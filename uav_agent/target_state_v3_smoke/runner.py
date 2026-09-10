"""Bounded positive-only overfit test, NOT formal Stage B or a deployable model.

Retain the verified V3 loader and temporal CNN/GRU. Train pixel/depth heads
against the SAME reference time's centre, never project a moving reference
target into old frames and pretend it was stationary. Oracle labels enter the
objective only. Validity/variance heads are frozen and explicitly uncalibrated.
"""
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from datasets.target_state.dataset import split_for_episode
from target_state_v3.data import TargetStateV3Dataset, sensor_observations
from target_state_v3.measurement import MeasurementPolicy, measure_window
from target_state_v3.verify_tar import file_sha
from target_state_v3_derived.ground_overlay import (
    CONFIG, ROOT, check_overlay, derived_view, load_source, verified_overlay_episode,
)
from training.target_state.config import load_training_config
from training.target_state.data import GEOMETRY_INPUT_FIELDS
from training.target_state.geometry import corrected_ray_to_world, project_world_to_pixel
from training.target_state.model import TemporalRayDepthNet, TemporalRayDepthOutput
from training.target_state.sharded_trainer import _atomic_write_json

PROTOCOL = "v3_geometry_overfit_diagnostic_v1"
NETWORK_INPUTS = ("roi_rgbd", "geometry", "missing_mask")
MAX_POSITIVE_WINDOWS_PER_SPLIT = 64


@dataclass(frozen=True)
class SmokeOptions:
    steps: int = 300
    batch_size: int = 6
    learning_rate: float = 1e-3
    device: str = "cpu"
    threads: int = 4
    seed: int = 42
    report_every: int = 25

    def __post_init__(self):
        for name, lo, hi in (("steps", 2, 1000), ("batch_size", 1, 18),
                             ("threads", 1, 8), ("report_every", 1, 1000), ("seed", 0, 2**31-1)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer within [{lo}, {hi}]")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .01:
            raise ValueError("learning rate must be within (0, 0.01]")
        device = torch.device(self.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("smoke device must be CPU or CUDA")


def code_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths.append(ROOT/"scripts/train_target_state_v3_smoke.py")
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def separate_output(output, *sources):
    output = output.resolve()
    if any(output == p.resolve() or output in p.resolve().parents or p.resolve() in output.parents
           for p in sources):
        raise ValueError("smoke output must be separate from original data and derived annotations")
    if output.exists():
        raise ValueError("output already exists; use a new version, no implicit resume or overwrite")
    return output


def collate(samples, device="cpu"):
    return {k: v.to(device) for k, v in default_collate(samples).items()}


def forward_sensor_only(model, batch):
    return model(*(batch[k] for k in NETWORK_INPUTS))


def ray_result(output, batch):
    return corrected_ray_to_world(anchor_uv_px=batch["anchor_uv_px"], raw_depth_m=batch["raw_depth_m"],
        delta_uv_px=output.delta_uv_px, depth_residual_m=output.depth_residual_m,
        intrinsics_fx_fy_cx_cy=batch["intrinsics_fx_fy_cx_cy"],
        camera_position_world_m=batch["camera_position_world_m"],
        camera_orientation_world_wxyz=batch["camera_orientation_world_wxyz"])


def zero_correction(batch):
    z = batch["raw_depth_m"]
    return TemporalRayDepthOutput(torch.zeros_like(batch["anchor_uv_px"]), torch.zeros_like(z),
                                  torch.zeros((*z.shape, 3), device=z.device), torch.full_like(z, -20.))


def reference_objective(output, batch):
    if any(not torch.isfinite(v).all() for v in output.as_dict().values()):
        raise ValueError("non-finite model output")
    if not all(batch[k].bool().all() for k in
               ("measurement_valid", "target_present_mask", "label_valid_mask", "reference_center_in_image")):
        raise ValueError("regression smoke only accepts eligible positive reference windows")
    uv, target_depth, valid = project_world_to_pixel(position_world_m=batch["target_position_world_m"],
        intrinsics_fx_fy_cx_cy=batch["intrinsics_fx_fy_cx_cy"],
        camera_position_world_m=batch["camera_position_world_m"],
        camera_orientation_world_wxyz=batch["camera_orientation_world_wxyz"])
    if (not valid.all() or not torch.allclose(uv, batch["history_center_uv_px"][:, -1], atol=1e-3, rtol=1e-5)
            or not torch.allclose(target_depth, batch["target_depth_m"], atol=1e-4, rtol=1e-5)):
        raise ValueError("reference centre/depth labels disagree with same-time geometry")
    world, depth, ray_valid = ray_result(output, batch)
    if (not ray_valid.all() or not torch.isfinite(world).all() or
            torch.any(depth < batch["depth_range_m"][:, 0]) or torch.any(depth > batch["depth_range_m"][:, 1])):
        raise ValueError("invalid corrected ray; never hide invalid predictions by masking the loss")
    pixel = F.smooth_l1_loss(batch["anchor_uv_px"]+output.delta_uv_px, uv)
    depth_loss = F.smooth_l1_loss(depth, target_depth)
    position = F.smooth_l1_loss(world, batch["target_position_world_m"])
    total = .1*pixel + depth_loss + 2.*position
    if not torch.isfinite(total):
        raise ValueError("non-finite regression objective")
    return total, {"pixel_huber": pixel, "depth_huber": depth_loss, "position_huber": position}, world, depth, uv


def make_model(kwargs):
    model = TemporalRayDepthNet(**kwargs)
    # Start exactly at the geometric surface baseline; do not load legacy weights.
    for head in (model.delta_uv_head, model.depth_head, model.log_variance_head, model.validity_head):
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    model.validity_head.bias.data.fill_(-20.)  # reject-by-default, not a learned gate
    for head in (model.log_variance_head, model.validity_head):
        head.requires_grad_(False)
    return model


def prepare_data(derived):
    """Independently validate all evidence; only cache eligible train/val positives."""
    derived = derived.resolve()
    acceptance_path = derived/"acceptance_report.json"
    saved_sha = file_sha(acceptance_path)
    saved = json.loads(acceptance_path.read_text())
    fresh = check_overlay(derived)  # read-only: does not rewrite the acceptance file
    if fresh != saved or file_sha(acceptance_path) != saved_sha:
        raise ValueError("saved acceptance report differs from fresh replay")
    manifest = json.loads((derived/"manifest.json").read_text())
    source, state, source_sha = load_source(Path(manifest["source_session"]))
    cfg = replace(load_training_config(CONFIG), device="cpu", num_workers=0, initial_checkpoint_path=None,
        expected_yolo_model_sha256=state["contract"]["detector_deployment"]["model_sha256"])
    samples = {"train": [], "validation": []}
    inventory, counts, seen = [], {}, set()
    expected = {(w["episode_id"], w["reference_frame_id"]): w for w in fresh["windows"]}
    for source_entry, overlay_entry in zip(state["episodes"], manifest["episodes"]):
        folder, records, _ = verified_overlay_episode(source, state, source_entry, derived, overlay_entry)
        split = split_for_episode(source_entry["episode_id"], seed=cfg.seed)
        original = TargetStateV3Dataset(cfg, episode_root=folder, split=split)
        view = derived_view(original, records)
        split_counts = counts.setdefault(split, Counter(windows=0, positive_measurement_windows=0))
        for i, seq in enumerate(view.sequences):
            key = (seq.reference.episode_id, seq.reference.frame_id)
            w = expected[key]
            seen.add(key)
            eligible = w["positive_measurement"]
            split_counts.update(windows=1, positive_measurement_windows=int(eligible))
            inventory.append({"sequence_id": seq.sequence_id, "reference_frame_id": seq.reference.frame_id,
                "episode_id": seq.reference.episode_id, "split": split, "positive_measurement": eligible,
                "used_for_optimization": eligible and split == "train"})
            if not eligible or split == "test":
                continue
            if len(samples[split]) >= MAX_POSITIVE_WINDOWS_PER_SPLIT:
                raise ValueError("too many positives for bounded smoke test; do not truncate or expand silently")
            sample = view[i]
            if not bool(sample["measurement_valid"]) or any(not torch.isfinite(v).all() for v in sample.values()):
                raise ValueError("materialized sample differs from acceptance")
            decision = measure_window(sensor_observations((*seq.history, seq.reference), folder))[-1]
            batch = collate([sample])
            _, _, world, _, _ = reference_objective(zero_correction(batch), batch)
            if (not decision.accepted or
                    not torch.allclose(sample["anchor_uv_px"], torch.tensor(decision.surface.uv_px)) or
                    not torch.allclose(sample["raw_depth_m"], torch.tensor(decision.surface.depth_m)) or
                    not np.allclose(world[0].numpy(), decision.surface.world_m, atol=1e-4, rtol=1e-5)):
                raise ValueError("zero-residual ray differs from shared sensor-only measurement")
            samples[split].append(sample)
        print(f"Cached {source_entry['episode_id']}: train={len(samples['train'])}, validation={len(samples['validation'])}", flush=True)
    if seen != set(expected) or not samples["train"]:
        raise ValueError("window ledger mismatch or no positive training windows")
    ids = {split: {r["episode_id"] for r in inventory if r["split"] == split} for split in ("train", "validation", "test")}
    if ids["train"] & (ids["validation"] | ids["test"]) or ids["validation"] & ids["test"]:
        raise ValueError("episode split leakage")
    kwargs = {name: getattr(cfg, name) for name in
              ("geometry_input_dim", "roi_feature_dim", "geometry_feature_dim", "hidden_dim", "gru_layers")}
    provenance = {"source_session": str(source), "source_session_sha256": source_sha,
        "overlay_root": str(derived), "overlay_manifest_sha256": file_sha(derived/"manifest.json"),
        "acceptance_report_sha256": saved_sha, "source_producer_sha256": state["contract"]["source_sha256"],
        "derivation_code_sha256": manifest["derivation_code_sha256"], "config_sha256": file_sha(CONFIG),
        "physical_captures": fresh["physical_captures"], "split_seed": cfg.seed,
        "counts": {k: dict(v) for k, v in counts.items()}, "windows": inventory,
        "model_kwargs": kwargs, "network_inputs": NETWORK_INPUTS, "geometry_fields": GEOMETRY_INPUT_FIELDS,
        "roi_size_px": cfg.roi_size_px, "history_size": cfg.history_size, "max_history_age_s": cfg.max_history_age_s,
        "measurement_preprocessing": MeasurementPolicy().contract(), "zero_residual_sensor_geometry_parity": True}
    return samples, provenance


@torch.no_grad()
def evaluate(model, samples, *, device, batch_size):
    if not samples:
        return {"count": 0, "status": "not_run_no_positive_windows"}
    if model is not None:
        model.eval()
    errors, pixel_errors, depth_errors, losses = [], [], [], []
    for start in range(0, len(samples), batch_size):
        batch = collate(samples[start:start+batch_size], device)
        output = zero_correction(batch) if model is None else forward_sensor_only(model, batch)
        loss, _, world, depth, uv = reference_objective(output, batch)
        n = len(world)
        losses.extend([float(loss)]*n)
        errors.extend(torch.linalg.vector_norm(world-batch["target_position_world_m"], dim=-1).cpu().tolist())
        pixel_errors.extend(torch.linalg.vector_norm(batch["anchor_uv_px"]+output.delta_uv_px-uv, dim=-1).cpu().tolist())
        depth_errors.extend(abs(depth-batch["target_depth_m"]).cpu().tolist())
    return {"count": len(samples), "loss": float(np.mean(losses)),
        "position_mean_m": float(np.mean(errors)), "position_median_m": float(np.median(errors)),
        "position_p95_m": float(np.quantile(errors, .95)), "position_max_m": float(np.max(errors)),
        "pixel_error_mean_px": float(np.mean(pixel_errors)), "depth_mae_m": float(np.mean(depth_errors)),
        "invalid_predictions": 0, "scope": "all_preselected_positive_windows_no_prediction_filtering"}


def train_fixed_steps(model, samples, options, progress=None):
    if not samples:
        raise ValueError("no training samples")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=options.learning_rate, weight_decay=0.)
    generator = torch.Generator().manual_seed(options.seed)
    pending, groups_seen = [], set()
    required = {"roi_encoder", "geometry_encoder", "temporal", "delta_uv_head", "depth_head"}
    for step in range(1, options.steps+1):
        if not pending:
            pending = torch.randperm(len(samples), generator=generator).tolist()
        indices, pending = pending[:options.batch_size], pending[options.batch_size:]
        batch = collate([samples[i] for i in indices], options.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = forward_sensor_only(model, batch)
        loss, _, _, _, _ = reference_objective(output, batch)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not parameter.requires_grad or not torch.isfinite(parameter.grad).all():
                    raise ValueError("frozen/non-finite gradient")
                if torch.count_nonzero(parameter.grad):
                    groups_seen.add(name.split(".", 1)[0])
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % options.report_every == 0 or step == options.steps:
            row = {"step": step, "optimization_loss": float(loss.detach()),
                "train": evaluate(model, samples, device=options.device, batch_size=options.batch_size)}
            print(json.dumps(row), flush=True)
            if progress is not None:
                progress(row)
    return {"required_groups": sorted(required), "nonzero_gradient_groups": sorted(groups_seen),
            "passed": required <= groups_seen}


def run_smoke(derived, output, options, *, acknowledge_diagnostic_only=False):
    if not acknowledge_diagnostic_only:
        raise ValueError("explicit --acknowledge-diagnostic-only is required; this pilot is not training-approved")
    started = time.monotonic()
    derived = derived.resolve()
    source = Path(json.loads((derived/"manifest.json").read_text())["source_session"])
    output = separate_output(output, derived, source)
    torch.set_num_threads(options.threads)
    device = torch.device(options.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; explicitly use --device cpu")
    samples, provenance = prepare_data(derived)
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    model = make_model(provenance["model_kwargs"]).to(device)
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
              if k.startswith(("log_variance_head.", "validity_head."))}
    manifest = {"protocol": PROTOCOL, "diagnostic_only": True, "production_approved": False,
        "formal_training": False, "initialization": "random_backbone_zero_residual_heads_no_legacy_checkpoint",
        "frozen_heads": ["log_variance_head", "validity_head"], "options": asdict(options),
        "objective": {"reference_pixel_huber": .1, "reference_depth_huber": 1., "reference_position_huber": 2.,
                      "history_reprojection": 0., "gaussian_nll": 0., "validity_bce": 0.},
        "fixed_steps_no_validation_selection": True, "test_model_evaluation": False,
        "pass_criteria": {"train_position_mean_max_baseline_ratio": .5, "train_loss_max_baseline_ratio": .5,
                          "all_trainable_groups_have_nonzero_gradient": True, "checkpoint_reload_verified": True},
        "smoke_code_sha256": code_hashes(), "data": provenance}
    output.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(output/"diagnostic_manifest.json", manifest)
    try:
        baseline = {s: evaluate(None, values, device=options.device, batch_size=options.batch_size)
                    for s, values in samples.items()}
        initial = evaluate(model, samples["train"], device=options.device, batch_size=options.batch_size)
        if not math.isclose(initial["position_mean_m"], baseline["train"]["position_mean_m"], abs_tol=1e-6):
            raise ValueError("model initialization differs from zero-residual baseline")
        history = []
        def progress(row):
            history.append(row)
            _atomic_write_json(output/"progress.json", {"complete": False, "history": history})
        flow = train_fixed_steps(model, samples["train"], options, progress)
        final = {s: evaluate(model, values, device=options.device, batch_size=options.batch_size)
                 for s, values in samples.items()}
        if any(not torch.equal(model.state_dict()[k].cpu(), v) for k, v in frozen.items()):
            raise ValueError("untrained validity/variance head was changed")
        checkpoint = {"model_type": PROTOCOL, "diagnostic_only": True, "production_approved": False,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_kwargs": provenance["model_kwargs"], "steps": options.steps,
            "measurement_preprocessing": provenance["measurement_preprocessing"],
            "diagnostic_manifest_sha256": file_sha(output/"diagnostic_manifest.json")}
        path = output/"diagnostic_checkpoint.pt"
        temporary = output/"diagnostic_checkpoint.pt.partial"
        with temporary.open("xb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        restored = torch.load(path, weights_only=True, map_location="cpu")
        replay = make_model(restored["model_kwargs"]).to(device)
        replay.load_state_dict(restored["state_dict"], strict=True)
        reloaded = evaluate(replay, samples["train"], device=options.device, batch_size=options.batch_size)
        if any(not math.isclose(reloaded[k], final["train"][k], abs_tol=1e-6, rel_tol=1e-5)
               for k in ("loss", "position_mean_m", "position_max_m")):
            raise ValueError("saved checkpoint replay differs")
        if (file_sha(derived/"manifest.json") != provenance["overlay_manifest_sha256"] or
                file_sha(source/"session.json") != provenance["source_session_sha256"] or
                file_sha(derived/"acceptance_report.json") != provenance["acceptance_report_sha256"] or
                code_hashes() != manifest["smoke_code_sha256"]):
            raise ValueError("source/overlay/runner changed during smoke test")
        base, end = baseline["train"], final["train"]
        fit = (base["position_mean_m"] > 1e-6 and end["position_mean_m"] <= .5*base["position_mean_m"]
               and end["loss"] <= .5*base["loss"])
        report = {"complete": True, "protocol": PROTOCOL, "smoke_passed": bool(fit and flow["passed"]),
            "regression_fit_passed": bool(fit), "gradient_flow": flow, "checkpoint_reload_verified": True,
            "diagnostic_manifest_sha256": file_sha(output/"diagnostic_manifest.json"),
            "checkpoint_sha256": file_sha(path), "steps": options.steps, "elapsed_s": time.monotonic()-started,
            "counts": provenance["counts"], "baseline": baseline, "initial_train": initial, "final": final,
            "history": history, "source_modified": False, "production_approved": False,
            "ready_for_bulk_collection": False, "validity_trained": False, "covariance_calibrated": False,
            "validation_role": "held_out_episodes_descriptive_only_not_used_for_selection",
            "test_model_evaluation": "not_run", "notes": [
                "This is a positive-only plumbing/overfit test, NOT a formal Stage B continuation.",
                "Overlapping windows are not independent samples; fitting them proves no generalization.",
                "Frozen validity and variance heads must not be deployed.",
                "Keep source RGB-D, oracle evidence and sidecars; no data was transferred or deleted."]}
        _atomic_write_json(output/"smoke_report.json", report)
        _atomic_write_json(output/"progress.json", {"complete": True, "history": history})
        return report
    except BaseException as error:
        _atomic_write_json(output/"failure.json", {"complete": False, "error": f"{type(error).__name__}: {error}"})
        raise
