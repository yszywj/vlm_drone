#!/usr/bin/env python3
"""Offline quality-head experiment on verified frozen TRAIN/validation features.

Never extract/upload data, change original supervision, or deploy a checkpoint.
Quality is conditional on a known valid measurement, NOT target identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import fit_target_state_validity_cached as cached

producer = cached.producer
PROTOCOL = "frozen_geometry_conditional_quality_v1"
ERROR_LIMIT_M = 1.0
THRESHOLD = 0.5
EXTRA_FEATURES = (
    "log1p_raw_depth", "log1p_corrected_depth", "signed_log1p_depth_residual",
    "delta_u_over_width", "delta_v_over_height", "anchor_u_over_width", "anchor_v_over_height",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "log_variance_x", "log_variance_y", "log_variance_z",
)
QUALITY_KEYS = {"epochs", "batch_size", "learning_rate", "weight_decay", "seed", "class_balance_power"}


def validate_config(config):
    if set(config) != {"experiment_name", "validity_fit", "quality_fit", "experiments"}:
        raise ValueError("invalid quality config keys")
    if not cached.safe_name(config["experiment_name"]):
        raise ValueError("unsafe experiment_name")
    cached.validate_fit(config["validity_fit"])
    if config["validity_fit"]["epochs"] != 1:
        raise ValueError("this protocol uses exactly one TRAIN-only validity warmup epoch")
    q = config["quality_fit"]
    if not isinstance(q, dict) or set(q) != QUALITY_KEYS:
        raise ValueError("invalid quality_fit keys")
    cached.validate_fit({**q, "hard_example_weight": 1.0, "anchor_logit_weight": 0.0})
    if not isinstance(config["experiments"], list) or not config["experiments"]:
        raise ValueError("at least one quality experiment is required")
    names = set()
    for exp in config["experiments"]:
        if (not isinstance(exp, dict) or set(exp) != {"name", "hidden_dim"}
                or not cached.safe_name(exp["name"]) or exp["name"] in names):
            raise ValueError("invalid/duplicate quality experiment")
        if isinstance(exp["hidden_dim"], bool) or not isinstance(exp["hidden_dim"], int) or not 0 <= exp["hidden_dim"] <= 128:
            raise ValueError("hidden_dim must be an integer in [0,128]")
        names.add(exp["name"])


def load_config(path):
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "base_config" not in raw:
        raise ValueError("quality config requires base_config")
    raw = dict(raw)
    base = Path(raw.pop("base_config"))
    if not base.is_absolute():
        base = path.resolve().parent / base
    validate_config(raw)
    return producer.load_settings(base.resolve()), raw


def quality_inputs(features, rows):
    """Explicit runtime-available allowlist; no IDs, world truth or labels.

    Invalid sensor/geometry rows use zeros and remain unconditionally gated out.
    Missing/non-finite inputs on an eligible row are an error, never imputed.
    """
    if (features.ndim != 2 or len(features) != len(rows) or features.device.type != "cpu"
            or features.requires_grad or not torch.isfinite(features).all()):
        raise ValueError("quality features must be finite detached CPU tensors")
    extra = []
    for row in rows:
        if not (row["reference_input_valid"] and row["model_geometry_valid"]):
            extra.append([0.0] * len(EXTRA_FEATURES))
            continue
        size = row["image_size_wh"]
        delta, anchor = row["delta_uv_px"], row["anchor_uv_px"]
        bbox, variance = row["bbox_xyxy_normalized"], row["position_variance_m2"]
        depths = [row["raw_depth_m"], row["corrected_depth_m"], row["depth_residual_m"]]
        if any(len(v) != n for v, n in ((size, 2), (delta, 2), (anchor, 2), (bbox, 4), (variance, 3))):
            raise ValueError("invalid runtime quality feature shape")
        values = [*size, *delta, *anchor, *bbox, *variance, *depths]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("non-finite runtime quality feature")
        if min(*size, *variance, *depths[:2]) <= 0:
            raise ValueError("quality sizes, variances and eligible depths must be positive")
        residual = depths[2]
        extra.append([math.log1p(depths[0]), math.log1p(depths[1]),
            math.copysign(math.log1p(abs(residual)), residual),
            delta[0]/size[0], delta[1]/size[1], anchor[0]/size[0], anchor[1]/size[1],
            *bbox, *(math.log(v) for v in variance)])
    result = torch.cat((features, torch.tensor(extra, dtype=features.dtype)), dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("non-finite transformed quality input")
    return result


def quality_supervision(rows):
    """TRAIN-only conditional accuracy labels, separate from validity labels."""
    if any(r["split"] != "train" for r in rows):
        raise ValueError("quality supervision requires TRAIN only")
    mask, labels = [], []
    for row in rows:
        selected = bool(row["probe_fit_mask"] and row["probe_label"])
        mask.append(selected)
        error = row["model_error_m"] if selected else None
        if selected and (isinstance(error, bool) or not isinstance(error, (int, float))
                         or not math.isfinite(error) or error < 0):
            raise ValueError("invalid TRAIN quality supervision error")
        labels.append(float(error <= ERROR_LIMIT_M) if selected else 0.0)
    return torch.tensor(mask, dtype=torch.bool), torch.tensor(labels, dtype=torch.float32)


def quality_weights(labels, power):
    good = int(labels.sum())
    bad = len(labels) - good
    if not good or not bad:
        raise ValueError("quality training requires both accurate and >1m TRAIN measurements")
    ratio = (good/bad)**power
    weights = torch.where(labels.bool(), 1.0, ratio)
    weights /= weights.mean()
    return weights, {"accurate_count": good, "over_1m_count": bad,
        "negative_to_positive_weight_ratio": ratio,
        "negative_loss_weight_fraction": float(weights[~labels.bool()].sum()/weights.sum())}


def prepare_data(prepared, original, config):
    if set(prepared) != set(producer.SPLITS):
        raise ValueError("only TRAIN and validation caches are permitted")
    episodes = [{r["episode_id"] for r in prepared[s][1]} for s in producer.SPLITS]
    if episodes[0] & episodes[1]:
        raise ValueError("TRAIN/validation episode overlap")
    inputs = {}
    for split, (features, rows) in prepared.items():
        cached.validate_cached_split(features, rows, split, original)
        inputs[split] = quality_inputs(features, rows)
    mask, all_labels = quality_supervision(prepared["train"][1])
    labels = all_labels[mask]
    weights, statistics = quality_weights(labels, config["quality_fit"]["class_balance_power"])
    train_inputs = inputs["train"][mask]
    # Normalization and class weights use qualified TRAIN only, never validation.
    mean = train_inputs.mean(0)
    scale = train_inputs.std(0, unbiased=False).clamp_min(1e-4)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise ValueError("non-finite TRAIN normalization")
    fit_rows = [r for r, selected in zip(prepared["train"][1], mask.tolist()) if selected]
    statistics.update(masked_count=len(mask)-int(mask.sum()),
        episode_count=len({r["episode_id"] for r in fit_rows}),
        risk_episode_counts=dict(Counter(r["episode_id"] for r in fit_rows if r["model_error_m"] > ERROR_LIMIT_M)),
        normalization_split="train", loss_split="train", error_limit_m=ERROR_LIMIT_M)
    return {"inputs": inputs, "mask": mask, "labels": labels, "weights": weights,
            "mean": mean, "scale": scale, "statistics": statistics}


class QualityHead(nn.Module):
    def __init__(self, mean, scale, hidden_dim):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        self.layers = (nn.Sequential(nn.Linear(mean.numel(), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
                       if hidden_dim else nn.Sequential(nn.Linear(mean.numel(), 1)))
        # Begin with a permissive quality proposal, not a silently active gate.
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.constant_(self.layers[-1].bias, 2.0)

    def forward(self, inputs):
        return self.layers(((inputs-self.mean)/self.scale).clamp(-10, 10)).flatten()


def clone_state(module):
    state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("cannot save non-finite head parameters or normalization")
    return state


def warmup_validity(train, original, fit):
    """One TRAIN epoch, then freeze; no validation data is accepted here."""
    x, rows = train
    cached.validate_cached_split(x, rows, "train", original)
    mask = torch.tensor([r["probe_fit_mask"] for r in rows], dtype=torch.bool)
    tx = x[mask]
    labels = torch.tensor([r["probe_label"] for r in rows], dtype=torch.float32)[mask]
    torch.manual_seed(fit["seed"])
    head = nn.Linear(x.shape[1], 1)
    head.load_state_dict(original, strict=True)
    with torch.no_grad():
        teacher = head(tx).flatten().detach().clone()
    weights, statistics = cached.sample_weights(labels, torch.sigmoid(teacher), fit)
    optimizer = torch.optim.AdamW(head.parameters(), lr=fit["learning_rate"], weight_decay=fit["weight_decay"])
    generator = torch.Generator().manual_seed(fit["seed"])
    total_loss = 0.0
    for ids in torch.randperm(len(tx), generator=generator).split(fit["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = cached.objective(head(tx[ids]).flatten(), labels[ids], weights[ids], teacher[ids], fit["anchor_logit_weight"])
        if not torch.isfinite(loss):
            raise ValueError("non-finite validity warmup loss")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(ids)
    return clone_state(head), {"train_loss": total_loss/len(tx), "weight_statistics": statistics,
        "epochs": 1, "original_validity_labels_unchanged": True, "frozen_during_quality_fit": True}


def validity_probabilities(features, state):
    with torch.no_grad():
        return torch.sigmoid(F.linear(features, state["weight"], state["bias"]).flatten())


def joint_rows(rows, validity, quality):
    if quality.shape != validity.shape or quality.ndim != 1 or not torch.isfinite(quality).all() or not ((quality >= 0) & (quality <= 1)).all():
        raise ValueError("invalid quality probabilities")
    result = producer.regate(rows, validity)
    for row, probability in zip(result, quality.tolist()):
        row["quality_probability"] = probability
        rejected = probability < THRESHOLD
        row["failure_flags"] = {**row["failure_flags"], "quality_head_rejected": rejected}
        if row["model_valid"] and rejected:
            row["model_valid"] = False
            row["model_failure_reason"] = "quality_head_rejected"
    return result


def validation_diagnostics(rows):
    score = producer.validation_score(rows)
    good = [r for r in rows if r["probe_eligible"] and r["probe_label"] and r["model_error_m"] <= ERROR_LIMIT_M]
    score["quality_diagnostics"] = {"eligible_accurate_positive_count": len(good),
        "accurate_positive_rejected_count": sum(not r["model_valid"] for r in good),
        "quality_rejected_after_validity_count": sum(r["model_failure_reason"] == "quality_head_rejected" for r in rows)}
    return score


def changed_rows(original, candidate):
    return [{"original_valid": a["model_valid"], "original_validity_probability": a["validity_probability"],
             "candidate": b} for a, b in zip(original, candidate) if a["model_valid"] != b["model_valid"]]


def fit_quality(prepared, original, proposal, data, fit, hidden_dim):
    x = data["inputs"]["train"][data["mask"]]
    vx, vrows = data["inputs"]["validation"], prepared["validation"][1]
    probabilities = validity_probabilities(prepared["validation"][0], proposal)
    baseline = validation_diagnostics(vrows)
    proposal_score = validation_diagnostics(producer.regate(vrows, probabilities))
    torch.manual_seed(fit["seed"])
    head = QualityHead(data["mean"], data["scale"], hidden_dim)
    optimizer = torch.optim.AdamW(head.parameters(), lr=fit["learning_rate"], weight_decay=fit["weight_decay"])
    generator = torch.Generator().manual_seed(fit["seed"])
    best_epoch, best_score, best_state, best_rows, history = 0, baseline, None, vrows, []
    last_rows = vrows
    for epoch in range(1, fit["epochs"]+1):
        loss_sum = 0.0
        for ids in torch.randperm(len(x), generator=generator).split(fit["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = (F.binary_cross_entropy_with_logits(head(x[ids]), data["labels"][ids], reduction="none") * data["weights"][ids]).mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite quality loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        with torch.no_grad():
            last_rows = joint_rows(vrows, probabilities, torch.sigmoid(head(vx)))
        score = validation_diagnostics(last_rows)
        admissible = producer.admissible(score, baseline)
        if admissible and score["positive_supervision_rejected_count"] < best_score["positive_supervision_rejected_count"]:
            best_epoch, best_score, best_state, best_rows = epoch, score, clone_state(head), last_rows
        history.append({"epoch": epoch, "train_loss": loss_sum/len(x), "admissible": admissible, "validation": score})
        print(f"[quality {epoch}/{fit['epochs']}] loss={loss_sum/len(x):.5f} "
              f"positive_rejected={score['positive_supervision_rejected_count']} "
              f"false_positive={score['no_target_false_positive_count']} "
              f"over1m={score['accepted_over_1m_count']} admissible={admissible}", flush=True)
    report = {"complete": True, "candidate_improved": best_epoch > 0, "best_epoch": best_epoch,
        "baseline_validation": baseline, "proposal_validity_only_validation": proposal_score,
        "candidate_validation": best_score, "history": history,
        "trainable_quality_parameter_count": sum(p.numel() for p in head.parameters()),
        "quality_weight_statistics": data["statistics"], "hidden_dim": hidden_dim,
        "quality_input_dim": x.shape[1], "quality_extra_features": list(EXTRA_FEATURES),
        "quality_good_error_max_m": ERROR_LIMIT_M, "validity_threshold": THRESHOLD, "quality_threshold": THRESHOLD,
        "original_validity_labels_unchanged": True, "geometry_parameters_changed": False,
        "geometry_bn_buffers_changed": False, "selection_split": "validation", "test_split_used": False,
        "promotion_passed": False, "pc_upload_required": False}
    # None means the actual ORIGINAL pipeline, not original validity AND a new quality gate.
    selected_validity = proposal if best_state is not None else original
    states = {"best": {"validity_state_dict": selected_validity, "quality_state_dict": best_state,
                       "quality_enabled": best_state is not None, "epoch": best_epoch},
              "last": {"validity_state_dict": proposal, "quality_state_dict": clone_state(head),
                       "quality_enabled": True, "epoch": fit["epochs"]}}
    changes = {"offline_only": True, "selection_split": "validation", "error_used_as_input": False,
        "best": changed_rows(vrows, best_rows), "last": changed_rows(vrows, last_rows)}
    return states, report, changes


def source_hashes():
    return {str(p.relative_to(ROOT)): producer.sha256_file(p) for p in (Path(__file__), Path(cached.__file__))}


def check_output(output, contract):
    if output.is_symlink():
        raise ValueError("quality output must not be a symlink")
    cp = output / "contract.json"
    if cp.is_symlink():
        raise ValueError("quality contract must not be a symlink")
    if cp.exists():
        if json.loads(cp.read_text()) != contract:
            raise ValueError("quality contract changed; choose a NEW experiment_name")
    elif output.exists():
        if not output.is_dir() or any(p.name != cached._RUN_LOCK_FILENAME or p.is_symlink()
                or not p.is_file() or p.stat().st_size != 0 for p in output.iterdir()):
            raise ValueError("output exists without ownership; choose a NEW experiment_name")


def load_completed(output, contract):
    path = output / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    expected_files = {"best_quality_bundle.pt", "last_quality_bundle.pt", "validation_changes.json"}
    if (path.is_symlink() or report.get("complete") is not True or report.get("contract") != contract
            or set(report.get("artifact_sha256", {})) != expected_files):
        raise ValueError("completed quality report mismatch")
    for name, digest in report["artifact_sha256"].items():
        artifact = output / name
        if artifact.is_symlink() or producer.sha256_file(artifact) != digest:
            raise ValueError("completed quality artifact SHA256 mismatch")
    return report


def save_experiment(output, contract, prepared, original, proposal, warmup, data, fit, hidden_dim):
    check_output(output, contract)
    with producer._exclusive_run_lock(output):
        check_output(output, contract)
        done = load_completed(output, contract)
        if done is not None:
            print(f"Already complete: {output}", flush=True)
            return done
        producer._atomic_write_json(output / "contract.json", contract)
        before = {k: v.clone() for k, v in original.items()}
        states, report, changes = fit_quality(prepared, original, proposal, data, fit, hidden_dim)
        if any(not torch.equal(v, before[k]) for k, v in original.items()):
            raise ValueError("original head was modified")
        hashes = {}
        for name, state in states.items():
            filename = f"{name}_quality_bundle.pt"
            artifact = {**state, "artifact_type": "offline_conditional_quality_heads_only", "protocol": PROTOCOL,
                "deployable": False, "selected_candidate": name == "best" and report["candidate_improved"],
                "base_checkpoint_sha256": contract["cache"]["base_checkpoint_sha256"], "contract": contract,
                "hidden_dim": hidden_dim, "input_dim": data["mean"].numel(), "extra_features": list(EXTRA_FEATURES),
                "quality_good_error_max_m": ERROR_LIMIT_M, "validity_threshold": THRESHOLD, "quality_threshold": THRESHOLD}
            producer.save_receipt(output / filename, artifact)
            hashes[filename] = producer.sha256_file(output / filename)
        producer._atomic_write_json(output / "validation_changes.json", changes)
        hashes["validation_changes.json"] = producer.sha256_file(output / "validation_changes.json")
        report.update(contract=contract, validity_warmup=warmup, artifact_sha256=hashes)
        producer._atomic_write_json(output / "report.json", report)
        # Re-read all committed evidence before declaring this experiment done.
        return load_completed(output, contract)


def run(settings, config, *, dry_run=False):
    validate_config(config)
    torch.set_num_threads(4)
    prepared, original, identity = cached.read_cache(settings)
    data = prepare_data(prepared, original, config)
    contract = {"protocol": PROTOCOL, "cache": identity, "config": config, "source_sha256": source_hashes(),
        "quality_error_limit_m": ERROR_LIMIT_M, "validity_threshold": THRESHOLD, "quality_threshold": THRESHOLD,
        "extra_features": list(EXTRA_FEATURES), "normalization": "qualified_TRAIN_mean_std_floor1e-4_clip10"}
    output = settings["output_dir"] / config["experiment_name"]
    check_output(output, contract)
    plan = {"cache_verified": True, "receipt_count": identity["receipt_count"],
        "sample_counts": {s: len(rows) for s, (_, rows) in prepared.items()},
        "quality_training": data["statistics"], "quality_input_dim": data["mean"].numel(),
        "experiments": config["experiments"], "epochs_per_quality_experiment": config["quality_fit"]["epochs"],
        "output_dir": str(output), "execution_device": "cpu", "pc_upload_required": False, "test_split_used": False}
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return plan
    with producer._exclusive_run_lock(output):
        check_output(output, contract)
        producer._atomic_write_json(output / "contract.json", contract)
        results, proposal, warmup = {}, None, None
        for experiment in config["experiments"]:
            destination = output / experiment["name"]
            exp_contract = {**contract, "experiment": experiment}
            check_output(destination, exp_contract)
            done = load_completed(destination, exp_contract)
            if done is not None:
                results[experiment["name"]] = done
                print(f"Already complete: {destination}", flush=True)
                continue
            if proposal is None:
                proposal, warmup = warmup_validity(prepared["train"], original, config["validity_fit"])
            print(f"Running quality experiment: {experiment['name']}", flush=True)
            results[experiment["name"]] = save_experiment(destination, exp_contract, prepared, original, proposal,
                warmup, data, config["quality_fit"], experiment["hidden_dim"])
        improved = [name for name, result in results.items() if result["candidate_improved"]]
        selected = min(improved, key=lambda n: results[n]["candidate_validation"]["positive_supervision_rejected_count"]) if improved else None
        summary = {"complete": True, "contract": contract, "candidate_improved": selected is not None,
            "selected_experiment": selected, "selection_split": "validation", "test_split_used": False,
            "promotion_passed": False, "pc_upload_required": False, "quality_training": data["statistics"],
            "baseline_validation": next(iter(results.values()))["baseline_validation"],
            "experiments": {name: {k: result[k] for k in ("best_epoch", "candidate_improved", "candidate_validation", "artifact_sha256")}
                            for name, result in results.items()},
            "notes": ["Frozen backbone/geometry/covariance; original measurement-valid labels are unchanged.",
                "Only known supervised TRAIN positives teach conditional quality; negatives/unknowns are not relabelled.",
                "Quality TRAIN errors are in-sample for the frozen backbone, not independent calibration.",
                "Repeated-development validation, not independent generalization or task identity certification.",
                "No admissible improvement means ORIGINAL validity with quality disabled.",
                "Last bundles are diagnostic only; even selected best bundles are not deployable."]}
        producer._atomic_write_json(output / "report.json", summary)
    print(f"Cached quality experiment complete: {output / 'report.json'}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="verify cache and supervision without training, uploads or writes")
    args = parser.parse_args(argv)
    settings, config = load_config(args.config)
    run(settings, config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
