#!/usr/bin/env python3
"""Offline, resumable measurement audit; never train or change promotion gates.

Requests only validation/test shards under dedicated audit_* run IDs. Durable
per-shard evidence is committed before deleting that audit's server cache.
PC source archives and the model's training directory are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.target_state.config import LossWeights, TargetStateTrainingConfig
from training.target_state.geometry import corrected_ray_to_world, project_world_to_pixel
from training.target_state.measurement_gate import gate_batch
from perception.ray_measurement_gate import MEASUREMENT_PROTOCOL
from training.target_state.shard_runtime import materialize_shard, cleanup_materialized_shard
from training.target_state.sharded_trainer import (
    PCTransCLI, ShardedTrainingOptions, _atomic_write_json, _fsync_directory,
    _dataset_loader, _exclusive_run_lock, _new_model, _prepare_active_shard,
    _release_loader, _state_name, validate_shard_index_for_training,
)
from training.target_state.shards import load_shard_index
from training.target_state.trainer import (
    TargetStateEvaluationAccumulator, _device, _to_device, sha256_file,
)
from scripts.target_state_audit_evidence import (
    CaseExportOptions, compare_metrics, export_cases, verify_case_assets,
)


@torch.no_grad()
def diagnostic_rows(batch, output, *, maximum_depth_m):
    """Mirror official acceptance masks, but preserve per-sample diagnostics."""
    common = dict(
        anchor_uv_px=batch["anchor_uv_px"], raw_depth_m=batch["raw_depth_m"],
        intrinsics_fx_fy_cx_cy=batch["intrinsics_fx_fy_cx_cy"],
        camera_position_world_m=batch["camera_position_world_m"],
        camera_orientation_world_wxyz=batch["camera_orientation_world_wxyz"],
    )
    world, depth, ray_valid = corrected_ray_to_world(
        **common, delta_uv_px=output.delta_uv_px,
        depth_residual_m=output.depth_residual_m,
    )
    base_world, base_depth, base_ray = corrected_ray_to_world(
        **common, delta_uv_px=torch.zeros_like(output.delta_uv_px),
        depth_residual_m=torch.zeros_like(output.depth_residual_m),
    )
    probability = torch.sigmoid(output.measurement_valid_logit)
    model_gate = gate_batch(batch, corrected_depth_m=depth, delta_uv_px=output.delta_uv_px,
                           validity_probability=probability, maximum_depth_m=maximum_depth_m)
    baseline_gate = gate_batch(batch, corrected_depth_m=base_depth,
                              delta_uv_px=torch.zeros_like(output.delta_uv_px),
                              validity_probability=torch.ones_like(probability),
                              maximum_depth_m=maximum_depth_m)
    geometry_valid = (ray_valid & torch.isfinite(world).all(-1)
                      & torch.isfinite(depth) & model_gate.geometry_valid)
    base_valid = (base_ray & torch.isfinite(base_world).all(-1)
                  & torch.isfinite(base_depth) & baseline_gate.accepted)
    target = batch["target_position_world_m"]
    minimum = batch["depth_range_m"][:, 0]
    maximum = batch["depth_range_m"][:, 1].clamp_max(maximum_depth_m)
    size = batch["image_size_wh"]
    image_ok = torch.isfinite(size).all(-1) & (size > 0).all(-1)
    def uv_ok(uv):
        return (image_ok & torch.isfinite(uv).all(-1)
                & (uv >= 0).all(-1) & (uv < size).all(-1))
    def depth_ok(z):
        return (torch.isfinite(z) & torch.isfinite(minimum) & torch.isfinite(maximum)
                & (minimum > 0) & (maximum > minimum) & (z >= minimum) & (z <= maximum))
    detected = ~batch["missing_mask"][:, -1].bool()
    consistent = batch.get("reference_sensor_consistent", torch.ones_like(detected))
    # These independent flags describe the SAME gate; they do not replace it.
    flags = {
        "detector_miss": ~detected,
        "raw_depth_invalid_or_out_of_range": ~depth_ok(batch["raw_depth_m"]),
        "rgbd_consistency_rejected": ~consistent,
        "reference_anchor_or_image_invalid": ~uv_ok(batch["anchor_uv_px"]),
        "corrected_pixel_invalid_or_out_of_image": ~uv_ok(batch["anchor_uv_px"] + output.delta_uv_px),
        "corrected_depth_invalid_or_out_of_range": ~depth_ok(depth),
        "ray_or_world_invalid": ~ray_valid | ~torch.isfinite(world).all(-1),
        "validity_head_rejected": ~torch.isfinite(probability) | (probability < .5) | (probability > 1),
    }
    # Privileged evidence is computed AFTER sensor-only acceptance. It must
    # never turn an observation on/off or remove it from evaluation denominators.
    target_uv, target_depth, target_ray = project_world_to_pixel(
        position_world_m=target, intrinsics_fx_fy_cx_cy=batch["intrinsics_fx_fy_cx_cy"],
        camera_position_world_m=batch["camera_position_world_m"],
        camera_orientation_world_wxyz=batch["camera_orientation_world_wxyz"],
    )
    fields = {
        "evaluated_visible_target": (batch["label_valid_mask"].bool()
            & batch["target_present_mask"].bool()
            & batch["history_visible_mask"][:, -1].bool()
            & torch.isfinite(target).all(-1)),
        "no_target": ~batch["target_present_mask"].bool(),
        "model_valid": geometry_valid & model_gate.accepted,
        "baseline_valid": base_valid,
        "model_geometry_valid": geometry_valid,
        "validity_probability": probability,
        "raw_depth_m": batch["raw_depth_m"],
        "reference_input_valid": model_gate.input_valid,
        "reference_detected": detected,
        "reference_sensor_consistent": consistent,
        "reference_anchor_valid": uv_ok(batch["anchor_uv_px"]),
        "raw_depth_valid_in_range": depth_ok(batch["raw_depth_m"]),
        "image_size_wh": batch["image_size_wh"],
        "depth_range_m": batch["depth_range_m"],
        "corrected_depth_m": depth,
        "delta_uv_px": output.delta_uv_px,
        "depth_residual_m": output.depth_residual_m,
        "position_variance_m2": torch.exp(output.position_log_variance),
        "model_error_m": torch.linalg.vector_norm(world - target, dim=-1),
        "baseline_error_m": torch.linalg.vector_norm(base_world - target, dim=-1),
        "model_position_world_m": world,
        "baseline_position_world_m": base_world,
        "target_position_world_m": target,
        "anchor_uv_px": batch["anchor_uv_px"],
        "occlusion_ratio": batch["occlusion_ratio"],
        "bbox_jitter_score": batch["bbox_jitter_score"],
        "offline_target_center_uv_px": target_uv,
        "offline_target_depth_m": target_depth,
        "offline_target_in_output_domain": (batch["target_present_mask"].bool()
            & target_ray & uv_ok(target_uv) & depth_ok(target_depth)),
        "offline_measurement_supervision_positive": batch["measurement_valid"],
        "offline_validity_supervised": batch.get("validity_supervision_mask", torch.ones_like(detected)),
    }
    values = {key: value.detach().cpu().tolist() for key, value in fields.items()}
    flag_values = {key: value.detach().cpu().tolist() for key, value in flags.items()}
    rows = []
    for i in range(len(values["raw_depth_m"])):
        row = {key: value[i] for key, value in values.items()}
        row["failure_flags"] = {key: value[i] for key, value in flag_values.items()}
        row["model_failure_reason"] = None if row["model_valid"] else next(
            (key for key, failed in row["failure_flags"].items() if failed), "unclassified_gate_rejection")
        rows.append(row)
    return rows


def error_summary(rows, field):
    values = [row[field] for row in rows if math.isfinite(row[field])]
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(rows), "finite_count": len(values),
        "median_m": float(tensor.quantile(0.5)) if values else None,
        "p95_m": float(tensor.quantile(0.95)) if values else None,
        "over_1m_count": sum(value > 1 for value in values),
        "over_5m_count": sum(value > 5 for value in values),
        "over_10m_count": sum(value > 10 for value in values),
    }


def summarize_rows(rows):
    visible = [row for row in rows if row["evaluated_visible_target"]]
    both = [row for row in visible if row["model_valid"] and row["baseline_valid"]]
    failed = [row for row in visible if not row["model_valid"]]
    reasons = Counter()
    overlapping = Counter()
    regressions = Counter()
    for row in failed:
        # Ordered attribution is descriptive, not a causal proof. Independent
        # flags and raw heads remain in samples.json for overlapping failures.
        if row.get("model_failure_reason"):
            reason = row["model_failure_reason"]
        elif not row["detected"]:
            reason = "detector_miss"
        elif not math.isfinite(row["raw_depth_m"]) or row["raw_depth_m"] <= 0:
            reason = "no_valid_raw_depth"
        elif row.get("reference_input_valid") is False:
            reason = "invalid_reference_input"
        elif not row["model_geometry_valid"]:
            reason = "corrected_geometry_invalid"
        else:
            reason = "validity_head_rejected"
        reasons[reason] += 1
        overlapping.update(key for key, value in row.get("failure_flags", {}).items() if value)
        if row["baseline_valid"]:
            regressions[reason] += 1
    domains = {}
    if visible and "offline_target_in_output_domain" in visible[0]:
        for label, selected in (
            ("target_in_output_domain", [r for r in visible if r["offline_target_in_output_domain"]]),
            ("target_outside_output_domain", [r for r in visible if not r["offline_target_in_output_domain"]]),
            ("unknown_association_supervision", [r for r in visible if not r["offline_validity_supervised"]]),
            ("positive_measurement_supervision", [r for r in visible if r["offline_measurement_supervision_positive"]]),
        ):
            domains[label] = {"count": len(selected),
                             "model_failed_count": sum(not r["model_valid"] for r in selected),
                             "baseline_failed_count": sum(not r["baseline_valid"] for r in selected),
                             "both_accepted_count": sum(r["model_valid"] and r["baseline_valid"] for r in selected)}
    return {
        "sample_count": len(rows), "visible_target_count": len(visible),
        "no_target_count": sum(row["no_target"] for row in rows),
        "model_failed_count": len(failed),
        "baseline_failed_count": sum(not row["baseline_valid"] for row in visible),
        "model_only_failed_count": sum(row["baseline_valid"] and not row["model_valid"] for row in visible),
        "baseline_only_failed_count": sum(row["model_valid"] and not row["baseline_valid"] for row in visible),
        "both_failed_count": sum(not row["model_valid"] and not row["baseline_valid"] for row in visible),
        "model_false_positive_count": sum(row["no_target"] and row["model_valid"] for row in rows),
        "baseline_false_positive_count": sum(row["no_target"] and row["baseline_valid"] for row in rows),
        "model_failure_reasons_ordered": dict(reasons),
        "model_failure_flags_overlapping": dict(overlapping),
        "model_only_failure_reasons_ordered": dict(regressions),
        "offline_label_subgroups_overlapping": domains,
        "model_accepted_only": error_summary([r for r in visible if r["model_valid"]], "model_error_m"),
        "baseline_accepted_only": error_summary([r for r in visible if r["baseline_valid"]], "baseline_error_m"),
        "both_accepted": {"model": error_summary(both, "model_error_m"),
                          "baseline": error_summary(both, "baseline_error_m"),
                          "model_better_count": sum(r["model_error_m"] < r["baseline_error_m"] for r in both),
                          "model_worse_count": sum(r["model_error_m"] > r["baseline_error_m"] for r in both)},
    }


def evaluate_archive(archive, *, index, entry, config, model, device,
                     case_options=None, output_dir=None):
    materialized = materialize_shard(archive, index=index)
    loader = None
    try:
        dataset, loader = _dataset_loader(
            config=config, dataset_root=materialized.dataset_root, split=entry.split,
            device=device, shuffle=False, generator_seed=None, split_seed=index.split_seed,
        )
        accumulator = TargetStateEvaluationAccumulator(config.maximum_depth_m)
        rows = []
        with torch.inference_mode():
            for batch in loader:
                batch = _to_device(batch, device)
                output = model(batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
                accumulator.add_batch(batch=batch, output=output)
                batch_rows = diagnostic_rows(batch, output, maximum_depth_m=config.maximum_depth_m)
                for sequence, row in zip(dataset.sequences[len(rows):], batch_rows):
                    frame = sequence.reference
                    detection = frame.detector_prediction
                    row.update(
                        shard=entry.filename, sequence_id=sequence.sequence_id,
                        episode_id=frame.episode_id, frame_id=frame.frame_id,
                        timestamp_s=frame.timestamp_s, assignment_id=frame.assignment_id,
                        candidate_id=detection.candidate_id, tracker_id=detection.tracker_id,
                        detected=detection.detected,
                        bbox_xyxy_normalized=detection.bbox_xyxy_normalized,
                        rgb_path=frame.sensor_input.rgb_path, depth_path=frame.sensor_input.depth_path,
                    )
                rows.extend(batch_rows)
        if len(rows) != len(dataset) or len(rows) != entry.sequence_count:
            raise ValueError(f"sequence count mismatch: {entry.filename}")
        evidence = {"cases": [], "assets": {}}
        if case_options is not None:
            evidence = export_cases(rows, dataset.sequences,
                dataset_root=materialized.dataset_root, output_dir=output_dir,
                entry=entry, options=case_options)
        return materialized, {"rows": rows, "accumulator": accumulator.state_dict(),
                              "case_exports": evidence}
    finally:
        _release_loader(loader)


def load_receipt(path, *, contract_sha, entry, maximum_depth_m):
    receipt = torch.load(path, map_location="cpu", weights_only=True)
    if receipt.get("contract_sha256") != contract_sha or receipt.get("entry") != entry.to_dict():
        raise ValueError(f"audit receipt identity mismatch: {path}")
    if len(receipt["rows"]) != entry.sequence_count:
        raise ValueError(f"audit receipt sequence count mismatch: {path}")
    accumulator = TargetStateEvaluationAccumulator.from_state_dict(
        receipt["accumulator"], maximum_depth_m=maximum_depth_m,
    )
    return receipt, accumulator


def save_receipt(path, receipt):
    """Atomic durable evidence, independent of training-checkpoint schemas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(receipt, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def audit_split(*, split, config, options, index, lifecycle, model, device,
                output_dir, contract_sha, case_options=None):
    run_id = f"{options.run_id_prefix}.{split}"
    entries = index.shards_for_split(split)
    lifecycle.request(run_id, [entry.filename for entry in entries])
    total = TargetStateEvaluationAccumulator(config.maximum_depth_m)
    all_rows = []
    all_cases, all_assets = [], {}
    for ordinal, entry in enumerate(entries, 1):
        receipt_path = output_dir / "receipts" / (entry.filename + ".pt")
        materialized = None
        state = lifecycle.shard_state(run_id, entry.filename)
        name = _state_name(state)
        if receipt_path.exists():
            receipt, partial = load_receipt(receipt_path, contract_sha=contract_sha,
                                           entry=entry, maximum_depth_m=config.maximum_depth_m)
            if name != "consumed" or state["deleted"] is not True:
                archive = _prepare_active_shard(lifecycle=lifecycle, options=options,
                                                run_id=run_id, entry=entry)
                materialized = materialize_shard(archive, index=index)
        else:
            if name == "consumed":
                raise ValueError(f"consumed audit shard has no durable receipt: {entry.filename}; use a new audit prefix and output directory")
            print(f"[{split} {ordinal}/{len(entries)}] waiting/evaluating {entry.filename}", flush=True)
            archive = _prepare_active_shard(lifecycle=lifecycle, options=options,
                                            run_id=run_id, entry=entry)
            materialized, receipt = evaluate_archive(
                archive, index=index, entry=entry, config=config, model=model, device=device,
                case_options=case_options, output_dir=output_dir,
            )
            receipt.update(contract_sha256=contract_sha, entry=entry.to_dict())
            save_receipt(receipt_path, receipt)
            # Re-read and validate committed evidence BEFORE either deletion.
            receipt, partial = load_receipt(receipt_path, contract_sha=contract_sha,
                                           entry=entry, maximum_depth_m=config.maximum_depth_m)
        evidence = receipt.get("case_exports", {"cases": [], "assets": {}})
        if case_options is not None and "case_exports" not in receipt:
            raise ValueError("receipt lacks required case evidence; use a new audit run")
        verify_case_assets(evidence, output_dir)
        if materialized is not None:
            cleanup_materialized_shard(materialized)
            lifecycle.consume(run_id, entry.filename, delete=True)
        total.merge(partial)
        all_rows.extend(receipt["rows"])
        all_cases.extend(evidence["cases"])
        all_assets.update(evidence["assets"])
        print(f"[{split} {ordinal}/{len(entries)}] saved; audit server cache consumed", flush=True)
    report = {"measurement_protocol": MEASUREMENT_PROTOCOL,
              "evaluation_metrics": total.finalize(), "diagnostics": summarize_rows(all_rows),
              "case_exports": {"selected_count": len(all_cases),
                 "rgbd_complete_count": sum(c["rgbd_export_complete"] for c in all_cases),
                 "rgbd_omitted_budget_count": sum(not c["rgbd_export_complete"] for c in all_cases),
                 "unique_asset_bytes": sum(a["size_bytes"] for a in all_assets.values()),
                 "cases_file": f"{split}_cases.json"}}
    _atomic_write_json(output_dir / f"{split}_samples.json", {"samples": all_rows})
    _atomic_write_json(output_dir / f"{split}_cases.json", {
        "offline_only": True, "cases": all_cases, "assets": all_assets,
        "notes": ["RGB-D files are unchanged source bytes; labels/poses are OFFLINE evidence only.",
                  "Budget-omitted cases retain metadata but are NOT fully replayable without PC data."]})
    _atomic_write_json(output_dir / f"{split}_report.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("model-manifest", "shard-index", "pc-trans-root", "pc-trans-config", "bridge-root", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--run-id-prefix", required=True, help="must start with audit_")
    result.add_argument("--pc-trans-python", type=Path, default=Path(sys.executable))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--wait-timeout", type=float, default=86400)
    result.add_argument("--case-assets-max-gib", type=float, default=2.0)
    result.add_argument("--case-error-m", type=float, default=1.0)
    result.add_argument("--case-episode", action="append", default=[])
    result.add_argument("--dry-run", action="store_true", help="validate without requesting data or writing files")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not math.isfinite(args.case_assets_max_gib) or args.case_assets_max_gib < 0:
        raise ValueError("case asset budget must be finite and non-negative")
    case_options = CaseExportOptions(int(args.case_assets_max_gib * 1024**3),
                                     args.case_error_m, tuple(sorted(set(args.case_episode))))
    manifest_path = args.model_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    raw_config = dict(manifest["config"])
    raw_config["loss_weights"] = LossWeights(**raw_config["loss_weights"])
    config = replace(TargetStateTrainingConfig(**raw_config), device=args.device, num_workers=args.num_workers)
    options = ShardedTrainingOptions(
        shard_index_path=args.shard_index, pc_trans_root=args.pc_trans_root,
        pc_trans_config=args.pc_trans_config, bridge_root=args.bridge_root,
        run_id_prefix=args.run_id_prefix, wait_timeout_s=args.wait_timeout,
        pc_trans_python=args.pc_trans_python,
    )
    if not options.run_id_prefix.startswith("audit_"):
        raise ValueError("use a dedicated audit_* prefix, not an old training run ID")
    output_dir = args.output_dir.resolve()
    training_dir = config.output_dir / config.run_name
    if output_dir == training_dir or training_dir in output_dir.parents or output_dir in training_dir.parents:
        raise ValueError("audit output must be separate from the training directory")
    bridge_config = json.loads(options.pc_trans_config.read_text())
    if Path(bridge_config["bridge_root"]).resolve() != options.bridge_root:
        raise ValueError("--bridge-root disagrees with pc_trans config")
    index = load_shard_index(options.shard_index_path)
    validate_shard_index_for_training(index, config)
    checkpoint_path = Path(manifest["checkpoint_path"]).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 does not match model manifest")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key, expected in (("shard_index_sha256", index.index_sha256),
                          ("parent_dataset_sha256", index.parent_dataset_sha256),
                          ("training_stage", config.stage.value)):
        if manifest[key] != expected or checkpoint[key] != expected:
            raise ValueError(f"manifest/checkpoint/index mismatch: {key}")
    for key, expected, declared in (
        ("supervision_protocol", config.supervision_protocol, manifest.get("supervision_protocol", "legacy_v1")),
        ("reference_guard_protocol", config.reference_guard_protocol,
         manifest.get("preprocessing", {}).get("reference_guard_protocol", "none")),
    ):
        if checkpoint.get(key, "legacy_v1" if key == "supervision_protocol" else "none") != expected or declared != expected:
            raise ValueError(f"manifest/checkpoint/config mismatch: {key}")
    model = _new_model(config, torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint
    sources = [Path(__file__), ROOT / "scripts" / "target_state_audit_evidence.py",
               ROOT / "scripts" / "inspect_target_state_episode.py",
               ROOT / "perception" / "ray_measurement_gate.py",
               ROOT / "perception" / "rgbd_consistency.py",
               *sorted((ROOT / "training" / "target_state").glob("*.py")),
               *sorted((ROOT / "datasets" / "target_state").glob("*.py"))]
    contract = {
        "schema_version": 3, "measurement_protocol": MEASUREMENT_PROTOCOL,
        "case_export": {"max_bytes": case_options.max_bytes, "error_m": case_options.outlier_error_m,
                        "episode_ids": list(case_options.episode_ids)},
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": checkpoint_sha, "index_sha256": index.index_sha256,
        "run_id_prefix": options.run_id_prefix, "output_dir": str(output_dir),
        "bridge_root": str(options.bridge_root), "device": args.device,
        "torch_version": str(torch.__version__),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
    }
    contract_sha = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    entries = [entry for entry in index.shards if entry.split in ("validation", "test")]
    available = {episode for entry in entries for episode in entry.episode_ids}
    if set(case_options.episode_ids) - available:
        raise ValueError("requested case episode is not in validation/test; do not add training data to this evaluation")
    plan = {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
            "run_ids": [f"{options.run_id_prefix}.{split}" for split in ("validation", "test")],
            "shards": len(entries), "archive_bytes": sum(e.archive_size_bytes for e in entries),
            "sequences": sum(e.sequence_count for e in entries), "output_dir": str(output_dir),
            "case_assets_max_bytes": case_options.max_bytes, "case_episode_ids": list(case_options.episode_ids)}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    device = _device(args.device)
    model.to(device)
    lifecycle = PCTransCLI(options.pc_trans_root, options.pc_trans_config, options.pc_trans_python)
    # A prefix has one persistent owner, so changing output directories cannot
    # silently reuse already-consumed audit shards or race another evaluator.
    owner_dir = options.bridge_root / "control" / "audits" / options.run_id_prefix
    with _exclusive_run_lock(owner_dir), _exclusive_run_lock(output_dir):
        for path in (owner_dir / "contract.json", output_dir / "contract.json"):
            if path.exists() and json.loads(path.read_text()) != contract:
                raise ValueError(f"audit contract changed: {path}; choose a new audit prefix/output directory")
        for path in (owner_dir / "contract.json", output_dir / "contract.json"):
            _atomic_write_json(path, contract)
        results = {}
        for split in ("validation", "test"):
            results[split] = audit_split(
                split=split, config=config, options=options, index=index, lifecycle=lifecycle,
                model=model, device=device, output_dir=output_dir, contract_sha=contract_sha,
                case_options=case_options,
            )
            differences = compare_metrics(results[split]["evaluation_metrics"], manifest[f"{split}_metrics"])
            results[split]["training_metric_replay"] = {"matched": not differences, "differences": differences}
        _atomic_write_json(output_dir / "report.json", {
            "complete": True, "contract": contract, "results": results,
            "original_promotion_unchanged": manifest["promotion"],
            "original_metrics": {s: manifest[f"{s}_metrics"] for s in ("validation", "test")},
            "notes": ["Accepted means offline evaluator acceptance, not runtime Kalman acceptance.",
                      "Failure attribution is ordered/descriptive, not a causal proof.",
                      "Offline label subgroups and failure flags overlap; they never gate predictions.",
                      "No probability threshold sweep or test-set tuning was performed.",
                      "Ground truth is used only as offline evaluation labels.",
                      "V2 position metrics exclude rejected samples; failure rates still use all visible targets.",
                      "Use both_accepted for matched comparison; original promotion and artifacts are unchanged."],
        })
    print(f"Audit complete: {output_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
