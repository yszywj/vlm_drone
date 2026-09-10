#!/usr/bin/env python3
"""TRAIN/validation-only frozen-feature validity experiment; never deploy.

An uncertain association is masked, not relabelled. Runtime geometry and its
0.5 acceptance threshold are unchanged. No test split is requested or fitted.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from collections import Counter
from hashlib import sha256
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

from scripts.audit_target_state_sharded import diagnostic_rows, save_receipt, summarize_rows
from scripts.target_state_audit_evidence import CaseExportOptions, export_cases, verify_case_assets
from training.target_state.config import LossWeights, TargetStateTrainingConfig, TrainingStage
from training.target_state.sharded_trainer import (
    PCTransCLI, ShardedTrainingOptions, _atomic_write_json, _dataset_loader,
    _exclusive_run_lock, _model_config, _new_model, _prepare_active_shard, _release_loader,
    _state_name, validate_shard_index_for_training,
)
from training.target_state.shard_runtime import materialize_shard, cleanup_materialized_shard
from training.target_state.shards import load_shard_index
from training.target_state.trainer import _to_device, sha256_file

SPLITS = ("train", "validation")
PROTOCOL = "frozen_geometry_validity_probe_v1"


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_settings(path):
    settings = yaml.safe_load(path.read_text())
    required = {"model_manifest", "shard_index", "train_review_manifest", "pc_trans_root",
                "pc_trans_config", "bridge_root", "output_dir", "run_id_prefix", "experiment_name",
                "device", "num_workers", "wait_timeout_s", "case_assets_max_gib", "fit"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("invalid validity probe config keys")
    for key in ("model_manifest", "shard_index", "train_review_manifest", "pc_trans_root",
                "pc_trans_config", "bridge_root", "output_dir"):
        settings[key] = Path(settings[key]).expanduser().resolve()
    if not settings["run_id_prefix"].startswith("audit_validity_"):
        raise ValueError("use a dedicated audit_validity_* run prefix")
    name = settings["experiment_name"]
    if not isinstance(name, str) or not name or name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("experiment_name must be a safe single directory name")
    fit = settings["fit"]
    if set(fit) != {"epochs", "batch_size", "learning_rate", "weight_decay", "hard_example_weight", "seed"}:
        raise ValueError("invalid fit config keys")
    for k in ("epochs", "batch_size"):
        if isinstance(fit[k], bool) or not isinstance(fit[k], int) or fit[k] <= 0:
            raise ValueError(f"fit.{k} must be a positive integer")
    for k in ("learning_rate", "weight_decay", "hard_example_weight"):
        if not math.isfinite(fit[k]) or fit[k] < 0 or (k != "weight_decay" and fit[k] == 0):
            raise ValueError(f"invalid fit.{k}")
    if isinstance(fit["seed"], bool) or not isinstance(fit["seed"], int) or fit["seed"] < 0:
        raise ValueError("fit.seed must be a non-negative integer")
    budget = settings["case_assets_max_gib"]
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("case_assets_max_gib must be finite and non-negative")
    return settings


def training_findings(path, index):
    review = json.loads(path.read_text())
    if review.get("offline_only") is not True or review.get("action") != "review_only_no_automatic_relabel_or_deletion":
        raise ValueError("association source must be an offline review manifest")
    entries = {entry.filename: entry for entry in index.shards_for_split("train")}
    if review.get("counts", {}).get("frame_records") != sum(e.frame_count for e in entries.values()):
        raise ValueError("training review frame count differs from index")
    # Every cited finding must resolve to TRAIN, not validation or test.
    result = {}
    for row in review["findings"]:
        entry = entries.get(row["shard"])
        if entry is None or row["episode_id"] not in entry.episode_ids:
            raise ValueError("association finding is outside the training split")
        key = (row["episode_id"], row["frame_id"])
        if key in result:
            raise ValueError("duplicate training review finding")
        result[key] = tuple(row["reasons"])
    # Bind old evidence to the original index through its companion contract.
    parent_report = json.loads((path.parent / "report.json").read_text())
    contract = parent_report["contract"]
    if (parent_report.get("complete") is not True or contract["index_sha256"] != index.index_sha256
            or contract["parent_dataset_sha256"] != index.parent_dataset_sha256
            or digest(contract) != review["contract_sha256"]):
        raise ValueError("training association review contract mismatch")
    return result


def review_window(sequence, findings):
    """Offline ambiguity hints only; none of these fields is a model input."""
    frames = (*sequence.history, sequence.reference)
    reasons = set()
    for frame in frames:
        reasons.update(findings.get((frame.episode_id, frame.frame_id), ()))
        if frame.association_review_required:
            reasons.add("collector_unresolved")
    instances = {frame.training_label.instance_id for frame in frames
                 if frame.training_label is not None and frame.training_label.instance_id is not None}
    if len(instances) > 1:
        reasons.add("multiple_label_instances_in_window")
    present = [frame.training_label is not None for frame in frames]
    if any(a != b for a, b in zip(present, present[1:])):
        reasons.add("null_positive_transition_in_window")
    return sorted(reasons)


def feature_contract(settings, config, index, checkpoint_sha):
    sources = [Path(__file__), ROOT / "scripts/audit_target_state_sharded.py",
               ROOT / "scripts/target_state_audit_evidence.py", ROOT / "scripts/inspect_target_state_episode.py",
               ROOT / "perception/rgbd_consistency.py", ROOT / "perception/ray_measurement_gate.py",
               *sorted((ROOT / "training/target_state").glob("*.py")),
               *sorted((ROOT / "datasets/target_state").glob("*.py"))]
    return {"protocol": PROTOCOL, "splits": list(SPLITS), "test_split_used": False,
            "checkpoint_sha256": checkpoint_sha, "manifest_sha256": sha256_file(settings["model_manifest"]),
            "index_sha256": index.index_sha256, "parent_dataset_sha256": index.parent_dataset_sha256,
            "review_sha256": sha256_file(settings["train_review_manifest"]),
            "run_id_prefix": settings["run_id_prefix"], "output_dir": str(settings["output_dir"]),
            "bridge_root": str(settings["bridge_root"]), "device": settings["device"],
            "torch_version": str(torch.__version__), "model_config": _model_config(config),
            "case_assets_max_bytes": int(settings["case_assets_max_gib"] * 1024**3),
            "supervision_policy": "v2_labels_mask_train_review_and_window_ambiguity_no_relabel_v1",
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}}


def preflight(settings):
    manifest = json.loads(settings["model_manifest"].read_text())
    raw = dict(manifest["config"])
    raw["loss_weights"] = LossWeights(**raw["loss_weights"])
    config = replace(TargetStateTrainingConfig(**raw), device=settings["device"], num_workers=settings["num_workers"])
    if (config.stage != TrainingStage.YOLO_DEPLOYMENT or config.supervision_protocol != "projected_center_v2"
            or config.reference_guard_protocol != "rgbd_consistency_v1"):
        raise ValueError("this experiment requires a Geometry V2 Stage B model")
    index = load_shard_index(settings["shard_index"])
    validate_shard_index_for_training(index, config)
    checkpoint_path = Path(manifest["checkpoint_path"]).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key, expected in (("shard_index_sha256", index.index_sha256),
                          ("parent_dataset_sha256", index.parent_dataset_sha256),
                          ("training_stage", config.stage.value),
                          ("supervision_protocol", config.supervision_protocol),
                          ("reference_guard_protocol", config.reference_guard_protocol)):
        declared = manifest.get("preprocessing", {}).get(key) if key == "reference_guard_protocol" else manifest.get(key)
        if checkpoint.get(key) != expected or declared != expected:
            raise ValueError(f"model/manifest contract mismatch: {key}")
    model = _new_model(config, torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    findings = training_findings(settings["train_review_manifest"], index)
    bridge = Path(json.loads(settings["pc_trans_config"].read_text())["bridge_root"]).resolve()
    if bridge != settings["bridge_root"]:
        raise ValueError("bridge config mismatch")
    output = settings["output_dir"]
    for protected in (bridge, checkpoint_path.parent, settings["model_manifest"].parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("probe output must be separate from model and bridge")
    contract = feature_contract(settings, config, index, checkpoint_sha)
    return config, index, model, findings, contract


@torch.no_grad()
def extract_archive(archive, *, index, entry, config, model, device, findings, output_dir, case_options):
    if entry.split not in SPLITS:
        raise ValueError("test extraction is forbidden in validity probe")
    materialized = materialize_shard(archive, index=index)
    loader = None
    captured = []
    hook = model.validity_head.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().cpu().clone()))
    try:
        model.eval()
        dataset, loader = _dataset_loader(config=config, dataset_root=materialized.dataset_root,
            split=entry.split, device=device, shuffle=False, generator_seed=None, split_seed=index.split_seed)
        rows, features = [], []
        for batch in loader:
            captured.clear()
            batch = _to_device(batch, device)
            output = model(batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
            if len(captured) != 1:
                raise ValueError("validity feature hook was not called exactly once")
            feature = captured[0]
            reconstructed = F.linear(feature, model.validity_head.weight.detach().cpu(), model.validity_head.bias.detach().cpu()).flatten()
            if not torch.allclose(reconstructed, output.measurement_valid_logit.cpu(), atol=1e-5, rtol=1e-5):
                raise ValueError("cached features do not reproduce original validity logits")
            batch_rows = diagnostic_rows(batch, output, maximum_depth_m=config.maximum_depth_m)
            start = len(rows)
            for sequence, row in zip(dataset.sequences[start:start+len(batch_rows)], batch_rows):
                frame = sequence.reference
                detection = frame.detector_prediction
                reasons = review_window(sequence, findings if entry.split == "train" else {})
                eligible = row["reference_input_valid"] and row["model_geometry_valid"]
                row.update(shard=entry.filename, split=entry.split, sequence_id=sequence.sequence_id,
                    episode_id=frame.episode_id, frame_id=frame.frame_id, timestamp_s=frame.timestamp_s,
                    tracker_id=detection.tracker_id, candidate_id=detection.candidate_id,
                    detected=detection.detected, bbox_xyxy_normalized=detection.bbox_xyxy_normalized,
                    rgb_path=frame.sensor_input.rgb_path, depth_path=frame.sensor_input.depth_path,
                    probe_review_reasons=reasons, probe_eligible=bool(eligible),
                    probe_fit_mask=bool(entry.split == "train" and eligible
                        and row["offline_validity_supervised"] and not reasons),
                    probe_label=bool(row["offline_measurement_supervision_positive"]))
            features.append(feature)
            rows.extend(batch_rows)
        if len(rows) != entry.sequence_count or len(rows) != len(dataset):
            raise ValueError("probe sequence count mismatch")
        evidence = export_cases(rows, dataset.sequences, dataset_root=materialized.dataset_root,
            output_dir=output_dir, entry=entry, options=case_options)
        return materialized, {"rows": rows, "features": torch.cat(features), "case_exports": evidence}
    finally:
        hook.remove()
        _release_loader(loader)


def load_feature_receipt(path, *, contract_sha, entry, output_dir):
    receipt = torch.load(path, map_location="cpu", weights_only=True)
    if receipt.get("contract_sha256") != contract_sha or receipt.get("entry") != entry.to_dict():
        raise ValueError("feature receipt identity mismatch")
    rows, features = receipt["rows"], receipt["features"]
    if (len(rows) != entry.sequence_count or features.ndim != 2 or features.shape[0] != len(rows)
            or not torch.isfinite(features).all() or features.device.type != "cpu"):
        raise ValueError("invalid feature receipt tensors/counts")
    if any(r["split"] != entry.split or r["episode_id"] not in entry.episode_ids for r in rows):
        raise ValueError("feature receipt split leakage")
    verify_case_assets(receipt["case_exports"], output_dir)
    return receipt


def prepare(settings, *, dry_run=False, lifecycle=None):
    config, index, model, findings, contract = preflight(settings)
    entries = [e for s in SPLITS for e in index.shards_for_split(s)]
    plan = {"checkpoint_sha256": contract["checkpoint_sha256"], "splits": list(SPLITS),
            "run_ids": [f"{settings['run_id_prefix']}.{s}" for s in SPLITS],
            "shards": len(entries), "archive_bytes": sum(e.archive_size_bytes for e in entries),
            "sequences": sum(e.sequence_count for e in entries), "test_split_used": False,
            "output_dir": str(settings["output_dir"]), "case_asset_budget_bytes": contract["case_assets_max_bytes"]}
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return plan
    device = torch.device(settings["device"])
    model.to(device)
    options = ShardedTrainingOptions(shard_index_path=settings["shard_index"],
        pc_trans_root=settings["pc_trans_root"], pc_trans_config=settings["pc_trans_config"],
        bridge_root=settings["bridge_root"], run_id_prefix=settings["run_id_prefix"],
        wait_timeout_s=settings["wait_timeout_s"])
    client = lifecycle or PCTransCLI(options.pc_trans_root, options.pc_trans_config, options.pc_trans_python)
    output = settings["output_dir"]
    owner = options.bridge_root / "control/audits" / options.run_id_prefix
    contract_sha = digest(contract)
    case_options = CaseExportOptions(contract["case_assets_max_bytes"])
    with _exclusive_run_lock(owner), _exclusive_run_lock(output):
        for p in (owner / "contract.json", output / "contract.json"):
            if p.exists() and json.loads(p.read_text()) != contract:
                raise ValueError("feature contract changed; use a new prefix/output")
        for p in (owner / "contract.json", output / "contract.json"):
            _atomic_write_json(p, contract)
        receipt_hashes, results = {}, {}
        for split in SPLITS:
            rows = []
            run_id = f"{options.run_id_prefix}.{split}"
            split_entries = index.shards_for_split(split)
            client.request(run_id, [e.filename for e in split_entries])
            for ordinal, entry in enumerate(split_entries, 1):
                path = output / "receipts" / (entry.filename + ".pt")
                state = client.shard_state(run_id, entry.filename)
                materialized = None
                if path.exists():
                    receipt = load_feature_receipt(path, contract_sha=contract_sha, entry=entry, output_dir=output)
                    if _state_name(state) != "consumed" or state["deleted"] is not True:
                        archive = _prepare_active_shard(lifecycle=client, options=options, run_id=run_id, entry=entry)
                        materialized = materialize_shard(archive, index=index)
                else:
                    if _state_name(state) == "consumed":
                        raise ValueError("consumed shard has no feature receipt; use a new prefix/output")
                    print(f"[{split} {ordinal}/{len(split_entries)}] waiting/extracting {entry.filename}", flush=True)
                    archive = _prepare_active_shard(lifecycle=client, options=options, run_id=run_id, entry=entry)
                    materialized, receipt = extract_archive(archive, index=index, entry=entry, config=config,
                        model=model, device=device, findings=findings, output_dir=output, case_options=case_options)
                    receipt.update(contract_sha256=contract_sha, entry=entry.to_dict())
                    save_receipt(path, receipt)
                    receipt = load_feature_receipt(path, contract_sha=contract_sha, entry=entry, output_dir=output)
                receipt_hashes[entry.filename] = sha256_file(path)
                if materialized is not None:
                    cleanup_materialized_shard(materialized)
                    client.consume(run_id, entry.filename, delete=True)
                rows.extend(receipt["rows"])
                print(f"[{split} {ordinal}/{len(split_entries)}] features/evidence saved; server cache consumed", flush=True)
            results[split] = supervision_summary(rows)
            _atomic_write_json(output / f"{split}_review.json", {"offline_only": True,
                "summary": results[split], "review_rows": [r for r in rows if r["probe_review_reasons"]
                    or (r["no_target"] and r["model_valid"])
                    or (r["offline_measurement_supervision_positive"] and r["baseline_valid"] and not r["model_valid"])]})
        _atomic_write_json(output / "cache_manifest.json", {"complete": True, "contract": contract,
            "receipt_sha256": receipt_hashes, "results": results, "test_split_used": False,
            "note": "Review flags are ambiguity, not proven wrong labels. No source data was relabelled."})
    print(f"Feature cache complete: {output / 'cache_manifest.json'}", flush=True)
    return results


def supervision_summary(rows):
    return {"sample_count": len(rows), "diagnostics": summarize_rows(rows),
            "fit_positive_count": sum(r["probe_fit_mask"] and r["probe_label"] for r in rows),
            "fit_negative_count": sum(r["probe_fit_mask"] and not r["probe_label"] for r in rows),
            "review_window_count": sum(bool(r["probe_review_reasons"]) for r in rows),
            "review_reason_counts_overlapping": dict(Counter(reason for r in rows for reason in r["probe_review_reasons"])),
            "positive_supervision_rejected_count": sum(r["offline_measurement_supervision_positive"]
                and r["baseline_valid"] and not r["model_valid"] for r in rows)}


def regate(rows, probabilities):
    result = []
    for old, p in zip(rows, probabilities.tolist()):
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("non-finite/out-of-range validity probability")
        row = dict(old)
        row["validity_probability"] = p
        row["model_valid"] = bool(row["probe_eligible"] and p >= .5)
        flags = dict(row["failure_flags"])
        flags["validity_head_rejected"] = p < .5
        row["failure_flags"] = flags
        row["model_failure_reason"] = None if row["model_valid"] else next(
            (name for name, failed in flags.items() if failed), "unclassified_gate_rejection")
        result.append(row)
    if len(result) != len(rows):
        raise ValueError("probability count mismatch")
    return result


def validation_score(rows):
    return {
        "positive_supervision_rejected_count": sum(r["evaluated_visible_target"]
            and r["offline_measurement_supervision_positive"] and r["baseline_valid"] and not r["model_valid"] for r in rows),
        "no_target_false_positive_count": sum(r["no_target"] and r["model_valid"] for r in rows),
        "out_of_domain_accepted_count": sum(r["evaluated_visible_target"] and r["model_valid"]
            and not r["offline_target_in_output_domain"] for r in rows),
        "accepted_over_1m_count": sum(r["evaluated_visible_target"] and r["model_valid"] and r["model_error_m"] > 1 for r in rows),
        "diagnostics": summarize_rows(rows),
    }


def admissible(score, baseline):
    # Tune on validation ONLY; never relax the current fixed 0.5 sensor gate.
    return all(score[key] <= baseline[key] for key in (
        "no_target_false_positive_count", "out_of_domain_accepted_count", "accepted_over_1m_count"))


def fit_head(train, validation, initial_head, fit):
    """Only a separate 129-parameter Linear head is trained. No encoder/BN/GRU."""
    x, rows = train
    vx, vrows = validation
    if any(r["split"] != "train" for r in rows) or any(r["split"] != "validation" for r in vrows):
        raise ValueError("fit_head requires TRAIN gradients and validation-only selection")
    mask = torch.tensor([r["probe_fit_mask"] for r in rows], dtype=torch.bool)
    y = torch.tensor([r["probe_label"] for r in rows], dtype=torch.float32)[mask]
    tx = x[mask]
    positives, negatives = int(y.sum()), int((1-y).sum())
    if not positives or not negatives:
        raise ValueError("no qualified positive or negative training examples; review supervision before fitting")
    torch.manual_seed(fit["seed"])
    head = nn.Linear(x.shape[1], 1)
    head.load_state_dict(initial_head, strict=True)
    with torch.no_grad():
        original_train = torch.sigmoid(head(tx).flatten())
        original_validation = torch.sigmoid(head(vx).flatten())
    baseline_rows = regate(vrows, original_validation)
    if [r["model_valid"] for r in baseline_rows] != [r["model_valid"] for r in vrows]:
        raise ValueError("cached baseline acceptance cannot be reproduced")
    baseline = validation_score(baseline_rows)
    hard = (original_train >= .5) != y.bool()
    weights = torch.where(y.bool(), .5 / positives, .5 / negatives)
    weights *= torch.where(hard, fit["hard_example_weight"], 1.0)
    weights *= len(weights) / weights.sum()
    optimizer = torch.optim.AdamW(head.parameters(), lr=fit["learning_rate"], weight_decay=fit["weight_decay"])
    generator = torch.Generator().manual_seed(fit["seed"])
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best_score, best_epoch, history = baseline, 0, []
    for epoch in range(1, fit["epochs"]+1):
        order = torch.randperm(len(tx), generator=generator)
        loss_sum = 0.0
        for batch_indices in order.split(fit["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = head(tx[batch_indices]).flatten()
            loss = (F.binary_cross_entropy_with_logits(logits, y[batch_indices], reduction="none") * weights[batch_indices]).mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite head loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_indices)
        with torch.no_grad():
            score = validation_score(regate(vrows, torch.sigmoid(head(vx).flatten())))
        eligible = admissible(score, baseline)
        if eligible and score["positive_supervision_rejected_count"] < best_score["positive_supervision_rejected_count"]:
            best_epoch, best_score = epoch, score
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": loss_sum / len(tx),
                        "admissible": eligible, "validation": score})
        print(f"[head {epoch}/{fit['epochs']}] loss={loss_sum/len(tx):.5f} "
              f"positive_rejected={score['positive_supervision_rejected_count']} "
              f"false_positive={score['no_target_false_positive_count']} admissible={eligible}", flush=True)
    return best_state, {"complete": True, "candidate_improved": best_epoch > 0, "best_epoch": best_epoch,
        "train_positive_count": positives, "train_negative_count": negatives,
        "train_hard_positive_count": int((hard & y.bool()).sum()),
        "train_hard_negative_count": int((hard & ~y.bool()).sum()),
        "baseline_validation": baseline, "candidate_validation": best_score, "history": history,
        "trainable_parameter_count": sum(p.numel() for p in head.parameters()),
        "geometry_parameters_changed": False, "geometry_bn_buffers_changed": False,
        "test_split_used": False, "promotion_passed": False,
        "notes": ["A frozen-feature classifier experiment, NOT a deployable checkpoint or a semantic identity fix.",
                  "Only TRAIN rows contribute gradients/weights. Validation chooses an epoch at the fixed 0.5 threshold.",
                  "Unknown/review-flagged TRAIN windows are masked, never relabelled as negatives.",
                  "Existing labels may remain imperfect; masking is not a certified dataset repair.",
                  "If no admissible improvement exists, epoch 0 retains the original head.",
                  "Fresh independent scenes and runtime validation are still required before promotion."]}


def train(settings):
    config, index, model, _, contract = preflight(settings)
    root = settings["output_dir"]
    cache_path = root / "cache_manifest.json"
    cache = json.loads(cache_path.read_text())
    if cache.get("complete") is not True or cache.get("contract") != contract or cache.get("test_split_used") is not False:
        raise ValueError("complete matching feature cache required before fitting")
    expected_entries = [entry for split in SPLITS for entry in index.shards_for_split(split)]
    if set(cache["receipt_sha256"]) != {e.filename for e in expected_entries}:
        raise ValueError("cache contains missing/extra split receipts")
    prepared = {}
    for split in SPLITS:
        features, rows = [], []
        for entry in index.shards_for_split(split):
            path = root / "receipts" / (entry.filename + ".pt")
            if sha256_file(path) != cache["receipt_sha256"][entry.filename]:
                raise ValueError("feature receipt SHA256 mismatch")
            receipt = load_feature_receipt(path, contract_sha=digest(contract), entry=entry, output_dir=root)
            features.append(receipt["features"])
            rows.extend(receipt["rows"])
        prepared[split] = torch.cat(features), rows
    experiment = root / settings["experiment_name"]
    fit_contract = {"protocol": PROTOCOL, "cache_manifest_sha256": sha256_file(cache_path),
                    "fit": settings["fit"], "source_sha256": sha256_file(Path(__file__)),
                    "checkpoint_sha256": contract["checkpoint_sha256"]}
    with _exclusive_run_lock(experiment):
        cp = experiment / "contract.json"
        if cp.exists() and json.loads(cp.read_text()) != fit_contract:
            raise ValueError("fit contract changed; choose a NEW experiment_name to reuse the cache")
        result_path = experiment / "report.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("complete") is not True or result["head_sha256"] != sha256_file(experiment / "best_validity_head.pt"):
                raise ValueError("existing experiment evidence mismatch")
            print(f"Head experiment already complete: {result_path}", flush=True)
            return result
        _atomic_write_json(cp, fit_contract)
        torch.set_num_threads(4)
        frozen_before = {k: v.clone() for k, v in model.state_dict().items()}
        head, result = fit_head(prepared["train"], prepared["validation"], model.validity_head.state_dict(), settings["fit"])
        if any(not torch.equal(v, model.state_dict()[k]) for k, v in frozen_before.items()):
            raise ValueError("frozen source model changed during fitting")
        head_path = experiment / "best_validity_head.pt"
        save_receipt(head_path, {"artifact_type": "offline_validity_head_probe_only", "protocol": PROTOCOL,
            "base_checkpoint_sha256": contract["checkpoint_sha256"], "head_state_dict": head,
            "fit_contract": fit_contract, "best_epoch": result["best_epoch"], "deployable": False})
        result.update(fit_contract=fit_contract, head_sha256=sha256_file(head_path))
        _atomic_write_json(result_path, result)
    print(f"Validity head experiment complete: {result_path}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.command == "prepare":
        prepare(settings, dry_run=args.dry_run)
    elif args.dry_run:
        raise ValueError("--dry-run is supported by prepare only")
    else:
        train(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
