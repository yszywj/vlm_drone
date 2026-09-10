#!/usr/bin/env python3
"""CPU-only head experiments on an immutable V1 feature cache.

The original producer stays byte-for-byte unchanged: no cache hash bypass or
manifest migration, no extraction, PC requests, raw-data cleanup or test fit.
"""
from __future__ import annotations

import argparse
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

from scripts import target_state_validity_probe as producer
from training.target_state.sharded_trainer import _RUN_LOCK_FILENAME

FIT_PROTOCOL = "cached_validity_weight_anchor_v2"
FIT_KEYS = {"epochs", "batch_size", "learning_rate", "weight_decay", "hard_example_weight",
            "seed", "class_balance_power", "anchor_logit_weight"}


def safe_name(value):
    return (isinstance(value, str) and bool(value)
            and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value))


def validate_fit(fit):
    if not isinstance(fit, dict) or set(fit) != FIT_KEYS:
        raise ValueError("invalid cached-fit settings")
    for key in ("epochs", "batch_size", "seed"):
        v = fit[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < (0 if key == "seed" else 1):
            raise ValueError(f"invalid fit.{key}")
    for key in FIT_KEYS - {"epochs", "batch_size", "seed"}:
        v = fit[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"invalid fit.{key}")
    if fit["learning_rate"] == 0 or fit["hard_example_weight"] == 0:
        raise ValueError("learning_rate and hard_example_weight must be positive")
    if not 0 <= fit["class_balance_power"] <= 1:
        raise ValueError("class_balance_power must be within [0,1]")


def load_suite(path):
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {"base_config", "suite_name", "fit", "experiments"}:
        raise ValueError("invalid suite config keys")
    if not safe_name(raw["suite_name"]):
        raise ValueError("invalid suite_name")
    base_path = Path(raw["base_config"])
    if not base_path.is_absolute():
        base_path = path.resolve().parent / base_path
    settings = producer.load_settings(base_path.resolve())
    if not isinstance(raw["experiments"], list) or not raw["experiments"]:
        raise ValueError("at least one experiment is required")
    validate_fit(raw["fit"])
    experiments, names = [], set()
    for experiment in raw["experiments"]:
        if not isinstance(experiment, dict) or set(experiment) != {"name", "overrides"}:
            raise ValueError("experiment requires name and overrides")
        name, overrides = experiment["name"], experiment["overrides"]
        if not safe_name(name) or name in names:
            raise ValueError("duplicate or unsafe experiment name")
        if not isinstance(overrides, dict) or set(overrides) - FIT_KEYS:
            raise ValueError("unknown experiment fit override")
        fit = {**raw["fit"], **overrides}
        validate_fit(fit)
        experiments.append({"name": name, "fit": fit})
        names.add(name)
    return settings, {"suite_name": raw["suite_name"], "experiments": experiments}


def validate_cached_split(features, rows, split, initial_head):
    if (split not in producer.SPLITS or not isinstance(features, torch.Tensor)
            or features.ndim != 2 or len(features) != len(rows) or features.requires_grad
            or features.device.type != "cpu" or not torch.isfinite(features).all()):
        raise ValueError("invalid detached CPU cache features")
    if any(row["split"] != split for row in rows):
        raise ValueError("cached split contamination")
    with torch.no_grad():
        probabilities = torch.sigmoid(F.linear(features, initial_head["weight"], initial_head["bias"]).flatten())
    old_probabilities = torch.tensor([r["validity_probability"] for r in rows], dtype=features.dtype)
    if not torch.allclose(probabilities, old_probabilities, atol=1e-5, rtol=1e-5):
        raise ValueError("features do not reproduce the original head probabilities")
    replay = producer.regate(rows, probabilities)
    if [r["model_valid"] for r in replay] != [r["model_valid"] for r in rows]:
        raise ValueError("cached original acceptance cannot be reproduced")
    for row in rows:
        eligible = bool(row["reference_input_valid"] and row["model_geometry_valid"])
        fit_mask = bool(split == "train" and eligible and row["offline_validity_supervised"]
                        and not row["probe_review_reasons"])
        if (row["probe_eligible"] != eligible or row["probe_fit_mask"] != fit_mask
                or row["probe_label"] != bool(row["offline_measurement_supervision_positive"])):
            raise ValueError("cached supervision contract mismatch")


def read_cache(settings):
    """Keep strict V1 source/model/receipt/asset verification; never regenerate."""
    _, index, model, _, expected_contract = producer.preflight(settings)
    root = settings["output_dir"]
    path = root / "cache_manifest.json"
    cache = json.loads(path.read_text())
    if (cache.get("complete") is not True or cache.get("contract") != expected_contract
            or cache.get("test_split_used") is not False):
        raise ValueError("a complete, unchanged V1 cache is required; no automatic extraction is performed")
    entries = [entry for split in producer.SPLITS for entry in index.shards_for_split(split)]
    if set(cache["receipt_sha256"]) != {e.filename for e in entries}:
        raise ValueError("missing/extra cache split receipts")
    initial_head = {key: value.detach().cpu().clone() for key, value in model.validity_head.state_dict().items()}
    prepared = {}
    for split in producer.SPLITS:
        features, rows = [], []
        for entry in index.shards_for_split(split):
            receipt_path = root / "receipts" / (entry.filename + ".pt")
            if producer.sha256_file(receipt_path) != cache["receipt_sha256"][entry.filename]:
                raise ValueError("feature receipt SHA256 mismatch")
            receipt = producer.load_feature_receipt(receipt_path,
                contract_sha=producer.digest(expected_contract), entry=entry, output_dir=root)
            features.append(receipt["features"])
            rows.extend(receipt["rows"])
        x = torch.cat(features)
        validate_cached_split(x, rows, split, initial_head)
        prepared[split] = x, rows
    return prepared, initial_head, {
        "cache_manifest_sha256": producer.sha256_file(path),
        "feature_contract_sha256": producer.digest(expected_contract),
        "base_checkpoint_sha256": expected_contract["checkpoint_sha256"],
        "producer_source_sha256": expected_contract["source_sha256"]["scripts/target_state_validity_probe.py"],
        "receipt_count": len(entries), "test_split_used": False,
    }


def sample_weights(labels, original_probabilities, fit):
    """power=0: empirical per-example BCE; power=1: old full balancing."""
    positives, negatives = int(labels.sum()), int((1-labels).sum())
    if not positives or not negatives:
        raise ValueError("qualified positive and negative TRAIN examples are required")
    negative_scale = (positives / negatives) ** fit["class_balance_power"]
    hard = (original_probabilities >= .5) != labels.bool()
    weights = torch.where(labels.bool(), 1.0, negative_scale)
    weights *= torch.where(hard, fit["hard_example_weight"], 1.0)
    weights *= len(weights) / weights.sum()
    return weights, {
        "train_positive_count": positives, "train_negative_count": negatives,
        "train_hard_positive_count": int((hard & labels.bool()).sum()),
        "train_hard_negative_count": int((hard & ~labels.bool()).sum()),
        "negative_to_positive_base_weight_ratio": negative_scale,
        "negative_sample_fraction": negatives / len(labels),
        "negative_loss_weight_fraction": float(weights[~labels.bool()].sum() / weights.sum()),
    }


def objective(logits, labels, weights, original_logits, anchor_weight):
    bce = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean()
    # Soft output preservation on qualified TRAIN only, not validation/test.
    # Work with logits so confident examples still constrain large score drift.
    anchor_mse = (logits - original_logits.detach()).square().mean()
    return bce + anchor_weight * anchor_mse, bce, anchor_mse


def shift_summary(delta):
    values = delta.detach().abs()
    return {"count": values.numel(), "mean_abs_logit_shift": float(values.mean()) if values.numel() else None,
            "max_abs_logit_shift": float(values.max()) if values.numel() else None}


def fit_head(train, validation, initial_head, fit):
    validate_fit(fit)
    x, rows = train
    vx, vrows = validation
    validate_cached_split(x, rows, "train", initial_head)
    validate_cached_split(vx, vrows, "validation", initial_head)
    mask = torch.tensor([r["probe_fit_mask"] for r in rows], dtype=torch.bool)
    tx = x[mask]
    y = torch.tensor([r["probe_label"] for r in rows], dtype=torch.float32)[mask]
    torch.manual_seed(fit["seed"])
    head = nn.Linear(x.shape[1], 1)
    head.load_state_dict(initial_head, strict=True)
    with torch.no_grad():
        original_logits = head(tx).flatten().detach().clone()
        original_validation_logits = head(vx).flatten().detach().clone()
    weights, weight_stats = sample_weights(y, torch.sigmoid(original_logits), fit)
    baseline = producer.validation_score(vrows)
    optimizer = torch.optim.AdamW(head.parameters(), lr=fit["learning_rate"], weight_decay=fit["weight_decay"])
    generator = torch.Generator().manual_seed(fit["seed"])
    best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    best_epoch, best_score, history = 0, baseline, []
    for epoch in range(1, fit["epochs"]+1):
        sums = dict(objective=0.0, bce=0.0, anchor_mse=0.0)
        for indices in torch.randperm(len(tx), generator=generator).split(fit["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = head(tx[indices]).flatten()
            total, bce, anchor = objective(logits, y[indices], weights[indices], original_logits[indices], fit["anchor_logit_weight"])
            if not torch.isfinite(total):
                raise ValueError("non-finite cached-fit objective")
            total.backward()
            optimizer.step()
            for name, value in (("objective", total), ("bce", bce), ("anchor_mse", anchor)):
                sums[name] += float(value.detach()) * len(indices)
        with torch.no_grad():
            validation_logits = head(vx).flatten()
            score = producer.validation_score(producer.regate(vrows, torch.sigmoid(validation_logits)))
            shifts = {"train_fit": shift_summary(head(tx).flatten() - original_logits),
                      "validation": shift_summary(validation_logits - original_validation_logits)}
        admissible = producer.admissible(score, baseline)
        if admissible and score["positive_supervision_rejected_count"] < best_score["positive_supervision_rejected_count"]:
            best_epoch, best_score = epoch, score
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": {k: v / len(tx) for k, v in sums.items()},
                        "admissible": admissible, "validation": score, "logit_shift": shifts})
        print(f"[head {epoch}/{fit['epochs']}] loss={sums['objective']/len(tx):.5f} "
              f"positive_rejected={score['positive_supervision_rejected_count']} "
              f"false_positive={score['no_target_false_positive_count']} admissible={admissible}", flush=True)
    return best_state, {
        "complete": True, "protocol": FIT_PROTOCOL, "candidate_improved": best_epoch > 0,
        "best_epoch": best_epoch, "baseline_validation": baseline, "candidate_validation": best_score,
        "fit": fit, "weight_statistics": weight_stats, "history": history,
        "trainable_parameter_count": sum(p.numel() for p in head.parameters()),
        "acceptance_threshold": .5, "geometry_parameters_changed": False,
        "geometry_bn_buffers_changed": False, "test_split_used": False, "promotion_passed": False,
        "notes": ["Only a separate validity head is trained from fixed cached features.",
                  "TRAIN alone supplies BCE, class counts, hard weights and logit-anchor loss.",
                  "The anchor is a soft constraint, not a hard bound on probability/logit changes.",
                  "Validation-only epoch selection uses unchanged safety constraints and threshold 0.5.",
                  "No admissible improvement retains the original head at epoch 0.",
                  "A development experiment, not a target-identity repair or independent test result."]}


def run_experiment(output, prepared, initial_head, fit, cache_identity):
    contract = {"protocol": FIT_PROTOCOL, "cache": cache_identity, "fit": fit,
                "fitter_source_sha256": producer.sha256_file(Path(__file__))}
    with producer._exclusive_run_lock(output):
        contract_path, report_path = output / "contract.json", output / "report.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("experiment contract changed; use a new suite_name")
        if report_path.exists():
            result = json.loads(report_path.read_text())
            if (result.get("complete") is not True or result.get("contract") != contract
                    or result["head_sha256"] != producer.sha256_file(output / "best_validity_head.pt")):
                raise ValueError("existing experiment evidence mismatch")
            print(f"Already complete: {report_path}", flush=True)
            return result
        producer._atomic_write_json(contract_path, contract)
        source_before = {k: v.clone() for k,v in initial_head.items()}
        state, result = fit_head(prepared["train"], prepared["validation"], initial_head, fit)
        if any(not torch.equal(source_before[k], initial_head[k]) for k in source_before):
            raise ValueError("source head changed")
        path = output / "best_validity_head.pt"
        producer.save_receipt(path, {"artifact_type": "offline_validity_head_probe_only", "protocol": FIT_PROTOCOL,
            "base_checkpoint_sha256": cache_identity["base_checkpoint_sha256"], "head_state_dict": state,
            "fit_contract": contract, "best_epoch": result["best_epoch"], "deployable": False})
        result.update(contract=contract, head_sha256=producer.sha256_file(path))
        producer._atomic_write_json(report_path, result)
        return result


def run(settings, suite, *, dry_run=False):
    # CPU only; the producer's recorded cuda:0 setting remains provenance, not
    # a request to allocate GPU memory or run the original encoder again.
    torch.set_num_threads(4)
    prepared, initial_head, identity = read_cache(settings)
    output = settings["output_dir"] / suite["suite_name"]
    if output.is_symlink():
        raise ValueError("suite output must not be a symlink")
    if output.exists() and not (output / "suite_contract.json").is_file():
        # A crash can occur after lock creation but before contract commit.
        # Reuse only an empty/lock-only directory, never unrelated artifacts.
        if not output.is_dir() or any(p.name != _RUN_LOCK_FILENAME or p.is_symlink()
                or not p.is_file() or p.stat().st_size != 0 for p in output.iterdir()):
            raise ValueError("suite output already exists without ownership; use a new suite_name")
    plan = {"cache_verified": True, "receipt_count": identity["receipt_count"],
            "sample_counts": {s: len(rows) for s, (_, rows) in prepared.items()},
            "experiments": suite["experiments"], "output_dir": str(output),
            "execution_device": "cpu", "pc_upload_required": False, "test_split_used": False}
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return plan
    suite_contract = {"protocol": FIT_PROTOCOL, "cache": identity, "suite": suite,
                      "fitter_source_sha256": producer.sha256_file(Path(__file__))}
    with producer._exclusive_run_lock(output):
        cp = output / "suite_contract.json"
        if cp.exists() and json.loads(cp.read_text()) != suite_contract:
            raise ValueError("suite contract changed; use a new suite_name")
        producer._atomic_write_json(cp, suite_contract)
        results = {}
        for experiment in suite["experiments"]:
            print(f"Running cached experiment: {experiment['name']}", flush=True)
            results[experiment["name"]] = run_experiment(output / experiment["name"], prepared,
                initial_head, experiment["fit"], identity)
        improved = [name for name, result in results.items() if result["candidate_improved"]]
        selected = min(improved, key=lambda name: results[name]["candidate_validation"]["positive_supervision_rejected_count"]) if improved else None
        summary = {"complete": True, "suite_contract": suite_contract, "candidate_improved": bool(improved),
            "selected_experiment": selected, "selection_split": "validation", "test_split_used": False,
            "promotion_passed": False, "pc_upload_required": False,
            "baseline_validation": next(iter(results.values()))["baseline_validation"],
            "experiments": {name: {k: result[k] for k in ("best_epoch", "candidate_improved",
                "candidate_validation", "weight_statistics", "head_sha256")} for name, result in results.items()},
            "note": "Existing models/cache and head_balanced_v1 are unchanged; new heads are not deployable."}
        producer._atomic_write_json(output / "report.json", summary)
    print(f"Cached validity suite complete: {output / 'report.json'}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="verify the entire existing cache without training, uploads or writes")
    args = parser.parse_args(argv)
    settings, suite = load_suite(args.config)
    run(settings, suite, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
