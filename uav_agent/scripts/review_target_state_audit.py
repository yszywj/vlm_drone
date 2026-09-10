#!/usr/bin/env python3
"""Reanalyse saved predictions without GPU, PC transfer, or artifact changes.

Old rows lack image dimensions, so this is explicitly a partial gate replay,
not a full V2 evaluation or a production promotion decision. Threshold sweeps
and the candidate covariance scale are derived from validation only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_target_state_sharded import summarize_rows
from perception.ray_measurement_gate import MEASUREMENT_PROTOCOL, reference_depth_valid
from training.target_state.sharded_trainer import _atomic_write_json
from training.target_state.trainer import sha256_file


def reference_ok(row, minimum, maximum):
    return bool(reference_depth_valid(reference_detected=row["detected"],
        raw_depth_m=row["raw_depth_m"], minimum_depth_m=minimum, maximum_depth_m=maximum))


def regate_rows(rows, *, minimum, maximum, threshold=None):
    result = []
    for original in rows:
        row = dict(original)
        eligible = reference_ok(row, minimum, maximum)
        geometry = row["model_geometry_valid"] and minimum <= row["corrected_depth_m"] <= maximum
        learned = row["model_valid"] if threshold is None else row["validity_probability"] >= threshold
        row["model_valid"] = bool(eligible and geometry and learned)
        row["baseline_valid"] = bool(eligible and row["baseline_valid"])
        result.append(row)
    return result


def covariance_stats(rows, scale=1.0):
    selected = [r for r in rows if r["evaluated_visible_target"] and r["model_valid"]]
    if not selected:
        return {"count": 0, "q95_normalized_squared_error": None}
    error = np.asarray([np.subtract(r["model_position_world_m"], r["target_position_world_m"])
                        for r in selected], dtype=np.float64)
    variance = np.asarray([r["position_variance_m2"] for r in selected], dtype=np.float64)
    if not np.isfinite(variance).all() or not (variance > 0).all():
        raise ValueError("invalid covariance in accepted audit rows")
    normalized = (error ** 2 / (variance * scale)).sum(axis=1)
    return {"count": len(selected), "variance_scale": scale,
            "q95_normalized_squared_error": float(np.quantile(normalized, 0.95)),
            "coverage_68_percent_ellipsoid": float(np.mean(normalized <= chi2.ppf(0.68, 3))),
            "coverage_95_percent_ellipsoid": float(np.mean(normalized <= chi2.ppf(0.95, 3))),
            "maximum_normalized_squared_error": float(normalized.max())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    source = args.audit_dir.resolve()
    destination = args.output_dir.resolve()
    if source == destination:
        raise ValueError("write review to a separate directory, never overwrite the original audit")
    original = json.loads((source / "report.json").read_text())
    manifest = json.loads(args.model_manifest.read_text())
    if not original["complete"]:
        raise ValueError("original audit is incomplete")
    if original["contract"]["manifest_sha256"] != sha256_file(args.model_manifest):
        raise ValueError("model manifest differs from original audit")
    if sha256_file(Path(manifest["checkpoint_path"])) != manifest["checkpoint_sha256"]:
        raise ValueError("model checkpoint hash mismatch")
    minimum, maximum = manifest["config"]["minimum_depth_m"], manifest["config"]["maximum_depth_m"]
    raw = {s: json.loads((source / f"{s}_samples.json").read_text())["samples"] for s in ("validation", "test")}
    guarded = {s: regate_rows(rows, minimum=minimum, maximum=maximum) for s, rows in raw.items()}
    validation_covariance = covariance_stats(guarded["validation"])
    q95 = validation_covariance["q95_normalized_squared_error"]
    if q95 is None:
        raise ValueError("validation has no accepted samples for covariance analysis")
    candidate_scale = max(1.0, q95 / float(chi2.ppf(0.95, 3)))
    sweep = []
    for threshold in (0.05, 0.1, 0.12, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.95, 0.99):
        stats = summarize_rows(regate_rows(raw["validation"], minimum=minimum, maximum=maximum, threshold=threshold))
        sweep.append({"threshold": threshold, "failed_count": stats["model_failed_count"],
                      "false_positive_count": stats["model_false_positive_count"],
                      "accepted_error": stats["model_accepted_only"]})
    report = {
        "analysis_complete": True, "full_v2_evaluation": False,
        "measurement_protocol_under_review": MEASUREMENT_PROTOCOL,
        "scope": "saved-output reference/depth guards only; image bounds and runtime/Kalman were not replayed",
        "source_sha256": {f"{s}_samples.json": sha256_file(source / f"{s}_samples.json") for s in raw},
        "source_report_sha256": sha256_file(source / "report.json"),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "original_promotion_unchanged": manifest["promotion"],
        "results": {}, "validation_only_threshold_sweep": sweep,
        "covariance_candidate": {"fit_split": "validation", "variance_scale": candidate_scale,
                                 "applied_to_model_or_runtime": False,
                                 "assumption": "3D diagonal Gaussian; global coverage is not an outlier guarantee"},
    }
    for split, rows in guarded.items():
        blocked = [r for old, r in zip(raw[split], rows) if old["model_valid"] and not r["model_valid"]]
        outliers = sorted([r for r in rows if r["evaluated_visible_target"] and r["model_valid"]],
                          key=lambda r: r["model_error_m"], reverse=True)[:20]
        report["results"][split] = {
            "diagnostics": summarize_rows(rows), "newly_blocked": blocked,
            "covariance_original": covariance_stats(rows),
            "covariance_validation_scale_only": covariance_stats(rows, scale=candidate_scale),
            "largest_accepted_errors": outliers,
        }
    _atomic_write_json(destination / "review.json", report)
    print(json.dumps({"report": str(destination / "review.json"), "full_v2_evaluation": False,
        "blocked_counts": {s: len(v["newly_blocked"]) for s, v in report["results"].items()},
        "covariance_candidate": report["covariance_candidate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
