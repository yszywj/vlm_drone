"""Model hashes and JSON manifests for reproducible YOLO artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from training.yolo.config import YoloTrainConfig
from training.yolo.dataset import DatasetValidationReport


class ModelRegistryError(ValueError):
    """Raised when model provenance or a validation gate is invalid."""


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise ModelRegistryError(f"cannot hash missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dataset_manifest_hash(report: DatasetValidationReport) -> tuple[str, str]:
    """Hash the dataset manifest, accepting the collector's JSONL manifest."""

    for name in ("manifest.json", "manifest.jsonl"):
        manifest = report.dataset_root / name
        if manifest.is_file():
            return str(manifest), sha256_file(manifest)
    # Third-party Ultralytics datasets often have no manifest. Keep provenance
    # reproducible and label the exact fallback path instead of returning no hash.
    return str(report.data_yaml), sha256_file(report.data_yaml)


def build_model_manifest(
    *,
    config: YoloTrainConfig,
    dataset_report: DatasetValidationReport,
    best_model_path: str | Path,
    started_at: str,
    finished_at: str,
    elapsed_s: float,
    metrics: Mapping[str, Any],
    ultralytics_version: str,
    torch_version: str,
    git_commit: str | None = None,
) -> dict[str, Any]:
    if config.base_model_path is None:
        raise ModelRegistryError("base model path is required for a model manifest")
    best = Path(best_model_path)
    manifest_path, manifest_hash = dataset_manifest_hash(dataset_report)
    commit = git_commit if git_commit is not None else os.getenv("GIT_COMMIT")
    return {
        "schema_version": 1,
        "training": {
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_s": float(elapsed_s),
        },
        "git_commit": commit,
        "model_family": config.model_family,
        "task": config.task,
        "base_model": {
            "path": str(config.base_model_path),
            "sha256": sha256_file(config.base_model_path),
        },
        "best_model": {
            "path": str(best),
            "sha256": sha256_file(best),
        },
        "dataset": {
            "name": dataset_report.dataset_root.name,
            "data_yaml": str(dataset_report.data_yaml),
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_hash,
            "classes": list(dataset_report.class_names),
            "split_image_counts": dict(dataset_report.split_image_counts),
            "class_counts": dict(dataset_report.class_counts),
        },
        "training_parameters": config.to_dict(),
        "versions": {
            "ultralytics": ultralytics_version,
            "torch": torch_version,
        },
        "validation_metrics": _json_safe(metrics),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def write_model_manifest(run_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    return write_json(Path(run_dir) / "model_manifest.json", payload)


def write_validation_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return write_json(path, payload)


def load_validation_gate(
    validation_report: str | Path,
    *,
    model_path: str | Path,
    model_family: str | None = None,
) -> Mapping[str, Any]:
    report_path = Path(validation_report)
    if not report_path.is_file():
        raise ModelRegistryError(
            "export requires an explicit validation report produced by validate_yolo.py"
        )
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(f"invalid validation report {report_path}: {exc}") from exc
    if not isinstance(raw, Mapping) or raw.get("passed") is not True:
        raise ModelRegistryError("model validation did not pass; export is blocked")
    required_fields = {
        "schema_version",
        "validated_at",
        "model_path",
        "model_sha256",
        "dataset_yaml",
        "model_family",
        "task",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "per_class",
        "small_target_metrics",
        "latency_ms",
    }
    missing = sorted(required_fields - set(raw))
    if missing:
        raise ModelRegistryError(
            "validation report is incomplete; missing: " + ", ".join(missing)
        )
    if raw.get("schema_version") != 1:
        raise ModelRegistryError("unsupported validation report schema_version")
    if not isinstance(raw.get("validated_at"), str) or not raw["validated_at"].strip():
        raise ModelRegistryError("validation report has no valid validation timestamp")
    for key in ("model_path", "dataset_yaml"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ModelRegistryError(f"validation report has invalid {key}")
    report_family = raw.get("model_family")
    if report_family not in {"yolo", "yoloe"}:
        raise ModelRegistryError("validation report has invalid model_family")
    if model_family is not None and report_family != model_family:
        raise ModelRegistryError(
            "validation report model_family does not match the requested export family"
        )
    if raw.get("task") not in {"detect", "segment"}:
        raise ModelRegistryError("validation report has invalid task")
    for key in ("mAP50", "mAP50-95", "precision", "recall"):
        value = raw.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ModelRegistryError(f"validation report has invalid {key}")
    for key in ("per_class", "small_target_metrics", "latency_ms"):
        if not isinstance(raw.get(key), Mapping):
            raise ModelRegistryError(f"validation report has invalid {key}")
    small_metrics = raw["small_target_metrics"]
    if small_metrics.get("available") is not True:
        raise ModelRegistryError(
            "validation report has no complete small-target metrics; export is blocked"
        )
    for key in ("precision", "recall", "mAP50", "mAP50-95"):
        value = small_metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ModelRegistryError(
                f"validation report has invalid small-target {key}"
            )
    expected_hash = raw.get("model_sha256")
    actual_hash = sha256_file(model_path)
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ModelRegistryError(
            "validation report model hash does not match the requested export model"
        )
    return raw
