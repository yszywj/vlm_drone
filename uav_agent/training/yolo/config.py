"""Strict configuration for isolated Ultralytics training jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


class YoloTrainingConfigError(ValueError):
    """Raised when a training configuration cannot be trusted."""


_ALLOWED_KEYS = frozenset(
    {
        "model_family",
        "task",
        "base_model_path",
        "dataset_yaml",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "workers",
        "patience",
        "seed",
        "deterministic",
        "amp",
        "cache",
        "resume",
        "project_dir",
        "run_name",
    }
)
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ENVIRONMENT_KEYS: Mapping[str, tuple[str, ...]] = {
    "base_model_path": ("UAV_AGENT_YOLO_MODEL", "UAV_AGENT_YOLO_BASE_MODEL"),
    "dataset_yaml": ("UAV_AGENT_YOLO_DATA", "UAV_AGENT_YOLO_DATASET"),
    "device": ("UAV_AGENT_YOLO_DEVICE",),
    "epochs": ("UAV_AGENT_YOLO_EPOCHS",),
    "imgsz": ("UAV_AGENT_YOLO_IMGSZ",),
    "batch": ("UAV_AGENT_YOLO_BATCH",),
    "run_name": ("UAV_AGENT_YOLO_RUN_NAME",),
    "project_dir": ("UAV_AGENT_YOLO_PROJECT_DIR",),
    "resume": ("UAV_AGENT_YOLO_RESUME",),
}


def _plain_path(value: Any, field_name: str) -> Path | None:
    if value is None or value is False:
        return None
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise YoloTrainingConfigError(f"{field_name} must be a non-empty path or null")
    return Path(value).expanduser()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise YoloTrainingConfigError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise YoloTrainingConfigError(f"{field_name} must be a non-negative integer")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise YoloTrainingConfigError(f"{field_name} must be true or false")
    return value


def _device(value: Any) -> str | int:
    if isinstance(value, bool):
        raise YoloTrainingConfigError("device must be an integer GPU index or device string")
    if isinstance(value, int):
        if value < -1:
            raise YoloTrainingConfigError("device integer must be -1 or greater")
        return value
    if not isinstance(value, str) or not value.strip():
        raise YoloTrainingConfigError("device must be an integer GPU index or device string")
    normalized = value.strip()
    if normalized.lstrip("-").isdigit():
        return int(normalized)
    if normalized.lower() not in {"cpu", "mps"} and not re.fullmatch(
        r"(?:cuda:)?\d+(?:,(?:cuda:)?\d+)*", normalized.lower()
    ):
        raise YoloTrainingConfigError(
            "device must be cpu, mps, a GPU index, or a comma-separated GPU list"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class YoloTrainConfig:
    """Complete inputs for one reproducible Ultralytics training run."""

    model_family: str = "yolo"
    task: str = "detect"
    base_model_path: Path | None = None
    dataset_yaml: Path | None = None
    epochs: int = 100
    imgsz: int = 960
    batch: int = 16
    device: str | int = 0
    workers: int = 8
    patience: int = 20
    seed: int = 42
    deterministic: bool = True
    amp: bool = True
    cache: bool = False
    resume: Path | None = None
    project_dir: Path = Path("../outputs/perception/yolo")
    run_name: str = "yolo26s_uav_target"

    def __post_init__(self) -> None:
        family = str(self.model_family).strip().lower()
        task = str(self.task).strip().lower()
        if family not in {"yolo", "yoloe"}:
            raise YoloTrainingConfigError("model_family must be 'yolo' or 'yoloe'")
        if task not in {"detect", "segment"}:
            raise YoloTrainingConfigError("task must be 'detect' or 'segment'")
        object.__setattr__(self, "model_family", family)
        object.__setattr__(self, "task", task)
        object.__setattr__(
            self,
            "base_model_path",
            _plain_path(self.base_model_path, "base_model_path"),
        )
        object.__setattr__(
            self,
            "dataset_yaml",
            _plain_path(self.dataset_yaml, "dataset_yaml"),
        )
        object.__setattr__(self, "epochs", _positive_int(self.epochs, "epochs"))
        object.__setattr__(self, "imgsz", _positive_int(self.imgsz, "imgsz"))
        object.__setattr__(self, "batch", _positive_int(self.batch, "batch"))
        object.__setattr__(self, "device", _device(self.device))
        object.__setattr__(self, "workers", _nonnegative_int(self.workers, "workers"))
        object.__setattr__(self, "patience", _nonnegative_int(self.patience, "patience"))
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))
        object.__setattr__(self, "deterministic", _bool(self.deterministic, "deterministic"))
        object.__setattr__(self, "amp", _bool(self.amp, "amp"))
        object.__setattr__(self, "cache", _bool(self.cache, "cache"))
        resume = _plain_path(self.resume, "resume")
        if resume is not None and resume.name != "last.pt":
            raise YoloTrainingConfigError("resume must point to an explicit last.pt file")
        object.__setattr__(self, "resume", resume)
        project_dir = _plain_path(self.project_dir, "project_dir")
        if project_dir is None:
            raise YoloTrainingConfigError("project_dir cannot be null")
        object.__setattr__(self, "project_dir", project_dir)
        if not isinstance(self.run_name, str) or not _RUN_NAME_RE.fullmatch(self.run_name):
            raise YoloTrainingConfigError(
                "run_name must contain only letters, digits, '.', '_' or '-'"
            )

    @property
    def run_dir(self) -> Path:
        return self.project_dir / self.run_name

    def require_runtime_paths(self) -> None:
        if self.base_model_path is None:
            raise YoloTrainingConfigError(
                "base_model_path is required; provide --model, UAV_AGENT_YOLO_MODEL, or YAML"
            )
        if self.dataset_yaml is None:
            raise YoloTrainingConfigError(
                "dataset_yaml is required; provide --data, UAV_AGENT_YOLO_DATA, or YAML"
            )
        if not self.base_model_path.is_file():
            raise YoloTrainingConfigError(
                f"base model does not exist: {self.base_model_path}; automatic downloads are disabled"
            )
        if self.base_model_path.suffix.lower() != ".pt":
            raise YoloTrainingConfigError("base_model_path must be a local .pt checkpoint")
        if not self.dataset_yaml.is_file():
            raise YoloTrainingConfigError(f"dataset YAML does not exist: {self.dataset_yaml}")
        if self.resume is not None and not self.resume.is_file():
            raise YoloTrainingConfigError(f"resume checkpoint does not exist: {self.resume}")

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        for key in ("base_model_path", "dataset_yaml", "resume", "project_dir"):
            raw[key] = None if raw[key] is None else str(raw[key])
        return raw


def _yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise YoloTrainingConfigError(f"cannot read training config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise YoloTrainingConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise YoloTrainingConfigError("training config root must be a mapping")
    unknown = sorted(set(loaded) - _ALLOWED_KEYS)
    if unknown:
        raise YoloTrainingConfigError(
            "training config contains unknown keys: " + ", ".join(str(key) for key in unknown)
        )
    return dict(loaded)


def _parse_environment_value(field_name: str, value: str) -> Any:
    if field_name in {"epochs", "imgsz", "batch"}:
        try:
            return int(value)
        except ValueError as exc:
            raise YoloTrainingConfigError(
                f"environment override for {field_name} must be an integer"
            ) from exc
    if field_name == "device":
        return value
    return value


def load_yolo_train_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> YoloTrainConfig:
    """Load config using CLI overrides > environment > YAML > defaults."""

    raw = _yaml_mapping(Path(path))
    environment = os.environ if environ is None else environ
    for field_name, variable_names in _ENVIRONMENT_KEYS.items():
        for variable_name in variable_names:
            if variable_name in environment and environment[variable_name] != "":
                raw[field_name] = _parse_environment_value(
                    field_name, environment[variable_name]
                )
                break
    if overrides:
        unknown = sorted(set(overrides) - _ALLOWED_KEYS)
        if unknown:
            raise YoloTrainingConfigError(
                "training overrides contain unknown keys: " + ", ".join(unknown)
            )
        raw.update({key: value for key, value in overrides.items() if value is not None})
    return YoloTrainConfig(**raw)

