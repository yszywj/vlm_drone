"""Validated service-only configuration for the isolated YOLO process."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any


class YoloServiceConfigurationError(ValueError):
    """Raised before a model or server socket is created."""


class ModelFamily(str, Enum):
    YOLO = "yolo"
    YOLOE = "yoloe"

    @classmethod
    def parse(cls, value: object) -> "ModelFamily":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise YoloServiceConfigurationError("model_family must be yolo or yoloe")
        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            raise YoloServiceConfigurationError(
                "model_family must be yolo or yoloe"
            ) from exc


def _positive_int(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise YoloServiceConfigurationError(f"{name} must be an integer")
    if value <= 0 or value > maximum:
        raise YoloServiceConfigurationError(
            f"{name} must be between 1 and {maximum}"
        )
    return value


def _unit_float(value: object, name: str, *, allow_one: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise YoloServiceConfigurationError(f"{name} must be a number")
    result = float(value)
    upper_ok = result <= 1.0 if allow_one else result < 1.0
    if not isfinite(result) or result <= 0.0 or not upper_ok:
        suffix = "]" if allow_one else ")"
        raise YoloServiceConfigurationError(f"{name} must be within (0, 1{suffix}")
    return result


def _device(value: object) -> str:
    if isinstance(value, bool):
        raise YoloServiceConfigurationError("device must be an index or device name")
    if isinstance(value, int):
        if value < 0:
            raise YoloServiceConfigurationError("device index must be non-negative")
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise YoloServiceConfigurationError("device must be an index or device name")
    normalized = value.strip().lower()
    if normalized not in {"cpu", "mps"} and not normalized.isdigit():
        raise YoloServiceConfigurationError("device must be cpu, mps, or a GPU index")
    return normalized


@dataclass(frozen=True, slots=True)
class YoloServiceSettings:
    """Settings safe to commit; deliberately excludes the model path."""

    schema_version: int = 1
    host: str = "127.0.0.1"
    port: int = 8011
    model_family: ModelFamily = ModelFamily.YOLO
    device: str = "0"
    tracker_path: str = "configs/yolo/botsort_uav.yaml"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.70
    image_size_px: int = 960
    max_image_bytes: int = 8_388_608
    max_image_width_px: int = 4096
    max_image_height_px: int = 4096

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise YoloServiceConfigurationError("schema_version must be 1")
        if self.host != "127.0.0.1":
            raise YoloServiceConfigurationError(
                "YOLO service host is restricted to 127.0.0.1"
            )
        object.__setattr__(self, "port", _positive_int(self.port, "port", 65_535))
        object.__setattr__(self, "model_family", ModelFamily.parse(self.model_family))
        object.__setattr__(self, "device", _device(self.device))
        if not isinstance(self.tracker_path, str) or not self.tracker_path.strip():
            raise YoloServiceConfigurationError("tracker_path must be non-empty")
        object.__setattr__(self, "tracker_path", self.tracker_path.strip())
        object.__setattr__(
            self,
            "confidence_threshold",
            _unit_float(self.confidence_threshold, "confidence_threshold"),
        )
        object.__setattr__(
            self,
            "iou_threshold",
            _unit_float(self.iou_threshold, "iou_threshold"),
        )
        object.__setattr__(
            self, "image_size_px", _positive_int(self.image_size_px, "image_size_px", 8192)
        )
        object.__setattr__(
            self,
            "max_image_bytes",
            _positive_int(self.max_image_bytes, "max_image_bytes", 268_435_456),
        )
        object.__setattr__(
            self,
            "max_image_width_px",
            _positive_int(self.max_image_width_px, "max_image_width_px", 16_384),
        )
        object.__setattr__(
            self,
            "max_image_height_px",
            _positive_int(self.max_image_height_px, "max_image_height_px", 16_384),
        )

    def with_overrides(self, **values: object) -> "YoloServiceSettings":
        supplied = {name: value for name, value in values.items() if value is not None}
        unknown = sorted(set(supplied) - set(self.__dataclass_fields__))
        if unknown:
            raise YoloServiceConfigurationError(
                "unknown service overrides: " + ", ".join(unknown)
            )
        return replace(self, **supplied)


@dataclass(frozen=True, slots=True)
class YoloServiceConfig:
    """Resolved startup config with explicit, existing local paths."""

    model_path: Path
    settings: YoloServiceSettings = YoloServiceSettings()

    def __post_init__(self) -> None:
        path = Path(self.model_path).expanduser().resolve()
        if not path.is_file():
            raise YoloServiceConfigurationError(
                f"model path is not an existing file: {path}"
            )
        tracker = Path(self.settings.tracker_path).expanduser()
        if not tracker.is_absolute():
            tracker = (Path.cwd() / tracker).resolve()
        else:
            tracker = tracker.resolve()
        if not tracker.is_file():
            raise YoloServiceConfigurationError(
                f"tracker path is not an existing file: {tracker}"
            )
        object.__setattr__(self, "model_path", path)
        object.__setattr__(
            self,
            "settings",
            replace(self.settings, tracker_path=str(tracker)),
        )


ServiceConfig = YoloServiceConfig


_SETTINGS_FIELDS = frozenset(
    {
        "schema_version",
        "host",
        "port",
        "model_family",
        "device",
        "tracker_path",
        "confidence_threshold",
        "iou_threshold",
        "image_size_px",
        "max_image_bytes",
        "max_image_width_px",
        "max_image_height_px",
    }
)


def load_service_settings(path: str | Path) -> YoloServiceSettings:
    """Load the committed service settings with exact-key validation."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - deployment dependency check
        raise YoloServiceConfigurationError("PyYAML is required to read service config") from exc
    config_path = Path(path).expanduser()
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise YoloServiceConfigurationError(
            f"service config not found: {config_path}"
        ) from exc
    except (OSError, yaml.YAMLError) as exc:
        raise YoloServiceConfigurationError(
            f"could not read service config {config_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise YoloServiceConfigurationError("service config must be a mapping")
    if any(not isinstance(key, str) for key in raw):
        raise YoloServiceConfigurationError("service config keys must be strings")
    missing = sorted(_SETTINGS_FIELDS - set(raw))
    unknown = sorted(set(raw) - _SETTINGS_FIELDS)
    if missing:
        raise YoloServiceConfigurationError(
            "service config is missing keys: " + ", ".join(missing)
        )
    if unknown:
        raise YoloServiceConfigurationError(
            "service config contains unknown keys: " + ", ".join(unknown)
        )
    return YoloServiceSettings(**raw)


def file_sha256(path: str | Path, *, chunk_size: int = 1_048_576) -> str:
    file_path = Path(path)
    digest = sha256()
    with file_path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ModelFamily",
    "ServiceConfig",
    "YoloServiceConfig",
    "YoloServiceConfigurationError",
    "YoloServiceSettings",
    "file_sha256",
    "load_service_settings",
]
