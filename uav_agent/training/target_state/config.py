"""Validated configuration for two-stage temporal target-state training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Mapping

import yaml


class TrainingStage(str, Enum):
    ORACLE_CLEAN = "oracle_clean"
    YOLO_DEPLOYMENT = "yolo_deployment"


@dataclass(frozen=True, slots=True)
class LossWeights:
    depth: float = 1.0
    position_3d: float = 2.0
    reprojection: float = 0.5
    gaussian_nll: float = 0.25
    validity_bce: float = 0.5

    def __post_init__(self) -> None:
        for name in ("depth", "position_3d", "reprojection", "gaussian_nll", "validity_bce"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)) or value < 0:
                raise ValueError(f"loss_weights.{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TargetStateTrainingConfig:
    dataset_root: Path
    output_dir: Path
    initial_checkpoint_path: Path | None = None
    stage: TrainingStage = TrainingStage.YOLO_DEPLOYMENT
    history_size: int = 6
    max_history_age_s: float = 2.0
    roi_size_px: int = 128
    geometry_input_dim: int = 25
    roi_feature_dim: int = 96
    geometry_feature_dim: int = 64
    hidden_dim: int = 128
    gru_layers: int = 2
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda:0"
    run_name: str = "temporal_ray_depth_v1"
    camera_convention: str = "camera_optical_x_right_y_down_z_forward"
    coordinate_convention: str = "world_flu_x_forward_y_left_z_up"
    save_figures: int = 3
    validation_interval: int = 1
    promotion_p95_max_ratio: float = 1.05
    promotion_min_covariance_correlation: float = 0.1
    maximum_depth_m: float = 200.0
    minimum_depth_m: float = 0.2
    require_dataset_manifest: bool = True
    expected_yolo_model_sha256: str | None = None
    loss_weights: LossWeights = LossWeights()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser().resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        if self.initial_checkpoint_path is not None:
            object.__setattr__(
                self,
                "initial_checkpoint_path",
                Path(self.initial_checkpoint_path).expanduser().resolve(),
            )
        if not isinstance(self.stage, TrainingStage):
            object.__setattr__(self, "stage", TrainingStage(self.stage))
        if not 4 <= self.history_size <= 8:
            raise ValueError("history_size must be within [4, 8]")
        if not isfinite(self.max_history_age_s) or self.max_history_age_s <= 0.0:
            raise ValueError("max_history_age_s must be finite and positive")
        if self.roi_size_px < 32 or self.roi_size_px > 512:
            raise ValueError("roi_size_px must be within [32, 512]")
        if self.geometry_input_dim != 25:
            raise ValueError("geometry_input_dim must be 25 for schema version 1")
        for name in ("geometry_input_dim", "roi_feature_dim", "geometry_feature_dim", "hidden_dim", "gru_layers", "epochs", "batch_size"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not isinstance(self.require_dataset_manifest, bool):
            raise TypeError("require_dataset_manifest must be bool")
        if self.expected_yolo_model_sha256 is not None:
            if not isinstance(self.expected_yolo_model_sha256, str):
                raise TypeError("expected_yolo_model_sha256 must be a string or null")
            digest = self.expected_yolo_model_sha256.casefold()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "expected_yolo_model_sha256 must be a 64-character hexadecimal digest"
                )
            object.__setattr__(self, "expected_yolo_model_sha256", digest)
        if (
            self.stage is TrainingStage.YOLO_DEPLOYMENT
            and self.require_dataset_manifest
            and self.expected_yolo_model_sha256 is None
        ):
            raise ValueError(
                "yolo_deployment with a required dataset manifest must declare "
                "expected_yolo_model_sha256"
            )
        if self.save_figures < 0 or self.save_figures > 10:
            raise ValueError("save_figures must be within [0, 10]")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0 or (name == "learning_rate" and value == 0.0):
                raise ValueError(f"{name} must be finite and {'positive' if name == 'learning_rate' else 'non-negative'}")
        if not self.run_name or self.run_name != self.run_name.strip() or "/" in self.run_name:
            raise ValueError("run_name must be a non-empty path-safe name")
        if self.camera_convention != "camera_optical_x_right_y_down_z_forward":
            raise ValueError("unsupported camera_convention")
        if self.coordinate_convention != "world_flu_x_forward_y_left_z_up":
            raise ValueError("unsupported coordinate_convention")
        if not isfinite(self.promotion_p95_max_ratio) or self.promotion_p95_max_ratio < 1.0:
            raise ValueError("promotion_p95_max_ratio must be finite and at least 1")
        if not -1.0 <= self.promotion_min_covariance_correlation <= 1.0:
            raise ValueError("promotion_min_covariance_correlation must be within [-1, 1]")
        if not isfinite(self.maximum_depth_m) or self.maximum_depth_m <= 0.0:
            raise ValueError("maximum_depth_m must be finite and positive")
        if not isfinite(self.minimum_depth_m) or self.minimum_depth_m <= 0.0:
            raise ValueError("minimum_depth_m must be finite and positive")
        if self.minimum_depth_m >= self.maximum_depth_m:
            raise ValueError("minimum_depth_m must be below maximum_depth_m")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def load_training_config(path: str | Path) -> TargetStateTrainingConfig:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, "config")
    unknown = set(root) - {
        "dataset_root", "output_dir", "initial_checkpoint_path", "stage", "history_size", "max_history_age_s",
        "roi_size_px", "geometry_input_dim", "roi_feature_dim", "geometry_feature_dim",
        "hidden_dim", "gru_layers", "epochs", "batch_size", "learning_rate",
        "weight_decay", "num_workers", "seed", "device", "run_name",
        "camera_convention", "coordinate_convention", "loss_weights",
        "save_figures", "validation_interval", "promotion_p95_max_ratio",
        "promotion_min_covariance_correlation", "maximum_depth_m",
        "require_dataset_manifest",
        "expected_yolo_model_sha256",
        "minimum_depth_m",
    }
    if unknown:
        raise ValueError(f"unknown target-state training fields: {sorted(unknown)}")
    values = dict(root)
    if "loss_weights" in values:
        weights = _mapping(values["loss_weights"], "loss_weights")
        unknown_weights = set(weights) - {
            "depth", "position_3d", "reprojection", "gaussian_nll", "validity_bce"
        }
        if unknown_weights:
            raise ValueError(f"unknown loss weight fields: {sorted(unknown_weights)}")
        values["loss_weights"] = LossWeights(**weights)
    if "stage" in values:
        values["stage"] = TrainingStage(values["stage"])
    return TargetStateTrainingConfig(**values)


__all__ = ["LossWeights", "TargetStateTrainingConfig", "TrainingStage", "load_training_config"]
