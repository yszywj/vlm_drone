"""Immutable public schema for the unified UAV-agent configuration.

Keeping the value types separate from YAML parsing lets tools and adapters
depend on the configuration contract without importing the loader itself.
``configs.loader`` re-exports these names for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    headless: bool
    physics_dt_s: float
    rendering_dt_s: float
    stage_units_in_meters: float
    demo_steps: int


@dataclass(frozen=True)
class SceneConfig:
    size_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class UavConfig:
    initial_position_xyz_m: tuple[float, float, float]
    max_speed_mps: float
    max_yaw_rate_deg_s: float


@dataclass(frozen=True)
class CameraConfig:
    resolution_wh_px: tuple[int, int]
    frequency_hz: int
    horizontal_fov_deg: float
    focal_length_m: float | None
    pitch_deg: float


@dataclass(frozen=True)
class TargetRegionConfig:
    min_xyz_m: tuple[float, float, float]
    max_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class TargetMotionConfig:
    mode: str
    region: TargetRegionConfig
    speed_mps: float
    initial_heading_deg: float
    direction_change_interval_s: float
    seed: int


@dataclass(frozen=True)
class TargetConfig:
    initial_region: TargetRegionConfig
    max_speed_mps: float
    motion: TargetMotionConfig


@dataclass(frozen=True)
class SearchConfig:
    radius_m: float
    timeout_s: float
    transit_yaw_mode: str


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_root: str | None


@dataclass(frozen=True)
class LoggingConfig:
    terminal: bool
    console_log_interval_updates: int
    print_every_episode: bool
    debug_logging: bool
    csv: bool


@dataclass(frozen=True)
class TensorboardConfig:
    enabled: bool
    scalars_only: bool
    log_interval_updates: int
    flush_interval_s: float


@dataclass(frozen=True)
class CheckpointConfig:
    save_best: bool
    save_latest: bool
    latest_interval_steps: int
    save_periodic: bool
    save_full_base_model: bool
    save_adapter_only: bool
    save_optimizer_in_latest_only: bool


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool
    interval_steps: int
    num_validation_episodes: int
    num_test_episodes: int
    deterministic: bool
    fixed_validation_seeds: bool
    fixed_test_seeds: bool


@dataclass(frozen=True)
class ArtifactsConfig:
    save_images: bool
    save_videos: bool
    save_trajectories: bool
    save_observations: bool
    save_raw_frames: bool


@dataclass(frozen=True)
class FiguresConfig:
    enabled: bool
    format: str
    save_pdf: bool


@dataclass(frozen=True)
class StorageConfig:
    min_free_space_gb_before_start: float
    min_free_space_gb_during_run: float
    warning_run_size_gb: float
    max_run_size_gb: float


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    simulation: SimulationConfig
    scene: SceneConfig
    uav: UavConfig
    camera: CameraConfig
    target: TargetConfig
    search: SearchConfig
    experiment: ExperimentConfig
    logging: LoggingConfig
    tensorboard: TensorboardConfig
    checkpoint: CheckpointConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactsConfig
    figures: FiguresConfig
    storage: StorageConfig


__all__ = [
    "AppConfig",
    "CameraConfig",
    "CheckpointConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ArtifactsConfig",
    "FiguresConfig",
    "LoggingConfig",
    "SceneConfig",
    "SearchConfig",
    "StorageConfig",
    "SimulationConfig",
    "TargetConfig",
    "TargetMotionConfig",
    "TargetRegionConfig",
    "TensorboardConfig",
    "UavConfig",
]
