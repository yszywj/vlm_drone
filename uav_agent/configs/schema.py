"""Immutable public schema for the unified UAV-agent configuration.

Keeping the value types separate from YAML parsing lets tools and adapters
depend on the configuration contract without importing the loader itself.
``configs.loader`` re-exports these names for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from planner.policy import PlannerLimits, PlannerPolicy


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


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Trusted upper bounds for the constrained dynamic Skill planner."""

    max_plan_steps: int = 10
    max_goto_calls: int = 5
    max_search_calls: int = 1
    max_track_calls: int = 2
    max_reacquire_attempts_per_track: int = 2
    max_total_reacquire_attempts: int = 4
    min_track_duration_s: float = 1.0
    max_track_duration_s: float = 600.0
    default_on_target_lost: str = "REACQUIRE"
    default_reacquire_max_attempts: int = 2
    default_reacquire_search_radius_m: float = 10.0
    default_reacquire_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        limits = PlannerLimits.from_config(self)
        policy = PlannerPolicy.from_config(self, limits)
        object.__setattr__(
            self,
            "min_track_duration_s",
            limits.min_track_duration_s,
        )
        object.__setattr__(
            self,
            "max_track_duration_s",
            limits.max_track_duration_s,
        )
        object.__setattr__(
            self,
            "default_on_target_lost",
            policy.default_on_target_lost.value,
        )
        object.__setattr__(
            self,
            "default_reacquire_search_radius_m",
            policy.default_reacquire_search_radius_m,
        )
        object.__setattr__(
            self,
            "default_reacquire_timeout_s",
            policy.default_reacquire_timeout_s,
        )


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
    planner: PlannerConfig
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
    "PlannerConfig",
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
