"""Immutable public schema for the unified UAV-agent configuration.

Keeping the value types separate from YAML parsing lets tools and adapters
depend on the configuration contract without importing the loader itself.
``configs.loader`` re-exports these names for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real


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

    def __post_init__(self) -> None:
        for name in (
            "max_plan_steps",
            "max_goto_calls",
            "max_search_calls",
            "max_track_calls",
            "max_reacquire_attempts_per_track",
            "max_total_reacquire_attempts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_plan_steps < 2:
            raise ValueError("max_plan_steps must be at least 2")
        hard_caps = {
            "max_plan_steps": 10,
            "max_goto_calls": 5,
            "max_track_calls": 2,
            "max_reacquire_attempts_per_track": 2,
            "max_total_reacquire_attempts": 4,
        }
        for name, hard_cap in hard_caps.items():
            if getattr(self, name) > hard_cap:
                raise ValueError(f"{name} must not exceed {hard_cap} in planner v1")
        if self.max_search_calls != 1:
            raise ValueError("max_search_calls must be 1 in planner v1")
        if (
            self.max_reacquire_attempts_per_track
            > self.max_total_reacquire_attempts
        ):
            raise ValueError(
                "max_reacquire_attempts_per_track must not exceed "
                "max_total_reacquire_attempts"
            )
        durations: list[float] = []
        for name in ("min_track_duration_s", "max_track_duration_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite number")
            durations.append(float(value))
            object.__setattr__(self, name, float(value))
        if durations[0] > durations[1]:
            raise ValueError(
                "min_track_duration_s must not exceed max_track_duration_s"
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
