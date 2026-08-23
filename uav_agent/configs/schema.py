"""Immutable public schema for the unified UAV-agent configuration.

Keeping the value types separate from YAML parsing lets tools and adapters
depend on the configuration contract without importing the loader itself.
``configs.loader`` re-exports these names for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common.ids import validate_uav_id
from common.obstacle_types import ObstacleSpec
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
    obstacles: tuple[ObstacleSpec, ...] = ()


@dataclass(frozen=True)
class UavConfig:
    initial_position_xyz_m: tuple[float, float, float]
    max_speed_mps: float
    max_yaw_rate_deg_s: float
    id: str = "uav_1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_uav_id(self.id))


@dataclass(frozen=True)
class CameraConfig:
    resolution_wh_px: tuple[int, int]
    frequency_hz: int
    horizontal_fov_deg: float
    focal_length_m: float | None
    pitch_deg: float


@dataclass(frozen=True, slots=True)
class ObstaclePerceptionConfig:
    """Trusted visibility limits for the privileged ideal-camera backend."""

    mode: str = "disabled"
    max_distance_m: float = 40.0
    min_bbox_area_px: int = 64
    max_occlusion_ratio: float = 0.95


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
class ModelWorkerConfig:
    """Trusted limits for the per-UAV asynchronous model worker."""

    max_inflight_per_uav: int = 1
    request_timeout_s: float = 20.0


@dataclass(frozen=True, slots=True)
class QwenVisualReviewConfig:
    """Sampling and image limits for Qwen visual review."""

    enabled: bool = False
    mode: str = "shadow"
    goto_interval_s: float = 5.0
    search_interval_s: float = 2.0
    inspect_interval_s: float = 1.0
    track_interval_s: float = 5.0
    max_recent_frames: int = 3
    max_image_side_px: int = 1024
    jpeg_quality: int = 80
    hover_position_tolerance_m: float = 0.25
    hover_max_correction_speed_mps: float = 0.5
    blocking_hover_timeout_s: float = 75.0
    blocking_timeout_fallback: str = "CANCEL_AND_LAND"


@dataclass(frozen=True, slots=True)
class PlanRevisionConfig:
    """Hard cap and cooldown for controlled plan-suffix revisions."""

    enabled: bool = True
    max_revisions: int = 3
    cooldown_s: float = 5.0


@dataclass(frozen=True, slots=True)
class FrameStoreConfig:
    """Three independent bounds for the in-memory image store."""

    max_frames: int = 24
    max_bytes: int = 67_108_864
    max_age_s: float = 20.0


@dataclass(frozen=True, slots=True)
class DebugImagesConfig:
    """Opt-in and bounded sampled debug-image output."""

    enabled: bool = False
    max_images_per_run: int = 20


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
    # These runtime-vision blocks were added after the public v1 config.  A
    # default factory keeps construction of older trusted AppConfig fixtures
    # backward compatible while the loader validates any explicit YAML block.
    model_worker: ModelWorkerConfig = field(default_factory=ModelWorkerConfig)
    qwen_visual_review: QwenVisualReviewConfig = field(
        default_factory=QwenVisualReviewConfig
    )
    plan_revision: PlanRevisionConfig = field(default_factory=PlanRevisionConfig)
    frame_store: FrameStoreConfig = field(default_factory=FrameStoreConfig)
    debug_images: DebugImagesConfig = field(default_factory=DebugImagesConfig)
    obstacle_perception: ObstaclePerceptionConfig = field(
        default_factory=ObstaclePerceptionConfig
    )


__all__ = [
    "AppConfig",
    "CameraConfig",
    "CheckpointConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ArtifactsConfig",
    "FiguresConfig",
    "FrameStoreConfig",
    "DebugImagesConfig",
    "LoggingConfig",
    "ModelWorkerConfig",
    "ObstaclePerceptionConfig",
    "PlanRevisionConfig",
    "PlannerConfig",
    "SceneConfig",
    "SearchConfig",
    "QwenVisualReviewConfig",
    "StorageConfig",
    "SimulationConfig",
    "TargetConfig",
    "TargetMotionConfig",
    "TargetRegionConfig",
    "TensorboardConfig",
    "UavConfig",
]
