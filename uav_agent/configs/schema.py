"""Immutable public schema for the unified UAV-agent configuration.

Keeping the value types separate from YAML parsing lets tools and adapters
depend on the configuration contract without importing the loader itself.
``configs.loader`` re-exports these names for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from common.ids import validate_routing_id, validate_uav_id
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
    display_name: str | None = None
    home_name: str | None = None
    camera_profile: str = "default"

    def __post_init__(self) -> None:
        uav_id = validate_uav_id(self.id)
        display_name = uav_id if self.display_name is None else self.display_name
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError("display_name must be a non-empty string")
        home_name = f"home_{uav_id}" if self.home_name is None else self.home_name
        object.__setattr__(self, "id", uav_id)
        object.__setattr__(self, "display_name", display_name.strip())
        object.__setattr__(
            self,
            "home_name",
            validate_routing_id(home_name, "home_name"),
        )
        object.__setattr__(
            self,
            "camera_profile",
            validate_routing_id(self.camera_profile, "camera_profile"),
        )


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


@dataclass(frozen=True, slots=True)
class YoloServiceClientConfig:
    """Bounded local HTTP client settings for target perception."""

    url: str = "http://127.0.0.1:8011"
    request_timeout_s: float = 0.5
    max_result_age_s: float = 0.5
    jpeg_quality: int = 90
    max_inflight_per_uav: int = 1


@dataclass(frozen=True, slots=True)
class TargetDetectorConfig:
    """Detector family and audited target-query mapping settings."""

    model_family: str = "yolo"
    proposal_mode: str = "closed_set"
    confidence_threshold: float = 0.25
    class_aliases_path: str = "configs/yolo/class_aliases.yaml"


@dataclass(frozen=True, slots=True)
class TargetTrackerConfig:
    """Image-space tracker evidence requirements."""

    type: str = "botsort"
    min_track_observations: int = 3
    min_track_duration_s: float = 0.5


@dataclass(frozen=True, slots=True)
class TargetGeometryConfig:
    """Trusted RGB-D candidate-position resolver settings."""

    mode: str = "isaac_depth"
    depth_anchor: str = "bbox_bottom_center"
    depth_patch_radius_px: int = 4
    min_depth_m: float = 0.2
    max_depth_m: float = 200.0
    max_measurement_age_s: float = 0.5


@dataclass(frozen=True, slots=True)
class TargetStateEstimatorConfig:
    """World-space target filter limits, independent of BoT-SORT state."""

    type: str = "constant_velocity_kalman"
    max_prediction_age_s: float = 2.0
    max_position_jump_m: float = 10.0
    process_noise: float = 1.0
    measurement_noise: float = 0.5


@dataclass(frozen=True, slots=True)
class VisualConfirmationConfig:
    """Rules for turning visual candidates into a stable target lock."""

    mode: str = "class_track_or_qwen"
    require_qwen_for_attributes: bool = True
    require_qwen_for_reacquire_new_track_id: bool = True


@dataclass(frozen=True, slots=True)
class TargetPerceptionConfig:
    """Independent target-perception backend and its bounded subcomponents."""

    backend: str = "disabled"
    yolo_service: YoloServiceClientConfig = field(
        default_factory=YoloServiceClientConfig
    )
    detector: TargetDetectorConfig = field(default_factory=TargetDetectorConfig)
    tracker: TargetTrackerConfig = field(default_factory=TargetTrackerConfig)
    geometry: TargetGeometryConfig = field(default_factory=TargetGeometryConfig)
    state_estimator: TargetStateEstimatorConfig = field(
        default_factory=TargetStateEstimatorConfig
    )
    confirmation: VisualConfirmationConfig = field(
        default_factory=VisualConfirmationConfig
    )


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
class TargetAppearanceConfig:
    shape: str = "CUBE"
    color_name: str = "red"
    color_rgb: tuple[float, float, float] = (1.0, 0.12, 0.05)
    size_xyz_m: tuple[float, float, float] = (0.6, 0.6, 1.0)


@dataclass(frozen=True)
class TargetConfig:
    initial_region: TargetRegionConfig
    max_speed_mps: float
    motion: TargetMotionConfig
    id: str = "target"
    semantic_alias: str | None = None
    appearance: TargetAppearanceConfig = field(default_factory=TargetAppearanceConfig)

    def __post_init__(self) -> None:
        target_id = validate_routing_id(self.id, "target_id")
        alias = target_id if self.semantic_alias is None else self.semantic_alias
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("semantic_alias must be a non-empty string")
        object.__setattr__(self, "id", target_id)
        object.__setattr__(self, "semantic_alias", alias.strip())


@dataclass(frozen=True)
class SearchConfig:
    radius_m: float
    timeout_s: float
    transit_yaw_mode: str


@dataclass(frozen=True, slots=True)
class ModelWorkerConfig:
    """Trusted limits for the per-UAV asynchronous model worker."""

    max_inflight_per_uav: int = 1
    request_timeout_s: float = 60.0


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


@dataclass(frozen=True, slots=True)
class ResultsConfig:
    """Hard-bounded, image-free Fleet mission result policy."""

    detail_level: str = "standard"
    state_sample_hz: float = 1.0
    save_summary_figures: bool = True
    save_camera_images: bool = False
    save_videos: bool = False
    save_raw_frames: bool = False
    save_observations: bool = False
    retain_model_proposals: bool = True
    retain_prompts: bool = False
    max_record_bytes: int = 32_768
    max_stream_bytes: int = 8_388_608
    max_run_bytes: int = 33_554_432

    def __post_init__(self) -> None:
        if self.detail_level not in {"minimal", "standard"}:
            raise ValueError("results.detail_level must be minimal or standard")
        if (
            isinstance(self.state_sample_hz, bool)
            or not isinstance(self.state_sample_hz, (int, float))
            or not isfinite(float(self.state_sample_hz))
            or not 0.0 < float(self.state_sample_hz) <= 10.0
        ):
            raise ValueError("results.state_sample_hz must be within (0, 10]")
        for name in (
            "save_summary_figures",
            "save_camera_images",
            "save_videos",
            "save_raw_frames",
            "save_observations",
            "retain_model_proposals",
            "retain_prompts",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"results.{name} must be a boolean")
        if any(
            (
                self.save_camera_images,
                self.save_videos,
                self.save_raw_frames,
                self.save_observations,
                self.retain_prompts,
            )
        ):
            raise ValueError(
                "camera images, videos, raw frames, observations, and prompts "
                "must not be persisted"
            )
        for name in ("max_record_bytes", "max_stream_bytes", "max_run_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"results.{name} must be a positive integer")
        if self.max_record_bytes > 32_768:
            raise ValueError("results.max_record_bytes must not exceed 32768")
        if self.max_stream_bytes > 8_388_608:
            raise ValueError("results.max_stream_bytes must not exceed 8388608")
        if self.max_run_bytes > 33_554_432:
            raise ValueError("results.max_run_bytes must not exceed 33554432")
        if not self.max_record_bytes <= self.max_stream_bytes <= self.max_run_bytes:
            raise ValueError("results byte limits must satisfy record <= stream <= run")


@dataclass(frozen=True)
class FleetConfig:
    minimum_uav_separation_m: float = 5.0
    target_claim_policy: str = "EXCLUSIVE"
    route_conflict_policy: str = "LOWER_PRIORITY_HOLDS"
    assignment_failure_policy: str = "REPORT_AND_REPLAN"

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_uav_separation_m, bool)
            or not isinstance(self.minimum_uav_separation_m, (int, float))
            or not isfinite(float(self.minimum_uav_separation_m))
            or self.minimum_uav_separation_m <= 0.0
        ):
            raise ValueError("minimum_uav_separation_m must be greater than 0")
        if self.target_claim_policy != "EXCLUSIVE":
            raise ValueError("target_claim_policy must be EXCLUSIVE")
        if self.route_conflict_policy != "LOWER_PRIORITY_HOLDS":
            raise ValueError(
                "route_conflict_policy must be LOWER_PRIORITY_HOLDS"
            )
        if self.assignment_failure_policy != "REPORT_AND_REPLAN":
            raise ValueError(
                "assignment_failure_policy must be REPORT_AND_REPLAN"
            )


@dataclass(frozen=True)
class ModelBrokerConfig:
    max_inflight_global: int = 4
    max_inflight_per_uav: int = 1
    max_pending_per_uav: int = 2
    starvation_timeout_s: float = 15.0

    def __post_init__(self) -> None:
        for name in (
            "max_inflight_global",
            "max_inflight_per_uav",
            "max_pending_per_uav",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_inflight_per_uav != 1:
            raise ValueError("max_inflight_per_uav must be exactly 1")
        if (
            isinstance(self.starvation_timeout_s, bool)
            or not isinstance(self.starvation_timeout_s, (int, float))
            or not isfinite(float(self.starvation_timeout_s))
            or self.starvation_timeout_s <= 0.0
        ):
            raise ValueError("starvation_timeout_s must be greater than 0")


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    simulation: SimulationConfig
    scene: SceneConfig
    # Legacy singleton constructor inputs are retained for compatibility with
    # existing ``dataclasses.replace(config, uav=...)`` call sites.  The
    # canonical public inventory is always the plural fields below.
    uav: UavConfig | None = field(
        repr=False,
        compare=False,
        metadata={"run_manager_exclude": True},
    )
    camera: CameraConfig | None = field(
        repr=False,
        compare=False,
        metadata={"run_manager_exclude": True},
    )
    target: TargetConfig | None = field(
        repr=False,
        compare=False,
        metadata={"run_manager_exclude": True},
    )
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
    results: ResultsConfig = field(default_factory=ResultsConfig)
    qwen_visual_review: QwenVisualReviewConfig = field(
        default_factory=QwenVisualReviewConfig
    )
    plan_revision: PlanRevisionConfig = field(default_factory=PlanRevisionConfig)
    frame_store: FrameStoreConfig = field(default_factory=FrameStoreConfig)
    debug_images: DebugImagesConfig = field(default_factory=DebugImagesConfig)
    obstacle_perception: ObstaclePerceptionConfig = field(
        default_factory=ObstaclePerceptionConfig
    )
    target_perception: TargetPerceptionConfig = field(
        default_factory=TargetPerceptionConfig
    )
    uavs: tuple[UavConfig, ...] = ()
    targets: tuple[TargetConfig, ...] = ()
    camera_profiles: Mapping[str, CameraConfig] = field(default_factory=dict)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    model_broker: ModelBrokerConfig = field(default_factory=ModelBrokerConfig)

    def __post_init__(self) -> None:
        raw_uav = object.__getattribute__(self, "uav")
        raw_camera = object.__getattribute__(self, "camera")
        raw_target = object.__getattribute__(self, "target")
        uavs = tuple(self.uavs)
        targets = tuple(self.targets)
        profiles = dict(self.camera_profiles)

        # A legacy value wins for singleton construction/replacement.  Multi
        # inventory construction passes these three compatibility inputs as
        # ``None`` and therefore cannot be silently collapsed to one item.
        if raw_uav is not None:
            if len(uavs) > 1:
                raise ValueError("legacy uav cannot be combined with multiple uavs")
            uavs = (raw_uav,)
        if raw_target is not None:
            if len(targets) > 1:
                raise ValueError("legacy target cannot be combined with multiple targets")
            targets = (raw_target,)
        if raw_camera is not None:
            if len(uavs) != 1:
                raise ValueError("legacy camera requires exactly one UAV")
            profiles[uavs[0].camera_profile] = raw_camera

        if not uavs:
            raise ValueError("uavs must contain at least one UAV")
        if not targets:
            raise ValueError("targets must contain at least one target")
        if not profiles:
            raise ValueError("camera_profiles must contain at least one profile")
        if len({item.id for item in uavs}) != len(uavs):
            raise ValueError("uavs must have unique IDs")
        if len({item.id for item in targets}) != len(targets):
            raise ValueError("targets must have unique IDs")
        for name in profiles:
            validate_routing_id(name, "camera_profile")
        for item in uavs:
            if item.camera_profile not in profiles:
                raise ValueError(
                    f"UAV {item.id!r} references unknown camera profile "
                    f"{item.camera_profile!r}"
                )

        object.__setattr__(self, "uavs", uavs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "camera_profiles", MappingProxyType(profiles))

    def __getattribute__(self, name: str) -> object:
        if name == "uav":
            uavs = object.__getattribute__(self, "uavs")
            if len(uavs) != 1:
                raise ValueError(
                    "config.uav is ambiguous for a multi-UAV config; use config.uavs"
                )
            return uavs[0]
        if name == "target":
            targets = object.__getattribute__(self, "targets")
            if len(targets) != 1:
                raise ValueError(
                    "config.target is ambiguous for a multi-target config; use config.targets"
                )
            return targets[0]
        if name == "camera":
            uavs = object.__getattribute__(self, "uavs")
            if len(uavs) != 1:
                raise ValueError(
                    "config.camera is ambiguous for a multi-UAV config; "
                    "use config.camera_profiles"
                )
            profiles = object.__getattribute__(self, "camera_profiles")
            return profiles[uavs[0].camera_profile]
        return object.__getattribute__(self, name)


__all__ = [
    "AppConfig",
    "CameraConfig",
    "CheckpointConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ArtifactsConfig",
    "FiguresConfig",
    "FleetConfig",
    "FrameStoreConfig",
    "DebugImagesConfig",
    "LoggingConfig",
    "ModelWorkerConfig",
    "ObstaclePerceptionConfig",
    "TargetDetectorConfig",
    "TargetGeometryConfig",
    "TargetPerceptionConfig",
    "TargetStateEstimatorConfig",
    "TargetTrackerConfig",
    "VisualConfirmationConfig",
    "YoloServiceClientConfig",
    "PlanRevisionConfig",
    "PlannerConfig",
    "SceneConfig",
    "SearchConfig",
    "QwenVisualReviewConfig",
    "ResultsConfig",
    "StorageConfig",
    "SimulationConfig",
    "TargetConfig",
    "TargetAppearanceConfig",
    "TargetMotionConfig",
    "TargetRegionConfig",
    "TensorboardConfig",
    "UavConfig",
]
