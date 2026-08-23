"""Load and validate the unified UAV-agent YAML configuration."""

from __future__ import annotations

from dataclasses import replace
from math import isclose, isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from common.ids import validate_uav_id
from configs.schema import (
    AppConfig,
    ArtifactsConfig,
    CameraConfig,
    CheckpointConfig,
    DebugImagesConfig,
    EvaluationConfig,
    ExperimentConfig,
    FiguresConfig,
    FrameStoreConfig,
    LoggingConfig,
    ModelWorkerConfig,
    ObstaclePerceptionConfig,
    PlanRevisionConfig,
    PlannerConfig,
    SceneConfig,
    SearchConfig,
    QwenVisualReviewConfig,
    StorageConfig,
    SimulationConfig,
    TargetConfig,
    TargetMotionConfig,
    TargetRegionConfig,
    TensorboardConfig,
    UavConfig,
)
from common.obstacle_types import ObstacleMotionState, ObstacleSpec


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_DEFAULT_PLANNER_CONFIG = PlannerConfig()
_DEFAULT_MODEL_WORKER_CONFIG = ModelWorkerConfig()
_DEFAULT_QWEN_VISUAL_REVIEW_CONFIG = QwenVisualReviewConfig()
_DEFAULT_PLAN_REVISION_CONFIG = PlanRevisionConfig()
_DEFAULT_FRAME_STORE_CONFIG = FrameStoreConfig()
_DEFAULT_DEBUG_IMAGES_CONFIG = DebugImagesConfig()
_DEFAULT_OBSTACLE_PERCEPTION_CONFIG = ObstaclePerceptionConfig()

_MAX_REQUEST_TIMEOUT_S = 300.0
_MAX_REVIEW_INTERVAL_S = 3_600.0
_MAX_IMAGE_SIDE_PX = 4_096
_MAX_HOVER_POSITION_TOLERANCE_M = 5.0
_MAX_HOVER_CORRECTION_SPEED_MPS = 10.0
# Keep the configuration boundary aligned with planner.revision.RevisionLimits;
# a value accepted here must remain constructible by the trusted runtime.
_MAX_PLAN_REVISIONS = 3
_MAX_REVISION_COOLDOWN_S = 3_600.0
_MAX_FRAME_STORE_FRAMES = 4_096
_MAX_FRAME_STORE_BYTES = 1_073_741_824
_MAX_FRAME_STORE_AGE_S = 3_600.0
_MAX_DEBUG_IMAGES_PER_RUN = 10_000
_MAX_OBSTACLE_PERCEPTION_DISTANCE_M = 10_000.0
_MAX_OBSTACLE_BBOX_AREA_PX = 100_000_000


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key: {path}.{key}")
    return mapping[key]


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ConfigError(f"{path} must be finite")
    return result


def _positive_number(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if result <= 0.0:
        raise ConfigError(f"{path} must be greater than 0")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false")
    return value


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{path} must be a non-negative integer")
    return value


def _bounded_positive_number(value: Any, path: str, maximum: float) -> float:
    result = _positive_number(value, path)
    if result > maximum:
        raise ConfigError(f"{path} must not exceed {maximum:g}")
    return result


def _bounded_positive_integer(value: Any, path: str, maximum: int) -> int:
    result = _positive_integer(value, path)
    if result > maximum:
        raise ConfigError(f"{path} must not exceed {maximum}")
    return result


def _strict_optional_block(
    root: Mapping[str, Any],
    name: str,
    expected_keys: frozenset[str],
) -> Mapping[str, Any] | None:
    """Return a supplied post-v1 block after exact-key validation.

    Omitting the entire block is backward compatible.  Supplying a partial or
    extended block is rejected so a typo cannot silently weaken a limit.
    """

    if name not in root:
        return None
    raw = _mapping(root[name], name)
    if any(not isinstance(key, str) for key in raw):
        raise ConfigError(f"{name} keys must be strings")
    missing = sorted(expected_keys - set(raw))
    unknown = sorted(set(raw) - expected_keys)
    if missing:
        raise ConfigError(f"{name} is missing required keys: " + ", ".join(missing))
    if unknown:
        raise ConfigError(f"{name} contains unknown keys: " + ", ".join(unknown))
    return raw


def _uav_id(value: Any, path: str) -> str:
    try:
        return validate_uav_id(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _nonnegative_number(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if result < 0.0:
        raise ConfigError(f"{path} must be greater than or equal to 0")
    return result


def _float_vector(value: Any, path: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ConfigError(f"{path} must contain exactly {length} numbers")
    return tuple(_finite_number(item, f"{path}[{index}]") for index, item in enumerate(value))


def _resolution(value: Any, path: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ConfigError(f"{path} must be [width, height]")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ConfigError(f"{path} values must be positive integers")
    return int(value[0]), int(value[1])


def _validate_spatial_bounds(config: AppConfig) -> None:
    size_x, size_y, size_z = config.scene.size_xyz_m
    uav_x, uav_y, uav_z = config.uav.initial_position_xyz_m
    if not (-size_x / 2.0 <= uav_x <= size_x / 2.0):
        raise ConfigError("uav.initial_position_xyz_m x is outside the scene")
    if not (-size_y / 2.0 <= uav_y <= size_y / 2.0):
        raise ConfigError("uav.initial_position_xyz_m y is outside the scene")
    if not (0.0 <= uav_z <= size_z):
        raise ConfigError("uav.initial_position_xyz_m z is outside the scene")

    region_min = config.target.initial_region.min_xyz_m
    region_max = config.target.initial_region.max_xyz_m
    if any(low > high for low, high in zip(region_min, region_max)):
        raise ConfigError("target.initial_region min_xyz_m must not exceed max_xyz_m")
    if region_min[0] < -size_x / 2.0 or region_max[0] > size_x / 2.0:
        raise ConfigError("target.initial_region x bounds are outside the scene")
    if region_min[1] < -size_y / 2.0 or region_max[1] > size_y / 2.0:
        raise ConfigError("target.initial_region y bounds are outside the scene")
    if region_min[2] < 0.0 or region_max[2] > size_z:
        raise ConfigError("target.initial_region z bounds are outside the scene")

    motion_min = config.target.motion.region.min_xyz_m
    motion_max = config.target.motion.region.max_xyz_m
    if any(low > high for low, high in zip(motion_min, motion_max)):
        raise ConfigError("target.motion.region min_xyz_m must not exceed max_xyz_m")
    if motion_min[0] < -size_x / 2.0 or motion_max[0] > size_x / 2.0:
        raise ConfigError("target.motion.region x bounds are outside the scene")
    if motion_min[1] < -size_y / 2.0 or motion_max[1] > size_y / 2.0:
        raise ConfigError("target.motion.region y bounds are outside the scene")
    if motion_min[2] < 0.0 or motion_max[2] > size_z:
        raise ConfigError("target.motion.region z bounds are outside the scene")
    if motion_min[0] == motion_max[0] or motion_min[1] == motion_max[1]:
        raise ConfigError("target.motion.region must have positive x and y width")
    initial_inside_motion = all(
        initial_low >= motion_low and initial_high <= motion_high
        for initial_low, initial_high, motion_low, motion_high in zip(
            region_min,
            region_max,
            motion_min,
            motion_max,
        )
    )
    if not initial_inside_motion:
        raise ConfigError("target.initial_region must be contained in target.motion.region")

    for spec in config.scene.obstacles:
        minimum = spec.aabb.min_xyz_m
        maximum = spec.aabb.max_xyz_m
        if minimum[0] < -size_x / 2.0 or maximum[0] > size_x / 2.0:
            raise ConfigError(f"scene obstacle {spec.obstacle_id!r} x bounds are outside the scene")
        if minimum[1] < -size_y / 2.0 or maximum[1] > size_y / 2.0:
            raise ConfigError(f"scene obstacle {spec.obstacle_id!r} y bounds are outside the scene")
        if minimum[2] < 0.0 or maximum[2] > size_z:
            raise ConfigError(f"scene obstacle {spec.obstacle_id!r} z bounds are outside the scene")


def load_config(path: str | Path) -> AppConfig:
    """Read ``path`` once and return an immutable, validated configuration."""

    config_path = Path(path).expanduser()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    root = _mapping(raw, "config")
    schema_version = _required(root, "schema_version", "config")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        raise ConfigError("schema_version must be 1")

    simulation_raw = _mapping(_required(root, "simulation", "config"), "simulation")
    headless = _required(simulation_raw, "headless", "simulation")
    if not isinstance(headless, bool):
        raise ConfigError("simulation.headless must be true or false")
    demo_steps = _required(simulation_raw, "demo_steps", "simulation")
    if isinstance(demo_steps, bool) or not isinstance(demo_steps, int) or demo_steps <= 0:
        raise ConfigError("simulation.demo_steps must be a positive integer")
    simulation = SimulationConfig(
        headless=headless,
        physics_dt_s=_positive_number(
            _required(simulation_raw, "physics_dt_s", "simulation"), "simulation.physics_dt_s"
        ),
        rendering_dt_s=_positive_number(
            _required(simulation_raw, "rendering_dt_s", "simulation"), "simulation.rendering_dt_s"
        ),
        stage_units_in_meters=_positive_number(
            _required(simulation_raw, "stage_units_in_meters", "simulation"),
            "simulation.stage_units_in_meters",
        ),
        demo_steps=demo_steps,
    )
    if simulation.stage_units_in_meters != 1.0:
        raise ConfigError(
            "simulation.stage_units_in_meters must be 1.0 because all *_m values are passed as stage units"
        )
    dt_ratio = simulation.rendering_dt_s / simulation.physics_dt_s
    if not isclose(dt_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ConfigError(
            "simulation.physics_dt_s and simulation.rendering_dt_s must match in this kinematic environment"
        )
    requested_rendering_hz = 1.0 / simulation.rendering_dt_s
    rendering_hz = round(requested_rendering_hz)
    if rendering_hz <= 0 or not isclose(
        requested_rendering_hz,
        rendering_hz,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ConfigError("1 / simulation.rendering_dt_s must resolve to an integer frequency")
    canonical_dt_s = 1.0 / rendering_hz
    simulation = replace(
        simulation,
        physics_dt_s=canonical_dt_s,
        rendering_dt_s=canonical_dt_s,
    )

    scene_raw = _mapping(_required(root, "scene", "config"), "scene")
    obstacles_raw = scene_raw.get("obstacles", ())
    if isinstance(obstacles_raw, (str, bytes)) or not isinstance(obstacles_raw, Sequence):
        raise ConfigError("scene.obstacles must be a sequence")
    obstacle_specs: list[ObstacleSpec] = []
    obstacle_keys = frozenset(
        {
            "obstacle_id",
            "center_xyz_m",
            "size_xyz_m",
            "color_rgb",
            "collidable",
            "motion_state",
        }
    )
    for index, raw_obstacle in enumerate(obstacles_raw):
        path = f"scene.obstacles[{index}]"
        obstacle = _mapping(raw_obstacle, path)
        missing = sorted(obstacle_keys - set(obstacle))
        unknown = sorted(set(obstacle) - obstacle_keys)
        if missing:
            raise ConfigError(f"{path} is missing required keys: " + ", ".join(missing))
        if unknown:
            raise ConfigError(f"{path} contains unknown keys: " + ", ".join(unknown))
        obstacle_id = obstacle["obstacle_id"]
        if not isinstance(obstacle_id, str):
            raise ConfigError(f"{path}.obstacle_id must be a string")
        motion_state = obstacle["motion_state"]
        if not isinstance(motion_state, str):
            raise ConfigError(f"{path}.motion_state must be a string")
        try:
            obstacle_specs.append(
                ObstacleSpec(
                    obstacle_id=obstacle_id,
                    center_xyz_m=_float_vector(
                        obstacle["center_xyz_m"], f"{path}.center_xyz_m", 3
                    ),
                    size_xyz_m=_float_vector(
                        obstacle["size_xyz_m"], f"{path}.size_xyz_m", 3
                    ),
                    color_rgb=_float_vector(
                        obstacle["color_rgb"], f"{path}.color_rgb", 3
                    ),
                    collidable=_boolean(
                        obstacle["collidable"], f"{path}.collidable"
                    ),
                    motion_state=ObstacleMotionState(motion_state),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid {path}: {exc}") from exc
    obstacle_ids = tuple(spec.obstacle_id for spec in obstacle_specs)
    if len(obstacle_ids) != len(set(obstacle_ids)):
        raise ConfigError("invalid scene.obstacles: duplicate obstacle_id")
    scene = SceneConfig(
        size_xyz_m=_float_vector(_required(scene_raw, "size_xyz_m", "scene"), "scene.size_xyz_m", 3),
        obstacles=tuple(obstacle_specs),
    )
    if any(value <= 0.0 for value in scene.size_xyz_m):
        raise ConfigError("scene.size_xyz_m values must be greater than 0")

    uav_raw = _mapping(_required(root, "uav", "config"), "uav")
    uav = UavConfig(
        id=_uav_id(uav_raw.get("id", "uav_1"), "uav.id"),
        initial_position_xyz_m=_float_vector(
            _required(uav_raw, "initial_position_xyz_m", "uav"), "uav.initial_position_xyz_m", 3
        ),
        max_speed_mps=_positive_number(_required(uav_raw, "max_speed_mps", "uav"), "uav.max_speed_mps"),
        max_yaw_rate_deg_s=_positive_number(
            _required(uav_raw, "max_yaw_rate_deg_s", "uav"), "uav.max_yaw_rate_deg_s"
        ),
    )

    camera_raw = _mapping(_required(root, "camera", "config"), "camera")
    frequency_hz = _required(camera_raw, "frequency_hz", "camera")
    if isinstance(frequency_hz, bool) or not isinstance(frequency_hz, int) or frequency_hz <= 0:
        raise ConfigError("camera.frequency_hz must be a positive integer")
    focal_length_raw = _required(camera_raw, "focal_length_m", "camera")
    camera = CameraConfig(
        resolution_wh_px=_resolution(
            _required(camera_raw, "resolution_wh_px", "camera"), "camera.resolution_wh_px"
        ),
        frequency_hz=frequency_hz,
        horizontal_fov_deg=_finite_number(
            _required(camera_raw, "horizontal_fov_deg", "camera"), "camera.horizontal_fov_deg"
        ),
        focal_length_m=(
            None
            if focal_length_raw is None
            else _positive_number(focal_length_raw, "camera.focal_length_m")
        ),
        pitch_deg=_finite_number(_required(camera_raw, "pitch_deg", "camera"), "camera.pitch_deg"),
    )
    if not 0.0 < camera.horizontal_fov_deg < 180.0:
        raise ConfigError("camera.horizontal_fov_deg must be between 0 and 180")
    if not -90.0 <= camera.pitch_deg <= 90.0:
        raise ConfigError("camera.pitch_deg must be between -90 and 90")
    if rendering_hz % camera.frequency_hz != 0:
        raise ConfigError("camera.frequency_hz must be an integer divisor of the rendering frequency")

    obstacle_perception_raw = _strict_optional_block(
        root,
        "obstacle_perception",
        frozenset(
            {
                "mode",
                "max_distance_m",
                "min_bbox_area_px",
                "max_occlusion_ratio",
            }
        ),
    )
    if obstacle_perception_raw is None:
        obstacle_perception = _DEFAULT_OBSTACLE_PERCEPTION_CONFIG
    else:
        mode = obstacle_perception_raw["mode"]
        if not isinstance(mode, str) or mode not in {"disabled", "ideal_camera"}:
            raise ConfigError(
                "obstacle_perception.mode must be disabled or ideal_camera"
            )
        max_occlusion = _finite_number(
            obstacle_perception_raw["max_occlusion_ratio"],
            "obstacle_perception.max_occlusion_ratio",
        )
        if not 0.0 <= max_occlusion < 1.0:
            raise ConfigError(
                "obstacle_perception.max_occlusion_ratio must be in [0, 1)"
            )
        obstacle_perception = ObstaclePerceptionConfig(
            mode=mode,
            max_distance_m=_bounded_positive_number(
                obstacle_perception_raw["max_distance_m"],
                "obstacle_perception.max_distance_m",
                _MAX_OBSTACLE_PERCEPTION_DISTANCE_M,
            ),
            min_bbox_area_px=_bounded_positive_integer(
                obstacle_perception_raw["min_bbox_area_px"],
                "obstacle_perception.min_bbox_area_px",
                _MAX_OBSTACLE_BBOX_AREA_PX,
            ),
            max_occlusion_ratio=max_occlusion,
        )

    target_raw = _mapping(_required(root, "target", "config"), "target")
    region_raw = _mapping(_required(target_raw, "initial_region", "target"), "target.initial_region")
    initial_region = TargetRegionConfig(
        min_xyz_m=_float_vector(
            _required(region_raw, "min_xyz_m", "target.initial_region"),
            "target.initial_region.min_xyz_m",
            3,
        ),
        max_xyz_m=_float_vector(
            _required(region_raw, "max_xyz_m", "target.initial_region"),
            "target.initial_region.max_xyz_m",
            3,
        ),
    )
    max_target_speed = _positive_number(
        _required(target_raw, "max_speed_mps", "target"), "target.max_speed_mps"
    )
    motion_raw = _mapping(_required(target_raw, "motion", "target"), "target.motion")
    motion_mode = _required(motion_raw, "mode", "target.motion")
    if not isinstance(motion_mode, str) or motion_mode.upper() not in {"STATIC", "LINEAR", "RANDOM_WALK"}:
        raise ConfigError("target.motion.mode must be STATIC, LINEAR, or RANDOM_WALK")
    motion_region_raw = _mapping(
        _required(motion_raw, "region", "target.motion"), "target.motion.region"
    )
    motion_region = TargetRegionConfig(
        min_xyz_m=_float_vector(
            _required(motion_region_raw, "min_xyz_m", "target.motion.region"),
            "target.motion.region.min_xyz_m",
            3,
        ),
        max_xyz_m=_float_vector(
            _required(motion_region_raw, "max_xyz_m", "target.motion.region"),
            "target.motion.region.max_xyz_m",
            3,
        ),
    )
    motion_speed = _nonnegative_number(
        _required(motion_raw, "speed_mps", "target.motion"), "target.motion.speed_mps"
    )
    if motion_speed > max_target_speed:
        raise ConfigError("target.motion.speed_mps must not exceed target.max_speed_mps")
    motion_seed = _required(motion_raw, "seed", "target.motion")
    if isinstance(motion_seed, bool) or not isinstance(motion_seed, int) or motion_seed < 0:
        raise ConfigError("target.motion.seed must be a non-negative integer")
    target = TargetConfig(
        initial_region=initial_region,
        max_speed_mps=max_target_speed,
        motion=TargetMotionConfig(
            mode=motion_mode.upper(),
            region=motion_region,
            speed_mps=motion_speed,
            initial_heading_deg=_finite_number(
                _required(motion_raw, "initial_heading_deg", "target.motion"),
                "target.motion.initial_heading_deg",
            ),
            direction_change_interval_s=_positive_number(
                _required(motion_raw, "direction_change_interval_s", "target.motion"),
                "target.motion.direction_change_interval_s",
            ),
            seed=motion_seed,
        ),
    )

    search_raw = _mapping(_required(root, "search", "config"), "search")
    transit_yaw_mode = _required(search_raw, "transit_yaw_mode", "search")
    if not isinstance(transit_yaw_mode, str) or transit_yaw_mode.upper() not in {
        "FACE_POINT",
        "COURSE_ALIGNED",
        "KEEP_CURRENT",
    }:
        raise ConfigError(
            "search.transit_yaw_mode must be FACE_POINT, COURSE_ALIGNED, or KEEP_CURRENT"
        )
    search = SearchConfig(
        radius_m=_positive_number(_required(search_raw, "radius_m", "search"), "search.radius_m"),
        timeout_s=_positive_number(_required(search_raw, "timeout_s", "search"), "search.timeout_s"),
        transit_yaw_mode=transit_yaw_mode.upper(),
    )

    # ``planner`` was introduced after the original unified configuration.
    # Its absence is intentionally backward compatible, while any supplied
    # value is parsed strictly and in full.
    if "planner" not in root:
        planner = _DEFAULT_PLANNER_CONFIG
    else:
        planner_raw_value = root["planner"]
        planner_raw = _mapping(planner_raw_value, "planner")
        if any(not isinstance(key, str) for key in planner_raw):
            raise ConfigError("planner keys must be strings")
        required_planner_keys = {
            "max_plan_steps",
            "max_goto_calls",
            "max_search_calls",
            "max_track_calls",
            "max_reacquire_attempts_per_track",
            "max_total_reacquire_attempts",
            "min_track_duration_s",
            "max_track_duration_s",
        }
        policy_planner_keys = {
            "default_on_target_lost",
            "default_reacquire_max_attempts",
            "default_reacquire_search_radius_m",
            "default_reacquire_timeout_s",
        }
        expected_planner_keys = required_planner_keys | policy_planner_keys
        unknown_planner_keys = sorted(set(planner_raw) - expected_planner_keys)
        if unknown_planner_keys:
            raise ConfigError(
                "planner contains unknown keys: "
                + ", ".join(str(key) for key in unknown_planner_keys)
            )
        # Policy fields were added after the original dynamic planner limits.
        # Old public configs that contain the limit block therefore inherit
        # the new trusted policy defaults rather than becoming unreadable.
        missing_planner_keys = sorted(required_planner_keys - set(planner_raw))
        if missing_planner_keys:
            raise ConfigError(
                "planner is missing required keys: "
                + ", ".join(missing_planner_keys)
            )
        try:
            planner = PlannerConfig(
                max_plan_steps=_positive_integer(
                    planner_raw["max_plan_steps"], "planner.max_plan_steps"
                ),
                max_goto_calls=_positive_integer(
                    planner_raw["max_goto_calls"], "planner.max_goto_calls"
                ),
                max_search_calls=_positive_integer(
                    planner_raw["max_search_calls"], "planner.max_search_calls"
                ),
                max_track_calls=_positive_integer(
                    planner_raw["max_track_calls"], "planner.max_track_calls"
                ),
                max_reacquire_attempts_per_track=_positive_integer(
                    planner_raw["max_reacquire_attempts_per_track"],
                    "planner.max_reacquire_attempts_per_track",
                ),
                max_total_reacquire_attempts=_positive_integer(
                    planner_raw["max_total_reacquire_attempts"],
                    "planner.max_total_reacquire_attempts",
                ),
                min_track_duration_s=_positive_number(
                    planner_raw["min_track_duration_s"],
                    "planner.min_track_duration_s",
                ),
                max_track_duration_s=_positive_number(
                    planner_raw["max_track_duration_s"],
                    "planner.max_track_duration_s",
                ),
                default_on_target_lost=planner_raw.get(
                    "default_on_target_lost",
                    _DEFAULT_PLANNER_CONFIG.default_on_target_lost,
                ),
                default_reacquire_max_attempts=_positive_integer(
                    planner_raw.get(
                        "default_reacquire_max_attempts",
                        _DEFAULT_PLANNER_CONFIG.default_reacquire_max_attempts,
                    ),
                    "planner.default_reacquire_max_attempts",
                ),
                default_reacquire_search_radius_m=_positive_number(
                    planner_raw.get(
                        "default_reacquire_search_radius_m",
                        _DEFAULT_PLANNER_CONFIG.default_reacquire_search_radius_m,
                    ),
                    "planner.default_reacquire_search_radius_m",
                ),
                default_reacquire_timeout_s=_positive_number(
                    planner_raw.get(
                        "default_reacquire_timeout_s",
                        _DEFAULT_PLANNER_CONFIG.default_reacquire_timeout_s,
                    ),
                    "planner.default_reacquire_timeout_s",
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid planner configuration: {exc}") from exc

    model_worker_raw = _strict_optional_block(
        root,
        "model_worker",
        frozenset({"max_inflight_per_uav", "request_timeout_s"}),
    )
    if model_worker_raw is None:
        model_worker = _DEFAULT_MODEL_WORKER_CONFIG
    else:
        max_inflight = _positive_integer(
            model_worker_raw["max_inflight_per_uav"],
            "model_worker.max_inflight_per_uav",
        )
        # The worker contract deliberately supports one in-flight request per
        # UAV.  A larger value would reintroduce out-of-order control results.
        if max_inflight != 1:
            raise ConfigError("model_worker.max_inflight_per_uav must be exactly 1")
        model_worker = ModelWorkerConfig(
            max_inflight_per_uav=max_inflight,
            request_timeout_s=_bounded_positive_number(
                model_worker_raw["request_timeout_s"],
                "model_worker.request_timeout_s",
                _MAX_REQUEST_TIMEOUT_S,
            ),
        )

    visual_review_raw = _strict_optional_block(
        root,
        "qwen_visual_review",
        frozenset(
            {
                "enabled",
                "mode",
                "goto_interval_s",
                "search_interval_s",
                "inspect_interval_s",
                "track_interval_s",
                "max_recent_frames",
                "max_image_side_px",
                "jpeg_quality",
                "hover_position_tolerance_m",
                "hover_max_correction_speed_mps",
                "blocking_hover_timeout_s",
                "blocking_timeout_fallback",
            }
        ),
    )
    if visual_review_raw is None:
        qwen_visual_review = _DEFAULT_QWEN_VISUAL_REVIEW_CONFIG
    else:
        mode = visual_review_raw["mode"]
        if not isinstance(mode, str) or mode not in {"shadow", "gate"}:
            raise ConfigError("qwen_visual_review.mode must be shadow or gate")
        max_recent_frames = _bounded_positive_integer(
            visual_review_raw["max_recent_frames"],
            "qwen_visual_review.max_recent_frames",
            3,
        )
        image_side = _bounded_positive_integer(
            visual_review_raw["max_image_side_px"],
            "qwen_visual_review.max_image_side_px",
            _MAX_IMAGE_SIDE_PX,
        )
        jpeg_quality = _bounded_positive_integer(
            visual_review_raw["jpeg_quality"],
            "qwen_visual_review.jpeg_quality",
            95,
        )
        blocking_timeout_fallback = visual_review_raw[
            "blocking_timeout_fallback"
        ]
        if not isinstance(blocking_timeout_fallback, str) or blocking_timeout_fallback not in {
            "RESUME_PREVIOUS",
            "CANCEL_AND_LAND",
        }:
            raise ConfigError(
                "qwen_visual_review.blocking_timeout_fallback must be "
                "RESUME_PREVIOUS or CANCEL_AND_LAND"
            )
        qwen_visual_review = QwenVisualReviewConfig(
            enabled=_boolean(
                visual_review_raw["enabled"], "qwen_visual_review.enabled"
            ),
            mode=mode,
            goto_interval_s=_bounded_positive_number(
                visual_review_raw["goto_interval_s"],
                "qwen_visual_review.goto_interval_s",
                _MAX_REVIEW_INTERVAL_S,
            ),
            search_interval_s=_bounded_positive_number(
                visual_review_raw["search_interval_s"],
                "qwen_visual_review.search_interval_s",
                _MAX_REVIEW_INTERVAL_S,
            ),
            inspect_interval_s=_bounded_positive_number(
                visual_review_raw["inspect_interval_s"],
                "qwen_visual_review.inspect_interval_s",
                _MAX_REVIEW_INTERVAL_S,
            ),
            track_interval_s=_bounded_positive_number(
                visual_review_raw["track_interval_s"],
                "qwen_visual_review.track_interval_s",
                _MAX_REVIEW_INTERVAL_S,
            ),
            max_recent_frames=max_recent_frames,
            max_image_side_px=image_side,
            jpeg_quality=jpeg_quality,
            hover_position_tolerance_m=_bounded_positive_number(
                visual_review_raw["hover_position_tolerance_m"],
                "qwen_visual_review.hover_position_tolerance_m",
                _MAX_HOVER_POSITION_TOLERANCE_M,
            ),
            hover_max_correction_speed_mps=_bounded_positive_number(
                visual_review_raw["hover_max_correction_speed_mps"],
                "qwen_visual_review.hover_max_correction_speed_mps",
                _MAX_HOVER_CORRECTION_SPEED_MPS,
            ),
            blocking_hover_timeout_s=_bounded_positive_number(
                visual_review_raw["blocking_hover_timeout_s"],
                "qwen_visual_review.blocking_hover_timeout_s",
                _MAX_REQUEST_TIMEOUT_S,
            ),
            blocking_timeout_fallback=blocking_timeout_fallback,
        )

    plan_revision_raw = _strict_optional_block(
        root,
        "plan_revision",
        frozenset({"enabled", "max_revisions", "cooldown_s"}),
    )
    if plan_revision_raw is None:
        plan_revision = _DEFAULT_PLAN_REVISION_CONFIG
    else:
        max_revisions = _nonnegative_integer(
            plan_revision_raw["max_revisions"], "plan_revision.max_revisions"
        )
        if max_revisions > _MAX_PLAN_REVISIONS:
            raise ConfigError(
                f"plan_revision.max_revisions must not exceed {_MAX_PLAN_REVISIONS}"
            )
        revision_enabled = _boolean(
            plan_revision_raw["enabled"],
            "plan_revision.enabled",
        )
        if revision_enabled and max_revisions == 0:
            raise ConfigError(
                "plan_revision.max_revisions must be positive when revision is enabled"
            )
        plan_revision = PlanRevisionConfig(
            enabled=revision_enabled,
            max_revisions=max_revisions,
            cooldown_s=_nonnegative_number(
                plan_revision_raw["cooldown_s"], "plan_revision.cooldown_s"
            ),
        )
        if plan_revision.cooldown_s > _MAX_REVISION_COOLDOWN_S:
            raise ConfigError(
                "plan_revision.cooldown_s must not exceed "
                f"{_MAX_REVISION_COOLDOWN_S:g}"
            )

    frame_store_raw = _strict_optional_block(
        root,
        "frame_store",
        frozenset({"max_frames", "max_bytes", "max_age_s"}),
    )
    if frame_store_raw is None:
        frame_store = _DEFAULT_FRAME_STORE_CONFIG
    else:
        frame_store = FrameStoreConfig(
            max_frames=_bounded_positive_integer(
                frame_store_raw["max_frames"],
                "frame_store.max_frames",
                _MAX_FRAME_STORE_FRAMES,
            ),
            max_bytes=_bounded_positive_integer(
                frame_store_raw["max_bytes"],
                "frame_store.max_bytes",
                _MAX_FRAME_STORE_BYTES,
            ),
            max_age_s=_bounded_positive_number(
                frame_store_raw["max_age_s"],
                "frame_store.max_age_s",
                _MAX_FRAME_STORE_AGE_S,
            ),
        )

    debug_images_raw = _strict_optional_block(
        root,
        "debug_images",
        frozenset({"enabled", "max_images_per_run"}),
    )
    if debug_images_raw is None:
        debug_images = _DEFAULT_DEBUG_IMAGES_CONFIG
    else:
        max_debug_images = _nonnegative_integer(
            debug_images_raw["max_images_per_run"],
            "debug_images.max_images_per_run",
        )
        if max_debug_images > _MAX_DEBUG_IMAGES_PER_RUN:
            raise ConfigError(
                "debug_images.max_images_per_run must not exceed "
                f"{_MAX_DEBUG_IMAGES_PER_RUN}"
            )
        debug_images = DebugImagesConfig(
            enabled=_boolean(debug_images_raw["enabled"], "debug_images.enabled"),
            max_images_per_run=max_debug_images,
        )

    experiment_raw = _mapping(_required(root, "experiment", "config"), "experiment")
    experiment_name = _required(experiment_raw, "name", "experiment")
    if not isinstance(experiment_name, str) or not _EXPERIMENT_NAME_RE.fullmatch(experiment_name):
        raise ConfigError("experiment.name may contain only letters, digits, '_' and '-'")
    experiment_seed = _required(experiment_raw, "seed", "experiment")
    if isinstance(experiment_seed, bool) or not isinstance(experiment_seed, int) or experiment_seed < 0:
        raise ConfigError("experiment.seed must be a non-negative integer")
    output_root = _required(experiment_raw, "output_root", "experiment")
    if output_root is not None and (not isinstance(output_root, str) or not output_root.strip()):
        raise ConfigError("experiment.output_root must be null or a non-empty path string")
    experiment = ExperimentConfig(
        name=experiment_name,
        seed=experiment_seed,
        output_root=output_root,
    )

    logging_raw = _mapping(_required(root, "logging", "config"), "logging")
    logging = LoggingConfig(
        terminal=_boolean(_required(logging_raw, "terminal", "logging"), "logging.terminal"),
        console_log_interval_updates=_positive_integer(
            _required(logging_raw, "console_log_interval_updates", "logging"),
            "logging.console_log_interval_updates",
        ),
        print_every_episode=_boolean(
            _required(logging_raw, "print_every_episode", "logging"),
            "logging.print_every_episode",
        ),
        debug_logging=_boolean(
            _required(logging_raw, "debug_logging", "logging"), "logging.debug_logging"
        ),
        csv=_boolean(_required(logging_raw, "csv", "logging"), "logging.csv"),
    )

    tensorboard_raw = _mapping(_required(root, "tensorboard", "config"), "tensorboard")
    tensorboard = TensorboardConfig(
        enabled=_boolean(_required(tensorboard_raw, "enabled", "tensorboard"), "tensorboard.enabled"),
        scalars_only=_boolean(
            _required(tensorboard_raw, "scalars_only", "tensorboard"),
            "tensorboard.scalars_only",
        ),
        log_interval_updates=_positive_integer(
            _required(tensorboard_raw, "log_interval_updates", "tensorboard"),
            "tensorboard.log_interval_updates",
        ),
        flush_interval_s=_positive_number(
            _required(tensorboard_raw, "flush_interval_s", "tensorboard"),
            "tensorboard.flush_interval_s",
        ),
    )
    if not tensorboard.scalars_only:
        raise ConfigError("tensorboard.scalars_only must remain true")

    checkpoint_raw = _mapping(_required(root, "checkpoint", "config"), "checkpoint")
    checkpoint = CheckpointConfig(
        save_best=_boolean(_required(checkpoint_raw, "save_best", "checkpoint"), "checkpoint.save_best"),
        save_latest=_boolean(
            _required(checkpoint_raw, "save_latest", "checkpoint"), "checkpoint.save_latest"
        ),
        latest_interval_steps=_positive_integer(
            _required(checkpoint_raw, "latest_interval_steps", "checkpoint"),
            "checkpoint.latest_interval_steps",
        ),
        save_periodic=_boolean(
            _required(checkpoint_raw, "save_periodic", "checkpoint"), "checkpoint.save_periodic"
        ),
        save_full_base_model=_boolean(
            _required(checkpoint_raw, "save_full_base_model", "checkpoint"),
            "checkpoint.save_full_base_model",
        ),
        save_adapter_only=_boolean(
            _required(checkpoint_raw, "save_adapter_only", "checkpoint"),
            "checkpoint.save_adapter_only",
        ),
        save_optimizer_in_latest_only=_boolean(
            _required(checkpoint_raw, "save_optimizer_in_latest_only", "checkpoint"),
            "checkpoint.save_optimizer_in_latest_only",
        ),
    )
    if checkpoint.save_periodic:
        raise ConfigError("checkpoint.save_periodic must remain false")
    if checkpoint.save_full_base_model:
        raise ConfigError("checkpoint.save_full_base_model must remain false")
    if not checkpoint.save_adapter_only:
        raise ConfigError("checkpoint.save_adapter_only must remain true")

    evaluation_raw = _mapping(_required(root, "evaluation", "config"), "evaluation")
    evaluation = EvaluationConfig(
        enabled=_boolean(_required(evaluation_raw, "enabled", "evaluation"), "evaluation.enabled"),
        interval_steps=_positive_integer(
            _required(evaluation_raw, "interval_steps", "evaluation"), "evaluation.interval_steps"
        ),
        num_validation_episodes=_positive_integer(
            _required(evaluation_raw, "num_validation_episodes", "evaluation"),
            "evaluation.num_validation_episodes",
        ),
        num_test_episodes=_positive_integer(
            _required(evaluation_raw, "num_test_episodes", "evaluation"),
            "evaluation.num_test_episodes",
        ),
        deterministic=_boolean(
            _required(evaluation_raw, "deterministic", "evaluation"),
            "evaluation.deterministic",
        ),
        fixed_validation_seeds=_boolean(
            _required(evaluation_raw, "fixed_validation_seeds", "evaluation"),
            "evaluation.fixed_validation_seeds",
        ),
        fixed_test_seeds=_boolean(
            _required(evaluation_raw, "fixed_test_seeds", "evaluation"),
            "evaluation.fixed_test_seeds",
        ),
    )

    artifacts_raw = _mapping(_required(root, "artifacts", "config"), "artifacts")
    artifacts = ArtifactsConfig(
        save_images=_boolean(_required(artifacts_raw, "save_images", "artifacts"), "artifacts.save_images"),
        save_videos=_boolean(_required(artifacts_raw, "save_videos", "artifacts"), "artifacts.save_videos"),
        save_trajectories=_boolean(
            _required(artifacts_raw, "save_trajectories", "artifacts"),
            "artifacts.save_trajectories",
        ),
        save_observations=_boolean(
            _required(artifacts_raw, "save_observations", "artifacts"),
            "artifacts.save_observations",
        ),
        save_raw_frames=_boolean(
            _required(artifacts_raw, "save_raw_frames", "artifacts"),
            "artifacts.save_raw_frames",
        ),
    )
    if any((
        artifacts.save_images,
        artifacts.save_videos,
        artifacts.save_trajectories,
        artifacts.save_observations,
        artifacts.save_raw_frames,
    )):
        raise ConfigError("raw image, video, trajectory, observation and frame artifacts are disabled")

    figures_raw = _mapping(_required(root, "figures", "config"), "figures")
    figure_format = _required(figures_raw, "format", "figures")
    if not isinstance(figure_format, str) or figure_format.lower() != "png":
        raise ConfigError("figures.format must be png")
    figures = FiguresConfig(
        enabled=_boolean(_required(figures_raw, "enabled", "figures"), "figures.enabled"),
        format="png",
        save_pdf=_boolean(_required(figures_raw, "save_pdf", "figures"), "figures.save_pdf"),
    )
    if figures.save_pdf:
        raise ConfigError("figures.save_pdf must remain false by default")

    storage_raw = _mapping(_required(root, "storage", "config"), "storage")
    storage = StorageConfig(
        min_free_space_gb_before_start=_nonnegative_number(
            _required(storage_raw, "min_free_space_gb_before_start", "storage"),
            "storage.min_free_space_gb_before_start",
        ),
        min_free_space_gb_during_run=_nonnegative_number(
            _required(storage_raw, "min_free_space_gb_during_run", "storage"),
            "storage.min_free_space_gb_during_run",
        ),
        warning_run_size_gb=_positive_number(
            _required(storage_raw, "warning_run_size_gb", "storage"),
            "storage.warning_run_size_gb",
        ),
        max_run_size_gb=_positive_number(
            _required(storage_raw, "max_run_size_gb", "storage"),
            "storage.max_run_size_gb",
        ),
    )
    if storage.min_free_space_gb_before_start < storage.min_free_space_gb_during_run:
        raise ConfigError("storage start free-space threshold must be >= the during-run threshold")
    if storage.warning_run_size_gb >= storage.max_run_size_gb:
        raise ConfigError("storage.warning_run_size_gb must be smaller than storage.max_run_size_gb")

    config = AppConfig(
        schema_version=schema_version,
        simulation=simulation,
        scene=scene,
        uav=uav,
        camera=camera,
        target=target,
        search=search,
        planner=planner,
        experiment=experiment,
        logging=logging,
        tensorboard=tensorboard,
        checkpoint=checkpoint,
        evaluation=evaluation,
        artifacts=artifacts,
        figures=figures,
        storage=storage,
        model_worker=model_worker,
        qwen_visual_review=qwen_visual_review,
        plan_revision=plan_revision,
        frame_store=frame_store,
        debug_images=debug_images,
        obstacle_perception=obstacle_perception,
    )
    _validate_spatial_bounds(config)
    return config
