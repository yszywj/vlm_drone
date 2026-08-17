"""Load and validate the unified UAV-agent YAML configuration."""

from __future__ import annotations

from dataclasses import replace
from math import isclose, isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from configs.schema import (
    AppConfig,
    ArtifactsConfig,
    CameraConfig,
    CheckpointConfig,
    EvaluationConfig,
    ExperimentConfig,
    FiguresConfig,
    LoggingConfig,
    PlannerConfig,
    SceneConfig,
    SearchConfig,
    StorageConfig,
    SimulationConfig,
    TargetConfig,
    TargetMotionConfig,
    TargetRegionConfig,
    TensorboardConfig,
    UavConfig,
)


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_DEFAULT_PLANNER_CONFIG = PlannerConfig()


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
    scene = SceneConfig(
        size_xyz_m=_float_vector(_required(scene_raw, "size_xyz_m", "scene"), "scene.size_xyz_m", 3)
    )
    if any(value <= 0.0 for value in scene.size_xyz_m):
        raise ConfigError("scene.size_xyz_m values must be greater than 0")

    uav_raw = _mapping(_required(root, "uav", "config"), "uav")
    uav = UavConfig(
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
        expected_planner_keys = {
            "max_plan_steps",
            "max_goto_calls",
            "max_search_calls",
            "max_track_calls",
            "max_reacquire_attempts_per_track",
            "max_total_reacquire_attempts",
            "min_track_duration_s",
            "max_track_duration_s",
        }
        unknown_planner_keys = sorted(set(planner_raw) - expected_planner_keys)
        if unknown_planner_keys:
            raise ConfigError(
                "planner contains unknown keys: "
                + ", ".join(str(key) for key in unknown_planner_keys)
            )
        missing_planner_keys = sorted(expected_planner_keys - set(planner_raw))
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
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid planner configuration: {exc}") from exc

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
    )
    _validate_spatial_bounds(config)
    return config
