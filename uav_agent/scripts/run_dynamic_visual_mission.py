#!/usr/bin/env python3
"""Run the routed dynamic mission with opt-in asynchronous Qwen vision.

This is an experimental Isaac standalone entry point.  Production target
perception uses an isolated loopback YOLO/YOLOE service and fails closed when
that service is unavailable.  The retained Oracle execution path is available
only through the two-part ``oracle_evaluation`` acknowledgement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = Path(
    os.environ.get(
        "UAV_AGENT_CONDA_ENV",
        "/home/amax/miniconda3/envs/r_isaac_sim",
    )
).expanduser()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep module import pure Python.  In particular, no Isaac-backed environment
# module is imported until main() has created SimulationApp.
from common.ids import (  # noqa: E402
    generate_routing_id,
    validate_mission_id,
    validate_uav_id,
)
from configs.loader import ConfigError, load_config  # noqa: E402


_DEFAULT_UAV_SAFETY_RADIUS_M = 0.5


def _build_oracle_evaluation_backend(uav_id: str) -> object:
    """Build the explicitly privileged backend with one trusted UAV route."""

    from perception import (
        GuardedPerceptionBackend,
        OraclePerception,
        PerceptionRuntimeProfile,
    )

    trusted_uav_id = validate_uav_id(uav_id)
    return GuardedPerceptionBackend(
        OraclePerception(uav_id=trusted_uav_id, target_id="target"),
        profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
        acknowledge_privileged_oracle=True,
    )


class LaunchConfigurationError(ValueError):
    """Raised before Isaac/Qwen startup when an opt-in is incomplete."""


@dataclass(frozen=True, slots=True)
class TestInjectionSpec:
    """One explicitly synthetic, simulation-time event trigger."""

    __test__ = False

    event_type: str
    trigger_at_s: float
    flag_name: str

    def __post_init__(self) -> None:
        if self.event_type not in {
            "PATH_BLOCKED",
            "SKILL_PROGRESS_STALLED",
            "TARGET_IDENTITY_UNCERTAIN",
        }:
            raise ValueError("unsupported test-injection event type")
        if (
            isinstance(self.trigger_at_s, bool)
            or not isinstance(self.trigger_at_s, (int, float))
            or not isfinite(float(self.trigger_at_s))
            or float(self.trigger_at_s) < 0.0
        ):
            raise ValueError("trigger_at_s must be finite and non-negative")
        object.__setattr__(self, "trigger_at_s", float(self.trigger_at_s))


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug-visualization",
        action="store_true",
        help=(
            "draw search geometry, camera frustum, planned route, "
            "and skill-colored executed trajectory in the GUI viewport"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="unified YAML config (default: %(default)s)",
    )
    parser.add_argument(
        "--planner",
        choices=("scripted", "llm", "dynamic_scripted", "dynamic_llm"),
        default="dynamic_llm",
        help="high-level planner (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=(
            "scripted_baseline",
            "classical_baseline",
            "qwen_open_sim",
            "qwen_critic_sim",
            "qwen_strict",
        ),
        default=None,
        help=(
            "closed benchmark condition; implies planner, planning contract, "
            "route backend and validation policy"
        ),
    )
    parser.add_argument(
        "--planning-contract",
        choices=("v2", "v3"),
        default="v2",
        help="initial structured planner contract (default: %(default)s)",
    )
    parser.add_argument(
        "--route-validation-mode",
        choices=("open_sim", "critic_sim", "strict"),
        default="strict",
        help="Qwen route validation policy (default: %(default)s)",
    )
    parser.add_argument(
        "--obstacle-perception",
        choices=("disabled", "ideal_camera"),
        default=None,
        help="override obstacle_perception.mode from config",
    )
    parser.add_argument(
        "--runtime-program",
        choices=("linear", "graph"),
        default="linear",
        help=(
            "mission control-flow representation; graph uses the validated "
            "linear TaskPlan adapter and owns SkillManager advancement "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI-compatible API key (never printed or persisted)",
    )
    parser.add_argument(
        "--uav-id",
        default="uav_1",
        type=validate_uav_id,
        help="trusted single-Agent routing ID (default: %(default)s)",
    )
    parser.add_argument("--takeoff-altitude", type=_positive_float, default=10.0)
    parser.add_argument("--track-duration", type=_positive_float, default=30.0)
    parser.add_argument("--max-sim-time", type=_positive_float, default=360.0)
    parser.add_argument("--start-altitude", type=_finite_float, default=0.0)

    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=None)
    parser.add_argument("--debug-ground-truth", action="store_true")

    parser.add_argument(
        "--enable-qwen-vision",
        action="store_true",
        help="enable asynchronous RGB review; disabled regardless of YAML without this flag",
    )
    parser.add_argument(
        "--enable-qwen-next-best-view",
        action="store_true",
        help=(
            "advertise and run asynchronous ADAPTIVE_NEXT_BEST_VIEW for "
            "Spatial V3; disabled by default"
        ),
    )
    parser.add_argument(
        "--debug-model-responses",
        action="store_true",
        help="persist only response length and a sanitized final 500 characters",
    )
    parser.add_argument(
        "--vision-review-mode",
        choices=("shadow", "gate"),
        default=None,
        help="override qwen_visual_review.mode from config",
    )
    parser.add_argument(
        "--acknowledge-vision-gate",
        action="store_true",
        help="explicitly authorize gate-mode semantic control effects",
    )
    parser.add_argument(
        "--perception-runtime-profile",
        choices=("production", "oracle_evaluation"),
        default="production",
        help="production fails closed until a real detector/tracker is installed",
    )
    parser.add_argument(
        "--acknowledge-privileged-oracle",
        action="store_true",
        help="second explicit opt-in required with oracle_evaluation",
    )
    parser.add_argument(
        "--target-perception-backend",
        choices=("disabled", "oracle_evaluation", "ultralytics_service"),
        default=None,
        help="override target_perception.backend from config",
    )
    parser.add_argument(
        "--yolo-service-url",
        default=None,
        help="override the loopback YOLO service URL",
    )
    parser.add_argument(
        "--yolo-request-timeout-s",
        type=_positive_float,
        default=None,
        help="override the bounded YOLO HTTP timeout",
    )
    parser.add_argument(
        "--yolo-max-result-age-s",
        type=_positive_float,
        default=None,
        help="override maximum accepted YOLO result age",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/dynamic_visual_missions",
        help="parent directory for bounded image-free run logs",
    )
    parser.add_argument(
        "--unseen-spatial-instruction",
        action="store_true",
        help="mark this episode as held-out spatial-language evaluation",
    )

    parser.add_argument(
        "--inject-path-blocked-at-s",
        type=_nonnegative_float,
        default=None,
        metavar="SECONDS",
        help="inject PATH_BLOCKED with source=test_injection",
    )
    parser.add_argument(
        "--inject-progress-stall-at-s",
        type=_nonnegative_float,
        default=None,
        metavar="SECONDS",
        help="inject SKILL_PROGRESS_STALLED with source=test_injection",
    )
    parser.add_argument(
        "--inject-identity-conflict-at-s",
        type=_nonnegative_float,
        default=None,
        metavar="SECONDS",
        help="inject TARGET_IDENTITY_UNCERTAIN with source=test_injection",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = build_argument_parser().parse_args(raw_argv)
    for destination in (
        "experiment_mode",
        "planner",
        "planning_contract",
        "route_validation_mode",
    ):
        flag = "--" + destination.replace("_", "-")
        setattr(
            args,
            f"_explicit_{destination}",
            any(item == flag or item.startswith(flag + "=") for item in raw_argv),
        )
    return resolve_experiment_launch_args(args)


def resolve_experiment_launch_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply one experiment profile or annotate a legacy launch deterministically."""

    from experiments.spatial_benchmark import (
        SpatialBenchmarkError,
        experiment_mode_profile,
        infer_experiment_mode,
        resolve_experiment_profile,
    )

    requested = getattr(args, "experiment_mode", None)
    args.experiment_mode_explicit = bool(
        getattr(args, "_explicit_experiment_mode", requested is not None)
    )
    try:
        if requested is None:
            mode = infer_experiment_mode(
                planner=args.planner,
                route_validation_mode=args.route_validation_mode,
            )
            profile = experiment_mode_profile(mode)
            exact_profile = (
                args.planner == profile.planner
                and args.planning_contract == profile.planning_contract
                and args.route_validation_mode == profile.route_validation_mode
                and not (
                    profile.mode.value
                    in {"scripted_baseline", "classical_baseline"}
                    and bool(
                        args.enable_qwen_vision
                        or args.enable_qwen_next_best_view
                    )
                )
                and not bool(args.enable_qwen_next_best_view)
            )
            # Preserve all legacy defaults, but never label a non-comparable
            # legacy combination as one of the five benchmark conditions.
            args.experiment_mode = mode.value if exact_profile else "unspecified"
            args.inferred_experiment_mode = mode.value
        else:
            profile = resolve_experiment_profile(
                requested,
                planner=(
                    args.planner
                    if getattr(args, "_explicit_planner", False)
                    else None
                ),
                planning_contract=(
                    args.planning_contract
                    if getattr(args, "_explicit_planning_contract", False)
                    else None
                ),
                route_validation_mode=(
                    args.route_validation_mode
                    if getattr(args, "_explicit_route_validation_mode", False)
                    else None
                ),
            )
            args.planner = profile.planner
            args.planning_contract = profile.planning_contract
            args.route_validation_mode = profile.route_validation_mode
        args.route_planner_backend = profile.route_planner_backend
    except SpatialBenchmarkError as exc:
        raise LaunchConfigurationError(str(exc)) from None
    return args


def build_test_injection_specs(
    args: argparse.Namespace,
) -> tuple[TestInjectionSpec, ...]:
    pairs = (
        (
            "PATH_BLOCKED",
            args.inject_path_blocked_at_s,
            "--inject-path-blocked-at-s",
        ),
        (
            "SKILL_PROGRESS_STALLED",
            args.inject_progress_stall_at_s,
            "--inject-progress-stall-at-s",
        ),
        (
            "TARGET_IDENTITY_UNCERTAIN",
            args.inject_identity_conflict_at_s,
            "--inject-identity-conflict-at-s",
        ),
    )
    return tuple(
        TestInjectionSpec(event_type, value, flag)
        for event_type, value, flag in pairs
        if value is not None
    )


def validate_launch_args(
    args: argparse.Namespace,
    *,
    configured_review_mode: str = "shadow",
) -> tuple[str, tuple[TestInjectionSpec, ...]]:
    """Return the effective review mode or reject an unsafe combination."""

    if not isinstance(args.instruction, str) or not args.instruction.strip():
        raise LaunchConfigurationError("--instruction must not be empty")
    if not hasattr(args, "route_planner_backend"):
        resolve_experiment_launch_args(args)
    if (
        args.experiment_mode in {"scripted_baseline", "classical_baseline"}
        and bool(getattr(args, "experiment_mode_explicit", False))
        and (args.enable_qwen_vision or args.enable_qwen_next_best_view)
    ):
        raise LaunchConfigurationError(
            f"--experiment-mode {args.experiment_mode} forbids "
            "Qwen visual capabilities so the non-Qwen baseline remains isolated"
        )
    if args.enable_qwen_next_best_view and (
        args.planner != "dynamic_llm" or args.planning_contract != "v3"
    ):
        raise LaunchConfigurationError(
            "--enable-qwen-next-best-view requires --planner dynamic_llm "
            "and --planning-contract v3"
        )
    if args.planning_contract == "v3" and args.planner != "dynamic_llm":
        raise LaunchConfigurationError(
            "--planning-contract v3 currently requires --planner dynamic_llm"
        )
    mode = args.vision_review_mode or configured_review_mode
    if mode not in {"shadow", "gate"}:
        raise LaunchConfigurationError("visual review mode must be shadow or gate")
    if args.acknowledge_vision_gate and mode != "gate":
        raise LaunchConfigurationError(
            "--acknowledge-vision-gate is only valid with gate mode"
        )
    if mode == "gate" and not args.enable_qwen_vision:
        raise LaunchConfigurationError("gate mode requires --enable-qwen-vision")
    if mode == "gate" and not args.acknowledge_vision_gate:
        raise LaunchConfigurationError(
            "gate mode requires --acknowledge-vision-gate"
        )
    if getattr(args, "runtime_program", "linear") == "graph":
        obstacle_mode = getattr(
            args,
            "effective_obstacle_perception_mode",
            args.obstacle_perception or "disabled",
        )
        if mode == "gate":
            raise LaunchConfigurationError(
                "--runtime-program graph does not accept TaskPlan gate "
                "revisions; a ProgramPatch coordinator is required"
            )
        if obstacle_mode != "disabled":
            raise LaunchConfigurationError(
                "--runtime-program graph does not accept obstacle TaskPlan "
                "replanning; use --obstacle-perception disabled until the "
                "route coordinator emits ProgramPatch"
            )

    oracle_selected = args.perception_runtime_profile == "oracle_evaluation"
    if oracle_selected != bool(args.acknowledge_privileged_oracle):
        if oracle_selected:
            raise LaunchConfigurationError(
                "oracle_evaluation requires --acknowledge-privileged-oracle"
            )
        raise LaunchConfigurationError(
            "Oracle acknowledgement is invalid in production profile"
        )
    target_backend = getattr(args, "effective_target_perception_backend", None)
    if target_backend is None:
        target_backend = getattr(args, "target_perception_backend", None)
    if target_backend is None and oracle_selected:
        # Backward-compatible explicit Oracle command: the profile plus its
        # acknowledgement remains the required two-part opt-in.
        target_backend = "oracle_evaluation"
    if target_backend is None:
        raise LaunchConfigurationError(
            "production visual geometry is unavailable: no real detector/tracker "
            "backend was selected"
        )
    if oracle_selected and target_backend != "oracle_evaluation":
        raise LaunchConfigurationError(
            "oracle_evaluation profile requires target backend oracle_evaluation"
        )
    if not oracle_selected and target_backend == "oracle_evaluation":
        raise LaunchConfigurationError(
            "production profile forbids target backend oracle_evaluation"
        )
    if not oracle_selected and target_backend not in {
        "disabled",
        "ultralytics_service",
    }:
        raise LaunchConfigurationError("invalid production target backend")

    injections = build_test_injection_specs(args)
    if injections and not args.enable_qwen_vision:
        raise LaunchConfigurationError(
            "test-injection review events require --enable-qwen-vision"
        )
    return mode, injections


def make_test_injection_event(
    spec: TestInjectionSpec,
    *,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    timestamp_s: float,
) -> object:
    """Build one routed event with an unambiguous synthetic source label."""

    from runtime.events import EventSeverity, MissionEvent, MissionEventType

    return MissionEvent(
        event_id=generate_routing_id("event"),
        mission_id=validate_mission_id(mission_id),
        uav_id=validate_uav_id(uav_id),
        plan_version=plan_version,
        timestamp_s=timestamp_s,
        event_type=MissionEventType(spec.event_type),
        severity=EventSeverity.WARNING,
        payload={
            "source": "test_injection",
            "trigger_flag": spec.flag_name,
            "trigger_at_s": spec.trigger_at_s,
        },
    )


def startup_fields(
    *,
    uav_id: str,
    mission_id: str,
    planner: str,
    review_enabled: bool,
    review_mode: str,
    perception_profile: str,
    oracle_acknowledged: bool,
    target_perception_backend: str = "unspecified",
    yolo_service_url: str | None = None,
    model_family: str | None = None,
    geometry_mode: str | None = None,
    state_estimator: str | None = None,
    confirmation_mode: str | None = None,
) -> tuple[tuple[str, object], ...]:
    """Stable launch metadata used by both the terminal and unit tests."""

    return (
        ("uav_id", validate_uav_id(uav_id)),
        ("mission_id", validate_mission_id(mission_id)),
        ("planner", planner),
        (
            "qwen_visual_review",
            f"enabled:{review_mode}" if review_enabled else "disabled",
        ),
        ("perception_runtime_profile", perception_profile),
        ("oracle_acknowledged", oracle_acknowledged),
        ("target_perception_backend", target_perception_backend),
        ("yolo_service_url", yolo_service_url or "none"),
        ("model_family", model_family or "none"),
        ("geometry_mode", geometry_mode or "none"),
        ("state_estimator", state_estimator or "none"),
        ("confirmation_mode", confirmation_mode or "none"),
    )


def _json_compatible(value: object) -> object:
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"manifest value {type(value).__name__} is not JSON-compatible")


def _read_git_commit() -> str | None:
    """Best-effort read-only provenance; never mutates the repository."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def _active_flight_corridor(
    manager: object,
    observation: object,
    *,
    safety_radius_m: float = _DEFAULT_UAV_SAFETY_RADIUS_M,
) -> object | None:
    """Build a short trusted corridor without consulting target ground truth."""

    from common.obstacle_types import FlightCorridor
    from skills.follow_route import FollowRouteGoal
    from skills.goto import GotoGoal
    from skills.search import SearchGoal, SearchGoalV3
    from skills.search_geometry import region_center

    invocation = getattr(manager, "active_invocation", None)
    if invocation is None:
        return None
    pose = observation.uav_pose
    start = (float(pose.x), float(pose.y), float(pose.z))
    goal = invocation.goal
    end: tuple[float, float, float] | None = None
    if isinstance(goal, GotoGoal):
        end = tuple(float(item) for item in goal.position)
    elif isinstance(goal, FollowRouteGoal):
        try:
            index = int(manager.get_feedback().data.get("waypoint_index", 0))
        except Exception:
            index = 0
        if 0 <= index < len(goal.waypoints):
            end = tuple(float(item) for item in goal.waypoints[index])
    elif isinstance(goal, (SearchGoal, SearchGoalV3)):
        try:
            feedback = manager.get_feedback().data
            active_waypoint = feedback.get("active_waypoint_xyz_m")
            if (
                isinstance(active_waypoint, (tuple, list))
                and len(active_waypoint) == 3
            ):
                end = tuple(float(item) for item in active_waypoint)
        except Exception:
            end = None
        if end is None and isinstance(goal, SearchGoal):
            end = (
                float(goal.center[0]),
                float(goal.center[1]),
                float(goal.search_altitude),
            )
        elif end is None:
            center = region_center(goal.region)
            end = (
                float(center[0]),
                float(center[1]),
                float(goal.search_altitude_m),
            )
    if end is None:
        velocity = observation.uav_velocity
        speed_sq = float(sum(float(item) ** 2 for item in velocity))
        if speed_sq > 1e-12:
            end = tuple(start[index] + 5.0 * float(velocity[index]) for index in range(3))
    if end is None or sum((right - left) ** 2 for left, right in zip(start, end)) <= 1e-12:
        return None
    return FlightCorridor(start, end, safety_radius_m)


def _route_debug_records(
    manager: object,
    visual_runtime: object | None,
) -> tuple[object, ...]:
    """Return rejected raw proposals before registered route lifecycles."""

    registry = getattr(manager, "route_registry", None)
    coordinator = (
        None
        if visual_runtime is None
        else getattr(visual_runtime, "obstacle_revision_coordinator", None)
    )
    proposal_records = () if coordinator is None else tuple(coordinator.records)
    registry_records = () if registry is None else tuple(registry.records)
    return (*proposal_records, *registry_records)


def _camera_geometry_from_observation(
    observation: object,
    *,
    uav_id: str,
    camera_config: object,
    far_clip_m: float,
) -> object | None:
    """Project one routed Observation onto the pure-Python camera contract."""

    from common.obstacle_types import CameraGeometry

    if (
        observation.camera_position_m is None
        or observation.camera_orientation_wxyz is None
    ):
        return None
    return CameraGeometry(
        frame_id=generate_routing_id("camera_frame"),
        uav_id=uav_id,
        timestamp_s=float(observation.timestamp),
        position_world_m=tuple(float(item) for item in observation.camera_position_m),
        orientation_world_from_camera_wxyz=tuple(
            float(item) for item in observation.camera_orientation_wxyz
        ),
        resolution_wh_px=tuple(camera_config.resolution_wh_px),
        horizontal_fov_deg=float(camera_config.horizontal_fov_deg),
        near_clip_m=0.1,
        far_clip_m=max(0.11, float(far_clip_m)),
    )


def _build_initial_spatial_resolver(
    world_context: object,
    uav_pose: object,
) -> object:
    """Bind Spatial V3 relative frames to trusted post-reset geometry.

    The pose is sampled from the controller after environment setup, so
    ``UAV_START_FLU`` never relies on an assumed yaw.  ``UAV_HOLD_FLU`` and
    ``CAMERA_FLU`` deliberately remain unavailable during initial planning.
    """

    from planner.spatial_resolver import FramePose, SpatialResolver

    landing_zones = getattr(world_context, "landing_zones", None)
    if not isinstance(landing_zones, dict) and not hasattr(
        landing_zones, "__getitem__"
    ):
        raise LaunchConfigurationError(
            "Spatial V3 requires trusted landing-zone geometry"
        )
    try:
        home = landing_zones["home"]
    except (KeyError, TypeError):
        raise LaunchConfigurationError(
            "Spatial V3 HOME_ENU requires a trusted 'home' landing zone"
        ) from None

    named_locations: dict[str, tuple[float, float, float]] = {
        str(name): (
            float(zone.position_xy_m[0]),
            float(zone.position_xy_m[1]),
            float(zone.ground_altitude_m),
        )
        for name, zone in landing_zones.items()
    }
    for name, point in getattr(world_context, "navigation_points", {}).items():
        named_locations[str(name)] = tuple(
            float(item) for item in point.position_xyz_m
        )

    start = FramePose(
        (
            float(uav_pose.x),
            float(uav_pose.y),
            float(uav_pose.z),
        ),
        yaw_rad=float(uav_pose.yaw),
    )
    return SpatialResolver(
        home_pose=FramePose(
            (
                float(home.position_xy_m[0]),
                float(home.position_xy_m[1]),
                float(home.ground_altitude_m),
            )
        ),
        uav_start_pose=start,
        named_locations=named_locations,
    )


def visual_manifest_context(
    *,
    args: argparse.Namespace,
    config: object,
    mission_id: str,
    review_mode: str,
    model_name: str,
) -> dict[str, object]:
    """Return bounded launch provenance without credentials or image data."""

    blocks = {
        name: getattr(config, name)
        for name in (
            "model_worker",
            "qwen_visual_review",
            "plan_revision",
            "frame_store",
            "debug_images",
            "target_perception",
        )
    }
    obstacle_mode = getattr(
        args,
        "effective_obstacle_perception_mode",
        args.obstacle_perception or config.obstacle_perception.mode,
    )
    return {
        "mission_id": validate_mission_id(mission_id),
        "uav_id": validate_uav_id(args.uav_id),
        "planner": args.planner,
        "model": model_name,
        "git_commit": _read_git_commit(),
        "perception_runtime_profile": args.perception_runtime_profile,
        "oracle_acknowledged": bool(args.acknowledge_privileged_oracle),
        "target_perception_backend": getattr(
            args,
            "effective_target_perception_backend",
            config.target_perception.backend,
        ),
        "target_evaluator_enabled": bool(
            args.debug_ground_truth
            or args.perception_runtime_profile == "oracle_evaluation"
        ),
        "target_evaluator_ground_truth_to_control": False,
        "qwen_visual_review_mode": review_mode,
        "qwen_next_best_view_enabled": bool(
            args.enable_qwen_next_best_view
        ),
        "experiment_mode": args.experiment_mode,
        "route_planner_backend": args.route_planner_backend,
        "unseen_spatial_instruction": bool(args.unseen_spatial_instruction),
        "planning_contract": args.planning_contract,
        "runtime_program": args.runtime_program,
        "route_validation_mode": args.route_validation_mode,
        "obstacle_perception_mode": obstacle_mode,
        "obstacle_perception_source": (
            "ideal_camera_obstacle_perception"
            if obstacle_mode == "ideal_camera"
            else "disabled"
        ),
        "obstacle_perception_privileged": obstacle_mode == "ideal_camera",
        "obstacle_coordinate_frame": (
            "CAMERA_FLU" if obstacle_mode == "ideal_camera" else None
        ),
        "prompt_schema_versions": {
            "initial_planner": 3 if args.planning_contract == "v3" else 2,
            "visual_review": 1,
            "runtime_visual_assessment": 2,
            "obstacle_route_revision": 3,
            "mission_program": 1,
            "next_best_view": 1,
        },
        "configuration": _json_compatible(blocks),
    }


@dataclass(slots=True)
class _VisualRuntime:
    coordinator: object | None
    worker: object | None
    event_bus: object
    output_parent: Path
    model_name: str
    revision_coordinator: object | None = None
    obstacle_revision_coordinator: object | None = None
    runtime_visual_assessment_coordinator: object | None = None
    next_best_view_provider: object | None = None
    target_debug_image_writer: object | None = None
    extra_workers: list[object] = field(default_factory=list)
    seen_review_ids: frozenset[str] = frozenset()
    seen_event_ids: frozenset[str] = frozenset()
    seen_revision_keys: frozenset[str] = frozenset()
    seen_obstacle_revision_keys: frozenset[str] = frozenset()
    seen_runtime_assessment_request_ids: frozenset[str] = frozenset()
    seen_next_best_view_request_ids: frozenset[str] = frozenset()
    transition_cursor: int = 0
    search_report_cursor: int = 0
    logger: object | None = None
    run_dir: Path | None = None
    hover_started_at_s: float | None = None
    manifest_context: dict[str, object] = field(default_factory=dict)
    terminal_manifest: dict[str, object] = field(default_factory=dict)

    def begin_logging(
        self,
        mission_id: str,
        *,
        manifest_context: dict[str, object] | None = None,
    ) -> None:
        from experiments.sparse_mission_logger import (
            RunManifestMetadata,
            SparseMissionLogger,
        )

        self.run_dir = self.output_parent.expanduser() / validate_mission_id(mission_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_context = dict(manifest_context or {})
        try:
            self.logger = SparseMissionLogger(
                self.run_dir,
                manifest_metadata=RunManifestMetadata(
                    experiment_mode=str(
                        self.manifest_context.get("experiment_mode", "unspecified")
                    ),
                    route_planner_backend=str(
                        self.manifest_context.get(
                            "route_planner_backend", "unspecified"
                        )
                    ),
                    planning_contract=str(
                        self.manifest_context.get("planning_contract", "unspecified")
                    ),
                    runtime_program=str(
                        self.manifest_context.get("runtime_program", "linear")
                    ),
                    route_validation_mode=str(
                        self.manifest_context.get(
                            "route_validation_mode", "unspecified"
                        )
                    ),
                    obstacle_perception_mode=str(
                        self.manifest_context.get(
                            "obstacle_perception_mode", "unspecified"
                        )
                    ),
                    prompt_schema_versions=dict(
                        self.manifest_context.get("prompt_schema_versions", {})
                    ),
                    git_commit=self.manifest_context.get("git_commit"),
                ),
            )
        except Exception as exc:
            # Persistent logging is observational.  A filesystem failure after
            # takeoff must never stop the control loop or strand the vehicle.
            self.logger = None
            print(
                f"[Logging] sparse logs disabled: {type(exc).__name__}",
                file=sys.stderr,
            )
        self._write_manifest()

    def _disable_failed_logger(self, exc: Exception) -> None:
        logger = self.logger
        self.logger = None
        if logger is not None:
            try:
                logger.close()
            except Exception:
                pass
        print(
            f"[Logging] sparse log append disabled: {type(exc).__name__}",
            file=sys.stderr,
        )

    def emit_new_reviews(
        self,
        *,
        step_id: str | None,
        skill: str,
    ) -> None:
        from experiments.sparse_mission_logger import QwenReviewLogRecord

        if self.coordinator is None:
            return
        records = self.coordinator.records
        for record in records:
            if record.review_id in self.seen_review_ids:
                continue
            payload = record.to_dict()
            print(
                "[QwenVisualReview] "
                f"mission_id={record.mission_id} uav_id={record.uav_id} "
                f"plan_version={record.plan_version} "
                f"step_id={step_id or 'none'} skill={skill} payload="
                + json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
            )
            if self.logger is not None:
                usage = dict(record.token_usage)
                try:
                    self.logger.log_qwen_review(
                        QwenReviewLogRecord(
                            review_id=record.review_id,
                            request_id=record.request_id,
                            mission_id=record.mission_id,
                            uav_id=record.uav_id,
                            plan_version=record.plan_version,
                            step_id=step_id,
                            frame_id=record.frame_id,
                            observation_timestamp_s=record.observation_timestamp_s,
                            decision=record.decision or record.error_code or "UNKNOWN",
                            bbox_xyxy_normalized=record.bbox_xyxy_normalized,
                            prompt_tokens=int(
                                usage.get("prompt_tokens", usage.get("input_tokens", 0))
                            ),
                            completion_tokens=int(
                                usage.get("completion_tokens", usage.get("output_tokens", 0))
                            ),
                            total_tokens=int(usage.get("total_tokens", 0)),
                            latency_s=record.latency_s,
                            stale=record.stale,
                            accepted=record.accepted_for_control,
                            timeout=record.error_code == "TIMEOUT",
                            semantic_source=record.semantic_source,
                            geometry_source=record.geometry_source,
                            error_code=record.error_code,
                            stale_reasons=record.stale_reasons,
                            response_text_length=record.response_text_length,
                            response_text_tail=record.response_text_tail,
                        )
                    )
                except Exception as exc:
                    self._disable_failed_logger(exc)
        self.seen_review_ids = frozenset(record.review_id for record in records)

    def emit_new_runtime_assessments(self, *, step_id: str | None) -> None:
        coordinator = self.runtime_visual_assessment_coordinator
        if coordinator is None:
            return
        from experiments.sparse_mission_logger import QwenReviewLogRecord

        seen = set(self.seen_runtime_assessment_request_ids)
        for record in tuple(coordinator.records):
            if record.request_id in seen:
                continue
            assessment = record.assessment
            decision = (
                record.error_code or "UNKNOWN"
                if assessment is None
                else assessment.decision.value
            )
            grounded = bool(
                assessment is not None
                and any(item.geometry_grounded for item in assessment.hazards)
            )
            bbox = (
                None
                if assessment is None
                else assessment.target_assessment.bbox_xyxy_normalized
            )
            print(
                "[RuntimeVisualAssessmentV2] "
                f"mission_id={record.mission_id} uav_id={record.uav_id} "
                f"plan_version={record.plan_version} "
                f"step_id={step_id or 'none'} decision={decision} "
                f"stale={record.stale} applied={record.applied_to_control}"
            )
            if self.logger is not None:
                try:
                    self.logger.log_qwen_review(
                        QwenReviewLogRecord(
                            review_id=record.review_id,
                            request_id=record.request_id,
                            mission_id=record.mission_id,
                            uav_id=record.uav_id,
                            plan_version=record.plan_version,
                            step_id=step_id,
                            frame_id=record.frame_id,
                            observation_timestamp_s=(
                                record.observation_timestamp_s
                            ),
                            decision=decision,
                            bbox_xyxy_normalized=bbox,
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            latency_s=max(
                                0.0,
                                (record.completed_timestamp_s or record.submitted_timestamp_s)
                                - record.submitted_timestamp_s,
                            ),
                            stale=record.stale,
                            accepted=record.applied_to_control,
                            timeout=record.error_code == "TIMEOUT",
                            geometry_source=(
                                "ideal_camera_obstacle_perception"
                                if grounded
                                else "none"
                            ),
                            error_code=record.error_code,
                        )
                    )
                except Exception as exc:
                    self._disable_failed_logger(exc)
            seen.add(record.request_id)
        self.seen_runtime_assessment_request_ids = frozenset(seen)

    def emit_new_next_best_view_proposals(self) -> None:
        """Persist completed macro-view proposals without retaining RGB data."""

        provider = self.next_best_view_provider
        if provider is None:
            return
        from experiments.sparse_mission_logger import (
            ModelCallKind,
            ModelProposalLogRecord,
        )

        seen = set(self.seen_next_best_view_request_ids)
        for record in tuple(getattr(provider, "records", ())):
            if record.request_id in seen:
                continue
            payload = record.to_dict()
            # The provider audit contract contains only structured text and
            # coordinates. SparseMissionLogger independently rejects image or
            # base64 keys if that invariant ever regresses.
            print(
                "[QwenNextBestView] "
                f"mission_id={record.mission_id} uav_id={record.uav_id} "
                f"plan_version={record.plan_version} "
                f"request_id={record.request_id} "
                f"decision={record.decision or record.error_code}"
            )
            if self.logger is not None:
                try:
                    self.logger.log_model_proposal(
                        ModelProposalLogRecord(
                            proposal_id=generate_routing_id("proposal_nbv"),
                            mission_id=record.mission_id,
                            uav_id=record.uav_id,
                            plan_version=record.plan_version,
                            timestamp_s=record.observation_timestamp_s,
                            call_kind=ModelCallKind.NEXT_BEST_VIEW,
                            proposal_index=record.proposal_index,
                            proposal=payload,
                            final_proposal=record.error_code is None,
                        )
                    )
                except Exception as exc:
                    self._disable_failed_logger(exc)
            seen.add(record.request_id)
        self.seen_next_best_view_request_ids = frozenset(seen)

    def log_injection(self, event: object, *, skill: str, step_id: str) -> None:
        payload = event.to_dict()
        print(
            "[TestInjection] "
            f"mission_id={event.mission_id} uav_id={event.uav_id} "
            f"plan_version={event.plan_version} step_id={step_id} "
            f"skill={skill} payload="
            + json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )

    def emit_new_events(self, *, skill: str, step_id: str) -> None:
        from experiments.sparse_mission_logger import MissionEventLogRecord
        from runtime.events import json_payload_to_dict

        events = self.event_bus.recent()
        for event in events:
            if event.event_id in self.seen_event_ids:
                continue
            payload = json_payload_to_dict(event.payload)
            source = payload.get("source", "runtime")
            record = MissionEventLogRecord(
                timestamp_s=event.timestamp_s,
                mission_id=event.mission_id,
                uav_id=event.uav_id,
                plan_version=event.plan_version,
                step_id=step_id,
                skill=skill,
                event=event.event_type.value,
                status=event.severity.value,
                reason=f"source={source}",
            )
            print(record.to_terminal_line())
            if self.logger is not None:
                try:
                    self.logger.log_mission_event(record)
                except Exception as exc:
                    self._disable_failed_logger(exc)
        self.seen_event_ids = frozenset(event.event_id for event in events)

    def emit_new_transitions(self, manager: object) -> None:
        from experiments.sparse_mission_logger import SkillTransitionLogRecord
        from skills.types import SkillName

        transitions = manager.transition_log
        for transition in transitions[self.transition_cursor :]:
            step_id = transition.new_step_id or transition.old_step_id or "none"
            record = SkillTransitionLogRecord(
                timestamp_s=transition.timestamp,
                mission_id=transition.mission_id,
                uav_id=transition.uav_id,
                plan_version=transition.plan_version,
                step_id=step_id,
                old_skill=(
                    "NONE" if transition.old_skill is None else transition.old_skill.value
                ),
                new_skill=(
                    "NONE" if transition.new_skill is None else transition.new_skill.value
                ),
                old_status=(
                    "NONE" if transition.old_status is None else transition.old_status.name
                ),
                result_code=(
                    "NONE"
                    if transition.result_code is None
                    else transition.result_code.name
                ),
                reason=transition.reason,
            )
            terminal_skill = record.new_skill
            if terminal_skill == "NONE":
                terminal_skill = record.old_skill
            print(
                "[SkillTransition] "
                f"mission_id={record.mission_id} uav_id={record.uav_id} "
                f"plan_version={record.plan_version} step_id={record.step_id} "
                f"skill={terminal_skill} old_skill={record.old_skill} "
                f"new_skill={record.new_skill} reason={record.reason}"
            )
            if self.logger is not None:
                try:
                    self.logger.log_skill_transition(record)
                    if (
                        transition.new_skill is SkillName.HOVER
                        and transition.old_skill is not SkillName.HOVER
                        and self.hover_started_at_s is None
                    ):
                        self.hover_started_at_s = transition.timestamp
                        self.logger.record_supervisory_hover_started()
                    elif (
                        transition.old_skill is SkillName.HOVER
                        and transition.new_skill is not SkillName.HOVER
                        and self.hover_started_at_s is not None
                    ):
                        self.logger.record_supervisory_hover_duration(
                            max(0.0, transition.timestamp - self.hover_started_at_s)
                        )
                        self.hover_started_at_s = None
                except Exception as exc:
                    self._disable_failed_logger(exc)
            self.transition_cursor += 1

    def emit_new_search_metrics(self, manager: object) -> None:
        """Record terminal SEARCH coverage from routed execution reports."""

        from skills.types import SkillName

        reports = manager.execution_reports
        plan = manager.task_plan
        steps_by_id = (
            {} if plan is None else {step.step_id: step for step in plan.steps}
        )
        for report in reports[self.search_report_cursor :]:
            self.search_report_cursor += 1
            if report.skill_name is not SkillName.SEARCH:
                continue
            result = report.feedback_or_result
            data = result.get("data", result)
            if not isinstance(data, dict):
                data = {}
            coverage = data.get("coverage_ratio", 0.0)
            visited = data.get("visited_viewpoints", ())
            elapsed = data.get("elapsed_time")
            step = steps_by_id.get(report.step_id)
            params = {} if step is None else step.params
            region = params.get("region")
            raw_shape = getattr(region, "shape", None)
            region_shape = (
                str(getattr(raw_shape, "value", raw_shape))
                if raw_shape is not None
                else str(region or "UNKNOWN")
            )
            strategy = params.get("strategy")
            raw_strategy = getattr(strategy, "kind", None)
            search_strategy = (
                str(getattr(raw_strategy, "value", raw_strategy))
                if raw_strategy is not None
                else "PERIMETER_V1"
            )
            visited_count = (
                len(visited)
                if isinstance(visited, (tuple, list))
                else int(data.get("visited_viewpoint_count", 0))
            )
            target_detection_time_s = (
                float(elapsed)
                if report.result_code is not None
                and report.result_code.value == "TARGET_FOUND"
                and isinstance(elapsed, (int, float))
                and not isinstance(elapsed, bool)
                else None
            )
            print(
                "[SearchMetrics] "
                f"mission_id={report.mission_id} uav_id={report.uav_id} "
                f"plan_version={report.plan_version} step_id={report.step_id} "
                f"region_shape={region_shape} strategy={search_strategy} "
                f"coverage_ratio={float(coverage):.3f} "
                f"visited_viewpoint_count={visited_count}"
            )
            if self.logger is not None:
                try:
                    self.logger.record_search_metrics(
                        region_shape=region_shape,
                        search_strategy=search_strategy,
                        coverage_ratio=float(coverage),
                        visited_viewpoint_count=visited_count,
                        target_detection_time_s=target_detection_time_s,
                    )
                except Exception as exc:
                    self._disable_failed_logger(exc)
                    return

    def emit_new_revisions(self, *, skill: str, step_id: str) -> None:
        coordinator = self.revision_coordinator
        if coordinator is None:
            return
        records = tuple(coordinator.records)
        seen = set(self.seen_revision_keys)
        for record in records:
            key = f"{record.request_id}:{record.outcome}:{record.timestamp_s}"
            if key in seen:
                continue
            print(
                "[PlanRevision] "
                f"mission_id={record.mission_id} uav_id={record.uav_id} "
                f"plan_version={record.new_plan_version or record.old_plan_version} "
                f"step_id={record.new_step_id or step_id} skill={skill} "
                f"outcome={record.outcome} request_id={record.request_id}"
            )
            if self.logger is not None:
                try:
                    self.logger.record_plan_revision_model_call()
                    if record.outcome == "ACCEPTED":
                        self.logger.record_plan_revision()
                except Exception as exc:
                    self._disable_failed_logger(exc)
            seen.add(key)
        self.seen_revision_keys = frozenset(seen)

    def emit_new_obstacle_revisions(
        self,
        *,
        mission_id: str,
        uav_id: str,
        plan_version: int,
    ) -> None:
        """Persist every route proposal/Critic pair without retaining RGB."""

        coordinator = self.obstacle_revision_coordinator
        if coordinator is None:
            return
        from experiments.sparse_mission_logger import (
            ModelCallKind,
            ModelProposalLogRecord,
        )

        seen = set(self.seen_obstacle_revision_keys)
        records = tuple(coordinator.records)
        first_index_by_round: dict[int, int] = {}
        for item in records:
            round_index = int(getattr(item, "round_index", 0))
            first_index_by_round.setdefault(round_index, item.proposal_index)
        for record in records:
            round_index = int(getattr(record, "round_index", 0))
            key = (
                f"{round_index}:{record.request_id}:{record.proposal_index}:"
                f"{record.submitted_timestamp_s}"
            )
            if key in seen:
                continue
            proposal = (
                {"error_code": record.error_code or record.outcome}
                if record.proposal is None
                else dict(record.proposal)
            )
            critique = (
                None if record.critique is None else dict(record.critique)
            )
            shadow_strict_critique = getattr(
                record,
                "shadow_strict_critique",
                None,
            )
            if shadow_strict_critique is not None:
                raw_shadow = dict(shadow_strict_critique)
                raw_violations = raw_shadow.get("violations", ())
                violation_types = sorted(
                    {
                        str(item.get("type"))
                        for item in raw_violations
                        if isinstance(item, dict) and item.get("type") is not None
                    }
                )
                shadow_strict_critique = {
                    "status": raw_shadow.get("status"),
                    "route_id": raw_shadow.get("route_id"),
                    "violation_count": (
                        len(raw_violations)
                        if isinstance(raw_violations, (tuple, list))
                        else 0
                    ),
                    "violation_types": violation_types[:16],
                    "route_length_m": raw_shadow.get("route_length_m"),
                    "minimum_clearance_m": raw_shadow.get(
                        "minimum_clearance_m"
                    ),
                }
            raw_version = proposal.get("new_plan_version", plan_version)
            proposal_version = (
                raw_version
                if isinstance(raw_version, int) and not isinstance(raw_version, bool)
                else plan_version
            )
            route_data = proposal.get("route_draft")
            route_id = (
                route_data.get("route_id")
                if isinstance(route_data, dict)
                and isinstance(route_data.get("route_id"), str)
                else None
            )
            if self.manifest_context.get("route_planner_backend") == "classical":
                call_kind = ModelCallKind.CLASSICAL_ROUTE_PLANNER
            else:
                call_kind = (
                    ModelCallKind.ROUTE_PLANNER
                    if record.proposal_index == first_index_by_round[round_index]
                    else ModelCallKind.ROUTE_REPAIR
                )
            completed = record.completed_timestamp_s
            latency_s = (
                None
                if completed is None
                else max(0.0, completed - record.submitted_timestamp_s)
            )
            print(
                "[ObstacleRouteRevision] "
                f"mission_id={mission_id} uav_id={uav_id} "
                f"plan_version={proposal_version} route_id={route_id or 'none'} "
                f"proposal_index={record.proposal_index} outcome={record.outcome}"
            )
            if self.logger is not None:
                try:
                    self.logger.log_model_proposal(
                        ModelProposalLogRecord(
                            proposal_id=generate_routing_id("proposal_route"),
                            mission_id=mission_id,
                            uav_id=uav_id,
                            plan_version=proposal_version,
                            timestamp_s=(
                                record.submitted_timestamp_s
                                if completed is None
                                else completed
                            ),
                            call_kind=call_kind,
                            proposal_index=record.proposal_index,
                            proposal=proposal,
                            critique=critique,
                            shadow_strict_critique=shadow_strict_critique,
                            route_id=route_id,
                            final_proposal=record.outcome == "ACCEPTED",
                            latency_s=latency_s,
                        )
                    )
                    if record.outcome == "ACCEPTED":
                        self.logger.record_plan_revision()
                    if critique is not None:
                        invalid = sum(
                            1
                            for violation in critique.get("violations", ())
                            if isinstance(violation, dict)
                            and violation.get("type")
                            in {
                                "OUTSIDE_SCENE",
                                "ALTITUDE_OUT_OF_BOUNDS",
                                "TOO_MANY_WAYPOINTS",
                            }
                        )
                        if invalid:
                            self.logger.record_invalid_waypoints(invalid)
                except Exception as exc:
                    self._disable_failed_logger(exc)
            seen.add(key)
        if self.logger is not None and records:
            try:
                state = getattr(coordinator.snapshot().state, "value", "")
                terminal = records[-1]
                shadow = getattr(terminal, "shadow_strict_critique", None)
                if state in {"ACCEPTED", "EXHAUSTED"} and isinstance(
                    shadow, dict
                ):
                    status = shadow.get("status")
                    if status in {"ACCEPT", "REVISE"}:
                        self.logger.record_shadow_strict_route_validity(
                            status == "ACCEPT"
                        )
                elif state == "EXHAUSTED":
                    # Repeated schema/contract-invalid outputs do not produce
                    # evaluable geometry, but are still invalid route plans.
                    self.logger.record_shadow_strict_route_validity(False)
            except Exception as exc:
                self._disable_failed_logger(exc)
        self.seen_obstacle_revision_keys = frozenset(seen)

    def add_worker(self, worker: object) -> None:
        if not callable(getattr(worker, "close", None)):
            raise TypeError("worker must provide close")
        self.extra_workers.append(worker)

    def set_terminal_manifest(
        self,
        *,
        agent_status: str,
        task_status: str,
        plan_version: int,
        guard_error: str | None,
        mission_success: bool | None = None,
    ) -> None:
        if mission_success is None:
            mission_success = (
                agent_status == "SUCCEEDED"
                and task_status == "SUCCEEDED"
                and guard_error is None
            )
        if not isinstance(mission_success, bool):
            raise TypeError("mission_success must be a boolean or None")
        self.terminal_manifest = {
            "agent_status": str(agent_status),
            "task_status": str(task_status),
            "final_plan_version": int(plan_version),
            "guard_error": guard_error,
            "mission_success": mission_success,
        }
        if self.logger is not None:
            try:
                self.logger.set_final_plan_version(int(plan_version))
            except Exception as exc:
                self._disable_failed_logger(exc)
        self._write_manifest()

    def record_target_perception_metrics(self, metrics: object) -> None:
        """Persist one bounded image-free row and mirror it in the manifest."""

        converter = getattr(metrics, "to_dict", None)
        if not callable(converter):
            raise TypeError("target perception metrics must provide to_dict()")
        raw = converter()
        if not isinstance(raw, dict) or not raw:
            raise TypeError("target perception metrics must be a non-empty mapping")
        payload = {
            str(key): _json_compatible(value)
            for key, value in raw.items()
        }
        self.terminal_manifest["target_perception_metrics"] = payload
        run_dir = self.run_dir
        if run_dir is not None:
            temporary = run_dir / ".target_perception_metrics.csv.tmp"
            destination = run_dir / "target_perception_metrics.csv"
            try:
                with temporary.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=tuple(payload))
                    writer.writeheader()
                    writer.writerow(
                        {
                            key: "" if value is None else value
                            for key, value in payload.items()
                        }
                    )
                os.replace(temporary, destination)
            except Exception:
                try:
                    temporary.unlink(missing_ok=True)
                except Exception:
                    pass
                raise
        self._write_manifest()

    def bind_target_debug_image_writer(self, writer: object) -> None:
        """Expose bounded writer accounting to ``run_manifest.json``."""

        stats = getattr(writer, "stats", None)
        if stats is None:
            raise TypeError("target debug image writer must provide stats")
        self.target_debug_image_writer = writer
        self._write_manifest()

    def record_initial_planner_calls(self, count: int) -> None:
        if self.logger is None or count <= 0:
            return
        try:
            self.logger.record_initial_planner_model_call(count=count)
        except Exception as exc:
            self._disable_failed_logger(exc)

    def log_initial_planner_proposals(
        self,
        planner: object,
        compiled_mission: object,
        *,
        timestamp_s: float,
        fallback_call_count: int,
    ) -> None:
        """Persist raw structured drafts and the exact compiled execution plan."""

        if self.logger is None:
            return
        proposals = tuple(getattr(planner, "model_proposals", ()))
        if not proposals:
            self.record_initial_planner_calls(fallback_call_count)
            return
        from experiments.sparse_mission_logger import (
            ModelCallKind,
            ModelProposalLogRecord,
        )

        plan = compiled_mission.task_plan
        for index, raw_record in enumerate(proposals):
            final = bool(raw_record.get("accepted", False)) and index == len(proposals) - 1
            payload: dict[str, object] = {
                "raw_model_proposal": raw_record,
            }
            if final:
                payload["final_executed_task_plan"] = plan.to_dict()
            try:
                self.logger.log_model_proposal(
                    ModelProposalLogRecord(
                        proposal_id=generate_routing_id("proposal_initial"),
                        mission_id=plan.mission_id,
                        uav_id=plan.uav_id,
                        plan_version=plan.plan_version,
                        timestamp_s=float(timestamp_s),
                        call_kind=ModelCallKind.INITIAL_PLANNER,
                        proposal_index=index,
                        proposal=payload,
                        final_proposal=final,
                    )
                )
            except Exception as exc:
                self._disable_failed_logger(exc)
                return

    def record_obstacle_hold_metrics(self, snapshot: object) -> None:
        if self.logger is None:
            return
        requested = getattr(snapshot, "hold_requested_timestamp_s", None)
        established = getattr(snapshot, "hold_established_timestamp_s", None)
        sources = tuple(getattr(snapshot, "hold_trigger_sources", ()))
        if requested is None or not sources:
            return
        try:
            self.logger.record_hold_metrics(
                trigger_source="+".join(str(item) for item in sources),
                hazard_detection_latency_s=0.0,
                hold_establishment_latency_s=(
                    None
                    if established is None
                    else max(0.0, float(established) - float(requested))
                ),
            )
        except Exception as exc:
            self._disable_failed_logger(exc)

    def record_route_collision(self) -> None:
        """Increment the sparse metric without putting I/O in safety control."""

        if self.logger is None:
            return
        try:
            self.logger.record_collision()
        except Exception as exc:
            self._disable_failed_logger(exc)
            return

    def record_path_position(self, position_xyz_m: tuple[float, float, float]) -> None:
        if self.logger is None:
            return
        try:
            self.logger.record_path_position(position_xyz_m)
        except Exception as exc:
            self._disable_failed_logger(exc)

    def _write_manifest(self) -> None:
        run_dir = self.run_dir
        if run_dir is None:
            return
        debug_images_count = 0
        debug_images_bytes = 0
        writer = self.target_debug_image_writer
        if writer is not None:
            writer_stats = getattr(writer, "stats", None)
            raw_count = getattr(writer_stats, "count", 0)
            raw_bytes = getattr(writer_stats, "bytes", 0)
            if (
                isinstance(raw_count, int)
                and not isinstance(raw_count, bool)
                and raw_count >= 0
            ):
                debug_images_count = raw_count
            if (
                isinstance(raw_bytes, int)
                and not isinstance(raw_bytes, bool)
                and raw_bytes >= 0
            ):
                debug_images_bytes = raw_bytes
        stats: dict[str, object]
        if self.logger is None:
            stats = {
                "qwen_visual_reviews": {
                    "count": 0,
                    "accepted": 0,
                    "stale": 0,
                    "timeout": 0,
                },
                "plan_revisions": 0,
                "next_best_view_model_calls": 0,
                "shadow_strict_route_valid": None,
                "route_validity_source": "shadow_strict_route_valid",
                "supervisory_hover": {"count": 0, "total_time_s": 0.0},
                "debug_images": {
                    "count": debug_images_count,
                    "bytes": debug_images_bytes,
                },
                "dropped_log_records": 0,
            }
        else:
            try:
                stats = self.logger.snapshot().to_manifest_dict(
                    debug_images_count=debug_images_count,
                    debug_images_bytes=debug_images_bytes,
                )
            except Exception:
                return
        payload = {
            "schema_version": 2,
            **self.manifest_context,
            **stats,
            **self.terminal_manifest,
        }
        temporary = run_dir / ".run_manifest.json.tmp"
        destination = run_dir / "run_manifest.json"
        try:
            temporary.write_text(
                json.dumps(
                    _json_compatible(payload),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass
            print(
                f"[Logging] run manifest disabled: {type(exc).__name__}",
                file=sys.stderr,
            )

    def close(self, *, timeout_s: float) -> None:
        close_error: Exception | None = None
        for worker in (self.worker, *tuple(self.extra_workers)):
            if worker is None:
                continue
            try:
                worker.close(timeout_s=timeout_s)
            except Exception as exc:  # pragma: no cover - integration shutdown path
                if close_error is None:
                    close_error = exc
        try:
            if self.logger is not None:
                self._write_manifest()
                self.logger.close()
        finally:
            if close_error is not None:
                raise close_error


def _create_visual_runtime(
    *,
    args: argparse.Namespace,
    config: object,
    manager: object,
    target_manager: object,
    candidate_bank: object,
    frame_store: object,
    review_mode: str,
    await_revision_completion: bool,
) -> _VisualRuntime:
    from agents.visual_review_coordinator import VisualReviewCoordinator
    from models import AsyncModelWorker, OpenAICompatibleClient
    from perception import QwenVLMVerifier, VisualReviewGate
    from runtime import MissionEventBus, ReviewScheduler

    visual_client = OpenAICompatibleClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_s=config.model_worker.request_timeout_s,
        max_images_per_request=config.qwen_visual_review.max_recent_frames,
    )
    worker = AsyncModelWorker(visual_client, uav_id=args.uav_id)
    scheduler = ReviewScheduler(
        intervals_s={
            "GOTO": config.qwen_visual_review.goto_interval_s,
            "SEARCH": config.qwen_visual_review.search_interval_s,
            "INSPECT": config.qwen_visual_review.inspect_interval_s,
            "TRACK": config.qwen_visual_review.track_interval_s,
        }
    )
    verifier = QwenVLMVerifier(
        max_image_side_px=config.qwen_visual_review.max_image_side_px,
        jpeg_quality=config.qwen_visual_review.jpeg_quality,
    )
    event_bus = MissionEventBus(max_events=256)
    coordinator = VisualReviewCoordinator(
        uav_id=args.uav_id,
        scheduler=scheduler,
        frame_store=frame_store,
        worker=worker,
        verifier=verifier,
        gate=VisualReviewGate(mode=review_mode, min_consistent_matches=2),
        skill_manager=manager,
        target_manager=target_manager,
        candidate_bank=candidate_bank,
        require_stable_search_candidate=True,
        min_search_candidate_observations=(
            config.target_perception.tracker.min_track_observations
        ),
        min_search_candidate_duration_s=(
            config.target_perception.tracker.min_track_duration_s
        ),
        event_bus=event_bus,
        max_recent_frames=config.qwen_visual_review.max_recent_frames,
        review_timeout_s=config.qwen_visual_review.blocking_hover_timeout_s,
        hover_position_tolerance_m=(
            config.qwen_visual_review.hover_position_tolerance_m
        ),
        hover_max_correction_speed_mps=(
            config.qwen_visual_review.hover_max_correction_speed_mps
        ),
        blocking_timeout_fallback=(
            config.qwen_visual_review.blocking_timeout_fallback
        ),
        max_result_age_s=config.frame_store.max_age_s,
        await_revision_completion=await_revision_completion,
        debug_model_responses=args.debug_model_responses,
    )
    return _VisualRuntime(
        coordinator=coordinator,
        worker=worker,
        event_bus=event_bus,
        output_parent=Path(args.output_dir),
        model_name=visual_client.model,
    )


def _create_logging_runtime(*, args: argparse.Namespace) -> _VisualRuntime:
    """Create image-free experiment logging without starting any Qwen worker."""

    from runtime import MissionEventBus

    uses_model = (
        getattr(args, "planner", None) in {"llm", "dynamic_llm"}
        or getattr(args, "route_planner_backend", None) == "qwen"
        or bool(getattr(args, "enable_qwen_next_best_view", False))
    )
    configured_model = getattr(args, "model", None)
    model_name = (
        configured_model.strip()
        if uses_model
        and isinstance(configured_model, str)
        and configured_model.strip()
        else "none"
    )

    return _VisualRuntime(
        coordinator=None,
        worker=None,
        event_bus=MissionEventBus(max_events=256),
        output_parent=Path(args.output_dir),
        model_name=model_name,
    )


def _run_until_terminal(
    *,
    simulation_app: object,
    debug_visualizer: object | None,
    environment: object,
    perception: object,
    agent: object,
    manager: object,
    clock: object,
    task_start_s: float,
    max_sim_time_s: float,
    shutdown_guard_s: float,
    debug_ground_truth: bool,
    injections: tuple[TestInjectionSpec, ...],
    visual_runtime: _VisualRuntime | None,
    obstacle_runtime: object | None = None,
    obstacle_route_runtime: object | None = None,
    route_collision_monitor: object | None = None,
    target_perception_coordinator: object | None = None,
    target_manager: object | None = None,
    production_target_perception: bool = False,
    target_estimate_evaluator: object | None = None,
) -> object:
    """Advance on fresh Camera samples; all HTTP work stays in the worker."""

    from scripts.run_llm_oracle_pipeline import _LoopResult, _print_ground_truth

    snapshot = agent.snapshot()
    fired: set[str] = set()
    cancel_requested = False
    landing_deadline_s: float | None = None
    next_debug_time_s = task_start_s
    obstacle_route_failed = False
    route_collision_failed = False
    route_collision_terminal = False
    target_perception_failed = False

    while simulation_app.is_running() and snapshot.status.value == "RUNNING":
        now = clock.now()
        if snapshot.active_skill == "LAND" and landing_deadline_s is None:
            landing_deadline_s = now + shutdown_guard_s
        if not cancel_requested and now - task_start_s > max_sim_time_s:
            print(
                "[MissionAgent] external max-sim-time reached; requesting fail-safe LAND",
                file=sys.stderr,
            )
            cancel_requested = True
            snapshot = agent.cancel()
            landing_deadline_s = now + shutdown_guard_s
        if landing_deadline_s is not None and now > landing_deadline_s:
            return _LoopResult(
                snapshot,
                "fail-safe LAND exceeded its separate shutdown guard",
            )

        # environment.step() is the fresh-camera gate.  No coordinator/Agent
        # call occurs on rendering ticks that did not produce a new RGB sample.
        if not environment.step():
            continue
        if production_target_perception:
            base_observation = environment.get_skill_observation(
                include_oracle=False
            )
            estimate = None
            if target_perception_coordinator is not None:
                if not target_perception_failed:
                    if target_manager is None:
                        raise RuntimeError(
                            "production target coordinator requires TargetManager"
                        )
                    target_spec = target_manager.target_spec
                    try:
                        if target_spec is not None:
                            target_perception_coordinator.submit_frame(
                                camera_sample=environment.get_camera_sample(),
                                target_spec=target_spec,
                            )
                        estimate = target_perception_coordinator.poll(
                            now_s=float(base_observation.timestamp),
                            target_manager=target_manager,
                        )
                    except Exception as exc:
                        # Import locally to keep the script's pre-Isaac startup
                        # boundary pure Python. Unsupported closed-set targets
                        # never become ordinary "not found" results.
                        from perception.target_perception_coordinator import (
                            TargetPerceptionError,
                        )

                        if not isinstance(exc, TargetPerceptionError):
                            raise
                        target_perception_failed = True
                        print(
                            "[TargetPerception] failed closed: "
                            f"{type(exc).__name__}; action=CANCEL_AND_LAND",
                            file=sys.stderr,
                        )
                        if not cancel_requested:
                            cancel_requested = True
                            snapshot = agent.cancel()
                            landing_deadline_s = now + shutdown_guard_s
                observation = perception.attach_target_estimate(
                    base_observation,
                    estimate,
                )
            else:
                observation = perception.observe(base_observation)
            frame = (
                environment.get_evaluator_frame()
                if debug_ground_truth
                else None
            )
        else:
            frame = environment.get_evaluator_frame()
            observation = perception.observe(frame)
        if target_estimate_evaluator is not None:
            if frame is None:
                raise RuntimeError(
                    "target evaluator requires an explicit synchronized evaluator frame"
                )
            evaluate = getattr(target_estimate_evaluator, "evaluate", None)
            if not callable(evaluate):
                raise TypeError("target_estimate_evaluator must provide evaluate()")
            evaluation_result = evaluate(
                getattr(observation, "target_estimate", None),
                _target_ground_truth_from_evaluator_frame(frame),
            )
            if evaluation_result is not None:
                raise RuntimeError(
                    "target evaluator must be a write-only side channel returning None"
                )
        if visual_runtime is not None:
            visual_runtime.record_path_position(
                (
                    float(observation.uav_pose.x),
                    float(observation.uav_pose.y),
                    float(observation.uav_pose.z),
                )
            )
        if (
            debug_ground_truth
            and frame is not None
            and observation.timestamp >= next_debug_time_s
        ):
            _print_ground_truth(frame, observation.timestamp)
            next_debug_time_s = observation.timestamp + 1.0

        if (
            route_collision_monitor is not None
            and not route_collision_failed
            and not route_collision_terminal
        ):
            try:
                collision = route_collision_monitor.observe(
                    (
                        float(observation.uav_pose.x),
                        float(observation.uav_pose.y),
                        float(observation.uav_pose.z),
                    ),
                    timestamp_s=float(observation.timestamp),
                )
                if collision is not None:
                    route_collision_terminal = True
            except Exception as exc:
                # Registry/geometry/callback inconsistencies are trusted
                # runtime failures.  Never keep flying a route after the
                # collision closure itself becomes unavailable.
                route_collision_failed = True
                route_collision_terminal = True
                print(
                    "[RouteCollision] monitor failed closed: "
                    f"{type(exc).__name__}",
                    file=sys.stderr,
                )
                try:
                    manager.cancel_task()
                except Exception:
                    pass

        elapsed_s = max(0.0, clock.now() - task_start_s)
        flight_corridor = _active_flight_corridor(manager, observation)
        camera_geometry = None
        if obstacle_runtime is not None and not route_collision_terminal:
            if obstacle_route_runtime is not None:
                obstacle_route_runtime.observe_active_corridor(
                    flight_corridor,
                    collision_state=obstacle_runtime.state,
                )
            current = agent.snapshot()
            camera_geometry = _camera_geometry_from_observation(
                observation,
                uav_id=current.uav_id,
                camera_config=environment.config.camera,
                far_clip_m=environment.config.obstacle_perception.max_distance_m,
            )
            if camera_geometry is not None:
                assert current.mission_id is not None
                assert current.plan_version is not None
                obstacle_runtime.process_camera_frame(
                    camera_geometry,
                    mission_id=current.mission_id,
                    uav_id=current.uav_id,
                    plan_version=current.plan_version,
                    active_corridor=flight_corridor,
                    uav_velocity_world_mps=tuple(
                        float(item) for item in observation.uav_velocity
                    ),
                )
            runtime_assessment = (
                None
                if visual_runtime is None
                else visual_runtime.runtime_visual_assessment_coordinator
            )
            if runtime_assessment is not None and camera_geometry is not None:
                try:
                    runtime_assessment.tick(
                        manager=manager,
                        obstacle_runtime=obstacle_runtime,
                        rgb=observation.camera_rgb,
                        frame_id=camera_geometry.frame_id,
                        timestamp_s=observation.timestamp,
                        mission_elapsed_s=elapsed_s,
                        obstacle_observation=(
                            obstacle_runtime.latest_observation
                        ),
                        safety_state=obstacle_runtime.state,
                        uav_speed_mps=float(
                            sum(
                                float(item) ** 2
                                for item in observation.uav_velocity
                            )
                            ** 0.5
                        ),
                    )
                except Exception as exc:
                    # V2 periodic assessment is auxiliary; trusted low-level
                    # collision supervision remains active if its worker or
                    # schema fails.
                    print(
                        "[RuntimeVisualAssessmentV2] disabled: "
                        f"{type(exc).__name__}",
                        file=sys.stderr,
                    )
                    visual_runtime.runtime_visual_assessment_coordinator = None
        if visual_runtime is not None and not route_collision_terminal:
            for spec in injections:
                if spec.flag_name in fired or elapsed_s < spec.trigger_at_s:
                    continue
                current = agent.snapshot()
                assert current.mission_id is not None
                assert current.plan_version is not None
                event = make_test_injection_event(
                    spec,
                    mission_id=current.mission_id,
                    uav_id=current.uav_id,
                    plan_version=current.plan_version,
                    timestamp_s=observation.timestamp,
                )
                agent.submit_review_event(event)
                step_id = manager.active_planned_step_id or "none"
                visual_runtime.log_injection(
                    event,
                    skill=current.active_skill or "NONE",
                    step_id=step_id,
                )
                fired.add(spec.flag_name)

        # MissionAgent feeds the same fresh frame to the non-blocking visual
        # coordinator only after hard Safety permits it, then advances Manager.
        snapshot = agent.tick(observation)
        if (
            route_collision_monitor is not None
            and not route_collision_failed
            and not route_collision_terminal
        ):
            try:
                # Seed a FOLLOW_ROUTE that became active during this Agent
                # tick, so its first movement interval is swept next frame.
                collision = route_collision_monitor.observe(
                    (
                        float(observation.uav_pose.x),
                        float(observation.uav_pose.y),
                        float(observation.uav_pose.z),
                    ),
                    timestamp_s=float(observation.timestamp),
                )
                if collision is not None:
                    route_collision_terminal = True
            except Exception as exc:
                route_collision_failed = True
                route_collision_terminal = True
                print(
                    "[RouteCollision] monitor failed closed: "
                    f"{type(exc).__name__}",
                    file=sys.stderr,
                )
                try:
                    manager.cancel_task()
                except Exception:
                    pass
        manager_plan = manager.task_plan
        if (
            not route_collision_terminal
            and manager_plan is not None
            and snapshot.plan_version is not None
            and manager_plan.plan_version == snapshot.plan_version + 1
        ):
            try:
                snapshot = agent.adopt_runtime_task_plan(manager_plan)
            except Exception as exc:
                print(
                    "[ObstacleRouteRevision] Agent adoption failed closed: "
                    f"{type(exc).__name__}",
                    file=sys.stderr,
                )
                manager.cancel_task()
                snapshot = agent.snapshot()
        obstacle_snapshot = None
        if obstacle_runtime is not None and not route_collision_terminal:
            obstacle_snapshot = obstacle_runtime.observe_skill_transitions(
                manager.transition_log
            )
            if visual_runtime is not None:
                visual_runtime.record_obstacle_hold_metrics(obstacle_snapshot)
        if (
            obstacle_route_runtime is not None
            and obstacle_snapshot is not None
            and camera_geometry is not None
            and not obstacle_route_failed
        ):
            from planner.spatial_resolver import FramePose

            try:
                obstacle_route_runtime.tick(
                    obstacle_snapshot=obstacle_snapshot,
                    manager=manager,
                    rgb=observation.camera_rgb,
                    frame_id=camera_geometry.frame_id,
                    timestamp_s=observation.timestamp,
                    mission_elapsed_s=elapsed_s,
                    hold_pose=FramePose(
                        (
                            float(observation.uav_pose.x),
                            float(observation.uav_pose.y),
                            float(observation.uav_pose.z),
                        ),
                        yaw_rad=float(observation.uav_pose.yaw),
                    ),
                )
            except Exception as exc:
                # A malformed/unavailable route context is a trusted-runtime
                # failure.  Never continue the interrupted flight or expose
                # request/image contents in this boundary log.
                obstacle_route_failed = True
                print(
                    "[ObstacleRouteRevision] failed closed: "
                    f"{type(exc).__name__}",
                    file=sys.stderr,
                )
                manager.cancel_task()
        if debug_visualizer is not None:
            # 如果运行时计划修订成功，更新灰色计划路线和目标点。
            revision_coordinator = (
                None
                if visual_runtime is None
                else visual_runtime.revision_coordinator
            )
            accepted_revision = (
                None
                if revision_coordinator is None
                else revision_coordinator.latest_accepted_revision
            )
            if (
                accepted_revision is not None
                and accepted_revision.compiled_mission.task_plan.plan_version
                != debug_visualizer.plan_version
            ):
                debug_visualizer.set_plan(
                    accepted_revision.compiled_mission
                )
            active_task_plan = manager.task_plan
            if (
                active_task_plan is not None
                and active_task_plan.plan_version != debug_visualizer.plan_version
            ):
                debug_visualizer.set_plan(
                    SimpleNamespace(task_plan=active_task_plan)
                )

            debug_visualizer.set_safety_corridor(flight_corridor)
            debug_visualizer.set_hold_point(
                (
                    getattr(
                        getattr(obstacle_route_runtime, "active_hold_pose", None),
                        "xyz_m",
                        (
                            float(observation.uav_pose.x),
                            float(observation.uav_pose.y),
                            float(observation.uav_pose.z),
                        ),
                    )
                    if bool(getattr(manager, "is_supervisory_paused", False))
                    else None
                )
            )
            debug_visualizer.set_route_records(
                _route_debug_records(manager, visual_runtime)
            )
            obstacle_state = (
                None if obstacle_runtime is None else obstacle_runtime.state
            )
            hazard_active = (
                obstacle_state is not None
                and getattr(obstacle_state, "value", str(obstacle_state)) != "CLEAR"
            )
            debug_visualizer.update(
                uav_pose=observation.uav_pose,
                camera_position_m=observation.camera_position_m,
                camera_orientation_wxyz=(
                    observation.camera_orientation_wxyz
                ),
                active_skill=snapshot.active_skill,
                active_step_id=manager.active_planned_step_id,
                target_lifecycle=snapshot.target.lifecycle,
                hazard_active=hazard_active,
                hold_active=bool(
                    getattr(manager, "is_supervisory_paused", False)
                ),
            )
        if visual_runtime is not None:
            visual_runtime.emit_new_reviews(
                step_id=manager.active_planned_step_id,
                skill=snapshot.active_skill or "NONE",
            )
            visual_runtime.emit_new_runtime_assessments(
                step_id=manager.active_planned_step_id,
            )
            visual_runtime.emit_new_next_best_view_proposals()
            visual_runtime.emit_new_events(
                skill=snapshot.active_skill or "NONE",
                step_id=manager.active_planned_step_id or "none",
            )
            visual_runtime.emit_new_transitions(manager)
            visual_runtime.emit_new_search_metrics(manager)
            visual_runtime.emit_new_revisions(
                skill=snapshot.active_skill or "NONE",
                step_id=manager.active_planned_step_id or "none",
            )
            if snapshot.mission_id is not None and snapshot.plan_version is not None:
                visual_runtime.emit_new_obstacle_revisions(
                    mission_id=snapshot.mission_id,
                    uav_id=snapshot.uav_id,
                    plan_version=snapshot.plan_version,
                )

    if snapshot.status.value == "RUNNING":
        return _LoopResult(
            snapshot,
            "SimulationApp stopped before the mission reached a terminal state",
        )
    return _LoopResult(snapshot)


def _best_effort_production_failure_land(
    *,
    simulation_app: object,
    environment: object,
    perception: object,
    agent: object,
    clock: object,
    shutdown_guard_s: float,
) -> object:
    """Tick trusted cancel-and-LAND after an unexpected production failure."""

    from scripts.run_llm_oracle_pipeline import _LoopResult

    snapshot = agent.snapshot()
    if snapshot.status.value != "RUNNING":
        return _LoopResult(snapshot)
    try:
        snapshot = agent.cancel()
    except Exception as exc:
        snapshot = agent.snapshot()
        if snapshot.status.value == "RUNNING" and snapshot.active_skill != "LAND":
            stop = getattr(environment.uav_controller, "stop", None)
            if callable(stop):
                stop()
            return _LoopResult(
                snapshot,
                "runtime failure could not start fail-safe LAND: "
                f"{type(exc).__name__}",
            )

    deadline_s = float(clock.now()) + float(shutdown_guard_s)
    while simulation_app.is_running() and snapshot.status.value == "RUNNING":
        if float(clock.now()) > deadline_s:
            stop = getattr(environment.uav_controller, "stop", None)
            if callable(stop):
                stop()
            return _LoopResult(
                snapshot,
                "runtime failure LAND exceeded its shutdown guard",
            )
        try:
            if not environment.step():
                continue
            base = environment.get_skill_observation(include_oracle=False)
            observation = perception.attach_target_estimate(base, None)
            snapshot = agent.tick(observation)
        except Exception as exc:
            stop = getattr(environment.uav_controller, "stop", None)
            if callable(stop):
                stop()
            return _LoopResult(
                snapshot,
                "runtime failure LAND observation/tick failed: "
                f"{type(exc).__name__}",
            )
    if snapshot.status.value == "RUNNING":
        return _LoopResult(
            snapshot,
            "SimulationApp stopped before runtime failure LAND completed",
        )
    return _LoopResult(snapshot)


def _target_ground_truth_from_evaluator_frame(frame: object) -> object:
    """Copy privileged Isaac truth into the evaluator-only pure value type."""

    from perception.evaluation import TargetGroundTruth

    try:
        timestamp_s = float(frame.observation.camera_timestamp_s)
        position = tuple(float(value) for value in frame.target_position_m)
        velocity = tuple(float(value) for value in frame.target_velocity_mps)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "target evaluator requires a synchronized environment EvaluatorFrame"
        ) from exc
    return TargetGroundTruth(
        timestamp_s=timestamp_s,
        position_world_m=position,
        velocity_world_mps=velocity,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except LaunchConfigurationError as exc:
        print(f"launch configuration error: {exc}", file=sys.stderr)
        return 2
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve():
        print(
            "error: run this standalone demo through "
            "./python.sh scripts/run_dynamic_visual_mission.py",
            file=sys.stderr,
        )
        return 2

    try:
        config = load_config(args.config)
        config = replace(config, uav=replace(config.uav, id=args.uav_id))
        effective_target_backend = (
            args.target_perception_backend
            or (
                "oracle_evaluation"
                if args.perception_runtime_profile == "oracle_evaluation"
                else config.target_perception.backend
            )
        )
        from common.loopback_url import validate_loopback_http_url

        yolo_service_config = replace(
            config.target_perception.yolo_service,
            url=validate_loopback_http_url(
                (
                    args.yolo_service_url
                    if args.yolo_service_url is not None
                    else config.target_perception.yolo_service.url
                ),
                "--yolo-service-url",
            ),
            request_timeout_s=(
                args.yolo_request_timeout_s
                if args.yolo_request_timeout_s is not None
                else config.target_perception.yolo_service.request_timeout_s
            ),
            max_result_age_s=(
                args.yolo_max_result_age_s
                if args.yolo_max_result_age_s is not None
                else config.target_perception.yolo_service.max_result_age_s
            ),
        )
        config = replace(
            config,
            target_perception=replace(
                config.target_perception,
                backend=effective_target_backend,
                yolo_service=yolo_service_config,
            ),
        )
        args.effective_target_perception_backend = effective_target_backend
        args.effective_obstacle_perception_mode = (
            args.obstacle_perception or config.obstacle_perception.mode
        )
        review_mode, injections = validate_launch_args(
            args,
            configured_review_mode=config.qwen_visual_review.mode,
        )
    except (ConfigError, LaunchConfigurationError) as exc:
        print(f"launch configuration error: {exc}", file=sys.stderr)
        return 2
    if not 0.0 <= args.start_altitude <= config.scene.size_xyz_m[2]:
        print("error: --start-altitude is outside scene Z bounds", file=sys.stderr)
        return 2
    if not 0.0 < args.takeoff_altitude <= config.scene.size_xyz_m[2]:
        print("error: --takeoff-altitude is outside scene Z bounds", file=sys.stderr)
        return 2
    if args.start_altitude > args.takeoff_altitude:
        print("error: --start-altitude exceeds --takeoff-altitude", file=sys.stderr)
        return 2

    # Pure-Python planning and policy setup precede SimulationApp.
    from models import OpenAICompatibleClient
    from planner import (
        DynamicLLMPlanner,
        LLMPlanner,
        MissionIntent,
        PlannerPolicy,
        QwenPlanRevisionPlanner,
        RevisionLimits,
        RevisionValidator,
        ScriptedDynamicPlanner,
        ScriptedPlanner,
        SkillPlanDraft,
        build_default_skill_catalog,
    )
    from runtime import PlanValidator, PlannerLimits, SafetySupervisor
    from runtime.world_context_builder import (
        WorldContextBuildError,
        build_planner_world_context,
    )
    from skills.search_strategy import SearchRuntimeCapabilities
    from scripts.run_llm_oracle_pipeline import (
        IsaacSimulationClock,
        _LoopResult,
        _CountingModelClient,
        _best_effort_interrupt_land,
        _landing_zone_name,
        _print_plan,
        _print_runtime_summary,
        _random_target_spawn,
        _shutdown_guard_s,
        _standard_dynamic_draft_data,
    )

    try:
        world_context = build_planner_world_context(
            config,
            takeoff_altitude_m=args.takeoff_altitude,
            track_duration_s=args.track_duration,
            start_altitude_m=args.start_altitude,
            land_timeout_s=max(60.0, 3.0 * config.scene.size_xyz_m[2]),
        )
    except WorldContextBuildError as exc:
        print(f"world context error: {exc}", file=sys.stderr)
        return 2

    planner_limits = PlannerLimits.from_config(config.planner)
    planner_policy = PlannerPolicy.from_config(config.planner, planner_limits)
    search_runtime_capabilities = SearchRuntimeCapabilities(
        adaptive_next_best_view=bool(args.enable_qwen_next_best_view)
    )
    counting_client: _CountingModelClient | None = None
    selected_model: str | None = None
    if args.planner == "scripted":
        planner = ScriptedPlanner(
            MissionIntent(
                target_description="moving target",
                search_region="search_area",
                track_duration_s=args.track_duration,
                landing_zone="home",
                takeoff_altitude_m=args.takeoff_altitude,
            )
        )
    elif args.planner == "dynamic_scripted":
        planner = ScriptedDynamicPlanner(
            SkillPlanDraft.from_dict(_standard_dynamic_draft_data(args))
        )
    else:
        planner_client = OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )
        selected_model = planner_client.model
        counting_client = _CountingModelClient(planner_client)
        if args.planner == "llm":
            planner = LLMPlanner(
                counting_client,
                PROJECT_ROOT / "prompts" / "mission_planner_system.txt",
            )
        else:
            planner = DynamicLLMPlanner(
                counting_client,
                (
                    PROJECT_ROOT / "prompts" / "dynamic_skill_planner_v3_system.txt"
                    if args.planning_contract == "v3"
                    else PROJECT_ROOT / "prompts" / "dynamic_skill_planner_system.txt"
                ),
                planner_limits=planner_limits,
                planner_policy=planner_policy,
                planning_contract=args.planning_contract,
                search_runtime_capabilities=search_runtime_capabilities,
            )

    shutdown_guard_s = _shutdown_guard_s(
        max(args.takeoff_altitude, config.scene.size_xyz_m[2]),
        args.start_altitude,
    )
    safety = SafetySupervisor(
        world_context.scene_min_xyz_m,
        world_context.scene_max_xyz_m,
        max_mission_time_s=args.max_sim_time + shutdown_guard_s + 1.0,
        position_margin_m=0.25,
        max_safe_altitude_m=world_context.scene_max_xyz_m[2],
        planner_limits=planner_limits,
        search_runtime_capabilities=search_runtime_capabilities,
    )

    # Standalone ordering boundary: this is the first Isaac import.  All
    # env/simple_uav_search_env and omni/pxr-backed imports remain below the
    # successfully constructed SimulationApp.
    from isaacsim import SimulationApp

    headless = (
        config.simulation.headless
        if args.headless is None
        else args.headless
    )

    if args.debug_visualization and headless:
        print(
            "error: --debug-visualization requires --no-headless",
            file=sys.stderr,
        )
        return 2

    simulation_app = SimulationApp({"headless": headless})

    if args.debug_visualization:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.util.debug_draw")
        simulation_app.update()
    environment = None
    agent = None
    manager = None
    perception = None
    clock = None
    visual_runtime: _VisualRuntime | None = None
    debug_visualizer = None
    obstacle_runtime = None
    obstacle_route_runtime = None
    route_collision_monitor = None
    initial_spatial_resolver = None
    next_best_view_worker = None
    next_best_view_provider = None
    next_best_view_worker_registered = False
    target_perception_coordinator = None
    target_estimate_evaluator = None
    target_evaluation_metrics = None
    interrupted = False
    try:
        from agents.mission_agent import MissionAgent
        from agents.plan_revision_coordinator import PlanRevisionCoordinator
        from env.simple_uav_search_env import SimpleUavSearchEnv
        from perception import (
            CandidateBank,
            OracleEvaluationCandidateResolver,
            PerceptionRuntimeProfile,
        )
        from runtime import (
            CollisionSupervisor,
            FrameStore,
            HazardFusion,
            ObstacleHazardRuntime,
            RouteCollisionMonitor,
            RouteRegistry,
        )
        from skills.inspect import InspectSkill
        from skills.manager import SkillManager, create_default_skill_registry
        from target.target_manager import TargetManager

        environment = SimpleUavSearchEnv(config)
        environment.setup()
        if not headless:
            environment.configure_overview_viewport()
        configured_start = config.uav.initial_position_xyz_m
        environment.set_uav_pose(
            (configured_start[0], configured_start[1], args.start_altitude)
        )
        initial_spatial_resolver = _build_initial_spatial_resolver(
            world_context,
            environment.uav_controller.get_pose(),
        )
        validator = PlanValidator(
            planner_limits,
            planner_policy,
            spatial_resolver=initial_spatial_resolver,
            search_runtime_capabilities=search_runtime_capabilities,
        )
        target_spawn = _random_target_spawn(config)
        environment.set_target_pose(target_spawn)
        if args.debug_ground_truth:
            print(f"[GroundTruth] target_spawn_m={target_spawn}")

        profile = (
            PerceptionRuntimeProfile.ORACLE_EVALUATION
            if args.perception_runtime_profile == "oracle_evaluation"
            else PerceptionRuntimeProfile.PRODUCTION
        )
        frame_store = FrameStore(
            max_frames=config.frame_store.max_frames,
            max_bytes=config.frame_store.max_bytes,
            max_age_s=config.frame_store.max_age_s,
        )
        candidate_bank = CandidateBank(uav_id=args.uav_id)
        from perception.factory import build_target_perception_backend

        perception = build_target_perception_backend(
            config,
            runtime_profile=profile,
            acknowledge_privileged_oracle=args.acknowledge_privileged_oracle,
            uav_id=args.uav_id,
        )
        if profile is PerceptionRuntimeProfile.ORACLE_EVALUATION:
            print(
                "[Perception] PRIVILEGED ORACLE_EVALUATION explicitly enabled; "
                "geometry_source=oracle_evaluation, never qwen_vl"
            )

            def oracle_candidate_position(
                resolved_uav_id: str,
                candidate_id: str,
                timestamp_s: float,
            ) -> object:
                # Evaluator-only label/INSPECT resolver. It is never created
                # for production and its value never enters Qwen or YOLO.
                del candidate_id, timestamp_s
                if resolved_uav_id != args.uav_id:
                    raise ValueError("candidate resolver UAV route mismatch")
                return environment.get_evaluator_frame().target_position_m

            candidate_resolver = OracleEvaluationCandidateResolver(
                oracle_candidate_position,
                profile=profile,
                acknowledge_privileged_oracle=True,
            )
        elif config.target_perception.backend == "ultralytics_service":
            from perception.depth_geometry import DepthCandidateResolver
            from perception.target_perception_coordinator import (
                TargetPerceptionCoordinator,
            )

            candidate_resolver = DepthCandidateResolver(
                frame_store,
                sampling_strategy=config.target_perception.geometry.depth_anchor,
                patch_radius_px=(
                    config.target_perception.geometry.depth_patch_radius_px
                ),
                min_depth_m=config.target_perception.geometry.min_depth_m,
                max_depth_m=config.target_perception.geometry.max_depth_m,
            )
            target_perception_coordinator = TargetPerceptionCoordinator(
                config.target_perception,
                frame_store=frame_store,
                candidate_bank=candidate_bank,
                resolver=candidate_resolver,
            )
            print(
                "[Perception] production ultralytics_service enabled; "
                "geometry_source=isaac_depth, oracle_fallback=forbidden"
            )
        else:
            from perception.grounding import ProductionCandidateResolver

            candidate_resolver = ProductionCandidateResolver()
            print("[Perception] production target perception disabled")

        if (
            args.debug_ground_truth
            or profile is PerceptionRuntimeProfile.ORACLE_EVALUATION
        ):
            from perception.evaluation import (
                TargetEvaluationMode,
                TargetEstimateEvaluator,
            )
            from perception.target_perception_coordinator import (
                TargetPerceptionMetrics,
            )

            target_evaluation_metrics = (
                target_perception_coordinator.metrics
                if target_perception_coordinator is not None
                else TargetPerceptionMetrics()
            )
            allowed_sources = (
                frozenset({"oracle_evaluation"})
                if profile is PerceptionRuntimeProfile.ORACLE_EVALUATION
                else frozenset(
                    {
                        "yolo26_botsort",
                        "yoloe26_botsort",
                        "kalman_prediction",
                    }
                )
            )
            target_estimate_evaluator = TargetEstimateEvaluator(
                target_evaluation_metrics,
                mode=TargetEvaluationMode.ORACLE_GROUND_TRUTH,
                allowed_estimate_sources=allowed_sources,
            )
            print(
                "[TargetEvaluator] privileged RMSE side-channel enabled; "
                "ground_truth_to_control=false"
            )

        clock = IsaacSimulationClock(environment)
        context = environment.make_skill_context(clock, perception=perception)
        inspect_skill = InspectSkill(
            candidate_bank=candidate_bank,
            candidate_resolver=candidate_resolver,
            frame_store=frame_store,
        )
        route_registry = RouteRegistry()
        if args.enable_qwen_next_best_view:
            from models import AsyncModelWorker
            from planner.qwen_next_best_view import (
                NextBestViewRouting,
                QwenNextBestViewProvider,
            )

            next_best_view_client = OpenAICompatibleClient(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                timeout_s=config.model_worker.request_timeout_s,
                max_images_per_request=1,
            )
            next_best_view_worker = AsyncModelWorker(
                next_best_view_client,
                uav_id=args.uav_id,
            )

            def current_next_best_view_routing() -> NextBestViewRouting:
                if manager is None or manager.task_plan is None:
                    raise RuntimeError(
                        "next-best-view routing is unavailable before task start"
                    )
                return NextBestViewRouting(
                    manager.task_plan.mission_id,
                    manager.task_plan.plan_version,
                )

            next_best_view_provider = QwenNextBestViewProvider(
                worker=next_best_view_worker,
                uav_id=args.uav_id,
                routing_context=current_next_best_view_routing,
                max_image_side_px=(
                    config.qwen_visual_review.max_image_side_px
                ),
                jpeg_quality=config.qwen_visual_review.jpeg_quality,
            )
        manager = SkillManager(
            context,
            registry=create_default_skill_registry(
                transit_yaw_mode=config.search.transit_yaw_mode,
                inspect_skill=inspect_skill,
                next_best_view_provider=next_best_view_provider,
            ),
            route_registry=route_registry,
        )
        target_manager = TargetManager()
        revision_enabled = bool(
            args.enable_qwen_vision
            and review_mode == "gate"
            and config.plan_revision.enabled
            and args.planner in {"dynamic_scripted", "dynamic_llm"}
        )
        if args.enable_qwen_vision:
            visual_runtime = _create_visual_runtime(
                args=args,
                config=config,
                manager=manager,
                target_manager=target_manager,
                candidate_bank=candidate_bank,
                frame_store=frame_store,
                review_mode=review_mode,
                await_revision_completion=revision_enabled,
            )
        else:
            # Metrics, transitions and manifests are experiment outputs, not
            # a side effect of enabling the periodic Qwen vision reviewer.
            visual_runtime = _create_logging_runtime(args=args)
        if (
            target_perception_coordinator is not None
            and visual_runtime.coordinator is not None
        ):
            review_provider = getattr(
                visual_runtime.coordinator,
                "latest_candidate_review",
                None,
            )
            reference_provider = getattr(
                visual_runtime.coordinator,
                "latest_candidate_review_reference_handles",
                None,
            )
            if not callable(review_provider):
                raise RuntimeError(
                    "visual review coordinator lacks typed candidate evidence bridge"
                )
            if not callable(reference_provider):
                raise RuntimeError(
                    "visual review coordinator lacks identity-reference bridge"
                )
            target_perception_coordinator.bind_visual_review_provider(
                review_provider,
                reference_provider,
            )
            print(
                "[Perception] accepted routed Qwen candidate reviews are "
                "available as typed semantic evidence"
            )
        if next_best_view_worker is not None:
            assert visual_runtime is not None
            visual_runtime.add_worker(next_best_view_worker)
            next_best_view_worker_registered = True
            visual_runtime.next_best_view_provider = next_best_view_provider
            visual_runtime.model_name = next_best_view_client.model
        if args.effective_obstacle_perception_mode == "ideal_camera":
            from perception import IdealObstaclePerception

            assert environment.scene is not None
            ideal_obstacles = IdealObstaclePerception.from_config(
                environment.scene.obstacle_registry,
                config.obstacle_perception,
            )
            collision_supervisor = CollisionSupervisor()
            obstacle_runtime = ObstacleHazardRuntime(
                perception=ideal_obstacles,
                hazard_fusion=HazardFusion(),
                collision_supervisor=collision_supervisor,
                skill_manager=manager,
                event_sink=(
                    None
                    if visual_runtime is None
                    else visual_runtime.event_bus.publish
                ),
                hover_timeout_s=config.qwen_visual_review.blocking_hover_timeout_s,
                hover_position_tolerance_m=(
                    config.qwen_visual_review.hover_position_tolerance_m
                ),
                hover_max_correction_speed_mps=(
                    config.qwen_visual_review.hover_max_correction_speed_mps
                ),
                hover_timeout_fallback=(
                    config.qwen_visual_review.blocking_timeout_fallback
                ),
            )
            if (
                visual_runtime is not None
                and args.route_planner_backend == "qwen"
            ):
                from agents.obstacle_revision_coordinator import (
                    ObstacleRevisionCoordinator,
                )
                from models import AsyncModelWorker
                from planner.obstacle_revision import ObstacleAwareRevisionPlanner

                route_client = OpenAICompatibleClient(
                    base_url=args.base_url,
                    model=args.model,
                    api_key=args.api_key,
                    timeout_s=config.model_worker.request_timeout_s,
                    max_images_per_request=3,
                )
                route_worker = AsyncModelWorker(
                    route_client,
                    uav_id=args.uav_id,
                )
                visual_runtime.add_worker(route_worker)
                obstacle_revision_coordinator = ObstacleRevisionCoordinator(
                    uav_id=args.uav_id,
                    planner=ObstacleAwareRevisionPlanner(
                        max_image_side_px=(
                            config.qwen_visual_review.max_image_side_px
                        ),
                        jpeg_quality=config.qwen_visual_review.jpeg_quality,
                    ),
                    worker=route_worker,
                    route_registry=route_registry,
                    collision_supervisor=collision_supervisor,
                    skill_manager=manager,
                    safety_preflight=safety.preflight,
                    route_validation_mode=args.route_validation_mode,
                    max_proposals=3,
                    event_sink=visual_runtime.event_bus.publish,
                )
                visual_runtime.obstacle_revision_coordinator = (
                    obstacle_revision_coordinator
                )
            elif (
                visual_runtime is not None
                and args.route_planner_backend == "classical"
            ):
                from agents.classical_obstacle_revision_coordinator import (
                    ClassicalObstacleRevisionCoordinator,
                )
                from planner.classical_route_planner import ClassicalRoutePlanner

                visual_runtime.obstacle_revision_coordinator = (
                    ClassicalObstacleRevisionCoordinator(
                        uav_id=args.uav_id,
                        planner=ClassicalRoutePlanner(),
                        route_registry=route_registry,
                        collision_supervisor=collision_supervisor,
                        skill_manager=manager,
                        safety_preflight=safety.preflight,
                        event_sink=visual_runtime.event_bus.publish,
                    )
                )

        agent_kwargs: dict[str, object] = {}
        if visual_runtime is not None and visual_runtime.coordinator is not None:
            agent_kwargs["visual_review_coordinator"] = visual_runtime.coordinator
        if revision_enabled:
            assert visual_runtime is not None
            revision_limits = RevisionLimits(
                max_plan_revisions=config.plan_revision.max_revisions,
                cooldown_s=config.plan_revision.cooldown_s,
                max_added_steps_per_revision=3,
                max_total_plan_steps=planner_limits.max_plan_steps,
            )
            revision_validator = RevisionValidator(
                validator,
                revision_limits=revision_limits,
                safety_preflight=safety,
            )
            revision_planner = QwenPlanRevisionPlanner(
                world_context=world_context,
                skill_catalog=build_default_skill_catalog(),
                limits=planner_limits,
                policy=planner_policy,
            )
            plan_revision_coordinator = PlanRevisionCoordinator(
                uav_id=args.uav_id,
                planner=revision_planner,
                worker=visual_runtime.worker,
                validator=revision_validator,
                skill_manager=manager,
                candidate_bank=candidate_bank,
                world_context=world_context,
                safety_preflight=safety,
                original_instruction=args.instruction,
                clock=clock.now,
                request_timeout_s=config.model_worker.request_timeout_s,
            )
            visual_runtime.revision_coordinator = plan_revision_coordinator
            agent_kwargs["plan_revision_coordinator"] = plan_revision_coordinator
        agent = MissionAgent(
            planner,
            validator,
            safety,
            manager,
            target_manager,
            clock,
            perception_runtime_profile=profile,
            acknowledge_privileged_oracle=args.acknowledge_privileged_oracle,
            target_perception_backend=config.target_perception.backend,
            runtime_program=args.runtime_program,
            **agent_kwargs,
        )

        task_start_s = clock.now()
        compiled = agent.start(args.instruction, world_context)
        if target_perception_coordinator is not None:
            target_perception_coordinator.reset(
                mission_id=compiled.task_plan.mission_id,
                uav_id=args.uav_id,
            )
        assert environment.scene is not None

        def on_route_collision(collision: object) -> None:
            impact = getattr(collision, "impact_position_world_m", None)
            print(
                "[RouteCollision] "
                f"mission_id={getattr(collision, 'mission_id', 'unknown')} "
                f"uav_id={getattr(collision, 'uav_id', 'unknown')} "
                f"plan_version={getattr(collision, 'plan_version', 'unknown')} "
                f"route_id={getattr(collision, 'route_id', 'unknown')} "
                f"obstacle_id={getattr(collision, 'obstacle_id', 'unknown')} "
                f"impact_world_m={impact} action=CANCEL_AND_LAND",
                file=sys.stderr,
            )
            if visual_runtime is not None:
                visual_runtime.record_route_collision()

        if (
            obstacle_runtime is not None
            and visual_runtime is not None
            and visual_runtime.obstacle_revision_coordinator is not None
        ):
            from runtime.obstacle_route_runtime import (
                ObstacleRouteReplanRuntime,
            )

            assert environment.scene is not None
            assert initial_spatial_resolver is not None
            obstacle_route_runtime = ObstacleRouteReplanRuntime(
                coordinator=visual_runtime.obstacle_revision_coordinator,
                initial_resolver=initial_spatial_resolver,
                obstacles=environment.scene.obstacle_registry,
                scene_min_xyz_m=world_context.scene_min_xyz_m,
                scene_max_xyz_m=world_context.scene_max_xyz_m,
                original_instruction=args.instruction,
                original_plan_summary=compiled.task_plan.to_dict(),
            )
            if args.route_planner_backend == "qwen":
                from agents.runtime_visual_assessment_coordinator import (
                    RuntimeVisualAssessmentCoordinator,
                )
                from models import AsyncModelWorker
                from perception.runtime_visual_assessment import (
                    QwenRuntimeVisualVerifierV2,
                )

                assessment_client = OpenAICompatibleClient(
                    base_url=args.base_url,
                    model=args.model,
                    api_key=args.api_key,
                    timeout_s=config.model_worker.request_timeout_s,
                    max_images_per_request=3,
                )
                assessment_worker = AsyncModelWorker(
                    assessment_client,
                    uav_id=args.uav_id,
                )
                visual_runtime.add_worker(assessment_worker)
                visual_runtime.runtime_visual_assessment_coordinator = (
                    RuntimeVisualAssessmentCoordinator(
                        uav_id=args.uav_id,
                        worker=assessment_worker,
                        verifier=QwenRuntimeVisualVerifierV2(
                            max_image_side_px=(
                                config.qwen_visual_review.max_image_side_px
                            ),
                            jpeg_quality=config.qwen_visual_review.jpeg_quality,
                        ),
                        original_instruction=args.instruction,
                        target_spec=compiled.target_spec,
                        intervals_s={
                            "GOTO": config.qwen_visual_review.goto_interval_s,
                            "SEARCH": config.qwen_visual_review.search_interval_s,
                            "INSPECT": config.qwen_visual_review.inspect_interval_s,
                            "TRACK": config.qwen_visual_review.track_interval_s,
                        },
                        max_result_age_s=config.frame_store.max_age_s,
                        apply_to_control=review_mode == "gate",
                    )
                )
        if args.debug_visualization:
            from visualization import MissionDebugDraw, MissionStatusOverlay

            debug_visualizer = MissionDebugDraw(
                world_context=world_context,
                camera_config=config.camera,
                status_overlay=(
                    None if headless else MissionStatusOverlay()
                ),
            )
            debug_visualizer.set_plan(compiled)
            assert environment.scene is not None
            debug_visualizer.set_obstacles(
                environment.scene.obstacle_registry.specs
            )
        launch_snapshot = agent.snapshot()
        assert launch_snapshot.mission_id is not None
        for key, value in startup_fields(
            uav_id=args.uav_id,
            mission_id=launch_snapshot.mission_id,
            planner=args.planner,
            review_enabled=args.enable_qwen_vision,
            review_mode=review_mode,
            perception_profile=args.perception_runtime_profile,
            oracle_acknowledged=args.acknowledge_privileged_oracle,
            target_perception_backend=config.target_perception.backend,
            yolo_service_url=(
                config.target_perception.yolo_service.url
                if config.target_perception.backend == "ultralytics_service"
                else None
            ),
            model_family=config.target_perception.detector.model_family,
            geometry_mode=config.target_perception.geometry.mode,
            state_estimator=config.target_perception.state_estimator.type,
            confirmation_mode=config.target_perception.confirmation.mode,
        ):
            print(f"[Launch] {key}={value}")
        print(
            "[Launch] qwen_next_best_view="
            + (
                "enabled:async_macro_only"
                if args.enable_qwen_next_best_view
                else "disabled"
            )
        )
        _print_plan(compiled)
        if visual_runtime is not None:
            visual_runtime.begin_logging(
                launch_snapshot.mission_id,
                manifest_context=visual_manifest_context(
                    args=args,
                    config=config,
                    mission_id=launch_snapshot.mission_id,
                    review_mode=review_mode,
                    model_name=visual_runtime.model_name,
                ),
            )
            if target_perception_coordinator is not None:
                # This branch exists only for the production
                # ultralytics_service backend.  Oracle evaluation never
                # creates or binds a target-image writer.
                from perception.target_debug_images import (
                    BoundedTargetDebugImageWriter,
                )

                assert visual_runtime.run_dir is not None
                target_debug_writer = BoundedTargetDebugImageWriter(
                    visual_runtime.run_dir / "debug_images" / "target_perception",
                    enabled=config.debug_images.enabled,
                    max_images_per_run=(
                        config.debug_images.max_images_per_run
                    ),
                )
                target_perception_coordinator.bind_debug_image_writer(
                    target_debug_writer
                )
                visual_runtime.bind_target_debug_image_writer(
                    target_debug_writer
                )
            visual_runtime.log_initial_planner_proposals(
                planner,
                compiled,
                timestamp_s=task_start_s,
                fallback_call_count=(
                    0 if counting_client is None else counting_client.chat_calls
                ),
            )
            visual_runtime.emit_new_transitions(manager)

        # Construct this for every planner/perception mode. In particular,
        # open_sim deliberately executes unchecked model geometry and relies
        # on this runtime closure to measure and terminate real collisions.
        route_collision_monitor = RouteCollisionMonitor(
            obstacle_registry=environment.scene.obstacle_registry,
            route_registry=route_registry,
            skill_manager=manager,
            uav_radius_m=_DEFAULT_UAV_SAFETY_RADIUS_M,
            event_sink=(
                None
                if visual_runtime is None
                else visual_runtime.event_bus.publish
            ),
            collision_sink=on_route_collision,
        )
        initial_pose = environment.uav_controller.get_pose()
        route_collision_monitor.observe(
            (initial_pose.x, initial_pose.y, initial_pose.z),
            timestamp_s=task_start_s,
        )

        try:
            loop_result = _run_until_terminal(
                simulation_app=simulation_app,
                debug_visualizer=debug_visualizer,
                environment=environment,
                perception=perception,
                agent=agent,
                manager=manager,
                clock=clock,
                task_start_s=task_start_s,
                max_sim_time_s=args.max_sim_time,
                shutdown_guard_s=shutdown_guard_s,
                debug_ground_truth=args.debug_ground_truth,
                injections=injections,
                visual_runtime=visual_runtime,
                obstacle_runtime=obstacle_runtime,
                obstacle_route_runtime=obstacle_route_runtime,
                route_collision_monitor=route_collision_monitor,
                target_perception_coordinator=target_perception_coordinator,
                target_manager=target_manager,
                production_target_perception=(
                    profile is PerceptionRuntimeProfile.PRODUCTION
                ),
                target_estimate_evaluator=target_estimate_evaluator,
            )
        except KeyboardInterrupt:
            interrupted = True
            if profile is PerceptionRuntimeProfile.ORACLE_EVALUATION:
                loop_result = _best_effort_interrupt_land(
                    simulation_app=simulation_app,
                    environment=environment,
                    oracle=perception,
                    agent=agent,
                    clock=clock,
                    shutdown_guard_s=shutdown_guard_s,
                )
            else:
                loop_result = _best_effort_production_failure_land(
                    simulation_app=simulation_app,
                    environment=environment,
                    perception=perception,
                    agent=agent,
                    clock=clock,
                    shutdown_guard_s=shutdown_guard_s,
                )

        final_pose = environment.uav_controller.get_pose()
        final_xyz = (final_pose.x, final_pose.y, final_pose.z)
        home = world_context.landing_zones[_landing_zone_name(compiled)]
        _, _, checks_passed = _print_runtime_summary(
            args=args,
            compiled=compiled,
            selected_model=(
                selected_model
                if selected_model is not None
                else (None if visual_runtime is None else visual_runtime.model_name)
            ),
            model_calls=(0 if counting_client is None else counting_client.chat_calls),
            manager=manager,
            target_manager=target_manager,
            snapshot=loop_result.snapshot,
            final_xyz_m=final_xyz,
            home_xy_m=home.position_xy_m,
            ground_altitude_m=home.ground_altitude_m,
            elapsed_s=max(0.0, clock.now() - task_start_s),
            guard_error=loop_result.guard_error,
        )
        if visual_runtime is not None:
            terminal_snapshot = loop_result.snapshot
            visual_runtime.emit_new_transitions(manager)
            visual_runtime.emit_new_search_metrics(manager)
            visual_runtime.emit_new_runtime_assessments(
                step_id=manager.active_planned_step_id,
            )
            visual_runtime.emit_new_next_best_view_proposals()
            visual_runtime.emit_new_revisions(
                skill=terminal_snapshot.active_skill or "NONE",
                step_id=manager.active_planned_step_id or "none",
            )
            if (
                terminal_snapshot.mission_id is not None
                and terminal_snapshot.plan_version is not None
            ):
                visual_runtime.emit_new_obstacle_revisions(
                    mission_id=terminal_snapshot.mission_id,
                    uav_id=terminal_snapshot.uav_id,
                    plan_version=terminal_snapshot.plan_version,
                )
            visual_runtime.set_terminal_manifest(
                agent_status=terminal_snapshot.status.value,
                task_status=terminal_snapshot.task_status,
                plan_version=terminal_snapshot.plan_version or 1,
                guard_error=loop_result.guard_error,
                mission_success=bool(checks_passed),
            )
        if interrupted:
            return 130
        return 0 if checks_passed else 1
    except KeyboardInterrupt:
        if (
            args.perception_runtime_profile == "oracle_evaluation"
            and all(value is not None for value in (agent, environment, perception, clock))
        ):
            _best_effort_interrupt_land(
                simulation_app=simulation_app,
                environment=environment,
                oracle=perception,
                agent=agent,
                clock=clock,
                shutdown_guard_s=shutdown_guard_s,
            )
        elif all(
            value is not None
            for value in (agent, environment, perception, clock)
        ):
            _best_effort_production_failure_land(
                simulation_app=simulation_app,
                environment=environment,
                perception=perception,
                agent=agent,
                clock=clock,
                shutdown_guard_s=shutdown_guard_s,
            )
        elif agent is not None:
            try:
                agent.cancel()
            except Exception:
                pass
        return 130
    except Exception as exc:
        # Never include prompts, API headers, base64 image content, or raw
        # evaluator frames in this error boundary.
        print(f"dynamic visual pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if all(
            value is not None
            for value in (agent, environment, perception, clock)
        ):
            try:
                if args.perception_runtime_profile == "oracle_evaluation":
                    _best_effort_interrupt_land(
                        simulation_app=simulation_app,
                        environment=environment,
                        oracle=perception,
                        agent=agent,
                        clock=clock,
                        shutdown_guard_s=shutdown_guard_s,
                    )
                else:
                    _best_effort_production_failure_land(
                        simulation_app=simulation_app,
                        environment=environment,
                        perception=perception,
                        agent=agent,
                        clock=clock,
                        shutdown_guard_s=shutdown_guard_s,
                    )
            except Exception as landing_exc:
                # A final direct stop is safer than leaving a previous
                # velocity command latched when LAND observation assembly is
                # itself unavailable.
                try:
                    stop = getattr(environment.uav_controller, "stop", None)
                    if callable(stop):
                        stop()
                except Exception:
                    pass
                print(
                    "fail-safe LAND cleanup failed: "
                    f"{type(landing_exc).__name__}",
                    file=sys.stderr,
                )
        return 1
    finally:
        try:
            if (
                next_best_view_worker is not None
                and not next_best_view_worker_registered
            ):
                try:
                    next_best_view_worker.close(
                        timeout_s=config.model_worker.request_timeout_s + 1.0
                    )
                except Exception as exc:
                    print(
                        "next-best-view worker shutdown failed: "
                        f"{type(exc).__name__}",
                        file=sys.stderr,
                    )
            if target_perception_coordinator is not None:
                perception_metrics = target_perception_coordinator.metrics
                try:
                    print(
                        "[TargetPerceptionMetrics] "
                        + json.dumps(
                            perception_metrics.to_dict(),
                            sort_keys=True,
                            allow_nan=False,
                        )
                    )
                    if visual_runtime is not None:
                        visual_runtime.record_target_perception_metrics(
                            perception_metrics
                        )
                except Exception as exc:
                    print(
                        "target perception metrics output failed: "
                        f"{type(exc).__name__}",
                        file=sys.stderr,
                    )
                try:
                    target_perception_coordinator.close()
                except Exception as exc:
                    print(
                        "target perception shutdown failed: "
                        f"{type(exc).__name__}",
                        file=sys.stderr,
                    )
            elif target_evaluation_metrics is not None:
                # Oracle upper-bound runs have no production coordinator, but
                # their evaluator-only RMSE output follows the same bounded
                # metrics schema.  No truth object is persisted.
                try:
                    print(
                        "[TargetPerceptionMetrics] "
                        + json.dumps(
                            target_evaluation_metrics.to_dict(),
                            sort_keys=True,
                            allow_nan=False,
                        )
                    )
                    if visual_runtime is not None:
                        visual_runtime.record_target_perception_metrics(
                            target_evaluation_metrics
                        )
                except Exception as exc:
                    print(
                        "target evaluator metrics output failed: "
                        f"{type(exc).__name__}",
                        file=sys.stderr,
                    )
            if visual_runtime is not None:
                try:
                    visual_runtime.close(
                        timeout_s=config.model_worker.request_timeout_s + 1.0
                    )
                except Exception as exc:
                    print(
                        f"visual worker shutdown failed: {type(exc).__name__}",
                        file=sys.stderr,
                    )
            if environment is not None:
                if debug_visualizer is not None:
                    try:
                        debug_visualizer.close()
                    except Exception as exc:
                        print(
                            f"debug visualization shutdown failed: "
                            f"{type(exc).__name__}",
                            file=sys.stderr,
                        )
                environment.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
