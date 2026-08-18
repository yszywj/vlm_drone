#!/usr/bin/env python3
"""Run the complete Stage-0 Oracle Skill pipeline in Isaac Sim standalone."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
import random
import sys
from dataclasses import asdict, is_dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = Path(
    os.environ.get(
        "UAV_AGENT_CONDA_ENV",
        "/home/amax/miniconda3/envs/r_isaac_sim",
    )
).expanduser()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import ConfigError, load_config  # noqa: E402
from common.ids import validate_uav_id  # noqa: E402


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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uav-id", default="uav_1", type=validate_uav_id)
    parser.add_argument("--config", required=True, help="path to the unified YAML config")
    parser.add_argument(
        "--track-duration",
        type=_positive_float,
        default=30.0,
        help="seconds of successful tracking before TRACK_COMPLETE (default: 30)",
    )
    parser.add_argument(
        "--max-sim-time",
        type=_positive_float,
        default=360.0,
        help="task-time guard before requesting fail-safe LAND (default: 360)",
    )
    parser.add_argument(
        "--start-altitude",
        type=_finite_float,
        default=0.0,
        help="episode start world-Z in metres; reset/debug override (default: 0)",
    )
    parser.add_argument(
        "--takeoff-altitude",
        type=_positive_float,
        default=10.0,
        help="world-Z TAKEOFF target in metres (default: 10)",
    )
    parser.add_argument(
        "--target-description",
        default="moving target",
        help="description passed to the deterministic SEARCH Skill",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=None)
    return parser.parse_args()


class IsaacSimulationClock:
    """Expose Isaac World simulation time through the SkillClock protocol."""

    def __init__(self, environment: object) -> None:
        self._environment = environment

    def now(self) -> float:
        world = getattr(self._environment, "world", None)
        if world is None:
            raise RuntimeError("environment World is not available")
        return float(world.current_time)


def _task_plan_dicts(config: object, args: argparse.Namespace) -> list[dict[str, object]]:
    """Build the one hand-written Stage-0 plan in a single adaptation point."""

    target_region = config.target.initial_region
    center = tuple(
        (low + high) / 2.0
        for low, high in zip(target_region.min_xyz_m, target_region.max_xyz_m)
    )
    search_altitude = float(args.takeoff_altitude)
    # Approach the west edge so SEARCH's FACE_POINT transit initially observes
    # the region rather than placing a downward-looking Camera directly above it.
    approach_x = max(
        -config.scene.size_xyz_m[0] / 2.0,
        center[0] - config.search.radius_m,
    )
    approach_position = (approach_x, center[1], search_altitude)
    return [
        {
            "skill": "TAKEOFF",
            "target_altitude": search_altitude,
            "timeout": max(20.0, 2.0 * abs(search_altitude - args.start_altitude)),
        },
        {
            "skill": "GOTO",
            "position": list(approach_position),
            "timeout": 60.0,
        },
        {
            "skill": "SEARCH",
            "center": list(center),
            "radius": config.search.radius_m,
            "target_description": args.target_description,
            "search_altitude": search_altitude,
            "timeout": config.search.timeout_s,
        },
        {
            "skill": "TRACK",
            "target_id": "$SEARCH.result.target_id",
            "desired_altitude": search_altitude,
            "track_duration": args.track_duration,
        },
        {
            "skill": "LAND",
            "ground_altitude": 0.0,
            "timeout": max(30.0, 3.0 * search_altitude),
        },
    ]


def _entry_mapping(entry: object) -> Mapping[str, object]:
    if isinstance(entry, Mapping):
        return entry
    to_dict = getattr(entry, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    if is_dataclass(entry) and not isinstance(entry, type):
        return asdict(entry)
    return {"transition": entry}


def _display_value(value: object) -> object:
    enum_name = getattr(value, "name", None)
    return enum_name if isinstance(enum_name, str) else value


def _print_new_transitions(manager: object, start_index: int) -> int:
    entries = tuple(getattr(manager, "transition_log", ()))
    for entry in entries[start_index:]:
        data = _entry_mapping(entry)
        timestamp = data.get("timestamp", data.get("timestamp_s", "?"))
        old_skill = _display_value(data.get("old_skill", "NONE"))
        new_skill = _display_value(data.get("new_skill", "NONE"))
        old_status = _display_value(data.get("old_status", "-"))
        result_code = _display_value(data.get("result_code", "-"))
        reason = data.get("reason", "")
        print(
            "[SkillManager] "
            f"t={timestamp} {old_skill}({old_status}/{result_code}) -> "
            f"{new_skill} reason={reason}"
        )
    return len(entries)


def main() -> int:
    args = parse_args()
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve():
        print(
            "error: use ./python.sh scripts/run_oracle_pipeline.py "
            "--config configs/default.yaml",
            file=sys.stderr,
        )
        return 2
    if not args.target_description.strip():
        print("error: --target-description must not be empty", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
        config = replace(config, uav=replace(config.uav, id=args.uav_id))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if not 0.0 <= args.start_altitude <= config.scene.size_xyz_m[2]:
        print("error: --start-altitude is outside scene Z bounds", file=sys.stderr)
        return 2
    if not 0.0 < args.takeoff_altitude <= config.scene.size_xyz_m[2]:
        print("error: --takeoff-altitude is outside scene Z bounds", file=sys.stderr)
        return 2
    if args.start_altitude > args.takeoff_altitude:
        print("error: --start-altitude must not exceed --takeoff-altitude", file=sys.stderr)
        return 2

    # Standalone ordering is mandatory: no Isaac-backed environment module is
    # imported before SimulationApp has been constructed.
    from isaacsim import SimulationApp

    headless = config.simulation.headless if args.headless is None else args.headless
    simulation_app = SimulationApp({"headless": headless})
    environment = None
    try:
        from env.simple_uav_search_env import SimpleUavSearchEnv
        from skills.manager import (
            SkillManager,
            TaskPlan,
            TaskStatus,
            create_default_skill_registry,
        )

        environment = SimpleUavSearchEnv(config)
        environment.setup()
        if not headless:
            environment.configure_overview_viewport()

        initial = config.uav.initial_position_xyz_m
        environment.set_uav_pose((initial[0], initial[1], args.start_altitude))
        spawn_rng = random.Random(config.target.motion.seed)
        target_region = config.target.initial_region
        target_start = tuple(
            spawn_rng.uniform(low, high)
            for low, high in zip(
                target_region.min_xyz_m,
                target_region.max_xyz_m,
            )
        )
        environment.set_target_pose(target_start)
        print(f"[TaskManager] target_spawn={target_start}")
        print(
            "[Perception] PRIVILEGED ORACLE_EVALUATION profile enabled; "
            "not a production visual-perception runtime"
        )
        oracle = _build_oracle_evaluation_backend(args.uav_id)
        clock = IsaacSimulationClock(environment)
        context = environment.make_skill_context(clock, perception=oracle)
        registry = create_default_skill_registry(
            transit_yaw_mode=config.search.transit_yaw_mode
        )
        manager = SkillManager(context, registry=registry)
        plan = TaskPlan.from_dicts(
            _task_plan_dicts(config, args),
            mission_id="mission_oracle_demo",
            uav_id=args.uav_id,
            plan_version=1,
        )

        print("[TaskManager] Task started")
        manager.start_task(plan)
        transition_index = _print_new_transitions(manager, 0)
        deadline = clock.now() + args.max_sim_time
        landing_deadline: float | None = None

        while simulation_app.is_running() and manager.task_status is TaskStatus.RUNNING:
            now = clock.now()
            if landing_deadline is None and now > deadline:
                print(
                    f"[TaskManager] task guard reached at {now:.3f}s; requesting LAND",
                    file=sys.stderr,
                )
                manager.cancel_task()
                transition_index = _print_new_transitions(
                    manager,
                    transition_index,
                )
                # The external guard does not bypass the fail-safe LAND. Give
                # the configured descent a separate bounded shutdown window.
                landing_deadline = now + max(30.0, 3.0 * args.takeoff_altitude)
            if landing_deadline is not None and now > landing_deadline:
                print(
                    "[TaskManager] emergency LAND did not complete before its guard",
                    file=sys.stderr,
                )
                return 1
            if not environment.step():
                continue
            frame = environment.get_evaluator_frame()
            observation = oracle.observe(frame)
            manager.tick(observation)
            transition_index = _print_new_transitions(manager, transition_index)

        status = manager.task_status
        final_pose = environment.uav_controller.get_pose()
        print(f"[TaskManager] Task {status.name}")
        print(
            "[TaskManager] "
            f"final_uav_position=({final_pose.x:.3f}, {final_pose.y:.3f}, {final_pose.z:.3f})"
        )
        return 0 if status is TaskStatus.SUCCEEDED else 1
    except KeyboardInterrupt:
        print("Oracle pipeline interrupted by user")
        return 130
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
