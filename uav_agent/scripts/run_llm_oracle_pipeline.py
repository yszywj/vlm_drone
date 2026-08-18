#!/usr/bin/env python3
"""Run MissionAgent with a legacy or dynamic text planner in Isaac Sim."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from math import hypot, isfinite
import os
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = Path(
    os.environ.get(
        "UAV_AGENT_CONDA_ENV",
        "/home/amax/miniconda3/envs/r_isaac_sim",
    )
).expanduser()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Config loading is deliberately the only project import at module scope.
# In particular, no Isaac-backed environment module may be imported until
# ``SimulationApp`` has been constructed in :func:`main`.
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
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than zero"
        )
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uav-id",
        default="uav_1",
        type=validate_uav_id,
        help="trusted UAV routing ID (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="unified YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--planner",
        choices=("scripted", "llm", "dynamic_scripted", "dynamic_llm"),
        default="scripted",
        help="high-level mission planner (default: scripted)",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="natural-language mission instruction",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base; otherwise QWEN_API_BASE/default",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="text model name; otherwise QWEN_MODEL/default",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key; otherwise QWEN_API_KEY/default (never logged)",
    )
    parser.add_argument(
        "--takeoff-altitude",
        type=_positive_float,
        default=10.0,
        help="TAKEOFF/search/return world-Z in metres (default: 10)",
    )
    parser.add_argument(
        "--track-duration",
        type=_positive_float,
        default=30.0,
        help="successful TRACK duration in simulation seconds (default: 30)",
    )
    parser.add_argument(
        "--max-sim-time",
        type=_positive_float,
        default=360.0,
        help="external task guard before cancel-and-LAND (default: 360)",
    )
    parser.add_argument(
        "--start-altitude",
        type=_finite_float,
        default=0.0,
        help="episode reset world-Z in metres (default: 0)",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="run without a visible viewport",
    )
    display.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="show the Isaac Sim viewport",
    )
    parser.set_defaults(headless=None)
    parser.add_argument(
        "--debug-ground-truth",
        action="store_true",
        help="print evaluator target truth at most once per simulation second",
    )
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


class _CountingModelClient:
    """Count logical planner chat calls without retaining prompts or outputs."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.chat_calls = 0

    def healthcheck(self) -> None:
        self._delegate.healthcheck()

    def chat(self, messages: object, *, options: object = None) -> object:
        self.chat_calls += 1
        return self._delegate.chat(messages, options=options)


@dataclass(frozen=True, slots=True)
class _LoopResult:
    snapshot: object
    guard_error: str | None = None


def _shutdown_guard_s(takeoff_altitude_m: float, start_altitude_m: float) -> float:
    # LAND descends at 0.5 m/s by default.  Keep ample finite margin for the
    # current altitude plus Camera sampling and renderer startup jitter.
    highest_expected = max(takeoff_altitude_m, start_altitude_m, 0.0)
    return max(60.0, 4.0 * highest_expected + 20.0)


def _random_target_spawn(config: object) -> tuple[float, float, float]:
    """Sample only after planner context construction, never for its prompt."""

    region = config.target.initial_region
    rng = random.Random(config.target.motion.seed)
    return tuple(
        rng.uniform(lower, upper)
        for lower, upper in zip(region.min_xyz_m, region.max_xyz_m)
    )


def _standard_dynamic_draft_data(args: argparse.Namespace) -> dict[str, object]:
    """Model-free dynamic baseline; geometry is intentionally absent."""

    return {
        "schema_version": 1,
        "steps": [
            {
                "id": "takeoff_1",
                "skill": "TAKEOFF",
                "args": {"altitude_m": args.takeoff_altitude},
            },
            {
                "id": "goto_search",
                "skill": "GOTO",
                "args": {
                    "destination": "search_area",
                    "altitude_m": args.takeoff_altitude,
                    "yaw_mode": "COURSE_ALIGNED",
                },
            },
            {
                "id": "search_1",
                "skill": "SEARCH",
                "args": {
                    "region": "search_area",
                    "target_description": "moving target",
                    "altitude_m": args.takeoff_altitude,
                },
            },
            {
                "id": "track_1",
                "skill": "TRACK",
                "args": {
                    "target_ref": "$search_1.target_id",
                    "duration_s": args.track_duration,
                    "desired_altitude_m": args.takeoff_altitude,
                    "desired_distance_m": 6.0,
                },
                "recovery": {
                    "skill": "REACQUIRE",
                    "max_attempts": 2,
                    "search_radius_m": 10.0,
                    "timeout_s": 30.0,
                },
            },
            {
                "id": "goto_home",
                "skill": "GOTO",
                "args": {
                    "destination": "home",
                    "altitude_m": args.takeoff_altitude,
                    "yaw_mode": "COURSE_ALIGNED",
                },
            },
            {
                "id": "land_1",
                "skill": "LAND",
                "args": {"zone": "home"},
            },
        ],
    }


def _print_ground_truth(frame: object, timestamp_s: float) -> None:
    position = tuple(float(value) for value in frame.target_position_m)
    velocity = tuple(float(value) for value in frame.target_velocity_mps)
    visible = bool(frame.target_projection.visible[0])
    print(
        "[GroundTruth] "
        f"t={timestamp_s:.3f} target_position_m={position} "
        f"target_velocity_mps={velocity} visible={visible}"
    )


def _run_until_terminal(
    *,
    simulation_app: object,
    environment: object,
    oracle: object,
    agent: object,
    clock: IsaacSimulationClock,
    task_start_s: float,
    max_sim_time_s: float,
    shutdown_guard_s: float,
    debug_ground_truth: bool,
) -> _LoopResult:
    """Advance only on fresh RGB samples and honor the external task guard."""

    snapshot = agent.snapshot()
    cancel_requested = False
    landing_deadline_s: float | None = None
    next_debug_time_s = task_start_s

    while simulation_app.is_running() and snapshot.status.value == "RUNNING":
        now = clock.now()
        # Bound every LAND, including normal completion, task-failure landing,
        # and SafetySupervisor-triggered shutdown.  This also covers the case
        # where safety entered LAND before the external task deadline.
        if snapshot.active_skill == "LAND" and landing_deadline_s is None:
            landing_deadline_s = now + shutdown_guard_s

        if not cancel_requested and now - task_start_s > max_sim_time_s:
            print(
                "[MissionAgent] external max-sim-time reached; "
                "requesting fail-safe LAND",
                file=sys.stderr,
            )
            cancel_requested = True
            try:
                snapshot = agent.cancel()
            except Exception as exc:
                # A prior safety decision may already have latched shutdown.
                # It is safe to keep advancing that existing LAND, but do not
                # hide any rejection while another Skill remains active.
                snapshot = agent.snapshot()
                if snapshot.active_skill != "LAND":
                    return _LoopResult(
                        snapshot,
                        "external timeout could not request fail-safe LAND: "
                        f"{exc}",
                    )
                print(
                    "[MissionAgent] fail-safe LAND was already in progress",
                    file=sys.stderr,
                )
            if landing_deadline_s is None:
                landing_deadline_s = now + shutdown_guard_s

        if landing_deadline_s is not None and now > landing_deadline_s:
            return _LoopResult(
                snapshot,
                "fail-safe LAND exceeded its separate shutdown guard",
            )

        # This gate is the synchronization boundary: neither Agent nor Oracle
        # is called until the environment has produced a new Camera sample.
        if not environment.step():
            continue
        frame = environment.get_evaluator_frame()
        observation = oracle.observe(frame)
        if debug_ground_truth and observation.timestamp >= next_debug_time_s:
            _print_ground_truth(frame, observation.timestamp)
            next_debug_time_s = observation.timestamp + 1.0
        snapshot = agent.tick(observation)

    if snapshot.status.value == "RUNNING":
        return _LoopResult(
            snapshot,
            "SimulationApp stopped before the mission reached a terminal state",
        )
    return _LoopResult(snapshot)


def _best_effort_interrupt_land(
    *,
    simulation_app: object,
    environment: object,
    oracle: object,
    agent: object,
    clock: IsaacSimulationClock,
    shutdown_guard_s: float,
) -> _LoopResult:
    """Request cancel on Ctrl-C and keep ticking a bounded fail-safe LAND."""

    snapshot = agent.snapshot()
    if snapshot.status.value != "RUNNING":
        return _LoopResult(snapshot)

    try:
        snapshot = agent.cancel()
        print(
            "[MissionAgent] KeyboardInterrupt: cancel requested; "
            "continuing bounded LAND",
            file=sys.stderr,
        )
    except Exception as exc:
        # A cancellation may already be latched and LAND may already be active.
        # Continue it if possible, but state the inability to request a new one.
        snapshot = agent.snapshot()
        print(
            "[MissionAgent] KeyboardInterrupt: cancel request was not accepted: "
            f"{exc}",
            file=sys.stderr,
        )
        if (
            snapshot.status.value == "RUNNING"
            and snapshot.active_skill != "LAND"
        ):
            return _LoopResult(
                snapshot,
                "KeyboardInterrupt could not start fail-safe LAND",
            )

    deadline_s = clock.now() + shutdown_guard_s
    try:
        while simulation_app.is_running() and snapshot.status.value == "RUNNING":
            if clock.now() > deadline_s:
                return _LoopResult(
                    snapshot,
                    "KeyboardInterrupt LAND exceeded its shutdown guard",
                )
            if not environment.step():
                continue
            frame = environment.get_evaluator_frame()
            observation = oracle.observe(frame)
            snapshot = agent.tick(observation)
    except KeyboardInterrupt:
        return _LoopResult(
            snapshot,
            "a second KeyboardInterrupt stopped the best-effort LAND",
        )

    if snapshot.status.value == "RUNNING":
        return _LoopResult(
            snapshot,
            "SimulationApp stopped before KeyboardInterrupt LAND completed",
        )
    return _LoopResult(snapshot)


def _print_plan(compiled: object) -> None:
    planner_output = compiled.planner_output
    dynamic = compiled.skill_plan_draft is not None
    print(
        "[Planner] SkillPlanDraft (planner-selected)"
        if dynamic
        else "[Planner] MissionIntent"
    )
    print(
        json.dumps(
            planner_output.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    print("[Planner] Compiled TaskPlan (trusted geometry/policy/timeouts)")
    print(
        json.dumps(
            compiled.task_plan.to_dicts(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    if dynamic:
        print("[Planner] Compiler Notes")
        print(
            json.dumps(
                list(compiled.compiler_notes),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
        )


def _landing_zone_name(compiled: object) -> str:
    if compiled.intent is not None:
        return compiled.intent.landing_zone
    draft = compiled.skill_plan_draft
    if draft is None or not draft.steps:
        raise RuntimeError("compiled mission has no planner output")
    final_step = draft.steps[-1]
    zone = final_step.args.get("zone")
    if final_step.skill != "LAND" or not isinstance(zone, str) or not zone:
        raise RuntimeError("dynamic plan has no final named landing zone")
    return zone


def _print_runtime_summary(
    *,
    args: argparse.Namespace,
    compiled: object,
    selected_model: str | None,
    model_calls: int,
    manager: object,
    target_manager: object,
    snapshot: object,
    final_xyz_m: tuple[float, float, float],
    home_xy_m: tuple[float, float],
    ground_altitude_m: float,
    elapsed_s: float,
    guard_error: str | None,
) -> tuple[float, float, bool]:
    home_xy_error = hypot(
        final_xyz_m[0] - home_xy_m[0],
        final_xyz_m[1] - home_xy_m[1],
    )
    altitude_error = abs(final_xyz_m[2] - ground_altitude_m)

    print("[Summary] Skill transitions")
    for transition in manager.transition_log:
        print(
            "[SkillTransition] "
            + json.dumps(
                transition.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
    print("[Summary] Target lifecycle events")
    for event in target_manager.events():
        print(
            "[TargetLifecycle] "
            + json.dumps(
                event.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )

    last_result = manager.last_result
    last_result_code = None if last_result is None else last_result.code.name
    task_status = manager.task_status.name
    print(f"[Summary] planner={args.planner}")
    draft = compiled.skill_plan_draft
    print(
        "[Summary] planner_output_type="
        f"{type(compiled.planner_output).__name__}"
    )
    print(
        "[Summary] draft_step_count="
        f"{0 if draft is None else len(draft.steps)}"
    )
    print(f"[Summary] compiled_step_count={len(compiled.task_plan.steps)}")
    recovery_budget = 0
    if draft is not None:
        recovery_budget = sum(
            0 if step.recovery is None else step.recovery.max_attempts
            for step in draft.steps
        )
    print(f"[Summary] planned_recovery_budget={recovery_budget}")
    print(f"[Summary] model={selected_model or 'none'}")
    print(f"[Summary] model_chat_calls={model_calls}")
    print(f"[Summary] agent_status={snapshot.status.value}")
    print(f"[Summary] task_status={task_status}")
    print(f"[Summary] last_result_code={last_result_code}")
    print(
        "[Summary] "
        f"final_uav_position_m=({final_xyz_m[0]:.3f}, "
        f"{final_xyz_m[1]:.3f}, {final_xyz_m[2]:.3f})"
    )
    print(f"[Summary] home_xy_error_m={home_xy_error:.3f}")
    print(f"[Summary] altitude_error_m={altitude_error:.3f}")
    print(f"[Summary] mission_sim_time_s={elapsed_s:.3f}")
    if guard_error is not None:
        print(f"[Summary] shutdown_error={guard_error}", file=sys.stderr)

    final_checks = (
        snapshot.status.value == "SUCCEEDED"
        and task_status == "SUCCEEDED"
        and last_result_code == "LAND_COMPLETE"
        and home_xy_error <= 1.25
        and altitude_error <= 0.20
        and guard_error is None
    )
    print(f"[Summary] final_checks_passed={final_checks}")
    return home_xy_error, altitude_error, final_checks


def main() -> int:
    args = parse_args()
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve():
        print(
            "error: run this standalone demo through "
            "./python.sh scripts/run_llm_oracle_pipeline.py",
            file=sys.stderr,
        )
        return 2
    if not args.instruction.strip():
        print("error: --instruction must not be empty", file=sys.stderr)
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
        print(
            "error: --start-altitude must not exceed --takeoff-altitude",
            file=sys.stderr,
        )
        return 2

    # These are all pure-Python planning/runtime definitions.  Construct the
    # planner's safe world view before importing or instantiating Isaac Sim.
    from models import OpenAICompatibleClient
    from planner import (
        DynamicLLMPlanner,
        LLMPlanner,
        MissionIntent,
        PlannerPolicy,
        ScriptedDynamicPlanner,
        ScriptedPlanner,
        SkillPlanDraft,
    )
    from runtime import PlanValidator, PlannerLimits, SafetySupervisor
    from runtime.world_context_builder import (
        WorldContextBuildError,
        build_planner_world_context,
    )

    try:
        world_context = build_planner_world_context(
            config,
            takeoff_altitude_m=args.takeoff_altitude,
            track_duration_s=args.track_duration,
            start_altitude_m=args.start_altitude,
            # A text planner may choose any safe scene altitude.  At the
            # current default descent speed (0.5 m/s), three seconds per metre
            # leaves deterministic margin even from the scene ceiling.
            land_timeout_s=max(60.0, 3.0 * config.scene.size_xyz_m[2]),
        )
    except WorldContextBuildError as exc:
        print(f"world context error: {exc}", file=sys.stderr)
        return 2

    planner_limits = PlannerLimits.from_config(config.planner)
    planner_policy = PlannerPolicy.from_config(config.planner, planner_limits)
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
        raw_client = OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )
        selected_model = raw_client.model
        counting_client = _CountingModelClient(raw_client)
        if args.planner == "llm":
            planner = LLMPlanner(
                counting_client,
                PROJECT_ROOT / "prompts" / "mission_planner_system.txt",
            )
        else:
            planner = DynamicLLMPlanner(
                counting_client,
                PROJECT_ROOT / "prompts" / "dynamic_skill_planner_system.txt",
                planner_limits=planner_limits,
                planner_policy=planner_policy,
            )

    validator = PlanValidator(planner_limits, planner_policy)
    shutdown_guard_s = _shutdown_guard_s(
        # An LLM may validly select another altitude within the trusted scene.
        # Size the emergency guard for the highest validator-accepted Z.
        max(args.takeoff_altitude, config.scene.size_xyz_m[2]),
        args.start_altitude,
    )
    safety = SafetySupervisor(
        world_context.scene_min_xyz_m,
        world_context.scene_max_xyz_m,
        # The explicit CLI guard owns cancellation.  Safety remains active
        # through the separately bounded landing interval.
        max_mission_time_s=args.max_sim_time + shutdown_guard_s + 1.0,
        position_margin_m=0.25,
        max_safe_altitude_m=world_context.scene_max_xyz_m[2],
        planner_limits=planner_limits,
    )

    # Standalone ordering boundary: every Isaac-backed import is below the
    # constructed SimulationApp.  Do not move environment imports above it.
    from isaacsim import SimulationApp

    headless = config.simulation.headless if args.headless is None else args.headless
    simulation_app = SimulationApp({"headless": headless})
    environment = None
    agent = None
    oracle = None
    clock = None
    interrupted = False
    try:
        from agents.mission_agent import MissionAgent
        from env.simple_uav_search_env import SimpleUavSearchEnv
        from perception import (
            PerceptionRuntimeProfile,
        )
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
        target_spawn = _random_target_spawn(config)
        environment.set_target_pose(target_spawn)
        if args.debug_ground_truth:
            print(f"[GroundTruth] target_spawn_m={target_spawn}")

        print(
            "[Perception] PRIVILEGED ORACLE_EVALUATION profile enabled; "
            "this is an upper-bound/regression path, not production vision"
        )
        oracle = _build_oracle_evaluation_backend(args.uav_id)
        clock = IsaacSimulationClock(environment)
        context = environment.make_skill_context(clock, perception=oracle)
        manager = SkillManager(
            context,
            registry=create_default_skill_registry(
                transit_yaw_mode=config.search.transit_yaw_mode
            ),
        )
        target_manager = TargetManager()
        agent = MissionAgent(
            planner,
            validator,
            safety,
            manager,
            target_manager,
            clock,
            perception_runtime_profile=(
                PerceptionRuntimeProfile.ORACLE_EVALUATION
            ),
            acknowledge_privileged_oracle=True,
        )

        task_start_s = clock.now()
        compiled = agent.start(args.instruction, world_context)
        _print_plan(compiled)

        try:
            loop_result = _run_until_terminal(
                simulation_app=simulation_app,
                environment=environment,
                oracle=oracle,
                agent=agent,
                clock=clock,
                task_start_s=task_start_s,
                max_sim_time_s=args.max_sim_time,
                shutdown_guard_s=shutdown_guard_s,
                debug_ground_truth=args.debug_ground_truth,
            )
        except KeyboardInterrupt:
            interrupted = True
            loop_result = _best_effort_interrupt_land(
                simulation_app=simulation_app,
                environment=environment,
                oracle=oracle,
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
            selected_model=selected_model,
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
        if interrupted:
            print(
                "[Summary] process interrupted; best-effort LAND result reported above",
                file=sys.stderr,
            )
            return 130
        return 0 if checks_passed else 1
    except KeyboardInterrupt:
        if (
            agent is not None
            and environment is not None
            and oracle is not None
            and clock is not None
        ):
            result = _best_effort_interrupt_land(
                simulation_app=simulation_app,
                environment=environment,
                oracle=oracle,
                agent=agent,
                clock=clock,
                shutdown_guard_s=shutdown_guard_s,
            )
            if result.guard_error is not None:
                print(
                    f"pipeline interrupt shutdown failed: {result.guard_error}",
                    file=sys.stderr,
                )
        else:
            print(
                "pipeline interrupted before a flight task was available to land",
                file=sys.stderr,
            )
        return 130
    except Exception as exc:
        # Model-client and planner errors intentionally omit API keys.  Do not
        # add raw HTTP headers, prompts, RGB arrays, or evaluator frames here.
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "[Summary] model_chat_calls="
            f"{0 if counting_client is None else counting_client.chat_calls}",
            file=sys.stderr,
        )
        return 1
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
