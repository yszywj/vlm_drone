#!/usr/bin/env python3
"""Run one prevalidated multi-UAV Fleet mission in Isaac Sim.

Fleet decomposition, every local Spatial V3 model call, strict compilation,
target-perception preflight, and Safety preflight all finish before this module
performs its first Isaac import.  Isaac runtime execution then replays the
request-bound plans through the normal FleetMissionRuntime and MissionAgent
validation boundaries without calling either planner a second time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import cos, radians, sin
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Mapping, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Every import in this section is pure Python.  In particular, importing this
# entry point for argument or planning tests must never import ``isaacsim``.
from configs.loader import load_config  # noqa: E402
from fleet.compiler import FleetAssignmentCompiler  # noqa: E402
from fleet.llm_planner import LLMFleetPlanner  # noqa: E402
from fleet.logging import FleetMissionLogger  # noqa: E402
from fleet.local_spatial_planner import (  # noqa: E402
    RoutedPreplannedSpatialPlanner,
    ScriptedAssignmentSpatialPlanner,
)
from fleet.preplanned_planner import RoutedPreplannedFleetPlanner  # noqa: E402
from fleet.request_builder import (  # noqa: E402
    FleetRequestBuildError,
    build_agent_world_contexts,
    build_fleet_mission_request,
    parse_explicit_assignment_instruction,
)
from fleet.schemas import validate_fleet_mission_plan  # noqa: E402
from fleet.scripted_planner import ScriptedFleetPlanner  # noqa: E402
from fleet.types import (  # noqa: E402
    AssignmentCompilation,
    FleetMissionPlan,
    FleetMissionRequest,
)
from models.adapter_registry import (  # noqa: E402
    AdapterRegistry,
    AdapterSelection,
    DEFAULT_ADAPTER_CONFIG,
    ModelCallRole,
)
from models.model_client_factory import ModelClientFactory  # noqa: E402
from perception.factory import validate_target_perception_preflight  # noqa: E402
from planner.dynamic_llm_planner import DynamicLLMPlanner  # noqa: E402
from planner.policy import PlannerLimits, PlannerPolicy  # noqa: E402
from planner.schemas import PlannerWorldContext  # noqa: E402
from runtime.plan_validator import PlanValidator  # noqa: E402
from runtime.safety_supervisor import SafetyAction, SafetySupervisor  # noqa: E402


DEFAULT_INSTRUCTION = (
    "无人机A前往世界坐标二十、三十附近十五米范围搜索并跟踪目标i二十秒；"
    "无人机B前往世界坐标负二十五、十附近十二米范围搜索并跟踪目标j二十秒；"
    "完成后分别返回各自起点降落"
)


class FleetLaunchConfigurationError(ValueError):
    """Raised before Isaac startup when a launch is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class PreparedFleetMission:
    """Pure-Python result that is safe to hand across the Isaac boundary."""

    config: object
    request: FleetMissionRequest
    plan: FleetMissionPlan
    world_contexts: Mapping[str, PlannerWorldContext]
    compilations: Mapping[str, AssignmentCompilation]
    compiler: FleetAssignmentCompiler
    planner_limits: PlannerLimits
    planner_policy: PlannerPolicy
    fleet_planner_source: str
    local_planner_source: str
    adapter_selections: tuple[Mapping[str, object], ...]
    model_call_records: tuple[Mapping[str, object], ...]
    visual_clients: Mapping[str, object]
    runtime_visual_selection: AdapterSelection | None
    planned_routes: Mapping[str, tuple[tuple[float, float, float], ...]]
    headless: bool
    vision_review_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "world_contexts",
            MappingProxyType(dict(self.world_contexts)),
        )
        object.__setattr__(
            self,
            "compilations",
            MappingProxyType(dict(self.compilations)),
        )
        object.__setattr__(
            self,
            "visual_clients",
            MappingProxyType(dict(self.visual_clients)),
        )
        object.__setattr__(
            self,
            "planned_routes",
            MappingProxyType(
                {
                    uav_id: tuple(tuple(point) for point in route)
                    for uav_id, route in self.planned_routes.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "adapter_selections",
            tuple(MappingProxyType(dict(item)) for item in self.adapter_selections),
        )
        object.__setattr__(
            self,
            "model_call_records",
            tuple(MappingProxyType(dict(item)) for item in self.model_call_records),
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs/multi_uav_demo.yaml",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--fleet-planner",
        choices=("scripted", "llm"),
        default="scripted",
    )
    parser.add_argument(
        "--local-planner",
        choices=("dynamic_scripted", "dynamic_llm"),
        default="dynamic_scripted",
    )
    parser.add_argument("--planning-contract", choices=("v3",), default="v3")
    parser.add_argument(
        "--runtime-program",
        choices=("linear", "graph"),
        default="linear",
    )
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=DEFAULT_ADAPTER_CONFIG,
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--enable-qwen-vision", action="store_true")
    parser.add_argument(
        "--vision-review-mode",
        choices=("shadow", "gate"),
        default=None,
    )
    parser.add_argument(
        "--perception-runtime-profile",
        choices=("production", "oracle_evaluation"),
        default="production",
    )
    parser.add_argument(
        "--acknowledge-privileged-oracle",
        action="store_true",
    )
    parser.add_argument("--debug-visualization", action="store_true")
    headless = parser.add_mutually_exclusive_group()
    headless.add_argument("--headless", dest="headless", action="store_true")
    headless.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=None)
    parser.add_argument("--max-sim-time", type=float, default=300.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_PROJECT_ROOT / "logs/fleet_missions",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def prepare_fleet_mission(args: argparse.Namespace) -> PreparedFleetMission:
    """Plan, validate, compile, and preflight without importing Isaac."""

    if not isinstance(args.instruction, str) or not args.instruction.strip():
        raise FleetLaunchConfigurationError("--instruction must not be empty")
    if not isinstance(args.max_sim_time, (int, float)) or isinstance(
        args.max_sim_time, bool
    ):
        raise FleetLaunchConfigurationError("--max-sim-time must be a number")
    if not 0.0 < float(args.max_sim_time) <= 86_400.0:
        raise FleetLaunchConfigurationError(
            "--max-sim-time must be within (0, 86400] seconds"
        )

    config = load_config(args.config)
    resolved_headless = (
        bool(config.simulation.headless)
        if args.headless is None
        else bool(args.headless)
    )
    if args.debug_visualization and resolved_headless:
        raise FleetLaunchConfigurationError(
            "--debug-visualization requires --no-headless"
        )
    is_oracle = args.perception_runtime_profile == "oracle_evaluation"
    if is_oracle and not args.acknowledge_privileged_oracle:
        raise FleetLaunchConfigurationError(
            "oracle_evaluation requires --acknowledge-privileged-oracle"
        )
    if not is_oracle and args.acknowledge_privileged_oracle:
        raise FleetLaunchConfigurationError(
            "Oracle acknowledgement is forbidden in production"
        )
    review_mode = args.vision_review_mode or config.qwen_visual_review.mode
    if review_mode not in {"shadow", "gate"}:
        raise FleetLaunchConfigurationError(
            "vision review mode must be shadow or gate"
        )

    registry = AdapterRegistry(args.adapter_config)
    if args.model is not None and args.model != registry.base_model_name:
        raise FleetLaunchConfigurationError(
            "--model must match adapters.json base_model.served_model_name"
        )
    selections: list[dict[str, object]] = []
    model_call_records: list[dict[str, object]] = []
    client_factory = ModelClientFactory(
        registry,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_s=config.model_worker.request_timeout_s,
        selection_logger=lambda value: selections.append(dict(value)),
        call_logger=lambda value: model_call_records.append(dict(value)),
    )

    directives = None
    try:
        directives = parse_explicit_assignment_instruction(
            args.instruction,
            config,
        )
    except FleetRequestBuildError:
        if args.fleet_planner == "scripted":
            raise
    request = build_fleet_mission_request(
        config,
        args.instruction.strip(),
        directives=directives,
    )

    if args.fleet_planner == "scripted":
        fleet_planner = ScriptedFleetPlanner()
        selection = {
            **registry.resolve(ModelCallRole.FLEET_PLAN).to_dict(),
            "used": False,
        }
        selections.append(selection)
        model_call_records.append(
            _not_called_model_record(
                selection,
                fleet_mission_id=request.fleet_mission_id,
                call_id="not_called_fleet_plan",
            )
        )
    else:
        fleet_planner = LLMFleetPlanner(
            client_factory.for_role(
                ModelCallRole.FLEET_PLAN,
                fleet_mission_id=request.fleet_mission_id,
            )
        )
    plan = validate_fleet_mission_plan(fleet_planner.plan(request), request)
    if not plan.assignments:
        raise FleetLaunchConfigurationError(
            "Fleet Planner returned no executable assignments"
        )

    limits = PlannerLimits.from_config(config.planner)
    policy = PlannerPolicy.from_config(config.planner, limits)
    contexts = build_agent_world_contexts(config, plan)
    local_planners: dict[str, object] = {}
    if args.local_planner == "dynamic_scripted":
        for assignment in plan.assignments:
            local_planners[assignment.uav_id] = ScriptedAssignmentSpatialPlanner()
        selection = {
            **registry.resolve(ModelCallRole.AGENT_SPATIAL_PLAN).to_dict(),
            "used": False,
        }
        selections.append(selection)
        model_call_records.append(
            _not_called_model_record(
                selection,
                fleet_mission_id=request.fleet_mission_id,
                call_id="not_called_agent_spatial_plan",
            )
        )
    else:
        prompt = _PROJECT_ROOT / "prompts/dynamic_skill_planner_v3_system.txt"
        for assignment in plan.assignments:
            local_planners[assignment.uav_id] = DynamicLLMPlanner(
                client_factory.for_role(
                    ModelCallRole.AGENT_SPATIAL_PLAN,
                    fleet_mission_id=request.fleet_mission_id,
                    assignment_id=assignment.assignment_id,
                    uav_id=assignment.uav_id,
                ),
                system_prompt_path=prompt,
                planner_limits=limits,
                planner_policy=policy,
                planning_contract="v3",
            )
    compiler = FleetAssignmentCompiler(
        local_planners,
        validator=PlanValidator(limits, policy),
    )
    compilations = compiler.compile(request, plan, contexts)

    # Compilation is not enough: perform deterministic perception and Safety
    # preflight on every final TaskPlan before allowing the first Isaac import.
    for assignment in plan.assignments:
        result = compilations[assignment.uav_id]
        compiled = result.compiled_mission
        if compiled is None:
            raise FleetLaunchConfigurationError(
                f"local plan for {assignment.uav_id} was not compiled"
            )
        if not is_oracle:
            validate_target_perception_preflight(
                config.target_perception.backend,
                (step.skill for step in compiled.task_plan.steps),
            )
        context = contexts[assignment.uav_id]
        safety = SafetySupervisor(
            context.scene_min_xyz_m,
            context.scene_max_xyz_m,
            max_mission_time_s=float(args.max_sim_time) + 120.0,
            position_margin_m=0.25,
            max_safe_altitude_m=context.scene_max_xyz_m[2],
            planner_limits=limits,
        )
        decision = safety.preflight(compiled)
        if decision.action is not SafetyAction.CONTINUE:
            raise FleetLaunchConfigurationError(
                f"Safety preflight rejected {assignment.uav_id}: {decision.reason}"
            )

    visual_clients: dict[str, object] = {}
    runtime_visual_selection: AdapterSelection | None = None
    if args.enable_qwen_vision:
        # Construct role-bound clients during pure-Python preparation so URL,
        # credential shape, Adapter routing and per-UAV isolation all fail
        # before the first Isaac import.  The clients do not own threads and
        # cannot execute until the runtime wraps them behind the shared Broker
        # dispatcher below.
        visual_factory = ModelClientFactory(
            registry,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=config.model_worker.request_timeout_s,
        )
        for assignment in plan.assignments:
            visual_clients[assignment.uav_id] = visual_factory.for_role(
                ModelCallRole.RUNTIME_VISUAL_REVIEW,
                fleet_mission_id=request.fleet_mission_id,
                assignment_id=assignment.assignment_id,
                uav_id=assignment.uav_id,
            )
        runtime_visual_selection = registry.resolve(
            ModelCallRole.RUNTIME_VISUAL_REVIEW
        )
        selections.append(
            {
                **runtime_visual_selection.to_dict(),
                "used": True,
                "brokered": True,
            }
        )

    planned_routes = _build_planned_routes(contexts, compilations)

    return PreparedFleetMission(
        config=config,
        request=request,
        plan=plan,
        world_contexts=contexts,
        compilations=compilations,
        compiler=compiler,
        planner_limits=limits,
        planner_policy=policy,
        fleet_planner_source=fleet_planner.source,
        local_planner_source=args.local_planner,
        adapter_selections=tuple(selections),
        model_call_records=tuple(model_call_records),
        visual_clients=visual_clients,
        runtime_visual_selection=runtime_visual_selection,
        planned_routes=planned_routes,
        headless=resolved_headless,
        vision_review_mode=review_mode,
    )


def _region_representative_world(region: object) -> tuple[float, float, float]:
    """Return a conservative representative point from a compiled WORLD region."""

    frame = getattr(region, "frame", None)
    if frame is not None and getattr(frame, "value", frame) != "WORLD_ENU":
        raise FleetLaunchConfigurationError(
            "compiled SEARCH region must use WORLD_ENU before route extraction"
        )
    if hasattr(region, "entry_point_xyz_m") and region.entry_point_xyz_m is not None:
        return tuple(float(value) for value in region.entry_point_xyz_m)
    if hasattr(region, "center_xyz_m"):
        return tuple(float(value) for value in region.center_xyz_m)
    if hasattr(region, "origin_xyz_m"):
        distance_range = region.distance_range_m
        distance_m = (float(distance_range[0]) + float(distance_range[1])) / 2.0
        angle = radians(float(region.azimuth_center_deg))
        origin = region.origin_xyz_m
        return (
            float(origin[0]) + distance_m * cos(angle),
            float(origin[1]) + distance_m * sin(angle),
            float(origin[2]),
        )
    if hasattr(region, "vertices_xyz_m"):
        vertices = tuple(region.vertices_xyz_m)
        return tuple(
            sum(float(point[index]) for point in vertices) / len(vertices)
            for index in range(3)
        )
    if hasattr(region, "centerline_xyz_m"):
        return tuple(float(value) for value in region.centerline_xyz_m[0])
    raise FleetLaunchConfigurationError(
        "compiled SEARCH region has no trusted route representative"
    )


def _build_planned_routes(
    contexts: Mapping[str, PlannerWorldContext],
    compilations: Mapping[str, AssignmentCompilation],
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    """Extract image/Oracle-free high-level routes from compiled TaskPlans."""

    routes: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for uav_id, result in compilations.items():
        context = contexts[uav_id]
        points: list[tuple[float, float, float]] = [
            tuple(float(value) for value in context.initial_uav_xyz_m)
        ]
        compiled = result.compiled_mission
        if compiled is None:
            raise FleetLaunchConfigurationError(
                f"cannot extract route from uncompiled assignment {uav_id}"
            )
        for step in compiled.task_plan.steps:
            if step.skill.value == "TAKEOFF":
                altitude = float(step.params["target_altitude"])
                points.append((points[-1][0], points[-1][1], altitude))
            elif step.skill.value == "SEARCH":
                representative = _region_representative_world(step.params["region"])
                points.append(
                    (
                        representative[0],
                        representative[1],
                        float(step.params["search_altitude_m"]),
                    )
                )
            elif step.skill.value == "GOTO":
                points.append(
                    tuple(float(value) for value in step.params["position"])
                )
            elif step.skill.value == "LAND":
                xy = step.params.get("expected_position_xy")
                if xy is not None:
                    points.append(
                        (
                            float(xy[0]),
                            float(xy[1]),
                            float(step.params["ground_altitude"]),
                        )
                    )
        deduplicated: list[tuple[float, float, float]] = []
        for point in points:
            if not deduplicated or point != deduplicated[-1]:
                deduplicated.append(point)
        if len(deduplicated) < 2:
            raise FleetLaunchConfigurationError(
                f"planned route for {uav_id} contains fewer than two points"
            )
        routes[uav_id] = tuple(deduplicated)
    return routes


class _FleetSimulationClock:
    def __init__(self, environment: object) -> None:
        self._environment = environment

    def now(self) -> float:
        world = getattr(self._environment, "world", None)
        if world is None:
            raise RuntimeError("Fleet environment World is unavailable")
        return float(world.current_time)


def _spatial_resolver(context: PlannerWorldContext, home_name: str) -> object:
    from planner.spatial_resolver import FramePose, SpatialResolver

    home = context.landing_zones[home_name]
    home_xyz = (
        home.position_xy_m[0],
        home.position_xy_m[1],
        home.ground_altitude_m,
    )
    start = FramePose(context.initial_uav_xyz_m, 0.0)
    return SpatialResolver(
        home_pose=FramePose(home_xyz, 0.0),
        uav_start_pose=start,
        named_locations={home_name: home_xyz},
    )


def _build_visual_review_coordinator(
    *,
    prepared: PreparedFleetMission,
    uav_id: str,
    manager: object,
    target_manager: object,
    worker: object,
) -> object:
    from agents.visual_review_coordinator import VisualReviewCoordinator
    from perception import CandidateBank, QwenVLMVerifier, VisualReviewGate
    from runtime import FrameStore, MissionEventBus, ReviewScheduler

    config = prepared.config
    scheduler = ReviewScheduler(
        intervals_s={
            "GOTO": config.qwen_visual_review.goto_interval_s,
            "SEARCH": config.qwen_visual_review.search_interval_s,
            "INSPECT": config.qwen_visual_review.inspect_interval_s,
            "TRACK": config.qwen_visual_review.track_interval_s,
        }
    )
    frame_store = FrameStore(
        max_frames=config.frame_store.max_frames,
        max_bytes=config.frame_store.max_bytes,
        max_age_s=config.frame_store.max_age_s,
    )
    return VisualReviewCoordinator(
        uav_id=uav_id,
        scheduler=scheduler,
        frame_store=frame_store,
        worker=worker,
        verifier=QwenVLMVerifier(
            max_image_side_px=config.qwen_visual_review.max_image_side_px,
            jpeg_quality=config.qwen_visual_review.jpeg_quality,
        ),
        gate=VisualReviewGate(
            mode=prepared.vision_review_mode,
            min_consistent_matches=2,
        ),
        skill_manager=manager,
        target_manager=target_manager,
        candidate_bank=CandidateBank(uav_id=uav_id),
        event_bus=MissionEventBus(max_events=256),
        review_timeout_s=config.qwen_visual_review.blocking_hover_timeout_s,
        max_result_age_s=config.frame_store.max_age_s,
        hover_position_tolerance_m=(
            config.qwen_visual_review.hover_position_tolerance_m
        ),
        hover_max_correction_speed_mps=(
            config.qwen_visual_review.hover_max_correction_speed_mps
        ),
        blocking_timeout_fallback=(
            config.qwen_visual_review.blocking_timeout_fallback
        ),
        max_recent_frames=config.qwen_visual_review.max_recent_frames,
        require_stable_search_candidate=True,
        min_search_candidate_observations=(
            config.target_perception.tracker.min_track_observations
        ),
        min_search_candidate_duration_s=(
            config.target_perception.tracker.min_track_duration_s
        ),
    )


def _build_brokered_visual_workers(
    prepared: PreparedFleetMission,
    broker: object,
    logger: object,
    *,
    worker_factory: object | None = None,
    dispatcher_factory: object | None = None,
) -> tuple[object, Mapping[str, object]]:
    """Create per-UAV clients behind one shared Broker acquire owner.

    This helper is deliberately pure Python and injectable so the entrypoint's
    exact ownership wiring can be tested without importing Isaac Sim.
    """

    if prepared.runtime_visual_selection is None:
        raise FleetLaunchConfigurationError(
            "runtime visual Adapter selection was not preflighted"
        )
    expected_uav_ids = {
        assignment.uav_id for assignment in prepared.plan.assignments
    }
    if set(prepared.visual_clients) != expected_uav_ids:
        raise FleetLaunchConfigurationError(
            "runtime visual clients do not exactly cover Fleet assignments"
        )
    record_logger = getattr(logger, "log_model_call", None)
    if not callable(record_logger):
        raise TypeError("logger must provide log_model_call()")
    if worker_factory is None:
        from models import AsyncModelWorker

        worker_factory = AsyncModelWorker
    if dispatcher_factory is None:
        from fleet.model_request_dispatcher import ModelRequestDispatcher

        dispatcher_factory = ModelRequestDispatcher
    if not callable(worker_factory) or not callable(dispatcher_factory):
        raise TypeError("visual worker and dispatcher factories must be callable")

    raw_workers: dict[str, object] = {}
    dispatcher: object | None = None
    close_timeout_s = float(prepared.config.model_worker.request_timeout_s)
    try:
        for uav_id in sorted(expected_uav_ids):
            raw_workers[uav_id] = worker_factory(
                prepared.visual_clients[uav_id],
                uav_id=uav_id,
            )
        dispatcher = dispatcher_factory(
            broker,
            raw_workers,
            adapter_selection=prepared.runtime_visual_selection,
            record_logger=record_logger,
        )
        worker_for = getattr(dispatcher, "worker_for", None)
        if not callable(worker_for):
            raise TypeError("dispatcher must provide worker_for()")
        facades = {
            assignment.uav_id: worker_for(
                assignment.uav_id,
                assignment_id=assignment.assignment_id,
            )
            for assignment in prepared.plan.assignments
        }
        return dispatcher, MappingProxyType(facades)
    except BaseException:
        owners = (dispatcher,) if dispatcher is not None else tuple(raw_workers.values())
        for owner in owners:
            close = getattr(owner, "close", None)
            if callable(close):
                try:
                    close(timeout_s=close_timeout_s)
                except BaseException:
                    pass
        raise


def _not_called_model_record(
    selection: Mapping[str, object],
    *,
    fleet_mission_id: str,
    call_id: str,
) -> dict[str, object]:
    """Represent a Scripted role selection without pretending a call ran."""

    return {
        "call_id": call_id,
        "call_role": selection["call_role"],
        "fleet_mission_id": fleet_mission_id,
        "assignment_id": None,
        "uav_id": None,
        "requested_adapter": selection["requested_adapter"],
        "adapter_status": selection["adapter_status"],
        "effective_model": selection["effective_model"],
        "fallback_used": selection["fallback_used"],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_s": 0.0,
        "finish_reason": "not_called",
        "error_code": None,
        "stale_reasons": [],
    }


def _write_prepared_logs(logger: object, prepared: PreparedFleetMission, args: argparse.Namespace) -> None:
    logger.write_run_manifest(
        {
            "schema_version": 1,
            "fleet_mission_id": prepared.request.fleet_mission_id,
            "config_path": str(Path(args.config).expanduser().resolve()),
            "fleet_planner": args.fleet_planner,
            "fleet_planner_source": prepared.fleet_planner_source,
            "local_planner": args.local_planner,
            "planning_contract": args.planning_contract,
            "runtime_program": args.runtime_program,
            "perception_runtime_profile": args.perception_runtime_profile,
            "acknowledge_privileged_oracle": bool(
                args.acknowledge_privileged_oracle
            ),
            "qwen_visual_review_enabled": bool(args.enable_qwen_vision),
            "vision_review_mode": prepared.vision_review_mode,
            "headless": prepared.headless,
            "debug_visualization": bool(args.debug_visualization),
            "max_sim_time_s": float(args.max_sim_time),
            "adapter_selections": [
                dict(selection) for selection in prepared.adapter_selections
            ],
        }
    )
    for record in prepared.model_call_records:
        logger.log_model_call(record)
    logger.write_fleet_plan(prepared.plan)
    logger.write_assignments(_prepared_assignment_rows(prepared, status="PENDING"))
    for uav_id, result in prepared.compilations.items():
        logger.write_local_plan(
            uav_id,
            result.agent_request.local_plan_version,
            {
                "agent_planner_request": result.agent_request.to_dict(),
                "spatial_plan_draft_v3": result.planner_output.to_dict(),
                "compiled_task_plan": result.compiled_mission.task_plan.to_dict(),
            },
        )


def _prepared_assignment_rows(
    prepared: PreparedFleetMission,
    *,
    status: str,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "assignment_id": assignment.assignment_id,
            "uav_id": assignment.uav_id,
            "target_alias": assignment.target_alias,
            "priority": assignment.priority,
            "status": status,
            "local_plan_version": prepared.compilations[
                assignment.uav_id
            ].agent_request.local_plan_version,
        }
        for assignment in prepared.plan.assignments
    )


_TERMINAL_SECRET_PATTERNS = (
    re.compile(r"(?i)(--api-key(?:=|\s+))\S+"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|access[_-]?token|password|passwd|"
        r"client[_-]?secret)\b\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@"),
)


def _redact_terminal_error(value: object) -> str:
    """Remove credential-shaped text before stderr or persistent logging."""

    result = str(value)
    result = _TERMINAL_SECRET_PATTERNS[0].sub(r"\1[REDACTED]", result)
    result = _TERMINAL_SECRET_PATTERNS[1].sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        result,
    )
    result = _TERMINAL_SECRET_PATTERNS[2].sub("Bearer [REDACTED]", result)
    result = _TERMINAL_SECRET_PATTERNS[3].sub("sk-[REDACTED]", result)
    result = _TERMINAL_SECRET_PATTERNS[4].sub(r"\1[REDACTED]@", result)
    return result


def _append_terminal_error(current: str | None, value: object) -> str:
    addition = _redact_terminal_error(value)
    return addition if current is None else f"{current}; {addition}"


def _apply_terminal_outcome(
    summary: dict[str, object],
    *,
    exit_code: int,
    terminal_error: str | None,
    interrupted: bool,
) -> None:
    if terminal_error is not None:
        runtime_status = str(summary.get("status", "UNKNOWN"))
        summary.setdefault("shutdown_runtime_status", runtime_status)
        summary["status"] = "CANCELED" if interrupted else "FAILED"
        summary["last_error"] = _redact_terminal_error(terminal_error)
    summary["exit_code"] = exit_code
    summary["interrupted"] = interrupted


def _terminal_log_payload(
    prepared: PreparedFleetMission,
    runtime: object | None,
    *,
    exit_code: int,
    terminal_error: str | None,
    interrupted: bool,
) -> tuple[tuple[Mapping[str, object], ...], dict[str, object]]:
    """Build one bounded summary for success, failure, or interruption."""

    terminal_error = (
        None
        if terminal_error is None
        else _redact_terminal_error(terminal_error)
    )
    prepared_rows = {
        str(row["assignment_id"]): dict(row)
        for row in _prepared_assignment_rows(prepared, status="PENDING")
    }
    if runtime is None:
        terminal_status = "CANCELED" if interrupted else "FAILED"
        rows = tuple(
            {**row, "status": terminal_status}
            for row in prepared_rows.values()
        )
        summary: dict[str, object] = {
            "status": terminal_status,
            "fleet_mission_id": prepared.request.fleet_mission_id,
            "fleet_plan_version": prepared.plan.fleet_plan_version,
            "agent_plan_versions": {
                uav_id: result.agent_request.local_plan_version
                for uav_id, result in prepared.compilations.items()
            },
            "agent_statuses": {
                uav_id: "NOT_STARTED" for uav_id in prepared.compilations
            },
            "assignments": {
                str(row["assignment_id"]): dict(row) for row in rows
            },
            "last_airspace_decision": None,
            "event_count": 0,
            "last_error": terminal_error,
            "target_registry": {
                "claim_policy": (
                    prepared.plan.coordination_policy.target_claim_policy.value
                ),
                "targets": {},
                "event_count": 0,
            },
            "model_broker": {
                "pending": [],
                "inflight": [],
                "log_count": 0,
            },
        }
    else:
        snapshot = runtime.snapshot()
        runtime_rows = {
            str(assignment_id): dict(row)
            for assignment_id, row in snapshot.assignments.items()
            if str(assignment_id) in prepared_rows
        }
        missing_status = (
            "CANCELED"
            if interrupted
            else "FAILED"
            if terminal_error is not None
            else "PENDING"
        )
        rows = tuple(
            (
                {**prepared_row, **runtime_rows[assignment_id]}
                if assignment_id in runtime_rows
                else {**prepared_row, "status": missing_status}
            )
            for assignment_id, prepared_row in prepared_rows.items()
        )
        summary = {
            **snapshot.to_summary_dict(),
            "fleet_mission_id": prepared.request.fleet_mission_id,
            "fleet_plan_version": prepared.plan.fleet_plan_version,
            "agent_plan_versions": {
                uav_id: snapshot.agent_plan_versions.get(
                    uav_id,
                    result.agent_request.local_plan_version,
                )
                for uav_id, result in prepared.compilations.items()
            },
            "agent_statuses": {
                uav_id: snapshot.agent_statuses.get(uav_id, "NOT_STARTED")
                for uav_id in prepared.compilations
            },
            "assignments": {
                str(row["assignment_id"]): dict(row) for row in rows
            },
            "target_registry": runtime.targets.summary_snapshot(),
            "model_broker": runtime.model_broker.summary_snapshot(),
        }
    _apply_terminal_outcome(
        summary,
        exit_code=exit_code,
        terminal_error=terminal_error,
        interrupted=interrupted,
    )
    return tuple(rows), summary


def _drain_runtime_logs(
    logger: object,
    visual_coordinators: Mapping[str, object],
    visual_log_cursors: dict[str, int],
    managers: Mapping[str, object],
    transition_log_cursors: dict[str, int],
) -> None:
    """Persist every record exactly once, including records emitted at shutdown."""

    for uav_id in sorted(visual_coordinators):
        records = tuple(visual_coordinators[uav_id].records)
        cursor = visual_log_cursors.get(uav_id, 0)
        if cursor > len(records):
            raise RuntimeError(f"visual log moved backwards for {uav_id}")
        while cursor < len(records):
            logger.log_visual_review(uav_id, records[cursor])
            cursor += 1
            visual_log_cursors[uav_id] = cursor

    for uav_id in sorted(managers):
        records = tuple(managers[uav_id].transition_log)
        cursor = transition_log_cursors.get(uav_id, 0)
        if cursor > len(records):
            raise RuntimeError(f"transition log moved backwards for {uav_id}")
        while cursor < len(records):
            logger.log_agent_transition(uav_id, records[cursor])
            cursor += 1
            transition_log_cursors[uav_id] = cursor


def _best_effort_cancel_and_land(
    runtime: object,
    simulation_app: object,
    clock: _FleetSimulationClock,
    *,
    guard_s: float = 120.0,
) -> None:
    """Keep advancing an interrupted Fleet until fail-safe LAND terminates."""

    from fleet.runtime import FleetStatus

    if getattr(runtime, "status", None) is not FleetStatus.RUNNING:
        return
    if not bool(getattr(runtime, "cancel_requested", False)):
        runtime.cancel()
    deadline_s = clock.now() + guard_s
    while (
        simulation_app.is_running()
        and runtime.status is FleetStatus.RUNNING
        and clock.now() < deadline_s
    ):
        runtime.tick()
    if runtime.status is FleetStatus.RUNNING:
        raise RuntimeError("Fleet cancel-and-land exceeded shutdown guard")


def _close_execution_resources(
    runtime: object | None,
    environment: object | None,
    simulation_app: object | None,
) -> None:
    """Close the runtime owner and always release ``SimulationApp`` last."""

    first_error: BaseException | None = None
    owner = runtime if runtime is not None else environment
    close_owner = getattr(owner, "close", None)
    if callable(close_owner):
        try:
            close_owner()
        except BaseException as exc:
            first_error = exc
            # FleetMissionRuntime normally owns Environment.close().  If an
            # earlier Agent.close() failed, explicitly attempt the environment
            # so World/Camera resources are still released before Isaac exits.
            if runtime is not None and environment is not None:
                close_environment = getattr(environment, "close", None)
                if callable(close_environment):
                    try:
                        close_environment()
                    except BaseException:
                        # Preserve the first failure; SimulationApp still must
                        # be closed after this best-effort fallback.
                        pass
    if simulation_app is not None:
        try:
            simulation_app.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _create_fleet_debug_visualizer(
    plan: FleetMissionPlan,
    agent_plan_versions: Mapping[str, int | None],
    *,
    headless: bool,
    draw_factory: object | None = None,
    overlay_factory: object | None = None,
) -> object:
    """Build viewport-only Fleet geometry and GUI text after Isaac startup."""

    if not isinstance(headless, bool):
        raise TypeError("headless must be bool")
    if draw_factory is None:
        from visualization import FleetDebugDraw

        draw_factory = FleetDebugDraw
    if not callable(draw_factory):
        raise TypeError("draw_factory must be callable")
    status_overlay = None
    if not headless:
        if overlay_factory is None:
            from visualization import FleetStatusOverlay

            overlay_factory = FleetStatusOverlay
        if not callable(overlay_factory):
            raise TypeError("overlay_factory must be callable")
        status_overlay = overlay_factory()
    try:
        visualizer = draw_factory(status_overlay=status_overlay)
    except BaseException:
        # Ownership transfers to FleetDebugDraw only after its constructor
        # returns.  Do not leak the omni.ui window on constructor failure.
        close_overlay = getattr(status_overlay, "close", None)
        if callable(close_overlay):
            close_overlay()
        raise
    try:
        set_plan = getattr(visualizer, "set_plan", None)
        if not callable(set_plan):
            raise TypeError("draw_factory must return an object with set_plan()")
        set_plan(plan, agent_plan_versions=agent_plan_versions)
    except BaseException:
        close = getattr(visualizer, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                # Preserve the construction/set_plan failure that explains
                # why viewport setup could not complete.
                pass
        else:
            close_overlay = getattr(status_overlay, "close", None)
            if callable(close_overlay):
                close_overlay()
        raise
    return visualizer


def _finalize_fleet_execution(
    prepared: PreparedFleetMission,
    logger: object,
    *,
    runtime: object | None,
    simulation_app: object | None,
    environment: object | None,
    clock: _FleetSimulationClock | None,
    debug_draw: object | None,
    workers: Sequence[object],
    visual_coordinators: Mapping[str, object],
    visual_log_cursors: dict[str, int],
    managers: Mapping[str, object],
    transition_log_cursors: dict[str, int],
    exit_code: int,
    terminal_error: str | None,
    interrupted: bool,
    print_terminal_summary: bool,
    primary_error_present: bool,
) -> int:
    """Finish one run without allowing cleanup/log failures to mask its cause."""

    def record_finalization_error(label: str, exc: BaseException) -> None:
        nonlocal exit_code, interrupted, terminal_error
        detail = _redact_terminal_error(f"{label}: {type(exc).__name__}: {exc}")
        terminal_error = _append_terminal_error(terminal_error, detail)
        if isinstance(exc, KeyboardInterrupt) and not primary_error_present:
            interrupted = True
            exit_code = 130
        elif not primary_error_present and not interrupted:
            exit_code = 1
        print(f"[Fleet] {detail}", file=sys.stderr)

    if runtime is not None and clock is not None and simulation_app is not None:
        try:
            _best_effort_cancel_and_land(runtime, simulation_app, clock)
        except BaseException as exc:
            record_finalization_error("best-effort cancel-and-land failed", exc)

    try:
        _drain_runtime_logs(
            logger,
            visual_coordinators,
            visual_log_cursors,
            managers,
            transition_log_cursors,
        )
    except BaseException as exc:
        record_finalization_error("terminal runtime log drain failed", exc)

    assignment_rows: tuple[Mapping[str, object], ...] | None = None
    final_summary: dict[str, object] | None = None
    try:
        assignment_rows, final_summary = _terminal_log_payload(
            prepared,
            runtime,
            exit_code=exit_code,
            terminal_error=terminal_error,
            interrupted=interrupted,
        )
    except BaseException as exc:
        record_finalization_error("terminal snapshot failed", exc)
        try:
            assignment_rows, final_summary = _terminal_log_payload(
                prepared,
                None,
                exit_code=exit_code,
                terminal_error=terminal_error,
                interrupted=interrupted,
            )
        except BaseException as fallback_exc:
            record_finalization_error(
                "prepared terminal snapshot fallback failed",
                fallback_exc,
            )

    if debug_draw is not None:
        close_debug_draw = getattr(debug_draw, "close", None)
        if callable(close_debug_draw):
            try:
                close_debug_draw()
            except BaseException as exc:
                record_finalization_error("debug visualization cleanup failed", exc)
    for worker in workers:
        close_worker = getattr(worker, "close", None)
        if callable(close_worker):
            try:
                close_worker()
            except BaseException as exc:
                record_finalization_error("visual worker cleanup failed", exc)
    try:
        _close_execution_resources(runtime, environment, simulation_app)
    except BaseException as exc:
        record_finalization_error("execution resource cleanup failed", exc)

    # Dispatcher.close() can complete or stale outstanding model work after
    # the first terminal snapshot was built.  Refresh only the bounded Broker
    # scheduler summary so pending/inflight/log_count describe durable final
    # model_calls.csv state rather than the pre-cleanup instant.
    if final_summary is not None and runtime is not None:
        try:
            final_summary["model_broker"] = (
                runtime.model_broker.summary_snapshot()
            )
        except BaseException as exc:
            record_finalization_error("terminal Broker summary refresh failed", exc)

    if final_summary is not None:
        _apply_terminal_outcome(
            final_summary,
            exit_code=exit_code,
            terminal_error=terminal_error,
            interrupted=interrupted,
        )

    if assignment_rows is not None:
        try:
            logger.write_assignments(assignment_rows)
        except BaseException as exc:
            record_finalization_error("terminal assignments write failed", exc)
            if final_summary is not None:
                _apply_terminal_outcome(
                    final_summary,
                    exit_code=exit_code,
                    terminal_error=terminal_error,
                    interrupted=interrupted,
                )

    summary_written = False
    if final_summary is not None:
        try:
            logger.write_summary(final_summary)
            summary_written = True
        except BaseException as exc:
            record_finalization_error("terminal summary write failed", exc)

    if print_terminal_summary and summary_written:
        try:
            print(json.dumps(final_summary, ensure_ascii=False, indent=2))
        except Exception as exc:
            # The durable summary is authoritative; a closed stdout pipe must
            # not retroactively change a fully recorded mission outcome.
            print(
                "[Fleet] terminal summary display failed: "
                + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
                file=sys.stderr,
            )
    return exit_code


def run_prepared_fleet_mission(
    prepared: PreparedFleetMission,
    args: argparse.Namespace,
) -> int:
    """Cross the Isaac boundary and execute an already prepared mission."""

    # Create the sparse run record before crossing the Isaac boundary so an
    # import, SimulationApp, or environment setup failure is still auditable.
    logger = FleetMissionLogger(
        args.output_root,
        prepared.request.fleet_mission_id,
        uav_ids=tuple(prepared.compilations),
    )
    _write_prepared_logs(logger, prepared, args)

    simulation_app = None
    environment = None
    runtime = None
    clock = None
    debug_draw = None
    workers: list[object] = []
    visual_coordinators: dict[str, object] = {}
    visual_log_cursors: dict[str, int] = {}
    managers: dict[str, object] = {}
    transition_log_cursors: dict[str, int] = {}
    exit_code = 1
    terminal_error: str | None = None
    interrupted = False
    print_terminal_summary = False
    primary_error: BaseException | None = None
    primary_traceback = None
    try:
        # FIRST ISAAC IMPORT.  All planning, validation, compilation, model
        # calls, perception preflight, Safety preflight, and initial logging
        # have completed above this line.
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": prepared.headless})
        if args.debug_visualization:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.util.debug_draw")
            simulation_app.update()
        from agents.mission_agent import MissionAgent
        from env.fleet_uav_search_env import FleetUavSearchEnv
        from fleet.airspace_manager import (
            FleetAirspaceManager,
            coerce_fleet_pose_snapshot,
        )
        from fleet.model_request_broker import GlobalModelRequestBroker
        from fleet.runtime import FleetMissionRuntime, FleetStatus
        from fleet.target_registry import SharedTargetRegistry
        from perception.factory import build_target_perception_backend
        from perception.runtime import (
            GuardedPerceptionBackend,
            PerceptionRuntimeProfile,
        )
        from runtime.route_registry import RouteRegistry
        from skills.manager import SkillManager, create_default_skill_registry
        from target.target_manager import TargetManager

        assignments = {
            assignment.uav_id: assignment.target_alias
            for assignment in prepared.plan.assignments
        }
        environment = FleetUavSearchEnv(
            prepared.config,
            assignments=assignments,
        )
        environment.setup()
        if not prepared.headless and environment.scene is not None:
            environment.scene.configure_overview_viewport()

        profile = (
            PerceptionRuntimeProfile.ORACLE_EVALUATION
            if args.perception_runtime_profile == "oracle_evaluation"
            else PerceptionRuntimeProfile.PRODUCTION
        )
        agents: dict[str, object] = {}
        perceptions: dict[str, object] = {}
        clock = _FleetSimulationClock(environment)

        # One Broker owns admission for the entire Fleet runtime.  Visual
        # clients remain isolated per UAV, but no AsyncModelWorker is exposed
        # directly to a coordinator: each receives only its Broker facade.
        broker = GlobalModelRequestBroker(
            max_inflight_global=(
                prepared.config.model_broker.max_inflight_global
            ),
            max_inflight_per_uav=(
                prepared.config.model_broker.max_inflight_per_uav
            ),
            max_pending_per_uav=(
                prepared.config.model_broker.max_pending_per_uav
            ),
            starvation_timeout_s=(
                prepared.config.model_broker.starvation_timeout_s
            ),
        )
        brokered_visual_workers: Mapping[str, object] = {}
        if args.enable_qwen_vision:
            dispatcher, brokered_visual_workers = (
                _build_brokered_visual_workers(prepared, broker, logger)
            )
            workers.append(dispatcher)
        for assignment in prepared.plan.assignments:
            uav_id = assignment.uav_id
            if profile is PerceptionRuntimeProfile.ORACLE_EVALUATION:
                raw_perception = environment.make_oracle_perception(uav_id)
                if (
                    raw_perception.uav_id != uav_id
                    or raw_perception.target_id != assignment.target_alias
                ):
                    raise RuntimeError("Oracle backend is outside its assignment")
                runtime_perception = GuardedPerceptionBackend(
                    raw_perception,
                    profile=profile,
                    acknowledge_privileged_oracle=True,
                )
                backend_name = "oracle_evaluation"
            else:
                raw_perception = build_target_perception_backend(
                    prepared.config,
                    runtime_profile=profile,
                    acknowledge_privileged_oracle=False,
                    uav_id=uav_id,
                )
                runtime_perception = raw_perception
                backend_name = prepared.config.target_perception.backend
            context = environment.make_skill_context(
                uav_id,
                clock,
                # Apply the same information-policy boundary to direct
                # SkillContext access and to FleetRuntime observations.
                perception=runtime_perception,
            )
            manager = SkillManager(
                context,
                registry=create_default_skill_registry(
                    transit_yaw_mode=prepared.config.search.transit_yaw_mode,
                ),
                route_registry=RouteRegistry(),
            )
            managers[uav_id] = manager
            transition_log_cursors[uav_id] = 0
            target_manager = TargetManager()
            visual_coordinator = None
            if args.enable_qwen_vision:
                visual_coordinator = _build_visual_review_coordinator(
                    prepared=prepared,
                    uav_id=uav_id,
                    manager=manager,
                    target_manager=target_manager,
                    worker=brokered_visual_workers[uav_id],
                )
                visual_coordinators[uav_id] = visual_coordinator
                visual_log_cursors[uav_id] = 0
            world_context = prepared.world_contexts[uav_id]
            home_name = prepared.request.uav(uav_id).home_name
            validator = PlanValidator(
                prepared.planner_limits,
                prepared.planner_policy,
                spatial_resolver=_spatial_resolver(world_context, home_name),
            )
            safety = SafetySupervisor(
                world_context.scene_min_xyz_m,
                world_context.scene_max_xyz_m,
                max_mission_time_s=float(args.max_sim_time) + 120.0,
                position_margin_m=0.25,
                max_safe_altitude_m=world_context.scene_max_xyz_m[2],
                planner_limits=prepared.planner_limits,
            )
            replay = RoutedPreplannedSpatialPlanner(
                prepared.compilations[uav_id].planner_output,
                source=prepared.local_planner_source,
            )
            agents[uav_id] = MissionAgent(
                replay,
                validator,
                safety,
                manager,
                target_manager,
                clock,
                perception_runtime_profile=profile,
                acknowledge_privileged_oracle=bool(
                    args.acknowledge_privileged_oracle
                ),
                uav_id=uav_id,
                visual_review_coordinator=visual_coordinator,
                runtime_program=args.runtime_program,
                target_perception_backend=backend_name,
            )
            perceptions[uav_id] = runtime_perception

        targets = SharedTargetRegistry(
            prepared.plan.coordination_policy.target_claim_policy.value
        )
        airspace = FleetAirspaceManager(
            prepared.plan.coordination_policy.minimum_uav_separation_m,
            policy=prepared.plan.coordination_policy.route_conflict_policy.value,
        )
        runtime = FleetMissionRuntime(
            environment,
            RoutedPreplannedFleetPlanner(
                prepared.request,
                prepared.plan,
                source=prepared.fleet_planner_source,
            ),
            agents,
            assignment_compiler=prepared.compiler,
            world_contexts=prepared.world_contexts,
            perceptions=perceptions,
            planned_routes=prepared.planned_routes,
            targets=targets,
            airspace=airspace,
            model_broker=broker,
            logger=logger,
        )
        runtime.start(args.instruction.strip(), request=prepared.request)

        # Persist the real NONE -> TAKEOFF records produced by MissionAgent;
        # later drains append only newly emitted records.
        _drain_runtime_logs(
            logger,
            visual_coordinators,
            visual_log_cursors,
            managers,
            transition_log_cursors,
        )

        if args.debug_visualization:
            debug_draw = _create_fleet_debug_visualizer(
                prepared.plan,
                runtime.snapshot().agent_plan_versions,
                headless=prepared.headless,
            )

        cancel_requested = False
        shutdown_deadline_s = float(args.max_sim_time) + 120.0
        last_printed_second = -1
        while simulation_app.is_running() and runtime.status is FleetStatus.RUNNING:
            snapshot = runtime.tick()
            now_s = clock.now()
            whole_second = int(now_s)
            if whole_second != last_printed_second:
                last_printed_second = whole_second
                statuses = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(snapshot.agent_statuses.items())
                )
                print(
                    f"[Fleet] t={now_s:.1f}s status={snapshot.status.value} {statuses}"
                )
            if (
                not cancel_requested
                and runtime.status is FleetStatus.RUNNING
                and now_s >= float(args.max_sim_time)
            ):
                print(
                    "[Fleet] max-sim-time reached; requesting cancel-and-land",
                    file=sys.stderr,
                )
                runtime.cancel()
                cancel_requested = True
            if cancel_requested and now_s >= shutdown_deadline_s:
                raise RuntimeError("Fleet cancel-and-land exceeded shutdown guard")

            _drain_runtime_logs(
                logger,
                visual_coordinators,
                visual_log_cursors,
                managers,
                transition_log_cursors,
            )

            if debug_draw is not None:
                try:
                    pose_snapshot = coerce_fleet_pose_snapshot(
                        environment.get_fleet_pose_snapshot()
                    )
                    fleet_pose = environment.get_fleet_pose_snapshot()
                    target_positions = {
                        target_id: (
                            state.x,
                            state.y,
                            state.z,
                        )
                        for target_id, state in fleet_pose.target_states.items()
                    }
                    debug_draw.update(
                        poses=pose_snapshot,
                        target_records=runtime.targets.records,
                        target_positions_world_m=target_positions,
                        airspace_decision=runtime.airspace.last_decision,
                        agent_plan_versions=snapshot.agent_plan_versions,
                    )
                except RuntimeError:
                    # Camera/pose warm-up may leave no complete first snapshot.
                    pass

        final_snapshot = runtime.snapshot()
        exit_code = 0 if final_snapshot.status is FleetStatus.SUCCEEDED else 1
        print_terminal_summary = True
    except KeyboardInterrupt:
        print("[Fleet] interrupted", file=sys.stderr)
        terminal_error = "KeyboardInterrupt: fleet mission interrupted"
        interrupted = True
        exit_code = 130
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
        terminal_error = _redact_terminal_error(
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        try:
            exit_code = _finalize_fleet_execution(
                prepared,
                logger,
                runtime=runtime,
                simulation_app=simulation_app,
                environment=environment,
                clock=clock,
                debug_draw=debug_draw,
                workers=workers,
                visual_coordinators=visual_coordinators,
                visual_log_cursors=visual_log_cursors,
                managers=managers,
                transition_log_cursors=transition_log_cursors,
                exit_code=exit_code,
                terminal_error=terminal_error,
                interrupted=interrupted,
                print_terminal_summary=print_terminal_summary,
                primary_error_present=primary_error is not None,
            )
        except BaseException as exc:
            # This is a final containment boundary.  The helper handles every
            # expected stage independently; an internal bug here still cannot
            # replace the original mission exception or an earlier Ctrl+C.
            print(
                "[Fleet] terminal finalization failed unexpectedly: "
                + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
                file=sys.stderr,
            )
            if primary_error is None and not interrupted:
                exit_code = 1
    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        prepared = prepare_fleet_mission(args)
    except Exception as exc:
        print(
            "fleet launch configuration error: "
            + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
            file=sys.stderr,
        )
        return 2
    try:
        return run_prepared_fleet_mission(prepared, args)
    except Exception as exc:
        print(
            "fleet mission failed: "
            + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
