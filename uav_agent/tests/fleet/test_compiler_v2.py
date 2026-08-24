from __future__ import annotations

import json

import pytest

from fleet.compiler import AssignmentCompilerError, FleetAssignmentCompiler
from fleet.task_spec import (
    ConstraintStrength,
    FleetTaskSpecV1,
    GoalType,
    MissionGoal,
)
from fleet.types import (
    FleetCoordinationPolicy,
    FleetStartPolicy,
    FleetUavCapability,
)
from fleet.types_v2 import (
    AssignmentCompilationV2,
    FleetAssignmentV2,
    FleetMissionPlanV2,
    FleetMissionRequestV2,
)
from planner.schemas import LandingZoneSpec, PlannerRequest, PlannerWorldContext
from planner.schemas_v3 import SkillPlanDraftV3
from planner.spatial import CircleRegion, CoordinateFrame
from runtime.plan_validator import PlanValidator
from runtime.validation_codes import ValidationCode
from runtime.validation_report import ValidationSeverity
from target.types import TargetSpec


class _GoalDrivenSpatialPlanner:
    source = "dynamic_llm"

    def __init__(self, mode: str = "cover") -> None:
        self.mode = mode
        self.requests: list[PlannerRequest] = []

    def plan(self, request: PlannerRequest) -> SkillPlanDraftV3:
        self.requests.append(request)
        payload = json.loads(request.instruction)
        goals = payload["assigned_goals"]
        specs = payload["trusted_target_specs"]
        target_spec = next(iter(specs.values()), None)
        output_uav = "uav_tampered" if self.mode == "routing" else request.uav_id
        output_mission = (
            "mission_tampered" if self.mode == "routing" else request.mission_id
        )
        steps: list[dict[str, object]] = [
            {
                "id": "takeoff_1",
                "uav_id": output_uav,
                "skill": "TAKEOFF",
                "args": {"altitude_m": 10},
            }
        ]
        search_id: str | None = None
        if self.mode != "omit" and any(
            goal["goal_type"] == "SEARCH_TARGET" for goal in goals
        ):
            goal = next(
                item for item in goals if item["goal_type"] == "SEARCH_TARGET"
            )
            search_id = "search_1"
            steps.append(
                {
                    "id": search_id,
                    "uav_id": output_uav,
                    "skill": "SEARCH",
                    "args": {
                        "region": goal["spatial_constraint"],
                        "strategy": {"kind": "SPIRAL_OUT", "spacing_m": 4},
                        "entry_policy": "START_IN_PLACE_IF_INSIDE",
                        "target_description": (
                            "target"
                            if target_spec is None
                            else target_spec["original_description"]
                        ),
                        "search_altitude_m": 10,
                        "timeout_s": 60,
                    },
                }
            )
        if self.mode != "omit" and any(
            goal["goal_type"] == "TRACK_TARGET" for goal in goals
        ):
            goal = next(
                item for item in goals if item["goal_type"] == "TRACK_TARGET"
            )
            steps.append(
                {
                    "id": "track_1",
                    "uav_id": output_uav,
                    "skill": "TRACK",
                    "args": {
                        "target_ref": (
                            "$trusted_target.target_id"
                            if search_id is None
                            else f"${search_id}.target_id"
                        ),
                        "duration_s": goal["duration_s"],
                    },
                }
            )
        home = payload["own_home"]
        steps.extend(
            (
                {
                    "id": "goto_home",
                    "uav_id": output_uav,
                    "skill": "GOTO",
                    "args": {
                        "target": {"kind": "NAMED_LOCATION", "name": home}
                    },
                },
                {
                    "id": "land_1",
                    "uav_id": output_uav,
                    "skill": "LAND",
                    "args": {"zone": home},
                },
            )
        )
        if self.mode == "untrusted_target":
            target_spec = TargetSpec("invented target").to_dict()
        raw: dict[str, object] = {
            "schema_version": 3,
            "mission_id": output_mission,
            "uav_id": output_uav,
            "plan_version": request.plan_version,
            "assumptions": [],
            "steps": steps,
        }
        if target_spec is not None:
            raw["target_spec"] = target_spec
        return SkillPlanDraftV3.from_dict(raw)


def _uav(uav_id: str) -> FleetUavCapability:
    return FleetUavCapability(
        uav_id=uav_id,
        display_name=uav_id.upper(),
        available=True,
        home_name=f"home_{uav_id[-1]}",
        max_speed_mps=5,
        max_altitude_m=30,
    )


def _goal(
    goal_id: str,
    goal_type: GoalType,
    *,
    alias: str = "target_i",
    duration_s: float | None = None,
) -> MissionGoal:
    return MissionGoal(
        goal_id=goal_id,
        goal_type=goal_type,
        target_alias=alias,
        spatial_constraint=(
            CircleRegion(CoordinateFrame.WORLD_ENU, (15, 5, 0), 8)
            if goal_type is GoalType.SEARCH_TARGET
            else None
        ),
        duration_s=duration_s,
        distance_m=None,
        strength=ConstraintStrength.MUST,
    )


def _contracts(
    *goals: MissionGoal,
    assigned_goal_ids: tuple[str, ...] | None = None,
) -> tuple[FleetMissionRequestV2, FleetMissionPlanV2, FleetAssignmentV2]:
    task = FleetTaskSpecV1(
        source_text="Search target_i and track target_i when requested.",
        goals=tuple(goals),
    )
    policy = FleetCoordinationPolicy()
    request = FleetMissionRequestV2(
        fleet_mission_id="fleet_mission_v2_compile",
        fleet_plan_version=2,
        task_spec=task,
        uav_inventory=(_uav("uav_a"), _uav("uav_b")),
        trusted_fleet_state=(),
        coordination_policy=policy,
    )
    goal_ids = assigned_goal_ids or tuple(goal.goal_id for goal in goals)
    assignment = FleetAssignmentV2(
        assignment_id="assignment_a",
        uav_id="uav_a",
        goal_ids=goal_ids,
        priority=100,
        start_policy=FleetStartPolicy.PARALLEL,
    )
    plan = FleetMissionPlanV2(
        fleet_mission_id=request.fleet_mission_id,
        fleet_plan_version=request.fleet_plan_version,
        assignments=(assignment,),
        coordination_policy=policy,
        unassigned_goal_ids=tuple(
            goal.goal_id for goal in goals if goal.goal_id not in goal_ids
        ),
    )
    return request, plan, assignment


def _world() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-100, -100, 0),
        scene_max_xyz_m=(100, 100, 40),
        initial_uav_xyz_m=(0, 0, 0),
        search_regions={},
        landing_zones={
            "home_a": LandingZoneSpec("home_a", (0, 0), 0),
            "home_b": LandingZoneSpec("home_b", (20, 0), 0),
        },
        default_takeoff_altitude_m=10,
        default_track_duration_s=30,
        search_timeout_s=75,
    )


def _catalog() -> dict[str, TargetSpec]:
    return {
        "target_i": TargetSpec(
            "red target i", immutable_identity_summary="target i"
        ),
        "target_j": TargetSpec(
            "blue target j", immutable_identity_summary="target j"
        ),
    }


def test_v2_request_contains_only_assignment_goals_and_no_fixed_six_step_chain() -> None:
    search = _goal("goal_search", GoalType.SEARCH_TARGET)
    track = _goal("goal_track", GoalType.TRACK_TARGET, duration_s=12)
    request, plan, assignment = _contracts(
        search,
        track,
        assigned_goal_ids=(search.goal_id,),
    )
    planner = _GoalDrivenSpatialPlanner()
    compiler = FleetAssignmentCompiler(planner, validator=PlanValidator())

    result = compiler.compile_assignment_v2(
        request,
        plan,
        assignment,
        _world(),
        target_catalog=_catalog(),
    )

    assert isinstance(result, AssignmentCompilationV2)
    assert tuple(goal.goal_id for goal in result.agent_request.goals) == (
        "goal_search",
    )
    focused = json.loads(result.planner_request.instruction)
    assert [item["goal_id"] for item in focused["assigned_goals"]] == [
        "goal_search"
    ]
    assert "goal_track" not in result.planner_request.instruction
    assert "target_j" not in result.planner_request.instruction
    assert "Take off, search, track, return home, and land." not in (
        result.planner_request.instruction
    )
    assert focused["trusted_runtime_safety_completion"] is True
    assert result.planner_request.allow_trusted_safety_completion is True
    assert result.planner_request.world_context.search_regions == {}
    assert set(result.planner_request.world_context.landing_zones) == {"home_a"}
    assert result.executable is True
    assert result.semantically_valid is True
    assert all(step.skill != "TRACK" for step in result.planner_output.steps)


def test_v2_semantic_coverage_gap_retains_safe_compiled_plan() -> None:
    search = _goal("goal_search", GoalType.SEARCH_TARGET)
    request, plan, assignment = _contracts(search)
    compiler = FleetAssignmentCompiler(
        _GoalDrivenSpatialPlanner("omit"), validator=PlanValidator()
    )

    result = compiler.compile_assignment_v2(
        request,
        plan,
        assignment,
        _world(),
        target_catalog=_catalog(),
        proposal_id="proposal_1",
    )

    assert result.compiled_mission is not None
    assert result.executable is True
    assert result.semantically_valid is False
    assert result.uncovered_goal_ids == ("goal_search",)
    assert result.validation_report.hard_blocked is False
    assert any(
        finding.severity is ValidationSeverity.RECOVERABLE_SEMANTIC_ERROR
        and finding.code is ValidationCode.GOAL_NOT_COVERED
        for finding in result.validation_report.findings
    )


def test_v2_routing_mismatch_is_a_hard_block_without_executable_plan() -> None:
    search = _goal("goal_search", GoalType.SEARCH_TARGET)
    request, plan, assignment = _contracts(search)
    compiler = FleetAssignmentCompiler(
        _GoalDrivenSpatialPlanner("routing"), validator=PlanValidator()
    )

    result = compiler.compile_assignment_v2(
        request,
        plan,
        assignment,
        _world(),
        target_catalog=_catalog(),
    )

    assert result.compiled_mission is None
    assert result.validation_report.hard_blocked is True
    assert any(
        finding.code is ValidationCode.ROUTING_MISMATCH
        for finding in result.validation_report.findings
    )


def test_v2_untrusted_target_spec_is_a_hard_compiler_finding() -> None:
    search = _goal("goal_search", GoalType.SEARCH_TARGET)
    request, plan, assignment = _contracts(search)
    compiler = FleetAssignmentCompiler(
        _GoalDrivenSpatialPlanner("untrusted_target"), validator=PlanValidator()
    )

    result = compiler.compile_assignment_v2(
        request,
        plan,
        assignment,
        _world(),
        target_catalog=_catalog(),
    )

    assert result.compiled_mission is None
    assert result.validation_report.hard_blocked is True
    assert any(
        finding.code is ValidationCode.UNKNOWN_ENTITY
        and finding.stage == "LOCAL_COMPILATION"
        for finding in result.validation_report.findings
    )


def test_v2_trusted_target_lock_allows_track_without_search() -> None:
    track = _goal("goal_track", GoalType.TRACK_TARGET, duration_s=10)
    request, plan, assignment = _contracts(track)
    compiler = FleetAssignmentCompiler(
        _GoalDrivenSpatialPlanner(), validator=PlanValidator()
    )

    result = compiler.compile_assignment_v2(
        request,
        plan,
        assignment,
        _world(),
        target_catalog=_catalog(),
        trusted_target_id="target_locked_1",
    )

    assert result.executable is True
    assert result.semantically_valid is True
    assert all(step.skill != "SEARCH" for step in result.planner_output.steps)
    assert result.goal_coverage.coverages[0].evidence_step_ids == ("track_1",)


def test_v2_semantic_retry_payload_is_projected_and_bounded() -> None:
    search = _goal("goal_search", GoalType.SEARCH_TARGET)
    request, plan, assignment = _contracts(search)
    planner = _GoalDrivenSpatialPlanner()
    compiler = FleetAssignmentCompiler(planner, validator=PlanValidator())

    compiler.compile_assignment_v2(
        request,
        plan,
        assignment,
        _world(),
        target_catalog=_catalog(),
        semantic_repair_findings=(
            {
                "goal_id": "goal_search",
                "code": ValidationCode.GOAL_NOT_COVERED,
                "message": "Add a realizable path for this assigned Goal.",
                # Extra report metadata is intentionally not projected.
                "previous_prompt": "must not survive",
                "previous_output": {"steps": ["must not survive"]},
            },
        ),
    )
    focused = json.loads(planner.requests[0].instruction)
    assert focused["semantic_repair_findings"] == [
        {
            "goal_id": "goal_search",
            "code": "GOAL_NOT_COVERED",
            "message": "Add a realizable path for this assigned Goal.",
        }
    ]
    assert "previous_prompt" not in planner.requests[0].instruction
    assert "previous_output" not in planner.requests[0].instruction

    agent_request = compiler.build_agent_request_v2(
        request,
        plan,
        assignment,
        target_catalog=_catalog(),
    )
    with pytest.raises(AssignmentCompilerError, match="at most 32"):
        compiler.build_planner_request_v2(
            agent_request,
            _world(),
            request.uav_inventory[0],
            semantic_repair_findings=tuple(
                {
                    "goal_id": f"goal_{index}",
                    "code": "GOAL_NOT_COVERED",
                    "message": "missing",
                }
                for index in range(33)
            ),
        )


def test_v2_proposal_retry_payload_is_separate_projected_and_private_safe() -> None:
    search = _goal("goal_search", GoalType.SEARCH_TARGET)
    request, plan, assignment = _contracts(search)
    planner = _GoalDrivenSpatialPlanner()
    compiler = FleetAssignmentCompiler(planner, validator=PlanValidator())

    compiler.compile_assignment_v2(
        request,
        plan,
        assignment,
        _world(),
        target_catalog=_catalog(),
        semantic_repair_findings=(
            {
                "goal_id": "goal_search",
                "code": "GOAL_NOT_COVERED",
                "message": "Cover the assigned search Goal.",
            },
        ),
        proposal_repair_findings=(
            {
                "code": "INVALID_JSON",
                "message": "Emit one complete strict JSON object.\nNo Markdown.",
                "goal_id": "goal_search",
                "raw_output": "must not survive",
                "previous_prompt": "must not survive either",
            },
        ),
    )

    focused = json.loads(planner.requests[0].instruction)
    assert focused["proposal_repair_findings"] == [
        {
            "code": "INVALID_JSON",
            "message": "Emit one complete strict JSON object. No Markdown.",
        }
    ]
    assert focused["semantic_repair_findings"][0]["goal_id"] == "goal_search"
    assert "raw_output" not in planner.requests[0].instruction
    assert "previous_prompt" not in planner.requests[0].instruction

    agent_request = compiler.build_agent_request_v2(
        request,
        plan,
        assignment,
        target_catalog=_catalog(),
    )
    with pytest.raises(AssignmentCompilerError, match="at most 32"):
        compiler.build_planner_request_v2(
            agent_request,
            _world(),
            request.uav_inventory[0],
            proposal_repair_findings=tuple(
                {"code": "INVALID_JSON", "message": f"invalid {index}"}
                for index in range(33)
            ),
        )
    with pytest.raises(AssignmentCompilerError, match="private/media"):
        compiler.build_planner_request_v2(
            agent_request,
            _world(),
            request.uav_inventory[0],
            proposal_repair_findings=(
                {
                    "code": "SCHEMA_INVALID",
                    "message": "Copy private oracle base64 data into the retry.",
                },
            ),
        )
    with pytest.raises(AssignmentCompilerError, match="code is invalid"):
        compiler.build_planner_request_v2(
            agent_request,
            _world(),
            request.uav_inventory[0],
            proposal_repair_findings=(
                {
                    "code": "GOAL_NOT_COVERED",
                    "message": "This belongs to semantic Goal feedback.",
                },
            ),
        )
