from __future__ import annotations

from fleet.task_spec import (
    ConstraintStrength,
    GoalType,
    MissionGoal,
    TerminationGoal,
)
from planner.goal_checker import GoalSatisfactionChecker
from planner.schemas import LandingZoneSpec, PlannerWorldContext
from planner.schemas_v3 import SkillPlanDraftV3
from runtime.plan_validator import PlanValidator
from runtime.safety_supervisor import SafetyAction, SafetySupervisor
from runtime.validation_codes import ValidationCode
from runtime.validation_report import ValidationSeverity


def _goal(
    goal_id: str,
    goal_type: GoalType,
    *,
    duration_s: float | None = None,
) -> MissionGoal | TerminationGoal:
    if goal_type in {
        GoalType.RETURN_HOME,
        GoalType.LAND,
        GoalType.RETURN_HOME_AND_LAND,
        GoalType.WAIT,
        GoalType.REPORT,
    }:
        return TerminationGoal(
            goal_id=goal_id,
            goal_type=goal_type,
            uav_id=None,
            duration_s=duration_s,
            strength=ConstraintStrength.MUST,
        )
    return MissionGoal(
        goal_id=goal_id,
        goal_type=goal_type,
        target_alias=(
            "target_i"
            if goal_type
            in {GoalType.SEARCH_TARGET, GoalType.TRACK_TARGET, GoalType.INSPECT_TARGET}
            else None
        ),
        spatial_constraint=None,
        duration_s=duration_s,
        distance_m=None,
        strength=ConstraintStrength.MUST,
    )


def _plan(*steps: dict[str, object]) -> dict[str, object]:
    return {
        "mission_id": "mission_goal_test",
        "uav_id": "uav_a",
        "steps": list(steps),
    }


def _step(step_id: str, skill: str, **args: object) -> dict[str, object]:
    return {
        "id": step_id,
        "uav_id": "uav_a",
        "skill": skill,
        "args": args,
    }


def test_trusted_lock_allows_track_without_search() -> None:
    report = GoalSatisfactionChecker().check(
        [_goal("goal_track", GoalType.TRACK_TARGET, duration_s=10.0)],
        _plan(
            _step("takeoff", "TAKEOFF"),
            _step(
                "track",
                "TRACK",
                target_ref="$trusted_target.target_id",
                duration_s=10.0,
            ),
            _step("land", "LAND", zone="home"),
        ),
        mission_id="mission_goal_test",
        uav_id="uav_a",
        trusted_target_locked=True,
    )

    assert report.complete
    assert report.uncovered_goal_ids == ()
    assert report.validation_report.accepted
    assert report.coverages[0].evidence_step_ids == ("track",)


def test_track_without_trusted_lock_or_search_is_recoverable_semantic_error() -> None:
    report = GoalSatisfactionChecker().check(
        [_goal("goal_track", GoalType.TRACK_TARGET, duration_s=10.0)],
        _plan(
            _step("takeoff", "TAKEOFF"),
            _step("land", "LAND", zone="home"),
        ),
        mission_id="mission_goal_test",
        assignment_id="assignment_a",
        uav_id="uav_a",
    )

    assert not report.complete
    assert report.validation_report.executable
    assert not report.validation_report.semantically_valid
    finding = report.findings[0]
    assert finding.goal_id == "goal_track"
    assert finding.severity is ValidationSeverity.RECOVERABLE_SEMANTIC_ERROR
    assert finding.code is ValidationCode.GOAL_NOT_COVERED


def test_no_track_goal_does_not_require_track_or_search() -> None:
    report = GoalSatisfactionChecker().check(
        [_goal("goal_wait", GoalType.WAIT, duration_s=5.0)],
        _plan(
            _step("takeoff", "TAKEOFF"),
            _step("wait", "HOVER", duration_s=5.0),
            _step("land", "LAND", zone="home"),
        ),
        mission_id="mission_goal_test",
        uav_id="uav_a",
    )

    assert report.complete
    assert all(step_id != "track" for step_id in report.coverages[0].evidence_step_ids)


def test_search_to_track_is_a_possible_confirmed_runtime_path() -> None:
    report = GoalSatisfactionChecker().check(
        [
            _goal("goal_search", GoalType.SEARCH_TARGET),
            _goal("goal_track", GoalType.TRACK_TARGET, duration_s=12.0),
        ],
        _plan(
            _step("takeoff", "TAKEOFF"),
            _step("search", "SEARCH", region={"shape": "CIRCLE"}),
            _step("track", "TRACK", target_ref="$search.target_id", duration_s=12.0),
            _step("land", "LAND", zone="home"),
        ),
        mission_id="mission_goal_test",
        uav_id="uav_a",
    )

    assert report.complete
    assert report.validation_report.accepted


def test_short_track_duration_requests_repair_not_action_block() -> None:
    report = GoalSatisfactionChecker().check(
        [_goal("goal_track", GoalType.TRACK_TARGET, duration_s=20.0)],
        _plan(
            _step("takeoff", "TAKEOFF"),
            _step("search", "SEARCH"),
            _step("track", "TRACK", target_ref="$search.target_id", duration_s=5.0),
            _step("land", "LAND", zone="home"),
        ),
        mission_id="mission_goal_test",
        uav_id="uav_a",
    )

    assert report.validation_report.executable
    assert report.findings[0].code is ValidationCode.TRACK_DURATION_UNDERSHOOT


def test_unknown_skill_nonfinite_and_routing_mismatch_are_hard_blocks() -> None:
    plan = _plan(
        _step("takeoff", "TAKEOFF"),
        _step("unknown", "SET_MOTOR", pwm=0.5),
        _step("nan_hover", "HOVER", duration_s=float("nan")),
        {
            "id": "wrong_route",
            "uav_id": "uav_b",
            "skill": "LAND",
            "args": {"zone": "home"},
        },
    )
    report = GoalSatisfactionChecker().check(
        [],
        plan,
        mission_id="mission_goal_test",
        uav_id="uav_a",
    ).validation_report

    assert report.hard_blocked
    assert not report.executable
    assert {item.code for item in report.findings} == {
        ValidationCode.UNKNOWN_SKILL,
        ValidationCode.NON_FINITE_NUMBER,
        ValidationCode.ROUTING_MISMATCH,
    }
    assert all(
        item.severity is ValidationSeverity.HARD_ACTION_BLOCK
        for item in report.findings
    )


def test_validator_compiles_trusted_lock_but_blocks_untrusted_symbol() -> None:
    context = PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={},
        landing_zones={
            "home": LandingZoneSpec(
                name="home",
                position_xy_m=(0.0, 0.0),
                ground_altitude_m=0.0,
            )
        },
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=10.0,
        search_timeout_s=60.0,
    )
    draft = SkillPlanDraftV3.from_dict(
        {
            "schema_version": 3,
            "mission_id": "mission_goal_test",
            "uav_id": "uav_a",
            "plan_version": 1,
            "assumptions": [],
            "steps": [
                {
                    "id": "takeoff",
                    "uav_id": "uav_a",
                    "skill": "TAKEOFF",
                    "args": {"altitude_m": 10.0},
                },
                {
                    "id": "track",
                    "uav_id": "uav_a",
                    "skill": "TRACK",
                    "args": {
                        "target_ref": "$trusted_target.target_id",
                        "duration_s": 10.0,
                    },
                },
                {
                    "id": "goto_home",
                    "uav_id": "uav_a",
                    "skill": "GOTO",
                    "args": {
                        "target": {"kind": "NAMED_LOCATION", "name": "home"}
                    },
                },
                {
                    "id": "land",
                    "uav_id": "uav_a",
                    "skill": "LAND",
                    "args": {"zone": "home"},
                },
            ],
        }
    )
    validator = PlanValidator()
    common = {
        "source": "dynamic_llm",
        "mission_id": "mission_goal_test",
        "uav_id": "uav_a",
        "plan_version": 1,
        "assignment_id": "assignment_a",
        "goals": (_goal("goal_track", GoalType.TRACK_TARGET, duration_s=10.0),),
    }

    blocked, blocked_report = validator.validate_and_compile_with_report(
        draft,
        context,
        **common,
    )
    assert blocked is None
    assert blocked_report.hard_blocked
    assert blocked_report.findings[0].code is ValidationCode.STEP_REFERENCE_INVALID

    compiled, report = validator.validate_and_compile_with_report(
        draft,
        context,
        trusted_target_id="target_i",
        **common,
    )
    assert compiled is not None
    assert report.accepted
    decision, safety_report = SafetySupervisor(
        context.scene_min_xyz_m,
        context.scene_max_xyz_m,
    ).preflight_with_report(compiled, assignment_id="assignment_a")
    assert decision.action is SafetyAction.CONTINUE
    assert safety_report.accepted


def test_trusted_safety_epilogue_does_not_satisfy_user_land_goal() -> None:
    context = PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={},
        landing_zones={
            "home": LandingZoneSpec(
                name="home",
                position_xy_m=(0.0, 0.0),
                ground_altitude_m=0.0,
            )
        },
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=10.0,
        search_timeout_s=60.0,
    )
    draft = SkillPlanDraftV3.from_dict(
        {
            "schema_version": 3,
            "mission_id": "mission_goal_test",
            "uav_id": "uav_a",
            "plan_version": 1,
            "assumptions": [],
            "steps": [
                {
                    "id": "takeoff",
                    "uav_id": "uav_a",
                    "skill": "TAKEOFF",
                    "args": {"altitude_m": 10.0},
                },
                {
                    "id": "wait",
                    "uav_id": "uav_a",
                    "skill": "HOVER",
                    "args": {"duration_s": 5.0},
                },
            ],
        }
    )
    goals = (
        _goal("goal_wait", GoalType.WAIT, duration_s=5.0),
        _goal("goal_land", GoalType.LAND),
    )
    validator = PlanValidator()

    blocked, blocked_report = validator.validate_and_compile_with_report(
        draft,
        context,
        source="dynamic_llm",
        mission_id="mission_goal_test",
        uav_id="uav_a",
        plan_version=1,
        assignment_id="assignment_a",
        goals=goals,
    )
    assert blocked is None
    assert blocked_report.hard_blocked

    compiled, report = validator.validate_and_compile_with_report(
        draft,
        context,
        source="dynamic_llm",
        mission_id="mission_goal_test",
        uav_id="uav_a",
        plan_version=1,
        assignment_id="assignment_a",
        goals=goals,
        allow_trusted_safety_completion=True,
    )

    assert compiled is not None
    assert tuple(step.skill.value for step in compiled.task_plan.steps) == (
        "TAKEOFF",
        "HOVER",
        "GOTO",
        "LAND",
    )
    assert compiled.task_plan.steps[-2].step_id.startswith("trusted_return_home")
    assert compiled.task_plan.steps[-1].step_id.startswith("trusted_land")
    assert not report.hard_blocked
    assert {item.goal_id for item in report.findings} == {"goal_land"}
    raw_coverage = GoalSatisfactionChecker().check(
        goals,
        draft,
        mission_id="mission_goal_test",
        assignment_id="assignment_a",
        uav_id="uav_a",
        home_name="home",
    )
    assert raw_coverage.coverages[0].covered
    assert not raw_coverage.coverages[1].covered

    decision = SafetySupervisor(
        context.scene_min_xyz_m,
        context.scene_max_xyz_m,
    ).preflight(compiled)
    assert decision.action is SafetyAction.CONTINUE
