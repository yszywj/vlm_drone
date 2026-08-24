from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from configs.loader import load_config
from env.fleet_uav_search_env import FleetUavSearchEnv
from fleet.runtime import FleetMissionRuntime
from fleet.task_spec import (
    ConstraintStrength,
    FleetTaskSpecV1,
    GoalType,
    MissionGoal,
    TerminationGoal,
)
from fleet.types import FleetCoordinationPolicy, FleetStartPolicy, FleetUavCapability
from fleet.types_v2 import (
    FleetAssignmentV2,
    FleetMissionPlanV2,
    FleetMissionRequestV2,
)
from planner.spatial import CircleRegion, CoordinateFrame
from scripts.run_fleet_mission import (
    FleetLaunchConfigurationError,
    RuntimeEnvelopeMetadata,
    _lower_v2_target_plan_for_runtime,
)


ROOT = Path(__file__).resolve().parents[2]


def _contracts(
    *,
    goals: tuple[MissionGoal, ...] = (),
    termination_goals: tuple[TerminationGoal, ...] = (),
) -> tuple[object, FleetMissionRequestV2, FleetMissionPlanV2]:
    config = load_config(ROOT / "configs/multi_uav_demo.yaml")
    task = FleetTaskSpecV1(
        source_text="execute the assigned semantic goals",
        goals=goals,
        termination_goals=termination_goals,
    )
    request = FleetMissionRequestV2(
        fleet_mission_id="fleet_mission_targetless",
        fleet_plan_version=1,
        task_spec=task,
        uav_inventory=(
            FleetUavCapability("uav_a", "A", True, "home_a", 5.0, 30.0),
        ),
        trusted_fleet_state=(),
        coordination_policy=FleetCoordinationPolicy(),
    )
    assignment = FleetAssignmentV2(
        assignment_id="assignment_targetless",
        uav_id="uav_a",
        goal_ids=task.all_goal_ids,
        priority=100,
        start_policy=FleetStartPolicy.PARALLEL,
    )
    plan = FleetMissionPlanV2(
        fleet_mission_id=request.fleet_mission_id,
        fleet_plan_version=request.fleet_plan_version,
        assignments=(assignment,),
        coordination_policy=request.coordination_policy,
    )
    return config, request, plan


def _targetless_goal(goal_type: GoalType) -> MissionGoal | TerminationGoal:
    if goal_type is GoalType.NAVIGATE:
        return MissionGoal(
            "goal_navigate",
            goal_type,
            None,
            CircleRegion(CoordinateFrame.WORLD_ENU, (12.0, 5.0, 0.0), 2.0),
            None,
            None,
            ConstraintStrength.MUST,
        )
    return TerminationGoal(
        "goal_" + goal_type.value.lower(),
        goal_type,
        "uav_a",
        3.0 if goal_type is GoalType.WAIT else None,
        ConstraintStrength.MUST,
    )


@pytest.mark.parametrize(
    "goal_type",
    (
        GoalType.NAVIGATE,
        GoalType.WAIT,
        GoalType.LAND,
        GoalType.RETURN_HOME,
        GoalType.RETURN_HOME_AND_LAND,
    ),
)
def test_only_bounded_targetless_goal_types_receive_quarantined_envelope(
    goal_type: GoalType,
) -> None:
    goal = _targetless_goal(goal_type)
    config, request, plan = _contracts(
        goals=(goal,) if isinstance(goal, MissionGoal) else (),
        termination_goals=(goal,) if isinstance(goal, TerminationGoal) else (),
    )

    runtime_request, runtime_plan, metadata = _lower_v2_target_plan_for_runtime(
        config,
        request,
        plan,
        include_metadata=True,
    )

    assert isinstance(metadata, RuntimeEnvelopeMetadata)
    assert metadata.non_target_assignment_ids == {"assignment_targetless"}
    assert metadata.semantic_target_by_assignment == {
        "assignment_targetless": None
    }
    assert metadata.required_by_assignment == {"assignment_targetless": True}
    anchor = metadata.compatibility_anchor_by_assignment["assignment_targetless"]
    assert anchor.startswith("compat_non_target_")
    assert anchor not in {target.id for target in config.targets}
    assert runtime_plan.assignments[0].target_alias == anchor
    assert runtime_request.target_request(anchor).required is True


def test_targetless_prefer_goal_keeps_optional_requiredness() -> None:
    goal = TerminationGoal(
        "goal_wait",
        GoalType.WAIT,
        "uav_a",
        2.0,
        ConstraintStrength.PREFER,
    )
    config, request, plan = _contracts(termination_goals=(goal,))

    runtime_request, _, metadata = _lower_v2_target_plan_for_runtime(
        config,
        request,
        plan,
        include_metadata=True,
    )

    anchor = metadata.compatibility_anchor_by_assignment["assignment_targetless"]
    assert metadata.required_by_assignment["assignment_targetless"] is False
    assert runtime_request.target_request(anchor).required is False


def test_targetless_report_is_explicitly_non_executable() -> None:
    report = _targetless_goal(GoalType.REPORT)
    assert isinstance(report, TerminationGoal)
    config, request, plan = _contracts(termination_goals=(report,))

    with pytest.raises(
        FleetLaunchConfigurationError,
        match="targetless.*REPORT",
    ):
        _lower_v2_target_plan_for_runtime(config, request, plan)


def test_multiple_semantic_targets_still_fail_closed() -> None:
    goals = tuple(
        MissionGoal(
            f"goal_search_{suffix}",
            GoalType.SEARCH_TARGET,
            f"target_{suffix}",
            CircleRegion(CoordinateFrame.WORLD_ENU, (x, 0.0, 0.0), 5.0),
            None,
            None,
            ConstraintStrength.MUST,
        )
        for suffix, x in (("i", 10.0), ("j", -10.0))
    )
    config, request, plan = _contracts(goals=goals)

    with pytest.raises(FleetLaunchConfigurationError, match="multiple semantic targets"):
        _lower_v2_target_plan_for_runtime(config, request, plan)


@dataclass
class _Agent:
    uav_id: str = "uav_a"

    def __post_init__(self) -> None:
        self.status = "IDLE"

    def start_assignment(self, assignment: object) -> None:
        self.status = "RUNNING"

    def snapshot(self) -> dict[str, object]:
        return {"status": self.status, "plan_version": 1}


class _Environment:
    def __init__(self) -> None:
        self.assignment_publications: list[dict[str, str]] = []

    def set_assignments(self, assignments: object) -> None:
        self.assignment_publications.append(dict(assignments))  # type: ignore[arg-type]

    def start(self, plan: object) -> None:
        self.plan = plan


class _Planner:
    def __init__(self, plan: object) -> None:
        self.plan_value = plan

    def plan(self, request: object) -> object:
        return self.plan_value


def test_runtime_never_publishes_or_claims_targetless_anchor() -> None:
    goal = _targetless_goal(GoalType.LAND)
    assert isinstance(goal, TerminationGoal)
    config, request_v2, plan_v2 = _contracts(termination_goals=(goal,))
    runtime_request, runtime_plan, metadata = _lower_v2_target_plan_for_runtime(
        config,
        request_v2,
        plan_v2,
        include_metadata=True,
    )
    environment = _Environment()
    runtime = FleetMissionRuntime(
        environment,
        _Planner(runtime_plan),
        {"uav_a": _Agent()},
        precomputed_start_inputs={"uav_a": (request_v2.task_spec.source_text, None)},
        non_target_assignment_ids=tuple(metadata.non_target_assignment_ids),
        assignment_requiredness=metadata.required_by_assignment,
    )

    runtime.start(runtime_request.original_instruction, request=runtime_request)
    snapshot = runtime.snapshot()

    assert environment.assignment_publications == [{}]
    assert runtime.targets.records == ()
    assert runtime.assignments.by_id("assignment_targetless").required is True
    assignment_row = snapshot.assignments["assignment_targetless"]
    assert assignment_row["target_alias"] is None
    assert assignment_row["target_spec"] is None
    assert assignment_row["non_target_assignment"] is True


def test_explicit_empty_environment_assignments_do_not_restore_defaults() -> None:
    config = load_config(ROOT / "configs/multi_uav_demo.yaml")

    environment = FleetUavSearchEnv(config, assignments={})

    assert dict(environment.assignments) == {}
