from __future__ import annotations

from dataclasses import dataclass

import pytest

from fleet.planning_recovery import (
    FleetReplanRequest,
    PlanningRecoveryCoordinator,
    RecoveryDisposition,
    execute_fleet_replan,
)
from fleet.planning_state import (
    AssignmentPlanningState,
    PlanningRepairBudget,
    PlanningStage,
    PlanningStateError,
    PlanningStateTracker,
)


def test_repair_budgets_are_independent_and_versions_increase() -> None:
    tracker = PlanningStateTracker()
    tracker.register("assignment_a", "uav_a")
    tracker.transition("assignment_a", AssignmentPlanningState.PLANNING)
    tracker.transition("assignment_a", AssignmentPlanningState.VALIDATING)
    budget = PlanningRepairBudget(interpreter=1, fleet_plan=2, local_plan=2)

    first = tracker.begin_repair(
        "assignment_a",
        PlanningStage.LOCAL_PLANNING,
        budget=budget,
        error_code="GOAL_NOT_COVERED",
    )
    assert first.plan_version == 2
    assert first.repair_attempts[PlanningStage.LOCAL_PLANNING] == 1
    second = tracker.begin_repair(
        "assignment_a",
        PlanningStage.LOCAL_PLANNING,
        budget=budget,
        error_code="GOAL_NOT_COVERED",
    )
    assert second.plan_version == 3
    with pytest.raises(PlanningStateError, match="budget exhausted"):
        tracker.begin_repair(
            "assignment_a",
            PlanningStage.LOCAL_PLANNING,
            budget=budget,
            error_code="GOAL_NOT_COVERED",
        )


def test_recoverable_exhaustion_degrades_only_when_safe_partial_exists() -> None:
    coordinator = PlanningRecoveryCoordinator(
        PlanningRepairBudget(interpreter=0, fleet_plan=0, local_plan=1)
    )
    finding = {"severity": "RECOVERABLE_SEMANTIC_ERROR", "code": "TRACK_MISSING"}
    repair = coordinator.decide(
        stage=PlanningStage.LOCAL_PLANNING,
        plan_version=1,
        assignment_id="assignment_a",
        findings=(finding,),
        uncovered_goal_ids=("goal_track",),
    )
    assert repair.disposition is RecoveryDisposition.REPAIR
    assert repair.next_plan_version == 2
    degraded = coordinator.decide(
        stage=PlanningStage.LOCAL_PLANNING,
        plan_version=2,
        assignment_id="assignment_a",
        findings=(finding,),
        safe_partial_available=True,
        uncovered_goal_ids=("goal_track",),
    )
    assert degraded.disposition is RecoveryDisposition.DEGRADED_EXECUTABLE
    assert degraded.uncovered_goal_ids == ("goal_track",)


@pytest.mark.parametrize(
    ("airborne", "severity", "expected"),
    (
        (False, "HARD_ACTION_BLOCK", RecoveryDisposition.KEEP_GROUNDED),
        (True, "HARD_ACTION_BLOCK", RecoveryDisposition.HOVER_AND_REPLAN),
        (True, "FATAL_SAFETY", RecoveryDisposition.SAFE_LAND),
    ),
)
def test_hard_and_fatal_findings_never_become_executable(
    airborne: bool, severity: str, expected: RecoveryDisposition
) -> None:
    decision = PlanningRecoveryCoordinator().decide(
        stage=PlanningStage.LOCAL_PLANNING,
        plan_version=1,
        assignment_id="assignment_a",
        findings=({"severity": severity, "code": "UNSAFE_GOTO"},),
        airborne=airborne,
        safe_partial_available=True,
    )
    assert decision.disposition is expected


@dataclass(frozen=True)
class _Assignment:
    assignment_id: str


@dataclass(frozen=True)
class _Patch:
    new_fleet_plan_version: int
    replacement_assignments: tuple[_Assignment, ...]


class _Planner:
    def replan(self, request):
        return _Patch(
            request.base_fleet_plan_version + 1,
            (_Assignment("assignment_replanned"), _Assignment("assignment_bad")),
        )


class _Compiler:
    def compile_reassignment(self, request, proposal, assignment):
        if assignment.assignment_id == "assignment_bad":
            raise ValueError("bad local proposal")
        return {"assignment_id": assignment.assignment_id, "ready": True}


def test_real_fleet_replan_compiles_assignments_independently() -> None:
    request = FleetReplanRequest(
        fleet_mission_id="fleet_mission_replan",
        base_fleet_plan_version=3,
        incomplete_goal_ids=("goal_track",),
        available_uav_ids=("uav_b",),
        trusted_fleet_state={"uav_b": {"available": True}},
        reason_codes=("ASSIGNMENT_FAILED",),
    )
    outcome = execute_fleet_replan(request, planner=_Planner(), compiler=_Compiler())
    assert outcome.new_fleet_plan_version == 4
    assert outcome.compilations["assignment_replanned"]["ready"] is True
    assert isinstance(outcome.compilations["assignment_bad"], ValueError)

