from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from fleet.runtime import (
    AssignmentStatus,
    FleetMissionRuntime,
    FleetReplanPublication,
    FleetStatus,
    ReplannedAssignment,
)
from fleet.scripted_planner import ScriptedFleetPlanner
from tests.fleet.test_fleet_runtime import _FakeAgent, _FakeEnvironment, _request
from fleet.types import FleetUavCapability


def _assignment_ids():
    request = _request()
    plan = ScriptedFleetPlanner().plan(request)
    return request, {item.uav_id: item.assignment_id for item in plan.assignments}


def test_one_failed_local_plan_does_not_prevent_other_uav_start() -> None:
    request, assignment_ids = _assignment_ids()
    environment = _FakeEnvironment()
    agent_b = _FakeAgent("uav_b", terminal_status="RUNNING")
    runtime = FleetMissionRuntime(
        environment,
        ScriptedFleetPlanner(),
        {"uav_b": agent_b},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        initial_assignment_states={
            assignment_ids["uav_a"]: AssignmentStatus.REPAIRING,
        },
    )

    runtime.start(request.original_instruction, request=request)

    assert environment.started == 1
    assert agent_b.started == 1
    assert runtime.status is FleetStatus.RUNNING
    assert runtime.assignments.for_uav("uav_a").status is AssignmentStatus.REPAIRING
    assert runtime.assignments.for_uav("uav_b").status is AssignmentStatus.RUNNING


def test_repaired_local_plan_starts_at_next_barrier() -> None:
    request, assignment_ids = _assignment_ids()
    environment = _FakeEnvironment()
    agent_b = _FakeAgent("uav_b", terminal_status="RUNNING")
    runtime = FleetMissionRuntime(
        environment,
        ScriptedFleetPlanner(),
        {"uav_b": agent_b},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        initial_assignment_states={
            assignment_ids["uav_a"]: AssignmentStatus.REPAIRING,
        },
    )
    runtime.start(request.original_instruction, request=request)
    repaired = _FakeAgent("uav_a", terminal_status="SUCCEEDED", plan_version=2)

    ready = runtime.register_ready_agent(
        assignment_ids["uav_a"],
        repaired,
        start_input=("trusted repaired local plan", None),
    )
    assert ready.status is AssignmentStatus.READY
    assert repaired.started == 0

    snapshot = runtime.tick()
    assert repaired.started == 1
    assert repaired.ticks == 1
    assert snapshot.assignments[assignment_ids["uav_a"]]["status"] == "SUCCEEDED"
    assert snapshot.assignments[assignment_ids["uav_b"]]["status"] == "RUNNING"


def test_all_failed_local_plans_never_start_environment() -> None:
    request, assignment_ids = _assignment_ids()
    environment = _FakeEnvironment()
    runtime = FleetMissionRuntime(
        environment,
        ScriptedFleetPlanner(),
        {},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        initial_assignment_states={
            assignment_id: AssignmentStatus.REPAIRING
            for assignment_id in assignment_ids.values()
        },
    )

    runtime.start(request.original_instruction, request=request)

    assert environment.started == 0
    assert runtime.status is FleetStatus.FAILED_NO_EXECUTABLE_PLAN
    assert {
        record.status for record in runtime.assignments.records
    } == {AssignmentStatus.FAILED}
    assert runtime.targets.records == ()


def test_report_and_replan_runs_real_handler_without_stopping_other_uav() -> None:
    request, assignment_ids = _assignment_ids()
    environment = _FakeEnvironment()
    failing = _FakeAgent("uav_a", fail_tick=True)
    healthy = _FakeAgent("uav_b")
    replacement = _FakeAgent("uav_a", plan_version=2)
    calls: list[tuple[str, object]] = []

    def replan_handler(record, world_belief):
        calls.append((record.assignment.assignment_id, world_belief))
        return ReplannedAssignment(
            assignment_id=record.assignment.assignment_id,
            agent=replacement,
            start_input=("trusted Fleet replan output", None),
        )

    runtime = FleetMissionRuntime(
        environment,
        ScriptedFleetPlanner(),
        {"uav_a": failing, "uav_b": healthy},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        replan_handler=replan_handler,
    )
    runtime.start(request.original_instruction, request=request)

    waiting = runtime.tick()
    assert healthy.ticks == 1
    assert len(calls) == 1
    assert waiting.assignments[assignment_ids["uav_a"]]["status"] == "READY", (
        waiting.assignments[assignment_ids["uav_a"]]
    )
    assert waiting.assignments[assignment_ids["uav_b"]]["status"] == "SUCCEEDED"
    assert waiting.status is FleetStatus.RUNNING

    finished = runtime.tick()
    assert replacement.started == 1
    assert replacement.ticks == 1
    assert finished.status is FleetStatus.SUCCEEDED
    event_types = [event["event_type"] for event in finished.events]
    assert "FLEET_REPLAN_REQUESTED" in event_types
    assert "FLEET_REPLAN_ACCEPTED" in event_types
    assert "REPAIRED_ASSIGNMENT_STARTED" in event_types


def _request_with_idle_uav():
    request = _request()
    return replace(
        request,
        uav_inventory=request.uav_inventory
        + (
            FleetUavCapability(
                "uav_c", "无人机C", True, "home_c", 5.0, 30.0
            ),
        ),
    )


def test_atomic_fleet_replan_reassigns_to_idle_uav_and_publishes_version() -> None:
    request = _request_with_idle_uav()
    original_plan = ScriptedFleetPlanner().plan(request)
    failed_assignment = next(
        item for item in original_plan.assignments if item.uav_id == "uav_a"
    )
    failing = _FakeAgent("uav_a", fail_tick=True)
    healthy = _FakeAgent("uav_b")
    replacement_agent = _FakeAgent("uav_c", plan_version=2)
    replacement_assignment = replace(
        failed_assignment,
        assignment_id="assignment_replanned_i",
        uav_id="uav_c",
    )

    def replan_handler(record, world_belief):
        assert world_belief is not None
        assert world_belief.fleet_plan_version == 1
        return FleetReplanPublication(
            base_fleet_plan_version=1,
            new_fleet_plan_version=2,
            replacements=(
                ReplannedAssignment(
                    assignment_id=record.assignment.assignment_id,
                    replacement_assignment=replacement_assignment,
                    agent=replacement_agent,
                    start_input=("trusted reassigned local plan", None),
                ),
            ),
        )

    runtime = FleetMissionRuntime(
        _FakeEnvironment(),
        ScriptedFleetPlanner(),
        {"uav_a": failing, "uav_b": healthy},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        replan_handler=replan_handler,
    )
    runtime.start(request.original_instruction, request=request)

    published = runtime.tick()
    assert published.fleet_plan_version == 2
    assert failed_assignment.assignment_id not in published.assignments
    assert published.assignments[replacement_assignment.assignment_id]["uav_id"] == "uav_c"
    assert published.assignments[replacement_assignment.assignment_id]["status"] == "READY"
    assert published.assignments[
        next(
            item.assignment_id
            for item in original_plan.assignments
            if item.uav_id == "uav_b"
        )
    ]["status"] == "SUCCEEDED"
    assert replacement_agent.started == 0
    assert failing.cancels == 1
    assert runtime.fleet_plan is not None
    assert runtime.fleet_plan.fleet_plan_version == 2
    assert runtime.world_belief is not None
    assert runtime.world_belief.fleet_plan_version == 2

    finished = runtime.tick()
    assert replacement_agent.started == 1
    assert replacement_agent.ticks == 1
    assert finished.status is FleetStatus.SUCCEEDED
    assert any(
        event["event_type"] == "FLEET_PLAN_VERSION_PUBLISHED"
        and event["fleet_plan_version"] == 2
        for event in finished.events
    )


def test_invalid_multi_replacement_is_not_partially_published() -> None:
    request = _request_with_idle_uav()
    original_plan = ScriptedFleetPlanner().plan(request)
    failed_assignment = next(
        item for item in original_plan.assignments if item.uav_id == "uav_a"
    )
    failing = _FakeAgent("uav_a", fail_tick=True)
    healthy = _FakeAgent("uav_b", terminal_status="RUNNING")
    idle_slot = _FakeAgent("uav_c", terminal_status="RUNNING")
    replacement_agent = _FakeAgent("uav_c", plan_version=2)
    valid_replacement = replace(
        failed_assignment,
        assignment_id="assignment_replanned_i",
        uav_id="uav_c",
    )
    unrelated_replacement = replace(
        failed_assignment,
        assignment_id="assignment_unrelated",
    )

    def replan_handler(record, world_belief):
        return FleetReplanPublication(
            base_fleet_plan_version=1,
            new_fleet_plan_version=2,
            replacements=(
                ReplannedAssignment(
                    assignment_id=record.assignment.assignment_id,
                    replacement_assignment=valid_replacement,
                    agent=replacement_agent,
                    start_input=("valid compiled plan", None),
                ),
                ReplannedAssignment(
                    assignment_id="assignment_missing",
                    replacement_assignment=unrelated_replacement,
                    agent=_FakeAgent("uav_a"),
                    start_input=("unrelated compiled plan", None),
                ),
            ),
        )

    runtime = FleetMissionRuntime(
        _FakeEnvironment(),
        ScriptedFleetPlanner(),
        {"uav_a": failing, "uav_b": healthy, "uav_c": idle_slot},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        replan_handler=replan_handler,
    )
    runtime.start(request.original_instruction, request=request)

    snapshot = runtime.tick()
    assert snapshot.fleet_plan_version == 1
    assert valid_replacement.assignment_id not in snapshot.assignments
    assert snapshot.assignments[failed_assignment.assignment_id]["status"] == "FAILED"
    assert runtime.assignments.for_uav("uav_b").status is AssignmentStatus.RUNNING
    assert runtime.agents["uav_c"] is idle_slot
    assert replacement_agent.started == 0
    assert not any(
        event["event_type"] == "FLEET_PLAN_VERSION_PUBLISHED"
        for event in snapshot.events
    )
    assert any(
        event["event_type"] == "FLEET_REPLAN_FAILED"
        for event in snapshot.events
    )


def test_fleet_replan_publication_rejects_nonincrementing_version() -> None:
    with pytest.raises(ValueError, match="base_fleet_plan_version \\+ 1"):
        FleetReplanPublication(
            base_fleet_plan_version=2,
            new_fleet_plan_version=4,
            replacements=(),
        )


def test_waiting_replan_holds_then_locally_lands_only_failed_uav() -> None:
    @dataclass
    class _LandingAgent(_FakeAgent):
        landing: bool = False

        def cancel(self):
            self.cancels += 1
            self.landing = True
            self.status = "RUNNING"
            return self.snapshot()

        def tick(self, observation):
            self.ticks += 1
            if self.landing:
                self.status = "CANCELED"
                return self.snapshot()
            raise RuntimeError("local controller path failed")

    request = _request()
    failing = _LandingAgent("uav_a")
    healthy = _FakeAgent("uav_b", terminal_status="RUNNING")
    handler_calls = 0

    def replan_handler(record, world_belief):
        nonlocal handler_calls
        handler_calls += 1
        if handler_calls == 1:
            return None
        raise RuntimeError("bounded Fleet replan exhausted")

    environment = _FakeEnvironment()
    runtime = FleetMissionRuntime(
        environment,
        ScriptedFleetPlanner(),
        {"uav_a": failing, "uav_b": healthy},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        replan_handler=replan_handler,
    )
    runtime.start(request.original_instruction, request=request)

    waiting = runtime.tick()
    failed_id = runtime.assignments.for_uav("uav_a").assignment.assignment_id
    assert waiting.assignments[failed_id]["status"] == "WAITING_REASSIGNMENT"
    assert failing.ticks == 1
    assert failing.cancels == 0
    assert healthy.ticks == 1
    assert environment.held[-1] == "uav_a"

    landing = runtime.tick()
    assert landing.assignments[failed_id]["status"] == "CANCELING"
    assert failing.ticks == 1  # WAITING never re-executes the failed local plan.
    assert failing.cancels == 1
    assert healthy.ticks == 2
    assert healthy.cancels == 0
    assert environment.held.count("uav_a") >= 2

    terminal = runtime.tick()
    assert terminal.assignments[failed_id]["status"] == "FAILED"
    assert failing.ticks == 2
    assert healthy.ticks == 3
    assert healthy.cancels == 0
    assert any(
        event["event_type"] == "LOCAL_FAILSAFE_LAND_COMPLETED"
        for event in terminal.events
    )
