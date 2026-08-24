from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from fleet.airspace_manager import FleetPoseSnapshot, FleetUavPose
from fleet.runtime import (
    AssignmentStatus,
    FleetMissionRuntime,
    FleetRuntimeError,
    FleetStatus,
)
from fleet.scripted_planner import ScriptedFleetPlanner
from fleet.types import (
    AssignmentFailurePolicy,
    FleetCoordinationPolicy,
    FleetMissionRequest,
    FleetStartPolicy,
    FleetTargetRequest,
    FleetUavCapability,
)
from planner.spatial import CircleRegion, CoordinateFrame
from target.types import TargetSpec


def _request() -> FleetMissionRequest:
    return FleetMissionRequest(
        fleet_mission_id="fleet_mission_test",
        fleet_plan_version=1,
        original_instruction="两架无人机分别搜索两个目标",
        uav_inventory=(
            FleetUavCapability(
                "uav_a", "无人机A", True, "home_a", 5.0, 30.0
            ),
            FleetUavCapability(
                "uav_b", "无人机B", True, "home_b", 5.0, 30.0
            ),
        ),
        target_requests=(
            FleetTargetRequest(
                "target_i",
                TargetSpec("红色移动目标i"),
                requested_uav_id="uav_a",
                search_region=CircleRegion(
                    CoordinateFrame.WORLD_ENU, (20.0, 30.0, 0.0), 15.0
                ),
                track_duration_s=20.0,
                priority=100,
            ),
            FleetTargetRequest(
                "target_j",
                TargetSpec("蓝色移动目标j"),
                requested_uav_id="uav_b",
                search_region=CircleRegion(
                    CoordinateFrame.WORLD_ENU, (-25.0, 10.0, 0.0), 12.0
                ),
                track_duration_s=20.0,
                priority=10,
            ),
        ),
        coordination_policy=FleetCoordinationPolicy(),
    )


@dataclass
class _FakeAgent:
    uav_id: str
    terminal_status: str = "SUCCEEDED"
    plan_version: int = 1
    fail_tick: bool = False
    locked_target_id: str | None = None

    def __post_init__(self) -> None:
        self.status = "IDLE"
        self.started = 0
        self.ticks = 0
        self.cancels = 0

    def start_assignment(self, assignment: object) -> None:
        self.started += 1
        self.assignment = assignment
        self.status = "RUNNING"

    def tick(self, observation: object) -> dict[str, object]:
        self.ticks += 1
        if self.fail_tick:
            raise RuntimeError("local failure")
        self.status = self.terminal_status
        return self.snapshot()

    def cancel(self) -> None:
        self.cancels += 1
        self.status = "CANCELED"

    def snapshot(self) -> dict[str, object]:
        target_id = self.locked_target_id
        return {
            "status": self.status,
            "plan_version": self.plan_version if self.status != "IDLE" else None,
            "last_error": None,
            "target": {
                "target_id": target_id,
                "lifecycle": "LOCKED" if target_id else "SEARCHING",
                "confidence": 0.95 if target_id else None,
                "last_seen_time_s": 1.0 if target_id else None,
            },
        }


class _FakeEnvironment:
    def __init__(self, *, conflict: bool = False) -> None:
        self.started = 0
        self.closed = 0
        self.steps = 0
        self.held: list[str] = []
        self.conflict = conflict

    def start(self, plan: object) -> None:
        self.started += 1

    def step(self) -> None:
        self.steps += 1

    def get_agent_observation(self, uav_id: str) -> dict[str, object]:
        return {"uav_id": uav_id, "timestamp": float(self.steps)}

    def get_fleet_pose_snapshot(self) -> FleetPoseSnapshot:
        second_x = 1.0 if self.conflict else 20.0
        return FleetPoseSnapshot(
            float(self.steps),
            {
                "uav_a": FleetUavPose(
                    "uav_a", (0.0, 0.0, 10.0), priority=100
                ),
                "uav_b": FleetUavPose(
                    "uav_b", (second_x, 0.0, 10.0), priority=10
                ),
            },
        )

    def hold_uav(self, uav_id: str) -> None:
        self.held.append(uav_id)

    def close(self) -> None:
        self.closed += 1


class _RecordingLogger:
    def __init__(self) -> None:
        self.assignment_snapshots: list[tuple[dict[str, object], ...]] = []
        self.summaries: list[dict[str, object]] = []

    def write_fleet_plan(self, plan: object) -> None:
        self.plan = plan

    def write_assignments(self, rows: object) -> None:
        self.assignment_snapshots.append(
            tuple(dict(row) for row in rows)  # type: ignore[arg-type]
        )

    def write_summary(self, summary: object) -> None:
        self.summaries.append(dict(summary))  # type: ignore[arg-type]

    def log_fleet_event(self, event: object) -> None:
        pass

    def log_airspace_decision(self, decision: object) -> None:
        pass


def _runtime(
    environment: _FakeEnvironment,
    agent_a: _FakeAgent,
    agent_b: _FakeAgent,
) -> FleetMissionRuntime:
    request = _request()
    return FleetMissionRuntime(
        environment,
        ScriptedFleetPlanner(),
        {"uav_a": agent_a, "uav_b": agent_b},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
    )


def test_two_agents_begin_before_first_tick_and_fleet_succeeds() -> None:
    env = _FakeEnvironment()
    a = _FakeAgent("uav_a", plan_version=3)
    b = _FakeAgent("uav_b", plan_version=1)
    runtime = _runtime(env, a, b)
    runtime.start(_request().original_instruction)

    assert env.started == 1
    assert a.started == b.started == 1
    assert a.ticks == b.ticks == 0
    snapshot = runtime.tick()
    assert snapshot.status is FleetStatus.SUCCEEDED
    assert snapshot.agent_plan_versions == {"uav_a": 3, "uav_b": 1}
    assert all(
        row["status"] == "SUCCEEDED" for row in snapshot.assignments.values()
    )
    assert runtime.world_belief is not None
    assert runtime.world_belief.fleet_plan_version == 1
    assert runtime.world_belief.agents["uav_a"].plan_version == 3
    assert runtime.world_belief.local_safety_summary("uav_a") == (
        {
            "uav_id": "uav_b",
            "assignment_id": runtime.assignments.for_uav("uav_b").assignment.assignment_id,
            "current_region": "CIRCLE",
            "altitude_layer": None,
            "status": "SUCCEEDED",
        },
    )


def test_terminal_logger_refreshes_assignments_and_keeps_summary_bounded() -> None:
    request = _request()
    logger = _RecordingLogger()
    runtime = FleetMissionRuntime(
        _FakeEnvironment(),
        ScriptedFleetPlanner(),
        {"uav_a": _FakeAgent("uav_a"), "uav_b": _FakeAgent("uav_b")},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        logger=logger,
    )

    runtime.start(request.original_instruction, request=request)
    runtime.tick()

    assert len(logger.assignment_snapshots) == 2
    assert {row["status"] for row in logger.assignment_snapshots[0]} == {"RUNNING"}
    assert {row["status"] for row in logger.assignment_snapshots[-1]} == {
        "SUCCEEDED"
    }
    assert logger.summaries[-1]["status"] == "SUCCEEDED"
    assert logger.summaries[-1]["event_count"] > 0
    assert "events" not in logger.summaries[-1]


def test_local_failure_isolated_and_reports_partial_success() -> None:
    env = _FakeEnvironment()
    a = _FakeAgent("uav_a", fail_tick=True)
    b = _FakeAgent("uav_b")
    runtime = _runtime(env, a, b)
    runtime.start(_request().original_instruction)
    snapshot = runtime.tick()

    assert snapshot.status is FleetStatus.PARTIAL_SUCCESS
    assert a.cancels == b.cancels == 0
    assert b.ticks == 1
    statuses = {row["uav_id"]: row["status"] for row in snapshot.assignments.values()}
    assert statuses == {
        "uav_a": "REASSIGNMENT_REQUIRED",
        "uav_b": "SUCCEEDED",
    }
    assert any(event["event_type"] == "REASSIGNMENT_REQUIRED" for event in snapshot.events)


def test_airspace_conflict_immediately_holds_lower_priority_before_agent_tick() -> None:
    env = _FakeEnvironment(conflict=True)
    a = _FakeAgent("uav_a", terminal_status="RUNNING")
    b = _FakeAgent("uav_b", terminal_status="RUNNING")
    runtime = _runtime(env, a, b)
    runtime.start(_request().original_instruction)
    snapshot = runtime.tick()

    assert env.held == ["uav_b"]
    assert a.ticks == 1
    assert b.ticks == 0
    b_record = next(row for row in snapshot.assignments.values() if row["uav_id"] == "uav_b")
    assert b_record["status"] == AssignmentStatus.HOLDING.value
    assert snapshot.last_airspace_decision["event_type"] == "AIRSPACE_CONFLICT"


def test_preplanned_routes_are_injected_into_live_airspace_snapshots() -> None:
    env = _FakeEnvironment()
    a = _FakeAgent("uav_a", terminal_status="RUNNING")
    b = _FakeAgent("uav_b", terminal_status="RUNNING")
    request = _request()
    runtime = FleetMissionRuntime(
        env,
        ScriptedFleetPlanner(),
        {"uav_a": a, "uav_b": b},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        planned_routes={
            "uav_a": ((0.0, 0.0, 10.0), (20.0, 20.0, 10.0)),
            "uav_b": ((20.0, 0.0, 10.0), (0.0, 20.0, 10.0)),
        },
    )
    runtime.start(request.original_instruction, request=request)

    snapshot = runtime.tick()
    conflict = snapshot.last_airspace_decision["conflicts"][0]
    assert conflict["routes_intersect"] is True
    assert snapshot.last_airspace_decision["hold_uav_ids"] == ["uav_b"]


def test_terminal_assignment_drops_stale_route_and_releases_other_uav() -> None:
    env = _FakeEnvironment()
    a = _FakeAgent("uav_a", terminal_status="SUCCEEDED")
    b = _FakeAgent("uav_b", terminal_status="RUNNING")
    request = _request()
    runtime = FleetMissionRuntime(
        env,
        ScriptedFleetPlanner(),
        {"uav_a": a, "uav_b": b},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        planned_routes={
            "uav_a": ((0.0, 0.0, 10.0), (20.0, 20.0, 10.0)),
            "uav_b": ((20.0, 0.0, 10.0), (0.0, 20.0, 10.0)),
        },
    )
    runtime.start(request.original_instruction, request=request)

    first = runtime.tick()
    assert first.last_airspace_decision["hold_uav_ids"] == ["uav_b"]
    assert a.ticks == 1
    assert b.ticks == 0

    second = runtime.tick()
    assert second.last_airspace_decision["hold_uav_ids"] == []
    assert b.ticks == 1


def test_wrong_target_lock_generates_claim_conflict_and_hold() -> None:
    env = _FakeEnvironment()
    a = _FakeAgent("uav_a", terminal_status="RUNNING", locked_target_id="target_i")
    b = _FakeAgent("uav_b", terminal_status="RUNNING", locked_target_id="target_i")
    runtime = _runtime(env, a, b)
    runtime.start(_request().original_instruction)
    snapshot = runtime.tick()

    assert "uav_b" in env.held
    assert any(event["event_type"] == "TARGET_CLAIM_CONFLICT" for event in snapshot.events)
    b_record = next(row for row in snapshot.assignments.values() if row["uav_id"] == "uav_b")
    assert b_record["status"] == "HOLDING"


def test_planner_failure_does_not_start_environment() -> None:
    class _BrokenPlanner:
        def plan(self, request: object) -> object:
            raise RuntimeError("bad fleet JSON")

    env = _FakeEnvironment()
    request = _request()
    runtime = FleetMissionRuntime(
        env,
        _BrokenPlanner(),
        {"uav_a": _FakeAgent("uav_a"), "uav_b": _FakeAgent("uav_b")},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
    )
    with pytest.raises(FleetRuntimeError, match="fleet planning failed"):
        runtime.start(request.original_instruction, request=request)
    assert env.started == 0
    assert runtime.status is FleetStatus.FAILED


def test_cancel_is_fleet_wide_only_when_explicitly_requested() -> None:
    env = _FakeEnvironment()
    a = _FakeAgent("uav_a", terminal_status="RUNNING")
    b = _FakeAgent("uav_b", terminal_status="RUNNING")
    runtime = _runtime(env, a, b)
    runtime.start(_request().original_instruction)
    snapshot = runtime.cancel()
    assert snapshot.status is FleetStatus.CANCELED
    assert a.cancels == b.cancels == 1
    runtime.close()
    runtime.close()
    assert env.closed == 1


def test_cancel_keeps_fleet_running_until_every_agent_finishes_landing() -> None:
    @dataclass
    class _LandingAgent(_FakeAgent):
        def cancel(self) -> dict[str, object]:
            self.cancels += 1
            self.status = "RUNNING"
            self._landing = True
            return self.snapshot()

        def tick(self, observation: object) -> dict[str, object]:
            self.ticks += 1
            if getattr(self, "_landing", False):
                self.status = "CANCELED"
            return self.snapshot()

    env = _FakeEnvironment()
    a = _LandingAgent("uav_a", terminal_status="RUNNING")
    b = _LandingAgent("uav_b", terminal_status="RUNNING")
    runtime = _runtime(env, a, b)
    runtime.start(_request().original_instruction)

    canceling = runtime.cancel()
    assert canceling.status is FleetStatus.RUNNING
    assert {
        row["status"] for row in canceling.assignments.values()
    } == {AssignmentStatus.CANCELING.value}

    landed = runtime.tick()
    assert landed.status is FleetStatus.CANCELED
    assert {row["status"] for row in landed.assignments.values()} == {
        AssignmentStatus.CANCELED.value
    }
    assert [event["event_type"] for event in landed.events].count(
        "FLEET_CANCEL_REQUESTED"
    ) == 1
    assert [event["event_type"] for event in landed.events].count(
        "FLEET_CANCELED"
    ) == 1


def test_airspace_hold_does_not_deadlock_cancel_fail_safe_land() -> None:
    @dataclass
    class _LandingAgent(_FakeAgent):
        def cancel(self) -> dict[str, object]:
            self.cancels += 1
            self.status = "RUNNING"
            self._landing = True
            return self.snapshot()

        def tick(self, observation: object) -> dict[str, object]:
            self.ticks += 1
            if getattr(self, "_landing", False):
                self.status = "CANCELED"
            return self.snapshot()

        def snapshot(self) -> dict[str, object]:
            result = super().snapshot()
            result["active_skill"] = (
                "LAND" if getattr(self, "_landing", False) else "SEARCH"
            )
            return result

    env = _FakeEnvironment(conflict=True)
    a = _LandingAgent("uav_a", terminal_status="RUNNING")
    b = _LandingAgent("uav_b", terminal_status="RUNNING")
    runtime = _runtime(env, a, b)
    runtime.start(_request().original_instruction)
    assert runtime.cancel().status is FleetStatus.RUNNING

    landed = runtime.tick()

    assert landed.status is FleetStatus.CANCELED
    assert a.ticks == b.ticks == 1
    assert "uav_b" not in env.held
    assert landed.last_airspace_decision["hold_uav_ids"] == ["uav_b"]
    override_events = [
        event
        for event in landed.events
        if event["event_type"]
        == "AIRSPACE_HOLD_OVERRIDDEN_FOR_FAILSAFE_LAND"
    ]
    assert [event["uav_id"] for event in override_events] == ["uav_b"]


def test_runtime_rejects_sequential_plan_before_environment_start() -> None:
    base = _request()
    sequential_request = replace(
        base,
        target_requests=(
            replace(
                base.target_requests[0],
                start_policy=FleetStartPolicy.SEQUENTIAL,
            ),
            base.target_requests[1],
        ),
    )
    env = _FakeEnvironment()
    a = _FakeAgent("uav_a")
    b = _FakeAgent("uav_b")
    runtime = FleetMissionRuntime(
        env,
        ScriptedFleetPlanner(),
        {"uav_a": a, "uav_b": b},
        inventory=sequential_request.uav_inventory,
        target_requests=sequential_request.target_requests,
        coordination_policy=sequential_request.coordination_policy,
    )

    with pytest.raises(FleetRuntimeError, match="does not support SEQUENTIAL"):
        runtime.start(sequential_request.original_instruction)

    assert env.started == 0
    assert a.started == b.started == 0


def test_optional_assignment_failure_does_not_block_required_success() -> None:
    base = _request()
    request = replace(
        base,
        target_requests=(
            base.target_requests[0],
            replace(base.target_requests[1], required=False),
        ),
    )
    runtime = FleetMissionRuntime(
        _FakeEnvironment(),
        ScriptedFleetPlanner(),
        {
            "uav_a": _FakeAgent("uav_a"),
            "uav_b": _FakeAgent("uav_b", fail_tick=True),
        },
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
    )
    runtime.start(request.original_instruction)

    snapshot = runtime.tick()

    assert snapshot.status is FleetStatus.SUCCEEDED
    assert {
        row["uav_id"]: row["status"] for row in snapshot.assignments.values()
    } == {"uav_a": "SUCCEEDED", "uav_b": "REASSIGNMENT_REQUIRED"}


def test_agent_plan_versions_advance_independently_and_never_decrease() -> None:
    @dataclass
    class _VersionAgent(_FakeAgent):
        next_version: int = 1

        def tick(self, observation: object) -> dict[str, object]:
            self.ticks += 1
            self.plan_version = self.next_version
            self.status = self.terminal_status
            return self.snapshot()

    env = _FakeEnvironment()
    a = _VersionAgent(
        "uav_a",
        terminal_status="RUNNING",
        plan_version=3,
        next_version=4,
    )
    b = _VersionAgent(
        "uav_b",
        terminal_status="RUNNING",
        plan_version=1,
        next_version=1,
    )
    runtime = _runtime(env, a, b)
    runtime.start(_request().original_instruction)

    advanced = runtime.tick()
    assert advanced.agent_plan_versions == {"uav_a": 4, "uav_b": 1}
    assert runtime.assignments.for_uav("uav_a").local_plan_version == 4
    assert runtime.assignments.for_uav("uav_b").local_plan_version == 1

    a.next_version = 2
    stale = runtime.tick()
    assert stale.status is FleetStatus.RUNNING
    assert runtime.assignments.for_uav("uav_a").status is (
        AssignmentStatus.REASSIGNMENT_REQUIRED
    )
    assert runtime.assignments.for_uav("uav_a").local_plan_version == 4
    assert runtime.assignments.for_uav("uav_b").local_plan_version == 1


def test_successful_locked_target_claim_stays_terminated() -> None:
    env = _FakeEnvironment()
    runtime = _runtime(
        env,
        _FakeAgent("uav_a", locked_target_id="target_i"),
        _FakeAgent("uav_b", locked_target_id="target_j"),
    )
    runtime.start(_request().original_instruction)

    snapshot = runtime.tick()

    assert snapshot.status is FleetStatus.SUCCEEDED
    for target_id in ("target_i", "target_j"):
        record = runtime.targets.record(target_id)
        assert not record.active_claims
        assert {claim.state.value for claim in record.claims} == {"TERMINATED"}


def test_target_conflict_hold_persists_until_trusted_intervention() -> None:
    env = _FakeEnvironment()
    a = _FakeAgent(
        "uav_a",
        terminal_status="RUNNING",
        locked_target_id="target_i",
    )
    b = _FakeAgent(
        "uav_b",
        terminal_status="RUNNING",
        locked_target_id="target_i",
    )
    runtime = _runtime(env, a, b)
    runtime.start(_request().original_instruction)

    first = runtime.tick()
    env.conflict = True
    second = runtime.tick()

    assert first.status is second.status is FleetStatus.RUNNING
    assert b.ticks == 1
    assert runtime.assignments.for_uav("uav_b").status is AssignmentStatus.HOLDING
    assert runtime.assignments.for_uav("uav_b").last_error == (
        "TARGET_CLAIM_CONFLICT"
    )


def test_cancel_fleet_policy_requests_cancel_on_other_running_agent() -> None:
    base = _request()
    policy = replace(
        base.coordination_policy,
        assignment_failure_policy=AssignmentFailurePolicy.CANCEL_FLEET,
    )
    request = replace(base, coordination_policy=policy)
    env = _FakeEnvironment()
    a = _FakeAgent("uav_a", fail_tick=True)
    b = _FakeAgent("uav_b", terminal_status="RUNNING")
    runtime = FleetMissionRuntime(
        env,
        ScriptedFleetPlanner(),
        {"uav_a": a, "uav_b": b},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=policy,
    )
    runtime.start(request.original_instruction)

    snapshot = runtime.tick()

    assert snapshot.status is FleetStatus.FAILED
    assert b.cancels == 1
    assert b.ticks == 0
    assert not runtime.targets.record("target_i").active_claims
    assert not runtime.targets.record("target_j").active_claims


def test_explicit_required_unassigned_target_can_only_finish_partial() -> None:
    request = _request()
    full = ScriptedFleetPlanner().plan(request)
    partial = replace(
        full,
        assignments=(full.assignments[0],),
        unassigned_requirements=("target_j: no eligible UAV",),
    )

    class _PartialPlanner:
        def plan(self, supplied: object) -> object:
            return partial

    env = _FakeEnvironment()
    runtime = FleetMissionRuntime(
        env,
        _PartialPlanner(),
        {"uav_a": _FakeAgent("uav_a"), "uav_b": _FakeAgent("uav_b")},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
    )
    runtime.start(request.original_instruction, request=request)

    snapshot = runtime.tick()

    assert snapshot.status is FleetStatus.PARTIAL_SUCCESS
    assert snapshot.status is not FleetStatus.SUCCEEDED


def test_missing_required_target_fails_before_environment_start() -> None:
    request = _request()
    full = ScriptedFleetPlanner().plan(request)
    incomplete = replace(full, assignments=(full.assignments[0],))

    class _IncompletePlanner:
        def plan(self, supplied: object) -> object:
            return incomplete

    env = _FakeEnvironment()
    runtime = FleetMissionRuntime(
        env,
        _IncompletePlanner(),
        {"uav_a": _FakeAgent("uav_a"), "uav_b": _FakeAgent("uav_b")},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
    )

    with pytest.raises(FleetRuntimeError, match="required target"):
        runtime.start(request.original_instruction, request=request)

    assert runtime.status is FleetStatus.FAILED
    assert env.started == 0
