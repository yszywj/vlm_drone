from __future__ import annotations

from dataclasses import replace

import pytest

from fleet.runtime import (
    AssignmentStatus,
    FleetMissionRuntime,
    FleetReplanPublication,
    ReplannedAssignment,
)
from fleet.scripted_planner import ScriptedFleetPlanner
from perception.mode import TargetPerceptionMode
from perception.runtime_provider import YoloTargetPerceptionRuntime
from target import TargetSpec
from tests.fleet.test_fleet_runtime import _FakeAgent, _FakeEnvironment, _request
from tests.perception.test_yolo_runtime_provider import _Bridge


class _AssignmentScopedOracleRuntime:
    mode = TargetPerceptionMode.ORACLE
    backend_name = "oracle_evaluation"

    def __init__(
        self,
        uav_id: str,
        metrics: dict[str, object] | None = None,
    ) -> None:
        self.uav_id = uav_id
        self.metric_values = {} if metrics is None else dict(metrics)
        self.reset_calls: list[dict[str, object]] = []
        self.closed = 0
        self.target_id: str | None = None

    def reset(
        self,
        *,
        mission_id: str,
        assignment_id: str,
        uav_id: str,
        target_alias: str,
        target_spec: TargetSpec,
    ) -> None:
        if self.closed:
            raise RuntimeError("old Oracle binding is closed")
        if uav_id != self.uav_id:
            raise PermissionError("Oracle runtime routed to another UAV")
        self.target_id = target_alias
        self.reset_calls.append(
            {
                "mission_id": mission_id,
                "assignment_id": assignment_id,
                "uav_id": uav_id,
                "target_alias": target_alias,
                "target_spec": target_spec,
            }
        )

    def close(self) -> None:
        self.closed += 1

    def metrics(self) -> dict[str, object]:
        return dict(self.metric_values)


def _single_target_request():
    request = _request()
    return replace(request, target_requests=(request.target_requests[0],))


def test_cross_uav_oracle_replan_retires_old_binding_and_resets_new_one() -> None:
    request = _single_target_request()
    old_oracle = _AssignmentScopedOracleRuntime("uav_a")
    runtime = FleetMissionRuntime(
        _FakeEnvironment(),
        ScriptedFleetPlanner(),
        {
            "uav_a": _FakeAgent("uav_a", terminal_status="RUNNING"),
            "uav_b": _FakeAgent("uav_b", terminal_status="RUNNING"),
        },
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        perceptions={"uav_a": old_oracle},
    )
    plan = runtime.start(request.original_instruction, request=request)
    source = plan.assignments[0]
    assert old_oracle.target_id == "target_i"
    assert old_oracle.reset_calls[0]["assignment_id"] == source.assignment_id

    runtime.assignments.update(
        source.assignment_id,
        AssignmentStatus.WAITING_REASSIGNMENT,
    )
    replacement_assignment = replace(
        source,
        assignment_id="assignment_replanned_target_i",
        uav_id="uav_b",
    )
    new_oracle = _AssignmentScopedOracleRuntime("uav_b")
    replacement_agent = _FakeAgent(
        "uav_b",
        terminal_status="RUNNING",
        plan_version=2,
    )
    publication = FleetReplanPublication(
        base_fleet_plan_version=1,
        new_fleet_plan_version=2,
        replacements=(
            ReplannedAssignment(
                assignment_id=source.assignment_id,
                replacement_assignment=replacement_assignment,
                agent=replacement_agent,
                perception=new_oracle,
                start_input=("continue target_i mission", None),
            ),
        ),
    )

    published = runtime._publish_fleet_replan(  # noqa: SLF001
        source.assignment_id,
        publication,
    )

    assert published == (replacement_assignment.assignment_id,)
    assert old_oracle.closed == 1
    assert "uav_a" not in runtime._perceptions  # noqa: SLF001
    assert runtime._perceptions["uav_b"] is new_oracle  # noqa: SLF001
    assert new_oracle.reset_calls == []

    runtime._start_pending_ready_assignments()  # noqa: SLF001
    assert new_oracle.target_id == "target_i"
    assert new_oracle.reset_calls == [
        {
            "mission_id": request.fleet_mission_id,
            "assignment_id": replacement_assignment.assignment_id,
            "uav_id": "uav_b",
            "target_alias": "target_i",
            "target_spec": source.target_spec,
        }
    ]
    with pytest.raises(RuntimeError, match="closed"):
        old_oracle.reset(
            mission_id=request.fleet_mission_id,
            assignment_id=source.assignment_id,
            uav_id="uav_a",
            target_alias="target_i",
            target_spec=source.target_spec,
        )
    runtime.close()


def test_same_uav_yolo_replan_closes_old_runtime_and_resets_new_stream() -> None:
    request = _single_target_request()
    old_bridge = _Bridge("uav_a")
    old_runtime = YoloTargetPerceptionRuntime(
        uav_id="uav_a",
        bridge=old_bridge,
    )
    runtime = FleetMissionRuntime(
        _FakeEnvironment(),
        ScriptedFleetPlanner(),
        {
            "uav_a": _FakeAgent("uav_a", terminal_status="RUNNING"),
            "uav_b": _FakeAgent("uav_b", terminal_status="RUNNING"),
        },
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        perceptions={"uav_a": old_runtime},
    )
    plan = runtime.start(request.original_instruction, request=request)
    assignment = plan.assignments[0]
    assert old_bridge.reset_calls == [
        (request.fleet_mission_id, assignment.target_spec, "target_i")
    ]

    runtime.assignments.update(
        assignment.assignment_id,
        AssignmentStatus.WAITING_REASSIGNMENT,
    )
    new_bridge = _Bridge("uav_a")
    new_runtime = YoloTargetPerceptionRuntime(
        uav_id="uav_a",
        bridge=new_bridge,
    )
    runtime.register_ready_agent(
        assignment.assignment_id,
        _FakeAgent(
            "uav_a",
            terminal_status="RUNNING",
            plan_version=2,
        ),
        perception=new_runtime,
        start_input=("retry target_i mission", None),
    )

    assert old_bridge.closed == 1
    assert new_bridge.reset_calls == []
    runtime._start_pending_ready_assignments()  # noqa: SLF001
    assert new_bridge.reset_calls == [
        (request.fleet_mission_id, assignment.target_spec, "target_i")
    ]
    assert new_runtime.target_id == "target_i"
    assert new_runtime._assignment_id == assignment.assignment_id  # noqa: SLF001
    runtime.close()
    assert new_bridge.closed == 1


def test_replan_preserves_and_aggregates_retired_perception_metrics() -> None:
    request = _single_target_request()
    old_runtime = _AssignmentScopedOracleRuntime(
        "uav_a",
        {
            "oracle_visible_frames": 4,
            "oracle_total_frames": 10,
            "oracle_visible_ratio": 0.4,
            "time_to_first_oracle_visibility_s": 8.0,
            "target_lost_count": 1,
        },
    )
    runtime = FleetMissionRuntime(
        _FakeEnvironment(),
        ScriptedFleetPlanner(),
        {
            "uav_a": _FakeAgent("uav_a", terminal_status="RUNNING"),
            "uav_b": _FakeAgent("uav_b", terminal_status="RUNNING"),
        },
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        perceptions={"uav_a": old_runtime},
    )
    plan = runtime.start(request.original_instruction, request=request)
    assignment = plan.assignments[0]
    new_runtime = _AssignmentScopedOracleRuntime(
        "uav_a",
        {
            "oracle_visible_frames": 3,
            "oracle_total_frames": 5,
            "oracle_visible_ratio": 0.6,
            "time_to_first_oracle_visibility_s": 12.0,
            "target_lost_count": 2,
        },
    )
    runtime.assignments.update(
        assignment.assignment_id,
        AssignmentStatus.WAITING_REASSIGNMENT,
    )
    runtime.register_ready_agent(
        assignment.assignment_id,
        _FakeAgent("uav_a", terminal_status="RUNNING", plan_version=2),
        perception=new_runtime,
        start_input=("retry target_i mission", None),
    )

    metrics = runtime.perception_metrics_snapshot()["uav_a"]
    assert old_runtime.closed == 1
    assert metrics["perception_segment_count"] == 2
    assert metrics["mode_transition_detected"] is False
    assert metrics["oracle_visible_frames"] == 7
    assert metrics["oracle_total_frames"] == 15
    assert metrics["oracle_visible_ratio"] == pytest.approx(7 / 15)
    assert metrics["time_to_first_oracle_visibility_s"] == 8.0
    assert metrics["target_lost_count"] == 3
    runtime.close()
