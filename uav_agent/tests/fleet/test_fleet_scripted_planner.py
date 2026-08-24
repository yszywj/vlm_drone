from __future__ import annotations

from dataclasses import replace

import pytest

from fleet.planner_base import FleetPlanner, FleetPlannerError
from fleet.scripted_planner import ScriptedFleetPlanner
from fleet.types import (
    FleetMissionRequest,
    FleetStartPolicy,
    FleetTargetRequest,
    FleetUavCapability,
)
from planner.spatial import CircleRegion, CoordinateFrame
from target.types import TargetSpec


def _uav(uav_id: str, *, available: bool = True) -> FleetUavCapability:
    return FleetUavCapability(
        uav_id,
        uav_id,
        available,
        f"home_{uav_id[-1]}",
        5,
        30,
    )


def _target(
    alias: str,
    x: float,
    *,
    requested_uav_id: str | None = None,
    start_policy: FleetStartPolicy = FleetStartPolicy.PARALLEL,
) -> FleetTargetRequest:
    return FleetTargetRequest(
        alias,
        TargetSpec(alias, immutable_identity_summary=alias),
        requested_uav_id,
        CircleRegion(CoordinateFrame.WORLD_ENU, (x, 0, 0), 10),
        20,
        start_policy=start_policy,
    )


def _request(*targets: FleetTargetRequest) -> FleetMissionRequest:
    return FleetMissionRequest(
        "fleet_mission_scripted",
        1,
        "structured fleet request",
        (_uav("uav_b"), _uav("uav_a")),
        targets,
    )


def test_scripted_planner_is_deterministic_and_keeps_plan_layered() -> None:
    planner = ScriptedFleetPlanner()
    assert isinstance(planner, FleetPlanner)
    request = _request(
        _target("target_i", 10, requested_uav_id="uav_a"),
        _target("target_j", 30, requested_uav_id="uav_b"),
    )
    first = planner.plan(request)
    second = planner.plan(request)
    assert first == second
    assert [(item.uav_id, item.target_alias) for item in first.assignments] == [
        ("uav_a", "target_i"),
        ("uav_b", "target_j"),
    ]
    assert [item.assignment_id for item in first.assignments] == [
        "assignment_uav_a_target_i",
        "assignment_uav_b_target_j",
    ]
    assert "steps" not in repr(first.to_dict()).lower()


def test_scripted_planner_uses_stable_sorted_available_uav_selection() -> None:
    request = _request(_target("target_i", 10), _target("target_j", 30))
    plan = ScriptedFleetPlanner().plan(request)
    assert [assignment.uav_id for assignment in plan.assignments] == [
        "uav_a",
        "uav_b",
    ]


def test_scripted_planner_requires_structured_geometry_and_duration() -> None:
    missing_region = replace(_target("target_i", 10), search_region=None)
    with pytest.raises(FleetPlannerError, match="structured search_region"):
        ScriptedFleetPlanner().plan(_request(missing_region))
    missing_duration = replace(_target("target_i", 10), track_duration_s=None)
    with pytest.raises(FleetPlannerError, match="track_duration_s"):
        ScriptedFleetPlanner().plan(_request(missing_duration))


def test_scripted_planner_rejects_unavailable_and_parallel_over_capacity() -> None:
    unavailable_request = FleetMissionRequest(
        "fleet_mission_unavailable",
        1,
        "structured fleet request",
        (_uav("uav_a", available=False), _uav("uav_b")),
        (_target("target_i", 10, requested_uav_id="uav_a"),),
    )
    with pytest.raises(FleetPlannerError, match="unavailable"):
        ScriptedFleetPlanner().plan(unavailable_request)

    over_capacity = FleetMissionRequest(
        "fleet_mission_capacity",
        1,
        "structured fleet request",
        (_uav("uav_a"),),
        (_target("target_i", 10), _target("target_j", 30)),
    )
    with pytest.raises(FleetPlannerError, match="exceed available UAV"):
        ScriptedFleetPlanner().plan(over_capacity)


def test_v1_scripted_planner_rejects_explicit_sequential_targets() -> None:
    request = FleetMissionRequest(
        "fleet_mission_sequential",
        1,
        "two sequential tasks",
        (_uav("uav_a"),),
        (
            _target(
                "target_i",
                10,
                requested_uav_id="uav_a",
                start_policy=FleetStartPolicy.SEQUENTIAL,
            ),
            _target(
                "target_j",
                30,
                requested_uav_id="uav_a",
                start_policy=FleetStartPolicy.SEQUENTIAL,
            ),
        ),
    )
    with pytest.raises(FleetPlannerError, match="does not support SEQUENTIAL"):
        ScriptedFleetPlanner().plan(request)
