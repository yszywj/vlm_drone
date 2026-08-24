from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from fleet.types import (
    FleetAssignment,
    FleetCoordinationPolicy,
    FleetMissionError,
    FleetMissionPlan,
    FleetMissionRequest,
    FleetPlanPatch,
    FleetStartPolicy,
    FleetTargetRequest,
    FleetUavCapability,
    TargetClaimPolicy,
)
from planner.spatial import CircleRegion, CoordinateFrame
from target.types import TargetSpec


def _target(name: str) -> TargetSpec:
    return TargetSpec(
        f"red moving target {name}",
        category="vehicle",
        hard_attributes=("red", "moving"),
        immutable_identity_summary=f"red moving target {name}",
    )


def _region(x: float) -> CircleRegion:
    return CircleRegion(CoordinateFrame.WORLD_ENU, (x, 10.0, 0.0), 8.0)


def _uav(uav_id: str) -> FleetUavCapability:
    return FleetUavCapability(
        uav_id=uav_id,
        display_name=uav_id.upper(),
        available=True,
        home_name=f"home_{uav_id[-1]}",
        max_speed_mps=5.0,
        max_altitude_m=30.0,
    )


def _assignment(
    assignment_id: str,
    uav_id: str,
    alias: str,
    *,
    start_policy: FleetStartPolicy = FleetStartPolicy.PARALLEL,
) -> FleetAssignment:
    suffix = alias[-1]
    return FleetAssignment(
        assignment_id=assignment_id,
        uav_id=uav_id,
        target_alias=alias,
        target_spec=_target(suffix),
        search_region=_region(10.0 if suffix == "i" else 30.0),
        track_duration_s=20.0,
        priority=100,
        start_policy=start_policy,
    )


def test_capability_and_request_are_immutable_exact_round_trips() -> None:
    capability = _uav("uav_a")
    payload = capability.to_dict()
    assert FleetUavCapability.from_dict(payload) == capability
    assert set(payload) == {
        "uav_id",
        "display_name",
        "available",
        "home_name",
        "max_speed_mps",
        "max_altitude_m",
        "camera_modalities",
        "payload_capabilities",
        "remaining_energy_ratio",
        "current_assignment_id",
    }
    assert not {"pid", "motor", "thrust", "yaw_rate"} & set(payload)

    request = FleetMissionRequest(
        fleet_mission_id="fleet_mission_types",
        fleet_plan_version=2,
        original_instruction="assign the two trusted targets",
        uav_inventory=(capability, _uav("uav_b")),
        target_requests=(
            FleetTargetRequest("target_i", _target("i"), "uav_a", _region(10), 20),
            FleetTargetRequest("target_j", _target("j"), "uav_b", _region(30), 20),
        ),
        assumptions=("both UAVs are ready",),
    )
    assert FleetMissionRequest.from_dict(request.to_dict()) == request
    assert request.available_uav_ids == ("uav_a", "uav_b")
    assert request.target_aliases == ("target_i", "target_j")
    assert request.uav("uav_a") == capability
    assert request.target_request("target_j").target_spec == _target("j")
    with pytest.raises(FrozenInstanceError):
        request.fleet_plan_version = 3  # type: ignore[misc]


def test_request_rejects_duplicate_and_unknown_inventory_relations() -> None:
    target = FleetTargetRequest("target_i", _target("i"), "uav_a", _region(10), 20)
    with pytest.raises(FleetMissionError, match="duplicate uav_id"):
        FleetMissionRequest(
            "fleet_mission_dup_uav",
            1,
            "task",
            (_uav("uav_a"), _uav("uav_a")),
            (target,),
        )
    with pytest.raises(FleetMissionError, match="duplicate target_alias"):
        FleetMissionRequest(
            "fleet_mission_dup_target",
            1,
            "task",
            (_uav("uav_a"),),
            (target, target),
        )
    with pytest.raises(FleetMissionError, match="unknown requested UAV"):
        FleetMissionRequest(
            "fleet_mission_unknown_uav",
            1,
            "task",
            (_uav("uav_a"),),
            (
                FleetTargetRequest(
                    "target_i", _target("i"), "uav_missing", _region(10), 20
                ),
            ),
        )


def test_plan_rejects_duplicate_assignment_uav_and_exclusive_claims() -> None:
    first = _assignment("assignment_a_i", "uav_a", "target_i")
    second = _assignment("assignment_b_j", "uav_b", "target_j")
    policy = FleetCoordinationPolicy()
    plan = FleetMissionPlan(
        "fleet_mission_plan",
        1,
        (first, second),
        policy,
    )
    assert FleetMissionPlan.from_dict(plan.to_dict()) == plan

    with pytest.raises(FleetMissionError, match="duplicate assignment_id"):
        FleetMissionPlan("fleet_mission_plan", 1, (first, first), policy)

    same_uav = replace(second, uav_id="uav_a")
    with pytest.raises(FleetMissionError, match="multiple active assignments"):
        FleetMissionPlan("fleet_mission_plan", 1, (first, same_uav), policy)

    same_target = replace(
        second,
        target_alias=first.target_alias,
        target_spec=first.target_spec,
        search_region=first.search_region,
    )
    with pytest.raises(FleetMissionError, match="EXCLUSIVE"):
        FleetMissionPlan("fleet_mission_plan", 1, (first, same_target), policy)

    shared = replace(policy, target_claim_policy=TargetClaimPolicy.SHARED)
    assert len(
        FleetMissionPlan(
            "fleet_mission_shared", 1, (first, same_target), shared
        ).assignments
    ) == 2


def test_sequential_assignments_and_patch_version_are_explicit() -> None:
    first = _assignment(
        "assignment_a_i",
        "uav_a",
        "target_i",
        start_policy=FleetStartPolicy.SEQUENTIAL,
    )
    second = _assignment(
        "assignment_a_j",
        "uav_a",
        "target_j",
        start_policy=FleetStartPolicy.SEQUENTIAL,
    )
    policy = FleetCoordinationPolicy()
    plan = FleetMissionPlan(
        "fleet_mission_sequence", 3, (first, second), policy
    )
    assert [item.uav_id for item in plan.assignments] == ["uav_a", "uav_a"]
    patch = FleetPlanPatch(
        fleet_mission_id=plan.fleet_mission_id,
        base_fleet_plan_version=3,
        new_fleet_plan_version=4,
        replacement_assignments=(second,),
        coordination_policy=policy,
        reason_codes=("UAV_UNAVAILABLE",),
    )
    assert patch.to_dict()["new_fleet_plan_version"] == 4
    with pytest.raises(FleetMissionError, match=r"base_fleet_plan_version \+ 1"):
        replace(patch, new_fleet_plan_version=5)
