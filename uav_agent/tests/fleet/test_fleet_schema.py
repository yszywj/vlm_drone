from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from fleet.json_schema import build_fleet_mission_plan_json_schema
from fleet.schemas import (
    parse_fleet_mission_plan,
    parse_fleet_mission_request,
    validate_fleet_mission_plan,
)
from fleet.scripted_planner import ScriptedFleetPlanner
from fleet.types import (
    FleetAssignment,
    FleetMissionError,
    FleetMissionPlan,
    FleetMissionRequest,
    FleetStartPolicy,
    FleetTargetRequest,
    FleetUavCapability,
)
from planner.spatial import CircleRegion, CoordinateFrame, RectangleRegion
from target.types import TargetSpec


def _request() -> FleetMissionRequest:
    target_i = TargetSpec("red moving target i", immutable_identity_summary="target i")
    target_j = TargetSpec("blue moving target j", immutable_identity_summary="target j")
    return FleetMissionRequest(
        fleet_mission_id="fleet_mission_schema",
        fleet_plan_version=7,
        original_instruction="uav_a handles i and uav_b handles j",
        uav_inventory=(
            FleetUavCapability(
                "uav_a", "A", True, "home_a", 5, 30, ("RGB",), ()
            ),
            FleetUavCapability(
                "uav_b", "B", True, "home_b", 5, 30, ("RGB",), ()
            ),
        ),
        target_requests=(
            FleetTargetRequest(
                "target_i",
                target_i,
                "uav_a",
                CircleRegion(CoordinateFrame.WORLD_ENU, (20, 30, 0), 15),
                20,
            ),
            FleetTargetRequest(
                "target_j",
                target_j,
                "uav_b",
                RectangleRegion(
                    CoordinateFrame.HOME_ENU, (5, 10, 0), 12, 8
                ),
                15,
            ),
        ),
    )


def test_plan_parser_round_trips_regions_and_rejects_unknown_fields() -> None:
    request = _request()
    payload = ScriptedFleetPlanner().plan(request).to_dict()
    parsed = parse_fleet_mission_plan(payload, request=request)
    assert parsed.to_dict() == payload
    assert isinstance(parsed.assignments[0].search_region, CircleRegion)
    assert isinstance(parsed.assignments[1].search_region, RectangleRegion)

    tampered = deepcopy(payload)
    tampered["unexpected"] = True
    with pytest.raises(FleetMissionError, match="unknown fields"):
        parse_fleet_mission_plan(tampered, request=request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("uav_id", "uav_c", "unknown uav_id"),
        ("target_alias", "target_unknown", "target allowlist"),
    ),
)
def test_plan_parser_rejects_inventory_and_target_allowlist_violations(
    field: str, value: str, message: str
) -> None:
    request = _request()
    payload = ScriptedFleetPlanner().plan(request).to_dict()
    payload["assignments"][0][field] = value  # type: ignore[index]
    with pytest.raises(FleetMissionError, match=message):
        parse_fleet_mission_plan(payload, request=request)


def test_plan_parser_rejects_trusted_routing_and_relationship_changes() -> None:
    request = _request()
    payload = ScriptedFleetPlanner().plan(request).to_dict()
    payload["fleet_plan_version"] = 8
    with pytest.raises(FleetMissionError, match="routing/version"):
        parse_fleet_mission_plan(payload, request=request)

    payload = ScriptedFleetPlanner().plan(request).to_dict()
    payload["assignments"][0]["uav_id"] = "uav_b"  # type: ignore[index]
    payload["assignments"][1]["uav_id"] = "uav_a"  # type: ignore[index]
    with pytest.raises(FleetMissionError, match="requested UAV relation"):
        parse_fleet_mission_plan(payload, request=request)


def test_request_bound_json_schema_has_only_high_level_assignments() -> None:
    request = _request()
    schema = build_fleet_mission_plan_json_schema(request)
    properties = schema["properties"]
    assert properties["fleet_mission_id"]["const"] == request.fleet_mission_id
    assert properties["fleet_plan_version"]["const"] == 7
    assert properties["assignments"]["maxItems"] == 2
    variants = properties["assignments"]["items"]["oneOf"]
    assert {item["properties"]["target_alias"]["const"] for item in variants} == {
        "target_i",
        "target_j",
    }
    assert variants[0]["properties"]["uav_id"] == {
        "type": "string",
        "const": "uav_a",
    }
    serialized = repr(schema).lower()
    assert "steps" not in serialized
    assert "pid" not in serialized
    assert "motor" not in serialized


def test_json_schema_region_frames_are_explicitly_constrained() -> None:
    request = _request()
    unbound_target = FleetTargetRequest(
        "target_free",
        TargetSpec("free target"),
        requested_uav_id=None,
        search_region=None,
        track_duration_s=None,
    )
    free_request = FleetMissionRequest(
        "fleet_mission_free_region",
        1,
        "choose a trusted region",
        request.uav_inventory,
        (unbound_target,),
    )
    schema = build_fleet_mission_plan_json_schema(free_request)
    region = schema["properties"]["assignments"]["items"]["oneOf"][0][
        "properties"
    ]["search_region"]
    frames = {
        tuple(variant["properties"]["frame"]["enum"])
        for variant in region["oneOf"]
        if "frame" in variant["properties"]
    }
    assert frames == {("WORLD_ENU", "HOME_ENU")}


def test_parser_and_validator_reject_region_frame_outside_json_schema() -> None:
    request = _request()
    free_request = replace(
        request,
        target_requests=(
            FleetTargetRequest(
                "target_i",
                request.target_requests[0].target_spec,
                requested_uav_id="uav_a",
                search_region=None,
                track_duration_s=20,
            ),
        ),
    )
    assignment = FleetAssignment(
        "assignment_uav_a_target_i",
        "uav_a",
        "target_i",
        free_request.target_requests[0].target_spec,
        CircleRegion(CoordinateFrame.CAMERA_FLU, (1, 2, 3), 5),
        20,
    )
    plan = FleetMissionPlan(
        fleet_mission_id=free_request.fleet_mission_id,
        fleet_plan_version=free_request.fleet_plan_version,
        assignments=(assignment,),
        coordination_policy=free_request.coordination_policy,
    )

    with pytest.raises(FleetMissionError, match="WORLD_ENU or HOME_ENU"):
        parse_fleet_mission_plan(plan.to_dict(), request=free_request)
    with pytest.raises(FleetMissionError, match="WORLD_ENU or HOME_ENU"):
        validate_fleet_mission_plan(plan, free_request)


def test_required_target_must_be_assigned_or_explicitly_named_unassigned() -> None:
    request = _request()
    payload = ScriptedFleetPlanner().plan(request).to_dict()
    payload["assignments"] = payload["assignments"][:1]

    with pytest.raises(FleetMissionError, match="required target"):
        parse_fleet_mission_plan(payload, request=request)

    payload["unassigned_requirements"] = ["target_j: no eligible UAV"]
    parsed = parse_fleet_mission_plan(payload, request=request)
    assert [item.target_alias for item in parsed.assignments] == ["target_i"]


def test_required_target_cannot_be_both_assigned_and_unassigned() -> None:
    request = _request()
    payload = ScriptedFleetPlanner().plan(request).to_dict()
    payload["unassigned_requirements"] = ["target_j: contradictory output"]

    with pytest.raises(FleetMissionError, match="but not both"):
        parse_fleet_mission_plan(payload, request=request)


def test_unassigned_target_alias_requires_an_exact_prefix() -> None:
    request = _request()
    payload = ScriptedFleetPlanner().plan(request).to_dict()
    payload["assignments"] = payload["assignments"][:1]
    payload["unassigned_requirements"] = [
        "target_j_extra: text merely contains target_j"
    ]

    with pytest.raises(FleetMissionError, match="explicitly named"):
        parse_fleet_mission_plan(payload, request=request)


def test_v1_parsers_and_schema_reject_sequential_policy() -> None:
    request = _request()
    request_payload = request.to_dict()
    request_payload["target_requests"][0]["start_policy"] = "SEQUENTIAL"
    with pytest.raises(FleetMissionError, match="SEQUENTIAL target requests"):
        parse_fleet_mission_request(request_payload)

    sequential_request = replace(
        request,
        target_requests=(
            replace(
                request.target_requests[0],
                start_policy=FleetStartPolicy.SEQUENTIAL,
            ),
            request.target_requests[1],
        ),
    )
    with pytest.raises(ValueError, match="SEQUENTIAL target requests"):
        build_fleet_mission_plan_json_schema(sequential_request)

    plan_payload = ScriptedFleetPlanner().plan(request).to_dict()
    plan_payload["assignments"][0]["start_policy"] = "SEQUENTIAL"
    with pytest.raises(FleetMissionError, match="SEQUENTIAL assignments"):
        parse_fleet_mission_plan(plan_payload)


def test_unbound_start_policy_is_parallel_and_never_expands_capacity() -> None:
    request = _request()
    extra = FleetTargetRequest(
        "target_k",
        TargetSpec("green target k", immutable_identity_summary="target k"),
        requested_uav_id=None,
        search_region=None,
        track_duration_s=None,
        start_policy=None,
        required=True,
    )
    unbound = replace(
        request,
        target_requests=(
            replace(request.target_requests[0], start_policy=None),
            replace(request.target_requests[1], start_policy=None),
            extra,
        ),
    )

    schema = build_fleet_mission_plan_json_schema(unbound)
    assignments = schema["properties"]["assignments"]
    assert assignments["maxItems"] == len(request.available_uav_ids)
    assert {
        variant["properties"]["start_policy"]["const"]
        for variant in assignments["items"]["oneOf"]
    } == {"PARALLEL"}
