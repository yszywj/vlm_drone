"""Exact JSON-like parsers and trusted-request validation for Fleet plans."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from planner.spatial import CoordinateFrame, region_spec_from_dict
from target.types import TargetSpec

from fleet.types import (
    FleetAssignment,
    FleetCoordinationPolicy,
    FleetMissionError,
    FleetMissionPlan,
    FleetMissionRequest,
    FleetStartPolicy,
    FleetTargetRequest,
    FleetUavCapability,
)


def _exact(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    fields = frozenset(value)
    unknown = fields - required - optional
    missing = required - fields
    if unknown:
        raise FleetMissionError(
            f"{name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise FleetMissionError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )
    return value


def _array(value: object, name: str, *, maximum: int = 64) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    if len(value) > maximum:
        raise FleetMissionError(f"{name} must contain at most {maximum} items")
    return tuple(value)


def parse_fleet_uav_capability(value: object) -> FleetUavCapability:
    data = _exact(
        value,
        name="FleetUavCapability",
        required=frozenset(
            {
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
        ),
    )
    return FleetUavCapability(
        uav_id=data["uav_id"],  # type: ignore[arg-type]
        display_name=data["display_name"],  # type: ignore[arg-type]
        available=data["available"],  # type: ignore[arg-type]
        home_name=data["home_name"],  # type: ignore[arg-type]
        max_speed_mps=data["max_speed_mps"],  # type: ignore[arg-type]
        max_altitude_m=data["max_altitude_m"],  # type: ignore[arg-type]
        camera_modalities=_array(
            data["camera_modalities"], "camera_modalities", maximum=16
        ),  # type: ignore[arg-type]
        payload_capabilities=_array(
            data["payload_capabilities"], "payload_capabilities", maximum=32
        ),  # type: ignore[arg-type]
        remaining_energy_ratio=data["remaining_energy_ratio"],  # type: ignore[arg-type]
        current_assignment_id=data["current_assignment_id"],  # type: ignore[arg-type]
    )


def parse_fleet_coordination_policy(value: object) -> FleetCoordinationPolicy:
    data = _exact(
        value,
        name="FleetCoordinationPolicy",
        required=frozenset(
            {
                "target_claim_policy",
                "minimum_uav_separation_m",
                "route_conflict_policy",
                "assignment_failure_policy",
            }
        ),
    )
    return FleetCoordinationPolicy(
        target_claim_policy=data["target_claim_policy"],  # type: ignore[arg-type]
        minimum_uav_separation_m=data["minimum_uav_separation_m"],  # type: ignore[arg-type]
        route_conflict_policy=data["route_conflict_policy"],  # type: ignore[arg-type]
        assignment_failure_policy=data["assignment_failure_policy"],  # type: ignore[arg-type]
    )


def parse_fleet_target_request(value: object) -> FleetTargetRequest:
    data = _exact(
        value,
        name="FleetTargetRequest",
        required=frozenset(
            {
                "target_alias",
                "target_spec",
                "requested_uav_id",
                "search_region",
                "track_duration_s",
                "priority",
                "start_policy",
                "required",
            }
        ),
    )
    target_data = data["target_spec"]
    if not isinstance(target_data, Mapping):
        raise TypeError("FleetTargetRequest.target_spec must be an object")
    region_data = data["search_region"]
    target = FleetTargetRequest(
        target_alias=data["target_alias"],  # type: ignore[arg-type]
        target_spec=TargetSpec.from_dict(target_data),
        requested_uav_id=data["requested_uav_id"],  # type: ignore[arg-type]
        search_region=(
            None if region_data is None else region_spec_from_dict(region_data)
        ),
        track_duration_s=data["track_duration_s"],  # type: ignore[arg-type]
        priority=data["priority"],  # type: ignore[arg-type]
        start_policy=data["start_policy"],  # type: ignore[arg-type]
        required=data["required"],  # type: ignore[arg-type]
    )
    _validate_v1_target_start_policies((target,))
    return target


def parse_fleet_mission_request(value: object) -> FleetMissionRequest:
    data = _exact(
        value,
        name="FleetMissionRequest",
        required=frozenset(
            {
                "fleet_mission_id",
                "fleet_plan_version",
                "original_instruction",
                "uav_inventory",
                "target_requests",
                "coordination_policy",
                "assumptions",
            }
        ),
    )
    inventory = _array(data["uav_inventory"], "uav_inventory")
    targets = _array(data["target_requests"], "target_requests")
    request = FleetMissionRequest(
        fleet_mission_id=data["fleet_mission_id"],  # type: ignore[arg-type]
        fleet_plan_version=data["fleet_plan_version"],  # type: ignore[arg-type]
        original_instruction=data["original_instruction"],  # type: ignore[arg-type]
        uav_inventory=tuple(parse_fleet_uav_capability(item) for item in inventory),
        target_requests=tuple(parse_fleet_target_request(item) for item in targets),
        coordination_policy=parse_fleet_coordination_policy(
            data["coordination_policy"]
        ),
        assumptions=_array(data["assumptions"], "assumptions", maximum=32),  # type: ignore[arg-type]
    )
    _validate_v1_request_start_policies(request)
    return request


def parse_fleet_assignment(value: object) -> FleetAssignment:
    data = _exact(
        value,
        name="FleetAssignment",
        required=frozenset(
            {
                "assignment_id",
                "uav_id",
                "target_alias",
                "target_spec",
                "search_region",
                "track_duration_s",
                "priority",
                "start_policy",
            }
        ),
    )
    target_data = data["target_spec"]
    if not isinstance(target_data, Mapping):
        raise TypeError("FleetAssignment.target_spec must be an object")
    assignment = FleetAssignment(
        assignment_id=data["assignment_id"],  # type: ignore[arg-type]
        uav_id=data["uav_id"],  # type: ignore[arg-type]
        target_alias=data["target_alias"],  # type: ignore[arg-type]
        target_spec=TargetSpec.from_dict(target_data),
        search_region=region_spec_from_dict(data["search_region"]),
        track_duration_s=data["track_duration_s"],  # type: ignore[arg-type]
        priority=data["priority"],  # type: ignore[arg-type]
        start_policy=data["start_policy"],  # type: ignore[arg-type]
    )
    _validate_v1_assignment_start_policies((assignment,))
    _validate_v1_assignment_region_frames((assignment,))
    return assignment


def parse_fleet_mission_plan(
    value: object,
    *,
    request: FleetMissionRequest | None = None,
) -> FleetMissionPlan:
    data = _exact(
        value,
        name="FleetMissionPlan",
        required=frozenset(
            {
                "schema_version",
                "fleet_mission_id",
                "fleet_plan_version",
                "assignments",
                "coordination_policy",
                "assumptions",
                "unassigned_requirements",
            }
        ),
    )
    raw_assignments = _array(data["assignments"], "assignments")
    plan = FleetMissionPlan(
        schema_version=data["schema_version"],  # type: ignore[arg-type]
        fleet_mission_id=data["fleet_mission_id"],  # type: ignore[arg-type]
        fleet_plan_version=data["fleet_plan_version"],  # type: ignore[arg-type]
        assignments=tuple(
            parse_fleet_assignment(item) for item in raw_assignments
        ),
        coordination_policy=parse_fleet_coordination_policy(
            data["coordination_policy"]
        ),
        assumptions=_array(data["assumptions"], "assumptions", maximum=32),  # type: ignore[arg-type]
        unassigned_requirements=_array(
            data["unassigned_requirements"],
            "unassigned_requirements",
            maximum=64,
        ),  # type: ignore[arg-type]
    )
    _validate_v1_assignment_start_policies(plan.assignments)
    if request is not None:
        validate_fleet_mission_plan(plan, request)
    return plan


def validate_fleet_mission_plan(
    plan: FleetMissionPlan,
    request: FleetMissionRequest,
) -> FleetMissionPlan:
    """Validate model/scripted assignments against trusted allow-lists."""

    if not isinstance(plan, FleetMissionPlan):
        raise TypeError("plan must be a FleetMissionPlan")
    if not isinstance(request, FleetMissionRequest):
        raise TypeError("request must be a FleetMissionRequest")
    if (
        plan.fleet_mission_id != request.fleet_mission_id
        or plan.fleet_plan_version != request.fleet_plan_version
    ):
        raise FleetMissionError(
            "FleetMissionPlan routing/version must exactly echo the request"
        )
    if plan.coordination_policy != request.coordination_policy:
        raise FleetMissionError(
            "FleetMissionPlan coordination_policy must exactly echo trusted limits"
        )
    _validate_v1_request_start_policies(request)
    _validate_v1_assignment_start_policies(plan.assignments)
    _validate_v1_assignment_region_frames(plan.assignments)
    inventory = {item.uav_id: item for item in request.uav_inventory}
    target_directory = {
        item.target_alias: item for item in request.target_requests
    }
    available_ids = set(request.available_uav_ids)
    for assignment in plan.assignments:
        capability = inventory.get(assignment.uav_id)
        if capability is None:
            raise FleetMissionError(
                f"assignment references unknown uav_id: {assignment.uav_id}"
            )
        if assignment.uav_id not in available_ids:
            raise FleetMissionError(
                f"assignment references unavailable uav_id: {assignment.uav_id}"
            )
        if (
            capability.current_assignment_id is not None
            and capability.current_assignment_id != assignment.assignment_id
        ):
            raise FleetMissionError(
                f"uav_id is already assigned: {assignment.uav_id}"
            )
        target = target_directory.get(assignment.target_alias)
        if target is None:
            raise FleetMissionError(
                "assignment target_alias is outside the trusted target allowlist: "
                + assignment.target_alias
            )
        if assignment.target_spec != target.target_spec:
            raise FleetMissionError(
                f"assignment target_spec changed for {assignment.target_alias}"
            )
        if (
            target.requested_uav_id is not None
            and assignment.uav_id != target.requested_uav_id
        ):
            raise FleetMissionError(
                f"assignment changed requested UAV relation for {target.target_alias}"
            )
        if (
            target.search_region is not None
            and assignment.search_region != target.search_region
        ):
            raise FleetMissionError(
                f"assignment changed requested RegionSpec for {target.target_alias}"
            )
        if (
            target.track_duration_s is not None
            and assignment.track_duration_s != target.track_duration_s
        ):
            raise FleetMissionError(
                f"assignment changed requested track duration for {target.target_alias}"
            )
        if (
            target.priority is not None
            and assignment.priority != target.priority
        ) or (
            target.start_policy is not None
            and assignment.start_policy is not target.start_policy
        ):
            raise FleetMissionError(
                f"assignment changed requested priority/start policy for {target.target_alias}"
            )
    if len(plan.assignments) > len(available_ids):
        raise FleetMissionError(
            "Fleet Planner v1 assignments exceed the available UAV count"
        )
    assignment_counts = Counter(assignment.target_alias for assignment in plan.assignments)
    explicitly_unassigned = _explicitly_unassigned_aliases(
        plan.unassigned_requirements,
        request,
    )
    invalid_required_coverage = tuple(
        (
            target.target_alias,
            assignment_counts[target.target_alias],
            target.target_alias in explicitly_unassigned,
        )
        for target in request.target_requests
        if target.required
        and (
            assignment_counts[target.target_alias] > 1
            or (
                assignment_counts[target.target_alias] == 0
                and target.target_alias not in explicitly_unassigned
            )
            or (
                assignment_counts[target.target_alias] == 1
                and target.target_alias in explicitly_unassigned
            )
        )
    )
    if invalid_required_coverage:
        details = ", ".join(
            f"{alias}(assignments={count}, explicitly_unassigned={unassigned})"
            for alias, count, unassigned in invalid_required_coverage
        )
        raise FleetMissionError(
            "each required target must be covered by exactly one assignment "
            "or explicitly named in unassigned_requirements, but not both; "
            + details
        )
    return plan


def _validate_v1_request_start_policies(
    request: FleetMissionRequest,
) -> None:
    _validate_v1_target_start_policies(request.target_requests)


def _validate_v1_target_start_policies(
    targets: Sequence[FleetTargetRequest],
) -> None:
    sequential_targets = tuple(
        target.target_alias
        for target in targets
        if target.start_policy is FleetStartPolicy.SEQUENTIAL
    )
    if sequential_targets:
        raise FleetMissionError(
            "Fleet Planner v1 does not support SEQUENTIAL target requests: "
            + ", ".join(sequential_targets)
        )


def _validate_v1_assignment_start_policies(
    assignments: Sequence[FleetAssignment],
) -> None:
    sequential_assignments = tuple(
        assignment.assignment_id
        for assignment in assignments
        if assignment.start_policy is FleetStartPolicy.SEQUENTIAL
    )
    if sequential_assignments:
        raise FleetMissionError(
            "Fleet Planner v1 does not support SEQUENTIAL assignments: "
            + ", ".join(sequential_assignments)
        )


def _validate_v1_assignment_region_frames(
    assignments: Sequence[FleetAssignment],
) -> None:
    allowed = frozenset(
        {
            CoordinateFrame.WORLD_ENU,
            CoordinateFrame.HOME_ENU,
        }
    )
    invalid = tuple(
        f"{assignment.assignment_id}={assignment.search_region.frame.value}"
        for assignment in assignments
        if assignment.search_region.frame not in allowed
    )
    if invalid:
        raise FleetMissionError(
            "Fleet Planner v1 assignment RegionSpec frame must be WORLD_ENU "
            "or HOME_ENU: "
            + ", ".join(invalid)
        )


def _explicitly_unassigned_aliases(
    entries: Sequence[str],
    request: FleetMissionRequest,
) -> frozenset[str]:
    """Extract aliases only from an exact ``alias`` or ``alias: reason`` prefix."""

    known = frozenset(request.target_aliases)
    result: set[str] = set()
    for entry in entries:
        prefix, separator, reason = entry.partition(":")
        alias = prefix.strip()
        if alias not in known:
            continue
        if separator and not reason.strip():
            continue
        if not separator and entry.strip() != alias:
            continue
        result.add(alias)
    return frozenset(result)


__all__ = [
    "parse_fleet_assignment",
    "parse_fleet_coordination_policy",
    "parse_fleet_mission_plan",
    "parse_fleet_mission_request",
    "parse_fleet_target_request",
    "parse_fleet_uav_capability",
    "validate_fleet_mission_plan",
]
