"""Request-bound JSON Schema for structured FleetMissionPlan output."""

from __future__ import annotations

from copy import deepcopy

from common.ids import ROUTING_ID_PATTERN_TEXT
from planner.json_schema_v3 import region_spec_json_schema
from planner.spatial import CoordinateFrame

from fleet.types import FleetMissionRequest, FleetStartPolicy


def _constrain_region_frames(
    schema: dict[str, object],
    trusted_frames: tuple[str, ...],
) -> dict[str, object]:
    result = deepcopy(schema)
    variants = result.get("oneOf")
    if not isinstance(variants, list):
        raise TypeError("RegionSpec JSON schema has an unexpected shape")
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        properties = variant.get("properties")
        if not isinstance(properties, dict) or "frame" not in properties:
            continue
        properties["frame"] = {"type": "string", "enum": list(trusted_frames)}
    return result


def build_fleet_mission_plan_json_schema(
    request: FleetMissionRequest,
    *,
    trusted_spatial_frames: tuple[CoordinateFrame | str, ...] = (
        CoordinateFrame.WORLD_ENU,
        CoordinateFrame.HOME_ENU,
    ),
) -> dict[str, object]:
    """Bind routing, inventory, target directory, and user relationships."""

    if not isinstance(request, FleetMissionRequest):
        raise TypeError("request must be a FleetMissionRequest")
    normalized_frames: list[str] = []
    for value in trusted_spatial_frames:
        try:
            frame = value if isinstance(value, CoordinateFrame) else CoordinateFrame(value)
        except (TypeError, ValueError):
            raise ValueError("trusted_spatial_frames contains an invalid frame") from None
        if frame.value not in normalized_frames:
            normalized_frames.append(frame.value)
    if not normalized_frames:
        raise ValueError("trusted_spatial_frames must not be empty")
    sequential_targets = tuple(
        target.target_alias
        for target in request.target_requests
        if target.start_policy is FleetStartPolicy.SEQUENTIAL
    )
    if sequential_targets:
        raise ValueError(
            "Fleet Planner v1 does not support SEQUENTIAL target requests: "
            + ", ".join(sequential_targets)
        )
    available = list(request.available_uav_ids)
    if not available:
        raise ValueError("Fleet planning requires at least one available UAV")
    region_schema = _constrain_region_frames(
        region_spec_json_schema(), tuple(normalized_frames)
    )
    assignment_variants: list[dict[str, object]] = []
    for target in request.target_requests:
        uav_schema: dict[str, object]
        if target.requested_uav_id is None:
            uav_schema = {"type": "string", "enum": available}
        else:
            uav_schema = {"type": "string", "const": target.requested_uav_id}
        assignment_variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "assignment_id",
                    "uav_id",
                    "target_alias",
                    "target_spec",
                    "search_region",
                    "track_duration_s",
                    "priority",
                    "start_policy",
                ],
                "properties": {
                    "assignment_id": {
                        "type": "string",
                        "pattern": ROUTING_ID_PATTERN_TEXT,
                    },
                    "uav_id": uav_schema,
                    "target_alias": {
                        "type": "string",
                        "const": target.target_alias,
                    },
                    "target_spec": {
                        "type": "object",
                        "const": target.target_spec.to_dict(),
                    },
                    "search_region": (
                        {"const": target.search_region.to_dict()}
                        if target.search_region is not None
                        else deepcopy(region_schema)
                    ),
                    "track_duration_s": (
                        {"type": "number", "const": target.track_duration_s}
                        if target.track_duration_s is not None
                        else {
                            "type": "number",
                            "exclusiveMinimum": 0.0,
                            "maximum": 3600.0,
                        }
                    ),
                    "priority": (
                        {"type": "integer", "const": target.priority}
                        if target.priority is not None
                        else {"type": "integer", "minimum": 0, "maximum": 1000}
                    ),
                    "start_policy": (
                        {
                            "type": "string",
                            "const": target.start_policy.value,
                        }
                        if target.start_policy is not None
                        else {
                            "type": "string",
                            "const": FleetStartPolicy.PARALLEL.value,
                        }
                    ),
                },
            }
        )
    maximum_assignments = min(len(available), len(request.target_requests))
    policy = request.coordination_policy
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "fleet_mission_id",
            "fleet_plan_version",
            "assignments",
            "coordination_policy",
            "assumptions",
            "unassigned_requirements",
        ],
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "fleet_mission_id": {
                "type": "string",
                "const": request.fleet_mission_id,
            },
            "fleet_plan_version": {
                "type": "integer",
                "const": request.fleet_plan_version,
            },
            "assignments": {
                "type": "array",
                "maxItems": maximum_assignments,
                "items": {"oneOf": assignment_variants},
            },
            "coordination_policy": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "target_claim_policy",
                    "minimum_uav_separation_m",
                    "route_conflict_policy",
                    "assignment_failure_policy",
                ],
                "properties": {
                    "target_claim_policy": {
                        "type": "string",
                        "const": policy.target_claim_policy.value,
                    },
                    "minimum_uav_separation_m": {
                        "type": "number",
                        "const": policy.minimum_uav_separation_m,
                    },
                    "route_conflict_policy": {
                        "type": "string",
                        "const": policy.route_conflict_policy.value,
                    },
                    "assignment_failure_policy": {
                        "type": "string",
                        "const": policy.assignment_failure_policy.value,
                    },
                },
            },
            "assumptions": {
                "type": "array",
                "maxItems": 32,
                "items": {"type": "string", "minLength": 1, "maxLength": 512},
            },
            "unassigned_requirements": {
                "type": "array",
                "maxItems": 64,
                "items": {"type": "string", "minLength": 1, "maxLength": 512},
            },
        },
    }


__all__ = ["build_fleet_mission_plan_json_schema"]
