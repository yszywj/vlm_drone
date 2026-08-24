"""Request-bound Structured Output Schema for Fleet Planner V2."""

from __future__ import annotations

from common.ids import ROUTING_ID_PATTERN_TEXT

from fleet.types import FleetStartPolicy
from fleet.types_v2 import (
    DeviationReasonCode,
    FleetMissionRequestV2,
    MAX_ASSIGNMENT_DEVIATIONS,
    MAX_ASSIGNMENT_GOALS,
)


def _object(properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _text(maximum: int = 512) -> dict[str, object]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def build_fleet_mission_request_v2_json_schema(
    request: FleetMissionRequestV2,
) -> dict[str, object]:
    """Describe the exact trusted request snapshot supplied to a planner."""

    if not isinstance(request, FleetMissionRequestV2):
        raise TypeError("request must be a FleetMissionRequestV2")
    payload = request.to_dict()
    return _object(
        {
            key: {"const": value}
            for key, value in payload.items()
        },
        list(payload),
    )


def build_fleet_mission_plan_v2_json_schema(
    request: FleetMissionRequestV2,
) -> dict[str, object]:
    """Bind routing and trusted enums without locking user assignment wishes.

    ``uniqueItems`` is deliberately absent because vLLM 0.27.1 returns HTTP
    500 while compiling that keyword.  The strict V2 Python contracts enforce
    all duplicate invariants after decoding.
    """

    if not isinstance(request, FleetMissionRequestV2):
        raise TypeError("request must be a FleetMissionRequestV2")
    available_uavs = list(request.available_uav_ids)
    if not available_uavs:
        raise ValueError("Fleet V2 planning requires at least one available UAV")
    goal_ids = list(request.task_spec.all_goal_ids)
    if not goal_ids:
        raise ValueError("Fleet V2 planning requires at least one Goal")
    constraint_ids = [
        item.constraint_id for item in request.task_spec.assignment_constraints
    ]
    evidence_ids = list(request.trusted_evidence_ids)
    # xgrammar also raises an internal error for ``enum: []``.  When no
    # trusted deviation can be expressed, make the array structurally empty
    # without compiling unreachable child enums.  Python still validates all
    # deviation references when this branch is enabled.
    if constraint_ids and evidence_ids:
        deviation = _object(
            {
                "constraint_id": {
                    "type": "string",
                    "enum": constraint_ids,
                },
                "reason_code": {
                    "type": "string",
                    "enum": [item.value for item in DeviationReasonCode],
                },
                "evidence_refs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "items": {"type": "string", "enum": evidence_ids},
                },
            },
            ["constraint_id", "reason_code", "evidence_refs"],
        )
        deviations_schema: dict[str, object] = {
            "type": "array",
            "maxItems": min(MAX_ASSIGNMENT_DEVIATIONS, len(constraint_ids)),
            "items": deviation,
        }
    else:
        deviations_schema = {
            "type": "array",
            "maxItems": 0,
        }
    assignment = _object(
        {
            "assignment_id": {
                "type": "string",
                "pattern": ROUTING_ID_PATTERN_TEXT,
            },
            # This is intentionally an enum, never a const derived from an
            # AssignmentConstraint.  The Fleet model owns the final choice.
            "uav_id": {"type": "string", "enum": available_uavs},
            "goal_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": min(MAX_ASSIGNMENT_GOALS, len(goal_ids)),
                "items": {"type": "string", "enum": goal_ids},
            },
            "priority": {"type": "integer", "minimum": 0, "maximum": 1000},
            "start_policy": {
                "type": "string",
                # FleetMissionRequest/FleetMissionPlan V1 lowering, used by
                # both current linear and graph runtimes, intentionally has no
                # SEQUENTIAL execution semantics.  Do not advertise a V2
                # value that the trusted runtime must reject later.
                "enum": [FleetStartPolicy.PARALLEL.value],
            },
            "deviations": deviations_schema,
        },
        [
            "assignment_id",
            "uav_id",
            "goal_ids",
            "priority",
            "start_policy",
            "deviations",
        ],
    )
    policy = request.coordination_policy.to_dict()
    coordination = _object(
        {key: {"const": value} for key, value in policy.items()},
        list(policy),
    )
    return _object(
        {
            "schema_version": {"type": "integer", "const": 2},
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
                "maxItems": min(len(available_uavs), len(goal_ids)),
                "items": assignment,
            },
            "coordination_policy": coordination,
            "assumptions": {
                "type": "array",
                "maxItems": 32,
                "items": _text(),
            },
            "unassigned_goal_ids": {
                "type": "array",
                "maxItems": len(goal_ids),
                "items": {"type": "string", "enum": goal_ids},
            },
        },
        [
            "schema_version",
            "fleet_mission_id",
            "fleet_plan_version",
            "assignments",
            "coordination_policy",
            "assumptions",
            "unassigned_goal_ids",
        ],
    )


__all__ = [
    "build_fleet_mission_plan_v2_json_schema",
    "build_fleet_mission_request_v2_json_schema",
]
