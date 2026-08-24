"""Request-bound Structured Output Schema for :mod:`fleet.task_spec`."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence

from common.ids import ROUTING_ID_PATTERN_TEXT, validate_routing_id, validate_uav_id
from planner.json_schema_v3 import region_spec_json_schema, spatial_target_json_schema
from planner.spatial import CoordinateFrame

from fleet.task_spec import (
    ConstraintStrength,
    MAX_AMBIGUITIES,
    MAX_ASSIGNMENT_CONSTRAINTS,
    MAX_GOALS,
    MAX_ORDERING_CONSTRAINTS,
    MAX_SOURCE_EVIDENCE,
    MAX_TERMINATION_GOALS,
    GoalType,
    TERMINATION_GOAL_TYPES,
)


def _object(properties: dict[str, object], required: Sequence[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _nullable(schema: dict[str, object]) -> dict[str, object]:
    return {"oneOf": [schema, {"type": "null"}]}


def _bounded_text(maximum: int = 512) -> dict[str, object]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _id() -> dict[str, object]:
    return {"type": "string", "pattern": ROUTING_ID_PATTERN_TEXT}


def _id_array(maximum: int = 16, *, minimum: int = 0) -> dict[str, object]:
    # vLLM 0.27.1's structured-output grammar returns HTTP 500 when it sees
    # ``uniqueItems``.  The strict trusted TaskSpec parser independently
    # rejects duplicates for every corresponding tuple.
    return {
        "type": "array",
        "minItems": minimum,
        "maxItems": maximum,
        "items": _id(),
    }


def _constrain_frames(schema: dict[str, object], frames: tuple[str, ...]) -> dict[str, object]:
    result = deepcopy(schema)

    def walk(value: object) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict) and "frame" in properties:
                properties["frame"] = {"type": "string", "enum": list(frames)}
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(result)
    return result


def _normalized_ids(values: Sequence[str], name: str, *, uav: bool) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = validate_uav_id(value) if uav else validate_routing_id(value, name)
        if normalized not in result:
            result.append(normalized)
    if uav and not result:
        raise ValueError("trusted_uav_ids must not be empty")
    return tuple(result)


def _normalized_frames(
    values: Sequence[CoordinateFrame | str],
) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        try:
            normalized = (
                value if isinstance(value, CoordinateFrame) else CoordinateFrame(value)
            ).value
        except (TypeError, ValueError):
            raise ValueError(
                "supported_coordinate_frames contains an invalid frame"
            ) from None
        if normalized not in result:
            result.append(normalized)
    if not result:
        raise ValueError("supported_coordinate_frames must not be empty")
    return tuple(result)


def build_fleet_task_spec_json_schema(
    *,
    source_text: str,
    trusted_uav_ids: Sequence[str],
    trusted_target_aliases: Sequence[str],
    supported_coordinate_frames: Sequence[CoordinateFrame | str] = (
        CoordinateFrame.WORLD_ENU,
        CoordinateFrame.HOME_ENU,
        CoordinateFrame.UAV_START_FLU,
    ),
) -> dict[str, object]:
    """Build a strict schema whose entity/frame fields use trusted enums only."""

    if not isinstance(source_text, str) or not source_text.strip() or len(source_text) > 8192:
        raise ValueError("source_text must contain between 1 and 8192 characters")
    uav_ids = _normalized_ids(trusted_uav_ids, "uav_id", uav=True)
    targets = _normalized_ids(
        trusted_target_aliases, "target_alias", uav=False
    )
    frames = _normalized_frames(supported_coordinate_frames)
    strength = {
        "type": "string",
        "enum": [item.value for item in ConstraintStrength],
    }
    target_schema: dict[str, object]
    if targets:
        target_schema = _nullable({"type": "string", "enum": list(targets)})
    else:
        target_schema = {"type": "null"}
    general_spatial = {
        "oneOf": [
            {"type": "null"},
            _constrain_frames(region_spec_json_schema(), frames),
            _constrain_frames(spatial_target_json_schema(), frames),
        ]
    }
    positive_seconds = _nullable(
        {"type": "number", "exclusiveMinimum": 0.0, "maximum": 86400.0}
    )
    positive_distance = _nullable(
        {"type": "number", "exclusiveMinimum": 0.0, "maximum": 100000.0}
    )
    common_goal_properties: dict[str, object] = {
        "goal_id": _id(),
        "target_alias": target_schema,
        "duration_s": positive_seconds,
        "distance_m": positive_distance,
        "strength": strength,
        "evidence_refs": _id_array(),
    }
    goal_required = [
        "goal_id",
        "goal_type",
        "target_alias",
        "spatial_constraint",
        "duration_s",
        "distance_m",
        "strength",
        "evidence_refs",
    ]

    def mission_goal_variant(
        goal_type: GoalType,
        spatial_constraint: dict[str, object],
    ) -> dict[str, object]:
        properties = deepcopy(common_goal_properties)
        properties["goal_type"] = {"const": goal_type.value}
        properties["spatial_constraint"] = spatial_constraint
        return _object(properties, goal_required)

    # Keep the semantic distinction in the wire grammar.  In particular, a
    # point is a navigation destination, not a search region: accepting it for
    # SEARCH_TARGET loses the user's "within N metres" extent before Fleet
    # planning even starts.  ``goal_type`` is a const in every branch so the
    # oneOf is discriminated without the less-portable JSON Schema
    # ``discriminator`` extension.
    mission_goal = {
        "oneOf": [
            mission_goal_variant(
                GoalType.SEARCH_TARGET,
                _nullable(_constrain_frames(region_spec_json_schema(), frames)),
            ),
            mission_goal_variant(GoalType.TRACK_TARGET, deepcopy(general_spatial)),
            mission_goal_variant(GoalType.INSPECT_TARGET, deepcopy(general_spatial)),
            mission_goal_variant(
                GoalType.NAVIGATE,
                _constrain_frames(spatial_target_json_schema(), frames),
            ),
        ]
    }
    assignment_constraint = _object(
        {
            "constraint_id": _id(),
            "uav_id": {"type": "string", "enum": list(uav_ids)},
            "goal_ids": _id_array(MAX_GOALS, minimum=1),
            "strength": deepcopy(strength),
            "evidence_refs": _id_array(),
        },
        ["constraint_id", "uav_id", "goal_ids", "strength", "evidence_refs"],
    )
    ordering_constraint = _object(
        {
            "constraint_id": _id(),
            "before_goal_id": _id(),
            "after_goal_id": _id(),
            "strength": deepcopy(strength),
            "evidence_refs": _id_array(),
        },
        [
            "constraint_id",
            "before_goal_id",
            "after_goal_id",
            "strength",
            "evidence_refs",
        ],
    )
    termination_goal = _object(
        {
            "goal_id": _id(),
            "goal_type": {
                "type": "string",
                "enum": [item.value for item in TERMINATION_GOAL_TYPES],
            },
            "uav_id": _nullable({"type": "string", "enum": list(uav_ids)}),
            "duration_s": deepcopy(positive_seconds),
            "strength": deepcopy(strength),
            "evidence_refs": _id_array(),
        },
        [
            "goal_id",
            "goal_type",
            "uav_id",
            "duration_s",
            "strength",
            "evidence_refs",
        ],
    )
    ambiguity = _object(
        {
            "ambiguity_id": _id(),
            "field_path": _bounded_text(128),
            "description": _bounded_text(512),
            "candidate_values": {
                "type": "array",
                "maxItems": 8,
                "items": _bounded_text(128),
            },
            "resolution_required": {"type": "boolean"},
            "evidence_refs": _id_array(),
        },
        [
            "ambiguity_id",
            "field_path",
            "description",
            "candidate_values",
            "resolution_required",
            "evidence_refs",
        ],
    )
    evidence = _object(
        {"evidence_id": _id(), "quote": _bounded_text(512)},
        ["evidence_id", "quote"],
    )
    return _object(
        {
            "schema_version": {"type": "integer", "const": 1},
            "source_text": {"type": "string", "const": source_text},
            "goals": {
                "type": "array",
                "maxItems": MAX_GOALS,
                "items": mission_goal,
            },
            "assignment_constraints": {
                "type": "array",
                "maxItems": MAX_ASSIGNMENT_CONSTRAINTS,
                "items": assignment_constraint,
            },
            "ordering_constraints": {
                "type": "array",
                "maxItems": MAX_ORDERING_CONSTRAINTS,
                "items": ordering_constraint,
            },
            "termination_goals": {
                "type": "array",
                "maxItems": MAX_TERMINATION_GOALS,
                "items": termination_goal,
            },
            "ambiguities": {
                "type": "array",
                "maxItems": MAX_AMBIGUITIES,
                "items": ambiguity,
            },
            "source_evidence": {
                "type": "array",
                "maxItems": MAX_SOURCE_EVIDENCE,
                "items": evidence,
            },
        },
        [
            "schema_version",
            "source_text",
            "goals",
            "assignment_constraints",
            "ordering_constraints",
            "termination_goals",
            "ambiguities",
            "source_evidence",
        ],
    )


__all__ = ["build_fleet_task_spec_json_schema"]
