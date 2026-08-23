"""Strict structured-output schema for versioned MissionProgram patches."""

from __future__ import annotations

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from planner.mission_program import ProgramEvent
from skills.types import SkillName


def build_program_patch_json_schema(
    *,
    mission_id: str,
    uav_id: str,
    base_plan_version: int,
    replace_from_node_id: str,
    trigger_event: ProgramEvent | str,
    max_replacement_nodes: int = 16,
) -> dict[str, object]:
    """Return a portable vLLM schema; Python enforces graph semantics.

    JSON Schema cannot express equality between ``node_id`` and ``step.id``
    or completed-prefix immutability.  Those invariants are intentionally
    checked again by :class:`ProgramPatch` and ``apply_program_patch``.
    """

    mission = validate_mission_id(mission_id)
    routed_uav = validate_uav_id(uav_id)
    current = validate_routing_id(
        replace_from_node_id, "replace_from_node_id"
    )
    if (
        isinstance(base_plan_version, bool)
        or not isinstance(base_plan_version, int)
        or base_plan_version <= 0
    ):
        raise ValueError("base_plan_version must be a positive integer")
    try:
        event = (
            trigger_event
            if isinstance(trigger_event, ProgramEvent)
            else ProgramEvent(trigger_event)
        )
    except (TypeError, ValueError):
        raise ValueError("trigger_event is unsupported") from None
    if (
        isinstance(max_replacement_nodes, bool)
        or not isinstance(max_replacement_nodes, int)
        or not 1 <= max_replacement_nodes <= 32
    ):
        raise ValueError("max_replacement_nodes must be within [1, 32]")

    node_id = {
        "type": "string",
        "pattern": "^[a-z][a-z0-9_]{0,31}$",
    }
    reason = {
        "type": "string",
        "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
    }
    recovery = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "skill",
            "max_attempts",
            "search_radius_m",
            "timeout_s",
        ],
        "properties": {
            "skill": {"type": "string", "const": "REACQUIRE"},
            "max_attempts": {"type": "integer", "minimum": 0, "maximum": 2},
            "search_radius_m": {"type": "number", "minimum": 3, "maximum": 20},
            "timeout_s": {"type": "number", "minimum": 5, "maximum": 60},
        },
    }
    step = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "skill", "args"],
        "properties": {
            "id": dict(node_id),
            "skill": {
                "type": "string",
                "enum": [
                    skill.value
                    for skill in SkillName
                    if skill is not SkillName.REACQUIRE
                ],
            },
            # Goal-specific types and bounds are preflighted by SkillManager
            # before a patch may be staged.  Keep this grammar general enough
            # for every registered Skill while bounding its transport size.
            "args": {
                "type": "object",
                "maxProperties": 32,
            },
            "recovery": recovery,
        },
    }
    edge = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_node_id", "target_node_id", "on"],
        "properties": {
            "source_node_id": dict(node_id),
            "target_node_id": dict(node_id),
            "on": {
                "type": "string",
                "enum": [candidate.value for candidate in ProgramEvent],
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "mission_id",
            "uav_id",
            "base_plan_version",
            "new_plan_version",
            "replace_from_node_id",
            "replacement_nodes",
            "replacement_edges",
            "reason_codes",
        ],
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "mission_id": {"type": "string", "const": mission},
            "uav_id": {"type": "string", "const": routed_uav},
            "base_plan_version": {
                "type": "integer",
                "const": base_plan_version,
            },
            "new_plan_version": {
                "type": "integer",
                "const": base_plan_version + 1,
            },
            "replace_from_node_id": {"type": "string", "const": current},
            "replacement_nodes": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_replacement_nodes,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["node_id", "step"],
                    "properties": {
                        "node_id": dict(node_id),
                        "step": step,
                    },
                },
            },
            "replacement_edges": {
                "type": "array",
                "maxItems": 64,
                "items": edge,
            },
            "reason_codes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": reason,
            },
        },
    }


__all__ = ["build_program_patch_json_schema"]
