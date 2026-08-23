"""Transport schema for the experimental MissionProgram envelope."""

from __future__ import annotations


def build_mission_program_json_schema() -> dict[str, object]:
    """Return a strict structural schema; TaskStep semantics remain trusted."""

    routing_id = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_.-]{0,63}$"}
    event = {
        "type": "string",
        "enum": [
            "SUCCESS",
            "FAILURE",
            "TARGET_CONFIRMED",
            "TARGET_LOST",
            "PATH_BLOCKED",
            "TIMEOUT",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "mission_id",
            "uav_id",
            "plan_version",
            "entry_node_id",
            "spatial_entities",
            "nodes",
            "edges",
            "event_handlers",
        ],
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "mission_id": dict(routing_id),
            "uav_id": dict(routing_id),
            "plan_version": {"type": "integer", "minimum": 1},
            "entry_node_id": dict(routing_id),
            "spatial_entities": {"type": "array", "maxItems": 128},
            "nodes": {"type": "array", "minItems": 1, "maxItems": 100},
            "edges": {
                "type": "array",
                "maxItems": 256,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source_node_id", "target_node_id", "on"],
                    "properties": {
                        "source_node_id": dict(routing_id),
                        "target_node_id": dict(routing_id),
                        "on": event,
                    },
                },
            },
            "event_handlers": {"type": "array", "maxItems": 6},
        },
    }


__all__ = ["build_mission_program_json_schema"]
