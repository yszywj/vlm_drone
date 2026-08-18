"""Canonical validation and generation for cross-module routing identifiers.

Every boundary type imports this module instead of maintaining a local ID
regular expression.  IDs are deliberately opaque: their only contract is a
small, log-safe ASCII representation.
"""

from __future__ import annotations

import re
from uuid import uuid4


ROUTING_ID_PATTERN_TEXT = r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$"
ROUTING_ID_PATTERN = re.compile(ROUTING_ID_PATTERN_TEXT)


def validate_routing_id(value: object, field_name: str = "routing_id") -> str:
    """Return *value* unchanged when it is a valid routing identifier.

    Whitespace is not silently stripped because routing comparisons must be
    byte-for-byte stable across planner, runtime, model worker, and logs.
    """

    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if ROUTING_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must match {ROUTING_ID_PATTERN_TEXT}"
        )
    return value


def validate_uav_id(value: object) -> str:
    return validate_routing_id(value, "uav_id")


def validate_mission_id(value: object) -> str:
    return validate_routing_id(value, "mission_id")


def validate_plan_id(value: object) -> str:
    return validate_routing_id(value, "plan_id")


def validate_review_id(value: object) -> str:
    return validate_routing_id(value, "review_id")


def validate_request_id(value: object) -> str:
    return validate_routing_id(value, "request_id")


def validate_invocation_id(value: object) -> str:
    return validate_routing_id(value, "invocation_id")


def generate_routing_id(prefix: str) -> str:
    """Generate a trusted opaque ID with a validated human-readable prefix."""

    normalized_prefix = validate_routing_id(prefix, "prefix")
    # A UUID hex suffix is stable, collision-resistant, and still leaves room
    # for prefixes used by this project (mission, plan, request, ...).
    max_prefix_length = 64 - 1 - 32
    if len(normalized_prefix) > max_prefix_length:
        raise ValueError(
            f"prefix must contain at most {max_prefix_length} characters"
        )
    return validate_routing_id(f"{normalized_prefix}_{uuid4().hex}")


__all__ = [
    "ROUTING_ID_PATTERN",
    "ROUTING_ID_PATTERN_TEXT",
    "generate_routing_id",
    "validate_invocation_id",
    "validate_mission_id",
    "validate_plan_id",
    "validate_request_id",
    "validate_review_id",
    "validate_routing_id",
    "validate_uav_id",
]
