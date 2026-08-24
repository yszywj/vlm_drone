"""Stable validation codes shared by planning, compilation, and runtime.

The code is deliberately separate from the human-readable message.  Result
recorders and recovery policies must branch on a bounded enum, never on error
prose emitted by a model or an exception.
"""

from __future__ import annotations

from enum import Enum


class ValidationCode(str, Enum):
    """Machine-readable validation outcomes for protocol version 1."""

    # Semantic coverage: these findings may be repaired or reported as a
    # degraded result without authorizing an unsafe action.
    GOAL_NOT_COVERED = "GOAL_NOT_COVERED"
    GOAL_PATH_INFEASIBLE = "GOAL_PATH_INFEASIBLE"
    TRACK_DURATION_UNDERSHOOT = "TRACK_DURATION_UNDERSHOOT"
    WAIT_DURATION_UNDERSHOOT = "WAIT_DURATION_UNDERSHOOT"
    RETURN_HOME_NOT_COVERED = "RETURN_HOME_NOT_COVERED"
    LAND_NOT_COVERED = "LAND_NOT_COVERED"
    ASSIGNMENT_CONSTRAINT_DEVIATION = "ASSIGNMENT_CONSTRAINT_DEVIATION"
    PREFERRED_CONSTRAINT_DEVIATION = "PREFERRED_CONSTRAINT_DEVIATION"
    UNSUPPORTED_GOAL_TYPE = "UNSUPPORTED_GOAL_TYPE"
    AMBIGUOUS_GOAL = "AMBIGUOUS_GOAL"

    # Proposal/action boundary: the affected proposal must never reach a
    # SkillManager or controller.
    INVALID_JSON = "INVALID_JSON"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    UNKNOWN_SKILL = "UNKNOWN_SKILL"
    UNKNOWN_ENTITY = "UNKNOWN_ENTITY"
    NON_FINITE_NUMBER = "NON_FINITE_NUMBER"
    ROUTING_MISMATCH = "ROUTING_MISMATCH"
    PLAN_VERSION_MISMATCH = "PLAN_VERSION_MISMATCH"
    STEP_REFERENCE_INVALID = "STEP_REFERENCE_INVALID"
    CALL_LIMIT_EXCEEDED = "CALL_LIMIT_EXCEEDED"
    LOW_LEVEL_CONTROL_FORBIDDEN = "LOW_LEVEL_CONTROL_FORBIDDEN"
    ORACLE_FIELD_FORBIDDEN = "ORACLE_FIELD_FORBIDDEN"
    OUT_OF_BOUNDS_GOTO = "OUT_OF_BOUNDS_GOTO"
    INVALID_LANDING_ZONE = "INVALID_LANDING_ZONE"
    UNSAFE_ACTION = "UNSAFE_ACTION"

    # Live safety boundary.  These represent loss of the assumptions under
    # which an already-running assignment was admitted.
    COLLISION_IMMINENT = "COLLISION_IMMINENT"
    GEOFENCE_BREACH = "GEOFENCE_BREACH"
    ALTITUDE_LIMIT_BREACH = "ALTITUDE_LIMIT_BREACH"
    MISSION_TIMEOUT = "MISSION_TIMEOUT"
    WORLD_STATE_INVALID = "WORLD_STATE_INVALID"
    AIRSPACE_SYSTEM_FAILURE = "AIRSPACE_SYSTEM_FAILURE"

    INTERNAL_ERROR = "INTERNAL_ERROR"


__all__ = ["ValidationCode"]
