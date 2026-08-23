"""Shared, dependency-free runtime boundary helpers."""

from common.ids import (
    ROUTING_ID_PATTERN,
    ROUTING_ID_PATTERN_TEXT,
    generate_routing_id,
    validate_invocation_id,
    validate_mission_id,
    validate_plan_id,
    validate_request_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from common.provenance import is_privileged_oracle_source
from common.obstacle_types import (
    CAMERA_COORDINATE_FRAME,
    CameraGeometry,
    FlightCorridor,
    IDEAL_CAMERA_OBSTACLE_SOURCE,
    MotionState,
    ObstacleAABB,
    ObstacleMotionState,
    ObstacleObservation,
    ObstacleSpec,
    VisibleObstacle,
)

__all__ = [
    "ROUTING_ID_PATTERN",
    "ROUTING_ID_PATTERN_TEXT",
    "CAMERA_COORDINATE_FRAME",
    "CameraGeometry",
    "FlightCorridor",
    "IDEAL_CAMERA_OBSTACLE_SOURCE",
    "MotionState",
    "ObstacleAABB",
    "ObstacleMotionState",
    "ObstacleObservation",
    "ObstacleSpec",
    "VisibleObstacle",
    "generate_routing_id",
    "is_privileged_oracle_source",
    "validate_invocation_id",
    "validate_mission_id",
    "validate_plan_id",
    "validate_request_id",
    "validate_review_id",
    "validate_routing_id",
    "validate_uav_id",
]
