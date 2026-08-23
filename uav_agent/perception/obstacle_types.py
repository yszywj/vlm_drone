"""Public perception namespace for shared obstacle geometry contracts.

The canonical definitions live in :mod:`common.obstacle_types` so low-level
configuration and scene code can import them without executing the eager
``perception`` package initialiser.  Re-exporting preserves the documented
``perception.obstacle_types`` API and class identity.
"""

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
]
