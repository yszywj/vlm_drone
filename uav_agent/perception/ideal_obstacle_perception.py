"""Camera-constrained privileged obstacle perception in pure Python.

The backend may use configured ground-truth geometry internally, but it emits
only obstacles that pass clipping, frustum, projected-area, distance, and
occlusion checks for one fresh camera frame.  It never imports Isaac Sim and
never serialises world-space obstacle centres.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan
from numbers import Real
from typing import Protocol, Sequence

import numpy as np

from common.ids import generate_routing_id, validate_uav_id
from env.obstacle_registry import ObstacleRegistry
from common.obstacle_types import (
    CameraGeometry,
    FlightCorridor,
    IDEAL_CAMERA_OBSTACLE_SOURCE,
    ObstacleAABB,
    ObstacleObservation,
    ObstacleSpec,
    VisibleObstacle,
)


class IdealObstaclePerceptionConfigLike(Protocol):
    max_distance_m: float
    min_bbox_area_px: int
    max_occlusion_ratio: float


@dataclass(frozen=True, slots=True)
class _Projection:
    spec: ObstacleSpec
    bbox_xyxy_px: tuple[float, float, float, float]
    relative_center_m: tuple[float, float, float]
    depth_m: float
    area_px: float


class IdealObstaclePerception:
    """Filter a shared registry through a fresh pinhole-camera geometry."""

    source = IDEAL_CAMERA_OBSTACLE_SOURCE
    privileged = True

    def __init__(
        self,
        registry: ObstacleRegistry,
        *,
        max_distance_m: float = 40.0,
        min_bbox_area_px: int = 64,
        max_occlusion_ratio: float = 0.95,
    ) -> None:
        if not isinstance(registry, ObstacleRegistry):
            raise TypeError("registry must be an ObstacleRegistry")
        self.registry = registry
        self.max_distance_m = _positive(max_distance_m, "max_distance_m")
        if (
            isinstance(min_bbox_area_px, bool)
            or not isinstance(min_bbox_area_px, int)
            or min_bbox_area_px <= 0
        ):
            raise ValueError("min_bbox_area_px must be a positive integer")
        self.min_bbox_area_px = min_bbox_area_px
        self.max_occlusion_ratio = _ratio(
            max_occlusion_ratio,
            "max_occlusion_ratio",
            upper_inclusive=False,
        )
        self._last_frame_by_uav: dict[str, tuple[str, float]] = {}

    @classmethod
    def from_config(
        cls,
        registry: ObstacleRegistry,
        config: IdealObstaclePerceptionConfigLike,
    ) -> "IdealObstaclePerception":
        return cls(
            registry,
            max_distance_m=config.max_distance_m,
            min_bbox_area_px=config.min_bbox_area_px,
            max_occlusion_ratio=config.max_occlusion_ratio,
        )

    def observe(
        self,
        camera: CameraGeometry,
        *,
        active_corridor: FlightCorridor | None = None,
        uav_velocity_world_mps: Sequence[float] = (0.0, 0.0, 0.0),
        observation_id: str | None = None,
    ) -> ObstacleObservation | None:
        """Return one result for a new frame, or ``None`` for a stale duplicate.

        A caller invokes this method at Camera frame frequency.  Repeated or
        time-regressing frames are deliberately suppressed so a fast physics
        loop cannot manufacture extra privileged observations between renders.
        """

        if not isinstance(camera, CameraGeometry):
            raise TypeError("camera must be a CameraGeometry")
        if active_corridor is not None and not isinstance(active_corridor, FlightCorridor):
            raise TypeError("active_corridor must be a FlightCorridor or None")
        velocity = _vector3(uav_velocity_world_mps, "uav_velocity_world_mps")
        prior = self._last_frame_by_uav.get(camera.uav_id)
        if prior is not None and (
            camera.frame_id == prior[0] or camera.timestamp_s <= prior[1]
        ):
            return None

        rotation_world_from_camera = _quaternion_rotation_matrix(
            camera.orientation_world_from_camera_wxyz
        )
        camera_position = np.asarray(camera.position_world_m, dtype=np.float64)
        projections: list[_Projection] = []
        for spec in self.registry:
            projection = _project_obstacle(
                spec,
                camera=camera,
                camera_position=camera_position,
                rotation_world_from_camera=rotation_world_from_camera,
                max_distance_m=self.max_distance_m,
            )
            if projection is not None and projection.area_px >= self.min_bbox_area_px:
                projections.append(projection)

        projections.sort(key=lambda item: (item.depth_m, item.spec.obstacle_id))
        accepted: list[VisibleObstacle] = []
        nearer: list[_Projection] = []
        width, height = camera.resolution_wh_px
        for projection in projections:
            occlusion = min(
                1.0,
                sum(
                    _intersection_area(
                        projection.bbox_xyxy_px,
                        foreground.bbox_xyxy_px,
                    )
                    / projection.area_px
                    for foreground in nearer
                    if foreground.depth_m < projection.depth_m
                ),
            )
            nearer.append(projection)
            if occlusion > self.max_occlusion_ratio:
                continue
            if projection.area_px * (1.0 - occlusion) < self.min_bbox_area_px:
                continue

            spec = projection.spec
            aabb = spec.aabb
            corridor_intersection = (
                active_corridor.intersects(aabb)
                if active_corridor is not None
                else False
            )
            ttc = (
                _time_to_collision(
                    active_corridor.start_world_m,
                    velocity,
                    aabb.expanded((active_corridor.radius_m,) * 3),
                    active_corridor.end_world_m,
                )
                if active_corridor is not None and corridor_intersection
                else None
            )
            x1, y1, x2, y2 = projection.bbox_xyxy_px
            accepted.append(
                VisibleObstacle(
                    obstacle_id=spec.obstacle_id,
                    bbox_xyxy_normalized=(
                        x1 / width,
                        y1 / height,
                        x2 / width,
                        y2 / height,
                    ),
                    relative_center_m=projection.relative_center_m,
                    relative_size_m=spec.size_xyz_m,
                    depth_m=projection.depth_m,
                    occlusion_ratio=occlusion,
                    motion_state=spec.motion_state,
                    active_corridor_intersection=corridor_intersection,
                    time_to_collision_s=ttc,
                )
            )

        self._last_frame_by_uav[camera.uav_id] = (
            camera.frame_id,
            camera.timestamp_s,
        )
        return ObstacleObservation(
            observation_id=(
                generate_routing_id("observation")
                if observation_id is None
                else observation_id
            ),
            frame_id=camera.frame_id,
            uav_id=camera.uav_id,
            timestamp_s=camera.timestamp_s,
            visible_obstacles=tuple(accepted),
        )

    def reset(self, *, uav_id: str | None = None) -> None:
        """Forget freshness state for one UAV or for a new global episode."""

        if uav_id is None:
            self._last_frame_by_uav.clear()
        else:
            self._last_frame_by_uav.pop(validate_uav_id(uav_id), None)

    def manifest_fields(self) -> dict[str, object]:
        return {
            "obstacle_perception_source": self.source,
            "obstacle_perception_privileged": self.privileged,
            "obstacle_perception_max_distance_m": self.max_distance_m,
            "obstacle_perception_min_bbox_area_px": self.min_bbox_area_px,
            "obstacle_perception_max_occlusion_ratio": self.max_occlusion_ratio,
        }


def _project_obstacle(
    spec: ObstacleSpec,
    *,
    camera: CameraGeometry,
    camera_position: np.ndarray,
    rotation_world_from_camera: np.ndarray,
    max_distance_m: float,
) -> _Projection | None:
    aabb = spec.aabb
    minimum = np.asarray(aabb.min_xyz_m)
    maximum = np.asarray(aabb.max_xyz_m)
    closest = np.minimum(np.maximum(camera_position, minimum), maximum)
    if float(np.linalg.norm(closest - camera_position)) > max_distance_m:
        return None

    corners_world = np.asarray(aabb.corners_xyz_m(), dtype=np.float64)
    corners_camera = (rotation_world_from_camera.T @ (corners_world - camera_position).T).T
    center_camera = rotation_world_from_camera.T @ (
        np.asarray(spec.center_xyz_m, dtype=np.float64) - camera_position
    )
    forward = corners_camera[:, 0]
    if float(np.max(forward)) < camera.near_clip_m:
        return None
    if float(np.min(forward)) > camera.far_clip_m:
        return None

    # Clamp near-plane-crossing corners before projection.  Screen clipping
    # below then retains partially visible boxes whose corners all lie outside
    # the final image rectangle.
    projection_points = corners_camera[forward > 0.0].copy()
    if projection_points.size == 0:
        return None
    projection_points[:, 0] = np.clip(
        projection_points[:, 0], camera.near_clip_m, camera.far_clip_m
    )
    width, height = camera.resolution_wh_px
    focal_px = width / (2.0 * tan(radians(camera.horizontal_fov_deg) / 2.0))
    u = width / 2.0 - focal_px * projection_points[:, 1] / projection_points[:, 0]
    v = height / 2.0 - focal_px * projection_points[:, 2] / projection_points[:, 0]
    raw_x1 = float(np.min(u))
    raw_y1 = float(np.min(v))
    raw_x2 = float(np.max(u))
    raw_y2 = float(np.max(v))
    x1 = max(0.0, min(float(width), raw_x1))
    y1 = max(0.0, min(float(height), raw_y1))
    x2 = max(0.0, min(float(width), raw_x2))
    y2 = max(0.0, min(float(height), raw_y2))
    if x1 >= x2 or y1 >= y2:
        return None
    depth = float(center_camera[0])
    if depth <= 0.0:
        positive_depths = forward[forward > 0.0]
        if positive_depths.size == 0:
            return None
        depth = float(np.min(positive_depths))
    return _Projection(
        spec=spec,
        bbox_xyxy_px=(x1, y1, x2, y2),
        relative_center_m=tuple(float(item) for item in center_camera),
        depth_m=depth,
        area_px=(x2 - x1) * (y2 - y1),
    )


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _time_to_collision(
    start_xyz_m: Sequence[float],
    velocity_xyz_mps: np.ndarray,
    obstacle: ObstacleAABB,
    corridor_end_xyz_m: Sequence[float],
) -> float | None:
    speed = float(np.linalg.norm(velocity_xyz_mps))
    if speed <= 1e-12:
        return None
    start = np.asarray(start_xyz_m, dtype=np.float64)
    end = np.asarray(corridor_end_xyz_m, dtype=np.float64)
    duration = float(np.linalg.norm(end - start)) / speed
    entry = 0.0
    exit_ = duration
    for axis in range(3):
        velocity = velocity_xyz_mps[axis]
        if abs(velocity) <= 1e-15:
            if start[axis] < obstacle.min_xyz_m[axis] or start[axis] > obstacle.max_xyz_m[axis]:
                return None
            continue
        first = (obstacle.min_xyz_m[axis] - start[axis]) / velocity
        second = (obstacle.max_xyz_m[axis] - start[axis]) / velocity
        entry = max(entry, min(first, second))
        exit_ = min(exit_, max(first, second))
        if entry > exit_:
            return None
    if exit_ < 0.0 or entry > duration:
        return None
    return max(0.0, entry)


def _quaternion_rotation_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return result


def _ratio(value: object, name: str, *, upper_inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    valid = 0.0 <= result <= 1.0 if upper_inclusive else 0.0 <= result < 1.0
    if not np.isfinite(result) or not valid:
        bracket = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{name} must be in {bracket}")
    return result


__all__ = ["IdealObstaclePerception", "IdealObstaclePerceptionConfigLike"]
