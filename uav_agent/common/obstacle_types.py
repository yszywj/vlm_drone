"""Pure-Python obstacle geometry and camera-constrained observation types.

The types in this module deliberately have no Isaac Sim dependency.  Scene
construction, ideal-camera perception, collision supervision, visualisation,
and route critics can therefore share one immutable geometry contract without
crossing the simulator import boundary.

Camera-relative vectors use ``CAMERA_FLU`` throughout: +X forward, +Y left,
+Z up.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Iterable, Mapping, Sequence

from common.ids import validate_routing_id, validate_uav_id


IDEAL_CAMERA_OBSTACLE_SOURCE = "ideal_camera_obstacle_perception"
CAMERA_COORDINATE_FRAME = "CAMERA_FLU"


class ObstacleMotionState(str, Enum):
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    UNKNOWN = "UNKNOWN"


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _vector(
    value: object,
    field_name: str,
    *,
    length: int,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must contain exactly {length} numbers")
    materialized = tuple(value)
    if len(materialized) != length:
        raise ValueError(f"{field_name} must contain exactly {length} numbers")
    return tuple(
        _finite(item, f"{field_name}[{index}]")
        for index, item in enumerate(materialized)
    )


def _positive_vector3(value: object, field_name: str) -> tuple[float, float, float]:
    result = _vector(value, field_name, length=3)
    if any(item <= 0.0 for item in result):
        raise ValueError(f"{field_name} values must be greater than zero")
    return result  # type: ignore[return-value]


def _motion_state(value: object) -> ObstacleMotionState:
    if isinstance(value, ObstacleMotionState):
        return value
    if isinstance(value, str):
        try:
            return ObstacleMotionState(value)
        except ValueError:
            pass
    raise ValueError("motion_state must be STATIC, DYNAMIC, or UNKNOWN")


@dataclass(frozen=True, slots=True)
class ObstacleAABB:
    """Closed world-space axis-aligned bounding box in metres."""

    min_xyz_m: tuple[float, float, float]
    max_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = _vector(self.min_xyz_m, "min_xyz_m", length=3)
        maximum = _vector(self.max_xyz_m, "max_xyz_m", length=3)
        if any(low >= high for low, high in zip(minimum, maximum)):
            raise ValueError("min_xyz_m must be strictly less than max_xyz_m")
        object.__setattr__(self, "min_xyz_m", minimum)
        object.__setattr__(self, "max_xyz_m", maximum)

    @classmethod
    def from_center_size(
        cls,
        center_xyz_m: Sequence[float],
        size_xyz_m: Sequence[float],
    ) -> "ObstacleAABB":
        center = _vector(center_xyz_m, "center_xyz_m", length=3)
        size = _positive_vector3(size_xyz_m, "size_xyz_m")
        half = tuple(item / 2.0 for item in size)
        return cls(
            min_xyz_m=tuple(value - extent for value, extent in zip(center, half)),
            max_xyz_m=tuple(value + extent for value, extent in zip(center, half)),
        )

    @property
    def center_xyz_m(self) -> tuple[float, float, float]:
        return tuple(
            (low + high) / 2.0
            for low, high in zip(self.min_xyz_m, self.max_xyz_m)
        )

    @property
    def size_xyz_m(self) -> tuple[float, float, float]:
        return tuple(
            high - low for low, high in zip(self.min_xyz_m, self.max_xyz_m)
        )

    def expanded(self, half_extent_xyz_m: Sequence[float]) -> "ObstacleAABB":
        extent = _vector(half_extent_xyz_m, "half_extent_xyz_m", length=3)
        if any(item < 0.0 for item in extent):
            raise ValueError("half_extent_xyz_m values must be non-negative")
        return ObstacleAABB(
            min_xyz_m=tuple(low - item for low, item in zip(self.min_xyz_m, extent)),
            max_xyz_m=tuple(high + item for high, item in zip(self.max_xyz_m, extent)),
        )

    def contains(self, point_xyz_m: Sequence[float], *, strict: bool = False) -> bool:
        point = _vector(point_xyz_m, "point_xyz_m", length=3)
        if strict:
            return all(
                low < value < high
                for low, value, high in zip(self.min_xyz_m, point, self.max_xyz_m)
            )
        return all(
            low <= value <= high
            for low, value, high in zip(self.min_xyz_m, point, self.max_xyz_m)
        )

    def intersects(self, other: "ObstacleAABB") -> bool:
        if not isinstance(other, ObstacleAABB):
            raise TypeError("other must be an ObstacleAABB")
        return all(
            first_low <= second_high and second_low <= first_high
            for first_low, first_high, second_low, second_high in zip(
                self.min_xyz_m,
                self.max_xyz_m,
                other.min_xyz_m,
                other.max_xyz_m,
            )
        )

    def segment_intersection_fraction(
        self,
        start_xyz_m: Sequence[float],
        end_xyz_m: Sequence[float],
    ) -> float | None:
        """Return the first closed-segment intersection fraction, if any."""

        start = _vector(start_xyz_m, "start_xyz_m", length=3)
        end = _vector(end_xyz_m, "end_xyz_m", length=3)
        direction = tuple(b - a for a, b in zip(start, end))
        entry = 0.0
        exit_ = 1.0
        for axis in range(3):
            delta = direction[axis]
            low = self.min_xyz_m[axis]
            high = self.max_xyz_m[axis]
            if abs(delta) <= 1e-15:
                if start[axis] < low or start[axis] > high:
                    return None
                continue
            first = (low - start[axis]) / delta
            second = (high - start[axis]) / delta
            axis_entry = min(first, second)
            axis_exit = max(first, second)
            entry = max(entry, axis_entry)
            exit_ = min(exit_, axis_exit)
            if entry > exit_:
                return None
        if exit_ < 0.0 or entry > 1.0:
            return None
        return max(0.0, entry)

    def corners_xyz_m(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            (x, y, z)
            for x in (self.min_xyz_m[0], self.max_xyz_m[0])
            for y in (self.min_xyz_m[1], self.max_xyz_m[1])
            for z in (self.min_xyz_m[2], self.max_xyz_m[2])
        )


@dataclass(frozen=True, slots=True)
class ObstacleSpec:
    """Immutable scene obstacle specification shared by every subsystem."""

    obstacle_id: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    color_rgb: tuple[float, float, float]
    collidable: bool = True
    motion_state: ObstacleMotionState = ObstacleMotionState.STATIC

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obstacle_id",
            validate_routing_id(self.obstacle_id, "obstacle_id"),
        )
        object.__setattr__(
            self,
            "center_xyz_m",
            _vector(self.center_xyz_m, "center_xyz_m", length=3),
        )
        object.__setattr__(
            self,
            "size_xyz_m",
            _positive_vector3(self.size_xyz_m, "size_xyz_m"),
        )
        color = _vector(self.color_rgb, "color_rgb", length=3)
        if any(item < 0.0 or item > 1.0 for item in color):
            raise ValueError("color_rgb values must be between zero and one")
        object.__setattr__(self, "color_rgb", color)
        if not isinstance(self.collidable, bool):
            raise TypeError("collidable must be a bool")
        object.__setattr__(self, "motion_state", _motion_state(self.motion_state))

    @property
    def aabb(self) -> ObstacleAABB:
        return ObstacleAABB.from_center_size(self.center_xyz_m, self.size_xyz_m)


@dataclass(frozen=True, slots=True)
class CameraGeometry:
    """One rendered camera pose and pinhole model, independent of Isaac."""

    frame_id: str
    uav_id: str
    timestamp_s: float
    position_world_m: tuple[float, float, float]
    orientation_world_from_camera_wxyz: tuple[float, float, float, float]
    resolution_wh_px: tuple[int, int]
    horizontal_fov_deg: float
    near_clip_m: float
    far_clip_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", validate_routing_id(self.frame_id, "frame_id"))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        timestamp = _finite(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(
            self,
            "position_world_m",
            _vector(self.position_world_m, "position_world_m", length=3),
        )
        quaternion = _vector(
            self.orientation_world_from_camera_wxyz,
            "orientation_world_from_camera_wxyz",
            length=4,
        )
        norm = sum(item * item for item in quaternion) ** 0.5
        if norm <= 1e-12:
            raise ValueError("orientation_world_from_camera_wxyz must have non-zero norm")
        object.__setattr__(
            self,
            "orientation_world_from_camera_wxyz",
            tuple(item / norm for item in quaternion),
        )
        resolution = self.resolution_wh_px
        if (
            isinstance(resolution, (str, bytes))
            or not isinstance(resolution, Sequence)
            or len(resolution) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in resolution)
        ):
            raise ValueError("resolution_wh_px must contain two positive integers")
        object.__setattr__(self, "resolution_wh_px", (int(resolution[0]), int(resolution[1])))
        fov = _finite(self.horizontal_fov_deg, "horizontal_fov_deg")
        if not 0.0 < fov < 180.0:
            raise ValueError("horizontal_fov_deg must be between zero and 180")
        object.__setattr__(self, "horizontal_fov_deg", fov)
        near = _finite(self.near_clip_m, "near_clip_m")
        far = _finite(self.far_clip_m, "far_clip_m")
        if near <= 0.0 or far <= near:
            raise ValueError("clipping range must satisfy 0 < near_clip_m < far_clip_m")
        object.__setattr__(self, "near_clip_m", near)
        object.__setattr__(self, "far_clip_m", far)


@dataclass(frozen=True, slots=True)
class FlightCorridor:
    """Finite active flight segment swept by a spherical safety radius."""

    start_world_m: tuple[float, float, float]
    end_world_m: tuple[float, float, float]
    radius_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_world_m",
            _vector(self.start_world_m, "start_world_m", length=3),
        )
        object.__setattr__(
            self,
            "end_world_m",
            _vector(self.end_world_m, "end_world_m", length=3),
        )
        radius = _finite(self.radius_m, "radius_m")
        if radius < 0.0:
            raise ValueError("radius_m must be non-negative")
        object.__setattr__(self, "radius_m", radius)

    def intersects(self, obstacle: ObstacleAABB) -> bool:
        if not isinstance(obstacle, ObstacleAABB):
            raise TypeError("obstacle must be an ObstacleAABB")
        expanded = obstacle.expanded((self.radius_m,) * 3)
        return expanded.segment_intersection_fraction(
            self.start_world_m,
            self.end_world_m,
        ) is not None


@dataclass(frozen=True, slots=True)
class VisibleObstacle:
    """One obstacle observation containing only camera-relative geometry."""

    obstacle_id: str
    bbox_xyxy_normalized: tuple[float, float, float, float]
    relative_center_m: tuple[float, float, float]
    relative_size_m: tuple[float, float, float]
    depth_m: float
    occlusion_ratio: float
    motion_state: ObstacleMotionState
    active_corridor_intersection: bool
    time_to_collision_s: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obstacle_id",
            validate_routing_id(self.obstacle_id, "obstacle_id"),
        )
        bbox = _vector(self.bbox_xyxy_normalized, "bbox_xyxy_normalized", length=4)
        if any(item < 0.0 or item > 1.0 for item in bbox):
            raise ValueError("bbox_xyxy_normalized values must be between zero and one")
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("bbox_xyxy_normalized must have positive area")
        object.__setattr__(self, "bbox_xyxy_normalized", bbox)
        object.__setattr__(
            self,
            "relative_center_m",
            _vector(self.relative_center_m, "relative_center_m", length=3),
        )
        object.__setattr__(
            self,
            "relative_size_m",
            _positive_vector3(self.relative_size_m, "relative_size_m"),
        )
        depth = _finite(self.depth_m, "depth_m")
        if depth <= 0.0:
            raise ValueError("depth_m must be greater than zero")
        object.__setattr__(self, "depth_m", depth)
        occlusion = _finite(self.occlusion_ratio, "occlusion_ratio")
        if not 0.0 <= occlusion <= 1.0:
            raise ValueError("occlusion_ratio must be between zero and one")
        object.__setattr__(self, "occlusion_ratio", occlusion)
        object.__setattr__(self, "motion_state", _motion_state(self.motion_state))
        if not isinstance(self.active_corridor_intersection, bool):
            raise TypeError("active_corridor_intersection must be a bool")
        if self.time_to_collision_s is not None:
            ttc = _finite(self.time_to_collision_s, "time_to_collision_s")
            if ttc < 0.0:
                raise ValueError("time_to_collision_s must be non-negative or None")
            object.__setattr__(self, "time_to_collision_s", ttc)

    def to_dict(self) -> dict[str, object]:
        return {
            "obstacle_id": self.obstacle_id,
            "bbox_xyxy_normalized": list(self.bbox_xyxy_normalized),
            "relative_center_m": list(self.relative_center_m),
            "relative_size_m": list(self.relative_size_m),
            "depth_m": self.depth_m,
            "occlusion_ratio": self.occlusion_ratio,
            "motion_state": self.motion_state.value,
            "active_corridor_intersection": self.active_corridor_intersection,
            "time_to_collision_s": self.time_to_collision_s,
        }


@dataclass(frozen=True, slots=True)
class ObstacleObservation:
    """Bounded result emitted once for each accepted fresh camera frame."""

    observation_id: str
    frame_id: str
    uav_id: str
    timestamp_s: float
    visible_obstacles: tuple[VisibleObstacle, ...]
    source: str = IDEAL_CAMERA_OBSTACLE_SOURCE
    privileged: bool = True
    coordinate_frame: str = CAMERA_COORDINATE_FRAME

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            validate_routing_id(self.observation_id, "observation_id"),
        )
        object.__setattr__(self, "frame_id", validate_routing_id(self.frame_id, "frame_id"))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        timestamp = _finite(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        object.__setattr__(self, "timestamp_s", timestamp)
        obstacles = tuple(self.visible_obstacles)
        if any(not isinstance(item, VisibleObstacle) for item in obstacles):
            raise TypeError("visible_obstacles must contain VisibleObstacle values")
        ids = tuple(item.obstacle_id for item in obstacles)
        if len(ids) != len(set(ids)):
            raise ValueError("visible_obstacles must not contain duplicate obstacle_id values")
        object.__setattr__(self, "visible_obstacles", obstacles)
        if self.source != IDEAL_CAMERA_OBSTACLE_SOURCE:
            raise ValueError(
                f"source must be {IDEAL_CAMERA_OBSTACLE_SOURCE!r} for ideal observations"
            )
        if self.privileged is not True:
            raise ValueError("ideal camera obstacle observations must be marked privileged")
        if self.coordinate_frame != CAMERA_COORDINATE_FRAME:
            raise ValueError(f"coordinate_frame must be {CAMERA_COORDINATE_FRAME}")

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "frame_id": self.frame_id,
            "uav_id": self.uav_id,
            "timestamp_s": self.timestamp_s,
            "source": self.source,
            "privileged": self.privileged,
            "coordinate_frame": self.coordinate_frame,
            "visible_obstacles": [item.to_dict() for item in self.visible_obstacles],
        }

    def manifest_fields(self) -> Mapping[str, object]:
        return {
            "obstacle_perception_source": self.source,
            "obstacle_perception_privileged": self.privileged,
            "obstacle_coordinate_frame": self.coordinate_frame,
        }


# Short public alias matching the checklist terminology.
MotionState = ObstacleMotionState


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
