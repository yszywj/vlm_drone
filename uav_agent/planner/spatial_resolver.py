"""Trusted coordinate-frame and landmark resolution for Spatial Contract V3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import cos, isfinite, radians, sin
from numbers import Real
from types import MappingProxyType

from planner.spatial import (
    CircleRegion,
    CoordinateFrame,
    CorridorRegion,
    NamedLocationTarget,
    PointTarget,
    PolygonRegion,
    RectangleRegion,
    RegionSpec,
    RelationalPointTarget,
    RelationalRegion,
    RouteTarget,
    SectorRegion,
    SpatialContractError,
    SpatialRelation,
    SpatialTarget,
)


class SpatialResolutionError(ValueError):
    """Base error for geometry that cannot be safely resolved."""


class MissingFramePoseError(SpatialResolutionError):
    """Raised when a relative frame has no trusted runtime pose."""


class UnresolvedSpatialReferenceError(SpatialResolutionError):
    """Raised when a named location or visually grounded landmark is absent."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _xyz(value: object, name: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ValueError(f"{name} must contain exactly three numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class FramePose:
    """Trusted planar FLU pose expressed in WORLD_ENU.

    Search and route geometry is currently yaw-only.  A future camera adapter
    may provide a full SE(3) transform without changing the model contract.
    """

    xyz_m: tuple[float, float, float]
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz_m", _xyz(self.xyz_m, "FramePose.xyz_m"))
        object.__setattr__(self, "yaw_rad", _finite(self.yaw_rad, "FramePose.yaw_rad"))


def _point_mapping(
    values: Mapping[str, PointTarget | Sequence[float]],
    name: str,
) -> Mapping[str, PointTarget]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, PointTarget] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError(f"{name} keys must be non-empty strings")
        point = (
            value
            if isinstance(value, PointTarget)
            else PointTarget(CoordinateFrame.WORLD_ENU, _xyz(value, f"{name}[{key!r}]"))
        )
        if point.frame is not CoordinateFrame.WORLD_ENU:
            raise ValueError(f"{name}[{key!r}] must already use WORLD_ENU")
        result[key.strip()] = point
    return MappingProxyType(result)


class SpatialResolver:
    """Resolve V3 spatial objects without inventing frame or landmark facts."""

    def __init__(
        self,
        *,
        home_pose: FramePose,
        uav_start_pose: FramePose,
        uav_hold_pose: FramePose | None = None,
        camera_pose: FramePose | None = None,
        named_locations: Mapping[str, PointTarget | Sequence[float]] | None = None,
        landmarks: Mapping[str, PointTarget | Sequence[float]] | None = None,
    ) -> None:
        if not isinstance(home_pose, FramePose):
            raise TypeError("home_pose must be a FramePose")
        if not isinstance(uav_start_pose, FramePose):
            raise TypeError("uav_start_pose must be a FramePose")
        if uav_hold_pose is not None and not isinstance(uav_hold_pose, FramePose):
            raise TypeError("uav_hold_pose must be a FramePose or None")
        if camera_pose is not None and not isinstance(camera_pose, FramePose):
            raise TypeError("camera_pose must be a FramePose or None")
        self._home_pose = home_pose
        self._uav_start_pose = uav_start_pose
        self._uav_hold_pose = uav_hold_pose
        self._camera_pose = camera_pose
        self._named_locations = _point_mapping(
            {} if named_locations is None else named_locations,
            "named_locations",
        )
        # ``landmarks`` is the trusted output of visual grounding, never model
        # prose alone. An absent id therefore stays unresolved/fail-closed.
        self._landmarks = _point_mapping(
            {} if landmarks is None else landmarks,
            "landmarks",
        )

    @property
    def named_locations(self) -> Mapping[str, PointTarget]:
        return self._named_locations

    @property
    def grounded_landmarks(self) -> Mapping[str, PointTarget]:
        return self._landmarks

    def with_uav_hold_pose(self, hold_pose: FramePose) -> "SpatialResolver":
        """Return an equivalent resolver with one trusted HOLD-frame anchor.

        The initial mission resolver deliberately has no ``UAV_HOLD_FLU``
        pose.  Obstacle replanning may add that pose only after the
        supervisory HOVER handshake has completed.  Returning a new resolver
        keeps the original initial-planning boundary immutable.
        """

        if not isinstance(hold_pose, FramePose):
            raise TypeError("hold_pose must be a FramePose")
        return SpatialResolver(
            home_pose=self._home_pose,
            uav_start_pose=self._uav_start_pose,
            uav_hold_pose=hold_pose,
            camera_pose=self._camera_pose,
            named_locations=self._named_locations,
            landmarks=self._landmarks,
        )

    def resolve_point(
        self,
        frame: CoordinateFrame,
        xyz_m: Sequence[float],
    ) -> tuple[float, float, float]:
        if not isinstance(frame, CoordinateFrame):
            try:
                frame = CoordinateFrame(frame)
            except (TypeError, ValueError) as exc:
                raise SpatialResolutionError("unknown coordinate frame") from exc
        point = _xyz(xyz_m, "xyz_m")
        if frame is CoordinateFrame.WORLD_ENU:
            return point
        pose = self._pose_for_frame(frame)
        # HOME_ENU explicitly retains ENU axes. FLU frames rotate about +z.
        yaw = 0.0 if frame is CoordinateFrame.HOME_ENU else pose.yaw_rad
        c, s = cos(yaw), sin(yaw)
        return (
            pose.xyz_m[0] + c * point[0] - s * point[1],
            pose.xyz_m[1] + s * point[0] + c * point[1],
            pose.xyz_m[2] + point[2],
        )

    def resolve_target(self, target: SpatialTarget) -> PointTarget | RouteTarget:
        if isinstance(target, NamedLocationTarget):
            try:
                return self._named_locations[target.name]
            except KeyError:
                raise UnresolvedSpatialReferenceError(
                    f"named location is not registered: {target.name}"
                ) from None
        if isinstance(target, PointTarget):
            return PointTarget(CoordinateFrame.WORLD_ENU, self.resolve_point(target.frame, target.xyz_m))
        if isinstance(target, RouteTarget):
            return RouteTarget(
                CoordinateFrame.WORLD_ENU,
                tuple(self.resolve_point(target.frame, point) for point in target.waypoints_xyz_m),
            )
        if isinstance(target, RelationalPointTarget):
            reference = self._grounded_reference(target.reference_id)
            direction = self._relation_direction(target.relation, target.frame)
            return PointTarget(
                CoordinateFrame.WORLD_ENU,
                tuple(reference.xyz_m[index] + target.distance_m * direction[index] for index in range(3)),
            )
        raise TypeError("target must be a SpatialTarget")

    def resolve_region(self, region: RegionSpec) -> RegionSpec:
        if isinstance(region, RelationalRegion):
            reference = self._grounded_reference(region.reference_id)
            direction = self._relation_direction(region.relation, region.frame)
            center = tuple(
                reference.xyz_m[index] + region.distance_m * direction[index]
                for index in range(3)
            )
            return RectangleRegion(
                CoordinateFrame.WORLD_ENU,
                center,
                region.extent_m[0],
                region.extent_m[1],
            )
        frame = region.frame
        yaw_offset_deg = self._yaw_offset_deg(frame)
        if isinstance(region, CircleRegion):
            return CircleRegion(CoordinateFrame.WORLD_ENU, self.resolve_point(frame, region.center_xyz_m), region.radius_m)
        if isinstance(region, RectangleRegion):
            return RectangleRegion(
                CoordinateFrame.WORLD_ENU,
                self.resolve_point(frame, region.center_xyz_m),
                region.width_m,
                region.height_m,
                region.yaw_deg + yaw_offset_deg,
                None if region.entry_point_xyz_m is None else self.resolve_point(frame, region.entry_point_xyz_m),
            )
        if isinstance(region, SectorRegion):
            return SectorRegion(
                CoordinateFrame.WORLD_ENU,
                self.resolve_point(frame, region.origin_xyz_m),
                region.azimuth_center_deg + yaw_offset_deg,
                region.azimuth_span_deg,
                region.distance_range_m,
            )
        if isinstance(region, PolygonRegion):
            return PolygonRegion(
                CoordinateFrame.WORLD_ENU,
                tuple(self.resolve_point(frame, point) for point in region.vertices_xyz_m),
            )
        if isinstance(region, CorridorRegion):
            return CorridorRegion(
                CoordinateFrame.WORLD_ENU,
                tuple(self.resolve_point(frame, point) for point in region.centerline_xyz_m),
                region.half_width_m,
            )
        raise TypeError("region must be a RegionSpec")

    def _pose_for_frame(self, frame: CoordinateFrame) -> FramePose:
        value = {
            CoordinateFrame.HOME_ENU: self._home_pose,
            CoordinateFrame.UAV_START_FLU: self._uav_start_pose,
            CoordinateFrame.UAV_HOLD_FLU: self._uav_hold_pose,
            CoordinateFrame.CAMERA_FLU: self._camera_pose,
        }.get(frame)
        if value is None:
            raise MissingFramePoseError(f"trusted pose is unavailable for {frame.value}")
        return value

    def _yaw_offset_deg(self, frame: CoordinateFrame) -> float:
        if frame in {CoordinateFrame.WORLD_ENU, CoordinateFrame.HOME_ENU}:
            return 0.0
        return self._pose_for_frame(frame).yaw_rad * 180.0 / 3.141592653589793

    def _grounded_reference(self, reference_id: str) -> PointTarget:
        try:
            return self._landmarks[reference_id]
        except KeyError:
            raise UnresolvedSpatialReferenceError(
                f"landmark has not been visually grounded: {reference_id}"
            ) from None

    def _relation_direction(
        self,
        relation: SpatialRelation,
        frame: CoordinateFrame | None,
    ) -> tuple[float, float, float]:
        cardinal = {
            SpatialRelation.NORTH_OF: (0.0, 1.0, 0.0),
            SpatialRelation.SOUTH_OF: (0.0, -1.0, 0.0),
            SpatialRelation.EAST_OF: (1.0, 0.0, 0.0),
            SpatialRelation.WEST_OF: (-1.0, 0.0, 0.0),
            SpatialRelation.ABOVE: (0.0, 0.0, 1.0),
            SpatialRelation.BELOW: (0.0, 0.0, -1.0),
        }
        if relation in cardinal:
            return cardinal[relation]
        if frame is None:
            raise SpatialResolutionError(
                f"{relation.value} is ambiguous without an explicit frame and logged assumption"
            )
        local = {
            SpatialRelation.IN_FRONT_OF: (1.0, 0.0, 0.0),
            SpatialRelation.BEHIND: (-1.0, 0.0, 0.0),
            SpatialRelation.LEFT_OF: (0.0, 1.0, 0.0),
            SpatialRelation.RIGHT_OF: (0.0, -1.0, 0.0),
        }.get(relation)
        if local is None:
            raise SpatialContractError(f"unsupported relation: {relation.value}")
        if frame in {CoordinateFrame.WORLD_ENU, CoordinateFrame.HOME_ENU}:
            yaw = 0.0
        else:
            yaw = self._pose_for_frame(frame).yaw_rad
        c, s = cos(yaw), sin(yaw)
        return (c * local[0] - s * local[1], s * local[0] + c * local[1], local[2])


__all__ = [
    "FramePose",
    "MissingFramePoseError",
    "SpatialResolutionError",
    "SpatialResolver",
    "UnresolvedSpatialReferenceError",
]
