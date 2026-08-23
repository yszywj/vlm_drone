"""Spatial Contract V3 value objects.

The V3 contract is deliberately independent from the named-location-only V2
schemas.  Every numeric coordinate carries an explicit coordinate frame and
all model-facing parsers are exact: unknown fields, non-finite values and
degenerate geometry fail at the trust boundary.

Frame conventions
-----------------
``WORLD_ENU`` uses x east, y north and z up. ``HOME_ENU`` has the same axes
with ``home`` as its origin. ``UAV_START_FLU`` and ``UAV_HOLD_FLU`` use x
forward, y left and z up at the corresponding UAV pose. ``CAMERA_FLU`` uses
the current camera origin and FLU axes.  Relative frames therefore require a
trusted pose before they can be resolved to ``WORLD_ENU``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import ClassVar, TypeAlias


class SpatialContractError(ValueError):
    """Raised when model-supplied V3 spatial data is structurally invalid."""


class CoordinateFrame(str, Enum):
    WORLD_ENU = "WORLD_ENU"
    HOME_ENU = "HOME_ENU"
    UAV_START_FLU = "UAV_START_FLU"
    UAV_HOLD_FLU = "UAV_HOLD_FLU"
    CAMERA_FLU = "CAMERA_FLU"


class SpatialRelation(str, Enum):
    LEFT_OF = "LEFT_OF"
    RIGHT_OF = "RIGHT_OF"
    IN_FRONT_OF = "IN_FRONT_OF"
    BEHIND = "BEHIND"
    NORTH_OF = "NORTH_OF"
    SOUTH_OF = "SOUTH_OF"
    EAST_OF = "EAST_OF"
    WEST_OF = "WEST_OF"
    ABOVE = "ABOVE"
    BELOW = "BELOW"


class RectangleAnchor(str, Enum):
    CENTER = "CENTER"
    NORTH_EDGE_MIDPOINT = "NORTH_EDGE_MIDPOINT"
    SOUTH_EDGE_MIDPOINT = "SOUTH_EDGE_MIDPOINT"
    EAST_EDGE_MIDPOINT = "EAST_EDGE_MIDPOINT"
    WEST_EDGE_MIDPOINT = "WEST_EDGE_MIDPOINT"
    NORTHWEST_CORNER = "NORTHWEST_CORNER"
    NORTHEAST_CORNER = "NORTHEAST_CORNER"
    SOUTHWEST_CORNER = "SOUTHWEST_CORNER"
    SOUTHEAST_CORNER = "SOUTHEAST_CORNER"
    ENTRY_POINT = "ENTRY_POINT"


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    allowed = ", ".join(item.value for item in enum_type)
    raise SpatialContractError(f"{name} must be one of: {allowed}")


def _text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise SpatialContractError(f"{name} must be non-empty")
    if len(result) > maximum:
        raise SpatialContractError(f"{name} must contain at most {maximum} characters")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise SpatialContractError(f"{name} must be a finite number")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise SpatialContractError(f"{name} must be greater than zero")
    return result


def _vector(value: object, length: int, name: str) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise SpatialContractError(f"{name} must contain exactly {length} numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


def _vectors(
    value: object,
    *,
    minimum: int,
    maximum: int,
    name: str,
) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    if not minimum <= len(value) <= maximum:
        raise SpatialContractError(
            f"{name} must contain between {minimum} and {maximum} points"
        )
    return tuple(_vector(item, 3, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _exact(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    fields = frozenset(value)
    unknown = fields - required - optional
    missing = required - fields
    if unknown:
        raise SpatialContractError(f"{name} contains unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise SpatialContractError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    return value


@dataclass(frozen=True, slots=True)
class NamedLocationTarget:
    name: str
    kind: ClassVar[str] = "NAMED_LOCATION"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name", maximum=128))

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "name": self.name}


@dataclass(frozen=True, slots=True)
class PointTarget:
    frame: CoordinateFrame
    xyz_m: tuple[float, float, float]
    kind: ClassVar[str] = "POINT"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _enum(self.frame, CoordinateFrame, "frame"))
        object.__setattr__(self, "xyz_m", _vector(self.xyz_m, 3, "xyz_m"))

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "frame": self.frame.value, "xyz_m": list(self.xyz_m)}


@dataclass(frozen=True, slots=True)
class RelationalPointTarget:
    relation: SpatialRelation
    reference_id: str
    distance_m: float
    frame: CoordinateFrame | None = None
    kind: ClassVar[str] = "RELATIONAL_POINT"

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", _enum(self.relation, SpatialRelation, "relation"))
        object.__setattr__(self, "reference_id", _text(self.reference_id, "reference_id", maximum=128))
        object.__setattr__(self, "distance_m", _positive(self.distance_m, "distance_m"))
        if self.frame is not None:
            object.__setattr__(self, "frame", _enum(self.frame, CoordinateFrame, "frame"))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind,
            "relation": self.relation.value,
            "reference_id": self.reference_id,
            "distance_m": self.distance_m,
        }
        if self.frame is not None:
            result["frame"] = self.frame.value
        return result


@dataclass(frozen=True, slots=True)
class RouteTarget:
    frame: CoordinateFrame
    waypoints_xyz_m: tuple[tuple[float, float, float], ...]
    kind: ClassVar[str] = "ROUTE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _enum(self.frame, CoordinateFrame, "frame"))
        object.__setattr__(
            self,
            "waypoints_xyz_m",
            _vectors(self.waypoints_xyz_m, minimum=2, maximum=64, name="waypoints_xyz_m"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "frame": self.frame.value,
            "waypoints_xyz_m": [list(point) for point in self.waypoints_xyz_m],
        }


SpatialTarget: TypeAlias = NamedLocationTarget | PointTarget | RelationalPointTarget | RouteTarget


@dataclass(frozen=True, slots=True)
class CircleRegion:
    frame: CoordinateFrame
    center_xyz_m: tuple[float, float, float]
    radius_m: float
    shape: ClassVar[str] = "CIRCLE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _enum(self.frame, CoordinateFrame, "frame"))
        object.__setattr__(self, "center_xyz_m", _vector(self.center_xyz_m, 3, "center_xyz_m"))
        object.__setattr__(self, "radius_m", _positive(self.radius_m, "radius_m"))

    def to_dict(self) -> dict[str, object]:
        return {"shape": self.shape, "frame": self.frame.value, "center_xyz_m": list(self.center_xyz_m), "radius_m": self.radius_m}


@dataclass(frozen=True, slots=True)
class RectangleRegion:
    frame: CoordinateFrame
    center_xyz_m: tuple[float, float, float]
    width_m: float
    height_m: float
    yaw_deg: float = 0.0
    entry_point_xyz_m: tuple[float, float, float] | None = None
    shape: ClassVar[str] = "RECTANGLE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _enum(self.frame, CoordinateFrame, "frame"))
        object.__setattr__(self, "center_xyz_m", _vector(self.center_xyz_m, 3, "center_xyz_m"))
        object.__setattr__(self, "width_m", _positive(self.width_m, "width_m"))
        object.__setattr__(self, "height_m", _positive(self.height_m, "height_m"))
        object.__setattr__(self, "yaw_deg", _finite(self.yaw_deg, "yaw_deg"))
        if self.entry_point_xyz_m is not None:
            object.__setattr__(self, "entry_point_xyz_m", _vector(self.entry_point_xyz_m, 3, "entry_point_xyz_m"))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "shape": self.shape,
            "frame": self.frame.value,
            "center_xyz_m": list(self.center_xyz_m),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "yaw_deg": self.yaw_deg,
        }
        if self.entry_point_xyz_m is not None:
            result["entry_point_xyz_m"] = list(self.entry_point_xyz_m)
        return result


@dataclass(frozen=True, slots=True)
class SectorRegion:
    frame: CoordinateFrame
    origin_xyz_m: tuple[float, float, float]
    azimuth_center_deg: float
    azimuth_span_deg: float
    distance_range_m: tuple[float, float]
    shape: ClassVar[str] = "SECTOR"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _enum(self.frame, CoordinateFrame, "frame"))
        object.__setattr__(self, "origin_xyz_m", _vector(self.origin_xyz_m, 3, "origin_xyz_m"))
        object.__setattr__(self, "azimuth_center_deg", _finite(self.azimuth_center_deg, "azimuth_center_deg"))
        span = _positive(self.azimuth_span_deg, "azimuth_span_deg")
        if span > 360.0:
            raise SpatialContractError("azimuth_span_deg must be at most 360")
        object.__setattr__(self, "azimuth_span_deg", span)
        ranges = _vector(self.distance_range_m, 2, "distance_range_m")
        if ranges[0] < 0.0 or ranges[0] >= ranges[1]:
            raise SpatialContractError("distance_range_m must satisfy 0 <= near < far")
        object.__setattr__(self, "distance_range_m", ranges)

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "frame": self.frame.value,
            "origin_xyz_m": list(self.origin_xyz_m),
            "azimuth_center_deg": self.azimuth_center_deg,
            "azimuth_span_deg": self.azimuth_span_deg,
            "distance_range_m": list(self.distance_range_m),
        }


@dataclass(frozen=True, slots=True)
class PolygonRegion:
    frame: CoordinateFrame
    vertices_xyz_m: tuple[tuple[float, float, float], ...]
    shape: ClassVar[str] = "POLYGON"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _enum(self.frame, CoordinateFrame, "frame"))
        points = _vectors(self.vertices_xyz_m, minimum=3, maximum=64, name="vertices_xyz_m")
        area_twice = sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        if abs(area_twice) <= 1e-9:
            raise SpatialContractError("vertices_xyz_m must form a non-degenerate XY polygon")
        object.__setattr__(self, "vertices_xyz_m", points)

    def to_dict(self) -> dict[str, object]:
        return {"shape": self.shape, "frame": self.frame.value, "vertices_xyz_m": [list(point) for point in self.vertices_xyz_m]}


@dataclass(frozen=True, slots=True)
class CorridorRegion:
    frame: CoordinateFrame
    centerline_xyz_m: tuple[tuple[float, float, float], ...]
    half_width_m: float
    shape: ClassVar[str] = "CORRIDOR"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _enum(self.frame, CoordinateFrame, "frame"))
        points = _vectors(self.centerline_xyz_m, minimum=2, maximum=64, name="centerline_xyz_m")
        if all(points[index] == points[index + 1] for index in range(len(points) - 1)):
            raise SpatialContractError("centerline_xyz_m must contain a non-zero segment")
        object.__setattr__(self, "centerline_xyz_m", points)
        object.__setattr__(self, "half_width_m", _positive(self.half_width_m, "half_width_m"))

    def to_dict(self) -> dict[str, object]:
        return {"shape": self.shape, "frame": self.frame.value, "centerline_xyz_m": [list(point) for point in self.centerline_xyz_m], "half_width_m": self.half_width_m}


@dataclass(frozen=True, slots=True)
class RelationalRegion:
    relation: SpatialRelation
    reference_id: str
    distance_m: float
    extent_m: tuple[float, float]
    frame: CoordinateFrame | None = None
    shape: ClassVar[str] = "RELATIONAL"

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", _enum(self.relation, SpatialRelation, "relation"))
        object.__setattr__(self, "reference_id", _text(self.reference_id, "reference_id", maximum=128))
        object.__setattr__(self, "distance_m", _positive(self.distance_m, "distance_m"))
        extents = _vector(self.extent_m, 2, "extent_m")
        if extents[0] <= 0.0 or extents[1] <= 0.0:
            raise SpatialContractError("extent_m values must be greater than zero")
        object.__setattr__(self, "extent_m", extents)
        if self.frame is not None:
            object.__setattr__(self, "frame", _enum(self.frame, CoordinateFrame, "frame"))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "shape": self.shape,
            "relation": self.relation.value,
            "reference_id": self.reference_id,
            "distance_m": self.distance_m,
            "extent_m": list(self.extent_m),
        }
        if self.frame is not None:
            result["frame"] = self.frame.value
        return result


RegionSpec: TypeAlias = CircleRegion | RectangleRegion | SectorRegion | PolygonRegion | CorridorRegion | RelationalRegion


@dataclass(frozen=True, slots=True)
class SpatialAssumption:
    source_text: str
    interpretation: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_text", _text(self.source_text, "source_text", maximum=256))
        object.__setattr__(self, "interpretation", _text(self.interpretation, "interpretation", maximum=512))
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise SpatialContractError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def from_dict(cls, value: object) -> SpatialAssumption:
        data = _exact(value, name="SpatialAssumption", required=frozenset({"source_text", "interpretation", "confidence"}))
        return cls(data["source_text"], data["interpretation"], data["confidence"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"source_text": self.source_text, "interpretation": self.interpretation, "confidence": self.confidence}


def spatial_target_from_dict(value: object) -> SpatialTarget:
    if not isinstance(value, Mapping):
        raise TypeError("SpatialTarget must be an object")
    kind = value.get("kind")
    if kind == NamedLocationTarget.kind:
        data = _exact(value, name="NamedLocationTarget", required=frozenset({"kind", "name"}))
        return NamedLocationTarget(data["name"])  # type: ignore[arg-type]
    if kind == PointTarget.kind:
        data = _exact(value, name="PointTarget", required=frozenset({"kind", "frame", "xyz_m"}))
        return PointTarget(data["frame"], data["xyz_m"])  # type: ignore[arg-type]
    if kind == RelationalPointTarget.kind:
        data = _exact(value, name="RelationalPointTarget", required=frozenset({"kind", "relation", "reference_id", "distance_m"}), optional=frozenset({"frame"}))
        return RelationalPointTarget(data["relation"], data["reference_id"], data["distance_m"], data.get("frame"))  # type: ignore[arg-type]
    if kind == RouteTarget.kind:
        data = _exact(value, name="RouteTarget", required=frozenset({"kind", "frame", "waypoints_xyz_m"}))
        return RouteTarget(data["frame"], data["waypoints_xyz_m"])  # type: ignore[arg-type]
    raise SpatialContractError("SpatialTarget.kind is unsupported")


def region_spec_from_dict(value: object) -> RegionSpec:
    if not isinstance(value, Mapping):
        raise TypeError("RegionSpec must be an object")
    shape = value.get("shape")
    if shape == CircleRegion.shape:
        data = _exact(value, name="CircleRegion", required=frozenset({"shape", "frame", "center_xyz_m", "radius_m"}))
        return CircleRegion(data["frame"], data["center_xyz_m"], data["radius_m"])  # type: ignore[arg-type]
    if shape == RectangleRegion.shape:
        data = _exact(value, name="RectangleRegion", required=frozenset({"shape", "frame", "center_xyz_m", "width_m", "height_m"}), optional=frozenset({"yaw_deg", "entry_point_xyz_m"}))
        return RectangleRegion(data["frame"], data["center_xyz_m"], data["width_m"], data["height_m"], data.get("yaw_deg", 0.0), data.get("entry_point_xyz_m"))  # type: ignore[arg-type]
    if shape == SectorRegion.shape:
        data = _exact(value, name="SectorRegion", required=frozenset({"shape", "frame", "origin_xyz_m", "azimuth_center_deg", "azimuth_span_deg", "distance_range_m"}))
        return SectorRegion(data["frame"], data["origin_xyz_m"], data["azimuth_center_deg"], data["azimuth_span_deg"], data["distance_range_m"])  # type: ignore[arg-type]
    if shape == PolygonRegion.shape:
        data = _exact(value, name="PolygonRegion", required=frozenset({"shape", "frame", "vertices_xyz_m"}))
        return PolygonRegion(data["frame"], data["vertices_xyz_m"])  # type: ignore[arg-type]
    if shape == CorridorRegion.shape:
        data = _exact(value, name="CorridorRegion", required=frozenset({"shape", "frame", "centerline_xyz_m", "half_width_m"}))
        return CorridorRegion(data["frame"], data["centerline_xyz_m"], data["half_width_m"])  # type: ignore[arg-type]
    if shape == RelationalRegion.shape:
        data = _exact(value, name="RelationalRegion", required=frozenset({"shape", "relation", "reference_id", "distance_m", "extent_m"}), optional=frozenset({"frame"}))
        return RelationalRegion(data["relation"], data["reference_id"], data["distance_m"], data["extent_m"], data.get("frame"))  # type: ignore[arg-type]
    raise SpatialContractError("RegionSpec.shape is unsupported")


__all__ = [
    "CircleRegion", "CoordinateFrame", "CorridorRegion", "NamedLocationTarget",
    "PointTarget", "PolygonRegion", "RectangleAnchor", "RectangleRegion",
    "RegionSpec", "RelationalPointTarget", "RelationalRegion", "RouteTarget",
    "SectorRegion", "SpatialAssumption", "SpatialContractError", "SpatialRelation",
    "SpatialTarget", "region_spec_from_dict", "spatial_target_from_dict",
]
