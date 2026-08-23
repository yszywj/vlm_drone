"""Pure geometry generation for SEARCH V3 strategies.

This module never reads Oracle target state and never emits controller-rate
commands.  It converts one macro strategy into bounded observation points.
"""

from __future__ import annotations

from math import atan2, ceil, cos, degrees, hypot, pi, radians, sin, sqrt
import random

from planner.spatial import (
    CircleRegion,
    CoordinateFrame,
    CorridorRegion,
    PolygonRegion,
    RectangleAnchor,
    RectangleRegion,
    RegionSpec,
    RelationalRegion,
    SectorRegion,
)
from skills.search_strategy import SearchStrategyError, SearchStrategySpec, SearchStrategyType


Point3 = tuple[float, float, float]
_EPS = 1e-9
_V1_ANGLES_DEG = (30.0, 90.0, 150.0, 210.0, 270.0, 330.0)


def _require_world(region: RegionSpec) -> None:
    if isinstance(region, RelationalRegion):
        raise SearchStrategyError("RELATIONAL region must be grounded before SEARCH")
    if region.frame is not CoordinateFrame.WORLD_ENU:
        raise SearchStrategyError("SEARCH geometry requires a resolved WORLD_ENU region")


def _altitude(point: Point3, altitude_m: float) -> Point3:
    return (float(point[0]), float(point[1]), float(altitude_m))


def region_center(region: RegionSpec, altitude_m: float | None = None) -> Point3:
    _require_world(region)
    if isinstance(region, (CircleRegion, RectangleRegion)):
        center = region.center_xyz_m
    elif isinstance(region, SectorRegion):
        radius = sum(region.distance_range_m) / 2.0
        angle = radians(region.azimuth_center_deg)
        center = (
            region.origin_xyz_m[0] + radius * cos(angle),
            region.origin_xyz_m[1] + radius * sin(angle),
            region.origin_xyz_m[2],
        )
    elif isinstance(region, PolygonRegion):
        center = (
            sum(point[0] for point in region.vertices_xyz_m) / len(region.vertices_xyz_m),
            sum(point[1] for point in region.vertices_xyz_m) / len(region.vertices_xyz_m),
            sum(point[2] for point in region.vertices_xyz_m) / len(region.vertices_xyz_m),
        )
    elif isinstance(region, CorridorRegion):
        lengths = [
            hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(region.centerline_xyz_m, region.centerline_xyz_m[1:])
        ]
        half = sum(lengths) / 2.0
        traveled = 0.0
        center = region.centerline_xyz_m[0]
        for a, b, length in zip(region.centerline_xyz_m, region.centerline_xyz_m[1:], lengths):
            if traveled + length >= half and length > _EPS:
                ratio = (half - traveled) / length
                center = tuple(a[index] + ratio * (b[index] - a[index]) for index in range(3))  # type: ignore[assignment]
                break
            traveled += length
    else:  # pragma: no cover - exhaustive union
        raise TypeError("unsupported RegionSpec")
    return center if altitude_m is None else _altitude(center, altitude_m)


def point_inside_region(region: RegionSpec, point_xyz_m: Point3) -> bool:
    _require_world(region)
    x, y = point_xyz_m[:2]
    if isinstance(region, CircleRegion):
        return hypot(x - region.center_xyz_m[0], y - region.center_xyz_m[1]) <= region.radius_m + _EPS
    if isinstance(region, RectangleRegion):
        lx, ly = _world_to_rectangle_local(region, point_xyz_m)
        return abs(lx) <= region.width_m / 2.0 + _EPS and abs(ly) <= region.height_m / 2.0 + _EPS
    if isinstance(region, SectorRegion):
        dx, dy = x - region.origin_xyz_m[0], y - region.origin_xyz_m[1]
        distance = hypot(dx, dy)
        angle_delta = _wrap_deg(degrees(atan2(dy, dx)) - region.azimuth_center_deg)
        return (
            region.distance_range_m[0] - _EPS <= distance <= region.distance_range_m[1] + _EPS
            and abs(angle_delta) <= region.azimuth_span_deg / 2.0 + _EPS
        )
    if isinstance(region, PolygonRegion):
        return _point_in_polygon(x, y, region.vertices_xyz_m)
    if isinstance(region, CorridorRegion):
        return min(
            _distance_to_segment_xy(point_xyz_m, a, b)
            for a, b in zip(region.centerline_xyz_m, region.centerline_xyz_m[1:])
        ) <= region.half_width_m + _EPS
    raise TypeError("unsupported RegionSpec")


def rectangle_anchor_point(
    region: RectangleRegion,
    anchor: RectangleAnchor,
    *,
    altitude_m: float,
) -> Point3:
    _require_world(region)
    if not isinstance(anchor, RectangleAnchor):
        anchor = RectangleAnchor(anchor)
    if anchor is RectangleAnchor.ENTRY_POINT:
        if region.entry_point_xyz_m is None:
            raise SearchStrategyError("rectangle has no ENTRY_POINT")
        if not point_inside_region(region, region.entry_point_xyz_m):
            raise SearchStrategyError("rectangle ENTRY_POINT lies outside the region")
        return _altitude(region.entry_point_xyz_m, altitude_m)
    half_x, half_y = region.width_m / 2.0, region.height_m / 2.0
    local = {
        RectangleAnchor.CENTER: (0.0, 0.0),
        RectangleAnchor.NORTH_EDGE_MIDPOINT: (0.0, half_y),
        RectangleAnchor.SOUTH_EDGE_MIDPOINT: (0.0, -half_y),
        RectangleAnchor.EAST_EDGE_MIDPOINT: (half_x, 0.0),
        RectangleAnchor.WEST_EDGE_MIDPOINT: (-half_x, 0.0),
        RectangleAnchor.NORTHWEST_CORNER: (-half_x, half_y),
        RectangleAnchor.NORTHEAST_CORNER: (half_x, half_y),
        RectangleAnchor.SOUTHWEST_CORNER: (-half_x, -half_y),
        RectangleAnchor.SOUTHEAST_CORNER: (half_x, -half_y),
    }[anchor]
    return _altitude(_rectangle_local_to_world(region, local), altitude_m)


def nearest_boundary_point(region: RegionSpec, point_xyz_m: Point3, altitude_m: float) -> Point3:
    _require_world(region)
    if isinstance(region, CircleRegion):
        dx = point_xyz_m[0] - region.center_xyz_m[0]
        dy = point_xyz_m[1] - region.center_xyz_m[1]
        length = hypot(dx, dy)
        dx, dy = ((1.0, 0.0) if length <= _EPS else (dx / length, dy / length))
        return (region.center_xyz_m[0] + region.radius_m * dx, region.center_xyz_m[1] + region.radius_m * dy, altitude_m)
    if isinstance(region, RectangleRegion):
        lx, ly = _world_to_rectangle_local(region, point_xyz_m)
        hx, hy = region.width_m / 2.0, region.height_m / 2.0
        if abs(lx) <= hx and abs(ly) <= hy:
            distances = ((hx - lx, (hx, ly)), (lx + hx, (-hx, ly)), (hy - ly, (lx, hy)), (ly + hy, (lx, -hy)))
            local = min(distances, key=lambda item: item[0])[1]
        else:
            local = (min(hx, max(-hx, lx)), min(hy, max(-hy, ly)))
        return _altitude(_rectangle_local_to_world(region, local), altitude_m)
    if isinstance(region, PolygonRegion):
        candidates = [
            _nearest_on_segment_xy(point_xyz_m, a, b, altitude_m)
            for a, b in zip(region.vertices_xyz_m, region.vertices_xyz_m[1:] + region.vertices_xyz_m[:1])
        ]
        return min(candidates, key=lambda item: hypot(item[0] - point_xyz_m[0], item[1] - point_xyz_m[1]))
    if isinstance(region, SectorRegion):
        dx, dy = point_xyz_m[0] - region.origin_xyz_m[0], point_xyz_m[1] - region.origin_xyz_m[1]
        radius = min(region.distance_range_m[1], max(region.distance_range_m[0], hypot(dx, dy)))
        delta = max(-region.azimuth_span_deg / 2.0, min(region.azimuth_span_deg / 2.0, _wrap_deg(degrees(atan2(dy, dx)) - region.azimuth_center_deg)))
        angle = radians(region.azimuth_center_deg + delta)
        return (region.origin_xyz_m[0] + radius * cos(angle), region.origin_xyz_m[1] + radius * sin(angle), altitude_m)
    if isinstance(region, CorridorRegion):
        center = min(
            (_nearest_on_segment_xy(point_xyz_m, a, b, altitude_m) for a, b in zip(region.centerline_xyz_m, region.centerline_xyz_m[1:])),
            key=lambda item: hypot(item[0] - point_xyz_m[0], item[1] - point_xyz_m[1]),
        )
        dx, dy = point_xyz_m[0] - center[0], point_xyz_m[1] - center[1]
        length = hypot(dx, dy)
        if length <= _EPS:
            dx, dy, length = 0.0, 1.0, 1.0
        return (center[0] + region.half_width_m * dx / length, center[1] + region.half_width_m * dy / length, altitude_m)
    raise TypeError("unsupported RegionSpec")


def generate_search_waypoints(
    region: RegionSpec,
    strategy: SearchStrategySpec,
    *,
    altitude_m: float,
) -> tuple[Point3, ...]:
    """Compile a bounded macro strategy to observation waypoints."""

    _require_world(region)
    if not isinstance(strategy, SearchStrategySpec):
        raise TypeError("strategy must be a SearchStrategySpec")
    if altitude_m <= 0.0:
        raise SearchStrategyError("altitude_m must be greater than zero")
    kind = strategy.kind
    if kind is SearchStrategyType.PERIMETER_V1:
        if not isinstance(region, CircleRegion):
            raise SearchStrategyError("PERIMETER_V1 is defined only for CIRCLE")
        points = tuple(
            (region.center_xyz_m[0] + region.radius_m * cos(radians(angle)), region.center_xyz_m[1] + region.radius_m * sin(radians(angle)), altitude_m)
            for angle in _V1_ANGLES_DEG
        )
    elif kind is SearchStrategyType.PERIMETER:
        points = _perimeter(region, strategy.spacing_m, altitude_m, strategy.max_viewpoints)
    elif kind is SearchStrategyType.LAWNMOWER:
        points = _lawnmower(region, strategy.spacing_m, altitude_m, strategy.max_viewpoints)
    elif kind in {SearchStrategyType.SPIRAL_IN, SearchStrategyType.SPIRAL_OUT}:
        points = _spiral(region, strategy.spacing_m, altitude_m, strategy.max_viewpoints, inward=kind is SearchStrategyType.SPIRAL_IN)
    elif kind is SearchStrategyType.SECTOR_SWEEP:
        if not isinstance(region, SectorRegion):
            raise SearchStrategyError("SECTOR_SWEEP requires a SECTOR region")
        points = _sector_sweep(region, strategy.spacing_m, altitude_m, strategy.max_viewpoints)
    elif kind is SearchStrategyType.CORRIDOR_FOLLOW:
        if not isinstance(region, CorridorRegion):
            raise SearchStrategyError("CORRIDOR_FOLLOW requires a CORRIDOR region")
        points = tuple(_altitude(point, altitude_m) for point in region.centerline_xyz_m)
    elif kind is SearchStrategyType.RANDOM_COVERAGE:
        points = _random_points(region, altitude_m, strategy.max_viewpoints, strategy.random_seed)
    elif kind is SearchStrategyType.MODEL_WAYPOINTS:
        points = tuple(_altitude(point, altitude_m) for point in strategy.model_waypoints_xyz_m)
        outside = [index for index, point in enumerate(points) if not point_inside_region(region, point)]
        if outside:
            raise SearchStrategyError(
                "MODEL_WAYPOINTS contains points outside the region at indexes: "
                + ", ".join(str(index) for index in outside)
            )
    elif kind is SearchStrategyType.ADAPTIVE_NEXT_BEST_VIEW:
        raise SearchStrategyError(
            "ADAPTIVE_NEXT_BEST_VIEW requires a runtime next-best-view provider"
        )
    else:  # pragma: no cover - exhaustive enum
        raise SearchStrategyError(f"unsupported strategy: {kind.value}")
    if not points:
        raise SearchStrategyError("strategy produced no waypoints")
    return tuple(points[: strategy.max_viewpoints])


def _perimeter(region: RegionSpec, spacing: float, altitude: float, limit: int) -> tuple[Point3, ...]:
    if isinstance(region, CircleRegion):
        count = min(limit, max(8, int(ceil(2.0 * pi * region.radius_m / spacing))))
        return tuple((region.center_xyz_m[0] + region.radius_m * cos(2.0 * pi * i / count), region.center_xyz_m[1] + region.radius_m * sin(2.0 * pi * i / count), altitude) for i in range(count))
    if isinstance(region, RectangleRegion):
        anchors = (RectangleAnchor.SOUTHWEST_CORNER, RectangleAnchor.SOUTHEAST_CORNER, RectangleAnchor.NORTHEAST_CORNER, RectangleAnchor.NORTHWEST_CORNER)
        return tuple(rectangle_anchor_point(region, anchor, altitude_m=altitude) for anchor in anchors)
    if isinstance(region, PolygonRegion):
        return tuple(_altitude(point, altitude) for point in region.vertices_xyz_m[:limit])
    if isinstance(region, SectorRegion):
        near, far = region.distance_range_m
        half = region.azimuth_span_deg / 2.0
        samples = max(2, min(limit // 2, int(ceil(radians(region.azimuth_span_deg) * far / spacing)) + 1))
        angles = [region.azimuth_center_deg - half + region.azimuth_span_deg * i / (samples - 1) for i in range(samples)]
        points = [(region.origin_xyz_m[0] + far * cos(radians(angle)), region.origin_xyz_m[1] + far * sin(radians(angle)), altitude) for angle in angles]
        points += [(region.origin_xyz_m[0] + near * cos(radians(angle)), region.origin_xyz_m[1] + near * sin(radians(angle)), altitude) for angle in reversed(angles)]
        return tuple(points[:limit])
    if isinstance(region, CorridorRegion):
        return tuple(_altitude(point, altitude) for point in region.centerline_xyz_m[:limit])
    raise TypeError("unsupported RegionSpec")


def _lawnmower(region: RegionSpec, spacing: float, altitude: float, limit: int) -> tuple[Point3, ...]:
    min_x, max_x, min_y, max_y = _bounds(region)
    row_count = max(1, int(ceil((max_y - min_y) / spacing)))
    points: list[Point3] = []
    for row in range(row_count + 1):
        y = min(max_y, min_y + row * spacing)
        samples = max(2, int(ceil((max_x - min_x) / spacing)) + 1)
        xs = [min_x + (max_x - min_x) * i / (samples - 1) for i in range(samples)]
        if row % 2:
            xs.reverse()
        inside = [(x, y, altitude) for x in xs if point_inside_region(region, (x, y, altitude))]
        if inside:
            points.extend((inside[0], inside[-1]) if len(inside) > 1 else inside)
        if len(points) >= limit:
            break
    return tuple(points[:limit]) or (region_center(region, altitude),)


def _spiral(region: RegionSpec, spacing: float, altitude: float, limit: int, *, inward: bool) -> tuple[Point3, ...]:
    center = region_center(region, altitude)
    min_x, max_x, min_y, max_y = _bounds(region)
    radius = max(max_x - min_x, max_y - min_y) / 2.0
    turns = max(1.0, radius / spacing)
    candidates: list[Point3] = []
    count = min(limit * 4, max(12, int(ceil(turns * 16))))
    for index in range(count):
        fraction = index / max(1, count - 1)
        radial = radius * (1.0 - fraction if inward else fraction)
        angle = 2.0 * pi * turns * fraction
        point = (center[0] + radial * cos(angle), center[1] + radial * sin(angle), altitude)
        if point_inside_region(region, point):
            candidates.append(point)
            if len(candidates) >= limit:
                break
    return tuple(candidates) or (center,)


def _sector_sweep(region: SectorRegion, spacing: float, altitude: float, limit: int) -> tuple[Point3, ...]:
    near, far = region.distance_range_m
    arc = radians(region.azimuth_span_deg) * far
    count = max(2, min(limit, int(ceil(arc / spacing)) + 1))
    result: list[Point3] = []
    for index in range(count):
        fraction = index / (count - 1)
        angle = radians(region.azimuth_center_deg - region.azimuth_span_deg / 2.0 + fraction * region.azimuth_span_deg)
        radius = far if index % 2 == 0 else near
        result.append((region.origin_xyz_m[0] + radius * cos(angle), region.origin_xyz_m[1] + radius * sin(angle), altitude))
    return tuple(result)


def _random_points(region: RegionSpec, altitude: float, count: int, seed: int | None) -> tuple[Point3, ...]:
    generator = random.Random(0 if seed is None else seed)
    min_x, max_x, min_y, max_y = _bounds(region)
    points: list[Point3] = []
    for _ in range(count * 100):
        point = (generator.uniform(min_x, max_x), generator.uniform(min_y, max_y), altitude)
        if point_inside_region(region, point):
            points.append(point)
            if len(points) == count:
                break
    if not points:
        points.append(region_center(region, altitude))
    return tuple(points)


def _bounds(region: RegionSpec) -> tuple[float, float, float, float]:
    if isinstance(region, CircleRegion):
        return (region.center_xyz_m[0] - region.radius_m, region.center_xyz_m[0] + region.radius_m, region.center_xyz_m[1] - region.radius_m, region.center_xyz_m[1] + region.radius_m)
    if isinstance(region, RectangleRegion):
        corners = [rectangle_anchor_point(region, anchor, altitude_m=region.center_xyz_m[2]) for anchor in (RectangleAnchor.NORTHWEST_CORNER, RectangleAnchor.NORTHEAST_CORNER, RectangleAnchor.SOUTHWEST_CORNER, RectangleAnchor.SOUTHEAST_CORNER)]
    elif isinstance(region, PolygonRegion):
        corners = list(region.vertices_xyz_m)
    elif isinstance(region, SectorRegion):
        far = region.distance_range_m[1]
        return (region.origin_xyz_m[0] - far, region.origin_xyz_m[0] + far, region.origin_xyz_m[1] - far, region.origin_xyz_m[1] + far)
    elif isinstance(region, CorridorRegion):
        return (min(point[0] for point in region.centerline_xyz_m) - region.half_width_m, max(point[0] for point in region.centerline_xyz_m) + region.half_width_m, min(point[1] for point in region.centerline_xyz_m) - region.half_width_m, max(point[1] for point in region.centerline_xyz_m) + region.half_width_m)
    else:
        raise TypeError("unsupported RegionSpec")
    return (min(point[0] for point in corners), max(point[0] for point in corners), min(point[1] for point in corners), max(point[1] for point in corners))


def _world_to_rectangle_local(region: RectangleRegion, point: Point3) -> tuple[float, float]:
    dx, dy = point[0] - region.center_xyz_m[0], point[1] - region.center_xyz_m[1]
    c, s = cos(radians(region.yaw_deg)), sin(radians(region.yaw_deg))
    return (c * dx + s * dy, -s * dx + c * dy)


def _rectangle_local_to_world(region: RectangleRegion, local: tuple[float, float]) -> Point3:
    c, s = cos(radians(region.yaw_deg)), sin(radians(region.yaw_deg))
    return (region.center_xyz_m[0] + c * local[0] - s * local[1], region.center_xyz_m[1] + s * local[0] + c * local[1], region.center_xyz_m[2])


def _point_in_polygon(x: float, y: float, vertices: tuple[Point3, ...]) -> bool:
    inside = False
    previous = vertices[-1]
    for current in vertices:
        if _distance_to_segment_xy((x, y, 0.0), previous, current) <= _EPS:
            return True
        if ((current[1] > y) != (previous[1] > y)) and x < (previous[0] - current[0]) * (y - current[1]) / (previous[1] - current[1]) + current[0]:
            inside = not inside
        previous = current
    return inside


def _nearest_on_segment_xy(point: Point3, a: Point3, b: Point3, altitude: float) -> Point3:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    ratio = 0.0 if length_sq <= _EPS else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq))
    return (a[0] + ratio * dx, a[1] + ratio * dy, altitude)


def _distance_to_segment_xy(point: Point3, a: Point3, b: Point3) -> float:
    nearest = _nearest_on_segment_xy(point, a, b, point[2])
    return hypot(point[0] - nearest[0], point[1] - nearest[1])


def _wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


__all__ = [
    "Point3", "generate_search_waypoints", "nearest_boundary_point",
    "point_inside_region", "rectangle_anchor_point", "region_center",
]
