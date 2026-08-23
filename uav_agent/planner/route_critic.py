"""Configurable counterexample-only critic for Qwen spatial routes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist, sqrt

from env.obstacle_registry import ObstacleRegistry
from common.obstacle_types import ObstacleAABB
from planner.route_types import RouteConstraints, RouteDraft
from planner.spatial import CoordinateFrame
from planner.spatial_resolver import SpatialResolutionError, SpatialResolver


class RouteValidationMode(str, Enum):
    OPEN_SIM = "open_sim"
    CRITIC_SIM = "critic_sim"
    STRICT = "strict"


class RouteCriticStatus(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"


class RouteViolationType(str, Enum):
    PATH_INTERSECTS_OBSTACLE = "PATH_INTERSECTS_OBSTACLE"
    INSUFFICIENT_CLEARANCE = "INSUFFICIENT_CLEARANCE"
    OUTSIDE_SCENE = "OUTSIDE_SCENE"
    ALTITUDE_OUT_OF_BOUNDS = "ALTITUDE_OUT_OF_BOUNDS"
    SEGMENT_TOO_LONG = "SEGMENT_TOO_LONG"
    ROUTE_TOO_LONG = "ROUTE_TOO_LONG"
    DOES_NOT_REJOIN_GOAL = "DOES_NOT_REJOIN_GOAL"
    UNRESOLVED_FRAME = "UNRESOLVED_FRAME"
    TOO_MANY_WAYPOINTS = "TOO_MANY_WAYPOINTS"


@dataclass(frozen=True, slots=True)
class RouteViolation:
    type: RouteViolationType
    segment_index: int | None = None
    obstacle_id: str | None = None
    required_m: float | None = None
    actual_m: float | None = None
    waypoint_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"type": self.type.value}
        for name in (
            "segment_index",
            "obstacle_id",
            "required_m",
            "actual_m",
            "waypoint_id",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class RouteCritique:
    status: RouteCriticStatus
    route_id: str
    violations: tuple[RouteViolation, ...]
    minimum_clearance_m: float | None
    route_length_m: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "route_id": self.route_id,
            "violations": [item.to_dict() for item in self.violations],
            "minimum_clearance_m": self.minimum_clearance_m,
            "route_length_m": self.route_length_m,
        }


@dataclass(frozen=True, slots=True)
class RouteValidationContext:
    resolver: SpatialResolver
    obstacles: ObstacleRegistry
    scene_min_xyz_m: tuple[float, float, float]
    scene_max_xyz_m: tuple[float, float, float]
    route_start_world_m: tuple[float, float, float]
    original_goal_world_m: tuple[float, float, float]
    constraints: RouteConstraints = RouteConstraints()

    def __post_init__(self) -> None:
        if not isinstance(self.resolver, SpatialResolver):
            raise TypeError("resolver must be a SpatialResolver")
        if not isinstance(self.obstacles, ObstacleRegistry):
            raise TypeError("obstacles must be an ObstacleRegistry")
        if not isinstance(self.constraints, RouteConstraints):
            raise TypeError("constraints must be RouteConstraints")
        for name in (
            "scene_min_xyz_m",
            "scene_max_xyz_m",
            "route_start_world_m",
            "original_goal_world_m",
        ):
            value = tuple(getattr(self, name))
            if len(value) != 3:
                raise ValueError(f"{name} must contain three coordinates")
            object.__setattr__(self, name, tuple(float(item) for item in value))
        if any(a >= b for a, b in zip(self.scene_min_xyz_m, self.scene_max_xyz_m)):
            raise ValueError("scene bounds must be strictly ordered")


class RouteCritic:
    """Return violations only; never clamp, reroute, or suggest waypoints."""

    def __init__(self, mode: RouteValidationMode | str) -> None:
        try:
            self._mode = mode if isinstance(mode, RouteValidationMode) else RouteValidationMode(mode)
        except (TypeError, ValueError):
            raise ValueError("mode must be open_sim, critic_sim, or strict") from None

    @property
    def mode(self) -> RouteValidationMode:
        return self._mode

    def evaluate(
        self,
        route: RouteDraft,
        context: RouteValidationContext,
    ) -> RouteCritique:
        if not isinstance(route, RouteDraft):
            raise TypeError("route must be a RouteDraft")
        if not isinstance(context, RouteValidationContext):
            raise TypeError("context must be a RouteValidationContext")
        try:
            world = tuple(
                context.resolver.resolve_point(route.frame, item.xyz_m)
                for item in route.waypoints
            )
        except SpatialResolutionError:
            return RouteCritique(
                RouteCriticStatus.REVISE,
                route.route_id,
                (RouteViolation(RouteViolationType.UNRESOLVED_FRAME),),
                None,
                None,
            )

        points = (context.route_start_world_m,) + world
        route_length = sum(dist(left, right) for left, right in zip(points, points[1:]))
        # open_sim measures the unmodified proposal and deliberately defers all
        # geometric safety outcomes to the runtime collision monitor.
        if self._mode is RouteValidationMode.OPEN_SIM:
            return RouteCritique(
                RouteCriticStatus.ACCEPT,
                route.route_id,
                (),
                self._minimum_clearance(points, context.obstacles),
                route_length,
            )

        violations: list[RouteViolation] = []
        constraints = context.constraints
        if len(route.waypoints) > constraints.max_waypoints:
            violations.append(
                RouteViolation(
                    RouteViolationType.TOO_MANY_WAYPOINTS,
                    required_m=float(constraints.max_waypoints),
                    actual_m=float(len(route.waypoints)),
                )
            )
        for waypoint, point in zip(route.waypoints, world):
            if not all(
                low <= value <= high
                for low, value, high in zip(
                    context.scene_min_xyz_m,
                    point,
                    context.scene_max_xyz_m,
                )
            ):
                kind = (
                    RouteViolationType.ALTITUDE_OUT_OF_BOUNDS
                    if not context.scene_min_xyz_m[2] <= point[2] <= context.scene_max_xyz_m[2]
                    else RouteViolationType.OUTSIDE_SCENE
                )
                violations.append(RouteViolation(kind, waypoint_id=waypoint.waypoint_id))
        for index, (start, end) in enumerate(zip(points, points[1:])):
            length = dist(start, end)
            if length > constraints.max_segment_length_m:
                violations.append(
                    RouteViolation(
                        RouteViolationType.SEGMENT_TOO_LONG,
                        segment_index=index,
                        required_m=constraints.max_segment_length_m,
                        actual_m=length,
                    )
                )
            for spec in context.obstacles.collidable_specs:
                aabb = spec.aabb
                if aabb.segment_intersection_fraction(start, end) is not None:
                    violations.append(
                        RouteViolation(
                            RouteViolationType.PATH_INTERSECTS_OBSTACLE,
                            segment_index=index,
                            obstacle_id=spec.obstacle_id,
                            required_m=constraints.minimum_clearance_m,
                            actual_m=0.0,
                        )
                    )
                    continue
                clearance = _segment_aabb_distance(start, end, aabb)
                if clearance < constraints.minimum_clearance_m:
                    violations.append(
                        RouteViolation(
                            RouteViolationType.INSUFFICIENT_CLEARANCE,
                            segment_index=index,
                            obstacle_id=spec.obstacle_id,
                            required_m=constraints.minimum_clearance_m,
                            actual_m=clearance,
                        )
                    )
        if route_length > constraints.max_detour_distance_m:
            violations.append(
                RouteViolation(
                    RouteViolationType.ROUTE_TOO_LONG,
                    required_m=constraints.max_detour_distance_m,
                    actual_m=route_length,
                )
            )
        rejoin_distance = dist(world[-1], context.original_goal_world_m)
        if constraints.must_rejoin_original_goal and rejoin_distance > constraints.rejoin_tolerance_m:
            violations.append(
                RouteViolation(
                    RouteViolationType.DOES_NOT_REJOIN_GOAL,
                    required_m=constraints.rejoin_tolerance_m,
                    actual_m=rejoin_distance,
                    waypoint_id=route.waypoints[-1].waypoint_id,
                )
            )
        minimum = self._minimum_clearance(points, context.obstacles)
        return RouteCritique(
            RouteCriticStatus.ACCEPT if not violations else RouteCriticStatus.REVISE,
            route.route_id,
            tuple(violations),
            minimum,
            route_length,
        )

    @staticmethod
    def _minimum_clearance(
        points: tuple[tuple[float, float, float], ...],
        obstacles: ObstacleRegistry,
    ) -> float | None:
        values = [
            _segment_aabb_distance(start, end, spec.aabb)
            for start, end in zip(points, points[1:])
            for spec in obstacles.collidable_specs
        ]
        return None if not values else min(values)


def _point_aabb_distance(point: tuple[float, float, float], box: ObstacleAABB) -> float:
    squared = 0.0
    for value, low, high in zip(point, box.min_xyz_m, box.max_xyz_m):
        delta = low - value if value < low else value - high if value > high else 0.0
        squared += delta * delta
    return sqrt(squared)


def _segment_aabb_distance(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    box: ObstacleAABB,
) -> float:
    if box.segment_intersection_fraction(start, end) is not None:
        return 0.0
    # Squared point-to-box distance along a segment is convex. A bounded
    # ternary search is deterministic and sufficiently conservative for this
    # ideal simulator critic; it never creates an alternate route.
    left, right = 0.0, 1.0
    direction = tuple(b - a for a, b in zip(start, end))
    for _ in range(72):
        first = left + (right - left) / 3.0
        second = right - (right - left) / 3.0
        p1 = tuple(start[i] + first * direction[i] for i in range(3))
        p2 = tuple(start[i] + second * direction[i] for i in range(3))
        if _point_aabb_distance(p1, box) <= _point_aabb_distance(p2, box):
            right = second
        else:
            left = first
    t = (left + right) / 2.0
    point = tuple(start[i] + t * direction[i] for i in range(3))
    return _point_aabb_distance(point, box)


__all__ = [
    "RouteCritic",
    "RouteCriticStatus",
    "RouteCritique",
    "RouteValidationContext",
    "RouteValidationMode",
    "RouteViolation",
    "RouteViolationType",
]
