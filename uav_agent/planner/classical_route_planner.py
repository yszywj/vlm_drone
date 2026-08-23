"""Deterministic classical obstacle-route baseline.

This module is deliberately independent from Qwen and Isaac Sim.  It plans on
the explicitly supplied, camera-grounded ``UAV_HOLD_FLU`` AABBs, then runs the
result through the existing STRICT :class:`RouteCritic`.  The validation
context may contain additional trusted safety geometry; that geometry is used
only by the final critic and never as an implicit source of visibility-graph
nodes.

The planner owns every waypoint it creates.  No waypoint is clamped or
rewritten after construction, and failure is represented by the typed
``ClassicalNoFeasibleRoute`` result rather than a hidden fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from math import ceil, dist, isfinite
from numbers import Real
from typing import TypeAlias

from common.ids import validate_routing_id
from common.obstacle_types import ObstacleAABB
from planner.obstacle_revision import GroundedObstacleGeometry
from planner.route_critic import (
    RouteCritic,
    RouteCriticStatus,
    RouteCritique,
    RouteValidationContext,
    RouteValidationMode,
)
from planner.route_types import RouteDraft, RouteWaypoint
from planner.spatial import CoordinateFrame, PointTarget
from planner.spatial_resolver import SpatialResolutionError


_GEOMETRY_EPSILON = 1e-9
_TRUSTED_POSE_TOLERANCE_M = 1e-6
_MAX_GROUNDED_OBSTACLES = 32


class ClassicalRouteFailureCode(str, Enum):
    """Stable, experiment-facing reasons for an explicit no-route result."""

    NO_GROUNDED_OBSTACLES = "CLASSICAL_NO_GROUNDED_OBSTACLES"
    UNKNOWN_GROUNDED_OBSTACLE = "CLASSICAL_UNKNOWN_GROUNDED_OBSTACLE"
    HOLD_FRAME_UNAVAILABLE = "CLASSICAL_HOLD_FRAME_UNAVAILABLE"
    HOLD_START_MISMATCH = "CLASSICAL_HOLD_START_MISMATCH"
    REJOIN_GOAL_MISMATCH = "CLASSICAL_REJOIN_GOAL_MISMATCH"
    DEGENERATE_REJOIN_TARGET = "CLASSICAL_DEGENERATE_REJOIN_TARGET"
    START_OUTSIDE_SCENE = "CLASSICAL_START_OUTSIDE_SCENE"
    GOAL_OUTSIDE_SCENE = "CLASSICAL_GOAL_OUTSIDE_SCENE"
    START_WITHIN_CLEARANCE = "CLASSICAL_START_WITHIN_CLEARANCE"
    GOAL_WITHIN_CLEARANCE = "CLASSICAL_GOAL_WITHIN_CLEARANCE"
    GRAPH_DISCONNECTED = "CLASSICAL_GRAPH_DISCONNECTED"
    WAYPOINT_BUDGET_EXCEEDED = "CLASSICAL_WAYPOINT_BUDGET_EXCEEDED"
    STRICT_CRITIC_REJECTED = "CLASSICAL_STRICT_CRITIC_REJECTED"


@dataclass(frozen=True, slots=True)
class ClassicalRouteSolution:
    """One unchanged typed route accepted by the mandatory STRICT critic."""

    route: RouteDraft
    critique: RouteCritique
    grounded_obstacle_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.route, RouteDraft):
            raise TypeError("route must be a RouteDraft")
        if not isinstance(self.critique, RouteCritique):
            raise TypeError("critique must be a RouteCritique")
        if (
            self.critique.route_id != self.route.route_id
            or self.critique.status is not RouteCriticStatus.ACCEPT
        ):
            raise ValueError("solution requires a matching ACCEPT critique")
        _validate_obstacle_ids(self.grounded_obstacle_ids)


@dataclass(frozen=True, slots=True)
class ClassicalNoFeasibleRoute:
    """Explicit terminal result; an optional rejected candidate is audit-only."""

    reason_code: ClassicalRouteFailureCode
    grounded_obstacle_ids: tuple[str, ...]
    candidate_route: RouteDraft | None = None
    critique: RouteCritique | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason_code, ClassicalRouteFailureCode):
            raise TypeError("reason_code must be ClassicalRouteFailureCode")
        _validate_obstacle_ids(self.grounded_obstacle_ids, allow_empty=True)
        if self.candidate_route is not None and not isinstance(
            self.candidate_route, RouteDraft
        ):
            raise TypeError("candidate_route must be a RouteDraft or None")
        if self.critique is not None:
            if not isinstance(self.critique, RouteCritique):
                raise TypeError("critique must be a RouteCritique or None")
            if self.candidate_route is None:
                raise ValueError("a critique requires its rejected candidate_route")
            if self.critique.route_id != self.candidate_route.route_id:
                raise ValueError("critique and candidate route_id must match")
            if self.critique.status is not RouteCriticStatus.REVISE:
                raise ValueError("no-feasible critique must have REVISE status")


ClassicalRoutePlanningResult: TypeAlias = (
    ClassicalRouteSolution | ClassicalNoFeasibleRoute
)


class ClassicalRoutePlanner:
    """Inflated-AABB visibility graph plus deterministic Dijkstra.

    Candidate geometry is generated only from ``grounded_obstacles``.  Scene
    bounds and the complete ``RouteValidationContext`` remain authoritative at
    the safety boundary.  This class has no model API and is never a Qwen
    error-recovery path.
    """

    def __init__(self, *, clearance_epsilon_m: float = 1e-3) -> None:
        if (
            isinstance(clearance_epsilon_m, bool)
            or not isinstance(clearance_epsilon_m, Real)
        ):
            raise TypeError("clearance_epsilon_m must be a finite number")
        normalized = float(clearance_epsilon_m)
        if not isfinite(normalized) or not 0.0 < normalized <= 0.25:
            raise ValueError("clearance_epsilon_m must be within (0, 0.25]")
        self._clearance_epsilon_m = normalized
        self._critic = RouteCritic(RouteValidationMode.STRICT)

    @property
    def validation_mode(self) -> RouteValidationMode:
        return RouteValidationMode.STRICT

    def plan(
        self,
        *,
        route_id: str,
        rejoin_target: PointTarget,
        grounded_obstacles: tuple[GroundedObstacleGeometry, ...],
        validation_context: RouteValidationContext,
    ) -> ClassicalRoutePlanningResult:
        """Return a STRICT-accepted route or an explicit no-feasible result."""

        route_id = validate_routing_id(route_id, "route_id")
        if not isinstance(rejoin_target, PointTarget):
            raise TypeError("rejoin_target must be a PointTarget")
        if rejoin_target.frame is not CoordinateFrame.UAV_HOLD_FLU:
            raise ValueError("rejoin_target must use UAV_HOLD_FLU")
        if not isinstance(grounded_obstacles, tuple) or any(
            not isinstance(item, GroundedObstacleGeometry)
            for item in grounded_obstacles
        ):
            raise TypeError(
                "grounded_obstacles must be a tuple of GroundedObstacleGeometry"
            )
        if not isinstance(validation_context, RouteValidationContext):
            raise TypeError("validation_context must be RouteValidationContext")

        ordered = tuple(sorted(grounded_obstacles, key=lambda item: item.obstacle_id))
        obstacle_ids = tuple(item.obstacle_id for item in ordered)
        _validate_obstacle_ids(obstacle_ids, allow_empty=True)
        if not ordered:
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.NO_GROUNDED_OBSTACLES,
                obstacle_ids,
            )
        if any(item.frame is not CoordinateFrame.UAV_HOLD_FLU for item in ordered):
            # GroundedObstacleGeometry normally enforces this at construction;
            # retain the check here as an explicit planner trust boundary.
            raise ValueError("all grounded obstacles must use UAV_HOLD_FLU")
        for obstacle_id in obstacle_ids:
            if obstacle_id not in validation_context.obstacles:
                return ClassicalNoFeasibleRoute(
                    ClassicalRouteFailureCode.UNKNOWN_GROUNDED_OBSTACLE,
                    obstacle_ids,
                )

        start = (0.0, 0.0, 0.0)
        goal = rejoin_target.xyz_m
        if dist(start, goal) <= _GEOMETRY_EPSILON:
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.DEGENERATE_REJOIN_TARGET,
                obstacle_ids,
            )
        try:
            start_world = validation_context.resolver.resolve_point(
                CoordinateFrame.UAV_HOLD_FLU,
                start,
            )
            goal_world = validation_context.resolver.resolve_point(
                CoordinateFrame.UAV_HOLD_FLU,
                goal,
            )
        except SpatialResolutionError:
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.HOLD_FRAME_UNAVAILABLE,
                obstacle_ids,
            )
        if dist(start_world, validation_context.route_start_world_m) > _TRUSTED_POSE_TOLERANCE_M:
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.HOLD_START_MISMATCH,
                obstacle_ids,
            )
        if dist(goal_world, validation_context.original_goal_world_m) > _TRUSTED_POSE_TOLERANCE_M:
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.REJOIN_GOAL_MISMATCH,
                obstacle_ids,
            )
        if not _within_scene(start_world, validation_context):
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.START_OUTSIDE_SCENE,
                obstacle_ids,
            )
        if not _within_scene(goal_world, validation_context):
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.GOAL_OUTSIDE_SCENE,
                obstacle_ids,
            )

        expansion = (
            validation_context.constraints.minimum_clearance_m
            + self._clearance_epsilon_m
        )
        inflated = tuple(
            ObstacleAABB(
                item.relative_aabb_min_m,
                item.relative_aabb_max_m,
            ).expanded((expansion, expansion, expansion))
            for item in ordered
        )
        if any(box.contains(start) for box in inflated):
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.START_WITHIN_CLEARANCE,
                obstacle_ids,
            )
        if any(box.contains(goal) for box in inflated):
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.GOAL_WITHIN_CLEARANCE,
                obstacle_ids,
            )

        nodes = _visibility_nodes(
            start=start,
            goal=goal,
            inflated=inflated,
            context=validation_context,
        )
        path_indices = _shortest_visibility_path(nodes, inflated)
        if path_indices is None:
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.GRAPH_DISCONNECTED,
                obstacle_ids,
            )
        local_path = tuple(nodes[index] for index in path_indices)
        waypoints = _subdivide_path(
            local_path,
            maximum_segment_m=validation_context.constraints.max_segment_length_m,
        )
        if len(waypoints) > validation_context.constraints.max_waypoints:
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.WAYPOINT_BUDGET_EXCEEDED,
                obstacle_ids,
            )
        route = RouteDraft(
            route_id,
            CoordinateFrame.UAV_HOLD_FLU,
            tuple(
                RouteWaypoint(f"wp_classical_{index:02d}", point)
                for index, point in enumerate(waypoints, 1)
            ),
        )
        critique = self._critic.evaluate(route, validation_context)
        if critique.status is not RouteCriticStatus.ACCEPT:
            return ClassicalNoFeasibleRoute(
                ClassicalRouteFailureCode.STRICT_CRITIC_REJECTED,
                obstacle_ids,
                candidate_route=route,
                critique=critique,
            )
        return ClassicalRouteSolution(route, critique, obstacle_ids)


def _validate_obstacle_ids(
    values: tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError("grounded_obstacle_ids must be a tuple")
    if not values and not allow_empty:
        raise ValueError("grounded_obstacle_ids must not be empty")
    if len(values) > _MAX_GROUNDED_OBSTACLES:
        raise ValueError(
            f"grounded_obstacle_ids must contain at most {_MAX_GROUNDED_OBSTACLES} values"
        )
    normalized = tuple(
        validate_routing_id(item, f"grounded_obstacle_ids[{index}]")
        for index, item in enumerate(values)
    )
    if normalized != values:
        raise ValueError("grounded_obstacle_ids must already be normalized")
    if len(values) != len(set(values)):
        raise ValueError("grounded_obstacle_ids must be unique")


def _within_scene(
    point: tuple[float, float, float],
    context: RouteValidationContext,
) -> bool:
    return all(
        low <= value <= high
        for low, value, high in zip(
            context.scene_min_xyz_m,
            point,
            context.scene_max_xyz_m,
        )
    )


def _visibility_nodes(
    *,
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    inflated: tuple[ObstacleAABB, ...],
    context: RouteValidationContext,
) -> tuple[tuple[float, float, float], ...]:
    # Start and goal retain fixed indices 0 and 1.  Stable obstacle ordering
    # plus lexicographic corner order makes equal-cost Dijkstra ties repeatable.
    candidates = [start, goal]
    for box in inflated:
        candidates.extend(sorted(box.corners_xyz_m()))
    result: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for index, point in enumerate(candidates):
        if point in seen:
            continue
        try:
            world = context.resolver.resolve_point(
                CoordinateFrame.UAV_HOLD_FLU,
                point,
            )
        except SpatialResolutionError:
            continue
        if not _within_scene(world, context):
            continue
        # A corner may lie on its source AABB, but never inside another
        # inflated obstacle.  Boundary-only contact remains a visibility node.
        if index >= 2 and any(box.contains(point, strict=True) for box in inflated):
            continue
        seen.add(point)
        result.append(point)
    # Start and goal were prevalidated and must remain indices 0 and 1.
    if len(result) < 2 or result[0] != start or result[1] != goal:
        return (start, goal)
    return tuple(result)


def _shortest_visibility_path(
    nodes: tuple[tuple[float, float, float], ...],
    inflated: tuple[ObstacleAABB, ...],
) -> tuple[int, ...] | None:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if not _segment_visible(nodes[left], nodes[right], inflated):
                continue
            length = dist(nodes[left], nodes[right])
            if length <= _GEOMETRY_EPSILON:
                continue
            adjacency[left].append((right, length))
            adjacency[right].append((left, length))
    for edges in adjacency:
        edges.sort(key=lambda item: item[0])

    # The full node-index path is part of the heap key.  Equal-length choices
    # are therefore resolved lexicographically and never depend on hash order.
    best: dict[int, tuple[float, int, tuple[int, ...]]] = {
        0: (0.0, 0, (0,))
    }
    heap: list[tuple[float, int, tuple[int, ...], int]] = [
        (0.0, 0, (0,), 0)
    ]
    while heap:
        cost, hops, path, node = heappop(heap)
        if best.get(node) != (cost, hops, path):
            continue
        if node == 1:
            return path
        for neighbor, edge_cost in adjacency[node]:
            if neighbor in path:
                continue
            candidate = (cost + edge_cost, hops + 1, (*path, neighbor))
            incumbent = best.get(neighbor)
            if incumbent is None or candidate < incumbent:
                best[neighbor] = candidate
                heappush(heap, (*candidate, neighbor))
    return None


def _segment_visible(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    inflated: tuple[ObstacleAABB, ...],
) -> bool:
    for box in inflated:
        # Shrinking by a numerical epsilon models intersection with the open
        # AABB interior.  A visibility edge may touch a vertex/face of the
        # clearance envelope, but may not enter it.
        interior = ObstacleAABB(
            tuple(value + _GEOMETRY_EPSILON for value in box.min_xyz_m),
            tuple(value - _GEOMETRY_EPSILON for value in box.max_xyz_m),
        )
        if interior.segment_intersection_fraction(start, end) is not None:
            return False
    return True


def _subdivide_path(
    path: tuple[tuple[float, float, float], ...],
    *,
    maximum_segment_m: float,
) -> tuple[tuple[float, float, float], ...]:
    if len(path) < 2:
        raise ValueError("path must contain start and goal")
    waypoints: list[tuple[float, float, float]] = []
    for start, end in zip(path, path[1:]):
        length = dist(start, end)
        if length <= _GEOMETRY_EPSILON:
            continue
        divisions = max(
            1,
            ceil(length / (maximum_segment_m * (1.0 - 1e-12))),
        )
        for step in range(1, divisions + 1):
            fraction = step / divisions
            waypoints.append(
                tuple(
                    left + fraction * (right - left)
                    for left, right in zip(start, end)
                )
            )
    # RouteDraft requires at least two explicit points.  A clear direct edge
    # normally has only the goal, so add a planner-owned midpoint rather than
    # duplicating the implicit route start.
    if len(waypoints) == 1:
        midpoint = tuple(
            (left + right) / 2.0 for left, right in zip(path[0], path[-1])
        )
        waypoints.insert(0, midpoint)
    if len(waypoints) < 2:
        raise ValueError("start and rejoin target must be distinct")
    return tuple(waypoints)


__all__ = [
    "ClassicalNoFeasibleRoute",
    "ClassicalRouteFailureCode",
    "ClassicalRoutePlanner",
    "ClassicalRoutePlanningResult",
    "ClassicalRouteSolution",
]
