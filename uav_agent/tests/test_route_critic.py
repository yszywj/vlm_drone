from __future__ import annotations

import unittest

from env.obstacle_registry import ObstacleRegistry
from perception.obstacle_types import ObstacleSpec
from planner.route_critic import (
    RouteCritic,
    RouteCriticStatus,
    RouteValidationContext,
    RouteValidationMode,
    RouteViolationType,
)
from planner.route_types import RouteConstraints, RouteDraft, RouteWaypoint
from planner.spatial import CoordinateFrame
from planner.spatial_resolver import FramePose, SpatialResolver


def _context(*, clearance: float = 1.0) -> RouteValidationContext:
    resolver = SpatialResolver(
        home_pose=FramePose((0.0, 0.0, 0.0)),
        uav_start_pose=FramePose((0.0, 0.0, 0.0)),
        uav_hold_pose=FramePose((0.0, 0.0, 10.0)),
    )
    obstacles = ObstacleRegistry(
        (
            ObstacleSpec(
                "box_1",
                (5.0, 0.0, 10.0),
                (2.0, 2.0, 4.0),
                (0.7, 0.2, 0.2),
            ),
        )
    )
    return RouteValidationContext(
        resolver,
        obstacles,
        (-20.0, -20.0, 0.0),
        (20.0, 20.0, 30.0),
        (0.0, 0.0, 10.0),
        (10.0, 0.0, 10.0),
        RouteConstraints(minimum_clearance_m=clearance, max_detour_distance_m=40.0),
    )


def _route(*points: tuple[float, float, float]) -> RouteDraft:
    return RouteDraft(
        "route_1",
        CoordinateFrame.UAV_HOLD_FLU,
        tuple(RouteWaypoint(f"wp_{index}", point) for index, point in enumerate(points, 1)),
    )


class RouteCriticTest(unittest.TestCase):
    def test_safe_left_bypass_is_accepted_unchanged(self) -> None:
        route = _route((2, 3, 0), (8, 3, 0), (10, 0, 0))
        result = RouteCritic("critic_sim").evaluate(route, _context())
        self.assertEqual(result.status, RouteCriticStatus.ACCEPT)
        self.assertEqual(route.waypoints[0].xyz_m, (2.0, 3.0, 0.0))

    def test_intersection_returns_counterexample_without_waypoints(self) -> None:
        route = _route((4, 0, 0), (6, 0, 0), (10, 0, 0))
        result = RouteCritic(RouteValidationMode.CRITIC_SIM).evaluate(route, _context())
        self.assertEqual(result.status, RouteCriticStatus.REVISE)
        self.assertIn(RouteViolationType.PATH_INTERSECTS_OBSTACLE, {v.type for v in result.violations})
        self.assertNotIn("replacement_waypoints", result.to_dict())

    def test_open_sim_does_not_pre_reject_intersection(self) -> None:
        route = _route((4, 0, 0), (6, 0, 0), (10, 0, 0))
        result = RouteCritic("open_sim").evaluate(route, _context())
        self.assertEqual(result.status, RouteCriticStatus.ACCEPT)
        self.assertEqual(result.violations, ())
        self.assertEqual(result.minimum_clearance_m, 0.0)

    def test_out_of_scene_and_missing_rejoin_are_explicit(self) -> None:
        route = _route((0, 3, 0), (30, 3, 0))
        result = RouteCritic("strict").evaluate(route, _context())
        types = {item.type for item in result.violations}
        self.assertIn(RouteViolationType.OUTSIDE_SCENE, types)
        self.assertIn(RouteViolationType.DOES_NOT_REJOIN_GOAL, types)
        self.assertIn(RouteViolationType.SEGMENT_TOO_LONG, types)

    def test_unavailable_hold_frame_is_revise_not_crash(self) -> None:
        resolver = SpatialResolver(
            home_pose=FramePose((0, 0, 0)),
            uav_start_pose=FramePose((0, 0, 0)),
        )
        context = RouteValidationContext(
            resolver,
            ObstacleRegistry(),
            (-20, -20, 0),
            (20, 20, 30),
            (0, 0, 10),
            (10, 0, 10),
        )
        result = RouteCritic("critic_sim").evaluate(_route((1, 1, 0), (10, 0, 0)), context)
        self.assertEqual(result.violations[0].type, RouteViolationType.UNRESOLVED_FRAME)


if __name__ == "__main__":
    unittest.main()
