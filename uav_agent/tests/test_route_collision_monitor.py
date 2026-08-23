from __future__ import annotations

import unittest
from types import SimpleNamespace

from common.obstacle_types import ObstacleSpec
from env.obstacle_registry import ObstacleRegistry
from planner.route_critic import RouteCriticStatus, RouteCritique
from planner.route_types import RouteDraft, RouteState, RouteWaypoint
from planner.spatial import CoordinateFrame
from planner.spatial_resolver import FramePose
from runtime.events import EventSeverity, MissionEventType
from runtime.route_collision_monitor import (
    ROUTE_COLLISION_SOURCE,
    RouteCollisionMonitor,
)
from runtime.route_registry import RouteRegistry
from skills.manager import (
    SkillManager,
    TaskStatus,
    create_default_skill_registry,
)
from skills.plan import TaskPlan, TaskStep
from skills.types import SkillName
from tests.test_goto import ManualClock, make_context, make_uav


def _accepted_route(route_id: str = "route_1") -> RouteRegistry:
    route = RouteDraft(
        route_id,
        CoordinateFrame.WORLD_ENU,
        (
            RouteWaypoint("wp_1", (0.0, 0.0, 10.0)),
            RouteWaypoint("wp_2", (10.0, 0.0, 10.0)),
        ),
    )
    registry = RouteRegistry()
    registry.register(
        route,
        frame_snapshot=FramePose((0.0, 0.0, 0.0)),
        raw_proposal={"route_draft": route.to_dict()},
        plan_version=2,
        proposal_timestamp_s=1.0,
    )
    registry.record_critique(
        route_id,
        RouteCritique(RouteCriticStatus.ACCEPT, route_id, (), 10.0, 4.0),
    )
    return registry


def _route_manager(registry: RouteRegistry) -> SkillManager:
    manager = SkillManager(
        make_context(make_uav((0.0, 0.0, 10.0)), ManualClock()),
        registry=create_default_skill_registry(),
        route_registry=registry,
    )
    manager.start_task(
        TaskPlan(
            (
                TaskStep(
                    "follow_route",
                    SkillName.FOLLOW_ROUTE,
                    {"route_ref": "route_1", "tolerance_m": 0.1},
                ),
                TaskStep("land_home", SkillName.LAND, {}),
            ),
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=2,
        )
    )
    return manager


def _obstacle(
    obstacle_id: str,
    center: tuple[float, float, float],
    *,
    collidable: bool = True,
) -> ObstacleSpec:
    return ObstacleSpec(
        obstacle_id,
        center,
        (1.0, 1.0, 2.0),
        (1.0, 0.0, 0.0),
        collidable=collidable,
    )


class _CollisionCounter:
    def __init__(self) -> None:
        self.count = 0

    def record_collision(self) -> None:
        self.count += 1


class RouteCollisionMonitorTest(unittest.TestCase):
    def test_swept_collision_closes_route_and_starts_safe_landing(self) -> None:
        registry = _accepted_route()
        manager = _route_manager(registry)
        events = []
        collisions = []
        counter = _CollisionCounter()
        monitor = RouteCollisionMonitor(
            obstacle_registry=ObstacleRegistry((_obstacle("box_red", (5, 0, 10)),)),
            route_registry=registry,
            skill_manager=manager,
            uav_radius_m=0.5,
            event_sink=events.append,
            collision_sink=lambda collision: (
                collisions.append(collision),
                counter.record_collision(),
            ),
        )

        self.assertEqual(registry.get("route_1").state, RouteState.EXECUTING)
        self.assertIsNone(monitor.observe((0, 0, 10), timestamp_s=1.0))
        collision = monitor.observe((10, 0, 10), timestamp_s=2.0)

        self.assertIsNotNone(collision)
        assert collision is not None
        self.assertEqual(collision.obstacle_id, "box_red")
        self.assertAlmostEqual(collision.segment_fraction, 0.4)
        self.assertEqual(collision.impact_position_world_m, (4.0, 0.0, 10.0))
        self.assertEqual(registry.get("route_1").state, RouteState.COLLIDED)
        self.assertEqual(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.task_status, TaskStatus.RUNNING)
        self.assertEqual(manager.pending_task_result, TaskStatus.CANCELED)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, MissionEventType.ROUTE_COLLISION)
        self.assertEqual(events[0].severity, EventSeverity.CRITICAL)
        self.assertEqual(events[0].payload["source"], ROUTE_COLLISION_SOURCE)
        self.assertEqual(events[0].payload["geometry_source"], "scene_obstacle_registry")
        self.assertEqual(collisions, [collision])
        self.assertEqual(counter.count, 1)

        # LAND is now active and the route is terminal: the same geometry may
        # never produce a second collision or a second logger increment.
        self.assertIsNone(monitor.observe((10, 0, 10), timestamp_s=3.0))
        self.assertEqual(len(monitor.records), 1)
        self.assertEqual(counter.count, 1)

    def test_uav_radius_expands_only_collidable_aabbs(self) -> None:
        registry = _accepted_route()
        manager = _route_manager(registry)
        obstacles = ObstacleRegistry(
            (
                _obstacle("ignored", (3.0, 1.4, 10.0), collidable=False),
                _obstacle("near_miss", (6.0, 1.4, 10.0)),
            )
        )
        point_monitor = RouteCollisionMonitor(
            obstacle_registry=obstacles,
            route_registry=registry,
            skill_manager=manager,
            uav_radius_m=0.0,
        )
        self.assertIsNone(point_monitor.observe((0, 0, 10), timestamp_s=1.0))
        self.assertIsNone(point_monitor.observe((10, 0, 10), timestamp_s=2.0))
        self.assertEqual(registry.get("route_1").state, RouteState.EXECUTING)

        radius_monitor = RouteCollisionMonitor(
            obstacle_registry=obstacles,
            route_registry=registry,
            skill_manager=manager,
            uav_radius_m=1.0,
        )
        self.assertIsNone(radius_monitor.observe((0, 0, 10), timestamp_s=3.0))
        collision = radius_monitor.observe((10, 0, 10), timestamp_s=4.0)
        self.assertIsNotNone(collision)
        assert collision is not None
        self.assertEqual(collision.obstacle_id, "near_miss")

    def test_accepted_completed_and_non_follow_route_states_do_not_report(self) -> None:
        accepted_registry = _accepted_route()
        accepted_plan = TaskPlan(
            (
                TaskStep(
                    "follow_route",
                    SkillName.FOLLOW_ROUTE,
                    {"route_ref": "route_1"},
                ),
                TaskStep("land_home", SkillName.LAND, {}),
            ),
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=2,
        )
        accepted_manager = SimpleNamespace(
            task_status=TaskStatus.RUNNING,
            active_name=SkillName.FOLLOW_ROUTE,
            active_planned_step_id="follow_route",
            task_plan=accepted_plan,
            uav_id="uav_1",
            cancel_task=lambda: None,
        )
        accepted_monitor = RouteCollisionMonitor(
            obstacle_registry=ObstacleRegistry((_obstacle("box_red", (5, 0, 10)),)),
            route_registry=accepted_registry,
            skill_manager=accepted_manager,
        )
        accepted_monitor.observe((0, 0, 10), timestamp_s=0.0)
        self.assertIsNone(
            accepted_monitor.observe((10, 0, 10), timestamp_s=1.0)
        )
        self.assertEqual(
            accepted_registry.get("route_1").state,
            RouteState.ACCEPTED,
        )

        registry = _accepted_route()
        manager = _route_manager(registry)
        monitor = RouteCollisionMonitor(
            obstacle_registry=ObstacleRegistry((_obstacle("box_red", (5, 0, 10)),)),
            route_registry=registry,
            skill_manager=manager,
        )
        monitor.observe((0, 0, 10), timestamp_s=1.0)
        registry.transition("route_1", RouteState.COMPLETED)
        self.assertIsNone(monitor.observe((10, 0, 10), timestamp_s=2.0))
        self.assertEqual(registry.get("route_1").state, RouteState.COMPLETED)
        self.assertEqual(monitor.records, ())

        # Once Manager is canceled for an unrelated reason, LAND rather than
        # FOLLOW_ROUTE is active. Even a stale externally supplied position
        # inside the obstacle must remain silent.
        manager.cancel_task()
        self.assertIsNone(monitor.observe((5, 0, 10), timestamp_s=3.0))
        self.assertEqual(monitor.records, ())

    def test_first_sample_does_not_sweep_from_a_previous_non_route_skill(self) -> None:
        registry = _accepted_route()
        manager = _route_manager(registry)
        monitor = RouteCollisionMonitor(
            obstacle_registry=ObstacleRegistry((_obstacle("box_red", (5, 0, 10)),)),
            route_registry=registry,
            skill_manager=manager,
        )

        # A fresh route starts sampling on the far side of the obstacle. There
        # is no prior FOLLOW_ROUTE sample, so no invented segment is tested.
        self.assertIsNone(monitor.observe((10, 0, 10), timestamp_s=1.0))
        self.assertEqual(registry.get("route_1").state, RouteState.EXECUTING)


if __name__ == "__main__":
    unittest.main()
