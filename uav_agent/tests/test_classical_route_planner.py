from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from agents.classical_obstacle_revision_coordinator import (
    ClassicalObstacleRevisionCoordinator,
)
from common.obstacle_types import FlightCorridor, ObstacleSpec
from env.obstacle_registry import ObstacleRegistry
from planner.classical_route_planner import (
    ClassicalNoFeasibleRoute,
    ClassicalRouteFailureCode,
    ClassicalRoutePlanner,
    ClassicalRouteSolution,
)
from planner.obstacle_revision import GroundedObstacleGeometry
from planner.route_critic import (
    RouteCriticStatus,
    RouteValidationContext,
    RouteValidationMode,
)
from planner.route_types import RouteConstraints
from planner.spatial import CoordinateFrame, PointTarget
from planner.spatial_resolver import FramePose, SpatialResolver
from runtime.collision_supervisor import CollisionSupervisor
from runtime.hazard_fusion import HazardFusion
from runtime.obstacle_route_runtime import ObstacleRouteReplanRuntime
from runtime.route_registry import RouteRegistry
from runtime.safety_supervisor import SafetyAction, SafetyDecision
from skills.plan import TaskPlan, TaskStep
from skills.types import SkillName


def _context(
    specs: tuple[ObstacleSpec, ...],
    *,
    scene_min: tuple[float, float, float] = (-20.0, -20.0, 0.0),
    scene_max: tuple[float, float, float] = (20.0, 20.0, 20.0),
    start: tuple[float, float, float] = (0.0, 0.0, 5.0),
    goal: tuple[float, float, float] = (10.0, 0.0, 5.0),
    constraints: RouteConstraints = RouteConstraints(
        max_waypoints=8,
        minimum_clearance_m=1.0,
    ),
) -> RouteValidationContext:
    return RouteValidationContext(
        resolver=SpatialResolver(
            home_pose=FramePose((0.0, 0.0, 0.0)),
            uav_start_pose=FramePose((0.0, 0.0, 0.0)),
            uav_hold_pose=FramePose(start),
        ),
        obstacles=ObstacleRegistry(specs),
        scene_min_xyz_m=scene_min,
        scene_max_xyz_m=scene_max,
        route_start_world_m=start,
        original_goal_world_m=goal,
        constraints=constraints,
    )


def _grounded(
    obstacle_id: str,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> GroundedObstacleGeometry:
    return GroundedObstacleGeometry(
        obstacle_id,
        CoordinateFrame.UAV_HOLD_FLU,
        minimum,
        maximum,
    )


class ClassicalRoutePlannerTest(unittest.TestCase):
    def test_visibility_graph_detour_is_deterministic_and_strict_accepted(self) -> None:
        obstacle = ObstacleSpec(
            "box_red",
            (5.0, 0.0, 5.0),
            (2.0, 2.0, 2.0),
            (1.0, 0.0, 0.0),
        )
        context = _context((obstacle,))
        geometry = _grounded("box_red", (4.0, -1.0, -1.0), (6.0, 1.0, 1.0))
        planner = ClassicalRoutePlanner()

        first = planner.plan(
            route_id="route_classical_1",
            rejoin_target=PointTarget(
                CoordinateFrame.UAV_HOLD_FLU,
                (10.0, 0.0, 0.0),
            ),
            grounded_obstacles=(geometry,),
            validation_context=context,
        )
        second = planner.plan(
            route_id="route_classical_1",
            rejoin_target=PointTarget(
                CoordinateFrame.UAV_HOLD_FLU,
                (10.0, 0.0, 0.0),
            ),
            grounded_obstacles=(geometry,),
            validation_context=context,
        )

        self.assertIsInstance(first, ClassicalRouteSolution)
        self.assertIsInstance(second, ClassicalRouteSolution)
        assert isinstance(first, ClassicalRouteSolution)
        assert isinstance(second, ClassicalRouteSolution)
        self.assertEqual(first.route.to_dict(), second.route.to_dict())
        self.assertEqual(first.route.frame, CoordinateFrame.UAV_HOLD_FLU)
        self.assertEqual(first.route.waypoints[-1].xyz_m, (10.0, 0.0, 0.0))
        self.assertTrue(
            any(
                abs(point.xyz_m[1]) > 1e-6 or abs(point.xyz_m[2]) > 1e-6
                for point in first.route.waypoints[:-1]
            )
        )
        self.assertEqual(first.critique.status, RouteCriticStatus.ACCEPT)
        self.assertEqual(planner.validation_mode, RouteValidationMode.STRICT)

    def test_final_strict_critic_sees_context_geometry_not_used_as_graph_nodes(self) -> None:
        grounded_spec = ObstacleSpec(
            "box_visible",
            (5.0, 8.0, 5.0),
            (2.0, 2.0, 2.0),
            (1.0, 0.0, 0.0),
        )
        ungrounded_safety_spec = ObstacleSpec(
            "box_safety_only",
            (5.0, 0.0, 5.0),
            (2.0, 2.0, 2.0),
            (0.0, 0.0, 1.0),
        )
        result = ClassicalRoutePlanner().plan(
            route_id="route_strict_reject",
            rejoin_target=PointTarget(
                CoordinateFrame.UAV_HOLD_FLU,
                (10.0, 0.0, 0.0),
            ),
            grounded_obstacles=(
                _grounded(
                    "box_visible",
                    (4.0, 7.0, -1.0),
                    (6.0, 9.0, 1.0),
                ),
            ),
            validation_context=_context(
                (grounded_spec, ungrounded_safety_spec)
            ),
        )

        self.assertIsInstance(result, ClassicalNoFeasibleRoute)
        assert isinstance(result, ClassicalNoFeasibleRoute)
        self.assertEqual(
            result.reason_code,
            ClassicalRouteFailureCode.STRICT_CRITIC_REJECTED,
        )
        self.assertIsNotNone(result.candidate_route)
        self.assertIsNotNone(result.critique)
        assert result.critique is not None
        self.assertTrue(
            any(
                violation.obstacle_id == "box_safety_only"
                for violation in result.critique.violations
            )
        )

    def test_scene_spanning_wall_returns_explicit_graph_disconnected(self) -> None:
        wall = ObstacleSpec(
            "wall",
            (5.0, 0.0, 2.0),
            (2.0, 4.0, 4.0),
            (0.5, 0.5, 0.5),
        )
        result = ClassicalRoutePlanner().plan(
            route_id="route_none",
            rejoin_target=PointTarget(
                CoordinateFrame.UAV_HOLD_FLU,
                (10.0, 0.0, 0.0),
            ),
            grounded_obstacles=(
                _grounded("wall", (4.0, -2.0, -2.0), (6.0, 2.0, 2.0)),
            ),
            validation_context=_context(
                (wall,),
                scene_min=(-1.0, -2.0, 0.0),
                scene_max=(11.0, 2.0, 4.0),
                start=(0.0, 0.0, 2.0),
                goal=(10.0, 0.0, 2.0),
                constraints=RouteConstraints(
                    max_waypoints=8,
                    minimum_clearance_m=0.25,
                ),
            ),
        )

        self.assertIsInstance(result, ClassicalNoFeasibleRoute)
        assert isinstance(result, ClassicalNoFeasibleRoute)
        self.assertEqual(
            result.reason_code,
            ClassicalRouteFailureCode.GRAPH_DISCONNECTED,
        )
        self.assertIsNone(result.candidate_route)

    def test_segment_subdivision_obeys_waypoint_budget_or_fails_explicitly(self) -> None:
        off_path = ObstacleSpec(
            "box_visible",
            (5.0, 8.0, 5.0),
            (2.0, 2.0, 2.0),
            (1.0, 0.0, 0.0),
        )
        result = ClassicalRoutePlanner().plan(
            route_id="route_budget",
            rejoin_target=PointTarget(
                CoordinateFrame.UAV_HOLD_FLU,
                (10.0, 0.0, 0.0),
            ),
            grounded_obstacles=(
                _grounded(
                    "box_visible",
                    (4.0, 7.0, -1.0),
                    (6.0, 9.0, 1.0),
                ),
            ),
            validation_context=_context(
                (off_path,),
                constraints=RouteConstraints(
                    max_waypoints=2,
                    max_segment_length_m=3.0,
                    minimum_clearance_m=1.0,
                ),
            ),
        )

        self.assertIsInstance(result, ClassicalNoFeasibleRoute)
        assert isinstance(result, ClassicalNoFeasibleRoute)
        self.assertEqual(
            result.reason_code,
            ClassicalRouteFailureCode.WAYPOINT_BUDGET_EXCEEDED,
        )


class _Manager:
    def __init__(self, plan: TaskPlan) -> None:
        self.is_supervisory_paused = True
        self.active_planned_step_id = "goto_search"
        self.task_plan = plan
        self.replacement: TaskPlan | None = None
        self.cancel_count = 0

    def replace_interrupted_step_and_suffix(self, plan: TaskPlan) -> None:
        self.replacement = plan
        self.task_plan = plan
        self.is_supervisory_paused = False

    def cancel_task(self) -> None:
        self.cancel_count += 1


def _plan() -> TaskPlan:
    return TaskPlan(
        (
            TaskStep("takeoff", SkillName.TAKEOFF, {"altitude": 10.0}),
            TaskStep(
                "goto_search",
                SkillName.GOTO,
                {
                    "position": (10.0, 0.0, 10.0),
                    "tolerance": 0.75,
                    "timeout": 120.0,
                },
            ),
            TaskStep(
                "land_home",
                SkillName.LAND,
                {"ground_altitude": 0.0, "timeout": 60.0},
            ),
        ),
        mission_id="mission_classical",
        uav_id="uav_1",
        plan_version=1,
    )


def _grounded_supervisor() -> CollisionSupervisor:
    report = HazardFusion().low_level_report(
        hazard_detected=True,
        geometry_grounded=True,
        obstacle_ids=("box_red",),
    )
    fusion = HazardFusion().fuse(
        (report,),
        mission_id="mission_classical",
        uav_id="uav_1",
        plan_version=1,
        timestamp_s=5.0,
    )
    supervisor = CollisionSupervisor()
    supervisor.evaluate(fusion)
    supervisor.mark_hold_established(timestamp_s=5.05)
    supervisor.mark_geometry_grounded()
    return supervisor


class ClassicalCoordinatorRuntimeIntegrationTest(unittest.TestCase):
    def test_runtime_uses_registry_compiler_and_strict_publication_without_worker(self) -> None:
        obstacle = ObstacleSpec(
            "box_red",
            (5.0, 0.0, 10.0),
            (2.0, 2.0, 4.0),
            (1.0, 0.0, 0.0),
        )
        obstacles = ObstacleRegistry((obstacle,))
        registry = RouteRegistry()
        manager = _Manager(_plan())
        supervisor = _grounded_supervisor()
        coordinator = ClassicalObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ClassicalRoutePlanner(),
            route_registry=registry,
            collision_supervisor=supervisor,
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(
                SafetyAction.CONTINUE,
                "safe",
            ),
        )
        runtime = ObstacleRouteReplanRuntime(
            coordinator=coordinator,
            initial_resolver=SpatialResolver(
                home_pose=FramePose((0.0, 0.0, 0.0)),
                uav_start_pose=FramePose((0.0, 0.0, 0.0)),
            ),
            obstacles=obstacles,
            scene_min_xyz_m=(-20.0, -20.0, 0.0),
            scene_max_xyz_m=(20.0, 20.0, 30.0),
            original_instruction="search then return home",
            original_plan_summary=_plan().to_dict(),
        )
        runtime.observe_active_corridor(
            FlightCorridor(
                (0.0, 0.0, 10.0),
                (10.0, 0.0, 10.0),
                0.5,
            ),
            collision_state="CLEAR",
        )

        result = runtime.tick(
            obstacle_snapshot=SimpleNamespace(
                state=SimpleNamespace(value="GEOMETRY_GROUNDED"),
                fusion=SimpleNamespace(obstacle_ids=("box_red",)),
            ),
            manager=manager,
            rgb=np.zeros((8, 12, 3), dtype=np.uint8),
            frame_id="frame_classical",
            timestamp_s=5.2,
            mission_elapsed_s=4.8,
            hold_pose=FramePose((0.0, 0.0, 10.0)),
        )

        self.assertEqual(result.coordinator_state, "ACCEPTED")
        self.assertTrue(result.revision_started)
        self.assertIsNotNone(manager.replacement)
        assert manager.replacement is not None
        self.assertEqual(manager.replacement.plan_version, 2)
        self.assertEqual(manager.replacement.steps[1].skill, SkillName.FOLLOW_ROUTE)
        self.assertEqual(manager.replacement.steps[-1].skill, SkillName.LAND)
        self.assertEqual(manager.cancel_count, 0)
        self.assertEqual(len(registry.records), 1)
        route_record = registry.records[0]
        self.assertEqual(route_record.critique.status, RouteCriticStatus.ACCEPT)
        self.assertEqual(route_record.state.value, "ACCEPTED")
        self.assertEqual(coordinator.records[0].outcome, "ACCEPTED")
        self.assertEqual(
            coordinator.records[0].proposal["avoidance_strategy"]["reason_codes"],
            ["CLASSICAL_VISIBILITY_GRAPH_V1"],
        )
        # Publication resumes the collision state; there is no model worker to
        # poll and no Qwen fallback hidden behind the baseline.
        self.assertEqual(supervisor.state.value, "CLEAR")

    def test_no_visibility_graph_path_is_explicit_and_cancels_without_registry_entry(self) -> None:
        wall = ObstacleSpec(
            "box_red",
            (5.0, 0.0, 10.0),
            (2.0, 40.0, 30.0),
            (1.0, 0.0, 0.0),
        )
        obstacles = ObstacleRegistry((wall,))
        registry = RouteRegistry()
        manager = _Manager(_plan())
        supervisor = _grounded_supervisor()
        coordinator = ClassicalObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ClassicalRoutePlanner(),
            route_registry=registry,
            collision_supervisor=supervisor,
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(
                SafetyAction.CONTINUE,
                "safe",
            ),
        )
        runtime = ObstacleRouteReplanRuntime(
            coordinator=coordinator,
            initial_resolver=SpatialResolver(
                home_pose=FramePose((0.0, 0.0, 0.0)),
                uav_start_pose=FramePose((0.0, 0.0, 0.0)),
            ),
            obstacles=obstacles,
            scene_min_xyz_m=(-20.0, -20.0, 0.0),
            scene_max_xyz_m=(20.0, 20.0, 30.0),
            original_instruction="search then return home",
            original_plan_summary=_plan().to_dict(),
        )
        runtime.observe_active_corridor(
            FlightCorridor(
                (0.0, 0.0, 10.0),
                (10.0, 0.0, 10.0),
                0.5,
            ),
            collision_state="CLEAR",
        )

        result = runtime.tick(
            obstacle_snapshot=SimpleNamespace(
                state=SimpleNamespace(value="GEOMETRY_GROUNDED"),
                fusion=SimpleNamespace(obstacle_ids=("box_red",)),
            ),
            manager=manager,
            rgb=np.zeros((8, 12, 3), dtype=np.uint8),
            frame_id="frame_classical_wall",
            timestamp_s=5.2,
            mission_elapsed_s=4.8,
            hold_pose=FramePose((0.0, 0.0, 10.0)),
        )

        self.assertEqual(result.coordinator_state, "EXHAUSTED")
        self.assertEqual(
            result.error_code,
            ClassicalRouteFailureCode.GRAPH_DISCONNECTED.value,
        )
        self.assertEqual(manager.cancel_count, 1)
        self.assertIsNone(manager.replacement)
        self.assertEqual(registry.records, ())
        self.assertEqual(coordinator.records[0].outcome, "NO_FEASIBLE_ROUTE")
        self.assertEqual(
            coordinator.records[0].error_code,
            ClassicalRouteFailureCode.GRAPH_DISCONNECTED.value,
        )

    def test_safety_preflight_rejection_fails_closed_without_alternate_route(self) -> None:
        obstacle = ObstacleSpec(
            "box_red",
            (5.0, 0.0, 10.0),
            (2.0, 2.0, 4.0),
            (1.0, 0.0, 0.0),
        )
        obstacles = ObstacleRegistry((obstacle,))
        registry = RouteRegistry()
        manager = _Manager(_plan())
        coordinator = ClassicalObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ClassicalRoutePlanner(),
            route_registry=registry,
            collision_supervisor=_grounded_supervisor(),
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(
                SafetyAction.CANCEL_AND_LAND,
                "test rejection",
            ),
        )
        runtime = ObstacleRouteReplanRuntime(
            coordinator=coordinator,
            initial_resolver=SpatialResolver(
                home_pose=FramePose((0.0, 0.0, 0.0)),
                uav_start_pose=FramePose((0.0, 0.0, 0.0)),
            ),
            obstacles=obstacles,
            scene_min_xyz_m=(-20.0, -20.0, 0.0),
            scene_max_xyz_m=(20.0, 20.0, 30.0),
            original_instruction="search then return home",
            original_plan_summary=_plan().to_dict(),
        )
        runtime.observe_active_corridor(
            FlightCorridor(
                (0.0, 0.0, 10.0),
                (10.0, 0.0, 10.0),
                0.5,
            ),
            collision_state="CLEAR",
        )

        result = runtime.tick(
            obstacle_snapshot=SimpleNamespace(
                state=SimpleNamespace(value="GEOMETRY_GROUNDED"),
                fusion=SimpleNamespace(obstacle_ids=("box_red",)),
            ),
            manager=manager,
            rgb=np.zeros((8, 12, 3), dtype=np.uint8),
            frame_id="frame_classical_safety",
            timestamp_s=5.2,
            mission_elapsed_s=4.8,
            hold_pose=FramePose((0.0, 0.0, 10.0)),
        )

        self.assertEqual(result.coordinator_state, "FAILED")
        self.assertEqual(result.error_code, "CLASSICAL_ROUTE_PUBLICATION_FAILED")
        self.assertEqual(manager.cancel_count, 1)
        self.assertIsNone(manager.replacement)
        self.assertEqual(registry.records, ())
        self.assertEqual(coordinator.records[0].outcome, "PUBLICATION_FAILED")


if __name__ == "__main__":
    unittest.main()
