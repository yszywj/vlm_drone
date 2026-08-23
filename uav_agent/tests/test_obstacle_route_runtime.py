from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from common.obstacle_types import FlightCorridor, ObstacleSpec
from env.obstacle_registry import ObstacleRegistry
from planner.obstacle_revision import (
    ObstacleAwareRevisionPlanner,
    ObstacleReplacementStep,
    ObstacleRouteRevisionDraft,
)
from planner.route_types import (
    AvoidanceStrategy,
    AvoidanceStrategyType,
    RouteDraft,
    RouteWaypoint,
)
from planner.spatial import CoordinateFrame
from planner.spatial_resolver import FramePose, SpatialResolver
from runtime.obstacle_route_runtime import ObstacleRouteReplanRuntime
from skills.plan import RecoveryPolicy, StepOutputRef, TaskPlan, TaskStep
from skills.types import SkillName


class _Coordinator:
    def __init__(self) -> None:
        self.state = "IDLE"
        self.captured: dict[str, object] = {}
        self.records: tuple[object, ...] = ()
        self.reset_preserve: bool | None = None

    def snapshot(self) -> object:
        return SimpleNamespace(
            state=SimpleNamespace(value=self.state),
            request_id=self.captured.get("request_id"),
            accepted_route_id=None,
            error_code=None,
        )

    def begin(self, request: object, **kwargs: object) -> object:
        self.captured = {"request": request, **kwargs}
        self.captured["request_id"] = "request_pending"
        self.state = "AWAITING_MODEL"
        return self.snapshot()

    def tick(self, *, timestamp_s: float) -> object:
        self.captured["tick_timestamp_s"] = timestamp_s
        return self.snapshot()

    def reset(self, *, preserve_records: bool = False) -> None:
        self.reset_preserve = preserve_records
        self.state = "IDLE"


class _Manager:
    def __init__(
        self,
        plan: TaskPlan,
        *,
        paused: bool = True,
        active_step_id: str = "goto_search",
    ) -> None:
        self.is_supervisory_paused = paused
        self.active_planned_step_id = active_step_id
        self.task_plan = plan


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
        mission_id="mission_1",
        uav_id="uav_1",
        plan_version=1,
    )


def _tracking_plan() -> TaskPlan:
    return TaskPlan(
        (
            TaskStep("takeoff", SkillName.TAKEOFF, {"altitude": 10.0}),
            TaskStep(
                "goto_search",
                SkillName.GOTO,
                {"position": (12.0, 0.0, 10.0), "timeout": 120.0},
            ),
            TaskStep(
                "search_1",
                SkillName.SEARCH,
                {
                    "center": (12.0, 0.0, 0.0),
                    "radius": 8.0,
                    "target_description": "moving target",
                    "search_altitude": 10.0,
                    "timeout": 60.0,
                },
            ),
            TaskStep(
                "track_1",
                SkillName.TRACK,
                {
                    "target_id": StepOutputRef("search_1"),
                    "desired_distance": 6.0,
                    "track_duration": 15.0,
                    "timeout": 30.0,
                },
                RecoveryPolicy(SkillName.REACQUIRE, 2, 10.0, 30.0),
            ),
            TaskStep(
                "goto_home",
                SkillName.GOTO,
                {"position": (0.0, 0.0, 10.0), "timeout": 120.0},
            ),
            TaskStep(
                "land_home",
                SkillName.LAND,
                {"ground_altitude": 0.0, "timeout": 60.0},
            ),
        ),
        mission_id="mission_1",
        uav_id="uav_1",
        plan_version=1,
    )


class ObstacleRouteRuntimeTest(unittest.TestCase):
    def _runtime(
        self,
        coordinator: _Coordinator,
        *,
        original_plan: TaskPlan | None = None,
    ) -> ObstacleRouteReplanRuntime:
        plan = _plan() if original_plan is None else original_plan
        return ObstacleRouteReplanRuntime(
            coordinator=coordinator,
            initial_resolver=SpatialResolver(
                home_pose=FramePose((0.0, 0.0, 0.0)),
                uav_start_pose=FramePose((0.0, 0.0, 0.0)),
            ),
            obstacles=ObstacleRegistry(
                (
                    ObstacleSpec(
                        "box_red",
                        (5.0, 0.0, 10.0),
                        (2.0, 2.0, 4.0),
                        (1.0, 0.0, 0.0),
                    ),
                )
            ),
            scene_min_xyz_m=(-20.0, -20.0, 0.0),
            scene_max_xyz_m=(20.0, 20.0, 30.0),
            original_instruction="search then return home",
            original_plan_summary=plan.to_dict(),
        )

    def test_grounded_hold_starts_multimodal_route_without_changing_waypoints(self) -> None:
        coordinator = _Coordinator()
        runtime = self._runtime(coordinator)
        corridor = FlightCorridor((0.0, 0.0, 10.0), (10.0, 0.0, 10.0), 0.5)
        runtime.observe_active_corridor(corridor, collision_state="CLEAR")
        result = runtime.tick(
            obstacle_snapshot=SimpleNamespace(
                state=SimpleNamespace(value="GEOMETRY_GROUNDED"),
                fusion=SimpleNamespace(obstacle_ids=("box_red",)),
            ),
            manager=_Manager(_plan()),
            rgb=np.zeros((8, 12, 3), dtype=np.uint8),
            frame_id="frame_route_1",
            timestamp_s=5.2,
            mission_elapsed_s=4.8,
            hold_pose=FramePose((0.0, 0.0, 10.0), yaw_rad=0.0),
        )
        self.assertTrue(result.revision_started)
        self.assertEqual(result.coordinator_state, "AWAITING_MODEL")
        request = coordinator.captured["request"]
        self.assertEqual(request.replace_from_step_id, "goto_search")
        self.assertEqual(request.grounded_obstacle_geometry.frame.value, "UAV_HOLD_FLU")
        self.assertEqual(
            request.active_corridor_rejoin_target.to_dict(),
            {
                "kind": "POINT",
                "frame": "UAV_HOLD_FLU",
                "xyz_m": [10.0, 0.0, 0.0],
            },
        )
        self.assertEqual(request.frames[0].rgb.shape, (8, 12, 3))
        # The trusted request freezes nested plan summaries.  The concrete
        # planner must still be able to thaw and serialize them before the
        # first HTTP request (the real Isaac path previously raised TypeError
        # here while fake coordinators appeared healthy).
        async_request = ObstacleAwareRevisionPlanner(
            max_image_side_px=64
        ).build_async_request(request, request_id="request_route_runtime")
        self.assertEqual(async_request.plan_version, 1)
        self.assertEqual(
            async_request.options.response_format.name,
            "obstacle_route_revision_v3",
        )
        context = coordinator.captured["validation_context"]
        self.assertEqual(context.original_goal_world_m, (10.0, 0.0, 10.0))

        # The compiler retains these exact model waypoints in RouteRegistry;
        # the executable plan references only their route ID.
        route = RouteDraft(
            request.route_id,
            CoordinateFrame.UAV_HOLD_FLU,
            (
                RouteWaypoint("wp_left", (2.0, 3.0, 0.0)),
                RouteWaypoint("wp_rejoin", (10.0, 0.0, 0.0)),
            ),
        )
        draft = ObstacleRouteRevisionDraft(
            "mission_1",
            "uav_1",
            1,
            2,
            "goto_search",
            AvoidanceStrategy(
                AvoidanceStrategyType.BYPASS_LEFT,
                "original_goto_target",
                ("LEFT_CLEARANCE_VISIBLE",),
            ),
            route,
            (
                ObstacleReplacementStep(
                    "follow_detour",
                    "uav_1",
                    "FOLLOW_ROUTE",
                    {"route_ref": request.route_id},
                ),
                ObstacleReplacementStep(
                    "land_home_reused", "uav_1", "LAND", {}
                ),
            ),
        )
        compiled = coordinator.captured["compile_replacement"](draft)
        self.assertEqual(compiled.plan_version, 2)
        self.assertEqual(compiled.steps[0].step_id, "takeoff")
        self.assertEqual(compiled.steps[1].skill, SkillName.FOLLOW_ROUTE)
        self.assertEqual(compiled.steps[1].params["route_ref"], request.route_id)
        self.assertEqual(compiled.steps[-1].params, _plan().steps[-1].params)

    def test_accepted_handoff_resets_for_next_hazard_but_keeps_history(self) -> None:
        coordinator = _Coordinator()
        coordinator.state = "ACCEPTED"
        runtime = self._runtime(coordinator)
        result = runtime.tick(
            obstacle_snapshot=SimpleNamespace(
                state=SimpleNamespace(value="CLEAR"),
                fusion=None,
            ),
            manager=_Manager(_plan(), paused=False),
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            frame_id="frame_route_2",
            timestamp_s=6.0,
            mission_elapsed_s=5.6,
            hold_pose=FramePose((0.0, 0.0, 10.0)),
        )
        self.assertEqual(result.coordinator_state, "IDLE")
        self.assertTrue(coordinator.reset_preserve)

    def test_search_and_track_continuations_preserve_corridor_and_plan_version(self) -> None:
        plan = _tracking_plan()
        cases = (
            (
                "search_1",
                (
                    ObstacleReplacementStep(
                        "restart_search",
                        "uav_1",
                        "SEARCH",
                        {"target_continuation": "RESTART_SEARCH"},
                    ),
                    ObstacleReplacementStep("track_after_search", "uav_1", "TRACK", {}),
                ),
                (SkillName.SEARCH, SkillName.TRACK),
            ),
            (
                "track_1",
                (
                    ObstacleReplacementStep(
                        "continue_track",
                        "uav_1",
                        "TRACK",
                        {"target_continuation": "CONTINUE_TRACK"},
                    ),
                ),
                (SkillName.TRACK,),
            ),
            (
                "track_1",
                (
                    ObstacleReplacementStep(
                        "search_again",
                        "uav_1",
                        "SEARCH",
                        {"target_continuation": "RESTART_SEARCH"},
                    ),
                    ObstacleReplacementStep("track_again", "uav_1", "TRACK", {}),
                ),
                (SkillName.SEARCH, SkillName.TRACK),
            ),
            (
                "track_1",
                (
                    ObstacleReplacementStep(
                        "recover_track",
                        "uav_1",
                        "TRACK",
                        {"target_continuation": "REACQUIRE"},
                    ),
                ),
                (SkillName.TRACK,),
            ),
        )
        for active_step_id, target_steps, expected_skills in cases:
            with self.subTest(
                active_step_id=active_step_id,
                action=target_steps[0].target_continuation,
            ):
                coordinator = _Coordinator()
                runtime = self._runtime(coordinator, original_plan=plan)
                runtime.observe_active_corridor(
                    FlightCorridor(
                        (0.0, 0.0, 10.0),
                        (12.0, 0.0, 10.0),
                        0.5,
                    ),
                    collision_state="CLEAR",
                )
                result = runtime.tick(
                    obstacle_snapshot=SimpleNamespace(
                        state=SimpleNamespace(value="GEOMETRY_GROUNDED"),
                        fusion=SimpleNamespace(obstacle_ids=("box_red",)),
                    ),
                    manager=_Manager(plan, active_step_id=active_step_id),
                    rgb=np.zeros((8, 12, 3), dtype=np.uint8),
                    frame_id=f"frame_{active_step_id}_{target_steps[0].step_id}",
                    timestamp_s=7.0,
                    mission_elapsed_s=6.5,
                    hold_pose=FramePose((0.0, 0.0, 10.0)),
                )

                self.assertTrue(result.revision_started)
                request = coordinator.captured["request"]
                self.assertEqual(request.base_plan_version, 1)
                self.assertEqual(request.new_plan_version, 2)
                self.assertEqual(request.replace_from_step_id, active_step_id)
                self.assertEqual(
                    request.active_corridor_rejoin_target.xyz_m,
                    (12.0, 0.0, 0.0),
                )
                route = RouteDraft(
                    request.route_id,
                    CoordinateFrame.UAV_HOLD_FLU,
                    (
                        RouteWaypoint("wp_detour", (2.0, 3.0, 0.0)),
                        RouteWaypoint("wp_rejoin", (12.0, 0.0, 0.0)),
                    ),
                )
                draft = ObstacleRouteRevisionDraft(
                    "mission_1",
                    "uav_1",
                    1,
                    2,
                    active_step_id,
                    AvoidanceStrategy(
                        AvoidanceStrategyType.BYPASS_LEFT,
                        "original_goto_target",
                        ("LEFT_CLEARANCE_VISIBLE",),
                    ),
                    route,
                    (
                        ObstacleReplacementStep(
                            "follow_detour",
                            "uav_1",
                            "FOLLOW_ROUTE",
                            {"route_ref": request.route_id},
                        ),
                        *target_steps,
                        ObstacleReplacementStep("return_home", "uav_1", "GOTO", {}),
                        ObstacleReplacementStep("land_home_v2", "uav_1", "LAND", {}),
                    ),
                )

                compiled = coordinator.captured["compile_replacement"](draft)

                self.assertEqual(compiled.plan_version, 2)
                replacement_index = next(
                    index
                    for index, step in enumerate(compiled.steps)
                    if step.skill is SkillName.FOLLOW_ROUTE
                )
                self.assertEqual(
                    tuple(
                        step.skill
                        for step in compiled.steps[
                            replacement_index + 1 : replacement_index + 1 + len(expected_skills)
                        ]
                    ),
                    expected_skills,
                )
                self.assertNotIn(
                    SkillName.REACQUIRE,
                    [step.skill for step in compiled.steps],
                )


if __name__ == "__main__":
    unittest.main()
