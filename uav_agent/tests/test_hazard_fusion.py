from __future__ import annotations

import unittest

from common.obstacle_types import (
    CameraGeometry,
    FlightCorridor,
    ObstacleMotionState,
    ObstacleObservation,
    ObstacleSpec,
    VisibleObstacle,
)
from env.obstacle_registry import ObstacleRegistry
from perception.ideal_obstacle_perception import IdealObstaclePerception
from runtime.collision_supervisor import (
    CollisionSupervisor,
    CollisionSupervisorAction,
    CollisionSupervisorState,
)
from runtime.events import MissionEventType
from runtime.hazard_fusion import HazardFusion, HazardReport, HazardSource
from runtime.obstacle_runtime import ObstacleHazardRuntime


def _visible(
    obstacle_id: str = "box_red",
    *,
    corridor: bool = True,
    ttc_s: float | None = 2.0,
    depth_m: float = 4.0,
) -> VisibleObstacle:
    return VisibleObstacle(
        obstacle_id=obstacle_id,
        bbox_xyxy_normalized=(0.2, 0.2, 0.8, 0.8),
        relative_center_m=(depth_m, 0.0, 0.0),
        relative_size_m=(2.0, 2.0, 2.0),
        depth_m=depth_m,
        occlusion_ratio=0.0,
        motion_state=ObstacleMotionState.STATIC,
        active_corridor_intersection=corridor,
        time_to_collision_s=ttc_s if corridor else None,
    )


def _observation(*visible: VisibleObstacle) -> ObstacleObservation:
    return ObstacleObservation(
        observation_id="observation_1",
        frame_id="frame_1",
        uav_id="uav_1",
        timestamp_s=1.0,
        visible_obstacles=tuple(visible),
    )


def _fuse(fusion: HazardFusion, reports: tuple[HazardReport, ...]):
    return fusion.fuse(
        reports,
        mission_id="mission_1",
        uav_id="uav_1",
        plan_version=1,
        timestamp_s=1.0,
        uav_speed_mps=2.0,
    )


class HazardFusionTest(unittest.TestCase):
    def test_visible_but_nonblocking_obstacle_does_not_hold(self) -> None:
        fusion = HazardFusion()
        report = fusion.report_from_observation(
            _observation(_visible(corridor=False)),
            uav_speed_mps=2.0,
        )
        result = _fuse(fusion, (report,))
        self.assertFalse(result.should_hold)
        self.assertFalse(result.can_generate_route)
        self.assertEqual(result.visible_obstacle_ids, ("box_red",))
        decision = CollisionSupervisor().evaluate(result)
        self.assertEqual(decision.state, CollisionSupervisorState.CLEAR)
        self.assertEqual(
            tuple(event.event_type for event in decision.events),
            (MissionEventType.OBSTACLE_VISIBLE,),
        )

    def test_repeated_visibility_is_coalesced_until_obstacle_leaves_view(self) -> None:
        fusion = HazardFusion()
        visible_report = fusion.report_from_observation(
            _observation(_visible(corridor=False)),
            uav_speed_mps=2.0,
        )
        visible = _fuse(fusion, (visible_report,))
        empty_report = fusion.report_from_observation(
            _observation(),
            uav_speed_mps=2.0,
        )
        empty = _fuse(fusion, (empty_report,))
        supervisor = CollisionSupervisor()

        self.assertEqual(len(supervisor.evaluate(visible).events), 1)
        self.assertEqual(supervisor.evaluate(visible).events, ())
        self.assertEqual(supervisor.evaluate(empty).events, ())
        self.assertEqual(
            tuple(event.event_type for event in supervisor.evaluate(visible).events),
            (MissionEventType.OBSTACLE_VISIBLE,),
        )

    def test_low_level_or_qwen_hazard_each_requests_hold(self) -> None:
        fusion = HazardFusion()
        low_level = fusion.low_level_report(hazard_detected=True)
        qwen = fusion.qwen_report(hazard_detected=True)
        for report, low, model in (
            (low_level, True, False),
            (qwen, False, True),
        ):
            with self.subTest(source=report.source):
                result = _fuse(fusion, (report,))
                self.assertTrue(result.should_hold)
                self.assertEqual(result.low_level_hazard_detected, low)
                self.assertEqual(result.qwen_hazard_detected, model)

    def test_two_sources_for_same_obstacle_are_deduplicated(self) -> None:
        fusion = HazardFusion()
        ideal = fusion.report_from_observation(
            _observation(_visible()),
            uav_speed_mps=2.0,
        )
        qwen = fusion.qwen_report(
            hazard_detected=True,
            geometry_grounded=True,
            obstacle_ids=("box_red",),
        )
        result = _fuse(fusion, (ideal, qwen))
        self.assertTrue(result.low_level_hazard_detected)
        self.assertTrue(result.qwen_hazard_detected)
        self.assertEqual(result.obstacle_ids, ("box_red",))
        self.assertTrue(result.can_generate_route)
        provenance = result.to_dict()["source_reports"]
        self.assertEqual(provenance[0]["source"], "ideal_camera_obstacle_perception")
        self.assertTrue(provenance[0]["privileged"])

    def test_braking_distance_or_ttc_and_corridor_gate_low_level_stop(self) -> None:
        fusion = HazardFusion(path_blocked_ttc_s=5.0)
        too_far = fusion.report_from_observation(
            _observation(_visible(ttc_s=8.0, depth_m=20.0)),
            uav_speed_mps=1.0,
        )
        braking = fusion.report_from_observation(
            _observation(_visible(ttc_s=None, depth_m=1.0)),
            uav_speed_mps=3.0,
        )
        self.assertFalse(too_far.hazard_detected)
        self.assertTrue(braking.hazard_detected)
        self.assertTrue(braking.imminent_collision)


class CollisionSupervisorTest(unittest.TestCase):
    def test_grounded_hazard_executes_full_hold_replan_resume_state_machine(self) -> None:
        fusion = HazardFusion()
        report = fusion.report_from_observation(
            _observation(_visible(ttc_s=1.0)),
            uav_speed_mps=2.0,
        )
        result = _fuse(fusion, (report,))
        supervisor = CollisionSupervisor()
        braking = supervisor.evaluate(result)
        self.assertEqual(braking.state, CollisionSupervisorState.BRAKING)
        self.assertEqual(braking.action, CollisionSupervisorAction.REQUEST_HOLD)
        self.assertEqual(
            braking.transitions,
            (
                CollisionSupervisorState.HAZARD_SUSPECTED,
                CollisionSupervisorState.BRAKING,
            ),
        )
        types = {event.event_type for event in braking.events}
        self.assertIn(MissionEventType.OBSTACLE_VISIBLE, types)
        self.assertIn(MissionEventType.IMMINENT_COLLISION, types)
        self.assertIn(MissionEventType.HOLD_REQUESTED, types)

        held = supervisor.mark_hold_established(timestamp_s=1.1)
        self.assertEqual(held.state, CollisionSupervisorState.HOLDING)
        self.assertEqual(held.events[0].event_type, MissionEventType.HOLD_ESTABLISHED)
        self.assertEqual(
            supervisor.mark_geometry_grounded().state,
            CollisionSupervisorState.GEOMETRY_GROUNDED,
        )
        self.assertEqual(
            supervisor.begin_replanning().state,
            CollisionSupervisorState.REPLANNING,
        )
        proposed = supervisor.route_proposed("route_1", timestamp_s=2.0)
        self.assertEqual(proposed.events[0].event_type, MissionEventType.ROUTE_PROPOSED)
        accepted = supervisor.route_accepted(
            validation_mode="critic_sim",
            required_checks_passed=True,
            timestamp_s=2.1,
        )
        self.assertTrue(accepted.may_resume)
        self.assertEqual(accepted.events[0].event_type, MissionEventType.ROUTE_ACCEPTED)
        resumed = supervisor.resume(required_checks_passed=True)
        self.assertEqual(resumed.state, CollisionSupervisorState.CLEAR)
        self.assertFalse(resumed.should_hold)

    def test_qwen_suspicion_holds_but_cannot_generate_ungrounded_route(self) -> None:
        fusion = HazardFusion()
        result = _fuse(fusion, (fusion.qwen_report(hazard_detected=True),))
        supervisor = CollisionSupervisor()
        supervisor.evaluate(result)
        supervisor.mark_hold_established(timestamp_s=1.1)
        with self.assertRaisesRegex(RuntimeError, "geometry-grounded"):
            supervisor.mark_geometry_grounded()
        self.assertEqual(supervisor.state, CollisionSupervisorState.HOLDING)

    def test_rejected_route_stays_holding_and_can_be_reproposed(self) -> None:
        fusion = HazardFusion()
        report = fusion.low_level_report(
            hazard_detected=True,
            geometry_grounded=True,
            obstacle_ids=("box_red",),
        )
        supervisor = CollisionSupervisor()
        supervisor.evaluate(_fuse(fusion, (report,)))
        supervisor.mark_hold_established(timestamp_s=1.1)
        supervisor.mark_geometry_grounded()
        supervisor.begin_replanning()
        supervisor.route_proposed("route_1", timestamp_s=2.0)
        rejected = supervisor.route_rejected(
            reason_codes=("SEGMENT_INTERSECTS_AABB",),
            timestamp_s=2.1,
        )
        self.assertEqual(rejected.state, CollisionSupervisorState.REPLANNING)
        self.assertEqual(rejected.events[0].event_type, MissionEventType.ROUTE_REJECTED)
        with self.assertRaises(RuntimeError):
            supervisor.route_accepted(
                validation_mode="strict",
                required_checks_passed=True,
                timestamp_s=2.2,
            )
        supervisor.route_proposed("route_2", timestamp_s=2.3)

    def test_unchecked_route_cannot_resume_and_active_routing_cannot_change(self) -> None:
        fusion = HazardFusion()
        report = fusion.low_level_report(
            hazard_detected=True,
            geometry_grounded=True,
            obstacle_ids=("box_red",),
        )
        supervisor = CollisionSupervisor()
        supervisor.evaluate(_fuse(fusion, (report,)))
        with self.assertRaises(ValueError):
            supervisor.evaluate(
                fusion.fuse(
                    (report,),
                    mission_id="mission_other",
                    uav_id="uav_1",
                    plan_version=1,
                    timestamp_s=2.0,
                )
            )
        supervisor.mark_hold_established(timestamp_s=1.1)
        supervisor.mark_geometry_grounded()
        supervisor.begin_replanning()
        supervisor.route_proposed("route_1", timestamp_s=2.0)
        with self.assertRaisesRegex(RuntimeError, "required checks"):
            supervisor.route_accepted(
                validation_mode="strict",
                required_checks_passed=False,
                timestamp_s=2.1,
            )

    def test_all_required_hazard_sources_and_event_types_are_public(self) -> None:
        self.assertEqual(
            {source.value for source in HazardSource},
            {
                "ideal_camera_obstacle_perception",
                "future_low_level_detector",
                "qwen_visual_review",
            },
        )
        required = {
            "OBSTACLE_VISIBLE",
            "PATH_BLOCKED",
            "IMMINENT_COLLISION",
            "HOLD_REQUESTED",
            "HOLD_ESTABLISHED",
            "ROUTE_PROPOSED",
            "ROUTE_REJECTED",
            "ROUTE_ACCEPTED",
        }
        self.assertTrue(required <= {item.value for item in MissionEventType})


class ObstacleHazardRuntimeTest(unittest.TestCase):
    class _Manager:
        is_supervisory_paused = False

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def interrupt_with_hover(self, reason_code: str, **kwargs: object) -> None:
            self.calls.append((reason_code, kwargs))
            self.is_supervisory_paused = True

    def test_low_level_camera_hazard_requests_hold_before_any_model_result(self) -> None:
        manager = self._Manager()
        events: list[object] = []
        runtime = ObstacleHazardRuntime(
            perception=IdealObstaclePerception(
                ObstacleRegistry(
                    (
                        ObstacleSpec(
                            "box_red",
                            (4.0, 0.0, 0.0),
                            (2.0, 2.0, 2.0),
                            (0.8, 0.2, 0.1),
                        ),
                    )
                )
            ),
            hazard_fusion=HazardFusion(),
            collision_supervisor=CollisionSupervisor(),
            skill_manager=manager,
            event_sink=events.append,
        )
        camera = CameraGeometry(
            "frame_1",
            "uav_1",
            1.0,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (640, 480),
            90.0,
            0.1,
            30.0,
        )
        snapshot = runtime.process_camera_frame(
            camera,
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            active_corridor=FlightCorridor(
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
                0.25,
            ),
            uav_velocity_world_mps=(2.0, 0.0, 0.0),
        )
        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(manager.calls[0][0], "LOW_LEVEL_PATH_BLOCKED")
        self.assertEqual(
            manager.calls[0][1]["defer_observation_timestamp_s"],
            1.0,
        )
        self.assertEqual(snapshot.state, CollisionSupervisorState.BRAKING)
        self.assertTrue(snapshot.geometry_grounded)
        self.assertIn(
            MissionEventType.HOLD_REQUESTED,
            {event.event_type for event in events},
        )
        # The next image is sampled after HOVER has zeroed velocity and the
        # active flight corridor disappears.  That clear observation must not
        # erase the grounded hazard which authorized route generation.
        clear_snapshot = runtime.process_camera_frame(
            CameraGeometry(
                "frame_2",
                "uav_1",
                1.1,
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (640, 480),
                90.0,
                0.1,
                30.0,
            ),
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            active_corridor=None,
            uav_velocity_world_mps=(0.0, 0.0, 0.0),
        )
        self.assertTrue(clear_snapshot.geometry_grounded)
        self.assertEqual(clear_snapshot.fusion.obstacle_ids, ("box_red",))

        held = runtime.mark_hold_established(timestamp_s=1.2)
        self.assertEqual(held.state, CollisionSupervisorState.GEOMETRY_GROUNDED)

    def test_ungrounded_qwen_hazard_holds_but_cannot_ground_route(self) -> None:
        manager = self._Manager()
        fusion = HazardFusion()
        runtime = ObstacleHazardRuntime(
            perception=IdealObstaclePerception(ObstacleRegistry()),
            hazard_fusion=fusion,
            collision_supervisor=CollisionSupervisor(),
            skill_manager=manager,
        )
        snapshot = runtime.add_qwen_hazard(
            fusion.qwen_report(hazard_detected=True),
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            timestamp_s=1.0,
        )
        self.assertTrue(snapshot.hold_requested)
        self.assertFalse(snapshot.geometry_grounded)
        held = runtime.mark_hold_established(timestamp_s=1.1)
        self.assertEqual(held.state, CollisionSupervisorState.HOLDING)


if __name__ == "__main__":
    unittest.main()
