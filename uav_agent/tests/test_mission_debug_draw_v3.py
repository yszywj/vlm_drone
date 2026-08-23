from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from common.obstacle_types import FlightCorridor, ObstacleSpec
from planner.route_types import RouteDraft, RouteState, RouteWaypoint
from planner.spatial import (
    CircleRegion,
    CoordinateFrame,
    CorridorRegion,
    PolygonRegion,
    RectangleRegion,
    SectorRegion,
)
from planner.spatial_resolver import FramePose
from runtime.route_registry import RouteRecord
from skills.plan import TaskPlan, TaskStep
from skills.search_strategy import SearchStrategySpec, SearchStrategyType
from skills.types import SkillName
from visualization.mission_debug_draw import (
    DebugDrawOptions,
    DebugDrawSnapshot,
    MissionDebugDraw,
    MissionStatusOverlay,
    route_state_color,
)


class _FakeDraw:
    def __init__(self) -> None:
        self.starts: list[object] = []
        self.ends: list[object] = []
        self.colors: list[object] = []
        self.points: list[object] = []

    def clear_lines(self) -> None:
        self.starts = []
        self.ends = []
        self.colors = []

    def clear_points(self) -> None:
        self.points = []

    def draw_lines(self, starts: list[object], ends: list[object], colors: list[object], widths: list[object]) -> None:
        self.starts = list(starts)
        self.ends = list(ends)
        self.colors = list(colors)
        self.widths = list(widths)

    def draw_points(self, points: list[object], colors: list[object], sizes: list[object]) -> None:
        self.points = list(points)
        self.point_colors = list(colors)
        self.point_sizes = list(sizes)


class _FakeFrame:
    def __enter__(self) -> "_FakeFrame":
        return self

    def __exit__(self, *args: object) -> None:
        del args


class _FakeWindow:
    def __init__(self, title: str, **kwargs: object) -> None:
        self.title = title
        self.options = kwargs
        self.frame = _FakeFrame()
        self.visible = True
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class _FakeLabel:
    def __init__(self, text: str, **kwargs: object) -> None:
        self.text = text
        self.options = kwargs


class _FakeUI:
    Window = _FakeWindow
    Label = _FakeLabel


class _RecordingOverlay:
    def __init__(self) -> None:
        self.snapshots: list[DebugDrawSnapshot] = []
        self.closed = False

    def update(self, snapshot: DebugDrawSnapshot) -> None:
        self.snapshots.append(snapshot)

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _Pose:
    x: float
    y: float
    z: float


def _visualizer() -> tuple[MissionDebugDraw, _FakeDraw]:
    draw = _FakeDraw()
    visualizer = MissionDebugDraw(
        world_context=SimpleNamespace(
            landing_zones={},
            initial_uav_xyz_m=(0.0, 0.0, 0.0),
        ),
        camera_config=SimpleNamespace(
            resolution_wh_px=(640, 480),
            horizontal_fov_deg=90.0,
        ),
        options=DebugDrawOptions(circle_segments=12, corridor_segments=8),
        draw_interface=draw,
    )
    return visualizer, draw


class MissionDebugDrawV3Test(unittest.TestCase):
    def test_status_overlay_shows_plan_version_and_current_skill_as_text(self) -> None:
        overlay = MissionStatusOverlay(ui_module=_FakeUI)
        overlay.update(
            DebugDrawSnapshot(
                plan_version=3,
                current_skill="FOLLOW_ROUTE",
                search_region_shapes=("SECTOR",),
                search_strategies=("SECTOR_SWEEP",),
                search_waypoint_count=7,
                obstacle_count=2,
                route_states=("EXECUTING",),
                hold_active=True,
                trajectory_segment_count=42,
            )
        )

        self.assertIn("Plan version: 3", overlay.text)
        self.assertIn("Current Skill: FOLLOW_ROUTE", overlay.text)
        self.assertIn("SECTOR / SECTOR_SWEEP", overlay.text)
        self.assertIn("HOLD: ACTIVE", overlay.text)
        window = overlay._window
        overlay.close()
        self.assertTrue(window.destroyed)
        self.assertFalse(window.visible)

    def test_visualizer_updates_and_closes_injected_status_overlay(self) -> None:
        overlay = _RecordingOverlay()
        draw = _FakeDraw()
        visualizer = MissionDebugDraw(
            world_context=SimpleNamespace(
                landing_zones={},
                initial_uav_xyz_m=(0.0, 0.0, 0.0),
            ),
            camera_config=SimpleNamespace(
                resolution_wh_px=(640, 480),
                horizontal_fov_deg=90.0,
            ),
            draw_interface=draw,
            status_overlay=overlay,
        )
        visualizer.update(
            uav_pose=_Pose(0, 0, 5),
            camera_position_m=None,
            camera_orientation_wxyz=None,
            active_skill="SEARCH",
            active_step_id=None,
            target_lifecycle="SEARCHING",
        )
        self.assertEqual(overlay.snapshots[-1].current_skill, "SEARCH")
        visualizer.close()
        self.assertTrue(overlay.closed)

    def test_all_resolved_region_shapes_and_model_waypoints_are_drawn(self) -> None:
        visualizer, draw = _visualizer()
        regions_and_strategies = (
            (
                CircleRegion(CoordinateFrame.WORLD_ENU, (0, 0, 0), 5),
                SearchStrategySpec(
                    SearchStrategyType.MODEL_WAYPOINTS,
                    model_waypoints_xyz_m=((0, 0, 8), (2, 0, 8)),
                ),
            ),
            (
                RectangleRegion(CoordinateFrame.WORLD_ENU, (0, 0, 0), 8, 4, 30),
                SearchStrategySpec(SearchStrategyType.LAWNMOWER, spacing_m=2),
            ),
            (
                SectorRegion(CoordinateFrame.WORLD_ENU, (0, 0, 0), 0, 90, (1, 5)),
                SearchStrategySpec(SearchStrategyType.SECTOR_SWEEP, spacing_m=2),
            ),
            (
                PolygonRegion(
                    CoordinateFrame.WORLD_ENU,
                    ((0, 0, 0), (6, 0, 0), (4, 4, 0), (0, 3, 0)),
                ),
                SearchStrategySpec(SearchStrategyType.PERIMETER, spacing_m=2),
            ),
            (
                CorridorRegion(
                    CoordinateFrame.WORLD_ENU,
                    ((0, 0, 0), (4, 0, 0), (8, 2, 0)),
                    1.5,
                ),
                SearchStrategySpec(SearchStrategyType.CORRIDOR_FOLLOW),
            ),
        )
        for region, strategy in regions_and_strategies:
            with self.subTest(shape=region.shape):
                visualizer.set_search_region(
                    region,
                    strategy=strategy,
                    altitude_m=8.0,
                )
                snapshot = visualizer.snapshot()
                self.assertEqual(snapshot.search_region_shapes, (region.shape,))
                self.assertEqual(snapshot.search_strategies, (strategy.kind.value,))
                self.assertGreater(snapshot.search_waypoint_count, 0)
                self.assertTrue(draw.starts)
                self.assertTrue(draw.points)
        self.assertTrue(visualizer.viewport_only)

    def test_obstacles_corridor_hold_and_route_state_colors_are_independent_overlays(self) -> None:
        visualizer, draw = _visualizer()
        visualizer.set_obstacles(
            (ObstacleSpec("box_red", (5, 0, 3), (2, 2, 4), (1, 0, 0)),)
        )
        visualizer.set_safety_corridor(FlightCorridor((0, 0, 3), (8, 0, 3), 1.0))
        visualizer.set_hold_point((0, 0, 3))

        records = []
        for index, state in enumerate(
            (
                RouteState.PROPOSED,
                RouteState.REJECTED,
                RouteState.ACCEPTED,
                RouteState.EXECUTING,
            )
        ):
            route_id = f"route_{index}"
            route = RouteDraft(
                route_id,
                CoordinateFrame.WORLD_ENU,
                (
                    RouteWaypoint(f"wp_{index}_a", (0, index, 5)),
                    RouteWaypoint(f"wp_{index}_b", (8, index, 5)),
                ),
            )
            records.append(
                RouteRecord(
                    route_id=route_id,
                    frame_snapshot=FramePose((0, 0, 0)),
                    raw_proposal={"route_id": route_id},
                    route=route,
                    plan_version=1,
                    proposal_timestamp_s=float(index),
                    state=state,
                )
            )
        visualizer.set_route_records(records)
        snapshot = visualizer.snapshot()
        self.assertEqual(snapshot.obstacle_count, 1)
        self.assertTrue(snapshot.hold_active)
        self.assertEqual(snapshot.route_states, tuple(state.value for state in (
            RouteState.PROPOSED,
            RouteState.REJECTED,
            RouteState.ACCEPTED,
            RouteState.EXECUTING,
        )))
        for state in snapshot.route_states:
            self.assertIn(route_state_color(state), draw.colors)

    def test_serialized_v3_plan_and_follow_route_goal_are_visible(self) -> None:
        visualizer, draw = _visualizer()
        region = RectangleRegion(CoordinateFrame.WORLD_ENU, (4, 5, 0), 8, 6)
        strategy = SearchStrategySpec(SearchStrategyType.LAWNMOWER, spacing_m=2)
        plan = TaskPlan(
            (
                TaskStep(
                    "search_1",
                    SkillName.SEARCH,
                    {
                        "region": region,
                        "strategy": strategy,
                        "search_altitude_m": 8.0,
                    },
                ),
                TaskStep(
                    "follow_1",
                    SkillName.FOLLOW_ROUTE,
                    {"route_ref": "route_relative"},
                ),
            ),
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=3,
        )
        visualizer.set_plan(SimpleNamespace(task_plan=plan))
        route = RouteDraft(
            "route_relative",
            CoordinateFrame.UAV_HOLD_FLU,
            (
                RouteWaypoint("wp_a", (1, 0, 0)),
                RouteWaypoint("wp_b", (3, 0, 0)),
            ),
        )
        visualizer.set_route_records(
            (
                RouteRecord(
                    route_id="route_relative",
                    frame_snapshot=FramePose((10, 20, 8), yaw_rad=0.0),
                    raw_proposal={"route_id": "route_relative"},
                    route=route,
                    plan_version=3,
                    proposal_timestamp_s=1.0,
                    state=RouteState.EXECUTING,
                ),
            )
        )
        visualizer.update(
            uav_pose=_Pose(10, 20, 8),
            camera_position_m=None,
            camera_orientation_wxyz=None,
            active_skill=SkillName.FOLLOW_ROUTE,
            active_step_id="follow_1",
            target_lifecycle=None,
        )
        self.assertEqual(visualizer.snapshot().plan_version, 3)
        self.assertEqual(visualizer.snapshot().current_skill, "FOLLOW_ROUTE")
        self.assertEqual(visualizer.snapshot().search_region_shapes, ("RECTANGLE",))
        self.assertEqual(draw.ends[-1], (13.0, 20.0, 8.0))

    def test_rejected_coordinator_proposal_is_drawn_even_if_never_registered(self) -> None:
        visualizer, draw = _visualizer()
        route = RouteDraft(
            "route_rejected_raw",
            CoordinateFrame.UAV_HOLD_FLU,
            (
                RouteWaypoint("wp_raw_a", (1, 0, 0)),
                RouteWaypoint("wp_raw_b", (3, 0, 0)),
            ),
        )
        visualizer.set_route_records(
            (
                SimpleNamespace(
                    proposal={"route_draft": route.to_dict()},
                    outcome="REVISE",
                    frame_snapshot=FramePose((10, 20, 8)),
                ),
            )
        )

        self.assertEqual(
            visualizer.snapshot().route_states,
            ("PROPOSED", "REJECTED"),
        )
        self.assertIn(route_state_color("PROPOSED"), draw.colors)
        self.assertIn(route_state_color("REJECTED"), draw.colors)

    def test_frustum_color_reflects_search_lock_and_hazard_hold(self) -> None:
        visualizer, draw = _visualizer()

        def update(skill: str, lifecycle: str, **flags: bool) -> tuple[float, float, float, float]:
            visualizer.update(
                uav_pose=_Pose(0, 0, 5),
                camera_position_m=(0, 0, 5),
                camera_orientation_wxyz=(1, 0, 0, 0),
                active_skill=skill,
                active_step_id=None,
                target_lifecycle=lifecycle,
                **flags,
            )
            self.assertEqual(len(set(draw.colors)), 1)
            return draw.colors[0]  # type: ignore[return-value]

        self.assertEqual(update("GOTO", "NONE"), (0.10, 0.85, 1.00, 0.80))
        self.assertEqual(update("SEARCH", "NONE"), (1.00, 0.75, 0.05, 0.95))
        self.assertEqual(update("TRACK", "LOCKED"), (0.10, 1.00, 0.20, 0.95))
        self.assertEqual(
            update("HOVER", "LOCKED", hold_active=True),
            (1.00, 0.10, 0.10, 0.95),
        )
        visualizer.close()
        self.assertFalse(draw.starts)

    def test_executed_trajectory_remains_colored_by_active_skill(self) -> None:
        visualizer, draw = _visualizer()
        for pose, skill in (
            (_Pose(0, 0, 5), "SEARCH"),
            (_Pose(1, 0, 5), "SEARCH"),
            (_Pose(2, 0, 5), "FOLLOW_ROUTE"),
        ):
            visualizer.update(
                uav_pose=pose,
                camera_position_m=None,
                camera_orientation_wxyz=None,
                active_skill=skill,
                active_step_id=None,
                target_lifecycle="NONE",
            )
        self.assertEqual(visualizer.snapshot().trajectory_segment_count, 2)
        self.assertEqual(draw.colors[0], (1.00, 0.75, 0.05, 1.00))
        self.assertEqual(draw.colors[1], (0.00, 1.00, 1.00, 1.00))


if __name__ == "__main__":
    unittest.main()
