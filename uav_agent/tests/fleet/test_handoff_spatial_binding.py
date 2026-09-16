"""Task geometry must survive an aircraft change without reinterpretation."""
from dataclasses import replace
from math import pi

import pytest

from fleet.handoff import freeze_handoff_task
from fleet.task_spec import FleetTaskSpecV1, MissionGoal, TerminationGoal
from planner.spatial import (
    CircleRegion, CoordinateFrame, NamedLocationTarget, PointTarget,
    RectangleRegion, RouteTarget,
)
from planner.spatial_resolver import FramePose, MissingFramePoseError, SpatialResolver


def _resolver():
    return SpatialResolver(
        home_pose=FramePose((10, 20, 1)),
        uav_start_pose=FramePose((30, 40, 2), pi / 2),
        named_locations={"home_source": (10, 20, 1)},
    )


@pytest.mark.parametrize("target, expected", [
    (PointTarget("HOME_ENU", (3, 4, 5)), (13, 24, 6)),
    (PointTarget("UAV_START_FLU", (3, 4, 5)), (26, 43, 7)),
    (NamedLocationTarget("home_source"), (10, 20, 1)),
])
def test_original_navigation_geometry_is_world_bound_before_handoff(target, expected):
    original = FleetTaskSpecV1(
        source_text="Navigate, then return to the executing aircraft's home.",
        goals=(MissionGoal("goal_move", "NAVIGATE", None, target, None, None, "MUST"),),
        termination_goals=(TerminationGoal("goal_return", "RETURN_HOME_AND_LAND", None, None, "MUST"),),
    )
    frozen = freeze_handoff_task(original, _resolver())
    assert frozen.goals[0].spatial_constraint.frame is CoordinateFrame.WORLD_ENU
    assert frozen.goals[0].spatial_constraint.xyz_m == pytest.approx(expected)
    assert frozen.termination_goals == original.termination_goals
    assert original.goals[0].spatial_constraint == target
    # Repeated handoff must not reapply any launch offset or yaw.
    spare = SpatialResolver(home_pose=FramePose((100, 100, 0)),
                            uav_start_pose=FramePose((200, 200, 0), -pi / 2))
    assert freeze_handoff_task(frozen, spare) == frozen


def test_original_search_region_retains_rotation_extent_and_entry():
    region = RectangleRegion("UAV_START_FLU", (3, 4, 0), 12, 8, 15, (3, 0, 0))
    task = FleetTaskSpecV1(source_text="Search the assigned region.", goals=(
        MissionGoal("goal_search", "SEARCH_TARGET", "target_a", region, None, None, "MUST"),
    ))
    frozen = freeze_handoff_task(task, _resolver()).goals[0].spatial_constraint
    assert frozen.frame is CoordinateFrame.WORLD_ENU
    assert frozen.center_xyz_m == pytest.approx((26, 43, 2))
    assert frozen.entry_point_xyz_m == pytest.approx((30, 43, 2))
    assert (frozen.width_m, frozen.height_m, frozen.yaw_deg) == (12, 8, 105)


def test_route_waypoints_are_resolved_in_original_order():
    route = RouteTarget("UAV_START_FLU", ((1, 0, 3), (2, 3, 4)))
    task = FleetTaskSpecV1(source_text="Follow the required route.", goals=(
        MissionGoal("goal_route", "NAVIGATE", None, route, None, None, "MUST"),
    ))
    frozen = freeze_handoff_task(task, _resolver()).goals[0].spatial_constraint
    assert frozen.waypoints_xyz_m[0] == pytest.approx((30, 41, 5))
    assert frozen.waypoints_xyz_m[1] == pytest.approx((27, 42, 6))


def test_missing_original_hold_reference_is_rejected_not_guessed():
    task = FleetTaskSpecV1(source_text="Search around the original hold.", goals=(
        MissionGoal("goal_search", "SEARCH_TARGET", "target_a",
                    CircleRegion("UAV_HOLD_FLU", (1, 2, 0), 5), None, None, "MUST"),
    ))
    with pytest.raises(MissingFramePoseError):
        freeze_handoff_task(task, _resolver())
