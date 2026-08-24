from __future__ import annotations

import numpy as np
import pytest

from fleet.airspace_manager import (
    AirspaceRisk,
    FleetAirspaceManager,
    FleetPoseSnapshot,
    FleetUavPose,
)


def _snapshot(*poses: FleetUavPose) -> FleetPoseSnapshot:
    return FleetPoseSnapshot(10.0, {pose.uav_id: pose for pose in poses})


def test_prediction_uses_same_timestamp_and_holds_lower_priority() -> None:
    manager = FleetAirspaceManager(
        minimum_separation_m=5.0,
        warning_separation_m=7.0,
        prediction_horizon_s=10.0,
    )
    decision = manager.evaluate(
        _snapshot(
            FleetUavPose(
                "uav_a",
                (-10.0, 0.0, 10.0),
                (2.0, 0.0, 0.0),
                priority=100,
            ),
            FleetUavPose(
                "uav_b",
                (10.0, 0.0, 10.0),
                (-2.0, 0.0, 0.0),
                priority=10,
            ),
        )
    )

    assert decision.event_type == "AIRSPACE_CONFLICT"
    assert decision.hold_uav_ids == ("uav_b",)
    conflict = decision.conflicts[0]
    assert conflict.current_distance_m == pytest.approx(20.0)
    assert conflict.horizontal_distance_m == pytest.approx(20.0)
    assert conflict.vertical_distance_m == pytest.approx(0.0)
    assert conflict.relative_speed_mps == pytest.approx(4.0)
    assert conflict.time_to_closest_s == pytest.approx(5.0)
    assert conflict.predicted_collision_time_s == pytest.approx(3.75)
    assert conflict.predicted_closest_distance_m == pytest.approx(0.0)
    assert conflict.risk is AirspaceRisk.COLLISION_PREDICTED


def test_warning_conflict_is_logged_without_deadlocking_safe_approach() -> None:
    manager = FleetAirspaceManager(
        minimum_separation_m=5.0,
        warning_separation_m=7.5,
        prediction_horizon_s=10.0,
    )
    decision = manager.evaluate(
        _snapshot(
            FleetUavPose(
                "uav_a",
                (-2.8, 0.0, 0.0),
                priority=100,
                landing_zone_id="home_a",
            ),
            FleetUavPose(
                "uav_b",
                (2.8, 0.0, 8.0),
                (0.0, 0.0, -0.5),
                priority=100,
                landing_zone_id="home_b",
            ),
        )
    )

    conflict = decision.conflicts[0]
    assert conflict.risk is AirspaceRisk.CONFLICT
    assert conflict.predicted_closest_distance_m > manager.minimum_separation_m
    assert conflict.predicted_closest_distance_m < manager.warning_separation_m
    assert conflict.routes_intersect is False
    assert conflict.shared_landing_zone is False
    assert decision.event_type == "AIRSPACE_CONFLICT"
    assert decision.hold_uav_ids == ()
    assert conflict.hold_uav_id is None


def test_crossing_routes_are_explicit_conflict() -> None:
    manager = FleetAirspaceManager(5.0)
    decision = manager.evaluate(
        _snapshot(
            FleetUavPose(
                "uav_a",
                (-20.0, -20.0, 10.0),
                priority=5,
                route_xyz_m=((-5.0, 0.0, 10.0), (5.0, 0.0, 10.0)),
            ),
            FleetUavPose(
                "uav_b",
                (20.0, 20.0, 10.0),
                priority=10,
                route_xyz_m=((0.0, -5.0, 10.0), (0.0, 5.0, 10.0)),
            ),
        )
    )
    assert decision.conflicts[0].routes_intersect
    assert decision.hold_uav_ids == ("uav_a",)


def test_well_separated_stationary_uavs_are_clear() -> None:
    decision = FleetAirspaceManager(5.0).evaluate(
        _snapshot(
            FleetUavPose("uav_a", (0.0, 0.0, 10.0)),
            FleetUavPose("uav_b", (20.0, 0.0, 15.0)),
        )
    )
    assert decision.clear
    assert decision.hold_uav_ids == ()
    assert decision.conflicts[0].risk is AirspaceRisk.CLEAR


def test_existing_hard_separation_breach_reports_collision_time_zero() -> None:
    decision = FleetAirspaceManager(5.0).evaluate(
        _snapshot(
            FleetUavPose("uav_a", (0.0, 0.0, 10.0)),
            FleetUavPose("uav_b", (4.0, 0.0, 10.0)),
        )
    )

    assert decision.conflicts[0].predicted_collision_time_s == 0.0


def test_disjoint_vertical_route_legs_do_not_intersect_in_xy() -> None:
    decision = FleetAirspaceManager(5.0).evaluate(
        _snapshot(
            FleetUavPose(
                "uav_a",
                (-3.0, 0.0, 0.0),
                route_xyz_m=((-3.0, 0.0, 0.0), (-3.0, 0.0, 10.0)),
            ),
            FleetUavPose(
                "uav_b",
                (3.0, 0.0, 0.0),
                route_xyz_m=((3.0, 0.0, 0.0), (3.0, 0.0, 10.0)),
            ),
        )
    )
    assert decision.conflicts[0].routes_intersect is False
    assert decision.hold_uav_ids == ()


def test_pose_accepts_three_element_numpy_controller_vectors() -> None:
    pose = FleetUavPose(
        "uav_a",
        np.array([1.0, 2.0, 3.0], dtype=np.float32),  # type: ignore[arg-type]
        np.array([0.5, -0.25, 0.0], dtype=np.float32),  # type: ignore[arg-type]
    )

    assert pose.position_xyz_m == pytest.approx((1.0, 2.0, 3.0))
    assert pose.velocity_xyz_mps == pytest.approx((0.5, -0.25, 0.0))
