from __future__ import annotations

import math
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.land import LandGoal, LandSkill
from skills.motion_types import YawMode
from skills.types import Observation, SkillContext, SkillResultCode, SkillStatus


class ManualClock:
    def __init__(self, time_s: float = 0.0) -> None:
        self.time_s = time_s

    def now(self) -> float:
        return self.time_s

    def advance(self, dt_s: float) -> None:
        self.time_s += dt_s


class FakeCamera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


def make_uav(
    position_xyz_m: tuple[float, float, float],
    *,
    yaw: float = 0.0,
    max_speed: float = 5.0,
    max_yaw_rate: float = 1.0,
) -> KinematicUAV:
    return KinematicUAV(
        UAVState(*position_xyz_m, yaw),
        max_speed_mps=max_speed,
        max_yaw_rate_rad_s=max_yaw_rate,
    )


def make_context(uav: KinematicUAV, clock: ManualClock) -> SkillContext:
    return SkillContext(
        uav=uav,
        camera=FakeCamera(),
        perception=None,
        clock=clock,
        uav_id="uav_1",
    )


def make_observation(uav: KinematicUAV, clock: ManualClock) -> Observation:
    return Observation(
        uav_id="uav_1",
        timestamp=clock.now(),
        uav_pose=uav.get_pose(),
        uav_velocity=uav.get_velocity(),
        camera_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def wrapped_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


class LandSkillTest(unittest.TestCase):
    def test_goal_defaults_and_validation(self) -> None:
        defaults = LandGoal()
        self.assertEqual(defaults.ground_altitude, 0.0)
        self.assertEqual(defaults.tolerance, 0.1)
        self.assertEqual(defaults.descent_speed, 0.5)
        self.assertIs(defaults.yaw_mode, YawMode.KEEP_CURRENT)
        self.assertIsNone(defaults.yaw_value)
        self.assertEqual(defaults.timeout, 30.0)
        self.assertIsNone(defaults.expected_position_xy)
        self.assertEqual(defaults.zone_tolerance_m, 0.75)

        with self.assertRaises(ValueError):
            LandGoal(expected_position_xy=(0.0, float("nan")))
        with self.assertRaises(ValueError):
            LandGoal(expected_position_xy=[0.0, 0.0])  # type: ignore[arg-type]

        invalid_goals = (
            LandGoal(ground_altitude=float("nan")),
            LandGoal(tolerance=0.0),
            LandGoal(descent_speed=0.0),
            LandGoal(timeout=0.0),
            LandGoal(zone_tolerance_m=0.0),
            LandGoal(yaw_mode=YawMode.COURSE_ALIGNED),
            LandGoal(yaw_mode=YawMode.FIXED),
        )
        for invalid_goal in invalid_goals:
            with self.subTest(goal=invalid_goal):
                uav = make_uav((0.0, 0.0, 2.0))
                skill = LandSkill()
                skill.start(invalid_goal, make_context(uav, ManualClock()))
                self.assertIs(skill.status, SkillStatus.FAILED)
                self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)

    def test_normal_land_requires_trusted_zone_at_start(self) -> None:
        inside_uav = make_uav((3.2, -4.1, 2.0))
        inside = LandSkill()
        inside.start(
            LandGoal(
                expected_position_xy=(3.0, -4.0),
                zone_tolerance_m=0.75,
            ),
            make_context(inside_uav, ManualClock()),
        )
        self.assertIs(inside.status, SkillStatus.RUNNING)
        feedback = inside.get_feedback().data
        self.assertEqual(feedback["expected_position_xy"], (3.0, -4.0))
        self.assertAlmostEqual(feedback["zone_error_m"], math.hypot(0.2, -0.1))
        self.assertEqual(feedback["zone_tolerance_m"], 0.75)

        outside_uav = make_uav((4.0, -4.0, 2.0))
        outside = LandSkill()
        outside.start(
            LandGoal(
                expected_position_xy=(3.0, -4.0),
                zone_tolerance_m=0.75,
            ),
            make_context(outside_uav, ManualClock()),
        )
        self.assertIs(outside.status, SkillStatus.FAILED)
        self.assertIs(
            outside.get_result().code,
            SkillResultCode.INVALID_STATE,
        )
        outside_feedback = outside.get_feedback()
        self.assertEqual(
            outside_feedback.message,
            "UAV is outside the trusted landing zone",
        )
        self.assertEqual(
            outside_feedback.data["expected_position_xy"],
            (3.0, -4.0),
        )
        self.assertEqual(outside_feedback.data["zone_error_m"], 1.0)
        self.assertEqual(outside_feedback.data["zone_tolerance_m"], 0.75)

    def test_normal_land_fails_if_observation_leaves_trusted_zone(self) -> None:
        uav = make_uav((0.0, 0.0, 2.0))
        clock = ManualClock()
        skill = LandSkill()
        skill.start(
            LandGoal(expected_position_xy=(0.0, 0.0), zone_tolerance_m=0.75),
            make_context(uav, clock),
        )
        drifted = Observation(
            uav_id="uav_1",
            timestamp=0.0,
            uav_pose=UAVState(0.8, 0.0, 2.0, 0.0),
            uav_velocity=np.zeros(3),
            camera_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        )

        self.assertIs(skill.tick(drifted), SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_STATE)

    def test_emergency_land_without_expected_zone_starts_in_place(self) -> None:
        uav = make_uav((40.0, -40.0, 2.0))
        skill = LandSkill()

        skill.start(LandGoal(expected_position_xy=None), make_context(uav, ManualClock()))

        self.assertIs(skill.status, SkillStatus.RUNNING)
        self.assertIsNone(skill.get_feedback().data["expected_position_xy"])
        self.assertIsNone(skill.get_feedback().data["zone_error_m"])

    def test_normal_land_is_continuous_locks_xy_and_keeps_default_yaw(self) -> None:
        uav = make_uav(
            (3.0, -4.0, 2.5),
            yaw=0.73,
        )
        initial = uav.get_pose()
        clock = ManualClock()
        skill = LandSkill()
        goal = LandGoal(
            ground_altitude=0.5,
            tolerance=0.001,
            descent_speed=0.5,
        )
        skill.start(goal, make_context(uav, clock))

        self.assertIs(skill.status, SkillStatus.RUNNING)
        self.assertEqual(uav.get_pose(), initial)
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.RUNNING,
        )
        self.assertEqual(uav.get_pose(), initial)
        np.testing.assert_allclose(uav.get_velocity(), (0.0, 0.0, -0.5))

        states = [initial]
        for _ in range(30):
            states.append(uav.step(0.2))
            clock.advance(0.2)
            status = skill.tick(make_observation(uav, clock))
            if status is not SkillStatus.RUNNING:
                break
        else:
            self.fail("LAND did not reach a terminal state")

        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertIs(skill.get_result().code, SkillResultCode.LAND_COMPLETE)
        self.assertGreaterEqual(states[-1].z, goal.ground_altitude - 1e-12)
        self.assertLessEqual(abs(states[-1].z - goal.ground_altitude), goal.tolerance)
        for previous, current in zip(states, states[1:]):
            self.assertLess(current.z, previous.z)
            self.assertLessEqual(previous.z - current.z, 0.1 + 1e-12)
            self.assertAlmostEqual(current.x, initial.x)
            self.assertAlmostEqual(current.y, initial.y)
            self.assertAlmostEqual(current.yaw, initial.yaw)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        result = skill.get_result().data
        self.assertEqual(result["landing_position_xy"], (initial.x, initial.y))
        self.assertFalse(result["is_airborne"])

    def test_fixed_yaw_rotates_while_descending(self) -> None:
        uav = make_uav(
            (0.0, 0.0, 2.0),
            yaw=-0.5,
            max_yaw_rate=0.4,
        )
        clock = ManualClock()
        skill = LandSkill()
        skill.start(
            LandGoal(
                ground_altitude=0.0,
                tolerance=0.001,
                descent_speed=0.5,
                yaw_mode=YawMode.FIXED,
                yaw_value=0.5,
            ),
            make_context(uav, clock),
        )
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.RUNNING,
        )
        first = uav.step(0.5)
        clock.advance(0.5)
        self.assertAlmostEqual(first.z, 1.75)
        self.assertAlmostEqual(first.yaw, -0.3)

        previous = first
        for _ in range(10):
            status = skill.tick(make_observation(uav, clock))
            if status is not SkillStatus.RUNNING:
                break
            current = uav.step(0.5)
            clock.advance(0.5)
            self.assertLess(current.z, previous.z)
            self.assertLessEqual(abs(wrapped_delta(current.yaw, previous.yaw)), 0.2 + 1e-12)
            previous = current
        else:
            self.fail("LAND with FIXED yaw did not terminate")
        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertAlmostEqual(uav.get_pose().yaw, 0.5)

    def test_large_step_and_speed_do_not_cross_ground(self) -> None:
        uav = make_uav((0.0, 0.0, 1.0), max_speed=100.0)
        clock = ManualClock()
        skill = LandSkill()
        skill.start(
            LandGoal(
                ground_altitude=0.0,
                tolerance=0.001,
                descent_speed=50.0,
            ),
            make_context(uav, clock),
        )
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.RUNNING,
        )
        self.assertAlmostEqual(uav.get_velocity()[2], -50.0)
        landed = uav.step(1.0)
        clock.advance(1.0)
        self.assertAlmostEqual(landed.z, 0.0)
        self.assertGreaterEqual(landed.z, 0.0)
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.SUCCEEDED,
        )

    def test_timeout_fails_and_stops(self) -> None:
        uav = make_uav((0.0, 0.0, 10.0))
        clock = ManualClock()
        skill = LandSkill()
        skill.start(
            LandGoal(
                ground_altitude=0.0,
                tolerance=0.01,
                descent_speed=0.5,
                timeout=0.5,
            ),
            make_context(uav, clock),
        )
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.RUNNING,
        )
        uav.step(0.1)
        self.assertAlmostEqual(uav.get_pose().z, 9.95)

        clock.time_s = 1.0
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.FAILED,
        )
        self.assertIs(skill.get_result().code, SkillResultCode.TIMEOUT)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        stopped_pose = uav.get_pose()
        self.assertEqual(uav.step(1.0), stopped_pose)

    def test_cancel_stops_descent_and_unfinished_yaw(self) -> None:
        uav = make_uav((0.0, 0.0, 2.0), max_yaw_rate=0.5)
        clock = ManualClock()
        skill = LandSkill()
        skill.start(
            LandGoal(yaw_mode=YawMode.FIXED, yaw_value=1.0),
            make_context(uav, clock),
        )
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.RUNNING,
        )
        uav.step(0.2)
        clock.advance(0.2)
        self.assertAlmostEqual(uav.get_pose().z, 1.9)
        self.assertAlmostEqual(uav.get_pose().yaw, 0.1)

        skill.cancel()
        self.assertIs(skill.status, SkillStatus.CANCELED)
        self.assertIs(skill.get_result().code, SkillResultCode.CANCELED)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        stopped_pose = uav.get_pose()
        self.assertEqual(uav.step(1.0), stopped_pose)

    def test_first_tick_succeeds_when_already_at_ground_tolerance(self) -> None:
        uav = make_uav((1.0, 2.0, 0.05))
        clock = ManualClock()
        skill = LandSkill()
        skill.start(LandGoal(ground_altitude=0.0, tolerance=0.1), make_context(uav, clock))
        self.assertIs(skill.status, SkillStatus.RUNNING)
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.SUCCEEDED,
        )
        self.assertIs(skill.get_result().code, SkillResultCode.LAND_COMPLETE)

    def test_landing_at_timeout_deadline_succeeds(self) -> None:
        uav = make_uav((0.0, 0.0, 0.5))
        clock = ManualClock()
        skill = LandSkill()
        skill.start(
            LandGoal(
                ground_altitude=0.0,
                tolerance=0.001,
                descent_speed=0.5,
                timeout=1.0,
            ),
            make_context(uav, clock),
        )
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.RUNNING,
        )
        uav.step(1.0)
        clock.time_s = 1.0
        self.assertIs(
            skill.tick(make_observation(uav, clock)),
            SkillStatus.SUCCEEDED,
        )
        self.assertIs(skill.get_result().code, SkillResultCode.LAND_COMPLETE)

    def test_start_below_ground_is_invalid_state(self) -> None:
        uav = make_uav((0.0, 0.0, -0.2))
        skill = LandSkill()
        skill.start(
            LandGoal(ground_altitude=0.0, tolerance=0.1),
            make_context(uav, ManualClock()),
        )
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_STATE)


if __name__ == "__main__":
    unittest.main()
