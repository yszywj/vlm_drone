from __future__ import annotations

import math
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.motion_types import YawMode
from skills.takeoff import TakeoffGoal, TakeoffSkill
from skills.types import Observation, SkillContext, SkillResultCode, SkillStatus


class FakeCamera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


class ManualClock:
    def __init__(self, time_s: float = 0.0) -> None:
        self.time_s = time_s

    def now(self) -> float:
        return self.time_s

    def advance(self, dt_s: float) -> None:
        self.time_s += dt_s


def make_uav(
    *,
    altitude: float,
    yaw: float = 0.0,
    max_speed: float = 5.0,
    max_yaw_rate: float = 1.0,
) -> KinematicUAV:
    return KinematicUAV(
        UAVState(0.0, 0.0, altitude, yaw),
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


def run_until_terminal(
    skill: TakeoffSkill,
    uav: KinematicUAV,
    clock: ManualClock,
    *,
    dt_s: float,
    max_steps: int = 1000,
) -> tuple[list[UAVState], SkillStatus]:
    states = [uav.get_pose()]
    for _ in range(max_steps):
        status = skill.tick(make_observation(uav, clock))
        if status is not SkillStatus.RUNNING:
            return states, status
        states.append(uav.step(dt_s))
        clock.advance(dt_s)
    raise AssertionError("TAKEOFF did not reach a terminal state")


class TakeoffSkillTest(unittest.TestCase):
    def test_takeoff_from_ground_is_continuous_and_not_teleported(self) -> None:
        uav = make_uav(altitude=0.0)
        clock = ManualClock()
        skill = TakeoffSkill()
        skill.start(
            TakeoffGoal(target_altitude=1.0, tolerance=0.01, climb_speed=0.5),
            make_context(uav, clock),
        )

        self.assertIs(skill.status, SkillStatus.RUNNING)
        self.assertEqual(uav.get_pose().z, 0.0)
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.RUNNING)
        self.assertEqual(uav.get_pose().z, 0.0)
        np.testing.assert_allclose(uav.get_velocity(), [0.0, 0.0, 0.5])

        first = uav.step(0.2)
        clock.advance(0.2)
        self.assertAlmostEqual(first.z, 0.1)
        remaining_states, status = run_until_terminal(skill, uav, clock, dt_s=0.2)
        states = [UAVState(0.0, 0.0, 0.0, 0.0), first, *remaining_states[1:]]

        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertIs(skill.get_result().code, SkillResultCode.TAKEOFF_COMPLETE)
        self.assertLessEqual(abs(states[-1].z - 1.0), 0.01)
        for previous, current in zip(states, states[1:]):
            self.assertGreater(current.z, previous.z)
            self.assertLessEqual(current.z - previous.z, 0.1 + 1e-12)
            self.assertAlmostEqual(current.x, 0.0)
            self.assertAlmostEqual(current.y, 0.0)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))

    def test_takeoff_from_nonzero_initial_height(self) -> None:
        uav = make_uav(altitude=1.25)
        clock = ManualClock()
        skill = TakeoffSkill()
        skill.start(
            TakeoffGoal(target_altitude=2.25, tolerance=0.001, climb_speed=0.5),
            make_context(uav, clock),
        )

        states, status = run_until_terminal(skill, uav, clock, dt_s=0.25)
        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertAlmostEqual(states[1].z, 1.375)
        self.assertAlmostEqual(states[-1].z, 2.25)
        self.assertEqual(len(states) - 1, 8)

    def test_keep_current_yaw_is_default(self) -> None:
        initial_yaw = 0.73
        uav = make_uav(altitude=0.0, yaw=initial_yaw)
        clock = ManualClock()
        skill = TakeoffSkill()
        defaults = TakeoffGoal(target_altitude=1.0)
        self.assertEqual(defaults.tolerance, 0.2)
        self.assertEqual(defaults.climb_speed, 1.0)
        self.assertIs(defaults.yaw_mode, YawMode.KEEP_CURRENT)
        self.assertIsNone(defaults.yaw_value)
        self.assertEqual(defaults.timeout, 20.0)
        goal = TakeoffGoal(target_altitude=1.0, tolerance=0.01, climb_speed=0.5)
        self.assertIs(goal.yaw_mode, YawMode.KEEP_CURRENT)
        skill.start(
            goal,
            make_context(uav, clock),
        )

        states, status = run_until_terminal(skill, uav, clock, dt_s=0.2)
        self.assertIs(status, SkillStatus.SUCCEEDED)
        for state in states:
            self.assertAlmostEqual(state.yaw, initial_yaw)

    def test_fixed_yaw_rotates_gradually_while_climbing(self) -> None:
        uav = make_uav(altitude=0.0, yaw=-0.5, max_yaw_rate=0.4)
        clock = ManualClock()
        skill = TakeoffSkill()
        skill.start(
            TakeoffGoal(
                target_altitude=2.0,
                tolerance=0.01,
                climb_speed=0.5,
                yaw_mode=YawMode.FIXED,
                yaw_value=0.5,
            ),
            make_context(uav, clock),
        )

        states, status = run_until_terminal(skill, uav, clock, dt_s=0.5)
        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertAlmostEqual(states[1].yaw, -0.3)
        self.assertAlmostEqual(states[-1].yaw, 0.5)
        for previous, current in zip(states, states[1:]):
            yaw_step = (current.yaw - previous.yaw + math.pi) % (2.0 * math.pi) - math.pi
            self.assertLessEqual(abs(yaw_step), 0.2 + 1e-12)
            self.assertGreater(current.z, previous.z)

    def test_timeout_fails_and_stops(self) -> None:
        uav = make_uav(altitude=0.0)
        clock = ManualClock()
        skill = TakeoffSkill()
        skill.start(
            TakeoffGoal(
                target_altitude=10.0,
                tolerance=0.01,
                climb_speed=0.5,
                timeout=0.5,
            ),
            make_context(uav, clock),
        )
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.RUNNING)
        uav.step(0.1)
        clock.advance(0.1)
        self.assertAlmostEqual(uav.get_pose().z, 0.05)

        clock.time_s = 1.0
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.TIMEOUT)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        stopped_pose = uav.get_pose()
        self.assertEqual(uav.step(1.0), stopped_pose)

    def test_cancel_stops_takeoff_motion(self) -> None:
        uav = make_uav(altitude=0.0, yaw=0.0, max_yaw_rate=0.5)
        clock = ManualClock()
        skill = TakeoffSkill()
        skill.start(
            TakeoffGoal(
                target_altitude=2.0,
                yaw_mode=YawMode.FIXED,
                yaw_value=1.0,
            ),
            make_context(uav, clock),
        )
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.RUNNING)
        self.assertGreater(uav.get_velocity()[2], 0.0)
        uav.step(0.2)
        clock.advance(0.2)
        self.assertAlmostEqual(uav.get_pose().yaw, 0.1)

        skill.cancel()
        self.assertIs(skill.status, SkillStatus.CANCELED)
        self.assertIs(skill.get_result().code, SkillResultCode.CANCELED)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        stopped_pose = uav.get_pose()
        self.assertEqual(uav.step(1.0), stopped_pose)

    def test_fixed_yaw_requires_yaw_value(self) -> None:
        uav = make_uav(altitude=0.0)
        skill = TakeoffSkill()
        skill.start(
            TakeoffGoal(target_altitude=1.0, yaw_mode=YawMode.FIXED),
            make_context(uav, ManualClock()),
        )
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)

    def test_large_step_and_climb_speed_do_not_overshoot_target(self) -> None:
        uav = make_uav(altitude=0.0, max_speed=100.0)
        clock = ManualClock()
        skill = TakeoffSkill()
        skill.start(
            TakeoffGoal(
                target_altitude=1.0,
                tolerance=0.001,
                climb_speed=50.0,
            ),
            make_context(uav, clock),
        )
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.RUNNING)
        self.assertAlmostEqual(uav.get_velocity()[2], 50.0)

        state = uav.step(1.0)
        clock.advance(1.0)
        self.assertAlmostEqual(state.z, 1.0)
        self.assertLessEqual(state.z, 1.0)
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
