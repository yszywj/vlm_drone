from __future__ import annotations

import math
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.goto import GotoGoal, GotoSkill
from skills.motion_types import MotionPolicy, YawMode
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
    position: tuple[float, float, float],
    *,
    yaw: float = 0.0,
    max_speed: float = 5.0,
    max_yaw_rate: float = 2.0,
) -> KinematicUAV:
    return KinematicUAV(
        UAVState(*position, yaw),
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


def position(state: UAVState) -> np.ndarray:
    return np.asarray([state.x, state.y, state.z], dtype=np.float64)


def wrapped_error(actual: float, expected: float) -> float:
    return (actual - expected + math.pi) % (2.0 * math.pi) - math.pi


def run_until_terminal(
    skill: GotoSkill,
    uav: KinematicUAV,
    clock: ManualClock,
    *,
    dt_s: float,
    max_steps: int = 2000,
) -> tuple[list[UAVState], SkillStatus]:
    states = [uav.get_pose()]
    for _ in range(max_steps):
        status = skill.tick(make_observation(uav, clock))
        if status is not SkillStatus.RUNNING:
            return states, status
        states.append(uav.step(dt_s))
        clock.advance(dt_s)
    raise AssertionError("GOTO did not reach a terminal state")


class GotoSkillTest(unittest.TestCase):
    def test_goal_defaults(self) -> None:
        goal = GotoGoal(position=(1.0, 2.0, 3.0))
        self.assertEqual(goal.tolerance, 1.0)
        self.assertEqual(goal.timeout, 60.0)
        self.assertIs(goal.motion_policy.yaw_mode, YawMode.COURSE_ALIGNED)
        self.assertIsNone(goal.motion_policy.max_speed)

    def test_ten_random_initial_positions_reach_goal_continuously(self) -> None:
        rng = np.random.default_rng(20260815)
        for case_index in range(10):
            with self.subTest(case=case_index):
                initial = rng.uniform((-10.0, -10.0, 5.0), (10.0, 10.0, 8.0))
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                goal_position = initial + 3.0 * direction

                uav = make_uav(tuple(initial), max_speed=2.0, max_yaw_rate=10.0)
                clock = ManualClock()
                skill = GotoSkill()
                goal = GotoGoal(
                    position=tuple(goal_position),
                    tolerance=0.02,
                    motion_policy=MotionPolicy(max_speed=1.5),
                    timeout=10.0,
                )
                skill.start(goal, make_context(uav, clock))
                states, status = run_until_terminal(skill, uav, clock, dt_s=0.1)

                self.assertIs(status, SkillStatus.SUCCEEDED)
                result = skill.get_result()
                self.assertIsNotNone(result)
                self.assertIs(result.code, SkillResultCode.GOAL_REACHED)
                distances = [
                    float(np.linalg.norm(goal_position - position(state)))
                    for state in states
                ]
                self.assertGreaterEqual(len(states) - 1, 2)
                self.assertLessEqual(distances[-1], goal.tolerance + 1e-12)
                for previous, current in zip(states, states[1:]):
                    displacement = float(np.linalg.norm(position(current) - position(previous)))
                    self.assertLessEqual(displacement, 1.5 * 0.1 + 1e-12)
                for previous, current in zip(distances, distances[1:]):
                    self.assertLess(current, previous)
                np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))

    def test_course_aligned_translation_and_yaw_change_in_same_step(self) -> None:
        uav = make_uav((0.0, 0.0, 1.0), yaw=0.0, max_yaw_rate=1.0)
        clock = ManualClock()
        skill = GotoSkill()
        initial = uav.get_pose()
        skill.start(
            GotoGoal(
                position=(0.0, 4.0, 1.0),
                tolerance=0.01,
                motion_policy=MotionPolicy(
                    max_speed=1.0,
                    max_yaw_rate=0.5,
                    yaw_mode=YawMode.COURSE_ALIGNED,
                ),
            ),
            make_context(uav, clock),
        )

        self.assertEqual(uav.get_pose(), initial)
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.RUNNING)
        self.assertEqual(uav.get_pose(), initial)
        np.testing.assert_allclose(uav.get_velocity(), [0.0, 1.0, 0.0])

        first = uav.step(0.2)
        clock.advance(0.2)
        self.assertAlmostEqual(first.y, 0.2)
        self.assertAlmostEqual(first.yaw, 0.1)
        self.assertGreater(first.y, initial.y)
        self.assertGreater(first.yaw, initial.yaw)

    def test_keep_current_preserves_start_yaw_while_side_flying(self) -> None:
        initial_yaw = 0.7
        uav = make_uav((0.0, 0.0, 1.0), yaw=initial_yaw)
        clock = ManualClock()
        skill = GotoSkill()
        skill.start(
            GotoGoal(
                position=(0.0, 2.0, 1.0),
                tolerance=0.001,
                motion_policy=MotionPolicy(
                    max_speed=1.0,
                    yaw_mode=YawMode.KEEP_CURRENT,
                ),
            ),
            make_context(uav, clock),
        )

        states, status = run_until_terminal(skill, uav, clock, dt_s=0.2)
        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertGreater(states[-1].y, states[0].y)
        for state in states:
            self.assertAlmostEqual(state.yaw, initial_yaw)

    def test_fixed_yaw_zero_rotates_while_translating(self) -> None:
        uav = make_uav((0.0, 0.0, 1.0), yaw=0.8, max_yaw_rate=1.0)
        clock = ManualClock()
        skill = GotoSkill()
        skill.start(
            GotoGoal(
                position=(0.0, 5.0, 1.0),
                tolerance=0.001,
                motion_policy=MotionPolicy(
                    max_speed=1.0,
                    max_yaw_rate=0.4,
                    yaw_mode=YawMode.FIXED,
                    yaw_value=0.0,
                ),
            ),
            make_context(uav, clock),
        )

        states, status = run_until_terminal(skill, uav, clock, dt_s=0.1)
        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertAlmostEqual(states[1].y, 0.1)
        self.assertAlmostEqual(wrapped_error(states[1].yaw, states[0].yaw), -0.04)
        self.assertAlmostEqual(states[-1].yaw, 0.0)

    def test_face_point_tracks_world_point_while_moving(self) -> None:
        goal_position = np.asarray([20.0, 30.0, 10.0])
        look_at_point = np.asarray([30.0, 40.0, 0.0])
        uav = make_uav(
            (0.0, 0.0, 10.0),
            yaw=0.0,
            max_speed=10.0,
            max_yaw_rate=100.0,
        )
        clock = ManualClock()
        skill = GotoSkill()
        skill.start(
            GotoGoal(
                position=tuple(goal_position),
                tolerance=0.01,
                motion_policy=MotionPolicy(
                    max_speed=10.0,
                    max_yaw_rate=100.0,
                    yaw_mode=YawMode.FACE_POINT,
                    look_at_point=tuple(look_at_point),
                ),
            ),
            make_context(uav, clock),
        )

        states, status = run_until_terminal(skill, uav, clock, dt_s=0.1)
        self.assertIs(status, SkillStatus.SUCCEEDED)
        for state in states[1:]:
            expected_yaw = math.atan2(
                look_at_point[1] - state.y,
                look_at_point[0] - state.x,
            )
            self.assertAlmostEqual(wrapped_error(state.yaw, expected_yaw), 0.0)
        self.assertLessEqual(
            float(np.linalg.norm(goal_position - position(states[-1]))),
            0.01,
        )
        self.assertAlmostEqual(states[-1].yaw, math.pi / 4.0)
        self.assertNotAlmostEqual(states[-1].yaw, math.atan2(30.0, 20.0))

    def test_start_and_tick_do_not_teleport(self) -> None:
        uav = make_uav((0.0, 0.0, 1.0), max_speed=5.0)
        clock = ManualClock()
        skill = GotoSkill()
        initial = uav.get_pose()
        skill.start(
            GotoGoal(
                position=(10.0, 0.0, 1.0),
                tolerance=0.01,
                motion_policy=MotionPolicy(max_speed=2.0),
            ),
            make_context(uav, clock),
        )
        self.assertEqual(uav.get_pose(), initial)
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.RUNNING)
        self.assertEqual(uav.get_pose(), initial)
        self.assertAlmostEqual(np.linalg.norm(uav.get_velocity()), 2.0)

        first = uav.step(0.25)
        self.assertAlmostEqual(np.linalg.norm(position(first) - position(initial)), 0.5)
        self.assertNotEqual(first, UAVState(10.0, 0.0, 1.0, 0.0))

    def test_timeout_fails_and_stops(self) -> None:
        uav = make_uav((0.0, 0.0, 1.0), max_speed=1.0)
        clock = ManualClock()
        skill = GotoSkill()
        skill.start(
            GotoGoal(
                position=(100.0, 0.0, 1.0),
                tolerance=0.01,
                motion_policy=MotionPolicy(max_speed=0.5),
                timeout=0.5,
            ),
            make_context(uav, clock),
        )
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.RUNNING)
        uav.step(0.1)
        clock.advance(0.1)
        self.assertAlmostEqual(uav.get_pose().x, 0.05)

        clock.time_s = 1.0
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.TIMEOUT)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        stopped_pose = uav.get_pose()
        self.assertEqual(uav.step(1.0), stopped_pose)

    def test_cancel_stops_translation_and_unfinished_yaw(self) -> None:
        uav = make_uav(
            (0.0, 0.0, 1.0),
            yaw=1.0,
            max_speed=2.0,
            max_yaw_rate=1.0,
        )
        clock = ManualClock()
        skill = GotoSkill()
        skill.start(
            GotoGoal(
                position=(0.0, 10.0, 1.0),
                motion_policy=MotionPolicy(
                    max_speed=1.0,
                    max_yaw_rate=1.0,
                    yaw_mode=YawMode.FIXED,
                    yaw_value=0.0,
                ),
            ),
            make_context(uav, clock),
        )
        self.assertIs(skill.tick(make_observation(uav, clock)), SkillStatus.RUNNING)
        moving_pose = uav.step(0.2)
        clock.advance(0.2)
        self.assertGreater(moving_pose.y, 0.0)
        self.assertLess(moving_pose.yaw, 1.0)

        skill.cancel()
        self.assertIs(skill.status, SkillStatus.CANCELED)
        self.assertIs(skill.get_result().code, SkillResultCode.CANCELED)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        stopped_pose = uav.get_pose()
        self.assertEqual(uav.step(1.0), stopped_pose)


if __name__ == "__main__":
    unittest.main()
