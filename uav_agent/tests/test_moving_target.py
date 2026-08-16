from __future__ import annotations

import unittest

import numpy as np

from env.moving_target import MovingTarget, TargetMotionMode


def make_target(mode: TargetMotionMode, seed: int = 7) -> MovingTarget:
    return MovingTarget(
        mode=mode,
        initial_position_xyz_m=[0.0, 0.0, 0.5],
        bounds_min_xyz_m=[-2.0, -1.0, 0.5],
        bounds_max_xyz_m=[2.0, 1.0, 0.5],
        speed_mps=1.0,
        max_speed_mps=1.5,
        direction_change_interval_s=0.7,
        seed=seed,
        initial_heading_rad=0.0,
    )


class MovingTargetTest(unittest.TestCase):
    def test_static_does_not_move(self) -> None:
        target = make_target(TargetMotionMode.STATIC)
        initial = target.get_pose()
        for _ in range(100):
            target.step(1.0)
        self.assertEqual(target.get_pose(), initial)
        np.testing.assert_array_equal(target.get_velocity(), np.zeros(3))

    def test_linear_reflects_and_stays_in_bounds(self) -> None:
        target = make_target(TargetMotionMode.LINEAR)
        for _ in range(20_000):
            state = target.step(0.037)
            self.assertGreaterEqual(state.x, -2.0)
            self.assertLessEqual(state.x, 2.0)
            self.assertGreaterEqual(state.y, -1.0)
            self.assertLessEqual(state.y, 1.0)
        self.assertLessEqual(np.linalg.norm(target.get_velocity()), 1.0 + 1e-12)

    def test_linear_does_not_reflect_before_wall(self) -> None:
        target = make_target(TargetMotionMode.LINEAR)
        state = target.step(1.99999)
        self.assertAlmostEqual(state.x, 1.99999)
        self.assertGreater(target.get_velocity()[0], 0.0)

    def test_random_walk_is_seeded_and_bounded_for_minutes(self) -> None:
        first = make_target(TargetMotionMode.RANDOM_WALK, seed=11)
        second = make_target(TargetMotionMode.RANDOM_WALK, seed=11)
        initial = first.get_pose()
        moved = False
        for _ in range(18_000):  # five simulated minutes at 60 Hz
            pose_a = first.step(1.0 / 60.0)
            pose_b = second.step(1.0 / 60.0)
            moved = moved or pose_a != initial
            self.assertEqual(pose_a, pose_b)
            self.assertGreaterEqual(pose_a.x, -2.0)
            self.assertLessEqual(pose_a.x, 2.0)
            self.assertGreaterEqual(pose_a.y, -1.0)
            self.assertLessEqual(pose_a.y, 1.0)
            self.assertEqual(pose_a.z, 0.5)
        self.assertTrue(moved)

    def test_failed_reset_is_atomic(self) -> None:
        target = make_target(TargetMotionMode.LINEAR)
        target.step(0.25)
        pose_before = target.get_pose()
        velocity_before = target.get_velocity()
        mode_before = target.mode
        with self.assertRaises(ValueError):
            target.reset(mode=TargetMotionMode.STATIC, position_m=[99.0, 0.0, 0.5])
        self.assertEqual(target.get_pose(), pose_before)
        np.testing.assert_array_equal(target.get_velocity(), velocity_before)
        self.assertIs(target.mode, mode_before)


if __name__ == "__main__":
    unittest.main()
