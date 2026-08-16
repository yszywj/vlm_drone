from __future__ import annotations

import math
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState


class KinematicUAVTest(unittest.TestCase):
    def test_goal_is_reached_continuously(self) -> None:
        uav = KinematicUAV(
            UAVState(0.0, 0.0, 10.0, 0.0),
            max_speed_mps=5.0,
            max_yaw_rate_rad_s=math.radians(60.0),
            goal_tolerance_m=0.05,
        )
        goal = np.asarray([10.0, 5.0, 8.0])
        dt = 1.0 / 60.0
        uav.move_toward(goal)
        self.assertEqual(uav.get_pose(), UAVState(0.0, 0.0, 10.0, 0.0))

        first = uav.step(dt)
        self.assertFalse(np.allclose([first.x, first.y, first.z], goal))
        self.assertLessEqual(
            np.linalg.norm(np.asarray([first.x, first.y, first.z]) - np.asarray([0.0, 0.0, 10.0])),
            5.0 * dt + 1e-12,
        )

        steps = 1
        while not uav.goal_reached() and steps < 1000:
            previous = uav.get_pose()
            current = uav.step(dt)
            displacement = np.linalg.norm(
                np.asarray([current.x - previous.x, current.y - previous.y, current.z - previous.z])
            )
            self.assertLessEqual(displacement, 5.0 * dt + 1e-12)
            steps += 1

        self.assertTrue(uav.goal_reached())
        self.assertGreater(steps, 100)
        self.assertLess(steps, 200)

    def test_speed_and_yaw_rate_are_limited(self) -> None:
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, 0.0), 2.0, math.radians(30.0))
        uav.set_velocity([20.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(uav.get_velocity()), 2.0)
        uav.rotate_yaw(math.pi)
        state = uav.step(0.1)
        self.assertLessEqual(abs(state.yaw), math.radians(3.0) + 1e-12)

    def test_stop_holds_pose(self) -> None:
        uav = KinematicUAV(UAVState(1.0, 2.0, 3.0, 0.0), 3.0, 1.0)
        uav.set_velocity([1.0, 1.0, 0.0], 0.5)
        uav.stop()
        self.assertEqual(uav.step(1.0), UAVState(1.0, 2.0, 3.0, 0.0))

    def test_yaw_uses_shortest_wrapped_direction(self) -> None:
        uav = KinematicUAV(
            UAVState(0.0, 0.0, 1.0, math.radians(179.0)),
            1.0,
            math.radians(10.0),
        )
        uav.rotate_yaw(math.radians(-179.0))
        before = uav.get_pose().yaw
        after = uav.step(0.1).yaw
        delta = (after - before + math.pi) % (2.0 * math.pi) - math.pi
        self.assertAlmostEqual(delta, math.radians(1.0))

    def test_invalid_goal_command_does_not_corrupt_active_navigation(self) -> None:
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, 0.0), 1.0, 1.0)
        uav.move_toward([10.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            uav.move_toward([0.0, 10.0, 1.0], speed_mps=-1.0)
        state = uav.step(1.0)
        self.assertEqual(state, UAVState(1.0, 0.0, 1.0, 0.0))

    def test_new_navigation_clears_old_yaw_target(self) -> None:
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, 0.0), 1.0, 1.0)
        uav.rotate_yaw(math.pi / 2.0)
        uav.move_toward([1.0, 0.0, 1.0], face_goal=False)
        self.assertEqual(uav.step(0.1).yaw, 0.0)


if __name__ == "__main__":
    unittest.main()
