from __future__ import annotations

from threading import Thread
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.hover import HoverGoal, HoverMode, HoverSkill
from skills.motion_types import MotionPolicy, YawMode
from skills.types import Observation, SkillContext, SkillResultCode, SkillStatus


class FakeCamera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))


class ManualClock:
    def __init__(self) -> None:
        self.time_s = 0.0

    def now(self) -> float:
        return self.time_s


class CountingUAV(KinematicUAV):
    def __init__(self) -> None:
        super().__init__(
            UAVState(1.0, 2.0, 3.0, 0.4),
            max_speed_mps=3.0,
            max_yaw_rate_rad_s=2.0,
        )
        self.velocity_commands = 0

    def set_velocity(
        self,
        velocity_xyz_mps: object,
        yaw_rate_rad_s: float = 0.0,
    ) -> None:
        self.velocity_commands += 1
        super().set_velocity(velocity_xyz_mps, yaw_rate_rad_s)  # type: ignore[arg-type]


def context(uav: CountingUAV, clock: ManualClock) -> SkillContext:
    return SkillContext(uav, FakeCamera(), None, clock, uav_id="uav_7")


def observation(uav: CountingUAV, clock: ManualClock) -> Observation:
    return Observation(
        timestamp=clock.now(),
        uav_pose=uav.get_pose(),
        uav_velocity=uav.get_velocity(),
        camera_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        uav_id="uav_7",
    )


class HoverSkillTest(unittest.TestCase):
    def test_timed_hover_commands_every_tick_and_completes(self) -> None:
        uav = CountingUAV()
        clock = ManualClock()
        skill = HoverSkill()
        skill.start(
            HoverGoal(mode=HoverMode.TIMED, duration_s=0.3, max_wait_s=1.0),
            context(uav, clock),
        )

        for timestamp in (0.0, 0.1, 0.2):
            clock.time_s = timestamp
            self.assertIs(skill.tick(observation(uav, clock)), SkillStatus.RUNNING)
        clock.time_s = 0.3
        self.assertIs(skill.tick(observation(uav, clock)), SkillStatus.SUCCEEDED)

        self.assertEqual(uav.velocity_commands, 4)
        self.assertIs(skill.get_result().code, SkillResultCode.HOVER_COMPLETE)
        self.assertEqual(skill.get_feedback().data["uav_id"], "uav_7")
        self.assertEqual(
            skill.get_feedback().data["captured_hold_position"],
            (1.0, 2.0, 3.0),
        )

    def test_drift_is_corrected_toward_captured_position(self) -> None:
        uav = CountingUAV()
        clock = ManualClock()
        skill = HoverSkill()
        skill.start(
            HoverGoal(
                duration_s=2.0,
                max_wait_s=3.0,
                position_tolerance_m=0.1,
                max_correction_speed_mps=0.4,
            ),
            context(uav, clock),
        )
        uav.set_pose(2.0, 2.0, 3.0, 0.4)

        skill.tick(observation(uav, clock))

        self.assertLess(uav.get_velocity()[0], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(uav.get_velocity())), 0.4)
        self.assertAlmostEqual(skill.get_feedback().data["position_drift_m"], 1.0)

    def test_until_released_uses_thread_safe_flag_then_completes_on_tick(self) -> None:
        uav = CountingUAV()
        clock = ManualClock()
        skill = HoverSkill()
        skill.start(
            HoverGoal(
                mode=HoverMode.UNTIL_RELEASED,
                duration_s=None,
                max_wait_s=2.0,
                reason_code="BLOCKING_REVIEW",
            ),
            context(uav, clock),
        )
        worker = Thread(target=skill.request_release)
        worker.start()
        worker.join()

        self.assertIs(skill.status, SkillStatus.RUNNING)
        self.assertIs(skill.tick(observation(uav, clock)), SkillStatus.SUCCEEDED)
        self.assertIs(skill.get_result().code, SkillResultCode.HOVER_COMPLETE)

    def test_until_released_timeout_and_goal_validation(self) -> None:
        uav = CountingUAV()
        clock = ManualClock()
        skill = HoverSkill()
        skill.start(
            HoverGoal(
                mode=HoverMode.UNTIL_RELEASED,
                duration_s=None,
                max_wait_s=0.5,
            ),
            context(uav, clock),
        )
        clock.time_s = 0.5
        self.assertIs(skill.tick(observation(uav, clock)), SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.TIMEOUT)

        for invalid in (
            HoverGoal(mode=HoverMode.UNTIL_RELEASED, duration_s=1.0),
            HoverGoal(duration_s=2.0, max_wait_s=1.0),
            HoverGoal(
                motion_policy=MotionPolicy(yaw_mode=YawMode.COURSE_ALIGNED)
            ),
        ):
            with self.subTest(goal=invalid):
                candidate = HoverSkill()
                candidate.start(invalid, context(CountingUAV(), ManualClock()))
                self.assertIs(candidate.status, SkillStatus.FAILED)
                self.assertIs(
                    candidate.get_result().code,
                    SkillResultCode.INVALID_GOAL,
                )


if __name__ == "__main__":
    unittest.main()
