from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.base import Skill
from skills.motion_types import (
    MotionPolicy,
    MotionPolicyValidationError,
    YawMode,
    apply_motion_policy,
)
from skills.types import Observation, SkillContext, SkillGoal, SkillResultCode, SkillStatus
from tests.test_skill_base import FakeCamera, FakeClock, make_observation


@dataclass(frozen=True, slots=True)
class PolicyGoal(SkillGoal):
    motion_policy: MotionPolicy = field(default_factory=MotionPolicy)


class PolicySkill(Skill):
    goal_type = PolicyGoal

    def _on_tick(self, observation: Observation) -> None:
        pass

    def command(self, velocity_xyz_mps: tuple[float, float, float]) -> np.ndarray:
        return self._apply_motion_policy(velocity_xyz_mps)


def context_for(uav: KinematicUAV) -> SkillContext:
    return SkillContext(
        uav=uav,
        camera=FakeCamera(),
        perception=None,
        clock=FakeClock(),
        uav_id="uav_1",
    )


class MotionPolicyTest(unittest.TestCase):
    def test_fixed_requires_yaw_value_at_skill_start(self) -> None:
        skill = PolicySkill()
        goal = PolicyGoal(MotionPolicy(yaw_mode=YawMode.FIXED))
        skill.start(goal, context_for(KinematicUAV(UAVState(0, 0, 1, 0), 5, 1)))
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)

    def test_face_point_requires_look_at_point_at_skill_start(self) -> None:
        skill = PolicySkill()
        goal = PolicyGoal(MotionPolicy(yaw_mode=YawMode.FACE_POINT))
        skill.start(goal, context_for(KinematicUAV(UAVState(0, 0, 1, 0), 5, 1)))
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)

    def test_course_aligned_moves_and_turns_with_policy_limits(self) -> None:
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, 0.0), 5.0, 2.0)
        policy = MotionPolicy(
            max_speed=1.0,
            max_yaw_rate=0.25,
            yaw_mode=YawMode.COURSE_ALIGNED,
        )
        commanded = apply_motion_policy(uav, [0.0, 4.0, 0.0], policy)
        self.assertAlmostEqual(np.linalg.norm(commanded), 1.0)
        state = uav.step(1.0)
        self.assertAlmostEqual(state.y, 1.0)
        self.assertAlmostEqual(state.yaw, 0.25)

    def test_keep_current_allows_side_flight(self) -> None:
        initial_yaw = 0.7
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, initial_yaw), 5.0, 2.0)
        apply_motion_policy(
            uav,
            [0.0, 1.0, 0.0],
            MotionPolicy(yaw_mode=YawMode.KEEP_CURRENT),
            initial_yaw=initial_yaw,
        )
        state = uav.step(0.5)
        self.assertAlmostEqual(state.x, 0.0)
        self.assertAlmostEqual(state.y, 0.5)
        self.assertAlmostEqual(state.yaw, initial_yaw)

    def test_fixed_yaw_is_independent_of_sideways_translation(self) -> None:
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, -0.5), 5.0, 2.0)
        apply_motion_policy(
            uav,
            [0.0, 1.0, 0.0],
            MotionPolicy(yaw_mode=YawMode.FIXED, yaw_value=0.0),
        )
        state = uav.step(1.0)
        self.assertAlmostEqual(state.x, 0.0)
        self.assertAlmostEqual(state.y, 1.0)
        self.assertAlmostEqual(state.yaw, 0.0)

    def test_face_point_recomputes_heading_as_uav_moves(self) -> None:
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, 0.0), 5.0, 10.0)
        apply_motion_policy(
            uav,
            [1.0, 0.0, 0.0],
            MotionPolicy(
                yaw_mode=YawMode.FACE_POINT,
                look_at_point=(10.0, 10.0, 0.0),
            ),
        )
        first = uav.step(1.0)
        second = uav.step(1.0)
        self.assertAlmostEqual(first.yaw, math.atan2(10.0, 9.0))
        self.assertAlmostEqual(second.yaw, math.atan2(10.0, 8.0))
        self.assertGreater(second.yaw, first.yaw)

    def test_policy_yaw_rate_cannot_exceed_uav_limit_and_does_not_leak(self) -> None:
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, 0.0), 5.0, 0.5)
        apply_motion_policy(
            uav,
            [0.0, 0.0, 0.0],
            MotionPolicy(
                max_yaw_rate=10.0,
                yaw_mode=YawMode.FIXED,
                yaw_value=math.pi / 2.0,
            ),
        )
        self.assertAlmostEqual(uav.step(1.0).yaw, 0.5)

        uav.rotate_yaw(math.pi / 2.0, max_yaw_rate_rad_s=0.1)
        self.assertAlmostEqual(uav.step(1.0).yaw, 0.6)
        uav.rotate_yaw(math.pi / 2.0)
        self.assertAlmostEqual(uav.step(1.0).yaw, 1.1)

    def test_motion_vectors_reject_bool_and_numeric_strings(self) -> None:
        with self.assertRaises(MotionPolicyValidationError):
            MotionPolicy(
                yaw_mode=YawMode.FACE_POINT,
                look_at_point=(True, 0.0, 0.0),
            ).validate()
        with self.assertRaises(MotionPolicyValidationError):
            apply_motion_policy(
                KinematicUAV(UAVState(0, 0, 1, 0), 5, 1),
                ("1", 0.0, 0.0),
                MotionPolicy(),
            )

    def test_invalid_commands_do_not_replace_existing_motion(self) -> None:
        uav = KinematicUAV(UAVState(0, 0, 1, 0), 5, 1)
        uav.set_velocity((0.5, 0.0, 0.0))
        bad_point = (value for value in (1.0, 2.0, 3.0))
        with self.assertRaises(MotionPolicyValidationError):
            apply_motion_policy(
                uav,
                (0.0, 1.0, 0.0),
                MotionPolicy(
                    yaw_mode=YawMode.FACE_POINT,
                    look_at_point=bad_point,  # type: ignore[arg-type]
                ),
            )
        np.testing.assert_array_equal(uav.get_velocity(), [0.5, 0.0, 0.0])

        with self.assertRaises(ValueError):
            uav.move_toward(
                (10.0, 0.0, 1.0),
                face_goal=np.asarray([True, False]),  # type: ignore[arg-type]
            )
        np.testing.assert_array_equal(uav.get_velocity(), [0.5, 0.0, 0.0])

    def test_world_velocity_is_independent_of_heading(self) -> None:
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, 0.0), 5.0, math.pi)
        uav.set_velocity([0.0, 1.0, 0.0])
        state = uav.step(0.5)
        self.assertEqual(state, UAVState(0.0, 0.5, 1.0, 0.0))

    def test_keep_current_yaw_is_captured_when_skill_starts(self) -> None:
        uav = KinematicUAV(UAVState(0.0, 0.0, 1.0, -0.4), 5.0, 1.0)
        skill = PolicySkill()
        skill.start(
            PolicyGoal(MotionPolicy(yaw_mode=YawMode.KEEP_CURRENT)),
            context_for(uav),
        )
        self.assertIs(skill.status, SkillStatus.RUNNING)
        self.assertAlmostEqual(skill.initial_yaw, -0.4)
        skill.command((0.0, 1.0, 0.0))
        state = uav.step(0.5)
        self.assertAlmostEqual(state.y, 0.5)
        self.assertAlmostEqual(state.yaw, -0.4)
        self.assertIs(skill.tick(make_observation()), SkillStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()
