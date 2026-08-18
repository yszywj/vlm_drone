from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from math import atan2
from typing import Sequence

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from env.uav_controller import UAVState as CanonicalUAVState
from skills.base import Skill
from skills.goto import GotoGoal, GotoSkill
from skills.motion_types import MotionPolicy, YawMode, apply_motion_policy
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillResultCode,
    SkillStatus,
    UAVController,
)


class FakeCamera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


class FakeClock:
    def now(self) -> float:
        return 0.0


class FakeUAVController:
    """Independent adapter fake; deliberately does not inherit KinematicUAV."""

    def __init__(self) -> None:
        self.max_speed_mps = 4.0
        self.max_yaw_rate_rad_s = 1.5
        self._pose = UAVState(1.0, 2.0, 3.0, 0.4)
        self._velocity = np.zeros(3, dtype=np.float64)
        self.last_move: dict[str, object] | None = None
        self.last_yaw: dict[str, object] | None = None
        self.last_face_point: dict[str, object] | None = None
        self.stop_count = 0

    def get_pose(self) -> UAVState:
        return self._pose

    def get_velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def set_velocity(
        self,
        velocity_xyz_mps: Sequence[float],
        yaw_rate_rad_s: float = 0.0,
    ) -> None:
        self._velocity = np.asarray(velocity_xyz_mps, dtype=np.float64).copy()
        self.last_yaw = {
            "rate_command": float(yaw_rate_rad_s),
        }

    def move_toward(
        self,
        goal_xyz_m: Sequence[float],
        speed_mps: float | None = None,
        *,
        face_goal: bool = True,
        tolerance_m: float | None = None,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        goal = np.asarray(goal_xyz_m, dtype=np.float64)
        current = np.asarray(
            [self._pose.x, self._pose.y, self._pose.z],
            dtype=np.float64,
        )
        delta = goal - current
        distance = float(np.linalg.norm(delta))
        requested_speed = self.max_speed_mps if speed_mps is None else float(speed_mps)
        self._velocity = (
            np.zeros(3, dtype=np.float64)
            if distance <= float(tolerance_m or 0.0)
            else delta / distance * min(requested_speed, self.max_speed_mps)
        )
        self.last_move = {
            "goal": tuple(float(value) for value in goal),
            "speed_mps": speed_mps,
            "face_goal": face_goal,
            "tolerance_m": tolerance_m,
            "max_yaw_rate_rad_s": max_yaw_rate_rad_s,
        }

    def rotate_yaw(
        self,
        target_yaw_rad: float,
        *,
        relative: bool = False,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        self.last_yaw = {
            "target": float(target_yaw_rad),
            "relative": relative,
            "max_yaw_rate_rad_s": max_yaw_rate_rad_s,
        }

    def face_point(
        self,
        point_xyz_m: Sequence[float],
        *,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        point = tuple(float(value) for value in point_xyz_m)
        self.last_face_point = {
            "point": point,
            "max_yaw_rate_rad_s": max_yaw_rate_rad_s,
        }
        self.last_yaw = {
            "target": atan2(point[1] - self._pose.y, point[0] - self._pose.x),
        }

    def stop(self) -> None:
        self._velocity.fill(0.0)
        self.stop_count += 1


class IncompleteController:
    """Looks controller-like but intentionally lacks FACE_POINT support."""

    max_speed_mps = 1.0
    max_yaw_rate_rad_s = 1.0

    def get_pose(self) -> UAVState:
        return UAVState(0.0, 0.0, 0.0, 0.0)

    def get_velocity(self) -> np.ndarray:
        return np.zeros(3)

    def set_velocity(self, velocity_xyz_mps: Sequence[float], yaw_rate_rad_s: float = 0.0) -> None:
        pass

    def move_toward(
        self,
        goal_xyz_m: Sequence[float],
        speed_mps: float | None = None,
        **kwargs: object,
    ) -> None:
        pass

    def rotate_yaw(self, target_yaw_rad: float, **kwargs: object) -> None:
        pass

    def stop(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class PolicyGoal(SkillGoal):
    motion_policy: MotionPolicy = field(default_factory=MotionPolicy)


class PolicySkill(Skill):
    goal_type = PolicyGoal

    def _on_tick(self, observation: Observation) -> None:
        pass

    def command(self, velocity_xyz_mps: Sequence[float]) -> np.ndarray:
        return self._apply_motion_policy(velocity_xyz_mps)


def make_context(uav: object) -> SkillContext:
    return SkillContext(
        uav=uav,  # type: ignore[arg-type]
        camera=FakeCamera(),
        perception=None,
        clock=FakeClock(),
        uav_id="uav_1",
    )


def make_observation(pose: UAVState) -> Observation:
    return Observation(
        uav_id="uav_1",
        timestamp=0.1,
        uav_pose=pose,
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
    )


class UAVControllerProtocolTest(unittest.TestCase):
    def test_kinematic_module_reexports_simulator_independent_state(self) -> None:
        self.assertIs(UAVState, CanonicalUAVState)
        self.assertEqual(UAVState.__module__, "env.uav_controller")

    def test_kinematic_and_independent_adapter_satisfy_protocol(self) -> None:
        kinematic = KinematicUAV(UAVState(0.0, 0.0, 1.0, 0.0), 2.0, 1.0)
        adapter = FakeUAVController()

        self.assertIsInstance(kinematic, UAVController)
        self.assertIsInstance(adapter, UAVController)
        make_context(adapter).validate()

    def test_context_rejects_incomplete_structural_controller(self) -> None:
        controller = IncompleteController()
        self.assertNotIsInstance(controller, UAVController)
        with self.assertRaisesRegex(TypeError, "must satisfy UAVController"):
            make_context(controller).validate()

        skill = PolicySkill()
        skill.start(PolicyGoal(), make_context(controller))
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_STATE)

    def test_motion_policy_uses_non_kinematic_controller(self) -> None:
        controller = FakeUAVController()
        commanded = apply_motion_policy(
            controller,
            (0.0, 6.0, 0.0),
            MotionPolicy(
                max_speed=2.0,
                max_yaw_rate=0.5,
                yaw_mode=YawMode.FACE_POINT,
                look_at_point=(5.0, 8.0, 0.0),
            ),
        )

        self.assertAlmostEqual(float(np.linalg.norm(commanded)), 2.0)
        self.assertEqual(controller.last_face_point["point"], (5.0, 8.0, 0.0))
        self.assertEqual(controller.last_face_point["max_yaw_rate_rad_s"], 0.5)

    def test_base_skill_and_goto_accept_non_kinematic_controller(self) -> None:
        controller = FakeUAVController()
        policy_skill = PolicySkill()
        policy_skill.start(
            PolicyGoal(MotionPolicy(yaw_mode=YawMode.KEEP_CURRENT)),
            make_context(controller),
        )
        self.assertIs(policy_skill.status, SkillStatus.RUNNING)
        policy_skill.command((0.0, 1.0, 0.0))
        self.assertEqual(controller.last_yaw["target"], 0.4)
        policy_skill.cancel()

        goto = GotoSkill()
        goto.start(GotoGoal(position=(8.0, 2.0, 3.0)), make_context(controller))
        self.assertIs(goto.status, SkillStatus.RUNNING)
        self.assertIs(
            goto.tick(make_observation(controller.get_pose())),
            SkillStatus.RUNNING,
        )
        self.assertEqual(controller.last_move["goal"], (8.0, 2.0, 3.0))
        self.assertTrue(controller.last_move["face_goal"])
        goto.cancel()


if __name__ == "__main__":
    unittest.main()
