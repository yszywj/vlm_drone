from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.base import Skill
from skills.hover import HoverSkill, HoverTimeoutFallback
from skills.manager import ExecutionKind, SkillManager, TaskStatus
from skills.plan import TaskPlan
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillName,
    SkillStatus,
)


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def now(self) -> float:
        return self.value


class _Camera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))


class _RecordingUAV(KinematicUAV):
    def __init__(self) -> None:
        self.velocity_command_count = 0
        super().__init__(
            UAVState(0.0, 0.0, 5.0, 0.0),
            max_speed_mps=3.0,
            max_yaw_rate_rad_s=2.0,
        )

    def set_velocity(self, velocity_xyz_mps, yaw_rate_rad_s: float = 0.0) -> None:
        self.velocity_command_count += 1
        super().set_velocity(velocity_xyz_mps, yaw_rate_rad_s)


class _HoldingSkill(Skill):
    goal_type = SkillGoal

    def __init__(self) -> None:
        super().__init__()
        self.started_goals: list[SkillGoal] = []

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        del context
        self.started_goals.append(deepcopy(goal))

    def _on_tick(self, observation: Observation) -> None:
        del observation


def _plan(*, version: int = 1, x: float = 10.0) -> TaskPlan:
    return TaskPlan.from_dicts(
        [
            {
                "id": "goto_active",
                "skill": "GOTO",
                "position": [x, 0.0, 5.0],
                "timeout": 60.0,
            },
            {"id": "land_home", "skill": "LAND"},
        ],
        mission_id="mission_hover_regression",
        uav_id="uav_1",
        plan_version=version,
    )


def _observation(uav: KinematicUAV, timestamp_s: float) -> Observation:
    return Observation(
        timestamp=timestamp_s,
        uav_pose=uav.get_pose(),
        uav_velocity=uav.get_velocity(),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        uav_id="uav_1",
    )


class SupervisoryHoverRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.uav = _RecordingUAV()
        self.goto = _HoldingSkill()
        self.land = _HoldingSkill()
        self.manager = SkillManager(
            SkillContext(
                self.uav,
                _Camera(),
                None,
                self.clock,
                uav_id="uav_1",
            ),
            registry={
                SkillName.GOTO: self.goto,
                SkillName.HOVER: HoverSkill(),
                SkillName.LAND: self.land,
            },
        )
        self.manager.start_task(_plan())

    def test_supervisory_hover_is_not_ticked_with_pre_start_observation(self) -> None:
        self.manager.interrupt_with_hover(
            "review_path_blocked",
            max_wait_s=5.0,
            defer_observation_timestamp_s=10.0,
        )

        # This frame was sampled before HOVER captured its start pose/clock.
        self.manager.tick(_observation(self.uav, 10.0))

        self.assertIs(self.manager.active_name, SkillName.HOVER)
        self.assertIs(self.manager.active_status, SkillStatus.RUNNING)
        self.assertIs(self.manager.task_status, TaskStatus.RUNNING)
        self.assertNotIn(
            "HOLD_ESTABLISHED",
            [record.reason for record in self.manager.transition_log],
        )

        self.clock.value = 10.1
        self.manager.tick(_observation(self.uav, 10.1))
        self.assertIn(
            "HOLD_ESTABLISHED",
            [record.reason for record in self.manager.transition_log],
        )

    def test_path_blocked_starts_hover_without_immediate_failure(self) -> None:
        self.manager.interrupt_with_hover("review_path_blocked", max_wait_s=5.0)
        self.manager.tick(_observation(self.uav, 9.95))
        for index in range(1, 11):
            self.clock.value = 10.0 + index * 0.1
            self.manager.tick(_observation(self.uav, self.clock.value))
            self.assertIs(self.manager.active_name, SkillName.HOVER)
            self.assertIs(self.manager.active_status, SkillStatus.RUNNING)
            self.assertIs(self.manager.task_status, TaskStatus.RUNNING)
            self.assertIsNone(self.manager.pending_task_result)
        self.assertGreaterEqual(self.uav.velocity_command_count, 10)
        self.assertFalse(
            any(
                record.result_code is not None
                and record.result_code.name == "INVALID_STATE"
                for record in self.manager.transition_log
            )
        )

    def test_supervisory_hover_preserves_interrupted_goal(self) -> None:
        original_goal = deepcopy(self.manager.active_invocation.goal)
        original_outputs = self.manager.step_outputs
        original_step = self.manager.active_planned_step_id

        self.manager.interrupt_with_hover("review_path_blocked")

        self.assertEqual(self.manager.step_outputs, original_outputs)
        self.assertEqual(self.manager.active_planned_step_id, original_step)
        self.assertTrue(self.manager.is_supervisory_paused)
        self.assertNotEqual(self.manager.active_invocation.goal, original_goal)

    def test_supervisory_hover_resume_restores_exact_goal(self) -> None:
        original_goal = deepcopy(self.manager.active_invocation.goal)
        self.manager.interrupt_with_hover("review_path_blocked")
        self.manager.resume_interrupted_step()
        self.clock.value = 10.1
        self.manager.tick(_observation(self.uav, 10.1))

        self.assertIs(self.manager.active_name, SkillName.GOTO)
        self.assertEqual(self.manager.active_invocation.goal, original_goal)
        self.assertFalse(self.manager.is_supervisory_paused)

    def test_supervisory_hover_replacement_uses_new_plan_version(self) -> None:
        self.manager.interrupt_with_hover("review_path_blocked")
        self.manager.replace_interrupted_step_and_suffix(_plan(version=2, x=20.0))
        self.clock.value = 10.1
        self.manager.tick(_observation(self.uav, 10.1))

        self.assertIs(self.manager.active_name, SkillName.GOTO)
        self.assertEqual(self.manager.task_plan.plan_version, 2)
        self.assertEqual(self.manager.active_invocation.plan_version, 2)
        self.assertEqual(self.manager.active_invocation.goal.position, (20.0, 0.0, 5.0))

    def test_hover_timeout_uses_configured_fallback(self) -> None:
        self.manager.interrupt_with_hover(
            "review_path_blocked",
            max_wait_s=0.5,
            timeout_fallback=HoverTimeoutFallback.RESUME_PREVIOUS,
        )
        self.manager.tick(_observation(self.uav, 9.95))
        self.clock.value = 10.5
        self.manager.tick(_observation(self.uav, 10.5))

        self.assertIs(self.manager.active_name, SkillName.GOTO)
        self.assertIs(self.manager.task_status, TaskStatus.RUNNING)
        self.assertIn(
            "supervisory_hover_timeout_resume_previous",
            [record.reason for record in self.manager.transition_log],
        )

    def test_three_delayed_repair_rounds_do_not_trigger_75_second_fallback(self) -> None:
        self.manager.interrupt_with_hover(
            "review_path_blocked",
            max_wait_s=75.0,
            timeout_fallback=HoverTimeoutFallback.CANCEL_AND_LAND,
        )
        self.manager.tick(_observation(self.uav, 9.95))

        # Representative model completions at roughly 20/40/60 simulation
        # seconds remain inside the independent supervisory deadline.
        for timestamp_s in (30.0, 50.0, 70.0):
            self.clock.value = timestamp_s
            self.manager.tick(_observation(self.uav, timestamp_s))
            self.assertIs(self.manager.active_name, SkillName.HOVER)
            self.assertIs(self.manager.task_status, TaskStatus.RUNNING)

        self.clock.value = 85.1
        self.manager.tick(_observation(self.uav, 85.1))
        self.assertIs(self.manager.active_name, SkillName.LAND)
        self.assertIn(
            "supervisory_hover_timeout_cancel_and_land",
            [record.reason for record in self.manager.transition_log],
        )


if __name__ == "__main__":
    unittest.main()
