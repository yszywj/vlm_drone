from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.base import Skill
from skills.goto import GotoGoal
from skills.hover import HoverSkill, HoverTimeoutFallback
from skills.manager import ExecutionKind, SkillManager, SkillManagerError, TaskStatus
from skills.plan import TaskPlan
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillName,
    SkillResultCode,
    SkillStatus,
)


class _Camera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value


@dataclass(frozen=True)
class _Outcome:
    status: SkillStatus
    code: SkillResultCode
    data: dict[str, object]


class _ScriptedSkill(Skill):
    goal_type = SkillGoal

    def __init__(self, *outcomes: _Outcome) -> None:
        super().__init__()
        self.outcomes = deque(outcomes)
        self.started_goals: list[SkillGoal] = []

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        del context
        self.started_goals.append(deepcopy(goal))

    def _on_tick(self, observation: Observation) -> None:
        del observation
        outcome = self.outcomes.popleft()
        if outcome.status is SkillStatus.SUCCEEDED:
            self._succeed(outcome.code, "done", outcome.data)
        else:
            self._fail(outcome.code, "failed", outcome.data)


def _ok(code: SkillResultCode, **data: object) -> _Outcome:
    return _Outcome(SkillStatus.SUCCEEDED, code, data)


def _context() -> tuple[SkillContext, _Clock]:
    clock = _Clock()
    return (
        SkillContext(
            KinematicUAV(
                UAVState(0.0, 0.0, 5.0, 0.0),
                max_speed_mps=5.0,
                max_yaw_rate_rad_s=2.0,
            ),
            _Camera(),
            None,
            clock,
            uav_id="uav_1",
        ),
        clock,
    )


def _observation(clock: _Clock) -> Observation:
    return Observation(
        timestamp=clock.value,
        uav_pose=UAVState(0.0, 0.0, 5.0, 0.0),
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        uav_id="uav_1",
    )


def _goto_plan(*, version: int = 1, x: float = 2.0) -> TaskPlan:
    return TaskPlan.from_dicts(
        [
            {"id": "goto", "skill": "GOTO", "position": [x, 0.0, 5.0]},
            {"id": "land", "skill": "LAND"},
        ],
        mission_id="mission_hover",
        uav_id="uav_1",
        plan_version=version,
    )


class SkillManagerHoverTest(unittest.TestCase):
    def _manager(self) -> tuple[SkillManager, _Clock, _ScriptedSkill, _ScriptedSkill]:
        context, clock = _context()
        goto = _ScriptedSkill(
            _ok(SkillResultCode.GOAL_REACHED),
            _ok(SkillResultCode.GOAL_REACHED),
        )
        land = _ScriptedSkill(_ok(SkillResultCode.LAND_COMPLETE))
        manager = SkillManager(
            context,
            registry={
                SkillName.GOTO: goto,
                SkillName.HOVER: HoverSkill(),
                SkillName.LAND: land,
            },
        )
        return manager, clock, goto, land

    def test_supervisory_hover_releases_and_resumes_exact_goal(self) -> None:
        manager, clock, goto, _ = self._manager()
        manager.start_task(_goto_plan())
        original = deepcopy(goto.started_goals[-1])

        manager.interrupt_with_hover(
            "BLOCKING_REVIEW",
            max_wait_s=5.0,
            timeout_fallback=HoverTimeoutFallback.CANCEL_AND_LAND,
        )
        self.assertIs(manager.task_status, TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.HOVER)
        self.assertIs(manager.active_execution_kind, ExecutionKind.SUPERVISORY)

        manager.resume_interrupted_step()
        manager.tick(_observation(clock))

        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertEqual(goto.started_goals[-1], original)
        self.assertFalse(manager.is_supervisory_paused)
        self.assertIn(
            "supervisory_hover_release_requested",
            [record.reason for record in manager.transition_log],
        )
        self.assertIn(
            "interrupted_step_resumed",
            [record.reason for record in manager.transition_log],
        )
        self.assertTrue(all(record.uav_id == "uav_1" for record in manager.transition_log))

    def test_release_can_wait_for_explicit_continuation(self) -> None:
        manager, clock, _, _ = self._manager()
        manager.start_task(_goto_plan())
        manager.interrupt_with_hover("REVIEW", max_wait_s=5.0)
        manager.release_supervisory_hover()
        manager.tick(_observation(clock))

        self.assertIsNone(manager.active_name)
        self.assertIs(manager.tick(_observation(clock)), TaskStatus.RUNNING)
        manager.resume_interrupted_step()
        self.assertIs(manager.active_name, SkillName.GOTO)

    def test_timeout_fallback_resume_or_cancel_and_land(self) -> None:
        with self.subTest(fallback="resume"):
            manager, clock, _, _ = self._manager()
            manager.start_task(_goto_plan())
            manager.interrupt_with_hover(
                "PERIODIC_REVIEW",
                max_wait_s=0.5,
                timeout_fallback=HoverTimeoutFallback.RESUME_PREVIOUS,
            )
            clock.value = 0.5
            manager.tick(_observation(clock))
            self.assertIs(manager.active_name, SkillName.GOTO)
            self.assertIs(manager.task_status, TaskStatus.RUNNING)

        with self.subTest(fallback="land"):
            manager, clock, _, _ = self._manager()
            manager.start_task(_goto_plan())
            manager.interrupt_with_hover(
                "BLOCKING_CONFIRMATION",
                max_wait_s=0.5,
                timeout_fallback=HoverTimeoutFallback.CANCEL_AND_LAND,
            )
            clock.value = 0.5
            manager.tick(_observation(clock))
            self.assertIs(manager.active_name, SkillName.LAND)
            self.assertIs(manager.pending_task_result, TaskStatus.FAILED)

    def test_replacement_advances_version_and_starts_new_current_step(self) -> None:
        manager, clock, goto, _ = self._manager()
        manager.start_task(_goto_plan())
        manager.interrupt_with_hover("REVISION", max_wait_s=5.0)
        manager.replace_interrupted_step_and_suffix(_goto_plan(version=2, x=9.0))
        manager.tick(_observation(clock))

        self.assertEqual(manager.task_plan.plan_version, 2)
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertIsInstance(goto.started_goals[-1], GotoGoal)
        self.assertEqual(goto.started_goals[-1].position, (9.0, 0.0, 5.0))

    def test_interrupted_track_resumes_with_same_target_id(self) -> None:
        context, clock = _context()
        search = _ScriptedSkill(
            _ok(SkillResultCode.TARGET_FOUND, target_id="target_7")
        )
        track = _ScriptedSkill(
            _ok(SkillResultCode.TRACK_COMPLETE),
            _ok(SkillResultCode.TRACK_COMPLETE),
        )
        land = _ScriptedSkill(_ok(SkillResultCode.LAND_COMPLETE))
        manager = SkillManager(
            context,
            registry={
                SkillName.SEARCH: search,
                SkillName.TRACK: track,
                SkillName.HOVER: HoverSkill(),
                SkillName.LAND: land,
            },
        )
        manager.start_task(
            TaskPlan.from_dicts(
                [
                    {
                        "id": "search",
                        "skill": "SEARCH",
                        "center": [0.0, 0.0, 0.0],
                        "radius": 5.0,
                        "target_description": "target",
                    },
                    {
                        "id": "track",
                        "skill": "TRACK",
                        "target_id": "$search.target_id",
                        "track_duration": 3.0,
                    },
                    {"id": "land", "skill": "LAND"},
                ]
            )
        )
        manager.tick(_observation(clock))
        self.assertIs(manager.active_name, SkillName.TRACK)
        manager.interrupt_with_hover("IDENTITY_REVIEW")
        manager.resume_interrupted_step()
        manager.tick(_observation(clock))

        self.assertIs(manager.active_name, SkillName.TRACK)
        self.assertEqual(manager.active_target_id, "target_7")
        self.assertEqual(track.started_goals[-1].target_id, "target_7")

    def test_task_cancel_overrides_supervisory_hover_and_lands(self) -> None:
        manager, clock, _, _ = self._manager()
        manager.start_task(_goto_plan())
        manager.interrupt_with_hover("SOFT_REVIEW")

        manager.cancel_task()

        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(manager.pending_task_result, TaskStatus.CANCELED)
        self.assertFalse(manager.is_supervisory_paused)
        manager.tick(_observation(clock))
        self.assertIs(manager.task_status, TaskStatus.CANCELED)

    def test_takeoff_and_land_are_not_interruptible(self) -> None:
        context, clock = _context()
        takeoff = _ScriptedSkill(_ok(SkillResultCode.TAKEOFF_COMPLETE))
        land = _ScriptedSkill(_ok(SkillResultCode.LAND_COMPLETE))
        manager = SkillManager(
            context,
            registry={
                SkillName.TAKEOFF: takeoff,
                SkillName.HOVER: HoverSkill(),
                SkillName.LAND: land,
            },
        )
        manager.start_task(
            TaskPlan.from_dicts(
                [
                    {"skill": "TAKEOFF", "target_altitude": 5.0},
                    {"skill": "LAND"},
                ]
            )
        )
        with self.assertRaises(SkillManagerError):
            manager.interrupt_with_hover("REVIEW")
        manager.tick(_observation(clock))
        self.assertIs(manager.active_name, SkillName.LAND)
        with self.assertRaises(SkillManagerError):
            manager.interrupt_with_hover("REVIEW")

    def test_planned_timed_hover_executes_linearly(self) -> None:
        context, clock = _context()
        land = _ScriptedSkill(_ok(SkillResultCode.LAND_COMPLETE))
        manager = SkillManager(
            context,
            registry={SkillName.HOVER: HoverSkill(), SkillName.LAND: land},
        )
        manager.start_task(
            TaskPlan.from_dicts(
                [
                    {
                        "skill": "HOVER",
                        "mode": "TIMED",
                        "duration_s": 0.25,
                        "max_wait_s": 1.0,
                    },
                    {"skill": "LAND"},
                ]
            )
        )
        clock.value = 0.25
        manager.tick(_observation(clock))
        self.assertIs(manager.active_name, SkillName.LAND)
        manager.tick(_observation(clock))
        self.assertIs(manager.task_status, TaskStatus.SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
