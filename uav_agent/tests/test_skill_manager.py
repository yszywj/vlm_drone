from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.base import Skill
from skills.goto import GotoGoal
from skills.land import LandGoal
from skills.manager import (
    SkillManager,
    SkillManagerError,
    TaskPlan,
    TaskPlanError,
    TaskStatus,
    create_default_skill_registry,
)
from skills.reacquire import ReacquireGoal
from skills.search import SearchGoal
from skills.takeoff import TakeoffGoal
from skills.track import TrackGoal
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillName,
    SkillResultCode,
    SkillStatus,
)


class FakeCamera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


class ManualClock:
    def __init__(self) -> None:
        self.time_s = 0.0

    def now(self) -> float:
        return self.time_s

    def advance(self, dt_s: float = 1.0) -> None:
        self.time_s += dt_s


@dataclass(frozen=True, slots=True)
class ScriptedOutcome:
    status: SkillStatus
    code: SkillResultCode
    data: dict[str, object] = field(default_factory=dict)


class ScriptedSkill(Skill):
    """Accept concrete Goals but finish with one queued result per run."""

    goal_type = SkillGoal

    def __init__(self, *outcomes: ScriptedOutcome) -> None:
        super().__init__()
        self._outcomes = deque(outcomes)
        self.started_goals: list[SkillGoal] = []

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        self.started_goals.append(deepcopy(goal))

    def _on_tick(self, observation: Observation) -> None:
        if not self._outcomes:
            raise AssertionError("ScriptedSkill has no result queued for this run")
        outcome = self._outcomes.popleft()
        if outcome.status is SkillStatus.SUCCEEDED:
            self._succeed(outcome.code, "scripted success", outcome.data)
        elif outcome.status is SkillStatus.FAILED:
            self._fail(outcome.code, "scripted failure", outcome.data)
        else:
            raise AssertionError("scripted terminal status must be SUCCEEDED or FAILED")


class ResetErrorScriptedSkill(ScriptedSkill):
    def _on_reset(self) -> None:
        raise RuntimeError("scripted reset failure")


def succeeded(
    code: SkillResultCode,
    data: dict[str, object] | None = None,
) -> ScriptedOutcome:
    return ScriptedOutcome(SkillStatus.SUCCEEDED, code, data or {})


def failed(
    code: SkillResultCode,
    data: dict[str, object] | None = None,
) -> ScriptedOutcome:
    return ScriptedOutcome(SkillStatus.FAILED, code, data or {})


def standard_plan(*, track_duration: float = 5.0) -> TaskPlan:
    return TaskPlan.from_dicts(
        [
            {
                "skill": "TAKEOFF",
                "target_altitude": 10.0,
            },
            {
                "skill": "GOTO",
                "position": [20.0, 30.0, 10.0],
                "tolerance": 0.5,
            },
            {
                "skill": "SEARCH",
                "center": [20.0, 30.0, 0.0],
                "radius": 15.0,
                "target_description": "moving target",
                # Intentionally omit search_altitude: Manager should propagate
                # the planned TAKEOFF altitude into the typed SearchGoal.
            },
            {
                "skill": "TRACK",
                "target_id": "$SEARCH.result.target_id",
                "track_duration": track_duration,
            },
            {"skill": "LAND"},
        ]
    )


def six_step_plan(*, track_duration: float = 5.0) -> TaskPlan:
    entries = standard_plan(track_duration=track_duration).to_dicts()
    entries.insert(
        -1,
        {
            "skill": "GOTO",
            "position": [2.0, -3.0, 10.0],
            "tolerance": 0.25,
            "timeout": 45.0,
        },
    )
    return TaskPlan.from_dicts(entries)


def default_outcomes() -> dict[SkillName, list[ScriptedOutcome]]:
    return {
        SkillName.TAKEOFF: [succeeded(SkillResultCode.TAKEOFF_COMPLETE)],
        SkillName.GOTO: [succeeded(SkillResultCode.GOAL_REACHED)],
        SkillName.SEARCH: [
            succeeded(SkillResultCode.TARGET_FOUND, {"target_id": "target_0"})
        ],
        SkillName.TRACK: [succeeded(SkillResultCode.TRACK_COMPLETE)],
        SkillName.REACQUIRE: [
            succeeded(SkillResultCode.TARGET_FOUND, {"target_id": "target_0"})
        ],
        SkillName.LAND: [succeeded(SkillResultCode.LAND_COMPLETE)],
    }


def make_registry(
    overrides: dict[SkillName, list[ScriptedOutcome]] | None = None,
) -> tuple[dict[SkillName, ScriptedSkill], dict[SkillName, Skill]]:
    outcomes = default_outcomes()
    if overrides is not None:
        outcomes.update(overrides)
    scripted = {
        name: ScriptedSkill(*name_outcomes)
        for name, name_outcomes in outcomes.items()
    }
    return scripted, dict(scripted)


def make_context() -> tuple[SkillContext, ManualClock]:
    clock = ManualClock()
    context = SkillContext(
        uav=KinematicUAV(
            UAVState(0.0, 0.0, 0.0, 0.0),
            max_speed_mps=5.0,
            max_yaw_rate_rad_s=2.0,
        ),
        camera=FakeCamera(),
        perception=None,
        clock=clock,
    )
    return context, clock


def observation(timestamp: float) -> Observation:
    return Observation(
        timestamp=timestamp,
        uav_pose=UAVState(0.0, 0.0, 10.0, 0.0),
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def tick_once(manager: SkillManager, clock: ManualClock) -> TaskStatus:
    clock.advance()
    status = manager.tick(observation(clock.now()))
    if not isinstance(status, TaskStatus):
        raise AssertionError("task-mode tick must return TaskStatus")
    return status


class SkillManagerTaskTest(unittest.TestCase):
    def test_task_plan_accepts_only_the_five_or_six_step_sequences(self) -> None:
        self.assertEqual(len(standard_plan().steps), 5)
        self.assertEqual(len(six_step_plan().steps), 6)

        base = standard_plan().to_dicts()
        invalid_plans = {
            "missing takeoff": base[1:],
            "missing search": [base[0], base[1], base[3], base[4]],
            "missing track": [base[0], base[1], base[2], base[4]],
            "land not last": [base[0], base[1], base[2], base[4], base[3]],
            "arbitrary order": [base[0], base[2], base[1], base[3], base[4]],
            "duplicate search": [base[0], base[1], base[2], base[2], base[4]],
            "duplicate track": [base[0], base[1], base[3], base[3], base[4]],
            "explicit reacquire": [
                base[0],
                base[1],
                base[2],
                base[3],
                {"skill": "REACQUIRE"},
                base[4],
            ],
        }
        for label, entries in invalid_plans.items():
            with self.subTest(label=label), self.assertRaises(TaskPlanError):
                TaskPlan.from_dicts(entries)

    def test_complete_success_task_and_typed_parameter_passing(self) -> None:
        context, clock = make_context()
        scripted, registry = make_registry()
        console: list[str] = []
        manager = SkillManager(context, registry=registry, logger=console.append)

        self.assertEqual(manager.task_status, TaskStatus.IDLE)
        self.assertEqual(
            set(create_default_skill_registry()),
            set(SkillName),
        )
        self.assertEqual(manager.start_task(standard_plan()), TaskStatus.RUNNING)
        self.assertEqual(manager.active_name, SkillName.TAKEOFF)

        for _ in range(4):
            self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.pending_task_result, TaskStatus.SUCCEEDED)
        self.assertEqual(tick_once(manager, clock), TaskStatus.SUCCEEDED)

        takeoff_goal = scripted[SkillName.TAKEOFF].started_goals[0]
        goto_goal = scripted[SkillName.GOTO].started_goals[0]
        search_goal = scripted[SkillName.SEARCH].started_goals[0]
        track_goal = scripted[SkillName.TRACK].started_goals[0]
        land_goal = scripted[SkillName.LAND].started_goals[0]
        self.assertIsInstance(takeoff_goal, TakeoffGoal)
        self.assertIsInstance(goto_goal, GotoGoal)
        self.assertEqual(goto_goal.position, (20.0, 30.0, 10.0))
        self.assertIsInstance(search_goal, SearchGoal)
        self.assertEqual(search_goal.search_altitude, 10.0)
        self.assertIsInstance(track_goal, TrackGoal)
        self.assertEqual(track_goal.target_id, "target_0")
        self.assertEqual(track_goal.track_duration, 5.0)
        self.assertIsInstance(land_goal, LandGoal)
        self.assertEqual(manager.active_target_id, "target_0")

        expected_new_skills = [
            SkillName.TAKEOFF,
            SkillName.GOTO,
            SkillName.SEARCH,
            SkillName.TRACK,
            SkillName.LAND,
            None,
        ]
        self.assertEqual(
            [record.new_skill for record in manager.transition_log],
            expected_new_skills,
        )
        self.assertEqual(
            [record.reason for record in manager.transition_log],
            [
                "task_started",
                "takeoff_complete",
                "goal_reached",
                "target_found",
                "track_complete",
                "task_completed",
            ],
        )
        self.assertEqual(
            [record.timestamp for record in manager.transition_log],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        search_to_track = manager.transition_log[3]
        self.assertIs(search_to_track.old_skill, SkillName.SEARCH)
        self.assertIs(search_to_track.old_status, SkillStatus.SUCCEEDED)
        self.assertIs(search_to_track.result_code, SkillResultCode.TARGET_FOUND)
        self.assertIs(search_to_track.new_skill, SkillName.TRACK)
        self.assertIsNone(manager.active_name)
        self.assertEqual(len(console), len(manager.transition_log))
        self.assertIs(
            manager.tick(observation(clock.now())),
            TaskStatus.SUCCEEDED,
        )

    def test_six_step_task_returns_to_landing_zone_before_land(self) -> None:
        context, clock = make_context()
        scripted, registry = make_registry(
            {
                SkillName.GOTO: [
                    succeeded(SkillResultCode.GOAL_REACHED),
                    succeeded(SkillResultCode.GOAL_REACHED),
                ]
            }
        )
        manager = SkillManager(context, registry=registry)
        manager.start_task(six_step_plan())

        for _ in range(4):
            self.assertIs(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertIs(manager.pending_task_result, TaskStatus.SUCCEEDED)
        self.assertEqual(
            [goal.position for goal in scripted[SkillName.GOTO].started_goals],
            [(20.0, 30.0, 10.0), (2.0, -3.0, 10.0)],
        )

        self.assertIs(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(tick_once(manager, clock), TaskStatus.SUCCEEDED)
        self.assertEqual(
            [record.new_skill for record in manager.transition_log],
            [
                SkillName.TAKEOFF,
                SkillName.GOTO,
                SkillName.SEARCH,
                SkillName.TRACK,
                SkillName.GOTO,
                SkillName.LAND,
                None,
            ],
        )
        self.assertEqual(manager.transition_log[4].reason, "track_complete")

    def test_six_step_recovery_still_returns_to_landing_zone(self) -> None:
        lost_data = {
            "target_id": "target_0",
            "last_seen_position": (11.0, 12.0, 0.5),
            "last_seen_velocity": (0.4, -0.2, 0.0),
            "last_seen_time": 3.5,
            "tracking_duration": 1.25,
        }
        context, clock = make_context()
        scripted, registry = make_registry(
            {
                SkillName.GOTO: [
                    succeeded(SkillResultCode.GOAL_REACHED),
                    succeeded(SkillResultCode.GOAL_REACHED),
                ],
                SkillName.TRACK: [
                    failed(SkillResultCode.TARGET_LOST, lost_data),
                    succeeded(SkillResultCode.TRACK_COMPLETE),
                ],
            }
        )
        manager = SkillManager(context, registry=registry)
        manager.start_task(six_step_plan(track_duration=5.0))

        for _ in range(6):
            self.assertIs(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertEqual(len(scripted[SkillName.TRACK].started_goals), 2)
        self.assertEqual(len(scripted[SkillName.GOTO].started_goals), 2)

        self.assertIs(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(tick_once(manager, clock), TaskStatus.SUCCEEDED)
        self.assertEqual(
            [record.new_skill for record in manager.transition_log],
            [
                SkillName.TAKEOFF,
                SkillName.GOTO,
                SkillName.SEARCH,
                SkillName.TRACK,
                SkillName.REACQUIRE,
                SkillName.TRACK,
                SkillName.GOTO,
                SkillName.LAND,
                None,
            ],
        )

    def test_second_goto_failure_skips_to_fail_safe_land(self) -> None:
        context, clock = make_context()
        scripted, registry = make_registry(
            {
                SkillName.GOTO: [
                    succeeded(SkillResultCode.GOAL_REACHED),
                    failed(SkillResultCode.TIMEOUT),
                ]
            }
        )
        manager = SkillManager(context, registry=registry)
        manager.start_task(six_step_plan())

        for _ in range(5):
            self.assertIs(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(manager.pending_task_result, TaskStatus.FAILED)
        self.assertIs(manager.task_failure_result.code, SkillResultCode.TIMEOUT)
        self.assertEqual(len(scripted[SkillName.GOTO].started_goals), 2)
        self.assertEqual(len(scripted[SkillName.LAND].started_goals), 1)
        self.assertEqual(manager.transition_log[-1].reason, "goto_timeout")

        self.assertIs(tick_once(manager, clock), TaskStatus.FAILED)
        self.assertEqual(manager.transition_log[-1].reason, "failure_landing_complete")

    def test_target_lost_reacquires_and_resumes_saved_track_goal(self) -> None:
        lost_data = {
            "target_id": "target_0",
            "last_seen_position": (11.0, 12.0, 0.5),
            "last_seen_velocity": (0.4, -0.2, 0.0),
            "last_seen_time": 3.5,
            "tracking_duration": 1.25,
        }
        overrides = {
            SkillName.TRACK: [
                failed(SkillResultCode.TARGET_LOST, lost_data),
                succeeded(SkillResultCode.TRACK_COMPLETE),
            ],
            SkillName.REACQUIRE: [
                succeeded(
                    SkillResultCode.TARGET_FOUND,
                    {"target_id": "target_0"},
                )
            ],
        }
        context, clock = make_context()
        scripted, registry = make_registry(overrides)
        manager = SkillManager(
            context,
            registry=registry,
            reacquire_search_radius=7.5,
            reacquire_timeout=9.0,
        )
        manager.start_task(standard_plan(track_duration=5.0))

        for _ in range(4):
            self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(manager.active_name, SkillName.REACQUIRE)
        reacquire_goal = scripted[SkillName.REACQUIRE].started_goals[0]
        self.assertIsInstance(reacquire_goal, ReacquireGoal)
        self.assertEqual(reacquire_goal.target_id, "target_0")
        self.assertEqual(reacquire_goal.last_seen_position, (11.0, 12.0, 0.5))
        self.assertEqual(reacquire_goal.last_seen_velocity, (0.4, -0.2, 0.0))
        self.assertEqual(reacquire_goal.last_seen_time, 3.5)
        self.assertEqual(reacquire_goal.search_radius, 7.5)
        self.assertEqual(reacquire_goal.timeout, 9.0)

        self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(manager.active_name, SkillName.TRACK)
        resumed_goal = scripted[SkillName.TRACK].started_goals[1]
        self.assertIsInstance(resumed_goal, TrackGoal)
        self.assertEqual(resumed_goal.target_id, "target_0")
        self.assertAlmostEqual(resumed_goal.track_duration, 3.75)

        self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(manager.active_name, SkillName.LAND)
        self.assertEqual(tick_once(manager, clock), TaskStatus.SUCCEEDED)
        self.assertEqual(
            [record.new_skill for record in manager.transition_log],
            [
                SkillName.TAKEOFF,
                SkillName.GOTO,
                SkillName.SEARCH,
                SkillName.TRACK,
                SkillName.REACQUIRE,
                SkillName.TRACK,
                SkillName.LAND,
                None,
            ],
        )
        self.assertEqual(manager.transition_log[4].reason, "target_lost")
        self.assertIs(
            manager.transition_log[4].result_code,
            SkillResultCode.TARGET_LOST,
        )
        self.assertEqual(manager.transition_log[5].reason, "target_reacquired")

    def test_search_exhausted_lands_before_task_failed(self) -> None:
        context, clock = make_context()
        scripted, registry = make_registry(
            {
                SkillName.SEARCH: [
                    failed(SkillResultCode.SEARCH_EXHAUSTED)
                ]
            }
        )
        manager = SkillManager(context, registry=registry)
        manager.start_task(standard_plan())

        self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.pending_task_result, TaskStatus.FAILED)
        self.assertEqual(manager.task_status, TaskStatus.RUNNING)
        self.assertEqual(tick_once(manager, clock), TaskStatus.FAILED)
        self.assertEqual(manager.task_failure_result.code, SkillResultCode.SEARCH_EXHAUSTED)
        self.assertEqual(len(scripted[SkillName.TRACK].started_goals), 0)
        self.assertEqual(len(scripted[SkillName.REACQUIRE].started_goals), 0)
        self.assertEqual(manager.transition_log[-2].reason, "search_exhausted")
        self.assertEqual(manager.transition_log[-1].reason, "failure_landing_complete")

    def test_reacquire_timeout_lands_before_task_failed(self) -> None:
        lost_data = {
            "target_id": "target_0",
            "last_seen_position": (5.0, 6.0, 0.5),
            "last_seen_velocity": (0.1, 0.0, 0.0),
            "last_seen_time": 2.0,
            "tracking_duration": 0.5,
        }
        context, clock = make_context()
        scripted, registry = make_registry(
            {
                SkillName.TRACK: [
                    failed(SkillResultCode.TARGET_LOST, lost_data)
                ],
                SkillName.REACQUIRE: [failed(SkillResultCode.TIMEOUT)],
            }
        )
        manager = SkillManager(context, registry=registry)
        manager.start_task(standard_plan())

        for _ in range(5):
            self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.pending_task_result, TaskStatus.FAILED)
        self.assertEqual(tick_once(manager, clock), TaskStatus.FAILED)
        self.assertEqual(manager.task_failure_result.code, SkillResultCode.TIMEOUT)
        self.assertEqual(len(scripted[SkillName.TRACK].started_goals), 1)
        self.assertEqual(manager.transition_log[-2].reason, "reacquire_timeout")
        self.assertEqual(manager.transition_log[-1].reason, "failure_landing_complete")

    def test_goto_timeout_lands_before_task_failed(self) -> None:
        context, clock = make_context()
        scripted, registry = make_registry(
            {SkillName.GOTO: [failed(SkillResultCode.TIMEOUT)]}
        )
        manager = SkillManager(context, registry=registry)
        manager.start_task(standard_plan())

        self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.pending_task_result, TaskStatus.FAILED)
        # A later cancel request must not hide the already-recorded failure.
        self.assertIs(manager.cancel_task(), TaskStatus.RUNNING)
        self.assertIs(manager.pending_task_result, TaskStatus.FAILED)
        self.assertEqual(tick_once(manager, clock), TaskStatus.FAILED)
        self.assertEqual(manager.task_failure_result.code, SkillResultCode.TIMEOUT)
        self.assertEqual(len(scripted[SkillName.SEARCH].started_goals), 0)
        self.assertEqual(len(scripted[SkillName.TRACK].started_goals), 0)
        self.assertEqual(manager.transition_log[-2].reason, "goto_timeout")
        self.assertEqual(manager.transition_log[-1].reason, "failure_landing_complete")

    def test_cancel_during_land_finishes_landing_then_commits_canceled(self) -> None:
        context, clock = make_context()
        _, registry = make_registry()
        manager = SkillManager(context, registry=registry)
        manager.start_task(standard_plan())
        for _ in range(4):
            self.assertIs(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)

        self.assertIs(manager.cancel_task(), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(manager.pending_task_result, TaskStatus.CANCELED)
        self.assertIs(tick_once(manager, clock), TaskStatus.CANCELED)
        self.assertEqual(manager.transition_log[-1].reason, "cancel_landing_complete")

    def test_task_mode_rejects_external_active_reset(self) -> None:
        context, _ = make_context()
        _, registry = make_registry()
        manager = SkillManager(context, registry=registry)
        manager.start_task(standard_plan())
        with self.assertRaises(SkillManagerError):
            manager.reset_active()
        with self.assertRaises(SkillManagerError):
            manager.cancel_active()

    def test_reset_hook_failure_still_enters_fail_safe_land(self) -> None:
        context, clock = make_context()
        scripted, registry = make_registry()
        broken = ResetErrorScriptedSkill(
            succeeded(SkillResultCode.GOAL_REACHED)
        )
        scripted[SkillName.GOTO] = broken
        registry[SkillName.GOTO] = broken
        manager = SkillManager(context, registry=registry)
        manager.start_task(standard_plan())

        self.assertIs(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertIs(tick_once(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(manager.pending_task_result, TaskStatus.FAILED)
        self.assertIs(manager.task_failure_result.code, SkillResultCode.INTERNAL_ERROR)
        self.assertEqual(manager.transition_log[-1].reason, "skill_reset_failed")
        self.assertIs(tick_once(manager, clock), TaskStatus.FAILED)

    def test_cancel_reset_hook_failure_still_enters_land(self) -> None:
        context, clock = make_context()
        _, registry = make_registry()
        broken = ResetErrorScriptedSkill(
            succeeded(SkillResultCode.TAKEOFF_COMPLETE)
        )
        registry[SkillName.TAKEOFF] = broken
        manager = SkillManager(context, registry=registry)
        manager.start_task(standard_plan())

        self.assertIs(manager.cancel_task(), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(manager.pending_task_result, TaskStatus.FAILED)
        self.assertIs(manager.task_failure_result.code, SkillResultCode.INTERNAL_ERROR)
        self.assertEqual(manager.transition_log[-1].reason, "canceled_skill_reset_failed")
        self.assertIs(tick_once(manager, clock), TaskStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
