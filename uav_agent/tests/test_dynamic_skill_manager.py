from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.base import Skill
from skills.manager import ExecutionKind, SkillManager, TaskStatus
from skills.plan import RecoveryPolicy, StepOutputRef, TaskPlan, TaskStep
from skills.reacquire import ReacquireGoal
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
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


class ManualClock:
    def __init__(self) -> None:
        self.time_s = 0.0

    def now(self) -> float:
        return self.time_s

    def advance(self) -> None:
        self.time_s += 1.0


@dataclass(frozen=True, slots=True)
class Outcome:
    status: SkillStatus
    code: SkillResultCode
    data: dict[str, object] = field(default_factory=dict)


class CountingScriptedSkill(Skill):
    goal_type = SkillGoal

    def __init__(self, *outcomes: Outcome) -> None:
        super().__init__()
        self.outcomes = deque(outcomes)
        self.started_goals: list[SkillGoal] = []
        self.tick_count = 0

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        self.started_goals.append(deepcopy(goal))

    def _on_tick(self, observation: Observation) -> None:
        self.tick_count += 1
        outcome = self.outcomes.popleft()
        if outcome.status is SkillStatus.SUCCEEDED:
            self._succeed(outcome.code, "ok", outcome.data)
        else:
            self._fail(outcome.code, "failed", outcome.data)


def ok(code: SkillResultCode, **data: object) -> Outcome:
    return Outcome(SkillStatus.SUCCEEDED, code, data)


def fail(code: SkillResultCode, **data: object) -> Outcome:
    return Outcome(SkillStatus.FAILED, code, data)


def lost(*, tracking_duration: float = 1.0) -> Outcome:
    return fail(
        SkillResultCode.TARGET_LOST,
        target_id="target_7",
        last_seen_position=(4.0, 5.0, 0.0),
        last_seen_velocity=(0.2, 0.0, 0.0),
        last_seen_time=1.0,
        tracking_duration=tracking_duration,
    )


def context() -> tuple[SkillContext, ManualClock]:
    clock = ManualClock()
    return (
        SkillContext(
            uav=KinematicUAV(
                UAVState(0.0, 0.0, 0.0, 0.0),
                max_speed_mps=5.0,
                max_yaw_rate_rad_s=2.0,
            ),
            camera=FakeCamera(),
            perception=None,
            clock=clock,
        ),
        clock,
    )


def observation(timestamp: float) -> Observation:
    return Observation(
        timestamp=timestamp,
        uav_pose=UAVState(0.0, 0.0, 8.0, 0.0),
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def tick(manager: SkillManager, clock: ManualClock) -> TaskStatus:
    clock.advance()
    result = manager.tick(observation(clock.now()))
    assert isinstance(result, TaskStatus)
    return result


def registry(
    *,
    goto_count: int = 1,
    track: list[Outcome] | None = None,
    reacquire: list[Outcome] | None = None,
) -> tuple[dict[SkillName, CountingScriptedSkill], dict[SkillName, Skill]]:
    outcomes = {
        SkillName.TAKEOFF: [ok(SkillResultCode.TAKEOFF_COMPLETE)],
        SkillName.GOTO: [ok(SkillResultCode.GOAL_REACHED)] * goto_count,
        SkillName.SEARCH: [
            ok(SkillResultCode.TARGET_FOUND, target_id="target_7")
        ],
        SkillName.TRACK: track or [ok(SkillResultCode.TRACK_COMPLETE)],
        SkillName.REACQUIRE: reacquire
        or [ok(SkillResultCode.TARGET_FOUND, target_id="target_7")],
        SkillName.LAND: [ok(SkillResultCode.LAND_COMPLETE)],
    }
    scripted = {
        name: CountingScriptedSkill(*values) for name, values in outcomes.items()
    }
    return scripted, dict(scripted)


def navigation_plan() -> TaskPlan:
    return TaskPlan(
        (
            TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 8.0}),
            TaskStep("goto_a", SkillName.GOTO, {"position": (2.0, 0.0, 8.0)}),
            TaskStep("goto_b", SkillName.GOTO, {"position": (3.0, 1.0, 8.0)}),
            TaskStep("land", SkillName.LAND, {}),
        )
    )


def tracking_plan(recovery: RecoveryPolicy | None) -> TaskPlan:
    return TaskPlan(
        (
            TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 8.0}),
            TaskStep(
                "search",
                SkillName.SEARCH,
                {
                    "center": (5.0, 5.0, 0.0),
                    "radius": 3.0,
                    "target_description": "moving target",
                    "search_altitude": 8.0,
                },
            ),
            # SEARCH and TRACK deliberately need not be adjacent.
            TaskStep("goto_mid", SkillName.GOTO, {"position": (5.0, 1.0, 8.0)}),
            TaskStep(
                "track",
                SkillName.TRACK,
                {
                    "target_id": StepOutputRef("search"),
                    "track_duration": 10.0,
                },
                recovery,
            ),
            TaskStep("land", SkillName.LAND, {}),
        )
    )


class DynamicSkillManagerTests(unittest.TestCase):
    def test_start_owns_plan_snapshot_against_future_param_mutation(self) -> None:
        plan = navigation_plan()
        ctx, clock = context()
        scripted, skills = registry(goto_count=2)
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(plan)

        # Mutate both the caller's original plan and a value returned by the
        # Manager's inspection property before the future GOTO is started.
        plan.steps[1].params["position"] = (99.0, 99.0, 99.0)
        exposed = manager.task_plan
        self.assertIsNotNone(exposed)
        exposed.steps[1].params["position"] = (-99.0, -99.0, -99.0)

        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        first_goto = scripted[SkillName.GOTO].started_goals[0]
        self.assertEqual(first_goto.position, (2.0, 0.0, 8.0))
        self.assertEqual(
            manager.task_plan.steps[1].params["position"],
            (2.0, 0.0, 8.0),
        )

    def test_multiple_gotos_execute_in_order_and_one_skill_ticks_per_frame(self) -> None:
        ctx, clock = context()
        scripted, skills = registry(goto_count=2)
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(navigation_plan())

        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertEqual(scripted[SkillName.TAKEOFF].tick_count, 1)
        self.assertEqual(scripted[SkillName.GOTO].tick_count, 0)
        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(tick(manager, clock), TaskStatus.SUCCEEDED)

        self.assertEqual(
            [goal.position for goal in scripted[SkillName.GOTO].started_goals],
            [(2.0, 0.0, 8.0), (3.0, 1.0, 8.0)],
        )
        self.assertEqual(
            [record.new_step_id for record in manager.transition_log],
            ["takeoff", "goto_a", "goto_b", "land", None],
        )

    def test_search_output_is_saved_and_track_reference_resolves_by_step(self) -> None:
        ctx, clock = context()
        scripted, skills = registry()
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(tracking_plan(None))
        for _ in range(4):
            self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.step_outputs["search"]["target_id"], "target_7")
        goal = scripted[SkillName.TRACK].started_goals[0]
        self.assertIsInstance(goal, TrackGoal)
        self.assertEqual(goal.target_id, "target_7")
        self.assertIs(tick(manager, clock), TaskStatus.SUCCEEDED)

    def test_dynamic_recovery_disabled_lands_and_fails(self) -> None:
        ctx, clock = context()
        _, skills = registry(track=[lost()])
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(tracking_plan(None))
        for _ in range(4):
            self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.recovery_attempts, {})
        self.assertEqual(manager.transition_log[-1].reason, "target_lost_recovery_unavailable")
        self.assertIs(tick(manager, clock), TaskStatus.FAILED)

    def test_bounded_recovery_resumes_same_track_and_records_attempt(self) -> None:
        policy = RecoveryPolicy(SkillName.REACQUIRE, 2, 7.0, 11.0)
        ctx, clock = context()
        scripted, skills = registry(
            track=[lost(tracking_duration=2.0), ok(SkillResultCode.TRACK_COMPLETE)]
        )
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(tracking_plan(policy))
        for _ in range(4):
            self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.REACQUIRE)
        self.assertIs(manager.active_execution_kind, ExecutionKind.RECOVERY)
        reacquire_goal = scripted[SkillName.REACQUIRE].started_goals[0]
        self.assertIsInstance(reacquire_goal, ReacquireGoal)
        self.assertEqual(reacquire_goal.search_radius, 7.0)
        self.assertEqual(reacquire_goal.timeout, 11.0)
        self.assertEqual(manager.transition_log[-1].old_step_id, "track")
        self.assertEqual(manager.transition_log[-1].new_step_id, "track")
        self.assertEqual(manager.transition_log[-1].recovery_attempt, 1)

        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.TRACK)
        resumed = scripted[SkillName.TRACK].started_goals[1]
        self.assertAlmostEqual(resumed.track_duration, 8.0)
        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(tick(manager, clock), TaskStatus.SUCCEEDED)

    def test_recovery_attempts_exhaust_without_looping(self) -> None:
        policy = RecoveryPolicy(SkillName.REACQUIRE, 2, 10.0, 30.0)
        ctx, clock = context()
        scripted, skills = registry(
            track=[lost(), lost(), lost()],
            reacquire=[
                ok(SkillResultCode.TARGET_FOUND, target_id="target_7"),
                ok(SkillResultCode.TARGET_FOUND, target_id="target_7"),
            ],
        )
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(tracking_plan(policy))
        # takeoff, search, goto, track-loss, reacquire, track-loss,
        # reacquire, track-loss -> fail-safe LAND
        for _ in range(8):
            self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.recovery_attempts["track"], 2)
        self.assertEqual(len(scripted[SkillName.REACQUIRE].started_goals), 2)
        self.assertEqual(manager.transition_log[-1].recovery_attempt, 2)
        self.assertIs(tick(manager, clock), TaskStatus.FAILED)

    def test_reacquire_cannot_silently_switch_to_a_decoy_target(self) -> None:
        policy = RecoveryPolicy(SkillName.REACQUIRE, 1, 10.0, 30.0)
        ctx, clock = context()
        _, skills = registry(
            track=[lost()],
            reacquire=[
                ok(SkillResultCode.TARGET_FOUND, target_id="decoy_target")
            ],
        )
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(tracking_plan(policy))
        for _ in range(4):
            self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.REACQUIRE)
        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.active_target_id, "target_7")
        self.assertEqual(manager.transition_log[-1].reason, "reacquire_target_mismatch")
        self.assertIs(manager.task_failure_result.code, SkillResultCode.INTERNAL_ERROR)
        self.assertIs(tick(manager, clock), TaskStatus.FAILED)

    def test_reacquire_success_must_explicitly_return_target_id(self) -> None:
        policy = RecoveryPolicy(SkillName.REACQUIRE, 1, 10.0, 30.0)
        ctx, clock = context()
        scripted, skills = registry(
            track=[lost()],
            reacquire=[ok(SkillResultCode.TARGET_FOUND)],
        )
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(tracking_plan(policy))
        for _ in range(4):
            self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.REACQUIRE)

        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.active_target_id, "target_7")
        self.assertEqual(len(scripted[SkillName.TRACK].started_goals), 1)
        self.assertEqual(manager.transition_log[-1].reason, "reacquire_target_invalid")
        self.assertIs(manager.task_failure_result.code, SkillResultCode.INTERNAL_ERROR)
        self.assertIs(tick(manager, clock), TaskStatus.FAILED)

    def test_track_loss_cannot_replace_identity_before_reacquire(self) -> None:
        policy = RecoveryPolicy(SkillName.REACQUIRE, 1, 10.0, 30.0)
        decoy_loss = fail(
            SkillResultCode.TARGET_LOST,
            target_id="decoy_target",
            last_seen_position=(4.0, 5.0, 0.0),
            last_seen_velocity=(0.2, 0.0, 0.0),
            last_seen_time=1.0,
            tracking_duration=1.0,
        )
        ctx, clock = context()
        scripted, skills = registry(track=[decoy_loss])
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(tracking_plan(policy))
        for _ in range(4):
            self.assertIs(tick(manager, clock), TaskStatus.RUNNING)

        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.active_target_id, "target_7")
        self.assertEqual(len(scripted[SkillName.REACQUIRE].started_goals), 0)
        self.assertEqual(manager.recovery_attempts, {"track": 0})
        self.assertEqual(manager.transition_log[-1].reason, "track_lost_target_mismatch")
        self.assertIs(manager.task_failure_result.code, SkillResultCode.INTERNAL_ERROR)
        self.assertIs(tick(manager, clock), TaskStatus.FAILED)

    def test_reset_and_new_task_do_not_retain_runtime_outputs(self) -> None:
        policy = RecoveryPolicy(SkillName.REACQUIRE, 1, 10.0, 30.0)
        ctx, clock = context()
        _, skills = registry()
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(tracking_plan(policy))
        for _ in range(5):
            tick(manager, clock)
        self.assertIs(manager.task_status, TaskStatus.SUCCEEDED)
        self.assertTrue(manager.step_outputs)
        self.assertEqual(manager.recovery_attempts, {"track": 0})
        self.assertIsNotNone(manager.last_result)

        manager.reset_task()
        self.assertIs(manager.task_status, TaskStatus.IDLE)
        self.assertEqual(manager.step_outputs, {})
        self.assertEqual(manager.recovery_attempts, {})
        self.assertIsNone(manager.last_result)
        manager.start_task(navigation_plan())
        self.assertEqual(manager.step_outputs, {})
        self.assertEqual(manager.recovery_attempts, {})
        self.assertIsNone(manager.last_result)

    def test_cancel_skips_to_fail_safe_land(self) -> None:
        ctx, clock = context()
        _, skills = registry(goto_count=2)
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(navigation_plan())
        self.assertIs(manager.cancel_task(), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.transition_log[-1].new_step_id, "land")
        self.assertIs(tick(manager, clock), TaskStatus.CANCELED)

    def test_plan_ending_without_land_uses_default_fail_safe_land(self) -> None:
        plan = TaskPlan(
            (
                TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 8.0}),
                TaskStep("goto", SkillName.GOTO, {"position": (1.0, 0.0, 8.0)}),
            )
        )
        ctx, clock = context()
        _, skills = registry()
        manager = SkillManager(ctx, registry=skills)
        manager.start_task(plan)
        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIsNone(manager.transition_log[-1].new_step_id)
        self.assertEqual(manager.transition_log[-1].reason, "plan_ended_without_land")
        self.assertIs(manager.task_failure_result.code, SkillResultCode.INTERNAL_ERROR)
        self.assertIs(tick(manager, clock), TaskStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
