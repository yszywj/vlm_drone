from __future__ import annotations

import unittest
from dataclasses import dataclass, field

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from skills.base import Skill, SkillLifecycleError
from skills.goto import GotoGoal, GotoSkill
from skills.land import LandGoal, LandSkill
from skills.manager import SkillManager, SkillManagerError
from skills.motion_types import MotionPolicy
from skills.reacquire import ReacquireGoal, ReacquireSkill
from skills.search import SearchGoal, SearchSkill
from skills.takeoff import TakeoffGoal, TakeoffSkill
from skills.track import TrackGoal, TrackSkill
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillName,
    SkillResult,
    SkillResultCode,
    SkillStatus,
)


class FakeCamera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


class FakeClock:
    def __init__(self) -> None:
        self.time_s = 0.0

    def now(self) -> float:
        return self.time_s


@dataclass(frozen=True, slots=True)
class ScriptedGoal(SkillGoal):
    outcome: str = "success"
    motion_policy: MotionPolicy = field(default_factory=MotionPolicy)


class ScriptedSkill(Skill):
    goal_type = ScriptedGoal

    def _on_tick(self, observation: Observation) -> None:
        goal = self._active_goal
        if not isinstance(goal, ScriptedGoal):
            raise AssertionError("typed goal was not retained")
        if goal.outcome == "success":
            self._set_feedback(
                1.0,
                "goal reached",
                {"distance_remaining": 0.0, "samples": [1, 2]},
            )
            self._succeed(
                SkillResultCode.GOAL_REACHED,
                "goal reached",
                {"final_x": observation.uav_pose.x, "route": [0.0, 1.0]},
            )
        elif goal.outcome == "failure":
            self._fail(SkillResultCode.TIMEOUT, "timed out")
        elif goal.outcome == "bad_result_code":
            self._succeed(SkillResultCode.TIMEOUT, "inconsistent result")


class CancelHookErrorSkill(ScriptedSkill):
    def _on_cancel(self) -> None:
        raise RuntimeError("cancel cleanup failed")


class ResetHookErrorSkill(ScriptedSkill):
    def _on_reset(self) -> None:
        raise RuntimeError("reset cleanup failed")


class ConcreteTrackSkill(TrackSkill):
    def _on_tick(self, observation: Observation) -> None:
        pass


class CompleteThenRaiseSkill(ScriptedSkill):
    def _on_tick(self, observation: Observation) -> None:
        self._succeed(SkillResultCode.GOAL_REACHED, "premature success")
        raise RuntimeError("post-completion failure")


class ImmediateCompleteSkill(ScriptedSkill):
    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        self._succeed(SkillResultCode.GOAL_REACHED, "completed during start")


class StartInterruptSkill(ScriptedSkill):
    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        raise KeyboardInterrupt("simulated external interruption")


class CancelDuringTickSkill(ScriptedSkill):
    def _on_tick(self, observation: Observation) -> None:
        self.cancel()


def make_context() -> SkillContext:
    return SkillContext(
        uav=KinematicUAV(UAVState(0.0, 0.0, 1.0, 0.3), 5.0, 1.0),
        camera=FakeCamera(),
        perception=None,
        clock=FakeClock(),
    )


def make_observation() -> Observation:
    return Observation(
        timestamp=0.1,
        uav_pose=UAVState(0.0, 0.0, 1.0, 0.3),
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def make_concrete_contract(contract_type: type[Skill]) -> Skill:
    def _on_tick(self: Skill, observation: Observation) -> None:
        pass

    concrete_type = type(
        f"Concrete{contract_type.__name__}",
        (contract_type,),
        {"_on_tick": _on_tick},
    )
    return concrete_type()


class SkillBaseTest(unittest.TestCase):
    def test_six_typed_goal_contracts_accept_valid_goals(self) -> None:
        cases = (
            (TakeoffSkill, TakeoffGoal(target_altitude=5.0)),
            (GotoSkill, GotoGoal(position=(1.0, 2.0, 3.0))),
            (
                SearchSkill,
                SearchGoal(
                    center=(0.0, 0.0, 0.5),
                    radius=5.0,
                    target_description="person",
                    search_altitude=5.0,
                ),
            ),
            (TrackSkill, TrackGoal(target_id="person-1")),
            (
                ReacquireSkill,
                ReacquireGoal(
                    target_id="person-1",
                    last_seen_position=(1.0, 2.0, 0.5),
                    last_seen_velocity=(0.1, 0.0, 0.0),
                    last_seen_time=0.0,
                    search_radius=3.0,
                ),
            ),
            (LandSkill, LandGoal()),
        )
        for contract_type, goal in cases:
            with self.subTest(contract=contract_type.__name__):
                skill = make_concrete_contract(contract_type)
                skill.start(goal, make_context())
                self.assertIs(skill.status, SkillStatus.RUNNING)
                skill.cancel()
                skill.reset()

    def test_six_typed_goal_contracts_reject_representative_bad_goals(self) -> None:
        cases = (
            (TakeoffSkill, TakeoffGoal(target_altitude=float("nan"))),
            (
                GotoSkill,
                GotoGoal(position=(True, 2.0, 3.0)),  # type: ignore[arg-type]
            ),
            (
                SearchSkill,
                SearchGoal(
                    center=(0.0, 0.0, 0.5),
                    radius=0.0,
                    target_description="person",
                    search_altitude=5.0,
                ),
            ),
            (TrackSkill, TrackGoal(target_id=123)),  # type: ignore[arg-type]
            (
                ReacquireSkill,
                ReacquireGoal(
                    target_id="person-1",
                    last_seen_position=(1.0, float("inf"), 0.5),
                    last_seen_velocity=(0.1, 0.0, 0.0),
                    last_seen_time=0.0,
                    search_radius=3.0,
                ),
            ),
            (LandSkill, LandGoal(timeout=0.0)),
        )
        for contract_type, goal in cases:
            with self.subTest(contract=contract_type.__name__):
                skill = make_concrete_contract(contract_type)
                skill.start(goal, make_context())
                self.assertIs(skill.status, SkillStatus.FAILED)
                self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)
                skill.reset()

    def test_initial_start_success_and_reset_lifecycle(self) -> None:
        skill = ScriptedSkill()
        self.assertIs(skill.status, SkillStatus.IDLE)
        self.assertIsNone(skill.get_result())

        skill.start(ScriptedGoal(outcome="success"), make_context())
        self.assertIs(skill.status, SkillStatus.RUNNING)
        self.assertAlmostEqual(skill.initial_yaw, 0.3)

        self.assertIs(skill.tick(make_observation()), SkillStatus.SUCCEEDED)
        result = skill.get_result()
        self.assertIsNotNone(result)
        self.assertIs(result.status, SkillStatus.SUCCEEDED)
        self.assertIs(result.code, SkillResultCode.GOAL_REACHED)
        self.assertEqual(skill.get_feedback().progress, 1.0)

        result.data["route"].append(2.0)
        feedback = skill.get_feedback()
        feedback.data["samples"].append(3)
        self.assertEqual(skill.get_result().data["route"], [0.0, 1.0])
        self.assertEqual(skill.get_feedback().data["samples"], [1, 2])
        self.assertEqual(result.to_dict()["status"], "SUCCEEDED")
        self.assertEqual(result.to_dict()["code"], "GOAL_REACHED")

        skill.reset()
        self.assertIs(skill.status, SkillStatus.IDLE)
        self.assertIsNone(skill.get_result())

    def test_failure_has_failed_status_and_result_code(self) -> None:
        skill = ScriptedSkill()
        skill.start(ScriptedGoal(outcome="failure"), make_context())
        self.assertIs(skill.tick(make_observation()), SkillStatus.FAILED)
        result = skill.get_result()
        self.assertIsNotNone(result)
        self.assertIs(result.status, SkillStatus.FAILED)
        self.assertIs(result.code, SkillResultCode.TIMEOUT)

    def test_inconsistent_terminal_code_becomes_internal_error(self) -> None:
        skill = ScriptedSkill()
        skill.start(ScriptedGoal(outcome="bad_result_code"), make_context())
        self.assertIs(skill.tick(make_observation()), SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INTERNAL_ERROR)

        with self.assertRaises(ValueError):
            SkillResult(
                status=SkillStatus.SUCCEEDED,
                code=SkillResultCode.TIMEOUT,
                message="invalid pair",
            )

    def test_exception_after_completion_overrides_success(self) -> None:
        skill = CompleteThenRaiseSkill()
        skill.start(ScriptedGoal(), make_context())
        self.assertIs(skill.tick(make_observation()), SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INTERNAL_ERROR)

    def test_valid_start_cannot_skip_running_and_complete_immediately(self) -> None:
        skill = ImmediateCompleteSkill()
        skill.start(ScriptedGoal(), make_context())
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INTERNAL_ERROR)

    def test_tick_cannot_impersonate_external_cancel(self) -> None:
        skill = CancelDuringTickSkill()
        skill.start(ScriptedGoal(), make_context())
        self.assertIs(skill.tick(make_observation()), SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INTERNAL_ERROR)

    def test_bad_track_id_is_invalid_goal_not_internal_error(self) -> None:
        skill = ConcreteTrackSkill()
        skill.start(TrackGoal(target_id=123), make_context())  # type: ignore[arg-type]
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)

    def test_malformed_observation_is_invalid_state(self) -> None:
        skill = ScriptedSkill()
        skill.start(ScriptedGoal(outcome="running"), make_context())
        bad_observation = Observation(
            timestamp=float("nan"),
            uav_pose=UAVState(0.0, 0.0, 1.0, 0.0),
            uav_velocity=np.zeros(3),
            camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        self.assertIs(skill.tick(bad_observation), SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_STATE)

    def test_cancel_transitions_to_canceled(self) -> None:
        context = make_context()
        context.uav.set_velocity([1.0, 0.0, 0.0])
        skill = ScriptedSkill()
        skill.start(ScriptedGoal(outcome="running"), context)
        skill.cancel()
        self.assertIs(skill.status, SkillStatus.CANCELED)
        self.assertIs(skill.get_result().code, SkillResultCode.CANCELED)
        np.testing.assert_array_equal(context.uav.get_velocity(), np.zeros(3))

    def test_cancel_cleanup_error_still_transitions_to_canceled(self) -> None:
        skill = CancelHookErrorSkill()
        skill.start(ScriptedGoal(outcome="running"), make_context())
        skill.cancel()
        result = skill.get_result()
        self.assertIs(skill.status, SkillStatus.CANCELED)
        self.assertIs(result.code, SkillResultCode.CANCELED)
        self.assertIn("cleanup_errors", result.data)

    def test_reset_cleanup_error_still_returns_to_idle(self) -> None:
        skill = ResetHookErrorSkill()
        skill.start(ScriptedGoal(outcome="success"), make_context())
        skill.tick(make_observation())
        with self.assertRaises(SkillLifecycleError):
            skill.reset()
        self.assertIs(skill.status, SkillStatus.IDLE)
        self.assertIsNone(skill.get_result())

    def test_illegal_lifecycle_transitions_are_rejected(self) -> None:
        skill = ScriptedSkill()
        with self.assertRaises(SkillLifecycleError):
            skill.tick(make_observation())
        with self.assertRaises(SkillLifecycleError):
            skill.cancel()
        with self.assertRaises(SkillLifecycleError):
            skill.reset()

        skill.start(ScriptedGoal(outcome="running"), make_context())
        with self.assertRaises(SkillLifecycleError):
            skill.start(ScriptedGoal(), make_context())
        with self.assertRaises(SkillLifecycleError):
            skill.reset()

        skill.cancel()
        with self.assertRaises(SkillLifecycleError):
            skill.start(ScriptedGoal(), make_context())
        with self.assertRaises(SkillLifecycleError):
            skill.tick(make_observation())
        with self.assertRaises(SkillLifecycleError):
            skill.cancel()

    def test_manager_requires_terminal_reset_before_reuse(self) -> None:
        manager = SkillManager(make_context())
        skill = ScriptedSkill()
        manager.register(SkillName.GOTO, skill)
        self.assertEqual(manager.available_skills(), (SkillName.GOTO,))
        self.assertIs(
            manager.start(SkillName.GOTO, ScriptedGoal(outcome="success")),
            SkillStatus.RUNNING,
        )
        self.assertIs(manager.tick(make_observation()), SkillStatus.SUCCEEDED)
        with self.assertRaises(SkillManagerError):
            manager.start(SkillName.GOTO, ScriptedGoal())

        manager.reset_active()
        self.assertIsNone(manager.active_name)
        self.assertIs(manager.last_result.code, SkillResultCode.GOAL_REACHED)
        self.assertIs(
            manager.start("goto", ScriptedGoal(outcome="running")),
            SkillStatus.RUNNING,
        )
        manager.cancel_active()
        manager.reset_active()

    def test_manager_rejects_duplicate_skill_instance(self) -> None:
        manager = SkillManager(make_context())
        skill = ScriptedSkill()
        manager.register(SkillName.GOTO, skill)
        with self.assertRaises(SkillManagerError):
            manager.register(SkillName.SEARCH, skill)

    def test_manager_releases_active_skill_when_reset_cleanup_fails(self) -> None:
        manager = SkillManager(make_context())
        manager.register(SkillName.GOTO, ResetHookErrorSkill())
        manager.start(SkillName.GOTO, ScriptedGoal(outcome="success"))
        manager.tick(make_observation())
        with self.assertRaises(SkillLifecycleError):
            manager.reset_active()
        self.assertIsNone(manager.active_name)
        self.assertIs(manager.last_result.code, SkillResultCode.GOAL_REACHED)

    def test_manager_retains_ownership_when_start_is_interrupted(self) -> None:
        manager = SkillManager(make_context())
        manager.register(SkillName.GOTO, StartInterruptSkill())
        with self.assertRaises(KeyboardInterrupt):
            manager.start(SkillName.GOTO, ScriptedGoal())
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertIs(manager.active_status, SkillStatus.RUNNING)
        manager.cancel_active()
        manager.reset_active()


if __name__ == "__main__":
    unittest.main()
