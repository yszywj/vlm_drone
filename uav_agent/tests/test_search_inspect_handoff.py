"""Narrow trusted SEARCH-candidate to INSPECT Manager handoff tests."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from env.moving_target import TargetState
from skills.base import Skill
from skills.hover import HoverSkill
from skills.inspect import InspectGoal
from skills.manager import SkillManager, SkillManagerError, TaskStatus
from skills.plan import TaskPlan
from skills.search import SearchSkill
from skills.track import TrackGoal
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
        self._outcomes = deque(outcomes)
        self.started_goals: list[SkillGoal] = []

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        del context
        self.started_goals.append(deepcopy(goal))

    def _on_tick(self, observation: Observation) -> None:
        del observation
        outcome = self._outcomes.popleft()
        if outcome.status is SkillStatus.SUCCEEDED:
            self._succeed(outcome.code, "done", outcome.data)
        else:
            self._fail(outcome.code, "failed", outcome.data)


def _success(code: SkillResultCode, **data: object) -> _Outcome:
    return _Outcome(SkillStatus.SUCCEEDED, code, data)


def _plan(
    *,
    version: int,
    candidate_id: str = "candidate_1",
    change_search: bool = False,
    include_inspect: bool = True,
    include_track: bool = True,
) -> TaskPlan:
    search_radius = 6.0 if change_search else 5.0
    entries: list[dict[str, object]] = [
        {"id": "goto_search", "skill": "GOTO", "position": [1.0, 0.0, 5.0]},
        {
            "id": "search",
            "skill": "SEARCH",
            "center": [0.0, 0.0, 5.0],
            "radius": search_radius,
            "target_description": "red moving target",
            "search_altitude": 5.0,
        },
    ]
    if include_inspect:
        entries.append(
            {
                "id": "inspect_candidate",
                "skill": "INSPECT",
                "candidate_id": candidate_id,
            }
        )
    if include_track:
        entries.append(
            {
                "id": "track",
                "skill": "TRACK",
                "target_id": "$search.target_id",
                "track_duration": 1.0,
            }
        )
    entries.append({"id": "land", "skill": "LAND"})
    return TaskPlan.from_dicts(
        entries,
        mission_id="mission_candidate_handoff",
        uav_id="uav_1",
        plan_version=version,
    )


class SearchInspectHandoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        context = SkillContext(
            KinematicUAV(
                UAVState(0.0, 0.0, 5.0, 0.0),
                max_speed_mps=5.0,
                max_yaw_rate_rad_s=2.0,
            ),
            _Camera(),
            None,
            self.clock,
            uav_id="uav_1",
        )
        self.goto = _ScriptedSkill(_success(SkillResultCode.GOAL_REACHED))
        self.search = SearchSkill()
        self.inspect = _ScriptedSkill(_success(SkillResultCode.GOAL_REACHED))
        self.track = _ScriptedSkill(_success(SkillResultCode.TRACK_COMPLETE))
        self.land = _ScriptedSkill(_success(SkillResultCode.LAND_COMPLETE))
        self.manager = SkillManager(
            context,
            registry={
                SkillName.GOTO: self.goto,
                SkillName.SEARCH: self.search,
                SkillName.HOVER: HoverSkill(),
                SkillName.INSPECT: self.inspect,
                SkillName.TRACK: self.track,
                SkillName.LAND: self.land,
            },
        )
        self.original = _plan(version=1, include_inspect=False)
        self.manager.start_task(self.original)
        self.manager.tick(self._observation())
        self.assertIs(self.manager.active_name, SkillName.SEARCH)
        self.manager.report_candidate_pending("candidate_1", source="qwen_vl")
        self.manager.interrupt_with_hover("candidate_confirmation")

    def _observation(self, *, target_visible: bool | None = None) -> Observation:
        return Observation(
            timestamp=self.clock.value,
            uav_pose=UAVState(0.0, 0.0, 5.0, 0.0),
            uav_velocity=np.zeros(3),
            camera_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            camera_position_m=(
                None if target_visible is not True else np.zeros(3)
            ),
            camera_orientation_wxyz=(
                None
                if target_visible is not True
                else np.asarray((1.0, 0.0, 0.0, 0.0))
            ),
            oracle_target_id=(
                "real_target" if target_visible is True else None
            ),
            oracle_target_visible=target_visible,
            oracle_target_pose=(
                TargetState(1.0, 0.0, 0.0, 0.0)
                if target_visible is True
                else None
            ),
            uav_id="uav_1",
        )

    def test_candidate_handoff_preserves_prefix_and_starts_matching_inspect(self) -> None:
        completed_prefix = deepcopy(self.manager.step_outputs["goto_search"])
        original_search = self.original.steps[1].to_dict()
        revised = _plan(version=2)

        self.manager.handoff_interrupted_search_candidate_to_inspect(
            revised,
            candidate_id="candidate_1",
            source="qwen_vl",
        )
        self.manager.tick(self._observation())

        self.assertIs(self.manager.task_status, TaskStatus.RUNNING)
        self.assertIs(self.manager.active_name, SkillName.INSPECT)
        self.assertEqual(self.manager.task_plan.plan_version, 2)  # type: ignore[union-attr]
        self.assertEqual(
            self.manager.task_plan.steps[1].to_dict(),  # type: ignore[union-attr]
            original_search,
        )
        self.assertEqual(self.manager.step_outputs["goto_search"], completed_prefix)
        self.assertNotIn("search", self.manager.step_outputs)
        self.assertIsNone(self.manager.active_target_id)
        self.assertIsInstance(self.inspect.started_goals[-1], InspectGoal)
        self.assertEqual(
            self.inspect.started_goals[-1].candidate_id,  # type: ignore[attr-defined]
            "candidate_1",
        )
        self.assertIn(
            "search_candidate_handoff_to_inspect",
            tuple(record.reason for record in self.manager.transition_log),
        )

        # INSPECT only contributes evidence, then Manager restarts the exact
        # saved SEARCH. It does not complete SEARCH or fabricate target_id.
        self.manager.tick(self._observation())
        self.assertIs(self.manager.active_name, SkillName.SEARCH)
        self.assertNotIn("search", self.manager.step_outputs)
        self.assertIn("inspect_candidate", self.manager.step_outputs)
        self.assertIn(
            "inspection_evidence_collected_search_resumed",
            tuple(record.reason for record in self.manager.transition_log),
        )

        # Only the real SEARCH result resolves TRACK's output reference; the
        # already-consumed INSPECT is skipped and cannot execute twice.
        self.manager.tick(self._observation(target_visible=True))
        self.assertIs(self.manager.active_name, SkillName.TRACK)
        self.assertEqual(self.manager.step_outputs["search"]["target_id"], "real_target")
        self.assertEqual(len(self.inspect.started_goals), 1)
        self.assertIsInstance(self.track.started_goals[-1], TrackGoal)
        self.assertEqual(self.track.started_goals[-1].target_id, "real_target")  # type: ignore[attr-defined]
        self.assertIn(
            "target_found_after_inspection_detour",
            tuple(record.reason for record in self.manager.transition_log),
        )

    def test_rejected_or_timed_out_inspection_returns_to_search(self) -> None:
        self.inspect._outcomes.clear()
        self.inspect._outcomes.append(
            _Outcome(
                SkillStatus.FAILED,
                SkillResultCode.TIMEOUT,
                {"candidate_id": "candidate_1", "inspection_confirmed_target": False},
            )
        )
        self.manager.handoff_interrupted_search_candidate_to_inspect(
            _plan(version=2),
            candidate_id="candidate_1",
            source="qwen_vl",
        )
        self.manager.tick(self._observation())
        self.manager.tick(self._observation())

        self.assertIs(self.manager.task_status, TaskStatus.RUNNING)
        self.assertIs(self.manager.active_name, SkillName.SEARCH)
        self.assertNotIn("search", self.manager.step_outputs)
        self.assertNotIn("inspect_candidate", self.manager.step_outputs)
        self.assertIn(
            "inspection_rejected_search_resumed",
            tuple(record.reason for record in self.manager.transition_log),
        )

    def test_inspection_internal_error_fails_safe_without_search_output(self) -> None:
        self.inspect._outcomes.clear()
        self.inspect._outcomes.append(
            _Outcome(
                SkillStatus.FAILED,
                SkillResultCode.INTERNAL_ERROR,
                {"candidate_id": "candidate_1"},
            )
        )
        self.manager.handoff_interrupted_search_candidate_to_inspect(
            _plan(version=2),
            candidate_id="candidate_1",
            source="qwen_vl",
        )
        self.manager.tick(self._observation())
        self.manager.tick(self._observation())

        self.assertIs(self.manager.active_name, SkillName.LAND)
        self.assertIs(self.manager.pending_task_result, TaskStatus.FAILED)
        self.assertNotIn("search", self.manager.step_outputs)
        self.assertIsNone(self.manager.active_target_id)

    def test_cancel_during_inspection_lands_and_reset_clears_detour(self) -> None:
        self.manager.handoff_interrupted_search_candidate_to_inspect(
            _plan(version=2),
            candidate_id="candidate_1",
            source="qwen_vl",
        )
        self.manager.tick(self._observation())
        self.assertIs(self.manager.active_name, SkillName.INSPECT)

        self.manager.cancel_task()
        self.assertIs(self.manager.active_name, SkillName.LAND)
        self.manager.tick(self._observation())
        self.assertIs(self.manager.task_status, TaskStatus.CANCELED)
        self.manager.reset_task()
        self.assertIs(self.manager.task_status, TaskStatus.IDLE)

        self.manager.start_task(self.original)
        self.assertIs(self.manager.active_name, SkillName.GOTO)

    def test_invalid_handoffs_are_atomic_and_leave_hover_owned(self) -> None:
        invalid = (
            ("wrong candidate", _plan(version=2, candidate_id="candidate_2"), "candidate_1"),
            ("changed search", _plan(version=2, change_search=True), "candidate_1"),
            ("missing inspect", _plan(version=2, include_inspect=False), "candidate_1"),
            ("version jump", _plan(version=3), "candidate_1"),
        )
        for label, plan, candidate_id in invalid:
            with self.subTest(label=label):
                before_plan = self.manager.task_plan.to_dict()  # type: ignore[union-attr]
                before_outputs = self.manager.step_outputs
                before_transitions = self.manager.transition_log
                with self.assertRaises(SkillManagerError):
                    self.manager.handoff_interrupted_search_candidate_to_inspect(
                        plan,
                        candidate_id=candidate_id,
                        source="qwen_vl",
                    )
                self.assertEqual(
                    self.manager.task_plan.to_dict(),  # type: ignore[union-attr]
                    before_plan,
                )
                self.assertEqual(self.manager.step_outputs, before_outputs)
                self.assertEqual(self.manager.transition_log, before_transitions)
                self.assertIs(self.manager.active_name, SkillName.HOVER)
                self.assertTrue(self.manager.is_supervisory_paused)

    def test_untrusted_candidate_source_is_rejected_atomically(self) -> None:
        before = self.manager.task_plan.to_dict()  # type: ignore[union-attr]
        with self.assertRaises(SkillManagerError):
            self.manager.handoff_interrupted_search_candidate_to_inspect(
                _plan(version=2),
                candidate_id="candidate_1",
                source="detector",
            )
        self.assertEqual(self.manager.task_plan.to_dict(), before)  # type: ignore[union-attr]
        self.assertIs(self.manager.active_name, SkillName.HOVER)


if __name__ == "__main__":
    unittest.main()
