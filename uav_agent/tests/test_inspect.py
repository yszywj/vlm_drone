from __future__ import annotations

from dataclasses import fields
from math import atan2, pi
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from perception.candidate_bank import CandidateBank, CandidateLifecycle
from perception.grounding import (
    OracleEvaluationCandidateResolver,
    ProductionCandidateResolver,
    ResolvedCandidatePosition,
)
from perception.runtime import PerceptionRuntimeProfile
from runtime.frame_store import FrameStore
from skills.inspect import (
    InspectApproachPolicy,
    InspectGoal,
    InspectPhase,
    InspectPolicy,
    InspectSkill,
)
from skills.types import Observation, SkillContext, SkillResultCode, SkillStatus


class ManualClock:
    def __init__(self, timestamp_s: float = 0.0) -> None:
        self.timestamp_s = timestamp_s

    def now(self) -> float:
        return self.timestamp_s

    def advance(self, dt_s: float) -> None:
        self.timestamp_s += dt_s


class FakeCamera:
    def __init__(self, uav: KinematicUAV) -> None:
        self._uav = uav

    def get_rgb(self) -> np.ndarray:
        return np.zeros((12, 16, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = self._uav.get_pose()
        return (
            np.asarray([state.x, state.y, state.z], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )


class MismatchedResolver:
    profile = PerceptionRuntimeProfile.ORACLE_EVALUATION

    def resolve(self, candidate, *, timestamp_s: float):
        del timestamp_s
        return ResolvedCandidatePosition(
            uav_id="uav_2",
            candidate_id=candidate.candidate_id,
            position_xyz_m=(6.0, 0.0, 0.0),
            source="oracle_evaluation",
        )


class IllicitProductionOracleResolver:
    profile = PerceptionRuntimeProfile.PRODUCTION

    def resolve(self, candidate, *, timestamp_s: float):
        del timestamp_s
        return ResolvedCandidatePosition(
            uav_id=candidate.uav_id,
            candidate_id=candidate.candidate_id,
            position_xyz_m=(6.0, 0.0, 0.0),
            source="oracle_evaluation",
        )


def _oracle_resolver(
    position_xyz_m: tuple[float, float, float] = (6.0, 0.0, 0.0),
) -> OracleEvaluationCandidateResolver:
    return OracleEvaluationCandidateResolver(
        lambda uav_id, candidate_id, timestamp_s: position_xyz_m,
        profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
        acknowledge_privileged_oracle=True,
    )


def _fixture(
    *,
    policy: InspectPolicy | None = None,
    resolver=None,
    candidate_id: str = "candidate_1",
) -> tuple[
    InspectSkill,
    KinematicUAV,
    ManualClock,
    CandidateBank,
    FrameStore,
    SkillContext,
]:
    uav = KinematicUAV(
        UAVState(0.0, 0.0, 5.0, 0.0),
        max_speed_mps=5.0,
        max_yaw_rate_rad_s=5.0,
    )
    clock = ManualClock()
    frame_store = FrameStore(max_frames=16, max_bytes=1_000_000, max_age_s=30)
    initial_ref = frame_store.add_frame(
        uav_id="uav_1",
        frame_id="frame_initial",
        timestamp_s=0.0,
        rgb=np.zeros((12, 16, 3), dtype=np.uint8),
    )
    bank = CandidateBank(uav_id="uav_1", stale_after_s=30.0)
    proposed = bank.propose(
        candidate_id=candidate_id,
        timestamp_s=0.0,
        bbox_xyxy_normalized=(0.2, 0.2, 0.6, 0.8),
        frame_ref=initial_ref,
        source="oracle_evaluation",
    )
    if proposed is None:
        raise AssertionError("fresh candidate was unexpectedly suppressed")
    selected_policy = policy or InspectPolicy(
        approach_speed_mps=2.0,
        position_tolerance_m=0.15,
        max_yaw_rate_rad_s=4.0,
        stabilization_duration_s=0.4,
        max_evidence_frames=2,
    )
    skill = InspectSkill(
        candidate_bank=bank,
        candidate_resolver=resolver or _oracle_resolver(),
        frame_store=frame_store,
        policy=selected_policy,
    )
    context = SkillContext(
        uav=uav,
        camera=FakeCamera(uav),
        perception=None,
        clock=clock,
        uav_id="uav_1",
    )
    return skill, uav, clock, bank, frame_store, context


def _observation(
    uav: KinematicUAV,
    clock: ManualClock,
    *,
    pixel_value: int = 0,
) -> Observation:
    state = uav.get_pose()
    return Observation(
        timestamp=clock.now(),
        uav_pose=state,
        uav_velocity=uav.get_velocity(),
        camera_rgb=np.full((12, 16, 3), pixel_value, dtype=np.uint8),
        uav_id="uav_1",
    )


def _run(
    skill: InspectSkill,
    uav: KinematicUAV,
    clock: ManualClock,
    *,
    dt_s: float = 0.1,
    max_steps: int = 300,
) -> tuple[SkillStatus, list[InspectPhase | None], float]:
    phases: list[InspectPhase | None] = []
    largest_step = 0.0
    for step_index in range(max_steps):
        before = uav.get_pose()
        status = skill.tick(
            _observation(uav, clock, pixel_value=step_index % 255)
        )
        after_tick = uav.get_pose()
        # INSPECT may only command motion; a tick itself must never teleport.
        np.testing.assert_allclose(
            [after_tick.x, after_tick.y, after_tick.z],
            [before.x, before.y, before.z],
            atol=0.0,
        )
        phases.append(skill.phase)
        if status is not SkillStatus.RUNNING:
            return status, phases, largest_step
        uav.step(dt_s)
        after_step = uav.get_pose()
        largest_step = max(
            largest_step,
            float(
                np.linalg.norm(
                    np.asarray(
                        [
                            after_step.x - before.x,
                            after_step.y - before.y,
                            after_step.z - before.z,
                        ],
                        dtype=np.float64,
                    )
                )
            ),
        )
        clock.advance(dt_s)
    raise AssertionError("INSPECT did not reach a terminal state")


def _contains_forbidden_runtime_value(value: object) -> bool:
    if isinstance(value, (np.ndarray, bytes, bytearray, memoryview)):
        return True
    if isinstance(value, dict):
        return any(
            _contains_forbidden_runtime_value(key)
            or _contains_forbidden_runtime_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_forbidden_runtime_value(item) for item in value)
    return False


class InspectSkillTest(unittest.TestCase):
    def test_goal_exposes_no_world_coordinate_or_low_level_control(self) -> None:
        self.assertEqual(
            {field.name for field in fields(InspectGoal)},
            {
                "candidate_id",
                "desired_observation_distance_m",
                "viewpoint_change_rad",
                "max_duration_s",
                "approach_policy",
            },
        )
        goal = InspectGoal(candidate_id="candidate_1")
        self.assertIs(
            goal.approach_policy,
            InspectApproachPolicy.MAINTAIN_ALTITUDE_ORBIT,
        )
        for forbidden in (
            "position",
            "coordinate",
            "velocity",
            "waypoint",
            "yaw",
            "motor",
        ):
            self.assertFalse(
                any(forbidden in field.name.casefold() for field in fields(goal))
            )

    def test_ideal_inspect_moves_continuously_and_emits_bounded_frame_refs(self) -> None:
        skill, uav, clock, bank, frame_store, context = _fixture()
        skill.start(
            InspectGoal(
                candidate_id="candidate_1",
                desired_observation_distance_m=2.0,
                viewpoint_change_rad=pi / 2.0,
                max_duration_s=10.0,
            ),
            context,
        )
        self.assertIs(skill.status, SkillStatus.RUNNING)

        status, phases, largest_step = _run(skill, uav, clock)
        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertIn(InspectPhase.APPROACH, phases)
        self.assertIn(InspectPhase.VIEWPOINT_CHANGE, phases)
        self.assertIn(InspectPhase.STABILIZING, phases)
        self.assertLessEqual(largest_step, 0.2 + 1e-9)

        final = uav.get_pose()
        self.assertAlmostEqual(final.z, 5.0, places=9)
        self.assertAlmostEqual(
            float(np.hypot(final.x - 6.0, final.y)),
            2.0,
            delta=0.2,
        )
        # The positive pi/2 request changed the approach bearing by a finite,
        # policy-bounded amount rather than orbiting indefinitely.
        final_bearing = atan2(final.y, final.x - 6.0)
        self.assertAlmostEqual(final_bearing, -pi / 2.0, delta=0.15)

        result = skill.get_result()
        self.assertIsNotNone(result)
        self.assertIs(result.code, SkillResultCode.GOAL_REACHED)
        self.assertIs(result.data["inspection_confirmed_target"], False)
        evidence = result.data["evidence_handle"]
        self.assertIsInstance(evidence, dict)
        frame_values = evidence["frame_refs"]
        self.assertGreaterEqual(len(frame_values), 1)
        self.assertLessEqual(len(frame_values), skill.policy.max_evidence_frames)
        self.assertFalse(_contains_forbidden_runtime_value(result.data))
        for frame_value in frame_values:
            self.assertEqual(
                set(frame_value),
                {"uav_id", "frame_id", "timestamp_s", "width", "height"},
            )
        handle = skill.evidence_handle
        self.assertIsNotNone(handle)
        self.assertLessEqual(
            len(handle.frame_refs),
            skill.policy.max_evidence_frames,
        )
        self.assertTrue(all(frame_store.contains(ref) for ref in handle.frame_refs))

        candidate = bank.get("candidate_1")
        self.assertIsNotNone(candidate)
        # INSPECT produces evidence for review.  It never verifies its own
        # semantic target hypothesis, and it releases motion ownership while
        # the independent semantic review is still pending.
        self.assertIs(candidate.lifecycle, CandidateLifecycle.PROVISIONAL)

    def test_invalid_goals_fail_before_resolving_or_moving(self) -> None:
        invalid_goals = (
            InspectGoal(candidate_id=""),
            InspectGoal(
                candidate_id="candidate_1",
                desired_observation_distance_m=1.0,
            ),
            InspectGoal(
                candidate_id="candidate_1",
                desired_observation_distance_m=21.0,
            ),
            InspectGoal(
                candidate_id="candidate_1",
                viewpoint_change_rad=pi,
            ),
            InspectGoal(
                candidate_id="candidate_1",
                viewpoint_change_rad=0.0,
            ),
            InspectGoal(candidate_id="candidate_1", max_duration_s=0.0),
            InspectGoal(candidate_id="candidate_1", max_duration_s=61.0),
            InspectGoal(
                candidate_id="candidate_1",
                approach_policy="free_form",  # type: ignore[arg-type]
            ),
        )
        for goal in invalid_goals:
            with self.subTest(goal=goal):
                skill, uav, _, bank, _, context = _fixture()
                initial = uav.get_pose()
                skill.start(goal, context)
                self.assertIs(skill.status, SkillStatus.FAILED)
                self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)
                self.assertEqual(uav.get_pose(), initial)
                self.assertIs(
                    bank.get("candidate_1").lifecycle,
                    CandidateLifecycle.PROVISIONAL,
                )

    def test_unknown_candidate_fails_closed(self) -> None:
        skill, _, _, bank, _, context = _fixture()
        skill.start(InspectGoal(candidate_id="candidate_missing"), context)
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_STATE)
        self.assertIs(
            bank.get("candidate_1").lifecycle,
            CandidateLifecycle.PROVISIONAL,
        )

    def test_production_resolver_is_unavailable_and_does_not_mutate_candidate(self) -> None:
        skill, _, _, bank, _, context = _fixture(
            resolver=ProductionCandidateResolver()
        )
        skill.start(InspectGoal(candidate_id="candidate_1"), context)
        self.assertIs(skill.status, SkillStatus.FAILED)
        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.INVALID_STATE)
        self.assertIn("not implemented", result.message)
        self.assertIs(
            bank.get("candidate_1").lifecycle,
            CandidateLifecycle.PROVISIONAL,
        )

    def test_resolver_routing_and_profile_mismatches_fail_closed(self) -> None:
        for resolver in (MismatchedResolver(), IllicitProductionOracleResolver()):
            with self.subTest(resolver=type(resolver).__name__):
                skill, _, _, bank, _, context = _fixture(resolver=resolver)
                skill.start(InspectGoal(candidate_id="candidate_1"), context)
                self.assertIs(skill.status, SkillStatus.FAILED)
                self.assertIs(
                    skill.get_result().code,
                    SkillResultCode.INVALID_STATE,
                )
                self.assertIs(
                    bank.get("candidate_1").lifecycle,
                    CandidateLifecycle.PROVISIONAL,
                )

    def test_timeout_does_not_fabricate_confirmation(self) -> None:
        policy = InspectPolicy(
            approach_speed_mps=0.25,
            stabilization_duration_s=0.2,
            max_evidence_frames=2,
        )
        skill, uav, clock, bank, _, context = _fixture(policy=policy)
        skill.start(
            InspectGoal(candidate_id="candidate_1", max_duration_s=0.3),
            context,
        )
        status, _, _ = _run(skill, uav, clock, max_steps=20)
        self.assertIs(status, SkillStatus.FAILED)
        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.TIMEOUT)
        self.assertIs(result.data["inspection_confirmed_target"], False)
        self.assertEqual(result.data["evidence_frame_count"], 0)
        self.assertIs(
            bank.get("candidate_1").lifecycle,
            CandidateLifecycle.REJECTED,
        )

    def test_cancel_releases_candidate_from_under_inspection(self) -> None:
        skill, _, _, bank, _, context = _fixture()
        skill.start(InspectGoal(candidate_id="candidate_1"), context)
        self.assertIs(
            bank.get("candidate_1").lifecycle,
            CandidateLifecycle.UNDER_INSPECTION,
        )

        skill.cancel()

        self.assertIs(skill.status, SkillStatus.CANCELED)
        self.assertIs(
            bank.get("candidate_1").lifecycle,
            CandidateLifecycle.REJECTED,
        )

    def test_candidate_rejected_while_running_fails_without_confirmation(self) -> None:
        skill, uav, clock, bank, _, context = _fixture()
        skill.start(InspectGoal(candidate_id="candidate_1"), context)
        bank.reject("candidate_1", timestamp_s=clock.now())
        status = skill.tick(_observation(uav, clock))
        self.assertIs(status, SkillStatus.FAILED)
        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.INVALID_STATE)
        self.assertNotIn("target_id", result.data)
        self.assertNotIn("inspection_confirmed_target", result.data)

    def test_policy_keeps_evidence_cap_small_and_explicit(self) -> None:
        self.assertEqual(InspectPolicy().max_evidence_frames, 3)
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            InspectPolicy(max_evidence_frames=9)
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            InspectPolicy(max_evidence_frames=0)


if __name__ == "__main__":
    unittest.main()
