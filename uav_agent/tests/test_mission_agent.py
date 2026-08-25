"""Pure-Python integration tests for the high-level MissionAgent.

These tests intentionally use no Isaac Sim, model server, image model, or
environment facade.  The Agent receives only its narrow planner/runtime
dependencies, while deterministic Skills exercise the real SkillManager and
TargetManager state machines.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import unittest

import numpy as np

from agents.mission_agent import (
    AgentStatus,
    MissionAgent,
    MissionAgentError,
    MissionAgentSnapshot,
)
from common.target_estimate import TargetEstimate
from env.kinematic_uav import KinematicUAV, UAVState
from planner.base import MissionPlanner
from perception.runtime import PerceptionRuntimeProfile
from perception.confirmation import CandidateConfirmationCoordinator
from perception.types import (
    DetectionCandidate,
    IdentityConsistencyEvidence,
    SemanticVerification,
    ShortTrackEvidence,
)
from planner.schemas import (
    CompiledMission,
    LandingZoneSpec,
    MissionIntent,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
)
from runtime.plan_validator import PlanValidationError, PlanValidator
from runtime.safety_supervisor import (
    SafetyAction,
    SafetyDecision,
    SafetySupervisor,
)
from skills.base import Skill
from skills.manager import SkillManager, TaskPlan, TaskStatus, TransitionRecord
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillName,
    SkillResultCode,
    SkillStatus,
)
from target.target_manager import TargetManager
from target.types import TargetLifecycle


class FakeClock:
    """Manually controlled monotonic clock satisfying ``SkillClock``."""

    def __init__(self, time_s: float = 0.0) -> None:
        self.time_s = float(time_s)

    def now(self) -> float:
        return self.time_s

    def set(self, time_s: float) -> None:
        self.time_s = float(time_s)


class FakeCamera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


class FakePlanner(MissionPlanner):
    """Return one immutable high-level intent and count boundary calls."""

    source = "scripted"

    def __init__(
        self,
        intent: MissionIntent,
        *,
        error: Exception | None = None,
    ) -> None:
        self._intent = intent
        self._error = error
        self.calls = 0
        self.requests: list[PlannerRequest] = []

    def plan(self, request: PlannerRequest) -> MissionIntent:
        self.calls += 1
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return MissionIntent.from_dict(self._intent.to_dict())


class CountingValidator(PlanValidator):
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error
        self.sources: list[str] = []

    def validate_and_compile(
        self,
        intent: MissionIntent,
        context: PlannerWorldContext,
        *,
        source: str,
    ) -> CompiledMission:
        self.calls += 1
        self.sources.append(source)
        if self.error is not None:
            raise self.error
        return super().validate_and_compile(intent, context, source=source)


class LegacyFiveStepValidator(CountingValidator):
    """Compile the pre-task-1 five-step plan for compatibility coverage."""

    def validate_and_compile(
        self,
        intent: MissionIntent,
        context: PlannerWorldContext,
        *,
        source: str,
    ) -> CompiledMission:
        six_step = super().validate_and_compile(intent, context, source=source)
        entries = six_step.task_plan.to_dicts()
        del entries[-2]
        return CompiledMission(
            intent=six_step.intent,
            task_plan=TaskPlan.from_dicts(entries),
            source=six_step.source,
        )


class CountingSafetySupervisor(SafetySupervisor):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.preflight_calls = 0
        self.evaluate_calls = 0
        self.decisions: list[SafetyDecision] = []

    def preflight(self, mission: CompiledMission | TaskPlan) -> SafetyDecision:
        self.preflight_calls += 1
        decision = super().preflight(mission)
        self.decisions.append(decision)
        return decision

    def evaluate(
        self,
        observation: Observation,
        *,
        mission_elapsed_s: float,
    ) -> SafetyDecision:
        self.evaluate_calls += 1
        decision = super().evaluate(
            observation,
            mission_elapsed_s=mission_elapsed_s,
        )
        self.decisions.append(decision)
        return decision


class RejectingSafetySupervisor(CountingSafetySupervisor):
    def preflight(self, mission: CompiledMission | TaskPlan) -> SafetyDecision:
        self.preflight_calls += 1
        decision = SafetyDecision(SafetyAction.ABORT, "scripted preflight rejection")
        self.decisions.append(decision)
        return decision


@dataclass(frozen=True, slots=True)
class ScriptedOutcome:
    status: SkillStatus
    code: SkillResultCode | None = None
    data: dict[str, object] = field(default_factory=dict)


class ScriptedSkill(Skill):
    """Finish or remain running according to one queued outcome per tick."""

    goal_type = SkillGoal

    def __init__(self, *outcomes: ScriptedOutcome) -> None:
        super().__init__()
        self._outcomes = deque(outcomes)
        self.started_goals: list[SkillGoal] = []
        self.tick_count = 0

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        self.started_goals.append(deepcopy(goal))

    @property
    def next_outcome(self) -> ScriptedOutcome | None:
        """Expose the next scripted frame to the test perception fixture."""

        return None if not self._outcomes else self._outcomes[0]

    def _on_tick(self, observation: Observation) -> None:
        self.tick_count += 1
        if not self._outcomes:
            raise AssertionError("ScriptedSkill has no queued outcome")
        outcome = self._outcomes.popleft()
        if outcome.status is SkillStatus.RUNNING:
            self._set_feedback(
                0.5,
                "scripted running",
                {"samples": [self.tick_count]},
            )
            return
        if outcome.status is SkillStatus.SUCCEEDED and outcome.code is not None:
            self._succeed(outcome.code, "scripted success", outcome.data)
            return
        if outcome.status is SkillStatus.FAILED and outcome.code is not None:
            self._fail(outcome.code, "scripted failure", outcome.data)
            return
        raise AssertionError("invalid ScriptedOutcome")


class CountingSkillManager(SkillManager):
    def __init__(self, *args: object, start_error: Exception | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.start_error = start_error
        self.start_task_calls = 0
        self.tick_calls = 0
        self.cancel_task_calls = 0
        self.reset_task_calls = 0

    def start_task(self, plan: TaskPlan) -> TaskStatus:
        self.start_task_calls += 1
        if self.start_error is not None:
            raise self.start_error
        return super().start_task(plan)

    def tick(self, observation: Observation) -> SkillStatus | TaskStatus:
        self.tick_calls += 1
        return super().tick(observation)

    def cancel_task(self) -> TaskStatus:
        self.cancel_task_calls += 1
        return super().cancel_task()

    def reset_task(self) -> None:
        self.reset_task_calls += 1
        super().reset_task()


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


def running() -> ScriptedOutcome:
    return ScriptedOutcome(SkillStatus.RUNNING)


def world_context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={
            "search_area": SearchRegionSpec(
                name="search_area",
                center_xyz_m=(20.0, 20.0, 0.0),
                radius_m=10.0,
                approach_xyz_m=(20.0, 5.0, 10.0),
                description="known semantic search area",
            )
        },
        landing_zones={
            "home": LandingZoneSpec(
                name="home",
                position_xy_m=(0.0, 0.0),
                ground_altitude_m=0.0,
                description="home landing pad",
            )
        },
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=30.0,
        search_timeout_s=60.0,
        goto_timeout_s=120.0,
        land_timeout_s=60.0,
    )


def mission_intent(**overrides: object) -> MissionIntent:
    values: dict[str, object] = {
        "target_description": "moving red target",
        "search_region": "search_area",
        "track_duration_s": 30.0,
        "landing_zone": "home",
        "takeoff_altitude_m": 10.0,
    }
    values.update(overrides)
    return MissionIntent.from_dict(values)


def default_outcomes() -> dict[SkillName, list[ScriptedOutcome]]:
    return {
        SkillName.TAKEOFF: [succeeded(SkillResultCode.TAKEOFF_COMPLETE)],
        # The six-step plan runs GOTO twice.
        SkillName.GOTO: [
            succeeded(SkillResultCode.GOAL_REACHED),
            succeeded(SkillResultCode.GOAL_REACHED),
        ],
        SkillName.SEARCH: [
            succeeded(SkillResultCode.TARGET_FOUND, {"target_id": "target_0"})
        ],
        SkillName.TRACK: [succeeded(SkillResultCode.TRACK_COMPLETE)],
        SkillName.REACQUIRE: [
            succeeded(SkillResultCode.TARGET_FOUND, {"target_id": "target_0"})
        ],
        SkillName.LAND: [succeeded(SkillResultCode.LAND_COMPLETE)],
    }


def observation(
    timestamp: float,
    *,
    pose: UAVState | None = None,
) -> Observation:
    return Observation(
        uav_id="uav_1",
        timestamp=float(timestamp),
        uav_pose=pose or UAVState(0.0, 0.0, 10.0, 0.0),
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def privileged_target_estimate(
    timestamp: float,
    source: str,
) -> TargetEstimate:
    return TargetEstimate(
        timestamp_s=timestamp,
        target_id="target_0",
        candidate_id="candidate_0",
        tracker_id="track_0",
        visible=False,
        confirmed=True,
        predicted_only=True,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox_xyxy_normalized=None,
        position_world_m=(1.0, 0.0, 0.0),
        velocity_world_mps=(0.0, 0.0, 0.0),
        measurement_age_s=0.0,
        source=source,
    )


@dataclass(slots=True)
class Harness:
    agent: MissionAgent
    planner: FakePlanner
    validator: CountingValidator
    safety: CountingSafetySupervisor
    manager: CountingSkillManager
    target: TargetManager
    clock: FakeClock
    skills: dict[SkillName, ScriptedSkill]
    context: PlannerWorldContext
    auto_commit_oracle_provider: bool

    def start(self) -> CompiledMission:
        return self.agent.start("find, track, and return", self.context)

    def tick(
        self,
        timestamp: float,
        *,
        pose: UAVState | None = None,
    ) -> MissionAgentSnapshot:
        self.clock.set(timestamp)
        self._commit_oracle_provider_lock(timestamp)
        return self.agent.tick(observation(timestamp, pose=pose))

    def _commit_oracle_provider_lock(self, timestamp: float) -> None:
        """Model the provider-before-Agent ordering used by the real runtime."""

        if not self.auto_commit_oracle_provider:
            return
        active = self.manager.active_name
        if active not in {SkillName.SEARCH, SkillName.REACQUIRE}:
            return
        outcome = self.skills[active].next_outcome
        if (
            outcome is None
            or outcome.status is not SkillStatus.SUCCEEDED
            or outcome.code is not SkillResultCode.TARGET_FOUND
        ):
            return
        target_id = outcome.data.get("target_id")
        if not isinstance(target_id, str) or not target_id.strip():
            return
        provider_values = {
            "timestamp_s": timestamp,
            "confidence": 1.0,
            "last_seen_position": (1.0, 2.0, 0.5),
            "last_seen_velocity": (0.0, 0.0, 0.0),
        }
        if self.target.lifecycle is TargetLifecycle.SEARCHING:
            self.target.lock_oracle_from_search(
                target_id.strip(),
                **provider_values,
            )
        elif self.target.lifecycle is TargetLifecycle.REACQUIRING:
            self.target.mark_reacquired_oracle(
                target_id.strip(),
                **provider_values,
            )


def make_harness(
    *,
    outcomes: dict[SkillName, list[ScriptedOutcome]] | None = None,
    planner: FakePlanner | None = None,
    validator: CountingValidator | None = None,
    safety: CountingSafetySupervisor | None = None,
    start_error: Exception | None = None,
    logger: object | None = None,
    # The fixture models the Oracle provider committing TargetManager before
    # MissionAgent consumes a TARGET_FOUND Skill transition. Individual
    # production-boundary tests disable that fixture through their profile.
    perception_runtime_profile: PerceptionRuntimeProfile = (
        PerceptionRuntimeProfile.ORACLE_EVALUATION
    ),
    acknowledge_privileged_oracle: bool = True,
    runtime_program: str = "linear",
) -> Harness:
    clock = FakeClock()
    context = SkillContext(
        uav=KinematicUAV(
            UAVState(0.0, 0.0, 0.0, 0.0),
            max_speed_mps=5.0,
            max_yaw_rate_rad_s=2.0,
        ),
        camera=FakeCamera(),
        perception=None,
        clock=clock,
        uav_id="uav_1",
    )
    configured = default_outcomes()
    if outcomes is not None:
        configured.update(outcomes)
    skills = {
        name: ScriptedSkill(*skill_outcomes)
        for name, skill_outcomes in configured.items()
    }
    manager = CountingSkillManager(
        context,
        registry=skills,
        start_error=start_error,
    )
    planner = planner or FakePlanner(mission_intent())
    validator = validator or CountingValidator()
    safety = safety or CountingSafetySupervisor(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        max_mission_time_s=300.0,
        max_safe_altitude_m=25.0,
    )
    target = TargetManager()
    agent = MissionAgent(
        planner=planner,
        validator=validator,
        safety=safety,
        skill_manager=manager,
        target_manager=target,
        clock=clock,
        logger=logger,  # type: ignore[arg-type]
        perception_runtime_profile=perception_runtime_profile,
        acknowledge_privileged_oracle=acknowledge_privileged_oracle,
        runtime_program=runtime_program,
    )
    return Harness(
        agent=agent,
        planner=planner,
        validator=validator,
        safety=safety,
        manager=manager,
        target=target,
        clock=clock,
        skills=skills,
        context=world_context(),
        auto_commit_oracle_provider=(
            perception_runtime_profile
            is PerceptionRuntimeProfile.ORACLE_EVALUATION
        ),
    )


def run_to_terminal(harness: Harness, *, start_timestamp: int = 1) -> MissionAgentSnapshot:
    snapshot = harness.agent.snapshot()
    for timestamp in range(start_timestamp, start_timestamp + 20):
        snapshot = harness.tick(float(timestamp))
        if snapshot.status in {
            AgentStatus.SUCCEEDED,
            AgentStatus.FAILED,
            AgentStatus.CANCELED,
        }:
            return snapshot
    raise AssertionError("mission did not become terminal within 20 scripted ticks")


class MissionAgentStartTests(unittest.TestCase):
    def test_start_success_calls_planner_once_and_keeps_target_uninitialized(self) -> None:
        harness = make_harness()

        compiled = harness.start()
        snapshot = harness.agent.snapshot()

        self.assertIsInstance(compiled, CompiledMission)
        self.assertEqual(len(compiled.task_plan.steps), 6)
        self.assertEqual(harness.planner.calls, 1)
        self.assertEqual(harness.validator.calls, 1)
        self.assertEqual(harness.validator.sources, ["scripted"])
        self.assertEqual(harness.safety.preflight_calls, 1)
        self.assertEqual(harness.manager.start_task_calls, 1)
        self.assertIs(snapshot.status, AgentStatus.RUNNING)
        self.assertEqual(snapshot.task_status, "RUNNING")
        self.assertEqual(snapshot.active_skill, "TAKEOFF")
        self.assertIs(snapshot.target.lifecycle, TargetLifecycle.UNINITIALIZED)
        request = harness.planner.requests[0]
        self.assertIs(request.world_context, harness.context)

    def test_graph_start_binds_real_manager_program_executor(self) -> None:
        harness = make_harness(runtime_program="graph")

        harness.start()

        self.assertTrue(harness.manager.is_graph_runtime)
        program = harness.manager.program_snapshot
        assert program is not None
        assert harness.manager.task_plan is not None
        self.assertEqual(
            program.current_node_id,
            harness.manager.task_plan.steps[0].step_id,
        )
        terminal = run_to_terminal(harness)
        self.assertIs(terminal.status, AgentStatus.SUCCEEDED)
        assert harness.manager.program_snapshot is not None
        self.assertTrue(harness.manager.program_snapshot.terminal)

    def test_planner_failure_sets_failed_and_does_not_start_manager(self) -> None:
        planner = FakePlanner(mission_intent(), error=RuntimeError("planner offline"))
        harness = make_harness(planner=planner)

        with self.assertRaises(MissionAgentError):
            harness.start()

        self.assertIs(harness.agent.snapshot().status, AgentStatus.FAILED)
        self.assertIn("planner offline", harness.agent.snapshot().last_error or "")
        self.assertEqual(planner.calls, 1)
        self.assertEqual(harness.validator.calls, 0)
        self.assertEqual(harness.manager.start_task_calls, 0)

    def test_reset_after_planner_failure_restores_idle_without_invalid_resets(self) -> None:
        planner = FakePlanner(mission_intent(), error=RuntimeError("planner offline"))
        harness = make_harness(planner=planner)
        with self.assertRaises(MissionAgentError):
            harness.start()

        harness.agent.reset()

        snapshot = harness.agent.snapshot()
        self.assertIs(snapshot.status, AgentStatus.IDLE)
        self.assertEqual(snapshot.task_status, "IDLE")
        self.assertIsNone(snapshot.last_error)
        self.assertIs(snapshot.target.lifecycle, TargetLifecycle.UNINITIALIZED)
        self.assertEqual(harness.manager.reset_task_calls, 0)
        self.assertEqual(planner.calls, 1)

    def test_validator_failure_does_not_start_manager(self) -> None:
        validator = CountingValidator(error=PlanValidationError("invalid intent"))
        harness = make_harness(validator=validator)

        with self.assertRaises(MissionAgentError):
            harness.start()

        self.assertIs(harness.agent.snapshot().status, AgentStatus.FAILED)
        self.assertEqual(harness.planner.calls, 1)
        self.assertEqual(validator.calls, 1)
        self.assertEqual(harness.safety.preflight_calls, 0)
        self.assertEqual(harness.manager.start_task_calls, 0)

    def test_preflight_failure_does_not_start_manager(self) -> None:
        safety = RejectingSafetySupervisor(
            scene_min_xyz_m=(-50.0, -50.0, 0.0),
            scene_max_xyz_m=(50.0, 50.0, 30.0),
        )
        harness = make_harness(safety=safety)

        with self.assertRaises(MissionAgentError):
            harness.start()

        self.assertIs(harness.agent.snapshot().status, AgentStatus.FAILED)
        self.assertEqual(safety.preflight_calls, 1)
        self.assertEqual(harness.manager.start_task_calls, 0)

    def test_skill_manager_start_failure_sets_failed(self) -> None:
        harness = make_harness(start_error=RuntimeError("dispatch failed"))

        with self.assertRaises(MissionAgentError):
            harness.start()

        self.assertIs(harness.agent.snapshot().status, AgentStatus.FAILED)
        self.assertIn("dispatch failed", harness.agent.snapshot().last_error or "")
        self.assertEqual(harness.manager.start_task_calls, 1)

    def test_start_rejects_non_idle_without_replanning(self) -> None:
        harness = make_harness()
        harness.start()

        with self.assertRaises(MissionAgentError):
            harness.start()

        self.assertEqual(harness.planner.calls, 1)


class MissionAgentTickAndTargetTests(unittest.TestCase):
    def test_inspection_detour_return_preserves_existing_search_lifecycle(self) -> None:
        harness = make_harness()
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)
        self.assertIs(harness.target.lifecycle, TargetLifecycle.SEARCHING)
        events_before = harness.target.events()

        # Manager's trusted detour emits INSPECT -> SEARCH without completing
        # SEARCH. The Agent must retain the existing TargetSpec/search state,
        # not create a second target lifecycle or claim a lock.
        harness.agent._apply_target_transition(  # type: ignore[attr-defined]
            TransitionRecord(
                timestamp=2.5,
                old_skill=SkillName.INSPECT,
                old_status=SkillStatus.SUCCEEDED,
                result_code=SkillResultCode.GOAL_REACHED,
                new_skill=SkillName.SEARCH,
                reason="inspection_evidence_collected_search_resumed",
                old_step_id="inspect_candidate",
                new_step_id="search",
            )
        )

        self.assertIs(harness.target.lifecycle, TargetLifecycle.SEARCHING)
        self.assertEqual(harness.target.events(), events_before)
        self.assertIsNone(harness.target.snapshot().target_id)

    def test_duplicate_timestamp_does_not_tick_or_consume_transition_twice(self) -> None:
        harness = make_harness(
            outcomes={
                SkillName.TAKEOFF: [
                    running(),
                    succeeded(SkillResultCode.TAKEOFF_COMPLETE),
                ]
            }
        )
        harness.start()

        first = harness.tick(1.0)
        events_before = harness.target.events()
        transitions_before = harness.manager.transition_log
        duplicate = harness.agent.tick(observation(1.0))

        self.assertEqual(harness.manager.tick_calls, 1)
        self.assertEqual(harness.target.events(), events_before)
        self.assertEqual(harness.manager.transition_log, transitions_before)
        self.assertEqual(duplicate, first)

    def test_tick_never_calls_planner_again(self) -> None:
        harness = make_harness()
        harness.start()
        run_to_terminal(harness)
        self.assertEqual(harness.planner.calls, 1)

    def test_target_starts_search_only_when_search_skill_is_entered(self) -> None:
        harness = make_harness()
        harness.start()
        self.assertIs(harness.target.lifecycle, TargetLifecycle.UNINITIALIZED)

        harness.tick(1.0)  # TAKEOFF -> GOTO
        self.assertIs(harness.target.lifecycle, TargetLifecycle.UNINITIALIZED)
        harness.tick(2.0)  # GOTO -> SEARCH

        snapshot = harness.agent.snapshot().target
        self.assertIs(snapshot.lifecycle, TargetLifecycle.SEARCHING)
        self.assertEqual(snapshot.description, "moving red target")
        self.assertEqual(
            [event.new_state for event in harness.target.events()],
            [TargetLifecycle.SEARCHING],
        )

    def test_search_success_accepts_provider_lock_then_starts_tracking(self) -> None:
        harness = make_harness()
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)
        harness.tick(3.0)  # SEARCH -> TRACK

        target = harness.agent.snapshot().target
        self.assertIs(target.lifecycle, TargetLifecycle.TRACKING)
        self.assertEqual(target.target_id, "target_0")
        self.assertEqual(target.confidence, 1.0)
        self.assertEqual(target.source, "oracle")
        self.assertEqual(
            [event.new_state for event in harness.target.events()],
            [
                TargetLifecycle.SEARCHING,
                TargetLifecycle.LOCKED,
                TargetLifecycle.TRACKING,
            ],
        )

    def test_oracle_search_result_without_provider_lock_fails_closed(self) -> None:
        harness = make_harness()
        harness.auto_commit_oracle_provider = False
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)

        snapshot = harness.tick(3.0)

        self.assertEqual(snapshot.active_skill, "LAND")
        self.assertIn("perception provider", snapshot.last_error or "")
        self.assertNotEqual(snapshot.target.source, "oracle")

    def test_production_search_cannot_directly_oracle_lock(self) -> None:
        harness = make_harness(
            perception_runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
            acknowledge_privileged_oracle=False,
        )
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)

        snapshot = harness.tick(3.0)  # SEARCH result tries to enter TRACK

        self.assertEqual(snapshot.active_skill, "LAND")
        self.assertIn("perception provider", snapshot.last_error or "")
        self.assertNotEqual(snapshot.target.source, "oracle")

    def test_production_search_accepts_confirmed_candidate_lock(self) -> None:
        harness = make_harness(
            perception_runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
            acknowledge_privileged_oracle=False,
        )
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)  # target lifecycle enters SEARCHING
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(
            DetectionCandidate("tracklet_0", 2.1, 0.8),
            harness.target,
        )
        result = coordinator.evaluate(
            target_manager=harness.target,
            track=ShortTrackEvidence("tracklet_0", 2.7, 4, 0.6, True, 0.9),
            semantic=SemanticVerification(
                "tracklet_0",
                2.8,
                "moving red target",
                True,
                0.9,
                "qwen-vl",
            ),
            identity=IdentityConsistencyEvidence(
                "tracklet_0",
                "target_0",
                2.9,
                True,
                True,
                4,
                0.9,
            ),
        )
        self.assertEqual(result.decision.value, "CONFIRMED")

        snapshot = harness.tick(3.0)

        self.assertEqual(snapshot.active_skill, "TRACK")
        self.assertIs(snapshot.target.lifecycle, TargetLifecycle.TRACKING)
        self.assertEqual(snapshot.target.source, "confirmed_vision")

    def test_production_visual_lock_rejects_all_oracle_source_aliases(self) -> None:
        for source in ("oracle", "oracle_truth", "OrAcLe_BrIdGe"):
            with self.subTest(source=source):
                harness = make_harness(
                    perception_runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
                    acknowledge_privileged_oracle=False,
                )
                harness.start()
                harness.tick(1.0)
                harness.tick(2.0)
                coordinator = CandidateConfirmationCoordinator()
                coordinator.register_candidate(
                    DetectionCandidate("tracklet_0", 2.1, 0.8),
                    harness.target,
                )
                coordinator.evaluate(
                    target_manager=harness.target,
                    track=ShortTrackEvidence(
                        "tracklet_0", 2.7, 4, 0.6, True, 0.9
                    ),
                    semantic=SemanticVerification(
                        "tracklet_0",
                        2.8,
                        "moving red target",
                        True,
                        0.9,
                        "qwen-vl",
                    ),
                    identity=IdentityConsistencyEvidence(
                        "tracklet_0",
                        "target_0",
                        2.9,
                        True,
                        True,
                        4,
                        0.9,
                    ),
                )
                # Adversarially relabel an otherwise valid visual lock.  This
                # models an upstream plugin trying to bypass the source gate.
                harness.target._source = source

                snapshot = harness.tick(3.0)

                self.assertEqual(snapshot.active_skill, "LAND")
                self.assertIn("Oracle target lock", snapshot.last_error or "")

    def test_track_lost_enters_reacquiring_with_real_last_seen_data(self) -> None:
        lost_data = {
            "target_id": "target_0",
            "last_seen_position": (7.0, 8.0, 0.0),
            "last_seen_velocity": (0.5, 0.0, 0.0),
            "last_seen_time": 3.5,
            "tracking_duration": 0.5,
        }
        harness = make_harness(
            outcomes={
                SkillName.TRACK: [
                    failed(SkillResultCode.TARGET_LOST, lost_data),
                    succeeded(SkillResultCode.TRACK_COMPLETE),
                ]
            }
        )
        harness.start()
        for timestamp in (1.0, 2.0, 3.0, 4.0):
            harness.tick(timestamp)

        target = harness.agent.snapshot().target
        self.assertEqual(harness.manager.active_name, SkillName.REACQUIRE)
        self.assertIs(target.lifecycle, TargetLifecycle.REACQUIRING)
        self.assertEqual(target.last_seen_position, (7.0, 8.0, 0.0))
        self.assertEqual(target.last_seen_velocity, (0.5, 0.0, 0.0))
        self.assertEqual(target.last_seen_time_s, 3.5)

    def test_missing_last_seen_data_is_not_fabricated(self) -> None:
        harness = make_harness(
            outcomes={
                SkillName.TRACK: [failed(SkillResultCode.TARGET_LOST, {})]
            }
        )
        harness.start()
        for timestamp in (1.0, 2.0, 3.0):
            harness.tick(timestamp)
        before_loss = harness.agent.snapshot().target
        harness.tick(4.0)

        target = harness.agent.snapshot().target
        # A missing TRACK-loss payload must not overwrite the provider's last
        # confirmed estimate while fail-safe LAND begins.
        self.assertIs(target.lifecycle, TargetLifecycle.TERMINATED)
        self.assertEqual(target.last_seen_position, before_loss.last_seen_position)
        self.assertEqual(target.last_seen_velocity, before_loss.last_seen_velocity)
        self.assertEqual(target.last_seen_time_s, before_loss.last_seen_time_s)

    def test_reacquire_success_returns_target_to_tracking(self) -> None:
        lost_data = {
            "last_seen_position": (7.0, 8.0, 0.0),
            "last_seen_velocity": (0.5, 0.0, 0.0),
            "last_seen_time": 3.5,
        }
        harness = make_harness(
            outcomes={
                SkillName.TRACK: [
                    failed(SkillResultCode.TARGET_LOST, lost_data),
                    succeeded(SkillResultCode.TRACK_COMPLETE),
                ]
            }
        )
        harness.start()
        for timestamp in (1.0, 2.0, 3.0, 4.0, 5.0):
            harness.tick(timestamp)

        self.assertEqual(harness.manager.active_name, SkillName.TRACK)
        self.assertIs(
            harness.agent.snapshot().target.lifecycle,
            TargetLifecycle.TRACKING,
        )
        self.assertEqual(
            [event.new_state for event in harness.target.events()][-4:],
            [
                TargetLifecycle.LOST,
                TargetLifecycle.REACQUIRING,
                TargetLifecycle.LOCKED,
                TargetLifecycle.TRACKING,
            ],
        )

    def test_production_reacquire_requires_confirmed_candidate(self) -> None:
        lost_data = {
            "last_seen_position": (7.0, 8.0, 0.0),
            "last_seen_velocity": (0.5, 0.0, 0.0),
            "last_seen_time": 3.5,
        }
        harness = make_harness(
            outcomes={
                SkillName.TRACK: [
                    failed(SkillResultCode.TARGET_LOST, lost_data),
                    succeeded(SkillResultCode.TRACK_COMPLETE),
                ]
            },
            perception_runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
            acknowledge_privileged_oracle=False,
        )
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(
            DetectionCandidate("initial_tracklet", 2.1, 0.8),
            harness.target,
        )
        coordinator.evaluate(
            target_manager=harness.target,
            track=ShortTrackEvidence("initial_tracklet", 2.7, 4, 0.6, True, 0.9),
            semantic=SemanticVerification(
                "initial_tracklet", 2.8, "moving red target", True, 0.9, "qwen-vl"
            ),
            identity=IdentityConsistencyEvidence(
                "initial_tracklet", "target_0", 2.9, True, True, 4, 0.9
            ),
        )
        harness.tick(3.0)  # confirmed SEARCH -> TRACK
        harness.tick(4.0)  # TRACK lost -> REACQUIRE
        self.assertIs(harness.target.lifecycle, TargetLifecycle.REACQUIRING)

        coordinator.register_candidate(
            DetectionCandidate("reacquire_tracklet", 4.1, 0.85),
            harness.target,
        )
        coordinator.evaluate(
            target_manager=harness.target,
            track=ShortTrackEvidence("reacquire_tracklet", 4.7, 4, 0.6, True, 0.9),
            semantic=SemanticVerification(
                "reacquire_tracklet", 4.8, "moving red target", True, 0.9, "qwen-vl"
            ),
            identity=IdentityConsistencyEvidence(
                "reacquire_tracklet", "target_0", 4.9, True, True, 4, 0.9
            ),
        )

        snapshot = harness.tick(5.0)

        self.assertEqual(snapshot.active_skill, "TRACK")
        self.assertIs(snapshot.target.lifecycle, TargetLifecycle.TRACKING)
        self.assertEqual(snapshot.target.target_id, "target_0")
        self.assertEqual(snapshot.target.source, "confirmed_vision")

    def test_track_complete_terminates_target_before_return_goto(self) -> None:
        harness = make_harness()
        harness.start()
        for timestamp in (1.0, 2.0, 3.0, 4.0):
            harness.tick(timestamp)

        self.assertEqual(harness.manager.active_name, SkillName.GOTO)
        self.assertIs(
            harness.agent.snapshot().target.lifecycle,
            TargetLifecycle.TERMINATED,
        )
        self.assertEqual(harness.target.events()[-1].reason, "tracking_complete")

    def test_transition_cursor_is_stable_across_running_ticks(self) -> None:
        harness = make_harness(
            outcomes={
                SkillName.TAKEOFF: [
                    running(),
                    running(),
                    succeeded(SkillResultCode.TAKEOFF_COMPLETE),
                ]
            }
        )
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)

        self.assertEqual(len(harness.manager.transition_log), 1)
        self.assertEqual(harness.target.events(), ())


class MissionAgentSafetyAndLifecycleTests(unittest.TestCase):
    def test_production_agent_rejects_oracle_before_safety_or_skill(self) -> None:
        harness = make_harness(
            perception_runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
            acknowledge_privileged_oracle=False,
        )
        harness.start()
        privileged = replace(
            observation(1.0),
            oracle_target_visible=False,
        )

        snapshot = harness.agent.tick(privileged)

        self.assertEqual(harness.safety.evaluate_calls, 0)
        self.assertEqual(harness.manager.tick_calls, 0)
        self.assertEqual(harness.manager.cancel_task_calls, 1)
        self.assertEqual(snapshot.active_skill, "LAND")
        self.assertIn("perception boundary", snapshot.last_error or "")

    def test_production_agent_rejects_oracle_target_estimate_aliases(self) -> None:
        for source in ("oracle", "oracle_truth", "OrAcLe_BrIdGe"):
            with self.subTest(source=source):
                harness = make_harness(
                    perception_runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
                    acknowledge_privileged_oracle=False,
                )
                harness.start()
                privileged = replace(
                    observation(1.0),
                    target_estimate=privileged_target_estimate(1.0, source),
                )

                snapshot = harness.agent.tick(privileged)

                self.assertEqual(harness.safety.evaluate_calls, 0)
                self.assertEqual(harness.manager.tick_calls, 0)
                self.assertEqual(snapshot.active_skill, "LAND")
                self.assertIn("perception boundary", snapshot.last_error or "")

    def test_repeated_oracle_violation_cannot_starve_fail_safe_land(self) -> None:
        harness = make_harness(
            perception_runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
            acknowledge_privileged_oracle=False,
        )
        harness.start()

        first = harness.agent.tick(
            replace(observation(1.0), oracle_target_visible=False)
        )
        self.assertEqual(first.active_skill, "LAND")
        self.assertEqual(harness.manager.tick_calls, 0)

        final = harness.agent.tick(
            replace(observation(2.0), oracle_target_visible=False)
        )

        self.assertEqual(harness.manager.tick_calls, 1)
        self.assertIs(final.status, AgentStatus.FAILED)

    def test_oracle_evaluation_agent_requires_explicit_acknowledgement(self) -> None:
        with self.assertRaisesRegex(MissionAgentError, "explicit"):
            make_harness(
                perception_runtime_profile=(
                    PerceptionRuntimeProfile.ORACLE_EVALUATION
                ),
                acknowledge_privileged_oracle=False,
            )

        harness = make_harness(
            perception_runtime_profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            acknowledge_privileged_oracle=True,
        )
        harness.start()
        privileged = replace(
            observation(1.0),
            oracle_target_visible=False,
        )
        harness.clock.set(1.0)

        harness.agent.tick(privileged)

        self.assertEqual(harness.safety.evaluate_calls, 1)
        self.assertEqual(harness.manager.tick_calls, 1)

    def test_oracle_evaluation_profile_accepts_oracle_source_alias(self) -> None:
        harness = make_harness(
            perception_runtime_profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            acknowledge_privileged_oracle=True,
        )
        harness.start()
        harness.clock.set(1.0)

        harness.agent.tick(
            replace(
                observation(1.0),
                target_estimate=privileged_target_estimate(
                    1.0,
                    "OrAcLe_TrUtH",
                ),
            )
        )

        self.assertEqual(harness.safety.evaluate_calls, 1)
        self.assertEqual(harness.manager.tick_calls, 1)

    def test_runtime_boundary_violation_cancels_into_land(self) -> None:
        harness = make_harness()
        harness.start()

        snapshot = harness.tick(
            1.0,
            pose=UAVState(60.0, 0.0, 10.0, 0.0),
        )

        self.assertEqual(harness.manager.tick_calls, 0)
        self.assertEqual(harness.manager.cancel_task_calls, 1)
        self.assertEqual(harness.manager.active_name, SkillName.LAND)
        self.assertIs(snapshot.status, AgentStatus.RUNNING)
        final = harness.tick(2.0)
        self.assertIs(final.status, AgentStatus.CANCELED)
        self.assertEqual(final.task_status, "CANCELED")

    def test_timestamp_rollback_triggers_safety_without_manager_tick(self) -> None:
        harness = make_harness(
            outcomes={
                SkillName.TAKEOFF: [running()],
            }
        )
        harness.start()
        harness.tick(2.0)
        before = harness.manager.tick_calls

        snapshot = harness.agent.tick(observation(1.0))

        self.assertEqual(harness.manager.tick_calls, before)
        self.assertIs(harness.safety.decisions[-1].action, SafetyAction.ABORT)
        self.assertEqual(harness.manager.cancel_task_calls, 1)
        self.assertEqual(harness.manager.active_name, SkillName.LAND)
        self.assertIs(snapshot.status, AgentStatus.RUNNING)

        # The rejected frame is never dispatched.  A later valid timestamp may
        # advance fail-safe LAND, but the pending safety ABORT determines the
        # Agent's final result rather than Manager's intermediate CANCELED.
        final = harness.tick(3.0)
        self.assertIs(final.status, AgentStatus.FAILED)

    def test_damaged_observation_reaches_safety_and_never_ticks_skill(self) -> None:
        harness = make_harness()
        harness.start()
        damaged = object.__new__(Observation)
        object.__setattr__(damaged, "uav_id", "uav_1")

        snapshot = harness.agent.tick(damaged)

        self.assertEqual(harness.safety.evaluate_calls, 1)
        self.assertIs(harness.safety.decisions[-1].action, SafetyAction.ABORT)
        self.assertEqual(harness.manager.tick_calls, 0)
        self.assertEqual(harness.manager.cancel_task_calls, 1)
        self.assertEqual(harness.manager.active_name, SkillName.LAND)
        self.assertIs(snapshot.status, AgentStatus.RUNNING)

        final = harness.tick(1.0)
        self.assertIs(final.status, AgentStatus.FAILED)

    def test_shutdown_still_rejects_rollback_before_advancing_land(self) -> None:
        harness = make_harness()
        harness.start()
        harness.tick(2.0, pose=UAVState(60.0, 0.0, 10.0, 0.0))
        self.assertEqual(harness.manager.active_name, SkillName.LAND)
        self.assertEqual(harness.manager.tick_calls, 0)

        rollback = harness.agent.tick(observation(1.0))

        self.assertIs(harness.safety.decisions[-1].action, SafetyAction.ABORT)
        self.assertEqual(harness.manager.tick_calls, 0)
        self.assertIs(rollback.status, AgentStatus.RUNNING)

        final = harness.tick(3.0)
        self.assertEqual(harness.manager.tick_calls, 1)
        self.assertIs(final.status, AgentStatus.FAILED)

    def test_cancel_keeps_agent_running_until_fail_safe_land_finishes(self) -> None:
        harness = make_harness()
        harness.start()

        canceled = harness.agent.cancel()

        self.assertEqual(harness.manager.cancel_task_calls, 1)
        self.assertEqual(harness.manager.active_name, SkillName.LAND)
        self.assertIs(harness.agent.snapshot().status, AgentStatus.RUNNING)
        self.assertIs(
            harness.agent.snapshot().target.lifecycle,
            TargetLifecycle.TERMINATED,
        )
        self.assertNotIn(
            TargetLifecycle.SEARCHING,
            [event.new_state for event in harness.target.events()],
        )
        if canceled is not None:
            self.assertIsInstance(canceled, MissionAgentSnapshot)
        final = harness.tick(1.0)
        self.assertIs(final.status, AgentStatus.CANCELED)
        self.assertIs(final.target.lifecycle, TargetLifecycle.TERMINATED)

    def test_cancel_terminates_an_active_target(self) -> None:
        harness = make_harness()
        harness.start()
        for timestamp in (1.0, 2.0, 3.0):
            harness.tick(timestamp)
        self.assertIs(
            harness.agent.snapshot().target.lifecycle,
            TargetLifecycle.TRACKING,
        )

        harness.clock.set(3.5)
        harness.agent.cancel()
        final = harness.tick(4.0)

        self.assertIs(final.status, AgentStatus.CANCELED)
        self.assertIs(final.target.lifecycle, TargetLifecycle.TERMINATED)

    def test_six_step_mission_succeeds(self) -> None:
        harness = make_harness()
        compiled = harness.start()
        final = run_to_terminal(harness)

        self.assertEqual(len(compiled.task_plan.steps), 6)
        self.assertIs(final.status, AgentStatus.SUCCEEDED)
        self.assertEqual(final.task_status, "SUCCEEDED")
        self.assertIs(final.target.lifecycle, TargetLifecycle.TERMINATED)
        self.assertEqual(harness.manager.tick_calls, 6)

    def test_legacy_five_step_mission_still_succeeds(self) -> None:
        harness = make_harness(validator=LegacyFiveStepValidator())
        compiled = harness.start()
        final = run_to_terminal(harness)

        self.assertEqual(len(compiled.task_plan.steps), 5)
        self.assertIs(final.status, AgentStatus.SUCCEEDED)
        self.assertEqual(harness.manager.tick_calls, 5)

    def test_skill_failure_lands_then_marks_agent_failed(self) -> None:
        harness = make_harness(
            outcomes={
                SkillName.SEARCH: [failed(SkillResultCode.SEARCH_EXHAUSTED)]
            }
        )
        harness.start()
        final = run_to_terminal(harness)

        self.assertIs(final.status, AgentStatus.FAILED)
        self.assertEqual(final.task_status, "FAILED")
        self.assertIs(final.target.lifecycle, TargetLifecycle.TERMINATED)

    def test_reset_terminal_mission_returns_all_managers_to_idle(self) -> None:
        harness = make_harness()
        harness.start()
        run_to_terminal(harness)

        reset_value = harness.agent.reset()

        if reset_value is not None:
            self.assertIsInstance(reset_value, MissionAgentSnapshot)
        snapshot = harness.agent.snapshot()
        self.assertIs(snapshot.status, AgentStatus.IDLE)
        self.assertEqual(snapshot.task_status, "IDLE")
        self.assertIsNone(snapshot.active_skill)
        self.assertIsNone(snapshot.feedback)
        self.assertIsNone(snapshot.last_error)
        self.assertIs(snapshot.target.lifecycle, TargetLifecycle.UNINITIALIZED)
        self.assertEqual(harness.manager.reset_task_calls, 1)

    def test_snapshot_is_defensive(self) -> None:
        harness = make_harness(
            outcomes={
                SkillName.TAKEOFF: [running(), running()],
            }
        )
        harness.start()
        first = harness.tick(1.0)
        self.assertIsNotNone(first.feedback)
        assert first.feedback is not None
        first.feedback["data"]["samples"].append(999)  # type: ignore[index,union-attr]

        second = harness.agent.snapshot()
        assert second.feedback is not None
        self.assertEqual(second.feedback["data"]["samples"], [1])  # type: ignore[index]

        encoded = second.to_dict()
        self.assertEqual(encoded["status"], "RUNNING")
        self.assertEqual(encoded["task_status"], "RUNNING")
        self.assertEqual(encoded["target"]["lifecycle"], "UNINITIALIZED")  # type: ignore[index]
        self.assertEqual(json.loads(json.dumps(encoded)), encoded)
        encoded["feedback"]["data"]["samples"].append(1000)  # type: ignore[index,union-attr]
        third = harness.agent.snapshot()
        assert third.feedback is not None
        self.assertEqual(third.feedback["data"]["samples"], [1])  # type: ignore[index]

    def test_logger_failure_does_not_affect_execution(self) -> None:
        def broken_logger(message: str) -> None:
            raise RuntimeError(f"logger rejected {len(message)} bytes")

        harness = make_harness(logger=broken_logger)
        harness.start()
        final = run_to_terminal(harness)

        self.assertIs(final.status, AgentStatus.SUCCEEDED)
        self.assertEqual(harness.planner.calls, 1)

    def test_illegal_cancel_reset_and_tick_raise_clear_agent_error(self) -> None:
        harness = make_harness()

        with self.assertRaises(MissionAgentError):
            harness.agent.cancel()
        with self.assertRaises(MissionAgentError):
            harness.agent.reset()
        with self.assertRaises(MissionAgentError):
            harness.agent.tick(observation(1.0))


if __name__ == "__main__":
    unittest.main()
