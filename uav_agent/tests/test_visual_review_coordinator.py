from __future__ import annotations

from collections import deque
import json
import unittest

import numpy as np

from agents.mission_agent import MissionAgent
from agents.visual_review_coordinator import (
    RevisionCompletionAction,
    VisualReviewCoordinator,
)
from env.kinematic_uav import KinematicUAV, UAVState
from models import AsyncModelRequest, AsyncModelResult, ModelResponse
from perception.visual_review import VisualReviewGate, VisualReviewMode
from perception.candidate_bank import CandidateBank, CandidateLifecycle
from planner.base import MissionPlanner
from planner.schemas import (
    LandingZoneSpec,
    MissionIntent,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
)
from runtime.events import (
    EventSeverity,
    MissionEvent,
    MissionEventBus,
    MissionEventType,
)
from runtime.frame_store import FrameStore
from runtime.review_scheduler import ReviewScheduler
from runtime.plan_validator import PlanValidator
from runtime.safety_supervisor import SafetyAction, SafetyDecision
from runtime.safety_supervisor import SafetySupervisor
from skills.base import Skill
from skills.hover import HoverSkill, HoverTimeoutFallback
from skills.manager import SkillManager, TaskStatus
from skills.plan import TaskPlan
from skills.search import SearchPhase, SearchSkill
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillName,
    SkillResultCode,
)
from target.target_manager import TargetManager
from target.types import TargetLifecycle, TargetSnapshot, TargetSpec


class ManualClock:
    def __init__(self) -> None:
        self.time_s = 0.0

    def now(self) -> float:
        return self.time_s


class FakeCamera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))


class HoldingSkill(Skill):
    """A deterministic Skill that stays RUNNING until Manager interrupts it."""

    goal_type = SkillGoal

    def _on_tick(self, observation: Observation) -> None:
        del observation


class HoldingSearchSkill(SearchSkill):
    """SEARCH-compatible fake that needs no privileged Oracle fields."""

    def _on_tick(self, observation: Observation) -> None:
        del observation


class SuccessSkill(Skill):
    goal_type = SkillGoal

    def __init__(self, code: SkillResultCode) -> None:
        super().__init__()
        self._code = code

    def _on_tick(self, observation: Observation) -> None:
        del observation
        self._succeed(self._code, "scripted success")


class StaticPlanner(MissionPlanner):
    source = "scripted"

    def plan(self, request: PlannerRequest) -> MissionIntent:
        del request
        return MissionIntent(
            target_description="moving target",
            search_region="search_area",
            track_duration_s=10.0,
            landing_zone="home",
            takeoff_altitude_m=10.0,
        )


class FakeAsyncWorker:
    def __init__(self, uav_id: str = "uav_1") -> None:
        self.uav_id = uav_id
        self.requests: list[AsyncModelRequest] = []
        self.results: deque[AsyncModelResult] = deque()

    def submit(self, request: AsyncModelRequest) -> None:
        self.requests.append(request)

    def poll(
        self,
        *,
        expected_request_id: str | None = None,
        expected_review_id: str | None = None,
        minimum_observation_timestamp_s: float | None = None,
        include_stale: bool = False,
    ) -> AsyncModelResult | None:
        del (
            expected_request_id,
            expected_review_id,
            minimum_observation_timestamp_s,
            include_stale,
        )
        return None if not self.results else self.results.popleft()

    def complete_latest(
        self,
        *,
        decision: str = "NO_RELEVANT_CHANGE",
        recommended_action: str = "CONTINUE",
        description: str = "red weathered target",
        bbox_xyxy_normalized: tuple[float, float, float, float] = (
            0.1,
            0.2,
            0.4,
            0.6,
        ),
        stale: bool = False,
        plan_version: int | None = None,
    ) -> None:
        request = self.requests[-1]
        present = decision in {"POSSIBLE_TARGET", "TARGET_MATCH"}
        content = {
            "schema_version": 1,
            "review_id": request.review_id,
            "mission_id": request.mission_id,
            "uav_id": request.uav_id,
            "plan_version": (
                request.plan_version if plan_version is None else plan_version
            ),
            "observation_timestamp_s": request.observation_timestamp_s,
            "frame_id": request.frame_id,
            "decision": decision,
            "candidate": {
                "present": present,
                "bbox_xyxy_normalized": (
                    list(bbox_xyxy_normalized) if present else None
                ),
                "description": description if present else None,
                "self_reported_confidence": 0.8 if present else None,
            },
            "scene_observations": [],
            "reason_codes": ["test_review"],
            "recommended_action": recommended_action,
        }
        self.results.append(
            AsyncModelResult(
                request_id=request.request_id,
                review_id=request.review_id,
                mission_id=request.mission_id,
                uav_id=request.uav_id,
                plan_version=request.plan_version,
                observation_timestamp_s=request.observation_timestamp_s,
                frame_id=request.frame_id,
                response=ModelResponse(
                    json.dumps(content),
                    "fake-qwen",
                    "stop",
                    {"total_tokens": 11},
                ),
                error_code=None,
                error_message=None,
                stale=stale,
            )
        )


def make_runtime(
    first_skill: SkillName = SkillName.GOTO,
) -> tuple[SkillManager, ManualClock, KinematicUAV]:
    clock = ManualClock()
    uav = KinematicUAV(
        UAVState(0.0, 0.0, 10.0, 0.0),
        max_speed_mps=3.0,
        max_yaw_rate_rad_s=2.0,
    )
    context = SkillContext(uav, FakeCamera(), None, clock, uav_id="uav_1")
    registry = {
        first_skill: (
            SearchSkill() if first_skill is SkillName.SEARCH else HoldingSkill()
        ),
        SkillName.INSPECT: HoldingSkill(),
        SkillName.HOVER: HoverSkill(),
        SkillName.LAND: HoldingSkill(),
    }
    manager = SkillManager(context, registry=registry)
    if first_skill is SkillName.TRACK:
        first = {
            "skill": "TRACK",
            "target_id": "target_1",
            "track_duration": 30.0,
        }
    elif first_skill is SkillName.SEARCH:
        first = {
            "skill": "SEARCH",
            "center": [0.0, 0.0, 0.0],
            "radius": 5.0,
            "target_description": "moving target",
            "search_altitude": 10.0,
            "timeout": 60.0,
        }
    else:
        first = {
            "skill": "GOTO",
            "position": [20.0, 30.0, 10.0],
            "timeout": 60.0,
        }
    manager.start_task(
        TaskPlan.from_dicts(
            [first, {"skill": "LAND"}],
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
        )
    )
    return manager, clock, uav


def observation(clock: ManualClock, uav: KinematicUAV) -> Observation:
    return Observation(
        timestamp=clock.now(),
        uav_pose=uav.get_pose(),
        uav_velocity=uav.get_velocity(),
        camera_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        uav_id="uav_1",
    )


def event(event_type: MissionEventType, timestamp_s: float = 0.0) -> MissionEvent:
    return MissionEvent(
        event_id=f"event_{event_type.value.lower()}",
        mission_id="mission_1",
        uav_id="uav_1",
        plan_version=1,
        timestamp_s=timestamp_s,
        event_type=event_type,
        severity=EventSeverity.WARNING,
        payload={"source": "test_injection"},
    )


def target_snapshot(target_id: str = "target_1") -> TargetSnapshot:
    return TargetSnapshot(
        target_id=target_id,
        description="moving target",
        lifecycle=TargetLifecycle.TRACKING,
        confidence=0.9,
        last_seen_position=None,
        last_seen_velocity=None,
        last_seen_time_s=None,
        source="confirmed_vision",
    )


def tick_coordinator(
    coordinator: VisualReviewCoordinator,
    manager: SkillManager,
    clock: ManualClock,
    uav: KinematicUAV,
    *,
    snapshot: TargetSnapshot | None = None,
    safety_action: SafetyAction = SafetyAction.CONTINUE,
) -> None:
    coordinator.tick(
        observation(clock, uav),
        mission_id="mission_1",
        plan_version=1,
        active_skill=manager.active_name,
        active_step_id=manager.active_planned_step_id,
        target_spec=TargetSpec("moving target"),
        target_snapshot=snapshot or target_snapshot(),
        safety_decision=SafetyDecision(safety_action, "test"),
        mission_elapsed_s=clock.now(),
    )


class VisualReviewCoordinatorTests(unittest.TestCase):
    def make_coordinator(
        self,
        manager: SkillManager,
        worker: FakeAsyncWorker,
        *,
        mode: VisualReviewMode,
        intervals: dict[str, float] | None = None,
        timeout: float = 2.0,
        target_manager: TargetManager | None = None,
        fallback: HoverTimeoutFallback = HoverTimeoutFallback.CANCEL_AND_LAND,
        event_bus: MissionEventBus | None = None,
        candidate_bank: CandidateBank | None = None,
        await_revision_completion: bool = False,
    ) -> VisualReviewCoordinator:
        return VisualReviewCoordinator(
            uav_id="uav_1",
            scheduler=ReviewScheduler(
                intervals_s=intervals or {"GOTO": 1.0, "TRACK": 1.0},
                cooldown_s=0.0,
            ),
            frame_store=FrameStore(max_frames=6, max_bytes=100_000, max_age_s=10.0),
            worker=worker,
            gate=VisualReviewGate(mode=mode, min_consistent_matches=2),
            skill_manager=manager,
            target_manager=target_manager,
            candidate_bank=candidate_bank,
            event_bus=event_bus,
            review_timeout_s=timeout,
            max_result_age_s=5.0,
            blocking_timeout_fallback=fallback,
            await_revision_completion=await_revision_completion,
        )

    def test_shadow_blocking_event_records_without_hover_or_target_effect(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.SHADOW,
        )
        coordinator.submit_event(event(MissionEventType.MULTIPLE_CANDIDATES))

        tick_coordinator(coordinator, manager, clock, uav)

        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertFalse(manager.is_supervisory_paused)
        self.assertEqual(len(worker.requests), 1)
        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)
        record = coordinator.records[-1]
        self.assertEqual(record.disposition, "SHADOW_RECORDED")
        self.assertFalse(record.accepted_for_control)
        self.assertEqual(record.semantic_source, "qwen_vl")
        self.assertEqual(record.geometry_source, "none")
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertEqual(coordinator.revision_events, ())

    def test_blocking_gate_event_starts_hover_only_after_safety_continue(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
        )
        coordinator.submit_event(event(MissionEventType.TARGET_IDENTITY_UNCERTAIN))

        tick_coordinator(
            coordinator,
            manager,
            clock,
            uav,
            safety_action=SafetyAction.CANCEL_AND_LAND,
        )
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertEqual(len(worker.requests), 0)

        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)
        self.assertIs(manager.active_name, SkillName.HOVER)
        self.assertTrue(manager.is_supervisory_paused)
        worker.complete_latest(decision="NO_RELEVANT_CHANGE")

        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)
        manager.tick(observation(clock, uav))
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertTrue(
            any(
                record.reason == "interrupted_step_resumed"
                for record in manager.transition_log
            )
        )

    def test_blocking_timeout_uses_trusted_cancel_and_land_fallback(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            timeout=1.0,
        )
        coordinator.submit_event(event(MissionEventType.MULTIPLE_CANDIDATES))
        tick_coordinator(coordinator, manager, clock, uav)
        self.assertIs(manager.active_name, SkillName.HOVER)

        clock.time_s = 1.0
        tick_coordinator(coordinator, manager, clock, uav)
        manager.tick(observation(clock, uav))

        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(manager.pending_task_result, TaskStatus.FAILED)
        self.assertEqual(coordinator.records[-1].error_code, "TIMEOUT")

    def test_periodic_track_review_is_nonblocking_and_never_ticks_qwen(self) -> None:
        manager, clock, uav = make_runtime(SkillName.TRACK)
        worker = FakeAsyncWorker()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            intervals={"TRACK": 1.0},
        )
        tick_coordinator(coordinator, manager, clock, uav)
        self.assertEqual(len(worker.requests), 0)
        clock.time_s = 1.0
        tick_coordinator(coordinator, manager, clock, uav)

        self.assertEqual(len(worker.requests), 1)
        self.assertIs(manager.active_name, SkillName.TRACK)
        self.assertFalse(manager.is_supervisory_paused)
        # No ModelClient exists in this test: coordinator tick performed only
        # worker submit/poll, proving TRACK never synchronously calls Qwen.

    def test_two_track_mismatches_queue_blocking_identity_review_without_switch(self) -> None:
        manager, clock, uav = make_runtime(SkillName.TRACK)
        worker = FakeAsyncWorker()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            intervals={"TRACK": 0.1},
        )
        tracked = target_snapshot("target_1")
        tick_coordinator(coordinator, manager, clock, uav, snapshot=tracked)
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav, snapshot=tracked)
        worker.complete_latest(decision="TARGET_MISMATCH")
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav, snapshot=tracked)
        self.assertEqual(coordinator.snapshot().consecutive_track_mismatches, 1)
        self.assertIs(manager.active_name, SkillName.TRACK)

        worker.complete_latest(decision="TARGET_MISMATCH")
        clock.time_s = 0.3
        tick_coordinator(coordinator, manager, clock, uav, snapshot=tracked)

        self.assertEqual(coordinator.snapshot().consecutive_track_mismatches, 2)
        self.assertIs(manager.active_name, SkillName.HOVER)
        self.assertEqual(tracked.target_id, "target_1")
        self.assertNotIn("reid", json.dumps(coordinator.records[-1].to_dict()).lower())

    def test_shadow_mismatches_never_publish_identity_control_event(self) -> None:
        manager, clock, uav = make_runtime(SkillName.TRACK)
        worker = FakeAsyncWorker()
        bus = MissionEventBus()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.SHADOW,
            intervals={"TRACK": 0.1},
            event_bus=bus,
        )
        tracked = target_snapshot("target_1")
        tick_coordinator(coordinator, manager, clock, uav, snapshot=tracked)
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav, snapshot=tracked)
        worker.complete_latest(decision="TARGET_MISMATCH")
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav, snapshot=tracked)
        worker.complete_latest(decision="TARGET_MISMATCH")
        clock.time_s = 0.3
        tick_coordinator(coordinator, manager, clock, uav, snapshot=tracked)

        self.assertEqual(coordinator.snapshot().consecutive_track_mismatches, 2)
        self.assertIs(manager.active_name, SkillName.TRACK)
        self.assertFalse(manager.is_supervisory_paused)
        self.assertFalse(
            any(
                item.event_type is MissionEventType.TARGET_IDENTITY_UNCERTAIN
                for item in bus.recent()
            )
        )

    def test_changed_target_and_plan_frame_contract_discard_stale_result(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
        )
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(
            coordinator,
            manager,
            clock,
            uav,
            snapshot=target_snapshot("target_1"),
        )
        worker.complete_latest(decision="TARGET_MATCH")

        clock.time_s = 0.1
        tick_coordinator(
            coordinator,
            manager,
            clock,
            uav,
            snapshot=target_snapshot("target_2"),
        )

        self.assertTrue(coordinator.records[-1].stale)
        self.assertEqual(coordinator.records[-1].error_code, "STALE")
        self.assertFalse(coordinator.records[-1].accepted_for_control)
        self.assertEqual(coordinator.revision_events, ())

    def test_consensus_updates_only_mutable_appearance_notes(self) -> None:
        manager, clock, uav = make_runtime(SkillName.TRACK)
        worker = FakeAsyncWorker()
        target_manager = TargetManager()
        spec = TargetSpec(
            original_description="moving red vehicle",
            immutable_identity_summary="red vehicle assigned by mission",
        )
        target_manager.start_search(spec, 0.0)
        target_manager.set_candidate(
            "target_1",
            timestamp_s=0.0,
            confidence=0.8,
            source="qwen_vl",
        )
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            intervals={"TRACK": 0.1},
            target_manager=target_manager,
        )
        snap = target_manager.snapshot()
        tick_coordinator(coordinator, manager, clock, uav, snapshot=snap)
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav, snapshot=snap)
        worker.complete_latest(decision="TARGET_MATCH", description="dusty red shell")
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav, snapshot=snap)
        self.assertEqual(target_manager.target_spec.mutable_appearance_notes, ())

        worker.complete_latest(decision="TARGET_MATCH", description="dusty red shell")
        clock.time_s = 0.3
        tick_coordinator(coordinator, manager, clock, uav, snapshot=snap)

        updated = target_manager.target_spec
        assert updated is not None
        self.assertEqual(updated.original_description, spec.original_description)
        self.assertEqual(
            updated.immutable_identity_summary,
            spec.immutable_identity_summary,
        )
        self.assertEqual(updated.mutable_appearance_notes, ("dusty red shell",))

    def test_typed_path_blocked_authorizes_exactly_one_replan(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        bus = MissionEventBus()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            event_bus=bus,
            await_revision_completion=True,
        )
        trusted_event = MissionEvent(
            event_id="event_trusted_path_blocked",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            timestamp_s=0.0,
            event_type=MissionEventType.PATH_BLOCKED,
            severity=EventSeverity.WARNING,
            payload={
                "recommended_action": "CONTINUE",
                "event_type": "LOW_VISIBILITY",
                "injected_action": "DO_NOT_REPLAN",
            },
        )
        coordinator.submit_event(trusted_event)
        tick_coordinator(coordinator, manager, clock, uav)
        serialized_prompt = json.dumps(
            [message.to_dict() for message in worker.requests[-1].messages]
        )
        self.assertIn(MissionEventType.PATH_BLOCKED.value, serialized_prompt)
        self.assertNotIn("DO_NOT_REPLAN", serialized_prompt)
        self.assertNotIn("injected_action", serialized_prompt)

        # The routed visual result is valid but grants no candidate consensus.
        worker.complete_latest(
            decision="NO_RELEVANT_CHANGE",
            recommended_action="CONTINUE",
        )
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)

        requested = [
            item
            for item in bus.recent()
            if item.event_type is MissionEventType.PLAN_REVISION_REQUESTED
        ]
        self.assertEqual(len(requested), 1)
        payload = requested[0].to_dict()["payload"]
        self.assertEqual(payload["source"], "trusted_runtime_event")
        self.assertTrue(payload["trusted_runtime_event"])
        self.assertEqual(payload["trigger_event_type"], "PATH_BLOCKED")
        self.assertEqual(payload["trigger_event_id"], trusted_event.event_id)
        self.assertEqual(payload["action"], "REQUEST_REPLAN")
        self.assertNotIn("steps", payload)
        self.assertNotIn("plan", payload)
        self.assertIs(manager.active_name, SkillName.HOVER)
        self.assertIsNotNone(coordinator.pending_revision)

        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)
        self.assertEqual(len(coordinator.revision_events), 1)

    def test_oracle_target_state_is_removed_from_visual_prompt(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.SHADOW,
        )
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        privileged = TargetSnapshot(
            target_id="oracle_truth_target_99",
            description="moving target",
            lifecycle=TargetLifecycle.TRACKING,
            confidence=1.0,
            last_seen_position=(1.0, 2.0, 3.0),
            last_seen_velocity=(4.0, 5.0, 6.0),
            last_seen_time_s=0.0,
            source="oracle",
        )
        tick_coordinator(
            coordinator,
            manager,
            clock,
            uav,
            snapshot=privileged,
        )

        serialized = json.dumps(
            [message.to_dict() for message in worker.requests[-1].messages]
        ).casefold()
        self.assertNotIn("oracle_truth_target_99", serialized)
        self.assertNotIn("oracle_target", serialized)
        self.assertNotIn("[1.0, 2.0, 3.0]", serialized)

    def test_mission_agent_automatically_reviews_each_distinct_frame_once(self) -> None:
        clock = ManualClock()
        uav = KinematicUAV(
            UAVState(0.0, 0.0, 10.0, 0.0),
            max_speed_mps=3.0,
            max_yaw_rate_rad_s=2.0,
        )
        context = SkillContext(uav, FakeCamera(), None, clock, uav_id="uav_1")
        registry = {
            SkillName.TAKEOFF: SuccessSkill(SkillResultCode.TAKEOFF_COMPLETE),
            SkillName.GOTO: HoldingSkill(),
            SkillName.SEARCH: HoldingSkill(),
            SkillName.TRACK: HoldingSkill(),
            SkillName.REACQUIRE: HoldingSkill(),
            SkillName.LAND: HoldingSkill(),
            SkillName.HOVER: HoverSkill(),
        }
        manager = SkillManager(context, registry=registry)
        target = TargetManager()
        worker = FakeAsyncWorker()
        coordinator = VisualReviewCoordinator(
            uav_id="uav_1",
            scheduler=ReviewScheduler(
                intervals_s={"GOTO": 0.5},
                cooldown_s=0.0,
            ),
            frame_store=FrameStore(max_frames=6, max_bytes=100_000, max_age_s=10.0),
            worker=worker,
            gate=VisualReviewGate(mode=VisualReviewMode.SHADOW),
            skill_manager=manager,
            target_manager=target,
        )
        agent = MissionAgent(
            planner=StaticPlanner(),
            validator=PlanValidator(),
            safety=SafetySupervisor(
                scene_min_xyz_m=(-50.0, -50.0, 0.0),
                scene_max_xyz_m=(50.0, 50.0, 30.0),
                max_mission_time_s=300.0,
                max_safe_altitude_m=25.0,
            ),
            skill_manager=manager,
            target_manager=target,
            clock=clock,
            uav_id="uav_1",
            visual_review_coordinator=coordinator,
        )
        world = PlannerWorldContext(
            scene_min_xyz_m=(-50.0, -50.0, 0.0),
            scene_max_xyz_m=(50.0, 50.0, 30.0),
            initial_uav_xyz_m=(0.0, 0.0, 0.0),
            search_regions={
                "search_area": SearchRegionSpec(
                    name="search_area",
                    center_xyz_m=(20.0, 20.0, 0.0),
                    radius_m=10.0,
                    approach_xyz_m=(20.0, 5.0, 10.0),
                    description="search area",
                )
            },
            landing_zones={
                "home": LandingZoneSpec(
                    name="home",
                    position_xy_m=(0.0, 0.0),
                    ground_altitude_m=0.0,
                    description="home pad",
                )
            },
            default_takeoff_altitude_m=10.0,
            default_track_duration_s=10.0,
            search_timeout_s=60.0,
            goto_timeout_s=120.0,
            land_timeout_s=60.0,
        )
        agent.start("find and track the moving target", world)

        clock.time_s = 0.1
        agent.tick(observation(clock, uav))  # TAKEOFF -> GOTO
        clock.time_s = 0.2
        agent.tick(observation(clock, uav))  # establish GOTO review interval
        clock.time_s = 0.7
        first = agent.tick(observation(clock, uav))
        self.assertEqual(len(worker.requests), 1)
        self.assertIsNotNone(first.visual_review)
        self.assertEqual(
            first.visual_review["inflight_request_id"],
            worker.requests[-1].request_id,
        )

        duplicate = agent.tick(observation(clock, uav))
        self.assertEqual(len(worker.requests), 1)
        self.assertEqual(
            duplicate.visual_review["latest_frame_ref"],
            first.visual_review["latest_frame_ref"],
        )

    def test_qwen_candidate_requires_consensus_before_inspect_event(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        bank = CandidateBank(uav_id="uav_1")
        bus = MissionEventBus()
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            candidate_bank=bank,
            event_bus=bus,
        )
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        coordinator.submit_event(
            MissionEvent(
                event_id="event_low_visibility_second",
                mission_id="mission_1",
                uav_id="uav_1",
                plan_version=1,
                timestamp_s=0.1,
                event_type=MissionEventType.LOW_VISIBILITY,
                severity=EventSeverity.WARNING,
                payload={"source": "test_injection"},
            )
        )
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)

        candidates = bank.snapshots()
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertIs(candidate.lifecycle, CandidateLifecycle.PROVISIONAL)
        self.assertEqual(candidate.source, "qwen_vl")
        self.assertEqual(candidate.review_history[0].decision, "POSSIBLE_TARGET")
        self.assertEqual(len(candidate.bbox_history), 1)
        self.assertEqual(len(candidate.frame_history), 1)
        self.assertFalse(hasattr(candidate, "reid_evidence"))
        self.assertEqual(coordinator.revision_events, ())
        self.assertIs(manager.active_name, SkillName.GOTO)

        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)

        revision = coordinator.revision_events[-1]
        payload = revision.to_dict()["payload"]
        self.assertEqual(payload["candidate_id"], candidate.candidate_id)
        self.assertEqual(payload["action"], "INSPECT")
        self.assertEqual(payload["authorization_source"], "visual_consensus")
        self.assertFalse(payload["trusted_runtime_event"])
        self.assertNotIn("position", payload)
        self.assertNotIn("steps", payload)
        self.assertIs(manager.active_name, SkillName.GOTO)
        updated = bank.get(candidate.candidate_id)
        assert updated is not None
        self.assertEqual(len(updated.review_history), 2)

    def test_nonblocking_search_candidate_updates_only_pending_phase(self) -> None:
        manager, clock, uav = make_runtime(SkillName.SEARCH)
        worker = FakeAsyncWorker()
        bank = CandidateBank(uav_id="uav_1")
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            intervals={"SEARCH": 100.0},
            candidate_bank=bank,
        )
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(coordinator, manager, clock, uav)
        self.assertIs(manager.active_name, SkillName.SEARCH)
        self.assertFalse(manager.is_supervisory_paused)

        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="CONTINUE",
        )
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)

        candidate = bank.snapshots()[0]
        feedback = manager.get_feedback()
        self.assertIs(manager.active_name, SkillName.SEARCH)
        self.assertFalse(manager.is_supervisory_paused)
        self.assertEqual(feedback.data["phase"], SearchPhase.CANDIDATE_PENDING.value)
        self.assertEqual(feedback.data["candidate_id"], candidate.candidate_id)
        self.assertEqual(feedback.data["candidate_source"], "qwen_vl")

    def test_search_consensus_without_revision_owner_never_starts_hover(self) -> None:
        manager, clock, uav = make_runtime(SkillName.SEARCH)
        worker = FakeAsyncWorker()
        bank = CandidateBank(uav_id="uav_1")
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            intervals={"SEARCH": 0.1},
            candidate_bank=bank,
            await_revision_completion=False,
        )

        # Establish cadence, then complete two reviews of the same image-space
        # candidate. Consensus may be logged, but no configured revision owner
        # means it cannot take supervisory control or strand SEARCH in HOVER.
        tick_coordinator(coordinator, manager, clock, uav)
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        clock.time_s = 0.3
        tick_coordinator(coordinator, manager, clock, uav)

        self.assertIs(manager.active_name, SkillName.SEARCH)
        self.assertFalse(manager.is_supervisory_paused)
        self.assertIsNone(coordinator.pending_revision)
        self.assertEqual(len(coordinator.revision_events), 1)

    def test_search_consensus_does_not_cross_different_candidates(self) -> None:
        manager, clock, uav = make_runtime(SkillName.SEARCH)
        worker = FakeAsyncWorker()
        bank = CandidateBank(uav_id="uav_1")
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            intervals={"SEARCH": 100.0},
            candidate_bank=bank,
        )
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.0))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(decision="TARGET_MATCH")
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.1))
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)

        worker.complete_latest(
            decision="TARGET_MATCH",
            description="different object",
            bbox_xyxy_normalized=(0.7, 0.7, 0.9, 0.9),
        )
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)

        candidates = bank.snapshots()
        self.assertEqual(len(candidates), 2)
        self.assertNotEqual(candidates[0].candidate_id, candidates[1].candidate_id)
        self.assertEqual(coordinator.records[-2].disposition, "PENDING")
        self.assertEqual(coordinator.records[-1].disposition, "PENDING")
        self.assertFalse(coordinator.records[-1].accepted_for_control)
        self.assertEqual(coordinator.revision_events, ())
        self.assertIs(manager.active_name, SkillName.SEARCH)
        self.assertFalse(manager.is_supervisory_paused)

    def test_search_consensus_accumulates_for_same_associated_candidate(self) -> None:
        manager, clock, uav = make_runtime(SkillName.SEARCH)
        worker = FakeAsyncWorker()
        bank = CandidateBank(uav_id="uav_1")
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            intervals={"SEARCH": 100.0},
            candidate_bank=bank,
        )
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.0))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(decision="TARGET_MATCH")
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.1))
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)

        worker.complete_latest(decision="TARGET_MATCH")
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)

        candidates = bank.snapshots()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0].review_history), 2)
        self.assertEqual(coordinator.records[-1].disposition, "CONSENSUS_REACHED")
        self.assertTrue(coordinator.records[-1].accepted_for_control)

    def test_rejected_candidate_cooldown_cannot_complete_consensus(self) -> None:
        manager, clock, uav = make_runtime(SkillName.SEARCH)
        worker = FakeAsyncWorker()
        bank = CandidateBank(uav_id="uav_1", rejected_cooldown_s=5.0)
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            intervals={"SEARCH": 100.0},
            candidate_bank=bank,
        )
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.0))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(decision="TARGET_MATCH")
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.1))
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)
        candidate_id = bank.snapshots()[0].candidate_id
        bank.reject(candidate_id, timestamp_s=0.1)

        worker.complete_latest(decision="TARGET_MATCH")
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)

        candidate = bank.get(candidate_id)
        assert candidate is not None
        self.assertIs(candidate.lifecycle, CandidateLifecycle.REJECTED)
        self.assertEqual(len(candidate.review_history), 1)
        self.assertIsNone(coordinator.records[-1].candidate_id)
        self.assertEqual(coordinator.records[-1].disposition, "PENDING")
        self.assertFalse(coordinator.records[-1].accepted_for_control)
        self.assertEqual(coordinator.revision_events, ())

    def test_single_blocking_search_candidate_resumes_without_revision(self) -> None:
        manager, clock, uav = make_runtime(SkillName.SEARCH)
        worker = FakeAsyncWorker()
        bank = CandidateBank(uav_id="uav_1")
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            intervals={"SEARCH": 100.0},
            candidate_bank=bank,
            await_revision_completion=True,
        )
        search_skill = manager.skill_registry[SkillName.SEARCH]
        assert isinstance(search_skill, SearchSkill)
        coordinator.submit_event(event(MissionEventType.MULTIPLE_CANDIDATES))
        tick_coordinator(coordinator, manager, clock, uav)
        self.assertIs(manager.active_name, SkillName.HOVER)

        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)

        self.assertIs(manager.active_name, SkillName.HOVER)
        self.assertTrue(manager.is_supervisory_paused)
        self.assertIsNot(search_skill.phase, SearchPhase.CANDIDATE_PENDING)
        self.assertIsNone(coordinator.pending_revision)
        self.assertEqual(coordinator.revision_events, ())

        manager.tick(observation(clock, uav))
        self.assertIs(manager.active_name, SkillName.SEARCH)
        self.assertFalse(manager.is_supervisory_paused)

    def test_overlapping_reviews_reuse_candidate_and_rejection_cooldown(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        bank = CandidateBank(uav_id="uav_1", rejected_cooldown_s=5.0)
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.SHADOW,
            candidate_bank=bank,
        )

        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.0))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(decision="POSSIBLE_TARGET")
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)
        first_id = bank.snapshots()[0].candidate_id

        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.1))
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(decision="POSSIBLE_TARGET")
        clock.time_s = 0.3
        tick_coordinator(coordinator, manager, clock, uav)

        candidates = bank.snapshots()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_id, first_id)
        self.assertEqual(len(candidates[0].bbox_history), 2)
        self.assertEqual(len(candidates[0].review_history), 2)
        revision_count_before_rejection = len(coordinator.revision_events)

        bank.reject(first_id, timestamp_s=0.3)
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.3))
        clock.time_s = 0.4
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(decision="POSSIBLE_TARGET")
        clock.time_s = 0.5
        tick_coordinator(coordinator, manager, clock, uav)

        after_rejection = bank.snapshots()
        self.assertEqual(len(after_rejection), 1)
        self.assertEqual(after_rejection[0].candidate_id, first_id)
        self.assertIs(after_rejection[0].lifecycle, CandidateLifecycle.REJECTED)
        self.assertEqual(len(after_rejection[0].review_history), 2)
        # The cooldown-suppressed review cannot mint a fresh candidate ID.
        self.assertIsNone(coordinator.records[-1].candidate_id)
        self.assertEqual(
            len(coordinator.revision_events),
            revision_count_before_rejection,
        )

    def test_configured_recent_frame_limit_is_enforced(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = VisualReviewCoordinator(
            uav_id="uav_1",
            scheduler=ReviewScheduler(
                intervals_s={"GOTO": 100.0},
                cooldown_s=0.0,
            ),
            frame_store=FrameStore(max_frames=6, max_bytes=100_000, max_age_s=10.0),
            worker=worker,
            gate=VisualReviewGate(mode=VisualReviewMode.SHADOW),
            skill_manager=manager,
            max_recent_frames=2,
        )
        tick_coordinator(coordinator, manager, clock, uav)
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.1))
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)

        content = worker.requests[-1].messages[-1].content
        self.assertIsInstance(content, tuple)
        # One text part plus exactly two configured recent image parts.
        self.assertEqual(len(content), 3)
        with self.assertRaises(ValueError):
            VisualReviewCoordinator(
                uav_id="uav_1",
                scheduler=ReviewScheduler(),
                frame_store=FrameStore(),
                worker=FakeAsyncWorker(),
                max_recent_frames=4,
            )

    def test_deferred_blocking_inspect_revision_keeps_hover_until_resume(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        bank = CandidateBank(uav_id="uav_1")
        coordinator = self.make_coordinator(
            manager,
            worker,
            mode=VisualReviewMode.GATE,
            candidate_bank=bank,
            await_revision_completion=True,
        )
        # First non-blocking observation is provisional only.
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY, 0.0))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        coordinator.submit_event(event(MissionEventType.MULTIPLE_CANDIDATES, 0.1))
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)
        self.assertEqual(coordinator.revision_events, ())
        self.assertIs(manager.active_name, SkillName.HOVER)

        # The second routed review is associated to the same candidate and
        # reaches gate consensus while the trusted blocking HOVER is active.
        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        clock.time_s = 0.2
        tick_coordinator(coordinator, manager, clock, uav)
        manager.tick(observation(clock, uav))

        pending = coordinator.pending_revision
        self.assertIsNotNone(pending)
        assert pending is not None and pending.candidate_id is not None
        self.assertIs(manager.active_name, SkillName.HOVER)
        self.assertTrue(manager.is_supervisory_paused)
        with self.assertRaisesRegex(Exception, "event_id mismatch"):
            coordinator.acknowledge_revision_handoff(event_id="event_wrong")

        coordinator.complete_revision(
            RevisionCompletionAction.RESUME,
        )
        manager.tick(observation(clock, uav))

        self.assertIsNone(coordinator.pending_revision)
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertEqual(manager.task_plan.plan_version, 1)

    def test_mission_agent_periodic_search_consensus_auto_hovers_for_revision(self) -> None:
        clock = ManualClock()
        uav = KinematicUAV(
            UAVState(0.0, 0.0, 10.0, 0.0),
            max_speed_mps=3.0,
            max_yaw_rate_rad_s=2.0,
        )
        context = SkillContext(uav, FakeCamera(), None, clock, uav_id="uav_1")
        registry = {
            SkillName.TAKEOFF: SuccessSkill(SkillResultCode.TAKEOFF_COMPLETE),
            SkillName.GOTO: SuccessSkill(SkillResultCode.GOAL_REACHED),
            SkillName.SEARCH: HoldingSearchSkill(),
            SkillName.INSPECT: HoldingSkill(),
            SkillName.TRACK: HoldingSkill(),
            SkillName.REACQUIRE: HoldingSkill(),
            SkillName.LAND: HoldingSkill(),
            SkillName.HOVER: HoverSkill(),
        }
        manager = SkillManager(context, registry=registry)
        target = TargetManager()
        bank = CandidateBank(uav_id="uav_1")
        worker = FakeAsyncWorker()
        coordinator = VisualReviewCoordinator(
            uav_id="uav_1",
            scheduler=ReviewScheduler(
                intervals_s={"SEARCH": 0.1, "INSPECT": 100.0},
                cooldown_s=0.0,
            ),
            frame_store=FrameStore(max_frames=8, max_bytes=100_000, max_age_s=10.0),
            worker=worker,
            gate=VisualReviewGate(mode=VisualReviewMode.GATE),
            skill_manager=manager,
            target_manager=target,
            candidate_bank=bank,
            await_revision_completion=True,
        )
        safety = SafetySupervisor(
            scene_min_xyz_m=(-50.0, -50.0, 0.0),
            scene_max_xyz_m=(50.0, 50.0, 30.0),
            max_mission_time_s=300.0,
            max_safe_altitude_m=25.0,
        )
        agent = MissionAgent(
            planner=StaticPlanner(),
            validator=PlanValidator(),
            safety=safety,
            skill_manager=manager,
            target_manager=target,
            clock=clock,
            uav_id="uav_1",
            visual_review_coordinator=coordinator,
        )
        world = PlannerWorldContext(
            scene_min_xyz_m=(-50.0, -50.0, 0.0),
            scene_max_xyz_m=(50.0, 50.0, 30.0),
            initial_uav_xyz_m=(0.0, 0.0, 0.0),
            search_regions={
                "search_area": SearchRegionSpec(
                    name="search_area",
                    center_xyz_m=(20.0, 20.0, 0.0),
                    radius_m=10.0,
                    approach_xyz_m=(20.0, 5.0, 10.0),
                    description="search area",
                )
            },
            landing_zones={
                "home": LandingZoneSpec(
                    name="home",
                    position_xy_m=(0.0, 0.0),
                    ground_altitude_m=0.0,
                    description="home pad",
                )
            },
            default_takeoff_altitude_m=10.0,
            default_track_duration_s=10.0,
            search_timeout_s=60.0,
            goto_timeout_s=120.0,
            land_timeout_s=60.0,
        )
        agent.start("find and inspect the moving target", world)
        clock.time_s = 0.1
        agent.tick(observation(clock, uav))  # TAKEOFF -> GOTO
        clock.time_s = 0.2
        search_snapshot = agent.tick(observation(clock, uav))  # GOTO -> SEARCH
        self.assertEqual(search_snapshot.active_skill, "SEARCH")
        self.assertIs(target.lifecycle, TargetLifecycle.SEARCHING)

        # Establish SEARCH's periodic cadence; no candidate event is injected.
        clock.time_s = 0.3
        agent.tick(observation(clock, uav))
        clock.time_s = 0.4
        first_request = agent.tick(observation(clock, uav))
        self.assertEqual(
            first_request.active_skill,
            "SEARCH",
            first_request.to_dict(),
        )
        self.assertEqual(len(worker.requests), 1)

        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        clock.time_s = 0.5
        provisional = agent.tick(observation(clock, uav))
        self.assertEqual(provisional.active_skill, "SEARCH")
        self.assertEqual(coordinator.revision_events, ())
        self.assertIsNone(coordinator.pending_revision)
        self.assertEqual(len(worker.requests), 2)

        worker.complete_latest(
            decision="POSSIBLE_TARGET",
            recommended_action="INSPECT",
        )
        clock.time_s = 0.6
        waiting = agent.tick(observation(clock, uav))
        self.assertEqual(waiting.active_skill, "HOVER")
        self.assertIsNotNone(coordinator.pending_revision)
        pending = coordinator.pending_revision
        assert pending is not None and pending.candidate_id is not None

        revision_payload = pending.event.to_dict()["payload"]
        self.assertEqual(revision_payload["action"], "INSPECT")
        self.assertEqual(revision_payload["candidate_id"], pending.candidate_id)
        self.assertNotIn("steps", revision_payload)
        self.assertNotIn("position", revision_payload)
        self.assertIs(target.lifecycle, TargetLifecycle.SEARCHING)
        candidate = bank.get(pending.candidate_id)
        assert candidate is not None
        self.assertIs(candidate.lifecycle, CandidateLifecycle.PROVISIONAL)


if __name__ == "__main__":
    unittest.main()
