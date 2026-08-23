"""Main-thread orchestration for sparse asynchronous visual reviews.

The coordinator is deliberately narrower than :class:`MissionAgent`: it owns
frame buffering, review scheduling, routed result acceptance, and the trusted
supervisory-HOVER handshake.  Model HTTP work remains in
``AsyncModelWorker``; this module never calls ``ModelClient.chat`` and never
passes Oracle fields to the verifier.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from numbers import Real
import os
import re
from typing import Protocol
from types import MappingProxyType

from common.ids import (
    generate_routing_id,
    validate_mission_id,
    validate_routing_id,
    validate_uav_id,
)
from models import AsyncModelRequest, AsyncModelResult, ModelProtocolError
from perception.qwen_vlm_verifier import (
    QwenVLMVerifier,
    VisualReviewFrame,
    VisualReviewInput,
)
from perception.candidate_bank import (
    CandidateBank,
    CandidateLifecycle,
    CandidateReviewRef,
)
from perception.visual_review import (
    QwenVisualReview,
    ReviewDisposition,
    VisualReviewAcceptance,
    VisualReviewAction,
    VisualReviewDecision,
    VisualReviewExpectation,
    VisualReviewGate,
    VisualReviewMode,
    VisualReviewParseErrorCode,
    VisualReviewProtocolError,
    VisualReviewStaleReason,
)
from runtime.events import (
    EventSeverity,
    MissionEvent,
    MissionEventBus,
    MissionEventType,
)
from runtime.frame_store import FrameRef, FrameStore
from runtime.review_scheduler import (
    ReviewScheduleReason,
    ReviewScheduler,
    ReviewTicket,
)
from runtime.safety_supervisor import SafetyAction, SafetyDecision
from skills.hover import HoverTimeoutFallback
from skills.manager import ExecutionKind, SkillManager, TaskStatus, TransitionRecord
from skills.plan import TaskPlan
from skills.types import Observation, SkillName, SkillStatus
from target.target_manager import TargetManager, TargetStateError
from target.types import TargetLifecycle, TargetSnapshot, TargetSpec


class _AsyncReviewWorker(Protocol):
    uav_id: str

    def submit(self, request: AsyncModelRequest) -> None: ...

    def poll(
        self,
        *,
        expected_request_id: str | None = None,
        expected_review_id: str | None = None,
        minimum_observation_timestamp_s: float | None = None,
        include_stale: bool = False,
    ) -> AsyncModelResult | None: ...


class _VisualReviewVerifier(Protocol):
    def build_async_request(
        self,
        review_input: VisualReviewInput,
        *,
        request_id: str,
    ) -> AsyncModelRequest: ...

    def parse_async_result(
        self,
        result: AsyncModelResult,
        *,
        expectation: VisualReviewExpectation,
    ) -> QwenVisualReview: ...


class VisualReviewCoordinatorError(RuntimeError):
    """Raised when a trusted review-orchestration invariant is violated."""


class RevisionCompletionAction(str, Enum):
    RESUME = "RESUME"
    REPLACE = "REPLACE"


@dataclass(frozen=True, slots=True)
class PendingPlanRevision:
    """A structured hand-off to an independent asynchronous revision planner."""

    event: MissionEvent
    request_id: str
    review_id: str
    candidate_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event.to_dict(),
            "request_id": self.request_id,
            "review_id": self.review_id,
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True, slots=True)
class VisualReviewRecord:
    """Sparse review log entry; it intentionally contains no image bytes."""

    request_id: str
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    observation_timestamp_s: float
    frame_id: str
    completed_timestamp_s: float
    blocking: bool
    stale: bool
    stale_reasons: tuple[str, ...]
    decision: str | None
    disposition: str | None
    accepted_for_control: bool
    bbox_xyxy_normalized: tuple[float, float, float, float] | None
    token_usage: Mapping[str, int]
    latency_s: float
    error_code: str | None
    semantic_source: str = "qwen_vl"
    geometry_source: str = "none"
    candidate_id: str | None = None
    response_text_length: int | None = None
    response_text_tail: str | None = None

    def __post_init__(self) -> None:
        reasons = tuple(self.stale_reasons)
        allowed_reasons = {item.value for item in VisualReviewStaleReason}
        if any(
            not isinstance(reason, str) or reason not in allowed_reasons
            for reason in reasons
        ):
            raise ValueError("stale_reasons contains an unsupported reason")
        if len(set(reasons)) != len(reasons):
            raise ValueError("stale_reasons must not contain duplicates")
        if self.stale != bool(reasons):
            raise ValueError("stale must equal bool(stale_reasons)")
        object.__setattr__(self, "stale_reasons", reasons)
        if self.candidate_id is not None:
            object.__setattr__(
                self,
                "candidate_id",
                validate_routing_id(self.candidate_id, "candidate_id"),
            )
        copied_usage: dict[str, int] = {}
        for key, value in self.token_usage.items():
            if not isinstance(key, str):
                raise TypeError("token_usage keys must be strings")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("token_usage values must be non-negative integers")
            copied_usage[key] = value
        object.__setattr__(self, "token_usage", MappingProxyType(copied_usage))
        debug_pair = (self.response_text_length, self.response_text_tail)
        if (debug_pair[0] is None) != (debug_pair[1] is None):
            raise ValueError(
                "response_text_length and response_text_tail must be set together"
            )
        if self.response_text_length is not None:
            if (
                isinstance(self.response_text_length, bool)
                or not isinstance(self.response_text_length, int)
                or self.response_text_length < 0
            ):
                raise ValueError("response_text_length must be non-negative")
            if not isinstance(self.response_text_tail, str):
                raise TypeError("response_text_tail must be a string")
            if len(self.response_text_tail) > 500:
                raise ValueError("response_text_tail must contain at most 500 characters")
            lowered = self.response_text_tail.casefold()
            if "base64," in lowered or "api_key" in lowered or "authorization" in lowered:
                raise ValueError("response_text_tail contains forbidden secret/image data")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "review_id": self.review_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "observation_timestamp_s": self.observation_timestamp_s,
            "frame_id": self.frame_id,
            "completed_timestamp_s": self.completed_timestamp_s,
            "blocking": self.blocking,
            "stale": self.stale,
            "stale_reasons": list(self.stale_reasons),
            "decision": self.decision,
            "disposition": self.disposition,
            "accepted_for_control": self.accepted_for_control,
            "bbox_xyxy_normalized": (
                None
                if self.bbox_xyxy_normalized is None
                else list(self.bbox_xyxy_normalized)
            ),
            "token_usage": dict(self.token_usage),
            "latency_s": self.latency_s,
            "error_code": self.error_code,
            "semantic_source": self.semantic_source,
            "geometry_source": self.geometry_source,
            "candidate_id": self.candidate_id,
            "response_text_length": self.response_text_length,
            "response_text_tail": self.response_text_tail,
        }


@dataclass(frozen=True, slots=True)
class VisualReviewCoordinatorSnapshot:
    uav_id: str
    latest_frame_ref: FrameRef | None
    inflight_request_id: str | None
    inflight_review_id: str | None
    inflight_blocking: bool
    supervisory_hover_active: bool
    consecutive_track_mismatches: int
    review_count: int
    stale_count: int
    last_record: VisualReviewRecord | None
    last_candidate_id: str | None
    pending_revision: PendingPlanRevision | None

    def to_dict(self) -> dict[str, object]:
        return {
            "uav_id": self.uav_id,
            "latest_frame_ref": (
                None if self.latest_frame_ref is None else self.latest_frame_ref.to_dict()
            ),
            "inflight_request_id": self.inflight_request_id,
            "inflight_review_id": self.inflight_review_id,
            "inflight_blocking": self.inflight_blocking,
            "supervisory_hover_active": self.supervisory_hover_active,
            "consecutive_track_mismatches": self.consecutive_track_mismatches,
            "review_count": self.review_count,
            "stale_count": self.stale_count,
            "last_record": (
                None if self.last_record is None else self.last_record.to_dict()
            ),
            "last_candidate_id": self.last_candidate_id,
            "pending_revision": (
                None
                if self.pending_revision is None
                else self.pending_revision.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class _PendingReview:
    ticket: ReviewTicket
    expectation: VisualReviewExpectation
    frame_ref: FrameRef
    step_id: str
    target_id: str | None
    blocking_hover_started: bool
    trigger_event: MissionEvent | None


_REVIEWABLE_SKILL_NAMES = frozenset({"GOTO", "SEARCH", "INSPECT", "TRACK"})


class VisualReviewCoordinator:
    """Coordinate asynchronous Qwen reviews for one routed UAV mission.

    ``tick`` is non-blocking with respect to the model: it only stores a frame,
    polls already completed work, builds/submits a request, and returns.  A
    blocking event may request trusted supervisory HOVER only after the caller
    supplies a ``SafetyDecision(CONTINUE, ...)``.  Shadow mode never changes
    Skill or target state, including for events labelled blocking.
    """

    def __init__(
        self,
        *,
        uav_id: str,
        scheduler: ReviewScheduler,
        frame_store: FrameStore,
        worker: _AsyncReviewWorker,
        verifier: _VisualReviewVerifier | None = None,
        gate: VisualReviewGate | None = None,
        skill_manager: SkillManager | None = None,
        target_manager: TargetManager | None = None,
        candidate_bank: CandidateBank | None = None,
        event_bus: MissionEventBus | None = None,
        review_timeout_s: float = 20.0,
        max_result_age_s: float = 20.0,
        hover_position_tolerance_m: float = 0.25,
        hover_max_correction_speed_mps: float = 0.5,
        blocking_timeout_fallback: HoverTimeoutFallback | str = (
            HoverTimeoutFallback.CANCEL_AND_LAND
        ),
        max_pending_events: int = 32,
        max_records: int = 256,
        track_mismatch_threshold: int = 2,
        await_revision_completion: bool = False,
        max_recent_frames: int = 3,
        candidate_association_iou_threshold: float = 0.5,
        debug_model_responses: bool | None = None,
    ) -> None:
        self._uav_id = validate_uav_id(uav_id)
        if not isinstance(scheduler, ReviewScheduler):
            raise TypeError("scheduler must be a ReviewScheduler")
        if not isinstance(frame_store, FrameStore):
            raise TypeError("frame_store must be a FrameStore")
        if getattr(worker, "uav_id", None) != self._uav_id:
            raise ValueError("worker.uav_id must match coordinator uav_id")
        if not callable(getattr(worker, "submit", None)) or not callable(
            getattr(worker, "poll", None)
        ):
            raise TypeError("worker must provide non-blocking submit and poll")
        selected_verifier = QwenVLMVerifier() if verifier is None else verifier
        if not callable(getattr(selected_verifier, "build_async_request", None)) or not callable(
            getattr(selected_verifier, "parse_async_result", None)
        ):
            raise TypeError("verifier must build and parse asynchronous reviews")
        selected_gate = VisualReviewGate() if gate is None else gate
        if not isinstance(selected_gate, VisualReviewGate):
            raise TypeError("gate must be a VisualReviewGate")
        if skill_manager is not None:
            if not isinstance(skill_manager, SkillManager):
                raise TypeError("skill_manager must be a SkillManager or None")
            if skill_manager.uav_id != self._uav_id:
                raise ValueError("SkillManager uav_id must match coordinator uav_id")
        if target_manager is not None and not isinstance(target_manager, TargetManager):
            raise TypeError("target_manager must be a TargetManager or None")
        if candidate_bank is not None:
            if not isinstance(candidate_bank, CandidateBank):
                raise TypeError("candidate_bank must be a CandidateBank or None")
            if candidate_bank.uav_id != self._uav_id:
                raise ValueError("CandidateBank uav_id must match coordinator uav_id")
        if event_bus is not None and not isinstance(event_bus, MissionEventBus):
            raise TypeError("event_bus must be a MissionEventBus or None")
        if not isinstance(await_revision_completion, bool):
            raise TypeError("await_revision_completion must be bool")
        if debug_model_responses is not None and not isinstance(
            debug_model_responses, bool
        ):
            raise TypeError("debug_model_responses must be bool or None")

        self._review_timeout_s = _positive_finite(review_timeout_s, "review_timeout_s")
        self._max_result_age_s = _positive_finite(max_result_age_s, "max_result_age_s")
        self._hover_position_tolerance_m = _positive_finite(
            hover_position_tolerance_m, "hover_position_tolerance_m"
        )
        self._hover_max_correction_speed_mps = _positive_finite(
            hover_max_correction_speed_mps, "hover_max_correction_speed_mps"
        )
        try:
            self._blocking_timeout_fallback = HoverTimeoutFallback(
                blocking_timeout_fallback
            )
        except (TypeError, ValueError):
            raise ValueError(
                "blocking_timeout_fallback must be RESUME_PREVIOUS or CANCEL_AND_LAND"
            ) from None
        _positive_int(max_pending_events, "max_pending_events")
        _positive_int(max_records, "max_records")
        self._track_mismatch_threshold = _positive_int(
            track_mismatch_threshold, "track_mismatch_threshold"
        )
        self._max_recent_frames = _positive_int(
            max_recent_frames,
            "max_recent_frames",
        )
        if self._max_recent_frames > 3:
            raise ValueError("max_recent_frames must be between 1 and 3")
        self._candidate_association_iou_threshold = _unit_interval(
            candidate_association_iou_threshold,
            "candidate_association_iou_threshold",
        )
        self._debug_model_responses = (
            _debug_visual_review_from_environment()
            if debug_model_responses is None
            else debug_model_responses
        )

        self._scheduler = scheduler
        self._frame_store = frame_store
        self._worker = worker
        self._verifier = selected_verifier
        self._gate = selected_gate
        self._skill_manager = skill_manager
        self._target_manager = target_manager
        self._candidate_bank = candidate_bank
        self._event_bus = event_bus
        self._await_revision_completion = await_revision_completion
        self._pending_events: deque[MissionEvent] = deque(maxlen=max_pending_events)
        self._records: deque[VisualReviewRecord] = deque(maxlen=max_records)
        self._pending: _PendingReview | None = None
        self._latest_frame_ref: FrameRef | None = None
        self._last_frame_timestamp_s: float | None = None
        self._route: tuple[str, int] | None = None
        self._gate_target_id: str | None = None
        self._consecutive_track_mismatches = 0
        self._stale_count = 0
        self._last_candidate_id: str | None = None
        self._revision_events: deque[MissionEvent] = deque(maxlen=max_records)
        self._pending_revision: PendingPlanRevision | None = None

    @property
    def uav_id(self) -> str:
        return self._uav_id

    @property
    def mode(self) -> VisualReviewMode:
        return self._gate.mode

    @property
    def records(self) -> tuple[VisualReviewRecord, ...]:
        return tuple(self._records)

    @property
    def revision_events(self) -> tuple[MissionEvent, ...]:
        return tuple(self._revision_events)

    @property
    def pending_revision(self) -> PendingPlanRevision | None:
        return self._pending_revision

    def validate_agent_bindings(
        self,
        skill_manager: SkillManager,
        target_manager: TargetManager,
    ) -> None:
        """Reject accidental cross-agent manager wiring at construction time."""

        if self._skill_manager is not None and self._skill_manager is not skill_manager:
            raise VisualReviewCoordinatorError(
                "coordinator and MissionAgent must share the same SkillManager"
            )
        if self._target_manager is not None and self._target_manager is not target_manager:
            raise VisualReviewCoordinatorError(
                "coordinator and MissionAgent must share the same TargetManager"
            )

    def submit_event(self, event: MissionEvent) -> None:
        """Queue one routed, image-free event for the next new frame."""

        if not isinstance(event, MissionEvent):
            raise TypeError("event must be a MissionEvent")
        if event.uav_id != self._uav_id:
            raise VisualReviewCoordinatorError("event uav_id does not match coordinator")
        if self._route is not None and (
            event.mission_id != self._route[0] or event.plan_version != self._route[1]
        ):
            raise VisualReviewCoordinatorError("event route is stale")
        self._pending_events.append(event)
        if self._event_bus is not None:
            self._event_bus.publish(event)

    def observe_skill_transition(
        self,
        record: TransitionRecord,
    ) -> MissionEvent | None:
        """Publish the routed HOLD-established handshake exactly once.

        MissionAgent calls this while consuming each Manager transition, so a
        hold established while the revision worker owns model polling is not
        delayed until visual-review polling resumes.
        """

        if not isinstance(record, TransitionRecord):
            raise TypeError("record must be a TransitionRecord")
        if record.uav_id != self._uav_id:
            raise VisualReviewCoordinatorError("transition uav_id mismatch")
        if record.reason != "HOLD_ESTABLISHED":
            return None
        if (
            record.old_skill is not SkillName.HOVER
            or record.new_skill is not SkillName.HOVER
            or record.old_status is not SkillStatus.RUNNING
        ):
            raise VisualReviewCoordinatorError(
                "HOLD_ESTABLISHED transition has invalid HOVER lifecycle"
            )
        if self._route is not None and self._route != (
            record.mission_id,
            record.plan_version,
        ):
            raise VisualReviewCoordinatorError(
                "HOLD_ESTABLISHED transition route is stale"
            )
        event = MissionEvent(
            event_id=generate_routing_id("event"),
            mission_id=record.mission_id,
            uav_id=record.uav_id,
            plan_version=record.plan_version,
            timestamp_s=record.timestamp,
            event_type=MissionEventType.HOLD_ESTABLISHED,
            severity=EventSeverity.INFO,
            payload={
                "source": "skill_manager",
                "step_id": record.new_step_id,
                "invocation_id": record.invocation_id,
            },
        )
        if self._event_bus is not None:
            self._event_bus.publish(event)
        return event

    def complete_revision(
        self,
        action: RevisionCompletionAction | str,
        *,
        replacement_plan: TaskPlan | None = None,
    ) -> TaskStatus:
        """Complete one deferred revision on the MissionAgent main thread.

        ``REPLACE`` accepts only a TaskPlan already compiled, validated, and
        preflight-approved by the trusted revision pipeline. SkillManager then
        rechecks routing/version/prefix invariants before atomic replacement.
        The model worker never calls this method.
        """

        try:
            selected = RevisionCompletionAction(action)
        except (TypeError, ValueError):
            raise ValueError("action must be RESUME or REPLACE") from None
        if self._pending_revision is None:
            raise VisualReviewCoordinatorError("no plan revision is awaiting completion")
        manager = self._skill_manager
        if manager is None:
            raise VisualReviewCoordinatorError("revision completion requires SkillManager")
        if selected is RevisionCompletionAction.RESUME:
            if replacement_plan is not None:
                raise ValueError("RESUME does not accept replacement_plan")
            status = manager.resume_interrupted_step()
        else:
            if not isinstance(replacement_plan, TaskPlan):
                raise TypeError("REPLACE requires a validated replacement TaskPlan")
            status = manager.replace_interrupted_step_and_suffix(replacement_plan)
        self._pending_revision = None
        return status

    def acknowledge_revision_handoff(
        self,
        *,
        event_id: str,
    ) -> PendingPlanRevision:
        """Transfer an outstanding wait to ``PlanRevisionCoordinator``.

        The caller must invoke this only after the independent coordinator has
        accepted the same structured event and assumed responsibility for the
        existing supervisory HOVER plus its trusted fallback. This method has
        no Skill/controller side effect.
        """

        normalized_event_id = validate_routing_id(event_id, "event_id")
        pending = self._pending_revision
        if pending is None:
            raise VisualReviewCoordinatorError("no plan revision is awaiting handoff")
        if pending.event.event_id != normalized_event_id:
            raise VisualReviewCoordinatorError("revision handoff event_id mismatch")
        if (
            self._skill_manager is not None
            and not self._skill_manager.is_supervisory_paused
        ):
            raise VisualReviewCoordinatorError(
                "cannot hand off after supervisory HOVER has resolved"
            )
        self._pending_revision = None
        return pending

    def tick(
        self,
        observation: Observation,
        *,
        mission_id: str,
        plan_version: int,
        active_skill: SkillName | None,
        active_step_id: str | None,
        target_spec: TargetSpec,
        target_snapshot: TargetSnapshot,
        safety_decision: SafetyDecision,
        skill_feedback: Mapping[str, object] | None = None,
        mission_elapsed_s: float = 0.0,
    ) -> VisualReviewCoordinatorSnapshot:
        """Ingest one new frame, poll, and possibly submit without HTTP waiting."""

        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        observation.validate()
        if observation.uav_id != self._uav_id:
            raise VisualReviewCoordinatorError("Observation uav_id mismatch")
        mission = validate_mission_id(mission_id)
        version = _positive_int(plan_version, "plan_version")
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        if not isinstance(target_snapshot, TargetSnapshot):
            raise TypeError("target_snapshot must be a TargetSnapshot")
        if not isinstance(safety_decision, SafetyDecision):
            raise TypeError("safety_decision must be a SafetyDecision")
        now = _nonnegative_finite(observation.timestamp, "observation.timestamp")
        elapsed = _nonnegative_finite(mission_elapsed_s, "mission_elapsed_s")
        route = (mission, version)
        if self._route is None:
            self._route = route
        elif self._route != route:
            # An in-flight response from an older revision/mission can never
            # affect the new route.  Keep it drainable and reset semantic
            # consensus, but never silently rewrite its expectation.
            self._route = route
            self._gate.reset()
            self._gate_target_id = None
            self._consecutive_track_mismatches = 0

        if self._last_frame_timestamp_s is not None and now <= self._last_frame_timestamp_s:
            if now == self._last_frame_timestamp_s:
                return self.snapshot()
            raise VisualReviewCoordinatorError("visual frame timestamp moved backwards")

        frame_ref = self._frame_store.add_frame(
            uav_id=self._uav_id,
            frame_id=generate_routing_id("frame"),
            timestamp_s=now,
            rgb=observation.camera_rgb,
        )
        self._latest_frame_ref = (
            frame_ref
            if self._frame_store.contains(frame_ref)
            else self._frame_store.latest_ref(uav_id=self._uav_id)
        )
        self._last_frame_timestamp_s = now
        if (
            self._pending_revision is not None
            and self._skill_manager is not None
            and not self._skill_manager.is_supervisory_paused
        ):
            # The trusted HOVER timeout/cancel policy resolved the wait before
            # a revision result arrived. A late revision can no longer apply.
            self._pending_revision = None

        # Hard safety always wins. Reviews are not submitted, and especially
        # cannot start HOVER, until the trusted supervisor allows CONTINUE.
        if safety_decision.action is not SafetyAction.CONTINUE:
            self.abort_pending(
                reason_code="SAFETY_SUPERSEDED",
                timestamp_s=now,
            )
            return self.snapshot()
        poll_failed = self._poll_result(
            now=now,
            mission_id=mission,
            plan_version=version,
            current_target_id=target_snapshot.target_id,
            current_step_id=active_step_id,
        )
        timed_out = self._handle_timeout(now)
        if poll_failed or timed_out:
            # Transport/timeout failures must not immediately create another
            # request on the same frame, especially with zero test cooldown.
            return self.snapshot()

        # Polling may have promoted repeated ordinary SEARCH evidence into a
        # trusted supervisory HOVER. Do not schedule from the caller's now-
        # stale active_skill argument after that state transition.
        if (
            self._skill_manager is not None
            and self._skill_manager.is_supervisory_paused
        ):
            return self.snapshot()

        if (
            self._pending is not None
            or active_skill is None
            or active_skill.value not in _REVIEWABLE_SKILL_NAMES
        ):
            self._drain_orphan_result(now)
            return self.snapshot()
        if active_step_id is None:
            raise VisualReviewCoordinatorError("reviewable Skill has no active step_id")

        event = self._next_event_for_route(mission, version, now)
        decision = self._scheduler.schedule(
            mission_id=mission,
            uav_id=self._uav_id,
            plan_version=version,
            skill_name=active_skill.value,
            timestamp_s=now,
            event=event,
        )
        if not decision.should_submit:
            if event is not None and decision.reason in {
                ReviewScheduleReason.COOLDOWN,
                ReviewScheduleReason.IN_FLIGHT,
            }:
                self._pending_events.appendleft(event)
            self._drain_orphan_result(now)
            return self.snapshot()
        ticket = decision.ticket
        assert ticket is not None
        blocking_hover_started = False
        frame_pinned = False
        try:
            # Building the multimodal input is part of the scheduler
            # reservation transaction.  A tiny store or explicit eviction
            # may reject the latest frame; that failure must still release
            # the reserved per-UAV ticket.
            review_input = self._build_input(
                ticket=ticket,
                latest=frame_ref,
                target_spec=target_spec,
                target_snapshot=_prompt_safe_target_snapshot(target_snapshot),
                active_step_id=active_step_id,
                skill_feedback=skill_feedback,
                mission_elapsed_s=elapsed,
                trigger_event_type=(None if event is None else event.event_type),
            )
            request = self._verifier.build_async_request(
                review_input,
                request_id=ticket.request_id,
            )
            self._frame_store.pin(frame_ref)
            frame_pinned = True
            self._worker.submit(request)
            if ticket.blocking and self._gate.mode is VisualReviewMode.GATE:
                if self._skill_manager is None:
                    raise VisualReviewCoordinatorError(
                        "gate-mode blocking review requires a SkillManager"
                    )
                self._skill_manager.interrupt_with_hover(
                    _hover_reason(event),
                    max_wait_s=self._review_timeout_s,
                    position_tolerance_m=self._hover_position_tolerance_m,
                    max_correction_speed_mps=self._hover_max_correction_speed_mps,
                    timeout_fallback=self._blocking_timeout_fallback,
                    defer_observation_timestamp_s=now,
                )
                blocking_hover_started = True
        except Exception:
            if frame_pinned:
                self._frame_store.unpin(frame_ref)
            self._scheduler.mark_timed_out(
                uav_id=self._uav_id,
                request_id=ticket.request_id,
                review_id=ticket.review_id,
                timestamp_s=now,
            )
            raise
        self._pending = _PendingReview(
            ticket=ticket,
            expectation=review_input.expectation,
            frame_ref=frame_ref,
            step_id=active_step_id,
            target_id=target_snapshot.target_id,
            blocking_hover_started=blocking_hover_started,
            # Authorization is retained from the typed immutable event. Its
            # arbitrary payload is never consulted by review control policy.
            trigger_event=event,
        )
        self._publish_status_event(
            MissionEventType.MODEL_REVIEW_STARTED,
            ticket,
            now,
            {"blocking": ticket.blocking, "frame_id": frame_ref.frame_id},
        )
        return self.snapshot()

    def abort_pending(
        self,
        *,
        reason_code: str = "MISSION_SHUTDOWN",
        timestamp_s: float | None = None,
    ) -> VisualReviewCoordinatorSnapshot:
        """Release an in-flight review when trusted mission control supersedes it."""

        if reason_code not in {"MISSION_SHUTDOWN", "SAFETY_SUPERSEDED"}:
            raise ValueError("unsupported visual-review abort reason")
        pending = self._pending
        if pending is None:
            return self.snapshot()
        selected_time = (
            self._last_frame_timestamp_s
            if timestamp_s is None
            else _nonnegative_finite(timestamp_s, "timestamp_s")
        )
        now = max(
            pending.ticket.submitted_timestamp_s,
            0.0 if selected_time is None else selected_time,
        )
        self._finish_pending_with_error(
            pending,
            now=now,
            error_code=reason_code,
            event_type=MissionEventType.MODEL_REVIEW_COMPLETED,
        )
        return self.snapshot()

    def snapshot(self) -> VisualReviewCoordinatorSnapshot:
        pending = self._pending
        hover_active = bool(
            self._skill_manager is not None
            and self._skill_manager.is_supervisory_paused
        )
        return VisualReviewCoordinatorSnapshot(
            uav_id=self._uav_id,
            latest_frame_ref=self._latest_frame_ref,
            inflight_request_id=(None if pending is None else pending.ticket.request_id),
            inflight_review_id=(None if pending is None else pending.ticket.review_id),
            inflight_blocking=(False if pending is None else pending.ticket.blocking),
            supervisory_hover_active=hover_active,
            consecutive_track_mismatches=self._consecutive_track_mismatches,
            review_count=len(self._records),
            stale_count=self._stale_count,
            last_record=(None if not self._records else self._records[-1]),
            last_candidate_id=self._last_candidate_id,
            pending_revision=self._pending_revision,
        )

    def reset(self) -> None:
        # The background HTTP call cannot be killed safely. Release the
        # scheduler reservation and forget its expectation; any eventual
        # response is drained as an orphan-stale result in a later mission.
        if self._pending is not None:
            pending = self._pending
            completion_time = max(
                pending.ticket.submitted_timestamp_s,
                self._last_frame_timestamp_s or 0.0,
            )
            self._scheduler.mark_timed_out(
                uav_id=self._uav_id,
                request_id=pending.ticket.request_id,
                review_id=pending.ticket.review_id,
                timestamp_s=completion_time,
            )
            self._pending = None
        self._scheduler.reset_uav(uav_id=self._uav_id)
        self._frame_store.clear(uav_id=self._uav_id)
        self._pending_events.clear()
        self._records.clear()
        self._latest_frame_ref = None
        self._last_frame_timestamp_s = None
        self._route = None
        self._gate_target_id = None
        self._consecutive_track_mismatches = 0
        self._stale_count = 0
        self._last_candidate_id = None
        self._revision_events.clear()
        self._pending_revision = None
        self._gate.reset()

    def _build_input(
        self,
        *,
        ticket: ReviewTicket,
        latest: FrameRef,
        target_spec: TargetSpec,
        target_snapshot: TargetSnapshot,
        active_step_id: str,
        skill_feedback: Mapping[str, object] | None,
        mission_elapsed_s: float,
        trigger_event_type: MissionEventType | None,
    ) -> VisualReviewInput:
        refs = [
            ref
            for ref in self._frame_store.refs(uav_id=self._uav_id)
            if ref.timestamp_s <= latest.timestamp_s
        ][-self._max_recent_frames :]
        frames: list[VisualReviewFrame] = []
        for ref in refs:
            rgb = self._frame_store.get_frame(ref)
            if rgb is not None:
                frames.append(VisualReviewFrame(ref, rgb))
        if not frames or frames[-1].ref != latest:
            raise VisualReviewCoordinatorError("latest frame was evicted before submission")
        environment_context: dict[str, object] = {
            "mission_elapsed_s": mission_elapsed_s,
        }
        if trigger_event_type is not None:
            if not isinstance(trigger_event_type, MissionEventType):
                raise TypeError("trigger_event_type must be a MissionEventType")
            # Only this closed enum value is projected; arbitrary event
            # payload content is deliberately excluded from the model input.
            environment_context["trigger_event_type"] = trigger_event_type.value
        return VisualReviewInput(
            review_id=ticket.review_id,
            mission_id=ticket.mission_id,
            uav_id=ticket.uav_id,
            plan_version=ticket.plan_version,
            observation_timestamp_s=latest.timestamp_s,
            frame_id=latest.frame_id,
            target_spec=target_spec,
            current_skill=ticket.skill_name,
            current_step_id=active_step_id,
            frames=tuple(frames),
            target_snapshot=target_snapshot,
            skill_feedback_summary=_feedback_summary(skill_feedback),
            environment_context=environment_context,
        )

    def _poll_result(
        self,
        *,
        now: float,
        mission_id: str,
        plan_version: int,
        current_target_id: str | None,
        current_step_id: str | None,
    ) -> bool:
        pending = self._pending
        if pending is None:
            return False
        try:
            result = self._worker.poll(
                expected_request_id=pending.ticket.request_id,
                expected_review_id=pending.ticket.review_id,
                minimum_observation_timestamp_s=pending.expectation.observation_timestamp_s,
                include_stale=True,
            )
        except Exception:
            # Worker transport failures are data-plane failures, not a reason
            # to strand the scheduler reservation or its pinned image.
            self._finish_pending_with_error(
                pending,
                now=now,
                error_code=VisualReviewParseErrorCode.MODEL_REQUEST_FAILED.value,
                event_type=MissionEventType.MODEL_REVIEW_COMPLETED,
            )
            return True
        if result is None:
            return False

        orphan_reasons: list[str] = []
        if result.stale:
            orphan_reasons.append(
                VisualReviewStaleReason.WORKER_MARKED_STALE.value
            )
        if result.request_id != pending.ticket.request_id:
            orphan_reasons.append(
                VisualReviewStaleReason.REQUEST_ID_MISMATCH.value
            )
        if result.review_id != pending.ticket.review_id:
            orphan_reasons.append(
                VisualReviewStaleReason.REVIEW_ID_MISMATCH.value
            )
        if orphan_reasons:
            # A timed-out/aborted HTTP call can finish after a new request has
            # acquired coordinator ownership.  Record that old result as an
            # orphan, but never let it complete or unpin the new request.
            self._record_orphan_result(
                result,
                now=now,
                stale_reasons=tuple(dict.fromkeys(orphan_reasons)),
            )
            return False

        stale_reasons = _collect_stale_reasons(
            result=result,
            pending=pending,
            mission_id=mission_id,
            plan_version=plan_version,
            current_step_id=current_step_id,
            current_target_id=current_target_id,
            now=now,
            max_result_age_s=self._max_result_age_s,
            frame_present=self._frame_store.contains(pending.frame_ref),
        )
        review: QwenVisualReview | None = None
        acceptance: VisualReviewAcceptance | None = None
        candidate_id: str | None = None
        candidate_suppressed = False
        error_code: str | None = None
        if stale_reasons:
            error_code = "STALE"
        elif result.error_code is not None:
            error_code = VisualReviewParseErrorCode.MODEL_REQUEST_FAILED.value
        else:
            try:
                review = self._verifier.parse_async_result(
                    result,
                    expectation=pending.expectation,
                )
                if pending.target_id != self._gate_target_id:
                    self._gate.reset()
                    self._consecutive_track_mismatches = 0
                    self._gate_target_id = pending.target_id
                candidate_id, candidate_suppressed = self._associate_candidate(
                    review=review,
                )
                acceptance = self._gate.evaluate(
                    review,
                    pending.expectation,
                    consensus_key=self._review_consensus_key(
                        pending=pending,
                        candidate_id=candidate_id,
                        candidate_suppressed=candidate_suppressed,
                    ),
                )
                if acceptance.disposition is ReviewDisposition.STALE:
                    error_code = "STALE"
                    # parse_async_result already enforces this route. Retain a
                    # specific fail-closed reason if a custom gate violates
                    # that invariant rather than emitting an unexplained bool.
                    stale_reasons = (
                        VisualReviewStaleReason.WORKER_MARKED_STALE.value,
                    )
            except VisualReviewProtocolError:
                error_code = VisualReviewParseErrorCode.ROUTING_MISMATCH.value
            except Exception as exc:
                error_code = _classify_model_or_parse_error(result, exc)

        stale = bool(stale_reasons)
        if stale:
            self._stale_count += 1
        self._scheduler.mark_completed(
            uav_id=self._uav_id,
            request_id=pending.ticket.request_id,
            review_id=pending.ticket.review_id,
            timestamp_s=now,
        )
        # An explicit store clear is itself a legitimate staleness signal;
        # it also removes the pin, so release only while the exact frame is
        # still present.
        if self._frame_store.contains(pending.frame_ref):
            self._frame_store.unpin(pending.frame_ref)
        self._pending = None
        candidate_id = self._record_candidate(
            review=review,
            pending=pending,
            completed_timestamp_s=now,
            candidate_id=candidate_id,
            discard=error_code is not None,
        )
        self._report_search_candidate_pending(
            pending=pending,
            candidate_id=candidate_id,
        )
        threshold_crossed = self._update_track_semantics(
            review,
            pending.ticket.skill_name,
            stale,
            acceptance,
        )
        self._append_record(
            pending=pending,
            result=result,
            review=review,
            acceptance=acceptance,
            now=now,
            stale=stale,
            error_code=error_code,
            stale_reasons=stale_reasons,
            candidate_id=candidate_id,
        )
        self._publish_status_event(
            MissionEventType.MODEL_RESPONSE_STALE
            if stale
            else MissionEventType.MODEL_REVIEW_COMPLETED,
            pending.ticket,
            now,
            {
                "stale": stale,
                "stale_reasons": list(stale_reasons),
                "error_code": error_code,
            },
        )
        revision_action: str | None = None
        authorization_source: str | None = None
        trigger = pending.trigger_event
        trusted_path_blocked = bool(
            trigger is not None
            and trigger.event_type is MissionEventType.PATH_BLOCKED
        )
        review_is_valid = bool(
            self._gate.mode is VisualReviewMode.GATE
            and review is not None
            and acceptance is not None
            and error_code is None
            and result.error_code is None
            and not stale
        )
        if review_is_valid and trusted_path_blocked:
            # PATH_BLOCKED is an allowlisted typed runtime authorization. The
            # visual result only has to satisfy routing/protocol validity; its
            # action, candidate, and the original event payload grant nothing.
            revision_action = VisualReviewAction.REQUEST_REPLAN.value
            authorization_source = "trusted_runtime_event"
        elif (
            review_is_valid
            and acceptance is not None
            and acceptance.accepted_for_control
            and not candidate_suppressed
            and self._is_valid_revision_candidate(candidate_id)
        ):
            # Model semantics may request revision only after repeated
            # evidence has reached consensus for this exact CandidateBank ID.
            revision_action = _revision_action(review, stale=False)
            if revision_action is not None:
                authorization_source = "visual_consensus"
        revision_event: MissionEvent | None = None
        automatic_revision_hover = False
        if (
            revision_action is not None
            and authorization_source == "visual_consensus"
            and not pending.blocking_hover_started
        ):
            automatic_revision_hover = self._interrupt_search_for_revision(
                pending=pending,
            )
        if (
            revision_action is not None
            and review is not None
            and authorization_source is not None
        ):
            # This event is the only bridge to a separate asynchronous
            # Revision Planner. The visual-review JSON never carries or
            # directly applies an unvalidated flight plan.
            revision_event = self._publish_revision_event(
                pending.ticket,
                now,
                review=review,
                candidate_id=candidate_id,
                action=revision_action,
                authorization_source=authorization_source,
                trigger_event=trigger,
            )
        if threshold_crossed and self._gate.mode is VisualReviewMode.GATE:
            self._queue_identity_conflict_event(pending.ticket, now)
        # Qwen never manipulates the controller. A trusted continuation merely
        # releases HOVER and restarts the exact saved Goal on the next manager
        # tick. Plan replacement is a separate validated two-stage operation.
        hover_started = pending.blocking_hover_started or automatic_revision_hover
        if hover_started and self._skill_manager is not None:
            if revision_event is not None and (
                self._await_revision_completion or automatic_revision_hover
            ):
                self._pending_revision = PendingPlanRevision(
                    event=revision_event,
                    request_id=pending.ticket.request_id,
                    review_id=pending.ticket.review_id,
                    candidate_id=candidate_id,
                )
            else:
                # No revision owner was configured: resume the exact saved
                # Goal immediately so a model recommendation cannot starve the
                # mission in HOVER.
                self._skill_manager.resume_interrupted_step()
        return False

    def _handle_timeout(self, now: float) -> bool:
        pending = self._pending
        if pending is None or now - pending.ticket.submitted_timestamp_s < self._review_timeout_s:
            return False
        self._finish_pending_with_error(
            pending,
            now=now,
            error_code="TIMEOUT",
            event_type=MissionEventType.MODEL_REVIEW_TIMEOUT,
        )
        return True

    def _finish_pending_with_error(
        self,
        pending: _PendingReview,
        *,
        now: float,
        error_code: str,
        event_type: MissionEventType,
    ) -> None:
        """Atomically release scheduler/frame ownership and record the failure."""

        if self._pending is not pending:
            raise VisualReviewCoordinatorError("pending review ownership changed")
        self._scheduler.mark_timed_out(
            uav_id=self._uav_id,
            request_id=pending.ticket.request_id,
            review_id=pending.ticket.review_id,
            timestamp_s=now,
        )
        if self._frame_store.contains(pending.frame_ref):
            self._frame_store.unpin(pending.frame_ref)
        self._pending = None
        self._records.append(
            VisualReviewRecord(
                request_id=pending.ticket.request_id,
                review_id=pending.ticket.review_id,
                mission_id=pending.ticket.mission_id,
                uav_id=pending.ticket.uav_id,
                plan_version=pending.ticket.plan_version,
                observation_timestamp_s=pending.expectation.observation_timestamp_s,
                frame_id=pending.expectation.frame_id,
                completed_timestamp_s=now,
                blocking=pending.ticket.blocking,
                stale=False,
                stale_reasons=(),
                decision=None,
                disposition=None,
                accepted_for_control=False,
                bbox_xyxy_normalized=None,
                token_usage={},
                latency_s=now - pending.ticket.submitted_timestamp_s,
                error_code=error_code,
            )
        )
        self._publish_status_event(
            event_type,
            pending.ticket,
            now,
            {
                "blocking": pending.ticket.blocking,
                "error_code": error_code,
            },
        )
        # For blocking reviews, the Manager's trusted HOVER goal owns timeout
        # fallback. Its next tick chooses RESUME_PREVIOUS or CANCEL_AND_LAND.
        # Non-blocking reviews simply leave the current Skill untouched.

    def _update_track_semantics(
        self,
        review: QwenVisualReview | None,
        reviewed_skill_name: str,
        stale: bool,
        acceptance: VisualReviewAcceptance | None,
    ) -> bool:
        if stale or review is None or reviewed_skill_name != SkillName.TRACK.value:
            return False
        if review.decision is VisualReviewDecision.TARGET_MISMATCH:
            self._consecutive_track_mismatches += 1
            # Even reaching the threshold emits policy evidence only; it does
            # not drop/switch the target or fabricate identity/ReID evidence.
            return self._consecutive_track_mismatches == self._track_mismatch_threshold
        if review.decision is VisualReviewDecision.TARGET_MATCH:
            self._consecutive_track_mismatches = 0
            if (
                self._gate.mode is VisualReviewMode.GATE
                and acceptance is not None
                and acceptance.accepted_for_control
            ):
                self._append_controlled_appearance_note(review)
        return False

    def _record_candidate(
        self,
        *,
        review: QwenVisualReview | None,
        pending: _PendingReview,
        completed_timestamp_s: float,
        candidate_id: str | None,
        discard: bool,
    ) -> str | None:
        bank = self._candidate_bank
        if (
            bank is None
            or discard
            or review is None
            or candidate_id is None
            or not review.candidate.present
            or review.candidate.bbox_xyxy_normalized is None
        ):
            return None
        candidate = bank.propose(
            candidate_id=candidate_id,
            timestamp_s=review.observation_timestamp_s,
            bbox_xyxy_normalized=review.candidate.bbox_xyxy_normalized,
            frame_ref=pending.frame_ref,
            source="qwen_vl",
        )
        if candidate is None:
            return None
        bank.add_review(
            candidate_id,
            CandidateReviewRef(
                review_id=review.review_id,
                timestamp_s=completed_timestamp_s,
                decision=review.decision.value,
            ),
        )
        self._last_candidate_id = candidate_id
        return candidate_id

    def _associate_candidate(
        self,
        *,
        review: QwenVisualReview,
    ) -> tuple[str | None, bool]:
        """Resolve a Qwen box before gate evaluation without adding evidence."""

        bank = self._candidate_bank
        bbox = review.candidate.bbox_xyxy_normalized
        if bank is None or not review.candidate.present or bbox is None:
            return None, False
        candidate_id = bank.associate_proposal(
            timestamp_s=review.observation_timestamp_s,
            bbox_xyxy_normalized=bbox,
            proposed_candidate_id=generate_routing_id("candidate"),
            min_iou=self._candidate_association_iou_threshold,
            source="qwen_vl",
        )
        # A None association is a trusted negative-memory suppression, not an
        # invitation to fall back to route-level semantic consensus.
        return candidate_id, candidate_id is None

    @staticmethod
    def _review_consensus_key(
        *,
        pending: _PendingReview,
        candidate_id: str | None,
        candidate_suppressed: bool,
    ) -> str | None:
        if candidate_suppressed:
            return None
        if pending.ticket.skill_name == SkillName.TRACK.value:
            # Preserve legacy target-scoped TRACK consensus when no
            # CandidateBank is configured. When a candidate is available,
            # binding the key to it also prevents cross-box revision evidence.
            return candidate_id or pending.target_id
        return candidate_id

    def _is_valid_revision_candidate(self, candidate_id: str | None) -> bool:
        bank = self._candidate_bank
        if bank is None or candidate_id is None:
            return False
        candidate = bank.get(candidate_id)
        return bool(
            candidate is not None
            and candidate.lifecycle
            not in {CandidateLifecycle.REJECTED, CandidateLifecycle.STALE}
        )

    def _interrupt_search_for_revision(
        self,
        *,
        pending: _PendingReview,
    ) -> bool:
        """Start trusted HOVER after accepted ordinary SEARCH consensus."""

        manager = self._skill_manager
        if (
            not self._await_revision_completion
            or manager is None
            or pending.ticket.skill_name != SkillName.SEARCH.value
            or manager.task_status is not TaskStatus.RUNNING
            or manager.active_name is not SkillName.SEARCH
            or manager.active_execution_kind is not ExecutionKind.PLANNED
            or manager.active_status is not SkillStatus.RUNNING
            or manager.active_planned_step_id != pending.step_id
        ):
            return False
        plan = manager.task_plan
        if (
            plan is None
            or plan.mission_id != pending.ticket.mission_id
            or plan.uav_id != pending.ticket.uav_id
            or plan.plan_version != pending.ticket.plan_version
        ):
            return False
        manager.interrupt_with_hover(
            "review_candidate_consensus",
            max_wait_s=self._review_timeout_s,
            position_tolerance_m=self._hover_position_tolerance_m,
            max_correction_speed_mps=self._hover_max_correction_speed_mps,
            timeout_fallback=self._blocking_timeout_fallback,
            defer_observation_timestamp_s=self._last_frame_timestamp_s,
        )
        return True

    def _report_search_candidate_pending(
        self,
        *,
        pending: _PendingReview,
        candidate_id: str | None,
    ) -> None:
        """Expose Qwen's provisional evidence to a still-running SEARCH.

        This remains a main-thread, gate-mode state annotation.  It never
        confirms identity or starts INSPECT/HOVER, and it is intentionally
        skipped after a blocking review has already interrupted SEARCH.
        """

        manager = self._skill_manager
        if (
            candidate_id is None
            or self._gate.mode is not VisualReviewMode.GATE
            or pending.blocking_hover_started
            or pending.ticket.skill_name != SkillName.SEARCH.value
            or manager is None
            or manager.task_status is not TaskStatus.RUNNING
            or manager.active_name is not SkillName.SEARCH
            or manager.active_execution_kind is not ExecutionKind.PLANNED
            or manager.active_status is not SkillStatus.RUNNING
        ):
            return
        manager.report_candidate_pending(candidate_id, source="qwen_vl")

    def _queue_identity_conflict_event(
        self,
        ticket: ReviewTicket,
        timestamp_s: float,
    ) -> None:
        event = MissionEvent(
            event_id=generate_routing_id("event"),
            mission_id=ticket.mission_id,
            uav_id=ticket.uav_id,
            plan_version=ticket.plan_version,
            timestamp_s=timestamp_s,
            event_type=MissionEventType.TARGET_IDENTITY_UNCERTAIN,
            severity=EventSeverity.WARNING,
            payload={
                "source": "qwen_vl",
                "request_id": ticket.request_id,
                "consecutive_mismatches": self._consecutive_track_mismatches,
            },
        )
        self._pending_events.append(event)
        if self._event_bus is not None:
            self._event_bus.publish(event)

    def _append_controlled_appearance_note(self, review: QwenVisualReview) -> None:
        if self._target_manager is None or not review.candidate.present:
            return
        description = review.candidate.description
        spec = self._target_manager.target_spec
        if description is None or spec is None:
            return
        try:
            if (
                description not in spec.mutable_appearance_notes
                and len(spec.mutable_appearance_notes) >= 32
            ):
                return
            updated = spec.append_appearance_note(description)
            self._target_manager.update_mutable_appearance_notes(
                updated.mutable_appearance_notes
            )
        except (TargetStateError, TypeError, ValueError):
            # A review completing after target termination cannot rewrite any
            # semantic state; routing acceptance remains separately logged.
            return

    def _append_record(
        self,
        *,
        pending: _PendingReview,
        result: AsyncModelResult,
        review: QwenVisualReview | None,
        acceptance: VisualReviewAcceptance | None,
        now: float,
        stale: bool,
        error_code: str | None,
        stale_reasons: tuple[str, ...],
        candidate_id: str | None,
    ) -> None:
        response = result.response
        self._records.append(
            VisualReviewRecord(
                request_id=pending.ticket.request_id,
                review_id=pending.ticket.review_id,
                mission_id=pending.ticket.mission_id,
                uav_id=pending.ticket.uav_id,
                plan_version=pending.ticket.plan_version,
                observation_timestamp_s=pending.expectation.observation_timestamp_s,
                frame_id=pending.expectation.frame_id,
                completed_timestamp_s=now,
                blocking=pending.ticket.blocking,
                stale=stale,
                stale_reasons=stale_reasons,
                decision=None if review is None else review.decision.value,
                disposition=(
                    None if acceptance is None else acceptance.disposition.value
                ),
                accepted_for_control=(
                    False if acceptance is None else acceptance.accepted_for_control
                ),
                bbox_xyxy_normalized=(
                    None if review is None else review.candidate.bbox_xyxy_normalized
                ),
                token_usage={} if response is None else dict(response.usage),
                latency_s=now - pending.ticket.submitted_timestamp_s,
                error_code=error_code or result.error_code,
                candidate_id=candidate_id,
                **_response_debug_fields(
                    response.content if response is not None else None,
                    enabled=self._debug_model_responses,
                ),
            )
        )

    def _next_event_for_route(
        self,
        mission_id: str,
        plan_version: int,
        now: float,
    ) -> MissionEvent | None:
        while self._pending_events:
            event = self._pending_events.popleft()
            if (
                event.mission_id == mission_id
                and event.uav_id == self._uav_id
                and event.plan_version == plan_version
                and event.timestamp_s <= now
            ):
                return event
            self._stale_count += 1
        return None

    def _drain_orphan_result(self, now: float) -> None:
        if self._pending is not None:
            return
        result = self._worker.poll(include_stale=True)
        if result is None:
            return
        self._record_orphan_result(
            result,
            now=now,
            stale_reasons=(
                VisualReviewStaleReason.WORKER_MARKED_STALE.value
                if result.stale
                else VisualReviewStaleReason.REQUEST_ID_MISMATCH.value,
            ),
        )

    def _record_orphan_result(
        self,
        result: AsyncModelResult,
        *,
        now: float,
        stale_reasons: tuple[str, ...],
    ) -> None:
        """Log a result that has no ownership claim on the current request."""

        self._stale_count += 1
        # Results from timed-out/superseded requests are deliberately not
        # parsed: no trusted expectation remains against which to accept them.
        self._records.append(
            VisualReviewRecord(
                request_id=result.request_id,
                review_id=result.review_id,
                mission_id=result.mission_id,
                uav_id=result.uav_id,
                plan_version=result.plan_version,
                observation_timestamp_s=result.observation_timestamp_s,
                frame_id=result.frame_id,
                completed_timestamp_s=now,
                blocking=False,
                stale=True,
                stale_reasons=stale_reasons,
                decision=None,
                disposition=ReviewDisposition.STALE.value,
                accepted_for_control=False,
                bbox_xyxy_normalized=None,
                token_usage=(
                    {} if result.response is None else dict(result.response.usage)
                ),
                latency_s=max(0.0, now - result.observation_timestamp_s),
                error_code="ORPHAN_STALE",
                **_response_debug_fields(
                    (
                        None
                        if result.response is None
                        else result.response.content
                    ),
                    enabled=self._debug_model_responses,
                ),
            )
        )

    def _publish_status_event(
        self,
        event_type: MissionEventType,
        ticket: ReviewTicket,
        timestamp_s: float,
        payload: Mapping[str, object],
    ) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish(
            MissionEvent(
                event_id=generate_routing_id("event"),
                mission_id=ticket.mission_id,
                uav_id=ticket.uav_id,
                plan_version=ticket.plan_version,
                timestamp_s=timestamp_s,
                event_type=event_type,
                severity=(
                    EventSeverity.WARNING
                    if event_type
                    in {
                        MissionEventType.MODEL_REVIEW_TIMEOUT,
                        MissionEventType.MODEL_RESPONSE_STALE,
                    }
                    else EventSeverity.INFO
                ),
                payload=dict(payload),
            )
        )

    def _publish_revision_event(
        self,
        ticket: ReviewTicket,
        timestamp_s: float,
        *,
        review: QwenVisualReview,
        candidate_id: str | None,
        action: str,
        authorization_source: str,
        trigger_event: MissionEvent | None,
    ) -> MissionEvent:
        trusted_runtime_event = authorization_source == "trusted_runtime_event"
        trigger_event_type = (
            None if trigger_event is None else trigger_event.event_type.value
        )
        trigger_event_id = (
            None if trigger_event is None else trigger_event.event_id
        )
        event = MissionEvent(
            event_id=generate_routing_id("event"),
            mission_id=ticket.mission_id,
            uav_id=ticket.uav_id,
            plan_version=ticket.plan_version,
            timestamp_s=timestamp_s,
            event_type=MissionEventType.PLAN_REVISION_REQUESTED,
            severity=EventSeverity.INFO,
            payload={
                "request_id": ticket.request_id,
                "review_id": ticket.review_id,
                "candidate_id": candidate_id,
                "action": action,
                "decision": review.decision.value,
                "reason_codes": list(review.reason_codes),
                "source": (
                    "trusted_runtime_event"
                    if trusted_runtime_event
                    else "qwen_vl"
                ),
                "authorization_source": authorization_source,
                "trusted_runtime_event": trusted_runtime_event,
                "trigger_event_type": trigger_event_type,
                "trigger_event_id": trigger_event_id,
            },
        )
        self._revision_events.append(event)
        if self._event_bus is not None:
            self._event_bus.publish(event)
        return event


def _feedback_summary(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
    result: dict[str, object] = {}
    for key in ("progress", "message", "phase", "elapsed_time", "waypoint_index", "waypoint_count"):
        item = value.get(key)
        if item is None:
            data = value.get("data")
            if isinstance(data, Mapping):
                item = data.get(key)
        if item is None or isinstance(item, (str, bool, int)):
            if item is not None:
                result[key] = item
        elif isinstance(item, float) and isfinite(item):
            result[key] = item
    return result


def _collect_stale_reasons(
    *,
    result: AsyncModelResult,
    pending: _PendingReview,
    mission_id: str,
    plan_version: int,
    current_step_id: str | None,
    current_target_id: str | None,
    now: float,
    max_result_age_s: float,
    frame_present: bool,
) -> tuple[str, ...]:
    """Evaluate every staleness invariant independently in stable order."""

    reasons: list[str] = []

    def add(reason: VisualReviewStaleReason, condition: bool) -> None:
        if condition:
            reasons.append(reason.value)

    add(VisualReviewStaleReason.WORKER_MARKED_STALE, result.stale)
    add(
        VisualReviewStaleReason.MISSION_ID_CHANGED,
        mission_id != pending.ticket.mission_id
        or result.mission_id != pending.ticket.mission_id,
    )
    add(
        VisualReviewStaleReason.PLAN_VERSION_CHANGED,
        plan_version != pending.ticket.plan_version
        or result.plan_version != pending.ticket.plan_version,
    )
    add(
        VisualReviewStaleReason.STEP_ID_CHANGED,
        current_step_id != pending.step_id,
    )
    add(
        VisualReviewStaleReason.TARGET_ID_CHANGED,
        current_target_id != pending.target_id,
    )
    add(
        VisualReviewStaleReason.RESULT_TOO_OLD,
        now - pending.expectation.observation_timestamp_s > max_result_age_s,
    )
    add(VisualReviewStaleReason.FRAME_EVICTED, not frame_present)
    add(
        VisualReviewStaleReason.REQUEST_ID_MISMATCH,
        result.request_id != pending.ticket.request_id,
    )
    add(
        VisualReviewStaleReason.REVIEW_ID_MISMATCH,
        result.review_id != pending.ticket.review_id,
    )
    return tuple(reasons)


class _DuplicateJSONField(ValueError):
    pass


def _diagnostic_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONField(key)
        result[key] = value
    return result


def _reject_diagnostic_constant(value: str) -> object:
    raise ValueError(f"unsupported JSON constant: {value}")


def _classify_model_or_parse_error(
    result: AsyncModelResult,
    error: Exception,
) -> str:
    """Classify without retaining exception text or the full model response."""

    response = result.response
    if response is None:
        return VisualReviewParseErrorCode.MODEL_REQUEST_FAILED.value
    try:
        parsed = json.loads(
            response.content,
            parse_constant=_reject_diagnostic_constant,
            object_pairs_hook=_diagnostic_json_object,
        )
    except _DuplicateJSONField:
        return VisualReviewParseErrorCode.DUPLICATE_FIELD.value
    except (json.JSONDecodeError, TypeError, ValueError):
        return VisualReviewParseErrorCode.INVALID_JSON.value
    if isinstance(parsed, Mapping):
        decision = parsed.get("decision")
        action = parsed.get("recommended_action")
        if (
            isinstance(decision, str)
            and decision not in {item.value for item in VisualReviewDecision}
        ) or (
            isinstance(action, str)
            and action not in {item.value for item in VisualReviewAction}
        ):
            return VisualReviewParseErrorCode.UNSUPPORTED_ENUM.value
    if isinstance(error, ModelProtocolError):
        return VisualReviewParseErrorCode.SCHEMA_INVALID.value
    return VisualReviewParseErrorCode.UNKNOWN_PARSE_ERROR.value


_IMAGE_DATA_PATTERN = re.compile(
    r"data:image/[^;,\s\"']+;base64,[^\s\"']+",
    flags=re.IGNORECASE,
)
_CREDENTIAL_LABEL_PATTERN = re.compile(
    r"api[\s_-]*key|authorization",
    flags=re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r'''(["']?(?:api[\s_-]*key|authorization)["']?\s*[:=]\s*)'''
    r'''(?:"[^"]*"|'[^']*'|[^,\s}]+)''',
    flags=re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(r"bearer\s+[^\s\"']+", flags=re.IGNORECASE)


def _response_debug_fields(
    content: str | None,
    *,
    enabled: bool,
) -> dict[str, object]:
    if not enabled or content is None:
        return {
            "response_text_length": None,
            "response_text_tail": None,
        }
    # Redact before slicing: otherwise a long data URL whose prefix lies
    # outside the tail could leak an unlabelled base64 suffix.
    redacted = _IMAGE_DATA_PATTERN.sub("[REDACTED_IMAGE_DATA]", content)
    redacted = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(
        '"credential":"[REDACTED_CREDENTIAL]"',
        redacted,
    )
    redacted = _BEARER_PATTERN.sub("[REDACTED_CREDENTIAL]", redacted)
    redacted = _CREDENTIAL_LABEL_PATTERN.sub("credential", redacted)
    return {
        "response_text_length": len(content),
        "response_text_tail": redacted[-500:],
    }


def _debug_visual_review_from_environment() -> bool:
    raw = os.environ.get("UAV_AGENT_DEBUG_VISUAL_REVIEW")
    if raw is None:
        return False
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError(
        "UAV_AGENT_DEBUG_VISUAL_REVIEW must be one of 0/1/false/true/no/yes/off/on"
    )


def _prompt_safe_target_snapshot(snapshot: TargetSnapshot) -> TargetSnapshot:
    """Remove an Oracle-derived target state before constructing a prompt."""

    source = snapshot.source
    if source is None or "oracle" not in source.casefold():
        return snapshot
    return TargetSnapshot(
        target_id=None,
        description=snapshot.description,
        lifecycle=TargetLifecycle.UNINITIALIZED,
        confidence=None,
        last_seen_position=None,
        last_seen_velocity=None,
        last_seen_time_s=None,
        source=None,
    )


def _revision_action(
    review: QwenVisualReview | None,
    *,
    stale: bool,
) -> str | None:
    if review is None or stale:
        return None
    if review.recommended_action is VisualReviewAction.REQUEST_REPLAN:
        return VisualReviewAction.REQUEST_REPLAN.value
    if review.recommended_action is VisualReviewAction.INSPECT:
        return VisualReviewAction.INSPECT.value
    if review.decision in {
        VisualReviewDecision.POSSIBLE_TARGET,
        VisualReviewDecision.AMBIGUOUS,
    }:
        # Trusted orchestration maps uncertain semantics to a request for the
        # revision planner to consider INSPECT. It does not manufacture an
        # INSPECT Goal, candidate geometry, or flight plan itself.
        return VisualReviewAction.INSPECT.value
    return None


def _hover_reason(event: MissionEvent | None) -> str:
    if event is None:
        return "visual_review_blocking"
    text = "review_" + event.event_type.value.lower()
    return text[:64]


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _nonnegative_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return normalized


def _positive_finite(value: object, field_name: str) -> float:
    normalized = _nonnegative_finite(value, field_name)
    if normalized == 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return normalized


def _unit_interval(value: object, field_name: str) -> float:
    normalized = _nonnegative_finite(value, field_name)
    if normalized > 1.0:
        raise ValueError(f"{field_name} must be within [0, 1]")
    return normalized


__all__ = [
    "PendingPlanRevision",
    "RevisionCompletionAction",
    "VisualReviewCoordinator",
    "VisualReviewCoordinatorError",
    "VisualReviewCoordinatorSnapshot",
    "VisualReviewRecord",
]
