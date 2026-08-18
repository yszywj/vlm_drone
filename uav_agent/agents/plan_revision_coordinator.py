"""Non-blocking orchestration for trusted, model-proposed plan revisions.

The visual-review path may request a revision, but it is never allowed to
carry or apply one.  This coordinator is the second-stage boundary: it builds
one text-only request through :class:`QwenPlanRevisionPlanner`, submits it to
an asynchronous worker, validates the returned suffix in full, and only then
asks ``SkillManager`` to atomically replace the interrupted step/suffix.

No method in this module calls a synchronous model client.  ``tick`` performs
bounded in-memory work and at most one non-blocking worker poll.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Protocol

from common.ids import generate_routing_id, validate_routing_id, validate_uav_id
from models import AsyncModelRequest, AsyncModelResult
from perception.candidate_bank import CandidateBank, CandidateLifecycle, CandidateSnapshot
from planner.revision import (
    PlanRevisionRequest,
    QwenPlanRevisionPlanner,
    RevisionValidationError,
    RevisionValidator,
    ValidatedPlanRevision,
)
from planner.schemas import PlannerWorldContext, SkillPlanDraftV2
from runtime.events import MissionEvent, MissionEventType
from common.provenance import is_privileged_oracle_source
from runtime.world_belief import WorldBelief
from skills.hover import HoverTimeoutFallback
from skills.plan import TaskPlan


class _AsyncRevisionWorker(Protocol):
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


class _RevisionPlanner(Protocol):
    def build_async_request(
        self,
        revision_request: PlanRevisionRequest,
        *,
        request_id: str,
        review_id: str,
    ) -> AsyncModelRequest: ...

    def parse_async_result(
        self,
        result: AsyncModelResult,
        *,
        revision_request: PlanRevisionRequest,
        expected_request_id: str,
        expected_review_id: str,
    ) -> object: ...


class _RevisionValidator(Protocol):
    @property
    def revision_limits(self) -> object: ...

    def validate_and_apply(self, revision: object, **kwargs: object) -> ValidatedPlanRevision: ...


class _SupervisorySkillManager(Protocol):
    uav_id: str

    @property
    def task_plan(self) -> TaskPlan | None: ...

    @property
    def task_status(self) -> object: ...

    @property
    def pending_task_result(self) -> object | None: ...

    @property
    def active_planned_step_id(self) -> str | None: ...

    @property
    def active_name(self) -> object | None: ...

    @property
    def is_supervisory_paused(self) -> bool: ...

    @property
    def step_outputs(self) -> Mapping[str, Mapping[str, object]]: ...

    def interrupt_with_hover(self, reason_code: str, **kwargs: object) -> object: ...

    def resume_interrupted_step(self) -> object: ...

    def replace_interrupted_step_and_suffix(self, plan: TaskPlan) -> object: ...

    def handoff_interrupted_search_candidate_to_inspect(
        self,
        plan: TaskPlan,
        *,
        candidate_id: str,
        source: str,
    ) -> object: ...

    def cancel_task(self) -> object: ...


class PlanRevisionCoordinatorError(RuntimeError):
    """Raised for invalid coordinator construction or direct API misuse."""


class PlanRevisionState(str, Enum):
    IDLE = "IDLE"
    IN_FLIGHT = "IN_FLIGHT"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"


class PlanRevisionFallback(str, Enum):
    RESUME_INTERRUPTED = "RESUME_INTERRUPTED"
    CANCEL_AND_LAND = "CANCEL_AND_LAND"


@dataclass(frozen=True, slots=True)
class PlanRevisionRecord:
    """Sparse audit record.  It deliberately contains no prompt or image."""

    mission_id: str
    uav_id: str
    request_id: str
    review_id: str
    old_plan_version: int
    new_plan_version: int | None
    old_step_id: str
    new_step_id: str | None
    timestamp_s: float
    outcome: str
    reason_codes: tuple[str, ...]
    fallback: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "request_id": self.request_id,
            "review_id": self.review_id,
            "old_plan_version": self.old_plan_version,
            "new_plan_version": self.new_plan_version,
            "old_step_id": self.old_step_id,
            "new_step_id": self.new_step_id,
            "timestamp_s": self.timestamp_s,
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "fallback": self.fallback,
        }


@dataclass(frozen=True, slots=True)
class PlanRevisionCoordinatorSnapshot:
    state: PlanRevisionState
    uav_id: str
    request_id: str | None
    review_id: str | None
    mission_id: str | None
    base_plan_version: int | None
    expected_plan_version: int | None
    current_step_id: str | None
    submitted_timestamp_s: float | None
    deadline_timestamp_s: float | None
    revision_count: int
    last_revision_timestamp_s: float | None
    last_error_code: str | None
    last_error: str | None
    fallback_applied: str | None
    last_record: PlanRevisionRecord | None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "uav_id": self.uav_id,
            "request_id": self.request_id,
            "review_id": self.review_id,
            "mission_id": self.mission_id,
            "base_plan_version": self.base_plan_version,
            "expected_plan_version": self.expected_plan_version,
            "current_step_id": self.current_step_id,
            "submitted_timestamp_s": self.submitted_timestamp_s,
            "deadline_timestamp_s": self.deadline_timestamp_s,
            "revision_count": self.revision_count,
            "last_revision_timestamp_s": self.last_revision_timestamp_s,
            "last_error_code": self.last_error_code,
            "last_error": self.last_error,
            "fallback_applied": self.fallback_applied,
            "last_record": (
                None if self.last_record is None else self.last_record.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class _PendingRevision:
    event: MissionEvent
    request: PlanRevisionRequest
    request_id: str
    review_id: str
    submitted_timestamp_s: float
    deadline_timestamp_s: float


class PlanRevisionCoordinator:
    """Coordinate one routed asynchronous revision request at a time."""

    def __init__(
        self,
        *,
        uav_id: str,
        planner: QwenPlanRevisionPlanner | _RevisionPlanner,
        worker: _AsyncRevisionWorker,
        validator: RevisionValidator | _RevisionValidator,
        skill_manager: _SupervisorySkillManager,
        candidate_bank: CandidateBank | None = None,
        world_context: PlannerWorldContext,
        safety_preflight: object,
        original_instruction: str,
        clock: Callable[[], float],
        request_timeout_s: float = 20.0,
        fallback: PlanRevisionFallback | str = (
            PlanRevisionFallback.RESUME_INTERRUPTED
        ),
        hover_position_tolerance_m: float = 0.25,
        hover_max_correction_speed_mps: float = 0.5,
        max_records: int = 32,
        logger: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        self._uav_id = validate_uav_id(uav_id)
        if not callable(getattr(planner, "build_async_request", None)) or not callable(
            getattr(planner, "parse_async_result", None)
        ):
            raise TypeError("planner must provide build_async_request and parse_async_result")
        if not callable(getattr(worker, "submit", None)) or not callable(
            getattr(worker, "poll", None)
        ):
            raise TypeError("worker must provide non-blocking submit and poll methods")
        if validate_uav_id(getattr(worker, "uav_id", None)) != self._uav_id:
            raise ValueError("worker uav_id does not match coordinator")
        if not callable(getattr(validator, "validate_and_apply", None)):
            raise TypeError("validator must provide validate_and_apply")
        limits = getattr(validator, "revision_limits", None)
        if limits is None or not all(
            hasattr(limits, field)
            for field in ("max_plan_revisions", "cooldown_s")
        ):
            raise TypeError("validator must expose trusted revision_limits")
        if validate_uav_id(getattr(skill_manager, "uav_id", None)) != self._uav_id:
            raise ValueError("SkillManager uav_id does not match coordinator")
        for method in (
            "interrupt_with_hover",
            "resume_interrupted_step",
            "replace_interrupted_step_and_suffix",
            "cancel_task",
        ):
            if not callable(getattr(skill_manager, method, None)):
                raise TypeError(f"skill_manager must provide {method}()")
        if candidate_bank is not None:
            if not isinstance(candidate_bank, CandidateBank):
                raise TypeError("candidate_bank must be a CandidateBank or None")
            if candidate_bank.uav_id != self._uav_id:
                raise ValueError("CandidateBank uav_id does not match coordinator")
            if not callable(
                getattr(
                    skill_manager,
                    "handoff_interrupted_search_candidate_to_inspect",
                    None,
                )
            ):
                raise TypeError(
                    "skill_manager must provide "
                    "handoff_interrupted_search_candidate_to_inspect() when a "
                    "CandidateBank is configured"
                )
        if not isinstance(world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")
        if not callable(safety_preflight) and not callable(
            getattr(safety_preflight, "preflight", None)
        ):
            raise TypeError(
                "safety_preflight must be callable or expose preflight()"
            )
        if not isinstance(original_instruction, str):
            raise TypeError("original_instruction must be a string")
        instruction = original_instruction.strip()
        if not instruction or len(instruction) > 4096:
            raise ValueError(
                "original_instruction must contain between 1 and 4096 characters"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        timeout = _positive_finite(request_timeout_s, "request_timeout_s")
        if timeout > 300.0:
            raise ValueError("request_timeout_s must not exceed 300 seconds")
        position_tolerance = _positive_finite(
            hover_position_tolerance_m,
            "hover_position_tolerance_m",
        )
        correction_speed = _positive_finite(
            hover_max_correction_speed_mps,
            "hover_max_correction_speed_mps",
        )
        try:
            selected_fallback = PlanRevisionFallback(fallback)
        except (TypeError, ValueError):
            raise ValueError(
                "fallback must be RESUME_INTERRUPTED or CANCEL_AND_LAND"
            ) from None
        if isinstance(max_records, bool) or not isinstance(max_records, int):
            raise TypeError("max_records must be an integer")
        if not 1 <= max_records <= 1024:
            raise ValueError("max_records must be between 1 and 1024")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable or None")

        self._planner = planner
        self._worker = worker
        self._validator = validator
        self._skill_manager = skill_manager
        self._candidate_bank = candidate_bank
        self._world_context = world_context
        self._safety_preflight = safety_preflight
        self._original_instruction = instruction
        self._clock = clock
        self._request_timeout_s = timeout
        self._fallback = selected_fallback
        self._hover_position_tolerance_m = position_tolerance
        self._hover_max_correction_speed_mps = correction_speed
        self._logger = logger
        self._records: deque[PlanRevisionRecord] = deque(maxlen=max_records)

        self._state = PlanRevisionState.IDLE
        self._pending: _PendingRevision | None = None
        self._revision_count = 0
        self._last_revision_timestamp_s: float | None = None
        self._last_error_code: str | None = None
        self._last_error: str | None = None
        self._fallback_applied: str | None = None
        self._last_route: tuple[str, int, str] | None = None
        self._latest_accepted: ValidatedPlanRevision | None = None

    @property
    def uav_id(self) -> str:
        return self._uav_id

    @property
    def records(self) -> tuple[PlanRevisionRecord, ...]:
        return tuple(self._records)

    @property
    def latest_accepted_revision(self) -> ValidatedPlanRevision | None:
        """Return the immutable accepted result for MissionAgent bookkeeping."""

        return self._latest_accepted

    @property
    def is_inflight(self) -> bool:
        return self._pending is not None and self._state is PlanRevisionState.IN_FLIGHT

    def validate_agent_bindings(
        self,
        skill_manager: _SupervisorySkillManager,
        safety_preflight: object | None = None,
    ) -> None:
        """Reject accidentally wiring one coordinator across UAV agents."""

        if skill_manager is not self._skill_manager:
            raise PlanRevisionCoordinatorError(
                "coordinator and MissionAgent must share the same SkillManager"
            )
        if (
            safety_preflight is not None
            and safety_preflight is not self._safety_preflight
        ):
            raise PlanRevisionCoordinatorError(
                "coordinator and MissionAgent must share the same SafetySupervisor"
            )

    def validate_mission_start(
        self,
        instruction: str,
        world_context: PlannerWorldContext,
    ) -> None:
        """Verify that the trusted second-stage context matches start()."""

        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        if instruction.strip() != self._original_instruction:
            raise PlanRevisionCoordinatorError(
                "revision coordinator original instruction does not match mission"
            )
        if not isinstance(world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")
        if world_context != self._world_context:
            raise PlanRevisionCoordinatorError(
                "revision coordinator WorldContext does not match mission"
            )

    def submit_event(
        self,
        event: MissionEvent,
        *,
        current_plan: SkillPlanDraftV2,
        world_belief: WorldBelief,
    ) -> PlanRevisionCoordinatorSnapshot:
        """Start one blocking asynchronous revision, never an HTTP call.

        Operational rejection is represented in the returned snapshot.  Type
        errors remain exceptions because they indicate an integration bug.
        """

        if not isinstance(event, MissionEvent):
            raise TypeError("event must be a MissionEvent")
        if not isinstance(current_plan, SkillPlanDraftV2):
            raise TypeError("current_plan must be a SkillPlanDraftV2")
        if not isinstance(world_belief, WorldBelief):
            raise TypeError("world_belief must be a WorldBelief")
        if self._pending is not None:
            # New evidence cannot supersede a request that already owns the
            # blocking HOVER.  Keep the authoritative in-flight route/state
            # unchanged; its result will be accepted, timed out, or rejected
            # through the normal path.
            return self.snapshot()

        route_error = self._submission_route_error(event, current_plan, world_belief)
        if route_error is not None:
            return self._reject_submission(
                event,
                current_plan,
                route_error[0],
                route_error[1],
                fallback=bool(self._skill_manager.is_supervisory_paused),
            )

        current_step_id = world_belief.current_step_id
        assert current_step_id is not None
        current_step = next(
            step for step in current_plan.steps if step.id == current_step_id
        )
        if current_step.skill == "LAND" or _enum_value(
            getattr(self._skill_manager, "active_name", None)
        ) == "LAND":
            return self._reject_submission(
                event,
                current_plan,
                "REVISION_DURING_LAND",
                "ordinary semantic revision is forbidden during LAND",
                fallback=False,
            )
        inspect_candidate_id, inspect_error = self._inspect_trigger_candidate(event)
        if inspect_error is not None:
            return self._reject_submission(
                event,
                current_plan,
                inspect_error[0],
                inspect_error[1],
                fallback=bool(self._skill_manager.is_supervisory_paused),
            )
        if inspect_candidate_id is not None:
            if current_step.skill != "SEARCH":
                return self._reject_submission(
                    event,
                    current_plan,
                    "INSPECT_TRIGGER_REQUIRES_SEARCH",
                    "INSPECT candidate handoff requires the active SEARCH step",
                    fallback=bool(self._skill_manager.is_supervisory_paused),
                )
            candidate_error = self._candidate_error(
                event,
                inspect_candidate_id,
            )
            if candidate_error is not None:
                return self._reject_submission(
                    event,
                    current_plan,
                    candidate_error[0],
                    candidate_error[1],
                    fallback=bool(self._skill_manager.is_supervisory_paused),
                )

        now = self._now()
        limits = self._validator.revision_limits
        if self._revision_count >= int(limits.max_plan_revisions):
            return self._reject_submission(
                event,
                current_plan,
                "REVISION_BUDGET_EXCEEDED",
                "maximum plan revision count has been reached",
                fallback=True,
            )
        if (
            self._last_revision_timestamp_s is not None
            and now - self._last_revision_timestamp_s < float(limits.cooldown_s)
        ):
            return self._reject_submission(
                event,
                current_plan,
                "REVISION_COOLDOWN",
                "plan revision cooldown has not elapsed",
                fallback=True,
            )

        plan_ids = tuple(step.id for step in current_plan.steps)
        current_index = plan_ids.index(current_step_id)
        completed = plan_ids[:current_index]
        safe_outputs = _safe_completed_outputs(
            getattr(self._skill_manager, "step_outputs", {}),
            completed,
        )
        try:
            revision_request = PlanRevisionRequest(
                original_instruction=self._original_instruction,
                original_plan=current_plan,
                current_step_id=current_step_id,
                completed_step_ids=completed,
                completed_step_outputs=safe_outputs,
                replaceable_step_ids=plan_ids[current_index:],
                world_belief=world_belief,
                trigger_event=event,
                trusted_inspect_candidate_id=inspect_candidate_id,
            )
            request_id = generate_routing_id("revision_request")
            review_id = generate_routing_id("revision_review")
            async_request = self._planner.build_async_request(
                revision_request,
                request_id=request_id,
                review_id=review_id,
            )
        except Exception as exc:
            return self._reject_submission(
                event,
                current_plan,
                "REQUEST_BUILD_FAILED",
                _safe_error(exc),
                fallback=bool(self._skill_manager.is_supervisory_paused),
            )

        pending = _PendingRevision(
            event=event,
            request=revision_request,
            request_id=request_id,
            review_id=review_id,
            submitted_timestamp_s=now,
            deadline_timestamp_s=now + self._request_timeout_s,
        )
        hover_was_active = bool(self._skill_manager.is_supervisory_paused)
        try:
            if not hover_was_active:
                self._skill_manager.interrupt_with_hover(
                    "plan_revision_requested",
                    max_wait_s=self._request_timeout_s,
                    position_tolerance_m=self._hover_position_tolerance_m,
                    max_correction_speed_mps=(
                        self._hover_max_correction_speed_mps
                    ),
                    timeout_fallback=(
                        HoverTimeoutFallback.RESUME_PREVIOUS
                        if self._fallback
                        is PlanRevisionFallback.RESUME_INTERRUPTED
                        else HoverTimeoutFallback.CANCEL_AND_LAND
                    ),
                )
            self._pending = pending
            self._worker.submit(async_request)
        except Exception as exc:
            self._pending = pending
            self._finish_rejected(
                state=PlanRevisionState.REJECTED,
                code="SUBMIT_OR_HOVER_FAILED",
                message=_safe_error(exc),
                now=now,
                apply_fallback=bool(self._skill_manager.is_supervisory_paused),
            )
            return self.snapshot()

        self._state = PlanRevisionState.IN_FLIGHT
        self._last_error_code = None
        self._last_error = None
        self._fallback_applied = None
        self._last_route = (
            current_plan.mission_id,
            current_plan.plan_version,
            current_step_id,
        )
        self._log(
            {
                "event": "plan_revision_submitted",
                "mission_id": current_plan.mission_id,
                "uav_id": self._uav_id,
                "plan_version": current_plan.plan_version,
                "step_id": current_step_id,
                "request_id": request_id,
                "review_id": review_id,
            }
        )
        return self.snapshot()

    def tick(
        self,
        *,
        current_plan: SkillPlanDraftV2,
        world_belief: WorldBelief,
    ) -> PlanRevisionCoordinatorSnapshot:
        """Poll once and apply only a fully validated, still-current result."""

        if not isinstance(current_plan, SkillPlanDraftV2):
            raise TypeError("current_plan must be a SkillPlanDraftV2")
        if not isinstance(world_belief, WorldBelief):
            raise TypeError("world_belief must be a WorldBelief")
        pending = self._pending
        if pending is None:
            return self.snapshot()
        now = self._now()

        if now < pending.submitted_timestamp_s:
            self._finish_rejected(
                state=PlanRevisionState.REJECTED,
                code="REVISION_CLOCK_REGRESSION",
                message="revision clock moved backwards while request was in flight",
                now=now,
                apply_fallback=True,
            )
            return self.snapshot()

        stale = self._pending_route_error(pending, current_plan, world_belief)
        if stale is not None:
            self._finish_rejected(
                state=PlanRevisionState.REJECTED,
                code="STALE_REVISION",
                message=stale,
                now=now,
                apply_fallback=bool(self._skill_manager.is_supervisory_paused),
            )
            return self.snapshot()
        if now >= pending.deadline_timestamp_s:
            self._finish_rejected(
                state=PlanRevisionState.TIMED_OUT,
                code="REVISION_TIMEOUT",
                message="plan revision request exceeded its trusted timeout",
                now=now,
                apply_fallback=True,
            )
            return self.snapshot()

        result = self._worker.poll(
            expected_request_id=pending.request_id,
            expected_review_id=pending.review_id,
            minimum_observation_timestamp_s=pending.request.anchor_timestamp_s,
            include_stale=True,
        )
        if result is None:
            return self.snapshot()

        try:
            revision = self._planner.parse_async_result(
                result,
                revision_request=pending.request,
                expected_request_id=pending.request_id,
                expected_review_id=pending.review_id,
            )
            validated = self._validator.validate_and_apply(
                revision,
                original=pending.request.original_plan,
                world_context=self._world_context,
                current_step_id=pending.request.current_step_id,
                completed_step_ids=pending.request.completed_step_ids,
                completed_step_outputs=pending.request.completed_step_outputs,
                revision_count=self._revision_count,
                now_s=now,
                last_revision_timestamp_s=self._last_revision_timestamp_s,
                expected_new_plan_version=(
                    pending.request.original_plan.plan_version + 1
                ),
                source="dynamic_llm",
                safety_preflight=self._safety_preflight,
                trusted_inspect_candidate_id=(
                    pending.request.trusted_inspect_candidate_id
                ),
            )
            # Re-check immediately before the only state-changing call.  This
            # keeps the original Manager plan intact if another trusted path
            # canceled, landed, or advanced the task while validation ran.
            stale_after_validation = self._pending_route_error(
                pending,
                current_plan,
                world_belief,
            )
            if stale_after_validation is not None:
                raise PlanRevisionCoordinatorError(stale_after_validation)
            inspect_candidate_id, inspect_error = self._inspect_trigger_candidate(
                pending.event
            )
            if inspect_error is not None:
                raise PlanRevisionCoordinatorError(inspect_error[1])
            if inspect_candidate_id is None:
                self._skill_manager.replace_interrupted_step_and_suffix(
                    validated.compiled_mission.task_plan
                )
            else:
                candidate_error = self._candidate_error(
                    pending.event,
                    inspect_candidate_id,
                )
                if candidate_error is not None:
                    raise PlanRevisionCoordinatorError(candidate_error[1])
                candidate = self._require_candidate(inspect_candidate_id)
                self._skill_manager.handoff_interrupted_search_candidate_to_inspect(
                    validated.compiled_mission.task_plan,
                    candidate_id=candidate.candidate_id,
                    source=candidate.source,
                )
        except RevisionValidationError as exc:
            self._finish_rejected(
                state=PlanRevisionState.REJECTED,
                code=exc.code.value,
                message=_safe_error(exc),
                now=now,
                apply_fallback=True,
            )
            return self.snapshot()
        except Exception as exc:
            self._finish_rejected(
                state=PlanRevisionState.REJECTED,
                code="MODEL_OR_REVISION_REJECTED",
                message=_safe_error(exc),
                now=now,
                apply_fallback=True,
            )
            return self.snapshot()

        old_plan = pending.request.original_plan
        old_index = tuple(step.id for step in old_plan.steps).index(
            pending.request.current_step_id
        )
        accepted_inspect_candidate_id, _inspect_error = (
            self._inspect_trigger_candidate(pending.event)
        )
        new_index = (
            old_index + 1
            if accepted_inspect_candidate_id is not None
            else old_index
        )
        new_step_id = validated.revised_plan.steps[new_index].id
        self._pending = None
        self._state = PlanRevisionState.ACCEPTED
        self._revision_count = validated.revision_count
        self._last_revision_timestamp_s = now
        self._last_error_code = None
        self._last_error = None
        self._fallback_applied = None
        self._latest_accepted = validated
        record = PlanRevisionRecord(
            mission_id=old_plan.mission_id,
            uav_id=old_plan.uav_id,
            request_id=pending.request_id,
            review_id=pending.review_id,
            old_plan_version=old_plan.plan_version,
            new_plan_version=validated.revised_plan.plan_version,
            old_step_id=pending.request.current_step_id,
            new_step_id=new_step_id,
            timestamp_s=now,
            outcome=PlanRevisionState.ACCEPTED.value,
            reason_codes=tuple(validated.revision.reason_codes),
            fallback=None,
        )
        self._records.append(record)
        self._last_route = (
            old_plan.mission_id,
            validated.revised_plan.plan_version,
            new_step_id,
        )
        self._log({"event": "plan_revision_accepted", **record.to_dict()})
        return self.snapshot()

    def snapshot(self) -> PlanRevisionCoordinatorSnapshot:
        pending = self._pending
        if pending is not None:
            plan = pending.request.original_plan
            mission_id = plan.mission_id
            base_version = plan.plan_version
            expected_version = plan.plan_version + 1
            current_step = pending.request.current_step_id
            request_id = pending.request_id
            review_id = pending.review_id
            submitted = pending.submitted_timestamp_s
            deadline = pending.deadline_timestamp_s
        elif self._last_route is not None:
            mission_id, route_version, current_step = self._last_route
            base_version = (
                route_version - 1
                if self._state is PlanRevisionState.ACCEPTED
                else route_version
            )
            expected_version = (
                route_version
                if self._state is PlanRevisionState.ACCEPTED
                else route_version + 1
            )
            last = None if not self._records else self._records[-1]
            request_id = None if last is None else last.request_id
            review_id = None if last is None else last.review_id
            submitted = None
            deadline = None
        else:
            mission_id = None
            base_version = None
            expected_version = None
            current_step = None
            request_id = None
            review_id = None
            submitted = None
            deadline = None
        return PlanRevisionCoordinatorSnapshot(
            state=self._state,
            uav_id=self._uav_id,
            request_id=request_id,
            review_id=review_id,
            mission_id=mission_id,
            base_plan_version=base_version,
            expected_plan_version=expected_version,
            current_step_id=current_step,
            submitted_timestamp_s=submitted,
            deadline_timestamp_s=deadline,
            revision_count=self._revision_count,
            last_revision_timestamp_s=self._last_revision_timestamp_s,
            last_error_code=self._last_error_code,
            last_error=self._last_error,
            fallback_applied=self._fallback_applied,
            last_record=(None if not self._records else self._records[-1]),
        )

    def reset(self) -> None:
        """Reset mission-scoped budgets and forget any orphanable response."""

        if self._pending is not None and self._skill_manager.is_supervisory_paused:
            self._apply_fallback()
        self._pending = None
        self._state = PlanRevisionState.IDLE
        self._revision_count = 0
        self._last_revision_timestamp_s = None
        self._last_error_code = None
        self._last_error = None
        self._fallback_applied = None
        self._last_route = None
        self._latest_accepted = None
        self._records.clear()

    def _inspect_trigger_candidate(
        self,
        event: MissionEvent,
    ) -> tuple[str | None, tuple[str, str] | None]:
        """Return the trusted INSPECT route or a fail-closed input error."""

        if event.payload.get("action") != "INSPECT":
            return None, None
        try:
            candidate_id = validate_routing_id(
                event.payload.get("candidate_id"),
                "candidate_id",
            )
        except (TypeError, ValueError):
            return None, (
                "INSPECT_CANDIDATE_INVALID",
                "INSPECT event requires a valid candidate_id",
            )
        if self._candidate_bank is None:
            return None, (
                "INSPECT_CANDIDATE_BANK_REQUIRED",
                "INSPECT candidate handoff requires a routed CandidateBank",
            )
        return candidate_id, None

    def _candidate_error(
        self,
        event: MissionEvent,
        candidate_id: str,
    ) -> tuple[str, str] | None:
        bank = self._candidate_bank
        if bank is None:
            return (
                "INSPECT_CANDIDATE_BANK_REQUIRED",
                "INSPECT candidate handoff requires a routed CandidateBank",
            )
        candidate = bank.get(candidate_id)
        if candidate is None:
            return (
                "INSPECT_CANDIDATE_UNKNOWN",
                "INSPECT event candidate_id is absent from CandidateBank",
            )
        if candidate.uav_id != self._uav_id:
            return (
                "INSPECT_CANDIDATE_ROUTING_MISMATCH",
                "INSPECT candidate belongs to another UAV",
            )
        if candidate.lifecycle is not CandidateLifecycle.PROVISIONAL:
            return (
                "INSPECT_CANDIDATE_NOT_PROVISIONAL",
                "INSPECT candidate must still be PROVISIONAL",
            )
        if is_privileged_oracle_source(candidate.source):
            return (
                "INSPECT_CANDIDATE_SOURCE_PRIVILEGED",
                "Oracle candidate provenance cannot enter a Qwen revision request",
            )
        if candidate.source != "qwen_vl":
            return (
                "INSPECT_CANDIDATE_SOURCE_INVALID",
                "INSPECT candidate source is not an allowed trusted boundary",
            )
        event_source = event.payload.get("source")
        if event_source is not None and event_source != candidate.source:
            return (
                "INSPECT_CANDIDATE_SOURCE_MISMATCH",
                "INSPECT event source does not match CandidateBank",
            )
        return None

    def _require_candidate(self, candidate_id: str) -> CandidateSnapshot:
        bank = self._candidate_bank
        if bank is None:
            raise PlanRevisionCoordinatorError(
                "INSPECT candidate handoff has no CandidateBank"
            )
        candidate = bank.get(candidate_id)
        if candidate is None:
            raise PlanRevisionCoordinatorError(
                "INSPECT candidate disappeared from CandidateBank"
            )
        return candidate

    def _submission_route_error(
        self,
        event: MissionEvent,
        plan: SkillPlanDraftV2,
        belief: WorldBelief,
    ) -> tuple[str, str] | None:
        if event.event_type is not MissionEventType.PLAN_REVISION_REQUESTED:
            return (
                "EVENT_TYPE_INVALID",
                "only PLAN_REVISION_REQUESTED events are accepted",
            )
        expected = (plan.mission_id, plan.uav_id, plan.plan_version)
        if event.uav_id != self._uav_id or belief.uav_id != self._uav_id:
            return ("ROUTING_MISMATCH", "revision request uav_id mismatch")
        if (
            (event.mission_id, event.uav_id, event.plan_version) != expected
            or (belief.mission_id, belief.uav_id, belief.plan_version) != expected
        ):
            return ("ROUTING_MISMATCH", "revision request route is stale")
        if belief.current_step_id is None or belief.current_skill is None:
            return ("CURRENT_STEP_INVALID", "WorldBelief has no active step")
        plan_ids = tuple(step.id for step in plan.steps)
        if belief.current_step_id not in plan_ids:
            return (
                "CURRENT_STEP_INVALID",
                "WorldBelief current step is absent from current_plan",
            )
        manager_plan = self._skill_manager.task_plan
        if manager_plan is None or (
            manager_plan.mission_id,
            manager_plan.uav_id,
            manager_plan.plan_version,
        ) != expected:
            return ("ROUTING_MISMATCH", "SkillManager plan route is stale")
        if self._skill_manager.active_planned_step_id != belief.current_step_id:
            return (
                "CURRENT_STEP_INVALID",
                "SkillManager active step does not match WorldBelief",
            )
        if _enum_value(self._skill_manager.task_status) != "RUNNING":
            return ("TASK_NOT_RUNNING", "SkillManager task is not RUNNING")
        if self._skill_manager.pending_task_result is not None:
            return (
                "TASK_TERMINATING",
                "task cancel/landing already takes priority over revision",
            )
        return None

    def _pending_route_error(
        self,
        pending: _PendingRevision,
        plan: SkillPlanDraftV2,
        belief: WorldBelief,
    ) -> str | None:
        expected = pending.request.original_plan
        if (
            plan.mission_id != expected.mission_id
            or plan.uav_id != expected.uav_id
            or plan.plan_version != expected.plan_version
        ):
            return "current semantic plan changed while revision was in flight"
        if (
            belief.mission_id != expected.mission_id
            or belief.uav_id != expected.uav_id
            or belief.plan_version != expected.plan_version
            or belief.current_step_id != pending.request.current_step_id
        ):
            return "WorldBelief route/current step changed while revision was in flight"
        manager_plan = self._skill_manager.task_plan
        if manager_plan is None or (
            manager_plan.mission_id != expected.mission_id
            or manager_plan.uav_id != expected.uav_id
            or manager_plan.plan_version != expected.plan_version
        ):
            return "SkillManager plan route changed while revision was in flight"
        if self._skill_manager.active_planned_step_id != pending.request.current_step_id:
            return "SkillManager current step changed while revision was in flight"
        if self._skill_manager.pending_task_result is not None:
            return "hard safety/cancel landing took priority over revision"
        if not self._skill_manager.is_supervisory_paused:
            return "supervisory HOVER ended before revision was accepted"
        return None

    def _reject_submission(
        self,
        event: MissionEvent,
        plan: SkillPlanDraftV2,
        code: str,
        message: str,
        *,
        fallback: bool,
    ) -> PlanRevisionCoordinatorSnapshot:
        now = self._now()
        self._state = PlanRevisionState.REJECTED
        self._last_error_code = code
        self._last_error = message
        self._fallback_applied = self._apply_fallback() if fallback else None
        current = getattr(self._skill_manager, "active_planned_step_id", None)
        if not isinstance(current, str):
            current = plan.steps[-1].id
        request_id = generate_routing_id("revision_rejected")
        review_id = generate_routing_id("revision_review")
        record = PlanRevisionRecord(
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            request_id=request_id,
            review_id=review_id,
            old_plan_version=plan.plan_version,
            new_plan_version=None,
            old_step_id=current,
            new_step_id=None,
            timestamp_s=now,
            outcome=PlanRevisionState.REJECTED.value,
            reason_codes=(code,),
            fallback=self._fallback_applied,
        )
        self._records.append(record)
        self._last_route = (plan.mission_id, plan.plan_version, current)
        self._log({"event": "plan_revision_rejected", **record.to_dict()})
        return self.snapshot()

    def _finish_rejected(
        self,
        *,
        state: PlanRevisionState,
        code: str,
        message: str,
        now: float,
        apply_fallback: bool,
    ) -> None:
        pending = self._pending
        if pending is None:
            raise PlanRevisionCoordinatorError("no pending revision to reject")
        plan = pending.request.original_plan
        fallback = self._apply_fallback() if apply_fallback else None
        self._pending = None
        self._state = state
        self._last_error_code = code
        self._last_error = message
        self._fallback_applied = fallback
        record = PlanRevisionRecord(
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            request_id=pending.request_id,
            review_id=pending.review_id,
            old_plan_version=plan.plan_version,
            new_plan_version=None,
            old_step_id=pending.request.current_step_id,
            new_step_id=None,
            timestamp_s=now,
            outcome=state.value,
            reason_codes=(code,),
            fallback=fallback,
        )
        self._records.append(record)
        self._last_route = (
            plan.mission_id,
            plan.plan_version,
            pending.request.current_step_id,
        )
        self._log({"event": "plan_revision_rejected", **record.to_dict()})

    def _apply_fallback(self) -> str | None:
        if self._fallback is PlanRevisionFallback.RESUME_INTERRUPTED:
            if not self._skill_manager.is_supervisory_paused:
                return None
            try:
                self._skill_manager.resume_interrupted_step()
                return self._fallback.value
            except Exception:
                # A failed resume must not strand a vehicle in an unowned
                # supervisory state.  Existing fail-safe landing remains the
                # trusted final fallback.
                try:
                    self._skill_manager.cancel_task()
                    return PlanRevisionFallback.CANCEL_AND_LAND.value
                except Exception:
                    return "FALLBACK_FAILED"
        try:
            if _enum_value(self._skill_manager.task_status) == "RUNNING":
                self._skill_manager.cancel_task()
                return self._fallback.value
        except Exception:
            return "FALLBACK_FAILED"
        return None

    def _now(self) -> float:
        return _nonnegative_finite(self._clock(), "clock timestamp")

    def _log(self, payload: Mapping[str, object]) -> None:
        if self._logger is None:
            return
        try:
            self._logger(dict(payload))
        except Exception:
            return


def _safe_completed_outputs(
    raw: object,
    completed_step_ids: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Project completed outputs onto a tiny prompt-safe scalar allow-list."""

    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, dict[str, object]] = {}
    allowed = frozenset({"target_id", "status", "result_code"})
    for step_id in completed_step_ids:
        value = raw.get(step_id)
        if not isinstance(value, Mapping):
            continue
        projected: dict[str, object] = {}
        for key in allowed:
            item = value.get(key)
            if item is None or isinstance(item, (str, bool, int)):
                if item is not None:
                    projected[key] = item
            elif isinstance(item, float) and isfinite(item):
                projected[key] = item
        if projected:
            result[step_id] = projected
    return result


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return raw if isinstance(raw, str) else str(raw)


def _safe_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if not text:
        return type(exc).__name__
    # Never retain model responses or giant transport errors in normal state.
    return text[:512]


def _positive_finite(value: object, field_name: str) -> float:
    number = _nonnegative_finite(value, field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return number


def _nonnegative_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


__all__ = [
    "PlanRevisionCoordinator",
    "PlanRevisionCoordinatorError",
    "PlanRevisionCoordinatorSnapshot",
    "PlanRevisionFallback",
    "PlanRevisionRecord",
    "PlanRevisionState",
]
