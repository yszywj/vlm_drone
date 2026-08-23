"""Non-blocking obstacle-route proposal, critique, repair, and publication."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from numbers import Real
from typing import Protocol

from common.ids import generate_routing_id, validate_uav_id
from models import AsyncModelResult, ModelProtocolError
from planner.obstacle_revision import (
    ObstacleAwareRevisionPlanner,
    ObstacleAwareRevisionRequest,
    ObstacleParseRepairFeedback,
    ObstacleRevisionError,
    ObstacleRevisionSession,
    ObstacleRevisionSessionState,
    ObstacleRouteRevisionDraft,
    is_repairable_obstacle_parse_error,
)
from planner.route_critic import (
    RouteCritic,
    RouteCritique,
    RouteValidationContext,
    RouteValidationMode,
)
from planner.route_types import RouteContractError
from planner.spatial_resolver import FramePose
from runtime.collision_supervisor import CollisionSupervisor
from runtime.events import MissionEvent, json_payload_to_dict, validated_json_payload
from runtime.route_registry import RouteRegistry
from runtime.safety_supervisor import SafetyAction, SafetyDecision
from skills.plan import TaskPlan


class _Worker(Protocol):
    uav_id: str

    def submit(self, request: object) -> None: ...

    def poll(self, **kwargs: object) -> AsyncModelResult | None: ...


class ObstacleRevisionCoordinatorState(str, Enum):
    IDLE = "IDLE"
    AWAITING_MODEL = "AWAITING_MODEL"
    ACCEPTED = "ACCEPTED"
    EXHAUSTED = "EXHAUSTED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ObstacleRevisionCoordinatorRecord:
    request_id: str
    proposal_index: int
    submitted_timestamp_s: float
    completed_timestamp_s: float | None
    outcome: str
    proposal: dict[str, object] | None
    critique: dict[str, object] | None
    error_code: str | None = None
    round_index: int = 0
    frame_snapshot: FramePose | None = None
    shadow_strict_critique: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ObstacleRevisionCoordinatorSnapshot:
    state: ObstacleRevisionCoordinatorState
    request_id: str | None
    proposal_count: int
    accepted_route_id: str | None
    error_code: str | None


class ObstacleRevisionCoordinator:
    """Keep Qwen off the control loop while preserving every route proposal."""

    def __init__(
        self,
        *,
        uav_id: str,
        planner: ObstacleAwareRevisionPlanner,
        worker: _Worker,
        route_registry: RouteRegistry,
        collision_supervisor: CollisionSupervisor,
        skill_manager: object,
        safety_preflight: Callable[[TaskPlan], SafetyDecision],
        route_validation_mode: RouteValidationMode | str = RouteValidationMode.STRICT,
        max_proposals: int = 3,
        event_sink: Callable[[MissionEvent], object] | None = None,
    ) -> None:
        self._uav_id = validate_uav_id(uav_id)
        if not isinstance(planner, ObstacleAwareRevisionPlanner):
            raise TypeError("planner must be ObstacleAwareRevisionPlanner")
        if getattr(worker, "uav_id", None) != self._uav_id:
            raise ValueError("worker.uav_id must match coordinator uav_id")
        if not callable(getattr(worker, "submit", None)) or not callable(
            getattr(worker, "poll", None)
        ):
            raise TypeError("worker must provide submit and poll")
        if not isinstance(route_registry, RouteRegistry):
            raise TypeError("route_registry must be RouteRegistry")
        if not isinstance(collision_supervisor, CollisionSupervisor):
            raise TypeError("collision_supervisor must be CollisionSupervisor")
        for method in (
            "replace_interrupted_step_and_suffix",
            "cancel_task",
        ):
            if not callable(getattr(skill_manager, method, None)):
                raise TypeError(f"skill_manager must provide {method}")
        if not callable(safety_preflight):
            raise TypeError("safety_preflight must be callable")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        try:
            mode = RouteValidationMode(route_validation_mode)
        except (TypeError, ValueError):
            raise ValueError(
                "route_validation_mode must be open_sim, critic_sim, or strict"
            ) from None
        self._planner = planner
        self._worker = worker
        self._registry = route_registry
        self._supervisor = collision_supervisor
        self._manager = skill_manager
        self._safety_preflight = safety_preflight
        self._mode = mode
        self._max_proposals = max_proposals
        # Validate the proposal budget eagerly using the authoritative session.
        ObstacleRevisionSession(mode=mode, max_proposals=max_proposals)
        self._event_sink = event_sink
        self._state = ObstacleRevisionCoordinatorState.IDLE
        self._request: ObstacleAwareRevisionRequest | None = None
        self._validation_context: RouteValidationContext | None = None
        self._frame_snapshot: FramePose | None = None
        self._compile_replacement: Callable[[ObstacleRouteRevisionDraft], TaskPlan] | None = None
        self._session: ObstacleRevisionSession | None = None
        self._request_id: str | None = None
        self._submitted_timestamp_s: float | None = None
        self._prior_proposal: ObstacleRouteRevisionDraft | None = None
        self._prior_critique: RouteCritique | None = None
        self._parse_repair: ObstacleParseRepairFeedback | None = None
        self._records: list[ObstacleRevisionCoordinatorRecord] = []
        self._archived_histories: list[dict[str, object]] = []
        self._round_index = 0
        self._next_proposal_index = 0
        self._round_proposal_count = 0
        self._accepted_route_id: str | None = None
        self._error_code: str | None = None

    @property
    def records(self) -> tuple[ObstacleRevisionCoordinatorRecord, ...]:
        return tuple(self._records)

    @property
    def history_dict(self) -> dict[str, object]:
        current = self._current_round_history()
        rounds = deepcopy(self._archived_histories)
        if current is not None:
            rounds.append(current)

        # Preserve the legacy, single-round top-level view while exposing every
        # retained round under an unambiguous namespace.
        if self._session is None:
            history: dict[str, object] = {"state": self._state.value}
        else:
            history = self._session.history_dict()
        history["round_index"] = self._round_index
        history["rounds"] = rounds
        return history

    def begin(
        self,
        request: ObstacleAwareRevisionRequest,
        *,
        validation_context: RouteValidationContext,
        frame_snapshot: FramePose,
        compile_replacement: Callable[[ObstacleRouteRevisionDraft], TaskPlan],
        timestamp_s: float,
    ) -> ObstacleRevisionCoordinatorSnapshot:
        if self._state is not ObstacleRevisionCoordinatorState.IDLE:
            raise RuntimeError("obstacle revision coordinator is already active")
        if not isinstance(request, ObstacleAwareRevisionRequest):
            raise TypeError("request must be ObstacleAwareRevisionRequest")
        if request.uav_id != self._uav_id:
            raise ValueError("request.uav_id does not match coordinator")
        if not isinstance(validation_context, RouteValidationContext):
            raise TypeError("validation_context must be RouteValidationContext")
        if not isinstance(frame_snapshot, FramePose):
            raise TypeError("frame_snapshot must be FramePose")
        if not callable(compile_replacement):
            raise TypeError("compile_replacement must be callable")
        if not bool(getattr(self._manager, "is_supervisory_paused", False)):
            raise RuntimeError("route revision requires an established supervisory pause")
        self._request = request
        self._validation_context = validation_context
        self._frame_snapshot = frame_snapshot
        self._compile_replacement = compile_replacement
        self._session = ObstacleRevisionSession(
            mode=self._mode,
            max_proposals=self._max_proposals,
        )
        self._round_proposal_count = 0
        self._publish_events(self._supervisor.begin_replanning().events)
        self._submit(timestamp_s=timestamp_s)
        return self.snapshot()

    def tick(self, *, timestamp_s: float) -> ObstacleRevisionCoordinatorSnapshot:
        if self._state is not ObstacleRevisionCoordinatorState.AWAITING_MODEL:
            return self.snapshot()
        assert self._request_id is not None
        result = self._worker.poll(
            expected_request_id=self._request_id,
            include_stale=True,
        )
        if result is None:
            return self.snapshot()
        completed = _timestamp(timestamp_s)
        try:
            proposal = self._planner.parse_async_result(
                result,
                request=self._require_request(),
            )
        except Exception as exc:
            parse_error_code = _normalized_parse_error_code(exc, result=result)
            failed_proposal = _failed_response_audit(
                result,
                parse_error_code=parse_error_code,
            )
            repairable = is_repairable_obstacle_parse_error(parse_error_code)
            has_budget = self._round_proposal_count + 1 < self._max_proposals
            self._append_record(
                completed_timestamp_s=completed,
                outcome=(
                    "REVISE_STRUCTURE"
                    if repairable and has_budget
                    else (
                        "EXHAUSTED_STRUCTURE"
                        if repairable
                        else "MODEL_OR_SCHEMA_ERROR"
                    )
                ),
                proposal=failed_proposal,
                critique=None,
                error_code=parse_error_code,
            )
            if repairable and has_budget:
                self._prior_proposal = None
                self._prior_critique = None
                self._parse_repair = _parse_repair_feedback(
                    failed_proposal,
                    parse_error_code=parse_error_code,
                    repair_attempt_index=self._round_proposal_count,
                    repeated_unchanged=_latest_structural_output_repeated(
                        self._records,
                        round_index=self._round_index,
                    ),
                )
                self._submit(timestamp_s=completed)
            elif repairable:
                self._state = ObstacleRevisionCoordinatorState.EXHAUSTED
                self._error_code = "ROUTE_REPAIR_BUDGET_EXHAUSTED"
                self._trusted_fallback_if_required()
            else:
                self._fail_or_fallback(parse_error_code)
            return self.snapshot()

        self._parse_repair = None
        session = self._require_session()
        validation_context = self._require_validation_context()
        critique = session.evaluate(proposal, validation_context)
        # Benchmark-only measurement: never controls acceptance, never edits
        # the proposal, and remains STRICT even when execution uses open_sim.
        shadow_strict_critique = RouteCritic(RouteValidationMode.STRICT).evaluate(
            proposal.route_draft,
            validation_context,
        )
        self._publish_events(
            self._supervisor.route_proposed(
                proposal.route_draft.route_id,
                timestamp_s=completed,
            ).events
        )
        accepted = session.state is ObstacleRevisionSessionState.ACCEPTED
        self._append_record(
            completed_timestamp_s=completed,
            outcome="ACCEPTED" if accepted else "REVISE",
            proposal=proposal.to_dict(),
            critique=critique.to_dict(),
            shadow_strict_critique=shadow_strict_critique.to_dict(),
        )
        if accepted:
            self._publish_accepted(
                proposal,
                critique,
                timestamp_s=completed,
            )
        elif (
            session.state is ObstacleRevisionSessionState.EXHAUSTED
            or self._round_proposal_count >= self._max_proposals
        ):
            self._publish_events(
                self._supervisor.route_rejected(
                    reason_codes=_critique_reason_codes(critique),
                    timestamp_s=completed,
                ).events
            )
            self._state = ObstacleRevisionCoordinatorState.EXHAUSTED
            self._error_code = "ROUTE_REPAIR_BUDGET_EXHAUSTED"
            self._trusted_fallback_if_required()
        else:
            self._publish_events(
                self._supervisor.route_rejected(
                    reason_codes=_critique_reason_codes(critique),
                    timestamp_s=completed,
                ).events
            )
            self._prior_proposal = proposal
            self._prior_critique = critique
            self._parse_repair = None
            self._submit(timestamp_s=completed)
        return self.snapshot()

    def snapshot(self) -> ObstacleRevisionCoordinatorSnapshot:
        return ObstacleRevisionCoordinatorSnapshot(
            self._state,
            self._request_id,
            self._round_proposal_count,
            self._accepted_route_id,
            self._error_code,
        )

    def reset(self, *, preserve_records: bool = False) -> None:
        """Return to IDLE, optionally retaining all completed revision rounds.

        ``preserve_records=True`` is intended for another obstacle event in the
        same mission.  It archives the current session history and keeps global
        proposal indices monotonic.  The default retains the historical reset
        semantics used at an episode boundary.
        """

        if self._state is ObstacleRevisionCoordinatorState.AWAITING_MODEL:
            raise RuntimeError("cannot reset while a model request is active")
        if not isinstance(preserve_records, bool):
            raise TypeError("preserve_records must be a bool")
        if preserve_records:
            current = self._current_round_history()
            if current is not None:
                self._archived_histories.append(current)
                self._round_index += 1
        else:
            self._records.clear()
            self._archived_histories.clear()
            self._round_index = 0
            self._next_proposal_index = 0
        self._state = ObstacleRevisionCoordinatorState.IDLE
        self._request = None
        self._validation_context = None
        self._frame_snapshot = None
        self._compile_replacement = None
        self._session = None
        self._request_id = None
        self._submitted_timestamp_s = None
        self._prior_proposal = None
        self._prior_critique = None
        self._parse_repair = None
        self._round_proposal_count = 0
        self._accepted_route_id = None
        self._error_code = None

    def _current_round_history(self) -> dict[str, object] | None:
        if self._session is None:
            return None
        history = self._session.history_dict()
        return {
            "round_index": self._round_index,
            "coordinator_state": self._state.value,
            **history,
        }

    def _append_record(
        self,
        *,
        completed_timestamp_s: float,
        outcome: str,
        proposal: dict[str, object] | None,
        critique: dict[str, object] | None,
        error_code: str | None = None,
        shadow_strict_critique: dict[str, object] | None = None,
    ) -> None:
        if self._request_id is None:
            raise RuntimeError("revision request id is unavailable")
        record = ObstacleRevisionCoordinatorRecord(
            request_id=self._request_id,
            proposal_index=self._next_proposal_index,
            submitted_timestamp_s=self._require_submitted_time(),
            completed_timestamp_s=completed_timestamp_s,
            outcome=outcome,
            proposal=proposal,
            critique=critique,
            error_code=error_code,
            round_index=self._round_index,
            frame_snapshot=self._frame_snapshot,
            shadow_strict_critique=shadow_strict_critique,
        )
        self._records.append(record)
        self._next_proposal_index += 1
        self._round_proposal_count += 1

    def _submit(self, *, timestamp_s: float) -> None:
        request_id = generate_routing_id("request_route")
        request = self._planner.build_async_request(
            self._require_request(),
            request_id=request_id,
            prior_proposal=self._prior_proposal,
            critique=self._prior_critique,
            parse_repair=self._parse_repair,
        )
        self._worker.submit(request)
        self._request_id = request_id
        self._submitted_timestamp_s = _timestamp(timestamp_s)
        self._state = ObstacleRevisionCoordinatorState.AWAITING_MODEL

    def _publish_accepted(
        self,
        proposal: ObstacleRouteRevisionDraft,
        critique: RouteCritique,
        *,
        timestamp_s: float,
    ) -> None:
        request = self._require_request()
        frame_snapshot = self._require_frame_snapshot()
        registered_route_id: str | None = None
        try:
            plan = self._require_compiler()(proposal)
            if not isinstance(plan, TaskPlan):
                raise TypeError("compile_replacement must return TaskPlan")
            if (
                plan.mission_id != proposal.mission_id
                or plan.uav_id != proposal.uav_id
                or plan.plan_version != proposal.new_plan_version
            ):
                raise ValueError("compiled replacement route/version mismatch")
            safety = self._safety_preflight(plan)
            if not isinstance(safety, SafetyDecision) or safety.action is not SafetyAction.CONTINUE:
                raise ValueError("compiled obstacle replacement failed safety preflight")
            record = self._registry.register(
                proposal.route_draft,
                frame_snapshot=frame_snapshot,
                raw_proposal=proposal.to_dict(),
                plan_version=proposal.new_plan_version,
                proposal_timestamp_s=request.frames[-1].ref.timestamp_s,
            )
            self._registry.record_critique(record.route_id, critique)
            registered_route_id = record.route_id
            self._manager.replace_interrupted_step_and_suffix(plan)
            self._publish_events(
                self._supervisor.route_accepted(
                    validation_mode=self._mode.value,
                    required_checks_passed=True,
                    timestamp_s=timestamp_s,
                ).events
            )
            self._supervisor.resume(required_checks_passed=True)
        except Exception:
            if registered_route_id is not None:
                try:
                    self._registry.transition(registered_route_id, "REJECTED")
                except Exception:
                    pass
            self._state = ObstacleRevisionCoordinatorState.FAILED
            self._error_code = "ACCEPTED_ROUTE_PUBLICATION_FAILED"
            self._trusted_fallback_if_required()
            return
        self._accepted_route_id = proposal.route_draft.route_id
        self._state = ObstacleRevisionCoordinatorState.ACCEPTED

    def _fail_or_fallback(self, error_code: str) -> None:
        self._state = ObstacleRevisionCoordinatorState.FAILED
        self._error_code = error_code
        self._trusted_fallback_if_required()

    def _trusted_fallback_if_required(self) -> None:
        # No experiment mode may remain in an indefinite HOVER after a
        # terminal schema/repair/publication failure.  The trusted fallback is
        # always cancel-and-land; only STRICT promises that unsafe geometry is
        # pre-rejected rather than measured at runtime.
        self._manager.cancel_task()

    def _publish_events(self, events: tuple[MissionEvent, ...]) -> None:
        if self._event_sink is None:
            return
        for event in events:
            self._event_sink(event)

    def _require_request(self) -> ObstacleAwareRevisionRequest:
        if self._request is None:
            raise RuntimeError("revision request is unavailable")
        return self._request

    def _require_session(self) -> ObstacleRevisionSession:
        if self._session is None:
            raise RuntimeError("revision session is unavailable")
        return self._session

    def _require_validation_context(self) -> RouteValidationContext:
        if self._validation_context is None:
            raise RuntimeError("route validation context is unavailable")
        return self._validation_context

    def _require_frame_snapshot(self) -> FramePose:
        if self._frame_snapshot is None:
            raise RuntimeError("route frame snapshot is unavailable")
        return self._frame_snapshot

    def _require_compiler(self) -> Callable[[ObstacleRouteRevisionDraft], TaskPlan]:
        if self._compile_replacement is None:
            raise RuntimeError("replacement compiler is unavailable")
        return self._compile_replacement

    def _require_submitted_time(self) -> float:
        if self._submitted_timestamp_s is None:
            raise RuntimeError("request submission timestamp is unavailable")
        return self._submitted_timestamp_s


def _critique_reason_codes(critique: RouteCritique) -> tuple[str, ...]:
    reasons = tuple(dict.fromkeys(item.type.value for item in critique.violations))
    return reasons or ("ROUTE_REJECTED",)


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timestamp_s must be finite and non-negative")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError("timestamp_s must be finite and non-negative")
    return normalized


_AUDIT_TAIL_CHARS = 500
_AUDIT_JSON_INPUT_CHARS = 16_384
_SENSITIVE_TEXT_MARKERS = (
    "data:image/",
    "base64,",
    "authorization",
    "api_key",
    "api-key",
    "apikey",
    "access_token",
    "access-token",
    "bearer ",
    "api key",
    "credential",
    "private_key",
    "password",
    "request_id",
    "request-id",
    "secret",
    "sk-",
)
_SENSITIVE_JSON_FIELDS = {
    "image",
    "images",
    "image_url",
    "rgb_image",
    "pixels",
    "camera_pixels",
    "frame_data",
    "frames",
    "messages",
    "request",
    "request_id",
    "rgb",
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "token",
    "credential",
    "credentials",
    "private_key",
    "password",
    "secret",
    "secret_key",
}


def _normalized_parse_error_code(
    exc: Exception,
    *,
    result: AsyncModelResult,
) -> str:
    """Map untrusted parse failures to stable codes without logging details."""

    if result.stale:
        return "ROUTE_RESPONSE_STALE"
    if result.response is None:
        source_code = result.error_code
        if (
            isinstance(source_code, str)
            and 1 <= len(source_code) <= 64
            and all(
                character.isupper()
                or character.isdigit()
                or character == "_"
                for character in source_code
            )
        ):
            return source_code
        return "ROUTE_MODEL_REQUEST_ERROR"
    message = str(exc).casefold()
    if isinstance(exc, RouteContractError):
        if "adjacent route waypoints must be distinct" in message:
            return "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE"
        if "waypoint ids must be unique" in message:
            return "ROUTE_WAYPOINT_IDS_DUPLICATE"
        return "ROUTE_CONTRACT_ERROR"
    if isinstance(exc, ObstacleRevisionError):
        if "async route result routing/version mismatch" in message:
            return "ROUTE_ASYNC_RESULT_ROUTING_MISMATCH"
        if "replacement must begin with follow_route" in message:
            return "ROUTE_FIRST_STEP_NOT_FOLLOW_ROUTE"
        if "replacement step ids must be unique" in message:
            return "ROUTE_REPLACEMENT_STEP_IDS_DUPLICATE"
        if "schema_version must equal" in message:
            return "ROUTE_SCHEMA_VERSION_INVALID"
        if "route draft exceeds trusted waypoint count" in message:
            return "ROUTE_WAYPOINT_BUDGET_EXCEEDED"
        if "replacement terminal suffix" in message:
            return "ROUTE_TERMINAL_SUFFIX_INVALID"
        if (
            "route draft does not echo trusted" in message
            or "new_plan_version must equal" in message
        ):
            return "ROUTE_TRUSTED_METADATA_MISMATCH"
        return "ROUTE_SCHEMA_VALUE_ERROR"
    if isinstance(exc, ModelProtocolError):
        if "invalid obstacle revision json" in message:
            return "ROUTE_JSON_INVALID"
        return "ROUTE_MODEL_PROTOCOL_ERROR"
    if isinstance(exc, TypeError):
        return "ROUTE_SCHEMA_TYPE_ERROR"
    # Other bounded contract ValueErrors deliberately share one stable
    # category. Exception text may contain model output and is never retained.
    if isinstance(exc, ValueError):
        return "ROUTE_SCHEMA_VALUE_ERROR"
    return "ROUTE_PARSE_ERROR"


def _parse_repair_feedback(
    failed_proposal: dict[str, object],
    *,
    parse_error_code: str,
    repair_attempt_index: int = 1,
    repeated_unchanged: bool = False,
) -> ObstacleParseRepairFeedback:
    """Convert one audit envelope into the only feedback shape sent to Qwen."""

    audit = failed_proposal.get("raw_model_response_audit")
    if not isinstance(audit, dict):
        raise RuntimeError("failed model proposal audit is unavailable")
    status = audit.get("structured_payload_status")
    raw = audit.get("raw_json_object")
    tail = audit.get("response_text_tail")
    # Feeding a complete, semantically invalid route back to a deterministic
    # decoder strongly anchors it to the bad coordinates.  Keep that object in
    # the coordinator audit record, but for duplicate geometry send only the
    # stable error facts and correction contract.
    if (
        parse_error_code
        in {
            "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE",
            "ROUTE_TERMINAL_SUFFIX_INVALID",
        }
        and status == "PRESERVED"
        and isinstance(raw, dict)
    ):
        output_kind = "OMITTED_REPETITION_RISK"
        output = None
    elif status == "PRESERVED" and isinstance(raw, dict):
        output_kind = "JSON_OBJECT"
        output: dict[str, object] | str | None = raw
    elif status == "REDACTED_SENSITIVE_CONTENT":
        output_kind = "OMITTED_SENSITIVE"
        output = None
    elif isinstance(tail, str) and tail:
        output_kind = "TEXT_TAIL"
        output = tail
    else:
        output_kind = "OMITTED_UNAVAILABLE"
        output = None
    length = audit.get("response_text_length", 0)
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        length = 0
    bounded_length = min(length, 1_000_000_000)
    truncated = bool(audit.get("response_text_truncated", False)) or bounded_length != length
    return ObstacleParseRepairFeedback(
        error_code=parse_error_code,
        rejected_output_kind=output_kind,
        rejected_model_output=output,
        response_text_length=bounded_length,
        response_text_truncated=truncated,
        repair_attempt_index=repair_attempt_index,
        repeated_unchanged=repeated_unchanged,
    )


def _latest_structural_output_repeated(
    records: list[ObstacleRevisionCoordinatorRecord],
    *,
    round_index: int,
) -> bool:
    """Compare bounded preserved JSON only; never retain or compare images."""

    candidates = [
        record
        for record in records
        if record.round_index == round_index
        and record.outcome in {"REVISE_STRUCTURE", "EXHAUSTED_STRUCTURE"}
        and isinstance(record.proposal, dict)
    ]
    if len(candidates) < 2:
        return False

    def _raw(record: ObstacleRevisionCoordinatorRecord) -> object:
        assert isinstance(record.proposal, dict)
        audit = record.proposal.get("raw_model_response_audit")
        if not isinstance(audit, dict) or audit.get("structured_payload_status") != "PRESERVED":
            return None
        return audit.get("raw_json_object")

    previous = _raw(candidates[-2])
    current = _raw(candidates[-1])
    return previous is not None and current is not None and previous == current


def _failed_response_audit(
    result: AsyncModelResult,
    *,
    parse_error_code: str,
) -> dict[str, object]:
    """Return a bounded, image-free audit envelope for one rejected output.

    Safe JSON objects are preserved semantically.  Malformed, oversized, or
    sensitive responses retain only their original character count and a
    bounded redacted tail.  Request messages and RGB are never copied here.
    """

    content = "" if result.response is None else result.response.content
    tail, redacted = _redacted_audit_tail(content)
    audit: dict[str, object] = {
        "audit_schema_version": 1,
        "parse_error_code": parse_error_code,
        "response_available": result.response is not None,
        "response_text_length": len(content),
        "response_text_tail": tail,
        "response_text_truncated": redacted or len(content) > len(tail),
        "structured_payload_status": "NO_RESPONSE",
    }
    if result.response is not None:
        status, raw_object = _safe_raw_json_object(content)
        audit["structured_payload_status"] = status
        if raw_object is not None:
            audit["raw_json_object"] = raw_object
    return {"raw_model_response_audit": audit}


def _redacted_audit_tail(content: str) -> tuple[str, bool]:
    """Return at most 500 characters, dropping all text after a secret marker."""

    lowered = content.casefold()
    offsets = [
        offset
        for marker in _SENSITIVE_TEXT_MARKERS
        if (offset := lowered.find(marker)) >= 0
    ]
    redacted = bool(offsets)
    safe = content
    if offsets:
        safe = content[: min(offsets)] + "[REDACTED_SENSITIVE_CONTENT]"
    return safe[-_AUDIT_TAIL_CHARS:], redacted


def _safe_raw_json_object(content: str) -> tuple[str, dict[str, object] | None]:
    if len(content) > _AUDIT_JSON_INPUT_CHARS:
        return "OMITTED_SIZE_LIMIT", None
    try:
        parsed = json.loads(
            content,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            object_pairs_hook=_strict_audit_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return "NOT_JSON_OBJECT", None
    if not isinstance(parsed, dict):
        return "NOT_JSON_OBJECT", None
    if _contains_sensitive_json(parsed):
        return "REDACTED_SENSITIVE_CONTENT", None
    try:
        frozen = validated_json_payload(parsed, field_name="raw_model_response")
    except (TypeError, ValueError):
        return "OMITTED_BOUNDS", None
    return "PRESERVED", json_payload_to_dict(frozen)


def _strict_audit_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _contains_sensitive_json(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = key.casefold().replace("-", "_")
            if normalized_key in _SENSITIVE_JSON_FIELDS or any(
                marker in normalized_key
                for marker in (
                    "api_key",
                    "authorization",
                    "credential",
                    "private_key",
                    "password",
                    "secret",
                )
            ) or normalized_key.endswith("_token"):
                return True
            if _contains_sensitive_json(nested):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_sensitive_json(item) for item in value)
    if isinstance(value, str):
        lowered = value.casefold()
        return any(marker in lowered for marker in _SENSITIVE_TEXT_MARKERS)
    return False


__all__ = [
    "ObstacleRevisionCoordinator",
    "ObstacleRevisionCoordinatorRecord",
    "ObstacleRevisionCoordinatorSnapshot",
    "ObstacleRevisionCoordinatorState",
]
