"""Priority-aware global broker for per-UAV model calls.

The broker owns scheduling metadata only; it never shares or mutates a model
client.  Callers acquire a request, execute it with the role-specific client
created by ``ModelClientFactory``, then complete it with the requested and
effective model names for auditable logging.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from math import isfinite
from numbers import Integral, Real
from threading import RLock
from time import monotonic

from common.ids import (
    generate_routing_id,
    validate_request_id,
    validate_routing_id,
    validate_uav_id,
)


class ModelRequestPriority(IntEnum):
    P0_IMMINENT_COLLISION = 0
    P1_FLEET_REPLAN = 1
    P2_AGENT_RUNTIME_REPLAN = 2
    P3_RUNTIME_VISUAL_REVIEW = 3
    P4_PERIODIC_REVIEW = 4


class BrokerRequestState(str):
    PENDING = "PENDING"
    INFLIGHT = "INFLIGHT"
    COMPLETED = "COMPLETED"
    STALE = "STALE"
    FAILED = "FAILED"


class ModelRequestBrokerError(RuntimeError):
    pass


def _role_text(value: object) -> str:
    if hasattr(value, "value"):
        value = getattr(value, "value")
    return validate_routing_id(value, "call_role")


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


@dataclass(frozen=True, slots=True)
class ModelBrokerRequest:
    call_role: str
    priority: ModelRequestPriority
    uav_id: str | None = None
    assignment_id: str | None = None
    requested_adapter: str | None = None
    payload: Mapping[str, object] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: generate_routing_id("request"))
    submitted_at_s: float = field(default_factory=monotonic)
    control_related: bool = True
    replaceable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_role", _role_text(self.call_role))
        if isinstance(self.priority, bool):
            raise TypeError("priority must be a ModelRequestPriority or integer")
        if not isinstance(self.priority, ModelRequestPriority):
            object.__setattr__(self, "priority", ModelRequestPriority(self.priority))
        if self.uav_id is not None:
            object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if self.assignment_id is not None:
            object.__setattr__(
                self,
                "assignment_id",
                validate_routing_id(self.assignment_id, "assignment_id"),
            )
        if self.requested_adapter is not None:
            object.__setattr__(
                self,
                "requested_adapter",
                validate_routing_id(self.requested_adapter, "requested_adapter"),
            )
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        object.__setattr__(self, "payload", deepcopy(dict(self.payload)))
        object.__setattr__(self, "request_id", validate_request_id(self.request_id))
        object.__setattr__(
            self,
            "submitted_at_s",
            _finite_nonnegative(self.submitted_at_s, "submitted_at_s"),
        )
        if not isinstance(self.control_related, bool):
            raise TypeError("control_related must be bool")
        if not isinstance(self.replaceable, bool):
            raise TypeError("replaceable must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "call_role": self.call_role,
            "uav_id": self.uav_id,
            "assignment_id": self.assignment_id,
            "priority": self.priority.name,
            "requested_adapter": self.requested_adapter,
            "submitted_at_s": self.submitted_at_s,
            "control_related": self.control_related,
            "replaceable": self.replaceable,
        }


@dataclass(frozen=True, slots=True)
class ModelCallLogRecord:
    request_id: str
    call_role: str
    uav_id: str | None
    assignment_id: str | None
    priority: str
    requested_adapter: str | None
    adapter_status: str | None
    effective_model: str | None
    fallback_used: bool
    latency_s: float
    state: str
    stale_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", validate_request_id(self.request_id))
        object.__setattr__(self, "call_role", _role_text(self.call_role))
        if self.uav_id is not None:
            object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if self.assignment_id is not None:
            object.__setattr__(self, "assignment_id", validate_routing_id(self.assignment_id, "assignment_id"))
        object.__setattr__(self, "latency_s", _finite_nonnegative(self.latency_s, "latency_s"))
        object.__setattr__(self, "prompt_tokens", _nonnegative_int(self.prompt_tokens, "prompt_tokens"))
        object.__setattr__(self, "completion_tokens", _nonnegative_int(self.completion_tokens, "completion_tokens"))
        if not isinstance(self.fallback_used, bool):
            raise TypeError("fallback_used must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "call_id": self.request_id,
            "call_role": self.call_role,
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "priority": self.priority,
            "requested_adapter": self.requested_adapter,
            "adapter_status": self.adapter_status,
            "effective_model": self.effective_model,
            "fallback_used": self.fallback_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": self.latency_s,
            "finish_reason": self.finish_reason,
            "error_code": self.error_code,
            "state": self.state,
            "stale_reasons": [] if self.stale_reason is None else [self.stale_reason],
        }


class GlobalModelRequestBroker:
    """Bounded priority queue with per-UAV isolation and frame replacement."""

    def __init__(
        self,
        *,
        max_inflight_global: int = 4,
        max_inflight_per_uav: int = 1,
        max_pending_per_uav: int = 2,
        starvation_timeout_s: float = 15.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        for name, value in (
            ("max_inflight_global", max_inflight_global),
            ("max_inflight_per_uav", max_inflight_per_uav),
            ("max_pending_per_uav", max_pending_per_uav),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.max_inflight_global = max_inflight_global
        self.max_inflight_per_uav = max_inflight_per_uav
        self.max_pending_per_uav = max_pending_per_uav
        self.starvation_timeout_s = _finite_nonnegative(
            starvation_timeout_s,
            "starvation_timeout_s",
        )
        if self.starvation_timeout_s <= 0.0:
            raise ValueError("starvation_timeout_s must be greater than zero")
        self._clock = clock
        self._pending: list[ModelBrokerRequest] = []
        self._inflight: dict[str, tuple[ModelBrokerRequest, float]] = {}
        self._preempted_inflight: dict[str, ModelCallLogRecord] = {}
        self._logs: list[ModelCallLogRecord] = []
        self._lock = RLock()

    @property
    def logs(self) -> tuple[ModelCallLogRecord, ...]:
        with self._lock:
            return tuple(self._logs)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

    def submit(self, request: ModelBrokerRequest) -> str:
        if not isinstance(request, ModelBrokerRequest):
            raise TypeError("request must be a ModelBrokerRequest")
        with self._lock:
            if request.request_id in self._inflight or any(
                item.request_id == request.request_id for item in self._pending
            ) or any(
                record.request_id == request.request_id for record in self._logs
            ):
                raise ModelRequestBrokerError("request_id already exists")

            if request.replaceable:
                replace_index = next(
                    (
                        index
                        for index, existing in enumerate(self._pending)
                        if existing.replaceable
                        and existing.uav_id == request.uav_id
                        and existing.call_role == request.call_role
                        and existing.assignment_id == request.assignment_id
                        # A less urgent refresh must never overwrite queued
                        # emergency/replan work, even if a caller mistakenly
                        # marked that older request as replaceable.
                        and request.priority <= existing.priority
                    ),
                    None,
                )
                if replace_index is not None:
                    stale = self._pending.pop(replace_index)
                    self._record_stale(stale, "SUPERSEDED_BY_NEWER_FRAME")

            scoped_pending = [item for item in self._pending if item.uav_id == request.uav_id]
            if len(scoped_pending) >= self.max_pending_per_uav:
                # Urgent work may evict the least urgent replaceable review,
                # but it can never overwrite another urgent/replan request.
                candidates = [
                    item
                    for item in scoped_pending
                    if item.replaceable and item.priority > request.priority
                ]
                if not candidates:
                    raise ModelRequestBrokerError("per-UAV pending queue is full")
                evicted = max(candidates, key=lambda item: (item.priority, -item.submitted_at_s))
                self._pending.remove(evicted)
                self._record_stale(evicted, "EVICTED_BY_HIGHER_PRIORITY")
            self._preempt_replaceable_inflight_for_urgent(request)
            self._pending.append(request)
            return request.request_id

    def acquire_next(self) -> ModelBrokerRequest | None:
        with self._lock:
            if len(self._inflight) >= self.max_inflight_global:
                return None
            now = _finite_nonnegative(self._clock(), "clock result")
            eligible = [item for item in self._pending if self._eligible(item)]
            if not eligible:
                return None
            selected = min(
                eligible,
                key=lambda item: (
                    self._effective_priority(item, now),
                    item.submitted_at_s,
                    item.request_id,
                ),
            )
            self._pending.remove(selected)
            self._inflight[selected.request_id] = (selected, now)
            return selected

    def _eligible(self, request: ModelBrokerRequest) -> bool:
        if request.uav_id is None:
            return True
        same_uav = [
            item
            for item, _ in self._inflight.values()
            if item.uav_id == request.uav_id
        ]
        if len(same_uav) >= self.max_inflight_per_uav:
            return False
        if request.control_related and any(item.control_related for item in same_uav):
            return False
        return True

    def _effective_priority(self, request: ModelBrokerRequest, now: float) -> int:
        base_priority = int(request.priority)
        # P0--P2 are safety/replanning work.  Aging a visual P3/P4 request
        # across that boundary would let an old periodic review run before a
        # newly submitted collision response, contradicting the Broker's
        # primary safety guarantee.  Aging therefore provides fairness only
        # inside the visual-review band (P4 may rise to P3).
        if request.priority <= ModelRequestPriority.P2_AGENT_RUNTIME_REPLAN:
            return base_priority
        waited = max(0.0, now - request.submitted_at_s)
        boost = int(waited // self.starvation_timeout_s)
        return max(int(ModelRequestPriority.P3_RUNTIME_VISUAL_REVIEW), base_priority - boost)

    def _preempt_replaceable_inflight_for_urgent(
        self,
        request: ModelBrokerRequest,
    ) -> None:
        """Logically preempt stale visual work so P0/P1 can be dispatched.

        An HTTP/model call may not be synchronously cancelable.  Removing only a
        request explicitly marked ``replaceable`` makes its eventual response
        stale and frees the broker slot immediately; :meth:`complete` then
        returns the existing STALE record instead of resurrecting that result.
        """

        if request.priority > ModelRequestPriority.P1_FLEET_REPLAN:
            return
        now = _finite_nonnegative(self._clock(), "clock result")
        while True:
            same_uav = tuple(
                (request_id, item, started_at)
                for request_id, (item, started_at) in self._inflight.items()
                if request.uav_id is not None and item.uav_id == request.uav_id
            )
            per_uav_blocked = (
                request.uav_id is not None
                and len(same_uav) >= self.max_inflight_per_uav
            )
            global_blocked = len(self._inflight) >= self.max_inflight_global
            if not per_uav_blocked and not global_blocked:
                return
            scope = same_uav if per_uav_blocked else tuple(
                (request_id, item, started_at)
                for request_id, (item, started_at) in self._inflight.items()
            )
            candidates = tuple(
                entry
                for entry in scope
                if entry[1].replaceable
                and entry[1].priority > request.priority
            )
            if not candidates:
                return
            request_id, victim, started_at = max(
                candidates,
                key=lambda entry: (
                    entry[1].priority,
                    -entry[2],
                    entry[0],
                ),
            )
            self._inflight.pop(request_id)
            stale = self._record_stale(
                victim,
                "PREEMPTED_BY_HIGHER_PRIORITY",
                latency_s=max(0.0, now - started_at),
            )
            self._preempted_inflight[request_id] = stale

    def complete(
        self,
        request_id: str,
        *,
        requested_adapter: str | None = None,
        adapter_status: str | None = None,
        effective_model: str | None = None,
        fallback_used: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: str | None = None,
        error_code: str | None = None,
    ) -> ModelCallLogRecord:
        request_id = validate_request_id(request_id)
        with self._lock:
            try:
                request, started_at = self._inflight.pop(request_id)
            except KeyError:
                preempted = self._preempted_inflight.pop(request_id, None)
                if preempted is not None:
                    return preempted
                raise ModelRequestBrokerError("request is not inflight") from None
            now = _finite_nonnegative(self._clock(), "clock result")
            record = ModelCallLogRecord(
                request_id=request.request_id,
                call_role=request.call_role,
                uav_id=request.uav_id,
                assignment_id=request.assignment_id,
                priority=request.priority.name,
                requested_adapter=requested_adapter or request.requested_adapter,
                adapter_status=adapter_status,
                effective_model=effective_model,
                fallback_used=fallback_used,
                latency_s=max(0.0, now - started_at),
                state=(BrokerRequestState.FAILED if error_code else BrokerRequestState.COMPLETED),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                error_code=error_code,
            )
            self._logs.append(record)
            return record

    def cancel_pending(self, request_id: str, *, reason: str = "CANCELED") -> None:
        request_id = validate_request_id(request_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        with self._lock:
            request = next((item for item in self._pending if item.request_id == request_id), None)
            if request is None:
                raise ModelRequestBrokerError("request is not pending")
            self._pending.remove(request)
            self._record_stale(request, reason.strip())

    def _record_stale(
        self,
        request: ModelBrokerRequest,
        reason: str,
        *,
        latency_s: float = 0.0,
    ) -> ModelCallLogRecord:
        record = ModelCallLogRecord(
            request_id=request.request_id,
            call_role=request.call_role,
            uav_id=request.uav_id,
            assignment_id=request.assignment_id,
            priority=request.priority.name,
            requested_adapter=request.requested_adapter,
            adapter_status=None,
            effective_model=None,
            fallback_used=False,
            latency_s=latency_s,
            state=BrokerRequestState.STALE,
            stale_reason=reason,
        )
        self._logs.append(record)
        return record

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "pending": [item.to_dict() for item in sorted(self._pending, key=lambda item: item.request_id)],
                "inflight": [
                    item.to_dict()
                    for item, _ in sorted(self._inflight.values(), key=lambda pair: pair[0].request_id)
                ],
                "logs": [record.to_dict() for record in self._logs],
                "limits": {
                    "max_inflight_global": self.max_inflight_global,
                    "max_inflight_per_uav": self.max_inflight_per_uav,
                    "max_pending_per_uav": self.max_pending_per_uav,
                    "starvation_timeout_s": self.starvation_timeout_s,
                },
            }

    def summary_snapshot(self) -> dict[str, object]:
        """Return bounded scheduler state without duplicating per-call logs."""

        with self._lock:
            return {
                "pending": [
                    item.to_dict()
                    for item in sorted(self._pending, key=lambda item: item.request_id)
                ],
                "inflight": [
                    item.to_dict()
                    for item, _ in sorted(
                        self._inflight.values(),
                        key=lambda pair: pair[0].request_id,
                    )
                ],
                "log_count": len(self._logs),
                "limits": {
                    "max_inflight_global": self.max_inflight_global,
                    "max_inflight_per_uav": self.max_inflight_per_uav,
                    "max_pending_per_uav": self.max_pending_per_uav,
                    "starvation_timeout_s": self.starvation_timeout_s,
                },
            }


__all__ = [
    "BrokerRequestState",
    "GlobalModelRequestBroker",
    "ModelBrokerRequest",
    "ModelCallLogRecord",
    "ModelRequestBrokerError",
    "ModelRequestPriority",
]
