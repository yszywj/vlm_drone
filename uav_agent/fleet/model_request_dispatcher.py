"""Thread-safe bridge from the global Broker to per-UAV async workers.

The dispatcher deliberately exposes the same non-blocking ``submit``/``poll``
surface consumed by :class:`VisualReviewCoordinator`.  Requests cannot cross
the HTTP/model boundary until :class:`GlobalModelRequestBroker` has admitted
and acquired them.  Polling any UAV services every worker, completes Broker
accounting, and opens newly available global/per-UAV slots.

Scheduling is pumped from caller threads under one re-entrant lock rather than
from another background thread.  Actual model calls remain asynchronous inside
``AsyncModelWorker``; this keeps admission deterministic and makes shutdown
ownership explicit.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import replace
from math import isfinite
from numbers import Real
from threading import RLock
from time import monotonic

from common.ids import (
    validate_request_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from fleet.model_request_broker import (
    BrokerRequestState,
    GlobalModelRequestBroker,
    ModelBrokerRequest,
    ModelCallLogRecord,
    ModelRequestBrokerError,
    ModelRequestPriority,
)
from models.adapter_registry import AdapterSelection, ModelCallRole
from models.async_worker import AsyncModelRequest, AsyncModelResult


class ModelRequestDispatcherError(RuntimeError):
    """Raised when Dispatcher ownership or routing invariants are violated."""


def _timestamp(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


class BrokeredAsyncModelWorker:
    """One UAV-scoped facade compatible with ``VisualReviewCoordinator``."""

    __slots__ = ("_dispatcher", "uav_id", "assignment_id")

    def __init__(
        self,
        dispatcher: ModelRequestDispatcher,
        uav_id: str,
        assignment_id: str | None,
    ) -> None:
        self._dispatcher = dispatcher
        self.uav_id = validate_uav_id(uav_id)
        self.assignment_id = assignment_id

    @property
    def is_busy(self) -> bool:
        return self._dispatcher._is_uav_busy(self.uav_id)

    @property
    def discarded_result_count(self) -> int:
        return self._dispatcher._discarded_for(self.uav_id)

    def submit(self, request: AsyncModelRequest) -> None:
        self._dispatcher._submit(self, request)

    def poll(
        self,
        *,
        expected_request_id: str | None = None,
        expected_review_id: str | None = None,
        minimum_observation_timestamp_s: float | None = None,
        include_stale: bool = False,
    ) -> AsyncModelResult | None:
        return self._dispatcher._poll(
            self,
            expected_request_id=expected_request_id,
            expected_review_id=expected_review_id,
            minimum_observation_timestamp_s=minimum_observation_timestamp_s,
            include_stale=include_stale,
        )


class ModelRequestDispatcher:
    """Own all visual workers and enforce one shared Broker admission path."""

    def __init__(
        self,
        broker: GlobalModelRequestBroker,
        workers: Mapping[str, object],
        *,
        adapter_selection: AdapterSelection,
        clock: Callable[[], float] = monotonic,
        max_completed_results_per_uav: int = 16,
        record_logger: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(broker, GlobalModelRequestBroker):
            raise TypeError("broker must be a GlobalModelRequestBroker")
        if not isinstance(workers, Mapping) or not workers:
            raise TypeError("workers must be a non-empty mapping")
        if not isinstance(adapter_selection, AdapterSelection):
            raise TypeError("adapter_selection must be an AdapterSelection")
        if adapter_selection.call_role is not ModelCallRole.RUNTIME_VISUAL_REVIEW:
            raise ValueError(
                "adapter_selection must route RUNTIME_VISUAL_REVIEW"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        if record_logger is not None and not callable(record_logger):
            raise TypeError("record_logger must be callable or None")
        if (
            isinstance(max_completed_results_per_uav, bool)
            or not isinstance(max_completed_results_per_uav, int)
            or max_completed_results_per_uav <= 0
        ):
            raise ValueError("max_completed_results_per_uav must be a positive integer")

        normalized_workers: dict[str, object] = {}
        for raw_uav_id, worker in workers.items():
            uav_id = validate_uav_id(raw_uav_id)
            for method in ("submit", "poll", "close"):
                if not callable(getattr(worker, method, None)):
                    raise TypeError(f"workers[{uav_id!r}] must provide {method}()")
            if getattr(worker, "uav_id", None) != uav_id:
                raise ValueError(f"workers[{uav_id!r}] has mismatched uav_id")
            normalized_workers[uav_id] = worker

        self.broker = broker
        self.adapter_selection = adapter_selection
        self._workers = normalized_workers
        self._clock = clock
        self._record_logger = record_logger
        self._max_completed_results_per_uav = max_completed_results_per_uav
        self._requests: dict[str, AsyncModelRequest] = {}
        self._dispatched_request_ids: set[str] = set()
        self._results = {uav_id: deque() for uav_id in normalized_workers}
        self._discarded_results = {uav_id: 0 for uav_id in normalized_workers}
        self._facades: dict[str, BrokeredAsyncModelWorker] = {}
        self._assignment_ids: dict[str, str | None] = {}
        self._emitted_record_ids: set[str] = set()
        self._closed = False
        self._lock = RLock()

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def workers(self) -> Mapping[str, object]:
        with self._lock:
            return dict(self._workers)

    def worker_for(
        self,
        uav_id: str,
        *,
        assignment_id: str | None = None,
    ) -> BrokeredAsyncModelWorker:
        normalized_uav = validate_uav_id(uav_id)
        normalized_assignment = (
            None
            if assignment_id is None
            else validate_routing_id(assignment_id, "assignment_id")
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("model request dispatcher is closed")
            if normalized_uav not in self._workers:
                raise KeyError(f"no model worker is registered for {normalized_uav!r}")
            existing_assignment = self._assignment_ids.get(normalized_uav)
            if (
                normalized_uav in self._assignment_ids
                and existing_assignment != normalized_assignment
            ):
                raise ValueError(
                    f"UAV {normalized_uav!r} facade is already bound to assignment "
                    f"{existing_assignment!r}"
                )
            facade = self._facades.get(normalized_uav)
            if facade is None:
                facade = BrokeredAsyncModelWorker(
                    self,
                    normalized_uav,
                    normalized_assignment,
                )
                self._facades[normalized_uav] = facade
                self._assignment_ids[normalized_uav] = normalized_assignment
            return facade

    def close(self, timeout_s: float | None = None) -> None:
        """Cancel queued work, drain active calls, and close every worker once."""

        if timeout_s is not None:
            timeout_s = _timestamp(timeout_s, "timeout_s")
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for request_id in tuple(self._requests):
                if request_id in self._dispatched_request_ids:
                    continue
                try:
                    self.broker.cancel_pending(
                        request_id,
                        reason="DISPATCHER_CLOSED",
                    )
                except ModelRequestBrokerError:
                    # A concurrent Broker replacement may already have made it
                    # stale; reconciliation below owns the one synthetic result.
                    pass
            self._reconcile_pending_stale()

            first_error: BaseException | None = None
            for uav_id in sorted(self._workers):
                try:
                    self._workers[uav_id].close(timeout_s=timeout_s)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            self._collect_worker_results()
            self._reconcile_pending_stale()
            if first_error is None and self._dispatched_request_ids:
                first_error = ModelRequestDispatcherError(
                    "worker close returned without a result for acquired request(s): "
                    + ", ".join(sorted(self._dispatched_request_ids))
                )
            if first_error is not None:
                raise first_error

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "closed": self._closed,
                "owned_request_ids": sorted(self._requests),
                "dispatched_request_ids": sorted(self._dispatched_request_ids),
                "queued_result_counts": {
                    uav_id: len(queue) for uav_id, queue in self._results.items()
                },
                "discarded_result_counts": dict(self._discarded_results),
                "emitted_record_ids": sorted(self._emitted_record_ids),
            }

    def __enter__(self) -> ModelRequestDispatcher:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _submit(
        self,
        facade: BrokeredAsyncModelWorker,
        request: AsyncModelRequest,
    ) -> None:
        if not isinstance(request, AsyncModelRequest):
            raise TypeError("request must be an AsyncModelRequest")
        if request.uav_id != facade.uav_id:
            raise ValueError(
                f"request uav_id {request.uav_id!r} does not match facade "
                f"uav_id {facade.uav_id!r}"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("model request dispatcher is closed")
            submitted_at_s = _timestamp(self._clock(), "clock result")
            broker_request = ModelBrokerRequest(
                request_id=request.request_id,
                call_role=ModelCallRole.RUNTIME_VISUAL_REVIEW.value,
                priority=ModelRequestPriority(request.broker_priority),
                uav_id=request.uav_id,
                assignment_id=facade.assignment_id,
                requested_adapter=self.adapter_selection.requested_adapter,
                submitted_at_s=submitted_at_s,
                control_related=False,
                replaceable=request.broker_replaceable,
                # No messages or image bytes cross into global Fleet state.
                payload={
                    "review_id": request.review_id,
                    "mission_id": request.mission_id,
                    "plan_version": request.plan_version,
                    "observation_timestamp_s": request.observation_timestamp_s,
                    "frame_id": request.frame_id,
                },
            )
            self.broker.submit(broker_request)
            self._requests[request.request_id] = request
            self._reconcile_pending_stale()
            self._pump()

    def _poll(
        self,
        facade: BrokeredAsyncModelWorker,
        *,
        expected_request_id: str | None,
        expected_review_id: str | None,
        minimum_observation_timestamp_s: float | None,
        include_stale: bool,
    ) -> AsyncModelResult | None:
        if expected_request_id is not None:
            expected_request_id = validate_request_id(expected_request_id)
        if expected_review_id is not None:
            expected_review_id = validate_review_id(expected_review_id)
        if minimum_observation_timestamp_s is not None:
            minimum_observation_timestamp_s = _timestamp(
                minimum_observation_timestamp_s,
                "minimum_observation_timestamp_s",
            )
        if not isinstance(include_stale, bool):
            raise TypeError("include_stale must be bool")

        with self._lock:
            self._service()
            queue = self._results[facade.uav_id]
            while queue:
                result = queue.popleft()
                stale = (
                    result.stale
                    or (
                        expected_request_id is not None
                        and result.request_id != expected_request_id
                    )
                    or (
                        expected_review_id is not None
                        and result.review_id != expected_review_id
                    )
                    or (
                        minimum_observation_timestamp_s is not None
                        and result.observation_timestamp_s
                        < minimum_observation_timestamp_s
                    )
                )
                if stale and not include_stale:
                    self._discarded_results[facade.uav_id] += 1
                    continue
                return replace(result, stale=stale)
            return None

    def _service(self) -> None:
        self._collect_worker_results()
        self._reconcile_pending_stale()
        if not self._closed:
            self._pump()

    def _pump(self) -> None:
        while True:
            broker_request = self.broker.acquire_next()
            if broker_request is None:
                return
            request = self._requests.get(broker_request.request_id)
            if request is None:
                raise ModelRequestDispatcherError(
                    "Broker acquired a request not owned by this dispatcher: "
                    f"{broker_request.request_id}"
                )
            worker = self._workers[request.uav_id]
            self._dispatched_request_ids.add(request.request_id)
            try:
                worker.submit(request)
            except Exception as exc:
                self._dispatched_request_ids.discard(request.request_id)
                self._requests.pop(request.request_id, None)
                record = self.broker.complete(
                    request.request_id,
                    requested_adapter=self.adapter_selection.requested_adapter,
                    adapter_status=self.adapter_selection.adapter_status.value,
                    effective_model=self.adapter_selection.effective_model,
                    fallback_used=self.adapter_selection.fallback_used,
                    error_code="WORKER_SUBMIT_FAILED",
                )
                self._enqueue_result(
                    AsyncModelResult(
                        request_id=request.request_id,
                        review_id=request.review_id,
                        mission_id=request.mission_id,
                        uav_id=request.uav_id,
                        plan_version=request.plan_version,
                        observation_timestamp_s=request.observation_timestamp_s,
                        frame_id=request.frame_id,
                        response=None,
                        error_code="MODEL_REQUEST_FAILED",
                        error_message=type(exc).__name__,
                    )
                )
                self._emit_record(record)

    def _collect_worker_results(self) -> None:
        for uav_id in sorted(self._workers):
            worker = self._workers[uav_id]
            while True:
                result = worker.poll(include_stale=True)
                if result is None:
                    break
                request = self._requests.get(result.request_id)
                if request is None or result.request_id not in self._dispatched_request_ids:
                    raise ModelRequestDispatcherError(
                        "worker returned a result without Dispatcher ownership: "
                        f"{result.request_id}"
                    )
                response = result.response
                usage = {} if response is None else response.usage
                record = self.broker.complete(
                    result.request_id,
                    requested_adapter=self.adapter_selection.requested_adapter,
                    adapter_status=self.adapter_selection.adapter_status.value,
                    effective_model=self.adapter_selection.effective_model,
                    fallback_used=self.adapter_selection.fallback_used,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    finish_reason=(None if response is None else response.finish_reason),
                    error_code=result.error_code,
                )
                self._dispatched_request_ids.remove(result.request_id)
                self._requests.pop(result.request_id, None)
                self._enqueue_result(
                    replace(
                        result,
                        stale=(
                            result.stale
                            or record.state == BrokerRequestState.STALE
                        ),
                    )
                )
                self._emit_record(record)

    def _reconcile_pending_stale(self) -> None:
        stale_by_id: dict[str, ModelCallLogRecord] = {
            record.request_id: record
            for record in self.broker.logs
            if record.state == BrokerRequestState.STALE
        }
        for request_id, request in tuple(self._requests.items()):
            if request_id in self._dispatched_request_ids:
                continue
            record = stale_by_id.get(request_id)
            if record is None:
                continue
            self._requests.pop(request_id, None)
            self._enqueue_result(
                AsyncModelResult(
                    request_id=request.request_id,
                    review_id=request.review_id,
                    mission_id=request.mission_id,
                    uav_id=request.uav_id,
                    plan_version=request.plan_version,
                    observation_timestamp_s=request.observation_timestamp_s,
                    frame_id=request.frame_id,
                    response=None,
                    error_code="BROKER_STALE",
                    error_message=record.stale_reason or "STALE",
                    stale=True,
                )
            )
            self._emit_record(record)

    def _emit_record(self, record: ModelCallLogRecord) -> None:
        """Publish one terminal Broker record at most once per request."""

        if record.request_id in self._emitted_record_ids:
            return
        self._emitted_record_ids.add(record.request_id)
        if self._record_logger is not None:
            self._record_logger(record.to_dict())

    def _enqueue_result(self, result: AsyncModelResult) -> None:
        queue = self._results[result.uav_id]
        if len(queue) >= self._max_completed_results_per_uav:
            queue.popleft()
            self._discarded_results[result.uav_id] += 1
        queue.append(result)

    def _is_uav_busy(self, uav_id: str) -> bool:
        with self._lock:
            return any(request.uav_id == uav_id for request in self._requests.values())

    def _discarded_for(self, uav_id: str) -> int:
        with self._lock:
            return self._discarded_results[uav_id]


__all__ = [
    "BrokeredAsyncModelWorker",
    "ModelRequestDispatcher",
    "ModelRequestDispatcherError",
]
