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
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from threading import RLock, Thread, get_ident
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

    def prepare_worker_for(self, uav_id: str, *, assignment_id: str) -> BrokeredAsyncModelWorker:
        """Construct an inert candidate facade without binding official routing.

        The candidate owner keeps it private until publication. No request can
        be produced by an idle MissionAgent; rejected candidates leave the
        dispatcher assignment map untouched.
        """
        normalized_uav = validate_uav_id(uav_id)
        normalized_assignment = validate_routing_id(assignment_id, "assignment_id")
        with self._lock:
            if self._closed:
                raise RuntimeError("model request dispatcher is closed")
            if normalized_uav not in self._workers:
                raise KeyError("no model worker is registered for " + normalized_uav)
            existing = self._assignment_ids.get(normalized_uav)
            if normalized_uav in self._assignment_ids and existing != normalized_assignment:
                raise ValueError("candidate UAV already has another visual assignment")
            return BrokeredAsyncModelWorker(self, normalized_uav, normalized_assignment)

    def close(self, timeout_s: float | None = None) -> None:
        """Cancel queued work and close workers within one shared time budget."""

        timeout_s = 0.1 if timeout_s is None else _timestamp(timeout_s, "timeout_s")
        shutdown_deadline = monotonic() + timeout_s
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
                    self._workers[uav_id].close(
                        timeout_s=max(0.0, shutdown_deadline - monotonic())
                    )
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
            for _ in range(len(queue)):
                result = queue.popleft()
                matches = (
                    (expected_request_id is None or result.request_id == expected_request_id)
                    and (expected_review_id is None or result.review_id == expected_review_id)
                )
                if not matches:
                    queue.append(result)
                    continue
                stale = result.stale or (
                    minimum_observation_timestamp_s is not None
                    and result.observation_timestamp_s < minimum_observation_timestamp_s
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
            broker_request = self.broker.acquire_next(request_ids=self._requests)
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



@dataclass(frozen=True, slots=True)
class TextTaskResult:
    """A request-owned compute outcome; stale values must never be published.

    Exceptions are returned only to the owner, never copied into Broker logs.
    """

    request_id: str
    value: object = None
    exception: BaseException | None = None
    stale: bool = False
    reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return not self.stale and self.exception is None


@dataclass(frozen=True, slots=True)
class _TextTask:
    request: ModelBrokerRequest
    prepare: Callable[[], Callable[[], object] | None]
    deadline_at_s: float | None
    adapter_selection: AdapterSelection | None


class BrokeredTextTaskRunner:
    """Bounded pure text computation using the visual dispatcher's Broker.

    ``submit`` only queues. ``pump``/``poll`` run on the constructing thread;
    immediately before launching an admitted task they call ``prepare`` there.
    That callback refreshes and approves the owner's snapshot, and returns a
    pure callable (or None to reject it). Only that callable runs off-thread.

    Cancel and deadlines revoke publication eligibility. Running calls retain
    their Broker resource leases until their threads actually return, even
    after close. Results are consumed independently by request ID.
    """

    _ROLE_PRIORITIES = {
        ModelCallRole.FLEET_REPLAN.value: ModelRequestPriority.P1_FLEET_REPLAN,
        ModelCallRole.RUNTIME_REPLAN.value: ModelRequestPriority.P2_AGENT_RUNTIME_REPLAN,
    }

    def __init__(
        self,
        broker: GlobalModelRequestBroker,
        *,
        clock: Callable[[], float] = monotonic,
        max_workers: int | None = None,
        max_outstanding_tasks: int = 64,
        record_logger: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not isinstance(broker, GlobalModelRequestBroker):
            raise TypeError("broker must be a GlobalModelRequestBroker")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if max_workers is None:
            max_workers = broker.max_inflight_global
        for value, name in ((max_workers, "max_workers"),
                            (max_outstanding_tasks, "max_outstanding_tasks")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if record_logger is not None and not callable(record_logger):
            raise TypeError("record_logger must be callable or None")
        self.broker = broker
        self._clock = clock
        self._max_workers = max_workers
        self._max_outstanding_tasks = max_outstanding_tasks
        self._record_logger = record_logger
        self._owner_thread_id = get_ident()
        self._tasks: dict[str, _TextTask] = {}
        self._active: dict[str, Thread | None] = {}
        self._done: deque[TextTaskResult] = deque()
        self._results: dict[str, TextTaskResult] = {}
        self._revoked: dict[str, str] = {}
        self._emitted_record_ids: set[str] = set()
        self._closed = False
        self._lock = RLock()

    def _assert_owner(self) -> None:
        if get_ident() != self._owner_thread_id:
            raise RuntimeError("text task admission must run on its owner thread")

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def inflight_count(self) -> int:
        """Actual occupied/unacknowledged calls, including revoked calls."""
        with self._lock:
            return len(self._active)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._tasks) - len(self._active)

    def submit(
        self,
        request: ModelBrokerRequest,
        prepare: Callable[[], Callable[[], object] | None],
        *,
        deadline_at_s: float | None = None,
        adapter_selection: AdapterSelection | None = None,
    ) -> str:
        self._assert_owner()
        if not isinstance(request, ModelBrokerRequest):
            raise TypeError("request must be a ModelBrokerRequest")
        allowed_priority = self._ROLE_PRIORITIES.get(request.call_role)
        if allowed_priority is None or request.priority != allowed_priority:
            raise ValueError("text task role and priority must be trusted replanning roles")
        if not request.control_related:
            raise ValueError("text replanning tasks must be control_related")
        if not callable(prepare):
            raise TypeError("prepare must be callable")
        if deadline_at_s is not None:
            deadline_at_s = _timestamp(deadline_at_s, "deadline_at_s")
        if adapter_selection is not None:
            if not isinstance(adapter_selection, AdapterSelection):
                raise TypeError("adapter_selection must be an AdapterSelection")
            if adapter_selection.call_role.value != request.call_role:
                raise ValueError("adapter_selection role does not match text task")
            if (request.requested_adapter is not None
                    and request.requested_adapter != adapter_selection.requested_adapter):
                raise ValueError("adapter_selection does not match requested_adapter")
        with self._lock:
            if self._closed:
                raise RuntimeError("text task runner is closed")
            if len(self._tasks.keys() | self._results.keys()) >= self._max_outstanding_tasks:
                raise ModelRequestDispatcherError("text task outstanding capacity is full")
            self.broker.submit(request)
            self._tasks[request.request_id] = _TextTask(
                request, prepare, deadline_at_s, adapter_selection
            )
            return request.request_id

    def pump(self) -> None:
        """Nonblocking owner-thread admission, timeout and completion service."""
        self._assert_owner()
        with self._lock:
            now = _timestamp(self._clock(), "clock result")
            for request_id, task in tuple(self._tasks.items()):
                if task.deadline_at_s is not None and now >= task.deadline_at_s:
                    self._cancel_locked(request_id, "DEADLINE_EXCEEDED")
            # Another Broker owner may have superseded/preempted our request.
            for record in self.broker.logs:
                if (record.state == BrokerRequestState.STALE
                        and record.request_id in self._tasks):
                    self._cancel_locked(record.request_id, record.stale_reason or "STALE")
            self._collect_done()
            while not self._closed and len(self._active) < self._max_workers:
                request = self.broker.acquire_next(request_ids=self._tasks)
                if request is None:
                    break
                task = self._tasks[request.request_id]
                self._active[request.request_id] = None
                try:
                    compute = task.prepare()
                    if compute is not None and not callable(compute):
                        raise TypeError("prepare must return a callable or None")
                except Exception as exc:
                    self._finish(TextTaskResult(request.request_id, exception=exc))
                    continue
                if compute is None:
                    self._cancel_locked(request.request_id, "SNAPSHOT_REJECTED")
                if (task.deadline_at_s is not None
                        and _timestamp(self._clock(), "clock result") >= task.deadline_at_s):
                    self._cancel_locked(request.request_id, "DEADLINE_EXCEEDED")
                if request.request_id in self._revoked:
                    # No callable was started, so this reserved lease is free.
                    self._finish(TextTaskResult(request.request_id))
                    continue
                thread = Thread(
                    target=self._run_compute,
                    args=(request.request_id, compute),
                    name=f"text-model-{request.request_id}",
                    daemon=True,
                )
                self._active[request.request_id] = thread
                try:
                    thread.start()
                except Exception as exc:
                    self._finish(TextTaskResult(request.request_id, exception=exc))

    def poll(self, request_id: str) -> TextTaskResult | None:
        request_id = validate_request_id(request_id)
        self.pump()
        with self._lock:
            return self._results.pop(request_id, None)

    def cancel(self, request_id: str, *, reason: str = "CANCELED") -> bool:
        self._assert_owner()
        request_id = validate_request_id(request_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        with self._lock:
            return self._cancel_locked(request_id, reason.strip())

    def _cancel_locked(self, request_id: str, reason: str) -> bool:
        if request_id in self._revoked:
            return False
        task = self._tasks.get(request_id)
        if task is None:
            result = self._results.get(request_id)
            if result is None or result.stale:
                return False
            self._results[request_id] = TextTaskResult(request_id, stale=True, reason=reason)
            return True
        if request_id in self._active:
            record = self.broker.cancel_inflight(request_id, reason=reason)
            self._revoked[request_id] = record.stale_reason or reason
            self._emit_record(record)
        else:
            try:
                self.broker.cancel_pending(request_id, reason=reason)
            except ModelRequestBrokerError:
                # A Broker replacement may already have removed queued work.
                pass
            self._tasks.pop(request_id)
            for record in reversed(self.broker.logs):
                if record.request_id == request_id:
                    reason = record.stale_reason or reason
                    self._emit_record(record)
                    break
        self._results[request_id] = TextTaskResult(request_id, stale=True, reason=reason)
        return True

    def _run_compute(self, request_id: str, compute: Callable[[], object]) -> None:
        try:
            result = TextTaskResult(request_id, value=compute())
        except BaseException as exc:
            result = TextTaskResult(request_id, exception=exc)
        with self._lock:
            self._done.append(result)

    def _collect_done(self) -> None:
        while self._done:
            self._finish(self._done.popleft())

    def _finish(self, result: TextTaskResult) -> None:
        task = self._tasks.pop(result.request_id)
        self._active.pop(result.request_id)
        selection = task.adapter_selection
        record = self.broker.complete(
            result.request_id,
            requested_adapter=(None if selection is None else selection.requested_adapter),
            adapter_status=(None if selection is None else selection.adapter_status.value),
            effective_model=(None if selection is None else selection.effective_model),
            fallback_used=False if selection is None else selection.fallback_used,
            error_code="TEXT_TASK_FAILED" if result.exception is not None else None,
        )
        revoked = self._revoked.pop(result.request_id, None)
        if revoked is None:
            self._results[result.request_id] = replace(
                result,
                stale=record.state == BrokerRequestState.STALE,
                reason=record.stale_reason,
            )
        self._emit_record(record)

    def _emit_record(self, record: ModelCallLogRecord) -> None:
        if record.request_id in self._emitted_record_ids:
            return
        self._emitted_record_ids.add(record.request_id)
        if self._record_logger is not None:
            self._record_logger(record.to_dict())

    def close(self, timeout_s: float | None = 0.1) -> None:
        """Revoke work and wait at most one real monotonic shutdown budget.

        A fake scheduling clock must not make shutdown wait forever. Daemon
        threads still running after this budget keep their actual Broker slots;
        a later owner-thread pump/close may acknowledge their completion.
        """
        self._assert_owner()
        budget = 0.1 if timeout_s is None else _timestamp(timeout_s, "timeout_s")
        deadline = monotonic() + budget
        with self._lock:
            self._closed = True
            for request_id in tuple(self._tasks):
                self._cancel_locked(request_id, "RUNNER_CLOSED")
            threads = tuple(thread for thread in self._active.values() if thread is not None)
        for thread in threads:
            thread.join(max(0.0, deadline - monotonic()))
        with self._lock:
            self._collect_done()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "closed": self._closed,
                "pending_request_ids": sorted(self._tasks.keys() - self._active.keys()),
                "inflight_request_ids": sorted(self._active),
                "revoked_request_ids": sorted(self._revoked),
                "completed_request_ids": sorted(self._results),
            }

__all__ = [
    "BrokeredAsyncModelWorker",
    "BrokeredTextTaskRunner",
    "TextTaskResult",
    "ModelRequestDispatcher",
    "ModelRequestDispatcherError",
]
