from __future__ import annotations

from collections import deque
import threading

import pytest

from fleet.model_request_broker import (
    GlobalModelRequestBroker,
    ModelBrokerRequest,
    ModelRequestPriority,
)
from fleet.model_request_dispatcher import ModelRequestDispatcher
from models import (
    AdapterSelection,
    AdapterStatus,
    AsyncModelRequest,
    AsyncModelResult,
    ChatMessage,
    ModelCallRole,
    ModelResponse,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now


class _Worker:
    def __init__(self, uav_id: str) -> None:
        self.uav_id = uav_id
        self.submitted: deque[AsyncModelRequest] = deque()
        self.results: deque[AsyncModelResult] = deque()
        self.closed = False
        self.close_calls = 0

    def submit(self, request: AsyncModelRequest) -> None:
        if self.closed:
            raise RuntimeError("worker closed")
        self.submitted.append(request)

    def poll(self, *, include_stale: bool = False, **kwargs: object):
        del include_stale, kwargs
        return None if not self.results else self.results.popleft()

    def finish_next(
        self,
        *,
        usage: dict[str, int] | None = None,
        model: str = "server-model",
    ) -> AsyncModelResult:
        request = self.submitted.popleft()
        result = AsyncModelResult(
            request_id=request.request_id,
            review_id=request.review_id,
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            plan_version=request.plan_version,
            observation_timestamp_s=request.observation_timestamp_s,
            frame_id=request.frame_id,
            response=ModelResponse(
                content="{}",
                model=model,
                finish_reason="stop",
                usage={} if usage is None else usage,
            ),
            error_code=None,
            error_message=None,
        )
        self.results.append(result)
        return result

    def fail_next(self) -> AsyncModelResult:
        request = self.submitted.popleft()
        result = AsyncModelResult(
            request_id=request.request_id,
            review_id=request.review_id,
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            plan_version=request.plan_version,
            observation_timestamp_s=request.observation_timestamp_s,
            frame_id=request.frame_id,
            response=None,
            error_code="MODEL_REQUEST_FAILED",
            error_message="ModelHTTPError",
        )
        self.results.append(result)
        return result

    def close(self, timeout_s: float | None = None) -> None:
        del timeout_s
        self.close_calls += 1
        self.closed = True
        while self.submitted:
            self.finish_next()


def _selection() -> AdapterSelection:
    return AdapterSelection(
        call_role=ModelCallRole.RUNTIME_VISUAL_REVIEW,
        requested_adapter="runtime_visual",
        adapter_status=AdapterStatus.PLACEHOLDER,
        effective_model="Qwen3-VL-4B-Instruct",
        fallback_used=True,
    )


def _request(
    request_id: str,
    uav_id: str,
    *,
    priority: int,
    timestamp_s: float = 1.0,
) -> AsyncModelRequest:
    return AsyncModelRequest(
        request_id=request_id,
        review_id=f"review_{request_id}",
        mission_id="mission_dispatcher",
        uav_id=uav_id,
        plan_version=1,
        observation_timestamp_s=timestamp_s,
        frame_id=f"frame_{request_id}",
        messages=(ChatMessage("user", "inspect"),),
        broker_priority=priority,
        broker_replaceable=True,
    )


def _dispatcher(
    uav_ids: tuple[str, ...],
    *,
    max_inflight_global: int = 1,
    record_logger=None,
) -> tuple[
    ModelRequestDispatcher,
    GlobalModelRequestBroker,
    dict[str, _Worker],
    _Clock,
]:
    clock = _Clock()
    broker = GlobalModelRequestBroker(
        max_inflight_global=max_inflight_global,
        max_inflight_per_uav=1,
        max_pending_per_uav=3,
        clock=clock,
    )
    workers = {uav_id: _Worker(uav_id) for uav_id in uav_ids}
    dispatcher = ModelRequestDispatcher(
        broker,
        workers,
        adapter_selection=_selection(),
        clock=clock,
        record_logger=record_logger,
    )
    return dispatcher, broker, workers, clock


def test_global_slot_priority_and_completion_metadata_are_broker_owned() -> None:
    dispatcher, broker, workers, clock = _dispatcher(("uav_a", "uav_b"))
    facade_a = dispatcher.worker_for("uav_a", assignment_id="assignment_a")
    facade_b = dispatcher.worker_for("uav_b", assignment_id="assignment_b")
    periodic = _request("request_periodic", "uav_a", priority=4)
    runtime = _request("request_runtime", "uav_b", priority=3)

    facade_a.submit(periodic)
    facade_b.submit(runtime)
    assert [item.request_id for item in workers["uav_a"].submitted] == [
        periodic.request_id
    ]
    assert workers["uav_b"].submitted == deque()
    assert broker.inflight_count == 1
    assert broker.pending_count == 1

    clock.now = 102.5
    workers["uav_a"].finish_next(
        usage={"prompt_tokens": 17, "completion_tokens": 4}
    )
    result = facade_a.poll(expected_request_id=periodic.request_id)

    assert result is not None and result.succeeded and not result.stale
    # Completing P4 opens the one global slot and the queued P3 request is
    # dispatched through the Broker before poll returns.
    assert [item.request_id for item in workers["uav_b"].submitted] == [
        runtime.request_id
    ]
    record = next(item for item in broker.logs if item.request_id == periodic.request_id)
    assert record.assignment_id == "assignment_a"
    assert record.priority == "P4_PERIODIC_REVIEW"
    assert record.requested_adapter == "runtime_visual"
    assert record.adapter_status == "placeholder"
    assert record.effective_model == "Qwen3-VL-4B-Instruct"
    assert record.fallback_used is True
    assert record.prompt_tokens == 17
    assert record.completion_tokens == 4
    assert record.finish_reason == "stop"
    assert record.latency_s == pytest.approx(2.5)

    dispatcher.close()
    assert all(worker.close_calls == 1 for worker in workers.values())


def test_new_periodic_frame_stales_pending_request_before_worker_boundary() -> None:
    dispatcher, broker, workers, _ = _dispatcher(("uav_a", "uav_blocker"))
    blocker = dispatcher.worker_for("uav_blocker")
    visual = dispatcher.worker_for("uav_a", assignment_id="assignment_a")
    blocker.submit(_request("request_blocker", "uav_blocker", priority=3))
    old = _request("request_old_frame", "uav_a", priority=4, timestamp_s=1.0)
    new = _request("request_new_frame", "uav_a", priority=4, timestamp_s=2.0)

    visual.submit(old)
    visual.submit(new)

    stale = visual.poll(include_stale=True)
    assert stale is not None
    assert stale.request_id == old.request_id
    assert stale.stale is True
    assert stale.error_code == "BROKER_STALE"
    assert stale.error_message == "SUPERSEDED_BY_NEWER_FRAME"
    assert workers["uav_a"].submitted == deque()
    assert broker.logs[-1].request_id == old.request_id
    assert broker.logs[-1].state == "STALE"

    workers["uav_blocker"].finish_next()
    assert blocker.poll() is not None
    assert [item.request_id for item in workers["uav_a"].submitted] == [new.request_id]
    dispatcher.close()


def test_model_failure_completes_broker_with_error_and_routing_metadata() -> None:
    dispatcher, broker, workers, clock = _dispatcher(("uav_a",))
    facade = dispatcher.worker_for("uav_a", assignment_id="assignment_a")
    request = _request("request_failure", "uav_a", priority=3)
    facade.submit(request)
    clock.now = 103.0
    workers["uav_a"].fail_next()

    result = facade.poll(expected_request_id=request.request_id)

    assert result is not None
    assert result.error_code == "MODEL_REQUEST_FAILED"
    assert result.response is None
    record = broker.logs[-1]
    assert record.request_id == request.request_id
    assert record.state == "FAILED"
    assert record.error_code == "MODEL_REQUEST_FAILED"
    assert record.assignment_id == "assignment_a"
    assert record.adapter_status == "placeholder"
    assert record.effective_model == "Qwen3-VL-4B-Instruct"
    assert record.latency_s == pytest.approx(3.0)
    dispatcher.close()


def test_preempted_inflight_late_worker_result_remains_stale() -> None:
    logged: list[dict[str, object]] = []
    dispatcher, broker, workers, clock = _dispatcher(
        ("uav_a", "uav_b"),
        record_logger=lambda value: logged.append(dict(value)),
    )
    visual = dispatcher.worker_for("uav_a", assignment_id="assignment_a")
    request = _request("request_visual", "uav_a", priority=4)
    visual.submit(request)
    assert broker.inflight_count == 1

    clock.now = 101.0
    urgent = ModelBrokerRequest(
        request_id="request_urgent",
        call_role="FLEET_REPLAN",
        priority=ModelRequestPriority.P1_FLEET_REPLAN,
        uav_id="uav_b",
        assignment_id="assignment_b",
        submitted_at_s=clock.now,
    )
    broker.submit(urgent)
    assert broker.acquire_next() == urgent
    broker.complete(urgent.request_id, effective_model="trusted-replanner")

    workers["uav_a"].finish_next(usage={"prompt_tokens": 99})
    late = visual.poll(include_stale=True)

    assert late is not None
    assert late.request_id == request.request_id
    assert late.stale is True
    records = [item for item in broker.logs if item.request_id == request.request_id]
    assert len(records) == 1
    assert records[0].state == "STALE"
    assert records[0].stale_reason == "PREEMPTED_BY_HIGHER_PRIORITY"
    visual_logs = [item for item in logged if item["call_id"] == request.request_id]
    assert len(visual_logs) == 1
    assert visual_logs[0]["state"] == "STALE"
    # The late completion cannot emit another callback or resurrect usage.
    assert visual_logs[0]["prompt_tokens"] == 0
    dispatcher.close()


def test_visual_priority_metadata_is_restricted_to_p3_and_p4() -> None:
    base = _request("request_valid", "uav_a", priority=3)
    assert base.broker_priority == 3
    assert base.broker_replaceable is True

    values = {
        name: getattr(base, name) for name in base.__dataclass_fields__
    }
    for invalid in (0, 1, 2, 5, True, "3"):
        with pytest.raises((TypeError, ValueError)):
            AsyncModelRequest(**(values | {"broker_priority": invalid}))
    with pytest.raises(TypeError, match="broker_replaceable"):
        AsyncModelRequest(**(values | {"broker_replaceable": 1}))


def test_concurrent_facade_submit_respects_limits_and_shared_close_owns_workers() -> None:
    uav_ids = ("uav_a", "uav_b", "uav_c", "uav_d")
    dispatcher, broker, workers, _ = _dispatcher(
        uav_ids,
        max_inflight_global=2,
    )
    facades = {uav_id: dispatcher.worker_for(uav_id) for uav_id in uav_ids}
    barrier = threading.Barrier(len(uav_ids))
    errors: list[BaseException] = []

    def submit(uav_id: str) -> None:
        try:
            barrier.wait(timeout=2.0)
            facades[uav_id].submit(
                _request(f"request_{uav_id}", uav_id, priority=4)
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(uav_id,)) for uav_id in uav_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2.0)

    assert errors == []
    assert broker.inflight_count == 2
    assert broker.pending_count == 2
    assert sum(len(worker.submitted) for worker in workers.values()) == 2

    dispatcher.close()
    assert broker.inflight_count == 0
    assert broker.pending_count == 0
    assert all(worker.closed and worker.close_calls == 1 for worker in workers.values())
    assert all(not facade.is_busy for facade in facades.values())


def test_record_logger_receives_each_terminal_state_exactly_once() -> None:
    records: list[dict[str, object]] = []
    dispatcher, broker, workers, _ = _dispatcher(
        ("uav_a", "uav_b"),
        record_logger=lambda value: records.append(dict(value)),
    )
    facade_a = dispatcher.worker_for("uav_a", assignment_id="assignment_a")
    facade_b = dispatcher.worker_for("uav_b", assignment_id="assignment_b")

    completed = _request("request_completed", "uav_a", priority=3)
    facade_a.submit(completed)
    workers["uav_a"].finish_next(
        usage={"prompt_tokens": 5, "completion_tokens": 2}
    )
    assert facade_a.poll() is not None

    failed = _request("request_failed", "uav_a", priority=3)
    facade_a.submit(failed)
    workers["uav_a"].fail_next()
    assert facade_a.poll() is not None

    blocker = _request("request_blocker_callback", "uav_b", priority=3)
    facade_b.submit(blocker)
    stale = _request("request_stale_callback", "uav_a", priority=4)
    replacement = _request("request_replacement_callback", "uav_a", priority=4)
    facade_a.submit(stale)
    facade_a.submit(replacement)
    assert facade_a.poll(include_stale=True) is not None

    # Repeated service/poll calls and close must never duplicate a terminal
    # callback already emitted by completion or pending reconciliation.
    assert facade_a.poll(include_stale=True) is None
    dispatcher.close()

    by_id = {record["call_id"]: record for record in records}
    assert len(records) == len(by_id)
    assert by_id[completed.request_id]["state"] == "COMPLETED"
    assert by_id[completed.request_id]["prompt_tokens"] == 5
    assert by_id[completed.request_id]["completion_tokens"] == 2
    assert by_id[completed.request_id]["effective_model"] == (
        "Qwen3-VL-4B-Instruct"
    )
    assert by_id[failed.request_id]["state"] == "FAILED"
    assert by_id[failed.request_id]["error_code"] == "MODEL_REQUEST_FAILED"
    assert by_id[stale.request_id]["state"] == "STALE"
    assert by_id[stale.request_id]["stale_reasons"] == [
        "SUPERSEDED_BY_NEWER_FRAME"
    ]
    assert set(by_id) == {record.request_id for record in broker.logs}
