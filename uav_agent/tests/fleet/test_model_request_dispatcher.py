from __future__ import annotations

from collections import deque
import threading

import pytest

from fleet.model_request_broker import (
    GlobalModelRequestBroker,
    ModelBrokerRequest,
    ModelRequestPriority,
)
from fleet.model_request_dispatcher import (
    BrokeredTextTaskRunner, ModelRequestDispatcher, ModelRequestDispatcherError,
)
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

    assert visual.poll(expected_request_id=new.request_id) is None
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
    assert broker.acquire_next() is None
    assert broker.inflight_count == 1

    workers["uav_a"].finish_next(usage={"prompt_tokens": 99})
    # The visual dispatcher collects its late result without acquiring text
    # work owned by another dispatcher, then the real slot can be reused.
    late = visual.poll(include_stale=True)
    assert broker.acquire_next() == urgent
    broker.complete(urgent.request_id, effective_model="trusted-replanner")

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


def _text_request(request_id: str, uav_id: str = "uav_text", **kwargs: object) -> ModelBrokerRequest:
    return ModelBrokerRequest(
        request_id=request_id,
        call_role=ModelCallRole.RUNTIME_REPLAN,
        priority=ModelRequestPriority.P2_AGENT_RUNTIME_REPLAN,
        uav_id=uav_id,
        submitted_at_s=100.0,
        **kwargs,
    )


def _join_text_thread(runner: BrokeredTextTaskRunner, request_id: str) -> None:
    # Explicit synchronization in tests only: production poll never joins.
    thread = runner._active[request_id]
    assert thread is not None
    thread.join(2.0)
    assert not thread.is_alive()


def test_dispatcher_filtered_poll_preserves_other_completed_requests() -> None:
    dispatcher, _, workers, _ = _dispatcher(("uav_a",))
    facade = dispatcher.worker_for("uav_a")
    first = _request("request_first_owned", "uav_a", priority=3)
    second = _request("request_second_owned", "uav_a", priority=3)
    facade.submit(first)
    workers["uav_a"].finish_next()
    assert facade.poll(expected_request_id=second.request_id, include_stale=True) is None
    facade.submit(second)
    workers["uav_a"].finish_next()
    assert facade.poll(expected_request_id=second.request_id).request_id == second.request_id
    assert facade.poll(expected_request_id=first.request_id).request_id == first.request_id
    assert facade.discarded_result_count == 0
    dispatcher.close()


def test_text_admission_refreshes_snapshot_on_owner_only_when_real_slot_is_ready() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(max_inflight_global=1, clock=clock)
    blocker = _text_request("request_blocker", "uav_other")
    broker.submit(blocker)
    assert broker.acquire_next() == blocker
    runner = BrokeredTextTaskRunner(broker, clock=clock)
    state = {"version": 1}
    preparations: list[tuple[int, int]] = []
    owner = threading.get_ident()

    def prepare():
        version = state["version"]
        preparations.append((version, threading.get_ident()))
        return lambda: (version, threading.get_ident())

    request = _text_request("request_refresh")
    runner.submit(request, prepare)
    runner.pump()
    assert preparations == []
    state["version"] = 2
    broker.complete(blocker.request_id)
    runner.pump()
    assert preparations == [(2, owner)]
    _join_text_thread(runner, request.request_id)
    result = runner.poll(request.request_id)
    assert result.succeeded
    assert result.value[0] == 2
    assert result.value[1] != owner
    runner.close()


def test_cancel_active_text_revokes_result_but_preserves_underlying_capacity() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(max_inflight_global=1, clock=clock)
    runner = BrokeredTextTaskRunner(broker, clock=clock)
    started, release, second_started = threading.Event(), threading.Event(), threading.Event()

    def slow():
        started.set()
        assert release.wait(2.0)
        return "late value"

    first = _text_request("request_canceled", "uav_a")
    second = _text_request("request_waits", "uav_b")
    runner.submit(first, lambda: slow)
    runner.pump()
    assert started.wait(2.0)
    runner.submit(second, lambda: lambda: second_started.set())
    try:
        assert runner.cancel(first.request_id)
        canceled = runner.poll(first.request_id)
        assert canceled.stale and canceled.reason == "CANCELED"
        assert runner.inflight_count == broker.inflight_count == 1
        assert not second_started.is_set()
        runner.pump()
        assert not second_started.is_set()
    finally:
        release.set()
    _join_text_thread(runner, first.request_id)
    runner.pump()
    assert second_started.wait(2.0)
    _join_text_thread(runner, second.request_id)
    assert runner.poll(first.request_id) is None
    assert runner.poll(second.request_id).succeeded
    assert broker.inflight_count == 0
    runner.close()


def test_text_deadline_uses_injected_wall_clock_and_late_result_cannot_return() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(max_inflight_global=1, clock=clock)
    runner = BrokeredTextTaskRunner(broker, clock=clock)
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        assert release.wait(2.0)
        return "expired candidate"

    request = _text_request("request_deadline")
    runner.submit(request, lambda: slow, deadline_at_s=101.0)
    runner.pump()
    assert started.wait(2.0)
    try:
        clock.now = 101.0
        result = runner.poll(request.request_id)
        assert result.stale and result.reason == "DEADLINE_EXCEEDED"
        assert runner.inflight_count == broker.inflight_count == 1
    finally:
        release.set()
    _join_text_thread(runner, request.request_id)
    assert runner.poll(request.request_id) is None
    assert broker.inflight_count == 0
    assert broker.logs[-1].state == "STALE"
    runner.close()


def test_queued_deadline_and_rejected_snapshot_do_not_invoke_compute() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(clock=clock)
    runner = BrokeredTextTaskRunner(broker, clock=clock)
    prepared: list[str] = []
    expired = _text_request("request_expired")
    rejected = _text_request("request_rejected")
    runner.submit(expired, lambda: prepared.append("expired"), deadline_at_s=100.0)
    runner.submit(rejected, lambda: prepared.append("rejected"))
    assert runner.poll(expired.request_id).reason == "DEADLINE_EXCEEDED"
    assert runner.poll(rejected.request_id).reason == "SNAPSHOT_REJECTED"
    assert prepared == ["rejected"]
    assert broker.inflight_count == 0
    runner.close()


def test_text_out_of_order_results_are_consumed_by_request_and_exceptions_are_private() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(max_inflight_global=2, clock=clock)
    runner = BrokeredTextTaskRunner(broker, clock=clock)
    release = threading.Event()
    error = ValueError("private compute detail")

    def slow():
        assert release.wait(2.0)
        return "first"

    def fail():
        raise error

    first = _text_request("request_slow", "uav_a")
    second = _text_request("request_failure", "uav_b")
    runner.submit(first, lambda: slow)
    runner.submit(second, lambda: fail)
    runner.pump()
    _join_text_thread(runner, second.request_id)
    try:
        assert runner.poll(first.request_id) is None
        result = runner.poll(second.request_id)
        assert result.exception is error and not result.succeeded
        assert "private compute detail" not in str(broker.snapshot())
    finally:
        release.set()
    _join_text_thread(runner, first.request_id)
    assert runner.poll(first.request_id).value == "first"
    runner.close()


def test_text_and_visual_dispatchers_share_limits_without_acquiring_each_others_work() -> None:
    dispatcher, broker, workers, clock = _dispatcher(("uav_a",))
    runner = BrokeredTextTaskRunner(broker, clock=clock)
    facade = dispatcher.worker_for("uav_a")
    visual = _request("request_visual", "uav_a", priority=4)
    facade.submit(visual)
    text = _text_request("request_text")
    runner.submit(text, lambda: lambda: "candidate")
    runner.pump()
    assert broker.inflight_count == 1
    assert runner.inflight_count == 0
    workers["uav_a"].finish_next()
    assert facade.poll(expected_request_id=visual.request_id) is not None
    assert broker.pending_count == 1
    runner.pump()
    _join_text_thread(runner, text.request_id)
    assert runner.poll(text.request_id).value == "candidate"
    assert broker.inflight_count == 0
    runner.close()
    dispatcher.close()


def test_text_runner_rejects_visual_priority_escalation_and_nonowner_admission() -> None:
    broker = GlobalModelRequestBroker(clock=_Clock())
    runner = BrokeredTextTaskRunner(broker, clock=_Clock())
    visual = ModelBrokerRequest(
        request_id="request_no_escalation", call_role=ModelCallRole.RUNTIME_VISUAL_REVIEW,
        priority=ModelRequestPriority.P1_FLEET_REPLAN, uav_id="uav_a",
    )
    with pytest.raises(ValueError, match="role and priority"):
        runner.submit(visual, lambda: lambda: None)
    errors: list[BaseException] = []

    def wrong_thread():
        try:
            runner.pump()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join(2.0)
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "owner thread" in str(errors[0])
    runner.close()


def test_text_runner_shutdown_is_bounded_with_a_frozen_scheduling_clock() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(max_inflight_global=1, clock=clock)
    runner = BrokeredTextTaskRunner(broker, clock=clock)
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        assert release.wait(2.0)
        return "late"

    request = _text_request("request_shutdown")
    runner.submit(request, lambda: slow)
    runner.pump()
    assert started.wait(2.0)
    try:
        # No real waiting at all: this must return with the task still running.
        runner.close(timeout_s=0.0)
        assert not release.is_set()
        assert runner.is_closed
        assert runner.inflight_count == broker.inflight_count == 1
        assert runner.poll(request.request_id).reason == "RUNNER_CLOSED"
        with pytest.raises(RuntimeError, match="closed"):
            runner.submit(_text_request("request_after_close"), lambda: lambda: None)
    finally:
        release.set()
    _join_text_thread(runner, request.request_id)
    runner.close(timeout_s=0.0)
    assert broker.inflight_count == 0
    assert runner.poll(request.request_id) is None


def test_text_runner_bounds_unconsumed_results_without_silently_discarding_them() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(clock=clock)
    runner = BrokeredTextTaskRunner(broker, clock=clock, max_outstanding_tasks=1)
    first = _text_request("request_retained")
    runner.submit(first, lambda: lambda: "retained")
    runner.pump()
    _join_text_thread(runner, first.request_id)
    runner.pump()
    with pytest.raises(ModelRequestDispatcherError, match="capacity"):
        runner.submit(_text_request("request_capacity"), lambda: lambda: None)
    assert runner.poll(first.request_id).value == "retained"
    runner.submit(_text_request("request_capacity"), lambda: None)
    runner.close()
