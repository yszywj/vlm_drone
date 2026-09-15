from __future__ import annotations

from dataclasses import replace

import pytest

from fleet.model_request_broker import (
    GlobalModelRequestBroker,
    ModelBrokerRequest,
    ModelRequestBrokerError,
    ModelRequestPriority,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _request(
    role: str,
    priority: ModelRequestPriority,
    uav_id: str,
    **kwargs: object,
) -> ModelBrokerRequest:
    return ModelBrokerRequest(
        call_role=role,
        priority=priority,
        uav_id=uav_id,
        submitted_at_s=100.0,
        **kwargs,
    )


def test_urgent_uav_b_replan_precedes_uav_a_periodic_review() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(clock=clock)
    periodic = _request(
        "RUNTIME_VISUAL_REVIEW",
        ModelRequestPriority.P4_PERIODIC_REVIEW,
        "uav_a",
        replaceable=True,
        control_related=False,
    )
    urgent = _request(
        "FLEET_REPLAN",
        ModelRequestPriority.P1_FLEET_REPLAN,
        "uav_b",
    )
    broker.submit(periodic)
    broker.submit(urgent)
    assert broker.acquire_next() == urgent


def test_all_five_priority_classes_are_dispatched_in_strict_order() -> None:
    broker = GlobalModelRequestBroker(max_inflight_global=5, clock=_Clock())
    requests = (
        _request(
            "RUNTIME_VISUAL_REVIEW",
            ModelRequestPriority.P4_PERIODIC_REVIEW,
            "uav_p4",
            replaceable=True,
            control_related=False,
        ),
        _request(
            "RUNTIME_REPLAN",
            ModelRequestPriority.P2_AGENT_RUNTIME_REPLAN,
            "uav_p2",
        ),
        _request(
            "AIRSPACE_CONFLICT",
            ModelRequestPriority.P0_IMMINENT_COLLISION,
            "uav_p0",
        ),
        _request(
            "RUNTIME_VISUAL_REVIEW",
            ModelRequestPriority.P3_RUNTIME_VISUAL_REVIEW,
            "uav_p3",
            replaceable=True,
            control_related=False,
        ),
        _request(
            "FLEET_REPLAN",
            ModelRequestPriority.P1_FLEET_REPLAN,
            "uav_p1",
        ),
    )
    for request in requests:
        broker.submit(request)

    acquired = tuple(broker.acquire_next() for _ in requests)

    assert [request.priority for request in acquired if request is not None] == list(
        ModelRequestPriority
    )


def test_visual_starvation_aging_never_crosses_replanning_safety_boundary() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(
        max_inflight_global=2,
        starvation_timeout_s=15.0,
        clock=clock,
    )
    old_periodic = _request(
        "RUNTIME_VISUAL_REVIEW",
        ModelRequestPriority.P4_PERIODIC_REVIEW,
        "uav_visual",
        replaceable=True,
        control_related=False,
    )
    broker.submit(old_periodic)
    clock.now = 200.0
    imminent = replace(
        _request(
            "AIRSPACE_CONFLICT",
            ModelRequestPriority.P0_IMMINENT_COLLISION,
            "uav_emergency",
        ),
        submitted_at_s=clock.now,
    )
    broker.submit(imminent)

    # Even after many starvation intervals, visual work cannot overtake P0.
    assert broker.acquire_next() == imminent
    assert broker.acquire_next() == old_periodic

    fair = GlobalModelRequestBroker(
        max_inflight_global=2,
        starvation_timeout_s=15.0,
        clock=clock,
    )
    old_periodic = replace(old_periodic, request_id="request_old_periodic")
    fair.submit(old_periodic)
    runtime_review = replace(
        _request(
            "RUNTIME_VISUAL_REVIEW",
            ModelRequestPriority.P3_RUNTIME_VISUAL_REVIEW,
            "uav_runtime",
            replaceable=True,
            control_related=False,
        ),
        request_id="request_new_runtime",
        submitted_at_s=clock.now,
    )
    fair.submit(runtime_review)

    # Aging still prevents P4 starvation inside the visual-review band.
    assert fair.acquire_next() == old_periodic


def test_boolean_priority_is_rejected_instead_of_becoming_p1() -> None:
    with pytest.raises(TypeError, match="priority"):
        ModelBrokerRequest(
            call_role="FLEET_REPLAN",
            priority=True,  # type: ignore[arg-type]
            uav_id="uav_a",
        )


def test_same_uav_is_bounded_but_another_uav_can_run() -> None:
    broker = GlobalModelRequestBroker(max_inflight_global=4, clock=_Clock())
    first = _request("RUNTIME_REPLAN", ModelRequestPriority.P2_AGENT_RUNTIME_REPLAN, "uav_a")
    second = _request("FLEET_REPLAN", ModelRequestPriority.P1_FLEET_REPLAN, "uav_a")
    other = _request("FLEET_REPLAN", ModelRequestPriority.P1_FLEET_REPLAN, "uav_b")
    for item in (first, second, other):
        broker.submit(item)
    acquired = (broker.acquire_next(), broker.acquire_next())
    assert {item.request_id for item in acquired if item is not None} == {
        second.request_id,
        other.request_id,
    }
    # The lower-priority request for uav_a stays queued while uav_a already
    # owns its single per-UAV inflight slot.
    assert broker.acquire_next() is None


def test_new_periodic_frame_replaces_old_pending_frame() -> None:
    broker = GlobalModelRequestBroker(clock=_Clock())
    old = _request(
        "RUNTIME_VISUAL_REVIEW",
        ModelRequestPriority.P4_PERIODIC_REVIEW,
        "uav_a",
        replaceable=True,
        control_related=False,
    )
    new = replace(old, request_id="request_new", submitted_at_s=101.0)
    broker.submit(old)
    broker.submit(new)
    assert broker.pending_count == 1
    assert broker.acquire_next() == new
    assert broker.logs[-1].stale_reason == "SUPERSEDED_BY_NEWER_FRAME"


def test_normal_review_cannot_overwrite_urgent_request() -> None:
    broker = GlobalModelRequestBroker(max_pending_per_uav=1, clock=_Clock())
    urgent = _request("FLEET_REPLAN", ModelRequestPriority.P1_FLEET_REPLAN, "uav_a")
    normal = _request(
        "RUNTIME_VISUAL_REVIEW",
        ModelRequestPriority.P4_PERIODIC_REVIEW,
        "uav_a",
        replaceable=True,
        control_related=False,
    )
    broker.submit(urgent)
    with pytest.raises(ModelRequestBrokerError, match="queue is full"):
        broker.submit(normal)


def test_less_urgent_refresh_cannot_supersede_urgent_same_stream() -> None:
    broker = GlobalModelRequestBroker(clock=_Clock())
    urgent = _request(
        "RUNTIME_VISUAL_REVIEW",
        ModelRequestPriority.P1_FLEET_REPLAN,
        "uav_a",
        replaceable=True,
        control_related=False,
    )
    normal = replace(
        urgent,
        request_id="request_normal_refresh",
        priority=ModelRequestPriority.P4_PERIODIC_REVIEW,
        submitted_at_s=101.0,
    )

    broker.submit(urgent)
    broker.submit(normal)

    assert broker.pending_count == 2
    assert broker.logs == ()
    assert broker.acquire_next() == urgent


def test_request_id_cannot_be_reused_after_completion() -> None:
    broker = GlobalModelRequestBroker(clock=_Clock())
    request = _request(
        "FLEET_REPLAN",
        ModelRequestPriority.P1_FLEET_REPLAN,
        "uav_a",
    )
    broker.submit(request)
    assert broker.acquire_next() == request
    broker.complete(request.request_id, effective_model="base")

    with pytest.raises(ModelRequestBrokerError, match="already exists"):
        broker.submit(request)


def test_urgent_request_revokes_replaceable_result_but_waits_for_real_global_slot() -> None:
    clock = _Clock()
    broker = GlobalModelRequestBroker(max_inflight_global=1, clock=clock)
    periodic = _request(
        "RUNTIME_VISUAL_REVIEW",
        ModelRequestPriority.P4_PERIODIC_REVIEW,
        "uav_a",
        replaceable=True,
        control_related=False,
    )
    urgent = _request(
        "FLEET_REPLAN",
        ModelRequestPriority.P1_FLEET_REPLAN,
        "uav_b",
    )
    broker.submit(periodic)
    assert broker.acquire_next() == periodic
    clock.now = 102.5

    broker.submit(urgent)

    assert broker.acquire_next() is None
    assert broker.inflight_count == 1
    assert broker.pending_count == 1
    stale = next(
        record for record in broker.logs if record.request_id == periodic.request_id
    )
    assert stale.state == "STALE"
    assert stale.stale_reason == "PREEMPTED_BY_HIGHER_PRIORITY"
    assert stale.latency_s == pytest.approx(2.5)
    # A late HTTP result is acknowledged as stale and cannot become COMPLETED.
    assert broker.complete(periodic.request_id) == stale
    assert broker.acquire_next() == urgent


def test_same_uav_urgent_request_waits_for_revoked_visual_call_to_finish() -> None:
    broker = GlobalModelRequestBroker(max_inflight_global=4, clock=_Clock())
    periodic = _request(
        "RUNTIME_VISUAL_REVIEW",
        ModelRequestPriority.P4_PERIODIC_REVIEW,
        "uav_a",
        replaceable=True,
        control_related=False,
    )
    urgent = _request(
        "FLEET_REPLAN",
        ModelRequestPriority.P1_FLEET_REPLAN,
        "uav_a",
    )
    broker.submit(periodic)
    assert broker.acquire_next() == periodic

    broker.submit(urgent)

    assert broker.acquire_next() is None
    assert broker.inflight_count == 1
    assert broker.logs[-1].stale_reason == "PREEMPTED_BY_HIGHER_PRIORITY"
    broker.complete(periodic.request_id)
    assert broker.acquire_next() == urgent


def test_urgent_request_never_preempts_non_replaceable_inflight_work() -> None:
    broker = GlobalModelRequestBroker(max_inflight_global=1, clock=_Clock())
    protected = _request(
        "RUNTIME_VISUAL_REVIEW",
        ModelRequestPriority.P4_PERIODIC_REVIEW,
        "uav_a",
        replaceable=False,
        control_related=False,
    )
    urgent = _request(
        "FLEET_REPLAN",
        ModelRequestPriority.P1_FLEET_REPLAN,
        "uav_b",
    )
    broker.submit(protected)
    assert broker.acquire_next() == protected

    broker.submit(urgent)

    assert broker.acquire_next() is None
    assert broker.inflight_count == 1
    assert broker.pending_count == 1
    assert broker.logs == ()
    broker.complete(protected.request_id)
    assert broker.acquire_next() == urgent


def test_owner_acquisition_cannot_consume_or_bypass_another_owners_priority() -> None:
    broker = GlobalModelRequestBroker(clock=_Clock())
    visual = _request("RUNTIME_VISUAL_REVIEW", ModelRequestPriority.P4_PERIODIC_REVIEW, "uav_a")
    text = _request("RUNTIME_REPLAN", ModelRequestPriority.P2_AGENT_RUNTIME_REPLAN, "uav_b")
    broker.submit(visual)
    broker.submit(text)
    assert broker.acquire_next(request_ids={visual.request_id}) is None
    assert broker.pending_count == 2
    assert broker.inflight_count == 0
    assert broker.acquire_next(request_ids={text.request_id}) == text
    assert broker.acquire_next(request_ids={visual.request_id}) == visual


def test_cancel_inflight_is_idempotent_and_keeps_real_lease_until_completion() -> None:
    broker = GlobalModelRequestBroker(max_inflight_global=1, clock=_Clock())
    first = _request("RUNTIME_REPLAN", ModelRequestPriority.P2_AGENT_RUNTIME_REPLAN, "uav_a")
    second = _request("FLEET_REPLAN", ModelRequestPriority.P1_FLEET_REPLAN, "uav_b")
    broker.submit(first)
    assert broker.acquire_next() == first
    record = broker.cancel_inflight(first.request_id, reason="DEADLINE_EXCEEDED")
    assert broker.cancel_inflight(first.request_id, reason="CANCELED") is record
    broker.submit(second)
    assert broker.acquire_next() is None
    assert broker.inflight_count == 1
    assert broker.complete(first.request_id) is record
    assert broker.acquire_next() == second
    assert len(broker.logs) == 1


def test_one_urgent_request_does_not_revoke_every_occupied_visual_call() -> None:
    broker = GlobalModelRequestBroker(max_inflight_global=2, clock=_Clock())
    visual_a = _request("RUNTIME_VISUAL_REVIEW", ModelRequestPriority.P4_PERIODIC_REVIEW,
                        "uav_a", replaceable=True, control_related=False)
    visual_b = _request("RUNTIME_VISUAL_REVIEW", ModelRequestPriority.P4_PERIODIC_REVIEW,
                        "uav_b", replaceable=True, control_related=False)
    broker.submit(visual_a)
    broker.submit(visual_b)
    assert broker.acquire_next() is not None
    assert broker.acquire_next() is not None
    urgent = _request("FLEET_REPLAN", ModelRequestPriority.P1_FLEET_REPLAN, "uav_c")
    broker.submit(urgent)
    assert broker.inflight_count == 2
    assert broker.acquire_next() is None
    assert len(broker.logs) == 1
    revoked_id = broker.logs[0].request_id
    broker.complete(revoked_id)
    assert broker.acquire_next() == urgent
