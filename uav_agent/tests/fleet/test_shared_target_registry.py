from __future__ import annotations

import pytest

from fleet.target_registry import (
    SharedTargetRegistry,
    TargetClaimError,
    TargetClaimState,
)


def test_assignment_claims_are_target_isolated() -> None:
    registry = SharedTargetRegistry("EXCLUSIVE")
    registry.bind_assignment(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="target_i",
        semantic_alias="目标i",
        priority=100,
    )
    registry.bind_assignment(
        assignment_id="assignment_b_j",
        uav_id="uav_b",
        target_runtime_id="target_j",
        semantic_alias="目标j",
        priority=90,
    )

    with pytest.raises(TargetClaimError, match="trusted assignment"):
        registry.claim(
            assignment_id="assignment_a_i",
            uav_id="uav_a",
            target_runtime_id="target_j",
            confidence=1.0,
            timestamp_s=1.0,
        )
    with pytest.raises(TargetClaimError, match="outside its assignment"):
        registry.validate_oracle_binding(
            assignment_id="assignment_a_i",
            uav_id="uav_a",
            assigned_target_id="target_j",
        )


def test_exclusive_conflict_selects_priority_winner_and_holds_loser() -> None:
    registry = SharedTargetRegistry("EXCLUSIVE")
    registry.bind_assignment(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="target_i",
        semantic_alias="目标i",
        priority=100,
    )
    registry.bind_assignment(
        assignment_id="assignment_b_i",
        uav_id="uav_b",
        target_runtime_id="target_i",
        semantic_alias="目标i",
        priority=10,
    )

    winner = registry.claim(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="target_i",
        confidence=0.9,
        timestamp_s=2.0,
        state=TargetClaimState.EXCLUSIVE,
    )
    loser = registry.claim(
        assignment_id="assignment_b_i",
        uav_id="uav_b",
        target_runtime_id="target_i",
        confidence=0.95,
        timestamp_s=2.1,
        state=TargetClaimState.EXCLUSIVE,
    )

    assert winner.accepted
    assert loser.conflict
    assert not loser.accepted
    assert loser.winner is not None
    assert loser.winner.uav_id == "uav_a"
    assert loser.hold_uav_id == "uav_b"
    record = registry.record("target_i")
    assert [claim.uav_id for claim in record.active_claims] == ["uav_a"]
    assert next(
        claim for claim in record.claims if claim.uav_id == "uav_b"
    ).state is TargetClaimState.RELEASED


def test_exclusive_winner_releases_every_other_provisional_claim() -> None:
    registry = SharedTargetRegistry("EXCLUSIVE")
    for uav_id, priority in (("uav_a", 20), ("uav_b", 10), ("uav_c", 100)):
        registry.bind_assignment(
            assignment_id=f"assignment_{uav_id}_i",
            uav_id=uav_id,
            target_runtime_id="target_i",
            semantic_alias="目标i",
            priority=priority,
        )

    decision = registry.claim(
        assignment_id="assignment_uav_c_i",
        uav_id="uav_c",
        target_runtime_id="target_i",
        confidence=0.9,
        timestamp_s=2.0,
        state=TargetClaimState.EXCLUSIVE,
    )

    assert decision.accepted
    record = registry.record("target_i")
    assert [claim.uav_id for claim in record.active_claims] == ["uav_c"]
    assert {
        claim.uav_id: claim.state for claim in record.claims
    } == {
        "uav_a": TargetClaimState.RELEASED,
        "uav_b": TargetClaimState.RELEASED,
        "uav_c": TargetClaimState.EXCLUSIVE,
    }


def test_terminated_claim_is_monotonic_and_cannot_be_reactivated() -> None:
    registry = SharedTargetRegistry("EXCLUSIVE")
    registry.bind_assignment(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="target_i",
        semantic_alias="目标i",
    )
    registry.claim(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="target_i",
        confidence=0.9,
        timestamp_s=5.0,
    )
    terminal = registry.terminate("assignment_a_i", timestamp_s=4.0)
    assert terminal.timestamp_s == 5.0

    with pytest.raises(TargetClaimError, match="cannot be reactivated"):
        registry.claim(
            assignment_id="assignment_a_i",
            uav_id="uav_a",
            target_runtime_id="target_i",
            confidence=1.0,
            timestamp_s=6.0,
        )


def test_one_uav_cannot_hold_two_active_assignments() -> None:
    registry = SharedTargetRegistry()
    registry.bind_assignment(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="target_i",
        semantic_alias="目标i",
    )
    with pytest.raises(TargetClaimError, match="another active assignment"):
        registry.bind_assignment(
            assignment_id="assignment_a_j",
            uav_id="uav_a",
            target_runtime_id="target_j",
            semantic_alias="目标j",
        )


def test_records_are_read_only_and_deterministically_ordered() -> None:
    registry = SharedTargetRegistry()
    registry.bind_assignment(
        assignment_id="assignment_b_j",
        uav_id="uav_b",
        target_runtime_id="target_j",
        semantic_alias="目标j",
    )
    registry.bind_assignment(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="target_i",
        semantic_alias="目标i",
    )

    records = registry.records
    assert isinstance(records, tuple)
    assert [record.target_runtime_id for record in records] == [
        "target_i",
        "target_j",
    ]


def test_event_history_is_bounded_and_terminal_summary_omits_history() -> None:
    registry = SharedTargetRegistry()
    registry.bind_assignment(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="target_i",
        semantic_alias="目标i",
    )
    for index in range(registry.MAX_RETAINED_EVENTS + 17):
        registry.claim(
            assignment_id="assignment_a_i",
            uav_id="uav_a",
            target_runtime_id="target_i",
            confidence=0.9,
            timestamp_s=float(index + 1),
        )

    snapshot = registry.snapshot()
    summary = registry.summary_snapshot()
    assert snapshot["event_count"] == registry.MAX_RETAINED_EVENTS + 18
    assert len(snapshot["events"]) == registry.MAX_RETAINED_EVENTS
    assert summary["event_count"] == snapshot["event_count"]
    assert "events" not in summary
