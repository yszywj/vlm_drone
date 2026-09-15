from __future__ import annotations

import pytest

from fleet.target_registry import SharedTargetRegistry, TargetClaimError


def _bound_registry():
    registry = SharedTargetRegistry()
    registry.bind_assignment(
        assignment_id="assignment_1", uav_id="uav_1",
        target_runtime_id="target_1", semantic_alias="target one",
    )
    return registry


def test_failing_candidate_bind_cannot_pollute_live_registry():
    live = _bound_registry()
    before = live.snapshot()
    candidate = live.clone_for_staging()
    # bind_assignment registers a new target before detecting duplicate UAV;
    # the partial mutation must stay solely in the staging object.
    with pytest.raises(TargetClaimError, match="another active assignment"):
        candidate.bind_assignment(
            assignment_id="assignment_2", uav_id="uav_1",
            target_runtime_id="target_2", semantic_alias="target two",
        )
    assert candidate.snapshot() != before
    assert live.snapshot() == before
    assert live.assigned_target(assignment_id="assignment_1", uav_id="uav_1") == "target_1"


def test_adopt_swaps_validated_state_and_preserves_live_identity():
    live = _bound_registry()
    reader_reference = live
    candidate = live.clone_for_staging()
    assert candidate._lock is not live._lock
    candidate.bind_assignment(
        assignment_id="assignment_2", uav_id="uav_2",
        target_runtime_id="target_2", semantic_alias="target two",
    )
    expected = candidate.snapshot()
    live.adopt_staged(candidate)
    assert reader_reference is live
    assert reader_reference.snapshot() == expected
    assert live.assigned_target(assignment_id="assignment_2", uav_id="uav_2") == "target_2"
    assert live.assigned_target(assignment_id="assignment_1", uav_id="uav_1") == "target_1"
    # The consumed object's old containers cannot mutate the published state.
    candidate.register_target("target_3", "target three")
    assert live.snapshot() == expected
    with pytest.raises(TargetClaimError, match="unused candidate"):
        live.adopt_staged(candidate)
    assert live.snapshot() == expected


def test_wrong_owner_and_policy_are_rejected_before_publication():
    live = _bound_registry()
    before = live.snapshot()
    stranger = _bound_registry().clone_for_staging()
    with pytest.raises(TargetClaimError, match="unused candidate"):
        live.adopt_staged(stranger)
    shared = SharedTargetRegistry("SHARED").clone_for_staging()
    with pytest.raises(TargetClaimError, match="policy mismatch"):
        live.adopt_staged(shared)
    assert live.snapshot() == before
