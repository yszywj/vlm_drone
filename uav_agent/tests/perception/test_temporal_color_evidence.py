from __future__ import annotations

import pytest

from configs.schema import TargetColorAttributeConfig
from perception.attribute_types import (
    AttributeDecision,
    AttributeObservation,
    AttributeRequirement,
)
from perception.attribute_verifier import AttributeRouteMismatch, AttributeTimeError
from perception.color_attribute_verifier import TemporalColorEvidenceAccumulator
from perception.runtime import PerceptionRuntimeProfile


def _requirement(
    *,
    mission_id: str = "mission_a",
    uav_id: str = "uav_a",
    candidate_id: str = "candidate_a",
    tracker_id: int = 7,
) -> AttributeRequirement:
    return AttributeRequirement(
        mission_id=mission_id,
        uav_id=uav_id,
        assignment_id="assignment_a",
        candidate_id=candidate_id,
        tracker_id=tracker_id,
        attribute_name="color",
        expected_value="red",
    )


def _observation(
    timestamp_s: float,
    *,
    decision: AttributeDecision = AttributeDecision.MATCH,
    observed: str | None = "red",
    mission_id: str = "mission_a",
    uav_id: str = "uav_a",
    candidate_id: str = "candidate_a",
    tracker_id: int = 7,
    source: str = "hsv_depth_mask",
    runtime_profile: PerceptionRuntimeProfile = PerceptionRuntimeProfile.PRODUCTION,
) -> AttributeObservation:
    return AttributeObservation(
        mission_id=mission_id,
        uav_id=uav_id,
        assignment_id="assignment_a",
        candidate_id=candidate_id,
        tracker_id=tracker_id,
        timestamp_s=timestamp_s,
        attribute_name="color",
        expected_value="red",
        observed_value=observed,
        decision=decision,
        confidence=0.9 if decision is not AttributeDecision.PENDING else 0.0,
        observation_count=1,
        duration_s=0.0,
        valid_sample_ratio=0.8 if decision is not AttributeDecision.PENDING else 0.0,
        source=source,
        reason_code="single_frame",
        runtime_profile=runtime_profile,
    )


def test_single_frame_never_finalizes_and_three_frames_do() -> None:
    accumulator = TemporalColorEvidenceAccumulator(
        min_observations=3,
        min_duration_s=0.4,
    )
    first = accumulator.update(_observation(1.0))
    second = accumulator.update(_observation(1.2))
    third = accumulator.update(_observation(1.4))
    assert first.decision is AttributeDecision.PENDING
    assert second.decision is AttributeDecision.PENDING
    assert third.decision is AttributeDecision.MATCH
    assert third.observation_count == 3
    assert third.duration_s == pytest.approx(0.4)
    assert third.observed_value == "red"


def test_accumulator_builds_from_public_color_config() -> None:
    accumulator = TemporalColorEvidenceAccumulator.from_config(
        TargetColorAttributeConfig(min_observations=2, min_duration_s=0.1)
    )
    assert accumulator.update(_observation(1.0)).decision is AttributeDecision.PENDING
    assert accumulator.update(_observation(1.1)).decision is AttributeDecision.MATCH


def test_stable_other_color_only_mismatches_after_temporal_thresholds() -> None:
    accumulator = TemporalColorEvidenceAccumulator(
        min_observations=3,
        min_duration_s=0.4,
    )
    results = [
        accumulator.update(
            _observation(
                timestamp,
                decision=AttributeDecision.MISMATCH,
                observed="blue",
            )
        )
        for timestamp in (2.0, 2.2, 2.4)
    ]
    assert results[0].decision is AttributeDecision.PENDING
    assert results[1].decision is AttributeDecision.PENDING
    assert results[2].decision is AttributeDecision.MISMATCH
    assert results[2].observed_value == "blue"


def test_unknown_is_not_negative_evidence() -> None:
    accumulator = TemporalColorEvidenceAccumulator(
        min_observations=3,
        min_duration_s=0.4,
    )
    for timestamp in (1.0, 1.3, 1.7, 2.0):
        result = accumulator.update(
            _observation(
                timestamp,
                decision=AttributeDecision.PENDING,
                observed="unknown",
            )
        )
    assert result.decision is AttributeDecision.PENDING
    assert result.reason_code == "no_decisive_color_evidence"
    assert result.observation_count == 4


def test_timestamps_increase_strictly() -> None:
    accumulator = TemporalColorEvidenceAccumulator()
    accumulator.update(_observation(1.0))
    with pytest.raises(AttributeTimeError, match="increase strictly"):
        accumulator.update(_observation(1.0))
    with pytest.raises(AttributeTimeError, match="increase strictly"):
        accumulator.update(_observation(0.9))


def test_tracker_change_starts_a_fresh_epoch() -> None:
    accumulator = TemporalColorEvidenceAccumulator(
        min_observations=2,
        min_duration_s=0.1,
    )
    accumulator.update(_observation(1.0, tracker_id=7))
    accumulator.update(_observation(1.2, tracker_id=7))
    switched = accumulator.update(_observation(1.3, tracker_id=8))
    assert switched.decision is AttributeDecision.PENDING
    assert switched.observation_count == 1
    assert accumulator.history(_requirement(tracker_id=7)) == ()
    assert len(accumulator.history(_requirement(tracker_id=8))) == 1


def test_candidate_reject_reset_removes_votes_and_timestamp_epoch() -> None:
    accumulator = TemporalColorEvidenceAccumulator(
        min_observations=2,
        min_duration_s=0.1,
    )
    accumulator.update(_observation(1.0))
    accumulator.reject_candidate(
        mission_id="mission_a",
        uav_id="uav_a",
        assignment_id="assignment_a",
        candidate_id="candidate_a",
    )
    assert accumulator.history(_requirement()) == ()
    # A lifecycle reset creates a new epoch, so a restarted simulation clock
    # cannot inherit stale evidence.
    restarted = accumulator.update(_observation(1.0))
    assert restarted.observation_count == 1


def test_mission_switch_clears_old_uav_history() -> None:
    accumulator = TemporalColorEvidenceAccumulator()
    accumulator.update(_observation(5.0, mission_id="mission_a"))
    accumulator.begin_mission(mission_id="mission_b", uav_id="uav_a")
    accumulator.update(_observation(0.1, mission_id="mission_b"))
    assert accumulator.history(_requirement(mission_id="mission_a")) == ()
    assert len(accumulator.history(_requirement(mission_id="mission_b"))) == 1
    with pytest.raises(AttributeRouteMismatch, match="active UAV mission"):
        accumulator.update(_observation(5.1, mission_id="mission_a"))


def test_temporal_accumulator_rejects_provenance_laundering() -> None:
    accumulator = TemporalColorEvidenceAccumulator()
    with pytest.raises(PermissionError, match="non-production"):
        accumulator.update(
            _observation(
                1.0,
                source="oracle_color",
                runtime_profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            )
        )
    with pytest.raises(ValueError, match="hsv_depth_mask"):
        accumulator.update(_observation(1.0, source="other_vision"))


def test_history_capacity_must_be_able_to_reach_threshold() -> None:
    with pytest.raises(ValueError, match="at least min_observations"):
        TemporalColorEvidenceAccumulator(
            min_observations=4,
            max_history_per_candidate=3,
        )


def test_same_tracker_id_on_two_uavs_never_shares_history() -> None:
    accumulator = TemporalColorEvidenceAccumulator(
        min_observations=2,
        min_duration_s=0.1,
    )
    accumulator.update(_observation(1.0, uav_id="uav_a"))
    other = accumulator.update(_observation(1.0, uav_id="uav_b"))
    assert other.observation_count == 1
    assert len(accumulator.history(_requirement(uav_id="uav_a"))) == 1
    assert len(accumulator.history(_requirement(uav_id="uav_b"))) == 1


def test_bundle_checks_requirement_tracker_route() -> None:
    accumulator = TemporalColorEvidenceAccumulator()
    accumulator.update(_observation(1.0, tracker_id=7))
    bundle = accumulator.bundle(_requirement(tracker_id=7))
    assert bundle.evidence.observation_count == 1
    assert bundle.to_dict()["schema_version"] == 1
    with pytest.raises(AttributeRouteMismatch):
        accumulator.bundle(_requirement(tracker_id=8))
