from __future__ import annotations

import numpy as np
import pytest

from configs.schema import TargetPerceptionConfig, VisualConfirmationConfig
from env.camera_types import CameraIntrinsics, CameraSample
from perception.candidate_bank import CandidateLifecycle, CandidateSnapshot
from perception.color_attribute_verifier import (
    RgbdColorAttributeVerifier,
    TemporalColorEvidenceAccumulator,
)
from perception.attribute_types import AttributeDecision, AttributeEvidence
from perception.attribute_verifier import AttributeRouteMismatch
from perception.semantic_fusion import (
    AttributeSemanticVerificationPending,
    AttributeSemanticVerificationRequiresQwen,
    DeterministicAttributeSemanticVerifier,
    TemporalRgbdAttributeSemanticProvider,
)
from perception.types import SemanticVerification
from target.types import TargetSpec
from runtime.frame_store import FrameRef, FrameStore
from yolo_service.protocol import TrackDetection


def _target(*attributes: str) -> TargetSpec:
    return TargetSpec(
        "红色立方体",
        category="cube",
        hard_attributes=attributes,
    )


def _detection(
    *, class_name: str = "cube", class_id: int = 0, confidence: float = 0.8
) -> TrackDetection:
    return TrackDetection(
        track_id=7,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy_normalized=(0.1, 0.1, 0.9, 0.9),
    )


def _evidence(
    decision: AttributeDecision,
    *, observed: str | None,
    confidence: float = 0.7,
    timestamp_s: float = 2.0,
    candidate_id: str = "candidate_a",
    observation_count: int = 3,
    duration_s: float = 0.5,
) -> AttributeEvidence:
    return AttributeEvidence(
        mission_id="mission_a",
        uav_id="uav_a",
        assignment_id="assignment_a",
        candidate_id=candidate_id,
        tracker_id=7,
        timestamp_s=timestamp_s,
        attribute_name="color",
        expected_value="red",
        observed_value=observed,
        decision=decision,
        confidence=confidence,
        observation_count=observation_count,
        duration_s=duration_s,
        valid_sample_ratio=0.8,
        source="hsv_depth_mask",
        reason_code="temporal_color_stable",
    )


def _verify(
    evidence: AttributeEvidence | None,
    **kwargs: object,
) -> SemanticVerification:
    return DeterministicAttributeSemanticVerifier(
        expected_class_id=0
    ).verify(
        candidate_id="candidate_a",
        timestamp_s=2.0,
        target_spec=_target("color=red"),
        detection=_detection(),
        attribute_evidence=evidence,
        mission_id="mission_a",
        uav_id="uav_a",
        assignment_id="assignment_a",
        **kwargs,  # type: ignore[arg-type]
    )


def test_non_cube_class_is_semantic_mismatch() -> None:
    result = DeterministicAttributeSemanticVerifier(
        expected_class_id=0
    ).verify(
        candidate_id="candidate_a",
        timestamp_s=2.0,
        target_spec=_target("color=red"),
        detection=_detection(class_name="person", class_id=1),
    )
    assert result.matches is False
    assert result.confidence == 0.8
    assert result.verifier == "closed_set_class+temporal_rgbd_color"


def test_temporal_color_match_and_mismatch_use_min_confidence() -> None:
    match = _verify(
        _evidence(AttributeDecision.MATCH, observed="red", confidence=0.7)
    )
    mismatch = _verify(
        _evidence(AttributeDecision.MISMATCH, observed="blue", confidence=0.6)
    )
    assert match.matches is True
    assert match.confidence == 0.7
    assert mismatch.matches is False
    assert mismatch.confidence == 0.6
    assert match.verifier == "closed_set_class+temporal_rgbd_color"


def test_pending_color_keeps_candidate_pending_instead_of_rejecting() -> None:
    with pytest.raises(AttributeSemanticVerificationPending):
        _verify(
            _evidence(
                AttributeDecision.PENDING,
                observed="unknown",
                confidence=0.0,
            )
        )


def test_claimed_single_frame_mismatch_cannot_reject_candidate() -> None:
    with pytest.raises(AttributeSemanticVerificationPending):
        _verify(
            _evidence(
                AttributeDecision.MISMATCH,
                observed="blue",
                observation_count=1,
                duration_s=0.0,
            )
        )


def test_unsupported_attribute_requests_qwen() -> None:
    verifier = DeterministicAttributeSemanticVerifier(expected_class_id=0)
    with pytest.raises(AttributeSemanticVerificationRequiresQwen):
        verifier.verify(
            candidate_id="candidate_a",
            timestamp_s=2.0,
            target_spec=_target("color=red", "logo=acme"),
            detection=_detection(),
            attribute_evidence=_evidence(AttributeDecision.MATCH, observed="red"),
            mission_id="mission_a",
            uav_id="uav_a",
            assignment_id="assignment_a",
        )


def test_qwen_gate_requires_acknowledgement_and_shadow_never_confirms() -> None:
    qwen = SemanticVerification(
        candidate_id="candidate_a",
        timestamp_s=2.0,
        target_description="红色立方体",
        matches=True,
        confidence=0.9,
        verifier="qwen_vlm",
    )
    pending = _evidence(
        AttributeDecision.PENDING,
        observed="unknown",
        confidence=0.0,
    )
    with pytest.raises(AttributeSemanticVerificationPending):
        _verify(pending, qwen_verification=qwen, qwen_mode="shadow")
    with pytest.raises(PermissionError, match="acknowledgement"):
        _verify(pending, qwen_verification=qwen, qwen_mode="gate")
    result = _verify(
        pending,
        qwen_verification=qwen,
        qwen_mode="gate",
        acknowledge_vision_gate=True,
    )
    assert result.matches is True
    assert result.confidence == 0.8
    assert result.verifier == "closed_set_class+temporal_rgbd_color"


def test_qwen_gate_cannot_bypass_early_color_accumulation() -> None:
    qwen = SemanticVerification(
        candidate_id="candidate_a",
        timestamp_s=2.0,
        target_description="红色立方体",
        matches=True,
        confidence=0.9,
        verifier="qwen_vlm",
    )
    early = _evidence(
        AttributeDecision.PENDING,
        observed="unknown",
        confidence=0.0,
        observation_count=1,
        duration_s=0.0,
    )
    with pytest.raises(AttributeSemanticVerificationPending):
        _verify(
            early,
            qwen_verification=qwen,
            qwen_mode="gate",
            acknowledge_vision_gate=True,
        )


def test_stale_qwen_and_attribute_routes_fail_closed() -> None:
    stale_qwen = SemanticVerification(
        candidate_id="candidate_old",
        timestamp_s=2.0,
        target_description="红色立方体",
        matches=True,
        confidence=0.9,
        verifier="qwen_vlm",
    )
    with pytest.raises(ValueError, match="different candidate epoch"):
        _verify(
            _evidence(AttributeDecision.PENDING, observed="unknown", confidence=0.0),
            qwen_verification=stale_qwen,
            qwen_mode="gate",
            acknowledge_vision_gate=True,
        )
    with pytest.raises(ValueError, match="candidate_id mismatch"):
        _verify(
            _evidence(
                AttributeDecision.MATCH,
                observed="red",
                candidate_id="candidate_old",
            )
        )


def test_terminal_deterministic_evidence_forbids_redundant_qwen_gate() -> None:
    qwen = SemanticVerification(
        candidate_id="candidate_a",
        timestamp_s=2.0,
        target_description="红色立方体",
        matches=False,
        confidence=0.9,
        verifier="qwen_vlm",
    )
    with pytest.raises(ValueError, match="must not be called"):
        _verify(
            _evidence(AttributeDecision.MATCH, observed="red"),
            qwen_verification=qwen,
            qwen_mode="gate",
            acknowledge_vision_gate=True,
        )


def test_class_only_cube_target_needs_no_attribute_or_qwen() -> None:
    result = DeterministicAttributeSemanticVerifier(expected_class_id=0).verify(
        candidate_id="candidate_a",
        timestamp_s=2.0,
        target_spec=_target(),
        detection=_detection(confidence=0.77),
    )
    assert result.matches is True
    assert result.confidence == 0.77


@pytest.mark.parametrize(
    "target",
    [
        TargetSpec(
            "不是蓝色立方体",
            category="cube",
            hard_attributes=("color=red",),
            negative_constraints=("color=blue",),
        ),
        TargetSpec(
            "车辆旁的红色立方体",
            category="cube",
            hard_attributes=("color=red",),
            relation_constraints=("beside=vehicle",),
        ),
    ],
)
def test_relations_and_negative_constraints_require_qwen(target: TargetSpec) -> None:
    with pytest.raises(AttributeSemanticVerificationRequiresQwen):
        DeterministicAttributeSemanticVerifier(expected_class_id=0).verify(
            candidate_id="candidate_a",
            timestamp_s=2.0,
            target_spec=target,
            detection=_detection(),
            attribute_evidence=_evidence(AttributeDecision.MATCH, observed="red"),
            mission_id="mission_a",
            uav_id="uav_a",
            assignment_id="assignment_a",
        )


def test_attribute_evidence_requires_explicit_external_route() -> None:
    with pytest.raises(ValueError, match="routed attribute evidence"):
        DeterministicAttributeSemanticVerifier(expected_class_id=0).verify(
            candidate_id="candidate_a",
            timestamp_s=2.0,
            target_spec=_target("color=red"),
            detection=_detection(),
            attribute_evidence=_evidence(AttributeDecision.MATCH, observed="red"),
        )


def _camera_sample(timestamp_s: float, rgb: tuple[int, int, int]) -> CameraSample:
    pixels = np.empty((20, 20, 3), dtype=np.uint8)
    pixels[:] = rgb
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=pixels,
        depth_to_image_plane_m=np.full((20, 20), 10.0, dtype=np.float32),
        camera_position_world_m=(0.0, 0.0, 0.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=CameraIntrinsics(
            fx=10.0,
            fy=10.0,
            cx=10.0,
            cy=10.0,
            width=20,
            height=20,
        ),
    )


def _candidate(frames: tuple[FrameRef, ...]) -> CandidateSnapshot:
    bbox = (0.1, 0.1, 0.9, 0.9)
    return CandidateSnapshot(
        uav_id="uav_a",
        candidate_id="candidate_a",
        first_seen_timestamp_s=frames[0].timestamp_s,
        last_seen_timestamp_s=frames[-1].timestamp_s,
        bbox_history=(bbox,) * len(frames),
        frame_history=frames,
        source="ultralytics_service",
        lifecycle=CandidateLifecycle.UNDER_INSPECTION,
        review_history=(),
    )


def _provider(store: FrameStore) -> TemporalRgbdAttributeSemanticProvider:
    return TemporalRgbdAttributeSemanticProvider(
        mission_id="mission_a",
        uav_id="uav_a",
        assignment_id="assignment_a",
        frame_store=store,
        color_verifier=RgbdColorAttributeVerifier(store),
        accumulator=TemporalColorEvidenceAccumulator(
            min_observations=3,
            min_duration_s=0.4,
        ),
        semantic_verifier=DeterministicAttributeSemanticVerifier(
            expected_class_id=0,
            min_color_observations=3,
            min_color_duration_s=0.4,
        ),
    )


def test_stateful_provider_is_directly_callable_by_target_coordinator() -> None:
    store = FrameStore()
    refs = tuple(
        store.add_sample(
            uav_id="uav_a",
            frame_id=f"frame_{index}",
            sample=_camera_sample(timestamp, (255, 0, 0)),
        )
        for index, timestamp in enumerate((1.0, 1.2, 1.4), start=1)
    )
    provider = _provider(store)
    outputs = [
        provider(
            _candidate(refs[:index]),
            _target("color=red"),
            _detection(),
            ref.timestamp_s,
        )
        for index, ref in enumerate(refs, start=1)
    ]
    assert outputs[:2] == [None, None]
    assert outputs[2] is not None and outputs[2].matches is True
    assert outputs[2].verifier == "closed_set_class+temporal_rgbd_color"
    records = provider.evidence_records()
    assert len(records) == 3
    assert all(
        not ({"rgb", "depth", "path", "base64"} & record.to_dict().keys())
        for record in records
    )
    metrics = provider.metrics.to_dict()
    assert metrics["observations_total"] == 3
    assert metrics["semantic_match"] == 1
    assert metrics["semantic_pending"] == 2


def test_stateful_provider_builds_from_yolo_attribute_config() -> None:
    store = FrameStore()
    provider = TemporalRgbdAttributeSemanticProvider.from_target_perception_config(
        TargetPerceptionConfig(
            backend="ultralytics_service",
            confirmation=VisualConfirmationConfig(
                mode="class_track_attribute_or_qwen"
            ),
        ),
        mission_id="mission_a",
        uav_id="uav_a",
        assignment_id="assignment_a",
        frame_store=store,
        expected_class_id=0,
    )
    assert provider.metrics.observations_total == 0


def test_stateful_provider_tracker_epoch_resets_color_votes() -> None:
    store = FrameStore()
    refs = tuple(
        store.add_sample(
            uav_id="uav_a",
            frame_id=f"switch_frame_{index}",
            sample=_camera_sample(timestamp, (255, 0, 0)),
        )
        for index, timestamp in enumerate((2.0, 2.2, 2.4), start=1)
    )
    provider = _provider(store)
    for index, ref in enumerate(refs[:2], start=1):
        assert (
            provider(
                _candidate(refs[:index]),
                _target("color=red"),
                _detection(),
                ref.timestamp_s,
            )
            is None
        )
    switched_detection = TrackDetection(
        track_id=8,
        class_id=0,
        class_name="cube",
        confidence=0.8,
        bbox_xyxy_normalized=(0.1, 0.1, 0.9, 0.9),
    )
    assert (
        provider(
            _candidate(refs),
            _target("color=red"),
            switched_detection,
            refs[-1].timestamp_s,
        )
        is None
    )
    assert provider.evidence_records()[-1].observation_count == 1
    assert provider.metrics.tracker_epoch_resets == 1
    assert provider.requires_qwen("candidate_a") is True
    assert provider.qwen_requirement_reason("candidate_a") == "tracker_epoch_changed"


def test_stateful_provider_reset_exposes_clean_scalar_state() -> None:
    store = FrameStore()
    ref = store.add_sample(
        uav_id="uav_a",
        frame_id="reset_frame",
        sample=_camera_sample(1.0, (255, 0, 0)),
    )
    provider = _provider(store)
    provider(_candidate((ref,)), _target("color=red"), _detection(), 1.0)
    provider.reset_candidate("candidate_a")
    assert provider.evidence_records()  # audit records survive candidate reset
    drained = provider.drain_evidence_records()
    assert len(drained) == 1
    assert provider.evidence_records() == ()
    provider.reset(
        mission_id="mission_b",
        uav_id="uav_a",
        assignment_id="assignment_b",
    )
    assert provider.evidence_records() == ()
    assert provider.metrics.observations_total == 0


def test_stateful_provider_exposes_only_long_pending_as_qwen_required() -> None:
    store = FrameStore()
    refs = tuple(
        store.add_sample(
            uav_id="uav_a",
            frame_id=f"gray_frame_{index}",
            sample=_camera_sample(timestamp, (128, 128, 128)),
        )
        for index, timestamp in enumerate((3.0, 3.2, 3.4), start=1)
    )
    provider = _provider(store)
    assert (
        provider(
            _candidate(refs[:1]),
            _target("color=red"),
            _detection(),
            refs[0].timestamp_s,
        )
        is None
    )
    assert provider.requires_qwen("candidate_a") is False
    for index in (2, 3):
        assert (
            provider(
                _candidate(refs[:index]),
                _target("color=red"),
                _detection(),
                refs[index - 1].timestamp_s,
            )
            is None
        )
    assert provider.requires_qwen("candidate_a") is True
    assert (
        provider.qwen_requirement_reason("candidate_a")
        == "persistent_attribute_pending"
    )
    provider.clear_qwen_requirement("candidate_a")
    assert provider.requires_qwen("candidate_a") is False


def test_stateful_provider_reset_rejects_other_uav() -> None:
    provider = _provider(FrameStore())
    with pytest.raises(AttributeRouteMismatch, match="different UAV"):
        provider.reset(
            mission_id="mission_b",
            uav_id="uav_b",
            assignment_id="assignment_b",
        )
