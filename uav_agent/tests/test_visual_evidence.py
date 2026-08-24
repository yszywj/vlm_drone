from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from perception.candidate_bank import CandidateLifecycle, CandidateSnapshot
from perception.class_aliases import ClassAliasMapper
from perception.visual_evidence import (
    ClosedSetClassSemanticVerifier,
    NonMonotonicTrackTime,
    QwenEvidencePending,
    QwenReacquireIdentityVerifierAdapter,
    QwenSemanticVerifierAdapter,
    ReacquireIdentityRequiresQwen,
    SemanticVerificationRequiresQwen,
    TemporalTrackIdentityVerifier,
    TrackBoxObservation,
    UltralyticsShortTrackEvidenceBuilder,
)
from perception.visual_review import (
    QwenVisualReview,
    VisualReviewAction,
    VisualReviewCandidate,
    VisualReviewDecision,
)
from runtime.frame_store import FrameRef
from target import TargetSpec
from yolo_service.protocol import TrackDetection


def alias_mapper() -> ClassAliasMapper:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "aliases.yaml"
        path.write_text(
            "person:\n  aliases: [person, pedestrian, 人]\n"
            "car:\n  aliases: [car, 汽车]\n",
            encoding="utf-8",
        )
        return ClassAliasMapper.from_yaml(path)


def detection(
    track_id: int = 7,
    *,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.9,
    bbox: tuple[float, float, float, float] = (0.2, 0.2, 0.4, 0.6),
) -> TrackDetection:
    return TrackDetection(track_id, class_id, class_name, confidence, bbox)


def review(
    decision: VisualReviewDecision,
    *,
    timestamp_s: float = 1.2,
    present: bool = True,
    confidence: float = 0.85,
) -> QwenVisualReview:
    candidate = (
        VisualReviewCandidate(
            True,
            (0.2, 0.2, 0.4, 0.6),
            "person",
            confidence,
        )
        if present
        else VisualReviewCandidate(False, None, None, None)
    )
    return QwenVisualReview(
        schema_version=1,
        review_id="review_1",
        mission_id="mission_1",
        uav_id="uav_1",
        plan_version=1,
        observation_timestamp_s=timestamp_s,
        frame_id="frame_3",
        decision=decision,
        candidate=candidate,
        scene_observations=("candidate visible",),
        reason_codes=("semantic_match",),
        recommended_action=VisualReviewAction.CONTINUE,
    )


class ShortTrackEvidenceBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = UltralyticsShortTrackEvidenceBuilder(
            min_observations=3,
            min_duration_s=0.5,
            max_center_jump_normalized=0.2,
            max_observation_gap_s=0.5,
        )

    def test_incremental_same_track_becomes_stable(self) -> None:
        evidence = None
        for timestamp in (0.0, 0.3, 0.6):
            evidence = self.builder.update(
                candidate_id="candidate_7",
                timestamp_s=timestamp,
                detection=detection(),
            )
        assert evidence is not None
        self.assertTrue(evidence.stable)
        self.assertEqual(evidence.observation_count, 3)
        self.assertAlmostEqual(evidence.duration_s, 0.6)
        self.assertEqual(evidence.confidence, 0.9)

    def test_track_id_change_or_bbox_jump_is_unstable(self) -> None:
        changed_track = self.builder.build_history(
            (
                TrackBoxObservation("candidate_7", 7, 0.0, (0.1, 0.1, 0.2, 0.2), 0.9),
                TrackBoxObservation("candidate_7", 8, 0.3, (0.1, 0.1, 0.2, 0.2), 0.9),
                TrackBoxObservation("candidate_7", 8, 0.6, (0.1, 0.1, 0.2, 0.2), 0.9),
            )
        )
        jumped = self.builder.build_history(
            (
                TrackBoxObservation("candidate_7", 7, 0.0, (0.1, 0.1, 0.2, 0.2), 0.9),
                TrackBoxObservation("candidate_7", 7, 0.3, (0.7, 0.7, 0.8, 0.8), 0.9),
                TrackBoxObservation("candidate_7", 7, 0.6, (0.7, 0.7, 0.8, 0.8), 0.9),
            )
        )
        self.assertFalse(changed_track.stable)
        self.assertFalse(jumped.stable)

    def test_non_monotonic_history_is_not_stable_and_live_update_is_rejected(self) -> None:
        history = (
            TrackBoxObservation("candidate_7", 7, 0.0, (0.1, 0.1, 0.2, 0.2), 0.9),
            TrackBoxObservation("candidate_7", 7, 0.6, (0.1, 0.1, 0.2, 0.2), 0.9),
            TrackBoxObservation("candidate_7", 7, 0.4, (0.1, 0.1, 0.2, 0.2), 0.9),
        )
        self.assertFalse(self.builder.build_history(history).stable)

        self.builder.update(
            candidate_id="candidate_7",
            timestamp_s=1.0,
            detection=detection(),
        )
        with self.assertRaises(NonMonotonicTrackTime):
            self.builder.update(
                candidate_id="candidate_7",
                timestamp_s=0.9,
                detection=detection(),
            )

    def test_candidate_bank_history_is_reused_without_fabricated_confidence(self) -> None:
        frames = tuple(
            FrameRef("uav_1", f"frame_{index}", timestamp, 100, 80)
            for index, timestamp in enumerate((0.0, 0.3, 0.6))
        )
        candidate = CandidateSnapshot(
            uav_id="uav_1",
            candidate_id="candidate_7",
            first_seen_timestamp_s=0.0,
            last_seen_timestamp_s=0.6,
            bbox_history=((0.2, 0.2, 0.4, 0.6),) * 3,
            frame_history=frames,
            source="ultralytics_service",
            lifecycle=CandidateLifecycle.PROVISIONAL,
            review_history=(),
        )
        honest = self.builder.build(candidate)
        scored = self.builder.build(candidate, confidences=(0.9, 0.8, 0.7))
        self.assertTrue(honest.stable)
        self.assertEqual(honest.confidence, 0.0)
        self.assertEqual(scored.confidence, 0.7)


class SemanticAndIdentityEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = ClosedSetClassSemanticVerifier(
            alias_mapper(),
            {0: "person", 1: "car"},
        )
        self.track = UltralyticsShortTrackEvidenceBuilder().build_history(
            (
                TrackBoxObservation("candidate_7", 7, 0.0, (0.2, 0.2, 0.4, 0.6), 0.9),
                TrackBoxObservation("candidate_7", 7, 0.3, (0.2, 0.2, 0.4, 0.6), 0.9),
                TrackBoxObservation("candidate_7", 7, 0.6, (0.2, 0.2, 0.4, 0.6), 0.9),
            )
        )

    def test_closed_set_verifier_accepts_only_plain_exact_category(self) -> None:
        result = self.verifier.verify(
            candidate_id="candidate_7",
            timestamp_s=0.7,
            target_spec=TargetSpec("person", category="person"),
            detection=detection(),
        )
        mismatch = self.verifier.verify(
            candidate_id="candidate_7",
            timestamp_s=0.7,
            target_spec=TargetSpec("person", category="person"),
            detection=detection(class_id=1, class_name="car"),
        )
        self.assertTrue(result.matches)
        self.assertFalse(mismatch.matches)

    def test_attributes_relations_negatives_and_specific_identity_require_qwen(self) -> None:
        specs = (
            TargetSpec("person", category="person", hard_attributes=("red coat",)),
            TargetSpec("person", category="person", negative_constraints=("no bag",)),
            TargetSpec("person", category="person", relation_constraints=("near car",)),
            TargetSpec(
                "Alice in the red coat",
                category="person",
                immutable_identity_summary="Alice",
            ),
        )
        for spec in specs:
            with self.subTest(spec=spec):
                with self.assertRaises(SemanticVerificationRequiresQwen):
                    self.verifier.verify(
                        candidate_id="candidate_7",
                        timestamp_s=0.7,
                        target_spec=spec,
                        detection=detection(),
                    )

    def test_new_tracker_id_during_reacquire_requires_qwen(self) -> None:
        temporal = TemporalTrackIdentityVerifier()
        initial = temporal.verify(
            track=self.track,
            target_id="target_1",
            reference_track_id=7,
            current_track_id=7,
        )
        self.assertTrue(initial.reidentified)
        with self.assertRaises(ReacquireIdentityRequiresQwen):
            temporal.verify(
                track=self.track,
                target_id="target_1",
                reference_track_id=7,
                current_track_id=9,
                reacquiring=True,
            )

    def test_qwen_semantic_adapter_preserves_target_description(self) -> None:
        adapter = QwenSemanticVerifierAdapter()
        target_spec = TargetSpec(
            "person wearing a red coat",
            category="person",
            hard_attributes=("red coat",),
        )
        result = adapter.from_review(
            candidate_id="candidate_7",
            target_spec=target_spec,
            review=review(VisualReviewDecision.TARGET_MATCH),
            expected_bbox=(0.2, 0.2, 0.4, 0.6),
        )
        self.assertTrue(result.matches)
        self.assertEqual(result.target_description, target_spec.description)
        self.assertEqual(result.verifier, "qwen_vl")

    def test_qwen_reacquire_adapter_requires_reference_and_terminal_review(self) -> None:
        adapter = QwenReacquireIdentityVerifierAdapter()
        with self.assertRaises(ReacquireIdentityRequiresQwen):
            adapter.from_review(
                track=self.track,
                target_id="target_1",
                review=review(VisualReviewDecision.TARGET_MATCH),
                reference_handles=(),
            )
        result = adapter.from_review(
            track=self.track,
            target_id="target_1",
            review=review(VisualReviewDecision.TARGET_MATCH),
            reference_handles=("frame_reference_1",),
            expected_bbox=(0.2, 0.2, 0.4, 0.6),
        )
        self.assertTrue(result.reidentified)
        self.assertTrue(result.temporally_consistent)
        self.assertEqual(result.source, "qwen_reacquire")

        with self.assertRaises(QwenEvidencePending):
            adapter.from_review(
                track=self.track,
                target_id="target_1",
                review=review(VisualReviewDecision.AMBIGUOUS),
                reference_handles=("frame_reference_1",),
            )


if __name__ == "__main__":
    unittest.main()
