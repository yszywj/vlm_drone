from __future__ import annotations

import json
import unittest

from perception.visual_review import (
    QwenVisualReview,
    ReviewDisposition,
    VisualReviewAction,
    VisualReviewCandidate,
    VisualReviewDecision,
    VisualReviewExpectation,
    VisualReviewGate,
    VisualReviewMode,
    VisualReviewProtocolError,
    build_qwen_visual_review_json_schema,
)


def valid_review() -> dict[str, object]:
    return {
        "schema_version": 1,
        "review_id": "review_001",
        "mission_id": "mission_001",
        "uav_id": "uav_1",
        "plan_version": 1,
        "observation_timestamp_s": 12.5,
        "frame_id": "frame_0081",
        "decision": "POSSIBLE_TARGET",
        "candidate": {
            "present": True,
            "bbox_xyxy_normalized": [0.1, 0.2, 0.4, 0.6],
            "description": "red cube-like object",
            "self_reported_confidence": 0.82,
        },
        "scene_observations": ["target is small"],
        "reason_codes": ["NEEDS_TEMPORAL_CONFIRMATION"],
        "recommended_action": "INSPECT",
    }


class VisualReviewTypesTest(unittest.TestCase):
    def test_valid_review_round_trip(self) -> None:
        review = QwenVisualReview.from_dict(valid_review())

        self.assertIs(review.decision, VisualReviewDecision.POSSIBLE_TARGET)
        self.assertIs(review.recommended_action, VisualReviewAction.INSPECT)
        self.assertEqual(json.loads(json.dumps(review.to_dict())), valid_review())

    def test_routing_id_and_version_mismatch_values_are_strict(self) -> None:
        for field, value in (
            ("uav_id", "wrong id"),
            ("mission_id", ""),
            ("review_id", "_review"),
            ("frame_id", "frame/id"),
            ("plan_version", 0),
        ):
            with self.subTest(field=field):
                data = valid_review()
                data[field] = value
                with self.assertRaises((TypeError, ValueError)):
                    QwenVisualReview.from_dict(data)

    def test_bbox_and_candidate_consistency_are_strict(self) -> None:
        for bbox in (
            [-0.1, 0.2, 0.4, 0.6],
            [0.4, 0.2, 0.1, 0.6],
            [0.1, 0.2, 0.4],
            [0.1, 0.2, float("nan"), 0.6],
        ):
            with self.subTest(bbox=bbox):
                data = valid_review()
                data["candidate"] = dict(data["candidate"])  # type: ignore[arg-type]
                data["candidate"]["bbox_xyxy_normalized"] = bbox  # type: ignore[index]
                with self.assertRaises((TypeError, ValueError)):
                    QwenVisualReview.from_dict(data)
        with self.assertRaises(ValueError):
            VisualReviewCandidate(False, (0.1, 0.2, 0.3, 0.4), None, None)

    def test_unknown_fields_and_unapproved_enums_are_rejected(self) -> None:
        data = valid_review()
        data["velocity"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            QwenVisualReview.from_dict(data)

        data = valid_review()
        data["recommended_action"] = "FLY_TO_XYZ"
        with self.assertRaisesRegex(ValueError, "not supported"):
            QwenVisualReview.from_dict(data)

    def test_response_schema_contains_only_normalized_image_geometry(self) -> None:
        schema = build_qwen_visual_review_json_schema(
            review_id="review_001",
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=1,
            frame_id="frame_0081",
            observation_timestamp_s=12.5,
        )
        encoded = json.dumps(schema, sort_keys=True)

        self.assertIn("bbox_xyxy_normalized", encoded)
        self.assertNotIn("world_position", encoded)
        self.assertNotIn("velocity", encoded)
        self.assertNotIn("acceleration", encoded)
        self.assertEqual(schema["properties"]["uav_id"], {"const": "uav_1"})  # type: ignore[index]

    def test_shadow_mode_records_but_never_affects_control(self) -> None:
        review = QwenVisualReview.from_dict(valid_review())
        expected = VisualReviewExpectation(
            review_id="review_001",
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=12.5,
            frame_id="frame_0081",
        )

        outcome = VisualReviewGate(mode=VisualReviewMode.SHADOW).evaluate(
            review,
            expected,
        )

        self.assertIs(outcome.disposition, ReviewDisposition.SHADOW_RECORDED)
        self.assertFalse(outcome.accepted_for_control)

    def test_gate_requires_multiple_consistent_matches_without_reid(self) -> None:
        data = valid_review()
        data["decision"] = "TARGET_MATCH"
        data["recommended_action"] = "CONTINUE"
        review = QwenVisualReview.from_dict(data)
        expected = VisualReviewExpectation(
            review_id="review_001",
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=12.5,
            frame_id="frame_0081",
        )
        gate = VisualReviewGate(mode="gate", min_consistent_matches=2)

        first = gate.evaluate(review, expected, consensus_key="target_001")
        duplicate = gate.evaluate(review, expected, consensus_key="target_001")
        second_data = valid_review()
        second_data.update(
            {
                "review_id": "review_002",
                "frame_id": "frame_0082",
                "observation_timestamp_s": 13.0,
                "decision": "TARGET_MATCH",
                "recommended_action": "CONTINUE",
            }
        )
        second_review = QwenVisualReview.from_dict(second_data)
        second_expected = VisualReviewExpectation(
            review_id="review_002",
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=13.0,
            frame_id="frame_0082",
        )
        second = gate.evaluate(
            second_review,
            second_expected,
            consensus_key="target_001",
        )

        self.assertIs(first.disposition, ReviewDisposition.PENDING)
        self.assertFalse(first.accepted_for_control)
        self.assertIs(duplicate.disposition, ReviewDisposition.STALE)
        self.assertEqual(duplicate.consistent_match_count, 1)
        self.assertIs(second.disposition, ReviewDisposition.CONSENSUS_REACHED)
        self.assertTrue(second.accepted_for_control)
        self.assertNotIn("reid", second.reason.casefold())

    def test_gate_never_combines_different_consensus_identities(self) -> None:
        first_data = valid_review()
        first_data.update(
            {
                "decision": "TARGET_MATCH",
                "recommended_action": "CONTINUE",
            }
        )
        first_review = QwenVisualReview.from_dict(first_data)
        first_expected = VisualReviewExpectation(
            review_id="review_001",
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=12.5,
            frame_id="frame_0081",
        )
        second_data = valid_review()
        second_data.update(
            {
                "review_id": "review_002",
                "frame_id": "frame_0082",
                "observation_timestamp_s": 13.0,
                "decision": "TARGET_MATCH",
                "recommended_action": "CONTINUE",
            }
        )
        second_review = QwenVisualReview.from_dict(second_data)
        second_expected = VisualReviewExpectation(
            review_id="review_002",
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=13.0,
            frame_id="frame_0082",
        )
        gate = VisualReviewGate(mode="gate", min_consistent_matches=2)

        first = gate.evaluate(
            first_review,
            first_expected,
            consensus_key="candidate_001",
        )
        second = gate.evaluate(
            second_review,
            second_expected,
            consensus_key="candidate_002",
        )

        self.assertEqual(first.consistent_match_count, 1)
        self.assertEqual(second.consistent_match_count, 1)
        self.assertIs(second.disposition, ReviewDisposition.PENDING)
        self.assertFalse(second.accepted_for_control)

    def test_stale_and_wrong_uav_results_are_rejected(self) -> None:
        review = QwenVisualReview.from_dict(valid_review())
        stale = VisualReviewExpectation(
            review_id="review_001",
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=2,
            observation_timestamp_s=12.5,
            frame_id="frame_0081",
        )
        outcome = VisualReviewGate(mode="gate").evaluate(review, stale)
        self.assertIs(outcome.disposition, ReviewDisposition.STALE)

        wrong_uav = VisualReviewExpectation(
            review_id="review_001",
            mission_id="mission_001",
            uav_id="uav_2",
            plan_version=1,
            observation_timestamp_s=12.5,
            frame_id="frame_0081",
        )
        with self.assertRaisesRegex(VisualReviewProtocolError, "uav_id"):
            VisualReviewGate(mode="gate").evaluate(review, wrong_uav)


if __name__ == "__main__":
    unittest.main()
