from __future__ import annotations

import unittest

from perception.candidate_bank import (
    CandidateBank,
    CandidateLifecycle,
    CandidateReviewRef,
)
from runtime.frame_store import FrameRef


def frame(frame_id: str, timestamp_s: float, *, uav_id: str = "uav_1") -> FrameRef:
    return FrameRef(uav_id, frame_id, timestamp_s, 64, 48)


class CandidateBankTest(unittest.TestCase):
    def test_lifecycle_and_histories_are_bounded(self) -> None:
        bank = CandidateBank(
            uav_id="uav_1",
            max_history_per_candidate=2,
            max_review_history=2,
        )
        for index in range(3):
            bank.propose(
                candidate_id="candidate_1",
                timestamp_s=float(index),
                bbox_xyxy_normalized=(0.1, 0.2, 0.4, 0.6),
                frame_ref=frame(f"frame_{index}", float(index)),
                source="qwen_vl",
            )
        bank.mark_under_inspection("candidate_1")
        for index in range(3):
            bank.add_review(
                "candidate_1",
                CandidateReviewRef(
                    f"review_{index}",
                    float(index),
                    "AMBIGUOUS",
                ),
            )

        snapshot = bank.get("candidate_1")
        self.assertIsNotNone(snapshot)
        self.assertIs(snapshot.lifecycle, CandidateLifecycle.UNDER_INSPECTION)  # type: ignore[union-attr]
        self.assertEqual(len(snapshot.bbox_history), 2)  # type: ignore[union-attr]
        self.assertEqual(len(snapshot.frame_history), 2)  # type: ignore[union-attr]
        self.assertEqual(len(snapshot.review_history), 2)  # type: ignore[union-attr]

        released = bank.release_inspection_pending_review("candidate_1")
        self.assertIs(released.lifecycle, CandidateLifecycle.PROVISIONAL)
        with self.assertRaisesRegex(ValueError, "PROVISIONAL -> PROVISIONAL"):
            bank.release_inspection_pending_review("candidate_1")

    def test_rejected_candidate_obeys_negative_memory_cooldown(self) -> None:
        bank = CandidateBank(uav_id="uav_1", rejected_cooldown_s=10.0)
        bank.propose(
            candidate_id="candidate_1",
            timestamp_s=1.0,
            bbox_xyxy_normalized=(0.1, 0.2, 0.4, 0.6),
            frame_ref=frame("frame_1", 1.0),
            source="qwen_vl",
        )
        bank.reject("candidate_1", timestamp_s=2.0)

        suppressed = bank.propose(
            candidate_id="candidate_1",
            timestamp_s=5.0,
            bbox_xyxy_normalized=(0.1, 0.2, 0.4, 0.6),
            frame_ref=frame("frame_2", 5.0),
            source="qwen_vl",
        )
        restored = bank.propose(
            candidate_id="candidate_1",
            timestamp_s=12.0,
            bbox_xyxy_normalized=(0.1, 0.2, 0.4, 0.6),
            frame_ref=frame("frame_3", 12.0),
            source="qwen_vl",
        )

        self.assertIsNone(suppressed)
        self.assertIsNotNone(restored)
        self.assertIs(restored.lifecycle, CandidateLifecycle.PROVISIONAL)  # type: ignore[union-attr]

    def test_bank_count_and_staleness_are_bounded(self) -> None:
        bank = CandidateBank(
            uav_id="uav_1",
            max_candidates=2,
            stale_after_s=5.0,
        )
        for index in range(3):
            bank.propose(
                candidate_id=f"candidate_{index}",
                timestamp_s=float(index),
                bbox_xyxy_normalized=(0.1, 0.2, 0.4, 0.6),
                frame_ref=frame(f"frame_{index}", float(index)),
                source="detector",
            )

        self.assertEqual(len(bank.snapshots()), 2)
        self.assertNotIn("candidate_0", {item.candidate_id for item in bank.snapshots()})
        expired = bank.expire(8.0)
        self.assertEqual(set(expired), {"candidate_1", "candidate_2"})

    def test_uav_routing_and_oracle_source_are_explicit(self) -> None:
        bank = CandidateBank(uav_id="uav_1")
        with self.assertRaisesRegex(ValueError, "uav_id"):
            bank.propose(
                candidate_id="candidate_1",
                timestamp_s=1.0,
                bbox_xyxy_normalized=(0.1, 0.2, 0.4, 0.6),
                frame_ref=frame("frame_1", 1.0, uav_id="uav_2"),
                source="detector",
            )
        with self.assertRaisesRegex(ValueError, "oracle_evaluation"):
            bank.propose(
                candidate_id="candidate_1",
                timestamp_s=1.0,
                bbox_xyxy_normalized=(0.1, 0.2, 0.4, 0.6),
                frame_ref=frame("frame_1", 1.0),
                source="oracle",
            )


if __name__ == "__main__":
    unittest.main()
