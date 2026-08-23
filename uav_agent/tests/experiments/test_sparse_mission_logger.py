from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.sparse_mission_logger import (
    MAX_RECORD_BYTES,
    MissionEventLogRecord,
    QwenReviewLogRecord,
    SkillTransitionLogRecord,
    SparseMissionLogger,
)


class SparseMissionLoggerTest(unittest.TestCase):
    @staticmethod
    def _review(**overrides: object) -> QwenReviewLogRecord:
        values: dict[str, object] = {
            "review_id": "review_1",
            "request_id": "request_1",
            "mission_id": "mission_1",
            "uav_id": "uav_1",
            "plan_version": 2,
            "step_id": "step_3",
            "frame_id": "frame_9",
            "observation_timestamp_s": 12.5,
            "decision": "TARGET_MATCH",
            "bbox_xyxy_normalized": (0.1, 0.2, 0.4, 0.6),
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "latency_s": 0.75,
            "stale": False,
            "accepted": True,
            "timeout": False,
        }
        values.update(overrides)
        return QwenReviewLogRecord(**values)  # type: ignore[arg-type]

    def test_writes_only_fixed_sparse_fields_and_flushes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = SparseMissionLogger(temporary)
            self.assertTrue(logger.log_qwen_review(self._review()))
            self.assertTrue(
                logger.log_mission_event(
                    event_record := MissionEventLogRecord(
                        12.5,
                        "mission_1",
                        "uav_1",
                        2,
                        "step_3",
                        "SEARCH",
                        "MODEL_REVIEW_COMPLETED",
                        "RUNNING",
                        "periodic_review",
                    )
                )
            )
            terminal_line = event_record.to_terminal_line()
            for expected in (
                "mission_id=mission_1",
                "uav_id=uav_1",
                "plan_version=2",
                "step_id=step_3",
                "skill=SEARCH",
            ):
                self.assertIn(expected, terminal_line)
            self.assertTrue(
                logger.log_skill_transition(
                    SkillTransitionLogRecord(
                        12.6,
                        "mission_1",
                        "uav_1",
                        2,
                        "step_3",
                        "SEARCH",
                        "INSPECT",
                        "RUNNING",
                        "TARGET_FOUND",
                        "candidate_persistent",
                    )
                )
            )

            review_path = Path(temporary) / "qwen_reviews.jsonl"
            payload = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {
                    "review_id",
                    "request_id",
                    "mission_id",
                    "uav_id",
                    "plan_version",
                    "step_id",
                    "frame_id",
                    "observation_timestamp_s",
                    "decision",
                    "semantic_source",
                    "geometry_source",
                    "bbox_xyxy_normalized",
                    "token_usage",
                    "latency_s",
                    "stale",
                    "stale_reasons",
                    "accepted",
                    "timeout",
                    "error_code",
                    "response_text_length",
                    "response_text_tail",
                },
            )
            serialized = review_path.read_text(encoding="utf-8").casefold()
            self.assertNotIn("base64", serialized)
            self.assertNotIn("image", serialized)
            with (Path(temporary) / "mission_events.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            with (Path(temporary) / "skill_transitions.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            logger.close()

    def test_record_and_byte_budgets_drop_without_growing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = SparseMissionLogger(
                temporary,
                max_records_per_stream=1,
                max_bytes_per_stream=MAX_RECORD_BYTES,
            )
            self.assertTrue(logger.log_qwen_review(self._review()))
            initial_size = (Path(temporary) / "qwen_reviews.jsonl").stat().st_size
            self.assertFalse(
                logger.log_qwen_review(self._review(review_id="review_2"))
            )
            self.assertEqual(
                (Path(temporary) / "qwen_reviews.jsonl").stat().st_size,
                initial_size,
            )
            self.assertEqual(logger.snapshot().dropped_log_record_count, 1)
            self.assertEqual(logger.snapshot().review_count, 2)
            logger.close()

    def test_manifest_summary_counts_review_revision_hover_and_debug(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with SparseMissionLogger(temporary) as logger:
                logger.log_qwen_review(
                    self._review(stale=True, accepted=False, timeout=True)
                )
                logger.record_plan_revision()
                logger.record_supervisory_hover(2.25)
                summary = logger.snapshot().to_manifest_dict(
                    debug_images_count=2,
                    debug_images_bytes=1000,
                )
            self.assertEqual(summary["qwen_visual_reviews"]["count"], 1)  # type: ignore[index]
            self.assertEqual(summary["qwen_visual_reviews"]["stale"], 1)  # type: ignore[index]
            self.assertEqual(summary["qwen_visual_reviews"]["timeout"], 1)  # type: ignore[index]
            self.assertEqual(summary["plan_revisions"], 1)
            self.assertEqual(summary["supervisory_hover"]["total_time_s"], 2.25)  # type: ignore[index]
            self.assertEqual(summary["debug_images"]["bytes"], 1000)  # type: ignore[index]

    def test_inflight_hover_is_counted_before_its_duration_is_known(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with SparseMissionLogger(temporary) as logger:
                logger.record_supervisory_hover_started()
                during = logger.snapshot()
                self.assertEqual(during.hover_count, 1)
                self.assertEqual(during.hover_total_time_s, 0.0)
                logger.record_supervisory_hover_duration(1.5)
                after = logger.snapshot()
            self.assertEqual(after.hover_count, 1)
            self.assertEqual(after.hover_total_time_s, 1.5)

    def test_image_or_base64_text_and_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._review(decision="data:image/jpeg;base64,AAAA")
        with self.assertRaises(ValueError):
            self._review(latency_s=float("inf"))
        with self.assertRaises(ValueError):
            self._review(bbox_xyxy_normalized=(0.5, 0.2, 0.4, 0.6))
        with self.assertRaises(TypeError):
            self._review(prompt_tokens=True)
        with self.assertRaises(ValueError):
            self._review(semantic_source="oracle")
        with self.assertRaises(ValueError):
            self._review(geometry_source="qwen_vl")
        with self.assertRaises(ValueError):
            self._review(stale_reasons=("step_id_changed",))
        with self.assertRaises(ValueError):
            self._review(
                stale=True,
                accepted=False,
                stale_reasons=("not_a_stale_reason",),
            )
        with self.assertRaises(ValueError):
            self._review(
                response_text_length=100,
                response_text_tail="data:image/jpeg;base64,AAAA",
            )

    def test_specific_stale_reasons_and_safe_debug_tail_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with SparseMissionLogger(temporary) as logger:
                logger.log_qwen_review(
                    self._review(
                        stale=True,
                        accepted=False,
                        error_code="STALE",
                        stale_reasons=("step_id_changed",),
                        response_text_length=640,
                        response_text_tail='{"decision":"NO_RELEVANT_CHANGE"}',
                    )
                )
            payload = json.loads(
                (Path(temporary) / "qwen_reviews.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["error_code"], "STALE")
            self.assertEqual(payload["stale_reasons"], ["step_id_changed"])
            self.assertEqual(payload["response_text_length"], 640)
            self.assertLessEqual(len(payload["response_text_tail"]), 500)

    def test_close_is_idempotent_and_prevents_further_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = SparseMissionLogger(temporary)
            logger.close()
            logger.close()
            with self.assertRaises(RuntimeError):
                logger.log_qwen_review(self._review())


if __name__ == "__main__":
    unittest.main()
