from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from typing import Sequence

import numpy as np
from PIL import Image

from datasets.target_state.dataset import (
    build_manifest,
    check_dataset,
    compute_dataset_sha256,
    split_for_episode,
)
from datasets.target_state.schema import TargetStateFrameRecord
from datasets.target_state.sequence import build_sequences
from scripts.check_target_state_dataset import main as checker_main
from tests.training.target_state.test_dataset_schema import make_record


class TargetStateDatasetCheckerTest(unittest.TestCase):
    def _write_dataset(
        self,
        root: Path,
        count: int = 7,
        *,
        records: Sequence[TargetStateFrameRecord] | None = None,
    ) -> None:
        (root / "rgb").mkdir(parents=True)
        (root / "depth").mkdir(parents=True)
        stored_records = (
            list(records)
            if records is not None
            else [make_record(index) for index in range(count)]
        )
        for record in stored_records:
            Image.fromarray(np.full((24, 32, 3), 127, dtype=np.uint8)).save(
                root / record.sensor_input.rgb_path
            )
            np.save(
                root / record.sensor_input.depth_path,
                np.full((24, 32), 4.0, dtype=np.float32),
                allow_pickle=False,
            )
        with (root / "frames.jsonl").open("w", encoding="utf-8") as stream:
            for record in stored_records:
                stream.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")

    def test_checker_hashes_files_and_manifest_reports_required_statistics(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_dataset(root)
            records = [make_record(index) for index in range(7)]
            sequences = build_sequences(records, history_size=6)

            report = check_dataset(root)
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.frame_count, 7)
            self.assertEqual(report.sequence_count, 1)
            self.assertEqual(len(report.dataset_sha256 or ""), 64)
            self.assertEqual(sum(report.split_frame_counts.values()), 7)

            manifest = build_manifest(
                records,
                sequences,
                dataset_sha256=report.dataset_sha256 or "",
                generation_commit_sha="nogit",
            )
            self.assertEqual(manifest["scene_count"], 1)
            self.assertEqual(manifest["sequence_count"], 1)
            self.assertEqual(manifest["target_color_distribution"], {"red": 7})
            self.assertEqual(manifest["yolo_miss_rate"], 0.0)
            self.assertEqual(set(manifest["episode_splits"].values()), {split_for_episode("episode_1")})

    def test_checker_uses_manifest_parameters_and_cli_reports_real_sequences(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_dataset(root)
            records = [make_record(index) for index in range(7)]
            sequences = build_sequences(records, history_size=4, max_history_age_s=1.0)
            manifest = build_manifest(
                records,
                sequences,
                dataset_sha256=compute_dataset_sha256(root, records),
            )
            manifest.update(
                {
                    "history_size": 4,
                    "max_history_age_s": 1.0,
                    "detector_prediction_source": "external_capture_spool_unverified",
                    "candidate_id_source": "external_capture_spool_unverified",
                    "detector_truth_association": "external_capture_spool_unverified",
                    "detector_deployment": None,
                    "yolo_model_sha256": "a" * 64,
                    "oracle_usage": "offline_training_labels_only",
                }
            )
            (root / "dataset_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                exit_code = checker_main(["--dataset", str(root)])

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"], payload["errors"])
            self.assertEqual(payload["sequence_count"], 3)

    def test_checker_rejects_incorrect_missing_and_tracker_switch_masks(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = [make_record(index) for index in range(7)]
            records[2] = make_record(2, detected=False)
            records[4] = make_record(4, tracker_id="tracker_2")
            records[5] = make_record(5, tracker_id="tracker_2")
            records[6] = make_record(6, tracker_id="tracker_2")
            self._write_dataset(root, records=records)
            sequence = build_sequences(records, history_size=6)[0]
            corrupted = replace(
                sequence,
                missing_mask=(False,) * 7,
                tracker_id_changed=(False,) * 7,
            )

            report = check_dataset(root, sequences=(corrupted,))

            self.assertFalse(report.ok)
            self.assertTrue(any("missing_mask" in error for error in report.errors))
            self.assertTrue(any("tracker_id_changed" in error for error in report.errors))

    def test_checker_rejects_target_center_outside_camera_resolution(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_dataset(root)
            frames_path = root / "frames.jsonl"
            payloads = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
            payloads[0]["training_label"]["center_pixel_uv"] = [32.0, 12.0]
            frames_path.write_text(
                "".join(json.dumps(payload, allow_nan=False) + "\n" for payload in payloads),
                encoding="utf-8",
            )

            report = check_dataset(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("center_pixel_uv" in error for error in report.errors))

    def test_checker_fails_on_resolution_or_missing_depth(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_dataset(root, count=1)
            np.save(root / "depth/frame_0.npy", np.ones((2, 2), dtype=np.float32))

            report = check_dataset(root)
            self.assertFalse(report.ok)
            self.assertTrue(any("depth resolution mismatch" in error for error in report.errors))

    def test_episode_split_is_deterministic_and_never_frame_based(self) -> None:
        expected = split_for_episode("episode_123", seed=9)
        self.assertEqual(expected, split_for_episode("episode_123", seed=9))
        self.assertIn(expected, {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
