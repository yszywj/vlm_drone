"""Regression tests for strict target-state dataset manifest validation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from datasets.target_state.dataset import (
    build_manifest,
    check_dataset,
    compute_dataset_sha256,
)
from datasets.target_state.schema import TargetStateFrameRecord
from datasets.target_state.sequence import build_sequences
from tests.training.target_state.test_dataset_schema import make_record


_FLOAT_FIELDS = (
    "mean_detector_bbox_center_step_normalized",
    "mean_occlusion_ratio",
    "yolo_miss_rate",
)
_MODEL_SHA = "a" * 64


def _write_records(root: Path, *, count: int) -> list[TargetStateFrameRecord]:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    records = [make_record(index) for index in range(count)]
    for record in records:
        Image.fromarray(np.full((24, 32, 3), 127, dtype=np.uint8)).save(
            root / record.sensor_input.rgb_path
        )
        np.save(
            root / record.sensor_input.depth_path,
            np.full((24, 32), 4.0, dtype=np.float32),
            allow_pickle=False,
        )
    (root / "frames.jsonl").write_text(
        "".join(
            json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return records


def _write_valid_manifest(
    root: Path,
    records: list[TargetStateFrameRecord],
) -> dict[str, object]:
    manifest = build_manifest(
        records,
        build_sequences(records, history_size=6, max_history_age_s=2.0),
        dataset_sha256=compute_dataset_sha256(root, records),
        generation_commit_sha="testcommit",
    )
    manifest.update(
        {
            "detector_prediction_source": "real_yolo_deployment_output",
            "candidate_id_source": "sensor_only_bbox_color_temporal_linker",
            "detector_truth_association": (
                "offline_privileged_one_to_one_iou_after_worker_inference"
            ),
            "detector_deployment": {
                "preflight_verified": True,
                "model_family": "yolo",
                "model_names": {"0": "cube"},
                "model_sha256": _MODEL_SHA,
                "worker_url": "http://127.0.0.1:8011",
            },
            "yolo_model_sha256": _MODEL_SHA,
            "oracle_usage": "offline_training_labels_only",
            "history_size": 6,
            "max_history_age_s": 2.0,
        }
    )
    _store_manifest(root, manifest)
    report = check_dataset(root)
    if not report.ok:
        raise AssertionError(report.errors)
    return manifest


def _store_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


class TargetStateDatasetManifestToleranceTest(unittest.TestCase):
    def test_real_collection_roundoff_at_e_minus_18_is_accepted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = _write_records(root, count=7)
            manifest = _write_valid_manifest(root, records)
            manifest["mean_detector_bbox_center_step_normalized"] = (
                float(manifest["mean_detector_bbox_center_step_normalized"])
                + 6.938893903907228e-18
            )
            _store_manifest(root, manifest)

            report = check_dataset(root)

            self.assertTrue(report.ok, report.errors)

    def test_roundoff_near_absolute_tolerance_is_accepted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = _write_records(root, count=7)
            manifest = _write_valid_manifest(root, records)
            manifest["yolo_miss_rate"] = (
                float(manifest["yolo_miss_rate"]) + 1e-15
            )
            _store_manifest(root, manifest)

            report = check_dataset(root)

            self.assertTrue(report.ok, report.errors)

    def test_large_error_is_rejected_for_every_derived_float(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = _write_records(root, count=7)
            original = _write_valid_manifest(root, records)
            for field in _FLOAT_FIELDS:
                with self.subTest(field=field):
                    manifest = dict(original)
                    manifest[field] = float(original[field]) + 1e-4
                    _store_manifest(root, manifest)

                    report = check_dataset(root)

                    self.assertFalse(report.ok)
                    self.assertTrue(
                        any(field in error for error in report.errors),
                        report.errors,
                    )

    def test_none_matches_none(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = _write_records(root, count=0)
            manifest = _write_valid_manifest(root, records)
            self.assertTrue(all(manifest[field] is None for field in _FLOAT_FIELDS))

            report = check_dataset(root)

            self.assertTrue(report.ok, report.errors)

    def test_none_and_float_never_match_in_either_direction(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = _write_records(root, count=7)
            manifest = _write_valid_manifest(root, records)
            manifest["mean_occlusion_ratio"] = None
            _store_manifest(root, manifest)
            actual_none = check_dataset(root)
            self.assertFalse(actual_none.ok)
            self.assertTrue(
                any("mean_occlusion_ratio" in error for error in actual_none.errors)
            )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = _write_records(root, count=0)
            manifest = _write_valid_manifest(root, records)
            manifest["mean_occlusion_ratio"] = 0.0
            _store_manifest(root, manifest)
            actual_float = check_dataset(root)
            self.assertFalse(actual_float.ok)
            self.assertTrue(
                any("mean_occlusion_ratio" in error for error in actual_float.errors)
            )

    def test_non_float_derived_values_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = _write_records(root, count=7)
            original = _write_valid_manifest(root, records)
            for value in ("0.2", 0, True, {}, []):
                with self.subTest(value=value):
                    manifest = dict(original)
                    manifest["mean_occlusion_ratio"] = value
                    _store_manifest(root, manifest)

                    report = check_dataset(root)

                    self.assertFalse(report.ok)
                    self.assertTrue(
                        any(
                            "mean_occlusion_ratio" in error
                            for error in report.errors
                        ),
                        report.errors,
                    )

    def test_integrity_fields_remain_exact(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = _write_records(root, count=7)
            original = _write_valid_manifest(root, records)
            wrong_split = "test"
            if next(iter(original["episode_splits"].values())) == wrong_split:
                wrong_split = "train"
            mutations = {
                "dataset_sha256": "0" * 64,
                "frame_count": int(original["frame_count"]) + 1,
                "episode_splits": {"episode_1": wrong_split},
                "yolo_model_sha256": "b" * 64,
            }
            for field, value in mutations.items():
                with self.subTest(field=field):
                    manifest = dict(original)
                    manifest[field] = value
                    _store_manifest(root, manifest)

                    report = check_dataset(root)

                    self.assertFalse(report.ok)
                    self.assertTrue(
                        any(field in error or "deployment SHA" in error for error in report.errors),
                        report.errors,
                    )


if __name__ == "__main__":
    unittest.main()
