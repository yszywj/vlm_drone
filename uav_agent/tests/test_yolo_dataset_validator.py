from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.yolo.dataset import YoloDatasetValidator  # noqa: E402


class YoloDatasetValidatorTest(unittest.TestCase):
    def _dataset(self, root: Path, *, include_test: bool = True) -> Path:
        for split in ("train", "val") + (("test",) if include_test else ()):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        descriptor = {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "red_cube", 1: "person"},
        }
        if include_test:
            descriptor["test"] = "images/test"
        data_yaml = root / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
        if include_test:
            self._image(root / "images/test/baseline.png", (91, 92, 93))
            (root / "labels/test/baseline.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        return data_yaml

    @staticmethod
    def _image(path: Path, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (20, 10), color=color).save(path)

    def test_valid_dataset_reports_distribution_area_and_empty_background(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_yaml = self._dataset(root)
            self._image(root / "images/train/object.png", (255, 0, 0))
            self._image(root / "images/train/background.png", (0, 255, 0))
            self._image(root / "images/val/person.png", (0, 0, 255))
            (root / "labels/train/object.txt").write_text(
                "0 0.5 0.5 0.4 0.6\n", encoding="utf-8"
            )
            (root / "labels/train/background.txt").write_text("", encoding="utf-8")
            (root / "labels/val/person.txt").write_text(
                "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            label_before = (root / "labels/train/object.txt").read_bytes()

            report = YoloDatasetValidator().validate(data_yaml)

            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.split_image_counts, {"train": 2, "val": 1, "test": 1})
            self.assertEqual(report.split_annotation_counts["train"], 1)
            self.assertEqual(report.class_counts, {"red_cube": 2, "person": 1})
            self.assertEqual(report.target_area_px["count"], 3)
            self.assertEqual(report.target_area_px["small_count"], 3)
            self.assertEqual(report.target_area_px["medium_count"], 0)
            self.assertEqual(report.target_area_px["large_count"], 0)
            self.assertEqual(report.target_area_px_by_split["train"]["count"], 1)
            self.assertEqual(report.target_area_px_by_split["val"]["small_count"], 1)
            self.assertEqual(report.empty_label_files, 1)
            self.assertIn("empty_label", {issue.code for issue in report.warnings})

            statistics_path = report.write_statistics()
            payload = json.loads(statistics_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["target_area_px"]["count"], 3)
            self.assertEqual(
                (root / "labels/train/object.txt").read_bytes(), label_before
            )

    def test_bad_image_missing_label_orphan_and_invalid_labels_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_yaml = self._dataset(root)
            (root / "images/train/broken.png").write_bytes(b"not an image")
            (root / "labels/train/broken.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            self._image(root / "images/train/missing.png", (1, 2, 3))
            self._image(root / "images/train/invalid.png", (4, 5, 6))
            (root / "labels/train/invalid.txt").write_text(
                "2 0.5 0.5 0.2 0.2\n"
                "0 0.5 0.5 0.0 0.2\n"
                "0 1.2 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            (root / "labels/train/orphan.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            self._image(root / "images/val/valid.png", (7, 8, 9))
            (root / "labels/val/valid.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )

            report = YoloDatasetValidator().validate(data_yaml)

            self.assertFalse(report.ok)
            codes = {issue.code for issue in report.errors}
            self.assertTrue(
                {
                    "unreadable_image",
                    "missing_label",
                    "orphan_label",
                    "class_id_out_of_range",
                    "non_positive_box",
                    "coordinate_out_of_range",
                }.issubset(codes),
                codes,
            )

    def test_duplicate_content_and_cross_split_hash_leakage_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_yaml = self._dataset(root)
            image = Image.new("RGB", (12, 12), color=(12, 34, 56))
            image.save(root / "images/train/a.png")
            image.save(root / "images/train/b.png")
            image.save(root / "images/val/c.png")
            for path in (
                root / "labels/train/a.txt",
                root / "labels/train/b.txt",
                root / "labels/val/c.txt",
            ):
                path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            report = YoloDatasetValidator().validate(data_yaml)

            codes = {issue.code for issue in report.errors}
            self.assertIn("split_hash_leakage", codes)

    def test_box_extending_outside_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_yaml = self._dataset(root)
            self._image(root / "images/train/a.png", (1, 0, 0))
            self._image(root / "images/val/b.png", (0, 1, 0))
            (root / "labels/train/a.txt").write_text(
                "0 0.1 0.5 0.4 0.2\n", encoding="utf-8"
            )
            (root / "labels/val/b.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )

            report = YoloDatasetValidator().validate(data_yaml)

            self.assertIn("box_outside_image", {issue.code for issue in report.errors})

    def test_test_split_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = YoloDatasetValidator().validate(
                self._dataset(Path(temporary), include_test=False)
            )

            self.assertIn("missing_split", {issue.code for issue in report.errors})

    def test_dataset_with_only_empty_labels_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_yaml = self._dataset(root)
            (root / "labels/test/baseline.txt").write_text("", encoding="utf-8")
            for split, color in (("train", (11, 12, 13)), ("val", (21, 22, 23))):
                self._image(root / f"images/{split}/background.png", color)
                (root / f"labels/{split}/background.txt").write_text("", encoding="utf-8")

            report = YoloDatasetValidator().validate(data_yaml)

            self.assertIn(
                "no_positive_annotations", {issue.code for issue in report.errors}
            )


if __name__ == "__main__":
    unittest.main()
