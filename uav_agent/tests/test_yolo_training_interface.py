from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.yolo.config import (  # noqa: E402
    YoloTrainConfig,
    YoloTrainingConfigError,
    load_yolo_train_config,
)
from training.yolo.registry import sha256_file, write_validation_report  # noqa: E402
from training.yolo.trainer import (  # noqa: E402
    UltralyticsTrainingBackend,
    YoloTrainingError,
    YoloTrainingBackend,
)


class _FakeModel:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.names = {0: "person"}
        self.train_kwargs: dict[str, object] | None = None
        self.export_kwargs: dict[str, object] | None = None

    def train(self, **kwargs: object) -> object:
        self.train_kwargs = kwargs
        weights = self.run_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        (weights / "epoch001.pt").write_bytes(b"intermediate")
        (self.run_dir / "results.csv").write_text(
            "epoch,metrics/mAP50(B)\n0,0.8\n", encoding="utf-8"
        )
        (self.run_dir / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
        (self.run_dir / "confusion_matrix.png").write_bytes(b"figure")
        return SimpleNamespace(
            save_dir=self.run_dir,
            results_dict={"metrics/mAP50(B)": 0.8},
        )

    def val(self, **kwargs: object) -> object:
        box = SimpleNamespace(
            map50=0.8,
            map=0.6,
            mp=0.7,
            mr=0.65,
            maps=[0.6],
            ap_class_index=[0],
            class_result=lambda index: (0.71, 0.66, 0.81, 0.61),
        )
        return SimpleNamespace(
            box=box,
            speed={"inference": 4.5},
            results_dict={
                "metrics/mAP50(B)": 0.8,
                "metrics/mAP50-95(B)": 0.6,
                "metrics/precision(B)": 0.7,
                "metrics/recall(B)": 0.65,
                "metrics/precision(small)": 0.51,
                "metrics/recall(small)": 0.52,
                "metrics/mAP50(small)": 0.53,
                "metrics/mAP50-95(small)": 0.31,
            },
        )

    def predict(self, **kwargs: object) -> list[object]:
        return [SimpleNamespace(boxes=[])]

    def export(self, **kwargs: object) -> str:
        self.export_kwargs = kwargs
        path = self.run_dir / ("model.onnx" if kwargs["format"] == "onnx" else "model.engine")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"export")
        return str(path)


class YoloTrainingInterfaceTest(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path]:
        model = root / "base.pt"
        model.write_bytes(b"model")
        for split, color in (
            ("train", (1, 2, 3)),
            ("val", (4, 5, 6)),
            ("test", (7, 8, 9)),
        ):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            Image.new("RGB", (16, 16), color=color).save(root / f"images/{split}/a.png")
            (root / f"labels/{split}/a.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
            )
        data = root / "data.yaml"
        data.write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {0: "person"},
                }
            ),
            encoding="utf-8",
        )
        (root / "manifest.jsonl").write_text(
            '{"sample_id":"train/a"}\n{"sample_id":"val/a"}\n'
            '{"sample_id":"test/a"}\n',
            encoding="utf-8",
        )
        return model, data

    @staticmethod
    def _passing_validation_payload(model_path: Path) -> dict[str, object]:
        return {
            "schema_version": 1,
            "validated_at": "2026-01-01T00:00:00Z",
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "dataset_yaml": "/tmp/data.yaml",
            "model_family": "yolo",
            "task": "detect",
            "passed": True,
            "mAP50": 0.8,
            "mAP50-95": 0.6,
            "precision": 0.7,
            "recall": 0.65,
            "per_class": {"person": {"mAP50-95": 0.6}},
            "small_target_metrics": {
                "available": True,
                "precision": 0.51,
                "recall": 0.52,
                "mAP50": 0.53,
                "mAP50-95": 0.31,
            },
            "latency_ms": {"inference": 4.5},
        }

    def test_config_precedence_and_explicit_resume_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "train.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model_family": "yolo",
                        "task": "detect",
                        "epochs": 10,
                        "device": "cpu",
                        "project_dir": str(root / "output"),
                        "run_name": "yaml-name",
                    }
                ),
                encoding="utf-8",
            )
            config = load_yolo_train_config(
                config_path,
                environ={"UAV_AGENT_YOLO_EPOCHS": "20", "UAV_AGENT_YOLO_RUN_NAME": "env-name"},
                overrides={"epochs": 30, "run_name": "cli-name"},
            )
            self.assertEqual(config.epochs, 30)
            self.assertEqual(config.run_name, "cli-name")
            with self.assertRaises(YoloTrainingConfigError):
                YoloTrainConfig(resume=root / "best.pt")

    def test_dry_run_and_injected_backend_train_validate_predict_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, data_yaml = self._inputs(root)
            run_dir = root / "outputs" / "unit"
            fake = _FakeModel(run_dir)
            backend = UltralyticsTrainingBackend(
                model_factory=lambda path, family, task: fake
            )
            self.assertIsInstance(backend, YoloTrainingBackend)
            config = YoloTrainConfig(
                base_model_path=model_path,
                dataset_yaml=data_yaml,
                epochs=1,
                imgsz=64,
                batch=1,
                device="cpu",
                workers=0,
                project_dir=root / "outputs",
                run_name="unit",
            )

            preflight = backend.preflight(config)
            self.assertTrue(preflight.ok, preflight.diagnostics)
            result = backend.train(config)
            self.assertTrue(result.best_model_path.is_file())
            self.assertTrue(result.last_model_path.is_file())
            self.assertTrue(result.model_manifest_path.is_file())
            manifest = json.loads(result.model_manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(
                {
                    "training",
                    "git_commit",
                    "base_model",
                    "best_model",
                    "dataset",
                    "training_parameters",
                    "versions",
                    "validation_metrics",
                }.issubset(manifest)
            )
            self.assertTrue(
                {"started_at", "finished_at", "elapsed_s"}.issubset(
                    manifest["training"]
                )
            )
            self.assertEqual(manifest["base_model"]["sha256"], sha256_file(model_path))
            self.assertEqual(
                manifest["best_model"]["sha256"], sha256_file(result.best_model_path)
            )
            self.assertEqual(manifest["versions"]["ultralytics"], "injected")
            self.assertTrue(manifest["dataset"]["manifest_path"].endswith("manifest.jsonl"))
            self.assertEqual(manifest["dataset"]["classes"], ["person"])
            self.assertEqual(fake.train_kwargs["save_period"], -1)
            self.assertTrue((run_dir / "figures/confusion_matrix.png").is_file())
            self.assertEqual(
                sorted(path.name for path in (run_dir / "weights").glob("*.pt")),
                ["best.pt", "last.pt"],
            )

            validation = backend.validate(
                model_path=model_path,
                dataset_yaml=data_yaml,
                device="cpu",
                imgsz=64,
            )
            self.assertTrue(validation.passed)
            self.assertEqual(validation.map50, 0.8)
            self.assertEqual(validation.latency_ms["inference"], 4.5)
            self.assertEqual(validation.per_class["person"]["precision"], 0.71)
            self.assertEqual(validation.per_class["person"]["mAP50"], 0.81)
            self.assertTrue(validation.small_target_metrics["available"])
            self.assertEqual(validation.small_target_metrics["mAP50-95"], 0.31)
            self.assertEqual(
                validation.small_target_metrics["validation_ground_truth"]["small_count"],
                1,
            )
            validation_path = write_validation_report(
                root / "validation.json", validation.to_dict()
            )
            prediction = backend.predict(
                model_path=model_path,
                source=root / "images/val/a.png",
                device="cpu",
            )
            self.assertEqual(prediction.image_count, 1)
            exported = backend.export(
                model_path=model_path,
                validation_report=validation_path,
                format="onnx",
                device="cpu",
            )
            self.assertTrue(exported.exported_path.is_file())
            self.assertTrue(exported.export_manifest_path.is_file())
            export_manifest = json.loads(
                exported.export_manifest_path.read_text(encoding="utf-8")
            )
            self.assertFalse(exported.dynamic_prompts_supported)
            self.assertFalse(export_manifest["dynamic_prompts_supported"])
            self.assertFalse(export_manifest["yoloe_prompts_statically_frozen"])
            self.assertFalse(fake.export_kwargs["simplify"])

    def test_export_rejects_wrong_hash_and_unacknowledged_yoloe_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, _ = self._inputs(root)
            fake = _FakeModel(root / "output")
            backend = UltralyticsTrainingBackend(
                model_factory=lambda path, family, task: fake
            )
            wrong = write_validation_report(
                root / "wrong.json",
                {
                    **self._passing_validation_payload(model_path),
                    "model_sha256": "0" * 64,
                },
            )
            with self.assertRaises(YoloTrainingError):
                backend.export(
                    model_path=model_path,
                    validation_report=wrong,
                    format="onnx",
                    device="cpu",
                )
            valid = write_validation_report(
                root / "valid.json",
                {
                    **self._passing_validation_payload(model_path),
                    "model_family": "yoloe",
                },
            )
            with self.assertRaisesRegex(YoloTrainingError, "statically freezes prompts"):
                backend.export(
                    model_path=model_path,
                    validation_report=valid,
                    format="onnx",
                    device="cpu",
                    model_family="yoloe",
                )
            frozen = backend.export(
                model_path=model_path,
                validation_report=valid,
                format="onnx",
                device="cpu",
                model_family="yoloe",
                freeze_yoloe_prompts=True,
            )
            self.assertFalse(frozen.dynamic_prompts_supported)
            frozen_manifest = json.loads(
                frozen.export_manifest_path.read_text(encoding="utf-8")
            )
            self.assertTrue(frozen_manifest["yoloe_prompts_statically_frozen"])
            self.assertEqual(frozen_manifest["frozen_classes"], ["person"])

    def test_export_rejects_incomplete_passing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, _ = self._inputs(root)
            backend = UltralyticsTrainingBackend(
                model_factory=lambda path, family, task: _FakeModel(root / "output")
            )
            incomplete = write_validation_report(
                root / "incomplete.json",
                {"passed": True, "model_sha256": sha256_file(model_path)},
            )
            with self.assertRaisesRegex(YoloTrainingError, "incomplete"):
                backend.export(
                    model_path=model_path,
                    validation_report=incomplete,
                    format="onnx",
                    device="cpu",
                )

    def test_export_rejects_unavailable_small_target_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, _ = self._inputs(root)
            backend = UltralyticsTrainingBackend(
                model_factory=lambda path, family, task: _FakeModel(root / "output")
            )
            report = write_validation_report(
                root / "no-small.json",
                {
                    **self._passing_validation_payload(model_path),
                    "small_target_metrics": {"available": False},
                },
            )
            with self.assertRaisesRegex(YoloTrainingError, "small-target metrics"):
                backend.export(
                    model_path=model_path,
                    validation_report=report,
                    format="onnx",
                    device="cpu",
                )

    def test_sparse_per_class_metrics_follow_ap_class_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.pt"
            model.write_bytes(b"model")
            box = SimpleNamespace(
                map50=0.8,
                map=0.6,
                mp=0.7,
                mr=0.65,
                maps=[0.6],
                ap_class_index=[1],
                class_result=lambda result_index: (0.2, 0.3, 0.4, 0.5),
            )
            result = SimpleNamespace(
                box=box,
                speed={},
                results_dict={
                    "metrics/precision(small)": 0.1,
                    "metrics/recall(small)": 0.2,
                    "metrics/mAP50(small)": 0.3,
                    "metrics/mAP50-95(small)": 0.4,
                },
            )

            parsed = UltralyticsTrainingBackend._parse_validation_result(
                result,
                model_path=model,
                dataset_yaml=root / "data.yaml",
                model_family="yolo",
                task="detect",
                class_names=("red_cube", "person"),
                validation_target_area_px={},
                validation_image_count=1,
                total_elapsed_ms=1.0,
            )

            self.assertIsNone(parsed.per_class["red_cube"]["precision"])
            self.assertEqual(parsed.per_class["person"]["precision"], 0.2)
            self.assertEqual(parsed.per_class["person"]["mAP50-95"], 0.5)

    def test_tensorrt_export_runtime_is_optional_and_fails_fast(self) -> None:
        backend = UltralyticsTrainingBackend()

        def fake_import(name: str) -> object:
            if name == "onnx":
                return object()
            if name == "torch":
                return SimpleNamespace(
                    cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
                )
            if name == "tensorrt":
                raise ImportError("not installed")
            raise AssertionError(name)

        with patch("training.yolo.trainer.importlib.import_module", side_effect=fake_import):
            with self.assertRaisesRegex(YoloTrainingError, "optional.*not installed"):
                backend._require_export_runtime("engine", 0)

    def test_onnx_export_runtime_fails_before_ultralytics_autoinstall(self) -> None:
        backend = UltralyticsTrainingBackend()
        with patch(
            "training.yolo.trainer.importlib.import_module",
            side_effect=ImportError("onnx missing"),
        ):
            with self.assertRaisesRegex(YoloTrainingError, "onnx==1.22.0"):
                backend._require_export_runtime("onnx", "cpu")

    def test_preflight_rejects_existing_non_resume_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, data_yaml = self._inputs(root)
            run_dir = root / "outputs" / "already-there"
            run_dir.mkdir(parents=True)
            config = YoloTrainConfig(
                base_model_path=model_path,
                dataset_yaml=data_yaml,
                device="cpu",
                project_dir=root / "outputs",
                run_name="already-there",
            )
            preflight = UltralyticsTrainingBackend(
                model_factory=lambda path, family, task: _FakeModel(run_dir)
            ).preflight(config)
            self.assertFalse(preflight.ok)
            self.assertTrue(
                any("already exists" in message for message in preflight.diagnostics)
            )

    def test_preflight_loads_checkpoint_and_reports_incompatible_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, data_yaml = self._inputs(root)
            config = YoloTrainConfig(
                base_model_path=model_path,
                dataset_yaml=data_yaml,
                device="cpu",
                project_dir=root / "outputs",
                run_name="model-check",
            )

            def reject_model(path: Path, family: str, task: str) -> object:
                raise RuntimeError("incompatible checkpoint payload")

            preflight = UltralyticsTrainingBackend(
                model_factory=reject_model
            ).preflight(config)
            self.assertFalse(preflight.ok)
            self.assertTrue(
                any("incompatible checkpoint" in message for message in preflight.diagnostics)
            )

    def test_modules_do_not_import_isaac(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                PROJECT_ROOT / "training/yolo/config.py",
                PROJECT_ROOT / "training/yolo/dataset.py",
                PROJECT_ROOT / "training/yolo/trainer.py",
                PROJECT_ROOT / "training/yolo/registry.py",
                PROJECT_ROOT / "scripts/train_yolo.py",
            )
        )
        self.assertNotIn("isaacsim", source.lower())
        self.assertNotIn("omni.", source.lower())


if __name__ == "__main__":
    unittest.main()
