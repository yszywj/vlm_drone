from __future__ import annotations

import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

from configs.loader import ConfigError, load_config
from configs.schema import (
    DebugImagesConfig,
    FrameStoreConfig,
    ModelWorkerConfig,
    PlanRevisionConfig,
    QwenVisualReviewConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class VisualRuntimeConfigTest(unittest.TestCase):
    def _raw(self) -> dict[str, object]:
        value = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)
        return value

    def _load(self, raw: dict[str, object]):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            return load_config(path)

    def test_default_blocks_are_loaded(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config.model_worker, ModelWorkerConfig())
        self.assertGreaterEqual(config.model_worker.request_timeout_s, 60.0)
        self.assertEqual(config.qwen_visual_review, QwenVisualReviewConfig())
        self.assertEqual(config.plan_revision, PlanRevisionConfig())
        self.assertEqual(config.frame_store, FrameStoreConfig())
        self.assertEqual(config.debug_images, DebugImagesConfig())
        self.assertFalse(config.qwen_visual_review.enabled)
        self.assertEqual(config.qwen_visual_review.mode, "shadow")
        self.assertLessEqual(config.qwen_visual_review.max_recent_frames, 3)

    def test_config_loader_imports_in_a_cold_interpreter(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from configs import load_config; "
                    "print(load_config('configs/default.yaml').uav.id)"
                ),
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "uav_1")

    def test_legacy_config_inherits_all_new_trusted_defaults(self) -> None:
        raw = self._raw()
        for name in (
            "model_worker",
            "qwen_visual_review",
            "plan_revision",
            "frame_store",
            "debug_images",
        ):
            raw.pop(name)
        config = self._load(raw)
        self.assertEqual(config.model_worker, ModelWorkerConfig())
        self.assertEqual(config.qwen_visual_review, QwenVisualReviewConfig())
        self.assertEqual(config.plan_revision, PlanRevisionConfig())
        self.assertEqual(config.frame_store, FrameStoreConfig())
        self.assertEqual(config.debug_images, DebugImagesConfig())

    def test_explicit_blocks_require_exact_keys(self) -> None:
        for section in (
            "model_worker",
            "qwen_visual_review",
            "plan_revision",
            "frame_store",
            "debug_images",
        ):
            with self.subTest(section=section):
                raw = self._raw()
                raw[section].pop(next(iter(raw[section])))  # type: ignore[union-attr]
                with self.assertRaises(ConfigError):
                    self._load(raw)
                raw = self._raw()
                raw[section]["typo"] = 1  # type: ignore[index]
                with self.assertRaises(ConfigError):
                    self._load(raw)

    def test_unbounded_worker_and_visual_values_are_rejected(self) -> None:
        mutations = (
            ("model_worker", "max_inflight_per_uav", 2),
            ("model_worker", "request_timeout_s", float("inf")),
            ("model_worker", "request_timeout_s", 301.0),
            ("qwen_visual_review", "mode", "active"),
            ("qwen_visual_review", "goto_interval_s", float("inf")),
            ("qwen_visual_review", "max_recent_frames", 4),
            ("qwen_visual_review", "max_image_side_px", 4097),
            ("qwen_visual_review", "jpeg_quality", 96),
            ("qwen_visual_review", "jpeg_quality", 100),
            ("qwen_visual_review", "hover_position_tolerance_m", 5.01),
            ("qwen_visual_review", "hover_max_correction_speed_mps", 10.01),
            ("qwen_visual_review", "blocking_hover_timeout_s", 0),
            ("qwen_visual_review", "blocking_timeout_fallback", "CONTINUE"),
            ("qwen_visual_review", "enabled", "false"),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key, value=value):
                raw = self._raw()
                raw[section][key] = value  # type: ignore[index]
                with self.assertRaises(ConfigError):
                    self._load(raw)

    def test_revision_frame_and_debug_caps_are_rejected(self) -> None:
        mutations = (
            ("plan_revision", "max_revisions", 4),
            ("plan_revision", "cooldown_s", float("inf")),
            ("frame_store", "max_frames", 0),
            ("frame_store", "max_bytes", 1_073_741_825),
            ("frame_store", "max_age_s", float("inf")),
            ("debug_images", "max_images_per_run", -1),
            ("debug_images", "max_images_per_run", 10_001),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key, value=value):
                raw = self._raw()
                raw[section][key] = value  # type: ignore[index]
                with self.assertRaises(ConfigError):
                    self._load(raw)

        raw = self._raw()
        raw["plan_revision"]["enabled"] = True  # type: ignore[index]
        raw["plan_revision"]["max_revisions"] = 0  # type: ignore[index]
        with self.assertRaises(ConfigError):
            self._load(raw)


if __name__ == "__main__":
    unittest.main()
