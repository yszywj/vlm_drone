from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import AppConfig as LoaderAppConfig  # noqa: E402
from configs.loader import ConfigError, load_config  # noqa: E402
from configs.schema import AppConfig  # noqa: E402


class DefaultConfigTest(unittest.TestCase):
    def _load_mutated(self, mutate: object) -> None:
        raw = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
        mutate(raw)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            load_config(path)

    def test_loader_reexports_the_public_schema(self) -> None:
        self.assertIs(LoaderAppConfig, AppConfig)

    def test_default_config_is_valid_and_complete(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

        self.assertIsInstance(config, AppConfig)

        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.simulation.stage_units_in_meters, 1.0)
        self.assertEqual(config.simulation.physics_dt_s, config.simulation.rendering_dt_s)
        self.assertEqual(config.simulation.physics_dt_s, 1.0 / 60.0)
        self.assertEqual(config.scene.size_xyz_m, (100.0, 100.0, 30.0))
        self.assertEqual(config.uav.initial_position_xyz_m, (0.0, 0.0, 10.0))
        self.assertEqual(config.camera.resolution_wh_px, (640, 480))
        self.assertEqual(config.camera.frequency_hz, 10)
        self.assertIsNone(config.camera.focal_length_m)
        self.assertEqual(config.target.motion.mode, "RANDOM_WALK")
        self.assertLessEqual(config.target.motion.speed_mps, config.target.max_speed_mps)
        self.assertEqual(config.target.motion.region.min_xyz_m, (-40.0, -40.0, 0.5))
        self.assertEqual(config.search.transit_yaw_mode, "FACE_POINT")
        self.assertGreater(config.search.radius_m, 0.0)
        self.assertGreater(config.search.timeout_s, 0.0)
        self.assertEqual(config.experiment.name, "oracle_baseline")
        self.assertEqual(config.experiment.seed, 42)
        self.assertIsNone(config.experiment.output_root)
        self.assertTrue(config.tensorboard.scalars_only)
        self.assertFalse(config.checkpoint.save_periodic)
        self.assertFalse(config.checkpoint.save_full_base_model)
        self.assertTrue(config.checkpoint.save_adapter_only)
        self.assertTrue(config.evaluation.fixed_validation_seeds)
        self.assertTrue(config.evaluation.fixed_test_seeds)
        self.assertFalse(config.artifacts.save_images)
        self.assertFalse(config.artifacts.save_videos)
        self.assertFalse(config.artifacts.save_trajectories)
        self.assertEqual(config.figures.format, "png")
        self.assertGreater(
            config.storage.min_free_space_gb_before_start,
            config.storage.min_free_space_gb_during_run,
        )

    def test_heavy_artifacts_and_periodic_or_full_checkpoints_are_rejected(self) -> None:
        for section, key, value in (
            ("artifacts", "save_images", True),
            ("artifacts", "save_videos", True),
            ("checkpoint", "save_periodic", True),
            ("checkpoint", "save_full_base_model", True),
            ("tensorboard", "scalars_only", False),
        ):
            with self.subTest(section=section, key=key):
                with self.assertRaises(ConfigError):
                    self._load_mutated(
                        lambda raw, section=section, key=key, value=value: raw[section].__setitem__(key, value)
                    )

    def test_invalid_experiment_name_and_storage_relationships_are_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load_mutated(lambda raw: raw["experiment"].__setitem__("name", "bad/name"))
        with self.assertRaises(ConfigError):
            self._load_mutated(
                lambda raw: raw["storage"].update(
                    min_free_space_gb_before_start=5,
                    min_free_space_gb_during_run=10,
                )
            )


if __name__ == "__main__":
    unittest.main()
