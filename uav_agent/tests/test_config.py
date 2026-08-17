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
from configs.schema import PlannerConfig  # noqa: E402


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
        self.assertEqual(config.planner, PlannerConfig())
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

    def test_legacy_config_without_planner_uses_trusted_defaults(self) -> None:
        config: AppConfig | None = None

        def remove_planner(raw: dict[str, object]) -> None:
            raw.pop("planner")

        raw = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        )
        remove_planner(raw)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.planner, PlannerConfig())

    def test_planner_limits_are_strictly_validated(self) -> None:
        mutations = (
            ("max_plan_steps", 1),
            ("max_plan_steps", 11),
            ("max_plan_steps", True),
            ("max_goto_calls", 0),
            ("max_goto_calls", 6),
            ("max_search_calls", 2),
            ("max_track_calls", 0),
            ("max_track_calls", 3),
            ("max_reacquire_attempts_per_track", 5),
            ("max_total_reacquire_attempts", 0),
            ("max_total_reacquire_attempts", 5),
            ("min_track_duration_s", 0.0),
            ("max_track_duration_s", float("inf")),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ConfigError):
                    self._load_mutated(
                        lambda raw, key=key, value=value: raw["planner"].__setitem__(
                            key, value
                        )
                    )

        with self.assertRaises(ConfigError):
            self._load_mutated(
                lambda raw: raw["planner"].update(
                    min_track_duration_s=30.0,
                    max_track_duration_s=10.0,
                )
            )
        with self.assertRaises(ConfigError):
            self._load_mutated(
                lambda raw: raw["planner"].__setitem__("unknown", 1)
            )
        with self.assertRaises(ConfigError):
            self._load_mutated(lambda raw: raw.__setitem__("planner", None))

    def test_planner_config_direct_construction_is_strict(self) -> None:
        for kwargs in (
            {"max_plan_steps": True},
            {"max_plan_steps": 1},
            {"max_plan_steps": 11},
            {"max_search_calls": 2},
            {"min_track_duration_s": float("nan")},
            {"min_track_duration_s": 20.0, "max_track_duration_s": 10.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PlannerConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
