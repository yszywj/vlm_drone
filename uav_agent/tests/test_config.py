from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config  # noqa: E402


class DefaultConfigTest(unittest.TestCase):
    def test_default_config_is_valid_and_complete(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

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


if __name__ == "__main__":
    unittest.main()
