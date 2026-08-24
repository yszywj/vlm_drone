from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CameraSensorRgbdSourceTest(unittest.TestCase):
    """Verify Isaac integration without importing Isaac before SimulationApp."""

    def test_sensor_uses_isaac_51_image_plane_depth_api_and_atomic_clone(self) -> None:
        source = (PROJECT_ROOT / "env" / "camera_sensor.py").read_text(encoding="utf-8")
        self.assertIn("camera.add_distance_to_image_plane_to_frame()", source)
        self.assertIn('frame.get("distance_to_image_plane")', source)
        self.assertIn("get_current_frame(clone=True)", source)
        self.assertNotIn("add_distance_to_camera_to_frame", source)
        self.assertIn("RGB/depth resolution mismatch", source)
        self.assertIn("depth[invalid] = np.nan", source)
        self.assertIn("CameraFrameNotReady", source)

    def test_environment_enables_depth_only_after_world_reset(self) -> None:
        fleet_source = (PROJECT_ROOT / "env" / "fleet_uav_search_env.py").read_text(
            encoding="utf-8"
        )
        reset_index = fleet_source.index("world.reset()")
        enable_index = fleet_source.index("sensor.enable_depth()")
        self.assertLess(reset_index, enable_index)
        self.assertIn("sample = self._require_camera_sensor(uav_id).get_sample()", fleet_source)
        self.assertIn("camera_sample=sample", fleet_source)

        singleton_source = (
            PROJECT_ROOT / "env" / "simple_uav_search_env.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self._fleet_env.setup()", singleton_source)
        self.assertNotIn("from isaacsim", singleton_source)

    def test_collector_script_imports_simulation_app_only_inside_main(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "collect_yolo_dataset.py").read_text(
            encoding="utf-8"
        )
        gate_index = source.index("require_oracle_label_acknowledgements(")
        simulation_index = source.index("from isaacsim import SimulationApp")
        self.assertLess(gate_index, simulation_index)
        prefix = source[:simulation_index]
        self.assertNotIn("from env.simple_uav_search_env", prefix)


if __name__ == "__main__":
    unittest.main()
