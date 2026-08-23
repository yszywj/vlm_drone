from __future__ import annotations

import subprocess
import sys
import unittest

from common.obstacle_types import (
    CameraGeometry,
    FlightCorridor,
    ObstacleSpec,
)
from env.obstacle_registry import ObstacleRegistry
from perception.ideal_obstacle_perception import IdealObstaclePerception


def _spec(
    obstacle_id: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> ObstacleSpec:
    return ObstacleSpec(obstacle_id, center, size, (0.8, 0.2, 0.1))


def _camera(
    frame_id: str = "frame_1",
    *,
    timestamp_s: float = 1.0,
    far_clip_m: float = 100.0,
) -> CameraGeometry:
    return CameraGeometry(
        frame_id=frame_id,
        uav_id="uav_1",
        timestamp_s=timestamp_s,
        position_world_m=(0.0, 0.0, 0.0),
        orientation_world_from_camera_wxyz=(1.0, 0.0, 0.0, 0.0),
        resolution_wh_px=(640, 480),
        horizontal_fov_deg=90.0,
        near_clip_m=0.1,
        far_clip_m=far_clip_m,
    )


class IdealObstaclePerceptionTest(unittest.TestCase):
    def test_visible_obstacle_is_camera_flu_relative_and_privileged(self) -> None:
        backend = IdealObstaclePerception(
            ObstacleRegistry((_spec("box_red", (5.0, 1.0, 1.0)),))
        )
        observation = backend.observe(
            _camera(),
            observation_id="observation_1",
        )
        assert observation is not None
        self.assertEqual(observation.source, "ideal_camera_obstacle_perception")
        self.assertTrue(observation.privileged)
        self.assertEqual(observation.coordinate_frame, "CAMERA_FLU")
        visible = observation.visible_obstacles[0]
        self.assertEqual(visible.relative_center_m, (5.0, 1.0, 1.0))
        self.assertLess(sum(visible.bbox_xyxy_normalized[::2]) / 2.0, 0.5)
        self.assertLess(sum(visible.bbox_xyxy_normalized[1::2]) / 2.0, 0.5)
        encoded = observation.to_dict()
        self.assertNotIn("center_xyz_m", str(encoded))
        self.assertNotIn("world", str(encoded).lower())
        self.assertEqual(
            observation.manifest_fields()["obstacle_perception_privileged"],
            True,
        )

    def test_frustum_clipping_distance_and_area_filters_are_enforced(self) -> None:
        registry = ObstacleRegistry(
            (
                _spec("behind", (-5.0, 0.0, 0.0)),
                _spec("outside_fov", (5.0, 20.0, 0.0)),
                _spec("past_far", (30.0, 0.0, 0.0)),
                _spec("too_small", (10.0, 0.0, 0.0), (0.01, 0.01, 0.01)),
                _spec("visible", (5.0, 0.0, 0.0)),
            )
        )
        backend = IdealObstaclePerception(
            registry,
            max_distance_m=20.0,
            min_bbox_area_px=64,
        )
        result = backend.observe(
            _camera(far_clip_m=20.0),
            observation_id="observation_1",
        )
        assert result is not None
        self.assertEqual(
            tuple(item.obstacle_id for item in result.visible_obstacles),
            ("visible",),
        )

    def test_partial_screen_projection_is_retained(self) -> None:
        backend = IdealObstaclePerception(
            ObstacleRegistry((_spec("edge", (5.0, -5.0, 0.0), (3.0, 3.0, 2.0)),)),
            min_bbox_area_px=16,
        )
        result = backend.observe(_camera(), observation_id="observation_1")
        assert result is not None
        self.assertEqual(len(result.visible_obstacles), 1)
        self.assertEqual(result.visible_obstacles[0].bbox_xyxy_normalized[2], 1.0)

    def test_nearer_projected_box_occludes_far_box(self) -> None:
        backend = IdealObstaclePerception(
            ObstacleRegistry(
                (
                    _spec("near", (4.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                    _spec("far", (8.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
                )
            ),
            max_occlusion_ratio=0.8,
        )
        result = backend.observe(_camera(), observation_id="observation_1")
        assert result is not None
        self.assertEqual(
            tuple(item.obstacle_id for item in result.visible_obstacles),
            ("near",),
        )

    def test_active_corridor_and_ttc_are_computed_but_visibility_alone_is_not_hazard(self) -> None:
        backend = IdealObstaclePerception(
            ObstacleRegistry(
                (
                    _spec("blocking", (5.0, 0.0, 0.0)),
                    _spec("visible_only", (5.0, 3.0, 0.0), (1.0, 1.0, 1.0)),
                )
            )
        )
        result = backend.observe(
            _camera(),
            active_corridor=FlightCorridor((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), 0.2),
            uav_velocity_world_mps=(2.0, 0.0, 0.0),
            observation_id="observation_1",
        )
        assert result is not None
        by_id = {item.obstacle_id: item for item in result.visible_obstacles}
        self.assertTrue(by_id["blocking"].active_corridor_intersection)
        self.assertAlmostEqual(by_id["blocking"].time_to_collision_s or -1.0, 1.9)
        self.assertFalse(by_id["visible_only"].active_corridor_intersection)
        self.assertIsNone(by_id["visible_only"].time_to_collision_s)

    def test_only_fresh_camera_frames_emit_and_reset_reopens_stream(self) -> None:
        backend = IdealObstaclePerception(ObstacleRegistry())
        first = backend.observe(_camera(), observation_id="observation_1")
        self.assertIsNotNone(first)
        self.assertIsNone(backend.observe(_camera(), observation_id="observation_2"))
        self.assertIsNone(
            backend.observe(
                _camera("frame_2", timestamp_s=0.5),
                observation_id="observation_2",
            )
        )
        backend.reset(uav_id="uav_1")
        self.assertIsNotNone(
            backend.observe(_camera(), observation_id="observation_3")
        )

    def test_module_import_does_not_cross_isaac_boundary(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import perception.ideal_obstacle_perception; "
                "assert 'isaacsim' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
