from __future__ import annotations

import unittest

import numpy as np

from env.camera_types import CameraIntrinsics, CameraSample


class CameraSampleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(
            fx=120.0,
            fy=121.0,
            cx=1.0,
            cy=1.0,
            width=3,
            height=2,
        )

    def test_sample_owns_read_only_synchronized_arrays(self) -> None:
        rgb = np.full((2, 3, 3), 7, dtype=np.uint8)
        depth = np.full((2, 3), 4.0, dtype=np.float32)
        sample = CameraSample(
            timestamp_s=1.5,
            rgb=rgb,
            depth_to_image_plane_m=depth,
            camera_position_world_m=(1.0, 2.0, 3.0),
            camera_orientation_world_wxyz=(2.0, 0.0, 0.0, 0.0),
            intrinsics=self.intrinsics,
            render_frame_id=(90, 60),
        )
        rgb[:] = 99
        depth[:] = 99.0

        self.assertTrue(np.all(sample.rgb == 7))
        self.assertTrue(np.all(sample.depth_to_image_plane_m == 4.0))
        self.assertFalse(sample.rgb.flags.writeable)
        self.assertFalse(sample.depth_to_image_plane_m.flags.writeable)
        self.assertEqual(sample.camera_orientation_world_wxyz, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(sample.render_frame_id, (90, 60))

    def test_nonpositive_and_nonfinite_depth_are_invalid_not_fake_zero(self) -> None:
        depth = np.asarray([[0.0, np.inf, 2.0], [np.nan, -1.0, 20.0]])
        sample = CameraSample(
            timestamp_s=0.0,
            rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            depth_to_image_plane_m=depth,
            camera_position_world_m=(0.0, 0.0, 0.0),
            camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            intrinsics=self.intrinsics,
        )
        assert sample.depth_to_image_plane_m is not None
        self.assertTrue(np.isnan(sample.depth_to_image_plane_m[0, 0]))
        self.assertTrue(np.isnan(sample.depth_to_image_plane_m[0, 1]))
        mask = sample.valid_depth_mask(min_depth_m=1.0, max_depth_m=10.0)
        assert mask is not None
        self.assertEqual(int(mask.sum()), 1)

    def test_resolution_and_intrinsics_are_strict(self) -> None:
        with self.assertRaises(ValueError):
            CameraSample(
                timestamp_s=0.0,
                rgb=np.zeros((3, 3, 3), dtype=np.uint8),
                depth_to_image_plane_m=None,
                camera_position_world_m=(0.0, 0.0, 0.0),
                camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
                intrinsics=self.intrinsics,
            )
        with self.assertRaises(ValueError):
            CameraIntrinsics(fx=0.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1)
        with self.assertRaises(ValueError):
            CameraSample(
                timestamp_s=0.0,
                rgb=np.zeros((2, 3, 3), dtype=np.uint8),
                depth_to_image_plane_m=None,
                camera_position_world_m=(0.0, 0.0, 0.0),
                camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
                intrinsics=self.intrinsics,
                render_frame_id=(1, 0),
            )


if __name__ == "__main__":
    unittest.main()
