from __future__ import annotations

import unittest

import numpy as np

from env.camera_types import CameraIntrinsics, CameraSample
from runtime.frame_store import FrameStore


def _sample(timestamp_s: float, value: int = 1) -> CameraSample:
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=np.full((2, 2, 3), value, dtype=np.uint8),
        depth_to_image_plane_m=np.full((2, 2), 3.0, dtype=np.float32),
        camera_position_world_m=(1.0, 2.0, 3.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=CameraIntrinsics(
            fx=10.0,
            fy=10.0,
            cx=0.5,
            cy=0.5,
            width=2,
            height=2,
        ),
    )


class FrameStoreDepthTest(unittest.TestCase):
    def test_uav_self_motion_is_synchronized_and_all_or_none(self) -> None:
        store = FrameStore(max_frames=2, max_bytes=10_000, max_age_s=10.0)
        camera = _sample(1.0)
        ref = store.add_sample(
            uav_id="uav_1",
            frame_id="frame_motion",
            sample=camera,
            uav_linear_velocity_world_mps=(1.0, 2.0, 3.0),
            uav_angular_velocity_body_radps=(0.1, 0.2, 0.3),
        )

        motion = store.get_uav_self_motion(ref)
        self.assertIsNotNone(motion)
        assert motion is not None
        self.assertEqual(motion.linear_velocity_world_mps, (1.0, 2.0, 3.0))
        self.assertEqual(motion.angular_velocity_body_radps, (0.1, 0.2, 0.3))
        synchronized = store.get_temporal_inputs(ref)
        self.assertIsNotNone(synchronized)
        assert synchronized is not None
        rgb, depth, geometry, synchronized_motion = synchronized
        self.assertFalse(rgb.flags.writeable)
        self.assertFalse(depth.flags.writeable)
        self.assertEqual(geometry.timestamp_s, ref.timestamp_s)
        self.assertEqual(synchronized_motion, motion)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            store.add_sample(
                uav_id="uav_1",
                frame_id="frame_bad",
                sample=_sample(2.0),
                uav_linear_velocity_world_mps=(1.0, 2.0, 3.0),
            )

    def test_rgb_depth_and_geometry_share_one_reference(self) -> None:
        store = FrameStore(max_frames=2, max_bytes=100, max_age_s=10.0)
        sample = _sample(1.0)
        ref = store.add_sample(uav_id="uav_1", frame_id="frame_1", sample=sample)

        self.assertEqual(ref.timestamp_s, sample.timestamp_s)
        self.assertEqual(store.total_bytes, sample.rgb.nbytes + sample.depth_to_image_plane_m.nbytes)
        depth = store.get_depth(ref)
        geometry = store.get_camera_geometry(ref)
        assert depth is not None and geometry is not None
        depth[:] = 99.0
        self.assertTrue(np.all(store.get_depth(ref) == 3.0))
        self.assertEqual(geometry.timestamp_s, ref.timestamp_s)
        self.assertEqual(geometry.intrinsics, sample.intrinsics)
        self.assertEqual(geometry.camera_position_world_m, (1.0, 2.0, 3.0))

    def test_depth_bytes_participate_in_hard_eviction_bound(self) -> None:
        # Each sample is 12 RGB bytes plus 16 float32 depth bytes.
        store = FrameStore(max_frames=4, max_bytes=40, max_age_s=10.0)
        first = store.add_sample(
            uav_id="uav_1",
            frame_id="frame_1",
            sample=_sample(1.0, 1),
        )
        second = store.add_sample(
            uav_id="uav_1",
            frame_id="frame_2",
            sample=_sample(2.0, 2),
        )
        self.assertIsNone(store.get_frame(first))
        self.assertIsNotNone(store.get_depth(second))
        self.assertEqual(store.total_bytes, 28)

    def test_legacy_rgb_only_api_is_unchanged(self) -> None:
        store = FrameStore(max_frames=1, max_bytes=20, max_age_s=5.0)
        ref = store.add_frame(
            uav_id="uav_1",
            frame_id="legacy_1",
            timestamp_s=0.0,
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        )
        self.assertIsNotNone(store.get_frame(ref))
        self.assertIsNone(store.get_depth(ref))
        self.assertIsNone(store.get_camera_geometry(ref))

    def test_depth_cannot_be_detached_from_geometry(self) -> None:
        store = FrameStore(max_frames=1, max_bytes=100, max_age_s=5.0)
        with self.assertRaisesRegex(ValueError, "requires camera geometry"):
            store.add_frame(
                uav_id="uav_1",
                frame_id="bad_1",
                timestamp_s=0.0,
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                depth_to_image_plane_m=np.ones((2, 2), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
