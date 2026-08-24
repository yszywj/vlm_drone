from __future__ import annotations

import math
import unittest

import numpy as np

from env.camera_types import CameraIntrinsics, CameraSample
from perception.candidate_bank import CandidateLifecycle, CandidateSnapshot
from perception.depth_geometry import (
    DepthCandidateResolver,
    backproject_pixel_to_camera_optical,
    camera_flu_to_world,
    optical_to_camera_flu,
    project_world_to_pixel,
)
from perception.grounding import CandidateResolutionUnavailable
from runtime.frame_store import FrameCameraGeometry, FrameRef, FrameStore


def _candidate(ref: FrameRef) -> CandidateSnapshot:
    return CandidateSnapshot(
        uav_id=ref.uav_id,
        candidate_id="candidate_1",
        first_seen_timestamp_s=ref.timestamp_s,
        last_seen_timestamp_s=ref.timestamp_s,
        bbox_history=((0.36, 0.36, 0.64, 0.64),),
        frame_history=(ref,),
        source="yolo26_botsort",
        lifecycle=CandidateLifecycle.PROVISIONAL,
        review_history=(),
    )


class DepthGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(
            fx=100.0,
            fy=100.0,
            cx=5.0,
            cy=5.0,
            width=11,
            height=11,
        )

    def test_known_depth_backprojects_through_flu_world_axes(self) -> None:
        optical = backproject_pixel_to_camera_optical(
            u_px=5.0,
            v_px=5.0,
            depth_m=4.0,
            intrinsics=self.intrinsics,
        )
        self.assertEqual(optical, (0.0, 0.0, 4.0))
        geometry = FrameCameraGeometry(
            timestamp_s=1.0,
            intrinsics=self.intrinsics,
            camera_position_world_m=(1.0, 2.0, 3.0),
            camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        world = camera_flu_to_world(optical_to_camera_flu(optical), geometry)
        np.testing.assert_allclose(world, (5.0, 2.0, 3.0), atol=1e-12)

    def test_world_pixel_round_trip_with_rotated_camera(self) -> None:
        half = math.sqrt(0.5)
        geometry = FrameCameraGeometry(
            timestamp_s=1.0,
            intrinsics=self.intrinsics,
            camera_position_world_m=(2.0, -1.0, 0.5),
            camera_orientation_world_wxyz=(half, 0.0, 0.0, half),
        )
        expected = camera_flu_to_world((6.0, -0.06, 0.03), geometry)
        u, v, depth = project_world_to_pixel(
            position_world_m=expected,
            geometry=geometry,
        )
        optical = backproject_pixel_to_camera_optical(
            u_px=u,
            v_px=v,
            depth_m=depth,
            intrinsics=self.intrinsics,
        )
        actual = camera_flu_to_world(optical_to_camera_flu(optical), geometry)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_resolver_uses_bbox_patch_median_and_world_pose(self) -> None:
        store = FrameStore(max_frames=2, max_bytes=10_000, max_age_s=10.0)
        depth = np.full((11, 11), 4.0, dtype=np.float32)
        depth[7, 5] = 1000.0  # invalid far outlier at the nominal anchor
        sample = CameraSample(
            timestamp_s=1.0,
            rgb=np.zeros((11, 11, 3), dtype=np.uint8),
            depth_to_image_plane_m=depth,
            camera_position_world_m=(1.0, 2.0, 3.0),
            camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            intrinsics=self.intrinsics,
        )
        ref = store.add_sample(uav_id="uav_1", frame_id="frame_1", sample=sample)
        resolved = DepthCandidateResolver(
            store,
            patch_radius_px=1,
            min_depth_m=0.2,
            max_depth_m=20.0,
        ).resolve(_candidate(ref), timestamp_s=1.0)

        # Bottom-center pixel is (5, 7): optical (0, .08, 4), then FLU.
        np.testing.assert_allclose(resolved.position_xyz_m, (5.0, 2.0, 2.92), atol=1e-6)
        self.assertEqual(resolved.source, "isaac_depth_bbox_bottom_center")

    def test_missing_or_invalid_depth_never_fabricates_zero_position(self) -> None:
        store = FrameStore(max_frames=2, max_bytes=10_000, max_age_s=10.0)
        ref = store.add_frame(
            uav_id="uav_1",
            frame_id="frame_1",
            timestamp_s=1.0,
            rgb=np.zeros((11, 11, 3), dtype=np.uint8),
        )
        with self.assertRaises(CandidateResolutionUnavailable):
            DepthCandidateResolver(store).resolve(_candidate(ref), timestamp_s=1.0)


if __name__ == "__main__":
    unittest.main()
