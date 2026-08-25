from __future__ import annotations

import numpy as np
import pytest

from env.camera_types import CameraIntrinsics, CameraSample
from perception.candidate_bank import CandidateLifecycle, CandidateSnapshot
from perception.depth_geometry import DepthCandidateResolver, DepthSamplingStrategy
from perception.grounding import CandidateResolutionUnavailable
from perception.measurement import TargetMeasurement
from runtime.frame_store import FrameRef, FrameStore


def _candidate(ref: FrameRef) -> CandidateSnapshot:
    return CandidateSnapshot(
        uav_id=ref.uav_id,
        candidate_id="candidate_1",
        first_seen_timestamp_s=ref.timestamp_s,
        last_seen_timestamp_s=ref.timestamp_s,
        bbox_history=((0.25, 0.25, 0.75, 0.75),),
        frame_history=(ref,),
        source="ultralytics_service",
        lifecycle=CandidateLifecycle.PROVISIONAL,
        review_history=(),
    )


def _store(depth: np.ndarray) -> tuple[FrameStore, FrameRef]:
    intrinsics = CameraIntrinsics(
        fx=120.0,
        fy=120.0,
        cx=20.0,
        cy=20.0,
        width=40,
        height=40,
    )
    sample = CameraSample(
        timestamp_s=1.0,
        rgb=np.zeros((40, 40, 3), dtype=np.uint8),
        depth_to_image_plane_m=depth,
        camera_position_world_m=(1.0, 2.0, 3.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=intrinsics,
    )
    store = FrameStore(max_frames=2, max_bytes=1_000_000, max_age_s=10.0)
    ref = store.add_sample(uav_id="uav_1", frame_id="frame_1", sample=sample)
    return store, ref


def test_foreground_cluster_uses_center_target_not_bottom_ground() -> None:
    depth = np.full((40, 40), 20.0, dtype=np.float32)
    depth[13:26, 13:27] = 6.0  # target in the center of the bbox
    depth[26:30, 10:30] = 2.0  # closer ground along bbox bottom
    depth[18, 20] = np.nan
    store, ref = _store(depth)

    measurement = DepthCandidateResolver(store).resolve(
        _candidate(ref),
        timestamp_s=1.0,
    )

    assert isinstance(measurement, TargetMeasurement)
    assert measurement.corrected_depth_m == pytest.approx(6.0)
    assert measurement.source.endswith("foreground_cluster_median")
    assert 0.0 < measurement.measurement_quality <= 1.0
    covariance = np.asarray(measurement.covariance_world_m2)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-10


@pytest.mark.parametrize(
    "strategy",
    (
        DepthSamplingStrategy.BBOX_CENTER,
        DepthSamplingStrategy.BBOX_BOTTOM_CENTER,
        DepthSamplingStrategy.BBOX_PATCH_MEDIAN,
    ),
)
def test_legacy_bbox_anchors_remain_selectable(
    strategy: DepthSamplingStrategy,
) -> None:
    store, ref = _store(np.full((40, 40), 7.0, dtype=np.float32))
    measurement = DepthCandidateResolver(
        store,
        sampling_strategy=strategy,
    ).resolve(_candidate(ref), timestamp_s=1.0)
    assert measurement.corrected_depth_m == pytest.approx(7.0)
    assert measurement.source.endswith(strategy.value)


def test_invalid_depth_and_out_of_bounds_bbox_fail_with_reasons() -> None:
    store, ref = _store(np.full((40, 40), np.nan, dtype=np.float32))
    resolver = DepthCandidateResolver(store)
    with pytest.raises(
        CandidateResolutionUnavailable,
        match="foreground_cluster_insufficient_valid_depth_samples",
    ):
        resolver.resolve(_candidate(ref), timestamp_s=1.0)

    with pytest.raises(
        CandidateResolutionUnavailable,
        match="candidate_bbox_out_of_bounds",
    ):
        resolver._sample_depth(  # noqa: SLF001 - explicit geometry boundary test
            np.ones((40, 40), dtype=np.float32),
            (-0.1, 0.1, 0.5, 0.5),
        )

