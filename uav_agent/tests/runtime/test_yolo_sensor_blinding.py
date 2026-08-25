from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace

import numpy as np

from configs.loader import load_config
from env.camera_types import CameraIntrinsics, CameraSample
from perception.target_perception_coordinator import TargetPerceptionCoordinator
from perception.yolo_client import YoloModelInfo
from target import TargetLifecycle, TargetManager, TargetSpec
from yolo_service.protocol import TargetQuery, TimingMs, TrackDetection, TrackResponse


class _InlineExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:  # pragma: no cover - asserted through poll
            future.set_exception(exc)
        return future


class _SensorOnlyClient:
    def __init__(self, detections: tuple[TrackDetection, ...]) -> None:
        self.detections = detections

    def health(self):
        return {"schema_version": 1, "status": "ok", "ready": True}

    def model_info(self) -> YoloModelInfo:
        return YoloModelInfo("yolo", ((0, "cube"),))

    def reset_stream(self, request) -> None:
        del request

    def track(self, request, rgb) -> TrackResponse:
        # The service receives only pixels.  No environment or target truth is
        # available through this boundary.
        assert isinstance(rgb, np.ndarray)
        return TrackResponse(
            schema_version=1,
            request_id=request.request_id,
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            stream_id=request.stream_id,
            frame_id=request.frame_id,
            timestamp_s=request.timestamp_s,
            detections=self.detections,
            timing_ms=TimingMs(0.0, 0.0, 0.0, 0.0),
        )


def _sample(timestamp_s: float, *, blank_depth: bool) -> CameraSample:
    depth = np.full((20, 20), 8.0, dtype=np.float32)
    if blank_depth:
        depth.fill(np.nan)
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=np.zeros((20, 20, 3), dtype=np.uint8),
        depth_to_image_plane_m=depth,
        camera_position_world_m=(0.0, 0.0, 10.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=CameraIntrinsics(10.0, 10.0, 9.5, 9.5, 20, 20),
    )


def _coordinator(client: _SensorOnlyClient) -> TargetPerceptionCoordinator:
    config = replace(
        load_config("configs/default.yaml").target_perception,
        backend="ultralytics_service",
    )
    return TargetPerceptionCoordinator(
        config,
        client=client,
        executor=_InlineExecutor(),
        model_names={0: "cube"},
        query_compiler=lambda spec, names: TargetQuery((0,), ()),
    )


def test_blank_rgb_and_blank_depth_cannot_create_a_world_position_or_find_target() -> None:
    detection = TrackDetection(
        track_id=1,
        class_id=0,
        class_name="cube",
        confidence=0.99,
        bbox_xyxy_normalized=(0.25, 0.2, 0.75, 0.8),
    )
    coordinator = _coordinator(_SensorOnlyClient((detection,)))
    manager = TargetManager()
    spec = TargetSpec("cube", category="cube")
    manager.start_search(spec, 0.0)
    coordinator.reset(mission_id="mission_blank_sensors", uav_id="uav_1")

    estimates = []
    for timestamp_s in (0.0, 0.3, 0.6):
        coordinator.submit_frame(
            camera_sample=_sample(timestamp_s, blank_depth=True),
            target_spec=spec,
        )
        estimates.append(coordinator.poll(now_s=timestamp_s, target_manager=manager))

    assert all(
        estimate is None or estimate.position_world_m is None
        for estimate in estimates
    )
    # A 2-D detector proposal may remain an explicit candidate, but without a
    # finite 3-D measurement it must never bind/lock the Assignment target.
    assert manager.lifecycle in {
        TargetLifecycle.SEARCHING,
        TargetLifecycle.CANDIDATE,
    }
    assert manager.snapshot().target_id != "target"
    metrics = coordinator.runtime_metrics()
    assert metrics["measurement_created"] == 0
    assert metrics["position_world_outputs"] == 0
    assert metrics["search_target_found"] == 0
    coordinator.close()


def test_off_fov_empty_yolo_response_cannot_find_target_despite_valid_depth() -> None:
    coordinator = _coordinator(_SensorOnlyClient(()))
    manager = TargetManager()
    spec = TargetSpec("cube", category="cube")
    manager.start_search(spec, 0.0)
    coordinator.reset(mission_id="mission_off_fov", uav_id="uav_1")

    for timestamp_s in (0.0, 0.3, 0.6):
        coordinator.submit_frame(
            camera_sample=_sample(timestamp_s, blank_depth=False),
            target_spec=spec,
        )
        assert coordinator.poll(now_s=timestamp_s, target_manager=manager) is None

    assert manager.lifecycle is TargetLifecycle.SEARCHING
    metrics = coordinator.runtime_metrics()
    assert metrics["detections_total"] == 0
    assert metrics["candidate_created"] == 0
    assert metrics["measurement_created"] == 0
    assert metrics["search_target_found"] == 0
    coordinator.close()
