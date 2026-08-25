from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from configs.loader import load_config
from env.camera_types import CameraIntrinsics, CameraSample
from perception.candidate_bank import CandidateBank
from perception.runtime_bridge import CoordinatedVisionPerceptionBackend
from perception.runtime_provider import YoloTargetPerceptionRuntime
from perception.semantic_fusion import TemporalRgbdAttributeSemanticProvider
from perception.target_perception_coordinator import TargetPerceptionCoordinator
from perception.vision_backend import VisionPerceptionBackend
from perception.visual_review import (
    QwenVisualReview,
    VisualReviewAction,
    VisualReviewCandidate,
    VisualReviewDecision,
)
from perception.yolo_client import YoloModelInfo
from runtime.frame_store import FrameStore
from target import TargetLifecycle, TargetManager, TargetSpec
from tests.perception.test_yolo_runtime_provider import (
    _Bridge,
    _base,
    _query,
    _sample,
)
from yolo_service.protocol import (
    TargetQuery,
    TimingMs,
    TrackDetection,
    TrackResponse,
)


ROOT = Path(__file__).resolve().parents[2]


def test_two_yolo_runtimes_keep_identical_tracker_ids_isolated() -> None:
    bridge_a = _Bridge("uav_a")
    bridge_b = _Bridge("uav_b")
    runtime_a = YoloTargetPerceptionRuntime(uav_id="uav_a", bridge=bridge_a)
    runtime_b = YoloTargetPerceptionRuntime(uav_id="uav_b", bridge=bridge_b)
    spec_a = TargetSpec(
        "red cube",
        category="cube",
        hard_attributes=("color=red",),
    )
    spec_b = TargetSpec(
        "blue cube",
        category="cube",
        hard_attributes=("color=blue",),
    )
    runtime_a.reset(
        mission_id="fleet_mission_1",
        assignment_id="assignment_a",
        uav_id="uav_a",
        target_query=_query(spec_a, "target_i"),
    )
    runtime_b.reset(
        mission_id="fleet_mission_1",
        assignment_id="assignment_b",
        uav_id="uav_b",
        target_query=_query(spec_b, "target_j"),
    )
    sample = _sample(1.0)

    observed_a = runtime_a.observe(
        base_observation=_base(sample, uav_id="uav_a"),
        camera_sample=sample,
        target_manager=TargetManager(),
    )
    observed_b = runtime_b.observe(
        base_observation=_base(sample, uav_id="uav_b"),
        camera_sample=sample,
        target_manager=TargetManager(),
    )

    estimate_a = observed_a.target_estimate
    estimate_b = observed_b.target_estimate
    assert estimate_a is not None and estimate_b is not None
    assert estimate_a.tracker_id == estimate_b.tracker_id == "tracker_7"
    assert estimate_a.target_id == "target_i"
    assert estimate_b.target_id == "target_j"
    assert runtime_a._bridge is bridge_a  # noqa: SLF001
    assert runtime_b._bridge is bridge_b  # noqa: SLF001
    assert bridge_a.inputs is not bridge_b.inputs
    runtime_a.close()
    assert bridge_a.closed == 1
    assert bridge_b.closed == 0
    runtime_b.close()


class _InlineExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:  # pragma: no cover - surfaced by coordinator
            future.set_exception(exc)
        return future


class _CubeClient:
    def __init__(self) -> None:
        self.reset_calls: list[str] = []

    def health(self) -> dict[str, object]:
        return {"schema_version": 1, "status": "ok", "ready": True}

    def model_info(self) -> YoloModelInfo:
        return YoloModelInfo(
            "yolo",
            ((0, "cube"),),
            "895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07",
        )

    def reset_stream(self, request) -> None:
        self.reset_calls.append(request.stream_id)

    def track(self, request, rgb: np.ndarray) -> TrackResponse:
        del rgb
        return TrackResponse(
            schema_version=1,
            request_id=request.request_id,
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            stream_id=request.stream_id,
            frame_id=request.frame_id,
            timestamp_s=request.timestamp_s,
            detections=(
                TrackDetection(
                    track_id=7,
                    class_id=0,
                    class_name="cube",
                    confidence=0.9,
                    bbox_xyxy_normalized=(0.1, 0.1, 0.9, 0.9),
                ),
            ),
            timing_ms=TimingMs(0.0, 1.0, 1.0, 2.0),
        )


def _color_sample(timestamp_s: float, rgb: tuple[int, int, int]) -> CameraSample:
    pixels = np.empty((20, 20, 3), dtype=np.uint8)
    pixels[:] = rgb
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=pixels,
        depth_to_image_plane_m=np.full((20, 20), 10.0, dtype=np.float32),
        camera_position_world_m=(0.0, 0.0, 0.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=CameraIntrinsics(10.0, 10.0, 10.0, 10.0, 20, 20),
    )


def _color_coordinator(
    *,
    review_timestamp: list[float],
    review_calls: list[str],
) -> tuple[TargetPerceptionCoordinator, _CubeClient]:
    app = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")
    color = replace(
        app.target_perception.attributes.color,
        min_observations=3,
        min_duration_s=0.4,
    )
    config = replace(
        app.target_perception,
        tracker=replace(
            app.target_perception.tracker,
            min_track_observations=1,
            min_track_duration_s=0.0,
        ),
        attributes=replace(
            app.target_perception.attributes,
            color=color,
        ),
    )
    store = FrameStore()
    semantic = TemporalRgbdAttributeSemanticProvider.from_target_perception_config(
        config,
        mission_id="mission_color",
        uav_id="uav_a",
        assignment_id="assignment_color",
        frame_store=store,
        expected_class_id=0,
    )
    client = _CubeClient()
    coordinator = TargetPerceptionCoordinator(
        config,
        client=client,
        executor=_InlineExecutor(),
        frame_store=store,
        candidate_bank=CandidateBank(uav_id="uav_a"),
        model_names={0: "cube"},
        query_compiler=lambda spec, names: TargetQuery((0,), ()),
        semantic_evidence_provider=semantic,
    )

    def already_available_positive_review(candidate_id: str) -> QwenVisualReview:
        review_calls.append(candidate_id)
        timestamp_s = review_timestamp[0]
        return QwenVisualReview(
            schema_version=1,
            review_id=f"review_{len(review_calls)}",
            mission_id="mission_color",
            uav_id="uav_a",
            plan_version=1,
            observation_timestamp_s=timestamp_s,
            frame_id=f"review_frame_{len(review_calls)}",
            decision=VisualReviewDecision.TARGET_MATCH,
            candidate=VisualReviewCandidate(
                present=True,
                bbox_xyxy_normalized=(0.1, 0.1, 0.9, 0.9),
                description="red cube",
                self_reported_confidence=0.99,
            ),
            scene_observations=("Qwen says target match",),
            reason_codes=("early_positive_review",),
            recommended_action=VisualReviewAction.CONTINUE,
        )

    coordinator.bind_visual_review_provider(already_available_positive_review)
    coordinator.reset(
        mission_id="mission_color",
        uav_id="uav_a",
        assignment_id="assignment_color",
    )
    return coordinator, client


def _run_color_sequence(
    rgb: tuple[int, int, int],
) -> tuple[
    TargetPerceptionCoordinator,
    TargetManager,
    list[object],
    list[str],
]:
    review_timestamp = [0.0]
    review_calls: list[str] = []
    coordinator, _ = _color_coordinator(
        review_timestamp=review_timestamp,
        review_calls=review_calls,
    )
    manager = TargetManager()
    spec = TargetSpec(
        "red cube",
        category="cube",
        hard_attributes=("color=red",),
    )
    manager.start_search(spec, 0.0)
    estimates: list[object] = []
    for timestamp_s in (0.0, 0.1, 0.2, 0.5):
        review_timestamp[0] = timestamp_s
        coordinator.submit_frame(
            camera_sample=_color_sample(timestamp_s, rgb),
            target_spec=spec,
        )
        estimates.append(
            coordinator.poll(now_s=timestamp_s, target_manager=manager)
        )
    return coordinator, manager, estimates, review_calls


def test_color_temporal_threshold_blocks_class_only_and_early_qwen_lock(
    capsys,
) -> None:
    coordinator, manager, estimates, review_calls = _run_color_sequence(
        (255, 0, 0)
    )

    events = manager.events()
    assert events[0].new_state is TargetLifecycle.SEARCHING
    # The first three class=cube detections include two semantic evaluations,
    # but deterministic RGB-D evidence is still below 3 frames / 0.4 s.
    assert all(not estimate.confirmed for estimate in estimates[:3])
    assert events[-2].new_state is TargetLifecycle.CANDIDATE
    assert review_calls == []
    assert estimates[-1].confirmed is True
    assert manager.lifecycle is TargetLifecycle.LOCKED
    assert coordinator.metrics.candidates_confirmed == 1
    metrics = coordinator.runtime_metrics()
    assert metrics["attribute_confirmed"] == 1
    assert metrics["attribute_ambiguous"] == 2
    assert metrics["measurement_created"] == 4
    assert metrics["kalman_updates_accepted"] == 1
    assert metrics["search_target_found"] == 1

    prefix = "[PerceptionCandidate] "
    candidate_events = [
        json.loads(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    confirmed = next(
        event
        for event in candidate_events
        if event["transition"] == "candidate_confirmed"
    )
    assert confirmed["tracker_id"] == "track_7"
    assert confirmed["attribute_state"] == "match"
    assert confirmed["color_result"] == "red"
    assert confirmed["geometry_state"] == "measurement_created"
    assert confirmed["measurement_source"] == "isaac_depth_foreground_cluster_median"
    assert confirmed["confirmed"] is True
    assert confirmed["position_world_m"] is not None
    assert confirmed["estimate_source"] == "yolo26_botsort"
    coordinator.close()


def test_stable_wrong_color_rejects_candidate_without_qwen_override() -> None:
    coordinator, manager, estimates, review_calls = _run_color_sequence(
        (0, 0, 255)
    )

    assert all(not estimate.confirmed for estimate in estimates)
    assert manager.lifecycle is TargetLifecycle.SEARCHING
    assert coordinator.metrics.candidates_confirmed == 0
    assert coordinator.metrics.candidates_rejected == 1
    metrics = coordinator.runtime_metrics()
    assert metrics["attribute_confirmed"] == 0
    assert metrics["attribute_ambiguous"] == 2
    assert review_calls == []
    coordinator.close()


def test_assignment_alias_is_authoritative_for_yolo_lock_and_estimate() -> None:
    review_timestamp = [0.0]
    review_calls: list[str] = []
    coordinator, _ = _color_coordinator(
        review_timestamp=review_timestamp,
        review_calls=review_calls,
    )
    app = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")
    runtime = YoloTargetPerceptionRuntime(
        uav_id="uav_a",
        bridge=CoordinatedVisionPerceptionBackend(
            uav_id="uav_a",
            coordinator=coordinator,
            vision_backend=VisionPerceptionBackend(
                app.target_perception,
                uav_id="uav_a",
            ),
        ),
    )
    spec = TargetSpec(
        "red cube",
        category="cube",
        hard_attributes=("color=red",),
    )
    manager = TargetManager()
    manager.start_search(spec, 0.0)
    runtime.reset(
        mission_id="mission_color",
        assignment_id="assignment_color",
        uav_id="uav_a",
        target_query=_query(spec, "target_i"),
    )

    observed = None
    for timestamp_s in (0.0, 0.1, 0.2, 0.5, 0.6):
        review_timestamp[0] = timestamp_s
        sample = _color_sample(timestamp_s, (255, 0, 0))
        observed = runtime.observe(
            base_observation=_base(sample, uav_id="uav_a"),
            camera_sample=sample,
            target_manager=manager,
        )

    assert observed is not None
    assert observed.target_estimate is not None
    assert observed.target_estimate.target_id == "target_i"
    assert manager.lifecycle is TargetLifecycle.LOCKED
    assert manager.snapshot().target_id == "target_i"
    assert runtime.target_id == "target_i"
    assert review_calls == []
    runtime.close()
