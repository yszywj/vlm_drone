from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from configs.loader import load_config
from env.camera_types import CameraIntrinsics, CameraSample
from perception.target_debug_images import (
    BoundedTargetDebugImageWriter,
    TARGET_DEBUG_EVENTS,
    TargetDebugAnnotation,
    TargetDebugImageStats,
)
from perception.target_perception_coordinator import TargetPerceptionCoordinator
from perception.types import IdentityConsistencyEvidence, SemanticVerification
from runtime.frame_store import FrameStore
from scripts.run_dynamic_visual_mission import _VisualRuntime
from target import TargetLifecycle, TargetManager, TargetSpec
from yolo_service.protocol import (
    TargetQuery,
    TimingMs,
    TrackDetection,
    TrackResponse,
)


ROOT = Path(__file__).resolve().parents[1]


def _annotation(*, source: str = "yolo26_botsort") -> TargetDebugAnnotation:
    return TargetDebugAnnotation(
        bbox_xyxy_normalized=(0.2, 0.15, 0.8, 0.9),
        class_id=0,
        class_name="person",
        confidence=0.875,
        track_id="track_7",
        candidate_id="candidate_7",
        confirmed=True,
        position_world_m=(1.0, 2.0, 3.0),
        measurement_age_s=0.125,
        source=source,
    )


def _frame_store() -> tuple[FrameStore, object]:
    store = FrameStore(max_frames=8, max_bytes=4_000_000, max_age_s=10.0)
    rgb = np.full((180, 320, 3), 96, dtype=np.uint8)
    ref = store.add_frame(
        uav_id="uav_1",
        frame_id="frame_1",
        timestamp_s=0.0,
        rgb=rgb,
    )
    return store, ref


def test_writer_is_default_off_and_does_not_create_directory(tmp_path: Path) -> None:
    store, ref = _frame_store()
    output = tmp_path / "debug_images"
    writer = BoundedTargetDebugImageWriter(output)

    assert not writer.capture(
        event="first_detection",
        frame_store=store,
        frame_ref=ref,
        annotation=_annotation(),
    )
    assert writer.stats == TargetDebugImageStats(0, 0, ())
    assert not output.exists()


def test_writer_enforces_event_dedup_global_cap_and_oracle_boundary(
    tmp_path: Path,
) -> None:
    store, ref = _frame_store()
    output = tmp_path / "debug_images"
    writer = BoundedTargetDebugImageWriter(
        output,
        enabled=True,
        max_images_per_run=2,
    )

    # Oracle cannot consume a slot or create a file.
    assert not writer.capture(
        event="target_lost",
        frame_store=store,
        frame_ref=ref,
        annotation=_annotation(source="oracle_evaluation"),
    )
    assert writer.capture(
        event="first_detection",
        frame_store=store,
        frame_ref=ref,
        annotation=_annotation(),
    )
    assert not writer.capture(
        event="first_detection",
        frame_store=store,
        frame_ref=ref,
        annotation=_annotation(),
    )
    assert writer.capture(
        event="confirmation_success",
        frame_store=store,
        frame_ref=ref,
        annotation=_annotation(),
    )
    assert not writer.capture(
        event="reacquire_success",
        frame_store=store,
        frame_ref=ref,
        annotation=_annotation(),
    )

    files = sorted(output.glob("*.jpg"))
    assert [item.name for item in files] == [
        "01_first_detection.jpg",
        "02_confirmation_success.jpg",
    ]
    assert writer.stats.count == 2
    assert writer.stats.bytes == sum(item.stat().st_size for item in files)
    assert writer.stats.events == (
        "confirmation_success",
        "first_detection",
    )


def test_annotation_contains_every_required_field() -> None:
    lines = "\n".join(_annotation().label_lines("first_candidate"))
    for field in (
        "bbox",  # bbox is visibly rendered; remaining values are text labels.
        "class",
        "confidence",
        "track_id",
        "candidate_id",
        "confirmed",
        "position_world_m",
        "measurement_age",
        "sampled_pixel_uv",
        "raw_depth",
    ):
        if field != "bbox":
            assert f"{field}=" in lines
    assert _annotation().bbox_xyxy_normalized is not None
    assert TARGET_DEBUG_EVENTS == {
        "first_detection",
        "first_candidate",
        "confirmation_success",
        "candidate_rejected",
        "target_lost",
        "reacquire_success",
    }


def test_debug_image_marks_sampled_pixel_and_foreground_depth_without_saving_depth(
    tmp_path: Path,
) -> None:
    store = FrameStore(max_frames=2, max_bytes=4_000_000, max_age_s=10.0)
    rgb = np.full((40, 60, 3), 80, dtype=np.uint8)
    depth = np.full((40, 60), 9.0, dtype=np.float32)
    depth[10:30, 15:45] = 4.0
    ref = store.add_frame(
        uav_id="uav_1",
        frame_id="frame_depth_1",
        timestamp_s=0.0,
        rgb=rgb,
        depth_to_image_plane_m=depth,
        intrinsics=CameraIntrinsics(fx=50.0, fy=50.0, cx=29.5, cy=19.5, width=60, height=40),
        camera_position_world_m=(0.0, 0.0, 0.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    writer = BoundedTargetDebugImageWriter(tmp_path, enabled=True, max_images_per_run=1)
    annotation = replace(
        _annotation(),
        bbox_xyxy_normalized=(0.25, 0.25, 0.75, 0.75),
        sampled_pixel_uv=(30.0, 20.0),
        raw_depth_m=4.0,
    )

    assert writer.capture(
        event="confirmation_success",
        frame_store=store,
        frame_ref=ref,
        annotation=annotation,
    )
    assert len(list(tmp_path.glob("*.jpg"))) == 1
    assert not list(tmp_path.glob("*.npy"))


class _InlineExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:  # pragma: no cover - surfaced through coordinator
            future.set_exception(exc)
        return future


class _FakeYoloClient:
    def __init__(self) -> None:
        self.track_id = 7

    def reset_stream(self, request) -> None:
        del request

    def health(self):
        return {"schema_version": 1, "status": "ok", "ready": True}

    def model_info(self):
        from perception.yolo_client import YoloModelInfo

        return YoloModelInfo("yolo", ((0, "person"),))

    def track(self, request, rgb) -> TrackResponse:
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
                    track_id=self.track_id,
                    class_id=0,
                    class_name="person",
                    confidence=0.9,
                    bbox_xyxy_normalized=(0.25, 0.2, 0.75, 0.8),
                ),
            ),
            timing_ms=TimingMs(0.0, 1.0, 0.0, 1.0),
        )


class _EventSink:
    def __init__(self) -> None:
        self.events: list[str] = []

    def capture(self, **kwargs) -> bool:
        self.events.append(kwargs["event"])
        assert isinstance(kwargs["annotation"], TargetDebugAnnotation)
        assert kwargs["annotation"].source == "yolo26_botsort"
        return True


def _sample(timestamp_s: float) -> CameraSample:
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=np.zeros((32, 32, 3), dtype=np.uint8),
        depth_to_image_plane_m=np.full((32, 32), 8.0, dtype=np.float32),
        camera_position_world_m=(0.0, 0.0, 5.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=CameraIntrinsics(20.0, 20.0, 15.5, 15.5, 32, 32),
    )


def _coordinator(
    sink: _EventSink,
    *,
    semantic_matches: bool = True,
    client: _FakeYoloClient | None = None,
) -> TargetPerceptionCoordinator:
    app = load_config(ROOT / "configs" / "default.yaml")
    config = replace(app.target_perception, backend="ultralytics_service")

    def semantic(candidate, target_spec, detection, timestamp_s):
        del detection
        return SemanticVerification(
            candidate_id=candidate.candidate_id,
            timestamp_s=timestamp_s,
            target_description=target_spec.description,
            matches=semantic_matches,
            confidence=0.9,
            verifier="test_semantic",
        )

    def identity(
        candidate,
        target_id,
        tracker_id,
        new_reacquire_track,
        timestamp_s,
    ):
        del tracker_id, new_reacquire_track
        return IdentityConsistencyEvidence(
            candidate_id=candidate.candidate_id,
            target_id=target_id,
            timestamp_s=timestamp_s,
            reidentified=True,
            temporally_consistent=True,
            consistent_observations=3,
            confidence=0.9,
            source="test_identity",
        )

    return TargetPerceptionCoordinator(
        config,
        client=client or _FakeYoloClient(),
        executor=_InlineExecutor(),
        model_names={0: "person"},
        query_compiler=lambda spec, names: TargetQuery((0,), ()),
        semantic_evidence_provider=semantic,
        identity_evidence_provider=identity,
        debug_image_writer=sink,
    )


def test_coordinator_emits_detection_candidate_confirmation_and_lost_events() -> None:
    sink = _EventSink()
    coordinator = _coordinator(sink)
    manager = TargetManager()
    spec = TargetSpec("person", category="person")
    manager.start_search(spec, 0.0)
    coordinator.reset(mission_id="mission_debug", uav_id="uav_1")

    for timestamp in (0.0, 0.3, 0.6):
        coordinator.submit_frame(camera_sample=_sample(timestamp), target_spec=spec)
        coordinator.poll(now_s=timestamp, target_manager=manager)
    assert manager.lifecycle is TargetLifecycle.LOCKED
    manager.start_tracking(0.7)
    coordinator.poll(now_s=0.7, target_manager=manager)
    manager.mark_lost(timestamp_s=0.8)
    coordinator.poll(now_s=0.8, target_manager=manager)

    assert "first_detection" in sink.events
    assert "first_candidate" in sink.events
    assert "confirmation_success" in sink.events
    assert "target_lost" in sink.events
    coordinator.close()


def test_coordinator_emits_candidate_rejected_event() -> None:
    sink = _EventSink()
    coordinator = _coordinator(sink, semantic_matches=False)
    manager = TargetManager()
    spec = TargetSpec("person", category="person")
    manager.start_search(spec, 0.0)
    coordinator.reset(mission_id="mission_reject", uav_id="uav_1")

    for timestamp in (0.0, 0.3, 0.6):
        coordinator.submit_frame(camera_sample=_sample(timestamp), target_spec=spec)
        coordinator.poll(now_s=timestamp, target_manager=manager)

    assert "candidate_rejected" in sink.events
    assert coordinator.metrics.candidates_rejected == 1
    coordinator.close()


def test_coordinator_emits_reacquire_success_for_confirmed_new_track() -> None:
    sink = _EventSink()
    client = _FakeYoloClient()
    coordinator = _coordinator(sink, client=client)
    manager = TargetManager()
    spec = TargetSpec("person", category="person")
    manager.start_search(spec, 0.0)
    coordinator.reset(mission_id="mission_reacquire", uav_id="uav_1")
    for timestamp in (0.0, 0.3, 0.6):
        coordinator.submit_frame(camera_sample=_sample(timestamp), target_spec=spec)
        coordinator.poll(now_s=timestamp, target_manager=manager)
    manager.start_tracking(0.7)
    coordinator.poll(now_s=0.7, target_manager=manager)
    manager.mark_lost(timestamp_s=0.8)
    coordinator.poll(now_s=0.8, target_manager=manager)
    manager.start_reacquiring(0.9)
    coordinator.poll(now_s=0.9, target_manager=manager)

    client.track_id = 8
    for timestamp in (1.0, 1.3, 1.6):
        coordinator.submit_frame(camera_sample=_sample(timestamp), target_spec=spec)
        coordinator.poll(now_s=timestamp, target_manager=manager)

    assert manager.lifecycle is TargetLifecycle.LOCKED
    assert coordinator.metrics.reacquire_successes == 1
    assert "reacquire_success" in sink.events
    coordinator.close()


@dataclass(frozen=True)
class _FakeWriter:
    stats: TargetDebugImageStats


def test_visual_runtime_manifest_uses_writer_count_and_bytes(tmp_path: Path) -> None:
    runtime = _VisualRuntime(
        coordinator=None,
        worker=None,
        event_bus=object(),
        output_parent=tmp_path,
        model_name="none",
    )
    runtime.begin_logging("mission_manifest")
    runtime.bind_target_debug_image_writer(
        _FakeWriter(TargetDebugImageStats(2, 1234, ("first_detection",)))
    )
    manifest = json.loads(
        (tmp_path / "mission_manifest" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["debug_images"] == {"count": 2, "bytes": 1234}
    runtime.close(timeout_s=1.0)
