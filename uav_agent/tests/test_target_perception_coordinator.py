from __future__ import annotations

from concurrent.futures import Future
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from time import monotonic
import unittest

import numpy as np

from configs.loader import load_config
from env.camera_types import CameraIntrinsics, CameraSample
from perception.candidate_bank import CandidateBank
from perception.grounding import CandidateResolutionUnavailable
from perception.semantic_fusion import DETERMINISTIC_ATTRIBUTE_VERIFIER
from perception.target_perception_coordinator import (
    TargetPerceptionCoordinator,
    TargetPerceptionError,
    TargetPerceptionNotReady,
)
from perception.target_state_estimator import TargetStateEstimator
from perception.types import SemanticVerification
from perception.yolo_client import (
    YoloClientRequestTimeout,
    YoloClientResponseError,
    YoloClientStreamBusy,
    YoloClientUnavailable,
    YoloModelInfo,
)
from perception.visual_review import (
    QwenVisualReview,
    VisualReviewAction,
    VisualReviewCandidate,
    VisualReviewDecision,
)
from target import TargetLifecycle, TargetManager, TargetSpec
from yolo_service.protocol import (
    TargetQuery,
    TimingMs,
    TrackDetection,
    TrackResponse,
)


ROOT = Path(__file__).resolve().parents[1]


class InlineExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


class ManualExecutor:
    """Deterministic single-worker executor for lifecycle race tests."""

    def __init__(self) -> None:
        self.tasks: list[tuple[Future, object, tuple[object, ...]]] = []

    def submit(self, function, *args):
        future = Future()
        self.tasks.append((future, function, args))
        return future

    def run_next(self) -> None:
        future, function, args = self.tasks.pop(0)
        if future.cancelled():
            return
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)


class FakeClient:
    def __init__(self) -> None:
        self.reset_calls: list[str] = []
        self.health_calls = 0
        self.model_info_calls = 0
        self.track_calls = 0
        self.track_id = 7
        self.detections_override: tuple[TrackDetection, ...] | None = None

    def health(self) -> dict[str, object]:
        return {"schema_version": 1, "status": "ok", "ready": True}

    def model_info(self) -> YoloModelInfo:
        return YoloModelInfo("yolo", ((0, "person"),))

    def health(self):
        self.health_calls += 1
        return {"schema_version": 1, "status": "ok", "ready": True}

    def model_info(self) -> YoloModelInfo:
        self.model_info_calls += 1
        return YoloModelInfo("yolo", ((0, "person"),))

    def reset_stream(self, request) -> None:
        self.reset_calls.append(request.stream_id)

    def track(self, request, rgb) -> TrackResponse:
        self.track_calls += 1
        return TrackResponse(
            schema_version=1,
            request_id=request.request_id,
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            stream_id=request.stream_id,
            frame_id=request.frame_id,
            timestamp_s=request.timestamp_s,
            detections=(
                self.detections_override
                if self.detections_override is not None
                else (
                    TrackDetection(
                        track_id=self.track_id,
                        class_id=0,
                        class_name="person",
                        confidence=0.9,
                        bbox_xyxy_normalized=(0.35, 0.2, 0.65, 0.8),
                    ),
                )
            ),
            timing_ms=TimingMs(0.0, 2.0, 1.0, 3.0),
        )


class UnavailableResolver:
    def __init__(self, reason: str = "foreground_depth_cluster_empty") -> None:
        self.reason = reason

    def resolve(self, candidate, *, timestamp_s: float):
        raise CandidateResolutionUnavailable(self.reason)


def sample(timestamp_s: float, *, depth_m: float = 10.0) -> CameraSample:
    intrinsics = CameraIntrinsics(10.0, 10.0, 9.5, 9.5, 20, 20)
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=np.zeros((20, 20, 3), dtype=np.uint8),
        depth_to_image_plane_m=np.full((20, 20), depth_m, dtype=np.float32),
        camera_position_world_m=(0.0, 0.0, 10.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=intrinsics,
    )


class TargetPerceptionCoordinatorTest(unittest.TestCase):
    def make_coordinator(self, client: FakeClient) -> TargetPerceptionCoordinator:
        app = load_config(ROOT / "configs/default.yaml")
        config = replace(app.target_perception, backend="ultralytics_service")
        return TargetPerceptionCoordinator(
            config,
            client=client,
            executor=InlineExecutor(),
            model_names={0: "person"},
            query_compiler=lambda spec, names: TargetQuery((0,), ()),
        )

    def test_candidate_requires_short_track_before_lock_and_has_3d_state(self) -> None:
        client = FakeClient()
        coordinator = self.make_coordinator(client)
        manager = TargetManager()
        spec = TargetSpec("a person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_1", uav_id="uav_1")

        estimates = []
        for timestamp in (0.0, 0.3, 0.6):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            estimates.append(coordinator.poll(now_s=timestamp, target_manager=manager))

        self.assertFalse(estimates[0].confirmed)
        self.assertFalse(estimates[1].confirmed)
        self.assertTrue(estimates[2].confirmed)
        self.assertIsNotNone(estimates[2].position_world_m)
        self.assertIsNotNone(estimates[2].velocity_world_mps)
        self.assertEqual(manager.lifecycle, TargetLifecycle.LOCKED)
        self.assertEqual(client.track_calls, 3)
        self.assertEqual(coordinator.metrics.candidates_confirmed, 1)
        coordinator.close()
        self.assertEqual(client.reset_calls, ["mission_1:uav_1", "mission_1:uav_1"])

    def test_runtime_evidence_metrics_and_candidate_logs_are_bounded(self) -> None:
        client = FakeClient()
        coordinator = self.make_coordinator(client)
        manager = TargetManager()
        spec = TargetSpec("a person", category="person")
        manager.start_search(spec, 0.0)

        output = StringIO()
        with redirect_stdout(output):
            coordinator.reset(
                mission_id="mission_metrics",
                uav_id="uav_metrics",
                assignment_id="assignment_metrics",
            )
            for timestamp in (0.0, 0.3, 0.6):
                coordinator.submit_frame(
                    camera_sample=sample(timestamp),
                    target_spec=spec,
                )
                coordinator.poll(now_s=timestamp, target_manager=manager)

            manager.start_tracking(0.7)
            coordinator.submit_frame(camera_sample=sample(0.9), target_spec=spec)
            visible = coordinator.poll(now_s=0.9, target_manager=manager)
            predicted = coordinator.poll(now_s=1.5, target_manager=manager)

        self.assertIsNotNone(visible)
        self.assertTrue(visible.visible)
        self.assertIsNotNone(predicted)
        self.assertTrue(predicted.predicted_only)

        metrics = coordinator.runtime_metrics()
        expected_counts = {
            "camera_frames_received": 4,
            "yolo_requests_submitted": 4,
            "yolo_results_received": 4,
            "detections_total": 4,
            "tracked_detections_total": 4,
            "candidate_created": 1,
            "candidate_confirmed": 1,
            "candidate_rejected": 0,
            "depth_resolution_attempts": 4,
            "depth_resolution_successes": 4,
            "depth_resolution_failures": 0,
            "measurement_created": 4,
            "measurement_rejected": 0,
            "kalman_updates_accepted": 2,
            "kalman_updates_rejected": 0,
            "position_world_outputs": 5,
            "predicted_only_outputs": 1,
            "search_target_found": 1,
            "track_visible_updates": 1,
            "track_predicted_updates": 1,
            "attribute_confirmed": 0,
            "attribute_ambiguous": 0,
        }
        for name, expected in expected_counts.items():
            self.assertEqual(metrics[name], expected, name)

        prefix = "[PerceptionCandidate] "
        records = [
            json.loads(line.removeprefix(prefix))
            for line in output.getvalue().splitlines()
            if line.startswith(prefix)
        ]
        self.assertEqual(
            [record["transition"] for record in records],
            ["candidate_created", "candidate_confirmed"],
        )
        expected_fields = {
            "timestamp",
            "uav_id",
            "assignment_id",
            "transition",
            "tracker_id",
            "candidate_id",
            "bbox",
            "detector_confidence",
            "attribute_state",
            "color_result",
            "geometry_state",
            "measurement_source",
            "position_world_m",
            "confirmed",
            "target_id",
            "estimate_source",
        }
        for record in records:
            self.assertEqual(set(record), expected_fields)
            self.assertEqual(record["uav_id"], "uav_metrics")
            self.assertEqual(record["assignment_id"], "assignment_metrics")
            self.assertEqual(
                record["measurement_source"],
                "isaac_depth_foreground_cluster_median",
            )
        self.assertEqual(
            list(coordinator.candidate_transition_records()),
            records,
        )
        self.assertEqual(
            list(coordinator.drain_candidate_transition_records()),
            records,
        )
        self.assertEqual(coordinator.drain_candidate_transition_records(), ())
        scalar_log = output.getvalue().casefold()
        for forbidden in (
            "ground_truth",
            "oracle_target",
            "prim_path",
            "motion_seed",
        ):
            self.assertNotIn(forbidden, scalar_log)
        coordinator.close()

    def test_reset_is_atomic_fail_fast_handshake_and_checks_model_family(self) -> None:
        class MismatchedClient(FakeClient):
            def model_info(self) -> YoloModelInfo:
                return YoloModelInfo("yoloe", ((0, "person"),))

        client = MismatchedClient()
        coordinator = self.make_coordinator(client)
        with self.assertRaisesRegex(TargetPerceptionError, "model family mismatch"):
            coordinator.reset(mission_id="mission_bad_model", uav_id="uav_1")
        self.assertIn("yolo_startup_handshake_failed", coordinator.last_error or "")
        with self.assertRaises(TargetPerceptionNotReady):
            coordinator.submit_frame(
                camera_sample=sample(0.0),
                target_spec=TargetSpec("person", category="person"),
            )
        coordinator.close()

    def test_reset_retries_busy_stream_within_bound(self) -> None:
        class BusyOnceClient(FakeClient):
            def __init__(self) -> None:
                super().__init__()
                self.busy = True

            def reset_stream(self, request) -> None:
                if self.busy:
                    self.busy = False
                    raise YoloClientResponseError("HTTP 409 stream_busy")
                super().reset_stream(request)

        client = BusyOnceClient()
        coordinator = self.make_coordinator(client)
        coordinator.reset(mission_id="mission_busy", uav_id="uav_1")
        self.assertEqual(client.reset_calls, ["mission_busy:uav_1"])
        self.assertEqual(client.health_calls, 1)
        self.assertEqual(client.model_info_calls, 1)
        coordinator.close()

    def test_second_reset_retires_completed_inflight_stream_before_new_task(self) -> None:
        client = FakeClient()
        coordinator = self.make_coordinator(client)
        spec = TargetSpec("person", category="person")
        coordinator.reset(mission_id="mission_first", uav_id="uav_1")
        coordinator.submit_frame(camera_sample=sample(0.0), target_spec=spec)

        coordinator.reset(mission_id="mission_second", uav_id="uav_1")

        self.assertEqual(
            client.reset_calls,
            [
                "mission_first:uav_1",
                "mission_first:uav_1",
                "mission_second:uav_1",
            ],
        )
        self.assertEqual(client.health_calls, 2)
        coordinator.close()

    def test_close_busy_stream_cleanup_is_bounded_and_records_failure(self) -> None:
        class CleanupBusyClient(FakeClient):
            def reset_stream(self, request) -> None:
                if self.reset_calls:
                    raise YoloClientResponseError("HTTP 409 stream_busy")
                super().reset_stream(request)

        client = CleanupBusyClient()
        coordinator = self.make_coordinator(client)
        coordinator.reset(mission_id="mission_close_busy", uav_id="uav_1")
        started = monotonic()
        coordinator.close()
        elapsed = monotonic() - started

        self.assertLess(elapsed, 1.5)
        self.assertIn("yolo_cleanup_reset_failed", coordinator.last_error or "")

    def test_one_inflight_one_pending_drops_only_intermediate_frame(self) -> None:
        # The public bound is exercised separately from HTTP: an unresolved
        # future leaves one newest pending slot and replaces its predecessor.
        class HoldingExecutor:
            def __init__(self):
                self.futures = []

            def submit(self, function, *args):
                future = Future()
                self.futures.append((future, function, args))
                return future

        app = load_config(ROOT / "configs/default.yaml")
        config = replace(app.target_perception, backend="ultralytics_service")
        holding = HoldingExecutor()
        coordinator = TargetPerceptionCoordinator(
            config,
            client=FakeClient(),
            executor=holding,
            model_names={0: "person"},
            query_compiler=lambda spec, names: TargetQuery((0,), ()),
        )
        coordinator.reset(mission_id="mission_2", uav_id="uav_1")
        spec = TargetSpec("person", category="person")
        for timestamp in (0.0, 0.1, 0.2):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
        self.assertEqual(len(holding.futures), 1)
        self.assertEqual(coordinator.metrics.yolo_dropped_frames, 1)
        coordinator.close()

    def test_timeout_serializes_reset_barrier_before_newest_pending_track(self) -> None:
        """A local timeout must not be mistaken for remote inference completion."""

        class OrphanedInferenceClient(FakeClient):
            def __init__(self) -> None:
                super().__init__()
                self.first_track = True
                self.remote_inference_active = False
                self.concurrent_track_attempts = 0
                self.busy_reset_attempts = 0

            def track(self, request, rgb) -> TrackResponse:
                if self.first_track:
                    self.first_track = False
                    self.track_calls += 1
                    self.remote_inference_active = True
                    raise YoloClientRequestTimeout(
                        "deadline elapsed while model.track still runs"
                    )
                if self.remote_inference_active:
                    self.concurrent_track_attempts += 1
                    raise YoloClientStreamBusy("HTTP 409 STREAM_BUSY")
                return super().track(request, rgb)

            def reset_stream(self, request) -> None:
                if self.remote_inference_active:
                    self.busy_reset_attempts += 1
                    if self.busy_reset_attempts == 1:
                        raise YoloClientStreamBusy("HTTP 409 STREAM_BUSY")
                    self.remote_inference_active = False
                super().reset_stream(request)

        app = load_config(ROOT / "configs/default.yaml")
        config = replace(app.target_perception, backend="ultralytics_service")
        executor = ManualExecutor()
        client = OrphanedInferenceClient()
        coordinator = TargetPerceptionCoordinator(
            config,
            client=client,
            executor=executor,
            model_names={0: "person"},
            query_compiler=lambda spec, names: TargetQuery((0,), ()),
        )
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_timeout_race", uav_id="uav_1")

        coordinator.submit_frame(camera_sample=sample(0.0), target_spec=spec)
        coordinator.submit_frame(camera_sample=sample(0.1), target_spec=spec)
        executor.run_next()  # local timeout; remote inference remains active
        coordinator.poll(now_s=0.1, target_manager=manager)

        # While the asynchronous reset barrier is queued, only the newest
        # camera frame may replace the pending slot.  No Track call is queued.
        coordinator.submit_frame(camera_sample=sample(0.2), target_spec=spec)
        self.assertEqual(len(executor.tasks), 1)
        self.assertEqual(client.track_calls, 1)
        self.assertEqual(client.concurrent_track_attempts, 0)

        executor.run_next()  # reset observes one 409, then retires the stream
        coordinator.poll(now_s=0.2, target_manager=manager)
        self.assertEqual(len(executor.tasks), 1)  # newest Track launches now
        executor.run_next()
        coordinator.poll(now_s=0.2, target_manager=manager)

        self.assertEqual(client.track_calls, 2)
        self.assertEqual(client.concurrent_track_attempts, 0)
        self.assertEqual(client.busy_reset_attempts, 2)
        self.assertEqual(coordinator.metrics.yolo_timeouts, 1)
        self.assertEqual(coordinator.metrics.yolo_stream_recoveries, 1)
        self.assertEqual(coordinator.metrics.yolo_stream_recovery_failures, 0)
        coordinator.close()

    def test_track_stream_busy_is_recovered_instead_of_failed_closed(self) -> None:
        class BusyOnceTrackClient(FakeClient):
            def __init__(self) -> None:
                super().__init__()
                self.busy_once = True

            def track(self, request, rgb) -> TrackResponse:
                if self.busy_once:
                    self.busy_once = False
                    raise YoloClientStreamBusy(
                        "YOLO service rejected request (409, STREAM_BUSY)"
                    )
                return super().track(request, rgb)

        client = BusyOnceTrackClient()
        coordinator = self.make_coordinator(client)
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_track_busy", uav_id="uav_1")

        coordinator.submit_frame(camera_sample=sample(0.0), target_spec=spec)
        self.assertIsNone(
            coordinator.poll(now_s=0.0, target_manager=manager)
        )
        coordinator.submit_frame(camera_sample=sample(0.1), target_spec=spec)
        estimate = coordinator.poll(now_s=0.1, target_manager=manager)

        self.assertIsNotNone(estimate)
        self.assertEqual(coordinator.metrics.yolo_stream_busy_responses, 1)
        self.assertEqual(coordinator.metrics.yolo_response_errors, 0)
        self.assertEqual(coordinator.metrics.yolo_stream_recoveries, 1)
        self.assertIsNone(coordinator._fatal_error)
        coordinator.close()

    def test_stable_attribute_candidate_uses_typed_qwen_semantic_review(self) -> None:
        client = FakeClient()
        coordinator = self.make_coordinator(client)
        manager = TargetManager()
        spec = TargetSpec(
            "person wearing red",
            category="person",
            hard_attributes=("wearing red",),
        )
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_qwen", uav_id="uav_1")
        accepted_review = QwenVisualReview(
            schema_version=1,
            review_id="review_attribute",
            mission_id="mission_qwen",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=0.6,
            frame_id="frame_review",
            decision=VisualReviewDecision.TARGET_MATCH,
            candidate=VisualReviewCandidate(
                True,
                (0.35, 0.2, 0.65, 0.8),
                "person wearing red",
                0.9,
            ),
            scene_observations=("red clothing visible",),
            reason_codes=("attribute_match",),
            recommended_action=VisualReviewAction.CONTINUE,
        )
        coordinator.bind_visual_review_provider(
            lambda candidate_id: (
                accepted_review
                if candidate_id == "mission_qwen_uav_1_track_7"
                else None
            )
        )

        estimates = []
        for timestamp in (0.0, 0.3, 0.6):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            estimates.append(coordinator.poll(now_s=timestamp, target_manager=manager))

        self.assertFalse(estimates[1].confirmed)
        self.assertTrue(estimates[2].confirmed)
        self.assertEqual(manager.lifecycle, TargetLifecycle.LOCKED)
        coordinator.close()

    def test_repeated_qwen_terminal_mismatch_rejects_stable_candidate(self) -> None:
        client = FakeClient()
        coordinator = self.make_coordinator(client)
        manager = TargetManager()
        spec = TargetSpec(
            "person wearing red",
            category="person",
            hard_attributes=("wearing red",),
        )
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_qwen_negative", uav_id="uav_1")
        negative_review = QwenVisualReview(
            schema_version=1,
            review_id="review_attribute_mismatch_2",
            mission_id="mission_qwen_negative",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=0.6,
            frame_id="frame_review_negative",
            decision=VisualReviewDecision.TARGET_MISMATCH,
            candidate=VisualReviewCandidate(
                True,
                (0.35, 0.2, 0.65, 0.8),
                "person wearing blue",
                0.9,
            ),
            scene_observations=("red clothing is absent",),
            reason_codes=("attribute_mismatch_consensus",),
            recommended_action=VisualReviewAction.CONTINUE,
        )
        coordinator.bind_visual_review_provider(
            lambda candidate_id: (
                negative_review
                if candidate_id == "mission_qwen_negative_uav_1_track_7"
                else None
            )
        )

        for timestamp in (0.0, 0.3, 0.6):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            estimate = coordinator.poll(now_s=timestamp, target_manager=manager)

        self.assertFalse(estimate.confirmed)
        self.assertEqual(manager.lifecycle, TargetLifecycle.SEARCHING)
        self.assertEqual(coordinator.metrics.candidates_rejected, 1)
        coordinator.close()

    def test_new_reacquire_track_requires_qwen_review_with_reference(self) -> None:
        client = FakeClient()
        coordinator = self.make_coordinator(client)
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_reid", uav_id="uav_1")
        for timestamp in (0.0, 0.3, 0.6):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            coordinator.poll(now_s=timestamp, target_manager=manager)
        original_target_id = manager.snapshot().target_id
        manager.start_tracking(0.7)
        manager.mark_lost(timestamp_s=0.8)
        manager.start_reacquiring(0.9)
        client.track_id = 8

        review = QwenVisualReview(
            schema_version=1,
            review_id="review_reid",
            mission_id="mission_reid",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=1.6,
            frame_id="frame_reid",
            decision=VisualReviewDecision.TARGET_MATCH,
            candidate=VisualReviewCandidate(
                True,
                (0.35, 0.2, 0.65, 0.8),
                "same person as reference",
                0.9,
            ),
            scene_observations=("identity consistent with reference",),
            reason_codes=("reid_match",),
            recommended_action=VisualReviewAction.CONTINUE,
        )
        coordinator.bind_visual_review_provider(
            lambda candidate_id: (
                review
                if candidate_id == "mission_reid_uav_1_track_8"
                else None
            ),
            lambda candidate_id: (
                ("frame_verified_reference",)
                if candidate_id == "mission_reid_uav_1_track_8"
                else ()
            ),
        )
        estimates = []
        for timestamp in (1.0, 1.3, 1.6):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            estimates.append(coordinator.poll(now_s=timestamp, target_manager=manager))
            if timestamp == 1.0:
                self.assertTrue(
                    coordinator.qwen_fallback_required(
                        "mission_reid_uav_1_track_8"
                    )
                )

        self.assertFalse(estimates[1].confirmed)
        self.assertTrue(estimates[2].confirmed)
        self.assertEqual(manager.snapshot().target_id, original_target_id)
        self.assertEqual(coordinator.metrics.reacquire_successes, 1)
        self.assertFalse(
            coordinator.qwen_fallback_required("mission_reid_uav_1_track_8")
        )
        coordinator.close()

    def test_new_reacquire_track_can_use_deterministic_color_position_time(self) -> None:
        client = FakeClient()
        app = load_config(ROOT / "configs/default.yaml")
        config = replace(
            app.target_perception,
            backend="ultralytics_service",
            confirmation=replace(
                app.target_perception.confirmation,
                require_qwen_for_reacquire_new_track_id=False,
            ),
        )

        def deterministic_color(candidate, spec, detection, timestamp_s):
            return SemanticVerification(
                candidate_id=candidate.candidate_id,
                timestamp_s=timestamp_s,
                target_description=spec.description,
                matches=True,
                confidence=0.9,
                verifier=DETERMINISTIC_ATTRIBUTE_VERIFIER,
            )

        coordinator = TargetPerceptionCoordinator(
            config,
            client=client,
            executor=InlineExecutor(),
            model_names={0: "person"},
            query_compiler=lambda spec, names: TargetQuery((0,), ()),
            semantic_evidence_provider=deterministic_color,
        )
        manager = TargetManager()
        spec = TargetSpec(
            "red person",
            category="person",
            hard_attributes=("color=red",),
        )
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_deterministic_reid", uav_id="uav_1")
        for timestamp in (0.0, 0.3, 0.6):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            coordinator.poll(now_s=timestamp, target_manager=manager)
        original_target_id = manager.snapshot().target_id
        self.assertIsNotNone(original_target_id)
        manager.start_tracking(0.7)
        manager.mark_lost(timestamp_s=0.8)
        manager.start_reacquiring(0.9)
        client.track_id = 8

        estimates = []
        for timestamp in (1.0, 1.3, 1.6):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            estimates.append(coordinator.poll(now_s=timestamp, target_manager=manager))

        self.assertFalse(estimates[1].confirmed)
        self.assertTrue(estimates[2].confirmed)
        self.assertEqual(manager.snapshot().target_id, original_target_id)
        self.assertEqual(coordinator.metrics.reacquire_successes, 1)
        self.assertFalse(
            coordinator.qwen_fallback_required(
                "mission_deterministic_reid_uav_1_track_8"
            )
        )
        coordinator.close()

    def test_new_reacquire_track_without_each_deterministic_proof_stays_candidate(
        self,
    ) -> None:
        for missing_proof in ("color", "position", "time"):
            with self.subTest(missing_proof=missing_proof):
                client = FakeClient()
                app = load_config(ROOT / "configs/default.yaml")
                config = replace(
                    app.target_perception,
                    backend="ultralytics_service",
                    confirmation=replace(
                        app.target_perception.confirmation,
                        require_qwen_for_reacquire_new_track_id=False,
                    ),
                )

                def semantic(candidate, spec, detection, timestamp_s):
                    return SemanticVerification(
                        candidate_id=candidate.candidate_id,
                        timestamp_s=timestamp_s,
                        target_description=spec.description,
                        matches=True,
                        confidence=0.9,
                        verifier=(
                            "class_only_test"
                            if missing_proof == "color"
                            else DETERMINISTIC_ATTRIBUTE_VERIFIER
                        ),
                    )

                coordinator = TargetPerceptionCoordinator(
                    config,
                    client=client,
                    executor=InlineExecutor(),
                    model_names={0: "person"},
                    query_compiler=lambda spec, names: TargetQuery((0,), ()),
                    semantic_evidence_provider=semantic,
                )
                manager = TargetManager()
                spec = TargetSpec(
                    "red person",
                    category="person",
                    hard_attributes=("color=red",),
                )
                manager.start_search(spec, 0.0)
                coordinator.reset(
                    mission_id=f"mission_missing_{missing_proof}",
                    uav_id="uav_1",
                )
                for timestamp in (0.0, 0.3, 0.6):
                    coordinator.submit_frame(
                        camera_sample=sample(timestamp), target_spec=spec
                    )
                    coordinator.poll(now_s=timestamp, target_manager=manager)
                original_target_id = manager.snapshot().target_id
                manager.start_tracking(0.7)
                manager.mark_lost(timestamp_s=0.8)
                manager.start_reacquiring(0.9)
                client.track_id = 8

                timestamps = (
                    (3.0, 3.3, 3.6)
                    if missing_proof == "time"
                    else (1.0, 1.3, 1.6)
                )
                estimate = None
                for timestamp in timestamps:
                    coordinator.submit_frame(
                        camera_sample=sample(
                            timestamp,
                            depth_m=100.0 if missing_proof == "position" else 10.0,
                        ),
                        target_spec=spec,
                    )
                    estimate = coordinator.poll(
                        now_s=timestamp,
                        target_manager=manager,
                    )

                self.assertIsNotNone(estimate)
                self.assertFalse(estimate.confirmed)
                self.assertEqual(manager.lifecycle, TargetLifecycle.CANDIDATE)
                self.assertNotEqual(manager.snapshot().target_id, original_target_id)
                self.assertEqual(coordinator.metrics.reacquire_successes, 0)
                coordinator.close()

    def test_resolution_failure_reason_is_persisted_and_logs_are_throttled(
        self,
    ) -> None:
        client = FakeClient()
        app = load_config(ROOT / "configs/default.yaml")
        config = replace(app.target_perception, backend="ultralytics_service")
        coordinator = TargetPerceptionCoordinator(
            config,
            client=client,
            resolver=UnavailableResolver("foreground_depth_cluster_empty"),
            executor=InlineExecutor(),
            model_names={0: "person"},
            query_compiler=lambda spec, names: TargetQuery((0,), ()),
        )
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        output = StringIO()
        with redirect_stdout(output):
            coordinator.reset(mission_id="mission_depth_reason", uav_id="uav_1")
            for index in range(10):
                timestamp = index * 0.1
                coordinator.submit_frame(
                    camera_sample=sample(timestamp), target_spec=spec
                )
                coordinator.poll(now_s=timestamp, target_manager=manager)

        reason = (
            "CandidateResolutionUnavailable:foreground_depth_cluster_empty"
        )
        metrics = coordinator.runtime_metrics()
        self.assertEqual(metrics["depth_resolution_failures"], 10)
        self.assertEqual(metrics["measurement_rejected"], 10)
        self.assertEqual(metrics["depth_resolution_last_failure_reason"], reason)
        self.assertEqual(
            metrics["depth_resolution_failure_reason_counts"],
            {reason: 10},
        )
        records = [
            json.loads(line.removeprefix("[PerceptionDepthFailure] "))
            for line in output.getvalue().splitlines()
            if line.startswith("[PerceptionDepthFailure] ")
        ]
        self.assertEqual(
            [record["reason_occurrence"] for record in records],
            [1, 2, 4, 8],
        )
        self.assertTrue(all(record["reason"] == reason for record in records))
        self.assertEqual(manager.lifecycle, TargetLifecycle.CANDIDATE)
        coordinator.close()

    def test_unconfirmed_tracker_cannot_contaminate_locked_control_prediction(self) -> None:
        client = FakeClient()
        coordinator = self.make_coordinator(client)
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_identity_boundary", uav_id="uav_1")
        for timestamp in (0.0, 0.3, 0.6):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            locked = coordinator.poll(now_s=timestamp, target_manager=manager)
        assert locked is not None
        self.assertTrue(locked.confirmed)
        self.assertEqual(locked.tracker_id, "track_7")
        locked_position = locked.position_world_m

        # A same-class distractor appears at a very different depth under a
        # new BoT-SORT ID while TargetManager still owns the original lock.
        client.track_id = 99
        coordinator.submit_frame(
            camera_sample=sample(0.9, depth_m=100.0),
            target_spec=spec,
        )
        estimate = coordinator.poll(now_s=0.9, target_manager=manager)

        assert estimate is not None
        self.assertTrue(estimate.confirmed)
        self.assertTrue(estimate.predicted_only)
        self.assertFalse(estimate.visible)
        self.assertEqual(estimate.tracker_id, "track_7")
        self.assertEqual(estimate.position_world_m, locked_position)
        self.assertEqual(manager.lifecycle, TargetLifecycle.LOCKED)
        coordinator.close()

    def test_rejected_depth_jump_never_bypasses_state_estimator(self) -> None:
        client = FakeClient()
        app = load_config(ROOT / "configs/default.yaml")
        config = replace(app.target_perception, backend="ultralytics_service")
        coordinator = TargetPerceptionCoordinator(
            config,
            client=client,
            executor=InlineExecutor(),
            model_names={0: "person"},
            query_compiler=lambda spec, names: TargetQuery((0,), ()),
            state_estimator=TargetStateEstimator(
                max_position_jump_m=1.0,
                max_prediction_age_s=1.0,
            ),
        )
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_jump_gate", uav_id="uav_1")

        for timestamp in (0.0, 0.3, 0.6):
            coordinator.submit_frame(
                camera_sample=sample(timestamp, depth_m=10.0),
                target_spec=spec,
            )
            accepted = coordinator.poll(now_s=timestamp, target_manager=manager)

        assert accepted is not None
        assert accepted.position_world_m is not None
        coordinator.submit_frame(
            camera_sample=sample(0.9, depth_m=100.0),
            target_spec=spec,
        )
        rejected = coordinator.poll(now_s=0.9, target_manager=manager)

        assert rejected is not None
        assert rejected.position_world_m is not None
        self.assertTrue(rejected.confirmed)
        self.assertTrue(rejected.predicted_only)
        self.assertFalse(rejected.visible)
        self.assertIsNone(rejected.bbox_xyxy_normalized)
        self.assertEqual(rejected.source, "kalman_prediction")
        self.assertGreater(rejected.measurement_age_s, 0.0)
        self.assertLess(
            float(
                np.linalg.norm(
                    np.asarray(rejected.position_world_m)
                    - np.asarray(accepted.position_world_m)
                )
            ),
            1.0,
        )
        self.assertEqual(coordinator.metrics.depth_resolution_failures, 1)
        coordinator.close()

    def test_candidate_cooldown_starts_new_short_track_evidence_epoch(self) -> None:
        client = FakeClient()
        app = load_config(ROOT / "configs/default.yaml")
        config = replace(app.target_perception, backend="ultralytics_service")
        bank = CandidateBank(uav_id="uav_1", rejected_cooldown_s=0.1)
        coordinator = TargetPerceptionCoordinator(
            config,
            client=client,
            executor=InlineExecutor(),
            candidate_bank=bank,
            model_names={0: "person"},
            query_compiler=lambda spec, names: TargetQuery((0,), ()),
        )
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_epoch", uav_id="uav_1")

        for timestamp in (0.0, 0.3):
            coordinator.submit_frame(
                camera_sample=sample(timestamp),
                target_spec=spec,
            )
            estimate = coordinator.poll(now_s=timestamp, target_manager=manager)
        assert estimate is not None and estimate.candidate_id is not None
        candidate_id = estimate.candidate_id
        bank.reject(candidate_id, timestamp_s=0.31)

        coordinator.submit_frame(camera_sample=sample(0.7), target_spec=spec)
        restarted = coordinator.poll(now_s=0.7, target_manager=manager)

        assert restarted is not None
        self.assertFalse(restarted.confirmed)
        self.assertEqual(
            coordinator._track_snapshots[candidate_id][-1].observation_count,
            1,
        )
        snapshot = bank.get(candidate_id)
        assert snapshot is not None
        self.assertEqual(snapshot.first_seen_timestamp_s, 0.7)
        self.assertEqual(len(snapshot.frame_history), 1)
        self.assertEqual(snapshot.review_history, ())
        coordinator.close()

    def test_rejected_high_confidence_candidate_does_not_block_next_detection(self) -> None:
        client = FakeClient()
        client.detections_override = (
            TrackDetection(7, 0, "person", 0.95, (0.05, 0.2, 0.35, 0.8)),
            TrackDetection(8, 0, "person", 0.85, (0.60, 0.2, 0.90, 0.8)),
        )
        app = load_config(ROOT / "configs/default.yaml")
        config = replace(app.target_perception, backend="ultralytics_service")

        def semantic(candidate, spec, detection, timestamp_s):
            return SemanticVerification(
                candidate_id=candidate.candidate_id,
                timestamp_s=timestamp_s,
                target_description=spec.description,
                matches=detection.track_id == 8,
                confidence=0.9,
                verifier="test_semantic",
            )

        coordinator = TargetPerceptionCoordinator(
            config,
            client=client,
            executor=InlineExecutor(),
            model_names={0: "person"},
            query_compiler=lambda spec, names: TargetQuery((0,), ()),
            semantic_evidence_provider=semantic,
        )
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_multi", uav_id="uav_1")

        estimates = []
        for timestamp in (0.0, 0.3, 0.6, 0.9, 1.2):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            estimates.append(coordinator.poll(now_s=timestamp, target_manager=manager))

        self.assertEqual(coordinator.metrics.candidates_rejected, 1)
        self.assertEqual(coordinator.metrics.candidates_confirmed, 1)
        self.assertTrue(estimates[-1].confirmed)
        self.assertEqual(estimates[-1].tracker_id, "track_8")
        self.assertEqual(manager.lifecycle, TargetLifecycle.LOCKED)
        coordinator.close()

    def test_qwen_required_mode_never_uses_class_only_confirmation(self) -> None:
        client = FakeClient()
        app = load_config(ROOT / "configs/default.yaml")
        config = replace(
            app.target_perception,
            backend="ultralytics_service",
            confirmation=replace(
                app.target_perception.confirmation,
                mode="qwen_required",
            ),
        )
        coordinator = TargetPerceptionCoordinator(
            config,
            client=client,
            executor=InlineExecutor(),
            model_names={0: "person"},
            query_compiler=lambda spec, names: TargetQuery((0,), ()),
        )
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_qwen_required", uav_id="uav_1")
        for timestamp in (0.0, 0.3, 0.6, 0.9):
            coordinator.submit_frame(camera_sample=sample(timestamp), target_spec=spec)
            estimate = coordinator.poll(now_s=timestamp, target_manager=manager)

        self.assertFalse(estimate.confirmed)
        self.assertEqual(manager.lifecycle, TargetLifecycle.CANDIDATE)
        self.assertEqual(coordinator.metrics.candidates_confirmed, 0)
        coordinator.close()

    def test_poll_separates_availability_and_response_error_metrics(self) -> None:
        class FailingClient(FakeClient):
            def __init__(self, failure: Exception | None, *, mismatched: bool = False):
                super().__init__()
                self.failure = failure
                self.mismatched = mismatched

            def track(self, request, rgb) -> TrackResponse:
                if self.failure is not None:
                    raise self.failure
                response = super().track(request, rgb)
                if self.mismatched:
                    return replace(response, request_id="request_wrong_route")
                return response

        cases = (
            (FailingClient(YoloClientUnavailable("offline")), 1, 0, False),
            (FailingClient(TimeoutError("deadline")), 1, 0, False),
            (FailingClient(YoloClientResponseError("HTTP 422")), 0, 1, True),
            (FailingClient(None, mismatched=True), 0, 1, True),
            (FailingClient(RuntimeError("internal")), 0, 0, True),
        )
        for index, (
            client,
            expected_timeouts,
            expected_response_errors,
            fatal,
        ) in enumerate(cases):
            with self.subTest(index=index):
                coordinator = self.make_coordinator(client)
                manager = TargetManager()
                spec = TargetSpec("person", category="person")
                manager.start_search(spec, 0.0)
                coordinator.reset(
                    mission_id=f"mission_metric_{index}",
                    uav_id="uav_1",
                )
                coordinator.submit_frame(
                    camera_sample=sample(0.0),
                    target_spec=spec,
                )
                if fatal:
                    with self.assertRaises(TargetPerceptionError):
                        coordinator.poll(now_s=0.0, target_manager=manager)
                else:
                    coordinator.poll(now_s=0.0, target_manager=manager)

                metrics = coordinator.metrics.to_dict()
                self.assertEqual(metrics["yolo_timeouts"], expected_timeouts)
                self.assertEqual(
                    metrics["yolo_response_errors"],
                    expected_response_errors,
                )
                coordinator.close()

    def test_consecutive_service_unavailability_fails_closed_on_third_request(self) -> None:
        class OfflineClient(FakeClient):
            def track(self, request, rgb) -> TrackResponse:
                del request, rgb
                raise YoloClientUnavailable("service offline")

        coordinator = self.make_coordinator(OfflineClient())
        manager = TargetManager()
        spec = TargetSpec("person", category="person")
        manager.start_search(spec, 0.0)
        coordinator.reset(mission_id="mission_offline", uav_id="uav_1")

        for timestamp in (0.0, 0.1):
            coordinator.submit_frame(
                camera_sample=sample(timestamp),
                target_spec=spec,
            )
            self.assertIsNone(
                coordinator.poll(now_s=timestamp, target_manager=manager)
            )

        coordinator.submit_frame(camera_sample=sample(0.2), target_spec=spec)
        with self.assertRaisesRegex(
            TargetPerceptionError,
            "3 consecutive availability failures",
        ):
            coordinator.poll(now_s=0.2, target_manager=manager)
        with self.assertRaisesRegex(TargetPerceptionError, "failed closed"):
            coordinator.submit_frame(
                camera_sample=sample(0.3),
                target_spec=spec,
            )
        self.assertEqual(coordinator.metrics.yolo_timeouts, 3)
        coordinator.close()


if __name__ == "__main__":
    unittest.main()
