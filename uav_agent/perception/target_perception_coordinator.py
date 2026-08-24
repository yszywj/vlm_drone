"""Bounded asynchronous target-perception pipeline.

The coordinator owns model I/O, candidate evidence, trusted RGB-D geometry and
world-state filtering outside MissionAgent and Skill code.  It keeps at most
one request in flight and one newest pending frame per UAV.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from numbers import Real
from statistics import fmean
from threading import RLock
from time import monotonic, sleep

from common.ids import generate_routing_id, validate_mission_id, validate_uav_id
from common.target_estimate import TargetEstimate
from configs.schema import TargetPerceptionConfig
from env.camera_types import CameraSample
from perception.candidate_bank import CandidateBank, CandidateLifecycle, CandidateSnapshot
from perception.class_aliases import (
    ClassAliasMapper,
    UnsupportedTargetCategory,
    compile_target_query,
)
from perception.confirmation import (
    CandidateConfirmationCoordinator,
    ConfirmationDecision,
    ConfirmationPolicy,
)
from perception.depth_geometry import DepthCandidateResolver
from perception.grounding import (
    CandidateResolutionUnavailable,
    GroundingProposal,
    UltralyticsGrounder,
)
from perception.target_state_estimator import (
    TargetStateEstimator,
    TargetStateMeasurementRejected,
)
from perception.target_debug_images import (
    BoundedTargetDebugImageWriter,
    TargetDebugAnnotation,
)
from perception.types import (
    DetectionCandidate,
    IdentityConsistencyEvidence,
    SemanticVerification,
    ShortTrackEvidence,
)
from perception.yolo_client import (
    YoloClientResponseError,
    YoloClientUnavailable,
    YoloServiceClient,
)
from perception.visual_review import QwenVisualReview
from perception.visual_evidence import (
    ClosedSetClassSemanticVerifier,
    QwenEvidencePending,
    QwenReacquireIdentityVerifierAdapter,
    QwenSemanticVerifierAdapter,
    ReacquireIdentityRequiresQwen,
    SemanticVerificationRequiresQwen,
    TemporalTrackIdentityVerifier,
    UltralyticsShortTrackEvidenceBuilder,
    VisualEvidenceError,
)
from runtime.frame_store import FrameRef, FrameStore
from target import TargetLifecycle, TargetManager
from target.types import TargetSpec
from yolo_service.protocol import (
    ProtocolValidationError,
    ResetStreamRequest,
    TargetQuery,
    TrackDetection,
    TrackRequest,
    TrackResponse,
)


class TargetPerceptionError(RuntimeError):
    """Trusted integration failure that never triggers an Oracle fallback."""


class TargetPerceptionNotReady(TargetPerceptionError):
    """Raised when submit occurs before a mission reset."""


class TargetQueryUnsupported(TargetPerceptionError):
    code = "UNSUPPORTED_TARGET_CATEGORY"


_MAX_CONSECUTIVE_AVAILABILITY_FAILURES = 3


SemanticEvidenceProvider = Callable[
    [CandidateSnapshot, TargetSpec, TrackDetection, float], SemanticVerification | None
]
IdentityEvidenceProvider = Callable[
    [CandidateSnapshot, str, int, bool, float], IdentityConsistencyEvidence | None
]
VisualReviewProvider = Callable[[str], QwenVisualReview | None]
VisualReviewReferenceProvider = Callable[[str], Sequence[str]]


@dataclass(slots=True)
class TargetPerceptionMetrics:
    yolo_requests: int = 0
    yolo_successful_responses: int = 0
    yolo_timeouts: int = 0
    yolo_response_errors: int = 0
    yolo_stale_results: int = 0
    yolo_dropped_frames: int = 0
    detections_total: int = 0
    candidates_total: int = 0
    candidates_rejected: int = 0
    candidates_confirmed: int = 0
    track_id_switches: int = 0
    track_fragmentations: int = 0
    target_visible_frames: int = 0
    target_total_frames: int = 0
    target_lost_count: int = 0
    reacquire_attempts: int = 0
    reacquire_successes: int = 0
    depth_resolution_failures: int = 0
    _latencies_ms: list[float] = field(default_factory=list, repr=False)
    _measurement_ages_s: list[float] = field(default_factory=list, repr=False)
    _position_squared_errors: list[float] = field(default_factory=list, repr=False)
    _velocity_squared_errors: list[float] = field(default_factory=list, repr=False)

    def record_latency(self, value_ms: float) -> None:
        value = _nonnegative(value_ms, "latency_ms")
        self._latencies_ms.append(value)
        del self._latencies_ms[:-10_000]

    def record_measurement_age(self, value_s: float) -> None:
        self._measurement_ages_s.append(_nonnegative(value_s, "measurement_age_s"))
        del self._measurement_ages_s[:-10_000]

    def record_evaluator_error(
        self,
        *,
        position_error_m: float | None,
        velocity_error_mps: float | None,
        evaluator_mode: bool,
    ) -> None:
        if evaluator_mode is not True:
            raise PermissionError(
                "ground-truth RMSE metrics are evaluator-only and cannot feed runtime"
            )
        if position_error_m is None and velocity_error_mps is None:
            raise ValueError("at least one evaluator error must be provided")
        if position_error_m is not None:
            position = _nonnegative(position_error_m, "position_error_m")
            self._position_squared_errors.append(position * position)
            del self._position_squared_errors[:-10_000]
        if velocity_error_mps is not None:
            velocity = _nonnegative(velocity_error_mps, "velocity_error_mps")
            self._velocity_squared_errors.append(velocity * velocity)
            del self._velocity_squared_errors[:-10_000]

    def to_dict(self) -> dict[str, int | float | None]:
        latency_sorted = sorted(self._latencies_ms)
        p95 = (
            None
            if not latency_sorted
            else latency_sorted[min(len(latency_sorted) - 1, int(0.95 * len(latency_sorted)))]
        )
        return {
            "yolo_requests": self.yolo_requests,
            "yolo_successful_responses": self.yolo_successful_responses,
            "yolo_timeouts": self.yolo_timeouts,
            "yolo_response_errors": self.yolo_response_errors,
            "yolo_stale_results": self.yolo_stale_results,
            "yolo_dropped_frames": self.yolo_dropped_frames,
            "yolo_inference_latency_ms_mean": _mean_or_none(self._latencies_ms),
            "yolo_inference_latency_ms_p95": p95,
            "detections_total": self.detections_total,
            "candidates_total": self.candidates_total,
            "candidates_rejected": self.candidates_rejected,
            "candidates_confirmed": self.candidates_confirmed,
            "track_id_switches": self.track_id_switches,
            "track_fragmentations": self.track_fragmentations,
            "target_visible_ratio": (
                0.0
                if self.target_total_frames == 0
                else self.target_visible_frames / self.target_total_frames
            ),
            "target_lost_count": self.target_lost_count,
            "reacquire_attempts": self.reacquire_attempts,
            "reacquire_successes": self.reacquire_successes,
            "depth_resolution_failures": self.depth_resolution_failures,
            "position_measurement_age_mean": _mean_or_none(self._measurement_ages_s),
            "position_rmse_m": _rmse_or_none(self._position_squared_errors),
            "velocity_rmse_mps": _rmse_or_none(self._velocity_squared_errors),
        }


@dataclass(frozen=True, slots=True)
class _Submission:
    request: TrackRequest
    frame_ref: FrameRef
    target_spec: TargetSpec


@dataclass(frozen=True, slots=True)
class _Inflight:
    submission: _Submission
    future: Future[TrackResponse]


class TargetPerceptionCoordinator:
    """One-mission, one-UAV target perception state machine."""

    def __init__(
        self,
        config: TargetPerceptionConfig,
        *,
        client: YoloServiceClient | object | None = None,
        frame_store: FrameStore | None = None,
        candidate_bank: CandidateBank | None = None,
        resolver: DepthCandidateResolver | None = None,
        state_estimator: TargetStateEstimator | None = None,
        executor: Executor | None = None,
        model_names: Mapping[int, str] | Sequence[str] | None = None,
        query_compiler: Callable[[TargetSpec, Mapping[int, str]], TargetQuery] | None = None,
        semantic_evidence_provider: SemanticEvidenceProvider | None = None,
        identity_evidence_provider: IdentityEvidenceProvider | None = None,
        visual_review_provider: VisualReviewProvider | None = None,
        visual_review_reference_provider: VisualReviewReferenceProvider | None = None,
        debug_image_writer: BoundedTargetDebugImageWriter | object | None = None,
    ) -> None:
        if not isinstance(config, TargetPerceptionConfig):
            raise TypeError("config must be a TargetPerceptionConfig")
        if config.backend != "ultralytics_service":
            raise ValueError(
                "TargetPerceptionCoordinator requires ultralytics_service backend"
            )
        self._config = config
        self._client = client or YoloServiceClient(
            base_url=config.yolo_service.url,
            request_timeout_s=config.yolo_service.request_timeout_s,
            jpeg_quality=config.yolo_service.jpeg_quality,
        )
        self._frame_store = frame_store or FrameStore()
        if candidate_bank is not None and not isinstance(candidate_bank, CandidateBank):
            raise TypeError("candidate_bank must be a CandidateBank or None")
        self._provided_candidate_bank = candidate_bank
        self._resolver = resolver or DepthCandidateResolver(
            self._frame_store,
            sampling_strategy=config.geometry.depth_anchor,
            patch_radius_px=config.geometry.depth_patch_radius_px,
            min_depth_m=config.geometry.min_depth_m,
            max_depth_m=config.geometry.max_depth_m,
        )
        estimator_cfg = config.state_estimator
        self._estimator = state_estimator or TargetStateEstimator(
            max_prediction_age_s=estimator_cfg.max_prediction_age_s,
            max_position_jump_m=estimator_cfg.max_position_jump_m,
            process_noise=estimator_cfg.process_noise,
            measurement_noise=estimator_cfg.measurement_noise,
        )
        self._confirmation = CandidateConfirmationCoordinator(
            ConfirmationPolicy(
                min_track_observations=config.tracker.min_track_observations,
                min_track_duration_s=config.tracker.min_track_duration_s,
                min_track_confidence=config.detector.confidence_threshold,
                min_semantic_confidence=0.5,
                min_identity_confidence=0.5,
                min_consistent_observations=config.tracker.min_track_observations,
            )
        )
        self._alias_mapper = ClassAliasMapper.from_yaml(
            config.detector.class_aliases_path
        )
        self._track_evidence = UltralyticsShortTrackEvidenceBuilder(
            min_observations=config.tracker.min_track_observations,
            min_duration_s=config.tracker.min_track_duration_s,
        )
        self._identity_verifier = TemporalTrackIdentityVerifier()
        self._qwen_semantic_adapter = QwenSemanticVerifierAdapter()
        self._qwen_identity_adapter = QwenReacquireIdentityVerifierAdapter()
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="yolo-target",
        )
        self._model_names = _normalize_model_names(model_names)
        self._query_compiler = query_compiler
        self._semantic_provider = semantic_evidence_provider
        self._identity_provider = identity_evidence_provider
        if visual_review_provider is not None and not callable(visual_review_provider):
            raise TypeError("visual_review_provider must be callable or None")
        if (
            visual_review_reference_provider is not None
            and not callable(visual_review_reference_provider)
        ):
            raise TypeError(
                "visual_review_reference_provider must be callable or None"
            )
        self._visual_review_provider = visual_review_provider
        self._visual_review_reference_provider = visual_review_reference_provider
        if debug_image_writer is not None and not callable(
            getattr(debug_image_writer, "capture", None)
        ):
            raise TypeError("debug_image_writer must provide capture or be None")
        self._debug_image_writer = debug_image_writer
        self._mission_id: str | None = None
        self._uav_id: str | None = None
        self._stream_id: str | None = None
        self._candidate_bank: CandidateBank | None = None
        self._frame_sequence = 0
        self._inflight: _Inflight | None = None
        self._pending: _Submission | None = None
        self._latest_estimate: TargetEstimate | None = None
        # Prediction used for control may inherit state only from a confirmed
        # logical identity.  The newest detector result can instead be an
        # unconfirmed same-class distractor.
        self._latest_confirmed_estimate: TargetEstimate | None = None
        self._latest_confirmed_frame_ref: FrameRef | None = None
        self._track_snapshots: dict[str, list[ShortTrackEvidence]] = {}
        self._locked_tracker_id: int | None = None
        self._estimator_tracker_id: int | None = None
        self._last_observed_tracker_id: int | None = None
        self._last_target_lifecycle: TargetLifecycle | None = None
        self._expected_reacquire_target_id: str | None = None
        self._identity_reference_refs: tuple[FrameRef, ...] = ()
        self._last_error: str | None = None
        self._fatal_error: str | None = None
        self._consecutive_availability_failures = 0
        self._closed = False
        self._lock = RLock()
        self.metrics = TargetPerceptionMetrics()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def latest_estimate(self) -> TargetEstimate | None:
        return self._latest_estimate

    @property
    def frame_store(self) -> FrameStore:
        return self._frame_store

    def reset(self, *, mission_id: str, uav_id: str) -> None:
        mission = validate_mission_id(mission_id)
        uav = validate_uav_id(uav_id)
        with self._lock:
            self._ensure_open()
            previous_mission = self._mission_id
            previous_uav = self._uav_id
            previous_stream = self._stream_id
            inflight = self._inflight
            self._inflight = None
            if inflight is not None:
                inflight.future.cancel()
            self._pending = None
            # A failed handshake must not leave a coordinator that appears
            # ready.  Publish routing state only after health/reset/model-info
            # have all succeeded.
            self._mission_id = None
            self._uav_id = None
            self._stream_id = None
            self._candidate_bank = None
            self._fatal_error = None
            self._consecutive_availability_failures = 0
            if previous_uav is not None:
                self._frame_store.clear(uav_id=previous_uav)
            if self._provided_candidate_bank is not None:
                if self._provided_candidate_bank.uav_id != uav:
                    raise ValueError("candidate_bank uav_id does not match reset uav_id")
            self._last_error = None

        cleanup_wait_s = self._cleanup_wait_s()
        self._drain_future(inflight, timeout_s=cleanup_wait_s)
        try:
            # A reused coordinator must first retire the previous persistent
            # BoT-SORT stream.  Refuse the next mission if this cannot be done
            # within the bounded cleanup interval.
            if (
                previous_mission is not None
                and previous_uav is not None
                and previous_stream is not None
            ):
                self._reset_stream_with_retry(
                    mission_id=previous_mission,
                    uav_id=previous_uav,
                    stream_id=previous_stream,
                    timeout_s=cleanup_wait_s,
                )
            health = self._client.health()
            if health.get("ready") is not True:
                raise YoloClientResponseError("YOLO service is not ready")
            self._reset_stream_with_retry(
                mission_id=mission,
                uav_id=uav,
                stream_id=f"{mission}:{uav}",
                timeout_s=cleanup_wait_s,
            )
            info = self._client.model_info()
            expected_family = self._config.detector.model_family
            if info.model_family != expected_family:
                raise YoloClientResponseError(
                    "YOLO model family mismatch: "
                    f"service={info.model_family!r}, config={expected_family!r}"
                )
            model_names = _normalize_model_names(info.names)
        except Exception as exc:
            message = _safe_error("yolo_startup_handshake_failed", exc)
            with self._lock:
                self._last_error = message
            raise TargetPerceptionError(
                f"YOLO startup handshake failed: {type(exc).__name__}: {exc}"
            ) from exc

        with self._lock:
            self._ensure_open()
            self._mission_id = mission
            self._uav_id = uav
            self._stream_id = f"{mission}:{uav}"
            self._model_names = model_names
            if self._provided_candidate_bank is not None:
                self._provided_candidate_bank.clear()
                self._candidate_bank = self._provided_candidate_bank
            else:
                self._candidate_bank = CandidateBank(uav_id=uav)
            self._frame_sequence = 0
            self._pending = None
            self._latest_estimate = None
            self._latest_confirmed_estimate = None
            self._latest_confirmed_frame_ref = None
            self._track_evidence.reset()
            self._track_snapshots.clear()
            self._locked_tracker_id = None
            self._estimator_tracker_id = None
            self._last_observed_tracker_id = None
            self._last_target_lifecycle = None
            self._expected_reacquire_target_id = None
            self._identity_reference_refs = ()
            self._last_error = None
            self._fatal_error = None
            self._consecutive_availability_failures = 0
            self._estimator.reset()

    def submit_frame(
        self,
        *,
        camera_sample: CameraSample,
        target_spec: TargetSpec,
    ) -> None:
        if not isinstance(camera_sample, CameraSample):
            raise TypeError("camera_sample must be a CameraSample")
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        with self._lock:
            self._ensure_ready()
            assert self._mission_id is not None
            assert self._uav_id is not None
            assert self._stream_id is not None
            self._frame_sequence += 1
            frame_id = f"frame_{self._frame_sequence:010d}"
            frame_ref = self._frame_store.add_sample(
                uav_id=self._uav_id,
                frame_id=frame_id,
                sample=camera_sample,
            )
            query = self._compile_query(target_spec)
            submission = _Submission(
                TrackRequest(
                    schema_version=1,
                    request_id=generate_routing_id("request"),
                    mission_id=self._mission_id,
                    uav_id=self._uav_id,
                    stream_id=self._stream_id,
                    frame_id=frame_id,
                    timestamp_s=camera_sample.timestamp_s,
                    target_query=query,
                ),
                frame_ref,
                target_spec,
            )
            if self._inflight is None:
                self._launch(submission)
            else:
                if self._pending is not None:
                    self.metrics.yolo_dropped_frames += 1
                self._pending = submission

    def poll(
        self,
        *,
        now_s: float,
        target_manager: TargetManager,
    ) -> TargetEstimate | None:
        now = _nonnegative(now_s, "now_s")
        if not isinstance(target_manager, TargetManager):
            raise TypeError("target_manager must be a TargetManager")
        with self._lock:
            self._ensure_ready()
            self._record_lifecycle(
                target_manager.snapshot().lifecycle,
                timestamp_s=now,
            )
            completed = self._inflight
            if completed is not None and completed.future.done():
                self._inflight = None
                try:
                    response = completed.future.result()
                    response.assert_matches(completed.submission.request)
                    self._consecutive_availability_failures = 0
                    self.metrics.yolo_successful_responses += 1
                    self.metrics.record_latency(response.timing_ms.inference)
                    if now - response.timestamp_s > self._config.yolo_service.max_result_age_s:
                        self.metrics.yolo_stale_results += 1
                        self._last_error = "stale_yolo_result"
                    else:
                        self._process_response(
                            response,
                            completed.submission,
                            target_manager,
                            now,
                        )
                        self._record_lifecycle(
                            target_manager.snapshot().lifecycle,
                            timestamp_s=now,
                        )
                except Exception as exc:
                    if isinstance(exc, (TimeoutError, YoloClientUnavailable)):
                        self.metrics.yolo_timeouts += 1
                        self._consecutive_availability_failures += 1
                    elif isinstance(
                        exc,
                        (YoloClientResponseError, ProtocolValidationError),
                    ):
                        self.metrics.yolo_response_errors += 1
                    self._last_error = _safe_error("yolo_request_failed", exc)

                    # Availability failures can be transient, so tolerate a
                    # strictly bounded streak while the existing trusted
                    # prediction ages out.  Protocol/HTTP contract failures
                    # and unexpected processing faults are immediately fatal:
                    # continuing would reinterpret corrupt model evidence as
                    # an ordinary "not found" result.
                    fatal = not isinstance(
                        exc,
                        (TimeoutError, YoloClientUnavailable),
                    ) or (
                        self._consecutive_availability_failures
                        >= _MAX_CONSECUTIVE_AVAILABILITY_FAILURES
                    )
                    if fatal:
                        self._pending = None
                        self._fatal_error = self._last_error
                        raise TargetPerceptionError(
                            "YOLO perception failed closed after "
                            f"{self._consecutive_availability_failures} "
                            "consecutive availability failures"
                            if isinstance(
                                exc,
                                (TimeoutError, YoloClientUnavailable),
                            )
                            else "YOLO perception contract or processing failure"
                        ) from exc
                if self._pending is not None:
                    pending, self._pending = self._pending, None
                    self._launch(pending)

            estimate = self._estimate_for_now(now, target_manager)
            self._latest_estimate = estimate
            self.metrics.target_total_frames += 1
            if estimate is not None and estimate.visible:
                self.metrics.target_visible_frames += 1
            return estimate

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            mission, uav, stream = self._mission_id, self._uav_id, self._stream_id
            inflight = self._inflight
            self._inflight = None
            if inflight is not None:
                inflight.future.cancel()
            self._pending = None
            self._estimator.reset()
            self._track_evidence.reset()
            self._track_snapshots.clear()
            if uav is not None:
                self._frame_store.clear(uav_id=uav)
            self._closed = True
        # A running executor future cannot be cancelled.  Drain it for a
        # strictly bounded cleanup interval before asking the single-stream
        # service to destroy its persistent BoT-SORT state.
        cleanup_wait_s = self._cleanup_wait_s()
        self._drain_future(inflight, timeout_s=cleanup_wait_s)
        if mission is not None and uav is not None and stream is not None:
            try:
                self._reset_stream_with_retry(
                    mission_id=mission,
                    uav_id=uav,
                    stream_id=stream,
                    timeout_s=cleanup_wait_s,
                )
            except Exception as exc:
                self._last_error = _safe_error("yolo_cleanup_reset_failed", exc)
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _launch(self, submission: _Submission) -> None:
        rgb = self._frame_store.get_frame(submission.frame_ref)
        if rgb is None:
            self.metrics.yolo_dropped_frames += 1
            self._last_error = "frame_evicted_before_yolo_submit"
            return
        self.metrics.yolo_requests += 1
        future = self._executor.submit(
            self._client.track,
            submission.request,
            rgb,
        )
        self._inflight = _Inflight(submission, future)

    def bind_visual_review_provider(
        self,
        provider: VisualReviewProvider,
        reference_handles_provider: VisualReviewReferenceProvider | None = None,
    ) -> None:
        """Attach the existing asynchronous Qwen review gate as typed evidence."""

        if not callable(provider):
            raise TypeError("provider must be callable")
        if reference_handles_provider is not None and not callable(
            reference_handles_provider
        ):
            raise TypeError("reference_handles_provider must be callable or None")
        with self._lock:
            self._ensure_open()
            if self._visual_review_provider is not None:
                raise TargetPerceptionError("visual review provider is already bound")
            self._visual_review_provider = provider
            self._visual_review_reference_provider = reference_handles_provider

    def bind_debug_image_writer(
        self,
        writer: BoundedTargetDebugImageWriter | object,
    ) -> None:
        """Attach one production-only, opt-in representative-image sink."""

        if not callable(getattr(writer, "capture", None)):
            raise TypeError("writer must provide capture")
        with self._lock:
            self._ensure_open()
            if self._debug_image_writer is not None:
                raise TargetPerceptionError("debug image writer is already bound")
            self._debug_image_writer = writer

    def _process_response(
        self,
        response: TrackResponse,
        submission: _Submission,
        target_manager: TargetManager,
        now_s: float,
    ) -> None:
        self.metrics.detections_total += len(response.detections)
        if not response.detections:
            return
        proposals = UltralyticsGrounder.from_response(
            response,
            submission.frame_ref,
        )
        snapshot = target_manager.snapshot()
        active_candidate_id = (
            snapshot.target_id
            if snapshot.lifecycle is TargetLifecycle.CANDIDATE
            else None
        )
        pairs = list(zip(response.detections, proposals))
        # Continue the active evidence track first.  If it is rejected in this
        # response, later proposals can immediately register the next
        # candidate instead of one high-confidence distractor blocking all
        # alternatives indefinitely.
        pairs.sort(
            key=lambda pair: (
                pair[1].candidate_id == active_candidate_id,
                pair[0].track_id == self._locked_tracker_id,
                pair[0].confidence,
                -pair[0].track_id,
            ),
            reverse=True,
        )
        outcomes: list[tuple[TargetEstimate, TrackDetection]] = []
        for detection, proposal in pairs:
            estimate = self._process_detection(
                detection=detection,
                proposal=proposal,
                submission=submission,
                target_manager=target_manager,
                now_s=now_s,
            )
            if estimate is not None:
                outcomes.append((estimate, detection))
        if not outcomes:
            return

        final_snapshot = target_manager.snapshot()
        estimate, selected_detection = max(
            outcomes,
            key=lambda outcome: (
                outcome[0].confirmed
                and final_snapshot.target_id is not None
                and outcome[0].target_id == final_snapshot.target_id,
                final_snapshot.lifecycle is TargetLifecycle.CANDIDATE
                and outcome[0].candidate_id == final_snapshot.target_id,
                0.0
                if outcome[0].confidence is None
                else outcome[0].confidence,
                -outcome[1].track_id,
            ),
        )
        control_related = estimate.confirmed or (
            final_snapshot.lifecycle is TargetLifecycle.CANDIDATE
            and estimate.candidate_id == final_snapshot.target_id
        )
        if control_related:
            if (
                self._last_observed_tracker_id is not None
                and self._last_observed_tracker_id != selected_detection.track_id
            ):
                self.metrics.track_id_switches += 1
            self._last_observed_tracker_id = selected_detection.track_id
        self._latest_estimate = estimate
        if estimate.confirmed:
            self._latest_confirmed_estimate = estimate
            self._latest_confirmed_frame_ref = submission.frame_ref

    def _process_detection(
        self,
        *,
        detection: TrackDetection,
        proposal: GroundingProposal,
        submission: _Submission,
        target_manager: TargetManager,
        now_s: float,
    ) -> TargetEstimate | None:
        timestamp_s = submission.request.timestamp_s
        self._capture_debug_image(
            "first_detection",
            frame_ref=submission.frame_ref,
            detection=detection,
            candidate_id=None,
            estimate=None,
        )
        bank = self._require_candidate_bank()
        previous_candidate = bank.get(proposal.candidate_id)
        candidate = bank.propose(
            candidate_id=proposal.candidate_id,
            timestamp_s=timestamp_s,
            bbox_xyxy_normalized=proposal.bbox_xyxy_normalized,
            frame_ref=proposal.frame_ref,
            source=proposal.source,
        )
        if candidate is None:
            return None
        starts_new_evidence_epoch = bool(
            previous_candidate is not None
            and previous_candidate.lifecycle
            in {CandidateLifecycle.REJECTED, CandidateLifecycle.STALE}
            and candidate.first_seen_timestamp_s == timestamp_s
        )
        if starts_new_evidence_epoch:
            self._track_evidence.reset(candidate.candidate_id)
            self._track_snapshots.pop(candidate.candidate_id, None)
        track = self._track_evidence.update(
            candidate_id=candidate.candidate_id,
            timestamp_s=timestamp_s,
            detection=detection,
        )
        snapshots = self._track_snapshots.setdefault(candidate.candidate_id, [])
        snapshots.append(track)
        del snapshots[:-32]
        position = self._resolve_position(
            candidate,
            timestamp_s,
            now_s=now_s,
        )
        candidates_before = self.metrics.candidates_total
        rejected_before = self.metrics.candidates_rejected
        confirmed_before = self.metrics.candidates_confirmed
        reacquired_before = self.metrics.reacquire_successes
        confirmed, target_id = self._confirmation_state(
            target_manager=target_manager,
            candidate=candidate,
            detection=detection,
            target_spec=submission.target_spec,
            position=position,
            track=track,
        )
        # The 3-D estimator is a flight-control input.  Never contaminate it
        # with a provisional candidate, including a new tracker observed while
        # another target remains LOCKED/TRACKING.
        filtered = None
        if confirmed and position is not None:
            try:
                if self._estimator_tracker_id != detection.track_id:
                    self._estimator.reset()
                    self._estimator_tracker_id = detection.track_id
                filtered = self._estimator.update(
                    timestamp_s=timestamp_s,
                    position_world_m=position,
                    confidence=detection.confidence,
                )
                self.metrics.record_measurement_age(filtered.measurement_age_s)
            except (TargetStateMeasurementRejected, ValueError):
                self.metrics.depth_resolution_failures += 1

                # Rejected depth/geometry measurements are untrusted control
                # inputs.  Never fall back to the raw position after the
                # estimator's innovation or timestamp gate rejects it.  A
                # bounded Kalman prediction is safe to expose when one is
                # available; otherwise publish no 3-D control position for
                # this visible detection.
                try:
                    filtered = self._estimator.predict(timestamp_s)
                except ValueError:
                    filtered = None
                if filtered is not None:
                    self.metrics.record_measurement_age(
                        filtered.measurement_age_s
                    )

        state = filtered
        prediction_only = bool(
            confirmed and state is not None and state.predicted_only
        )
        estimate = TargetEstimate(
            timestamp_s=timestamp_s,
            target_id=target_id if confirmed else None,
            candidate_id=candidate.candidate_id,
            tracker_id=f"track_{detection.track_id}",
            visible=not prediction_only,
            confirmed=confirmed,
            predicted_only=prediction_only,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox_xyxy_normalized=(
                None
                if prediction_only
                else detection.bbox_xyxy_normalized
            ),
            position_world_m=(
                (
                    position
                    if not confirmed
                    else None
                )
                if state is None
                else state.position_world_m
            ),
            velocity_world_mps=(None if state is None else state.velocity_world_mps),
            measurement_age_s=(0.0 if state is None else state.measurement_age_s),
            source=("kalman_prediction" if prediction_only else self._source_name),
        )
        if self.metrics.candidates_total > candidates_before:
            self._capture_debug_image(
                "first_candidate",
                frame_ref=submission.frame_ref,
                detection=detection,
                candidate_id=candidate.candidate_id,
                estimate=estimate,
            )
        if self.metrics.candidates_rejected > rejected_before:
            self._capture_debug_image(
                "candidate_rejected",
                frame_ref=submission.frame_ref,
                detection=detection,
                candidate_id=candidate.candidate_id,
                estimate=estimate,
            )
        if self.metrics.candidates_confirmed > confirmed_before:
            self._capture_debug_image(
                "confirmation_success",
                frame_ref=submission.frame_ref,
                detection=detection,
                candidate_id=candidate.candidate_id,
                estimate=estimate,
            )
        if self.metrics.reacquire_successes > reacquired_before:
            self._capture_debug_image(
                "reacquire_success",
                frame_ref=submission.frame_ref,
                detection=detection,
                candidate_id=candidate.candidate_id,
                estimate=estimate,
            )
        return estimate

    def _confirmation_state(
        self,
        *,
        target_manager: TargetManager,
        candidate: CandidateSnapshot,
        detection: TrackDetection,
        target_spec: TargetSpec,
        position: tuple[float, float, float] | None,
        track: ShortTrackEvidence,
    ) -> tuple[bool, str | None]:
        snapshot = target_manager.snapshot()
        if snapshot.lifecycle in {TargetLifecycle.LOCKED, TargetLifecycle.TRACKING}:
            if (
                snapshot.target_id is not None
                and self._locked_tracker_id == detection.track_id
            ):
                return True, snapshot.target_id
            return False, None
        if snapshot.lifecycle not in {
            TargetLifecycle.SEARCHING,
            TargetLifecycle.CANDIDATE,
            TargetLifecycle.REACQUIRING,
        }:
            return False, None
        if snapshot.lifecycle is TargetLifecycle.REACQUIRING:
            self._expected_reacquire_target_id = snapshot.target_id
        if snapshot.lifecycle in {TargetLifecycle.SEARCHING, TargetLifecycle.REACQUIRING}:
            # A secondary track may have accumulated history while another
            # candidate owned TargetManager.  Its activation cannot be
            # backdated across the preceding rejection transition, and old
            # observations must not count toward the newly active evidence
            # window.
            registration_timestamp = track.timestamp_s
            if candidate.first_seen_timestamp_s < registration_timestamp:
                self._track_evidence.reset(candidate.candidate_id)
                restarted = self._track_evidence.update(
                    candidate_id=candidate.candidate_id,
                    timestamp_s=registration_timestamp,
                    detection=detection,
                )
                self._track_snapshots[candidate.candidate_id] = [restarted]
            detection_candidate = DetectionCandidate(
                candidate_id=candidate.candidate_id,
                timestamp_s=registration_timestamp,
                confidence=detection.confidence,
                source=self._source_name,
                estimated_position=position,
            )
            self._confirmation.register_candidate(detection_candidate, target_manager)
            self.metrics.candidates_total += 1
            return False, None
        active_id = target_manager.snapshot().target_id
        if active_id != candidate.candidate_id:
            return False, None

        # The concrete builder deliberately marks a history unstable until
        # its minimum count and duration are both present.  That is a pending
        # candidate, not negative identity evidence.  Once those minima are
        # available, ``stable=False`` represents a genuine continuity/jump
        # failure and is passed to the confirmation coordinator for rejection.
        if (
            track.observation_count < self._config.tracker.min_track_observations
            or track.duration_s < self._config.tracker.min_track_duration_s
        ):
            return False, None

        semantic = self._semantic_evidence(
            candidate,
            target_spec,
            detection,
            track.timestamp_s,
        )
        if semantic is None:
            semantic = SemanticVerification(
                candidate_id=candidate.candidate_id,
                timestamp_s=track.timestamp_s,
                target_description=target_spec.description,
                matches=True,
                confidence=0.0,
                verifier="semantic_review_pending",
            )
        confirmation_track = track
        if semantic.timestamp_s < track.timestamp_s:
            compatible = [
                value
                for value in self._track_snapshots.get(candidate.candidate_id, ())
                if value.timestamp_s <= semantic.timestamp_s
            ]
            if not compatible:
                return False, None
            confirmation_track = compatible[-1]
            if (
                confirmation_track.observation_count
                < self._config.tracker.min_track_observations
                or confirmation_track.duration_s
                < self._config.tracker.min_track_duration_s
            ):
                return False, None
        logical_target_id = (
            self._expected_reacquire_target_id
            or _logical_target_id(target_spec)
        )
        is_new_reacquire_track = (
            self._expected_reacquire_target_id is not None
            and self._locked_tracker_id is not None
            and self._locked_tracker_id != detection.track_id
        )
        identity = self._identity_evidence(
            candidate,
            logical_target_id,
            detection.track_id,
            is_new_reacquire_track,
            max(confirmation_track.timestamp_s, semantic.timestamp_s),
            confirmation_track,
        )
        if identity is None:
            identity = IdentityConsistencyEvidence(
                candidate_id=candidate.candidate_id,
                target_id=logical_target_id,
                timestamp_s=max(
                    confirmation_track.timestamp_s,
                    semantic.timestamp_s,
                ),
                reidentified=True,
                temporally_consistent=True,
                consistent_observations=min(
                    confirmation_track.observation_count,
                    self._config.tracker.min_track_observations,
                ),
                confidence=(
                    0.0 if is_new_reacquire_track else confirmation_track.confidence
                ),
                source=(
                    "qwen_reacquire_pending"
                    if is_new_reacquire_track
                    else "temporal_track"
                ),
            )
        result = self._confirmation.evaluate(
            target_manager=target_manager,
            track=confirmation_track,
            semantic=semantic,
            identity=identity,
        )
        if result.decision is ConfirmationDecision.REJECTED:
            self.metrics.candidates_rejected += 1
            bank_candidate = self._require_candidate_bank().get(candidate.candidate_id)
            if bank_candidate is not None and bank_candidate.lifecycle not in {
                CandidateLifecycle.REJECTED,
                CandidateLifecycle.STALE,
            }:
                self._require_candidate_bank().reject(
                    candidate.candidate_id,
                    timestamp_s=confirmation_track.timestamp_s,
                )
            return False, None
        if result.decision is ConfirmationDecision.CONFIRMED:
            self.metrics.candidates_confirmed += 1
            if self._expected_reacquire_target_id is not None:
                self.metrics.reacquire_successes += 1
            self._locked_tracker_id = detection.track_id
            self._expected_reacquire_target_id = None
            self._retain_identity_reference(candidate)
            current = self._require_candidate_bank().get(candidate.candidate_id)
            if current is not None and current.lifecycle in {
                CandidateLifecycle.PROVISIONAL,
                CandidateLifecycle.UNDER_INSPECTION,
            }:
                self._require_candidate_bank().verify(candidate.candidate_id)
            return True, result.target_id
        return False, None

    def _semantic_evidence(
        self,
        candidate: CandidateSnapshot,
        target_spec: TargetSpec,
        detection: TrackDetection,
        timestamp_s: float,
    ) -> SemanticVerification | None:
        if self._semantic_provider is not None:
            supplied = self._semantic_provider(
                candidate, target_spec, detection, timestamp_s
            )
            if supplied is not None and (
                self._config.confirmation.mode != "qwen_required"
                or supplied.verifier == "qwen_vl"
            ):
                return supplied
        review = self._latest_visual_review(candidate)
        if review is not None:
            try:
                return self._qwen_semantic_adapter.from_review(
                    candidate_id=candidate.candidate_id,
                    target_spec=target_spec,
                    review=review,
                    expected_bbox=_candidate_bbox_at(
                        candidate,
                        review.observation_timestamp_s,
                    ),
                )
            except (QwenEvidencePending, VisualEvidenceError, ValueError):
                # A stale/misaligned typed review remains pending.  It is never
                # converted into class-only approval or an Oracle fallback.
                pass
        if self._config.confirmation.mode == "qwen_required":
            return None
        if self._config.detector.model_family == "yolo":
            try:
                return ClosedSetClassSemanticVerifier(
                    self._alias_mapper,
                    self._model_names,
                ).verify(
                    candidate_id=candidate.candidate_id,
                    timestamp_s=timestamp_s,
                    target_spec=target_spec,
                    detection=detection,
                )
            except SemanticVerificationRequiresQwen:
                return None

        # YOLOE supplies open-vocabulary proposals, but a proposal alone is
        # not proof of required attributes, exclusions, relations, or a
        # specific identity.  Those cases remain pending until the injected
        # Qwen verifier returns typed evidence.
        requires_qwen = bool(
            target_spec.hard_attributes
            or target_spec.negative_constraints
            or target_spec.relation_constraints
            or target_spec.original_description.casefold()
            != target_spec.category.casefold()
            or target_spec.immutable_identity_summary.casefold()
            != target_spec.category.casefold()
        )
        if requires_qwen and self._config.confirmation.require_qwen_for_attributes:
            return None
        return SemanticVerification(
            candidate_id=candidate.candidate_id,
            timestamp_s=timestamp_s,
            target_description=target_spec.description,
            matches=True,
            confidence=detection.confidence,
            verifier="yoloe_text_grounding",
        )

    def _identity_evidence(
        self,
        candidate: CandidateSnapshot,
        target_id: str,
        tracker_id: int,
        new_reacquire_track: bool,
        timestamp_s: float,
        track: ShortTrackEvidence,
    ) -> IdentityConsistencyEvidence | None:
        if self._identity_provider is not None:
            supplied = self._identity_provider(
                candidate,
                target_id,
                tracker_id,
                new_reacquire_track,
                timestamp_s,
            )
            if supplied is not None:
                return supplied
        if (
            new_reacquire_track
            and self._config.confirmation.require_qwen_for_reacquire_new_track_id
        ):
            review = self._latest_visual_review(candidate)
            reference_provider = self._visual_review_reference_provider
            references = (
                ()
                if reference_provider is None
                else tuple(reference_provider(candidate.candidate_id))
            )
            if review is None or not references:
                return None
            try:
                return self._qwen_identity_adapter.from_review(
                    track=track,
                    target_id=target_id,
                    review=review,
                    reference_handles=references,
                    expected_bbox=_candidate_bbox_at(
                        candidate,
                        review.observation_timestamp_s,
                    ),
                )
            except (QwenEvidencePending, VisualEvidenceError, ValueError, TypeError):
                return None
        reference_track_id = (
            tracker_id if self._locked_tracker_id is None else self._locked_tracker_id
        )
        try:
            return self._identity_verifier.verify(
                track=track,
                target_id=target_id,
                reference_track_id=reference_track_id,
                current_track_id=tracker_id,
                reacquiring=self._expected_reacquire_target_id is not None,
                timestamp_s=timestamp_s,
            )
        except ReacquireIdentityRequiresQwen:
            return None

    def _latest_visual_review(
        self,
        candidate: CandidateSnapshot,
    ) -> QwenVisualReview | None:
        provider = self._visual_review_provider
        if provider is None:
            return None
        review = provider(candidate.candidate_id)
        if review is not None and not isinstance(review, QwenVisualReview):
            raise TypeError("visual_review_provider must return QwenVisualReview or None")
        if (
            review is not None
            and review.observation_timestamp_s
            < candidate.first_seen_timestamp_s
        ):
            # A candidate ID may reappear only after its rejection cooldown.
            # Typed Qwen evidence from the prior epoch cannot confirm the new
            # track, even if an external bounded provider still retains it.
            return None
        return review

    def _retain_identity_reference(self, candidate: CandidateSnapshot) -> None:
        for retained in self._identity_reference_refs:
            if self._frame_store.contains(retained):
                try:
                    self._frame_store.unpin(retained)
                except ValueError:
                    pass
        self._identity_reference_refs = ()
        for frame_ref in reversed(candidate.frame_history):
            if not self._frame_store.contains(frame_ref):
                continue
            self._frame_store.pin(frame_ref)
            self._identity_reference_refs = (frame_ref,)
            return

    def _resolve_position(
        self,
        candidate: CandidateSnapshot,
        timestamp_s: float,
        *,
        now_s: float,
    ) -> tuple[float, float, float] | None:
        if now_s - timestamp_s > self._config.geometry.max_measurement_age_s:
            self.metrics.depth_resolution_failures += 1
            return None
        try:
            return self._resolver.resolve(
                candidate,
                timestamp_s=timestamp_s,
            ).position_xyz_m
        except CandidateResolutionUnavailable:
            self.metrics.depth_resolution_failures += 1
            return None

    def _estimate_for_now(
        self,
        now_s: float,
        target_manager: TargetManager,
    ) -> TargetEstimate | None:
        snapshot = target_manager.snapshot()
        latest = self._latest_estimate
        latest_is_fresh = (
            latest is not None
            and now_s - latest.timestamp_s
            <= self._config.yolo_service.max_result_age_s
        )
        if latest_is_fresh:
            assert latest is not None
            # SEARCH/CANDIDATE consumes provisional estimates to expose
            # CANDIDATE_PENDING.  After lock, only the matching confirmed
            # identity can replace the control observation.
            if snapshot.lifecycle in {
                TargetLifecycle.SEARCHING,
                TargetLifecycle.CANDIDATE,
            } or (
                latest.confirmed
                and snapshot.target_id is not None
                and latest.target_id == snapshot.target_id
            ):
                return latest
        if snapshot.lifecycle not in {
            TargetLifecycle.LOCKED,
            TargetLifecycle.TRACKING,
            TargetLifecycle.LOST,
            TargetLifecycle.REACQUIRING,
        } or snapshot.target_id is None:
            return None
        try:
            predicted = self._estimator.predict(now_s)
        except ValueError:
            return None
        if predicted is None:
            return None
        self.metrics.record_measurement_age(predicted.measurement_age_s)
        reference = self._latest_confirmed_estimate
        return TargetEstimate(
            timestamp_s=now_s,
            target_id=snapshot.target_id,
            candidate_id=None if reference is None else reference.candidate_id,
            tracker_id=None if reference is None else reference.tracker_id,
            visible=False,
            confirmed=True,
            predicted_only=True,
            class_id=None if reference is None else reference.class_id,
            class_name=None if reference is None else reference.class_name,
            confidence=None if reference is None else reference.confidence,
            bbox_xyxy_normalized=None,
            position_world_m=predicted.position_world_m,
            velocity_world_mps=predicted.velocity_world_mps,
            measurement_age_s=predicted.measurement_age_s,
            source="kalman_prediction",
        )

    def _record_lifecycle(
        self,
        lifecycle: TargetLifecycle,
        *,
        timestamp_s: float,
    ) -> None:
        previous = self._last_target_lifecycle
        if lifecycle is previous:
            return
        if lifecycle is TargetLifecycle.LOST:
            self.metrics.target_lost_count += 1
            if previous in {TargetLifecycle.LOCKED, TargetLifecycle.TRACKING}:
                self.metrics.track_fragmentations += 1
            reference = self._latest_confirmed_estimate
            frame_ref = self._latest_confirmed_frame_ref
            if frame_ref is not None and not self._frame_store.contains(frame_ref):
                frame_ref = None
            if frame_ref is not None:
                self._capture_debug_image(
                    "target_lost",
                    frame_ref=frame_ref,
                    detection=None,
                    candidate_id=(
                        None if reference is None else reference.candidate_id
                    ),
                    estimate=reference,
                    measurement_age_s=(
                        None
                        if reference is None
                        else max(
                            reference.measurement_age_s,
                            timestamp_s - reference.timestamp_s,
                        )
                    ),
                )
        if lifecycle is TargetLifecycle.REACQUIRING:
            self.metrics.reacquire_attempts += 1
        self._last_target_lifecycle = lifecycle

    def _capture_debug_image(
        self,
        event: str,
        *,
        frame_ref: FrameRef,
        detection: TrackDetection | None,
        candidate_id: str | None,
        estimate: TargetEstimate | None,
        measurement_age_s: float | None = None,
    ) -> None:
        writer = self._debug_image_writer
        if writer is None:
            return
        resolved_bbox = (
            detection.bbox_xyxy_normalized
            if detection is not None
            else (None if estimate is None else estimate.bbox_xyxy_normalized)
        )
        resolved_class_id = (
            detection.class_id
            if detection is not None
            else (None if estimate is None else estimate.class_id)
        )
        resolved_class_name = (
            detection.class_name
            if detection is not None
            else (None if estimate is None else estimate.class_name)
        )
        resolved_confidence = (
            detection.confidence
            if detection is not None
            else (None if estimate is None else estimate.confidence)
        )
        resolved_track_id = (
            f"track_{detection.track_id}"
            if detection is not None
            else (None if estimate is None else estimate.tracker_id)
        )
        try:
            writer.capture(
                event=event,
                frame_store=self._frame_store,
                frame_ref=frame_ref,
                annotation=TargetDebugAnnotation(
                    bbox_xyxy_normalized=resolved_bbox,
                    class_id=resolved_class_id,
                    class_name=resolved_class_name,
                    confidence=resolved_confidence,
                    track_id=resolved_track_id,
                    candidate_id=(
                        candidate_id
                        if candidate_id is not None
                        else (None if estimate is None else estimate.candidate_id)
                    ),
                    confirmed=False if estimate is None else estimate.confirmed,
                    position_world_m=(
                        None if estimate is None else estimate.position_world_m
                    ),
                    measurement_age_s=(
                        measurement_age_s
                        if measurement_age_s is not None
                        else (
                            None
                            if estimate is None
                            else estimate.measurement_age_s
                        )
                    ),
                    source=self._source_name,
                ),
            )
        except Exception:
            # Debug persistence is observational and must not alter control.
            return

    @property
    def _source_name(self) -> str:
        return (
            "yoloe26_botsort"
            if self._config.detector.model_family == "yoloe"
            else "yolo26_botsort"
        )

    def _compile_query(self, target_spec: TargetSpec) -> TargetQuery:
        try:
            if self._query_compiler is not None:
                query = self._query_compiler(target_spec, self._model_names)
                if not isinstance(query, TargetQuery):
                    raise TypeError("query_compiler must return TargetQuery")
                return query
            return compile_target_query(
                target_spec,
                self._config.detector.model_family,
                self._model_names,
                self._alias_mapper,
            )
        except UnsupportedTargetCategory as exc:
            raise TargetQueryUnsupported(
                str(exc)
            ) from exc

    def _cleanup_wait_s(self) -> float:
        return min(
            5.0,
            max(0.5, self._config.yolo_service.request_timeout_s + 0.25),
        )

    @staticmethod
    def _drain_future(inflight: _Inflight | None, *, timeout_s: float) -> None:
        if inflight is None or inflight.future.cancelled():
            return
        try:
            inflight.future.result(timeout=timeout_s)
        except Exception:
            # Cancellation, request failure and timeout are all followed by a
            # stream reset.  The caller decides whether reset failure is fatal.
            pass

    def _reset_stream_with_retry(
        self,
        *,
        mission_id: str,
        uav_id: str,
        stream_id: str,
        timeout_s: float,
    ) -> None:
        deadline = monotonic() + timeout_s
        while True:
            try:
                self._client.reset_stream(
                    ResetStreamRequest(
                        schema_version=1,
                        request_id=generate_routing_id("request"),
                        mission_id=mission_id,
                        uav_id=uav_id,
                        stream_id=stream_id,
                    )
                )
                return
            except YoloClientResponseError as exc:
                message = str(exc).casefold()
                retryable = any(
                    marker in message for marker in ("stream_busy", "busy", "409")
                )
                if not retryable or monotonic() >= deadline:
                    raise
                sleep(min(0.05, max(0.0, deadline - monotonic())))

    def _require_candidate_bank(self) -> CandidateBank:
        if self._candidate_bank is None:
            raise TargetPerceptionNotReady("coordinator has not been reset")
        return self._candidate_bank

    def _ensure_open(self) -> None:
        if self._closed:
            raise TargetPerceptionError("coordinator is closed")

    def _ensure_ready(self) -> None:
        self._ensure_open()
        if self._fatal_error is not None:
            raise TargetPerceptionError(
                "coordinator is failed closed; reset is required"
            )
        if self._mission_id is None or self._uav_id is None:
            raise TargetPerceptionNotReady(
                "reset(mission_id=..., uav_id=...) is required first"
            )


def _logical_target_id(target_spec: TargetSpec) -> str:
    digest = sha256(
        target_spec.immutable_identity_summary.encode("utf-8")
    ).hexdigest()[:24]
    return f"target_{digest}"


def _candidate_bbox_at(
    candidate: CandidateSnapshot,
    timestamp_s: float,
) -> tuple[float, float, float, float]:
    matches = [
        bbox
        for frame, bbox in zip(candidate.frame_history, candidate.bbox_history)
        if frame.timestamp_s <= timestamp_s + 1e-9
    ]
    if not matches:
        raise ValueError("visual review predates retained candidate frame history")
    return matches[-1]


def _normalize_model_names(
    values: Mapping[int, str] | Sequence[str] | None,
) -> dict[int, str]:
    if values is None:
        return {}
    items = values.items() if isinstance(values, Mapping) else enumerate(values)
    result: dict[int, str] = {}
    for raw_id, raw_name in items:
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 0:
            raise ValueError("model class IDs must be non-negative integers")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("model class names must be non-empty strings")
        result[int(raw_id)] = raw_name.strip()
    return result


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(fmean(values))


def _rmse_or_none(squared_errors: Sequence[float]) -> float | None:
    return None if not squared_errors else float(fmean(squared_errors) ** 0.5)


def _safe_error(prefix: str, exc: Exception) -> str:
    return f"{prefix}:{type(exc).__name__}"[:256]


__all__ = [
    "TargetPerceptionCoordinator",
    "TargetPerceptionError",
    "TargetPerceptionMetrics",
    "TargetPerceptionNotReady",
    "TargetQueryUnsupported",
]
