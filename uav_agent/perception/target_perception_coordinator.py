"""Bounded asynchronous target-perception pipeline.

The coordinator owns model I/O, candidate evidence, trusted RGB-D geometry and
world-state filtering outside MissionAgent and Skill code.  It keeps at most
one request in flight and one newest pending frame per UAV.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from statistics import fmean
from threading import RLock
from time import monotonic, sleep

from common.ids import (
    generate_routing_id,
    validate_mission_id,
    validate_routing_id,
    validate_uav_id,
)
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
from perception.grounding import (
    CandidateResolutionUnavailable,
    CandidateResolver,
    GroundingProposal,
    UltralyticsGrounder,
)
from perception.measurement import TargetMeasurement
from perception.target_state_estimator import (
    TargetStateEstimator,
    TargetStateMeasurementRejected,
)
from perception.target_debug_images import (
    BoundedTargetDebugImageWriter,
    TargetDebugAnnotation,
)
from perception.target_query import TargetQuerySpec
from perception.types import (
    DetectionCandidate,
    IdentityConsistencyEvidence,
    SemanticVerification,
    ShortTrackEvidence,
)
from perception.yolo_client import (
    YoloClientRequestTimeout,
    YoloClientResponseError,
    YoloClientStreamBusy,
    YoloClientUnavailable,
    YoloServiceClient,
    validate_yolo_model_identity,
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
from perception.semantic_fusion import AttributeSemanticVerificationPending
from perception.semantic_fusion import DETERMINISTIC_ATTRIBUTE_VERIFIER
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
_MAX_CANDIDATE_TRANSITION_LOGS_PER_ASSIGNMENT = 256
_MAX_DEPTH_RESOLUTION_FAILURE_LOGS_PER_ASSIGNMENT = 32
_MAX_DEPTH_RESOLUTION_FAILURE_REASON_KEYS = 16


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
    camera_frames_received: int = 0
    yolo_requests_submitted: int = 0
    yolo_results_received: int = 0
    yolo_requests: int = 0
    yolo_successful_responses: int = 0
    yolo_timeouts: int = 0
    yolo_response_errors: int = 0
    yolo_stream_busy_responses: int = 0
    yolo_stream_recoveries: int = 0
    yolo_stream_recovery_failures: int = 0
    yolo_stale_results: int = 0
    yolo_dropped_frames: int = 0
    detections_total: int = 0
    tracked_detections_total: int = 0
    candidate_created: int = 0
    candidate_confirmed: int = 0
    candidate_rejected: int = 0
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
    depth_resolution_attempts: int = 0
    depth_resolution_successes: int = 0
    depth_resolution_last_failure_reason: str | None = None
    depth_resolution_failure_reason_counts: dict[str, int] = field(
        default_factory=dict
    )
    measurement_created: int = 0
    measurement_rejected: int = 0
    kalman_updates_accepted: int = 0
    kalman_updates_rejected: int = 0
    position_world_outputs: int = 0
    predicted_only_outputs: int = 0
    search_target_found: int = 0
    track_visible_updates: int = 0
    track_predicted_updates: int = 0
    qwen_attribute_fallback_count: int = 0
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

    def to_dict(self) -> dict[str, object]:
        latency_sorted = sorted(self._latencies_ms)
        p95 = (
            None
            if not latency_sorted
            else latency_sorted[min(len(latency_sorted) - 1, int(0.95 * len(latency_sorted)))]
        )
        return {
            "camera_frames_received": self.camera_frames_received,
            "yolo_requests_submitted": self.yolo_requests_submitted,
            "yolo_results_received": self.yolo_results_received,
            "yolo_requests": self.yolo_requests,
            "yolo_successful_responses": self.yolo_successful_responses,
            "yolo_timeouts": self.yolo_timeouts,
            "yolo_response_errors": self.yolo_response_errors,
            "yolo_stream_busy_responses": self.yolo_stream_busy_responses,
            "yolo_stream_recoveries": self.yolo_stream_recoveries,
            "yolo_stream_recovery_failures": self.yolo_stream_recovery_failures,
            "yolo_stale_results": self.yolo_stale_results,
            "yolo_dropped_frames": self.yolo_dropped_frames,
            "yolo_inference_latency_ms_mean": _mean_or_none(self._latencies_ms),
            "yolo_inference_latency_ms_p95": p95,
            "detections_total": self.detections_total,
            "tracked_detections_total": self.tracked_detections_total,
            "candidate_created": self.candidate_created,
            "candidate_confirmed": self.candidate_confirmed,
            "candidate_rejected": self.candidate_rejected,
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
            "depth_resolution_attempts": self.depth_resolution_attempts,
            "depth_resolution_successes": self.depth_resolution_successes,
            "depth_resolution_last_failure_reason": (
                self.depth_resolution_last_failure_reason
            ),
            "depth_resolution_failure_reason_counts": dict(
                sorted(self.depth_resolution_failure_reason_counts.items())
            ),
            "measurement_created": self.measurement_created,
            "measurement_rejected": self.measurement_rejected,
            "kalman_updates_accepted": self.kalman_updates_accepted,
            "kalman_updates_rejected": self.kalman_updates_rejected,
            "position_world_outputs": self.position_world_outputs,
            "predicted_only_outputs": self.predicted_only_outputs,
            "search_target_found": self.search_target_found,
            "track_visible_updates": self.track_visible_updates,
            "track_predicted_updates": self.track_predicted_updates,
            "qwen_attribute_fallback_count": self.qwen_attribute_fallback_count,
            "target_visible_frames": self.target_visible_frames,
            "target_total_frames": self.target_total_frames,
            "yolo_latency_sample_count": len(self._latencies_ms),
            "position_measurement_age_count": len(self._measurement_ages_s),
            "position_error_sample_count": len(self._position_squared_errors),
            "velocity_error_sample_count": len(self._velocity_squared_errors),
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
        resolver: CandidateResolver | None = None,
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
        if frame_store is not None and not isinstance(frame_store, FrameStore):
            raise TypeError("frame_store must be a FrameStore or None")
        # FrameStore implements __len__, so a newly created, intentionally
        # shared store is falsey while empty.  An ``or FrameStore()`` fallback
        # would silently detach the RGB-D semantic provider from the
        # coordinator on every fresh runtime.
        self._frame_store = FrameStore() if frame_store is None else frame_store
        if candidate_bank is not None and not isinstance(candidate_bank, CandidateBank):
            raise TypeError("candidate_bank must be a CandidateBank or None")
        self._provided_candidate_bank = candidate_bank
        if resolver is not None:
            self._resolver = resolver
        else:
            # Local import avoids a module cycle: the factory imports this
            # coordinator only inside the production runtime builder.
            from perception.factory import build_target_candidate_resolver

            self._resolver = build_target_candidate_resolver(
                config,
                frame_store=self._frame_store,
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
        self._assignment_id: str | None = None
        self._target_alias: str | None = None
        self._target_query: TargetQuerySpec | None = None
        self._stream_id: str | None = None
        self._candidate_bank: CandidateBank | None = None
        self._frame_sequence = 0
        self._inflight: _Inflight | None = None
        # A client timeout only ends the local HTTP wait; the worker's
        # model.track() thread can still own its process-wide stream.  Recovery
        # is therefore a first-class asynchronous barrier.  No later Track
        # request may launch until its reset future has completed successfully.
        self._stream_recovery: Future[None] | None = None
        self._pending: _Submission | None = None
        self._latest_estimate: TargetEstimate | None = None
        # Prediction used for control may inherit state only from a confirmed
        # logical identity.  The newest detector result can instead be an
        # unconfirmed same-class distractor.
        self._latest_confirmed_estimate: TargetEstimate | None = None
        self._latest_confirmed_frame_ref: FrameRef | None = None
        self._track_snapshots: dict[str, list[ShortTrackEvidence]] = {}
        self._tracker_id_by_candidate: dict[str, int] = {}
        self._locked_tracker_id: int | None = None
        self._estimator_tracker_id: int | None = None
        self._last_observed_tracker_id: int | None = None
        self._last_target_lifecycle: TargetLifecycle | None = None
        self._expected_reacquire_target_id: str | None = None
        self._identity_reference_refs: tuple[FrameRef, ...] = ()
        self._last_error: str | None = None
        self._fatal_error: str | None = None
        self._consecutive_availability_failures = 0
        self._candidate_transition_logs_emitted = 0
        self._candidate_transition_records: deque[dict[str, object]] = deque(
            maxlen=_MAX_CANDIDATE_TRANSITION_LOGS_PER_ASSIGNMENT
        )
        self._depth_resolution_failure_logs_emitted = 0
        self._depth_resolution_failure_log_counts: dict[str, int] = {}
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

    @property
    def candidate_bank(self) -> CandidateBank | None:
        return self._candidate_bank or self._provided_candidate_bank

    def qwen_fallback_required(self, candidate_id: str) -> bool:
        candidate = validate_routing_id(candidate_id, "candidate_id")
        required = getattr(self._semantic_provider, "requires_qwen", None)
        if callable(required) and bool(required(candidate)):
            return True
        # Reacquiring under a new BoT-SORT ID is an identity question even
        # when class and colour are deterministic.  Expose that condition to
        # the shared visual scheduler so it can request one low-frequency,
        # candidate-bound review with the retained verified reference.
        with self._lock:
            tracker_id = self._tracker_id_by_candidate.get(candidate)
            return bool(
                self._config.confirmation.require_qwen_for_reacquire_new_track_id
                and self._expected_reacquire_target_id is not None
                and self._locked_tracker_id is not None
                and tracker_id is not None
                and tracker_id != self._locked_tracker_id
            )

    def runtime_metrics(self) -> Mapping[str, object]:
        """Return bounded detector and semantic-provider scalar diagnostics."""

        result: dict[str, object] = dict(self.metrics.to_dict())
        provider_metrics = getattr(self._semantic_provider, "metrics", None)
        if provider_metrics is not None:
            extra_value = (
                provider_metrics()
                if callable(provider_metrics)
                else provider_metrics
            )
            to_dict = getattr(extra_value, "to_dict", None)
            extra = to_dict() if callable(to_dict) else extra_value
            if not isinstance(extra, Mapping):
                raise TypeError("semantic provider metrics must be a mapping")
            extra = dict(extra)
            aliases = {
                "color_observations": "observations_total",
                "color_matches": "evidence_match",
                "color_mismatches": "evidence_mismatch",
                "color_pending": "evidence_pending",
                "qwen_attribute_fallback_required": "qwen_fallback_required",
            }
            for public_name, internal_name in aliases.items():
                if internal_name in extra:
                    extra[public_name] = extra[internal_name]
            extra["attribute_confirmed"] = int(extra.get("semantic_match", 0))
            extra["attribute_ambiguous"] = int(extra.get("semantic_pending", 0))
            overlap = set(result).intersection(extra)
            if overlap:
                raise ValueError(
                    "semantic provider metrics collide with coordinator metrics: "
                    + ", ".join(sorted(overlap))
                )
            result.update(extra)
        result.setdefault("attribute_confirmed", 0)
        result.setdefault("attribute_ambiguous", 0)
        resolver_statistics = getattr(self._resolver, "statistics", None)
        if resolver_statistics is not None:
            resolver_value = (
                resolver_statistics()
                if callable(resolver_statistics)
                else resolver_statistics
            )
            resolver_to_dict = getattr(resolver_value, "to_dict", None)
            resolver_metrics = (
                resolver_to_dict() if callable(resolver_to_dict) else resolver_value
            )
            if not isinstance(resolver_metrics, Mapping):
                raise TypeError("resolver statistics must be a mapping")
            overlap = set(result).intersection(resolver_metrics)
            if overlap:
                raise ValueError(
                    "resolver metrics collide with coordinator metrics: "
                    + ", ".join(sorted(overlap))
                )
            result.update(dict(resolver_metrics))
        return result

    def attribute_evidence_records(self) -> tuple[object, ...]:
        """Return the provider's bounded scalar evidence snapshot, if any."""

        records = getattr(self._semantic_provider, "evidence_records", None)
        if not callable(records):
            return ()
        values = tuple(records())
        if len(values) > 10_000:
            raise ValueError("semantic provider returned unbounded evidence records")
        return values

    def drain_attribute_evidence_records(self) -> tuple[object, ...]:
        """Atomically consume the provider's bounded scalar evidence records."""

        drain = getattr(self._semantic_provider, "drain_evidence_records", None)
        if not callable(drain):
            return ()
        values = tuple(drain())
        if len(values) > 10_000:
            raise ValueError("semantic provider returned unbounded evidence records")
        return values

    def candidate_transition_records(self) -> tuple[Mapping[str, object], ...]:
        """Return a copy of the bounded scalar candidate-transition buffer."""

        with self._lock:
            return tuple(dict(record) for record in self._candidate_transition_records)

    def drain_candidate_transition_records(
        self,
    ) -> tuple[Mapping[str, object], ...]:
        """Atomically consume candidate lifecycle edges for experiment output."""

        with self._lock:
            values = tuple(
                dict(record) for record in self._candidate_transition_records
            )
            self._candidate_transition_records.clear()
            return values

    def reset(
        self,
        *,
        mission_id: str,
        uav_id: str,
        assignment_id: str | None = None,
        target_alias: str | None = None,
        target_query: TargetQuerySpec | None = None,
    ) -> None:
        mission = validate_mission_id(mission_id)
        uav = validate_uav_id(uav_id)
        assignment = (
            None
            if assignment_id is None
            else validate_routing_id(assignment_id, "assignment_id")
        )
        routed_target = (
            None
            if target_alias is None
            else validate_routing_id(target_alias, "target_alias")
        )
        if target_query is not None and not isinstance(
            target_query,
            TargetQuerySpec,
        ):
            raise TypeError("target_query must be a TargetQuerySpec or None")
        if target_query is not None:
            if routed_target is None:
                routed_target = target_query.target_alias
            elif routed_target != target_query.target_alias:
                raise ValueError("target_alias must match target_query.target_alias")
        if routed_target is not None and assignment is None:
            raise ValueError("target_alias requires assignment_id")
        with self._lock:
            self._ensure_open()
            previous_mission = self._mission_id
            previous_uav = self._uav_id
            previous_stream = self._stream_id
            inflight = self._inflight
            self._inflight = None
            if inflight is not None:
                inflight.future.cancel()
            recovery = self._stream_recovery
            self._stream_recovery = None
            if recovery is not None:
                recovery.cancel()
            self._pending = None
            # A failed handshake must not leave a coordinator that appears
            # ready.  Publish routing state only after health/reset/model-info
            # have all succeeded.
            self._mission_id = None
            self._uav_id = None
            self._assignment_id = None
            self._target_alias = None
            self._target_query = None
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
        self._drain_executor_future(recovery, timeout_s=cleanup_wait_s)
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
            detector = self._config.detector
            validate_yolo_model_identity(
                info,
                expected_model_family=(
                    detector.expected_model_family or detector.model_family
                ),
                expected_model_names=detector.expected_model_names,
                expected_model_sha256=detector.expected_model_sha256,
                worker_url=self._config.yolo_service.url,
            )
            model_names = _normalize_model_names(info.names)
            if target_query is not None:
                self._validate_target_query_model(target_query, model_names)
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
            self._assignment_id = assignment
            self._target_alias = routed_target
            self._target_query = target_query
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
            self._tracker_id_by_candidate.clear()
            self._locked_tracker_id = None
            self._estimator_tracker_id = None
            self._last_observed_tracker_id = None
            self._last_target_lifecycle = None
            self._expected_reacquire_target_id = None
            self._identity_reference_refs = ()
            self._last_error = None
            self._fatal_error = None
            self._consecutive_availability_failures = 0
            self._candidate_transition_logs_emitted = 0
            self._depth_resolution_failure_logs_emitted = 0
            self._depth_resolution_failure_log_counts.clear()
            self._estimator.reset()
            resolver_reset = getattr(self._resolver, "reset", None)
            if callable(resolver_reset):
                resolver_reset(uav_id=uav, assignment_id=assignment)
            semantic_reset = getattr(self._semantic_provider, "reset", None)
            if assignment is not None and callable(semantic_reset):
                semantic_reset(
                    mission_id=mission,
                    uav_id=uav,
                    assignment_id=assignment,
                )

    def submit_frame(
        self,
        *,
        camera_sample: CameraSample,
        target_spec: TargetSpec | None = None,
        uav_linear_velocity_world_mps: Sequence[float] | None = None,
        uav_angular_velocity_body_radps: Sequence[float] | None = None,
    ) -> None:
        if not isinstance(camera_sample, CameraSample):
            raise TypeError("camera_sample must be a CameraSample")
        if self._config.geometry.mode == "temporal_ray_depth" and (
            uav_linear_velocity_world_mps is None
            or uav_angular_velocity_body_radps is None
        ):
            raise ValueError(
                "temporal_ray_depth requires synchronized UAV linear and "
                "angular self-motion"
            )
        with self._lock:
            self._ensure_ready()
            self.metrics.camera_frames_received += 1
            assert self._mission_id is not None
            assert self._uav_id is not None
            assert self._stream_id is not None
            self._frame_sequence += 1
            frame_id = f"frame_{self._frame_sequence:010d}"
            frame_ref = self._frame_store.add_sample(
                uav_id=self._uav_id,
                frame_id=frame_id,
                sample=camera_sample,
                uav_linear_velocity_world_mps=(
                    uav_linear_velocity_world_mps
                ),
                uav_angular_velocity_body_radps=(
                    uav_angular_velocity_body_radps
                ),
            )
            # Production reset binds a closed TargetQuerySpec.  In that mode
            # a complete TargetSpec is rejected even if a caller attempts to
            # bypass the public runtime bridge.  The optional TargetSpec path
            # remains only for isolated legacy coordinator tests which reset
            # without a production query.
            if self._target_query is not None:
                if target_spec is not None:
                    raise PermissionError(
                        "production coordinator rejects complete TargetSpec input"
                    )
                semantic_target_spec = self._target_query.to_semantic_target_spec()
                query = self._compile_target_query_spec(self._target_query)
            else:
                if not isinstance(target_spec, TargetSpec):
                    raise TypeError(
                        "legacy direct coordinator use requires target_spec; "
                        "production reset requires target_query"
                    )
                semantic_target_spec = target_spec
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
                semantic_target_spec,
            )
            if self._inflight is None and self._stream_recovery is None:
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
            recovery = self._stream_recovery
            if recovery is not None and recovery.done():
                self._stream_recovery = None
                try:
                    recovery.result()
                    self.metrics.yolo_stream_recoveries += 1
                except Exception as exc:
                    self.metrics.yolo_stream_recovery_failures += 1
                    self._pending = None
                    self._last_error = _safe_error(
                        "yolo_stream_recovery_failed",
                        exc,
                    )
                    self._fatal_error = self._last_error
                    raise TargetPerceptionError(
                        "YOLO stream ownership could not be recovered"
                    ) from exc
                if self._pending is not None:
                    pending, self._pending = self._pending, None
                    self._launch(pending)

            completed = self._inflight
            if completed is not None and completed.future.done():
                self._inflight = None
                try:
                    response = completed.future.result()
                    response.assert_matches(completed.submission.request)
                    self._consecutive_availability_failures = 0
                    self.metrics.yolo_successful_responses += 1
                    self.metrics.yolo_results_received += 1
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
                    stream_busy = isinstance(exc, YoloClientStreamBusy)
                    remote_completion_unknown = isinstance(
                        exc,
                        (TimeoutError, YoloClientRequestTimeout, YoloClientStreamBusy),
                    )
                    if stream_busy:
                        self.metrics.yolo_stream_busy_responses += 1
                        self._consecutive_availability_failures += 1
                    elif isinstance(exc, (TimeoutError, YoloClientUnavailable)):
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
                    if remote_completion_unknown:
                        self._start_stream_recovery()
                if self._pending is not None and self._stream_recovery is None:
                    pending, self._pending = self._pending, None
                    self._launch(pending)

            estimate = self._estimate_for_now(now, target_manager)
            self._latest_estimate = estimate
            self.metrics.target_total_frames += 1
            if estimate is not None and estimate.visible:
                self.metrics.target_visible_frames += 1
            if estimate is not None and estimate.position_world_m is not None:
                self.metrics.position_world_outputs += 1
            if estimate is not None and estimate.predicted_only:
                self.metrics.predicted_only_outputs += 1
            if target_manager.lifecycle is TargetLifecycle.TRACKING:
                if estimate is not None and estimate.predicted_only:
                    self.metrics.track_predicted_updates += 1
                elif estimate is not None and estimate.visible:
                    self.metrics.track_visible_updates += 1
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
            recovery = self._stream_recovery
            self._stream_recovery = None
            if recovery is not None:
                recovery.cancel()
            self._pending = None
            self._estimator.reset()
            self._track_evidence.reset()
            self._track_snapshots.clear()
            self._target_alias = None
            self._target_query = None
            self._assignment_id = None
            if uav is not None:
                self._frame_store.clear(uav_id=uav)
            self._closed = True
        # A running executor future cannot be cancelled.  Drain it for a
        # strictly bounded cleanup interval before asking the single-stream
        # service to destroy its persistent BoT-SORT state.
        cleanup_wait_s = self._cleanup_wait_s()
        self._drain_future(inflight, timeout_s=cleanup_wait_s)
        self._drain_executor_future(recovery, timeout_s=cleanup_wait_s)
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
        close_semantic = getattr(self._semantic_provider, "close", None)
        if callable(close_semantic):
            close_semantic()

    def _launch(self, submission: _Submission) -> None:
        if self._stream_recovery is not None:
            raise RuntimeError("cannot launch YOLO while stream recovery is active")
        rgb = self._frame_store.get_frame(submission.frame_ref)
        if rgb is None:
            self.metrics.yolo_dropped_frames += 1
            self._last_error = "frame_evicted_before_yolo_submit"
            return
        self.metrics.yolo_requests += 1
        self.metrics.yolo_requests_submitted += 1
        future = self._executor.submit(
            self._client.track,
            submission.request,
            rgb,
        )
        self._inflight = _Inflight(submission, future)

    def _start_stream_recovery(self) -> None:
        """Serialize remote stream retirement ahead of the newest frame.

        HTTP request cancellation cannot cancel an inference already running in
        the service thread pool.  The reset helper waits through exact
        ``STREAM_BUSY`` responses and only then releases the executor queue for
        another Track request.
        """

        if self._stream_recovery is not None:
            return
        if self._mission_id is None or self._uav_id is None or self._stream_id is None:
            raise TargetPerceptionNotReady("stream recovery requires an active binding")
        self._stream_recovery = self._executor.submit(
            self._recover_stream,
            self._mission_id,
            self._uav_id,
            self._stream_id,
            self._cleanup_wait_s(),
        )

    def _recover_stream(
        self,
        mission_id: str,
        uav_id: str,
        stream_id: str,
        timeout_s: float,
    ) -> None:
        self._reset_stream_with_retry(
            mission_id=mission_id,
            uav_id=uav_id,
            stream_id=stream_id,
            timeout_s=timeout_s,
        )

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
        self.metrics.tracked_detections_total += len(response.detections)
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
            confidence=detection.confidence,
            tracker_id=f"track_{detection.track_id}",
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
            semantic_reset_candidate = getattr(
                self._semantic_provider,
                "reset_candidate",
                None,
            )
            if callable(semantic_reset_candidate):
                semantic_reset_candidate(candidate.candidate_id)
        track = self._track_evidence.update(
            candidate_id=candidate.candidate_id,
            timestamp_s=timestamp_s,
            detection=detection,
        )
        snapshots = self._track_snapshots.setdefault(candidate.candidate_id, [])
        self._tracker_id_by_candidate[candidate.candidate_id] = detection.track_id
        snapshots.append(track)
        del snapshots[:-32]
        measurement = self._resolve_measurement(
            candidate,
            timestamp_s,
            now_s=now_s,
        )
        position = (
            None if measurement is None else measurement.position_world_m
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
        if confirmed and measurement is not None:
            try:
                if self._estimator_tracker_id != detection.track_id:
                    self._estimator.reset()
                    self._estimator_tracker_id = detection.track_id
                filtered = self._estimator.update(
                    replace(
                        measurement,
                        tracker_id=f"track_{detection.track_id}",
                    )
                )
                self.metrics.kalman_updates_accepted += 1
                self.metrics.record_measurement_age(filtered.measurement_age_s)
            except (TargetStateMeasurementRejected, ValueError):
                self.metrics.depth_resolution_failures += 1
                self.metrics.measurement_rejected += 1
                self.metrics.kalman_updates_rejected += 1

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
            self._log_candidate_transition(
                "candidate_created",
                timestamp_s=timestamp_s,
                detection=detection,
                candidate_id=candidate.candidate_id,
                measurement=measurement,
                estimate=estimate,
            )
            self._capture_debug_image(
                "first_candidate",
                frame_ref=submission.frame_ref,
                detection=detection,
                candidate_id=candidate.candidate_id,
                estimate=estimate,
                measurement=measurement,
            )
        if self.metrics.candidates_rejected > rejected_before:
            self._log_candidate_transition(
                "candidate_rejected",
                timestamp_s=timestamp_s,
                detection=detection,
                candidate_id=candidate.candidate_id,
                measurement=measurement,
                estimate=estimate,
            )
            self._capture_debug_image(
                "candidate_rejected",
                frame_ref=submission.frame_ref,
                detection=detection,
                candidate_id=candidate.candidate_id,
                estimate=estimate,
                measurement=measurement,
            )
        if self.metrics.candidates_confirmed > confirmed_before:
            self._log_candidate_transition(
                "candidate_confirmed",
                timestamp_s=timestamp_s,
                detection=detection,
                candidate_id=candidate.candidate_id,
                measurement=measurement,
                estimate=estimate,
            )
            self._capture_debug_image(
                "confirmation_success",
                frame_ref=submission.frame_ref,
                detection=detection,
                candidate_id=candidate.candidate_id,
                estimate=estimate,
                measurement=measurement,
            )
        if self.metrics.reacquire_successes > reacquired_before:
            self._capture_debug_image(
                "reacquire_success",
                frame_ref=submission.frame_ref,
                detection=detection,
                candidate_id=candidate.candidate_id,
                estimate=estimate,
                measurement=measurement,
            )
        return estimate

    def _log_candidate_transition(
        self,
        transition: str,
        *,
        timestamp_s: float,
        detection: TrackDetection,
        candidate_id: str,
        measurement: TargetMeasurement | None,
        estimate: TargetEstimate,
    ) -> None:
        """Emit one scalar-only event per lifecycle edge, never per frame.

        The event is intentionally restricted to production evidence. It does
        not contain target truth, simulator object identity, prim paths, motion
        seeds, images, or depth arrays.
        """

        if (
            self._candidate_transition_logs_emitted
            >= _MAX_CANDIDATE_TRANSITION_LOGS_PER_ASSIGNMENT
        ):
            return
        self._candidate_transition_logs_emitted += 1

        attribute_state = "pending"
        color_result: str | None = None
        records = getattr(self._semantic_provider, "evidence_records", None)
        if callable(records):
            for record in reversed(tuple(records())):
                if getattr(record, "candidate_id", None) != candidate_id:
                    continue
                raw_decision = getattr(record, "decision", "pending")
                attribute_state = str(
                    getattr(raw_decision, "value", raw_decision)
                ).casefold()
                raw_color = getattr(record, "observed_value", None)
                color_result = None if raw_color is None else str(raw_color)
                break
        position = estimate.position_world_m
        record = {
            "timestamp": float(timestamp_s),
            "uav_id": self._uav_id,
            "assignment_id": self._assignment_id,
            "transition": transition,
            "tracker_id": f"track_{detection.track_id}",
            "candidate_id": candidate_id,
            "bbox": list(detection.bbox_xyxy_normalized),
            "detector_confidence": float(detection.confidence),
            "attribute_state": attribute_state,
            "color_result": color_result,
            "geometry_state": (
                "measurement_created"
                if measurement is not None
                else "measurement_rejected"
            ),
            "measurement_source": (
                None if measurement is None else measurement.source
            ),
            "position_world_m": None if position is None else list(position),
            "confirmed": bool(estimate.confirmed),
            "target_id": estimate.target_id if estimate.confirmed else None,
            "estimate_source": estimate.source,
        }
        self._candidate_transition_records.append(dict(record))
        print(
            "[PerceptionCandidate] "
            + json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

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
            self.metrics.candidate_created += 1
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

        # A detector/semantic match without a finite current 3-D measurement
        # is not actionable identity evidence.  Keep initial acquisition and
        # REACQUIRE candidates pending; the early LOCKED/TRACKING same-ID path
        # above still permits bounded estimator prediction during a transient
        # depth outage.
        if position is None:
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
            or self._target_alias
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
            semantic,
            target_spec,
            position,
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
                    (
                        "qwen_reacquire_pending"
                        if self._config.confirmation.require_qwen_for_reacquire_new_track_id
                        else "deterministic_reacquire_pending"
                    )
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
            self.metrics.candidate_rejected += 1
            bank_candidate = self._require_candidate_bank().get(candidate.candidate_id)
            if bank_candidate is not None and bank_candidate.lifecycle not in {
                CandidateLifecycle.REJECTED,
                CandidateLifecycle.STALE,
            }:
                self._require_candidate_bank().reject(
                    candidate.candidate_id,
                    timestamp_s=confirmation_track.timestamp_s,
                )
            semantic_reject = getattr(
                self._semantic_provider,
                "reject_candidate",
                None,
            )
            if callable(semantic_reject):
                semantic_reject(candidate.candidate_id)
            return False, None
        if result.decision is ConfirmationDecision.CONFIRMED:
            self.metrics.candidates_confirmed += 1
            self.metrics.candidate_confirmed += 1
            if snapshot.lifecycle is TargetLifecycle.CANDIDATE:
                self.metrics.search_target_found += 1
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
            try:
                supplied = self._semantic_provider(
                    candidate, target_spec, detection, timestamp_s
                )
            except AttributeSemanticVerificationPending:
                # Insufficient deterministic evidence is neither a mismatch
                # nor permission to ask Qwen on every frame.
                return None
            except SemanticVerificationRequiresQwen:
                # Unsupported or persistently ambiguous attributes may use
                # the already-routed, asynchronous Qwen gate below.
                supplied = None
            if supplied is not None and (
                self._config.confirmation.mode != "qwen_required"
                or supplied.verifier == "qwen_vl"
            ):
                return supplied
        # A typed Qwen review may influence the control path only when the
        # deterministic provider explicitly declares that this candidate is
        # unsupported or persistently ambiguous.  In particular, a review
        # that happens to arrive during the normal multi-frame RGB-D warm-up
        # cannot short-circuit the color verifier.
        allow_qwen_fallback = (
            self._semantic_provider is None
            or self.qwen_fallback_required(candidate.candidate_id)
        )
        review = self._latest_visual_review(candidate) if allow_qwen_fallback else None
        if review is not None:
            try:
                verification = self._qwen_semantic_adapter.from_review(
                    candidate_id=candidate.candidate_id,
                    target_spec=target_spec,
                    review=review,
                    expected_bbox=_candidate_bbox_at(
                        candidate,
                        review.observation_timestamp_s,
                    ),
                )
                if self.qwen_fallback_required(candidate.candidate_id):
                    self.metrics.qwen_attribute_fallback_count += 1
                    clear_requirement = getattr(
                        self._semantic_provider,
                        "clear_qwen_requirement",
                        None,
                    )
                    if callable(clear_requirement):
                        clear_requirement(candidate.candidate_id)
                return verification
            except (QwenEvidencePending, VisualEvidenceError, ValueError):
                # A stale/misaligned typed review remains pending.  It is never
                # converted into class-only approval or an Oracle fallback.
                pass
        if self._semantic_provider is not None:
            # The configured provider owns semantic authority for this
            # runtime.  ``None`` means its temporal evidence is still pending
            # (or an authorised Qwen fallback has not arrived); falling
            # through to the legacy class-only verifier would silently bypass
            # RGB-D attributes.
            return None
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
        semantic: SemanticVerification,
        target_spec: TargetSpec,
        position: tuple[float, float, float] | None,
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
        if new_reacquire_track:
            return self._deterministic_reacquire_identity(
                candidate=candidate,
                target_id=target_id,
                timestamp_s=timestamp_s,
                track=track,
                semantic=semantic,
                target_spec=target_spec,
                position=position,
            )
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

    def _deterministic_reacquire_identity(
        self,
        *,
        candidate: CandidateSnapshot,
        target_id: str,
        timestamp_s: float,
        track: ShortTrackEvidence,
        semantic: SemanticVerification,
        target_spec: TargetSpec,
        position: tuple[float, float, float] | None,
    ) -> IdentityConsistencyEvidence | None:
        """Rebind a fragmented track only from bounded production evidence.

        This path is used only when the explicit Qwen-new-track gate is off.
        A class match alone is deliberately insufficient: a supported exact
        colour assertion must have terminal deterministic RGB-D evidence, the
        measured position must remain close to the last confirmed kinematic
        state, and the gap must fit inside the configured prediction horizon.
        Returning ``None`` keeps TargetManager in CANDIDATE.
        """

        expected_color: str | None = None
        for assertion in target_spec.hard_attributes:
            if assertion.count("=") != 1:
                return None
            name, value = assertion.split("=", 1)
            if name.casefold() == "color":
                if expected_color is not None:
                    return None
                expected_color = value.casefold()
        supported_colors = {
            value.casefold()
            for value in self._config.attributes.color.supported_values
        }
        color_consistent = bool(
            expected_color in supported_colors
            and semantic.matches
            and semantic.verifier == DETERMINISTIC_ATTRIBUTE_VERIFIER
            and semantic.confidence >= self._confirmation.policy.min_semantic_confidence
        )
        reference = self._latest_confirmed_estimate
        if (
            not color_consistent
            or reference is None
            or reference.target_id != target_id
            or reference.position_world_m is None
            or position is None
        ):
            return None
        elapsed_s = timestamp_s - reference.timestamp_s
        if (
            elapsed_s < 0.0
            or elapsed_s > self._config.state_estimator.max_prediction_age_s
        ):
            return None
        reference_velocity = reference.velocity_world_mps or (0.0, 0.0, 0.0)
        expected_position = tuple(
            reference.position_world_m[index]
            + reference_velocity[index] * elapsed_s
            for index in range(3)
        )
        position_error_m = sum(
            (position[index] - expected_position[index]) ** 2
            for index in range(3)
        ) ** 0.5
        if position_error_m > self._config.state_estimator.max_position_jump_m:
            return None
        return IdentityConsistencyEvidence(
            candidate_id=candidate.candidate_id,
            target_id=target_id,
            timestamp_s=timestamp_s,
            reidentified=True,
            temporally_consistent=track.stable,
            consistent_observations=track.observation_count,
            confidence=min(track.confidence, semantic.confidence),
            source="deterministic_color_position_time_reacquire",
        )

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

    def _resolve_measurement(
        self,
        candidate: CandidateSnapshot,
        timestamp_s: float,
        *,
        now_s: float,
    ) -> TargetMeasurement | None:
        self.metrics.depth_resolution_attempts += 1
        if now_s - timestamp_s > self._config.geometry.max_measurement_age_s:
            self._record_depth_resolution_failure(
                "measurement_too_old",
                candidate=candidate,
                timestamp_s=timestamp_s,
            )
            return None
        try:
            measurement = self._resolver.resolve(
                candidate,
                timestamp_s=timestamp_s,
            )
            if measurement is None:
                self._record_depth_resolution_failure(
                    "resolver_returned_none",
                    candidate=candidate,
                    timestamp_s=timestamp_s,
                )
                return None
            if not isinstance(measurement, TargetMeasurement):
                raise TypeError(
                    "production CandidateResolver must return TargetMeasurement or None"
                )
            self.metrics.depth_resolution_successes += 1
            self.metrics.measurement_created += 1
            return measurement
        except CandidateResolutionUnavailable as exc:
            self._record_depth_resolution_failure(
                _resolution_failure_reason(exc),
                candidate=candidate,
                timestamp_s=timestamp_s,
            )
            return None

    def _record_depth_resolution_failure(
        self,
        reason: str,
        *,
        candidate: CandidateSnapshot,
        timestamp_s: float,
    ) -> None:
        """Count every failure and emit exponentially throttled bounded logs."""

        normalized = _bounded_reason(reason)
        counts = self.metrics.depth_resolution_failure_reason_counts
        metric_key = normalized
        if metric_key not in counts and len(counts) >= _MAX_DEPTH_RESOLUTION_FAILURE_REASON_KEYS:
            metric_key = "other_resolution_failure"
        counts[metric_key] = counts.get(metric_key, 0) + 1
        self.metrics.depth_resolution_last_failure_reason = normalized
        self.metrics.depth_resolution_failures += 1
        self.metrics.measurement_rejected += 1

        assignment_count = self._depth_resolution_failure_log_counts.get(normalized, 0) + 1
        self._depth_resolution_failure_log_counts[normalized] = assignment_count
        # Emit occurrences 1, 2, 4, 8, ... for each reason, subject to a hard
        # per-assignment cap. Persistent failures remain visible without a
        # frame-rate log flood.
        should_emit = assignment_count & (assignment_count - 1) == 0
        if (
            not should_emit
            or self._depth_resolution_failure_logs_emitted
            >= _MAX_DEPTH_RESOLUTION_FAILURE_LOGS_PER_ASSIGNMENT
        ):
            return
        self._depth_resolution_failure_logs_emitted += 1
        print(
            "[PerceptionDepthFailure] "
            + json.dumps(
                {
                    "timestamp": timestamp_s,
                    "uav_id": self._uav_id,
                    "assignment_id": self._assignment_id,
                    "candidate_id": candidate.candidate_id,
                    "reason": normalized,
                    "reason_occurrence": assignment_count,
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

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
        measurement: TargetMeasurement | None = None,
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
                    sampled_pixel_uv=(
                        None if measurement is None else measurement.pixel_uv
                    ),
                    raw_depth_m=(
                        None if measurement is None else measurement.raw_depth_m
                    ),
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

    def _compile_target_query_spec(
        self,
        target_query: TargetQuerySpec,
    ) -> TargetQuery:
        """Compile the already-whitelisted production query."""

        if not isinstance(target_query, TargetQuerySpec):
            raise TypeError("target_query must be a TargetQuerySpec")
        self._validate_target_query_model(target_query, self._model_names)
        if self._config.detector.model_family == "yolo":
            return TargetQuery(
                class_ids=(target_query.detector_class_id,),
                text_prompts=(),
            )
        # YOLOE remains open-vocabulary, but its prompt is compiled only from
        # the same whitelisted semantic value.  No scene/evaluator data is
        # available at this boundary.
        return compile_target_query(
            target_query.to_semantic_target_spec(),
            self._config.detector.model_family,
            self._model_names,
            self._alias_mapper,
        )

    def _validate_target_query_model(
        self,
        target_query: TargetQuerySpec,
        model_names: Mapping[int, str],
    ) -> None:
        if self._config.detector.model_family != "yolo":
            return
        actual_name = model_names.get(target_query.detector_class_id)
        if actual_name != target_query.detector_class_name:
            raise TargetQueryUnsupported(
                "production target query class does not exactly match loaded "
                "model.names: "
                f"requested={target_query.detector_class_id}:"
                f"{target_query.detector_class_name!r}, actual={actual_name!r}"
            )

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

    @staticmethod
    def _drain_executor_future(
        future: Future[object] | None,
        *,
        timeout_s: float,
    ) -> None:
        if future is None or future.cancelled():
            return
        try:
            future.result(timeout=timeout_s)
        except Exception:
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
            except (YoloClientStreamBusy, YoloClientResponseError) as exc:
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


def _bounded_reason(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("resolution failure reason must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "candidate_resolution_unavailable"
    return normalized[:256]


def _resolution_failure_reason(exc: CandidateResolutionUnavailable) -> str:
    message = _bounded_reason(str(exc))
    return _bounded_reason(f"{type(exc).__name__}:{message}")


__all__ = [
    "TargetPerceptionCoordinator",
    "TargetPerceptionError",
    "TargetPerceptionMetrics",
    "TargetPerceptionNotReady",
    "TargetQueryUnsupported",
]
