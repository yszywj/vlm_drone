"""Concrete adapters from Ultralytics/Qwen results to confirmation evidence.

The adapters in this module only build the existing evidence values consumed
by :class:`perception.confirmation.CandidateConfirmationCoordinator`.  They do
not lock targets or maintain a second lifecycle.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot, isfinite
from numbers import Integral, Real

from common.ids import validate_routing_id
from perception.candidate_bank import CandidateSnapshot
from perception.class_aliases import ClassAliasMapper
from perception.qwen_vlm_verifier import QwenVLMVerifier
from perception.types import (
    DetectionCandidate,
    IdentityConsistencyEvidence,
    SemanticVerification,
    ShortTrackEvidence,
)
from perception.visual_review import QwenVisualReview, VisualReviewDecision
from target import TargetSpec
from yolo_service.protocol import TrackDetection


SEMANTIC_REQUIRES_QWEN = "SEMANTIC_REQUIRES_QWEN"
REACQUIRE_REQUIRES_QWEN = "REACQUIRE_REQUIRES_QWEN"
QWEN_EVIDENCE_PENDING = "QWEN_EVIDENCE_PENDING"
NON_MONOTONIC_TRACK_TIME = "NON_MONOTONIC_TRACK_TIME"


class VisualEvidenceError(RuntimeError):
    """Base error for evidence that cannot safely enter confirmation."""

    code = "VISUAL_EVIDENCE_ERROR"


class SemanticVerificationRequiresQwen(VisualEvidenceError):
    """Closed-set class equality cannot prove the requested semantics."""

    code = SEMANTIC_REQUIRES_QWEN


class ReacquireIdentityRequiresQwen(VisualEvidenceError):
    """A new tracker ID cannot inherit an old target identity by category."""

    code = REACQUIRE_REQUIRES_QWEN


class QwenEvidencePending(VisualEvidenceError):
    """A non-terminal Qwen review must remain pending, not become rejection."""

    code = QWEN_EVIDENCE_PENDING


class NonMonotonicTrackTime(VisualEvidenceError):
    """Incremental tracker history attempted to move backwards in time."""

    code = NON_MONOTONIC_TRACK_TIME


def _timestamp(value: object, name: str = "timestamp_s") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _confidence(value: object) -> float:
    result = _timestamp(value, "confidence")
    if result > 1.0:
        raise ValueError("confidence must be within [0, 1]")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _bbox(value: object) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("bbox must contain four normalized coordinates")
    if len(value) != 4:
        raise ValueError("bbox must contain four normalized coordinates")
    coordinates = tuple(_timestamp(item, "bbox coordinate") for item in value)
    if any(item > 1.0 for item in coordinates):
        raise ValueError("bbox coordinates must be within [0, 1]")
    x1, y1, x2, y2 = coordinates
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
    return coordinates  # type: ignore[return-value]


def _track_id(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("track_id must be a non-negative integer or routing string")
    if isinstance(value, Integral):
        if int(value) < 0:
            raise ValueError("track_id must be non-negative")
        return str(int(value))
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return str(int(value))
    return validate_routing_id(value, "track_id")


def _reference_handle(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reference_handle must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > 512:
        raise ValueError("reference_handle must contain at most 512 characters")
    return normalized


@dataclass(frozen=True, slots=True)
class TrackBoxObservation:
    """One bounded tracker observation used to prove short-track stability."""

    candidate_id: str
    track_id: str | int
    timestamp_s: float
    bbox_xyxy_normalized: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            validate_routing_id(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(self, "track_id", _track_id(self.track_id))
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s))
        object.__setattr__(
            self,
            "bbox_xyxy_normalized",
            _bbox(self.bbox_xyxy_normalized),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))


class UltralyticsShortTrackEvidenceBuilder:
    """Build stable-track evidence from bounded ordered box observations.

    ``build_history`` is stateless and marks a non-monotonic sequence
    unstable.  ``update`` is the live bounded form and rejects time reversal
    before it can corrupt its retained history.
    """

    def __init__(
        self,
        *,
        min_observations: int = 3,
        min_duration_s: float = 0.5,
        max_center_jump_normalized: float = 0.35,
        max_observation_gap_s: float = 1.0,
        max_history_per_track: int = 32,
        max_tracks: int = 32,
    ) -> None:
        self._min_observations = _positive_int(
            min_observations, "min_observations"
        )
        self._min_duration_s = _timestamp(min_duration_s, "min_duration_s")
        self._max_center_jump = _timestamp(
            max_center_jump_normalized,
            "max_center_jump_normalized",
        )
        if self._max_center_jump > 2**0.5:
            raise ValueError("max_center_jump_normalized exceeds image diagonal")
        self._max_gap_s = _timestamp(
            max_observation_gap_s,
            "max_observation_gap_s",
        )
        if self._max_gap_s == 0.0:
            raise ValueError("max_observation_gap_s must be greater than zero")
        self._max_history = _positive_int(
            max_history_per_track,
            "max_history_per_track",
        )
        self._max_tracks = _positive_int(max_tracks, "max_tracks")
        self._histories: dict[str, deque[TrackBoxObservation]] = {}

    def update(
        self,
        *,
        candidate_id: str,
        timestamp_s: float,
        detection: TrackDetection | None = None,
        track_id: str | int | None = None,
        bbox_xyxy_normalized: Sequence[float] | None = None,
        confidence: float | None = None,
    ) -> ShortTrackEvidence:
        """Append one detection and return evidence for its bounded history."""

        candidate_id = validate_routing_id(candidate_id, "candidate_id")
        if detection is not None:
            if not isinstance(detection, TrackDetection):
                raise TypeError("detection must be a TrackDetection or None")
            if any(
                item is not None
                for item in (track_id, bbox_xyxy_normalized, confidence)
            ):
                raise ValueError(
                    "raw track fields cannot be combined with detection"
                )
            track_id = detection.track_id
            bbox_xyxy_normalized = detection.bbox_xyxy_normalized
            confidence = detection.confidence
        if track_id is None or bbox_xyxy_normalized is None or confidence is None:
            raise ValueError(
                "track_id, bbox_xyxy_normalized, and confidence are required"
            )
        observation = TrackBoxObservation(
            candidate_id=candidate_id,
            track_id=track_id,
            timestamp_s=timestamp_s,
            bbox_xyxy_normalized=tuple(bbox_xyxy_normalized),  # type: ignore[arg-type]
            confidence=confidence,
        )
        history = self._histories.get(candidate_id)
        if history is not None and observation.timestamp_s <= history[-1].timestamp_s:
            raise NonMonotonicTrackTime(
                f"{NON_MONOTONIC_TRACK_TIME}: candidate {candidate_id!r} "
                "timestamp must increase strictly"
            )
        if history is None:
            if len(self._histories) >= self._max_tracks:
                oldest_id = min(
                    self._histories,
                    key=lambda item: (
                        self._histories[item][-1].timestamp_s,
                        item,
                    ),
                )
                del self._histories[oldest_id]
            history = deque(maxlen=self._max_history)
            self._histories[candidate_id] = history
        history.append(observation)
        return self.build_history(tuple(history))

    def reset(self, candidate_id: str | None = None) -> None:
        """Drop one track or all bounded histories at a mission boundary."""

        if candidate_id is None:
            self._histories.clear()
            return
        normalized = validate_routing_id(candidate_id, "candidate_id")
        self._histories.pop(normalized, None)

    def build(
        self,
        candidate_or_history: CandidateSnapshot | Sequence[TrackBoxObservation] | str,
        *,
        confidences: Sequence[float] | float | None = None,
        track_id: str | int | None = None,
    ) -> ShortTrackEvidence:
        """Build from CandidateBank state, explicit history, or retained ID."""

        if isinstance(candidate_or_history, CandidateSnapshot):
            return self.build_candidate(
                candidate_or_history,
                confidences=confidences,
                track_id=track_id,
            )
        if isinstance(candidate_or_history, str):
            candidate_id = validate_routing_id(
                candidate_or_history,
                "candidate_id",
            )
            try:
                history = tuple(self._histories[candidate_id])
            except KeyError:
                raise ValueError(
                    f"no retained track history for {candidate_id!r}"
                ) from None
            return self.build_history(history)
        if confidences is not None or track_id is not None:
            raise ValueError(
                "confidences and track_id only apply to CandidateSnapshot input"
            )
        return self.build_history(candidate_or_history)

    def build_candidate(
        self,
        candidate: CandidateSnapshot,
        *,
        confidences: Sequence[float] | float | None = None,
        track_id: str | int | None = None,
    ) -> ShortTrackEvidence:
        if not isinstance(candidate, CandidateSnapshot):
            raise TypeError("candidate must be a CandidateSnapshot")
        if len(candidate.bbox_history) != len(candidate.frame_history):
            raise ValueError("candidate bbox and frame histories must be aligned")
        count = len(candidate.frame_history)
        if confidences is None:
            # CandidateBank intentionally stores no detector confidence.  Zero
            # keeps the evidence honest and prevents confirmation until the
            # caller supplies the corresponding detector scores.
            scores = (0.0,) * count
        elif isinstance(confidences, Real) and not isinstance(confidences, bool):
            scores = (_confidence(confidences),) * count
        else:
            if isinstance(confidences, (str, bytes)) or not isinstance(
                confidences, Sequence
            ):
                raise TypeError("confidences must be a number or score sequence")
            if len(confidences) != count:
                raise ValueError("confidences must align with candidate history")
            scores = tuple(_confidence(value) for value in confidences)
        actual_track_id = candidate.candidate_id if track_id is None else track_id
        observations = tuple(
            TrackBoxObservation(
                candidate_id=candidate.candidate_id,
                track_id=actual_track_id,
                timestamp_s=frame.timestamp_s,
                bbox_xyxy_normalized=bbox,
                confidence=score,
            )
            for frame, bbox, score in zip(
                candidate.frame_history,
                candidate.bbox_history,
                scores,
            )
        )
        return self.build_history(observations)

    def build_history(
        self,
        history: Sequence[TrackBoxObservation],
    ) -> ShortTrackEvidence:
        if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
            raise TypeError("history must be a sequence of TrackBoxObservation")
        observations = tuple(history)
        if not observations:
            raise ValueError("history must not be empty")
        if any(not isinstance(item, TrackBoxObservation) for item in observations):
            raise TypeError("history must contain TrackBoxObservation values")
        first = observations[0]
        timestamps = tuple(item.timestamp_s for item in observations)
        monotonic = all(
            current > previous
            for previous, current in zip(timestamps, timestamps[1:])
        )
        same_candidate = all(
            item.candidate_id == first.candidate_id for item in observations
        )
        same_track = all(item.track_id == first.track_id for item in observations)
        duration = (
            timestamps[-1] - timestamps[0]
            if monotonic
            else max(timestamps) - min(timestamps)
        )
        gaps = tuple(
            current - previous
            for previous, current in zip(timestamps, timestamps[1:])
        )
        continuous = monotonic and all(gap <= self._max_gap_s for gap in gaps)
        boxes = tuple(item.bbox_xyxy_normalized for item in observations)
        centers = tuple(
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            for x1, y1, x2, y2 in boxes
        )
        no_jump = all(
            hypot(current[0] - previous[0], current[1] - previous[1])
            <= self._max_center_jump
            for previous, current in zip(centers, centers[1:])
        )
        stable = all(
            (
                same_candidate,
                same_track,
                monotonic,
                continuous,
                no_jump,
                len(observations) >= self._min_observations,
                duration >= self._min_duration_s,
            )
        )
        return ShortTrackEvidence(
            candidate_id=first.candidate_id,
            timestamp_s=max(timestamps),
            observation_count=len(observations),
            duration_s=duration,
            stable=stable,
            confidence=min(item.confidence for item in observations),
        )


def _requires_specific_identity_check(
    target_spec: TargetSpec,
    mapper: ClassAliasMapper,
    canonical_name: str,
) -> bool:
    # Exact aliases describe a class.  Any richer free-form identity text is
    # treated as specific and therefore cannot be proven by class equality.
    return not (
        mapper.category_is_exact_alias(
            target_spec.original_description,
            canonical_name,
        )
        and mapper.category_is_exact_alias(
            target_spec.immutable_identity_summary,
            canonical_name,
        )
    )


class ClosedSetClassSemanticVerifier:
    """Verify category equality only when TargetSpec needs no richer proof."""

    def __init__(
        self,
        mapper: ClassAliasMapper,
        model_names: Mapping[int, str] | Sequence[str],
    ) -> None:
        if not isinstance(mapper, ClassAliasMapper):
            raise TypeError("mapper must be a ClassAliasMapper")
        self._mapper = mapper
        # Resolution performs strict model-name validation.  Keep a detached
        # value so a caller cannot mutate the live class table after audit.
        self._model_names = (
            dict(model_names)
            if isinstance(model_names, Mapping)
            else tuple(model_names)
        )

    def requires_qwen(self, target_spec: TargetSpec) -> bool:
        """Return whether category equality is insufficient for this target."""

        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        resolved = self._mapper.resolve(target_spec.category, self._model_names)
        return bool(
            target_spec.hard_attributes
            or target_spec.negative_constraints
            or target_spec.relation_constraints
            or _requires_specific_identity_check(
                target_spec,
                self._mapper,
                resolved.canonical_name,
            )
        )

    def verify(
        self,
        *,
        candidate_id: str,
        timestamp_s: float,
        target_spec: TargetSpec,
        detection: TrackDetection,
    ) -> SemanticVerification:
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        if not isinstance(detection, TrackDetection):
            raise TypeError("detection must be a TrackDetection")
        resolved = self._mapper.resolve(target_spec.category, self._model_names)
        needs_qwen = self.requires_qwen(target_spec)
        if needs_qwen:
            raise SemanticVerificationRequiresQwen(
                f"{SEMANTIC_REQUIRES_QWEN}: target contains attributes, "
                "relations, exclusions, or specific identity text"
            )
        matches = (
            detection.class_id == resolved.class_id
            and detection.class_name == resolved.class_name
        )
        return SemanticVerification(
            candidate_id=validate_routing_id(candidate_id, "candidate_id"),
            timestamp_s=_timestamp(timestamp_s),
            target_description=target_spec.description,
            matches=matches,
            confidence=detection.confidence,
            verifier="closed_set_model_class",
        )

    def verify_candidate(
        self,
        candidate: DetectionCandidate,
        target_spec: TargetSpec,
        detection: TrackDetection,
    ) -> SemanticVerification:
        if not isinstance(candidate, DetectionCandidate):
            raise TypeError("candidate must be a DetectionCandidate")
        return self.verify(
            candidate_id=candidate.candidate_id,
            timestamp_s=candidate.timestamp_s,
            target_spec=target_spec,
            detection=detection,
        )


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _validate_review_candidate(
    review: QwenVisualReview,
    expected_bbox: Sequence[float] | None,
    min_bbox_iou: float,
) -> None:
    if expected_bbox is None or not review.candidate.present:
        return
    expected = _bbox(expected_bbox)
    candidate_bbox = review.candidate.bbox_xyxy_normalized
    if candidate_bbox is None or _bbox_iou(expected, candidate_bbox) < min_bbox_iou:
        raise VisualEvidenceError(
            "Qwen review candidate does not overlap the expected tracker box"
        )


def _terminal_review_match(review: QwenVisualReview) -> tuple[bool, float]:
    if review.decision is VisualReviewDecision.TARGET_MATCH:
        if not review.candidate.present:
            raise VisualEvidenceError("TARGET_MATCH requires a present candidate")
        confidence = review.candidate.self_reported_confidence
        if confidence is None:  # guarded by VisualReviewCandidate, defensive here
            raise VisualEvidenceError("TARGET_MATCH has no confidence")
        return True, confidence
    if review.decision in {
        VisualReviewDecision.TARGET_MISMATCH,
        VisualReviewDecision.NO_TARGET,
    }:
        confidence = review.candidate.self_reported_confidence
        return False, 0.0 if confidence is None else confidence
    raise QwenEvidencePending(
        f"{QWEN_EVIDENCE_PENDING}: review decision "
        f"{review.decision.value} is not terminal identity evidence"
    )


class QwenSemanticVerifierAdapter:
    """Thin Qwen request/parser delegate plus typed semantic conversion."""

    def __init__(self, verifier: QwenVLMVerifier | None = None) -> None:
        self._verifier = verifier or QwenVLMVerifier()
        if not isinstance(self._verifier, QwenVLMVerifier):
            raise TypeError("verifier must be a QwenVLMVerifier")

    @property
    def verifier(self) -> QwenVLMVerifier:
        return self._verifier

    def build_async_request(self, *args: object, **kwargs: object) -> object:
        return self._verifier.build_async_request(*args, **kwargs)  # type: ignore[arg-type]

    def parse_async_result(self, *args: object, **kwargs: object) -> QwenVisualReview:
        return self._verifier.parse_async_result(*args, **kwargs)  # type: ignore[arg-type]

    def from_review(
        self,
        *,
        candidate_id: str,
        target_spec: TargetSpec,
        review: QwenVisualReview,
        expected_bbox: Sequence[float] | None = None,
        min_bbox_iou: float = 0.25,
    ) -> SemanticVerification:
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        if not isinstance(review, QwenVisualReview):
            raise TypeError("review must be a QwenVisualReview")
        threshold = _confidence(min_bbox_iou)
        _validate_review_candidate(review, expected_bbox, threshold)
        matches, confidence = _terminal_review_match(review)
        return SemanticVerification(
            candidate_id=validate_routing_id(candidate_id, "candidate_id"),
            timestamp_s=review.observation_timestamp_s,
            target_description=target_spec.description,
            matches=matches,
            confidence=confidence,
            verifier="qwen_vl",
        )


class TemporalTrackIdentityVerifier:
    """Prove initial identity only by one stable tracker ID."""

    def verify(
        self,
        *,
        track: ShortTrackEvidence,
        target_id: str,
        reference_track_id: str | int,
        current_track_id: str | int,
        reacquiring: bool = False,
        timestamp_s: float | None = None,
    ) -> IdentityConsistencyEvidence:
        if not isinstance(track, ShortTrackEvidence):
            raise TypeError("track must be ShortTrackEvidence")
        if not isinstance(reacquiring, bool):
            raise TypeError("reacquiring must be bool")
        reference_id = _track_id(reference_track_id)
        current_id = _track_id(current_track_id)
        same_track = reference_id == current_id
        if reacquiring and not same_track:
            raise ReacquireIdentityRequiresQwen(
                f"{REACQUIRE_REQUIRES_QWEN}: tracker ID changed from "
                f"{reference_id!r} to {current_id!r}"
            )
        evidence_time = (
            track.timestamp_s if timestamp_s is None else _timestamp(timestamp_s)
        )
        if evidence_time < track.timestamp_s:
            raise ValueError("identity timestamp cannot predate short-track evidence")
        return IdentityConsistencyEvidence(
            candidate_id=track.candidate_id,
            target_id=validate_routing_id(target_id, "target_id"),
            timestamp_s=evidence_time,
            reidentified=same_track,
            temporally_consistent=same_track and track.stable,
            consistent_observations=track.observation_count,
            confidence=track.confidence,
            source="temporal_track",
        )


class QwenReacquireIdentityVerifierAdapter:
    """Convert a reference-frame Qwen review for a changed tracker ID."""

    def __init__(self, verifier: QwenVLMVerifier | None = None) -> None:
        self._semantic_adapter = QwenSemanticVerifierAdapter(verifier)

    @property
    def verifier(self) -> QwenVLMVerifier:
        return self._semantic_adapter.verifier

    def build_async_request(self, *args: object, **kwargs: object) -> object:
        return self.verifier.build_async_request(*args, **kwargs)  # type: ignore[arg-type]

    def parse_async_result(self, *args: object, **kwargs: object) -> QwenVisualReview:
        return self.verifier.parse_async_result(*args, **kwargs)  # type: ignore[arg-type]

    def from_review(
        self,
        *,
        track: ShortTrackEvidence,
        target_id: str,
        review: QwenVisualReview,
        reference_handles: Sequence[str],
        expected_bbox: Sequence[float] | None = None,
        min_bbox_iou: float = 0.25,
    ) -> IdentityConsistencyEvidence:
        if not isinstance(track, ShortTrackEvidence):
            raise TypeError("track must be ShortTrackEvidence")
        if not isinstance(review, QwenVisualReview):
            raise TypeError("review must be a QwenVisualReview")
        if isinstance(reference_handles, (str, bytes)) or not isinstance(
            reference_handles,
            Sequence,
        ):
            raise TypeError("reference_handles must be a sequence of frame handles")
        handles = tuple(
            _reference_handle(value)
            for value in reference_handles
        )
        if not handles:
            raise ReacquireIdentityRequiresQwen(
                f"{REACQUIRE_REQUIRES_QWEN}: historical reference frames are required"
            )
        if review.observation_timestamp_s < track.timestamp_s:
            raise VisualEvidenceError("Qwen identity review predates short-track evidence")
        threshold = _confidence(min_bbox_iou)
        _validate_review_candidate(review, expected_bbox, threshold)
        matches, confidence = _terminal_review_match(review)
        return IdentityConsistencyEvidence(
            candidate_id=track.candidate_id,
            target_id=validate_routing_id(target_id, "target_id"),
            timestamp_s=review.observation_timestamp_s,
            reidentified=matches,
            temporally_consistent=matches and track.stable,
            consistent_observations=track.observation_count,
            confidence=min(confidence, track.confidence),
            source="qwen_reacquire",
        )


__all__ = [
    "ClosedSetClassSemanticVerifier",
    "NON_MONOTONIC_TRACK_TIME",
    "NonMonotonicTrackTime",
    "QWEN_EVIDENCE_PENDING",
    "QwenEvidencePending",
    "QwenReacquireIdentityVerifierAdapter",
    "QwenSemanticVerifierAdapter",
    "REACQUIRE_REQUIRES_QWEN",
    "ReacquireIdentityRequiresQwen",
    "SEMANTIC_REQUIRES_QWEN",
    "SemanticVerificationRequiresQwen",
    "TemporalTrackIdentityVerifier",
    "TrackBoxObservation",
    "UltralyticsShortTrackEvidenceBuilder",
    "VisualEvidenceError",
]
