"""Deterministic fusion of closed-set cube and temporal RGB-D attributes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from threading import RLock

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from perception.attribute_types import (
    AttributeDecision,
    AttributeEvidence,
    AttributeRequirement,
    AttributeVerificationBundle,
)
from perception.attribute_verifier import (
    AttributeRouteMismatch,
    AttributeVerificationRoute,
)
from perception.candidate_bank import CandidateSnapshot
from perception.color_attribute_verifier import (
    RgbdColorAttributeVerifier,
    TemporalColorEvidenceAccumulator,
)
from perception.types import SemanticVerification
from perception.visual_evidence import (
    SemanticVerificationRequiresQwen,
    VisualEvidenceError,
)
from target.types import TargetSpec
from runtime.frame_store import FrameStore
from yolo_service.protocol import TrackDetection


DETERMINISTIC_ATTRIBUTE_VERIFIER = "closed_set_class+temporal_rgbd_color"
ATTRIBUTE_SEMANTIC_PENDING = "ATTRIBUTE_SEMANTIC_PENDING"
ATTRIBUTE_SEMANTIC_REQUIRES_QWEN = "ATTRIBUTE_SEMANTIC_REQUIRES_QWEN"


class AttributeSemanticVerificationPending(VisualEvidenceError):
    """The candidate must remain CANDIDATE while color evidence accumulates."""

    code = ATTRIBUTE_SEMANTIC_PENDING


class AttributeSemanticVerificationRequiresQwen(SemanticVerificationRequiresQwen):
    """A deterministic attribute is unsupported or persistently ambiguous."""

    code = ATTRIBUTE_SEMANTIC_REQUIRES_QWEN


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timestamp_s must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError("timestamp_s must be finite and non-negative")
    return result


def _class_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("expected_class_id must be an integer or None")
    result = int(value)
    if result < 0:
        raise ValueError("expected_class_id must be non-negative")
    return result


def _hard_attributes(target_spec: TargetSpec) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in target_spec.hard_attributes:
        if raw.count("=") != 1:
            raise AttributeSemanticVerificationRequiresQwen(
                f"{ATTRIBUTE_SEMANTIC_REQUIRES_QWEN}: hard attribute {raw!r} "
                "is not an exact name=value assertion"
            )
        name, value = raw.split("=", 1)
        if not name or not value or name != name.strip() or value != value.strip():
            raise AttributeSemanticVerificationRequiresQwen(
                f"{ATTRIBUTE_SEMANTIC_REQUIRES_QWEN}: malformed hard attribute"
            )
        name, value = name.casefold(), value.casefold()
        if name in parsed:
            raise ValueError(f"duplicate hard attribute {name!r}")
        parsed[name] = value
    return parsed


class DeterministicAttributeSemanticVerifier:
    """Fuse exact YOLO class equality with final temporal color evidence.

    ``PENDING`` never becomes ``matches=False`` because doing so would reject a
    valid candidate from one weak frame.  Callers keep the target in CANDIDATE
    when :class:`AttributeSemanticVerificationPending` is raised.  Unsupported
    attributes use the distinct ``RequiresQwen`` exception so a rate-limited,
    acknowledged production gate can decide whether to request Qwen.
    """

    def __init__(
        self,
        *,
        expected_class_name: str = "cube",
        expected_class_id: int | None = None,
        supported_attributes: tuple[str, ...] = ("color",),
        min_color_observations: int = 3,
        min_color_duration_s: float = 0.4,
        qwen_pending_min_observations: int | None = None,
        qwen_pending_min_duration_s: float | None = None,
    ) -> None:
        if (
            not isinstance(expected_class_name, str)
            or not expected_class_name
            or expected_class_name != expected_class_name.strip()
        ):
            raise ValueError("expected_class_name must be a non-empty string")
        if not isinstance(supported_attributes, tuple) or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in supported_attributes
        ):
            raise TypeError("supported_attributes must be a tuple of strings")
        normalized_attributes = tuple(value.casefold() for value in supported_attributes)
        if len(set(normalized_attributes)) != len(normalized_attributes):
            raise ValueError("supported_attributes must not contain duplicates")
        self._expected_class_name = expected_class_name
        self._expected_class_id = _class_id(expected_class_id)
        self._supported_attributes = normalized_attributes
        if isinstance(min_color_observations, bool) or not isinstance(
            min_color_observations, Integral
        ):
            raise TypeError("min_color_observations must be an integer")
        if int(min_color_observations) <= 0:
            raise ValueError("min_color_observations must be positive")
        self._min_color_observations = int(min_color_observations)
        self._min_color_duration_s = _timestamp(min_color_duration_s)
        qwen_count = (
            self._min_color_observations
            if qwen_pending_min_observations is None
            else qwen_pending_min_observations
        )
        if isinstance(qwen_count, bool) or not isinstance(qwen_count, Integral):
            raise TypeError("qwen_pending_min_observations must be an integer or None")
        if int(qwen_count) <= 0:
            raise ValueError("qwen_pending_min_observations must be positive")
        self._qwen_pending_min_observations = int(qwen_count)
        self._qwen_pending_min_duration_s = (
            self._min_color_duration_s
            if qwen_pending_min_duration_s is None
            else _timestamp(qwen_pending_min_duration_s)
        )

    @property
    def verifier_name(self) -> str:
        return DETERMINISTIC_ATTRIBUTE_VERIFIER

    def qwen_fallback_eligible(self, evidence: AttributeEvidence) -> bool:
        """Return whether PENDING evidence has reached the bounded Qwen gate."""

        if not isinstance(evidence, AttributeEvidence):
            raise TypeError("evidence must be AttributeEvidence")
        return bool(
            evidence.decision is AttributeDecision.UNSUPPORTED
            or (
                evidence.decision is AttributeDecision.PENDING
                and evidence.observation_count
                >= self._qwen_pending_min_observations
                and evidence.duration_s + 1e-12
                >= self._qwen_pending_min_duration_s
            )
        )

    @classmethod
    def from_color_config(
        cls,
        config: object,
        *,
        expected_class_name: str = "cube",
        expected_class_id: int | None = None,
    ) -> "DeterministicAttributeSemanticVerifier":
        from configs.schema import TargetColorAttributeConfig

        if not isinstance(config, TargetColorAttributeConfig):
            raise TypeError("config must be a TargetColorAttributeConfig")
        if not config.enabled:
            raise ValueError("color attribute verification is disabled")
        return cls(
            expected_class_name=expected_class_name,
            expected_class_id=expected_class_id,
            min_color_observations=config.min_observations,
            min_color_duration_s=config.min_duration_s,
        )

    def verify(
        self,
        *,
        candidate_id: str,
        timestamp_s: float,
        target_spec: TargetSpec,
        detection: TrackDetection,
        attribute_evidence: AttributeEvidence | AttributeVerificationBundle | None = None,
        mission_id: str | None = None,
        uav_id: str | None = None,
        assignment_id: str | None = None,
        qwen_verification: SemanticVerification | None = None,
        qwen_mode: str = "disabled",
        acknowledge_vision_gate: bool = False,
    ) -> SemanticVerification:
        candidate = validate_routing_id(candidate_id, "candidate_id")
        timestamp = _timestamp(timestamp_s)
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        if not isinstance(detection, TrackDetection):
            raise TypeError("detection must be a TrackDetection")
        if not isinstance(qwen_mode, str) or qwen_mode not in {
            "disabled",
            "shadow",
            "gate",
        }:
            raise ValueError("qwen_mode must be disabled, shadow, or gate")
        if not isinstance(acknowledge_vision_gate, bool):
            raise TypeError("acknowledge_vision_gate must be bool")
        if qwen_mode == "gate" and not acknowledge_vision_gate:
            raise PermissionError("Qwen vision gate requires explicit acknowledgement")
        route_values = (mission_id, uav_id, assignment_id)
        if any(value is None for value in route_values) and not all(
            value is None for value in route_values
        ):
            raise ValueError(
                "mission_id, uav_id, and assignment_id must be supplied together"
            )
        if mission_id is not None:
            mission_id = validate_mission_id(mission_id)
            assert uav_id is not None and assignment_id is not None
            uav_id = validate_uav_id(uav_id)
            assignment_id = validate_routing_id(assignment_id, "assignment_id")
        if qwen_mode == "disabled" and qwen_verification is not None:
            raise ValueError("Qwen evidence cannot be supplied when qwen_mode is disabled")

        evidence: AttributeEvidence | None
        if isinstance(attribute_evidence, AttributeVerificationBundle):
            evidence = attribute_evidence.evidence
        elif isinstance(attribute_evidence, AttributeEvidence):
            evidence = attribute_evidence
        elif attribute_evidence is None:
            evidence = None
        else:
            raise TypeError(
                "attribute_evidence must be AttributeEvidence, "
                "AttributeVerificationBundle, or None"
            )
        if evidence is not None and mission_id is None:
            raise ValueError(
                "routed attribute evidence requires mission_id, uav_id, and assignment_id"
            )
        hard_attributes = _hard_attributes(target_spec)
        expected_color = hard_attributes.get("color")
        if evidence is not None and expected_color is not None:
            self._validate_evidence_route(
                evidence=evidence,
                candidate_id=candidate,
                timestamp_s=timestamp,
                detection=detection,
                expected_color=expected_color,
                mission_id=mission_id,
                uav_id=uav_id,
                assignment_id=assignment_id,
            )

        class_matches = detection.class_name == self._expected_class_name and (
            self._expected_class_id is None
            or detection.class_id == self._expected_class_id
        )
        if not class_matches:
            if qwen_verification is not None and qwen_mode == "gate":
                raise ValueError(
                    "deterministic class mismatch is terminal; Qwen gate must not be called"
                )
            return SemanticVerification(
                candidate_id=candidate,
                timestamp_s=timestamp,
                target_description=target_spec.description,
                matches=False,
                confidence=detection.confidence,
                verifier=DETERMINISTIC_ATTRIBUTE_VERIFIER,
            )

        if target_spec.negative_constraints or target_spec.relation_constraints:
            reasons: list[str] = []
            if target_spec.negative_constraints:
                reasons.append("negative_constraints")
            if target_spec.relation_constraints:
                reasons.append("relation_constraints")
            return self._qwen_or_raise(
                candidate_id=candidate,
                timestamp_s=timestamp,
                target_spec=target_spec,
                detector_confidence=detection.confidence,
                qwen_verification=qwen_verification,
                qwen_mode=qwen_mode,
                acknowledge_vision_gate=acknowledge_vision_gate,
                reason="unsupported_semantics:" + ",".join(reasons),
            )

        unsupported = tuple(
            sorted(name for name in hard_attributes if name not in self._supported_attributes)
        )
        if unsupported:
            return self._qwen_or_raise(
                candidate_id=candidate,
                timestamp_s=timestamp,
                target_spec=target_spec,
                detector_confidence=detection.confidence,
                qwen_verification=qwen_verification,
                qwen_mode=qwen_mode,
                acknowledge_vision_gate=acknowledge_vision_gate,
                reason="unsupported_attributes:" + ",".join(unsupported),
            )

        if expected_color is None:
            if qwen_verification is not None and qwen_mode == "gate":
                raise ValueError(
                    "deterministic class evidence is complete; Qwen gate must not be called"
                )
            return SemanticVerification(
                candidate_id=candidate,
                timestamp_s=timestamp,
                target_description=target_spec.description,
                matches=True,
                confidence=detection.confidence,
                verifier=DETERMINISTIC_ATTRIBUTE_VERIFIER,
            )

        if evidence is None:
            return self._qwen_or_raise(
                candidate_id=candidate,
                timestamp_s=timestamp,
                target_spec=target_spec,
                detector_confidence=detection.confidence,
                qwen_verification=qwen_verification,
                qwen_mode=qwen_mode,
                acknowledge_vision_gate=acknowledge_vision_gate,
                reason="color_evidence_absent",
                pending=True,
                allow_gate=False,
            )
        if evidence.decision is AttributeDecision.PENDING:
            return self._qwen_or_raise(
                candidate_id=candidate,
                timestamp_s=timestamp,
                target_spec=target_spec,
                detector_confidence=detection.confidence,
                qwen_verification=qwen_verification,
                qwen_mode=qwen_mode,
                acknowledge_vision_gate=acknowledge_vision_gate,
                reason=evidence.reason_code,
                pending=True,
                allow_gate=(
                    evidence.observation_count
                    >= self._qwen_pending_min_observations
                    and evidence.duration_s + 1e-12
                    >= self._qwen_pending_min_duration_s
                ),
            )
        if evidence.decision is AttributeDecision.UNSUPPORTED:
            return self._qwen_or_raise(
                candidate_id=candidate,
                timestamp_s=timestamp,
                target_spec=target_spec,
                detector_confidence=detection.confidence,
                qwen_verification=qwen_verification,
                qwen_mode=qwen_mode,
                acknowledge_vision_gate=acknowledge_vision_gate,
                reason=evidence.reason_code,
            )
        if (
            evidence.observation_count < self._min_color_observations
            or evidence.duration_s + 1e-12 < self._min_color_duration_s
        ):
            return self._qwen_or_raise(
                candidate_id=candidate,
                timestamp_s=timestamp,
                target_spec=target_spec,
                detector_confidence=detection.confidence,
                qwen_verification=qwen_verification,
                qwen_mode=qwen_mode,
                acknowledge_vision_gate=acknowledge_vision_gate,
                reason="terminal_color_evidence_below_temporal_threshold",
                pending=True,
                allow_gate=False,
            )
        if qwen_verification is not None and qwen_mode == "gate":
            raise ValueError(
                "deterministic color evidence is terminal; Qwen gate must not be called"
            )
        return SemanticVerification(
            candidate_id=candidate,
            timestamp_s=timestamp,
            target_description=target_spec.description,
            matches=evidence.decision is AttributeDecision.MATCH,
            confidence=min(detection.confidence, evidence.confidence),
            verifier=DETERMINISTIC_ATTRIBUTE_VERIFIER,
        )

    @staticmethod
    def _validate_evidence_route(
        *,
        evidence: AttributeEvidence,
        candidate_id: str,
        timestamp_s: float,
        detection: TrackDetection,
        expected_color: str,
        mission_id: str | None,
        uav_id: str | None,
        assignment_id: str | None,
    ) -> None:
        if evidence.candidate_id != candidate_id:
            raise ValueError("attribute evidence candidate_id mismatch")
        tracker_id = str(detection.track_id)
        if evidence.tracker_id != tracker_id:
            raise ValueError("attribute evidence tracker_id mismatch")
        if evidence.timestamp_s != timestamp_s:
            raise ValueError(
                "attribute evidence must describe the current frame timestamp"
            )
        if (
            evidence.attribute_name != "color"
            or evidence.expected_value != expected_color
        ):
            raise ValueError(
                "attribute evidence does not match the target color requirement"
            )
        if mission_id is not None:
            assert uav_id is not None and assignment_id is not None
            if (
                evidence.mission_id != mission_id
                or evidence.uav_id != uav_id
                or evidence.assignment_id != assignment_id
            ):
                raise ValueError(
                    "attribute evidence mission/UAV/Assignment route mismatch"
                )

    @staticmethod
    def _qwen_or_raise(
        *,
        candidate_id: str,
        timestamp_s: float,
        target_spec: TargetSpec,
        detector_confidence: float,
        qwen_verification: SemanticVerification | None,
        qwen_mode: str,
        acknowledge_vision_gate: bool,
        reason: str,
        pending: bool = False,
        allow_gate: bool = True,
    ) -> SemanticVerification:
        if qwen_mode == "shadow":
            # Shadow evidence is deliberately ignored even when supplied.
            qwen_verification = None
        if not allow_gate or qwen_mode != "gate" or qwen_verification is None:
            exception = (
                AttributeSemanticVerificationPending
                if pending
                else AttributeSemanticVerificationRequiresQwen
            )
            raise exception(f"{exception.code}: {reason}")
        if not isinstance(qwen_verification, SemanticVerification):
            raise TypeError("qwen_verification must be SemanticVerification")
        if qwen_verification.candidate_id != candidate_id:
            raise ValueError(
                "stale Qwen result belongs to a different candidate epoch"
            )
        if qwen_verification.timestamp_s != timestamp_s:
            raise ValueError(
                "stale Qwen result does not match the current frame timestamp"
            )
        if qwen_verification.target_description != target_spec.description:
            raise ValueError("Qwen result target description mismatch")
        return SemanticVerification(
            candidate_id=candidate_id,
            timestamp_s=timestamp_s,
            target_description=target_spec.description,
            matches=qwen_verification.matches,
            confidence=min(detector_confidence, qwen_verification.confidence),
            verifier=DETERMINISTIC_ATTRIBUTE_VERIFIER,
        )


@dataclass(frozen=True, slots=True)
class AttributeSemanticProviderMetrics:
    observations_total: int = 0
    evidence_match: int = 0
    evidence_mismatch: int = 0
    evidence_pending: int = 0
    evidence_unsupported: int = 0
    semantic_match: int = 0
    semantic_mismatch: int = 0
    semantic_pending: int = 0
    qwen_fallback_required: int = 0
    tracker_epoch_resets: int = 0
    candidate_epoch_resets: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must equal 1")
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, int(value))

    def to_dict(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


class TemporalRgbdAttributeSemanticProvider:
    """Stateful adapter matching ``TargetPerceptionCoordinator``'s callback.

    The provider consumes only the candidate's current ``FrameRef`` and its
    matching tracker box.  It retains scalar observations/evidence, never
    pixels, and returns ``None`` while deterministic evidence is pending or
    needs the coordinator-owned Qwen gate.
    """

    def __init__(
        self,
        *,
        mission_id: str,
        uav_id: str,
        assignment_id: str,
        frame_store: FrameStore,
        color_verifier: RgbdColorAttributeVerifier,
        accumulator: TemporalColorEvidenceAccumulator,
        semantic_verifier: DeterministicAttributeSemanticVerifier,
        max_evidence_records: int = 256,
    ) -> None:
        self._mission_id = validate_mission_id(mission_id)
        self._uav_id = validate_uav_id(uav_id)
        self._assignment_id = validate_routing_id(
            assignment_id, "assignment_id"
        )
        if not isinstance(frame_store, FrameStore):
            raise TypeError("frame_store must be a FrameStore")
        if not isinstance(color_verifier, RgbdColorAttributeVerifier):
            raise TypeError("color_verifier must be RgbdColorAttributeVerifier")
        if color_verifier.frame_store is not frame_store:
            raise ValueError(
                "color_verifier must use the provider's exact FrameStore"
            )
        if not isinstance(accumulator, TemporalColorEvidenceAccumulator):
            raise TypeError("accumulator must be TemporalColorEvidenceAccumulator")
        if not isinstance(
            semantic_verifier, DeterministicAttributeSemanticVerifier
        ):
            raise TypeError(
                "semantic_verifier must be DeterministicAttributeSemanticVerifier"
            )
        if isinstance(max_evidence_records, bool) or not isinstance(
            max_evidence_records, Integral
        ):
            raise TypeError("max_evidence_records must be an integer")
        if int(max_evidence_records) <= 0:
            raise ValueError("max_evidence_records must be positive")
        self._frame_store = frame_store
        self._color_verifier = color_verifier
        self._accumulator = accumulator
        self._semantic_verifier = semantic_verifier
        self._records: deque[AttributeEvidence] = deque(
            maxlen=int(max_evidence_records)
        )
        self._epoch_by_candidate: dict[str, tuple[float, str]] = {}
        self._qwen_requirement_by_candidate: dict[str, str] = {}
        self._counts = {
            field: 0
            for field in AttributeSemanticProviderMetrics.__dataclass_fields__
            if field != "schema_version"
        }
        self._lock = RLock()
        self._accumulator.begin_mission(
            mission_id=self._mission_id,
            uav_id=self._uav_id,
        )

    @classmethod
    def from_target_perception_config(
        cls,
        config: object,
        *,
        mission_id: str,
        uav_id: str,
        assignment_id: str,
        frame_store: FrameStore,
        expected_class_name: str = "cube",
        expected_class_id: int | None = None,
        min_bbox_area_px: int = 64,
    ) -> "TemporalRgbdAttributeSemanticProvider":
        from configs.schema import TargetPerceptionConfig

        if not isinstance(config, TargetPerceptionConfig):
            raise TypeError("config must be a TargetPerceptionConfig")
        if config.backend != "ultralytics_service":
            raise ValueError("attribute semantic provider is production YOLO-only")
        if not config.attributes.enabled:
            raise ValueError("target attribute verification is disabled")
        if config.attributes.mode != "deterministic_then_qwen":
            raise ValueError(
                "attribute semantic provider requires deterministic_then_qwen mode"
            )
        if config.confirmation.mode != "class_track_attribute_or_qwen":
            raise ValueError(
                "attribute semantic provider requires "
                "class_track_attribute_or_qwen confirmation mode"
            )
        color_config = config.attributes.color
        return cls(
            mission_id=mission_id,
            uav_id=uav_id,
            assignment_id=assignment_id,
            frame_store=frame_store,
            color_verifier=RgbdColorAttributeVerifier.from_config(
                color_config,
                frame_store=frame_store,
                min_bbox_area_px=min_bbox_area_px,
            ),
            accumulator=TemporalColorEvidenceAccumulator.from_config(
                color_config
            ),
            semantic_verifier=(
                DeterministicAttributeSemanticVerifier.from_color_config(
                    color_config,
                    expected_class_name=expected_class_name,
                    expected_class_id=expected_class_id,
                )
            ),
        )

    def __call__(
        self,
        candidate: CandidateSnapshot,
        target_spec: TargetSpec,
        detection: TrackDetection,
        timestamp_s: float,
    ) -> SemanticVerification | None:
        if not isinstance(candidate, CandidateSnapshot):
            raise TypeError("candidate must be a CandidateSnapshot")
        if candidate.uav_id != self._uav_id:
            raise AttributeRouteMismatch("candidate belongs to another UAV")
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        if not isinstance(detection, TrackDetection):
            raise TypeError("detection must be a TrackDetection")
        timestamp = _timestamp(timestamp_s)
        if len(candidate.frame_history) != len(candidate.bbox_history):
            raise ValueError("candidate frame and bbox histories must be aligned")
        matches = [
            index
            for index, frame in enumerate(candidate.frame_history)
            if frame.timestamp_s == timestamp
        ]
        if not matches:
            raise AttributeRouteMismatch(
                "candidate has no FrameRef for semantic timestamp"
            )
        index = matches[-1]
        frame_ref = candidate.frame_history[index]
        candidate_bbox = candidate.bbox_history[index]
        if tuple(detection.bbox_xyxy_normalized) != tuple(candidate_bbox):
            raise AttributeRouteMismatch(
                "detection bbox does not match candidate FrameRef epoch"
            )

        tracker_id = str(detection.track_id)
        epoch = (candidate.first_seen_timestamp_s, tracker_id)
        with self._lock:
            previous_epoch = self._epoch_by_candidate.get(candidate.candidate_id)
            if previous_epoch is not None and previous_epoch != epoch:
                self._accumulator.reject_candidate(
                    mission_id=self._mission_id,
                    uav_id=self._uav_id,
                    assignment_id=self._assignment_id,
                    candidate_id=candidate.candidate_id,
                )
                if previous_epoch[0] != epoch[0]:
                    self._counts["candidate_epoch_resets"] += 1
                else:
                    self._counts["tracker_epoch_resets"] += 1
                    if candidate.candidate_id not in self._qwen_requirement_by_candidate:
                        self._counts["qwen_fallback_required"] += 1
                    self._qwen_requirement_by_candidate[
                        candidate.candidate_id
                    ] = "tracker_epoch_changed"
            self._epoch_by_candidate[candidate.candidate_id] = epoch

        try:
            attributes = _hard_attributes(target_spec)
        except AttributeSemanticVerificationRequiresQwen:
            with self._lock:
                self._counts["qwen_fallback_required"] += 1
                self._qwen_requirement_by_candidate[
                    candidate.candidate_id
                ] = "unsupported_target_semantics"
            return None
        expected_color = attributes.get("color")
        evidence: AttributeEvidence | None = None
        if expected_color is not None:
            requirement = AttributeRequirement(
                mission_id=self._mission_id,
                uav_id=self._uav_id,
                assignment_id=self._assignment_id,
                candidate_id=candidate.candidate_id,
                tracker_id=detection.track_id,
                attribute_name="color",
                expected_value=expected_color,
            )
            route = AttributeVerificationRoute(
                mission_id=self._mission_id,
                uav_id=self._uav_id,
                assignment_id=self._assignment_id,
                candidate_id=candidate.candidate_id,
                tracker_id=detection.track_id,
            )
            observation = self._color_verifier.verify(
                requirement=requirement,
                detection=detection,
                route=route,
                frame_ref=frame_ref,
            )
            evidence = self._accumulator.update(observation)
            with self._lock:
                self._records.append(evidence)
                self._counts["observations_total"] += 1
                self._counts[
                    {
                        AttributeDecision.MATCH: "evidence_match",
                        AttributeDecision.MISMATCH: "evidence_mismatch",
                        AttributeDecision.PENDING: "evidence_pending",
                        AttributeDecision.UNSUPPORTED: "evidence_unsupported",
                    }[evidence.decision]
                ] += 1

        try:
            semantic = self._semantic_verifier.verify(
                candidate_id=candidate.candidate_id,
                timestamp_s=timestamp,
                target_spec=target_spec,
                detection=detection,
                attribute_evidence=evidence,
                mission_id=self._mission_id if evidence is not None else None,
                uav_id=self._uav_id if evidence is not None else None,
                assignment_id=(
                    self._assignment_id if evidence is not None else None
                ),
            )
        except AttributeSemanticVerificationPending:
            with self._lock:
                self._counts["semantic_pending"] += 1
                if (
                    evidence is not None
                    and self._semantic_verifier.qwen_fallback_eligible(evidence)
                ):
                    if candidate.candidate_id not in self._qwen_requirement_by_candidate:
                        self._counts["qwen_fallback_required"] += 1
                    self._qwen_requirement_by_candidate[
                        candidate.candidate_id
                    ] = "persistent_attribute_pending"
            return None
        except AttributeSemanticVerificationRequiresQwen:
            with self._lock:
                self._counts["qwen_fallback_required"] += 1
                self._qwen_requirement_by_candidate[
                    candidate.candidate_id
                ] = "unsupported_attribute_or_semantics"
            return None
        with self._lock:
            key = "semantic_match" if semantic.matches else "semantic_mismatch"
            self._counts[key] += 1
            self._qwen_requirement_by_candidate.pop(candidate.candidate_id, None)
        return semantic

    @property
    def metrics(self) -> AttributeSemanticProviderMetrics:
        with self._lock:
            return AttributeSemanticProviderMetrics(**self._counts)

    def evidence_records(self) -> tuple[AttributeEvidence, ...]:
        with self._lock:
            return tuple(self._records)

    def drain_evidence_records(self) -> tuple[AttributeEvidence, ...]:
        """Return and clear the bounded scalar evidence queue for logging."""

        with self._lock:
            records = tuple(self._records)
            self._records.clear()
            return records

    def requires_qwen(self, candidate_id: str) -> bool:
        normalized = validate_routing_id(candidate_id, "candidate_id")
        with self._lock:
            return normalized in self._qwen_requirement_by_candidate

    def qwen_requirement_reason(self, candidate_id: str) -> str | None:
        normalized = validate_routing_id(candidate_id, "candidate_id")
        with self._lock:
            return self._qwen_requirement_by_candidate.get(normalized)

    def clear_qwen_requirement(self, candidate_id: str) -> None:
        normalized = validate_routing_id(candidate_id, "candidate_id")
        with self._lock:
            self._qwen_requirement_by_candidate.pop(normalized, None)

    def reset_candidate(self, candidate_id: str) -> None:
        normalized = validate_routing_id(candidate_id, "candidate_id")
        with self._lock:
            self._accumulator.reject_candidate(
                mission_id=self._mission_id,
                uav_id=self._uav_id,
                assignment_id=self._assignment_id,
                candidate_id=normalized,
            )
            self._epoch_by_candidate.pop(normalized, None)
            self._qwen_requirement_by_candidate.pop(normalized, None)

    def reject_candidate(self, candidate_id: str) -> None:
        self.reset_candidate(candidate_id)

    def reset_mission(
        self,
        *,
        mission_id: str,
        assignment_id: str | None = None,
    ) -> None:
        normalized_mission = validate_mission_id(mission_id)
        normalized_assignment = (
            self._assignment_id
            if assignment_id is None
            else validate_routing_id(assignment_id, "assignment_id")
        )
        with self._lock:
            self._mission_id = normalized_mission
            self._assignment_id = normalized_assignment
            self._accumulator.begin_mission(
                mission_id=normalized_mission,
                uav_id=self._uav_id,
            )
            self._epoch_by_candidate.clear()
            self._qwen_requirement_by_candidate.clear()
            self._records.clear()
            for key in self._counts:
                self._counts[key] = 0

    def reset(
        self,
        *,
        mission_id: str,
        uav_id: str,
        assignment_id: str,
    ) -> None:
        normalized_uav = validate_uav_id(uav_id)
        if normalized_uav != self._uav_id:
            raise AttributeRouteMismatch(
                "cannot reset attribute provider for a different UAV"
            )
        self.reset_mission(
            mission_id=mission_id,
            assignment_id=assignment_id,
        )

__all__ = [
    "ATTRIBUTE_SEMANTIC_PENDING",
    "ATTRIBUTE_SEMANTIC_REQUIRES_QWEN",
    "AttributeSemanticVerificationPending",
    "AttributeSemanticVerificationRequiresQwen",
    "AttributeSemanticProviderMetrics",
    "DETERMINISTIC_ATTRIBUTE_VERIFIER",
    "DeterministicAttributeSemanticVerifier",
    "TemporalRgbdAttributeSemanticProvider",
]
