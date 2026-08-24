"""Visual grounding and trusted candidate-position resolution boundaries.

No learned detector is implemented in this module.  In particular there is
no YOLOE fallback hidden behind the protocol.  Production position grounding
fails explicitly until a real geometry backend is supplied.  The temporary
Oracle resolver requires the existing two-part ORACLE_EVALUATION opt-in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np

from common.ids import validate_routing_id, validate_uav_id
from perception.candidate_bank import CandidateSnapshot
from perception.prompt_types import PromptBundle
from perception.runtime import (
    PerceptionBoundaryError,
    PerceptionRuntimeProfile,
)
from runtime.frame_store import FrameRef
from target.types import TargetSpec
from yolo_service.protocol import TrackResponse


class CandidateResolutionUnavailable(RuntimeError):
    """Raised when no trusted candidate-to-world resolver is available."""


def _finite_position(value: object) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(
        value,
        (Sequence, np.ndarray),
    ):
        raise ValueError("candidate position must contain three finite numbers")
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(
            "candidate position must contain three finite numbers"
        ) from None
    if len(items) != 3:
        raise ValueError("candidate position must contain three finite numbers")
    normalized: list[float] = []
    for component in items:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise TypeError("candidate position must contain finite numbers")
        number = float(component)
        if not isfinite(number):
            raise ValueError("candidate position must contain finite numbers")
        normalized.append(number)
    return normalized[0], normalized[1], normalized[2]


def _bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("bbox must contain four normalized coordinates")
    result: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise TypeError("bbox components must be finite numbers")
        number = float(component)
        if not isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("bbox components must be within [0, 1]")
        result.append(number)
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
    return result[0], result[1], result[2], result[3]


@dataclass(frozen=True, slots=True)
class GroundingProposal:
    """One image-space proposal; never a trusted flight coordinate."""

    uav_id: str
    candidate_id: str
    frame_ref: FrameRef
    bbox_xyxy_normalized: tuple[float, float, float, float]
    source: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "candidate_id",
            validate_routing_id(self.candidate_id, "candidate_id"),
        )
        if not isinstance(self.frame_ref, FrameRef):
            raise TypeError("frame_ref must be a FrameRef")
        if self.frame_ref.uav_id != self.uav_id:
            raise ValueError("frame_ref uav_id does not match proposal uav_id")
        object.__setattr__(
            self,
            "bbox_xyxy_normalized",
            _bbox(self.bbox_xyxy_normalized),
        )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        object.__setattr__(self, "source", self.source.strip())
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence,
                Real,
            ):
                raise TypeError("confidence must be a finite number or None")
            confidence = float(self.confidence)
            if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "uav_id": self.uav_id,
            "candidate_id": self.candidate_id,
            "frame_ref": self.frame_ref.to_dict(),
            "bbox_xyxy_normalized": list(self.bbox_xyxy_normalized),
            "source": self.source,
            "confidence": self.confidence,
        }


@runtime_checkable
class GroundingBackend(Protocol):
    """Future detector/Qwen/learned-grounder boundary.

    Implementations may inspect an ephemeral RGB array, but must return only
    bounded image-space proposals.  They cannot return flight coordinates or
    invoke a UAV controller.  Qwen implementations are scheduled externally
    by ``ReviewScheduler`` and therefore remain low-frequency.
    """

    def propose(
        self,
        *,
        uav_id: str,
        frame_ref: FrameRef,
        rgb: np.ndarray,
        target_spec: TargetSpec,
        prompt_bundle: PromptBundle,
    ) -> Sequence[GroundingProposal]: ...


class GroundingBackendUnavailable(NotImplementedError):
    """Raised by named future slots instead of returning fabricated boxes."""


class _UnimplementedGrounder:
    backend_name = "grounding backend"

    def propose(
        self,
        *,
        uav_id: str,
        frame_ref: FrameRef,
        rgb: np.ndarray,
        target_spec: TargetSpec,
        prompt_bundle: PromptBundle,
    ) -> Sequence[GroundingProposal]:
        del uav_id, frame_ref, rgb, target_spec, prompt_bundle
        raise GroundingBackendUnavailable(
            f"{self.backend_name} is an explicit future integration slot"
        )


class OracleEvaluationGrounder(_UnimplementedGrounder):
    """Privileged proposal slot; no evaluator projection is fabricated here."""

    backend_name = "OracleEvaluationGrounder"

    def __init__(
        self,
        *,
        profile: PerceptionRuntimeProfile,
        acknowledge_privileged_oracle: bool,
    ) -> None:
        if not isinstance(profile, PerceptionRuntimeProfile):
            raise TypeError("profile must be a PerceptionRuntimeProfile")
        if not isinstance(acknowledge_privileged_oracle, bool):
            raise TypeError("acknowledge_privileged_oracle must be bool")
        if profile is not PerceptionRuntimeProfile.ORACLE_EVALUATION:
            raise PerceptionBoundaryError(
                "OracleEvaluationGrounder is forbidden in PRODUCTION"
            )
        if not acknowledge_privileged_oracle:
            raise PerceptionBoundaryError(
                "OracleEvaluationGrounder requires explicit Oracle acknowledgement"
            )


class QwenVLGrounder(_UnimplementedGrounder):
    """Low-frequency semantic proposal slot, scheduled outside Skill code."""

    backend_name = "QwenVLGrounder"


class LearnedGrounder(_UnimplementedGrounder):
    """Future learned image-space grounding slot."""

    backend_name = "LearnedGrounder"


class UltralyticsGrounder(_UnimplementedGrounder):
    """Pure adapter from a strict YOLO service response to image proposals.

    Network scheduling remains in the target-perception coordinator.  Keeping
    response parsing here prevents service-specific tensors or response
    objects from leaking into Skills and preserves the existing
    ``GroundingBackend`` boundary.
    """

    backend_name = "UltralyticsGrounder response provider"

    @classmethod
    def from_response(
        cls,
        response: TrackResponse,
        frame_ref: FrameRef,
    ) -> tuple[GroundingProposal, ...]:
        if not isinstance(response, TrackResponse):
            raise TypeError("response must be a TrackResponse")
        if not isinstance(frame_ref, FrameRef):
            raise TypeError("frame_ref must be a FrameRef")
        mismatched: list[str] = []
        if response.uav_id != frame_ref.uav_id:
            mismatched.append("uav_id")
        if response.frame_id != frame_ref.frame_id:
            mismatched.append("frame_id")
        if abs(response.timestamp_s - frame_ref.timestamp_s) > 1e-9:
            mismatched.append("timestamp_s")
        if mismatched:
            raise ValueError(
                "YOLO response does not match FrameRef fields: "
                + ", ".join(mismatched)
            )
        track_ids = [item.track_id for item in response.detections]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("YOLO response contains duplicate track_id values")
        return tuple(
            GroundingProposal(
                uav_id=response.uav_id,
                candidate_id=_ultralytics_candidate_id(
                    response.mission_id,
                    response.uav_id,
                    detection.track_id,
                ),
                frame_ref=frame_ref,
                bbox_xyxy_normalized=detection.bbox_xyxy_normalized,
                source="ultralytics_service",
                confidence=detection.confidence,
            )
            for detection in response.detections
        )


def _ultralytics_candidate_id(
    mission_id: str,
    uav_id: str,
    track_id: int,
) -> str:
    """Return a stable, routing-safe candidate ID for one mission stream."""

    readable = f"{mission_id}_{uav_id}_track_{track_id}"
    try:
        return validate_routing_id(readable, "candidate_id")
    except ValueError:
        # Input IDs are individually bounded but their concatenation can exceed
        # the 64-character routing contract.  A stable digest retains stream
        # isolation without silently truncating two IDs to the same prefix.
        digest = sha256(
            f"{mission_id}:{uav_id}:track:{track_id}".encode("utf-8")
        ).hexdigest()[:20]
        return validate_routing_id(f"track_{track_id}_{digest}", "candidate_id")


class YOLOEGrounder(UltralyticsGrounder):
    """Compatibility name for the shared YOLO/YOLOE response adapter."""

    backend_name = "YOLOEGrounder response provider"


@dataclass(frozen=True, slots=True)
class ResolvedCandidatePosition:
    """Trusted internal geometry returned to INSPECT, never to Qwen."""

    uav_id: str
    candidate_id: str
    position_xyz_m: tuple[float, float, float]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "candidate_id",
            validate_routing_id(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "position_xyz_m",
            _finite_position(self.position_xyz_m),
        )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        object.__setattr__(self, "source", self.source.strip())


@runtime_checkable
class CandidateResolver(Protocol):
    """Resolve a candidate to trusted internal geometry for a motion Skill."""

    @property
    def profile(self) -> PerceptionRuntimeProfile: ...

    def resolve(
        self,
        candidate: CandidateSnapshot,
        *,
        timestamp_s: float,
    ) -> ResolvedCandidatePosition: ...


OraclePositionProvider = Callable[
    [str, str, float],
    Sequence[float] | np.ndarray,
]


class OracleEvaluationCandidateResolver:
    """Privileged evaluator-only candidate geometry adapter.

    Construction requires both the explicit ``ORACLE_EVALUATION`` profile and
    a positive acknowledgement.  Merely passing an Oracle provider cannot
    enable it in production.
    """

    def __init__(
        self,
        position_provider: OraclePositionProvider,
        *,
        profile: PerceptionRuntimeProfile,
        acknowledge_privileged_oracle: bool,
    ) -> None:
        if not callable(position_provider):
            raise TypeError("position_provider must be callable")
        if not isinstance(profile, PerceptionRuntimeProfile):
            raise TypeError("profile must be a PerceptionRuntimeProfile")
        if not isinstance(acknowledge_privileged_oracle, bool):
            raise TypeError("acknowledge_privileged_oracle must be a bool")
        if profile is not PerceptionRuntimeProfile.ORACLE_EVALUATION:
            raise PerceptionBoundaryError(
                "Oracle candidate resolution is forbidden in PRODUCTION"
            )
        if not acknowledge_privileged_oracle:
            raise PerceptionBoundaryError(
                "ORACLE_EVALUATION candidate resolution requires explicit "
                "acknowledge_privileged_oracle=True"
            )
        self._position_provider = position_provider
        self._profile = profile

    @property
    def profile(self) -> PerceptionRuntimeProfile:
        return self._profile

    def resolve(
        self,
        candidate: CandidateSnapshot,
        *,
        timestamp_s: float,
    ) -> ResolvedCandidatePosition:
        if not isinstance(candidate, CandidateSnapshot):
            raise TypeError("candidate must be a CandidateSnapshot")
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, Real):
            raise TypeError("timestamp_s must be a finite non-negative number")
        normalized_time = float(timestamp_s)
        if not isfinite(normalized_time) or normalized_time < 0.0:
            raise ValueError("timestamp_s must be a finite non-negative number")
        position = self._position_provider(
            candidate.uav_id,
            candidate.candidate_id,
            normalized_time,
        )
        return ResolvedCandidatePosition(
            uav_id=candidate.uav_id,
            candidate_id=candidate.candidate_id,
            position_xyz_m=_finite_position(position),
            source="oracle_evaluation",
        )


class ProductionCandidateResolver:
    """Explicit production placeholder; no detector geometry is fabricated."""

    @property
    def profile(self) -> PerceptionRuntimeProfile:
        return PerceptionRuntimeProfile.PRODUCTION

    def resolve(
        self,
        candidate: CandidateSnapshot,
        *,
        timestamp_s: float,
    ) -> ResolvedCandidatePosition:
        if not isinstance(candidate, CandidateSnapshot):
            raise TypeError("candidate must be a CandidateSnapshot")
        del timestamp_s
        raise CandidateResolutionUnavailable(
            "production candidate position resolution is not implemented; "
            "configure a real learned geometry backend"
        )


__all__ = [
    "CandidateResolutionUnavailable",
    "CandidateResolver",
    "GroundingBackend",
    "GroundingBackendUnavailable",
    "GroundingProposal",
    "LearnedGrounder",
    "OracleEvaluationGrounder",
    "OracleEvaluationCandidateResolver",
    "OraclePositionProvider",
    "ProductionCandidateResolver",
    "QwenVLGrounder",
    "ResolvedCandidatePosition",
    "UltralyticsGrounder",
    "YOLOEGrounder",
]
