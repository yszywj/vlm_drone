"""Deterministic coordinator for multi-stage visual target confirmation.

No model is implemented here.  Detector/tracker, VLM, and ReID components
must supply their own typed evidence.  The coordinator only validates that
evidence and advances ``TargetManager`` through CANDIDATE and LOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Integral, Real

from perception.types import (
    DetectionCandidate,
    IdentityConsistencyEvidence,
    SemanticVerification,
    ShortTrackEvidence,
)
from target import TargetLifecycle, TargetManager


class CandidateConfirmationError(RuntimeError):
    """Raised for inconsistent evidence or an invalid target lifecycle."""


class ConfirmationDecision(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


def _threshold(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return normalized


@dataclass(frozen=True, slots=True)
class ConfirmationPolicy:
    """Minimum evidence needed before a visual target can be locked."""

    min_track_observations: int = 3
    min_track_duration_s: float = 0.5
    min_track_confidence: float = 0.5
    min_semantic_confidence: float = 0.5
    min_identity_confidence: float = 0.5
    min_consistent_observations: int = 3

    def __post_init__(self) -> None:
        for name in ("min_track_observations", "min_consistent_observations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"{name} must be greater than zero")
            object.__setattr__(self, name, int(value))
        duration = self.min_track_duration_s
        if isinstance(duration, bool) or not isinstance(duration, Real):
            raise TypeError("min_track_duration_s must be a finite number")
        duration = float(duration)
        if not isfinite(duration) or duration < 0.0:
            raise ValueError("min_track_duration_s must be finite and non-negative")
        object.__setattr__(self, "min_track_duration_s", duration)
        for name in (
            "min_track_confidence",
            "min_semantic_confidence",
            "min_identity_confidence",
        ):
            object.__setattr__(self, name, _threshold(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    decision: ConfirmationDecision
    candidate_id: str
    target_id: str | None
    confidence: float | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ConfirmationDecision):
            raise TypeError("decision must be a ConfirmationDecision")
        for name in ("candidate_id", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if self.target_id is not None:
            if not isinstance(self.target_id, str) or not self.target_id.strip():
                raise ValueError("target_id must be a non-empty string or None")
            object.__setattr__(self, "target_id", self.target_id.strip())
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                _threshold(self.confidence, "confidence"),
            )


class CandidateConfirmationCoordinator:
    """Advance SEARCHING -> CANDIDATE -> LOCKED from typed visual evidence."""

    def __init__(self, policy: ConfirmationPolicy | None = None) -> None:
        self._policy = policy or ConfirmationPolicy()
        if not isinstance(self._policy, ConfirmationPolicy):
            raise TypeError("policy must be a ConfirmationPolicy")

    @property
    def policy(self) -> ConfirmationPolicy:
        return self._policy

    def register_candidate(
        self,
        candidate: DetectionCandidate,
        target_manager: TargetManager,
    ) -> ConfirmationResult:
        if not isinstance(candidate, DetectionCandidate):
            raise TypeError("candidate must be a DetectionCandidate")
        if not isinstance(target_manager, TargetManager):
            raise TypeError("target_manager must be a TargetManager")
        if target_manager.lifecycle not in {
            TargetLifecycle.SEARCHING,
            TargetLifecycle.REACQUIRING,
        }:
            raise CandidateConfirmationError(
                "candidate registration requires TargetManager SEARCHING or "
                "REACQUIRING"
            )
        target_manager.set_candidate(
            candidate.candidate_id,
            timestamp_s=candidate.timestamp_s,
            confidence=candidate.confidence,
            source=candidate.source,
            last_seen_position=candidate.estimated_position,
            last_seen_velocity=candidate.estimated_velocity,
        )
        return ConfirmationResult(
            ConfirmationDecision.PENDING,
            candidate.candidate_id,
            None,
            candidate.confidence,
            "candidate_registered",
        )

    def evaluate(
        self,
        *,
        target_manager: TargetManager,
        track: ShortTrackEvidence,
        semantic: SemanticVerification,
        identity: IdentityConsistencyEvidence,
    ) -> ConfirmationResult:
        """Validate one complete evidence bundle and update the target state.

        A negative boolean result rejects the candidate.  Positive but
        insufficient-duration/confidence evidence remains PENDING.  Only a
        bundle satisfying every policy threshold becomes LOCKED.
        """

        if not isinstance(target_manager, TargetManager):
            raise TypeError("target_manager must be a TargetManager")
        if not isinstance(track, ShortTrackEvidence):
            raise TypeError("track must be ShortTrackEvidence")
        if not isinstance(semantic, SemanticVerification):
            raise TypeError("semantic must be SemanticVerification")
        if not isinstance(identity, IdentityConsistencyEvidence):
            raise TypeError("identity must be IdentityConsistencyEvidence")
        snapshot = target_manager.snapshot()
        if snapshot.lifecycle is not TargetLifecycle.CANDIDATE:
            raise CandidateConfirmationError(
                "confirmation requires TargetManager CANDIDATE"
            )
        candidate_id = snapshot.target_id
        if candidate_id is None:  # defensive against a corrupt manager
            raise CandidateConfirmationError("CANDIDATE has no candidate_id")
        evidence_ids = {track.candidate_id, semantic.candidate_id, identity.candidate_id}
        if evidence_ids != {candidate_id}:
            raise CandidateConfirmationError(
                "track, semantic, and identity evidence must match the active candidate"
            )
        if semantic.target_description != snapshot.description:
            raise CandidateConfirmationError(
                "semantic evidence target_description does not match the mission target"
            )
        candidate_time = snapshot.last_seen_time_s
        if candidate_time is None:
            raise CandidateConfirmationError("CANDIDATE has no detection timestamp")
        evidence_times = (track.timestamp_s, semantic.timestamp_s, identity.timestamp_s)
        if any(timestamp < candidate_time for timestamp in evidence_times):
            raise CandidateConfirmationError(
                "confirmation evidence cannot predate the candidate detection"
            )
        if not (
            track.timestamp_s <= semantic.timestamp_s <= identity.timestamp_s
        ):
            raise CandidateConfirmationError(
                "confirmation evidence must be ordered detector -> short track "
                "-> semantic verification -> identity verification"
            )
        available_track_time = track.timestamp_s - candidate_time
        if track.duration_s > available_track_time + 1e-12:
            raise CandidateConfirmationError(
                "short-track duration cannot exceed elapsed time since detection"
            )
        if identity.consistent_observations > track.observation_count:
            raise CandidateConfirmationError(
                "identity consistent_observations cannot exceed short-track "
                "observation_count"
            )
        result_time = max(evidence_times)

        rejection_reason: str | None = None
        if not track.stable:
            rejection_reason = "short_track_unstable"
        elif not semantic.matches:
            rejection_reason = "semantic_mismatch"
        elif not identity.reidentified:
            rejection_reason = "reid_failed"
        elif not identity.temporally_consistent:
            rejection_reason = "temporal_identity_inconsistent"
        if rejection_reason is not None:
            target_manager.reject_candidate(
                timestamp_s=result_time,
                reason=rejection_reason,
            )
            return ConfirmationResult(
                ConfirmationDecision.REJECTED,
                candidate_id,
                None,
                None,
                rejection_reason,
            )

        policy = self._policy
        shortfalls: list[str] = []
        if track.observation_count < policy.min_track_observations:
            shortfalls.append("track_observation_count")
        if track.duration_s < policy.min_track_duration_s:
            shortfalls.append("track_duration")
        if track.confidence < policy.min_track_confidence:
            shortfalls.append("track_confidence")
        if semantic.confidence < policy.min_semantic_confidence:
            shortfalls.append("semantic_confidence")
        if identity.confidence < policy.min_identity_confidence:
            shortfalls.append("identity_confidence")
        if identity.consistent_observations < policy.min_consistent_observations:
            shortfalls.append("consistent_observations")
        confidence = min(track.confidence, semantic.confidence, identity.confidence)
        if shortfalls:
            return ConfirmationResult(
                ConfirmationDecision.PENDING,
                candidate_id,
                None,
                confidence,
                "insufficient_evidence:" + ",".join(shortfalls),
            )

        target_manager._lock_confirmed_candidate(
            identity.target_id,
            timestamp_s=result_time,
            confidence=confidence,
            source="confirmed_vision",
        )
        return ConfirmationResult(
            ConfirmationDecision.CONFIRMED,
            candidate_id,
            identity.target_id,
            confidence,
            "track_semantic_reid_confirmed",
        )


__all__ = [
    "CandidateConfirmationCoordinator",
    "CandidateConfirmationError",
    "ConfirmationDecision",
    "ConfirmationPolicy",
    "ConfirmationResult",
]
