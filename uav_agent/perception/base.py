"""Isaac-independent structural interface for perception backends.

The interface is intentionally small at this stage: a runtime frame goes in
and the shared :class:`~skills.types.Observation` consumed by Skills comes
out.  Detector boxes, masks, identities, and model-specific outputs do not
belong in this boundary yet.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from perception.types import (
    DetectionCandidate,
    IdentityConsistencyEvidence,
    SemanticVerification,
    ShortTrackEvidence,
)
from skills.types import Observation


@runtime_checkable
class PerceptionBackend(Protocol):
    """Convert one synchronized runtime frame into an ``Observation``."""

    def observe(self, frame: object) -> Observation:
        """Return the observation represented by ``frame``."""


@runtime_checkable
class DetectorTrackerBackend(Protocol):
    """Learned proposal and short-track boundary; no ground truth allowed."""

    def detect(self, observation: Observation) -> DetectionCandidate | None: ...

    def update_track(
        self,
        candidate_id: str,
        observation: Observation,
    ) -> ShortTrackEvidence: ...


@runtime_checkable
class SemanticVerifierBackend(Protocol):
    """Verify that one candidate matches the requested semantics."""

    def verify_candidate(
        self,
        candidate: DetectionCandidate,
        target_description: str,
        camera_rgb: object,
    ) -> SemanticVerification: ...


@runtime_checkable
class IdentityVerifierBackend(Protocol):
    """Provide ReID plus temporal identity-consistency evidence."""

    def verify_identity(
        self,
        candidate_id: str,
        observation: Observation,
    ) -> IdentityConsistencyEvidence: ...


__all__ = [
    "DetectorTrackerBackend",
    "IdentityVerifierBackend",
    "PerceptionBackend",
    "SemanticVerifierBackend",
]
