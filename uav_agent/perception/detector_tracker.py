"""Future detector/tracker perception backend.

No detector or tracker model is loaded in Stage 0.  Keeping this class as an
explicitly failing backend prevents callers from mistaking placeholder data
for a real visual observation.
"""

from __future__ import annotations

from perception.runtime import PerceptionCapability
from perception.types import DetectionCandidate, ShortTrackEvidence
from skills.types import Observation


class DetectorTrackerPerception:
    """Placeholder for a learned detector plus temporal target tracker."""

    capability = PerceptionCapability.VISION

    def observe(self, frame: object) -> Observation:
        """Reject use until a real detector/tracker backend is implemented."""

        del frame
        raise NotImplementedError(
            "DetectorTrackerPerception is not implemented; no detector or "
            "tracker result is available"
        )

    def detect(self, observation: Observation) -> DetectionCandidate | None:
        del observation
        raise NotImplementedError(
            "DetectorTrackerPerception is not implemented; no candidate is available"
        )

    def update_track(
        self,
        candidate_id: str,
        observation: Observation,
    ) -> ShortTrackEvidence:
        del candidate_id, observation
        raise NotImplementedError(
            "DetectorTrackerPerception is not implemented; no short-track "
            "evidence is available"
        )


__all__ = ["DetectorTrackerPerception"]
