"""Future visual ReID and temporal-consistency backend."""

from __future__ import annotations

from perception.types import IdentityConsistencyEvidence
from skills.types import Observation


class ReIDVerifier:
    """Honest placeholder which never fabricates identity evidence."""

    def verify_identity(
        self,
        candidate_id: str,
        observation: Observation,
    ) -> IdentityConsistencyEvidence:
        del candidate_id, observation
        raise NotImplementedError(
            "ReIDVerifier is not implemented; no temporal identity evidence "
            "is available"
        )


__all__ = ["ReIDVerifier"]
