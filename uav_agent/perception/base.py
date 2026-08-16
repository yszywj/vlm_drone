"""Isaac-independent structural interface for perception backends.

The interface is intentionally small at this stage: a runtime frame goes in
and the shared :class:`~skills.types.Observation` consumed by Skills comes
out.  Detector boxes, masks, identities, and model-specific outputs do not
belong in this boundary yet.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from skills.types import Observation


@runtime_checkable
class PerceptionBackend(Protocol):
    """Convert one synchronized runtime frame into an ``Observation``."""

    def observe(self, frame: object) -> Observation:
        """Return the observation represented by ``frame``."""


__all__ = ["PerceptionBackend"]
