"""Abstract interface for high-level mission planners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from planner.schemas import MissionIntent, PlannerRequest


class PlannerError(RuntimeError):
    """Base class for mission-planning failures."""


class PlannerOutputError(PlannerError):
    """Raised when a planner produces an invalid high-level intent."""


class MissionPlanner(ABC):
    """Convert a natural-language request into a high-level mission intent."""

    @abstractmethod
    def plan(self, request: PlannerRequest) -> MissionIntent:
        """Return an intent only; Skill execution belongs to the runtime."""

