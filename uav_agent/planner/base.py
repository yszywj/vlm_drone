"""Abstract interface for high-level mission planners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from planner.schemas import PlannerOutput, PlannerRequest


class PlannerError(RuntimeError):
    """Base class for mission-planning failures."""


class PlannerOutputError(PlannerError):
    """Raised when a planner produces invalid high-level JSON output."""


class MissionPlanner(ABC):
    """Convert natural language into a high-level semantic plan.

    A planner may return the legacy ``MissionIntent`` or a dynamic
    ``SkillPlanDraft``/routed ``SkillPlanDraftV2``.  It must never return world-coordinate-resolved
    ``SkillGoal`` or ``TaskPlan`` values; those belong to trusted runtime code.
    """

    source: str

    @abstractmethod
    def plan(self, request: PlannerRequest) -> PlannerOutput:
        """Return a high-level planner output; execution belongs to runtime."""
