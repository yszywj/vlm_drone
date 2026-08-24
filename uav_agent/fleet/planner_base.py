"""Abstract high-level Fleet Planner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fleet.types import FleetMissionPlan, FleetMissionRequest


class FleetPlannerError(RuntimeError):
    """Base error for fleet mission decomposition."""


class FleetPlannerOutputError(FleetPlannerError):
    """Raised when a planner returns an invalid FleetMissionPlan."""


class FleetPlanner(ABC):
    """Produce assignments and coordination policy, never local Skill steps."""

    source: str

    @abstractmethod
    def plan(self, request: FleetMissionRequest) -> FleetMissionPlan:
        """Return a high-level fleet assignment plan."""


__all__ = ["FleetPlanner", "FleetPlannerError", "FleetPlannerOutputError"]
