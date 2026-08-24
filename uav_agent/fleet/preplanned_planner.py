"""Strict replay boundary for a Fleet plan prepared before Isaac starts."""

from __future__ import annotations

from copy import deepcopy

from fleet.planner_base import FleetPlanner
from fleet.schemas import validate_fleet_mission_plan
from fleet.types import FleetMissionPlan, FleetMissionRequest


class RoutedPreplannedFleetPlanner(FleetPlanner):
    """Replay one validated plan only for its exact original trusted request.

    The standalone Fleet runner completes every model call before importing
    Isaac.  :class:`fleet.runtime.FleetMissionRuntime` still owns its normal
    planning boundary, so this adapter verifies that none of the request
    inventory, target requirements, coordination limits, routing IDs, or user
    text changed between the two phases.
    """

    def __init__(
        self,
        request: FleetMissionRequest,
        plan: FleetMissionPlan,
        *,
        source: str,
    ) -> None:
        if not isinstance(request, FleetMissionRequest):
            raise TypeError("request must be a FleetMissionRequest")
        if not isinstance(plan, FleetMissionPlan):
            raise TypeError("plan must be a FleetMissionPlan")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        self._request_payload = deepcopy(request.to_dict())
        self._plan_payload = deepcopy(
            validate_fleet_mission_plan(plan, request).to_dict()
        )
        self.source = source.strip()

    def plan(self, request: FleetMissionRequest) -> FleetMissionPlan:
        if not isinstance(request, FleetMissionRequest):
            raise TypeError("request must be a FleetMissionRequest")
        if request.to_dict() != self._request_payload:
            raise ValueError(
                "runtime Fleet request differs from the preplanned trusted request"
            )
        # Parsing a fresh immutable object prevents a runtime consumer from
        # observing or retaining the constructor's original object identity.
        return FleetMissionPlan.from_dict(
            deepcopy(self._plan_payload),
            request=request,
        )


__all__ = ["RoutedPreplannedFleetPlanner"]
