"""Deterministic, model-free planner used by the B0 baseline."""

from __future__ import annotations

from planner.base import MissionPlanner
from planner.schemas import MissionIntent, PlannerRequest


class ScriptedPlanner(MissionPlanner):
    """Return a fresh copy of one constructor-supplied MissionIntent."""

    source = "scripted"

    def __init__(self, intent: MissionIntent) -> None:
        if not isinstance(intent, MissionIntent):
            raise TypeError("intent must be a MissionIntent")
        self._intent_data = intent.to_dict()

    def plan(self, request: PlannerRequest) -> MissionIntent:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        return MissionIntent.from_dict(self._intent_data)
