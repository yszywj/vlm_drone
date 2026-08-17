"""Deterministic, model-free planner used by the B0 baseline."""

from __future__ import annotations

from planner.base import MissionPlanner
from planner.diagnostics import PlannerDiagnostics, PlannerExecution
from planner.schemas import MissionIntent, PlannerRequest


class ScriptedPlanner(MissionPlanner):
    """Return a fresh copy of one constructor-supplied MissionIntent."""

    source = "scripted"

    def __init__(self, intent: MissionIntent) -> None:
        if not isinstance(intent, MissionIntent):
            raise TypeError("intent must be a MissionIntent")
        self._intent_data = intent.to_dict()

    def plan(self, request: PlannerRequest) -> MissionIntent:
        return self.plan_with_diagnostics(request).output

    def plan_with_diagnostics(self, request: PlannerRequest) -> PlannerExecution:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        output = MissionIntent.from_dict(self._intent_data)
        return PlannerExecution(output=output, diagnostics=_scripted_diagnostics())


def _scripted_diagnostics() -> PlannerDiagnostics:
    return PlannerDiagnostics(
        model_calls=0,
        repair_used=False,
        repair_succeeded=False,
        initial_output_valid=True,
        final_output_valid=True,
        initial_error_code=None,
        initial_error_message=None,
        structured_output_enabled=False,
    )
