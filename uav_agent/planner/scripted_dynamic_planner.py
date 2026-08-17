"""Deterministic model-free planner for dynamic runtime tests and demos."""

from __future__ import annotations

from planner.base import MissionPlanner
from planner.diagnostics import PlannerDiagnostics, PlannerExecution
from planner.schemas import PlannerRequest, SkillPlanDraft


class ScriptedDynamicPlanner(MissionPlanner):
    """Return a fresh copy of one constructor-supplied SkillPlanDraft."""

    source = "dynamic_scripted"

    def __init__(self, skill_plan_draft: SkillPlanDraft) -> None:
        if not isinstance(skill_plan_draft, SkillPlanDraft):
            raise TypeError("skill_plan_draft must be a SkillPlanDraft")
        self._draft_data = skill_plan_draft.to_dict()

    def plan(self, request: PlannerRequest) -> SkillPlanDraft:
        return self.plan_with_diagnostics(request).output

    def plan_with_diagnostics(self, request: PlannerRequest) -> PlannerExecution:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        output = SkillPlanDraft.from_dict(self._draft_data)
        diagnostics = PlannerDiagnostics(
            model_calls=0,
            repair_used=False,
            repair_succeeded=False,
            initial_output_valid=True,
            final_output_valid=True,
            initial_error_code=None,
            initial_error_message=None,
            structured_output_enabled=False,
        )
        return PlannerExecution(output=output, diagnostics=diagnostics)


__all__ = ["ScriptedDynamicPlanner"]
