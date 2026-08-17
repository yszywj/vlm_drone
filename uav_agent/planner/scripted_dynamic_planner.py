"""Deterministic model-free planner for dynamic runtime tests and demos."""

from __future__ import annotations

from planner.base import MissionPlanner
from planner.schemas import PlannerRequest, SkillPlanDraft


class ScriptedDynamicPlanner(MissionPlanner):
    """Return a fresh copy of one constructor-supplied SkillPlanDraft."""

    source = "dynamic_scripted"

    def __init__(self, skill_plan_draft: SkillPlanDraft) -> None:
        if not isinstance(skill_plan_draft, SkillPlanDraft):
            raise TypeError("skill_plan_draft must be a SkillPlanDraft")
        self._draft_data = skill_plan_draft.to_dict()

    def plan(self, request: PlannerRequest) -> SkillPlanDraft:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        return SkillPlanDraft.from_dict(self._draft_data)


__all__ = ["ScriptedDynamicPlanner"]
