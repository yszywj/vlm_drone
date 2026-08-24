"""Deterministic model-free planner for dynamic runtime tests and demos."""

from __future__ import annotations

from planner.base import MissionPlanner
from planner.diagnostics import PlannerDiagnostics, PlannerExecution
from planner.schemas import (
    PlannerRequest,
    SkillPlanDraft,
    SkillPlanDraftV2,
    migrate_plan_v1_to_v2,
)
from planner.scripted_target_semantics import target_spec_from_scripted_instruction


class ScriptedDynamicPlanner(MissionPlanner):
    """Return a fresh baseline with conservatively bound target semantics.

    The operational Skill order remains constructor-supplied.  A small,
    deterministic grammar may replace only SEARCH.target_description from the
    request; it never invents geometry, calls a model, or fuzzily guesses a
    detector category.
    """

    source = "dynamic_scripted"

    def __init__(self, skill_plan_draft: SkillPlanDraft) -> None:
        if not isinstance(skill_plan_draft, SkillPlanDraft):
            raise TypeError("skill_plan_draft must be a SkillPlanDraft")
        self._draft_data = skill_plan_draft.to_dict()

    def plan(self, request: PlannerRequest) -> SkillPlanDraft | SkillPlanDraftV2:
        return self.plan_with_diagnostics(request).output

    def plan_with_diagnostics(self, request: PlannerRequest) -> PlannerExecution:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        output = SkillPlanDraft.from_dict(self._draft_data)
        target_spec = target_spec_from_scripted_instruction(request.instruction)
        if target_spec is not None:
            rebound = output.to_dict()
            rebound_steps = rebound["steps"]
            assert isinstance(rebound_steps, list)
            for raw_step in rebound_steps:
                assert isinstance(raw_step, dict)
                if raw_step.get("skill") != "SEARCH":
                    continue
                raw_args = raw_step.get("args")
                assert isinstance(raw_args, dict)
                raw_args["target_description"] = target_spec.original_description
            output = SkillPlanDraft.from_dict(rebound)
        if request.has_routing_ids:
            assert request.mission_id is not None
            assert request.uav_id is not None
            assert request.plan_version is not None
            output = migrate_plan_v1_to_v2(
                output,
                mission_id=request.mission_id,
                uav_id=request.uav_id,
                plan_version=request.plan_version,
            )
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
