"""Trusted analysis for a Fleet-local Spatial V3 safety epilogue.

The model-produced draft remains the semantic user plan.  This module only
decides how many trusted return/landing steps the runtime compiler must add so
the executable ``TaskPlan`` still ends safely.  Keeping this analysis shared
prevents the planner, symbolic checker, and compiler from disagreeing about
step and GOTO budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

from planner.schemas import PlannerWorldContext
from planner.schemas_v3 import SkillPlanDraftV3
from planner.spatial import NamedLocationTarget


@dataclass(frozen=True, slots=True)
class TrustedSafetyCompletion:
    """Minimal trusted suffix needed after one model-produced V3 draft."""

    zone_name: str | None
    append_goto: bool
    append_land: bool

    @property
    def additional_steps(self) -> int:
        return int(self.append_goto) + int(self.append_land)

    @property
    def additional_gotos(self) -> int:
        return int(self.append_goto)


def analyze_trusted_safety_completion(
    draft: SkillPlanDraftV3,
    world_context: PlannerWorldContext,
) -> TrustedSafetyCompletion:
    """Return the trusted suffix required to make ``draft`` land safely.

    A draft which already contains ``LAND`` is left untouched; ordinary
    symbolic validation remains responsible for rejecting duplicate,
    non-final, or mismatched landing steps.  A draft without ``LAND`` can use
    this mode only with one unambiguous trusted landing zone.  Trailing HOVER
    steps preserve position, so an earlier matching home GOTO can be reused.
    """

    if not isinstance(draft, SkillPlanDraftV3):
        raise TypeError("draft must be a SkillPlanDraftV3")
    if not isinstance(world_context, PlannerWorldContext):
        raise TypeError("world_context must be a PlannerWorldContext")

    if any(step.skill == "LAND" for step in draft.steps):
        return TrustedSafetyCompletion(None, False, False)

    zone_names = tuple(world_context.landing_zones)
    if len(zone_names) != 1:
        raise ValueError(
            "trusted runtime safety completion requires exactly one trusted "
            "landing zone in the focused world context"
        )
    zone_name = zone_names[0]

    return_index = len(draft.steps) - 1
    while return_index >= 0 and draft.steps[return_index].skill == "HOVER":
        return_index -= 1
    already_at_zone = False
    if return_index >= 0 and draft.steps[return_index].skill == "GOTO":
        target = draft.steps[return_index].args.get("target")
        already_at_zone = (
            isinstance(target, NamedLocationTarget)
            and target.name == zone_name
        )
    return TrustedSafetyCompletion(
        zone_name=zone_name,
        append_goto=not already_at_zone,
        append_land=True,
    )


__all__ = [
    "TrustedSafetyCompletion",
    "analyze_trusted_safety_completion",
]
