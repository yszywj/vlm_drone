"""Shared cross-step validation for constrained dynamic Skill plans.

This checker deliberately operates before trusted geometry compilation.  It
therefore reasons only about symbols, ordering, references and bounded call
budgets.  Runtime safety remains an independent validation boundary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import hypot
import re

from planner.policy import PlannerLimits, PlannerPolicy
from planner.schemas import PlannerWorldContext, SkillPlanDraft, SkillPlanDraftV2


_TARGET_REF_PATTERN = re.compile(
    r"^\$(?P<step_id>[a-z][a-z0-9_]{0,31})\.target_id$"
)


class PlanIssueCode(str, Enum):
    PLAN_TOO_SHORT = "PLAN_TOO_SHORT"
    PLAN_TOO_LONG = "PLAN_TOO_LONG"
    TAKEOFF_NOT_FIRST = "TAKEOFF_NOT_FIRST"
    TAKEOFF_COUNT_INVALID = "TAKEOFF_COUNT_INVALID"
    LAND_NOT_FINAL = "LAND_NOT_FINAL"
    LAND_COUNT_INVALID = "LAND_COUNT_INVALID"
    LAND_GOTO_MISSING = "LAND_GOTO_MISSING"
    LAND_ZONE_MISMATCH = "LAND_ZONE_MISMATCH"
    GOTO_LIMIT_EXCEEDED = "GOTO_LIMIT_EXCEEDED"
    SEARCH_LIMIT_EXCEEDED = "SEARCH_LIMIT_EXCEEDED"
    TRACK_LIMIT_EXCEEDED = "TRACK_LIMIT_EXCEEDED"
    TOP_LEVEL_REACQUIRE_FORBIDDEN = "TOP_LEVEL_REACQUIRE_FORBIDDEN"
    TRACK_WITHOUT_SEARCH = "TRACK_WITHOUT_SEARCH"
    INSPECT_WITHOUT_SEARCH = "INSPECT_WITHOUT_SEARCH"
    TARGET_REF_INVALID = "TARGET_REF_INVALID"
    TARGET_REF_FORWARD = "TARGET_REF_FORWARD"
    TARGET_REF_NOT_SEARCH = "TARGET_REF_NOT_SEARCH"
    RECOVERY_ON_NON_TRACK = "RECOVERY_ON_NON_TRACK"
    RECOVERY_BUDGET_EXCEEDED = "RECOVERY_BUDGET_EXCEEDED"
    RECOVERY_CONFLICTS_WITH_FAIL = "RECOVERY_CONFLICTS_WITH_FAIL"
    UNKNOWN_NAMED_LOCATION = "UNKNOWN_NAMED_LOCATION"
    ROUTE_TARGET_REQUIRES_FOLLOW_ROUTE = "ROUTE_TARGET_REQUIRES_FOLLOW_ROUTE"
    # JSON Schema cannot express uniqueness of an object field across array
    # items, so the shared symbolic boundary owns this rule as well.
    STEP_ID_DUPLICATE = "STEP_ID_DUPLICATE"


@dataclass(frozen=True, slots=True)
class PlanIssue:
    code: PlanIssueCode
    message: str
    step_id: str | None
    repairable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.code, PlanIssueCode):
            raise TypeError("code must be a PlanIssueCode")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string")
        if self.step_id is not None and not isinstance(self.step_id, str):
            raise TypeError("step_id must be a string or None")
        if not isinstance(self.repairable, bool):
            raise TypeError("repairable must be a bool")


@dataclass(frozen=True, slots=True)
class SymbolicCheckResult:
    issues: tuple[PlanIssue, ...]

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if any(not isinstance(issue, PlanIssue) for issue in issues):
            raise TypeError("issues must contain only PlanIssue values")
        object.__setattr__(self, "issues", issues)

    @property
    def valid(self) -> bool:
        return not self.issues


class SymbolicPlanChecker:
    """Validate all model-draft rules that depend on multiple steps."""

    def check(
        self,
        draft: object,
        *,
        world_context: PlannerWorldContext,
        limits: PlannerLimits,
        policy: PlannerPolicy,
    ) -> SymbolicCheckResult:
        from planner.schemas_v3 import SkillPlanDraftV3

        if isinstance(draft, SkillPlanDraftV3):
            return self._check_v3(
                draft,
                world_context=world_context,
                limits=limits,
                policy=policy,
            )
        if isinstance(draft, SkillPlanDraftV2):
            draft = draft.to_v1()
        elif not isinstance(draft, SkillPlanDraft):
            raise TypeError("draft must be a SkillPlanDraft or SkillPlanDraftV2")
        if not isinstance(world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")
        if not isinstance(limits, PlannerLimits):
            raise TypeError("limits must be a PlannerLimits")
        if not isinstance(policy, PlannerPolicy):
            raise TypeError("policy must be a PlannerPolicy")
        policy.validate_against(limits)

        steps = draft.steps
        issues: list[PlanIssue] = []

        def add(
            code: PlanIssueCode,
            message: str,
            step_id: str | None = None,
        ) -> None:
            issues.append(
                PlanIssue(
                    code=code,
                    message=message,
                    step_id=step_id,
                    repairable=True,
                )
            )

        if len(steps) < 2:
            add(PlanIssueCode.PLAN_TOO_SHORT, "plan must contain at least 2 steps")
        if len(steps) > limits.max_plan_steps:
            add(
                PlanIssueCode.PLAN_TOO_LONG,
                f"plan exceeds the trusted {limits.max_plan_steps}-step limit",
            )

        ids = [step.id for step in steps]
        seen_ids: set[str] = set()
        reported_duplicates: set[str] = set()
        for step_id in ids:
            if step_id in seen_ids and step_id not in reported_duplicates:
                add(
                    PlanIssueCode.STEP_ID_DUPLICATE,
                    "step id must be unique",
                    step_id,
                )
                reported_duplicates.add(step_id)
            seen_ids.add(step_id)

        counts = Counter(step.skill for step in steps)
        if counts["TAKEOFF"] != 1:
            add(
                PlanIssueCode.TAKEOFF_COUNT_INVALID,
                "TAKEOFF must appear exactly once",
            )
        takeoff_steps = [step for step in steps if step.skill == "TAKEOFF"]
        if takeoff_steps and steps and steps[0].skill != "TAKEOFF":
            add(
                PlanIssueCode.TAKEOFF_NOT_FIRST,
                "TAKEOFF must be the first step",
                takeoff_steps[0].id,
            )
        if counts["LAND"] != 1:
            add(
                PlanIssueCode.LAND_COUNT_INVALID,
                "LAND must appear exactly once",
            )
        land_steps = [step for step in steps if step.skill == "LAND"]
        if land_steps and steps and steps[-1].skill != "LAND":
            add(
                PlanIssueCode.LAND_NOT_FINAL,
                "LAND must be the final step",
                land_steps[-1].id,
            )

        for skill, limit, code in (
            ("GOTO", limits.max_goto_calls, PlanIssueCode.GOTO_LIMIT_EXCEEDED),
            (
                "SEARCH",
                limits.max_search_calls,
                PlanIssueCode.SEARCH_LIMIT_EXCEEDED,
            ),
            (
                "TRACK",
                limits.max_track_calls,
                PlanIssueCode.TRACK_LIMIT_EXCEEDED,
            ),
        ):
            if counts[skill] > limit:
                add(code, f"{skill} exceeds the trusted call limit of {limit}")

        for step in steps:
            if step.skill == "REACQUIRE":
                add(
                    PlanIssueCode.TOP_LEVEL_REACQUIRE_FORBIDDEN,
                    "REACQUIRE is recovery-only and cannot be top-level",
                    step.id,
                )

        known_goto_names = (
            set(world_context.search_regions)
            | set(world_context.landing_zones)
            | set(world_context.navigation_points)
        )
        for step in steps:
            if step.skill == "GOTO":
                destination = step.args.get("destination")
                if destination not in known_goto_names:
                    add(
                        PlanIssueCode.UNKNOWN_NAMED_LOCATION,
                        "GOTO destination is not a trusted named location",
                        step.id,
                    )
            elif step.skill == "SEARCH":
                region = step.args.get("region")
                if region not in world_context.search_regions:
                    add(
                        PlanIssueCode.UNKNOWN_NAMED_LOCATION,
                        "SEARCH region is not a trusted search-region name",
                        step.id,
                    )
            elif step.skill == "LAND":
                zone = step.args.get("zone")
                if zone not in world_context.landing_zones:
                    add(
                        PlanIssueCode.UNKNOWN_NAMED_LOCATION,
                        "LAND zone is not a trusted landing-zone name",
                        step.id,
                    )

        for index, step in enumerate(steps):
            if step.skill != "LAND":
                continue
            if self._is_valid_takeoff_land_exception(
                steps,
                index=index,
                world_context=world_context,
            ):
                continue
            if index == 0 or steps[index - 1].skill != "GOTO":
                add(
                    PlanIssueCode.LAND_GOTO_MISSING,
                    "LAND must be preceded by matching GOTO",
                    step.id,
                )
                continue
            previous = steps[index - 1]
            if previous.args.get("destination") != step.args.get("zone"):
                add(
                    PlanIssueCode.LAND_ZONE_MISMATCH,
                    "LAND zone must match the preceding GOTO destination",
                    step.id,
                )

        total_recovery_attempts = 0
        indexes_by_id: dict[str, list[int]] = {}
        for index, step in enumerate(steps):
            indexes_by_id.setdefault(step.id, []).append(index)

        for index, step in enumerate(steps):
            recovery = step.recovery
            if recovery is not None and step.skill != "TRACK":
                add(
                    PlanIssueCode.RECOVERY_ON_NON_TRACK,
                    "recovery is only allowed on TRACK steps",
                    step.id,
                )

            if step.skill != "TRACK":
                if step.skill == "INSPECT" and not any(
                    previous.skill == "SEARCH" for previous in steps[:index]
                ):
                    add(
                        PlanIssueCode.INSPECT_WITHOUT_SEARCH,
                        "INSPECT requires a prior SEARCH step",
                        step.id,
                    )
                continue

            reference = step.args.get("target_ref")
            match = (
                _TARGET_REF_PATTERN.fullmatch(reference)
                if isinstance(reference, str)
                else None
            )
            if match is None:
                add(
                    PlanIssueCode.TARGET_REF_INVALID,
                    "TRACK target_ref must use $<step_id>.target_id",
                    step.id,
                )
            else:
                source_id = match.group("step_id")
                source_indexes = indexes_by_id.get(source_id, [])
                prior_indexes = [item for item in source_indexes if item < index]
                if not source_indexes:
                    add(
                        PlanIssueCode.TARGET_REF_INVALID,
                        "TRACK target_ref names no step in this plan",
                        step.id,
                    )
                elif not prior_indexes:
                    add(
                        PlanIssueCode.TARGET_REF_FORWARD,
                        "TRACK target_ref must point to an earlier step",
                        step.id,
                    )
                else:
                    source = steps[prior_indexes[-1]]
                    if source.skill != "SEARCH":
                        add(
                            PlanIssueCode.TARGET_REF_NOT_SEARCH,
                            "TRACK target_ref must point to a SEARCH step",
                            step.id,
                        )

            if not any(previous.skill == "SEARCH" for previous in steps[:index]):
                add(
                    PlanIssueCode.TRACK_WITHOUT_SEARCH,
                    "TRACK requires a prior SEARCH step",
                    step.id,
                )

            action = step.args.get("on_target_lost")
            if action == "FAIL" and recovery is not None:
                add(
                    PlanIssueCode.RECOVERY_CONFLICTS_WITH_FAIL,
                    "on_target_lost=FAIL cannot be combined with recovery",
                    step.id,
                )
            if action == "FAIL":
                effective_attempts = 0
            elif recovery is not None:
                # This also preserves the deprecated hand-written
                # max_attempts=0 compatibility boundary.
                effective_attempts = recovery.max_attempts
            else:
                effective_action = (
                    action
                    if action is not None
                    else policy.default_on_target_lost.value
                )
                effective_attempts = (
                    policy.default_reacquire_max_attempts
                    if effective_action == "REACQUIRE"
                    else 0
                )
            total_recovery_attempts += effective_attempts
            if effective_attempts > limits.max_reacquire_attempts_per_track:
                add(
                    PlanIssueCode.RECOVERY_BUDGET_EXCEEDED,
                    "TRACK recovery exceeds the per-TRACK attempt budget",
                    step.id,
                )

        if total_recovery_attempts > limits.max_total_reacquire_attempts:
            add(
                PlanIssueCode.RECOVERY_BUDGET_EXCEEDED,
                "total REACQUIRE attempt budget exceeds the trusted planner limit",
            )

        return SymbolicCheckResult(tuple(issues))

    def _check_v3(
        self,
        draft: object,
        *,
        world_context: PlannerWorldContext,
        limits: PlannerLimits,
        policy: PlannerPolicy,
    ) -> SymbolicCheckResult:
        """Validate V3 program structure without reapplying V2 location rules."""

        from planner.schemas_v3 import SkillPlanDraftV3
        from planner.spatial import NamedLocationTarget, RouteTarget

        if not isinstance(draft, SkillPlanDraftV3):
            raise TypeError("draft must be a SkillPlanDraftV3")
        if not isinstance(world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")
        if not isinstance(limits, PlannerLimits):
            raise TypeError("limits must be a PlannerLimits")
        if not isinstance(policy, PlannerPolicy):
            raise TypeError("policy must be a PlannerPolicy")
        policy.validate_against(limits)

        steps = draft.steps
        issues: list[PlanIssue] = []

        def add(
            code: PlanIssueCode,
            message: str,
            step_id: str | None = None,
        ) -> None:
            issues.append(PlanIssue(code, message, step_id, True))

        if len(steps) < 2:
            add(PlanIssueCode.PLAN_TOO_SHORT, "plan must contain at least 2 steps")
        if len(steps) > limits.max_plan_steps:
            add(
                PlanIssueCode.PLAN_TOO_LONG,
                f"plan exceeds the trusted {limits.max_plan_steps}-step limit",
            )
        if len({step.id for step in steps}) != len(steps):
            seen: set[str] = set()
            for step in steps:
                if step.id in seen:
                    add(PlanIssueCode.STEP_ID_DUPLICATE, "step id must be unique", step.id)
                    break
                seen.add(step.id)

        counts = Counter(step.skill for step in steps)
        if counts["TAKEOFF"] != 1:
            add(PlanIssueCode.TAKEOFF_COUNT_INVALID, "TAKEOFF must appear exactly once")
        if steps and steps[0].skill != "TAKEOFF":
            add(PlanIssueCode.TAKEOFF_NOT_FIRST, "TAKEOFF must be the first step", steps[0].id)
        if counts["LAND"] != 1:
            add(PlanIssueCode.LAND_COUNT_INVALID, "LAND must appear exactly once")
        if steps and steps[-1].skill != "LAND":
            add(PlanIssueCode.LAND_NOT_FINAL, "LAND must be the final step", steps[-1].id)
        if counts["GOTO"] > limits.max_goto_calls:
            add(
                PlanIssueCode.GOTO_LIMIT_EXCEEDED,
                f"GOTO exceeds the trusted call limit of {limits.max_goto_calls}",
            )
        # V3 intentionally does not reuse PlannerLimits.max_search_calls=1.
        # Multiple searches remain bounded by max_plan_steps and compiler time.
        if counts["TRACK"] > limits.max_track_calls:
            add(
                PlanIssueCode.TRACK_LIMIT_EXCEEDED,
                f"TRACK exceeds the trusted call limit of {limits.max_track_calls}",
            )
        for step in steps:
            if step.skill == "REACQUIRE":
                add(
                    PlanIssueCode.TOP_LEVEL_REACQUIRE_FORBIDDEN,
                    "REACQUIRE is recovery-only and cannot be top-level",
                    step.id,
                )

        known_names = (
            set(world_context.search_regions)
            | set(world_context.landing_zones)
            | set(world_context.navigation_points)
        )
        for step in steps:
            if step.skill == "GOTO":
                target = step.args.get("target")
                if isinstance(target, NamedLocationTarget) and target.name not in known_names:
                    add(
                        PlanIssueCode.UNKNOWN_NAMED_LOCATION,
                        "GOTO named target is not registered in trusted context",
                        step.id,
                    )
                elif isinstance(target, RouteTarget):
                    add(
                        PlanIssueCode.ROUTE_TARGET_REQUIRES_FOLLOW_ROUTE,
                        "multi-waypoint routes must use FOLLOW_ROUTE, not GOTO",
                        step.id,
                    )
            elif step.skill == "LAND":
                if step.args.get("zone") not in world_context.landing_zones:
                    add(
                        PlanIssueCode.UNKNOWN_NAMED_LOCATION,
                        "LAND zone is not a trusted landing zone",
                        step.id,
                    )

        indexes_by_id = {step.id: index for index, step in enumerate(steps)}
        total_recovery_attempts = 0
        for index, step in enumerate(steps):
            recovery = step.recovery
            if recovery is not None and step.skill != "TRACK":
                add(
                    PlanIssueCode.RECOVERY_ON_NON_TRACK,
                    "recovery is only allowed on TRACK steps",
                    step.id,
                )
            if step.skill == "INSPECT" and not any(
                previous.skill == "SEARCH" for previous in steps[:index]
            ):
                add(
                    PlanIssueCode.INSPECT_WITHOUT_SEARCH,
                    "INSPECT requires a prior SEARCH step",
                    step.id,
                )
            if step.skill != "TRACK":
                continue

            reference = step.args.get("target_ref")
            match = (
                _TARGET_REF_PATTERN.fullmatch(reference)
                if isinstance(reference, str)
                else None
            )
            if match is None:
                add(
                    PlanIssueCode.TARGET_REF_INVALID,
                    "TRACK target_ref must use $<step_id>.target_id",
                    step.id,
                )
            else:
                source_id = match.group("step_id")
                source_index = indexes_by_id.get(source_id)
                if source_index is None:
                    add(
                        PlanIssueCode.TARGET_REF_INVALID,
                        "TRACK target_ref names no step in this plan",
                        step.id,
                    )
                elif source_index >= index:
                    add(
                        PlanIssueCode.TARGET_REF_FORWARD,
                        "TRACK target_ref must point to an earlier step",
                        step.id,
                    )
                elif steps[source_index].skill != "SEARCH":
                    add(
                        PlanIssueCode.TARGET_REF_NOT_SEARCH,
                        "TRACK target_ref must point to a SEARCH step",
                        step.id,
                    )
            if not any(previous.skill == "SEARCH" for previous in steps[:index]):
                add(
                    PlanIssueCode.TRACK_WITHOUT_SEARCH,
                    "TRACK requires a prior SEARCH step",
                    step.id,
                )

            action = step.args.get("on_target_lost")
            if action == "FAIL" and recovery is not None:
                add(
                    PlanIssueCode.RECOVERY_CONFLICTS_WITH_FAIL,
                    "on_target_lost=FAIL cannot be combined with recovery",
                    step.id,
                )
            if action == "FAIL":
                attempts = 0
            elif recovery is not None:
                attempts = recovery.max_attempts
            else:
                effective = action or policy.default_on_target_lost.value
                attempts = (
                    policy.default_reacquire_max_attempts
                    if effective == "REACQUIRE"
                    else 0
                )
            total_recovery_attempts += attempts
            if attempts > limits.max_reacquire_attempts_per_track:
                add(
                    PlanIssueCode.RECOVERY_BUDGET_EXCEEDED,
                    "TRACK recovery exceeds the per-TRACK attempt budget",
                    step.id,
                )

        if total_recovery_attempts > limits.max_total_reacquire_attempts:
            add(
                PlanIssueCode.RECOVERY_BUDGET_EXCEEDED,
                "total REACQUIRE attempt budget exceeds the trusted planner limit",
            )
        return SymbolicCheckResult(tuple(issues))

    @staticmethod
    def _is_valid_takeoff_land_exception(
        steps: tuple[object, ...],
        *,
        index: int,
        world_context: PlannerWorldContext,
    ) -> bool:
        if (
            len(steps) != 2
            or index != 1
            or getattr(steps[0], "skill", None) != "TAKEOFF"
        ):
            return False
        land = steps[1]
        zone_name = getattr(land, "args", {}).get("zone")
        zone = world_context.landing_zones.get(zone_name)
        if zone is None:
            return False
        tolerance = getattr(zone, "horizontal_tolerance_m", 0.75)
        error = hypot(
            world_context.initial_uav_xyz_m[0] - zone.position_xy_m[0],
            world_context.initial_uav_xyz_m[1] - zone.position_xy_m[1],
        )
        return error <= tolerance


__all__ = [
    "PlanIssue",
    "PlanIssueCode",
    "SymbolicCheckResult",
    "SymbolicPlanChecker",
]
