"""Goal coverage checking without prescribing one fixed Skill chain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from numbers import Real
import re
from typing import Protocol, runtime_checkable

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from planner.goal_effects import GoalFact, skill_effect
from runtime.validation_codes import ValidationCode
from runtime.validation_report import (
    RecoveryRecommendation,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
)


SUPPORTED_GOAL_TYPES = frozenset(
    {
        "SEARCH_TARGET",
        "TRACK_TARGET",
        "INSPECT_TARGET",
        "NAVIGATE",
        "RETURN_HOME",
        "LAND",
        "RETURN_HOME_AND_LAND",
        "WAIT",
        "REPORT",
    }
)
_STEP_TARGET_REF = re.compile(
    r"^\$(?P<step_id>[a-z][a-z0-9_]{0,31})\.target_id$"
)


@runtime_checkable
class GoalLike(Protocol):
    """Narrow boundary used to avoid importing Fleet TaskSpec at runtime."""

    goal_id: str
    goal_type: object


@dataclass(frozen=True, slots=True)
class GoalCoverage:
    schema_version: int
    goal_id: str
    goal_type: str
    covered: bool
    evidence_step_ids: tuple[str, ...]
    message: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("GoalCoverage.schema_version must equal 1")
        object.__setattr__(self, "goal_id", validate_routing_id(self.goal_id, "goal_id"))
        if self.goal_type not in SUPPORTED_GOAL_TYPES:
            # Unknown goal types are representable so the report can carry a
            # recoverable finding instead of crashing the whole Fleet.
            if not isinstance(self.goal_type, str) or not self.goal_type:
                raise ValueError("goal_type must be a non-empty string")
        if not isinstance(self.covered, bool):
            raise TypeError("covered must be bool")
        evidence = tuple(
            validate_routing_id(item, f"evidence_step_ids[{index}]")
            for index, item in enumerate(self.evidence_step_ids)
        )
        if len(evidence) != len(set(evidence)):
            raise ValueError("evidence_step_ids must not contain duplicates")
        object.__setattr__(self, "evidence_step_ids", evidence)
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be non-empty")

    @property
    def satisfied(self) -> bool:
        return self.covered

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "goal_id": self.goal_id,
            "goal_type": self.goal_type,
            "covered": self.covered,
            "evidence_step_ids": list(self.evidence_step_ids),
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class GoalCoverageReport:
    schema_version: int
    mission_id: str
    assignment_id: str | None
    uav_id: str | None
    coverages: tuple[GoalCoverage, ...]
    validation_report: ValidationReport

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("GoalCoverageReport.schema_version must equal 1")
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        if self.assignment_id is not None:
            object.__setattr__(
                self,
                "assignment_id",
                validate_routing_id(self.assignment_id, "assignment_id"),
            )
        if self.uav_id is not None:
            object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        coverages = tuple(self.coverages)
        if any(not isinstance(item, GoalCoverage) for item in coverages):
            raise TypeError("coverages must contain GoalCoverage values")
        if len({item.goal_id for item in coverages}) != len(coverages):
            raise ValueError("goal_id values must be unique")
        object.__setattr__(self, "coverages", coverages)
        if not isinstance(self.validation_report, ValidationReport):
            raise TypeError("validation_report must be a ValidationReport")
        if self.validation_report.mission_id != self.mission_id:
            raise ValueError("validation report mission_id does not match")

    @property
    def complete(self) -> bool:
        return all(item.covered for item in self.coverages)

    @property
    def satisfied(self) -> bool:
        return self.complete

    @property
    def uncovered_goal_ids(self) -> tuple[str, ...]:
        return tuple(item.goal_id for item in self.coverages if not item.covered)

    @property
    def findings(self) -> tuple[ValidationFinding, ...]:
        return self.validation_report.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mission_id": self.mission_id,
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "coverages": [item.to_dict() for item in self.coverages],
            "validation_report": self.validation_report.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _StepView:
    step_id: str
    skill: str
    args: Mapping[str, object]
    uav_id: str | None


class GoalSatisfactionChecker:
    """Check for at least one realizable path, not one privileged ordering.

    ``trusted_target_locked`` is trusted runtime state.  It is never inferred
    from user text or a model's claim.  SEARCH is considered a possible path
    to a confirmed target because its success handoff is gated by the trusted
    detector/tracker/verifier pipeline before TRACK is dispatched.
    """

    def check(
        self,
        goals: Sequence[GoalLike | Mapping[str, object]],
        plan: object,
        *,
        mission_id: str | None = None,
        assignment_id: str | None = None,
        uav_id: str | None = None,
        timestamp: float = 0.0,
        proposal_id: str | None = None,
        trusted_target_locked: bool = False,
        valid_landing_zone: bool = True,
        home_name: str | None = None,
    ) -> GoalCoverageReport:
        if isinstance(goals, (str, bytes)) or not isinstance(goals, Sequence):
            raise TypeError("goals must be a sequence")
        if not isinstance(trusted_target_locked, bool):
            raise TypeError("trusted_target_locked must be bool")
        if not isinstance(valid_landing_zone, bool):
            raise TypeError("valid_landing_zone must be bool")
        timestamp_s = _finite_nonnegative(timestamp, "timestamp")
        resolved_mission = validate_mission_id(
            mission_id or getattr(plan, "mission_id", None) or "mission_validation"
        )
        resolved_uav = uav_id or getattr(plan, "uav_id", None)
        if resolved_uav is not None:
            resolved_uav = validate_uav_id(resolved_uav)
        if assignment_id is not None:
            assignment_id = validate_routing_id(assignment_id, "assignment_id")
        if proposal_id is not None:
            proposal_id = validate_routing_id(proposal_id, "proposal_id")
        normalized_goals = tuple(_goal_view(item) for item in goals)
        if len(normalized_goals) > 32:
            raise ValueError("goals must contain at most 32 values")
        if len({item[0] for item in normalized_goals}) != len(normalized_goals):
            raise ValueError("goal_id values must be unique")
        steps = _plan_steps(plan)

        hard_findings: list[ValidationFinding] = []
        facts = {GoalFact.ON_GROUND}
        if valid_landing_zone:
            facts.add(GoalFact.VALID_LANDING_ZONE)
        if trusted_target_locked:
            facts.add(GoalFact.CONFIRMED_TARGET_AVAILABLE)
        executable_steps: list[_StepView] = []
        search_success_path = trusted_target_locked
        prior_steps: dict[str, _StepView] = {}

        for step in steps:
            effect = skill_effect(step.skill)
            if effect is None:
                hard_findings.append(
                    _finding(
                        mission_id=resolved_mission,
                        assignment_id=assignment_id,
                        uav_id=resolved_uav,
                        goal_id=None,
                        step_id=step.step_id,
                        proposal_id=proposal_id,
                        timestamp=timestamp_s,
                        severity=ValidationSeverity.HARD_ACTION_BLOCK,
                        code=ValidationCode.UNKNOWN_SKILL,
                        message=f"step {step.step_id} uses an unknown Skill",
                        action=RecoveryRecommendation.REPAIR_LOCAL_PLAN,
                    )
                )
                continue
            if step.skill == "TRACK":
                reference = step.args.get("target_ref")
                reference_error: str | None = None
                if reference == "$trusted_target.target_id":
                    if not trusted_target_locked:
                        reference_error = (
                            "trusted target_ref was used without a trusted runtime lock"
                        )
                elif isinstance(reference, str):
                    match = _STEP_TARGET_REF.fullmatch(reference)
                    source = None if match is None else prior_steps.get(match.group("step_id"))
                    if match is None or source is None or source.skill != "SEARCH":
                        reference_error = (
                            "TRACK target_ref must name an earlier SEARCH result"
                        )
                else:
                    reference_error = "TRACK target_ref is missing or invalid"
                if reference_error is not None:
                    hard_findings.append(
                        _finding(
                            mission_id=resolved_mission,
                            assignment_id=assignment_id,
                            uav_id=resolved_uav,
                            goal_id=None,
                            step_id=step.step_id,
                            proposal_id=proposal_id,
                            timestamp=timestamp_s,
                            severity=ValidationSeverity.HARD_ACTION_BLOCK,
                            code=ValidationCode.STEP_REFERENCE_INVALID,
                            message=reference_error,
                            action=RecoveryRecommendation.REPAIR_LOCAL_PLAN,
                        )
                    )
                    prior_steps[step.step_id] = step
                    continue
            bad_number = _first_nonfinite_path(step.args)
            if bad_number is not None:
                hard_findings.append(
                    _finding(
                        mission_id=resolved_mission,
                        assignment_id=assignment_id,
                        uav_id=resolved_uav,
                        goal_id=None,
                        step_id=step.step_id,
                        proposal_id=proposal_id,
                        timestamp=timestamp_s,
                        severity=ValidationSeverity.HARD_ACTION_BLOCK,
                        code=ValidationCode.NON_FINITE_NUMBER,
                        message=f"step {step.step_id} contains a non-finite number",
                        action=RecoveryRecommendation.REPAIR_LOCAL_PLAN,
                    )
                )
                continue
            if resolved_uav is not None and step.uav_id not in {None, resolved_uav}:
                hard_findings.append(
                    _finding(
                        mission_id=resolved_mission,
                        assignment_id=assignment_id,
                        uav_id=resolved_uav,
                        goal_id=None,
                        step_id=step.step_id,
                        proposal_id=proposal_id,
                        timestamp=timestamp_s,
                        severity=ValidationSeverity.HARD_ACTION_BLOCK,
                        code=ValidationCode.ROUTING_MISMATCH,
                        message=f"step {step.step_id} changed trusted UAV routing",
                        action=RecoveryRecommendation.REPAIR_LOCAL_PLAN,
                    )
                )
                continue

            preconditions = set(effect.preconditions)
            # A successful SEARCH is released as TARGET_FOUND only after the
            # trusted confirmation pipeline locks the identity.  This keeps
            # SEARCH->TRACK feasible without pretending every detection is a
            # confirmed target.
            if step.skill == "TRACK" and search_success_path:
                preconditions.discard(GoalFact.CONFIRMED_TARGET_AVAILABLE)
            if not preconditions.issubset(facts):
                continue
            executable_steps.append(step)
            facts.update(effect.effects)
            facts.update(effect.possible_effects)
            if step.skill in {"SEARCH", "INSPECT"}:
                search_success_path = True
                facts.add(GoalFact.CONFIRMED_TARGET_AVAILABLE)
            if step.skill == "LAND":
                facts.discard(GoalFact.AIRBORNE)
            elif step.skill == "TAKEOFF":
                facts.discard(GoalFact.ON_GROUND)
            prior_steps[step.step_id] = step

        coverages: list[GoalCoverage] = []
        semantic_findings: list[ValidationFinding] = []
        for goal_id, goal_type, goal in normalized_goals:
            covered, evidence, message, code = _coverage_for_goal(
                goal_type,
                goal,
                executable_steps,
                home_name=home_name,
            )
            coverages.append(
                GoalCoverage(1, goal_id, goal_type, covered, evidence, message)
            )
            if not covered:
                semantic_findings.append(
                    _finding(
                        mission_id=resolved_mission,
                        assignment_id=assignment_id,
                        uav_id=resolved_uav,
                        goal_id=goal_id,
                        step_id=None,
                        proposal_id=proposal_id,
                        timestamp=timestamp_s,
                        severity=ValidationSeverity.RECOVERABLE_SEMANTIC_ERROR,
                        code=code,
                        message=message,
                        action=RecoveryRecommendation.REPAIR_LOCAL_PLAN,
                        evidence_refs=evidence,
                    )
                )

        findings = tuple(hard_findings + semantic_findings)
        report_id = _stable_id(
            "report",
            resolved_mission,
            assignment_id or "none",
            str(timestamp_s),
            *(item.finding_id for item in findings),
        )
        validation = ValidationReport(
            schema_version=1,
            report_id=report_id,
            timestamp=timestamp_s,
            stage="LOCAL_GOAL_VALIDATION",
            mission_id=resolved_mission,
            assignment_id=assignment_id,
            uav_id=resolved_uav,
            findings=findings,
        )
        return GoalCoverageReport(
            schema_version=1,
            mission_id=resolved_mission,
            assignment_id=assignment_id,
            uav_id=resolved_uav,
            coverages=tuple(coverages),
            validation_report=validation,
        )

    evaluate = check


def _goal_view(
    goal: GoalLike | Mapping[str, object],
) -> tuple[str, str, object]:
    if isinstance(goal, Mapping):
        goal_id = goal.get("goal_id")
        goal_type = goal.get("goal_type")
    else:
        goal_id = getattr(goal, "goal_id", None)
        goal_type = getattr(goal, "goal_type", None)
    normalized_id = validate_routing_id(goal_id, "goal_id")
    raw_type = getattr(goal_type, "value", goal_type)
    if not isinstance(raw_type, str) or not raw_type:
        raise TypeError("goal_type must be a string or Enum value")
    return normalized_id, raw_type, goal


def _plan_steps(plan: object) -> tuple[_StepView, ...]:
    raw_steps = plan.get("steps") if isinstance(plan, Mapping) else getattr(plan, "steps", None)
    if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
        raise TypeError("plan must expose a sequence of steps")
    if len(raw_steps) > 10:
        raise ValueError("plan steps exceed the trusted limit of 10")
    result: list[_StepView] = []
    for index, raw in enumerate(raw_steps):
        if isinstance(raw, Mapping):
            step_id = raw.get("id", raw.get("step_id"))
            skill = raw.get("skill")
            args = raw.get("args", raw.get("params", {}))
            step_uav = raw.get("uav_id")
        else:
            step_id = getattr(raw, "id", getattr(raw, "step_id", None))
            skill = getattr(raw, "skill", None)
            args = getattr(raw, "args", getattr(raw, "params", {}))
            step_uav = getattr(raw, "uav_id", None)
        normalized_id = validate_routing_id(step_id, f"steps[{index}].id")
        raw_skill = getattr(skill, "value", skill)
        if not isinstance(raw_skill, str) or not raw_skill:
            raw_skill = "__UNKNOWN__"
        if not isinstance(args, Mapping) or any(not isinstance(key, str) for key in args):
            raise TypeError(f"steps[{index}].args must be an object")
        if step_uav is not None:
            step_uav = validate_uav_id(step_uav)
        result.append(_StepView(normalized_id, raw_skill.upper(), args, step_uav))
    return tuple(result)


def _value(goal: object, *names: str) -> object:
    for name in names:
        if isinstance(goal, Mapping) and name in goal:
            return goal[name]
        if hasattr(goal, name):
            return getattr(goal, name)
    parameters = goal.get("parameters") if isinstance(goal, Mapping) else getattr(goal, "parameters", None)
    if isinstance(parameters, Mapping):
        for name in names:
            if name in parameters:
                return parameters[name]
    return None


def _duration(goal: object) -> float | None:
    value = _value(goal, "duration_s", "track_duration_s", "wait_duration_s")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if isfinite(result) and result > 0.0 else None


def _coverage_for_goal(
    goal_type: str,
    goal: object,
    steps: Sequence[_StepView],
    *,
    home_name: str | None,
) -> tuple[bool, tuple[str, ...], str, ValidationCode]:
    if goal_type not in SUPPORTED_GOAL_TYPES:
        return (
            False,
            (),
            f"goal uses unsupported GoalType {goal_type}",
            ValidationCode.UNSUPPORTED_GOAL_TYPE,
        )

    if goal_type == "SEARCH_TARGET":
        candidates = tuple(step for step in steps if step.skill == "SEARCH")
        candidates = _matching_spatial_steps(candidates, goal, "region")
        return _simple_coverage(goal_type, candidates)

    if goal_type == "TRACK_TARGET":
        candidates = tuple(step for step in steps if step.skill == "TRACK")
        requested = _duration(goal)
        actual = sum(_positive_arg(step.args, "duration_s", "track_duration") for step in candidates)
        if candidates and (requested is None or actual + 1e-9 >= requested):
            return True, tuple(step.step_id for step in candidates), "TRACK goal has a realizable confirmed-target path", ValidationCode.GOAL_NOT_COVERED
        if candidates and requested is not None:
            return False, tuple(step.step_id for step in candidates), f"TRACK duration {actual:g}s is shorter than requested {requested:g}s", ValidationCode.TRACK_DURATION_UNDERSHOOT
        return False, (), "TRACK goal is not covered by a realizable TRACK step", ValidationCode.GOAL_NOT_COVERED

    if goal_type == "INSPECT_TARGET":
        return _simple_coverage(goal_type, tuple(step for step in steps if step.skill == "INSPECT"))

    if goal_type == "NAVIGATE":
        candidates = _matching_spatial_steps(
            tuple(step for step in steps if step.skill in {"GOTO", "FOLLOW_ROUTE"}),
            goal,
            "target",
        )
        return _simple_coverage(goal_type, candidates)

    if goal_type in {"RETURN_HOME", "RETURN_HOME_AND_LAND"}:
        expected_home = home_name or _value(goal, "home_name", "destination")
        gotos = tuple(step for step in steps if step.skill == "GOTO")
        returns = tuple(
            step for step in gotos if _is_home_target(step.args, expected_home)
        )
        if goal_type == "RETURN_HOME":
            if returns:
                return True, tuple(step.step_id for step in returns), "RETURN_HOME goal is covered", ValidationCode.RETURN_HOME_NOT_COVERED
            return False, (), "RETURN_HOME goal has no matching GOTO", ValidationCode.RETURN_HOME_NOT_COVERED
        lands = tuple(step for step in steps if step.skill == "LAND")
        for returned in returns:
            return_index = steps.index(returned)
            later_land = next((item for item in lands if steps.index(item) > return_index), None)
            if later_land is not None:
                return True, (returned.step_id, later_land.step_id), "RETURN_HOME_AND_LAND goal is covered", ValidationCode.GOAL_NOT_COVERED
        return False, (), "RETURN_HOME_AND_LAND goal lacks an ordered home GOTO and LAND", ValidationCode.RETURN_HOME_NOT_COVERED

    if goal_type == "LAND":
        return _simple_coverage(goal_type, tuple(step for step in steps if step.skill == "LAND"), missing_code=ValidationCode.LAND_NOT_COVERED)

    if goal_type == "WAIT":
        candidates = tuple(step for step in steps if step.skill == "HOVER")
        requested = _duration(goal)
        actual = sum(_positive_arg(step.args, "duration_s") for step in candidates)
        if candidates and (requested is None or actual + 1e-9 >= requested):
            return True, tuple(step.step_id for step in candidates), "WAIT goal is covered", ValidationCode.GOAL_NOT_COVERED
        if candidates and requested is not None:
            return False, tuple(step.step_id for step in candidates), f"WAIT duration {actual:g}s is shorter than requested {requested:g}s", ValidationCode.WAIT_DURATION_UNDERSHOOT
        return False, (), "WAIT goal has no HOVER step", ValidationCode.GOAL_NOT_COVERED

    return _simple_coverage(goal_type, tuple(step for step in steps if step.skill == "REPORT"))


def _simple_coverage(
    goal_type: str,
    steps: Sequence[_StepView],
    *,
    missing_code: ValidationCode = ValidationCode.GOAL_NOT_COVERED,
) -> tuple[bool, tuple[str, ...], str, ValidationCode]:
    if steps:
        return True, tuple(item.step_id for item in steps), f"{goal_type} goal has a realizable Skill path", missing_code
    return False, (), f"{goal_type} goal is not covered by the plan", missing_code


def _matching_spatial_steps(
    steps: Sequence[_StepView],
    goal: object,
    arg_name: str,
) -> tuple[_StepView, ...]:
    expected = _value(goal, "spatial_constraint", "region", "target", "destination")
    if expected is None:
        return tuple(steps)
    expected_json = _json_value(expected)
    return tuple(
        step
        for step in steps
        if _json_value(step.args.get(arg_name, step.args.get("destination"))) == expected_json
    )


def _json_value(value: object) -> object:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _is_home_target(args: Mapping[str, object], expected_home: object) -> bool:
    target = args.get("target", args.get("destination"))
    name = getattr(target, "name", target)
    if isinstance(target, Mapping):
        name = target.get("name", target.get("destination"))
    if expected_home is None:
        return isinstance(name, str) and "home" in name.casefold()
    expected = getattr(expected_home, "value", expected_home)
    return name == expected


def _positive_arg(args: Mapping[str, object], *names: str) -> float:
    for name in names:
        value = args.get(name)
        if isinstance(value, Real) and not isinstance(value, bool):
            result = float(value)
            if isfinite(result) and result > 0.0:
                return result
    return 0.0


def _first_nonfinite_path(value: object, path: str = "args") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            result = _first_nonfinite_path(item, f"{path}.{key}")
            if result is not None:
                return result
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result = _first_nonfinite_path(item, f"{path}[{index}]")
            if result is not None:
                return result
    elif isinstance(value, Real) and not isinstance(value, (bool, int)):
        if not isfinite(float(value)):
            return path
    return None


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _finding(
    *,
    mission_id: str,
    assignment_id: str | None,
    uav_id: str | None,
    goal_id: str | None,
    step_id: str | None,
    proposal_id: str | None,
    timestamp: float,
    severity: ValidationSeverity,
    code: ValidationCode,
    message: str,
    action: RecoveryRecommendation,
    evidence_refs: Sequence[str] = (),
) -> ValidationFinding:
    finding_id = _stable_id(
        "finding",
        mission_id,
        assignment_id or "none",
        goal_id or "none",
        step_id or "none",
        code.value,
    )
    return ValidationFinding(
        schema_version=1,
        finding_id=finding_id,
        timestamp=timestamp,
        stage="LOCAL_GOAL_VALIDATION",
        scope="ASSIGNMENT" if assignment_id is not None else "AGENT",
        severity=severity,
        code=code,
        message=message,
        mission_id=mission_id,
        assignment_id=assignment_id,
        uav_id=uav_id,
        goal_id=goal_id,
        step_id=step_id,
        proposal_id=proposal_id,
        evidence_refs=tuple(evidence_refs),
        recommended_action=action,
    )


__all__ = [
    "GoalCoverage",
    "GoalCoverageReport",
    "GoalLike",
    "GoalSatisfactionChecker",
    "SUPPORTED_GOAL_TYPES",
]
