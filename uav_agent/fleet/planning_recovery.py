"""Trusted repair decisions and Fleet-replan orchestration.

The coordinator never executes controller actions and never treats model text
as authority.  It consumes structured findings, enforces independent repair
budgets, and can invoke injected pure-Python planner/compiler boundaries for a
new, monotonically-versioned Fleet proposal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from common.ids import validate_routing_id, validate_uav_id
from fleet.planning_state import PlanningRepairBudget, PlanningStage


class PlanningRecoveryError(RuntimeError):
    pass


class RecoveryDisposition(str, Enum):
    REPAIR = "REPAIR"
    DEGRADED_EXECUTABLE = "DEGRADED_EXECUTABLE"
    WAITING_REASSIGNMENT = "WAITING_REASSIGNMENT"
    KEEP_GROUNDED = "KEEP_GROUNDED"
    HOVER_AND_REPLAN = "HOVER_AND_REPLAN"
    SAFE_LAND = "SAFE_LAND"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class PlanningRecoveryDecision:
    disposition: RecoveryDisposition
    stage: PlanningStage
    assignment_id: str | None
    repair_attempt: int
    next_plan_version: int
    reason_codes: tuple[str, ...]
    uncovered_goal_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("schema_version must equal integer 1")
        object.__setattr__(self, "disposition", RecoveryDisposition(self.disposition))
        object.__setattr__(self, "stage", PlanningStage(self.stage))
        if self.assignment_id is not None:
            object.__setattr__(
                self,
                "assignment_id",
                validate_routing_id(self.assignment_id, "assignment_id"),
            )
        for name in ("repair_attempt", "next_plan_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < (0 if name == "repair_attempt" else 1):
                raise ValueError(f"{name} is outside its valid range")
        reasons = tuple(self.reason_codes)
        if not reasons or len(reasons) > 64 or any(
            not isinstance(value, str) or not value.strip() or len(value) > 128
            for value in reasons
        ):
            raise ValueError("reason_codes must contain 1..64 bounded strings")
        object.__setattr__(self, "reason_codes", reasons)
        goals = tuple(
            validate_routing_id(value, "uncovered_goal_id")
            for value in self.uncovered_goal_ids
        )
        if len(goals) != len(set(goals)) or len(goals) > 64:
            raise ValueError("uncovered_goal_ids must be unique and bounded")
        object.__setattr__(self, "uncovered_goal_ids", goals)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "disposition": self.disposition.value,
            "stage": self.stage.value,
            "assignment_id": self.assignment_id,
            "repair_attempt": self.repair_attempt,
            "next_plan_version": self.next_plan_version,
            "reason_codes": list(self.reason_codes),
            "uncovered_goal_ids": list(self.uncovered_goal_ids),
        }


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _severity(value: object) -> str:
    raw = _field(value, "severity", "RECOVERABLE_SEMANTIC_ERROR")
    return str(getattr(raw, "value", raw)).upper()


def _code(value: object) -> str:
    raw = _field(value, "code", "UNKNOWN_VALIDATION_ERROR")
    text = str(getattr(raw, "value", raw)).strip()
    return text[:128] or "UNKNOWN_VALIDATION_ERROR"


class PlanningRecoveryCoordinator:
    """Choose repair/degrade/reassign without broadening action authority."""

    def __init__(self, budget: PlanningRepairBudget | None = None) -> None:
        self.budget = budget or PlanningRepairBudget()
        self._attempts: dict[tuple[str | None, PlanningStage], int] = {}

    def decide(
        self,
        *,
        stage: PlanningStage | str,
        plan_version: int,
        findings: Sequence[object],
        assignment_id: str | None = None,
        airborne: bool = False,
        safe_partial_available: bool = False,
        uncovered_goal_ids: Sequence[str] = (),
    ) -> PlanningRecoveryDecision:
        normalized_stage = PlanningStage(stage)
        if isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version <= 0:
            raise ValueError("plan_version must be a positive integer")
        if assignment_id is not None:
            assignment_id = validate_routing_id(assignment_id, "assignment_id")
        if not isinstance(airborne, bool) or not isinstance(safe_partial_available, bool):
            raise TypeError("airborne and safe_partial_available must be bool")
        values = tuple(findings)
        codes = tuple(dict.fromkeys(_code(value) for value in values)) or (
            "PLANNING_OUTPUT_REJECTED",
        )
        severities = {_severity(value) for value in values}
        hard = bool(severities & {"HARD_ACTION_BLOCK", "FATAL_SAFETY"})
        key = (assignment_id, normalized_stage)
        used = self._attempts.get(key, 0)
        limit = self.budget.for_stage(normalized_stage)

        if hard:
            disposition = (
                RecoveryDisposition.SAFE_LAND
                if airborne and "FATAL_SAFETY" in severities
                else RecoveryDisposition.HOVER_AND_REPLAN
                if airborne
                else RecoveryDisposition.KEEP_GROUNDED
            )
            return PlanningRecoveryDecision(
                disposition,
                normalized_stage,
                assignment_id,
                used,
                plan_version,
                codes,
                tuple(uncovered_goal_ids),
            )

        if used < limit:
            used += 1
            self._attempts[key] = used
            return PlanningRecoveryDecision(
                RecoveryDisposition.REPAIR,
                normalized_stage,
                assignment_id,
                used,
                plan_version + 1,
                codes,
                tuple(uncovered_goal_ids),
            )
        if safe_partial_available:
            return PlanningRecoveryDecision(
                RecoveryDisposition.DEGRADED_EXECUTABLE,
                normalized_stage,
                assignment_id,
                used,
                plan_version,
                codes,
                tuple(uncovered_goal_ids),
            )
        return PlanningRecoveryDecision(
            RecoveryDisposition.WAITING_REASSIGNMENT,
            normalized_stage,
            assignment_id,
            used,
            plan_version,
            codes,
            tuple(uncovered_goal_ids),
        )

    def attempts_for(
        self, assignment_id: str | None, stage: PlanningStage | str
    ) -> int:
        if assignment_id is not None:
            assignment_id = validate_routing_id(assignment_id, "assignment_id")
        return self._attempts.get((assignment_id, PlanningStage(stage)), 0)


@dataclass(frozen=True, slots=True)
class FleetReplanRequest:
    fleet_mission_id: str
    base_fleet_plan_version: int
    incomplete_goal_ids: tuple[str, ...]
    available_uav_ids: tuple[str, ...]
    trusted_fleet_state: Mapping[str, object]
    reason_codes: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        from common.ids import validate_mission_id

        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("schema_version must equal integer 1")
        object.__setattr__(
            self, "fleet_mission_id", validate_mission_id(self.fleet_mission_id)
        )
        if (
            isinstance(self.base_fleet_plan_version, bool)
            or not isinstance(self.base_fleet_plan_version, int)
            or self.base_fleet_plan_version <= 0
        ):
            raise ValueError("base_fleet_plan_version must be positive")
        goals = tuple(
            validate_routing_id(value, "incomplete_goal_id")
            for value in self.incomplete_goal_ids
        )
        uavs = tuple(validate_uav_id(value) for value in self.available_uav_ids)
        if not goals or len(goals) != len(set(goals)):
            raise ValueError("incomplete_goal_ids must be non-empty and unique")
        if not uavs or len(uavs) != len(set(uavs)):
            raise ValueError("available_uav_ids must be non-empty and unique")
        reasons = tuple(self.reason_codes)
        if not reasons or any(
            not isinstance(value, str) or not value.strip() or len(value) > 128
            for value in reasons
        ):
            raise ValueError("reason_codes must contain bounded strings")
        if not isinstance(self.trusted_fleet_state, Mapping):
            raise TypeError("trusted_fleet_state must be a mapping")
        # A deep JSON policy is enforced by the V2 schema/parser.  This object
        # only exposes an immutable shallow routing snapshot to injected code.
        object.__setattr__(self, "incomplete_goal_ids", goals)
        object.__setattr__(self, "available_uav_ids", uavs)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "trusted_fleet_state",
            MappingProxyType(dict(self.trusted_fleet_state)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "fleet_mission_id": self.fleet_mission_id,
            "base_fleet_plan_version": self.base_fleet_plan_version,
            "incomplete_goal_ids": list(self.incomplete_goal_ids),
            "available_uav_ids": list(self.available_uav_ids),
            "trusted_fleet_state": dict(self.trusted_fleet_state),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class FleetReplanOutcome:
    proposal: object
    compilations: Mapping[str, object]
    new_fleet_plan_version: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("schema_version must equal integer 1")
        if (
            isinstance(self.new_fleet_plan_version, bool)
            or not isinstance(self.new_fleet_plan_version, int)
            or self.new_fleet_plan_version <= 1
        ):
            raise ValueError("new_fleet_plan_version must be greater than one")
        if not isinstance(self.compilations, Mapping):
            raise TypeError("compilations must be a mapping")
        if len(self.compilations) > 64:
            raise ValueError("compilations must contain at most 64 entries")
        normalized: dict[str, object] = {}
        for raw_assignment_id, compilation in self.compilations.items():
            assignment_id = validate_routing_id(
                raw_assignment_id, "compilation_assignment_id"
            )
            normalized[assignment_id] = compilation
        object.__setattr__(
            self, "compilations", MappingProxyType(normalized)
        )


def execute_fleet_replan(
    request: FleetReplanRequest,
    *,
    planner: object,
    compiler: object,
) -> FleetReplanOutcome:
    """Run a real injected replan and independently compile each assignment.

    ``planner`` may expose ``replan`` or ``plan``.  Its proposal must increase
    the Fleet version by exactly one.  ``compiler`` may expose
    ``compile_reassignment`` or ``compile_assignment``; failures stay scoped to
    their assignment and are returned as exception values instead of aborting
    successful compilations.
    """

    if not isinstance(request, FleetReplanRequest):
        raise TypeError("request must be a FleetReplanRequest")
    plan_method = getattr(planner, "replan", None) or getattr(planner, "plan", None)
    if not callable(plan_method):
        raise TypeError("planner must provide replan() or plan()")
    proposal = plan_method(request)
    raw_version = _field(
        proposal,
        "new_fleet_plan_version",
        _field(proposal, "fleet_plan_version", None),
    )
    expected_version = request.base_fleet_plan_version + 1
    if raw_version != expected_version:
        raise PlanningRecoveryError(
            "Fleet replan proposal must increment fleet_plan_version by exactly one"
        )
    assignments = _field(
        proposal,
        "replacement_assignments",
        _field(proposal, "assignments", ()),
    )
    if isinstance(assignments, (str, bytes)) or not isinstance(assignments, Sequence):
        raise PlanningRecoveryError("Fleet replan proposal assignments must be an array")
    if not assignments or len(assignments) > 64:
        raise PlanningRecoveryError(
            "Fleet replan proposal must contain 1..64 assignments"
        )
    compile_method = getattr(compiler, "compile_reassignment", None) or getattr(
        compiler, "compile_assignment", None
    )
    if not callable(compile_method):
        raise TypeError("compiler must provide compile_reassignment() or compile_assignment()")
    results: dict[str, object] = {}
    for assignment in assignments:
        assignment_id = _field(assignment, "assignment_id", None)
        if not isinstance(assignment_id, str):
            raise PlanningRecoveryError("replacement assignment lacks assignment_id")
        assignment_id = validate_routing_id(assignment_id, "assignment_id")
        if assignment_id in results:
            raise PlanningRecoveryError(
                "Fleet replan proposal contains duplicate assignment_id"
            )
        try:
            results[assignment_id] = compile_method(request, proposal, assignment)
        except Exception as exc:  # independently auditable assignment failure
            results[assignment_id] = exc
    return FleetReplanOutcome(
        proposal=proposal,
        compilations=results,
        new_fleet_plan_version=expected_version,
    )


__all__ = [
    "FleetReplanOutcome",
    "FleetReplanRequest",
    "PlanningRecoveryCoordinator",
    "PlanningRecoveryDecision",
    "PlanningRecoveryError",
    "RecoveryDisposition",
    "execute_fleet_replan",
]
