"""Bounded, auditable state for assignment-level planning and repair.

This module is deliberately independent from Isaac Sim and model clients.  It
tracks *planning* lifecycle only; flight lifecycle remains owned by
``FleetMissionRuntime`` and ``MissionAgent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from common.ids import validate_routing_id, validate_uav_id


class PlanningStateError(RuntimeError):
    """Raised when a planning lifecycle transition is not legal."""


class AssignmentPlanningState(str, Enum):
    PENDING = "PENDING"
    INTERPRETING = "INTERPRETING"
    PLANNING = "PLANNING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    REPAIRING = "REPAIRING"
    DEGRADED_EXECUTABLE = "DEGRADED_EXECUTABLE"
    RUNNING = "RUNNING"
    HOLDING = "HOLDING"
    WAITING_REASSIGNMENT = "WAITING_REASSIGNMENT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class PlanningStage(str, Enum):
    MISSION_INTERPRETATION = "MISSION_INTERPRETATION"
    FLEET_PLANNING = "FLEET_PLANNING"
    LOCAL_PLANNING = "LOCAL_PLANNING"


class PlanningAttemptOutcome(str, Enum):
    IN_FLIGHT = "IN_FLIGHT"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    DEGRADED_ACCEPTED = "DEGRADED_ACCEPTED"


@dataclass(frozen=True, slots=True)
class PlanningRepairBudget:
    interpreter: int = 1
    fleet_plan: int = 2
    local_plan: int = 2

    def __post_init__(self) -> None:
        for name in ("interpreter", "fleet_plan", "local_plan"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} repair budget must be a non-negative integer")

    def for_stage(self, stage: PlanningStage | str) -> int:
        normalized = PlanningStage(stage)
        return {
            PlanningStage.MISSION_INTERPRETATION: self.interpreter,
            PlanningStage.FLEET_PLANNING: self.fleet_plan,
            PlanningStage.LOCAL_PLANNING: self.local_plan,
        }[normalized]


@dataclass(frozen=True, slots=True)
class PlanningAttempt:
    attempt_id: str
    proposal_id: str
    stage: PlanningStage
    plan_version: int
    timestamp_s: float
    outcome: PlanningAttemptOutcome = PlanningAttemptOutcome.IN_FLIGHT
    assignment_id: str | None = None
    uav_id: str | None = None
    repair_of_proposal_id: str | None = None
    finding_codes: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("schema_version must equal integer 1")
        object.__setattr__(
            self, "attempt_id", validate_routing_id(self.attempt_id, "attempt_id")
        )
        object.__setattr__(
            self, "proposal_id", validate_routing_id(self.proposal_id, "proposal_id")
        )
        object.__setattr__(self, "stage", PlanningStage(self.stage))
        object.__setattr__(self, "outcome", PlanningAttemptOutcome(self.outcome))
        if isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int):
            raise TypeError("plan_version must be an integer")
        if self.plan_version <= 0:
            raise ValueError("plan_version must be positive")
        if (
            isinstance(self.timestamp_s, bool)
            or not isinstance(self.timestamp_s, (int, float))
            or not isfinite(float(self.timestamp_s))
            or float(self.timestamp_s) < 0.0
        ):
            raise ValueError("timestamp_s must be finite and non-negative")
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        if self.assignment_id is not None:
            object.__setattr__(
                self,
                "assignment_id",
                validate_routing_id(self.assignment_id, "assignment_id"),
            )
        if self.uav_id is not None:
            object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if self.repair_of_proposal_id is not None:
            object.__setattr__(
                self,
                "repair_of_proposal_id",
                validate_routing_id(
                    self.repair_of_proposal_id, "repair_of_proposal_id"
                ),
            )
        codes = tuple(self.finding_codes)
        if len(codes) > 64 or any(
            not isinstance(code, str) or not code.strip() or len(code) > 128
            for code in codes
        ):
            raise ValueError("finding_codes must contain at most 64 bounded strings")
        object.__setattr__(self, "finding_codes", codes)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "attempt_id": self.attempt_id,
            "proposal_id": self.proposal_id,
            "stage": self.stage.value,
            "plan_version": self.plan_version,
            "timestamp_s": self.timestamp_s,
            "outcome": self.outcome.value,
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "repair_of_proposal_id": self.repair_of_proposal_id,
            "finding_codes": list(self.finding_codes),
        }


@dataclass(frozen=True, slots=True)
class AssignmentPlanningRecord:
    assignment_id: str
    uav_id: str
    state: AssignmentPlanningState = AssignmentPlanningState.PENDING
    plan_version: int = 1
    repair_attempts: Mapping[PlanningStage, int] = field(default_factory=dict)
    uncovered_goal_ids: tuple[str, ...] = ()
    last_error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assignment_id",
            validate_routing_id(self.assignment_id, "assignment_id"),
        )
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "state", AssignmentPlanningState(self.state))
        if isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int):
            raise TypeError("plan_version must be an integer")
        if self.plan_version <= 0:
            raise ValueError("plan_version must be positive")
        attempts: dict[PlanningStage, int] = {}
        for raw_stage, value in dict(self.repair_attempts).items():
            stage = PlanningStage(raw_stage)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("repair attempt counts must be non-negative integers")
            attempts[stage] = value
        object.__setattr__(self, "repair_attempts", MappingProxyType(attempts))
        goals = tuple(
            validate_routing_id(value, "uncovered_goal_id")
            for value in self.uncovered_goal_ids
        )
        if len(goals) != len(set(goals)) or len(goals) > 64:
            raise ValueError("uncovered_goal_ids must be unique and bounded")
        object.__setattr__(self, "uncovered_goal_ids", goals)
        if self.last_error_code is not None and (
            not isinstance(self.last_error_code, str)
            or not self.last_error_code.strip()
            or len(self.last_error_code) > 128
        ):
            raise ValueError("last_error_code must be a bounded string or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "state": self.state.value,
            "plan_version": self.plan_version,
            "repair_attempts": {
                key.value: value for key, value in self.repair_attempts.items()
            },
            "uncovered_goal_ids": list(self.uncovered_goal_ids),
            "last_error_code": self.last_error_code,
        }


_ALLOWED_TRANSITIONS: Mapping[AssignmentPlanningState, frozenset[AssignmentPlanningState]] = {
    AssignmentPlanningState.PENDING: frozenset(
        {
            AssignmentPlanningState.INTERPRETING,
            AssignmentPlanningState.PLANNING,
            AssignmentPlanningState.CANCELED,
        }
    ),
    AssignmentPlanningState.INTERPRETING: frozenset(
        {
            AssignmentPlanningState.PLANNING,
            AssignmentPlanningState.REPAIRING,
            AssignmentPlanningState.FAILED,
        }
    ),
    AssignmentPlanningState.PLANNING: frozenset(
        {
            AssignmentPlanningState.VALIDATING,
            AssignmentPlanningState.REPAIRING,
            AssignmentPlanningState.WAITING_REASSIGNMENT,
            AssignmentPlanningState.FAILED,
        }
    ),
    AssignmentPlanningState.VALIDATING: frozenset(
        {
            AssignmentPlanningState.READY,
            AssignmentPlanningState.DEGRADED_EXECUTABLE,
            AssignmentPlanningState.REPAIRING,
            AssignmentPlanningState.WAITING_REASSIGNMENT,
            AssignmentPlanningState.FAILED,
        }
    ),
    AssignmentPlanningState.REPAIRING: frozenset(
        {
            AssignmentPlanningState.VALIDATING,
            AssignmentPlanningState.REPAIRING,
            AssignmentPlanningState.DEGRADED_EXECUTABLE,
            AssignmentPlanningState.WAITING_REASSIGNMENT,
            AssignmentPlanningState.FAILED,
        }
    ),
    AssignmentPlanningState.READY: frozenset(
        {AssignmentPlanningState.RUNNING, AssignmentPlanningState.CANCELED}
    ),
    AssignmentPlanningState.DEGRADED_EXECUTABLE: frozenset(
        {AssignmentPlanningState.RUNNING, AssignmentPlanningState.CANCELED}
    ),
    AssignmentPlanningState.RUNNING: frozenset(
        {
            AssignmentPlanningState.HOLDING,
            AssignmentPlanningState.WAITING_REASSIGNMENT,
            AssignmentPlanningState.SUCCEEDED,
            AssignmentPlanningState.FAILED,
            AssignmentPlanningState.CANCELED,
        }
    ),
    AssignmentPlanningState.HOLDING: frozenset(
        {
            AssignmentPlanningState.RUNNING,
            AssignmentPlanningState.WAITING_REASSIGNMENT,
            AssignmentPlanningState.FAILED,
            AssignmentPlanningState.CANCELED,
        }
    ),
    AssignmentPlanningState.WAITING_REASSIGNMENT: frozenset(
        {
            AssignmentPlanningState.PLANNING,
            AssignmentPlanningState.CANCELED,
            AssignmentPlanningState.FAILED,
        }
    ),
    AssignmentPlanningState.SUCCEEDED: frozenset(),
    AssignmentPlanningState.FAILED: frozenset(),
    AssignmentPlanningState.CANCELED: frozenset(),
}


class PlanningStateTracker:
    """Strict per-assignment planning state with monotonic repair versions."""

    def __init__(self) -> None:
        self._records: dict[str, AssignmentPlanningRecord] = {}
        self._attempts: list[PlanningAttempt] = []

    def register(self, assignment_id: str, uav_id: str) -> AssignmentPlanningRecord:
        record = AssignmentPlanningRecord(assignment_id=assignment_id, uav_id=uav_id)
        if record.assignment_id in self._records:
            raise PlanningStateError(f"duplicate assignment_id: {record.assignment_id}")
        self._records[record.assignment_id] = record
        return record

    def record(self, assignment_id: str) -> AssignmentPlanningRecord:
        normalized = validate_routing_id(assignment_id, "assignment_id")
        try:
            return self._records[normalized]
        except KeyError:
            raise PlanningStateError(f"unknown assignment_id: {normalized}") from None

    def transition(
        self,
        assignment_id: str,
        state: AssignmentPlanningState | str,
        *,
        error_code: str | None = None,
        uncovered_goal_ids: tuple[str, ...] | None = None,
    ) -> AssignmentPlanningRecord:
        current = self.record(assignment_id)
        target = AssignmentPlanningState(state)
        if target is not current.state and target not in _ALLOWED_TRANSITIONS[current.state]:
            raise PlanningStateError(
                f"invalid planning state transition {current.state.value} -> {target.value}"
            )
        updated = replace(
            current,
            state=target,
            last_error_code=error_code,
            uncovered_goal_ids=(
                current.uncovered_goal_ids
                if uncovered_goal_ids is None
                else uncovered_goal_ids
            ),
        )
        self._records[current.assignment_id] = updated
        return updated

    def begin_repair(
        self,
        assignment_id: str,
        stage: PlanningStage | str,
        *,
        budget: PlanningRepairBudget,
        error_code: str,
    ) -> AssignmentPlanningRecord:
        current = self.record(assignment_id)
        normalized_stage = PlanningStage(stage)
        used = current.repair_attempts.get(normalized_stage, 0)
        if used >= budget.for_stage(normalized_stage):
            raise PlanningStateError(
                f"repair budget exhausted for {normalized_stage.value}"
            )
        attempts = dict(current.repair_attempts)
        attempts[normalized_stage] = used + 1
        updated = replace(
            current,
            state=AssignmentPlanningState.REPAIRING,
            plan_version=current.plan_version + 1,
            repair_attempts=attempts,
            last_error_code=error_code,
        )
        self._records[current.assignment_id] = updated
        return updated

    def add_attempt(self, attempt: PlanningAttempt) -> None:
        if not isinstance(attempt, PlanningAttempt):
            raise TypeError("attempt must be a PlanningAttempt")
        if any(item.attempt_id == attempt.attempt_id for item in self._attempts):
            raise PlanningStateError(f"duplicate attempt_id: {attempt.attempt_id}")
        self._attempts.append(attempt)

    @property
    def attempts(self) -> tuple[PlanningAttempt, ...]:
        return tuple(self._attempts)

    def snapshot(self) -> Mapping[str, Mapping[str, object]]:
        return MappingProxyType(
            {
                key: MappingProxyType(record.to_dict())
                for key, record in sorted(self._records.items())
            }
        )


__all__ = [
    "AssignmentPlanningRecord",
    "AssignmentPlanningState",
    "PlanningAttempt",
    "PlanningAttemptOutcome",
    "PlanningRepairBudget",
    "PlanningStage",
    "PlanningStateError",
    "PlanningStateTracker",
]
