"""Strict, bounded validation reports used across Fleet planning stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real

from common.ids import (
    validate_mission_id,
    validate_routing_id,
    validate_uav_id,
)
from runtime.validation_codes import ValidationCode


_SCHEMA_VERSION = 1
_MAX_FINDINGS = 256
_MAX_EVIDENCE_REFS = 32
_MAX_MESSAGE_CHARS = 2048


class ValidationSeverity(str, Enum):
    WARNING = "WARNING"
    RECOVERABLE_SEMANTIC_ERROR = "RECOVERABLE_SEMANTIC_ERROR"
    HARD_ACTION_BLOCK = "HARD_ACTION_BLOCK"
    FATAL_SAFETY = "FATAL_SAFETY"


class RecoveryRecommendation(str, Enum):
    """Bounded recovery choices; it is not a controller command."""

    NONE = "NONE"
    REPAIR_LOCAL_PLAN = "REPAIR_LOCAL_PLAN"
    DEGRADE_EXECUTABLE = "DEGRADE_EXECUTABLE"
    HOLD_POSITION = "HOLD_POSITION"
    REQUEST_FLEET_REPLAN = "REQUEST_FLEET_REPLAN"
    CANCEL_AND_LAND = "CANCEL_AND_LAND"
    ABORT_FLEET = "ABORT_FLEET"


def _exact_object(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    keys = frozenset(value)
    unknown = keys - required - optional
    missing = required - keys
    if unknown:
        raise ValueError(
            f"{name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ValueError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )
    return value


def _schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("schema_version must be an integer")
    if value != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {_SCHEMA_VERSION}")
    return value


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timestamp must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError("timestamp must be a finite non-negative number")
    return result


def _text(value: object, name: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result or len(result) > maximum:
        raise ValueError(
            f"{name} must contain between 1 and {maximum} characters"
        )
    return result


def _optional_id(value: object, name: str) -> str | None:
    if value is None:
        return None
    return validate_routing_id(value, name)


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} is not supported") from None


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """One scoped finding; optional routing fields are explicit ``None``."""

    schema_version: int
    finding_id: str
    timestamp: float
    stage: str
    scope: str
    severity: ValidationSeverity
    code: ValidationCode
    message: str
    mission_id: str
    assignment_id: str | None
    uav_id: str | None
    goal_id: str | None
    step_id: str | None
    proposal_id: str | None
    evidence_refs: tuple[str, ...]
    recommended_action: RecoveryRecommendation

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(
            self,
            "finding_id",
            validate_routing_id(self.finding_id, "finding_id"),
        )
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        object.__setattr__(self, "stage", _text(self.stage, "stage"))
        object.__setattr__(self, "scope", _text(self.scope, "scope"))
        object.__setattr__(
            self,
            "severity",
            _enum(self.severity, ValidationSeverity, "severity"),
        )
        object.__setattr__(self, "code", _enum(self.code, ValidationCode, "code"))
        object.__setattr__(
            self,
            "message",
            _text(self.message, "message", maximum=_MAX_MESSAGE_CHARS),
        )
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(
            self,
            "assignment_id",
            _optional_id(self.assignment_id, "assignment_id"),
        )
        if self.uav_id is not None:
            object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        for name in ("goal_id", "step_id", "proposal_id"):
            object.__setattr__(self, name, _optional_id(getattr(self, name), name))
        if isinstance(self.evidence_refs, (str, bytes)) or not isinstance(
            self.evidence_refs, Sequence
        ):
            raise TypeError("evidence_refs must be an array")
        refs = tuple(
            validate_routing_id(item, f"evidence_refs[{index}]")
            for index, item in enumerate(self.evidence_refs)
        )
        if len(refs) > _MAX_EVIDENCE_REFS:
            raise ValueError(
                f"evidence_refs must contain at most {_MAX_EVIDENCE_REFS} values"
            )
        if len(refs) != len(set(refs)):
            raise ValueError("evidence_refs must not contain duplicates")
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(
            self,
            "recommended_action",
            _enum(
                self.recommended_action,
                RecoveryRecommendation,
                "recommended_action",
            ),
        )

    @classmethod
    def from_dict(cls, value: object) -> "ValidationFinding":
        fields = frozenset(
            {
                "schema_version", "finding_id", "timestamp", "stage", "scope",
                "severity", "code", "message", "mission_id", "assignment_id",
                "uav_id", "goal_id", "step_id", "proposal_id", "evidence_refs",
                "recommended_action",
            }
        )
        data = _exact_object(value, name="ValidationFinding", required=fields)
        refs = data["evidence_refs"]
        if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence):
            raise TypeError("evidence_refs must be an array")
        return cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            finding_id=data["finding_id"],  # type: ignore[arg-type]
            timestamp=data["timestamp"],  # type: ignore[arg-type]
            stage=data["stage"],  # type: ignore[arg-type]
            scope=data["scope"],  # type: ignore[arg-type]
            severity=data["severity"],  # type: ignore[arg-type]
            code=data["code"],  # type: ignore[arg-type]
            message=data["message"],  # type: ignore[arg-type]
            mission_id=data["mission_id"],  # type: ignore[arg-type]
            assignment_id=data["assignment_id"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            goal_id=data["goal_id"],  # type: ignore[arg-type]
            step_id=data["step_id"],  # type: ignore[arg-type]
            proposal_id=data["proposal_id"],  # type: ignore[arg-type]
            evidence_refs=tuple(refs),  # type: ignore[arg-type]
            recommended_action=data["recommended_action"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "finding_id": self.finding_id,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "scope": self.scope,
            "severity": self.severity.value,
            "code": self.code.value,
            "message": self.message,
            "mission_id": self.mission_id,
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "goal_id": self.goal_id,
            "step_id": self.step_id,
            "proposal_id": self.proposal_id,
            "evidence_refs": list(self.evidence_refs),
            "recommended_action": self.recommended_action.value,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregate verdict whose severity controls action admission."""

    schema_version: int
    report_id: str
    timestamp: float
    stage: str
    mission_id: str
    assignment_id: str | None
    uav_id: str | None
    findings: tuple[ValidationFinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(
            self,
            "report_id",
            validate_routing_id(self.report_id, "report_id"),
        )
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        object.__setattr__(self, "stage", _text(self.stage, "stage"))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(
            self,
            "assignment_id",
            _optional_id(self.assignment_id, "assignment_id"),
        )
        if self.uav_id is not None:
            object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if isinstance(self.findings, (str, bytes)) or not isinstance(
            self.findings, Sequence
        ):
            raise TypeError("findings must be an array")
        findings = tuple(self.findings)
        if len(findings) > _MAX_FINDINGS:
            raise ValueError(f"findings must contain at most {_MAX_FINDINGS} values")
        if any(not isinstance(item, ValidationFinding) for item in findings):
            raise TypeError("findings must contain ValidationFinding values")
        if len({item.finding_id for item in findings}) != len(findings):
            raise ValueError("finding_id values must be unique within a report")
        for item in findings:
            if item.mission_id != self.mission_id:
                raise ValueError("finding mission_id does not match report")
            if self.assignment_id is not None and item.assignment_id not in {
                None,
                self.assignment_id,
            }:
                raise ValueError("finding assignment_id does not match report")
            if self.uav_id is not None and item.uav_id not in {None, self.uav_id}:
                raise ValueError("finding uav_id does not match report")
        object.__setattr__(self, "findings", findings)

    @property
    def hard_blocked(self) -> bool:
        return any(
            item.severity
            in {ValidationSeverity.HARD_ACTION_BLOCK, ValidationSeverity.FATAL_SAFETY}
            for item in self.findings
        )

    @property
    def fatal(self) -> bool:
        return any(
            item.severity is ValidationSeverity.FATAL_SAFETY
            for item in self.findings
        )

    @property
    def executable(self) -> bool:
        """Semantic errors are repairable and do not admit blocked actions."""

        return not self.hard_blocked

    @property
    def semantically_valid(self) -> bool:
        return not any(
            item.severity is ValidationSeverity.RECOVERABLE_SEMANTIC_ERROR
            for item in self.findings
        )

    @property
    def accepted(self) -> bool:
        return self.executable and self.semantically_valid

    @classmethod
    def from_dict(cls, value: object) -> "ValidationReport":
        required = frozenset(
            {
                "schema_version", "report_id", "timestamp", "stage",
                "mission_id", "assignment_id", "uav_id", "findings",
            }
        )
        data = _exact_object(value, name="ValidationReport", required=required)
        raw_findings = data["findings"]
        if isinstance(raw_findings, (str, bytes)) or not isinstance(
            raw_findings, Sequence
        ):
            raise TypeError("findings must be an array")
        return cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            report_id=data["report_id"],  # type: ignore[arg-type]
            timestamp=data["timestamp"],  # type: ignore[arg-type]
            stage=data["stage"],  # type: ignore[arg-type]
            mission_id=data["mission_id"],  # type: ignore[arg-type]
            assignment_id=data["assignment_id"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            findings=tuple(
                ValidationFinding.from_dict(item) for item in raw_findings
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "mission_id": self.mission_id,
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "findings": [item.to_dict() for item in self.findings],
        }


__all__ = [
    "RecoveryRecommendation",
    "ValidationFinding",
    "ValidationReport",
    "ValidationSeverity",
]
