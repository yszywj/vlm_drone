"""Immutable structural types for executable linear Skill plans.

This module deliberately knows nothing about world coordinates or mission
semantics.  Those checks belong to the trusted plan compiler and the safety
supervisor.  The runtime representation only guarantees stable step identity,
safe output references, bounded recovery metadata, and serialization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from numbers import Real
import re

from common.ids import validate_mission_id, validate_uav_id
from skills.types import SkillName


_STEP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_STEP_OUTPUT_PATTERN = re.compile(
    r"^\$(?P<step_id>[a-z][a-z0-9_]{0,31})\.(?P<field>[a-z][a-z0-9_]*)$"
)
_LEGACY_SEARCH_TARGET_REF = "$SEARCH.result.target_id"


class TaskPlanError(ValueError):
    """Raised when the runtime representation of a task plan is invalid."""


def _skill_name(value: SkillName | str | object, *, field_name: str) -> SkillName:
    if isinstance(value, SkillName):
        return value
    if isinstance(value, str):
        try:
            return SkillName(value.upper())
        except ValueError as exc:
            raise TaskPlanError(f"unknown {field_name}: {value}") from exc
    raise TaskPlanError(f"{field_name} must be a SkillName or string")


def _step_id(value: object, *, field_name: str = "step_id") -> str:
    if not isinstance(value, str) or _STEP_ID_PATTERN.fullmatch(value) is None:
        raise TaskPlanError(
            f"{field_name} must match ^[a-z][a-z0-9_]{{0,31}}$"
        )
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TaskPlanError(f"{name} must be a finite number")
    parsed = float(value)
    if not isfinite(parsed):
        raise TaskPlanError(f"{name} must be a finite number")
    return parsed


@dataclass(frozen=True, slots=True)
class StepOutputRef:
    """A reference to one field produced by a prior planned step."""

    step_id: str
    field: str = "target_id"

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _step_id(self.step_id))
        if self.field != "target_id":
            raise TaskPlanError("StepOutputRef.field must be 'target_id'")

    @classmethod
    def from_string(cls, value: str) -> StepOutputRef:
        if not isinstance(value, str):
            raise TaskPlanError("step output reference must be a string")
        match = _STEP_OUTPUT_PATTERN.fullmatch(value)
        if match is None:
            raise TaskPlanError(
                "step output reference must use $<step_id>.target_id"
            )
        return cls(match.group("step_id"), match.group("field"))

    def to_string(self) -> str:
        return f"${self.step_id}.{self.field}"

    def __str__(self) -> str:
        return self.to_string()


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """Bounded internal recovery attached to a planned TRACK step."""

    skill: SkillName
    max_attempts: int
    search_radius_m: float
    timeout_s: float

    def __post_init__(self) -> None:
        skill = _skill_name(self.skill, field_name="recovery skill")
        if skill is not SkillName.REACQUIRE:
            raise TaskPlanError("RecoveryPolicy.skill must be REACQUIRE")
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise TaskPlanError("RecoveryPolicy.max_attempts must be an integer")
        if not 0 <= self.max_attempts <= 2:
            raise TaskPlanError("RecoveryPolicy.max_attempts must be between 0 and 2")
        radius = _finite_number(self.search_radius_m, "RecoveryPolicy.search_radius_m")
        timeout = _finite_number(self.timeout_s, "RecoveryPolicy.timeout_s")
        if not 3.0 <= radius <= 20.0:
            raise TaskPlanError(
                "RecoveryPolicy.search_radius_m must be between 3 and 20"
            )
        if not 5.0 <= timeout <= 60.0:
            raise TaskPlanError("RecoveryPolicy.timeout_s must be between 5 and 60")
        object.__setattr__(self, "skill", skill)
        object.__setattr__(self, "search_radius_m", radius)
        object.__setattr__(self, "timeout_s", timeout)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RecoveryPolicy:
        if not isinstance(value, Mapping):
            raise TaskPlanError("recovery must be a mapping")
        normalized = dict(value)
        required = {"skill", "max_attempts", "search_radius_m", "timeout_s"}
        unknown = sorted(set(normalized) - required)
        missing = sorted(required - set(normalized))
        if unknown:
            raise TaskPlanError(
                "unknown recovery field(s): " + ", ".join(unknown)
            )
        if missing:
            raise TaskPlanError(
                "missing recovery field(s): " + ", ".join(missing)
            )
        return cls(
            skill=_skill_name(normalized["skill"], field_name="recovery skill"),
            max_attempts=normalized["max_attempts"],  # type: ignore[arg-type]
            search_radius_m=normalized["search_radius_m"],  # type: ignore[arg-type]
            timeout_s=normalized["timeout_s"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "skill": self.skill.value,
            "max_attempts": self.max_attempts,
            "search_radius_m": self.search_radius_m,
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True, slots=True, init=False)
class TaskStep:
    """One typed, identified, top-level step in a linear task plan.

    The preferred constructor is ``TaskStep(step_id, skill, params,
    recovery=None)``.  ``TaskStep(skill, params)`` remains accepted for old
    callers and receives a deterministic standalone legacy id; plans parsed
    through :meth:`TaskPlan.from_dicts` use ``step_01``, ``step_02``, ... .
    """

    step_id: str
    skill: SkillName
    params: Mapping[str, object]
    recovery: RecoveryPolicy | None

    def __init__(
        self,
        step_id: str | SkillName,
        skill: SkillName | str | Mapping[str, object],
        params: Mapping[str, object] | None = None,
        recovery: RecoveryPolicy | Mapping[str, object] | None = None,
    ) -> None:
        # Backward-compatible two-positional-argument form:
        # TaskStep(SkillName.GOTO, {"position": ...}).
        if isinstance(skill, Mapping) and params is None:
            actual_skill = _skill_name(step_id, field_name="Skill name")
            actual_step_id = f"legacy_{actual_skill.value.lower()}"
            actual_params = skill
        else:
            actual_step_id = _step_id(step_id)
            actual_skill = _skill_name(skill, field_name="Skill name")
            actual_params = params

        if actual_skill is SkillName.REACQUIRE:
            raise TaskPlanError("REACQUIRE cannot be a top-level TaskStep")
        if not isinstance(actual_params, Mapping):
            raise TaskPlanError("TaskStep.params must be a mapping")
        if any(not isinstance(key, str) for key in actual_params):
            raise TaskPlanError("TaskStep.params keys must be strings")
        normalized_params = deepcopy(dict(actual_params))
        reserved = sorted({"id", "skill", "recovery"} & set(normalized_params))
        if reserved:
            raise TaskPlanError(
                "TaskStep.params contains reserved field(s): " + ", ".join(reserved)
            )
        if actual_skill is SkillName.TRACK:
            target = normalized_params.get("target_id")
            if (
                isinstance(target, str)
                and target != _LEGACY_SEARCH_TARGET_REF
                and target.startswith("$")
            ):
                normalized_params["target_id"] = StepOutputRef.from_string(target)

        if recovery is None:
            normalized_recovery = None
        elif isinstance(recovery, RecoveryPolicy):
            normalized_recovery = recovery
        elif isinstance(recovery, Mapping):
            normalized_recovery = RecoveryPolicy.from_dict(recovery)
        else:
            raise TaskPlanError("TaskStep.recovery must be a RecoveryPolicy or mapping")
        if normalized_recovery is not None and actual_skill is not SkillName.TRACK:
            raise TaskPlanError("recovery is only allowed on TRACK TaskStep values")

        object.__setattr__(self, "step_id", actual_step_id)
        object.__setattr__(self, "skill", actual_skill)
        object.__setattr__(self, "params", normalized_params)
        object.__setattr__(self, "recovery", normalized_recovery)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.step_id,
            "skill": self.skill.value,
        }
        result.update(_to_json_compatible(self.params))
        if self.recovery is not None:
            result["recovery"] = self.recovery.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class TaskPlan:
    """A non-empty, finite linear runtime plan.

    Ordering, count, world, and flight-safety rules intentionally live at the
    trusted compiler and supervisor boundaries, not in this transport type.
    """

    steps: tuple[TaskStep, ...]
    mission_id: str = "mission_legacy"
    uav_id: str = "uav_1"
    plan_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple) or not self.steps:
            raise TaskPlanError("TaskPlan.steps must be a non-empty tuple")
        if not all(isinstance(step, TaskStep) for step in self.steps):
            raise TaskPlanError("TaskPlan.steps must contain only TaskStep values")
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise TaskPlanError("TaskPlan step ids must be unique")
        if any(step.skill is SkillName.REACQUIRE for step in self.steps):
            raise TaskPlanError("TaskPlan must not contain top-level REACQUIRE")
        try:
            validate_mission_id(self.mission_id)
            validate_uav_id(self.uav_id)
        except (TypeError, ValueError) as exc:
            raise TaskPlanError(str(exc)) from exc
        if isinstance(self.plan_version, bool) or not isinstance(
            self.plan_version, int
        ) or self.plan_version <= 0:
            raise TaskPlanError("TaskPlan.plan_version must be a positive integer")

    @classmethod
    def from_dicts(
        cls,
        entries: Sequence[Mapping[str, object]],
        *,
        mission_id: str = "mission_legacy",
        uav_id: str = "uav_1",
        plan_version: int = 1,
    ) -> TaskPlan:
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise TaskPlanError("task plan must be a sequence of mappings")
        explicit_ids = {
            entry.get("id")
            for entry in entries
            if (
                isinstance(entry, Mapping)
                and "id" in entry
                and isinstance(entry.get("id"), str)
            )
        }
        generated_ids: set[str] = set()
        parsed: list[TaskStep] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise TaskPlanError(f"task plan entry {index} must be a mapping")
            if any(not isinstance(key, str) for key in entry):
                raise TaskPlanError(f"task plan entry {index} keys must be strings")
            normalized = deepcopy(dict(entry))
            if "skill" not in normalized:
                raise TaskPlanError(f"task plan entry {index} is missing skill")
            if "id" in normalized:
                step_id = normalized.pop("id")
            else:
                candidate_number = index + 1
                step_id = f"step_{candidate_number:02d}"
                while step_id in explicit_ids or step_id in generated_ids:
                    candidate_number += 1
                    step_id = f"step_{candidate_number:02d}"
                generated_ids.add(step_id)
            skill = normalized.pop("skill")
            recovery = normalized.pop("recovery", None)
            parsed.append(TaskStep(step_id, skill, normalized, recovery))
        return cls(
            tuple(parsed),
            mission_id=mission_id,
            uav_id=uav_id,
            plan_version=plan_version,
        )

    def to_dicts(self) -> list[dict[str, object]]:
        return [step.to_dict() for step in self.steps]

    def to_dict(self) -> dict[str, object]:
        """Return the routed runtime-plan envelope."""

        return {
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "steps": self.to_dicts(),
        }


def _to_json_compatible(value: object) -> object:
    if isinstance(value, StepOutputRef):
        return value.to_string()
    if isinstance(value, Enum):
        # Names are stable across both string-valued SkillName and auto-valued
        # control enums such as YawMode.  Numeric ``auto()`` values are an
        # implementation detail and must never leak into serialized plans.
        return value.name
    if isinstance(value, Mapping):
        return {
            str(key): _to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_json_compatible(getattr(value, field.name))
            for field in fields(value)
        }
    return deepcopy(value)


__all__ = [
    "RecoveryPolicy",
    "StepOutputRef",
    "TaskPlan",
    "TaskPlanError",
    "TaskStep",
]
