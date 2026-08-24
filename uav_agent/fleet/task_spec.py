"""Strict, simulator-free semantic task contract for Fleet planning.

``FleetTaskSpecV1`` is the only output of mission interpretation.  It records
what the user asked for, not how a UAV controller should execute it.  The
contract therefore has no Skill steps, velocities, PID gains, motor commands,
camera payloads, or privileged Oracle state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import TypeAlias

from common.ids import validate_routing_id, validate_uav_id
from planner.spatial import (
    CoordinateFrame,
    RegionSpec,
    SpatialTarget,
    region_spec_from_dict,
    spatial_target_from_dict,
)


MAX_GOALS = 32
MAX_ASSIGNMENT_CONSTRAINTS = 32
MAX_ORDERING_CONSTRAINTS = 64
MAX_TERMINATION_GOALS = 16
MAX_AMBIGUITIES = 16
MAX_SOURCE_EVIDENCE = 64


class FleetTaskSpecError(ValueError):
    """Raised when interpreted task semantics cross an unsafe boundary."""


class ConstraintStrength(str, Enum):
    MUST = "MUST"
    PREFER = "PREFER"
    OPEN = "OPEN"


class GoalType(str, Enum):
    SEARCH_TARGET = "SEARCH_TARGET"
    TRACK_TARGET = "TRACK_TARGET"
    INSPECT_TARGET = "INSPECT_TARGET"
    NAVIGATE = "NAVIGATE"
    RETURN_HOME = "RETURN_HOME"
    LAND = "LAND"
    RETURN_HOME_AND_LAND = "RETURN_HOME_AND_LAND"
    WAIT = "WAIT"
    REPORT = "REPORT"


MISSION_GOAL_TYPES = (
    GoalType.SEARCH_TARGET,
    GoalType.TRACK_TARGET,
    GoalType.INSPECT_TARGET,
    GoalType.NAVIGATE,
)
TERMINATION_GOAL_TYPES = (
    GoalType.RETURN_HOME,
    GoalType.LAND,
    GoalType.RETURN_HOME_AND_LAND,
    GoalType.WAIT,
    GoalType.REPORT,
)


SpatialConstraint: TypeAlias = RegionSpec | SpatialTarget


_TARGET_GOAL_TYPES = frozenset(
    {GoalType.SEARCH_TARGET, GoalType.TRACK_TARGET, GoalType.INSPECT_TARGET}
)
_MISSION_TYPES = frozenset(MISSION_GOAL_TYPES)
_TERMINATION_TYPES = frozenset(TERMINATION_GOAL_TYPES)
_FORBIDDEN_KEYS = frozenset(
    {
        "velocity",
        "velocity_mps",
        "yaw_rate",
        "yaw_rate_rad_s",
        "pid",
        "kp",
        "ki",
        "kd",
        "motor",
        "motors",
        "motor_thrust",
        "thrust",
        "pwm",
        "oracle",
        "oracle_target_pose",
        "oracle_target_velocity",
        "oracle_target_visible",
        "oracle_target_id",
        "reasoning",
        "chain_of_thought",
        "analysis",
        "thoughts",
        "rationale",
        "camera_rgb",
        "rgb",
        "image",
        "images",
        "base64",
    }
)


def reject_forbidden_task_fields(value: object, path: str = "FleetTaskSpecV1") -> None:
    """Reject low-level, image, and privileged fields before normal parsing."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} field names must be strings")
            folded = key.casefold()
            if folded in _FORBIDDEN_KEYS or folded.startswith("oracle_target_"):
                raise FleetTaskSpecError(f"{path}.{key} is forbidden")
            reject_forbidden_task_fields(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_forbidden_task_fields(nested, f"{path}[{index}]")
    elif isinstance(value, float) and not isfinite(value):
        raise FleetTaskSpecError(f"{path} must not contain NaN or Infinity")


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    raise FleetTaskSpecError(
        f"{name} must be one of: " + ", ".join(item.value for item in enum_type)
    )


def _text(
    value: object,
    name: str,
    *,
    maximum: int = 512,
    preserve: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value if preserve else value.strip()
    if not normalized.strip() or len(normalized) > maximum:
        raise FleetTaskSpecError(
            f"{name} must contain between 1 and {maximum} characters"
        )
    if any(character == "\x00" for character in normalized):
        raise FleetTaskSpecError(f"{name} must not contain NUL characters")
    return normalized


def _optional_text(value: object, name: str, *, maximum: int = 128) -> str | None:
    if value is None:
        return None
    return _text(value, name, maximum=maximum)


def _finite_optional(
    value: object,
    name: str,
    *,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be null or a finite number")
    result = float(value)
    if not isfinite(result) or result <= 0.0 or result > maximum:
        raise FleetTaskSpecError(f"{name} must be within (0, {maximum}]")
    return result


def _text_tuple(
    value: object,
    name: str,
    *,
    maximum_items: int,
    maximum_chars: int = 128,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array of strings")
    if len(value) > maximum_items:
        raise FleetTaskSpecError(f"{name} must contain at most {maximum_items} items")
    result = tuple(
        _text(item, f"{name}[{index}]", maximum=maximum_chars)
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise FleetTaskSpecError(f"{name} must not contain duplicates")
    return result


def _exact(
    value: object,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    fields = frozenset(value)
    unknown = fields - required
    missing = required - fields
    if unknown:
        raise FleetTaskSpecError(
            f"{name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise FleetTaskSpecError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )
    return value


def _array(value: object, name: str, *, maximum: int) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    if len(value) > maximum:
        raise FleetTaskSpecError(f"{name} must contain at most {maximum} items")
    return tuple(value)


def _spatial_constraint(value: object) -> SpatialConstraint | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("spatial_constraint must be an object or null")
    if "shape" in value:
        return region_spec_from_dict(value)
    if "kind" in value:
        return spatial_target_from_dict(value)
    raise FleetTaskSpecError(
        "spatial_constraint must be a Spatial V3 RegionSpec or SpatialTarget"
    )


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    evidence_id: str
    quote: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            validate_routing_id(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(self, "quote", _text(self.quote, "quote", maximum=512))

    def to_dict(self) -> dict[str, object]:
        return {"evidence_id": self.evidence_id, "quote": self.quote}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SourceEvidence":
        data = _exact(
            value,
            name="SourceEvidence",
            required=frozenset({"evidence_id", "quote"}),
        )
        return cls(data["evidence_id"], data["quote"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class MissionGoal:
    goal_id: str
    goal_type: GoalType
    target_alias: str | None
    spatial_constraint: SpatialConstraint | None
    duration_s: float | None
    distance_m: float | None
    strength: ConstraintStrength
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "goal_id", validate_routing_id(self.goal_id, "goal_id")
        )
        goal_type = _enum(self.goal_type, GoalType, "goal_type")
        if goal_type not in _MISSION_TYPES:
            raise FleetTaskSpecError(
                f"{goal_type.value} is a termination goal and must be placed in "
                "FleetTaskSpecV1.termination_goals, not goals"
            )
        object.__setattr__(self, "goal_type", goal_type)
        target_alias = _optional_text(
            self.target_alias, "target_alias", maximum=64
        )
        if target_alias is not None:
            target_alias = validate_routing_id(target_alias, "target_alias")
        if goal_type in _TARGET_GOAL_TYPES and target_alias is None:
            raise FleetTaskSpecError(f"{goal_type.value} requires target_alias")
        object.__setattr__(self, "target_alias", target_alias)
        spatial_constraint = self.spatial_constraint
        if spatial_constraint is not None and not isinstance(
            spatial_constraint, RegionSpec | SpatialTarget
        ):
            raise TypeError(
                "spatial_constraint must be a Spatial V3 value or None"
            )
        if (
            goal_type is GoalType.SEARCH_TARGET
            and spatial_constraint is not None
            and not isinstance(spatial_constraint, RegionSpec)
        ):
            raise FleetTaskSpecError(
                "SEARCH_TARGET spatial_constraint must be a RegionSpec or None"
            )
        if goal_type is GoalType.NAVIGATE and not isinstance(
            spatial_constraint, SpatialTarget
        ):
            raise FleetTaskSpecError(
                "NAVIGATE requires a SpatialTarget spatial_constraint"
            )
        object.__setattr__(
            self,
            "duration_s",
            _finite_optional(self.duration_s, "duration_s", maximum=86400.0),
        )
        object.__setattr__(
            self,
            "distance_m",
            _finite_optional(self.distance_m, "distance_m", maximum=100000.0),
        )
        object.__setattr__(
            self, "strength", _enum(self.strength, ConstraintStrength, "strength")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _text_tuple(
                self.evidence_refs,
                "evidence_refs",
                maximum_items=16,
                maximum_chars=64,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type.value,
            "target_alias": self.target_alias,
            "spatial_constraint": (
                None
                if self.spatial_constraint is None
                else self.spatial_constraint.to_dict()
            ),
            "duration_s": self.duration_s,
            "distance_m": self.distance_m,
            "strength": self.strength.value,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MissionGoal":
        data = _exact(
            value,
            name="MissionGoal",
            required=frozenset(
                {
                    "goal_id",
                    "goal_type",
                    "target_alias",
                    "spatial_constraint",
                    "duration_s",
                    "distance_m",
                    "strength",
                    "evidence_refs",
                }
            ),
        )
        return cls(
            goal_id=data["goal_id"],  # type: ignore[arg-type]
            goal_type=data["goal_type"],  # type: ignore[arg-type]
            target_alias=data["target_alias"],  # type: ignore[arg-type]
            spatial_constraint=_spatial_constraint(data["spatial_constraint"]),
            duration_s=data["duration_s"],  # type: ignore[arg-type]
            distance_m=data["distance_m"],  # type: ignore[arg-type]
            strength=data["strength"],  # type: ignore[arg-type]
            evidence_refs=_array(
                data["evidence_refs"], "evidence_refs", maximum=16
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AssignmentConstraint:
    constraint_id: str
    uav_id: str
    goal_ids: tuple[str, ...]
    strength: ConstraintStrength
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constraint_id",
            validate_routing_id(self.constraint_id, "constraint_id"),
        )
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        goals = _text_tuple(
            self.goal_ids, "goal_ids", maximum_items=MAX_GOALS, maximum_chars=64
        )
        if not goals:
            raise FleetTaskSpecError("AssignmentConstraint.goal_ids must not be empty")
        object.__setattr__(self, "goal_ids", goals)
        object.__setattr__(
            self, "strength", _enum(self.strength, ConstraintStrength, "strength")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _text_tuple(
                self.evidence_refs,
                "evidence_refs",
                maximum_items=16,
                maximum_chars=64,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "uav_id": self.uav_id,
            "goal_ids": list(self.goal_ids),
            "strength": self.strength.value,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AssignmentConstraint":
        data = _exact(
            value,
            name="AssignmentConstraint",
            required=frozenset(
                {"constraint_id", "uav_id", "goal_ids", "strength", "evidence_refs"}
            ),
        )
        return cls(
            constraint_id=data["constraint_id"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            goal_ids=_array(data["goal_ids"], "goal_ids", maximum=MAX_GOALS),  # type: ignore[arg-type]
            strength=data["strength"],  # type: ignore[arg-type]
            evidence_refs=_array(
                data["evidence_refs"], "evidence_refs", maximum=16
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class OrderingConstraint:
    constraint_id: str
    before_goal_id: str
    after_goal_id: str
    strength: ConstraintStrength
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("constraint_id", "before_goal_id", "after_goal_id"):
            object.__setattr__(
                self, name, validate_routing_id(getattr(self, name), name)
            )
        if self.before_goal_id == self.after_goal_id:
            raise FleetTaskSpecError("ordering endpoints must be different goals")
        object.__setattr__(
            self, "strength", _enum(self.strength, ConstraintStrength, "strength")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _text_tuple(
                self.evidence_refs,
                "evidence_refs",
                maximum_items=16,
                maximum_chars=64,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "before_goal_id": self.before_goal_id,
            "after_goal_id": self.after_goal_id,
            "strength": self.strength.value,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "OrderingConstraint":
        data = _exact(
            value,
            name="OrderingConstraint",
            required=frozenset(
                {
                    "constraint_id",
                    "before_goal_id",
                    "after_goal_id",
                    "strength",
                    "evidence_refs",
                }
            ),
        )
        return cls(
            constraint_id=data["constraint_id"],  # type: ignore[arg-type]
            before_goal_id=data["before_goal_id"],  # type: ignore[arg-type]
            after_goal_id=data["after_goal_id"],  # type: ignore[arg-type]
            strength=data["strength"],  # type: ignore[arg-type]
            evidence_refs=_array(
                data["evidence_refs"], "evidence_refs", maximum=16
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TerminationGoal:
    goal_id: str
    goal_type: GoalType
    uav_id: str | None
    duration_s: float | None
    strength: ConstraintStrength
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "goal_id", validate_routing_id(self.goal_id, "goal_id")
        )
        goal_type = _enum(self.goal_type, GoalType, "goal_type")
        if goal_type not in _TERMINATION_TYPES:
            raise FleetTaskSpecError(
                "TerminationGoal.goal_type must be RETURN_HOME, LAND, "
                "RETURN_HOME_AND_LAND, WAIT, or REPORT"
            )
        object.__setattr__(self, "goal_type", goal_type)
        if self.uav_id is not None:
            object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "duration_s",
            _finite_optional(self.duration_s, "duration_s", maximum=86400.0),
        )
        if goal_type is GoalType.WAIT and self.duration_s is None:
            raise FleetTaskSpecError("WAIT termination requires duration_s")
        object.__setattr__(
            self, "strength", _enum(self.strength, ConstraintStrength, "strength")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _text_tuple(
                self.evidence_refs,
                "evidence_refs",
                maximum_items=16,
                maximum_chars=64,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type.value,
            "uav_id": self.uav_id,
            "duration_s": self.duration_s,
            "strength": self.strength.value,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TerminationGoal":
        data = _exact(
            value,
            name="TerminationGoal",
            required=frozenset(
                {
                    "goal_id",
                    "goal_type",
                    "uav_id",
                    "duration_s",
                    "strength",
                    "evidence_refs",
                }
            ),
        )
        return cls(
            goal_id=data["goal_id"],  # type: ignore[arg-type]
            goal_type=data["goal_type"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            duration_s=data["duration_s"],  # type: ignore[arg-type]
            strength=data["strength"],  # type: ignore[arg-type]
            evidence_refs=_array(
                data["evidence_refs"], "evidence_refs", maximum=16
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TaskAmbiguity:
    ambiguity_id: str
    field_path: str
    description: str
    candidate_values: tuple[str, ...]
    resolution_required: bool
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ambiguity_id",
            validate_routing_id(self.ambiguity_id, "ambiguity_id"),
        )
        object.__setattr__(
            self, "field_path", _text(self.field_path, "field_path", maximum=128)
        )
        object.__setattr__(
            self,
            "description",
            _text(self.description, "description", maximum=512),
        )
        object.__setattr__(
            self,
            "candidate_values",
            _text_tuple(
                self.candidate_values,
                "candidate_values",
                maximum_items=8,
                maximum_chars=128,
            ),
        )
        if not isinstance(self.resolution_required, bool):
            raise TypeError("resolution_required must be bool")
        object.__setattr__(
            self,
            "evidence_refs",
            _text_tuple(
                self.evidence_refs,
                "evidence_refs",
                maximum_items=16,
                maximum_chars=64,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ambiguity_id": self.ambiguity_id,
            "field_path": self.field_path,
            "description": self.description,
            "candidate_values": list(self.candidate_values),
            "resolution_required": self.resolution_required,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TaskAmbiguity":
        data = _exact(
            value,
            name="TaskAmbiguity",
            required=frozenset(
                {
                    "ambiguity_id",
                    "field_path",
                    "description",
                    "candidate_values",
                    "resolution_required",
                    "evidence_refs",
                }
            ),
        )
        return cls(
            ambiguity_id=data["ambiguity_id"],  # type: ignore[arg-type]
            field_path=data["field_path"],  # type: ignore[arg-type]
            description=data["description"],  # type: ignore[arg-type]
            candidate_values=_array(
                data["candidate_values"], "candidate_values", maximum=8
            ),  # type: ignore[arg-type]
            resolution_required=data["resolution_required"],  # type: ignore[arg-type]
            evidence_refs=_array(
                data["evidence_refs"], "evidence_refs", maximum=16
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class FleetTaskSpecV1:
    source_text: str
    goals: tuple[MissionGoal, ...]
    assignment_constraints: tuple[AssignmentConstraint, ...] = ()
    ordering_constraints: tuple[OrderingConstraint, ...] = ()
    termination_goals: tuple[TerminationGoal, ...] = ()
    ambiguities: tuple[TaskAmbiguity, ...] = ()
    source_evidence: tuple[SourceEvidence, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise FleetTaskSpecError("schema_version must equal integer 1")
        object.__setattr__(
            self,
            "source_text",
            _text(self.source_text, "source_text", maximum=8192, preserve=True),
        )
        collections = (
            ("goals", self.goals, MissionGoal, MAX_GOALS),
            (
                "assignment_constraints",
                self.assignment_constraints,
                AssignmentConstraint,
                MAX_ASSIGNMENT_CONSTRAINTS,
            ),
            (
                "ordering_constraints",
                self.ordering_constraints,
                OrderingConstraint,
                MAX_ORDERING_CONSTRAINTS,
            ),
            (
                "termination_goals",
                self.termination_goals,
                TerminationGoal,
                MAX_TERMINATION_GOALS,
            ),
            ("ambiguities", self.ambiguities, TaskAmbiguity, MAX_AMBIGUITIES),
            (
                "source_evidence",
                self.source_evidence,
                SourceEvidence,
                MAX_SOURCE_EVIDENCE,
            ),
        )
        for name, values, expected, maximum in collections:
            normalized = tuple(values)
            if len(normalized) > maximum or any(
                not isinstance(item, expected) for item in normalized
            ):
                raise FleetTaskSpecError(
                    f"{name} must contain at most {maximum} {expected.__name__} values"
                )
            object.__setattr__(self, name, normalized)
        if not self.goals and not self.termination_goals:
            raise FleetTaskSpecError(
                "FleetTaskSpecV1 requires at least one mission or termination goal"
            )
        self._validate_graph()

    def _validate_graph(self) -> None:
        all_goal_ids = [goal.goal_id for goal in self.goals] + [
            goal.goal_id for goal in self.termination_goals
        ]
        if len(all_goal_ids) != len(set(all_goal_ids)):
            raise FleetTaskSpecError("goal_id values must be globally unique")
        constraint_ids = [item.constraint_id for item in self.assignment_constraints]
        constraint_ids += [item.constraint_id for item in self.ordering_constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise FleetTaskSpecError("constraint_id values must be globally unique")
        ambiguity_ids = [item.ambiguity_id for item in self.ambiguities]
        if len(ambiguity_ids) != len(set(ambiguity_ids)):
            raise FleetTaskSpecError("ambiguity_id values must be unique")
        evidence_ids = [item.evidence_id for item in self.source_evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise FleetTaskSpecError("evidence_id values must be unique")
        known_goals = set(all_goal_ids)
        known_evidence = set(evidence_ids)
        for item in self.assignment_constraints:
            self._known_refs(item.goal_ids, known_goals, "assignment goal")
            self._known_refs(item.evidence_refs, known_evidence, "source evidence")
        for item in self.ordering_constraints:
            self._known_refs(
                (item.before_goal_id, item.after_goal_id), known_goals, "ordering goal"
            )
            self._known_refs(item.evidence_refs, known_evidence, "source evidence")
        for item in (*self.goals, *self.termination_goals, *self.ambiguities):
            self._known_refs(item.evidence_refs, known_evidence, "source evidence")
        for evidence in self.source_evidence:
            if evidence.quote not in self.source_text:
                raise FleetTaskSpecError(
                    f"source evidence {evidence.evidence_id!r} is not verbatim source_text"
                )

    @staticmethod
    def _known_refs(values: Sequence[str], known: set[str], name: str) -> None:
        unknown = sorted(set(values) - known)
        if unknown:
            raise FleetTaskSpecError(
                f"unknown {name} reference(s): {', '.join(unknown)}"
            )

    @property
    def all_goal_ids(self) -> tuple[str, ...]:
        return tuple(goal.goal_id for goal in self.goals) + tuple(
            goal.goal_id for goal in self.termination_goals
        )

    def goal(self, goal_id: str) -> MissionGoal | TerminationGoal:
        normalized = validate_routing_id(goal_id, "goal_id")
        for goal in (*self.goals, *self.termination_goals):
            if goal.goal_id == normalized:
                return goal
        raise FleetTaskSpecError(f"unknown goal_id: {normalized}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_text": self.source_text,
            "goals": [item.to_dict() for item in self.goals],
            "assignment_constraints": [
                item.to_dict() for item in self.assignment_constraints
            ],
            "ordering_constraints": [
                item.to_dict() for item in self.ordering_constraints
            ],
            "termination_goals": [item.to_dict() for item in self.termination_goals],
            "ambiguities": [item.to_dict() for item in self.ambiguities],
            "source_evidence": [item.to_dict() for item in self.source_evidence],
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        trusted_uav_ids: Sequence[str] | None = None,
        trusted_target_aliases: Sequence[str] | None = None,
        supported_coordinate_frames: Sequence[CoordinateFrame | str] | None = None,
        expected_source_text: str | None = None,
    ) -> "FleetTaskSpecV1":
        return parse_fleet_task_spec(
            value,
            trusted_uav_ids=trusted_uav_ids,
            trusted_target_aliases=trusted_target_aliases,
            supported_coordinate_frames=supported_coordinate_frames,
            expected_source_text=expected_source_text,
        )


def _normalize_frames(
    values: Sequence[CoordinateFrame | str] | None,
) -> frozenset[CoordinateFrame] | None:
    if values is None:
        return None
    result: set[CoordinateFrame] = set()
    for value in values:
        try:
            result.add(
                value if isinstance(value, CoordinateFrame) else CoordinateFrame(value)
            )
        except (TypeError, ValueError):
            raise FleetTaskSpecError(
                "supported_coordinate_frames contains an invalid frame"
            ) from None
    if not result:
        raise FleetTaskSpecError("supported_coordinate_frames must not be empty")
    return frozenset(result)


def validate_task_spec_trust(
    task_spec: FleetTaskSpecV1,
    *,
    trusted_uav_ids: Sequence[str] | None = None,
    trusted_target_aliases: Sequence[str] | None = None,
    supported_coordinate_frames: Sequence[CoordinateFrame | str] | None = None,
    expected_source_text: str | None = None,
) -> FleetTaskSpecV1:
    """Apply request-bound entity/frame allow-lists after structural parsing."""

    if not isinstance(task_spec, FleetTaskSpecV1):
        raise TypeError("task_spec must be a FleetTaskSpecV1")
    if expected_source_text is not None and task_spec.source_text != expected_source_text:
        raise FleetTaskSpecError("source_text must exactly echo the original instruction")
    if trusted_uav_ids is not None:
        trusted_uavs = {validate_uav_id(value) for value in trusted_uav_ids}
        referenced_uavs = {
            item.uav_id for item in task_spec.assignment_constraints
        } | {
            item.uav_id
            for item in task_spec.termination_goals
            if item.uav_id is not None
        }
        unknown = sorted(referenced_uavs - trusted_uavs)
        if unknown:
            raise FleetTaskSpecError(
                "task spec references unknown UAV(s): " + ", ".join(unknown)
            )
    if trusted_target_aliases is not None:
        trusted_targets = {
            validate_routing_id(value, "target_alias")
            for value in trusted_target_aliases
        }
        unknown = sorted(
            {
                goal.target_alias
                for goal in task_spec.goals
                if goal.target_alias is not None
            }
            - trusted_targets
        )
        if unknown:
            raise FleetTaskSpecError(
                "task spec references unknown target(s): " + ", ".join(unknown)
            )
    supported_frames = _normalize_frames(supported_coordinate_frames)
    if supported_frames is not None:
        used: set[CoordinateFrame] = set()
        for goal in task_spec.goals:
            spatial = goal.spatial_constraint
            if spatial is None:
                continue
            frame = getattr(spatial, "frame", None)
            if frame is not None:
                used.add(frame)
        unknown_frames = sorted(frame.value for frame in used - supported_frames)
        if unknown_frames:
            raise FleetTaskSpecError(
                "task spec uses unsupported coordinate frame(s): "
                + ", ".join(unknown_frames)
            )
    return task_spec


def parse_fleet_task_spec(
    value: object,
    *,
    trusted_uav_ids: Sequence[str] | None = None,
    trusted_target_aliases: Sequence[str] | None = None,
    supported_coordinate_frames: Sequence[CoordinateFrame | str] | None = None,
    expected_source_text: str | None = None,
) -> FleetTaskSpecV1:
    reject_forbidden_task_fields(value)
    data = _exact(
        value,
        name="FleetTaskSpecV1",
        required=frozenset(
            {
                "schema_version",
                "source_text",
                "goals",
                "assignment_constraints",
                "ordering_constraints",
                "termination_goals",
                "ambiguities",
                "source_evidence",
            }
        ),
    )
    raw_goals = _array(data["goals"], "goals", maximum=MAX_GOALS)
    raw_assignments = _array(
        data["assignment_constraints"],
        "assignment_constraints",
        maximum=MAX_ASSIGNMENT_CONSTRAINTS,
    )
    raw_ordering = _array(
        data["ordering_constraints"],
        "ordering_constraints",
        maximum=MAX_ORDERING_CONSTRAINTS,
    )
    raw_termination = _array(
        data["termination_goals"],
        "termination_goals",
        maximum=MAX_TERMINATION_GOALS,
    )
    raw_ambiguities = _array(
        data["ambiguities"], "ambiguities", maximum=MAX_AMBIGUITIES
    )
    raw_evidence = _array(
        data["source_evidence"],
        "source_evidence",
        maximum=MAX_SOURCE_EVIDENCE,
    )
    spec = FleetTaskSpecV1(
        schema_version=data["schema_version"],  # type: ignore[arg-type]
        source_text=data["source_text"],  # type: ignore[arg-type]
        goals=tuple(MissionGoal.from_dict(item) for item in raw_goals),  # type: ignore[arg-type]
        assignment_constraints=tuple(
            AssignmentConstraint.from_dict(item) for item in raw_assignments  # type: ignore[arg-type]
        ),
        ordering_constraints=tuple(
            OrderingConstraint.from_dict(item) for item in raw_ordering  # type: ignore[arg-type]
        ),
        termination_goals=tuple(
            TerminationGoal.from_dict(item) for item in raw_termination  # type: ignore[arg-type]
        ),
        ambiguities=tuple(
            TaskAmbiguity.from_dict(item) for item in raw_ambiguities  # type: ignore[arg-type]
        ),
        source_evidence=tuple(
            SourceEvidence.from_dict(item) for item in raw_evidence  # type: ignore[arg-type]
        ),
    )
    return validate_task_spec_trust(
        spec,
        trusted_uav_ids=trusted_uav_ids,
        trusted_target_aliases=trusted_target_aliases,
        supported_coordinate_frames=supported_coordinate_frames,
        expected_source_text=expected_source_text,
    )


__all__ = [
    "AssignmentConstraint",
    "ConstraintStrength",
    "FleetTaskSpecError",
    "FleetTaskSpecV1",
    "GoalType",
    "MAX_AMBIGUITIES",
    "MAX_ASSIGNMENT_CONSTRAINTS",
    "MAX_GOALS",
    "MAX_ORDERING_CONSTRAINTS",
    "MAX_SOURCE_EVIDENCE",
    "MAX_TERMINATION_GOALS",
    "MISSION_GOAL_TYPES",
    "MissionGoal",
    "OrderingConstraint",
    "SourceEvidence",
    "SpatialConstraint",
    "TaskAmbiguity",
    "TERMINATION_GOAL_TYPES",
    "TerminationGoal",
    "parse_fleet_task_spec",
    "reject_forbidden_task_fields",
    "validate_task_spec_trust",
]
