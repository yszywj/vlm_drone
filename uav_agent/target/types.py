"""Pure-Python target lifecycle value types.

The types in this module deliberately contain only scalar values, enums, and
fixed-size tuples.  They do not carry frames, detector internals, or references
to an environment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from numbers import Real


class TargetLifecycle(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    SEARCHING = "SEARCHING"
    CANDIDATE = "CANDIDATE"
    LOCKED = "LOCKED"
    TRACKING = "TRACKING"
    LOST = "LOST"
    REACQUIRING = "REACQUIRING"
    TERMINATED = "TERMINATED"


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _optional_vector3(
    value: object,
    field_name: str,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ValueError(f"{field_name} must contain exactly three finite numbers")
    normalized = tuple(
        _finite_number(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )
    return normalized  # type: ignore[return-value]


def _optional_confidence(value: object, field_name: str = "confidence") -> float | None:
    if value is None:
        return None
    normalized = _finite_number(value, field_name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1 inclusive")
    return normalized


def _optional_non_empty_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field_name)


_TARGET_TEXT_MAX_LENGTH = 512
_TARGET_LIST_MAX_ITEMS = 32


def _bounded_text(value: object, field_name: str) -> str:
    normalized = _non_empty_string(value, field_name)
    if len(normalized) > _TARGET_TEXT_MAX_LENGTH:
        raise ValueError(
            f"{field_name} must contain at most {_TARGET_TEXT_MAX_LENGTH} characters"
        )
    return normalized


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    if len(value) > _TARGET_LIST_MAX_ITEMS:
        raise ValueError(
            f"{field_name} must contain at most {_TARGET_LIST_MAX_ITEMS} items"
        )
    normalized = tuple(
        _bounded_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate items")
    return normalized


@dataclass(frozen=True, slots=True, init=False)
class TargetSpec:
    """Versioned target semantics without geometry or mutable identity drift.

    ``TargetSpec("moving target")`` and the legacy ``description=`` keyword
    remain accepted. Appearance observations are deliberately separated from
    the immutable mission identity and can only be changed by constructing a
    new value through :meth:`with_mutable_appearance_notes`.
    """

    original_description: str
    description: str
    category: str
    hard_attributes: tuple[str, ...]
    soft_attributes: tuple[str, ...]
    negative_constraints: tuple[str, ...]
    relation_constraints: tuple[str, ...]
    query_ladder: tuple[str, ...]
    inspection_questions: tuple[str, ...]
    immutable_identity_summary: str
    mutable_appearance_notes: tuple[str, ...]

    _SERIALIZED_FIELDS = frozenset(
        {
            "original_description",
            "category",
            "hard_attributes",
            "soft_attributes",
            "negative_constraints",
            "relation_constraints",
            "query_ladder",
            "inspection_questions",
            "immutable_identity_summary",
            "mutable_appearance_notes",
        }
    )

    def __init__(
        self,
        original_description: str | None = None,
        *,
        category: str = "unspecified",
        hard_attributes: Sequence[str] = (),
        soft_attributes: Sequence[str] = (),
        negative_constraints: Sequence[str] = (),
        relation_constraints: Sequence[str] = (),
        query_ladder: Sequence[str] = (),
        inspection_questions: Sequence[str] = (),
        immutable_identity_summary: str | None = None,
        mutable_appearance_notes: Sequence[str] = (),
        description: str | None = None,
    ) -> None:
        if original_description is None:
            original_description = description
        elif description is not None:
            if _bounded_text(description, "TargetSpec.description") != _bounded_text(
                original_description,
                "TargetSpec.original_description",
            ):
                raise ValueError(
                    "description and original_description must match when both are set"
                )
        if original_description is None:
            raise TypeError("TargetSpec requires original_description")
        original = _bounded_text(
            original_description,
            "TargetSpec.original_description",
        )
        identity = (
            original
            if immutable_identity_summary is None
            else _bounded_text(
                immutable_identity_summary,
                "TargetSpec.immutable_identity_summary",
            )
        )
        object.__setattr__(self, "original_description", original)
        object.__setattr__(self, "description", original)
        object.__setattr__(
            self,
            "category",
            _bounded_text(category, "TargetSpec.category"),
        )
        for field_name, value in (
            ("hard_attributes", hard_attributes),
            ("soft_attributes", soft_attributes),
            ("negative_constraints", negative_constraints),
            ("relation_constraints", relation_constraints),
            ("query_ladder", query_ladder),
            ("inspection_questions", inspection_questions),
            ("mutable_appearance_notes", mutable_appearance_notes),
        ):
            object.__setattr__(self, field_name, _text_tuple(value, field_name))
        object.__setattr__(self, "immutable_identity_summary", identity)

    def with_mutable_appearance_notes(
        self,
        notes: Sequence[str],
    ) -> "TargetSpec":
        """Return a copy with appearance notes, never changing identity fields."""

        return replace(
            self,
            mutable_appearance_notes=_text_tuple(
                notes,
                "TargetSpec.mutable_appearance_notes",
            ),
        )

    def append_appearance_note(self, note: str) -> "TargetSpec":
        """Append one bounded note while preserving immutable identity."""

        normalized = _bounded_text(note, "TargetSpec.mutable_appearance_note")
        if normalized in self.mutable_appearance_notes:
            return self
        return self.with_mutable_appearance_notes(
            (*self.mutable_appearance_notes, normalized)
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TargetSpec":
        """Parse the exact schema-v2 target identity object.

        This parser deliberately requires every field.  Compatibility defaults
        belong to the trusted v1-to-v2 migration path, never to Qwen response
        parsing.
        """

        if not isinstance(data, Mapping):
            raise TypeError("TargetSpec input must be a mapping")
        if any(not isinstance(key, str) for key in data):
            raise TypeError("TargetSpec field names must be strings")
        keys = frozenset(data)
        unknown = keys - cls._SERIALIZED_FIELDS
        if unknown:
            raise ValueError(
                "TargetSpec contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        missing = cls._SERIALIZED_FIELDS - keys
        if missing:
            raise ValueError(
                "TargetSpec is missing required fields: "
                + ", ".join(sorted(missing))
            )
        return cls(
            original_description=data["original_description"],
            category=data["category"],
            hard_attributes=data["hard_attributes"],
            soft_attributes=data["soft_attributes"],
            negative_constraints=data["negative_constraints"],
            relation_constraints=data["relation_constraints"],
            query_ladder=data["query_ladder"],
            inspection_questions=data["inspection_questions"],
            immutable_identity_summary=data["immutable_identity_summary"],
            mutable_appearance_notes=data["mutable_appearance_notes"],
        )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {
            "original_description": self.original_description,
            "category": self.category,
            "hard_attributes": list(self.hard_attributes),
            "soft_attributes": list(self.soft_attributes),
            "negative_constraints": list(self.negative_constraints),
            "relation_constraints": list(self.relation_constraints),
            "query_ladder": list(self.query_ladder),
            "inspection_questions": list(self.inspection_questions),
            "immutable_identity_summary": self.immutable_identity_summary,
            "mutable_appearance_notes": list(self.mutable_appearance_notes),
        }


@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    """Immutable snapshot of one logical target lifecycle."""

    target_id: str | None
    description: str
    lifecycle: TargetLifecycle
    confidence: float | None
    last_seen_position: tuple[float, float, float] | None
    last_seen_velocity: tuple[float, float, float] | None
    last_seen_time_s: float | None
    source: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, TargetLifecycle):
            raise TypeError("TargetSnapshot.lifecycle must be a TargetLifecycle")

        object.__setattr__(
            self,
            "description",
            _non_empty_string(
                self.description,
                "TargetSnapshot.description",
            ),
        )
        object.__setattr__(
            self,
            "target_id",
            _optional_non_empty_string(self.target_id, "TargetSnapshot.target_id"),
        )
        object.__setattr__(
            self,
            "confidence",
            _optional_confidence(self.confidence, "TargetSnapshot.confidence"),
        )
        object.__setattr__(
            self,
            "last_seen_position",
            _optional_vector3(
                self.last_seen_position,
                "TargetSnapshot.last_seen_position",
            ),
        )
        object.__setattr__(
            self,
            "last_seen_velocity",
            _optional_vector3(
                self.last_seen_velocity,
                "TargetSnapshot.last_seen_velocity",
            ),
        )
        if self.last_seen_time_s is not None:
            object.__setattr__(
                self,
                "last_seen_time_s",
                _finite_number(
                    self.last_seen_time_s,
                    "TargetSnapshot.last_seen_time_s",
                ),
            )
        object.__setattr__(
            self,
            "source",
            _optional_non_empty_string(self.source, "TargetSnapshot.source"),
        )

        if self.lifecycle is TargetLifecycle.UNINITIALIZED and any(
            value is not None
            for value in (
                self.target_id,
                self.confidence,
                self.last_seen_position,
                self.last_seen_velocity,
                self.last_seen_time_s,
                self.source,
            )
        ):
            raise ValueError("UNINITIALIZED target snapshot cannot contain target data")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {
            "target_id": self.target_id,
            "description": self.description,
            "lifecycle": self.lifecycle.value,
            "confidence": self.confidence,
            "last_seen_position": (
                list(self.last_seen_position)
                if self.last_seen_position is not None
                else None
            ),
            "last_seen_velocity": (
                list(self.last_seen_velocity)
                if self.last_seen_velocity is not None
                else None
            ),
            "last_seen_time_s": self.last_seen_time_s,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class TargetEvent:
    """One validated target lifecycle transition."""

    timestamp_s: float
    old_state: TargetLifecycle
    new_state: TargetLifecycle
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_s",
            _finite_number(self.timestamp_s, "TargetEvent.timestamp_s"),
        )
        if not isinstance(self.old_state, TargetLifecycle):
            raise TypeError("TargetEvent.old_state must be a TargetLifecycle")
        if not isinstance(self.new_state, TargetLifecycle):
            raise TypeError("TargetEvent.new_state must be a TargetLifecycle")
        if self.old_state is self.new_state:
            raise ValueError("TargetEvent must change lifecycle state")
        object.__setattr__(
            self,
            "reason",
            _non_empty_string(self.reason, "TargetEvent.reason"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {
            "timestamp_s": self.timestamp_s,
            "old_state": self.old_state.value,
            "new_state": self.new_state.value,
            "reason": self.reason,
        }
