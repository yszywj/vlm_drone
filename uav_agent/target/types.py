"""Pure-Python target lifecycle value types.

The types in this module deliberately contain only scalar values, enums, and
fixed-size tuples.  They do not carry frames, detector internals, or references
to an environment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Target semantics supplied by the mission, without privileged geometry."""

    description: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "description",
            _non_empty_string(self.description, "TargetSpec.description"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {"description": self.description}


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
