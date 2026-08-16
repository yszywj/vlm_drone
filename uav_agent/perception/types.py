"""Pure value types for the visual target-confirmation boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative(value: object, name: str) -> float:
    normalized = _finite(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _confidence(value: object, name: str = "confidence") -> float:
    normalized = _finite(value, name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return normalized


def _positive_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return normalized


def _optional_vector3(
    value: object,
    name: str,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    normalized = tuple(
        _finite(component, f"{name}[{index}]")
        for index, component in enumerate(value)
    )
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    """One detector proposal; it is not yet an identified mission target."""

    candidate_id: str
    timestamp_s: float
    confidence: float
    source: str = "detector"
    estimated_position: tuple[float, float, float] | None = None
    estimated_velocity: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "source", _text(self.source, "source"))
        if self.source.casefold() == "oracle":
            raise ValueError("a visual DetectionCandidate cannot use source='oracle'")
        object.__setattr__(
            self,
            "estimated_position",
            _optional_vector3(self.estimated_position, "estimated_position"),
        )
        object.__setattr__(
            self,
            "estimated_velocity",
            _optional_vector3(self.estimated_velocity, "estimated_velocity"),
        )


@dataclass(frozen=True, slots=True)
class ShortTrackEvidence:
    """Temporal evidence that a detector candidate forms a stable short track."""

    candidate_id: str
    timestamp_s: float
    observation_count: int
    duration_s: float
    stable: bool
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(
            self,
            "observation_count",
            _positive_count(self.observation_count, "observation_count"),
        )
        object.__setattr__(self, "duration_s", _nonnegative(self.duration_s, "duration_s"))
        if not isinstance(self.stable, bool):
            raise TypeError("stable must be bool")
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True, slots=True)
class SemanticVerification:
    """VLM result for matching the mission's semantic target description."""

    candidate_id: str
    timestamp_s: float
    target_description: str
    matches: bool
    confidence: float
    verifier: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(
            self,
            "target_description",
            _text(self.target_description, "target_description"),
        )
        if not isinstance(self.matches, bool):
            raise TypeError("matches must be bool")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "verifier", _text(self.verifier, "verifier"))
        if self.verifier.casefold() == "oracle":
            raise ValueError("visual semantic verification cannot use Oracle")


@dataclass(frozen=True, slots=True)
class IdentityConsistencyEvidence:
    """ReID and temporal-consistency result for an already tracked candidate."""

    candidate_id: str
    target_id: str
    timestamp_s: float
    reidentified: bool
    temporally_consistent: bool
    consistent_observations: int
    confidence: float
    source: str = "reid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        if not isinstance(self.reidentified, bool):
            raise TypeError("reidentified must be bool")
        if not isinstance(self.temporally_consistent, bool):
            raise TypeError("temporally_consistent must be bool")
        object.__setattr__(
            self,
            "consistent_observations",
            _positive_count(self.consistent_observations, "consistent_observations"),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "source", _text(self.source, "source"))
        if self.source.casefold() == "oracle":
            raise ValueError("visual identity confirmation cannot use Oracle")


__all__ = [
    "DetectionCandidate",
    "IdentityConsistencyEvidence",
    "SemanticVerification",
    "ShortTrackEvidence",
]
