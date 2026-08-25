"""Scalar-only protocols for deterministic target-attribute evidence.

These values deliberately contain no image arrays, encoded images, frame paths,
or simulator objects.  Pixels remain in ``CameraSample``/``FrameStore`` and only
the bounded result of an attribute measurement crosses this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Integral, Real

from common.ids import (
    validate_mission_id,
    validate_routing_id,
    validate_uav_id,
)
from common.provenance import is_privileged_oracle_source
from perception.runtime import PerceptionRuntimeProfile


ATTRIBUTE_SCHEMA_VERSION = 1


class AttributeDecision(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    PENDING = "PENDING"
    UNSUPPORTED = "UNSUPPORTED"


def _schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("schema_version must be an integer")
    if int(value) != ATTRIBUTE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must equal {ATTRIBUTE_SCHEMA_VERSION}"
        )
    return ATTRIBUTE_SCHEMA_VERSION


def _text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(
            f"{name} must be non-empty and have no surrounding whitespace"
        )
    if len(value) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _attribute_text(value: object, name: str) -> str:
    return _text(value, name, maximum=64).casefold()


def _optional_attribute_text(value: object, name: str) -> str | None:
    return None if value is None else _attribute_text(value, name)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _positive_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _tracker_id(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("tracker_id must be an integer or routing string")
    if isinstance(value, Integral):
        result = int(value)
        if result < 0:
            raise ValueError("tracker_id must be non-negative")
        return str(result)
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return str(int(value))
    return validate_routing_id(value, "tracker_id")


def _decision(value: object) -> AttributeDecision:
    if isinstance(value, AttributeDecision):
        return value
    if isinstance(value, str):
        try:
            return AttributeDecision(value)
        except ValueError:
            pass
    raise ValueError("decision must be a valid AttributeDecision")


def _runtime_profile(value: object) -> PerceptionRuntimeProfile:
    if isinstance(value, PerceptionRuntimeProfile):
        return value
    if isinstance(value, str):
        aliases = {
            "production": PerceptionRuntimeProfile.PRODUCTION,
            "oracle_evaluation": PerceptionRuntimeProfile.ORACLE_EVALUATION,
        }
        normalized = aliases.get(value.casefold())
        if normalized is not None:
            return normalized
        try:
            return PerceptionRuntimeProfile(value)
        except ValueError:
            pass
    raise ValueError("runtime_profile must be PRODUCTION or ORACLE_EVALUATION")


def _strict_mapping(
    value: object,
    name: str,
    fields: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    actual = frozenset(value)
    unknown = actual - fields
    missing = fields - actual
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    return value


@dataclass(frozen=True, slots=True)
class AttributeRequirement:
    """One immutable attribute assertion for one routed tracker candidate."""

    mission_id: str
    uav_id: str
    assignment_id: str
    candidate_id: str
    tracker_id: str | int
    attribute_name: str
    expected_value: str
    schema_version: int = ATTRIBUTE_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "schema_version",
            "mission_id",
            "uav_id",
            "assignment_id",
            "candidate_id",
            "tracker_id",
            "attribute_name",
            "expected_value",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "assignment_id",
            validate_routing_id(self.assignment_id, "assignment_id"),
        )
        object.__setattr__(
            self,
            "candidate_id",
            validate_routing_id(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(self, "tracker_id", _tracker_id(self.tracker_id))
        object.__setattr__(
            self,
            "attribute_name",
            _attribute_text(self.attribute_name, "attribute_name"),
        )
        object.__setattr__(
            self,
            "expected_value",
            _attribute_text(self.expected_value, "expected_value"),
        )

    @classmethod
    def from_dict(cls, value: object) -> "AttributeRequirement":
        raw = _strict_mapping(value, "AttributeRequirement", cls._FIELDS)
        return cls(**raw)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "assignment_id": self.assignment_id,
            "candidate_id": self.candidate_id,
            "tracker_id": self.tracker_id,
            "attribute_name": self.attribute_name,
            "expected_value": self.expected_value,
        }


@dataclass(frozen=True, slots=True)
class AttributeObservation:
    """One scalar RGB-D attribute measurement for one synchronized frame."""

    mission_id: str
    uav_id: str
    assignment_id: str
    candidate_id: str
    tracker_id: str | int
    timestamp_s: float
    attribute_name: str
    expected_value: str
    observed_value: str | None
    decision: AttributeDecision
    confidence: float
    observation_count: int
    duration_s: float
    valid_sample_ratio: float
    source: str
    reason_code: str
    runtime_profile: PerceptionRuntimeProfile = PerceptionRuntimeProfile.PRODUCTION
    schema_version: int = ATTRIBUTE_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "schema_version",
            "mission_id",
            "uav_id",
            "assignment_id",
            "candidate_id",
            "tracker_id",
            "timestamp_s",
            "attribute_name",
            "expected_value",
            "observed_value",
            "decision",
            "confidence",
            "observation_count",
            "duration_s",
            "valid_sample_ratio",
            "source",
            "reason_code",
            "runtime_profile",
        }
    )

    def __post_init__(self) -> None:
        _normalize_measurement(self)

    @classmethod
    def from_dict(cls, value: object) -> "AttributeObservation":
        raw = dict(_strict_mapping(value, "AttributeObservation", cls._FIELDS))
        return cls(**raw)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return _measurement_to_dict(self)


@dataclass(frozen=True, slots=True)
class AttributeEvidence:
    """Temporal attribute decision consumed by semantic confirmation."""

    mission_id: str
    uav_id: str
    assignment_id: str
    candidate_id: str
    tracker_id: str | int
    timestamp_s: float
    attribute_name: str
    expected_value: str
    observed_value: str | None
    decision: AttributeDecision
    confidence: float
    observation_count: int
    duration_s: float
    valid_sample_ratio: float
    source: str
    reason_code: str
    runtime_profile: PerceptionRuntimeProfile = PerceptionRuntimeProfile.PRODUCTION
    schema_version: int = ATTRIBUTE_SCHEMA_VERSION

    _FIELDS = AttributeObservation._FIELDS

    def __post_init__(self) -> None:
        _normalize_measurement(self)

    @classmethod
    def from_dict(cls, value: object) -> "AttributeEvidence":
        raw = dict(_strict_mapping(value, "AttributeEvidence", cls._FIELDS))
        return cls(**raw)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return _measurement_to_dict(self)


def _normalize_measurement(value: AttributeObservation | AttributeEvidence) -> None:
    object.__setattr__(value, "schema_version", _schema_version(value.schema_version))
    object.__setattr__(value, "mission_id", validate_mission_id(value.mission_id))
    object.__setattr__(value, "uav_id", validate_uav_id(value.uav_id))
    object.__setattr__(
        value,
        "assignment_id",
        validate_routing_id(value.assignment_id, "assignment_id"),
    )
    object.__setattr__(
        value,
        "candidate_id",
        validate_routing_id(value.candidate_id, "candidate_id"),
    )
    object.__setattr__(value, "tracker_id", _tracker_id(value.tracker_id))
    object.__setattr__(
        value,
        "timestamp_s",
        _finite_nonnegative(value.timestamp_s, "timestamp_s"),
    )
    object.__setattr__(
        value,
        "attribute_name",
        _attribute_text(value.attribute_name, "attribute_name"),
    )
    object.__setattr__(
        value,
        "expected_value",
        _attribute_text(value.expected_value, "expected_value"),
    )
    observed = _optional_attribute_text(value.observed_value, "observed_value")
    object.__setattr__(value, "observed_value", observed)
    decision = _decision(value.decision)
    object.__setattr__(value, "decision", decision)
    object.__setattr__(value, "confidence", _unit_interval(value.confidence, "confidence"))
    object.__setattr__(
        value,
        "observation_count",
        _positive_count(value.observation_count, "observation_count"),
    )
    object.__setattr__(
        value,
        "duration_s",
        _finite_nonnegative(value.duration_s, "duration_s"),
    )
    object.__setattr__(
        value,
        "valid_sample_ratio",
        _unit_interval(value.valid_sample_ratio, "valid_sample_ratio"),
    )
    source = _text(value.source, "source", maximum=128)
    object.__setattr__(value, "source", source)
    object.__setattr__(
        value,
        "reason_code",
        validate_routing_id(value.reason_code, "reason_code"),
    )
    profile = _runtime_profile(value.runtime_profile)
    object.__setattr__(value, "runtime_profile", profile)
    if isinstance(value, AttributeObservation):
        if value.observation_count != 1 or value.duration_s != 0.0:
            raise ValueError(
                "a single-frame AttributeObservation requires "
                "observation_count=1 and duration_s=0"
            )
    if profile is PerceptionRuntimeProfile.PRODUCTION and is_privileged_oracle_source(source):
        raise ValueError("production attribute evidence cannot use an Oracle source")
    if decision in {AttributeDecision.MATCH, AttributeDecision.MISMATCH} and observed is None:
        raise ValueError("MATCH/MISMATCH attribute evidence requires observed_value")
    if (
        decision in {AttributeDecision.MATCH, AttributeDecision.MISMATCH}
        and value.confidence <= 0.0
    ):
        raise ValueError("MATCH/MISMATCH attribute evidence requires positive confidence")
    if (
        decision in {AttributeDecision.MATCH, AttributeDecision.MISMATCH}
        and value.valid_sample_ratio <= 0.0
    ):
        raise ValueError(
            "MATCH/MISMATCH attribute evidence requires a positive valid_sample_ratio"
        )
    if decision is AttributeDecision.MATCH and observed != value.expected_value:
        raise ValueError("MATCH requires observed_value to equal expected_value")
    if decision is AttributeDecision.MISMATCH and observed == value.expected_value:
        raise ValueError("MISMATCH requires observed_value to differ from expected_value")
    if observed == "unknown" and decision is not AttributeDecision.PENDING:
        raise ValueError("unknown cannot be used as positive or negative evidence")


def _measurement_to_dict(
    value: AttributeObservation | AttributeEvidence,
) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "mission_id": value.mission_id,
        "uav_id": value.uav_id,
        "assignment_id": value.assignment_id,
        "candidate_id": value.candidate_id,
        "tracker_id": value.tracker_id,
        "timestamp_s": value.timestamp_s,
        "attribute_name": value.attribute_name,
        "expected_value": value.expected_value,
        "observed_value": value.observed_value,
        "decision": value.decision.value,
        "confidence": value.confidence,
        "observation_count": value.observation_count,
        "duration_s": value.duration_s,
        "valid_sample_ratio": value.valid_sample_ratio,
        "source": value.source,
        "reason_code": value.reason_code,
        "runtime_profile": value.runtime_profile.value,
    }


def _route_tuple(
    value: AttributeRequirement | AttributeObservation | AttributeEvidence,
) -> tuple[str, str, str, str, str, str, str]:
    return (
        value.mission_id,
        value.uav_id,
        value.assignment_id,
        value.candidate_id,
        value.tracker_id,
        value.attribute_name,
        value.expected_value,
    )


@dataclass(frozen=True, slots=True)
class AttributeVerificationBundle:
    """A strictly routed, strictly ordered temporal evidence bundle."""

    requirement: AttributeRequirement
    observations: tuple[AttributeObservation, ...]
    evidence: AttributeEvidence
    schema_version: int = ATTRIBUTE_SCHEMA_VERSION

    _FIELDS = frozenset(
        {"schema_version", "requirement", "observations", "evidence"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if not isinstance(self.requirement, AttributeRequirement):
            raise TypeError("requirement must be an AttributeRequirement")
        if isinstance(self.observations, (str, bytes)) or not isinstance(
            self.observations, Sequence
        ):
            raise TypeError("observations must be a sequence")
        observations = tuple(self.observations)
        if not observations:
            raise ValueError("observations must not be empty")
        if any(not isinstance(item, AttributeObservation) for item in observations):
            raise TypeError("observations must contain AttributeObservation values")
        if not isinstance(self.evidence, AttributeEvidence):
            raise TypeError("evidence must be AttributeEvidence")
        expected_route = _route_tuple(self.requirement)
        if any(_route_tuple(item) != expected_route for item in observations):
            raise ValueError(
                "candidate, tracker, UAV, Assignment, and attribute routing must match"
            )
        if _route_tuple(self.evidence) != expected_route:
            raise ValueError(
                "evidence candidate, tracker, UAV, Assignment, and attribute routing must match"
            )
        timestamps = tuple(item.timestamp_s for item in observations)
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("attribute observation timestamps must increase strictly")
        if self.evidence.timestamp_s != timestamps[-1]:
            raise ValueError("evidence timestamp must equal the newest observation")
        if self.evidence.observation_count != len(observations):
            raise ValueError("evidence observation_count must match observations")
        elapsed = timestamps[-1] - timestamps[0]
        if self.evidence.duration_s > elapsed + 1e-12:
            raise ValueError("evidence duration_s cannot exceed retained history")
        object.__setattr__(self, "observations", observations)

    @classmethod
    def from_dict(cls, value: object) -> "AttributeVerificationBundle":
        raw = _strict_mapping(value, "AttributeVerificationBundle", cls._FIELDS)
        observations_raw = raw["observations"]
        if isinstance(observations_raw, (str, bytes)) or not isinstance(
            observations_raw, Sequence
        ):
            raise TypeError("observations must be an array")
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            requirement=AttributeRequirement.from_dict(raw["requirement"]),
            observations=tuple(
                AttributeObservation.from_dict(item) for item in observations_raw
            ),
            evidence=AttributeEvidence.from_dict(raw["evidence"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "requirement": self.requirement.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "evidence": self.evidence.to_dict(),
        }


__all__ = [
    "ATTRIBUTE_SCHEMA_VERSION",
    "AttributeDecision",
    "AttributeEvidence",
    "AttributeObservation",
    "AttributeRequirement",
    "AttributeVerificationBundle",
]
