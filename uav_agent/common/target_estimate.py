"""Neutral, dependency-free target state shared by perception and Skills.

``TargetEstimate`` deliberately contains only small immutable Python values.
Images, tensors, masks, and model result objects stay behind their respective
perception boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Mapping

from common.ids import validate_routing_id


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _nonnegative(value: object, field_name: str) -> float:
    normalized = _finite(value, field_name)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized


def _optional_id(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return validate_routing_id(value, field_name)


def _optional_class_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("class_id must be a non-negative integer or None")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("class_id must be non-negative")
    return normalized


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty without surrounding whitespace")
    return value


def _optional_confidence(value: object) -> float | None:
    if value is None:
        return None
    normalized = _finite(value, "confidence")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError("confidence must be between zero and one")
    return normalized


def _optional_bbox(
    value: object,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("bbox_xyxy_normalized must be a four-number tuple or None")
    normalized = tuple(
        _finite(component, f"bbox_xyxy_normalized[{index}]")
        for index, component in enumerate(value)
    )
    if any(component < 0.0 or component > 1.0 for component in normalized):
        raise ValueError("bbox_xyxy_normalized components must be within [0, 1]")
    x1, y1, x2, y2 = normalized
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox_xyxy_normalized must satisfy x1 < x2 and y1 < y2")
    return x1, y1, x2, y2


def _optional_vector3(
    value: object,
    field_name: str,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    # Requiring a tuple (instead of accepting arbitrary sequences) keeps
    # ndarray/tensor payloads out of this otherwise serialization-safe type.
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{field_name} must be a three-number tuple or None")
    normalized = tuple(
        _finite(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TargetEstimate:
    """One perception-time estimate with explicit visibility and provenance."""

    timestamp_s: float
    target_id: str | None
    candidate_id: str | None
    tracker_id: str | None

    visible: bool
    confirmed: bool
    predicted_only: bool

    class_id: int | None
    class_name: str | None
    confidence: float | None

    bbox_xyxy_normalized: tuple[float, float, float, float] | None

    position_world_m: tuple[float, float, float] | None
    velocity_world_mps: tuple[float, float, float] | None

    measurement_age_s: float
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _nonnegative(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "target_id", _optional_id(self.target_id, "target_id"))
        object.__setattr__(
            self,
            "candidate_id",
            _optional_id(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(self, "tracker_id", _optional_id(self.tracker_id, "tracker_id"))
        for field_name in ("visible", "confirmed", "predicted_only"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        object.__setattr__(self, "class_id", _optional_class_id(self.class_id))
        object.__setattr__(self, "class_name", _optional_text(self.class_name, "class_name"))
        object.__setattr__(self, "confidence", _optional_confidence(self.confidence))
        bbox = _optional_bbox(self.bbox_xyxy_normalized)
        object.__setattr__(self, "bbox_xyxy_normalized", bbox)
        object.__setattr__(
            self,
            "position_world_m",
            _optional_vector3(self.position_world_m, "position_world_m"),
        )
        object.__setattr__(
            self,
            "velocity_world_mps",
            _optional_vector3(self.velocity_world_mps, "velocity_world_mps"),
        )
        object.__setattr__(
            self,
            "measurement_age_s",
            _nonnegative(self.measurement_age_s, "measurement_age_s"),
        )
        source = _optional_text(self.source, "source")
        assert source is not None
        object.__setattr__(self, "source", source)

        if self.visible and bbox is None:
            raise ValueError("visible=True requires bbox_xyxy_normalized")
        if self.confirmed and self.target_id is None:
            raise ValueError("confirmed=True requires a stable target_id")
        if self.predicted_only and self.visible:
            raise ValueError("predicted_only=True requires visible=False")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation without model-specific values."""

        return {
            "timestamp_s": self.timestamp_s,
            "target_id": self.target_id,
            "candidate_id": self.candidate_id,
            "tracker_id": self.tracker_id,
            "visible": self.visible,
            "confirmed": self.confirmed,
            "predicted_only": self.predicted_only,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy_normalized": (
                None
                if self.bbox_xyxy_normalized is None
                else list(self.bbox_xyxy_normalized)
            ),
            "position_world_m": (
                None if self.position_world_m is None else list(self.position_world_m)
            ),
            "velocity_world_mps": (
                None
                if self.velocity_world_mps is None
                else list(self.velocity_world_mps)
            ),
            "measurement_age_s": self.measurement_age_s,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TargetEstimate":
        """Strictly parse a mapping, rejecting missing and unknown fields."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                f"TargetEstimate fields mismatch; missing={missing}, unknown={unknown}"
            )
        values = dict(payload)
        for field_name in (
            "bbox_xyxy_normalized",
            "position_world_m",
            "velocity_world_mps",
        ):
            value = values[field_name]
            if isinstance(value, list):
                values[field_name] = tuple(value)
        return cls(**values)  # type: ignore[arg-type]


__all__ = ["TargetEstimate"]
