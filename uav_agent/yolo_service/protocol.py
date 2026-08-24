"""Strict, dependency-free wire types for the local Ultralytics service.

The service intentionally keeps model objects, tensors, file paths and image
locations out of the protocol.  A JPEG is carried as a multipart part while
the JSON metadata is parsed with the value objects in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from numbers import Integral, Real
from typing import Any

from common.ids import (
    validate_mission_id,
    validate_request_id,
    validate_routing_id,
    validate_uav_id,
)


SCHEMA_VERSION = 1


class ProtocolValidationError(ValueError):
    """Raised when untrusted JSON violates the public service schema."""


class RouteMismatchError(ProtocolValidationError):
    """Raised when a response does not belong to the submitted request."""


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ProtocolValidationError(f"{name} keys must be strings")
    return value


def _strict_fields(
    value: object,
    name: str,
    required: frozenset[str],
) -> Mapping[str, object]:
    obj = _object(value, name)
    keys = set(obj)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ProtocolValidationError(
            f"{name} is missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise ProtocolValidationError(
            f"{name} contains unknown fields: {', '.join(unknown)}"
        )
    return obj


def _schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProtocolValidationError("schema_version must be integer 1")
    if int(value) != SCHEMA_VERSION:
        raise ProtocolValidationError(
            f"unsupported schema_version {value!r}; expected {SCHEMA_VERSION}"
        )
    return SCHEMA_VERSION


def _routing_id(value: object, name: str) -> str:
    try:
        return validate_routing_id(value, name)
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(str(exc)) from exc


def _mission_id(value: object) -> str:
    try:
        return validate_mission_id(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(str(exc)) from exc


def _uav_id(value: object) -> str:
    try:
        return validate_uav_id(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(str(exc)) from exc


def _request_id(value: object) -> str:
    try:
        return validate_request_id(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(str(exc)) from exc


def _stream_id(value: object, mission_id: str, uav_id: str) -> str:
    expected = f"{mission_id}:{uav_id}"
    if not isinstance(value, str):
        raise ProtocolValidationError("stream_id must be a string")
    if value != expected:
        raise ProtocolValidationError(
            "stream_id must exactly equal '<mission_id>:<uav_id>'"
        )
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProtocolValidationError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ProtocolValidationError(f"{name} must be finite and non-negative")
    return result


def _confidence(value: object) -> float:
    result = _finite_nonnegative(value, "confidence")
    if result > 1.0:
        raise ProtocolValidationError("confidence must be within [0, 1]")
    return result


def _bounded_text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ProtocolValidationError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ProtocolValidationError(
            f"{name} must be non-empty and must not have surrounding whitespace"
        )
    if len(value) > maximum:
        raise ProtocolValidationError(f"{name} exceeds {maximum} characters")
    if any(ord(character) < 32 for character in value):
        raise ProtocolValidationError(f"{name} contains control characters")
    return value


def _integer(value: object, name: str, *, maximum: int = 2**31 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProtocolValidationError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0 or result > maximum:
        raise ProtocolValidationError(
            f"{name} must be a non-negative integer no greater than {maximum}"
        )
    return result


def _sequence(value: object, name: str, *, maximum: int) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise ProtocolValidationError(f"{name} must be a JSON array")
    if len(value) > maximum:
        raise ProtocolValidationError(f"{name} exceeds {maximum} entries")
    return value


@dataclass(frozen=True, slots=True)
class TargetQuery:
    """Audited closed-set IDs or YOLOE prompts, never model paths."""

    class_ids: tuple[int, ...] = ()
    text_prompts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        class_ids = tuple(
            _integer(value, "class_ids entry", maximum=65_535)
            for value in self.class_ids
        )
        prompts = tuple(
            _bounded_text(value, "text_prompts entry")
            for value in self.text_prompts
        )
        if len(class_ids) > 256:
            raise ProtocolValidationError("class_ids exceeds 256 entries")
        if len(prompts) > 64:
            raise ProtocolValidationError("text_prompts exceeds 64 entries")
        if len(set(class_ids)) != len(class_ids):
            raise ProtocolValidationError("class_ids must not contain duplicates")
        if len(set(prompts)) != len(prompts):
            raise ProtocolValidationError("text_prompts must not contain duplicates")
        if not class_ids and not prompts:
            raise ProtocolValidationError(
                "target_query must contain class_ids or text_prompts"
            )
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "text_prompts", prompts)

    @classmethod
    def from_dict(cls, value: object) -> "TargetQuery":
        raw = _strict_fields(
            value,
            "target_query",
            frozenset({"class_ids", "text_prompts"}),
        )
        class_ids = _sequence(raw["class_ids"], "class_ids", maximum=256)
        prompts = _sequence(raw["text_prompts"], "text_prompts", maximum=64)
        return cls(tuple(class_ids), tuple(prompts))

    def to_dict(self) -> dict[str, object]:
        return {
            "class_ids": list(self.class_ids),
            "text_prompts": list(self.text_prompts),
        }


@dataclass(frozen=True, slots=True)
class TrackRequest:
    schema_version: int
    request_id: str
    mission_id: str
    uav_id: str
    stream_id: str
    frame_id: str
    timestamp_s: float
    target_query: TargetQuery

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        mission_id = _mission_id(self.mission_id)
        uav_id = _uav_id(self.uav_id)
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        object.__setattr__(self, "mission_id", mission_id)
        object.__setattr__(self, "uav_id", uav_id)
        object.__setattr__(
            self,
            "stream_id",
            _stream_id(self.stream_id, mission_id, uav_id),
        )
        object.__setattr__(self, "frame_id", _routing_id(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "timestamp_s",
            _finite_nonnegative(self.timestamp_s, "timestamp_s"),
        )
        if not isinstance(self.target_query, TargetQuery):
            raise TypeError("target_query must be a TargetQuery")

    @classmethod
    def from_dict(cls, value: object) -> "TrackRequest":
        raw = _strict_fields(
            value,
            "track request",
            frozenset(
                {
                    "schema_version",
                    "request_id",
                    "mission_id",
                    "uav_id",
                    "stream_id",
                    "frame_id",
                    "timestamp_s",
                    "target_query",
                }
            ),
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            request_id=raw["request_id"],  # type: ignore[arg-type]
            mission_id=raw["mission_id"],  # type: ignore[arg-type]
            uav_id=raw["uav_id"],  # type: ignore[arg-type]
            stream_id=raw["stream_id"],  # type: ignore[arg-type]
            frame_id=raw["frame_id"],  # type: ignore[arg-type]
            timestamp_s=raw["timestamp_s"],  # type: ignore[arg-type]
            target_query=TargetQuery.from_dict(raw["target_query"]),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "TrackRequest":
        return cls.from_dict(loads_strict_object(value, "track request"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "stream_id": self.stream_id,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "target_query": self.target_query.to_dict(),
        }


def _bbox(value: object) -> tuple[float, float, float, float]:
    values = _sequence(value, "bbox_xyxy_normalized", maximum=4)
    if len(values) != 4:
        raise ProtocolValidationError(
            "bbox_xyxy_normalized must contain exactly four coordinates"
        )
    result = tuple(_finite_nonnegative(item, "bbox coordinate") for item in values)
    if any(item > 1.0 for item in result):
        raise ProtocolValidationError("bbox coordinates must be within [0, 1]")
    x1, y1, x2, y2 = result
    if x1 >= x2 or y1 >= y2:
        raise ProtocolValidationError("bbox must satisfy x1 < x2 and y1 < y2")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TrackDetection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy_normalized: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _integer(self.track_id, "track_id"))
        object.__setattr__(
            self,
            "class_id",
            _integer(self.class_id, "class_id", maximum=65_535),
        )
        object.__setattr__(
            self,
            "class_name",
            _bounded_text(self.class_name, "class_name"),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "bbox_xyxy_normalized",
            _bbox(self.bbox_xyxy_normalized),
        )

    @classmethod
    def from_dict(cls, value: object) -> "TrackDetection":
        raw = _strict_fields(
            value,
            "detection",
            frozenset(
                {
                    "track_id",
                    "class_id",
                    "class_name",
                    "confidence",
                    "bbox_xyxy_normalized",
                }
            ),
        )
        return cls(
            track_id=raw["track_id"],  # type: ignore[arg-type]
            class_id=raw["class_id"],  # type: ignore[arg-type]
            class_name=raw["class_name"],  # type: ignore[arg-type]
            confidence=raw["confidence"],  # type: ignore[arg-type]
            bbox_xyxy_normalized=raw["bbox_xyxy_normalized"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy_normalized": list(self.bbox_xyxy_normalized),
        }


# Short compatibility name useful to clients without compromising the field
# meaning: every returned detection already belongs to the tracker response.
Detection = TrackDetection


@dataclass(frozen=True, slots=True)
class TimingMs:
    decode: float
    inference: float
    tracking: float
    total: float

    def __post_init__(self) -> None:
        for name in ("decode", "inference", "tracking", "total"):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        subtotal = self.decode + self.inference + self.tracking
        if self.total + 1e-6 < subtotal:
            raise ProtocolValidationError(
                "timing_ms.total cannot be less than its component sum"
            )

    @classmethod
    def from_dict(cls, value: object) -> "TimingMs":
        raw = _strict_fields(
            value,
            "timing_ms",
            frozenset({"decode", "inference", "tracking", "total"}),
        )
        return cls(
            decode=raw["decode"],  # type: ignore[arg-type]
            inference=raw["inference"],  # type: ignore[arg-type]
            tracking=raw["tracking"],  # type: ignore[arg-type]
            total=raw["total"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "decode": self.decode,
            "inference": self.inference,
            "tracking": self.tracking,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class TrackResponse:
    schema_version: int
    request_id: str
    mission_id: str
    uav_id: str
    stream_id: str
    frame_id: str
    timestamp_s: float
    detections: tuple[TrackDetection, ...]
    timing_ms: TimingMs

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        mission_id = _mission_id(self.mission_id)
        uav_id = _uav_id(self.uav_id)
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        object.__setattr__(self, "mission_id", mission_id)
        object.__setattr__(self, "uav_id", uav_id)
        object.__setattr__(
            self,
            "stream_id",
            _stream_id(self.stream_id, mission_id, uav_id),
        )
        object.__setattr__(self, "frame_id", _routing_id(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "timestamp_s",
            _finite_nonnegative(self.timestamp_s, "timestamp_s"),
        )
        detections = tuple(self.detections)
        if len(detections) > 10_000:
            raise ProtocolValidationError("detections exceeds 10000 entries")
        if any(not isinstance(item, TrackDetection) for item in detections):
            raise TypeError("detections must contain TrackDetection values")
        object.__setattr__(self, "detections", detections)
        if not isinstance(self.timing_ms, TimingMs):
            raise TypeError("timing_ms must be a TimingMs")

    @classmethod
    def from_dict(cls, value: object) -> "TrackResponse":
        raw = _strict_fields(
            value,
            "track response",
            frozenset(
                {
                    "schema_version",
                    "request_id",
                    "mission_id",
                    "uav_id",
                    "stream_id",
                    "frame_id",
                    "timestamp_s",
                    "detections",
                    "timing_ms",
                }
            ),
        )
        detections = _sequence(raw["detections"], "detections", maximum=10_000)
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            request_id=raw["request_id"],  # type: ignore[arg-type]
            mission_id=raw["mission_id"],  # type: ignore[arg-type]
            uav_id=raw["uav_id"],  # type: ignore[arg-type]
            stream_id=raw["stream_id"],  # type: ignore[arg-type]
            frame_id=raw["frame_id"],  # type: ignore[arg-type]
            timestamp_s=raw["timestamp_s"],  # type: ignore[arg-type]
            detections=tuple(TrackDetection.from_dict(item) for item in detections),
            timing_ms=TimingMs.from_dict(raw["timing_ms"]),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "TrackResponse":
        return cls.from_dict(loads_strict_object(value, "track response"))

    def assert_matches(self, request: TrackRequest) -> None:
        if not isinstance(request, TrackRequest):
            raise TypeError("request must be a TrackRequest")
        fields = (
            "schema_version",
            "request_id",
            "mission_id",
            "uav_id",
            "stream_id",
            "frame_id",
            "timestamp_s",
        )
        mismatched = [
            name for name in fields if getattr(self, name) != getattr(request, name)
        ]
        if mismatched:
            raise RouteMismatchError(
                "track response does not match request fields: "
                + ", ".join(mismatched)
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "stream_id": self.stream_id,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "detections": [item.to_dict() for item in self.detections],
            "timing_ms": self.timing_ms.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ResetStreamRequest:
    schema_version: int
    request_id: str
    mission_id: str
    uav_id: str
    stream_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        mission_id = _mission_id(self.mission_id)
        uav_id = _uav_id(self.uav_id)
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        object.__setattr__(self, "mission_id", mission_id)
        object.__setattr__(self, "uav_id", uav_id)
        object.__setattr__(
            self,
            "stream_id",
            _stream_id(self.stream_id, mission_id, uav_id),
        )

    @classmethod
    def from_dict(cls, value: object) -> "ResetStreamRequest":
        raw = _strict_fields(
            value,
            "stream reset request",
            frozenset(
                {"schema_version", "request_id", "mission_id", "uav_id", "stream_id"}
            ),
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            request_id=raw["request_id"],  # type: ignore[arg-type]
            mission_id=raw["mission_id"],  # type: ignore[arg-type]
            uav_id=raw["uav_id"],  # type: ignore[arg-type]
            stream_id=raw["stream_id"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "stream_id": self.stream_id,
        }


def loads_strict_object(value: str | bytes, name: str = "payload") -> Mapping[str, object]:
    if not isinstance(value, (str, bytes)):
        raise ProtocolValidationError(f"{name} must be JSON text")
    try:
        decoded: Any = json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolValidationError(f"{name} is invalid strict JSON: {exc}") from exc
    return _object(decoded, name)


__all__ = [
    "Detection",
    "ProtocolValidationError",
    "ResetStreamRequest",
    "RouteMismatchError",
    "SCHEMA_VERSION",
    "TargetQuery",
    "TimingMs",
    "TrackDetection",
    "TrackRequest",
    "TrackResponse",
    "loads_strict_object",
]
