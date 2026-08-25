"""Structural interface and fail-closed errors for attribute verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from env.camera_types import CameraSample
from perception.attribute_types import (
    ATTRIBUTE_SCHEMA_VERSION,
    AttributeObservation,
    AttributeRequirement,
)
from runtime.frame_store import FrameRef
from yolo_service.protocol import TrackDetection


class AttributeVerificationError(RuntimeError):
    """Base class for trusted RGB-D attribute-pipeline failures."""

    code = "ATTRIBUTE_VERIFICATION_ERROR"


class AttributeRouteMismatch(AttributeVerificationError):
    """Evidence was offered to a different mission/UAV/candidate route."""

    code = "ATTRIBUTE_ROUTE_MISMATCH"


class AttributeTimeError(AttributeVerificationError):
    """Attribute evidence attempted to move backwards or reuse a timestamp."""

    code = "ATTRIBUTE_TIME_ERROR"


class AttributeFrameUnavailable(AttributeVerificationError):
    """The exact synchronized frame referenced by a request is unavailable."""

    code = "ATTRIBUTE_FRAME_UNAVAILABLE"


def _tracker_id(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("tracker_id must be an integer or routing string")
    if isinstance(value, Integral):
        if int(value) < 0:
            raise ValueError("tracker_id must be non-negative")
        return str(int(value))
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return str(int(value))
    return validate_routing_id(value, "tracker_id")


@dataclass(frozen=True, slots=True)
class AttributeVerificationRoute:
    """Ephemeral scalar routing envelope for one camera/tracker input."""

    mission_id: str
    uav_id: str
    assignment_id: str
    candidate_id: str
    tracker_id: str | int
    schema_version: int = ATTRIBUTE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
        ):
            raise TypeError("schema_version must be an integer")
        if int(self.schema_version) != ATTRIBUTE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {ATTRIBUTE_SCHEMA_VERSION}")
        object.__setattr__(self, "schema_version", ATTRIBUTE_SCHEMA_VERSION)
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

    @classmethod
    def from_dict(cls, value: object) -> "AttributeVerificationRoute":
        if not isinstance(value, Mapping):
            raise TypeError("AttributeVerificationRoute must be a mapping")
        fields = {
            "schema_version",
            "mission_id",
            "uav_id",
            "assignment_id",
            "candidate_id",
            "tracker_id",
        }
        if any(not isinstance(key, str) for key in value):
            raise TypeError("AttributeVerificationRoute keys must be strings")
        unknown = set(value) - fields
        missing = fields - set(value)
        if unknown:
            raise ValueError(
                "AttributeVerificationRoute has unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if missing:
            raise ValueError(
                "AttributeVerificationRoute is missing fields: "
                + ", ".join(sorted(missing))
            )
        return cls(**value)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "assignment_id": self.assignment_id,
            "candidate_id": self.candidate_id,
            "tracker_id": self.tracker_id,
        }

    def validate(self, requirement: AttributeRequirement, detection: TrackDetection) -> None:
        if not isinstance(requirement, AttributeRequirement):
            raise TypeError("requirement must be an AttributeRequirement")
        if not isinstance(detection, TrackDetection):
            raise TypeError("detection must be a TrackDetection")
        expected = (
            requirement.mission_id,
            requirement.uav_id,
            requirement.assignment_id,
            requirement.candidate_id,
            requirement.tracker_id,
        )
        actual = (
            self.mission_id,
            self.uav_id,
            self.assignment_id,
            self.candidate_id,
            self.tracker_id,
        )
        if actual != expected or self.tracker_id != _tracker_id(detection.track_id):
            raise AttributeRouteMismatch(
                "camera, candidate, tracker, UAV, and Assignment routing must match"
            )


@runtime_checkable
class AttributeVerifier(Protocol):
    """Verify one detector box against one synchronized RGB-D sample.

    Implementations accept either the original ``CameraSample`` or a
    ``FrameRef`` resolving both channels from one ``FrameStore`` entry.  They
    must never accept independently supplied RGB and depth arrays.
    """

    def verify(
        self,
        *,
        requirement: AttributeRequirement,
        detection: TrackDetection,
        route: AttributeVerificationRoute,
        frame_ref: FrameRef | None = None,
        camera_sample: CameraSample | None = None,
    ) -> AttributeObservation: ...


__all__ = [
    "AttributeFrameUnavailable",
    "AttributeRouteMismatch",
    "AttributeTimeError",
    "AttributeVerificationRoute",
    "AttributeVerificationError",
    "AttributeVerifier",
]
