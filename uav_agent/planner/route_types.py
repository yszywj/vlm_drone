"""Strict model-facing and trusted value types for spatial routes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import dist, isfinite
from numbers import Real

from common.ids import validate_routing_id
from planner.spatial import CoordinateFrame


class RouteContractError(ValueError):
    """Raised when an untrusted route violates the V3 transport contract."""


class AvoidanceStrategyType(str, Enum):
    BYPASS_LEFT = "BYPASS_LEFT"
    BYPASS_RIGHT = "BYPASS_RIGHT"
    BYPASS_ABOVE = "BYPASS_ABOVE"
    BACKTRACK = "BACKTRACK"
    HOLD_POSITION = "HOLD_POSITION"


class RouteState(str, Enum):
    PROPOSED = "PROPOSED"
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    COLLIDED = "COLLIDED"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise RouteContractError(f"{name} must be finite")
    return normalized


def _positive(value: object, name: str) -> float:
    normalized = _finite(value, name)
    if normalized <= 0.0:
        raise RouteContractError(f"{name} must be greater than zero")
    return normalized


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise RouteContractError(f"{name} must contain exactly three numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _strict(data: Mapping[str, object], required: set[str], name: str) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be an object")
    missing, unknown = required - set(data), set(data) - required
    if missing:
        raise RouteContractError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise RouteContractError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


@dataclass(frozen=True, slots=True)
class RouteWaypoint:
    waypoint_id: str
    xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "waypoint_id", validate_routing_id(self.waypoint_id, "waypoint_id")
        )
        object.__setattr__(self, "xyz_m", _vector3(self.xyz_m, "xyz_m"))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RouteWaypoint":
        _strict(data, {"waypoint_id", "xyz_m"}, "RouteWaypoint")
        return cls(data["waypoint_id"], data["xyz_m"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"waypoint_id": self.waypoint_id, "xyz_m": list(self.xyz_m)}


@dataclass(frozen=True, slots=True)
class RouteDraft:
    route_id: str
    frame: CoordinateFrame
    waypoints: tuple[RouteWaypoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_id", validate_routing_id(self.route_id, "route_id"))
        if not isinstance(self.frame, CoordinateFrame):
            try:
                object.__setattr__(self, "frame", CoordinateFrame(self.frame))
            except (TypeError, ValueError):
                raise RouteContractError("frame must be a supported CoordinateFrame") from None
        waypoints = tuple(self.waypoints)
        if not 2 <= len(waypoints) <= 16 or any(
            not isinstance(item, RouteWaypoint) for item in waypoints
        ):
            raise RouteContractError("waypoints must contain 2..16 RouteWaypoint values")
        ids = [item.waypoint_id for item in waypoints]
        if len(ids) != len(set(ids)):
            raise RouteContractError("waypoint IDs must be unique")
        for left, right in zip(waypoints, waypoints[1:]):
            if dist(left.xyz_m, right.xyz_m) <= 1e-9:
                raise RouteContractError("adjacent route waypoints must be distinct")
        object.__setattr__(self, "waypoints", waypoints)

    @property
    def length_m(self) -> float:
        return sum(
            dist(left.xyz_m, right.xyz_m)
            for left, right in zip(self.waypoints, self.waypoints[1:])
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RouteDraft":
        _strict(data, {"route_id", "frame", "waypoints"}, "RouteDraft")
        raw = data["waypoints"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError("RouteDraft.waypoints must be an array")
        return cls(
            data["route_id"],  # type: ignore[arg-type]
            data["frame"],  # type: ignore[arg-type]
            tuple(RouteWaypoint.from_dict(item) for item in raw),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "frame": self.frame.value,
            "waypoints": [item.to_dict() for item in self.waypoints],
        }


@dataclass(frozen=True, slots=True)
class RouteConstraints:
    max_waypoints: int = 5
    max_detour_distance_m: float = 40.0
    minimum_clearance_m: float = 1.5
    max_segment_length_m: float = 25.0
    rejoin_tolerance_m: float = 2.0
    must_rejoin_original_goal: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_waypoints, bool) or not isinstance(self.max_waypoints, int) or not 2 <= self.max_waypoints <= 16:
            raise RouteContractError("max_waypoints must be an integer within [2, 16]")
        for name in (
            "max_detour_distance_m",
            "minimum_clearance_m",
            "max_segment_length_m",
            "rejoin_tolerance_m",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if not isinstance(self.must_rejoin_original_goal, bool):
            raise TypeError("must_rejoin_original_goal must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_waypoints": self.max_waypoints,
            "max_detour_distance_m": self.max_detour_distance_m,
            "minimum_clearance_m": self.minimum_clearance_m,
            "max_segment_length_m": self.max_segment_length_m,
            "rejoin_tolerance_m": self.rejoin_tolerance_m,
            "must_rejoin_original_goal": self.must_rejoin_original_goal,
        }


@dataclass(frozen=True, slots=True)
class AvoidanceStrategy:
    strategy: AvoidanceStrategyType
    rejoin_target: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, AvoidanceStrategyType):
            try:
                object.__setattr__(self, "strategy", AvoidanceStrategyType(self.strategy))
            except (TypeError, ValueError):
                raise RouteContractError("strategy is not supported") from None
        object.__setattr__(
            self,
            "rejoin_target",
            validate_routing_id(self.rejoin_target, "rejoin_target"),
        )
        reasons = tuple(self.reason_codes)
        if not reasons or len(reasons) > 16:
            raise RouteContractError("reason_codes must contain 1..16 values")
        for reason in reasons:
            if not isinstance(reason, str) or not reason or len(reason) > 64:
                raise RouteContractError("reason_codes must be bounded non-empty strings")
        if len(reasons) != len(set(reasons)):
            raise RouteContractError("reason_codes must be unique")
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.value,
            "rejoin_target": self.rejoin_target,
            "reason_codes": list(self.reason_codes),
        }


__all__ = [
    "AvoidanceStrategy",
    "AvoidanceStrategyType",
    "RouteConstraints",
    "RouteContractError",
    "RouteDraft",
    "RouteState",
    "RouteWaypoint",
]
