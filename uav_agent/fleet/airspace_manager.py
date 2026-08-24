"""Deterministic fleet-wide separation checks over one synchronized snapshot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import hypot, isfinite, sqrt
from numbers import Real
from types import MappingProxyType

from common.ids import validate_routing_id, validate_uav_id


class AirspaceRisk(str, Enum):
    CLEAR = "CLEAR"
    ADVISORY = "ADVISORY"
    CONFLICT = "CONFLICT"
    COLLISION_PREDICTED = "COLLISION_PREDICTED"


class RouteConflictPolicy(str, Enum):
    LOWER_PRIORITY_HOLDS = "LOWER_PRIORITY_HOLDS"


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be a {qualifier} number")
    return result


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(
            f"{name} must contain exactly three finite numbers"
        ) from None
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return tuple(
        _finite(item, f"{name}[{index}]")
        for index, item in enumerate(items)
    )  # type: ignore[return-value]


def _route(value: object) -> tuple[tuple[float, float, float], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("route_xyz_m must be a sequence")
    result = tuple(_vector3(point, f"route_xyz_m[{index}]") for index, point in enumerate(value))
    if len(result) == 1:
        raise ValueError("route_xyz_m must be empty or contain at least two points")
    return result


@dataclass(frozen=True, slots=True)
class FleetUavPose:
    uav_id: str
    position_xyz_m: tuple[float, float, float]
    velocity_xyz_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    priority: int = 0
    assignment_id: str | None = None
    route_xyz_m: tuple[tuple[float, float, float], ...] = ()
    landing_zone_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "position_xyz_m", _vector3(self.position_xyz_m, "position_xyz_m"))
        object.__setattr__(self, "velocity_xyz_mps", _vector3(self.velocity_xyz_mps, "velocity_xyz_mps"))
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")
        if self.assignment_id is not None:
            object.__setattr__(
                self,
                "assignment_id",
                validate_routing_id(self.assignment_id, "assignment_id"),
            )
        object.__setattr__(self, "route_xyz_m", _route(self.route_xyz_m))
        if self.landing_zone_id is not None:
            object.__setattr__(
                self,
                "landing_zone_id",
                validate_routing_id(self.landing_zone_id, "landing_zone_id"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "uav_id": self.uav_id,
            "position_xyz_m": list(self.position_xyz_m),
            "velocity_xyz_mps": list(self.velocity_xyz_mps),
            "priority": self.priority,
            "assignment_id": self.assignment_id,
            "route_xyz_m": [list(point) for point in self.route_xyz_m],
            "landing_zone_id": self.landing_zone_id,
        }


@dataclass(frozen=True, slots=True)
class FleetPoseSnapshot:
    timestamp_s: float
    poses: Mapping[str, FleetUavPose]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s", nonnegative=True))
        if not isinstance(self.poses, Mapping):
            raise TypeError("poses must be a mapping")
        normalized: dict[str, FleetUavPose] = {}
        for raw_id, pose in self.poses.items():
            uav_id = validate_uav_id(raw_id)
            if not isinstance(pose, FleetUavPose):
                pose = _coerce_pose(uav_id, pose)
            if pose.uav_id != uav_id:
                raise ValueError("pose mapping key must equal FleetUavPose.uav_id")
            if uav_id in normalized:
                raise ValueError("poses contains a duplicate UAV ID")
            normalized[uav_id] = pose
        object.__setattr__(self, "poses", MappingProxyType(dict(sorted(normalized.items()))))

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_s": self.timestamp_s,
            "poses": {uav_id: pose.to_dict() for uav_id, pose in self.poses.items()},
        }


@dataclass(frozen=True, slots=True)
class PairwiseConflict:
    uav_a_id: str
    uav_b_id: str
    current_distance_m: float
    horizontal_distance_m: float
    vertical_distance_m: float
    relative_speed_mps: float
    predicted_closest_distance_m: float
    time_to_closest_s: float
    predicted_collision_time_s: float | None
    routes_intersect: bool
    shared_landing_zone: bool
    risk: AirspaceRisk
    hold_uav_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uav_a_id", validate_uav_id(self.uav_a_id))
        object.__setattr__(self, "uav_b_id", validate_uav_id(self.uav_b_id))
        if self.uav_a_id >= self.uav_b_id:
            raise ValueError("pair IDs must be stored in sorted order")
        for name in (
            "current_distance_m",
            "horizontal_distance_m",
            "vertical_distance_m",
            "relative_speed_mps",
            "predicted_closest_distance_m",
            "time_to_closest_s",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, nonnegative=True))
        if self.predicted_collision_time_s is not None:
            object.__setattr__(
                self,
                "predicted_collision_time_s",
                _finite(self.predicted_collision_time_s, "predicted_collision_time_s", nonnegative=True),
            )
        if not isinstance(self.risk, AirspaceRisk):
            object.__setattr__(self, "risk", AirspaceRisk(self.risk))
        if self.hold_uav_id is not None:
            object.__setattr__(self, "hold_uav_id", validate_uav_id(self.hold_uav_id))
            if self.hold_uav_id not in {self.uav_a_id, self.uav_b_id}:
                raise ValueError("hold_uav_id must belong to the pair")

    @property
    def is_conflict(self) -> bool:
        return self.risk in {AirspaceRisk.CONFLICT, AirspaceRisk.COLLISION_PREDICTED}

    def to_dict(self) -> dict[str, object]:
        return {
            "uav_a_id": self.uav_a_id,
            "uav_b_id": self.uav_b_id,
            "current_distance_m": self.current_distance_m,
            "horizontal_distance_m": self.horizontal_distance_m,
            "vertical_distance_m": self.vertical_distance_m,
            "relative_speed_mps": self.relative_speed_mps,
            "predicted_closest_distance_m": self.predicted_closest_distance_m,
            "time_to_closest_s": self.time_to_closest_s,
            "predicted_collision_time_s": self.predicted_collision_time_s,
            "routes_intersect": self.routes_intersect,
            "shared_landing_zone": self.shared_landing_zone,
            "risk": self.risk.value,
            "hold_uav_id": self.hold_uav_id,
        }


@dataclass(frozen=True, slots=True)
class FleetAirspaceDecision:
    timestamp_s: float
    conflicts: tuple[PairwiseConflict, ...] = ()
    hold_uav_ids: tuple[str, ...] = ()
    event_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s", nonnegative=True))
        conflicts = tuple(self.conflicts)
        if any(not isinstance(item, PairwiseConflict) for item in conflicts):
            raise TypeError("conflicts must contain PairwiseConflict values")
        object.__setattr__(self, "conflicts", conflicts)
        holds = tuple(sorted({validate_uav_id(item) for item in self.hold_uav_ids}))
        object.__setattr__(self, "hold_uav_ids", holds)
        if self.event_type is not None and self.event_type != "AIRSPACE_CONFLICT":
            raise ValueError("event_type must be AIRSPACE_CONFLICT or None")

    @property
    def clear(self) -> bool:
        return not any(conflict.is_conflict for conflict in self.conflicts)

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_s": self.timestamp_s,
            "event_type": self.event_type,
            "hold_uav_ids": list(self.hold_uav_ids),
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
        }


class FleetAirspaceManager:
    """Evaluate current and constant-velocity future separation immediately."""

    def __init__(
        self,
        minimum_separation_m: float = 5.0,
        *,
        warning_separation_m: float | None = None,
        prediction_horizon_s: float = 10.0,
        route_altitude_tolerance_m: float = 2.0,
        policy: RouteConflictPolicy | str = RouteConflictPolicy.LOWER_PRIORITY_HOLDS,
    ) -> None:
        self.minimum_separation_m = _finite(
            minimum_separation_m,
            "minimum_separation_m",
            nonnegative=True,
        )
        if self.minimum_separation_m <= 0.0:
            raise ValueError("minimum_separation_m must be greater than zero")
        warning = (
            self.minimum_separation_m * 1.5
            if warning_separation_m is None
            else _finite(warning_separation_m, "warning_separation_m", nonnegative=True)
        )
        if warning < self.minimum_separation_m:
            raise ValueError("warning_separation_m must be at least minimum_separation_m")
        self.warning_separation_m = warning
        self.prediction_horizon_s = _finite(
            prediction_horizon_s,
            "prediction_horizon_s",
            nonnegative=True,
        )
        if self.prediction_horizon_s <= 0.0:
            raise ValueError("prediction_horizon_s must be greater than zero")
        self.route_altitude_tolerance_m = _finite(
            route_altitude_tolerance_m,
            "route_altitude_tolerance_m",
            nonnegative=True,
        )
        self.policy = RouteConflictPolicy(policy)
        self._last_decision: FleetAirspaceDecision | None = None

    @property
    def last_decision(self) -> FleetAirspaceDecision | None:
        return self._last_decision

    def evaluate(self, snapshot: FleetPoseSnapshot | object) -> FleetAirspaceDecision:
        snapshot = coerce_fleet_pose_snapshot(snapshot)
        poses = tuple(snapshot.poses.values())
        conflicts: list[PairwiseConflict] = []
        for index, first in enumerate(poses):
            for second in poses[index + 1 :]:
                conflicts.append(self._evaluate_pair(first, second))
        holds = tuple(
            sorted(
                {
                    conflict.hold_uav_id
                    for conflict in conflicts
                    if conflict.is_conflict and conflict.hold_uav_id is not None
                }
            )
        )
        decision = FleetAirspaceDecision(
            timestamp_s=snapshot.timestamp_s,
            conflicts=tuple(conflicts),
            hold_uav_ids=holds,
            event_type=(
                "AIRSPACE_CONFLICT"
                if any(conflict.is_conflict for conflict in conflicts)
                else None
            ),
        )
        self._last_decision = decision
        return decision

    def check(self, snapshot: FleetPoseSnapshot | object) -> FleetAirspaceDecision:
        return self.evaluate(snapshot)

    def _evaluate_pair(self, first: FleetUavPose, second: FleetUavPose) -> PairwiseConflict:
        if first.uav_id > second.uav_id:
            first, second = second, first
        relative_position = tuple(
            second.position_xyz_m[index] - first.position_xyz_m[index]
            for index in range(3)
        )
        relative_velocity = tuple(
            second.velocity_xyz_mps[index] - first.velocity_xyz_mps[index]
            for index in range(3)
        )
        current = _norm(relative_position)
        horizontal = hypot(relative_position[0], relative_position[1])
        vertical = abs(relative_position[2])
        relative_speed = _norm(relative_velocity)
        velocity_sq = sum(component * component for component in relative_velocity)
        if velocity_sq <= 1e-12:
            time_to_closest = 0.0
        else:
            time_to_closest = max(
                0.0,
                min(
                    self.prediction_horizon_s,
                    -sum(a * b for a, b in zip(relative_position, relative_velocity))
                    / velocity_sq,
                ),
            )
        closest_vector = tuple(
            relative_position[index] + relative_velocity[index] * time_to_closest
            for index in range(3)
        )
        closest = _norm(closest_vector)
        routes_intersect = _routes_intersect(
            first.route_xyz_m,
            second.route_xyz_m,
            altitude_tolerance_m=self.route_altitude_tolerance_m,
        )
        shared_landing = bool(
            first.landing_zone_id
            and first.landing_zone_id == second.landing_zone_id
        )

        closing = sum(
            a * b for a, b in zip(relative_position, relative_velocity)
        ) < 0.0
        collision = current < self.minimum_separation_m or (
            closing
            and 0.0 < time_to_closest <= self.prediction_horizon_s
            and closest < self.minimum_separation_m
        )
        collision_time = (
            _time_to_separation_breach(
                relative_position,
                relative_velocity,
                separation_m=self.minimum_separation_m,
                horizon_s=self.prediction_horizon_s,
            )
            if collision
            else None
        )
        predicted_warning = (
            closing
            and 0.0 < time_to_closest <= self.prediction_horizon_s
            and closest < self.warning_separation_m
        )
        if collision:
            risk = AirspaceRisk.COLLISION_PREDICTED
        elif predicted_warning or routes_intersect or shared_landing:
            risk = AirspaceRisk.CONFLICT
        elif current < self.warning_separation_m:
            risk = AirspaceRisk.ADVISORY
        else:
            risk = AirspaceRisk.CLEAR
        hold = None
        # Entering the warning envelope is an explicit Fleet conflict event,
        # but it is not by itself a stop condition.  Holding on every warning
        # can deadlock two UAVs whose distinct recovery zones are safely above
        # the hard minimum yet intentionally closer than the advisory radius.
        # Immediate deterministic HOLD remains fail-closed for a current or
        # predicted hard-separation breach, an intersecting active route, or a
        # shared landing zone.
        if collision or routes_intersect or shared_landing:
            hold = _lower_priority(first, second).uav_id
        return PairwiseConflict(
            uav_a_id=first.uav_id,
            uav_b_id=second.uav_id,
            current_distance_m=current,
            horizontal_distance_m=horizontal,
            vertical_distance_m=vertical,
            relative_speed_mps=relative_speed,
            predicted_closest_distance_m=closest,
            time_to_closest_s=time_to_closest,
            predicted_collision_time_s=collision_time,
            routes_intersect=routes_intersect,
            shared_landing_zone=shared_landing,
            risk=risk,
            hold_uav_id=hold,
        )


def coerce_fleet_pose_snapshot(value: FleetPoseSnapshot | object) -> FleetPoseSnapshot:
    if isinstance(value, FleetPoseSnapshot):
        return value
    if isinstance(value, Mapping):
        timestamp = value.get("timestamp_s", value.get("timestamp", 0.0))
        poses = value.get("poses", value.get("uavs", value.get("uav_states")))
        velocities = value.get("uav_velocities_mps", {})
    else:
        timestamp = getattr(value, "timestamp_s", getattr(value, "timestamp", 0.0))
        poses = getattr(
            value,
            "poses",
            getattr(value, "uavs", getattr(value, "uav_states", None)),
        )
        velocities = getattr(value, "uav_velocities_mps", {})
    if not isinstance(poses, Mapping):
        raise TypeError("fleet pose snapshot must expose a poses mapping")
    if velocities and not isinstance(velocities, Mapping):
        raise TypeError("uav_velocities_mps must be a mapping")
    normalized: dict[str, object] = {}
    for raw_uav_id, pose in poses.items():
        if isinstance(pose, Mapping):
            item = dict(pose)
            item.setdefault("velocity_xyz_mps", velocities.get(raw_uav_id, (0.0, 0.0, 0.0)))
        else:
            item = FleetUavPose(
                uav_id=raw_uav_id,
                position_xyz_m=_position_from_pose(pose),
                velocity_xyz_mps=velocities.get(raw_uav_id, (0.0, 0.0, 0.0)),
            )
        normalized[raw_uav_id] = item
    return FleetPoseSnapshot(timestamp_s=timestamp, poses=normalized)  # type: ignore[arg-type]


def _coerce_pose(uav_id: str, value: object) -> FleetUavPose:
    if isinstance(value, Mapping):
        position = value.get("position_xyz_m", value.get("position", value.get("pose")))
        velocity = value.get("velocity_xyz_mps", value.get("velocity", (0.0, 0.0, 0.0)))
        return FleetUavPose(
            uav_id=uav_id,
            position_xyz_m=_position_from_pose(position),
            velocity_xyz_mps=velocity,  # type: ignore[arg-type]
            priority=value.get("priority", 0),  # type: ignore[arg-type]
            assignment_id=value.get("assignment_id"),  # type: ignore[arg-type]
            route_xyz_m=value.get("route_xyz_m", ()),  # type: ignore[arg-type]
            landing_zone_id=value.get("landing_zone_id"),  # type: ignore[arg-type]
        )
    position = getattr(value, "position_xyz_m", getattr(value, "position", getattr(value, "pose", None)))
    return FleetUavPose(
        uav_id=uav_id,
        position_xyz_m=_position_from_pose(position),
        velocity_xyz_mps=getattr(value, "velocity_xyz_mps", getattr(value, "velocity", (0.0, 0.0, 0.0))),
        priority=getattr(value, "priority", 0),
        assignment_id=getattr(value, "assignment_id", None),
        route_xyz_m=getattr(value, "route_xyz_m", ()),
        landing_zone_id=getattr(value, "landing_zone_id", None),
    )


def _position_from_pose(value: object) -> object:
    if value is None:
        raise TypeError("pose must expose position_xyz_m")
    if isinstance(value, Mapping):
        if "position_xyz_m" in value:
            return value["position_xyz_m"]
        if all(key in value for key in ("x", "y", "z")):
            return (value["x"], value["y"], value["z"])
    if all(hasattr(value, name) for name in ("x", "y", "z")):
        return (getattr(value, "x"), getattr(value, "y"), getattr(value, "z"))
    return value


def _norm(vector: tuple[float, float, float]) -> float:
    return sqrt(sum(component * component for component in vector))


def _time_to_separation_breach(
    relative_position: tuple[float, float, float],
    relative_velocity: tuple[float, float, float],
    *,
    separation_m: float,
    horizon_s: float,
) -> float | None:
    """Return first entry into the hard separation sphere.

    ``time_to_closest`` is not a collision time: for a head-on encounter it is
    later than the instant the vehicles first cross the safety boundary.  Solve
    ``|r + v*t|^2 = separation^2`` and report the earlier non-negative root.
    """

    current_sq = sum(component * component for component in relative_position)
    separation_sq = separation_m * separation_m
    if current_sq < separation_sq:
        return 0.0
    velocity_sq = sum(component * component for component in relative_velocity)
    if velocity_sq <= 1e-12:
        return None
    dot = sum(
        position * velocity
        for position, velocity in zip(relative_position, relative_velocity)
    )
    discriminant = dot * dot - velocity_sq * (current_sq - separation_sq)
    if discriminant < 0.0:
        return None
    entry = (-dot - sqrt(max(0.0, discriminant))) / velocity_sq
    if entry < 0.0 or entry > horizon_s:
        return None
    return entry


def _lower_priority(first: FleetUavPose, second: FleetUavPose) -> FleetUavPose:
    if first.priority != second.priority:
        return first if first.priority < second.priority else second
    # Deterministic tie break: the lexicographically later UAV yields.
    return first if first.uav_id > second.uav_id else second


def _routes_intersect(
    first: tuple[tuple[float, float, float], ...],
    second: tuple[tuple[float, float, float], ...],
    *,
    altitude_tolerance_m: float,
) -> bool:
    if len(first) < 2 or len(second) < 2:
        return False
    for a0, a1 in zip(first, first[1:]):
        for b0, b1 in zip(second, second[1:]):
            if max(min(a0[2], a1[2]), min(b0[2], b1[2])) - min(
                max(a0[2], a1[2]), max(b0[2], b1[2])
            ) > altitude_tolerance_m:
                continue
            if _segments_intersect_xy(a0, a1, b0, b1):
                return True
    return False


def _segments_intersect_xy(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    d: tuple[float, float, float],
) -> bool:
    def orientation(p: tuple[float, float, float], q: tuple[float, float, float], r: tuple[float, float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(
        p: tuple[float, float, float],
        q: tuple[float, float, float],
        r: tuple[float, float, float],
    ) -> bool:
        epsilon = 1e-9
        return (
            min(p[0], r[0]) - epsilon <= q[0] <= max(p[0], r[0]) + epsilon
            and min(p[1], r[1]) - epsilon
            <= q[1]
            <= max(p[1], r[1]) + epsilon
        )

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    epsilon = 1e-9
    if ((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon)) and (
        (o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon)
    ):
        return True
    return (
        (abs(o1) <= epsilon and on_segment(a, c, b))
        or (abs(o2) <= epsilon and on_segment(a, d, b))
        or (abs(o3) <= epsilon and on_segment(c, a, d))
        or (abs(o4) <= epsilon and on_segment(c, b, d))
    )


__all__ = [
    "AirspaceRisk",
    "FleetAirspaceDecision",
    "FleetAirspaceManager",
    "FleetPoseSnapshot",
    "FleetUavPose",
    "PairwiseConflict",
    "RouteConflictPolicy",
    "coerce_fleet_pose_snapshot",
]
