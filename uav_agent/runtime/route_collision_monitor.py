"""Trusted swept-volume collision closure for executing model routes.

The monitor is intentionally independent of Isaac Sim.  It consumes sampled
UAV world positions, uses the same immutable :class:`ObstacleRegistry` as the
scene and route critic, and closes an actual collision through trusted runtime
state: ``EXECUTING -> COLLIDED -> SkillManager.cancel_task()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol

from common.ids import generate_routing_id
from env.obstacle_registry import ObstacleRegistry
from planner.route_types import RouteState
from runtime.events import EventSeverity, MissionEvent, MissionEventType
from runtime.route_registry import RouteRecord, RouteRegistry


ROUTE_COLLISION_SOURCE = "route_collision_monitor"


class RouteCollisionMonitorError(RuntimeError):
    """Raised when routed collision state cannot be closed safely."""


class _CancelableRouteManager(Protocol):
    active_name: object
    active_planned_step_id: str | None
    task_plan: object | None
    task_status: object

    def cancel_task(self) -> object: ...


@dataclass(frozen=True, slots=True)
class RouteCollision:
    """One immutable, routed collision detected from a swept UAV segment."""

    route_id: str
    obstacle_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    timestamp_s: float
    previous_position_world_m: tuple[float, float, float]
    current_position_world_m: tuple[float, float, float]
    impact_position_world_m: tuple[float, float, float]
    segment_fraction: float
    uav_radius_m: float
    event: MissionEvent

    def to_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "obstacle_id": self.obstacle_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "timestamp_s": self.timestamp_s,
            "previous_position_world_m": list(self.previous_position_world_m),
            "current_position_world_m": list(self.current_position_world_m),
            "impact_position_world_m": list(self.impact_position_world_m),
            "segment_fraction": self.segment_fraction,
            "uav_radius_m": self.uav_radius_m,
            "event_id": self.event.event_id,
        }


class RouteCollisionMonitor:
    """Detect collisions only for the Manager-owned executing FOLLOW_ROUTE.

    ``event_sink`` is suitable for :class:`MissionEventBus.publish`.
    ``collision_sink`` receives the richer :class:`RouteCollision`; a sparse
    experiment logger can be adapted with
    ``lambda collision: logger.record_collision()``.
    """

    def __init__(
        self,
        *,
        obstacle_registry: ObstacleRegistry,
        route_registry: RouteRegistry,
        skill_manager: _CancelableRouteManager,
        uav_radius_m: float = 0.0,
        event_sink: Callable[[MissionEvent], object] | None = None,
        collision_sink: Callable[[RouteCollision], object] | None = None,
    ) -> None:
        if not isinstance(obstacle_registry, ObstacleRegistry):
            raise TypeError("obstacle_registry must be an ObstacleRegistry")
        if not isinstance(route_registry, RouteRegistry):
            raise TypeError("route_registry must be a RouteRegistry")
        if not callable(getattr(skill_manager, "cancel_task", None)):
            raise TypeError("skill_manager must provide cancel_task")
        radius = _finite_number(uav_radius_m, "uav_radius_m")
        if radius < 0.0:
            raise ValueError("uav_radius_m must be non-negative")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        if collision_sink is not None and not callable(collision_sink):
            raise TypeError("collision_sink must be callable or None")
        self._obstacles = obstacle_registry
        self._routes = route_registry
        self._manager = skill_manager
        self._uav_radius_m = radius
        self._event_sink = event_sink
        self._collision_sink = collision_sink
        self._tracked_route_id: str | None = None
        self._previous_position_world_m: tuple[float, float, float] | None = None
        self._previous_timestamp_s: float | None = None
        self._records: list[RouteCollision] = []

    @property
    def uav_radius_m(self) -> float:
        return self._uav_radius_m

    @property
    def tracked_route_id(self) -> str | None:
        return self._tracked_route_id

    @property
    def records(self) -> tuple[RouteCollision, ...]:
        return tuple(self._records)

    def observe(
        self,
        position_world_m: Iterable[float],
        *,
        timestamp_s: float,
    ) -> RouteCollision | None:
        """Consume one world-position sample and close the first collision."""

        current = _xyz(position_world_m, "position_world_m")
        timestamp = _nonnegative(timestamp_s, "timestamp_s")
        active = self._active_executing_route()
        if active is None:
            self._clear_tracking()
            return None

        record, plan = active
        if record.route_id != self._tracked_route_id:
            start = current
            self._tracked_route_id = record.route_id
            self._previous_timestamp_s = timestamp
        else:
            start = self._previous_position_world_m or current
            if (
                self._previous_timestamp_s is not None
                and timestamp < self._previous_timestamp_s
            ):
                raise ValueError("timestamp_s cannot move backwards during FOLLOW_ROUTE")

        self._previous_position_world_m = current
        self._previous_timestamp_s = timestamp
        hit = self._first_collision(start, current)
        if hit is None:
            return None

        obstacle_id, fraction = hit
        impact = tuple(
            begin + fraction * (end - begin)
            for begin, end in zip(start, current)
        )
        updated = self._routes.transition(record.route_id, RouteState.COLLIDED)
        if updated.state is not RouteState.COLLIDED:
            raise RouteCollisionMonitorError("route registry did not enter COLLIDED")
        event = MissionEvent(
            event_id=generate_routing_id("event_collision"),
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            plan_version=plan.plan_version,
            timestamp_s=timestamp,
            event_type=MissionEventType.ROUTE_COLLISION,
            severity=EventSeverity.CRITICAL,
            payload={
                "source": ROUTE_COLLISION_SOURCE,
                "geometry_source": "scene_obstacle_registry",
                "route_id": record.route_id,
                "obstacle_id": obstacle_id,
                "previous_position_world_m": list(start),
                "current_position_world_m": list(current),
                "impact_position_world_m": list(impact),
                "segment_fraction": fraction,
                "uav_radius_m": self._uav_radius_m,
            },
        )
        collision = RouteCollision(
            route_id=record.route_id,
            obstacle_id=obstacle_id,
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            plan_version=plan.plan_version,
            timestamp_s=timestamp,
            previous_position_world_m=start,
            current_position_world_m=current,
            impact_position_world_m=impact,  # type: ignore[arg-type]
            segment_fraction=fraction,
            uav_radius_m=self._uav_radius_m,
            event=event,
        )
        self._records.append(collision)
        self._close_collision(collision)
        return collision

    tick = observe

    def reset(self) -> None:
        """Clear sampled positions and collision history at an episode boundary."""

        self._clear_tracking()
        self._records.clear()

    def _active_executing_route(self) -> tuple[RouteRecord, object] | None:
        if _enum_label(getattr(self._manager, "task_status", None)) != "RUNNING":
            return None
        if _enum_label(getattr(self._manager, "active_name", None)) != "FOLLOW_ROUTE":
            return None
        plan = getattr(self._manager, "task_plan", None)
        step_id = getattr(self._manager, "active_planned_step_id", None)
        steps = getattr(plan, "steps", ())
        step = next(
            (candidate for candidate in steps if getattr(candidate, "step_id", None) == step_id),
            None,
        )
        if step is None or _enum_label(getattr(step, "skill", None)) != "FOLLOW_ROUTE":
            return None
        params = getattr(step, "params", None)
        route_id = None if params is None else params.get("route_ref")
        if not isinstance(route_id, str):
            raise RouteCollisionMonitorError("active FOLLOW_ROUTE has no route_ref")
        try:
            record = self._routes.get(route_id)
        except KeyError as exc:
            raise RouteCollisionMonitorError(
                f"active FOLLOW_ROUTE references unknown route {route_id}"
            ) from exc
        if record.state is not RouteState.EXECUTING:
            return None
        if record.plan_version != getattr(plan, "plan_version", None):
            raise RouteCollisionMonitorError(
                "executing route plan_version does not match active TaskPlan"
            )
        if getattr(plan, "uav_id", None) != getattr(self._manager, "uav_id", None):
            raise RouteCollisionMonitorError(
                "active TaskPlan uav_id does not match SkillManager"
            )
        return record, plan

    def _first_collision(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> tuple[str, float] | None:
        radius = (self._uav_radius_m,) * 3
        first: tuple[str, float] | None = None
        for obstacle in self._obstacles.collidable_specs:
            fraction = obstacle.aabb.expanded(radius).segment_intersection_fraction(
                start,
                end,
            )
            if fraction is None:
                continue
            if first is None or fraction < first[1]:
                first = obstacle.obstacle_id, fraction
        return first

    def _close_collision(self, collision: RouteCollision) -> None:
        failures: list[Exception] = []
        try:
            # SkillManager owns cancellation and immediately replaces the
            # active route with its trusted emergency LAND lifecycle.
            self._manager.cancel_task()
        except Exception as exc:
            failures.append(exc)
        if self._event_sink is not None:
            try:
                self._event_sink(collision.event)
            except Exception as exc:
                failures.append(exc)
        if self._collision_sink is not None:
            try:
                self._collision_sink(collision)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RouteCollisionMonitorError(
                "route collision was recorded but a closure callback failed"
            ) from failures[0]

    def _clear_tracking(self) -> None:
        self._tracked_route_id = None
        self._previous_position_world_m = None
        self._previous_timestamp_s = None


def _enum_label(value: object) -> str | None:
    raw = getattr(value, "value", value)
    if isinstance(raw, str):
        return raw
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else None


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative(value: object, name: str) -> float:
    normalized = _finite_number(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _xyz(value: Iterable[float], name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain three finite numbers")
    try:
        items = tuple(value)
    except TypeError:
        raise TypeError(f"{name} must contain three finite numbers") from None
    if len(items) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    return tuple(
        _finite_number(item, f"{name}[{index}]")
        for index, item in enumerate(items)
    )  # type: ignore[return-value]


__all__ = [
    "ROUTE_COLLISION_SOURCE",
    "RouteCollision",
    "RouteCollisionMonitor",
    "RouteCollisionMonitorError",
]
