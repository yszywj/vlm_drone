"""Routing-safe mission events and a bounded in-memory event bus."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from numbers import Real
from threading import RLock

from common.ids import (
    validate_mission_id,
    validate_routing_id,
    validate_uav_id,
)


class MissionEventType(str, Enum):
    PERIODIC_REVIEW_DUE = "PERIODIC_REVIEW_DUE"
    CANDIDATE_PERSISTENT = "CANDIDATE_PERSISTENT"
    MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATES"
    TARGET_CONFIRMATION_REQUIRED = "TARGET_CONFIRMATION_REQUIRED"
    TARGET_IDENTITY_UNCERTAIN = "TARGET_IDENTITY_UNCERTAIN"
    TRACK_CONFIDENCE_DROP = "TRACK_CONFIDENCE_DROP"
    TRACK_LOST = "TRACK_LOST"
    OBSTACLE_VISIBLE = "OBSTACLE_VISIBLE"
    PATH_BLOCKED = "PATH_BLOCKED"
    IMMINENT_COLLISION = "IMMINENT_COLLISION"
    HOLD_REQUESTED = "HOLD_REQUESTED"
    HOLD_ESTABLISHED = "HOLD_ESTABLISHED"
    ROUTE_PROPOSED = "ROUTE_PROPOSED"
    ROUTE_REJECTED = "ROUTE_REJECTED"
    ROUTE_ACCEPTED = "ROUTE_ACCEPTED"
    ROUTE_COLLISION = "ROUTE_COLLISION"
    SKILL_PROGRESS_STALLED = "SKILL_PROGRESS_STALLED"
    LOW_VISIBILITY = "LOW_VISIBILITY"
    TASK_COMPLETION_UNCERTAIN = "TASK_COMPLETION_UNCERTAIN"
    MODEL_REVIEW_STARTED = "MODEL_REVIEW_STARTED"
    MODEL_REVIEW_COMPLETED = "MODEL_REVIEW_COMPLETED"
    MODEL_REVIEW_TIMEOUT = "MODEL_REVIEW_TIMEOUT"
    MODEL_RESPONSE_STALE = "MODEL_RESPONSE_STALE"
    PLAN_REVISION_REQUESTED = "PLAN_REVISION_REQUESTED"
    PLAN_REVISION_ACCEPTED = "PLAN_REVISION_ACCEPTED"
    PLAN_REVISION_REJECTED = "PLAN_REVISION_REJECTED"


class EventSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class _FrozenJSONDict(dict[str, object]):
    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("event JSON payloads are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


_MAX_JSON_DEPTH = 8
_MAX_JSON_ITEMS = 256
_MAX_JSON_STRING_CHARS = 4096
_MAX_JSON_ENCODED_BYTES = 65_536


def _finite_timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timestamp_s must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError("timestamp_s must be a finite non-negative number")
    return normalized


def _positive_plan_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("plan_version must be an integer")
    if value <= 0:
        raise ValueError("plan_version must be greater than zero")
    return value


def _freeze_json(
    value: object,
    *,
    field_name: str,
    depth: int,
    item_budget: list[int],
    active: set[int],
) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{field_name} exceeds the maximum JSON depth")
    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in active:
            raise ValueError(f"{field_name} contains a circular reference")
        active.add(object_id)
        try:
            copied: dict[str, object] = {}
            for key, item in value.items():
                item_budget[0] += 1
                if item_budget[0] > _MAX_JSON_ITEMS:
                    raise ValueError(f"{field_name} contains too many JSON items")
                if not isinstance(key, str):
                    raise TypeError(f"{field_name} keys must be strings")
                if len(key) > _MAX_JSON_STRING_CHARS:
                    raise ValueError(f"{field_name} contains an oversized key")
                copied[key] = _freeze_json(
                    item,
                    field_name=f"{field_name}.{key}",
                    depth=depth + 1,
                    item_budget=item_budget,
                    active=active,
                )
        finally:
            active.remove(object_id)
        return _FrozenJSONDict(copied)
    if isinstance(value, (list, tuple)):
        object_id = id(value)
        if object_id in active:
            raise ValueError(f"{field_name} contains a circular reference")
        active.add(object_id)
        try:
            copied_items: list[object] = []
            for index, item in enumerate(value):
                item_budget[0] += 1
                if item_budget[0] > _MAX_JSON_ITEMS:
                    raise ValueError(f"{field_name} contains too many JSON items")
                copied_items.append(
                    _freeze_json(
                        item,
                        field_name=f"{field_name}[{index}]",
                        depth=depth + 1,
                        item_budget=item_budget,
                        active=active,
                    )
                )
        finally:
            active.remove(object_id)
        return tuple(copied_items)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field_name} must not contain NaN or Infinity")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_JSON_STRING_CHARS:
            raise ValueError(f"{field_name} contains an oversized string")
        return value
    raise TypeError(f"{field_name} contains a non-JSON value")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def validated_json_payload(
    value: object,
    *,
    field_name: str = "payload",
) -> Mapping[str, object]:
    """Return an immutable bounded copy of one JSON object.

    Numpy arrays, bytes, model/controller instances, and arbitrary Python
    objects are rejected by the primitive allow-list.
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object")
    frozen = _freeze_json(
        value,
        field_name=field_name,
        depth=0,
        item_budget=[0],
        active=set(),
    )
    assert isinstance(frozen, Mapping)
    encoded = json.dumps(frozen, ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_JSON_ENCODED_BYTES:
        raise ValueError(f"{field_name} exceeds the encoded byte limit")
    return frozen


def json_payload_to_dict(value: Mapping[str, object]) -> dict[str, object]:
    copied = _thaw_json(value)
    assert isinstance(copied, dict)
    return copied


@dataclass(frozen=True, slots=True)
class MissionEvent:
    event_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    timestamp_s: float
    event_type: MissionEventType
    severity: EventSeverity
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_id",
            validate_routing_id(self.event_id, "event_id"),
        )
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "plan_version",
            _positive_plan_version(self.plan_version),
        )
        object.__setattr__(
            self,
            "timestamp_s",
            _finite_timestamp(self.timestamp_s),
        )
        if not isinstance(self.event_type, MissionEventType):
            try:
                object.__setattr__(
                    self,
                    "event_type",
                    MissionEventType(self.event_type),
                )
            except (TypeError, ValueError):
                raise ValueError("event_type must be a supported MissionEventType") from None
        if not isinstance(self.severity, EventSeverity):
            try:
                object.__setattr__(
                    self,
                    "severity",
                    EventSeverity(self.severity),
                )
            except (TypeError, ValueError):
                raise ValueError("severity must be a supported EventSeverity") from None
        object.__setattr__(
            self,
            "payload",
            validated_json_payload(self.payload),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "timestamp_s": self.timestamp_s,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "payload": json_payload_to_dict(self.payload),
        }


class MissionEventBus:
    """Small bounded event history with routing-aware reads."""

    def __init__(self, *, max_events: int = 256) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise TypeError("max_events must be an integer")
        if max_events <= 0:
            raise ValueError("max_events must be greater than zero")
        self._events: deque[MissionEvent] = deque(maxlen=max_events)
        self._event_ids: set[str] = set()
        self._lock = RLock()

    @property
    def max_events(self) -> int:
        maxlen = self._events.maxlen
        assert maxlen is not None
        return maxlen

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def publish(self, event: MissionEvent) -> None:
        if not isinstance(event, MissionEvent):
            raise TypeError("event must be a MissionEvent")
        with self._lock:
            if event.event_id in self._event_ids:
                raise ValueError("event_id has already been published")
            if len(self._events) == self.max_events:
                evicted = self._events[0]
                self._event_ids.remove(evicted.event_id)
            self._events.append(event)
            self._event_ids.add(event.event_id)

    def recent(
        self,
        *,
        uav_id: str | None = None,
        mission_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[MissionEvent, ...]:
        normalized_uav = None if uav_id is None else validate_uav_id(uav_id)
        normalized_mission = (
            None if mission_id is None else validate_mission_id(mission_id)
        )
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an integer or None")
            if limit <= 0:
                raise ValueError("limit must be greater than zero")
        with self._lock:
            selected = [
                event
                for event in self._events
                if (normalized_uav is None or event.uav_id == normalized_uav)
                and (
                    normalized_mission is None
                    or event.mission_id == normalized_mission
                )
            ]
            if limit is not None:
                selected = selected[-limit:]
            return tuple(selected)


EventBus = MissionEventBus


__all__ = [
    "EventBus",
    "EventSeverity",
    "MissionEvent",
    "MissionEventBus",
    "MissionEventType",
    "json_payload_to_dict",
    "validated_json_payload",
]
