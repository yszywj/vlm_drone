"""Trusted scheduling policy for sparse asynchronous Qwen visual reviews."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from threading import RLock
from types import MappingProxyType

from common.ids import (
    generate_routing_id,
    validate_mission_id,
    validate_request_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from runtime.events import MissionEvent, MissionEventType


DEFAULT_REVIEW_INTERVALS_S: Mapping[str, float] = MappingProxyType(
    {
        "GOTO": 5.0,
        "SEARCH": 2.0,
        "INSPECT": 1.0,
        "TRACK": 5.0,
    }
)

DEFAULT_EVENT_TRIGGERS = frozenset(
    {
        MissionEventType.PERIODIC_REVIEW_DUE,
        MissionEventType.CANDIDATE_PERSISTENT,
        MissionEventType.MULTIPLE_CANDIDATES,
        MissionEventType.TARGET_CONFIRMATION_REQUIRED,
        MissionEventType.TARGET_IDENTITY_UNCERTAIN,
        MissionEventType.TRACK_CONFIDENCE_DROP,
        MissionEventType.TRACK_LOST,
        MissionEventType.PATH_BLOCKED,
        MissionEventType.SKILL_PROGRESS_STALLED,
        MissionEventType.LOW_VISIBILITY,
        MissionEventType.TASK_COMPLETION_UNCERTAIN,
    }
)

DEFAULT_BLOCKING_EVENT_TYPES = frozenset(
    {
        MissionEventType.CANDIDATE_PERSISTENT,
        MissionEventType.MULTIPLE_CANDIDATES,
        MissionEventType.TARGET_CONFIRMATION_REQUIRED,
        MissionEventType.TARGET_IDENTITY_UNCERTAIN,
        MissionEventType.PATH_BLOCKED,
        MissionEventType.TASK_COMPLETION_UNCERTAIN,
    }
)


def _timestamp(value: object, field_name: str = "timestamp_s") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return normalized


def _positive_plan_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("plan_version must be an integer")
    if value <= 0:
        raise ValueError("plan_version must be greater than zero")
    return value


def _skill_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("skill_name must be a non-empty string")
    normalized = value.upper()
    if value != normalized:
        raise ValueError("skill_name must use its canonical uppercase name")
    return normalized


class ReviewTrigger(str, Enum):
    PERIODIC = "PERIODIC"
    EVENT = "EVENT"


class ReviewScheduleReason(str, Enum):
    SCHEDULED = "SCHEDULED"
    NOT_DUE = "NOT_DUE"
    COOLDOWN = "COOLDOWN"
    IN_FLIGHT = "IN_FLIGHT"


@dataclass(frozen=True, slots=True)
class ReviewTicket:
    request_id: str
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    skill_name: str
    submitted_timestamp_s: float
    trigger: ReviewTrigger
    blocking: bool
    event_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", validate_request_id(self.request_id))
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _positive_plan_version(self.plan_version))
        object.__setattr__(self, "skill_name", _skill_name(self.skill_name))
        object.__setattr__(
            self,
            "submitted_timestamp_s",
            _timestamp(self.submitted_timestamp_s, "submitted_timestamp_s"),
        )
        if not isinstance(self.trigger, ReviewTrigger):
            raise TypeError("trigger must be a ReviewTrigger")
        if not isinstance(self.blocking, bool):
            raise TypeError("blocking must be a bool")
        if self.event_id is not None:
            object.__setattr__(
                self,
                "event_id",
                validate_routing_id(self.event_id, "event_id"),
            )

    @property
    def hover_required(self) -> bool:
        return self.blocking

    @property
    def requires_supervisory_hover(self) -> bool:
        return self.blocking

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "review_id": self.review_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "skill_name": self.skill_name,
            "submitted_timestamp_s": self.submitted_timestamp_s,
            "trigger": self.trigger.value,
            "blocking": self.blocking,
            "hover_required": self.hover_required,
            "event_id": self.event_id,
        }


@dataclass(frozen=True, slots=True)
class ReviewScheduleDecision:
    reason: ReviewScheduleReason
    ticket: ReviewTicket | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, ReviewScheduleReason):
            raise TypeError("reason must be a ReviewScheduleReason")
        if self.reason is ReviewScheduleReason.SCHEDULED:
            if not isinstance(self.ticket, ReviewTicket):
                raise ValueError("a scheduled decision requires a ReviewTicket")
        elif self.ticket is not None:
            raise ValueError("an unscheduled decision cannot contain a ticket")

    @property
    def should_submit(self) -> bool:
        return self.ticket is not None

    @property
    def blocking(self) -> bool:
        return self.ticket.blocking if self.ticket is not None else False

    @property
    def hover_required(self) -> bool:
        return self.blocking

    @property
    def requires_supervisory_hover(self) -> bool:
        return self.blocking


class ReviewScheduler:
    """Schedule periodic/event reviews and atomically reserve one per UAV."""

    def __init__(
        self,
        *,
        intervals_s: Mapping[str, float] | None = None,
        cooldown_s: float = 1.0,
        event_triggers: object = DEFAULT_EVENT_TRIGGERS,
        blocking_event_types: object = (
            DEFAULT_BLOCKING_EVENT_TYPES
        ),
    ) -> None:
        raw_intervals = (
            DEFAULT_REVIEW_INTERVALS_S if intervals_s is None else intervals_s
        )
        if not isinstance(raw_intervals, Mapping):
            raise TypeError("intervals_s must be a mapping")
        normalized_intervals: dict[str, float] = {}
        for raw_skill, raw_interval in raw_intervals.items():
            skill = _skill_name(raw_skill)
            interval = _timestamp(raw_interval, f"{skill} interval")
            if interval == 0.0:
                raise ValueError("review intervals must be greater than zero")
            normalized_intervals[skill] = interval
        self._intervals = normalized_intervals
        self._cooldown_s = _timestamp(cooldown_s, "cooldown_s")
        self._event_triggers = self._validated_event_set(
            event_triggers,
            "event_triggers",
        )
        self._blocking_event_types = self._validated_event_set(
            blocking_event_types,
            "blocking_event_types",
        )
        if not self._blocking_event_types <= self._event_triggers:
            raise ValueError("blocking event types must also be event triggers")

        self._active_skill: dict[str, str] = {}
        self._next_periodic_due: dict[str, float | None] = {}
        self._cooldown_until: dict[str, float] = {}
        self._last_timestamp: dict[str, float] = {}
        self._inflight: dict[str, ReviewTicket] = {}
        self._lock = RLock()

    @staticmethod
    def _validated_event_set(
        value: object,
        field_name: str,
    ) -> frozenset[MissionEventType]:
        if not isinstance(value, (frozenset, set, tuple, list)):
            raise TypeError(f"{field_name} must be a collection")
        if any(not isinstance(item, MissionEventType) for item in value):
            raise TypeError(f"{field_name} must contain MissionEventType values")
        return frozenset(value)

    @property
    def intervals_s(self) -> dict[str, float]:
        return dict(self._intervals)

    @property
    def cooldown_s(self) -> float:
        return self._cooldown_s

    def schedule(
        self,
        *,
        mission_id: str,
        uav_id: str,
        plan_version: int,
        skill_name: str,
        timestamp_s: float,
        event: MissionEvent | None = None,
        request_id: str | None = None,
        review_id: str | None = None,
    ) -> ReviewScheduleDecision:
        """Evaluate triggers and reserve a review atomically when due."""

        mission = validate_mission_id(mission_id)
        uav = validate_uav_id(uav_id)
        version = _positive_plan_version(plan_version)
        skill = _skill_name(skill_name)
        now = _timestamp(timestamp_s)
        if event is not None:
            if not isinstance(event, MissionEvent):
                raise TypeError("event must be a MissionEvent or None")
            if event.mission_id != mission or event.uav_id != uav:
                raise ValueError("event routing IDs do not match review request")
            if event.plan_version != version:
                raise ValueError("event plan_version does not match review request")
            if event.timestamp_s > now:
                raise ValueError("event timestamp cannot be in the future")
        normalized_request_id = (
            None if request_id is None else validate_request_id(request_id)
        )
        normalized_review_id = (
            None if review_id is None else validate_review_id(review_id)
        )

        with self._lock:
            self._require_monotonic(uav, now)
            if self._active_skill.get(uav) != skill:
                self._active_skill[uav] = skill
                interval = self._intervals.get(skill)
                self._next_periodic_due[uav] = (
                    None if interval is None else now + interval
                )

            if uav in self._inflight:
                return ReviewScheduleDecision(ReviewScheduleReason.IN_FLIGHT)

            trigger: ReviewTrigger | None = None
            blocking = False
            event_id: str | None = None
            if event is not None and event.event_type in self._event_triggers:
                trigger = (
                    ReviewTrigger.PERIODIC
                    if event.event_type is MissionEventType.PERIODIC_REVIEW_DUE
                    else ReviewTrigger.EVENT
                )
                blocking = (
                    event.event_type is not MissionEventType.PERIODIC_REVIEW_DUE
                    and event.event_type in self._blocking_event_types
                )
                event_id = event.event_id
            else:
                due = self._next_periodic_due.get(uav)
                if due is not None and now >= due:
                    trigger = ReviewTrigger.PERIODIC

            if trigger is None:
                return ReviewScheduleDecision(ReviewScheduleReason.NOT_DUE)
            if now < self._cooldown_until.get(uav, 0.0):
                return ReviewScheduleDecision(ReviewScheduleReason.COOLDOWN)

            selected_request_id = (
                generate_routing_id("request")
                if normalized_request_id is None
                else normalized_request_id
            )
            selected_review_id = (
                generate_routing_id("review")
                if normalized_review_id is None
                else normalized_review_id
            )
            ticket = ReviewTicket(
                request_id=selected_request_id,
                review_id=selected_review_id,
                mission_id=mission,
                uav_id=uav,
                plan_version=version,
                skill_name=skill,
                submitted_timestamp_s=now,
                trigger=trigger,
                blocking=blocking,
                event_id=event_id,
            )
            self._inflight[uav] = ticket
            self._cooldown_until[uav] = now + self._cooldown_s
            interval = self._intervals.get(skill)
            self._next_periodic_due[uav] = (
                None if interval is None else now + interval
            )
            return ReviewScheduleDecision(
                ReviewScheduleReason.SCHEDULED,
                ticket,
            )

    def mark_completed(
        self,
        *,
        uav_id: str,
        request_id: str,
        review_id: str,
        timestamp_s: float,
    ) -> ReviewTicket:
        """Release exactly the matching per-UAV reservation."""

        return self._finish(
            uav_id=uav_id,
            request_id=request_id,
            review_id=review_id,
            timestamp_s=timestamp_s,
        )

    def mark_timed_out(
        self,
        *,
        uav_id: str,
        request_id: str,
        review_id: str,
        timestamp_s: float,
    ) -> ReviewTicket:
        return self._finish(
            uav_id=uav_id,
            request_id=request_id,
            review_id=review_id,
            timestamp_s=timestamp_s,
        )

    def inflight(self, *, uav_id: str) -> ReviewTicket | None:
        uav = validate_uav_id(uav_id)
        with self._lock:
            return self._inflight.get(uav)

    def reset_uav(self, *, uav_id: str) -> None:
        """Clear scheduler state only when no request is in flight."""

        uav = validate_uav_id(uav_id)
        with self._lock:
            if uav in self._inflight:
                raise RuntimeError("cannot reset a UAV with an in-flight review")
            self._active_skill.pop(uav, None)
            self._next_periodic_due.pop(uav, None)
            self._cooldown_until.pop(uav, None)
            self._last_timestamp.pop(uav, None)

    def _finish(
        self,
        *,
        uav_id: str,
        request_id: str,
        review_id: str,
        timestamp_s: float,
    ) -> ReviewTicket:
        uav = validate_uav_id(uav_id)
        request = validate_request_id(request_id)
        review = validate_review_id(review_id)
        now = _timestamp(timestamp_s)
        with self._lock:
            ticket = self._inflight.get(uav)
            if ticket is None:
                raise ValueError("no review is in flight for this uav_id")
            if ticket.request_id != request or ticket.review_id != review:
                raise ValueError("review completion IDs do not match in-flight request")
            if now < ticket.submitted_timestamp_s:
                raise ValueError("review completion timestamp precedes submission")
            self._require_monotonic(uav, now)
            del self._inflight[uav]
            self._cooldown_until[uav] = max(
                self._cooldown_until.get(uav, 0.0),
                now + self._cooldown_s,
            )
            return ticket

    def _require_monotonic(self, uav_id: str, timestamp_s: float) -> None:
        previous = self._last_timestamp.get(uav_id)
        if previous is not None and timestamp_s < previous:
            raise ValueError("review scheduler timestamp moved backwards")
        self._last_timestamp[uav_id] = timestamp_s


__all__ = [
    "DEFAULT_BLOCKING_EVENT_TYPES",
    "DEFAULT_EVENT_TRIGGERS",
    "DEFAULT_REVIEW_INTERVALS_S",
    "ReviewScheduleDecision",
    "ReviewScheduleReason",
    "ReviewScheduler",
    "ReviewTicket",
    "ReviewTrigger",
]
