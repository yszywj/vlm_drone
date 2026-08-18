"""Immutable, serialization-safe mission belief snapshots.

``WorldBelief`` contains only value objects and bounded JSON summaries.  It
cannot carry an environment, controller, or model instance.  The optional
``WorldBeliefStore`` enforces that mutations are serialized by the thread that
created the store; asynchronous model workers may read snapshots and return
results, but cannot update the authoritative belief directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from numbers import Real
from threading import RLock, get_ident

from common.ids import (
    validate_mission_id,
    validate_plan_id,
    validate_request_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from common.provenance import is_privileged_oracle_source
from runtime.events import (
    MissionEvent,
    json_payload_to_dict,
    validated_json_payload,
)
from runtime.frame_store import FrameRef
from target.types import TargetSnapshot, TargetSpec


def _finite_nonnegative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return normalized


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _optional_confidence(value: object) -> float | None:
    if value is None:
        return None
    confidence = _finite_nonnegative(value, "confidence")
    if confidence > 1.0:
        raise ValueError("confidence must be at most 1.0")
    return confidence


class QwenRequestState(str, Enum):
    IDLE = "IDLE"
    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    COMPLETED = "COMPLETED"
    TIMED_OUT = "TIMED_OUT"
    STALE = "STALE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class QwenRequestStatus:
    state: QwenRequestState = QwenRequestState.IDLE
    request_id: str | None = None
    review_id: str | None = None
    blocking: bool = False
    submitted_timestamp_s: float | None = None

    def __post_init__(self) -> None:
        state = self.state
        if not isinstance(state, QwenRequestState):
            try:
                state = QwenRequestState(state)
            except (TypeError, ValueError):
                raise ValueError("state must be a supported QwenRequestState") from None
            object.__setattr__(self, "state", state)
        if not isinstance(self.blocking, bool):
            raise TypeError("blocking must be a bool")

        if state is QwenRequestState.IDLE:
            if any(
                value is not None
                for value in (
                    self.request_id,
                    self.review_id,
                    self.submitted_timestamp_s,
                )
            ) or self.blocking:
                raise ValueError("IDLE Qwen request status cannot contain request data")
            return

        object.__setattr__(
            self,
            "request_id",
            validate_request_id(self.request_id),
        )
        object.__setattr__(
            self,
            "review_id",
            validate_review_id(self.review_id),
        )
        object.__setattr__(
            self,
            "submitted_timestamp_s",
            _finite_nonnegative(
                self.submitted_timestamp_s,
                "submitted_timestamp_s",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "request_id": self.request_id,
            "review_id": self.review_id,
            "blocking": self.blocking,
            "submitted_timestamp_s": self.submitted_timestamp_s,
        }


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    candidate_id: str
    confidence: float | None
    last_seen_timestamp_s: float
    source: str
    observation_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            validate_routing_id(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "confidence",
            _optional_confidence(self.confidence),
        )
        object.__setattr__(
            self,
            "last_seen_timestamp_s",
            _finite_nonnegative(
                self.last_seen_timestamp_s,
                "last_seen_timestamp_s",
            ),
        )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(
            self,
            "observation_count",
            _positive_integer(self.observation_count, "observation_count"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "confidence": self.confidence,
            "last_seen_timestamp_s": self.last_seen_timestamp_s,
            "source": self.source,
            "observation_count": self.observation_count,
        }


@dataclass(frozen=True, slots=True)
class WorldBelief:
    mission_id: str
    uav_id: str
    plan_version: int
    current_step_id: str | None
    current_skill: str | None
    skill_feedback: Mapping[str, object] | None
    target_spec: TargetSpec | None
    target_snapshot: TargetSnapshot | None
    candidate_summaries: tuple[CandidateSummary, ...]
    recent_events: tuple[MissionEvent, ...]
    qwen_request_status: QwenRequestStatus
    latest_frame_ref: FrameRef | None
    mission_elapsed_s: float
    plan_id: str | None = None

    MAX_CANDIDATE_SUMMARIES = 64
    MAX_RECENT_EVENTS = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "plan_version",
            _positive_integer(self.plan_version, "plan_version"),
        )
        if self.plan_id is not None:
            object.__setattr__(self, "plan_id", validate_plan_id(self.plan_id))

        if (self.current_step_id is None) != (self.current_skill is None):
            raise ValueError(
                "current_step_id and current_skill must both be set or both be None"
            )
        if self.current_step_id is not None:
            object.__setattr__(
                self,
                "current_step_id",
                validate_routing_id(self.current_step_id, "current_step_id"),
            )
            normalized_skill = validate_routing_id(
                self.current_skill,
                "current_skill",
            )
            if normalized_skill != normalized_skill.upper():
                raise ValueError("current_skill must use its canonical uppercase name")
            object.__setattr__(self, "current_skill", normalized_skill)

        if self.skill_feedback is not None:
            object.__setattr__(
                self,
                "skill_feedback",
                validated_json_payload(
                    self.skill_feedback,
                    field_name="skill_feedback",
                ),
            )
        if self.target_spec is not None and not isinstance(
            self.target_spec,
            TargetSpec,
        ):
            raise TypeError("target_spec must be a TargetSpec or None")
        if self.target_snapshot is not None and not isinstance(
            self.target_snapshot,
            TargetSnapshot,
        ):
            raise TypeError("target_snapshot must be a TargetSnapshot or None")
        if (
            self.target_spec is not None
            and self.target_snapshot is not None
            and self.target_spec.description != self.target_snapshot.description
        ):
            raise ValueError("target snapshot description must match TargetSpec")

        if isinstance(self.candidate_summaries, (str, bytes)) or not isinstance(
            self.candidate_summaries,
            Sequence,
        ):
            raise TypeError("candidate_summaries must be a sequence")
        candidates = tuple(self.candidate_summaries)
        if len(candidates) > self.MAX_CANDIDATE_SUMMARIES:
            raise ValueError("candidate_summaries exceeds its bounded limit")
        if any(not isinstance(item, CandidateSummary) for item in candidates):
            raise TypeError(
                "candidate_summaries must contain only CandidateSummary values"
            )
        candidate_ids = tuple(item.candidate_id for item in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_summaries must have unique candidate_id values")
        object.__setattr__(self, "candidate_summaries", candidates)

        if isinstance(self.recent_events, (str, bytes)) or not isinstance(
            self.recent_events,
            Sequence,
        ):
            raise TypeError("recent_events must be a sequence")
        events = tuple(self.recent_events)
        if len(events) > self.MAX_RECENT_EVENTS:
            raise ValueError("recent_events exceeds its bounded limit")
        for event in events:
            if not isinstance(event, MissionEvent):
                raise TypeError("recent_events must contain only MissionEvent values")
            if event.mission_id != self.mission_id or event.uav_id != self.uav_id:
                raise ValueError("recent event routing IDs do not match WorldBelief")
            if event.plan_version > self.plan_version:
                raise ValueError("recent event cannot use a future plan_version")
        object.__setattr__(self, "recent_events", events)

        if not isinstance(self.qwen_request_status, QwenRequestStatus):
            raise TypeError("qwen_request_status must be a QwenRequestStatus")
        if self.latest_frame_ref is not None:
            if not isinstance(self.latest_frame_ref, FrameRef):
                raise TypeError("latest_frame_ref must be a FrameRef or None")
            if self.latest_frame_ref.uav_id != self.uav_id:
                raise ValueError("latest FrameRef uav_id does not match WorldBelief")
        object.__setattr__(
            self,
            "mission_elapsed_s",
            _finite_nonnegative(self.mission_elapsed_s, "mission_elapsed_s"),
        )

    def to_dict(self) -> dict[str, object]:
        target_spec_data: dict[str, object] | None = None
        if self.target_spec is not None:
            target_spec_data = self.target_spec.to_dict()
            # Preserve the stable legacy summary key while TargetSpec v2 also
            # exposes its richer original/immutable/mutable semantic fields.
            target_spec_data.setdefault("description", self.target_spec.description)
        return {
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_id": self.plan_id,
            "plan_version": self.plan_version,
            "current_step_id": self.current_step_id,
            "current_skill": self.current_skill,
            "skill_feedback": (
                None
                if self.skill_feedback is None
                else json_payload_to_dict(self.skill_feedback)
            ),
            "target_spec": target_spec_data,
            "target_snapshot": (
                None
                if self.target_snapshot is None
                else self.target_snapshot.to_dict()
            ),
            "candidate_summaries": [
                candidate.to_dict() for candidate in self.candidate_summaries
            ],
            "recent_events": [event.to_dict() for event in self.recent_events],
            "qwen_request_status": self.qwen_request_status.to_dict(),
            "latest_frame_ref": (
                None
                if self.latest_frame_ref is None
                else self.latest_frame_ref.to_dict()
            ),
            "mission_elapsed_s": self.mission_elapsed_s,
        }


WorldBeliefSnapshot = WorldBelief


class WorldBeliefThreadError(RuntimeError):
    """Raised when a worker thread attempts to mutate authoritative belief."""


class WorldBeliefStore:
    """Owner-thread serialized holder for immutable WorldBelief snapshots."""

    def __init__(self, initial: WorldBelief) -> None:
        if not isinstance(initial, WorldBelief):
            raise TypeError("initial must be a WorldBelief")
        self._current = initial
        self._owner_thread_id = get_ident()
        self._lock = RLock()

    def snapshot(self) -> WorldBelief:
        with self._lock:
            return self._current

    def update(self, **changes: object) -> WorldBelief:
        self._require_owner_thread()
        with self._lock:
            if (
                "mission_id" in changes
                and changes["mission_id"] != self._current.mission_id
            ) or (
                "uav_id" in changes
                and changes["uav_id"] != self._current.uav_id
            ):
                raise ValueError("WorldBelief routing IDs are immutable")
            updated = replace(self._current, **changes)
            self._require_same_route(updated)
            if updated.plan_version < self._current.plan_version:
                raise ValueError("WorldBelief plan_version must not decrease")
            self._current = updated
            return updated

    def set_snapshot(self, snapshot: WorldBelief) -> None:
        self._require_owner_thread()
        if not isinstance(snapshot, WorldBelief):
            raise TypeError("snapshot must be a WorldBelief")
        with self._lock:
            self._require_same_route(snapshot)
            if snapshot.plan_version < self._current.plan_version:
                raise ValueError("WorldBelief plan_version must not decrease")
            self._current = snapshot

    def _require_same_route(self, snapshot: WorldBelief) -> None:
        if (
            snapshot.mission_id != self._current.mission_id
            or snapshot.uav_id != self._current.uav_id
        ):
            raise ValueError("WorldBelief routing IDs are immutable")

    def _require_owner_thread(self) -> None:
        if get_ident() != self._owner_thread_id:
            raise WorldBeliefThreadError(
                "WorldBelief may only be updated by its owner thread"
            )


__all__ = [
    "CandidateSummary",
    "is_privileged_oracle_source",
    "QwenRequestState",
    "QwenRequestStatus",
    "WorldBelief",
    "WorldBeliefSnapshot",
    "WorldBeliefStore",
    "WorldBeliefThreadError",
]
