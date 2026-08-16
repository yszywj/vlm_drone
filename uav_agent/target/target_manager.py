"""Deterministic lifecycle manager for one mission target."""

from __future__ import annotations

from target.types import (
    TargetEvent,
    TargetLifecycle,
    TargetSnapshot,
    TargetSpec,
    _finite_number,
    _non_empty_string,
    _optional_confidence,
    _optional_non_empty_string,
    _optional_vector3,
)


class TargetStateError(RuntimeError):
    """Raised when a target lifecycle operation is invalid in the current state."""


_ACTIVE_STATES = frozenset(
    {
        TargetLifecycle.SEARCHING,
        TargetLifecycle.CANDIDATE,
        TargetLifecycle.LOCKED,
        TargetLifecycle.TRACKING,
        TargetLifecycle.LOST,
        TargetLifecycle.REACQUIRING,
    }
)

# A mission may be canceled or fail before SEARCH begins.  Task-level
# termination is still a real lifecycle event, but it must not fabricate an
# intermediate SEARCHING state or target identity.
_TERMINATABLE_STATES = _ACTIVE_STATES | {
    TargetLifecycle.UNINITIALIZED,
}

_UNINITIALIZED_DESCRIPTION = "uninitialized"


class TargetManager:
    """Own target metadata and lifecycle transitions without reading an environment."""

    def __init__(self) -> None:
        self._lifecycle = TargetLifecycle.UNINITIALIZED
        self._description = _UNINITIALIZED_DESCRIPTION
        self._target_id: str | None = None
        self._confidence: float | None = None
        self._last_seen_position: tuple[float, float, float] | None = None
        self._last_seen_velocity: tuple[float, float, float] | None = None
        self._last_seen_time_s: float | None = None
        self._source: str | None = None
        self._last_event_time_s: float | None = None
        self._events: list[TargetEvent] = []

    @property
    def lifecycle(self) -> TargetLifecycle:
        return self._lifecycle

    def start_search(self, target_spec: TargetSpec, timestamp_s: float) -> None:
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        timestamp = self._validated_transition(
            TargetLifecycle.SEARCHING,
            timestamp_s,
            allowed_from={TargetLifecycle.UNINITIALIZED},
        )

        self._description = target_spec.description
        self._target_id = None
        self._confidence = None
        self._last_seen_position = None
        self._last_seen_velocity = None
        self._last_seen_time_s = None
        self._source = None
        self._commit_transition(
            TargetLifecycle.SEARCHING,
            timestamp,
            "search_started",
        )

    def set_candidate(
        self,
        target_id: str,
        *,
        timestamp_s: float,
        confidence: float | None = None,
        source: str,
        last_seen_position: tuple[float, float, float] | None = None,
        last_seen_velocity: tuple[float, float, float] | None = None,
    ) -> None:
        """Record a perception candidate without claiming that it is locked."""

        timestamp = self._validated_transition(
            TargetLifecycle.CANDIDATE,
            timestamp_s,
            allowed_from={TargetLifecycle.SEARCHING},
        )
        values = self._validated_observation_values(
            target_id=target_id,
            confidence=confidence,
            source=source,
            last_seen_position=last_seen_position,
            last_seen_velocity=last_seen_velocity,
        )
        self._apply_observation_values(*values, timestamp_s=timestamp)
        self._commit_transition(
            TargetLifecycle.CANDIDATE,
            timestamp,
            "candidate_detected",
        )

    def lock(
        self,
        target_id: str,
        *,
        timestamp_s: float,
        confidence: float | None = None,
        source: str = "oracle",
        last_seen_position: tuple[float, float, float] | None = None,
        last_seen_velocity: tuple[float, float, float] | None = None,
    ) -> None:
        """Lock a candidate, including the Stage-0 SEARCHING-to-LOCKED path."""

        timestamp = self._validated_transition(
            TargetLifecycle.LOCKED,
            timestamp_s,
            allowed_from={
                TargetLifecycle.SEARCHING,
                TargetLifecycle.CANDIDATE,
            },
        )
        values = self._validated_observation_values(
            target_id=target_id,
            confidence=confidence,
            source=source,
            last_seen_position=last_seen_position,
            last_seen_velocity=last_seen_velocity,
        )
        self._apply_observation_values(*values, timestamp_s=timestamp)
        reason = (
            "target_locked_from_candidate"
            if self._lifecycle is TargetLifecycle.CANDIDATE
            else "target_locked"
        )
        self._commit_transition(TargetLifecycle.LOCKED, timestamp, reason)

    def start_tracking(self, timestamp_s: float) -> None:
        timestamp = self._validated_transition(
            TargetLifecycle.TRACKING,
            timestamp_s,
            allowed_from={TargetLifecycle.LOCKED},
        )
        if self._target_id is None:
            raise TargetStateError("cannot start tracking without a locked target_id")
        self._commit_transition(
            TargetLifecycle.TRACKING,
            timestamp,
            "tracking_started",
        )

    def mark_lost(
        self,
        *,
        timestamp_s: float,
        last_seen_position: tuple[float, float, float] | None = None,
        last_seen_velocity: tuple[float, float, float] | None = None,
        last_seen_time_s: float | None = None,
    ) -> None:
        timestamp = self._validated_transition(
            TargetLifecycle.LOST,
            timestamp_s,
            allowed_from={TargetLifecycle.TRACKING},
        )
        position = _optional_vector3(last_seen_position, "last_seen_position")
        velocity = _optional_vector3(last_seen_velocity, "last_seen_velocity")
        if last_seen_time_s is None:
            # The compact Stage-0 API historically supplied pose/velocity with
            # only the loss-event timestamp.  Treat that observation as sampled
            # at the event time instead of pairing new vectors with an old time.
            seen_time = (
                timestamp
                if position is not None or velocity is not None
                else None
            )
        else:
            seen_time = _finite_number(last_seen_time_s, "last_seen_time_s")
        if seen_time is not None and seen_time > timestamp:
            raise ValueError("last_seen_time_s cannot be later than timestamp_s")
        if (
            seen_time is not None
            and self._last_seen_time_s is not None
            and seen_time < self._last_seen_time_s
        ):
            raise TargetStateError(
                "last_seen_time_s cannot move backward: "
                f"{seen_time} < {self._last_seen_time_s}"
            )

        if position is not None:
            self._last_seen_position = position
        if velocity is not None:
            self._last_seen_velocity = velocity
        if seen_time is not None:
            self._last_seen_time_s = seen_time
        self._commit_transition(TargetLifecycle.LOST, timestamp, "target_lost")

    def start_reacquiring(self, timestamp_s: float) -> None:
        timestamp = self._validated_transition(
            TargetLifecycle.REACQUIRING,
            timestamp_s,
            allowed_from={TargetLifecycle.LOST},
        )
        self._commit_transition(
            TargetLifecycle.REACQUIRING,
            timestamp,
            "reacquisition_started",
        )

    def mark_reacquired(
        self,
        target_id: str | None = None,
        *,
        timestamp_s: float,
        confidence: float | None = None,
        source: str = "oracle",
        last_seen_position: tuple[float, float, float] | None = None,
        last_seen_velocity: tuple[float, float, float] | None = None,
    ) -> None:
        timestamp = self._validated_transition(
            TargetLifecycle.LOCKED,
            timestamp_s,
            allowed_from={TargetLifecycle.REACQUIRING},
        )
        resolved_target_id = self._target_id if target_id is None else target_id
        if resolved_target_id is None:
            raise TargetStateError("cannot mark target reacquired without a target_id")
        values = self._validated_observation_values(
            target_id=resolved_target_id,
            confidence=confidence,
            source=source,
            last_seen_position=last_seen_position,
            last_seen_velocity=last_seen_velocity,
        )
        self._apply_observation_values(*values, timestamp_s=timestamp)
        self._commit_transition(
            TargetLifecycle.LOCKED,
            timestamp,
            "target_reacquired",
        )

    def terminate(self, timestamp_s: float, reason: str) -> None:
        normalized_reason = _non_empty_string(reason, "reason")
        timestamp = self._validated_transition(
            TargetLifecycle.TERMINATED,
            timestamp_s,
            allowed_from=_TERMINATABLE_STATES,
        )
        self._commit_transition(
            TargetLifecycle.TERMINATED,
            timestamp,
            normalized_reason,
        )

    def reset(self) -> None:
        """Clear a terminated target so the manager can serve another mission."""

        if self._lifecycle is not TargetLifecycle.TERMINATED:
            raise TargetStateError(
                "reset requires lifecycle TERMINATED; "
                f"current lifecycle is {self._lifecycle.value}"
            )
        self._lifecycle = TargetLifecycle.UNINITIALIZED
        self._description = _UNINITIALIZED_DESCRIPTION
        self._target_id = None
        self._confidence = None
        self._last_seen_position = None
        self._last_seen_velocity = None
        self._last_seen_time_s = None
        self._source = None
        self._last_event_time_s = None
        self._events.clear()

    def snapshot(self) -> TargetSnapshot:
        """Return a fresh immutable snapshot of the current target state."""

        return TargetSnapshot(
            target_id=self._target_id,
            description=self._description,
            lifecycle=self._lifecycle,
            confidence=self._confidence,
            last_seen_position=self._last_seen_position,
            last_seen_velocity=self._last_seen_velocity,
            last_seen_time_s=self._last_seen_time_s,
            source=self._source,
        )

    def events(self) -> tuple[TargetEvent, ...]:
        """Return a read-only snapshot of lifecycle events."""

        return tuple(self._events)

    def _validated_transition(
        self,
        new_state: TargetLifecycle,
        timestamp_s: float,
        *,
        allowed_from: set[TargetLifecycle] | frozenset[TargetLifecycle],
    ) -> float:
        if self._lifecycle not in allowed_from:
            allowed = ", ".join(sorted(state.value for state in allowed_from))
            raise TargetStateError(
                f"cannot transition {self._lifecycle.value} -> {new_state.value}; "
                f"expected one of: {allowed}"
            )
        timestamp = _finite_number(timestamp_s, "timestamp_s")
        if (
            self._last_event_time_s is not None
            and timestamp < self._last_event_time_s
        ):
            raise TargetStateError(
                "timestamp_s cannot move backward: "
                f"{timestamp} < {self._last_event_time_s}"
            )
        return timestamp

    @staticmethod
    def _validated_observation_values(
        *,
        target_id: object,
        confidence: object,
        source: object,
        last_seen_position: object,
        last_seen_velocity: object,
    ) -> tuple[
        str,
        float | None,
        str,
        tuple[float, float, float] | None,
        tuple[float, float, float] | None,
    ]:
        normalized_target_id = _non_empty_string(target_id, "target_id")
        normalized_confidence = _optional_confidence(confidence)
        normalized_source = _optional_non_empty_string(source, "source")
        if normalized_source is None:  # defensive for future callers/type erasure
            raise ValueError("source must be non-empty")
        return (
            normalized_target_id,
            normalized_confidence,
            normalized_source,
            _optional_vector3(last_seen_position, "last_seen_position"),
            _optional_vector3(last_seen_velocity, "last_seen_velocity"),
        )

    def _apply_observation_values(
        self,
        target_id: str,
        confidence: float | None,
        source: str,
        last_seen_position: tuple[float, float, float] | None,
        last_seen_velocity: tuple[float, float, float] | None,
        *,
        timestamp_s: float,
    ) -> None:
        self._target_id = target_id
        self._confidence = confidence
        self._source = source
        if last_seen_position is not None:
            self._last_seen_position = last_seen_position
        if last_seen_velocity is not None:
            self._last_seen_velocity = last_seen_velocity
        self._last_seen_time_s = timestamp_s

    def _commit_transition(
        self,
        new_state: TargetLifecycle,
        timestamp_s: float,
        reason: str,
    ) -> None:
        event = TargetEvent(
            timestamp_s=timestamp_s,
            old_state=self._lifecycle,
            new_state=new_state,
            reason=reason,
        )
        self._lifecycle = new_state
        self._last_event_time_s = timestamp_s
        self._events.append(event)
