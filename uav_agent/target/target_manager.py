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
        # When REACQUIRE evaluates a new visual candidate, rejection must
        # restore the previously tracked identity/last-seen snapshot.
        self._candidate_restore: TargetSnapshot | None = None
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
        self._candidate_restore = None
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

        candidate_restore = (
            self.snapshot()
            if self._lifecycle is TargetLifecycle.REACQUIRING
            else None
        )
        timestamp = self._validated_transition(
            TargetLifecycle.CANDIDATE,
            timestamp_s,
            allowed_from={
                TargetLifecycle.SEARCHING,
                TargetLifecycle.REACQUIRING,
            },
        )
        values = self._validated_observation_values(
            target_id=target_id,
            confidence=confidence,
            source=source,
            last_seen_position=last_seen_position,
            last_seen_velocity=last_seen_velocity,
        )
        if values[2].casefold() == "oracle":
            raise ValueError(
                "set_candidate() does not accept Oracle evidence"
            )
        self._candidate_restore = candidate_restore
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
        source: str = "confirmed_vision",
        last_seen_position: tuple[float, float, float] | None = None,
        last_seen_velocity: tuple[float, float, float] | None = None,
    ) -> None:
        """Reject legacy direct locks that bypass confirmation evidence."""

        del (
            target_id,
            timestamp_s,
            confidence,
            source,
            last_seen_position,
            last_seen_velocity,
        )
        raise TargetStateError(
            "direct visual lock is disabled; use "
            "CandidateConfirmationCoordinator"
        )

    def _lock_confirmed_candidate(
        self,
        target_id: str,
        *,
        timestamp_s: float,
        confidence: float | None,
        source: str,
        last_seen_position: tuple[float, float, float] | None = None,
        last_seen_velocity: tuple[float, float, float] | None = None,
    ) -> None:
        """Commit a coordinator-validated visual evidence bundle.

        This is intentionally internal.  The public confirmation entry point
        is :class:`perception.confirmation.CandidateConfirmationCoordinator`.
        """

        timestamp = self._validated_transition(
            TargetLifecycle.LOCKED,
            timestamp_s,
            allowed_from={TargetLifecycle.CANDIDATE},
        )
        values = self._validated_observation_values(
            target_id=target_id,
            confidence=confidence,
            source=source,
            last_seen_position=last_seen_position,
            last_seen_velocity=last_seen_velocity,
        )
        if values[2].casefold() == "oracle":
            raise ValueError(
                "confirmed visual lock does not accept Oracle evidence; use "
                "an explicit Oracle evaluation shortcut"
            )
        restore = self._candidate_restore
        if (
            restore is not None
            and restore.target_id is not None
            and values[0] != restore.target_id
        ):
            raise TargetStateError(
                "reacquired visual identity does not match the previously "
                "tracked target_id"
            )
        reacquired = restore is not None
        self._apply_observation_values(*values, timestamp_s=timestamp)
        self._candidate_restore = None
        self._commit_transition(
            TargetLifecycle.LOCKED,
            timestamp,
            (
                "target_reacquired_from_candidate"
                if reacquired
                else "target_locked_from_candidate"
            ),
        )

    def lock_oracle_from_search(
        self,
        target_id: str,
        *,
        timestamp_s: float,
        confidence: float | None = 1.0,
        last_seen_position: tuple[float, float, float] | None = None,
        last_seen_velocity: tuple[float, float, float] | None = None,
    ) -> None:
        """Explicit privileged shortcut for Oracle evaluation pipelines only.

        Production visual code must use ``CandidateConfirmationCoordinator``.
        Keeping the legacy expert shortcut under an Oracle-named method
        prevents a detector caller from silently bypassing confirmation.
        """

        timestamp = self._validated_transition(
            TargetLifecycle.LOCKED,
            timestamp_s,
            allowed_from={TargetLifecycle.SEARCHING},
        )
        values = self._validated_observation_values(
            target_id=target_id,
            confidence=confidence,
            source="oracle",
            last_seen_position=last_seen_position,
            last_seen_velocity=last_seen_velocity,
        )
        self._apply_observation_values(*values, timestamp_s=timestamp)
        self._candidate_restore = None
        self._commit_transition(
            TargetLifecycle.LOCKED,
            timestamp,
            "target_locked_by_oracle_evaluation",
        )

    def reject_candidate(self, *, timestamp_s: float, reason: str) -> None:
        """Reject a candidate and return to SEARCHING or REACQUIRING.

        A SEARCH candidate is cleared so a later proposal cannot inherit its
        evidence.  A REACQUIRE candidate restores the previously tracked
        identity and last-seen state.  The mission description is preserved.
        """

        normalized_reason = _non_empty_string(reason, "reason")
        restore = self._candidate_restore
        return_state = (
            TargetLifecycle.REACQUIRING
            if restore is not None
            else TargetLifecycle.SEARCHING
        )
        timestamp = self._validated_transition(
            return_state,
            timestamp_s,
            allowed_from={TargetLifecycle.CANDIDATE},
        )
        if restore is None:
            self._target_id = None
            self._confidence = None
            self._last_seen_position = None
            self._last_seen_velocity = None
            self._last_seen_time_s = None
            self._source = None
        else:
            self._description = restore.description
            self._target_id = restore.target_id
            self._confidence = restore.confidence
            self._last_seen_position = restore.last_seen_position
            self._last_seen_velocity = restore.last_seen_velocity
            self._last_seen_time_s = restore.last_seen_time_s
            self._source = restore.source
        self._candidate_restore = None
        self._commit_transition(
            return_state,
            timestamp,
            f"candidate_rejected:{normalized_reason}",
        )

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

    def finish_tracking_segment(self, timestamp_s: float) -> None:
        """Keep the identity locked after one bounded TRACK segment ends.

        Dynamic plans may navigate between two TRACK calls for the same
        mission target.  During that navigation the target is still known,
        but it is not actively being tracked.
        """

        timestamp = self._validated_transition(
            TargetLifecycle.LOCKED,
            timestamp_s,
            allowed_from={TargetLifecycle.TRACKING},
        )
        if self._target_id is None:
            raise TargetStateError(
                "cannot finish a tracking segment without a target_id"
            )
        self._commit_transition(
            TargetLifecycle.LOCKED,
            timestamp,
            "tracking_segment_complete",
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
        source: str = "confirmed_vision",
        last_seen_position: tuple[float, float, float] | None = None,
        last_seen_velocity: tuple[float, float, float] | None = None,
    ) -> None:
        """Reject legacy direct visual reacquisition without evidence."""

        del (
            target_id,
            timestamp_s,
            confidence,
            source,
            last_seen_position,
            last_seen_velocity,
        )
        raise TargetStateError(
            "direct visual reacquisition is disabled; use "
            "CandidateConfirmationCoordinator"
        )

    def mark_reacquired_oracle(
        self,
        target_id: str | None = None,
        *,
        timestamp_s: float,
        confidence: float | None = 1.0,
        last_seen_position: tuple[float, float, float] | None = None,
        last_seen_velocity: tuple[float, float, float] | None = None,
    ) -> None:
        """Explicit privileged REACQUIRE shortcut for Oracle evaluation."""

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
            source="oracle",
            last_seen_position=last_seen_position,
            last_seen_velocity=last_seen_velocity,
        )
        self._apply_observation_values(*values, timestamp_s=timestamp)
        self._candidate_restore = None
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
        self._candidate_restore = None
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
        self._candidate_restore = None
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
