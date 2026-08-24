"""Shared target claims for a fleet of otherwise independent MissionAgents.

The registry is deliberately simulator- and perception-independent.  A local
agent may only claim the target bound to its trusted assignment, and an
``EXCLUSIVE`` conflict is returned as an explicit decision instead of being
silently counted as two successful missions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from numbers import Real
from threading import RLock

from common.ids import validate_routing_id, validate_uav_id


class TargetClaimState(str, Enum):
    UNCLAIMED = "UNCLAIMED"
    PROVISIONAL = "PROVISIONAL"
    EXCLUSIVE = "EXCLUSIVE"
    SHARED = "SHARED"
    RELEASED = "RELEASED"
    TERMINATED = "TERMINATED"


class TargetClaimPolicy(str, Enum):
    EXCLUSIVE = "EXCLUSIVE"
    SHARED = "SHARED"


class TargetClaimError(ValueError):
    """Raised when a claim violates trusted assignment routing."""


def _target_id(value: object, name: str = "target_runtime_id") -> str:
    return validate_routing_id(value, name)


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("confidence must be a finite number")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be between zero and one")
    return result


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timestamp_s must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError("timestamp_s must be a finite non-negative number")
    return result


@dataclass(frozen=True, slots=True)
class TargetClaim:
    target_runtime_id: str
    semantic_alias: str
    assignment_id: str
    uav_id: str
    state: TargetClaimState
    confidence: float
    timestamp_s: float
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_runtime_id", _target_id(self.target_runtime_id))
        if not isinstance(self.semantic_alias, str) or not self.semantic_alias.strip():
            raise ValueError("semantic_alias must be a non-empty string")
        object.__setattr__(self, "semantic_alias", self.semantic_alias.strip())
        object.__setattr__(
            self,
            "assignment_id",
            validate_routing_id(self.assignment_id, "assignment_id"),
        )
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if not isinstance(self.state, TargetClaimState):
            object.__setattr__(self, "state", TargetClaimState(self.state))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s))
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")

    @property
    def active(self) -> bool:
        return self.state in {
            TargetClaimState.PROVISIONAL,
            TargetClaimState.EXCLUSIVE,
            TargetClaimState.SHARED,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "target_runtime_id": self.target_runtime_id,
            "semantic_alias": self.semantic_alias,
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "state": self.state.value,
            "confidence": self.confidence,
            "timestamp_s": self.timestamp_s,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class SharedTargetRecord:
    target_runtime_id: str
    semantic_alias: str
    claims: tuple[TargetClaim, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_runtime_id", _target_id(self.target_runtime_id))
        if not isinstance(self.semantic_alias, str) or not self.semantic_alias.strip():
            raise ValueError("semantic_alias must be a non-empty string")
        object.__setattr__(self, "semantic_alias", self.semantic_alias.strip())
        claims = tuple(self.claims)
        if any(not isinstance(claim, TargetClaim) for claim in claims):
            raise TypeError("claims must contain TargetClaim values")
        if any(claim.target_runtime_id != self.target_runtime_id for claim in claims):
            raise ValueError("claim target_runtime_id does not match record")
        object.__setattr__(self, "claims", claims)

    @property
    def active_claims(self) -> tuple[TargetClaim, ...]:
        return tuple(claim for claim in self.claims if claim.active)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_runtime_id": self.target_runtime_id,
            "semantic_alias": self.semantic_alias,
            "claims": [claim.to_dict() for claim in self.claims],
        }


@dataclass(frozen=True, slots=True)
class TargetClaimDecision:
    accepted: bool
    event_type: str
    claim: TargetClaim
    winner: TargetClaim | None = None
    loser_uav_id: str | None = None
    reason: str = ""

    @property
    def conflict(self) -> bool:
        return self.event_type == "TARGET_CLAIM_CONFLICT"

    @property
    def hold_uav_id(self) -> str | None:
        return self.loser_uav_id if self.conflict else None

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "event_type": self.event_type,
            "claim": self.claim.to_dict(),
            "winner": None if self.winner is None else self.winner.to_dict(),
            "loser_uav_id": self.loser_uav_id,
            "reason": self.reason,
        }


class SharedTargetRegistry:
    """Thread-safe assignment allowlist and shared target claim registry."""

    MAX_RETAINED_EVENTS = 256

    def __init__(
        self,
        claim_policy: TargetClaimPolicy | str = TargetClaimPolicy.EXCLUSIVE,
    ) -> None:
        self._policy = TargetClaimPolicy(claim_policy)
        self._records: dict[str, SharedTargetRecord] = {}
        self._assignments: dict[str, tuple[str, str, str, int]] = {}
        self._events: list[TargetClaimDecision] = []
        self._event_count = 0
        self._lock = RLock()

    @property
    def claim_policy(self) -> TargetClaimPolicy:
        return self._policy

    @property
    def events(self) -> tuple[TargetClaimDecision, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def records(self) -> tuple[SharedTargetRecord, ...]:
        """Return an immutable, deterministically ordered registry view."""

        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def register_target(self, target_runtime_id: str, semantic_alias: str) -> None:
        target_runtime_id = _target_id(target_runtime_id)
        if not isinstance(semantic_alias, str) or not semantic_alias.strip():
            raise ValueError("semantic_alias must be a non-empty string")
        semantic_alias = semantic_alias.strip()
        with self._lock:
            existing = self._records.get(target_runtime_id)
            if existing is not None and existing.semantic_alias != semantic_alias:
                raise TargetClaimError("target ID is already registered with another alias")
            if existing is None:
                self._records[target_runtime_id] = SharedTargetRecord(
                    target_runtime_id,
                    semantic_alias,
                )

    def bind_assignment(
        self,
        *,
        assignment_id: str,
        uav_id: str,
        target_runtime_id: str,
        semantic_alias: str,
        priority: int = 0,
        timestamp_s: float = 0.0,
        provisional: bool = True,
    ) -> TargetClaim:
        """Create the trusted UAV->target binding and its pre-bound claim."""

        assignment_id = validate_routing_id(assignment_id, "assignment_id")
        uav_id = validate_uav_id(uav_id)
        target_runtime_id = _target_id(target_runtime_id)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError("priority must be an integer")
        self.register_target(target_runtime_id, semantic_alias)
        with self._lock:
            previous = self._assignments.get(assignment_id)
            binding = (uav_id, target_runtime_id, semantic_alias.strip(), priority)
            if previous is not None and previous != binding:
                raise TargetClaimError("assignment_id is already bound differently")
            if any(
                bound_uav == uav_id and other_assignment != assignment_id
                for other_assignment, (bound_uav, _, _, _) in self._assignments.items()
            ):
                raise TargetClaimError("UAV already has another active assignment")
            self._assignments[assignment_id] = binding
        return self.claim(
            assignment_id=assignment_id,
            uav_id=uav_id,
            target_runtime_id=target_runtime_id,
            confidence=1.0,
            timestamp_s=timestamp_s,
            state=(
                TargetClaimState.PROVISIONAL
                if provisional
                else (
                    TargetClaimState.EXCLUSIVE
                    if self._policy is TargetClaimPolicy.EXCLUSIVE
                    else TargetClaimState.SHARED
                )
            ),
        ).claim

    def claim(
        self,
        *,
        assignment_id: str,
        uav_id: str,
        target_runtime_id: str,
        confidence: float,
        timestamp_s: float,
        state: TargetClaimState | str | None = None,
    ) -> TargetClaimDecision:
        assignment_id = validate_routing_id(assignment_id, "assignment_id")
        uav_id = validate_uav_id(uav_id)
        target_runtime_id = _target_id(target_runtime_id)
        confidence = _confidence(confidence)
        timestamp_s = _timestamp(timestamp_s)
        desired_state = (
            TargetClaimState.EXCLUSIVE
            if state is None and self._policy is TargetClaimPolicy.EXCLUSIVE
            else TargetClaimState.SHARED
            if state is None
            else TargetClaimState(state)
        )
        if desired_state not in {
            TargetClaimState.PROVISIONAL,
            TargetClaimState.EXCLUSIVE,
            TargetClaimState.SHARED,
        }:
            raise TargetClaimError("claim state must be PROVISIONAL, EXCLUSIVE, or SHARED")

        with self._lock:
            binding = self._assignments.get(assignment_id)
            if binding is None:
                raise TargetClaimError("assignment is not registered")
            bound_uav, bound_target, alias, priority = binding
            if bound_uav != uav_id or bound_target != target_runtime_id:
                raise TargetClaimError(
                    "claim does not match the trusted assignment UAV/target binding"
                )
            record = self._records[target_runtime_id]
            previous_for_assignment = next(
                (
                    existing
                    for existing in record.claims
                    if existing.assignment_id == assignment_id
                ),
                None,
            )
            if (
                previous_for_assignment is not None
                and previous_for_assignment.state is TargetClaimState.TERMINATED
            ):
                raise TargetClaimError("terminated claim cannot be reactivated")
            if (
                previous_for_assignment is not None
                and timestamp_s < previous_for_assignment.timestamp_s
            ):
                raise TargetClaimError("claim timestamp must not decrease")
            claim = TargetClaim(
                target_runtime_id=target_runtime_id,
                semantic_alias=alias,
                assignment_id=assignment_id,
                uav_id=uav_id,
                state=desired_state,
                confidence=confidence,
                timestamp_s=timestamp_s,
                priority=priority,
            )
            others = tuple(
                existing
                for existing in record.active_claims
                if existing.assignment_id != assignment_id
            )
            if (
                self._policy is TargetClaimPolicy.EXCLUSIVE
                and desired_state is not TargetClaimState.PROVISIONAL
                and others
            ):
                contenders = (*others, claim)
                winner = min(
                    contenders,
                    key=lambda item: (-item.priority, item.uav_id, item.assignment_id),
                )
                accepted = winner.assignment_id == assignment_id
                loser = claim if not accepted else min(
                    others,
                    key=lambda item: (item.priority, item.uav_id, item.assignment_id),
                )
                if accepted:
                    losing_assignment_ids = {
                        existing.assignment_id for existing in others
                    }
                    claims = tuple(
                        replace(existing, state=TargetClaimState.RELEASED)
                        if existing.assignment_id in losing_assignment_ids
                        else existing
                        for existing in record.claims
                    )
                    claims = _upsert_claim(claims, claim)
                else:
                    # A rejected contender must not retain the PROVISIONAL
                    # claim created by bind_assignment().  Leaving it active
                    # makes the registry report two live exclusive owners even
                    # though this decision explicitly rejected one of them.
                    claims = _upsert_claim(
                        record.claims,
                        replace(claim, state=TargetClaimState.RELEASED),
                    )
                self._records[target_runtime_id] = replace(record, claims=claims)
                decision = TargetClaimDecision(
                    accepted=accepted,
                    event_type="TARGET_CLAIM_CONFLICT",
                    claim=claim,
                    winner=winner,
                    loser_uav_id=loser.uav_id,
                    reason="exclusive target already has another active claimant",
                )
                self._record_event(decision)
                return decision

            self._records[target_runtime_id] = replace(
                record,
                claims=_upsert_claim(record.claims, claim),
            )
            decision = TargetClaimDecision(
                accepted=True,
                event_type="TARGET_CLAIM_ACCEPTED",
                claim=claim,
                winner=claim,
            )
            self._record_event(decision)
            return decision

    def _record_event(self, decision: TargetClaimDecision) -> None:
        self._event_count += 1
        self._events.append(decision)
        overflow = len(self._events) - self.MAX_RETAINED_EVENTS
        if overflow > 0:
            del self._events[:overflow]

    def release(self, assignment_id: str, *, timestamp_s: float) -> TargetClaim:
        return self._set_terminal_state(
            assignment_id,
            TargetClaimState.RELEASED,
            timestamp_s,
        )

    def terminate(self, assignment_id: str, *, timestamp_s: float) -> TargetClaim:
        return self._set_terminal_state(
            assignment_id,
            TargetClaimState.TERMINATED,
            timestamp_s,
        )

    def _set_terminal_state(
        self,
        assignment_id: str,
        state: TargetClaimState,
        timestamp_s: float,
    ) -> TargetClaim:
        assignment_id = validate_routing_id(assignment_id, "assignment_id")
        timestamp_s = _timestamp(timestamp_s)
        with self._lock:
            binding = self._assignments.get(assignment_id)
            if binding is None:
                raise TargetClaimError("assignment is not registered")
            _, target_id, _, _ = binding
            record = self._records[target_id]
            previous = next(
                (claim for claim in record.claims if claim.assignment_id == assignment_id),
                None,
            )
            if previous is None:
                raise TargetClaimError("assignment has no claim")
            if previous.state is TargetClaimState.TERMINATED:
                if state is TargetClaimState.TERMINATED:
                    return previous
                raise TargetClaimError("terminated claim cannot change state")
            updated = replace(
                previous,
                state=state,
                timestamp_s=max(previous.timestamp_s, timestamp_s),
            )
            self._records[target_id] = replace(
                record,
                claims=_upsert_claim(record.claims, updated),
            )
            return updated

    def assigned_target(self, *, assignment_id: str, uav_id: str) -> str:
        assignment_id = validate_routing_id(assignment_id, "assignment_id")
        uav_id = validate_uav_id(uav_id)
        with self._lock:
            binding = self._assignments.get(assignment_id)
            if binding is None or binding[0] != uav_id:
                raise TargetClaimError("assignment is not bound to this UAV")
            return binding[1]

    def validate_oracle_binding(
        self,
        *,
        assignment_id: str,
        uav_id: str,
        assigned_target_id: str,
    ) -> None:
        expected = self.assigned_target(assignment_id=assignment_id, uav_id=uav_id)
        if expected != _target_id(assigned_target_id, "assigned_target_id"):
            raise TargetClaimError("Oracle backend target is outside its assignment")

    def record(self, target_runtime_id: str) -> SharedTargetRecord:
        target_runtime_id = _target_id(target_runtime_id)
        with self._lock:
            try:
                return self._records[target_runtime_id]
            except KeyError:
                raise TargetClaimError("target is not registered") from None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "claim_policy": self._policy.value,
                "targets": {
                    target_id: record.to_dict()
                    for target_id, record in sorted(self._records.items())
                },
                "event_count": self._event_count,
                "events": [event.to_dict() for event in self._events],
            }

    def summary_snapshot(self) -> dict[str, object]:
        """Return current claims and a count, leaving history in event logs."""

        with self._lock:
            return {
                "claim_policy": self._policy.value,
                "targets": {
                    target_id: record.to_dict()
                    for target_id, record in sorted(self._records.items())
                },
                "event_count": self._event_count,
            }


def _upsert_claim(
    claims: tuple[TargetClaim, ...],
    replacement: TargetClaim,
) -> tuple[TargetClaim, ...]:
    result = [
        claim for claim in claims if claim.assignment_id != replacement.assignment_id
    ]
    result.append(replacement)
    return tuple(sorted(result, key=lambda claim: claim.assignment_id))


__all__ = [
    "SharedTargetRecord",
    "SharedTargetRegistry",
    "TargetClaim",
    "TargetClaimDecision",
    "TargetClaimError",
    "TargetClaimPolicy",
    "TargetClaimState",
]
