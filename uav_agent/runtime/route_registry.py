"""Owner-thread registry preserving raw and evaluated route proposals."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from threading import get_ident

from common.ids import validate_routing_id
from planner.route_critic import RouteCritique, RouteCriticStatus
from planner.route_types import RouteDraft, RouteState
from planner.spatial_resolver import FramePose
from runtime.events import validated_json_payload


class RouteRegistryError(RuntimeError):
    """Raised when route provenance or state transitions are invalid."""


@dataclass(frozen=True, slots=True)
class RouteRecord:
    route_id: str
    frame_snapshot: FramePose
    raw_proposal: Mapping[str, object]
    route: RouteDraft
    plan_version: int
    proposal_timestamp_s: float
    critique: RouteCritique | None = None
    state: RouteState = RouteState.PROPOSED

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_id", validate_routing_id(self.route_id, "route_id"))
        if not isinstance(self.frame_snapshot, FramePose):
            raise TypeError("frame_snapshot must be a FramePose")
        if not isinstance(self.route, RouteDraft) or self.route.route_id != self.route_id:
            raise RouteRegistryError("route must match route_id")
        if isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int) or self.plan_version <= 0:
            raise RouteRegistryError("plan_version must be a positive integer")
        if isinstance(self.proposal_timestamp_s, bool) or not isinstance(self.proposal_timestamp_s, Real) or not isfinite(self.proposal_timestamp_s) or self.proposal_timestamp_s < 0:
            raise RouteRegistryError("proposal_timestamp_s must be finite and non-negative")
        object.__setattr__(self, "proposal_timestamp_s", float(self.proposal_timestamp_s))
        object.__setattr__(
            self,
            "raw_proposal",
            validated_json_payload(self.raw_proposal, field_name="raw_proposal"),
        )
        if self.critique is not None and not isinstance(self.critique, RouteCritique):
            raise TypeError("critique must be a RouteCritique or None")
        if not isinstance(self.state, RouteState):
            try:
                object.__setattr__(self, "state", RouteState(self.state))
            except (TypeError, ValueError):
                raise RouteRegistryError("state is not supported") from None

    def to_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "frame_snapshot": {
                "xyz_m": list(self.frame_snapshot.xyz_m),
                "yaw_rad": self.frame_snapshot.yaw_rad,
            },
            "raw_proposal": deepcopy(dict(self.raw_proposal)),
            "route": self.route.to_dict(),
            "plan_version": self.plan_version,
            "proposal_timestamp_s": self.proposal_timestamp_s,
            "critique": None if self.critique is None else self.critique.to_dict(),
            "state": self.state.value,
        }


class RouteRegistry:
    """Retain every proposal exactly; accepted execution never overwrites raw JSON."""

    _ALLOWED = {
        RouteState.PROPOSED: frozenset({RouteState.REJECTED, RouteState.ACCEPTED}),
        # Publication is transactional across the registry, SkillManager and
        # safety/collision supervisor.  If either downstream publication step
        # fails after a critic ACCEPT, retain the proposal but mark it rejected
        # instead of leaving a route that can never execute as ACCEPTED.
        RouteState.ACCEPTED: frozenset({RouteState.EXECUTING, RouteState.REJECTED}),
        RouteState.EXECUTING: frozenset({RouteState.COMPLETED, RouteState.COLLIDED}),
        RouteState.REJECTED: frozenset(),
        RouteState.COMPLETED: frozenset(),
        RouteState.COLLIDED: frozenset(),
    }

    def __init__(self, *, max_records: int = 128) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or not 1 <= max_records <= 1024:
            raise ValueError("max_records must be an integer within [1, 1024]")
        self._max_records = max_records
        self._owner_thread = get_ident()
        self._records: dict[str, RouteRecord] = {}
        self._order: list[str] = []

    def register(
        self,
        route: RouteDraft,
        *,
        frame_snapshot: FramePose,
        raw_proposal: Mapping[str, object],
        plan_version: int,
        proposal_timestamp_s: float,
    ) -> RouteRecord:
        self._require_owner()
        if route.route_id in self._records:
            raise RouteRegistryError("route_id is already registered")
        if len(self._records) >= self._max_records:
            raise RouteRegistryError("RouteRegistry record budget is exhausted")
        record = RouteRecord(
            route.route_id,
            frame_snapshot,
            raw_proposal,
            route,
            plan_version,
            proposal_timestamp_s,
        )
        self._records[record.route_id] = record
        self._order.append(record.route_id)
        return record

    def record_critique(self, route_id: str, critique: RouteCritique) -> RouteRecord:
        self._require_owner()
        record = self.get(route_id)
        if record.state is not RouteState.PROPOSED or record.critique is not None:
            raise RouteRegistryError("critique can only be attached once to a proposed route")
        if critique.route_id != record.route_id:
            raise RouteRegistryError("critique route_id does not match")
        state = RouteState.ACCEPTED if critique.status is RouteCriticStatus.ACCEPT else RouteState.REJECTED
        updated = replace(record, critique=critique, state=state)
        self._records[record.route_id] = updated
        return updated

    def transition(self, route_id: str, state: RouteState | str) -> RouteRecord:
        self._require_owner()
        record = self.get(route_id)
        try:
            normalized = state if isinstance(state, RouteState) else RouteState(state)
        except (TypeError, ValueError):
            raise RouteRegistryError("route state is not supported") from None
        if normalized not in self._ALLOWED[record.state]:
            raise RouteRegistryError(
                f"illegal route transition {record.state.value} -> {normalized.value}"
            )
        updated = replace(record, state=normalized)
        self._records[record.route_id] = updated
        return updated

    def get(self, route_id: str) -> RouteRecord:
        normalized = validate_routing_id(route_id, "route_id")
        try:
            return self._records[normalized]
        except KeyError:
            raise KeyError(f"unknown route_id: {normalized}") from None

    @property
    def records(self) -> tuple[RouteRecord, ...]:
        return tuple(self._records[item] for item in self._order)

    def _require_owner(self) -> None:
        if get_ident() != self._owner_thread:
            raise RouteRegistryError("RouteRegistry may only be mutated by its owner thread")


__all__ = ["RouteRecord", "RouteRegistry", "RouteRegistryError"]
