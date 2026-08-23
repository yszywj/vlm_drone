"""Trusted collision HOLD and route-resume state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from common.ids import generate_routing_id, validate_routing_id
from runtime.events import EventSeverity, MissionEvent, MissionEventType
from runtime.hazard_fusion import HazardFusionResult


class CollisionSupervisorState(str, Enum):
    CLEAR = "CLEAR"
    HAZARD_SUSPECTED = "HAZARD_SUSPECTED"
    BRAKING = "BRAKING"
    HOLDING = "HOLDING"
    GEOMETRY_GROUNDED = "GEOMETRY_GROUNDED"
    REPLANNING = "REPLANNING"
    READY_TO_RESUME = "READY_TO_RESUME"


class CollisionSupervisorAction(str, Enum):
    NONE = "NONE"
    REQUEST_HOLD = "REQUEST_HOLD"
    KEEP_HOLDING = "KEEP_HOLDING"
    START_REPLANNING = "START_REPLANNING"
    RESUME = "RESUME"


@dataclass(frozen=True, slots=True)
class CollisionSupervisorDecision:
    state: CollisionSupervisorState
    action: CollisionSupervisorAction
    should_hold: bool
    may_generate_route: bool
    may_resume: bool
    events: tuple[MissionEvent, ...] = ()
    transitions: tuple[CollisionSupervisorState, ...] = ()


class CollisionSupervisor:
    """Prevent ungrounded reports or unchecked routes from resuming flight."""

    def __init__(self) -> None:
        self._state = CollisionSupervisorState.CLEAR
        self._last_fusion: HazardFusionResult | None = None
        self._proposed_route_id: str | None = None
        self._last_visible_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    @property
    def state(self) -> CollisionSupervisorState:
        return self._state

    @property
    def should_hold(self) -> bool:
        return self._state is not CollisionSupervisorState.CLEAR

    def evaluate(self, fusion: HazardFusionResult) -> CollisionSupervisorDecision:
        if not isinstance(fusion, HazardFusionResult):
            raise TypeError("fusion must be a HazardFusionResult")
        if self._last_fusion is not None and self._state is not CollisionSupervisorState.CLEAR:
            if (
                fusion.mission_id != self._last_fusion.mission_id
                or fusion.uav_id != self._last_fusion.uav_id
                or fusion.plan_version != self._last_fusion.plan_version
            ):
                raise ValueError("active collision supervision cannot change routing")
        # Once HOLD supervision is active, a later frame sampled after the
        # controller has stopped naturally has no active corridor/TTC.  It
        # must not erase the geometry-grounded evidence that triggered HOLD
        # before route generation consumes it.
        if self._state is CollisionSupervisorState.CLEAR or fusion.should_hold:
            self._last_fusion = fusion
        events: list[MissionEvent] = []
        visible_signature = (
            tuple(fusion.visible_obstacle_ids),
            tuple(dict.fromkeys(report.source.value for report in fusion.reports)),
        )
        if fusion.visible_obstacle_ids and visible_signature != self._last_visible_signature:
            events.append(
                self._event(
                    MissionEventType.OBSTACLE_VISIBLE,
                    EventSeverity.INFO,
                    {
                        "obstacle_ids": list(fusion.visible_obstacle_ids),
                        "sources": [report.source.value for report in fusion.reports],
                    },
                    timestamp_s=fusion.timestamp_s,
                )
            )
            self._last_visible_signature = visible_signature
        elif not fusion.visible_obstacle_ids:
            # A later re-entry is a new visibility transition and should be
            # reported once.  Identical 30 Hz observations are deliberately
            # coalesced so they cannot evict safety-critical events/logs.
            self._last_visible_signature = None

        if not fusion.should_hold:
            action = (
                CollisionSupervisorAction.NONE
                if self._state is CollisionSupervisorState.CLEAR
                else CollisionSupervisorAction.KEEP_HOLDING
            )
            return self._decision(action=action, events=events)

        if self._state is not CollisionSupervisorState.CLEAR:
            return self._decision(
                action=CollisionSupervisorAction.KEEP_HOLDING,
                events=events,
            )

        transitions = [CollisionSupervisorState.HAZARD_SUSPECTED]
        self._state = CollisionSupervisorState.HAZARD_SUSPECTED
        risk_type = (
            MissionEventType.IMMINENT_COLLISION
            if fusion.imminent_collision
            else MissionEventType.PATH_BLOCKED
        )
        events.append(
            self._event(
                risk_type,
                (
                    EventSeverity.CRITICAL
                    if fusion.imminent_collision
                    else EventSeverity.ERROR
                ),
                self._hazard_payload(fusion),
                timestamp_s=fusion.timestamp_s,
            )
        )
        self._state = CollisionSupervisorState.BRAKING
        transitions.append(self._state)
        events.append(
            self._event(
                MissionEventType.HOLD_REQUESTED,
                EventSeverity.ERROR,
                self._hazard_payload(fusion),
                timestamp_s=fusion.timestamp_s,
            )
        )
        return self._decision(
            action=CollisionSupervisorAction.REQUEST_HOLD,
            events=events,
            transitions=transitions,
        )

    ingest = evaluate

    def mark_hold_established(self, *, timestamp_s: float) -> CollisionSupervisorDecision:
        self._require_state(CollisionSupervisorState.BRAKING)
        self._state = CollisionSupervisorState.HOLDING
        return self._decision(
            action=CollisionSupervisorAction.KEEP_HOLDING,
            events=(
                self._event(
                    MissionEventType.HOLD_ESTABLISHED,
                    EventSeverity.WARNING,
                    self._hazard_payload(self._require_fusion()),
                    timestamp_s=timestamp_s,
                ),
            ),
            transitions=(self._state,),
        )

    hold_established = mark_hold_established

    def mark_geometry_grounded(self) -> CollisionSupervisorDecision:
        self._require_state(CollisionSupervisorState.HOLDING)
        fusion = self._require_fusion()
        if not fusion.can_generate_route:
            raise RuntimeError(
                "route generation requires a geometry-grounded active hazard"
            )
        self._state = CollisionSupervisorState.GEOMETRY_GROUNDED
        return self._decision(
            action=CollisionSupervisorAction.KEEP_HOLDING,
            transitions=(self._state,),
        )

    geometry_grounded = mark_geometry_grounded

    def begin_replanning(self) -> CollisionSupervisorDecision:
        self._require_state(CollisionSupervisorState.GEOMETRY_GROUNDED)
        self._state = CollisionSupervisorState.REPLANNING
        return self._decision(
            action=CollisionSupervisorAction.START_REPLANNING,
            transitions=(self._state,),
        )

    def route_proposed(
        self,
        route_id: str,
        *,
        timestamp_s: float,
    ) -> CollisionSupervisorDecision:
        self._require_state(CollisionSupervisorState.REPLANNING)
        self._proposed_route_id = validate_routing_id(route_id, "route_id")
        return self._decision(
            action=CollisionSupervisorAction.KEEP_HOLDING,
            events=(
                self._event(
                    MissionEventType.ROUTE_PROPOSED,
                    EventSeverity.INFO,
                    {"route_id": self._proposed_route_id},
                    timestamp_s=timestamp_s,
                ),
            ),
        )

    def route_rejected(
        self,
        *,
        reason_codes: Iterable[str],
        timestamp_s: float,
    ) -> CollisionSupervisorDecision:
        self._require_state(CollisionSupervisorState.REPLANNING)
        route_id = self._require_route_id()
        reasons = tuple(
            validate_routing_id(item, f"reason_codes[{index}]")
            for index, item in enumerate(reason_codes)
        )
        if not reasons:
            raise ValueError("reason_codes must not be empty")
        self._proposed_route_id = None
        return self._decision(
            action=CollisionSupervisorAction.KEEP_HOLDING,
            events=(
                self._event(
                    MissionEventType.ROUTE_REJECTED,
                    EventSeverity.WARNING,
                    {"route_id": route_id, "reason_codes": list(reasons)},
                    timestamp_s=timestamp_s,
                ),
            ),
        )

    def route_accepted(
        self,
        *,
        validation_mode: str,
        required_checks_passed: bool,
        timestamp_s: float,
    ) -> CollisionSupervisorDecision:
        self._require_state(CollisionSupervisorState.REPLANNING)
        route_id = self._require_route_id()
        if validation_mode not in {"open_sim", "critic_sim", "strict"}:
            raise ValueError(
                "validation_mode must be open_sim, critic_sim, or strict"
            )
        if not isinstance(required_checks_passed, bool):
            raise TypeError("required_checks_passed must be a bool")
        if not required_checks_passed:
            raise RuntimeError(
                "route cannot be accepted until required checks have passed"
            )
        self._state = CollisionSupervisorState.READY_TO_RESUME
        return self._decision(
            action=CollisionSupervisorAction.KEEP_HOLDING,
            events=(
                self._event(
                    MissionEventType.ROUTE_ACCEPTED,
                    EventSeverity.INFO,
                    {"route_id": route_id, "validation_mode": validation_mode},
                    timestamp_s=timestamp_s,
                ),
            ),
            transitions=(self._state,),
        )

    def resume(self, *, required_checks_passed: bool) -> CollisionSupervisorDecision:
        self._require_state(CollisionSupervisorState.READY_TO_RESUME)
        if required_checks_passed is not True:
            raise RuntimeError("flight resume requires the accepted route checks")
        self._state = CollisionSupervisorState.CLEAR
        self._last_fusion = None
        self._proposed_route_id = None
        return self._decision(
            action=CollisionSupervisorAction.RESUME,
            transitions=(self._state,),
        )

    def reset(self) -> None:
        """Reset state only at an explicit episode/runtime boundary."""

        self._state = CollisionSupervisorState.CLEAR
        self._last_fusion = None
        self._proposed_route_id = None
        self._last_visible_signature = None

    def _decision(
        self,
        *,
        action: CollisionSupervisorAction,
        events: Iterable[MissionEvent] = (),
        transitions: Iterable[CollisionSupervisorState] = (),
    ) -> CollisionSupervisorDecision:
        return CollisionSupervisorDecision(
            state=self._state,
            action=action,
            should_hold=self._state is not CollisionSupervisorState.CLEAR,
            may_generate_route=self._state
            in {
                CollisionSupervisorState.GEOMETRY_GROUNDED,
                CollisionSupervisorState.REPLANNING,
            },
            may_resume=self._state is CollisionSupervisorState.READY_TO_RESUME,
            events=tuple(events),
            transitions=tuple(transitions),
        )

    def _event(
        self,
        event_type: MissionEventType,
        severity: EventSeverity,
        payload: dict[str, object],
        *,
        timestamp_s: float,
    ) -> MissionEvent:
        fusion = self._require_fusion()
        return MissionEvent(
            event_id=generate_routing_id("event"),
            mission_id=fusion.mission_id,
            uav_id=fusion.uav_id,
            plan_version=fusion.plan_version,
            timestamp_s=timestamp_s,
            event_type=event_type,
            severity=severity,
            payload=payload,
        )

    @staticmethod
    def _hazard_payload(fusion: HazardFusionResult) -> dict[str, object]:
        return {
            "obstacle_ids": list(fusion.obstacle_ids),
            "sources": [report.source.value for report in fusion.reports if report.hazard_detected],
            "source_provenance": [
                {
                    "source": report.source.value,
                    "privileged": report.privileged,
                }
                for report in fusion.reports
                if report.hazard_detected
            ],
            "geometry_grounded": fusion.geometry_grounded,
            "minimum_ttc_s": fusion.minimum_ttc_s,
            "braking_distance_m": fusion.braking_distance_m,
        }

    def _require_fusion(self) -> HazardFusionResult:
        if self._last_fusion is None:
            raise RuntimeError("a fused hazard report is required")
        return self._last_fusion

    def _require_state(self, expected: CollisionSupervisorState) -> None:
        if self._state is not expected:
            raise RuntimeError(
                f"collision supervisor state must be {expected.value}, got {self._state.value}"
            )

    def _require_route_id(self) -> str:
        if self._proposed_route_id is None:
            raise RuntimeError("a proposed route_id is required")
        return self._proposed_route_id


CollisionState = CollisionSupervisorState
CollisionAction = CollisionSupervisorAction
CollisionDecision = CollisionSupervisorDecision


__all__ = [
    "CollisionAction",
    "CollisionDecision",
    "CollisionState",
    "CollisionSupervisor",
    "CollisionSupervisorAction",
    "CollisionSupervisorDecision",
    "CollisionSupervisorState",
]
