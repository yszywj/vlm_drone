"""Main-thread orchestration for camera hazards and immediate trusted HOLD.

The runtime deliberately stops at the route-proposal boundary.  It never
calls a model, never invents obstacle geometry, and never resumes a paused
Skill.  A separate obstacle revision coordinator may move the collision state
machine from ``GEOMETRY_GROUNDED`` through an accepted route to resume.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol

from common.obstacle_types import CameraGeometry, FlightCorridor, ObstacleObservation
from perception.ideal_obstacle_perception import IdealObstaclePerception
from runtime.collision_supervisor import (
    CollisionSupervisor,
    CollisionSupervisorAction,
    CollisionSupervisorDecision,
    CollisionSupervisorState,
)
from runtime.events import MissionEvent
from runtime.hazard_fusion import HazardFusion, HazardFusionResult, HazardReport


class ObstacleRuntimeError(RuntimeError):
    """Raised when routed hazard state cannot be applied atomically."""


class _InterruptibleManager(Protocol):
    is_supervisory_paused: bool

    def interrupt_with_hover(self, reason_code: str, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class ObstacleRuntimeSnapshot:
    state: CollisionSupervisorState
    observation: ObstacleObservation | None
    fusion: HazardFusionResult | None
    hold_requested: bool
    geometry_grounded: bool
    emitted_event_ids: tuple[str, ...]
    hold_requested_timestamp_s: float | None
    hold_established_timestamp_s: float | None
    hold_trigger_sources: tuple[str, ...]


class ObstacleHazardRuntime:
    """Fuse fresh camera evidence and request HOVER without waiting for Qwen."""

    def __init__(
        self,
        *,
        perception: IdealObstaclePerception,
        hazard_fusion: HazardFusion,
        collision_supervisor: CollisionSupervisor,
        skill_manager: _InterruptibleManager,
        event_sink: Callable[[MissionEvent], object] | None = None,
        hover_timeout_s: float = 20.0,
        hover_position_tolerance_m: float = 0.25,
        hover_max_correction_speed_mps: float = 0.5,
        hover_timeout_fallback: str = "CANCEL_AND_LAND",
    ) -> None:
        if not isinstance(perception, IdealObstaclePerception):
            raise TypeError("perception must be IdealObstaclePerception")
        if not isinstance(hazard_fusion, HazardFusion):
            raise TypeError("hazard_fusion must be HazardFusion")
        if not isinstance(collision_supervisor, CollisionSupervisor):
            raise TypeError("collision_supervisor must be CollisionSupervisor")
        if not callable(getattr(skill_manager, "interrupt_with_hover", None)):
            raise TypeError("skill_manager must support interrupt_with_hover")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self._perception = perception
        self._fusion_engine = hazard_fusion
        self._supervisor = collision_supervisor
        self._manager = skill_manager
        self._event_sink = event_sink
        self._hover_timeout_s = _positive(hover_timeout_s, "hover_timeout_s")
        self._hover_position_tolerance_m = _positive(
            hover_position_tolerance_m,
            "hover_position_tolerance_m",
        )
        self._hover_max_correction_speed_mps = _positive(
            hover_max_correction_speed_mps,
            "hover_max_correction_speed_mps",
        )
        if hover_timeout_fallback not in {"RESUME_PREVIOUS", "CANCEL_AND_LAND"}:
            raise ValueError(
                "hover_timeout_fallback must be RESUME_PREVIOUS or CANCEL_AND_LAND"
            )
        self._hover_timeout_fallback = hover_timeout_fallback
        self._latest_observation: ObstacleObservation | None = None
        self._latest_fusion: HazardFusionResult | None = None
        self._event_ids: list[str] = []
        self._seen_hold_transitions: set[tuple[object, ...]] = set()
        self._hold_requested_timestamp_s: float | None = None
        self._hold_established_timestamp_s: float | None = None

    @property
    def state(self) -> CollisionSupervisorState:
        return self._supervisor.state

    @property
    def latest_observation(self) -> ObstacleObservation | None:
        return self._latest_observation

    @property
    def latest_fusion(self) -> HazardFusionResult | None:
        return self._latest_fusion

    @property
    def collision_supervisor(self) -> CollisionSupervisor:
        return self._supervisor

    def process_camera_frame(
        self,
        camera: CameraGeometry,
        *,
        mission_id: str,
        uav_id: str,
        plan_version: int,
        active_corridor: FlightCorridor | None,
        uav_velocity_world_mps: tuple[float, float, float],
        additional_reports: Iterable[HazardReport] = (),
    ) -> ObstacleRuntimeSnapshot:
        """Process one fresh frame and synchronously request HOLD when needed."""

        observation = self._perception.observe(
            camera,
            active_corridor=active_corridor,
            uav_velocity_world_mps=uav_velocity_world_mps,
        )
        if observation is None:
            return self.snapshot()
        self._latest_observation = observation
        speed = sum(value * value for value in uav_velocity_world_mps) ** 0.5
        reports = (
            self._fusion_engine.report_from_observation(
                observation,
                uav_speed_mps=speed,
            ),
            *tuple(additional_reports),
        )
        fusion = self._fusion_engine.fuse(
            reports,
            mission_id=mission_id,
            uav_id=uav_id,
            plan_version=plan_version,
            timestamp_s=camera.timestamp_s,
            uav_speed_mps=speed,
        )
        preserve_active_hazard = bool(
            self._supervisor.state is not CollisionSupervisorState.CLEAR
            and self._latest_fusion is not None
            and self._latest_fusion.should_hold
            and not fusion.should_hold
        )
        if not preserve_active_hazard:
            self._latest_fusion = fusion
        decision = self._supervisor.evaluate(fusion)
        self._publish(decision)
        active_fusion = self._latest_fusion
        if (
            self._supervisor.state is CollisionSupervisorState.HOLDING
            and active_fusion is not None
            and active_fusion.can_generate_route
        ):
            # A Qwen-only suspicion may have established HOLD before trusted
            # camera geometry became available.  Promote that later grounded
            # observation without requiring a second HOVER handshake.
            self._supervisor.mark_geometry_grounded()
        if decision.action is CollisionSupervisorAction.REQUEST_HOLD:
            if self._hold_requested_timestamp_s is None:
                self._hold_requested_timestamp_s = camera.timestamp_s
            # This is the hard real-time trust boundary: no model request is
            # submitted or awaited before the Manager starts HOVER.
            if not bool(getattr(self._manager, "is_supervisory_paused", False)):
                self._manager.interrupt_with_hover(
                    "LOW_LEVEL_PATH_BLOCKED",
                    max_wait_s=self._hover_timeout_s,
                    position_tolerance_m=self._hover_position_tolerance_m,
                    max_correction_speed_mps=self._hover_max_correction_speed_mps,
                    timeout_fallback=self._hover_timeout_fallback,
                    defer_observation_timestamp_s=camera.timestamp_s,
                )
        return self.snapshot()

    def mark_hold_established(self, *, timestamp_s: float) -> ObstacleRuntimeSnapshot:
        """Advance BRAKING only after SkillManager reports a real stable hold."""

        decision = self._supervisor.mark_hold_established(timestamp_s=timestamp_s)
        self._hold_established_timestamp_s = float(timestamp_s)
        self._publish(decision)
        fusion = self._latest_fusion
        if fusion is not None and fusion.can_generate_route:
            self._supervisor.mark_geometry_grounded()
        return self.snapshot()

    def observe_skill_transitions(
        self,
        records: Iterable[object],
    ) -> ObstacleRuntimeSnapshot:
        """Consume Manager transitions and bind the real HOLD handshake once."""

        for record in records:
            if getattr(record, "reason", None) != "HOLD_ESTABLISHED":
                continue
            key = (
                getattr(record, "mission_id", None),
                getattr(record, "plan_version", None),
                getattr(record, "invocation_id", None),
                getattr(record, "timestamp", None),
            )
            if key in self._seen_hold_transitions:
                continue
            self._seen_hold_transitions.add(key)
            if self._supervisor.state is CollisionSupervisorState.BRAKING:
                self.mark_hold_established(timestamp_s=float(record.timestamp))
        return self.snapshot()

    def add_qwen_hazard(
        self,
        report: HazardReport,
        *,
        mission_id: str,
        uav_id: str,
        plan_version: int,
        timestamp_s: float,
        uav_speed_mps: float = 0.0,
    ) -> ObstacleRuntimeSnapshot:
        """Fuse an independent Qwen report; ungrounded reports may HOLD only."""

        prior_reports = () if self._latest_fusion is None else self._latest_fusion.reports
        fusion = self._fusion_engine.fuse(
            (*prior_reports, report),
            mission_id=mission_id,
            uav_id=uav_id,
            plan_version=plan_version,
            timestamp_s=timestamp_s,
            uav_speed_mps=uav_speed_mps,
        )
        self._latest_fusion = fusion
        decision = self._supervisor.evaluate(fusion)
        self._publish(decision)
        if (
            decision.action is CollisionSupervisorAction.REQUEST_HOLD
            and not bool(getattr(self._manager, "is_supervisory_paused", False))
        ):
            if self._hold_requested_timestamp_s is None:
                self._hold_requested_timestamp_s = float(timestamp_s)
            self._manager.interrupt_with_hover(
                "QWEN_PATH_MAY_BE_BLOCKED",
                max_wait_s=self._hover_timeout_s,
                position_tolerance_m=self._hover_position_tolerance_m,
                max_correction_speed_mps=self._hover_max_correction_speed_mps,
                timeout_fallback=self._hover_timeout_fallback,
                defer_observation_timestamp_s=timestamp_s,
            )
        return self.snapshot()

    def snapshot(self) -> ObstacleRuntimeSnapshot:
        fusion = self._latest_fusion
        return ObstacleRuntimeSnapshot(
            state=self._supervisor.state,
            observation=self._latest_observation,
            fusion=fusion,
            hold_requested=self._supervisor.should_hold,
            geometry_grounded=bool(fusion is not None and fusion.geometry_grounded),
            emitted_event_ids=tuple(self._event_ids),
            hold_requested_timestamp_s=self._hold_requested_timestamp_s,
            hold_established_timestamp_s=self._hold_established_timestamp_s,
            hold_trigger_sources=(
                ()
                if fusion is None
                else tuple(
                    dict.fromkeys(
                        report.source.value
                        for report in fusion.reports
                        if report.hazard_detected
                    )
                )
            ),
        )

    def reset(self) -> None:
        self._perception.reset()
        self._supervisor.reset()
        self._latest_observation = None
        self._latest_fusion = None
        self._event_ids.clear()
        self._seen_hold_transitions.clear()
        self._hold_requested_timestamp_s = None
        self._hold_established_timestamp_s = None

    def _publish(self, decision: CollisionSupervisorDecision) -> None:
        for event in decision.events:
            self._event_ids.append(event.event_id)
            if self._event_sink is not None:
                self._event_sink(event)


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite number")
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return normalized


__all__ = [
    "ObstacleHazardRuntime",
    "ObstacleRuntimeError",
    "ObstacleRuntimeSnapshot",
]
