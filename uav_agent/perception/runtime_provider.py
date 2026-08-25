"""Unified Oracle/YOLO target-perception runtime contract.

MissionAgent, SkillManager and TargetManager consume only the resulting
Observation/TargetEstimate.  Simulator truth and detector-specific state stay
behind these per-assignment providers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from math import isclose
from typing import Protocol, runtime_checkable

import numpy as np

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from perception.mode import TargetPerceptionMode
from perception.runtime import (
    GuardedPerceptionBackend,
    PerceptionCapability,
    PerceptionRuntimeProfile,
    validate_observation_access,
)
from perception.runtime_bridge import (
    CoordinatedVisionPerceptionBackend,
    SynchronizedTargetPerceptionInput,
)
from skills.types import Observation
from target.target_manager import TargetManager
from target.types import TargetLifecycle, TargetSpec
from env.camera_types import CameraSample


@runtime_checkable
class TargetPerceptionRuntime(Protocol):
    @property
    def mode(self) -> TargetPerceptionMode: ...

    @property
    def backend_name(self) -> str: ...

    def reset(
        self,
        *,
        mission_id: str,
        assignment_id: str,
        uav_id: str,
        target_alias: str,
        target_spec: TargetSpec,
    ) -> None: ...

    def observe(
        self,
        *,
        base_observation: Observation,
        camera_sample: CameraSample | None,
        target_manager: TargetManager,
    ) -> Observation: ...

    def metrics(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class OracleTargetPerceptionRuntime:
    """Assignment-scoped privileged upper-bound provider.

    ``frame_provider`` is the sole evaluator capability.  It is injected only
    for Oracle mode and always receives the active UAV/target binding.
    """

    def __init__(
        self,
        *,
        uav_id: str,
        oracle_backend: object,
        frame_provider: Callable[[str, str], object],
    ) -> None:
        self._uav_id = validate_uav_id(uav_id)
        if not isinstance(oracle_backend, GuardedPerceptionBackend):
            raise TypeError(
                "oracle_backend must be a GuardedPerceptionBackend"
            )
        if (
            oracle_backend.profile
            is not PerceptionRuntimeProfile.ORACLE_EVALUATION
            or not oracle_backend.acknowledge_privileged_oracle
            or oracle_backend.capability
            is not PerceptionCapability.PRIVILEGED_ORACLE
        ):
            raise PermissionError(
                "Oracle runtime requires the acknowledged privileged guard"
            )
        if not callable(frame_provider):
            raise TypeError("frame_provider must be callable")
        inner = getattr(oracle_backend, "backend", oracle_backend)
        if getattr(inner, "uav_id", self._uav_id) != self._uav_id:
            raise ValueError("Oracle backend is routed to another UAV")
        self._backend = oracle_backend
        self._frame_provider = frame_provider
        self._mission_id: str | None = None
        self._assignment_id: str | None = None
        self._target_alias: str | None = None
        self._total_frames = 0
        self._visible_frames = 0
        self._first_visibility_s: float | None = None
        self._target_lost_count = 0
        self._reacquire_attempts = 0
        self._reacquire_successes = 0
        self._target_manager: TargetManager | None = None
        self._target_event_cursor = 0
        self._closed = False

    @property
    def mode(self) -> TargetPerceptionMode:
        return TargetPerceptionMode.ORACLE

    @property
    def backend_name(self) -> str:
        return "oracle_evaluation"

    @property
    def uav_id(self) -> str:
        return self._uav_id

    @property
    def target_id(self) -> str | None:
        return self._target_alias

    def reset(
        self,
        *,
        mission_id: str,
        assignment_id: str,
        uav_id: str,
        target_alias: str,
        target_spec: TargetSpec,
    ) -> None:
        if self._closed:
            raise RuntimeError("Oracle perception runtime is closed")
        mission = validate_mission_id(mission_id)
        assignment = validate_routing_id(assignment_id, "assignment_id")
        routed_uav = validate_uav_id(uav_id)
        target = validate_routing_id(target_alias, "target_alias")
        if routed_uav != self._uav_id:
            raise ValueError("Oracle reset uav_id does not match runtime")
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        inner = getattr(self._backend, "backend", self._backend)
        if (
            getattr(inner, "uav_id", None) != routed_uav
            or getattr(inner, "target_id", None) != target
        ):
            raise PermissionError(
                "Oracle backend must be rebound to the active Assignment target"
            )
        self._mission_id = mission
        self._assignment_id = assignment
        self._target_alias = target
        self._total_frames = 0
        self._visible_frames = 0
        self._first_visibility_s = None
        self._target_lost_count = 0
        self._reacquire_attempts = 0
        self._reacquire_successes = 0
        self._target_manager = None
        self._target_event_cursor = 0

    def observe(
        self,
        *,
        base_observation: Observation,
        camera_sample: CameraSample | None,
        target_manager: TargetManager,
    ) -> Observation:
        if self._closed:
            raise RuntimeError("Oracle perception runtime is closed")
        if self._mission_id is None or self._target_alias is None:
            raise RuntimeError("Oracle perception runtime must be reset before observe")
        if not isinstance(base_observation, Observation):
            raise TypeError("base_observation must be an Observation")
        validate_observation_access(base_observation)
        if base_observation.uav_id != self._uav_id:
            raise ValueError("Oracle base observation is routed to another UAV")
        if not isinstance(target_manager, TargetManager):
            raise TypeError("target_manager must be a TargetManager")
        self._observe_target_lifecycle_edges(target_manager)
        if camera_sample is not None:
            SynchronizedTargetPerceptionInput(base_observation, camera_sample)
        frame = self._frame_provider(self._uav_id, self._target_alias)
        observation = self._backend.observe(frame)
        if observation.uav_id != self._uav_id:
            raise PermissionError("Oracle output is routed to another UAV")
        if not isclose(
            float(observation.timestamp),
            float(base_observation.timestamp),
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not np.array_equal(
            observation.camera_rgb,
            base_observation.camera_rgb,
        ):
            raise PermissionError(
                "Oracle evaluator output is not synchronized with the agent frame"
            )
        estimate = observation.target_estimate
        if estimate is None:
            # An off-FOV evaluator frame is a valid Oracle query, but it must
            # carry no target side channel into Agent/Skill code.
            if any(
                value is not None
                for value in (
                    observation.oracle_target_id,
                    observation.oracle_target_visible,
                    observation.oracle_target_pose,
                    observation.oracle_target_velocity,
                )
            ):
                raise PermissionError(
                    "invisible Oracle output must not expose oracle_target_*"
                )
            self._total_frames += 1
            return observation
        if estimate.target_id != self._target_alias:
            raise PermissionError("Oracle output target does not match Assignment")
        if estimate.source != "oracle_evaluation":
            raise PermissionError(
                "Oracle output source must be 'oracle_evaluation'"
            )
        if estimate.visible:
            # Oracle is the one explicitly privileged lifecycle shortcut.
            # Advancing it here keeps Agent and Skill code backend-neutral.
            # TargetManager accepts equal timestamps, and after the first call
            # the lifecycle is LOCKED, so observing the same Camera frame more
            # than once cannot append a duplicate transition.
            if target_manager.lifecycle is TargetLifecycle.SEARCHING:
                target_manager.lock_oracle_from_search(
                    estimate.target_id,
                    timestamp_s=float(observation.timestamp),
                    confidence=estimate.confidence,
                    last_seen_position=estimate.position_world_m,
                    last_seen_velocity=estimate.velocity_world_mps,
                )
            elif target_manager.lifecycle is TargetLifecycle.REACQUIRING:
                target_manager.mark_reacquired_oracle(
                    estimate.target_id,
                    timestamp_s=float(observation.timestamp),
                    confidence=estimate.confidence,
                    last_seen_position=estimate.position_world_m,
                    last_seen_velocity=estimate.velocity_world_mps,
                )
            self._observe_target_lifecycle_edges(target_manager)
            self._total_frames += 1
            self._visible_frames += 1
            if self._first_visibility_s is None:
                self._first_visibility_s = float(observation.timestamp)
            return observation

        # Compatibility backends may still emit an explicitly invisible
        # TargetEstimate.  Normalize them to the same no-target public
        # contract as OraclePerception itself, without losing the frame from
        # the provider's visibility metrics.
        self._total_frames += 1
        return replace(
            observation,
            target_estimate=None,
            oracle_target_id=None,
            oracle_target_visible=None,
            oracle_target_pose=None,
            oracle_target_velocity=None,
        )

    def metrics(self) -> Mapping[str, object]:
        if self._target_manager is not None:
            # MissionAgent advances LOST/REACQUIRING after the provider call on
            # a frame. Pulling the append-only event tail here keeps terminal
            # reports exact even when no later Camera frame is observed.
            self._observe_target_lifecycle_edges(self._target_manager)
        return {
            "oracle_visible_frames": self._visible_frames,
            "oracle_total_frames": self._total_frames,
            "oracle_visible_ratio": (
                0.0
                if self._total_frames == 0
                else self._visible_frames / self._total_frames
            ),
            "time_to_first_oracle_visibility_s": self._first_visibility_s,
            "target_lost_count": self._target_lost_count,
            "reacquire_attempts": self._reacquire_attempts,
            "reacquire_successes": self._reacquire_successes,
        }

    def _observe_target_lifecycle_edges(
        self,
        target_manager: TargetManager,
    ) -> None:
        """Consume each TargetManager transition once for runtime metrics."""

        events = target_manager.events()
        if self._target_manager is None:
            self._target_manager = target_manager
            self._target_event_cursor = len(events)
            # The first provider call may legitimately begin while a recovery
            # state is already active. Count the state it actually observes,
            # without replaying unrelated history from before reset().
            if target_manager.lifecycle is TargetLifecycle.LOST:
                self._target_lost_count += 1
            elif target_manager.lifecycle is TargetLifecycle.REACQUIRING:
                self._reacquire_attempts += 1
            return
        if target_manager is not self._target_manager:
            raise RuntimeError(
                "Oracle runtime cannot switch TargetManager within an Assignment"
            )
        if len(events) < self._target_event_cursor:
            raise RuntimeError("TargetManager event log moved backwards")

        for event in events[self._target_event_cursor :]:
            if event.new_state is TargetLifecycle.LOST:
                self._target_lost_count += 1
            if event.new_state is TargetLifecycle.REACQUIRING:
                self._reacquire_attempts += 1
            if (
                event.old_state is TargetLifecycle.REACQUIRING
                and event.new_state is TargetLifecycle.LOCKED
            ):
                self._reacquire_successes += 1
        self._target_event_cursor = len(events)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


class YoloTargetPerceptionRuntime:
    """Production provider backed by one UAV's isolated YOLO coordinator."""

    def __init__(
        self,
        *,
        uav_id: str,
        bridge: CoordinatedVisionPerceptionBackend,
        attribute_evidence_sink: Callable[[object], None] | None = None,
    ) -> None:
        self._uav_id = validate_uav_id(uav_id)
        if not isinstance(bridge, CoordinatedVisionPerceptionBackend):
            raise TypeError("bridge must be a CoordinatedVisionPerceptionBackend")
        if bridge.uav_id != self._uav_id:
            raise ValueError("bridge.uav_id does not match runtime")
        if attribute_evidence_sink is not None and not callable(
            attribute_evidence_sink
        ):
            raise TypeError("attribute_evidence_sink must be callable or None")
        self._bridge = bridge
        self._attribute_evidence_sink = attribute_evidence_sink
        self._attribute_evidence_log_errors = 0
        self._assignment_id: str | None = None
        self._target_alias: str | None = None
        self._closed = False

    @property
    def mode(self) -> TargetPerceptionMode:
        return TargetPerceptionMode.YOLO

    @property
    def backend_name(self) -> str:
        return "ultralytics_service"

    @property
    def uav_id(self) -> str:
        return self._uav_id

    @property
    def target_id(self) -> str | None:
        return self._target_alias

    @property
    def coordinator(self) -> object:
        return self._bridge.coordinator

    def reset(
        self,
        *,
        mission_id: str,
        assignment_id: str,
        uav_id: str,
        target_alias: str,
        target_spec: TargetSpec,
    ) -> None:
        if self._closed:
            raise RuntimeError("YOLO perception runtime is closed")
        # Any rebind attempt retires the previous Assignment immediately.
        # Validation or service-handshake failure cannot revive it.
        self._assignment_id = None
        self._target_alias = None
        mission = validate_mission_id(mission_id)
        assignment = validate_routing_id(assignment_id, "assignment_id")
        routed_uav = validate_uav_id(uav_id)
        target = validate_routing_id(target_alias, "target_alias")
        if routed_uav != self._uav_id:
            raise ValueError("YOLO reset uav_id does not match runtime")
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        self._bridge.reset(
            mission_id=mission,
            target_spec=target_spec,
            assignment_id=assignment,
            target_alias=target,
        )
        self._assignment_id = assignment
        self._target_alias = target

    def observe(
        self,
        *,
        base_observation: Observation,
        camera_sample: CameraSample | None,
        target_manager: TargetManager,
    ) -> Observation:
        if self._closed:
            raise RuntimeError("YOLO perception runtime is closed")
        if self._assignment_id is None:
            raise RuntimeError("YOLO perception runtime must be reset before observe")
        if camera_sample is None:
            raise ValueError("YOLO target perception requires synchronized RGB-D")
        synchronized = SynchronizedTargetPerceptionInput(
            base_observation=base_observation,
            camera_sample=camera_sample,
        )
        observation = self._bridge.observe(
            synchronized,
            target_manager=target_manager,
        )
        # Keep the production capability boundary at the public runtime
        # provider as well as inside the concrete Vision backend.  This makes
        # a swapped/misconfigured bridge fail closed instead of allowing an
        # Oracle-sourced estimate to leak into YOLO mode.
        validate_observation_access(observation)
        if observation.uav_id != self._uav_id:
            raise PermissionError("YOLO output is routed to another UAV")
        self._flush_attribute_evidence()
        estimate = observation.target_estimate
        if estimate is None:
            return observation
        assert self._target_alias is not None
        # Provisional candidates deliberately have no stable target_id. Once
        # the coordinator marks an estimate confirmed, its routed identity
        # must already be the trusted Assignment alias; never relabel it here.
        if estimate.confirmed and estimate.target_id != self._target_alias:
            raise PermissionError(
                "YOLO output target does not match the active Assignment"
            )
        if not estimate.confirmed and estimate.target_id is not None:
            raise PermissionError(
                "unconfirmed YOLO output must not claim a stable target ID"
            )
        return observation

    def metrics(self) -> Mapping[str, object]:
        return {
            **dict(self._bridge.metrics()),
            "attribute_evidence_log_errors": self._attribute_evidence_log_errors,
        }

    def _flush_attribute_evidence(self) -> None:
        sink = self._attribute_evidence_sink
        if sink is None:
            return
        # Consume records instead of tracking a cursor into the provider's
        # bounded deque.  A cursor would stop advancing once a full deque
        # evicted its oldest record, silently dropping every later sample.
        for value in self._bridge.drain_attribute_evidence_records():
            try:
                sink(value)
            except Exception:
                # Experiment persistence is observational and must not alter
                # flight-control authority.
                self._attribute_evidence_log_errors += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bridge.close()


__all__ = [
    "OracleTargetPerceptionRuntime",
    "TargetPerceptionRuntime",
    "YoloTargetPerceptionRuntime",
]
