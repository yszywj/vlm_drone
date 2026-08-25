"""Synchronized, backend-neutral bridge for Fleet target perception.

The bridge owns no simulator APIs.  It accepts one atomic agent-facing RGB-D
sample, advances one per-UAV coordinator without blocking the control loop,
and attaches only a :class:`TargetEstimate` to the shared Observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isclose, sin
from typing import Mapping

import numpy as np

from common.ids import validate_routing_id, validate_uav_id
from env.camera_types import CameraSample
from perception.runtime import validate_observation_access
from perception.target_perception_coordinator import TargetPerceptionCoordinator
from perception.target_query import TargetQuerySpec
from perception.vision_backend import VisionPerceptionBackend
from skills.types import Observation
from target.target_manager import TargetManager


@dataclass(frozen=True, slots=True)
class SynchronizedTargetPerceptionInput:
    """One atomic production Camera batch with its neutral Skill observation."""

    base_observation: Observation
    camera_sample: CameraSample
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.base_observation, Observation):
            raise TypeError("base_observation must be an Observation")
        if not isinstance(self.camera_sample, CameraSample):
            raise TypeError("camera_sample must be a CameraSample")
        self.base_observation.validate()
        validate_observation_access(self.base_observation)
        if not isclose(
            float(self.base_observation.timestamp),
            self.camera_sample.timestamp_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Observation and CameraSample timestamps must match")
        if not np.array_equal(
            self.base_observation.camera_rgb,
            self.camera_sample.rgb,
        ):
            raise ValueError("Observation RGB must come from the same CameraSample")
        if self.base_observation.camera_position_m is None:
            raise ValueError("production target perception requires Camera pose")
        if not np.allclose(
            self.base_observation.camera_position_m,
            self.camera_sample.camera_position_world_m,
            rtol=0.0,
            atol=1e-9,
        ) or not np.allclose(
            self.base_observation.camera_orientation_wxyz,
            self.camera_sample.camera_orientation_world_wxyz,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("Observation pose must come from the same CameraSample")


def synchronized_uav_self_motion(
    observation: Observation,
    *,
    previous_yaw_rad: float | None,
    previous_timestamp_s: float | None,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return synchronized world-linear/body-angular UAV motion.

    The canonical UAV observation currently exposes yaw but not full body
    angular velocity.  Therefore the body x/y angular components are exactly
    zero and body-z is the wrap-safe finite difference of synchronized yaw.
    This is UAV state, not motion inferred from the camera extrinsics.
    """

    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation")
    if (previous_yaw_rad is None) != (previous_timestamp_s is None):
        raise ValueError("previous UAV yaw and timestamp must be paired")
    linear_velocity = tuple(float(value) for value in observation.uav_velocity)
    yaw = float(observation.uav_pose.yaw)
    timestamp = float(observation.timestamp)
    if previous_yaw_rad is None:
        yaw_rate = 0.0
    else:
        delta_t = timestamp - float(previous_timestamp_s)
        if delta_t <= 1e-9:
            raise ValueError(
                "new camera samples require increasing UAV state timestamps"
            )
        wrapped_delta = atan2(
            sin(yaw - float(previous_yaw_rad)),
            cos(yaw - float(previous_yaw_rad)),
        )
        yaw_rate = wrapped_delta / delta_t
    return linear_velocity, (0.0, 0.0, yaw_rate)


class CoordinatedVisionPerceptionBackend:
    """Per-UAV adapter around an asynchronous YOLO coordinator.

    Polling happens before submission so a completed response is consumed at
    the current control tick.  A camera timestamp is submitted at most once;
    repeated control ticks therefore do not fabricate duplicate detections.
    """

    def __init__(
        self,
        *,
        uav_id: str,
        coordinator: TargetPerceptionCoordinator,
        vision_backend: VisionPerceptionBackend,
    ) -> None:
        self._uav_id = validate_uav_id(uav_id)
        if not isinstance(coordinator, TargetPerceptionCoordinator):
            raise TypeError("coordinator must be a TargetPerceptionCoordinator")
        if not isinstance(vision_backend, VisionPerceptionBackend):
            raise TypeError("vision_backend must be a VisionPerceptionBackend")
        if vision_backend.uav_id != self._uav_id:
            raise ValueError("vision_backend.uav_id does not match bridge")
        self._coordinator = coordinator
        self._vision_backend = vision_backend
        self._target_query: TargetQuerySpec | None = None
        self._target_alias: str | None = None
        self._last_submitted_timestamp_s: float | None = None
        self._last_uav_yaw_rad: float | None = None
        self._last_uav_motion_timestamp_s: float | None = None
        self._closed = False

    @property
    def uav_id(self) -> str:
        return self._uav_id

    @property
    def coordinator(self) -> TargetPerceptionCoordinator:
        return self._coordinator

    @property
    def target_alias(self) -> str | None:
        return self._target_alias

    def attribute_evidence_records(self) -> tuple[object, ...]:
        records = getattr(self._coordinator, "attribute_evidence_records", None)
        return () if not callable(records) else tuple(records())

    def drain_attribute_evidence_records(self) -> tuple[object, ...]:
        drain = getattr(
            self._coordinator,
            "drain_attribute_evidence_records",
            None,
        )
        return () if not callable(drain) else tuple(drain())

    def candidate_transition_records(self) -> tuple[object, ...]:
        records = getattr(self._coordinator, "candidate_transition_records", None)
        return () if not callable(records) else tuple(records())

    def drain_candidate_transition_records(self) -> tuple[object, ...]:
        drain = getattr(
            self._coordinator,
            "drain_candidate_transition_records",
            None,
        )
        return () if not callable(drain) else tuple(drain())

    def reset(
        self,
        *,
        mission_id: str,
        target_query: TargetQuerySpec,
        assignment_id: str | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("perception bridge is closed")
        # Retire the previous binding before validating or handshaking the
        # replacement. No failed reset may leave an old Assignment usable.
        self._target_query = None
        self._target_alias = None
        self._last_submitted_timestamp_s = None
        self._last_uav_yaw_rad = None
        self._last_uav_motion_timestamp_s = None
        if not isinstance(target_query, TargetQuerySpec):
            raise TypeError("target_query must be a TargetQuerySpec")
        routed_target = validate_routing_id(
            target_query.target_alias,
            "target_alias",
        )
        self._coordinator.reset(
            mission_id=mission_id,
            uav_id=self._uav_id,
            assignment_id=assignment_id,
            target_alias=routed_target,
            target_query=target_query,
        )
        self._target_query = target_query
        self._target_alias = routed_target

    def observe(
        self,
        synchronized_input: SynchronizedTargetPerceptionInput,
        *,
        target_manager: TargetManager,
    ) -> Observation:
        if self._closed:
            raise RuntimeError("perception bridge is closed")
        if not isinstance(synchronized_input, SynchronizedTargetPerceptionInput):
            raise TypeError(
                "synchronized_input must be SynchronizedTargetPerceptionInput"
            )
        if not isinstance(target_manager, TargetManager):
            raise TypeError("target_manager must be a TargetManager")
        if synchronized_input.base_observation.uav_id != self._uav_id:
            raise ValueError("perception input is routed to another UAV")
        if self._target_query is None:
            raise RuntimeError("perception bridge must be reset before observe")

        now_s = float(synchronized_input.base_observation.timestamp)
        if (
            self._last_submitted_timestamp_s is not None
            and synchronized_input.camera_sample.timestamp_s
            < self._last_submitted_timestamp_s - 1e-9
        ):
            # Reject a stale batch before polling can advance any candidate or
            # TargetManager state.  This preserves the all-or-nothing atomic
            # camera-batch boundary on timestamp regressions.
            raise ValueError("Camera timestamps must be monotonically increasing")
        estimate = self._coordinator.poll(
            now_s=now_s,
            target_manager=target_manager,
        )
        if (
            self._last_submitted_timestamp_s is None
            or synchronized_input.camera_sample.timestamp_s
            > self._last_submitted_timestamp_s + 1e-9
        ):
            observation = synchronized_input.base_observation
            yaw = float(observation.uav_pose.yaw)
            previous_yaw = self._last_uav_yaw_rad
            previous_timestamp = self._last_uav_motion_timestamp_s
            linear_velocity, angular_velocity = synchronized_uav_self_motion(
                observation,
                previous_yaw_rad=previous_yaw,
                previous_timestamp_s=previous_timestamp,
            )
            self._coordinator.submit_frame(
                camera_sample=synchronized_input.camera_sample,
                uav_linear_velocity_world_mps=linear_velocity,
                uav_angular_velocity_body_radps=angular_velocity,
            )
            self._last_submitted_timestamp_s = (
                synchronized_input.camera_sample.timestamp_s
            )
            self._last_uav_yaw_rad = yaw
            self._last_uav_motion_timestamp_s = now_s
        return self._vision_backend.attach_target_estimate(
            synchronized_input.base_observation,
            estimate,
        )

    def metrics(self) -> Mapping[str, object]:
        runtime_metrics = getattr(self._coordinator, "runtime_metrics", None)
        if callable(runtime_metrics):
            return dict(runtime_metrics())
        metrics = getattr(self._coordinator, "metrics", None)
        to_dict = getattr(metrics, "to_dict", None)
        if not callable(to_dict):
            return {}
        return dict(to_dict())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._target_query = None
        self._target_alias = None
        self._last_submitted_timestamp_s = None
        self._last_uav_yaw_rad = None
        self._last_uav_motion_timestamp_s = None
        self._coordinator.close()


__all__ = [
    "CoordinatedVisionPerceptionBackend",
    "SynchronizedTargetPerceptionInput",
]
