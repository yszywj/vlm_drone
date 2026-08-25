"""Synchronized, backend-neutral bridge for Fleet target perception.

The bridge owns no simulator APIs.  It accepts one atomic agent-facing RGB-D
sample, advances one per-UAV coordinator without blocking the control loop,
and attaches only a :class:`TargetEstimate` to the shared Observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Mapping

import numpy as np

from common.ids import validate_routing_id, validate_uav_id
from env.camera_types import CameraSample
from perception.runtime import validate_observation_access
from perception.target_perception_coordinator import TargetPerceptionCoordinator
from perception.vision_backend import VisionPerceptionBackend
from skills.types import Observation
from target.target_manager import TargetManager
from target.types import TargetSpec


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
        self._target_spec: TargetSpec | None = None
        self._target_alias: str | None = None
        self._last_submitted_timestamp_s: float | None = None
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

    def reset(
        self,
        *,
        mission_id: str,
        target_spec: TargetSpec,
        assignment_id: str | None = None,
        target_alias: str,
    ) -> None:
        if self._closed:
            raise RuntimeError("perception bridge is closed")
        # Retire the previous binding before validating or handshaking the
        # replacement. No failed reset may leave an old Assignment usable.
        self._target_spec = None
        self._target_alias = None
        self._last_submitted_timestamp_s = None
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        routed_target = validate_routing_id(target_alias, "target_alias")
        self._coordinator.reset(
            mission_id=mission_id,
            uav_id=self._uav_id,
            assignment_id=assignment_id,
            target_alias=routed_target,
        )
        self._target_spec = target_spec
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
        if self._target_spec is None:
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
            self._coordinator.submit_frame(
                camera_sample=synchronized_input.camera_sample,
                target_spec=self._target_spec,
            )
            self._last_submitted_timestamp_s = (
                synchronized_input.camera_sample.timestamp_s
            )
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
        self._target_spec = None
        self._target_alias = None
        self._last_submitted_timestamp_s = None
        self._coordinator.close()


__all__ = [
    "CoordinatedVisionPerceptionBackend",
    "SynchronizedTargetPerceptionInput",
]
