"""Production adapter that attaches only standardized target estimates."""

from __future__ import annotations

from dataclasses import replace

from common.provenance import is_privileged_oracle_source
from common.target_estimate import TargetEstimate
from configs.schema import TargetPerceptionConfig
from perception.runtime import (
    PerceptionBoundaryError,
    PerceptionCapability,
    validate_observation_access,
)
from skills.types import Observation
from common.ids import validate_uav_id


class VisionPerceptionBackend:
    """Normalize a base Observation without exposing detector internals.

    Detection, candidate confirmation, depth resolution and filtering happen
    in ``TargetPerceptionCoordinator``.  This backend only joins its immutable
    result to the synchronized vehicle/camera observation.
    """

    capability = PerceptionCapability.VISION

    def __init__(self, config: TargetPerceptionConfig, *, uav_id: str) -> None:
        if not isinstance(config, TargetPerceptionConfig):
            raise TypeError("config must be a TargetPerceptionConfig")
        if config.backend != "ultralytics_service":
            raise ValueError(
                "VisionPerceptionBackend requires backend=ultralytics_service"
            )
        self._config = config
        self._uav_id = validate_uav_id(uav_id)

    @property
    def config(self) -> TargetPerceptionConfig:
        return self._config

    @property
    def uav_id(self) -> str:
        return self._uav_id

    def observe(self, frame: object) -> Observation:
        """Validate a pre-standardized Observation and strip legacy truth."""

        if not isinstance(frame, Observation):
            raise TypeError(
                "VisionPerceptionBackend.observe expects a synchronized Observation"
            )
        if frame.uav_id != self._uav_id:
            raise ValueError("Observation.uav_id does not match vision backend")
        if any(
            value is not None
            for value in (
                frame.oracle_target_id,
                frame.oracle_target_visible,
                frame.oracle_target_pose,
                frame.oracle_target_velocity,
            )
        ):
            raise PerceptionBoundaryError(
                "production vision input must not contain oracle_target_*"
            )
        if (
            frame.target_estimate is not None
            and is_privileged_oracle_source(frame.target_estimate.source)
        ):
            raise PerceptionBoundaryError(
                "production vision input rejects Oracle TargetEstimate"
            )
        validate_observation_access(frame)
        return frame

    def attach_target_estimate(
        self,
        observation: Observation,
        estimate: TargetEstimate | None,
    ) -> Observation:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        if estimate is not None and not isinstance(estimate, TargetEstimate):
            raise TypeError("estimate must be a TargetEstimate or None")
        return self.observe(
            replace(
                observation,
                target_estimate=estimate,
                oracle_target_id=None,
                oracle_target_visible=None,
                oracle_target_pose=None,
                oracle_target_velocity=None,
            )
        )


class DisabledTargetPerceptionBackend:
    """Explicit no-target backend; never fabricates a negative detection."""

    capability = PerceptionCapability.VISION

    def __init__(self, *, uav_id: str) -> None:
        self._uav_id = validate_uav_id(uav_id)

    @property
    def uav_id(self) -> str:
        return self._uav_id

    def observe(self, frame: object) -> Observation:
        if not isinstance(frame, Observation):
            raise TypeError(
                "DisabledTargetPerceptionBackend expects an Observation"
            )
        if frame.uav_id != self._uav_id:
            raise ValueError("Observation.uav_id does not match disabled backend")
        clean = replace(
            frame,
            target_estimate=None,
            oracle_target_id=None,
            oracle_target_visible=None,
            oracle_target_pose=None,
            oracle_target_velocity=None,
        )
        validate_observation_access(clean)
        return clean


__all__ = [
    "DisabledTargetPerceptionBackend",
    "VisionPerceptionBackend",
]
