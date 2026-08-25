"""Privileged evaluator-to-Observation adapter for Stage 0.

This module is intentionally Isaac-independent.  A standalone runtime obtains
an ``EvaluatorFrame`` from the environment only after a fresh Camera sample,
then passes that synchronized snapshot to :class:`OraclePerception`.  The
adapter does not query the scene, move the Target, or perform its own FOV
calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from common.target_estimate import TargetEstimate
from env.moving_target import TargetState
from perception.runtime import PerceptionCapability
from common.ids import validate_uav_id
from skills.types import Observation

if TYPE_CHECKING:
    # This import is erased at runtime.  Importing the Isaac-backed environment
    # before SimulationApp exists would violate standalone startup ordering.
    from env.simple_uav_search_env import EvaluatorFrame


class OraclePerceptionError(ValueError):
    """Raised when an evaluator snapshot is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class OraclePerception:
    """Convert one synchronized privileged frame into a Skill Observation.

    ``target_id`` is the simulator's stable logical identifier.  Visibility is
    copied from the Camera projection already stored in the evaluator frame;
    it means geometric in-frustum visibility and currently ignores occlusion.
    """

    # Routing is part of the observation producer contract.  Keeping this
    # required prevents a non-default UAV from silently receiving an
    # Observation labelled with the historical ``uav_1`` compatibility ID.
    uav_id: str
    target_id: str = "target"

    # Runtime policy code uses this declaration to prevent this adapter from
    # being selected by the default production profile.
    capability = PerceptionCapability.PRIVILEGED_ORACLE

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        except (TypeError, ValueError) as exc:
            raise OraclePerceptionError(str(exc)) from None
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise OraclePerceptionError("target_id must be a non-empty string")
        object.__setattr__(self, "target_id", self.target_id.strip())

    def observe(self, frame: object) -> Observation:
        """Return a defensive copy of the synchronized Oracle observation."""

        # Keep the public boundary identical to ``PerceptionBackend``.  The
        # cast is static-only; the guarded field access below remains the
        # runtime validation for an evaluator frame.
        evaluator_frame = cast("EvaluatorFrame", frame)

        try:
            agent = evaluator_frame.observation
            projection_visible = np.asarray(
                evaluator_frame.target_projection.visible
            )
            target_state = evaluator_frame.target_state
            target_velocity = np.asarray(evaluator_frame.target_velocity_mps)
        except (AttributeError, TypeError) as exc:
            raise OraclePerceptionError(
                "frame must be a synchronized environment EvaluatorFrame"
            ) from exc

        if projection_visible.shape != (1,):
            raise OraclePerceptionError(
                "EvaluatorFrame target projection must contain exactly one visibility value"
            )
        if not np.issubdtype(projection_visible.dtype, np.bool_):
            raise OraclePerceptionError(
                "EvaluatorFrame target visibility must be boolean"
            )
        if not isinstance(target_state, TargetState):
            raise OraclePerceptionError(
                "EvaluatorFrame target_state must be a TargetState"
            )
        if target_velocity.shape != (3,) or not np.all(np.isfinite(target_velocity)):
            raise OraclePerceptionError(
                "EvaluatorFrame target_velocity_mps must contain three finite values"
            )

        visible = bool(projection_visible[0])
        bbox = None
        if visible:
            projected = getattr(evaluator_frame.target_projection, "pixels_uv", None)
            if projected is None:
                # Legacy evaluator fixtures exposed only the already-computed
                # visibility bit.  A full-frame compatibility box preserves
                # their upper-bound semantics without affecting production.
                bbox = (0.0, 0.0, 1.0, 1.0)
            else:
                try:
                    pixels_uv = np.asarray(projected)
                    bbox = _point_bbox_normalized(
                        pixels_uv,
                        np.asarray(agent.rgb).shape,
                    )
                except (TypeError, ValueError) as exc:
                    raise OraclePerceptionError(
                        "visible Oracle target requires one finite projected pixel"
                    ) from exc

        try:
            timestamp_s = float(agent.camera_timestamp_s)
            # Camera visibility is the privilege-release boundary.  The
            # evaluator frame may retain off-FOV truth for scoring, but none of
            # that target state is allowed to cross into the Agent
            # Observation until the target is actually visible in this
            # Camera sample.
            estimate = (
                TargetEstimate(
                    timestamp_s=timestamp_s,
                    target_id=self.target_id,
                    candidate_id=None,
                    tracker_id=None,
                    visible=True,
                    confirmed=True,
                    predicted_only=False,
                    class_id=None,
                    class_name=None,
                    confidence=1.0,
                    bbox_xyxy_normalized=bbox,
                    position_world_m=(
                        float(target_state.x),
                        float(target_state.y),
                        float(target_state.z),
                    ),
                    velocity_world_mps=tuple(
                        float(value) for value in target_velocity
                    ),
                    measurement_age_s=0.0,
                    source="oracle_evaluation",
                )
                if visible
                else None
            )
            observation = Observation(
                uav_id=self.uav_id,
                timestamp=timestamp_s,
                uav_pose=agent.uav_state,
                uav_velocity=np.asarray(agent.uav_velocity_mps).copy(),
                camera_rgb=np.asarray(agent.rgb).copy(),
                camera_position_m=np.asarray(agent.camera_position_m).copy(),
                camera_orientation_wxyz=np.asarray(
                    agent.camera_orientation_wxyz
                ).copy(),
                target_estimate=estimate,
                oracle_target_id=self.target_id if visible else None,
                oracle_target_visible=True if visible else None,
                oracle_target_pose=target_state if visible else None,
                oracle_target_velocity=(
                    target_velocity.copy() if visible else None
                ),
            )
            observation.validate()
        except (AttributeError, TypeError, ValueError) as exc:
            raise OraclePerceptionError(
                f"invalid synchronized evaluator frame: {exc}"
            ) from exc
        return observation

    def get_observation(self, frame: object) -> Observation:
        """Compatibility alias for runtimes that use ``get_*`` naming."""

        return self.observe(frame)


def _point_bbox_normalized(
    pixels_uv: np.ndarray,
    rgb_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    """Represent the evaluator's projected target point as a tiny valid box."""

    if pixels_uv.shape != (1, 2) or not np.all(np.isfinite(pixels_uv)):
        raise ValueError("pixels_uv must contain one finite (u, v) point")
    if len(rgb_shape) != 3 or rgb_shape[0] <= 0 or rgb_shape[1] <= 0:
        raise ValueError("rgb shape must contain positive height and width")
    height, width = int(rgb_shape[0]), int(rgb_shape[1])
    u = min(max(float(pixels_uv[0, 0]), 0.0), max(0.0, width - 1.0))
    v = min(max(float(pixels_uv[0, 1]), 0.0), max(0.0, height - 1.0))
    half_width = max(0.5, min(2.0, width / 2.0))
    half_height = max(0.5, min(2.0, height / 2.0))
    x1 = max(0.0, (u - half_width) / width)
    x2 = min(1.0, (u + half_width) / width)
    y1 = max(0.0, (v - half_height) / height)
    y2 = min(1.0, (v + half_height) / height)
    if x1 >= x2:
        x1, x2 = (0.0, min(1.0, 1.0 / width)) if u <= 0.0 else (
            max(0.0, 1.0 - 1.0 / width),
            1.0,
        )
    if y1 >= y2:
        y1, y2 = (0.0, min(1.0, 1.0 / height)) if v <= 0.0 else (
            max(0.0, 1.0 - 1.0 / height),
            1.0,
        )
    return x1, y1, x2, y2


__all__ = ["OraclePerception", "OraclePerceptionError"]
