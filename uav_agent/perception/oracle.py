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

from env.moving_target import TargetState
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

    target_id: str = "target"

    def __post_init__(self) -> None:
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

        try:
            observation = Observation(
                timestamp=float(agent.camera_timestamp_s),
                uav_pose=agent.uav_state,
                uav_velocity=np.asarray(agent.uav_velocity_mps).copy(),
                camera_rgb=np.asarray(agent.rgb).copy(),
                camera_position_m=np.asarray(agent.camera_position_m).copy(),
                camera_orientation_wxyz=np.asarray(
                    agent.camera_orientation_wxyz
                ).copy(),
                oracle_target_id=self.target_id,
                oracle_target_visible=bool(projection_visible[0]),
                oracle_target_pose=target_state,
                oracle_target_velocity=target_velocity.copy(),
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


__all__ = ["OraclePerception", "OraclePerceptionError"]
