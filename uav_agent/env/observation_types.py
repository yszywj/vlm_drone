"""Simulator-independent observation contracts shared by singleton and fleet envs.

Keeping these value/capability types outside the Isaac-backed environment
modules prevents the fleet coordinator from importing ``SimulationApp``
dependencies merely to publish a synchronized observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from env.camera_types import CameraSample
from env.moving_target import TargetState
from env.uav_controller import UAVState

if TYPE_CHECKING:
    from env.camera_sensor import ImageProjection


@dataclass(frozen=True)
class AgentObservation:
    """Default agent-facing observation; deliberately excludes Target truth."""

    rgb: np.ndarray
    uav_state: UAVState
    uav_velocity_mps: np.ndarray
    camera_position_m: np.ndarray
    camera_orientation_wxyz: np.ndarray
    camera_timestamp_s: float
    camera_sample: CameraSample | None = None


@dataclass(frozen=True)
class EvaluatorFrame:
    """A synchronized Camera frame plus privileged Target truth."""

    observation: AgentObservation
    target_position_m: np.ndarray
    target_orientation_wxyz: np.ndarray
    target_state: TargetState
    target_velocity_mps: np.ndarray
    target_projection: ImageProjection


class AgentView:
    """Narrow Planner capability that cannot read evaluator Target APIs."""

    __slots__ = (
        "__observe",
        "__move_toward",
        "__set_velocity",
        "__rotate_yaw",
        "__stop",
        "__distance_to_goal",
        "__heading_error",
        "__goal_reached",
    )

    def __init__(
        self,
        *,
        observe: Callable[[], AgentObservation],
        move_toward: Callable[..., None],
        set_velocity: Callable[..., None],
        rotate_yaw: Callable[..., None],
        stop: Callable[[], None],
        distance_to_goal: Callable[..., float],
        heading_error: Callable[..., float],
        goal_reached: Callable[..., bool],
    ) -> None:
        self.__observe = observe
        self.__move_toward = move_toward
        self.__set_velocity = set_velocity
        self.__rotate_yaw = rotate_yaw
        self.__stop = stop
        self.__distance_to_goal = distance_to_goal
        self.__heading_error = heading_error
        self.__goal_reached = goal_reached

    def observe(self) -> AgentObservation:
        return self.__observe()

    def move_toward(
        self,
        goal_xyz_m: Sequence[float],
        speed_mps: float | None = None,
        *,
        face_goal: bool = True,
        tolerance_m: float | None = None,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        self.__move_toward(
            goal_xyz_m,
            speed_mps,
            face_goal=face_goal,
            tolerance_m=tolerance_m,
            max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        )

    def set_velocity(
        self,
        velocity_xyz_mps: Sequence[float],
        yaw_rate_rad_s: float = 0.0,
    ) -> None:
        self.__set_velocity(velocity_xyz_mps, yaw_rate_rad_s)

    def rotate_yaw(
        self,
        target_yaw_rad: float,
        *,
        relative: bool = False,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        self.__rotate_yaw(
            target_yaw_rad,
            relative=relative,
            max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        )

    def stop(self) -> None:
        self.__stop()

    def distance_to_goal(self, goal_xyz_m: Sequence[float] | None = None) -> float:
        return self.__distance_to_goal(goal_xyz_m)

    def heading_error(self, goal_xyz_m: Sequence[float] | None = None) -> float:
        return self.__heading_error(goal_xyz_m)

    def goal_reached(
        self,
        goal_xyz_m: Sequence[float] | None = None,
        tolerance_m: float | None = None,
    ) -> bool:
        return self.__goal_reached(goal_xyz_m, tolerance_m)


__all__ = ["AgentObservation", "AgentView", "EvaluatorFrame"]
