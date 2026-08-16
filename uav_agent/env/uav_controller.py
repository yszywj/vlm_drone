"""Simulator-independent UAV state and controller contract.

Adapters for ideal kinematics, PX4 offboard, Pegasus, MAVSDK, ROS 2, or a
real vehicle can implement this structural interface without importing a
specific simulator controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class UAVState:
    """Canonical world-frame pose state; yaw is in radians."""

    x: float
    y: float
    z: float
    yaw: float


@runtime_checkable
class UAVController(Protocol):
    """World-frame motion contract consumed by runtime Skills."""

    @property
    def max_speed_mps(self) -> float: ...

    @property
    def max_yaw_rate_rad_s(self) -> float: ...

    def get_pose(self) -> UAVState: ...

    def get_velocity(self) -> np.ndarray: ...

    def set_velocity(
        self,
        velocity_xyz_mps: Sequence[float],
        yaw_rate_rad_s: float = 0.0,
    ) -> None: ...

    def move_toward(
        self,
        goal_xyz_m: Sequence[float],
        speed_mps: float | None = None,
        *,
        face_goal: bool = True,
        tolerance_m: float | None = None,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None: ...

    def rotate_yaw(
        self,
        target_yaw_rad: float,
        *,
        relative: bool = False,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None: ...

    def face_point(
        self,
        point_xyz_m: Sequence[float],
        *,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None: ...

    def stop(self) -> None: ...


__all__ = ["UAVController", "UAVState"]
