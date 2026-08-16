"""Pure kinematic UAV controller with bounded translation and yaw rates.

The model intentionally excludes motors, thrust, roll/pitch dynamics, and
aerodynamic forces. ``set_pose`` is reserved for reset/debug; navigation must
use commands followed by repeated ``step`` calls. Yaw is stored in radians.
"""

from __future__ import annotations

from math import atan2, cos, isfinite, pi, sin
from numbers import Real
from typing import Callable, Sequence

import numpy as np

from env.uav_controller import UAVState


PoseWriter = Callable[[np.ndarray, np.ndarray], None]


class KinematicUAV:
    """Integrate velocity commands and synchronize a visual pose sink."""

    def __init__(
        self,
        initial_state: UAVState,
        max_speed_mps: float,
        max_yaw_rate_rad_s: float,
        goal_tolerance_m: float = 0.25,
        pose_writer: PoseWriter | None = None,
    ) -> None:
        self.max_speed_mps = _positive(max_speed_mps, "max_speed_mps")
        self.max_yaw_rate_rad_s = _positive(max_yaw_rate_rad_s, "max_yaw_rate_rad_s")
        self.goal_tolerance_m = _positive(goal_tolerance_m, "goal_tolerance_m")
        self._pose_writer = pose_writer
        self._state = UAVState(0.0, 0.0, 0.0, 0.0)
        self._velocity = np.zeros(3, dtype=np.float64)
        self._yaw_rate_rad_s = 0.0
        self._goal: np.ndarray | None = None
        self._goal_speed_mps = self.max_speed_mps
        self._active_goal_tolerance_m = self.goal_tolerance_m
        self._navigation_active = False
        self._face_goal = True
        self._yaw_target_rad: float | None = None
        self._yaw_face_point_xyz_m: np.ndarray | None = None
        self._yaw_target_rate_limit_rad_s = self.max_yaw_rate_rad_s
        self.set_pose(initial_state.x, initial_state.y, initial_state.z, initial_state.yaw)

    def set_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        """Teleport only for initialization/reset/debug and clear all commands."""

        position = _vector3([x, y, z], "pose position")
        yaw_rad = _finite(yaw, "yaw")
        self._state = UAVState(*position.tolist(), _wrap_angle(yaw_rad))
        self._velocity.fill(0.0)
        self._yaw_rate_rad_s = 0.0
        self._goal = None
        self._active_goal_tolerance_m = self.goal_tolerance_m
        self._navigation_active = False
        self._yaw_target_rad = None
        self._yaw_face_point_xyz_m = None
        self._yaw_target_rate_limit_rad_s = self.max_yaw_rate_rad_s
        self._publish_pose()

    def get_pose(self) -> UAVState:
        return UAVState(self._state.x, self._state.y, self._state.z, self._state.yaw)

    def get_velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def set_velocity(
        self,
        velocity_xyz_mps: Sequence[float],
        yaw_rate_rad_s: float = 0.0,
    ) -> None:
        """Command world-frame velocity, clamped by vector norm and yaw rate."""

        velocity = _vector3(velocity_xyz_mps, "velocity_xyz_mps")
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed_mps:
            velocity *= self.max_speed_mps / speed
        requested_yaw_rate = _finite(yaw_rate_rad_s, "yaw_rate_rad_s")
        self._velocity = velocity
        self._yaw_rate_rad_s = float(
            np.clip(requested_yaw_rate, -self.max_yaw_rate_rad_s, self.max_yaw_rate_rad_s)
        )
        self._goal = None
        self._active_goal_tolerance_m = self.goal_tolerance_m
        self._navigation_active = False
        self._yaw_target_rad = None
        self._yaw_face_point_xyz_m = None
        self._yaw_target_rate_limit_rad_s = self.max_yaw_rate_rad_s

    def move_toward(
        self,
        goal_xyz_m: Sequence[float],
        speed_mps: float | None = None,
        *,
        face_goal: bool = True,
        tolerance_m: float | None = None,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        """Set a navigation goal; actual motion occurs only in subsequent steps."""

        candidate_goal = _vector3(goal_xyz_m, "goal_xyz_m")
        if not isinstance(face_goal, bool):
            raise ValueError("face_goal must be a bool")
        candidate_face_goal = face_goal
        requested_speed = (
            self.max_speed_mps if speed_mps is None else _positive(speed_mps, "speed_mps")
        )
        candidate_speed = min(requested_speed, self.max_speed_mps)
        candidate_tolerance = (
            self.goal_tolerance_m
            if tolerance_m is None
            else _positive(tolerance_m, "tolerance_m")
        )
        requested_yaw_limit = (
            self.max_yaw_rate_rad_s
            if max_yaw_rate_rad_s is None
            else _positive(max_yaw_rate_rad_s, "max_yaw_rate_rad_s")
        )
        candidate_yaw_limit = min(requested_yaw_limit, self.max_yaw_rate_rad_s)
        delta = candidate_goal - self._position_array()
        distance = float(np.linalg.norm(delta))
        candidate_active = (
            distance > candidate_tolerance
        )
        candidate_velocity = (
            np.zeros(3, dtype=np.float64)
            if not candidate_active
            else delta / distance * candidate_speed
        )
        candidate_yaw_target = (
            atan2(delta[1], delta[0])
            if (
                candidate_active
                and candidate_face_goal
                and float(np.linalg.norm(delta[:2])) > 1e-12
            )
            else None
        )

        self._goal = candidate_goal
        self._goal_speed_mps = candidate_speed
        self._active_goal_tolerance_m = candidate_tolerance
        self._navigation_active = candidate_active
        self._face_goal = candidate_face_goal
        self._velocity = candidate_velocity
        self._yaw_rate_rad_s = 0.0
        self._yaw_target_rad = candidate_yaw_target
        self._yaw_face_point_xyz_m = None
        self._yaw_target_rate_limit_rad_s = candidate_yaw_limit

    def rotate_yaw(
        self,
        target_yaw_rad: float,
        *,
        relative: bool = False,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        """Queue a bounded yaw rotation without changing yaw immediately."""

        target = _finite(target_yaw_rad, "target_yaw_rad")
        requested_limit = (
            self.max_yaw_rate_rad_s
            if max_yaw_rate_rad_s is None
            else _positive(max_yaw_rate_rad_s, "max_yaw_rate_rad_s")
        )
        if relative:
            target += self._state.yaw
        self._yaw_target_rad = _wrap_angle(target)
        self._yaw_face_point_xyz_m = None
        self._yaw_target_rate_limit_rad_s = min(requested_limit, self.max_yaw_rate_rad_s)
        self._yaw_rate_rad_s = 0.0
        self._face_goal = False

    def face_point(
        self,
        point_xyz_m: Sequence[float],
        *,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        """Continuously face a fixed world point while translating independently."""

        point = _vector3(point_xyz_m, "point_xyz_m")
        requested_limit = (
            self.max_yaw_rate_rad_s
            if max_yaw_rate_rad_s is None
            else _positive(max_yaw_rate_rad_s, "max_yaw_rate_rad_s")
        )
        delta_xy = point[:2] - self._position_array()[:2]
        target_yaw = (
            self._state.yaw
            if float(np.linalg.norm(delta_xy)) <= 1e-12
            else atan2(delta_xy[1], delta_xy[0])
        )
        self._yaw_face_point_xyz_m = point
        self._yaw_target_rad = _wrap_angle(target_yaw)
        self._yaw_target_rate_limit_rad_s = min(requested_limit, self.max_yaw_rate_rad_s)
        self._yaw_rate_rad_s = 0.0
        self._face_goal = False

    def stop(self) -> None:
        self._velocity.fill(0.0)
        self._yaw_rate_rad_s = 0.0
        self._goal = None
        self._active_goal_tolerance_m = self.goal_tolerance_m
        self._navigation_active = False
        self._yaw_target_rad = None
        self._yaw_face_point_xyz_m = None
        self._yaw_target_rate_limit_rad_s = self.max_yaw_rate_rad_s

    def step(self, dt_s: float) -> UAVState:
        """Advance exactly one simulation step using p(t+1) = p(t) + v*dt."""

        dt = _positive(dt_s, "dt_s")
        position = self._position_array()

        if self._navigation_active and self._goal is not None:
            delta = self._goal - position
            distance = float(np.linalg.norm(delta))
            if distance <= self._active_goal_tolerance_m:
                self._navigation_active = False
                self._velocity.fill(0.0)
            else:
                effective_speed = min(self._goal_speed_mps, self.max_speed_mps, distance / dt)
                self._velocity = delta / distance * effective_speed
                if self._face_goal and float(np.linalg.norm(delta[:2])) > 1e-12:
                    self._yaw_target_rad = atan2(delta[1], delta[0])

        position = position + self._velocity * dt
        if self._navigation_active and self._goal is not None:
            if float(np.linalg.norm(self._goal - position)) <= self._active_goal_tolerance_m:
                self._navigation_active = False
                self._velocity.fill(0.0)

        yaw = self._state.yaw
        if self._yaw_face_point_xyz_m is not None:
            delta_xy = self._yaw_face_point_xyz_m[:2] - position[:2]
            self._yaw_target_rad = (
                yaw
                if float(np.linalg.norm(delta_xy)) <= 1e-12
                else atan2(delta_xy[1], delta_xy[0])
            )
        if self._yaw_target_rad is not None:
            error = _wrap_angle(self._yaw_target_rad - yaw)
            yaw_limit = self._yaw_target_rate_limit_rad_s
            yaw_rate = float(np.clip(error / dt, -yaw_limit, yaw_limit))
            yaw = _wrap_angle(yaw + yaw_rate * dt)
            self._yaw_rate_rad_s = yaw_rate
            if abs(_wrap_angle(self._yaw_target_rad - yaw)) <= 1e-9:
                self._yaw_rate_rad_s = 0.0
        else:
            yaw = _wrap_angle(yaw + self._yaw_rate_rad_s * dt)

        self._state = UAVState(*position.tolist(), yaw)
        self._publish_pose()
        return self.get_pose()

    def distance_to_goal(self, goal_xyz_m: Sequence[float] | None = None) -> float:
        goal = self._resolve_goal(goal_xyz_m)
        return float(np.linalg.norm(goal - self._position_array()))

    def heading_error(self, goal_xyz_m: Sequence[float] | None = None) -> float:
        if goal_xyz_m is None and self._yaw_target_rad is not None:
            return _wrap_angle(self._yaw_target_rad - self._state.yaw)
        goal = self._resolve_goal(goal_xyz_m)
        delta = goal[:2] - self._position_array()[:2]
        if float(np.linalg.norm(delta)) <= 1e-12:
            return 0.0
        return _wrap_angle(atan2(delta[1], delta[0]) - self._state.yaw)

    def goal_reached(
        self,
        goal_xyz_m: Sequence[float] | None = None,
        tolerance_m: float | None = None,
    ) -> bool:
        tolerance = (
            self._active_goal_tolerance_m
            if tolerance_m is None and goal_xyz_m is None and self._goal is not None
            else self.goal_tolerance_m
            if tolerance_m is None
            else _positive(tolerance_m, "tolerance_m")
        )
        return self.distance_to_goal(goal_xyz_m) <= tolerance

    def _resolve_goal(self, goal_xyz_m: Sequence[float] | None) -> np.ndarray:
        if goal_xyz_m is not None:
            return _vector3(goal_xyz_m, "goal_xyz_m")
        if self._goal is None:
            raise RuntimeError("no active or retained goal")
        return self._goal

    def _position_array(self) -> np.ndarray:
        return np.asarray([self._state.x, self._state.y, self._state.z], dtype=np.float64)

    def _publish_pose(self) -> None:
        if self._pose_writer is None:
            return
        half_yaw = self._state.yaw / 2.0
        orientation_wxyz = np.asarray([cos(half_yaw), 0.0, 0.0, sin(half_yaw)])
        self._pose_writer(self._position_array(), orientation_wxyz)


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain three finite values")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if len(items) != 3 or any(
        isinstance(item, bool) or not isinstance(item, Real) or not isfinite(item)
        for item in items
    ):
        raise ValueError(f"{name} must contain three finite values")
    return np.asarray(items, dtype=np.float64)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return result


def _wrap_angle(angle_rad: float) -> float:
    return (angle_rad + pi) % (2.0 * pi) - pi
