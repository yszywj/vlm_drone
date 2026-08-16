"""Pure kinematic target motion with deterministic bounded trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, isfinite, pi, sin
from numbers import Real
from typing import Callable, Sequence

import numpy as np


PoseWriter = Callable[[np.ndarray, np.ndarray], None]


class TargetMotionMode(str, Enum):
    STATIC = "STATIC"
    LINEAR = "LINEAR"
    RANDOM_WALK = "RANDOM_WALK"


@dataclass(frozen=True)
class TargetState:
    x: float
    y: float
    z: float
    yaw: float


class MovingTarget:
    """Move a target in XY and reflect it inside a closed world-space AABB."""

    def __init__(
        self,
        mode: TargetMotionMode | str,
        initial_position_xyz_m: Sequence[float],
        bounds_min_xyz_m: Sequence[float],
        bounds_max_xyz_m: Sequence[float],
        speed_mps: float,
        max_speed_mps: float,
        direction_change_interval_s: float,
        seed: int = 0,
        initial_heading_rad: float = 0.0,
        pose_writer: PoseWriter | None = None,
    ) -> None:
        self.mode = _mode(mode)
        self._initial_position = _vector3(initial_position_xyz_m, "initial_position_xyz_m")
        self._bounds_min = _vector3(bounds_min_xyz_m, "bounds_min_xyz_m")
        self._bounds_max = _vector3(bounds_max_xyz_m, "bounds_max_xyz_m")
        if np.any(self._bounds_min > self._bounds_max):
            raise ValueError("bounds_min_xyz_m must not exceed bounds_max_xyz_m")
        if self._bounds_min[0] == self._bounds_max[0] or self._bounds_min[1] == self._bounds_max[1]:
            raise ValueError("target bounds must have positive x and y width")
        if np.any(self._initial_position < self._bounds_min) or np.any(self._initial_position > self._bounds_max):
            raise ValueError("initial_position_xyz_m must be inside target bounds")

        self.max_speed_mps = _positive(max_speed_mps, "max_speed_mps")
        self.speed_mps = _nonnegative(speed_mps, "speed_mps")
        if self.speed_mps > self.max_speed_mps:
            raise ValueError("speed_mps must not exceed max_speed_mps")
        self.direction_change_interval_s = _positive(
            direction_change_interval_s, "direction_change_interval_s"
        )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self._base_seed = seed
        self._initial_heading_rad = _finite(initial_heading_rad, "initial_heading_rad")
        self._pose_writer = pose_writer
        self._position = self._initial_position.copy()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._yaw = self._initial_heading_rad
        self._rng = np.random.default_rng(seed)
        self._time_s = 0.0
        self._turn_index = 0
        self.reset()

    def get_pose(self) -> TargetState:
        return TargetState(*self._position.tolist(), self._yaw)

    def get_velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def reset(
        self,
        *,
        position_m: Sequence[float] | None = None,
        yaw_rad: float | None = None,
        seed: int | None = None,
        mode: TargetMotionMode | str | None = None,
    ) -> TargetState:
        """Reset deterministically and immediately publish the initial target pose."""

        candidate_mode = self.mode if mode is None else _mode(mode)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer")
        episode_seed = self._base_seed if seed is None else seed
        candidate_rng = np.random.default_rng(episode_seed)
        candidate_position = (
            self._initial_position if position_m is None else _vector3(position_m, "position_m")
        )
        if np.any(candidate_position < self._bounds_min) or np.any(candidate_position > self._bounds_max):
            raise ValueError("position_m must be inside target bounds")
        reset_heading = (
            self._initial_heading_rad if yaw_rad is None else _finite(yaw_rad, "yaw_rad")
        )
        candidate_yaw = _wrap_angle(reset_heading)
        if candidate_mode is TargetMotionMode.STATIC or self.speed_mps == 0.0:
            candidate_velocity = np.zeros(3, dtype=np.float64)
        else:
            # RANDOM_WALK starts from the configured heading, then samples a
            # new heading at each direction-change boundary.
            candidate_velocity = np.asarray(
                [
                    self.speed_mps * cos(candidate_yaw),
                    self.speed_mps * sin(candidate_yaw),
                    0.0,
                ],
                dtype=np.float64,
            )

        self.mode = candidate_mode
        self._rng = candidate_rng
        self._position = candidate_position.copy()
        self._time_s = 0.0
        self._turn_index = 0
        self._yaw = candidate_yaw
        self._velocity = candidate_velocity
        self._publish_pose()
        return self.get_pose()

    def step(self, dt_s: float) -> TargetState:
        dt = _nonnegative(dt_s, "dt_s")
        if dt == 0.0 or self.mode is TargetMotionMode.STATIC or self.speed_mps == 0.0:
            return self.get_pose()

        if self.mode is TargetMotionMode.LINEAR:
            self._advance(dt)
            self._time_s += dt
        else:
            remaining = dt
            epsilon = 1e-12
            while remaining > epsilon:
                next_turn_time = (self._turn_index + 1) * self.direction_change_interval_s
                time_to_turn = max(0.0, next_turn_time - self._time_s)
                segment = min(remaining, time_to_turn)
                if segment > epsilon:
                    self._advance(segment)
                    self._time_s += segment
                    remaining -= segment
                if time_to_turn <= segment + epsilon:
                    self._turn_index += 1
                    self._sample_random_heading()

        self._update_yaw_from_velocity()
        self._publish_pose()
        return self.get_pose()

    def _advance(self, dt_s: float) -> None:
        for axis in (0, 1):
            low = self._bounds_min[axis]
            high = self._bounds_max[axis]
            length = high - low
            unfolded = (self._position[axis] - low) + self._velocity[axis] * dt_s
            phase = unfolded % (2.0 * length)
            if phase <= length:
                new_position = low + phase
                new_velocity = self._velocity[axis]
            else:
                new_position = low + (2.0 * length - phase)
                new_velocity = -self._velocity[axis]
            wall_tolerance = 1e-12 * max(1.0, length)
            if np.isclose(new_position, low, rtol=0.0, atol=wall_tolerance):
                new_position = low
                if new_velocity < 0.0:
                    new_velocity = -new_velocity
            elif np.isclose(new_position, high, rtol=0.0, atol=wall_tolerance):
                new_position = high
                if new_velocity > 0.0:
                    new_velocity = -new_velocity
            self._position[axis] = np.clip(new_position, low, high)
            self._velocity[axis] = new_velocity
        self._position[2] = np.clip(self._position[2], self._bounds_min[2], self._bounds_max[2])
        self._velocity[2] = 0.0

    def _sample_random_heading(self) -> None:
        self._set_heading(float(self._rng.uniform(-pi, pi)))

    def _set_heading(self, heading_rad: float) -> None:
        self._yaw = _wrap_angle(heading_rad)
        self._velocity = np.asarray(
            [self.speed_mps * cos(self._yaw), self.speed_mps * sin(self._yaw), 0.0],
            dtype=np.float64,
        )

    def _update_yaw_from_velocity(self) -> None:
        if float(np.linalg.norm(self._velocity[:2])) > 1e-12:
            self._yaw = _wrap_angle(atan2(self._velocity[1], self._velocity[0]))

    def _publish_pose(self) -> None:
        if self._pose_writer is None:
            return
        half_yaw = self._yaw / 2.0
        orientation_wxyz = np.asarray([cos(half_yaw), 0.0, 0.0, sin(half_yaw)])
        self._pose_writer(self._position.copy(), orientation_wxyz)


def _mode(value: TargetMotionMode | str) -> TargetMotionMode:
    if isinstance(value, TargetMotionMode):
        return value
    if isinstance(value, str):
        try:
            return TargetMotionMode(value.upper())
        except ValueError as exc:
            raise ValueError("mode must be STATIC, LINEAR, or RANDOM_WALK") from exc
    raise ValueError("mode must be STATIC, LINEAR, or RANDOM_WALK")


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result.copy()


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be greater than or equal to 0")
    return result


def _wrap_angle(angle_rad: float) -> float:
    return (angle_rad + pi) % (2.0 * pi) - pi
