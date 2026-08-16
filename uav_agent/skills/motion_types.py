"""Translation-independent yaw policies shared by motion skills."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from enum import Enum, auto
from math import atan2, isfinite
from numbers import Real
from typing import Sequence

import numpy as np

from skills.types import UAVController


class YawMode(Enum):
    COURSE_ALIGNED = auto()
    KEEP_CURRENT = auto()
    FIXED = auto()
    FACE_POINT = auto()


class MotionPolicyValidationError(ValueError):
    """Raised when a MotionPolicy cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class MotionPolicy:
    """Per-Skill motion limits; yaw values and rates use radians."""

    max_speed: float | None = None
    max_yaw_rate: float | None = None
    yaw_mode: YawMode = YawMode.COURSE_ALIGNED
    yaw_value: float | None = None
    look_at_point: tuple[float, float, float] | None = None

    def validate(self) -> None:
        validate_motion_policy(self)


@dataclass(frozen=True, slots=True)
class _ValidatedMotionPolicy:
    yaw_mode: YawMode
    max_speed: float | None
    max_yaw_rate: float | None
    yaw_value: float | None
    look_at_point: np.ndarray | None


def validate_motion_policy(policy: MotionPolicy) -> None:
    """Validate at Skill start so errors map to ``INVALID_GOAL``."""

    _normalize_motion_policy(policy)


def _normalize_motion_policy(policy: MotionPolicy) -> _ValidatedMotionPolicy:
    """Validate once and retain replay-safe values before issuing any command."""

    if not isinstance(policy, MotionPolicy):
        raise MotionPolicyValidationError("motion_policy must be a MotionPolicy")
    if not isinstance(policy.yaw_mode, YawMode):
        raise MotionPolicyValidationError("yaw_mode must be a YawMode")
    max_speed = (
        None if policy.max_speed is None else _positive(policy.max_speed, "max_speed")
    )
    max_yaw_rate = (
        None
        if policy.max_yaw_rate is None
        else _positive(policy.max_yaw_rate, "max_yaw_rate")
    )
    yaw_value: float | None = None
    look_at_point: np.ndarray | None = None
    if policy.yaw_mode is YawMode.FIXED:
        if policy.yaw_value is None:
            raise MotionPolicyValidationError("FIXED yaw mode requires yaw_value")
        yaw_value = _finite(policy.yaw_value, "yaw_value")
    if policy.yaw_mode is YawMode.FACE_POINT:
        if policy.look_at_point is None:
            raise MotionPolicyValidationError("FACE_POINT yaw mode requires look_at_point")
        look_at_point = _vector3(policy.look_at_point, "look_at_point")
    return _ValidatedMotionPolicy(
        yaw_mode=policy.yaw_mode,
        max_speed=max_speed,
        max_yaw_rate=max_yaw_rate,
        yaw_value=yaw_value,
        look_at_point=look_at_point,
    )


def apply_motion_policy(
    uav: UAVController,
    velocity_xyz_mps: Sequence[float],
    policy: MotionPolicy,
    *,
    initial_yaw: float | None = None,
) -> np.ndarray:
    """Command world-frame translation first, then an independent yaw target."""

    validated = _normalize_motion_policy(policy)
    velocity = _vector3(velocity_xyz_mps, "velocity_xyz_mps")
    speed = float(np.linalg.norm(velocity))
    if validated.max_speed is not None and speed > validated.max_speed:
        velocity *= validated.max_speed / speed

    state = uav.get_pose()
    target_yaw: float | None = None
    horizontal_speed = float(np.linalg.norm(velocity[:2]))
    if validated.yaw_mode is YawMode.COURSE_ALIGNED:
        target_yaw = state.yaw if horizontal_speed <= 1e-12 else atan2(velocity[1], velocity[0])
    elif validated.yaw_mode is YawMode.KEEP_CURRENT:
        if initial_yaw is None:
            raise MotionPolicyValidationError("KEEP_CURRENT requires the Skill start yaw")
        target_yaw = _finite(initial_yaw, "initial_yaw")
    elif validated.yaw_mode is YawMode.FIXED:
        target_yaw = validated.yaw_value
    if validated.yaw_mode is not YawMode.FACE_POINT and target_yaw is None:
        raise MotionPolicyValidationError("yaw mode has no target implementation")
    if validated.yaw_mode is YawMode.FACE_POINT and validated.look_at_point is None:
        raise MotionPolicyValidationError("FACE_POINT has no normalized look_at_point")
    # UAVController defines world-frame velocity, so translation never depends on yaw.
    uav.set_velocity(velocity)
    if validated.yaw_mode is YawMode.FACE_POINT:
        uav.face_point(
            validated.look_at_point,
            max_yaw_rate_rad_s=validated.max_yaw_rate,
        )
    else:
        uav.rotate_yaw(
            target_yaw,
            max_yaw_rate_rad_s=validated.max_yaw_rate,
        )
    return uav.get_velocity()


def move_toward_with_policy(
    uav: UAVController,
    goal_xyz_m: Sequence[float],
    speed_mps: float,
    tolerance_m: float,
    policy: MotionPolicy,
    *,
    initial_yaw: float | None = None,
) -> np.ndarray:
    """Queue target-aware translation and an independent yaw policy."""

    validated = _normalize_motion_policy(policy)
    goal = _vector3(goal_xyz_m, "goal_xyz_m")
    requested_speed = _positive(speed_mps, "speed_mps")
    tolerance = _positive(tolerance_m, "tolerance_m")
    effective_speed = (
        requested_speed
        if validated.max_speed is None
        else min(requested_speed, validated.max_speed)
    )

    target_yaw: float | None = None
    if validated.yaw_mode is YawMode.KEEP_CURRENT:
        if initial_yaw is None:
            raise MotionPolicyValidationError("KEEP_CURRENT requires the Skill start yaw")
        target_yaw = _finite(initial_yaw, "initial_yaw")
    elif validated.yaw_mode is YawMode.FIXED:
        target_yaw = validated.yaw_value
    if validated.yaw_mode is YawMode.FIXED and target_yaw is None:
        raise MotionPolicyValidationError("FIXED has no normalized yaw_value")
    if validated.yaw_mode is YawMode.FACE_POINT and validated.look_at_point is None:
        raise MotionPolicyValidationError("FACE_POINT has no normalized look_at_point")

    uav.move_toward(
        goal,
        effective_speed,
        face_goal=validated.yaw_mode is YawMode.COURSE_ALIGNED,
        tolerance_m=tolerance,
        max_yaw_rate_rad_s=validated.max_yaw_rate,
    )
    if validated.yaw_mode in {YawMode.KEEP_CURRENT, YawMode.FIXED}:
        uav.rotate_yaw(
            target_yaw,
            max_yaw_rate_rad_s=validated.max_yaw_rate,
        )
    elif validated.yaw_mode is YawMode.FACE_POINT:
        uav.face_point(
            validated.look_at_point,
            max_yaw_rate_rad_s=validated.max_yaw_rate,
        )
    return uav.get_velocity()


def _vector3(value: Sequence[float] | None, name: str) -> np.ndarray:
    if (
        value is None
        or isinstance(value, (str, bytes))
        or not isinstance(value, (SequenceABC, np.ndarray))
    ):
        raise MotionPolicyValidationError(f"{name} must contain three finite values")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise MotionPolicyValidationError(
            f"{name} must contain three finite values"
        ) from exc
    if len(items) != 3 or any(
        isinstance(item, bool) or not isinstance(item, Real) or not isfinite(item)
        for item in items
    ):
        raise MotionPolicyValidationError(f"{name} must contain three finite values")
    return np.asarray(items, dtype=np.float64)


def _finite(value: float | None, name: str) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise MotionPolicyValidationError(f"{name} must be finite")
    return float(value)


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise MotionPolicyValidationError(f"{name} must be greater than 0")
    return result
