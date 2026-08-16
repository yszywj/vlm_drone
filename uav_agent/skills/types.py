"""Shared, Isaac-independent types for all callable UAV skills."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from math import isfinite
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np

from env.moving_target import TargetState
from env.uav_controller import UAVController, UAVState


class SkillName(str, Enum):
    TAKEOFF = "TAKEOFF"
    GOTO = "GOTO"
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    REACQUIRE = "REACQUIRE"
    LAND = "LAND"


class SkillStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELED = auto()


class SkillResultCode(Enum):
    NONE = auto()

    TAKEOFF_COMPLETE = auto()
    LAND_COMPLETE = auto()

    GOAL_REACHED = auto()
    TRACK_COMPLETE = auto()

    TARGET_FOUND = auto()
    TARGET_LOST = auto()

    SEARCH_EXHAUSTED = auto()

    TIMEOUT = auto()
    INVALID_GOAL = auto()
    INVALID_STATE = auto()

    CANCELED = auto()
    INTERNAL_ERROR = auto()


_SUCCESS_RESULT_CODES = frozenset(
    {
        SkillResultCode.TAKEOFF_COMPLETE,
        SkillResultCode.LAND_COMPLETE,
        SkillResultCode.GOAL_REACHED,
        SkillResultCode.TRACK_COMPLETE,
        SkillResultCode.TARGET_FOUND,
    }
)
_FAILURE_RESULT_CODES = frozenset(
    {
        SkillResultCode.TARGET_LOST,
        SkillResultCode.SEARCH_EXHAUSTED,
        SkillResultCode.TIMEOUT,
        SkillResultCode.INVALID_GOAL,
        SkillResultCode.INVALID_STATE,
        SkillResultCode.INTERNAL_ERROR,
    }
)


@dataclass(frozen=True, slots=True)
class SkillGoal:
    """Marker base for typed, Skill-specific goals."""


@dataclass(frozen=True, slots=True)
class SkillFeedback:
    progress: float | None = None
    message: str = ""
    data: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "progress": self.progress,
            "message": self.message,
            "data": deepcopy(self.data),
        }


@dataclass(frozen=True, slots=True)
class SkillResult:
    status: SkillStatus
    code: SkillResultCode
    message: str
    data: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_skill_result(self.status, self.code)
        if not isinstance(self.message, str):
            raise TypeError("SkillResult.message must be a string")
        if not isinstance(self.data, dict):
            raise TypeError("SkillResult.data must be a dict")

    def to_dict(self) -> dict[str, object]:
        """Return an enum-normalized mapping for the Qwen tool layer."""

        return {
            "status": self.status.name,
            "code": self.code.name,
            "message": self.message,
            "data": deepcopy(self.data),
        }


def validate_skill_result(status: SkillStatus, code: SkillResultCode) -> None:
    """Enforce the one-to-one lifecycle/result-code boundary."""

    if not isinstance(status, SkillStatus):
        raise TypeError("status must be a SkillStatus")
    if not isinstance(code, SkillResultCode):
        raise TypeError("code must be a SkillResultCode")
    valid_codes = {
        SkillStatus.SUCCEEDED: _SUCCESS_RESULT_CODES,
        SkillStatus.FAILED: _FAILURE_RESULT_CODES,
        SkillStatus.CANCELED: frozenset({SkillResultCode.CANCELED}),
    }.get(status)
    if valid_codes is None:
        raise ValueError("SkillResult status must be terminal")
    if code not in valid_codes:
        raise ValueError(f"{code.name} is invalid for terminal status {status.name}")


@runtime_checkable
class CameraSensor(Protocol):
    """Minimal structural interface; avoids importing Isaac before SimulationApp."""

    def get_rgb(self) -> np.ndarray: ...

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class SkillClock(Protocol):
    """Simulation-time clock supplied by the runtime."""

    def now(self) -> float: ...


@dataclass(frozen=True, slots=True)
class SkillContext:
    uav: UAVController
    camera: CameraSensor
    perception: object | None
    clock: SkillClock

    def validate(self) -> None:
        if not isinstance(self.uav, UAVController):
            raise TypeError("SkillContext.uav must satisfy UAVController")
        if not isinstance(self.camera, CameraSensor):
            raise TypeError("SkillContext.camera must satisfy CameraSensor")
        if not isinstance(self.clock, SkillClock):
            raise TypeError("SkillContext.clock must satisfy SkillClock")


@dataclass(frozen=True, slots=True)
class Observation:
    """Time-consistent Skill input; privileged fields always use ``oracle_``."""

    timestamp: float
    uav_pose: UAVState
    uav_velocity: np.ndarray
    camera_rgb: np.ndarray

    # These poses belong to the same sampled RGB frame. They are optional for
    # backward-compatible non-visual Skills, but visual Skills may require them.
    camera_position_m: np.ndarray | None = None
    camera_orientation_wxyz: np.ndarray | None = None

    oracle_target_id: str | None = None
    oracle_target_visible: bool | None = None
    oracle_target_pose: TargetState | None = None
    oracle_target_velocity: np.ndarray | None = None

    def validate(self) -> None:
        _finite_scalar(self.timestamp, "Observation.timestamp")
        if not isinstance(self.uav_pose, UAVState):
            raise TypeError("Observation.uav_pose must be a UAVState")
        _validate_pose(self.uav_pose, "Observation.uav_pose")
        _validate_vector3(self.uav_velocity, "Observation.uav_velocity")
        if (
            not isinstance(self.camera_rgb, np.ndarray)
            or self.camera_rgb.ndim != 3
            or self.camera_rgb.shape[0] <= 0
            or self.camera_rgb.shape[1] <= 0
            or self.camera_rgb.shape[2] != 3
        ):
            raise ValueError("Observation.camera_rgb must have shape (height, width, 3)")
        if (self.camera_position_m is None) != (self.camera_orientation_wxyz is None):
            raise ValueError(
                "Observation camera position and orientation must be provided together"
            )
        if self.camera_position_m is not None:
            _validate_vector3(self.camera_position_m, "Observation.camera_position_m")
            _validate_vector4(
                self.camera_orientation_wxyz,
                "Observation.camera_orientation_wxyz",
            )
            if float(np.linalg.norm(self.camera_orientation_wxyz)) <= 1e-12:
                raise ValueError("Observation.camera_orientation_wxyz must be non-zero")
        if self.oracle_target_id is not None and (
            not isinstance(self.oracle_target_id, str)
            or not self.oracle_target_id.strip()
        ):
            raise TypeError("Observation.oracle_target_id must be a non-empty string or None")
        if self.oracle_target_visible is not None and not isinstance(
            self.oracle_target_visible, bool
        ):
            raise TypeError("Observation.oracle_target_visible must be bool or None")
        if self.oracle_target_pose is not None:
            if not isinstance(self.oracle_target_pose, TargetState):
                raise TypeError("Observation.oracle_target_pose must be TargetState or None")
            _validate_pose(self.oracle_target_pose, "Observation.oracle_target_pose")
        if self.oracle_target_velocity is not None:
            _validate_vector3(
                self.oracle_target_velocity,
                "Observation.oracle_target_velocity",
            )


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_pose(value: UAVState | TargetState, name: str) -> None:
    for field_name in ("x", "y", "z", "yaw"):
        _finite_scalar(getattr(value, field_name), f"{name}.{field_name}")


def _validate_vector3(value: object, name: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (3,)
        or not (
            np.issubdtype(value.dtype, np.integer)
            or np.issubdtype(value.dtype, np.floating)
        )
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{name} must be a finite numeric numpy array with shape (3,)")


def _validate_vector4(value: object, name: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (4,)
        or not (
            np.issubdtype(value.dtype, np.integer)
            or np.issubdtype(value.dtype, np.floating)
        )
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{name} must be a finite numeric numpy array with shape (4,)")
