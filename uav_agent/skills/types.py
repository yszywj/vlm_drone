"""Shared, Isaac-independent types for all callable UAV skills."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from math import isfinite
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np

from common.target_estimate import TargetEstimate
from env.moving_target import TargetState
from env.uav_controller import UAVController, UAVState
from common.ids import (
    validate_invocation_id,
    validate_mission_id,
    validate_routing_id,
    validate_uav_id,
)


class SkillName(str, Enum):
    TAKEOFF = "TAKEOFF"
    GOTO = "GOTO"
    FOLLOW_ROUTE = "FOLLOW_ROUTE"
    HOVER = "HOVER"
    SEARCH = "SEARCH"
    INSPECT = "INSPECT"
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
    ROUTE_COMPLETE = auto()
    HOVER_COMPLETE = auto()
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
        SkillResultCode.ROUTE_COMPLETE,
        SkillResultCode.HOVER_COMPLETE,
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
    # Routing is a trust boundary: callers must bind the context explicitly.
    uav_id: str = field(kw_only=True)

    def validate(self) -> None:
        validate_uav_id(getattr(self, "uav_id", None))
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

    # Backend-neutral target state consumed by SEARCH/TRACK/REACQUIRE.  The
    # legacy oracle_target_* fields below remain evaluator-only compatibility
    # data and must never be read by production Skill logic.
    target_estimate: TargetEstimate | None = None

    oracle_target_id: str | None = None
    oracle_target_visible: bool | None = None
    oracle_target_pose: TargetState | None = None
    oracle_target_velocity: np.ndarray | None = None
    # Kept keyword-only so a frame cannot silently inherit another UAV's
    # identity from positional compatibility or a process-wide default.
    uav_id: str = field(kw_only=True)

    def validate(self) -> None:
        validate_uav_id(getattr(self, "uav_id", None))
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
        if self.target_estimate is not None:
            if not isinstance(self.target_estimate, TargetEstimate):
                raise TypeError(
                    "Observation.target_estimate must be TargetEstimate or None"
                )
            if self.target_estimate.timestamp_s > float(self.timestamp) + 1e-9:
                raise ValueError(
                    "Observation.target_estimate cannot be newer than the frame"
                )
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
        # Backward-compatible evaluator adapter for older pure-Python tests
        # and callers that construct the legacy Oracle fields directly.  New
        # backends (including OraclePerception) populate target_estimate
        # explicitly; Skills consume only that neutral field.
        if (
            self.target_estimate is None
            and isinstance(self.oracle_target_visible, bool)
            and isinstance(self.oracle_target_id, str)
            and self.oracle_target_id.strip()
            and isinstance(self.oracle_target_pose, TargetState)
        ):
            pose = self.oracle_target_pose
            velocity = self.oracle_target_velocity
            object.__setattr__(
                self,
                "target_estimate",
                TargetEstimate(
                    timestamp_s=float(self.timestamp),
                    target_id=self.oracle_target_id.strip(),
                    candidate_id=None,
                    tracker_id=None,
                    visible=self.oracle_target_visible,
                    confirmed=True,
                    predicted_only=False,
                    class_id=None,
                    class_name=None,
                    confidence=1.0,
                    bbox_xyxy_normalized=(0.0, 0.0, 1.0, 1.0)
                    if self.oracle_target_visible
                    else None,
                    position_world_m=(float(pose.x), float(pose.y), float(pose.z)),
                    velocity_world_mps=(
                        None
                        if velocity is None
                        else tuple(float(component) for component in velocity)
                    ),
                    measurement_age_s=0.0,
                    source="oracle_evaluation",
                ),
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


@dataclass(frozen=True, slots=True)
class SkillInvocation:
    """Immutable routing envelope for one concrete Skill start."""

    mission_id: str
    uav_id: str
    plan_version: int
    step_id: str
    invocation_id: str
    skill_name: SkillName
    goal: SkillGoal

    def __post_init__(self) -> None:
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if isinstance(self.plan_version, bool) or not isinstance(
            self.plan_version, int
        ) or self.plan_version <= 0:
            raise ValueError("plan_version must be a positive integer")
        object.__setattr__(
            self,
            "step_id",
            validate_routing_id(self.step_id, "step_id"),
        )
        object.__setattr__(
            self,
            "invocation_id",
            validate_invocation_id(self.invocation_id),
        )
        if not isinstance(self.skill_name, SkillName):
            raise TypeError("skill_name must be a SkillName")
        if not isinstance(self.goal, SkillGoal):
            raise TypeError("goal must be a SkillGoal")

    def to_dict(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "step_id": self.step_id,
            "invocation_id": self.invocation_id,
            "skill_name": self.skill_name.value,
            "goal_type": type(self.goal).__name__,
        }


@dataclass(frozen=True, slots=True)
class SkillExecutionReport:
    """Routed public feedback/result emitted for one Skill invocation."""

    mission_id: str
    uav_id: str
    plan_version: int
    step_id: str
    invocation_id: str
    skill_name: SkillName
    status: SkillStatus
    result_code: SkillResultCode | None
    feedback_or_result: dict[str, object]
    timestamp_s: float

    def __post_init__(self) -> None:
        validate_mission_id(self.mission_id)
        validate_uav_id(self.uav_id)
        if isinstance(self.plan_version, bool) or not isinstance(
            self.plan_version, int
        ) or self.plan_version <= 0:
            raise ValueError("plan_version must be a positive integer")
        validate_routing_id(self.step_id, "step_id")
        validate_invocation_id(self.invocation_id)
        if not isinstance(self.skill_name, SkillName):
            raise TypeError("skill_name must be a SkillName")
        if not isinstance(self.status, SkillStatus):
            raise TypeError("status must be a SkillStatus")
        if self.result_code is not None and not isinstance(
            self.result_code, SkillResultCode
        ):
            raise TypeError("result_code must be a SkillResultCode or None")
        if not isinstance(self.feedback_or_result, dict):
            raise TypeError("feedback_or_result must be a dict")
        _finite_scalar(self.timestamp_s, "timestamp_s")
        object.__setattr__(
            self,
            "feedback_or_result",
            deepcopy(self.feedback_or_result),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "step_id": self.step_id,
            "invocation_id": self.invocation_id,
            "skill_name": self.skill_name.value,
            "status": self.status.name,
            "result_code": (
                None if self.result_code is None else self.result_code.name
            ),
            "feedback_or_result": deepcopy(self.feedback_or_result),
            "timestamp_s": float(self.timestamp_s),
        }
