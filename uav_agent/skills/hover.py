"""Continuously commanded bounded HOVER for planned or supervisory pauses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from numbers import Real
from threading import Event

import numpy as np

from common.ids import validate_routing_id
from skills.base import (
    Skill,
    SkillExecutionStateError,
    SkillGoalValidationError,
    require_positive,
)
from skills.motion_types import MotionPolicy, YawMode, apply_motion_policy
from skills.types import Observation, SkillContext, SkillGoal, SkillResultCode, SkillStatus


class HoverMode(str, Enum):
    TIMED = "TIMED"
    UNTIL_RELEASED = "UNTIL_RELEASED"


class HoverTimeoutFallback(str, Enum):
    """Trusted policy applied by the Manager after supervisory timeout."""

    RESUME_PREVIOUS = "RESUME_PREVIOUS"
    CANCEL_AND_LAND = "CANCEL_AND_LAND"


@dataclass(frozen=True, slots=True)
class HoverGoal(SkillGoal):
    mode: HoverMode = HoverMode.TIMED
    duration_s: float | None = 1.0
    max_wait_s: float = 20.0
    position_tolerance_m: float = 0.25
    max_correction_speed_mps: float = 0.5
    reason_code: str = "PLANNED_HOVER"
    motion_policy: MotionPolicy = field(
        default_factory=lambda: MotionPolicy(yaw_mode=YawMode.KEEP_CURRENT)
    )


class HoverSkill(Skill):
    """Hold a captured world pose without blocking the simulation thread."""

    goal_type = HoverGoal

    def __init__(self) -> None:
        super().__init__()
        self._hold_position: np.ndarray | None = None
        self._start_time: float | None = None
        self._last_clock_time: float | None = None
        self._last_observation_timestamp: float | None = None
        self._hold_established = False
        self._release_requested = Event()

    @property
    def start_time_s(self) -> float | None:
        """Clock boundary after which an Observation may drive this HOVER."""

        return self._start_time

    @property
    def hold_established(self) -> bool:
        """Whether at least one post-start hold setpoint has been issued."""

        return self._hold_established

    def observation_precedes_start(self, timestamp_s: float) -> bool:
        """Return whether a sampled frame predates the captured hold pose.

        SkillManager uses this narrow query to defer a HOVER created midway
        through a runtime tick without weakening the timestamp checks in the
        Skill itself.
        """

        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, Real):
            raise TypeError("timestamp_s must be a finite number")
        normalized = float(timestamp_s)
        if not isfinite(normalized):
            raise ValueError("timestamp_s must be a finite number")
        return self._start_time is not None and normalized < self._start_time - 1e-12

    def request_release(self) -> None:
        """Thread-safe signal; lifecycle transition happens on the next tick."""

        if self.status is not SkillStatus.RUNNING:
            raise SkillExecutionStateError(
                "request_release requires a running supervisory HOVER"
            )
        goal = self._hover_goal(self._active_goal)
        if goal.mode is not HoverMode.UNTIL_RELEASED:
            raise SkillExecutionStateError(
                "request_release is only valid for UNTIL_RELEASED HOVER"
            )
        self._release_requested.set()

    def _validate_goal(self, goal: SkillGoal) -> None:
        typed = self._hover_goal(goal)
        if not isinstance(typed.mode, HoverMode):
            raise SkillGoalValidationError("mode must be a HoverMode")
        max_wait = require_positive(typed.max_wait_s, "max_wait_s")
        require_positive(typed.position_tolerance_m, "position_tolerance_m")
        require_positive(
            typed.max_correction_speed_mps,
            "max_correction_speed_mps",
        )
        try:
            validate_routing_id(typed.reason_code, "reason_code")
        except (TypeError, ValueError) as exc:
            raise SkillGoalValidationError(str(exc)) from exc
        if typed.mode is HoverMode.TIMED:
            duration = require_positive(typed.duration_s, "duration_s")
            if duration > max_wait:
                raise SkillGoalValidationError(
                    "duration_s must not exceed trusted max_wait_s"
                )
        elif typed.duration_s is not None:
            raise SkillGoalValidationError(
                "UNTIL_RELEASED HOVER requires duration_s=None"
            )
        if typed.motion_policy.yaw_mode not in {
            YawMode.KEEP_CURRENT,
            YawMode.FIXED,
        }:
            raise SkillGoalValidationError(
                "HOVER yaw_mode must be KEEP_CURRENT or FIXED"
            )

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        del goal
        pose = context.uav.get_pose()
        now = self._read_clock(context)
        self._release_requested.clear()
        self._hold_position = np.asarray(
            (float(pose.x), float(pose.y), float(pose.z)),
            dtype=np.float64,
        )
        self._start_time = now
        self._last_clock_time = now
        self._last_observation_timestamp = None
        self._hold_established = False
        context.uav.stop()
        self._set_feedback(
            0.0,
            "HOVER initialized",
            self._feedback_data(elapsed=0.0, drift=0.0),
        )

    def _on_tick(self, observation: Observation) -> None:
        goal = self._hover_goal(self._active_goal)
        if self._hold_position is None or self._start_time is None:
            raise SkillExecutionStateError("HOVER was not initialized")
        now = self._read_clock(self._active_context)
        if self._last_clock_time is not None and now < self._last_clock_time - 1e-12:
            raise SkillExecutionStateError("Skill clock moved backwards during HOVER")
        self._last_clock_time = now
        if (
            self._last_observation_timestamp is not None
            and observation.timestamp < self._last_observation_timestamp - 1e-12
        ):
            raise SkillExecutionStateError(
                "Observation timestamp moved backwards during HOVER"
            )
        self._last_observation_timestamp = float(observation.timestamp)
        if observation.timestamp < self._start_time - 1e-12:
            raise SkillExecutionStateError(
                "HOVER received an Observation captured before Skill start"
            )
        elapsed = max(
            0.0,
            now - self._start_time,
            float(observation.timestamp) - self._start_time,
        )
        current = np.asarray(
            (
                float(observation.uav_pose.x),
                float(observation.uav_pose.y),
                float(observation.uav_pose.z),
            ),
            dtype=np.float64,
        )
        delta = self._hold_position - current
        drift = float(np.linalg.norm(delta))
        velocity = np.zeros(3, dtype=np.float64)
        if drift > goal.position_tolerance_m:
            velocity = (
                delta
                / drift
                * min(
                    float(goal.max_correction_speed_mps),
                    float(self._active_context.uav.max_speed_mps),
                )
            )
        # This command is deliberately re-issued on every frame. HOVER is a
        # continuously supervised setpoint, not a one-shot stop().
        apply_motion_policy(
            self._active_context.uav,
            velocity,
            goal.motion_policy,
            initial_yaw=self.initial_yaw,
        )
        self._hold_established = True
        progress = (
            min(1.0, elapsed / float(goal.duration_s))
            if goal.mode is HoverMode.TIMED and goal.duration_s is not None
            else min(1.0, elapsed / float(goal.max_wait_s))
        )
        data = self._feedback_data(elapsed=elapsed, drift=drift)
        if goal.mode is HoverMode.UNTIL_RELEASED and self._release_requested.is_set():
            self._set_feedback(1.0, "HOVER released", data)
            self._succeed(
                SkillResultCode.HOVER_COMPLETE,
                "supervisory HOVER released",
                data,
            )
            return
        if goal.mode is HoverMode.TIMED and elapsed >= float(goal.duration_s):
            self._set_feedback(1.0, "HOVER duration complete", data)
            self._succeed(
                SkillResultCode.HOVER_COMPLETE,
                "timed HOVER complete",
                data,
            )
            return
        if elapsed >= goal.max_wait_s:
            self._set_feedback(progress, "HOVER timed out", data)
            self._fail(
                SkillResultCode.TIMEOUT,
                "HOVER exceeded trusted max_wait_s",
                data,
            )
            return
        self._set_feedback(progress, "Holding position", data)

    def _feedback_data(self, *, elapsed: float, drift: float) -> dict[str, object]:
        goal = self._hover_goal(self._active_goal)
        return {
            "uav_id": self._active_context.uav_id,
            "elapsed_time": float(elapsed),
            "position_drift_m": float(drift),
            "captured_hold_position": (
                None
                if self._hold_position is None
                else tuple(float(value) for value in self._hold_position)
            ),
            "mode": goal.mode.value,
            "reason_code": goal.reason_code,
            "hold_established": self._hold_established,
        }

    def _on_reset(self) -> None:
        self._hold_position = None
        self._start_time = None
        self._last_clock_time = None
        self._last_observation_timestamp = None
        self._hold_established = False
        self._release_requested.clear()

    @staticmethod
    def _hover_goal(goal: SkillGoal) -> HoverGoal:
        if not isinstance(goal, HoverGoal):
            raise SkillExecutionStateError("active HOVER goal has invalid type")
        return goal

    @staticmethod
    def _read_clock(context: SkillContext) -> float:
        value = context.clock.now()
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise SkillExecutionStateError("Skill clock must return a finite number")
        return float(value)


__all__ = ["HoverGoal", "HoverMode", "HoverSkill", "HoverTimeoutFallback"]
