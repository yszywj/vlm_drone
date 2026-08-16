"""Ideal-kinematic GOTO Skill with independent translation and yaw control."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

import numpy as np

from skills.base import (
    Skill,
    SkillExecutionStateError,
    require_positive,
    require_vector3,
)
from skills.motion_types import MotionPolicy
from skills.types import Observation, SkillContext, SkillGoal, SkillResultCode


@dataclass(frozen=True, slots=True)
class GotoGoal(SkillGoal):
    """World-frame position goal; lengths are metres and time is seconds."""

    position: tuple[float, float, float]
    tolerance: float = 1.0
    motion_policy: MotionPolicy = field(default_factory=MotionPolicy)
    timeout: float = 60.0


class GotoSkill(Skill):
    """Move continuously toward a 3-D point while applying a yaw policy."""

    goal_type = GotoGoal

    def __init__(self) -> None:
        super().__init__()
        self._goal_position: np.ndarray | None = None
        self._start_distance: float | None = None
        self._start_time: float | None = None
        self._last_clock_time: float | None = None

    def _validate_goal(self, goal: SkillGoal) -> None:
        typed_goal = goal
        if not isinstance(typed_goal, GotoGoal):
            return
        require_vector3(typed_goal.position, "position")
        require_positive(typed_goal.tolerance, "tolerance")
        require_positive(typed_goal.timeout, "timeout")

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        typed_goal = self._goto_goal(goal)
        goal_position = np.asarray(
            require_vector3(typed_goal.position, "position"),
            dtype=np.float64,
        )
        current_position = self._position(context.uav.get_pose())
        start_time = self._read_clock(context)
        start_distance = float(np.linalg.norm(goal_position - current_position))

        # A new Skill owns the controller commands, but never changes pose in
        # start(). Motion begins only when the simulation later calls step().
        context.uav.stop()
        self._goal_position = goal_position
        self._start_distance = start_distance
        self._start_time = start_time
        self._last_clock_time = start_time
        self._set_feedback(
            1.0 if start_distance <= typed_goal.tolerance else 0.0,
            "GOTO initialized",
            {
                "distance_remaining": start_distance,
                "goal_position": tuple(float(value) for value in goal_position),
            },
        )

    def _on_tick(self, observation: Observation) -> None:
        goal = self._goto_goal(self._active_goal)
        if (
            self._goal_position is None
            or self._start_distance is None
            or self._start_time is None
        ):
            raise SkillExecutionStateError("GOTO was not initialized")

        now = self._read_clock(self._active_context)
        if self._last_clock_time is not None and now < self._last_clock_time - 1e-12:
            raise SkillExecutionStateError("Skill clock moved backwards during GOTO")
        self._last_clock_time = now
        elapsed = max(0.0, now - self._start_time)

        observed_position = self._position(observation.uav_pose)
        distance = float(np.linalg.norm(self._goal_position - observed_position))
        progress = self._progress(distance, goal.tolerance)
        goal_position = tuple(float(value) for value in self._goal_position)
        current_position = tuple(float(value) for value in observed_position)

        # Position defines GOTO completion. A yaw target is allowed to remain
        # unfinished; successful completion stops all residual commands.
        if distance <= goal.tolerance:
            self._set_feedback(
                1.0,
                "GOTO position reached",
                {
                    "distance_remaining": distance,
                    "goal_position": goal_position,
                    "elapsed_time": elapsed,
                },
            )
            self._succeed(
                SkillResultCode.GOAL_REACHED,
                "GOTO goal reached",
                {
                    "final_position": current_position,
                    "goal_position": goal_position,
                    "distance_remaining": distance,
                    "elapsed_time": elapsed,
                },
            )
            return

        # Reaching the goal exactly at the deadline is a success, hence the
        # timeout check deliberately follows the position check.
        if elapsed >= goal.timeout:
            self._set_feedback(
                progress,
                "GOTO timed out",
                {
                    "distance_remaining": distance,
                    "goal_position": goal_position,
                    "elapsed_time": elapsed,
                },
            )
            self._fail(
                SkillResultCode.TIMEOUT,
                "GOTO timed out before reaching its position",
                {
                    "final_position": current_position,
                    "goal_position": goal_position,
                    "distance_remaining": distance,
                    "elapsed_time": elapsed,
                },
            )
            return

        # The controller receives translation and yaw targets together. It
        # integrates both during the same subsequent simulation step.
        commanded_velocity = self._move_toward_with_motion_policy(
            self._goal_position,
            self._active_context.uav.max_speed_mps,
            goal.tolerance,
        )
        self._set_feedback(
            progress,
            "Moving toward GOTO position",
            {
                "distance_remaining": distance,
                "goal_position": goal_position,
                "commanded_speed": float(np.linalg.norm(commanded_velocity)),
                "elapsed_time": elapsed,
            },
        )

    def _on_reset(self) -> None:
        self._goal_position = None
        self._start_distance = None
        self._start_time = None
        self._last_clock_time = None

    def _progress(self, distance: float, tolerance: float) -> float:
        if self._start_distance is None:
            return 0.0
        if self._start_distance <= tolerance:
            return 1.0
        travel_distance = self._start_distance - tolerance
        return min(1.0, max(0.0, (self._start_distance - distance) / travel_distance))

    @staticmethod
    def _position(pose: object) -> np.ndarray:
        try:
            values = (pose.x, pose.y, pose.z)
        except AttributeError as exc:
            raise SkillExecutionStateError("UAV pose has no world position") from exc
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            for value in values
        ):
            raise SkillExecutionStateError("UAV world position must be finite")
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _goto_goal(goal: SkillGoal) -> GotoGoal:
        if not isinstance(goal, GotoGoal):
            raise SkillExecutionStateError("active GOTO goal has an invalid type")
        return goal

    @staticmethod
    def _read_clock(context: SkillContext) -> float:
        try:
            value = context.clock.now()
        except Exception as exc:
            raise SkillExecutionStateError(f"could not read Skill clock: {exc}") from exc
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise SkillExecutionStateError("Skill clock must return a finite number")
        return float(value)
