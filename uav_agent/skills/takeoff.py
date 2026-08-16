"""Ideal-kinematic TAKEOFF Skill with bounded vertical motion."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from skills.base import (
    Skill,
    SkillExecutionStateError,
    SkillGoalValidationError,
    require_finite,
    require_positive,
)
from skills.motion_types import MotionPolicy, YawMode
from skills.types import Observation, SkillContext, SkillGoal, SkillResultCode


@dataclass(frozen=True, slots=True)
class TakeoffGoal(SkillGoal):
    """World-frame TAKEOFF goal; lengths are metres and time is seconds."""

    target_altitude: float
    tolerance: float = 0.2
    climb_speed: float = 1.0
    yaw_mode: YawMode = YawMode.KEEP_CURRENT
    yaw_value: float | None = None
    timeout: float = 20.0

    @property
    def motion_policy(self) -> MotionPolicy:
        """Adapt the public TAKEOFF fields to the unified motion contract."""

        return MotionPolicy(
            max_speed=self.climb_speed,
            yaw_mode=self.yaw_mode,
            yaw_value=self.yaw_value,
        )


class TakeoffSkill(Skill):
    """Climb vertically without motors, thrust, or pose teleportation."""

    goal_type = TakeoffGoal

    def __init__(self) -> None:
        super().__init__()
        self._start_altitude: float | None = None
        self._start_time: float | None = None
        self._last_clock_time: float | None = None

    def _validate_goal(self, goal: SkillGoal) -> None:
        typed_goal = goal
        if not isinstance(typed_goal, TakeoffGoal):
            return
        require_finite(typed_goal.target_altitude, "target_altitude")
        require_positive(typed_goal.tolerance, "tolerance")
        require_positive(typed_goal.climb_speed, "climb_speed")
        require_positive(typed_goal.timeout, "timeout")
        if typed_goal.yaw_mode not in {YawMode.KEEP_CURRENT, YawMode.FIXED}:
            raise SkillGoalValidationError(
                "TAKEOFF yaw_mode must be KEEP_CURRENT or FIXED"
            )

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        typed_goal = self._takeoff_goal(goal)
        start_altitude = context.uav.get_pose().z
        target_altitude = float(typed_goal.target_altitude)
        start_time = self._read_clock(context)
        if start_altitude > target_altitude + typed_goal.tolerance:
            raise SkillExecutionStateError(
                "TAKEOFF target_altitude is below the current UAV altitude"
            )
        context.uav.stop()
        self._start_altitude = start_altitude
        self._start_time = start_time
        self._last_clock_time = start_time
        self._set_feedback(
            0.0,
            "TAKEOFF initialized",
            {
                "altitude": start_altitude,
                "target_altitude": target_altitude,
            },
        )

    def _on_tick(self, observation: Observation) -> None:
        goal = self._takeoff_goal(self._active_goal)
        if self._start_altitude is None or self._start_time is None:
            raise SkillExecutionStateError("TAKEOFF was not initialized")

        now = self._read_clock(self._active_context)
        if self._last_clock_time is not None and now < self._last_clock_time - 1e-12:
            raise SkillExecutionStateError("Skill clock moved backwards during TAKEOFF")
        self._last_clock_time = now
        elapsed = max(0.0, now - self._start_time)
        altitude = float(observation.uav_pose.z)
        target_altitude = float(goal.target_altitude)
        altitude_error = target_altitude - altitude
        progress = self._progress(altitude, target_altitude)

        if abs(altitude_error) <= goal.tolerance:
            self._set_feedback(
                1.0,
                "Target altitude reached",
                {
                    "altitude": altitude,
                    "target_altitude": target_altitude,
                    "elapsed_time": elapsed,
                },
            )
            self._succeed(
                SkillResultCode.TAKEOFF_COMPLETE,
                "TAKEOFF complete",
                {
                    "final_altitude": altitude,
                    "target_altitude": target_altitude,
                    "elapsed_time": elapsed,
                },
            )
            return

        if elapsed >= goal.timeout:
            self._set_feedback(
                progress,
                "TAKEOFF timed out",
                {
                    "altitude": altitude,
                    "target_altitude": target_altitude,
                    "elapsed_time": elapsed,
                },
            )
            self._fail(
                SkillResultCode.TIMEOUT,
                "TAKEOFF timed out before reaching target altitude",
                {
                    "final_altitude": altitude,
                    "target_altitude": target_altitude,
                    "elapsed_time": elapsed,
                },
            )
            return

        if altitude_error < -goal.tolerance:
            self._set_feedback(
                progress,
                "UAV is above the TAKEOFF target altitude",
                {
                    "altitude": altitude,
                    "target_altitude": target_altitude,
                    "elapsed_time": elapsed,
                },
            )
            self._fail(
                SkillResultCode.INVALID_STATE,
                "UAV is above the TAKEOFF target altitude",
                {
                    "altitude": altitude,
                    "target_altitude": target_altitude,
                },
            )
            return

        controller_pose = self._active_context.uav.get_pose()
        if controller_pose.z > target_altitude + goal.tolerance:
            self._set_feedback(
                progress,
                "Kinematic UAV state is above the TAKEOFF target altitude",
                {
                    "altitude": controller_pose.z,
                    "target_altitude": target_altitude,
                    "elapsed_time": elapsed,
                },
            )
            self._fail(
                SkillResultCode.INVALID_STATE,
                "Kinematic UAV state is above the TAKEOFF target altitude",
                {
                    "altitude": controller_pose.z,
                    "target_altitude": target_altitude,
                },
            )
            return
        commanded_velocity = self._move_toward_with_motion_policy(
            (controller_pose.x, controller_pose.y, target_altitude),
            goal.climb_speed,
            goal.tolerance,
        )
        self._set_feedback(
            progress,
            "Climbing to target altitude",
            {
                "altitude": altitude,
                "target_altitude": target_altitude,
                "climb_speed": float(commanded_velocity[2]),
                "elapsed_time": elapsed,
            },
        )

    def _on_reset(self) -> None:
        self._start_altitude = None
        self._start_time = None
        self._last_clock_time = None

    def _progress(self, altitude: float, target_altitude: float) -> float:
        if self._start_altitude is None:
            return 0.0
        climb_distance = target_altitude - self._start_altitude
        if climb_distance <= 0.0:
            return 1.0 if abs(target_altitude - altitude) <= 1e-12 else 0.0
        return min(1.0, max(0.0, (altitude - self._start_altitude) / climb_distance))

    @staticmethod
    def _takeoff_goal(goal: SkillGoal) -> TakeoffGoal:
        if not isinstance(goal, TakeoffGoal):
            raise SkillExecutionStateError("active TAKEOFF goal has an invalid type")
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
