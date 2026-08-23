"""Trusted execution of a finite, pre-resolved waypoint route."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

import numpy as np

from common.ids import validate_routing_id
from skills.base import (
    Skill,
    SkillExecutionStateError,
    SkillGoalValidationError,
    require_positive,
    require_vector3,
)
from skills.motion_types import MotionPolicy
from skills.types import Observation, SkillContext, SkillGoal, SkillResultCode


@dataclass(frozen=True, slots=True)
class FollowRouteGoal(SkillGoal):
    """World-frame route produced by a trusted resolver, never directly by Qwen."""

    route_id: str
    waypoints: tuple[tuple[float, float, float], ...]
    tolerance_m: float = 0.75
    timeout_s: float = 120.0
    motion_policy: MotionPolicy = field(default_factory=MotionPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_id", validate_routing_id(self.route_id, "route_id"))
        waypoints = tuple(self.waypoints)
        if not 2 <= len(waypoints) <= 16:
            raise ValueError("FollowRouteGoal.waypoints must contain between 2 and 16 points")
        normalized = tuple(
            require_vector3(point, f"waypoints[{index}]")
            for index, point in enumerate(waypoints)
        )
        object.__setattr__(self, "waypoints", normalized)


class FollowRouteSkill(Skill):
    goal_type = FollowRouteGoal

    def __init__(self) -> None:
        super().__init__()
        self._waypoints: tuple[np.ndarray, ...] = ()
        self._waypoint_index = 0
        self._start_time: float | None = None
        self._last_clock_time: float | None = None
        self._travelled_m = 0.0
        self._last_position: np.ndarray | None = None

    def _validate_goal(self, goal: SkillGoal) -> None:
        if not isinstance(goal, FollowRouteGoal):
            return
        require_positive(goal.tolerance_m, "tolerance_m")
        require_positive(goal.timeout_s, "timeout_s")
        for left, right in zip(goal.waypoints, goal.waypoints[1:]):
            if float(np.linalg.norm(np.asarray(right) - np.asarray(left))) <= 1e-9:
                raise SkillGoalValidationError("adjacent route waypoints must be distinct")

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        typed = self._route_goal(goal)
        now = self._clock(context)
        current = self._position(context.uav.get_pose())
        self._waypoints = tuple(np.asarray(point, dtype=np.float64) for point in typed.waypoints)
        self._waypoint_index = 0
        self._start_time = self._last_clock_time = now
        self._travelled_m = 0.0
        self._last_position = current
        context.uav.stop()
        self._set_feedback(
            0.0,
            "FOLLOW_ROUTE initialized",
            {
                "route_id": typed.route_id,
                "waypoint_index": 0,
                "waypoint_count": len(self._waypoints),
                "distance_remaining": float(np.linalg.norm(self._waypoints[0] - current)),
            },
        )

    def _on_tick(self, observation: Observation) -> None:
        goal = self._route_goal(self._active_goal)
        if self._start_time is None or not self._waypoints:
            raise SkillExecutionStateError("FOLLOW_ROUTE was not initialized")
        now = self._clock(self._active_context)
        if self._last_clock_time is not None and now < self._last_clock_time - 1e-12:
            raise SkillExecutionStateError("Skill clock moved backwards during FOLLOW_ROUTE")
        self._last_clock_time = now
        elapsed = max(0.0, now - self._start_time)
        current = self._position(observation.uav_pose)
        if self._last_position is not None:
            self._travelled_m += float(np.linalg.norm(current - self._last_position))
        self._last_position = current

        while self._waypoint_index < len(self._waypoints):
            distance = float(np.linalg.norm(self._waypoints[self._waypoint_index] - current))
            if distance > goal.tolerance_m:
                break
            self._waypoint_index += 1

        if self._waypoint_index >= len(self._waypoints):
            self._succeed(
                SkillResultCode.ROUTE_COMPLETE,
                "FOLLOW_ROUTE completed",
                {
                    "route_id": goal.route_id,
                    "visited_waypoints": len(self._waypoints),
                    "elapsed_time": elapsed,
                    "travelled_distance_m": self._travelled_m,
                    "final_position": tuple(float(value) for value in current),
                },
            )
            return
        if elapsed >= goal.timeout_s:
            self._fail(
                SkillResultCode.TIMEOUT,
                "FOLLOW_ROUTE timed out",
                {
                    "route_id": goal.route_id,
                    "waypoint_index": self._waypoint_index,
                    "elapsed_time": elapsed,
                    "travelled_distance_m": self._travelled_m,
                },
            )
            return

        target = self._waypoints[self._waypoint_index]
        distance = float(np.linalg.norm(target - current))
        velocity = self._move_toward_with_motion_policy(
            target,
            self._active_context.uav.max_speed_mps,
            goal.tolerance_m,
        )
        completed = self._waypoint_index / len(self._waypoints)
        self._set_feedback(
            completed,
            "Following trusted route",
            {
                "route_id": goal.route_id,
                "waypoint_index": self._waypoint_index,
                "waypoint_count": len(self._waypoints),
                "distance_remaining": distance,
                "commanded_speed": float(np.linalg.norm(velocity)),
                "elapsed_time": elapsed,
                "travelled_distance_m": self._travelled_m,
            },
        )

    def _on_reset(self) -> None:
        self._waypoints = ()
        self._waypoint_index = 0
        self._start_time = None
        self._last_clock_time = None
        self._travelled_m = 0.0
        self._last_position = None

    @staticmethod
    def _route_goal(value: SkillGoal | None) -> FollowRouteGoal:
        if not isinstance(value, FollowRouteGoal):
            raise SkillExecutionStateError("active FOLLOW_ROUTE goal has an invalid type")
        return value

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
    def _clock(context: SkillContext) -> float:
        try:
            value = context.clock.now()
        except Exception as exc:
            raise SkillExecutionStateError(f"could not read Skill clock: {exc}") from exc
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise SkillExecutionStateError("Skill clock must return a finite number")
        return float(value)


__all__ = ["FollowRouteGoal", "FollowRouteSkill"]
