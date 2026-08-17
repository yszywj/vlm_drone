"""Ideal-kinematic LAND Skill with locked XY and bounded descent."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, sqrt
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
class LandGoal(SkillGoal):
    """World-frame LAND request; lengths are metres and time is seconds."""

    ground_altitude: float = 0.0
    tolerance: float = 0.1
    descent_speed: float = 0.5
    yaw_mode: YawMode = YawMode.KEEP_CURRENT
    yaw_value: float | None = None
    timeout: float = 30.0
    expected_position_xy: tuple[float, float] | None = None
    zone_tolerance_m: float = 0.75

    def __post_init__(self) -> None:
        if self.expected_position_xy is not None:
            value = self.expected_position_xy
            if (
                isinstance(value, (str, bytes))
                or not isinstance(value, tuple)
                or len(value) != 2
            ):
                raise ValueError(
                    "expected_position_xy must be a two-number tuple or None"
                )
            normalized: list[float] = []
            for index, component in enumerate(value):
                if (
                    isinstance(component, bool)
                    or not isinstance(component, Real)
                    or not isfinite(component)
                ):
                    raise ValueError(
                        "expected_position_xy must contain two finite numbers"
                    )
                normalized.append(float(component))
            object.__setattr__(
                self,
                "expected_position_xy",
                (normalized[0], normalized[1]),
            )

    @property
    def motion_policy(self) -> MotionPolicy:
        """Adapt the public LAND fields to the unified motion contract."""

        return MotionPolicy(
            max_speed=self.descent_speed,
            yaw_mode=self.yaw_mode,
            yaw_value=self.yaw_value,
        )


class LandSkill(Skill):
    """Descend to a configured ground height without PX4 or pose teleportation."""

    goal_type = LandGoal

    def __init__(self) -> None:
        super().__init__()
        self._landing_x: float | None = None
        self._landing_y: float | None = None
        self._start_altitude: float | None = None
        self._start_time: float | None = None
        self._last_clock_time: float | None = None

    def _validate_goal(self, goal: SkillGoal) -> None:
        typed_goal = goal
        if not isinstance(typed_goal, LandGoal):
            return
        require_finite(typed_goal.ground_altitude, "ground_altitude")
        require_positive(typed_goal.tolerance, "tolerance")
        require_positive(typed_goal.descent_speed, "descent_speed")
        require_positive(typed_goal.timeout, "timeout")
        require_positive(typed_goal.zone_tolerance_m, "zone_tolerance_m")
        if typed_goal.yaw_mode not in {YawMode.KEEP_CURRENT, YawMode.FIXED}:
            raise SkillGoalValidationError(
                "LAND yaw_mode must be KEEP_CURRENT or FIXED"
            )

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        typed_goal = self._land_goal(goal)
        pose = context.uav.get_pose()
        ground_altitude = float(typed_goal.ground_altitude)
        start_time = self._read_clock(context)
        if pose.z < ground_altitude - typed_goal.tolerance:
            raise SkillExecutionStateError(
                "LAND ground_altitude is above the current UAV altitude"
            )
        zone_error = self._zone_error(
            float(pose.x),
            float(pose.y),
            typed_goal.expected_position_xy,
        )
        if (
            zone_error is not None
            and zone_error > typed_goal.zone_tolerance_m
        ):
            self._set_feedback(
                0.0,
                "UAV is outside the trusted landing zone",
                {
                    "altitude": float(pose.z),
                    "ground_altitude": ground_altitude,
                    "landing_position_xy": (float(pose.x), float(pose.y)),
                    "horizontal_error": 0.0,
                    "expected_position_xy": typed_goal.expected_position_xy,
                    "zone_error_m": zone_error,
                    "zone_tolerance_m": float(typed_goal.zone_tolerance_m),
                },
            )
            raise SkillExecutionStateError(
                "normal LAND requires the UAV to already be inside the "
                "trusted landing zone"
            )

        context.uav.stop()
        self._landing_x = float(pose.x)
        self._landing_y = float(pose.y)
        self._start_altitude = float(pose.z)
        self._start_time = start_time
        self._last_clock_time = start_time
        self._set_feedback(
            0.0,
            "LAND initialized",
            {
                "altitude": float(pose.z),
                "ground_altitude": ground_altitude,
                "landing_position_xy": (self._landing_x, self._landing_y),
                "horizontal_error": 0.0,
                "expected_position_xy": typed_goal.expected_position_xy,
                "zone_error_m": zone_error,
                "zone_tolerance_m": float(typed_goal.zone_tolerance_m),
            },
        )

    def _on_tick(self, observation: Observation) -> None:
        goal = self._land_goal(self._active_goal)
        if (
            self._landing_x is None
            or self._landing_y is None
            or self._start_altitude is None
            or self._start_time is None
        ):
            raise SkillExecutionStateError("LAND was not initialized")

        now = self._read_clock(self._active_context)
        if self._last_clock_time is not None and now < self._last_clock_time - 1e-12:
            raise SkillExecutionStateError("Skill clock moved backwards during LAND")
        self._last_clock_time = now
        elapsed = max(0.0, now - self._start_time)

        pose = observation.uav_pose
        altitude = float(pose.z)
        ground_altitude = float(goal.ground_altitude)
        vertical_error = altitude - ground_altitude
        horizontal_error = hypot(
            float(pose.x) - self._landing_x,
            float(pose.y) - self._landing_y,
        )
        zone_error = self._zone_error(
            float(pose.x),
            float(pose.y),
            goal.expected_position_xy,
        )
        progress = self._progress(altitude, ground_altitude)
        feedback_data: dict[str, object] = {
            "altitude": altitude,
            "ground_altitude": ground_altitude,
            "vertical_distance_remaining": max(0.0, vertical_error),
            "horizontal_error": horizontal_error,
            "landing_position_xy": (self._landing_x, self._landing_y),
            "elapsed_time": elapsed,
            "expected_position_xy": goal.expected_position_xy,
            "zone_error_m": zone_error,
            "zone_tolerance_m": float(goal.zone_tolerance_m),
        }

        if zone_error is not None and zone_error > goal.zone_tolerance_m:
            self._set_feedback(
                progress,
                "UAV left the trusted landing zone",
                feedback_data,
            )
            self._fail(
                SkillResultCode.INVALID_STATE,
                "UAV moved outside the trusted landing zone during LAND",
                feedback_data,
            )
            return

        if abs(vertical_error) <= goal.tolerance:
            final_position = (float(pose.x), float(pose.y), altitude)
            self._set_feedback(1.0, "Ground altitude reached", feedback_data)
            self._succeed(
                SkillResultCode.LAND_COMPLETE,
                "LAND complete",
                {
                    "final_position": final_position,
                    "final_altitude": altitude,
                    "ground_altitude": ground_altitude,
                    "landing_position_xy": (self._landing_x, self._landing_y),
                    "expected_position_xy": goal.expected_position_xy,
                    "zone_error_m": zone_error,
                    "zone_tolerance_m": float(goal.zone_tolerance_m),
                    "is_airborne": altitude > ground_altitude + goal.tolerance,
                    "elapsed_time": elapsed,
                },
            )
            return

        if elapsed >= goal.timeout:
            self._set_feedback(progress, "LAND timed out", feedback_data)
            self._fail(
                SkillResultCode.TIMEOUT,
                "LAND timed out before reaching ground altitude",
                {
                    "final_position": (float(pose.x), float(pose.y), altitude),
                    "ground_altitude": ground_altitude,
                    "elapsed_time": elapsed,
                },
            )
            return

        if vertical_error < -goal.tolerance:
            self._set_feedback(progress, "UAV is below ground altitude", feedback_data)
            self._fail(
                SkillResultCode.INVALID_STATE,
                "UAV is below the configured LAND ground altitude",
                feedback_data,
            )
            return

        controller_pose = self._active_context.uav.get_pose()
        if controller_pose.z < ground_altitude - goal.tolerance:
            self._fail(
                SkillResultCode.INVALID_STATE,
                "UAV controller state is below the LAND ground altitude",
                {
                    "altitude": float(controller_pose.z),
                    "ground_altitude": ground_altitude,
                },
            )
            return

        commanded_velocity = self._move_toward_with_motion_policy(
            (self._landing_x, self._landing_y, ground_altitude),
            goal.descent_speed,
            goal.tolerance,
        )
        feedback_data["commanded_velocity_mps"] = tuple(
            float(value) for value in commanded_velocity
        )
        feedback_data["commanded_speed_mps"] = sqrt(
            sum(float(value) ** 2 for value in commanded_velocity)
        )
        self._set_feedback(progress, "Descending to ground altitude", feedback_data)

    def _progress(self, altitude: float, ground_altitude: float) -> float:
        if self._start_altitude is None:
            return 0.0
        descent_distance = self._start_altitude - ground_altitude
        if descent_distance <= 0.0:
            return 1.0 if abs(altitude - ground_altitude) <= 1e-12 else 0.0
        descended = self._start_altitude - altitude
        return min(1.0, max(0.0, descended / descent_distance))

    @staticmethod
    def _zone_error(
        x: float,
        y: float,
        expected_position_xy: tuple[float, float] | None,
    ) -> float | None:
        if expected_position_xy is None:
            return None
        return hypot(
            x - expected_position_xy[0],
            y - expected_position_xy[1],
        )

    def _on_reset(self) -> None:
        self._landing_x = None
        self._landing_y = None
        self._start_altitude = None
        self._start_time = None
        self._last_clock_time = None

    @staticmethod
    def _land_goal(goal: SkillGoal) -> LandGoal:
        if not isinstance(goal, LandGoal):
            raise SkillExecutionStateError("active LAND goal has an invalid type")
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
