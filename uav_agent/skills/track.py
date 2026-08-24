"""Ideal Oracle-backed TRACK Skill with bounded kinematic following."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isfinite, sin
from numbers import Real

import numpy as np

from skills.base import (
    Skill,
    SkillExecutionStateError,
    SkillGoalValidationError,
    require_positive,
)
from skills.motion_types import MotionPolicy, YawMode, move_toward_with_policy
from skills.types import Observation, SkillContext, SkillGoal, SkillResultCode


@dataclass(frozen=True, slots=True)
class TrackGoal(SkillGoal):
    """Single-target tracking request; lengths are metres and time is seconds."""

    target_id: str
    desired_distance: float = 6.0
    desired_altitude: float = 8.0
    max_speed: float = 2.0
    max_target_lost_time: float = 2.0
    timeout: float | None = None
    track_duration: float | None = None


class TrackSkill(Skill):
    """Follow one confirmed target while keeping the camera-facing yaw free."""

    goal_type = TrackGoal
    POSITION_TOLERANCE_M = 0.1

    def __init__(self) -> None:
        super().__init__()
        self._start_time: float | None = None
        self._last_clock_time: float | None = None
        self._last_observation_timestamp: float | None = None
        self._stand_off_direction_xy: np.ndarray | None = None
        self._last_seen_time: float | None = None
        self._last_seen_position: np.ndarray | None = None
        self._last_seen_velocity: np.ndarray | None = None
        self._perception_source: str | None = None
        self._tracker_id: str | None = None
        self._measurement_age_s: float | None = None
        self._predicted_only = False

    def _validate_goal(self, goal: SkillGoal) -> None:
        typed_goal = goal
        if not isinstance(typed_goal, TrackGoal):
            return
        if not isinstance(typed_goal.target_id, str) or not typed_goal.target_id.strip():
            raise SkillGoalValidationError("target_id must be a non-empty string")
        require_positive(typed_goal.desired_distance, "desired_distance")
        require_positive(typed_goal.desired_altitude, "desired_altitude")
        require_positive(typed_goal.max_speed, "max_speed")
        require_positive(
            typed_goal.max_target_lost_time,
            "max_target_lost_time",
        )
        if typed_goal.timeout is not None:
            require_positive(typed_goal.timeout, "timeout")
        if typed_goal.track_duration is not None:
            require_positive(typed_goal.track_duration, "track_duration")

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        typed_goal = self._track_goal(goal)
        start_time = self._read_clock(context)
        context.uav.stop()
        self._start_time = start_time
        self._last_clock_time = start_time
        self._last_observation_timestamp = None
        self._stand_off_direction_xy = None
        self._clear_last_seen()
        self._set_feedback(
            0.0 if typed_goal.track_duration is not None else None,
            "TRACK initialized",
            {
                "target_distance": None,
                "distance_error": None,
                "target_visible": False,
                "target_relative_bearing": None,
                "last_seen_age": 0.0,
                "tracking_duration": 0.0,
                "perception_source": None,
                "tracker_id": None,
                "measurement_age_s": None,
                "predicted_only": False,
                "position_available": False,
                "velocity_available": False,
            },
        )

    def _on_tick(self, observation: Observation) -> None:
        goal = self._track_goal(self._active_goal)
        if self._start_time is None:
            raise SkillExecutionStateError("TRACK was not initialized")

        now = self._read_clock(self._active_context)
        if self._last_clock_time is not None and now < self._last_clock_time - 1e-12:
            raise SkillExecutionStateError("Skill clock moved backwards during TRACK")
        self._last_clock_time = now
        previous_observation_time = self._last_observation_timestamp
        if (
            previous_observation_time is not None
            and observation.timestamp < previous_observation_time - 1e-12
        ):
            raise SkillExecutionStateError(
                "Observation timestamp moved backwards during TRACK"
            )
        is_new_frame = (
            previous_observation_time is None
            or observation.timestamp > previous_observation_time + 1e-12
        )
        self._last_observation_timestamp = float(observation.timestamp)

        observation_time = float(observation.timestamp)
        frame_duration = observation_time - self._start_time
        if frame_duration < -1e-12:
            raise SkillExecutionStateError(
                "TRACK received an Observation captured before Skill start"
            )
        effective_time = max(now, observation_time)
        tracking_duration = max(0.0, effective_time - self._start_time)

        estimate = observation.target_estimate
        matching_estimate = (
            estimate is not None
            and estimate.confirmed
            and estimate.target_id == goal.target_id.strip()
        )
        position_available = bool(
            matching_estimate
            and estimate.position_world_m is not None
            and (estimate.visible or estimate.predicted_only)
        )
        velocity_available = bool(
            matching_estimate and estimate.velocity_world_mps is not None
        )
        if position_available:
            assert estimate is not None and estimate.position_world_m is not None
            target_position = np.asarray(
                estimate.position_world_m,
                dtype=np.float64,
            )
            target_velocity = np.asarray(
                estimate.velocity_world_mps
                if estimate.velocity_world_mps is not None
                else (0.0, 0.0, 0.0),
                dtype=np.float64,
            )
            target_visible = bool(estimate.visible and not estimate.predicted_only)
            self._perception_source = estimate.source
            self._tracker_id = estimate.tracker_id
            self._measurement_age_s = estimate.measurement_age_s + max(
                0.0, observation_time - estimate.timestamp_s
            )
            self._predicted_only = estimate.predicted_only
        elif self._last_seen_position is not None:
            target_position = self._last_seen_position.copy()
            target_velocity = (
                np.zeros(3, dtype=np.float64)
                if self._last_seen_velocity is None
                else self._last_seen_velocity.copy()
            )
            target_visible = False
            self._predicted_only = False
            self._measurement_age_s = (
                None
                if self._last_seen_time is None
                else max(0.0, effective_time - self._last_seen_time)
            )
        else:
            self._active_context.uav.stop()
            self._fail(
                SkillResultCode.INVALID_STATE,
                "TRACK has no confirmed three-dimensional target position",
                {
                    "target_id": goal.target_id.strip(),
                    "position_available": False,
                    "perception_source": None if estimate is None else estimate.source,
                },
            )
            return

        # A delayed visible frame may refresh last-seen only if it was captured
        # before the previous grace period expired. This prevents a recovery
        # frame from reviving an already-lost Skill. The same simulation epoch
        # is used for the clock and Observation timestamps.
        previous_seen_anchor = (
            self._start_time
            if self._last_seen_time is None
            else self._last_seen_time
        )
        previous_loss_deadline = (
            previous_seen_anchor + goal.max_target_lost_time
        )
        timeout_deadline = (
            None
            if goal.timeout is None
            else self._start_time + goal.timeout
        )
        completion_deadline = (
            None
            if goal.track_duration is None
            else self._start_time + goal.track_duration
        )
        visible_frame_precedes_loss = (
            observation_time <= previous_loss_deadline + 1e-12
        )
        visible_frame_precedes_timeout = (
            timeout_deadline is None
            or observation_time <= timeout_deadline + 1e-12
        )
        visible_frame_precedes_completion = (
            completion_deadline is None
            or observation_time <= completion_deadline + 1e-12
        )
        if (
            target_visible
            and is_new_frame
            and visible_frame_precedes_loss
            and visible_frame_precedes_timeout
            and visible_frame_precedes_completion
        ):
            assert estimate is not None
            self._last_seen_time = max(
                self._start_time,
                estimate.timestamp_s - estimate.measurement_age_s,
            )
            self._last_seen_position = target_position.copy()
            self._last_seen_velocity = target_velocity.copy()

        seen_anchor = (
            self._start_time
            if self._last_seen_time is None
            else self._last_seen_time
        )
        loss_deadline = seen_anchor + goal.max_target_lost_time
        last_seen_age = max(0.0, effective_time - seen_anchor)
        current_uav_position = np.asarray(
            [
                observation.uav_pose.x,
                observation.uav_pose.y,
                observation.uav_pose.z,
            ],
            dtype=np.float64,
        )
        target_distance = float(
            np.linalg.norm(target_position[:2] - current_uav_position[:2])
        )
        distance_error = target_distance - float(goal.desired_distance)
        target_bearing = atan2(
            target_position[1] - current_uav_position[1],
            target_position[0] - current_uav_position[0],
        )
        relative_bearing = _wrap_angle(target_bearing - observation.uav_pose.yaw)
        feedback_data = {
            "target_distance": target_distance,
            "distance_error": distance_error,
            "target_visible": target_visible,
            "target_relative_bearing": relative_bearing,
            "last_seen_age": last_seen_age,
            "tracking_duration": tracking_duration,
            "perception_source": self._perception_source,
            "tracker_id": self._tracker_id,
            "measurement_age_s": self._measurement_age_s,
            "predicted_only": self._predicted_only,
            "position_available": position_available,
            "velocity_available": velocity_available,
        }
        progress = (
            min(1.0, tracking_duration / goal.track_duration)
            if goal.track_duration is not None
            else (
                None
                if goal.timeout is None
                else min(1.0, tracking_duration / goal.timeout)
            )
        )

        # The wording "exceeds" is deliberate: a loss exactly equal to the
        # configured grace period remains recoverable until the next frame.
        target_lost = effective_time > loss_deadline + 1e-12
        timed_out = (
            timeout_deadline is not None
            and effective_time >= timeout_deadline
        )
        track_complete = (
            completion_deadline is not None
            and effective_time >= completion_deadline
        )

        # Sampling may observe several expired deadlines at once. Resolve them
        # by the time at which they became eligible, rather than by tick order.
        # A completion at the same instant as a failure wins, like reaching a
        # GOTO goal at its timeout boundary. TIMEOUT retains precedence over a
        # same-time loss, whose contract deliberately requires strict `>`.
        terminal_events: list[tuple[float, int, SkillResultCode]] = []
        if track_complete and completion_deadline is not None:
            terminal_events.append(
                (completion_deadline, 0, SkillResultCode.TRACK_COMPLETE)
            )
        if timed_out and timeout_deadline is not None:
            terminal_events.append((timeout_deadline, 1, SkillResultCode.TIMEOUT))
        if target_lost:
            terminal_events.append((loss_deadline, 2, SkillResultCode.TARGET_LOST))

        terminal_code = min(terminal_events)[2] if terminal_events else None
        if terminal_code is SkillResultCode.TRACK_COMPLETE:
            self._set_feedback(1.0, "TRACK duration complete", feedback_data)
            self._succeed(
                SkillResultCode.TRACK_COMPLETE,
                "TRACK completed its requested duration",
                self._last_seen_result_data(goal, last_seen_age, tracking_duration),
            )
            return

        if terminal_code is SkillResultCode.TARGET_LOST:
            self._set_feedback(progress, "TRACK target lost", feedback_data)
            self._fail(
                SkillResultCode.TARGET_LOST,
                "TRACK target remained outside the Camera FOV too long",
                self._last_seen_result_data(goal, last_seen_age, tracking_duration),
            )
            return

        if terminal_code is SkillResultCode.TIMEOUT:
            self._set_feedback(
                1.0,
                "TRACK timed out",
                feedback_data,
            )
            timeout_data = self._last_seen_result_data(
                goal,
                last_seen_age,
                tracking_duration,
            )
            self._fail(
                SkillResultCode.TIMEOUT,
                "TRACK reached its configured timeout",
                timeout_data,
            )
            return

        if self._stand_off_direction_xy is None:
            self._stand_off_direction_xy = self._initial_stand_off_direction(
                current_uav_position,
                target_position,
                target_velocity,
                0.0,
            )
        desired_position = np.asarray(
            [
                target_position[0]
                + self._stand_off_direction_xy[0] * goal.desired_distance,
                target_position[1]
                + self._stand_off_direction_xy[1] * goal.desired_distance,
                goal.desired_altitude,
            ],
            dtype=np.float64,
        )
        policy = MotionPolicy(
            max_speed=goal.max_speed,
            yaw_mode=YawMode.FACE_POINT,
            look_at_point=tuple(float(value) for value in target_position),
        )
        commanded_velocity = move_toward_with_policy(
            self._active_context.uav,
            desired_position,
            goal.max_speed,
            self.POSITION_TOLERANCE_M,
            policy,
            initial_yaw=self.initial_yaw,
        )
        feedback_data["desired_position"] = tuple(
            float(value) for value in desired_position
        )
        feedback_data["commanded_speed"] = float(
            np.linalg.norm(commanded_velocity)
        )
        self._set_feedback(
            progress,
            "Tracking confirmed target estimate",
            feedback_data,
        )

    def _last_seen_result_data(
        self,
        goal: TrackGoal,
        last_seen_age: float,
        tracking_duration: float,
    ) -> dict[str, object]:
        return {
            "target_id": goal.target_id.strip(),
            "last_seen_position": (
                ()
                if self._last_seen_position is None
                else tuple(float(value) for value in self._last_seen_position)
            ),
            "last_seen_velocity": (
                ()
                if self._last_seen_velocity is None
                else tuple(float(value) for value in self._last_seen_velocity)
            ),
            "last_seen_time": self._last_seen_time,
            "last_seen_age": last_seen_age,
            "tracking_duration": tracking_duration,
            "track_duration": goal.track_duration,
            "perception_source": self._perception_source,
            "tracker_id": self._tracker_id,
            "measurement_age_s": self._measurement_age_s,
            "predicted_only": self._predicted_only,
            "position_available": self._last_seen_position is not None,
            "velocity_available": self._last_seen_velocity is not None,
        }

    @staticmethod
    def _initial_stand_off_direction(
        uav_position: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        target_yaw: float,
    ) -> np.ndarray:
        direction = uav_position[:2] - target_position[:2]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            return direction / norm
        velocity_norm = float(np.linalg.norm(target_velocity[:2]))
        if velocity_norm > 1e-12:
            return -target_velocity[:2] / velocity_norm
        return np.asarray([-cos(target_yaw), -sin(target_yaw)], dtype=np.float64)

    def _clear_last_seen(self) -> None:
        self._last_seen_time = None
        self._last_seen_position = None
        self._last_seen_velocity = None
        self._perception_source = None
        self._tracker_id = None
        self._measurement_age_s = None
        self._predicted_only = False

    def _on_reset(self) -> None:
        self._start_time = None
        self._last_clock_time = None
        self._last_observation_timestamp = None
        self._stand_off_direction_xy = None
        self._clear_last_seen()

    @staticmethod
    def _track_goal(goal: SkillGoal) -> TrackGoal:
        if not isinstance(goal, TrackGoal):
            raise SkillExecutionStateError("active TRACK goal has an invalid type")
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


def _wrap_angle(angle: float) -> float:
    return atan2(sin(angle), cos(angle))
