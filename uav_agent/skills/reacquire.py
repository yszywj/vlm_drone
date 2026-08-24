"""Ideal REACQUIRE Skill using frozen constant-velocity prediction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import atan2, cos, isfinite, pi, sin
from numbers import Real

import numpy as np

from skills.base import (
    Skill,
    SkillExecutionStateError,
    SkillGoalValidationError,
    require_finite,
    require_positive,
    require_vector3,
)
from skills.motion_types import MotionPolicy, YawMode, move_toward_with_policy
from skills.types import Observation, SkillContext, SkillGoal, SkillResultCode


@dataclass(frozen=True, slots=True)
class ReacquireGoal(SkillGoal):
    """Last-seen target state used for one deterministic local recovery."""

    target_id: str
    last_seen_position: tuple[float, float, float]
    last_seen_velocity: tuple[float, float, float]
    last_seen_time: float
    search_radius: float = 10.0
    timeout: float = 30.0


class _ReacquirePhase(Enum):
    TRANSITING = auto()
    SCANNING = auto()


class ReacquireSkill(Skill):
    """Approach a predicted region and scan without invoking TRACK or SEARCH."""

    goal_type = ReacquireGoal
    FULL_SCAN_RAD = 2.0 * pi
    SCAN_YAW_RATE_RAD_S = 0.5

    def __init__(self) -> None:
        super().__init__()
        self._predicted_position: np.ndarray | None = None
        self._transit_goal: np.ndarray | None = None
        self._transit_policy: MotionPolicy | None = None
        self._prediction_dt = 0.0
        self._phase: _ReacquirePhase | None = None
        self._start_time: float | None = None
        self._last_clock_time: float | None = None
        self._last_observation_timestamp: float | None = None
        self._scan_angle_rad = 0.0
        self._scan_last_yaw: float | None = None
        self._scan_last_timestamp: float | None = None
        self._effective_scan_rate_rad_s: float | None = None
        self._completed_scans = 0

    def _validate_goal(self, goal: SkillGoal) -> None:
        typed_goal = goal
        if not isinstance(typed_goal, ReacquireGoal):
            return
        if not isinstance(typed_goal.target_id, str) or not typed_goal.target_id.strip():
            raise SkillGoalValidationError("target_id must be a non-empty string")
        require_vector3(typed_goal.last_seen_position, "last_seen_position")
        require_vector3(typed_goal.last_seen_velocity, "last_seen_velocity")
        last_seen_time = require_finite(typed_goal.last_seen_time, "last_seen_time")
        if last_seen_time < 0.0:
            raise SkillGoalValidationError("last_seen_time must be non-negative")
        require_positive(typed_goal.search_radius, "search_radius")
        require_positive(typed_goal.timeout, "timeout")

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        typed_goal = self._reacquire_goal(goal)
        start_time = self._read_clock(context)
        if typed_goal.last_seen_time > start_time + 1e-12:
            raise SkillExecutionStateError(
                "last_seen_time is later than the REACQUIRE start clock"
            )
        prediction_dt = max(0.0, start_time - float(typed_goal.last_seen_time))
        last_position = np.asarray(
            require_vector3(typed_goal.last_seen_position, "last_seen_position"),
            dtype=np.float64,
        )
        last_velocity = np.asarray(
            require_vector3(typed_goal.last_seen_velocity, "last_seen_velocity"),
            dtype=np.float64,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            predicted_position = last_position + last_velocity * prediction_dt
        if not np.all(np.isfinite(predicted_position)):
            raise SkillExecutionStateError(
                "constant-velocity prediction produced a non-finite position"
            )

        start_pose = context.uav.get_pose()
        transit_goal = np.asarray(
            [predicted_position[0], predicted_position[1], start_pose.z],
            dtype=np.float64,
        )
        transit_policy = MotionPolicy(
            max_speed=context.uav.max_speed_mps,
            yaw_mode=YawMode.FACE_POINT,
            look_at_point=tuple(float(value) for value in predicted_position),
        )

        context.uav.stop()
        self._predicted_position = predicted_position
        self._transit_goal = transit_goal
        self._transit_policy = transit_policy
        self._prediction_dt = prediction_dt
        self._phase = _ReacquirePhase.TRANSITING
        self._start_time = start_time
        self._last_clock_time = start_time
        self._last_observation_timestamp = None
        self._completed_scans = 0
        self._clear_scan_state()
        self._set_feedback(
            0.0,
            "REACQUIRE initialized",
            self._feedback_data(target_visible=False, elapsed=0.0),
        )

    def _on_tick(self, observation: Observation) -> None:
        goal = self._reacquire_goal(self._active_goal)
        if (
            self._predicted_position is None
            or self._transit_goal is None
            or self._transit_policy is None
            or self._phase is None
            or self._start_time is None
        ):
            raise SkillExecutionStateError("REACQUIRE was not initialized")

        now = self._read_clock(self._active_context)
        if self._last_clock_time is not None and now < self._last_clock_time - 1e-12:
            raise SkillExecutionStateError(
                "Skill clock moved backwards during REACQUIRE"
            )
        self._last_clock_time = now
        if (
            self._last_observation_timestamp is not None
            and observation.timestamp < self._last_observation_timestamp - 1e-12
        ):
            raise SkillExecutionStateError(
                "Observation timestamp moved backwards during REACQUIRE"
            )
        self._last_observation_timestamp = float(observation.timestamp)
        frame_elapsed = float(observation.timestamp) - self._start_time
        if frame_elapsed < -1e-12:
            raise SkillExecutionStateError(
                "REACQUIRE received an Observation captured before Skill start"
            )
        frame_elapsed = max(0.0, frame_elapsed)
        elapsed = max(max(0.0, now - self._start_time), frame_elapsed)

        requested_target_visible = self._requested_target_visible(observation, goal)
        if requested_target_visible and frame_elapsed <= goal.timeout + 1e-12:
            self._complete_target_found(observation, elapsed)
            return

        if elapsed >= goal.timeout:
            feedback = self._feedback_data(target_visible=False, elapsed=elapsed)
            self._set_feedback(1.0, "REACQUIRE timed out", feedback)
            self._fail(
                SkillResultCode.TIMEOUT,
                "REACQUIRE timed out before finding the target",
                {
                    "target_id": goal.target_id.strip(),
                    "predicted_position": tuple(
                        float(value) for value in self._predicted_position
                    ),
                    "prediction_dt": self._prediction_dt,
                    "phase": self._phase.name,
                    "completed_scans": self._completed_scans,
                    "elapsed_time": elapsed,
                },
            )
            return

        if self._phase is _ReacquirePhase.TRANSITING:
            self._tick_transit(observation, goal, elapsed)
        elif self._phase is _ReacquirePhase.SCANNING:
            self._tick_scan(observation, elapsed)
        else:
            raise SkillExecutionStateError("REACQUIRE has an invalid internal phase")

    def _requested_target_visible(
        self,
        observation: Observation,
        goal: ReacquireGoal,
    ) -> bool:
        estimate = observation.target_estimate
        if (
            estimate is None
            or not estimate.visible
            or estimate.predicted_only
            or not estimate.confirmed
            or estimate.position_world_m is None
        ):
            return False
        return estimate.target_id == goal.target_id.strip()

    def _tick_transit(
        self,
        observation: Observation,
        goal: ReacquireGoal,
        elapsed: float,
    ) -> None:
        current = _uav_position(observation)
        horizontal_distance = float(
            np.linalg.norm(self._predicted_position[:2] - current[:2])
        )
        if horizontal_distance <= goal.search_radius:
            self._begin_scan(observation)
            self._set_feedback(
                min(1.0, elapsed / goal.timeout),
                "Scanning predicted target region",
                self._feedback_data(
                    target_visible=False,
                    elapsed=elapsed,
                    horizontal_distance=horizontal_distance,
                ),
            )
            return

        move_toward_with_policy(
            self._active_context.uav,
            self._transit_goal,
            self._active_context.uav.max_speed_mps,
            goal.search_radius,
            self._transit_policy,
            initial_yaw=self.initial_yaw,
        )
        self._set_feedback(
            min(1.0, elapsed / goal.timeout),
            "Moving to predicted target region",
            self._feedback_data(
                target_visible=False,
                elapsed=elapsed,
                horizontal_distance=horizontal_distance,
            ),
        )

    def _begin_scan(self, observation: Observation) -> None:
        self._active_context.uav.stop()
        self._phase = _ReacquirePhase.SCANNING
        self._scan_angle_rad = 0.0
        self._scan_last_yaw = float(observation.uav_pose.yaw)
        self._scan_last_timestamp = float(observation.timestamp)
        self._effective_scan_rate_rad_s = min(
            self.SCAN_YAW_RATE_RAD_S,
            self._active_context.uav.max_yaw_rate_rad_s,
        )
        self._command_scan()

    def _tick_scan(self, observation: Observation, elapsed: float) -> None:
        if (
            self._scan_last_yaw is None
            or self._scan_last_timestamp is None
            or self._effective_scan_rate_rad_s is None
        ):
            raise SkillExecutionStateError("REACQUIRE yaw scan was not initialized")

        sample_dt = float(observation.timestamp) - self._scan_last_timestamp
        if sample_dt < -1e-12:
            raise SkillExecutionStateError(
                "REACQUIRE scan timestamp moved backwards"
            )
        sample_dt = max(0.0, sample_dt)
        observed_delta = _wrap_angle(
            float(observation.uav_pose.yaw) - self._scan_last_yaw
        )
        commanded_delta = self._effective_scan_rate_rad_s * sample_dt
        residual = _wrap_angle(observed_delta - _wrap_angle(commanded_delta))
        if abs(residual) > 1e-6:
            raise SkillExecutionStateError(
                "REACQUIRE yaw observation does not match the commanded scan rate"
            )

        self._scan_angle_rad += commanded_delta
        self._scan_last_yaw = float(observation.uav_pose.yaw)
        self._scan_last_timestamp = float(observation.timestamp)
        while self._scan_angle_rad + 1e-9 >= self.FULL_SCAN_RAD:
            self._scan_angle_rad = max(
                0.0,
                self._scan_angle_rad - self.FULL_SCAN_RAD,
            )
            self._completed_scans += 1

        self._command_scan()
        goal = self._reacquire_goal(self._active_goal)
        self._set_feedback(
            min(1.0, elapsed / goal.timeout),
            "Scanning predicted target region",
            self._feedback_data(target_visible=False, elapsed=elapsed),
        )

    def _command_scan(self) -> None:
        self._active_context.uav.set_velocity(
            (0.0, 0.0, 0.0),
            yaw_rate_rad_s=self.SCAN_YAW_RATE_RAD_S,
        )

    def _complete_target_found(
        self,
        observation: Observation,
        elapsed: float,
    ) -> None:
        goal = self._reacquire_goal(self._active_goal)
        estimate = observation.target_estimate
        if (
            estimate is None
            or not estimate.visible
            or not estimate.confirmed
            or estimate.target_id != goal.target_id.strip()
            or estimate.position_world_m is None
        ):
            raise SkillExecutionStateError(
                "visible requested target is missing a confirmed 3D TargetEstimate"
            )
        if (
            observation.camera_position_m is None
            or observation.camera_orientation_wxyz is None
        ):
            raise SkillExecutionStateError(
                "visible target frame is missing synchronized Camera pose"
            )

        data: dict[str, object] = {
            "target_id": goal.target_id.strip(),
            "found_timestamp": float(observation.timestamp),
            "predicted_position": tuple(
                float(value) for value in self._predicted_position
            ),
            "prediction_dt": self._prediction_dt,
            "uav_pose": _uav_pose_dict(observation),
            "camera_pose": {
                "position_m": tuple(
                    float(value) for value in observation.camera_position_m
                ),
                "orientation_wxyz": tuple(
                    float(value) for value in observation.camera_orientation_wxyz
                ),
            },
            "target_position_world_m": estimate.position_world_m,
            "target_velocity_world_mps": estimate.velocity_world_mps,
            "perception_source": estimate.source,
            "tracker_id": estimate.tracker_id,
            "candidate_id": estimate.candidate_id,
            "measurement_age_s": estimate.measurement_age_s,
            "completed_scans": self._completed_scans,
            "elapsed_time": elapsed,
        }
        if estimate.source == "oracle_evaluation":
            data["oracle_target_pose"] = {
                "x": estimate.position_world_m[0],
                "y": estimate.position_world_m[1],
                "z": estimate.position_world_m[2],
                "yaw": 0.0,
            }
            if estimate.velocity_world_mps is not None:
                data["oracle_target_velocity_mps"] = estimate.velocity_world_mps
        self._set_feedback(
            min(1.0, elapsed / goal.timeout),
            "Requested confirmed target visible in Camera FOV",
            self._feedback_data(target_visible=True, elapsed=elapsed),
        )
        self._succeed(
            SkillResultCode.TARGET_FOUND,
            "REACQUIRE found the requested target",
            data,
        )

    def _feedback_data(
        self,
        *,
        target_visible: bool,
        elapsed: float,
        horizontal_distance: float | None = None,
    ) -> dict[str, object]:
        data: dict[str, object] = {
            "phase": self._phase.name if self._phase is not None else "UNINITIALIZED",
            "predicted_position": (
                None
                if self._predicted_position is None
                else tuple(float(value) for value in self._predicted_position)
            ),
            "prediction_dt": self._prediction_dt,
            "target_visible": target_visible,
            "completed_scans": self._completed_scans,
            "scan_angle_rad": self._scan_angle_rad,
            "scan_target_rad": self.FULL_SCAN_RAD,
            "elapsed_time": elapsed,
        }
        if horizontal_distance is not None:
            data["horizontal_distance_to_prediction"] = horizontal_distance
            goal = self._reacquire_goal(self._active_goal)
            data["distance_to_search_region"] = max(
                0.0,
                horizontal_distance - goal.search_radius,
            )
        return data

    def _clear_scan_state(self) -> None:
        self._scan_angle_rad = 0.0
        self._scan_last_yaw = None
        self._scan_last_timestamp = None
        self._effective_scan_rate_rad_s = None

    def _on_reset(self) -> None:
        self._predicted_position = None
        self._transit_goal = None
        self._transit_policy = None
        self._prediction_dt = 0.0
        self._phase = None
        self._start_time = None
        self._last_clock_time = None
        self._last_observation_timestamp = None
        self._completed_scans = 0
        self._clear_scan_state()

    @staticmethod
    def _reacquire_goal(goal: SkillGoal) -> ReacquireGoal:
        if not isinstance(goal, ReacquireGoal):
            raise SkillExecutionStateError(
                "active REACQUIRE goal has an invalid type"
            )
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


def _uav_position(observation: Observation) -> np.ndarray:
    pose = observation.uav_pose
    return np.asarray([pose.x, pose.y, pose.z], dtype=np.float64)


def _uav_pose_dict(observation: Observation) -> dict[str, float]:
    pose = observation.uav_pose
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "yaw": float(pose.yaw),
    }


def _wrap_angle(angle: float) -> float:
    return atan2(sin(angle), cos(angle))
