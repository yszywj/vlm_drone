"""Deterministic waypoint SEARCH with active, continuous yaw scans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, isfinite, pi, radians, sin
from numbers import Real

import numpy as np

from env.moving_target import TargetState
from skills.base import (
    Skill,
    SkillExecutionStateError,
    SkillGoalValidationError,
    require_positive,
    require_vector3,
)
from skills.motion_types import MotionPolicy, YawMode, move_toward_with_policy
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillResultCode,
    SkillStatus,
)

_SEARCH_WAYPOINT_ANGLES_DEG: tuple[float, ...] = (
    30.0,
    90.0,
    150.0,
    210.0,
    270.0,
    330.0,
)


def build_search_waypoints(
    center_xyz_m: tuple[float, float, float],
    radius_m: float,
    altitude_m: float,
) -> tuple[tuple[float, float, float], ...]:
    """Build the deterministic six-point SEARCH perimeter route."""

    center = require_vector3(center_xyz_m, "center_xyz_m")
    radius = require_positive(radius_m, "radius_m")
    altitude = require_positive(altitude_m, "altitude_m")

    return tuple(
        (
            float(center[0] + radius * cos(radians(angle_deg))),
            float(center[1] + radius * sin(radians(angle_deg))),
            float(altitude),
        )
        for angle_deg in _SEARCH_WAYPOINT_ANGLES_DEG
    )

@dataclass(frozen=True, slots=True)
class SearchGoal(SkillGoal):
    """Known-region SEARCH request; lengths are metres and time is seconds."""

    center: tuple[float, float, float]
    radius: float
    target_description: str
    search_altitude: float
    transit_speed: float = 1.5
    scan_yaw_rate: float = 0.5
    timeout: float = 60.0


class SearchPhase(str, Enum):
    """Observable SEARCH phase without embedding another Skill lifecycle."""

    TRANSIT = "TRANSIT"
    SCANNING = "SCANNING"
    CANDIDATE_PENDING = "CANDIDATE_PENDING"
    WAITING_FOR_REVIEW = "WAITING_FOR_REVIEW"
    TARGET_LOCKED = "TARGET_LOCKED"


class SearchSkill(Skill):
    """Search six fixed perimeter points without reading hidden target position."""

    goal_type = SearchGoal
    WAYPOINT_COUNT = len(_SEARCH_WAYPOINT_ANGLES_DEG)
    FULL_SCAN_RAD = 2.0 * pi

    def __init__(
        self,
        transit_yaw_mode: YawMode | str = YawMode.FACE_POINT,
    ) -> None:
        super().__init__()
        self._transit_yaw_mode = _transit_mode(transit_yaw_mode)
        self._center: np.ndarray | None = None
        self._waypoints: tuple[np.ndarray, ...] = ()
        self._waypoint_index = 0
        self._waypoint_tolerance_m = 0.25
        self._phase: SearchPhase | None = None
        self._reported_phase: SearchPhase | None = None
        self._candidate_id: str | None = None
        self._candidate_source: str | None = None
        self._transit_policy: MotionPolicy | None = None
        self._start_time: float | None = None
        self._last_clock_time: float | None = None
        self._last_observation_timestamp: float | None = None
        self._scan_accumulated_rad = 0.0
        self._scan_last_yaw: float | None = None
        self._scan_last_timestamp: float | None = None
        self._effective_scan_rate_rad_s: float | None = None
        self._completed_scan_angles_rad: list[float] = []

    @property
    def transit_yaw_mode(self) -> YawMode:
        return self._transit_yaw_mode

    @property
    def phase(self) -> SearchPhase | None:
        return self._reported_phase

    def report_candidate_pending(self, candidate_id: str, *, source: str) -> None:
        """Expose a provisional candidate without declaring target success.

        Candidate confirmation, HOVER and INSPECT are orchestration concerns;
        this method deliberately does not call another Skill or fabricate an
        identity result.
        """

        if self.status is not SkillStatus.RUNNING:
            raise SkillExecutionStateError(
                "candidate reporting requires a running SEARCH"
            )
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("candidate source must be a non-empty string")
        self._candidate_id = candidate_id.strip()
        self._candidate_source = source.strip()
        self._reported_phase = SearchPhase.CANDIDATE_PENDING
        self._refresh_candidate_feedback("SEARCH candidate pending review")

    def mark_waiting_for_review(self, candidate_id: str) -> None:
        if self.status is not SkillStatus.RUNNING:
            raise SkillExecutionStateError(
                "review waiting requires a running SEARCH"
            )
        if self._candidate_id != candidate_id:
            raise ValueError("candidate_id does not match pending SEARCH candidate")
        self._reported_phase = SearchPhase.WAITING_FOR_REVIEW
        self._refresh_candidate_feedback("SEARCH waiting for visual review")

    def reject_candidate(self, candidate_id: str) -> None:
        if self.status is not SkillStatus.RUNNING:
            raise SkillExecutionStateError(
                "candidate rejection requires a running SEARCH"
            )
        if self._candidate_id != candidate_id:
            raise ValueError("candidate_id does not match pending SEARCH candidate")
        self._candidate_id = None
        self._candidate_source = None
        self._reported_phase = self._phase
        self._refresh_candidate_feedback("SEARCH candidate rejected")

    def _refresh_candidate_feedback(self, message: str) -> None:
        previous = self.get_feedback()
        data = dict(previous.data)
        data["phase"] = (
            self._reported_phase.value
            if self._reported_phase is not None
            else "UNINITIALIZED"
        )
        if self._candidate_id is None:
            data.pop("candidate_id", None)
            data.pop("candidate_source", None)
        else:
            data["candidate_id"] = self._candidate_id
            data["candidate_source"] = self._candidate_source
        self._set_feedback(previous.progress, message, data)

    def _validate_goal(self, goal: SkillGoal) -> None:
        typed_goal = goal
        if not isinstance(typed_goal, SearchGoal):
            return
        require_vector3(typed_goal.center, "center")
        require_positive(typed_goal.radius, "radius")
        if (
            not isinstance(typed_goal.target_description, str)
            or not typed_goal.target_description.strip()
        ):
            raise SkillGoalValidationError(
                "target_description must be a non-empty string"
            )
        require_positive(typed_goal.search_altitude, "search_altitude")
        require_positive(typed_goal.transit_speed, "transit_speed")
        require_positive(typed_goal.scan_yaw_rate, "scan_yaw_rate")
        require_positive(typed_goal.timeout, "timeout")

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        typed_goal = self._search_goal(goal)
        center = np.asarray(
            require_vector3(typed_goal.center, "center"),
            dtype=np.float64,
        )
        start_time = self._read_clock(context)
        waypoints = tuple(
            np.asarray(point, dtype=np.float64)
            for point in build_search_waypoints(
                tuple(float(value) for value in center),
                typed_goal.radius,
                typed_goal.search_altitude,
            )
        )
        look_at_point = (
            tuple(float(value) for value in center)
            if self._transit_yaw_mode is YawMode.FACE_POINT
            else None
        )

        # Waypoints and transit policy depend only on the requested region.
        # No Oracle pose is available or consulted during initialization.
        context.uav.stop()
        self._center = center
        self._waypoints = waypoints
        self._waypoint_index = 0
        self._waypoint_tolerance_m = min(
            0.5,
            max(0.05, 0.1 * float(typed_goal.radius)),
        )
        self._phase = SearchPhase.TRANSIT
        self._reported_phase = SearchPhase.TRANSIT
        self._candidate_id = None
        self._candidate_source = None
        self._transit_policy = MotionPolicy(
            max_speed=typed_goal.transit_speed,
            yaw_mode=self._transit_yaw_mode,
            look_at_point=look_at_point,
        )
        self._start_time = start_time
        self._last_clock_time = start_time
        self._last_observation_timestamp = None
        self._completed_scan_angles_rad = []
        self._clear_scan_state()
        self._set_feedback(
            0.0,
            "SEARCH initialized",
            {
                "phase": self._phase.name,
                "waypoint_index": 1,
                "waypoint_count": len(self._waypoints),
                "target_visible": False,
            },
        )

    def _on_tick(self, observation: Observation) -> None:
        goal = self._search_goal(self._active_goal)
        if (
            self._center is None
            or not self._waypoints
            or self._phase is None
            or self._transit_policy is None
            or self._start_time is None
        ):
            raise SkillExecutionStateError("SEARCH was not initialized")

        now = self._read_clock(self._active_context)
        if self._last_clock_time is not None and now < self._last_clock_time - 1e-12:
            raise SkillExecutionStateError("Skill clock moved backwards during SEARCH")
        self._last_clock_time = now
        if (
            self._last_observation_timestamp is not None
            and observation.timestamp < self._last_observation_timestamp - 1e-12
        ):
            raise SkillExecutionStateError(
                "Observation timestamp moved backwards during SEARCH"
            )
        self._last_observation_timestamp = float(observation.timestamp)
        clock_elapsed = max(0.0, now - self._start_time)
        frame_elapsed = float(observation.timestamp) - self._start_time
        if frame_elapsed < -1e-12:
            raise SkillExecutionStateError(
                "SEARCH received an Observation captured before Skill start"
            )
        frame_elapsed = max(0.0, frame_elapsed)
        elapsed = max(clock_elapsed, frame_elapsed)

        if observation.oracle_target_visible is None:
            raise SkillExecutionStateError(
                "SEARCH requires oracle_target_visible for ideal detection"
            )
        if (
            observation.oracle_target_visible
            and frame_elapsed <= goal.timeout + 1e-12
        ):
            self._complete_target_found(observation, elapsed)
            return

        # Visibility from the deadline frame has priority over timeout.
        if elapsed >= goal.timeout:
            self._set_feedback(
                self._overall_progress(),
                "SEARCH timed out",
                self._feedback_data(target_visible=False, elapsed=elapsed),
            )
            self._fail(
                SkillResultCode.TIMEOUT,
                "SEARCH timed out before finding the target",
                {
                    "target_description": goal.target_description.strip(),
                    "completed_scans": len(self._completed_scan_angles_rad),
                    "elapsed_time": elapsed,
                },
            )
            return

        if self._phase is SearchPhase.TRANSIT:
            self._tick_transit(observation, goal, elapsed)
        elif self._phase is SearchPhase.SCANNING:
            self._tick_scan(observation, goal, elapsed)
        else:
            raise SkillExecutionStateError("SEARCH has an invalid internal phase")

    def _tick_transit(
        self,
        observation: Observation,
        goal: SearchGoal,
        elapsed: float,
    ) -> None:
        waypoint = self._current_waypoint()
        current = _uav_position(observation)
        distance = float(np.linalg.norm(waypoint - current))
        if distance <= self._waypoint_tolerance_m:
            self._begin_scan(observation, goal)
            self._set_feedback(
                self._overall_progress(),
                "Scanning current waypoint",
                self._feedback_data(
                    target_visible=False,
                    elapsed=elapsed,
                    distance_to_waypoint=distance,
                ),
            )
            return

        move_toward_with_policy(
            self._active_context.uav,
            waypoint,
            goal.transit_speed,
            self._waypoint_tolerance_m,
            self._transit_policy,
            initial_yaw=self.initial_yaw,
        )
        self._set_feedback(
            self._overall_progress(),
            "Moving to search waypoint",
            self._feedback_data(
                target_visible=False,
                elapsed=elapsed,
                distance_to_waypoint=distance,
            ),
        )

    def _begin_scan(self, observation: Observation, goal: SearchGoal) -> None:
        self._active_context.uav.stop()
        self._phase = SearchPhase.SCANNING
        if self._reported_phase not in {
            SearchPhase.CANDIDATE_PENDING,
            SearchPhase.WAITING_FOR_REVIEW,
        }:
            self._reported_phase = SearchPhase.SCANNING
        self._scan_accumulated_rad = 0.0
        self._scan_last_yaw = float(observation.uav_pose.yaw)
        self._scan_last_timestamp = float(observation.timestamp)
        self._effective_scan_rate_rad_s = min(
            float(goal.scan_yaw_rate),
            self._active_context.uav.max_yaw_rate_rad_s,
        )
        self._command_scan(goal)

    def _tick_scan(
        self,
        observation: Observation,
        goal: SearchGoal,
        elapsed: float,
    ) -> None:
        if (
            self._scan_last_yaw is None
            or self._scan_last_timestamp is None
            or self._effective_scan_rate_rad_s is None
        ):
            raise SkillExecutionStateError("SEARCH yaw scan was not initialized")

        sample_dt = float(observation.timestamp) - self._scan_last_timestamp
        if sample_dt < -1e-12:
            raise SkillExecutionStateError("SEARCH scan timestamp moved backwards")
        sample_dt = max(0.0, sample_dt)
        observed_wrapped_delta = _wrap_angle(
            float(observation.uav_pose.yaw) - self._scan_last_yaw
        )
        # The kinematic controller executes a known constant positive yaw rate.
        # Integrating that command remains unambiguous even when a slow Camera
        # samples after crossing ±pi or after more than one revolution. The
        # wrapped observation still verifies that no external command changed
        # the executed yaw between samples.
        commanded_delta = self._effective_scan_rate_rad_s * sample_dt
        yaw_residual = _wrap_angle(
            observed_wrapped_delta - _wrap_angle(commanded_delta)
        )
        if abs(yaw_residual) > 1e-6:
            raise SkillExecutionStateError(
                "SEARCH yaw observation does not match the commanded scan rate"
            )

        self._scan_accumulated_rad += commanded_delta
        self._scan_last_yaw = float(observation.uav_pose.yaw)
        self._scan_last_timestamp = float(observation.timestamp)

        if self._scan_accumulated_rad + 1e-9 >= self.FULL_SCAN_RAD:
            self._active_context.uav.stop()
            self._completed_scan_angles_rad.append(self._scan_accumulated_rad)
            self._waypoint_index += 1
            if self._waypoint_index >= len(self._waypoints):
                self._set_feedback(
                    1.0,
                    "All search waypoints exhausted",
                    {
                        "phase": SearchPhase.SCANNING.value,
                        "waypoint_index": len(self._waypoints),
                        "waypoint_count": len(self._waypoints),
                        "scan_angle_rad": self._scan_accumulated_rad,
                        "target_visible": False,
                        "elapsed_time": elapsed,
                    },
                )
                self._fail(
                    SkillResultCode.SEARCH_EXHAUSTED,
                    "SEARCH completed every waypoint without finding the target",
                    {
                        "target_description": goal.target_description.strip(),
                        "completed_scans": len(self._completed_scan_angles_rad),
                        "scan_angles_rad": tuple(self._completed_scan_angles_rad),
                        "elapsed_time": elapsed,
                    },
                )
                return

            self._phase = SearchPhase.TRANSIT
            if self._reported_phase not in {
                SearchPhase.CANDIDATE_PENDING,
                SearchPhase.WAITING_FOR_REVIEW,
            }:
                self._reported_phase = SearchPhase.TRANSIT
            self._clear_scan_state()
            self._set_feedback(
                self._overall_progress(),
                "Waypoint scan complete",
                self._feedback_data(target_visible=False, elapsed=elapsed),
            )
            return

        self._command_scan(goal)
        self._set_feedback(
            self._overall_progress(),
            "Scanning current waypoint",
            self._feedback_data(target_visible=False, elapsed=elapsed),
        )

    def _command_scan(self, goal: SearchGoal) -> None:
        # Constant positive yaw rate produces a continuous scan. set_velocity
        # also guarantees zero translation throughout this phase.
        self._active_context.uav.set_velocity(
            (0.0, 0.0, 0.0),
            yaw_rate_rad_s=goal.scan_yaw_rate,
        )

    def _complete_target_found(
        self,
        observation: Observation,
        elapsed: float,
    ) -> None:
        goal = self._search_goal(self._active_goal)
        target_id = observation.oracle_target_id
        target_pose = observation.oracle_target_pose
        if not isinstance(target_id, str) or not target_id.strip():
            raise SkillExecutionStateError(
                "visible Oracle target is missing oracle_target_id"
            )
        if not isinstance(target_pose, TargetState):
            raise SkillExecutionStateError(
                "visible Oracle target is missing oracle_target_pose"
            )
        if (
            observation.camera_position_m is None
            or observation.camera_orientation_wxyz is None
        ):
            raise SkillExecutionStateError(
                "visible target frame is missing synchronized Camera pose"
            )

        data: dict[str, object] = {
            "target_id": target_id.strip(),
            "target_description": goal.target_description.strip(),
            "found_timestamp": float(observation.timestamp),
            "uav_pose": _uav_pose_dict(observation),
            "camera_pose": {
                "position_m": tuple(
                    float(value) for value in observation.camera_position_m
                ),
                "orientation_wxyz": tuple(
                    float(value) for value in observation.camera_orientation_wxyz
                ),
            },
            "oracle_target_pose": _target_pose_dict(target_pose),
            "elapsed_time": elapsed,
        }
        if observation.oracle_target_velocity is not None:
            data["oracle_target_velocity_mps"] = tuple(
                float(value) for value in observation.oracle_target_velocity
            )
        self._reported_phase = SearchPhase.TARGET_LOCKED
        self._set_feedback(
            self._overall_progress(),
            "Target visible in Camera FOV",
            self._feedback_data(target_visible=True, elapsed=elapsed),
        )
        self._succeed(
            SkillResultCode.TARGET_FOUND,
            "SEARCH found the requested target",
            data,
        )

    def _feedback_data(
        self,
        *,
        target_visible: bool,
        elapsed: float,
        distance_to_waypoint: float | None = None,
    ) -> dict[str, object]:
        data: dict[str, object] = {
            "phase": (
                self._reported_phase.value
                if self._reported_phase is not None
                else "UNINITIALIZED"
            ),
            "waypoint_index": min(self._waypoint_index + 1, len(self._waypoints)),
            "waypoint_count": len(self._waypoints),
            "target_visible": target_visible,
            "elapsed_time": elapsed,
        }
        if distance_to_waypoint is not None:
            data["distance_to_waypoint"] = float(distance_to_waypoint)
        if self._candidate_id is not None:
            data["candidate_id"] = self._candidate_id
            data["candidate_source"] = self._candidate_source
        if self._phase is SearchPhase.SCANNING:
            data["scan_angle_rad"] = self._scan_accumulated_rad
            data["scan_target_rad"] = self.FULL_SCAN_RAD
        return data

    def _overall_progress(self) -> float:
        if not self._waypoints:
            return 0.0
        scan_fraction = (
            min(1.0, self._scan_accumulated_rad / self.FULL_SCAN_RAD)
            if self._phase is SearchPhase.SCANNING
            else 0.0
        )
        return min(
            1.0,
            max(
                0.0,
                (self._waypoint_index + scan_fraction) / len(self._waypoints),
            ),
        )

    def _current_waypoint(self) -> np.ndarray:
        if not 0 <= self._waypoint_index < len(self._waypoints):
            raise SkillExecutionStateError("SEARCH waypoint index is out of range")
        return self._waypoints[self._waypoint_index]

    def _clear_scan_state(self) -> None:
        self._scan_accumulated_rad = 0.0
        self._scan_last_yaw = None
        self._scan_last_timestamp = None
        self._effective_scan_rate_rad_s = None

    def _on_reset(self) -> None:
        self._center = None
        self._waypoints = ()
        self._waypoint_index = 0
        self._waypoint_tolerance_m = 0.25
        self._phase = None
        self._reported_phase = None
        self._candidate_id = None
        self._candidate_source = None
        self._transit_policy = None
        self._start_time = None
        self._last_clock_time = None
        self._last_observation_timestamp = None
        self._completed_scan_angles_rad = []
        self._clear_scan_state()

    @staticmethod
    def _search_goal(goal: SkillGoal) -> SearchGoal:
        if not isinstance(goal, SearchGoal):
            raise SkillExecutionStateError("active SEARCH goal has an invalid type")
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


def _transit_mode(value: YawMode | str) -> YawMode:
    if isinstance(value, YawMode):
        mode = value
    elif isinstance(value, str):
        try:
            mode = YawMode[value.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown SEARCH transit yaw mode: {value}") from exc
    else:
        raise TypeError("transit_yaw_mode must be a YawMode or string")
    if mode not in {
        YawMode.FACE_POINT,
        YawMode.COURSE_ALIGNED,
        YawMode.KEEP_CURRENT,
    }:
        raise ValueError(
            "SEARCH transit yaw mode must be FACE_POINT, COURSE_ALIGNED, or KEEP_CURRENT"
        )
    return mode


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


def _target_pose_dict(pose: TargetState) -> dict[str, float]:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "yaw": float(pose.yaw),
    }


def _wrap_angle(angle: float) -> float:
    return atan2(sin(angle), cos(angle))
