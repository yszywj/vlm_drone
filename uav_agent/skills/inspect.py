"""Ideal bounded INSPECT behavior over trusted candidate geometry.

The model-visible goal contains no world coordinate.  A separately injected
``CandidateResolver`` resolves the selected ``candidate_id`` behind the trust
boundary, and the Skill emits only bounded ``FrameRef`` evidence metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, isfinite, pi, sin
from numbers import Real

import numpy as np

from common.ids import generate_routing_id, validate_routing_id, validate_uav_id
from perception.candidate_bank import (
    CandidateBank,
    CandidateLifecycle,
    CandidateSnapshot,
)
from perception.grounding import (
    CandidateResolutionUnavailable,
    CandidateResolver,
    ResolvedCandidatePosition,
)
from perception.runtime import (
    PerceptionRuntimeProfile,
)
from runtime.frame_store import FrameRef, FrameStore
from skills.base import (
    Skill,
    SkillExecutionStateError,
    SkillGoalValidationError,
    require_finite,
    require_positive,
)
from skills.motion_types import MotionPolicy, YawMode, move_toward_with_policy
from skills.types import Observation, SkillContext, SkillGoal, SkillResultCode


class InspectApproachPolicy(str, Enum):
    MAINTAIN_ALTITUDE_ORBIT = "MAINTAIN_ALTITUDE_ORBIT"


class InspectPhase(str, Enum):
    APPROACH = "APPROACH"
    VIEWPOINT_CHANGE = "VIEWPOINT_CHANGE"
    STABILIZING = "STABILIZING"


@dataclass(frozen=True, slots=True)
class InspectGoal(SkillGoal):
    """Semantic INSPECT request without a model-supplied world coordinate."""

    candidate_id: str
    desired_observation_distance_m: float = 4.0
    viewpoint_change_rad: float = pi / 4.0
    max_duration_s: float = 15.0
    approach_policy: InspectApproachPolicy = (
        InspectApproachPolicy.MAINTAIN_ALTITUDE_ORBIT
    )


@dataclass(frozen=True, slots=True)
class InspectPolicy:
    """Trusted motion/evidence limits unavailable to the language model."""

    min_observation_distance_m: float = 2.0
    max_observation_distance_m: float = 20.0
    max_viewpoint_change_rad: float = pi / 2.0
    max_duration_s: float = 60.0
    approach_speed_mps: float = 1.0
    position_tolerance_m: float = 0.25
    max_yaw_rate_rad_s: float = 1.0
    stabilization_duration_s: float = 1.0
    max_evidence_frames: int = 3

    def __post_init__(self) -> None:
        for name in (
            "min_observation_distance_m",
            "max_observation_distance_m",
            "max_viewpoint_change_rad",
            "max_duration_s",
            "approach_speed_mps",
            "position_tolerance_m",
            "max_yaw_rate_rad_s",
            "stabilization_duration_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite positive number")
            normalized = float(value)
            if not isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
            object.__setattr__(self, name, normalized)
        if self.min_observation_distance_m > self.max_observation_distance_m:
            raise ValueError(
                "min_observation_distance_m must not exceed its maximum"
            )
        if self.max_viewpoint_change_rad > pi:
            raise ValueError("max_viewpoint_change_rad must not exceed pi")
        if (
            isinstance(self.max_evidence_frames, bool)
            or not isinstance(self.max_evidence_frames, int)
        ):
            raise TypeError("max_evidence_frames must be an integer")
        if not 1 <= self.max_evidence_frames <= 8:
            raise ValueError("max_evidence_frames must be between 1 and 8")


@dataclass(frozen=True, slots=True)
class InspectionEvidenceHandle:
    evidence_id: str
    uav_id: str
    candidate_id: str
    frame_refs: tuple[FrameRef, ...]
    started_timestamp_s: float
    completed_timestamp_s: float
    geometry_source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            validate_routing_id(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "candidate_id",
            validate_routing_id(self.candidate_id, "candidate_id"),
        )
        frames = tuple(self.frame_refs)
        if not frames or len(frames) > 8:
            raise ValueError("frame_refs must contain between 1 and 8 entries")
        if any(
            not isinstance(frame, FrameRef) or frame.uav_id != self.uav_id
            for frame in frames
        ):
            raise ValueError("frame_refs must belong to evidence uav_id")
        object.__setattr__(self, "frame_refs", frames)
        started = require_finite(self.started_timestamp_s, "started_timestamp_s")
        completed = require_finite(
            self.completed_timestamp_s,
            "completed_timestamp_s",
        )
        if started < 0.0 or completed < started:
            raise ValueError("inspection evidence timestamps are inconsistent")
        object.__setattr__(self, "started_timestamp_s", started)
        object.__setattr__(self, "completed_timestamp_s", completed)
        if not isinstance(self.geometry_source, str) or not self.geometry_source.strip():
            raise ValueError("geometry_source must be a non-empty string")
        object.__setattr__(self, "geometry_source", self.geometry_source.strip())

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "uav_id": self.uav_id,
            "candidate_id": self.candidate_id,
            "frame_refs": [frame.to_dict() for frame in self.frame_refs],
            "started_timestamp_s": self.started_timestamp_s,
            "completed_timestamp_s": self.completed_timestamp_s,
            "geometry_source": self.geometry_source,
        }


class InspectSkill(Skill):
    """Approach, change viewpoint, stabilize, and collect bounded evidence."""

    goal_type = InspectGoal

    def __init__(
        self,
        *,
        candidate_bank: CandidateBank,
        candidate_resolver: CandidateResolver,
        frame_store: FrameStore,
        policy: InspectPolicy | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(candidate_bank, CandidateBank):
            raise TypeError("candidate_bank must be a CandidateBank")
        if not isinstance(candidate_resolver, CandidateResolver):
            raise TypeError("candidate_resolver must satisfy CandidateResolver")
        if not isinstance(frame_store, FrameStore):
            raise TypeError("frame_store must be a FrameStore")
        selected_policy = InspectPolicy() if policy is None else policy
        if not isinstance(selected_policy, InspectPolicy):
            raise TypeError("policy must be an InspectPolicy or None")
        self._candidate_bank = candidate_bank
        self._candidate_resolver = candidate_resolver
        self._frame_store = frame_store
        self._policy = selected_policy
        self._phase: InspectPhase | None = None
        self._candidate: CandidateSnapshot | None = None
        self._resolved: ResolvedCandidatePosition | None = None
        self._approach_position: np.ndarray | None = None
        self._viewpoint_position: np.ndarray | None = None
        self._inspect_motion_policy: MotionPolicy | None = None
        self._started_at_s: float | None = None
        self._phase_started_at_s: float | None = None
        self._last_clock_s: float | None = None
        self._last_observation_s: float | None = None
        self._evidence_refs: list[FrameRef] = []
        self._captured_timestamps: set[float] = set()
        self._evidence_handle: InspectionEvidenceHandle | None = None
        self._initial_approach_distance_m = 0.0
        self._initial_viewpoint_distance_m = 0.0

    @property
    def policy(self) -> InspectPolicy:
        return self._policy

    @property
    def phase(self) -> InspectPhase | None:
        return self._phase

    @property
    def evidence_handle(self) -> InspectionEvidenceHandle | None:
        return self._evidence_handle

    def _validate_goal(self, goal: SkillGoal) -> None:
        if not isinstance(goal, InspectGoal):
            return
        try:
            validate_routing_id(goal.candidate_id, "candidate_id")
        except (TypeError, ValueError) as exc:
            raise SkillGoalValidationError(str(exc)) from None
        if not isinstance(goal.approach_policy, InspectApproachPolicy):
            raise SkillGoalValidationError(
                "approach_policy must be an InspectApproachPolicy"
            )
        distance = require_positive(
            goal.desired_observation_distance_m,
            "desired_observation_distance_m",
        )
        if not (
            self._policy.min_observation_distance_m
            <= distance
            <= self._policy.max_observation_distance_m
        ):
            raise SkillGoalValidationError(
                "desired_observation_distance_m is outside trusted policy bounds"
            )
        angle = require_finite(goal.viewpoint_change_rad, "viewpoint_change_rad")
        if abs(angle) <= 1e-9:
            raise SkillGoalValidationError(
                "viewpoint_change_rad must request a non-zero viewpoint change"
            )
        if abs(angle) > self._policy.max_viewpoint_change_rad:
            raise SkillGoalValidationError(
                "viewpoint_change_rad exceeds trusted policy"
            )
        duration = require_positive(goal.max_duration_s, "max_duration_s")
        if duration > self._policy.max_duration_s:
            raise SkillGoalValidationError("max_duration_s exceeds trusted policy")

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        typed_goal = self._inspect_goal(goal)
        if self._candidate_bank.uav_id != context.uav_id:
            raise SkillExecutionStateError(
                "CandidateBank uav_id does not match SkillContext"
            )
        candidate = self._candidate_bank.get(typed_goal.candidate_id)
        if candidate is None:
            raise SkillExecutionStateError("INSPECT candidate_id is unknown")
        if candidate.lifecycle not in {
            CandidateLifecycle.PROVISIONAL,
            CandidateLifecycle.UNDER_INSPECTION,
        }:
            raise SkillExecutionStateError(
                "INSPECT candidate is not available for inspection"
            )
        start_time = self._read_clock(context)
        try:
            resolved = self._candidate_resolver.resolve(
                candidate,
                timestamp_s=start_time,
            )
        except CandidateResolutionUnavailable as exc:
            raise SkillExecutionStateError(str(exc)) from None
        if not isinstance(resolved, ResolvedCandidatePosition):
            raise SkillExecutionStateError(
                "CandidateResolver must return ResolvedCandidatePosition"
            )
        if (
            resolved.uav_id != context.uav_id
            or resolved.candidate_id != candidate.candidate_id
        ):
            raise SkillExecutionStateError(
                "CandidateResolver returned mismatched routing IDs"
            )
        if (
            resolved.source == "oracle_evaluation"
            and self._candidate_resolver.profile
            is not PerceptionRuntimeProfile.ORACLE_EVALUATION
        ):
            raise SkillExecutionStateError(
                "Oracle candidate geometry crossed a production resolver"
            )

        current = context.uav.get_pose()
        current_xyz = np.asarray([current.x, current.y, current.z], dtype=np.float64)
        target = np.asarray(resolved.position_xyz_m, dtype=np.float64)
        approach = self._maintain_altitude_position(
            current_xyz,
            target,
            typed_goal.desired_observation_distance_m,
        )
        viewpoint = self._rotated_viewpoint(
            approach,
            target,
            typed_goal.viewpoint_change_rad,
        )
        motion_policy = MotionPolicy(
            max_speed=self._policy.approach_speed_mps,
            max_yaw_rate=self._policy.max_yaw_rate_rad_s,
            yaw_mode=YawMode.FACE_POINT,
            look_at_point=tuple(float(value) for value in target),
        )
        motion_policy.validate()

        context.uav.stop()

        # Commit the lifecycle change only after all trusted resolution and
        # motion-policy checks have succeeded.  A missing production resolver
        # therefore cannot strand a candidate in UNDER_INSPECTION.
        if candidate.lifecycle is CandidateLifecycle.PROVISIONAL:
            candidate = self._candidate_bank.mark_under_inspection(
                candidate.candidate_id
            )

        self._phase = InspectPhase.APPROACH
        self._candidate = candidate
        self._resolved = resolved
        self._approach_position = approach
        self._viewpoint_position = viewpoint
        self._inspect_motion_policy = motion_policy
        self._started_at_s = start_time
        self._phase_started_at_s = start_time
        self._last_clock_s = start_time
        self._last_observation_s = None
        self._evidence_refs = []
        self._captured_timestamps = set()
        self._evidence_handle = None
        self._initial_approach_distance_m = float(
            np.linalg.norm(current_xyz - approach)
        )
        self._initial_viewpoint_distance_m = float(
            np.linalg.norm(viewpoint - approach)
        )
        self._set_feedback(
            0.0,
            "INSPECT approaching candidate",
            self._feedback_data(0.0),
        )

    def _on_tick(self, observation: Observation) -> None:
        try:
            self._tick_inspection(observation)
        except Exception:
            # Once INSPECT owns a provisional candidate, every abnormal exit
            # must release it from UNDER_INSPECTION.  Otherwise a resolver,
            # controller, or frame-store fault could strand the candidate and
            # suppress all bounded retry/fallback logic indefinitely.
            self._reject_candidate_if_inspecting(
                self._best_effort_terminal_timestamp(observation)
            )
            raise

    def _tick_inspection(self, observation: Observation) -> None:
        goal = self._inspect_goal(self._active_goal)
        if any(
            value is None
            for value in (
                self._phase,
                self._candidate,
                self._resolved,
                self._approach_position,
                self._viewpoint_position,
                self._inspect_motion_policy,
                self._started_at_s,
                self._phase_started_at_s,
            )
        ):
            raise SkillExecutionStateError("INSPECT was not initialized")
        now = self._read_clock(self._active_context)
        if self._last_clock_s is not None and now < self._last_clock_s - 1e-12:
            raise SkillExecutionStateError("Skill clock moved backwards during INSPECT")
        self._last_clock_s = now
        frame_time = float(observation.timestamp)
        if (
            self._last_observation_s is not None
            and frame_time < self._last_observation_s - 1e-12
        ):
            raise SkillExecutionStateError(
                "Observation timestamp moved backwards during INSPECT"
            )
        self._last_observation_s = frame_time
        if frame_time < self._started_at_s - 1e-12:
            raise SkillExecutionStateError(
                "INSPECT received an Observation captured before Skill start"
            )
        elapsed = max(now, frame_time) - self._started_at_s

        current_candidate = self._candidate_bank.get(self._candidate.candidate_id)
        if current_candidate is None or current_candidate.lifecycle in {
            CandidateLifecycle.REJECTED,
            CandidateLifecycle.STALE,
        }:
            raise SkillExecutionStateError(
                "INSPECT candidate became unavailable"
            )
        if elapsed >= goal.max_duration_s:
            self._reject_candidate_if_inspecting(max(now, frame_time))
            self._fail(
                SkillResultCode.TIMEOUT,
                "INSPECT timed out without confirming a target",
                {
                    "candidate_id": self._candidate.candidate_id,
                    "inspection_confirmed_target": False,
                    "evidence_frame_count": len(self._evidence_refs),
                },
            )
            return

        current_xyz = np.asarray(
            [
                observation.uav_pose.x,
                observation.uav_pose.y,
                observation.uav_pose.z,
            ],
            dtype=np.float64,
        )
        if self._phase is InspectPhase.APPROACH:
            self._tick_motion(
                current_xyz,
                self._approach_position,
                next_phase=InspectPhase.VIEWPOINT_CHANGE,
                elapsed=elapsed,
            )
        elif self._phase is InspectPhase.VIEWPOINT_CHANGE:
            self._tick_motion(
                current_xyz,
                self._viewpoint_position,
                next_phase=InspectPhase.STABILIZING,
                elapsed=elapsed,
            )
        else:
            self._tick_stabilizing(observation, now, elapsed)

    def _tick_motion(
        self,
        current_xyz: np.ndarray,
        destination: np.ndarray,
        *,
        next_phase: InspectPhase,
        elapsed: float,
    ) -> None:
        distance = float(np.linalg.norm(current_xyz - destination))
        if distance <= self._policy.position_tolerance_m:
            self._active_context.uav.set_velocity(np.zeros(3, dtype=np.float64))
            assert self._resolved is not None
            self._active_context.uav.face_point(
                self._resolved.position_xyz_m,
                max_yaw_rate_rad_s=self._policy.max_yaw_rate_rad_s,
            )
            self._phase = next_phase
            self._phase_started_at_s = self._last_clock_s
            self._set_feedback(
                self._progress(distance),
                f"INSPECT entered {next_phase.value}",
                self._feedback_data(elapsed),
            )
            return
        assert self._inspect_motion_policy is not None
        move_toward_with_policy(
            self._active_context.uav,
            destination,
            self._policy.approach_speed_mps,
            self._policy.position_tolerance_m,
            self._inspect_motion_policy,
            initial_yaw=self.initial_yaw,
        )
        self._set_feedback(
            self._progress(distance),
            f"INSPECT {self._phase.value.lower()}",
            self._feedback_data(elapsed, distance_to_phase_goal_m=distance),
        )

    def _tick_stabilizing(
        self,
        observation: Observation,
        now: float,
        elapsed: float,
    ) -> None:
        assert self._resolved is not None
        assert self._phase_started_at_s is not None
        self._active_context.uav.set_velocity(np.zeros(3, dtype=np.float64))
        self._active_context.uav.face_point(
            self._resolved.position_xyz_m,
            max_yaw_rate_rad_s=self._policy.max_yaw_rate_rad_s,
        )
        if (
            len(self._evidence_refs) < self._policy.max_evidence_frames
            and observation.timestamp not in self._captured_timestamps
        ):
            ref = self._frame_store.add_frame(
                uav_id=self._active_context.uav_id,
                frame_id=generate_routing_id("frame"),
                timestamp_s=observation.timestamp,
                rgb=observation.camera_rgb,
            )
            self._evidence_refs.append(ref)
            self._captured_timestamps.add(float(observation.timestamp))

        stable_elapsed = max(0.0, now - self._phase_started_at_s)
        if stable_elapsed >= self._policy.stabilization_duration_s:
            retained = tuple(
                frame
                for frame in self._evidence_refs
                if self._frame_store.contains(frame)
            )[-self._policy.max_evidence_frames :]
            if not retained:
                self._reject_candidate_if_inspecting(
                    max(now, float(observation.timestamp))
                )
                self._fail(
                    SkillResultCode.INVALID_STATE,
                    "INSPECT completed without retained frame evidence",
                    {
                        "candidate_id": self._candidate.candidate_id,
                        "inspection_confirmed_target": False,
                    },
                )
                return
            handle = InspectionEvidenceHandle(
                evidence_id=generate_routing_id("evidence"),
                uav_id=self._active_context.uav_id,
                candidate_id=self._candidate.candidate_id,
                frame_refs=retained,
                started_timestamp_s=self._started_at_s,
                completed_timestamp_s=max(now, float(observation.timestamp)),
                geometry_source=self._resolved.source,
            )
            # INSPECT owns motion and evidence capture only. Release its
            # lifecycle claim before publishing success; a later independent
            # semantic gate may verify/reject the still-provisional candidate.
            self._candidate = (
                self._candidate_bank.release_inspection_pending_review(
                    self._candidate.candidate_id
                )
            )
            self._evidence_handle = handle
            self._succeed(
                SkillResultCode.GOAL_REACHED,
                "INSPECT collected bounded candidate evidence",
                {
                    "candidate_id": self._candidate.candidate_id,
                    "inspection_confirmed_target": False,
                    "evidence_handle": handle.to_dict(),
                },
            )
            return
        self._set_feedback(
            min(0.99, 0.8 + 0.19 * stable_elapsed / self._policy.stabilization_duration_s),
            "INSPECT stabilizing and collecting evidence",
            self._feedback_data(elapsed),
        )

    def _on_cancel(self) -> None:
        self._reject_candidate_if_inspecting(self._best_effort_terminal_timestamp())

    def _best_effort_terminal_timestamp(
        self,
        observation: Observation | None = None,
    ) -> float:
        values = [
            value
            for value in (self._last_clock_s, self._last_observation_s)
            if value is not None
        ]
        if observation is not None:
            try:
                observed = float(observation.timestamp)
                if isfinite(observed) and observed >= 0.0:
                    values.append(observed)
            except (AttributeError, TypeError, ValueError):
                pass
        if self._candidate is not None:
            values.append(self._candidate.last_seen_timestamp_s)
        return max(values, default=0.0)

    def _reject_candidate_if_inspecting(self, timestamp_s: float) -> None:
        candidate = self._candidate
        if candidate is None:
            return
        current = self._candidate_bank.get(candidate.candidate_id)
        if (
            current is not None
            and current.lifecycle is CandidateLifecycle.UNDER_INSPECTION
        ):
            self._candidate = self._candidate_bank.reject(
                current.candidate_id,
                timestamp_s=max(float(timestamp_s), current.last_seen_timestamp_s),
            )

    def _feedback_data(
        self,
        elapsed: float,
        *,
        distance_to_phase_goal_m: float | None = None,
    ) -> dict[str, object]:
        return {
            "uav_id": self._active_context.uav_id,
            "candidate_id": (
                None if self._candidate is None else self._candidate.candidate_id
            ),
            "phase": None if self._phase is None else self._phase.value,
            "elapsed_time_s": float(elapsed),
            "distance_to_phase_goal_m": distance_to_phase_goal_m,
            "evidence_frame_count": len(self._evidence_refs),
            "inspection_confirmed_target": False,
        }

    def _progress(self, distance: float) -> float:
        if self._phase is InspectPhase.APPROACH:
            initial = max(self._initial_approach_distance_m, 1e-9)
            return min(0.5, 0.5 * max(0.0, 1.0 - distance / initial))
        initial = max(self._initial_viewpoint_distance_m, 1e-9)
        return min(0.8, 0.5 + 0.3 * max(0.0, 1.0 - distance / initial))

    @staticmethod
    def _maintain_altitude_position(
        current: np.ndarray,
        target: np.ndarray,
        distance_m: float,
    ) -> np.ndarray:
        radial = current[:2] - target[:2]
        norm = float(np.linalg.norm(radial))
        if norm <= 1e-9:
            radial = np.asarray([-1.0, 0.0], dtype=np.float64)
        else:
            radial = radial / norm
        return np.asarray(
            [
                target[0] + radial[0] * distance_m,
                target[1] + radial[1] * distance_m,
                current[2],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotated_viewpoint(
        approach: np.ndarray,
        target: np.ndarray,
        angle_change_rad: float,
    ) -> np.ndarray:
        radial = approach[:2] - target[:2]
        radius = float(np.linalg.norm(radial))
        angle = atan2(radial[1], radial[0]) + angle_change_rad
        return np.asarray(
            [
                target[0] + radius * cos(angle),
                target[1] + radius * sin(angle),
                approach[2],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _read_clock(context: SkillContext) -> float:
        value = context.clock.now()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise SkillExecutionStateError("Skill clock must return a finite number")
        normalized = float(value)
        if not isfinite(normalized) or normalized < 0.0:
            raise SkillExecutionStateError(
                "Skill clock must return a finite non-negative number"
            )
        return normalized

    @staticmethod
    def _inspect_goal(goal: SkillGoal) -> InspectGoal:
        if not isinstance(goal, InspectGoal):
            raise SkillExecutionStateError("active goal is not InspectGoal")
        return goal

    def _on_reset(self) -> None:
        self._phase = None
        self._candidate = None
        self._resolved = None
        self._approach_position = None
        self._viewpoint_position = None
        self._inspect_motion_policy = None
        self._started_at_s = None
        self._phase_started_at_s = None
        self._last_clock_s = None
        self._last_observation_s = None
        self._evidence_refs = []
        self._captured_timestamps = set()
        self._evidence_handle = None


__all__ = [
    "InspectApproachPolicy",
    "InspectGoal",
    "InspectPhase",
    "InspectPolicy",
    "InspectSkill",
    "InspectionEvidenceHandle",
]
