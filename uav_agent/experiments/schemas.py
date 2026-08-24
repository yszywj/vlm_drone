"""Shared, dependency-free schemas for lightweight experiment outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from typing import Mapping


class RunStatus(str, Enum):
    """Persistent lifecycle state of one experiment run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class StorageStatus(str, Enum):
    """Non-destructive result of a periodic run-storage check."""

    OK = "ok"
    WARNING = "warning"
    STOP_REQUIRED = "stop_required"


class MetricPhase(str, Enum):
    """Allowed phases in episode and failure CSV records."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class FailureReason(str, Enum):
    """Canonical terminal failure reasons used by every evaluator."""

    PLANNER_INVALID_OUTPUT = "PLANNER_INVALID_OUTPUT"
    PLAN_VALIDATION_FAILED = "PLAN_VALIDATION_FAILED"

    TAKEOFF_FAILED = "TAKEOFF_FAILED"
    TAKEOFF_TIMEOUT = "TAKEOFF_TIMEOUT"

    GOTO_SEARCH_FAILED = "GOTO_SEARCH_FAILED"
    GOTO_SEARCH_TIMEOUT = "GOTO_SEARCH_TIMEOUT"

    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    SEARCH_TIMEOUT = "SEARCH_TIMEOUT"
    FALSE_TARGET_LOCK = "FALSE_TARGET_LOCK"

    TRACK_FAILED = "TRACK_FAILED"
    TARGET_LOST = "TARGET_LOST"
    TRACK_DURATION_NOT_MET = "TRACK_DURATION_NOT_MET"

    REACQUIRE_FAILED = "REACQUIRE_FAILED"
    REACQUIRE_TIMEOUT = "REACQUIRE_TIMEOUT"

    RETURN_FAILED = "RETURN_FAILED"
    RETURN_TIMEOUT = "RETURN_TIMEOUT"

    LAND_FAILED = "LAND_FAILED"
    LAND_TIMEOUT = "LAND_TIMEOUT"

    COLLISION = "COLLISION"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    SAFETY_ABORT = "SAFETY_ABORT"
    MISSION_TIMEOUT = "MISSION_TIMEOUT"

    SIMULATOR_ERROR = "SIMULATOR_ERROR"
    CUDA_OUT_OF_MEMORY = "CUDA_OUT_OF_MEMORY"
    PROCESS_CRASH = "PROCESS_CRASH"
    INTERRUPTED = "INTERRUPTED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


# A single superset supports both RL and VLM/SFT training. MetricLogger writes
# missing optional values as empty CSV cells rather than manufacturing zeros.
TRAIN_METRIC_FIELDS: tuple[str, ...] = (
    "timestamp",
    "global_step",
    "update",
    "episodes_completed",
    "episode_return_mean",
    "episode_length_mean",
    "mission_success_rate_100",
    "learning_rate",
    "fps",
    "wall_time_s",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "train_loss",
    "validation_loss",
    "token_accuracy",
)


EVAL_METRIC_FIELDS: tuple[str, ...] = (
    "timestamp",
    "global_step",
    "checkpoint_step",
    "num_episodes",
    "mission_success_rate",
    "takeoff_success_rate",
    "goto_search_success_rate",
    "search_success_rate",
    "correct_lock_rate",
    "false_lock_rate",
    "track_success_rate",
    "reacquire_success_rate",
    "return_success_rate",
    "landing_success_rate",
    "collision_rate",
    "safety_abort_rate",
    "mean_mission_time_s",
    "mean_episode_return",
)


EPISODE_METRIC_FIELDS: tuple[str, ...] = (
    "run_id",
    "phase",
    "seed",
    "global_step",
    "episode_id",
    "scenario_id",
    "mission_success_strict",
    "failure_reason",
    "takeoff_success",
    "goto_search_success",
    "search_success",
    "correct_target_locked",
    "false_target_lock",
    "track_success",
    "reacquire_triggered",
    "reacquire_success",
    "return_success",
    "landing_success",
    "collision",
    "out_of_bounds",
    "safety_abort",
    "timeout",
    "time_to_first_detection_s",
    "time_to_correct_lock_s",
    "valid_track_duration_s",
    "mission_sim_time_s",
    "mission_wall_time_s",
    "path_length_m",
    "episode_return",
)


FAILURE_CASE_FIELDS: tuple[str, ...] = (
    "run_id",
    "phase",
    "global_step",
    "episode_id",
    "scenario_id",
    "failure_reason",
    "terminal_skill",
    "mission_sim_time_s",
    "message",
    "fleet_mission_id",
    "assignment_id",
    "uav_id",
    "goal_id",
    "stage",
    "code",
    "severity",
    "status",
)


FINAL_METRIC_FIELDS: tuple[str, ...] = (
    "run_id",
    "best_checkpoint_step",
    "num_test_episodes",
    "mission_success_rate",
    "mission_success_ci95_low",
    "mission_success_ci95_high",
    "search_success_rate",
    "correct_lock_rate",
    "false_lock_rate",
    "track_success_rate",
    "reacquire_success_rate",
    "return_success_rate",
    "landing_success_rate",
    "collision_rate",
    "safety_abort_rate",
    "mean_mission_time_s",
    "mean_episode_return",
)


# Fleet mission result tables deliberately contain only scalar or short textual
# values.  Camera data and full environment observations have no representable
# field in these schemas.
FLEET_METRIC_FIELDS: tuple[str, ...] = (
    "schema_version",
    "fleet_mission_id",
    "status",
    "strict_success",
    "semantic_success",
    "execution_success",
    "safety_success",
    "partial_success",
    "goal_count",
    "goals_completed",
    "goal_completion_rate",
    "assignment_count",
    "assignments_succeeded",
    "assignments_failed",
    "reassignment_count",
    "reassignments_succeeded",
    "repair_count",
    "repairs_succeeded",
    "repair_success_rate",
    "validation_finding_count",
    "collision_count",
    "out_of_bounds_count",
    "emergency_landing_count",
    "minimum_inter_uav_distance_m",
    "mission_sim_time_s",
    "mission_wall_time_s",
    "interpreter_schema_success",
    "fleet_plan_success",
    "local_plan_success",
    "prompt_tokens",
    "completion_tokens",
    "model_latency_s",
)


AGENT_METRIC_FIELDS: tuple[str, ...] = (
    "schema_version",
    "fleet_mission_id",
    "assignment_id",
    "uav_id",
    "status",
    "path_length_m",
    "airborne_time_s",
    "hover_time_s",
    "hold_time_s",
    "time_to_first_detection_s",
    "time_to_first_lock_s",
    "valid_track_duration_s",
    "target_lost_count",
    "target_reacquired_count",
    "returned_home",
    "landed",
)


GOAL_METRIC_FIELDS: tuple[str, ...] = (
    "schema_version",
    "fleet_mission_id",
    "assignment_id",
    "uav_id",
    "goal_id",
    "goal_type",
    "completed",
    "completion_time_s",
    "evidence_source",
    "unmet_reason",
    "constraint_deviation",
)


SKILL_EXECUTION_FIELDS: tuple[str, ...] = (
    "schema_version",
    "fleet_mission_id",
    "assignment_id",
    "uav_id",
    "goal_id",
    "step_id",
    "skill_name",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "result_code",
    "attempt",
    "recovery_action_id",
)


STATE_SAMPLE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "fleet_mission_id",
    "uav_id",
    "timestamp_s",
    "x_m",
    "y_m",
    "z_m",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "mode",
    "assignment_id",
    "goal_id",
    "step_id",
    "target_detected",
    "target_locked",
    "minimum_inter_uav_distance_m",
)


BATCH_EPISODE_METRIC_FIELDS: tuple[str, ...] = (
    "run_id",
    "status",
    "failure_reason",
    "strict_success",
    "semantic_success",
    "execution_success",
    "safety_success",
    "partial_success",
    "interpreter_schema_success",
    "fleet_plan_success",
    "local_plan_success",
    "repair_count",
    "repairs_succeeded",
    "reassignment_count",
    "reassignments_succeeded",
    "goal_count",
    "goals_completed",
    "goal_completion_rate",
    "prompt_tokens",
    "completion_tokens",
    "model_latency_s",
    "mission_sim_time_s",
    "mission_wall_time_s",
    "error_codes",
    "details_retained",
)


def _record_text(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > 1024 or any(char in normalized for char in "\x00\r\n"):
        raise ValueError(f"{name} must be a bounded single-line string")
    return normalized


def _record_time(value: object, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


class _FleetRecord:
    """Small convenience API shared by immutable fleet records."""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlanningAttemptRecord(_FleetRecord):
    attempt_id: str
    timestamp_s: float
    stage: str
    mission_id: str
    model_role: str
    prompt_sha256: str
    prompt_schema_version: str
    accepted: bool
    assignment_id: str | None = None
    uav_id: str | None = None
    proposal_id: str | None = None
    repaired_from_attempt_id: str | None = None
    error_codes: tuple[str, ...] = ()
    proposal: Mapping[str, object] | None = None
    raw_text_length: int | None = None
    raw_text_tail: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("attempt_id", "stage", "mission_id", "model_role", "prompt_schema_version"):
            object.__setattr__(self, name, _record_text(getattr(self, name), name))
        object.__setattr__(self, "timestamp_s", _record_time(self.timestamp_s, "timestamp_s"))
        digest = _record_text(self.prompt_sha256, "prompt_sha256")
        if len(digest or "") != 64 or any(c not in "0123456789abcdef" for c in digest or ""):
            raise ValueError("prompt_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "prompt_sha256", digest)
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean")
        for name in ("assignment_id", "uav_id", "proposal_id", "repaired_from_attempt_id"):
            object.__setattr__(self, name, _record_text(getattr(self, name), name, optional=True))
        if not isinstance(self.error_codes, tuple):
            object.__setattr__(self, "error_codes", tuple(self.error_codes))
        if len(self.error_codes) > 64:
            raise ValueError("error_codes must contain at most 64 entries")
        for code in self.error_codes:
            _record_text(code, "error_code")
        if (self.raw_text_length is None) != (self.raw_text_tail is None):
            raise ValueError("raw_text_length and raw_text_tail must be set together")
        if self.raw_text_length is not None:
            if isinstance(self.raw_text_length, bool) or not isinstance(self.raw_text_length, int) or self.raw_text_length < 0:
                raise ValueError("raw_text_length must be a non-negative integer")
            if not isinstance(self.raw_text_tail, str) or len(self.raw_text_tail) > 500:
                raise ValueError("raw_text_tail must contain at most 500 characters")


@dataclass(frozen=True, slots=True)
class ValidationFindingRecord(_FleetRecord):
    finding_id: str
    timestamp_s: float
    stage: str
    scope: str
    severity: str
    code: str
    message: str
    mission_id: str
    assignment_id: str | None = None
    uav_id: str | None = None
    goal_id: str | None = None
    step_id: str | None = None
    proposal_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    recommended_action: str | None = None
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class RecoveryActionRecord(_FleetRecord):
    recovery_action_id: str
    timestamp_s: float
    mission_id: str
    stage: str
    action: str
    outcome: str
    assignment_id: str | None = None
    uav_id: str | None = None
    goal_id: str | None = None
    source_attempt_id: str | None = None
    resulting_plan_version: int | None = None
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class GoalResultRecord(_FleetRecord):
    fleet_mission_id: str
    goal_id: str
    goal_type: str
    completed: bool
    assignment_id: str | None = None
    uav_id: str | None = None
    completion_time_s: float | None = None
    evidence_source: str | None = None
    unmet_reason: str | None = None
    constraint_deviation: str | None = None
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class AgentMetricRecord(_FleetRecord):
    fleet_mission_id: str
    uav_id: str
    status: str
    assignment_id: str | None = None
    path_length_m: float = 0.0
    airborne_time_s: float = 0.0
    hover_time_s: float = 0.0
    hold_time_s: float = 0.0
    time_to_first_detection_s: float | None = None
    time_to_first_lock_s: float | None = None
    valid_track_duration_s: float = 0.0
    target_lost_count: int = 0
    target_reacquired_count: int = 0
    returned_home: bool = False
    landed: bool = False
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class SkillExecutionRecord(_FleetRecord):
    fleet_mission_id: str
    uav_id: str
    step_id: str
    skill_name: str
    start_time_s: float
    end_time_s: float
    result_code: str
    attempt: int = 1
    assignment_id: str | None = None
    goal_id: str | None = None
    recovery_action_id: str | None = None
    schema_version: int = 1

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_time_s) - float(self.start_time_s))

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["duration_s"] = self.duration_s
        return result


@dataclass(frozen=True, slots=True)
class StateSampleRecord(_FleetRecord):
    fleet_mission_id: str
    uav_id: str
    timestamp_s: float
    position_xyz_m: tuple[float, float, float]
    velocity_xyz_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mode: str = "UNKNOWN"
    assignment_id: str | None = None
    goal_id: str | None = None
    step_id: str | None = None
    target_detected: bool = False
    target_locked: bool = False
    minimum_inter_uav_distance_m: float | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        x, y, z = self.position_xyz_m
        vx, vy, vz = self.velocity_xyz_mps
        return {
            "schema_version": self.schema_version,
            "fleet_mission_id": self.fleet_mission_id,
            "uav_id": self.uav_id,
            "timestamp_s": self.timestamp_s,
            "x_m": x,
            "y_m": y,
            "z_m": z,
            "vx_mps": vx,
            "vy_mps": vy,
            "vz_mps": vz,
            "mode": self.mode,
            "assignment_id": self.assignment_id,
            "goal_id": self.goal_id,
            "step_id": self.step_id,
            "target_detected": self.target_detected,
            "target_locked": self.target_locked,
            "minimum_inter_uav_distance_m": self.minimum_inter_uav_distance_m,
        }


__all__ = [
    "AGENT_METRIC_FIELDS",
    "AgentMetricRecord",
    "BATCH_EPISODE_METRIC_FIELDS",
    "EPISODE_METRIC_FIELDS",
    "EVAL_METRIC_FIELDS",
    "FAILURE_CASE_FIELDS",
    "FLEET_METRIC_FIELDS",
    "FINAL_METRIC_FIELDS",
    "FailureReason",
    "GOAL_METRIC_FIELDS",
    "GoalResultRecord",
    "MetricPhase",
    "PlanningAttemptRecord",
    "RecoveryActionRecord",
    "RunStatus",
    "SKILL_EXECUTION_FIELDS",
    "STATE_SAMPLE_FIELDS",
    "SkillExecutionRecord",
    "StateSampleRecord",
    "StorageStatus",
    "TRAIN_METRIC_FIELDS",
    "ValidationFindingRecord",
]
