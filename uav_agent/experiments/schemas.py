"""Shared, dependency-free schemas for lightweight experiment outputs."""

from __future__ import annotations

from enum import Enum


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


__all__ = [
    "EPISODE_METRIC_FIELDS",
    "EVAL_METRIC_FIELDS",
    "FAILURE_CASE_FIELDS",
    "FINAL_METRIC_FIELDS",
    "FailureReason",
    "MetricPhase",
    "RunStatus",
    "StorageStatus",
    "TRAIN_METRIC_FIELDS",
]
