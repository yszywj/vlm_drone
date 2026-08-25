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
    "target_perception_mode",
    "runtime_profile",
    "privileged_perception",
    "upper_bound_result",
    "production_vision_result",
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
    "search_success",
    "time_to_first_detection_s",
    "time_to_lock_s",
    "valid_track_duration_s",
    "target_lost_count",
    "reacquire_attempts",
    "reacquire_successes",
    "return_success",
    "landing_success",
    "path_length_m",
    "mission_sim_time_s",
    "mission_wall_time_s",
    "interpreter_schema_success",
    "fleet_plan_success",
    "local_plan_success",
    "prompt_tokens",
    "completion_tokens",
    "model_latency_s",
)


# Scalar-only production visual identity evidence.  This is JSONL rather than
# CSV so a future schema version can add bounded scalar reason codes without
# creating any route for Camera, crop, depth-plane, or base64 persistence.
ATTRIBUTE_EVIDENCE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "timestamp_s",
    "mission_id",
    "uav_id",
    "assignment_id",
    "candidate_id",
    "tracker_id",
    "attribute_name",
    "expected_value",
    "observed_value",
    "decision",
    "confidence",
    "observation_count",
    "duration_s",
    "valid_sample_ratio",
    "source",
    "reason_code",
)


# Throttled lifecycle edges emitted by TargetPerceptionCoordinator.  The
# whitelist deliberately excludes Camera/image/depth payloads and every
# simulator/evaluator identity surface (prim paths, motion seeds, truth).
TARGET_PERCEPTION_TRANSITION_FIELDS: tuple[str, ...] = (
    "schema_version",
    "timestamp",
    "uav_id",
    "assignment_id",
    "transition",
    "tracker_id",
    "candidate_id",
    "bbox",
    "detector_confidence",
    "attribute_state",
    "color_result",
    "geometry_state",
    "measurement_source",
    "position_world_m",
    "confirmed",
    "target_id",
    "estimate_source",
)


# Required on manifest.yaml, summary.json, and report.md whenever a fleet run
# declares a target-perception mode. ``backend_by_uav`` stays structured on
# JSON/YAML surfaces and is intentionally not flattened into the fleet CSV.
PERCEPTION_RESULT_FIELDS: tuple[str, ...] = (
    "target_perception_mode",
    "runtime_profile",
    "backend_by_uav",
    "privileged_perception",
    "oracle_acknowledged",
    "qwen_vision_mode",
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


def _record_unit_interval(value: object, name: str) -> float:
    normalized = _record_time(value, name)
    assert normalized is not None
    if normalized > 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return normalized


def _record_positive_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _record_vector(
    value: object,
    name: str,
    length: int,
    *,
    optional: bool = False,
) -> tuple[float, ...] | None:
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise TypeError(f"{name} must contain exactly {length} finite numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name}[{index}] must be a finite number")
        normalized = float(item)
        if not isfinite(normalized):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(normalized)
    return tuple(result)


class _FleetRecord:
    """Small convenience API shared by immutable fleet records."""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AttributeEvidenceRecord(_FleetRecord):
    """One scalar-only production attribute-evidence result row."""

    timestamp_s: float
    uav_id: str
    assignment_id: str
    candidate_id: str
    tracker_id: str | int
    attribute_name: str
    expected_value: str
    observed_value: str | None
    decision: str
    confidence: float
    observation_count: int
    duration_s: float
    valid_sample_ratio: float
    source: str
    mission_id: str | None = None
    reason_code: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("attribute evidence schema_version must be exactly 1")
        object.__setattr__(
            self,
            "timestamp_s",
            _record_time(self.timestamp_s, "timestamp_s"),
        )
        for name in (
            "uav_id",
            "assignment_id",
            "candidate_id",
            "attribute_name",
            "expected_value",
            "source",
        ):
            object.__setattr__(self, name, _record_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "mission_id",
            _record_text(self.mission_id, "mission_id", optional=True),
        )
        object.__setattr__(
            self,
            "reason_code",
            _record_text(self.reason_code, "reason_code", optional=True),
        )
        object.__setattr__(
            self,
            "observed_value",
            _record_text(self.observed_value, "observed_value", optional=True),
        )
        tracker_id = self.tracker_id
        if isinstance(tracker_id, bool) or not isinstance(tracker_id, (str, int)):
            raise TypeError("tracker_id must be a scalar string or integer")
        if isinstance(tracker_id, int):
            if tracker_id < 0:
                raise ValueError("tracker_id integer must be non-negative")
        else:
            object.__setattr__(
                self,
                "tracker_id",
                _record_text(tracker_id, "tracker_id"),
            )
        decision = _record_text(self.decision, "decision")
        assert decision is not None
        normalized_decision = decision.upper()
        if normalized_decision not in {
            "MATCH",
            "MISMATCH",
            "PENDING",
            "UNSUPPORTED",
        }:
            raise ValueError(
                "decision must be MATCH, MISMATCH, PENDING, or UNSUPPORTED"
            )
        object.__setattr__(self, "decision", normalized_decision)
        object.__setattr__(
            self,
            "confidence",
            _record_unit_interval(self.confidence, "confidence"),
        )
        object.__setattr__(
            self,
            "observation_count",
            _record_positive_count(self.observation_count, "observation_count"),
        )
        object.__setattr__(
            self,
            "duration_s",
            _record_time(self.duration_s, "duration_s"),
        )
        object.__setattr__(
            self,
            "valid_sample_ratio",
            _record_unit_interval(self.valid_sample_ratio, "valid_sample_ratio"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AttributeEvidenceRecord":
        if not isinstance(value, Mapping):
            raise TypeError("attribute evidence must be a mapping")
        raw = dict(value)
        unknown = sorted(set(raw) - set(ATTRIBUTE_EVIDENCE_FIELDS))
        if unknown:
            raise ValueError(
                "attribute evidence contains unknown fields: " + ", ".join(unknown)
            )
        required = set(ATTRIBUTE_EVIDENCE_FIELDS) - {"mission_id", "reason_code"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                "attribute evidence is missing required fields: " + ", ".join(missing)
            )
        return cls(**raw)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TargetPerceptionTransitionRecord(_FleetRecord):
    """One bounded production candidate lifecycle edge."""

    timestamp: float
    uav_id: str
    assignment_id: str
    transition: str
    tracker_id: str
    candidate_id: str
    bbox: tuple[float, float, float, float]
    detector_confidence: float
    attribute_state: str
    color_result: str | None
    geometry_state: str
    measurement_source: str | None
    position_world_m: tuple[float, float, float] | None
    confirmed: bool
    target_id: str | None
    estimate_source: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "target perception transition schema_version must be exactly 1"
            )
        object.__setattr__(self, "timestamp", _record_time(self.timestamp, "timestamp"))
        for name in (
            "uav_id",
            "assignment_id",
            "tracker_id",
            "candidate_id",
            "estimate_source",
        ):
            object.__setattr__(self, name, _record_text(getattr(self, name), name))

        transition = _record_text(self.transition, "transition")
        assert transition is not None
        transition = transition.casefold()
        if transition not in {
            "candidate_created",
            "candidate_rejected",
            "candidate_confirmed",
        }:
            raise ValueError("unsupported target perception transition")
        object.__setattr__(self, "transition", transition)

        bbox = _record_vector(self.bbox, "bbox", 4)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("bbox must be ordered normalized xyxy coordinates")
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(
            self,
            "detector_confidence",
            _record_unit_interval(self.detector_confidence, "detector_confidence"),
        )

        attribute_state = _record_text(self.attribute_state, "attribute_state")
        assert attribute_state is not None
        attribute_state = attribute_state.casefold()
        if attribute_state not in {"match", "mismatch", "pending", "unsupported"}:
            raise ValueError("unsupported attribute_state")
        object.__setattr__(self, "attribute_state", attribute_state)

        color_result = _record_text(
            self.color_result,
            "color_result",
            optional=True,
        )
        if color_result is not None:
            lowered = color_result.casefold()
            if len(color_result) > 64 or "base64," in lowered or lowered.startswith("data:"):
                raise ValueError("color_result must be a bounded scalar label")
        object.__setattr__(self, "color_result", color_result)

        geometry_state = _record_text(self.geometry_state, "geometry_state")
        assert geometry_state is not None
        geometry_state = geometry_state.casefold()
        if geometry_state not in {"measurement_created", "measurement_rejected"}:
            raise ValueError("unsupported geometry_state")
        object.__setattr__(self, "geometry_state", geometry_state)
        measurement_source = _record_text(
            self.measurement_source,
            "measurement_source",
            optional=True,
        )
        allowed_measurement_sources = {
            "isaac_depth",
            "isaac_depth_bbox_center",
            "isaac_depth_bbox_bottom_center",
            "isaac_depth_bbox_patch_median",
            "isaac_depth_foreground_cluster_median",
            "rgbd_depth_geometry",
            "rgbd_depth_geometry_bbox_center",
            "rgbd_depth_geometry_bbox_bottom_center",
            "rgbd_depth_geometry_bbox_patch_median",
            "rgbd_depth_geometry_foreground_cluster_median",
            "rgbd_depth_geometry_fallback",
            "temporal_ray_depth",
        }
        if (
            measurement_source is not None
            and measurement_source not in allowed_measurement_sources
        ):
            raise ValueError("unsupported measurement_source")
        if geometry_state == "measurement_created" and measurement_source is None:
            raise ValueError("measurement_created requires measurement_source")
        if geometry_state == "measurement_rejected" and measurement_source is not None:
            raise ValueError("measurement_rejected cannot carry measurement_source")
        object.__setattr__(self, "measurement_source", measurement_source)
        object.__setattr__(
            self,
            "position_world_m",
            _record_vector(
                self.position_world_m,
                "position_world_m",
                3,
                optional=True,
            ),
        )
        if type(self.confirmed) is not bool:
            raise TypeError("confirmed must be bool")
        target_id = _record_text(self.target_id, "target_id", optional=True)
        if self.confirmed and target_id is None:
            raise ValueError("confirmed transition requires target_id")
        if not self.confirmed and target_id is not None:
            raise ValueError("unconfirmed transition cannot carry target_id")
        object.__setattr__(self, "target_id", target_id)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "TargetPerceptionTransitionRecord":
        if not isinstance(value, Mapping):
            raise TypeError("target perception transition must be a mapping")
        raw = dict(value)
        unknown = sorted(set(raw) - set(TARGET_PERCEPTION_TRANSITION_FIELDS))
        if unknown:
            raise ValueError(
                "target perception transition contains unknown fields: "
                + ", ".join(unknown)
            )
        missing = sorted(set(TARGET_PERCEPTION_TRANSITION_FIELDS) - set(raw))
        if missing:
            raise ValueError(
                "target perception transition is missing required fields: "
                + ", ".join(missing)
            )
        return cls(**raw)  # type: ignore[arg-type]


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
    "ATTRIBUTE_EVIDENCE_FIELDS",
    "AttributeEvidenceRecord",
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
    "PERCEPTION_RESULT_FIELDS",
    "RecoveryActionRecord",
    "RunStatus",
    "SKILL_EXECUTION_FIELDS",
    "STATE_SAMPLE_FIELDS",
    "TARGET_PERCEPTION_TRANSITION_FIELDS",
    "SkillExecutionRecord",
    "StateSampleRecord",
    "StorageStatus",
    "TargetPerceptionTransitionRecord",
    "TRAIN_METRIC_FIELDS",
    "ValidationFindingRecord",
]
