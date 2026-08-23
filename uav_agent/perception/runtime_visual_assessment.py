"""Strict V2 protocol for asynchronous runtime visual mission assessment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
import json

from common.ids import (
    validate_mission_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from common.obstacle_types import ObstacleObservation
from perception.qwen_vlm_verifier import VisualReviewFrame
from target.types import TargetSpec
from models import (
    AsyncModelRequest,
    AsyncModelResult,
    ChatMessage,
    GenerationOptions,
    ImageURLContentPart,
    JsonSchemaResponseFormat,
    ModelProtocolError,
    TextContentPart,
    encode_rgb_to_data_url,
)


class RuntimeVisualProtocolError(ValueError):
    """Raised when a V2 assessment message is malformed or unrouted."""


class RuntimeVisualDecision(str, Enum):
    NO_RELEVANT_CHANGE = "NO_RELEVANT_CHANGE"
    NO_TARGET = "NO_TARGET"
    POSSIBLE_TARGET = "POSSIBLE_TARGET"
    TARGET_MATCH = "TARGET_MATCH"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    PATH_MAY_BE_BLOCKED = "PATH_MAY_BE_BLOCKED"
    PATH_BLOCKED = "PATH_BLOCKED"
    TASK_REVIEW_REQUIRED = "TASK_REVIEW_REQUIRED"


class RuntimeVisualAction(str, Enum):
    CONTINUE = "CONTINUE"
    HOLD_AND_INSPECT = "HOLD_AND_INSPECT"
    REQUEST_REPLAN = "REQUEST_REPLAN"
    REQUEST_LAND = "REQUEST_LAND"


class TargetAssessmentStatus(str, Enum):
    NO_TARGET = "NO_TARGET"
    POSSIBLE_TARGET = "POSSIBLE_TARGET"
    TARGET_MATCH = "TARGET_MATCH"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    AMBIGUOUS = "AMBIGUOUS"


class RuntimeSafetyState(str, Enum):
    CLEAR = "CLEAR"
    HAZARD_SUSPECTED = "HAZARD_SUSPECTED"
    BRAKING = "BRAKING"
    HOLDING = "HOLDING"
    GEOMETRY_GROUNDED = "GEOMETRY_GROUNDED"
    REPLANNING = "REPLANNING"
    READY_TO_RESUME = "READY_TO_RESUME"


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized) or (minimum is not None and normalized < minimum):
        raise RuntimeVisualProtocolError(f"{name} is outside its valid range")
    return normalized


def _text(value: object, name: str, *, maximum: int = 2048) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeVisualProtocolError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise RuntimeVisualProtocolError(f"{name} exceeds {maximum} characters")
    return normalized


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    raise RuntimeVisualProtocolError(f"{name} is not supported")


def _strict(data: Mapping[str, object], required: set[str], name: str) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be an object")
    missing, unknown = required - set(data), set(data) - required
    if missing:
        raise RuntimeVisualProtocolError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise RuntimeVisualProtocolError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


@dataclass(frozen=True, slots=True)
class CompletedStepSummary:
    step_id: str
    skill: str
    result_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", validate_routing_id(self.step_id, "step_id"))
        object.__setattr__(self, "skill", validate_routing_id(self.skill, "skill"))
        object.__setattr__(self, "result_code", validate_routing_id(self.result_code, "result_code"))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.step_id, "skill": self.skill, "result_code": self.result_code}


@dataclass(frozen=True, slots=True)
class CurrentStepSummary:
    step_id: str
    skill: str
    progress: float | None
    elapsed_time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", validate_routing_id(self.step_id, "step_id"))
        object.__setattr__(self, "skill", validate_routing_id(self.skill, "skill"))
        if self.progress is not None:
            progress = _finite(self.progress, "progress", minimum=0.0)
            if progress > 1.0:
                raise RuntimeVisualProtocolError("progress must be within [0, 1]")
            object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "elapsed_time_s", _finite(self.elapsed_time_s, "elapsed_time_s", minimum=0.0))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.step_id, "skill": self.skill, "progress": self.progress, "elapsed_time_s": self.elapsed_time_s}


@dataclass(frozen=True, slots=True)
class RemainingStepSummary:
    skill: str
    duration_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill", validate_routing_id(self.skill, "skill"))
        if self.duration_s is not None:
            object.__setattr__(self, "duration_s", _finite(self.duration_s, "duration_s", minimum=0.0))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"skill": self.skill}
        if self.duration_s is not None:
            result["duration_s"] = self.duration_s
        return result


@dataclass(frozen=True, slots=True)
class PlanProgressSummary:
    completed_steps: tuple[CompletedStepSummary, ...]
    current_step: CurrentStepSummary
    remaining_steps: tuple[RemainingStepSummary, ...]

    def __post_init__(self) -> None:
        completed, remaining = tuple(self.completed_steps), tuple(self.remaining_steps)
        if len(completed) > 100 or any(not isinstance(item, CompletedStepSummary) for item in completed):
            raise RuntimeVisualProtocolError("completed_steps is invalid or unbounded")
        if not isinstance(self.current_step, CurrentStepSummary):
            raise TypeError("current_step must be a CurrentStepSummary")
        if len(remaining) > 100 or any(not isinstance(item, RemainingStepSummary) for item in remaining):
            raise RuntimeVisualProtocolError("remaining_steps is invalid or unbounded")
        object.__setattr__(self, "completed_steps", completed)
        object.__setattr__(self, "remaining_steps", remaining)

    def to_dict(self) -> dict[str, object]:
        return {
            "completed_steps": [item.to_dict() for item in self.completed_steps],
            "current_step": self.current_step.to_dict(),
            "remaining_steps": [item.to_dict() for item in self.remaining_steps],
        }


@dataclass(frozen=True, slots=True)
class RuntimeVisualReviewInputV2:
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    observation_timestamp_s: float
    frame_id: str
    original_instruction: str
    target_spec: TargetSpec
    plan_progress: PlanProgressSummary
    frames: tuple[VisualReviewFrame, ...]
    obstacle_observations: tuple[ObstacleObservation, ...]
    safety_state: RuntimeSafetyState
    mission_elapsed_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int) or self.plan_version <= 0:
            raise RuntimeVisualProtocolError("plan_version must be a positive integer")
        object.__setattr__(self, "observation_timestamp_s", _finite(self.observation_timestamp_s, "observation_timestamp_s", minimum=0.0))
        object.__setattr__(self, "frame_id", validate_routing_id(self.frame_id, "frame_id"))
        object.__setattr__(self, "original_instruction", _text(self.original_instruction, "original_instruction"))
        if not isinstance(self.target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        if not isinstance(self.plan_progress, PlanProgressSummary):
            raise TypeError("plan_progress must be a PlanProgressSummary")
        frames = tuple(self.frames)
        if not 1 <= len(frames) <= 3 or any(not isinstance(item, VisualReviewFrame) for item in frames):
            raise RuntimeVisualProtocolError("frames must contain 1..3 VisualReviewFrame values")
        if any(item.ref.uav_id != self.uav_id for item in frames) or frames[-1].ref.frame_id != self.frame_id:
            raise RuntimeVisualProtocolError("frame routing does not match runtime review input")
        object.__setattr__(self, "frames", frames)
        observations = tuple(self.obstacle_observations)
        if len(observations) > 3 or any(not isinstance(item, ObstacleObservation) for item in observations):
            raise RuntimeVisualProtocolError("obstacle_observations must contain at most three typed values")
        if any(item.uav_id != self.uav_id for item in observations):
            raise RuntimeVisualProtocolError("obstacle observation uav_id mismatch")
        object.__setattr__(self, "obstacle_observations", observations)
        object.__setattr__(self, "safety_state", _enum(self.safety_state, RuntimeSafetyState, "safety_state"))
        object.__setattr__(self, "mission_elapsed_s", _finite(self.mission_elapsed_s, "mission_elapsed_s", minimum=0.0))

    def text_payload(self) -> dict[str, object]:
        """Return the model-visible, image-free mission-level summary."""

        return {
            "routing": {
                "review_id": self.review_id,
                "mission_id": self.mission_id,
                "uav_id": self.uav_id,
                "plan_version": self.plan_version,
                "frame_id": self.frame_id,
                "observation_timestamp_s": self.observation_timestamp_s,
            },
            "original_instruction": self.original_instruction,
            "target_spec": self.target_spec.to_dict(),
            "plan_progress": self.plan_progress.to_dict(),
            "obstacle_observations": [item.to_dict() for item in self.obstacle_observations],
            "safety_state": {"state": self.safety_state.value},
            "mission_elapsed_s": self.mission_elapsed_s,
        }


@dataclass(frozen=True, slots=True)
class TaskProgressAssessment:
    current_step_consistent: bool
    current_step_blocked: bool
    original_mission_still_achievable: bool

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in (
            self.current_step_consistent,
            self.current_step_blocked,
            self.original_mission_still_achievable,
        )):
            raise TypeError("task progress assessment fields must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "current_step_consistent": self.current_step_consistent,
            "current_step_blocked": self.current_step_blocked,
            "original_mission_still_achievable": self.original_mission_still_achievable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TaskProgressAssessment":
        _strict(data, {"current_step_consistent", "current_step_blocked", "original_mission_still_achievable"}, "task_progress_assessment")
        return cls(
            data["current_step_consistent"],  # type: ignore[arg-type]
            data["current_step_blocked"],  # type: ignore[arg-type]
            data["original_mission_still_achievable"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RuntimeTargetAssessment:
    status: TargetAssessmentStatus
    candidate_present: bool
    bbox_xyxy_normalized: tuple[float, float, float, float] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(self.status, TargetAssessmentStatus, "target status"))
        if not isinstance(self.candidate_present, bool):
            raise TypeError("candidate_present must be bool")
        if self.bbox_xyxy_normalized is None:
            if self.candidate_present:
                raise RuntimeVisualProtocolError("a present candidate requires bbox")
            return
        values = tuple(_finite(value, "bbox") for value in self.bbox_xyxy_normalized)
        if len(values) != 4 or any(value < 0 or value > 1 for value in values) or values[0] >= values[2] or values[1] >= values[3]:
            raise RuntimeVisualProtocolError("bbox_xyxy_normalized is invalid")
        if not self.candidate_present:
            raise RuntimeVisualProtocolError("an absent candidate cannot contain bbox")
        object.__setattr__(self, "bbox_xyxy_normalized", values)

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "candidate_present": self.candidate_present, "bbox_xyxy_normalized": None if self.bbox_xyxy_normalized is None else list(self.bbox_xyxy_normalized)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RuntimeTargetAssessment":
        _strict(data, {"status", "candidate_present", "bbox_xyxy_normalized"}, "target_assessment")
        return cls(
            data["status"],  # type: ignore[arg-type]
            data["candidate_present"],  # type: ignore[arg-type]
            data["bbox_xyxy_normalized"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RuntimeHazardAssessment:
    obstacle_id: str
    present: bool
    blocks_active_corridor: bool
    visual_confidence: float
    geometry_grounded: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "obstacle_id", validate_routing_id(self.obstacle_id, "obstacle_id"))
        if any(not isinstance(value, bool) for value in (self.present, self.blocks_active_corridor, self.geometry_grounded)):
            raise TypeError("hazard flags must be bool")
        confidence = _finite(self.visual_confidence, "visual_confidence", minimum=0.0)
        if confidence > 1.0:
            raise RuntimeVisualProtocolError("visual_confidence must be within [0, 1]")
        object.__setattr__(self, "visual_confidence", confidence)
        if self.geometry_grounded and not self.present:
            raise RuntimeVisualProtocolError("absent hazards cannot be geometry-grounded")

    def to_dict(self) -> dict[str, object]:
        return {"obstacle_id": self.obstacle_id, "present": self.present, "blocks_active_corridor": self.blocks_active_corridor, "visual_confidence": self.visual_confidence, "geometry_grounded": self.geometry_grounded}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RuntimeHazardAssessment":
        _strict(data, {"obstacle_id", "present", "blocks_active_corridor", "visual_confidence", "geometry_grounded"}, "hazard")
        return cls(
            data["obstacle_id"],  # type: ignore[arg-type]
            data["present"],  # type: ignore[arg-type]
            data["blocks_active_corridor"],  # type: ignore[arg-type]
            data["visual_confidence"],  # type: ignore[arg-type]
            data["geometry_grounded"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RuntimeVisualAssessmentV2:
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    observation_timestamp_s: float
    frame_id: str
    decision: RuntimeVisualDecision
    task_progress_assessment: TaskProgressAssessment
    target_assessment: RuntimeTargetAssessment
    hazards: tuple[RuntimeHazardAssessment, ...]
    recommended_action: RuntimeVisualAction
    reason_codes: tuple[str, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2 or isinstance(self.schema_version, bool):
            raise RuntimeVisualProtocolError("schema_version must equal 2")
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int) or self.plan_version <= 0:
            raise RuntimeVisualProtocolError("plan_version must be positive")
        object.__setattr__(self, "observation_timestamp_s", _finite(self.observation_timestamp_s, "observation_timestamp_s", minimum=0.0))
        object.__setattr__(self, "frame_id", validate_routing_id(self.frame_id, "frame_id"))
        object.__setattr__(self, "decision", _enum(self.decision, RuntimeVisualDecision, "decision"))
        if not isinstance(self.task_progress_assessment, TaskProgressAssessment):
            raise TypeError("task_progress_assessment must be typed")
        if not isinstance(self.target_assessment, RuntimeTargetAssessment):
            raise TypeError("target_assessment must be typed")
        hazards = tuple(self.hazards)
        if len(hazards) > 32 or any(not isinstance(item, RuntimeHazardAssessment) for item in hazards):
            raise RuntimeVisualProtocolError("hazards is invalid or unbounded")
        ids = [item.obstacle_id for item in hazards]
        if len(ids) != len(set(ids)):
            raise RuntimeVisualProtocolError("hazard obstacle IDs must be unique")
        object.__setattr__(self, "hazards", hazards)
        object.__setattr__(self, "recommended_action", _enum(self.recommended_action, RuntimeVisualAction, "recommended_action"))
        reasons = tuple(self.reason_codes)
        if len(reasons) > 32 or any(not isinstance(item, str) or not item or len(item) > 64 for item in reasons):
            raise RuntimeVisualProtocolError("reason_codes is invalid or unbounded")
        if len(reasons) != len(set(reasons)):
            raise RuntimeVisualProtocolError("reason_codes must be unique")
        object.__setattr__(self, "reason_codes", reasons)
        if self.decision is RuntimeVisualDecision.PATH_MAY_BE_BLOCKED and any(item.geometry_grounded for item in hazards):
            raise RuntimeVisualProtocolError("PATH_MAY_BE_BLOCKED must not claim grounded geometry")
        if self.recommended_action is RuntimeVisualAction.REQUEST_REPLAN and not any(item.geometry_grounded for item in hazards):
            raise RuntimeVisualProtocolError("REQUEST_REPLAN requires grounded hazard geometry")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "review_id": self.review_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "observation_timestamp_s": self.observation_timestamp_s,
            "frame_id": self.frame_id,
            "decision": self.decision.value,
            "task_progress_assessment": self.task_progress_assessment.to_dict(),
            "target_assessment": self.target_assessment.to_dict(),
            "hazards": [item.to_dict() for item in self.hazards],
            "recommended_action": self.recommended_action.value,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RuntimeVisualAssessmentV2":
        required = {
            "schema_version", "review_id", "mission_id", "uav_id",
            "plan_version", "observation_timestamp_s", "frame_id", "decision",
            "task_progress_assessment", "target_assessment", "hazards",
            "recommended_action", "reason_codes",
        }
        _strict(data, required, "RuntimeVisualAssessmentV2")
        hazards, reasons = data["hazards"], data["reason_codes"]
        if isinstance(hazards, (str, bytes)) or not isinstance(hazards, Sequence):
            raise TypeError("hazards must be an array")
        if isinstance(reasons, (str, bytes)) or not isinstance(reasons, Sequence):
            raise TypeError("reason_codes must be an array")
        return cls(
            review_id=data["review_id"],  # type: ignore[arg-type]
            mission_id=data["mission_id"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            plan_version=data["plan_version"],  # type: ignore[arg-type]
            observation_timestamp_s=data["observation_timestamp_s"],  # type: ignore[arg-type]
            frame_id=data["frame_id"],  # type: ignore[arg-type]
            decision=data["decision"],  # type: ignore[arg-type]
            task_progress_assessment=TaskProgressAssessment.from_dict(data["task_progress_assessment"]),  # type: ignore[arg-type]
            target_assessment=RuntimeTargetAssessment.from_dict(data["target_assessment"]),  # type: ignore[arg-type]
            hazards=tuple(RuntimeHazardAssessment.from_dict(item) for item in hazards),  # type: ignore[arg-type]
            recommended_action=data["recommended_action"],  # type: ignore[arg-type]
            reason_codes=tuple(reasons),  # type: ignore[arg-type]
            schema_version=data["schema_version"],  # type: ignore[arg-type]
        )


def build_runtime_visual_assessment_v2_schema(
    *,
    review_id: str,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    frame_id: str,
    observation_timestamp_s: float,
) -> dict[str, object]:
    """Build the frozen routed response schema used by Qwen structured output."""

    validate_review_id(review_id)
    validate_mission_id(mission_id)
    validate_uav_id(uav_id)
    validate_routing_id(frame_id, "frame_id")
    hazard = {
        "type": "object",
        "additionalProperties": False,
        "required": ["obstacle_id", "present", "blocks_active_corridor", "visual_confidence", "geometry_grounded"],
        "properties": {
            "obstacle_id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_.-]{0,63}$"},
            "present": {"type": "boolean"},
            "blocks_active_corridor": {"type": "boolean"},
            "visual_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "geometry_grounded": {"type": "boolean"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "review_id", "mission_id", "uav_id", "plan_version", "observation_timestamp_s", "frame_id", "decision", "task_progress_assessment", "target_assessment", "hazards", "recommended_action", "reason_codes"],
        "properties": {
            "schema_version": {"type": "integer", "const": 2},
            "review_id": {"type": "string", "const": review_id},
            "mission_id": {"type": "string", "const": mission_id},
            "uav_id": {"type": "string", "const": uav_id},
            "plan_version": {"type": "integer", "const": plan_version},
            "observation_timestamp_s": {"type": "number", "const": observation_timestamp_s},
            "frame_id": {"type": "string", "const": frame_id},
            "decision": {"type": "string", "enum": [item.value for item in RuntimeVisualDecision]},
            "task_progress_assessment": {
                "type": "object", "additionalProperties": False,
                "required": ["current_step_consistent", "current_step_blocked", "original_mission_still_achievable"],
                "properties": {name: {"type": "boolean"} for name in ("current_step_consistent", "current_step_blocked", "original_mission_still_achievable")},
            },
            "target_assessment": {
                "type": "object", "additionalProperties": False,
                "required": ["status", "candidate_present", "bbox_xyxy_normalized"],
                "properties": {
                    "status": {"type": "string", "enum": [item.value for item in TargetAssessmentStatus]},
                    "candidate_present": {"type": "boolean"},
                    "bbox_xyxy_normalized": {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0, "maximum": 1}}]},
                },
            },
            "hazards": {"type": "array", "maxItems": 32, "items": hazard},
            "recommended_action": {"type": "string", "enum": [item.value for item in RuntimeVisualAction]},
            "reason_codes": {"type": "array", "maxItems": 32, "items": {"type": "string", "pattern": "^[A-Z][A-Z0-9_]{0,63}$"}},
        },
    }


class QwenRuntimeVisualVerifierV2:
    """Build and parse non-blocking multimodal runtime-assessment requests."""

    SYSTEM_PROMPT = (
        "Assess the current mission using only the supplied camera frames and "
        "bounded task-level summaries. Do not infer hidden obstacle depth. "
        "Only mark geometry_grounded when a supplied privileged camera-visible "
        "observation grounds that obstacle. Never emit world coordinates, "
        "velocities, control commands, or chain-of-thought. Return one JSON "
        "object matching the schema."
    )

    def __init__(self, *, max_image_side_px: int = 1024, jpeg_quality: int = 80) -> None:
        if isinstance(max_image_side_px, bool) or not isinstance(max_image_side_px, int) or not 32 <= max_image_side_px <= 4096:
            raise ValueError("max_image_side_px must be an integer within [32, 4096]")
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be an integer within [1, 95]")
        self._max_image_side_px = max_image_side_px
        self._jpeg_quality = jpeg_quality

    def build_async_request(
        self,
        review_input: RuntimeVisualReviewInputV2,
        *,
        request_id: str,
    ) -> AsyncModelRequest:
        if not isinstance(review_input, RuntimeVisualReviewInputV2):
            raise TypeError("review_input must be RuntimeVisualReviewInputV2")
        request_id = validate_routing_id(request_id, "request_id")
        schema = build_runtime_visual_assessment_v2_schema(
            review_id=review_input.review_id,
            mission_id=review_input.mission_id,
            uav_id=review_input.uav_id,
            plan_version=review_input.plan_version,
            frame_id=review_input.frame_id,
            observation_timestamp_s=review_input.observation_timestamp_s,
        )
        parts: list[TextContentPart | ImageURLContentPart] = [
            TextContentPart(json.dumps(review_input.text_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        ]
        for frame in review_input.frames:
            parts.append(ImageURLContentPart(encode_rgb_to_data_url(
                frame.rgb,
                max_side_px=self._max_image_side_px,
                jpeg_quality=self._jpeg_quality,
            )))
        return AsyncModelRequest(
            request_id=request_id,
            review_id=review_input.review_id,
            mission_id=review_input.mission_id,
            uav_id=review_input.uav_id,
            plan_version=review_input.plan_version,
            observation_timestamp_s=review_input.observation_timestamp_s,
            frame_id=review_input.frame_id,
            messages=(ChatMessage("system", self.SYSTEM_PROMPT), ChatMessage("user", tuple(parts))),
            options=GenerationOptions(
                temperature=0.0,
                max_tokens=768,
                response_format=JsonSchemaResponseFormat("runtime_visual_assessment_v2", schema),
            ),
        )

    def parse_async_result(
        self,
        result: AsyncModelResult,
        *,
        expectation: RuntimeVisualReviewInputV2,
    ) -> RuntimeVisualAssessmentV2:
        if not isinstance(result, AsyncModelResult) or not isinstance(expectation, RuntimeVisualReviewInputV2):
            raise TypeError("result and expectation must use typed V2 contracts")
        for name in ("review_id", "mission_id", "uav_id", "plan_version", "frame_id"):
            if getattr(result, name) != getattr(expectation, name):
                raise RuntimeVisualProtocolError(f"routing mismatch: {name}")
        if result.stale or abs(result.observation_timestamp_s - expectation.observation_timestamp_s) > 1e-9:
            raise RuntimeVisualProtocolError("runtime visual response is stale")
        if result.response is None:
            raise ModelProtocolError(f"runtime visual request failed: {result.error_code or 'UNKNOWN'}")
        try:
            parsed = json.loads(
                result.response.content,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")),
                object_pairs_hook=_strict_json_object,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelProtocolError(f"invalid runtime visual JSON: {exc}") from exc
        assessment = RuntimeVisualAssessmentV2.from_dict(parsed)
        for name in ("review_id", "mission_id", "uav_id", "plan_version", "frame_id", "observation_timestamp_s"):
            if getattr(assessment, name) != getattr(expectation, name):
                raise RuntimeVisualProtocolError(f"assessment routing mismatch: {name}")
        return assessment


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


__all__ = [
    "CompletedStepSummary",
    "CurrentStepSummary",
    "PlanProgressSummary",
    "RemainingStepSummary",
    "RuntimeHazardAssessment",
    "RuntimeSafetyState",
    "RuntimeTargetAssessment",
    "RuntimeVisualAction",
    "RuntimeVisualAssessmentV2",
    "RuntimeVisualDecision",
    "RuntimeVisualProtocolError",
    "RuntimeVisualReviewInputV2",
    "TargetAssessmentStatus",
    "TaskProgressAssessment",
    "build_runtime_visual_assessment_v2_schema",
    "QwenRuntimeVisualVerifierV2",
]
