"""Concrete multimodal Qwen visual-review request and response boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from numbers import Real

import numpy as np

from common.ids import (
    validate_mission_id,
    validate_request_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
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
from perception.visual_review import (
    QwenVisualReview,
    VisualReviewExpectation,
    VisualReviewProtocolError,
    build_qwen_visual_review_json_schema,
)
from runtime.frame_store import FrameRef
from runtime.events import MissionEventType
from target import TargetSnapshot, TargetSpec


_ALLOWED_FEEDBACK_KEYS = frozenset(
    {"progress", "message", "phase", "elapsed_time", "waypoint_index", "waypoint_count"}
)
_ALLOWED_CONTEXT_KEYS = frozenset(
    {
        "mission_elapsed_s",
        "visibility_summary",
        "lighting_summary",
        "weather_summary",
        "trigger_event_type",
        "detector_candidate_id",
        "detector_candidate_source",
        "detector_candidate_frame_id",
        "detector_candidate_timestamp_s",
        "detector_candidate_x1",
        "detector_candidate_y1",
        "detector_candidate_x2",
        "detector_candidate_y2",
    }
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _timestamp(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > 512:
        raise ValueError(f"{name} must contain at most 512 characters")
    return normalized


def _safe_summary(
    value: Mapping[str, object],
    *,
    allowed_keys: frozenset[str],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    unknown = set(value) - allowed_keys
    if unknown:
        raise ValueError(f"{name} contains unsupported fields")
    result: dict[str, object] = {}
    for key, item in value.items():
        if item is None or isinstance(item, (str, bool, int)):
            result[key] = item
        elif isinstance(item, float) and isfinite(item):
            result[key] = item
        else:
            raise TypeError(f"{name}.{key} must be a JSON scalar")
    return result


@dataclass(frozen=True, slots=True)
class VisualReviewFrame:
    ref: FrameRef
    rgb: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.ref, FrameRef):
            raise TypeError("ref must be a FrameRef")
        if (
            not isinstance(self.rgb, np.ndarray)
            or self.rgb.dtype != np.uint8
            or self.rgb.ndim != 3
            or self.rgb.shape[2] != 3
        ):
            raise ValueError("rgb must be a uint8 array with shape (H, W, 3)")
        if self.rgb.shape[:2] != (self.ref.height, self.ref.width):
            raise ValueError("rgb shape must match FrameRef dimensions")
        copied = np.ascontiguousarray(self.rgb).copy()
        copied.setflags(write=False)
        object.__setattr__(self, "rgb", copied)


@dataclass(frozen=True, slots=True)
class VisualReviewInput:
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    observation_timestamp_s: float
    frame_id: str
    target_spec: TargetSpec
    current_skill: str
    current_step_id: str
    frames: tuple[VisualReviewFrame, ...]
    target_snapshot: TargetSnapshot
    skill_feedback_summary: Mapping[str, object]
    environment_context: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "plan_version",
            _positive_int(self.plan_version, "plan_version"),
        )
        object.__setattr__(
            self,
            "observation_timestamp_s",
            _timestamp(self.observation_timestamp_s, "observation_timestamp_s"),
        )
        object.__setattr__(
            self,
            "frame_id",
            validate_routing_id(self.frame_id, "frame_id"),
        )
        if not isinstance(self.target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        object.__setattr__(self, "current_skill", _text(self.current_skill, "current_skill"))
        object.__setattr__(
            self,
            "current_step_id",
            validate_routing_id(self.current_step_id, "current_step_id"),
        )
        frames = tuple(self.frames)
        if not 1 <= len(frames) <= 3:
            raise ValueError("frames must contain between one and three frames")
        if any(
            not isinstance(frame, VisualReviewFrame)
            or frame.ref.uav_id != self.uav_id
            for frame in frames
        ):
            raise ValueError("every review frame must belong to the request uav_id")
        if frames[-1].ref.frame_id != self.frame_id:
            raise ValueError("the latest frame must match frame_id")
        if abs(frames[-1].ref.timestamp_s - self.observation_timestamp_s) > 1e-9:
            raise ValueError("the latest frame timestamp must match the observation")
        object.__setattr__(self, "frames", frames)
        if not isinstance(self.target_snapshot, TargetSnapshot):
            raise TypeError("target_snapshot must be a TargetSnapshot")
        object.__setattr__(
            self,
            "skill_feedback_summary",
            _safe_summary(
                self.skill_feedback_summary,
                allowed_keys=_ALLOWED_FEEDBACK_KEYS,
                name="skill_feedback_summary",
            ),
        )
        safe_context = _safe_summary(
            self.environment_context,
            allowed_keys=_ALLOWED_CONTEXT_KEYS,
            name="environment_context",
        )
        trigger_event_type = safe_context.get("trigger_event_type")
        if trigger_event_type is not None:
            if not isinstance(trigger_event_type, str):
                raise TypeError(
                    "environment_context.trigger_event_type must be a string"
                )
            try:
                MissionEventType(trigger_event_type)
            except ValueError:
                raise ValueError(
                    "environment_context.trigger_event_type is not supported"
                ) from None
        object.__setattr__(self, "environment_context", safe_context)

    @property
    def expectation(self) -> VisualReviewExpectation:
        return VisualReviewExpectation(
            review_id=self.review_id,
            mission_id=self.mission_id,
            uav_id=self.uav_id,
            plan_version=self.plan_version,
            observation_timestamp_s=self.observation_timestamp_s,
            frame_id=self.frame_id,
        )


class QwenVLMVerifier:
    """Build routed asynchronous Qwen requests and parse strict reviews."""

    SYSTEM_PROMPT = (
        "Review only the supplied camera frames against the immutable target "
        "identity. Do not guess invisible details. If the target is too small, "
        "return AMBIGUOUS. A single-frame color resemblance is insufficient "
        "for identity confirmation. Emit exactly one JSON object matching the "
        "provided schema. Never output world coordinates, velocities, flight "
        "commands, or control gains."
    )

    def __init__(
        self,
        *,
        max_image_side_px: int = 1024,
        jpeg_quality: int = 80,
    ) -> None:
        self._max_image_side_px = _positive_int(
            max_image_side_px,
            "max_image_side_px",
        )
        if (
            isinstance(jpeg_quality, bool)
            or not isinstance(jpeg_quality, int)
            or not 1 <= jpeg_quality <= 95
        ):
            raise ValueError("jpeg_quality must be an integer within [1, 95]")
        self._jpeg_quality = jpeg_quality

    def build_async_request(
        self,
        review_input: VisualReviewInput,
        *,
        request_id: str,
    ) -> AsyncModelRequest:
        if not isinstance(review_input, VisualReviewInput):
            raise TypeError("review_input must be a VisualReviewInput")
        request_id = validate_request_id(request_id)
        schema = build_qwen_visual_review_json_schema(
            review_id=review_input.review_id,
            mission_id=review_input.mission_id,
            uav_id=review_input.uav_id,
            plan_version=review_input.plan_version,
            frame_id=review_input.frame_id,
            observation_timestamp_s=review_input.observation_timestamp_s,
        )
        payload = {
            "routing": {
                "review_id": review_input.review_id,
                "mission_id": review_input.mission_id,
                "uav_id": review_input.uav_id,
                "plan_version": review_input.plan_version,
                "frame_id": review_input.frame_id,
                "observation_timestamp_s": review_input.observation_timestamp_s,
            },
            "target_spec": review_input.target_spec.to_dict(),
            "current_skill": review_input.current_skill,
            "current_step_id": review_input.current_step_id,
            "target_state": {
                "target_id": review_input.target_snapshot.target_id,
                "lifecycle": review_input.target_snapshot.lifecycle.value,
                "confidence": review_input.target_snapshot.confidence,
            },
            "skill_feedback": dict(review_input.skill_feedback_summary),
            "environment_context": dict(review_input.environment_context),
        }
        parts: list[TextContentPart | ImageURLContentPart] = [
            TextContentPart(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        ]
        for frame in review_input.frames:
            parts.append(
                ImageURLContentPart(
                    encode_rgb_to_data_url(
                        frame.rgb,
                        max_side_px=self._max_image_side_px,
                        jpeg_quality=self._jpeg_quality,
                    )
                )
            )
        # Prompt contents never grant scheduler priority.  The verifier emits
        # the conservative periodic default; VisualReviewCoordinator replaces
        # it from its trusted ReviewTicket when this is event/runtime work.
        return AsyncModelRequest(
            request_id=request_id,
            review_id=review_input.review_id,
            mission_id=review_input.mission_id,
            uav_id=review_input.uav_id,
            plan_version=review_input.plan_version,
            observation_timestamp_s=review_input.observation_timestamp_s,
            frame_id=review_input.frame_id,
            messages=(
                ChatMessage("system", self.SYSTEM_PROMPT),
                ChatMessage("user", tuple(parts)),
            ),
            options=GenerationOptions(
                temperature=0.0,
                response_format=JsonSchemaResponseFormat(
                    "qwen_visual_review_v1",
                    schema,
                ),
            ),
            broker_priority=4,
            broker_replaceable=True,
        )

    def parse_async_result(
        self,
        result: AsyncModelResult,
        *,
        expectation: VisualReviewExpectation,
    ) -> QwenVisualReview:
        if not isinstance(result, AsyncModelResult):
            raise TypeError("result must be an AsyncModelResult")
        if not isinstance(expectation, VisualReviewExpectation):
            raise TypeError("expectation must be a VisualReviewExpectation")
        for name in ("review_id", "mission_id", "uav_id"):
            if getattr(result, name) != getattr(expectation, name):
                raise VisualReviewProtocolError(
                    f"async visual review routing mismatch: {name}"
                )
        if (
            result.stale
            or result.plan_version != expectation.plan_version
            or result.frame_id != expectation.frame_id
            or abs(
                result.observation_timestamp_s
                - expectation.observation_timestamp_s
            )
            > 1e-9
        ):
            raise VisualReviewProtocolError("async visual review result is stale")
        if result.response is None:
            raise ModelProtocolError(
                f"visual review request failed: {result.error_code or 'UNKNOWN'}"
            )
        try:
            parsed = json.loads(
                result.response.content,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_strict_object,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            raise ModelProtocolError("visual review response is not strict JSON") from None
        try:
            review = QwenVisualReview.from_dict(parsed)
        except (TypeError, ValueError):
            raise ModelProtocolError("visual review response violates its schema") from None
        # Reuse the same routing check used by gate/shadow consumers.
        if (
            review.review_id != expectation.review_id
            or review.mission_id != expectation.mission_id
            or review.uav_id != expectation.uav_id
            or review.plan_version != expectation.plan_version
            or review.frame_id != expectation.frame_id
            or abs(
                review.observation_timestamp_s
                - expectation.observation_timestamp_s
            )
            > 1e-9
        ):
            raise VisualReviewProtocolError("visual review response metadata mismatch")
        return review


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field is forbidden")
        result[key] = value
    return result


__all__ = [
    "QwenVLMVerifier",
    "VisualReviewFrame",
    "VisualReviewInput",
]
