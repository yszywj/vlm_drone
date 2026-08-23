"""Bounded, image-free logs for dynamic visual missions.

The logger intentionally exposes typed records instead of accepting arbitrary
payload mappings.  This keeps base64 images, prompts, model responses and
environment objects out of persistent logs by construction.  Every append is
flushed immediately and every stream has both a record and byte budget.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
from math import dist, isfinite
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import IO

from common.ids import (
    validate_mission_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from perception.visual_review import (
    VisualReviewParseErrorCode,
    VisualReviewStaleReason,
)
from runtime.events import json_payload_to_dict, validated_json_payload


DEFAULT_MAX_RECORDS_PER_STREAM = 10_000
DEFAULT_MAX_BYTES_PER_STREAM = 16 * 1024 * 1024
MAX_RECORD_BYTES = 8 * 1024


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _positive_plan_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("plan_version must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("plan_version must be a positive integer")
    return normalized


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _bounded_text(value: object, name: str, *, maximum: int = 256) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    if any(character in normalized for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{name} must be a single line")
    # This catches image data URLs early even if somebody tries to disguise
    # one as a reason or event label.
    if "base64," in normalized.casefold() or normalized.casefold().startswith("data:image/"):
        raise ValueError(f"{name} must not contain image/base64 data")
    return normalized


def _optional_routing_id(value: str | None, name: str) -> str | None:
    return None if value is None else validate_routing_id(value, name)


def _optional_finite_nonnegative(value: object, name: str) -> float | None:
    return None if value is None else _finite_nonnegative(value, name)


def _safe_json_object(value: object, name: str) -> Mapping[str, object]:
    frozen = validated_json_payload(value, field_name=name)

    def inspect(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                lowered = str(key).casefold()
                if lowered in {
                    "image",
                    "images",
                    "image_url",
                    "rgb_image",
                    "pixels",
                    "camera_pixels",
                }:
                    raise ValueError(f"{path}.{key} must not contain image data")
                inspect(nested, f"{path}.{key}")
        elif isinstance(item, tuple):
            for index, nested in enumerate(item):
                inspect(nested, f"{path}[{index}]")
        elif isinstance(item, str):
            lowered = item.casefold()
            if "base64," in lowered or lowered.startswith("data:image/"):
                raise ValueError(f"{path} must not contain image/base64 data")

    inspect(frozen, name)
    return frozen


def _bbox(
    value: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("bbox_xyxy_normalized must be a four-number tuple or None")
    normalized = tuple(
        _finite_nonnegative(component, f"bbox_xyxy_normalized[{index}]")
        for index, component in enumerate(value)
    )
    if any(component > 1.0 for component in normalized):
        raise ValueError("bbox_xyxy_normalized values must be within [0, 1]")
    x1, y1, x2, y2 = normalized
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox_xyxy_normalized must satisfy x1 < x2 and y1 < y2")
    return x1, y1, x2, y2


@dataclass(frozen=True, slots=True)
class QwenReviewLogRecord:
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    frame_id: str
    observation_timestamp_s: float
    decision: str
    bbox_xyxy_normalized: tuple[float, float, float, float] | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    stale: bool = False
    accepted: bool = False
    timeout: bool = False
    request_id: str | None = None
    step_id: str | None = None
    semantic_source: str = "qwen_vl"
    geometry_source: str = "none"
    error_code: str | None = None
    stale_reasons: tuple[str, ...] = ()
    response_text_length: int | None = None
    response_text_tail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _positive_plan_version(self.plan_version))
        object.__setattr__(self, "frame_id", validate_routing_id(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "observation_timestamp_s",
            _finite_nonnegative(self.observation_timestamp_s, "observation_timestamp_s"),
        )
        object.__setattr__(self, "decision", _bounded_text(self.decision, "decision", maximum=64))
        object.__setattr__(self, "bbox_xyxy_normalized", _bbox(self.bbox_xyxy_normalized))
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(self, "latency_s", _finite_nonnegative(self.latency_s, "latency_s"))
        for name in ("stale", "accepted", "timeout"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(self, "request_id", _optional_routing_id(self.request_id, "request_id"))
        object.__setattr__(self, "step_id", _optional_routing_id(self.step_id, "step_id"))
        semantic_source = _bounded_text(
            self.semantic_source,
            "semantic_source",
            maximum=64,
        )
        if semantic_source != "qwen_vl":
            raise ValueError("semantic_source must be qwen_vl")
        geometry_source = _bounded_text(
            self.geometry_source,
            "geometry_source",
            maximum=64,
        )
        if geometry_source not in {
            "none",
            "oracle_evaluation",
            "ideal_camera_obstacle_perception",
        }:
            raise ValueError(
                "geometry_source must be none, oracle_evaluation, or "
                "ideal_camera_obstacle_perception"
            )
        object.__setattr__(self, "semantic_source", semantic_source)
        object.__setattr__(self, "geometry_source", geometry_source)
        if self.error_code is not None:
            object.__setattr__(
                self,
                "error_code",
                _bounded_text(self.error_code, "error_code", maximum=64),
            )
        reasons = tuple(self.stale_reasons)
        allowed_reasons = {item.value for item in VisualReviewStaleReason}
        if any(
            not isinstance(reason, str) or reason not in allowed_reasons
            for reason in reasons
        ):
            raise ValueError("stale_reasons contains an unsupported reason")
        if len(set(reasons)) != len(reasons):
            raise ValueError("stale_reasons must not contain duplicates")
        if reasons and not self.stale:
            raise ValueError("stale_reasons require stale=True")
        object.__setattr__(self, "stale_reasons", reasons)
        if (self.response_text_length is None) != (self.response_text_tail is None):
            raise ValueError(
                "response_text_length and response_text_tail must be set together"
            )
        if self.response_text_length is not None:
            object.__setattr__(
                self,
                "response_text_length",
                _nonnegative_int(self.response_text_length, "response_text_length"),
            )
            if not isinstance(self.response_text_tail, str):
                raise TypeError("response_text_tail must be a string")
            if len(self.response_text_tail) > 500:
                raise ValueError("response_text_tail must contain at most 500 characters")
            lowered = self.response_text_tail.casefold()
            if any(
                forbidden in lowered
                for forbidden in ("base64,", "data:image/", "api_key", "authorization")
            ):
                raise ValueError(
                    "response_text_tail must not contain image or credential data"
                )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "request_id": self.request_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "step_id": self.step_id,
            "frame_id": self.frame_id,
            "observation_timestamp_s": self.observation_timestamp_s,
            "decision": self.decision,
            "semantic_source": self.semantic_source,
            "geometry_source": self.geometry_source,
            "bbox_xyxy_normalized": (
                None if self.bbox_xyxy_normalized is None else list(self.bbox_xyxy_normalized)
            ),
            "token_usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "latency_s": self.latency_s,
            "stale": self.stale,
            "stale_reasons": list(self.stale_reasons),
            "accepted": self.accepted,
            "timeout": self.timeout,
            "error_code": self.error_code,
            "response_text_length": self.response_text_length,
            "response_text_tail": self.response_text_tail,
        }


class ModelCallKind(str, Enum):
    INITIAL_PLANNER = "initial_planner"
    NEXT_BEST_VIEW = "next_best_view"
    CLASSICAL_ROUTE_PLANNER = "classical_route_planner"
    ROUTE_PLANNER = "route_planner"
    ROUTE_REPAIR = "route_repair"
    PLAN_REVISION = "plan_revision"


@dataclass(frozen=True, slots=True)
class ModelProposalLogRecord:
    """One image-free model proposal paired with its trusted Critic result."""

    proposal_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    timestamp_s: float
    call_kind: ModelCallKind
    proposal_index: int
    proposal: Mapping[str, object]
    critique: Mapping[str, object] | None = None
    shadow_strict_critique: Mapping[str, object] | None = None
    route_id: str | None = None
    final_proposal: bool = False
    latency_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposal_id",
            validate_routing_id(self.proposal_id, "proposal_id"),
        )
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _positive_plan_version(self.plan_version))
        object.__setattr__(
            self,
            "timestamp_s",
            _finite_nonnegative(self.timestamp_s, "timestamp_s"),
        )
        if not isinstance(self.call_kind, ModelCallKind):
            try:
                object.__setattr__(self, "call_kind", ModelCallKind(self.call_kind))
            except (TypeError, ValueError):
                raise ValueError("call_kind is unsupported") from None
        object.__setattr__(
            self,
            "proposal_index",
            _nonnegative_int(self.proposal_index, "proposal_index"),
        )
        object.__setattr__(self, "proposal", _safe_json_object(self.proposal, "proposal"))
        if self.critique is not None:
            object.__setattr__(
                self,
                "critique",
                _safe_json_object(self.critique, "critique"),
            )
        if self.shadow_strict_critique is not None:
            object.__setattr__(
                self,
                "shadow_strict_critique",
                _safe_json_object(
                    self.shadow_strict_critique,
                    "shadow_strict_critique",
                ),
            )
        object.__setattr__(self, "route_id", _optional_routing_id(self.route_id, "route_id"))
        if not isinstance(self.final_proposal, bool):
            raise TypeError("final_proposal must be a boolean")
        object.__setattr__(
            self,
            "latency_s",
            _optional_finite_nonnegative(self.latency_s, "latency_s"),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "timestamp_s": self.timestamp_s,
            "call_kind": self.call_kind.value,
            "proposal_index": self.proposal_index,
            "route_id": self.route_id,
            "proposal": json_payload_to_dict(self.proposal),
            "critique": (
                None
                if self.critique is None
                else json_payload_to_dict(self.critique)
            ),
            "shadow_strict_critique": (
                None
                if self.shadow_strict_critique is None
                else json_payload_to_dict(self.shadow_strict_critique)
            ),
            "final_proposal": self.final_proposal,
            "latency_s": self.latency_s,
        }


@dataclass(frozen=True, slots=True)
class SearchRunMetrics:
    region_shape: str
    search_strategy: str
    coverage_ratio: float
    visited_viewpoint_count: int
    target_detection_time_s: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "region_shape",
            _bounded_text(self.region_shape, "region_shape", maximum=32),
        )
        object.__setattr__(
            self,
            "search_strategy",
            _bounded_text(self.search_strategy, "search_strategy", maximum=64),
        )
        coverage = _finite_nonnegative(self.coverage_ratio, "coverage_ratio")
        if coverage > 1.0:
            raise ValueError("coverage_ratio must be within [0, 1]")
        object.__setattr__(self, "coverage_ratio", coverage)
        object.__setattr__(
            self,
            "visited_viewpoint_count",
            _nonnegative_int(
                self.visited_viewpoint_count,
                "visited_viewpoint_count",
            ),
        )
        object.__setattr__(
            self,
            "target_detection_time_s",
            _optional_finite_nonnegative(
                self.target_detection_time_s,
                "target_detection_time_s",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "region_shape": self.region_shape,
            "search_strategy": self.search_strategy,
            "coverage_ratio": self.coverage_ratio,
            "visited_viewpoint_count": self.visited_viewpoint_count,
            "target_detection_time_s": self.target_detection_time_s,
        }


@dataclass(frozen=True, slots=True)
class RunManifestMetadata:
    """Launch provenance fields required by the V3 experiment contract."""

    experiment_mode: str = "unspecified"
    route_planner_backend: str = "unspecified"
    planning_contract: str = "unspecified"
    runtime_program: str = "linear"
    route_validation_mode: str = "unspecified"
    obstacle_perception_mode: str = "unspecified"
    prompt_schema_versions: Mapping[str, object] = field(default_factory=dict)
    git_commit: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "experiment_mode",
            "route_planner_backend",
            "planning_contract",
            "runtime_program",
            "route_validation_mode",
            "obstacle_perception_mode",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_text(getattr(self, name), name, maximum=64),
            )
        if self.experiment_mode not in {
            "unspecified",
            "scripted_baseline",
            "classical_baseline",
            "qwen_open_sim",
            "qwen_critic_sim",
            "qwen_strict",
        }:
            raise ValueError("experiment_mode is unsupported")
        if self.route_planner_backend not in {
            "unspecified",
            "none",
            "classical",
            "qwen",
        }:
            raise ValueError("route_planner_backend is unsupported")
        versions = _safe_json_object(
            dict(self.prompt_schema_versions),
            "prompt_schema_versions",
        )
        if any(isinstance(value, (Mapping, tuple)) for value in versions.values()):
            raise ValueError("prompt_schema_versions values must be scalar")
        object.__setattr__(self, "prompt_schema_versions", versions)
        if self.git_commit is not None:
            if (
                not isinstance(self.git_commit, str)
                or len(self.git_commit) != 40
                or any(character not in "0123456789abcdefABCDEF" for character in self.git_commit)
            ):
                raise ValueError("git_commit must be a 40-character hexadecimal SHA or None")
            object.__setattr__(self, "git_commit", self.git_commit.lower())

    def to_manifest_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "experiment_mode": self.experiment_mode,
            "route_planner_backend": self.route_planner_backend,
            "planning_contract": self.planning_contract,
            "runtime_program": self.runtime_program,
            "route_validation_mode": self.route_validation_mode,
            "obstacle_perception_mode": self.obstacle_perception_mode,
            "prompt_schema_versions": json_payload_to_dict(
                self.prompt_schema_versions
            ),
        }
        # An unspecified logger-level value must not overwrite the launcher's
        # independently collected read-only git provenance during dict merge.
        if self.git_commit is not None:
            result["git_commit"] = self.git_commit
        return result


@dataclass(frozen=True, slots=True)
class MissionEventLogRecord:
    timestamp_s: float
    mission_id: str
    uav_id: str
    plan_version: int
    step_id: str
    skill: str
    event: str
    status: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite_nonnegative(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _positive_plan_version(self.plan_version))
        object.__setattr__(self, "step_id", validate_routing_id(self.step_id, "step_id"))
        for name in ("skill", "event", "status", "reason"):
            object.__setattr__(self, name, _bounded_text(getattr(self, name), name))

    def to_csv_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _MISSION_EVENT_FIELDS}

    def to_terminal_line(self) -> str:
        """Render the mandatory routing context for one compact terminal line."""

        return (
            f"[MISSION] mission_id={self.mission_id} uav_id={self.uav_id} "
            f"plan_version={self.plan_version} step_id={self.step_id} "
            f"skill={self.skill} event={self.event} status={self.status} "
            f"reason={self.reason}"
        )


@dataclass(frozen=True, slots=True)
class SkillTransitionLogRecord:
    timestamp_s: float
    mission_id: str
    uav_id: str
    plan_version: int
    step_id: str
    old_skill: str
    new_skill: str
    old_status: str
    result_code: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite_nonnegative(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _positive_plan_version(self.plan_version))
        object.__setattr__(self, "step_id", validate_routing_id(self.step_id, "step_id"))
        for name in ("old_skill", "new_skill", "old_status", "result_code", "reason"):
            object.__setattr__(self, name, _bounded_text(getattr(self, name), name))

    def to_csv_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _SKILL_TRANSITION_FIELDS}


@dataclass(frozen=True, slots=True)
class VisualRunStats:
    review_count: int
    accepted_count: int
    stale_count: int
    timeout_count: int
    plan_revision_count: int
    hover_count: int
    hover_total_time_s: float
    dropped_log_record_count: int
    valid_review_count: int = 0
    initial_planner_model_calls: int = 0
    next_best_view_model_calls: int = 0
    visual_review_parse_errors: int = 0
    route_planner_model_calls: int = 0
    classical_route_planner_calls: int = 0
    route_repair_model_calls: int = 0
    plan_revision_model_calls: int = 0
    hold_trigger_source: str | None = None
    hazard_detection_latency_s: float | None = None
    hold_establishment_latency_s: float | None = None
    route_planning_latency_s: float | None = None
    route_repair_count: int = 0
    route_length_m: float | None = None
    path_length_m: float | None = None
    minimum_route_clearance_m: float | None = None
    collision_count: int = 0
    invalid_waypoint_count: int = 0
    shadow_strict_route_valid: bool | None = None
    final_plan_version: int | None = None
    search_metrics: SearchRunMetrics | None = None
    manifest_metadata: RunManifestMetadata = field(default_factory=RunManifestMetadata)

    def to_manifest_dict(
        self,
        *,
        debug_images_count: int = 0,
        debug_images_bytes: int = 0,
    ) -> dict[str, object]:
        search = (
            {
                "region_shape": None,
                "search_strategy": None,
                "coverage_ratio": None,
                "visited_viewpoint_count": 0,
                "target_detection_time_s": None,
            }
            if self.search_metrics is None
            else self.search_metrics.to_dict()
        )
        result = {
            "qwen_visual_reviews": {
                "count": self.review_count,
                "accepted": self.accepted_count,
                "stale": self.stale_count,
                "timeout": self.timeout_count,
            },
            "plan_revisions": self.plan_revision_count,
            "initial_planner_model_calls": self.initial_planner_model_calls,
            "next_best_view_model_calls": self.next_best_view_model_calls,
            "visual_review_model_calls": self.review_count,
            "visual_review_valid_results": self.valid_review_count,
            "visual_review_stale_results": self.stale_count,
            "visual_review_parse_errors": self.visual_review_parse_errors,
            "route_planner_model_calls": self.route_planner_model_calls,
            "classical_route_planner_calls": self.classical_route_planner_calls,
            "route_repair_model_calls": self.route_repair_model_calls,
            "plan_revision_model_calls": self.plan_revision_model_calls,
            "hold_trigger_source": self.hold_trigger_source,
            "hazard_detection_latency_s": self.hazard_detection_latency_s,
            "hold_establishment_latency_s": self.hold_establishment_latency_s,
            "route_planning_latency_s": self.route_planning_latency_s,
            "route_repair_count": self.route_repair_count,
            "route_length_m": self.route_length_m,
            "path_length_m": self.path_length_m,
            "minimum_route_clearance_m": self.minimum_route_clearance_m,
            "collision_count": self.collision_count,
            "invalid_waypoint_count": self.invalid_waypoint_count,
            # This is populated only by an independent STRICT shadow check.
            # It is intentionally unrelated to open_sim execution acceptance.
            "shadow_strict_route_valid": self.shadow_strict_route_valid,
            "route_validity_source": "shadow_strict_route_valid",
            "plan_revision_count": self.plan_revision_count,
            "final_plan_version": self.final_plan_version,
            "search": search,
            **search,
            "supervisory_hover": {
                "count": self.hover_count,
                "total_time_s": self.hover_total_time_s,
            },
            "debug_images": {
                "count": _nonnegative_int(debug_images_count, "debug_images_count"),
                "bytes": _nonnegative_int(debug_images_bytes, "debug_images_bytes"),
            },
            "dropped_log_records": self.dropped_log_record_count,
        }
        result.update(self.manifest_metadata.to_manifest_dict())
        return result


_MISSION_EVENT_FIELDS = (
    "timestamp_s",
    "mission_id",
    "uav_id",
    "plan_version",
    "step_id",
    "skill",
    "event",
    "status",
    "reason",
)
_SKILL_TRANSITION_FIELDS = (
    "timestamp_s",
    "mission_id",
    "uav_id",
    "plan_version",
    "step_id",
    "old_skill",
    "new_skill",
    "old_status",
    "result_code",
    "reason",
)


@dataclass(slots=True)
class _StreamBudget:
    path: Path
    stream: IO[str]
    record_count: int
    byte_count: int


class SparseMissionLogger:
    """Write fixed-schema, immediately flushed and bounded sparse streams."""

    def __init__(
        self,
        logs_dir: str | Path,
        *,
        max_records_per_stream: int = DEFAULT_MAX_RECORDS_PER_STREAM,
        max_bytes_per_stream: int = DEFAULT_MAX_BYTES_PER_STREAM,
        manifest_metadata: RunManifestMetadata | None = None,
    ) -> None:
        self._max_records = _nonnegative_int(max_records_per_stream, "max_records_per_stream")
        self._max_bytes = _nonnegative_int(max_bytes_per_stream, "max_bytes_per_stream")
        if self._max_records == 0 or self._max_bytes < MAX_RECORD_BYTES:
            raise ValueError(
                f"max_records_per_stream must be positive and max_bytes_per_stream at least {MAX_RECORD_BYTES}"
            )
        if self._max_records > 1_000_000 or self._max_bytes > 1_073_741_824:
            raise ValueError("log budgets exceed the trusted hard cap")
        directory = Path(logs_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise ValueError("logs_dir must be a directory")
        self._lock = RLock()
        self._closed = False
        if manifest_metadata is not None and not isinstance(
            manifest_metadata, RunManifestMetadata
        ):
            raise TypeError("manifest_metadata must be RunManifestMetadata or None")
        self._manifest_metadata = manifest_metadata or RunManifestMetadata()
        self._review_count = 0
        self._accepted_count = 0
        self._valid_review_count = 0
        self._stale_count = 0
        self._timeout_count = 0
        self._visual_review_parse_errors = 0
        self._initial_planner_model_calls = 0
        self._next_best_view_model_calls = 0
        self._route_planner_model_calls = 0
        self._classical_route_planner_calls = 0
        self._route_repair_model_calls = 0
        self._plan_revision_model_calls = 0
        self._plan_revision_count = 0
        self._hover_count = 0
        self._hover_total_time_s = 0.0
        self._hold_trigger_source: str | None = None
        self._hazard_detection_latency_s: float | None = None
        self._hold_establishment_latency_s: float | None = None
        self._route_planning_latency_s: float | None = None
        self._route_repair_count = 0
        self._route_length_m: float | None = None
        self._path_length_m: float | None = None
        self._last_path_xyz_m: tuple[float, float, float] | None = None
        self._minimum_route_clearance_m: float | None = None
        self._collision_count = 0
        self._invalid_waypoint_count = 0
        self._shadow_strict_route_valid: bool | None = None
        self._final_plan_version: int | None = None
        self._search_metrics: SearchRunMetrics | None = None
        self._dropped = 0
        opened: list[_StreamBudget] = []
        try:
            self._qwen = self._open_stream(directory / "qwen_reviews.jsonl")
            opened.append(self._qwen)
            if self._qwen.record_count > self._max_records:
                raise ValueError(
                    f"existing log exceeds configured record budget: {self._qwen.path}"
                )
            self._proposals = self._open_stream(directory / "model_proposals.jsonl")
            opened.append(self._proposals)
            if self._proposals.record_count > self._max_records:
                raise ValueError(
                    f"existing log exceeds configured record budget: {self._proposals.path}"
                )
            self._mission = self._open_stream(directory / "mission_events.csv")
            opened.append(self._mission)
            self._transitions = self._open_stream(directory / "skill_transitions.csv")
            opened.append(self._transitions)
        except Exception:
            for budget in opened:
                budget.stream.close()
            raise
        self._mission_writer = csv.DictWriter(self._mission.stream, fieldnames=_MISSION_EVENT_FIELDS)
        self._transition_writer = csv.DictWriter(
            self._transitions.stream, fieldnames=_SKILL_TRANSITION_FIELDS
        )
        self._ensure_csv_header(self._mission, self._mission_writer)
        self._ensure_csv_header(self._transitions, self._transition_writer)

    @staticmethod
    def _count_existing_records(path: Path) -> int:
        if not path.exists() or path.stat().st_size == 0:
            return 0
        with path.open("r", encoding="utf-8", newline="") as stream:
            return sum(1 for _ in stream)

    def _open_stream(self, path: Path) -> _StreamBudget:
        byte_count = path.stat().st_size if path.exists() else 0
        if byte_count > self._max_bytes:
            raise ValueError(f"existing log exceeds configured byte budget: {path}")
        lines = self._count_existing_records(path)
        if lines > self._max_records + 1:
            raise ValueError(f"existing log exceeds configured record budget: {path}")
        stream = path.open("a", encoding="utf-8", newline="")
        return _StreamBudget(path, stream, lines, byte_count)

    def _ensure_csv_header(self, budget: _StreamBudget, writer: csv.DictWriter) -> None:
        if budget.byte_count != 0:
            # Existing header is validated to prevent schema mixing on resume.
            with budget.path.open("r", encoding="utf-8", newline="") as stream:
                first = stream.readline().rstrip("\r\n")
            if first != ",".join(writer.fieldnames):
                self.close()
                raise ValueError(f"existing CSV header does not match schema: {budget.path}")
            budget.record_count = max(0, budget.record_count - 1)
            return
        rendered = ",".join(writer.fieldnames) + "\r\n"
        encoded_size = len(rendered.encode("utf-8"))
        if encoded_size > self._max_bytes:
            self.close()
            raise ValueError("CSV header exceeds byte budget")
        writer.writeheader()
        budget.stream.flush()
        budget.byte_count += encoded_size
        budget.record_count = 0

    def _can_append(self, budget: _StreamBudget, encoded_size: int) -> bool:
        if encoded_size > MAX_RECORD_BYTES:
            raise ValueError(f"log record exceeds {MAX_RECORD_BYTES} byte limit")
        if (
            budget.record_count >= self._max_records
            or budget.byte_count + encoded_size > self._max_bytes
        ):
            self._dropped += 1
            return False
        return True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SparseMissionLogger is closed")

    def log_qwen_review(self, record: QwenReviewLogRecord) -> bool:
        if not isinstance(record, QwenReviewLogRecord):
            raise TypeError("record must be QwenReviewLogRecord")
        rendered = json.dumps(
            record.to_json_dict(), ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ) + "\n"
        size = len(rendered.encode("utf-8"))
        with self._lock:
            self._require_open()
            # Manifest counters describe reviews that occurred, including a
            # record dropped by the bounded persistence budget.
            self._review_count += 1
            self._accepted_count += int(record.accepted)
            self._valid_review_count += int(
                not record.stale
                and not record.timeout
                and record.error_code is None
            )
            self._stale_count += int(record.stale)
            self._timeout_count += int(record.timeout)
            parse_codes = {item.value for item in VisualReviewParseErrorCode}
            self._visual_review_parse_errors += int(
                record.error_code in parse_codes
            )
            if not self._can_append(self._qwen, size):
                return False
            self._qwen.stream.write(rendered)
            self._qwen.stream.flush()
            self._qwen.record_count += 1
            self._qwen.byte_count += size
            return True

    def log_model_proposal(self, record: ModelProposalLogRecord) -> bool:
        """Persist one proposal/Critic pair and count its model call."""

        if not isinstance(record, ModelProposalLogRecord):
            raise TypeError("record must be ModelProposalLogRecord")
        rendered = json.dumps(
            record.to_json_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ) + "\n"
        size = len(rendered.encode("utf-8"))
        with self._lock:
            self._require_open()
            self._record_model_call_unlocked(record.call_kind, 1)
            if record.call_kind is ModelCallKind.ROUTE_REPAIR:
                self._route_repair_count += 1
            if (
                record.call_kind
                in {
                    ModelCallKind.CLASSICAL_ROUTE_PLANNER,
                    ModelCallKind.ROUTE_PLANNER,
                    ModelCallKind.ROUTE_REPAIR,
                }
                and record.latency_s is not None
            ):
                self._route_planning_latency_s = record.latency_s
            if record.critique is not None:
                critique = json_payload_to_dict(record.critique)
                selected_route = bool(record.final_proposal) or (
                    critique.get("status") == "ACCEPT"
                )
                raw_length = critique.get("route_length_m")
                if (
                    selected_route
                    and not isinstance(raw_length, bool)
                    and isinstance(raw_length, Real)
                    and isfinite(raw_length)
                    and raw_length >= 0.0
                ):
                    self._route_length_m = float(raw_length)
                raw_clearance = critique.get("minimum_clearance_m")
                if (
                    selected_route
                    and not isinstance(raw_clearance, bool)
                    and isinstance(raw_clearance, Real)
                    and isfinite(raw_clearance)
                    and raw_clearance >= 0.0
                ):
                    clearance = float(raw_clearance)
                    self._minimum_route_clearance_m = (
                        clearance
                        if self._minimum_route_clearance_m is None
                        else min(self._minimum_route_clearance_m, clearance)
                    )
            if record.final_proposal and record.shadow_strict_critique is not None:
                shadow = json_payload_to_dict(record.shadow_strict_critique)
                shadow_status = shadow.get("status")
                if shadow_status not in {"ACCEPT", "REVISE"}:
                    raise ValueError(
                        "shadow_strict_critique.status must be ACCEPT or REVISE"
                    )
                strict_valid = shadow_status == "ACCEPT"
                self._shadow_strict_route_valid = (
                    strict_valid
                    if self._shadow_strict_route_valid is None
                    else self._shadow_strict_route_valid and strict_valid
                )
            if not self._can_append(self._proposals, size):
                return False
            self._proposals.stream.write(rendered)
            self._proposals.stream.flush()
            self._proposals.record_count += 1
            self._proposals.byte_count += size
            return True

    def _log_csv(
        self,
        budget: _StreamBudget,
        writer: csv.DictWriter,
        row: dict[str, object],
    ) -> bool:
        # Render once into an in-memory buffer so the byte budget is checked
        # before touching the persistent file.
        import io

        buffer = io.StringIO(newline="")
        csv.DictWriter(buffer, fieldnames=writer.fieldnames).writerow(row)
        rendered = buffer.getvalue()
        size = len(rendered.encode("utf-8"))
        if not self._can_append(budget, size):
            return False
        budget.stream.write(rendered)
        budget.stream.flush()
        budget.record_count += 1
        budget.byte_count += size
        return True

    def log_mission_event(self, record: MissionEventLogRecord) -> bool:
        if not isinstance(record, MissionEventLogRecord):
            raise TypeError("record must be MissionEventLogRecord")
        with self._lock:
            self._require_open()
            return self._log_csv(
                self._mission, self._mission_writer, record.to_csv_dict()
            )

    def log_skill_transition(self, record: SkillTransitionLogRecord) -> bool:
        if not isinstance(record, SkillTransitionLogRecord):
            raise TypeError("record must be SkillTransitionLogRecord")
        with self._lock:
            self._require_open()
            return self._log_csv(
                self._transitions,
                self._transition_writer,
                record.to_csv_dict(),
            )

    def record_model_call(
        self,
        kind: ModelCallKind | str,
        *,
        count: int = 1,
    ) -> None:
        """Count a model call that did not produce a persisted proposal."""

        try:
            normalized = kind if isinstance(kind, ModelCallKind) else ModelCallKind(kind)
        except (TypeError, ValueError):
            raise ValueError("kind is unsupported") from None
        amount = _nonnegative_int(count, "count")
        if amount == 0:
            raise ValueError("count must be greater than zero")
        with self._lock:
            self._require_open()
            self._record_model_call_unlocked(normalized, amount)

    def _record_model_call_unlocked(
        self,
        kind: ModelCallKind,
        count: int,
    ) -> None:
        field_by_kind = {
            ModelCallKind.INITIAL_PLANNER: "_initial_planner_model_calls",
            ModelCallKind.NEXT_BEST_VIEW: "_next_best_view_model_calls",
            ModelCallKind.CLASSICAL_ROUTE_PLANNER: "_classical_route_planner_calls",
            ModelCallKind.ROUTE_PLANNER: "_route_planner_model_calls",
            ModelCallKind.ROUTE_REPAIR: "_route_repair_model_calls",
            ModelCallKind.PLAN_REVISION: "_plan_revision_model_calls",
        }
        field_name = field_by_kind[kind]
        setattr(self, field_name, getattr(self, field_name) + count)

    def record_initial_planner_model_call(self, *, count: int = 1) -> None:
        self.record_model_call(ModelCallKind.INITIAL_PLANNER, count=count)

    def record_route_planner_model_call(self, *, count: int = 1) -> None:
        self.record_model_call(ModelCallKind.ROUTE_PLANNER, count=count)

    def record_route_repair_model_call(self, *, count: int = 1) -> None:
        self.record_model_call(ModelCallKind.ROUTE_REPAIR, count=count)

    def record_plan_revision_model_call(self, *, count: int = 1) -> None:
        self.record_model_call(ModelCallKind.PLAN_REVISION, count=count)

    def record_plan_revision(self) -> None:
        with self._lock:
            self._require_open()
            self._plan_revision_count += 1

    def set_final_plan_version(self, plan_version: int) -> None:
        normalized = _positive_plan_version(plan_version)
        with self._lock:
            self._require_open()
            self._final_plan_version = normalized

    def record_hold_metrics(
        self,
        *,
        trigger_source: str,
        hazard_detection_latency_s: float | None = None,
        hold_establishment_latency_s: float | None = None,
    ) -> None:
        source = _bounded_text(trigger_source, "trigger_source", maximum=128)
        hazard_latency = _optional_finite_nonnegative(
            hazard_detection_latency_s,
            "hazard_detection_latency_s",
        )
        hold_latency = _optional_finite_nonnegative(
            hold_establishment_latency_s,
            "hold_establishment_latency_s",
        )
        with self._lock:
            self._require_open()
            self._hold_trigger_source = source
            if hazard_latency is not None:
                self._hazard_detection_latency_s = hazard_latency
            if hold_latency is not None:
                self._hold_establishment_latency_s = hold_latency

    def record_route_metrics(
        self,
        *,
        planning_latency_s: float | None = None,
        route_length_m: float | None = None,
        minimum_clearance_m: float | None = None,
    ) -> None:
        latency = _optional_finite_nonnegative(
            planning_latency_s,
            "planning_latency_s",
        )
        length = _optional_finite_nonnegative(route_length_m, "route_length_m")
        clearance = _optional_finite_nonnegative(
            minimum_clearance_m,
            "minimum_clearance_m",
        )
        with self._lock:
            self._require_open()
            if latency is not None:
                self._route_planning_latency_s = latency
            if length is not None:
                self._route_length_m = length
            if clearance is not None:
                self._minimum_route_clearance_m = (
                    clearance
                    if self._minimum_route_clearance_m is None
                    else min(self._minimum_route_clearance_m, clearance)
                )

    def record_route_repair(self, *, count: int = 1) -> None:
        amount = _nonnegative_int(count, "count")
        if amount == 0:
            raise ValueError("count must be greater than zero")
        with self._lock:
            self._require_open()
            self._route_repair_count += amount

    def record_path_position(self, position_xyz_m: object) -> None:
        """Accumulate executed 3-D path length from trusted pose samples."""

        if not isinstance(position_xyz_m, (tuple, list)) or len(position_xyz_m) != 3:
            raise TypeError("position_xyz_m must contain three finite numbers")
        values: list[float] = []
        for item in position_xyz_m:
            if isinstance(item, bool) or not isinstance(item, Real):
                raise TypeError("position_xyz_m must contain three finite numbers")
            value = float(item)
            if not isfinite(value):
                raise ValueError("position_xyz_m must contain three finite numbers")
            values.append(value)
        point = (values[0], values[1], values[2])
        with self._lock:
            self._require_open()
            if self._last_path_xyz_m is None:
                self._path_length_m = 0.0
            else:
                assert self._path_length_m is not None
                self._path_length_m += dist(self._last_path_xyz_m, point)
            self._last_path_xyz_m = point

    def record_collision(self, *, count: int = 1) -> None:
        amount = _nonnegative_int(count, "count")
        if amount == 0:
            raise ValueError("count must be greater than zero")
        with self._lock:
            self._require_open()
            self._collision_count += amount

    def record_invalid_waypoints(self, count: int = 1) -> None:
        amount = _nonnegative_int(count, "count")
        if amount == 0:
            raise ValueError("count must be greater than zero")
        with self._lock:
            self._require_open()
            self._invalid_waypoint_count += amount

    def record_shadow_strict_route_validity(self, valid: bool) -> None:
        """Store a route result produced by an independent STRICT evaluator.

        Execution acceptance, registry publication, and the configured route
        validation mode are not accepted as substitutes for this measurement.
        Multiple obstacle routes in one mission are reduced by conjunction:
        the episode is route-valid only when every independently evaluated
        executed/final route is STRICT-valid.
        """

        if not isinstance(valid, bool):
            raise TypeError("valid must be a boolean")
        with self._lock:
            self._require_open()
            self._shadow_strict_route_valid = (
                valid
                if self._shadow_strict_route_valid is None
                else self._shadow_strict_route_valid and valid
            )

    def record_search_metrics(
        self,
        *,
        region_shape: str,
        search_strategy: str,
        coverage_ratio: float,
        visited_viewpoint_count: int,
        target_detection_time_s: float | None,
    ) -> None:
        metrics = SearchRunMetrics(
            region_shape=region_shape,
            search_strategy=search_strategy,
            coverage_ratio=coverage_ratio,
            visited_viewpoint_count=visited_viewpoint_count,
            target_detection_time_s=target_detection_time_s,
        )
        with self._lock:
            self._require_open()
            self._search_metrics = metrics

    def record_supervisory_hover(self, duration_s: float) -> None:
        """Compatibility helper for one already-completed HOVER interval."""

        duration = _finite_nonnegative(duration_s, "duration_s")
        with self._lock:
            self._require_open()
            self._hover_count += 1
            self._hover_total_time_s += duration

    def record_supervisory_hover_started(self) -> None:
        """Count HOVER immediately so an interrupted run is still truthful."""

        with self._lock:
            self._require_open()
            self._hover_count += 1

    def record_supervisory_hover_duration(self, duration_s: float) -> None:
        """Add elapsed time when a previously counted HOVER interval exits."""

        duration = _finite_nonnegative(duration_s, "duration_s")
        with self._lock:
            self._require_open()
            self._hover_total_time_s += duration

    def snapshot(self) -> VisualRunStats:
        with self._lock:
            return VisualRunStats(
                review_count=self._review_count,
                accepted_count=self._accepted_count,
                stale_count=self._stale_count,
                timeout_count=self._timeout_count,
                plan_revision_count=self._plan_revision_count,
                hover_count=self._hover_count,
                hover_total_time_s=self._hover_total_time_s,
                dropped_log_record_count=self._dropped,
                valid_review_count=self._valid_review_count,
                initial_planner_model_calls=self._initial_planner_model_calls,
                next_best_view_model_calls=self._next_best_view_model_calls,
                visual_review_parse_errors=self._visual_review_parse_errors,
                route_planner_model_calls=self._route_planner_model_calls,
                classical_route_planner_calls=self._classical_route_planner_calls,
                route_repair_model_calls=self._route_repair_model_calls,
                plan_revision_model_calls=self._plan_revision_model_calls,
                hold_trigger_source=self._hold_trigger_source,
                hazard_detection_latency_s=self._hazard_detection_latency_s,
                hold_establishment_latency_s=self._hold_establishment_latency_s,
                route_planning_latency_s=self._route_planning_latency_s,
                route_repair_count=self._route_repair_count,
                route_length_m=self._route_length_m,
                path_length_m=self._path_length_m,
                minimum_route_clearance_m=self._minimum_route_clearance_m,
                collision_count=self._collision_count,
                invalid_waypoint_count=self._invalid_waypoint_count,
                shadow_strict_route_valid=self._shadow_strict_route_valid,
                final_plan_version=self._final_plan_version,
                search_metrics=self._search_metrics,
                manifest_metadata=self._manifest_metadata,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for budget in (
                self._qwen,
                self._proposals,
                self._mission,
                self._transitions,
            ):
                budget.stream.flush()
                budget.stream.close()
            self._closed = True

    def __enter__(self) -> "SparseMissionLogger":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "DEFAULT_MAX_BYTES_PER_STREAM",
    "DEFAULT_MAX_RECORDS_PER_STREAM",
    "MAX_RECORD_BYTES",
    "MissionEventLogRecord",
    "ModelCallKind",
    "ModelProposalLogRecord",
    "QwenReviewLogRecord",
    "RunManifestMetadata",
    "SearchRunMetrics",
    "SkillTransitionLogRecord",
    "SparseMissionLogger",
    "VisualRunStats",
]
