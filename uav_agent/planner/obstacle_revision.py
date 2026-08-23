"""Multimodal Qwen route proposals and counterexample-driven repair sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from numbers import Real

from common.ids import (
    validate_mission_id,
    validate_request_id,
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
from perception.qwen_vlm_verifier import VisualReviewFrame
from perception.runtime_visual_assessment import RuntimeVisualAssessmentV2
from planner.route_critic import (
    RouteCritic,
    RouteCritique,
    RouteCriticStatus,
    RouteValidationContext,
    RouteValidationMode,
)
from planner.route_types import (
    AvoidanceStrategy,
    AvoidanceStrategyType,
    RouteConstraints,
    RouteDraft,
    RouteWaypoint,
)
from planner.spatial import CoordinateFrame, PointTarget
from runtime.events import json_payload_to_dict, validated_json_payload


class ObstacleRevisionError(ValueError):
    """Raised for invalid routing, geometry, or proposal state."""


_PARSE_REPAIR_ERROR_CODES = frozenset(
    {
        "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE",
        "ROUTE_CONTRACT_ERROR",
        "ROUTE_FIRST_STEP_NOT_FOLLOW_ROUTE",
        "ROUTE_JSON_INVALID",
        "ROUTE_REPLACEMENT_STEP_IDS_DUPLICATE",
        "ROUTE_SCHEMA_TYPE_ERROR",
        "ROUTE_SCHEMA_VALUE_ERROR",
        "ROUTE_SCHEMA_VERSION_INVALID",
        "ROUTE_TRUSTED_METADATA_MISMATCH",
        "ROUTE_TERMINAL_SUFFIX_INVALID",
        "ROUTE_WAYPOINT_BUDGET_EXCEEDED",
        "ROUTE_WAYPOINT_IDS_DUPLICATE",
    }
)
_PARSE_REPAIR_OUTPUT_KINDS = frozenset(
    {
        "JSON_OBJECT",
        "TEXT_TAIL",
        "OMITTED_REPETITION_RISK",
        "OMITTED_SENSITIVE",
        "OMITTED_UNAVAILABLE",
    }
)
_PARSE_REPAIR_MAX_JSON_CHARS = 16_384
_PARSE_REPAIR_MAX_TEXT_CHARS = 500
_PARSE_REPAIR_SENSITIVE_MARKERS = (
    "data:image/",
    "base64,",
    "authorization",
    "api_key",
    "api-key",
    "apikey",
    "access_token",
    "access-token",
    "bearer ",
    "credential",
    "private_key",
    "password",
    "secret",
    "request_id",
    "request-id",
    "sk-",
)


def is_repairable_obstacle_parse_error(error_code: object) -> bool:
    """Return whether a model-output error may consume another proposal slot."""

    return isinstance(error_code, str) and error_code in _PARSE_REPAIR_ERROR_CODES


def _repair_feedback_contains_sensitive(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = key.casefold().replace("-", "_")
            if any(marker.replace("-", "_") in normalized for marker in _PARSE_REPAIR_SENSITIVE_MARKERS):
                return True
            if _repair_feedback_contains_sensitive(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_repair_feedback_contains_sensitive(item) for item in value)
    if isinstance(value, str):
        lowered = value.casefold()
        return any(marker in lowered for marker in _PARSE_REPAIR_SENSITIVE_MARKERS)
    return False


@dataclass(frozen=True, slots=True)
class ObstacleParseRepairFeedback:
    """Bounded, image-free feedback for one rejected model response.

    This type intentionally cannot carry an ``AsyncModelRequest``, a frame, or
    arbitrary bytes.  It is the only structural-repair value accepted by the
    planner request builder.
    """

    error_code: str
    rejected_output_kind: str
    rejected_model_output: Mapping[str, object] | str | None
    response_text_length: int
    response_text_truncated: bool
    repair_attempt_index: int = 1
    repeated_unchanged: bool = False

    def __post_init__(self) -> None:
        if not is_repairable_obstacle_parse_error(self.error_code):
            raise ObstacleRevisionError("parse repair error_code is not repairable")
        if self.rejected_output_kind not in _PARSE_REPAIR_OUTPUT_KINDS:
            raise ObstacleRevisionError("rejected_output_kind is not supported")
        if (
            isinstance(self.response_text_length, bool)
            or not isinstance(self.response_text_length, int)
            or self.response_text_length < 0
            or self.response_text_length > 1_000_000_000
        ):
            raise ObstacleRevisionError("response_text_length is out of range")
        if not isinstance(self.response_text_truncated, bool):
            raise TypeError("response_text_truncated must be bool")
        if (
            isinstance(self.repair_attempt_index, bool)
            or not isinstance(self.repair_attempt_index, int)
            or not 1 <= self.repair_attempt_index <= 3
        ):
            raise ObstacleRevisionError("repair_attempt_index must be within [1, 3]")
        if not isinstance(self.repeated_unchanged, bool):
            raise TypeError("repeated_unchanged must be bool")

        output = self.rejected_model_output
        if self.rejected_output_kind == "JSON_OBJECT":
            if not isinstance(output, Mapping):
                raise TypeError("JSON_OBJECT repair feedback must be an object")
            normalized = validated_json_payload(
                output,
                field_name="rejected_model_output",
            )
            encoded = json.dumps(
                normalized,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if len(encoded) > _PARSE_REPAIR_MAX_JSON_CHARS:
                raise ObstacleRevisionError("rejected_model_output exceeds repair bound")
            if _repair_feedback_contains_sensitive(normalized):
                raise ObstacleRevisionError("rejected_model_output contains sensitive content")
            object.__setattr__(self, "rejected_model_output", normalized)
            return
        if self.rejected_output_kind == "TEXT_TAIL":
            if not isinstance(output, str) or not output:
                raise TypeError("TEXT_TAIL repair feedback must be non-empty text")
            if len(output) > _PARSE_REPAIR_MAX_TEXT_CHARS:
                raise ObstacleRevisionError("repair response tail exceeds bound")
            if _repair_feedback_contains_sensitive(output):
                raise ObstacleRevisionError("repair response tail contains sensitive content")
            return
        if output is not None:
            raise ObstacleRevisionError("omitted repair feedback cannot carry model output")

    def to_prompt_dict(self) -> dict[str, object]:
        output = self.rejected_model_output
        if isinstance(output, Mapping):
            output = json_payload_to_dict(output)
        return {
            "parse_error_code": self.error_code,
            "required_corrections": list(
                _parse_repair_required_corrections(self.error_code)
            ),
            "rejected_output_kind": self.rejected_output_kind,
            "rejected_model_output": output,
            "response_text_length": self.response_text_length,
            "response_text_truncated": self.response_text_truncated,
            "repair_attempt_index": self.repair_attempt_index,
            "repeated_unchanged": self.repeated_unchanged,
        }


def _parse_repair_required_corrections(error_code: str) -> tuple[str, ...]:
    """Return structural counterexamples without inventing route geometry."""

    common = (
        "Generate a new complete proposal; do not copy the rejected proposal unchanged.",
        "The runtime will reject rather than edit, clamp, or deduplicate model waypoints.",
    )
    specific = {
        "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE": (
            "Every adjacent waypoint xyz_m must differ.",
            "Use active_corridor_rejoin_target only for the final route waypoint; every intermediate waypoint must be a distinct detour coordinate that clears grounded_obstacle_geometry by route_constraints.minimum_clearance_m.",
            "Prefer the fewest necessary waypoints and never pad a route by repeating a coordinate.",
        ),
        "ROUTE_WAYPOINT_IDS_DUPLICATE": (
            "Assign a unique waypoint_id to every route waypoint.",
        ),
        "ROUTE_REPLACEMENT_STEP_IDS_DUPLICATE": (
            "Assign a unique id to every replacement step.",
        ),
        "ROUTE_FIRST_STEP_NOT_FOLLOW_ROUTE": (
            "The first replacement step must be FOLLOW_ROUTE and route_ref must exactly equal routing.route_id.",
        ),
        "ROUTE_WAYPOINT_BUDGET_EXCEEDED": (
            "Reduce the waypoint count to route_constraints.max_waypoints or fewer without changing trusted routing fields.",
        ),
        "ROUTE_SCHEMA_VERSION_INVALID": (
            "Echo schema_version=3 and every trusted routing/version field exactly.",
        ),
        "ROUTE_TRUSTED_METADATA_MISMATCH": (
            "Echo every trusted routing/version/route/replace_from identifier exactly.",
        ),
        "ROUTE_TERMINAL_SUFFIX_INVALID": (
            "End replacement_steps with required_terminal_suffix in exactly the listed skill order.",
            "Use args={} for every required terminal suffix step so the compiler reuses its trusted parameters.",
            "Include exactly one LAND, as the final replacement step.",
        ),
    }.get(error_code, ())
    return (*specific, *common)


class ObstacleRevisionSessionState(str, Enum):
    AWAITING_PROPOSAL = "AWAITING_PROPOSAL"
    AWAITING_REPAIR = "AWAITING_REPAIR"
    ACCEPTED = "ACCEPTED"
    EXHAUSTED = "EXHAUSTED"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ObstacleRevisionError(f"{name} must be finite")
    return normalized


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ObstacleRevisionError(f"{name} must contain three numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _strict(data: Mapping[str, object], fields: set[str], name: str) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be an object")
    missing, unknown = fields - set(data), set(data) - fields
    if missing:
        raise ObstacleRevisionError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ObstacleRevisionError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


@dataclass(frozen=True, slots=True)
class GroundedObstacleGeometry:
    obstacle_id: str
    frame: CoordinateFrame
    relative_aabb_min_m: tuple[float, float, float]
    relative_aabb_max_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "obstacle_id", validate_routing_id(self.obstacle_id, "obstacle_id"))
        if self.frame is not CoordinateFrame.UAV_HOLD_FLU:
            raise ObstacleRevisionError("grounded obstacle geometry must use UAV_HOLD_FLU")
        minimum = _vector3(self.relative_aabb_min_m, "relative_aabb_min_m")
        maximum = _vector3(self.relative_aabb_max_m, "relative_aabb_max_m")
        if any(low >= high for low, high in zip(minimum, maximum)):
            raise ObstacleRevisionError("relative AABB bounds must be strictly ordered")
        object.__setattr__(self, "relative_aabb_min_m", minimum)
        object.__setattr__(self, "relative_aabb_max_m", maximum)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame.value,
            "obstacle_id": self.obstacle_id,
            "relative_aabb_min_m": list(self.relative_aabb_min_m),
            "relative_aabb_max_m": list(self.relative_aabb_max_m),
        }


@dataclass(frozen=True, slots=True)
class ObstacleReplacementStep:
    step_id: str
    uav_id: str
    skill: str
    args: Mapping[str, object]

    _ALLOWED = frozenset({"FOLLOW_ROUTE", "GOTO", "HOVER", "SEARCH", "TRACK", "LAND"})
    _TARGET_CONTINUATION_KEY = "target_continuation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", validate_routing_id(self.step_id, "step_id"))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if not isinstance(self.skill, str) or self.skill not in self._ALLOWED:
            raise ObstacleRevisionError("replacement skill is not supported")
        normalized = validated_json_payload(self.args, field_name="replacement args")
        continuation = normalized.get(self._TARGET_CONTINUATION_KEY)
        if continuation is not None:
            if set(normalized) != {self._TARGET_CONTINUATION_KEY}:
                raise ObstacleRevisionError(
                    "target_continuation cannot be combined with controller args"
                )
            allowed = {
                "SEARCH": frozenset({"RESTART_SEARCH"}),
                "TRACK": frozenset({"CONTINUE_TRACK", "REACQUIRE"}),
            }.get(self.skill, frozenset())
            if not isinstance(continuation, str) or continuation not in allowed:
                raise ObstacleRevisionError(
                    f"target_continuation is invalid for replacement {self.skill}"
                )
        object.__setattr__(self, "args", normalized)

    @property
    def target_continuation(self) -> str | None:
        value = self.args.get(self._TARGET_CONTINUATION_KEY)
        return value if isinstance(value, str) else None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ObstacleReplacementStep":
        _strict(data, {"id", "uav_id", "skill", "args"}, "replacement step")
        return cls(data["id"], data["uav_id"], data["skill"], data["args"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"id": self.step_id, "uav_id": self.uav_id, "skill": self.skill, "args": deepcopy(dict(self.args))}


@dataclass(frozen=True, slots=True)
class ObstacleRouteRevisionDraft:
    mission_id: str
    uav_id: str
    base_plan_version: int
    new_plan_version: int
    replace_from_step_id: str
    avoidance_strategy: AvoidanceStrategy
    route_draft: RouteDraft
    replacement_steps: tuple[ObstacleReplacementStep, ...]
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 3 or isinstance(self.schema_version, bool):
            raise ObstacleRevisionError("schema_version must equal 3")
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        for name in ("base_plan_version", "new_plan_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ObstacleRevisionError(f"{name} must be a positive integer")
        if self.new_plan_version != self.base_plan_version + 1:
            raise ObstacleRevisionError("new_plan_version must equal base_plan_version + 1")
        object.__setattr__(self, "replace_from_step_id", validate_routing_id(self.replace_from_step_id, "replace_from_step_id"))
        if not isinstance(self.avoidance_strategy, AvoidanceStrategy) or not isinstance(self.route_draft, RouteDraft):
            raise TypeError("avoidance_strategy and route_draft must be typed")
        if self.route_draft.frame is not CoordinateFrame.UAV_HOLD_FLU:
            raise ObstacleRevisionError("obstacle route must use UAV_HOLD_FLU")
        steps = tuple(self.replacement_steps)
        if not 1 <= len(steps) <= 16 or any(not isinstance(item, ObstacleReplacementStep) for item in steps):
            raise ObstacleRevisionError("replacement_steps must contain 1..16 typed steps")
        if any(item.uav_id != self.uav_id for item in steps):
            raise ObstacleRevisionError("every replacement step must echo uav_id")
        if steps[0].skill != "FOLLOW_ROUTE" or steps[0].args.get("route_ref") != self.route_draft.route_id:
            raise ObstacleRevisionError("replacement must begin with FOLLOW_ROUTE referencing route_draft")
        ids = [item.step_id for item in steps]
        if len(ids) != len(set(ids)):
            raise ObstacleRevisionError("replacement step IDs must be unique")
        object.__setattr__(self, "replacement_steps", steps)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ObstacleRouteRevisionDraft":
        fields = {"schema_version", "mission_id", "uav_id", "base_plan_version", "new_plan_version", "replace_from_step_id", "avoidance_strategy", "route_draft", "replacement_steps"}
        _strict(data, fields, "ObstacleRouteRevisionDraft")
        strategy_data = data["avoidance_strategy"]
        _strict(strategy_data, {"strategy", "rejoin_target", "reason_codes"}, "avoidance_strategy")  # type: ignore[arg-type]
        reasons = strategy_data["reason_codes"]  # type: ignore[index]
        if isinstance(reasons, (str, bytes)) or not isinstance(reasons, Sequence):
            raise TypeError("reason_codes must be an array")
        raw_steps = data["replacement_steps"]
        if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
            raise TypeError("replacement_steps must be an array")
        return cls(
            mission_id=data["mission_id"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            base_plan_version=data["base_plan_version"],  # type: ignore[arg-type]
            new_plan_version=data["new_plan_version"],  # type: ignore[arg-type]
            replace_from_step_id=data["replace_from_step_id"],  # type: ignore[arg-type]
            avoidance_strategy=AvoidanceStrategy(
                strategy_data["strategy"],  # type: ignore[index,arg-type]
                strategy_data["rejoin_target"],  # type: ignore[index,arg-type]
                tuple(reasons),  # type: ignore[arg-type]
            ),
            route_draft=RouteDraft.from_dict(data["route_draft"]),  # type: ignore[arg-type]
            replacement_steps=tuple(ObstacleReplacementStep.from_dict(item) for item in raw_steps),  # type: ignore[arg-type]
            schema_version=data["schema_version"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "base_plan_version": self.base_plan_version,
            "new_plan_version": self.new_plan_version,
            "replace_from_step_id": self.replace_from_step_id,
            "avoidance_strategy": self.avoidance_strategy.to_dict(),
            "route_draft": self.route_draft.to_dict(),
            "replacement_steps": [item.to_dict() for item in self.replacement_steps],
        }


@dataclass(frozen=True, slots=True)
class ObstacleAwareRevisionRequest:
    mission_id: str
    uav_id: str
    base_plan_version: int
    new_plan_version: int
    route_id: str
    replace_from_step_id: str
    original_instruction: str
    original_plan_summary: Mapping[str, object]
    completed_prefix_summary: tuple[Mapping[str, object], ...]
    current_step_summary: Mapping[str, object]
    remaining_plan_summary: tuple[Mapping[str, object], ...]
    frames: tuple[VisualReviewFrame, ...]
    visual_assessment: RuntimeVisualAssessmentV2
    grounded_obstacle_geometry: GroundedObstacleGeometry
    active_corridor_rejoin_target: PointTarget
    route_constraints: RouteConstraints
    mission_elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if self.new_plan_version != self.base_plan_version + 1:
            raise ObstacleRevisionError("new_plan_version must equal base_plan_version + 1")
        object.__setattr__(self, "route_id", validate_routing_id(self.route_id, "route_id"))
        object.__setattr__(self, "replace_from_step_id", validate_routing_id(self.replace_from_step_id, "replace_from_step_id"))
        if not isinstance(self.original_instruction, str) or not self.original_instruction.strip() or len(self.original_instruction) > 2048:
            raise ObstacleRevisionError("original_instruction must be bounded non-empty text")
        for name in ("original_plan_summary", "current_step_summary"):
            object.__setattr__(self, name, validated_json_payload(getattr(self, name), field_name=name))
        for name in ("completed_prefix_summary", "remaining_plan_summary"):
            values = tuple(getattr(self, name))
            if len(values) > 100:
                raise ObstacleRevisionError(f"{name} is unbounded")
            object.__setattr__(self, name, tuple(validated_json_payload(item, field_name=name) for item in values))
        frames = tuple(self.frames)
        if not 1 <= len(frames) <= 3 or any(not isinstance(item, VisualReviewFrame) or item.ref.uav_id != self.uav_id for item in frames):
            raise ObstacleRevisionError("frames must contain 1..3 values for the bound UAV")
        object.__setattr__(self, "frames", frames)
        if not isinstance(self.visual_assessment, RuntimeVisualAssessmentV2) or self.visual_assessment.uav_id != self.uav_id or self.visual_assessment.plan_version != self.base_plan_version:
            raise ObstacleRevisionError("visual assessment routing/version mismatch")
        if not isinstance(self.grounded_obstacle_geometry, GroundedObstacleGeometry):
            raise TypeError("grounded_obstacle_geometry must be typed")
        if not isinstance(self.active_corridor_rejoin_target, PointTarget):
            raise TypeError("active_corridor_rejoin_target must be a PointTarget")
        if self.active_corridor_rejoin_target.frame is not CoordinateFrame.UAV_HOLD_FLU:
            raise ObstacleRevisionError(
                "active_corridor_rejoin_target must use UAV_HOLD_FLU"
            )
        if not isinstance(self.route_constraints, RouteConstraints):
            raise TypeError("route_constraints must be RouteConstraints")
        elapsed = _finite(self.mission_elapsed_s, "mission_elapsed_s")
        if elapsed < 0.0:
            raise ObstacleRevisionError("mission_elapsed_s must be non-negative")
        object.__setattr__(self, "mission_elapsed_s", elapsed)

    def prompt_payload(self) -> dict[str, object]:
        required_terminal_suffix = self.required_terminal_suffix
        return {
            "routing": {"mission_id": self.mission_id, "uav_id": self.uav_id, "base_plan_version": self.base_plan_version, "new_plan_version": self.new_plan_version, "route_id": self.route_id},
            "original_instruction": self.original_instruction,
            # Request construction freezes these summaries at the trust
            # boundary.  ``deepcopy`` cannot reconstruct the immutable nested
            # mappings (and used to fail before the first HTTP submission for
            # any non-trivial runtime plan), so thaw them through the bounded
            # JSON helper instead.
            "original_plan": json_payload_to_dict(self.original_plan_summary),
            "completed_prefix": [
                json_payload_to_dict(item)
                for item in self.completed_prefix_summary
            ],
            "current_step": json_payload_to_dict(self.current_step_summary),
            "remaining_plan": [
                json_payload_to_dict(item)
                for item in self.remaining_plan_summary
            ],
            "required_terminal_suffix": [
                {"skill": skill, "args": {}}
                for skill in required_terminal_suffix
            ],
            "runtime_visual_assessment": self.visual_assessment.to_dict(),
            "grounded_obstacle_geometry": self.grounded_obstacle_geometry.to_dict(),
            "route_start_xyz_m": [0.0, 0.0, 0.0],
            "coordinate_convention": {
                "frame": "UAV_HOLD_FLU",
                "x_axis": "forward",
                "y_axis": "left",
                "z_axis": "up",
                "origin": "current hold pose",
                "altitude_note": (
                    "local z=0 maintains the current world altitude; do not "
                    "copy the mission's world flight altitude into local z"
                ),
            },
            "active_corridor_rejoin_target": (
                self.active_corridor_rejoin_target.to_dict()
            ),
            "route_constraints": self.route_constraints.to_dict(),
            "mission_elapsed_s": self.mission_elapsed_s,
        }

    @property
    def required_terminal_suffix(self) -> tuple[str, ...]:
        """Return the trusted tail that a model replacement must preserve.

        The route itself may substitute an interrupted GOTO. Otherwise the
        last remaining GOTO and every step through terminal LAND form one
        indivisible landing approach, matching the trusted compiler contract.
        """

        remaining_skills = tuple(
            item.get("skill")
            for item in self.remaining_plan_summary
            if isinstance(item.get("skill"), str)
        )
        later_gotos = tuple(
            index
            for index, skill in enumerate(remaining_skills)
            if skill == "GOTO"
        )
        if later_gotos:
            return remaining_skills[later_gotos[-1] :]
        if self.current_step_summary.get("skill") == "GOTO":
            return remaining_skills
        land_indices = tuple(
            index
            for index, skill in enumerate(remaining_skills)
            if skill == "LAND"
        )
        if land_indices:
            return remaining_skills[land_indices[-1] :]
        return ()


class ObstacleAwareRevisionPlanner:
    SYSTEM_PROMPT = (
        "Propose an obstacle detour in UAV_HOLD_FLU. You choose left, right, "
        "above, or backtrack, the exact waypoint count and coordinates, and "
        "where to rejoin the original mission. Output only one complete JSON "
        "object matching the supplied schema, including its trusted routing "
        "fields, avoidance_strategy, route_draft, and replacement_steps. Never output "
        "velocities, PID gains, controller commands, or hidden reasoning. A "
        "critic may return counterexamples; revise the route yourself and do "
        "not assume the runtime will clamp or replace your waypoints. "
        "Every waypoint_id must be unique, every replacement step id must be "
        "unique, and adjacent waypoint xyz_m values must differ. The first "
        "replacement_steps item must be FOLLOW_ROUTE and its route_ref must "
        "equal route_draft.route_id. "
        "For later SEARCH/TRACK/GOTO/LAND steps, args={} means reuse the "
        "trusted compiled parameters of one matching remaining step. "
        "When the interrupted step is SEARCH or TRACK, you may express one "
        "explicit target continuation on the first step after FOLLOW_ROUTE: "
        "SEARCH args={\"target_continuation\":\"RESTART_SEARCH\"}, or TRACK "
        "args={\"target_continuation\":\"CONTINUE_TRACK\"} / "
        "{\"target_continuation\":\"REACQUIRE\"}. REACQUIRE is not a "
        "top-level skill: it requests the interrupted TRACK step's trusted "
        "bounded recovery policy, and the runtime enters it only after TRACK "
        "reports TARGET_LOST with trusted last-seen evidence. "
        "You may add a bounded timed HOVER, but must not invent new GOTO or LAND "
        "parameters in this obstacle patch. LAND must use args={} so it reuses "
        "the trusted terminal landing step. "
        "Copy required_terminal_suffix to the end of replacement_steps in "
        "exactly that skill order, with args={} for each listed step. "
        "TRACK preserves its trusted target-loss recovery, including REACQUIRE. "
        "The final replacement step must be LAND; include exactly one LAND and "
        "no steps after it. Terminate with the trusted return approach. The "
        "last route_draft waypoint must rejoin active_corridor_rejoin_target "
        "within route_constraints.rejoin_tolerance_m; choose every intermediate "
        "detour waypoint yourself. avoidance_strategy.rejoin_target uses the "
        "protocol label original_goto_target for this active corridor endpoint; "
        "it does not mean the home or terminal landing target. "
        "route_start_xyz_m is the implicit route origin; do not copy it as the "
        "first route_draft waypoint. UAV_HOLD_FLU is relative to the aircraft: "
        "local z=0 keeps the current world altitude, while local z=10 climbs "
        "another ten metres."
    )

    def __init__(self, *, max_image_side_px: int = 1024, jpeg_quality: int = 80) -> None:
        if not 32 <= max_image_side_px <= 4096 or not 1 <= jpeg_quality <= 95:
            raise ValueError("image limits are out of range")
        self._max_image_side_px = max_image_side_px
        self._jpeg_quality = jpeg_quality

    def build_async_request(
        self,
        request: ObstacleAwareRevisionRequest,
        *,
        request_id: str,
        prior_proposal: ObstacleRouteRevisionDraft | None = None,
        critique: RouteCritique | None = None,
        parse_repair: ObstacleParseRepairFeedback | None = None,
    ) -> AsyncModelRequest:
        if not isinstance(request, ObstacleAwareRevisionRequest):
            raise TypeError("request must be an ObstacleAwareRevisionRequest")
        request_id = validate_request_id(request_id)
        if (prior_proposal is None) != (critique is None):
            raise ObstacleRevisionError("repair requires both prior_proposal and critique")
        if parse_repair is not None and not isinstance(
            parse_repair,
            ObstacleParseRepairFeedback,
        ):
            raise TypeError("parse_repair must be ObstacleParseRepairFeedback or None")
        if parse_repair is not None and prior_proposal is not None:
            raise ObstacleRevisionError(
                "critic repair and structural parse repair are mutually exclusive"
            )
        payload = request.prompt_payload()
        if prior_proposal is not None:
            if prior_proposal.mission_id != request.mission_id or prior_proposal.route_draft.route_id != request.route_id or critique.route_id != request.route_id:
                raise ObstacleRevisionError("repair proposal/critique routing mismatch")
            payload["counterexample"] = {
                "prior_proposal": prior_proposal.to_dict(),
                "critique": critique.to_dict(),
                "instruction": "Generate a new proposal; do not mutate the previous JSON in place.",
            }
        elif parse_repair is not None:
            payload["counterexample"] = {
                **parse_repair.to_prompt_dict(),
                "instruction": (
                    "The previous model output failed the Python transport "
                    "contract. Generate a complete new proposal yourself. Do "
                    "not ask the runtime to alter, clamp, deduplicate, or "
                    "otherwise rewrite any waypoint."
                ),
            }
        parts: list[TextContentPart | ImageURLContentPart] = [TextContentPart(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))]
        for frame in request.frames:
            parts.append(ImageURLContentPart(encode_rgb_to_data_url(frame.rgb, max_side_px=self._max_image_side_px, jpeg_quality=self._jpeg_quality)))
        system_prompt = self.SYSTEM_PROMPT
        if parse_repair is not None:
            system_prompt = self._structural_repair_system_prompt(parse_repair)
        elif prior_proposal is not None and critique is not None:
            system_prompt = self._critic_repair_system_prompt(critique)
        return AsyncModelRequest(
            request_id=request_id,
            review_id=f"review_route_{request.route_id}",
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            plan_version=request.base_plan_version,
            observation_timestamp_s=request.frames[-1].ref.timestamp_s,
            frame_id=request.frames[-1].ref.frame_id,
            messages=(ChatMessage("system", system_prompt), ChatMessage("user", tuple(parts))),
            options=GenerationOptions(temperature=0.0, max_tokens=1536, response_format=JsonSchemaResponseFormat("obstacle_route_revision_v3", build_obstacle_route_revision_schema(request))),
        )

    def _structural_repair_system_prompt(
        self,
        feedback: ObstacleParseRepairFeedback,
    ) -> str:
        header = (
            f"\nSTRUCTURAL REPAIR ATTEMPT {feedback.repair_attempt_index}: "
            f"the previous response was rejected as {feedback.error_code}. "
        )
        repeated = (
            "The immediately preceding repair repeated the same rejected JSON. "
            if feedback.repeated_unchanged
            else ""
        )
        if feedback.error_code == "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE":
            correction = (
                "Generate a genuinely different waypoint list. The route origin "
                "[0,0,0] is implicit and active_corridor_rejoin_target may appear "
                "only once, as the final waypoint. Every intermediate point must "
                "differ from both and from its neighbors. Calculate at least one "
                "detour point outside the obstacle plus minimum_clearance_m: left "
                "uses y above AABB max, right uses y below AABB min, and above uses "
                "z above AABB max. BACKTRACK must still leave the blocked AABB slab. "
            )
        elif feedback.error_code == "ROUTE_TERMINAL_SUFFIX_INVALID":
            correction = (
                "End replacement_steps by copying required_terminal_suffix exactly, "
                "in order, with args={} for every listed trusted step and exactly "
                "one final LAND. Do not invent a GOTO target or LAND zone. "
            )
        else:
            correction = (
                "Apply every required_corrections item in the counterexample and "
                "generate a complete new proposal rather than copying the old one. "
            )
        return self.SYSTEM_PROMPT + header + repeated + correction

    def _critic_repair_system_prompt(self, critique: RouteCritique) -> str:
        violation_types = tuple(
            dict.fromkeys(item.type.value for item in critique.violations)
        )
        header = (
            "\nCRITIC REPAIR: the previous route was rejected for "
            + ", ".join(violation_types)
            + ". Generate a different complete proposal; the runtime will not "
            "modify any waypoint. "
        )
        corrections: list[str] = []
        if "DOES_NOT_REJOIN_GOAL" in violation_types:
            corrections.append(
                "The previous final waypoint is wrong. For the new final route "
                "waypoint, copy active_corridor_rejoin_target.xyz_m from the "
                "current user payload exactly; do not reuse the previous final "
                "coordinate or the mission's world altitude. "
            )
        if any(
            item in violation_types
            for item in {"PATH_INTERSECTS_OBSTACLE", "INSUFFICIENT_CLEARANCE"}
        ):
            corrections.append(
                "Recalculate intermediate points from grounded_obstacle_geometry "
                "and route_constraints.minimum_clearance_m; do not ask the critic "
                "for replacement coordinates. "
            )
        if not corrections:
            corrections.append(
                "Resolve every structured critique violation yourself while "
                "preserving trusted routing and the required terminal suffix. "
            )
        return self.SYSTEM_PROMPT + header + "".join(corrections)

    def parse_async_result(self, result: AsyncModelResult, *, request: ObstacleAwareRevisionRequest) -> ObstacleRouteRevisionDraft:
        if result.mission_id != request.mission_id or result.uav_id != request.uav_id or result.plan_version != request.base_plan_version:
            raise ObstacleRevisionError("async route result routing/version mismatch")
        if result.stale or result.response is None:
            raise ModelProtocolError(result.error_code or "route revision response is stale")
        try:
            parsed = json.loads(result.response.content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)), object_pairs_hook=_strict_object)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ModelProtocolError(f"invalid obstacle revision JSON: {exc}") from exc
        draft = ObstacleRouteRevisionDraft.from_dict(parsed)
        expected = (request.mission_id, request.uav_id, request.base_plan_version, request.new_plan_version, request.route_id, request.replace_from_step_id)
        actual = (draft.mission_id, draft.uav_id, draft.base_plan_version, draft.new_plan_version, draft.route_draft.route_id, draft.replace_from_step_id)
        if actual != expected:
            raise ObstacleRevisionError("route draft does not echo trusted routing/version/IDs")
        if len(draft.route_draft.waypoints) > request.route_constraints.max_waypoints:
            raise ObstacleRevisionError("route draft exceeds trusted waypoint count")
        _validate_model_terminal_suffix(draft, request)
        return draft


def _validate_model_terminal_suffix(
    draft: ObstacleRouteRevisionDraft,
    request: ObstacleAwareRevisionRequest,
) -> None:
    skills = tuple(step.skill for step in draft.replacement_steps)
    if skills[-1:] != ("LAND",) or skills.count("LAND") != 1:
        raise ObstacleRevisionError(
            "replacement terminal suffix must contain exactly one final LAND"
        )
    required = request.required_terminal_suffix
    if not required or len(skills) < len(required) + 1:
        raise ObstacleRevisionError(
            "replacement terminal suffix does not preserve required trusted steps"
        )
    if skills[-len(required) :] != required:
        raise ObstacleRevisionError(
            "replacement terminal suffix does not match required trusted skill order"
        )
    terminal_steps = draft.replacement_steps[-len(required) :]
    if any(step.args for step in terminal_steps):
        raise ObstacleRevisionError(
            "replacement terminal suffix must use empty args for trusted reuse"
        )


@dataclass(frozen=True, slots=True)
class ObstacleRevisionAttempt:
    proposal: ObstacleRouteRevisionDraft
    critique: RouteCritique


class ObstacleRevisionSession:
    """Bound proposal/critique history; the caller owns all Qwen invocations."""

    def __init__(self, *, mode: RouteValidationMode | str, max_proposals: int = 3) -> None:
        if isinstance(max_proposals, bool) or not isinstance(max_proposals, int) or not 1 <= max_proposals <= 3:
            raise ValueError("max_proposals must be an integer within [1, 3]")
        self._critic = RouteCritic(mode)
        self._max_proposals = max_proposals
        self._attempts: list[ObstacleRevisionAttempt] = []
        self._state = ObstacleRevisionSessionState.AWAITING_PROPOSAL

    @property
    def state(self) -> ObstacleRevisionSessionState:
        return self._state

    @property
    def attempts(self) -> tuple[ObstacleRevisionAttempt, ...]:
        return tuple(self._attempts)

    @property
    def accepted_proposal(self) -> ObstacleRouteRevisionDraft | None:
        return self._attempts[-1].proposal if self._state is ObstacleRevisionSessionState.ACCEPTED else None

    def evaluate(self, proposal: ObstacleRouteRevisionDraft, context: RouteValidationContext) -> RouteCritique:
        if self._state not in {ObstacleRevisionSessionState.AWAITING_PROPOSAL, ObstacleRevisionSessionState.AWAITING_REPAIR}:
            raise ObstacleRevisionError("revision session is terminal")
        if self._attempts:
            first = self._attempts[0].proposal
            if (proposal.mission_id, proposal.uav_id, proposal.base_plan_version, proposal.new_plan_version, proposal.route_draft.route_id) != (first.mission_id, first.uav_id, first.base_plan_version, first.new_plan_version, first.route_draft.route_id):
                raise ObstacleRevisionError("repair changed immutable route routing metadata")
        critique = self._critic.evaluate(proposal.route_draft, context)
        self._attempts.append(ObstacleRevisionAttempt(proposal, critique))
        if critique.status is RouteCriticStatus.ACCEPT:
            self._state = ObstacleRevisionSessionState.ACCEPTED
        elif len(self._attempts) >= self._max_proposals:
            self._state = ObstacleRevisionSessionState.EXHAUSTED
        else:
            self._state = ObstacleRevisionSessionState.AWAITING_REPAIR
        return critique

    def history_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"state": self._state.value}
        for index, attempt in enumerate(self._attempts):
            result[f"proposal_{index}"] = attempt.proposal.to_dict()
            result[f"critique_{index}"] = attempt.critique.to_dict()
        if self._state is ObstacleRevisionSessionState.ACCEPTED:
            result["final_proposal"] = self._attempts[-1].proposal.to_dict()
        return result


def build_obstacle_route_revision_schema(request: ObstacleAwareRevisionRequest) -> dict[str, object]:
    point = {
        "type": "object", "additionalProperties": False,
        "required": ["waypoint_id", "xyz_m"],
        "properties": {
            "waypoint_id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_.-]{0,63}$"},
            "xyz_m": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
        },
    }
    follow = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "uav_id", "skill", "args"],
        "properties": {
            "id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_.-]{0,63}$"},
            "uav_id": {"type": "string", "const": request.uav_id},
            "skill": {"type": "string", "const": "FOLLOW_ROUTE"},
            "args": {"type": "object", "additionalProperties": False, "required": ["route_ref"], "properties": {"route_ref": {"type": "string", "const": request.route_id}}},
        },
    }
    step_id_schema = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_.-]{0,63}$"}
    def _step(skill: str, args: dict[str, object]) -> dict[str, object]:
        return {
            "type": "object", "additionalProperties": False,
            "required": ["id", "uav_id", "skill", "args"],
            "properties": {
                "id": dict(step_id_schema),
                "uav_id": {"type": "string", "const": request.uav_id},
                "skill": {"type": "string", "const": skill},
                "args": args,
            },
        }
    empty_args = {"type": "object", "additionalProperties": False, "properties": {}}
    current_skill = request.current_step_summary.get("skill")
    search_args: dict[str, object] = empty_args
    track_args: dict[str, object] = empty_args
    if current_skill in {"SEARCH", "TRACK"}:
        search_args = {
            "oneOf": [
                empty_args,
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["target_continuation"],
                    "properties": {
                        "target_continuation": {
                            "type": "string",
                            "const": "RESTART_SEARCH",
                        }
                    },
                },
            ]
        }
    if current_skill == "TRACK":
        track_args = {
            "oneOf": [
                empty_args,
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["target_continuation"],
                    "properties": {
                        "target_continuation": {
                            "type": "string",
                            "enum": ["CONTINUE_TRACK", "REACQUIRE"],
                        }
                    },
                },
            ]
        }
    generic = {
        "oneOf": [
            _step("GOTO", empty_args),
            _step("HOVER", {"type": "object", "additionalProperties": False, "required": ["duration_s"], "properties": {"duration_s": {"type": "number", "minimum": 0.1, "maximum": 60}}}),
            _step("SEARCH", search_args),
            _step("TRACK", track_args),
            _step("LAND", empty_args),
        ]
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "mission_id", "uav_id", "base_plan_version", "new_plan_version", "replace_from_step_id", "avoidance_strategy", "route_draft", "replacement_steps"],
        "properties": {
            "schema_version": {"type": "integer", "const": 3},
            "mission_id": {"type": "string", "const": request.mission_id},
            "uav_id": {"type": "string", "const": request.uav_id},
            "base_plan_version": {"type": "integer", "const": request.base_plan_version},
            "new_plan_version": {"type": "integer", "const": request.new_plan_version},
            "replace_from_step_id": {"type": "string", "const": request.replace_from_step_id},
            "avoidance_strategy": {
                "type": "object", "additionalProperties": False,
                "required": ["strategy", "rejoin_target", "reason_codes"],
                "properties": {
                    "strategy": {"type": "string", "enum": [item.value for item in AvoidanceStrategyType]},
                    "rejoin_target": {"type": "string", "const": "original_goto_target"},
                    "reason_codes": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "pattern": "^[A-Z][A-Z0-9_]{0,63}$"}},
                },
            },
            "route_draft": {
                "type": "object", "additionalProperties": False,
                "required": ["route_id", "frame", "waypoints"],
                "properties": {
                    "route_id": {"type": "string", "const": request.route_id},
                    "frame": {"type": "string", "const": "UAV_HOLD_FLU"},
                    "waypoints": {"type": "array", "minItems": 2, "maxItems": request.route_constraints.max_waypoints, "items": point},
                },
            },
            # The strict Python parser below additionally requires the first
            # item to be FOLLOW_ROUTE. Keeping one non-overlapping ``oneOf``
            # here is portable across deployed vLLM grammar backends.
            "replacement_steps": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"oneOf": [follow, *generic["oneOf"]]}},
        },
    }


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


__all__ = [
    "GroundedObstacleGeometry",
    "ObstacleAwareRevisionPlanner",
    "ObstacleAwareRevisionRequest",
    "ObstacleParseRepairFeedback",
    "ObstacleReplacementStep",
    "ObstacleRevisionAttempt",
    "ObstacleRevisionError",
    "ObstacleRevisionSession",
    "ObstacleRevisionSessionState",
    "ObstacleRouteRevisionDraft",
    "build_obstacle_route_revision_schema",
    "is_repairable_obstacle_parse_error",
]
