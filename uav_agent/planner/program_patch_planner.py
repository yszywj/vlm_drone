"""Structured, text-only Qwen planner for bounded MissionProgram patches."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from math import isfinite
from numbers import Real

from common.ids import validate_request_id, validate_routing_id
from models import (
    AsyncModelRequest,
    AsyncModelResult,
    ChatMessage,
    GenerationOptions,
    JsonSchemaResponseFormat,
    ModelProtocolError,
)
from planner.mission_program import MissionProgram, MissionProgramError, ProgramEvent
from planner.program_patch import ProgramPatch, apply_program_patch
from planner.program_patch_schema import build_program_patch_json_schema


class ProgramPatchPlannerError(ValueError):
    """Raised when a patch request or model response violates its contract."""


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("observation_timestamp_s must be a finite number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(
            "observation_timestamp_s must be finite and non-negative"
        )
    return result


@dataclass(frozen=True, slots=True)
class ProgramPatchRequest:
    """Trusted graph and runtime cursor supplied to the model planner."""

    program: MissionProgram
    current_node_id: str
    completed_node_ids: tuple[str, ...]
    trigger_event: ProgramEvent
    observation_timestamp_s: float
    frame_id: str
    max_replacement_nodes: int = 16

    def __post_init__(self) -> None:
        if not isinstance(self.program, MissionProgram):
            raise TypeError("program must be a MissionProgram")
        owned = deepcopy(self.program)
        current = validate_routing_id(self.current_node_id, "current_node_id")
        known = {node.node_id for node in owned.nodes}
        if current not in known:
            raise ProgramPatchPlannerError("current_node_id is unknown")
        completed = tuple(
            validate_routing_id(item, "completed_node_id")
            for item in self.completed_node_ids
        )
        if len(completed) != len(set(completed)) or not set(completed) <= known:
            raise ProgramPatchPlannerError(
                "completed_node_ids must be unique known nodes"
            )
        if current in completed:
            raise ProgramPatchPlannerError(
                "current_node_id cannot already be completed"
            )
        try:
            event = (
                self.trigger_event
                if isinstance(self.trigger_event, ProgramEvent)
                else ProgramEvent(self.trigger_event)
            )
        except (TypeError, ValueError):
            raise ProgramPatchPlannerError("trigger_event is unsupported") from None
        if event is not ProgramEvent.PATH_BLOCKED:
            raise ProgramPatchPlannerError(
                "ProgramPatch planner currently accepts only PATH_BLOCKED"
            )
        if (
            isinstance(self.max_replacement_nodes, bool)
            or not isinstance(self.max_replacement_nodes, int)
            or not 1 <= self.max_replacement_nodes <= 32
        ):
            raise ProgramPatchPlannerError(
                "max_replacement_nodes must be within [1, 32]"
            )
        object.__setattr__(self, "program", owned)
        object.__setattr__(self, "current_node_id", current)
        object.__setattr__(self, "completed_node_ids", completed)
        object.__setattr__(self, "trigger_event", event)
        object.__setattr__(
            self, "observation_timestamp_s", _timestamp(self.observation_timestamp_s)
        )
        object.__setattr__(
            self, "frame_id", validate_routing_id(self.frame_id, "frame_id")
        )

    def prompt_payload(self) -> dict[str, object]:
        return {
            "trigger_event": self.trigger_event.value,
            "current_node_id": self.current_node_id,
            "completed_node_ids": list(self.completed_node_ids),
            "base_plan_version": self.program.plan_version,
            "required_new_plan_version": self.program.plan_version + 1,
            "max_replacement_nodes": self.max_replacement_nodes,
            "authoritative_program": self.program.to_dict(),
        }


class QwenProgramPatchPlanner:
    """Build and parse one schema-constrained ProgramPatch request."""

    SYSTEM_PROMPT = (
        "Return exactly one JSON ProgramPatch matching the supplied schema. "
        "Replace only the current node and its future suffix. Never include, "
        "edit, or restate a completed node or an edge originating in the "
        "completed prefix. The first replacement node id must equal "
        "current_node_id. Echo mission_id, uav_id, and both plan versions "
        "exactly. Include PATH_BLOCKED in reason_codes. Preserve a reachable, "
        "acyclic graph ending in a terminal LAND node. Use only Skill calls and "
        "arguments in the authoritative program contract; never output "
        "velocities, controller gains, or hidden reasoning. The runtime will "
        "validate the proposal exactly as returned and will not add, remove, "
        "clamp, reorder, or otherwise repair any node, edge, or argument."
    )

    def build_async_request(
        self,
        request: ProgramPatchRequest,
        *,
        request_id: str,
    ) -> AsyncModelRequest:
        if not isinstance(request, ProgramPatchRequest):
            raise TypeError("request must be a ProgramPatchRequest")
        routed_request_id = validate_request_id(request_id)
        review_id = f"review_patch_{request.current_node_id}"
        payload = json.dumps(
            request.prompt_payload(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return AsyncModelRequest(
            request_id=routed_request_id,
            review_id=review_id,
            mission_id=request.program.mission_id,
            uav_id=request.program.uav_id,
            plan_version=request.program.plan_version,
            observation_timestamp_s=request.observation_timestamp_s,
            frame_id=request.frame_id,
            messages=(
                ChatMessage("system", self.SYSTEM_PROMPT),
                ChatMessage("user", payload),
            ),
            options=GenerationOptions(
                temperature=0.0,
                max_tokens=1536,
                response_format=JsonSchemaResponseFormat(
                    "mission_program_patch_v1",
                    build_program_patch_json_schema(
                        mission_id=request.program.mission_id,
                        uav_id=request.program.uav_id,
                        base_plan_version=request.program.plan_version,
                        replace_from_node_id=request.current_node_id,
                        trigger_event=request.trigger_event,
                        max_replacement_nodes=request.max_replacement_nodes,
                    ),
                ),
            ),
        )

    def parse_async_result(
        self,
        result: AsyncModelResult,
        *,
        request: ProgramPatchRequest,
        expected_request_id: str,
    ) -> ProgramPatch:
        if not isinstance(result, AsyncModelResult):
            raise TypeError("result must be an AsyncModelResult")
        if not isinstance(request, ProgramPatchRequest):
            raise TypeError("request must be a ProgramPatchRequest")
        expected_id = validate_request_id(expected_request_id)
        expected_review_id = f"review_patch_{request.current_node_id}"
        expected_route = (
            expected_id,
            expected_review_id,
            request.program.mission_id,
            request.program.uav_id,
            request.program.plan_version,
            request.observation_timestamp_s,
            request.frame_id,
        )
        actual_route = (
            result.request_id,
            result.review_id,
            result.mission_id,
            result.uav_id,
            result.plan_version,
            result.observation_timestamp_s,
            result.frame_id,
        )
        if actual_route != expected_route:
            raise ProgramPatchPlannerError(
                "async ProgramPatch result routing/version does not match request"
            )
        if result.stale:
            raise ProgramPatchPlannerError("async ProgramPatch result is stale")
        if result.response is None:
            raise ModelProtocolError(
                result.error_code or "ProgramPatch model request failed"
            )
        try:
            decoded = json.loads(
                result.response.content,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)
                ),
                object_pairs_hook=_strict_json_object,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelProtocolError(f"invalid ProgramPatch JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ModelProtocolError("ProgramPatch response must be a JSON object")
        try:
            patch = ProgramPatch.from_dict(decoded)
        except (MissionProgramError, TypeError, ValueError) as exc:
            raise ModelProtocolError(f"invalid ProgramPatch contract: {exc}") from exc
        if (
            patch.mission_id != request.program.mission_id
            or patch.uav_id != request.program.uav_id
            or patch.base_plan_version != request.program.plan_version
            or patch.new_plan_version != request.program.plan_version + 1
            or patch.replace_from_node_id != request.current_node_id
            or request.trigger_event.value not in patch.reason_codes
            or len(patch.replacement_nodes) > request.max_replacement_nodes
        ):
            raise ProgramPatchPlannerError(
                "ProgramPatch does not echo trusted routing/version/event fields"
            )
        # Full graph construction proves completed-prefix protection, edge
        # references, and version semantics without mutating request.program.
        apply_program_patch(
            request.program,
            patch,
            completed_node_ids=frozenset(request.completed_node_ids),
        )
        return patch


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


__all__ = [
    "ProgramPatchPlannerError",
    "ProgramPatchRequest",
    "QwenProgramPatchPlanner",
]
