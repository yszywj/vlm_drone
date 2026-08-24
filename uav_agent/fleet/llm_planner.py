"""Strict text-only LLM Fleet Planner with request-bound structured output."""

from __future__ import annotations

import json

from models import (
    ChatMessage,
    GenerationOptions,
    JsonSchemaResponseFormat,
    ModelClient,
    ModelResponse,
)

from fleet.json_schema import build_fleet_mission_plan_json_schema
from fleet.planner_base import FleetPlanner, FleetPlannerOutputError
from fleet.schemas import parse_fleet_mission_plan
from fleet.types import FleetMissionPlan, FleetMissionRequest


class _DuplicateJSONKeyError(ValueError):
    pass


class LLMFleetPlanner(FleetPlanner):
    """Ask a model only for UAV-target-region assignment decomposition."""

    source = "fleet_llm"
    SYSTEM_PROMPT = (
        "Create exactly one FleetMissionPlan JSON object matching the supplied "
        "schema. Only decompose the natural-language fleet task, assign one "
        "trusted UAV to each target requirement, preserve every explicitly "
        "requested UAV-target-region relation, choose assignment priority, use "
        "PARALLEL for every v1 assignment (SEQUENTIAL execution is not yet "
        "supported). If a required target cannot be assigned, start one "
        "unassigned_requirements entry with its exact target_alias and a colon. "
        "Echo trusted routing and coordination "
        "limits exactly. Do not generate TAKEOFF, GOTO, SEARCH, TRACK, LAND, "
        "MissionProgram nodes, local waypoints, per-frame velocity, PID, motor "
        "commands, camera internals, Oracle coordinates, images, or hidden "
        "reasoning. Output JSON only."
    )

    def __init__(
        self,
        model_client: ModelClient,
        *,
        logger: object | None = None,
        max_tokens: int = 2048,
    ) -> None:
        if not callable(getattr(model_client, "chat", None)):
            raise TypeError("model_client must provide chat()")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 256 <= max_tokens <= 8192
        ):
            raise ValueError("max_tokens must be within [256, 8192]")
        self._client = model_client
        self._logger = logger
        self._max_tokens = max_tokens
        self._last_response_text_length: int | None = None

    @property
    def last_response_text_length(self) -> int | None:
        return self._last_response_text_length

    def plan(self, request: FleetMissionRequest) -> FleetMissionPlan:
        if not isinstance(request, FleetMissionRequest):
            raise TypeError("request must be a FleetMissionRequest")
        schema = build_fleet_mission_plan_json_schema(request)
        payload = {
            "task": "Create one FleetMissionPlan JSON object.",
            "original_instruction": request.original_instruction,
            "fleet_inventory": [
                item.to_dict() for item in request.uav_inventory
            ],
            "target_requests": [
                item.to_dict() for item in request.target_requests
            ],
            "trusted_spatial_frames": ["WORLD_ENU", "HOME_ENU"],
            "coordination_limits": {
                "max_active_assignments_per_uav": 1,
                "allowed_start_policies": ["PARALLEL"],
                **request.coordination_policy.to_dict(),
            },
            "trusted_routing": {
                "fleet_mission_id": request.fleet_mission_id,
                "fleet_plan_version": request.fleet_plan_version,
            },
        }
        messages = (
            ChatMessage("system", self.SYSTEM_PROMPT),
            ChatMessage(
                "user",
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )
        options = GenerationOptions(
            temperature=0.0,
            max_tokens=self._max_tokens,
            response_format=JsonSchemaResponseFormat(
                "fleet_mission_plan_v1", schema
            ),
        )
        self._safe_log("debug", "fleet planner model call started")
        response = self._client.chat(messages, options=options)
        if not isinstance(response, ModelResponse):
            raise FleetPlannerOutputError(
                "model client returned an invalid response object"
            )
        raw = response.content
        self._last_response_text_length = len(raw)
        if len(raw) > 262_144:
            raise FleetPlannerOutputError("FleetMissionPlan response is oversized")
        try:
            decoded = json.loads(
                raw,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)
                ),
                object_pairs_hook=_strict_json_object,
            )
            if not isinstance(decoded, dict):
                raise TypeError("top-level response must be an object")
            plan = parse_fleet_mission_plan(decoded, request=request)
        except (
            _DuplicateJSONKeyError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self._safe_log("error", "fleet planner output rejected")
            raise FleetPlannerOutputError(
                f"invalid FleetMissionPlan output: {type(exc).__name__}: {exc}"
            ) from None
        self._safe_log("debug", "fleet planner model call succeeded")
        return plan

    def _safe_log(self, level: str, message: str) -> None:
        logger = self._logger
        if logger is None:
            return
        try:
            method = getattr(logger, level, None)
            if callable(method):
                method(message)
            elif callable(logger):
                logger(message)
        except Exception:
            return


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


__all__ = ["LLMFleetPlanner"]
