"""Goal-referenced, request-bound Qwen Fleet Planner V2."""

from __future__ import annotations

from copy import deepcopy
import json

from models import (
    ChatMessage,
    GenerationOptions,
    JsonSchemaResponseFormat,
    ModelClient,
    ModelResponse,
)
from planner.diagnostics import PlannerDiagnostics

from fleet.json_schema_v2 import build_fleet_mission_plan_v2_json_schema
from fleet.planner_base import FleetPlannerOutputError
from fleet.schemas_v2 import (
    FleetPlanSemanticIssue,
    fleet_plan_v2_semantic_findings,
    parse_fleet_mission_plan_v2,
)
from fleet.strict_json import DuplicateJSONKeyError, strict_json_object_loads
from fleet.task_spec import reject_forbidden_task_fields
from fleet.types import FleetMissionError
from fleet.types_v2 import FleetMissionPlanV2, FleetMissionRequestV2


MAX_PROPOSAL_BYTES = 32_768
MAX_RESPONSE_BYTES = 32_768


def _error_code(exc: Exception) -> str:
    if isinstance(exc, DuplicateJSONKeyError):
        return "DUPLICATE_JSON_KEY"
    if isinstance(exc, json.JSONDecodeError):
        return "INVALID_JSON"
    if isinstance(exc, FleetMissionError):
        return "FLEET_PLAN_V2_VALIDATION_ERROR"
    if isinstance(exc, (TypeError, ValueError)):
        return "STRUCTURE_VALIDATION_ERROR"
    return type(exc).__name__.upper()


class LLMFleetPlannerV2:
    """Let Qwen choose UAV assignments while code guards trusted entities."""

    source = "fleet_llm_v2"
    SYSTEM_PROMPT = (
        "Return exactly one FleetMissionPlanV2 JSON object matching the supplied "
        "schema. Assign semantic goal_ids to available UAVs using capability and "
        "trusted Fleet-state evidence. You own the final uav_id choice: user UAV "
        "preferences are AssignmentConstraints, not fixed assignments. If you "
        "deviate from a non-OPEN AssignmentConstraint, add its constraint_id, a "
        "bounded reason_code, and evidence_refs that cite only trusted Fleet-state "
        "evidence IDs. Account for goals that cannot be assigned in "
        "unassigned_goal_ids. Echo routing and coordination values exactly. Do not "
        "generate Skills, local routes, controller commands, images, Oracle fields, "
        "or hidden reasoning. Output JSON only."
    )

    def __init__(
        self,
        model_client: ModelClient,
        *,
        logger: object | None = None,
        max_tokens: int = 3072,
        repair_budget: int = 2,
    ) -> None:
        if not callable(getattr(model_client, "chat", None)):
            raise TypeError("model_client must provide chat()")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 256 <= max_tokens <= 8192
        ):
            raise ValueError("max_tokens must be within [256, 8192]")
        if (
            isinstance(repair_budget, bool)
            or not isinstance(repair_budget, int)
            or not 0 <= repair_budget <= 2
        ):
            raise ValueError("repair_budget must be within [0, 2]")
        self._client = model_client
        self._logger = logger
        self._max_tokens = max_tokens
        self._repair_budget = repair_budget
        self._last_diagnostics: PlannerDiagnostics | None = None
        self._last_semantic_findings: tuple[FleetPlanSemanticIssue, ...] = ()
        self._model_proposals: list[dict[str, object]] = []

    @property
    def last_diagnostics(self) -> PlannerDiagnostics | None:
        return self._last_diagnostics

    @property
    def last_semantic_findings(self) -> tuple[FleetPlanSemanticIssue, ...]:
        return self._last_semantic_findings

    @property
    def model_proposals(self) -> tuple[dict[str, object], ...]:
        return tuple(deepcopy(item) for item in self._model_proposals)

    def plan(self, request: FleetMissionRequestV2) -> FleetMissionPlanV2:
        if not isinstance(request, FleetMissionRequestV2):
            raise TypeError("request must be a FleetMissionRequestV2")
        schema = build_fleet_mission_plan_v2_json_schema(request)
        payload = {
            "task": "Create one FleetMissionPlanV2 goal assignment.",
            "trusted_request": request.to_dict(),
            "planner_limits": {
                "max_active_assignments_per_uav": 1,
                "maximum_assignments": min(
                    len(request.available_uav_ids), len(request.task_spec.all_goal_ids)
                ),
                "semantic_deviations_are_recoverable": True,
            },
        }
        initial_messages = (
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
            top_p=1.0,
            response_format=JsonSchemaResponseFormat(
                "fleet_mission_plan_v2", schema
            ),
        )
        self._model_proposals = []
        self._last_diagnostics = None
        self._last_semantic_findings = ()
        messages = initial_messages
        initial_error_code: str | None = None
        initial_error_message: str | None = None
        attempts = 1 + self._repair_budget
        for attempt_index in range(attempts):
            repair = attempt_index > 0
            raw = ""
            decoded: dict[str, object] | None = None
            try:
                self._safe_log(
                    "debug",
                    "fleet V2 repair call started"
                    if repair
                    else "fleet V2 planner call started",
                )
                response = self._client.chat(messages, options=options)
                if not isinstance(response, ModelResponse):
                    raise TypeError("model client returned an invalid response object")
                raw = response.content
                if len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
                    raise ValueError(
                        f"FleetMissionPlanV2 response exceeds {MAX_RESPONSE_BYTES} bytes"
                    )
                decoded = strict_json_object_loads(raw)
                reject_forbidden_task_fields(decoded, "FleetMissionPlanV2")
                plan = parse_fleet_mission_plan_v2(decoded, request=request)
            except (
                DuplicateJSONKeyError,
                FleetMissionError,
                OverflowError,
                RecursionError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                code = _error_code(exc)
                message = f"{type(exc).__name__}: {exc}"[:512]
                if attempt_index == 0:
                    initial_error_code = code
                    initial_error_message = message
                self._record_proposal(
                    attempt_index=attempt_index,
                    repair=repair,
                    accepted=False,
                    error_code=code,
                    decoded=decoded,
                    response_length=len(raw),
                )
                if attempt_index + 1 >= attempts:
                    self._last_diagnostics = PlannerDiagnostics(
                        model_calls=attempt_index + 1,
                        repair_used=repair,
                        repair_succeeded=False,
                        initial_output_valid=False,
                        final_output_valid=False,
                        initial_error_code=initial_error_code,
                        initial_error_message=initial_error_message,
                        structured_output_enabled=True,
                    )
                    self._safe_log("error", "fleet V2 planner output rejected")
                    raise FleetPlannerOutputError(
                        f"invalid FleetMissionPlanV2 output after "
                        f"{attempt_index + 1} attempt(s): {code}: {message}"
                    ) from None
                messages = (
                    *initial_messages,
                    ChatMessage(
                        "user",
                        "The prior proposal was rejected with "
                        f"{code}: {message}. Generate a fresh complete JSON object "
                        "matching the same schema. Do not add prose or reasoning.",
                    ),
                )
                continue
            self._record_proposal(
                attempt_index=attempt_index,
                repair=repair,
                accepted=True,
                error_code=None,
                decoded=decoded,
                response_length=len(raw),
            )
            self._last_semantic_findings = fleet_plan_v2_semantic_findings(
                plan, request
            )
            self._last_diagnostics = PlannerDiagnostics(
                model_calls=attempt_index + 1,
                repair_used=repair,
                repair_succeeded=repair,
                initial_output_valid=not repair,
                final_output_valid=True,
                initial_error_code=initial_error_code,
                initial_error_message=initial_error_message,
                structured_output_enabled=True,
            )
            self._safe_log("debug", "fleet V2 planner call succeeded")
            return plan
        raise AssertionError("unreachable Fleet V2 planner attempt loop")

    def _record_proposal(
        self,
        *,
        attempt_index: int,
        repair: bool,
        accepted: bool,
        error_code: str | None,
        decoded: dict[str, object] | None,
        response_length: int,
    ) -> None:
        proposal: dict[str, object] | None = None
        if decoded is not None:
            try:
                reject_forbidden_task_fields(decoded, "model_proposal")
                rendered = json.dumps(
                    decoded,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                if len(rendered) <= MAX_PROPOSAL_BYTES:
                    proposal = deepcopy(decoded)
            except (TypeError, ValueError, OverflowError):
                proposal = None
        self._model_proposals.append(
            {
                "attempt_index": attempt_index,
                "repair": repair,
                "accepted": accepted,
                "error_code": error_code,
                "response_length": max(0, response_length),
                "proposal": proposal,
            }
        )

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


__all__ = ["LLMFleetPlannerV2", "MAX_PROPOSAL_BYTES"]
