"""Text-only LLM planner for the constrained dynamic SkillPlanDraft protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import re

from models.base import (
    ChatMessage,
    GenerationOptions,
    JsonSchemaResponseFormat,
    ModelClient,
    ModelResponse,
)
from planner.base import MissionPlanner, PlannerError, PlannerOutputError
from planner.diagnostics import PlannerDiagnostics, PlannerExecution
from planner.json_schema import build_skill_plan_v2_json_schema
from planner.json_schema_v3 import build_skill_plan_v3_json_schema
from planner.policy import PlannerLimits, PlannerPolicy
from planner.prompt_builder import (
    build_dynamic_skill_planner_messages,
    build_spatial_v3_skill_planner_messages,
)
from planner.schemas import (
    PlannerRequest,
    PlannerWorldContext,
    SkillPlanDraft,
    SkillPlanDraftV2,
)
from planner.schemas_v3 import SkillPlanDraftV3
from planner.skill_catalog import (
    SkillArgumentSpec,
    SkillCatalog,
    build_default_skill_catalog,
    build_spatial_v3_skill_catalog,
    initial_planner_catalog,
)
from planner.spatial import (
    CircleRegion,
    CoordinateFrame,
    CorridorRegion,
    NamedLocationTarget,
    PointTarget,
    RouteTarget,
    SectorRegion,
)
from planner.symbolic_checker import PlanIssue, SymbolicPlanChecker
from planner.trusted_safety_completion import (
    analyze_trusted_safety_completion,
)
from skills.search_strategy import (
    SearchRuntimeCapabilities,
    SearchStrategySpec,
    SearchStrategyType,
)


_JSON_FENCE = re.compile(
    r"\A```json[ \t]*(?:\r?\n)?(?P<body>.*?)\r?\n?```\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_FIXED_YAW_CONDITION = "only allowed and required when yaw_mode is FIXED"
# The documented local Qwen service uses a 4096-token context.  Initial output
# has enough room for a typical draft.  Repair uses a compact context (without
# the redundant Skill Catalog and full initial system prompt) and has a larger
# bounded output budget so a complete corrected multi-step object is not cut
# off mid-JSON on the documented 4096-token local deployment.
_DYNAMIC_PLAN_MAX_TOKENS = 1024
_DYNAMIC_REPAIR_MAX_TOKENS = 1536
_COMPACT_REPAIR_SYSTEM_PROMPT = (
    "Repair one UAV Skill plan as strict compact JSON matching the bound "
    "response schema. Preserve the user instruction and every trusted "
    "routing field. The previous object was rejected: do not repeat it "
    "unchanged. Emit the complete mission, not a valid-looking prefix, and "
    "add every requested action that the rejected object omitted. Obey the "
    "supplied validation issues and requirements. "
    "Use explicit Spatial V3 coordinate frames when applicable. Never emit "
    "Markdown, hidden reasoning, Oracle facts, low-level controls, or extra "
    "text. Generate the complete corrected object yourself."
)
_REPAIR_CONTEXT_KEYS = (
    "trusted_routing",
    "trusted_world_context",
    "planner_limits",
    "trusted_planner_policy",
    "runtime_search_capabilities",
    "coordinate_frames",
    "spatial_output_policy",
    "mission_completeness_contract",
    "user_instruction",
)

_SPATIAL_FRAME_NAMES = (
    "WORLD_ENU",
    "HOME_ENU",
    "UAV_START_FLU",
    "UAV_HOLD_FLU",
    "CAMERA_FLU",
)
_SPATIAL_FRAME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(name) for name in _SPATIAL_FRAME_NAMES)
    + r")(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
# Deliberately conservative: cardinal ENU words are not relative-frame
# ambiguities, and bare Chinese ``左右`` is excluded so a bounded expression
# such as ``左右各三十度`` is not mistaken for two unresolved directions.
_RELATIVE_DIRECTION_PATTERN = re.compile(
    r"(?:"
    r"左(?:边|侧|方)|右(?:边|侧|方)|"
    r"前(?:边|面|侧|方)|后(?:边|面|侧)|后方(?!可|能|才|会)|"
    r"(?<![A-Za-z])(?:(?:(?:to|on)\s+the\s+|turn\s+)left|"
    r"left\s+(?:side|of)|(?:(?:to|on)\s+the\s+|turn\s+)right|"
    r"right\s+(?:side|of)|"
    r"in\s+front(?:\s+of)?|front\s+side|behind|back\s+side)(?![A-Za-z])"
    r")",
    flags=re.IGNORECASE,
)
_SYMMETRIC_ANGLE_PATTERN = re.compile(
    r"左(?:右|侧\s*(?:和|与|及|、)\s*右侧?)各"
    r"[^，。；;,.!?！？\n]{0,24}(?:度|°)"
)
_SPATIAL_CLAUSE_BOUNDARIES = frozenset("，,。；;！？!?\n\r")


class _DuplicateJSONKeyError(ValueError):
    """Internal marker for ambiguous JSON objects."""


class _DraftValidationFailure(ValueError):
    """Sanitized validation failure used to drive the one repair call."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        issues: tuple[PlanIssue, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.issues = issues


class PlanningContract(str, Enum):
    V2 = "v2"
    V3 = "v3"


def _clause_contains_explicit_spatial_frame(
    instruction: str,
    start: int,
    end: int,
) -> bool:
    clause_start = start
    while (
        clause_start > 0
        and instruction[clause_start - 1] not in _SPATIAL_CLAUSE_BOUNDARIES
    ):
        clause_start -= 1
    clause_end = end
    while (
        clause_end < len(instruction)
        and instruction[clause_end] not in _SPATIAL_CLAUSE_BOUNDARIES
    ):
        clause_end += 1
    return _SPATIAL_FRAME_PATTERN.search(
        instruction[clause_start:clause_end]
    ) is not None


def _ambiguous_relative_direction_spans(
    instruction: str,
) -> tuple[tuple[int, int, str], ...]:
    symmetric_angles = tuple(
        match.span() for match in _SYMMETRIC_ANGLE_PATTERN.finditer(instruction)
    )
    result: list[tuple[int, int, str]] = []
    for match in _RELATIVE_DIRECTION_PATTERN.finditer(instruction):
        start, end = match.span()
        if any(left <= start and end <= right for left, right in symmetric_angles):
            continue
        if _clause_contains_explicit_spatial_frame(instruction, start, end):
            continue
        result.append((start, end, match.group(0)))
    return tuple(result)


class DynamicLLMPlanner(MissionPlanner):
    """Use a text model to select a finite high-level linear Skill plan.

    The result is intentionally unresolved: named places remain names and a
    TRACK target references either a prior SEARCH output or an explicitly
    advertised trusted runtime lock.  Trusted Python code performs all
    geometry resolution and safety completion later.
    """

    source = "dynamic_llm"

    def __init__(
        self,
        model_client: ModelClient,
        system_prompt_path: str | os.PathLike[str],
        skill_catalog: SkillCatalog | None = None,
        planner_limits: object = None,
        logger: object | None = None,
        planner_policy: PlannerPolicy | Mapping[str, object] | None = None,
        *,
        planning_contract: PlanningContract | str = PlanningContract.V2,
        search_runtime_capabilities: SearchRuntimeCapabilities | None = None,
        repair_budget: int = 1,
    ) -> None:
        if not callable(getattr(model_client, "chat", None)):
            raise TypeError("model_client must provide a callable chat() method")
        try:
            contract = (
                planning_contract
                if isinstance(planning_contract, PlanningContract)
                else PlanningContract(planning_contract)
            )
        except (TypeError, ValueError):
            raise ValueError("planning_contract must be v2 or v3") from None
        if not isinstance(system_prompt_path, (str, os.PathLike)):
            raise TypeError("system_prompt_path must be a path-like value")
        try:
            prompt = Path(system_prompt_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise PlannerError(
                "could not read the dynamic Skill planner system prompt"
            ) from None
        if not prompt.strip():
            raise PlannerError(
                "dynamic Skill planner system prompt must be non-empty"
            )

        capabilities = (
            SearchRuntimeCapabilities()
            if search_runtime_capabilities is None
            else search_runtime_capabilities
        )
        if not isinstance(capabilities, SearchRuntimeCapabilities):
            raise TypeError(
                "search_runtime_capabilities must be a "
                "SearchRuntimeCapabilities or None"
            )
        if (
            isinstance(repair_budget, bool)
            or not isinstance(repair_budget, int)
            or repair_budget not in {0, 1}
        ):
            raise ValueError("repair_budget must be integer 0 or 1")
        if skill_catalog is None:
            skill_catalog = (
                build_default_skill_catalog()
                if contract is PlanningContract.V2
                else build_spatial_v3_skill_catalog(capabilities)
            )
        if not isinstance(skill_catalog, SkillCatalog):
            raise TypeError("skill_catalog must be a SkillCatalog")
        self._validate_catalog_conditions(skill_catalog)
        if contract is PlanningContract.V3:
            self._validate_v3_catalog_shape(skill_catalog)

        if planner_limits is None:
            limits = PlannerLimits()
        elif isinstance(planner_limits, PlannerLimits):
            limits = planner_limits
        elif isinstance(planner_limits, Mapping):
            limits = PlannerLimits(**dict(planner_limits))
        else:
            try:
                limits = PlannerLimits.from_config(planner_limits)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TypeError(
                    "planner_limits must be PlannerLimits, a mapping, or a "
                    "compatible config object"
                ) from exc

        if planner_policy is None:
            policy = PlannerPolicy()
        elif isinstance(planner_policy, PlannerPolicy):
            policy = planner_policy
        elif isinstance(planner_policy, Mapping):
            policy = PlannerPolicy(**dict(planner_policy))
        else:
            try:
                policy = PlannerPolicy.from_config(planner_policy)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TypeError(
                    "planner_policy must be PlannerPolicy, a mapping, or a "
                    "compatible config object"
                ) from exc
        policy.validate_against(limits)

        self._model_client = model_client
        self._system_prompt = prompt.strip()
        self._planning_contract = contract
        self._search_runtime_capabilities = capabilities
        # The complete catalog includes runtime-revision-only INSPECT.  An
        # initial model request has no trusted CandidateBank ID, so expose and
        # enforce only the initial-planning projection.
        self._skill_catalog = initial_planner_catalog(skill_catalog)
        self._planner_limits = limits
        self._planner_policy = policy
        self._symbolic_checker = SymbolicPlanChecker()
        self._logger = logger
        self._repair_budget = repair_budget
        self._last_diagnostics: PlannerDiagnostics | None = None
        self._model_proposals: list[dict[str, object]] = []

    @property
    def skill_catalog(self) -> SkillCatalog:
        return self._skill_catalog

    @property
    def last_diagnostics(self) -> PlannerDiagnostics | None:
        return self._last_diagnostics

    @property
    def planning_contract(self) -> str:
        return self._planning_contract.value

    @property
    def search_runtime_capabilities(self) -> SearchRuntimeCapabilities:
        return self._search_runtime_capabilities

    @property
    def model_proposals(self) -> tuple[dict[str, object], ...]:
        """Return bounded image-free raw structured proposal records."""

        return tuple(deepcopy(item) for item in self._model_proposals)

    @property
    def repair_budget(self) -> int:
        """Return the bounded number of planner-internal repair calls."""

        return self._repair_budget

    def plan(
        self,
        request: PlannerRequest,
    ) -> SkillPlanDraftV2 | SkillPlanDraftV3:
        return self.plan_with_diagnostics(request).output

    def plan_with_diagnostics(self, request: PlannerRequest) -> PlannerExecution:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")

        self._last_diagnostics = None
        self._model_proposals.clear()
        if not request.has_routing_ids:
            self._last_diagnostics = PlannerDiagnostics(
                model_calls=0,
                repair_used=False,
                repair_succeeded=False,
                initial_output_valid=False,
                final_output_valid=False,
                initial_error_code="ROUTING_IDS_REQUIRED",
                initial_error_message=(
                    "dynamic Qwen planning requires trusted routing IDs"
                ),
                structured_output_enabled=True,
            )
            raise PlannerError(
                "DynamicLLMPlanner requires trusted mission_id, uav_id, and "
                "plan_version before any model call"
            )

        assert request.mission_id is not None
        assert request.uav_id is not None
        assert request.plan_version is not None
        if self._planning_contract is PlanningContract.V2:
            initial_messages = build_dynamic_skill_planner_messages(
                request.instruction,
                request.world_context,
                self._skill_catalog,
                self._planner_limits,
                self._system_prompt,
                self._planner_policy,
                mission_id=request.mission_id,
                uav_id=request.uav_id,
                plan_version=request.plan_version,
            )
            schema = build_skill_plan_v2_json_schema(
                world_context=request.world_context,
                skill_catalog=self._skill_catalog,
                limits=self._planner_limits,
                mission_id=request.mission_id,
                uav_id=request.uav_id,
                plan_version=request.plan_version,
            )
            response_format_name = "skill_plan_draft_v2"
        else:
            initial_messages = build_spatial_v3_skill_planner_messages(
                request.instruction,
                request.world_context,
                self._skill_catalog,
                self._planner_limits,
                self._system_prompt,
                self._planner_policy,
                mission_id=request.mission_id,
                uav_id=request.uav_id,
                plan_version=request.plan_version,
                search_runtime_capabilities=self._search_runtime_capabilities,
                trusted_target_locked=request.trusted_target_id is not None,
                allow_trusted_safety_completion=(
                    request.allow_trusted_safety_completion
                ),
            )
            schema = build_skill_plan_v3_json_schema(
                mission_id=request.mission_id,
                uav_id=request.uav_id,
                plan_version=request.plan_version,
                search_runtime_capabilities=self._search_runtime_capabilities,
                trusted_target_spec=request.trusted_target_spec,
                require_empty_assumptions=(
                    request.require_empty_spatial_assumptions
                ),
                trusted_target_locked=request.trusted_target_id is not None,
            )
            response_format_name = "skill_plan_draft_v3"
        response_format = JsonSchemaResponseFormat(
            name=response_format_name,
            schema=schema,
        )
        generation_options = GenerationOptions(
            temperature=0.0,
            max_tokens=_DYNAMIC_PLAN_MAX_TOKENS,
            response_format=response_format,
        )
        repair_generation_options = GenerationOptions(
            temperature=0.0,
            max_tokens=_DYNAMIC_REPAIR_MAX_TOKENS,
            response_format=response_format,
        )
        self._safe_log("debug", "dynamic Skill plan model call started")
        try:
            first_output = self._chat_content(initial_messages, generation_options)
        except Exception:
            self._record_model_proposal(
                None,
                attempt_index=0,
                repair=False,
                accepted=False,
                error_code="MODEL_CLIENT_ERROR",
            )
            self._last_diagnostics = self._failure_diagnostics(
                model_calls=1,
                repair_used=False,
                initial_error_code="MODEL_CLIENT_ERROR",
                initial_error_message="model client call failed",
            )
            raise
        first_failure: _DraftValidationFailure | None = None
        try:
            draft = self._validate_model_output(first_output, request)
        except _DraftValidationFailure as exc:
            first_failure = exc
            self._record_model_proposal(
                first_output,
                attempt_index=0,
                repair=False,
                accepted=False,
                error_code=exc.code,
            )
            self._safe_log(
                "warning",
                (
                    "dynamic Skill plan output was invalid; requesting one repair"
                    if self._repair_budget
                    else "dynamic Skill plan output was invalid; internal repair "
                    "is disabled"
                ),
            )
        else:
            self._record_model_proposal(
                first_output,
                attempt_index=0,
                repair=False,
                accepted=True,
                error_code=None,
            )
            self._safe_log("debug", "dynamic Skill plan model call succeeded")
            diagnostics = PlannerDiagnostics(
                model_calls=1,
                repair_used=False,
                repair_succeeded=False,
                initial_output_valid=True,
                final_output_valid=True,
                initial_error_code=None,
                initial_error_message=None,
                structured_output_enabled=True,
            )
            self._last_diagnostics = diagnostics
            return PlannerExecution(output=draft, diagnostics=diagnostics)

        assert first_failure is not None

        if self._repair_budget == 0:
            self._safe_log(
                "error",
                "dynamic Skill plan output was invalid; internal repair is disabled",
            )
            self._last_diagnostics = self._failure_diagnostics(
                model_calls=1,
                repair_used=False,
                initial_error_code=first_failure.code,
                initial_error_message=first_failure.message,
            )
            raise PlannerOutputError(
                "model failed to produce a valid SkillPlanDraft and internal "
                f"repair is disabled: {first_failure.code}: "
                f"{first_failure.message}"
            ) from None

        # The response schema already carries the authoritative Skill shape.
        # Re-sending the full catalog alongside the prior JSON can overflow
        # the documented local Qwen 4096-token context and turn a recoverable
        # validation error into HTTP 400.  Retain only trusted/user semantics.
        repair_messages = (
            ChatMessage(role="system", content=_COMPACT_REPAIR_SYSTEM_PROMPT),
            self._compact_repair_context(initial_messages[1]),
            ChatMessage(
                role="user",
                content=self._build_repair_prompt(
                    first_output,
                    first_failure,
                ),
            ),
        )
        try:
            repaired_output = self._chat_content(
                repair_messages,
                repair_generation_options,
            )
        except Exception:
            self._record_model_proposal(
                None,
                attempt_index=1,
                repair=True,
                accepted=False,
                error_code="MODEL_CLIENT_ERROR",
            )
            self._last_diagnostics = self._failure_diagnostics(
                model_calls=2,
                repair_used=True,
                initial_error_code=first_failure.code,
                initial_error_message=first_failure.message,
            )
            raise
        try:
            draft = self._validate_model_output(repaired_output, request)
        except _DraftValidationFailure as second_error:
            self._record_model_proposal(
                repaired_output,
                attempt_index=1,
                repair=True,
                accepted=False,
                error_code=second_error.code,
            )
            self._safe_log("error", "dynamic Skill plan repair output was invalid")
            self._last_diagnostics = self._failure_diagnostics(
                model_calls=2,
                repair_used=True,
                initial_error_code=first_failure.code,
                initial_error_message=first_failure.message,
            )
            raise PlannerOutputError(
                "model failed to produce a valid SkillPlanDraft after one repair: "
                f"{second_error.code}: {second_error.message}"
            ) from None

        self._record_model_proposal(
            repaired_output,
            attempt_index=1,
            repair=True,
            accepted=True,
            error_code=None,
        )
        self._safe_log("debug", "dynamic Skill plan repair succeeded")
        diagnostics = PlannerDiagnostics(
            model_calls=2,
            repair_used=True,
            repair_succeeded=True,
            initial_output_valid=False,
            final_output_valid=True,
            initial_error_code=first_failure.code,
            initial_error_message=first_failure.message,
            structured_output_enabled=True,
        )
        self._last_diagnostics = diagnostics
        return PlannerExecution(output=draft, diagnostics=diagnostics)

    def _record_model_proposal(
        self,
        raw_output: str | None,
        *,
        attempt_index: int,
        repair: bool,
        accepted: bool,
        error_code: str | None,
    ) -> None:
        record: dict[str, object] = {
            "attempt_index": attempt_index,
            "repair": repair,
            "accepted": accepted,
            "error_code": error_code,
        }
        if raw_output is None:
            record["raw_proposal"] = None
            record["response_text_length"] = 0
        else:
            record["response_text_length"] = len(raw_output)
            try:
                parsed = self._parse_json_object(raw_output)
                # Round-trip removes custom Mapping implementations and keeps
                # only JSON values.  The model response is text-only here.
                record["raw_proposal"] = json.loads(
                    json.dumps(
                        parsed,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
            except Exception:
                record["raw_proposal"] = None
                record["response_text_tail"] = raw_output[-500:]
        self._model_proposals.append(record)

    def _chat_content(
        self,
        messages: tuple[ChatMessage, ...],
        options: GenerationOptions,
    ) -> str:
        response = self._model_client.chat(
            messages,
            options=options,
        )
        if not isinstance(response, ModelResponse):
            raise PlannerOutputError(
                "model client returned an invalid response object"
            )
        return response.content

    def _validate_model_output(
        self,
        raw_output: str,
        request: PlannerRequest,
    ) -> SkillPlanDraftV2 | SkillPlanDraftV3:
        try:
            parsed = self._parse_json_object(raw_output)
        except (
            json.JSONDecodeError,
            _DuplicateJSONKeyError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            raise _DraftValidationFailure(
                "INVALID_JSON",
                self._describe_validation_error(exc),
            ) from None
        try:
            if self._planning_contract is PlanningContract.V2:
                if "target_spec" not in parsed:
                    raise ValueError(
                        "schema-v2 initial plan must include target_spec"
                    )
                draft: SkillPlanDraftV2 | SkillPlanDraftV3 = (
                    SkillPlanDraftV2.from_dict(parsed)
                )
            else:
                self._reject_v3_goto_route_targets(parsed)
                draft = SkillPlanDraftV3.from_dict(parsed)
                if (
                    request.trusted_target_spec is not None
                    and draft.target_spec != request.trusted_target_spec
                ):
                    raise ValueError(
                        "target_spec must exactly echo the trusted PlannerRequest"
                    )
                if (
                    request.require_empty_spatial_assumptions
                    and draft.assumptions
                ):
                    raise ValueError(
                        "assumptions must be empty for this fully structured request"
                    )
            self._require_initial_plan_skills(draft)
            if draft.mission_id != request.mission_id:
                raise ValueError(
                    "mission_id must exactly echo the trusted PlannerRequest"
                )
            if draft.uav_id != request.uav_id:
                raise ValueError(
                    "uav_id must exactly echo the trusted PlannerRequest"
                )
            if draft.plan_version != request.plan_version:
                raise ValueError(
                    "plan_version must exactly echo the trusted PlannerRequest"
                )
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise _DraftValidationFailure(
                "SCHEMA_INVALID",
                self._describe_validation_error(exc),
            ) from None
        if isinstance(draft, SkillPlanDraftV2):
            try:
                self._validate_against_catalog(draft.to_v1())
            except (OverflowError, RecursionError, TypeError, ValueError) as exc:
                raise _DraftValidationFailure(
                    "CATALOG_CONTRACT_VIOLATION",
                    self._describe_validation_error(exc),
                ) from None
            result = self._symbolic_checker.check(
                draft.to_v1(),
                world_context=request.world_context,
                limits=self._planner_limits,
                policy=self._planner_policy,
            )
            if not result.valid:
                first_issue = result.issues[0]
                raise _DraftValidationFailure(
                    first_issue.code.value,
                    first_issue.message,
                    issues=result.issues,
                )
        else:
            try:
                self._validate_v3_against_catalog(draft)
                self._validate_v3_spatial_assumptions(
                    draft,
                    user_instruction=request.instruction,
                )
                self._validate_v3_plan_semantics(
                    draft,
                    world_context=request.world_context,
                    total_search_time_budget_s=(
                        request.world_context.search_timeout_s
                    ),
                    trusted_target_locked=request.trusted_target_id is not None,
                    allow_trusted_safety_completion=(
                        request.allow_trusted_safety_completion
                    ),
                )
            except (OverflowError, RecursionError, TypeError, ValueError) as exc:
                raise _DraftValidationFailure(
                    "V3_CONTRACT_VIOLATION",
                    self._describe_validation_error(exc),
                ) from None
        return draft

    @staticmethod
    def _failure_diagnostics(
        *,
        model_calls: int,
        repair_used: bool,
        initial_error_code: str,
        initial_error_message: str,
    ) -> PlannerDiagnostics:
        return PlannerDiagnostics(
            model_calls=model_calls,
            repair_used=repair_used,
            repair_succeeded=False,
            initial_output_valid=False,
            final_output_valid=False,
            initial_error_code=initial_error_code,
            initial_error_message=initial_error_message,
            structured_output_enabled=True,
        )

    def _validate_against_catalog(self, draft: SkillPlanDraft) -> None:
        """Enforce the exact catalog shown in this planner invocation."""

        contracts = {contract.name: contract for contract in self._skill_catalog}
        for step in draft.steps:
            contract = contracts.get(step.skill)
            if (
                contract is None
                or not contract.top_level_allowed
                or contract.recovery_only
            ):
                raise ValueError(
                    f"step {step.id} uses a Skill unavailable for top-level planning"
                )
            self._validate_catalog_arguments(
                step.args,
                contract.arguments,
                prefix=f"step {step.id}",
            )

            if step.recovery is None:
                continue
            recovery_contract = contracts.get(step.recovery.skill)
            if (
                recovery_contract is None
                or recovery_contract.top_level_allowed
                or not recovery_contract.recovery_only
            ):
                raise ValueError(
                    f"step {step.id} uses a recovery absent from the active catalog"
                )
            recovery_data = step.recovery.to_dict()
            recovery_data.pop("skill", None)
            self._validate_catalog_arguments(
                recovery_data,
                recovery_contract.arguments,
                prefix=f"step {step.id} recovery",
            )

    @staticmethod
    def _validate_v3_catalog_shape(catalog: SkillCatalog) -> None:
        """Reject accidentally supplying the named-location-only V2 catalog."""

        try:
            goto = catalog.get("GOTO")
            search = catalog.get("SEARCH")
        except KeyError as exc:
            raise ValueError("Spatial V3 catalog must include GOTO and SEARCH") from exc
        goto_arguments = {argument.name: argument for argument in goto.arguments}
        search_arguments = {argument.name: argument for argument in search.arguments}
        if (
            "target" not in goto_arguments
            or goto_arguments["target"].value_type != "object"
            or "destination" in goto_arguments
        ):
            raise ValueError(
                "Spatial V3 GOTO catalog must expose object target, not destination"
            )
        for name, expected_type in (("region", "object"), ("strategy", "object")):
            argument = search_arguments.get(name)
            if argument is None or argument.value_type != expected_type:
                raise ValueError(
                    f"Spatial V3 SEARCH catalog must expose {name} as an object"
                )

    def _validate_v3_against_catalog(self, draft: SkillPlanDraftV3) -> None:
        contracts = {contract.name: contract for contract in self._skill_catalog}
        for step in draft.steps:
            contract = contracts.get(step.skill)
            if (
                contract is None
                or not contract.top_level_allowed
                or contract.recovery_only
            ):
                raise ValueError(
                    f"step {step.id} uses a Skill unavailable for V3 initial planning"
                )
            specifications = {item.name: item for item in contract.arguments}
            if not set(step.args).issubset(specifications):
                raise ValueError(
                    f"step {step.id} uses arguments absent from the V3 catalog"
                )
            required = {
                item.name for item in contract.arguments if item.required
            }
            if not required.issubset(step.args):
                raise ValueError(
                    f"step {step.id} omits an argument required by the V3 catalog"
                )
            for name, value in step.args.items():
                specification = specifications[name]
                if specification.value_type == "object":
                    if not (
                        isinstance(value, Mapping)
                        or callable(getattr(value, "to_dict", None))
                    ):
                        raise ValueError(
                            f"step {step.id}.{name} violates the V3 object contract"
                        )
                elif specification.value_type == "array":
                    if isinstance(value, (str, bytes)) or not isinstance(
                        value, (tuple, list)
                    ):
                        raise ValueError(
                            f"step {step.id}.{name} violates the V3 array contract"
                        )
                else:
                    self._validate_catalog_value(
                        value,
                        specification,
                        prefix=f"step {step.id}",
                    )
            if step.skill == "GOTO" and isinstance(
                step.args.get("target"), RouteTarget
            ):
                raise ValueError(
                    f"step {step.id} uses a ROUTE target; use FOLLOW_ROUTE"
                )
            if step.skill == "GOTO" and isinstance(
                target := step.args.get("target"), PointTarget
            ) and (
                target.frame is CoordinateFrame.WORLD_ENU
                and target.xyz_m[2] <= 0.0
            ):
                raise ValueError(
                    f"GOTO step {step.id} WORLD_ENU POINT target z must be "
                    "greater than zero"
                )

            if step.recovery is None:
                continue
            recovery_contract = contracts.get(step.recovery.skill)
            if (
                recovery_contract is None
                or recovery_contract.top_level_allowed
                or not recovery_contract.recovery_only
            ):
                raise ValueError(
                    f"step {step.id} uses a recovery absent from the V3 catalog"
                )
            recovery_values = step.recovery.to_dict()
            recovery_values.pop("skill", None)
            self._validate_catalog_arguments(
                recovery_values,
                recovery_contract.arguments,
                prefix=f"step {step.id} recovery",
            )

    @staticmethod
    def _validate_v3_spatial_assumptions(
        draft: SkillPlanDraftV3,
        *,
        user_instruction: str,
    ) -> None:
        """Bind model assumptions to real text and explicit coordinate frames."""

        source_spans: list[tuple[int, int, bool]] = []
        for assumption in draft.assumptions:
            source_text = assumption.source_text
            if source_text not in user_instruction:
                raise ValueError(
                    "V3 assumption.source_text must be an exact substring of "
                    "the trusted user instruction"
                )
            has_explicit_frame = (
                _SPATIAL_FRAME_PATTERN.search(assumption.interpretation)
                is not None
            )
            offset = 0
            while True:
                start = user_instruction.find(source_text, offset)
                if start < 0:
                    break
                source_spans.append(
                    (start, start + len(source_text), has_explicit_frame)
                )
                offset = start + 1

        for start, end, phrase in _ambiguous_relative_direction_spans(
            user_instruction
        ):
            covering = tuple(
                has_frame
                for left, right, has_frame in source_spans
                if left <= start and end <= right
            )
            if covering:
                if any(covering):
                    continue
                raise ValueError(
                    f"ambiguous relative direction {phrase!r} requires its "
                    "V3 assumption interpretation to name an explicit "
                    "coordinate frame"
                )
            raise ValueError(
                f"ambiguous relative direction {phrase!r} requires a V3 "
                "assumption whose source_text covers the original phrase"
            )

    def _validate_v3_plan_semantics(
        self,
        draft: SkillPlanDraftV3,
        *,
        world_context: PlannerWorldContext,
        total_search_time_budget_s: float,
        trusted_target_locked: bool = False,
        allow_trusted_safety_completion: bool = False,
    ) -> None:
        """Apply bounded initial-plan invariants without compiling geometry."""

        if len(draft.steps) > self._planner_limits.max_plan_steps:
            raise ValueError("V3 plan exceeds max_plan_steps")
        skills = tuple(step.skill for step in draft.steps)
        if skills[0] != "TAKEOFF" or skills.count("TAKEOFF") != 1:
            raise ValueError("V3 plan must begin with exactly one TAKEOFF")
        if not isinstance(allow_trusted_safety_completion, bool):
            raise TypeError("allow_trusted_safety_completion must be bool")
        if allow_trusted_safety_completion:
            if skills.count("LAND") > 1:
                raise ValueError("V3 plan may contain at most one LAND")
            if skills.count("LAND") == 1 and skills[-1] != "LAND":
                raise ValueError("V3 LAND must be the final model step")
            completion = analyze_trusted_safety_completion(
                draft,
                world_context,
            )
            if (
                len(draft.steps) + completion.additional_steps
                > self._planner_limits.max_plan_steps
            ):
                raise ValueError(
                    "V3 model plan leaves no step budget for trusted safety "
                    "completion"
                )
        elif skills[-1] != "LAND" or skills.count("LAND") != 1:
            raise ValueError("V3 plan must end with exactly one LAND")
        for skill, maximum in (
            ("GOTO", self._planner_limits.max_goto_calls),
            ("SEARCH", max(1, self._planner_limits.max_plan_steps - 2)),
            ("TRACK", self._planner_limits.max_track_calls),
        ):
            if skills.count(skill) > maximum:
                raise ValueError(f"V3 plan exceeds {skill} call limit")
        if (
            allow_trusted_safety_completion
            and skills.count("GOTO") + completion.additional_gotos
            > self._planner_limits.max_goto_calls
        ):
            raise ValueError(
                "V3 model plan leaves no GOTO budget for trusted safety "
                "completion"
            )

        search_ids: set[str] = set()
        total_search_time_s = 0.0
        total_reacquire_attempts = 0
        for step in draft.steps:
            if step.skill == "SEARCH":
                strategy = step.args.get("strategy")
                if not isinstance(strategy, SearchStrategySpec):
                    raise ValueError(
                        f"SEARCH step {step.id} strategy must be a SearchStrategySpec"
                    )
                if not self._search_runtime_capabilities.supports(strategy.kind):
                    raise ValueError(
                        f"SEARCH step {step.id} strategy {strategy.kind.value} "
                        "is unavailable in the negotiated runtime capabilities"
                    )
                region = step.args.get("region")
                required_region_type = {
                    SearchStrategyType.PERIMETER_V1: (CircleRegion, "CIRCLE"),
                    SearchStrategyType.SECTOR_SWEEP: (SectorRegion, "SECTOR"),
                    SearchStrategyType.CORRIDOR_FOLLOW: (
                        CorridorRegion,
                        "CORRIDOR",
                    ),
                }.get(strategy.kind)
                if (
                    required_region_type is not None
                    and not isinstance(region, required_region_type[0])
                ):
                    raise ValueError(
                        f"SEARCH step {step.id} strategy {strategy.kind.value} "
                        f"requires a {required_region_type[1]} region"
                    )
                search_ids.add(step.id)
                timeout_s = step.args.get("timeout_s")
                if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real):
                    raise ValueError(
                        f"SEARCH step {step.id} timeout_s must be numeric"
                    )
                total_search_time_s += float(timeout_s)
            if step.skill != "TRACK":
                continue
            reference = step.args.get("target_ref")
            expected_refs = {f"${step_id}.target_id" for step_id in search_ids}
            if trusted_target_locked:
                expected_refs.add("$trusted_target.target_id")
            if reference not in expected_refs:
                raise ValueError(
                    f"TRACK step {step.id} must reference a prior SEARCH "
                    "target_id or an advertised trusted target lock"
                )
            duration_s = step.args.get("duration_s")
            if isinstance(duration_s, bool) or not isinstance(duration_s, Real):
                raise ValueError(f"TRACK step {step.id} duration_s must be numeric")
            duration = float(duration_s)
            if not (
                self._planner_limits.min_track_duration_s
                <= duration
                <= self._planner_limits.max_track_duration_s
            ):
                raise ValueError(
                    f"TRACK step {step.id} duration_s is outside planner limits"
                )
            if step.recovery is not None:
                attempts = step.recovery.max_attempts
                if attempts > self._planner_limits.max_reacquire_attempts_per_track:
                    raise ValueError("V3 TRACK recovery exceeds per-step budget")
                total_reacquire_attempts += attempts
        if total_reacquire_attempts > self._planner_limits.max_total_reacquire_attempts:
            raise ValueError("V3 plan exceeds total REACQUIRE budget")
        if (
            isinstance(total_search_time_budget_s, bool)
            or not isinstance(total_search_time_budget_s, Real)
            or not isfinite(float(total_search_time_budget_s))
            or float(total_search_time_budget_s) <= 0.0
        ):
            raise ValueError("V3 total SEARCH time budget must be positive and finite")
        if total_search_time_s > float(total_search_time_budget_s) + 1e-9:
            raise ValueError(
                "V3 plan exceeds total SEARCH time budget: "
                f"{total_search_time_s:g} > {float(total_search_time_budget_s):g}"
            )

        if allow_trusted_safety_completion and "LAND" not in skills:
            return

        return_index = len(draft.steps) - 2
        while return_index >= 0 and draft.steps[return_index].skill == "HOVER":
            return_index -= 1
        if return_index < 0 or draft.steps[return_index].skill != "GOTO":
            raise ValueError(
                "V3 LAND requires a prior landing-zone GOTO; only HOVER may intervene"
            )
        return_target = draft.steps[return_index].args.get("target")
        landing_zone = draft.steps[-1].args.get("zone")
        if (
            not isinstance(return_target, NamedLocationTarget)
            or return_target.name != landing_zone
        ):
            raise ValueError(
                "V3 landing-zone GOTO must target the same named LAND zone"
            )

    @staticmethod
    def _validate_catalog_conditions(catalog: SkillCatalog) -> None:
        """Reject condition prose that the v1 validator cannot execute."""

        for contract in catalog:
            for argument in contract.arguments:
                if argument.condition is None:
                    continue
                if not (
                    argument.name == "yaw_deg"
                    and argument.condition == _FIXED_YAW_CONDITION
                ):
                    raise ValueError(
                        f"{contract.name}.{argument.name} uses an unsupported "
                        "catalog condition"
                    )

    @classmethod
    def _validate_catalog_arguments(
        cls,
        values: Mapping[str, object],
        specifications: tuple[SkillArgumentSpec, ...],
        *,
        prefix: str,
    ) -> None:
        specs = {argument.name: argument for argument in specifications}
        if not set(values).issubset(specs):
            raise ValueError(
                f"{prefix} uses arguments absent from the active catalog"
            )
        required = {
            argument.name for argument in specifications if argument.required
        }
        if not required.issubset(values):
            raise ValueError(
                f"{prefix} omits an argument required by the active catalog"
            )
        for name, value in values.items():
            cls._validate_catalog_value(value, specs[name], prefix=prefix)

        # This is the only conditional expression supported by the v1 JSON
        # protocol.  SkillPlanDraft already enforces it; repeat the check here
        # so a custom catalog can never weaken or reinterpret the contract.
        for argument in specifications:
            if argument.condition != _FIXED_YAW_CONDITION:
                continue
            if values.get("yaw_mode") == "FIXED" and "yaw_deg" not in values:
                raise ValueError(f"{prefix} FIXED yaw_mode requires yaw_deg")
            if "yaw_deg" in values and values.get("yaw_mode") != "FIXED":
                raise ValueError(f"{prefix} yaw_deg is only allowed for FIXED yaw")

    @staticmethod
    def _validate_catalog_value(
        value: object,
        specification: SkillArgumentSpec,
        *,
        prefix: str,
    ) -> None:
        field = f"{prefix}.{specification.name}"
        if specification.value_type == "string":
            if not isinstance(value, str):
                raise ValueError(f"{field} violates the active catalog type")
        elif specification.value_type == "number":
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{field} violates the active catalog type")
            try:
                normalized_number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError(
                    f"{field} violates the active catalog type"
                ) from None
            if not isfinite(normalized_number):
                raise ValueError(f"{field} must be finite")
        elif specification.value_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field} violates the active catalog type")
            normalized_number = float(value)
        else:  # SkillArgumentSpec validates this at catalog construction.
            raise ValueError(f"{field} has an unsupported active catalog type")

        if (
            specification.allowed_values
            and value not in specification.allowed_values
        ):
            raise ValueError(
                f"{field} is outside the active catalog allowed_values"
            )

        if specification.minimum is not None or specification.maximum is not None:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"{field} cannot be checked against catalog numeric bounds"
                )
            try:
                normalized_number = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ValueError(
                    f"{field} cannot be checked against catalog numeric bounds"
                ) from None
            if not isfinite(normalized_number):
                raise ValueError(f"{field} must be finite")
            if (
                specification.minimum is not None
                and normalized_number < specification.minimum
            ):
                raise ValueError(f"{field} is below the active catalog minimum")
            if (
                specification.maximum is not None
                and normalized_number > specification.maximum
            ):
                raise ValueError(f"{field} exceeds the active catalog maximum")

    def _build_repair_prompt(
        self,
        original_output: str,
        validation_failure: _DraftValidationFailure,
    ) -> str:
        issues = validation_failure.issues or ()
        validation_issues = [
            {
                "code": issue.code.value,
                "step_id": issue.step_id,
                "message": issue.message,
            }
            for issue in issues
        ]
        if not validation_issues:
            validation_issues = [
                {
                    "code": validation_failure.code,
                    "step_id": None,
                    "message": validation_failure.message,
                }
            ]
        mandatory_repairs = [
            hint
            for issue in validation_issues
            if (
                hint := self._repair_hint(
                    issue["code"],
                    issue.get("message"),
                )
            )
            is not None
        ]
        schema_label = (
            "schema-v2" if self._planning_contract is PlanningContract.V2 else "schema-v3"
        )
        if self._planning_contract is PlanningContract.V2:
            requirements = (
                "Follow the system rules, trusted world context, Skill Catalog, "
                "and planner limits. Return only corrected JSON with no Markdown "
                "or explanation. Do not add coordinates or low-level controls."
            )
        else:
            requirements = (
                "Follow the Spatial V3 system rules, trusted routing, Skill Catalog, "
                "and planner limits. Return only corrected JSON with no Markdown or "
                "explanation; emit compact JSON without indentation or unnecessary "
                "whitespace. Framed V3 coordinates and region geometry are allowed; "
                "preserve explicit coordinate frames and spatial assumptions. Do not "
                "add low-level controls or hidden truth. A TRACK target_ref must "
                "use the exact id of an earlier SEARCH as '$<id>.target_id', or "
                "the exact '$trusted_target.target_id' token only when the trusted "
                "context advertises a confirmed target lock."
            )
        payload = {
            "task": (
                "Repair the previous output into one valid routed "
                f"SkillPlanDraft {schema_label} JSON object."
            ),
            "previous_output_was_rejected": True,
            "must_change_rejected_output": True,
            "steps_must_cover_entire_user_instruction": True,
            "original_output": original_output,
            "validation_issues": validation_issues,
            "mandatory_repairs": mandatory_repairs,
            "requirements": requirements,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _compact_repair_context(message: ChatMessage) -> ChatMessage:
        """Drop redundant catalog prose while retaining trusted semantics."""

        if not isinstance(message, ChatMessage) or message.role != "user":
            raise TypeError("initial repair context must be a user ChatMessage")
        if not isinstance(message.content, str):
            raise TypeError("initial repair context must be text-only")
        try:
            original = json.loads(message.content)
        except json.JSONDecodeError:
            raise ValueError("initial repair context must contain valid JSON") from None
        if not isinstance(original, Mapping):
            raise ValueError("initial repair context must be a JSON object")
        compact = {
            "task": "Preserve the original mission semantics while repairing the JSON.",
            **{
                key: original[key]
                for key in _REPAIR_CONTEXT_KEYS
                if key in original
            },
        }
        return ChatMessage(
            role="user",
            content=json.dumps(
                compact,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def _repair_hint(
        self,
        code: object,
        message: object = None,
    ) -> str | None:
        if (
            self._planning_contract is PlanningContract.V2
            and code == "LAND_GOTO_MISSING"
        ):
            return (
                "Insert exactly one GOTO immediately before the final LAND. "
                "Set GOTO.args.destination exactly equal to LAND.args.zone, "
                "use a new unique step id, and preserve the trusted uav_id."
            )
        if (
            self._planning_contract is PlanningContract.V3
            and code == "V3_CONTRACT_VIOLATION"
            and isinstance(message, str)
            and "POINT target z" in message
        ):
            return (
                "For every WORLD_ENU GOTO POINT, set target.xyz_m[2] to a "
                "positive flight altitude. Do not navigate to a ground-level "
                "search-region center; SEARCH enters its RegionSpec itself. "
                "Return home with NAMED_LOCATION, not a ground-level POINT."
            )
        if (
            self._planning_contract is PlanningContract.V3
            and code == "V3_CONTRACT_VIOLATION"
            and isinstance(message, str)
            and "strategy" in message.casefold()
            and "region" in message.casefold()
        ):
            return (
                "Use a SEARCH strategy compatible with region.shape: "
                "PERIMETER_V1 only with CIRCLE, SECTOR_SWEEP only with SECTOR, "
                "and CORRIDOR_FOLLOW only with CORRIDOR. Preserve the model's "
                "region geometry and choose the compatible macro strategy "
                "yourself; do not ask the runtime to replace it."
            )
        if (
            self._planning_contract is PlanningContract.V3
            and code == "V3_CONTRACT_VIOLATION"
            and isinstance(message, str)
            and "assumption" in message.casefold()
        ):
            return (
                "For every reported ambiguous relative direction, add an "
                "assumptions entry whose source_text is an exact substring of "
                "the user instruction and whose interpretation explicitly names "
                "WORLD_ENU, HOME_ENU, UAV_START_FLU, UAV_HOLD_FLU, or CAMERA_FLU."
            )
        if (
            self._planning_contract is PlanningContract.V3
            and code == "V3_CONTRACT_VIOLATION"
            and isinstance(message, str)
            and "LAND" in message
        ):
            return (
                "The previous steps are an incomplete prefix; expand them instead "
                "of repeating them. Add every omitted user-requested action "
                "(including TRACK when requested). End steps with exactly one "
                "LAND. Before that LAND, return with "
                "a GOTO NAMED_LOCATION target whose name equals LAND.args.zone; "
                "only position-preserving HOVER steps may appear between them. For "
                "For TRACK, use an earlier SEARCH '$<id>.target_id' reference, or "
                "'$trusted_target.target_id' only when the trusted context explicitly "
                "advertises a confirmed target lock."
            )
        return None

    @classmethod
    def _parse_plan_draft(cls, raw_output: str) -> SkillPlanDraft:
        """Parse the retained schema-v1 compatibility representation only."""

        return SkillPlanDraft.from_dict(cls._parse_json_object(raw_output))

    @classmethod
    def _parse_plan_draft_v2(cls, raw_output: str) -> SkillPlanDraftV2:
        """Parse one initial routed schema-v2 response without invoking a model.

        INSPECT is intentionally revision-only: an initial plan has no trusted
        CandidateBank identifier to bind its candidate argument to.
        """

        parsed = cls._parse_json_object(raw_output)
        if "target_spec" not in parsed:
            raise ValueError("schema-v2 initial plan must include target_spec")
        draft = SkillPlanDraftV2.from_dict(parsed)
        cls._require_initial_plan_skills(draft)
        return draft

    @classmethod
    def _parse_plan_draft_v3(cls, raw_output: str) -> SkillPlanDraftV3:
        """Parse one routed Spatial Contract V3 initial response."""

        parsed = cls._parse_json_object(raw_output)
        cls._reject_v3_goto_route_targets(parsed)
        draft = SkillPlanDraftV3.from_dict(parsed)
        cls._require_initial_plan_skills(draft)
        return draft

    @staticmethod
    def _reject_v3_goto_route_targets(parsed: Mapping[str, object]) -> None:
        """Reject multi-point GOTO at the raw V3 parsing boundary."""

        steps = parsed.get("steps")
        if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
            return
        for index, raw_step in enumerate(steps):
            if not isinstance(raw_step, Mapping) or raw_step.get("skill") != "GOTO":
                continue
            args = raw_step.get("args")
            target = args.get("target") if isinstance(args, Mapping) else None
            if not isinstance(target, Mapping) or target.get("kind") != "ROUTE":
                continue
            step_id = raw_step.get("id")
            label = step_id if isinstance(step_id, str) and step_id else f"index {index}"
            raise ValueError(
                f"V3 GOTO step {label} cannot use a ROUTE target; "
                "use FOLLOW_ROUTE with a trusted route_ref"
            )

    @staticmethod
    def _require_initial_plan_skills(
        draft: SkillPlanDraftV2 | SkillPlanDraftV3,
    ) -> None:
        if any(step.skill == "INSPECT" for step in draft.steps):
            raise ValueError(
                "INSPECT is unavailable in an initial plan without a trusted "
                "runtime candidate"
            )

    @classmethod
    def _parse_json_object(cls, raw_output: str) -> Mapping[str, object]:
        if not isinstance(raw_output, str):
            raise TypeError("model output must be a string")
        text = raw_output.strip()
        if not text:
            raise ValueError("model output must be non-empty")

        fence_match = _JSON_FENCE.fullmatch(text)
        if fence_match is not None:
            text = fence_match.group("body").strip()
            if not text:
                raise ValueError("JSON code fence must contain an object")
        elif text.startswith("```") or text.endswith("```"):
            raise ValueError("only a single ```json ... ``` wrapper is supported")

        parsed = json.loads(
            text,
            parse_constant=cls._reject_nonfinite_constant,
            object_pairs_hook=cls._strict_object,
        )
        if not isinstance(parsed, Mapping):
            raise TypeError("model output must be one JSON object")
        return parsed

    @staticmethod
    def _reject_nonfinite_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    @staticmethod
    def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, value in pairs:
            if key in parsed:
                raise _DuplicateJSONKeyError(f"duplicate JSON field: {key}")
            parsed[key] = value
        return parsed

    @staticmethod
    def _describe_validation_error(error: Exception) -> str:
        if isinstance(error, json.JSONDecodeError):
            return (
                f"invalid JSON at line {error.lineno}, column {error.colno}: "
                f"{error.msg}"
            )
        return f"{type(error).__name__}: {error}"

    def _safe_log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        try:
            method = getattr(self._logger, level, None)
            if callable(method):
                method(message)
            elif callable(self._logger):
                self._logger(message)
        except Exception:
            pass


__all__ = ["DynamicLLMPlanner", "PlanningContract"]
