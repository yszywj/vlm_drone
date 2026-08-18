"""Text-only LLM planner for the constrained dynamic SkillPlanDraft protocol."""

from __future__ import annotations

from collections.abc import Mapping
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
from planner.policy import PlannerLimits, PlannerPolicy
from planner.prompt_builder import build_dynamic_skill_planner_messages
from planner.schemas import PlannerRequest, SkillPlanDraft, SkillPlanDraftV2
from planner.skill_catalog import (
    SkillArgumentSpec,
    SkillCatalog,
    build_default_skill_catalog,
    initial_planner_catalog,
)
from planner.symbolic_checker import PlanIssue, SymbolicPlanChecker


_JSON_FENCE = re.compile(
    r"\A```json[ \t]*(?:\r?\n)?(?P<body>.*?)\r?\n?```\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_FIXED_YAW_CONDITION = "only allowed and required when yaw_mode is FIXED"
# The documented local Qwen service uses a 4096-token context.  Initial output
# has enough room for a ten-step draft; the one repair call uses a smaller
# bounded budget because its prompt also contains the prior JSON and issues.
_DYNAMIC_PLAN_MAX_TOKENS = 768
_DYNAMIC_REPAIR_MAX_TOKENS = 512


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


class DynamicLLMPlanner(MissionPlanner):
    """Use a text model to select a finite high-level linear Skill plan.

    The result is intentionally unresolved: named places remain names and a
    TRACK target remains a reference to a prior SEARCH output.  Trusted Python
    code performs all geometry resolution and safety completion later.
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
    ) -> None:
        if not callable(getattr(model_client, "chat", None)):
            raise TypeError("model_client must provide a callable chat() method")
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

        if skill_catalog is None:
            skill_catalog = build_default_skill_catalog()
        if not isinstance(skill_catalog, SkillCatalog):
            raise TypeError("skill_catalog must be a SkillCatalog")
        self._validate_catalog_conditions(skill_catalog)

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
        # The complete catalog includes runtime-revision-only INSPECT.  An
        # initial model request has no trusted CandidateBank ID, so expose and
        # enforce only the initial-planning projection.
        self._skill_catalog = initial_planner_catalog(skill_catalog)
        self._planner_limits = limits
        self._planner_policy = policy
        self._symbolic_checker = SymbolicPlanChecker()
        self._logger = logger
        self._last_diagnostics: PlannerDiagnostics | None = None

    @property
    def skill_catalog(self) -> SkillCatalog:
        return self._skill_catalog

    @property
    def last_diagnostics(self) -> PlannerDiagnostics | None:
        return self._last_diagnostics

    def plan(self, request: PlannerRequest) -> SkillPlanDraftV2:
        return self.plan_with_diagnostics(request).output

    def plan_with_diagnostics(self, request: PlannerRequest) -> PlannerExecution:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")

        self._last_diagnostics = None
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
        response_format = JsonSchemaResponseFormat(
            name="skill_plan_draft_v2",
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
            self._safe_log(
                "warning",
                "dynamic Skill plan output was invalid; requesting one repair",
            )
        else:
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

        repair_messages = (
            *initial_messages,
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
    ) -> SkillPlanDraftV2:
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
            if "target_spec" not in parsed:
                raise ValueError(
                    "schema-v2 initial plan must include target_spec"
                )
            draft = SkillPlanDraftV2.from_dict(parsed)
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

    @staticmethod
    def _build_repair_prompt(
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
            if (hint := DynamicLLMPlanner._repair_hint(issue["code"])) is not None
        ]
        payload = {
            "task": (
                "Repair the previous output into one valid routed "
                "SkillPlanDraft schema-v2 JSON object."
            ),
            "original_output": original_output,
            "validation_issues": validation_issues,
            "mandatory_repairs": mandatory_repairs,
            "requirements": (
                "Follow the system rules, trusted world context, Skill Catalog, "
                "and planner limits. Return only corrected JSON with no Markdown "
                "or explanation. Do not add coordinates or low-level controls."
            ),
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _repair_hint(code: object) -> str | None:
        if code == "LAND_GOTO_MISSING":
            return (
                "Insert exactly one GOTO immediately before the final LAND. "
                "Set GOTO.args.destination exactly equal to LAND.args.zone, "
                "use a new unique step id, and preserve the trusted uav_id."
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

    @staticmethod
    def _require_initial_plan_skills(draft: SkillPlanDraftV2) -> None:
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


__all__ = ["DynamicLLMPlanner"]
