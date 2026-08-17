"""Text-only LLM planner for the constrained dynamic SkillPlanDraft protocol."""

from __future__ import annotations

from collections.abc import Mapping
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import re

from models.base import ChatMessage, GenerationOptions, ModelClient, ModelResponse
from planner.base import MissionPlanner, PlannerError, PlannerOutputError
from planner.prompt_builder import build_dynamic_skill_planner_messages
from planner.schemas import PlannerRequest, SkillPlanDraft
from planner.skill_catalog import (
    SkillArgumentSpec,
    SkillCatalog,
    build_default_skill_catalog,
)


_JSON_FENCE = re.compile(
    r"\A```json[ \t]*(?:\r?\n)?(?P<body>.*?)\r?\n?```\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_FIXED_YAW_CONDITION = "only allowed and required when yaw_mode is FIXED"


class _DuplicateJSONKeyError(ValueError):
    """Internal marker for ambiguous JSON objects."""


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

        # Mappings are snapshotted so caller mutation cannot change a planner
        # after construction.  Immutable PlannerLimits dataclasses can be kept
        # directly and are projected through a fixed allow-list by the builder.
        if isinstance(planner_limits, Mapping):
            planner_limits = dict(planner_limits)

        self._model_client = model_client
        self._system_prompt = prompt.strip()
        self._skill_catalog = skill_catalog
        self._planner_limits = planner_limits
        self._logger = logger
        self._generation_options = GenerationOptions(temperature=0.0)

    @property
    def skill_catalog(self) -> SkillCatalog:
        return self._skill_catalog

    def plan(self, request: PlannerRequest) -> SkillPlanDraft:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")

        initial_messages = build_dynamic_skill_planner_messages(
            request.instruction,
            request.world_context,
            self._skill_catalog,
            self._planner_limits,
            self._system_prompt,
        )
        self._safe_log("debug", "dynamic Skill plan model call started")
        first_output = self._chat_content(initial_messages)
        try:
            draft = self._parse_plan_draft(first_output)
            self._validate_against_catalog(draft)
            self._validate_repairable_structure(draft)
        except (
            json.JSONDecodeError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
        ) as first_error:
            validation_error = self._describe_validation_error(first_error)
            self._safe_log(
                "warning",
                "dynamic Skill plan output was invalid; requesting one repair",
            )
        else:
            self._safe_log("debug", "dynamic Skill plan model call succeeded")
            return draft

        repair_messages = (
            *initial_messages,
            ChatMessage(role="assistant", content=first_output),
            ChatMessage(
                role="user",
                content=self._build_repair_prompt(
                    first_output,
                    validation_error,
                ),
            ),
        )
        repaired_output = self._chat_content(repair_messages)
        try:
            draft = self._parse_plan_draft(repaired_output)
            self._validate_against_catalog(draft)
            self._validate_repairable_structure(draft)
        except (
            json.JSONDecodeError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
        ) as second_error:
            detail = self._describe_validation_error(second_error)
            self._safe_log("error", "dynamic Skill plan repair output was invalid")
            raise PlannerOutputError(
                "model failed to produce a valid SkillPlanDraft after one repair: "
                f"{detail}"
            ) from None

        self._safe_log("debug", "dynamic Skill plan repair succeeded")
        return draft

    def _chat_content(self, messages: tuple[ChatMessage, ...]) -> str:
        response = self._model_client.chat(
            messages,
            options=self._generation_options,
        )
        if not isinstance(response, ModelResponse):
            raise PlannerOutputError(
                "model client returned an invalid response object"
            )
        return response.content

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
    def _validate_repairable_structure(draft: SkillPlanDraft) -> None:
        """Catch cross-step mistakes that should consume the one repair call.

        World geometry and final authority remain in PlanValidator.  This
        narrow check exists because LandSkill descends in place: a model that
        omits the matching return GOTO can be corrected before the draft is
        handed to the trusted compiler.  The compact TAKEOFF -> LAND form is
        left to PlanValidator, which alone knows the initial landing geometry.
        """

        steps = draft.steps
        for index, step in enumerate(steps):
            if step.skill != "LAND":
                continue
            if index == 1 and steps[0].skill == "TAKEOFF":
                continue
            if index == 0:
                raise ValueError(
                    "LAND must be immediately preceded by GOTO to the same zone"
                )
            previous = steps[index - 1]
            if (
                previous.skill != "GOTO"
                or previous.args.get("destination") != step.args.get("zone")
            ):
                raise ValueError(
                    "LAND must be immediately preceded by GOTO to the same zone"
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
        validation_error: str,
    ) -> str:
        payload = {
            "task": (
                "Repair the previous output into one valid SkillPlanDraft "
                "JSON object."
            ),
            "original_output": original_output,
            "validation_error": validation_error,
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

    @classmethod
    def _parse_plan_draft(cls, raw_output: str) -> SkillPlanDraft:
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
        return SkillPlanDraft.from_dict(parsed)

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
