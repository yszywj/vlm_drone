"""Text-only LLM planner that produces a strict high-level MissionIntent."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import re

from models.base import ChatMessage, GenerationOptions, ModelClient, ModelResponse
from planner.base import MissionPlanner, PlannerError, PlannerOutputError
from planner.diagnostics import PlannerDiagnostics, PlannerExecution
from planner.prompt_builder import build_mission_planner_messages
from planner.schemas import MissionIntent, PlannerRequest


_JSON_FENCE = re.compile(
    r"\A```json[ \t]*(?:\r?\n)?(?P<body>.*?)\r?\n?```\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


class _DuplicateJSONKeyError(ValueError):
    """Internal marker used to make JSON object parsing unambiguous."""


class LLMPlanner(MissionPlanner):
    """Parse natural language into a ``MissionIntent`` using a text model.

    The model receives only a deliberately reduced view of the trusted world:
    scene bounds, semantic region/zone names and descriptions, and two defaults.
    Geometry used by Skills remains behind ``PlanValidator`` and is never exposed
    here.  A malformed first response gets exactly one repair attempt.
    """

    source = "llm"

    def __init__(
        self,
        model_client: ModelClient,
        system_prompt_path: str | os.PathLike[str],
        logger: object | None = None,
    ) -> None:
        if not callable(getattr(model_client, "chat", None)):
            raise TypeError("model_client must provide a callable chat() method")
        if not isinstance(system_prompt_path, (str, os.PathLike)):
            raise TypeError("system_prompt_path must be a path-like value")

        try:
            system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            # Avoid including an environment-specific path in an exception that
            # may be persisted by a caller's logging configuration.
            raise PlannerError(
                "could not read the mission planner system prompt"
            ) from None
        if not system_prompt.strip():
            raise PlannerError("mission planner system prompt must be non-empty")

        self._model_client = model_client
        self._system_prompt = system_prompt.strip()
        self._logger = logger
        # Sampling must remain deterministic for both the initial and repair
        # calls.  The immutable object is safe to reuse between plan() calls.
        self._generation_options = GenerationOptions(temperature=0.0)
        self._last_diagnostics: PlannerDiagnostics | None = None

    @property
    def last_diagnostics(self) -> PlannerDiagnostics | None:
        return self._last_diagnostics

    def plan(self, request: PlannerRequest) -> MissionIntent:
        """Return a model-produced intent, never an executable ``TaskPlan``."""

        return self.plan_with_diagnostics(request).output

    def plan_with_diagnostics(self, request: PlannerRequest) -> PlannerExecution:
        """Return the intent plus sanitized, bounded-call diagnostics."""

        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")

        self._last_diagnostics = None

        initial_messages = build_mission_planner_messages(
            request.instruction,
            request.world_context,
            self._system_prompt,
        )

        self._safe_log("debug", "mission intent model call started")
        try:
            first_output = self._chat_content(initial_messages)
        except Exception:
            self._last_diagnostics = self._failure_diagnostics(
                model_calls=1,
                repair_used=False,
                initial_error_code="MODEL_CLIENT_ERROR",
                initial_error_message="model client call failed",
            )
            raise
        first_failure: tuple[str, str] | None = None
        try:
            intent = self._validate_model_output(first_output)
        except PlannerOutputError as exc:
            code, _, message = str(exc).partition(": ")
            first_failure = (code, message or code)
            self._safe_log(
                "warning",
                "mission intent model output was invalid; requesting one repair",
            )
        else:
            self._safe_log("debug", "mission intent model call succeeded")
            diagnostics = PlannerDiagnostics(
                model_calls=1,
                repair_used=False,
                repair_succeeded=False,
                initial_output_valid=True,
                final_output_valid=True,
                initial_error_code=None,
                initial_error_message=None,
                structured_output_enabled=False,
            )
            self._last_diagnostics = diagnostics
            return PlannerExecution(output=intent, diagnostics=diagnostics)

        assert first_failure is not None
        repair_prompt = self._build_repair_prompt(
            first_output,
            f"{first_failure[0]}: {first_failure[1]}",
        )
        repair_messages = (
            *initial_messages,
            ChatMessage(role="assistant", content=first_output),
            ChatMessage(role="user", content=repair_prompt),
        )
        try:
            repaired_output = self._chat_content(repair_messages)
        except Exception:
            self._last_diagnostics = self._failure_diagnostics(
                model_calls=2,
                repair_used=True,
                initial_error_code=first_failure[0],
                initial_error_message=first_failure[1],
            )
            raise
        try:
            intent = self._validate_model_output(repaired_output)
        except PlannerOutputError as second_error:
            self._safe_log("error", "mission intent repair output was invalid")
            self._last_diagnostics = self._failure_diagnostics(
                model_calls=2,
                repair_used=True,
                initial_error_code=first_failure[0],
                initial_error_message=first_failure[1],
            )
            raise PlannerOutputError(
                "model failed to produce a valid MissionIntent after one repair: "
                f"{second_error}"
            ) from None

        self._safe_log("debug", "mission intent repair succeeded")
        diagnostics = PlannerDiagnostics(
            model_calls=2,
            repair_used=True,
            repair_succeeded=True,
            initial_output_valid=False,
            final_output_valid=True,
            initial_error_code=first_failure[0],
            initial_error_message=first_failure[1],
            structured_output_enabled=False,
        )
        self._last_diagnostics = diagnostics
        return PlannerExecution(output=intent, diagnostics=diagnostics)

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
            structured_output_enabled=False,
        )

    @classmethod
    def _validate_model_output(cls, raw_output: str) -> MissionIntent:
        try:
            parsed = cls._parse_json_object(raw_output)
        except (
            json.JSONDecodeError,
            _DuplicateJSONKeyError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            raise PlannerOutputError(
                f"INVALID_JSON: {cls._describe_validation_error(exc)}"
            ) from None
        try:
            return MissionIntent.from_dict(parsed)
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise PlannerOutputError(
                f"SCHEMA_INVALID: {cls._describe_validation_error(exc)}"
            ) from None

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

    @staticmethod
    def _build_user_prompt(request: PlannerRequest) -> str:
        """Compatibility wrapper around the shared canonical builder."""

        messages = build_mission_planner_messages(
            request.instruction,
            request.world_context,
            "placeholder",
        )
        return str(messages[1].content)

    @staticmethod
    def _build_repair_prompt(original_output: str, validation_error: str) -> str:
        # The original response is present exactly as required, but exists only
        # for this bounded retry and is never written to logs or object state.
        payload = {
            "task": "Repair the previous output into one valid MissionIntent JSON object.",
            "original_output": original_output,
            "validation_error": validation_error,
            "requirements": (
                "Follow the system rules and trusted world context. Return only "
                "the corrected JSON object with no Markdown or explanation."
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
    def _parse_intent(cls, raw_output: str) -> MissionIntent:
        return MissionIntent.from_dict(cls._parse_json_object(raw_output))

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
        except Exception:
            # Logging is observational and must never alter planning semantics.
            pass


__all__ = ["LLMPlanner"]
