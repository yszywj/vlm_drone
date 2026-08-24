"""Qwen-backed natural-language to :class:`FleetTaskSpecV1` interpreter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import TypeAlias

from common.ids import validate_routing_id, validate_uav_id
from models import (
    ChatMessage,
    GenerationOptions,
    JsonSchemaResponseFormat,
    ModelClient,
    ModelResponse,
)
from planner.diagnostics import PlannerDiagnostics
from planner.spatial import CoordinateFrame

from fleet.task_spec import (
    FleetTaskSpecError,
    FleetTaskSpecV1,
    parse_fleet_task_spec,
    reject_forbidden_task_fields,
)
from fleet.task_spec_json_schema import build_fleet_task_spec_json_schema
from fleet.strict_json import DuplicateJSONKeyError, strict_json_object_loads


AliasDirectory: TypeAlias = Mapping[str, str] | Sequence[str]
DEFAULT_SYSTEM_PROMPT = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "fleet_task_interpreter_system.txt"
)
MAX_PROPOSAL_BYTES = 32_768
MAX_RESPONSE_BYTES = 32_768


class FleetTaskInterpretationError(RuntimeError):
    """Raised after the bounded interpretation/repair budget is exhausted."""


def _normalize_alias_directory(
    value: AliasDirectory,
    *,
    name: str,
    uav: bool,
    allow_empty: bool,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    if isinstance(value, Mapping):
        raw_items = tuple(value.items())
    elif not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
        raw_items = tuple((item, item) for item in value)
    else:
        raise TypeError(f"{name} must be a mapping or an array of canonical IDs")
    if len(raw_items) > 128:
        raise ValueError(f"{name} must contain at most 128 entries")
    aliases: list[dict[str, str]] = []
    ids: list[str] = []
    seen_aliases: set[str] = set()
    for raw_alias, raw_id in raw_items:
        if not isinstance(raw_alias, str) or not raw_alias.strip() or len(raw_alias) > 128:
            raise ValueError(f"{name} aliases must contain 1..128 characters")
        if not isinstance(raw_id, str):
            raise TypeError(f"{name} canonical IDs must be strings")
        alias = raw_alias.strip()
        canonical = (
            validate_uav_id(raw_id)
            if uav
            else validate_routing_id(raw_id, "target_alias")
        )
        folded = alias.casefold()
        if folded in seen_aliases:
            raise ValueError(f"{name} contains duplicate alias: {alias}")
        seen_aliases.add(folded)
        aliases.append({"alias": alias, "canonical_id": canonical})
        if canonical not in ids:
            ids.append(canonical)
    if not aliases and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return tuple(aliases), tuple(ids)


def _error_code(exc: Exception) -> str:
    if isinstance(exc, DuplicateJSONKeyError):
        return "DUPLICATE_JSON_KEY"
    if isinstance(exc, json.JSONDecodeError):
        return "INVALID_JSON"
    if isinstance(exc, FleetTaskSpecError):
        return "TASK_SPEC_VALIDATION_ERROR"
    if isinstance(exc, (TypeError, ValueError)):
        return "STRUCTURE_VALIDATION_ERROR"
    return type(exc).__name__.upper()


class LLMFleetTaskInterpreter:
    """Interpret mission semantics without assigning work or importing Isaac."""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        uav_alias_catalog: AliasDirectory | None = None,
        target_alias_catalog: AliasDirectory | None = None,
        supported_coordinate_frames: Sequence[CoordinateFrame | str] = (
            CoordinateFrame.WORLD_ENU,
            CoordinateFrame.HOME_ENU,
            CoordinateFrame.UAV_START_FLU,
        ),
        system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT,
        logger: object | None = None,
        max_tokens: int = 3072,
        repair_budget: int = 1,
    ) -> None:
        if not callable(getattr(model_client, "chat", None)):
            raise TypeError("model_client must provide chat()")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 256 <= max_tokens <= 8192
        ):
            raise ValueError("max_tokens must be within [256, 8192]")
        if isinstance(repair_budget, bool) or repair_budget not in (0, 1):
            raise ValueError("repair_budget must be 0 or 1")
        frames: list[CoordinateFrame] = []
        for value in supported_coordinate_frames:
            try:
                frame = (
                    value
                    if isinstance(value, CoordinateFrame)
                    else CoordinateFrame(value)
                )
            except (TypeError, ValueError):
                raise ValueError(
                    "supported_coordinate_frames contains an invalid frame"
                ) from None
            if frame not in frames:
                frames.append(frame)
        if not frames:
            raise ValueError("supported_coordinate_frames must not be empty")
        prompt_path = Path(system_prompt_path)
        try:
            system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"could not read task interpreter prompt: {exc}") from exc
        if not system_prompt:
            raise ValueError("task interpreter system prompt must not be empty")
        self._client = model_client
        self._default_uavs = uav_alias_catalog
        self._default_targets = target_alias_catalog
        self._frames = tuple(frames)
        self._system_prompt = system_prompt
        self._logger = logger
        self._max_tokens = max_tokens
        self._repair_budget = repair_budget
        self._last_diagnostics: PlannerDiagnostics | None = None
        self._model_proposals: list[dict[str, object]] = []

    @property
    def last_diagnostics(self) -> PlannerDiagnostics | None:
        return self._last_diagnostics

    @property
    def model_proposals(self) -> tuple[dict[str, object], ...]:
        return tuple(deepcopy(item) for item in self._model_proposals)

    def interpret(
        self,
        source_text: str,
        *,
        uav_alias_catalog: AliasDirectory | None = None,
        target_alias_catalog: AliasDirectory | None = None,
    ) -> FleetTaskSpecV1:
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError("source_text must be a non-empty string")
        if len(source_text) > 8192:
            raise ValueError("source_text must contain at most 8192 characters")
        uav_source = (
            self._default_uavs if uav_alias_catalog is None else uav_alias_catalog
        )
        target_source = (
            self._default_targets
            if target_alias_catalog is None
            else target_alias_catalog
        )
        if uav_source is None:
            raise ValueError("uav_alias_catalog is required")
        if target_source is None:
            target_source = ()
        uav_directory, uav_ids = _normalize_alias_directory(
            uav_source,
            name="uav_alias_catalog",
            uav=True,
            allow_empty=False,
        )
        target_directory, target_ids = _normalize_alias_directory(
            target_source,
            name="target_alias_catalog",
            uav=False,
            allow_empty=True,
        )
        schema = build_fleet_task_spec_json_schema(
            source_text=source_text,
            trusted_uav_ids=uav_ids,
            trusted_target_aliases=target_ids,
            supported_coordinate_frames=self._frames,
        )
        payload = {
            "task": "Interpret the instruction as FleetTaskSpecV1; do not assign work.",
            "source_text": source_text,
            "trusted_uav_alias_directory": list(uav_directory),
            "trusted_target_alias_directory": list(target_directory),
            "supported_coordinate_frames": [item.value for item in self._frames],
            "units": {"time": "seconds", "distance": "metres"},
        }
        initial_messages = (
            ChatMessage("system", self._system_prompt),
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
            response_format=JsonSchemaResponseFormat("fleet_task_spec_v1", schema),
        )
        self._model_proposals = []
        self._last_diagnostics = None
        messages = initial_messages
        initial_error_code: str | None = None
        initial_error_message: str | None = None
        attempts = 1 + self._repair_budget
        for attempt_index in range(attempts):
            repair = attempt_index > 0
            self._safe_log(
                "debug",
                "mission interpretation repair call started"
                if repair
                else "mission interpretation model call started",
            )
            raw = ""
            decoded: dict[str, object] | None = None
            try:
                response = self._client.chat(messages, options=options)
                if not isinstance(response, ModelResponse):
                    raise TypeError("model client returned an invalid response object")
                raw = response.content
                if len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
                    raise ValueError(
                        f"FleetTaskSpec response exceeds {MAX_RESPONSE_BYTES} bytes"
                    )
                decoded = strict_json_object_loads(raw)
                spec = parse_fleet_task_spec(
                    decoded,
                    trusted_uav_ids=uav_ids,
                    trusted_target_aliases=target_ids,
                    supported_coordinate_frames=self._frames,
                    expected_source_text=source_text,
                )
            except (
                FleetTaskSpecError,
                DuplicateJSONKeyError,
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
                    self._safe_log("error", "mission interpretation output rejected")
                    raise FleetTaskInterpretationError(
                        f"invalid FleetTaskSpecV1 output after {attempt_index + 1} "
                        f"attempt(s): {code}: {message}"
                    ) from None
                messages = (
                    *initial_messages,
                    ChatMessage(
                        "user",
                        "The prior proposal was rejected with "
                        f"{code}: {message}. Generate a fresh complete JSON object "
                        "that matches the same schema. Do not add prose or reasoning.",
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
            self._safe_log("debug", "mission interpretation model call succeeded")
            return spec
        raise AssertionError("unreachable interpretation attempt loop")

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


__all__ = [
    "AliasDirectory",
    "DEFAULT_SYSTEM_PROMPT",
    "FleetTaskInterpretationError",
    "LLMFleetTaskInterpreter",
    "MAX_PROPOSAL_BYTES",
]
