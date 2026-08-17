"""Pure-Python contracts shared by model client implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from numbers import Real
import re
from typing import Protocol


class ModelClientError(RuntimeError):
    """Base class for failures exposed by a model client."""


class ModelConnectionError(ModelClientError):
    """Raised when the model service cannot be reached before the deadline."""


class ModelHTTPError(ModelClientError):
    """Raised when the model service returns a non-success HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts


class ModelProtocolError(ModelClientError):
    """Raised when a response does not follow the expected JSON contract."""


_CHAT_ROLES = frozenset({"system", "user", "assistant"})
_JSON_SCHEMA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class _FrozenJSONDict(dict[str, object]):
    """A JSON-serializable dict snapshot that rejects in-place mutation."""

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("JSON Schema snapshots are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _copy_json_value(
    value: object,
    *,
    path: str,
    active_containers: set[int],
) -> object:
    """Validate and defensively freeze one JSON-compatible value.

    ``json.dumps`` accepts integer dictionary keys by coercing them to text,
    which is too permissive for a JSON Schema boundary.  This explicit walk
    therefore validates every key before the final serializer check and also
    produces an immutable snapshot of nested containers.
    """

    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError("schema must not contain circular references")
        active_containers.add(container_id)
        try:
            copied: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("schema object keys must be strings")
                copied[key] = _copy_json_value(
                    item,
                    path=f"{path}.{key}",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(container_id)
        return _FrozenJSONDict(copied)

    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError("schema must not contain circular references")
        active_containers.add(container_id)
        try:
            return tuple(
                _copy_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            )
        finally:
            active_containers.remove(container_id)

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must not contain NaN or Infinity")
        return value
    raise TypeError(f"{path} contains a value that is not JSON serializable")


def _thaw_json_value(value: object) -> object:
    """Return a fresh JSON-compatible copy of an internally frozen value."""

    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class JsonSchemaResponseFormat:
    """OpenAI-compatible named JSON Schema response constraint.

    The schema is recursively snapshotted and frozen at construction.  Use
    :meth:`to_dict` when a fresh mutable request representation is needed.
    """

    name: str
    schema: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if _JSON_SCHEMA_NAME_PATTERN.fullmatch(self.name) is None:
            raise ValueError(
                "name must be non-empty and contain only letters, numbers, "
                "underscores, and hyphens"
            )
        if not isinstance(self.schema, Mapping):
            raise TypeError("schema must be a JSON object")

        frozen_schema = _copy_json_value(
            self.schema,
            path="schema",
            active_containers=set(),
        )
        assert isinstance(frozen_schema, Mapping)

        # Keep an explicit strict serializer check at the public boundary.
        # The frozen dict/tuple snapshot intentionally remains serializable by
        # the standard-library encoder used by the HTTP client.
        try:
            json.dumps(frozen_schema, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            raise TypeError("schema must be a JSON-serializable object") from None
        object.__setattr__(self, "schema", frozen_schema)

    def to_dict(self) -> dict[str, object]:
        """Return a fresh OpenAI ``json_schema`` payload."""

        return {
            "name": self.name,
            "schema": _thaw_json_value(self.schema),
        }


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One OpenAI-compatible chat message.

    ``content`` is deliberately typed as ``object`` so a later phase can add
    OpenAI-compatible multimodal content lists without changing this public
    schema.  This phase only permits text.
    """

    role: str
    content: object

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in _CHAT_ROLES:
            allowed = ", ".join(sorted(_CHAT_ROLES))
            raise ValueError(f"role must be one of: {allowed}")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string in the text-only client")

        # Keep JSON serializability an explicit boundary invariant.  The check
        # remains useful when multimodal content objects are introduced later.
        try:
            json.dumps(self.content, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("content must be JSON serializable") from exc

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Validated sampling options for a single chat completion."""

    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    response_format: JsonSchemaResponseFormat | None = None

    def __post_init__(self) -> None:
        temperature = _finite_float(self.temperature, "temperature")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        object.__setattr__(self, "temperature", temperature)

        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        top_p = _finite_float(self.top_p, "top_p")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be greater than 0.0 and at most 1.0")
        object.__setattr__(self, "top_p", top_p)

        if self.response_format is not None and not isinstance(
            self.response_format,
            JsonSchemaResponseFormat,
        ):
            raise TypeError(
                "response_format must be a JsonSchemaResponseFormat or None"
            )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized text response returned by a model client."""

    content: str
    model: str | None
    finish_reason: str | None
    usage: dict[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        for field_name, value in (
            ("model", self.model),
            ("finish_reason", self.finish_reason),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")

        if not isinstance(self.usage, Mapping):
            raise TypeError("usage must be a mapping of string keys to integers")
        usage: dict[str, int] = {}
        for key, value in self.usage.items():
            if not isinstance(key, str):
                raise TypeError("usage keys must be strings")
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("usage values must be integers")
            if value < 0:
                raise ValueError("usage values must be non-negative")
            usage[key] = value
        object.__setattr__(self, "usage", usage)


class ModelClient(Protocol):
    """Minimal interface consumed by future planner/model integrations."""

    def healthcheck(self) -> None:
        """Raise ``ModelClientError`` unless the service is healthy."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        """Generate one text response for an ordered message sequence."""
