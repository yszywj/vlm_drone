"""Pure-Python contracts shared by model client implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from numbers import Real
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


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


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
