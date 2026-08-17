"""Model-client contracts and OpenAI-compatible HTTP implementation."""

from models.base import (
    ChatMessage,
    GenerationOptions,
    JsonSchemaResponseFormat,
    ModelClient,
    ModelClientError,
    ModelConnectionError,
    ModelHTTPError,
    ModelProtocolError,
    ModelResponse,
)
from models.openai_compatible_client import OpenAICompatibleClient

__all__ = [
    "ChatMessage",
    "GenerationOptions",
    "JsonSchemaResponseFormat",
    "ModelClient",
    "ModelClientError",
    "ModelConnectionError",
    "ModelHTTPError",
    "ModelProtocolError",
    "ModelResponse",
    "OpenAICompatibleClient",
]
