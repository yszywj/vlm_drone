"""Model-client contracts and OpenAI-compatible HTTP implementation."""

from models.async_worker import (
    AsyncModelRequest,
    AsyncModelResult,
    AsyncModelWorker,
    AsyncVisionWorker,
)
from models.base import (
    ChatContentPart,
    ChatMessage,
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_IMAGES_PER_REQUEST,
    GenerationOptions,
    ImageURLContentPart,
    JsonSchemaResponseFormat,
    ModelClient,
    ModelClientError,
    ModelConnectionError,
    ModelHTTPError,
    ModelProtocolError,
    ModelResponse,
    TextContentPart,
)
from models.openai_compatible_client import OpenAICompatibleClient
from models.image_encoding import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_SIDE_PX,
    encode_rgb_to_data_url,
)

__all__ = [
    "AsyncModelRequest",
    "AsyncModelResult",
    "AsyncModelWorker",
    "AsyncVisionWorker",
    "ChatContentPart",
    "ChatMessage",
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_MAX_IMAGE_BYTES",
    "DEFAULT_MAX_IMAGES_PER_REQUEST",
    "DEFAULT_MAX_SIDE_PX",
    "GenerationOptions",
    "ImageURLContentPart",
    "JsonSchemaResponseFormat",
    "ModelClient",
    "ModelClientError",
    "ModelConnectionError",
    "ModelHTTPError",
    "ModelProtocolError",
    "ModelResponse",
    "OpenAICompatibleClient",
    "TextContentPart",
    "encode_rgb_to_data_url",
]
