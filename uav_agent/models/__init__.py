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
from models.adapter_registry import (
    AdapterRegistry,
    AdapterRegistryError,
    AdapterSelection,
    AdapterSpec,
    AdapterStatus,
    ModelCallRole,
)
from models.model_client_factory import ModelClientFactory
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
    "AdapterRegistry",
    "AdapterRegistryError",
    "AdapterSelection",
    "AdapterSpec",
    "AdapterStatus",
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
    "ModelClientFactory",
    "ModelConnectionError",
    "ModelHTTPError",
    "ModelProtocolError",
    "ModelResponse",
    "ModelCallRole",
    "OpenAICompatibleClient",
    "TextContentPart",
    "encode_rgb_to_data_url",
]
