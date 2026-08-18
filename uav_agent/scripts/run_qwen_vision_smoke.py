#!/usr/bin/env python3
"""Send one local RGB image to Qwen through the strict multimodal boundary."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.ids import validate_uav_id  # noqa: E402
from models.base import (  # noqa: E402
    ChatMessage,
    GenerationOptions,
    ImageURLContentPart,
    JsonSchemaResponseFormat,
    ModelClient,
    ModelConnectionError,
    ModelHTTPError,
    ModelProtocolError,
    TextContentPart,
)
from models.image_encoding import (  # noqa: E402
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_SIDE_PX,
    encode_rgb_to_data_url,
)
from models.openai_compatible_client import OpenAICompatibleClient  # noqa: E402


VISION_SMOKE_SCHEMA_NAME = "qwen_vision_smoke_v1"


class VisionSmokeError(RuntimeError):
    """Stable, non-sensitive error emitted by the smoke-test boundary."""

    def __init__(self, code: str, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


class _StrictJSONError(ValueError):
    pass


def _reject_json_constant(value: str) -> object:
    raise _StrictJSONError(f"non-finite JSON constant {value} is forbidden")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError("duplicate JSON object field is forbidden")
        result[key] = value
    return result


def _parse_strict_json(text: object) -> object:
    if not isinstance(text, str):
        raise _StrictJSONError("model output must be text")
    return json.loads(
        text,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_strict_object,
    )


def build_vision_smoke_schema(uav_id: str) -> dict[str, object]:
    """Build the strict response schema with a trusted UAV-ID echo."""

    uav_id = validate_uav_id(uav_id)
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "uav_id": {"type": "string", "const": uav_id},
            "target_present": {"type": "boolean"},
            "bbox_xyxy_normalized": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "minItems": 4,
                        "maxItems": 4,
                    },
                ]
            },
            "description": {"type": "string", "maxLength": 512},
            "self_reported_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": [
            "schema_version",
            "uav_id",
            "target_present",
            "bbox_xyxy_normalized",
            "description",
            "self_reported_confidence",
        ],
        "additionalProperties": False,
    }


def _load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise VisionSmokeError(
            "IMAGE_NOT_FOUND",
            f"image file does not exist: {path}",
            exit_code=2,
        )
    if not path.is_file():
        raise VisionSmokeError(
            "IMAGE_NOT_FILE",
            f"image path is not a regular file: {path}",
            exit_code=2,
        )
    try:
        with Image.open(path) as image:
            image.load()
            with image.convert("RGB") as converted:
                rgb = np.array(converted, dtype=np.uint8, copy=True)
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ):
        raise VisionSmokeError(
            "IMAGE_DECODE_ERROR",
            "image file could not be decoded as RGB",
            exit_code=2,
        ) from None
    return rgb


def _validate_output(value: object, *, expected_uav_id: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise VisionSmokeError(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            "model output must be one JSON object",
        )
    required = {
        "schema_version",
        "uav_id",
        "target_present",
        "bbox_xyxy_normalized",
        "description",
        "self_reported_confidence",
    }
    if set(value) != required:
        raise VisionSmokeError(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            "model output has missing or unknown fields",
        )
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise VisionSmokeError(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            "schema_version must equal 1",
        )
    if value["uav_id"] != expected_uav_id:
        raise VisionSmokeError(
            "MODEL_OUTPUT_ROUTING_MISMATCH",
            "model output uav_id does not match the request",
        )
    present = value["target_present"]
    if not isinstance(present, bool):
        raise VisionSmokeError(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            "target_present must be a boolean",
        )
    description = value["description"]
    if not isinstance(description, str) or len(description) > 512:
        raise VisionSmokeError(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            "description must be a string of at most 512 characters",
        )
    confidence = value["self_reported_confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise VisionSmokeError(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            "self_reported_confidence must be a number",
        )
    normalized_confidence = float(confidence)
    if not isfinite(normalized_confidence) or not 0.0 <= normalized_confidence <= 1.0:
        raise VisionSmokeError(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            "self_reported_confidence must be between 0 and 1",
        )

    bbox = value["bbox_xyxy_normalized"]
    normalized_bbox: list[float] | None
    if bbox is None:
        normalized_bbox = None
    elif isinstance(bbox, list) and len(bbox) == 4:
        normalized_bbox = []
        for coordinate in bbox:
            if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
                raise VisionSmokeError(
                    "MODEL_OUTPUT_SCHEMA_INVALID",
                    "bbox coordinates must be numbers",
                )
            normalized = float(coordinate)
            if not isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise VisionSmokeError(
                    "MODEL_OUTPUT_SCHEMA_INVALID",
                    "bbox coordinates must be normalized to [0, 1]",
                )
            normalized_bbox.append(normalized)
        if not (
            normalized_bbox[0] < normalized_bbox[2]
            and normalized_bbox[1] < normalized_bbox[3]
        ):
            raise VisionSmokeError(
                "MODEL_OUTPUT_SCHEMA_INVALID",
                "bbox must satisfy x_min < x_max and y_min < y_max",
            )
    else:
        raise VisionSmokeError(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            "bbox_xyxy_normalized must be null or four numbers",
        )
    if present != (normalized_bbox is not None):
        raise VisionSmokeError(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            "target_present and bbox presence are inconsistent",
        )

    return {
        "schema_version": 1,
        "uav_id": expected_uav_id,
        "target_present": present,
        "bbox_xyxy_normalized": normalized_bbox,
        "description": description,
        "self_reported_confidence": normalized_confidence,
    }


def run_vision_smoke(
    *,
    image_path: Path,
    uav_id: str,
    target_description: str,
    client: ModelClient,
    max_side_px: int = DEFAULT_MAX_SIDE_PX,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict[str, object]:
    """Execute one strict vision request and return validated parsed JSON."""

    try:
        uav_id = validate_uav_id(uav_id)
    except (TypeError, ValueError):
        raise VisionSmokeError("INVALID_UAV_ID", "uav_id is invalid", exit_code=2)
    if not isinstance(target_description, str) or not target_description.strip():
        raise VisionSmokeError(
            "INVALID_TARGET_DESCRIPTION",
            "target description must be non-empty",
            exit_code=2,
        )
    if len(target_description) > 1000:
        raise VisionSmokeError(
            "INVALID_TARGET_DESCRIPTION",
            "target description must contain at most 1000 characters",
            exit_code=2,
        )
    if not callable(getattr(client, "chat", None)):
        raise TypeError("client must provide a callable chat method")

    try:
        normalized_image_path = Path(image_path)
    except TypeError:
        raise VisionSmokeError(
            "INVALID_IMAGE_PATH",
            "image path is invalid",
            exit_code=2,
        ) from None
    rgb = _load_rgb(normalized_image_path)
    try:
        data_url = encode_rgb_to_data_url(
            rgb,
            max_side_px=max_side_px,
            jpeg_quality=jpeg_quality,
        )
        image_part = ImageURLContentPart(data_url)
    except (TypeError, ValueError, RuntimeError):
        raise VisionSmokeError(
            "IMAGE_ENCODING_ERROR",
            "RGB image could not be prepared for the model",
            exit_code=2,
        ) from None

    schema = build_vision_smoke_schema(uav_id)
    system_prompt = (
        "Inspect exactly one provided image for the requested visual target. "
        "Return only the JSON object constrained by the response schema. "
        "The bbox is [x_min,y_min,x_max,y_max] normalized to [0,1]. "
        "Use null bbox when the target is absent. Do not output flight "
        "coordinates, velocity, acceleration, control commands, Markdown, or "
        "explanations. Echo the supplied uav_id exactly."
    )
    user_text = (
        f"uav_id: {uav_id}\n"
        f"target_description: {target_description.strip()}\n"
        "Determine whether this target is visible in the image."
    )
    try:
        response = client.chat(
            (
                ChatMessage("system", system_prompt),
                ChatMessage(
                    "user",
                    (TextContentPart(user_text), image_part),
                ),
            ),
            options=GenerationOptions(
                temperature=0.0,
                max_tokens=512,
                top_p=1.0,
                response_format=JsonSchemaResponseFormat(
                    VISION_SMOKE_SCHEMA_NAME,
                    schema,
                ),
            ),
        )
    except ModelConnectionError as exc:
        code = (
            "MODEL_TIMEOUT"
            if "timed out" in str(exc).lower()
            else "MODEL_CONNECTION_ERROR"
        )
        raise VisionSmokeError(code, "model request did not complete") from None
    except ModelHTTPError as exc:
        status = exc.status_code
        detail = "unknown" if status is None else str(status)
        raise VisionSmokeError(
            "MODEL_HTTP_ERROR",
            f"model service returned HTTP status {detail}",
        ) from None
    except ModelProtocolError:
        raise VisionSmokeError(
            "MODEL_PROTOCOL_ERROR",
            "model service returned an invalid protocol response",
        ) from None
    except Exception:
        # Do not surface an arbitrary exception string: a custom transport may
        # include request headers or the base64 payload in it.
        raise VisionSmokeError(
            "MODEL_REQUEST_FAILED",
            "model request failed unexpectedly",
        ) from None

    try:
        decoded = _parse_strict_json(response.content)
    except (json.JSONDecodeError, TypeError, _StrictJSONError):
        raise VisionSmokeError(
            "MODEL_OUTPUT_INVALID_JSON",
            "model output is not strict JSON",
        ) from None
    return _validate_output(decoded, expected_uav_id=uav_id)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _jpeg_quality(value: str) -> int:
    result = _positive_int(value)
    if result > 95:
        raise argparse.ArgumentTypeError("must be at most 95")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one strict, image-only Qwen visual smoke check."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--uav-id", default="uav_1")
    parser.add_argument("--target-description", required=True)
    parser.add_argument("--base-url", default=os.environ.get("QWEN_API_BASE"))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL"))
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"))
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("QWEN_REQUEST_TIMEOUT_S", "60")),
    )
    parser.add_argument("--max-side-px", type=_positive_int, default=1024)
    parser.add_argument("--jpeg-quality", type=_jpeg_quality, default=80)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        client = OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout_s=args.timeout,
        )
        result = run_vision_smoke(
            image_path=args.image,
            uav_id=args.uav_id,
            target_description=args.target_description,
            client=client,
            max_side_px=args.max_side_px,
            jpeg_quality=args.jpeg_quality,
        )
    except VisionSmokeError as exc:
        print(
            json.dumps(
                {"ok": False, "error_code": exc.code, "message": str(exc)},
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return exc.exit_code
    except Exception:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_code": "CONFIGURATION_ERROR",
                    "message": "vision smoke configuration is invalid",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2

    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
