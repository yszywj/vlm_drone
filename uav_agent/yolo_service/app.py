"""FastAPI application factory for the loopback-only YOLO service."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import numpy as np

from yolo_service.config import YoloServiceConfig
from yolo_service.engine import (
    ImageValidationError,
    StreamBusyError,
    StreamConflictError,
    StreamSequenceError,
    UltralyticsEngine,
    UnsupportedTargetQuery,
    YoloDependencyUnavailable,
    YoloEngineError,
)
from yolo_service.protocol import (
    ProtocolValidationError,
    ResetStreamRequest,
    SCHEMA_VERSION,
    TrackRequest,
    loads_strict_object,
)


class YoloWebDependencyUnavailable(RuntimeError):
    """Raised when service-only HTTP dependencies are not installed."""


class BoundedRequestBodyMiddleware:
    """Enforce a hard POST-body ceiling at the ASGI receive boundary.

    The header check only rejects oversized requests early.  Every body chunk
    is still counted and retained only within the configured ceiling before
    being replayed downstream.  Thus chunked requests and requests without
    ``Content-Length`` cannot make Starlette's multipart parser buffer an
    unbounded upload or create a partial oversized temporary file.
    """

    def __init__(self, asgi_app: Any, *, max_post_bytes: int) -> None:
        if not isinstance(max_post_bytes, int) or isinstance(max_post_bytes, bool):
            raise TypeError("max_post_bytes must be an integer")
        if max_post_bytes <= 0:
            raise ValueError("max_post_bytes must be positive")
        self._app = asgi_app
        self._max_post_bytes = max_post_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self._app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                await _send_asgi_error(
                    scope,
                    receive,
                    send,
                    code="INVALID_CONTENT_LENGTH",
                    message="Content-Length must be an integer",
                    status_code=400,
                )
                return
            if length < 0:
                await _send_asgi_error(
                    scope,
                    receive,
                    send,
                    code="INVALID_CONTENT_LENGTH",
                    message="Content-Length must be non-negative",
                    status_code=400,
                )
                return
            if length > self._max_post_bytes:
                await _send_asgi_error(
                    scope,
                    receive,
                    send,
                    code="IMAGE_TOO_LARGE",
                    message="request exceeds configured byte limit",
                    status_code=413,
                )
                return

        received = 0
        buffered_messages: list[Any] = []
        while True:
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_post_bytes:
                    await _send_asgi_error(
                        scope,
                        receive,
                        send,
                        code="IMAGE_TOO_LARGE",
                        message="request exceeds configured byte limit",
                        status_code=413,
                    )
                    return
                buffered_messages.append(message)
                if not message.get("more_body", False):
                    break
            elif message.get("type") == "http.disconnect":
                buffered_messages.append(message)
                break

        replay_index = 0

        async def replay_receive() -> Any:
            nonlocal replay_index
            if replay_index < len(buffered_messages):
                message = buffered_messages[replay_index]
                replay_index += 1
                return message
            return await receive()

        await self._app(scope, replay_receive, send)


async def _send_asgi_error(
    scope: Any,
    receive: Any,
    send: Any,
    *,
    code: str,
    message: str,
    status_code: int,
) -> None:
    body = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "error": {"code": code, "message": message},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": (
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ),
        }
    )
    await send({"type": "http.response.body", "body": body})


def decode_jpeg_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG directly to BGR exactly once using OpenCV.

    ``cv2.imdecode(..., IMREAD_COLOR)`` already produces the BGR array expected
    by Ultralytics.  The caller must not apply an additional channel swap.
    """

    if not isinstance(jpeg_bytes, bytes) or not jpeg_bytes:
        raise ImageValidationError("image part must contain non-empty JPEG bytes")
    try:
        import cv2
    except ImportError as exc:
        raise YoloWebDependencyUnavailable(
            "opencv-python-headless is required by the YOLO service"
        ) from exc
    encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("image part is not a decodable JPEG")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError("decoded JPEG must be uint8 HxWx3")
    return np.ascontiguousarray(image)


def create_app(
    config: YoloServiceConfig | None = None,
    *,
    engine: UltralyticsEngine | None = None,
) -> Any:
    """Build the ASGI app without importing FastAPI at module import time."""

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:  # pragma: no cover - deployment dependency check
        raise YoloWebDependencyUnavailable(
            "fastapi and starlette are required in the yolo_perception environment"
        ) from exc

    if engine is None:
        if config is None:
            raise ValueError("config is required when engine is not injected")
        engine = UltralyticsEngine(config)
    elif config is not None:
        raise ValueError("config and engine are mutually exclusive")
    if not isinstance(engine, UltralyticsEngine):
        raise TypeError("engine must be an UltralyticsEngine")
    settings = engine.settings

    app = FastAPI(
        title="UAV Agent local YOLO tracking service",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.yolo_engine = engine

    def error_response(code: str, message: str, status_code: int) -> Any:
        return JSONResponse(
            status_code=status_code,
            content={
                "schema_version": SCHEMA_VERSION,
                "error": {"code": code, "message": message},
            },
        )

    # Multipart framing and strict metadata receive a bounded 1 MiB allowance
    # in addition to the configured JPEG byte ceiling.
    app.add_middleware(
        BoundedRequestBodyMiddleware,
        max_post_bytes=settings.max_image_bytes + 1_048_576,
    )

    async def health_endpoint() -> Any:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "ready": True,
            "active_stream_id": engine.active_stream_id,
        }

    app.get("/health")(health_endpoint)

    async def model_info_endpoint() -> Any:
        return engine.model_info()

    app.get("/v1/model-info")(model_info_endpoint)

    async def reset_endpoint(request: Any) -> Any:
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type != "application/json":
            return error_response(
                "UNSUPPORTED_MEDIA_TYPE",
                "stream reset requires application/json",
                415,
            )
        try:
            raw = await request.body()
            if len(raw) > 65_536:
                return error_response("PAYLOAD_TOO_LARGE", "reset payload too large", 413)
            reset_request = ResetStreamRequest.from_dict(
                loads_strict_object(raw, "stream reset request")
            )
            await run_in_threadpool(engine.reset_stream, reset_request.stream_id)
        except ProtocolValidationError as exc:
            return error_response("INVALID_REQUEST", str(exc), 422)
        except (StreamConflictError, StreamBusyError) as exc:
            return error_response(exc.code, str(exc), 409)
        except YoloEngineError as exc:
            return error_response(exc.code, str(exc), 503)
        return {
            **reset_request.to_dict(),
            "reset": True,
        }

    reset_endpoint.__annotations__["request"] = Request
    app.post("/v1/streams/reset")(reset_endpoint)

    async def track_endpoint(request: Any) -> Any:
        content_type = request.headers.get("content-type", "").lower()
        if not content_type.startswith("multipart/form-data"):
            return error_response(
                "UNSUPPORTED_MEDIA_TYPE",
                "track requires multipart/form-data",
                415,
            )
        try:
            form = await request.form()
            items = list(form.multi_items())
            keys = [key for key, _ in items]
            if sorted(keys) != ["image", "request_json"]:
                raise ProtocolValidationError(
                    "multipart form must contain exactly request_json and image once"
                )
            request_json = form["request_json"]
            image_part = form["image"]
            if not isinstance(request_json, str):
                raise ProtocolValidationError("request_json must be a text form field")
            if len(request_json.encode("utf-8")) > 65_536:
                raise ProtocolValidationError("request_json exceeds 65536 bytes")
            track_request = TrackRequest.from_json(request_json)
            if not hasattr(image_part, "read"):
                raise ProtocolValidationError("image must be a multipart file")
            part_content_type = getattr(image_part, "content_type", None)
            if part_content_type not in {"image/jpeg", "image/jpg"}:
                return error_response(
                    "UNSUPPORTED_IMAGE_TYPE",
                    "image part must have image/jpeg content type",
                    415,
                )
            jpeg_bytes = await image_part.read(settings.max_image_bytes + 1)
            if len(jpeg_bytes) > settings.max_image_bytes:
                return error_response(
                    "IMAGE_TOO_LARGE",
                    "JPEG exceeds configured byte limit",
                    413,
                )
            decode_started = perf_counter()
            image_bgr = await run_in_threadpool(decode_jpeg_to_bgr, jpeg_bytes)
            decode_ms = max(0.0, (perf_counter() - decode_started) * 1000.0)
            if image_bgr.shape[1] > settings.max_image_width_px:
                raise ImageValidationError("image width exceeds configured maximum")
            if image_bgr.shape[0] > settings.max_image_height_px:
                raise ImageValidationError("image height exceeds configured maximum")
            response = await run_in_threadpool(
                engine.track,
                track_request,
                image_bgr,
                decode_ms=decode_ms,
            )
            response.assert_matches(track_request)
            return response.to_dict()
        except ProtocolValidationError as exc:
            return error_response("INVALID_REQUEST", str(exc), 422)
        except ImageValidationError as exc:
            return error_response(exc.code, str(exc), 422)
        except UnsupportedTargetQuery as exc:
            return error_response(exc.code, str(exc), 422)
        except (StreamConflictError, StreamBusyError, StreamSequenceError) as exc:
            return error_response(exc.code, str(exc), 409)
        except (YoloDependencyUnavailable, YoloWebDependencyUnavailable) as exc:
            return error_response("YOLO_DEPENDENCY_UNAVAILABLE", str(exc), 503)
        except YoloEngineError as exc:
            return error_response(exc.code, str(exc), 503)

    track_endpoint.__annotations__["request"] = Request
    app.post("/v1/track")(track_endpoint)
    return app


__all__ = [
    "BoundedRequestBodyMiddleware",
    "YoloWebDependencyUnavailable",
    "create_app",
    "decode_jpeg_to_bgr",
]
