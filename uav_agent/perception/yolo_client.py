"""Strict loopback client for the isolated YOLO tracking service.

The client is synchronous by design; :mod:`target_perception_coordinator`
runs it on one bounded worker so no HTTP call occurs in ``Skill.tick()``.
Only RGB JPEG bytes and routing metadata cross this boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import BytesIO
import json
from math import isfinite
from numbers import Real
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

from common.loopback_url import validate_loopback_http_url
from yolo_service.protocol import (
    ResetStreamRequest,
    TrackRequest,
    TrackResponse,
    loads_strict_object,
)


class YoloClientError(RuntimeError):
    """Base error for bounded service failures."""


class YoloClientUnavailable(YoloClientError):
    """Network, timeout, or service availability failure."""


class YoloClientRequestTimeout(YoloClientUnavailable):
    """The client deadline elapsed while the remote request may still run."""


class YoloClientResponseError(YoloClientError):
    """Malformed, mismatched, or explicitly rejected response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class YoloClientStreamBusy(YoloClientUnavailable):
    """The worker is finishing an earlier inference for the same process.

    This is service backpressure, not a malformed detector response.  In
    particular, a local HTTP timeout does not cancel ``model.track()`` in the
    worker process, so the next request can legitimately observe
    ``STREAM_BUSY`` until that orphaned request finishes.
    """

    status_code = 409
    error_code = "STREAM_BUSY"


@dataclass(frozen=True, slots=True)
class YoloModelInfo:
    model_family: str
    class_names: tuple[tuple[int, str], ...]
    model_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.model_family not in {"yolo", "yoloe"}:
            raise ValueError("model_family must be yolo or yoloe")
        names = tuple(self.class_names)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(name, str)
            or not name.strip()
            for index, name in names
        ):
            raise ValueError("class_names must contain non-negative IDs and names")
        if len({index for index, _ in names}) != len(names):
            raise ValueError("class_names contains duplicate IDs")
        object.__setattr__(self, "class_names", names)
        if self.model_sha256 is not None:
            if (
                not isinstance(self.model_sha256, str)
                or len(self.model_sha256) != 64
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in self.model_sha256
                )
            ):
                raise ValueError(
                    "model_sha256 must be a 64-character hexadecimal digest"
                )
            object.__setattr__(self, "model_sha256", self.model_sha256.lower())

    @property
    def names(self) -> dict[int, str]:
        return dict(self.class_names)


def validate_yolo_model_identity(
    info: YoloModelInfo,
    *,
    expected_model_family: str,
    expected_model_names: Mapping[int, str],
    expected_model_sha256: str | None,
    worker_url: str,
) -> None:
    """Fail closed when a worker does not expose the configured model.

    The diagnostic is deliberately complete and scalar-only so a launch log
    always identifies the contacted worker, both digests, and the actual class
    catalog without disclosing any frame or target data.
    """

    if not isinstance(info, YoloModelInfo):
        raise TypeError("info must be a YoloModelInfo")
    normalized_url = validate_loopback_http_url(worker_url, "worker_url")
    if expected_model_family not in {"yolo", "yoloe"}:
        raise ValueError("expected_model_family must be yolo or yoloe")
    normalized_expected: dict[int, str] = {}
    for raw_id, raw_name in expected_model_names.items():
        if (
            isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or raw_id < 0
            or not isinstance(raw_name, str)
            or not raw_name.strip()
        ):
            raise ValueError(
                "expected_model_names must map non-negative IDs to names"
            )
        normalized_expected[int(raw_id)] = raw_name.strip().casefold()
    expected_digest = None
    if expected_model_sha256 is not None:
        if (
            not isinstance(expected_model_sha256, str)
            or len(expected_model_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in expected_model_sha256
            )
        ):
            raise ValueError(
                "expected_model_sha256 must be a 64-character hexadecimal digest"
            )
        expected_digest = expected_model_sha256.lower()
    actual_names = dict(info.names)
    normalized_actual = {
        class_id: class_name.strip().casefold()
        for class_id, class_name in actual_names.items()
    }
    mismatches: list[str] = []
    if info.model_family != expected_model_family:
        mismatches.append("model_family")
    if normalized_expected and normalized_actual != normalized_expected:
        mismatches.append("model_names")
    if expected_digest is not None and info.model_sha256 != expected_digest:
        mismatches.append("model_sha256")
    if not mismatches:
        return
    compatibility_prefix = (
        "YOLO model family mismatch; "
        if "model_family" in mismatches
        else ""
    )
    if "model_names" in mismatches:
        compatibility_prefix += "worker must expose exactly class 0='cube'; "
    raise YoloClientResponseError(
        compatibility_prefix
        + "YOLO model identity mismatch: "
        f"fields={','.join(mismatches)}; worker_url={normalized_url}; "
        f"expected_model_family={expected_model_family!r}; "
        f"actual_model_family={info.model_family!r}; "
        f"expected_model_sha256={expected_digest!r}; "
        f"actual_model_sha256={info.model_sha256!r}; "
        f"expected_model_names={dict(sorted(normalized_expected.items()))!r}; "
        f"actual_model_names={dict(sorted(actual_names.items()))!r}"
    )


Transport = Callable[[str, str, bytes | None, Mapping[str, str], float], bytes]


# The service authority is already restricted to an explicit loopback address.
# Do not let ambient HTTP(S)_PROXY settings redirect detector traffic away from
# that local trust boundary or make an otherwise healthy worker look offline.
_DIRECT_LOOPBACK_OPENER = build_opener(ProxyHandler({}))


class YoloServiceClient:
    """Small strict HTTP adapter with an injectable transport for tests."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8011",
        request_timeout_s: float = 0.5,
        jpeg_quality: int = 90,
        transport: Transport | None = None,
    ) -> None:
        normalized_url = validate_loopback_http_url(base_url, "base_url")
        if (
            isinstance(request_timeout_s, bool)
            or not isinstance(request_timeout_s, Real)
            or not isfinite(float(request_timeout_s))
            or float(request_timeout_s) <= 0.0
        ):
            raise ValueError("request_timeout_s must be finite and positive")
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int):
            raise TypeError("jpeg_quality must be an integer")
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be within [1, 95]")
        if transport is not None and not callable(transport):
            raise TypeError("transport must be callable or None")
        self._base_url = normalized_url
        self._timeout_s = float(request_timeout_s)
        self._jpeg_quality = jpeg_quality
        self._transport = transport or _urllib_transport

    @property
    def base_url(self) -> str:
        return self._base_url

    def health(self) -> Mapping[str, object]:
        payload = self._request_json("GET", "/health", None, {})
        if (
            payload.get("schema_version") != 1
            or payload.get("status") != "ok"
            or payload.get("ready") is not True
        ):
            raise YoloClientResponseError(
                f"YOLO service health response from {self._base_url} is invalid"
            )
        return payload

    def model_info(self) -> YoloModelInfo:
        payload = self._request_json("GET", "/v1/model-info", None, {})
        if payload.get("schema_version") != 1:
            raise YoloClientResponseError("model-info schema_version must be 1")
        family = payload.get("model_family")
        raw_names = payload.get(
            "model_names",
            payload.get("class_names", payload.get("names")),
        )
        if isinstance(raw_names, list):
            names = tuple((index, value) for index, value in enumerate(raw_names))
        elif isinstance(raw_names, Mapping):
            try:
                names = tuple(
                    sorted((int(index), value) for index, value in raw_names.items())
                )
            except (TypeError, ValueError) as exc:
                raise YoloClientResponseError("invalid model-info class names") from exc
        else:
            raise YoloClientResponseError("model-info has no class names")
        try:
            return YoloModelInfo(
                str(family),
                names,  # type: ignore[arg-type]
                payload.get("model_sha256"),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise YoloClientResponseError(f"invalid model-info response: {exc}") from exc

    def reset_stream(self, request: ResetStreamRequest) -> None:
        if not isinstance(request, ResetStreamRequest):
            raise TypeError("request must be a ResetStreamRequest")
        body = _strict_json_bytes(request.to_dict())
        payload = self._request_json(
            "POST",
            "/v1/streams/reset",
            body,
            {"Content-Type": "application/json"},
        )
        for name in ("schema_version", "request_id", "mission_id", "uav_id", "stream_id"):
            if payload.get(name) != getattr(request, name):
                raise YoloClientResponseError(
                    f"stream reset response mismatches {name}"
                )
        if payload.get("reset") is not True:
            raise YoloClientResponseError("stream reset was not acknowledged")

    def track(self, request: TrackRequest, image_rgb: np.ndarray) -> TrackResponse:
        if not isinstance(request, TrackRequest):
            raise TypeError("request must be a TrackRequest")
        jpeg = encode_rgb_jpeg(image_rgb, jpeg_quality=self._jpeg_quality)
        boundary = f"uav-agent-{request.request_id}"
        body = _multipart_body(request, jpeg, boundary)
        raw = self._request(
            "POST",
            "/v1/track",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            response = TrackResponse.from_json(raw)
            response.assert_matches(request)
        except (TypeError, ValueError) as exc:
            raise YoloClientResponseError(f"invalid track response: {exc}") from exc
        return response

    def _request_json(
        self,
        method: str,
        path: str,
        body: bytes | None,
        headers: Mapping[str, str],
    ) -> Mapping[str, object]:
        raw = self._request(method, path, body, headers)
        try:
            return loads_strict_object(raw, path)
        except ValueError as exc:
            raise YoloClientResponseError(f"invalid JSON response from {path}: {exc}") from exc

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None,
        headers: Mapping[str, str],
    ) -> bytes:
        try:
            return self._transport(
                method,
                self._base_url + path,
                body,
                dict(headers),
                self._timeout_s,
            )
        except YoloClientError:
            raise
        except TimeoutError as exc:
            raise YoloClientRequestTimeout(
                "YOLO service request timed out; remote completion is unknown"
            ) from exc
        except OSError as exc:
            raise YoloClientUnavailable(
                f"YOLO service request failed: {type(exc).__name__}"
            ) from exc


def encode_rgb_jpeg(image_rgb: np.ndarray, *, jpeg_quality: int = 90) -> bytes:
    """Encode one RGB array without applying a client-side channel swap."""

    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a numpy.ndarray")
    if image_rgb.dtype != np.uint8:
        raise TypeError("image_rgb must have dtype uint8")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if image_rgb.shape[0] <= 0 or image_rgb.shape[1] <= 0:
        raise ValueError("image_rgb dimensions must be positive")
    if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int):
        raise TypeError("jpeg_quality must be an integer")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be within [1, 95]")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise YoloClientUnavailable("Pillow is required for JPEG encoding") from exc
    stream = BytesIO()
    Image.fromarray(np.ascontiguousarray(image_rgb)).save(
        stream,
        format="JPEG",
        quality=jpeg_quality,
    )
    return stream.getvalue()


def _multipart_body(request: TrackRequest, jpeg: bytes, boundary: str) -> bytes:
    if not boundary.isascii() or any(character in boundary for character in '\r\n"'):
        raise ValueError("invalid multipart boundary")
    marker = boundary.encode("ascii")
    metadata = _strict_json_bytes(request.to_dict())
    return b"".join(
        (
            b"--" + marker + b"\r\n",
            b'Content-Disposition: form-data; name="request_json"\r\n',
            b"Content-Type: application/json\r\n\r\n",
            metadata,
            b"\r\n--" + marker + b"\r\n",
            b'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            jpeg,
            b"\r\n--" + marker + b"--\r\n",
        )
    )


def _strict_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _urllib_transport(
    method: str,
    url: str,
    body: bytes | None,
    headers: Mapping[str, str],
    timeout_s: float,
) -> bytes:
    request = Request(url, data=body, headers=dict(headers), method=method)
    try:
        with _DIRECT_LOOPBACK_OPENER.open(
            request, timeout=timeout_s
        ) as response:  # noqa: S310 - loopback checked and proxies disabled
            return response.read()
    except HTTPError as exc:
        try:
            payload = loads_strict_object(exc.read(), "YOLO error response")
            error = payload.get("error")
            if isinstance(error, Mapping):
                code = str(error.get("code", "HTTP_ERROR"))
                message = str(error.get("message", "request rejected"))
            else:
                code, message = "HTTP_ERROR", "request rejected"
        except Exception:
            code, message = "HTTP_ERROR", "request rejected"
        detail = (
            f"YOLO service rejected request ({exc.code}, {code}): {message[:256]}"
        )
        if exc.code == 409 and code == "STREAM_BUSY":
            raise YoloClientStreamBusy(detail) from exc
        raise YoloClientResponseError(
            detail,
            status_code=exc.code,
            error_code=code,
        ) from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise YoloClientRequestTimeout(
                "YOLO service request timed out; remote completion is unknown"
            ) from exc
        raise YoloClientUnavailable(
            f"YOLO service is unavailable: {type(exc.reason).__name__}"
        ) from exc


__all__ = [
    "YoloClientError",
    "YoloClientRequestTimeout",
    "YoloClientResponseError",
    "YoloClientStreamBusy",
    "YoloClientUnavailable",
    "YoloModelInfo",
    "YoloServiceClient",
    "encode_rgb_jpeg",
    "validate_yolo_model_identity",
]
