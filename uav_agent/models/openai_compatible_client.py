"""Standard-library client for an OpenAI-compatible model service."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from math import isfinite
from numbers import Real
import os
import socket
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from models.base import (
    ChatMessage,
    GenerationOptions,
    ModelConnectionError,
    ModelHTTPError,
    ModelProtocolError,
    ModelResponse,
)


DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "Qwen3-VL-4B-Instruct"
DEFAULT_REQUEST_TIMEOUT_S = 60.0

Transport = Callable[..., object]


class OpenAICompatibleClient:
    """Small synchronous client for vLLM's OpenAI-compatible endpoints.

    ``transport`` is an optional ``urllib.request.urlopen``-compatible callable.
    It keeps unit tests entirely offline and also permits a future application
    to supply a custom transport without changing this client contract.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        max_retries: int = 2,
        *,
        transport: Transport | None = None,
    ) -> None:
        selected_base = (
            base_url
            if base_url is not None
            else os.environ.get("QWEN_API_BASE", DEFAULT_API_BASE)
        )
        selected_model = (
            model
            if model is not None
            else os.environ.get("QWEN_MODEL", DEFAULT_MODEL)
        )
        selected_key = (
            api_key
            if api_key is not None
            else os.environ.get("QWEN_API_KEY", DEFAULT_API_KEY)
        )
        selected_timeout: object = (
            timeout_s
            if timeout_s is not None
            else os.environ.get(
                "QWEN_REQUEST_TIMEOUT_S",
                str(DEFAULT_REQUEST_TIMEOUT_S),
            )
        )

        self.base_url = self._normalize_base_url(selected_base)
        self.model = self._validate_nonempty_string(selected_model, "model")
        if not isinstance(selected_key, str):
            raise TypeError("api_key must be a string or None")
        if any(not 0x20 <= ord(character) <= 0x7E for character in selected_key):
            # urllib eventually encodes headers as Latin-1. Restricting keys to
            # printable ASCII prevents both header injection and a raw encoding
            # exception whose diagnostic could echo credential material.
            raise ValueError("api_key must contain printable ASCII only")
        self._api_key = selected_key
        self.timeout_s = self._validate_timeout(selected_timeout)

        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries

        if transport is not None and not callable(transport):
            raise TypeError("transport must be callable")
        self._transport = transport

    @staticmethod
    def _validate_nonempty_string(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must be non-empty")
        return normalized

    @classmethod
    def _normalize_base_url(cls, value: object) -> str:
        raw = cls._validate_nonempty_string(value, "base_url")
        parts = urllib_parse.urlsplit(raw)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parts.username is not None or parts.password is not None:
            raise ValueError("base_url must not contain user credentials")
        if parts.query or parts.fragment:
            raise ValueError("base_url must not contain a query or fragment")

        path_segments = [segment for segment in parts.path.split("/") if segment]
        while path_segments and path_segments[-1].lower() == "v1":
            path_segments.pop()
        path_segments.append("v1")
        path = "/" + "/".join(path_segments)
        return urllib_parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    @staticmethod
    def _validate_timeout(value: object) -> float:
        if isinstance(value, bool):
            raise TypeError("timeout_s must be a finite number")
        if isinstance(value, str):
            try:
                timeout = float(value)
            except ValueError:
                raise ValueError(
                    "QWEN_REQUEST_TIMEOUT_S must be a finite number"
                ) from None
        elif isinstance(value, Real):
            timeout = float(value)
        else:
            raise TypeError("timeout_s must be a finite number")
        if not isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and greater than zero")
        return timeout

    def healthcheck(self) -> None:
        """Verify that the service responds with a JSON object at ``/models``."""

        payload = self._request_json("GET", "/models")
        if not isinstance(payload, Mapping):
            raise ModelProtocolError(
                "model healthcheck response must be a JSON object"
            )
        if not isinstance(payload.get("data"), list):
            raise ModelProtocolError(
                "model healthcheck response is missing a data list"
            )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        """Send a text-only OpenAI-compatible chat completion request."""

        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise TypeError("messages must be a sequence of ChatMessage objects")
        if not messages:
            raise ValueError("messages must not be empty")
        serialized_messages: list[dict[str, object]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, ChatMessage):
                raise TypeError(f"messages[{index}] must be a ChatMessage")
            serialized_messages.append(message.to_dict())

        selected_options = options if options is not None else GenerationOptions()
        if not isinstance(selected_options, GenerationOptions):
            raise TypeError("options must be GenerationOptions or None")

        request_payload = {
            "model": self.model,
            "messages": serialized_messages,
            "temperature": selected_options.temperature,
            "max_tokens": selected_options.max_tokens,
            "top_p": selected_options.top_p,
        }
        response_payload = self._request_json(
            "POST",
            "/chat/completions",
            request_payload,
        )
        return self._parse_chat_response(response_payload)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> object:
        url = f"{self.base_url}{path}"
        body: bytes | None = None
        if payload is not None:
            try:
                body = json.dumps(
                    payload,
                    # Escaping non-ASCII also makes lone surrogate code points
                    # safe to encode.  They can occur in otherwise valid Python
                    # strings and must not escape as UnicodeEncodeError.
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError):
                serialization_error = ModelProtocolError(
                    "model request payload is not JSON serializable"
                )
            else:
                serialization_error = None
            if serialization_error is not None:
                # Raise outside the handler so no implementation exception is
                # retained in the public exception's implicit context.
                raise serialization_error

        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt_index in range(self.max_retries + 1):
            request = urllib_request.Request(
                url=url,
                data=body,
                headers=headers,
                method=method,
            )
            request_error: ModelConnectionError | ModelHTTPError | None = None
            should_retry = False
            try:
                raw, status = self._send(request)
            except urllib_error.HTTPError as exc:
                status = int(exc.code)
                self._close_quietly(exc)
                if self._can_retry(status, attempt_index):
                    should_retry = True
                else:
                    request_error = self._http_error(status, attempt_index + 1)
            except (socket.timeout, TimeoutError):
                request_error = ModelConnectionError(
                    f"model request timed out after {self.timeout_s:g} seconds"
                )
            except urllib_error.URLError as exc:
                if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                    message = (
                        f"model request timed out after {self.timeout_s:g} seconds"
                    )
                else:
                    message = "could not connect to model service"
                request_error = ModelConnectionError(message)
            except (ConnectionError, OSError):
                request_error = ModelConnectionError(
                    "could not connect to model service"
                )

            # Do not raise while a secret-bearing urllib exception is active:
            # ``raise ... from None`` suppresses display but still retains the
            # original object in ``__context__``.
            if should_retry:
                continue
            if request_error is not None:
                raise request_error

            if not 200 <= status < 300:
                if self._can_retry(status, attempt_index):
                    continue
                raise self._http_error(status, attempt_index + 1) from None

            return self._decode_json(raw)

        # The bounded loop always returns or raises.  This protects the method
        # against future changes to retry logic while preserving error typing.
        raise ModelHTTPError("model request exhausted its retry budget")

    def _send(self, request: urllib_request.Request) -> tuple[bytes | str, int]:
        opener = self._transport or urllib_request.urlopen
        response: Any = opener(request, timeout=self.timeout_s)
        try:
            status_value = (
                response.getcode()
                if callable(getattr(response, "getcode", None))
                else getattr(response, "status", None)
            )
            status = 200 if status_value is None else int(status_value)
            read = getattr(response, "read", None)
            if not callable(read):
                raise ModelProtocolError(
                    "model service returned an invalid HTTP response"
                )
            raw = read()
            if not isinstance(raw, (bytes, bytearray, str)):
                raise ModelProtocolError(
                    "model service returned a non-text HTTP body"
                )
            return (bytes(raw) if isinstance(raw, bytearray) else raw), status
        finally:
            self._close_quietly(response)

    @staticmethod
    def _close_quietly(response: object) -> None:
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Closing a response must never replace the useful request error.
                pass

    def _can_retry(self, status: int, attempt_index: int) -> bool:
        retryable = status == 429 or 500 <= status <= 599
        return retryable and attempt_index < self.max_retries

    @staticmethod
    def _http_error(status: int, attempts: int) -> ModelHTTPError:
        return ModelHTTPError(
            f"model service returned HTTP status {status}",
            status_code=status,
            attempts=attempts,
        )

    @staticmethod
    def _decode_json(raw: bytes | str) -> object:
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            protocol_error = ModelProtocolError(
                "model service returned invalid JSON"
            )
        else:
            protocol_error = None
        if protocol_error is not None:
            raise protocol_error
        return payload

    @staticmethod
    def _parse_chat_response(payload: object) -> ModelResponse:
        if not isinstance(payload, Mapping):
            raise ModelProtocolError("chat completion response must be a JSON object")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelProtocolError(
                "chat completion response is missing a non-empty choices list"
            )
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise ModelProtocolError("chat completion choice must be a JSON object")

        message = first_choice.get("message")
        if not isinstance(message, Mapping) or "content" not in message:
            raise ModelProtocolError(
                "chat completion response is missing choices[0].message.content"
            )
        content = message["content"]
        if not isinstance(content, str):
            raise ModelProtocolError(
                "chat completion choices[0].message.content must be a string"
            )

        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ModelProtocolError("chat completion model must be a string or null")
        finish_reason = first_choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ModelProtocolError(
                "chat completion finish_reason must be a string or null"
            )

        usage_value = OpenAICompatibleClient._normalize_usage(payload.get("usage", {}))
        try:
            response = ModelResponse(
                content=content,
                model=model,
                finish_reason=finish_reason,
                usage=usage_value,
            )
        except (TypeError, ValueError) as exc:
            protocol_error = ModelProtocolError(
                f"chat completion response has invalid usage: {exc}"
            )
        else:
            protocol_error = None
        if protocol_error is not None:
            raise protocol_error
        return response

    @staticmethod
    def _normalize_usage(value: object) -> dict[str, int]:
        """Keep integer counters while tolerating optional vLLM detail fields."""

        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ModelProtocolError(
                "chat completion usage must be a JSON object or null"
            )

        normalized: dict[str, int] = {}
        for key, field_value in value.items():
            if not isinstance(key, str):
                raise ModelProtocolError("chat completion usage keys must be strings")
            if isinstance(field_value, bool):
                raise ModelProtocolError(
                    "chat completion usage integer fields must not be boolean"
                )
            if isinstance(field_value, int):
                if field_value < 0:
                    raise ModelProtocolError(
                        "chat completion usage integer fields must be non-negative"
                    )
                normalized[key] = field_value
                continue
            if field_value is None or isinstance(field_value, Mapping):
                # OpenAI-compatible servers may attach token-detail objects or
                # null placeholders.  ModelResponse intentionally exposes only
                # the stable, top-level integer counters.
                continue
            raise ModelProtocolError(
                "chat completion usage fields must be integers, detail objects, "
                "or null"
            )
        return normalized
