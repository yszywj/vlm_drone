from __future__ import annotations

import json
import math
import os
import socket
import unittest
from unittest.mock import patch
from urllib import error as urllib_error

from models.base import (
    ChatMessage,
    GenerationOptions,
    ModelConnectionError,
    ModelHTTPError,
    ModelProtocolError,
)
from models.openai_compatible_client import OpenAICompatibleClient


class FakeResponse:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        if isinstance(payload, bytes):
            self._body = payload
        elif isinstance(payload, str):
            self._body = payload.encode("utf-8")
        else:
            self._body = json.dumps(payload).encode("utf-8")
        self.status = status
        self.closed = False

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True


class QueueTransport:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request: object, *, timeout: float) -> object:
        self.calls.append((request, timeout))
        if not self.outcomes:
            raise AssertionError("transport called more often than expected")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def http_error(status: int) -> urllib_error.HTTPError:
    return urllib_error.HTTPError(
        url="http://127.0.0.1:8000/v1/models",
        code=status,
        msg="test status",
        hdrs=None,
        fp=None,
    )


def chat_payload(content: object = "{\"status\":\"ok\"}") -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "model": "Qwen3-VL-4B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 5,
            "total_tokens": 9,
        },
    }


class ModelSchemaTest(unittest.TestCase):
    def test_chat_message_is_text_only_and_json_compatible(self) -> None:
        message = ChatMessage("user", "只返回 JSON")
        self.assertEqual(
            message.to_dict(),
            {"role": "user", "content": "只返回 JSON"},
        )
        with self.assertRaises(ValueError):
            ChatMessage("tool", "hello")
        with self.assertRaises(TypeError):
            ChatMessage("user", [{"type": "text", "text": "future"}])

    def test_generation_options_validate_numeric_values_strictly(self) -> None:
        options = GenerationOptions(temperature=1, max_tokens=8, top_p=0.5)
        self.assertEqual(options.temperature, 1.0)
        self.assertEqual(options.top_p, 0.5)

        invalid_options = (
            {"temperature": True},
            {"temperature": math.nan},
            {"temperature": -0.1},
            {"temperature": 2.1},
            {"max_tokens": True},
            {"max_tokens": 1.5},
            {"max_tokens": 0},
            {"top_p": math.inf},
            {"top_p": 0},
            {"top_p": 1.1},
        )
        for values in invalid_options:
            with self.subTest(values=values), self.assertRaises(
                (TypeError, ValueError)
            ):
                GenerationOptions(**values)


class OpenAICompatibleClientTest(unittest.TestCase):
    def make_client(
        self,
        transport: QueueTransport,
        **overrides: object,
    ) -> OpenAICompatibleClient:
        values = {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "Qwen3-VL-4B-Instruct",
            "api_key": "test-key",
            "timeout_s": 3,
            "transport": transport,
        }
        values.update(overrides)
        return OpenAICompatibleClient(**values)

    def test_healthcheck_uses_models_endpoint(self) -> None:
        response = FakeResponse({"object": "list", "data": []})
        transport = QueueTransport(response)
        client = self.make_client(transport)

        self.assertIsNone(client.healthcheck())
        request, timeout = transport.calls[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/models")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(timeout, 3.0)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertTrue(response.closed)

    def test_healthcheck_requires_models_data_list(self) -> None:
        for payload in ({}, {"data": None}, {"data": {}}):
            with self.subTest(payload=payload):
                transport = QueueTransport(FakeResponse(payload))
                with self.assertRaisesRegex(ModelProtocolError, "data list"):
                    self.make_client(transport).healthcheck()

    def test_chat_completion_serializes_request_and_normalizes_response(self) -> None:
        transport = QueueTransport(FakeResponse(chat_payload()))
        client = self.make_client(transport)

        response = client.chat(
            [ChatMessage("system", "Be brief"), ChatMessage("user", "Ping")],
            options=GenerationOptions(temperature=0.2, max_tokens=16, top_p=0.8),
        )

        self.assertEqual(response.content, '{"status":"ok"}')
        self.assertEqual(response.model, "Qwen3-VL-4B-Instruct")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage["total_tokens"], 9)

        request, _ = transport.calls[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(request.get_method(), "POST")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "Qwen3-VL-4B-Instruct")
        self.assertEqual(body["messages"][1], {"role": "user", "content": "Ping"})
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["max_tokens"], 16)
        self.assertEqual(body["top_p"], 0.8)

    def test_chat_request_escapes_lone_surrogate_code_points(self) -> None:
        transport = QueueTransport(FakeResponse(chat_payload()))
        client = self.make_client(transport)

        client.chat([ChatMessage("user", "lone surrogate: \ud800")])

        request, _ = transport.calls[0]
        encoded_body = request.data.decode("ascii")
        self.assertIn(r"\ud800", encoded_body)

    def test_usage_keeps_integer_counters_and_ignores_optional_details(self) -> None:
        payload = chat_payload()
        payload["usage"] = {
            "prompt_tokens": 4,
            "completion_tokens": 5,
            "total_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": None,
        }
        transport = QueueTransport(FakeResponse(payload))

        response = self.make_client(transport).chat(
            [ChatMessage("user", "hello")]
        )

        self.assertEqual(
            response.usage,
            {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        )

    def test_invalid_usage_is_protocol_error(self) -> None:
        invalid_values = (True, -1, 1.5, "one", [])
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                payload = chat_payload()
                payload["usage"] = {"prompt_tokens": invalid_value}
                transport = QueueTransport(FakeResponse(payload))
                with self.assertRaises(ModelProtocolError):
                    self.make_client(transport).chat(
                        [ChatMessage("user", "hello")]
                    )

    def test_base_url_normalization_has_exactly_one_v1_suffix(self) -> None:
        examples = {
            "http://localhost:8000": "http://localhost:8000/v1",
            "http://localhost:8000/": "http://localhost:8000/v1",
            "http://localhost:8000/v1/": "http://localhost:8000/v1",
            "http://localhost:8000/v1/v1": "http://localhost:8000/v1",
            "https://example.test/api/v1/": "https://example.test/api/v1",
        }
        for source, expected in examples.items():
            with self.subTest(source=source):
                client = OpenAICompatibleClient(
                    base_url=source,
                    model="model",
                    timeout_s=1,
                    transport=QueueTransport(),
                )
                self.assertEqual(client.base_url, expected)

    def test_constructor_arguments_override_environment_defaults(self) -> None:
        environment = {
            "QWEN_API_BASE": "http://environment.invalid/v1",
            "QWEN_MODEL": "environment-model",
            "QWEN_API_KEY": "environment-key",
            "QWEN_REQUEST_TIMEOUT_S": "99",
        }
        transport = QueueTransport(FakeResponse({"data": []}))
        with patch.dict(os.environ, environment, clear=False):
            client = self.make_client(transport)

        client.healthcheck()
        request, timeout = transport.calls[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(timeout, 3.0)

    def test_environment_defaults_are_read_when_arguments_are_omitted(self) -> None:
        environment = {
            "QWEN_API_BASE": "http://127.0.0.1:9000/v1/",
            "QWEN_MODEL": "environment-model",
            "QWEN_API_KEY": "environment-key",
            "QWEN_REQUEST_TIMEOUT_S": "7.5",
        }
        transport = QueueTransport(FakeResponse({"data": []}))
        with patch.dict(os.environ, environment, clear=False):
            client = OpenAICompatibleClient(transport=transport)
        client.healthcheck()

        request, timeout = transport.calls[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:9000/v1/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer environment-key")
        self.assertEqual(timeout, 7.5)
        self.assertEqual(client.model, "environment-model")

    def test_connection_error_is_classified(self) -> None:
        transport = QueueTransport(urllib_error.URLError("connection refused"))
        with self.assertRaises(ModelConnectionError):
            self.make_client(transport).healthcheck()
        self.assertEqual(len(transport.calls), 1)

    def test_timeout_is_classified(self) -> None:
        transport = QueueTransport(socket.timeout("timed out"))
        with self.assertRaisesRegex(ModelConnectionError, "timed out") as raised:
            self.make_client(transport).healthcheck()
        self.assertEqual(len(transport.calls), 1)
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    def test_http_429_is_retried(self) -> None:
        transport = QueueTransport(http_error(429), FakeResponse({"data": []}))
        self.make_client(transport, max_retries=2).healthcheck()
        self.assertEqual(len(transport.calls), 2)

    def test_http_500_is_retried(self) -> None:
        transport = QueueTransport(http_error(500), FakeResponse({"data": []}))
        self.make_client(transport, max_retries=2).healthcheck()
        self.assertEqual(len(transport.calls), 2)

    def test_http_400_is_not_retried(self) -> None:
        transport = QueueTransport(http_error(400), FakeResponse({"data": []}))
        with self.assertRaises(ModelHTTPError) as raised:
            self.make_client(transport, max_retries=3).healthcheck()
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.attempts, 1)
        self.assertEqual(len(transport.calls), 1)
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    def test_invalid_json_is_protocol_error(self) -> None:
        transport = QueueTransport(FakeResponse(b"not-json"))
        with self.assertRaises(ModelProtocolError) as raised:
            self.make_client(transport).healthcheck()
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    def test_missing_choices_is_protocol_error(self) -> None:
        transport = QueueTransport(FakeResponse({"model": "model"}))
        with self.assertRaisesRegex(ModelProtocolError, "choices"):
            self.make_client(transport).chat([ChatMessage("user", "hello")])

    def test_missing_message_content_is_protocol_error(self) -> None:
        malformed_responses = (
            {"choices": [{}]},
            {"choices": [{"message": {}}]},
        )
        for payload in malformed_responses:
            with self.subTest(payload=payload):
                transport = QueueTransport(FakeResponse(payload))
                with self.assertRaisesRegex(ModelProtocolError, "message.content"):
                    self.make_client(transport).chat(
                        [ChatMessage("user", "hello")]
                    )

    def test_non_string_content_is_protocol_error(self) -> None:
        transport = QueueTransport(FakeResponse(chat_payload(["not", "text"])))
        with self.assertRaisesRegex(ModelProtocolError, "must be a string"):
            self.make_client(transport).chat([ChatMessage("user", "hello")])

    def test_api_key_is_redacted_from_exception_and_cause(self) -> None:
        api_key = "super-secret-api-key"
        transport = QueueTransport(
            urllib_error.URLError(f"connection failed with {api_key}")
        )
        with self.assertRaises(ModelConnectionError) as raised:
            self.make_client(transport, api_key=api_key).healthcheck()

        self.assertNotIn(api_key, str(raised.exception))
        self.assertNotIn(api_key, repr(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    def test_api_key_rejects_unsafe_characters_without_echoing_value(self) -> None:
        for api_key in (
            "secret-key\r\nInjected-Header: yes",
            "secret-key-密",
        ):
            with self.subTest(api_key=api_key), self.assertRaises(ValueError) as raised:
                self.make_client(QueueTransport(), api_key=api_key)

            self.assertNotIn(api_key, str(raised.exception))
            self.assertNotIn("secret-key", str(raised.exception))
            self.assertIsNone(raised.exception.__cause__)
            self.assertIsNone(raised.exception.__context__)

    def test_maximum_retry_count_is_respected(self) -> None:
        transport = QueueTransport(http_error(503), http_error(503), http_error(503))
        with self.assertRaises(ModelHTTPError) as raised:
            self.make_client(transport, max_retries=2).healthcheck()

        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.attempts, 3)


if __name__ == "__main__":
    unittest.main()
