from __future__ import annotations

import asyncio
import json
import unittest

from yolo_service.app import BoundedRequestBodyMiddleware


class BoundedRequestBodyMiddlewareTest(unittest.TestCase):
    def test_post_without_content_length_is_rejected_while_streaming(self) -> None:
        downstream_completed = False

        async def downstream(scope, receive, send) -> None:
            nonlocal downstream_completed
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if not message.get("more_body", False):
                    break
            downstream_completed = True
            await send({"type": "http.response.start", "status": 204, "headers": ()})
            await send({"type": "http.response.body", "body": b""})

        middleware = BoundedRequestBodyMiddleware(downstream, max_post_bytes=10)
        chunks = [b"123456", b"78901"]
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            body = chunks.pop(0)
            return {
                "type": "http.request",
                "body": body,
                "more_body": bool(chunks),
            }

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/track",
            "headers": ((b"content-type", b"multipart/form-data; boundary=x"),),
        }
        asyncio.run(middleware(scope, receive, send))

        self.assertFalse(downstream_completed)
        self.assertEqual(sent[0]["status"], 413)
        payload = json.loads(sent[1]["body"])
        self.assertEqual(payload["error"]["code"], "IMAGE_TOO_LARGE")

    def test_declared_body_limit_is_only_an_early_rejection(self) -> None:
        downstream_called = False

        async def downstream(scope, receive, send) -> None:
            nonlocal downstream_called
            downstream_called = True

        middleware = BoundedRequestBodyMiddleware(downstream, max_post_bytes=10)
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            raise AssertionError("oversized declared body must not be read")

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/track",
            "headers": ((b"content-length", b"11"),),
        }
        asyncio.run(middleware(scope, receive, send))

        self.assertFalse(downstream_called)
        self.assertEqual(sent[0]["status"], 413)


if __name__ == "__main__":
    unittest.main()
