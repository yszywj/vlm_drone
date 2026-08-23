from __future__ import annotations

import threading
import time
import unittest

from models.async_worker import AsyncModelRequest, AsyncModelWorker
from models.base import (
    ChatMessage,
    GenerationOptions,
    ModelHTTPError,
    ModelResponse,
)


def request(
    request_id: str,
    *,
    uav_id: str = "uav_1",
    timestamp_s: float = 1.0,
) -> AsyncModelRequest:
    return AsyncModelRequest(
        request_id=request_id,
        review_id=f"review_{request_id}",
        mission_id="mission_1",
        uav_id=uav_id,
        plan_version=1,
        observation_timestamp_s=timestamp_s,
        frame_id=f"frame_{request_id}",
        messages=(ChatMessage("user", "inspect"),),
        options=GenerationOptions(temperature=0.0),
    )


class ControlledModelClient:
    def __init__(self, *, block_first: bool = False) -> None:
        self.block_first = block_first
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.second_finished = threading.Event()
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def healthcheck(self) -> None:
        return None

    def chat(self, messages, *, options=None) -> ModelResponse:
        del messages, options
        with self._lock:
            self.calls += 1
            call_number = self.calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if call_number == 1:
                self.first_started.set()
                if self.block_first and not self.release_first.wait(2.0):
                    raise AssertionError("test did not release first model call")
            response = ModelResponse(
                content=f'{{"call":{call_number}}}',
                model="fake",
                finish_reason="stop",
                usage={},
            )
            if call_number == 2:
                self.second_finished.set()
            return response
        finally:
            with self._lock:
                self.active -= 1


class FailingModelClient:
    def __init__(self) -> None:
        self.started = threading.Event()

    def healthcheck(self) -> None:
        return None

    def chat(self, messages, *, options=None) -> ModelResponse:
        del messages, options
        self.started.set()
        raise ModelHTTPError(
            "Authorization: Bearer TOP_SECRET api_key=TOP_SECRET"
        )


class AsyncModelRequestTest(unittest.TestCase):
    def test_request_snapshots_messages_and_validates_metadata(self) -> None:
        source = [ChatMessage("user", "inspect")]
        item = AsyncModelRequest(
            request_id="request_1",
            review_id="review_1",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=0,
            frame_id="frame_1",
            messages=source,
        )
        source.append(ChatMessage("user", "later"))
        self.assertIsInstance(item.messages, tuple)
        self.assertEqual(len(item.messages), 1)

        invalid_values = (
            {"request_id": "bad id"},
            {"review_id": ""},
            {"mission_id": 1},
            {"uav_id": "_bad"},
            {"plan_version": 0},
            {"observation_timestamp_s": -1},
            {"frame_id": "bad/frame"},
            {"messages": []},
            {"messages": [{"role": "user"}]},
        )
        defaults = {
            "request_id": "request_1",
            "review_id": "review_1",
            "mission_id": "mission_1",
            "uav_id": "uav_1",
            "plan_version": 1,
            "observation_timestamp_s": 1.0,
            "frame_id": "frame_1",
            "messages": (ChatMessage("user", "inspect"),),
        }
        for override in invalid_values:
            with self.subTest(override=override), self.assertRaises(
                (TypeError, ValueError)
            ):
                AsyncModelRequest(**(defaults | override))


class AsyncModelWorkerTest(unittest.TestCase):
    def test_submit_returns_without_waiting_for_http(self) -> None:
        client = ControlledModelClient(block_first=True)
        worker = AsyncModelWorker(client, uav_id="uav_1")
        try:
            started_at = time.monotonic()
            worker.submit(request("request_1"))
            elapsed = time.monotonic() - started_at

            self.assertLess(elapsed, 0.1)
            self.assertTrue(client.first_started.wait(1.0))
            self.assertTrue(worker.is_busy)
            client.release_first.set()
        finally:
            client.release_first.set()
            worker.close(timeout_s=2.0)

        result = worker.poll()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.succeeded)
        self.assertEqual(result.request_id, "request_1")
        self.assertEqual(result.uav_id, "uav_1")

    def test_new_request_marks_running_response_stale_and_keeps_single_inflight(self) -> None:
        client = ControlledModelClient(block_first=True)
        worker = AsyncModelWorker(client, uav_id="uav_1")
        try:
            worker.submit(request("request_1", timestamp_s=1.0))
            self.assertTrue(client.first_started.wait(1.0))
            worker.submit(request("request_2", timestamp_s=2.0))
            client.release_first.set()
            self.assertTrue(client.second_finished.wait(1.0))
            worker.close(timeout_s=2.0)

            result = worker.poll(expected_request_id="request_2")
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.request_id, "request_2")
            self.assertFalse(result.stale)
            self.assertEqual(client.max_active, 1)
            self.assertEqual(worker.discarded_result_count, 1)
        finally:
            client.release_first.set()
            worker.close(timeout_s=2.0)

    def test_stale_response_can_be_inspected_explicitly(self) -> None:
        client = ControlledModelClient(block_first=True)
        worker = AsyncModelWorker(client, uav_id="uav_1")
        try:
            worker.submit(request("request_1"))
            self.assertTrue(client.first_started.wait(1.0))
            worker.submit(request("request_2"))
            client.release_first.set()
            self.assertTrue(client.second_finished.wait(1.0))
            worker.close(timeout_s=2.0)

            result = worker.poll(include_stale=True)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.request_id, "request_1")
            self.assertTrue(result.stale)
        finally:
            client.release_first.set()
            worker.close(timeout_s=2.0)

    def test_worker_rejects_cross_uav_request(self) -> None:
        client = ControlledModelClient()
        with AsyncModelWorker(client, uav_id="uav_1") as worker:
            with self.assertRaisesRegex(ValueError, "does not match"):
                worker.submit(request("request_2", uav_id="uav_2"))
        self.assertEqual(client.calls, 0)

    def test_context_manager_closes_worker_and_rejects_new_work(self) -> None:
        client = ControlledModelClient()
        with AsyncModelWorker(client, uav_id="uav_1") as worker:
            worker.submit(request("request_1"))
            self.assertTrue(client.first_started.wait(1.0))
        self.assertTrue(worker.is_closed)
        self.assertFalse(worker.is_busy)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            worker.submit(request("request_2"))

    def test_poll_discards_result_older_than_requested_observation(self) -> None:
        client = ControlledModelClient()
        worker = AsyncModelWorker(client, uav_id="uav_1")
        worker.submit(request("request_1", timestamp_s=1.0))
        self.assertTrue(client.first_started.wait(1.0))
        worker.close(timeout_s=2.0)

        self.assertIsNone(worker.poll(minimum_observation_timestamp_s=2.0))
        self.assertEqual(worker.discarded_result_count, 1)

    def test_model_failure_uses_stable_non_secret_error_category(self) -> None:
        client = FailingModelClient()
        worker = AsyncModelWorker(client, uav_id="uav_1")
        worker.submit(request("request_1"))
        self.assertTrue(client.started.wait(1.0))
        worker.close(timeout_s=2.0)

        result = worker.poll()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.error_code, "MODEL_REQUEST_FAILED")
        self.assertEqual(result.error_message, "ModelHTTPError")
        self.assertNotIn("TOP_SECRET", result.error_message)


if __name__ == "__main__":
    unittest.main()
