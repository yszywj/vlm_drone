from __future__ import annotations

import json
import threading
import time
import unittest

import numpy as np

from models import AsyncModelResult, AsyncModelWorker, ModelResponse
from planner.qwen_next_best_view import (
    NextBestViewRouting,
    QwenNextBestViewProvider,
    build_next_best_view_json_schema,
)
from planner.spatial import CircleRegion, CoordinateFrame
from skills.search_strategy import NextBestViewRequest


def _request(**overrides: object) -> NextBestViewRequest:
    values: dict[str, object] = {
        "region": CircleRegion(CoordinateFrame.WORLD_ENU, (0, 0, 0), 5),
        "target_description": "moving red vehicle",
        "observation_timestamp_s": 12.5,
        "uav_position_xyz_m": (0, 0, 5),
        "uav_yaw_rad": 0.25,
        "camera_rgb": np.zeros((8, 12, 3), dtype=np.uint8),
        "camera_position_m": (0, 0, 5),
        "camera_orientation_wxyz": (1, 0, 0, 0),
        "visited_viewpoints_xyz_m": ((0, 0, 5),),
        "coverage_ratio": 0.25,
        "max_viewpoints": 4,
        "search_altitude_m": 5,
    }
    values.update(overrides)
    return NextBestViewRequest(**values)  # type: ignore[arg-type]


class _Worker:
    def __init__(self) -> None:
        self.submitted = None
        self.result: AsyncModelResult | None = None

    def submit(self, request: object) -> None:
        self.submitted = request

    def poll(self, **kwargs: object) -> AsyncModelResult | None:
        del kwargs
        result, self.result = self.result, None
        return result


def _proposal_for(async_request: object, *, point: object = (2, 0, 5)) -> dict[str, object]:
    return {
        "schema_version": 1,
        "routing": {
            "request_id": async_request.request_id,
            "mission_id": async_request.mission_id,
            "uav_id": async_request.uav_id,
            "plan_version": async_request.plan_version,
            "frame_id": async_request.frame_id,
            "observation_timestamp_s": async_request.observation_timestamp_s,
        },
        "decision": "NEXT_VIEW",
        "coordinate_frame": "WORLD_ENU",
        "viewpoint_xyz_m": list(point),
        "rationale": "inspect an uncovered part of the region",
    }


def _result(async_request: object, proposal: object) -> AsyncModelResult:
    return AsyncModelResult(
        request_id=async_request.request_id,
        review_id=async_request.review_id,
        mission_id=async_request.mission_id,
        uav_id=async_request.uav_id,
        plan_version=async_request.plan_version,
        observation_timestamp_s=async_request.observation_timestamp_s,
        frame_id=async_request.frame_id,
        response=ModelResponse(
            json.dumps(proposal, allow_nan=False),
            "Qwen3-VL-4B-Instruct",
            "stop",
            {"total_tokens": 20},
        ),
        error_code=None,
        error_message=None,
    )


class QwenNextBestViewProviderTest(unittest.TestCase):
    def _provider(self, worker: object) -> QwenNextBestViewProvider:
        return QwenNextBestViewProvider(
            worker=worker,  # type: ignore[arg-type]
            uav_id="uav_1",
            routing_context=lambda: NextBestViewRouting("mission_1", 3),
            max_image_side_px=32,
        )

    def test_request_is_routed_strict_multimodal_and_prompt_safe(self) -> None:
        worker = _Worker()
        provider = self._provider(worker)
        provider.submit_next_best_view(_request())

        async_request = worker.submitted
        self.assertIsNotNone(async_request)
        self.assertEqual(async_request.plan_version, 3)
        self.assertEqual(async_request.messages[1].image_count, 1)
        options = async_request.options
        self.assertIsNotNone(options)
        self.assertEqual(options.response_format.name, "next_best_view_v1")
        schema = options.response_format.schema
        self.assertEqual(len(schema["oneOf"]), 2)
        prompt = async_request.messages[1].content[0].text
        self.assertIn('"coverage_ratio":0.25', prompt)
        self.assertIn('"shape":"CIRCLE"', prompt)
        self.assertIn("WORLD_ENU", prompt)
        self.assertNotIn("oracle_target_", prompt)

    def test_valid_macro_point_is_region_checked_and_record_has_no_image(self) -> None:
        worker = _Worker()
        provider = self._provider(worker)
        provider.submit_next_best_view(_request())
        worker.result = _result(
            worker.submitted,
            _proposal_for(worker.submitted),
        )

        poll = provider.poll_next_best_view()

        self.assertTrue(poll.completed)
        self.assertEqual(poll.viewpoint_xyz_m, (2.0, 0.0, 5.0))
        self.assertFalse(provider.request_in_flight)
        self.assertEqual(provider.records[0].decision, "NEXT_VIEW")
        rendered = json.dumps(provider.records[0].to_dict())
        self.assertNotIn("data:image/", rendered)
        self.assertNotIn("camera_rgb", rendered)

    def test_exhausted_is_distinct_from_nonblocking_pending(self) -> None:
        worker = _Worker()
        provider = self._provider(worker)
        provider.submit_next_best_view(_request())
        self.assertFalse(provider.poll_next_best_view().completed)
        proposal = _proposal_for(worker.submitted)
        proposal.update(
            decision="EXHAUSTED",
            viewpoint_xyz_m=None,
            rationale="coverage is sufficient",
        )
        worker.result = _result(worker.submitted, proposal)

        poll = provider.poll_next_best_view()

        self.assertTrue(poll.completed)
        self.assertIsNone(poll.viewpoint_xyz_m)
        self.assertEqual(provider.records[0].decision, "EXHAUSTED")

    def test_outside_nonfinite_wrong_frame_and_duplicate_fail_closed(self) -> None:
        cases = (
            {"viewpoint_xyz_m": [20, 0, 5]},
            {"viewpoint_xyz_m": [0, 0, 5]},
            {"viewpoint_xyz_m": [2, 0, float("nan")]},
            {"coordinate_frame": "UAV_START_FLU"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                worker = _Worker()
                provider = self._provider(worker)
                provider.submit_next_best_view(_request())
                proposal = _proposal_for(worker.submitted)
                proposal.update(changes)
                content = json.dumps(proposal, allow_nan=True)
                worker.result = AsyncModelResult(
                    request_id=worker.submitted.request_id,
                    review_id=worker.submitted.review_id,
                    mission_id=worker.submitted.mission_id,
                    uav_id=worker.submitted.uav_id,
                    plan_version=worker.submitted.plan_version,
                    observation_timestamp_s=worker.submitted.observation_timestamp_s,
                    frame_id=worker.submitted.frame_id,
                    response=ModelResponse(content, "qwen", "stop", {}),
                    error_code=None,
                    error_message=None,
                )
                with self.assertRaisesRegex(Exception, "strict spatial contract"):
                    provider.poll_next_best_view()
                self.assertEqual(
                    provider.records[0].error_code,
                    "INVALID_MODEL_PROPOSAL",
                )

    def test_actual_worker_keeps_blocking_model_call_off_submitter_thread(self) -> None:
        class BlockingClient:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()

            def chat(self, messages: object, *, options: object = None) -> ModelResponse:
                del messages, options
                self.started.set()
                self.release.wait(2.0)
                return ModelResponse("{}", "qwen", "stop", {})

        client = BlockingClient()
        worker = AsyncModelWorker(client, uav_id="uav_1")
        provider = self._provider(worker)
        try:
            started = time.monotonic()
            provider.submit_next_best_view(_request())
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.2)
            self.assertTrue(client.started.wait(1.0))
            self.assertFalse(provider.poll_next_best_view().completed)
        finally:
            provider.cancel_pending_next_best_view()
            client.release.set()
            worker.close(timeout_s=2.0)

    def test_schema_binds_every_routing_field(self) -> None:
        schema = build_next_best_view_json_schema(
            request_id="request_1",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=2,
            frame_id="frame_1",
            observation_timestamp_s=3.5,
        )
        for variant in schema["oneOf"]:
            routing = variant["properties"]["routing"]
            self.assertFalse(routing["additionalProperties"])
            self.assertEqual(
                routing["properties"]["observation_timestamp_s"]["const"],
                3.5,
            )


if __name__ == "__main__":
    unittest.main()
