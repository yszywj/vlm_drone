from __future__ import annotations

import json
from io import BytesIO
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

from perception.yolo_client import (
    YoloClientRequestTimeout,
    YoloClientResponseError,
    YoloClientStreamBusy,
    YoloModelInfo,
    YoloServiceClient,
    encode_rgb_jpeg,
    validate_yolo_model_identity,
)
from yolo_service.protocol import TargetQuery, TrackRequest


def request() -> TrackRequest:
    return TrackRequest(
        schema_version=1,
        request_id="request_1",
        mission_id="mission_1",
        uav_id="uav_1",
        stream_id="mission_1:uav_1",
        frame_id="frame_1",
        timestamp_s=1.0,
        target_query=TargetQuery((0,), ()),
    )


class YoloClientTest(unittest.TestCase):
    def test_health_requires_explicit_ready_true(self) -> None:
        def transport(method, url, body, headers, timeout):
            del method, url, body, headers, timeout
            return json.dumps(
                {"schema_version": 1, "status": "ok", "ready": False}
            ).encode()

        with self.assertRaisesRegex(YoloClientResponseError, "health response"):
            YoloServiceClient(transport=transport).health()

    def test_loopback_authority_cannot_be_bypassed_with_userinfo(self) -> None:
        invalid = (
            "http://127.0.0.1:8011@evil.example:80",
            "http://localhost:8011@evil.example:80",
            "http://127.0.0.1:8011/path",
            "http://127.0.0.1:8011?redirect=evil",
            "https://127.0.0.1:8011",
            "http://127.0.0.1",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                YoloServiceClient(base_url=value, transport=lambda *args: b"{}")

    def test_track_uses_multipart_jpeg_and_checks_all_routes(self) -> None:
        seen: dict[str, object] = {}

        def transport(method, url, body, headers, timeout):
            seen.update(method=method, url=url, body=body, headers=headers, timeout=timeout)
            payload = {
                **request().to_dict(),
                "detections": [],
                "timing_ms": {"decode": 1.0, "inference": 2.0, "tracking": 0.0, "total": 3.0},
            }
            payload.pop("target_query")
            return json.dumps(payload).encode()

        client = YoloServiceClient(transport=transport)
        response = client.track(
            request(),
            np.zeros((8, 8, 3), dtype=np.uint8),
        )
        self.assertEqual(response.request_id, "request_1")
        self.assertIn("multipart/form-data", seen["headers"]["Content-Type"])
        self.assertIn(b'name="request_json"', seen["body"])
        self.assertIn(b'name="image"', seen["body"])

    def test_mismatched_response_is_rejected(self) -> None:
        def transport(method, url, body, headers, timeout):
            payload = {
                **request().to_dict(),
                "request_id": "request_other",
                "detections": [],
                "timing_ms": {"decode": 0.0, "inference": 0.0, "tracking": 0.0, "total": 0.0},
            }
            payload.pop("target_query")
            return json.dumps(payload).encode()

        with self.assertRaises(YoloClientResponseError):
            YoloServiceClient(transport=transport).track(
                request(), np.zeros((2, 2, 3), dtype=np.uint8)
            )

    def test_transport_timeout_preserves_unknown_remote_completion(self) -> None:
        def transport(method, url, body, headers, timeout):
            del method, url, body, headers, timeout
            raise TimeoutError("client deadline")

        with self.assertRaises(YoloClientRequestTimeout):
            YoloServiceClient(transport=transport).track(
                request(), np.zeros((2, 2, 3), dtype=np.uint8)
            )

    @patch("perception.yolo_client._DIRECT_LOOPBACK_OPENER.open")
    def test_http_stream_busy_is_typed_as_transient_backpressure(
        self,
        mocked_open,
    ) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": "STREAM_BUSY",
                    "message": "only one inference may be active",
                }
            }
        ).encode("utf-8")
        mocked_open.side_effect = HTTPError(
            "http://127.0.0.1:8011/v1/track",
            409,
            "Conflict",
            None,
            BytesIO(body),
        )

        with self.assertRaises(YoloClientStreamBusy) as raised:
            YoloServiceClient().track(
                request(), np.zeros((2, 2, 3), dtype=np.uint8)
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.error_code, "STREAM_BUSY")

    def test_rgb_encoder_does_not_swap_channels(self) -> None:
        from PIL import Image
        from io import BytesIO

        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        decoded = np.asarray(Image.open(BytesIO(encode_rgb_jpeg(rgb))))
        self.assertGreater(float(decoded[:, :, 0].mean()), 240.0)
        self.assertLess(float(decoded[:, :, 2].mean()), 15.0)

    def test_model_identity_mismatch_reports_complete_audit_context(self) -> None:
        with self.assertRaises(YoloClientResponseError) as raised:
            validate_yolo_model_identity(
                YoloModelInfo("yolo", ((0, "person"),), "b" * 64),
                expected_model_family="yolo",
                expected_model_names={0: "cube"},
                expected_model_sha256="a" * 64,
                worker_url="http://127.0.0.1:8011",
            )
        message = str(raised.exception)
        self.assertIn("worker_url=http://127.0.0.1:8011", message)
        self.assertIn("expected_model_sha256='" + "a" * 64 + "'", message)
        self.assertIn("actual_model_sha256='" + "b" * 64 + "'", message)
        self.assertIn("actual_model_names={0: 'person'}", message)

    def test_model_identity_accepts_exact_configured_worker(self) -> None:
        validate_yolo_model_identity(
            YoloModelInfo("yolo", ((0, "cube"),), "a" * 64),
            expected_model_family="yolo",
            expected_model_names={0: "cube"},
            expected_model_sha256="a" * 64,
            worker_url="http://127.0.0.1:8011",
        )


if __name__ == "__main__":
    unittest.main()
