"""Pure-Python tests for the strict YOLO service wire contract."""

from __future__ import annotations

import json
import unittest

from yolo_service.protocol import (
    ProtocolValidationError,
    RouteMismatchError,
    TargetQuery,
    TimingMs,
    TrackDetection,
    TrackRequest,
    TrackResponse,
)


def _request() -> TrackRequest:
    return TrackRequest(
        schema_version=1,
        request_id="request_1",
        mission_id="mission_1",
        uav_id="uav_1",
        stream_id="mission_1:uav_1",
        frame_id="frame_1",
        timestamp_s=12.3,
        target_query=TargetQuery(class_ids=(0,)),
    )


class YoloProtocolTest(unittest.TestCase):
    def test_request_round_trip_and_strict_stream_route(self) -> None:
        request = _request()
        self.assertEqual(TrackRequest.from_dict(request.to_dict()), request)
        invalid = request.to_dict()
        invalid["stream_id"] = "mission_other:uav_1"
        with self.assertRaisesRegex(ProtocolValidationError, "stream_id"):
            TrackRequest.from_dict(invalid)

    def test_unknown_fields_and_non_finite_json_are_rejected(self) -> None:
        unknown = _request().to_dict()
        unknown["model_path"] = "/server/secret.pt"
        with self.assertRaisesRegex(ProtocolValidationError, "unknown"):
            TrackRequest.from_dict(unknown)
        encoded = json.dumps(_request().to_dict()).replace("12.3", "NaN")
        with self.assertRaisesRegex(ProtocolValidationError, "non-finite"):
            TrackRequest.from_json(encoded)

    def test_query_is_bounded_and_requires_one_unambiguous_mode(self) -> None:
        with self.assertRaisesRegex(ProtocolValidationError, "contain"):
            TargetQuery()
        with self.assertRaisesRegex(ProtocolValidationError, "duplicates"):
            TargetQuery(class_ids=(0, 0))
        with self.assertRaisesRegex(ProtocolValidationError, "surrounding"):
            TargetQuery(text_prompts=(" person ",))

    def test_bbox_confidence_and_unknown_detection_fields_are_rejected(self) -> None:
        valid = {
            "track_id": 7,
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.86,
            "bbox_xyxy_normalized": [0.3, 0.25, 0.42, 0.71],
        }
        self.assertEqual(TrackDetection.from_dict(valid).track_id, 7)
        for bbox in (
            [0.3, 0.25, 1.1, 0.71],
            [0.4, 0.25, 0.4, 0.71],
            [0.3, 0.25, float("nan"), 0.71],
        ):
            with self.subTest(bbox=bbox), self.assertRaises(ProtocolValidationError):
                TrackDetection.from_dict({**valid, "bbox_xyxy_normalized": bbox})
        with self.assertRaises(ProtocolValidationError):
            TrackDetection.from_dict({**valid, "confidence": float("inf")})
        with self.assertRaisesRegex(ProtocolValidationError, "unknown"):
            TrackDetection.from_dict({**valid, "tensor": [1, 2, 3]})

    def test_response_route_mismatch_is_explicit(self) -> None:
        request = _request()
        response = TrackResponse(
            1,
            "request_other",
            request.mission_id,
            request.uav_id,
            request.stream_id,
            request.frame_id,
            request.timestamp_s,
            (),
            TimingMs(0, 0, 0, 0),
        )
        with self.assertRaisesRegex(RouteMismatchError, "request_id"):
            response.assert_matches(request)


if __name__ == "__main__":
    unittest.main()
