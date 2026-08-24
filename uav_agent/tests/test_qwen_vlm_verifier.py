from __future__ import annotations

import json
import unittest

import numpy as np

from models import AsyncModelResult, ImageURLContentPart, ModelResponse
from perception.qwen_vlm_verifier import (
    QwenVLMVerifier,
    VisualReviewFrame,
    VisualReviewInput,
)
from perception.visual_review import VisualReviewProtocolError
from runtime.frame_store import FrameRef
from runtime.events import MissionEventType
from target import TargetLifecycle, TargetSnapshot, TargetSpec


def review_input() -> VisualReviewInput:
    ref = FrameRef("uav_1", "frame_1", 2.0, 4, 3)
    return VisualReviewInput(
        review_id="review_1",
        mission_id="mission_1",
        uav_id="uav_1",
        plan_version=1,
        observation_timestamp_s=2.0,
        frame_id="frame_1",
        target_spec=TargetSpec(
            "red cube",
            category="object",
            hard_attributes=("red", "cube-like"),
        ),
        current_skill="SEARCH",
        current_step_id="search_1",
        frames=(VisualReviewFrame(ref, np.zeros((3, 4, 3), dtype=np.uint8)),),
        target_snapshot=TargetSnapshot(
            target_id=None,
            description="red cube",
            lifecycle=TargetLifecycle.SEARCHING,
            confidence=None,
            last_seen_position=None,
            last_seen_velocity=None,
            last_seen_time_s=None,
            source=None,
        ),
        skill_feedback_summary={"phase": "SCANNING", "progress": 0.2},
        environment_context={"mission_elapsed_s": 2.0},
    )


def response_content(**overrides: object) -> str:
    data: dict[str, object] = {
        "schema_version": 1,
        "review_id": "review_1",
        "mission_id": "mission_1",
        "uav_id": "uav_1",
        "plan_version": 1,
        "observation_timestamp_s": 2.0,
        "frame_id": "frame_1",
        "decision": "AMBIGUOUS",
        "candidate": {
            "present": False,
            "bbox_xyxy_normalized": None,
            "description": None,
            "self_reported_confidence": None,
        },
        "scene_observations": ["target is too small"],
        "reason_codes": ["SMALL_TARGET"],
        "recommended_action": "INSPECT",
    }
    data.update(overrides)
    return json.dumps(data)


class QwenVLMVerifierTest(unittest.TestCase):
    def test_builds_multimodal_zero_temperature_structured_request(self) -> None:
        verifier = QwenVLMVerifier(max_image_side_px=64, jpeg_quality=80)
        value = review_input()

        request = verifier.build_async_request(value, request_id="request_1")

        self.assertEqual(request.uav_id, "uav_1")
        self.assertEqual(request.broker_priority, 4)
        self.assertTrue(request.broker_replaceable)
        self.assertEqual(request.options.temperature, 0.0)  # type: ignore[union-attr]
        self.assertIsNotNone(request.options.response_format)  # type: ignore[union-attr]
        user = request.messages[1]
        self.assertFalse(isinstance(user.content, str))
        self.assertTrue(any(isinstance(part, ImageURLContentPart) for part in user.content))  # type: ignore[union-attr]
        text_payload = user.content[0].text  # type: ignore[union-attr]
        self.assertNotIn("last_seen_position", text_payload)
        self.assertNotIn("last_seen_velocity", text_payload)
        self.assertNotIn("camera_rgb", text_payload)

    def test_context_and_frame_limits_are_fail_closed(self) -> None:
        value = review_input()
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            VisualReviewInput(
                **{
                    **{name: getattr(value, name) for name in value.__dataclass_fields__},
                    "environment_context": {"oracle_target_pose": [1, 2, 3]},
                }
            )
        with self.assertRaisesRegex(ValueError, "one and three"):
            VisualReviewInput(
                **{
                    **{name: getattr(value, name) for name in value.__dataclass_fields__},
                    "frames": (),
                }
            )

        allowed = VisualReviewInput(
            **{
                **{name: getattr(value, name) for name in value.__dataclass_fields__},
                "environment_context": {
                    "mission_elapsed_s": 2.0,
                    "trigger_event_type": MissionEventType.PATH_BLOCKED.value,
                },
            }
        )
        self.assertEqual(
            allowed.environment_context["trigger_event_type"],
            MissionEventType.PATH_BLOCKED.value,
        )
        event_request = QwenVLMVerifier().build_async_request(
            allowed,
            request_id="request_event_priority",
        )
        # Prompt/event context alone cannot elevate Broker priority.  The
        # coordinator applies P3 only from its trusted ReviewTicket.
        self.assertEqual(event_request.broker_priority, 4)
        with self.assertRaisesRegex(ValueError, "not supported"):
            VisualReviewInput(
                **{
                    **{
                        name: getattr(value, name)
                        for name in value.__dataclass_fields__
                    },
                    "environment_context": {
                        "mission_elapsed_s": 2.0,
                        "trigger_event_type": "payload_says_path_blocked",
                    },
                }
            )

    def test_parses_valid_result_and_rejects_wrong_or_stale_metadata(self) -> None:
        verifier = QwenVLMVerifier()
        value = review_input()
        result = AsyncModelResult(
            request_id="request_1",
            review_id="review_1",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=2.0,
            frame_id="frame_1",
            response=ModelResponse(response_content(), "model", "stop", {}),
            error_code=None,
            error_message=None,
        )

        parsed = verifier.parse_async_result(result, expectation=value.expectation)
        self.assertEqual(parsed.frame_id, "frame_1")

        stale = AsyncModelResult(
            request_id="request_1",
            review_id="review_1",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            observation_timestamp_s=2.0,
            frame_id="frame_1",
            response=ModelResponse(response_content(), "model", "stop", {}),
            error_code=None,
            error_message=None,
            stale=True,
        )
        with self.assertRaisesRegex(VisualReviewProtocolError, "stale"):
            verifier.parse_async_result(stale, expectation=value.expectation)

        wrong = AsyncModelResult(
            request_id="request_1",
            review_id="review_1",
            mission_id="mission_1",
            uav_id="uav_2",
            plan_version=1,
            observation_timestamp_s=2.0,
            frame_id="frame_1",
            response=ModelResponse(response_content(), "model", "stop", {}),
            error_code=None,
            error_message=None,
        )
        with self.assertRaisesRegex(VisualReviewProtocolError, "uav_id"):
            verifier.parse_async_result(wrong, expectation=value.expectation)


if __name__ == "__main__":
    unittest.main()
