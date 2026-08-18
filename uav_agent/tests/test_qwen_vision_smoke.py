from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from models.base import (
    GenerationOptions,
    ImageURLContentPart,
    ModelConnectionError,
    ModelResponse,
)
from scripts.run_qwen_vision_smoke import (
    VisionSmokeError,
    build_vision_smoke_schema,
    main,
    run_vision_smoke,
)


VALID_OUTPUT = {
    "schema_version": 1,
    "uav_id": "uav_1",
    "target_present": True,
    "bbox_xyxy_normalized": [0.1, 0.2, 0.5, 0.7],
    "description": "matching red target",
    "self_reported_confidence": 0.8,
}


class FakeClient:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[tuple[object, object]] = []

    def healthcheck(self) -> None:
        return None

    def chat(self, messages, *, options=None) -> ModelResponse:
        self.calls.append((messages, options))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        content = (
            self.outcome
            if isinstance(self.outcome, str)
            else json.dumps(self.outcome)
        )
        return ModelResponse(
            content=content,
            model="fake-qwen",
            finish_reason="stop",
            usage={},
        )


class VisionSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.image_path = Path(self.temporary.name) / "frame.png"
        Image.new("RGB", (20, 10), color=(220, 10, 5)).save(self.image_path)

    def test_schema_is_strict_and_binds_uav_id(self) -> None:
        schema = build_vision_smoke_schema("uav_1")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["uav_id"],
            {"type": "string", "const": "uav_1"},
        )
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_request_contains_typed_image_and_strict_response_format(self) -> None:
        client = FakeClient(VALID_OUTPUT)

        result = run_vision_smoke(
            image_path=self.image_path,
            uav_id="uav_1",
            target_description="red moving target",
            client=client,
            max_side_px=64,
            jpeg_quality=75,
        )

        self.assertEqual(result, VALID_OUTPUT)
        self.assertEqual(len(client.calls), 1)
        messages, options = client.calls[0]
        self.assertEqual(len(messages), 2)
        user_message = messages[1]
        self.assertIsInstance(user_message.content[1], ImageURLContentPart)
        data_url = user_message.content[1].url
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        self.assertNotIn(data_url, json.dumps(result))
        self.assertIsInstance(options, GenerationOptions)
        self.assertEqual(options.temperature, 0.0)
        response_format = options.response_format
        self.assertIsNotNone(response_format)
        assert response_format is not None
        self.assertEqual(response_format.name, "qwen_vision_smoke_v1")

    def test_missing_image_has_clear_error_without_model_call(self) -> None:
        client = FakeClient(VALID_OUTPUT)
        with self.assertRaises(VisionSmokeError) as raised:
            run_vision_smoke(
                image_path=Path(self.temporary.name) / "missing.png",
                uav_id="uav_1",
                target_description="target",
                client=client,
            )
        self.assertEqual(raised.exception.code, "IMAGE_NOT_FOUND")
        self.assertEqual(client.calls, [])

    def test_timeout_has_stable_error_code(self) -> None:
        client = FakeClient(ModelConnectionError("request timed out after 1 second"))
        with self.assertRaises(VisionSmokeError) as raised:
            run_vision_smoke(
                image_path=self.image_path,
                uav_id="uav_1",
                target_description="target",
                client=client,
            )
        self.assertEqual(raised.exception.code, "MODEL_TIMEOUT")

    def test_invalid_json_and_schema_are_rejected(self) -> None:
        examples = (
            ("```json\n{}\n```", "MODEL_OUTPUT_INVALID_JSON"),
            (
                '{"schema_version":1,"schema_version":1}',
                "MODEL_OUTPUT_INVALID_JSON",
            ),
            (
                json.dumps(VALID_OUTPUT).replace("0.8", "NaN"),
                "MODEL_OUTPUT_INVALID_JSON",
            ),
            ({"target_present": False}, "MODEL_OUTPUT_SCHEMA_INVALID"),
            (
                VALID_OUTPUT | {"uav_id": "uav_2"},
                "MODEL_OUTPUT_ROUTING_MISMATCH",
            ),
            (
                VALID_OUTPUT | {"bbox_xyxy_normalized": [0.8, 0.2, 0.1, 0.7]},
                "MODEL_OUTPUT_SCHEMA_INVALID",
            ),
            (
                VALID_OUTPUT
                | {"target_present": False, "bbox_xyxy_normalized": [0, 0, 1, 1]},
                "MODEL_OUTPUT_SCHEMA_INVALID",
            ),
        )
        for outcome, expected_code in examples:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(VisionSmokeError) as raised:
                    run_vision_smoke(
                        image_path=self.image_path,
                        uav_id="uav_1",
                        target_description="target",
                        client=FakeClient(outcome),
                    )
                self.assertEqual(raised.exception.code, expected_code)

    def test_cli_prints_only_parsed_json_without_base64_or_api_key(self) -> None:
        client = FakeClient(VALID_OUTPUT)
        stdout = StringIO()
        stderr = StringIO()
        with patch(
            "scripts.run_qwen_vision_smoke.OpenAICompatibleClient",
            return_value=client,
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "--image",
                    str(self.image_path),
                    "--uav-id",
                    "uav_1",
                    "--target-description",
                    "red target",
                    "--api-key",
                    "secret-key",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), VALID_OUTPUT)
        self.assertNotIn("base64", stdout.getvalue())
        self.assertNotIn("secret-key", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_arbitrary_transport_error_does_not_expose_payload_or_key(self) -> None:
        client = FakeClient(RuntimeError("secret-key data:image/jpeg;base64,pixels"))
        with self.assertRaises(VisionSmokeError) as raised:
            run_vision_smoke(
                image_path=self.image_path,
                uav_id="uav_1",
                target_description="target",
                client=client,
            )
        self.assertEqual(raised.exception.code, "MODEL_REQUEST_FAILED")
        self.assertNotIn("secret-key", str(raised.exception))
        self.assertNotIn("base64", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
