"""Offline tests for strict, text-only LLM mission-intent planning."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import unittest

from models.base import (
    ChatMessage,
    GenerationOptions,
    ModelConnectionError,
    ModelResponse,
)
from planner.base import MissionPlanner, PlannerOutputError
from planner.llm_planner import LLMPlanner
from planner.prompt_builder import build_mission_planner_messages
from planner.schemas import (
    LandingZoneSpec,
    MissionIntent,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
)
from runtime.plan_validator import PlanValidator
from skills.manager import TaskPlan


PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "mission_planner_system.txt"
)


def _intent_dict(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "target_description": "moving target",
        "search_region": "search_area",
        "track_duration_s": 30.0,
        "landing_zone": "home",
        "takeoff_altitude_m": 10.0,
    }
    value.update(updates)
    return value


def _intent_json(**updates: object) -> str:
    return json.dumps(_intent_dict(**updates), ensure_ascii=False)


def _world_context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50, -50, 0),
        scene_max_xyz_m=(50, 50, 30),
        initial_uav_xyz_m=(0, 0, 0),
        search_regions={
            "search_area": SearchRegionSpec(
                "search_area",
                center_xyz_m=(20, 30, 0),
                radius_m=15,
                approach_xyz_m=(20, 12, 10),
                description="north sector with open ground",
            )
        },
        landing_zones={
            "home": LandingZoneSpec(
                "home",
                position_xy_m=(0, 0),
                description="launch pad",
            )
        },
        default_takeoff_altitude_m=10,
        default_track_duration_s=30,
        search_timeout_s=60,
    )


def _request(context: PlannerWorldContext | None = None) -> PlannerRequest:
    return PlannerRequest(
        "起飞后前往 search_area 搜寻移动目标，跟踪三十秒后返回 home 降落",
        context or _world_context(),
    )


class FakeModelClient:
    """Queue-backed ModelClient test double that performs no I/O."""

    def __init__(self, outcomes: Sequence[str | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[tuple[ChatMessage, ...], GenerationOptions | None]] = []

    def healthcheck(self) -> None:
        return None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        self.calls.append((tuple(messages), options))
        if not self._outcomes:
            raise AssertionError("unexpected extra model call")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ModelResponse(
            content=outcome,
            model="fake-qwen",
            finish_reason="stop",
            usage={},
        )


class _ExplodingLogger:
    def __getattr__(self, name: str) -> object:
        def explode(message: str) -> None:
            raise RuntimeError(f"logger {name} failed")

        return explode


class LLMPlannerTest(unittest.TestCase):
    def _planner(
        self,
        outcomes: Sequence[str | Exception],
        *,
        logger: object | None = None,
    ) -> tuple[LLMPlanner, FakeModelClient]:
        client = FakeModelClient(outcomes)
        return LLMPlanner(client, PROMPT_PATH, logger=logger), client

    def test_first_response_valid_json_returns_mission_intent(self) -> None:
        planner, client = self._planner([_intent_json()])

        result = planner.plan(_request())

        self.assertIsInstance(planner, MissionPlanner)
        self.assertIsInstance(result, MissionIntent)
        self.assertEqual(result, MissionIntent.from_dict(_intent_dict()))
        self.assertEqual(len(client.calls), 1)

    def test_plan_with_diagnostics_reports_legacy_llm_calls(self) -> None:
        planner, _client = self._planner([_intent_json()])

        execution = planner.plan_with_diagnostics(_request())

        self.assertIsInstance(execution.output, MissionIntent)
        self.assertEqual(execution.diagnostics.model_calls, 1)
        self.assertTrue(execution.diagnostics.initial_output_valid)
        self.assertFalse(execution.diagnostics.repair_used)
        self.assertFalse(execution.diagnostics.structured_output_enabled)

    def test_repaired_intent_reports_stable_schema_diagnostics(self) -> None:
        planner, _client = self._planner([
            json.dumps({"target_description": "moving target"}),
            _intent_json(),
        ])

        execution = planner.plan_with_diagnostics(_request())

        self.assertEqual(execution.diagnostics.model_calls, 2)
        self.assertTrue(execution.diagnostics.repair_succeeded)
        self.assertEqual(
            execution.diagnostics.initial_error_code,
            "SCHEMA_INVALID",
        )

    def test_single_json_code_fence_is_accepted_without_repair(self) -> None:
        fenced = f"```json\n{_intent_json()}\n```"
        planner, client = self._planner([fenced])

        result = planner.plan(_request())

        self.assertEqual(result.search_region, "search_area")
        self.assertEqual(len(client.calls), 1)

    def test_explanation_or_wrong_code_fence_is_not_accepted_as_json(self) -> None:
        invalid_wrappers = (
            f"Here is the result:\n{_intent_json()}",
            f"```python\n{_intent_json()}\n```",
            f"```json\n```json\n{_intent_json()}\n```\n```",
        )
        for output in invalid_wrappers:
            with self.subTest(output=output):
                planner, client = self._planner([output, output])
                with self.assertRaises(PlannerOutputError):
                    planner.plan(_request())
                self.assertEqual(len(client.calls), 2)

    def test_first_invalid_response_is_repaired_once(self) -> None:
        invalid = '{"target_description": "moving target",}'
        planner, client = self._planner([invalid, _intent_json()])

        result = planner.plan(_request())

        self.assertEqual(result.landing_zone, "home")
        self.assertEqual(len(client.calls), 2)
        repair_messages = client.calls[1][0]
        repair_payload = json.loads(str(repair_messages[-1].content))
        self.assertEqual(repair_payload["original_output"], invalid)
        self.assertIn("invalid JSON", repair_payload["validation_error"])

    def test_two_invalid_responses_raise_planner_output_error(self) -> None:
        planner, client = self._planner(["not JSON", "[]"])

        with self.assertRaisesRegex(PlannerOutputError, "after one repair"):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 2)

    def test_missing_required_field_is_rejected(self) -> None:
        missing = _intent_dict()
        missing.pop("target_description")
        raw = json.dumps(missing)
        planner, client = self._planner([raw, raw])

        with self.assertRaisesRegex(PlannerOutputError, "missing required fields"):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 2)

    def test_unknown_field_is_rejected(self) -> None:
        raw = _intent_json(oracle_hint=[1, 2, 3])
        planner, client = self._planner([raw, raw])

        with self.assertRaisesRegex(PlannerOutputError, "unknown fields"):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 2)

    def test_illegal_numbers_are_rejected(self) -> None:
        outputs = (
            '{"target_description":"moving target","search_region":"search_area",'
            '"track_duration_s":NaN,"landing_zone":"home",'
            '"takeoff_altitude_m":10.0}'
        )
        planner, client = self._planner([outputs, outputs])

        with self.assertRaisesRegex(PlannerOutputError, "non-finite JSON number"):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 2)

    def test_overflowing_json_integer_is_rejected_as_planner_output(self) -> None:
        # Stay below CPython's JSON integer digit limit so MissionIntent's
        # numeric normalization, rather than json.loads(), exercises overflow.
        huge_integer = "9" * 1000
        output = (
            '{"target_description":"moving target","search_region":"search_area",'
            f'"track_duration_s":{huge_integer},"landing_zone":"home",'
            '"takeoff_altitude_m":10.0}'
        )
        planner, client = self._planner([output, output])

        with self.assertRaisesRegex(PlannerOutputError, "OverflowError"):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 2)

    def test_deeply_nested_json_is_rejected_as_planner_output(self) -> None:
        output = "[" * 2000 + "]" * 2000
        planner, client = self._planner([output, output])

        with self.assertRaisesRegex(PlannerOutputError, "RecursionError"):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 2)

    def test_empty_model_output_is_rejected_after_one_repair(self) -> None:
        planner, client = self._planner(["", "  "])

        with self.assertRaisesRegex(PlannerOutputError, "non-empty"):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 2)

    def test_model_connection_error_propagates_without_repair(self) -> None:
        failure = ModelConnectionError("service unavailable")
        planner, client = self._planner([failure])

        with self.assertRaises(ModelConnectionError) as raised:
            planner.plan(_request())

        self.assertIs(raised.exception, failure)
        self.assertEqual(len(client.calls), 1)

    def test_plan_never_calls_model_more_than_twice(self) -> None:
        planner, client = self._planner(["bad", "still bad", _intent_json()])

        with self.assertRaises(PlannerOutputError):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 2)

    def test_temperature_is_zero_for_initial_and_repair_calls(self) -> None:
        planner, client = self._planner(["bad", _intent_json()])

        planner.plan(_request())

        self.assertEqual(len(client.calls), 2)
        for _, options in client.calls:
            self.assertIsInstance(options, GenerationOptions)
            self.assertEqual(options.temperature, 0.0)

    def test_prompt_has_no_oracle_or_evaluator_data(self) -> None:
        planner, client = self._planner([_intent_json()])

        planner.plan(_request())

        prompt = "\n".join(
            str(message.content) for message in client.calls[0][0]
        ).casefold()
        self.assertNotIn("oracle_target", prompt)
        self.assertNotIn("evaluatorframe", prompt)
        self.assertNotIn("camera_rgb", prompt)

    def test_runtime_uses_shared_prompt_builder_byte_for_byte(self) -> None:
        planner, client = self._planner([_intent_json()])
        request = _request()
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

        planner.plan(request)
        expected = build_mission_planner_messages(
            request.instruction,
            request.world_context,
            system_prompt,
        )

        actual = client.calls[0][0]
        self.assertEqual(actual, expected)
        self.assertEqual(
            tuple(str(message.content).encode("utf-8") for message in actual),
            tuple(str(message.content).encode("utf-8") for message in expected),
        )

    def test_shared_prompt_builder_exposes_only_runtime_public_context(self) -> None:
        context = _world_context()
        messages = build_mission_planner_messages(
            _request(context).instruction,
            context,
            PROMPT_PATH.read_text(encoding="utf-8"),
        )

        payload = json.loads(str(messages[1].content))
        self.assertEqual(
            set(payload),
            {"task", "trusted_world_context", "user_instruction"},
        )
        trusted = payload["trusted_world_context"]
        self.assertEqual(
            set(trusted),
            {
                "scene_bounds_m",
                "search_regions",
                "landing_zones",
                "default_takeoff_altitude_m",
                "default_track_duration_s",
            },
        )
        serialized = str(messages[1].content).casefold()
        for forbidden in (
            "gold",
            "oracle",
            "spawn",
            "evaluator",
            "frame",
            "image",
            "initial_uav",
            "approach_xyz",
            "center_xyz",
            "position_xy",
            "timeout",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)

    def test_prompt_omits_geometry_and_random_spawn_coordinates(self) -> None:
        context = PlannerWorldContext(
            scene_min_xyz_m=(-100, -100, 0),
            scene_max_xyz_m=(100, 100, 50),
            initial_uav_xyz_m=(7.654321, -8.765432, 1.234567),
            search_regions={
                "search_area": SearchRegionSpec(
                    "search_area",
                    center_xyz_m=(12.345678, 23.456789, 2.345679),
                    radius_m=14.567891,
                    approach_xyz_m=(16.789123, 27.891234, 9.876543),
                    description="trusted northern sector",
                )
            },
            landing_zones={
                "home": LandingZoneSpec(
                    "home",
                    position_xy_m=(31.415926, -27.182818),
                    ground_altitude_m=0.123456,
                    description="trusted launch pad",
                )
            },
            default_takeoff_altitude_m=10,
            default_track_duration_s=30,
            search_timeout_s=60,
        )
        planner, client = self._planner([_intent_json()])

        planner.plan(_request(context))

        prompt = "\n".join(str(message.content) for message in client.calls[0][0])
        hidden_geometry = (
            7.654321,
            -8.765432,
            1.234567,
            12.345678,
            23.456789,
            2.345679,
            14.567891,
            16.789123,
            27.891234,
            9.876543,
            31.415926,
            -27.182818,
            0.123456,
        )
        for value in hidden_geometry:
            with self.subTest(value=value):
                self.assertNotIn(str(value), prompt)
        self.assertIn('"name":"search_area"', prompt)
        self.assertIn('"name":"home"', prompt)

    def test_prompt_builder_rejects_forbidden_world_metadata_markers(self) -> None:
        markers = (
            "oracle_target",
            "ORACLE",
            "target-spawn",
            "targetSpawn",
            "出生点",
            "真实位置",
            "target_pose",
            "target position",
            "targetVelocity",
            "目标坐标",
            "目标 的 坐标",
            "目标速度",
            "EvaluatorFrame",
            "evaluator",
            "image_rgb",
            "video-stream",
            "frame",
            "camera",
            "cameraFrame",
        )
        locations = (
            "search_key",
            "search_name",
            "search_description",
            "landing_key",
            "landing_name",
            "landing_description",
        )

        for marker in markers:
            for location in locations:
                with self.subTest(marker=marker, location=location):
                    search_key = marker if location == "search_key" else "search_area"
                    search_name = marker if location == "search_name" else "search_area"
                    search_description = (
                        marker
                        if location == "search_description"
                        else "north sector with open ground"
                    )
                    landing_key = marker if location == "landing_key" else "home"
                    landing_name = marker if location == "landing_name" else "home"
                    landing_description = (
                        marker
                        if location == "landing_description"
                        else "launch pad"
                    )
                    context = PlannerWorldContext(
                        scene_min_xyz_m=(-50, -50, 0),
                        scene_max_xyz_m=(50, 50, 30),
                        initial_uav_xyz_m=(0, 0, 0),
                        search_regions={
                            search_key: SearchRegionSpec(
                                search_name,
                                center_xyz_m=(20, 30, 0),
                                radius_m=15,
                                approach_xyz_m=(20, 12, 10),
                                description=search_description,
                            )
                        },
                        landing_zones={
                            landing_key: LandingZoneSpec(
                                landing_name,
                                position_xy_m=(0, 0),
                                description=landing_description,
                            )
                        },
                        default_takeoff_altitude_m=10,
                        default_track_duration_s=30,
                        search_timeout_s=60,
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        "forbidden hidden-state or media marker",
                    ):
                        build_mission_planner_messages(
                            _request(context).instruction,
                            context,
                            PROMPT_PATH.read_text(encoding="utf-8"),
                        )

    def test_llm_planner_rejects_tainted_context_before_model_call(self) -> None:
        context = _world_context()
        tainted = PlannerWorldContext(
            scene_min_xyz_m=context.scene_min_xyz_m,
            scene_max_xyz_m=context.scene_max_xyz_m,
            initial_uav_xyz_m=context.initial_uav_xyz_m,
            search_regions={
                "search_area": SearchRegionSpec(
                    "search_area",
                    center_xyz_m=(20, 30, 0),
                    radius_m=15,
                    approach_xyz_m=(20, 12, 10),
                    description="contains oracle_target_pose",
                )
            },
            landing_zones=context.landing_zones,
            default_takeoff_altitude_m=context.default_takeoff_altitude_m,
            default_track_duration_s=context.default_track_duration_s,
            search_timeout_s=context.search_timeout_s,
        )
        planner, client = self._planner([_intent_json()])

        with self.assertRaisesRegex(
            ValueError,
            "forbidden hidden-state or media marker",
        ):
            planner.plan(_request(tainted))

        self.assertEqual(client.calls, [])

    def test_llm_planner_returns_intent_and_validator_builds_six_steps(self) -> None:
        planner, _ = self._planner([_intent_json()])
        request = _request()

        intent = planner.plan(request)
        compiled = PlanValidator().validate_and_compile(
            intent,
            request.world_context,
            source="llm",
        )

        self.assertNotIsInstance(intent, TaskPlan)
        self.assertEqual(
            [step["skill"] for step in compiled.task_plan.to_dicts()],
            ["TAKEOFF", "GOTO", "SEARCH", "TRACK", "GOTO", "LAND"],
        )

    def test_logger_failure_never_changes_valid_planning_result(self) -> None:
        planner, client = self._planner(
            [_intent_json()],
            logger=_ExplodingLogger(),
        )

        result = planner.plan(_request())

        self.assertEqual(result.target_description, "moving target")
        self.assertEqual(len(client.calls), 1)

    def test_duplicate_json_fields_are_not_silently_overwritten(self) -> None:
        duplicate = (
            '{"target_description":"first","target_description":"second",'
            '"search_region":"search_area","track_duration_s":30,'
            '"landing_zone":"home","takeoff_altitude_m":10}'
        )
        planner, client = self._planner([duplicate, duplicate])

        with self.assertRaisesRegex(PlannerOutputError, "duplicate JSON field"):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
