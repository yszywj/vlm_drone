"""Offline tests for constrained text-only dynamic Skill planning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
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
from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.prompt_builder import (
    build_dynamic_skill_planner_messages,
    build_mission_planner_messages,
)
from planner.schemas import (
    LandingZoneSpec,
    NavigationPointSpec,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraft,
)
from planner.scripted_dynamic_planner import ScriptedDynamicPlanner
from planner.skill_catalog import SkillCatalog, build_default_skill_catalog
from skills.plan import TaskPlan


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "dynamic_skill_planner_system.txt"
)
LEGACY_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "mission_planner_system.txt"
)
LIMITS = {
    "max_plan_steps": 10,
    "max_goto_calls": 5,
    "max_search_calls": 1,
    "max_track_calls": 2,
    "max_reacquire_attempts_per_track": 2,
    "max_total_reacquire_attempts": 4,
    "min_track_duration_s": 1.0,
    "max_track_duration_s": 600.0,
}


def _draft_dict() -> dict[str, object]:
    return {
        "schema_version": 1,
        "steps": [
            {
                "id": "takeoff_1",
                "skill": "TAKEOFF",
                "args": {"altitude_m": 10},
            },
            {
                "id": "goto_search",
                "skill": "GOTO",
                "args": {"destination": "search_area"},
            },
            {
                "id": "search_1",
                "skill": "SEARCH",
                "args": {
                    "region": "search_area",
                    "target_description": "moving target",
                },
            },
            {
                "id": "track_1",
                "skill": "TRACK",
                "args": {
                    "target_ref": "$search_1.target_id",
                    "duration_s": 10,
                },
                "recovery": {
                    "skill": "REACQUIRE",
                    "max_attempts": 2,
                    "search_radius_m": 10,
                    "timeout_s": 30,
                },
            },
            {
                "id": "goto_home",
                "skill": "GOTO",
                "args": {"destination": "home"},
            },
            {"id": "land_1", "skill": "LAND", "args": {"zone": "home"}},
        ],
    }


def _draft_json() -> str:
    return json.dumps(_draft_dict(), ensure_ascii=False)


def _world_context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50, -50, 0),
        scene_max_xyz_m=(50, 50, 30),
        initial_uav_xyz_m=(7.654321, -8.765432, 1.234567),
        search_regions={
            "search_area": SearchRegionSpec(
                "search_area",
                center_xyz_m=(12.345678, 23.456789, 2.345679),
                radius_m=14.567891,
                approach_xyz_m=(16.789123, 27.891234, 9.876543),
                description="north sector",
            )
        },
        landing_zones={
            "home": LandingZoneSpec(
                "home",
                position_xy_m=(31.415926, -27.182818),
                ground_altitude_m=0.123456,
                description="launch pad",
            )
        },
        navigation_points={
            "checkpoint": NavigationPointSpec(
                "checkpoint",
                position_xyz_m=(-11.111111, 22.222222, 8.888888),
                description="visual checkpoint",
            )
        },
        default_takeoff_altitude_m=10,
        default_track_duration_s=30,
        search_timeout_s=60,
    )


def _request() -> PlannerRequest:
    return PlannerRequest(
        "起飞后前往 search_area 搜索目标，跟踪十秒后返回 home 降落",
        _world_context(),
    )


class FakeModelClient:
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
        return ModelResponse(outcome, "fake-qwen", "stop", {})


class _ExplodingLogger:
    def __getattr__(self, name: str):
        def explode(message: str) -> None:
            raise RuntimeError(f"logger {name} failed")

        return explode


class DynamicLLMPlannerTest(unittest.TestCase):
    def _planner(
        self,
        outcomes: Sequence[str | Exception],
        *,
        logger: object | None = None,
    ) -> tuple[DynamicLLMPlanner, FakeModelClient]:
        client = FakeModelClient(outcomes)
        planner = DynamicLLMPlanner(
            client,
            PROMPT_PATH,
            build_default_skill_catalog(),
            LIMITS,
            logger,
        )
        return planner, client

    def test_valid_json_returns_draft_in_one_call(self) -> None:
        planner, client = self._planner([_draft_json()])

        result = planner.plan(_request())

        self.assertIsInstance(planner, MissionPlanner)
        self.assertEqual(planner.source, "dynamic_llm")
        self.assertIsInstance(result, SkillPlanDraft)
        self.assertNotIsInstance(result, TaskPlan)
        self.assertEqual(len(client.calls), 1)

    def test_single_json_fence_is_accepted(self) -> None:
        planner, client = self._planner([f"```json\n{_draft_json()}\n```"])
        self.assertEqual(len(planner.plan(_request()).steps), 6)
        self.assertEqual(len(client.calls), 1)

    def test_one_invalid_output_can_be_repaired_once(self) -> None:
        invalid = '{"schema_version":1,"steps":[]}'
        planner, client = self._planner([invalid, _draft_json()])

        result = planner.plan(_request())

        self.assertEqual(len(result.steps), 6)
        self.assertEqual(len(client.calls), 2)
        repair_payload = json.loads(str(client.calls[1][0][-1].content))
        self.assertEqual(repair_payload["original_output"], invalid)
        self.assertIn("between 2 and 10", repair_payload["validation_error"])

    def test_landing_precondition_error_uses_the_single_repair_call(self) -> None:
        invalid = _draft_dict()
        invalid["steps"].pop(4)  # type: ignore[union-attr]
        invalid_output = json.dumps(invalid, ensure_ascii=False)
        planner, client = self._planner([invalid_output, _draft_json()])

        result = planner.plan(_request())

        self.assertEqual(len(result.steps), 6)
        self.assertEqual(len(client.calls), 2)
        repair_payload = json.loads(str(client.calls[1][0][-1].content))
        self.assertIn(
            "LAND must be immediately preceded by GOTO to the same zone",
            repair_payload["validation_error"],
        )

    def test_two_invalid_outputs_fail_and_never_make_third_call(self) -> None:
        planner, client = self._planner(["not json", "[]", _draft_json()])
        with self.assertRaisesRegex(PlannerOutputError, "after one repair"):
            planner.plan(_request())
        self.assertEqual(len(client.calls), 2)

    def test_active_catalog_is_an_enforced_allow_list(self) -> None:
        catalog = build_default_skill_catalog()
        without_track = SkillCatalog(
            tuple(contract for contract in catalog if contract.name != "TRACK")
        )
        client = FakeModelClient([_draft_json(), _draft_json()])
        planner = DynamicLLMPlanner(
            client,
            PROMPT_PATH,
            skill_catalog=without_track,
            planner_limits=LIMITS,
        )
        with self.assertRaises(PlannerOutputError):
            planner.plan(_request())
        self.assertEqual(len(client.calls), 2)

    def test_active_catalog_enforces_value_contracts(self) -> None:
        cases: list[tuple[str, str, dict[str, object], dict[str, object]]] = [
            (
                "TRACK",
                "duration_s",
                {"maximum": 5.0},
                {},
            ),
            (
                "GOTO",
                "yaw_mode",
                {"allowed_values": ("KEEP_CURRENT",)},
                {"goto_yaw_mode": "COURSE_ALIGNED"},
            ),
            (
                "TRACK",
                "duration_s",
                {"value_type": "string"},
                {},
            ),
            (
                "REACQUIRE",
                "timeout_s",
                {"maximum": 20.0},
                {},
            ),
        ]
        for skill, argument_name, changes, output_changes in cases:
            with self.subTest(skill=skill, argument=argument_name, changes=changes):
                catalog = build_default_skill_catalog()
                contracts = []
                for contract in catalog:
                    arguments = tuple(
                        replace(argument, **changes)
                        if contract.name == skill and argument.name == argument_name
                        else argument
                        for argument in contract.arguments
                    )
                    contracts.append(replace(contract, arguments=arguments))
                restricted_catalog = SkillCatalog(tuple(contracts))

                output = _draft_dict()
                if "goto_yaw_mode" in output_changes:
                    output["steps"][1]["args"]["yaw_mode"] = output_changes[  # type: ignore[index]
                        "goto_yaw_mode"
                    ]
                raw_output = json.dumps(output, ensure_ascii=False)
                client = FakeModelClient([raw_output, raw_output])
                planner = DynamicLLMPlanner(
                    client,
                    PROMPT_PATH,
                    skill_catalog=restricted_catalog,
                    planner_limits=LIMITS,
                )

                with self.assertRaises(PlannerOutputError):
                    planner.plan(_request())
                self.assertEqual(len(client.calls), 2)

    def test_unsupported_catalog_condition_is_rejected_before_model_call(self) -> None:
        catalog = build_default_skill_catalog()
        contracts = []
        for contract in catalog:
            arguments = tuple(
                replace(argument, condition="only when instructed")
                if contract.name == "TRACK" and argument.name == "duration_s"
                else argument
                for argument in contract.arguments
            )
            contracts.append(replace(contract, arguments=arguments))
        client = FakeModelClient([_draft_json()])
        with self.assertRaisesRegex(ValueError, "unsupported catalog condition"):
            DynamicLLMPlanner(
                client,
                PROMPT_PATH,
                skill_catalog=SkillCatalog(tuple(contracts)),
                planner_limits=LIMITS,
            )
        self.assertEqual(client.calls, [])

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self) -> None:
        duplicate = (
            '{"schema_version":1,"schema_version":1,"steps":'
            + json.dumps(_draft_dict()["steps"])
            + "}"
        )
        nonfinite = _draft_json().replace('"altitude_m": 10', '"altitude_m": NaN', 1)
        for output, message in (
            (duplicate, "duplicate JSON field"),
            (nonfinite, "non-finite JSON number"),
        ):
            with self.subTest(message=message):
                planner, client = self._planner([output, output])
                with self.assertRaisesRegex(PlannerOutputError, message):
                    planner.plan(_request())
                self.assertEqual(len(client.calls), 2)

    def test_schema_failure_repair_preserves_specific_error(self) -> None:
        invalid = _draft_dict()
        invalid["steps"][1]["args"]["position"] = [1, 2, 3]  # type: ignore[index]
        output = json.dumps(invalid)
        planner, _ = self._planner([output, output])
        with self.assertRaisesRegex(PlannerOutputError, "unknown fields: position"):
            planner.plan(_request())

    def test_connection_error_propagates_without_repair(self) -> None:
        error = ModelConnectionError("offline")
        planner, client = self._planner([error])
        with self.assertRaises(ModelConnectionError) as raised:
            planner.plan(_request())
        self.assertIs(raised.exception, error)
        self.assertEqual(len(client.calls), 1)

    def test_temperature_is_zero_for_initial_and_repair_calls(self) -> None:
        planner, client = self._planner(["bad", _draft_json()])
        planner.plan(_request())
        self.assertEqual(len(client.calls), 2)
        for _, options in client.calls:
            self.assertIsInstance(options, GenerationOptions)
            self.assertEqual(options.temperature, 0.0)

    def test_runtime_messages_equal_shared_builder_byte_for_byte(self) -> None:
        planner, client = self._planner([_draft_json()])
        request = _request()
        planner.plan(request)
        expected = build_dynamic_skill_planner_messages(
            request.instruction,
            request.world_context,
            build_default_skill_catalog(),
            LIMITS,
            PROMPT_PATH.read_text(encoding="utf-8"),
        )
        self.assertEqual(client.calls[0][0], expected)
        self.assertEqual(
            tuple(str(item.content).encode("utf-8") for item in client.calls[0][0]),
            tuple(str(item.content).encode("utf-8") for item in expected),
        )

    def test_prompt_exposes_only_named_world_metadata_and_limits(self) -> None:
        messages = build_dynamic_skill_planner_messages(
            _request().instruction,
            _request().world_context,
            build_default_skill_catalog(),
            LIMITS,
            PROMPT_PATH.read_text(encoding="utf-8"),
        )
        payload = json.loads(str(messages[1].content))
        self.assertEqual(
            set(payload),
            {
                "task",
                "trusted_world_context",
                "skill_catalog",
                "planner_limits",
                "user_instruction",
            },
        )
        trusted = payload["trusted_world_context"]
        self.assertEqual(
            set(trusted),
            {
                "scene_bounds_m",
                "search_regions",
                "landing_zones",
                "navigation_points",
                "default_takeoff_altitude_m",
                "default_track_duration_s",
            },
        )
        self.assertEqual(trusted["navigation_points"][0]["name"], "checkpoint")
        serialized = str(messages[1].content)
        for secret in (
            "7.654321",
            "-8.765432",
            "1.234567",
            "12.345678",
            "23.456789",
            "2.345679",
            "14.567891",
            "16.789123",
            "27.891234",
            "9.876543",
            "31.415926",
            "-27.182818",
            "0.123456",
            "-11.111111",
            "22.222222",
            "8.888888",
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, serialized)
        for forbidden_key in (
            "initial_uav_xyz_m",
            "center_xyz_m",
            "approach_xyz_m",
            "position_xy_m",
            "position_xyz_m",
            "oracle_target",
            "camera_rgb",
        ):
            with self.subTest(forbidden_key=forbidden_key):
                self.assertNotIn(forbidden_key, serialized)

    def test_prompt_rejects_coordinate_or_low_level_public_descriptions(self) -> None:
        base = _world_context()
        for description in (
            "sector coordinates [12.3,23.4,0.0]",
            "sector PID kp=9",
            "目标坐标是 12.3,23.4,0.0",
        ):
            region = replace(
                base.search_regions["search_area"],
                description=description,
            )
            context = replace(
                base,
                search_regions={"search_area": region},
            )
            with self.subTest(description=description), self.assertRaises(ValueError):
                build_dynamic_skill_planner_messages(
                    "search safely",
                    context,
                    build_default_skill_catalog(),
                    LIMITS,
                    PROMPT_PATH.read_text(encoding="utf-8"),
                )

    def test_prompt_limits_cannot_exceed_v1_hard_caps(self) -> None:
        for field, value in (
            ("max_goto_calls", 6),
            ("max_track_calls", 3),
            ("max_reacquire_attempts_per_track", 3),
            ("max_total_reacquire_attempts", 5),
        ):
            limits = dict(LIMITS)
            limits[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_dynamic_skill_planner_messages(
                    _request().instruction,
                    _request().world_context,
                    build_default_skill_catalog(),
                    limits,
                    PROMPT_PATH.read_text(encoding="utf-8"),
                )

    def test_legacy_prompt_protocol_remains_unchanged(self) -> None:
        request = _request()
        messages = build_mission_planner_messages(
            request.instruction,
            request.world_context,
            LEGACY_PROMPT_PATH.read_text(encoding="utf-8"),
        )
        payload = json.loads(str(messages[1].content))
        self.assertEqual(
            set(payload),
            {"task", "trusted_world_context", "user_instruction"},
        )
        self.assertNotIn("skill_catalog", str(messages[1].content))
        self.assertNotIn("navigation_points", str(messages[1].content))

    def test_logger_failure_does_not_change_planning(self) -> None:
        planner, client = self._planner([_draft_json()], logger=_ExplodingLogger())
        self.assertEqual(len(planner.plan(_request()).steps), 6)
        self.assertEqual(len(client.calls), 1)

    def test_scripted_dynamic_planner_returns_fresh_drafts(self) -> None:
        source = SkillPlanDraft.from_dict(_draft_dict())
        planner = ScriptedDynamicPlanner(source)
        first = planner.plan(_request())
        second = planner.plan(_request())
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(planner.source, "dynamic_scripted")


if __name__ == "__main__":
    unittest.main()
