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
from planner.base import MissionPlanner, PlannerError, PlannerOutputError
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
    SkillPlanDraftV2,
    migrate_plan_v1_to_v2,
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


MISSION_ID = "mission_test"
UAV_ID = "uav_1"
PLAN_VERSION = 1


def _v1_draft_dict() -> dict[str, object]:
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


def _draft_dict() -> dict[str, object]:
    legacy = _v1_draft_dict()
    steps = []
    for raw_step in legacy["steps"]:
        step = dict(raw_step)
        step["uav_id"] = UAV_ID
        steps.append(step)
    return {
        "schema_version": 2,
        "mission_id": MISSION_ID,
        "uav_id": UAV_ID,
        "plan_version": PLAN_VERSION,
        "target_spec": {
            "original_description": "moving target",
            "category": "unspecified",
            "hard_attributes": [],
            "soft_attributes": [],
            "negative_constraints": [],
            "relation_constraints": [],
            "query_ladder": [],
            "inspection_questions": [],
            "immutable_identity_summary": "moving target",
            "mutable_appearance_notes": [],
        },
        "steps": steps,
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
        mission_id=MISSION_ID,
        uav_id=UAV_ID,
        plan_version=PLAN_VERSION,
    )


def _unrouted_request() -> PlannerRequest:
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
        self.assertIsInstance(result, SkillPlanDraftV2)
        self.assertNotIsInstance(result, TaskPlan)
        self.assertEqual(len(client.calls), 1)

    def test_plan_with_diagnostics_reports_structured_direct_success(self) -> None:
        planner, client = self._planner([_draft_json()])

        execution = planner.plan_with_diagnostics(_request())

        self.assertIsInstance(execution.output, SkillPlanDraftV2)
        self.assertEqual(execution.diagnostics.model_calls, 1)
        self.assertFalse(execution.diagnostics.repair_used)
        self.assertTrue(execution.diagnostics.initial_output_valid)
        self.assertTrue(execution.diagnostics.final_output_valid)
        self.assertTrue(execution.diagnostics.structured_output_enabled)
        self.assertIsNotNone(client.calls[0][1].response_format)
        self.assertNotIn("original_output", execution.diagnostics.to_dict())

    def test_repair_diagnostics_use_stable_symbolic_code_and_same_schema(self) -> None:
        invalid = _draft_dict()
        invalid["steps"].pop(4)  # type: ignore[union-attr]
        planner, client = self._planner(
            [json.dumps(invalid, ensure_ascii=False), _draft_json()]
        )

        execution = planner.plan_with_diagnostics(_request())

        diagnostics = execution.diagnostics
        self.assertEqual(diagnostics.model_calls, 2)
        self.assertTrue(diagnostics.repair_used)
        self.assertTrue(diagnostics.repair_succeeded)
        self.assertEqual(diagnostics.initial_error_code, "LAND_GOTO_MISSING")
        self.assertEqual(
            client.calls[0][1].response_format,
            client.calls[1][1].response_format,
        )

    def test_failed_repair_keeps_sanitized_last_diagnostics(self) -> None:
        planner, _client = self._planner(["not json", "[]"])
        with self.assertRaises(PlannerOutputError):
            planner.plan_with_diagnostics(_request())
        diagnostics = planner.last_diagnostics
        self.assertIsNotNone(diagnostics)
        self.assertEqual(diagnostics.model_calls, 2)
        self.assertEqual(diagnostics.initial_error_code, "INVALID_JSON")
        self.assertFalse(diagnostics.final_output_valid)
        self.assertNotIn("not json", json.dumps(diagnostics.to_dict()))

    def test_single_json_fence_is_accepted(self) -> None:
        planner, client = self._planner([f"```json\n{_draft_json()}\n```"])
        self.assertEqual(len(planner.plan(_request()).steps), 6)
        self.assertEqual(len(client.calls), 1)

    def test_one_invalid_output_can_be_repaired_once(self) -> None:
        invalid_value = _draft_dict()
        invalid_value["steps"] = []
        invalid = json.dumps(invalid_value, ensure_ascii=False)
        planner, client = self._planner([invalid, _draft_json()])

        result = planner.plan(_request())

        self.assertEqual(len(result.steps), 6)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            [message.role for message in client.calls[1][0]],
            ["system", "user", "user"],
        )
        self.assertLess(len(str(client.calls[1][0][0].content)), 800)
        self.assertIn(
            "response schema",
            str(client.calls[1][0][0].content),
        )
        compact_context = json.loads(str(client.calls[1][0][1].content))
        self.assertEqual(
            compact_context["user_instruction"],
            _request().instruction,
        )
        self.assertEqual(
            compact_context["trusted_routing"]["mission_id"],
            MISSION_ID,
        )
        self.assertNotIn("skill_catalog", compact_context)
        repair_payload = json.loads(str(client.calls[1][0][-1].content))
        self.assertEqual(repair_payload["original_output"], invalid)
        self.assertEqual(
            repair_payload["validation_issues"][0]["code"],
            "SCHEMA_INVALID",
        )
        self.assertIn(
            "between 2 and 10",
            repair_payload["validation_issues"][0]["message"],
        )

    def test_landing_precondition_error_uses_the_single_repair_call(self) -> None:
        invalid = _draft_dict()
        invalid["steps"].pop(4)  # type: ignore[union-attr]
        invalid_output = json.dumps(invalid, ensure_ascii=False)
        planner, client = self._planner([invalid_output, _draft_json()])

        result = planner.plan(_request())

        self.assertEqual(len(result.steps), 6)
        self.assertEqual(len(client.calls), 2)
        repair_payload = json.loads(str(client.calls[1][0][-1].content))
        self.assertEqual(
            repair_payload["validation_issues"][0]["code"],
            "LAND_GOTO_MISSING",
        )
        self.assertIn(
            "LAND must be preceded by matching GOTO",
            repair_payload["validation_issues"][0]["message"],
        )
        self.assertIn(
            "Insert exactly one GOTO immediately before the final LAND",
            repair_payload["mandatory_repairs"][0],
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

    def test_initial_plan_cannot_invent_inspect_candidate_id(self) -> None:
        output = _draft_dict()
        output["steps"].insert(  # type: ignore[union-attr]
            3,
            {
                "id": "inspect_1",
                "uav_id": UAV_ID,
                "skill": "INSPECT",
                "args": {"candidate_id": "candidate_hallucinated"},
            },
        )
        raw = json.dumps(output, ensure_ascii=False)
        planner, client = self._planner([raw, raw])

        with self.assertRaisesRegex(
            PlannerOutputError,
            "INSPECT is unavailable in an initial plan",
        ):
            planner.plan(_request())
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            planner.last_diagnostics.initial_error_code,  # type: ignore[union-attr]
            "SCHEMA_INVALID",
        )

        schema = client.calls[0][1].response_format.schema  # type: ignore[union-attr]
        variants = schema["properties"]["steps"]["items"]["oneOf"]
        self.assertNotIn(
            "INSPECT",
            {variant["properties"]["skill"]["const"] for variant in variants},
        )
        prompt = json.loads(str(client.calls[0][0][1].content))
        self.assertNotIn(
            "INSPECT",
            {item["name"] for item in prompt["skill_catalog"]["skills"]},
        )
        with self.assertRaisesRegex(ValueError, "INSPECT is unavailable"):
            SkillPlanDraftV2.from_dict(output)
        with self.assertRaisesRegex(ValueError, "INSPECT is unavailable"):
            DynamicLLMPlanner._parse_plan_draft_v2(raw)

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
            '{"schema_version":2,"schema_version":2,"steps":'
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
        self.assertEqual(
            [options.max_tokens for _, options in client.calls],
            [1024, 1536],
        )

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
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            plan_version=request.plan_version,
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
            mission_id=MISSION_ID,
            uav_id=UAV_ID,
            plan_version=PLAN_VERSION,
        )
        payload = json.loads(str(messages[1].content))
        self.assertEqual(
            set(payload),
            {
                "task",
                "trusted_world_context",
                "skill_catalog",
                "planner_limits",
                "trusted_planner_policy",
                "trusted_routing",
                "user_instruction",
            },
        )
        self.assertEqual(
            payload["trusted_planner_policy"],
            {
                "default_on_target_lost": "REACQUIRE",
                "allowed_on_target_lost": ["REACQUIRE", "FAIL"],
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

    def test_unrouted_request_is_rejected_before_model_call(self) -> None:
        planner, client = self._planner([_draft_json()])

        with self.assertRaisesRegex(PlannerError, "trusted mission_id"):
            planner.plan(_unrouted_request())

        self.assertEqual(client.calls, [])
        self.assertIsNotNone(planner.last_diagnostics)
        self.assertEqual(planner.last_diagnostics.model_calls, 0)
        self.assertEqual(
            planner.last_diagnostics.initial_error_code,
            "ROUTING_IDS_REQUIRED",
        )

    def test_scripted_dynamic_planner_returns_fresh_drafts(self) -> None:
        source = SkillPlanDraft.from_dict(_v1_draft_dict())
        planner = ScriptedDynamicPlanner(source)
        first = planner.plan(_request())
        second = planner.plan(_request())
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(
            first,
            migrate_plan_v1_to_v2(
                source,
                mission_id=MISSION_ID,
                uav_id=UAV_ID,
                plan_version=PLAN_VERSION,
            ),
        )
        self.assertEqual(planner.source, "dynamic_scripted")
        execution = planner.plan_with_diagnostics(_request())
        self.assertEqual(execution.output, first)
        self.assertEqual(execution.diagnostics.model_calls, 0)
        self.assertFalse(execution.diagnostics.structured_output_enabled)


if __name__ == "__main__":
    unittest.main()
