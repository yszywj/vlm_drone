"""Offline model-contract tests for the opt-in Spatial Planner V3 path."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from models.base import ChatMessage, GenerationOptions, ModelResponse
from planner.base import PlannerOutputError
from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.json_schema_v3 import (
    build_skill_plan_v3_json_schema,
    search_strategy_json_schema,
)
from planner.schemas import (
    LandingZoneSpec,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraftV2,
)
from planner.schemas_v3 import SkillPlanDraftV3
from planner.skill_catalog import (
    build_default_skill_catalog,
    build_spatial_v3_skill_catalog,
)
from planner.spatial import CoordinateFrame, RectangleRegion, SectorRegion
from scripts import run_planner_demo
from skills.search_strategy import SearchRuntimeCapabilities


_ROOT = Path(__file__).resolve().parents[1]
_V2_PROMPT = _ROOT / "prompts" / "dynamic_skill_planner_system.txt"
_V3_PROMPT = _ROOT / "prompts" / "dynamic_skill_planner_v3_system.txt"
_MISSION_ID = "mission_test"
_UAV_ID = "uav_1"
_SECTOR_INSTRUCTION = (
    "起飞到十米，搜索home北侧二十到五十米、左右各三十度的扇形区域，"
    "寻找一个移动目标，找到后跟踪十五秒，最后返回home降落"
)
_LEFT_INSTRUCTION = (
    "起飞后搜索任务开始时无人机左侧十到三十米范围内的一块矩形区域，"
    "找到移动目标后跟踪十秒并返航"
)


def _world_context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={
            "search_area": SearchRegionSpec(
                name="search_area",
                center_xyz_m=(20.0, 30.0, 0.0),
                radius_m=15.0,
                approach_xyz_m=(20.0, 12.0, 10.0),
                description="designated outdoor search area",
            )
        },
        landing_zones={
            "home": LandingZoneSpec(
                name="home",
                position_xy_m=(0.0, 0.0),
                ground_altitude_m=0.0,
                description="launch and recovery zone",
            )
        },
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=30.0,
        search_timeout_s=75.0,
    )


def _request(instruction: str = _SECTOR_INSTRUCTION) -> PlannerRequest:
    return PlannerRequest(
        instruction=instruction,
        world_context=_world_context(),
        mission_id=_MISSION_ID,
        uav_id=_UAV_ID,
        plan_version=1,
    )


def _target_spec() -> dict[str, object]:
    return {
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
    }


def _v2_plan() -> dict[str, object]:
    return {
        "schema_version": 2,
        "mission_id": _MISSION_ID,
        "uav_id": _UAV_ID,
        "plan_version": 1,
        "target_spec": _target_spec(),
        "steps": [
            {"id": "takeoff_1", "uav_id": _UAV_ID, "skill": "TAKEOFF", "args": {"altitude_m": 10}},
            {"id": "goto_search", "uav_id": _UAV_ID, "skill": "GOTO", "args": {"destination": "search_area"}},
            {"id": "search_1", "uav_id": _UAV_ID, "skill": "SEARCH", "args": {"region": "search_area", "target_description": "moving target"}},
            {"id": "track_1", "uav_id": _UAV_ID, "skill": "TRACK", "args": {"target_ref": "$search_1.target_id", "duration_s": 15}},
            {"id": "goto_home", "uav_id": _UAV_ID, "skill": "GOTO", "args": {"destination": "home"}},
            {"id": "land_1", "uav_id": _UAV_ID, "skill": "LAND", "args": {"zone": "home"}},
        ],
    }


def _sector_v3_plan(*, mission_id: str = _MISSION_ID, uav_id: str = _UAV_ID) -> dict[str, object]:
    return {
        "schema_version": 3,
        "mission_id": mission_id,
        "uav_id": uav_id,
        "plan_version": 1,
        "assumptions": [],
        "steps": [
            {"id": "takeoff_1", "uav_id": uav_id, "skill": "TAKEOFF", "args": {"altitude_m": 10}},
            {
                "id": "search_1",
                "uav_id": uav_id,
                "skill": "SEARCH",
                "args": {
                    "region": {
                        "shape": "SECTOR",
                        "frame": "HOME_ENU",
                        "origin_xyz_m": [0, 0, 0],
                        "azimuth_center_deg": 90,
                        "azimuth_span_deg": 60,
                        "distance_range_m": [20, 50],
                    },
                    "strategy": {"kind": "SECTOR_SWEEP", "spacing_m": 5, "max_viewpoints": 32},
                    "entry_policy": "START_IN_PLACE_IF_INSIDE",
                    "target_description": "moving target",
                    "search_altitude_m": 10,
                    "timeout_s": 75,
                },
            },
            {"id": "track_1", "uav_id": uav_id, "skill": "TRACK", "args": {"target_ref": "$search_1.target_id", "duration_s": 15}},
            {"id": "goto_home", "uav_id": uav_id, "skill": "GOTO", "args": {"target": {"kind": "NAMED_LOCATION", "name": "home"}}},
            {"id": "land_1", "uav_id": uav_id, "skill": "LAND", "args": {"zone": "home"}},
        ],
    }


def _left_rectangle_v3_plan() -> dict[str, object]:
    result = _sector_v3_plan()
    result["assumptions"] = [
        {
            "source_text": "任务开始时无人机左侧十到三十米",
            "interpretation": "UAV_START_FLU left (+y), with radial distance 10m to 30m",
            "confidence": 0.8,
        }
    ]
    search = result["steps"][1]  # type: ignore[index]
    search["args"]["region"] = {  # type: ignore[index]
        "shape": "RECTANGLE",
        "frame": "UAV_START_FLU",
        "center_xyz_m": [0, 20, 0],
        "width_m": 20,
        "height_m": 20,
        "yaw_deg": 0,
    }
    search["args"]["strategy"] = {  # type: ignore[index]
        "kind": "LAWNMOWER",
        "spacing_m": 5,
        "max_viewpoints": 32,
    }
    search["args"]["entry_policy"] = "NEAREST_POINT"  # type: ignore[index]
    search["args"]["timeout_s"] = 60  # type: ignore[index]
    result["steps"][2]["args"]["duration_s"] = 10  # type: ignore[index]
    return result


def _multi_search_v3_plan(
    *,
    search_timeouts_s: tuple[float, float] = (35.0, 40.0),
    extra_hover: bool = False,
) -> dict[str, object]:
    result = _sector_v3_plan()
    takeoff = deepcopy(result["steps"][0])  # type: ignore[index]
    first_search = deepcopy(result["steps"][1])  # type: ignore[index]
    second_search = deepcopy(first_search)
    first_search["args"]["timeout_s"] = search_timeouts_s[0]  # type: ignore[index]
    second_search["id"] = "search_2"
    second_search["args"]["timeout_s"] = search_timeouts_s[1]  # type: ignore[index]
    goto_home = deepcopy(result["steps"][-2])  # type: ignore[index]
    land = deepcopy(result["steps"][-1])  # type: ignore[index]
    hover = {
        "id": "hover_before_land",
        "uav_id": _UAV_ID,
        "skill": "HOVER",
        "args": {"duration_s": 5, "yaw_mode": "KEEP_CURRENT"},
    }
    steps: list[object] = [takeoff, first_search, second_search]
    if extra_hover:
        steps.append(
            {
                "id": "hover_mid_mission",
                "uav_id": _UAV_ID,
                "skill": "HOVER",
                "args": {"duration_s": 3},
            }
        )
    steps.extend((goto_home, hover, land))
    result["steps"] = steps
    return result


def _wait_only_v3_plan() -> dict[str, object]:
    return {
        "schema_version": 3,
        "mission_id": _MISSION_ID,
        "uav_id": _UAV_ID,
        "plan_version": 1,
        "assumptions": [],
        "steps": [
            {
                "id": "takeoff_1",
                "uav_id": _UAV_ID,
                "skill": "TAKEOFF",
                "args": {"altitude_m": 10},
            },
            {
                "id": "wait_1",
                "uav_id": _UAV_ID,
                "skill": "HOVER",
                "args": {"duration_s": 5},
            },
        ],
    }


class _FakeModelClient:
    def __init__(self, outcomes: Sequence[dict[str, object] | str]) -> None:
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
        content = outcome if isinstance(outcome, str) else json.dumps(outcome)
        return ModelResponse(content, "fake-qwen", "stop", {})


class _RoutingEchoV3Client:
    """Build a deterministic V3 reply using routing from the actual CLI prompt."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        del options
        payload = json.loads(messages[1].content)
        routing = payload["trusted_routing"]
        result = _sector_v3_plan(
            mission_id=routing["mission_id"],
            uav_id=routing["uav_id"],
        )
        result["plan_version"] = routing["plan_version"]
        return ModelResponse(json.dumps(result), "fake-qwen", "stop", {})


class DynamicLLMPlannerV3Test(unittest.TestCase):
    def test_default_contract_remains_byte_compatible_v2_path(self) -> None:
        client = _FakeModelClient([_v2_plan()])
        planner = DynamicLLMPlanner(client, _V2_PROMPT)

        result = planner.plan(_request())

        self.assertEqual(planner.planning_contract, "v2")
        self.assertIsInstance(result, SkillPlanDraftV2)
        messages, options = client.calls[0]
        self.assertEqual(options.response_format.name, "skill_plan_draft_v2")  # type: ignore[union-attr]
        self.assertEqual(options.response_format.schema["properties"]["schema_version"]["const"], 2)  # type: ignore[union-attr,index]
        self.assertEqual(messages[0].content, _V2_PROMPT.read_text(encoding="utf-8").strip())
        self.assertIn("不得输出任何原始 XYZ", messages[0].content)
        prompt_payload = json.loads(messages[1].content)
        self.assertEqual(prompt_payload["planner_limits"]["max_search_calls"], 1)

    def test_v3_sector_uses_independent_prompt_schema_and_parser(self) -> None:
        client = _FakeModelClient([_sector_v3_plan()])
        planner = DynamicLLMPlanner(
            client,
            _V3_PROMPT,
            planning_contract="v3",
        )

        result = planner.plan(_request())

        self.assertIsInstance(result, SkillPlanDraftV3)
        self.assertEqual(planner.planning_contract, "v3")
        region = result.steps[1].region
        self.assertIsInstance(region, SectorRegion)
        self.assertEqual(region.frame, CoordinateFrame.HOME_ENU)
        self.assertEqual(region.distance_range_m, (20.0, 50.0))
        self.assertEqual(region.azimuth_span_deg, 60.0)
        self.assertEqual(result.steps[2].args["duration_s"], 15.0)
        self.assertEqual(len(planner.model_proposals), 1)
        self.assertTrue(planner.model_proposals[0]["accepted"])
        self.assertEqual(
            planner.model_proposals[0]["raw_proposal"]["schema_version"],
            3,
        )

        messages, options = client.calls[0]
        response_format = options.response_format  # type: ignore[union-attr]
        self.assertEqual(response_format.name, "skill_plan_draft_v3")
        schema = response_format.schema
        self.assertEqual(schema["properties"]["schema_version"]["const"], 3)  # type: ignore[index]
        self.assertEqual(schema["properties"]["mission_id"]["const"], _MISSION_ID)  # type: ignore[index]
        self.assertEqual(schema["properties"]["uav_id"]["const"], _UAV_ID)  # type: ignore[index]
        prompt_payload = json.loads(messages[1].content)
        self.assertEqual(prompt_payload["user_instruction"], _SECTOR_INSTRUCTION)
        self.assertTrue(prompt_payload["spatial_output_policy"]["framed_coordinates_allowed"])
        self.assertTrue(
            prompt_payload["mission_completeness_contract"][
                "emit_the_complete_mission_not_a_prefix"
            ]
        )
        goto = next(
            item
            for item in prompt_payload["skill_catalog"]["skills"]
            if item["name"] == "GOTO"
        )
        argument_names = {item["name"] for item in goto["arguments"]}
        self.assertIn("target", argument_names)
        self.assertNotIn("destination", argument_names)
        self.assertLess(len(messages[1].content), 5000)
        self.assertNotIn("不得输出任何原始 XYZ", messages[0].content)
        self.assertNotIn("Do not add coordinates", messages[0].content)
        self.assertFalse(
            prompt_payload["runtime_search_capabilities"][
                "adaptive_next_best_view"
            ]
        )
        self.assertNotIn("ADAPTIVE_NEXT_BEST_VIEW", json.dumps(schema))

    def test_fleet_goal_mode_allows_no_track_and_defers_unrequested_land(self) -> None:
        client = _FakeModelClient([_wait_only_v3_plan()])
        planner = DynamicLLMPlanner(
            client,
            _V3_PROMPT,
            planning_contract="v3",
            repair_budget=0,
        )
        request = PlannerRequest(
            instruction="wait for five seconds",
            world_context=_world_context(),
            mission_id=_MISSION_ID,
            uav_id=_UAV_ID,
            plan_version=1,
            allow_trusted_safety_completion=True,
        )

        result = planner.plan(request)

        self.assertEqual(
            tuple(step.skill for step in result.steps),
            ("TAKEOFF", "HOVER"),
        )
        self.assertNotIn("TRACK", tuple(step.skill for step in result.steps))
        prompt_payload = json.loads(client.calls[0][0][1].content)
        completeness = prompt_payload["mission_completeness_contract"]
        self.assertTrue(completeness["trusted_runtime_safety_completion"])
        self.assertTrue(completeness["omit_unrequested_return_or_land"])
        self.assertEqual(planner.repair_budget, 0)

    def test_repair_budget_zero_returns_outer_repair_diagnostics_after_one_call(self) -> None:
        invalid = deepcopy(_sector_v3_plan())
        del invalid["steps"][1]["args"]["region"]["frame"]  # type: ignore[index]
        client = _FakeModelClient([invalid])
        planner = DynamicLLMPlanner(
            client,
            _V3_PROMPT,
            planning_contract="v3",
            repair_budget=0,
        )

        with self.assertRaisesRegex(PlannerOutputError, "internal repair is disabled"):
            planner.plan(_request())

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(planner.last_diagnostics.model_calls, 1)  # type: ignore[union-attr]
        self.assertFalse(planner.last_diagnostics.repair_used)  # type: ignore[union-attr]
        self.assertEqual(planner.last_diagnostics.initial_error_code, "SCHEMA_INVALID")  # type: ignore[union-attr]
        self.assertTrue(planner.last_diagnostics.initial_error_message)  # type: ignore[union-attr]

        with self.assertRaisesRegex(ValueError, "repair_budget"):
            DynamicLLMPlanner(
                _FakeModelClient([]),
                _V3_PROMPT,
                planning_contract="v3",
                repair_budget=2,
            )

    def test_v3_adaptive_strategy_is_capability_negotiated_everywhere(self) -> None:
        disabled_schema = search_strategy_json_schema()
        self.assertNotIn("ADAPTIVE_NEXT_BEST_VIEW", json.dumps(disabled_schema))
        disabled_search = build_spatial_v3_skill_catalog().get("SEARCH")
        disabled_description = next(
            item.description
            for item in disabled_search.arguments
            if item.name == "strategy"
        )
        self.assertNotIn("ADAPTIVE_NEXT_BEST_VIEW", disabled_description)

        capabilities = SearchRuntimeCapabilities(adaptive_next_best_view=True)
        enabled_schema = search_strategy_json_schema(
            search_runtime_capabilities=capabilities
        )
        self.assertIn("ADAPTIVE_NEXT_BEST_VIEW", json.dumps(enabled_schema))
        enabled_search = build_spatial_v3_skill_catalog(capabilities).get("SEARCH")
        enabled_description = next(
            item.description
            for item in enabled_search.arguments
            if item.name == "strategy"
        )
        self.assertIn("ADAPTIVE_NEXT_BEST_VIEW", enabled_description)

        client = _FakeModelClient([_sector_v3_plan()])
        planner = DynamicLLMPlanner(
            client,
            _V3_PROMPT,
            planning_contract="v3",
            search_runtime_capabilities=capabilities,
        )
        planner.plan(_request())
        messages, options = client.calls[0]
        payload = json.loads(messages[1].content)
        self.assertTrue(
            payload["runtime_search_capabilities"][
                "adaptive_next_best_view"
            ]
        )
        self.assertIn(
            "ADAPTIVE_NEXT_BEST_VIEW",
            json.dumps(options.response_format.schema),  # type: ignore[union-attr]
        )

    def test_v3_schema_has_exactly_one_hover_step_variant(self) -> None:
        schema = build_skill_plan_v3_json_schema(
            mission_id=_MISSION_ID,
            uav_id=_UAV_ID,
            plan_version=1,
        )
        variants = schema["properties"]["steps"]["items"]["oneOf"]  # type: ignore[index]
        hover_variants = [
            variant
            for variant in variants
            if variant["properties"]["skill"].get("const") == "HOVER"
        ]
        self.assertEqual(len(hover_variants), 1)
        self.assertNotIn("allOf", schema["properties"]["steps"])  # type: ignore[index]
        self.assertNotIn("contains", schema["properties"]["steps"])  # type: ignore[index]

    def test_v3_adaptive_output_is_rejected_without_negotiated_provider(self) -> None:
        adaptive = deepcopy(_sector_v3_plan())
        adaptive["steps"][1]["args"]["strategy"] = {  # type: ignore[index]
            "kind": "ADAPTIVE_NEXT_BEST_VIEW",
            "max_viewpoints": 4,
        }
        disabled_client = _FakeModelClient([adaptive, adaptive])
        disabled = DynamicLLMPlanner(
            disabled_client,
            _V3_PROMPT,
            planning_contract="v3",
        )
        with self.assertRaises(PlannerOutputError):
            disabled.plan(_request())
        self.assertIn(
            "unavailable",
            disabled.last_diagnostics.initial_error_message,  # type: ignore[union-attr]
        )

        enabled_client = _FakeModelClient([adaptive])
        enabled = DynamicLLMPlanner(
            enabled_client,
            _V3_PROMPT,
            planning_contract="v3",
            search_runtime_capabilities=SearchRuntimeCapabilities(
                adaptive_next_best_view=True
            ),
        )
        result = enabled.plan(_request())
        self.assertEqual(
            result.steps[1].args["strategy"].kind.value,
            "ADAPTIVE_NEXT_BEST_VIEW",
        )

    def test_v3_left_rectangle_preserves_reference_assumption(self) -> None:
        client = _FakeModelClient([_left_rectangle_v3_plan()])
        planner = DynamicLLMPlanner(client, _V3_PROMPT, planning_contract="v3")

        result = planner.plan(_request(_LEFT_INSTRUCTION))

        self.assertIsInstance(result.steps[1].region, RectangleRegion)
        self.assertEqual(result.steps[1].region.frame, CoordinateFrame.UAV_START_FLU)
        self.assertEqual(len(result.assumptions), 1)
        self.assertIn("UAV_START_FLU", result.assumptions[0].interpretation)
        self.assertEqual(result.steps[2].args["duration_s"], 10.0)
        prompt_payload = json.loads(client.calls[0][0][1].content)
        self.assertEqual(prompt_payload["user_instruction"], _LEFT_INSTRUCTION)

    def test_v3_missing_relative_direction_assumption_uses_bounded_repair(self) -> None:
        missing = deepcopy(_left_rectangle_v3_plan())
        missing["assumptions"] = []
        client = _FakeModelClient([missing, _left_rectangle_v3_plan()])
        planner = DynamicLLMPlanner(client, _V3_PROMPT, planning_contract="v3")

        execution = planner.plan_with_diagnostics(_request(_LEFT_INSTRUCTION))

        self.assertIsInstance(execution.output, SkillPlanDraftV3)
        self.assertTrue(execution.diagnostics.repair_used)
        self.assertTrue(execution.diagnostics.repair_succeeded)
        self.assertIn(
            "ambiguous relative direction",
            execution.diagnostics.initial_error_message,
        )
        self.assertEqual(len(client.calls), 2)

        failed_client = _FakeModelClient([missing, missing])
        failed_planner = DynamicLLMPlanner(
            failed_client,
            _V3_PROMPT,
            planning_contract="v3",
        )
        with self.assertRaises(PlannerOutputError):
            failed_planner.plan(_request(_LEFT_INSTRUCTION))
        self.assertEqual(
            failed_planner.last_diagnostics.initial_error_code,  # type: ignore[union-attr]
            "V3_CONTRACT_VIOLATION",
        )
        self.assertEqual(len(failed_client.calls), 2)

    def test_v3_incompatible_search_strategy_is_repaired_before_runtime(self) -> None:
        incompatible = deepcopy(_sector_v3_plan())
        incompatible["steps"][1]["args"]["strategy"] = {  # type: ignore[index]
            "kind": "PERIMETER_V1",
            "spacing_m": 5,
            "max_viewpoints": 32,
        }
        client = _FakeModelClient([incompatible, _sector_v3_plan()])
        planner = DynamicLLMPlanner(client, _V3_PROMPT, planning_contract="v3")

        execution = planner.plan_with_diagnostics(_request())

        self.assertTrue(execution.diagnostics.repair_used)
        self.assertTrue(execution.diagnostics.repair_succeeded)
        self.assertIn(
            "requires a CIRCLE region",
            execution.diagnostics.initial_error_message,
        )
        repair_payload = json.loads(client.calls[1][0][-1].content)
        self.assertTrue(
            any(
                "SECTOR_SWEEP only with SECTOR" in item
                for item in repair_payload["mandatory_repairs"]
            )
        )
        self.assertEqual(
            execution.output.steps[1].args["strategy"].kind.value,  # type: ignore[union-attr]
            "SECTOR_SWEEP",
        )

    def test_v3_ground_level_world_point_is_repaired_before_runtime(self) -> None:
        invalid = deepcopy(_sector_v3_plan())
        invalid["steps"].insert(  # type: ignore[union-attr]
            1,
            {
                "id": "goto_ground_center",
                "uav_id": _UAV_ID,
                "skill": "GOTO",
                "args": {
                    "target": {
                        "kind": "POINT",
                        "frame": "WORLD_ENU",
                        "xyz_m": [20, 30, 0],
                    }
                },
            },
        )
        client = _FakeModelClient([invalid, _sector_v3_plan()])
        planner = DynamicLLMPlanner(client, _V3_PROMPT, planning_contract="v3")

        execution = planner.plan_with_diagnostics(_request())

        self.assertTrue(execution.diagnostics.repair_used)
        self.assertTrue(execution.diagnostics.repair_succeeded)
        self.assertIn(
            "WORLD_ENU POINT target z must be greater than zero",
            execution.diagnostics.initial_error_message,
        )
        repair_payload = json.loads(client.calls[1][0][-1].content)
        self.assertTrue(
            any(
                "positive flight altitude" in item
                for item in repair_payload["mandatory_repairs"]
            )
        )

    def test_v3_incomplete_prefix_repair_must_expand_complete_mission(self) -> None:
        incomplete = deepcopy(_sector_v3_plan())
        incomplete["steps"] = incomplete["steps"][:2]  # type: ignore[index]
        client = _FakeModelClient([incomplete, _sector_v3_plan()])
        planner = DynamicLLMPlanner(client, _V3_PROMPT, planning_contract="v3")

        execution = planner.plan_with_diagnostics(_request())

        self.assertTrue(execution.diagnostics.repair_succeeded)
        repair_messages = client.calls[1][0]
        self.assertIn("do not repeat it unchanged", repair_messages[0].content)
        repair_payload = json.loads(repair_messages[-1].content)
        self.assertTrue(repair_payload["previous_output_was_rejected"])
        self.assertTrue(repair_payload["must_change_rejected_output"])
        self.assertTrue(repair_payload["steps_must_cover_entire_user_instruction"])
        self.assertTrue(
            any(
                "incomplete prefix" in item
                for item in repair_payload["mandatory_repairs"]
            )
        )
        self.assertEqual(execution.output.steps[-1].skill, "LAND")

    def test_v3_assumption_must_quote_instruction_and_name_explicit_frame(self) -> None:
        fabricated_source = deepcopy(_left_rectangle_v3_plan())
        fabricated_source["assumptions"][0]["source_text"] = "无人机右侧"  # type: ignore[index]
        implicit_frame = deepcopy(_left_rectangle_v3_plan())
        implicit_frame["assumptions"][0]["interpretation"] = (  # type: ignore[index]
            "the model interprets this as the UAV's left side"
        )

        for invalid, message in (
            (fabricated_source, "exact substring"),
            (implicit_frame, "explicit coordinate frame"),
        ):
            with self.subTest(message=message):
                client = _FakeModelClient([invalid, invalid])
                planner = DynamicLLMPlanner(
                    client,
                    _V3_PROMPT,
                    planning_contract="v3",
                )
                with self.assertRaises(PlannerOutputError):
                    planner.plan(_request(_LEFT_INSTRUCTION))
                self.assertIn(
                    message,
                    planner.last_diagnostics.initial_error_message,  # type: ignore[union-attr]
                )
                self.assertEqual(len(client.calls), 2)

    def test_v3_non_spatial_assumption_does_not_require_coordinate_frame(self) -> None:
        draft = deepcopy(_left_rectangle_v3_plan())
        draft["assumptions"].append(  # type: ignore[union-attr]
            {
                "source_text": "跟踪十秒",
                "interpretation": "TRACK duration_s=10",
                "confidence": 0.95,
            }
        )
        client = _FakeModelClient([draft])
        planner = DynamicLLMPlanner(
            client,
            _V3_PROMPT,
            planning_contract="v3",
        )

        result = planner.plan(_request(_LEFT_INSTRUCTION))

        self.assertEqual(len(result.assumptions), 2)
        self.assertEqual(len(client.calls), 1)

    def test_v3_explicit_frame_and_symmetric_sector_angle_need_no_assumption(self) -> None:
        explicitly_framed = deepcopy(_left_rectangle_v3_plan())
        explicitly_framed["assumptions"] = []
        explicitly_framed["steps"][1]["args"]["region"]["frame"] = "HOME_ENU"  # type: ignore[index]
        explicit_instruction = (
            "起飞后在 HOME_ENU 左侧搜索一块矩形区域，找到目标后返航降落"
        )
        explicit_planner = DynamicLLMPlanner(
            _FakeModelClient([explicitly_framed]),
            _V3_PROMPT,
            planning_contract="v3",
        )
        explicit_result = explicit_planner.plan(_request(explicit_instruction))
        self.assertEqual(explicit_result.assumptions, ())

        symmetric_instruction = (
            "起飞后搜索home北侧二十到五十米、左侧和右侧各三十度的扇形区域，"
            "找到移动目标后跟踪十五秒，最后返回home降落"
        )
        symmetric_planner = DynamicLLMPlanner(
            _FakeModelClient([_sector_v3_plan()]),
            _V3_PROMPT,
            planning_contract="v3",
        )
        symmetric_result = symmetric_planner.plan(
            _request(symmetric_instruction)
        )
        self.assertEqual(symmetric_result.assumptions, ())

        temporal_instruction = (
            "起飞后方可搜索home北侧二十到五十米的扇形区域，"
            "找到移动目标后跟踪十五秒，最后返回home降落"
        )
        temporal_planner = DynamicLLMPlanner(
            _FakeModelClient([_sector_v3_plan()]),
            _V3_PROMPT,
            planning_contract="v3",
        )
        temporal_result = temporal_planner.plan(_request(temporal_instruction))
        self.assertEqual(temporal_result.assumptions, ())

    def test_v3_allows_multiple_searches_and_goto_hover_land_with_shared_budgets(self) -> None:
        plan = _multi_search_v3_plan()
        client = _FakeModelClient([plan])
        planner = DynamicLLMPlanner(
            client,
            _V3_PROMPT,
            planner_limits={"max_plan_steps": 6},
            planning_contract="v3",
        )

        result = planner.plan(_request())

        self.assertEqual(
            tuple(step.skill for step in result.steps),
            ("TAKEOFF", "SEARCH", "SEARCH", "GOTO", "HOVER", "LAND"),
        )
        prompt_payload = json.loads(client.calls[0][0][1].content)
        self.assertEqual(prompt_payload["planner_limits"]["max_search_calls"], 4)
        self.assertEqual(
            prompt_payload["trusted_world_context"]["total_search_time_budget_s"],
            75.0,
        )

    def test_v3_repeated_search_is_bounded_by_total_steps_and_search_time(self) -> None:
        over_time = _multi_search_v3_plan(search_timeouts_s=(40.0, 40.0))
        time_client = _FakeModelClient([over_time, over_time])
        time_planner = DynamicLLMPlanner(
            time_client,
            _V3_PROMPT,
            planner_limits={"max_plan_steps": 6},
            planning_contract="v3",
        )
        with self.assertRaises(PlannerOutputError):
            time_planner.plan(_request())
        self.assertEqual(len(time_client.calls), 2)
        self.assertIn(
            "total SEARCH time budget",
            time_planner.last_diagnostics.initial_error_message,  # type: ignore[union-attr]
        )

        over_steps = _multi_search_v3_plan(extra_hover=True)
        step_client = _FakeModelClient([over_steps, over_steps])
        step_planner = DynamicLLMPlanner(
            step_client,
            _V3_PROMPT,
            planner_limits={"max_plan_steps": 6},
            planning_contract="v3",
        )
        with self.assertRaises(PlannerOutputError):
            step_planner.plan(_request())
        self.assertEqual(len(step_client.calls), 2)
        self.assertIn(
            "max_plan_steps",
            step_planner.last_diagnostics.initial_error_message,  # type: ignore[union-attr]
        )

    def test_v3_goto_schema_and_parser_reject_route_target_for_follow_route(self) -> None:
        schema = build_skill_plan_v3_json_schema(
            mission_id=_MISSION_ID,
            uav_id=_UAV_ID,
            plan_version=1,
        )
        variants = schema["properties"]["steps"]["items"]["oneOf"]  # type: ignore[index]
        goto_schema = next(
            item
            for item in variants
            if item["properties"]["skill"]["const"] == "GOTO"
        )
        target_variants = goto_schema["properties"]["args"]["properties"]["target"]["oneOf"]
        target_kinds = {
            item["properties"]["kind"]["const"] for item in target_variants
        }
        self.assertEqual(
            target_kinds,
            {"NAMED_LOCATION", "POINT", "RELATIONAL_POINT"},
        )

        route_plan = _multi_search_v3_plan()
        route_plan["steps"][-3]["args"]["target"] = {  # type: ignore[index]
            "kind": "ROUTE",
            "frame": "WORLD_ENU",
            "waypoints_xyz_m": [[0, 0, 10], [0, 0, 0]],
        }
        raw = json.dumps(route_plan)
        with self.assertRaisesRegex(ValueError, "use FOLLOW_ROUTE.*trusted route_ref"):
            DynamicLLMPlanner._parse_plan_draft_v3(raw)

        client = _FakeModelClient([route_plan, route_plan])
        planner = DynamicLLMPlanner(client, _V3_PROMPT, planning_contract="v3")
        with self.assertRaises(PlannerOutputError):
            planner.plan(_request())
        self.assertEqual(planner.last_diagnostics.initial_error_code, "SCHEMA_INVALID")  # type: ignore[union-attr]
        self.assertIn("FOLLOW_ROUTE", planner.last_diagnostics.initial_error_message)  # type: ignore[union-attr]
        prompt_payload = json.loads(client.calls[0][0][1].content)
        goto_contract = next(
            item
            for item in prompt_payload["skill_catalog"]["skills"]
            if item["name"] == "GOTO"
        )
        self.assertIn("FOLLOW_ROUTE", goto_contract["description"])

    def test_v3_repair_reuses_bound_schema_and_allows_framed_geometry(self) -> None:
        invalid = deepcopy(_sector_v3_plan())
        del invalid["steps"][1]["args"]["region"]["frame"]  # type: ignore[index]
        client = _FakeModelClient([invalid, _sector_v3_plan()])
        planner = DynamicLLMPlanner(client, _V3_PROMPT, planning_contract="v3")

        execution = planner.plan_with_diagnostics(_request())

        self.assertIsInstance(execution.output, SkillPlanDraftV3)
        self.assertTrue(execution.diagnostics.repair_used)
        self.assertTrue(execution.diagnostics.repair_succeeded)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(planner.model_proposals), 2)
        self.assertFalse(planner.model_proposals[0]["accepted"])
        self.assertTrue(planner.model_proposals[1]["accepted"])
        self.assertTrue(planner.model_proposals[1]["repair"])
        first_format = client.calls[0][1].response_format  # type: ignore[union-attr]
        second_format = client.calls[1][1].response_format  # type: ignore[union-attr]
        self.assertEqual(first_format, second_format)
        self.assertEqual(first_format.name, "skill_plan_draft_v3")
        repair_payload = json.loads(client.calls[1][0][-1].content)
        self.assertIn("schema-v3", repair_payload["task"])
        self.assertIn("Framed V3 coordinates", repair_payload["requirements"])
        self.assertNotIn("Do not add coordinates", repair_payload["requirements"])

    def test_v3_rejects_v2_catalog_before_calling_model(self) -> None:
        client = _FakeModelClient([])

        with self.assertRaisesRegex(ValueError, "Spatial V3 GOTO catalog"):
            DynamicLLMPlanner(
                client,
                _V3_PROMPT,
                build_default_skill_catalog(),
                planning_contract="v3",
            )

        self.assertEqual(client.calls, [])

    def test_cli_default_is_v2_and_explicit_v3_reaches_planner(self) -> None:
        parser = run_planner_demo._build_parser()
        self.assertEqual(parser.parse_args([]).planning_contract, "v2")
        self.assertEqual(
            parser.parse_args(["--planning-contract", "v3"]).planning_contract,
            "v3",
        )

        stdout = io.StringIO()
        with patch(
            "models.openai_compatible_client.OpenAICompatibleClient",
            return_value=_RoutingEchoV3Client(),
        ), redirect_stdout(stdout):
            exit_code = run_planner_demo.main(
                [
                    "--planner", "dynamic_llm",
                    "--planning-contract", "v3",
                    "--instruction", _SECTOR_INSTRUCTION,
                    "--json-output",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["planner_output_type"], "SkillPlanDraftV3")
        self.assertEqual(payload["skill_plan_draft"]["schema_version"], 3)
        self.assertEqual(payload["skill_plan_draft"]["steps"][1]["args"]["region"]["shape"], "SECTOR")
        self.assertIsNone(payload["compiled_task_plan"])


if __name__ == "__main__":
    unittest.main()
