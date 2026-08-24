"""Regression tests for conservative dynamic-scripted target binding."""

from __future__ import annotations

import unittest

from planner.schemas import (
    LandingZoneSpec,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraft,
    SkillPlanDraftV2,
    migrate_plan_v1_to_v2,
)
from planner.scripted_dynamic_planner import ScriptedDynamicPlanner
from planner.scripted_target_semantics import (
    compile_scripted_target_description,
    target_spec_from_scripted_instruction,
)


def _world_context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={
            "search_area": SearchRegionSpec(
                "search_area",
                center_xyz_m=(20.0, 30.0, 0.0),
                radius_m=15.0,
                approach_xyz_m=(20.0, 12.0, 10.0),
            )
        },
        landing_zones={"home": LandingZoneSpec("home", (0.0, 0.0))},
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=10.0,
        search_timeout_s=60.0,
    )


def _draft(target_description: str = "moving target") -> SkillPlanDraft:
    return SkillPlanDraft.from_dict(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "takeoff_1",
                    "skill": "TAKEOFF",
                    "args": {"altitude_m": 10.0},
                },
                {
                    "id": "search_1",
                    "skill": "SEARCH",
                    "args": {
                        "region": "search_area",
                        "target_description": target_description,
                    },
                },
                {
                    "id": "land_1",
                    "skill": "LAND",
                    "args": {"zone": "home"},
                },
            ],
        }
    )


def _request(instruction: str) -> PlannerRequest:
    return PlannerRequest(
        instruction,
        _world_context(),
        mission_id="mission_scripted_semantics",
        uav_id="uav_1",
        plan_version=1,
    )


class ScriptedTargetSemanticsTests(unittest.TestCase):
    def test_moving_is_an_attribute_not_a_detector_category(self) -> None:
        generic = compile_scripted_target_description("moving target")
        self.assertEqual(generic.category, "unspecified")
        self.assertEqual(generic.hard_attributes, ("moving",))

        person = compile_scripted_target_description("一个正在移动的人")
        self.assertEqual(person.category, "person")
        self.assertEqual(person.hard_attributes, ("moving",))
        self.assertEqual(person.original_description, "一个正在移动的人")

    def test_exact_closed_set_categories_and_literal_attribute_are_kept(self) -> None:
        self.assertEqual(
            compile_scripted_target_description("a person").category,
            "person",
        )
        self.assertEqual(
            compile_scripted_target_description("car").category,
            "car",
        )
        described = compile_scripted_target_description("穿红衣的人")
        self.assertEqual(described.category, "person")
        self.assertEqual(described.hard_attributes, ("穿红衣",))

    def test_category_matching_is_anchored_and_unknowns_remain_unspecified(self) -> None:
        self.assertEqual(
            compile_scripted_target_description("carpet").category,
            "unspecified",
        )
        self.assertEqual(
            compile_scripted_target_description("personality").category,
            "unspecified",
        )
        self.assertEqual(
            compile_scripted_target_description("机器人").category,
            "unspecified",
        )
        unsupported = compile_scripted_target_description("红色立方体")
        self.assertEqual(unsupported.category, "unspecified")
        self.assertEqual(unsupported.original_description, "红色立方体")

    def test_instruction_extraction_is_limited_to_explicit_search_grammar(self) -> None:
        chinese = target_spec_from_scripted_instruction(
            "起飞后前往 search_area 搜寻一个人，确认以后跟踪十秒，最后返回 home 降落"
        )
        assert chinese is not None
        self.assertEqual(chinese.category, "person")

        english = target_spec_from_scripted_instruction(
            "go to search_area and search for a moving car, then track it"
        )
        assert english is not None
        self.assertEqual(english.category, "car")
        self.assertEqual(english.hard_attributes, ("moving",))

        self.assertIsNone(
            target_spec_from_scripted_instruction(
                "起飞后前往 search_area 搜索目标，跟踪十秒后返航"
            )
        )

    def test_dynamic_scripted_rebinds_search_and_final_target_spec(self) -> None:
        planner = ScriptedDynamicPlanner(_draft())
        output = planner.plan(
            _request(
                "起飞后前往 search_area 搜寻一个正在移动的人，"
                "确认以后跟踪十秒，最后返回 home 降落"
            )
        )
        self.assertIsInstance(output, SkillPlanDraftV2)
        assert isinstance(output, SkillPlanDraftV2)
        self.assertEqual(output.target_spec.category, "person")
        self.assertEqual(output.target_spec.hard_attributes, ("moving",))
        search = next(step for step in output.steps if step.skill == "SEARCH")
        self.assertEqual(
            search.args["target_description"],
            "一个正在移动的人",
        )

    def test_bare_target_keeps_constructor_baseline_and_oracle_semantics(self) -> None:
        planner = ScriptedDynamicPlanner(_draft("moving target"))
        output = planner.plan(
            _request("前往 search_area 搜索目标，跟踪后返回 home 降落")
        )
        assert isinstance(output, SkillPlanDraftV2)
        self.assertEqual(output.target_spec.category, "unspecified")
        self.assertEqual(output.target_spec.hard_attributes, ("moving",))
        self.assertEqual(output.target_spec.original_description, "moving target")

    def test_v1_migration_preserves_exact_category_semantics(self) -> None:
        person = migrate_plan_v1_to_v2(
            _draft("person"),
            mission_id="mission_person",
            uav_id="uav_1",
            plan_version=1,
        )
        self.assertEqual(person.target_spec.category, "person")
        self.assertEqual(person.target_spec.hard_attributes, ())


if __name__ == "__main__":
    unittest.main()
