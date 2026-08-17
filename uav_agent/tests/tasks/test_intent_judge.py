from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import unittest

from experiments.evaluator import compute_instruction_grounded_success
from planner.schemas import MissionIntent
from tasks.intent_judge import IntentErrorCode, IntentJudge
from tasks.schemas import GoldPlannerSpec, PlannerWorldCase
from tasks.target_ontology import TargetOntology


CONCEPT_ID = "person_upper_red_backpack_black"
CANONICAL = "穿红色上衣并背黑色背包的人"


def make_world() -> PlannerWorldCase:
    return PlannerWorldCase(
        context_id="world_01",
        search_regions={
            "east_area": "东侧区域",
            "west_area": "西侧区域",
            "north_area": "北侧区域",
            "south_area": "南侧区域",
        },
        landing_zones={
            "home": "起点",
            "north_pad": "北平台",
            "south_pad": "南平台",
        },
        default_takeoff_altitude_m=10,
        default_track_duration_s=30,
        scene_min_xyz_m=(-50, -50, 0),
        scene_max_xyz_m=(50, 50, 30),
    )


def make_gold(**updates: object) -> GoldPlannerSpec:
    values: dict[str, object] = {
        "spec_id": "spec_000001",
        "target_concept_id": CONCEPT_ID,
        "target_description": CANONICAL,
        "search_region": "east_area",
        "track_duration_s": 30.0,
        "landing_zone": "home",
        "takeoff_altitude_m": None,
        "explicit_fields": {
            "target_description",
            "search_region",
            "track_duration_s",
            "landing_zone",
        },
    }
    values.update(updates)
    return GoldPlannerSpec(**values)


def prediction(**updates: object) -> MissionIntent:
    values: dict[str, object] = {
        "target_description": CANONICAL,
        "search_region": "east_area",
        "track_duration_s": 30.0,
        "landing_zone": "home",
        "takeoff_altitude_m": None,
    }
    values.update(updates)
    return MissionIntent(**values)


def successful_execution() -> dict[str, bool]:
    return {
        "takeoff_success": True,
        "goto_search_success": True,
        "search_success": True,
        "correct_target_locked": True,
        "track_success": True,
        "return_success": True,
        "landing_success": True,
        "false_target_lock": False,
        "collision": False,
        "out_of_bounds": False,
        "safety_abort": False,
        "timeout": False,
    }


class IntentJudgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = TargetOntology.load_default()
        cls.judge = IntentJudge(cls.ontology)
        cls.world = make_world()

    def judge_prediction(self, value: MissionIntent | None, **kwargs: object):
        return self.judge.judge(
            gold=make_gold(),
            predicted=value,
            world=self.world,
            **kwargs,
        )

    def test_all_five_fields_exact_and_semantic_match(self) -> None:
        result = self.judge_prediction(prediction())
        self.assertTrue(result.output_valid)
        self.assertTrue(result.exact_match)
        self.assertTrue(result.semantic_match)
        self.assertEqual(result.error_codes, ())
        json.dumps(result.to_dict(), allow_nan=False)

    def test_each_field_mismatch_breaks_semantic_match(self) -> None:
        cases = (
            (
                {"target_description": "穿蓝色上衣并背黑色背包的人"},
                "target_match",
                IntentErrorCode.TARGET_MISMATCH.value,
            ),
            (
                {"search_region": "west_area"},
                "search_region_match",
                IntentErrorCode.SEARCH_REGION_MISMATCH.value,
            ),
            (
                {"track_duration_s": 10},
                "track_duration_match",
                IntentErrorCode.TRACK_DURATION_MISMATCH.value,
            ),
            (
                {"landing_zone": "north_pad"},
                "landing_zone_match",
                IntentErrorCode.LANDING_ZONE_MISMATCH.value,
            ),
            (
                {"takeoff_altitude_m": 12},
                "takeoff_altitude_match",
                IntentErrorCode.TAKEOFF_ALTITUDE_MISMATCH.value,
            ),
        )
        for updates, field_name, code in cases:
            with self.subTest(field=field_name):
                result = self.judge_prediction(prediction(**updates))
                self.assertFalse(result.semantic_match)
                self.assertFalse(getattr(result, field_name))
                self.assertIn(code, result.error_codes)

    def test_null_gold_and_numeric_default_are_semantically_equivalent_only(self) -> None:
        result = self.judge_prediction(prediction(takeoff_altitude_m=10.0))
        self.assertFalse(result.exact_match)
        self.assertTrue(result.semantic_match)
        self.assertTrue(result.takeoff_altitude_match)
        self.assertEqual(result.takeoff_altitude_error_m, 0.0)

    def test_registered_target_alias_is_semantic_but_not_exact(self) -> None:
        result = self.judge_prediction(prediction(target_description="红衣黑包的人"))
        self.assertFalse(result.exact_match)
        self.assertTrue(result.semantic_match)
        self.assertTrue(result.target_match)

    def test_unknown_target_is_not_guessed(self) -> None:
        result = self.judge_prediction(
            prediction(target_description="近似红色衣服的目标")
        )
        self.assertTrue(result.output_valid)
        self.assertFalse(result.target_match)
        self.assertFalse(result.semantic_match)
        self.assertIn(
            IntentErrorCode.UNKNOWN_TARGET_DESCRIPTION.value,
            result.error_codes,
        )
        self.assertIn(IntentErrorCode.TARGET_MISMATCH.value, result.error_codes)

    def test_missing_prediction_or_parse_error_marks_output_invalid(self) -> None:
        missing = self.judge_prediction(None)
        parsed_error = self.judge_prediction(
            None,
            parse_error=ValueError("invalid JSON"),
        )
        for result in (missing, parsed_error):
            self.assertFalse(result.output_valid)
            self.assertFalse(result.exact_match)
            self.assertFalse(result.semantic_match)
            self.assertFalse(result.target_match)
            self.assertIsNone(result.track_duration_error_s)
            self.assertEqual(
                result.error_codes,
                (IntentErrorCode.PLANNER_OUTPUT_INVALID.value,),
            )

    def test_judge_does_not_mutate_gold_or_prediction(self) -> None:
        gold = make_gold()
        predicted = prediction()
        gold_before = gold.to_dict()
        predicted_before = predicted.to_dict()
        result = self.judge.judge(
            gold=gold,
            predicted=predicted,
            world=self.world,
        )
        self.assertEqual(gold.to_dict(), gold_before)
        self.assertEqual(predicted.to_dict(), predicted_before)
        with self.assertRaises(FrozenInstanceError):
            result.semantic_match = False  # type: ignore[misc]


class InstructionGroundedSuccessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.judge = IntentJudge(TargetOntology.load_default())
        cls.world = make_world()

    def result(self, predicted: MissionIntent):
        return self.judge.judge(
            gold=make_gold(),
            predicted=predicted,
            world=self.world,
        )

    def test_matching_intent_and_every_execution_condition_succeeds(self) -> None:
        result = self.result(prediction())
        self.assertTrue(
            compute_instruction_grounded_success(successful_execution(), result)
        )

    def test_wrong_duration_fails_even_if_predicted_task_was_completed(self) -> None:
        result = self.result(prediction(track_duration_s=10))
        self.assertFalse(
            compute_instruction_grounded_success(successful_execution(), result)
        )

    def test_wrong_landing_zone_fails_even_after_safe_landing(self) -> None:
        result = self.result(prediction(landing_zone="north_pad"))
        self.assertFalse(
            compute_instruction_grounded_success(successful_execution(), result)
        )

    def test_wrong_search_region_fails_even_after_full_execution(self) -> None:
        result = self.result(prediction(search_region="west_area"))
        self.assertFalse(
            compute_instruction_grounded_success(successful_execution(), result)
        )

    def test_structurally_valid_wrong_target_attributes_fail(self) -> None:
        result = self.result(
            prediction(target_description="穿蓝色上衣并背黑色背包的人")
        )
        self.assertTrue(result.output_valid)
        self.assertFalse(
            compute_instruction_grounded_success(successful_execution(), result)
        )

    def test_any_execution_failure_still_fails_with_matching_intent(self) -> None:
        result = self.result(prediction())
        for field in (
            "takeoff_success",
            "goto_search_success",
            "search_success",
            "correct_target_locked",
            "track_success",
            "return_success",
            "landing_success",
        ):
            metrics = successful_execution()
            metrics[field] = False
            with self.subTest(field=field):
                self.assertFalse(
                    compute_instruction_grounded_success(metrics, result)
                )
        for field in (
            "false_target_lock",
            "collision",
            "out_of_bounds",
            "safety_abort",
            "timeout",
        ):
            metrics = successful_execution()
            metrics[field] = True
            with self.subTest(field=field):
                self.assertFalse(
                    compute_instruction_grounded_success(metrics, result)
                )

    def test_grounded_success_rejects_non_judge_result(self) -> None:
        with self.assertRaises(TypeError):
            compute_instruction_grounded_success(
                successful_execution(),
                {"semantic_match": True},  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
