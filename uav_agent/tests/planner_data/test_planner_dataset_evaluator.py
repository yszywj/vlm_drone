from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from models.base import (
    ChatMessage,
    GenerationOptions,
    ModelConnectionError,
    ModelResponse,
)
from planner_data.evaluator import (
    PlannerDatasetEvaluator,
    PlannerEvaluationError,
    PlannerEvaluationErrorCode,
    aggregate_planner_predictions,
    load_planner_dataset_split,
    load_planner_world_cases,
    planner_world_case_to_runtime_context,
)
from planner_data.renderers import world_case_to_runtime_context
from planner_data.schemas import (
    PLANNER_DATASET_SCHEMA_VERSION,
    PlannerDatasetSample,
    PlannerSampleMetadata,
)
from tasks.schemas import GoldPlannerSpec, PlannerWorldCase
from tasks.target_ontology import TargetOntology


CONCEPT_ID = "person_upper_red_backpack_black"
CANONICAL = "穿红色上衣并背黑色背包的人"


class FakeModelClient:
    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.options: list[GenerationOptions | None] = []
        self.messages: list[tuple[ChatMessage, ...]] = []

    def healthcheck(self) -> None:
        return None

    def chat(self, messages, *, options=None):
        self.calls += 1
        self.options.append(options)
        self.messages.append(tuple(messages))
        if not self.outcomes:
            raise AssertionError("unexpected model call")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ModelResponse(
            content=outcome,
            model="fake-qwen",
            finish_reason="stop",
            usage={},
        )


def world() -> PlannerWorldCase:
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
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=30.0,
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
    )


def intent_dict(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "target_description": CANONICAL,
        "search_region": "east_area",
        "track_duration_s": 30.0,
        "landing_zone": "home",
        "takeoff_altitude_m": None,
    }
    value.update(updates)
    return value


def output(**updates: object) -> str:
    return json.dumps(
        intent_dict(**updates),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def sample(index: int) -> PlannerDatasetSample:
    spec_id = f"spec_{index:06d}"
    gold = GoldPlannerSpec(
        spec_id=spec_id,
        target_concept_id=CONCEPT_ID,
        target_description=CANONICAL,
        search_region="east_area",
        track_duration_s=30.0,
        landing_zone="home",
        takeoff_altitude_m=None,
        explicit_fields=frozenset(
            {
                "target_description",
                "search_region",
                "track_duration_s",
                "landing_zone",
            }
        ),
    )
    instruction = f"第{index}项任务：去东区寻找红衣黑包的人并跟踪三十秒后返回起点。"
    return PlannerDatasetSample(
        schema_version=PLANNER_DATASET_SCHEMA_VERSION,
        sample_id=f"test_iid_{index:06d}",
        split="test_iid",
        language="zh-CN",
        world_context_id="world_01",
        gold_spec_id=spec_id,
        messages=(
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content=instruction),
            ChatMessage(role="assistant", content=output()),
        ),
        gold=gold,
        metadata=PlannerSampleMetadata(
            instruction=instruction,
            template_family="test_template",
            paraphrase_family=f"test_paraphrase_{index}",
            generation_source="human",
            difficulty="medium",
            seed=1000 + index,
            semantic_spec_family=f"semantic_{index}",
            group_id=f"group_{index}",
        ),
    )


class PlannerDatasetEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ontology = TargetOntology.load_default()
        self.worlds = {"world_01": world()}
        self.prompt = (
            Path(__file__).resolve().parents[2]
            / "prompts"
            / "mission_planner_system.txt"
        )

    def llm_evaluator(self, client: FakeModelClient) -> PlannerDatasetEvaluator:
        return PlannerDatasetEvaluator(
            planner="llm",
            world_cases=self.worlds,
            ontology=self.ontology,
            model_client=client,
            system_prompt_path=self.prompt,
        )

    def test_scripted_gold_is_exact_and_semantic_100_percent(self) -> None:
        evaluator = PlannerDatasetEvaluator(
            planner="scripted",
            world_cases=self.worlds,
            ontology=self.ontology,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run = evaluator.evaluate(
                (sample(1), sample(2)),
                output_root=temporary,
            )

            self.assertEqual(run.summary["num_samples"], 2)
            self.assertEqual(run.summary["output_valid_rate"], 1.0)
            self.assertEqual(run.summary["exact_match_rate"], 1.0)
            self.assertEqual(run.summary["semantic_match_rate"], 1.0)
            self.assertEqual(run.summary["repair_request_rate"], 0.0)
            for filename in (
                "summary.json",
                "predictions.jsonl",
                "errors.csv",
                "field_metrics.csv",
                "terminal.log",
            ):
                self.assertTrue((run.run_dir / filename).is_file(), filename)
            self.assertTrue(all(row["model_calls"] == 0 for row in run.predictions))

    def test_valid_llm_output_uses_temperature_zero(self) -> None:
        client = FakeModelClient([output()])
        evaluator = self.llm_evaluator(client)
        with tempfile.TemporaryDirectory() as temporary:
            run = evaluator.evaluate((sample(1),), output_root=temporary)

        self.assertEqual(client.calls, 1)
        self.assertEqual(client.options[0].temperature, 0.0)
        self.assertEqual(run.summary["semantic_match_rate"], 1.0)
        record = run.predictions[0]
        self.assertEqual(record["initial_model_output"], output())
        self.assertEqual(record["final_model_output"], output())
        self.assertEqual(record["model_calls"], 1)

    def test_fixed_wrong_planner_counts_field_error(self) -> None:
        client = FakeModelClient([output(search_region="west_area")])
        evaluator = self.llm_evaluator(client)
        with tempfile.TemporaryDirectory() as temporary:
            run = evaluator.evaluate((sample(1),), output_root=temporary)
            with (run.run_dir / "errors.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                errors = {row["error_code"]: int(row["count"]) for row in csv.DictReader(stream)}

        self.assertEqual(run.summary["output_valid_rate"], 1.0)
        self.assertEqual(run.summary["semantic_match_rate"], 0.0)
        self.assertEqual(run.summary["search_region_accuracy"], 0.0)
        self.assertEqual(
            errors[PlannerEvaluationErrorCode.SEARCH_REGION_MISMATCH.value],
            1,
        )
        self.assertEqual(
            errors[PlannerEvaluationErrorCode.PLANNER_SEMANTIC_MISMATCH.value],
            1,
        )

    def test_invalid_json_and_failed_repair_are_output_invalid(self) -> None:
        client = FakeModelClient(["not json", "still not json"])
        evaluator = self.llm_evaluator(client)
        with tempfile.TemporaryDirectory() as temporary:
            run = evaluator.evaluate((sample(1),), output_root=temporary)

        record = run.predictions[0]
        self.assertEqual(client.calls, 2)
        self.assertEqual(record["model_calls"], 2)
        self.assertTrue(record["repair_requested"])
        self.assertFalse(record["repair_succeeded"])
        self.assertFalse(record["judge_result"]["output_valid"])
        self.assertEqual(run.summary["output_valid_rate"], 0.0)
        self.assertEqual(run.summary["repair_success_rate"], 0.0)

    def test_repair_success_reports_initial_and_final_metrics(self) -> None:
        client = FakeModelClient(["not json", output()])
        evaluator = self.llm_evaluator(client)
        with tempfile.TemporaryDirectory() as temporary:
            run = evaluator.evaluate((sample(1),), output_root=temporary)

        record = run.predictions[0]
        self.assertEqual(record["initial_model_output"], "not json")
        self.assertEqual(record["final_model_output"], output())
        self.assertFalse(record["initial_judge_result"]["output_valid"])
        self.assertTrue(record["judge_result"]["semantic_match"])
        self.assertTrue(record["repair_succeeded"])
        self.assertEqual(run.summary["initial_output_valid_rate"], 0.0)
        self.assertEqual(run.summary["output_valid_rate"], 1.0)
        self.assertEqual(run.summary["repair_request_rate"], 1.0)
        self.assertEqual(run.summary["repair_success_rate"], 1.0)

    def test_request_failure_does_not_abort_later_sample(self) -> None:
        client = FakeModelClient(
            [ModelConnectionError("offline"), output()]
        )
        evaluator = self.llm_evaluator(client)
        with tempfile.TemporaryDirectory() as temporary:
            run = evaluator.evaluate(
                (sample(1), sample(2)),
                output_root=temporary,
            )

        self.assertEqual(client.calls, 2)
        self.assertEqual(len(run.predictions), 2)
        self.assertTrue(run.predictions[0]["request_failed"])
        self.assertFalse(run.predictions[0]["judge_result"]["output_valid"])
        self.assertTrue(run.predictions[1]["judge_result"]["semantic_match"])
        self.assertEqual(run.summary["output_valid_rate"], 0.5)

    def test_failed_repair_request_keeps_initial_and_final_responses_distinct(self) -> None:
        client = FakeModelClient(
            ["not json", ModelConnectionError("repair offline")]
        )
        evaluator = self.llm_evaluator(client)
        with tempfile.TemporaryDirectory() as temporary:
            run = evaluator.evaluate((sample(1),), output_root=temporary)

        record = run.predictions[0]
        self.assertEqual(record["model_calls"], 2)
        self.assertEqual(record["initial_model_output"], "not json")
        self.assertEqual(record["final_model_output"], "")
        self.assertTrue(record["request_failed"])
        self.assertTrue(record["repair_requested"])
        self.assertFalse(record["repair_succeeded"])

    def test_resume_skips_completed_ids_without_model_call(self) -> None:
        client = FakeModelClient([output(), output()])
        evaluator = self.llm_evaluator(client)
        samples = (sample(1), sample(2))
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "resume-run"
            first = evaluator.evaluate(
                samples,
                run_dir=run_dir,
                limit=1,
            )
            self.assertEqual(client.calls, 1)
            resumed = evaluator.evaluate(
                samples,
                run_dir=first.run_dir,
                resume=True,
            )

            lines = (resumed.run_dir / "predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(client.calls, 2)
        self.assertEqual(len(lines), 2)
        self.assertEqual(resumed.summary["num_samples"], 2)

    def test_dataset_loader_rejects_duplicate_json_keys(self) -> None:
        serialized = json.dumps(
            sample(1).to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        # Repeating the same value would otherwise survive last-key-wins and
        # produce a completely valid sample.
        corrupted = serialized[:-1] + ',"sample_id":"test_iid_000001"}'
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "test_iid.jsonl").write_text(
                corrupted + "\n", encoding="utf-8"
            )
            with self.assertRaises(PlannerEvaluationError):
                load_planner_dataset_split(temporary, "test_iid")

    def test_dataset_loader_rejects_non_finite_json_numbers(self) -> None:
        serialized = json.dumps(
            sample(1).to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        corrupted = serialized.replace(
            '"track_duration_s":30.0',
            '"track_duration_s":NaN',
            1,
        )
        self.assertNotEqual(corrupted, serialized)
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "test_iid.jsonl").write_text(
                corrupted + "\n", encoding="utf-8"
            )
            with self.assertRaises(PlannerEvaluationError):
                load_planner_dataset_split(temporary, "test_iid")

    def test_resume_rejects_duplicate_keys_and_non_finite_numbers(self) -> None:
        evaluator = PlannerDatasetEvaluator(
            planner="scripted",
            world_cases=self.worlds,
            ontology=self.ontology,
        )
        for corruption in ("duplicate", "non_finite"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "resume-run"
                run = evaluator.evaluate((sample(1),), run_dir=run_dir)
                path = run.run_dir / "predictions.jsonl"
                line = path.read_text(encoding="utf-8").rstrip("\n")
                if corruption == "duplicate":
                    line = line[:-1] + ',"sample_id":"test_iid_000001"}'
                else:
                    record = json.loads(line)
                    record["latency_ms"] = float("nan")
                    line = json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=True,
                        separators=(",", ":"),
                    )
                path.write_text(line + "\n", encoding="utf-8")
                with self.assertRaises(PlannerEvaluationError):
                    evaluator.evaluate(
                        (sample(1),),
                        run_dir=run.run_dir,
                        resume=True,
                    )

    def test_world_loader_rejects_duplicate_yaml_keys(self) -> None:
        duplicated = """\
schema_version: planner_world_contexts_v1
schema_version: planner_world_contexts_v1
worlds: []
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worlds.yaml"
            path.write_text(duplicated, encoding="utf-8")
            with self.assertRaises(PlannerEvaluationError):
                load_planner_world_cases(path)

    def test_runtime_world_projection_is_shared_with_renderer(self) -> None:
        self.assertEqual(
            planner_world_case_to_runtime_context(world()),
            world_case_to_runtime_context(world()),
        )

    def test_start_index_and_limit_select_expected_sample(self) -> None:
        client = FakeModelClient([output()])
        evaluator = self.llm_evaluator(client)
        with tempfile.TemporaryDirectory() as temporary:
            run = evaluator.evaluate(
                (sample(1), sample(2), sample(3)),
                output_root=temporary,
                start_index=1,
                limit=1,
            )
        self.assertEqual(client.calls, 1)
        self.assertEqual(run.predictions[0]["sample_id"], "test_iid_000002")

    def test_summary_matches_prediction_aggregation(self) -> None:
        client = FakeModelClient(
            [output(), output(track_duration_s=10.0), "bad", "still bad"]
        )
        evaluator = self.llm_evaluator(client)
        with tempfile.TemporaryDirectory() as temporary:
            run = evaluator.evaluate(
                (sample(1), sample(2), sample(3)),
                output_root=temporary,
            )
            persisted = [
                json.loads(line)
                for line in (run.run_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(dict(run.summary), aggregate_planner_predictions(persisted))
        self.assertEqual(run.summary["num_samples"], 3)
        self.assertAlmostEqual(run.summary["output_valid_rate"], 2 / 3)
        self.assertAlmostEqual(run.summary["semantic_match_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
