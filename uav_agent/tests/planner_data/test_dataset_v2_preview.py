from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from models.base import ChatMessage
from planner.schemas import SkillPlanDraftV2
from planner_data.generator import GeneratedPlannerDataset
from planner_data.schemas import (
    DatasetProfile,
    PLANNER_DATASET_SCHEMA_VERSION,
    PLANNER_DATASET_SPLITS,
    PlannerDatasetSample,
    PlannerSampleMetadata,
)
from planner_data.v2_preview import (
    PLANNER_DATASET_V2_PREVIEW_SCHEMA,
    build_v2_preview_record,
    write_v2_preview_dataset,
)
from scripts.generate_planner_dataset import main as generation_main
from tasks.schemas import GoldPlannerSpec, PlannerWorldCase
from tasks.target_ontology import TargetOntology


_CONCEPT_ID = "person_upper_red_backpack_black"
_DESCRIPTION = "穿红色上衣并背黑色背包的人"
_INSTRUCTION = "前往东区寻找穿红色上衣并背黑色背包的人，跟踪三十秒后返回起点。"


def _sample() -> PlannerDatasetSample:
    gold = GoldPlannerSpec(
        spec_id="spec_preview_1",
        target_concept_id=_CONCEPT_ID,
        target_description=_DESCRIPTION,
        search_region="east_area",
        track_duration_s=30.0,
        landing_zone="home",
        takeoff_altitude_m=10.0,
        explicit_fields=frozenset(
            {
                "target_description",
                "search_region",
                "track_duration_s",
                "landing_zone",
                "takeoff_altitude_m",
            }
        ),
    )
    return PlannerDatasetSample(
        schema_version=PLANNER_DATASET_SCHEMA_VERSION,
        sample_id="train_000001",
        split="train",
        language="zh-CN",
        world_context_id="world_01",
        gold_spec_id=gold.spec_id,
        messages=(
            ChatMessage(role="system", content="legacy system"),
            ChatMessage(role="user", content=_INSTRUCTION),
            ChatMessage(role="assistant", content="{}"),
        ),
        gold=gold,
        metadata=PlannerSampleMetadata(
            instruction=_INSTRUCTION,
            template_family="train_preview",
            paraphrase_family="literal",
            generation_source="template",
            difficulty="easy",
            seed=1,
            semantic_spec_family="semantic_preview",
            group_id="group_preview",
        ),
    )


def _dataset() -> GeneratedPlannerDataset:
    world = PlannerWorldCase(
        context_id="world_01",
        search_regions={
            "east_area": "东区",
            "west_area": "西区",
        },
        landing_zones={"home": "起点", "backup": "备用降落区"},
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=30.0,
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
    )
    counts = {split: int(split == "train") for split in PLANNER_DATASET_SPLITS}
    samples = {
        split: ((_sample(),) if split == "train" else ())
        for split in PLANNER_DATASET_SPLITS
    }
    return GeneratedPlannerDataset(
        profile=DatasetProfile(name="pilot", split_counts=counts),
        seed=42,
        samples_by_split=samples,
        worlds={world.context_id: world},
        statistics={},
        resource_sha256={},
    )


class PlannerDatasetV2PreviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = _dataset()
        self.ontology = TargetOntology.load_default()

    def test_record_is_routed_schema_v2_without_changing_v1_sample(self) -> None:
        original = self.dataset.samples_by_split["train"][0].to_dict()
        record = build_v2_preview_record(
            self.dataset.samples_by_split["train"][0],
            dataset=self.dataset,
            ontology=self.ontology,
            uav_id="uav_1",
            system_prompt="Return strict schema-v2 JSON.",
        )
        self.assertEqual(record["dataset_schema"], PLANNER_DATASET_V2_PREVIEW_SCHEMA)
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["uav_id"], "uav_1")
        self.assertEqual(len(record["messages"]), 3)
        draft = SkillPlanDraftV2.from_dict(record["gold_skill_plan"])
        self.assertTrue(all(step.uav_id == "uav_1" for step in draft.steps))
        self.assertEqual(draft.target_spec.category, "person")
        self.assertEqual(
            draft.target_spec.original_description,
            _DESCRIPTION,
        )
        self.assertEqual(
            self.dataset.samples_by_split["train"][0].to_dict(),
            original,
        )

    def test_writer_uses_only_selected_temporary_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "planner_v2_preview"
            manifest = write_v2_preview_dataset(
                self.dataset,
                output,
                uav_id="uav_2",
                ontology=self.ontology,
            )
            self.assertEqual(manifest["uav_id"], "uav_2")
            self.assertEqual(manifest["split_counts"]["train"], 1)
            self.assertTrue((output / "dataset_manifest.json").is_file())
            self.assertTrue((output / "checksums.sha256").is_file())
            rows = (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            record = json.loads(rows[0])
            self.assertTrue(
                all(step["uav_id"] == "uav_2" for step in record["gold_skill_plan"]["steps"])
            )
            with self.assertRaises(FileExistsError):
                write_v2_preview_dataset(
                    self.dataset,
                    output,
                    uav_id="uav_2",
                    ontology=self.ontology,
                )

    def test_cli_rejects_unreviewed_full_v2_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = generation_main(
                [
                    "--output-root",
                    str(Path(directory) / "dataset"),
                    "--schema-version",
                    "2",
                    "--profile",
                    "full",
                ]
            )
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
