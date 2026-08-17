from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from planner.schemas import MissionIntent
from planner_data.generator import (
    DatasetGenerationError,
    PlannerDatasetGenerator,
    build_statistics,
    load_generation_config,
    serialize_expected_intent,
    stage_external_candidates,
    validate_paraphrase_candidate,
    write_generated_dataset,
)
from planner_data.schemas import PLANNER_DATASET_SPLITS, PlannerDatasetSample
from scripts.generate_planner_dataset import _strict_json_loads as strict_candidate_json_loads


class PlannerDatasetGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = PlannerDatasetGenerator()
        cls.first = cls.generator.generate(seed=42, profile="pilot")

    def test_pilot_counts_are_exact(self) -> None:
        self.assertEqual(self.first.num_samples, 1900)
        self.assertEqual(
            {split: len(self.first.samples_by_split[split]) for split in PLANNER_DATASET_SPLITS},
            {
                "train": 1000,
                "validation": 200,
                "test_iid": 200,
                "test_compositional": 200,
                "test_language": 200,
                "test_robustness": 100,
            },
        )

    def test_same_seed_is_byte_reproducible(self) -> None:
        second = self.generator.generate(seed=42, profile="pilot")
        first_rows = [
            json.dumps(sample.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            for split in PLANNER_DATASET_SPLITS
            for sample in self.first.samples_by_split[split]
        ]
        second_rows = [
            json.dumps(sample.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            for split in PLANNER_DATASET_SPLITS
            for sample in second.samples_by_split[split]
        ]
        self.assertEqual(first_rows, second_rows)

    def test_different_seed_changes_instructions(self) -> None:
        other = self.generator.generate(seed=43, profile="pilot")
        self.assertNotEqual(
            self.first.samples_by_split["train"][0].metadata.instruction,
            other.samples_by_split["train"][0].metadata.instruction,
        )

    def test_assistant_label_is_exactly_gold_expected_intent(self) -> None:
        for split in PLANNER_DATASET_SPLITS:
            for sample in self.first.samples_by_split[split][:10]:
                assistant = sample.messages[-1].content
                self.assertEqual(assistant, serialize_expected_intent(sample.gold))
                parsed = MissionIntent.from_dict(json.loads(assistant))
                self.assertEqual(parsed, sample.gold.to_expected_intent())

    def test_all_samples_round_trip_with_fixed_field_order(self) -> None:
        sample = self.first.samples_by_split["train"][0]
        encoded = json.dumps(sample.to_dict(), ensure_ascii=False, allow_nan=False)
        decoded = PlannerDatasetSample.from_dict(json.loads(encoded))
        self.assertEqual(decoded.to_dict(), sample.to_dict())
        self.assertEqual(list(sample.to_dict()), [
            "schema_version", "sample_id", "split", "language",
            "world_context_id", "gold_spec_id", "messages", "gold", "metadata",
        ])

    def test_every_declared_field_has_minimum_coverage(self) -> None:
        stats = self.first.statistics
        for field in (
            "target_concept_counts",
            "search_region_counts",
            "landing_zone_counts",
            "track_duration_counts",
        ):
            self.assertGreater(len(stats[field]), 1)
            self.assertGreaterEqual(min(stats[field].values()), 1)
        self.assertEqual(stats["automatic_paraphrase_fraction"], 0.0)
        self.assertEqual(stats["generation_source_counts"], {"template": 1900})
        self.assertEqual(stats["review_status_counts"], {"unreviewed": 1900})
        self.assertEqual(stats["template_fraction"], 1.0)
        self.assertEqual(stats["external_candidate_fraction"], 0.0)
        self.assertEqual(stats["train_external_candidate_fraction"], 0.0)
        self.assertEqual(stats["non_train_external_candidate_count"], 0)
        self.assertEqual(stats["unreviewed_external_candidate_count"], 0)
        self.assertTrue(stats["external_candidate_policy_met"])
        self.assertEqual(
            stats["test_language_human_authored_or_reviewed_fraction"],
            0.0,
        )
        self.assertEqual(stats["test_robustness_human_reviewed_fraction"], 0.0)
        self.assertFalse(stats["full_review_requirements_met"])
        for split in PLANNER_DATASET_SPLITS:
            self.assertEqual(
                stats["split_provenance"][split]["generation_source_counts"],
                {"template": len(self.first.samples_by_split[split])},
            )
            self.assertEqual(
                stats["split_provenance"][split]["review_status_counts"],
                {"unreviewed": len(self.first.samples_by_split[split])},
            )

    def test_minimum_coverage_checks_every_closed_set_value(self) -> None:
        statistics = dict(self.first.statistics)
        target_counts = dict(statistics["target_concept_counts"])
        omitted = next(iter(self.generator.ontology.concepts))
        target_counts.pop(omitted)
        statistics["target_concept_counts"] = target_counts
        with self.assertRaisesRegex(DatasetGenerationError, omitted):
            self.generator.validate_minimum_coverage(statistics)

    def test_external_candidate_provenance_policy_is_fail_closed(self) -> None:
        base = {
            split: tuple(samples[:1])
            for split, samples in self.first.samples_by_split.items()
        }

        train = base["train"][0]
        reviewed_external = replace(
            train,
            metadata=replace(
                train.metadata,
                generation_source="external_candidate",
                review_status="human_reviewed_template",
            ),
        )
        over_half = dict(base)
        over_half["train"] = (reviewed_external,)
        stats = build_statistics(over_half)
        self.assertEqual(stats["train_external_candidate_fraction"], 1.0)
        self.assertFalse(stats["external_candidate_policy_met"])

        validation = base["validation"][0]
        non_train = dict(base)
        non_train["validation"] = (
            replace(
                validation,
                metadata=replace(
                    validation.metadata,
                    generation_source="external_candidate",
                    review_status="human_reviewed_template",
                ),
            ),
        )
        stats = build_statistics(non_train)
        self.assertEqual(stats["non_train_external_candidate_count"], 1)
        self.assertFalse(stats["external_candidate_policy_met"])

        unreviewed = dict(base)
        unreviewed["train"] = (
            replace(
                train,
                metadata=replace(
                    train.metadata,
                    generation_source="external_candidate",
                    review_status="unreviewed",
                ),
            ),
        )
        stats = build_statistics(unreviewed)
        self.assertEqual(stats["unreviewed_external_candidate_count"], 1)
        self.assertFalse(stats["external_candidate_policy_met"])

        official_splits = dict(self.first.samples_by_split)
        official_splits["train"] = (
            replace(
                self.first.samples_by_split["train"][0],
                metadata=replace(
                    self.first.samples_by_split["train"][0].metadata,
                    generation_source="external_candidate",
                    review_status="unreviewed",
                ),
            ),
            *self.first.samples_by_split["train"][1:],
        )
        invalid_official = replace(
            self.first,
            samples_by_split=official_splits,
            statistics=build_statistics(official_splits),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DatasetGenerationError, "human-reviewed"):
                write_generated_dataset(
                    invalid_official,
                    Path(directory) / "planner_v1",
                )

    def test_publisher_recomputes_statistics_instead_of_trusting_flags(self) -> None:
        forged_statistics = dict(self.first.statistics)
        forged_statistics["full_review_requirements_met"] = True
        forged = replace(self.first, statistics=forged_statistics)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DatasetGenerationError, "recomputed exactly"):
                write_generated_dataset(forged, Path(directory) / "planner_v1")

    def test_robustness_rows_preserve_real_structured_categories(self) -> None:
        required = {
            "prompt_injection",
            "extra_field",
            "format_interference",
            "irrelevant_text",
            "long_instruction",
            "repeated_requirement",
        }
        rows = self.first.samples_by_split["test_robustness"]
        categories = {sample.metadata.robustness_category for sample in rows}
        self.assertTrue(required.issubset(categories))
        self.assertEqual(
            self.first.statistics["robustness_category_counts"],
            {
                category: sum(
                    sample.metadata.robustness_category == category for sample in rows
                )
                for category in sorted(categories)
            },
        )
        injection_by_category = {}
        for item in self.generator.lexicon.robustness_injections:
            injection_by_category.setdefault(item.category, set()).add(item.text)
        for sample in rows:
            category = sample.metadata.robustness_category
            self.assertIsNotNone(category)
            self.assertTrue(
                any(
                    text in sample.metadata.instruction
                    for text in injection_by_category[category]
                )
            )

    def test_ordinary_template_ids_and_families_do_not_cross_splits(self) -> None:
        ordinary = ("train", "validation", "test_iid", "test_compositional")
        family_sets = {
            split: {
                sample.metadata.template_family
                for sample in self.first.samples_by_split[split]
            }
            for split in ordinary
        }
        for index, left in enumerate(ordinary):
            for right in ordinary[index + 1 :]:
                self.assertTrue(family_sets[left].isdisjoint(family_sets[right]))

    def test_full_profile_is_not_publishable_without_real_human_review(self) -> None:
        full = self.generator.generate(seed=42, profile="full")
        self.assertFalse(full.statistics["full_review_requirements_met"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DatasetGenerationError, "human-reviewed"):
                write_generated_dataset(full, Path(directory) / "planner_v1")

    def test_dataset_config_and_target_ontology_reject_duplicate_yaml_keys(self) -> None:
        resources = Path(__file__).resolve().parents[2] / "resources" / "planner_v1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "planner_v1"
            shutil.copytree(resources, root)
            config = root / "dataset_config.yaml"
            config.write_text(
                config.read_text(encoding="utf-8") + "\nlanguage: zh-CN\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetGenerationError, "duplicate key"):
                load_generation_config(config)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "planner_v1"
            shutil.copytree(resources, root)
            ontology = root / "target_ontology.yaml"
            ontology.write_text(
                ontology.read_text(encoding="utf-8")
                + "\nschema_version: planner_target_ontology_v1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetGenerationError, "duplicate key"):
                PlannerDatasetGenerator(root / "dataset_config.yaml")

    def test_external_paraphrases_cannot_enter_official_splits(self) -> None:
        class FakeParaphraser:
            def paraphrase(self, instruction, gold_spec, allowed_aliases, count):
                return [instruction]

        with self.assertRaisesRegex(DatasetGenerationError, "candidate-only"):
            self.generator.generate(seed=42, profile="pilot", paraphraser=FakeParaphraser())

    def test_external_candidate_is_conservatively_checked_and_staged(self) -> None:
        sample = self.first.samples_by_split["train"][0]
        world = self.first.worlds[sample.world_context_id]
        accepted = validate_paraphrase_candidate(
            sample.metadata.instruction,
            gold=sample.gold,
            world=world,
            ontology=self.generator.ontology,
            lexicon=self.generator.lexicon,
        )
        self.assertTrue(accepted.accepted, accepted.reasons)
        rejected = validate_paraphrase_candidate(
            sample.metadata.instruction + "改成去东区并跟踪10秒绿色衣服的人。",
            gold=sample.gold,
            world=world,
            ontology=self.generator.ontology,
            lexicon=self.generator.lexicon,
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("NEGATION_OR_SEMANTIC_CHANGE", rejected.reasons)
        self.assertIn("UNKNOWN_TARGET_ATTRIBUTE", rejected.reasons)
        with tempfile.TemporaryDirectory() as directory:
            fixed_temporary = (
                Path(directory)
                / "_candidates"
                / ".external_candidates.jsonl.tmp"
            )
            fixed_temporary.parent.mkdir(parents=True)
            fixed_temporary.write_text("unrelated user work", encoding="utf-8")
            path = stage_external_candidates(
                generated=self.first,
                candidates=[
                    {
                        "sample_id": sample.sample_id,
                        "candidate_instruction": sample.metadata.instruction,
                    }
                ],
                output_root=directory,
                ontology=self.generator.ontology,
                lexicon=self.generator.lexicon,
            )
            self.assertEqual(path.parent.name, "_candidates")
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["canonical_instruction"], sample.metadata.instruction)
            self.assertNotIn("messages", record)
            self.assertEqual(
                fixed_temporary.read_text(encoding="utf-8"),
                "unrelated user work",
            )

    def test_external_candidate_rejects_every_introduced_semantic_value(self) -> None:
        sample = self.first.samples_by_split["train"][0]
        world = self.first.worlds[sample.world_context_id]

        def validate(suffix: str):
            return validate_paraphrase_candidate(
                sample.metadata.instruction + suffix,
                gold=sample.gold,
                world=world,
                ontology=self.generator.ontology,
                lexicon=self.generator.lexicon,
            )

        cases = {
            "另外目标穿蓝色衣服。": "UNKNOWN_TARGET_ATTRIBUTE",
            "另外跟踪17秒。": "CONFLICTING_DURATION",
            "再去中央区搜索。": "UNKNOWN_REGION",
            "任务后在东侧新降落点降落。": "UNKNOWN_LANDING_ZONE",
            "起飞高度改成17米。": "CONFLICTING_ALTITUDE",
            "目标还戴棕色帽子。": "UNKNOWN_TARGET_ATTRIBUTE",
            "目标还背青色背包。": "UNKNOWN_TARGET_ATTRIBUTE",
            "目标还戴手表。": "UNKNOWN_TARGET_ATTRIBUTE",
            "目标手持雨伞。": "UNKNOWN_TARGET_ATTRIBUTE",
            "目标穿卡其外套。": "UNKNOWN_TARGET_ATTRIBUTE",
            "目标是短发。": "UNKNOWN_TARGET_ATTRIBUTE",
            "不要跟踪目标。": "NEGATION_OR_SEMANTIC_CHANGE",
        }
        for suffix, expected_reason in cases.items():
            with self.subTest(suffix=suffix):
                result = validate(suffix)
                self.assertFalse(result.accepted)
                self.assertIn(expected_reason, result.reasons)

    def test_external_candidate_rejects_multiple_conflicting_values(self) -> None:
        sample = self.first.samples_by_split["train"][0]
        result = validate_paraphrase_candidate(
            sample.metadata.instruction
            + "改成东区或南区，跟踪10秒或20秒，在基地降落。",
            gold=sample.gold,
            world=self.first.worlds[sample.world_context_id],
            ontology=self.generator.ontology,
            lexicon=self.generator.lexicon,
        )
        self.assertFalse(result.accepted)
        self.assertIn("CONFLICTING_REGION", result.reasons)
        self.assertIn("CONFLICTING_DURATION", result.reasons)
        self.assertIn("CONFLICTING_LANDING_ZONE", result.reasons)
        self.assertIn("NEGATION_OR_SEMANTIC_CHANGE", result.reasons)

    def test_external_candidate_rejects_runtime_truth_and_media_content(self) -> None:
        sample = self.first.samples_by_split["train"][0]
        world = self.first.worlds[sample.world_context_id]
        forbidden_suffixes = (
            "目标坐标是(1,2,3)。",
            "目标速度为(1,0,0)。",
            "图片路径/tmp/frame.png。",
            "视频video.mp4。",
            "附带帧数据。",
        )
        for suffix in forbidden_suffixes:
            with self.subTest(suffix=suffix):
                result = validate_paraphrase_candidate(
                    sample.metadata.instruction + suffix,
                    gold=sample.gold,
                    world=world,
                    ontology=self.generator.ontology,
                    lexicon=self.generator.lexicon,
                )
                self.assertFalse(result.accepted)
                self.assertIn("FORBIDDEN_RUNTIME_DATA", result.reasons)

    def test_all_official_instructions_are_grounded_in_gold(self) -> None:
        for split, samples in self.first.samples_by_split.items():
            for sample in samples:
                result = validate_paraphrase_candidate(
                    sample.metadata.instruction,
                    gold=sample.gold,
                    world=self.first.worlds[sample.world_context_id],
                    ontology=self.generator.ontology,
                    lexicon=self.generator.lexicon,
                    allow_approved_robustness=(split == "test_robustness"),
                )
                self.assertTrue(result.accepted, (sample.sample_id, result.reasons))

    def test_candidate_jsonl_parser_rejects_duplicate_keys_and_non_finite_numbers(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            strict_candidate_json_loads(
                '{"sample_id":"a","sample_id":"b","candidate_instruction":"x"}'
            )
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
                    strict_candidate_json_loads(
                        '{"sample_id":"a","candidate_instruction":"x","extra":'
                        + constant
                        + "}"
                    )

    def test_atomic_writer_refuses_overwrite_and_emits_only_two_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "planner_v1"
            manifest = write_generated_dataset(self.first, root)
            self.assertEqual(manifest.split_counts["train"], 1000)
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                [
                    "checksums.sha256",
                    "dataset_manifest.json",
                    "statistics.json",
                    "test_compositional.jsonl",
                    "test_iid.jsonl",
                    "test_language.jsonl",
                    "test_robustness.jsonl",
                    "train.jsonl",
                    "validation.jsonl",
                ],
            )
            with self.assertRaises(FileExistsError):
                write_generated_dataset(self.first, root)
            write_generated_dataset(self.first, root, overwrite=True)
            self.assertFalse(any(path.name.endswith(".tmp") for path in root.parent.iterdir()))

    def test_atomic_overwrite_never_touches_a_fixed_old_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "planner_v1"
            write_generated_dataset(self.first, root)
            unrelated = parent / ".planner_v1.old"
            unrelated.mkdir()
            sentinel = unrelated / "user-work.txt"
            sentinel.write_text("keep me", encoding="utf-8")
            write_generated_dataset(self.first, root, overwrite=True)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")
            self.assertFalse(any(path.name.endswith(".backup") for path in parent.iterdir()))

    def test_overwrite_refuses_to_delete_unreviewed_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "planner_v1"
            write_generated_dataset(self.first, root)
            candidate = root / "_candidates" / "work.jsonl"
            candidate.parent.mkdir()
            candidate.write_text("local work\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetGenerationError, "contains _candidates"):
                write_generated_dataset(self.first, root, overwrite=True)
            self.assertEqual(candidate.read_text(encoding="utf-8"), "local work\n")

    def test_same_dataset_written_twice_has_identical_public_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            write_generated_dataset(self.first, first)
            write_generated_dataset(self.first, second)
            self.assertEqual(
                (first / "checksums.sha256").read_bytes(),
                (second / "checksums.sha256").read_bytes(),
            )

    def test_failed_overwrite_restores_previous_official_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "planner_v1"
            write_generated_dataset(self.first, root)
            original = (root / "checksums.sha256").read_bytes()
            replacement = self.generator.generate(seed=43, profile="pilot")
            import planner_data.generator as generator_module

            real_replace = generator_module.os.replace

            def fail_directory_publish(source, destination):
                if Path(destination) == root and Path(source).name.endswith(".tmp"):
                    raise OSError("injected directory publish failure")
                return real_replace(source, destination)

            with mock.patch.object(generator_module.os, "replace", side_effect=fail_directory_publish):
                with self.assertRaisesRegex(OSError, "injected"):
                    write_generated_dataset(replacement, root, overwrite=True)
            self.assertEqual((root / "checksums.sha256").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
