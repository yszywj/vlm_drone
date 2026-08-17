from __future__ import annotations

from dataclasses import replace
import unittest

from planner_data.generator import PlannerDatasetGenerator
from planner_data.leakage_checker import (
    DatasetLeakageError,
    check_dataset_leakage,
    check_split_seed_sets,
    normalize_instruction,
)


class LeakageCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = PlannerDatasetGenerator().generate(seed=42, profile="pilot")
        cls.samples = tuple(
            sample
            for split_samples in dataset.samples_by_split.values()
            for sample in split_samples
        )

    def test_generated_pilot_has_no_leakage(self) -> None:
        report = check_dataset_leakage(self.samples)
        self.assertTrue(report.valid, report.to_dict())

    def test_normalization_detects_typographic_duplicate(self) -> None:
        self.assertEqual(normalize_instruction("请 去东区。"), normalize_instruction("请?去东区。"))

    def test_duplicate_instruction_is_reported(self) -> None:
        first = self.samples[0]
        second = self.samples[1000]
        changed_metadata = replace(second.metadata, instruction=first.metadata.instruction)
        changed = replace(second, metadata=changed_metadata)
        report = check_dataset_leakage((first, changed))
        self.assertIn("DUPLICATE_NORMALIZED_INSTRUCTION", {issue.code for issue in report.issues})
        with self.assertRaises(DatasetLeakageError):
            report.require_valid()

    def test_forged_stored_semantic_family_is_reported(self) -> None:
        first = self.samples[0]
        second = self.samples[1000]
        changed_metadata = replace(
            second.metadata,
            semantic_spec_family=first.metadata.semantic_spec_family,
        )
        report = check_dataset_leakage((first, replace(second, metadata=changed_metadata)))
        self.assertIn(
            "SEMANTIC_SPEC_FAMILY_MISMATCH",
            {issue.code for issue in report.issues},
        )

    def test_forged_metadata_cannot_hide_cross_split_gold_leakage(self) -> None:
        first = self.samples[0]
        second = self.samples[1000]
        shared_gold = replace(first.gold, spec_id=second.gold_spec_id)
        forged = replace(
            second,
            world_context_id=first.world_context_id,
            gold=shared_gold,
        )

        report = check_dataset_leakage((first, forged))
        codes = {issue.code for issue in report.issues}
        self.assertIn("SEMANTIC_SPEC_FAMILY_MISMATCH", codes)
        self.assertIn("GROUP_ID_MISMATCH", codes)
        self.assertIn("SEMANTIC_GROUP_LEAKAGE", codes)

    def test_forged_group_id_is_reported(self) -> None:
        first = self.samples[0]
        changed = replace(first, metadata=replace(first.metadata, group_id="group_forged"))
        report = check_dataset_leakage((changed,))
        self.assertIn("GROUP_ID_MISMATCH", {issue.code for issue in report.issues})

    def test_template_skeleton_family_cannot_cross_splits(self) -> None:
        first = self.samples[0]
        second = self.samples[1000]
        changed_metadata = replace(
            second.metadata,
            template_family=first.metadata.template_family,
        )
        report = check_dataset_leakage((first, replace(second, metadata=changed_metadata)))
        self.assertIn(
            "TEMPLATE_FAMILY_LEAKAGE",
            {issue.code for issue in report.issues},
        )

    def test_generation_seed_sets_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(DatasetLeakageError, "both train and validation"):
            check_split_seed_sets({"train": [10], "validation": [10]})


if __name__ == "__main__":
    unittest.main()
