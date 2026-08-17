from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from planner_data.schemas import compute_semantic_spec_family, make_group_id
from planner_data.splitter import (
    SplitterError,
    assign_groups,
    assign_splits,
    group_key,
    normalize_instruction,
    validate_holdout_invariants,
    validate_splits,
)
from tasks.schemas import GoldPlannerSpec


@dataclass(frozen=True, slots=True)
class FakeMetadata:
    instruction: str
    template_family: str
    paraphrase_family: str
    seed: int
    semantic_spec_family: str
    group_id: str
    difficulty: str = "medium"

    @property
    def group_key(self) -> str:
        return "|".join(
            (
                self.semantic_spec_family,
                self.template_family,
                self.paraphrase_family,
            )
        )


@dataclass(frozen=True, slots=True)
class FakeSample:
    sample_id: str
    split: str
    world_context_id: str
    gold: GoldPlannerSpec
    metadata: FakeMetadata


def make_sample(
    index: int,
    *,
    split: str = "train",
    semantic: str | None = None,
    concept: str | None = None,
    instruction: str | None = None,
    template: str = "standard",
    paraphrase: str = "common",
    group_id: str | None = None,
    seed: int | None = None,
    difficulty: str = "medium",
) -> FakeSample:
    concept_value = concept or semantic or f"concept_{index}"
    world_context_id = "world_default"
    gold = GoldPlannerSpec(
        spec_id=f"gold_{index}",
        target_concept_id=concept_value,
        target_description=f"目标 {concept_value}",
        search_region="east_area",
        track_duration_s=30.0,
        landing_zone="home",
        takeoff_altitude_m=None,
        explicit_fields=frozenset(
            {"target_description", "search_region", "landing_zone"}
        ),
    )
    semantic_value = compute_semantic_spec_family(gold, world_context_id)
    actual_group_id = make_group_id(semantic_value, template, paraphrase)
    return FakeSample(
        sample_id=f"sample_{index}",
        split=split,
        world_context_id=world_context_id,
        gold=gold,
        metadata=FakeMetadata(
            instruction=instruction or f"执行第 {index} 个任务",
            template_family=template,
            paraphrase_family=paraphrase,
            seed=index if seed is None else seed,
            semantic_spec_family=semantic_value,
            group_id=group_id or actual_group_id,
            difficulty=difficulty,
        ),
    )


class SplitterTests(unittest.TestCase):
    def test_group_key_uses_schema_contract(self) -> None:
        sample = make_sample(1, semantic="spec", template="order_a", paraphrase="p1")
        self.assertEqual(
            group_key(sample),
            "\x1f".join(
                (sample.metadata.semantic_spec_family, "order_a", "p1")
            ),
        )

    def test_prelabelled_exact_counts_are_validated_and_copied(self) -> None:
        samples = (
            make_sample(1, split="train"),
            make_sample(2, split="train"),
            make_sample(3, split="validation"),
        )
        result = assign_splits(
            samples,
            {"train": 2, "validation": 1},
            enforce_holdouts=False,
        ).as_dict()

        self.assertEqual(len(result["train"]), 2)
        self.assertEqual(len(result["validation"]), 1)
        self.assertTrue(all(sample.split == "train" for sample in result["train"]))
        self.assertIsNot(result["train"][0], samples[0])

    def test_generic_group_assignment_is_deterministic(self) -> None:
        samples = tuple(make_sample(index) for index in range(8))
        counts = {"train": 3, "validation": 3, "test_iid": 2}

        first = assign_groups(samples, counts, seed=42)
        second = assign_groups(tuple(reversed(samples)), counts, seed=42)

        first_ids = {
            name: tuple(sample.sample_id for sample in members)
            for name, members in first.items()
        }
        second_ids = {
            name: tuple(sample.sample_id for sample in members)
            for name, members in second.items()
        }
        self.assertEqual(first_ids, second_ids)
        self.assertEqual({name: len(value) for name, value in first.items()}, counts)
        for split_name, members in first.items():
            self.assertTrue(all(member.split == split_name for member in members))

    def test_same_gold_group_never_crosses_splits(self) -> None:
        samples = (
            make_sample(1, semantic="shared", instruction="共享任务甲"),
            make_sample(2, semantic="shared", instruction="共享任务乙"),
            make_sample(3),
            make_sample(4),
        )
        result = assign_groups(samples, {"train": 2, "validation": 2}, seed=7)
        owners = {
            split_name
            for split_name, members in result.items()
            if any(
                member.metadata.semantic_spec_family
                == samples[0].metadata.semantic_spec_family
                for member in members
            )
        }
        self.assertEqual(len(owners), 1)

    def test_conflicting_prelabel_for_same_semantic_group_is_rejected(self) -> None:
        samples = (
            make_sample(1, split="train", semantic="shared"),
            make_sample(2, split="validation", semantic="shared"),
        )
        with self.assertRaisesRegex(SplitterError, "multiple splits"):
            assign_splits(
                samples,
                {"train": 1, "validation": 1},
                enforce_holdouts=False,
            )

    def test_impossible_exact_group_sizes_fail_explicitly(self) -> None:
        samples = (
            make_sample(1, semantic="pair_a"),
            make_sample(2, semantic="pair_a"),
            make_sample(3, semantic="pair_b"),
            make_sample(4, semantic="pair_b"),
        )
        with self.assertRaisesRegex(SplitterError, "cannot satisfy exact"):
            assign_groups(samples, {"train": 3, "validation": 1}, seed=5)

    def test_requested_count_mismatch_fails_without_downgrade(self) -> None:
        with self.assertRaisesRegex(SplitterError, "requested 3 samples"):
            assign_groups(
                (make_sample(1), make_sample(2)),
                {"train": 2, "validation": 1},
            )

    def test_semantic_group_leakage_is_rejected(self) -> None:
        splits = {
            "train": (make_sample(1, split="train", semantic="shared"),),
            "validation": (
                make_sample(2, split="validation", semantic="shared"),
            ),
        }
        with self.assertRaisesRegex(SplitterError, "semantic Gold group"):
            validate_splits(splits, enforce_holdouts=False)

    def test_forged_group_id_is_rejected(self) -> None:
        splits = {
            "train": (
                make_sample(1, split="train", semantic="a", group_id="rewrite_1"),
            ),
            "validation": (
                make_sample(
                    2,
                    split="validation",
                    semantic="b",
                    group_id="rewrite_1",
                ),
            ),
        }
        with self.assertRaisesRegex(SplitterError, "stored group_id"):
            validate_splits(splits, enforce_holdouts=False)

    def test_forged_semantic_metadata_cannot_hide_cross_split_gold(self) -> None:
        first = make_sample(1, split="train", semantic="shared")
        second = make_sample(2, split="validation", semantic="shared")
        forged_family = "semantic_00000000000000000000"
        forged = replace(
            second,
            metadata=replace(
                second.metadata,
                semantic_spec_family=forged_family,
                group_id=make_group_id(
                    forged_family,
                    second.metadata.template_family,
                    second.metadata.paraphrase_family,
                ),
            ),
        )
        with self.assertRaisesRegex(SplitterError, "does not match Gold-derived"):
            validate_splits(
                {"train": (first,), "validation": (forged,)},
                enforce_holdouts=False,
            )

    def test_generation_seed_leakage_is_rejected(self) -> None:
        splits = {
            "train": (make_sample(1, split="train", seed=99),),
            "validation": (make_sample(2, split="validation", seed=99),),
        }
        with self.assertRaisesRegex(SplitterError, "generation seed"):
            validate_splits(splits, enforce_holdouts=False)

    def test_normalized_instruction_duplicate_is_rejected(self) -> None:
        self.assertEqual(normalize_instruction("请执行，任务 A。"), "请执行任务a")
        splits = {
            "train": (
                make_sample(1, split="train", instruction="请执行，任务 A。"),
            ),
            "validation": (
                make_sample(2, split="validation", instruction="请执行 任务 a"),
            ),
        }
        with self.assertRaisesRegex(SplitterError, "normalized instruction duplicate"):
            validate_splits(splits, enforce_holdouts=False)

    def test_compositional_concept_must_be_held_out(self) -> None:
        valid = {
            "train": (
                make_sample(1, concept="red_blue"),
                make_sample(2, concept="blue_black"),
            ),
            "test_compositional": (
                make_sample(
                    3,
                    split="test_compositional",
                    concept="red_black",
                    paraphrase="composition",
                ),
            ),
        }
        validate_holdout_invariants(valid)

        invalid = {
            **valid,
            "test_compositional": (
                make_sample(
                    4,
                    split="test_compositional",
                    concept="red_blue",
                    paraphrase="composition_2",
                ),
            ),
        }
        with self.assertRaisesRegex(SplitterError, "present in train"):
            validate_holdout_invariants(invalid)

    def test_language_paraphrase_family_must_be_held_out(self) -> None:
        valid = {
            "train": (make_sample(1, paraphrase="common"),),
            "test_language": (
                make_sample(2, split="test_language", paraphrase="heldout_alias"),
            ),
        }
        validate_holdout_invariants(valid)

        invalid = {
            "train": (make_sample(1, paraphrase="common"),),
            "test_language": (
                make_sample(2, split="test_language", paraphrase="common"),
            ),
        }
        with self.assertRaisesRegex(SplitterError, "no feature held out"):
            validate_holdout_invariants(invalid)

    def test_robustness_markers_are_isolated(self) -> None:
        valid = {
            "train": (make_sample(1),),
            "test_robustness": (
                make_sample(
                    2,
                    split="test_robustness",
                    difficulty="robustness",
                    template="prompt_injection",
                ),
            ),
        }
        validate_holdout_invariants(valid)

        unmarked = {
            "train": (make_sample(1),),
            "test_robustness": (make_sample(2, split="test_robustness"),),
        }
        with self.assertRaisesRegex(SplitterError, "lacks an explicit"):
            validate_holdout_invariants(unmarked)

        misplaced = {
            "train": (make_sample(3, difficulty="robustness"),),
        }
        with self.assertRaisesRegex(SplitterError, "non-robust split"):
            validate_holdout_invariants(misplaced)


if __name__ == "__main__":
    unittest.main()
