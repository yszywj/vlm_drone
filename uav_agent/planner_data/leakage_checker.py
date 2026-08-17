"""Cross-split leakage checks for Planner dataset v1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import unicodedata

from .schemas import (
    PLANNER_DATASET_SPLITS,
    PlannerDatasetSample,
    compute_semantic_spec_family,
    make_group_id,
)


class DatasetLeakageError(ValueError):
    """Raised when samples violate a declared split boundary."""


@dataclass(frozen=True, slots=True)
class LeakageIssue:
    code: str
    message: str
    sample_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "sample_ids": list(self.sample_ids),
        }


@dataclass(frozen=True, slots=True)
class LeakageReport:
    num_samples: int
    issues: tuple[LeakageIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def require_valid(self) -> None:
        if self.issues:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
            raise DatasetLeakageError(details)

    def to_dict(self) -> dict[str, object]:
        return {
            "num_samples": self.num_samples,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def normalize_instruction(instruction: str) -> str:
    """Normalize spelling, whitespace and punctuation for exact duplicate checks."""

    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    normalized = unicodedata.normalize("NFKC", instruction).casefold()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z"))
        and not character.isspace()
    )


def check_dataset_leakage(
    samples: Iterable[PlannerDatasetSample],
) -> LeakageReport:
    """Return all deterministic cross-split leakage violations."""

    materialized = tuple(samples)
    issues: list[LeakageIssue] = []
    by_id: dict[str, list[PlannerDatasetSample]] = defaultdict(list)
    by_instruction: dict[str, list[PlannerDatasetSample]] = defaultdict(list)
    by_semantic: dict[str, list[PlannerDatasetSample]] = defaultdict(list)
    by_group: dict[str, list[PlannerDatasetSample]] = defaultdict(list)
    by_template_family: dict[str, list[PlannerDatasetSample]] = defaultdict(list)
    by_seed: dict[int, list[PlannerDatasetSample]] = defaultdict(list)

    for sample in materialized:
        if not isinstance(sample, PlannerDatasetSample):
            raise TypeError("samples must contain PlannerDatasetSample objects")
        by_id[sample.sample_id].append(sample)
        normalized = normalize_instruction(sample.metadata.instruction)
        if not normalized:
            issues.append(
                LeakageIssue(
                    "EMPTY_NORMALIZED_INSTRUCTION",
                    f"instruction for {sample.sample_id!r} becomes empty",
                    (sample.sample_id,),
                )
            )
        else:
            by_instruction[normalized].append(sample)
        computed_semantic = compute_semantic_spec_family(
            sample.gold,
            sample.world_context_id,
        )
        if sample.metadata.semantic_spec_family != computed_semantic:
            issues.append(
                LeakageIssue(
                    "SEMANTIC_SPEC_FAMILY_MISMATCH",
                    "stored semantic_spec_family "
                    f"{sample.metadata.semantic_spec_family!r} does not match "
                    f"Gold-derived {computed_semantic!r}",
                    (sample.sample_id,),
                )
            )
        by_semantic[computed_semantic].append(sample)
        computed_group = make_group_id(
            computed_semantic,
            sample.metadata.template_family,
            sample.metadata.paraphrase_family,
        )
        if sample.metadata.group_id != computed_group:
            issues.append(
                LeakageIssue(
                    "GROUP_ID_MISMATCH",
                    f"stored group_id {sample.metadata.group_id!r} does not match "
                    f"computed {computed_group!r}",
                    (sample.sample_id,),
                )
            )
        by_group[computed_group].append(sample)
        by_template_family[sample.metadata.template_family].append(sample)
        by_seed[sample.metadata.seed].append(sample)

    _append_duplicate_issues(issues, by_id, "DUPLICATE_SAMPLE_ID", "sample ID")
    _append_duplicate_issues(
        issues,
        by_instruction,
        "DUPLICATE_NORMALIZED_INSTRUCTION",
        "normalized instruction",
    )
    _append_cross_split_issues(
        issues,
        by_semantic,
        "SEMANTIC_GROUP_LEAKAGE",
        "semantic spec family",
    )
    _append_cross_split_issues(
        issues,
        by_group,
        "PARAPHRASE_GROUP_LEAKAGE",
        "semantic/template/paraphrase group",
    )
    # A template family denotes one sentence skeleton and its light variants.
    # Keeping it split-local prevents a familiar skeleton from dominating both
    # train and held-out evaluation, even when the Gold semantics differ.
    _append_cross_split_issues(
        issues,
        by_template_family,
        "TEMPLATE_FAMILY_LEAKAGE",
        "template family",
    )
    _append_cross_split_issues(
        issues,
        by_seed,
        "SEED_LEAKAGE",
        "generation seed",
    )

    train = [sample for sample in materialized if sample.split == "train"]
    train_concepts = {sample.gold.target_concept_id for sample in train}
    train_paraphrases = {sample.metadata.paraphrase_family for sample in train}
    train_templates = {sample.metadata.template_family for sample in train}

    for sample in materialized:
        if sample.split == "test_compositional" and sample.gold.target_concept_id in train_concepts:
            issues.append(
                LeakageIssue(
                    "COMPOSITIONAL_COMBINATION_SEEN_IN_TRAIN",
                    f"concept {sample.gold.target_concept_id!r} is present in train",
                    (sample.sample_id,),
                )
            )
        if sample.split == "test_language" and sample.metadata.paraphrase_family in train_paraphrases:
            issues.append(
                LeakageIssue(
                    "LANGUAGE_FAMILY_SEEN_IN_TRAIN",
                    f"paraphrase family {sample.metadata.paraphrase_family!r} is present in train",
                    (sample.sample_id,),
                )
            )
        if sample.split == "test_robustness":
            if sample.metadata.difficulty != "robustness":
                issues.append(
                    LeakageIssue(
                        "ROBUSTNESS_SAMPLE_NOT_MARKED",
                        "test_robustness sample must use robustness difficulty",
                        (sample.sample_id,),
                    )
                )
            if sample.metadata.template_family in train_templates:
                issues.append(
                    LeakageIssue(
                        "ROBUSTNESS_TEMPLATE_SEEN_IN_TRAIN",
                        f"robustness template {sample.metadata.template_family!r} is present in train",
                        (sample.sample_id,),
                    )
                )

    issues.sort(key=lambda issue: (issue.code, issue.sample_ids, issue.message))
    return LeakageReport(num_samples=len(materialized), issues=tuple(issues))


def check_split_seed_sets(split_seed_sets: Mapping[str, Iterable[int]]) -> None:
    """Validate mutually exclusive declared generation seed pools."""

    unknown = set(split_seed_sets) - set(PLANNER_DATASET_SPLITS)
    if unknown:
        raise DatasetLeakageError("unknown split seed sets: " + ", ".join(sorted(unknown)))
    owners: dict[int, str] = {}
    for split in PLANNER_DATASET_SPLITS:
        for seed in split_seed_sets.get(split, ()):
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("generation seeds must be integers")
            previous = owners.get(seed)
            if previous is not None and previous != split:
                raise DatasetLeakageError(
                    f"generation seed {seed} appears in both {previous} and {split}"
                )
            owners[seed] = split


def _append_duplicate_issues(
    issues: list[LeakageIssue],
    groups: Mapping[object, list[PlannerDatasetSample]],
    code: str,
    label: str,
) -> None:
    for key, group in groups.items():
        if len(group) > 1:
            issues.append(
                LeakageIssue(
                    code,
                    f"{label} {key!r} occurs {len(group)} times",
                    tuple(sorted(sample.sample_id for sample in group)),
                )
            )


def _append_cross_split_issues(
    issues: list[LeakageIssue],
    groups: Mapping[object, list[PlannerDatasetSample]],
    code: str,
    label: str,
) -> None:
    for key, group in groups.items():
        splits = {sample.split for sample in group}
        if len(splits) > 1:
            issues.append(
                LeakageIssue(
                    code,
                    f"{label} {key!r} crosses splits {sorted(splits)}",
                    tuple(sorted(sample.sample_id for sample in group)),
                )
            )


__all__ = [
    "DatasetLeakageError",
    "LeakageIssue",
    "LeakageReport",
    "check_dataset_leakage",
    "check_split_seed_sets",
    "normalize_instruction",
]
