"""Deterministic, leakage-aware splitting for Planner datasets.

The splitter deliberately operates on a small structural interface rather than
depending on the concrete dataset dataclasses.  A sample may be either a
mapping or an object with attributes.  In both cases it is expected to expose
``sample_id``, ``gold`` and ``metadata``.  This keeps the splitting rules easy
to test and makes them reusable by the generator and the standalone validator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from hashlib import sha256
import json
import re
import unicodedata
from typing import Generic, TypeVar

from .schemas import compute_semantic_spec_family, make_group_id


DEFAULT_SPLIT_ORDER = (
    "train",
    "validation",
    "test_iid",
    "test_compositional",
    "test_language",
    "test_robustness",
)

_MISSING = object()
_ROBUSTNESS_FIELDS = (
    "robustness_type",
    "robustness_category",
    "adversarial_type",
    "prompt_injection",
    "is_robustness",
)
_LANGUAGE_FEATURE_FIELDS = (
    "region_alias",
    "region_alias_id",
    "landing_alias",
    "landing_alias_id",
    "target_alias",
    "target_alias_id",
    "duration_alias",
    "duration_expression",
    "instruction_order",
    "politeness",
    "language_variant_id",
)

T = TypeVar("T")


class SplitterError(ValueError):
    """Raised when an exact, leakage-free split cannot be produced."""


@dataclass(frozen=True, slots=True)
class SplitValidationReport:
    """Compact immutable summary returned after split validation."""

    counts: tuple[tuple[str, int], ...]
    num_samples: int
    num_semantic_groups: int
    num_generation_seeds: int

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": dict(self.counts),
            "num_samples": self.num_samples,
            "num_semantic_groups": self.num_semantic_groups,
            "num_generation_seeds": self.num_generation_seeds,
        }


@dataclass(frozen=True, slots=True)
class SplitAssignment(Generic[T]):
    """A deterministic split result without mutating the input samples."""

    splits: tuple[tuple[str, tuple[T, ...]], ...]

    def as_dict(self) -> dict[str, tuple[T, ...]]:
        return {name: samples for name, samples in self.splits}


def normalize_instruction(instruction: str) -> str:
    """Return the canonical form used for exact duplicate detection.

    NFKC removes harmless width/compatibility differences.  Separators,
    punctuation and whitespace are ignored so superficial typography cannot be
    used to move an otherwise identical instruction across a split boundary.
    """

    if not isinstance(instruction, str):
        raise SplitterError("instruction must be a string")
    normalized = unicodedata.normalize("NFKC", instruction).casefold()
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z"))
        and not character.isspace()
    )
    if not normalized:
        raise SplitterError("instruction must be non-empty")
    return normalized


def semantic_spec_key(sample: object) -> str:
    """Return Gold-derived semantics and reject forged stored metadata."""

    metadata = _metadata(sample)
    gold = _field(sample, "gold", _MISSING)
    context_id = _field(sample, "world_context_id", _MISSING)
    if gold is _MISSING or gold is None or context_id is _MISSING:
        raise SplitterError(
            f"sample {_sample_label(sample)!r} must define Gold and world_context_id"
        )
    try:
        computed = compute_semantic_spec_family(gold, context_id)
    except (TypeError, ValueError) as exc:
        raise SplitterError(
            f"sample {_sample_label(sample)!r} has invalid semantic Gold: {exc}"
        ) from exc

    stored = _field(metadata, "semantic_spec_family", _MISSING)
    if stored is _MISSING:
        raise SplitterError(
            f"sample {_sample_label(sample)!r} has no stored semantic_spec_family"
        )
    normalized_stored = _non_empty_text(stored, "semantic_spec_family")
    if normalized_stored != computed:
        raise SplitterError(
            f"sample {_sample_label(sample)!r} stored semantic_spec_family "
            f"{normalized_stored!r} does not match Gold-derived {computed!r}"
        )
    return computed


def group_key(sample: object) -> str:
    """Build the configured group key for near-paraphrase colocation.

    The prescribed key is ``semantic_spec_family + template_family +
    paraphrase_family``.  It is always rebuilt from Gold semantics; an
    attacker-controlled metadata ``group_key`` is never trusted.
    """

    metadata = _metadata(sample)
    template = _field(metadata, "template_family", _MISSING)
    paraphrase = _field(metadata, "paraphrase_family", _MISSING)
    if template is _MISSING or paraphrase is _MISSING:
        raise SplitterError(
            f"sample {_sample_label(sample)!r} must define template_family and "
            "paraphrase_family"
        )
    return "\x1f".join(
        (
            semantic_spec_key(sample),
            _non_empty_text(template, "template_family"),
            _non_empty_text(paraphrase, "paraphrase_family"),
        )
    )


def assign_splits(
    samples: Sequence[T],
    split_counts: Mapping[str, int],
    *,
    seed: int = 0,
    enforce_holdouts: bool = True,
    respect_existing_split: bool = True,
) -> SplitAssignment[T]:
    """Assign complete semantic groups to splits with exact requested counts.

    By default, a sample requests its prelabelled top-level ``split``; this is
    the Gold-first dataset path because holdout semantics are authored before
    splitting.  A sample can additionally request a split with metadata
    ``split_hint``/``intended_split``/``required_split``.  Holdout metadata can
    also force one of the three specialized test splits.  Remaining groups are
    assigned in a hash-seeded deterministic order.  Inputs are never mutated.

    The assignment unit is the complete Gold semantic key, which is stricter
    than the minimum near-paraphrase group and therefore guarantees that all
    variants of one Gold task remain together.
    """

    normalized_counts = _validate_requested_counts(split_counts)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    materialized = tuple(samples)
    requested_total = sum(normalized_counts.values())
    if len(materialized) != requested_total:
        raise SplitterError(
            f"requested {requested_total} samples but received {len(materialized)}"
        )

    semantic_groups: dict[str, list[T]] = {}
    for sample in materialized:
        # Calling group_key here validates the required grouping metadata even
        # though the stronger semantic key is the physical assignment unit.
        group_key(sample)
        semantic_groups.setdefault(semantic_spec_key(sample), []).append(sample)

    group_records: list[tuple[str, tuple[T, ...], str | None]] = []
    for key, members in semantic_groups.items():
        ordered_members = tuple(sorted(members, key=_sample_sort_key))
        hints = {
            _split_hint(
                member,
                normalized_counts,
                respect_existing_split=respect_existing_split,
            )
            for member in members
        }
        hints.discard(None)
        if len(hints) > 1:
            raise SplitterError(
                f"semantic group {key!r} requests multiple splits: {sorted(hints)!r}"
            )
        group_records.append(
            (key, ordered_members, next(iter(hints)) if hints else None)
        )

    assignments: dict[str, list[T]] = {name: [] for name in normalized_counts}
    remaining = dict(normalized_counts)
    unforced: list[tuple[str, tuple[T, ...]]] = []
    for key, members, hint in group_records:
        if hint is None:
            unforced.append((key, members))
            continue
        size = len(members)
        if size > remaining[hint]:
            raise SplitterError(
                f"forced semantic group {key!r} has {size} samples but split "
                f"{hint!r} has only {remaining[hint]} slots remaining"
            )
        assignments[hint].extend(
            _replace_sample_split(member, hint) for member in members
        )
        remaining[hint] -= size

    placements = _place_unforced_groups(unforced, remaining, seed)
    for split_name, groups in placements.items():
        for _, members in groups:
            assignments[split_name].extend(
                _replace_sample_split(member, split_name) for member in members
            )

    result_dict = {
        name: tuple(sorted(members, key=_sample_sort_key))
        for name, members in assignments.items()
    }
    validate_splits(
        result_dict,
        expected_counts=normalized_counts,
        enforce_holdouts=enforce_holdouts,
    )
    return SplitAssignment(tuple((name, result_dict[name]) for name in normalized_counts))


def split_samples(
    samples: Sequence[T],
    split_counts: Mapping[str, int],
    *,
    seed: int = 0,
    enforce_holdouts: bool = True,
    respect_existing_split: bool = True,
) -> dict[str, tuple[T, ...]]:
    """Convenience wrapper returning a regular mapping."""

    return assign_splits(
        samples,
        split_counts,
        seed=seed,
        enforce_holdouts=enforce_holdouts,
        respect_existing_split=respect_existing_split,
    ).as_dict()


def assign_groups(
    samples: Sequence[T],
    split_counts: Mapping[str, int],
    *,
    seed: int = 0,
    enforce_holdouts: bool = False,
) -> dict[str, tuple[T, ...]]:
    """Generically assign unlabelled groups, ignoring existing ``split`` values.

    Dataset generation normally uses :func:`assign_splits` to validate the
    deliberate holdout labels.  This helper is useful for tests and for future
    non-holdout partitions where deterministic bin assignment is desired.
    """

    return split_samples(
        samples,
        split_counts,
        seed=seed,
        enforce_holdouts=enforce_holdouts,
        respect_existing_split=False,
    )


def validate_splits(
    splits: Mapping[str, Sequence[object]],
    *,
    expected_counts: Mapping[str, int] | None = None,
    enforce_holdouts: bool = True,
) -> SplitValidationReport:
    """Validate exact counts and all cross-split leakage invariants."""

    if not isinstance(splits, Mapping) or not splits:
        raise SplitterError("splits must be a non-empty mapping")
    split_names = tuple(splits)
    if any(not isinstance(name, str) or not name for name in split_names):
        raise SplitterError("split names must be non-empty strings")

    if expected_counts is not None:
        counts = _validate_requested_counts(expected_counts)
        if set(counts) != set(split_names):
            raise SplitterError(
                "actual split names do not match the requested split names"
            )
        for name, expected in counts.items():
            actual = len(splits[name])
            if actual != expected:
                raise SplitterError(
                    f"split {name!r} has {actual} samples; expected {expected}"
                )

    sample_owners: dict[str, str] = {}
    instruction_owners: dict[str, tuple[str, str]] = {}
    semantic_owners: dict[str, str] = {}
    group_owners: dict[str, str] = {}
    paraphrase_group_owners: dict[str, str] = {}
    seed_owners: dict[int, str] = {}

    total = 0
    for split_name, split_samples_value in splits.items():
        if isinstance(split_samples_value, (str, bytes)) or not isinstance(
            split_samples_value, Sequence
        ):
            raise SplitterError(f"split {split_name!r} must contain a sequence")
        for sample in split_samples_value:
            total += 1
            sample_id = _sample_id(sample)
            declared_split = _field(sample, "split", _MISSING)
            if declared_split is not _MISSING and declared_split != split_name:
                raise SplitterError(
                    f"sample {sample_id!r} declares split {declared_split!r} but "
                    f"is stored under {split_name!r}"
                )
            _claim_unique(sample_owners, sample_id, split_name, "sample ID")

            instruction = _instruction(sample)
            normalized = normalize_instruction(instruction)
            previous_instruction = instruction_owners.get(normalized)
            if previous_instruction is not None:
                previous_split, previous_id = previous_instruction
                raise SplitterError(
                    "normalized instruction duplicate: "
                    f"{previous_id!r} ({previous_split}) and {sample_id!r} "
                    f"({split_name})"
                )
            instruction_owners[normalized] = (split_name, sample_id)

            semantic = semantic_spec_key(sample)
            _claim_cross_split(
                semantic_owners, semantic, split_name, "semantic Gold group"
            )
            near_group = group_key(sample)
            _claim_cross_split(group_owners, near_group, split_name, "group key")

            paraphrase_group = _paraphrase_group_key(sample)
            _claim_cross_split(
                paraphrase_group_owners,
                paraphrase_group,
                split_name,
                "paraphrase group",
            )

            generation_seed = _generation_seed(sample)
            _claim_cross_split(
                seed_owners, generation_seed, split_name, "generation seed"
            )

    if enforce_holdouts:
        validate_holdout_invariants(splits)

    ordered_counts = tuple((name, len(splits[name])) for name in split_names)
    return SplitValidationReport(
        counts=ordered_counts,
        num_samples=total,
        num_semantic_groups=len(semantic_owners),
        num_generation_seeds=len(seed_owners),
    )


def validate_holdout_invariants(
    splits: Mapping[str, Sequence[object]],
) -> None:
    """Validate compositional, language and robustness holdout semantics."""

    train = tuple(splits.get("train", ()))

    compositional = tuple(splits.get("test_compositional", ()))
    if compositional:
        if not train:
            raise SplitterError(
                "test_compositional requires train samples for holdout validation"
            )
        train_combinations = {_composition_key(sample) for sample in train}
        train_components: set[str] = set()
        for sample in train:
            train_components.update(_composition_components(sample))
        for sample in compositional:
            combination = _composition_key(sample)
            if combination in train_combinations:
                raise SplitterError(
                    f"compositional holdout {combination!r} is present in train"
                )
            components = _composition_components(sample)
            if components and not set(components).issubset(train_components):
                missing = sorted(set(components) - train_components)
                raise SplitterError(
                    f"compositional sample {_sample_label(sample)!r} uses unseen "
                    f"individual attributes: {missing!r}"
                )

    language = tuple(splits.get("test_language", ()))
    if language:
        if not train:
            raise SplitterError(
                "test_language requires train samples for holdout validation"
            )
        train_features: set[str] = set()
        for sample in train:
            train_features.update(_language_features(sample))
        for sample in language:
            features = _language_features(sample)
            if not features:
                raise SplitterError(
                    f"language holdout sample {_sample_label(sample)!r} has no "
                    "language feature metadata"
                )
            if set(features).issubset(train_features):
                raise SplitterError(
                    f"language holdout sample {_sample_label(sample)!r} has no "
                    "feature held out from train"
                )

    for split_name, samples in splits.items():
        train_templates = {
            _non_empty_text(
                _field(_metadata(sample), "template_family", _MISSING),
                "template_family",
            )
            for sample in train
        }
        for sample in samples:
            marker = _robustness_marker(sample)
            if split_name == "test_robustness":
                if marker is None:
                    raise SplitterError(
                        f"robustness sample {_sample_label(sample)!r} lacks an "
                        "explicit robustness category"
                    )
                template = _non_empty_text(
                    _field(_metadata(sample), "template_family", _MISSING),
                    "template_family",
                )
                if template in train_templates:
                    raise SplitterError(
                        f"robustness template {template!r} is present in train"
                    )
            elif marker is not None:
                raise SplitterError(
                    f"robustness sample {_sample_label(sample)!r} appears in "
                    f"non-robust split {split_name!r}"
                )


def _place_unforced_groups(
    groups: Sequence[tuple[str, tuple[T, ...]]],
    remaining: Mapping[str, int],
    seed: int,
) -> dict[str, list[tuple[str, tuple[T, ...]]]]:
    placements: dict[str, list[tuple[str, tuple[T, ...]]]] = {
        split: [] for split in remaining
    }
    if not groups:
        if any(remaining.values()):
            raise SplitterError(
                f"not enough unforced samples to fill requested counts: {dict(remaining)!r}"
            )
        return placements

    ordered = sorted(
        groups,
        key=lambda item: (
            -len(item[1]),
            sha256(f"{seed}\0{item[0]}".encode("utf-8")).hexdigest(),
        ),
    )
    large = [item for item in ordered if len(item[1]) > 1]
    singletons = [item for item in ordered if len(item[1]) == 1]
    capacities = tuple(remaining[name] for name in remaining)
    split_names = tuple(remaining)

    if sum(capacities) != sum(len(members) for _, members in ordered):
        raise SplitterError("unforced sample count does not match remaining capacity")

    large_assignment = _assign_large_groups(large, split_names, capacities, seed)
    updated = list(capacities)
    for group, split_index in zip(large, large_assignment):
        placements[split_names[split_index]].append(group)
        updated[split_index] -= len(group[1])

    singleton_index = 0
    for split_index, split_name in enumerate(split_names):
        take = updated[split_index]
        if take < 0:
            raise SplitterError("internal splitter capacity underflow")
        placements[split_name].extend(singletons[singleton_index : singleton_index + take])
        singleton_index += take
    if singleton_index != len(singletons):
        raise SplitterError(
            "group sizes cannot satisfy the exact requested split counts"
        )
    return placements


def _assign_large_groups(
    groups: Sequence[tuple[str, tuple[T, ...]]],
    split_names: tuple[str, ...],
    capacities: tuple[int, ...],
    seed: int,
) -> tuple[int, ...]:
    if not groups:
        return ()

    sizes = tuple(len(members) for _, members in groups)
    suffix_sizes = [0] * (len(sizes) + 1)
    for index in range(len(sizes) - 1, -1, -1):
        suffix_sizes[index] = suffix_sizes[index + 1] + sizes[index]

    failed: set[tuple[int, tuple[int, ...]]] = set()

    def search(index: int, remaining: tuple[int, ...]) -> tuple[int, ...] | None:
        if index == len(groups):
            # Singleton groups can fill every residual slot exactly.
            return ()
        state = (index, remaining)
        if state in failed:
            return None
        if suffix_sizes[index] > sum(remaining):
            failed.add(state)
            return None

        size = sizes[index]
        group_digest = sha256(
            f"{seed}\0{groups[index][0]}".encode("utf-8")
        ).hexdigest()
        choices = sorted(
            range(len(split_names)),
            key=lambda split_index: (
                -(remaining[split_index] - size),
                sha256(
                    f"{group_digest}\0{split_names[split_index]}".encode("utf-8")
                ).hexdigest(),
            ),
        )
        tried_capacities: set[int] = set()
        for split_index in choices:
            capacity = remaining[split_index]
            if capacity < size or capacity in tried_capacities:
                continue
            tried_capacities.add(capacity)
            next_remaining = list(remaining)
            next_remaining[split_index] -= size
            tail = search(index + 1, tuple(next_remaining))
            if tail is not None:
                return (split_index,) + tail
        failed.add(state)
        return None

    answer = search(0, capacities)
    if answer is None:
        sizes_text = [len(members) for _, members in groups]
        raise SplitterError(
            "semantic group sizes cannot satisfy exact split counts; "
            f"group_sizes={sizes_text!r}, remaining={dict(zip(split_names, capacities))!r}"
        )
    return answer


def _validate_requested_counts(split_counts: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(split_counts, Mapping) or not split_counts:
        raise SplitterError("split_counts must be a non-empty mapping")
    ordered_names = [name for name in DEFAULT_SPLIT_ORDER if name in split_counts]
    ordered_names.extend(sorted(set(split_counts) - set(ordered_names)))
    result: dict[str, int] = {}
    for name in ordered_names:
        if not isinstance(name, str) or not name:
            raise SplitterError("split names must be non-empty strings")
        count = split_counts[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SplitterError(f"count for split {name!r} must be a non-negative integer")
        result[name] = count
    return result


def _field(value: object, name: str, default: object = _MISSING) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata(sample: object) -> object:
    metadata = _field(sample, "metadata", _MISSING)
    if metadata is _MISSING or metadata is None:
        raise SplitterError(f"sample {_sample_label(sample)!r} has no metadata")
    return metadata


def _sample_label(sample: object) -> str:
    value = _field(sample, "sample_id", "<unknown>")
    return str(value)


def _sample_id(sample: object) -> str:
    return _non_empty_text(_field(sample, "sample_id", _MISSING), "sample_id")


def _sample_sort_key(sample: object) -> str:
    return _sample_id(sample)


def _instruction(sample: object) -> str:
    metadata = _metadata(sample)
    value = _field(metadata, "instruction", _MISSING)
    if value is _MISSING:
        value = _field(sample, "instruction", _MISSING)
    return _non_empty_text(value, "instruction")


def _generation_seed(sample: object) -> int:
    value = _field(_metadata(sample), "seed", _MISSING)
    if value is _MISSING:
        value = _field(sample, "generation_seed", _MISSING)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitterError(
            f"sample {_sample_label(sample)!r} must have an integer generation seed"
        )
    return value


def _split_hint(
    sample: object,
    allowed: Mapping[str, int],
    *,
    respect_existing_split: bool,
) -> str | None:
    metadata = _metadata(sample)
    values: list[str] = []
    containers: list[tuple[object, tuple[str, ...]]] = [
        (metadata, ("split_hint", "intended_split", "required_split"))
    ]
    if respect_existing_split:
        containers.insert(0, (sample, ("split",)))
    for container, names in containers:
        for name in names:
            value = _field(container, name, _MISSING)
            if value is not _MISSING and value is not None and str(value).strip():
                values.append(_non_empty_text(value, name))

    holdout = _field(metadata, "holdout_type", _MISSING)
    if holdout is not _MISSING and holdout is not None:
        normalized = _non_empty_text(holdout, "holdout_type").casefold()
        holdout_map = {
            "compositional": "test_compositional",
            "language": "test_language",
            "robustness": "test_robustness",
        }
        if normalized not in holdout_map:
            raise SplitterError(f"unknown holdout_type {holdout!r}")
        values.append(holdout_map[normalized])
    if _robustness_marker(sample) is not None:
        values.append("test_robustness")

    unique = set(values)
    if len(unique) > 1:
        raise SplitterError(
            f"sample {_sample_label(sample)!r} has conflicting split hints: "
            f"{sorted(unique)!r}"
        )
    if not unique:
        return None
    value = next(iter(unique))
    if value not in allowed:
        raise SplitterError(
            f"sample {_sample_label(sample)!r} requests unknown split {value!r}"
        )
    return value


def _paraphrase_group_key(sample: object) -> str:
    metadata = _metadata(sample)
    semantic = semantic_spec_key(sample)
    template = _non_empty_text(
        _field(metadata, "template_family", _MISSING), "template_family"
    )
    paraphrase = _non_empty_text(
        _field(metadata, "paraphrase_family", _MISSING), "paraphrase_family"
    )
    computed = make_group_id(semantic, template, paraphrase)
    stored = _field(metadata, "group_id", _MISSING)
    if stored is not _MISSING:
        normalized_stored = _non_empty_text(stored, "group_id")
        if normalized_stored != computed:
            raise SplitterError(
                f"sample {_sample_label(sample)!r} stored group_id "
                f"{normalized_stored!r} does not match computed {computed!r}"
            )
    return computed


def _composition_key(sample: object) -> str:
    metadata = _metadata(sample)
    for name in (
        "target_attribute_combination",
        "composition_key",
        "composition_id",
    ):
        value = _field(metadata, name, _MISSING)
        if value is not _MISSING:
            return f"{name}:{_canonical_json(value)}"
    gold = _field(sample, "gold", _MISSING)
    concept_id = _field(gold, "target_concept_id", _MISSING)
    if concept_id is _MISSING:
        raise SplitterError(
            f"sample {_sample_label(sample)!r} has no target composition key"
        )
    return "target_concept_id:" + _non_empty_text(
        concept_id, "gold.target_concept_id"
    )


def _composition_components(sample: object) -> tuple[str, ...]:
    metadata = _metadata(sample)
    value = _field(metadata, "composition_components", _MISSING)
    if value is _MISSING or value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(
            f"{key}={_canonical_json(component)}"
            for key, component in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SplitterError("composition_components must be a mapping or sequence")
    return tuple(sorted(_non_empty_text(item, "composition component") for item in value))


def _language_features(sample: object) -> tuple[str, ...]:
    metadata = _metadata(sample)
    explicit = _field(metadata, "language_holdout_keys", _MISSING)
    if explicit is _MISSING:
        explicit = _field(metadata, "language_feature_ids", _MISSING)
    if explicit is not _MISSING and explicit is not None:
        if isinstance(explicit, str):
            return ("explicit:" + _non_empty_text(explicit, "language feature"),)
        if isinstance(explicit, bytes) or not isinstance(explicit, Sequence):
            raise SplitterError("language feature IDs must be a sequence of strings")
        return tuple(
            sorted(
                "explicit:" + _non_empty_text(value, "language feature")
                for value in explicit
            )
        )

    features: list[str] = []
    for name in _LANGUAGE_FEATURE_FIELDS:
        value = _field(metadata, name, _MISSING)
        if value is not _MISSING and value is not None and str(value).strip():
            features.append(f"{name}:{_canonical_json(value)}")
    if features:
        return tuple(sorted(features))

    paraphrase = _field(metadata, "paraphrase_family", _MISSING)
    if paraphrase is _MISSING:
        return ()
    return (
        "paraphrase_family:"
        + _non_empty_text(paraphrase, "paraphrase_family"),
    )


def _robustness_marker(sample: object) -> str | None:
    metadata = _metadata(sample)
    difficulty = _field(metadata, "difficulty", _MISSING)
    if isinstance(difficulty, str) and difficulty.casefold() == "robustness":
        return "difficulty:robustness"
    for name in _ROBUSTNESS_FIELDS:
        value = _field(metadata, name, _MISSING)
        if value is _MISSING or value is None or value is False:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return f"{name}:{value}"
    for name in ("template_family", "paraphrase_family"):
        value = _field(metadata, name, _MISSING)
        if value is _MISSING or not isinstance(value, str):
            continue
        normalized = value.casefold()
        if re.search(r"(?:robust|inject|adversarial)", normalized):
            return f"{name}:{value}"
    return None


def _replace_sample_split(sample: T, split_name: str) -> T:
    """Return a split-labelled copy, never mutating a frozen dataset sample."""

    if is_dataclass(sample) and not isinstance(sample, type):
        try:
            return replace(sample, split=split_name)
        except TypeError as exc:
            raise SplitterError(
                f"sample {_sample_label(sample)!r} cannot be relabelled via "
                "dataclasses.replace"
            ) from exc
    if isinstance(sample, Mapping):
        copied = dict(sample)
        copied["split"] = split_name
        return copied  # type: ignore[return-value]
    raise SplitterError(
        f"sample {_sample_label(sample)!r} must be a dataclass or mapping to "
        "receive a split assignment"
    )


def _claim_unique(
    owners: dict[str, str], key: str, split_name: str, label: str
) -> None:
    previous = owners.get(key)
    if previous is not None:
        raise SplitterError(
            f"duplicate {label} {key!r} in splits {previous!r} and {split_name!r}"
        )
    owners[key] = split_name


def _claim_cross_split(
    owners: dict[object, str], key: object, split_name: str, label: str
) -> None:
    previous = owners.get(key)
    if previous is not None and previous != split_name:
        raise SplitterError(
            f"{label} {key!r} leaks across splits {previous!r} and {split_name!r}"
        )
    owners[key] = split_name


def _non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise SplitterError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise SplitterError(f"{field_name} must be non-empty")
    return normalized


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SplitterError(f"value is not canonical JSON data: {exc}") from exc


__all__ = [
    "DEFAULT_SPLIT_ORDER",
    "SplitAssignment",
    "SplitValidationReport",
    "SplitterError",
    "assign_groups",
    "assign_splits",
    "group_key",
    "normalize_instruction",
    "semantic_spec_key",
    "split_samples",
    "validate_holdout_invariants",
    "validate_splits",
]
