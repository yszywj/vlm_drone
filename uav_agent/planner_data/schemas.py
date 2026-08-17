"""Immutable schemas for the text-only Planner dataset.

The dataset keeps trusted Gold data separate from model-facing messages.  The
objects in this module contain no image, frame, target instance, spawn, or
Oracle fields and are safe to use without Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from models.base import ChatMessage
from tasks.schemas import GoldPlannerSpec


PLANNER_DATASET_SCHEMA_VERSION = "planner_v1"
PLANNER_DATASET_SPLITS = (
    "train",
    "validation",
    "test_iid",
    "test_compositional",
    "test_language",
    "test_robustness",
)
PLANNER_REVIEW_STATUSES = frozenset(
    {
        "unreviewed",
        "curated_template",
        "human_authored_template",
        "human_reviewed_template",
    }
)


class PlannerDataSchemaError(ValueError):
    """Raised when a Planner dataset object violates the v1 schema."""


def compute_semantic_spec_family(
    gold: GoldPlannerSpec,
    world_context_id: str,
) -> str:
    """Compute the trusted semantic family directly from Gold task fields.

    This is intentionally independent of generation metadata.  Split and
    leakage checks can therefore detect (and cannot be bypassed by) a forged
    ``metadata.semantic_spec_family`` value.
    """

    if not isinstance(gold, GoldPlannerSpec):
        raise TypeError("gold must be a GoldPlannerSpec")
    context_id = _text(world_context_id, "world_context_id")
    payload = {
        "world_context_id": context_id,
        "target_concept_id": gold.target_concept_id,
        "search_region": gold.search_region,
        "landing_zone": gold.landing_zone,
        "track_duration_s": gold.track_duration_s,
        "duration_explicit": "track_duration_s" in gold.explicit_fields,
        "takeoff_altitude_m": gold.takeoff_altitude_m,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "semantic_" + sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise PlannerDataSchemaError(f"{field_name} must be non-empty")
    return normalized


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise PlannerDataSchemaError(f"{field_name} must be finite")
    return normalized


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlannerDataSchemaError(f"{field_name} must be a non-negative integer")
    return value


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    normalized = tuple(_text(item, f"{field_name} item") for item in value)
    if len(set(normalized)) != len(normalized):
        raise PlannerDataSchemaError(f"{field_name} must not contain duplicates")
    return normalized


def _readonly_string_mapping(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, f"{field_name} key")
        result[key] = _text(raw_value, f"{field_name}[{key!r}]")
    if not result and not allow_empty:
        raise PlannerDataSchemaError(f"{field_name} must be non-empty")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class PlannerSampleMetadata:
    """Generation provenance and leakage-group identity for one sample."""

    instruction: str
    template_family: str
    paraphrase_family: str
    generation_source: str
    difficulty: str
    seed: int
    semantic_spec_family: str
    group_id: str
    review_status: str = "unreviewed"
    composition_components: tuple[str, ...] = ()
    language_feature_ids: tuple[str, ...] = ()
    robustness_category: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "instruction",
            "template_family",
            "paraphrase_family",
            "generation_source",
            "difficulty",
            "semantic_spec_family",
            "group_id",
            "review_status",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        if self.generation_source not in {"template", "external_candidate", "human"}:
            raise PlannerDataSchemaError(
                "generation_source must be template, external_candidate, or human"
            )
        if self.review_status not in PLANNER_REVIEW_STATUSES:
            raise PlannerDataSchemaError(
                "review_status must be one of: "
                + ", ".join(sorted(PLANNER_REVIEW_STATUSES))
            )
        if self.difficulty not in {"easy", "medium", "hard", "robustness"}:
            raise PlannerDataSchemaError(
                "difficulty must be easy, medium, hard, or robustness"
            )
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "composition_components",
            _text_tuple(self.composition_components, "composition_components"),
        )
        object.__setattr__(
            self,
            "language_feature_ids",
            _text_tuple(self.language_feature_ids, "language_feature_ids"),
        )
        if self.robustness_category is not None:
            object.__setattr__(
                self,
                "robustness_category",
                _text(self.robustness_category, "robustness_category"),
            )
        if self.difficulty == "robustness" and self.robustness_category is None:
            raise PlannerDataSchemaError(
                "robustness difficulty requires robustness_category"
            )
        if self.difficulty != "robustness" and self.robustness_category is not None:
            raise PlannerDataSchemaError(
                "robustness_category is only allowed for robustness samples"
            )

    @property
    def group_key(self) -> str:
        """The required semantic/template/paraphrase group key."""

        return "|".join(
            (
                self.semantic_spec_family,
                self.template_family,
                self.paraphrase_family,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "template_family": self.template_family,
            "paraphrase_family": self.paraphrase_family,
            "generation_source": self.generation_source,
            "review_status": self.review_status,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "semantic_spec_family": self.semantic_spec_family,
            "group_id": self.group_id,
            "composition_components": list(self.composition_components),
            "language_feature_ids": list(self.language_feature_ids),
            "robustness_category": self.robustness_category,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PlannerSampleMetadata":
        required = {
            "instruction",
            "template_family",
            "paraphrase_family",
            "generation_source",
            "review_status",
            "difficulty",
            "seed",
            "semantic_spec_family",
            "group_id",
            "composition_components",
            "language_feature_ids",
            "robustness_category",
        }
        _require_exact_keys(data, required, "metadata")
        return cls(**{key: data[key] for key in required})


@dataclass(frozen=True, slots=True)
class PlannerDatasetSample:
    """One deterministic Chat-SFT compatible Planner example."""

    schema_version: str
    sample_id: str
    split: str
    language: str
    world_context_id: str
    gold_spec_id: str
    messages: tuple[ChatMessage, ...]
    gold: GoldPlannerSpec
    metadata: PlannerSampleMetadata

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_DATASET_SCHEMA_VERSION:
            raise PlannerDataSchemaError(
                f"schema_version must be {PLANNER_DATASET_SCHEMA_VERSION!r}"
            )
        object.__setattr__(self, "sample_id", _text(self.sample_id, "sample_id"))
        split = _text(self.split, "split")
        if split not in PLANNER_DATASET_SPLITS:
            raise PlannerDataSchemaError(f"unknown Planner dataset split {split!r}")
        object.__setattr__(self, "split", split)
        language = _text(self.language, "language")
        if language != "zh-CN":
            raise PlannerDataSchemaError("Planner dataset v1 language must be 'zh-CN'")
        object.__setattr__(self, "language", language)
        object.__setattr__(
            self, "world_context_id", _text(self.world_context_id, "world_context_id")
        )
        object.__setattr__(self, "gold_spec_id", _text(self.gold_spec_id, "gold_spec_id"))
        if not isinstance(self.gold, GoldPlannerSpec):
            raise TypeError("gold must be a GoldPlannerSpec")
        if self.gold_spec_id != self.gold.spec_id:
            raise PlannerDataSchemaError("gold_spec_id must equal gold.spec_id")
        if not isinstance(self.metadata, PlannerSampleMetadata):
            raise TypeError("metadata must be PlannerSampleMetadata")
        if isinstance(self.messages, (str, bytes)) or not isinstance(
            self.messages, Sequence
        ):
            raise TypeError("messages must be a sequence of ChatMessage")
        messages = tuple(self.messages)
        if len(messages) != 3 or tuple(message.role for message in messages) != (
            "system",
            "user",
            "assistant",
        ):
            raise PlannerDataSchemaError(
                "messages must contain exactly system, user, assistant in that order"
            )
        if any(not isinstance(message, ChatMessage) for message in messages):
            raise TypeError("messages must contain ChatMessage objects")
        object.__setattr__(self, "messages", messages)

    def to_dict(self) -> dict[str, object]:
        """Return fields in the stable planner_v1 JSON order."""

        gold = self.gold.to_dict()
        # ``gold_spec_id`` is represented at the record level and intentionally
        # omitted inside ``gold`` to match the public dataset contract.
        gold.pop("spec_id", None)
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "split": self.split,
            "language": self.language,
            "world_context_id": self.world_context_id,
            "gold_spec_id": self.gold_spec_id,
            "messages": [message.to_dict() for message in self.messages],
            "gold": gold,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PlannerDatasetSample":
        required = {
            "schema_version",
            "sample_id",
            "split",
            "language",
            "world_context_id",
            "gold_spec_id",
            "messages",
            "gold",
            "metadata",
        }
        _require_exact_keys(data, required, "Planner dataset sample")
        raw_messages = data["messages"]
        if isinstance(raw_messages, (str, bytes)) or not isinstance(
            raw_messages, Sequence
        ):
            raise TypeError("messages must be a list")
        messages: list[ChatMessage] = []
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise TypeError(f"messages[{index}] must be an object")
            _require_exact_keys(raw_message, {"role", "content"}, f"messages[{index}]")
            messages.append(
                ChatMessage(role=raw_message["role"], content=raw_message["content"])
            )
        raw_gold = data["gold"]
        if not isinstance(raw_gold, Mapping):
            raise TypeError("gold must be an object")
        gold_with_id = {"spec_id": data["gold_spec_id"], **dict(raw_gold)}
        gold = GoldPlannerSpec.from_dict(gold_with_id)
        raw_metadata = data["metadata"]
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("metadata must be an object")
        return cls(
            schema_version=data["schema_version"],
            sample_id=data["sample_id"],
            split=data["split"],
            language=data["language"],
            world_context_id=data["world_context_id"],
            gold_spec_id=data["gold_spec_id"],
            messages=tuple(messages),
            gold=gold,
            metadata=PlannerSampleMetadata.from_dict(raw_metadata),
        )


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Exact target counts for the six public dataset splits."""

    name: str
    split_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "profile name"))
        if not isinstance(self.split_counts, Mapping):
            raise TypeError("split_counts must be a mapping")
        if set(self.split_counts) != set(PLANNER_DATASET_SPLITS):
            raise PlannerDataSchemaError(
                "split_counts must contain exactly the six planner_v1 splits"
            )
        counts = {
            split: _nonnegative_integer(self.split_counts[split], f"count {split}")
            for split in PLANNER_DATASET_SPLITS
        }
        if not any(counts.values()):
            raise PlannerDataSchemaError("a dataset profile cannot be empty")
        object.__setattr__(self, "split_counts", MappingProxyType(counts))

    @property
    def total_samples(self) -> int:
        return sum(self.split_counts.values())


@runtime_checkable
class InstructionParaphraser(Protocol):
    """Optional candidate-only instruction rewriter; never creates labels."""

    def paraphrase(
        self,
        instruction: str,
        gold_spec: GoldPlannerSpec,
        allowed_aliases: Mapping[str, object],
        count: int,
    ) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Portable manifest; all paths are dataset-relative resource names."""

    schema_version: str
    dataset_name: str
    profile: str
    seed: int
    split_counts: Mapping[str, int]
    resource_sha256: Mapping[str, str]
    split_sha256: Mapping[str, str]
    statistics_sha256: str
    generated_at_utc: str

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_DATASET_SCHEMA_VERSION:
            raise PlannerDataSchemaError("invalid manifest schema_version")
        object.__setattr__(self, "dataset_name", _text(self.dataset_name, "dataset_name"))
        object.__setattr__(self, "profile", _text(self.profile, "profile"))
        if self.dataset_name != "planner_v1":
            raise PlannerDataSchemaError("dataset_name must be 'planner_v1'")
        if self.profile not in {"pilot", "full"}:
            raise PlannerDataSchemaError("profile must be 'pilot' or 'full'")
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        if not isinstance(self.split_counts, Mapping):
            raise TypeError("split_counts must be a mapping")
        counts = {
            _text(key, "split count key"): _nonnegative_integer(value, f"count {key}")
            for key, value in self.split_counts.items()
        }
        if set(counts) != set(PLANNER_DATASET_SPLITS):
            raise PlannerDataSchemaError("manifest split_counts has wrong split names")
        object.__setattr__(self, "split_counts", MappingProxyType(counts))
        object.__setattr__(
            self,
            "resource_sha256",
            _readonly_string_mapping(self.resource_sha256, "resource_sha256"),
        )
        object.__setattr__(
            self,
            "split_sha256",
            _readonly_string_mapping(self.split_sha256, "split_sha256"),
        )
        object.__setattr__(
            self, "statistics_sha256", _text(self.statistics_sha256, "statistics_sha256")
        )
        object.__setattr__(
            self, "generated_at_utc", _text(self.generated_at_utc, "generated_at_utc")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "profile": self.profile,
            "seed": self.seed,
            "split_counts": dict(self.split_counts),
            "resource_sha256": dict(self.resource_sha256),
            "split_sha256": dict(self.split_sha256),
            "statistics_sha256": self.statistics_sha256,
            "generated_at_utc": self.generated_at_utc,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DatasetManifest":
        required = {
            "schema_version",
            "dataset_name",
            "profile",
            "seed",
            "split_counts",
            "resource_sha256",
            "split_sha256",
            "statistics_sha256",
            "generated_at_utc",
        }
        _require_exact_keys(data, required, "dataset manifest")
        return cls(**{key: data[key] for key in required})


def make_group_id(
    semantic_spec_family: str,
    template_family: str,
    paraphrase_family: str,
) -> str:
    """Return the stable semantic/template/paraphrase grouping identity."""

    values = (
        _text(semantic_spec_family, "semantic_spec_family"),
        _text(template_family, "template_family"),
        _text(paraphrase_family, "paraphrase_family"),
    )
    payload = "\x1f".join(values).encode("utf-8")
    return "group_" + sha256(payload).hexdigest()[:20]


def _require_exact_keys(
    data: object,
    required: set[str],
    field_name: str,
) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in data):
        raise TypeError(f"{field_name} keys must be strings")
    unknown = set(data) - required
    missing = required - set(data)
    if unknown:
        raise PlannerDataSchemaError(
            f"{field_name} contains unknown fields: " + ", ".join(sorted(unknown))
        )
    if missing:
        raise PlannerDataSchemaError(
            f"{field_name} is missing fields: " + ", ".join(sorted(missing))
        )


__all__ = [
    "DatasetManifest",
    "DatasetProfile",
    "InstructionParaphraser",
    "PLANNER_DATASET_SCHEMA_VERSION",
    "PLANNER_DATASET_SPLITS",
    "PLANNER_REVIEW_STATUSES",
    "PlannerDataSchemaError",
    "PlannerDatasetSample",
    "PlannerSampleMetadata",
    "compute_semantic_spec_family",
    "make_group_id",
]
