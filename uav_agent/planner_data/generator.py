"""Gold-first, deterministic Planner dataset generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from types import MappingProxyType

from models.base import ChatMessage
from planner.prompt_builder import build_mission_planner_messages
from tasks.schemas import GoldPlannerSpec, PlannerWorldCase
from tasks.target_ontology import TargetOntology

from .renderers import (
    InstructionRenderer,
    LanguageLexicon,
    StrictYamlError,
    load_language_lexicon,
    load_template_catalog,
    load_world_cases,
    load_yaml_strict,
    world_case_to_runtime_context,
)
from .schemas import (
    DatasetManifest,
    DatasetProfile,
    InstructionParaphraser,
    PLANNER_DATASET_SCHEMA_VERSION,
    PLANNER_DATASET_SPLITS,
    PlannerDatasetSample,
    PlannerSampleMetadata,
    compute_semantic_spec_family,
    make_group_id,
)
from .splitter import normalize_instruction, validate_splits


DEFAULT_RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "planner_v1"
DEFAULT_DATASET_CONFIG_PATH = DEFAULT_RESOURCE_ROOT / "dataset_config.yaml"
DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "mission_planner_system.txt"
)


class DatasetGenerationError(RuntimeError):
    """Raised when a valid dataset cannot be generated atomically."""


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    schema_version: str
    language: str
    resource_files: Mapping[str, str]
    profiles: Mapping[str, DatasetProfile]
    track_durations_s: tuple[float, ...]
    takeoff_altitudes_m: tuple[float, ...]
    split_seed_offsets: Mapping[str, int]
    compositional_holdout_concepts: frozenset[str]
    minimum_coverage: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class GeneratedPlannerDataset:
    profile: DatasetProfile
    seed: int
    samples_by_split: Mapping[str, tuple[PlannerDatasetSample, ...]]
    worlds: Mapping[str, PlannerWorldCase]
    statistics: Mapping[str, object]
    resource_sha256: Mapping[str, str]

    @property
    def num_samples(self) -> int:
        return sum(len(samples) for samples in self.samples_by_split.values())


@dataclass(frozen=True, slots=True)
class ParaphraseCandidateValidation:
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"accepted": self.accepted, "reasons": list(self.reasons)}


def load_generation_config(path: str | Path = DEFAULT_DATASET_CONFIG_PATH) -> GenerationConfig:
    config_path = Path(path)
    try:
        raw = load_yaml_strict(config_path, "Planner dataset config")
    except StrictYamlError as exc:
        raise DatasetGenerationError(str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise DatasetGenerationError("dataset config root must be an object")
    required_root = {"schema_version", "language", "resources", "profiles", "generation"}
    _exact_keys(raw, required_root, "dataset config")
    if raw["schema_version"] != "planner_dataset_config_v1":
        raise DatasetGenerationError("unsupported dataset config schema_version")
    if raw["language"] != "zh-CN":
        raise DatasetGenerationError("Planner dataset v1 supports only zh-CN")

    resources = _string_mapping(raw["resources"], "resources")
    expected_resources = {
        "target_ontology",
        "language_lexicon",
        "world_contexts",
        "templates",
    }
    if set(resources) != expected_resources:
        raise DatasetGenerationError("resources must name exactly the four v1 resources")
    if any(Path(name).is_absolute() or len(Path(name).parts) != 1 for name in resources.values()):
        raise DatasetGenerationError("resource paths must be portable file names")

    profiles_raw = raw["profiles"]
    if not isinstance(profiles_raw, Mapping) or set(profiles_raw) != {"pilot", "full"}:
        raise DatasetGenerationError("profiles must contain pilot and full")
    profiles: dict[str, DatasetProfile] = {}
    for name, value in profiles_raw.items():
        if not isinstance(value, Mapping):
            raise DatasetGenerationError(f"profile {name!r} must be an object")
        _exact_keys(value, {"split_counts"}, f"profile {name}")
        profiles[str(name)] = DatasetProfile(name=str(name), split_counts=value["split_counts"])

    generation = raw["generation"]
    if not isinstance(generation, Mapping):
        raise DatasetGenerationError("generation must be an object")
    allowed_generation = {
        "track_durations_s",
        "takeoff_altitudes_m",
        "explicit_duration_ratio",
        "explicit_altitude_ratio",
        "generation_source",
        "default_paraphraser",
        "split_seed_offsets",
        "compositional_holdout_concepts",
        "minimum_coverage",
    }
    _exact_keys(generation, allowed_generation, "generation")
    durations = _positive_number_sequence(generation["track_durations_s"], "track_durations_s")
    altitudes = _positive_number_sequence(
        generation["takeoff_altitudes_m"], "takeoff_altitudes_m"
    )
    # Ratios and switches are validated even though the generator enumerates
    # exact semantic combinations rather than sampling Bernoulli fields.
    for key in ("explicit_duration_ratio", "explicit_altitude_ratio"):
        ratio = _finite_number(generation[key], key)
        if not 0.0 <= ratio <= 1.0:
            raise DatasetGenerationError(f"{key} must be between 0 and 1")
    if generation["generation_source"] != "template":
        raise DatasetGenerationError("official Gold labels must use template generation")
    if generation["default_paraphraser"] != "none":
        raise DatasetGenerationError("default_paraphraser must be none")
    offsets = _integer_mapping(generation["split_seed_offsets"], "split_seed_offsets")
    if set(offsets) != set(PLANNER_DATASET_SPLITS) or len(set(offsets.values())) != len(offsets):
        raise DatasetGenerationError("split_seed_offsets must uniquely cover all splits")
    holdouts_raw = generation["compositional_holdout_concepts"]
    if isinstance(holdouts_raw, (str, bytes)) or not isinstance(holdouts_raw, Sequence):
        raise DatasetGenerationError("compositional_holdout_concepts must be a list")
    holdouts = frozenset(_non_empty_text(value, "holdout concept") for value in holdouts_raw)
    if not holdouts:
        raise DatasetGenerationError("at least one compositional holdout is required")
    coverage = _integer_mapping(generation["minimum_coverage"], "minimum_coverage")
    if any(value <= 0 for value in coverage.values()):
        raise DatasetGenerationError("minimum coverage values must be positive")
    return GenerationConfig(
        schema_version=str(raw["schema_version"]),
        language=str(raw["language"]),
        resource_files=MappingProxyType(resources),
        profiles=MappingProxyType(profiles),
        track_durations_s=durations,
        takeoff_altitudes_m=altitudes,
        split_seed_offsets=MappingProxyType(offsets),
        compositional_holdout_concepts=holdouts,
        minimum_coverage=MappingProxyType(coverage),
    )


class PlannerDatasetGenerator:
    """Create labels from trusted Gold first, then render instructions."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
        *,
        system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
    ) -> None:
        self.config_path = Path(config_path)
        self.resource_root = self.config_path.parent
        self.config = load_generation_config(self.config_path)
        try:
            self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise DatasetGenerationError("could not read Planner system prompt") from exc
        if not self.system_prompt:
            raise DatasetGenerationError("Planner system prompt must be non-empty")

        resource = lambda key: self.resource_root / self.config.resource_files[key]
        try:
            # TargetOntology owns semantic validation; this preflight adds the
            # Planner-v1 duplicate-key guarantee before its existing parser runs.
            load_yaml_strict(resource("target_ontology"), "target ontology")
        except StrictYamlError as exc:
            raise DatasetGenerationError(str(exc)) from exc
        self.ontology = TargetOntology.from_file(resource("target_ontology"))
        self.lexicon = load_language_lexicon(resource("language_lexicon"))
        self.templates = load_template_catalog(resource("templates"))
        self.worlds = load_world_cases(resource("world_contexts"))
        if len(next(iter(self.worlds.values())).search_regions) < 4:
            raise DatasetGenerationError("planner_v1 requires at least four search regions")
        if len(next(iter(self.worlds.values())).landing_zones) < 3:
            raise DatasetGenerationError("planner_v1 requires at least three landing zones")
        unknown_holdouts = self.config.compositional_holdout_concepts - set(self.ontology.concepts)
        if unknown_holdouts:
            raise DatasetGenerationError(
                "unknown compositional holdout concepts: " + ", ".join(sorted(unknown_holdouts))
            )
        self.renderer = InstructionRenderer(self.ontology, self.lexicon, self.templates)
        self._resource_paths = {
            "dataset_config.yaml": self.config_path,
            **{
                self.config.resource_files[key]: resource(key)
                for key in sorted(self.config.resource_files)
            },
            "mission_planner_system.txt": Path(system_prompt_path),
        }

    def generate(
        self,
        *,
        seed: int,
        profile: str = "pilot",
        paraphraser: InstructionParaphraser | None = None,
    ) -> GeneratedPlannerDataset:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise DatasetGenerationError("seed must be a non-negative integer")
        if profile not in self.config.profiles:
            raise DatasetGenerationError(f"unknown dataset profile {profile!r}")
        if paraphraser is not None:
            raise DatasetGenerationError(
                "external paraphrases are candidate-only and cannot enter official splits"
            )
        selected_profile = self.config.profiles[profile]
        used_semantics: set[str] = set()
        used_instructions: set[str] = set()
        samples_by_split: dict[str, tuple[PlannerDatasetSample, ...]] = {}
        for split_index, split in enumerate(PLANNER_DATASET_SPLITS):
            count = selected_profile.split_counts[split]
            samples = self._generate_split(
                split=split,
                count=count,
                master_seed=seed,
                split_index=split_index,
                used_semantics=used_semantics,
                used_instructions=used_instructions,
            )
            samples_by_split[split] = tuple(samples)

        # This is a validation pass over the deliberate Gold holdouts, not a
        # random line split.
        validate_splits(
            samples_by_split,
            expected_counts=selected_profile.split_counts,
            enforce_holdouts=True,
        )
        statistics = build_statistics(samples_by_split)
        self.validate_minimum_coverage(statistics)
        return GeneratedPlannerDataset(
            profile=selected_profile,
            seed=seed,
            samples_by_split=MappingProxyType(samples_by_split),
            worlds=MappingProxyType(dict(self.worlds)),
            statistics=MappingProxyType(statistics),
            resource_sha256=MappingProxyType(
                {name: sha256_file(path) for name, path in self._resource_paths.items()}
            ),
        )

    def validate_minimum_coverage(self, statistics: Mapping[str, object]) -> None:
        """Require every configured closed-set value, including zero-count ones."""

        expected = {
            "target_concept": set(self.ontology.concepts),
            "search_region": {
                name for world in self.worlds.values() for name in world.search_regions
            },
            "landing_zone": {
                name for world in self.worlds.values() for name in world.landing_zones
            },
            "track_duration": {
                f"{value:g}"
                for value in (
                    *self.config.track_durations_s,
                    *(world.default_track_duration_s for world in self.worlds.values()),
                )
            },
        }
        _validate_minimum_coverage(
            statistics,
            self.config.minimum_coverage,
            expected_values=expected,
        )

    def _generate_split(
        self,
        *,
        split: str,
        count: int,
        master_seed: int,
        split_index: int,
        used_semantics: set[str],
        used_instructions: set[str],
    ) -> list[PlannerDatasetSample]:
        holdouts = self.config.compositional_holdout_concepts
        if split == "test_compositional":
            concept_ids = sorted(holdouts)
        else:
            concept_ids = sorted(set(self.ontology.concepts) - holdouts)
        combinations = self._semantic_combinations(concept_ids)
        rng = random.Random(master_seed * 1009 + split_index * 1000003)
        rng.shuffle(combinations)
        result: list[PlannerDatasetSample] = []
        split_offset = self.config.split_seed_offsets[split]
        for combination in combinations:
            if len(result) == count:
                break
            semantic_family = _semantic_family(combination)
            if semantic_family in used_semantics:
                continue
            sample_index = len(result)
            generation_seed = master_seed * 1_000_000 + split_offset + sample_index
            sample = self._build_sample(
                split=split,
                sample_index=sample_index,
                generation_seed=generation_seed,
                semantic_family=semantic_family,
                combination=combination,
                used_instructions=used_instructions,
            )
            if sample is None:
                continue
            used_semantics.add(semantic_family)
            used_instructions.add(normalize_instruction(sample.metadata.instruction))
            result.append(sample)
        if len(result) != count:
            raise DatasetGenerationError(
                f"could generate only {len(result)} of {count} requested {split} samples"
            )
        return result

    def _semantic_combinations(self, concept_ids: Sequence[str]) -> list[dict[str, object]]:
        combinations: list[dict[str, object]] = []
        for world in self.worlds.values():
            duration_options = [(world.default_track_duration_s, False)] + [
                (duration, True) for duration in self.config.track_durations_s
            ]
            altitude_options: list[float | None] = [None, *self.config.takeoff_altitudes_m]
            for concept_id in concept_ids:
                for region in sorted(world.search_regions):
                    for landing in sorted(world.landing_zones):
                        for duration, duration_explicit in duration_options:
                            for altitude in altitude_options:
                                combinations.append(
                                    {
                                        "world": world,
                                        "concept_id": concept_id,
                                        "search_region": region,
                                        "landing_zone": landing,
                                        "track_duration_s": duration,
                                        "duration_explicit": duration_explicit,
                                        "takeoff_altitude_m": altitude,
                                    }
                                )
        return combinations

    def _build_sample(
        self,
        *,
        split: str,
        sample_index: int,
        generation_seed: int,
        semantic_family: str,
        combination: Mapping[str, object],
        used_instructions: set[str],
    ) -> PlannerDatasetSample | None:
        world = combination["world"]
        assert isinstance(world, PlannerWorldCase)
        concept = self.ontology.require_concept(str(combination["concept_id"]))
        explicit_fields = {
            "target_description",
            "search_region",
            "landing_zone",
        }
        if combination["duration_explicit"]:
            explicit_fields.add("track_duration_s")
        altitude = combination["takeoff_altitude_m"]
        if altitude is not None:
            explicit_fields.add("takeoff_altitude_m")
        spec_id = f"{split}_spec_{sample_index:06d}"
        gold = GoldPlannerSpec(
            spec_id=spec_id,
            target_concept_id=concept.concept_id,
            target_description=concept.canonical_description,
            search_region=str(combination["search_region"]),
            track_duration_s=float(combination["track_duration_s"]),
            landing_zone=str(combination["landing_zone"]),
            takeoff_altitude_m=None if altitude is None else float(altitude),
            explicit_fields=frozenset(explicit_fields),
        )
        self.ontology.validate_gold_spec(gold)
        trusted_semantic_family = compute_semantic_spec_family(gold, world.context_id)
        if semantic_family != trusted_semantic_family:
            raise DatasetGenerationError(
                "internal semantic family differs from the Gold-derived schema contract"
            )
        rendered = None
        for variant_index in range(256):
            candidate = self.renderer.render(
                gold,
                world,
                split=split,
                seed=generation_seed,
                variant_index=variant_index,
            )
            if normalize_instruction(candidate.instruction) not in used_instructions:
                rendered = candidate
                break
        if rendered is None:
            return None

        runtime_world = world_case_to_runtime_context(world)
        prompt_messages = build_mission_planner_messages(
            rendered.instruction,
            runtime_world,
            self.system_prompt,
        )
        assistant_json = serialize_expected_intent(gold)
        messages = (
            *prompt_messages,
            ChatMessage(role="assistant", content=assistant_json),
        )
        group_id = make_group_id(
            semantic_family,
            rendered.template_family,
            rendered.paraphrase_family,
        )
        components = tuple(
            f"{key}={value}" for key, value in sorted(concept.attributes.items())
        )
        language_features = tuple(
            sorted(
                {
                    f"paraphrase:{rendered.paraphrase_family}",
                    *(f"alias:{role}:{value}" for role, value in rendered.aliases.items()),
                }
            )
        )
        is_robustness = split == "test_robustness"
        metadata = PlannerSampleMetadata(
            instruction=rendered.instruction,
            template_family=rendered.template_family,
            paraphrase_family=rendered.paraphrase_family,
            generation_source="template",
            # These rows are deterministically generated from repository
            # templates.  No human review occurred in this pipeline.
            review_status="unreviewed",
            difficulty="robustness" if is_robustness else rendered.difficulty,
            seed=generation_seed,
            semantic_spec_family=semantic_family,
            group_id=group_id,
            composition_components=components,
            language_feature_ids=language_features,
            robustness_category=rendered.robustness_category,
        )
        return PlannerDatasetSample(
            schema_version=PLANNER_DATASET_SCHEMA_VERSION,
            sample_id=f"{split}_{sample_index:06d}",
            split=split,
            language="zh-CN",
            world_context_id=world.context_id,
            gold_spec_id=spec_id,
            messages=messages,
            gold=gold,
            metadata=metadata,
        )


def serialize_expected_intent(gold: GoldPlannerSpec) -> str:
    """Serialize the deterministic assistant label in MissionIntent field order."""

    return json.dumps(
        gold.to_expected_intent().to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def build_statistics(
    samples_by_split: Mapping[str, Sequence[PlannerDatasetSample]],
) -> dict[str, object]:
    all_samples = [sample for split in PLANNER_DATASET_SPLITS for sample in samples_by_split[split]]
    counter = lambda values: dict(sorted(Counter(values).items()))
    language_samples = samples_by_split["test_language"]
    robustness_samples = samples_by_split["test_robustness"]
    language_human_fraction = _review_fraction(
        language_samples,
        {"human_authored_template", "human_reviewed_template"},
    )
    robustness_human_reviewed_fraction = _review_fraction(
        robustness_samples,
        {"human_reviewed_template"},
    )
    train_external_fraction = _source_fraction(
        samples_by_split["train"], "external_candidate"
    )
    non_train_external_count = sum(
        sample.metadata.generation_source == "external_candidate"
        for split in PLANNER_DATASET_SPLITS
        if split != "train"
        for sample in samples_by_split[split]
    )
    unreviewed_external_count = sum(
        sample.metadata.generation_source == "external_candidate"
        and sample.metadata.review_status != "human_reviewed_template"
        for sample in all_samples
    )
    external_candidate_policy_met = (
        train_external_fraction <= 0.5
        and non_train_external_count == 0
        and unreviewed_external_count == 0
    )
    full_review_requirements_met = (
        language_human_fraction >= 0.5
        and robustness_human_reviewed_fraction >= 1.0
        and external_candidate_policy_met
    )
    split_provenance = {
        split: {
            "generation_source_counts": counter(
                sample.metadata.generation_source
                for sample in samples_by_split[split]
            ),
            "review_status_counts": counter(
                sample.metadata.review_status
                for sample in samples_by_split[split]
            ),
        }
        for split in PLANNER_DATASET_SPLITS
    }
    return {
        "schema_version": PLANNER_DATASET_SCHEMA_VERSION,
        "total_samples": len(all_samples),
        "split_counts": {split: len(samples_by_split[split]) for split in PLANNER_DATASET_SPLITS},
        "target_concept_counts": counter(sample.gold.target_concept_id for sample in all_samples),
        "search_region_counts": counter(sample.gold.search_region for sample in all_samples),
        "landing_zone_counts": counter(sample.gold.landing_zone for sample in all_samples),
        "track_duration_counts": counter(f"{sample.gold.track_duration_s:g}" for sample in all_samples),
        "takeoff_altitude_counts": counter(
            "null" if sample.gold.takeoff_altitude_m is None else f"{sample.gold.takeoff_altitude_m:g}"
            for sample in all_samples
        ),
        "template_family_counts": counter(sample.metadata.template_family for sample in all_samples),
        "generation_source_counts": counter(sample.metadata.generation_source for sample in all_samples),
        "review_status_counts": counter(sample.metadata.review_status for sample in all_samples),
        "robustness_category_counts": counter(
            sample.metadata.robustness_category
            for sample in robustness_samples
            if sample.metadata.robustness_category is not None
        ),
        "split_provenance": split_provenance,
        "template_fraction": _source_fraction(all_samples, "template"),
        "external_candidate_fraction": _source_fraction(
            all_samples, "external_candidate"
        ),
        "automatic_paraphrase_fraction": train_external_fraction,
        "train_external_candidate_fraction": train_external_fraction,
        "non_train_external_candidate_count": non_train_external_count,
        "unreviewed_external_candidate_count": unreviewed_external_count,
        "external_candidate_policy_met": external_candidate_policy_met,
        "test_language_human_authored_or_reviewed_fraction": language_human_fraction,
        "test_robustness_human_reviewed_fraction": robustness_human_reviewed_fraction,
        "full_review_requirements_met": full_review_requirements_met,
    }


def write_generated_dataset(
    dataset: GeneratedPlannerDataset,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> DatasetManifest:
    """Atomically publish a complete official dataset directory."""

    destination = Path(output_root)
    recomputed_statistics = build_statistics(dataset.samples_by_split)
    if dict(dataset.statistics) != recomputed_statistics:
        raise DatasetGenerationError(
            "dataset statistics must be recomputed exactly from the samples before publish"
        )
    if not bool(recomputed_statistics.get("external_candidate_policy_met", False)):
        raise DatasetGenerationError(
            "external candidates must be human-reviewed, train-only, and no more "
            "than 50% of train"
        )
    if (
        dataset.profile.name == "full"
        and not bool(recomputed_statistics.get("full_review_requirements_met", False))
    ):
        raise DatasetGenerationError(
            "full Planner dataset cannot be published until test_language is "
            "at least 50% human-authored/reviewed and test_robustness is 100% "
            "human-reviewed"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"dataset already exists: {destination}")
    if destination.exists() and overwrite and (destination / "_candidates").exists():
        raise DatasetGenerationError(
            "refusing to overwrite a dataset that contains _candidates; move or "
            "review the local candidate work first"
        )
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent))
    backup_container: Path | None = None
    backup: Path | None = None
    published = False
    try:
        split_hashes: dict[str, str] = {}
        for split in PLANNER_DATASET_SPLITS:
            path = stage / f"{split}.jsonl"
            _write_jsonl(path, dataset.samples_by_split[split])
            split_hashes[path.name] = sha256_file(path)
        statistics_path = stage / "statistics.json"
        _write_json(statistics_path, recomputed_statistics)
        statistics_hash = sha256_file(statistics_path)
        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest = DatasetManifest(
            schema_version=PLANNER_DATASET_SCHEMA_VERSION,
            dataset_name="planner_v1",
            profile=dataset.profile.name,
            seed=dataset.seed,
            split_counts=dataset.profile.split_counts,
            resource_sha256=dataset.resource_sha256,
            split_sha256=split_hashes,
            statistics_sha256=statistics_hash,
            generated_at_utc=generated_at,
        )
        _write_json(stage / "dataset_manifest.json", manifest.to_dict())
        checksums = {**split_hashes, "statistics.json": statistics_hash}
        _write_text(
            stage / "checksums.sha256",
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        )
        if destination.exists():
            # Own a fresh sibling namespace for this invocation.  A fixed
            # ``.old`` name risks deleting an unrelated user directory and is
            # unsafe under concurrent/retried publishes.
            backup_container = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.",
                    suffix=".backup",
                    dir=destination.parent,
                )
            )
            backup = backup_container / "previous"
            os.replace(destination, backup)
        os.replace(stage, destination)
        published = True
        if backup_container is not None:
            shutil.rmtree(backup_container)
            backup_container = None
            backup = None
        return manifest
    except Exception as exc:
        if backup is not None and backup.exists():
            # ``os.replace(stage, destination)`` is atomic.  If a path has
            # appeared here after that operation failed, it may belong to a
            # concurrent actor; never remove it to force restoration.
            if destination.exists():
                if not published:
                    raise DatasetGenerationError(
                        "dataset publish failed and the previous dataset could not be "
                        f"safely restored; backup retained at {backup}"
                    ) from exc
                raise
            os.replace(backup, destination)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup_container is not None and backup_container.exists():
            # Remove only a directory created and owned by this invocation,
            # and only after its previous dataset has been restored/moved.
            if backup is None or not backup.exists():
                shutil.rmtree(backup_container)


def generate_and_write_dataset(
    *,
    config_path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
    output_root: str | Path,
    seed: int,
    profile: str = "pilot",
    overwrite: bool = False,
) -> DatasetManifest:
    generator = PlannerDatasetGenerator(config_path)
    generated = generator.generate(seed=seed, profile=profile)
    return write_generated_dataset(generated, output_root, overwrite=overwrite)


def validate_paraphrase_candidate(
    candidate_instruction: str,
    *,
    gold: GoldPlannerSpec,
    world: PlannerWorldCase,
    ontology: TargetOntology,
    lexicon: LanguageLexicon,
    allow_approved_robustness: bool = False,
    strict_external: bool = True,
) -> ParaphraseCandidateValidation:
    """Conservatively validate an external instruction-only rewrite.

    Only phrases in the closed lexicon/ontology are accepted.  This function
    never infers or changes a Gold label and deliberately rejects ambiguous
    free-form rewrites for later human review.
    """

    candidate = _non_empty_text(candidate_instruction, "candidate_instruction")
    reasons: list[str] = []

    # The robustness split deliberately appends a small, reviewed set of
    # adversarial strings.  They are test noise rather than task semantics, so
    # remove only those exact, resource-owned strings before interpreting the
    # instruction.  Arbitrary lookalikes remain visible to the checks below.
    semantic_text = candidate
    if allow_approved_robustness:
        for injection in _robustness_injection_texts(lexicon):
            semantic_text = semantic_text.replace(injection, " ")

    if _contains_forbidden_instruction_data(semantic_text):
        reasons.append("FORBIDDEN_RUNTIME_DATA")

    if any(token in semantic_text for token in ("取消", "改成", "换成", "删掉", "移除")) or re.search(
        r"(?:不要|无需|不用|不必)"
        r"[^\uff0c。\uff1b;]{0,12}"
        r"(?:起飞|搜索|寻找|搜寻|跟踪|跟随|返回|返航|降落|着陆|落地|去|到|执行)",
        semantic_text,
    ) or re.search(
        r"别(?:再)?(?:起飞|搜索|寻找|搜寻|找|跟踪|跟随|返回|返航|降落|着陆|落地|去|到|执行)",
        semantic_text,
    ):
        reasons.append("NEGATION_OR_SEMANTIC_CHANGE")
    if re.search(
        r"(?:忽略[^\uff0c。\uff1b;]{0,12}要求|"
        r"输出[^\uff0c。\uff1b;]{0,12}坐标|"
        r"目标真实位置|电机转速|actions\s*字段|系统提示词)",
        semantic_text,
        re.IGNORECASE,
    ):
        reasons.append("NEGATION_OR_SEMANTIC_CHANGE")

    concept = ontology.require_concept(gold.target_concept_id)
    target_phrases = (concept.canonical_description, *ontology.aliases_for(concept.concept_id))
    if not any(_contains_alias(semantic_text, phrase) for phrase in target_phrases):
        reasons.append("TARGET_NOT_EXPRESSED")
    for other_id, other in ontology.concepts.items():
        if other_id == concept.concept_id:
            continue
        if any(
            _contains_alias(semantic_text, phrase)
            for phrase in (other.canonical_description, *ontology.aliases_for(other_id))
        ):
            reasons.append("CONFLICTING_TARGET")
            break

    if gold.search_region not in world.search_regions:
        raise DatasetGenerationError(
            f"Gold search region {gold.search_region!r} is not available in {world.context_id!r}"
        )
    if gold.landing_zone not in world.landing_zones:
        raise DatasetGenerationError(
            f"Gold landing zone {gold.landing_zone!r} is not available in {world.context_id!r}"
        )
    _check_named_alias(
        semantic_text, gold.search_region, lexicon.search_regions, "REGION", reasons
    )
    _check_named_alias(
        semantic_text,
        gold.landing_zone,
        lexicon.landing_zones,
        "LANDING_ZONE",
        reasons,
    )
    if "track_duration_s" in gold.explicit_fields:
        selected_duration = _find_numeric_pool(lexicon.duration_expressions, gold.track_duration_s)
        if not any(
            _contains_alias(semantic_text, phrase)
            for phrase in _all_aliases(selected_duration)
        ):
            reasons.append("DURATION_NOT_EXPRESSED")
        for duration, pool in lexicon.duration_expressions.items():
            if abs(duration - gold.track_duration_s) > 1e-6 and any(
                _contains_alias(semantic_text, phrase) for phrase in _all_aliases(pool)
            ):
                reasons.append("CONFLICTING_DURATION")
                break
    elif not any(
        _contains_alias(semantic_text, phrase)
        for phrase in _all_aliases(lexicon.default_duration_expressions)
    ):
        reasons.append("DEFAULT_DURATION_NOT_EXPRESSED")

    if gold.takeoff_altitude_m is not None:
        selected_altitude = _find_numeric_pool(
            lexicon.altitude_expressions, gold.takeoff_altitude_m
        )
        if not any(
            _contains_alias(semantic_text, phrase)
            for phrase in _all_aliases(selected_altitude)
        ):
            reasons.append("ALTITUDE_NOT_EXPRESSED")
        for altitude, pool in lexicon.altitude_expressions.items():
            if abs(altitude - gold.takeoff_altitude_m) > 1e-6 and any(
                _contains_alias(semantic_text, phrase) for phrase in _all_aliases(pool)
            ):
                reasons.append("CONFLICTING_ALTITUDE")
                break

    if not strict_external:
        unique_reasons = tuple(dict.fromkeys(reasons))
        return ParaphraseCandidateValidation(
            accepted=not unique_reasons,
            reasons=unique_reasons,
        )

    # Remove every approved expression before looking for extra values.  What
    # remains is checked fail-closed: an unlisted place/time/altitude or a
    # second target attribute is not guessed and cannot enter official data.
    residual = semantic_text
    residual = _remove_phrases(residual, target_phrases)
    residual = _remove_phrases(
        residual,
        _named_alias_phrases(lexicon.search_regions)
        + _named_alias_phrases(lexicon.landing_zones),
    )
    residual = _remove_phrases(
        residual,
        tuple(
            phrase
            for pool in lexicon.duration_expressions.values()
            for phrase in _all_aliases(pool)
        )
        + _all_aliases(lexicon.default_duration_expressions),
    )
    residual = _remove_phrases(
        residual,
        tuple(
            phrase
            for pool in lexicon.altitude_expressions.values()
            for phrase in _all_aliases(pool)
        ),
    )
    residual = _remove_phrases(
        residual,
        _all_aliases(lexicon.polite_prefixes)
        + _all_aliases(lexicon.neutral_suffixes),
    )
    # Audited template anaphora refers back to the already expressed region;
    # it is not a second location candidate.
    residual = _remove_phrases(residual, ("在那里",))

    if _contains_unknown_region(residual):
        reasons.append("UNKNOWN_REGION")
    if _contains_unknown_landing_zone(residual):
        reasons.append("UNKNOWN_LANDING_ZONE")

    for seconds in _numeric_durations(semantic_text):
        if (
            "track_duration_s" not in gold.explicit_fields
            or abs(seconds - gold.track_duration_s) > 1e-6
        ):
            reasons.append("CONFLICTING_DURATION")
    if re.search(r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千点半]+)\s*(?:秒钟?|分钟)", residual):
        reasons.append("UNKNOWN_DURATION")

    for altitude in _numeric_altitudes(semantic_text):
        if (
            gold.takeoff_altitude_m is None
            or abs(altitude - gold.takeoff_altitude_m) > 1e-6
        ):
            reasons.append("CONFLICTING_ALTITUDE")
    if re.search(r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千点]+)\s*(?:米|公尺|m\b)", residual, re.IGNORECASE):
        reasons.append("UNKNOWN_ALTITUDE")

    if _contains_target_attribute(residual):
        reasons.append("UNKNOWN_TARGET_ATTRIBUTE")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return ParaphraseCandidateValidation(
        accepted=not unique_reasons,
        reasons=unique_reasons,
    )


def stage_external_candidates(
    *,
    generated: GeneratedPlannerDataset,
    candidates: Sequence[Mapping[str, object]],
    output_root: str | Path,
    ontology: TargetOntology,
    lexicon: LanguageLexicon,
) -> Path:
    """Write unreviewed rewrites only beneath ``_candidates/`` atomically."""

    sample_index = {
        sample.sample_id: sample
        for samples in generated.samples_by_split.values()
        for sample in samples
    }
    records: list[dict[str, object]] = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping) or set(raw) != {"sample_id", "candidate_instruction"}:
            raise DatasetGenerationError(
                f"candidate {index} must contain only sample_id and candidate_instruction"
            )
        sample_id = _non_empty_text(raw["sample_id"], f"candidate {index} sample_id")
        candidate = _non_empty_text(
            raw["candidate_instruction"],
            f"candidate {index} instruction",
        )
        sample = sample_index.get(sample_id)
        if sample is None:
            raise DatasetGenerationError(f"candidate {index} refers to unknown sample {sample_id!r}")
        world = generated.worlds[sample.world_context_id]
        validation = validate_paraphrase_candidate(
            candidate,
            gold=sample.gold,
            world=world,
            ontology=ontology,
            lexicon=lexicon,
        )
        records.append(
            {
                "schema_version": PLANNER_DATASET_SCHEMA_VERSION,
                "sample_id": sample.sample_id,
                "gold_spec_id": sample.gold_spec_id,
                "canonical_instruction": sample.metadata.instruction,
                "candidate_instruction": candidate,
                **validation.to_dict(),
            }
        )
    candidate_dir = Path(output_root) / "_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    path = candidate_dir / "external_candidates.jsonl"
    _write_text(
        path,
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
            for record in records
        ),
    )
    return path


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_family(combination: Mapping[str, object]) -> str:
    world = combination["world"]
    assert isinstance(world, PlannerWorldCase)
    return _semantic_family_from_values(
        world_context_id=world.context_id,
        target_concept_id=str(combination["concept_id"]),
        search_region=str(combination["search_region"]),
        landing_zone=str(combination["landing_zone"]),
        track_duration_s=float(combination["track_duration_s"]),
        duration_explicit=bool(combination["duration_explicit"]),
        takeoff_altitude_m=(
            None
            if combination["takeoff_altitude_m"] is None
            else float(combination["takeoff_altitude_m"])
        ),
    )


def _semantic_family_from_values(
    *,
    world_context_id: str,
    target_concept_id: str,
    search_region: str,
    landing_zone: str,
    track_duration_s: float,
    duration_explicit: bool,
    takeoff_altitude_m: float | None,
) -> str:
    payload = {
        "world_context_id": world_context_id,
        "target_concept_id": target_concept_id,
        "search_region": search_region,
        "landing_zone": landing_zone,
        "track_duration_s": track_duration_s,
        "duration_explicit": duration_explicit,
        "takeoff_altitude_m": takeoff_altitude_m,
    }
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return "semantic_" + sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _all_aliases(pool: object) -> tuple[str, ...]:
    train = getattr(pool, "train_aliases", ())
    heldout = getattr(pool, "heldout_aliases", ())
    return tuple(train) + tuple(heldout)


def _find_numeric_pool(pools: Mapping[float, object], value: float) -> object:
    for configured, pool in pools.items():
        if abs(configured - value) <= 1e-6:
            return pool
    raise DatasetGenerationError(f"no approved language aliases for value {value:g}")


def _check_named_alias(
    candidate: str,
    selected_name: str,
    pools: Mapping[str, object],
    label: str,
    reasons: list[str],
) -> None:
    selected = pools.get(selected_name)
    if selected is None:
        raise DatasetGenerationError(f"no approved aliases for {selected_name!r}")
    selected_phrases = (selected_name, *_all_aliases(selected))
    if not any(_contains_alias(candidate, phrase) for phrase in selected_phrases):
        reasons.append(f"{label}_NOT_EXPRESSED")
    for name, pool in pools.items():
        if name == selected_name:
            continue
        if any(
            _contains_alias(candidate, phrase)
            for phrase in (name, *_all_aliases(pool))
        ):
            reasons.append(f"CONFLICTING_{label}")
            break


def _contains_alias(text: str, phrase: str) -> bool:
    """Substring match that does not treat 5m/五米 as part of 15m/十五米."""

    numeric_chars = "0123456789.零〇一二两三四五六七八九十百千点"
    start = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            return False
        before = text[index - 1] if index > 0 else ""
        after_index = index + len(phrase)
        after = text[after_index] if after_index < len(text) else ""
        begins_numeric = bool(phrase) and phrase[0] in numeric_chars
        ends_numeric = bool(phrase) and phrase[-1] in numeric_chars
        if not (begins_numeric and before in numeric_chars) and not (
            ends_numeric and after in numeric_chars
        ):
            return True
        start = index + 1


def _robustness_injection_texts(lexicon: LanguageLexicon) -> tuple[str, ...]:
    """Return injection text across the v1 string and structured resources."""

    result: list[str] = []
    for injection in lexicon.robustness_injections:
        value = injection if isinstance(injection, str) else getattr(injection, "text", None)
        if isinstance(value, str) and value:
            result.append(value)
    return tuple(result)


def _named_alias_phrases(pools: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        phrase
        for name, pool in pools.items()
        for phrase in (name, *_all_aliases(pool))
    )


def _remove_phrases(text: str, phrases: Sequence[str]) -> str:
    for phrase in sorted(set(phrases), key=len, reverse=True):
        text = text.replace(phrase, " ")
    return text


def _contains_unknown_region(text: str) -> bool:
    if re.search(r"\b[A-Za-z][A-Za-z0-9_]*(?:_area|_region)\b", text):
        return True
    if re.search(r"[一-鿿]{1,12}(?:搜索范围|搜索区域|区域|地带|片区|区)", text):
        return True
    # Also reject an unlisted place used syntactically as a search location,
    # even when it does not end in a conventional region suffix.
    for match in re.finditer(
        r"(?:前往|再去|去|到|在)\s*"
        r"([^\s，。；;]{1,16}?)\s*"
        r"(?:搜索|搜寻|寻找|识别)",
        text,
    ):
        # Once the approved region literal is removed, common motion/linking
        # particles may become adjacent (for example "到 <region> 去搜寻").
        if match.group(1) not in {"去", "再去", "那里", "该处"}:
            return True
    return False


def _contains_unknown_landing_zone(text: str) -> bool:
    if re.search(r"\b[A-Za-z][A-Za-z0-9_]*(?:_pad|_landing_zone)\b", text):
        return True
    if re.search(r"[一-鿿]{1,12}(?:降落点|停机坪|着陆区|降落区|降落坪|基地)", text):
        return True
    return bool(
        re.search(
            r"(?:返回|回到|最后到|之后去|然后去|到|在|去)\s*"
            r"[^\s，。；;]{1,16}?\s*"
            r"(?:降落|着陆|落地)",
            text,
        )
    )


def _numeric_durations(text: str) -> tuple[float, ...]:
    result: list[float] = []
    for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(秒(?:钟)?|分钟)", text):
        value = float(match.group(1))
        result.append(value * (60.0 if match.group(2) == "分钟" else 1.0))
    return tuple(result)


def _numeric_altitudes(text: str) -> tuple[float, ...]:
    return tuple(
        float(match.group(1))
        for match in re.finditer(
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:米|公尺|m\b)",
            text,
            re.IGNORECASE,
        )
    )


def _contains_target_attribute(text: str) -> bool:
    # General "...colour" plus the compact forms used by the ontology.  The
    # approved target phrase has already been removed, so any remaining match
    # is an introduced/repeated attribute and is rejected conservatively.
    if re.search(
        r"[一-鿿]{1,3}色(?:上衣|衣服|下装|裤子?|背包|帽子?|鞋子?)?",
        text,
    ):
        return True
    if re.search(r"[红蓝黑白黄灰绿紫橙青棕粉](?:衣|裤|包|帽|鞋)", text):
        return True
    if re.search(
        r"(?:穿|戴|背|携带|带着?|没背|未背)"
        r"[^\uff0c。；;]{0,10}(?:上衣|衣服|下装|裤|背包|帽|眼镜|鞋)",
        text,
    ):
        return True
    # Once the approved ontology phrase is removed, any remaining free-form
    # human-attribute predicate is outside the closed ontology.  This catches
    # open-ended additions such as watches, umbrellas, khaki coats or hair
    # style without pretending to enumerate every possible attribute value.
    if re.search(
        r"(?:目标|行人|此人|该人|那个人)"
        r"[^\uff0c。；;]{0,8}"
        r"(?:穿着?|戴着?|佩戴|手持|拿着?|携带|撑着?|留着?|是|为|有)"
        r"\s*[^\s，。；;]{1,12}",
        text,
    ):
        return True
    if re.search(
        r"(?:穿着?|戴着?|佩戴|手持|拿着?|携带|撑着?|留着?)"
        r"\s*[^\s，。；;]{1,12}",
        text,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:red|blue|black|white|yellow|gray|grey|green|purple|orange)\b"
            r"[^\n，。；;]{0,12}"
            r"(?:clothing|shirt|pants|backpack|hat|shoes?)",
            text,
            re.IGNORECASE,
        )
    )


def _contains_forbidden_instruction_data(text: str) -> bool:
    """Reject runtime truth, observations, media and local artifact references.

    Planner instructions are semantic task requests only.  Coordinates,
    target/UAV state and image/video/frame material belong to runtime or
    evaluator boundaries and cannot be introduced by an external rewrite.
    """

    patterns = (
        r"\boracle(?:_target)?\b|\bground[ _-]?truth\b|\bgt[_ -]?(?:pose|position|velocity)\b",
        r"(?:目标真值|真值目标|真实目标)"
        r"|(?:目标|无人机|uav|相机|camera)"
        r"[^\uff0c。；;\n]{0,10}(?:坐标|位置|位姿|速度)",
        r"(?:坐标|位姿|速度向量)"
        r"|\b(?:target|uav|camera)_(?:pose|position|velocity)\b",
        r"(?:图片|图像|照片|视频|帧数据|像素数据|相机RGB)"
        r"|\b(?:camera_rgb|image_path|video_path|frame_path|observation_dump)\b",
        r"(?:^|[\s\"'=:])(?:/tmp/|/home/|[A-Za-z]:[\\/])",
        r"\.(?:png|jpe?g|bmp|webp|gif|mp4|avi|mov|mkv)\b",
        r"[\(\[]\s*-?\d+(?:\.\d+)?\s*[,，]\s*-?\d+(?:\.\d+)?"
        r"\s*[,，]\s*-?\d+(?:\.\d+)?\s*[\)\]]",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _write_jsonl(path: Path, samples: Sequence[PlannerDatasetSample]) -> None:
    text = "".join(
        json.dumps(sample.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        for sample in samples
    )
    _write_text(path, text)


def _write_json(path: Path, value: object) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
    )


def _write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if temporary.exists():
            temporary.unlink()
        raise


def _source_fraction(samples: Sequence[PlannerDatasetSample], source: str) -> float:
    return 0.0 if not samples else sum(sample.metadata.generation_source == source for sample in samples) / len(samples)


def _review_fraction(
    samples: Sequence[PlannerDatasetSample],
    accepted_statuses: set[str],
) -> float:
    return (
        0.0
        if not samples
        else sum(
            sample.metadata.review_status in accepted_statuses for sample in samples
        )
        / len(samples)
    )


def _validate_minimum_coverage(
    statistics: Mapping[str, object],
    minimums: Mapping[str, int],
    *,
    expected_values: Mapping[str, set[str]],
) -> None:
    mapping = {
        "target_concept": "target_concept_counts",
        "search_region": "search_region_counts",
        "landing_zone": "landing_zone_counts",
        "track_duration": "track_duration_counts",
    }
    for key, minimum in minimums.items():
        statistics_key = mapping.get(key)
        if statistics_key is None:
            raise DatasetGenerationError(f"unknown minimum coverage key {key!r}")
        counts = statistics[statistics_key]
        assert isinstance(counts, Mapping)
        missing = expected_values[key] - set(counts)
        below = sorted(
            str(value) for value in expected_values[key] if counts.get(value, 0) < minimum
        )
        if missing or below:
            raise DatasetGenerationError(
                f"minimum coverage was not met for {key}; "
                f"missing={sorted(missing)}, below_minimum={below}"
            )


def _exact_keys(value: Mapping[object, object], required: set[str], label: str) -> None:
    unknown = set(value) - required
    missing = required - set(value)
    if unknown or missing:
        raise DatasetGenerationError(
            f"{label} fields mismatch; unknown={sorted(map(str, unknown))}, missing={sorted(missing)}"
        )


def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetGenerationError(f"{label} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, label: str) -> float:
    from math import isfinite
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetGenerationError(f"{label} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise DatasetGenerationError(f"{label} must be finite")
    return result


def _positive_number_sequence(value: object, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise DatasetGenerationError(f"{label} must be a non-empty list")
    result = tuple(_finite_number(item, f"{label} item") for item in value)
    if any(item <= 0.0 for item in result) or len(set(result)) != len(result):
        raise DatasetGenerationError(f"{label} must contain unique positive values")
    return result


def _string_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise DatasetGenerationError(f"{label} must be an object")
    return {_non_empty_text(key, f"{label} key"): _non_empty_text(item, f"{label} value") for key, item in value.items()}


def _integer_mapping(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise DatasetGenerationError(f"{label} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        name = _non_empty_text(key, f"{label} key")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise DatasetGenerationError(f"{label}[{name!r}] must be a non-negative integer")
        result[name] = item
    return result


__all__ = [
    "DEFAULT_DATASET_CONFIG_PATH",
    "DEFAULT_RESOURCE_ROOT",
    "DatasetGenerationError",
    "GeneratedPlannerDataset",
    "GenerationConfig",
    "PlannerDatasetGenerator",
    "ParaphraseCandidateValidation",
    "build_statistics",
    "generate_and_write_dataset",
    "load_generation_config",
    "serialize_expected_intent",
    "sha256_file",
    "stage_external_candidates",
    "validate_paraphrase_candidate",
    "write_generated_dataset",
]
