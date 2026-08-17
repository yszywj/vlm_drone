"""Strict, standalone validation for published Planner datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re

from planner.prompt_builder import build_mission_planner_messages
from planner.schemas import MissionIntent
from tasks.target_ontology import TargetOntology

from .generator import (
    DEFAULT_DATASET_CONFIG_PATH,
    DEFAULT_SYSTEM_PROMPT_PATH,
    DatasetGenerationError,
    PlannerDatasetGenerator,
    build_statistics,
    serialize_expected_intent,
    sha256_file,
    validate_paraphrase_candidate,
)
from .leakage_checker import check_dataset_leakage
from .renderers import world_case_to_runtime_context
from .schemas import (
    DatasetManifest,
    PLANNER_DATASET_SCHEMA_VERSION,
    PLANNER_DATASET_SPLITS,
    PlannerDatasetSample,
    compute_semantic_spec_family,
    make_group_id,
)
from .splitter import SplitterError, validate_splits


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be trusted as Planner v1 Gold data."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    path: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "line": self.line,
        }


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    dataset_root: str
    num_samples: int
    split_counts: Mapping[str, int]
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def require_valid(self) -> None:
        if not self.valid:
            detail = "; ".join(
                f"{issue.code}: {issue.message}" for issue in self.issues[:10]
            )
            raise DatasetValidationError(detail)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_root": self.dataset_root,
            "valid": self.valid,
            "num_samples": self.num_samples,
            "split_counts": dict(self.split_counts),
            "issues": [issue.to_dict() for issue in self.issues],
        }


_FORBIDDEN_KEY_FRAGMENTS = (
    "oracle",
    "target_spawn",
    "target_position",
    "target_pose",
    "target_velocity",
    "evaluator_frame",
    "camera_rgb",
    "image_path",
    "video_path",
    "frame_path",
    "observation_dump",
)


def load_split_samples(
    dataset_root: str | Path,
    split: str,
) -> tuple[PlannerDatasetSample, ...]:
    if split not in PLANNER_DATASET_SPLITS:
        raise DatasetValidationError(f"unknown split {split!r}")
    path = Path(dataset_root) / f"{split}.jsonl"
    samples: list[PlannerDatasetSample] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise DatasetValidationError(
                        f"{path.name}:{line_number} contains a blank JSONL line"
                    )
                raw = _strict_json_loads(line)
                if not isinstance(raw, Mapping):
                    raise DatasetValidationError(
                        f"{path.name}:{line_number} must contain one JSON object"
                    )
                samples.append(PlannerDatasetSample.from_dict(raw))
    except (OSError, UnicodeError) as exc:
        raise DatasetValidationError(f"could not read {path.name}: {exc}") from exc
    return tuple(samples)


class PlannerDatasetValidator:
    """Rebuild every trusted invariant instead of trusting stored metadata."""

    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
        system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
    ) -> None:
        self._generator = PlannerDatasetGenerator(
            config_path,
            system_prompt_path=system_prompt_path,
        )
        self._config_path = Path(config_path)
        self._system_prompt_path = Path(system_prompt_path)
        self._system_prompt = self._generator.system_prompt
        self._ontology: TargetOntology = self._generator.ontology
        self._worlds = self._generator.worlds

    def validate(self, dataset_root: str | Path) -> DatasetValidationReport:
        root = Path(dataset_root)
        issues: list[ValidationIssue] = []
        manifest = self._load_manifest(root, issues)
        samples_by_split: dict[str, tuple[PlannerDatasetSample, ...]] = {}
        for split in PLANNER_DATASET_SPLITS:
            path = root / f"{split}.jsonl"
            samples_by_split[split] = self._load_and_validate_split(path, split, issues)

        all_samples = tuple(
            sample for split in PLANNER_DATASET_SPLITS for sample in samples_by_split[split]
        )
        if manifest is not None:
            if manifest.dataset_name != "planner_v1":
                issues.append(
                    ValidationIssue(
                        "MANIFEST_DATASET_NAME_INVALID",
                        "dataset_name must be 'planner_v1'",
                        "dataset_manifest.json",
                    )
                )
            profile = self._generator.config.profiles.get(manifest.profile)
            if profile is None:
                issues.append(
                    ValidationIssue(
                        "MANIFEST_PROFILE_INVALID",
                        "profile must be one of: pilot, full",
                        "dataset_manifest.json",
                    )
                )
            elif dict(manifest.split_counts) != dict(profile.split_counts):
                issues.append(
                    ValidationIssue(
                        "MANIFEST_PROFILE_COUNTS_MISMATCH",
                        f"split_counts do not match configured {manifest.profile!r} profile",
                        "dataset_manifest.json",
                    )
                )
            actual_counts = {split: len(samples_by_split[split]) for split in PLANNER_DATASET_SPLITS}
            if actual_counts != dict(manifest.split_counts):
                issues.append(
                    ValidationIssue(
                        "SPLIT_COUNT_MISMATCH",
                        f"actual counts {actual_counts} differ from manifest {dict(manifest.split_counts)}",
                        "dataset_manifest.json",
                    )
                )
            self._validate_hashes(root, manifest, issues)

        if not any(issue.code == "SAMPLE_SCHEMA_INVALID" for issue in issues):
            try:
                expected = None if manifest is None else manifest.split_counts
                validate_splits(
                    samples_by_split,
                    expected_counts=expected,
                    enforce_holdouts=True,
                )
            except (SplitterError, TypeError, ValueError) as exc:
                issues.append(ValidationIssue("SPLIT_LEAKAGE", str(exc)))
            leakage = check_dataset_leakage(all_samples)
            for leakage_issue in leakage.issues:
                issues.append(
                    ValidationIssue(leakage_issue.code, leakage_issue.message)
                )

        self._validate_statistics(root, samples_by_split, manifest, issues)
        issues.sort(
            key=lambda issue: (
                issue.path or "",
                issue.line if issue.line is not None else -1,
                issue.code,
                issue.message,
            )
        )
        return DatasetValidationReport(
            dataset_root=root.name,
            num_samples=len(all_samples),
            split_counts={split: len(samples_by_split[split]) for split in PLANNER_DATASET_SPLITS},
            issues=tuple(issues),
        )

    def _load_manifest(
        self,
        root: Path,
        issues: list[ValidationIssue],
    ) -> DatasetManifest | None:
        path = root / "dataset_manifest.json"
        try:
            raw = _strict_json_loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise TypeError("manifest must be an object")
            return DatasetManifest.from_dict(raw)
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                ValidationIssue("MANIFEST_INVALID", str(exc), path.name)
            )
            return None

    def _load_and_validate_split(
        self,
        path: Path,
        split: str,
        issues: list[ValidationIssue],
    ) -> tuple[PlannerDatasetSample, ...]:
        samples: list[PlannerDatasetSample] = []
        try:
            stream = path.open("r", encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            issues.append(ValidationIssue("SPLIT_FILE_MISSING", str(exc), path.name))
            return ()
        with stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    issues.append(
                        ValidationIssue(
                            "JSONL_INVALID",
                            "blank lines are not allowed",
                            path.name,
                            line_number,
                        )
                    )
                    continue
                try:
                    raw = _strict_json_loads(line)
                    if not isinstance(raw, Mapping):
                        raise TypeError("JSONL row must be an object")
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(
                        ValidationIssue("JSONL_INVALID", str(exc), path.name, line_number)
                    )
                    continue
                forbidden = sorted(_find_forbidden_keys(raw))
                if forbidden:
                    issues.append(
                        ValidationIssue(
                            "FORBIDDEN_FIELD",
                            "forbidden data keys: " + ", ".join(forbidden),
                            path.name,
                            line_number,
                        )
                    )
                try:
                    sample = PlannerDatasetSample.from_dict(raw)
                except (TypeError, ValueError) as exc:
                    issues.append(
                        ValidationIssue(
                            "SAMPLE_SCHEMA_INVALID",
                            str(exc),
                            path.name,
                            line_number,
                        )
                    )
                    continue
                if sample.split != split:
                    issues.append(
                        ValidationIssue(
                            "SPLIT_DECLARATION_MISMATCH",
                            f"row declares {sample.split!r}, expected {split!r}",
                            path.name,
                            line_number,
                        )
                    )
                self._validate_sample(sample, path.name, line_number, issues)
                samples.append(sample)
        return tuple(samples)

    def _validate_sample(
        self,
        sample: PlannerDatasetSample,
        path: str,
        line: int,
        issues: list[ValidationIssue],
    ) -> None:
        gold_is_valid = True
        try:
            self._ontology.validate_gold_spec(sample.gold)
        except (TypeError, ValueError) as exc:
            gold_is_valid = False
            issues.append(ValidationIssue("GOLD_INVALID", str(exc), path, line))
        world = self._worlds.get(sample.world_context_id)
        if world is None:
            issues.append(
                ValidationIssue(
                    "UNKNOWN_WORLD_CONTEXT",
                    f"unknown world {sample.world_context_id!r}",
                    path,
                    line,
                )
            )
            return
        expected_semantic_family = compute_semantic_spec_family(
            sample.gold,
            world.context_id,
        )
        if sample.metadata.semantic_spec_family != expected_semantic_family:
            issues.append(
                ValidationIssue(
                    "SEMANTIC_SPEC_FAMILY_INVALID",
                    "metadata.semantic_spec_family was not independently derived from Gold + world",
                    path,
                    line,
                )
            )
        expected_group_id = make_group_id(
            expected_semantic_family,
            sample.metadata.template_family,
            sample.metadata.paraphrase_family,
        )
        if sample.metadata.group_id != expected_group_id:
            issues.append(
                ValidationIssue(
                    "GROUP_ID_INVALID",
                    "metadata.group_id is not the deterministic Gold/template/paraphrase key",
                    path,
                    line,
                )
            )
        if sample.gold.search_region not in world.search_regions:
            issues.append(
                ValidationIssue(
                    "UNKNOWN_SEARCH_REGION",
                    f"region {sample.gold.search_region!r} is not in this world",
                    path,
                    line,
                )
            )
        if sample.gold.landing_zone not in world.landing_zones:
            issues.append(
                ValidationIssue(
                    "UNKNOWN_LANDING_ZONE",
                    f"landing zone {sample.gold.landing_zone!r} is not in this world",
                    path,
                    line,
                )
            )
        required_explicit = {
            "target_description",
            "search_region",
            "landing_zone",
        }
        missing_explicit = required_explicit - sample.gold.explicit_fields
        if missing_explicit:
            issues.append(
                ValidationIssue(
                    "GOLD_EXPLICIT_FIELDS_INVALID",
                    "these Planner v1 fields must be explicitly expressed: "
                    + ", ".join(sorted(missing_explicit)),
                    path,
                    line,
                )
            )
        if (
            "track_duration_s" not in sample.gold.explicit_fields
            and abs(sample.gold.track_duration_s - world.default_track_duration_s) > 1e-6
        ):
            issues.append(
                ValidationIssue(
                    "IMPLICIT_DURATION_NOT_DEFAULT",
                    "an omitted duration must store the trusted world default",
                    path,
                    line,
                )
            )
        if (
            gold_is_valid
            and sample.gold.search_region in world.search_regions
            and sample.gold.landing_zone in world.landing_zones
        ):
            alias_errors = self._validate_recorded_aliases(sample)
            if alias_errors:
                issues.append(
                    ValidationIssue(
                        "INSTRUCTION_ALIAS_METADATA_INVALID",
                        "; ".join(alias_errors),
                        path,
                        line,
                    )
                )
            try:
                grounded = validate_paraphrase_candidate(
                    sample.metadata.instruction,
                    gold=sample.gold,
                    world=world,
                    ontology=self._ontology,
                    lexicon=self._generator.lexicon,
                    allow_approved_robustness=(sample.split == "test_robustness"),
                    strict_external=True,
                )
            except (DatasetGenerationError, TypeError, ValueError) as exc:
                issues.append(
                    ValidationIssue(
                        "INSTRUCTION_VALIDATION_FAILED",
                        str(exc),
                        path,
                        line,
                    )
                )
            else:
                if "FORBIDDEN_RUNTIME_DATA" in grounded.reasons:
                    issues.append(
                        ValidationIssue(
                            "FORBIDDEN_INSTRUCTION_CONTENT",
                            "instruction contains runtime truth, state, media, frame, "
                            "or local artifact data",
                            path,
                            line,
                        )
                    )
                if not grounded.accepted:
                    issues.append(
                        ValidationIssue(
                            "INSTRUCTION_GOLD_MISMATCH",
                            "instruction does not express exactly its Gold semantics: "
                            + ", ".join(grounded.reasons),
                            path,
                            line,
                        )
                    )
        assistant = sample.messages[2].content
        try:
            parsed_raw = _strict_json_loads(str(assistant))
            if not isinstance(parsed_raw, Mapping):
                raise TypeError("assistant label must be one JSON object")
            predicted = MissionIntent.from_dict(parsed_raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                ValidationIssue("ASSISTANT_LABEL_INVALID", str(exc), path, line)
            )
        else:
            if (
                predicted != sample.gold.to_expected_intent()
                or str(assistant) != serialize_expected_intent(sample.gold)
            ):
                issues.append(
                    ValidationIssue(
                        "ASSISTANT_GOLD_MISMATCH",
                        "assistant label is not the deterministic Gold expected intent",
                        path,
                        line,
                    )
                )
        try:
            expected_prompt = build_mission_planner_messages(
                sample.metadata.instruction,
                world_case_to_runtime_context(world),
                self._system_prompt,
            )
        except (TypeError, ValueError) as exc:
            issues.append(ValidationIssue("PROMPT_BUILD_FAILED", str(exc), path, line))
        else:
            if sample.messages[:2] != expected_prompt:
                issues.append(
                    ValidationIssue(
                        "PROMPT_MISMATCH",
                        "system/user messages differ from the runtime Prompt Builder",
                        path,
                        line,
                    )
                )

    def _validate_recorded_aliases(
        self,
        sample: PlannerDatasetSample,
    ) -> tuple[str, ...]:
        """Bind exact renderer-recorded phrases back to their Gold fields."""

        aliases: dict[str, str] = {}
        errors: list[str] = []
        for feature in sample.metadata.language_feature_ids:
            if not feature.startswith("alias:"):
                continue
            parts = feature.split(":", 2)
            if len(parts) != 3 or not parts[1] or not parts[2]:
                errors.append(f"malformed alias feature {feature!r}")
                continue
            role, value = parts[1], parts[2]
            if role in aliases:
                errors.append(f"duplicate alias role {role!r}")
                continue
            aliases[role] = value
            if value not in sample.metadata.instruction:
                errors.append(f"recorded {role} alias is absent from instruction")

        expected_features = tuple(
            sorted(
                {
                    f"paraphrase:{sample.metadata.paraphrase_family}",
                    *(f"alias:{role}:{value}" for role, value in aliases.items()),
                }
            )
        )
        if sample.metadata.language_feature_ids != expected_features:
            errors.append(
                "language_feature_ids are not the exact deterministic alias/paraphrase record"
            )

        expected_roles = {
            "target",
            "search_region",
            "track_duration",
            "landing_zone",
        }
        if sample.gold.takeoff_altitude_m is not None:
            expected_roles.add("takeoff_altitude")
        if sample.split == "test_robustness":
            expected_roles.add("robustness_injection")
        if set(aliases) != expected_roles:
            errors.append(
                "alias roles differ from expected roles: expected "
                + ", ".join(sorted(expected_roles))
                + "; got "
                + ", ".join(sorted(aliases))
            )

        concept = self._ontology.require_concept(sample.gold.target_concept_id)
        target = aliases.get("target")
        permitted_targets = _target_values_for_split(
            concept.canonical_description,
            self._ontology.aliases_for(concept.concept_id),
            sample.split,
        )
        if target is not None and target not in permitted_targets:
            errors.append("target alias is not permitted for Gold concept in this split")

        expected_components = tuple(
            f"{key}={value}" for key, value in sorted(concept.attributes.items())
        )
        if sample.metadata.composition_components != expected_components:
            errors.append("composition_components differ from ontology attributes")

        lexicon = self._generator.lexicon
        region = aliases.get("search_region")
        if region is not None and region not in _pool_values_for_split(
            lexicon.search_regions.get(sample.gold.search_region),
            sample.split,
        ):
            errors.append("search_region alias is not permitted for Gold region in this split")
        landing = aliases.get("landing_zone")
        if landing is not None and landing not in _pool_values_for_split(
            lexicon.landing_zones.get(sample.gold.landing_zone),
            sample.split,
        ):
            errors.append("landing_zone alias is not permitted for Gold landing zone in this split")

        duration = aliases.get("track_duration")
        if "track_duration_s" in sample.gold.explicit_fields:
            duration_pool = _numeric_pool_for_value(
                lexicon.duration_expressions,
                sample.gold.track_duration_s,
            )
        else:
            duration_pool = lexicon.default_duration_expressions
        if duration is not None and duration not in _pool_values_for_split(
            duration_pool, sample.split
        ):
            errors.append(
                "track_duration alias is not permitted for Gold duration in this split"
            )

        altitude = aliases.get("takeoff_altitude")
        if sample.gold.takeoff_altitude_m is not None:
            altitude_pool = _numeric_pool_for_value(
                lexicon.altitude_expressions,
                sample.gold.takeoff_altitude_m,
            )
            if altitude is not None and altitude not in _pool_values_for_split(
                altitude_pool, sample.split
            ):
                errors.append(
                    "takeoff_altitude alias is not permitted for Gold altitude in this split"
                )

        injection = aliases.get("robustness_injection")
        if injection is not None:
            approved = {
                item.text: item.category for item in lexicon.robustness_injections
            }
            if injection not in approved:
                errors.append("robustness injection is not in the approved resource")
            elif approved[injection] != sample.metadata.robustness_category:
                errors.append("robustness injection category differs from metadata")

        duration_mode = (
            "explicit"
            if "track_duration_s" in sample.gold.explicit_fields
            else "default"
        )
        altitude_mode = (
            "explicit" if sample.gold.takeoff_altitude_m is not None else "default"
        )
        matching_templates = tuple(
            template
            for template in self._generator.templates.templates
            if template.template_family == sample.metadata.template_family
            and template.paraphrase_family == sample.metadata.paraphrase_family
            and sample.split in template.splits
            and template.duration_mode == duration_mode
            and template.altitude_mode == altitude_mode
        )
        if not matching_templates:
            errors.append(
                "template/paraphrase family is not split-local for the Gold explicit modes"
            )
        elif sample.metadata.generation_source == "template":
            prefix_options = (
                "",
                *_pool_values_for_split(lexicon.polite_prefixes, sample.split),
            )
            suffix_options = (
                "",
                *(
                    value + "。"
                    for value in _pool_values_for_split(
                        lexicon.neutral_suffixes, sample.split
                    )
                ),
            )
            rendered_candidates: set[str] = set()
            for template in matching_templates:
                for prefix in prefix_options:
                    for suffix in suffix_options:
                        rendered = template.text.format(
                            prefix=prefix,
                            region=aliases.get("search_region", ""),
                            target=aliases.get("target", ""),
                            duration=aliases.get("track_duration", ""),
                            landing=aliases.get("landing_zone", ""),
                            altitude=aliases.get("takeoff_altitude", ""),
                            suffix=suffix,
                            injection=aliases.get("robustness_injection", ""),
                        )
                        rendered = re.sub(r"[ \t\r\n]+", " ", rendered).strip()
                        rendered = re.sub(r"\s+([，。；：！？])", r"\1", rendered)
                        rendered_candidates.add(rendered)
            if sample.metadata.instruction not in rendered_candidates:
                errors.append(
                    "template instruction cannot be reconstructed from recorded split-local aliases"
                )

        semantic_instruction = sample.metadata.instruction
        if injection is not None:
            semantic_instruction = semantic_instruction.replace(injection, " ")
        role_candidates = {
            "target": tuple(
                value
                for ontology_concept in self._ontology.concepts.values()
                for value in (
                    ontology_concept.canonical_description,
                    *self._ontology.aliases_for(ontology_concept.concept_id),
                )
            ),
            "search_region": tuple(
                value
                for name, pool in lexicon.search_regions.items()
                for value in (name, *_pool_values(pool))
            ),
            "landing_zone": tuple(
                value
                for name, pool in lexicon.landing_zones.items()
                for value in (name, *_pool_values(pool))
            ),
            "track_duration": tuple(
                value
                for pool in lexicon.duration_expressions.values()
                for value in _pool_values(pool)
            )
            + _pool_values(lexicon.default_duration_expressions),
            "takeoff_altitude": tuple(
                value
                for pool in lexicon.altitude_expressions.values()
                for value in _pool_values(pool)
            ),
        }
        for role, recorded in aliases.items():
            candidates = role_candidates.get(role)
            if candidates is None:
                continue
            remaining = semantic_instruction.replace(recorded, " ")
            unexpected = sorted(
                {
                    value
                    for value in candidates
                    if value != recorded and _contains_literal(remaining, value)
                }
            )
            if unexpected:
                errors.append(
                    f"instruction contains unrecorded {role} aliases: "
                    + ", ".join(unexpected)
                )
        return tuple(errors)

    def _validate_hashes(
        self,
        root: Path,
        manifest: DatasetManifest,
        issues: list[ValidationIssue],
    ) -> None:
        expected_split_files = {f"{split}.jsonl" for split in PLANNER_DATASET_SPLITS}
        if set(manifest.split_sha256) != expected_split_files:
            issues.append(
                ValidationIssue(
                    "SPLIT_CHECKSUM_SET_MISMATCH",
                    "manifest must contain exactly one checksum for every public split",
                    "dataset_manifest.json",
                )
            )
        for filename, expected in manifest.split_sha256.items():
            if Path(filename).name != filename:
                issues.append(
                    ValidationIssue(
                        "SPLIT_CHECKSUM_SET_MISMATCH",
                        f"non-portable split filename {filename!r}",
                        "dataset_manifest.json",
                    )
                )
                continue
            path = root / filename
            try:
                actual = sha256_file(path)
            except OSError as exc:
                issues.append(ValidationIssue("CHECKSUM_FILE_MISSING", str(exc), filename))
                continue
            if actual != expected:
                issues.append(
                    ValidationIssue("CHECKSUM_MISMATCH", f"expected {expected}, got {actual}", filename)
                )
        resource_paths = {
            "dataset_config.yaml": self._config_path,
            **{
                self._generator.config.resource_files[key]: (
                    self._config_path.parent / self._generator.config.resource_files[key]
                )
                for key in self._generator.config.resource_files
            },
            "mission_planner_system.txt": self._system_prompt_path,
        }
        if set(manifest.resource_sha256) != set(resource_paths):
            issues.append(
                ValidationIssue(
                    "RESOURCE_SET_MISMATCH",
                    "manifest resource names do not match the validator resources",
                    "dataset_manifest.json",
                )
            )
        for name, path in resource_paths.items():
            expected = manifest.resource_sha256.get(name)
            if expected is not None and sha256_file(path) != expected:
                issues.append(
                    ValidationIssue(
                        "RESOURCE_CHECKSUM_MISMATCH",
                        f"resource {name!r} differs from generation metadata",
                        "dataset_manifest.json",
                    )
                )
        checksum_path = root / "checksums.sha256"
        try:
            declared = _parse_checksum_file(checksum_path)
        except (OSError, UnicodeError, ValueError) as exc:
            issues.append(ValidationIssue("CHECKSUM_LIST_INVALID", str(exc), checksum_path.name))
        else:
            expected_list = {
                **dict(manifest.split_sha256),
                "statistics.json": manifest.statistics_sha256,
            }
            if declared != expected_list:
                issues.append(
                    ValidationIssue(
                        "CHECKSUM_LIST_MISMATCH",
                        "checksums.sha256 differs from the manifest",
                        checksum_path.name,
                    )
                )

    def _validate_statistics(
        self,
        root: Path,
        samples_by_split: Mapping[str, Sequence[PlannerDatasetSample]],
        manifest: DatasetManifest | None,
        issues: list[ValidationIssue],
    ) -> None:
        path = root / "statistics.json"
        try:
            stored = _strict_json_loads(path.read_text(encoding="utf-8"))
            expected = build_statistics(samples_by_split)
            try:
                self._generator.validate_minimum_coverage(expected)
            except DatasetGenerationError as exc:
                issues.append(
                    ValidationIssue(
                        "MINIMUM_COVERAGE_NOT_MET",
                        str(exc),
                        path.name,
                    )
                )
            if stored != expected:
                issues.append(
                    ValidationIssue("STATISTICS_MISMATCH", "statistics.json was not recomputed from rows", path.name)
                )
            if (
                manifest is not None
                and manifest.profile == "full"
                and not bool(expected.get("full_review_requirements_met", False))
            ):
                issues.append(
                    ValidationIssue(
                        "FULL_REVIEW_REQUIREMENTS_NOT_MET",
                        "full profile requires at least 50% human-authored/reviewed "
                        "test_language and 100% human-reviewed test_robustness",
                        path.name,
                    )
                )
            if not bool(expected.get("external_candidate_policy_met", False)):
                issues.append(
                    ValidationIssue(
                        "EXTERNAL_CANDIDATE_POLICY_INVALID",
                        "external candidates must be human-reviewed, train-only, "
                        "and no more than 50% of train",
                        path.name,
                    )
                )
            if manifest is not None and sha256_file(path) != manifest.statistics_sha256:
                issues.append(
                    ValidationIssue("CHECKSUM_MISMATCH", "statistics checksum differs from manifest", path.name)
                )
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(ValidationIssue("STATISTICS_INVALID", str(exc), path.name))


def validate_dataset(
    dataset_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
) -> DatasetValidationReport:
    return PlannerDatasetValidator(config_path=config_path).validate(dataset_root)


def _pool_values(pool: object, *, canonical: str | None = None) -> tuple[str, ...]:
    if pool is None:
        return () if canonical is None else (canonical,)
    values = (
        *getattr(pool, "train_aliases", ()),
        *getattr(pool, "heldout_aliases", ()),
    )
    return values if canonical is None else (canonical, *values)


def _pool_values_for_split(pool: object, split: str) -> tuple[str, ...]:
    if pool is None:
        return ()
    attribute = "heldout_aliases" if split == "test_language" else "train_aliases"
    return tuple(getattr(pool, attribute, ()))


def _target_values_for_split(
    canonical: str,
    aliases: tuple[str, ...],
    split: str,
) -> tuple[str, ...]:
    if split == "test_language" and aliases:
        return aliases[-1:]
    ordinary_aliases = aliases[:-1] if len(aliases) >= 2 else ()
    return (canonical, *ordinary_aliases)


def _contains_literal(text: str, value: str) -> bool:
    numeric_chars = "0123456789.零〇一二两三四五六七八九十百千点"
    start = 0
    while True:
        index = text.find(value, start)
        if index < 0:
            return False
        before = text[index - 1] if index > 0 else ""
        after_index = index + len(value)
        after = text[after_index] if after_index < len(text) else ""
        numeric_boundary = (
            value[0] in numeric_chars and before in numeric_chars
        ) or (value[-1] in numeric_chars and after in numeric_chars)
        ascii_boundary = (
            value[0].isascii()
            and value[0].isalnum()
            and before
            and (before.isascii() and (before.isalnum() or before == "_"))
        ) or (
            value[-1].isascii()
            and value[-1].isalnum()
            and after
            and (after.isascii() and (after.isalnum() or after == "_"))
        )
        if not numeric_boundary and not ascii_boundary:
            return True
        start = index + 1


def _numeric_pool_for_value(
    pools: Mapping[float, object],
    value: float,
) -> object | None:
    for configured, pool in pools.items():
        if abs(configured - value) <= 1e-6:
            return pool
    return None


def _strict_json_loads(text: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


def _find_forbidden_keys(value: object, prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(fragment in key_text for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                found.add(path)
            found.update(_find_forbidden_keys(item, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            found.update(_find_forbidden_keys(item, f"{prefix}[{index}]"))
    return found


def _parse_checksum_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or not all(
            character in "0123456789abcdef" for character in parts[0]
        ):
            raise ValueError(f"invalid checksum line {line_number}")
        digest, name = parts
        if not name or Path(name).name != name or name in result:
            raise ValueError(f"invalid checksum file name on line {line_number}")
        result[name] = digest
    return result


__all__ = [
    "DatasetValidationError",
    "DatasetValidationReport",
    "PlannerDatasetValidator",
    "ValidationIssue",
    "load_split_samples",
    "validate_dataset",
]
