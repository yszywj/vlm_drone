"""Strict pure-Python validation for Fleet Planner v1 JSONL.

The validator intentionally delegates request, plan, assignment, coordination,
target, and Spatial V3 parsing to the production Fleet contracts. Dataset-only
logic is limited to record envelopes, metadata, capability labels, split
isolation, and task-coverage accounting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Callable, TypeVar

from fleet.schemas import (
    parse_fleet_assignment,
    parse_fleet_coordination_policy,
    parse_fleet_mission_plan,
    parse_fleet_mission_request,
    validate_fleet_mission_plan,
)
from fleet.types import (
    FleetMissionPlan,
    FleetMissionRequest,
    FleetPlanPatch,
)


FLEET_DATASET_SPLITS = (
    "train",
    "validation",
    "test_iid",
    "test_language",
    "test_compositional",
    "test_conflict",
    "test_reassignment",
)
FLEET_DATASET_SCENARIOS = frozenset(
    {
        "explicit_assignment",
        "natural_alias",
        "sector_region",
        "polygon_region",
        "corridor_region",
        "different_track_duration",
        "surplus_uav",
        "more_tasks_than_uavs",
        "unavailable_uav",
        "capability_mismatch",
        "duplicate_target_request",
        "similar_targets",
        "overlapping_regions_auto_assignment",
        "failed_assignment_reassignment",
    }
)
FLEET_DATASET_SPLIT_SCENARIOS = {
    "train": ("explicit_assignment", "natural_alias"),
    "validation": ("sector_region", "polygon_region"),
    "test_iid": ("corridor_region", "different_track_duration"),
    "test_language": ("surplus_uav", "more_tasks_than_uavs"),
    "test_compositional": ("capability_mismatch", "similar_targets"),
    "test_conflict": (
        "duplicate_target_request",
        "overlapping_regions_auto_assignment",
    ),
    "test_reassignment": ("unavailable_uav", "failed_assignment_reassignment"),
}
FLEET_DATASET_SAMPLES_PER_SPLIT = 2
FLEET_DATASET_MANIFEST_SCHEMA_VERSION = 2
FLEET_DATASET_GENERATION_SOURCE = "deterministic_gold_template"
FLEET_DATASET_CONTRACT = "FleetMissionRequest/FleetMissionPlan/FleetPlanPatch-v1"
PLAN_OUTPUT_KIND = "fleet_mission_plan"
PATCH_OUTPUT_KIND = "fleet_plan_patch"
OUTPUT_KINDS = frozenset({PLAN_OUTPUT_KIND, PATCH_OUTPUT_KIND})
REGION_TYPES = frozenset(
    {"CIRCLE", "RECTANGLE", "SECTOR", "POLYGON", "CORRIDOR", "RELATIONAL"}
)
_T = TypeVar("_T")


class FleetDatasetValidationError(ValueError):
    """Raised for malformed Fleet Planner samples or datasets."""


@dataclass(frozen=True, slots=True)
class FleetDatasetValidationReport:
    valid: bool
    split_counts: Mapping[str, int]
    scenario_types: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "valid": self.valid,
            "split_counts": dict(self.split_counts),
            "scenario_types": list(self.scenario_types),
            "errors": list(self.errors),
        }


def _exact(value: Mapping[str, object], keys: set[str], context: str) -> None:
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise FleetDatasetValidationError(
            f"{context} keys invalid; missing={missing}, unknown={unknown}"
        )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FleetDatasetValidationError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FleetDatasetValidationError(f"{context} must be an array")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FleetDatasetValidationError(f"{context} must be non-empty text")
    return value.strip()


def _production_contract(
    operation: str,
    callback: Callable[[], _T],
) -> _T:
    try:
        return callback()
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise FleetDatasetValidationError(
            f"{operation} rejected by production Fleet contract: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def parse_input_request(input_value: object) -> FleetMissionRequest:
    """Parse one dataset input as an exact production FleetMissionRequest."""

    return _production_contract(
        "input",
        lambda: parse_fleet_mission_request(input_value),
    )


def validate_fleet_output(
    output: object,
    *,
    request: FleetMissionRequest,
) -> Mapping[str, object]:
    """Parse and request-bind a normal FleetMissionPlan gold/prediction."""

    value = _mapping(output, "output")
    _production_contract(
        "output",
        lambda: parse_fleet_mission_plan(value, request=request),
    )
    return value


def parse_fleet_plan_patch(
    output: object,
    *,
    request: FleetMissionRequest,
) -> FleetPlanPatch:
    """Parse a FleetPlanPatch using its production value objects.

    FleetPlanPatch predates a public JSON parser, so this tiny exact envelope
    parser constructs the production dataclass and then validates every
    replacement assignment through ``validate_fleet_mission_plan``.
    """

    value = _mapping(output, "fleet_plan_patch")
    _exact(
        value,
        {
            "schema_version",
            "fleet_mission_id",
            "base_fleet_plan_version",
            "new_fleet_plan_version",
            "replacement_assignments",
            "coordination_policy",
            "reason_codes",
        },
        "fleet_plan_patch",
    )
    assignments = _sequence(
        value["replacement_assignments"],
        "fleet_plan_patch.replacement_assignments",
    )
    reason_codes = _sequence(
        value["reason_codes"],
        "fleet_plan_patch.reason_codes",
    )
    patch = _production_contract(
        "fleet_plan_patch",
        lambda: FleetPlanPatch(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            fleet_mission_id=value["fleet_mission_id"],  # type: ignore[arg-type]
            base_fleet_plan_version=value["base_fleet_plan_version"],  # type: ignore[arg-type]
            new_fleet_plan_version=value["new_fleet_plan_version"],  # type: ignore[arg-type]
            replacement_assignments=tuple(
                parse_fleet_assignment(item) for item in assignments
            ),
            coordination_policy=parse_fleet_coordination_policy(
                value["coordination_policy"]
            ),
            reason_codes=tuple(reason_codes),  # type: ignore[arg-type]
        ),
    )
    if (
        patch.fleet_mission_id != request.fleet_mission_id
        or patch.base_fleet_plan_version != request.fleet_plan_version
        or patch.coordination_policy != request.coordination_policy
    ):
        raise FleetDatasetValidationError(
            "fleet_plan_patch mission/base version/policy must exactly echo input"
        )
    replacement_plan = FleetMissionPlan(
        fleet_mission_id=patch.fleet_mission_id,
        fleet_plan_version=patch.new_fleet_plan_version,
        assignments=patch.replacement_assignments,
        coordination_policy=patch.coordination_policy,
        # FleetPlanPatch has no unassigned_requirements field.  Preserve its
        # explicit reason-code coverage while reusing the production plan
        # validator so required targets omitted from replacement assignments do
        # not look like silent task loss.
        unassigned_requirements=patch.reason_codes,
    )
    _production_contract(
        "fleet_plan_patch.replacement_assignments",
        lambda: validate_fleet_mission_plan(
            replacement_plan,
            replace(
                request,
                fleet_plan_version=patch.new_fleet_plan_version,
            ),
        ),
    )
    return patch


def validate_fleet_plan_patch(
    output: object,
    *,
    request: FleetMissionRequest,
) -> Mapping[str, object]:
    value = _mapping(output, "fleet_plan_patch")
    parse_fleet_plan_patch(value, request=request)
    return value


def required_payloads_by_target(
    request: FleetMissionRequest,
) -> dict[str, tuple[str, ...]]:
    """Return the dataset-v1 capability convention carried by TargetSpec.

    ``hard_attributes`` is part of the production TargetSpec contract and the
    production Fleet validator binds that complete TargetSpec back to the
    trusted request.  Fleet Planner v1 does not yet define a generic capability
    expression language, so the dataset layer alone interprets the explicit
    ``required_payload:`` namespace against ``FleetUavCapability``.
    """

    result: dict[str, tuple[str, ...]] = {}
    prefix = "required_payload:"
    for target in request.target_requests:
        result[target.target_alias] = tuple(
            attribute[len(prefix) :]
            for attribute in target.target_spec.hard_attributes
            if attribute.startswith(prefix)
        )
    return result


def extract_explicit_target_aliases(
    entries: Sequence[object],
    target_aliases: Sequence[str] | frozenset[str],
) -> frozenset[str]:
    """Extract unique exact ``alias`` or ``alias: reason`` coverage entries."""

    known = frozenset(target_aliases)
    result: set[str] = set()
    for entry in entries:
        if not isinstance(entry, str):
            continue
        prefix, separator, reason = entry.partition(":")
        alias = prefix.strip()
        if alias not in known:
            continue
        if separator and not reason.strip():
            continue
        if not separator and entry.strip() != alias:
            continue
        result.add(alias)
    return frozenset(result)


def _validate_dataset_semantics(
    *,
    request: FleetMissionRequest,
    assignments: Sequence[object],
    task_count: int,
    coverage_notes: Sequence[object],
) -> None:
    inventory = {item.uav_id: item for item in request.uav_inventory}
    requirements = required_payloads_by_target(request)
    assigned_aliases: set[str] = set()
    for index, raw in enumerate(assignments):
        assignment = _production_contract(
            f"assignment[{index}]",
            lambda raw=raw: parse_fleet_assignment(raw),
        )
        uav = inventory[assignment.uav_id]
        alias = assignment.target_alias
        assigned_aliases.add(alias)
        missing = sorted(
            set(requirements.get(alias, ())) - set(uav.payload_capabilities)
        )
        if missing:
            raise FleetDatasetValidationError(
                f"UAV {uav.uav_id} lacks required payload(s) for {alias}: "
                + ", ".join(missing)
            )
    normalized_notes = tuple(_text(item, "coverage note") for item in coverage_notes)
    explicitly_unassigned = extract_explicit_target_aliases(
        normalized_notes,
        request.target_aliases,
    )
    if len(assigned_aliases | set(explicitly_unassigned)) < task_count:
        raise FleetDatasetValidationError(
            "output does not cover metadata.task_count through assignments and "
            "explicit target_alias-prefixed unassigned requirements"
        )


def validate_sample(sample: object) -> Mapping[str, object]:
    value = _mapping(sample, "sample")
    common_keys = {"sample_id", "input", "output_kind", "metadata"}
    output_kind = _text(value.get("output_kind"), "output_kind")
    if output_kind == PLAN_OUTPUT_KIND:
        _exact(value, common_keys | {"output"}, "sample")
    elif output_kind == PATCH_OUTPUT_KIND:
        _exact(value, common_keys | {"fleet_plan_patch"}, "sample")
    else:
        raise FleetDatasetValidationError(
            f"output_kind must be one of {sorted(OUTPUT_KINDS)}"
        )
    _text(value["sample_id"], "sample_id")
    request = parse_input_request(value["input"])

    metadata = _mapping(value["metadata"], "metadata")
    _exact(
        metadata,
        {
            "language",
            "task_count",
            "uav_count",
            "target_count",
            "difficulty",
            "scenario_type",
        },
        "metadata",
    )
    if metadata["language"] not in {"zh", "en"}:
        raise FleetDatasetValidationError("metadata.language must be zh or en")
    for field, expected in (
        ("task_count", len(request.target_requests)),
        ("uav_count", len(request.uav_inventory)),
        ("target_count", len(request.target_requests)),
    ):
        if metadata[field] != expected:
            raise FleetDatasetValidationError(
                f"metadata.{field} does not match input"
            )
    task_count = metadata["task_count"]
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count <= 0
    ):
        raise FleetDatasetValidationError("metadata.task_count must be positive")
    _text(metadata["difficulty"], "metadata.difficulty")
    _text(metadata["scenario_type"], "metadata.scenario_type")

    if output_kind == PLAN_OUTPUT_KIND:
        output = validate_fleet_output(value["output"], request=request)
        plan = parse_fleet_mission_plan(output, request=request)
        _validate_dataset_semantics(
            request=request,
            assignments=output["assignments"],  # type: ignore[arg-type]
            task_count=task_count,
            coverage_notes=plan.unassigned_requirements,
        )
    else:
        output = validate_fleet_plan_patch(
            value["fleet_plan_patch"],
            request=request,
        )
        patch_value = parse_fleet_plan_patch(output, request=request)
        _validate_dataset_semantics(
            request=request,
            assignments=output["replacement_assignments"],  # type: ignore[arg-type]
            task_count=task_count,
            coverage_notes=patch_value.reason_codes,
        )
    return value


def _load_line(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise FleetDatasetValidationError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        raw = json.loads(
            line,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                FleetDatasetValidationError(
                    f"non-standard constant {constant}"
                )
            ),
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, FleetDatasetValidationError) as exc:
        raise FleetDatasetValidationError(
            f"{path}:{line_number}: {exc}"
        ) from exc
    return validate_sample(raw)


def load_split(path: str | Path) -> tuple[Mapping[str, object], ...]:
    source = Path(path)
    if not source.is_file():
        raise FleetDatasetValidationError(f"missing split file: {source}")
    samples = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            raise FleetDatasetValidationError(
                f"{source}:{line_number}: blank line"
            )
        samples.append(
            _load_line(line, path=source, line_number=line_number)
        )
    if not samples:
        raise FleetDatasetValidationError(f"split is empty: {source}")
    return tuple(samples)


def _load_manifest(root: Path) -> Mapping[str, object]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FleetDatasetValidationError(f"missing dataset manifest: {path}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise FleetDatasetValidationError(
                    f"manifest contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                FleetDatasetValidationError(
                    f"manifest contains non-standard constant {constant}"
                )
            ),
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise FleetDatasetValidationError(f"invalid dataset manifest: {exc}") from exc
    manifest = _mapping(raw, "manifest")
    _exact(
        manifest,
        {
            "schema_version",
            "seed",
            "split_counts",
            "sha256",
            "generation_source",
            "dataset_contract",
        },
        "manifest",
    )
    if manifest["schema_version"] != FLEET_DATASET_MANIFEST_SCHEMA_VERSION or isinstance(
        manifest["schema_version"], bool
    ):
        raise FleetDatasetValidationError(
            "manifest.schema_version must equal integer "
            f"{FLEET_DATASET_MANIFEST_SCHEMA_VERSION}"
        )
    seed = manifest["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FleetDatasetValidationError(
            "manifest.seed must be a non-negative integer"
        )
    if manifest["generation_source"] != FLEET_DATASET_GENERATION_SOURCE:
        raise FleetDatasetValidationError(
            "manifest.generation_source does not identify the trusted generator"
        )
    if manifest["dataset_contract"] != FLEET_DATASET_CONTRACT:
        raise FleetDatasetValidationError(
            "manifest.dataset_contract does not match the production Fleet contract"
        )

    split_counts = _mapping(manifest["split_counts"], "manifest.split_counts")
    _exact(split_counts, set(FLEET_DATASET_SPLITS), "manifest.split_counts")
    for split in FLEET_DATASET_SPLITS:
        count = split_counts[split]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count != FLEET_DATASET_SAMPLES_PER_SPLIT
        ):
            raise FleetDatasetValidationError(
                f"manifest.split_counts.{split} must equal "
                f"{FLEET_DATASET_SAMPLES_PER_SPLIT}"
            )

    hashes = _mapping(manifest["sha256"], "manifest.sha256")
    expected_files = {f"{split}.jsonl" for split in FLEET_DATASET_SPLITS}
    _exact(hashes, expected_files, "manifest.sha256")
    hexadecimal = frozenset("0123456789abcdef")
    for filename in sorted(expected_files):
        digest = hashes[filename]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in hexadecimal for character in digest)
        ):
            raise FleetDatasetValidationError(
                f"manifest.sha256.{filename} must be a lowercase SHA-256 digest"
            )
    return manifest


def validate_dataset(dataset_root: str | Path) -> FleetDatasetValidationReport:
    root = Path(dataset_root).expanduser().resolve()
    counts: dict[str, int] = {}
    errors: list[str] = []
    seen_ids: set[str] = set()
    scenarios: set[str] = set()
    manifest: Mapping[str, object] | None = None
    try:
        manifest = _load_manifest(root)
    except (OSError, UnicodeError, FleetDatasetValidationError) as exc:
        errors.append(str(exc))
    for split in FLEET_DATASET_SPLITS:
        split_path = root / f"{split}.jsonl"
        try:
            samples = load_split(split_path)
            counts[split] = len(samples)
            if len(samples) != FLEET_DATASET_SAMPLES_PER_SPLIT:
                errors.append(
                    f"{split} must contain exactly "
                    f"{FLEET_DATASET_SAMPLES_PER_SPLIT} samples"
                )
            if manifest is not None:
                manifest_counts = _mapping(
                    manifest["split_counts"], "manifest.split_counts"
                )
                if manifest_counts[split] != len(samples):
                    errors.append(
                        f"manifest split count mismatch for {split}: "
                        f"expected {manifest_counts[split]}, got {len(samples)}"
                    )
                manifest_hashes = _mapping(manifest["sha256"], "manifest.sha256")
                actual_digest = sha256(split_path.read_bytes()).hexdigest()
                expected_digest = manifest_hashes[split_path.name]
                if actual_digest != expected_digest:
                    errors.append(
                        f"manifest SHA-256 mismatch for {split_path.name}: "
                        f"expected {expected_digest}, got {actual_digest}"
                    )
            for sample in samples:
                sample_id = str(sample["sample_id"])
                if sample_id in seen_ids:
                    errors.append(
                        f"sample_id {sample_id} appears in multiple splits"
                    )
                seen_ids.add(sample_id)
                metadata = _mapping(sample["metadata"], "metadata")
                scenario = str(metadata["scenario_type"])
                scenarios.add(scenario)
                output_kind = str(sample["output_kind"])
                expected_kind = (
                    PATCH_OUTPUT_KIND
                    if scenario == "failed_assignment_reassignment"
                    else PLAN_OUTPUT_KIND
                )
                if output_kind != expected_kind:
                    errors.append(
                        f"{sample_id} scenario {scenario} must use {expected_kind}"
                    )
                if split != "test_reassignment" and output_kind == PATCH_OUTPUT_KIND:
                    errors.append(
                        f"{sample_id} FleetPlanPatch leaked outside test_reassignment"
                    )
            if manifest is not None:
                expected_scenarios = list(FLEET_DATASET_SPLIT_SCENARIOS[split])
                split_seed = int(manifest["seed"]) + FLEET_DATASET_SPLITS.index(
                    split
                )
                random.Random(split_seed).shuffle(expected_scenarios)
                actual_scenarios = [
                    str(_mapping(sample["metadata"], "metadata")["scenario_type"])
                    for sample in samples
                ]
                if actual_scenarios != expected_scenarios:
                    errors.append(
                        f"manifest.seed does not reproduce {split} scenario order: "
                        f"expected {expected_scenarios}, got {actual_scenarios}"
                    )
        except (OSError, UnicodeError, FleetDatasetValidationError) as exc:
            counts[split] = 0
            errors.append(str(exc))
    missing_scenarios = sorted(FLEET_DATASET_SCENARIOS - scenarios)
    unexpected_scenarios = sorted(scenarios - FLEET_DATASET_SCENARIOS)
    if missing_scenarios:
        errors.append(
            "dataset is missing required scenario_type values: "
            + ", ".join(missing_scenarios)
        )
    if unexpected_scenarios:
        errors.append(
            "dataset contains unknown scenario_type values: "
            + ", ".join(unexpected_scenarios)
        )
    return FleetDatasetValidationReport(
        not errors,
        counts,
        tuple(sorted(scenarios)),
        tuple(errors),
    )


__all__ = [
    "FLEET_DATASET_CONTRACT",
    "FLEET_DATASET_GENERATION_SOURCE",
    "FLEET_DATASET_MANIFEST_SCHEMA_VERSION",
    "FLEET_DATASET_SAMPLES_PER_SPLIT",
    "FLEET_DATASET_SCENARIOS",
    "FLEET_DATASET_SPLIT_SCENARIOS",
    "FLEET_DATASET_SPLITS",
    "PATCH_OUTPUT_KIND",
    "PLAN_OUTPUT_KIND",
    "FleetDatasetValidationError",
    "FleetDatasetValidationReport",
    "extract_explicit_target_aliases",
    "load_split",
    "parse_fleet_plan_patch",
    "parse_input_request",
    "required_payloads_by_target",
    "validate_dataset",
    "validate_fleet_output",
    "validate_fleet_plan_patch",
    "validate_sample",
]
