"""Deterministic schema-v2 preview export built from trusted Planner-v1 Gold.

The public Planner-v1 corpus remains the source of semantic Gold.  This module
only projects freshly generated in-memory samples into routed, dynamic
``SkillPlanDraftV2`` Chat-SFT records and writes them to an explicitly selected
directory.  It never edits ``datasets/planner_v1`` and never calls a model.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from common.ids import validate_uav_id
from models.base import ChatMessage
from planner.policy import PlannerLimits, PlannerPolicy
from planner.prompt_builder import build_dynamic_skill_planner_messages
from planner.schemas import SkillPlanDraftV2, migrate_plan_v1_to_v2
from planner.skill_catalog import build_default_skill_catalog
from target.types import TargetSpec
from tasks.target_ontology import TargetOntology

from .dynamic_judge import build_gold_dynamic_draft
from .generator import DatasetGenerationError, GeneratedPlannerDataset
from .renderers import world_case_to_runtime_context
from .schemas import PLANNER_DATASET_SPLITS, PlannerDatasetSample


PLANNER_DATASET_V2_PREVIEW_SCHEMA = "planner_dataset_preview_v2"
DEFAULT_DYNAMIC_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "dynamic_skill_planner_system.txt"
)


def _instruction_from_sample(sample: PlannerDatasetSample) -> str:
    """Read the trusted renderer output from the existing v1 user message."""

    expected = sample.metadata.instruction
    content = sample.messages[1].content
    if not isinstance(content, str):
        raise DatasetGenerationError("Planner-v1 user message must be text")
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Some hand-authored v1 fixtures store the instruction directly.  The
        # immutable metadata remains the canonical rendered instruction.
        instruction = content.strip()
    else:
        if not isinstance(payload, Mapping):
            raise DatasetGenerationError(
                "Planner-v1 user message must be text or a JSON object"
            )
        instruction = payload.get("user_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise DatasetGenerationError(
            "Planner-v1 user message is missing user_instruction"
        )
    if instruction.strip() != expected:
        raise DatasetGenerationError(
            "Planner-v1 user message instruction does not match metadata"
        )
    return expected


def _mission_id(sample_id: str) -> str:
    digest = sha256(sample_id.encode("utf-8")).hexdigest()[:20]
    return f"mission_{digest}"


def _target_spec(ontology: TargetOntology, sample: PlannerDatasetSample) -> TargetSpec:
    concept = ontology.require_concept(sample.gold.target_concept_id)
    hard_attributes = tuple(
        f"{name}={value}" for name, value in sorted(concept.attributes.items())
    )
    return TargetSpec(
        original_description=sample.gold.target_description,
        category=concept.category,
        hard_attributes=hard_attributes,
        immutable_identity_summary=concept.canonical_description,
        query_ladder=(concept.canonical_description,),
        inspection_questions=(
            f"候选是否符合任务目标：{concept.canonical_description}",
        ),
    )
def build_v2_preview_record(
    sample: PlannerDatasetSample,
    *,
    dataset: GeneratedPlannerDataset,
    ontology: TargetOntology,
    uav_id: str,
    system_prompt: str,
) -> dict[str, object]:
    """Project one immutable v1 Gold sample into one routed v2 SFT record."""

    if not isinstance(sample, PlannerDatasetSample):
        raise TypeError("sample must be a PlannerDatasetSample")
    trusted_uav_id = validate_uav_id(uav_id)
    try:
        world = dataset.worlds[sample.world_context_id]
    except KeyError as exc:
        raise DatasetGenerationError(
            f"unknown sample world_context_id {sample.world_context_id!r}"
        ) from exc
    instruction = _instruction_from_sample(sample)
    mission_id = _mission_id(sample.sample_id)
    context = world_case_to_runtime_context(world)
    catalog = build_default_skill_catalog()
    limits = PlannerLimits()
    policy = PlannerPolicy()
    messages = build_dynamic_skill_planner_messages(
        instruction,
        context,
        catalog,
        limits,
        system_prompt,
        policy,
        mission_id=mission_id,
        uav_id=trusted_uav_id,
        plan_version=1,
    )
    routed = migrate_plan_v1_to_v2(
        build_gold_dynamic_draft(sample.gold),
        mission_id=mission_id,
        uav_id=trusted_uav_id,
        plan_version=1,
    )
    routed = replace(routed, target_spec=_target_spec(ontology, sample))
    # Reparse through the strict public schema before serializing a label.
    routed = SkillPlanDraftV2.from_dict(routed.to_dict())
    assistant = ChatMessage(
        role="assistant",
        content=json.dumps(
            routed.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return {
        "schema_version": 2,
        "dataset_schema": PLANNER_DATASET_V2_PREVIEW_SCHEMA,
        "sample_id": sample.sample_id,
        "split": sample.split,
        "language": sample.language,
        "world_context_id": sample.world_context_id,
        "mission_id": mission_id,
        "uav_id": trusted_uav_id,
        "plan_version": 1,
        "messages": [message.to_dict() for message in (*messages, assistant)],
        "gold_skill_plan": routed.to_dict(),
        "source_v1_gold_spec_id": sample.gold_spec_id,
        "metadata": sample.metadata.to_dict(),
    }


def write_v2_preview_dataset(
    dataset: GeneratedPlannerDataset,
    output_root: str | Path,
    *,
    uav_id: str,
    ontology: TargetOntology,
    overwrite: bool = False,
    system_prompt_path: str | Path = DEFAULT_DYNAMIC_SYSTEM_PROMPT_PATH,
) -> dict[str, object]:
    """Atomically write routed v2 preview files to a user-selected directory."""

    if not isinstance(dataset, GeneratedPlannerDataset):
        raise TypeError("dataset must be a GeneratedPlannerDataset")
    if not isinstance(ontology, TargetOntology):
        raise TypeError("ontology must be a TargetOntology")
    trusted_uav_id = validate_uav_id(uav_id)
    try:
        system_prompt = Path(system_prompt_path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise DatasetGenerationError("could not read dynamic Planner prompt") from exc
    if not system_prompt:
        raise DatasetGenerationError("dynamic Planner prompt must be non-empty")

    destination = Path(output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"dataset already exists: {destination}")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".v2tmp",
            dir=destination.parent,
        )
    )
    backup_container: Path | None = None
    backup: Path | None = None
    try:
        hashes: dict[str, str] = {}
        split_counts: dict[str, int] = {}
        for split in PLANNER_DATASET_SPLITS:
            rows = dataset.samples_by_split[split]
            split_counts[split] = len(rows)
            path = stage / f"{split}.jsonl"
            digest = sha256()
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for sample in rows:
                    record = build_v2_preview_record(
                        sample,
                        dataset=dataset,
                        ontology=ontology,
                        uav_id=trusted_uav_id,
                        system_prompt=system_prompt,
                    )
                    line = json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ) + "\n"
                    stream.write(line)
                    digest.update(line.encode("utf-8"))
            hashes[path.name] = digest.hexdigest()
        manifest: dict[str, object] = {
            "dataset_schema": PLANNER_DATASET_V2_PREVIEW_SCHEMA,
            "schema_version": 2,
            "source_dataset": "planner_v1",
            "profile": dataset.profile.name,
            "seed": dataset.seed,
            "uav_id": trusted_uav_id,
            "plan_version": 1,
            "split_counts": split_counts,
            "split_sha256": hashes,
        }
        (stage / "dataset_manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        (stage / "checksums.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
            encoding="utf-8",
        )
        if destination.exists():
            backup_container = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.",
                    suffix=".v2backup",
                    dir=destination.parent,
                )
            )
            backup = backup_container / "previous"
            os.replace(destination, backup)
        os.replace(stage, destination)
        if backup_container is not None:
            shutil.rmtree(backup_container)
            backup_container = None
            backup = None
        return manifest
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup_container is not None and backup_container.exists():
            if backup is None or not backup.exists():
                shutil.rmtree(backup_container)


__all__ = [
    "DEFAULT_DYNAMIC_SYSTEM_PROMPT_PATH",
    "PLANNER_DATASET_V2_PREVIEW_SCHEMA",
    "build_v2_preview_record",
    "write_v2_preview_dataset",
]
