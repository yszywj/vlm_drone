"""Integrity-checked SFT input for the three production planning roles.

Unlike the legacy Fleet V1 adapter, this loader preserves the captured system
and user messages verbatim.  The generator already validates answers against
the production contracts; the loader checks every persisted split, its source
snapshot, and task ownership before exposing any training subset.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from training.lora.dataset import dataset_manifest_sha256


PLANNING_ROLES = ("mission_interpreter", "fleet_planner", "spatial_mission")
PLANNING_SPLITS = ("train", "validation", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PlanningRoleSFTDatasetError(ValueError):
    """Raised when captured role data cannot safely enter training."""


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PlanningRoleSFTDatasetError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PlanningRoleSFTDatasetError(f"non-finite JSON value: {value}")

    value = json.loads(text, object_pairs_hook=_object_pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise PlanningRoleSFTDatasetError("expected a JSON object")
    # Also reject overflow written as valid JSON exponent syntax.
    json.dumps(value, allow_nan=False)
    return value


def _read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield _json(line)
            except (ValueError, TypeError) as exc:
                raise PlanningRoleSFTDatasetError(f"{path}:{number}: {exc}") from exc


def _digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _check_messages(row: Mapping[str, object]) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise PlanningRoleSFTDatasetError("messages must be system, user, assistant")
    for message, role in zip(messages, ("system", "user", "assistant")):
        if (
            not isinstance(message, dict) or set(message) != {"role", "content"}
            or message["role"] != role or not isinstance(message["content"], str)
            or not message["content"]
        ):
            raise PlanningRoleSFTDatasetError("invalid captured role/content messages")
    _json(messages[1]["content"])
    _json(messages[2]["content"])


class PlanningRoleSFTDataset(Sequence[Mapping[str, object]]):
    """Load a task-disjoint role split without converting its production prompt.

    Integrity checks cover the entire root before ``max_samples`` is applied,
    including unused roles and held-out splits.  No tokenizer or model is loaded
    here; :class:`AssistantOnlyDataCollator` enforces the actual token limit.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        role: str,
        split: str = "train",
        max_samples: int | None = None,
    ) -> None:
        if role not in PLANNING_ROLES:
            raise PlanningRoleSFTDatasetError(f"unsupported planning role: {role!r}")
        if split not in PLANNING_SPLITS:
            raise PlanningRoleSFTDatasetError(f"unsupported planning split: {split!r}")
        if max_samples is not None and (
            isinstance(max_samples, bool) or not isinstance(max_samples, int)
            or max_samples <= 0
        ):
            raise PlanningRoleSFTDatasetError("max_samples must be a positive integer or null")
        root = Path(dataset_root).expanduser().resolve()
        try:
            manifest = _json((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("dataset_schema_version") != 1:
                raise PlanningRoleSFTDatasetError("unsupported planning dataset schema version")
            checks = manifest.get("validation", {})
            if not isinstance(checks, dict) or not all(
                checks.get(key) is True for key in (
                    "all_labels_passed_production_contracts",
                    "all_labels_passed_independent_blueprint_audits",
                )
            ):
                raise PlanningRoleSFTDatasetError("dataset lacks successful generation validation")
            expected_paths = {"tasks.jsonl"} | {
                f"{item_role}/{item_split}.jsonl"
                for item_role in PLANNING_ROLES for item_split in PLANNING_SPLITS
            }
            if set(manifest.get("data_sha256", {})) != expected_paths:
                raise PlanningRoleSFTDatasetError("manifest must cover all role splits and tasks")
            actual_paths = {str(path.relative_to(root)) for path in root.rglob("*.jsonl")}
            if actual_paths != expected_paths:
                raise PlanningRoleSFTDatasetError("missing or unexpected JSONL files")
            for relative in sorted(expected_paths):
                if _digest(root / relative) != manifest["data_sha256"][relative]:
                    raise PlanningRoleSFTDatasetError(f"dataset hash mismatch: {relative}")
            sources = manifest.get("source_sha256")
            if not isinstance(sources, dict) or not sources:
                raise PlanningRoleSFTDatasetError("missing generation source snapshot")
            for relative, expected in sources.items():
                path = (PROJECT_ROOT / relative).resolve()
                if not path.is_relative_to(PROJECT_ROOT) or _digest(path) != expected:
                    raise PlanningRoleSFTDatasetError(f"generation source changed: {relative}")

            tasks = list(_read_rows(root / "tasks.jsonl"))
            task_by_id = {task["task_id"]: task for task in tasks}
            if len(task_by_id) != len(tasks) or len(tasks) != manifest["underlying_tasks"]:
                raise PlanningRoleSFTDatasetError("duplicate task IDs or incorrect task count")
            for key in ("semantic_hash", "instruction_semantic_hash"):
                if len({task[key] for task in tasks}) != len(tasks):
                    raise PlanningRoleSFTDatasetError(f"duplicate underlying task {key}")
            if dict(Counter(task["split"] for task in tasks)) != manifest["task_split_counts"]:
                raise PlanningRoleSFTDatasetError("task split counts differ from manifest")
            selected: list[dict[str, object]] = []
            seen = set()
            role_counts = {item_role: Counter() for item_role in PLANNING_ROLES}
            projection_counts = Counter()
            for item_role in PLANNING_ROLES:
                for item_split in PLANNING_SPLITS:
                    for row in _read_rows(root / item_role / f"{item_split}.jsonl"):
                        task = task_by_id.get(row.get("task_id"))
                        if not task or row.get("sample_id") in seen:
                            raise PlanningRoleSFTDatasetError("unknown task or duplicate sample ID")
                        if row.get("role") != item_role or row.get("split") != item_split:
                            raise PlanningRoleSFTDatasetError("sample role/split differs from its file")
                        for key in ("split", "semantic_hash", "instruction_semantic_hash", "scale", "family"):
                            if row.get(key) != task[key]:
                                raise PlanningRoleSFTDatasetError(f"sample disagrees with task {key}")
                        expected_id = f"{task['task_id']}/{item_role}"
                        if item_role == "spatial_mission":
                            if row.get("uav_id") not in {uav["uav_id"] for uav in task["uavs"]}:
                                raise PlanningRoleSFTDatasetError("sample UAV is absent from task")
                            expected_id += f"/{row['uav_id']}"
                        elif row.get("uav_id") is not None:
                            raise PlanningRoleSFTDatasetError("global role sample cannot identify a local UAV")
                        if row["sample_id"] != expected_id:
                            raise PlanningRoleSFTDatasetError("sample ID does not identify its role projection")
                        _check_messages(row)
                        seen.add(row["sample_id"])
                        role_counts[item_role][item_split] += 1
                        projection_counts[task["task_id"], item_role] += 1
                        if item_role == role and item_split == split:
                            selected.append(row)
            if {key: dict(value) for key, value in role_counts.items()} != manifest["role_split_counts"]:
                raise PlanningRoleSFTDatasetError("role split counts differ from manifest")
            if len(seen) != manifest["total_samples"]:
                raise PlanningRoleSFTDatasetError("sample count differs from manifest")
            for task in tasks:
                for item_role in PLANNING_ROLES:
                    expected_count = len(task["uavs"]) if item_role == "spatial_mission" else 1
                    if projection_counts[task["task_id"], item_role] != expected_count:
                        raise PlanningRoleSFTDatasetError("missing or extra task role projections")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, PlanningRoleSFTDatasetError):
                raise
            raise PlanningRoleSFTDatasetError(f"planning dataset validation failed: {exc}") from exc

        self.dataset_root = root
        self.role = role
        self.split = split
        self.manifest = manifest
        self.manifest_sha256 = dataset_manifest_sha256(root)
        self._rows = tuple(selected[:max_samples] if max_samples is not None else selected)

    @property
    def count(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        row = deepcopy(self._rows[index])
        row["input_json"] = row["messages"][1]["content"]
        row["assistant_json"] = row["messages"][2]["content"]
        row["request"] = _json(row["input_json"])
        row["target"] = _json(row["assistant_json"])
        row["output_kind"] = self.role
        return row


__all__ = ["PLANNING_ROLES", "PLANNING_SPLITS", "PlanningRoleSFTDataset", "PlanningRoleSFTDatasetError"]
