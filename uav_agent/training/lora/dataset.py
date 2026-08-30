"""Validated text-only SFT data for the production Fleet Planner contract.

The adapter in this module deliberately owns no Fleet schema.  Every source
row is accepted by :mod:`fleet_data.validator` first, and the JSON placed in
the conversation is reconstructed through the production Fleet value objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from fleet.schemas import parse_fleet_mission_plan
from fleet_data.validator import (
    FLEET_DATASET_SPLITS,
    PATCH_OUTPUT_KIND,
    PLAN_OUTPUT_KIND,
    FleetDatasetValidationError,
    load_split,
    parse_fleet_plan_patch,
    parse_input_request,
    validate_dataset,
    validate_sample,
)


SYSTEM_PROMPT = (
    "你是 UAV Fleet Planner。"
    "根据 FleetMissionRequest 输出严格合法的 FleetMissionPlan 或 FleetPlanPatch。"
    "只输出 JSON，不输出 Markdown 或解释。"
)


class FleetPlannerSFTDatasetError(ValueError):
    """Raised when validated Fleet data cannot become an SFT conversation."""


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically without accepting NaN/Infinity."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def dataset_manifest_sha256(dataset_root: str | Path) -> str:
    """Return the digest of the already-validated dataset manifest file."""

    path = Path(dataset_root).expanduser().resolve() / "manifest.json"
    if not path.is_file():
        raise FleetPlannerSFTDatasetError(f"missing dataset manifest: {path}")
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FleetPlannerSFTDatasetError(
            f"could not read dataset manifest {path}: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class FleetPlannerSFTExample:
    """One validated Fleet request and its pure-JSON assistant target."""

    sample_id: str
    output_kind: str
    request: Mapping[str, object]
    target: Mapping[str, object]
    messages: tuple[Mapping[str, str], ...]

    @property
    def input_json(self) -> str:
        return self.messages[1]["content"]

    @property
    def assistant_json(self) -> str:
        return self.messages[2]["content"]

    def to_record(self) -> dict[str, object]:
        """Return a fresh mapping suitable for a Hugging Face Dataset/Trainer."""

        return {
            "sample_id": self.sample_id,
            "output_kind": self.output_kind,
            "request": dict(self.request),
            "target": dict(self.target),
            "input_json": self.input_json,
            "assistant_json": self.assistant_json,
            "messages": [dict(message) for message in self.messages],
        }


def sample_to_sft_example(sample: object) -> FleetPlannerSFTExample:
    """Convert one validator-approved row to a Qwen chat conversation.

    Calling :func:`validate_sample` here also protects callers that construct
    an example directly instead of going through :class:`FleetPlannerSFTDataset`.
    """

    validated = validate_sample(sample)
    request_value = parse_input_request(validated["input"])
    request = request_value.to_dict()
    output_kind = str(validated["output_kind"])
    if output_kind == PLAN_OUTPUT_KIND:
        target = parse_fleet_mission_plan(
            validated["output"],
            request=request_value,
        ).to_dict()
    elif output_kind == PATCH_OUTPUT_KIND:
        target = parse_fleet_plan_patch(
            validated["fleet_plan_patch"],
            request=request_value,
        ).to_dict()
    else:  # validate_sample already rejects this; keep the boundary explicit.
        raise FleetPlannerSFTDatasetError(
            f"unsupported Fleet output_kind: {output_kind!r}"
        )

    messages: tuple[Mapping[str, str], ...] = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(request)},
        {"role": "assistant", "content": canonical_json(target)},
    )
    return FleetPlannerSFTExample(
        sample_id=str(validated["sample_id"]),
        output_kind=output_kind,
        request=request,
        target=target,
        messages=messages,
    )


class FleetPlannerSFTDataset(Sequence[Mapping[str, object]]):
    """Validated split adapter for text-only Qwen Fleet Planner SFT.

    ``max_samples``/``limit`` are applied only *after* validation of the whole
    Fleet Planner dataset root.  This makes tiny smoke runs convenient without
    allowing an invalid held-out split or manifest to go unnoticed.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str = "train",
        max_samples: int | None = None,
        limit: int | None = None,
    ) -> None:
        if max_samples is not None and limit is not None:
            raise FleetPlannerSFTDatasetError(
                "provide only one of max_samples or limit"
            )
        selected_limit = max_samples if max_samples is not None else limit
        if selected_limit is not None and (
            isinstance(selected_limit, bool)
            or not isinstance(selected_limit, int)
            or selected_limit <= 0
        ):
            raise FleetPlannerSFTDatasetError(
                "max_samples/limit must be a positive integer or null"
            )
        if split not in FLEET_DATASET_SPLITS:
            raise FleetPlannerSFTDatasetError(
                f"split must be one of {list(FLEET_DATASET_SPLITS)}, got {split!r}"
            )

        root = Path(dataset_root).expanduser().resolve()
        if not root.is_dir():
            raise FleetPlannerSFTDatasetError(
                f"dataset_root must be a Fleet dataset directory: {root}"
            )
        report = validate_dataset(root)
        if not report.valid:
            raise FleetPlannerSFTDatasetError(
                "Fleet dataset validation failed: " + "; ".join(report.errors)
            )
        try:
            rows = load_split(root / f"{split}.jsonl")
            examples = tuple(sample_to_sft_example(row) for row in rows)
        except (FleetDatasetValidationError, KeyError, TypeError, ValueError) as exc:
            raise FleetPlannerSFTDatasetError(
                f"could not adapt Fleet split {split!r}: {exc}"
            ) from exc
        if selected_limit is not None:
            examples = examples[:selected_limit]

        self.dataset_root = root
        self.split = split
        self.manifest_sha256 = dataset_manifest_sha256(root)
        self._examples = examples

    @property
    def examples(self) -> tuple[FleetPlannerSFTExample, ...]:
        return self._examples

    @property
    def count(self) -> int:
        return len(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(example.to_record() for example in self._examples[index])
        return self._examples[index].to_record()


__all__ = [
    "SYSTEM_PROMPT",
    "FleetPlannerSFTDataset",
    "FleetPlannerSFTDatasetError",
    "FleetPlannerSFTExample",
    "canonical_json",
    "dataset_manifest_sha256",
    "sample_to_sft_example",
]
