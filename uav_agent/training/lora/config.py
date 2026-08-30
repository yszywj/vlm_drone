"""Strict, placeholder-safe configuration for Fleet Planner LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Mapping

from fleet_data.validator import FLEET_DATASET_SPLITS


class LoraScaffoldError(ValueError):
    """Raised when a LoRA configuration is unsafe or internally inconsistent."""


_COMMON_KEYS = {
    "schema_version",
    "status",
    "base_model_path",
    "output_dir",
    "dataset_dir",
    "target_modules",
    "rank",
    "lora_alpha",
    "lora_dropout",
}

_ACTIVE_ONLY_KEYS = {
    "adapter_output_dir",
    "seed",
    "model_max_length",
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "lr_scheduler_type",
    "max_grad_norm",
    "bf16",
    "gradient_checkpointing",
    "logging_steps",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "dataloader_num_workers",
    "train_split",
    "validation_split",
    "resume_from_checkpoint",
    "max_train_samples",
    "max_validation_samples",
    "notes",
}

_ALL_KEYS = _COMMON_KEYS | _ACTIVE_ONLY_KEYS
_PLACEHOLDER_METADATA_KEYS = {"adapter_output_dir", "notes"}
_TARGET_PATTERN = re.compile(
    r"^model\.language_model\.[A-Za-z0-9_]+(?:\.(?:[A-Za-z0-9_]+|\*))*$"
)
_SCHEDULERS = frozenset({"linear", "cosine"})


@dataclass(frozen=True, slots=True)
class LoraScaffoldConfig:
    """Validated training configuration.

    All training fields are ``None`` in a legacy placeholder configuration.
    This deliberately makes it impossible for code to reinterpret a placeholder
    as a partially configured training run.
    """

    config_path: Path
    schema_version: int
    status: str
    base_model_path: Path
    output_dir: Path
    dataset_dir: Path
    target_modules: tuple[str, ...] | None
    rank: int | None
    lora_alpha: float | None
    lora_dropout: float | None
    adapter_output_dir: Path | None = None
    seed: int | None = None
    model_max_length: int | None = None
    num_train_epochs: float | None = None
    max_steps: int | None = None
    per_device_train_batch_size: int | None = None
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    warmup_ratio: float | None = None
    lr_scheduler_type: str | None = None
    max_grad_norm: float | None = None
    bf16: bool | None = None
    gradient_checkpointing: bool | None = None
    logging_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    save_total_limit: int | None = None
    dataloader_num_workers: int | None = None
    train_split: str | None = None
    validation_split: str | None = None
    resume_from_checkpoint: Path | None = None
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    notes: str | None = None

    @property
    def is_placeholder(self) -> bool:
        return self.status == "placeholder"

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def require_active(self) -> "LoraScaffoldConfig":
        """Return this config only when real training was explicitly activated."""

        if not self.is_active:
            raise LoraScaffoldError(
                "placeholder configuration cannot start training or create weights"
            )
        return self


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LoraScaffoldError(f"{field} must be non-empty text")
    return value.strip()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LoraScaffoldError(f"{field} must be a positive integer")
    return value


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field)


def _number(value: object, field: str, *, minimum: float, inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoraScaffoldError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LoraScaffoldError(f"{field} must be finite")
    valid = result >= minimum if inclusive else result > minimum
    if not valid:
        operator = ">=" if inclusive else ">"
        raise LoraScaffoldError(f"{field} must be {operator} {minimum}")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise LoraScaffoldError(f"{field} must be a boolean")
    return value


def _resolve_path(value: object, field: str, source: Path) -> Path:
    raw = Path(_text(value, field)).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    if raw.parts and raw.parts[0] == "datasets":
        # Committed datasets are relative to the uav_agent project root.
        return (Path(__file__).resolve().parents[2] / raw).resolve()
    return (source.parent / raw).resolve()


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LoraScaffoldError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise LoraScaffoldError(f"non-finite JSON constant is forbidden: {value}")


def _target_modules(value: object, *, placeholder: bool) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise LoraScaffoldError("target_modules must be null or a non-empty list")
    result = tuple(_text(item, "target_modules item") for item in value)
    if len(result) != len(set(result)):
        raise LoraScaffoldError("target_modules contains duplicates")
    if placeholder:
        raise LoraScaffoldError(
            "placeholder config must keep target_modules/rank/alpha/dropout null; "
            "do not guess training parameters"
        )
    for pattern in result:
        if not _TARGET_PATTERN.fullmatch(pattern) or ".." in pattern or "**" in pattern:
            raise LoraScaffoldError(
                "target_modules entries must be fully qualified language-only names or "
                "anchored globs beginning with 'model.language_model.'; unsafe entry: "
                f"{pattern!r}"
            )
    return result


def _placeholder_config(
    raw: Mapping[str, object], source: Path, base_model: Path, output: Path, dataset: Path
) -> LoraScaffoldConfig:
    missing = _COMMON_KEYS - set(raw)
    unknown = set(raw) - _ALL_KEYS
    if missing or unknown:
        raise LoraScaffoldError(
            f"LoRA placeholder config keys invalid; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    forbidden_values = {
        key: raw[key]
        for key in (_ACTIVE_ONLY_KEYS - _PLACEHOLDER_METADATA_KEYS)
        if key in raw and raw[key] is not None
    }
    if forbidden_values:
        raise LoraScaffoldError(
            "placeholder config must not define active training fields: "
            + ", ".join(sorted(forbidden_values))
        )
    target_modules = _target_modules(raw["target_modules"], placeholder=True)
    rank = raw["rank"]
    alpha = raw["lora_alpha"]
    dropout = raw["lora_dropout"]
    if any(value is not None for value in (rank, alpha, dropout)):
        raise LoraScaffoldError(
            "placeholder config must keep target_modules/rank/alpha/dropout null; "
            "do not guess training parameters"
        )
    adapter_output = (
        _resolve_path(raw["adapter_output_dir"], "adapter_output_dir", source)
        if raw.get("adapter_output_dir") is not None
        else None
    )
    notes = _text(raw["notes"], "notes") if raw.get("notes") is not None else None
    return LoraScaffoldConfig(
        config_path=source,
        schema_version=1,
        status="placeholder",
        base_model_path=base_model,
        output_dir=output,
        dataset_dir=dataset,
        target_modules=target_modules,
        rank=None,
        lora_alpha=None,
        lora_dropout=None,
        adapter_output_dir=adapter_output,
        notes=notes,
    )


def _active_config(
    raw: Mapping[str, object], source: Path, base_model: Path, output: Path, dataset: Path
) -> LoraScaffoldConfig:
    missing = _ALL_KEYS - set(raw)
    unknown = set(raw) - _ALL_KEYS
    if missing or unknown:
        raise LoraScaffoldError(
            f"LoRA active config keys invalid; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )

    target_modules = _target_modules(raw["target_modules"], placeholder=False)
    assert target_modules is not None
    rank = _positive_int(raw["rank"], "rank")
    alpha = _number(raw["lora_alpha"], "lora_alpha", minimum=0.0, inclusive=False)
    dropout = _number(
        raw["lora_dropout"], "lora_dropout", minimum=0.0, inclusive=True
    )
    if dropout >= 1.0:
        raise LoraScaffoldError("lora_dropout must be in [0, 1)")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LoraScaffoldError("seed must be a non-negative integer")
    epochs = _number(
        raw["num_train_epochs"], "num_train_epochs", minimum=0.0, inclusive=False
    )
    weight_decay = _number(
        raw["weight_decay"], "weight_decay", minimum=0.0, inclusive=True
    )
    warmup_ratio = _number(
        raw["warmup_ratio"], "warmup_ratio", minimum=0.0, inclusive=True
    )
    if warmup_ratio >= 1.0:
        raise LoraScaffoldError("warmup_ratio must be in [0, 1)")
    scheduler = _text(raw["lr_scheduler_type"], "lr_scheduler_type")
    if scheduler not in _SCHEDULERS:
        raise LoraScaffoldError(
            f"lr_scheduler_type must be one of {sorted(_SCHEDULERS)}"
        )
    workers = raw["dataloader_num_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise LoraScaffoldError("dataloader_num_workers must be a non-negative integer")
    train_split = _text(raw["train_split"], "train_split")
    validation_split = _text(raw["validation_split"], "validation_split")
    for field, split in (
        ("train_split", train_split),
        ("validation_split", validation_split),
    ):
        if split not in FLEET_DATASET_SPLITS:
            raise LoraScaffoldError(
                f"{field} must be one of {list(FLEET_DATASET_SPLITS)}, got {split!r}"
            )
    if train_split == validation_split:
        raise LoraScaffoldError("train_split and validation_split must differ")
    resume = (
        _resolve_path(
            raw["resume_from_checkpoint"], "resume_from_checkpoint", source
        )
        if raw["resume_from_checkpoint"] is not None
        else None
    )
    adapter_output = _resolve_path(
        raw["adapter_output_dir"], "adapter_output_dir", source
    )
    for field, candidate in (
        ("output_dir", output),
        ("adapter_output_dir", adapter_output),
    ):
        if _paths_overlap(candidate, base_model):
            raise LoraScaffoldError(
                f"{field} must not equal, contain, or be inside base_model_path"
            )
        if _paths_overlap(candidate, dataset):
            raise LoraScaffoldError(
                f"{field} must not equal, contain, or be inside dataset_dir"
            )
    if _paths_overlap(output, adapter_output):
        raise LoraScaffoldError(
            "output_dir and adapter_output_dir must be separate, non-nested paths"
        )

    return LoraScaffoldConfig(
        config_path=source,
        schema_version=1,
        status="active",
        base_model_path=base_model,
        output_dir=output,
        dataset_dir=dataset,
        target_modules=target_modules,
        rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        adapter_output_dir=adapter_output,
        seed=seed,
        model_max_length=_positive_int(raw["model_max_length"], "model_max_length"),
        num_train_epochs=epochs,
        max_steps=_optional_positive_int(raw["max_steps"], "max_steps"),
        per_device_train_batch_size=_positive_int(
            raw["per_device_train_batch_size"], "per_device_train_batch_size"
        ),
        per_device_eval_batch_size=_positive_int(
            raw["per_device_eval_batch_size"], "per_device_eval_batch_size"
        ),
        gradient_accumulation_steps=_positive_int(
            raw["gradient_accumulation_steps"], "gradient_accumulation_steps"
        ),
        learning_rate=_number(
            raw["learning_rate"], "learning_rate", minimum=0.0, inclusive=False
        ),
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=scheduler,
        max_grad_norm=_number(
            raw["max_grad_norm"], "max_grad_norm", minimum=0.0, inclusive=False
        ),
        bf16=_boolean(raw["bf16"], "bf16"),
        gradient_checkpointing=_boolean(
            raw["gradient_checkpointing"], "gradient_checkpointing"
        ),
        logging_steps=_positive_int(raw["logging_steps"], "logging_steps"),
        eval_steps=_positive_int(raw["eval_steps"], "eval_steps"),
        save_steps=_positive_int(raw["save_steps"], "save_steps"),
        save_total_limit=_positive_int(
            raw["save_total_limit"], "save_total_limit"
        ),
        dataloader_num_workers=workers,
        train_split=train_split,
        validation_split=validation_split,
        resume_from_checkpoint=resume,
        max_train_samples=_optional_positive_int(
            raw["max_train_samples"], "max_train_samples"
        ),
        max_validation_samples=_optional_positive_int(
            raw["max_validation_samples"], "max_validation_samples"
        ),
        notes=_text(raw["notes"], "notes"),
    )


def load_lora_config(path: str | Path) -> LoraScaffoldConfig:
    """Load a strict config without importing model or training dependencies."""

    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LoraScaffoldError(f"could not load LoRA config {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise LoraScaffoldError("LoRA config root must be an object")
    if raw.get("schema_version") != 1 or isinstance(raw.get("schema_version"), bool):
        raise LoraScaffoldError("LoRA config schema_version must be 1")
    status = _text(raw.get("status"), "status")
    if status not in {"placeholder", "active"}:
        raise LoraScaffoldError("status must be placeholder or active")

    # These paths are inert data in placeholder mode; resolving them does not
    # create directories or touch model weights.
    base_model = _resolve_path(raw.get("base_model_path"), "base_model_path", source)
    output = _resolve_path(raw.get("output_dir"), "output_dir", source)
    dataset = _resolve_path(raw.get("dataset_dir"), "dataset_dir", source)
    if status == "placeholder":
        return _placeholder_config(raw, source, base_model, output, dataset)
    return _active_config(raw, source, base_model, output, dataset)


__all__ = ["LoraScaffoldConfig", "LoraScaffoldError", "load_lora_config"]
