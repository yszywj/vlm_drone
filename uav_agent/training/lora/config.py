"""Strict configuration for the non-training Fleet Planner LoRA scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


class LoraScaffoldError(ValueError):
    """Raised when placeholder configuration could imply fake training."""


@dataclass(frozen=True, slots=True)
class LoraScaffoldConfig:
    config_path: Path
    status: str
    base_model_path: Path
    output_dir: Path
    dataset_dir: Path
    target_modules: tuple[str, ...] | None
    rank: int | None
    lora_alpha: float | None
    lora_dropout: float | None


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LoraScaffoldError(f"{field} must be non-empty text")
    return value.strip()


def load_lora_config(path: str | Path) -> LoraScaffoldConfig:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LoraScaffoldError(f"could not load LoRA config {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise LoraScaffoldError("LoRA config root must be an object")
    expected = {
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
    if set(raw) != expected:
        raise LoraScaffoldError(
            f"LoRA config keys invalid; missing={sorted(expected - set(raw))}, "
            f"unknown={sorted(set(raw) - expected)}"
        )
    if raw["schema_version"] != 1:
        raise LoraScaffoldError("LoRA config schema_version must be 1")
    status = _text(raw["status"], "status")
    if status not in {"placeholder", "active"}:
        raise LoraScaffoldError("status must be placeholder or active")
    base_model = Path(_text(raw["base_model_path"], "base_model_path")).expanduser()
    output = Path(_text(raw["output_dir"], "output_dir")).expanduser()
    dataset_raw = Path(_text(raw["dataset_dir"], "dataset_dir")).expanduser()
    if dataset_raw.is_absolute():
        dataset = dataset_raw.resolve()
    elif dataset_raw.parts and dataset_raw.parts[0] == "datasets":
        # The committed reproducibility pilot lives inside uav_agent/datasets.
        dataset = (Path(__file__).resolve().parents[2] / dataset_raw).resolve()
    else:
        dataset = (source.parent / dataset_raw).resolve()
    target_modules_raw = raw["target_modules"]
    target_modules: tuple[str, ...] | None
    if target_modules_raw is None:
        target_modules = None
    elif isinstance(target_modules_raw, list) and target_modules_raw:
        target_modules = tuple(_text(item, "target_modules item") for item in target_modules_raw)
        if len(target_modules) != len(set(target_modules)):
            raise LoraScaffoldError("target_modules contains duplicates")
    else:
        raise LoraScaffoldError("target_modules must be null or a non-empty list")
    rank = raw["rank"]
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0):
        raise LoraScaffoldError("rank must be null or a positive integer")
    alpha = raw["lora_alpha"]
    if alpha is not None and (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha <= 0):
        raise LoraScaffoldError("lora_alpha must be null or positive")
    dropout = raw["lora_dropout"]
    if dropout is not None and (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0 <= float(dropout) < 1
    ):
        raise LoraScaffoldError("lora_dropout must be null or in [0, 1)")
    if status == "placeholder" and any(
        value is not None for value in (target_modules, rank, alpha, dropout)
    ):
        raise LoraScaffoldError(
            "placeholder config must keep target_modules/rank/alpha/dropout null; "
            "do not guess training parameters"
        )
    return LoraScaffoldConfig(
        source,
        status,
        base_model.resolve(),
        output.resolve(),
        dataset,
        target_modules,
        rank,
        None if alpha is None else float(alpha),
        None if dropout is None else float(dropout),
    )


__all__ = ["LoraScaffoldConfig", "LoraScaffoldError", "load_lora_config"]
