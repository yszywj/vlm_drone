#!/usr/bin/env python3
"""Inspect actual local Qwen3-VL Linear modules without guessing LoRA suffixes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.lora.modeling import (  # noqa: E402
    CONNECTOR,
    LANGUAGE_ATTENTION,
    LANGUAGE_MLP,
    VISION,
    classify_module_name,
    inspect_model_linear_modules,
    validate_local_model_directory,
)


def _category(name: str) -> str:
    """Backward-compatible category labels used by the original scaffold tests."""

    category = {
        CONNECTOR: "connector",
        VISION: "vision_tower",
        LANGUAGE_ATTENTION: "language_attention_projection",
        LANGUAGE_MLP: "language_mlp_projection",
    }.get(classify_module_name(name))
    if category is not None:
        return category
    # The pre-Qwen3-VL scaffold test used a text-only ``model.layers`` example.
    # Keep its display classification compatible without admitting that shorter
    # namespace into the LoRA target validator.
    parts = {part.lower() for part in name.split(".")}
    if parts & {"self_attn", "attention", "attn"}:
        return "language_attention_projection"
    if parts & {"mlp", "feed_forward"}:
        return "language_mlp_projection"
    return "other_linear"


def inspect_loaded_model(
    model: object,
    *,
    model_path: str | Path,
    language_only: bool = False,
) -> dict[str, object]:
    """Build the stable JSON report from an already loaded model."""

    inventory = inspect_model_linear_modules(model)
    payload: dict[str, object] = {
        "schema_version": 1,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "selection_mode": "language_only" if language_only else "inventory",
        "language_attention_modules": list(inventory.language_attention_modules),
        "language_mlp_modules": list(inventory.language_mlp_modules),
        "language_candidates": list(inventory.language_candidates),
        "fleet_planner_language_candidates": list(inventory.language_candidates),
        "vision_modules": list(inventory.vision_modules),
        "connector_modules": list(inventory.connector_modules),
        "other_linear_modules": list(inventory.other_linear_modules),
        "counts": {
            "language_attention": len(inventory.language_attention_modules),
            "language_mlp": len(inventory.language_mlp_modules),
            "language_candidates": len(inventory.language_candidates),
            "vision": len(inventory.vision_modules),
            "connector": len(inventory.connector_modules),
            "other_linear": len(inventory.other_linear_modules),
        },
        "config_updated": False,
        "note": (
            "Copy only reviewed, fully qualified language module names or anchored "
            "model.language_model.* globs into target_modules; runtime validation is "
            "still mandatory before PEFT injection."
        ),
    }
    if language_only:
        payload["selected_modules"] = list(inventory.language_candidates)
    else:
        payload["selected_modules"] = []
    return payload


def _load_local_model(model_path: Path) -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    validation = validate_local_model_directory(model_path)
    if not validation["complete"]:
        raise ValueError(
            "local Qwen3-VL model directory is incomplete: "
            + "; ".join(str(item) for item in validation["errors"])
        )
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "the separate qwen_lora environment needs a Transformers build with "
            "Qwen3VLForConditionalGeneration"
        ) from exc
    try:
        return Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
    except Exception as exc:  # third-party loader exception surface is not stable
        raise RuntimeError(f"could not inspect local Qwen3-VL model: {exc}") from exc


def inspect(
    model_path: str | Path, *, language_only: bool = False
) -> dict[str, object]:
    """Load the local model and return its real Linear-module inventory."""

    path = Path(model_path).expanduser().resolve()
    model = _load_local_model(path)
    return inspect_loaded_model(model, model_path=path, language_only=language_only)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--language-only",
        action="store_true",
        help="mark only reviewed language attention/MLP modules as selected candidates",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = inspect(args.model, language_only=args.language_only)
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"LoRA target inspection error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
