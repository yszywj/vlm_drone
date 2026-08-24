#!/usr/bin/env python3
"""Inspect actual local Qwen modules; never infer or write target_modules."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def _category(name: str) -> str:
    lowered = name.lower()
    # Multimodal connector names commonly live below a ``visual`` namespace
    # (for example ``visual.merger``).  Classify the specific bridge before
    # the broad tower namespace so the report does not hide connectors.
    if any(token in lowered for token in ("connector", "merger", "projector")):
        return "connector"
    if any(token in lowered for token in ("visual", "vision", "vit")):
        return "vision_tower"
    if any(token in lowered for token in ("self_attn", "attention", "attn")):
        return "language_attention_projection"
    if any(token in lowered for token in ("mlp", "feed_forward")):
        return "language_mlp_projection"
    return "other_linear"


def inspect(model_path: str | Path) -> dict[str, object]:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"local model directory does not exist: {path}")
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required in the separate qwen_lora environment"
        ) from exc
    model = AutoModel.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    categories: dict[str, list[str]] = {
        "language_attention_projection": [],
        "language_mlp_projection": [],
        "vision_tower": [],
        "connector": [],
        "other_linear": [],
    }
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Linear":
            categories[_category(name)].append(name)
    return {
        "model_path": str(path),
        "actual_linear_modules": categories,
        "fleet_planner_language_candidates": sorted(
            categories["language_attention_projection"]
            + categories["language_mlp_projection"]
        ),
        "config_updated": False,
        "note": "review actual names before selecting target_modules",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = inspect(args.model)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
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
