#!/usr/bin/env python3
"""Validate the placeholder Fleet Planner LoRA contract; never train weights."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from fleet_data.validator import validate_dataset  # noqa: E402
from training.lora.config import LoraScaffoldError, load_lora_config  # noqa: E402


DEFAULT_CONFIG = _ROOT / "configs/lora/fleet_planner_lora.json"


def validate_placeholder(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = load_lora_config(config_path)
    if config.status != "placeholder":
        raise LoraScaffoldError(
            "training is intentionally not implemented; inspect real target modules and "
            "complete a reviewed training configuration before activation"
        )
    report = validate_dataset(config.dataset_dir)
    if not report.valid:
        raise LoraScaffoldError("Fleet Planner dataset is invalid: " + "; ".join(report.errors))
    return {
        "status": "placeholder",
        "training_started": False,
        "weights_created": False,
        "base_model_path": str(config.base_model_path),
        "output_dir": str(config.output_dir),
        "dataset": report.to_dict(),
        "target_modules": None,
        "note": "configuration and data validated; no model was loaded and no training ran",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate_placeholder(args.config), ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (LoraScaffoldError, OSError, TypeError, ValueError) as exc:
        print(f"LoRA scaffold validation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
