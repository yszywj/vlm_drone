#!/usr/bin/env python3
"""Validate Planner v1 JSONL, Gold, prompts, splits and checksums."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planner_data.generator import DEFAULT_DATASET_CONFIG_PATH  # noqa: E402
from planner_data.validator import PlannerDatasetValidator  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly validate a generated Planner v1 dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATASET_CONFIG_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = PlannerDatasetValidator(config_path=arguments.config).validate(
            arguments.dataset_root
        )
    except Exception as exc:
        print(f"validation error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
