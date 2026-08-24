#!/usr/bin/env python3
"""Offline field-level evaluation wrapper for future Fleet Planner LoRA output."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from fleet_data.evaluator import evaluate_predictions  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(evaluate_predictions(args.gold, args.predictions), sort_keys=True, allow_nan=False))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"LoRA evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
