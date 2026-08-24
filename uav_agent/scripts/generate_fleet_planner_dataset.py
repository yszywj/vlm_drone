#!/usr/bin/env python3
"""Generate the deterministic Fleet Planner v1 pilot dataset."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from fleet_data.generator import generate_fleet_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=_ROOT / "datasets/fleet_planner_v1",
        help="dataset destination (both option names are equivalent)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = generate_fleet_dataset(args.output, seed=args.seed, overwrite=args.overwrite)
    except (OSError, TypeError, ValueError) as exc:
        print(f"fleet dataset generation error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
