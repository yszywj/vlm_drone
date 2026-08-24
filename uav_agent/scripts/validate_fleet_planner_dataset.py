#!/usr/bin/env python3
"""Strictly validate all Fleet Planner v1 splits."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from fleet_data.validator import validate_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        "--dataset-dir",
        dest="dataset_root",
        type=Path,
        default=_ROOT / "datasets/fleet_planner_v1",
        help="dataset root (both option names are equivalent)",
    )
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset_root)
    print(json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
