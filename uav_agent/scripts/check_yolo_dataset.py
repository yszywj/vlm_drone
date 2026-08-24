#!/usr/bin/env python3
"""Validate a YOLO dataset and write statistics.json without changing labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from training.yolo.dataset import (  # noqa: E402
    YoloDatasetError,
    YoloDatasetValidator,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation for an Ultralytics-format YOLO dataset."
    )
    parser.add_argument("--data", type=Path, required=True, help="dataset data.yaml")
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="statistics path; defaults to <dataset-root>/statistics.json",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="also return non-zero for valid background labels and other warnings",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = YoloDatasetValidator(task=args.task).validate(args.data)
        statistics_path = report.write_statistics(args.output)
    except (OSError, RuntimeError, YoloDatasetError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    payload = report.to_statistics_dict()
    payload["statistics_path"] = str(statistics_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report.ok and not (args.fail_on_warning and report.warnings) else 1


if __name__ == "__main__":
    raise SystemExit(main())

