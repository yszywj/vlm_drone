#!/usr/bin/env python3
"""Validate a local YOLO checkpoint and emit an export-gating report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from training.yolo.registry import write_validation_report  # noqa: E402
from training.yolo.trainer import (  # noqa: E402
    UltralyticsTrainingBackend,
    YoloTrainingError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a local YOLO .pt model; no model is downloaded."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    parser.add_argument("--model-family", choices=("yolo", "yoloe"), default="yolo")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="validation JSON used by export_yolo.py",
    )
    return parser


def _default_output(model_path: Path) -> Path:
    resolved = model_path.expanduser().resolve()
    if resolved.parent.name == "weights":
        return resolved.parent.parent / "validation_report.json"
    return resolved.with_suffix(".validation.json")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.imgsz <= 0:
        parser.error("--imgsz must be greater than zero")
    try:
        result = UltralyticsTrainingBackend().validate(
            model_path=args.model,
            dataset_yaml=args.data,
            device=args.device,
            imgsz=args.imgsz,
            task=args.task,
            model_family=args.model_family,
        )
        output = _default_output(args.model) if args.output is None else args.output
        report_path = write_validation_report(output, result.to_dict())
    except (OSError, RuntimeError, ValueError, YoloTrainingError) as exc:
        print(
            json.dumps(
                {"passed": False, "error": str(exc)},
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    payload = {**result.to_dict(), "validation_report": str(report_path)}
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
