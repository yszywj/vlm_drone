#!/usr/bin/env python3
"""Export a validated YOLO checkpoint to ONNX or TensorRT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from training.yolo.trainer import (  # noqa: E402
    UltralyticsTrainingBackend,
    YoloTrainingError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export only a model whose SHA256 matches a passing validate_yolo.py report. "
            "Development runtime should continue using .pt by default."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "tensorrt", "engine"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--model-family", choices=("yolo", "yoloe"), default="yolo")
    parser.add_argument(
        "--freeze-yoloe-prompts",
        action="store_true",
        help="acknowledge that a YOLOE export no longer supports dynamic set_classes()",
    )
    parser.add_argument("--half", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.imgsz <= 0:
        parser.error("--imgsz must be greater than zero")
    try:
        result = UltralyticsTrainingBackend().export(
            model_path=args.model,
            validation_report=args.validation_report,
            format=args.format,
            device=args.device,
            imgsz=args.imgsz,
            model_family=args.model_family,
            freeze_yoloe_prompts=args.freeze_yoloe_prompts,
            half=args.half,
        )
    except (OSError, RuntimeError, ValueError, YoloTrainingError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {"ok": True, **result.to_dict()},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

