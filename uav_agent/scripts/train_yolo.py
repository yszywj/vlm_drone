#!/usr/bin/env python3
"""Train or dry-run a local YOLO checkpoint without importing Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from training.yolo.config import (  # noqa: E402
    YoloTrainingConfigError,
    load_yolo_train_config,
)
from training.yolo.trainer import (  # noqa: E402
    UltralyticsTrainingBackend,
    YoloTrainingError,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a local Ultralytics model. Model downloads are never automatic, "
            "and this command has no Isaac Sim dependency."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PACKAGE_ROOT / "configs" / "yolo" / "train_yolo26s.yaml",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=_positive_int, default=None)
    parser.add_argument("--imgsz", type=_positive_int, default=None)
    parser.add_argument("--batch", type=_positive_int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="LAST.PT",
        help="resume only from this explicit last.pt checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check config, dataset, model, GPU, and output path without training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    overrides = {
        "base_model_path": args.model,
        "dataset_yaml": args.data,
        "device": args.device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "run_name": args.run_name,
        "resume": args.resume,
    }
    try:
        config = load_yolo_train_config(args.config, overrides=overrides)
        backend = UltralyticsTrainingBackend()
        if args.dry_run:
            preflight = backend.preflight(config)
            if preflight.dataset_report is not None:
                preflight.dataset_report.write_statistics()
            print(
                json.dumps(
                    {"dry_run": True, "config": config.to_dict(), **preflight.to_dict()},
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0 if preflight.ok else 1
        result = backend.train(config)
    except (OSError, RuntimeError, ValueError, YoloTrainingConfigError, YoloTrainingError) as exc:
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

