#!/usr/bin/env python3
"""Train the lightweight temporal ray-depth residual network (no Isaac import)."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from training.target_state.config import TrainingStage, load_training_config  # noqa: E402
from training.target_state.data import TargetStateTorchDataset  # noqa: E402
from training.target_state.trainer import (  # noqa: E402
    TargetStateTrainingError,
    train_target_state,
    validate_initial_checkpoint,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PACKAGE_ROOT / "configs" / "target_state" / "train_yolo_deployment.yaml",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="explicit stage-A best.pt used to initialize stage-B fine-tuning",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--stage", choices=[item.value for item in TrainingStage])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config/data and materialize one sample without training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_training_config(args.config)
        overrides = {
            "dataset_root": args.dataset_root,
            "output_dir": args.output_dir,
            "initial_checkpoint_path": args.initial_checkpoint,
            "run_name": args.run_name,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "stage": TrainingStage(args.stage) if args.stage else None,
        }
        config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
        if args.dry_run:
            validate_initial_checkpoint(config, map_location="cpu")
            summaries = {}
            for split in ("train", "validation", "test"):
                dataset = TargetStateTorchDataset(config, split=split)
                summaries[split] = {
                    "frame_count": dataset.summary.frame_count,
                    "sequence_count": dataset.summary.sequence_count,
                    "episode_ids": list(dataset.summary.episode_ids),
                }
                if len(dataset):
                    sample = dataset[0]
                    summaries[split]["sample_shapes"] = {
                        name: list(value.shape) for name, value in sample.items()
                    }
            print(json.dumps({"ok": True, "dry_run": True, "splits": summaries}, indent=2, ensure_ascii=False))
            return 0 if all(summaries[item]["sequence_count"] for item in summaries) else 1
        result = train_target_state(config)
    except (OSError, ValueError, RuntimeError, TargetStateTrainingError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **result.to_dict()}, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
