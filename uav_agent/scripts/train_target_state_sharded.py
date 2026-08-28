#!/usr/bin/env python3
"""Train Target State Stage A/B from pc_trans episode shards (no Isaac import)."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys

import yaml


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from training.target_state.config import (  # noqa: E402
    TrainingStage,
    load_training_config,
)
from training.target_state.sharded_trainer import (  # noqa: E402
    ShardedTrainingError,
    ShardedTrainingOptions,
    train_target_state_sharded,
    validate_resume_checkpoint,
    validate_shard_index_for_training,
)
from training.target_state.shards import load_shard_index  # noqa: E402
from training.target_state.trainer import validate_initial_checkpoint  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PACKAGE_ROOT / "configs" / "target_state" / "train_oracle_clean.yaml",
    )
    parser.add_argument("--shard-index", type=Path, required=True)
    parser.add_argument("--pc-trans-root", type=Path, required=True)
    parser.add_argument("--pc-trans-config", type=Path, required=True)
    parser.add_argument("--pc-trans-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--num-workers", type=_nonnegative_int)
    parser.add_argument("--stage", choices=[item.value for item in TrainingStage])
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="model-only initialization for a new stage (for example Stage B from Stage A best.pt)",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="same-run latest.pt; restores model, optimizer, and shard-boundary progress",
    )
    parser.add_argument("--wait-timeout", type=_nonnegative_float, default=86400.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config/index/checkpoint contracts without pc_trans or training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.initial_checkpoint is not None and args.resume_checkpoint is not None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "--initial-checkpoint and --resume-checkpoint have different semantics and cannot be combined",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        config = load_training_config(args.config)
        overrides = {
            "output_dir": args.output_dir,
            "run_name": args.run_name,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "stage": TrainingStage(args.stage) if args.stage else None,
            "initial_checkpoint_path": args.initial_checkpoint,
        }
        config = replace(
            config,
            **{name: value for name, value in overrides.items() if value is not None},
        )
        options = ShardedTrainingOptions(
            shard_index_path=args.shard_index,
            pc_trans_root=args.pc_trans_root,
            pc_trans_config=args.pc_trans_config,
            pc_trans_python=args.pc_trans_python,
            bridge_root=args.bridge_root,
            run_id_prefix=args.run_id_prefix,
            resume_checkpoint=args.resume_checkpoint,
            wait_timeout_s=args.wait_timeout,
        )
        if args.dry_run:
            index = load_shard_index(options.shard_index_path)
            validate_shard_index_for_training(index, config)
            resume = None
            if options.resume_checkpoint is None:
                validate_initial_checkpoint(config, map_location="cpu")
            else:
                canonical_latest = (
                    config.output_dir / config.run_name / "latest.pt"
                ).resolve()
                if options.resume_checkpoint != canonical_latest:
                    raise ShardedTrainingError(
                        "--resume-checkpoint must be this run's canonical latest.pt; "
                        f"expected={canonical_latest}, actual={options.resume_checkpoint}"
                    )
                resume = validate_resume_checkpoint(
                    options.resume_checkpoint,
                    config=config,
                    index=index,
                )
                if resume.get("run_id_prefix") != options.run_id_prefix:
                    raise ShardedTrainingError(
                        "resume run_id_prefix does not match checkpoint"
                    )
            authoritative_initial_checkpoint = (
                resume.get("initial_checkpoint_path")
                if resume is not None
                else config.initial_checkpoint_path
            )
            result = {
                "ok": True,
                "dry_run": True,
                "training_protocol": "episode_sharded_v1",
                "parent_dataset_sha256": index.parent_dataset_sha256,
                "shard_index_sha256": index.index_sha256,
                "shard_counts": {
                    split: len(index.shards_for_split(split))
                    for split in ("train", "validation", "test")
                },
                "global_epochs": config.epochs,
                "initial_checkpoint": (
                    None
                    if authoritative_initial_checkpoint is None
                    else str(authoritative_initial_checkpoint)
                ),
                "resume_checkpoint": (
                    None
                    if options.resume_checkpoint is None
                    else str(options.resume_checkpoint)
                ),
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        result = train_target_state_sharded(config, options)
    except (
        OSError,
        EOFError,
        TypeError,
        ValueError,
        RuntimeError,
        pickle.UnpicklingError,
        yaml.YAMLError,
        ShardedTrainingError,
    ) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {"ok": True, **result.to_dict()},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
