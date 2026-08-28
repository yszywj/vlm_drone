#!/usr/bin/env python3
"""Build deterministic episode-atomic Target State tar shards (CPU only)."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from training.target_state.shards import (  # noqa: E402
    ShardFormatError,
    build_target_state_shards,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _history_size(value: str) -> int:
    parsed = int(value)
    if not 4 <= parsed <= 8:
        raise argparse.ArgumentTypeError("must be within [4, 8]")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-shard-size-mib",
        type=_positive_float,
        default=512.0,
        help="soft target; a complete oversized episode remains in one shard",
    )
    parser.add_argument("--history-size", type=_history_size, default=6)
    parser.add_argument("--max-history-age-s", type=_positive_float, default=2.0)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    target_bytes = int(args.target_shard_size_mib * 1024 * 1024)
    if target_bytes <= 0:
        print(
            json.dumps(
                {"ok": False, "error": "target shard size rounds below one byte"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        result = build_target_state_shards(
            args.dataset_root,
            args.output_dir,
            target_shard_size_bytes=target_bytes,
            history_size=args.history_size,
            max_history_age_s=args.max_history_age_s,
            split_seed=args.split_seed,
        )
    except (OSError, ValueError, ShardFormatError) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
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
