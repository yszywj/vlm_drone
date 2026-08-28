#!/usr/bin/env python3
"""Finalize verified collection tar files into one Target State parent dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from training.target_state.collection_finalize import (  # noqa: E402
    CollectionFinalizationError,
    finalize_target_state_collection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-index",
        type=Path,
        required=True,
        help="completed collection_index.json copied from the collection session",
    )
    parser.add_argument(
        "--shard-dir",
        "--collection-shard-dir",
        dest="shard_dir",
        type=Path,
        required=True,
        help="PC directory containing immutable collection .tar files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new destination directory for the complete parent dataset",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = finalize_target_state_collection(
            args.collection_index,
            args.shard_dir,
            args.output_dir,
        )
    except (CollectionFinalizationError, OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc), "archives_preserved": True},
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
