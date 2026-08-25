#!/usr/bin/env python3
"""Validate a temporal target-state dataset without importing Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from datasets.target_state.dataset import check_dataset  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frames", default="frames.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--history-size",
        type=int,
        default=None,
        help="history observations per sequence; defaults to dataset_manifest.json or 6",
    )
    parser.add_argument(
        "--max-history-age-s",
        type=float,
        default=None,
        help="maximum temporal span; defaults to dataset_manifest.json or 2.0",
    )
    return parser


def _manifest_sequence_parameters(dataset: Path) -> tuple[int, float]:
    """Read collection-time sequence parameters when a manifest is available."""

    manifest_path = dataset / "dataset_manifest.json"
    if not manifest_path.is_file():
        return 6, 2.0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("dataset_manifest.json must contain a JSON object")
    return int(payload.get("history_size", 6)), float(payload.get("max_history_age_s", 2.0))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    try:
        manifest_history_size, manifest_max_history_age_s = _manifest_sequence_parameters(
            dataset
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "ok": False,
            "error": f"cannot read dataset sequence parameters: {exc}",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 1
    report = check_dataset(
        dataset,
        frames_path=args.frames,
        history_size=(
            manifest_history_size if args.history_size is None else args.history_size
        ),
        max_history_age_s=(
            manifest_max_history_age_s
            if args.max_history_age_s is None
            else args.max_history_age_s
        ),
        split_seed=args.split_seed,
    )
    payload = report.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
