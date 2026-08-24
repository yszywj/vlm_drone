#!/usr/bin/env python3
"""Aggregate bounded Fleet mission result directories without Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.fleet_batch_evaluator import (  # noqa: E402
    FleetBatchEvaluationError,
    evaluate_fleet_run_directories,
)


def _discover(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {root}")
    return sorted({path.parent.resolve() for path in root.rglob("summary.json")})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path, help="individual Fleet run directories")
    parser.add_argument("--runs-root", type=Path, help="recursively discover summary.json files")
    parser.add_argument(
        "--evaluation-root",
        "--output-dir",
        dest="evaluation_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--no-summary-figures", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_dirs = [path.expanduser().resolve() for path in args.run_dirs]
        if args.runs_root is not None:
            run_dirs.extend(_discover(args.runs_root.expanduser().resolve()))
        run_dirs = sorted(set(run_dirs))
        if not run_dirs:
            raise ValueError("at least one run directory is required")
        summary = evaluate_fleet_run_directories(
            run_dirs,
            args.evaluation_root,
            save_summary_figures=not args.no_summary_figures,
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (FleetBatchEvaluationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"fleet mission evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
