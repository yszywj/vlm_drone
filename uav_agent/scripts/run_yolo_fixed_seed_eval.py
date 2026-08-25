#!/usr/bin/env python3
"""Run at least five fixed-seed production YOLO missions through Isaac Sim.

The isolated YOLO service must already be healthy.  Use the companion
``run_yolo_fixed_seed_eval.sh`` entrypoint to start exactly one checked worker,
reuse it for every episode, and clean it up afterward.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.yolo_fixed_seed_eval import (  # noqa: E402
    DEFAULT_EPISODE_TIMEOUT_S,
    DEFAULT_EXPECTED_TRACK_DURATION_S,
    DEFAULT_FIXED_SEEDS,
    parse_fixed_seeds,
    run_fixed_seed_evaluation,
)


DEFAULT_INSTRUCTION = (
    "uav_1起飞到十米，前往世界坐标10,0附近20米范围搜索红色立方体目标target，"
    "找到后保持约六米距离跟踪二十秒，完成后返回起点降落"
)


def _default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _PROJECT_ROOT / "logs/yolo_fixed_seed_eval" / stamp


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs/yolo/runtime_yolo26.yaml",
        help="complete single-UAV production AppConfig",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_FIXED_SEEDS),
        help="at least five unique comma-separated target motion seeds",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-sim-time", type=float, default=300.0)
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=DEFAULT_EPISODE_TIMEOUT_S,
        help="wall-clock seconds allowed for each Isaac process tree",
    )
    parser.add_argument(
        "--expected-track-duration",
        type=float,
        default=DEFAULT_EXPECTED_TRACK_DURATION_S,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        seeds = parse_fixed_seeds(args.seeds)
        output_root = (
            _default_output_root()
            if args.output_root is None
            else args.output_root.expanduser().resolve()
        )
        result = run_fixed_seed_evaluation(
            project_root=_PROJECT_ROOT,
            source_config=args.config,
            evaluation_root=output_root,
            seeds=seeds,
            instruction=args.instruction,
            max_sim_time_s=args.max_sim_time,
            expected_track_duration_s=args.expected_track_duration,
            episode_timeout_s=args.episode_timeout,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    for record in result.records:
        print(
            "[fixed-seed-yolo] episode="
            + json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    print(
        json.dumps(
            {
                "ok": result.exit_code == 0,
                "exit_code": result.exit_code,
                "evaluation_root": str(output_root),
                "episode_count": len(result.records),
                "strict_success_rate": result.summary["strict_success_rate"],
                "stage_success_rate": result.summary["stage_success_rate"],
                "failure_stage_counts": result.summary["failure_stage_counts"],
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
