#!/usr/bin/env python3
"""Run a bounded, diagnostic-only V3 centre-regression overfit test (CPU default)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from target_state_v3_smoke.runner import SmokeOptions, run_smoke


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--acknowledge-diagnostic-only", action="store_true")
    args = parser.parse_args()
    options = SmokeOptions(**{name: getattr(args, name) for name in SmokeOptions.__dataclass_fields__})
    report = run_smoke(args.derived, args.output, options,
                       acknowledge_diagnostic_only=args.acknowledge_diagnostic_only)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    print(f"Report: {args.output.resolve()/'smoke_report.json'}", flush=True)
    if not report["smoke_passed"]:
        print("Completed, but the predeclared regression/gradient smoke criteria were NOT met.", file=sys.stderr)
        sys.exit(2)
