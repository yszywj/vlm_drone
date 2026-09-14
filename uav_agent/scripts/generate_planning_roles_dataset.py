#!/usr/bin/env python3
"""Generate or validate offline Interpreter/Fleet V2/Spatial V3 SFT data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from planning_data.generator import DEFAULT_TOKENIZER, generate_dataset, validate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "datasets/planning_roles_v1")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--validate-only", type=Path)
    args = parser.parse_args(argv)
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    if args.validate_only:
        result = validate_dataset(args.validate_only, progress=progress)
    else:
        manifest = generate_dataset(args.output, count=args.count, seed=args.seed,
                                    tokenizer_path=args.tokenizer, max_length=args.max_length, progress=progress)
        result = {"output": str(args.output.resolve()), "underlying_tasks": manifest["underlying_tasks"],
                  "total_samples": manifest["total_samples"], "role_split_counts": manifest["role_split_counts"],
                  "token_audit": manifest["token_audit"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
