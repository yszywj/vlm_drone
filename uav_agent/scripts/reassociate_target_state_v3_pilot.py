#!/usr/bin/env python3
"""Create a separate, immutable V3 ground-role annotation sidecar (CPU only)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from target_state_v3_derived.ground_overlay import create_overlay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = create_overlay(args.session, args.output)
    print(json.dumps({"complete": True, "output": str(args.output.resolve()),
        "physical_captures": result["physical_captures"], "annotation_only": True,
        "source_modified": False, "ready_for_training": False}, indent=2))
