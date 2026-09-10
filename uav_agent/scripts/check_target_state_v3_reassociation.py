#!/usr/bin/env python3
"""Independently replay a V3 sidecar and check actual CPU training windows."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from target_state_v3_derived.ground_overlay import check_overlay
from training.target_state.sharded_trainer import _atomic_write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = check_overlay(args.derived)
    _atomic_write_json(args.derived.resolve()/"acceptance_report.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}, indent=2))
