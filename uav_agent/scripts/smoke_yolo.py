#!/usr/bin/env python3
"""Compatibility entry point for :mod:`yolo_service_smoke_test`."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.yolo_service_smoke_test import main


if __name__ == "__main__":
    raise SystemExit(main())
