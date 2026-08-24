#!/usr/bin/env python3
"""Thin production defaults for :mod:`run_dynamic_visual_mission`."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dynamic_visual_mission import main as _mission_main  # noqa: E402


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(value == flag or value.startswith(flag + "=") for value in argv)


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    defaults: list[str] = []
    if not _has_flag(supplied, "--perception-runtime-profile"):
        defaults.extend(("--perception-runtime-profile", "production"))
    if not _has_flag(supplied, "--target-perception-backend"):
        defaults.extend(("--target-perception-backend", "ultralytics_service"))
    return _mission_main((*defaults, *supplied))


if __name__ == "__main__":
    raise SystemExit(main())
