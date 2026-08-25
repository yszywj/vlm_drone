#!/usr/bin/env python3
"""Read-only preflight for Fleet YOLO workers.

The command deliberately does not import Isaac Sim and does not create any
mission runtime objects.  It validates the configured per-UAV routing and
queries only ``/health`` and ``/v1/model-info`` through the strict loopback
client used by the real launch preflight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from configs.loader import ConfigError, load_config  # noqa: E402
from perception.factory import (  # noqa: E402
    TargetPerceptionConfigurationError,
    preflight_fleet_yolo_services,
)
from perception.yolo_client import YoloClientError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate isolated Fleet YOLO workers without starting Isaac Sim."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PACKAGE_ROOT / "configs/multi_uav_cube_yolo.yaml",
        help="YOLO Fleet YAML config",
    )
    parser.add_argument(
        "--uav-id",
        action="append",
        default=None,
        help=(
            "active UAV to check; repeat for multiple UAVs (defaults to every "
            "config.uavs entry)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        active_uav_ids = (
            tuple(item.id for item in config.uavs)
            if args.uav_id is None
            else tuple(args.uav_id)
        )
        services = preflight_fleet_yolo_services(config, active_uav_ids)
    except (
        ConfigError,
        OSError,
        TypeError,
        ValueError,
        TargetPerceptionConfigurationError,
        YoloClientError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "config": str(args.config.resolve()),
                "active_uav_ids": list(active_uav_ids),
                "services": services,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

