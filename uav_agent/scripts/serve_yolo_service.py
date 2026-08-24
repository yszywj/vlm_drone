#!/usr/bin/env python3
"""Start the isolated loopback-only Ultralytics/BoT-SORT service."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_service.app import create_app  # noqa: E402
from yolo_service.config import (  # noqa: E402
    ModelFamily,
    YoloServiceConfig,
    YoloServiceConfigurationError,
    load_service_settings,
)
from yolo_service.engine import (  # noqa: E402
    UltralyticsEngine,
    YoloDependencyUnavailable,
    YoloEngineError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/yolo/service_yolo26.yaml",
        help="strict service settings YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("UAV_AGENT_YOLO_MODEL"),
        help="existing local .pt model (or UAV_AGENT_YOLO_MODEL)",
    )
    parser.add_argument("--host", default=None, help="must be 127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model-family", choices=["yolo", "yoloe"], default=None)
    parser.add_argument("--device", default=None, help="GPU index, cpu, or mps")
    parser.add_argument("--tracker", dest="tracker_path", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--imgsz", dest="image_size_px", type=int, default=None)
    parser.add_argument("--max-image-bytes", type=int, default=None)
    parser.add_argument("--max-image-width-px", type=int, default=None)
    parser.add_argument("--max-image-height-px", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    return parser


def _resolved_config(args: argparse.Namespace) -> YoloServiceConfig:
    if not args.model:
        raise YoloServiceConfigurationError(
            "--model or UAV_AGENT_YOLO_MODEL must name an existing local file"
        )
    settings = load_service_settings(args.config)
    overrides = {
        "host": args.host,
        "port": args.port,
        "model_family": (
            ModelFamily.parse(args.model_family) if args.model_family is not None else None
        ),
        "device": args.device,
        "tracker_path": args.tracker_path,
        "confidence_threshold": args.confidence_threshold,
        "iou_threshold": args.iou_threshold,
        "image_size_px": args.image_size_px,
        "max_image_bytes": args.max_image_bytes,
        "max_image_width_px": args.max_image_width_px,
        "max_image_height_px": args.max_image_height_px,
    }
    settings = replace(
        settings,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    return YoloServiceConfig(Path(args.model), settings)


def main(argv: list[str] | None = None, *, uvicorn_module: Any | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = _resolved_config(args)
        engine = UltralyticsEngine(config)
        info = engine.model_info()
        print("[YOLO service] resolved runtime:")
        print(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True))
        app = create_app(engine=engine)
        if uvicorn_module is None:
            try:
                import uvicorn as uvicorn_module
            except ImportError as exc:
                raise YoloDependencyUnavailable(
                    "uvicorn is required in the yolo_perception environment"
                ) from exc
        # workers=1 is an architectural requirement while persist=True state is
        # owned by one in-process tracker.
        uvicorn_module.run(
            app,
            host=config.settings.host,
            port=config.settings.port,
            workers=1,
            log_level=args.log_level,
        )
        return 0
    except (YoloServiceConfigurationError, YoloEngineError, RuntimeError) as exc:
        print(f"[YOLO service] FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
