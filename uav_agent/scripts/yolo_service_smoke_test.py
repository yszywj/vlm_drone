#!/usr/bin/env python3
"""Exercise health, model info, reset and two tracked JPEG frames."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
import sys
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_service.protocol import (  # noqa: E402
    ResetStreamRequest,
    TargetQuery,
    TrackRequest,
    TrackResponse,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--image", required=True, help="local smoke-test image")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--timeout-s", type=_positive_float, default=10.0)
    parser.add_argument(
        "--allow-no-detections",
        action="store_true",
        help="verify the protocol without requiring the chosen image to contain the class",
    )
    return parser


def _loopback_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("base URL must be a loopback HTTP URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a path, query, or fragment")
    return value.rstrip("/")


def _jpeg_bytes(path: str | Path) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required by the smoke test") from exc
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise ValueError(f"test image does not exist: {image_path}")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        output = BytesIO()
        rgb.save(output, format="JPEG", quality=95)
    return output.getvalue()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import httpx

        base_url = _loopback_url(args.base_url)
        jpeg = _jpeg_bytes(args.image)
        mission_id = "mission_yolo_smoke"
        uav_id = "uav_1"
        stream_id = f"{mission_id}:{uav_id}"
        reset = ResetStreamRequest(1, "request_reset_start", mission_id, uav_id, stream_id)
        responses: list[TrackResponse] = []
        with httpx.Client(timeout=args.timeout_s) as client:
            health = client.get(f"{base_url}/health")
            health.raise_for_status()
            model_info = client.get(f"{base_url}/v1/model-info")
            model_info.raise_for_status()
            start_reset = client.post(
                f"{base_url}/v1/streams/reset",
                json=reset.to_dict(),
            )
            start_reset.raise_for_status()
            try:
                for index in range(2):
                    request = TrackRequest(
                        schema_version=1,
                        request_id=f"request_frame_{index}",
                        mission_id=mission_id,
                        uav_id=uav_id,
                        stream_id=stream_id,
                        frame_id=f"frame_{index}",
                        timestamp_s=float(index),
                        target_query=TargetQuery(class_ids=(args.class_id,)),
                    )
                    raw = client.post(
                        f"{base_url}/v1/track",
                        data={"request_json": json.dumps(request.to_dict())},
                        files={"image": (f"frame_{index}.jpg", jpeg, "image/jpeg")},
                    )
                    raw.raise_for_status()
                    response = TrackResponse.from_dict(raw.json())
                    response.assert_matches(request)
                    responses.append(response)
            finally:
                finish_reset = ResetStreamRequest(
                    1,
                    "request_reset_finish",
                    mission_id,
                    uav_id,
                    stream_id,
                )
                cleanup = client.post(
                    f"{base_url}/v1/streams/reset",
                    json=finish_reset.to_dict(),
                )
                cleanup.raise_for_status()
        track_sets = [{item.track_id for item in response.detections} for response in responses]
        stable_track_ids = sorted(track_sets[0] & track_sets[1])
        detections = sum(len(response.detections) for response in responses)
        if detections == 0 and not args.allow_no_detections:
            raise RuntimeError(
                "no requested-class detections; use a known-class image or "
                "--allow-no-detections for protocol-only smoke"
            )
        if all(track_sets) and not stable_track_ids:
            raise RuntimeError("consecutive detections did not preserve a track ID")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "health": health.json(),
                    "model_family": model_info.json().get("model_family"),
                    "detections_total": detections,
                    "stable_track_ids": stable_track_ids,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"[YOLO smoke] FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
