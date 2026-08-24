"""Opt-in real-model smoke test for the isolated YOLO service.

This file has no fake engine seam.  Collection skips it unless the caller
provides a local model, a local known-class image, and either CUDA or an
explicit CPU acknowledgement.
"""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _prerequisites() -> tuple[str | None, Path | None, Path | None, bool, str]:
    model_value = os.environ.get("UAV_AGENT_YOLO_MODEL")
    image_value = os.environ.get("UAV_AGENT_YOLO_TEST_IMAGE")
    model_path = Path(model_value).expanduser() if model_value else None
    image_path = Path(image_value).expanduser() if image_value else None
    allow_cpu = os.environ.get("UAV_AGENT_YOLO_ALLOW_CPU", "").strip().lower() in _TRUE_VALUES

    if model_path is None:
        return "UAV_AGENT_YOLO_MODEL is not set", None, image_path, allow_cpu, "cpu"
    if not model_path.is_file():
        return f"YOLO model does not exist: {model_path}", model_path, image_path, allow_cpu, "cpu"
    if image_path is None:
        return "UAV_AGENT_YOLO_TEST_IMAGE is not set", model_path, None, allow_cpu, "cpu"
    if not image_path.is_file():
        return f"YOLO test image does not exist: {image_path}", model_path, image_path, allow_cpu, "cpu"

    try:
        import torch
    except ImportError:
        return "torch is not installed in this environment", model_path, image_path, allow_cpu, "cpu"
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available and not allow_cpu:
        return (
            "CUDA is unavailable and UAV_AGENT_YOLO_ALLOW_CPU is not acknowledged",
            model_path,
            image_path,
            allow_cpu,
            "cpu",
        )
    default_device = "0" if cuda_available else "cpu"
    device = os.environ.get("UAV_AGENT_YOLO_DEVICE", default_device).strip().lower()
    if device == "cpu" and not allow_cpu and not cuda_available:
        return "CPU inference was not explicitly acknowledged", model_path, image_path, allow_cpu, device
    return None, model_path, image_path, allow_cpu, device


_SKIP_REASON, _MODEL_PATH, _IMAGE_PATH, _ALLOW_CPU, _DEVICE = _prerequisites()


def _jpeg_bytes(path: Path) -> bytes:
    from PIL import Image

    with Image.open(path) as image:
        output = BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=95)
    return output.getvalue()


@pytest.mark.yolo_smoke
@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_real_model_fastapi_health_detection_and_stable_track_id() -> None:
    """Load one real checkpoint and track a known class in two identical frames."""

    from fastapi.testclient import TestClient

    from yolo_service.app import create_app
    from yolo_service.config import ModelFamily, YoloServiceConfig, YoloServiceSettings
    from yolo_service.engine import UltralyticsEngine
    from yolo_service.protocol import (
        ResetStreamRequest,
        TargetQuery,
        TrackRequest,
        TrackResponse,
    )

    assert _MODEL_PATH is not None
    assert _IMAGE_PATH is not None
    tracker_path = PROJECT_ROOT / "configs/yolo/botsort_uav.yaml"
    class_id = int(os.environ.get("UAV_AGENT_YOLO_TEST_CLASS_ID", "0"))
    image_size = int(os.environ.get("UAV_AGENT_YOLO_TEST_IMGSZ", "640"))
    settings = YoloServiceSettings(
        model_family=ModelFamily.YOLO,
        device=_DEVICE,
        tracker_path=str(tracker_path),
        confidence_threshold=0.10,
        image_size_px=image_size,
    )
    engine = UltralyticsEngine(YoloServiceConfig(_MODEL_PATH, settings))
    app = create_app(engine=engine)
    jpeg = _jpeg_bytes(_IMAGE_PATH)

    mission_id = "mission_yolo_smoke"
    uav_id = "uav_1"
    stream_id = f"{mission_id}:{uav_id}"
    reset = ResetStreamRequest(
        schema_version=1,
        request_id="request_smoke_reset_start",
        mission_id=mission_id,
        uav_id=uav_id,
        stream_id=stream_id,
    )
    responses: list[TrackResponse] = []
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["ready"] is True

        model_info = client.get("/v1/model-info")
        assert model_info.status_code == 200
        info = model_info.json()
        assert info["status"] == "ready"
        assert info["model_path"] == str(_MODEL_PATH.resolve())
        assert str(class_id) in {str(key) for key in info["model_names"]}

        start_reset = client.post("/v1/streams/reset", json=reset.to_dict())
        assert start_reset.status_code == 200, start_reset.text
        try:
            for index in range(2):
                request = TrackRequest(
                    schema_version=1,
                    request_id=f"request_smoke_frame_{index}",
                    mission_id=mission_id,
                    uav_id=uav_id,
                    stream_id=stream_id,
                    frame_id=f"frame_smoke_{index}",
                    timestamp_s=float(index + 1),
                    target_query=TargetQuery(class_ids=(class_id,)),
                )
                raw = client.post(
                    "/v1/track",
                    data={"request_json": json.dumps(request.to_dict())},
                    files={
                        "image": (
                            f"frame_{index}.jpg",
                            jpeg,
                            "image/jpeg",
                        )
                    },
                )
                assert raw.status_code == 200, raw.text
                response = TrackResponse.from_dict(raw.json())
                response.assert_matches(request)
                responses.append(response)
        finally:
            cleanup = ResetStreamRequest(
                schema_version=1,
                request_id="request_smoke_reset_finish",
                mission_id=mission_id,
                uav_id=uav_id,
                stream_id=stream_id,
            )
            finish_reset = client.post(
                "/v1/streams/reset", json=cleanup.to_dict()
            )
            assert finish_reset.status_code == 200, finish_reset.text

    first_ids = {detection.track_id for detection in responses[0].detections}
    second_ids = {detection.track_id for detection in responses[1].detections}
    assert first_ids, (
        "known-class image produced no tracked detection; set "
        "UAV_AGENT_YOLO_TEST_CLASS_ID for the image's COCO class"
    )
    assert second_ids
    assert first_ids & second_ids, "consecutive frames did not preserve a track ID"
