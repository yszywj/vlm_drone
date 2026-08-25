from __future__ import annotations

import pytest

from perception.factory import preflight_fleet_yolo_services
from perception.mode import TargetPerceptionModeError, resolve_target_perception_mode
from perception.yolo_client import YoloClientUnavailable
from configs.loader import load_config


def test_yolo_mode_forbids_oracle_acknowledgement() -> None:
    with pytest.raises(TargetPerceptionModeError, match="forbidden"):
        resolve_target_perception_mode(
            "yolo",
            runtime_profile="production",
            backend="ultralytics_service",
            acknowledge_privileged_oracle=True,
        )


def test_offline_worker_fails_closed_without_oracle_or_disabled() -> None:
    config = load_config("configs/yolo/runtime_yolo26.yaml")

    class Offline:
        def health(self):
            raise YoloClientUnavailable("offline")

    with pytest.raises(YoloClientUnavailable, match="failed closed") as error:
        preflight_fleet_yolo_services(
            config,
            ("uav_1",),
            client_factory=lambda **_: Offline(),
        )
    message = str(error.value).casefold()
    assert "oracle" not in message
    assert "backend=disabled" not in message
