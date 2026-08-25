from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from common.target_estimate import TargetEstimate
from configs.loader import load_config
from env.camera_types import CameraIntrinsics, CameraSample
from env.uav_controller import UAVState
from perception.factory import (
    TargetPerceptionConfigurationError,
    preflight_fleet_yolo_services,
)
from perception.runtime_bridge import (
    CoordinatedVisionPerceptionBackend,
    SynchronizedTargetPerceptionInput,
)
from perception.runtime_provider import YoloTargetPerceptionRuntime
from perception.target_perception_coordinator import TargetPerceptionCoordinator
from perception.vision_backend import VisionPerceptionBackend
from perception.yolo_client import YoloClientUnavailable, YoloModelInfo
from scripts import check_fleet_yolo_services
from skills.types import Observation
from target import TargetManager, TargetSpec


ROOT = Path(__file__).resolve().parents[2]


def _sample(timestamp_s: float, *, fill: int = 0) -> CameraSample:
    intrinsics = CameraIntrinsics(8.0, 8.0, 3.5, 2.5, 8, 6)
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=np.full((6, 8, 3), fill, dtype=np.uint8),
        depth_to_image_plane_m=np.full((6, 8), 6.0, dtype=np.float32),
        camera_position_world_m=(1.0, 2.0, 3.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=intrinsics,
    )


def _base(sample: CameraSample, uav_id: str) -> Observation:
    return Observation(
        uav_id=uav_id,
        timestamp=sample.timestamp_s,
        uav_pose=UAVState(0.0, 0.0, 3.0, 0.0),
        uav_velocity=np.zeros(3, dtype=float),
        camera_rgb=np.asarray(sample.rgb).copy(),
        camera_position_m=np.asarray(sample.camera_position_world_m),
        camera_orientation_wxyz=np.asarray(
            sample.camera_orientation_world_wxyz
        ),
    )


def _estimate(timestamp_s: float, *, tracker: str = "tracker_7") -> TargetEstimate:
    return TargetEstimate(
        timestamp_s=timestamp_s,
        target_id="candidate_7",
        candidate_id="candidate_7",
        tracker_id=tracker,
        visible=True,
        confirmed=True,
        predicted_only=False,
        class_id=0,
        class_name="cube",
        confidence=0.9,
        bbox_xyxy_normalized=(0.1, 0.1, 0.4, 0.5),
        position_world_m=(4.0, 5.0, 0.5),
        velocity_world_mps=(0.0, 0.0, 0.0),
        measurement_age_s=0.0,
        source="yolo26_botsort",
    )


class _Metrics:
    def __init__(self, coordinator: "_Coordinator") -> None:
        self.coordinator = coordinator

    def to_dict(self) -> dict[str, object]:
        return {"frames_submitted": len(self.coordinator.submissions)}


class _Coordinator(TargetPerceptionCoordinator):
    def __init__(self, estimate: TargetEstimate | None) -> None:
        self.estimate = estimate
        self.events: list[str] = []
        self.reset_calls: list[tuple[str, str, str]] = []
        self.submissions: list[tuple[CameraSample, TargetSpec]] = []
        self.closed = 0
        self.target_alias: str | None = None
        self.fail_reset = False
        self.metrics = _Metrics(self)

    def reset(
        self,
        *,
        mission_id: str,
        uav_id: str,
        assignment_id: str | None = None,
        target_alias: str | None = None,
    ) -> None:
        del assignment_id
        assert target_alias is not None
        self.reset_calls.append((mission_id, uav_id, target_alias))
        if self.fail_reset:
            raise RuntimeError("injected reset handshake failure")
        self.target_alias = target_alias

    def poll(self, *, now_s: float, target_manager: TargetManager):
        del now_s, target_manager
        self.events.append("poll")
        return (
            None
            if self.estimate is None
            else replace(self.estimate, target_id=self.target_alias)
        )

    def submit_frame(
        self, *, camera_sample: CameraSample, target_spec: TargetSpec
    ) -> bool:
        self.events.append("submit")
        self.submissions.append((camera_sample, target_spec))
        return True

    def close(self) -> None:
        self.closed += 1

    def runtime_metrics(self) -> dict[str, object]:
        return self.metrics.to_dict()


def _runtime(
    config,
    *,
    uav_id: str,
    estimate: TargetEstimate | None,
) -> tuple[YoloTargetPerceptionRuntime, _Coordinator]:
    coordinator = _Coordinator(estimate)
    bridge = CoordinatedVisionPerceptionBackend(
        uav_id=uav_id,
        coordinator=coordinator,
        vision_backend=VisionPerceptionBackend(
            config.target_perception,
            uav_id=uav_id,
        ),
    )
    return YoloTargetPerceptionRuntime(uav_id=uav_id, bridge=bridge), coordinator


def test_bridge_polls_before_submit_and_submits_each_camera_batch_once() -> None:
    config = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")
    runtime, coordinator = _runtime(
        config,
        uav_id="uav_a",
        estimate=None,
    )
    spec = TargetSpec("red cube", category="cube")
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_a",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )
    sample = _sample(1.0)
    manager = TargetManager()

    runtime.observe(
        base_observation=_base(sample, "uav_a"),
        camera_sample=sample,
        target_manager=manager,
    )
    runtime.observe(
        base_observation=_base(sample, "uav_a"),
        camera_sample=sample,
        target_manager=manager,
    )

    assert coordinator.events == ["poll", "submit", "poll"]
    assert len(coordinator.submissions) == 1
    submitted_sample, submitted_spec = coordinator.submissions[0]
    assert submitted_sample is sample
    assert submitted_spec is spec
    assert runtime.metrics()["frames_submitted"] == 1
    assert runtime.metrics()["attribute_evidence_log_errors"] == 0

    stale = _sample(0.5)
    with pytest.raises(ValueError, match="monotonically increasing"):
        runtime.observe(
            base_observation=_base(stale, "uav_a"),
            camera_sample=stale,
            target_manager=manager,
        )
    assert coordinator.events == ["poll", "submit", "poll"]


def test_two_uavs_keep_same_tracker_id_in_independent_runtimes() -> None:
    config = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")
    sample = _sample(2.0)
    estimate = _estimate(sample.timestamp_s, tracker="tracker_7")
    runtime_a, coordinator_a = _runtime(
        config, uav_id="uav_a", estimate=estimate
    )
    runtime_b, coordinator_b = _runtime(
        config, uav_id="uav_b", estimate=estimate
    )
    spec_a = TargetSpec("red cube", category="cube", hard_attributes=("color=red",))
    spec_b = TargetSpec("blue cube", category="cube", hard_attributes=("color=blue",))
    runtime_a.reset(
        mission_id="mission_1",
        assignment_id="assignment_a",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec_a,
    )
    runtime_b.reset(
        mission_id="mission_1",
        assignment_id="assignment_b",
        uav_id="uav_b",
        target_alias="target_j",
        target_spec=spec_b,
    )

    observed_a = runtime_a.observe(
        base_observation=_base(sample, "uav_a"),
        camera_sample=sample,
        target_manager=TargetManager(),
    )
    observed_b = runtime_b.observe(
        base_observation=_base(sample, "uav_b"),
        camera_sample=sample,
        target_manager=TargetManager(),
    )

    assert observed_a.target_estimate is not None
    assert observed_b.target_estimate is not None
    assert observed_a.target_estimate.tracker_id == "tracker_7"
    assert observed_b.target_estimate.tracker_id == "tracker_7"
    assert observed_a.target_estimate.target_id == "target_i"
    assert observed_b.target_estimate.target_id == "target_j"
    assert coordinator_a.reset_calls == [("mission_1", "uav_a", "target_i")]
    assert coordinator_b.reset_calls == [("mission_1", "uav_b", "target_j")]
    assert coordinator_a.submissions[0][1] is spec_a
    assert coordinator_b.submissions[0][1] is spec_b


def test_failed_yolo_rebind_invalidates_old_runtime_and_bridge_route() -> None:
    config = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")
    sample = _sample(2.0)
    runtime, coordinator = _runtime(
        config,
        uav_id="uav_a",
        estimate=_estimate(sample.timestamp_s),
    )
    original = TargetSpec("red cube", category="cube")
    replacement = TargetSpec("blue cube", category="cube")
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_a",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=original,
    )
    assert runtime.target_id == "target_i"
    assert runtime._bridge.target_alias == "target_i"  # noqa: SLF001

    coordinator.fail_reset = True
    with pytest.raises(RuntimeError, match="injected reset handshake failure"):
        runtime.reset(
            mission_id="mission_1",
            assignment_id="assignment_b",
            uav_id="uav_a",
            target_alias="target_j",
            target_spec=replacement,
        )

    assert runtime.target_id is None
    assert runtime._bridge.target_alias is None  # noqa: SLF001
    with pytest.raises(RuntimeError, match="must be reset before observe"):
        runtime.observe(
            base_observation=_base(sample, "uav_a"),
            camera_sample=sample,
            target_manager=TargetManager(),
        )


class _Service:
    def __init__(
        self,
        url: str,
        *,
        ready: bool = True,
        family: str = "yolo",
        names: tuple[tuple[int, str], ...] = ((0, "cube"),),
    ) -> None:
        self.url = url
        self.ready = ready
        self.family = family
        self.names = names
        self.calls: list[str] = []

    def health(self) -> dict[str, object]:
        self.calls.append("health")
        return {
            "schema_version": 1,
            "status": "ok",
            "ready": self.ready,
        }

    def model_info(self) -> YoloModelInfo:
        self.calls.append("model_info")
        return YoloModelInfo(
            self.family,
            self.names,
            "ab" * 32,
        )


def test_preflight_checks_both_isolated_workers_and_records_sha() -> None:
    config = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")
    services: dict[str, _Service] = {}

    def factory(*, base_url: str, request_timeout_s: float, jpeg_quality: int):
        assert request_timeout_s == 0.5
        assert jpeg_quality == 90
        service = _Service(base_url)
        services[base_url] = service
        return service

    result = preflight_fleet_yolo_services(
        config,
        ("uav_a", "uav_b"),
        client_factory=factory,
    )

    assert set(services) == {
        "http://127.0.0.1:8011",
        "http://127.0.0.1:8012",
    }
    assert all(service.calls == ["health", "model_info"] for service in services.values())
    assert result["uav_a"]["model_names"] == {0: "cube"}
    assert result["uav_b"]["model_sha256"] == "ab" * 32


def test_single_uav_preflight_uses_default_url() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    production = replace(
        config,
        target_perception=replace(
            config.target_perception,
            backend="ultralytics_service",
            yolo_service=replace(
                config.target_perception.yolo_service,
                url="http://127.0.0.1:8019",
                per_uav_urls={},
            ),
        ),
    )
    seen_urls: list[str] = []

    def factory(**kwargs):
        seen_urls.append(kwargs["base_url"])
        return _Service(kwargs["base_url"])

    uav_id = production.uavs[0].id
    result = preflight_fleet_yolo_services(
        production,
        (uav_id,),
        client_factory=factory,
    )
    assert seen_urls == ["http://127.0.0.1:8019"]
    assert result[uav_id]["url"] == "http://127.0.0.1:8019"


@pytest.mark.parametrize(
    ("service", "message"),
    (
        (_Service("unused", ready=False), "health"),
        (_Service("unused", family="yoloe"), "model_family"),
        (_Service("unused", names=((0, "person"),)), "class 0='cube'"),
        (_Service("unused", names=((1, "cube"),)), "class 0='cube'"),
        (_Service("unused", names=((0, "cube"), (1, "cube"))), "class 0='cube'"),
    ),
)
def test_preflight_rejects_invalid_worker_contract(
    service: _Service,
    message: str,
) -> None:
    config = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")
    with pytest.raises(TargetPerceptionConfigurationError, match=message):
        preflight_fleet_yolo_services(
            config,
            ("uav_a",),
            client_factory=lambda **kwargs: service,
        )


def test_preflight_fails_closed_without_oracle_fallback() -> None:
    config = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")

    class Offline:
        def health(self):
            raise YoloClientUnavailable("offline")

    with pytest.raises(YoloClientUnavailable, match="offline"):
        preflight_fleet_yolo_services(
            config,
            ("uav_a",),
            client_factory=lambda **kwargs: Offline(),
        )


def test_preflight_rejects_duplicate_resolved_worker_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")
    monkeypatch.setattr(
        "perception.factory.yolo_service_url_for_uav",
        lambda config, uav_id: "http://127.0.0.1:8011",
    )
    with pytest.raises(TargetPerceptionConfigurationError, match="must not share"):
        preflight_fleet_yolo_services(
            config,
            ("uav_a", "uav_b"),
            client_factory=lambda **kwargs: _Service(kwargs["base_url"]),
        )


def test_check_script_prints_machine_readable_service_map(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        check_fleet_yolo_services,
        "preflight_fleet_yolo_services",
        lambda config, active_uav_ids: {
            uav_id: {
                "url": config.target_perception.yolo_service.per_uav_urls[uav_id],
                "model_family": "yolo",
                "model_names": {0: "cube"},
                "model_sha256": None,
                "ready": True,
            }
            for uav_id in active_uav_ids
        },
    )
    code = check_fleet_yolo_services.main(
        [
            "--config",
            str(ROOT / "configs/multi_uav_cube_yolo.yaml"),
            "--uav-id",
            "uav_b",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["active_uav_ids"] == ["uav_b"]
    assert set(payload["services"]) == {"uav_b"}
    assert captured.err == ""
