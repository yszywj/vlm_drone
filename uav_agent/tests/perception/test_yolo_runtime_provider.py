from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from common.target_estimate import TargetEstimate
from configs.loader import load_config
from env.camera_types import CameraIntrinsics, CameraSample
from env.uav_controller import UAVState
from perception.factory import build_target_perception_runtime
from perception.mode import TargetPerceptionMode, resolve_target_perception_mode
from perception.runtime import PerceptionBoundaryError
from perception.runtime_bridge import CoordinatedVisionPerceptionBackend
from perception.runtime_provider import TargetPerceptionRuntime, YoloTargetPerceptionRuntime
from skills.types import Observation
from target import TargetManager, TargetSpec


ROOT = Path(__file__).resolve().parents[2]


def _sample(timestamp_s: float = 2.0, *, fill: int = 0) -> CameraSample:
    intrinsics = CameraIntrinsics(8.0, 8.0, 3.5, 2.5, 8, 6)
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=np.full((6, 8, 3), fill, dtype=np.uint8),
        depth_to_image_plane_m=np.full((6, 8), 7.0, dtype=np.float32),
        camera_position_world_m=(1.0, 2.0, 3.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=intrinsics,
    )


def _base(sample: CameraSample, *, uav_id: str = "uav_a") -> Observation:
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


def _estimate(
    timestamp_s: float,
    *,
    target_id: str = "candidate_7",
    source: str = "yolo26_botsort",
) -> TargetEstimate:
    return TargetEstimate(
        timestamp_s=timestamp_s,
        target_id=target_id,
        candidate_id="candidate_7",
        tracker_id="tracker_7",
        visible=True,
        confirmed=True,
        predicted_only=False,
        class_id=0,
        class_name="cube",
        confidence=0.9,
        bbox_xyxy_normalized=(0.2, 0.2, 0.6, 0.7),
        position_world_m=(5.0, 6.0, 0.5),
        velocity_world_mps=(0.1, 0.0, 0.0),
        measurement_age_s=0.0,
        source=source,
    )


class _Bridge(CoordinatedVisionPerceptionBackend):
    """Small structural double that still satisfies the concrete guard."""

    def __init__(self, uav_id: str, *, source: str = "yolo26_botsort") -> None:
        self._test_uav_id = uav_id
        self.source = source
        self.reset_calls: list[tuple[str, TargetSpec, str]] = []
        self._target_alias = None
        self.inputs: list[object] = []
        self.attribute_records: list[object] = []
        self.closed = 0

    @property
    def uav_id(self) -> str:
        return self._test_uav_id

    def reset(
        self,
        *,
        mission_id: str,
        target_spec: TargetSpec,
        assignment_id: str | None = None,
        target_alias: str,
    ) -> None:
        del assignment_id
        self._target_alias = target_alias
        self.reset_calls.append((mission_id, target_spec, target_alias))

    def observe(self, synchronized_input, *, target_manager: TargetManager) -> Observation:
        assert isinstance(target_manager, TargetManager)
        self.inputs.append(synchronized_input)
        return replace(
            synchronized_input.base_observation,
            target_estimate=_estimate(
                synchronized_input.base_observation.timestamp,
                target_id=self._target_alias or "unbound_target",
                source=self.source,
            ),
        )

    def metrics(self) -> dict[str, object]:
        return {"frames_submitted": len(self.inputs)}

    def drain_attribute_evidence_records(self) -> tuple[object, ...]:
        values = tuple(self.attribute_records)
        self.attribute_records.clear()
        return values

    def close(self) -> None:
        self.closed += 1


def test_yolo_runtime_uses_atomic_rgbd_and_maps_logical_target_alias() -> None:
    bridge = _Bridge("uav_a")
    runtime = YoloTargetPerceptionRuntime(uav_id="uav_a", bridge=bridge)
    sample = _sample()
    base = _base(sample)
    spec = TargetSpec("red cube", category="cube", hard_attributes=("color=red",))
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )

    observed = runtime.observe(
        base_observation=base,
        camera_sample=sample,
        target_manager=TargetManager(),
    )

    assert isinstance(runtime, TargetPerceptionRuntime)
    assert runtime.mode is TargetPerceptionMode.YOLO
    assert runtime.backend_name == "ultralytics_service"
    assert bridge.reset_calls == [("mission_1", spec, "target_i")]
    assert len(bridge.inputs) == 1
    synchronized = bridge.inputs[0]
    assert synchronized.camera_sample is sample
    assert np.array_equal(
        synchronized.base_observation.camera_rgb,
        sample.rgb,
    )
    assert observed.target_estimate is not None
    assert observed.target_estimate.target_id == "target_i"
    assert observed.target_estimate.class_name == "cube"
    assert observed.target_estimate.source == "yolo26_botsort"
    assert observed.oracle_target_id is None
    assert runtime.metrics()["frames_submitted"] == 1
    assert runtime.metrics()["attribute_evidence_log_errors"] == 0


def test_yolo_runtime_requires_synchronized_camera_sample() -> None:
    bridge = _Bridge("uav_a")
    runtime = YoloTargetPerceptionRuntime(uav_id="uav_a", bridge=bridge)
    spec = TargetSpec("red cube", category="cube")
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )
    sample = _sample()

    with pytest.raises(ValueError, match="requires synchronized RGB-D"):
        runtime.observe(
            base_observation=_base(sample),
            camera_sample=None,
            target_manager=TargetManager(),
        )
    with pytest.raises(ValueError, match="timestamps must match"):
        runtime.observe(
            base_observation=replace(_base(sample), timestamp=sample.timestamp_s + 1.0),
            camera_sample=sample,
            target_manager=TargetManager(),
        )
    with pytest.raises(ValueError, match="RGB must come from the same"):
        runtime.observe(
            base_observation=replace(
                _base(sample),
                camera_rgb=np.full_like(sample.rgb, 255),
            ),
            camera_sample=sample,
            target_manager=TargetManager(),
        )
    with pytest.raises(ValueError, match="pose must come from the same"):
        runtime.observe(
            base_observation=replace(
                _base(sample),
                camera_position_m=np.asarray((9.0, 9.0, 9.0)),
            ),
            camera_sample=sample,
            target_manager=TargetManager(),
        )
    assert bridge.inputs == []


def test_yolo_runtime_rejects_oracle_input_and_oracle_bridge_output() -> None:
    sample = _sample()
    spec = TargetSpec("red cube", category="cube")

    normal = YoloTargetPerceptionRuntime(
        uav_id="uav_a", bridge=_Bridge("uav_a")
    )
    normal.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )
    privileged_base = replace(
        _base(sample),
        target_estimate=_estimate(
            sample.timestamp_s,
            target_id="target_i",
            source="oracle_evaluation",
        ),
    )
    with pytest.raises(PerceptionBoundaryError, match="Oracle"):
        normal.observe(
            base_observation=privileged_base,
            camera_sample=sample,
            target_manager=TargetManager(),
        )

    malicious = YoloTargetPerceptionRuntime(
        uav_id="uav_a",
        bridge=_Bridge("uav_a", source="oracle_evaluation"),
    )
    malicious.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )
    with pytest.raises(PerceptionBoundaryError, match="Oracle"):
        malicious.observe(
            base_observation=_base(sample),
            camera_sample=sample,
            target_manager=TargetManager(),
        )


def test_yolo_runtime_rejects_bridge_target_outside_active_assignment() -> None:
    sample = _sample()
    spec = TargetSpec("red cube", category="cube")

    class WrongTargetBridge(_Bridge):
        def observe(
            self,
            synchronized_input,
            *,
            target_manager: TargetManager,
        ) -> Observation:
            observation = super().observe(
                synchronized_input,
                target_manager=target_manager,
            )
            assert observation.target_estimate is not None
            return replace(
                observation,
                target_estimate=replace(
                    observation.target_estimate,
                    target_id="target_j",
                ),
            )

    runtime = YoloTargetPerceptionRuntime(
        uav_id="uav_a",
        bridge=WrongTargetBridge("uav_a"),
    )
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )

    with pytest.raises(PermissionError, match="active Assignment"):
        runtime.observe(
            base_observation=_base(sample),
            camera_sample=sample,
            target_manager=TargetManager(),
        )


def test_yolo_runtime_rejects_unconfirmed_stable_target_claim() -> None:
    sample = _sample()
    spec = TargetSpec("red cube", category="cube")

    class UnconfirmedTargetBridge(_Bridge):
        def observe(
            self,
            synchronized_input,
            *,
            target_manager: TargetManager,
        ) -> Observation:
            observation = super().observe(
                synchronized_input,
                target_manager=target_manager,
            )
            assert observation.target_estimate is not None
            return replace(
                observation,
                target_estimate=replace(
                    observation.target_estimate,
                    confirmed=False,
                ),
            )

    runtime = YoloTargetPerceptionRuntime(
        uav_id="uav_a",
        bridge=UnconfirmedTargetBridge("uav_a"),
    )
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )

    with pytest.raises(PermissionError, match="unconfirmed.*stable target ID"):
        runtime.observe(
            base_observation=_base(sample),
            camera_sample=sample,
            target_manager=TargetManager(),
        )


def test_yolo_factory_never_reads_evaluator_and_uses_per_uav_url() -> None:
    config = load_config(ROOT / "configs/multi_uav_cube_yolo.yaml")

    class ProductionEnvironment:
        def __getattribute__(self, name: str):
            if name in {"get_evaluator_frame", "make_oracle_perception"}:
                raise AssertionError("YOLO construction accessed evaluator capability")
            return object.__getattribute__(self, name)

    runtime = build_target_perception_runtime(
        config,
        resolved_mode=resolve_target_perception_mode("yolo"),
        environment=ProductionEnvironment(),
        uav_id="uav_b",
    )
    assert isinstance(runtime, YoloTargetPerceptionRuntime)
    assert runtime._bridge.coordinator._config.yolo_service.url == (  # noqa: SLF001
        "http://127.0.0.1:8012"
    )
    runtime.close()


def test_yolo_runtime_close_is_idempotent() -> None:
    bridge = _Bridge("uav_a")
    runtime = YoloTargetPerceptionRuntime(uav_id="uav_a", bridge=bridge)
    runtime.close()
    runtime.close()
    assert bridge.closed == 1


def test_yolo_runtime_drains_each_attribute_record_exactly_once() -> None:
    bridge = _Bridge("uav_a")
    persisted: list[object] = []
    runtime = YoloTargetPerceptionRuntime(
        uav_id="uav_a",
        bridge=bridge,
        attribute_evidence_sink=persisted.append,
    )
    spec = TargetSpec(
        "red cube",
        category="cube",
        hard_attributes=("color=red",),
    )
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )

    first = object()
    bridge.attribute_records.append(first)
    sample_1 = _sample(2.0)
    runtime.observe(
        base_observation=_base(sample_1),
        camera_sample=sample_1,
        target_manager=TargetManager(),
    )
    second = object()
    bridge.attribute_records.append(second)
    sample_2 = _sample(2.1)
    runtime.observe(
        base_observation=_base(sample_2),
        camera_sample=sample_2,
        target_manager=TargetManager(),
    )

    assert persisted == [first, second]
    assert bridge.attribute_records == []
    assert runtime.metrics()["attribute_evidence_log_errors"] == 0
    runtime.close()
