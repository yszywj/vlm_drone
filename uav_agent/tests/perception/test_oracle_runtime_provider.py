from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.target_estimate import TargetEstimate
from configs.loader import load_config
from env.camera_types import CameraIntrinsics, CameraSample
from env.moving_target import TargetState
from env.uav_controller import UAVState
from perception.factory import build_target_perception_runtime
from perception.mode import TargetPerceptionMode, resolve_target_perception_mode
from perception.oracle import OraclePerception
from perception.runtime import (
    GuardedPerceptionBackend,
    PerceptionCapability,
    PerceptionRuntimeProfile,
)
from perception.runtime_provider import (
    OracleTargetPerceptionRuntime,
    TargetPerceptionRuntime,
)
from skills.types import Observation
from target import TargetLifecycle, TargetManager, TargetSpec, TargetStateError


ROOT = Path(__file__).resolve().parents[2]


def _sample(timestamp_s: float = 1.0) -> CameraSample:
    intrinsics = CameraIntrinsics(8.0, 8.0, 3.5, 2.5, 8, 6)
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=np.zeros((6, 8, 3), dtype=np.uint8),
        depth_to_image_plane_m=np.full((6, 8), 5.0, dtype=np.float32),
        camera_position_world_m=(1.0, 2.0, 3.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=intrinsics,
    )


def _base(sample: CameraSample, *, uav_id: str = "uav_a") -> Observation:
    return Observation(
        uav_id=uav_id,
        timestamp=sample.timestamp_s,
        uav_pose=UAVState(0.0, 0.0, 2.0, 0.0),
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
    target_id: str,
    visible: bool,
) -> TargetEstimate:
    return TargetEstimate(
        timestamp_s=timestamp_s,
        target_id=target_id,
        candidate_id=None,
        tracker_id=None,
        visible=visible,
        confirmed=True,
        predicted_only=False,
        class_id=None,
        class_name=None,
        confidence=1.0,
        bbox_xyxy_normalized=(0.2, 0.2, 0.3, 0.3) if visible else None,
        position_world_m=(10.0, 20.0, 0.5),
        velocity_world_mps=(0.0, 0.0, 0.0),
        measurement_age_s=0.0,
        source="oracle_evaluation",
    )


class _OracleBackend:
    capability = PerceptionCapability.PRIVILEGED_ORACLE

    def __init__(self, *, uav_id: str, target_id: str) -> None:
        self.uav_id = uav_id
        self.target_id = target_id
        self.calls: list[object] = []

    def observe(self, frame: object) -> Observation:
        self.calls.append(frame)
        assert isinstance(frame, Observation)
        return frame


def _runtime(
    *,
    target_id: str,
    frames: list[Observation],
    calls: list[tuple[str, str]],
) -> OracleTargetPerceptionRuntime:
    backend = _OracleBackend(uav_id="uav_a", target_id=target_id)
    guarded = GuardedPerceptionBackend(
        backend,
        profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
        acknowledge_privileged_oracle=True,
    )

    def frame_provider(uav_id: str, routed_target_id: str) -> Observation:
        calls.append((uav_id, routed_target_id))
        return frames.pop(0)

    return OracleTargetPerceptionRuntime(
        uav_id="uav_a",
        oracle_backend=guarded,
        frame_provider=frame_provider,
    )


def test_oracle_runtime_emits_unified_assignment_bound_estimate() -> None:
    sample = _sample()
    base = _base(sample)
    oracle_observation = replace(
        base,
        target_estimate=_estimate(
            sample.timestamp_s,
            target_id="target_i",
            visible=True,
        ),
    )
    routed_calls: list[tuple[str, str]] = []
    runtime = _runtime(
        target_id="target_i",
        frames=[oracle_observation],
        calls=routed_calls,
    )
    spec = TargetSpec("red cube", category="cube", hard_attributes=("color=red",))
    manager = TargetManager()
    manager.start_search(spec, 0.0)

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
        target_manager=manager,
    )

    assert isinstance(runtime, TargetPerceptionRuntime)
    assert runtime.mode is TargetPerceptionMode.ORACLE
    assert runtime.backend_name == "oracle_evaluation"
    assert routed_calls == [("uav_a", "target_i")]
    assert observed.target_estimate is not None
    assert observed.target_estimate.target_id == "target_i"
    assert observed.target_estimate.source == "oracle_evaluation"
    snapshot = manager.snapshot()
    assert snapshot.lifecycle is TargetLifecycle.LOCKED
    assert snapshot.target_id == "target_i"
    assert snapshot.last_seen_position == (10.0, 20.0, 0.5)
    assert snapshot.last_seen_velocity == (0.0, 0.0, 0.0)
    assert snapshot.last_seen_time_s == sample.timestamp_s
    assert manager.events()[-1].reason == "target_locked_by_oracle_evaluation"
    assert runtime.metrics() == {
        "oracle_visible_frames": 1,
        "oracle_total_frames": 1,
        "oracle_visible_ratio": 1.0,
        "time_to_first_oracle_visibility_s": 1.0,
        "target_lost_count": 0,
        "reacquire_attempts": 0,
        "reacquire_successes": 0,
    }


def test_invisible_oracle_frame_does_not_change_search_lifecycle() -> None:
    sample = _sample()
    base = _base(sample)
    runtime = _runtime(
        target_id="target_i",
        frames=[
            replace(
                base,
                target_estimate=_estimate(
                    sample.timestamp_s,
                    target_id="target_i",
                    visible=False,
                ),
            )
        ],
        calls=[],
    )
    spec = TargetSpec("red cube", category="cube")
    manager = TargetManager()
    manager.start_search(spec, 0.0)
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
        target_manager=manager,
    )

    assert observed.target_estimate is None
    assert observed.oracle_target_id is None
    assert observed.oracle_target_visible is None
    assert observed.oracle_target_pose is None
    assert observed.oracle_target_velocity is None
    assert manager.lifecycle is TargetLifecycle.SEARCHING
    assert runtime.metrics() == {
        "oracle_visible_frames": 0,
        "oracle_total_frames": 1,
        "oracle_visible_ratio": 0.0,
        "time_to_first_oracle_visibility_s": None,
        "target_lost_count": 0,
        "reacquire_attempts": 0,
        "reacquire_successes": 0,
    }


def test_visible_oracle_lifecycle_transition_is_same_frame_idempotent() -> None:
    sample = _sample()
    base = _base(sample)
    visible = replace(
        base,
        target_estimate=_estimate(
            sample.timestamp_s,
            target_id="target_i",
            visible=True,
        ),
    )
    runtime = _runtime(
        target_id="target_i",
        frames=[visible, visible],
        calls=[],
    )
    spec = TargetSpec("red cube", category="cube")
    manager = TargetManager()
    # Equal lifecycle/frame timestamps are valid and must not need an epsilon.
    manager.start_search(spec, sample.timestamp_s)
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )

    runtime.observe(
        base_observation=base,
        camera_sample=sample,
        target_manager=manager,
    )
    events_after_first_observe = manager.events()
    runtime.observe(
        base_observation=base,
        camera_sample=sample,
        target_manager=manager,
    )

    assert manager.lifecycle is TargetLifecycle.LOCKED
    assert manager.events() == events_after_first_observe
    assert [event.timestamp_s for event in manager.events()] == [1.0, 1.0]
    assert runtime.metrics()["oracle_total_frames"] == 2
    assert runtime.metrics()["oracle_visible_frames"] == 2


def test_visible_oracle_frame_advances_reacquiring_with_estimate_state() -> None:
    sample = _sample()
    base = _base(sample)
    visible = replace(
        base,
        target_estimate=_estimate(
            sample.timestamp_s,
            target_id="target_i",
            visible=True,
        ),
    )
    runtime = _runtime(
        target_id="target_i",
        frames=[visible],
        calls=[],
    )
    spec = TargetSpec("red cube", category="cube")
    manager = TargetManager()
    manager.start_search(spec, 0.0)
    manager.lock_oracle_from_search("target_i", timestamp_s=0.1)
    manager.start_tracking(timestamp_s=0.2)
    manager.mark_lost(timestamp_s=0.9)
    manager.start_reacquiring(timestamp_s=sample.timestamp_s)
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )

    runtime.observe(
        base_observation=base,
        camera_sample=sample,
        target_manager=manager,
    )

    snapshot = manager.snapshot()
    assert snapshot.lifecycle is TargetLifecycle.LOCKED
    assert snapshot.target_id == "target_i"
    assert snapshot.last_seen_position == (10.0, 20.0, 0.5)
    assert snapshot.last_seen_velocity == (0.0, 0.0, 0.0)
    assert snapshot.last_seen_time_s == sample.timestamp_s
    assert manager.events()[-1].reason == "target_reacquired"


def test_oracle_runtime_counts_real_lifecycle_edges_once_and_reset_clears() -> None:
    samples = [_sample(timestamp) for timestamp in (1.0, 2.0, 3.0)]
    bases = [_base(sample) for sample in samples]
    invisible = replace(
        bases[0],
        target_estimate=_estimate(1.0, target_id="target_i", visible=False),
    )
    visible_at_two = replace(
        bases[1],
        target_estimate=_estimate(2.0, target_id="target_i", visible=True),
    )
    visible_at_three = replace(
        bases[2],
        target_estimate=_estimate(3.0, target_id="target_i", visible=True),
    )
    runtime = _runtime(
        target_id="target_i",
        frames=[invisible, visible_at_two, visible_at_three, visible_at_three],
        calls=[],
    )
    spec = TargetSpec("red cube", category="cube")
    manager = TargetManager()
    manager.start_search(spec, 0.0)
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )

    runtime.observe(
        base_observation=bases[0],
        camera_sample=samples[0],
        target_manager=manager,
    )
    runtime.observe(
        base_observation=bases[1],
        camera_sample=samples[1],
        target_manager=manager,
    )
    manager.start_tracking(timestamp_s=2.0)
    manager.mark_lost(timestamp_s=2.5)
    manager.start_reacquiring(timestamp_s=2.5)
    runtime.observe(
        base_observation=bases[2],
        camera_sample=samples[2],
        target_manager=manager,
    )

    lifecycle_metrics = runtime.metrics()
    assert lifecycle_metrics["target_lost_count"] == 1
    assert lifecycle_metrics["reacquire_attempts"] == 1
    assert lifecycle_metrics["reacquire_successes"] == 1

    # A repeated provider call for the same frame increments frame-call
    # metrics but must never replay TargetManager lifecycle edges.
    runtime.observe(
        base_observation=bases[2],
        camera_sample=samples[2],
        target_manager=manager,
    )
    repeated_metrics = runtime.metrics()
    assert repeated_metrics["target_lost_count"] == 1
    assert repeated_metrics["reacquire_attempts"] == 1
    assert repeated_metrics["reacquire_successes"] == 1

    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )
    reset_metrics = runtime.metrics()
    assert reset_metrics["target_lost_count"] == 0
    assert reset_metrics["reacquire_attempts"] == 0
    assert reset_metrics["reacquire_successes"] == 0


def test_oracle_lifecycle_rejects_backward_frame_without_counting_it() -> None:
    sample = _sample()
    base = _base(sample)
    runtime = _runtime(
        target_id="target_i",
        frames=[
            replace(
                base,
                target_estimate=_estimate(
                    sample.timestamp_s,
                    target_id="target_i",
                    visible=True,
                ),
            )
        ],
        calls=[],
    )
    spec = TargetSpec("red cube", category="cube")
    manager = TargetManager()
    manager.start_search(spec, 2.0)
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )

    with pytest.raises(TargetStateError, match="cannot move backward"):
        runtime.observe(
            base_observation=base,
            camera_sample=sample,
            target_manager=manager,
        )

    assert manager.lifecycle is TargetLifecycle.SEARCHING
    assert runtime.metrics()["oracle_total_frames"] == 0


def test_oracle_perception_does_not_release_off_fov_ground_truth() -> None:
    sample = _sample()
    frame = SimpleNamespace(
        observation=SimpleNamespace(
            camera_timestamp_s=sample.timestamp_s,
            uav_state=UAVState(0.0, 0.0, 2.0, 0.0),
            uav_velocity_mps=np.zeros(3, dtype=float),
            rgb=np.asarray(sample.rgb).copy(),
            camera_position_m=np.asarray(sample.camera_position_world_m),
            camera_orientation_wxyz=np.asarray(
                sample.camera_orientation_world_wxyz
            ),
        ),
        target_projection=SimpleNamespace(
            visible=np.asarray([False], dtype=np.bool_),
        ),
        target_state=TargetState(99.0, 88.0, 0.5, 0.0),
        target_velocity_mps=np.asarray([7.0, 6.0, 0.0]),
    )

    observed = OraclePerception(
        uav_id="uav_a",
        target_id="target_i",
    ).observe(frame)

    assert observed.target_estimate is None
    assert observed.oracle_target_id is None
    assert observed.oracle_target_visible is None
    assert observed.oracle_target_pose is None
    assert observed.oracle_target_velocity is None


def test_oracle_runtime_rejects_stale_assignment_binding() -> None:
    runtime = _runtime(target_id="target_i", frames=[], calls=[])
    spec = TargetSpec("blue cube", category="cube")
    with pytest.raises(PermissionError, match="rebound"):
        runtime.reset(
            mission_id="mission_1",
            assignment_id="assignment_2",
            uav_id="uav_a",
            target_alias="target_j",
            target_spec=spec,
        )

    runtime.close()
    runtime.close()  # cleanup is idempotent
    with pytest.raises(RuntimeError, match="closed"):
        runtime.reset(
            mission_id="mission_1",
            assignment_id="assignment_1",
            uav_id="uav_a",
            target_alias="target_i",
            target_spec=spec,
        )


def test_oracle_runtime_requires_existing_privileged_guard() -> None:
    raw = _OracleBackend(uav_id="uav_a", target_id="target_i")
    with pytest.raises(TypeError, match="GuardedPerceptionBackend"):
        OracleTargetPerceptionRuntime(
            uav_id="uav_a",
            oracle_backend=raw,
            frame_provider=lambda uav_id, target_id: object(),
        )


def test_oracle_runtime_rejects_evaluator_frame_from_another_camera_tick() -> None:
    sample = _sample(1.0)
    later = _sample(2.0)
    base = _base(sample)
    stale_output = replace(
        _base(later),
        target_estimate=_estimate(2.0, target_id="target_i", visible=True),
    )
    runtime = _runtime(
        target_id="target_i",
        frames=[stale_output],
        calls=[],
    )
    spec = TargetSpec("red cube", category="cube")
    runtime.reset(
        mission_id="mission_1",
        assignment_id="assignment_1",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec,
    )
    with pytest.raises(PermissionError, match="not synchronized"):
        runtime.observe(
            base_observation=base,
            camera_sample=sample,
            target_manager=TargetManager(),
        )


def test_oracle_factory_uses_assignment_scoped_environment_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/multi_uav_oracle.yaml")

    class Environment:
        def __init__(self) -> None:
            self.created: list[str] = []

        def make_oracle_perception(self, uav_id: str) -> OraclePerception:
            self.created.append(uav_id)
            return OraclePerception(uav_id=uav_id, target_id="target_i")

        def get_evaluator_frame(self, uav_id: str, target_id: str) -> object:
            raise AssertionError("factory construction must not read evaluator data")

    def forbidden_yolo(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Oracle mode constructed a YOLO component")

    monkeypatch.setattr("perception.factory.VisionPerceptionBackend", forbidden_yolo)
    environment = Environment()
    runtime = build_target_perception_runtime(
        config,
        resolved_mode=resolve_target_perception_mode(
            "oracle", acknowledge_privileged_oracle=True
        ),
        environment=environment,
        uav_id="uav_a",
    )

    assert isinstance(runtime, OracleTargetPerceptionRuntime)
    assert environment.created == ["uav_a"]
    runtime.close()
