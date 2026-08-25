from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from configs.loader import load_config
from env.camera_types import CameraFrameNotReady, CameraIntrinsics, CameraSample
from env.fleet_uav_search_env import FleetUavSearchEnv
from env.fleet_uav_search_env import FleetPoseSnapshot
from env.moving_target import TargetState
from env.uav_controller import UAVState


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _State:
    entity_id: str
    step_count: int


class _World:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.current_time = 0.0

    def step(self, *, render: bool) -> None:
        assert render is True
        self.current_time += 1.0 / 60.0
        self.events.append("world")

    def reset(self) -> None:
        self.current_time = 0.0
        self.events.append("reset_world")

    def stop(self) -> None:
        self.events.append("stop_world")

    @classmethod
    def clear_instance(cls) -> None:
        pass


class _Entity:
    def __init__(self, entity_id: str, events: list[str]) -> None:
        self.entity_id = entity_id
        self.events = events
        self.step_count = 0

    def step(self, dt_s: float) -> _State:
        assert dt_s > 0
        self.step_count += 1
        prefix = "target" if self.entity_id.startswith("target") else "uav"
        self.events.append(f"{prefix}:{self.entity_id}")
        return self.get_pose()

    def get_pose(self) -> _State:
        return _State(self.entity_id, self.step_count)

    def get_velocity(self):
        return [float(self.step_count), 0.0, 0.0]

    def set_pose(self, *values, **kwargs) -> None:
        self.step_count = 0
        self.events.append(f"reset_uav:{self.entity_id}")

    def reset(self, **kwargs) -> None:
        self.step_count = 0
        self.events.append(f"reset_target:{self.entity_id}")


class _Sensor:
    def __init__(self) -> None:
        self.invalidations = 0
        self.destroyed = False

    def invalidate_frame(self) -> None:
        if self.destroyed:
            raise AssertionError("destroyed Camera must not be invalidated")
        self.invalidations += 1

    def destroy(self) -> None:
        self.destroyed = True


class _SampleSensor(_Sensor):
    def __init__(
        self,
        timestamp_s: float,
        render_frame_id: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.sample = CameraSample(
            timestamp_s=timestamp_s,
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_to_image_plane_m=np.ones((2, 2), dtype=np.float32),
            camera_position_world_m=(0.0, 0.0, 1.0),
            camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            intrinsics=CameraIntrinsics(1.0, 1.0, 0.5, 0.5, 2, 2),
            render_frame_id=render_frame_id,
        )

    def get_sample(self) -> CameraSample:
        return self.sample

    def get_render_metadata(self) -> tuple[float, tuple[int, int] | None]:
        return self.sample.timestamp_s, self.sample.render_frame_id


class _NotReadySampleSensor(_Sensor):
    def get_render_metadata(self):
        raise CameraFrameNotReady("annotator warm-up")

    def get_sample(self):
        raise CameraFrameNotReady("annotator warm-up")


def _environment() -> tuple[FleetUavSearchEnv, list[str]]:
    config = load_config(PROJECT_ROOT / "configs" / "multi_uav_demo.yaml")
    environment = FleetUavSearchEnv(
        config,
        assignments={"uav_a": "target_i", "uav_b": "target_j"},
    )
    events: list[str] = []
    environment.world = _World(events)
    environment.uav_controllers = {
        uav_id: _Entity(uav_id, events) for uav_id in reversed(environment.uav_ids)
    }
    environment.target_motions = {
        target_id: _Entity(target_id, events)
        for target_id in reversed(environment.target_ids)
    }
    environment.camera_sensors = {uav_id: _Sensor() for uav_id in environment.uav_ids}

    def capture(timestamp_s):
        events.append(f"capture:{environment._tick_index}")
        return {
            uav_id: SimpleNamespace(timestamp_s=timestamp_s)
            for uav_id in environment.uav_ids
        }

    def publish(snapshot, samples):
        assert set(samples) == set(environment.uav_ids)
        environment.latest_agent_observations = {
            uav_id: {"camera_owner": uav_id, "pixels": [uav_id]}
            for uav_id in environment.uav_ids
        }
        environment.latest_evaluator_frames = {
            (uav_id, target_id): {"uav_id": uav_id, "target_id": target_id}
            for uav_id, target_id in environment.assignments.items()
        }
        return len(environment.uav_ids)

    environment._capture_camera_batch = capture  # type: ignore[method-assign]
    environment._publish_observation_batch = publish  # type: ignore[method-assign]
    return environment, events


def test_unique_prim_paths_and_inventory_api() -> None:
    environment, _ = _environment()

    assert environment.uav_ids == ("uav_a", "uav_b")
    assert environment.target_ids == ("target_i", "target_j")
    assert len(set(environment.uav_prim_paths.values())) == 2
    assert len(set(environment.camera_prim_paths.values())) == 2
    assert len(set(environment.target_prim_paths.values())) == 2
    assert environment.camera_prim_paths["uav_a"] == "/World/UAVs/uav_a/Camera"


def test_tick_barrier_captures_once_then_safety_and_ticks_sorted_uavs() -> None:
    environment, events = _environment()
    seen_snapshot_ids: list[int] = []

    def safety(snapshot) -> None:
        events.append("safety")
        seen_snapshot_ids.append(id(environment._fleet_pose_snapshot))
        assert tuple(snapshot.uav_states) == environment.uav_ids
        assert tuple(snapshot.uav_velocities_mps) == environment.uav_ids
        assert tuple(snapshot.target_states) == environment.target_ids

    results = environment.tick_uavs(
        {
            "uav_b": lambda: events.append("agent:uav_b") or "b",
            "uav_a": lambda: events.append("agent:uav_a") or "a",
        },
        global_safety_check=safety,
    )

    assert events == [
        "target:target_i",
        "target:target_j",
        "uav:uav_a",
        "uav:uav_b",
        "world",
        "capture:1",
        "safety",
        "agent:uav_a",
        "agent:uav_b",
    ]
    assert tuple(results) == ("uav_a", "uav_b")
    assert environment.last_tick_order == ("uav_a", "uav_b")
    assert len(seen_snapshot_ids) == 1
    assert environment.get_fleet_pose_snapshot().timestamp_s == pytest.approx(1.0 / 60.0)


def _camera_batch_environment(
    timestamp_a_s: float,
    timestamp_b_s: float,
) -> tuple[FleetUavSearchEnv, FleetPoseSnapshot]:
    config = load_config(PROJECT_ROOT / "configs" / "multi_uav_demo.yaml")
    environment = FleetUavSearchEnv(
        config,
        assignments={"uav_a": "target_i", "uav_b": "target_j"},
    )
    environment.camera_sensors = {
        "uav_a": _SampleSensor(timestamp_a_s),
        "uav_b": _SampleSensor(timestamp_b_s),
    }
    environment.scene = SimpleNamespace(
        world_to_image_for=lambda uav_id, position: (uav_id, tuple(position))
    )
    environment.target_motions = {
        target_id: SimpleNamespace(
            get_velocity=lambda: np.zeros(3, dtype=np.float64)
        )
        for target_id in environment.target_ids
    }
    return environment, FleetPoseSnapshot(
        tick_index=72,
        timestamp_s=1.2,
        uav_states={
            uav_id: UAVState(0.0, 0.0, 1.0, 0.0)
            for uav_id in environment.uav_ids
        },
        uav_velocities_mps={
            uav_id: np.zeros(3, dtype=np.float64)
            for uav_id in environment.uav_ids
        },
        target_states={
            target_id: TargetState(1.0, 2.0, 0.5, 0.0)
            for target_id in environment.target_ids
        },
    )


def test_camera_batch_preserves_exact_renderer_timestamp_and_frame_id() -> None:
    environment, snapshot = _camera_batch_environment(1.2, 1.2)
    for sensor in environment.camera_sensors.values():
        sensor.sample = CameraSample(
            timestamp_s=sensor.sample.timestamp_s,
            rgb=sensor.sample.rgb,
            depth_to_image_plane_m=sensor.sample.depth_to_image_plane_m,
            camera_position_world_m=sensor.sample.camera_position_world_m,
            camera_orientation_world_wxyz=sensor.sample.camera_orientation_world_wxyz,
            intrinsics=sensor.sample.intrinsics,
            render_frame_id=(72, 60),
        )

    assert environment._refresh_all_observations(snapshot) == 2
    for observation in environment.latest_agent_observations.values():
        assert observation.camera_timestamp_s == 1.2
        assert observation.camera_sample.timestamp_s == 1.2
        assert observation.camera_sample.render_frame_id == (72, 60)
    assert environment._last_camera_timestamps_s == {
        "uav_a": 1.2,
        "uav_b": 1.2,
    }


def test_camera_batch_rejects_cross_camera_render_skew_without_publication() -> None:
    environment, snapshot = _camera_batch_environment(
        1.183333395048976,
        1.200000062584877,
    )

    with pytest.raises(RuntimeError, match="same render timestamp"):
        environment._refresh_all_observations(snapshot)

    assert environment.latest_agent_observations == {}
    assert environment.latest_evaluator_frames == {}
    assert environment._last_camera_timestamps_s == {}


def test_camera_batch_rejects_even_one_renderer_tick_world_lag() -> None:
    environment, snapshot = _camera_batch_environment(1.183333395048976, 1.183333395048976)

    with pytest.raises(RuntimeError, match="outside the current renderer barrier"):
        environment._refresh_all_observations(snapshot)

    assert environment.latest_agent_observations == {}
    assert environment._last_camera_timestamps_s == {}


def test_camera_batch_rejects_more_than_one_renderer_tick_world_lag() -> None:
    environment, snapshot = _camera_batch_environment(1.16, 1.16)

    with pytest.raises(RuntimeError, match="outside the current renderer barrier"):
        environment._refresh_all_observations(snapshot)

    assert environment.latest_agent_observations == {}
    assert environment.latest_evaluator_frames == {}
    assert environment._last_camera_timestamps_s == {}


def test_camera_batch_withholds_stale_reset_frame_until_renderer_catches_up() -> None:
    environment, snapshot = _camera_batch_environment(1.16, 1.16)
    environment._camera_warmup_pending = True

    assert environment._refresh_all_observations(snapshot) == 0
    assert environment.latest_agent_observations == {}
    assert environment.latest_evaluator_frames == {}
    assert environment._camera_warmup_pending is True

    current_timestamp_s = 1.2
    for sensor in environment.camera_sensors.values():
        sensor.sample = replace(sensor.sample, timestamp_s=current_timestamp_s)

    assert environment._refresh_all_observations(snapshot) == 2
    assert environment._last_camera_timestamps_s == {
        "uav_a": current_timestamp_s,
        "uav_b": current_timestamp_s,
    }
    assert environment._camera_warmup_pending is False


def test_camera_batch_rejects_cross_camera_render_id_even_at_same_timestamp() -> None:
    environment, snapshot = _camera_batch_environment(1.2, 1.2)
    environment.camera_sensors = {
        "uav_a": _SampleSensor(1.2, (72, 60)),
        "uav_b": _SampleSensor(1.2, (73, 60)),
    }

    with pytest.raises(RuntimeError, match="same renderer frame ID"):
        environment._refresh_all_observations(snapshot)

    assert environment.latest_agent_observations == {}
    assert environment.latest_evaluator_frames == {}


def test_camera_batch_uses_one_shared_software_delivery_cadence() -> None:
    environment, snapshot = _camera_batch_environment(1.0, 1.0)

    assert environment._refresh_all_observations(replace(snapshot, timestamp_s=1.0)) == 2

    for sensor in environment.camera_sensors.values():
        sensor.sample = replace(sensor.sample, timestamp_s=1.0 + 1.0 / 60.0)
    assert environment._refresh_all_observations(
        replace(snapshot, timestamp_s=1.0 + 2.0 / 60.0)
    ) == 0
    assert environment._last_camera_timestamps_s == {
        "uav_a": 1.0,
        "uav_b": 1.0,
    }

    for sensor in environment.camera_sensors.values():
        sensor.sample = replace(sensor.sample, timestamp_s=1.1)
    assert environment._refresh_all_observations(replace(snapshot, timestamp_s=1.1)) == 2
    assert environment._last_camera_timestamps_s == {
        "uav_a": 1.1,
        "uav_b": 1.1,
    }


def test_non_due_intermediate_camera_frame_can_withhold_two_tick_pipeline_lag() -> None:
    environment, snapshot = _camera_batch_environment(1.05, 1.05)
    environment._last_camera_timestamps_s = {"uav_a": 1.0, "uav_b": 1.0}

    assert environment._refresh_all_observations(
        replace(snapshot, timestamp_s=1.05 + 2.0 / 60.0)
    ) == 0
    assert environment._last_camera_timestamps_s == {
        "uav_a": 1.0,
        "uav_b": 1.0,
    }

    for sensor in environment.camera_sensors.values():
        sensor.sample = replace(sensor.sample, timestamp_s=1.1)
    assert environment._refresh_all_observations(replace(snapshot, timestamp_s=1.1)) == 2


def test_due_camera_batch_drains_renderer_without_advancing_world_or_poses() -> None:
    environment, _ = _camera_batch_environment(0.15, 0.15)
    # The cached renderer frame is only 0.05 s newer than the last published
    # frame, but the world clock is a full 0.10 s newer and therefore due.
    environment._last_camera_timestamps_s = {"uav_a": 0.1, "uav_b": 0.1}

    class CatchupWorld:
        current_time = 0.2

        def __init__(self) -> None:
            self.render_count = 0

        def render(self) -> None:
            self.render_count += 1
            for sensor in environment.camera_sensors.values():
                sensor.sample = replace(
                    sensor.sample,
                    timestamp_s=min(
                        self.current_time,
                        sensor.sample.timestamp_s + 1.0 / 60.0,
                    ),
                )

    world = CatchupWorld()
    environment.world = world

    samples = environment._capture_camera_batch(world.current_time)

    assert world.render_count == 3
    assert all(sample.timestamp_s == pytest.approx(0.2) for sample in samples.values())
    assert world.current_time == 0.2


def test_due_camera_batch_drains_temporary_cross_product_skew() -> None:
    environment, _ = _camera_batch_environment(0.15, 1.0 / 6.0)
    environment._last_camera_timestamps_s = {"uav_a": 0.1, "uav_b": 0.1}

    class SkewedCatchupWorld:
        current_time = 0.2

        def __init__(self) -> None:
            self.render_count = 0

        def render(self) -> None:
            self.render_count += 1
            for sensor in environment.camera_sensors.values():
                sensor.sample = replace(
                    sensor.sample,
                    timestamp_s=min(
                        self.current_time,
                        sensor.sample.timestamp_s + 1.0 / 60.0,
                    ),
                )

    world = SkewedCatchupWorld()
    environment.world = world

    samples = environment._capture_camera_batch(world.current_time)

    assert world.render_count == 3
    assert all(sample.timestamp_s == pytest.approx(0.2) for sample in samples.values())


def test_due_camera_batch_rejects_pipeline_that_never_reaches_world_time() -> None:
    environment, _ = _camera_batch_environment(0.18, 0.18)
    environment._last_camera_timestamps_s = {"uav_a": 0.1, "uav_b": 0.1}

    class StalledWorld:
        current_time = 0.2

        def __init__(self) -> None:
            self.render_count = 0

        def render(self) -> None:
            self.render_count += 1

    world = StalledWorld()
    environment.world = world

    with pytest.raises(RuntimeError, match="outside the current renderer barrier"):
        environment._capture_camera_batch(world.current_time)

    assert world.render_count == environment._MAX_CAMERA_RENDER_CATCHUP_STEPS


def test_camera_batch_warmup_watchdog_fails_instead_of_hanging_forever() -> None:
    environment, _ = _camera_batch_environment(0.0, 0.0)
    environment.camera_sensors = {
        uav_id: _NotReadySampleSensor() for uav_id in environment.uav_ids
    }

    assert environment._capture_camera_batch(0.0) == {}
    assert environment._capture_camera_batch(1.99) == {}
    with pytest.raises(RuntimeError, match="synchronization timed out"):
        environment._capture_camera_batch(2.0)


def test_failed_camera_batch_publishes_neither_snapshot_safety_nor_agent_ticks() -> None:
    environment, _ = _environment()
    environment.tick_uavs(
        {"uav_a": lambda: None, "uav_b": lambda: None},
    )
    prior_snapshot = environment.get_fleet_pose_snapshot()
    assert environment.last_tick_order == ("uav_a", "uav_b")
    callbacks: list[str] = []

    def fail_capture(timestamp_s):
        raise RuntimeError("camera projection failed")

    environment._capture_camera_batch = fail_capture  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="camera projection failed"):
        environment.tick_uavs(
            {
                "uav_a": lambda: callbacks.append("uav_a"),
                "uav_b": lambda: callbacks.append("uav_b"),
            },
            global_safety_check=lambda snapshot: callbacks.append("safety"),
        )

    assert callbacks == []
    assert environment.get_fleet_pose_snapshot().tick_index == prior_snapshot.tick_index
    assert environment.last_tick_order == ()


def test_observation_copies_are_per_uav_and_oracle_is_assignment_scoped() -> None:
    environment, _ = _environment()
    environment.step()

    first = environment.get_agent_observation("uav_a")
    second = environment.get_agent_observation("uav_b")
    assert first["camera_owner"] == "uav_a"
    assert second["camera_owner"] == "uav_b"
    first["pixels"].append("changed")
    assert environment.get_agent_observation("uav_a")["pixels"] == ["uav_a"]

    assert environment.get_evaluator_frame("uav_a", "target_i")["target_id"] == "target_i"
    oracle = environment.make_oracle_perception("uav_a")
    assert oracle.uav_id == "uav_a"
    assert oracle.target_id == "target_i"
    with pytest.raises(PermissionError, match="assigned target"):
        environment.get_evaluator_frame("uav_a", "target_j")


def test_skill_observation_contains_only_requested_uav_and_no_oracle_fields() -> None:
    environment, _ = _environment()
    environment.latest_agent_observations["uav_a"] = SimpleNamespace(
        camera_timestamp_s=1.25,
        uav_state=UAVState(1.0, 2.0, 3.0, 0.4),
        uav_velocity_mps=np.asarray([0.1, 0.2, 0.3]),
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        camera_position_m=np.asarray([1.0, 2.0, 2.5]),
        camera_orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
    )

    observation = environment.get_skill_observation("uav_a")

    observation.validate()
    assert observation.uav_id == "uav_a"
    assert observation.oracle_target_id is None
    assert observation.target_estimate is None
    with pytest.raises(PermissionError, match="assignment-scoped"):
        environment.get_skill_observation("uav_a", include_oracle=True)


def test_exclusive_assignment_and_complete_tick_inventory_are_enforced() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "multi_uav_demo.yaml")
    with pytest.raises(ValueError, match="must not share"):
        FleetUavSearchEnv(
            config,
            assignments={"uav_a": "target_i", "uav_b": "target_i"},
        )
    environment, _ = _environment()
    with pytest.raises(ValueError, match="cover every UAV"):
        environment.tick_uavs({"uav_a": lambda: None})


def test_reset_clears_all_snapshot_camera_and_evaluator_caches() -> None:
    environment, events = _environment()
    environment.step()
    assert environment.latest_agent_observations
    assert environment.latest_evaluator_frames
    assert environment.get_fleet_pose_snapshot().tick_index == 1

    sensors = tuple(environment.camera_sensors.values())
    environment._last_camera_timestamps_s = {"uav_a": 0.1, "uav_b": 0.1}
    environment.reset(target_seeds={"target_i": 11, "target_j": 12})

    assert environment.latest_agent_observations == {}
    assert environment.latest_evaluator_frames == {}
    assert environment._last_camera_timestamps_s == {}
    assert environment._camera_warmup_pending is True
    with pytest.raises(RuntimeError, match="no fleet pose snapshot"):
        environment.get_fleet_pose_snapshot()
    assert environment.last_tick_order == ()
    assert all(sensor.invalidations >= 2 for sensor in sensors)
    assert "reset_world" in events


def test_close_destroys_every_camera_before_clearing_owned_state() -> None:
    environment, events = _environment()
    environment.step()
    sensors = tuple(environment.camera_sensors.values())
    invalidations_before_close = tuple(sensor.invalidations for sensor in sensors)

    environment.close()

    assert "stop_world" in events
    assert all(sensor.destroyed for sensor in sensors)
    assert tuple(sensor.invalidations for sensor in sensors) == invalidations_before_close
    assert environment.world is None
    assert environment.scene is None
    assert environment.uav_controllers == {}
    assert environment.target_motions == {}
    assert environment.camera_sensors == {}
    assert environment.latest_agent_observations == {}
    assert environment.latest_evaluator_frames == {}
    with pytest.raises(RuntimeError, match="no fleet pose snapshot"):
        environment.get_fleet_pose_snapshot()


def test_setup_failure_closes_partially_built_world_and_camera(monkeypatch) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "multi_uav_demo.yaml")
    sensor = _Sensor()

    class FailingWorld(_World):
        instance = None

        def __init__(self, **kwargs) -> None:
            super().__init__([])
            type(self).instance = self

    class FailingScene:
        def __init__(self, world, supplied_config) -> None:
            assert world is FailingWorld.instance
            assert supplied_config is config
            self.camera_sensors = {"uav_a": sensor}

        def build(self) -> None:
            raise RuntimeError("scene build failed")

    isaacsim = ModuleType("isaacsim")
    isaacsim_core = ModuleType("isaacsim.core")
    isaacsim_api = ModuleType("isaacsim.core.api")
    isaacsim_api.World = FailingWorld
    scene_module = ModuleType("env.scene")
    scene_module.UavSearchScene = FailingScene
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "isaacsim.core", isaacsim_core)
    monkeypatch.setitem(sys.modules, "isaacsim.core.api", isaacsim_api)
    monkeypatch.setitem(sys.modules, "env.scene", scene_module)

    environment = FleetUavSearchEnv(config)
    with pytest.raises(RuntimeError, match="scene build failed"):
        environment.setup()

    assert FailingWorld.instance is not None
    assert FailingWorld.instance.events == ["stop_world"]
    assert sensor.destroyed is True
    assert environment.world is None
    assert environment.scene is None
    assert environment.camera_sensors == {}
