from __future__ import annotations

from dataclasses import replace
from math import cos, pi, sin
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np

from env.camera_types import CameraIntrinsics, CameraSample
from scripts.collect_yolo_dataset import (
    _SimpleSceneCollectionAdapter,
    _UsdCubeV1SceneDriver,
)
from training.yolo.collection_scene import (
    CUBE_COLORS,
    HARD_NEGATIVE_KINDS,
    CollectionSceneObject,
    build_cube_v1_scene_inventory,
    load_cube_collection_protocol,
    oriented_box_corners_world,
    transformed_local_bounds_corners,
)
from training.yolo.isaac_collector import EpisodeRandomizer, RandomizationBounds


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _protocol():
    return load_cube_collection_protocol(
        PROJECT_ROOT / "configs" / "yolo" / "collect_cube.yaml"
    )


def test_formal_scene_planner_covers_zero_to_three_cubes_colors_and_negatives() -> None:
    protocol = _protocol()
    cases = (
        (0, "negative", 0),
        (0, "positive", 1),
        (3, "positive", 2),
        (6, "positive", 3),
        (1, "partial_occlusion", 3),
    )
    colors: set[str] = set()
    for episode_index, sample_kind, expected_cubes in cases:
        inventory = build_cube_v1_scene_inventory(
            protocol,
            scene_seed=9,
            episode_index=episode_index,
            sample_kind=sample_kind,
            anchor_position_world_m=(1.0, 2.0, 0.8),
            target_scale=1.0,
        )
        cubes = tuple(item for item in inventory if item.shape == "cube")
        colors.update(item.color_name for item in cubes)
        assert len(cubes) == expected_cubes
        assert {
            item.object_id for item in inventory if item.shape != "cube"
        } == set(HARD_NEGATIVE_KINDS)

    for episode_index in (1, 4, 7):
        colors.update(
            item.color_name
            for item in build_cube_v1_scene_inventory(
                protocol,
                scene_seed=9,
                episode_index=episode_index,
                sample_kind="partial_occlusion",
                anchor_position_world_m=(0.0, 0.0, 1.0),
                target_scale=1.0,
            )
            if item.shape == "cube"
        )
    assert colors == set(CUBE_COLORS)


def test_local_usd_bounds_are_transformed_as_oriented_corners_not_world_aabb() -> None:
    yaw = pi / 4.0

    def transform(point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = point
        scaled_x, scaled_y, scaled_z = 2.0 * x, 1.0 * y, 3.0 * z
        return (
            5.0 + cos(yaw) * scaled_x - sin(yaw) * scaled_y,
            -2.0 + sin(yaw) * scaled_x + cos(yaw) * scaled_y,
            1.0 + scaled_z,
        )

    corners, dimensions = transformed_local_bounds_corners(
        (-0.5, -0.5, -0.5),
        (0.5, 0.5, 0.5),
        transform,
    )
    assert np.allclose(dimensions, (2.0, 1.0, 3.0))
    # A rotated world's aligned extent is wider than either local XY side;
    # retaining the eight transformed corners avoids that AABB inflation.
    assert np.ptp(corners, axis=0)[0] > dimensions[0]
    assert np.ptp(corners, axis=0)[1] > dimensions[1]


def test_usd_driver_applies_body_bound_and_world_transform_exactly_once(
    monkeypatch,
) -> None:
    class _Matrix:
        def __init__(self, transform) -> None:
            self._transform = transform

        def Transform(self, point):
            return self._transform(np.asarray(point, dtype=np.float64))

    class _Range:
        def GetMin(self):
            return (-0.5, -0.5, -0.5)

        def GetMax(self):
            return (0.5, 0.5, 0.5)

    class _Bound:
        def GetRange(self):
            return _Range()

        def GetMatrix(self):
            return _Matrix(lambda point: point)

    class _Body:
        def IsValid(self):
            return True

        def IsA(self, _schema):
            return True

    body = _Body()

    class _Stage:
        def GetPrimAtPath(self, path):
            assert path == "/World/CubeV1Collection/cube_0/Body"
            return body

    class _Root:
        def GetStage(self):
            return _Stage()

        def GetPath(self):
            return "/World/CubeV1Collection/cube_0"

    class _Cache:
        requested = None

        def ComputeUntransformedBound(self, requested):
            self.requested = requested
            return _Bound()

    cache = _Cache()
    yaw = pi / 4.0
    rotation = np.asarray(
        [
            [cos(yaw), -sin(yaw), 0.0],
            [sin(yaw), cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    class _XformCache:
        requested = None

        def GetLocalToWorldTransform(self, requested):
            self.requested = requested
            return _Matrix(
                lambda point: rotation @ (point * np.asarray([2.0, 1.0, 3.0]))
                + np.asarray([5.0, -2.0, 1.0])
            )

    xform_cache = _XformCache()
    pxr = ModuleType("pxr")
    pxr.Gf = SimpleNamespace(Vec3d=lambda *values: values)
    pxr.Usd = SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: object()))
    pxr.UsdGeom = SimpleNamespace(
        BBoxCache=lambda *_args, **_kwargs: cache,
        Boundable=object,
        Tokens=SimpleNamespace(default_="default", render="render"),
        XformCache=lambda *_args, **_kwargs: xform_cache,
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    planned = CollectionSceneObject(
        object_id="cube_0",
        shape="cube",
        color_name="red",
        position_world_m=(0.0, 0.0, 0.0),
        orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        dimensions_xyz_m=(1.0, 1.0, 1.0),
    )
    driver = object.__new__(_UsdCubeV1SceneDriver)
    driver._active_ids = {planned.object_id}
    driver._roots = {planned.object_id: _Root()}

    rendered, corners = driver.rendered_geometry(planned)

    assert cache.requested is body
    assert xform_cache.requested is body
    assert np.allclose(rendered.position_world_m, (5.0, -2.0, 1.0))
    assert np.allclose(rendered.dimensions_xyz_m, (2.0, 1.0, 3.0))
    assert corners.shape == (8, 3)


def test_usd_driver_hides_non_protocol_mission_obstacles(monkeypatch) -> None:
    hidden: list[object] = []

    class _Prim:
        def IsValid(self):
            return True

    obstacle_root = _Prim()

    class _Stage:
        def GetPrimAtPath(self, path):
            assert path == "/World/Obstacles"
            return obstacle_root

    class _Imageable:
        def __init__(self, prim):
            self.prim = prim

        def MakeInvisible(self):
            hidden.append(self.prim)

    pxr = ModuleType("pxr")
    pxr.UsdGeom = SimpleNamespace(Imageable=_Imageable)
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    driver = object.__new__(_UsdCubeV1SceneDriver)
    driver._stage = lambda: _Stage()

    driver._hide_mission_obstacles()

    assert hidden == [obstacle_root]


class _FakeMotion:
    def __init__(self) -> None:
        self.resets: list[dict[str, object]] = []

    def reset(self, **kwargs: object) -> None:
        self.resets.append(dict(kwargs))


class _FakeSceneDriver:
    def __init__(self) -> None:
        self.installed: tuple[CollectionSceneObject, ...] = ()
        self.current: dict[str, CollectionSceneObject] = {}
        self.updates: list[str] = []

    def install(self, objects) -> None:
        self.installed = tuple(objects)
        self.current = {item.object_id: item for item in self.installed}

    def update_pose(self, obj: CollectionSceneObject) -> None:
        self.current[obj.object_id] = obj
        self.updates.append(obj.object_id)

    def rendered_geometry(
        self,
        obj: CollectionSceneObject,
    ) -> tuple[CollectionSceneObject, np.ndarray]:
        current = self.current[obj.object_id]
        # Simulate USD reporting geometry different from a planned legacy size.
        actual = replace(
            current,
            dimensions_xyz_m=tuple(
                value * factor
                for value, factor in zip(
                    current.dimensions_xyz_m,
                    (1.0, 1.1, 1.2),
                    strict=True,
                )
            ),
        )
        return actual, oriented_box_corners_world(actual)


class _FakeEnvironment:
    def __init__(self) -> None:
        self.motion = _FakeMotion()
        self.steps = 0
        self.resets: list[int] = []
        self.world = SimpleNamespace(current_time=0.0)

    def reset(self, *, target_seed: int) -> None:
        self.resets.append(target_seed)

    def set_uav_pose(self, *_args) -> None:
        return None

    def set_uav_velocity(self, *_args) -> None:
        return None

    def _require_target_motion(self) -> _FakeMotion:
        return self.motion

    def step(self) -> bool:
        self.steps += 1
        self.world.current_time = self.steps * 0.1
        return True

    def get_camera_sample(self) -> CameraSample:
        return CameraSample(
            timestamp_s=float(self.world.current_time),
            rgb=np.full((64, 96, 3), 90, dtype=np.uint8),
            depth_to_image_plane_m=np.full((64, 96), 5.0, dtype=np.float32),
            camera_position_world_m=(0.0, 0.0, 4.0),
            camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            intrinsics=CameraIntrinsics(
                fx=80.0,
                fy=80.0,
                cx=48.0,
                cy=32.0,
                width=96,
                height=64,
            ),
        )

    def world_to_image(self, _points: np.ndarray):
        pixels = np.asarray(
            (
                (12.0, 12.0),
                (12.0, 28.0),
                (28.0, 12.0),
                (28.0, 28.0),
                (14.0, 14.0),
                (14.0, 26.0),
                (26.0, 14.0),
                (26.0, 26.0),
            ),
            dtype=np.float64,
        )
        return SimpleNamespace(
            pixels_uv=pixels,
            depth_m=np.full(8, 5.0, dtype=np.float64),
        )


class _SequencedCameraEnvironment(_FakeEnvironment):
    """Expose renderer freshness independently from physics/world progress."""

    def __init__(self, fresh_frames: tuple[bool, ...], *, dt_s: float) -> None:
        super().__init__()
        self._fresh_frames = iter(fresh_frames)
        self._dt_s = float(dt_s)
        self._camera_timestamp_s = 0.0
        self.world = SimpleNamespace(current_time=0.0)

    def step(self) -> bool:
        self.steps += 1
        self.world.current_time = self.steps * self._dt_s
        fresh = next(self._fresh_frames)
        if fresh:
            self._camera_timestamp_s = self.world.current_time
        return fresh

    def get_camera_sample(self) -> CameraSample:
        sample = super().get_camera_sample()
        return replace(sample, timestamp_s=self._camera_timestamp_s)


def test_formal_adapter_installs_complete_inventory_and_reports_driver_bounds() -> None:
    environment = _FakeEnvironment()
    driver = _FakeSceneDriver()
    config = SimpleNamespace(
        simulation=SimpleNamespace(physics_dt_s=0.1),
        camera=SimpleNamespace(frequency_hz=10.0),
        target=SimpleNamespace(
            motion=SimpleNamespace(
                region=SimpleNamespace(
                    min_xyz_m=(-20.0, -20.0, 0.2),
                    max_xyz_m=(20.0, 20.0, 5.0),
                )
            )
        ),
    )
    simulation_app = SimpleNamespace(is_running=lambda: True)
    adapter = _SimpleSceneCollectionAdapter(
        environment,
        simulation_app,
        config,
        protocol=_protocol(),
        scene_driver=driver,
    )
    # These two methods are the only Isaac render APIs outside the injected
    # driver; suppress them so the lifecycle can be tested in pure Python.
    adapter._apply_render_randomization = lambda _plan: None
    adapter._set_camera_view = lambda **_kwargs: None
    plan = replace(
        EpisodeRandomizer(RandomizationBounds(), scene_seed=5).plan(1),
        sample_kind="partial_occlusion",
    )

    adapter.begin_episode(plan)
    assert environment.motion.resets == [
        {"seed": plan.key.scene_seed, "mode": "STATIC"}
    ]
    assert len([obj for obj in driver.installed if obj.shape == "cube"]) == 3
    assert {
        obj.object_id for obj in driver.installed if obj.shape != "cube"
    } == set(HARD_NEGATIVE_KINDS)
    partial = next(obj for obj in driver.installed if obj.shape == "partial_noncube")
    assert partial.position_world_m != next(
        obj.position_world_m
        for obj in build_cube_v1_scene_inventory(
            _protocol(),
            scene_seed=plan.key.scene_seed,
            episode_index=1,
            sample_kind="partial_occlusion",
            anchor_position_world_m=plan.target_position_world_m,
            target_scale=plan.target_scale,
        )
        if obj.shape == "partial_noncube"
    )

    planned_dimensions = {
        item.object_id: item.dimensions_xyz_m for item in driver.installed
    }
    adapter.advance_to_next_sample(0.2)
    truth = adapter.capture_oracle_frame("frame_000001")
    assert len(truth.objects) == 10
    assert {obj.object_id for obj in truth.objects} == set(driver.current)
    assert all(
        obj.dimensions_xyz_m != planned_dimensions[obj.object_id]
        for obj in truth.objects
    )
    assert set(driver.updates) == {"cube_0", "cube_1", "cube_2"}
    assert all(
        obj.as_scene_object().detector_class_id == 0
        for obj in truth.objects
        if obj.shape == "cube"
    )
    assert all(
        obj.as_scene_object().detector_class_id is None
        for obj in truth.objects
        if obj.shape != "cube"
    )


def test_adapter_waits_for_fresh_frame_after_last_planned_step_is_stale() -> None:
    """Never pair an earlier camera frame with later world/Oracle geometry."""

    dt_s = 0.1
    # The three planned physics steps contain a fresh frame in the middle, but
    # the final step advances geometry without rendering.  The adapter must do
    # one bounded warm-up step to obtain a frame at the final world barrier.
    environment = _SequencedCameraEnvironment(
        (False, True, False, True),
        dt_s=dt_s,
    )
    driver = _FakeSceneDriver()
    config = SimpleNamespace(
        simulation=SimpleNamespace(physics_dt_s=dt_s),
        camera=SimpleNamespace(frequency_hz=10.0),
        target=SimpleNamespace(
            motion=SimpleNamespace(
                region=SimpleNamespace(
                    min_xyz_m=(-20.0, -20.0, 0.2),
                    max_xyz_m=(20.0, 20.0, 5.0),
                )
            )
        ),
    )
    adapter = _SimpleSceneCollectionAdapter(
        environment,
        SimpleNamespace(is_running=lambda: True),
        config,
        protocol=_protocol(),
        scene_driver=driver,
    )
    adapter._apply_render_randomization = lambda _plan: None
    adapter._set_camera_view = lambda **_kwargs: None
    plan = replace(
        EpisodeRandomizer(RandomizationBounds(), scene_seed=17).plan(0),
        sample_kind="positive",
        target_position_world_m=(0.0, 0.0, 1.0),
        target_heading_deg=0.0,
        target_speed_mps=1.0,
        target_direction_change_interval_s=100.0,
    )

    adapter.begin_episode(plan)
    initial_cube_x = next(
        obj.position_world_m[0] for obj in driver.installed if obj.shape == "cube"
    )
    adapter.advance_to_next_sample(0.25)
    truth = adapter.capture_oracle_frame("frame_000000")
    rendered_cube = next(obj for obj in truth.objects if obj.shape == "cube")

    assert environment.steps == 4
    assert np.isclose(
        truth.camera_sample.timestamp_s,
        environment.world.current_time,
    )
    assert np.isclose(
        rendered_cube.position_world_m[0] - initial_cube_x,
        truth.camera_sample.timestamp_s * plan.target_speed_mps,
    )


def test_target_state_crossing_profile_moves_two_cubes_toward_each_other() -> None:
    environment = _FakeEnvironment()
    driver = _FakeSceneDriver()
    config = SimpleNamespace(
        simulation=SimpleNamespace(physics_dt_s=0.1),
        camera=SimpleNamespace(frequency_hz=10.0),
        target=SimpleNamespace(
            motion=SimpleNamespace(
                region=SimpleNamespace(
                    min_xyz_m=(-20.0, -20.0, 0.2),
                    max_xyz_m=(20.0, 20.0, 5.0),
                )
            )
        ),
    )
    adapter = _SimpleSceneCollectionAdapter(
        environment,
        SimpleNamespace(is_running=lambda: True),
        config,
        protocol=_protocol(),
        scene_driver=driver,
        crossing_trajectories=True,
    )
    adapter._apply_render_randomization = lambda _plan: None
    adapter._set_camera_view = lambda **_kwargs: None
    plan = replace(
        EpisodeRandomizer(RandomizationBounds(), scene_seed=23).plan(1),
        sample_kind="partial_occlusion",
        target_position_world_m=(0.0, 0.0, 1.0),
        target_speed_mps=1.0,
        target_direction_change_interval_s=100.0,
    )
    adapter.begin_episode(plan)
    initial = {
        item.object_id: np.asarray(item.position_world_m)
        for item in driver.installed
        if item.object_id in {"cube_0", "cube_1"}
    }
    initial_distance = float(np.linalg.norm(initial["cube_1"] - initial["cube_0"]))

    adapter.advance_to_next_sample(0.2)
    truth = adapter.capture_oracle_frame("frame_1")
    cubes = {item.object_id: item for item in truth.objects if item.shape == "cube"}
    final_distance = float(
        np.linalg.norm(
            np.asarray(cubes["cube_1"].position_world_m)
            - np.asarray(cubes["cube_0"].position_world_m)
        )
    )

    assert final_distance < initial_distance
    assert np.dot(
        cubes["cube_0"].velocity_world_mps,
        cubes["cube_1"].velocity_world_mps,
    ) < 0.0
