from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

from configs.loader import load_config
from configs.schema import CameraConfig
from env.fleet_uav_search_env import FleetUavSearchEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERA_CONFIG = CameraConfig((160, 120), 10, 90.0, None, -45.0)


@pytest.fixture
def sensor_class(monkeypatch):
    """Load the real sensor wrapper without starting Isaac in unit tests."""

    for name in (
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.api",
        "isaacsim.core.api.objects",
        "isaacsim.core.utils",
        "isaacsim.core.utils.rotations",
        "isaacsim.sensors",
        "isaacsim.sensors.camera",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["isaacsim.core.api"].World = object
    sys.modules["isaacsim.core.api.objects"].VisualCuboid = object
    sys.modules["isaacsim.core.utils.rotations"].euler_angles_to_quat = object
    sys.modules["isaacsim.sensors.camera"].Camera = object
    name = "_fleet_camera_cleanup_test_sensor"
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "env/camera_sensor.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.RGBCameraSensor


class _World:
    cleared = False

    def stop(self):
        self.stopped = True

    @classmethod
    def clear_instance(cls):
        cls.cleared = True


class _SharedRenderGraph:
    """Model Isaac's shared-node lifetime after the first product removal."""

    product_removed = False


class _Camera:
    def __init__(self, graph, channels=("rgb", "distance_to_image_plane")):
        self.graph = graph
        self.frame = {"rendering_time": 0, **dict.fromkeys(channels)}
        self.destroyed = False
        self.detach_error = None

    def get_current_frame(self):
        return self.frame

    def detach_annotator(self, name):
        if self.detach_error is not None:
            raise self.detach_error
        if self.graph.product_removed:
            raise TypeError("Invalid NodeObj object in Py_Node in getAttributes")
        self.frame.pop(name)

    def destroy(self):
        # Isaac Camera.destroy detaches remaining frame annotators before
        # deleting the render product. That per-Camera order fails for a Fleet.
        for name in tuple(self.frame):
            if name != "rendering_time":
                self.detach_annotator(name)
        self.graph.product_removed = True
        self.destroyed = True


def _sensor(sensor_class, world, camera, name):
    sensor = sensor_class(world, CAMERA_CONFIG, sensor_name=name)
    sensor.camera = camera
    sensor._depth_enabled = "distance_to_image_plane" in camera.frame
    return sensor


def test_four_cameras_release_shared_annotators_before_render_products(sensor_class):
    environment = FleetUavSearchEnv(
        load_config(PROJECT_ROOT / "configs/multi_uav_demo.yaml")
    )
    world = _World()
    environment.world = world
    graph = _SharedRenderGraph()
    cameras = [_Camera(graph) for _ in range(4)]
    sensors = [
        _sensor(sensor_class, world, camera, f"camera_{index}")
        for index, camera in enumerate(cameras)
    ]
    environment.camera_sensors = dict(zip(("uav_a", "uav_b", "uav_c", "uav_d"), sensors))

    environment.close()
    environment.close()

    assert world.stopped
    assert all(camera.destroyed for camera in cameras)
    assert all(sensor.camera is None for sensor in sensors)
    assert environment.world is None
    assert environment.camera_sensors == {}


def test_camera_cleanup_handles_partial_initialization_and_repeated_calls(sensor_class):
    world = _World()
    sensor = sensor_class(world, CAMERA_CONFIG)
    sensor.detach_annotators()
    sensor.destroy()
    camera = _Camera(_SharedRenderGraph(), channels=("rgb",))
    sensor.camera = camera

    sensor.detach_annotators()
    sensor.detach_annotators()
    sensor.destroy()
    sensor.destroy()

    assert camera.destroyed
    assert camera.frame == {"rendering_time": 0}
    assert sensor.camera is None
    assert not sensor._depth_enabled


def test_cleanup_error_is_reported_after_other_cameras_are_released(sensor_class):
    environment = FleetUavSearchEnv(
        load_config(PROJECT_ROOT / "configs/multi_uav_demo.yaml")
    )
    world = _World()
    _World.cleared = False
    environment.world = world
    graph = _SharedRenderGraph()
    broken = _Camera(graph)
    error = RuntimeError("annotator detach failed")
    broken.detach_error = error
    healthy = _Camera(graph)
    environment.camera_sensors = {
        "uav_a": _sensor(sensor_class, world, broken, "broken"),
        "uav_b": _sensor(sensor_class, world, healthy, "healthy"),
    }

    with pytest.raises(RuntimeError) as caught:
        environment.close()

    assert caught.value is error
    assert healthy.destroyed
    assert environment.world is None
    assert environment.camera_sensors == {}
    assert _World.cleared
