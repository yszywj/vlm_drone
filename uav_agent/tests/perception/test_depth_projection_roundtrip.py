from __future__ import annotations

from math import cos, pi, sin, sqrt

import numpy as np

from env.camera_types import CameraIntrinsics
from perception.depth_geometry import (
    backproject_pixel_to_camera_optical,
    camera_flu_to_world,
    optical_to_camera_flu,
    project_world_to_pixel,
)
from runtime.frame_store import FrameCameraGeometry


INTRINSICS = CameraIntrinsics(
    fx=400.0,
    fy=410.0,
    cx=320.0,
    cy=240.0,
    width=640,
    height=480,
)


def _round_trip(
    point_camera_flu_m: tuple[float, float, float],
    geometry: FrameCameraGeometry,
) -> None:
    expected_world = camera_flu_to_world(point_camera_flu_m, geometry)
    u_px, v_px, depth_m = project_world_to_pixel(
        position_world_m=expected_world,
        geometry=geometry,
    )
    optical = backproject_pixel_to_camera_optical(
        u_px=u_px,
        v_px=v_px,
        depth_m=depth_m,
        intrinsics=geometry.intrinsics,
    )
    actual_world = camera_flu_to_world(
        optical_to_camera_flu(optical),
        geometry,
    )
    np.testing.assert_allclose(actual_world, expected_world, atol=1e-10)


def test_optical_to_camera_flu_axis_contract() -> None:
    # Optical is right/down/forward; CAMERA_FLU is forward/left/up.
    assert optical_to_camera_flu((2.0, 3.0, 7.0)) == (7.0, -2.0, -3.0)


def test_world_projection_round_trip_uses_wxyz_quaternion_order() -> None:
    half = sqrt(0.5)
    geometry = FrameCameraGeometry(
        timestamp_s=1.0,
        intrinsics=INTRINSICS,
        camera_position_world_m=(3.0, -2.0, 5.0),
        # +90 degree world-Z yaw in wxyz order.
        camera_orientation_world_wxyz=(half, 0.0, 0.0, half),
    )
    _round_trip((12.0, -0.8, 0.4), geometry)


def test_world_projection_round_trip_with_fixed_camera_pitch() -> None:
    pitch = -25.0 * pi / 180.0
    geometry = FrameCameraGeometry(
        timestamp_s=2.0,
        intrinsics=INTRINSICS,
        camera_position_world_m=(-4.0, 1.5, 12.0),
        # Camera's fixed pitch about its +Y axis, explicitly wxyz.
        camera_orientation_world_wxyz=(
            cos(pitch / 2.0),
            0.0,
            sin(pitch / 2.0),
            0.0,
        ),
    )
    _round_trip((10.0, 0.5, -0.75), geometry)

