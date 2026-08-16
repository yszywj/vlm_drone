"""Fixed RGB camera mounted below the UAV nose.

This module depends on Isaac Sim and must only be imported after
``SimulationApp`` has been created. Camera poses use the project world-camera
axes: +X forward, +Y left, +Z up, with scalar-first quaternions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from pathlib import Path
from typing import Sequence

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.camera import Camera
from PIL import Image

from configs.loader import CameraConfig


class CameraFrameNotReady(RuntimeError):
    """Raised only while the RGB annotator has not produced a fresh frame."""


@dataclass(frozen=True)
class ImageProjection:
    """Projection result for one or more world points."""

    pixels_uv: np.ndarray
    depth_m: np.ndarray
    visible: np.ndarray


class RGBCameraSensor:
    """Own a single, fixed RGB camera and its visible debug housing."""

    CAMERA_PRIM_PATH = "/World/UAV/Camera"
    HOUSING_PRIM_PATH = "/World/UAV/CameraHousing"

    def __init__(
        self,
        world: World,
        config: CameraConfig,
        *,
        camera_prim_path: str = CAMERA_PRIM_PATH,
        housing_prim_path: str = HOUSING_PRIM_PATH,
    ) -> None:
        self.world = world
        self.config = config
        self.camera_prim_path = camera_prim_path
        self.housing_prim_path = housing_prim_path
        self.camera: Camera | None = None
        self._built = False

    def build(self) -> Camera:
        """Create the housing and Camera prim before the caller resets World."""

        if self._built:
            raise RuntimeError("RGB camera sensor has already been built")

        # Config pitch is relative to the UAV body and negative means down.
        # World-camera axes use +X forward, so a downward pitch is +Ry here.
        pitch_rotation_deg = -self.config.pitch_deg
        orientation_wxyz = euler_angles_to_quat(
            np.asarray([0.0, pitch_rotation_deg, 0.0]),
            degrees=True,
        )
        housing_translation = np.asarray([0.58, 0.0, -0.11], dtype=np.float64)
        self.world.scene.add(
            VisualCuboid(
                prim_path=self.housing_prim_path,
                name="uav_camera_housing",
                translation=housing_translation,
                orientation=orientation_wxyz,
                scale=np.asarray([0.24, 0.32, 0.22]),
                size=1.0,
                color=np.asarray([0.05, 0.85, 0.95]),
            )
        )

        self.camera = self.world.scene.add(
            Camera(
                prim_path=self.camera_prim_path,
                name="uav_rgb_camera",
                resolution=self.config.resolution_wh_px,
                frequency=self.config.frequency_hz,
            )
        )
        pitch_rad = radians(pitch_rotation_deg)
        forward = np.asarray([cos(pitch_rad), 0.0, -sin(pitch_rad)])
        self.camera.set_local_pose(
            translation=housing_translation + 0.15 * forward,
            orientation=orientation_wxyz,
            camera_axes="world",
        )

        if self.config.focal_length_m is None:
            aperture = self.camera.get_horizontal_aperture()
            half_fov_rad = radians(self.config.horizontal_fov_deg) / 2.0
            focal_length = aperture / (2.0 * tan(half_fov_rad))
        else:
            focal_length = self.config.focal_length_m
        self.camera.set_focal_length(focal_length)
        self._built = True
        return self.camera

    def get_rgb(self) -> np.ndarray:
        """Return a copy of the latest frequency-governed RGB frame."""

        image, _ = self.get_sample()
        return image

    def get_sample(self) -> tuple[np.ndarray, float]:
        """Atomically copy the latest RGB frame and its simulation timestamp."""

        frame = self._require_camera().get_current_frame()
        image = frame.get("rgb")
        if image is None or np.asarray(image).size == 0:
            raise CameraFrameNotReady("camera frame is not ready; render simulation steps first")
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] < 3:
            raise RuntimeError(f"unexpected camera frame shape: {array.shape}")
        timestamp = frame.get("rendering_time")
        if timestamp is None or not np.isfinite(float(timestamp)):
            raise RuntimeError("camera rendering timestamp is not available or finite")
        return np.ascontiguousarray(array[:, :, :3]).copy(), float(timestamp)

    def invalidate_frame(self) -> None:
        """Prevent an image from a previous pose/episode from being reused."""

        frame = self._require_camera().get_current_frame()
        if "rgb" in frame:
            frame["rgb"] = None

    def save_rgb(self, path: str | Path, image: np.ndarray | None = None) -> Path:
        """Save the latest RGB frame, appending ``.png`` if no suffix is given."""

        output_path = Path(path).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = self.get_rgb() if image is None else np.asarray(image).copy()
        if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0 or rgb.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        if not np.all(np.isfinite(rgb)):
            raise ValueError("image must contain only finite values")
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.float64)
            if rgb.size and float(np.nanmax(rgb)) <= 1.0:
                rgb *= 255.0
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(rgb).save(output_path)
        return output_path

    def get_rendering_time_s(self) -> float:
        """Return the simulation timestamp associated with the latest RGB frame."""

        _, timestamp = self.get_sample()
        return timestamp

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return camera world position (m) and orientation (wxyz)."""

        position, orientation = self._require_camera().get_world_pose(camera_axes="world")
        return np.asarray(position).copy(), np.asarray(orientation).copy()

    def world_to_image(self, points_xyz_m: Sequence[float] | Sequence[Sequence[float]]) -> ImageProjection:
        """Project world points and mark points that are inside the camera frustum."""

        points = np.asarray(points_xyz_m, dtype=np.float64)
        if points.shape == (3,):
            points = points.reshape(1, 3)
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("points_xyz_m must have shape (3,) or (N, 3) with finite values")

        camera = self._require_camera()
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = np.asarray(camera.get_image_coords_from_world_points(points), dtype=np.float64)
        homogeneous = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
        )
        view_matrix = np.asarray(camera.get_view_matrix_ros(), dtype=np.float64)
        camera_points = (view_matrix @ homogeneous.T).T
        depth = camera_points[:, 2]
        near_m, far_m = camera.get_clipping_range()
        width, height = camera.get_resolution()
        visible = (
            np.all(np.isfinite(pixels), axis=1)
            & np.isfinite(depth)
            & (depth >= near_m)
            & (depth <= far_m)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < height)
        )
        return ImageProjection(pixels_uv=pixels.copy(), depth_m=depth.copy(), visible=visible)

    def destroy(self) -> None:
        """Detach render products and annotators before closing SimulationApp."""

        if self.camera is not None:
            self.camera.destroy()
            self.camera = None

    def _require_camera(self) -> Camera:
        if self.camera is None:
            raise RuntimeError("RGB camera sensor must be built before use")
        return self.camera
