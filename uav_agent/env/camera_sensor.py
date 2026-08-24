"""Fixed RGB camera mounted below the UAV nose.

This module depends on Isaac Sim and must only be imported after
``SimulationApp`` has been created. Camera poses use the project world-camera
axes: +X forward, +Y left, +Z up, with scalar-first quaternions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from numbers import Integral
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.camera import Camera
from PIL import Image

from configs.schema import CameraConfig
from env.camera_types import CameraFrameNotReady, CameraIntrinsics, CameraSample


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
        sensor_name: str = "uav_rgb_camera",
        acquisition_frequency_hz: int | None = None,
    ) -> None:
        self.world = world
        self.config = config
        self.camera_prim_path = camera_prim_path
        self.housing_prim_path = housing_prim_path
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError("sensor_name must be a non-empty string")
        self.sensor_name = re.sub(r"[^A-Za-z0-9_]", "_", sensor_name)
        if acquisition_frequency_hz is None:
            acquisition_frequency_hz = config.frequency_hz
        if (
            isinstance(acquisition_frequency_hz, bool)
            or not isinstance(acquisition_frequency_hz, Integral)
            or int(acquisition_frequency_hz) <= 0
        ):
            raise ValueError("acquisition_frequency_hz must be a positive integer")
        # Fleet Camera render products acquire every renderer frame.  The
        # environment performs one shared software downsample to the public
        # CameraConfig frequency after proving every UAV has the same render ID.
        self.acquisition_frequency_hz = int(acquisition_frequency_hz)
        self.camera: Camera | None = None
        self._built = False
        self._depth_enabled = False

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
                name=f"{self.sensor_name}_housing",
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
                name=self.sensor_name,
                resolution=self.config.resolution_wh_px,
                frequency=self.acquisition_frequency_hz,
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

    def enable_depth(self) -> None:
        """Attach Isaac Sim's image-plane Z-depth annotator after Camera init.

        ``World.reset()`` initializes scene sensors and creates the render
        product.  Isaac Sim 5.1 therefore requires this method to be called
        after that reset, rather than while :meth:`build` is constructing the
        prim.  Repeated calls are intentionally harmless for coordinated
        environment resets.
        """

        if self._depth_enabled:
            return
        camera = self._require_camera()
        camera.add_distance_to_image_plane_to_frame()
        self._depth_enabled = True

    def synchronize_acquisition_clock(self) -> None:
        """Reset this Camera's cadence after every Fleet annotator is attached."""

        if not self._depth_enabled:
            raise RuntimeError("enable_depth() must precede Camera clock synchronization")
        self._require_camera().post_reset()

    def get_rgb(self) -> np.ndarray:
        """Return a copy of the latest frequency-governed RGB frame."""

        frame = self._snapshot_frame()
        return self._rgb_from_frame(frame)

    def get_sample(self) -> CameraSample:
        """Return one atomic RGB-D, pose, calibration, and timestamp sample.

        The RGB and ``distance_to_image_plane`` arrays are copied from the
        same cloned Isaac frame.  Missing depth is treated as annotator warmup,
        never as an all-zero range image.
        """

        if not self._depth_enabled:
            raise CameraFrameNotReady(
                "distance_to_image_plane annotator is not enabled; reset the World "
                "and call enable_depth()"
            )
        frame = self._snapshot_frame()
        rgb = self._rgb_from_frame(frame)
        timestamp_s = self._timestamp_from_frame(frame)
        depth_value = frame.get("distance_to_image_plane")
        if depth_value is None or np.asarray(depth_value).size == 0:
            raise CameraFrameNotReady(
                "camera depth frame is not ready; render simulation steps first"
            )
        depth = np.asarray(depth_value)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        expected_depth_shape = rgb.shape[:2]
        if depth.shape != expected_depth_shape:
            raise RuntimeError(
                "RGB/depth resolution mismatch: "
                f"rgb={expected_depth_shape}, depth={depth.shape}"
            )
        if not np.issubdtype(depth.dtype, np.number):
            raise RuntimeError("camera depth frame must contain numeric values")
        depth = np.ascontiguousarray(depth, dtype=np.float32).copy()
        near_m, far_m = self._require_camera().get_clipping_range()
        invalid = (
            ~np.isfinite(depth)
            | (depth <= 0.0)
            | (depth < float(near_m))
            | (depth > float(far_m))
        )
        depth[invalid] = np.nan

        position, orientation = self.get_camera_pose()
        intrinsics = self.get_intrinsics()
        return CameraSample(
            timestamp_s=timestamp_s,
            rgb=rgb,
            depth_to_image_plane_m=depth,
            camera_position_world_m=tuple(float(value) for value in position),
            camera_orientation_world_wxyz=tuple(float(value) for value in orientation),
            intrinsics=intrinsics,
            render_frame_id=self._render_frame_id_from_frame(frame),
        )

    def invalidate_frame(self) -> None:
        """Prevent an image from a previous pose/episode from being reused."""

        frame = self._require_camera().get_current_frame()
        for channel in ("rgb", "distance_to_image_plane"):
            if channel in frame:
                frame[channel] = None

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

        return self.get_render_metadata()[0]

    def get_render_metadata(self) -> tuple[float, tuple[int, int] | None]:
        """Peek the latest renderer clock without copying RGB/depth arrays."""

        frame = self._require_camera().get_current_frame()
        return (
            self._timestamp_from_frame(frame),
            self._render_frame_id_from_frame(frame),
        )

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return calibrated pinhole intrinsics for the configured resolution."""

        camera = self._require_camera()
        matrix = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"unexpected Camera intrinsics matrix: {matrix!r}")
        width, height = camera.get_resolution()
        return CameraIntrinsics(
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
            width=int(width),
            height=int(height),
        )

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
        self._depth_enabled = False

    def _snapshot_frame(self) -> dict[str, object]:
        """Clone the mutable Isaac frame dictionary for a coherent read."""

        return self._require_camera().get_current_frame(clone=True)

    def _rgb_from_frame(self, frame: dict[str, object]) -> np.ndarray:
        image = frame.get("rgb")
        if image is None or np.asarray(image).size == 0:
            raise CameraFrameNotReady(
                "camera RGB frame is not ready; render simulation steps first"
            )
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] < 3:
            raise RuntimeError(f"unexpected camera RGB frame shape: {array.shape}")
        width, height = self._require_camera().get_resolution()
        expected_shape = (int(height), int(width), 3)
        rgb = np.ascontiguousarray(array[:, :, :3])
        if rgb.shape != expected_shape:
            raise RuntimeError(
                f"camera RGB resolution mismatch: expected {expected_shape}, got {rgb.shape}"
            )
        if not np.all(np.isfinite(rgb)):
            raise RuntimeError("camera RGB frame contains non-finite values")
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.float64)
            if rgb.size and float(np.max(rgb)) <= 1.0:
                rgb *= 255.0
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        return rgb.copy()

    @staticmethod
    def _timestamp_from_frame(frame: dict[str, object]) -> float:
        timestamp = frame.get("rendering_time")
        if timestamp is None or not np.isfinite(float(timestamp)):
            raise CameraFrameNotReady("camera rendering timestamp is not ready")
        return float(timestamp)

    @staticmethod
    def _render_frame_id_from_frame(
        frame: Mapping[str, object],
    ) -> tuple[int, int] | None:
        reference_time = frame.get("rendering_frame")
        if not isinstance(reference_time, Mapping):
            return None
        numerator = reference_time.get("referenceTimeNumerator")
        denominator = reference_time.get("referenceTimeDenominator")
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, Integral)
            or isinstance(denominator, bool)
            or not isinstance(denominator, Integral)
        ):
            raise CameraFrameNotReady("camera renderer frame ID is not ready")
        normalized = (int(numerator), int(denominator))
        if normalized[0] < 0 or normalized[1] <= 0:
            raise CameraFrameNotReady("camera renderer frame ID is invalid")
        return normalized

    def _require_camera(self) -> Camera:
        if self.camera is None:
            raise RuntimeError("RGB camera sensor must be built before use")
        return self.camera
