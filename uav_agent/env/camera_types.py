"""Simulator-independent synchronized RGB-D camera value types."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np


class CameraFrameNotReady(RuntimeError):
    """Raised while a required synchronized Camera annotator is not ready."""


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return normalized


def _vector(value: object, field_name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise TypeError(f"{field_name} must be a {length}-number tuple")
    return tuple(
        _finite(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics for one fixed-resolution rendered frame."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        fx = _finite(self.fx, "fx")
        fy = _finite(self.fy, "fy")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("fx and fy must be greater than zero")
        width = _positive_int(self.width, "width")
        height = _positive_int(self.height, "height")
        cx = _finite(self.cx, "cx")
        cy = _finite(self.cy, "cy")
        if not 0.0 <= cx < width or not 0.0 <= cy < height:
            raise ValueError("principal point must lie inside the image")
        object.__setattr__(self, "fx", fx)
        object.__setattr__(self, "fy", fy)
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class CameraSample:
    """One atomic RGB, depth, pose, and calibration observation.

    RGB and depth are copied into private, contiguous, read-only arrays.
    Non-positive and non-finite depth values are represented as ``NaN``;
    callers apply their configured near/far depth limits when consuming it.
    The camera quaternion maps the project's camera FLU axes (+X forward,
    +Y left, +Z up) into world axes.
    """

    timestamp_s: float
    rgb: np.ndarray
    depth_to_image_plane_m: np.ndarray | None
    camera_position_world_m: tuple[float, float, float]
    camera_orientation_world_wxyz: tuple[float, float, float, float]
    intrinsics: CameraIntrinsics
    render_frame_id: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        timestamp = _finite(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics must be CameraIntrinsics")
        if not isinstance(self.rgb, np.ndarray):
            raise TypeError("rgb must be a numpy.ndarray")
        if self.rgb.dtype != np.uint8:
            raise TypeError("rgb must have dtype uint8")
        expected_shape = (self.intrinsics.height, self.intrinsics.width, 3)
        if self.rgb.shape != expected_shape:
            raise ValueError(f"rgb must have shape {expected_shape}")
        rgb = np.ascontiguousarray(self.rgb).copy()
        rgb.setflags(write=False)

        depth: np.ndarray | None = None
        if self.depth_to_image_plane_m is not None:
            if not isinstance(self.depth_to_image_plane_m, np.ndarray):
                raise TypeError("depth_to_image_plane_m must be a numpy.ndarray or None")
            if self.depth_to_image_plane_m.shape != expected_shape[:2]:
                raise ValueError(
                    "depth_to_image_plane_m resolution must match the RGB image"
                )
            if not np.issubdtype(self.depth_to_image_plane_m.dtype, np.number):
                raise TypeError("depth_to_image_plane_m must contain numeric values")
            depth = np.ascontiguousarray(
                self.depth_to_image_plane_m,
                dtype=np.float32,
            ).copy()
            invalid = ~np.isfinite(depth) | (depth <= 0.0)
            depth[invalid] = np.nan
            depth.setflags(write=False)

        position = _vector(
            self.camera_position_world_m,
            "camera_position_world_m",
            3,
        )
        quaternion = _vector(
            self.camera_orientation_world_wxyz,
            "camera_orientation_world_wxyz",
            4,
        )
        norm = sum(component * component for component in quaternion) ** 0.5
        if norm <= 1e-12:
            raise ValueError("camera_orientation_world_wxyz must have non-zero norm")
        quaternion = tuple(component / norm for component in quaternion)

        render_frame_id: tuple[int, int] | None = None
        if self.render_frame_id is not None:
            if (
                not isinstance(self.render_frame_id, tuple)
                or len(self.render_frame_id) != 2
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, Integral)
                    for component in self.render_frame_id
                )
            ):
                raise TypeError(
                    "render_frame_id must be an (integer numerator, integer denominator) tuple or None"
                )
            numerator, denominator = (
                int(self.render_frame_id[0]),
                int(self.render_frame_id[1]),
            )
            if numerator < 0 or denominator <= 0:
                raise ValueError(
                    "render_frame_id numerator must be non-negative and denominator positive"
                )
            render_frame_id = (numerator, denominator)

        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_to_image_plane_m", depth)
        object.__setattr__(self, "camera_position_world_m", position)
        object.__setattr__(self, "camera_orientation_world_wxyz", quaternion)
        object.__setattr__(self, "render_frame_id", render_frame_id)

    def valid_depth_mask(
        self,
        *,
        min_depth_m: float = 0.0,
        max_depth_m: float = float("inf"),
    ) -> np.ndarray | None:
        """Return a fresh validity mask under explicit metric depth limits."""

        if self.depth_to_image_plane_m is None:
            return None
        minimum = _finite(min_depth_m, "min_depth_m")
        if minimum < 0.0:
            raise ValueError("min_depth_m must be non-negative")
        if isinstance(max_depth_m, bool) or not isinstance(max_depth_m, Real):
            raise TypeError("max_depth_m must be a number")
        maximum = float(max_depth_m)
        if maximum <= minimum or (not isfinite(maximum) and maximum != float("inf")):
            raise ValueError("max_depth_m must be greater than min_depth_m")
        return (
            np.isfinite(self.depth_to_image_plane_m)
            & (self.depth_to_image_plane_m >= minimum)
            & (self.depth_to_image_plane_m <= maximum)
        )


__all__ = ["CameraFrameNotReady", "CameraIntrinsics", "CameraSample"]
