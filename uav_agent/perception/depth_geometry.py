"""Trusted RGB-D candidate geometry, independent of detector internals.

The pinhole equations operate in optical coordinates (+X right, +Y down,
+Z forward).  Isaac camera poses in this project use CAMERA_FLU (+X forward,
+Y left, +Z up), so the fixed conversion is::

    [x_flu, y_flu, z_flu] = [z_optical, -x_optical, -y_optical]

Only then is the camera's world quaternion applied.  No bbox-size distance
guess is used anywhere in this module.
"""

from __future__ import annotations

from enum import Enum
from math import ceil, floor, isfinite
from numbers import Integral, Real

import numpy as np

from env.camera_types import CameraIntrinsics
from perception.candidate_bank import CandidateSnapshot
from perception.grounding import (
    CandidateResolutionUnavailable,
    ResolvedCandidatePosition,
)
from perception.runtime import PerceptionRuntimeProfile
from runtime.frame_store import FrameCameraGeometry, FrameStore


class DepthSamplingStrategy(str, Enum):
    BBOX_CENTER = "bbox_center"
    BBOX_BOTTOM_CENTER = "bbox_bottom_center"
    BBOX_PATCH_MEDIAN = "bbox_patch_median"
    MASK_MEDIAN = "mask_median"


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return normalized


def _strategy(value: object) -> DepthSamplingStrategy:
    if isinstance(value, DepthSamplingStrategy):
        return value
    if isinstance(value, str):
        try:
            return DepthSamplingStrategy(value.strip().lower())
        except ValueError:
            pass
    choices = ", ".join(item.value for item in DepthSamplingStrategy)
    raise ValueError(f"sampling_strategy must be one of: {choices}")


def backproject_pixel_to_camera_optical(
    *,
    u_px: float,
    v_px: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    """Backproject one image-plane Z-depth sample into optical coordinates."""

    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("intrinsics must be CameraIntrinsics")
    u = _finite(u_px, "u_px")
    v = _finite(v_px, "v_px")
    depth = _finite(depth_m, "depth_m")
    if depth <= 0.0:
        raise ValueError("depth_m must be greater than zero")
    if not 0.0 <= u < intrinsics.width or not 0.0 <= v < intrinsics.height:
        raise ValueError("pixel must lie inside the calibrated image")
    return (
        (u - intrinsics.cx) * depth / intrinsics.fx,
        (v - intrinsics.cy) * depth / intrinsics.fy,
        depth,
    )


def optical_to_camera_flu(
    point_optical_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert optical right/down/forward into project forward/left/up."""

    if not isinstance(point_optical_m, tuple) or len(point_optical_m) != 3:
        raise TypeError("point_optical_m must be a three-number tuple")
    x_optical, y_optical, z_optical = (
        _finite(value, f"point_optical_m[{index}]")
        for index, value in enumerate(point_optical_m)
    )
    return z_optical, -x_optical, -y_optical


def _quaternion_rotation_matrix(
    quaternion_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    if not isinstance(quaternion_wxyz, tuple) or len(quaternion_wxyz) != 4:
        raise TypeError("quaternion_wxyz must be a four-number tuple")
    q = np.asarray(
        [
            _finite(value, f"quaternion_wxyz[{index}]")
            for index, value in enumerate(quaternion_wxyz)
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion_wxyz must have non-zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_flu_to_world(
    point_camera_flu_m: tuple[float, float, float],
    geometry: FrameCameraGeometry,
) -> tuple[float, float, float]:
    """Transform one CAMERA_FLU point by the synchronized camera pose."""

    if not isinstance(geometry, FrameCameraGeometry):
        raise TypeError("geometry must be FrameCameraGeometry")
    if not isinstance(point_camera_flu_m, tuple) or len(point_camera_flu_m) != 3:
        raise TypeError("point_camera_flu_m must be a three-number tuple")
    point = np.asarray(
        [
            _finite(value, f"point_camera_flu_m[{index}]")
            for index, value in enumerate(point_camera_flu_m)
        ],
        dtype=np.float64,
    )
    rotation = _quaternion_rotation_matrix(
        geometry.camera_orientation_world_wxyz
    )
    world = rotation @ point + np.asarray(
        geometry.camera_position_world_m,
        dtype=np.float64,
    )
    return float(world[0]), float(world[1]), float(world[2])


def project_world_to_pixel(
    *,
    position_world_m: tuple[float, float, float],
    geometry: FrameCameraGeometry,
) -> tuple[float, float, float]:
    """Project a world point to ``(u, v, optical_z)`` for round-trip tests."""

    if not isinstance(position_world_m, tuple) or len(position_world_m) != 3:
        raise TypeError("position_world_m must be a three-number tuple")
    world = np.asarray(
        [
            _finite(value, f"position_world_m[{index}]")
            for index, value in enumerate(position_world_m)
        ],
        dtype=np.float64,
    )
    rotation = _quaternion_rotation_matrix(
        geometry.camera_orientation_world_wxyz
    )
    camera_flu = rotation.T @ (
        world - np.asarray(geometry.camera_position_world_m, dtype=np.float64)
    )
    # FLU -> optical: right=-left, down=-up, forward=forward.
    x_optical = -camera_flu[1]
    y_optical = -camera_flu[2]
    z_optical = camera_flu[0]
    if z_optical <= 0.0:
        raise CandidateResolutionUnavailable("world point is behind the camera")
    intrinsics = geometry.intrinsics
    u = intrinsics.fx * x_optical / z_optical + intrinsics.cx
    v = intrinsics.fy * y_optical / z_optical + intrinsics.cy
    return float(u), float(v), float(z_optical)


class DepthCandidateResolver:
    """Resolve a candidate bbox through synchronized metric Z-depth."""

    def __init__(
        self,
        frame_store: FrameStore,
        *,
        sampling_strategy: DepthSamplingStrategy | str = (
            DepthSamplingStrategy.BBOX_BOTTOM_CENTER
        ),
        patch_radius_px: int = 4,
        min_depth_m: float = 0.2,
        max_depth_m: float = 200.0,
        min_valid_samples: int = 3,
        fallback_to_bbox_median: bool = True,
        source: str = "isaac_depth",
    ) -> None:
        if not isinstance(frame_store, FrameStore):
            raise TypeError("frame_store must be a FrameStore")
        self._frame_store = frame_store
        self._sampling_strategy = _strategy(sampling_strategy)
        self._patch_radius_px = _positive_integer(
            patch_radius_px,
            "patch_radius_px",
        )
        minimum = _finite(min_depth_m, "min_depth_m")
        maximum = _finite(max_depth_m, "max_depth_m")
        if minimum <= 0.0 or maximum <= minimum:
            raise ValueError("depth limits must satisfy 0 < min_depth_m < max_depth_m")
        self._min_depth_m = minimum
        self._max_depth_m = maximum
        self._min_valid_samples = _positive_integer(
            min_valid_samples,
            "min_valid_samples",
        )
        if not isinstance(fallback_to_bbox_median, bool):
            raise TypeError("fallback_to_bbox_median must be a bool")
        self._fallback_to_bbox_median = fallback_to_bbox_median
        if not isinstance(source, str) or not source or source != source.strip():
            raise ValueError("source must be non-empty without surrounding whitespace")
        self._source = source

    @property
    def profile(self) -> PerceptionRuntimeProfile:
        return PerceptionRuntimeProfile.PRODUCTION

    @property
    def sampling_strategy(self) -> DepthSamplingStrategy:
        return self._sampling_strategy

    def resolve(
        self,
        candidate: CandidateSnapshot,
        *,
        timestamp_s: float,
    ) -> ResolvedCandidatePosition:
        if not isinstance(candidate, CandidateSnapshot):
            raise TypeError("candidate must be a CandidateSnapshot")
        timestamp = _finite(timestamp_s, "timestamp_s")
        if timestamp < candidate.last_seen_timestamp_s:
            raise ValueError("timestamp_s cannot predate the candidate observation")
        if self._sampling_strategy is DepthSamplingStrategy.MASK_MEDIAN:
            raise CandidateResolutionUnavailable(
                "mask_median requires a segmentation mask not present in CandidateSnapshot"
            )

        frame_ref = candidate.frame_history[-1]
        depth = self._frame_store.get_depth(frame_ref, copy=False)
        geometry = self._frame_store.get_camera_geometry(frame_ref)
        if depth is None or geometry is None:
            raise CandidateResolutionUnavailable(
                "candidate frame has no synchronized depth and camera geometry"
            )
        bbox = candidate.bbox_history[-1]
        depth_m, u_px, v_px = self._sample_depth(depth, bbox)
        optical = backproject_pixel_to_camera_optical(
            u_px=u_px,
            v_px=v_px,
            depth_m=depth_m,
            intrinsics=geometry.intrinsics,
        )
        world = camera_flu_to_world(optical_to_camera_flu(optical), geometry)
        return ResolvedCandidatePosition(
            uav_id=candidate.uav_id,
            candidate_id=candidate.candidate_id,
            position_xyz_m=world,
            source=f"{self._source}_{self._sampling_strategy.value}",
        )

    def _sample_depth(
        self,
        depth: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        height, width = depth.shape
        x1 = max(0, min(width - 1, floor(bbox[0] * width)))
        y1 = max(0, min(height - 1, floor(bbox[1] * height)))
        x2 = max(x1, min(width - 1, ceil(bbox[2] * width) - 1))
        y2 = max(y1, min(height - 1, ceil(bbox[3] * height) - 1))
        u = (x1 + x2) / 2.0
        v = (
            float(y2)
            if self._sampling_strategy is DepthSamplingStrategy.BBOX_BOTTOM_CENTER
            else (y1 + y2) / 2.0
        )

        if self._sampling_strategy is DepthSamplingStrategy.BBOX_PATCH_MEDIAN:
            values = depth[y1 : y2 + 1, x1 : x2 + 1]
        else:
            center_x = int(round(u))
            center_y = int(round(v))
            radius = self._patch_radius_px
            values = depth[
                max(0, center_y - radius) : min(height, center_y + radius + 1),
                max(0, center_x - radius) : min(width, center_x + radius + 1),
            ]
        valid = self._valid_values(values)
        if (
            valid.size < self._min_valid_samples
            and self._fallback_to_bbox_median
            and self._sampling_strategy is not DepthSamplingStrategy.BBOX_PATCH_MEDIAN
        ):
            valid = self._valid_values(depth[y1 : y2 + 1, x1 : x2 + 1])
        if valid.size < self._min_valid_samples:
            raise CandidateResolutionUnavailable(
                "candidate bbox has insufficient valid metric depth samples"
            )
        return float(np.median(valid)), u, v

    def _valid_values(self, values: np.ndarray) -> np.ndarray:
        return values[
            np.isfinite(values)
            & (values >= self._min_depth_m)
            & (values <= self._max_depth_m)
        ]


__all__ = [
    "DepthCandidateResolver",
    "DepthSamplingStrategy",
    "backproject_pixel_to_camera_optical",
    "camera_flu_to_world",
    "optical_to_camera_flu",
    "project_world_to_pixel",
]
