"""Trusted RGB-D candidate geometry, independent of detector internals.

The pinhole equations operate in optical coordinates (+X right, +Y down,
+Z forward).  Isaac camera poses in this project use CAMERA_FLU (+X forward,
+Y left, +Z up), so the fixed conversion is::

    [x_flu, y_flu, z_flu] = [z_optical, -x_optical, -y_optical]

Only then is the camera's world quaternion applied.  No bbox-size distance
guess is used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, floor, isfinite
from numbers import Integral, Real

import numpy as np

from env.camera_types import CameraIntrinsics
from perception.candidate_bank import CandidateSnapshot
from perception.grounding import (
    CandidateResolutionUnavailable,
)
from perception.measurement import TargetMeasurement
from perception.runtime import PerceptionRuntimeProfile
from runtime.frame_store import FrameCameraGeometry, FrameStore


class DepthSamplingStrategy(str, Enum):
    BBOX_CENTER = "bbox_center"
    BBOX_BOTTOM_CENTER = "bbox_bottom_center"
    BBOX_PATCH_MEDIAN = "bbox_patch_median"
    FOREGROUND_CLUSTER_MEDIAN = "foreground_cluster_median"
    MASK_MEDIAN = "mask_median"


@dataclass(frozen=True, slots=True)
class _DepthSample:
    depth_m: float
    u_px: float
    v_px: float
    robust_depth_sigma_m: float
    valid_fraction: float
    cluster_fraction: float
    bbox_width_px: int
    bbox_height_px: int


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
            DepthSamplingStrategy.FOREGROUND_CLUSTER_MEDIAN
        ),
        patch_radius_px: int = 4,
        min_depth_m: float = 0.2,
        max_depth_m: float = 200.0,
        min_valid_samples: int = 3,
        fallback_to_bbox_median: bool = True,
        foreground_inset_ratio: float = 0.1,
        foreground_bottom_exclusion_ratio: float = 0.15,
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
        inset = _finite(foreground_inset_ratio, "foreground_inset_ratio")
        bottom_exclusion = _finite(
            foreground_bottom_exclusion_ratio,
            "foreground_bottom_exclusion_ratio",
        )
        if not 0.0 <= inset < 0.5:
            raise ValueError("foreground_inset_ratio must be within [0, 0.5)")
        if not 0.0 <= bottom_exclusion < 0.5:
            raise ValueError(
                "foreground_bottom_exclusion_ratio must be within [0, 0.5)"
            )
        self._foreground_inset_ratio = inset
        self._foreground_bottom_exclusion_ratio = bottom_exclusion
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
    ) -> TargetMeasurement:
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
        sampled = self._sample_depth_details(depth, bbox)
        optical = backproject_pixel_to_camera_optical(
            u_px=sampled.u_px,
            v_px=sampled.v_px,
            depth_m=sampled.depth_m,
            intrinsics=geometry.intrinsics,
        )
        camera_flu = optical_to_camera_flu(optical)
        world = camera_flu_to_world(camera_flu, geometry)
        quality = self._measurement_quality(
            sampled,
            image_width=geometry.intrinsics.width,
            image_height=geometry.intrinsics.height,
        )
        covariance = self._world_covariance(
            sampled=sampled,
            quality=quality,
            geometry=geometry,
        )
        return TargetMeasurement(
            timestamp_s=frame_ref.timestamp_s,
            candidate_id=candidate.candidate_id,
            tracker_id=candidate.tracker_id_history[-1],
            pixel_uv=(sampled.u_px, sampled.v_px),
            raw_depth_m=sampled.depth_m,
            corrected_depth_m=sampled.depth_m,
            position_camera_flu_m=camera_flu,
            position_world_m=world,
            covariance_world_m2=covariance,
            measurement_quality=quality,
            source=f"{self._source}_{self._sampling_strategy.value}",
        )

    def _sample_depth(
        self,
        depth: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        """Compatibility wrapper returning only depth and sampled pixel."""

        sampled = self._sample_depth_details(depth, bbox)
        return sampled.depth_m, sampled.u_px, sampled.v_px

    def _sample_depth_details(
        self,
        depth: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> _DepthSample:
        if not isinstance(depth, np.ndarray) or depth.ndim != 2:
            raise CandidateResolutionUnavailable(
                "candidate depth image must be a two-dimensional array"
            )
        height, width = depth.shape
        if height <= 0 or width <= 0:
            raise CandidateResolutionUnavailable("candidate depth image is empty")
        x1, y1, x2, y2 = self._bbox_pixels(bbox, width=width, height=height)
        if self._sampling_strategy is DepthSamplingStrategy.FOREGROUND_CLUSTER_MEDIAN:
            return self._sample_foreground_cluster(
                depth,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
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
            values = depth[y1 : y2 + 1, x1 : x2 + 1]
            valid = self._valid_values(values)
        if valid.size < self._min_valid_samples:
            raise CandidateResolutionUnavailable(
                "insufficient_valid_metric_depth_samples"
            )
        median = float(np.median(valid))
        robust_sigma = float(1.4826 * np.median(np.abs(valid - median)))
        return _DepthSample(
            depth_m=median,
            u_px=u,
            v_px=v,
            robust_depth_sigma_m=robust_sigma,
            valid_fraction=float(valid.size / max(values.size, 1)),
            cluster_fraction=1.0,
            bbox_width_px=x2 - x1 + 1,
            bbox_height_px=y2 - y1 + 1,
        )

    def _bbox_pixels(
        self,
        bbox: tuple[float, float, float, float],
        *,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise CandidateResolutionUnavailable(
                "candidate_bbox_must_have_four_normalized_coordinates"
            )
        values = tuple(_finite(value, f"bbox[{index}]") for index, value in enumerate(bbox))
        if any(value < 0.0 or value > 1.0 for value in values):
            raise CandidateResolutionUnavailable("candidate_bbox_out_of_bounds")
        if values[0] >= values[2] or values[1] >= values[3]:
            raise CandidateResolutionUnavailable("candidate_bbox_has_non_positive_area")
        x1 = max(0, min(width - 1, floor(values[0] * width)))
        y1 = max(0, min(height - 1, floor(values[1] * height)))
        x2 = max(x1, min(width - 1, ceil(values[2] * width) - 1))
        y2 = max(y1, min(height - 1, ceil(values[3] * height) - 1))
        return x1, y1, x2, y2

    def _sample_foreground_cluster(
        self,
        depth: np.ndarray,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> _DepthSample:
        bbox_width = x2 - x1 + 1
        bbox_height = y2 - y1 + 1
        inset_x = min(
            max(int(round(bbox_width * self._foreground_inset_ratio)), 0),
            max((bbox_width - 1) // 2, 0),
        )
        inset_y = min(
            max(int(round(bbox_height * self._foreground_inset_ratio)), 0),
            max((bbox_height - 1) // 2, 0),
        )
        ix1, ix2 = x1 + inset_x, x2 - inset_x
        iy1 = y1 + inset_y
        bottom_rows = int(round(bbox_height * self._foreground_bottom_exclusion_ratio))
        iy2 = max(iy1, y2 - inset_y - bottom_rows)
        roi = depth[iy1 : iy2 + 1, ix1 : ix2 + 1]
        valid_mask = (
            np.isfinite(roi)
            & (roi >= self._min_depth_m)
            & (roi <= self._max_depth_m)
        )
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count < self._min_valid_samples:
            raise CandidateResolutionUnavailable(
                "foreground_cluster_insufficient_valid_depth_samples"
            )

        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        seed_radius = min(
            self._patch_radius_px,
            max(1, min(bbox_width, bbox_height) // 8),
        )
        seed_patch = depth[
            max(iy1, center_y - seed_radius) : min(iy2 + 1, center_y + seed_radius + 1),
            max(ix1, center_x - seed_radius) : min(ix2 + 1, center_x + seed_radius + 1),
        ]
        seed_values = self._valid_values(seed_patch)
        center_depth = depth[center_y, center_x]
        if (
            isfinite(float(center_depth))
            and self._min_depth_m <= float(center_depth) <= self._max_depth_m
        ):
            # A valid center ray is the least ambiguous seed.  A patch median
            # can otherwise let a thin foreground object be overwhelmed by
            # eight surrounding background pixels.
            seed_depth = float(center_depth)
            near_center = seed_values[
                np.abs(seed_values - seed_depth)
                <= max(0.15, 0.05 * seed_depth)
            ]
            seed_mad = (
                0.0
                if near_center.size == 0
                else float(np.median(np.abs(near_center - seed_depth)))
            )
        elif seed_values.size:
            seed_depth = float(np.median(seed_values))
            seed_mad = float(
                np.median(np.abs(seed_values - seed_depth))
            )
        else:
            # The center can be an invalid hole.  Select the valid samples
            # nearest the center in image space rather than guessing that the
            # globally nearest depth must be the target.
            valid_y, valid_x = np.nonzero(valid_mask)
            distances = (
                (valid_x + ix1 - center_x) ** 2
                + (valid_y + iy1 - center_y) ** 2
            )
            nearest_count = max(
                self._min_valid_samples,
                min(valid_count, int(ceil(valid_count * 0.1))),
            )
            nearest = np.argpartition(distances, nearest_count - 1)[:nearest_count]
            nearest_depths = roi[valid_y[nearest], valid_x[nearest]]
            seed_depth = float(np.median(nearest_depths))
            seed_mad = float(
                np.median(np.abs(nearest_depths - seed_depth))
            )

        tolerance_m = max(0.15, 0.05 * seed_depth, 3.0 * 1.4826 * seed_mad)
        tolerance_m = min(tolerance_m, max(0.5, 0.15 * seed_depth))
        cluster_mask = valid_mask & (np.abs(roi - seed_depth) <= tolerance_m)
        cluster_y, cluster_x = np.nonzero(cluster_mask)
        cluster_values = roi[cluster_y, cluster_x]
        if cluster_values.size < self._min_valid_samples:
            raise CandidateResolutionUnavailable(
                "foreground_depth_cluster_has_insufficient_support"
            )
        depth_m = float(np.median(cluster_values))
        robust_sigma = float(
            1.4826 * np.median(np.abs(cluster_values - depth_m))
        )
        return _DepthSample(
            depth_m=depth_m,
            u_px=float(np.median(cluster_x + ix1)),
            v_px=float(np.median(cluster_y + iy1)),
            robust_depth_sigma_m=robust_sigma,
            valid_fraction=float(valid_count / max(roi.size, 1)),
            cluster_fraction=float(cluster_values.size / valid_count),
            bbox_width_px=bbox_width,
            bbox_height_px=bbox_height,
        )

    @staticmethod
    def _measurement_quality(
        sampled: _DepthSample,
        *,
        image_width: int,
        image_height: int,
    ) -> float:
        relative_spread = sampled.robust_depth_sigma_m / max(sampled.depth_m, 1e-6)
        stability = 1.0 / (1.0 + 20.0 * relative_spread)
        area_fraction = (
            sampled.bbox_width_px * sampled.bbox_height_px
            / float(image_width * image_height)
        )
        size_score = min(1.0, np.sqrt(area_fraction / 0.05))
        quality = (
            0.30 * sampled.valid_fraction
            + 0.30 * sampled.cluster_fraction
            + 0.25 * stability
            + 0.15 * size_score
        )
        return float(np.clip(quality, 0.0, 1.0))

    @staticmethod
    def _world_covariance(
        *,
        sampled: _DepthSample,
        quality: float,
        geometry: FrameCameraGeometry,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        intrinsics = geometry.intrinsics
        quality_scale = 1.0 / np.sqrt(max(quality, 0.05))
        sigma_depth = max(
            0.02,
            0.005 * sampled.depth_m,
            sampled.robust_depth_sigma_m,
        ) * quality_scale
        sigma_pixel = max(
            0.5,
            0.03 * max(sampled.bbox_width_px, sampled.bbox_height_px),
        ) * quality_scale
        jacobian = np.asarray(
            [
                [
                    sampled.depth_m / intrinsics.fx,
                    0.0,
                    (sampled.u_px - intrinsics.cx) / intrinsics.fx,
                ],
                [
                    0.0,
                    sampled.depth_m / intrinsics.fy,
                    (sampled.v_px - intrinsics.cy) / intrinsics.fy,
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        input_covariance = np.diag(
            [sigma_pixel**2, sigma_pixel**2, sigma_depth**2]
        )
        optical_covariance = jacobian @ input_covariance @ jacobian.T
        optical_to_flu = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        rotation = _quaternion_rotation_matrix(
            geometry.camera_orientation_world_wxyz
        )
        transform = rotation @ optical_to_flu
        world_covariance = transform @ optical_covariance @ transform.T
        world_covariance = (world_covariance + world_covariance.T) / 2.0
        world_covariance += np.eye(3, dtype=np.float64) * 1e-9
        return tuple(
            tuple(float(value) for value in row)
            for row in world_covariance
        )  # type: ignore[return-value]

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
