"""Simulator-neutral interfaces for privileged Isaac YOLO data collection.

This module deliberately does not import :mod:`isaacsim`, ``omni`` or ``pxr``.
The executable creates ``SimulationApp`` first and supplies an adapter that
implements :class:`IsaacCollectionAdapter`.  Oracle access is label-only and
requires two independent acknowledgements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image
import yaml

from env.camera_types import CameraSample


class IsaacDatasetCollectionError(RuntimeError):
    """Raised when privileged collection would be unsafe or inconsistent."""


def require_oracle_label_acknowledgements(
    *,
    oracle_label_generation: bool,
    acknowledge_privileged_oracle: bool,
) -> None:
    """Require the dedicated purpose flag and the privileged-data warning."""

    if not isinstance(oracle_label_generation, bool):
        raise TypeError("oracle_label_generation must be a bool")
    if not isinstance(acknowledge_privileged_oracle, bool):
        raise TypeError("acknowledge_privileged_oracle must be a bool")
    if not oracle_label_generation or not acknowledge_privileged_oracle:
        raise IsaacDatasetCollectionError(
            "Isaac dataset collection is privileged label generation only; "
            "both --oracle-label-generation and "
            "--acknowledge-privileged-oracle are required"
        )


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _bounded_pair(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{field_name} must be a (minimum, maximum) tuple")
    low = _finite(value[0], f"{field_name}[0]")
    high = _finite(value[1], f"{field_name}[1]")
    if low > high:
        raise ValueError(f"{field_name} minimum cannot exceed maximum")
    if minimum is not None and low < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return low, high


@dataclass(frozen=True, slots=True)
class CollectionLimits:
    """Hard storage and sampling bounds suitable for a shared server."""

    max_samples: int = 2_000
    max_episodes: int = 100
    frames_per_episode: int = 20
    sample_hz: float = 2.0
    min_bbox_area_px: float = 16.0
    jpeg_quality: int = 92
    train_fraction: float = 0.70
    val_fraction: float = 0.20

    def __post_init__(self) -> None:
        integer_limits = {
            "max_samples": (self.max_samples, 1, 100_000),
            "max_episodes": (self.max_episodes, 1, 10_000),
            "frames_per_episode": (self.frames_per_episode, 1, 10_000),
            "jpeg_quality": (self.jpeg_quality, 50, 100),
        }
        for field_name, (value, low, high) in integer_limits.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if not low <= value <= high:
                raise ValueError(f"{field_name} must be in [{low}, {high}]")
        sample_hz = _finite(self.sample_hz, "sample_hz")
        if not 0.0 < sample_hz <= 30.0:
            raise ValueError("sample_hz must be in (0, 30]")
        min_area = _finite(self.min_bbox_area_px, "min_bbox_area_px")
        if min_area < 0.0:
            raise ValueError("min_bbox_area_px must be non-negative")
        train = _finite(self.train_fraction, "train_fraction")
        val = _finite(self.val_fraction, "val_fraction")
        if train <= 0.0 or val <= 0.0 or train + val >= 1.0:
            raise ValueError(
                "train_fraction and val_fraction must be positive and leave a test split"
            )

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.val_fraction


@dataclass(frozen=True, slots=True)
class RandomizationBounds:
    """Bounded domain-randomization ranges passed to the Isaac adapter."""

    uav_x_m: tuple[float, float] = (-12.0, 12.0)
    uav_y_m: tuple[float, float] = (-12.0, 12.0)
    uav_altitude_m: tuple[float, float] = (4.0, 16.0)
    uav_speed_mps: tuple[float, float] = (0.0, 3.0)
    camera_pitch_deg: tuple[float, float] = (-50.0, -15.0)
    camera_horizontal_fov_deg: tuple[float, float] = (55.0, 85.0)
    target_x_m: tuple[float, float] = (-10.0, 10.0)
    target_y_m: tuple[float, float] = (-10.0, 10.0)
    target_altitude_m: tuple[float, float] = (0.5, 2.0)
    target_speed_mps: tuple[float, float] = (0.2, 2.5)
    target_scale: tuple[float, float] = (0.65, 1.45)
    target_camera_distance_m: tuple[float, float] = (3.0, 30.0)
    light_intensity_scale: tuple[float, float] = (0.45, 1.65)
    motion_blur_strength: tuple[float, float] = (0.0, 0.35)
    max_target_turn_rate_deg_s: tuple[float, float] = (8.0, 45.0)
    direction_change_interval_s: tuple[float, float] = (1.5, 6.0)
    negative_fraction: float = 0.20
    partial_occlusion_fraction: float = 0.30

    def __post_init__(self) -> None:
        nonnegative = {
            "uav_altitude_m",
            "uav_speed_mps",
            "target_altitude_m",
            "target_speed_mps",
            "target_scale",
            "target_camera_distance_m",
            "light_intensity_scale",
            "motion_blur_strength",
            "max_target_turn_rate_deg_s",
            "direction_change_interval_s",
        }
        for field_name in (
            "uav_x_m",
            "uav_y_m",
            "uav_altitude_m",
            "uav_speed_mps",
            "camera_pitch_deg",
            "camera_horizontal_fov_deg",
            "target_x_m",
            "target_y_m",
            "target_altitude_m",
            "target_speed_mps",
            "target_scale",
            "target_camera_distance_m",
            "light_intensity_scale",
            "motion_blur_strength",
            "max_target_turn_rate_deg_s",
            "direction_change_interval_s",
        ):
            normalized = _bounded_pair(
                getattr(self, field_name),
                field_name,
                minimum=0.0 if field_name in nonnegative else None,
            )
            object.__setattr__(self, field_name, normalized)
        if self.camera_horizontal_fov_deg[0] <= 0.0 or self.camera_horizontal_fov_deg[1] >= 180.0:
            raise ValueError("camera_horizontal_fov_deg must stay inside (0, 180)")
        for field_name in ("negative_fraction", "partial_occlusion_fraction"):
            value = _finite(getattr(self, field_name), field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.negative_fraction + self.partial_occlusion_fraction > 1.0:
            raise ValueError("negative and partial-occlusion fractions cannot exceed one")


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    scene_seed: int
    episode_id: str
    trajectory_id: str

    def __post_init__(self) -> None:
        if isinstance(self.scene_seed, bool) or not isinstance(self.scene_seed, int):
            raise TypeError("scene_seed must be an integer")
        if self.scene_seed < 0:
            raise ValueError("scene_seed must be non-negative")
        for field_name in ("episode_id", "trajectory_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


def split_for_episode(key: EpisodeKey, limits: CollectionLimits) -> str:
    """Assign a complete scene/episode/trajectory group to one split."""

    if not isinstance(key, EpisodeKey):
        raise TypeError("key must be an EpisodeKey")
    if not isinstance(limits, CollectionLimits):
        raise TypeError("limits must be CollectionLimits")
    payload = f"{key.scene_seed}:{key.episode_id}:{key.trajectory_id}".encode("utf-8")
    bucket = int.from_bytes(sha256(payload).digest()[:8], "big") / float(2**64)
    if bucket < limits.train_fraction:
        return "train"
    if bucket < limits.train_fraction + limits.val_fraction:
        return "val"
    return "test"


@dataclass(frozen=True, slots=True)
class EpisodeRandomization:
    """One reproducible, bounded episode request for an Isaac adapter."""

    key: EpisodeKey
    uav_position_world_m: tuple[float, float, float]
    uav_speed_mps: float
    uav_yaw_deg: float
    camera_pitch_deg: float
    camera_horizontal_fov_deg: float
    target_position_world_m: tuple[float, float, float]
    target_speed_mps: float
    target_heading_deg: float
    target_scale: float
    target_camera_distance_m: float
    target_max_turn_rate_deg_s: float
    target_direction_change_interval_s: float
    background_variant: int
    material_variant: int
    light_intensity_scale: float
    motion_blur_strength: float
    sample_kind: str

    def __post_init__(self) -> None:
        if self.sample_kind not in {"positive", "partial_occlusion", "negative"}:
            raise ValueError("sample_kind is invalid")
        if self.target_direction_change_interval_s <= 0.0:
            raise ValueError("target direction changes must be interval-bounded")
        if self.target_max_turn_rate_deg_s <= 0.0:
            raise ValueError("target motion must have a finite positive turn-rate limit")

    def to_manifest_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["key"] = asdict(self.key)
        return value


class EpisodeRandomizer:
    """Create plans that include positives, partial occlusions, and negatives."""

    def __init__(self, bounds: RandomizationBounds, *, scene_seed: int) -> None:
        if not isinstance(bounds, RandomizationBounds):
            raise TypeError("bounds must be RandomizationBounds")
        if isinstance(scene_seed, bool) or not isinstance(scene_seed, int) or scene_seed < 0:
            raise ValueError("scene_seed must be a non-negative integer")
        self._bounds = bounds
        self._scene_seed = scene_seed

    def plan(self, episode_index: int) -> EpisodeRandomization:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError("episode_index must be a non-negative integer")
        # The first three episodes deterministically cover all required sample
        # kinds.  Later episodes follow the configured fractions.
        rng = random.Random((self._scene_seed << 32) ^ episode_index)
        if episode_index % 3 == 0:
            sample_kind = "positive"
        elif episode_index % 3 == 1:
            sample_kind = "partial_occlusion"
        elif episode_index % 3 == 2:
            sample_kind = "negative"
        else:  # pragma: no cover - modulo above is exhaustive
            sample_kind = "positive"
        if episode_index >= 3:
            draw = rng.random()
            if draw < self._bounds.negative_fraction:
                sample_kind = "negative"
            elif draw < self._bounds.negative_fraction + self._bounds.partial_occlusion_fraction:
                sample_kind = "partial_occlusion"
            else:
                sample_kind = "positive"

        uniform = rng.uniform
        target_heading = uniform(-180.0, 180.0)
        target_position = (
            uniform(*self._bounds.target_x_m),
            uniform(*self._bounds.target_y_m),
            uniform(*self._bounds.target_altitude_m),
        )
        target_camera_distance = uniform(*self._bounds.target_camera_distance_m)
        view_bearing_rad = uniform(-3.141592653589793, 3.141592653589793)
        uav_x = min(
            self._bounds.uav_x_m[1],
            max(
                self._bounds.uav_x_m[0],
                target_position[0] - target_camera_distance * np.cos(view_bearing_rad),
            ),
        )
        uav_y = min(
            self._bounds.uav_y_m[1],
            max(
                self._bounds.uav_y_m[0],
                target_position[1] - target_camera_distance * np.sin(view_bearing_rad),
            ),
        )
        key = EpisodeKey(
            scene_seed=self._scene_seed,
            episode_id=f"episode_{episode_index:06d}",
            trajectory_id=f"trajectory_{self._scene_seed:08x}_{episode_index:06d}",
        )
        return EpisodeRandomization(
            key=key,
            uav_position_world_m=(
                uav_x,
                uav_y,
                uniform(*self._bounds.uav_altitude_m),
            ),
            uav_speed_mps=uniform(*self._bounds.uav_speed_mps),
            # A bounded offset prevents every Target from being centered.
            uav_yaw_deg=(
                np.degrees(view_bearing_rad) + uniform(-55.0, 55.0)
            )
            % 360.0
            - 180.0,
            camera_pitch_deg=uniform(*self._bounds.camera_pitch_deg),
            camera_horizontal_fov_deg=uniform(*self._bounds.camera_horizontal_fov_deg),
            target_position_world_m=target_position,
            target_speed_mps=uniform(*self._bounds.target_speed_mps),
            target_heading_deg=target_heading,
            target_scale=uniform(*self._bounds.target_scale),
            target_camera_distance_m=target_camera_distance,
            target_max_turn_rate_deg_s=uniform(*self._bounds.max_target_turn_rate_deg_s),
            target_direction_change_interval_s=uniform(
                *self._bounds.direction_change_interval_s
            ),
            background_variant=rng.randrange(8),
            material_variant=rng.randrange(12),
            light_intensity_scale=uniform(*self._bounds.light_intensity_scale),
            motion_blur_strength=uniform(*self._bounds.motion_blur_strength),
            sample_kind=sample_kind,
        )


@dataclass(frozen=True, slots=True)
class OracleFrameTruth:
    """Privileged truth paired with one atomic Camera sample.

    ``projected_target_pixels_uv`` and depths must come from Isaac scene truth,
    never from a detector.  Negative samples set both to ``None``.
    """

    camera_sample: CameraSample
    target_position_world_m: tuple[float, float, float] | None
    target_orientation_world_wxyz: tuple[float, float, float, float] | None
    projected_target_pixels_uv: np.ndarray | None
    projected_target_depth_m: np.ndarray | None
    occlusion_ratio: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_sample, CameraSample):
            raise TypeError("camera_sample must be a CameraSample")
        absent = self.target_position_world_m is None
        if absent != (self.target_orientation_world_wxyz is None):
            raise ValueError("target position and orientation must be provided together")
        if absent:
            if self.projected_target_pixels_uv is not None or self.projected_target_depth_m is not None:
                raise ValueError("negative samples cannot contain projected Target truth")
        else:
            pixels = np.asarray(self.projected_target_pixels_uv)
            depths = np.asarray(self.projected_target_depth_m)
            if pixels.ndim != 2 or pixels.shape[0] < 2 or pixels.shape[1] != 2:
                raise ValueError("projected Target pixels must have shape (N, 2), N >= 2")
            if depths.shape != (pixels.shape[0],):
                raise ValueError("projected Target depths must have shape (N,)")
        if self.occlusion_ratio is not None:
            ratio = _finite(self.occlusion_ratio, "occlusion_ratio")
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("occlusion_ratio must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ProjectedYoloLabel:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float
    bbox_xyxy_px: tuple[float, float, float, float]
    raw_area_px: float
    clipped_area_px: float
    visibility: str

    def yolo_line(self) -> str:
        return (
            f"{self.class_id} {self.center_x:.8f} {self.center_y:.8f} "
            f"{self.width:.8f} {self.height:.8f}\n"
        )


@dataclass(frozen=True, slots=True)
class ProjectionDecision:
    label: ProjectedYoloLabel | None
    visibility: str
    raw_area_px: float
    clipped_area_px: float
    occlusion_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class DepthVisibilityDecision:
    """Visibility inferred from one synchronized metric-depth crop.

    ``trusted`` is false when depth cannot prove that the projected target is
    rendered.  Callers must treat that result as label-negative (fail closed).
    """

    occlusion_ratio: float
    trusted: bool
    reason: str
    valid_depth_fraction: float
    target_depth_pixels: int
    occluder_depth_pixels: int


def estimate_depth_visibility(
    camera_sample: CameraSample,
    projected_target_pixels_uv: np.ndarray,
    projected_target_depth_m: np.ndarray,
    *,
    absolute_tolerance_m: float = 0.15,
    relative_tolerance: float = 0.03,
    min_valid_depth_fraction: float = 0.25,
    min_assessable_pixels: int = 4,
) -> DepthVisibilityDecision:
    """Estimate target occlusion from synchronized depth and GT projection.

    Pixels in the target's projected rectangle are divided into target-depth
    evidence and strictly nearer occluder evidence.  Farther background pixels
    are ignored because the rectangle can contain space outside the projected
    silhouette.  Missing/invalid depth or insufficient evidence cannot prove
    visibility and therefore returns ``trusted=False, occlusion_ratio=1``.
    """

    if not isinstance(camera_sample, CameraSample):
        raise TypeError("camera_sample must be a CameraSample")
    tolerance_m = _finite(absolute_tolerance_m, "absolute_tolerance_m")
    tolerance_fraction = _finite(relative_tolerance, "relative_tolerance")
    minimum_valid = _finite(
        min_valid_depth_fraction,
        "min_valid_depth_fraction",
    )
    if tolerance_m < 0.0 or tolerance_fraction < 0.0:
        raise ValueError("depth tolerances must be non-negative")
    if not 0.0 <= minimum_valid <= 1.0:
        raise ValueError("min_valid_depth_fraction must be in [0, 1]")
    if (
        isinstance(min_assessable_pixels, bool)
        or not isinstance(min_assessable_pixels, int)
        or min_assessable_pixels <= 0
    ):
        raise ValueError("min_assessable_pixels must be a positive integer")

    pixels = np.asarray(projected_target_pixels_uv, dtype=np.float64)
    depths = np.asarray(projected_target_depth_m, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[0] < 2 or pixels.shape[1] != 2:
        raise ValueError("projected Target pixels must have shape (N, 2), N >= 2")
    if depths.shape != (pixels.shape[0],):
        raise ValueError("projected Target depths must have shape (N,)")
    projection_valid = (
        np.all(np.isfinite(pixels), axis=1)
        & np.isfinite(depths)
        & (depths > 0.0)
    )
    if not np.any(projection_valid):
        return DepthVisibilityDecision(1.0, False, "invalid_projection", 0.0, 0, 0)

    usable_pixels = pixels[projection_valid]
    usable_depths = depths[projection_valid]
    intrinsics = camera_sample.intrinsics
    x1 = max(0, min(intrinsics.width, int(np.floor(np.min(usable_pixels[:, 0])))))
    y1 = max(0, min(intrinsics.height, int(np.floor(np.min(usable_pixels[:, 1])))))
    x2 = max(0, min(intrinsics.width, int(np.ceil(np.max(usable_pixels[:, 0])))))
    y2 = max(0, min(intrinsics.height, int(np.ceil(np.max(usable_pixels[:, 1])))))
    if x2 <= x1 or y2 <= y1:
        return DepthVisibilityDecision(1.0, False, "out_of_frame", 0.0, 0, 0)

    depth_image = camera_sample.depth_to_image_plane_m
    if depth_image is None:
        return DepthVisibilityDecision(1.0, False, "depth_unavailable", 0.0, 0, 0)
    crop = np.asarray(depth_image[y1:y2, x1:x2], dtype=np.float64)
    valid_depth = np.isfinite(crop) & (crop > 0.0)
    valid_count = int(np.count_nonzero(valid_depth))
    valid_fraction = valid_count / float(crop.size)
    if valid_count == 0 or valid_fraction < minimum_valid:
        return DepthVisibilityDecision(
            1.0,
            False,
            "invalid_depth",
            valid_fraction,
            0,
            0,
        )

    target_near_m = float(np.min(usable_depths))
    target_far_m = float(np.max(usable_depths))
    tolerance = max(
        tolerance_m,
        tolerance_fraction * float(np.median(usable_depths)),
    )
    target_evidence = (
        valid_depth
        & (crop >= target_near_m - tolerance)
        & (crop <= target_far_m + tolerance)
    )
    occluder_evidence = valid_depth & (crop < target_near_m - tolerance)
    target_count = int(np.count_nonzero(target_evidence))
    occluder_count = int(np.count_nonzero(occluder_evidence))
    assessable_count = target_count + occluder_count
    if assessable_count < min_assessable_pixels:
        return DepthVisibilityDecision(
            1.0,
            False,
            "depth_unresolved",
            valid_fraction,
            target_count,
            occluder_count,
        )
    occlusion_ratio = occluder_count / float(assessable_count)
    return DepthVisibilityDecision(
        occlusion_ratio,
        True,
        "fully_occluded" if target_count == 0 else "depth_confirmed",
        valid_fraction,
        target_count,
        occluder_count,
    )


def project_oracle_bbox(
    truth: OracleFrameTruth,
    *,
    class_id: int,
    min_bbox_area_px: float,
) -> ProjectionDecision:
    """Clip an Isaac-GT projection and convert it to normalized YOLO form."""

    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
        raise ValueError("class_id must be a non-negative integer")
    minimum_area = _finite(min_bbox_area_px, "min_bbox_area_px")
    if minimum_area < 0.0:
        raise ValueError("min_bbox_area_px must be non-negative")
    if truth.target_position_world_m is None:
        return ProjectionDecision(None, "negative", 0.0, 0.0)
    if truth.occlusion_ratio is not None and truth.occlusion_ratio >= 1.0:
        return ProjectionDecision(None, "fully_occluded", 0.0, 0.0, 1.0)
    assert truth.projected_target_pixels_uv is not None
    assert truth.projected_target_depth_m is not None
    pixels = np.asarray(truth.projected_target_pixels_uv, dtype=np.float64)
    depths = np.asarray(truth.projected_target_depth_m, dtype=np.float64)
    valid = np.all(np.isfinite(pixels), axis=1) & np.isfinite(depths) & (depths > 0.0)
    if not np.any(valid):
        return ProjectionDecision(None, "behind_camera", 0.0, 0.0)
    usable = pixels[valid]
    raw_x1 = float(np.min(usable[:, 0]))
    raw_y1 = float(np.min(usable[:, 1]))
    raw_x2 = float(np.max(usable[:, 0]))
    raw_y2 = float(np.max(usable[:, 1]))
    raw_area = max(0.0, raw_x2 - raw_x1) * max(0.0, raw_y2 - raw_y1)
    intrinsics = truth.camera_sample.intrinsics
    x1 = min(float(intrinsics.width), max(0.0, raw_x1))
    y1 = min(float(intrinsics.height), max(0.0, raw_y1))
    x2 = min(float(intrinsics.width), max(0.0, raw_x2))
    y2 = min(float(intrinsics.height), max(0.0, raw_y2))
    clipped_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return ProjectionDecision(None, "out_of_frame", raw_area, 0.0)
    if clipped_area < minimum_area:
        return ProjectionDecision(None, "too_small", raw_area, clipped_area)
    depth_visibility = estimate_depth_visibility(
        truth.camera_sample,
        pixels,
        depths,
    )
    if not depth_visibility.trusted:
        return ProjectionDecision(
            None,
            depth_visibility.reason,
            raw_area,
            clipped_area,
            1.0,
        )
    effective_occlusion = max(
        depth_visibility.occlusion_ratio,
        0.0 if truth.occlusion_ratio is None else truth.occlusion_ratio,
    )
    if effective_occlusion >= 1.0:
        return ProjectionDecision(
            None,
            "fully_occluded",
            raw_area,
            clipped_area,
            1.0,
        )
    clipped = any(
        abs(a - b) > 1e-9
        for a, b in ((x1, raw_x1), (y1, raw_y1), (x2, raw_x2), (y2, raw_y2))
    )
    visibility = (
        "partially_occluded"
        if effective_occlusion > 0.0
        else "edge_clipped" if clipped else "visible"
    )
    width_px = x2 - x1
    height_px = y2 - y1
    label = ProjectedYoloLabel(
        class_id=class_id,
        center_x=((x1 + x2) * 0.5) / intrinsics.width,
        center_y=((y1 + y2) * 0.5) / intrinsics.height,
        width=width_px / intrinsics.width,
        height=height_px / intrinsics.height,
        bbox_xyxy_px=(x1, y1, x2, y2),
        raw_area_px=raw_area,
        clipped_area_px=clipped_area,
        visibility=visibility,
    )
    return ProjectionDecision(
        label,
        visibility,
        raw_area,
        clipped_area,
        effective_occlusion,
    )


@runtime_checkable
class IsaacCollectionAdapter(Protocol):
    """Adapter implemented only after ``SimulationApp`` has been created."""

    def begin_episode(self, randomization: EpisodeRandomization) -> None:
        """Apply every requested randomization and reset trajectory state."""

    def advance_to_next_sample(self, sample_period_s: float) -> None:
        """Advance smooth dynamics; never randomize heading frame-by-frame."""

    def capture_oracle_frame(self, frame_id: str) -> OracleFrameTruth:
        """Return GT projection paired with the current CameraSample."""


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    output_dir: Path
    total_samples: int
    positive_labels: int
    positive_samples: int
    partial_occlusion_samples: int
    negative_samples: int
    edge_clipped_samples: int
    too_small_samples: int
    split_counts: Mapping[str, int]
    manifest_path: Path


class IsaacYoloDatasetCollector:
    """Bounded collection loop over a late-bound Isaac adapter."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        class_names: Sequence[str],
        class_id: int,
        limits: CollectionLimits,
        bounds: RandomizationBounds,
        scene_seed: int,
        oracle_label_generation: bool,
        acknowledge_privileged_oracle: bool,
    ) -> None:
        require_oracle_label_acknowledgements(
            oracle_label_generation=oracle_label_generation,
            acknowledge_privileged_oracle=acknowledge_privileged_oracle,
        )
        if not isinstance(limits, CollectionLimits):
            raise TypeError("limits must be CollectionLimits")
        if not isinstance(bounds, RandomizationBounds):
            raise TypeError("bounds must be RandomizationBounds")
        names = tuple(str(name).strip() for name in class_names)
        if not names or any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("class_names must be unique non-empty strings")
        if isinstance(class_id, bool) or not isinstance(class_id, int) or not 0 <= class_id < len(names):
            raise ValueError("class_id must index class_names")
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._class_names = names
        self._class_id = class_id
        self._limits = limits
        self._randomizer = EpisodeRandomizer(bounds, scene_seed=scene_seed)

    def collect(self, adapter: IsaacCollectionAdapter) -> CollectionSummary:
        if not isinstance(adapter, IsaacCollectionAdapter):
            raise TypeError("adapter must satisfy IsaacCollectionAdapter")
        self._prepare_directories()
        manifest_path = self._output_dir / "manifest.jsonl"
        total = positive = positive_samples = partial = negative = clipped = too_small = 0
        split_counts = {"train": 0, "val": 0, "test": 0}
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for episode_index in range(self._limits.max_episodes):
                if total >= self._limits.max_samples:
                    break
                plan = self._randomizer.plan(episode_index)
                split = split_for_episode(plan.key, self._limits)
                adapter.begin_episode(plan)
                for episode_frame_index in range(self._limits.frames_per_episode):
                    if total >= self._limits.max_samples:
                        break
                    adapter.advance_to_next_sample(1.0 / self._limits.sample_hz)
                    frame_id = f"{plan.key.episode_id}_frame_{episode_frame_index:06d}"
                    truth = adapter.capture_oracle_frame(frame_id)
                    decision = project_oracle_bbox(
                        truth,
                        class_id=self._class_id,
                        min_bbox_area_px=self._limits.min_bbox_area_px,
                    )
                    self._write_sample(split, frame_id, truth, decision)
                    record = self._manifest_record(
                        split=split,
                        frame_id=frame_id,
                        plan=plan,
                        truth=truth,
                        decision=decision,
                    )
                    manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    total += 1
                    split_counts[split] += 1
                    positive += int(decision.label is not None)
                    positive_samples += int(
                        plan.sample_kind == "positive" and decision.label is not None
                    )
                    partial += int(
                        plan.sample_kind == "partial_occlusion"
                        and decision.label is not None
                        and decision.visibility == "partially_occluded"
                    )
                    negative += int(decision.visibility == "negative")
                    clipped += int(decision.visibility == "edge_clipped")
                    too_small += int(decision.visibility == "too_small")
        self._finalize_dataset(
            split_counts,
            positive_samples=positive_samples,
            partial_occlusion_samples=partial,
            negative_samples=negative,
        )
        return CollectionSummary(
            output_dir=self._output_dir,
            total_samples=total,
            positive_labels=positive,
            positive_samples=positive_samples,
            partial_occlusion_samples=partial,
            negative_samples=negative,
            edge_clipped_samples=clipped,
            too_small_samples=too_small,
            split_counts=split_counts,
            manifest_path=manifest_path,
        )

    def _prepare_directories(self) -> None:
        if self._output_dir.exists() and any(self._output_dir.iterdir()):
            raise IsaacDatasetCollectionError(
                f"output directory is not empty: {self._output_dir}; choose a new directory"
            )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            (self._output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self._output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _write_sample(
        self,
        split: str,
        frame_id: str,
        truth: OracleFrameTruth,
        decision: ProjectionDecision,
    ) -> None:
        image_path = self._output_dir / "images" / split / f"{frame_id}.jpg"
        label_path = self._output_dir / "labels" / split / f"{frame_id}.txt"
        Image.fromarray(truth.camera_sample.rgb).save(
            image_path,
            format="JPEG",
            quality=self._limits.jpeg_quality,
            optimize=True,
        )
        label_path.write_text(
            "" if decision.label is None else decision.label.yolo_line(),
            encoding="utf-8",
        )

    def _write_dataset_yaml(self) -> None:
        descriptor = {
            "path": str(self._output_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for index, name in enumerate(self._class_names)},
        }
        (self._output_dir / "data.yaml").write_text(
            yaml.safe_dump(descriptor, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _finalize_dataset(
        self,
        split_counts: Mapping[str, int],
        *,
        positive_samples: int,
        partial_occlusion_samples: int,
        negative_samples: int,
    ) -> None:
        empty_splits = tuple(
            split for split in ("train", "val", "test") if split_counts.get(split, 0) <= 0
        )
        if empty_splits:
            rendered = ", ".join(empty_splits)
            raise IsaacDatasetCollectionError(
                f"dataset split(s) {rendered} are empty after grouped assignment; "
                "increase --max-episodes and --max-samples so more complete "
                "episode/trajectory groups are assigned, or choose a different "
                "--scene-seed, then "
                "collect into a new output directory"
            )
        missing_kinds = tuple(
            name
            for name, count in (
                ("positive", positive_samples),
                ("partially_occluded", partial_occlusion_samples),
                ("negative", negative_samples),
            )
            if count <= 0
        )
        if missing_kinds:
            raise IsaacDatasetCollectionError(
                "dataset is missing required realized sample kind(s): "
                + ", ".join(missing_kinds)
                + "; planned randomization labels are insufficient when the "
                "rendered target is out of frame or depth visibility fails. "
                "Increase --max-episodes/--max-samples or choose a different "
                "--scene-seed, then collect into a new output directory"
            )
        self._write_dataset_yaml()

    def _manifest_record(
        self,
        *,
        split: str,
        frame_id: str,
        plan: EpisodeRandomization,
        truth: OracleFrameTruth,
        decision: ProjectionDecision,
    ) -> dict[str, Any]:
        label = decision.label
        return {
            "schema_version": 1,
            "oracle_label_generation": True,
            "split": split,
            "scene_seed": plan.key.scene_seed,
            "episode_id": plan.key.episode_id,
            "trajectory_id": plan.key.trajectory_id,
            "frame_id": frame_id,
            "timestamp_s": truth.camera_sample.timestamp_s,
            "camera_pose": {
                "position_world_m": list(truth.camera_sample.camera_position_world_m),
                "orientation_world_wxyz": list(
                    truth.camera_sample.camera_orientation_world_wxyz
                ),
            },
            "camera_intrinsics": truth.camera_sample.intrinsics.to_dict(),
            "target_pose": None
            if truth.target_position_world_m is None
            else {
                "position_world_m": list(truth.target_position_world_m),
                "orientation_world_wxyz": list(truth.target_orientation_world_wxyz or ()),
            },
            "visibility": decision.visibility,
            "occlusion_ratio": decision.occlusion_ratio,
            "class_id": self._class_id,
            "bbox_xyxy_px": None if label is None else list(label.bbox_xyxy_px),
            "raw_bbox_area_px": decision.raw_area_px,
            "clipped_bbox_area_px": decision.clipped_area_px,
            "randomization": plan.to_manifest_dict(),
        }


__all__ = [
    "CollectionLimits",
    "CollectionSummary",
    "DepthVisibilityDecision",
    "EpisodeKey",
    "EpisodeRandomization",
    "EpisodeRandomizer",
    "IsaacCollectionAdapter",
    "IsaacDatasetCollectionError",
    "IsaacYoloDatasetCollector",
    "OracleFrameTruth",
    "ProjectedYoloLabel",
    "ProjectionDecision",
    "RandomizationBounds",
    "estimate_depth_visibility",
    "project_oracle_bbox",
    "require_oracle_label_acknowledgements",
    "split_for_episode",
]
