"""Bounded in-memory synchronized RGB-D frame storage.

Only :class:`FrameRef` metadata is intended to leave this module.  Pixel data
is owned by :class:`FrameStore`, never embedded in mission events or world
belief snapshots, and is evicted by count, byte, and age limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from threading import RLock
from typing import Sequence

import numpy as np

from common.ids import validate_routing_id, validate_uav_id
from env.camera_types import CameraIntrinsics, CameraSample


def _timestamp(value: object, field_name: str = "timestamp_s") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return normalized


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _finite_vector3(
    value: Sequence[float] | np.ndarray,
    field_name: str,
) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must contain three finite numbers")
    try:
        components = tuple(value)
    except TypeError:
        raise TypeError(
            f"{field_name} must contain three finite numbers"
        ) from None
    if len(components) != 3:
        raise ValueError(f"{field_name} must contain three finite numbers")
    normalized: list[float] = []
    for component in components:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise TypeError(f"{field_name} must contain three finite numbers")
        number = float(component)
        if not isfinite(number):
            raise ValueError(f"{field_name} must contain three finite numbers")
        normalized.append(number)
    return normalized[0], normalized[1], normalized[2]


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Small routing-safe reference to RGB data retained by ``FrameStore``."""

    uav_id: str
    frame_id: str
    timestamp_s: float
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "frame_id",
            validate_routing_id(self.frame_id, "frame_id"),
        )
        object.__setattr__(
            self,
            "timestamp_s",
            _timestamp(self.timestamp_s),
        )
        object.__setattr__(self, "width", _positive_integer(self.width, "width"))
        object.__setattr__(
            self,
            "height",
            _positive_integer(self.height, "height"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "uav_id": self.uav_id,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class FrameCameraGeometry:
    """Camera calibration and world pose captured with one ``FrameRef``."""

    timestamp_s: float
    intrinsics: CameraIntrinsics
    camera_position_world_m: tuple[float, float, float]
    camera_orientation_world_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FrameUavSelfMotion:
    """Agent-visible UAV motion synchronized with one camera frame.

    These values come from the vehicle/Observation side of the production
    boundary.  They are deliberately separate from camera-pose finite
    differences so a temporal model cannot silently train on one feature
    meaning and receive another at deployment.
    """

    linear_velocity_world_mps: tuple[float, float, float]
    angular_velocity_body_radps: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "linear_velocity_world_mps",
            _finite_vector3(
                self.linear_velocity_world_mps,
                "uav_linear_velocity_world_mps",
            ),
        )
        object.__setattr__(
            self,
            "angular_velocity_body_radps",
            _finite_vector3(
                self.angular_velocity_body_radps,
                "uav_angular_velocity_body_radps",
            ),
        )


@dataclass(frozen=True, slots=True)
class _StoredFrame:
    ref: FrameRef
    rgb: np.ndarray
    depth_to_image_plane_m: np.ndarray | None
    camera_geometry: FrameCameraGeometry | None
    uav_self_motion: FrameUavSelfMotion | None
    byte_count: int
    sequence: int


class FrameStore:
    """Thread-safe bounded ring buffer keyed by ``(uav_id, frame_id)``.

    A contiguous private uint8 copy is made on insertion so later mutation of
    a simulator-owned array cannot alter a pending visual-review request.
    Retrieval returns a copy by default.  ``copy=False`` returns a read-only
    view owned by the store and is intended only for immediate encoding.
    """

    def __init__(
        self,
        *,
        max_frames: int = 24,
        max_bytes: int = 67_108_864,
        max_age_s: float = 20.0,
        max_total_bytes: int | None = None,
        max_frame_age_s: float | None = None,
    ) -> None:
        # ``max_bytes``/``max_age_s`` match the public configuration keys.
        # The longer aliases retain compatibility with the first runtime API.
        if max_total_bytes is not None:
            if max_bytes != 67_108_864 and max_bytes != max_total_bytes:
                raise ValueError("max_bytes conflicts with max_total_bytes")
            max_bytes = max_total_bytes
        if max_frame_age_s is not None:
            if max_age_s != 20.0 and max_age_s != max_frame_age_s:
                raise ValueError("max_age_s conflicts with max_frame_age_s")
            max_age_s = max_frame_age_s
        self._max_frames = _positive_integer(max_frames, "max_frames")
        self._max_total_bytes = _positive_integer(
            max_bytes,
            "max_bytes",
        )
        age = _timestamp(max_age_s, "max_age_s")
        if age == 0.0:
            raise ValueError("max_age_s must be greater than zero")
        self._max_frame_age_s = age

        self._frames: dict[tuple[str, str], _StoredFrame] = {}
        # In-flight model requests may outlive the count/byte retention
        # window when simulation time advances faster than wall-clock HTTP.
        # Pins protect only explicitly referenced frames; hard count and byte
        # bounds remain in force because newly inserted, unpinned frames are
        # still eligible for eviction.
        self._pins: dict[tuple[str, str], int] = {}
        self._total_bytes = 0
        self._sequence = 0
        self._latest_timestamp_by_uav: dict[str, float] = {}
        self._lock = RLock()

    @property
    def max_frames(self) -> int:
        return self._max_frames

    @property
    def max_total_bytes(self) -> int:
        return self._max_total_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_total_bytes

    @property
    def max_frame_age_s(self) -> float:
        return self._max_frame_age_s

    @property
    def max_age_s(self) -> float:
        return self._max_frame_age_s

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def add_frame(
        self,
        *,
        uav_id: str,
        frame_id: str,
        timestamp_s: float,
        rgb: np.ndarray,
        depth_to_image_plane_m: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
        camera_position_world_m: tuple[float, float, float] | None = None,
        camera_orientation_world_wxyz: tuple[float, float, float, float] | None = None,
        uav_linear_velocity_world_mps: Sequence[float] | np.ndarray | None = None,
        uav_angular_velocity_body_radps: Sequence[float] | np.ndarray | None = None,
    ) -> FrameRef:
        """Store one frame and evict the oldest entries as needed.

        Legacy RGB-only callers omit all optional arguments.  Geometry fields
        are all-or-none, and a depth plane is accepted only with matching
        intrinsics and pose so it can never become detached from its frame.
        """

        normalized_uav_id = validate_uav_id(uav_id)
        normalized_frame_id = validate_routing_id(frame_id, "frame_id")
        normalized_timestamp = _timestamp(timestamp_s)
        geometry_values = (
            intrinsics,
            camera_position_world_m,
            camera_orientation_world_wxyz,
        )
        has_geometry = any(value is not None for value in geometry_values)
        if has_geometry and not all(value is not None for value in geometry_values):
            raise ValueError(
                "intrinsics and camera world pose must be supplied together"
            )
        if depth_to_image_plane_m is not None and not has_geometry:
            raise ValueError("depth_to_image_plane_m requires camera geometry")
        motion_values = (
            uav_linear_velocity_world_mps,
            uav_angular_velocity_body_radps,
        )
        has_self_motion = any(value is not None for value in motion_values)
        if has_self_motion and not all(value is not None for value in motion_values):
            raise ValueError(
                "UAV linear and angular velocity must be supplied together"
            )
        if has_self_motion and not has_geometry:
            raise ValueError("UAV self-motion requires synchronized camera geometry")

        camera_sample: CameraSample | None = None
        if has_geometry:
            assert intrinsics is not None
            assert camera_position_world_m is not None
            assert camera_orientation_world_wxyz is not None
            camera_sample = CameraSample(
                timestamp_s=normalized_timestamp,
                rgb=rgb,
                depth_to_image_plane_m=depth_to_image_plane_m,
                camera_position_world_m=camera_position_world_m,
                camera_orientation_world_wxyz=camera_orientation_world_wxyz,
                intrinsics=intrinsics,
            )
            normalized_rgb = camera_sample.rgb
            normalized_depth = camera_sample.depth_to_image_plane_m
        else:
            if not isinstance(rgb, np.ndarray):
                raise TypeError("rgb must be a numpy.ndarray")
            if rgb.dtype != np.uint8:
                raise TypeError("rgb must have dtype uint8")
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("rgb must have shape (height, width, 3)")
            normalized_rgb = rgb
            normalized_depth = None
        height = _positive_integer(int(normalized_rgb.shape[0]), "rgb height")
        width = _positive_integer(int(normalized_rgb.shape[1]), "rgb width")
        byte_count = int(normalized_rgb.nbytes) + (
            0 if normalized_depth is None else int(normalized_depth.nbytes)
        )
        if byte_count > self._max_total_bytes:
            raise ValueError("one frame exceeds max_bytes")

        ref = FrameRef(
            uav_id=normalized_uav_id,
            frame_id=normalized_frame_id,
            timestamp_s=normalized_timestamp,
            width=width,
            height=height,
        )
        key = (ref.uav_id, ref.frame_id)

        with self._lock:
            if key in self._frames:
                raise ValueError(
                    "frame_id is already present for this uav_id"
                )
            latest_for_uav = self._latest_timestamp_by_uav.get(
                normalized_uav_id
            )
            if (
                latest_for_uav is not None
                and normalized_timestamp
                < latest_for_uav - self._max_frame_age_s
            ):
                raise ValueError("frame is older than the configured age window")

            stored_rgb = np.ascontiguousarray(normalized_rgb).copy()
            stored_rgb.setflags(write=False)
            stored_depth: np.ndarray | None = None
            stored_geometry: FrameCameraGeometry | None = None
            stored_self_motion: FrameUavSelfMotion | None = None
            if camera_sample is not None:
                if normalized_depth is not None:
                    stored_depth = np.ascontiguousarray(normalized_depth).copy()
                    stored_depth.setflags(write=False)
                stored_geometry = FrameCameraGeometry(
                    timestamp_s=camera_sample.timestamp_s,
                    intrinsics=camera_sample.intrinsics,
                    camera_position_world_m=camera_sample.camera_position_world_m,
                    camera_orientation_world_wxyz=(
                        camera_sample.camera_orientation_world_wxyz
                    ),
                )
                if has_self_motion:
                    assert uav_linear_velocity_world_mps is not None
                    assert uav_angular_velocity_body_radps is not None
                    stored_self_motion = FrameUavSelfMotion(
                        linear_velocity_world_mps=tuple(
                            uav_linear_velocity_world_mps
                        ),
                        angular_velocity_body_radps=tuple(
                            uav_angular_velocity_body_radps
                        ),
                    )
            self._sequence += 1
            self._frames[key] = _StoredFrame(
                ref=ref,
                rgb=stored_rgb,
                depth_to_image_plane_m=stored_depth,
                camera_geometry=stored_geometry,
                uav_self_motion=stored_self_motion,
                byte_count=byte_count,
                sequence=self._sequence,
            )
            self._total_bytes += byte_count
            latest_for_uav = (
                normalized_timestamp
                if latest_for_uav is None
                else max(latest_for_uav, normalized_timestamp)
            )
            self._latest_timestamp_by_uav[normalized_uav_id] = latest_for_uav
            self._evict_locked(normalized_uav_id, latest_for_uav)
        return ref

    def add_sample(
        self,
        *,
        uav_id: str,
        frame_id: str,
        sample: CameraSample,
        uav_linear_velocity_world_mps: Sequence[float] | np.ndarray | None = None,
        uav_angular_velocity_body_radps: Sequence[float] | np.ndarray | None = None,
    ) -> FrameRef:
        """Store all channels from one already-synchronized camera sample."""

        if not isinstance(sample, CameraSample):
            raise TypeError("sample must be a CameraSample")
        return self.add_frame(
            uav_id=uav_id,
            frame_id=frame_id,
            timestamp_s=sample.timestamp_s,
            rgb=sample.rgb,
            depth_to_image_plane_m=sample.depth_to_image_plane_m,
            intrinsics=sample.intrinsics,
            camera_position_world_m=sample.camera_position_world_m,
            camera_orientation_world_wxyz=sample.camera_orientation_world_wxyz,
            uav_linear_velocity_world_mps=uav_linear_velocity_world_mps,
            uav_angular_velocity_body_radps=uav_angular_velocity_body_radps,
        )

    # Explicit long name for call sites where ``sample`` might be ambiguous.
    add_camera_sample = add_sample

    def get_frame(
        self,
        ref: FrameRef,
        *,
        copy: bool = True,
    ) -> np.ndarray | None:
        """Return retained pixels for exactly ``ref``, or ``None`` if evicted."""

        if not isinstance(ref, FrameRef):
            raise TypeError("ref must be a FrameRef")
        if not isinstance(copy, bool):
            raise TypeError("copy must be a bool")
        with self._lock:
            stored = self._frames.get((ref.uav_id, ref.frame_id))
            if stored is None or stored.ref != ref:
                return None
            if copy:
                return stored.rgb.copy()
            view = stored.rgb.view()
            view.setflags(write=False)
            return view

    def get_camera_sample(self, ref: FrameRef) -> CameraSample | None:
        """Atomically retrieve one synchronized RGB-D sample.

        Both pixel channels and camera geometry are resolved under the same
        lock.  The returned ``CameraSample`` owns immutable copies, so an
        eviction immediately after this method returns cannot split RGB from
        depth or alter the measurement.
        """

        if not isinstance(ref, FrameRef):
            raise TypeError("ref must be a FrameRef")
        with self._lock:
            stored = self._frames.get((ref.uav_id, ref.frame_id))
            if (
                stored is None
                or stored.ref != ref
                or stored.camera_geometry is None
            ):
                return None
            geometry = stored.camera_geometry
            return CameraSample(
                timestamp_s=geometry.timestamp_s,
                rgb=stored.rgb,
                depth_to_image_plane_m=stored.depth_to_image_plane_m,
                camera_position_world_m=geometry.camera_position_world_m,
                camera_orientation_world_wxyz=(
                    geometry.camera_orientation_world_wxyz
                ),
                intrinsics=geometry.intrinsics,
            )

    def get_uav_self_motion(self, ref: FrameRef) -> FrameUavSelfMotion | None:
        """Return UAV self-motion captured atomically with ``ref``."""

        if not isinstance(ref, FrameRef):
            raise TypeError("ref must be a FrameRef")
        with self._lock:
            stored = self._frames.get((ref.uav_id, ref.frame_id))
            if stored is None or stored.ref != ref:
                return None
            return stored.uav_self_motion

    def get_temporal_inputs(
        self,
        ref: FrameRef,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        FrameCameraGeometry,
        FrameUavSelfMotion,
    ] | None:
        """Atomically borrow one complete temporal-model sensor input.

        Returned image arrays are read-only views owned by this bounded store
        and are valid for immediate inference preprocessing only.  A missing
        depth, pose, or UAV self-motion component rejects the entire sample;
        channels are never assembled across different frame generations.
        """

        if not isinstance(ref, FrameRef):
            raise TypeError("ref must be a FrameRef")
        with self._lock:
            stored = self._frames.get((ref.uav_id, ref.frame_id))
            if (
                stored is None
                or stored.ref != ref
                or stored.depth_to_image_plane_m is None
                or stored.camera_geometry is None
                or stored.uav_self_motion is None
            ):
                return None
            rgb = stored.rgb.view()
            depth = stored.depth_to_image_plane_m.view()
            rgb.setflags(write=False)
            depth.setflags(write=False)
            return (
                rgb,
                depth,
                stored.camera_geometry,
                stored.uav_self_motion,
            )

    def get_depth(
        self,
        ref: FrameRef,
        *,
        copy: bool = True,
    ) -> np.ndarray | None:
        """Return the synchronized metric Z-depth plane, if retained."""

        if not isinstance(ref, FrameRef):
            raise TypeError("ref must be a FrameRef")
        if not isinstance(copy, bool):
            raise TypeError("copy must be a bool")
        with self._lock:
            stored = self._frames.get((ref.uav_id, ref.frame_id))
            if (
                stored is None
                or stored.ref != ref
                or stored.depth_to_image_plane_m is None
            ):
                return None
            if copy:
                return stored.depth_to_image_plane_m.copy()
            view = stored.depth_to_image_plane_m.view()
            view.setflags(write=False)
            return view

    def get_camera_geometry(
        self,
        ref: FrameRef,
    ) -> FrameCameraGeometry | None:
        """Return immutable calibration/pose captured with exactly ``ref``."""

        if not isinstance(ref, FrameRef):
            raise TypeError("ref must be a FrameRef")
        with self._lock:
            stored = self._frames.get((ref.uav_id, ref.frame_id))
            if stored is None or stored.ref != ref:
                return None
            return stored.camera_geometry

    def contains(self, ref: FrameRef) -> bool:
        return self.get_frame(ref, copy=False) is not None

    def pin(self, ref: FrameRef) -> None:
        """Retain an existing frame until the matching request is resolved."""

        if not isinstance(ref, FrameRef):
            raise TypeError("ref must be a FrameRef")
        key = (ref.uav_id, ref.frame_id)
        with self._lock:
            stored = self._frames.get(key)
            if stored is None or stored.ref != ref:
                raise ValueError("cannot pin an absent or mismatched frame")
            self._pins[key] = self._pins.get(key, 0) + 1

    def unpin(self, ref: FrameRef) -> None:
        """Release one matching in-flight retention claim."""

        if not isinstance(ref, FrameRef):
            raise TypeError("ref must be a FrameRef")
        key = (ref.uav_id, ref.frame_id)
        with self._lock:
            count = self._pins.get(key)
            if count is None:
                raise ValueError("frame is not pinned")
            if count == 1:
                self._pins.pop(key, None)
            else:
                self._pins[key] = count - 1
            newest = self._latest_timestamp_by_uav.get(ref.uav_id)
            if newest is not None:
                self._evict_locked(ref.uav_id, newest)

    def refs(self, *, uav_id: str | None = None) -> tuple[FrameRef, ...]:
        normalized_uav_id = None if uav_id is None else validate_uav_id(uav_id)
        with self._lock:
            entries = sorted(
                self._frames.values(),
                key=lambda item: (item.ref.timestamp_s, item.sequence),
            )
            return tuple(
                item.ref
                for item in entries
                if normalized_uav_id is None
                or item.ref.uav_id == normalized_uav_id
            )

    def latest_ref(self, *, uav_id: str) -> FrameRef | None:
        normalized_uav_id = validate_uav_id(uav_id)
        with self._lock:
            matches = (
                item
                for item in self._frames.values()
                if item.ref.uav_id == normalized_uav_id
            )
            latest = max(
                matches,
                key=lambda item: (item.ref.timestamp_s, item.sequence),
                default=None,
            )
            return None if latest is None else latest.ref

    def evict_expired(
        self,
        *,
        timestamp_s: float,
        uav_id: str | None = None,
    ) -> int:
        """Advance the age watermark and return the number of evicted frames."""

        now = _timestamp(timestamp_s)
        normalized_uav_id = None if uav_id is None else validate_uav_id(uav_id)
        with self._lock:
            previous_count = len(self._frames)
            affected_uavs = (
                set(self._latest_timestamp_by_uav)
                if normalized_uav_id is None
                else {normalized_uav_id}
            )
            for affected_uav in affected_uavs:
                latest = max(
                    self._latest_timestamp_by_uav.get(affected_uav, now),
                    now,
                )
                self._latest_timestamp_by_uav[affected_uav] = latest
                self._evict_locked(affected_uav, latest)
            return previous_count - len(self._frames)

    def clear(self, *, uav_id: str | None = None) -> int:
        normalized_uav_id = None if uav_id is None else validate_uav_id(uav_id)
        with self._lock:
            if normalized_uav_id is None:
                removed = len(self._frames)
                self._frames.clear()
                self._pins.clear()
                self._total_bytes = 0
                self._latest_timestamp_by_uav.clear()
                return removed
            keys = [
                key for key in self._frames if key[0] == normalized_uav_id
            ]
            for key in keys:
                self._remove_locked(key)
            self._latest_timestamp_by_uav.pop(normalized_uav_id, None)
            return len(keys)

    def _evict_locked(self, uav_id: str, newest_timestamp_s: float) -> None:
        cutoff = newest_timestamp_s - self._max_frame_age_s
        expired = sorted(
            (
                item
                for item in self._frames.values()
                if (
                    item.ref.uav_id == uav_id
                    and item.ref.timestamp_s < cutoff
                    and (item.ref.uav_id, item.ref.frame_id) not in self._pins
                )
            ),
            key=lambda item: (item.ref.timestamp_s, item.sequence),
        )
        for item in expired:
            self._remove_locked((item.ref.uav_id, item.ref.frame_id))

        while (
            len(self._frames) > self._max_frames
            or self._total_bytes > self._max_total_bytes
        ):
            candidates = tuple(
                item
                for item in self._frames.values()
                if (item.ref.uav_id, item.ref.frame_id) not in self._pins
            )
            if not candidates:
                # Pinning never adds bytes or entries, so this can only occur
                # when every retained frame already fits the configured hard
                # bounds.  Keep the guard defensive for future callers.
                break
            oldest = min(
                candidates,
                key=lambda item: item.sequence,
            )
            self._remove_locked((oldest.ref.uav_id, oldest.ref.frame_id))

    def _remove_locked(self, key: tuple[str, str]) -> None:
        item = self._frames.pop(key)
        self._pins.pop(key, None)
        self._total_bytes -= item.byte_count


BoundedFrameStore = FrameStore


__all__ = [
    "BoundedFrameStore",
    "FrameCameraGeometry",
    "FrameRef",
    "FrameStore",
    "FrameUavSelfMotion",
]
