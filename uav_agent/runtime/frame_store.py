"""Bounded in-memory RGB frame storage for asynchronous visual review.

Only :class:`FrameRef` metadata is intended to leave this module.  Pixel data
is owned by :class:`FrameStore`, never embedded in mission events or world
belief snapshots, and is evicted by count, byte, and age limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from threading import RLock

import numpy as np

from common.ids import validate_routing_id, validate_uav_id


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
class _StoredFrame:
    ref: FrameRef
    rgb: np.ndarray
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
    ) -> FrameRef:
        """Store one RGB frame and evict the oldest entries as needed."""

        normalized_uav_id = validate_uav_id(uav_id)
        normalized_frame_id = validate_routing_id(frame_id, "frame_id")
        normalized_timestamp = _timestamp(timestamp_s)
        if not isinstance(rgb, np.ndarray):
            raise TypeError("rgb must be a numpy.ndarray")
        if rgb.dtype != np.uint8:
            raise TypeError("rgb must have dtype uint8")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        height = _positive_integer(int(rgb.shape[0]), "rgb height")
        width = _positive_integer(int(rgb.shape[1]), "rgb width")
        byte_count = int(rgb.nbytes)
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

            stored_rgb = np.ascontiguousarray(rgb).copy()
            stored_rgb.setflags(write=False)
            self._sequence += 1
            self._frames[key] = _StoredFrame(
                ref=ref,
                rgb=stored_rgb,
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

    def contains(self, ref: FrameRef) -> bool:
        return self.get_frame(ref, copy=False) is not None

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
                if item.ref.uav_id == uav_id and item.ref.timestamp_s < cutoff
            ),
            key=lambda item: (item.ref.timestamp_s, item.sequence),
        )
        for item in expired:
            self._remove_locked((item.ref.uav_id, item.ref.frame_id))

        while (
            len(self._frames) > self._max_frames
            or self._total_bytes > self._max_total_bytes
        ):
            oldest = min(
                self._frames.values(),
                key=lambda item: item.sequence,
            )
            self._remove_locked((oldest.ref.uav_id, oldest.ref.frame_id))

    def _remove_locked(self, key: tuple[str, str]) -> None:
        item = self._frames.pop(key)
        self._total_bytes -= item.byte_count


BoundedFrameStore = FrameStore


__all__ = ["BoundedFrameStore", "FrameRef", "FrameStore"]
