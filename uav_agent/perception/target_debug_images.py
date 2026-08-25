"""Opt-in, hard-bounded representative target-perception debug images.

This module deliberately writes neither continuous frames nor video.  A run
may retain at most one annotated RGB frame for each whitelisted lifecycle
event and never more than the configured global image cap.  Oracle-labelled
frames are rejected even if a caller accidentally binds the sink.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import isfinite
from numbers import Real
from pathlib import Path
from threading import RLock

import numpy as np

from runtime.frame_store import FrameRef, FrameStore


TARGET_DEBUG_EVENTS = frozenset(
    {
        "first_detection",
        "first_candidate",
        "confirmation_success",
        "candidate_rejected",
        "target_lost",
        "reacquire_success",
    }
)


@dataclass(frozen=True, slots=True)
class TargetDebugAnnotation:
    """Small, non-privileged label set rendered into one representative RGB."""

    bbox_xyxy_normalized: tuple[float, float, float, float] | None
    class_id: int | None
    class_name: str | None
    confidence: float | None
    track_id: str | None
    candidate_id: str | None
    confirmed: bool
    position_world_m: tuple[float, float, float] | None
    measurement_age_s: float | None
    source: str
    sampled_pixel_uv: tuple[float, float] | None = None
    raw_depth_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.confirmed, bool):
            raise TypeError("confirmed must be a bool")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if self.class_id is not None and (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id < 0
        ):
            raise ValueError("class_id must be a non-negative integer or None")
        for name in ("class_name", "track_id", "candidate_id"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty string or None")
        if self.confidence is not None:
            confidence = _finite(self.confidence, "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0, 1]")
        if self.measurement_age_s is not None and (
            _finite(self.measurement_age_s, "measurement_age_s") < 0.0
        ):
            raise ValueError("measurement_age_s must be non-negative")
        if self.sampled_pixel_uv is not None:
            if len(self.sampled_pixel_uv) != 2:
                raise ValueError("sampled_pixel_uv must contain exactly 2 values")
            for index, value in enumerate(self.sampled_pixel_uv):
                if _finite(value, f"sampled_pixel_uv[{index}]") < 0.0:
                    raise ValueError("sampled_pixel_uv components must be non-negative")
        if self.raw_depth_m is not None and _finite(self.raw_depth_m, "raw_depth_m") <= 0.0:
            raise ValueError("raw_depth_m must be positive")
        if self.position_world_m is not None:
            if len(self.position_world_m) != 3:
                raise ValueError("position_world_m must contain exactly 3 values")
            for index, value in enumerate(self.position_world_m):
                _finite(value, f"position_world_m[{index}]")
        if self.bbox_xyxy_normalized is not None:
            if len(self.bbox_xyxy_normalized) != 4:
                raise ValueError(
                    "bbox_xyxy_normalized must contain exactly 4 values"
                )
            x1, y1, x2, y2 = (
                _finite(value, f"bbox_xyxy_normalized[{index}]")
                for index, value in enumerate(self.bbox_xyxy_normalized)
            )
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                raise ValueError(
                    "bbox_xyxy_normalized must be ordered inside [0, 1]"
                )

    def label_lines(self, event: str) -> tuple[str, ...]:
        """Return the complete deterministic annotation rendered into RGB."""

        _validate_event(event)
        category = self.class_name or "N/A"
        class_id = "N/A" if self.class_id is None else str(self.class_id)
        confidence = (
            "N/A" if self.confidence is None else f"{self.confidence:.3f}"
        )
        position = (
            "N/A"
            if self.position_world_m is None
            else "(" + ", ".join(f"{value:.3f}" for value in self.position_world_m) + ")"
        )
        age = (
            "N/A"
            if self.measurement_age_s is None
            else f"{self.measurement_age_s:.3f}s"
        )
        sample = (
            "N/A"
            if self.sampled_pixel_uv is None
            else "(" + ", ".join(f"{value:.1f}" for value in self.sampled_pixel_uv) + ")"
        )
        raw_depth = "N/A" if self.raw_depth_m is None else f"{self.raw_depth_m:.3f}m"
        return (
            f"event={event}",
            f"class={category} class_id={class_id} confidence={confidence}",
            f"track_id={self.track_id or 'N/A'} candidate_id={self.candidate_id or 'N/A'}",
            f"confirmed={str(self.confirmed).lower()}",
            f"position_world_m={position}",
            f"measurement_age={age}",
            f"sampled_pixel_uv={sample} raw_depth={raw_depth}",
        )


@dataclass(frozen=True, slots=True)
class TargetDebugImageStats:
    count: int
    bytes: int
    events: tuple[str, ...]

    def to_manifest_dict(self) -> dict[str, int]:
        return {"count": self.count, "bytes": self.bytes}


class BoundedTargetDebugImageWriter:
    """Write at most one JPEG per representative event and obey a hard cap."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        enabled: bool = False,
        max_images_per_run: int = 20,
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if (
            isinstance(max_images_per_run, bool)
            or not isinstance(max_images_per_run, int)
            or max_images_per_run < 0
        ):
            raise ValueError("max_images_per_run must be a non-negative integer")
        self._output_dir = Path(output_dir)
        self._enabled = enabled and max_images_per_run > 0
        self._max_images = max_images_per_run
        self._captured_events: set[str] = set()
        self._count = 0
        self._bytes = 0
        self._lock = RLock()

    @property
    def stats(self) -> TargetDebugImageStats:
        with self._lock:
            return TargetDebugImageStats(
                count=self._count,
                bytes=self._bytes,
                events=tuple(sorted(self._captured_events)),
            )

    def capture(
        self,
        *,
        event: str,
        frame_store: FrameStore,
        frame_ref: FrameRef,
        annotation: TargetDebugAnnotation,
    ) -> bool:
        """Best-effort capture of one whitelisted representative frame.

        False means disabled, duplicate, capped, evicted, Oracle-labelled, or
        an observational encoding/filesystem failure.  Debug output can never
        fail the flight-control path.
        """

        _validate_event(event)
        if not isinstance(frame_store, FrameStore):
            raise TypeError("frame_store must be a FrameStore")
        if not isinstance(frame_ref, FrameRef):
            raise TypeError("frame_ref must be a FrameRef")
        if not isinstance(annotation, TargetDebugAnnotation):
            raise TypeError("annotation must be a TargetDebugAnnotation")
        # Defense in depth: runtime Oracle outputs may never be persisted by
        # this production visual-debug facility.
        if "oracle" in annotation.source.casefold():
            return False
        with self._lock:
            if (
                not self._enabled
                or self._count >= self._max_images
                or event in self._captured_events
            ):
                return False
            rgb = frame_store.get_frame(frame_ref)
            if rgb is None:
                return False
            depth = frame_store.get_depth(frame_ref, copy=False)
            temporary: Path | None = None
            try:
                payload = _annotated_jpeg(rgb, depth, event, annotation)
                self._output_dir.mkdir(parents=True, exist_ok=True)
                destination = self._output_dir / f"{self._count + 1:02d}_{event}.jpg"
                if destination.exists():
                    return False
                temporary = destination.with_suffix(".jpg.tmp")
                # Event names are fixed and capture is serialized, so this is
                # deterministic and never overwrites an earlier event image.
                with temporary.open("xb") as stream:
                    stream.write(payload)
                temporary.replace(destination)
            except Exception:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                return False
            self._captured_events.add(event)
            self._count += 1
            self._bytes += len(payload)
            return True


def _annotated_jpeg(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    event: str,
    annotation: TargetDebugAnnotation,
) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    # The bounded debug image visualizes the same depth cluster represented by
    # the measurement without persisting a depth array.  Pixels near the raw
    # sampled depth are overlaid only inside the detector bbox.
    if (
        depth is not None
        and annotation.raw_depth_m is not None
        and annotation.bbox_xyxy_normalized is not None
        and depth.shape == rgb.shape[:2]
    ):
        x1n, y1n, x2n, y2n = annotation.bbox_xyxy_normalized
        height, width = depth.shape
        x1 = max(0, min(width - 1, int(np.floor(x1n * width))))
        y1 = max(0, min(height - 1, int(np.floor(y1n * height))))
        x2 = max(x1 + 1, min(width, int(np.ceil(x2n * width))))
        y2 = max(y1 + 1, min(height, int(np.ceil(y2n * height))))
        tolerance = max(0.15, 0.05 * annotation.raw_depth_m)
        mask = np.zeros(depth.shape, dtype=bool)
        patch = depth[y1:y2, x1:x2]
        mask[y1:y2, x1:x2] = (
            np.isfinite(patch)
            & (patch > 0.0)
            & (np.abs(patch - annotation.raw_depth_m) <= tolerance)
        )
        if np.any(mask):
            pixels = np.asarray(image).copy()
            pixels[mask] = (
                0.45 * pixels[mask].astype(np.float32)
                + 0.55 * np.asarray((32, 255, 96), dtype=np.float32)
            ).astype(np.uint8)
            image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    bbox = annotation.bbox_xyxy_normalized
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        draw.rectangle(
            (
                round(x1 * (width - 1)),
                round(y1 * (height - 1)),
                round(x2 * (width - 1)),
                round(y2 * (height - 1)),
            ),
            outline=(255, 32, 32),
            width=max(1, min(width, height) // 160 + 1),
        )
    if annotation.sampled_pixel_uv is not None:
        u, v = annotation.sampled_pixel_uv
        radius = max(2, min(width, height) // 120)
        draw.ellipse(
            (round(u) - radius, round(v) - radius, round(u) + radius, round(v) + radius),
            outline=(255, 255, 0),
            width=max(1, radius // 2),
        )
    lines = annotation.label_lines(event)
    line_height = 12
    panel_height = min(height, 4 + line_height * len(lines))
    draw.rectangle((0, 0, width, panel_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        y = 2 + index * line_height
        if y >= height:
            break
        draw.text((3, y), line, fill=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return buffer.getvalue()


def _validate_event(event: object) -> str:
    if not isinstance(event, str) or event not in TARGET_DEBUG_EVENTS:
        raise ValueError("event is not a supported target debug event")
    return event


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


__all__ = [
    "BoundedTargetDebugImageWriter",
    "TARGET_DEBUG_EVENTS",
    "TargetDebugAnnotation",
    "TargetDebugImageStats",
]
