"""Pure-NumPy RGB-D color evidence and bounded temporal accumulation."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from threading import RLock

import numpy as np

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from env.camera_types import CameraSample
from perception.attribute_types import (
    AttributeDecision,
    AttributeEvidence,
    AttributeObservation,
    AttributeRequirement,
    AttributeVerificationBundle,
)
from perception.attribute_verifier import (
    AttributeFrameUnavailable,
    AttributeRouteMismatch,
    AttributeTimeError,
    AttributeVerificationRoute,
)
from perception.runtime import PerceptionRuntimeProfile
from runtime.frame_store import FrameRef, FrameStore
from yolo_service.protocol import TrackDetection


COLOR_EVIDENCE_SOURCE = "hsv_depth_mask"


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _ratio(value: object, name: str, *, upper_inclusive: bool = True) -> float:
    result = _finite(value, name, minimum=0.0)
    if result > 1.0 or (not upper_inclusive and result == 1.0):
        bound = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{name} must be within {bound}")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _canonical_tracker_id(value: str | int) -> str:
    if isinstance(value, bool):
        raise TypeError("tracker_id must be an integer or string")
    if isinstance(value, Integral):
        if int(value) < 0:
            raise ValueError("tracker_id must be non-negative")
        return str(int(value))
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return str(int(value))
    return validate_routing_id(value, "tracker_id")


def _normalize_hue_ranges(
    value: Mapping[str, Sequence[Sequence[float]]] | None,
) -> dict[str, tuple[tuple[float, float], ...]]:
    raw = value or {
        "red": ((0.0, 20.0), (340.0, 360.0)),
        "blue": ((190.0, 260.0),),
    }
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError("hue_ranges_deg must be a non-empty mapping")
    normalized: dict[str, tuple[tuple[float, float], ...]] = {}
    for name, ranges in raw.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("hue range names must be non-empty strings")
        canonical = name.casefold()
        if canonical in normalized:
            raise ValueError("hue_ranges_deg contains duplicate color names")
        if isinstance(ranges, (str, bytes)) or not isinstance(ranges, Sequence):
            raise TypeError("each hue range collection must be a sequence")
        converted: list[tuple[float, float]] = []
        for index, bounds in enumerate(ranges):
            if (
                isinstance(bounds, (str, bytes))
                or not isinstance(bounds, Sequence)
                or len(bounds) != 2
            ):
                raise ValueError(
                    f"hue range {canonical}[{index}] must contain two numbers"
                )
            lower = _finite(bounds[0], "hue lower", minimum=0.0)
            upper = _finite(bounds[1], "hue upper", minimum=0.0)
            if lower >= upper or upper > 360.0:
                raise ValueError("hue ranges must satisfy 0 <= lower < upper <= 360")
            converted.append((lower, upper))
        if not converted:
            raise ValueError("each color must have at least one hue range")
        normalized[canonical] = tuple(converted)
    intervals = [
        (lower, upper, color)
        for color, ranges in normalized.items()
        for lower, upper in ranges
    ]
    for index, (left_lower, left_upper, left_color) in enumerate(intervals):
        for right_lower, right_upper, right_color in intervals[index + 1 :]:
            if max(left_lower, right_lower) < min(left_upper, right_upper):
                raise ValueError(
                    "hue_ranges_deg intervals must not overlap: "
                    f"{left_color} and {right_color}"
                )
    return normalized


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hue in degrees and saturation/value in [0, 1]."""

    normalized = rgb.astype(np.float32, copy=False) / np.float32(255.0)
    red = normalized[..., 0]
    green = normalized[..., 1]
    blue = normalized[..., 2]
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    delta = maximum - minimum

    saturation = np.zeros_like(maximum)
    nonblack = maximum > 0.0
    saturation[nonblack] = delta[nonblack] / maximum[nonblack]
    hue = np.zeros_like(maximum)
    chromatic = delta > 0.0
    red_max = chromatic & (maximum == red)
    green_max = chromatic & (maximum == green)
    blue_max = chromatic & (maximum == blue)
    hue[red_max] = 60.0 * np.mod(
        (green[red_max] - blue[red_max]) / delta[red_max],
        6.0,
    )
    hue[green_max] = 60.0 * (
        (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    )
    hue[blue_max] = 60.0 * (
        (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    )
    return hue, saturation, maximum


class RgbdColorAttributeVerifier:
    """Measure red/blue color inside a tracker box from one atomic RGB-D frame."""

    def __init__(
        self,
        frame_store: FrameStore | None = None,
        *,
        supported_values: Sequence[str] = ("red", "blue"),
        roi_inset_ratio: float = 0.12,
        min_valid_pixel_ratio: float = 0.15,
        min_saturation: float = 0.25,
        min_value: float = 0.15,
        min_dominant_fraction: float = 0.55,
        min_score_margin: float = 0.15,
        depth_absolute_tolerance_m: float = 0.25,
        depth_relative_tolerance: float = 0.05,
        min_bbox_area_px: int = 64,
        depth_anchor_radius_px: int = 2,
        hue_ranges_deg: Mapping[str, Sequence[Sequence[float]]] | None = None,
    ) -> None:
        if frame_store is not None and not isinstance(frame_store, FrameStore):
            raise TypeError("frame_store must be a FrameStore or None")
        if isinstance(supported_values, (str, bytes)) or not isinstance(
            supported_values, Sequence
        ):
            raise TypeError("supported_values must be a sequence")
        supported = tuple(
            str(value).casefold()
            if isinstance(value, str) and value and value == value.strip()
            else (_ for _ in ()).throw(
                ValueError("supported_values entries must be non-empty strings")
            )
            for value in supported_values
        )
        if not supported or len(set(supported)) != len(supported):
            raise ValueError("supported_values must be non-empty and unique")
        ranges = _normalize_hue_ranges(hue_ranges_deg)
        if any(value not in ranges for value in supported):
            raise ValueError("every supported color requires hue_ranges_deg")

        self._frame_store = frame_store
        self._supported_values = supported
        self._roi_inset_ratio = _ratio(
            roi_inset_ratio, "roi_inset_ratio", upper_inclusive=False
        )
        if self._roi_inset_ratio >= 0.5:
            raise ValueError("roi_inset_ratio must be less than 0.5")
        self._min_valid_ratio = _ratio(
            min_valid_pixel_ratio, "min_valid_pixel_ratio"
        )
        self._min_saturation = _ratio(min_saturation, "min_saturation")
        self._min_value = _ratio(min_value, "min_value")
        self._min_dominant = _ratio(
            min_dominant_fraction, "min_dominant_fraction"
        )
        self._min_margin = _ratio(min_score_margin, "min_score_margin")
        self._absolute_tolerance = _finite(
            depth_absolute_tolerance_m,
            "depth_absolute_tolerance_m",
            minimum=0.0,
        )
        self._relative_tolerance = _ratio(
            depth_relative_tolerance, "depth_relative_tolerance"
        )
        if self._absolute_tolerance == 0.0 and self._relative_tolerance == 0.0:
            raise ValueError("at least one depth tolerance must be positive")
        self._min_bbox_area = _positive_int(min_bbox_area_px, "min_bbox_area_px")
        self._anchor_radius = _nonnegative_int(
            depth_anchor_radius_px, "depth_anchor_radius_px"
        )
        self._hue_ranges = ranges

    @property
    def supported_values(self) -> tuple[str, ...]:
        return self._supported_values

    @property
    def frame_store(self) -> FrameStore | None:
        return self._frame_store

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        frame_store: FrameStore | None = None,
        min_bbox_area_px: int = 64,
    ) -> "RgbdColorAttributeVerifier":
        """Build from ``TargetColorAttributeConfig`` without storing config state."""

        from configs.schema import TargetColorAttributeConfig

        if not isinstance(config, TargetColorAttributeConfig):
            raise TypeError("config must be a TargetColorAttributeConfig")
        if not config.enabled:
            raise ValueError("color attribute verification is disabled")
        if config.method != "hsv_depth_mask":
            raise ValueError("only hsv_depth_mask color verification is supported")
        return cls(
            frame_store,
            supported_values=config.supported_values,
            roi_inset_ratio=config.roi_inset_ratio,
            min_valid_pixel_ratio=config.min_valid_pixel_ratio,
            min_saturation=config.min_saturation,
            min_value=config.min_value,
            min_dominant_fraction=config.min_dominant_fraction,
            min_score_margin=config.min_score_margin,
            depth_absolute_tolerance_m=config.depth_absolute_tolerance_m,
            depth_relative_tolerance=config.depth_relative_tolerance,
            min_bbox_area_px=min_bbox_area_px,
            hue_ranges_deg=config.hue_ranges_deg,
        )

    def verify(
        self,
        *,
        requirement: AttributeRequirement,
        detection: TrackDetection,
        route: AttributeVerificationRoute,
        frame_ref: FrameRef | None = None,
        camera_sample: CameraSample | None = None,
    ) -> AttributeObservation:
        if not isinstance(requirement, AttributeRequirement):
            raise TypeError("requirement must be an AttributeRequirement")
        if not isinstance(detection, TrackDetection):
            raise TypeError("detection must be a TrackDetection")
        if not isinstance(route, AttributeVerificationRoute):
            raise TypeError("route must be an AttributeVerificationRoute")
        route.validate(requirement, detection)
        if (frame_ref is None) == (camera_sample is None):
            raise ValueError("provide exactly one of frame_ref or camera_sample")
        if _canonical_tracker_id(detection.track_id) != requirement.tracker_id:
            raise AttributeRouteMismatch(
                "detection tracker_id does not match AttributeRequirement"
            )

        if frame_ref is not None:
            if not isinstance(frame_ref, FrameRef):
                raise TypeError("frame_ref must be a FrameRef")
            if frame_ref.uav_id != requirement.uav_id:
                raise AttributeRouteMismatch(
                    "FrameRef uav_id does not match AttributeRequirement"
                )
            if self._frame_store is None:
                raise AttributeFrameUnavailable(
                    "FrameRef verification requires the owning FrameStore"
                )
            synchronized_sample = self._frame_store.get_camera_sample(frame_ref)
            timestamp_s = frame_ref.timestamp_s
            width, height = frame_ref.width, frame_ref.height
            if synchronized_sample is None:
                return self._pending(
                    requirement,
                    timestamp_s,
                    "frame_unavailable",
                )
            rgb = synchronized_sample.rgb
            depth = synchronized_sample.depth_to_image_plane_m
        else:
            if not isinstance(camera_sample, CameraSample):
                raise TypeError("camera_sample must be a CameraSample")
            # A single CameraSample owns both channels and is the only direct
            # input accepted; independently supplied arrays are impossible.
            rgb = camera_sample.rgb
            depth = camera_sample.depth_to_image_plane_m
            timestamp_s = camera_sample.timestamp_s
            width = camera_sample.intrinsics.width
            height = camera_sample.intrinsics.height

        if requirement.attribute_name != "color":
            return self._unsupported(requirement, timestamp_s, "unsupported_attribute")
        if requirement.expected_value not in self._supported_values:
            return self._unsupported(requirement, timestamp_s, "unsupported_value")
        if depth is None:
            return self._pending(requirement, timestamp_s, "depth_unavailable")
        if rgb.shape != (height, width, 3) or depth.shape != (height, width):
            raise AttributeFrameUnavailable(
                "RGB and depth resolution must match their synchronized sample"
            )

        x1n, y1n, x2n, y2n = detection.bbox_xyxy_normalized
        x1 = max(0, min(width, int(np.floor(x1n * width))))
        y1 = max(0, min(height, int(np.floor(y1n * height))))
        x2 = max(0, min(width, int(np.ceil(x2n * width))))
        y2 = max(0, min(height, int(np.ceil(y2n * height))))
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height <= 0:
            return self._pending(requirement, timestamp_s, "bbox_too_small")
        inset_x = int(np.floor(box_width * self._roi_inset_ratio))
        inset_y = int(np.floor(box_height * self._roi_inset_ratio))
        ix1, iy1 = x1 + inset_x, y1 + inset_y
        ix2, iy2 = x2 - inset_x, y2 - inset_y
        roi_width, roi_height = ix2 - ix1, iy2 - iy1
        if roi_width <= 0 or roi_height <= 0 or roi_width * roi_height < self._min_bbox_area:
            return self._pending(requirement, timestamp_s, "bbox_too_small")

        center_x = min(width - 1, max(0, (x1 + x2 - 1) // 2))
        center_y = min(height - 1, max(0, (y1 + y2 - 1) // 2))
        anchor_x1 = max(x1, center_x - self._anchor_radius)
        anchor_x2 = min(x2, center_x + self._anchor_radius + 1)
        anchor_y1 = max(y1, center_y - self._anchor_radius)
        anchor_y2 = min(y2, center_y + self._anchor_radius + 1)
        anchor_patch = depth[anchor_y1:anchor_y2, anchor_x1:anchor_x2]
        finite_anchor = anchor_patch[np.isfinite(anchor_patch) & (anchor_patch > 0.0)]
        if finite_anchor.size == 0:
            return self._pending(requirement, timestamp_s, "depth_anchor_unavailable")
        anchor_depth = float(np.median(finite_anchor))
        tolerance = max(
            self._absolute_tolerance,
            self._relative_tolerance * anchor_depth,
        )

        roi_rgb = rgb[iy1:iy2, ix1:ix2]
        roi_depth = depth[iy1:iy2, ix1:ix2]
        depth_mask = (
            np.isfinite(roi_depth)
            & (roi_depth > 0.0)
            & (np.abs(roi_depth - anchor_depth) <= tolerance)
        )
        hue, saturation, value = _rgb_to_hsv(roi_rgb)
        chromatic_mask = (saturation >= self._min_saturation) & (
            value >= self._min_value
        )
        valid_mask = depth_mask & chromatic_mask
        total_pixels = int(valid_mask.size)
        valid_pixels = int(np.count_nonzero(valid_mask))
        valid_ratio = 0.0 if total_pixels == 0 else valid_pixels / total_pixels
        if valid_pixels == 0 or valid_ratio < self._min_valid_ratio:
            depth_ratio = (
                0.0
                if total_pixels == 0
                else int(np.count_nonzero(depth_mask)) / total_pixels
            )
            reason = (
                "achromatic_or_dark"
                if depth_ratio >= self._min_valid_ratio
                else "insufficient_valid_pixels"
            )
            return self._pending(
                requirement,
                timestamp_s,
                reason,
                observed_value="unknown",
                valid_sample_ratio=valid_ratio,
            )

        scores: dict[str, float] = {}
        covered = np.zeros_like(valid_mask, dtype=bool)
        for color, ranges in self._hue_ranges.items():
            color_mask = np.zeros_like(valid_mask, dtype=bool)
            for lower, upper in ranges:
                # Half-open intervals avoid double counting adjacent bins;
                # 360 is represented as 0 by the HSV conversion.
                color_mask |= (hue >= lower) & (hue < upper)
            color_mask &= valid_mask
            covered |= color_mask
            scores[color] = int(np.count_nonzero(color_mask)) / valid_pixels
        scores["other"] = int(np.count_nonzero(valid_mask & ~covered)) / valid_pixels
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        dominant_name, dominant_fraction = ordered[0]
        runner_up = 0.0 if len(ordered) == 1 else ordered[1][1]
        margin = dominant_fraction - runner_up
        if (
            dominant_fraction < self._min_dominant
            or margin < self._min_margin
        ):
            return self._pending(
                requirement,
                timestamp_s,
                "ambiguous_color",
                observed_value="unknown",
                confidence=dominant_fraction,
                valid_sample_ratio=valid_ratio,
            )

        decision = (
            AttributeDecision.MATCH
            if dominant_name == requirement.expected_value
            else AttributeDecision.MISMATCH
        )
        return AttributeObservation(
            mission_id=requirement.mission_id,
            uav_id=requirement.uav_id,
            assignment_id=requirement.assignment_id,
            candidate_id=requirement.candidate_id,
            tracker_id=requirement.tracker_id,
            timestamp_s=timestamp_s,
            attribute_name=requirement.attribute_name,
            expected_value=requirement.expected_value,
            observed_value=dominant_name,
            decision=decision,
            confidence=dominant_fraction,
            observation_count=1,
            duration_s=0.0,
            valid_sample_ratio=valid_ratio,
            source=COLOR_EVIDENCE_SOURCE,
            reason_code="single_frame_color",
        )

    @staticmethod
    def _pending(
        requirement: AttributeRequirement,
        timestamp_s: float,
        reason: str,
        *,
        observed_value: str | None = None,
        confidence: float = 0.0,
        valid_sample_ratio: float = 0.0,
    ) -> AttributeObservation:
        return AttributeObservation(
            mission_id=requirement.mission_id,
            uav_id=requirement.uav_id,
            assignment_id=requirement.assignment_id,
            candidate_id=requirement.candidate_id,
            tracker_id=requirement.tracker_id,
            timestamp_s=timestamp_s,
            attribute_name=requirement.attribute_name,
            expected_value=requirement.expected_value,
            observed_value=observed_value,
            decision=AttributeDecision.PENDING,
            confidence=confidence,
            observation_count=1,
            duration_s=0.0,
            valid_sample_ratio=valid_sample_ratio,
            source=COLOR_EVIDENCE_SOURCE,
            reason_code=reason,
        )

    @staticmethod
    def _unsupported(
        requirement: AttributeRequirement,
        timestamp_s: float,
        reason: str,
    ) -> AttributeObservation:
        return AttributeObservation(
            mission_id=requirement.mission_id,
            uav_id=requirement.uav_id,
            assignment_id=requirement.assignment_id,
            candidate_id=requirement.candidate_id,
            tracker_id=requirement.tracker_id,
            timestamp_s=timestamp_s,
            attribute_name=requirement.attribute_name,
            expected_value=requirement.expected_value,
            observed_value=None,
            decision=AttributeDecision.UNSUPPORTED,
            confidence=0.0,
            observation_count=1,
            duration_s=0.0,
            valid_sample_ratio=0.0,
            source=COLOR_EVIDENCE_SOURCE,
            reason_code=reason,
        )


@dataclass(slots=True)
class _TemporalHistory:
    tracker_id: str
    observations: deque[AttributeObservation]


class TemporalColorEvidenceAccumulator:
    """Turn single-frame color observations into bounded temporal evidence."""

    def __init__(
        self,
        *,
        min_observations: int = 3,
        min_duration_s: float = 0.4,
        max_history_per_candidate: int = 16,
        min_dominant_fraction: float = 0.55,
        min_score_margin: float = 0.15,
        max_candidates: int = 64,
    ) -> None:
        self._min_observations = _positive_int(
            min_observations, "min_observations"
        )
        self._min_duration = _finite(
            min_duration_s, "min_duration_s", minimum=0.0
        )
        self._max_history = _positive_int(
            max_history_per_candidate, "max_history_per_candidate"
        )
        if self._max_history < self._min_observations:
            raise ValueError(
                "max_history_per_candidate must be at least min_observations"
            )
        self._min_dominant = _ratio(
            min_dominant_fraction, "min_dominant_fraction"
        )
        self._min_margin = _ratio(min_score_margin, "min_score_margin")
        self._max_candidates = _positive_int(max_candidates, "max_candidates")
        self._histories: dict[
            tuple[str, str, str, str, str, str], _TemporalHistory
        ] = {}
        self._last_timestamp_by_candidate: dict[
            tuple[str, str, str, str, str, str], float
        ] = {}
        self._active_mission_by_uav: dict[str, str] = {}
        self._lock = RLock()

    @classmethod
    def from_config(cls, config: object) -> "TemporalColorEvidenceAccumulator":
        from configs.schema import TargetColorAttributeConfig

        if not isinstance(config, TargetColorAttributeConfig):
            raise TypeError("config must be a TargetColorAttributeConfig")
        if not config.enabled:
            raise ValueError("color attribute verification is disabled")
        return cls(
            min_observations=config.min_observations,
            min_duration_s=config.min_duration_s,
            max_history_per_candidate=config.max_history_per_candidate,
            min_dominant_fraction=config.min_dominant_fraction,
            min_score_margin=config.min_score_margin,
        )

    @staticmethod
    def _key(
        value: AttributeRequirement | AttributeObservation,
    ) -> tuple[str, str, str, str, str, str]:
        return (
            value.mission_id,
            value.uav_id,
            value.assignment_id,
            value.candidate_id,
            value.attribute_name,
            value.expected_value,
        )

    def update(self, observation: AttributeObservation) -> AttributeEvidence:
        if not isinstance(observation, AttributeObservation):
            raise TypeError("observation must be an AttributeObservation")
        if observation.runtime_profile is not PerceptionRuntimeProfile.PRODUCTION:
            raise PermissionError(
                "production temporal color evidence rejects non-production provenance"
            )
        if observation.source != COLOR_EVIDENCE_SOURCE:
            raise ValueError(
                "TemporalColorEvidenceAccumulator only accepts hsv_depth_mask source"
            )
        key = self._key(observation)
        with self._lock:
            active_mission = self._active_mission_by_uav.get(observation.uav_id)
            if active_mission is not None and active_mission != observation.mission_id:
                raise AttributeRouteMismatch(
                    "observation mission_id is not the active UAV mission; "
                    "call begin_mission before accepting a new mission"
                )
            if active_mission is None:
                self._active_mission_by_uav[observation.uav_id] = observation.mission_id

            previous_timestamp = self._last_timestamp_by_candidate.get(key)
            if (
                previous_timestamp is not None
                and observation.timestamp_s <= previous_timestamp
            ):
                raise AttributeTimeError(
                    "attribute timestamps must increase strictly per candidate epoch"
                )
            self._last_timestamp_by_candidate[key] = observation.timestamp_s

            history = self._histories.get(key)
            if history is not None and history.tracker_id != observation.tracker_id:
                # A BoT-SORT ID switch creates a new color-evidence epoch.  No
                # vote from the previous tracker may confirm the new one.
                history = None
                self._histories.pop(key, None)
            if history is None:
                if len(self._histories) >= self._max_candidates:
                    oldest = min(
                        self._histories,
                        key=lambda item: (
                            self._histories[item].observations[-1].timestamp_s,
                            item,
                        ),
                    )
                    self._histories.pop(oldest, None)
                    self._last_timestamp_by_candidate.pop(oldest, None)
                history = _TemporalHistory(
                    tracker_id=observation.tracker_id,
                    observations=deque(maxlen=self._max_history),
                )
                self._histories[key] = history
            history.observations.append(observation)
            return self._build_evidence(tuple(history.observations))

    def _build_evidence(
        self,
        observations: tuple[AttributeObservation, ...],
    ) -> AttributeEvidence:
        latest = observations[-1]
        duration = latest.timestamp_s - observations[0].timestamp_s
        valid_ratio = sum(item.valid_sample_ratio for item in observations) / len(
            observations
        )
        if latest.decision is AttributeDecision.UNSUPPORTED:
            return self._evidence_from(
                latest,
                observations,
                decision=AttributeDecision.UNSUPPORTED,
                observed_value=None,
                confidence=0.0,
                duration_s=duration,
                valid_sample_ratio=valid_ratio,
                reason_code="attribute_unsupported",
            )

        decisive = tuple(
            item
            for item in observations
            if item.decision in {AttributeDecision.MATCH, AttributeDecision.MISMATCH}
            and item.observed_value not in {None, "unknown"}
        )
        if not decisive:
            return self._evidence_from(
                latest,
                observations,
                decision=AttributeDecision.PENDING,
                observed_value="unknown",
                confidence=0.0,
                duration_s=duration,
                valid_sample_ratio=valid_ratio,
                reason_code="no_decisive_color_evidence",
            )

        counts: dict[str, int] = {}
        confidence_sums: dict[str, float] = {}
        for item in decisive:
            assert item.observed_value is not None
            counts[item.observed_value] = counts.get(item.observed_value, 0) + 1
            confidence_sums[item.observed_value] = (
                confidence_sums.get(item.observed_value, 0.0) + item.confidence
            )
        ranked = sorted(counts, key=lambda name: (-counts[name], name))
        dominant = ranked[0]
        dominant_fraction = counts[dominant] / len(decisive)
        runner_fraction = 0.0 if len(ranked) == 1 else counts[ranked[1]] / len(decisive)
        confidence = min(
            1.0,
            (confidence_sums[dominant] / counts[dominant]) * dominant_fraction,
        )
        decisive_duration = decisive[-1].timestamp_s - decisive[0].timestamp_s
        shortfalls: list[str] = []
        if len(decisive) < self._min_observations:
            shortfalls.append("min_observations")
        if decisive_duration + 1e-12 < self._min_duration:
            shortfalls.append("min_duration")
        if dominant_fraction < self._min_dominant:
            shortfalls.append("dominant_fraction")
        if dominant_fraction - runner_fraction < self._min_margin:
            shortfalls.append("score_margin")
        if shortfalls:
            return self._evidence_from(
                latest,
                observations,
                decision=AttributeDecision.PENDING,
                observed_value=dominant,
                confidence=confidence,
                duration_s=duration,
                valid_sample_ratio=valid_ratio,
                reason_code="insufficient_temporal_evidence",
            )

        decision = (
            AttributeDecision.MATCH
            if dominant == latest.expected_value
            else AttributeDecision.MISMATCH
        )
        return self._evidence_from(
            latest,
            observations,
            decision=decision,
            observed_value=dominant,
            confidence=confidence,
            duration_s=duration,
            valid_sample_ratio=valid_ratio,
            reason_code="temporal_color_stable",
        )

    @staticmethod
    def _evidence_from(
        latest: AttributeObservation,
        observations: tuple[AttributeObservation, ...],
        *,
        decision: AttributeDecision,
        observed_value: str | None,
        confidence: float,
        duration_s: float,
        valid_sample_ratio: float,
        reason_code: str,
    ) -> AttributeEvidence:
        return AttributeEvidence(
            mission_id=latest.mission_id,
            uav_id=latest.uav_id,
            assignment_id=latest.assignment_id,
            candidate_id=latest.candidate_id,
            tracker_id=latest.tracker_id,
            timestamp_s=latest.timestamp_s,
            attribute_name=latest.attribute_name,
            expected_value=latest.expected_value,
            observed_value=observed_value,
            decision=decision,
            confidence=confidence,
            observation_count=len(observations),
            duration_s=duration_s,
            valid_sample_ratio=valid_sample_ratio,
            source=COLOR_EVIDENCE_SOURCE,
            reason_code=reason_code,
            runtime_profile=latest.runtime_profile,
        )

    def bundle(self, requirement: AttributeRequirement) -> AttributeVerificationBundle:
        if not isinstance(requirement, AttributeRequirement):
            raise TypeError("requirement must be an AttributeRequirement")
        key = self._key(requirement)
        with self._lock:
            history = self._histories.get(key)
            if history is None:
                raise ValueError("no retained evidence for AttributeRequirement")
            if history.tracker_id != requirement.tracker_id:
                raise AttributeRouteMismatch(
                    "requirement tracker_id does not match the current evidence epoch"
                )
            observations = tuple(history.observations)
            evidence = self._build_evidence(observations)
        return AttributeVerificationBundle(requirement, observations, evidence)

    def history(
        self,
        requirement: AttributeRequirement,
    ) -> tuple[AttributeObservation, ...]:
        if not isinstance(requirement, AttributeRequirement):
            raise TypeError("requirement must be an AttributeRequirement")
        with self._lock:
            history = self._histories.get(self._key(requirement))
            if history is None or history.tracker_id != requirement.tracker_id:
                return ()
            return tuple(history.observations)

    def reset(
        self,
        *,
        mission_id: str | None = None,
        uav_id: str | None = None,
        assignment_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        normalized_mission = (
            None if mission_id is None else validate_mission_id(mission_id)
        )
        normalized_uav = None if uav_id is None else validate_uav_id(uav_id)
        normalized_assignment = (
            None
            if assignment_id is None
            else validate_routing_id(assignment_id, "assignment_id")
        )
        normalized_candidate = (
            None
            if candidate_id is None
            else validate_routing_id(candidate_id, "candidate_id")
        )
        with self._lock:
            self._reset_locked(
                mission_id=normalized_mission,
                uav_id=normalized_uav,
                assignment_id=normalized_assignment,
                candidate_id=normalized_candidate,
            )

    def begin_mission(self, *, mission_id: str, uav_id: str) -> None:
        """Atomically clear one UAV's old histories and bind its new epoch."""

        normalized_mission = validate_mission_id(mission_id)
        normalized_uav = validate_uav_id(uav_id)
        with self._lock:
            active = self._active_mission_by_uav.get(normalized_uav)
            if active == normalized_mission:
                return
            self._reset_locked(uav_id=normalized_uav)
            self._active_mission_by_uav[normalized_uav] = normalized_mission

    def _reset_locked(
        self,
        *,
        mission_id: str | None = None,
        uav_id: str | None = None,
        assignment_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        def selected(key: tuple[str, str, str, str, str, str]) -> bool:
            return (
                (mission_id is None or key[0] == mission_id)
                and (uav_id is None or key[1] == uav_id)
                and (assignment_id is None or key[2] == assignment_id)
                and (candidate_id is None or key[3] == candidate_id)
            )

        for key in tuple(self._histories):
            if selected(key):
                self._histories.pop(key, None)
                self._last_timestamp_by_candidate.pop(key, None)
        if mission_id is None and assignment_id is None and candidate_id is None:
            if uav_id is None:
                self._active_mission_by_uav.clear()
            else:
                self._active_mission_by_uav.pop(uav_id, None)

    def reject_candidate(
        self,
        *,
        mission_id: str,
        uav_id: str,
        assignment_id: str,
        candidate_id: str,
    ) -> None:
        self.reset(
            mission_id=mission_id,
            uav_id=uav_id,
            assignment_id=assignment_id,
            candidate_id=candidate_id,
        )


__all__ = [
    "COLOR_EVIDENCE_SOURCE",
    "RgbdColorAttributeVerifier",
    "TemporalColorEvidenceAccumulator",
]
