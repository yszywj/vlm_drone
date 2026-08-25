"""Simulator-neutral assembly for deployed-YOLO Isaac training captures.

The module never imports Isaac.  A late-bound script owns the simulator and
passes one atomic :class:`OracleFrameTruth`, synchronized UAV state, and the
response returned by the deployed YOLO tracking worker.  Privileged truth is
used only after inference, solely to construct offline labels and one-to-one
detector/label associations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Sequence

import numpy as np

from datasets.target_state.schema import (
    CameraFrameInput,
    DetectorPrediction,
    SensorInput,
    TargetStateFrameRecord,
    TargetTrainingLabel,
    UavFrameInput,
)
from perception.yolo_client import (
    YoloServiceClient,
    validate_yolo_model_identity,
)
from training.target_state.collector import VerifiedYoloDeployment
from training.yolo.collection_scene import CUBE_CLASS_NAME
from training.yolo.isaac_collector import (
    OracleFrameTruth,
    OracleObjectTruth,
    ProjectionDecision,
    project_oracle_object_bbox,
)
from yolo_service.protocol import TrackDetection, TrackResponse


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite probability")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return result


def _bbox_iou(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def preflight_deployed_yolo(
    client: YoloServiceClient,
    *,
    expected_model_sha256: str,
) -> VerifiedYoloDeployment:
    """Contact health/model-info and return an exact, fail-closed receipt."""

    if not isinstance(client, YoloServiceClient):
        raise TypeError("client must be a YoloServiceClient")
    client.health()
    info = client.model_info()
    validate_yolo_model_identity(
        info,
        expected_model_family="yolo",
        expected_model_names={0: "cube"},
        expected_model_sha256=expected_model_sha256,
        worker_url=client.base_url,
    )
    if info.model_sha256 is None:  # defensive; validator rejects this when expected
        raise RuntimeError("deployed YOLO worker did not return model_sha256")
    return VerifiedYoloDeployment(
        worker_url=client.base_url,
        model_family=info.model_family,
        model_names=info.class_names,
        model_sha256=info.model_sha256,
    )


@dataclass(frozen=True, slots=True)
class DetectionTruthAssociation:
    object_id: str
    detection_index: int
    iou: float


def associate_detections_to_truth(
    detections: Sequence[TrackDetection],
    truth_decisions: Sequence[tuple[OracleObjectTruth, ProjectionDecision]],
    *,
    resolution_wh_px: tuple[int, int],
    minimum_iou: float = 0.1,
) -> tuple[DetectionTruthAssociation, ...]:
    """Greedy deterministic one-to-one association used only for labels."""

    threshold = _probability(minimum_iou, "minimum_iou")
    width, height = resolution_wh_px
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("resolution_wh_px must contain positive integers")
    proposals: list[tuple[float, str, int]] = []
    for truth, decision in truth_decisions:
        if truth.shape != CUBE_CLASS_NAME or decision.label is None:
            continue
        x1, y1, x2, y2 = decision.label.bbox_xyxy_px
        normalized = (x1 / width, y1 / height, x2 / width, y2 / height)
        for index, detection in enumerate(detections):
            if detection.class_id != 0 or detection.class_name.casefold() != "cube":
                continue
            overlap = _bbox_iou(normalized, detection.bbox_xyxy_normalized)
            if overlap >= threshold:
                proposals.append((-overlap, truth.object_id, index))
    used_truth: set[str] = set()
    used_detections: set[int] = set()
    result: list[DetectionTruthAssociation] = []
    for negative_iou, object_id, detection_index in sorted(proposals):
        if object_id in used_truth or detection_index in used_detections:
            continue
        used_truth.add(object_id)
        used_detections.add(detection_index)
        result.append(
            DetectionTruthAssociation(
                object_id=object_id,
                detection_index=detection_index,
                iou=-negative_iou,
            )
        )
    return tuple(result)


@dataclass(slots=True)
class _CandidateMemory:
    candidate_id: str
    bbox: tuple[float, float, float, float]
    timestamp_s: float
    tracker_id: str
    appearance_key: str | None


class DetectorCandidateLinker:
    """Bounded sensor-only continuity across tracker-ID switches."""

    def __init__(self, *, maximum_gap_s: float = 2.0, minimum_iou: float = 0.1) -> None:
        if not isfinite(maximum_gap_s) or maximum_gap_s <= 0.0:
            raise ValueError("maximum_gap_s must be finite and positive")
        self.maximum_gap_s = float(maximum_gap_s)
        self.minimum_iou = _probability(minimum_iou, "minimum_iou")
        self._next_id = 0
        self._by_tracker: dict[str, str] = {}
        self._memories: dict[str, _CandidateMemory] = {}

    def reset(self) -> None:
        self._next_id = 0
        self._by_tracker.clear()
        self._memories.clear()

    def assign_frame(
        self,
        detections: Sequence[TrackDetection],
        *,
        timestamp_s: float,
        appearance_keys: Sequence[str | None] | None = None,
    ) -> tuple[str, ...]:
        timestamp = float(timestamp_s)
        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        appearances = (
            (None,) * len(detections)
            if appearance_keys is None
            else tuple(appearance_keys)
        )
        if len(appearances) != len(detections) or any(
            value is not None
            and (not isinstance(value, str) or not value or value != value.strip())
            for value in appearances
        ):
            raise ValueError("appearance_keys must align with detections")
        assigned: list[str] = []
        used_candidates: set[str] = set()
        for detection, appearance_key in zip(detections, appearances):
            tracker_id = f"track_{detection.track_id}"
            candidate_id = self._by_tracker.get(tracker_id)
            if candidate_id is not None:
                memory = self._memories.get(candidate_id)
                if (
                    memory is None
                    or timestamp - memory.timestamp_s > self.maximum_gap_s
                    or (
                        memory.appearance_key is not None
                        and appearance_key is not None
                        and memory.appearance_key != appearance_key
                    )
                ):
                    self._by_tracker.pop(tracker_id, None)
                    candidate_id = None
            if candidate_id in used_candidates:
                candidate_id = None
            if candidate_id is None:
                matches = [
                    (
                        _bbox_iou(memory.bbox, detection.bbox_xyxy_normalized),
                        memory.timestamp_s,
                        memory.candidate_id,
                    )
                    for memory in self._memories.values()
                    if memory.candidate_id not in used_candidates
                    and 0.0 <= timestamp - memory.timestamp_s <= self.maximum_gap_s
                    and not (
                        memory.appearance_key is not None
                        and appearance_key is not None
                        and memory.appearance_key != appearance_key
                    )
                ]
                matches = [item for item in matches if item[0] >= self.minimum_iou]
                if matches:
                    matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
                    candidate_id = matches[0][2]
                else:
                    self._next_id += 1
                    candidate_id = f"candidate_{self._next_id:06d}"
            self._by_tracker[tracker_id] = candidate_id
            self._memories[candidate_id] = _CandidateMemory(
                candidate_id=candidate_id,
                bbox=detection.bbox_xyxy_normalized,
                timestamp_s=timestamp,
                tracker_id=tracker_id,
                appearance_key=appearance_key,
            )
            used_candidates.add(candidate_id)
            assigned.append(candidate_id)
        expired = [
            candidate_id
            for candidate_id, memory in self._memories.items()
            if timestamp - memory.timestamp_s > self.maximum_gap_s
        ]
        for candidate_id in expired:
            self._memories.pop(candidate_id, None)
            for tracker_id, linked in tuple(self._by_tracker.items()):
                if linked == candidate_id:
                    self._by_tracker.pop(tracker_id, None)
        return tuple(assigned)


class TargetStateFrameAssembler:
    """Build target/no-target records from one synchronized deployed-YOLO frame."""

    def __init__(
        self,
        *,
        uav_id: str = "uav_1",
        minimum_truth_iou: float = 0.1,
        maximum_candidate_gap_s: float = 2.0,
        minimum_bbox_area_px: float = 16.0,
    ) -> None:
        self.uav_id = uav_id
        self.minimum_truth_iou = _probability(minimum_truth_iou, "minimum_truth_iou")
        if not isfinite(minimum_bbox_area_px) or minimum_bbox_area_px <= 0.0:
            raise ValueError("minimum_bbox_area_px must be finite and positive")
        self.minimum_bbox_area_px = float(minimum_bbox_area_px)
        self._linker = DetectorCandidateLinker(
            maximum_gap_s=maximum_candidate_gap_s,
            minimum_iou=minimum_truth_iou,
        )
        self._truth_candidate: dict[str, tuple[str, float]] = {}

    def reset_episode(self) -> None:
        self._linker.reset()
        self._truth_candidate.clear()

    def assemble(
        self,
        *,
        capture_id: str,
        episode_id: str,
        assignment_id: str,
        truth: OracleFrameTruth,
        uav_input: UavFrameInput,
        response: TrackResponse,
    ) -> tuple[TargetStateFrameRecord, ...]:
        if not isinstance(truth, OracleFrameTruth):
            raise TypeError("truth must be OracleFrameTruth")
        if not isinstance(uav_input, UavFrameInput):
            raise TypeError("uav_input must be UavFrameInput")
        if not isinstance(response, TrackResponse):
            raise TypeError("response must be TrackResponse")
        sample = truth.camera_sample
        if abs(response.timestamp_s - sample.timestamp_s) > 1e-9:
            raise ValueError("YOLO response timestamp does not match CameraSample")
        width, height = sample.intrinsics.width, sample.intrinsics.height
        camera = CameraFrameInput(
            fx=sample.intrinsics.fx,
            fy=sample.intrinsics.fy,
            cx=sample.intrinsics.cx,
            cy=sample.intrinsics.cy,
            position_world_m=sample.camera_position_world_m,
            orientation_world_wxyz=sample.camera_orientation_world_wxyz,
            resolution_wh_px=(width, height),
        )
        sensor = SensorInput(
            camera=camera,
            uav=uav_input,
            rgb_path=f"pending/{capture_id}.jpg",
            depth_path=f"pending/{capture_id}.npy",
        )
        detections = tuple(response.detections)
        appearance_keys = tuple(
            self._appearance_key(sample.rgb, detection)
            for detection in detections
        )
        candidate_ids = self._linker.assign_frame(
            detections,
            timestamp_s=sample.timestamp_s,
            appearance_keys=appearance_keys,
        )

        projected: list[tuple[OracleObjectTruth, ProjectionDecision, tuple[float, float, float, float] | None]] = []
        for obj in truth.objects:
            if obj.shape != CUBE_CLASS_NAME:
                continue
            decision = project_oracle_object_bbox(
                sample,
                obj,
                class_id=0,
                min_bbox_area_px=self.minimum_bbox_area_px,
            )
            normalized = None
            if decision.label is not None:
                x1, y1, x2, y2 = decision.label.bbox_xyxy_px
                normalized = (x1 / width, y1 / height, x2 / width, y2 / height)
            projected.append((obj, decision, normalized))

        proposals: list[tuple[float, str, int]] = []
        for obj, _decision, normalized in projected:
            if normalized is None:
                continue
            for detection_index, detection in enumerate(detections):
                if detection.class_id != 0 or detection.class_name.casefold() != "cube":
                    continue
                overlap = _bbox_iou(normalized, detection.bbox_xyxy_normalized)
                if overlap >= self.minimum_truth_iou:
                    proposals.append((-overlap, obj.object_id, detection_index))
        associations: dict[str, int] = {}
        used_detections: set[int] = set()
        for _negative_iou, object_id, detection_index in sorted(proposals):
            if object_id in associations or detection_index in used_detections:
                continue
            associations[object_id] = detection_index
            used_detections.add(detection_index)

        records: list[TargetStateFrameRecord] = []
        for target_index, (obj, decision, normalized) in enumerate(projected):
            detection_index = associations.get(obj.object_id)
            if detection_index is None:
                remembered = self._truth_candidate.get(obj.object_id)
                candidate_id = (
                    remembered[0]
                    if remembered is not None
                    and sample.timestamp_s - remembered[1]
                    <= self._linker.maximum_gap_s
                    else None
                )
                detector = DetectorPrediction(False, None, None, None, candidate_id)
            else:
                detection = detections[detection_index]
                candidate_id = candidate_ids[detection_index]
                self._truth_candidate[obj.object_id] = (
                    candidate_id,
                    sample.timestamp_s,
                )
                detector = DetectorPrediction(
                    True,
                    detection.bbox_xyxy_normalized,
                    detection.confidence,
                    f"track_{detection.track_id}",
                    candidate_id,
                )
            visible = decision.label is not None
            center = None
            if visible and decision.label is not None:
                x1, y1, x2, y2 = decision.label.bbox_xyxy_px
                projected_center = obj.center_pixel_uv
                center = (
                    projected_center
                    if projected_center is not None
                    and 0.0 <= projected_center[0] < width
                    and 0.0 <= projected_center[1] < height
                    else ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                )
            label = TargetTrainingLabel(
                position_world_m=obj.position_world_m,
                velocity_world_mps=obj.velocity_world_mps,
                center_pixel_uv=center,
                visible=visible,
                occlusion_ratio=(
                    1.0
                    if decision.occlusion_ratio is None and not visible
                    else float(decision.occlusion_ratio or 0.0)
                ),
                color_name=obj.color_name,
                instance_id=obj.object_id,
            )
            records.append(
                TargetStateFrameRecord(
                    frame_id=f"{capture_id}_target_{target_index}",
                    episode_id=episode_id,
                    assignment_id=assignment_id,
                    uav_id=self.uav_id,
                    timestamp_s=sample.timestamp_s,
                    sensor_input=sensor,
                    detector_prediction=detector,
                    training_label=label,
                )
            )

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            records.append(
                TargetStateFrameRecord(
                    frame_id=f"{capture_id}_false_positive_{detection_index}",
                    episode_id=episode_id,
                    assignment_id=assignment_id,
                    uav_id=self.uav_id,
                    timestamp_s=sample.timestamp_s,
                    sensor_input=sensor,
                    detector_prediction=DetectorPrediction(
                        True,
                        detection.bbox_xyxy_normalized,
                        detection.confidence,
                        f"track_{detection.track_id}",
                        candidate_ids[detection_index],
                    ),
                    training_label=None,
                )
            )
        if not projected and not detections:
            records.append(
                TargetStateFrameRecord(
                    frame_id=f"{capture_id}_no_target",
                    episode_id=episode_id,
                    assignment_id=assignment_id,
                    uav_id=self.uav_id,
                    timestamp_s=sample.timestamp_s,
                    sensor_input=sensor,
                    detector_prediction=DetectorPrediction(
                        False, None, None, None, None
                    ),
                    training_label=None,
                )
            )
        return tuple(records)

    @staticmethod
    def _appearance_key(rgb: object, detection: TrackDetection) -> str | None:
        """Return a conservative red/blue key from detector pixels only."""

        image = np.asarray(rgb)
        height, width = image.shape[:2]
        x1, y1, x2, y2 = detection.bbox_xyxy_normalized
        left = max(0, min(width - 1, int(round(x1 * width))))
        top = max(0, min(height - 1, int(round(y1 * height))))
        right = max(left + 1, min(width, int(round(x2 * width))))
        bottom = max(top + 1, min(height, int(round(y2 * height))))
        inset_x = max(0, int(round((right - left) * 0.15)))
        inset_y = max(0, int(round((bottom - top) * 0.15)))
        roi = image[
            top + inset_y : max(top + inset_y + 1, bottom - inset_y),
            left + inset_x : max(left + inset_x + 1, right - inset_x),
        ].astype(np.float32)
        if roi.size == 0:
            return None
        channels = np.median(roi.reshape(-1, 3), axis=0)
        red, green, blue = (float(value) for value in channels)
        if red >= 50.0 and red >= 1.35 * max(green, blue, 1.0):
            return "red"
        if blue >= 50.0 and blue >= 1.35 * max(red, green, 1.0):
            return "blue"
        return None


__all__ = [
    "DetectionTruthAssociation",
    "DetectorCandidateLinker",
    "TargetStateFrameAssembler",
    "associate_detections_to_truth",
    "preflight_deployed_yolo",
]
