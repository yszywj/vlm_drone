"""PyTorch input pipeline for leakage-safe temporal target-state training.

Privileged labels are read only in this training package.  The feature tensor
contains sensor state and recorded detector output; labels are returned in
separate tensors used by the loss/evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from datasets.target_state.dataset import read_frame_records, split_for_episode
from datasets.target_state.schema import TargetStateFrameRecord
from datasets.target_state.sequence import TargetStateSequence, build_sequences
from training.target_state.config import TargetStateTrainingConfig, TrainingStage


GEOMETRY_INPUT_FIELDS = (
    "bbox_x1_normalized", "bbox_y1_normalized", "bbox_x2_normalized",
    "bbox_y2_normalized", "detector_confidence", "tracker_continuity",
    "fx_normalized", "fy_normalized", "cx_normalized", "cy_normalized",
    "camera_relative_x_m", "camera_relative_y_m", "camera_relative_z_m",
    "camera_relative_orientation_w", "camera_relative_orientation_x",
    "camera_relative_orientation_y", "camera_relative_orientation_z", "uav_velocity_x_mps",
    "uav_velocity_y_mps", "uav_velocity_z_mps", "uav_angular_x_radps",
    "uav_angular_y_radps", "uav_angular_z_radps", "delta_t_s", "missing",
)

if len(GEOMETRY_INPUT_FIELDS) != 25:  # pragma: no cover - developer invariant
    raise RuntimeError("target-state geometry feature contract must contain 25 fields")


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    split: str
    stage: str
    frame_count: int
    sequence_count: int
    episode_ids: tuple[str, ...]


def _load_depth(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"depth_m"}:
                raise ValueError(f"depth archive must contain only depth_m: {path}")
            value = archive["depth_m"]
    else:
        with Image.open(path) as image:
            value = np.asarray(image)
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"depth must be a 2-D array: {path}")
    return result


def _mask_bbox(path: Path, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not path.is_file():
        return None
    if path.suffix.lower() in {".npy", ".npz"}:
        if path.suffix.lower() == ".npy":
            mask = np.load(path, allow_pickle=False)
        else:
            with np.load(path, allow_pickle=False) as archive:
                if len(archive.files) != 1:
                    raise ValueError(f"instance mask archive must have one array: {path}")
                mask = archive[archive.files[0]]
    else:
        with Image.open(path) as image:
            mask = np.asarray(image)
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if not len(xs):
        return None
    return (
        float(xs.min()) / width,
        float(ys.min()) / height,
        float(xs.max() + 1) / width,
        float(ys.max() + 1) / height,
    )


def _clean_bbox(record: TargetStateFrameRecord, root: Path) -> tuple[float, float, float, float] | None:
    label = record.training_label
    if label is None or not label.visible or label.center_pixel_uv is None:
        return None
    width, height = record.sensor_input.camera.resolution_wh_px
    mask_path = record.sensor_input.instance_mask_path
    if mask_path is not None:
        bbox = _mask_bbox(root / mask_path, width, height)
        if bbox is not None:
            return bbox
    detector_bbox = record.detector_prediction.bbox_xyxy_normalized
    if detector_bbox is not None:
        box_width = max(detector_bbox[2] - detector_bbox[0], 2.0 / width)
        box_height = max(detector_bbox[3] - detector_bbox[1], 2.0 / height)
    else:
        box_width, box_height = 0.2, 0.2
    center_u, center_v = label.center_pixel_uv
    center_x, center_y = center_u / width, center_v / height
    x1, x2 = max(0.0, center_x - box_width / 2), min(1.0, center_x + box_width / 2)
    y1, y2 = max(0.0, center_y - box_height / 2), min(1.0, center_y + box_height / 2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _detector_view(
    record: TargetStateFrameRecord,
    *,
    root: Path,
    stage: TrainingStage,
) -> tuple[tuple[float, float, float, float] | None, float, bool, str | None]:
    if stage is TrainingStage.ORACLE_CLEAN:
        bbox = _clean_bbox(record, root)
        return bbox, (1.0 if bbox is not None else 0.0), bbox is not None, (
            "oracle_clean" if bbox is not None else None
        )
    prediction = record.detector_prediction
    return (
        prediction.bbox_xyxy_normalized,
        float(prediction.confidence or 0.0),
        prediction.detected,
        prediction.tracker_id,
    )


def _crop_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    *,
    size: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, float, float, tuple[float, float]]:
    height, width = depth.shape
    if rgb.shape[:2] != (height, width):
        raise ValueError("RGB and depth resolutions do not match")
    if bbox is None:
        return np.zeros((4, size, size), dtype=np.float32), 0.0, 0.0, (0.0, 0.0)
    x1 = max(0, min(width - 1, int(np.floor(bbox[0] * width))))
    y1 = max(0, min(height - 1, int(np.floor(bbox[1] * height))))
    x2 = max(x1 + 1, min(width, int(np.ceil(bbox[2] * width))))
    y2 = max(y1 + 1, min(height, int(np.ceil(bbox[3] * height))))
    rgb_crop = rgb[y1:y2, x1:x2].astype(np.float32) / 255.0
    depth_crop = depth[y1:y2, x1:x2]
    valid = np.isfinite(depth_crop) & (depth_crop >= min_depth_m) & (depth_crop <= max_depth_m)
    valid_fraction = float(np.mean(valid))
    normalized_depth = np.where(valid, depth_crop / max_depth_m, 0.0).astype(np.float32)
    channels = np.concatenate((rgb_crop, normalized_depth[..., None]), axis=-1)
    rgbd = F.interpolate(
        torch.from_numpy(channels).permute(2, 0, 1).unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).numpy()
    anchor = ((x1 + x2 - 1) * 0.5, (y1 + y2 - 1) * 0.5)
    return rgbd, 0.0, valid_fraction, anchor


def _foreground_cluster_anchor_depth(
    depth: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[float, tuple[float, float]]:
    """Mirror production ``foreground_cluster_median`` preprocessing."""

    if bbox is None:
        return 0.0, (0.0, 0.0)
    height, width = depth.shape
    x1 = max(0, min(width - 1, int(np.floor(bbox[0] * width))))
    y1 = max(0, min(height - 1, int(np.floor(bbox[1] * height))))
    x2 = max(x1, min(width - 1, int(np.ceil(bbox[2] * width)) - 1))
    y2 = max(y1, min(height - 1, int(np.ceil(bbox[3] * height)) - 1))
    bbox_width, bbox_height = x2 - x1 + 1, y2 - y1 + 1
    inset_x = min(max(int(round(bbox_width * 0.1)), 0), max((bbox_width - 1) // 2, 0))
    inset_y = min(max(int(round(bbox_height * 0.1)), 0), max((bbox_height - 1) // 2, 0))
    ix1, ix2 = x1 + inset_x, x2 - inset_x
    iy1 = y1 + inset_y
    iy2 = max(iy1, y2 - inset_y - int(round(bbox_height * 0.15)))
    roi = depth[iy1 : iy2 + 1, ix1 : ix2 + 1]
    valid_mask = np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < 3:
        return 0.0, (0.0, 0.0)
    center_x, center_y = int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
    seed_radius = min(4, max(1, min(bbox_width, bbox_height) // 8))
    seed_patch = depth[
        max(iy1, center_y - seed_radius) : min(iy2 + 1, center_y + seed_radius + 1),
        max(ix1, center_x - seed_radius) : min(ix2 + 1, center_x + seed_radius + 1),
    ]
    seed_values = seed_patch[
        np.isfinite(seed_patch) & (seed_patch >= min_depth_m) & (seed_patch <= max_depth_m)
    ]
    center_depth = float(depth[center_y, center_x])
    if np.isfinite(center_depth) and min_depth_m <= center_depth <= max_depth_m:
        seed_depth = center_depth
        near = seed_values[np.abs(seed_values - seed_depth) <= max(0.15, 0.05 * seed_depth)]
        seed_mad = 0.0 if near.size == 0 else float(np.median(np.abs(near - seed_depth)))
    elif seed_values.size:
        seed_depth = float(np.median(seed_values))
        seed_mad = float(np.median(np.abs(seed_values - seed_depth)))
    else:
        valid_y, valid_x = np.nonzero(valid_mask)
        distances = (valid_x + ix1 - center_x) ** 2 + (valid_y + iy1 - center_y) ** 2
        nearest_count = max(3, min(valid_count, int(np.ceil(valid_count * 0.1))))
        nearest = np.argpartition(distances, nearest_count - 1)[:nearest_count]
        nearest_depths = roi[valid_y[nearest], valid_x[nearest]]
        seed_depth = float(np.median(nearest_depths))
        seed_mad = float(np.median(np.abs(nearest_depths - seed_depth)))
    tolerance = max(0.15, 0.05 * seed_depth, 3.0 * 1.4826 * seed_mad)
    tolerance = min(tolerance, max(0.5, 0.15 * seed_depth))
    cluster_y, cluster_x = np.nonzero(valid_mask & (np.abs(roi - seed_depth) <= tolerance))
    values = roi[cluster_y, cluster_x]
    if values.size < 3:
        return 0.0, (0.0, 0.0)
    return (
        float(np.median(values)),
        (float(np.median(cluster_x + ix1)), float(np.median(cluster_y + iy1))),
    )


def _target_depth(record: TargetStateFrameRecord) -> float:
    """Return optical-z depth from the synchronized camera pose."""

    label = record.training_label
    if label is None:
        raise ValueError("target depth is undefined for a no-target training record")
    position = np.asarray(label.position_world_m, dtype=np.float64)
    camera = record.sensor_input.camera
    relative = position - np.asarray(camera.position_world_m, dtype=np.float64)
    w, x, y, z = camera.orientation_world_wxyz
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    camera_flu = rotation.T @ relative
    return float(camera_flu[0])


def _relative_quaternion_wxyz(
    reference: Sequence[float], current: Sequence[float]
) -> tuple[float, float, float, float]:
    rw, rx, ry, rz = (float(value) for value in reference)
    cw, cx, cy, cz = (float(value) for value in current)
    # inverse(reference) Hamilton-product current
    result = np.asarray(
        (
            rw * cw + rx * cx + ry * cy + rz * cz,
            rw * cx - rx * cw - ry * cz + rz * cy,
            rw * cy + rx * cz - ry * cw - rz * cx,
            rw * cz - rx * cy + ry * cx - rz * cw,
        ),
        dtype=np.float64,
    )
    result /= max(float(np.linalg.norm(result)), 1e-12)
    return tuple(float(value) for value in result)  # type: ignore[return-value]


def _inverse_rotate_wxyz(quaternion: Sequence[float], vector: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return rotation.T @ vector


def _relative_camera_pose(
    current_position: Sequence[float],
    current_orientation: Sequence[float],
    reference_position: Sequence[float],
    reference_orientation: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    position = _inverse_rotate_wxyz(
        reference_orientation,
        np.asarray(current_position, dtype=np.float64)
        - np.asarray(reference_position, dtype=np.float64),
    )
    return (
        tuple(float(value) for value in position),
        _relative_quaternion_wxyz(reference_orientation, current_orientation),
    )


class TargetStateTorchDataset(Dataset[dict[str, Tensor]]):
    """Materialize 4--8 history frames plus one reference observation."""

    def __init__(
        self,
        config: TargetStateTrainingConfig,
        *,
        split: str,
        records: Sequence[TargetStateFrameRecord] | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.config = config
        self.root = config.dataset_root
        if config.require_dataset_manifest:
            manifest_path = self.root / "dataset_manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"dataset manifest is required: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise TypeError("dataset manifest must be a JSON object")
            dataset_sha = manifest.get("dataset_sha256")
            if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
                raise ValueError("dataset manifest has no valid dataset_sha256")
            if config.stage is TrainingStage.YOLO_DEPLOYMENT:
                if manifest.get("detector_prediction_source") != "real_yolo_deployment_output":
                    raise ValueError(
                        "yolo_deployment stage requires recorded real YOLO deployment output"
                    )
                yolo_sha = manifest.get("yolo_model_sha256")
                if not isinstance(yolo_sha, str) or len(yolo_sha) != 64:
                    raise ValueError("yolo_deployment dataset must record yolo_model_sha256")
                if yolo_sha != config.expected_yolo_model_sha256:
                    raise ValueError(
                        "yolo_deployment dataset model SHA256 does not match "
                        "expected_yolo_model_sha256"
                    )
                deployment = manifest.get("detector_deployment")
                if not isinstance(deployment, dict) or not (
                    deployment.get("preflight_verified") is True
                    and deployment.get("model_family") == "yolo"
                    and deployment.get("model_names") == {"0": "cube"}
                    and deployment.get("model_sha256") == config.expected_yolo_model_sha256
                ):
                    raise ValueError(
                        "yolo_deployment dataset requires a preflight-verified detector_deployment receipt"
                    )
        all_records = tuple(records) if records is not None else read_frame_records(self.root / "frames.jsonl")
        all_sequences = build_sequences(
            all_records,
            history_size=config.history_size,
            max_history_age_s=config.max_history_age_s,
        )
        self.sequences = tuple(
            item for item in all_sequences
            if split_for_episode(item.reference.episode_id, seed=config.seed) == split
        )
        self.summary = DatasetSummary(
            split=split,
            stage=config.stage.value,
            frame_count=sum(1 for item in all_records if split_for_episode(item.episode_id, seed=config.seed) == split),
            sequence_count=len(self.sequences),
            episode_ids=tuple(sorted({item.reference.episode_id for item in self.sequences})),
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sequence = self.sequences[index]
        frames = (*sequence.history, sequence.reference)
        reference_camera_position = np.asarray(
            sequence.reference.sensor_input.camera.position_world_m, dtype=np.float32
        )
        reference_camera_orientation = sequence.reference.sensor_input.camera.orientation_world_wxyz
        rois: list[np.ndarray] = []
        geometry: list[np.ndarray] = []
        anchors: list[tuple[float, float]] = []
        raw_depths: list[float] = []
        detected_flags: list[bool] = []
        centers: list[tuple[float, float]] = []
        visible: list[bool] = []
        label_valid_flags: list[bool] = []
        occlusion_ratios: list[float] = []
        camera_intrinsics: list[tuple[float, float, float, float]] = []
        camera_positions: list[tuple[float, float, float]] = []
        camera_orientations: list[tuple[float, float, float, float]] = []
        bbox_centers: list[tuple[float, float]] = []
        tracker_change_flags: list[bool] = []
        previous_tracker: str | None = None
        for record, delta_t in zip(frames, sequence.delta_t_s):
            camera = record.sensor_input.camera
            uav = record.sensor_input.uav
            width, height = camera.resolution_wh_px
            rgb_path = self.root / record.sensor_input.rgb_path
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            depth = _load_depth(self.root / record.sensor_input.depth_path)
            bbox, confidence, detected, tracker_id = _detector_view(
                record, root=self.root, stage=self.config.stage
            )
            roi, _, _valid_fraction, _bbox_anchor = _crop_rgbd(
                rgb,
                depth,
                bbox,
                size=self.config.roi_size_px,
                min_depth_m=self.config.minimum_depth_m,
                max_depth_m=self.config.maximum_depth_m,
            )
            raw_depth, anchor = _foreground_cluster_anchor_depth(
                depth,
                bbox,
                min_depth_m=self.config.minimum_depth_m,
                max_depth_m=self.config.maximum_depth_m,
            )
            missing = not detected
            bbox_values = bbox or (0.0, 0.0, 0.0, 0.0)
            relative_position, relative_orientation = _relative_camera_pose(
                camera.position_world_m,
                camera.orientation_world_wxyz,
                reference_camera_position,
                reference_camera_orientation,
            )
            # These six features are synchronized UAV self-state, not a
            # finite-difference approximation from the camera trajectory.
            # Camera extrinsics still enter through the independently recorded
            # synchronized camera pose and its reference-relative transform.
            linear_velocity = uav.linear_velocity_world_mps
            angular_velocity = uav.angular_velocity_body_radps
            tracker_continuity = float(
                tracker_id is not None
                and previous_tracker is not None
                and tracker_id == previous_tracker
            )
            tracker_change_flags.append(
                tracker_id is not None
                and previous_tracker is not None
                and tracker_id != previous_tracker
            )
            feature = np.asarray(
                [
                    *bbox_values, confidence,
                    tracker_continuity,
                    camera.fx / width, camera.fy / height, camera.cx / width, camera.cy / height,
                    *relative_position, *relative_orientation,
                    *linear_velocity,
                    *angular_velocity,
                    delta_t, float(missing),
                ],
                dtype=np.float32,
            )
            if feature.shape != (self.config.geometry_input_dim,):
                raise ValueError(
                    f"geometry feature contract produced {feature.size} fields, "
                    f"config requires {self.config.geometry_input_dim}"
                )
            rois.append(roi)
            geometry.append(feature)
            anchors.append(anchor)
            raw_depths.append(raw_depth)
            detected_flags.append(detected)
            bbox_centers.append(((bbox_values[0] + bbox_values[2]) / 2, (bbox_values[1] + bbox_values[3]) / 2))
            label = record.training_label
            center = None if label is None else label.center_pixel_uv
            # Missing privileged labels are genuine negatives. Finite zero
            # placeholders keep collation stable, while the explicit masks
            # below ensure they never supervise geometric regression.
            centers.append((0.0, 0.0) if center is None else center)
            visible.append(bool(label is not None and label.visible))
            label_valid_flags.append(label is not None)
            occlusion_ratios.append(0.0 if label is None else label.occlusion_ratio)
            camera_intrinsics.append((camera.fx, camera.fy, camera.cx, camera.cy))
            camera_positions.append(camera.position_world_m)
            camera_orientations.append(camera.orientation_world_wxyz)
            if tracker_id is not None:
                previous_tracker = tracker_id
        reference = sequence.reference
        reference_label = reference.training_label
        target_present = reference_label is not None
        reference_visible = bool(target_present and reference_label.visible)
        reference_raw_depth = raw_depths[-1]
        reference_detected = detected_flags[-1]
        measurement_valid = reference_visible and reference_detected and reference_raw_depth > 0.0
        detected_centers = np.asarray(
            [value for value, is_detected in zip(bbox_centers, detected_flags) if is_detected],
            dtype=np.float32,
        )
        jitter = float(np.linalg.norm(np.std(detected_centers, axis=0))) if len(detected_centers) > 1 else 0.0
        target_position = (
            (0.0, 0.0, 0.0)
            if reference_label is None
            else reference_label.position_world_m
        )
        target_depth = 0.0 if reference_label is None else _target_depth(reference)
        reference_occlusion = (
            0.0 if reference_label is None else reference_label.occlusion_ratio
        )
        return {
            "roi_rgbd": torch.from_numpy(np.stack(rois)),
            "geometry": torch.from_numpy(np.stack(geometry)),
            "missing_mask": torch.tensor([not value for value in detected_flags], dtype=torch.bool),
            "anchor_uv_px": torch.tensor(anchors[-1], dtype=torch.float32),
            "raw_depth_m": torch.tensor(reference_raw_depth, dtype=torch.float32),
            "intrinsics_fx_fy_cx_cy": torch.tensor(camera_intrinsics[-1], dtype=torch.float32),
            "camera_position_world_m": torch.tensor(camera_positions[-1], dtype=torch.float32),
            "camera_orientation_world_wxyz": torch.tensor(camera_orientations[-1], dtype=torch.float32),
            "target_position_world_m": torch.tensor(target_position, dtype=torch.float32),
            "target_depth_m": torch.tensor(target_depth, dtype=torch.float32),
            "target_present_mask": torch.tensor(target_present, dtype=torch.bool),
            "label_valid_mask": torch.tensor(target_present, dtype=torch.bool),
            "measurement_valid": torch.tensor(measurement_valid, dtype=torch.bool),
            "valid_depth_mask": torch.tensor(reference_raw_depth > 0.0, dtype=torch.bool),
            "history_intrinsics_fx_fy_cx_cy": torch.tensor(camera_intrinsics, dtype=torch.float32),
            "history_camera_position_world_m": torch.tensor(camera_positions, dtype=torch.float32),
            "history_camera_orientation_world_wxyz": torch.tensor(camera_orientations, dtype=torch.float32),
            "history_center_uv_px": torch.tensor(centers, dtype=torch.float32),
            "history_visible_mask": torch.tensor(visible, dtype=torch.bool),
            "history_label_valid_mask": torch.tensor(label_valid_flags, dtype=torch.bool),
            "occlusion_ratio": torch.tensor(reference_occlusion, dtype=torch.float32),
            "history_occlusion_ratio": torch.tensor(occlusion_ratios, dtype=torch.float32),
            "bbox_jitter_score": torch.tensor(jitter, dtype=torch.float32),
            "tracker_changed": torch.tensor(any(tracker_change_flags), dtype=torch.bool),
        }


__all__ = ["DatasetSummary", "GEOMETRY_INPUT_FIELDS", "TargetStateTorchDataset"]
