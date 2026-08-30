"""Target-state dataset I/O, leakage-safe splitting, and manifest generation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from datasets.target_state.schema import SCHEMA_VERSION, TargetStateFrameRecord
from datasets.target_state.sequence import TargetStateSequence, build_sequences


_SPLITS = ("train", "validation", "test")
_DERIVED_FLOAT_MANIFEST_FIELDS = frozenset(
    {
        "mean_detector_bbox_center_step_normalized",
        "mean_occlusion_ratio",
        "yolo_miss_rate",
    }
)


def _derived_manifest_float_matches(actual: object, expected: object) -> bool:
    """Compare a derived numeric statistic without weakening type checks."""

    if actual is None or expected is None:
        return actual is None and expected is None
    if type(actual) is not float or type(expected) is not float:
        return False
    return math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


def split_for_episode(episode_id: str, *, seed: int = 42) -> str:
    """Assign a complete episode to one deterministic split."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    bucket = int.from_bytes(
        sha256(f"{seed}:{episode_id}".encode("utf-8")).digest()[:8],
        "big",
    ) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def read_frame_records(path: str | Path) -> tuple[TargetStateFrameRecord, ...]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"frame manifest does not exist: {source}")
    result: list[TargetStateFrameRecord] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {source}:{line_number}")
            try:
                payload = json.loads(line)
                result.append(TargetStateFrameRecord.from_dict(payload))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid target-state record at {source}:{line_number}: {exc}"
                ) from exc
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DatasetCheckReport:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    frame_count: int
    sequence_count: int
    split_frame_counts: Mapping[str, int]
    detected_frames: int
    visible_frames: int
    target_labeled_frames: int
    no_target_frames: int
    false_positive_frames: int
    dataset_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "frame_count": self.frame_count,
            "sequence_count": self.sequence_count,
            "split_frame_counts": dict(self.split_frame_counts),
            "detected_frames": self.detected_frames,
            "visible_frames": self.visible_frames,
            "target_labeled_frames": self.target_labeled_frames,
            "no_target_frames": self.no_target_frames,
            "false_positive_frames": self.false_positive_frames,
            "dataset_sha256": self.dataset_sha256,
        }


def _update_file_hash(digest: object, path: Path, relative: str) -> None:
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)


def compute_dataset_sha256(
    root: str | Path,
    records: Sequence[TargetStateFrameRecord],
) -> str:
    dataset_root = Path(root).expanduser().resolve()
    digest = sha256()
    for record in sorted(records, key=lambda item: item.frame_id):
        encoded = json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
        paths = [record.sensor_input.rgb_path, record.sensor_input.depth_path]
        if record.sensor_input.instance_mask_path is not None:
            paths.append(record.sensor_input.instance_mask_path)
        for relative in paths:
            _update_file_hash(digest, dataset_root / relative, relative)
    return digest.hexdigest()


def _validate_existing_manifest(
    dataset_root: Path,
    *,
    records: Sequence[TargetStateFrameRecord],
    sequences: Sequence[TargetStateSequence],
    dataset_sha256: str,
    split_seed: int,
) -> tuple[list[str], list[str]]:
    """Cross-check a finalized manifest against actual decoded data."""

    path = dataset_root / "dataset_manifest.json"
    if not path.is_file():
        # Writers run the checker before atomically creating the final manifest.
        return [], []
    errors: list[str] = []
    warnings: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot read dataset_manifest.json: {exc}"], []
    if not isinstance(payload, Mapping):
        return ["dataset_manifest.json must contain a JSON object"], []
    required = {
        "schema_version",
        "dataset_sha256",
        "generation_commit_sha",
        "frame_count",
        "sequence_count",
        "scene_count",
        "episode_splits",
        "target_color_distribution",
        "mean_occlusion_ratio",
        "yolo_miss_rate",
        "detector_prediction_source",
        "candidate_id_source",
        "detector_truth_association",
        "yolo_model_sha256",
        "oracle_usage",
        "history_size",
        "max_history_age_s",
    }
    missing = sorted(required - set(payload))
    if missing:
        errors.append(f"dataset manifest missing fields: {missing}")
    expected = build_manifest(
        records,
        sequences,
        dataset_sha256=dataset_sha256,
        split_seed=split_seed,
        generation_commit_sha=str(payload.get("generation_commit_sha", "")),
    )
    for field in (
        "schema_version",
        "dataset_sha256",
        "frame_count",
        "sequence_count",
        "scene_count",
        "episode_splits",
        "target_color_distribution",
        "target_labeled_frame_count",
        "no_target_frame_count",
        "false_positive_frame_count",
        "physical_capture_count",
        "multi_target_capture_count",
        "no_target_capture_count",
        "crossing_event_count",
        "tracker_id_switch_count",
        "mean_detector_bbox_center_step_normalized",
        "mean_occlusion_ratio",
        "yolo_miss_rate",
        "history_sizes",
        "negative_sequence_count",
        "mixed_presence_sequence_count",
    ):
        if field not in payload:
            continue
        matches = (
            _derived_manifest_float_matches(payload[field], expected[field])
            if field in _DERIVED_FLOAT_MANIFEST_FIELDS
            else payload[field] == expected[field]
        )
        if not matches:
            errors.append(
                f"dataset manifest {field} does not match decoded dataset"
            )
    generation_sha = payload.get("generation_commit_sha")
    if not isinstance(generation_sha, str) or not generation_sha.strip():
        errors.append("dataset manifest generation_commit_sha is invalid")
    if payload.get("oracle_usage") != "offline_training_labels_only":
        errors.append("dataset manifest oracle_usage must be offline_training_labels_only")
    model_sha = payload.get("yolo_model_sha256")
    if (
        not isinstance(model_sha, str)
        or len(model_sha) != 64
        or any(character not in "0123456789abcdef" for character in model_sha.lower())
    ):
        errors.append("dataset manifest yolo_model_sha256 is invalid")
    source = payload.get("detector_prediction_source")
    deployment = payload.get("detector_deployment")
    if source == "real_yolo_deployment_output":
        if payload.get("candidate_id_source") != "sensor_only_bbox_color_temporal_linker":
            errors.append("real YOLO dataset candidate_id_source is unsupported")
        if payload.get("detector_truth_association") != (
            "offline_privileged_one_to_one_iou_after_worker_inference"
        ):
            errors.append("real YOLO dataset truth association provenance is invalid")
        if not isinstance(deployment, Mapping):
            errors.append("real YOLO dataset has no detector_deployment receipt")
        else:
            if deployment.get("preflight_verified") is not True:
                errors.append("detector deployment receipt is not preflight-verified")
            if deployment.get("model_family") != "yolo":
                errors.append("detector deployment model_family is not yolo")
            if deployment.get("model_names") != {"0": "cube"}:
                errors.append("detector deployment model_names must be exactly {0: cube}")
            if deployment.get("model_sha256") != model_sha:
                errors.append("detector deployment SHA does not match dataset manifest")
            worker_url = deployment.get("worker_url")
            if not isinstance(worker_url, str) or not worker_url.startswith(
                ("http://127.0.0.1:", "http://localhost:")
            ):
                errors.append("detector deployment worker_url is not loopback HTTP")
    elif source == "external_capture_spool_unverified":
        warnings.append(
            "detector provenance is unverified; this dataset is not eligible for "
            "yolo_deployment training"
        )
    else:
        errors.append("dataset manifest detector_prediction_source is unsupported")
    return errors, warnings


def check_dataset(
    root: str | Path,
    *,
    frames_path: str = "frames.jsonl",
    sequences: Sequence[TargetStateSequence] | None = None,
    history_size: int = 6,
    max_history_age_s: float = 2.0,
    split_seed: int = 42,
) -> DatasetCheckReport:
    dataset_root = Path(root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        records = read_frame_records(dataset_root / frames_path)
    except (OSError, ValueError) as exc:
        return DatasetCheckReport(
            False,
            (str(exc),),
            (),
            0,
            0 if sequences is None else len(sequences),
            {name: 0 for name in _SPLITS},
            0,
            0,
            0,
            0,
            0,
            None,
        )
    ids: set[str] = set()
    timestamps: dict[tuple[str, str, str, str | None, str | None], float] = {}
    episode_splits: dict[str, str] = {}
    split_counts = {name: 0 for name in _SPLITS}
    synchronized_assets: dict[tuple[str, str], tuple[object, ...]] = {}
    for record in records:
        if record.frame_id in ids:
            errors.append(f"duplicate frame_id: {record.frame_id}")
        ids.add(record.frame_id)
        key = (
            record.uav_id,
            record.assignment_id,
            record.episode_id,
            record.detector_prediction.candidate_id,
            None
            if record.training_label is None
            else record.training_label.instance_id,
        )
        previous = timestamps.get(key)
        if previous is not None and record.timestamp_s <= previous:
            errors.append(
                "timestamps are not strictly increasing for "
                f"{record.uav_id}/{record.assignment_id}/{record.episode_id}/"
                f"{record.detector_prediction.candidate_id}"
            )
        timestamps[key] = record.timestamp_s
        split = split_for_episode(record.episode_id, seed=split_seed)
        if record.episode_id in episode_splits and episode_splits[record.episode_id] != split:
            errors.append(f"episode crosses splits: {record.episode_id}")
        episode_splits[record.episode_id] = split
        split_counts[split] += 1
        camera = record.sensor_input.camera
        rgb_path = dataset_root / record.sensor_input.rgb_path
        depth_path = dataset_root / record.sensor_input.depth_path
        try:
            with Image.open(rgb_path) as image:
                image.verify()
            with Image.open(rgb_path) as image:
                if image.size != camera.resolution_wh_px:
                    errors.append(f"RGB resolution mismatch: {record.frame_id}")
        except (OSError, ValueError) as exc:
            errors.append(f"cannot decode RGB for {record.frame_id}: {exc}")
        try:
            if depth_path.suffix.lower() == ".npy":
                depth = np.load(depth_path, allow_pickle=False)
            elif depth_path.suffix.lower() == ".npz":
                with np.load(depth_path, allow_pickle=False) as archive:
                    if set(archive.files) != {"depth_m"}:
                        raise ValueError("NPZ depth must contain only depth_m")
                    depth = archive["depth_m"]
            else:
                with Image.open(depth_path) as image:
                    depth = np.asarray(image)
            if depth.shape != (camera.resolution_wh_px[1], camera.resolution_wh_px[0]):
                errors.append(f"depth resolution mismatch: {record.frame_id}")
            if not np.issubdtype(depth.dtype, np.number):
                errors.append(f"depth is not numeric: {record.frame_id}")
            elif not np.any(np.isfinite(depth) & (depth > 0.0)):
                warnings.append(f"depth has no positive finite pixels: {record.frame_id}")
        except (OSError, ValueError) as exc:
            errors.append(f"cannot read depth for {record.frame_id}: {exc}")
        mask_path = record.sensor_input.instance_mask_path
        if mask_path is not None and not (dataset_root / mask_path).is_file():
            errors.append(f"instance mask is missing: {record.frame_id}")

        asset_key = (
            record.sensor_input.rgb_path,
            record.sensor_input.depth_path,
        )
        synchronization_signature = (
            record.timestamp_s,
            record.sensor_input.camera,
            record.sensor_input.uav,
        )
        prior_signature = synchronized_assets.get(asset_key)
        if prior_signature is not None and prior_signature != synchronization_signature:
            errors.append(
                "shared RGB/depth assets carry inconsistent synchronized state: "
                f"{asset_key}"
            )
        synchronized_assets[asset_key] = synchronization_signature

        label = record.training_label
        center = None if label is None else label.center_pixel_uv
        if center is not None:
            width, height = camera.resolution_wh_px
            if not (0.0 <= center[0] < width and 0.0 <= center[1] < height):
                errors.append(
                    "target center_pixel_uv is outside camera resolution for "
                    f"{record.frame_id}: center={center}, resolution={(width, height)}"
                )

    sequence_parameters_valid = True
    try:
        # Reuse the canonical builder's strict parameter validation even when
        # callers supply pre-built sequences.
        build_sequences(
            (),
            history_size=history_size,
            max_history_age_s=max_history_age_s,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid temporal sequence parameters: {exc}")
        sequence_parameters_valid = False

    resolved_sequences: tuple[TargetStateSequence, ...]
    if sequences is None and sequence_parameters_valid:
        try:
            resolved_sequences = build_sequences(
                records,
                history_size=history_size,
                max_history_age_s=max_history_age_s,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"cannot build temporal sequences: {exc}")
            resolved_sequences = ()
    elif sequences is None:
        resolved_sequences = ()
    else:
        resolved_sequences = tuple(sequences)

    records_by_id = {record.frame_id: record for record in records}
    sequence_ids: set[str] = set()
    for index, sequence in enumerate(resolved_sequences):
        if not isinstance(sequence, TargetStateSequence):
            errors.append(f"sequence at index {index} is not a TargetStateSequence")
            continue
        if sequence.sequence_id in sequence_ids:
            errors.append(f"duplicate sequence_id: {sequence.sequence_id}")
        sequence_ids.add(sequence.sequence_id)

        frames = (*sequence.history, sequence.reference)
        if not 4 <= len(sequence.history) <= 8:
            errors.append(f"sequence {sequence.sequence_id} history size is outside [4, 8]")
        if any(
            left.timestamp_s >= right.timestamp_s
            for left, right in zip(frames, frames[1:])
        ):
            errors.append(f"sequence {sequence.sequence_id} is not strictly time ordered")
        for field in ("uav_id", "assignment_id", "episode_id"):
            if len({getattr(frame, field) for frame in frames}) != 1:
                errors.append(f"sequence {sequence.sequence_id} crosses {field}")
        labels = tuple(frame.training_label for frame in frames)
        instance_ids = {
            label.instance_id
            for label in labels
            if label is not None and label.instance_id is not None
        }
        if len(instance_ids) > 1:
            errors.append(
                f"sequence {sequence.sequence_id} crosses target instance"
            )
        candidate_ids = tuple(
            frame.detector_prediction.candidate_id for frame in frames
        )
        nonempty_candidate_ids = {
            candidate_id for candidate_id in candidate_ids if candidate_id is not None
        }
        if len(nonempty_candidate_ids) > 1:
            errors.append(
                f"sequence {sequence.sequence_id} mixes candidate_id values"
            )
        expected_target_present = tuple(label is not None for label in labels)
        if sequence.target_present_mask != expected_target_present:
            errors.append(
                f"sequence {sequence.sequence_id} target_present_mask does not match labels"
            )
        if not nonempty_candidate_ids and any(expected_target_present):
            errors.append(
                f"sequence {sequence.sequence_id} has target labels without a candidate"
            )
        expected_group = (
            next(iter(nonempty_candidate_ids))
            if nonempty_candidate_ids
            else "negative_background"
        )
        if sequence.sequence_group_id != expected_group:
            errors.append(
                f"sequence {sequence.sequence_id} sequence_group_id is inconsistent"
            )

        missing_frame_ids = sorted(
            frame.frame_id for frame in frames if frame.frame_id not in records_by_id
        )
        if missing_frame_ids:
            errors.append(
                f"sequence {sequence.sequence_id} references missing frames: {missing_frame_ids}"
            )
        else:
            modified_frame_ids = sorted(
                frame.frame_id
                for frame in frames
                if records_by_id[frame.frame_id] != frame
            )
            if modified_frame_ids:
                errors.append(
                    f"sequence {sequence.sequence_id} frames differ from frames.jsonl: "
                    f"{modified_frame_ids}"
                )

        expected_length = len(frames)
        if len(sequence.delta_t_s) != expected_length:
            errors.append(f"sequence {sequence.sequence_id} has an invalid delta_t_s length")
        else:
            expected_delta_t = tuple(
                sequence.reference.timestamp_s - frame.timestamp_s for frame in frames
            )
            if any(
                not isfinite(actual) or abs(actual - expected) > 1e-6
                for actual, expected in zip(sequence.delta_t_s, expected_delta_t)
            ):
                errors.append(
                    f"sequence {sequence.sequence_id} delta_t_s is not relative to its reference"
                )
            if max(sequence.delta_t_s, default=0.0) > max_history_age_s + 1e-9:
                errors.append(
                    f"sequence {sequence.sequence_id} exceeds max_history_age_s="
                    f"{max_history_age_s}"
                )

        expected_missing = tuple(
            not frame.detector_prediction.detected for frame in frames
        )
        if sequence.missing_mask != expected_missing:
            errors.append(
                f"sequence {sequence.sequence_id} missing_mask does not match detector output"
            )
        tracker_ids = tuple(
            frame.detector_prediction.tracker_id for frame in frames
        )
        expected_tracker_changes = tuple(
            False if frame_index == 0 else tracker_ids[frame_index] != tracker_ids[frame_index - 1]
            for frame_index in range(len(tracker_ids))
        )
        if sequence.tracker_id_changed != expected_tracker_changes:
            errors.append(
                f"sequence {sequence.sequence_id} tracker_id_changed does not match tracker output"
            )

    if records and not resolved_sequences:
        warnings.append(
            "no temporal sequences could be built; check candidate continuity, history_size, "
            "and max_history_age_s"
        )
    digest: str | None = None
    if not errors:
        try:
            digest = compute_dataset_sha256(dataset_root, records)
        except OSError as exc:
            errors.append(f"cannot hash dataset: {exc}")
    if digest is not None and not errors:
        manifest_errors, manifest_warnings = _validate_existing_manifest(
            dataset_root,
            records=records,
            sequences=resolved_sequences,
            dataset_sha256=digest,
            split_seed=split_seed,
        )
        errors.extend(manifest_errors)
        warnings.extend(manifest_warnings)
    return DatasetCheckReport(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        frame_count=len(records),
        sequence_count=len(resolved_sequences),
        split_frame_counts=split_counts,
        detected_frames=sum(item.detector_prediction.detected for item in records),
        visible_frames=sum(
            item.training_label is not None and item.training_label.visible
            for item in records
        ),
        target_labeled_frames=sum(item.training_label is not None for item in records),
        no_target_frames=sum(item.training_label is None for item in records),
        false_positive_frames=sum(
            item.training_label is None and item.detector_prediction.detected
            for item in records
        ),
        dataset_sha256=digest,
    )


def build_manifest(
    records: Sequence[TargetStateFrameRecord],
    sequences: Sequence[TargetStateSequence],
    *,
    dataset_sha256: str,
    split_seed: int = 42,
    generation_commit_sha: str = "nogit",
) -> dict[str, object]:
    if len(dataset_sha256) != 64:
        raise ValueError("dataset_sha256 must be a SHA256 digest")
    colors: dict[str, int] = {}
    for record in records:
        if record.training_label is None:
            continue
        color = record.training_label.color_name or "unknown"
        colors[color] = colors.get(color, 0) + 1
    episode_splits = {
        episode: split_for_episode(episode, seed=split_seed)
        for episode in sorted({item.episode_id for item in records})
    }
    labeled_records = [item for item in records if item.training_label is not None]
    occlusion = [
        item.training_label.occlusion_ratio
        for item in labeled_records
        if item.training_label is not None
    ]
    captures: dict[tuple[str, str], list[TargetStateFrameRecord]] = {}
    for record in records:
        captures.setdefault(
            (record.episode_id, record.sensor_input.rgb_path), []
        ).append(record)
    multi_target_captures = 0
    no_target_captures = 0
    positions_by_episode_time: dict[
        tuple[str, float], dict[str, np.ndarray]
    ] = {}
    for (episode_id, _rgb_path), capture_records in captures.items():
        instances = {
            record.training_label.instance_id
            for record in capture_records
            if record.training_label is not None
            and record.training_label.instance_id is not None
        }
        multi_target_captures += int(len(instances) >= 2)
        no_target_captures += int(
            all(record.training_label is None for record in capture_records)
        )
        for record in capture_records:
            label = record.training_label
            if label is None or label.instance_id is None:
                continue
            positions_by_episode_time.setdefault(
                (episode_id, record.timestamp_s), {}
            )[label.instance_id] = np.asarray(label.position_world_m, dtype=np.float64)
    relative_by_pair: dict[
        tuple[str, str, str], list[tuple[float, np.ndarray]]
    ] = {}
    for (episode_id, timestamp_s), positions in positions_by_episode_time.items():
        instance_ids = sorted(positions)
        for left_index, left_id in enumerate(instance_ids):
            for right_id in instance_ids[left_index + 1 :]:
                relative_by_pair.setdefault(
                    (episode_id, left_id, right_id), []
                ).append((timestamp_s, positions[right_id] - positions[left_id]))
    crossing_events = 0
    for values in relative_by_pair.values():
        ordered = sorted(values, key=lambda item: item[0])
        crossing_events += sum(
            float(np.dot(previous[1][:2], current[1][:2])) < 0.0
            for previous, current in zip(ordered, ordered[1:])
        )
    tracker_switches = 0
    bbox_steps: list[float] = []
    by_candidate: dict[tuple[str, str], list[TargetStateFrameRecord]] = {}
    for record in records:
        candidate_id = record.detector_prediction.candidate_id
        if candidate_id is not None:
            by_candidate.setdefault((record.episode_id, candidate_id), []).append(record)
    for candidate_records in by_candidate.values():
        ordered = sorted(candidate_records, key=lambda item: item.timestamp_s)
        previous_tracker: str | None = None
        previous_center: tuple[float, float] | None = None
        for record in ordered:
            prediction = record.detector_prediction
            if (
                previous_tracker is not None
                and prediction.tracker_id is not None
                and prediction.tracker_id != previous_tracker
            ):
                tracker_switches += 1
            if prediction.tracker_id is not None:
                previous_tracker = prediction.tracker_id
            bbox = prediction.bbox_xyxy_normalized
            if bbox is not None:
                center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
                if previous_center is not None:
                    bbox_steps.append(
                        float(np.linalg.norm(np.asarray(center) - np.asarray(previous_center)))
                    )
                previous_center = center
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "generation_commit_sha": generation_commit_sha,
        "frame_count": len(records),
        "sequence_count": len(sequences),
        "scene_count": len(episode_splits),
        "episode_splits": episode_splits,
        "target_color_distribution": colors,
        "target_labeled_frame_count": len(labeled_records),
        "no_target_frame_count": sum(
            item.training_label is None for item in records
        ),
        "false_positive_frame_count": sum(
            item.training_label is None and item.detector_prediction.detected
            for item in records
        ),
        "physical_capture_count": len(captures),
        "multi_target_capture_count": multi_target_captures,
        "no_target_capture_count": no_target_captures,
        "crossing_event_count": crossing_events,
        "tracker_id_switch_count": tracker_switches,
        "mean_detector_bbox_center_step_normalized": (
            sum(bbox_steps) / len(bbox_steps) if bbox_steps else None
        ),
        "mean_occlusion_ratio": (sum(occlusion) / len(occlusion) if occlusion else None),
        "yolo_miss_rate": (
            sum(not item.detector_prediction.detected for item in labeled_records)
            / len(labeled_records)
            if labeled_records else None
        ),
        "history_sizes": sorted({len(item.history) for item in sequences}),
        "negative_sequence_count": sum(
            not any(item.target_present_mask) for item in sequences
        ),
        "mixed_presence_sequence_count": sum(
            any(item.target_present_mask) and not all(item.target_present_mask)
            for item in sequences
        ),
    }


__all__ = [
    "DatasetCheckReport",
    "build_manifest",
    "check_dataset",
    "compute_dataset_sha256",
    "read_frame_records",
    "split_for_episode",
]
