from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from env.camera_types import CameraIntrinsics, CameraSample
from perception.yolo_client import YoloServiceClient, YoloClientResponseError
from training.target_state.collector import (
    TargetStateDatasetWriter,
    VerifiedYoloDeployment,
)
from training.target_state.isaac_capture import (
    DetectorCandidateLinker,
    TargetStateFrameAssembler,
    preflight_deployed_yolo,
)
from training.yolo.isaac_collector import OracleFrameTruth, OracleObjectTruth
from datasets.target_state.schema import UavFrameInput
from datasets.target_state.dataset import check_dataset
from datasets.target_state.sequence import build_sequences
from yolo_service.protocol import TimingMs, TrackDetection, TrackResponse


MODEL_SHA = "8" * 64


def _sample(timestamp_s: float = 1.0) -> CameraSample:
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=np.full((48, 64, 3), 96, dtype=np.uint8),
        depth_to_image_plane_m=np.full((48, 64), 5.0, dtype=np.float32),
        camera_position_world_m=(0.0, 0.0, 2.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=CameraIntrinsics(
            fx=50.0, fy=50.0, cx=31.5, cy=23.5, width=64, height=48
        ),
    )


def _truth_object(
    object_id: str,
    *,
    x_offset: float = 0.0,
    color: str = "red",
) -> OracleObjectTruth:
    x1, x2 = 20.0 + x_offset, 40.0 + x_offset
    pixels = np.asarray(
        ((x1, 14.0), (x2, 14.0), (x2, 34.0), (x1, 34.0)),
        dtype=np.float64,
    )
    return OracleObjectTruth(
        object_id=object_id,
        shape="cube",
        color_name=color,
        position_world_m=(5.0, -x_offset * 0.01, 0.5),
        orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        dimensions_xyz_m=(1.0, 1.0, 1.0),
        projected_pixels_uv=pixels,
        projected_depth_m=np.full(4, 5.0, dtype=np.float64),
        velocity_world_mps=(0.3, 0.1, 0.0),
        center_pixel_uv=((x1 + x2) * 0.5, 24.0),
        occlusion_ratio=0.0,
    )


def _response(
    sample: CameraSample,
    detections: tuple[TrackDetection, ...],
    *,
    frame_id: str = "frame_1",
) -> TrackResponse:
    return TrackResponse(
        schema_version=1,
        request_id="request_1",
        mission_id="mission_1",
        uav_id="uav_1",
        stream_id="mission_1:uav_1",
        frame_id=frame_id,
        timestamp_s=sample.timestamp_s,
        detections=detections,
        timing_ms=TimingMs(0.1, 0.2, 0.1, 0.4),
    )


def _detection(track_id: int, bbox=(0.31, 0.29, 0.63, 0.72)) -> TrackDetection:
    return TrackDetection(
        track_id=track_id,
        class_id=0,
        class_name="cube",
        confidence=0.83,
        bbox_xyxy_normalized=bbox,
    )


def _uav() -> UavFrameInput:
    return UavFrameInput(
        position_world_m=(0.0, 0.0, 3.0),
        orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity_world_mps=(0.2, 0.0, 0.0),
        angular_velocity_body_radps=(0.0, 0.0, 0.0),
    )


def test_worker_preflight_creates_exact_receipt_and_rejects_wrong_sha() -> None:
    def transport(method, url, body, headers, timeout):
        del method, body, headers, timeout
        payload = (
            {"schema_version": 1, "status": "ok", "ready": True}
            if url.endswith("/health")
            else {
                "schema_version": 1,
                "model_family": "yolo",
                "model_names": {"0": "cube"},
                "model_sha256": MODEL_SHA,
            }
        )
        return json.dumps(payload).encode()

    client = YoloServiceClient(base_url="http://127.0.0.1:8011", transport=transport)
    receipt = preflight_deployed_yolo(client, expected_model_sha256=MODEL_SHA)
    assert receipt.model_names == ((0, "cube"),)
    assert receipt.model_sha256 == MODEL_SHA
    with np.testing.assert_raises(YoloClientResponseError):
        preflight_deployed_yolo(client, expected_model_sha256="1" * 64)


def test_tracker_switch_reuses_sensor_candidate_and_miss_keeps_short_continuity() -> None:
    assembler = TargetStateFrameAssembler(minimum_bbox_area_px=4.0)
    sample = _sample(1.0)
    truth = OracleFrameTruth(sample, objects=(_truth_object("cube_0"),))
    first = assembler.assemble(
        capture_id="capture_1",
        episode_id="episode_1",
        assignment_id="assignment_1",
        truth=truth,
        uav_input=_uav(),
        response=_response(sample, (_detection(1),)),
    )[0]
    sample_2 = _sample(1.2)
    truth_2 = OracleFrameTruth(sample_2, objects=(_truth_object("cube_0"),))
    switched = assembler.assemble(
        capture_id="capture_2",
        episode_id="episode_1",
        assignment_id="assignment_1",
        truth=truth_2,
        uav_input=_uav(),
        response=_response(sample_2, (_detection(99),), frame_id="frame_2"),
    )[0]
    sample_3 = _sample(1.4)
    missed = assembler.assemble(
        capture_id="capture_3",
        episode_id="episode_1",
        assignment_id="assignment_1",
        truth=OracleFrameTruth(sample_3, objects=(_truth_object("cube_0"),)),
        uav_input=_uav(),
        response=_response(sample_3, (), frame_id="frame_3"),
    )[0]

    assert first.detector_prediction.candidate_id == switched.detector_prediction.candidate_id
    assert first.detector_prediction.tracker_id != switched.detector_prediction.tracker_id
    assert missed.detector_prediction.detected is False
    assert missed.detector_prediction.candidate_id == first.detector_prediction.candidate_id
    assert missed.training_label is not None
    assert missed.training_label.velocity_world_mps == (0.3, 0.1, 0.0)


def test_multi_target_matching_is_one_to_one_and_false_positive_has_null_label() -> None:
    assembler = TargetStateFrameAssembler(minimum_bbox_area_px=4.0)
    sample = _sample()
    truth = OracleFrameTruth(
        sample,
        objects=(
            _truth_object("cube_0", x_offset=-12.0, color="red"),
            _truth_object("cube_1", x_offset=12.0, color="blue"),
        ),
    )
    detections = (
        _detection(1, (0.12, 0.29, 0.44, 0.72)),
        _detection(2, (0.50, 0.29, 0.82, 0.72)),
        _detection(3, (0.01, 0.02, 0.10, 0.12)),
    )
    records = assembler.assemble(
        capture_id="capture_1",
        episode_id="episode_1",
        assignment_id="assignment_1",
        truth=truth,
        uav_input=_uav(),
        response=_response(sample, detections),
    )

    labels = [record.training_label for record in records]
    assert sum(label is not None for label in labels) == 2
    assert sum(label is None for label in labels) == 1
    assert len(
        {
            record.detector_prediction.candidate_id
            for record in records
            if record.detector_prediction.detected
        }
    ) == 3


def test_no_target_frame_is_null_not_a_fabricated_zero_position() -> None:
    assembler = TargetStateFrameAssembler(minimum_bbox_area_px=4.0)
    sample = _sample()
    records = assembler.assemble(
        capture_id="capture_1",
        episode_id="episode_1",
        assignment_id="assignment_1",
        truth=OracleFrameTruth(sample, objects=()),
        uav_input=_uav(),
        response=_response(sample, ()),
    )
    assert len(records) == 1
    assert records[0].training_label is None
    assert records[0].detector_prediction.detected is False
    assert records[0].to_dict()["training_label"] is None


def test_verified_writer_shares_atomic_rgbd_and_records_receipt() -> None:
    assembler = TargetStateFrameAssembler(minimum_bbox_area_px=4.0)
    sample = _sample()
    truth = OracleFrameTruth(
        sample,
        objects=(
            _truth_object("cube_0", x_offset=-10.0),
            _truth_object("cube_1", x_offset=10.0, color="blue"),
        ),
    )
    records = assembler.assemble(
        capture_id="capture_1",
        episode_id="episode_1",
        assignment_id="assignment_1",
        truth=truth,
        uav_input=_uav(),
        response=_response(
            sample,
            (
                _detection(1, (0.15, 0.29, 0.47, 0.72)),
                _detection(2, (0.46, 0.29, 0.78, 0.72)),
            ),
        ),
    )
    receipt = VerifiedYoloDeployment(
        worker_url="http://127.0.0.1:8011",
        model_family="yolo",
        model_names=((0, "cube"),),
        model_sha256=MODEL_SHA,
    )
    with TemporaryDirectory() as directory:
        root = Path(directory) / "dataset"
        writer = TargetStateDatasetWriter(root, verified_yolo_deployment=receipt)
        for record in records:
            writer.append(
                record,
                rgb=sample.rgb,
                depth_m=sample.depth_to_image_plane_m,
                asset_id="capture_1",
            )
        manifest_path, report = writer.finalize()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert report.ok, report.errors
        assert len(list(root.glob("rgb/**/*.jpg"))) == 1
        assert len(list(root.glob("depth/**/*.npy"))) == 1
        assert manifest["detector_prediction_source"] == "real_yolo_deployment_output"
        assert manifest["detector_deployment"]["preflight_verified"] is True
        assert manifest["yolo_model_sha256"] == MODEL_SHA
        assert check_dataset(root).ok
        manifest["dataset_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        corrupted = check_dataset(root)
        assert not corrupted.ok
        assert any("dataset_sha256" in error for error in corrupted.errors)


def test_candidate_linker_does_not_merge_distant_tracks() -> None:
    linker = DetectorCandidateLinker(maximum_gap_s=2.0, minimum_iou=0.2)
    first = linker.assign_frame((_detection(1),), timestamp_s=1.0)[0]
    distant = linker.assign_frame(
        (_detection(2, (0.01, 0.01, 0.10, 0.10)),), timestamp_s=1.1
    )[0]
    assert first != distant
    after_timeout = linker.assign_frame((_detection(1),), timestamp_s=4.0)[0]
    assert after_timeout != first


def test_candidate_linker_uses_sensor_appearance_when_tracker_ids_swap() -> None:
    linker = DetectorCandidateLinker(maximum_gap_s=2.0, minimum_iou=0.1)
    left = _detection(1, (0.10, 0.20, 0.40, 0.70))
    right = _detection(2, (0.45, 0.20, 0.75, 0.70))
    red_candidate, blue_candidate = linker.assign_frame(
        (left, right),
        timestamp_s=1.0,
        appearance_keys=("red", "blue"),
    )
    swapped_blue = _detection(1, (0.44, 0.20, 0.74, 0.70))
    swapped_red = _detection(2, (0.11, 0.20, 0.41, 0.70))
    blue_after, red_after = linker.assign_frame(
        (swapped_blue, swapped_red),
        timestamp_s=1.1,
        appearance_keys=("blue", "red"),
    )
    assert blue_after == blue_candidate
    assert red_after == red_candidate


def test_negative_background_frames_form_explicit_masked_sequence() -> None:
    assembler = TargetStateFrameAssembler(minimum_bbox_area_px=4.0)
    records = []
    for index in range(5):
        sample = _sample(1.0 + index * 0.2)
        records.extend(
            assembler.assemble(
                capture_id=f"capture_{index}",
                episode_id="episode_1",
                assignment_id="assignment_1",
                truth=OracleFrameTruth(sample, objects=()),
                uav_input=_uav(),
                response=_response(sample, (), frame_id=f"frame_{index}"),
            )
        )
    sequences = build_sequences(records, history_size=4, max_history_age_s=2.0)
    assert len(sequences) == 1
    assert sequences[0].sequence_group_id == "negative_background"
    assert sequences[0].target_present_mask == (False,) * 5
    assert sequences[0].missing_mask == (True,) * 5
