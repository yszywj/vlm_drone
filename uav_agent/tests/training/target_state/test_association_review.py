"""Offline-only association, uncertain-window quarantine, safe streamed review."""
from dataclasses import replace
import json
from unittest import mock

import numpy as np
import pytest

from datasets.target_state.schema import TargetStateFrameRecord
from datasets.target_state.sequence import build_sequences, TargetStateSequence
from training.target_state.association_quality import AssociationPolicy, depth_support
from training.target_state.association_audit import scan_records
from training.target_state.isaac_capture import TargetStateFrameAssembler
from training.yolo.isaac_collector import OracleFrameTruth
from scripts import audit_target_state_associations as audit
from tests.training.target_state.test_dataset_schema import make_record
from tests.training.target_state.test_isaac_capture import _sample, _truth_object, _response, _detection, _uav
from tests.training.target_state.test_shards import _write_parent_dataset
from tests.training.target_state.test_sharded_trainer import _FakeLifecycle
from training.target_state.shards import build_target_state_shards


def test_depth_evidence_handles_foreground_sparse_and_partial_occlusion():
    depth = np.full((48, 64), 5.0)
    bbox = (0.1, 0.1, 0.9, 0.9)
    assert depth_support(depth, bbox, 4.5, 5.5)["supported"]
    depth[:, :20] = 2.0  # Partial occlusion still has a dominant target surface.
    assert depth_support(depth, bbox, 4.5, 5.5)["supported"]
    depth[:] = 2.0
    assert depth_support(depth, bbox, 4.5, 5.5)["status"] == "foreground_conflict_or_occlusion"
    depth[:] = np.nan
    depth[24, 32] = 5.0
    assert depth_support(depth, bbox, 4.5, 5.5)["status"] == "insufficient_depth"
    assert not depth_support(depth, bbox, -1, 1)["supported"]


def assemble(sample, objects):
    return TargetStateFrameAssembler(minimum_bbox_area_px=4).assemble(
        capture_id="capture_1", episode_id="episode_1", assignment_id="assignment_1",
        truth=OracleFrameTruth(sample, objects=objects), uav_input=_uav(),
        response=_response(sample, (_detection(3),)))


def test_foreground_detection_is_not_forced_positive_or_negative():
    sample = replace(_sample(), depth_to_image_plane_m=np.full((48, 64), 2.0, dtype=np.float32))
    records = assemble(sample, (_truth_object("cube_0"),))
    assert len(records) == 2
    target, uncertain = records
    assert target.training_label.instance_id == "cube_0"
    assert not target.detector_prediction.detected
    assert uncertain.training_label is None
    assert uncertain.detector_prediction.tracker_id == "track_3"
    assert uncertain.detector_prediction.candidate_id == "candidate_000001"
    assert all(r.association_review_required for r in records)


def test_competing_same_depth_truth_is_unresolved_not_greedy_assignment():
    records = assemble(_sample(), (_truth_object("cube_0"), _truth_object("cube_1")))
    assert len(records) == 3
    assert all(r.association_review_required for r in records)
    assert all(not r.detector_prediction.detected for r in records[:2])


def test_visible_target_sliver_overlapping_foreground_box_is_not_associated():
    # Truth remains visible, but only a sliver overlaps a foreground detection.
    # This exercises the old IoU-only false association, not full occlusion.
    sample = _sample()
    depth = sample.depth_to_image_plane_m.copy()
    depth[14:34, 36:46] = 2.0
    sample = replace(sample, depth_to_image_plane_m=depth)
    records = TargetStateFrameAssembler(minimum_bbox_area_px=4).assemble(
        capture_id="capture_1", episode_id="episode_1", assignment_id="assignment_1",
        truth=OracleFrameTruth(sample, objects=(_truth_object("cube_0"),)), uav_input=_uav(),
        response=_response(sample, (_detection(3, (36/64, 14/48, 46/64, 34/48)),)))
    assert records[0].training_label.visible
    assert not records[0].detector_prediction.detected
    assert all(r.association_review_required for r in records)


def test_schema_extension_preserves_legacy_and_entire_windows():
    legacy = make_record(0).to_dict()
    assert "association_review_required" not in legacy
    assert TargetStateFrameRecord.from_dict(legacy).to_dict() == legacy
    records = [make_record(i) for i in range(12)]
    records[4] = replace(records[4], association_review_required=True)
    restored = TargetStateFrameRecord.from_dict(records[4].to_dict())
    assert restored == records[4]
    assert "association_review_required" not in restored.sensor_input.to_dict()
    sequences = build_sequences(records, history_size=4)
    assert [s.reference.frame_id for s in sequences] == ["frame_9", "frame_10", "frame_11"]
    with pytest.raises(ValueError, match="unresolved"):
        TargetStateSequence("bad", tuple(records[:4]), records[4], (0.8, 0.6, 0.4, 0.2, 0),
                            (False,) * 5, (False,) * 5)


def test_review_in_history_marks_window_even_when_reference_depth_is_good(tmp_path):
    records = [make_record(i) for i in range(7)]
    for i, record in enumerate(records):
        path = tmp_path / record.sensor_input.depth_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Camera x=1; forward target depth=4+i*.1. Frame 2 sees an occluder.
        np.save(path, np.full((24, 32), 1.0 if i == 2 else 4.0 + i * 0.1))
    result = scan_records(records, dataset_root=tmp_path, history_size=6, max_history_age_s=2)
    assert result["counts"]["review_required_records"] == 1
    assert result["counts"]["review_affected_sequences"] == 1
    assert result["counts"]["review_reference_sequences"] == 0
    assert result["affected_sequences"][0]["review_frame_ids"] == ["frame_2"]


@pytest.fixture
def streamed(tmp_path):
    parent = tmp_path / "parent"
    _write_parent_dataset(parent)
    index = build_target_state_shards(parent, tmp_path / "shards", target_shard_size_bytes=1,
        history_size=4, max_history_age_s=2.0, split_seed=42).shard_index
    options = audit.ShardedTrainingOptions(shard_index_path=index.source_path,
        pc_trans_root=tmp_path / "unused", pc_trans_config=tmp_path / "unused.json",
        bridge_root=tmp_path / "bridge", run_id_prefix="audit_association_fixture", wait_timeout_s=0)
    return dict(split="train", options=options, index=index,
        lifecycle=_FakeLifecycle(tmp_path / "shards", tmp_path / "bridge"),
        output_dir=tmp_path / "review", contract_sha="fixture", policy=AssociationPolicy())


def test_scan_commit_before_consume_resume_and_keep_pc_source(streamed):
    lifecycle = streamed["lifecycle"]
    with mock.patch.object(lifecycle, "consume", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            audit.scan_split(**streamed)
    original = audit.scan_archive
    first = streamed["index"].shards_for_split("train")[0]
    def scan_remaining(archive, **kwargs):
        assert archive.name != first.filename, "first shard must resume from receipt"
        return original(archive, **kwargs)
    with mock.patch.object(audit, "scan_archive", side_effect=scan_remaining):
        result = audit.scan_split(**streamed)
    with mock.patch.object(audit, "scan_archive", side_effect=AssertionError("must reuse receipt")):
        assert audit.scan_split(**streamed) == result
    entry = streamed["index"].shards_for_split("train")[0]
    assert (lifecycle.source / entry.filename).is_file()
    assert not lifecycle._active("audit_association_fixture.train", entry.filename).exists()


def test_receipt_write_failure_never_consumes(streamed):
    with mock.patch.object(audit, "_atomic_write_json", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            audit.scan_split(**streamed)
    assert not streamed["lifecycle"].consumed


def test_receipt_corruption_and_changed_contract_fail_closed(streamed):
    audit.scan_split(**streamed)
    entry = streamed["index"].shards_for_split("train")[0]
    path = streamed["output_dir"] / "receipts" / (entry.filename + ".json")
    with pytest.raises(ValueError, match="identity/checksum"):
        audit.load_receipt(path, contract_sha="other", entry=entry)
    receipt = json.loads(path.read_text())
    receipt["result"]["counts"]["physical_captures"] = 999
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identity/checksum"):
        audit.scan_split(**streamed)
