"""Offline diagnosis must explain existing rules without changing source labels."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from datasets.target_state.schema import TargetStateFrameRecord
from scripts.diagnose_target_state_v3_pilot import detection_cases, validate_output, window_audit
from target_state_v3.association import AssociationPolicy, InstanceFrameAssembler, normalize_instances
from tests.test_target_state_v3 import assembled, catalog, instances
from tests.training.target_state.test_isaac_capture import _sample


def timeline(n, unknown_at=None):
    records, indices = [], {}
    assembler = InstanceFrameAssembler()
    for i in range(n):
        sample = replace(_sample(1.+.2*i), render_frame_id=(i+1, 30))
        mask, mapping = instances(np.full((48, 64), 1 if i == unknown_at else 70000, np.uint32))
        rows = assembled(sample, mask, mapping, assembler, capture=f"capture_{i}", episode="episode_0")
        for row in rows:
            r = TargetStateFrameRecord.from_dict(row["record"])
            records.append(r)
            indices[r.frame_id] = i
    return records, indices


def test_window_ledger_matches_actual_builder_and_positive_run():
    records, indices = timeline(9)
    counts, candidates, windows, accepted = window_audit(records, indices, history_size=6, max_history_age_s=2.)
    assert counts["proposed_windows"] == counts["eligible_windows"] == 3
    assert counts["eligible_positive_reference_windows"] == 3
    assert candidates[0]["longest_same_target_matched_run_consecutive_captures"] == 9
    assert len(windows) == len(accepted) == 3


def test_unresolved_row_is_not_removed_to_create_positive_history():
    records, indices = timeline(9, unknown_at=3)
    before = [r.to_dict() for r in records]
    counts, candidates, _, accepted = window_audit(records, indices, history_size=6, max_history_age_s=2.)
    assert counts["proposed_windows"] == counts["unresolved_in_window"] == 3
    assert counts["skipped_label_without_candidate"] == 1
    assert accepted == set()
    assert candidates[0]["longest_same_target_matched_run_consecutive_captures"] == 5
    assert [r.to_dict() for r in records] == before


def test_short_track_has_no_seven_frame_window():
    records, indices = timeline(4)
    counts, _, _, _ = window_audit(records, indices, history_size=6, max_history_age_s=2.)
    assert counts["insufficient_history_references"] == 4
    assert counts["proposed_windows"] == 0


def test_longest_run_does_not_bridge_missing_physical_captures():
    records, indices = timeline(9)
    records = [r for r in records if indices[r.frame_id] != 3]
    _, candidates, _, _ = window_audit(records, indices, history_size=6, max_history_age_s=2.)
    assert candidates[0]["longest_same_target_matched_run_consecutive_captures"] == 5


def ground_case():
    mask = np.full((48, 64), 70000, dtype=np.uint32)
    mask[:, :32] = 13
    mask, mapping = normalize_instances({"data": mask, "info": {"idToLabels": {
        "13": "/World/Ground/geom", "70000": "/World/CubeV1Collection/cube_0/Body"}}},
        shape_hw=(48, 64), catalog=catalog())
    rows = assembled(_sample(), mask, mapping)
    return {"records": rows, "capture_id": "capture_1", "episode_id": "episode_1",
            "timestamp_s": _sample().timestamp_s, "oracle_only": {"mapping": mapping}}, mask


def test_unknown_ground_is_attributed_but_not_reclassified():
    payload, mask = ground_case()
    before = deepcopy(payload)
    cases = detection_cases(payload, mask, AssociationPolicy())
    assert len(cases) == 1
    assert cases[0]["association"]["status"] == "unresolved"
    assert cases[0]["unknown_roi_pixels_by_prim"]["/World/Ground/geom"] > 0
    assert cases[0]["ground_alone_exceeds_unknown_limit"] is True
    assert payload == before


def test_replay_mismatch_is_not_silently_relabelled():
    payload, mask = ground_case()
    payload["records"][0]["association"]["status"] = "background"
    with pytest.raises(ValueError, match="replay mismatch"):
        detection_cases(payload, mask, AssociationPolicy())


def test_output_cannot_overwrite_or_live_inside_source_session(tmp_path):
    session = tmp_path/"session"
    session.mkdir()
    for output in (session, session/"report", tmp_path):
        with pytest.raises(ValueError, match="outside"):
            validate_output(session, output)
    existing = tmp_path/"existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        validate_output(session, existing)
    assert validate_output(session, tmp_path/"new-report") == tmp_path/"new-report"
    assert not (tmp_path/"new-report").exists()
