"""Ground-role sidecars retain raw evidence and replay every label change."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from datasets.target_state.dataset import split_for_episode
from target_state_v3 import DATA_PROTOCOL
from target_state_v3.association import (AssociationPolicy, INSTANCE_EVIDENCE_PROTOCOL,
                                       InstanceFrameAssembler, normalize_instances)
from target_state_v3.data import TargetStateV3Dataset
from target_state_v3.measurement import MeasurementPolicy
from target_state_v3.storage import EpisodeWriter, make_archive, verify_episode
from target_state_v3.verify_tar import file_sha
from target_state_v3_derived.ground_overlay import (
    GROUND, ROOT, check_overlay, create_overlay, derive_capture,
)
from tests.test_target_state_v3 import catalog, config_for
from tests.training.target_state.test_isaac_capture import _sample, _truth_object, _detection, _response, _uav
from training.target_state.collector import VerifiedYoloDeployment
from training.yolo.isaac_collector import OracleFrameTruth


def capture_fixture(i=0, *, assembler=None, ground_path=GROUND, ground_id=13,
                    duplicate=False, extra_unknown=False, no_hit=False):
    sample = replace(_sample(1.+.2*i), render_frame_id=(i+1, 30))
    mask = np.full((48, 64), 70000, np.uint32)
    mask[:, :24] = ground_id  # ~17% of the inner detection ROI, >5% unknown gate
    paths = {str(ground_id): ground_path, "70000": "/World/CubeV1Collection/cube_0/Body"}
    raw = sample.depth_to_image_plane_m.copy()
    if extra_unknown:
        mask[:, 24:28] = 31
        paths["31"] = "/World/Unknown/geom"
    if no_hit:
        mask[:8] = 0
        raw[:8] = np.inf
        published = raw.copy()
        published[~np.isfinite(published)] = np.nan
        sample = replace(sample, depth_to_image_plane_m=published)
    mask, mapping = normalize_instances({"data": mask, "info": {"idToLabels": paths}},
        shape_hw=mask.shape, catalog=catalog(), raw_depth_m=raw)
    mapping.update(render_frame_id=list(sample.render_frame_id), timestamp_s=sample.timestamp_s, offline_only=True)
    capture, episode = f"capture_{i}", "s20260910_episode_000000"
    detections = (_detection(1), _detection(2)) if duplicate else (_detection(1),)
    rows = (assembler or InstanceFrameAssembler()).assemble(capture_id=capture, episode_id=episode,
        truth=OracleFrameTruth(sample, objects=(_truth_object("cube_0"),)),
        response=_response(sample, detections, frame_id=capture), uav=_uav(), mask=mask, mapping=mapping)
    return {"capture_id": capture, "episode_id": episode, "records": rows,
            "oracle_only": {"mapping": mapping}}, mask, raw, sample


def source_fixture(root, *, count=7):
    episode = "s20260910_episode_000000"
    contract = {"protocol": DATA_PROTOCOL, "physical_captures": count,
        "instance_evidence_protocol": INSTANCE_EVIDENCE_PROTOCOL,
        "association_policy": AssociationPolicy().to_dict(),
        "measurement_preprocessing": MeasurementPolicy().contract(),
        "source_sha256": {"env/scene.py": file_sha(ROOT/"env/scene.py")},
        "detector_deployment": VerifiedYoloDeployment("http://127.0.0.1:8011", "yolo", ((0, "cube"),),
                                                      "8"*64).to_manifest_dict()}
    folder = root/episode
    writer = EpisodeWriter(folder, episode_id=episode, expected_captures=count, contract=contract)
    assembler = InstanceFrameAssembler()
    for i in range(count):
        payload, mask, raw, sample = capture_fixture(i, assembler=assembler, no_hit=True)
        writer.append(capture_id=payload["capture_id"], sample=sample, mask=mask,
            mapping=payload["oracle_only"]["mapping"], records=payload["records"],
            measurement_diagnostics={}, raw_depth_m=raw)
    stats = writer.finalize()
    entry = make_archive(folder, root/"archives/shard_v3_ground_review_000000.tar")
    entry.update(episode_id=episode, stats=stats)
    (root/"session.json").write_text(json.dumps({"protocol": DATA_PROTOCOL, "complete": True,
        "contract": contract, "episodes": [entry], "summary": stats}))
    return folder, episode


@pytest.mark.parametrize("ground_id", [1, 13, 100001])
def test_exact_ground_role_recovers_original_same_frame_label_without_sensor_edits(ground_id):
    payload, mask, raw, _ = capture_fixture(ground_id=ground_id, no_hit=True)
    before = deepcopy(payload)
    body, records = derive_capture(payload, mask, raw)
    assert payload == before
    override = body["overrides"][0]
    assert override["old_association"]["status"] == "unresolved"
    assert override["new_association"]["status"] == "matched_cube"
    assert override["training_label"] == payload["records"][1]["record"]["training_label"]
    assert override["label_source_frame_id"] == payload["records"][1]["record"]["frame_id"]
    assert body["ground_evidence"][0]["instance_id"] == str(ground_id)
    assert payload["oracle_only"]["mapping"]["instances"]["0"]["kind"] == "background"
    for original, derived in zip(payload["records"], records):
        expected = deepcopy(original["record"])
        actual = derived.to_dict()
        for key in ("training_label", "association_review_required"):
            expected.pop(key, None)
            actual.pop(key, None)
        assert expected == actual
    assert records[1].detector_prediction.candidate_id is None  # no truth-derived candidate


@pytest.mark.parametrize("other_path", ["/World/Ground/geom2", "/World/Ground/geometry", "/Other/Ground/geom"])
def test_similar_path_never_treated_as_ground(other_path):
    payload, mask, raw, _ = capture_fixture(ground_path=other_path)
    body, records = derive_capture(payload, mask, raw)
    assert not body["ground_evidence"]
    assert records[0].association_review_required
    assert records[0].training_label is None


@pytest.mark.parametrize("depth", [np.nan, np.inf, -np.inf, 0., -1.])
def test_ground_requires_real_positive_finite_surface_depth(depth):
    payload, mask, raw, _ = capture_fixture()
    raw[0, 0] = depth
    with pytest.raises(ValueError, match="positive finite raw depth"):
        derive_capture(payload, mask, raw)


@pytest.mark.parametrize("option", ["duplicate", "extra_unknown"])
def test_ground_rule_does_not_waive_duplicate_or_unknown_rejection(option):
    payload, mask, raw, _ = capture_fixture(**{option: True})
    body, records = derive_capture(payload, mask, raw)
    assert body["ground_evidence"]
    assert all(r.association_review_required and r.training_label is None
               for r in records if r.detector_prediction.detected)


def test_missing_label_evidence_or_forged_original_association_fails():
    payload, mask, raw, _ = capture_fixture()
    bad = deepcopy(payload)
    bad["records"] = bad["records"][:1]
    with pytest.raises(ValueError, match="no original same-frame label"):
        derive_capture(bad, mask, raw)
    payload["records"][0]["association"]["status"] = "matched_cube"
    with pytest.raises(ValueError, match="cannot be replayed"):
        derive_capture(payload, mask, raw)


def test_output_is_separate_and_never_overwrites(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    for output in (source, source/"derived", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            create_overlay(source, output)
    existing = tmp_path/"existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        create_overlay(source, existing)


def test_full_overlay_replays_actual_windows_without_touching_source(tmp_path):
    torch.set_num_threads(2)
    source, output = tmp_path/"source", tmp_path/"derived"
    folder, episode = source_fixture(source)
    before = {str(p.relative_to(source)): file_sha(p) for p in source.rglob("*") if p.is_file()}
    manifest = create_overlay(source, output)
    report = check_overlay(output)
    assert before == {str(p.relative_to(source)): file_sha(p) for p in source.rglob("*") if p.is_file()}
    assert manifest["annotation_only"] and not manifest["ready_for_training"]
    assert report["complete"] and report["overlay_replay_verified"]
    assert report["baseline_detections"] == {"unresolved": 7}
    assert report["derived_detections"] == {"matched_cube": 7}
    assert report["baseline_dataset"]["windows"] == 0
    assert report["derived_dataset"]["windows"] == 1
    assert report["derived_dataset"]["positive_measurement_windows"] == 1
    assert report["history_size"] == 6 and report["max_history_age_s"] == 2.
    assert not report["ready_for_training"] and not report["production_approved"]
    assert report["needs_manual_visual_review"]
    assert verify_episode(folder)["unresolved"] == 7
    assert len(TargetStateV3Dataset(config_for(folder), episode_root=folder,
                                   split=split_for_episode(episode, seed=42))) == 0


def test_forged_override_is_rejected_even_if_its_checksum_is_refreshed(tmp_path):
    source, output = tmp_path/"source", tmp_path/"derived"
    source_fixture(source, count=1)
    manifest = create_overlay(source, output)
    entry = manifest["episodes"][0]
    path = output/entry["filename"]
    body = json.loads(path.read_text())
    body["captures"][0]["overrides"][0]["label_source_frame_id"] = "invented_other_frame"
    path.write_text(json.dumps(body))
    entry["sha256"] = file_sha(path)
    (output/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="replay mismatch"):
        check_overlay(output)


@pytest.mark.parametrize("tamper", ["source", "code", "missing_episode", "unlisted_file"])
def test_overlay_fails_closed_on_source_or_manifest_changes(tmp_path, tamper):
    source, output = tmp_path/"source", tmp_path/"derived"
    source_fixture(source, count=1)
    manifest = create_overlay(source, output)
    if tamper == "source":
        state = source/"session.json"
        state.write_text(state.read_text()+"\n")
    elif tamper == "code":
        manifest["derivation_code_sha256"] = {}
    elif tamper == "missing_episode":
        manifest["episodes"] = []
    elif tamper == "unlisted_file":
        (output/"extra.json").write_text("{}")
    (output/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        check_overlay(output)
