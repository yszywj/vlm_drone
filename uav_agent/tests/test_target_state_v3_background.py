"""Renderer zero is not a missing object ID or a blanket invalid-depth label."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from target_state_v3.association import INSTANCE_EVIDENCE_PROTOCOL, normalize_instances
from target_state_v3.isaac_instances import snapshot_instances
from target_state_v3.storage import EpisodeWriter, verify_episode, verify_instance_evidence
from target_state_v3.verify_tar import file_sha
from training.yolo.isaac_collector import OracleFrameTruth
from tests.test_target_state_v3 import assembled, catalog
from tests.training.target_state.test_isaac_capture import _sample, _truth_object


def renderer_frame():
    mask = np.full((48, 64), 70000, dtype=np.uint32)
    mask[:8] = 0
    depth = np.full(mask.shape, 5., dtype=np.float32)
    depth[:8] = np.inf
    payload = {"data": mask, "info": {"idToLabels": {
        "70000": "/World/CubeV1Collection/cube_0/Body",
        "1": "/World/UAVs/uav_1/MotorFront",  # Real renderer IDs are not semantic IDs.
    }}}
    return payload, depth


def normalize(payload, depth):
    return normalize_instances(payload, shape_hw=(48, 64), catalog=catalog(), raw_depth_m=depth)


def test_missing_zero_uses_raw_no_hit_evidence_without_fabricating_prim():
    payload, depth = renderer_frame()
    original = deepcopy(payload["info"])
    mask, mapping = normalize(payload, depth)
    assert np.array_equal(mask, payload["data"])
    assert payload["info"] == original
    assert mapping["id_to_prim"] == original["idToLabels"]
    assert "0" not in mapping["id_to_prim"]
    assert mapping["instances"]["0"] == {
        "prim_path": None, "object_id": None, "shape": None,
        "kind": "background", "source": INSTANCE_EVIDENCE_PROTOCOL,
    }
    assert mapping["background_evidence"]["0"]["pixel_count"] == 8*64
    assert mapping["instances"]["70000"]["object_id"] == "cube_0"


@pytest.mark.parametrize("bad_depth", [np.nan, -np.inf, 0., -1., 5., 1e10])
def test_invalid_or_clipped_depth_is_not_no_hit_background(bad_depth):
    payload, depth = renderer_frame()
    depth[0, 0] = bad_depth
    with pytest.raises(ValueError, match="raw positive-infinity"):
        normalize(payload, depth)


def test_missing_zero_without_raw_depth_is_still_rejected():
    payload, _ = renderer_frame()
    with pytest.raises(ValueError, match="raw depth evidence"):
        normalize(payload, None)


@pytest.mark.parametrize("raw_mapping", [{}, {"70000": "/World/CubeV1Collection/cube_0/Body"},
                                         {"0": "BACKGROUND"}])
def test_all_zero_frame_with_no_visible_surface_is_not_ready(raw_mapping):
    payload, depth = renderer_frame()
    payload["data"][:] = 0
    payload["info"]["idToLabels"] = raw_mapping
    depth[:] = np.inf
    with pytest.raises(ValueError, match="uninitialized"):
        normalize(payload, depth)


@pytest.mark.parametrize("missing_id", [1, 17, 4294967295])
def test_missing_nonzero_instance_never_becomes_background(missing_id):
    payload, depth = renderer_frame()
    payload["info"]["idToLabels"].pop("1")
    payload["data"][0, 0] = missing_id
    with pytest.raises(ValueError, match=f"instance {missing_id} has no same-frame mapping"):
        normalize(payload, depth)


def test_renderer_id_one_can_be_a_real_cube():
    payload, depth = renderer_frame()
    payload["data"][8:] = 1
    payload["info"]["idToLabels"]["1"] = "/World/CubeV1Collection/cube_0/Body"
    _, mapping = normalize(payload, depth)
    assert mapping["instances"]["1"]["object_id"] == "cube_0"


@pytest.mark.parametrize("name", ["BACKGROUND", "background"])
def test_explicit_background_mapping_cannot_override_raw_surface(name):
    payload, depth = renderer_frame()
    payload["info"]["idToLabels"]["0"] = name
    depth[0, 0] = 5.
    with pytest.raises(ValueError, match="raw positive-infinity"):
        normalize(payload, depth)


@pytest.mark.parametrize("depth", [np.full((64, 48), np.inf), np.ones((48, 64), dtype=np.uint32)])
def test_raw_depth_requires_correct_shape_and_float_dtype(depth):
    payload, _ = renderer_frame()
    with pytest.raises(ValueError, match="raw instance depth"):
        normalize(payload, depth)


def snapshot_fixture():
    payload, depth = renderer_frame()
    published = depth.copy()
    published[~np.isfinite(published)] = np.nan
    sample = replace(_sample(), depth_to_image_plane_m=published, render_frame_id=(5, 30))
    frame = {"rgb": sample.rgb, "distance_to_image_plane": depth,
             "rendering_time": sample.timestamp_s, "instance_id_segmentation": payload}
    sensor = SimpleNamespace(camera=SimpleNamespace(get_current_frame=lambda clone: frame),
        _render_frame_id_from_frame=lambda f: (5, 30), _rgb_from_frame=lambda f: f["rgb"])
    truth = OracleFrameTruth(sample, objects=(_truth_object("cube_0"),))
    driver = SimpleNamespace(_roots={"cube_0": SimpleNamespace(GetPath=lambda: "/World/CubeV1Collection/cube_0")})
    return sensor, truth, driver, frame


def test_snapshot_retains_raw_inf_and_does_not_alias_camera_buffers():
    sensor, truth, driver, frame = snapshot_fixture()
    mask, mapping, raw_depth = snapshot_instances(sensor, truth, driver)
    assert np.isposinf(raw_depth[:8]).all()
    assert np.isnan(truth.camera_sample.depth_to_image_plane_m[:8]).all()
    assert mapping["render_frame_id"] == [5, 30]
    frame["distance_to_image_plane"][:] = 1.
    frame["instance_id_segmentation"]["data"][:] = 42
    assert raw_depth[10, 10] == 5.
    assert mask[10, 10] == 70000


def test_snapshot_does_not_mistake_clipped_finite_depth_for_background():
    sensor, truth, driver, frame = snapshot_fixture()
    frame["distance_to_image_plane"][0, 0] = 1e10
    # Published depth is still NaN, so checking it alone would silently pass.
    with pytest.raises(ValueError, match="raw positive-infinity"):
        snapshot_instances(sensor, truth, driver)


def test_zero_does_not_bypass_same_frame_barrier():
    sensor, truth, driver, frame = snapshot_fixture()
    frame["rendering_time"] += .1
    with pytest.raises(RuntimeError, match="barrier"):
        snapshot_instances(sensor, truth, driver)


def write_raw_episode(root):
    sensor, truth, driver, _ = snapshot_fixture()
    mask, mapping, raw_depth = snapshot_instances(sensor, truth, driver)
    sample = truth.camera_sample
    rows = assembled(sample, mask, mapping, capture="capture_0", episode="episode_0")
    contract = {"instance_evidence_protocol": INSTANCE_EVIDENCE_PROTOCOL}
    writer = EpisodeWriter(root, episode_id="episode_0", expected_captures=1, contract=contract)
    writer.append(capture_id="capture_0", sample=sample, mask=mask, mapping=mapping,
                  records=rows, measurement_diagnostics={}, raw_depth_m=raw_depth)
    writer.finalize()
    return mask, mapping, raw_depth, contract


def test_raw_depth_is_persisted_only_in_oracle_evidence_and_replayed(tmp_path):
    root = tmp_path/"episode"
    _, _, raw_depth, _ = write_raw_episode(root)
    with np.load(root/"oracle/capture_0.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"instance_id", "raw_depth_to_image_plane_m"}
        assert np.array_equal(archive["raw_depth_to_image_plane_m"], raw_depth)
    assert np.isnan(np.load(root/"depth/capture_0.npy")[:8]).all()
    assert verify_episode(root)["physical_captures"] == 1


def test_raw_depth_cannot_be_dropped_or_evidence_counts_forged(tmp_path):
    mask, mapping, raw_depth, contract = write_raw_episode(tmp_path/"episode")
    with pytest.raises(ValueError, match="missing retained raw depth"):
        verify_instance_evidence(mask, mapping, None, contract)
    mapping["background_evidence"]["0"]["pixel_count"] += 1
    with pytest.raises(ValueError, match="evidence replay mismatch"):
        verify_instance_evidence(mask, mapping, raw_depth, contract)


def test_rehashed_but_physically_inconsistent_raw_evidence_is_rejected(tmp_path):
    root = tmp_path/"episode"
    mask, _, raw_depth, _ = write_raw_episode(root)
    raw_depth[0, 0] = 5.
    asset = root/"oracle/capture_0.npz"
    np.savez_compressed(asset, instance_id=mask, raw_depth_to_image_plane_m=raw_depth)
    manifest_path = root/"episode_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["oracle/capture_0.npz"] = file_sha(asset)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="raw positive-infinity"):
        verify_episode(root)


def test_empty_supervised_dataset_is_not_reported_as_verified_parity(tmp_path):
    from unittest.mock import patch
    from scripts.check_target_state_v3_pilot import check
    from target_state_v3.storage import make_archive
    root = tmp_path/"session"
    episode = "episode_0"
    folder = root/episode
    _, _, _, contract = write_raw_episode(folder)
    contract.update(physical_captures=1, detector_deployment={"model_sha256": "8"*64})
    manifest_path = folder/"episode_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contract"] = contract
    manifest_path.write_text(json.dumps(manifest))
    entry = make_archive(folder, root/"archives/shard_v3_check_000000.tar")
    entry["episode_id"] = episode
    (root/"session.json").write_text(json.dumps({"complete": True, "episodes": [entry], "contract": contract}))
    with patch("scripts.check_target_state_v3_pilot.TargetStateV3Dataset", return_value=SimpleNamespace(sequences=())):
        report = check(root)
    assert report["structural_checks_passed"] is True
    assert report["counts"]["supervised_windows"] == 0
    assert report["training_runtime_preprocessing_identical"] is None
    assert report["training_runtime_preprocessing_check"] == "not_run_no_supervised_windows"
