from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from scripts import target_state_validity_probe as probe
from tests.training.target_state.test_shards import _write_parent_dataset
from tests.training.target_state.test_sharded_trainer import _FakeLifecycle
from tests.training.target_state.test_dataset_schema import make_record
from training.target_state.config import TargetStateTrainingConfig
from training.target_state.shards import build_target_state_shards


@pytest.fixture
def fixture_run(tmp_path):
    parent = tmp_path / "parent"
    _write_parent_dataset(parent)
    # Synthetic unit-test provenance exercises the Stage B loader contract.
    dataset_manifest = parent / "dataset_manifest.json"
    provenance = json.loads(dataset_manifest.read_text())
    provenance["detector_prediction_source"] = "real_yolo_deployment_output"
    provenance["candidate_id_source"] = "sensor_only_bbox_color_temporal_linker"
    provenance["detector_truth_association"] = "offline_privileged_one_to_one_iou_after_worker_inference"
    provenance["detector_deployment"] = dict(preflight_verified=True, model_family="yolo",
        model_names={"0": "cube"}, model_sha256="a"*64, worker_url="http://127.0.0.1:8011")
    dataset_manifest.write_text(json.dumps(provenance))
    index = build_target_state_shards(parent, tmp_path / "shards", target_shard_size_bytes=1,
        history_size=4, max_history_age_s=2, split_seed=42).shard_index
    config = TargetStateTrainingConfig(dataset_root=parent, output_dir=tmp_path / "models",
        history_size=4, roi_size_px=32, roi_feature_dim=8, geometry_feature_dim=8,
        hidden_dim=8, gru_layers=1, num_workers=0, device="cpu", expected_yolo_model_sha256="a"*64,
        supervision_protocol="projected_center_v2", reference_guard_protocol="rgbd_consistency_v1")
    model = probe._new_model(config, torch.device("cpu")).eval().requires_grad_(False)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.validity_head.bias.fill_(2)
    modeldir = config.output_dir / config.run_name
    modeldir.mkdir(parents=True)
    checkpoint = modeldir / "best.pt"
    metadata = dict(training_stage=config.stage.value, shard_index_sha256=index.index_sha256,
                    parent_dataset_sha256=index.parent_dataset_sha256,
                    supervision_protocol=config.supervision_protocol,
                    reference_guard_protocol=config.reference_guard_protocol)
    torch.save(dict(**metadata, model_state_dict=model.state_dict()), checkpoint)
    manifest = dict(metadata, checkpoint_path=str(checkpoint), checkpoint_sha256=probe.sha256_file(checkpoint),
                    config=json.loads(json.dumps(asdict(config), default=str)),
                    preprocessing={"reference_guard_protocol": config.reference_guard_protocol})
    manifest.pop("reference_guard_protocol")
    (modeldir / "model_manifest.json").write_text(json.dumps(manifest))
    association = tmp_path / "association"
    association.mkdir()
    association_contract = {"index_sha256": index.index_sha256, "parent_dataset_sha256": index.parent_dataset_sha256}
    (association / "report.json").write_text(json.dumps({"complete": True, "contract": association_contract}))
    review = dict(offline_only=True, action="review_only_no_automatic_relabel_or_deletion",
        contract_sha256=probe.digest(association_contract),
        counts={"frame_records": sum(e.frame_count for e in index.shards_for_split("train"))}, findings=[])
    reviewpath = association / "train_review_manifest.json"
    reviewpath.write_text(json.dumps(review))
    pc = tmp_path / "pc.json"
    bridge = tmp_path / "bridge"
    pc.write_text(json.dumps({"bridge_root": str(bridge)}))
    settings = dict(model_manifest=modeldir / "model_manifest.json", shard_index=index.source_path,
        train_review_manifest=reviewpath, pc_trans_root=tmp_path / "unused", pc_trans_config=pc,
        bridge_root=bridge, output_dir=tmp_path / "probe", run_id_prefix="audit_validity_fixture",
        experiment_name="head_v1", device="cpu", num_workers=0, wait_timeout_s=0,
        case_assets_max_gib=.01, fit=dict(epochs=4, batch_size=16, learning_rate=.01,
            weight_decay=0., hard_example_weight=3., seed=42))
    return settings, index, _FakeLifecycle(tmp_path / "shards", bridge)


def test_preflight_uses_train_validation_only_and_writes_nothing(fixture_run):
    settings, index, lifecycle = fixture_run
    plan = probe.prepare(settings, dry_run=True, lifecycle=lifecycle)
    assert plan["splits"] == ["train", "validation"] and plan["test_split_used"] is False
    assert plan["shards"] == 3
    assert not settings["output_dir"].exists() and not lifecycle.requests


def test_train_review_rejects_test_finding(fixture_run):
    settings, index, _ = fixture_run
    path = settings["train_review_manifest"]
    review = json.loads(path.read_text())
    test = index.shards_for_split("test")[0]
    review["findings"] = [dict(shard=test.filename, episode_id=test.episode_ids[0], frame_id="bad", reasons=[])]
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="outside the training"):
        probe.preflight(settings)


def test_window_ambiguity_is_review_not_negative():
    frames = [make_record(i) for i in range(5)]
    frames[0] = replace(frames[0], training_label=None)
    frames[1] = replace(frames[1], training_label=replace(frames[1].training_label, instance_id="another_cube"))
    frames[2] = replace(frames[2], training_label=replace(frames[2].training_label, instance_id="original_cube"))
    seq = SimpleNamespace(history=frames[:-1], reference=frames[-1])
    reasons = probe.review_window(seq, {(frames[2].episode_id, frames[2].frame_id): ["depth_conflict"]})
    assert set(reasons) == {"null_positive_transition_in_window", "multiple_label_instances_in_window", "depth_conflict"}
    assert frames[-1].training_label is not None


def test_prepare_preserves_model_and_commits_before_consume(fixture_run):
    settings, index, lifecycle = fixture_run
    before = probe.sha256_file(settings["model_manifest"].parent / "best.pt")
    consume = lifecycle.consume
    checked = []
    def verify_consume(run_id, filename, *, delete):
        receipt = settings["output_dir"] / "receipts" / (filename + ".pt")
        assert receipt.is_file()
        payload = torch.load(receipt, weights_only=True)
        assert payload["features"].shape[1] == 8
        assert not payload["features"].requires_grad
        assert payload["entry"]["split"] in probe.SPLITS
        checked.append(filename)
        consume(run_id, filename, delete=delete)
    with mock.patch.object(lifecycle, "consume", side_effect=verify_consume):
        result = probe.prepare(settings, lifecycle=lifecycle)
    assert len(checked) == 3 and set(result) == {"train", "validation"}
    assert all("test" not in k for k in lifecycle.requests)
    assert probe.sha256_file(settings["model_manifest"].parent / "best.pt") == before
    assert all((lifecycle.source / e.filename).is_file() for e in index.shards)
    with mock.patch.object(probe, "extract_archive", side_effect=AssertionError("must use receipts")):
        assert probe.prepare(settings, lifecycle=lifecycle) == result


def test_commit_failure_keeps_archive_and_retry_reuses_evidence(fixture_run):
    settings, index, lifecycle = fixture_run
    with mock.patch.object(probe, "save_receipt", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            probe.prepare(settings, lifecycle=lifecycle)
    assert not lifecycle.consumed
    entry = index.shards_for_split("train")[0]
    assert lifecycle._active("audit_validity_fixture.train", entry.filename).is_file()
    with mock.patch.object(lifecycle, "consume", side_effect=RuntimeError("after commit")):
        with pytest.raises(RuntimeError, match="after commit"):
            probe.prepare(settings, lifecycle=lifecycle)
    assert not lifecycle.consumed
    first = settings["output_dir"] / "receipts" / (entry.filename + ".pt")
    assert first.is_file()
    original = probe.extract_archive
    def extract_remaining(archive, **kwargs):
        assert kwargs["entry"].filename != entry.filename
        return original(archive, **kwargs)
    with mock.patch.object(probe, "extract_archive", side_effect=extract_remaining):
        probe.prepare(settings, lifecycle=lifecycle)


def test_corrupted_feature_receipt_cannot_be_trained(fixture_run):
    settings, index, lifecycle = fixture_run
    probe.prepare(settings, lifecycle=lifecycle)
    path = settings["output_dir"] / "receipts" / (index.shards_for_split("train")[0].filename + ".pt")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="receipt SHA256"):
        probe.train(settings)


def synthetic_rows(split, labels, logits):
    rows = []
    for y, p in zip(labels, torch.sigmoid(torch.tensor(logits)).tolist()):
        rows.append(dict(split=split, probe_fit_mask=split == "train", probe_label=bool(y), probe_eligible=True,
            evaluated_visible_target=bool(y), no_target=not y, model_valid=p>=.5, baseline_valid=True,
            model_geometry_valid=True, detected=True, raw_depth_m=4., validity_probability=p,
            model_error_m=.1, baseline_error_m=.5, offline_measurement_supervision_positive=bool(y),
            offline_validity_supervised=True, offline_target_in_output_domain=bool(y),
            failure_flags={"validity_head_rejected": p < .5}, model_failure_reason=None if p>=.5 else "validity_head_rejected"))
    return rows


def test_head_only_fit_improves_without_geometry_or_label_changes():
    # Separable frozen features; original head rejects everything.
    x = torch.tensor([[1., 0.], [-1., 0.]] * 8)
    labels = [1, 0] * 8
    original = {"weight": torch.zeros(1,2), "bias": torch.tensor([-1.])}
    train = (x, synthetic_rows("train", labels, [-1.] * 16))
    val = (x.clone(), synthetic_rows("validation", labels, [-1.] * 16))
    old_rows = json.dumps(val[1], sort_keys=True)
    fit = dict(epochs=15, batch_size=16, learning_rate=.2, weight_decay=0., hard_example_weight=3., seed=42)
    head, report = probe.fit_head(train, val, original, fit)
    assert report["candidate_improved"] and report["best_epoch"] > 0
    assert report["candidate_validation"]["positive_supervision_rejected_count"] == 0
    assert report["candidate_validation"]["no_target_false_positive_count"] == 0
    assert report["test_split_used"] is False and report["promotion_passed"] is False
    assert json.dumps(val[1], sort_keys=True) == old_rows
    assert original["weight"].abs().sum() == 0 and original["bias"].item() == -1
    assert set(head) == {"weight", "bias"}
    with pytest.raises(ValueError, match="TRAIN gradients"):
        probe.fit_head((x, synthetic_rows("test", labels, [-1.]*16)), val, original, fit)


def test_no_improvement_keeps_original_and_masks_unknowns():
    x = torch.zeros(4,2)
    rows = synthetic_rows("train", [1,0,1,0], [-1.]*4)
    rows[2]["probe_fit_mask"] = False
    rows[3]["probe_fit_mask"] = False
    original = {"weight": torch.zeros(1,2), "bias": torch.tensor([-1.])}
    fit = dict(epochs=1, batch_size=4, learning_rate=.001, weight_decay=0., hard_example_weight=3., seed=42)
    head, result = probe.fit_head((x, rows), (x, synthetic_rows("validation", [1,0,1,0], [-1.]*4)), original, fit)
    assert result["best_epoch"] == 0 and not result["candidate_improved"]
    assert result["train_positive_count"] == result["train_negative_count"] == 1
    assert all(torch.equal(head[k], v) for k,v in original.items())
    for row in rows:
        if not row["probe_label"]:
            row["probe_fit_mask"] = False
    with pytest.raises(ValueError, match="no qualified"):
        probe.fit_head((x, rows), (x, synthetic_rows("validation", [1,0,1,0], [-1.]*4)), original, fit)


def test_no_improvement_that_adds_false_positives_or_large_errors_is_admitted():
    base = dict(no_target_false_positive_count=0, out_of_domain_accepted_count=3, accepted_over_1m_count=5)
    assert probe.admissible(base, base)
    for key in base:
        assert not probe.admissible(dict(base, **{key: base[key]+1}), base)


def test_regating_cannot_bypass_sensor_guard():
    rows = synthetic_rows("validation", [1], [-1.])
    rows[0].update(probe_eligible=False)
    rows[0]["failure_flags"]["rgbd_consistency_rejected"] = True
    updated = probe.regate(rows, torch.tensor([.99]))
    assert not updated[0]["model_valid"]
    assert rows[0]["validity_probability"] < .5
