from copy import deepcopy
import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from scripts import fit_target_state_quality_cached as quality
from tests.training.target_state.test_validity_cached import rows_for
from tests.training.target_state.test_validity_probe import fixture_run


def config():
    return dict(experiment_name="quality_test", validity_fit=dict(epochs=1, batch_size=64,
        learning_rate=.1, weight_decay=0., hard_example_weight=1., seed=42,
        class_balance_power=0., anchor_logit_weight=0.),
        quality_fit=dict(epochs=12, batch_size=64, learning_rate=.1, weight_decay=0., seed=42, class_balance_power=1.),
        experiments=[dict(name="linear", hidden_dim=0), dict(name="mlp", hidden_dim=8)])


def synthetic():
    x = torch.tensor([[1., 0., 0.], [1., 1., 0.], [-1., 0., 0.], [1., -1., 0.]]*8)
    original = {"weight": torch.zeros(1, 3), "bias": torch.tensor([-.1])}
    prepared = {}
    for split in quality.producer.SPLITS:
        rows = rows_for(split, [1,1,0,1]*8, [-.1]*32)
        for i, row in enumerate(rows):
            row.update(episode_id=f"{split}_ep_{i//4}", sequence_id=f"{split}_seq_{i}",
                image_size_wh=[640,480], delta_uv_px=[.1,.2], anchor_uv_px=[320,240],
                bbox_xyxy_normalized=[.4,.4,.6,.6], position_variance_m2=[.01,.02,.03],
                corrected_depth_m=4.2, depth_residual_m=.2,
                model_error_m=2. if i%4 == 1 else .1)
        prepared[split] = x.clone(), rows
    return prepared, original


def identity():
    return dict(cache_manifest_sha256="a"*64, base_checkpoint_sha256="b"*64, receipt_count=3)


def test_quality_features_have_no_offline_truth_or_id_dependency():
    prepared, _ = synthetic()
    features, rows = prepared["train"]
    before = quality.quality_inputs(features, rows)
    contaminated = deepcopy(rows)
    for row in contaminated:
        for key in ("model_error_m", "baseline_error_m", "target_position_world_m", "offline_target_depth_m",
                    "offline_target_center_uv_px", "offline_target_in_output_domain", "probe_label", "probe_fit_mask",
                    "offline_validity_supervised", "offline_measurement_supervision_positive", "no_target",
                    "evaluated_visible_target", "episode_id", "sequence_id", "tracker_id", "candidate_id",
                    "occlusion_ratio", "probe_review_reasons", "initial_region", "motion_seed", "prim_path"):
            row[key] = "forbidden-field-sentinel"
    assert torch.equal(before, quality.quality_inputs(features, contaminated))
    assert before.shape[1] == features.shape[1]+14
    assert not before.requires_grad


@pytest.mark.parametrize("field,value", [("raw_depth_m", float("nan")),
    ("position_variance_m2", [1.,0.,1.]), ("image_size_wh", [0,480]),
    ("delta_uv_px", [float("inf"),0.]), ("bbox_xyxy_normalized", [1.,2.])])
def test_invalid_runtime_features_fail_closed(field, value):
    prepared, _ = synthetic()
    x, rows = prepared["train"]
    rows[0][field] = value
    with pytest.raises(ValueError):
        quality.quality_inputs(x, rows)
    rows[0]["reference_input_valid"] = False
    assert quality.quality_inputs(x, rows)[0, -14:].eq(0).all()


def test_quality_supervision_does_not_relabel_validity_or_use_unknown_negatives():
    prepared, _ = synthetic()
    rows = prepared["train"][1]
    rows[0]["model_error_m"] = 1.0
    rows[1]["model_error_m"] = 1.00001
    rows[2]["model_error_m"] = float("nan")  # no target: NOT a geometric risk label
    rows[3].update(probe_fit_mask=False, offline_validity_supervised=False)
    before = deepcopy(rows)
    mask, labels = quality.quality_supervision(rows)
    assert mask[:4].tolist() == [True, True, False, False]
    assert labels[:4].tolist() == [1.,0.,0.,0.]
    assert rows[1]["probe_label"] is True
    assert json.dumps(rows, sort_keys=True) == json.dumps(before, sort_keys=True)
    rows[1]["model_error_m"] = float("nan")
    with pytest.raises(ValueError, match="supervision error"):
        quality.quality_supervision(rows)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_quality_labels_refuse_non_train_split(split):
    prepared, _ = synthetic()
    rows = prepared["train"][1]
    rows[0]["split"] = split
    with pytest.raises(ValueError, match="TRAIN only"):
        quality.quality_supervision(rows)


def test_weights_and_train_normalization_are_explicit():
    prepared, original = synthetic()
    data = quality.prepare_data(prepared, original, config())
    assert data["statistics"]["accurate_count"] == 16
    assert data["statistics"]["over_1m_count"] == 8
    assert data["statistics"]["negative_loss_weight_fraction"] == pytest.approx(.5)
    assert data["statistics"]["negative_to_positive_weight_ratio"] == 2
    assert data["statistics"]["masked_count"] == 8
    assert data["weights"].mean() == pytest.approx(1)
    train = data["inputs"]["train"][data["mask"]]
    assert torch.equal(data["mean"], train.mean(0))
    assert (data["scale"] >= 1e-4).all()
    with pytest.raises(ValueError, match="both accurate"):
        quality.quality_weights(torch.ones(3), 1.)


def test_episode_overlap_and_forged_unknown_fit_mask_are_rejected():
    prepared, original = synthetic()
    prepared["validation"][1][0]["episode_id"] = prepared["train"][1][0]["episode_id"]
    with pytest.raises(ValueError, match="episode overlap"):
        quality.prepare_data(prepared, original, config())
    prepared, original = synthetic()
    prepared["train"][1][0]["offline_validity_supervised"] = False
    with pytest.raises(ValueError, match="supervision contract"):
        quality.prepare_data(prepared, original, config())


def test_nonfinite_parameters_cannot_be_saved():
    head = quality.QualityHead(torch.zeros(3), torch.ones(3), 0)
    with torch.no_grad():
        head.layers[-1].bias.fill_(float("inf"))
    with pytest.raises(ValueError, match="non-finite head"):
        quality.clone_state(head)


def test_guard_validity_and_quality_are_all_required_and_rows_immutable():
    prepared, _ = synthetic()
    rows = prepared["validation"][1][:4]
    rows[0]["probe_eligible"] = False
    before = deepcopy(rows)
    gated = quality.joint_rows(rows, torch.tensor([1., .4, 1., .5]), torch.tensor([1., 1., .4, .5]))
    assert [r["model_valid"] for r in gated] == [False, False, False, True]
    assert gated[2]["model_failure_reason"] == "quality_head_rejected"
    assert gated[1]["model_failure_reason"] == "validity_head_rejected"
    assert rows == before
    with pytest.raises(ValueError, match="probabilities"):
        quality.joint_rows(rows, torch.ones(4), torch.tensor([1.,1.,float("nan"),1.]))


def test_validation_changes_never_affect_loss_normalization_or_final_weights():
    prepared, original = synthetic()
    cfg = config()
    data = quality.prepare_data(prepared, original, cfg)
    proposal, _ = quality.warmup_validity(prepared["train"], original, cfg["validity_fit"])
    states, report, _ = quality.fit_quality(prepared, original, proposal, data, cfg["quality_fit"], 0)
    altered = deepcopy(prepared)
    altered["validation"][0].mul_(10)
    for row in altered["validation"][1]:
        row["model_error_m"] = 3.  # may change validation selection, not TRAIN
    other_data = quality.prepare_data(altered, original, cfg)
    other_states, other_report, _ = quality.fit_quality(altered, original, proposal, other_data, cfg["quality_fit"], 0)
    assert torch.equal(data["mean"], other_data["mean"])
    assert torch.equal(data["scale"], other_data["scale"])
    assert [r["train_loss"] for r in report["history"]] == [r["train_loss"] for r in other_report["history"]]
    assert all(torch.equal(v, other_states["last"]["quality_state_dict"][k]) for k,v in states["last"]["quality_state_dict"].items())
    assert report["geometry_parameters_changed"] is False
    assert original["weight"].eq(0).all() and original["bias"].item() == pytest.approx(-.1)
    contaminated = {**prepared, "test": prepared["validation"]}
    with pytest.raises(ValueError, match="only TRAIN"):
        quality.prepare_data(contaminated, original, cfg)


def test_warmup_replays_previous_mild_fit_without_changing_source():
    prepared, original = synthetic()
    cfg = config()
    before = deepcopy(prepared)
    proposal, warmup = quality.warmup_validity(prepared["train"], original, cfg["validity_fit"])
    _, previous = quality.cached.fit_head(prepared["train"], prepared["validation"], original, cfg["validity_fit"])
    score = quality.producer.validation_score(quality.producer.regate(prepared["validation"][1],
        quality.validity_probabilities(prepared["validation"][0], proposal)))
    assert score == previous["history"][0]["validation"]
    assert warmup["train_loss"] == pytest.approx(previous["history"][0]["train_loss"]["objective"])
    assert prepared["train"][1] == before["train"][1]
    assert prepared["validation"][1] == before["validation"][1]


@pytest.mark.parametrize("key,value", [("epochs",0), ("epochs",True), ("learning_rate",0),
    ("class_balance_power",1.01), ("weight_decay",float("nan"))])
def test_invalid_quality_fit_config_rejected(key, value):
    cfg = config()
    cfg["quality_fit"][key] = value
    with pytest.raises(ValueError):
        quality.validate_config(cfg)


def test_config_roundtrip_and_duplicate_or_multiepoch_warmup_rejected(tmp_path, fixture_run):
    settings, _, _ = fixture_run
    path = tmp_path / "quality.yaml"
    base = tmp_path / "base.yaml"
    base.write_text(json.dumps(settings, default=str))
    cfg = config()
    path.write_text(json.dumps({"base_config":"base.yaml", **cfg}))
    loaded, parsed = quality.load_config(path)
    assert loaded["output_dir"] == settings["output_dir"] and parsed == cfg
    cfg["experiments"].append(cfg["experiments"][0])
    with pytest.raises(ValueError, match="duplicate"):
        quality.validate_config(cfg)
    cfg = config()
    cfg["validity_fit"]["epochs"] = 2
    with pytest.raises(ValueError, match="exactly one"):
        quality.validate_config(cfg)


def test_real_producer_cache_hash_checks_are_not_bypassed(fixture_run):
    settings, index, lifecycle = fixture_run
    quality.producer.prepare(settings, lifecycle=lifecycle)
    root = settings["output_dir"]
    with mock.patch.object(quality.producer, "prepare", side_effect=AssertionError("no extraction")), \
         mock.patch.object(quality.producer, "PCTransCLI", side_effect=AssertionError("no PC")):
        prepared, original, _ = quality.cached.read_cache(settings)
        assert set(prepared) == {"train", "validation"}
        receipt = root / "receipts" / (index.shards_for_split("train")[0].filename+".pt")
        receipt.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="receipt SHA256"):
            quality.run(settings, config(), dry_run=True)


def test_dry_run_uses_cpu_cache_without_writes_uploads_or_training(tmp_path):
    prepared, original = synthetic()
    with mock.patch.object(quality.cached, "read_cache", return_value=(prepared,original,identity())), \
         mock.patch.object(quality.producer, "PCTransCLI", side_effect=AssertionError("no PC")), \
         mock.patch.object(quality, "warmup_validity", side_effect=AssertionError("no fit")):
        result = quality.run({"output_dir":tmp_path}, config(), dry_run=True)
    assert result["pc_upload_required"] is False and result["test_split_used"] is False
    assert result["quality_input_dim"] == 17 and result["execution_device"] == "cpu"
    assert list(tmp_path.iterdir()) == []


def test_artifact_replay_completion_and_idempotent_resume(tmp_path):
    prepared, original = synthetic()
    cfg = config()
    with mock.patch.object(quality.cached, "read_cache", return_value=(prepared,original,identity())):
        result = quality.run({"output_dir":tmp_path}, cfg)
        assert result["complete"] and result["promotion_passed"] is False
        assert result["candidate_improved"]  # exercise selection, not only fallback
        for exp in result["experiments"].values():
            if exp["candidate_improved"]:
                assert quality.producer.admissible(exp["candidate_validation"], result["baseline_validation"])
                assert exp["candidate_validation"]["positive_supervision_rejected_count"] < result["baseline_validation"]["positive_supervision_rejected_count"]
        output = tmp_path / cfg["experiment_name"]
        for exp in cfg["experiments"]:
            report = json.loads((output / exp["name"] / "report.json").read_text())
            for kind in ("best", "last"):
                artifact = torch.load(output / exp["name"] / f"{kind}_quality_bundle.pt", weights_only=True)
                assert artifact["deployable"] is False
                vx = quality.quality_inputs(*prepared["validation"])
                p = quality.validity_probabilities(prepared["validation"][0], artifact["validity_state_dict"])
                if artifact["quality_enabled"]:
                    state = artifact["quality_state_dict"]
                    head = quality.QualityHead(state["mean"], state["scale"], exp["hidden_dim"])
                    head.load_state_dict(state, strict=True)
                    with torch.no_grad():
                        predicted = quality.joint_rows(prepared["validation"][1], p, torch.sigmoid(head(vx)))
                else:
                    assert artifact["quality_state_dict"] is None
                    predicted = quality.producer.regate(prepared["validation"][1], p)
                expected = report["candidate_validation"] if kind == "best" else report["history"][-1]["validation"]
                assert quality.validation_diagnostics(predicted) == expected
        snapshots = {p:p.read_bytes() for p in output.rglob("*.pt")}
        with mock.patch.object(quality, "warmup_validity", side_effect=AssertionError("already complete")), \
             mock.patch.object(quality, "fit_quality", side_effect=AssertionError("already complete")):
            assert quality.run({"output_dir":tmp_path}, cfg) == result
        assert all(p.read_bytes()==data for p,data in snapshots.items())
        cfg["quality_fit"]["epochs"] += 1
        with pytest.raises(ValueError, match="contract changed"):
            quality.run({"output_dir":tmp_path}, cfg)


def test_no_improvement_really_disables_quality_and_keeps_original():
    prepared, original = synthetic()
    cfg = config()
    data = quality.prepare_data(prepared, original, cfg)
    # Original rejects everything: a quality gate AND this proposal cannot improve recall.
    states, report, changes = quality.fit_quality(prepared, original, original, data, cfg["quality_fit"], 0)
    assert report["best_epoch"] == 0 and not report["candidate_improved"]
    assert states["best"]["quality_state_dict"] is None and not states["best"]["quality_enabled"]
    assert all(torch.equal(v, states["best"]["validity_state_dict"][k]) for k,v in original.items())
    assert changes["best"] == []


def test_unowned_and_symlink_outputs_are_protected_and_lock_only_resumes(tmp_path):
    path = tmp_path / "suite"
    with quality.producer._exclusive_run_lock(path):
        pass
    quality.check_output(path, {})
    user = path / "notes.txt"
    user.write_text("keep")
    with pytest.raises(ValueError, match="without ownership"):
        quality.check_output(path, {})
    assert user.read_text() == "keep"
    symlink = tmp_path / "link"
    symlink.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        quality.check_output(symlink, {})


def test_interruption_resumes_and_corrupt_saved_artifact_stops(tmp_path):
    prepared, original = synthetic()
    cfg = config()
    cfg["experiments"] = cfg["experiments"][:1]
    with mock.patch.object(quality.cached, "read_cache", return_value=(prepared,original,identity())):
        with mock.patch.object(quality.producer, "save_receipt", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                quality.run({"output_dir":tmp_path}, cfg)
        output = tmp_path / cfg["experiment_name"] / "linear"
        assert not (output / "report.json").exists()
        result = quality.run({"output_dir":tmp_path}, cfg)
        assert result["complete"]
        (output / "last_quality_bundle.pt").write_bytes(b"broken")
        with pytest.raises(ValueError, match="artifact SHA256"):
            quality.run({"output_dir":tmp_path}, cfg)
