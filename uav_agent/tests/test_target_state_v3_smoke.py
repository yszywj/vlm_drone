"""Diagnostic V3 regression: no source mutations, no held-out optimization."""
from dataclasses import replace
import json

import pytest
import torch

from datasets.target_state.dataset import split_for_episode
from target_state_v3.data import TargetStateV3Dataset
from target_state_v3.measurement import require_v3_artifact
from target_state_v3.verify_tar import file_sha
from target_state_v3_derived.ground_overlay import check_overlay, create_overlay, derive_episode, derived_view
from target_state_v3_smoke.runner import (SmokeOptions, collate, evaluate, forward_sensor_only,
    make_model, reference_objective, run_smoke, separate_output, train_fixed_steps, zero_correction)
from training.target_state.model import TemporalRayDepthNet
from tests.test_target_state_v3 import config_for
from tests.test_target_state_v3_reassociation import source_fixture


@pytest.fixture
def materialized(tmp_path):
    torch.set_num_threads(2)
    source = tmp_path/"source"
    folder, episode = source_fixture(source, count=9)
    _, records = derive_episode(folder, json.loads((folder/"episode_manifest.json").read_text()))
    cfg = replace(config_for(folder), roi_size_px=32)
    original = TargetStateV3Dataset(cfg, episode_root=folder, split=split_for_episode(episode, seed=42))
    view = derived_view(original, records)
    return [view[i] for i in range(len(view))]


def small_model():
    return make_model({"roi_feature_dim": 8, "geometry_feature_dim": 8, "hidden_dim": 16, "gru_layers": 1})


def test_labels_and_oracle_metadata_are_never_forward_inputs(materialized):
    model = TemporalRayDepthNet(roi_feature_dim=8, geometry_feature_dim=8, hidden_dim=16, gru_layers=1).eval()
    batch = collate(materialized)
    original = forward_sensor_only(model, batch)
    poisoned = {k: v.clone() for k, v in batch.items()}
    for key in ("target_position_world_m", "target_depth_m", "history_center_uv_px", "occlusion_ratio"):
        poisoned[key].fill_(float("nan"))
    poisoned["oracle_instance_mask"] = torch.full((1,), float("nan"))
    changed = forward_sensor_only(model, poisoned)
    for key, value in original.as_dict().items():
        assert torch.equal(value, changed.as_dict()[key])


def test_initial_corrections_are_exactly_zero_and_untrained_heads_frozen(materialized):
    model = small_model().eval()
    batch = collate(materialized)
    output = forward_sensor_only(model, batch)
    assert not torch.count_nonzero(output.delta_uv_px)
    assert not torch.count_nonzero(output.depth_residual_m)
    assert not torch.count_nonzero(output.position_log_variance)
    assert torch.all(output.measurement_valid_logit == -20.)
    assert all(not p.requires_grad for name, p in model.named_parameters()
               if name.startswith(("log_variance_head.", "validity_head.")))
    assert reference_objective(output, batch)[0] == reference_objective(zero_correction(batch), batch)[0]


def test_loss_uses_reference_time_not_a_static_target_in_old_frames(materialized):
    batch = collate(materialized)
    output = zero_correction(batch)
    before = reference_objective(output, batch)[0]
    batch["history_center_uv_px"][:, :-1] += 10000.
    batch["history_camera_position_world_m"][:, :-1] += 1000.
    assert torch.equal(reference_objective(output, batch)[0], before)


@pytest.mark.parametrize("problem", ["nan", "negative_depth", "negative_label", "inconsistent_center"])
def test_bad_predictions_or_supervision_fail_instead_of_vanishing_from_loss(materialized, problem):
    batch = collate(materialized)
    output = zero_correction(batch)
    if problem == "nan":
        output.delta_uv_px[0, 0] = float("nan")
    elif problem == "negative_depth":
        output.depth_residual_m[0] = -1000.
    elif problem == "negative_label":
        batch["measurement_valid"][0] = False
    else:
        batch["history_center_uv_px"][0, -1, 0] += 10.
    with pytest.raises(ValueError):
        reference_objective(output, batch)


def test_regression_can_fit_and_gradients_reach_all_trainable_groups(materialized):
    torch.manual_seed(42)
    model = small_model()
    before = evaluate(None, materialized, device="cpu", batch_size=3)
    flow = train_fixed_steps(model, materialized, SmokeOptions(steps=100, batch_size=3,
        learning_rate=.005, threads=2, report_every=100))
    after = evaluate(model, materialized, device="cpu", batch_size=3)
    assert flow["passed"]
    assert after["position_mean_m"] < before["position_mean_m"]*.5
    assert after["loss"] < before["loss"]*.5
    assert model.validity_head.bias.item() == -20.
    assert not torch.count_nonzero(model.log_variance_head.weight)


@pytest.mark.parametrize("kwargs", [{"steps": 1001}, {"batch_size": 0}, {"threads": 9},
                                    {"learning_rate": float("nan")}, {"seed": -1}])
def test_budget_options_are_bounded(kwargs):
    with pytest.raises(ValueError):
        SmokeOptions(**kwargs)


def test_explicit_opt_in_and_separate_new_output_required(tmp_path):
    with pytest.raises(ValueError, match="acknowledge"):
        run_smoke(tmp_path/"missing", tmp_path/"output", SmokeOptions(steps=2))
    source = tmp_path/"source"
    source.mkdir()
    for output in (source, source/"nested", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            separate_output(output, source)
    existing = tmp_path/"existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        separate_output(existing, source)


def sealed_overlay(tmp_path):
    source, derived = tmp_path/"source", tmp_path/"derived"
    source_fixture(source, count=7)
    create_overlay(source, derived)
    report = check_overlay(derived)
    (derived/"acceptance_report.json").write_text(json.dumps(report))
    return source, derived


def test_end_to_end_diagnostic_artifact_is_not_deployable_and_inputs_stay_immutable(tmp_path):
    source, derived = sealed_overlay(tmp_path)
    original_hashes = {p: file_sha(p) for root in (source, derived) for p in root.rglob("*") if p.is_file()}
    output = tmp_path/"smoke"
    report = run_smoke(derived, output, SmokeOptions(steps=2, batch_size=1, threads=2),
                       acknowledge_diagnostic_only=True)
    assert report["complete"] and report["checkpoint_reload_verified"]
    assert report["gradient_flow"]["passed"]
    assert report["baseline"]["train"]["count"] == 1
    assert report["final"]["validation"]["count"] == 0
    assert not report["validity_trained"] and not report["covariance_calibrated"]
    assert not report["production_approved"] and not report["ready_for_bulk_collection"]
    assert report["test_model_evaluation"] == "not_run"
    assert original_hashes == {p: file_sha(p) for p in original_hashes}
    artifact = torch.load(output/"diagnostic_checkpoint.pt", weights_only=True)
    assert artifact["model_type"] != "temporal_ray_depth_residual"
    with pytest.raises(ValueError, match="promotion"):
        require_v3_artifact(artifact)
    manifest = json.loads((output/"diagnostic_manifest.json").read_text())
    assert manifest["fixed_steps_no_validation_selection"]
    assert all(w["split"] == "train" for w in manifest["data"]["windows"] if w["used_for_optimization"])


def test_changed_acceptance_is_rejected_before_creating_training_output(tmp_path):
    _, derived = sealed_overlay(tmp_path)
    path = derived/"acceptance_report.json"
    report = json.loads(path.read_text())
    report["derived_dataset"]["positive_measurement_windows"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differs from fresh replay"):
        run_smoke(derived, tmp_path/"output", SmokeOptions(steps=2), acknowledge_diagnostic_only=True)
    assert not (tmp_path/"output").exists()
