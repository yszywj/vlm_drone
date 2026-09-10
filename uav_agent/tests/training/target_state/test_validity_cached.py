import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from scripts import fit_target_state_validity_cached as cached
from scripts import target_state_validity_probe as producer
from tests.training.target_state.test_validity_probe import fixture_run, synthetic_rows


def settings_fit(**overrides):
    return dict(epochs=8, batch_size=16, learning_rate=.2, weight_decay=0.,
                hard_example_weight=1., seed=42, class_balance_power=0., anchor_logit_weight=.1, **overrides)


def rows_for(split, labels, logits):
    rows = synthetic_rows(split, labels, logits)
    for row in rows:
        row.update(reference_input_valid=True, probe_review_reasons=[])
    return rows


def synthetic_cache():
    x = torch.tensor([[1., 0.], [-1., 0.]] * 8)
    original = {"weight": torch.zeros(1,2), "bias": torch.tensor([-1.])}
    data = {s: (x.clone(), rows_for(s, [1,0]*8, [-1.]*16)) for s in producer.SPLITS}
    return data, original


@pytest.mark.parametrize("power,ratio", [(0.,1.), (.25,4**.25), (1.,4.)])
def test_configurable_balance_has_exact_class_ratio(power, ratio):
    fit = settings_fit()
    fit["class_balance_power"] = power
    labels = torch.tensor([1.,1.,1.,1.,0.])
    weights, stats = cached.sample_weights(labels, labels*.8+.1, fit)
    assert weights[-1] / weights[0] == pytest.approx(ratio)
    assert weights.mean() == pytest.approx(1)
    assert stats["negative_loss_weight_fraction"] == pytest.approx(ratio/(4+ratio))


def test_anchor_penalty_is_trainable_but_teacher_is_detached():
    logits = torch.tensor([2.,-2.], requires_grad=True)
    old = torch.zeros(2, requires_grad=True)
    total,bce,anchor = cached.objective(logits, torch.tensor([1.,0.]), torch.ones(2), old, .1)
    assert anchor.item() == 4
    assert total.item() == pytest.approx(bce.item()+.4)
    total.backward()
    assert old.grad is None
    assert logits.grad[0] > 0  # penalty pushes this changed score towards its teacher
    assert logits.grad[1] < 0


@pytest.mark.parametrize("key,value", [("class_balance_power",1.01), ("class_balance_power",-1),
    ("anchor_logit_weight",float("nan")), ("learning_rate",0), ("epochs",True), ("hard_example_weight",0)])
def test_invalid_fit_config_is_rejected(key,value):
    fit = settings_fit()
    fit[key] = value
    with pytest.raises(ValueError):
        cached.validate_fit(fit)


def test_head_fit_preserves_source_geometry_and_validation_is_not_in_loss():
    prepared, original = synthetic_cache()
    saved = {k:v.clone() for k,v in original.items()}
    old_rows = json.dumps(prepared["validation"][1], sort_keys=True)
    fit = settings_fit()
    state, result = cached.fit_head(prepared["train"], prepared["validation"], original, fit)
    assert result["test_split_used"] is False and result["promotion_passed"] is False
    assert result["geometry_parameters_changed"] is False
    assert result["acceptance_threshold"] == .5
    assert set(state) == {"weight","bias"}
    assert all(torch.equal(saved[k],v) for k,v in original.items())
    assert json.dumps(prepared["validation"][1], sort_keys=True) == old_rows
    # Original probabilities remain -1-logit everywhere with a zero-weight head.
    # Changing validation features changes selection, never the SGD trajectory.
    _, different = cached.fit_head(prepared["train"],
        (prepared["validation"][0]*10, prepared["validation"][1]), original, fit)
    assert [e["train_loss"] for e in result["history"]] == [e["train_loss"] for e in different["history"]]
    with pytest.raises(ValueError, match="contamination"):
        cached.fit_head((prepared["train"][0], rows_for("test",[1,0]*8,[-1.]*16)), prepared["validation"], original, fit)


def test_unknown_supervision_cannot_enter_training_mask():
    prepared, original = synthetic_cache()
    x, rows = prepared["train"]
    rows[0]["offline_validity_supervised"] = False
    with pytest.raises(ValueError, match="supervision contract"):
        cached.fit_head((x,rows), prepared["validation"], original, settings_fit())
    rows[0]["probe_fit_mask"] = False
    _, result = cached.fit_head((x,rows), prepared["validation"], original, settings_fit())
    assert result["weight_statistics"]["train_positive_count"] == 7


def test_complete_cache_verifies_without_transfer_or_cache_mutation(fixture_run):
    settings,index,lifecycle = fixture_run
    producer.prepare(settings,lifecycle=lifecycle)
    root = settings["output_dir"]
    snapshot = {p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}
    suite = dict(suite_name="new_suite",experiments=[dict(name="natural",fit=settings_fit())])
    with mock.patch.object(producer,"PCTransCLI",side_effect=AssertionError("no transfer")), \
         mock.patch.object(producer,"prepare",side_effect=AssertionError("no extraction")):
        plan = cached.run(settings,suite,dry_run=True)
    assert plan["receipt_count"] == 3
    assert plan["pc_upload_required"] is False and plan["execution_device"] == "cpu"
    assert snapshot == {p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_bad_hash_and_test_receipt_never_trigger_automatic_reextraction(fixture_run):
    settings,index,lifecycle = fixture_run
    producer.prepare(settings,lifecycle=lifecycle)
    root = settings["output_dir"]
    cp = root / "cache_manifest.json"
    content = cp.read_bytes()
    c = json.loads(content)
    c["receipt_sha256"][index.shards_for_split("test")[0].filename] = "0"*64
    cp.write_text(json.dumps(c))
    with pytest.raises(ValueError,match="split receipts"):
        cached.read_cache(settings)
    cp.write_bytes(content)
    p = root / "receipts" / (index.shards_for_split("train")[0].filename+".pt")
    p.write_bytes(b"broken")
    with mock.patch.object(producer,"prepare",side_effect=AssertionError("no fallback")):
        with pytest.raises(ValueError,match="receipt SHA256"):
            cached.read_cache(settings)


def test_changed_producer_contract_cannot_be_bypassed(fixture_run):
    settings,index,lifecycle = fixture_run
    producer.prepare(settings,lifecycle=lifecycle)
    preflight = producer.preflight
    def changed(cfg):
        a,b,c,d,contract = preflight(cfg)
        contract["source_sha256"]["scripts/target_state_validity_probe.py"] = "0"*64
        return a,b,c,d,contract
    with mock.patch.object(producer,"preflight",side_effect=changed):
        with pytest.raises(ValueError,match="unchanged V1"):
            cached.read_cache(settings)


def test_suite_persists_and_resumes_without_refit(tmp_path):
    prepared, original = synthetic_cache()
    identity = dict(cache_manifest_sha256="a"*64,base_checkpoint_sha256="b"*64,receipt_count=3)
    cfg = {"output_dir":tmp_path}
    suite = dict(suite_name="suite_v2",experiments=[dict(name="natural",fit=settings_fit())])
    with mock.patch.object(cached,"read_cache",return_value=(prepared,original,identity)):
        result = cached.run(cfg,suite)
        path = tmp_path / "suite_v2/natural/best_validity_head.pt"
        artifact = torch.load(path,weights_only=True)
        assert artifact["deployable"] is False
        assert artifact["base_checkpoint_sha256"] == identity["base_checkpoint_sha256"]
        before = path.read_bytes()
        with mock.patch.object(cached,"fit_head",side_effect=AssertionError("already complete")):
            assert cached.run(cfg,suite) == result
        assert path.read_bytes() == before
        suite["experiments"][0]["fit"]["anchor_logit_weight"] = .2
        with pytest.raises(ValueError,match="suite contract changed"):
            cached.run(cfg,suite)


def test_failed_fit_is_restartable_and_source_head_is_unchanged(tmp_path):
    prepared, original = synthetic_cache()
    identity = dict(base_checkpoint_sha256="b"*64)
    before = {k:v.clone() for k,v in original.items()}
    with mock.patch.object(producer,"save_receipt",side_effect=OSError("disk full")):
        with pytest.raises(OSError,match="disk full"):
            cached.run_experiment(tmp_path,prepared,original,settings_fit(),identity)
    assert not (tmp_path / "report.json").exists()
    result = cached.run_experiment(tmp_path,prepared,original,settings_fit(),identity)
    assert result["complete"]
    assert all(torch.equal(before[k],v) for k,v in original.items())


def test_suite_can_resume_after_lock_creation_but_never_overwrite_unowned_data(tmp_path):
    prepared, original = synthetic_cache()
    identity = dict(cache_manifest_sha256="a"*64,base_checkpoint_sha256="b"*64,receipt_count=3)
    cfg = {"output_dir":tmp_path}
    suite = dict(suite_name="suite_v2",experiments=[dict(name="natural",fit=settings_fit())])
    output = tmp_path / "suite_v2"
    with producer._exclusive_run_lock(output):
        pass  # simulated crash before suite_contract.json is written
    with mock.patch.object(cached,"read_cache",return_value=(prepared,original,identity)):
        assert cached.run(cfg,suite,dry_run=True)["cache_verified"]
        user_file = output / "notes.txt"
        user_file.write_text("keep this")
        with pytest.raises(ValueError,match="without ownership"):
            cached.run(cfg,suite)
        assert user_file.read_text() == "keep this"


def test_no_admissible_improvement_keeps_original_head():
    prepared, original = synthetic_cache()
    fit = settings_fit()
    fit.update(epochs=1,learning_rate=1e-10)
    head, result = cached.fit_head(prepared["train"], prepared["validation"], original, fit)
    assert result["best_epoch"] == 0 and not result["candidate_improved"]
    assert all(torch.equal(head[k],v) for k,v in original.items())


def test_config_loads_relative_base_and_rejects_duplicate_experiments(fixture_run):
    settings,_,_ = fixture_run
    parent = settings["output_dir"].parent
    base = parent / "base.yaml"
    base.write_text(json.dumps(settings, default=str))
    suite = dict(base_config="base.yaml",suite_name="new_suite",fit=settings_fit(),
                 experiments=[dict(name="natural",overrides={})])
    p = parent / "suite.yaml"
    p.write_text(json.dumps(suite))
    loaded, config = cached.load_suite(p)
    assert loaded["output_dir"] == settings["output_dir"]
    assert config["experiments"][0]["fit"]["class_balance_power"] == 0
    suite["experiments"].append(suite["experiments"][0])
    p.write_text(json.dumps(suite))
    with pytest.raises(ValueError,match="duplicate"):
        cached.load_suite(p)
