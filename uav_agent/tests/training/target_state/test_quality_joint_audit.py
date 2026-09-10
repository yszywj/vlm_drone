from copy import deepcopy
import json
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

from scripts import audit_target_state_quality_test as joint
from tests.training.target_state.test_quality_cached import config as toy_config, synthetic, identity
from tests.training.target_state.test_validity_probe import fixture_run


@pytest.fixture
def candidate(tmp_path):
    prepared, original = synthetic()
    cfg = toy_config()
    settings = {"output_dir":tmp_path/"cache"}
    with mock.patch.object(joint.quality.cached,"read_cache",return_value=(prepared,original,identity())):
        result = joint.quality.run(settings,cfg)
    name = result["selected_experiment"]
    assert name is not None
    path = settings["output_dir"]/cfg["experiment_name"]/name/"best_quality_bundle.pt"
    options = dict(candidate_bundle=path,candidate_sha256=joint.producer.sha256_file(path),
        base_checkpoint_sha256=identity()["base_checkpoint_sha256"], expected_epoch=result["experiments"][name]["best_epoch"])
    return options, settings, cfg, prepared, original


def test_pinned_candidate_verifies_and_replays_without_fitting(candidate):
    options, settings, cfg, prepared, original = candidate
    with mock.patch.object(joint.quality,"warmup_validity",side_effect=AssertionError("no training")), \
         mock.patch.object(joint.quality,"fit_quality",side_effect=AssertionError("no training")), \
         mock.patch.object(joint.audit,"PCTransCLI",side_effect=AssertionError("no requests")):
        heads, report = joint.verify_candidate(options,settings,cfg,prepared,original,identity())
    assert report["validation_replay_matched"]
    assert not heads.quality.training and all(not p.requires_grad for p in heads.quality.parameters())


@pytest.mark.parametrize("field,value", [("candidate_sha256","0"*64), ("base_checkpoint_sha256","0"*64), ("expected_epoch",999)])
def test_changed_pin_or_epoch_is_rejected(candidate,field,value):
    options,settings,cfg,prepared,original = candidate
    options[field] = value
    with pytest.raises(ValueError):
        joint.verify_candidate(options,settings,cfg,prepared,original,identity())


def test_last_bundle_and_forged_selected_score_cannot_pass(candidate):
    options,settings,cfg,prepared,original = candidate
    original_path = options["candidate_bundle"]
    options["candidate_bundle"] = original_path.with_name("last_quality_bundle.pt")
    options["candidate_sha256"] = joint.producer.sha256_file(options["candidate_bundle"])
    with pytest.raises(ValueError,match="provenance"):
        joint.verify_candidate(options,settings,cfg,prepared,original,identity())
    options["candidate_bundle"] = original_path
    options["candidate_sha256"] = joint.producer.sha256_file(original_path)
    path = original_path.parent/"report.json"
    payload = json.loads(path.read_text())
    payload["candidate_validation"]["positive_supervision_rejected_count"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="reproduced"):
        joint.verify_candidate(options,settings,cfg,prepared,original,identity())


@pytest.mark.parametrize("field,value", [("quality_enabled",False), ("selected_candidate",False),
    ("quality_threshold",.4), ("validity_threshold",.4), ("deployable",True)])
def test_unselected_bundle_or_gate_changes_fail(candidate,field,value):
    options,_,_,_,original = candidate
    bundle = torch.load(options["candidate_bundle"],weights_only=True)
    bundle[field] = value
    with pytest.raises(ValueError,match="unsupported"):
        joint.FrozenHeads(bundle,original["weight"].shape[1])


def test_quality_inference_is_truth_independent_and_both_gates_apply(candidate):
    options,settings,cfg,prepared,original = candidate
    heads,_ = joint.verify_candidate(options,settings,cfg,prepared,original,identity())
    x,rows = prepared["validation"]
    unchanged,logits = heads.apply(x,rows)
    modified = deepcopy(rows)
    for row in modified:
        for field in ("model_error_m","target_position_world_m","offline_target_depth_m","offline_target_center_uv_px",
                "probe_label","probe_fit_mask","offline_validity_supervised","offline_measurement_supervision_positive",
                "no_target","evaluated_visible_target","episode_id","candidate_id","tracker_id","prim_path","initial_region","motion_seed"):
            row[field] = "offline sentinel"
    changed,other_logits = heads.apply(x,modified)
    assert torch.equal(logits,other_logits)
    assert [r["model_valid"] for r in unchanged] == [r["model_valid"] for r in changed]
    rows[0]["probe_eligible"] = False
    gated,_ = heads.apply(x,rows)
    assert not gated[0]["model_valid"]


@pytest.fixture
def ctx(fixture_run):
    settings,index,lifecycle = fixture_run
    config,_,model,_,_ = joint.producer.preflight(settings)
    original = joint.quality.clone_state(model.validity_head)
    dim = original["weight"].shape[1]+len(joint.quality.EXTRA_FEATURES)
    head = joint.quality.QualityHead(torch.zeros(dim),torch.ones(dim),0)
    bundle = dict(artifact_type="offline_conditional_quality_heads_only",protocol=joint.quality.PROTOCOL,
        deployable=False,selected_candidate=True,quality_enabled=True,quality_threshold=.5,validity_threshold=.5,
        quality_good_error_max_m=1.,extra_features=list(joint.quality.EXTRA_FEATURES),input_dim=dim,
        hidden_dim=0,validity_state_dict=original,quality_state_dict=joint.quality.clone_state(head))
    options = joint.audit.ShardedTrainingOptions(shard_index_path=index.source_path,pc_trans_root=settings["pc_trans_root"],
        pc_trans_config=settings["pc_trans_config"],bridge_root=settings["bridge_root"],run_id_prefix="audit_quality_fixture",wait_timeout_s=0)
    test = index.shards_for_split("test")[0]
    value = dict(config=config,options=options,index=index,model=model,heads=joint.FrozenHeads(bundle,dim-14),original_head=original,
        manifest={"test_metrics":{}},output=settings["output_dir"].parent/"joint_audit",
        owner=settings["bridge_root"]/"control/audits/audit_quality_fixture",contract={"test_fixture":True},
        case_options=joint.evidence.CaseExportOptions(1024**2,1.,tuple(test.episode_ids)),
        plan={"shards":1,"sequences":test.sequence_count,"candidate_frozen_before_test":True})
    return value,lifecycle,test


def test_full_cpu_audit_commits_before_consumption_and_resumes_without_inference(ctx):
    value,client,entry = ctx
    before = {k:v.clone() for k,v in value["model"].state_dict().items()}
    consume = client.consume
    def verify(run_id,filename,*,delete):
        receipt,_ = joint.read_receipt(value,entry)
        assert receipt["evaluation_only"] and receipt["case_exports"]["cases"]
        consume(run_id,filename,delete=delete)
    with mock.patch.object(client,"consume",side_effect=verify), \
         mock.patch.object(torch.optim,"AdamW",side_effect=AssertionError("no optimizer")):
        report = joint.audit_test(value,client,torch.device("cpu"))
    assert set(client.requests) == {"audit_quality_fixture.test"}
    assert client.requests["audit_quality_fixture.test"] == (entry.filename,)
    assert (client.source/entry.filename).exists()
    assert report["complete"] and not report["promotion_passed"]
    assert not report["test_used_for_fit_or_selection"] and not report["fresh_independent_test"]
    assert report["comparison"]["geometry_predictions_identical"]
    assert report["comparison"]["scores"]["original"]["diagnostics"]["sample_count"] == entry.sequence_count
    assert all(torch.equal(v,before[k]) for k,v in value["model"].state_dict().items())
    assert not value["model"].validity_head._forward_pre_hooks
    for filename,digest in report["output_sha256"].items():
        assert joint.producer.sha256_file(value["output"]/filename) == digest
    with mock.patch.object(joint,"evaluate_archive",side_effect=AssertionError("must use receipt")):
        assert joint.audit_test(value,client,torch.device("cpu")) == report


def test_dry_run_does_not_request_evaluate_or_write(ctx):
    value,client,_ = ctx
    with mock.patch.object(joint,"preflight",return_value=value), \
         mock.patch.object(joint.audit,"PCTransCLI",side_effect=AssertionError("no PC")), \
         mock.patch.object(joint,"evaluate_archive",side_effect=AssertionError("no TEST evaluation")):
        assert joint.run({"device":"cpu"},dry_run=True) == value["plan"]
    assert not client.requests and not value["output"].exists() and not value["owner"].exists()


def test_no_train_or_validation_shard_can_be_evaluated(ctx):
    value,client,_ = ctx
    entry = value["index"].shards_for_split("validation")[0]
    with pytest.raises(ValueError,match="only evaluates TEST"):
        joint.evaluate_archive(client.source/entry.filename,ctx=value,entry=entry,device=torch.device("cpu"))


@pytest.mark.parametrize("phase", ["receipt","checksum","consume"])
def test_crash_recovery_at_each_commit_boundary(ctx,phase):
    value,client,entry = ctx
    if phase == "receipt":
        patch = mock.patch.object(joint.audit,"save_receipt",side_effect=OSError("disk full"))
    elif phase == "checksum":
        patch = mock.patch.object(joint.producer,"_atomic_write_json",side_effect=OSError("disk full"))
    else:
        patch = mock.patch.object(client,"consume",side_effect=OSError("interrupted consume"))
    with patch:
        with pytest.raises(OSError):
            joint.audit_test(value,client,torch.device("cpu"))
    assert not client.consumed and client._active("audit_quality_fixture.test",entry.filename).exists()
    if phase == "consume":
        with mock.patch.object(joint,"evaluate_archive",side_effect=AssertionError("reuse committed receipt")):
            assert joint.audit_test(value,client,torch.device("cpu"))["complete"]
    else:
        assert joint.audit_test(value,client,torch.device("cpu"))["complete"]


@pytest.mark.parametrize("damage", ["receipt","asset","missing_checksum"])
def test_corrupt_committed_evidence_never_silently_recomputed(ctx,damage):
    value,client,entry = ctx
    joint.audit_test(value,client,torch.device("cpu"))
    path,checksum = joint.receipt_paths(value,entry)
    if damage == "receipt":
        path.write_bytes(b"broken")
    elif damage == "missing_checksum":
        checksum.unlink()
    else:
        r=torch.load(path,weights_only=True)
        asset = value["output"]/next(iter(r["case_exports"]["assets"]))
        asset.write_bytes(b"broken")
    with mock.patch.object(joint,"evaluate_archive",side_effect=AssertionError("do not overwrite corrupted evidence")):
        with pytest.raises(ValueError):
            joint.audit_test(value,client,torch.device("cpu"))


def test_asset_failure_before_consume_keeps_server_archive(ctx):
    value,client,entry = ctx
    with mock.patch.object(joint.evidence,"verify_case_assets",side_effect=ValueError("missing asset")):
        with pytest.raises(ValueError,match="missing asset"):
            joint.audit_test(value,client,torch.device("cpu"))
    assert not client.consumed and client._active("audit_quality_fixture.test",entry.filename).exists()


def test_changed_contract_and_unowned_output_stop_before_requests(ctx):
    value,client,_ = ctx
    value["output"].mkdir()
    (value["output"]/"notes.txt").write_text("keep")
    with pytest.raises(ValueError,match="without ownership"):
        joint.quality.check_output(value["output"],value["contract"])
    assert not client.requests


def test_changed_only_cases_are_exported_without_forging_requested_tags(ctx):
    value,client,entry = ctx
    materialized, receipt = joint.evaluate_archive(client.source/entry.filename,ctx=value,entry=entry,device=torch.device("cpu"))
    dataset,loader = joint.audit._dataset_loader(config=value["config"],dataset_root=materialized.dataset_root,split="test",
        device=torch.device("cpu"),shuffle=False,generator_seed=None,split_seed=value["index"].split_seed)
    old,new = deepcopy(receipt["original_rows"]),deepcopy(receipt["candidate_rows"])
    for a,b in zip(old,new):
        a.update(evaluated_visible_target=False,no_target=False,model_valid=False,baseline_valid=False)
        b.update(evaluated_visible_target=False,no_target=False,model_valid=True,baseline_valid=False)
    result=joint.export_joint_cases(old,new,dataset.sequences,dataset_root=materialized.dataset_root,output_dir=value["output"],
        entry=entry,options=joint.evidence.CaseExportOptions(0))
    assert len(result["cases"]) == entry.sequence_count
    assert all(c["tags"] == ["acceptance_changed"] for c in result["cases"])
    joint.audit._release_loader(loader)
    joint.audit.cleanup_materialized_shard(materialized)


def test_test_summary_exposes_replaced_error_cases_and_does_not_select():
    prepared,_=synthetic()
    old=deepcopy(prepared["validation"][1][:3])
    for i,r in enumerate(old):
        r.update(model_valid=i==0,model_position_world_m=[1.,2.,3.],model_error_m=2. if i<2 else .1)
    new=deepcopy(old)
    new[0]["model_valid"]=False
    new[1]["model_valid"]=True
    report=joint.summarize_pair(old,new)
    assert report["safety_case_changes"]["1"]["original_count"] == report["safety_case_changes"]["1"]["candidate_count"] == 1
    assert report["safety_case_changes"]["1"]["new_ids"] == [new[1]["sequence_id"]]
    assert "selected_epoch" not in report and "promotion_passed" not in report
    new[1]["model_position_world_m"][0]=99
    with pytest.raises(ValueError,match="frozen geometry"):
        joint.summarize_pair(old,new)


def test_default_cli_contract_pins_known_candidate_and_test_only(tmp_path):
    root=Path(__file__).resolve().parents[3]
    path=root/"configs/target_state/audit_quality_joint_test_50k.yaml"
    cfg=joint.load_config(path)
    assert cfg["expected_epoch"] == 19
    assert cfg["candidate_sha256"] == "584c8902d513005b3064cdaf1673d9ba01c0e212decd2f6109243eeafc1eea9d"
    assert cfg["run_id_prefix"]+".test" == "audit_quality_joint_50k_v1.test"
    raw=yaml.safe_load(path.read_text())
    raw["splits"]=["train","test"]
    bad=tmp_path/"bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match="config keys"):
        joint.load_config(bad)
