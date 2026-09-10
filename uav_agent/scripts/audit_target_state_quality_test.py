#!/usr/bin/env python3
"""Pinned quality candidate vs original model on TEST; no fit or deployment.

This is a historical TEST audit, not a new independent-scene generalization
claim. TEST features are evaluation-only and never enter the TRAIN cache.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_target_state_sharded as audit
from scripts import fit_target_state_quality_cached as quality
from scripts import target_state_audit_evidence as evidence

producer = quality.producer
PROTOCOL = "pinned_quality_joint_test_audit_v1"
KEYS = {"selection_config", "candidate_bundle", "candidate_sha256", "base_checkpoint_sha256",
        "expected_epoch", "run_id_prefix", "output_dir", "device", "num_workers", "wait_timeout_s",
        "case_assets_max_gib", "case_episode_ids"}


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_config(path):
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or set(cfg) != KEYS:
        raise ValueError("invalid joint TEST config keys")
    for key in ("selection_config", "candidate_bundle", "output_dir"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = p if p.is_absolute() else path.resolve().parent / p
    for key in ("candidate_sha256", "base_checkpoint_sha256"):
        if not valid_sha(cfg[key]):
            raise ValueError(f"invalid pinned {key}")
    if not quality.cached.safe_name(cfg["run_id_prefix"]) or not cfg["run_id_prefix"].startswith("audit_quality_"):
        raise ValueError("use a dedicated audit_quality_* prefix")
    for key, minimum in (("expected_epoch", 1), ("num_workers", 0)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < minimum:
            raise ValueError(f"invalid {key}")
    for key in ("wait_timeout_s", "case_assets_max_gib"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"invalid {key}")
    if (not isinstance(cfg["case_episode_ids"], list) or any(not quality.cached.safe_name(v) for v in cfg["case_episode_ids"])
            or len(set(cfg["case_episode_ids"])) != len(cfg["case_episode_ids"])):
        raise ValueError("invalid case_episode_ids")
    torch.device(cfg["device"])
    return cfg


class FrozenHeads:
    def __init__(self, bundle, feature_dim):
        if (bundle.get("artifact_type") != "offline_conditional_quality_heads_only"
                or bundle.get("protocol") != quality.PROTOCOL or bundle.get("deployable") is not False
                or bundle.get("selected_candidate") is not True or bundle.get("quality_enabled") is not True
                or bundle.get("quality_threshold") != .5 or bundle.get("validity_threshold") != .5
                or bundle.get("quality_good_error_max_m") != 1.0
                or bundle.get("extra_features") != list(quality.EXTRA_FEATURES)
                or bundle.get("input_dim") != feature_dim + len(quality.EXTRA_FEATURES)):
            raise ValueError("unsupported/unselected quality bundle or changed gate")
        hidden = bundle["hidden_dim"]
        if isinstance(hidden, bool) or not isinstance(hidden, int) or not 0 <= hidden <= 128:
            raise ValueError("invalid quality architecture")
        self.validity = {k: v.detach().cpu().clone() for k, v in bundle["validity_state_dict"].items()}
        if (set(self.validity) != {"weight", "bias"} or self.validity["weight"].shape != (1, feature_dim)
                or self.validity["bias"].shape != (1,) or any(not torch.isfinite(v).all() for v in self.validity.values())):
            raise ValueError("invalid frozen validity head")
        state = bundle["quality_state_dict"]
        dim = bundle["input_dim"]
        if (state["mean"].shape != (dim,) or state["scale"].shape != (dim,)
                or any(not torch.isfinite(v).all() for v in state.values()) or not (state["scale"] >= 1e-4).all()):
            raise ValueError("invalid saved quality normalization/parameters")
        self.quality = quality.QualityHead(state["mean"], state["scale"], hidden)
        self.quality.load_state_dict(state, strict=True)
        self.quality.eval().requires_grad_(False)

    @torch.no_grad()
    def apply(self, features, rows):
        validity_logits = F.linear(features, self.validity["weight"], self.validity["bias"]).flatten()
        quality_logits = self.quality(quality.quality_inputs(features, rows))
        if not torch.isfinite(validity_logits).all() or not torch.isfinite(quality_logits).all():
            raise ValueError("non-finite joint TEST logits")
        candidate = quality.joint_rows(rows, torch.sigmoid(validity_logits), torch.sigmoid(quality_logits))
        # min(logits)>=0 is precisely the two unchanged >=0.5 gates, not a
        # calibrated joint probability. Only the official metric adapter uses it.
        return candidate, torch.minimum(validity_logits, quality_logits)


def verify_candidate(cfg, settings, selection, prepared, original, identity):
    path = cfg["candidate_bundle"]
    if path.is_symlink() or producer.sha256_file(path) != cfg["candidate_sha256"]:
        raise ValueError("pinned candidate SHA256 mismatch")
    root = settings["output_dir"] / selection["experiment_name"]
    summary_path = root / "report.json"
    summary = json.loads(summary_path.read_text())
    name = summary.get("selected_experiment")
    exp = next((e for e in selection["experiments"] if e["name"] == name), None)
    contract = summary["contract"]
    if (exp is None or path.resolve() != (root/name/"best_quality_bundle.pt").resolve()
            or summary.get("complete") is not True or summary.get("candidate_improved") is not True
            or summary.get("selection_split") != "validation" or summary.get("test_split_used") is not False
            or contract["config"] != selection or contract["cache"] != identity
            or contract["source_sha256"] != quality.source_hashes()
            or identity["base_checkpoint_sha256"] != cfg["base_checkpoint_sha256"]):
        raise ValueError("candidate selection/cache/base-model provenance mismatch")
    exp_contract = {**contract, "experiment": exp}
    report = quality.load_completed(path.parent, exp_contract)
    if (report is None or report["best_epoch"] != cfg["expected_epoch"] or not report["candidate_improved"]
            or report["test_split_used"] is not False
            or summary["experiments"][name]["artifact_sha256"] != report["artifact_sha256"]):
        raise ValueError("candidate best-epoch/selection report mismatch")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if (bundle["epoch"] != cfg["expected_epoch"] or bundle["contract"] != exp_contract
            or bundle["base_checkpoint_sha256"] != cfg["base_checkpoint_sha256"]):
        raise ValueError("candidate bundle identity mismatch")
    heads = FrozenHeads(bundle, original["weight"].shape[1])
    # Verification ONLY: no fitting, gradient or validation-derived normalization.
    norm = quality.prepare_data(prepared, original, selection)
    if not torch.equal(heads.quality.mean, norm["mean"]) or not torch.equal(heads.quality.scale, norm["scale"]):
        raise ValueError("saved normalization differs from qualified TRAIN")
    candidate, _ = heads.apply(*prepared["validation"])
    baseline = quality.validation_diagnostics(prepared["validation"][1])
    score = quality.validation_diagnostics(candidate)
    if (baseline != report["baseline_validation"] or score != report["candidate_validation"]
            or not producer.admissible(score, baseline)
            or score["positive_supervision_rejected_count"] >= baseline["positive_supervision_rejected_count"]):
        raise ValueError("selected validation result cannot be reproduced")
    changes = json.loads((path.parent/"validation_changes.json").read_text())["best"]
    if canonical(quality.changed_rows(prepared["validation"][1], candidate)) != changes:
        raise ValueError("selected validation case replay mismatch")
    return heads, {"selection_report_sha256": producer.sha256_file(summary_path),
        "candidate_report_sha256": producer.sha256_file(path.parent/"report.json"),
        "selected_experiment": name, "selected_epoch": cfg["expected_epoch"],
        "validation_replay_matched": True, "baseline_validation": baseline, "candidate_validation": score}


def preflight(cfg):
    settings, selection = quality.load_config(cfg["selection_config"])
    prepared, original, identity = quality.cached.read_cache(settings)
    heads, selection_evidence = verify_candidate(cfg, settings, selection, prepared, original, identity)
    model_cfg, index, model, _, feature_contract = producer.preflight(settings)
    if producer.digest(feature_contract) != identity["feature_contract_sha256"]:
        raise ValueError("base-model/cache contract changed during preflight")
    manifest = json.loads(settings["model_manifest"].read_text())
    if "test_metrics" not in manifest:
        raise ValueError("base model manifest lacks original TEST metrics")
    model_cfg = replace(model_cfg, device=cfg["device"], num_workers=cfg["num_workers"])
    options = audit.ShardedTrainingOptions(shard_index_path=settings["shard_index"],
        pc_trans_root=settings["pc_trans_root"], pc_trans_config=settings["pc_trans_config"],
        bridge_root=settings["bridge_root"], run_id_prefix=cfg["run_id_prefix"], wait_timeout_s=cfg["wait_timeout_s"])
    output = cfg["output_dir"]
    for protected in (settings["output_dir"], settings["model_manifest"].parent, options.bridge_root):
        a, b = output.resolve(), protected.resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("joint audit output overlaps protected cache/model/bridge")
    entries = index.shards_for_split("test")
    if not entries or set(cfg["case_episode_ids"]) - {e for s in entries for e in s.episode_ids}:
        raise ValueError("missing TEST shards or case episode outside TEST")
    case_options = evidence.CaseExportOptions(int(cfg["case_assets_max_gib"]*1024**3), 1., tuple(cfg["case_episode_ids"]))
    sources = {**feature_contract["source_sha256"], **quality.source_hashes(),
               str(Path(__file__).relative_to(ROOT)): producer.sha256_file(Path(__file__))}
    contract = {"protocol": PROTOCOL, "config": canonical(cfg | {k:str(cfg[k]) for k in ("selection_config","candidate_bundle","output_dir")}),
        "source_sha256": sources, "torch_version": str(torch.__version__), "feature_cache": identity,
        "candidate_sha256": cfg["candidate_sha256"], "base_checkpoint_sha256": cfg["base_checkpoint_sha256"],
        "model_manifest_sha256": producer.sha256_file(settings["model_manifest"]), "index_sha256": index.index_sha256,
        "selection_evidence": selection_evidence, "case_max_bytes": case_options.max_bytes,
        "test_used_for_fit_or_selection": False, "evaluation_splits": ["test"],
        "quality_threshold": .5, "validity_threshold": .5, "quality_error_limit_m": 1.}
    owner = options.bridge_root/"control/audits"/options.run_id_prefix
    for directory in (output, owner):
        quality.check_output(directory, contract)
    plan = {"candidate_path": str(cfg["candidate_bundle"]), "candidate_sha256": cfg["candidate_sha256"],
        "selected_epoch": cfg["expected_epoch"], "base_checkpoint_sha256": cfg["base_checkpoint_sha256"],
        "validation_replay_matched": True, "run_id": options.run_id_prefix+".test",
        "shards": len(entries), "sequences": sum(e.sequence_count for e in entries),
        "archive_bytes": sum(e.archive_size_bytes for e in entries), "output_dir": str(output),
        "case_assets_max_bytes": case_options.max_bytes, "device": cfg["device"],
        "test_used_for_fit_or_selection": False, "pc_upload_required": True, "promotion_passed": False}
    return dict(config=model_cfg, options=options, index=index, model=model, heads=heads, original_head=original,
        manifest=manifest, output=output, owner=owner, contract=contract, case_options=case_options, plan=plan)


def enrich_rows(rows, sequences, entry):
    if len(rows) != len(sequences):
        raise ValueError("batch/sequence alignment mismatch")
    for row, sequence in zip(rows, sequences):
        frame, det = sequence.reference, sequence.reference.detector_prediction
        row.update(shard=entry.filename, split="test", sequence_id=sequence.sequence_id,
            episode_id=frame.episode_id, frame_id=frame.frame_id, timestamp_s=frame.timestamp_s,
            assignment_id=frame.assignment_id, tracker_id=det.tracker_id, candidate_id=det.candidate_id,
            detected=det.detected, bbox_xyxy_normalized=det.bbox_xyxy_normalized,
            rgb_path=frame.sensor_input.rgb_path, depth_path=frame.sensor_input.depth_path,
            probe_eligible=bool(row["reference_input_valid"] and row["model_geometry_valid"]),
            probe_fit_mask=False, probe_label=bool(row["offline_measurement_supervision_positive"]))


def export_joint_cases(original, candidate, sequences, *, dataset_root, output_dir, entry, options):
    chosen, seqs, tags, originals = [], [], {}, {}
    for old, new, seq in zip(original, candidate, sequences):
        labels = [f"{name}_{tag}" for name, row in (("original",old),("candidate",new)) for tag in evidence.case_tags(row, options)]
        if old["model_valid"] != new["model_valid"]:
            labels.append("acceptance_changed")
        if labels:
            chosen.append(new)
            seqs.append(seq)
            tags[new["sequence_id"]] = labels
            originals[new["sequence_id"]] = old
    # Reuse the byte-preserving bounded copier. Already selected rows are forced
    # through its exporter; replace internal selection tags with the real union.
    forced = evidence.CaseExportOptions(options.max_bytes, options.outlier_error_m,
        tuple(sorted({r["episode_id"] for r in chosen})))
    result = evidence.export_cases(chosen, seqs, dataset_root=dataset_root, output_dir=output_dir, entry=entry, options=forced)
    for case in result["cases"]:
        sid = case["sample"]["sequence_id"]
        case["tags"] = tags[sid]
        case["original_sample"] = originals[sid]
    return result


@torch.inference_mode()
def evaluate_archive(archive, *, ctx, entry, device):
    if entry.split != "test":
        raise ValueError("joint audit only evaluates TEST archives")
    model, config = ctx["model"], ctx["config"]
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("backbone must be frozen in eval mode")
    materialized = audit.materialize_shard(archive, index=ctx["index"])
    loader, captured = None, []
    hook = model.validity_head.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().cpu().clone()))
    try:
        dataset, loader = audit._dataset_loader(config=config, dataset_root=materialized.dataset_root, split="test",
            device=device, shuffle=False, generator_seed=None, split_seed=ctx["index"].split_seed)
        originals, candidates, features = [], [], []
        accumulators = {k:audit.TargetStateEvaluationAccumulator(config.maximum_depth_m) for k in ("original","candidate")}
        for batch in loader:
            captured.clear()
            batch = audit._to_device(batch, device)
            output = model(batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
            if len(captured) != 1:
                raise ValueError("expected exactly one validity feature capture")
            feature = captured[0]
            replay = F.linear(feature, ctx["original_head"]["weight"], ctx["original_head"]["bias"]).flatten()
            if not torch.allclose(replay, output.measurement_valid_logit.cpu(), atol=1e-5, rtol=1e-5):
                raise ValueError("backbone feature/logit replay mismatch")
            rows = audit.diagnostic_rows(batch, output, maximum_depth_m=config.maximum_depth_m)
            enrich_rows(rows, dataset.sequences[len(originals):len(originals)+len(rows)], entry)
            joint, logits = ctx["heads"].apply(feature, rows)
            combined = replace(output, measurement_valid_logit=logits.to(device))
            combined_rows = audit.diagnostic_rows(batch, combined, maximum_depth_m=config.maximum_depth_m)
            if [r["model_valid"] for r in combined_rows] != [r["model_valid"] for r in joint]:
                raise ValueError("official joint metric gate mismatch")
            accumulators["original"].add_batch(batch=batch, output=output)
            accumulators["candidate"].add_batch(batch=batch, output=combined)
            originals.extend(rows)
            candidates.extend(joint)
            features.append(feature)
        if len(originals) != entry.sequence_count or len(originals) != len(dataset):
            raise ValueError("joint audit sequence count mismatch")
        cases = export_joint_cases(originals, candidates, dataset.sequences, dataset_root=materialized.dataset_root,
            output_dir=ctx["output"], entry=entry, options=ctx["case_options"])
        return materialized, {"evaluation_only": True, "entry": entry.to_dict(), "contract_sha256": producer.digest(ctx["contract"]),
            "features": torch.cat(features), "original_rows": originals, "candidate_rows": candidates,
            "accumulators": {k:v.state_dict() for k,v in accumulators.items()}, "case_exports": cases}
    finally:
        hook.remove()
        audit._release_loader(loader)


def receipt_paths(ctx, entry):
    p = ctx["output"]/"receipts"/(entry.filename+".pt")
    return p, p.with_suffix(".sha256.json")


def read_receipt(ctx, entry):
    path, checksum = receipt_paths(ctx, entry)
    if path.is_symlink() or checksum.is_symlink():
        raise ValueError("joint receipt must not be a symlink")
    expected = json.loads(checksum.read_text())
    digest = producer.digest(ctx["contract"])
    if expected != {"sha256": producer.sha256_file(path), "contract_sha256": digest}:
        raise ValueError("joint receipt SHA256 mismatch")
    r = torch.load(path, map_location="cpu", weights_only=True)
    if r.get("evaluation_only") is not True or r.get("entry") != entry.to_dict() or r.get("contract_sha256") != digest:
        raise ValueError("joint receipt identity mismatch")
    x, old, new = r["features"], r["original_rows"], r["candidate_rows"]
    if (entry.split != "test" or x.ndim != 2 or x.shape != (entry.sequence_count, ctx["original_head"]["weight"].shape[1])
            or x.device.type != "cpu" or x.requires_grad or not torch.isfinite(x).all()
            or len(old) != entry.sequence_count or len(new) != len(old)):
        raise ValueError("invalid TEST receipt dimensions")
    for rows in (old,new):
        if (len({v["sequence_id"] for v in rows}) != len(rows) or any(v["split"] != "test"
                or v["episode_id"] not in entry.episode_ids or v["shard"] != entry.filename or v["probe_fit_mask"] is not False
                or v["probe_eligible"] != bool(v["reference_input_valid"] and v["model_geometry_valid"]) for v in rows)):
            raise ValueError("TEST receipt split/identity/eligibility mismatch")
    probabilities = quality.validity_probabilities(x, ctx["original_head"])
    if not torch.allclose(probabilities, torch.tensor([v["validity_probability"] for v in old]), atol=1e-5, rtol=1e-5):
        raise ValueError("receipt original probability mismatch")
    if [v["model_valid"] for v in producer.regate(old, probabilities)] != [v["model_valid"] for v in old]:
        raise ValueError("receipt original gate mismatch")
    replay, _ = ctx["heads"].apply(x, old)
    # All non-probability metadata, geometry and decisions must match exactly.
    for a, b in zip(replay, new):
        for key in ("quality_probability", "validity_probability"):
            if not math.isclose(a[key], b[key], abs_tol=1e-6, rel_tol=1e-5):
                raise ValueError("receipt candidate probability mismatch")
        if canonical({k:v for k,v in a.items() if k not in ("quality_probability","validity_probability")}) != canonical({k:v for k,v in b.items() if k not in ("quality_probability","validity_probability")}):
            raise ValueError("receipt paired geometry/decision mismatch")
    restored = {k:audit.TargetStateEvaluationAccumulator.from_state_dict(r["accumulators"][k], maximum_depth_m=ctx["config"].maximum_depth_m)
                for k in ("original","candidate")}
    for name, rows in (("original",old),("candidate",new)):
        state = restored[name].state_dict()
        visible = [v for v in rows if v["evaluated_visible_target"]]
        if (torch.cat(state["model_failures"]).tolist() != [not v["model_valid"] for v in visible]
                or torch.cat(state["model_no_target_claims"]).tolist() != [v["model_valid"] for v in rows if v["no_target"]]):
            raise ValueError("receipt accumulator/decision mismatch")
    evidence.verify_case_assets(r["case_exports"], ctx["output"])
    return r, restored


def summarize_pair(old, new):
    if len(old) != len(new) or any(a["sequence_id"] != b["sequence_id"] for a,b in zip(old,new)):
        raise ValueError("unaligned paired TEST rows")
    scores = {"original":quality.validation_diagnostics(old), "candidate":quality.validation_diagnostics(new)}
    changes = quality.changed_rows(old,new)
    both = [a for a,b in zip(old,new) if a["evaluated_visible_target"] and a["model_valid"] and b["model_valid"]]
    geometry_same = all(a["model_position_world_m"] == b["model_position_world_m"] for a,b in zip(old,new))
    if not geometry_same:
        raise ValueError("candidate changed frozen geometry")
    def ids(rows, kind):
        return {r["sequence_id"] for r in rows if r["model_valid"] and
            ((r["no_target"]) if kind == "false_positive" else (r["evaluated_visible_target"] and r["model_error_m"] > float(kind)))}
    safety = {}
    for kind in ("false_positive", "1", "5"):
        before, after = ids(old,kind), ids(new,kind)
        safety[kind] = {"original_count":len(before), "candidate_count":len(after),
            "new_ids":sorted(after-before), "removed_ids":sorted(before-after)}
    unchanged = producer.admissible(scores["candidate"],scores["original"])
    return {"scores":scores, "shared_accepted_error":audit.error_summary(both,"model_error_m"),
        "geometry_predictions_identical":True, "acceptance_changes":len(changes),
        "newly_accepted_count":sum(c["candidate"]["model_valid"] for c in changes),
        "newly_rejected_count":sum(not c["candidate"]["model_valid"] for c in changes),
        "safety_case_changes":safety, "fixed_test_safety_counts_nonregressing":unchanged,
        "fixed_test_primary_improved": unchanged and scores["candidate"]["positive_supervision_rejected_count"] < scores["original"]["positive_supervision_rejected_count"]}


def audit_test(ctx, lifecycle, device):
    entries = ctx["index"].shards_for_split("test")
    run_id = ctx["options"].run_id_prefix+".test"
    lifecycle.request(run_id, [e.filename for e in entries])
    total = {k:audit.TargetStateEvaluationAccumulator(ctx["config"].maximum_depth_m) for k in ("original","candidate")}
    originals, candidates, cases, assets, hashes = [], [], [], {}, {}
    for i, entry in enumerate(entries, 1):
        path, checksum = receipt_paths(ctx, entry)
        state = lifecycle.shard_state(run_id, entry.filename)
        consumed = audit._state_name(state) == "consumed"
        materialized = None
        if path.exists() and checksum.exists():
            receipt, partial = read_receipt(ctx, entry)
            if not consumed or state["deleted"] is not True:
                archive = audit._prepare_active_shard(lifecycle=lifecycle, options=ctx["options"], run_id=run_id, entry=entry)
                materialized = audit.materialize_shard(archive,index=ctx["index"])
        else:
            if consumed:
                raise ValueError("consumed TEST shard lacks committed receipt/checksum; use a new audit prefix/output")
            print(f"[test {i}/{len(entries)}] waiting/evaluating {entry.filename}",flush=True)
            archive = audit._prepare_active_shard(lifecycle=lifecycle, options=ctx["options"], run_id=run_id, entry=entry)
            materialized, receipt = evaluate_archive(archive,ctx=ctx,entry=entry,device=device)
            audit.save_receipt(path,receipt)
            producer._atomic_write_json(checksum,{"sha256":producer.sha256_file(path),"contract_sha256":producer.digest(ctx["contract"])})
            receipt, partial = read_receipt(ctx,entry)
        # No server deletion before durable paired predictions AND promised
        # RGB-D evidence have all been re-read and verified.
        if materialized is not None:
            audit.cleanup_materialized_shard(materialized)
            lifecycle.consume(run_id,entry.filename,delete=True)
        for k in total:
            total[k].merge(partial[k])
        originals.extend(receipt["original_rows"])
        candidates.extend(receipt["candidate_rows"])
        cases.extend(receipt["case_exports"]["cases"])
        assets.update(receipt["case_exports"]["assets"])
        hashes[entry.filename] = producer.sha256_file(path)
        print(f"[test {i}/{len(entries)}] verified receipt; this audit's server cache consumed",flush=True)
    original_metrics, candidate_metrics = total["original"].finalize(), total["candidate"].finalize()
    metric_differences = evidence.compare_metrics(original_metrics,ctx["manifest"]["test_metrics"])
    result = {"complete":True,"contract":ctx["contract"],"candidate_frozen_before_test":True,
        "training_performed":False,"test_used_for_fit_or_selection":False,"fresh_independent_test":False,
        "promotion_passed":False,"receipt_sha256":hashes,"comparison":summarize_pair(originals,candidates),
        "original_evaluation_metrics":original_metrics,"candidate_evaluation_metrics":candidate_metrics,
        "original_training_metric_replay":{"matched":not metric_differences,"differences":metric_differences},
        "case_exports":{"count":len(cases),"rgbd_complete_count":sum(c["rgbd_export_complete"] for c in cases),
            "rgbd_budget_omitted_count":sum(not c["rgbd_export_complete"] for c in cases),
            "unique_asset_bytes":sum(a["size_bytes"] for a in assets.values())},
        "notes":["Historical TEST was previously inspected; this is not fresh independent generalization.",
            "Geometry and covariance are unchanged; compare identical accepted subsets and acceptance changes.",
            "Candidate metric adapter uses min(validity_logit,quality_logit); its mean_loss is diagnostic, not a training objective.",
            "Offline labels/IDs/poses in evidence are never quality inputs or runtime truth access.",
            "Metric replay mismatch must be investigated before interpreting candidate differences.",
            "No epoch/threshold selection or automatic deployment on TEST. Runtime Kalman/closed-loop checks remain outstanding."]}
    output = ctx["output"]
    files = {
        "test_original_samples.json":{"samples":originals},
        "test_candidate_samples.json":{"samples":candidates},
        "test_changes.json":{"offline_only":True,"changes":quality.changed_rows(originals,candidates)},
        "test_cases.json":{"offline_only":True,"cases":cases,"assets":assets},
    }
    result["output_sha256"] = {}
    for filename, payload in files.items():
        producer._atomic_write_json(output/filename,payload)
        result["output_sha256"][filename] = producer.sha256_file(output/filename)
    producer._atomic_write_json(output/"report.json",result)
    return result


def run(cfg, *, dry_run=False, lifecycle=None):
    torch.set_num_threads(4)
    ctx = preflight(cfg)
    print(json.dumps(ctx["plan"],indent=2),flush=True)
    if dry_run:
        return ctx["plan"]
    # Fail unavailable GPU before requesting any PC data; do not silently fall back.
    device = audit._device(cfg["device"])
    ctx["model"].to(device).eval().requires_grad_(False)
    client = lifecycle or audit.PCTransCLI(ctx["options"].pc_trans_root,ctx["options"].pc_trans_config,ctx["options"].pc_trans_python)
    with producer._exclusive_run_lock(ctx["owner"]), producer._exclusive_run_lock(ctx["output"]):
        for directory in (ctx["owner"],ctx["output"]):
            quality.check_output(directory,ctx["contract"])
            producer._atomic_write_json(directory/"contract.json",ctx["contract"])
        result = audit_test(ctx,client,device)
    print(f"Joint quality TEST audit complete: {ctx['output']/'report.json'}",flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--dry-run",action="store_true",help="verify the pinned candidate/cache without PC requests, TEST evaluation or writes")
    args = parser.parse_args(argv)
    run(load_config(args.config),dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
