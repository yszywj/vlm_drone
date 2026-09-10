#!/usr/bin/env python3
"""Scan ALL PC-hosted splits for association review; never rewrite source labels.

Only this run's verified server cache is consumed after a durable receipt.
Review findings are NOT a repaired dataset and NOT production gating inputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.target_state.dataset import read_frame_records
from training.target_state.association_audit import scan_records
from training.target_state.association_quality import ASSOCIATION_PROTOCOL, AssociationPolicy
from training.target_state.shard_runtime import materialize_shard, cleanup_materialized_shard
from training.target_state.sharded_trainer import (
    PCTransCLI, ShardedTrainingOptions, _atomic_write_json, _exclusive_run_lock,
    _prepare_active_shard, _state_name,
)
from training.target_state.shards import load_shard_index
from training.target_state.trainer import sha256_file

SPLITS = ("train", "validation", "test")


def scan_archive(archive, *, index, policy):
    materialized = materialize_shard(archive, index=index)
    records = read_frame_records(materialized.dataset_root / "frames.jsonl")
    result = scan_records(records, dataset_root=materialized.dataset_root,
        history_size=index.history_size, max_history_age_s=index.max_history_age_s, policy=policy)
    return materialized, result


def load_receipt(path, *, contract_sha, entry):
    receipt = json.loads(path.read_text())
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    digest = sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if (receipt.get("contract_sha256") != contract_sha or receipt.get("entry") != entry.to_dict()
            or receipt.get("receipt_sha256") != digest):
        raise ValueError(f"association receipt identity/checksum mismatch: {path}")
    counts = receipt["result"]["counts"]
    if counts["frame_records"] != entry.frame_count or counts["sequences"] != entry.sequence_count:
        raise ValueError(f"association receipt count mismatch: {path}")
    return receipt


def scan_split(*, split, options, index, lifecycle, output_dir, contract_sha, policy):
    run_id = f"{options.run_id_prefix}.{split}"
    entries = index.shards_for_split(split)
    lifecycle.request(run_id, [entry.filename for entry in entries])
    counts, reasons, findings, affected = Counter(), Counter(), [], []
    for ordinal, entry in enumerate(entries, 1):
        path = output_dir / "receipts" / (entry.filename + ".json")
        materialized = None
        state = lifecycle.shard_state(run_id, entry.filename)
        if path.exists():
            receipt = load_receipt(path, contract_sha=contract_sha, entry=entry)
            if _state_name(state) != "consumed" or state["deleted"] is not True:
                archive = _prepare_active_shard(lifecycle=lifecycle, options=options,
                                                run_id=run_id, entry=entry)
                materialized = materialize_shard(archive, index=index)
        else:
            if _state_name(state) == "consumed":
                raise ValueError("consumed shard has no receipt; use a new audit prefix/output")
            print(f"[{split} {ordinal}/{len(entries)}] waiting/scanning {entry.filename}", flush=True)
            archive = _prepare_active_shard(lifecycle=lifecycle, options=options,
                                            run_id=run_id, entry=entry)
            materialized, result = scan_archive(archive, index=index, policy=policy)
            receipt = dict(contract_sha256=contract_sha, entry=entry.to_dict(), result=result)
            receipt["receipt_sha256"] = sha256(json.dumps(receipt, sort_keys=True, allow_nan=False).encode()).hexdigest()
            _atomic_write_json(path, receipt)
            receipt = load_receipt(path, contract_sha=contract_sha, entry=entry)
        # Commit and validate evidence BEFORE deleting any of this run's cache.
        if materialized is not None:
            cleanup_materialized_shard(materialized)
            lifecycle.consume(run_id, entry.filename, delete=True)
        result = receipt["result"]
        counts.update(result["counts"])
        reasons.update(result["reason_counts"])
        findings.extend(dict(shard=entry.filename, **row) for row in result["findings"])
        affected.extend(dict(shard=entry.filename, **row) for row in result["affected_sequences"])
        print(f"[{split} {ordinal}/{len(entries)}] receipt saved; review records="
              f"{counts['review_required_records']}; server cache consumed", flush=True)
    report = dict(counts=dict(counts), reason_counts=dict(reasons))
    _atomic_write_json(output_dir / f"{split}_review_manifest.json", dict(
        offline_only=True, action="review_only_no_automatic_relabel_or_deletion",
        contract_sha256=contract_sha, **report, findings=findings, affected_sequences=affected))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="audit_association_50k_v1")
    parser.add_argument("--pc-trans-root", type=Path, default=ROOT.parent.parent / "pc_trans")
    parser.add_argument("--wait-timeout", type=float, default=86400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    index = load_shard_index(args.shard_index)
    pc_root = args.pc_trans_root.resolve()
    config = pc_root / "config" / "config.json"
    bridge = Path(json.loads(config.read_text())["bridge_root"]).resolve()
    output = args.output_dir.resolve()
    if output == bridge or bridge in output.parents or output in bridge.parents:
        raise ValueError("review output must be outside the transfer bridge")
    if not args.run_id_prefix.startswith("audit_association_"):
        raise ValueError("use a dedicated audit_association_* prefix")
    options = ShardedTrainingOptions(shard_index_path=args.shard_index, pc_trans_root=pc_root,
        pc_trans_config=config, bridge_root=bridge, run_id_prefix=args.run_id_prefix,
        wait_timeout_s=args.wait_timeout)
    policy = AssociationPolicy()
    sources = [Path(__file__), *sorted((ROOT / "training" / "target_state").glob("*.py")),
               *sorted((ROOT / "datasets" / "target_state").glob("*.py"))]
    contract = dict(protocol=ASSOCIATION_PROTOCOL, policy=policy.to_dict(),
        index_sha256=index.index_sha256, parent_dataset_sha256=index.parent_dataset_sha256,
        run_id_prefix=options.run_id_prefix, output_dir=str(output), bridge_root=str(bridge),
        source_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in sources})
    contract_sha = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    print(json.dumps(dict(run_ids=[f"{options.run_id_prefix}.{s}" for s in SPLITS],
        shards=len(index.shards), archive_bytes=sum(e.archive_size_bytes for e in index.shards),
        frame_records=index.frame_count, sequences=index.sequence_count,
        output_dir=str(output), policy=policy.to_dict()), indent=2), flush=True)
    if args.dry_run:
        return 0
    client = PCTransCLI(pc_root, config)
    owner = bridge / "control" / "audits" / options.run_id_prefix
    with _exclusive_run_lock(owner), _exclusive_run_lock(output):
        for path in (owner / "contract.json", output / "contract.json"):
            if path.exists() and json.loads(path.read_text()) != contract:
                raise ValueError("audit contract changed; choose a new prefix AND output directory")
        for path in (owner / "contract.json", output / "contract.json"):
            _atomic_write_json(path, contract)
        results = {split: scan_split(split=split, options=options, index=index,
            lifecycle=client, output_dir=output, contract_sha=contract_sha, policy=policy) for split in SPLITS}
        _atomic_write_json(output / "report.json", dict(complete=True, offline_only=True,
            contract=contract, results=results, original_dataset_unchanged=True,
            limitations=["Review flags are not proven mislabels; partial occlusion can also trigger them.",
                "Legacy data has no full object extent/instance-ID mask mapping; depth support is not identity proof.",
                "No repaired dataset, filtering, new model, or deployment change is produced."]))
    print(f"Association review complete: {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
