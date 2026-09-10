#!/usr/bin/env python3
"""Request one PC shard and export unchanged RGB-D plus offline labels.

This is an offline review only: it does not alter labels, infer target identity
for production, change a model, or delete the PC/server archive.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.target_state.dataset import read_frame_records
from training.target_state.shard_runtime import materialize_shard, cleanup_materialized_shard
from training.target_state.sharded_trainer import (
    PCTransCLI, ShardedTrainingOptions, _atomic_write_json, _exclusive_run_lock,
    _prepare_active_shard, _state_name,
)
from training.target_state.shards import load_shard_index
from training.target_state.trainer import sha256_file


def copy_verified_asset(source, destination):
    expected = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or sha256_file(destination) != expected:
            raise ValueError(f"existing review asset differs: {destination}")
        return expected
    descriptor, name = tempfile.mkstemp(prefix=".asset-", dir=destination.parent)
    temporary = Path(name)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if sha256_file(temporary) != expected:
            raise ValueError("asset copy checksum mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def export_episode(archive, *, index, episode_id, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = materialize_shard(archive, index=index,
                                    materialized_root=output_dir / ".materialized")
    records = [r for r in read_frame_records(materialized.dataset_root / "frames.jsonl")
               if r.episode_id == episode_id]
    if not records:
        raise ValueError(f"episode not in verified archive: {episode_id}")
    assets = sorted({path for r in records for path in
                    (r.sensor_input.rgb_path, r.sensor_input.depth_path,
                     r.sensor_input.instance_mask_path) if path is not None})
    digests = {path: copy_verified_asset(materialized.dataset_root / path, output_dir / "assets" / path)
               for path in assets}
    report = {"offline_only": True, "episode_id": episode_id,
              "archive_path": str(archive), "archive_sha256": materialized.entry.archive_sha256,
              "index_sha256": index.index_sha256, "asset_sha256": digests,
              "records": [r.to_dict() for r in records]}
    _atomic_write_json(output_dir / "episode_review.json", report)
    # Only remove the exact verified extraction; both the server archive and
    # copied original assets remain available for repeated visual inspection.
    cleanup_materialized_shard(materialized)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--shard-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pc-trans-root", type=Path, default=ROOT.parent.parent / "pc_trans")
    parser.add_argument("--run-id")
    parser.add_argument("--wait-timeout", type=float, default=86400)
    parser.add_argument("--request-only", action="store_true")
    args = parser.parse_args(argv)
    index = load_shard_index(args.shard_index)
    matches = [entry for entry in index.shards if args.episode_id in entry.episode_ids]
    if len(matches) != 1:
        raise ValueError("episode must resolve to exactly one indexed shard")
    entry = matches[0]
    pc_config = args.pc_trans_root / "config" / "config.json"
    bridge = Path(json.loads(pc_config.read_text())["bridge_root"]).resolve()
    run_id = args.run_id or f"audit_review_{args.episode_id}_v1"
    if not run_id.startswith("audit_review_"):
        raise ValueError("inspection requires a dedicated audit_review_* run ID")
    options = ShardedTrainingOptions(shard_index_path=args.shard_index,
        pc_trans_root=args.pc_trans_root, pc_trans_config=pc_config, bridge_root=bridge,
        run_id_prefix=run_id, wait_timeout_s=args.wait_timeout)
    destination = args.output_dir.resolve()
    if destination == bridge or bridge in destination.parents:
        raise ValueError("review output must be outside the transfer bridge")
    client = PCTransCLI(args.pc_trans_root, pc_config)
    contract = {"run_id": run_id, "episode_id": args.episode_id,
                "index_sha256": index.index_sha256, "archive_sha256": entry.archive_sha256,
                "output_dir": str(destination)}
    owner = bridge / "control" / "audits" / run_id
    with _exclusive_run_lock(owner), _exclusive_run_lock(destination):
        for path in (owner / "contract.json", destination / "contract.json"):
            if path.exists() and json.loads(path.read_text()) != contract:
                raise ValueError("inspection run ID/output belongs to a different contract")
        for path in (owner / "contract.json", destination / "contract.json"):
            _atomic_write_json(path, contract)
        client.request(run_id, [entry.filename])
        print(json.dumps({"run_id": run_id, "requested_shard": entry.filename,
                          "archive_bytes": entry.archive_size_bytes, "output_dir": str(destination)}, indent=2), flush=True)
        if args.request_only:
            return 0
        state = client.shard_state(run_id, entry.filename)
        if _state_name(state) == "consumed":
            if state["deleted"]:
                raise ValueError("review archive was deleted; use a new review run ID/output")
            archive = bridge / "train_cache" / "active" / run_id / entry.filename
        else:
            archive = _prepare_active_shard(lifecycle=client, options=options, run_id=run_id, entry=entry)
        report = export_episode(archive, index=index, episode_id=args.episode_id, output_dir=destination)
        if _state_name(state) != "consumed":
            client.consume(run_id, entry.filename, delete=False)
    print(json.dumps({"complete": True, "records": len(report["records"]),
                      "report": str(destination / "episode_review.json"),
                      "server_archive_retained": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
