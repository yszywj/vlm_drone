#!/usr/bin/env python3
"""CPU-only V3 association/window diagnosis; never rewrite collection evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from datasets.target_state.schema import TargetStateFrameRecord
from datasets.target_state.sequence import build_sequences
from target_state_v3.association import AssociationPolicy, associate
from target_state_v3.data import TargetStateV3Dataset
from target_state_v3.storage import contained, verify_episode
from target_state_v3.verify_tar import file_sha, verify_tar
from training.target_state.config import load_training_config
from training.target_state.sharded_trainer import _atomic_write_json


def detection_cases(payload, mask, policy):
    """Replay original decisions, then attribute unknown pixels to exact prims."""
    rows = [r for r in payload["records"] if r["record"]["detector_prediction"]["detected"]]
    detections = [SimpleNamespace(class_id=0, class_name="cube",
        bbox_xyxy_normalized=r["record"]["detector_prediction"]["bbox_xyxy_normalized"]) for r in rows]
    mapping = payload["oracle_only"]["mapping"]
    decisions = associate(detections, mask, mapping, policy)
    cases = []
    h, w = mask.shape
    for row, det, decision in zip(rows, detections, decisions):
        if decision != row["association"]:
            raise ValueError(f"association replay mismatch: {row['record']['frame_id']}")
        box = det.bbox_xyxy_normalized
        x1, y1 = math.floor(box[0]*w), math.floor(box[1]*h)
        x2, y2 = math.ceil(box[2]*w), math.ceil(box[3]*h)
        dx, dy = x2-x1, y2-y1
        roi = mask[y1+int(.1*dy):y2-int(.1*dy), x1+int(.1*dx):x2-int(.1*dx)]
        unknown = Counter()
        for k, n in zip(*np.unique(roi, return_counts=True)):
            info = mapping["instances"][str(int(k))]
            if info["kind"] == "unknown":
                unknown[info["prim_path"] or "<no_prim>"] += int(n)
        d = row["record"]["detector_prediction"]
        ground = unknown["/World/Ground/geom"]
        cases.append({"frame_id": row["record"]["frame_id"], "capture_id": payload["capture_id"],
            "episode_id": payload["episode_id"], "timestamp_s": payload["timestamp_s"],
            "candidate_id": d["candidate_id"], "tracker_id": d["tracker_id"],
            "bbox_xyxy_normalized": list(box), "association": decision,
            "unknown_roi_pixels_by_prim": dict(unknown),
            "ground_fraction": ground/roi.size if roi.size else 0.,
            "unknown_limit_exceeded": bool(roi.size and sum(unknown.values())/roi.size > policy.maximum_unknown_fraction),
            "ground_alone_exceeds_unknown_limit": bool(roi.size and ground/roi.size > policy.maximum_unknown_fraction),
            "stored_sensor_measurement": payload.get("sensor_measurement_diagnostics", {}).get(row["record"]["frame_id"]),
        })
    return cases


def window_audit(records, capture_indices, *, history_size, max_history_age_s):
    """Explain the existing builder's first rejection, without dropping rows."""
    actual = build_sequences(records, history_size=history_size, max_history_age_s=max_history_age_s)
    groups = defaultdict(list)
    counts = Counter(skipped_label_without_candidate=0, insufficient_history_references=0,
        proposed_windows=0, unresolved_in_window=0, history_too_old=0,
        mixed_target_instances=0, eligible_windows=0, eligible_positive_reference_windows=0)
    for r in records:
        candidate = r.detector_prediction.candidate_id
        if candidate is None and r.training_label is not None:
            counts["skipped_label_without_candidate"] += 1
            continue
        key = (r.uav_id, r.assignment_id, r.episode_id, candidate or "negative_background")
        groups[key].append(r)
    candidates, windows, accepted = [], [], set()
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda r: (r.timestamp_s, r.frame_id))
        counts["insufficient_history_references"] += min(history_size, len(ordered))
        run = longest = 0
        previous_index = previous_instance = None
        for r in ordered:
            label = r.training_label
            matched = bool(label is not None and r.detector_prediction.detected and not r.association_review_required)
            index = capture_indices[r.frame_id]
            if matched:
                same = index == previous_index+1 if previous_index is not None else False
                run = run+1 if same and previous_instance == label.instance_id else 1
                previous_instance = label.instance_id
            else:
                run, previous_instance = 0, None
            previous_index = index
            longest = max(longest, run)
        candidates.append({"episode_id": key[2], "candidate_id": key[3], "records": len(ordered),
            "matched_records": sum(r.training_label is not None and not r.association_review_required for r in ordered),
            "unresolved_records": sum(r.association_review_required for r in ordered),
            "tracker_ids": sorted({r.detector_prediction.tracker_id for r in ordered if r.detector_prediction.tracker_id}),
            "longest_same_target_matched_run_consecutive_captures": longest})
        for i in range(history_size, len(ordered)):
            frames = ordered[i-history_size:i+1]
            ref = frames[-1]
            counts["proposed_windows"] += 1
            ids = {r.training_label.instance_id for r in frames if r.training_label is not None
                   and r.training_label.instance_id is not None}
            reason = "eligible_windows"
            if any(r.association_review_required for r in frames):
                reason = "unresolved_in_window"
            elif ref.timestamp_s-frames[0].timestamp_s > max_history_age_s+1e-9:
                reason = "history_too_old"
            elif len(ids) > 1:
                reason = "mixed_target_instances"
            counts[reason] += 1
            if reason == "eligible_windows":
                accepted.add(ref.frame_id)
                counts["eligible_positive_reference_windows"] += int(ref.training_label is not None)
            windows.append({"episode_id": key[2], "candidate_id": key[3], "reference_frame_id": ref.frame_id,
                "frame_ids": [r.frame_id for r in frames], "first_gate_result": reason,
                "reference_has_target_label": ref.training_label is not None})
    if accepted != {s.reference.frame_id for s in actual} or len(accepted) != len(actual):
        raise ValueError("diagnostic window ledger differs from actual sequence builder")
    return dict(counts), candidates, windows, accepted


def diagnose(session):
    session = session.resolve()
    state_path = session/"session.json"
    source_sha = file_sha(state_path)
    state = json.loads(state_path.read_text())
    if state.get("complete") is not True or state.get("last_error"):
        raise ValueError("diagnosis requires a complete, error-free pilot session")
    contract = state["contract"]
    for name, expected in contract["source_sha256"].items():
        if file_sha(contained(ROOT, name)) != expected:
            raise ValueError(f"producer source changed; do not silently replay new rules: {name}")
    policy = AssociationPolicy()
    if policy.to_dict() != contract["association_policy"]:
        raise ValueError("association policy changed")
    config_path = ROOT/"configs/target_state/train_geometry_v2_stageb_50k.yaml"
    cfg = replace(load_training_config(config_path), device="cpu", num_workers=0,
        expected_yolo_model_sha256=contract["detector_deployment"]["model_sha256"])
    totals, window_counts, unknown_prims = Counter(), Counter(), Counter()
    dataset_counts = Counter(windows=0, positive_reference_windows=0, negative_reference_windows=0,
                            positive_measurement_windows=0)
    cases, candidates, windows, episodes = [], [], [], []
    for entry in state["episodes"]:
        folder = contained(session, entry["episode_id"])
        stats = verify_episode(folder)
        if stats != entry["stats"]:
            raise ValueError("episode statistics differ from journal")
        archive = contained(session, "archives/"+entry["filename"])
        if file_sha(archive) != entry["sha256"]:
            raise ValueError("archive checksum mismatch")
        manifest = verify_tar(archive, expected_contract=contract, expected_episode=entry["episode_id"],
                              expected_manifest_sha=file_sha(folder/"episode_manifest.json"))
        records, capture_indices, episode_cases = [], {}, []
        for index, capture in enumerate(manifest["captures"]):
            payload = json.loads(contained(folder, f"captures/{capture}.json").read_text())
            for row in payload["records"]:
                r = TargetStateFrameRecord.from_dict(row["record"])
                records.append(r)
                capture_indices[r.frame_id] = index
            with np.load(contained(folder, payload["oracle_only"]["mask_path"]), allow_pickle=False) as npz:
                episode_cases.extend(detection_cases(payload, npz["instance_id"], policy))
        for case in episode_cases:
            unknown_prims.update(case["unknown_roi_pixels_by_prim"])
        wc, cs, ws, eligible = window_audit(records, capture_indices,
            history_size=cfg.history_size, max_history_age_s=cfg.max_history_age_s)
        loaded = set()
        for split in ("train", "validation", "test"):
            dataset = TargetStateV3Dataset(cfg, episode_root=folder, split=split)
            for i, seq in enumerate(dataset.sequences):
                batch = dataset[i]
                loaded.add(seq.reference.frame_id)
                dataset_counts["windows"] += 1
                present = bool(batch["target_present_mask"])
                dataset_counts["positive_reference_windows"] += int(present)
                dataset_counts["negative_reference_windows"] += int(not present)
                dataset_counts["positive_measurement_windows"] += int(batch["measurement_valid"])
        if loaded != eligible:
            raise ValueError("diagnosed windows differ from actual V3 loader")
        totals.update(stats)
        window_counts.update(wc)
        cases.extend(episode_cases)
        candidates.extend(cs)
        windows.extend(ws)
        episodes.append({"episode_id": entry["episode_id"], "counts": stats, "window_gates": wc})
        print(f"Diagnosed {len(episodes)}/{len(state['episodes'])} episodes", flush=True)
    if totals["physical_captures"] != contract["physical_captures"] or dict(totals) != state["summary"]:
        raise ValueError("diagnosis counts differ from completed session")
    if file_sha(state_path) != source_sha:
        raise ValueError("session changed during diagnosis")
    return {"complete": True, "diagnosis_only": True, "source_session": str(session),
        "source_session_sha256": source_sha, "diagnostic_source_sha256": file_sha(Path(__file__)),
        "window_config_sha256": file_sha(config_path), "history_size": cfg.history_size,
        "max_history_age_s": cfg.max_history_age_s, "labels_modified": False, "training_started": False,
        "ready_for_training": False, "producer_sources_match": True,
        "summary": {"collection_counts": dict(totals), "detected_records": len(cases),
            "window_gates": dict(window_counts), "actual_dataset": dict(dataset_counts),
            "detections_exceeding_unknown_limit": sum(c["unknown_limit_exceeded"] for c in cases),
            "detections_ground_alone_exceeds_unknown_limit": sum(c["ground_alone_exceeds_unknown_limit"] for c in cases),
            "unknown_roi_pixels_by_prim": dict(unknown_prims.most_common()),
            "longest_same_target_matched_run_consecutive_captures": max((c["longest_same_target_matched_run_consecutive_captures"] for c in candidates), default=0)},
        "episodes": episodes, "detections": cases, "candidates": candidates, "windows": windows,
        "notes": ["Window rejections are counted at the FIRST failing gate, matching the actual builder.",
                  "Unknown pixel counts are ROI totals, not distinct physical pixels or accuracy metrics.",
                  "A ground contribution does not prove it is the only failed association criterion.",
                  "No threshold changes, relabeling, training or collection are performed."]}


def validate_output(session, output):
    session, output = session.resolve(), output.resolve()
    if session == output or session in output.parents or output in session.parents:
        raise ValueError("diagnostic output must be outside the source session")
    if output.exists():
        raise ValueError("diagnostic output already exists; choose a new --output, never overwrite evidence")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = validate_output(args.session, args.output)
    torch.set_num_threads(4)
    result = diagnose(args.session)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(output/"diagnosis.json", result)
    _atomic_write_json(output/"summary.json", {k: v for k, v in result.items()
                                              if k not in {"episodes", "detections", "candidates", "windows"}})
    print(json.dumps({"complete": True, "output": str(output), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
