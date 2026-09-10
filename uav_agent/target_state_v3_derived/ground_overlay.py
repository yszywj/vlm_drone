"""Versioned, replay-verifiable ground-role annotation overlays.

No original capture, association, manifest or model is rewritten. A sidecar is
NOT a self-contained dataset and is NOT discovered by the legacy trainers.
"""
from collections import Counter
from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from datasets.target_state.dataset import split_for_episode
from datasets.target_state.schema import TargetStateFrameRecord
from datasets.target_state.sequence import build_sequences
from target_state_v3 import DATA_PROTOCOL
from target_state_v3.association import AssociationPolicy, associate
from target_state_v3.data import TargetStateV3Dataset, sensor_observations
from target_state_v3.measurement import measure_window
from target_state_v3.storage import contained, verify_episode
from target_state_v3.verify_tar import file_sha, verify_tar
from training.target_state.config import load_training_config
from training.target_state.sharded_trainer import _atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/target_state/train_geometry_v2_stageb_50k.yaml"
PROTOCOL = "v3_exact_ground_role_overlay_v1"
GROUND = "/World/Ground/geom"


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def code_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [ROOT/"scripts/reassociate_target_state_v3_pilot.py",
              ROOT/"scripts/check_target_state_v3_reassociation.py"]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def rules():
    return {"ground_prim_path": GROUND, "ground_role": "known_non_target_environment_surface",
        "association_kind": "known_non_cube", "require_positive_finite_raw_ground_depth": True,
        "association_policy": AssociationPolicy().to_dict(), "sensor_candidate_ids_unchanged": True,
        "raw_oracle_rows_retained": True, "offline_only": True}


def load_source(session):
    session = session.resolve()
    sha = file_sha(session/"session.json")
    state = json.loads((session/"session.json").read_text())
    if state.get("complete") is not True or state.get("last_error"):
        raise ValueError("source pilot must be complete and error-free")
    if state.get("protocol") != DATA_PROTOCOL or state["contract"].get("protocol") != DATA_PROTOCOL:
        raise ValueError("source is not a V3 pilot")
    if not 1 <= state["contract"]["physical_captures"] <= 200:
        raise ValueError("ground role review is limited to the at-most-200 capture pilot")
    if state["contract"]["association_policy"] != AssociationPolicy().to_dict():
        raise ValueError("source association policy differs; no implicit threshold changes allowed")
    # This producer's pinned scene defines /World/Ground via add_ground_plane.
    if "env/scene.py" not in state["contract"]["source_sha256"]:
        raise ValueError("missing provenance for the known scene ground role")
    for name, expected in state["contract"]["source_sha256"].items():
        if file_sha(contained(ROOT, name)) != expected:
            raise ValueError(f"source producer changed: {name}")
    if len({e["episode_id"] for e in state["episodes"]}) != len(state["episodes"]):
        raise ValueError("duplicate source episode")
    return session, state, sha


def checked_source_episode(session, state, entry):
    folder = contained(session, entry["episode_id"])
    stats = verify_episode(folder)
    if stats != entry["stats"]:
        raise ValueError("source episode statistics differ from journal")
    archive = contained(session, "archives/"+entry["filename"])
    if file_sha(archive) != entry["sha256"] or archive.stat().st_size != entry["size_bytes"]:
        raise ValueError("source archive checksum/size mismatch")
    manifest = verify_tar(archive, expected_contract=state["contract"],
        expected_episode=entry["episode_id"], expected_manifest_sha=file_sha(folder/"episode_manifest.json"))
    return folder, manifest


def derive_capture(payload, mask, raw_depth):
    """Use only one frame's recorded labels; ground never supplies target identity."""
    if raw_depth.shape != mask.shape or not np.issubdtype(raw_depth.dtype, np.floating):
        raise ValueError("raw depth is required at instance-mask resolution")
    records = [TargetStateFrameRecord.from_dict(row["record"]) for row in payload["records"]]
    labels, label_sources = {}, {}
    for record in records:
        label = record.training_label
        if label is None:
            continue
        if label.instance_id in labels and labels[label.instance_id] != label:
            raise ValueError("conflicting same-frame target labels")
        labels[label.instance_id] = label
        label_sources.setdefault(label.instance_id, record.frame_id)
    positions = [i for i, r in enumerate(records) if r.detector_prediction.detected]
    detections = [SimpleNamespace(class_id=0, class_name="cube",
        bbox_xyxy_normalized=records[i].detector_prediction.bbox_xyxy_normalized) for i in positions]
    mapping = payload["oracle_only"]["mapping"]
    before = associate(detections, mask, mapping)
    for i, decision in zip(positions, before):
        if decision != payload["records"][i]["association"]:
            raise ValueError("source association cannot be replayed")
    trial = deepcopy(mapping)
    ground_evidence = []
    for instance, info in trial["instances"].items():
        if info["prim_path"] != GROUND:
            continue
        if (info["kind"] != "unknown" or info["object_id"] is not None or
                mapping["id_to_prim"].get(instance) != GROUND):
            raise ValueError("ground instance conflicts with source identity/role")
        pixels = mask == int(instance)
        if not pixels.any() or not (np.isfinite(raw_depth[pixels]) & (raw_depth[pixels] > 0)).all():
            raise ValueError("ground pixels lack positive finite raw depth")
        ground_evidence.append({"instance_id": instance, "prim_path": GROUND,
                                "positive_finite_depth_pixels": int(pixels.sum())})
        info["kind"] = "known_non_cube"
    after = associate(detections, mask, trial)
    overrides = []
    for i, old, new in zip(positions, before, after):
        original = records[i]
        target = new["object_id"] if new["status"] == "matched_cube" else None
        if target is not None and target not in labels:
            raise ValueError("new association has no original same-frame label evidence")
        label = labels[target] if target is not None else None
        record = replace(original, training_label=label,
                         association_review_required=new["status"] == "unresolved")
        if record.sensor_input != original.sensor_input or record.detector_prediction != original.detector_prediction:
            raise ValueError("derived annotation changed sensor/candidate data")
        records[i] = record
        overrides.append({"source_frame_id": original.frame_id,
            "source_record_sha256": digest_json(original.to_dict()),
            "old_association": old, "new_association": new,
            "training_label": None if label is None else label.to_dict(),
            "label_source_frame_id": label_sources[target] if target is not None else None,
            "association_review_required": record.association_review_required})
    body = {"capture_id": payload["capture_id"], "ground_evidence": ground_evidence, "overrides": overrides}
    return body, records


def derive_episode(folder, manifest):
    captures, records = [], []
    for capture in manifest["captures"]:
        path = contained(folder, f"captures/{capture}.json")
        payload = json.loads(path.read_text())
        with np.load(contained(folder, payload["oracle_only"]["mask_path"]), allow_pickle=False) as npz:
            if "raw_depth_to_image_plane_m" not in npz.files:
                raise ValueError("ground overlay requires retained raw-depth evidence")
            body, updated = derive_capture(payload, npz["instance_id"], npz["raw_depth_to_image_plane_m"])
        body["source_capture_sha256"] = file_sha(path)
        captures.append(body)
        records.extend(updated)
    return {"protocol": PROTOCOL, "episode_id": manifest["episode_id"],
            "source_episode_manifest_sha256": file_sha(folder/"episode_manifest.json"),
            "captures": captures}, records


def create_overlay(session, output):
    session, output = session.resolve(), output.resolve()
    if session == output or session in output.parents or output in session.parents:
        raise ValueError("derived output must be separate from the source session")
    if output.exists():
        raise ValueError("output already exists; choose a new version, never overwrite evidence")
    session, state, source_sha = load_source(session)
    bodies, entries, counts = [], [], Counter()
    for entry in state["episodes"]:
        folder, source_manifest = checked_source_episode(session, state, entry)
        body, _ = derive_episode(folder, source_manifest)
        bodies.append(body)
        counts.update(entry["stats"])
        print(f"Reassociated {len(bodies)}/{len(state['episodes'])} episodes", flush=True)
    if dict(counts) != state["summary"] or counts["physical_captures"] != state["contract"]["physical_captures"]:
        raise ValueError("source journal is incomplete/inconsistent")
    if file_sha(session/"session.json") != source_sha:
        raise ValueError("source session changed during derivation")
    output.mkdir(parents=True, exist_ok=False)
    (output/"overlays").mkdir()
    for body in bodies:
        name = f"overlays/{body['episode_id']}.json"
        path = contained(output, name)
        _atomic_write_json(path, body)
        entries.append({"episode_id": body["episode_id"], "filename": name, "sha256": file_sha(path)})
    result = {"protocol": PROTOCOL, "complete": True, "annotation_only": True,
        "source_session": str(session), "source_session_sha256": source_sha,
        "source_contract_sha256": digest_json(state["contract"]), "physical_captures": counts["physical_captures"],
        "rules": rules(), "derivation_code_sha256": code_hashes(),
        "window_config_sha256": file_sha(CONFIG), "episodes": entries,
        "source_modified": False, "rgbd_copied": False, "training_started": False,
        "ready_for_training": False, "needs_manual_visual_review": True}
    _atomic_write_json(output/"manifest.json", result)
    return result


def verified_overlay_episode(session, state, source_entry, overlay_root, overlay_entry):
    if overlay_entry["episode_id"] != source_entry["episode_id"]:
        raise ValueError("overlay/source episode mismatch")
    if overlay_entry["filename"] != f"overlays/{source_entry['episode_id']}.json":
        raise ValueError("unexpected overlay filename")
    path = contained(overlay_root, overlay_entry["filename"])
    if file_sha(path) != overlay_entry["sha256"]:
        raise ValueError("overlay checksum mismatch")
    folder, manifest = checked_source_episode(session, state, source_entry)
    expected, records = derive_episode(folder, manifest)
    if json.loads(path.read_text()) != expected:
        raise ValueError("overlay replay mismatch; labels/provenance must not be manually changed")
    return folder, records, expected


def derived_view(original, records):
    """Explicit opt-in view; root/sensor paths and measurement code stay original."""
    dataset = copy(original)
    dataset.sequences = tuple(s for s in build_sequences(records, history_size=original.config.history_size,
        max_history_age_s=original.config.max_history_age_s)
        if split_for_episode(s.reference.episode_id, seed=original.config.seed) == original.summary.split)
    dataset.summary = replace(original.summary, sequence_count=len(dataset.sequences),
        episode_ids=tuple(sorted({s.reference.episode_id for s in dataset.sequences})))
    return dataset


def audit_dataset(dataset):
    counts = Counter(windows=0, positive_reference_windows=0, negative_reference_windows=0,
                     positive_measurement_windows=0)
    reasons, rejected, windows = Counter(), [], []
    for i, seq in enumerate(dataset.sequences):
        batch = dataset[i]
        if any(not torch.isfinite(v).all() for v in batch.values()):
            raise ValueError("non-finite derived training batch")
        decision = measure_window(sensor_observations((*seq.history, seq.reference), dataset.root))[-1]
        if bool(batch["valid_depth_mask"]) != decision.accepted:
            raise ValueError("measurement acceptance differs from shared sensor-only code")
        if decision.accepted and not torch.allclose(batch["raw_depth_m"], torch.tensor(decision.surface.depth_m)):
            raise ValueError("measurement depth differs from shared sensor-only code")
        present, valid = bool(batch["target_present_mask"]), bool(batch["measurement_valid"])
        counts["windows"] += 1
        counts["positive_reference_windows"] += int(present)
        counts["negative_reference_windows"] += int(not present)
        counts["positive_measurement_windows"] += int(valid)
        reasons[decision.reason] += 1
        if present and not valid:
            r = seq.reference
            rejected.append({"reference_frame_id": r.frame_id, "visible": r.training_label.visible,
                "detected": r.detector_prediction.detected, "surface_accepted": decision.accepted,
                "center_in_image": bool(batch["reference_center_in_image"]),
                "target_depth_m": float(batch["target_depth_m"]), "measurement_reason": decision.reason})
        windows.append({**seq.to_dict(), "positive_measurement": valid, "measurement_reason": decision.reason})
    return dict(counts), dict(reasons), rejected, windows


def check_overlay(overlay_root):
    overlay_root = overlay_root.resolve()
    manifest_path = overlay_root/"manifest.json"
    manifest_sha = file_sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("complete") is not True:
        raise ValueError("not a complete ground-role overlay")
    if manifest["rules"] != rules() or manifest["derivation_code_sha256"] != code_hashes():
        raise ValueError("overlay rule/code version differs")
    if manifest["window_config_sha256"] != file_sha(CONFIG):
        raise ValueError("window/training configuration changed")
    session, state, source_sha = load_source(Path(manifest["source_session"]))
    if source_sha != manifest["source_session_sha256"] or digest_json(state["contract"]) != manifest["source_contract_sha256"]:
        raise ValueError("overlay belongs to a different source session/contract")
    if len(manifest["episodes"]) != len(state["episodes"]):
        raise ValueError("overlay has missing/extra episodes")
    listed = {"manifest.json", "acceptance_report.json"} | {e["filename"] for e in manifest["episodes"]}
    if any(str(p.relative_to(overlay_root)) not in listed for p in overlay_root.rglob("*") if p.is_file()):
        raise ValueError("unlisted overlay files")
    cfg = replace(load_training_config(CONFIG), device="cpu", num_workers=0,
        expected_yolo_model_sha256=state["contract"]["detector_deployment"]["model_sha256"])
    baseline, derived, old_status, new_status, measurement_reasons = Counter(), Counter(), Counter(), Counter(), Counter()
    rejected, windows, total_captures = [], [], 0
    for source_entry, overlay_entry in zip(state["episodes"], manifest["episodes"]):
        folder, records, body = verified_overlay_episode(session, state, source_entry, overlay_root, overlay_entry)
        total_captures += len(body["captures"])
        for capture in body["captures"]:
            for item in capture["overrides"]:
                old_status[item["old_association"]["status"]] += 1
                new_status[item["new_association"]["status"]] += 1
        split = split_for_episode(source_entry["episode_id"], seed=cfg.seed)
        original = TargetStateV3Dataset(cfg, episode_root=folder, split=split)
        base_counts, _, _, _ = audit_dataset(original)
        counts, reasons, bad, eligible = audit_dataset(derived_view(original, records))
        baseline.update(base_counts)
        derived.update(counts)
        measurement_reasons.update(reasons)
        rejected.extend(bad)
        windows.extend(eligible)
        print(f"Verified overlay: {source_entry['episode_id']}", flush=True)
    if total_captures != manifest["physical_captures"] or total_captures != state["contract"]["physical_captures"]:
        raise ValueError("overlay physical capture count mismatch")
    if file_sha(manifest_path) != manifest_sha or file_sha(session/"session.json") != source_sha:
        raise ValueError("source/overlay changed during acceptance")
    return {"complete": True, "overlay_replay_verified": True, "source_session_sha256": source_sha,
        "overlay_manifest_sha256": manifest_sha, "physical_captures": total_captures,
        "history_size": cfg.history_size, "max_history_age_s": cfg.max_history_age_s,
        "baseline_detections": dict(old_status), "derived_detections": dict(new_status),
        "baseline_dataset": dict(baseline), "derived_dataset": dict(derived),
        "measurement_reasons": dict(measurement_reasons), "positive_reference_rejections": rejected,
        "windows": windows, "source_modified": False, "sensor_detector_records_unchanged": True,
        "training_started": False, "production_approved": False, "ready_for_training": False,
        "needs_manual_visual_review": True,
        "notes": ["Ground is a known non-target surface, NOT infinite-depth no-hit background.",
                  "Original no-candidate oracle rows remain evidence; do not interpret their count as new detector misses.",
                  "Window eligibility and sensor-only parity do not establish correct physical labels or 3D accuracy.",
                  "This sidecar depends on the original RGB-D and oracle evidence; do not delete source data."]}
