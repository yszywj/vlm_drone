#!/usr/bin/env python3
"""CPU-only structural and preprocessing acceptance, NOT held-out model accuracy."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from target_state_v3.storage import verify_episode
from target_state_v3.verify_tar import verify_tar,file_sha
from target_state_v3.data import TargetStateV3Dataset,sensor_observations
from target_state_v3.measurement import measure_window
from training.target_state.config import load_training_config
from training.target_state.sharded_trainer import _atomic_write_json


def check(root):
    state=json.loads((root/"session.json").read_text())
    if not state["complete"]:
        raise ValueError("pilot collection is not complete")
    cfg=load_training_config(ROOT/"configs/target_state/train_geometry_v2_stageb_50k.yaml")
    cfg=replace(cfg,device="cpu",num_workers=0,
        expected_yolo_model_sha256=state["contract"]["detector_deployment"]["model_sha256"])
    counts=Counter(supervised_windows=0,positive_measurement_windows=0,negative_reference_windows=0)
    reasons=Counter()
    for entry in state["episodes"]:
        folder=root/entry["episode_id"]
        counts.update(verify_episode(folder))
        archive=root/"archives"/entry["filename"]
        if file_sha(archive)!=entry["sha256"]:
            raise ValueError("pilot archive hash mismatch")
        verify_tar(archive,expected_contract=state["contract"],expected_episode=entry["episode_id"],
                   expected_manifest_sha=file_sha(folder/"episode_manifest.json"))
        for split in ("train","validation","test"):
            dataset=TargetStateV3Dataset(cfg,episode_root=folder,split=split)
            for i,sequence in enumerate(dataset.sequences):
                batch=dataset[i]
                decisions=measure_window(sensor_observations((*sequence.history,sequence.reference),folder))
                d=decisions[-1]
                if bool(batch["valid_depth_mask"])!=d.accepted:
                    raise ValueError("training/runtime acceptance drift")
                if d.accepted and not torch.allclose(batch["raw_depth_m"],torch.tensor(d.surface.depth_m)):
                    raise ValueError("training/runtime selected depth drift")
                if any(not torch.isfinite(v).all() for v in batch.values()):
                    raise ValueError("non-finite training batch")
                reasons[d.reason]+=1
                counts["supervised_windows"]+=1
                counts["positive_measurement_windows"]+=int(batch["measurement_valid"])
                counts["negative_reference_windows"]+=int(not batch["target_present_mask"])
        print(f"Verified V3 episode: {entry['episode_id']}",flush=True)
    if counts["physical_captures"]!=state["contract"]["physical_captures"]:
        raise ValueError("physical capture count mismatch")
    result={"complete":True,"structural_checks_passed":True,"counts":dict(counts),"measurement_reasons":dict(reasons),
        "training_runtime_preprocessing_identical":True if counts["supervised_windows"] else None,
        "training_runtime_windows_checked":counts["supervised_windows"],
        "training_runtime_preprocessing_check":"passed" if counts["supervised_windows"] else "not_run_no_supervised_windows",
        "model_trained":False,"production_approved":False,
        "source_session_sha256":file_sha(root/"session.json"),
        "needs_manual_visual_review":True,"ready_for_bulk_collection":False,
        "notes":["Same-frame mapping and schema checks are necessary, not proof of every physical label.",
                 "Pilot counts are not model accuracy. Review uncertain labels/misses and diversity before expansion."]}
    _atomic_write_json(root/"acceptance_report.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session",type=Path,required=True)
    torch.set_num_threads(4)
    print(json.dumps(check(p.parse_args().session.resolve()),indent=2))
