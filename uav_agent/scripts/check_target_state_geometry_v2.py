#!/usr/bin/env python3
"""CPU regression on SHA-verified episode exports. No training or source mutation."""
import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader
from datasets.target_state.schema import TargetStateFrameRecord
from training.target_state.config import load_training_config
from training.target_state.data import TargetStateTorchDataset
from training.target_state.geometry import project_world_to_pixel
from training.target_state.measurement_gate import gate_batch
from training.target_state.sharded_trainer import _atomic_write_json


def check_episode(folder,config):
    path=folder/'episode_review.json'
    original=path.read_bytes(); payload=json.loads(original)
    if not payload.get('offline_only') or payload['episode_id']!=folder.name:
        raise ValueError('invalid episode export')
    assets=(folder/'assets').resolve()
    for relative,digest in payload['asset_sha256'].items():
        asset=(assets/relative).resolve(strict=True)
        if assets not in asset.parents or sha256(asset.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'export asset SHA mismatch: {relative}')
    records=[TargetStateFrameRecord.from_dict(row) for row in payload['records']]
    from datasets.target_state.dataset import split_for_episode
    split=split_for_episode(payload['episode_id'],seed=config.seed)
    cfg=replace(config,dataset_root=assets,require_dataset_manifest=False,device='cpu',num_workers=0)
    data=TargetStateTorchDataset(cfg,split=split,records=records)
    legacy=TargetStateTorchDataset(replace(cfg,supervision_protocol='legacy_v1',reference_guard_protocol='none'),
                                  split=split,records=records)
    if len(data)!=len(legacy): raise ValueError('overlay changed sequence count')
    counts=Counter(physical_captures=len({r.timestamp_s for r in records}),records=len(records),
                   verified_assets=len(payload['asset_sha256']),sequences=len(data))
    for batch in DataLoader(data,batch_size=8,shuffle=False):
        uv,z,valid=project_world_to_pixel(position_world_m=batch['target_position_world_m'],
            intrinsics_fx_fy_cx_cy=batch['intrinsics_fx_fy_cx_cy'],
            camera_position_world_m=batch['camera_position_world_m'],
            camera_orientation_world_wxyz=batch['camera_orientation_world_wxyz'])
        ideal=gate_batch(batch,corrected_depth_m=z,delta_uv_px=uv-batch['anchor_uv_px'],
                        validity_probability=torch.ones_like(z),maximum_depth_m=cfg.maximum_depth_m)
        contradictory=batch['measurement_valid'] & ~ideal.accepted
        error=torch.linalg.vector_norm(uv-batch['history_center_uv_px'][:,-1],dim=-1)
        if torch.any(contradictory) or torch.any(error[batch['target_present_mask'] & valid]>.002):
            raise ValueError('v2 positive/projection contract regression')
        counts.update(effective_positive_references=int(batch['measurement_valid'].sum()),
            center_outside_visible_references=int((batch['history_visible_mask'][:,-1] & ~batch['reference_center_in_image']).sum()),
            guarded_detected_references=int((~batch['missing_mask'][:,-1] & ~batch['reference_sensor_consistent']).sum()),
            unknown_association_supervision=int((~batch['validity_supervision_mask']).sum()),
            contradictory_positive_labels=int(contradictory.sum()))
    if path.read_bytes()!=original: raise ValueError('source labels changed')
    return dict(counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/target_state/train_geometry_v2_stageb_50k.yaml')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); config=load_training_config(args.config)
    if config.supervision_protocol!='projected_center_v2' or config.reference_guard_protocol!='rgbd_consistency_v1':
        raise ValueError('this check requires v2 supervision and RGB-D guard')
    torch.set_num_threads(4)
    folders=sorted(p.parent for p in args.review_root.glob('episode_*/episode_review.json'))
    if not folders: raise ValueError('no episode exports found')
    results={p.name:check_episode(p,config) for p in folders}
    report=dict(passed=True,offline_only=True,supervision_protocol=config.supervision_protocol,
                reference_guard_protocol=config.reference_guard_protocol,episodes=results,
                source_dataset_unchanged=True,model_trained=False,
                note='Structural regression only, not held-out model accuracy or a promotion gate.')
    if args.output:
        destination=args.output.resolve()
        if destination==args.review_root.resolve() or args.review_root.resolve() in destination.parents:
            raise ValueError('regression output must be outside the source review root')
        _atomic_write_json(destination,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
