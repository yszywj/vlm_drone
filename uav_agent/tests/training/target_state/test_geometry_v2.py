from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from datasets.target_state.projection import project_label_center
from datasets.target_state.schema import TargetStateFrameRecord
from training.target_state.data import TargetStateTorchDataset
from training.target_state.geometry import project_world_to_pixel
from training.target_state.measurement_gate import gate_batch
from training.target_state.losses import compute_target_state_losses
from training.target_state.sharded_trainer import _training_contract_sha256
from tests.training.target_state.test_data_and_trainer import _dataset, _config, _FixedEvaluationModel
from perception.rgbd_consistency import SensorObservation, guard_sequence


def test_projection_keeps_true_offscreen_center():
    uv, depth = project_label_center((2,0,-4), (0,0,0), (1,0,0,0), (100,100,32,24))
    assert uv == (32,224) and depth == 2


def test_read_overlay_keeps_archive_labels_and_sequence_count(tmp_path):
    root=tmp_path/'data'
    _dataset(root)
    path=root/'frames.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    # Correct position projects below the image. Legacy saved centre is clipped.
    for row in rows:
        row['training_label']['position_world_m']=[5,0,-10]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    original=path.read_bytes()
    legacy=_config(root,tmp_path/'out')
    old=TargetStateTorchDataset(legacy,split='train')
    new=TargetStateTorchDataset(replace(legacy,supervision_protocol='projected_center_v2',
        reference_guard_protocol='rgbd_consistency_v1'),split='train')
    assert len(old)==len(new)==1
    assert old[0]['measurement_valid']
    sample=new[0]
    assert not sample['measurement_valid'] and sample['validity_supervision_mask']
    assert sample['target_present_mask'] and sample['history_visible_mask'][-1]
    assert not sample['reference_center_in_image']
    assert sample['history_center_uv_px'][-1,1] > 24
    assert path.read_bytes()==original
    # Labels cannot affect Stage B features or its sensor-only guard.
    for field in ('roi_rgbd','geometry','missing_mask','reference_sensor_consistent'):
        assert torch.equal(sample[field],old[0][field])
    batch=next(iter(DataLoader(new,batch_size=1)))
    model=_FixedEvaluationModel(validity_logit=0.0)
    output=model(batch['roi_rgbd'],batch['geometry'],batch['missing_mask'])
    loss=compute_target_state_losses(output,batch)
    assert loss.depth_huber==0 and loss.position_3d_huber==0 and loss.reprojection_huber==0
    assert loss.validity_bce>0 and torch.isfinite(loss.total)


def test_v2_positive_labels_satisfy_ideal_sensor_gate(tmp_path):
    root=tmp_path/'data'; _dataset(root)
    cfg=replace(_config(root,tmp_path/'out'),supervision_protocol='projected_center_v2',
                reference_guard_protocol='rgbd_consistency_v1')
    data=TargetStateTorchDataset(cfg,split='train')
    batch=next(iter(DataLoader(data,batch_size=1)))
    uv,z,_=project_world_to_pixel(position_world_m=batch['target_position_world_m'],
        intrinsics_fx_fy_cx_cy=batch['intrinsics_fx_fy_cx_cy'],
        camera_position_world_m=batch['camera_position_world_m'],
        camera_orientation_world_wxyz=batch['camera_orientation_world_wxyz'])
    gate=gate_batch(batch,corrected_depth_m=z,delta_uv_px=uv-batch['anchor_uv_px'],
                    validity_probability=torch.ones_like(z),maximum_depth_m=cfg.maximum_depth_m)
    assert torch.all(~batch['measurement_valid'] | gate.accepted)
    assert batch['measurement_valid'].all()
    assert _training_contract_sha256(cfg)!=_training_contract_sha256(replace(cfg,reference_guard_protocol='none'))
    assert _training_contract_sha256(cfg)!=_training_contract_sha256(replace(cfg,supervision_protocol='legacy_v1'))


def observation(depth, rgb=None, time=0, position=(0,0,0)):
    if rgb is None: rgb=np.full((*depth.shape,3),(30,60,200),dtype=np.uint8)
    return SensorObservation(rgb,depth,(.1,.1,.9,.9),(80,80,16,16),position,(1,0,0,0),time,'track1')


def test_same_color_multi_depth_occluder_is_rejected():
    depth=np.full((32,32),5.,dtype=np.float32)
    depth[:,16:]=9.
    result=guard_sequence([observation(depth)])[0]
    assert not result.accepted and result.competing_surfaces==1
    assert result.reason=='multiple_depth_same_color_surfaces'


def test_rgb_depth_conflict_and_clear_single_surface():
    depth=np.full((32,32),5.,dtype=np.float32)
    rgb=np.full((32,32,3),(240,30,30),dtype=np.uint8)
    assert guard_sequence([observation(depth,rgb)])[0].accepted
    depth[:,16:]=9.; rgb[:,16:]=(30,30,240)
    assert guard_sequence([observation(depth,rgb)])[0].reason=='multiple_depth_rgb_surfaces'


def test_temporal_surface_jump_rejected_but_ego_motion_compensated():
    first=observation(np.full((32,32),5.),time=0)
    jump=observation(np.full((32,32),10.),time=.2)
    assert guard_sequence([first,jump])[-1].reason=='temporal_surface_jump'
    # Camera moved back five metres; world surface remains stationary.
    compensated=replace(jump,position=(-5,0,0))
    assert guard_sequence([first,compensated])[-1].accepted


def test_explicit_outside_flag_roundtrips_without_altering_legacy():
    from tests.training.target_state.test_dataset_schema import make_record
    legacy=make_record(0)
    assert 'center_in_image' not in legacy.to_dict()['training_label']
    updated=replace(legacy,training_label=replace(legacy.training_label,
                    center_pixel_uv=(32,200),center_in_image=False))
    assert TargetStateFrameRecord.from_dict(updated.to_dict())==updated


def test_v2_sharded_training_manifest_and_semantic_resume(tmp_path):
    from tests.training.target_state.test_shards import _write_parent_dataset
    from tests.training.target_state.test_sharded_trainer import _FakeLifecycle
    from training.target_state.shards import build_target_state_shards
    from training.target_state.sharded_trainer import train_target_state_sharded, ShardedTrainingOptions, validate_resume_checkpoint
    from training.target_state.config import TrainingStage
    parent=tmp_path/'parent'; _write_parent_dataset(parent)
    index=build_target_state_shards(parent,tmp_path/'shards',target_shard_size_bytes=1,
        history_size=4,max_history_age_s=2,split_seed=42).shard_index
    config=replace(_config(parent,tmp_path/'output',stage=TrainingStage.ORACLE_CLEAN),
        history_size=4,supervision_protocol='projected_center_v2',reference_guard_protocol='rgbd_consistency_v1')
    options=ShardedTrainingOptions(shard_index_path=index.source_path,pc_trans_root=tmp_path/'unused',
        pc_trans_config=tmp_path/'unused.json',bridge_root=tmp_path/'bridge',run_id_prefix='v2_fixture',wait_timeout_s=0)
    lifecycle=_FakeLifecycle(tmp_path/'shards',tmp_path/'bridge')
    result=train_target_state_sharded(config,options,lifecycle=lifecycle)
    manifest=json.loads(result.model_manifest.read_text())
    assert manifest['preprocessing']['reference_guard_protocol']=='rgbd_consistency_v1'
    assert manifest['supervision_protocol']=='projected_center_v2'
    assert manifest['input_fields']['geometry_25d']
    checkpoint=torch.load(result.best_checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['reference_guard_protocol']=='rgbd_consistency_v1'
    validate_resume_checkpoint(result.latest_checkpoint,config=config,index=index)
    with pytest.raises(Exception,match='training_contract_sha256'):
        validate_resume_checkpoint(result.latest_checkpoint,
            config=replace(config,reference_guard_protocol='none'),index=index)
