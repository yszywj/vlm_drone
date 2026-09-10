"""V3 opt-in pilot contracts; all tests are CPU/synthetic, no Isaac or SSH."""
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from target_state_v3.association import normalize_instances,associate,InstanceFrameAssembler
from target_state_v3.measurement import MeasurementPolicy,measure_window,require_v3_artifact
from target_state_v3.isaac_instances import snapshot_instances
from target_state_v3.storage import EpisodeWriter,verify_episode,make_archive,publish_archive
from target_state_v3.verify_tar import verify_tar,file_sha
from target_state_v3.data import TargetStateV3Dataset,sensor_observations
from target_state_v3.runtime import MultiClusterMeasurementResolver
from perception.rgbd_consistency import SensorObservation
from perception.grounding import CandidateResolutionUnavailable
from runtime.frame_store import FrameStore
from datasets.target_state.dataset import split_for_episode
from training.target_state.config import TargetStateTrainingConfig,TrainingStage
from training.target_state.collector import VerifiedYoloDeployment
from training.yolo.isaac_collector import OracleFrameTruth
from tests.training.target_state.test_isaac_capture import _sample,_truth_object,_detection,_response,_uav


def obs(t=0.,depth=None,rgb=None,tracker="track_1"):
    if depth is None:
        depth=np.full((48,64),10.,dtype=np.float32)
    if rgb is None:
        rgb=np.full((48,64,3),(15,30,220),dtype=np.uint8)
    return SensorObservation(rgb,depth,(.1,.1,.9,.9),(80.,80.,32.,24.),(0.,0.,1.),(1.,0.,0.,0.),t,tracker)


def test_multicluster_does_not_seed_at_occluded_bbox_centre():
    first=obs()
    depth=first.depth.copy(); depth[:,24:40]=4.
    rgb=first.rgb.copy(); rgb[:,24:40]=(220,220,10)
    current=obs(.2,depth,rgb)
    assert not measure_window([current])[-1].accepted
    result=measure_window([first,current])[-1]
    assert result.accepted and result.surface.depth_m==10.
    u,v=map(int,result.surface.uv_px)
    assert depth[v,u]==10.  # anchor belongs to selected pixels


def test_same_appearance_competing_surfaces_abstain():
    depth=np.full((48,64),9.5,dtype=np.float32);depth[:,32:]=10.5
    result=measure_window([obs(),obs(.2,depth)],MeasurementPolicy(motion_slack_m=.6))[-1]
    assert not result.accepted


@pytest.mark.parametrize("change",["jump","tracker","appearance"])
def test_temporal_discontinuity_never_silently_accepts(change):
    second=obs(.2)
    if change=="jump": second=replace(second,depth=np.full((48,64),30.))
    if change=="tracker": second=replace(second,tracker_id="track_9")
    if change=="appearance": second=replace(second,rgb=np.full((48,64,3),(220,20,5),dtype=np.uint8))
    assert not measure_window([obs(),second])[-1].accepted


@pytest.mark.parametrize("observations",[[obs(1),obs(1)],[obs(3),obs(1)],[obs(),obs(3)],[]])
def test_window_time_and_size_fail_closed(observations):
    with pytest.raises(ValueError): measure_window(observations)


@pytest.mark.parametrize("depth",[np.full((48,64),np.nan),np.zeros((48,64)),np.full((48,64),201.)])
def test_invalid_depth_never_outputs_position(depth):
    assert not measure_window([obs(depth=depth)])[-1].accepted


def test_legacy_or_unapproved_artifacts_are_rejected():
    with pytest.raises(ValueError,match="preprocessing"): require_v3_artifact({})
    with pytest.raises(ValueError,match="promotion"):
        require_v3_artifact({"measurement_preprocessing":MeasurementPolicy().contract()})


def catalog():
    return [{"object_id":key,"prim_path":"/World/CubeV1Collection/"+key,"shape":shape}
            for key,shape in (("cube_0","cube"),("cube_1","cube"),("panel","partial_noncube"))]


def instances(mask=None):
    if mask is None: mask=np.full((48,64),70000,dtype=np.uint32)
    return normalize_instances({"data":mask,"info":{"idToLabels":{
        "0":"BACKGROUND","1":"UNLABELLED","70000":"/World/CubeV1Collection/cube_0/Body",
        "70001":"/World/CubeV1Collection/cube_1/Body","90000":"/World/CubeV1Collection/panel/Body"}}},
        shape_hw=(48,64),catalog=catalog())


def test_integer_ids_over_16bit_preserved():
    mask,mapping=instances()
    assert mask.dtype==np.uint32 and mask.max()==70000
    assert mapping["instances"]["70000"]["object_id"]=="cube_0"


@pytest.mark.parametrize("bad",[np.zeros((48,64),np.uint8),np.zeros((48,64,4),np.uint8),np.zeros((64,48),np.uint32)])
def test_colorized_or_transposed_masks_rejected(bad):
    with pytest.raises(ValueError): instances(bad)


def test_missing_mapping_and_substring_identity_fail_closed():
    with pytest.raises(ValueError,match="mapping"):
        instances(np.full((48,64),123,dtype=np.uint32))
    mask,mapping=normalize_instances({"data":np.full((48,64),5,dtype=np.uint32),
        "info":{"idToLabels":{"5":"/World/CubeV1Collection/cube_01/Body"}}},shape_hw=(48,64),catalog=catalog())
    assert mapping["instances"]["5"]["kind"]=="unknown"
    assert associate([_detection(1)],mask,mapping)[0]["status"]=="unresolved"


@pytest.mark.parametrize("instance,status",[(70000,"matched_cube"),(90000,"known_non_cube"),(0,"background"),(1,"unresolved")])
def test_cube_non_cube_background_unknown_are_distinct(instance,status):
    mask,mapping=instances(np.full((48,64),instance,dtype=np.uint32))
    assert associate([_detection(1)],mask,mapping)[0]["status"]==status


def test_two_cubes_or_duplicate_detections_are_not_greedily_labeled():
    mask=np.full((48,64),70000,dtype=np.uint32);mask[:,30:]=70001
    mask,mapping=instances(mask)
    assert associate([_detection(1)],mask,mapping)[0]["status"]=="unresolved"
    mask,mapping=instances()
    assert all(d["status"]=="unresolved" for d in associate([_detection(1),_detection(2)],mask,mapping))


def assembled(sample,mask,mapping,assembler=None,capture="capture_1",episode="episode_1"):
    return (assembler or InstanceFrameAssembler()).assemble(capture_id=capture,episode_id=episode,
        truth=OracleFrameTruth(sample,objects=(_truth_object("cube_0"),)),
        response=_response(sample,(_detection(1),),frame_id=capture),uav=_uav(),mask=mask,mapping=mapping)


def test_unresolved_is_not_negative_and_truth_never_changes_sensor_candidate():
    a,b=InstanceFrameAssembler(),InstanceFrameAssembler()
    mask,mapping=instances()
    known=assembled(_sample(),mask,mapping,a)[0]
    mask2,map2=instances(np.ones((48,64),np.uint32))
    unknown=assembled(_sample(),mask2,map2,b)[0]
    assert known["record"]["detector_prediction"]==unknown["record"]["detector_prediction"]
    assert unknown["association"]["status"]=="unresolved"
    assert unknown["record"]["association_review_required"]
    assert unknown["record"]["training_label"] is None
    assert known["record"]["sensor_input"]["instance_mask_path"] is None


def test_invisible_cube_preserved_without_fake_sensor_candidate():
    mask,mapping=instances(np.zeros((48,64),np.uint32))
    rows=assembled(_sample(),mask,mapping)
    missed=next(r for r in rows if r["association"]["status"]=="missed_cube")
    assert missed["record"]["detector_prediction"]["candidate_id"] is None
    assert missed["record"]["training_label"]["center_pixel_uv"] is None


def test_same_callback_mask_rgb_depth_checks_reject_drift():
    sample=replace(_sample(),render_frame_id=(5,30))
    truth=OracleFrameTruth(sample,objects=(_truth_object("cube_0"),))
    frame={"rgb":sample.rgb,"distance_to_image_plane":sample.depth_to_image_plane_m,
        "rendering_time":sample.timestamp_s,"instance_id_segmentation":{
            "data":np.full((48,64),70000,np.uint32),
            "info":{"idToLabels":{"70000":"/World/CubeV1Collection/cube_0/Body"}}}}
    sensor=SimpleNamespace(camera=SimpleNamespace(get_current_frame=lambda clone:frame),
        _render_frame_id_from_frame=lambda f:(5,30),_rgb_from_frame=lambda f:f["rgb"])
    driver=SimpleNamespace(_roots={"cube_0":SimpleNamespace(GetPath=lambda:"/World/CubeV1Collection/cube_0")})
    mask,_,raw_depth=snapshot_instances(sensor,truth,driver)
    assert mask.max()==70000
    assert np.array_equal(raw_depth,sample.depth_to_image_plane_m)
    frame["rendering_time"]+=.1
    with pytest.raises(RuntimeError,match="barrier"):snapshot_instances(sensor,truth,driver)
    frame["rendering_time"]=sample.timestamp_s
    frame["distance_to_image_plane"]=np.full((48,64),4.)
    with pytest.raises(RuntimeError,match="depth"):snapshot_instances(sensor,truth,driver)


def write_episode(root,count=7,unknown_at=None):
    episode="s20260910_episode_000000"
    deployment=VerifiedYoloDeployment("http://127.0.0.1:8011","yolo",((0,"cube"),),"8"*64)
    contract={"measurement_preprocessing":MeasurementPolicy().contract(),"detector_deployment":deployment.to_manifest_dict()}
    writer=EpisodeWriter(root,episode_id=episode,expected_captures=count,contract=contract)
    assembler=InstanceFrameAssembler()
    for i in range(count):
        capture=f"capture_{i}"
        sample=replace(_sample(1.+.2*i),render_frame_id=(i+1,30))
        mask,mapping=instances(np.full((48,64),1 if i==unknown_at else 70000,np.uint32))
        mapping.update(render_frame_id=list(sample.render_frame_id),timestamp_s=sample.timestamp_s,offline_only=True)
        rows=assembled(sample,mask,mapping,assembler,capture,episode)
        writer.append(capture_id=capture,sample=sample,mask=mask,mapping=mapping,records=rows,measurement_diagnostics={})
    writer.finalize()
    return episode,contract


def config_for(root):
    return TargetStateTrainingConfig(dataset_root=root,output_dir=root.parent/"model",
        stage=TrainingStage.YOLO_DEPLOYMENT,expected_yolo_model_sha256="8"*64,require_dataset_manifest=False,
        minimum_depth_m=.2,maximum_depth_m=200.,device="cpu",history_size=6,max_history_age_s=2.)


def test_episode_checksums_integer_archive_and_train_runtime_parity(tmp_path):
    folder=tmp_path/"episode"
    episode,contract=write_episode(folder)
    assert verify_episode(folder)["matched_cube"]==7
    archive=tmp_path/"shard_v3_pilot_000000.tar"
    entry=make_archive(folder,archive)
    verified=verify_tar(archive,expected_contract=contract,expected_episode=episode)
    assert entry["sha256"]==file_sha(archive)
    dataset=TargetStateV3Dataset(config_for(folder),episode_root=folder,split=split_for_episode(episode,seed=42))
    assert len(dataset)==1
    batch=dataset[0]
    seq=dataset.sequences[0]
    observations=sensor_observations((*seq.history,seq.reference),folder)
    decision=measure_window(observations)[-1]
    assert decision.accepted
    assert float(batch["raw_depth_m"])==decision.surface.depth_m
    assert np.allclose(batch["anchor_uv_px"],decision.surface.uv_px)
    assert batch["geometry"].shape==(7,25)
    assert batch["roi_rgbd"].shape[0]==7
    with patch("numpy.load",side_effect=AssertionError("oracle mask read")):
        # Sensor layer is already materialized; no file/label access at runtime.
        assert measure_window(observations)==measure_window(observations)
    with pytest.raises(ValueError,match="contract"):verify_tar(archive,expected_contract={})


def test_unknown_history_cannot_be_removed_to_bridge_a_gap(tmp_path):
    folder=tmp_path/"episode"
    episode,_=write_episode(folder,count=9,unknown_at=3)
    dataset=TargetStateV3Dataset(config_for(folder),episode_root=folder,split=split_for_episode(episode,seed=42))
    assert len(dataset)==0


def test_corrupt_oracle_asset_stops_before_training_or_archive(tmp_path):
    folder=tmp_path/"episode";write_episode(folder)
    (folder/"oracle/capture_0.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="checksum"):verify_episode(folder)
    with pytest.raises(ValueError,match="checksum"):make_archive(folder,tmp_path/"shard_bad.tar")


def test_partial_episode_is_not_sealed(tmp_path):
    writer=EpisodeWriter(tmp_path/"episode",episode_id="episode_1",expected_captures=20,contract={})
    with pytest.raises(ValueError,match="partial"):writer.finalize()


def test_archive_no_overwrite_and_publication_keeps_original(tmp_path):
    folder=tmp_path/"episode";write_episode(folder)
    archive=tmp_path/"shard_v3_pilot_000000.tar";make_archive(folder,archive)
    bridge=tmp_path/"bridge"
    (bridge/"collection_spool/writing").mkdir(parents=True)
    (bridge/"collection_spool/ready").mkdir(parents=True)
    with patch("target_state_v3.storage.subprocess.run") as seal:
        publish_archive(archive,pc_root=tmp_path,pc_config=tmp_path/"config.json",bridge=bridge)
        seal.assert_called_once()
    assert archive.exists() and folder.exists()
    assert file_sha(bridge/"collection_spool/writing"/(archive.name+".tmp"))==file_sha(archive)


def test_runtime_adapter_requires_full_history_and_blocks_legacy_artifact():
    store=FrameStore(max_frames=10,max_bytes=10_000_000,max_age_s=10.)
    refs=[]
    for i in range(7):
        refs.append(store.add_sample(uav_id="uav_1",frame_id=f"frame_{i}",sample=_sample(.2*i)))
    candidate=SimpleNamespace(frame_history=tuple(refs),bbox_history=((.2,.2,.8,.8),)*7,
        tracker_id_history=("track_1",)*7,last_seen_timestamp_s=1.2,candidate_id="candidate_1")
    resolver=MultiClusterMeasurementResolver(store)
    surface=resolver.resolve(candidate,timestamp_s=1.2,diagnostic_surface_only=True)
    assert surface.corrected_depth_m==5.
    assert "surface_diagnostic" in surface.source
    with pytest.raises(ValueError,match="preprocessing"):resolver.resolve(candidate,timestamp_s=1.2,artifact_manifest={})
    with pytest.raises(CandidateResolutionUnavailable,match="stale"):resolver.resolve(candidate,timestamp_s=10.,diagnostic_surface_only=True)


def test_collector_preflight_is_read_only_and_before_isaac(tmp_path):
    import scripts.collect_target_state_v3_pilot as collector
    args=collector.parser().parse_args(["--output",str(tmp_path/"v3_pilot_200_v2"),"--preflight-only",
        "--oracle-label-generation","--acknowledge-privileged-oracle"])
    receipt=VerifiedYoloDeployment("http://127.0.0.1:8011","yolo",((0,"cube"),),"8"*64)
    with patch.object(collector,"preflight_deployed_yolo",return_value=receipt):
        assert collector.run(args)==0
    assert not args.output.exists()


def test_collector_refuses_bulk_or_unacknowledged_collection_before_contact(tmp_path):
    import scripts.collect_target_state_v3_pilot as collector
    args=collector.parser().parse_args(["--output",str(tmp_path/"v3_pilot_200_v2"),"--captures","50000",
        "--oracle-label-generation","--acknowledge-privileged-oracle"])
    with patch.object(collector,"preflight_deployed_yolo") as preflight:
        with pytest.raises(ValueError,match="at most 200"):collector.run(args)
        preflight.assert_not_called()
    args.captures=200;args.acknowledge_privileged_oracle=False
    with pytest.raises(Exception,match="acknowledge"):collector.run(args)


def test_tar_symlinks_traversal_and_tampering_rejected(tmp_path):
    import tarfile
    for name,kind in (("../evil",tarfile.REGTYPE),("symlink",tarfile.SYMTYPE)):
        archive=tmp_path/("link.tar" if kind==tarfile.SYMTYPE else "traversal.tar")
        with tarfile.open(archive,"w") as tar:
            entry=tarfile.TarInfo(name);entry.type=kind;entry.linkname="/tmp/evil"
            tar.addfile(entry)
        with pytest.raises(ValueError,match="unsafe"):verify_tar(archive)


def test_training_features_are_independent_of_labels_and_oracle_metadata(tmp_path):
    folder=tmp_path/"episode";episode,_=write_episode(folder)
    dataset=TargetStateV3Dataset(config_for(folder),episode_root=folder,split=split_for_episode(episode,seed=42))
    before=dataset[0]
    seq=dataset.sequences[0]
    def relabel(r):
        return replace(r,training_label=replace(r.training_label,position_world_m=(30.,3.,7.),
                                               velocity_world_mps=(100.,20.,0.),color_name="yellow"))
    dataset.sequences=(replace(seq,history=tuple(relabel(r) for r in seq.history),reference=relabel(seq.reference)),)
    after=dataset[0]
    for field in ("roi_rgbd","geometry","raw_depth_m","anchor_uv_px","reference_sensor_consistent","missing_mask"):
        assert torch.equal(before[field],after[field]),field


def test_new_batch_runs_frozen_network_and_loss_on_cpu(tmp_path):
    from training.target_state.model import TemporalRayDepthNet
    from training.target_state.losses import compute_target_state_losses
    folder=tmp_path/"episode";episode,_=write_episode(folder)
    cfg=config_for(folder)
    dataset=TargetStateV3Dataset(cfg,episode_root=folder,split=split_for_episode(episode,seed=42))
    batch={k:v.unsqueeze(0) for k,v in dataset[0].items()}
    model=TemporalRayDepthNet().eval()
    with torch.no_grad():
        output=model(batch["roi_rgbd"],batch["geometry"],batch["missing_mask"])
        loss=compute_target_state_losses(output,batch)
    assert torch.isfinite(output.depth_residual_m).all()
    assert torch.isfinite(loss.total)
