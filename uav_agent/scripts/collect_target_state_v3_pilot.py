#!/usr/bin/env python3
"""Collect at most 200 physical RGB-D/instance captures; no model training/deployment."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from configs.loader import load_config
from perception.yolo_client import YoloServiceClient
from perception.rgbd_consistency import SensorObservation
from scripts.collect_target_state_dataset import (_uav_input, EXPECTED_ENV, DEFAULT_MODEL_SHA256)
from scripts.collect_yolo_dataset import (_SimpleSceneCollectionAdapter,_UsdCubeV1SceneDriver,_randomization_bounds)
from training.yolo.collection_scene import load_cube_collection_protocol
from training.yolo.isaac_collector import EpisodeRandomizer
from training.target_state.collector import require_privileged_collection_acknowledgements
from training.target_state.isaac_capture import preflight_deployed_yolo
from training.target_state.sharded_trainer import _atomic_write_json,_exclusive_run_lock
from training.target_state.trainer import sha256_file
from yolo_service.protocol import ResetStreamRequest,TrackRequest,TargetQuery
from target_state_v3 import DATA_PROTOCOL
from target_state_v3.association import InstanceFrameAssembler,AssociationPolicy,INSTANCE_EVIDENCE_PROTOCOL
from target_state_v3.isaac_instances import enable_instances,snapshot_instances
from target_state_v3.measurement import MeasurementPolicy,measure_window
from target_state_v3.storage import EpisodeWriter,verify_episode,make_archive,publish_archive,safe_run_name


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--run-id",default="v3_pilot_200_v2")
    p.add_argument("--captures",type=int,default=200)
    p.add_argument("--scene-seed",type=int,default=20260910)
    p.add_argument("--config",type=Path,default=ROOT/"configs/default.yaml")
    p.add_argument("--collection-config",type=Path,default=ROOT/"configs/yolo/collect_cube.yaml")
    p.add_argument("--yolo-model-sha256",default=DEFAULT_MODEL_SHA256)
    p.add_argument("--yolo-url",default="http://127.0.0.1:8011")
    p.add_argument("--request-timeout-s",type=float,default=30.0)
    p.add_argument("--gpu-device",type=int,default=0)
    p.add_argument("--pc-trans-root",type=Path,default=Path("/home/amax/ry/pc_trans"))
    p.add_argument("--publish",action="store_true")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--oracle-label-generation",action="store_true")
    p.add_argument("--acknowledge-privileged-oracle",action="store_true")
    return p


def contract_for(args,receipt):
    # Versioned outside the old source glob; never invalidate existing caches.
    sources=[*sorted((ROOT/"target_state_v3").glob("*.py")),Path(__file__),
        ROOT/"scripts/collect_yolo_dataset.py",ROOT/"scripts/collect_target_state_dataset.py",
        *sorted((ROOT/"training/target_state").glob("*.py")),
        *sorted((ROOT/"training/yolo").glob("*.py")),
        *sorted((ROOT/"datasets/target_state").glob("*.py")),
        *sorted((ROOT/"env").glob("*.py")),ROOT/"perception/depth_geometry.py",
        ROOT/"perception/rgbd_consistency.py",ROOT/"runtime/frame_store.py",ROOT/"yolo_service/protocol.py"]
    return {"protocol":DATA_PROTOCOL,"run_id":args.run_id,"physical_captures":args.captures,
        "instance_evidence_protocol":INSTANCE_EVIDENCE_PROTOCOL,
        "frames_per_episode":20,"sample_hz":5,"scene_seed":args.scene_seed,
        "output":str(args.output.resolve()),"publish":args.publish,
        "pc_trans_root":str(args.pc_trans_root.resolve()),
        "config_sha256":sha256_file(args.config),"collection_config_sha256":sha256_file(args.collection_config),
        "measurement_preprocessing":MeasurementPolicy().contract(),
        "association_policy":AssociationPolicy().to_dict(),"detector_deployment":receipt.to_manifest_dict(),
        "source_sha256":{str(p.relative_to(ROOT)):sha256_file(p) for p in sources}}


def wait_for_space(root,bridge,pc):
    while True:
        free=shutil.disk_usage(root).free
        spool=bridge/"collection_spool"
        used=sum(p.stat().st_size for folder in ("ready","writing") for p in (spool/folder).glob("*") if p.is_file())
        if (not (bridge/"control/pause_collection.flag").exists()
                and free>=float(pc["min_filesystem_free_gib"])*1024**3+512*1024**2
                and used<float(pc["collection_spool_quota_gib"])*float(pc["collection_pause_ratio"])*1024**3):
            return
        print("Paused before episode: disk/collection backpressure; keep PC transfer running.",flush=True)
        time.sleep(5)


def run(args):
    require_privileged_collection_acknowledgements(oracle_label_generation=args.oracle_label_generation,
        acknowledge_privileged_oracle=args.acknowledge_privileged_oracle)
    safe_run_name(args.run_id)
    if args.captures<=0 or args.captures>200 or args.captures%20:
        raise ValueError("pilot captures must be a positive multiple of 20, at most 200")
    if args.scene_seed<0 or args.gpu_device<0:
        raise ValueError("invalid seed/GPU")
    output=args.output.resolve()
    if output==ROOT or output in ROOT.parents or output.name!=args.run_id:
        raise ValueError("output must be a dedicated directory named exactly as --run-id")
    config=load_config(args.config)
    protocol=load_cube_collection_protocol(args.collection_config)
    pc_config=args.pc_trans_root/"config/config.json"
    pc=json.loads(pc_config.read_text())
    bridge=Path(pc["bridge_root"]).resolve()
    if bridge==output or bridge in output.parents or output in bridge.parents:
        raise ValueError("pilot originals must be outside the transfer bridge")
    if args.publish and not (bridge/"collection_spool/writing").is_dir():
        raise ValueError("pc_trans bridge is not initialized")
    client=YoloServiceClient(base_url=args.yolo_url,request_timeout_s=args.request_timeout_s)
    receipt=preflight_deployed_yolo(client,expected_model_sha256=args.yolo_model_sha256)
    contract=contract_for(args,receipt)
    state_file=output/"session.json"
    if state_file.exists() and json.loads(state_file.read_text())["contract"]!=contract:
        raise ValueError("pilot contract changed; use a NEW run ID/output, never overwrite prior evidence")
    print(json.dumps({"preflight":True,"captures":args.captures,"episodes":args.captures//20,
        "output":str(output),"sha256":receipt.model_sha256,"publish":args.publish,
        "server_originals_retained":True,"training":False,"production_changed":False},indent=2),flush=True)
    if args.preflight_only:
        return 0
    if Path(sys.prefix).resolve()!=EXPECTED_ENV.resolve():
        raise RuntimeError("run with ./python.sh")
    with _exclusive_run_lock(output):
        if not state_file.exists() and any(p.name!=".sharded_training.lock" for p in output.iterdir()):
            raise ValueError("unowned nonempty pilot output; choose a new run ID/output")
        state=json.loads(state_file.read_text()) if state_file.exists() else {
            "protocol":DATA_PROTOCOL,"contract":contract,"episodes":[],"complete":False}
        state.pop("last_error",None)
        _atomic_write_json(state_file,state)
        # Verify even a completed rerun; do not contact PC or start Isaac if done.
        if len(state["episodes"])>args.captures//20:
            raise ValueError("pilot journal exceeds capture budget")
        for i,entry in enumerate(state["episodes"]):
            if (entry["episode_id"]!=f"s{args.scene_seed}_episode_{i:06d}" or
                    entry["filename"]!=f"shard_{args.run_id}_{i:06d}.tar"):
                raise ValueError("pilot journal identity mismatch")
            stats=verify_episode(output/entry["episode_id"])
            if stats!=entry["stats"] or stats["physical_captures"]!=20:
                raise ValueError("pilot journal count mismatch")
            archive=output/"archives"/entry["filename"]
            if sha256_file(archive)!=entry["sha256"]:
                raise ValueError("retained pilot archive checksum mismatch")
        if state["complete"]:
            if len(state["episodes"])*20!=args.captures:
                raise ValueError("completed pilot has missing episodes")
            print("Pilot already complete and local evidence verified.",flush=True)
            return 0
        from isaacsim import SimulationApp
        app=SimulationApp({"headless":True,"active_gpu":args.gpu_device,"physics_gpu":args.gpu_device,
            "anti_aliasing":0,"multi_gpu":False,"fast_shutdown":True,
            "extra_args":["--/rtx/post/aa/op=0","--/rtx-defaults/post/aa/op=0",
                "--/rtx-transient/post/aa/limitedOps=false","--/app/hydra/renderSettings/useUsdAttributes=false",
                "--/app/hydra/renderSettings/useFabricAttributes=false",
                "--/log/channels/isaacsim.core.simulation_manager.plugin=error",
                "--/log/channels/isaacsim.sensors.camera.camera=error"]})
        environment=None
        try:
            from env.simple_uav_search_env import SimpleUavSearchEnv
            environment=SimpleUavSearchEnv(config)
            environment.setup()
            driver=_UsdCubeV1SceneDriver(environment)
            adapter=_SimpleSceneCollectionAdapter(environment,app,config,protocol=protocol,
                                                  scene_driver=driver,crossing_trajectories=True)
            randomizer=EpisodeRandomizer(_randomization_bounds(config),scene_seed=args.scene_seed)
            mission=f"v3s{args.scene_seed}"
            stream=f"{mission}:uav_1"
            for ordinal in range(args.captures//20):
                if ordinal<len(state["episodes"]):
                    entry=state["episodes"][ordinal]
                else:
                    wait_for_space(output,bridge,pc)
                    episode=f"s{args.scene_seed}_episode_{ordinal:06d}"
                    folder=output/episode
                    plan=randomizer.plan(ordinal)
                    kind=("positive","partial_occlusion","negative")[ordinal%3]
                    plan=replace(plan,key=replace(plan.key,episode_id=episode),sample_kind=kind,
                        target_speed_mps=max(plan.target_speed_mps,min(.75,float(config.target.max_speed_mps)))
                        if kind=="partial_occlusion" else plan.target_speed_mps)
                    if folder.exists() and not (folder/"episode_manifest.json").exists():
                        recovery=output/"recovery"/f"{episode}_{time.time_ns()}"
                        recovery.parent.mkdir(exist_ok=True)
                        os.replace(folder,recovery)
                        print(f"Preserved incomplete episode: {recovery}",flush=True)
                    if not folder.exists():
                        writer=EpisodeWriter(folder,episode_id=episode,expected_captures=20,contract=contract)
                        adapter.begin_episode(plan)
                        sensor=enable_instances(environment)
                        assembler=InstanceFrameAssembler()
                        histories=defaultdict(list)
                        client.reset_stream(ResetStreamRequest(1,f"reset_{ordinal:06d}",mission,"uav_1",stream))
                        for frame in range(20):
                            adapter.advance_to_next_sample(.2)
                            capture=f"{episode}_f{frame:03d}"
                            truth=adapter.capture_oracle_frame(capture)
                            sample=truth.camera_sample
                            mask,mapping,raw_depth=snapshot_instances(sensor,truth,driver)
                            request=TrackRequest(1,f"req_{ordinal:06d}_{frame:03d}",mission,"uav_1",stream,
                                                 capture,sample.timestamp_s,TargetQuery(class_ids=(0,),text_prompts=()))
                            response=client.track(request,sample.rgb)
                            rows=assembler.assemble(capture_id=capture,episode_id=episode,truth=truth,response=response,
                                uav=_uav_input(environment),mask=mask,mapping=mapping)
                            diagnostics={}
                            for row in rows:
                                r=row["record"]
                                det=r["detector_prediction"]
                                candidate=det["candidate_id"]
                                if candidate is None:
                                    continue
                                k=sample.intrinsics
                                obs=SensorObservation(sample.rgb,sample.depth_to_image_plane_m,det["bbox_xyxy_normalized"],
                                    (k.fx,k.fy,k.cx,k.cy),sample.camera_position_world_m,sample.camera_orientation_world_wxyz,
                                    sample.timestamp_s,det["tracker_id"])
                                history=histories[candidate]
                                history.append(obs)
                                history[:]=[h for h in history[-7:] if obs.timestamp_s-h.timestamp_s<=2.0]
                                decision=measure_window(history)[-1]
                                diagnostics[r["frame_id"]]=asdict(decision)
                            writer.append(capture_id=capture,sample=sample,mask=mask,mapping=mapping,records=rows,
                                          measurement_diagnostics=diagnostics,raw_depth_m=raw_depth)
                        stats=writer.finalize()
                    else:
                        stats=verify_episode(folder)
                    filename=f"shard_{args.run_id}_{ordinal:06d}.tar"
                    archive=output/"archives"/filename
                    if archive.exists():
                        # A crash after archive completion but before the journal
                        # commit must not replace or blindly trust that archive.
                        from target_state_v3.verify_tar import verify_tar
                        verify_tar(archive,expected_contract=contract,expected_episode=episode,
                                   expected_manifest_sha=sha256_file(folder/"episode_manifest.json"))
                        entry={"filename":filename,"sha256":sha256_file(archive),"size_bytes":archive.stat().st_size}
                    else:
                        temporary=archive.with_suffix(".tar.tmp")
                        if temporary.exists():
                            recovery=output/"recovery"/f"{temporary.name}_{time.time_ns()}"
                            recovery.parent.mkdir(exist_ok=True)
                            os.replace(temporary,recovery)
                        entry=make_archive(folder,archive)
                    entry.update(episode_id=episode,stats=stats,published=False)
                    state["episodes"].append(entry)
                    _atomic_write_json(state_file,state)
                if args.publish and not entry["published"]:
                    publish_archive(output/"archives"/entry["filename"],pc_root=args.pc_trans_root,
                                    pc_config=pc_config,bridge=bridge)
                    entry["published"]=True
                    _atomic_write_json(state_file,state)
                print(f"Verified {ordinal+1}/{args.captures//20} episodes; {(ordinal+1)*20} physical captures",flush=True)
            state["complete"]=True
            summary=Counter()
            for entry in state["episodes"]:
                summary.update(entry["stats"])
            state["summary"]=dict(summary)
            state["server_originals_retained"]=True
            _atomic_write_json(state_file,state)
            print(json.dumps({"complete":True,"session":str(state_file),"summary":dict(summary)},indent=2),flush=True)
        except BaseException as exc:
            state["last_error"]=f"{type(exc).__name__}: {exc}"
            _atomic_write_json(state_file,state)
            print(f"V3 pilot stopped; committed episodes retained: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
            raise
        finally:
            try:
                if environment is not None:
                    environment.close()
            finally:
                app.close()
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
