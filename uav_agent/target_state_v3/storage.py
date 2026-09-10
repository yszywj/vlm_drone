"""Small pilot storage: retain source evidence; publish only verified complete episodes.

Not a V1 collection shard. Old finalizers/trainers must not consume this format.
The bounded pilot keeps server originals, including incomplete work for review.
"""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image

from datasets.target_state.schema import TargetStateFrameRecord
from target_state_v3 import DATA_PROTOCOL
from target_state_v3.association import normalize_instances, INSTANCE_EVIDENCE_PROTOCOL
from training.target_state.sharded_trainer import _atomic_write_json, _fsync_directory
from training.target_state.trainer import sha256_file


def contained(root, relative):
    p=Path(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("unsafe relative path")
    path=root/p
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError("evidence path escapes root")
    return path


def durable(path):
    with path.open("rb") as f:
        os.fsync(f.fileno())
    _fsync_directory(path.parent)


def verify_instance_evidence(mask, mapping, raw_depth_m, contract):
    if (mapping.get("depth_evidence_protocol") is not None or
            contract.get("instance_evidence_protocol") == INSTANCE_EVIDENCE_PROTOCOL) and raw_depth_m is None:
        raise ValueError("missing retained raw depth for instance background evidence")
    _, expected = normalize_instances({"data":mask,"info":{"idToLabels":mapping["id_to_prim"]}},
        shape_hw=mask.shape,catalog=mapping["objects"],raw_depth_m=raw_depth_m)
    for key in ("instances","background_evidence","depth_evidence_protocol"):
        if mapping.get(key) != expected.get(key):
            raise ValueError(f"instance evidence replay mismatch: {key}")


class EpisodeWriter:
    def __init__(self,path,*,episode_id,expected_captures,contract):
        self.root=path
        path.mkdir(parents=True,exist_ok=False)
        self.episode_id=episode_id
        self.expected_captures=expected_captures
        self.contract=contract
        self.captures=[]

    def append(self,*,capture_id,sample,mask,mapping,records,measurement_diagnostics,raw_depth_m=None):
        if capture_id in self.captures:
            raise ValueError("duplicate physical capture")
        verify_instance_evidence(mask,mapping,raw_depth_m,self.contract)
        for folder in ("rgb","depth","oracle","captures"):
            (self.root/folder).mkdir(exist_ok=True)
        rgb=self.root/"rgb"/f"{capture_id}.png"
        depth=self.root/"depth"/f"{capture_id}.npy"
        instance=self.root/"oracle"/f"{capture_id}.npz"
        if mask.dtype!=np.uint32 or mask.shape!=sample.rgb.shape[:2]:
            raise ValueError("lossless integer mask required")
        Image.fromarray(sample.rgb).save(rgb)
        np.save(depth,sample.depth_to_image_plane_m,allow_pickle=False)
        evidence={"instance_id":mask}
        if raw_depth_m is not None:
            # Raw +/-inf/NaN distinctions are evidence, not network inputs.
            evidence["raw_depth_to_image_plane_m"]=raw_depth_m
        np.savez_compressed(instance,**evidence)
        for asset in (rgb,depth,instance):
            durable(asset)
        payload={"protocol":DATA_PROTOCOL,"capture_id":capture_id,"episode_id":self.episode_id,
            "timestamp_s":sample.timestamp_s,"render_frame_id":list(sample.render_frame_id),
            "oracle_only":{"mask_path":str(instance.relative_to(self.root)),"mapping":mapping},
            "records":records,"sensor_measurement_diagnostics":measurement_diagnostics}
        _atomic_write_json(self.root/"captures"/f"{capture_id}.json",payload)
        self.captures.append(capture_id)

    def finalize(self):
        if len(self.captures)!=self.expected_captures:
            raise ValueError("partial episodes cannot be sealed")
        files={str(p.relative_to(self.root)):sha256_file(p) for p in sorted(self.root.rglob("*")) if p.is_file()}
        manifest={"protocol":DATA_PROTOCOL,"episode_id":self.episode_id,"contract":self.contract,
                  "physical_capture_count":len(self.captures),"captures":self.captures,"sha256":files}
        _atomic_write_json(self.root/"episode_manifest.json",manifest)
        return verify_episode(self.root)


def verify_episode(root):
    manifest=json.loads((root/"episode_manifest.json").read_text())
    if manifest.get("protocol")!=DATA_PROTOCOL:
        raise ValueError("not a V3 episode")
    listed=set(manifest["sha256"])
    actual={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual!=listed|{"episode_manifest.json"}:
        raise ValueError("unlisted/missing episode assets")
    for name,digest in manifest["sha256"].items():
        path=contained(root,name)
        if sha256_file(path)!=digest:
            raise ValueError(f"episode asset checksum mismatch: {name}")
    if len(set(manifest["captures"]))!=manifest["physical_capture_count"]:
        raise ValueError("capture count mismatch")
    stats=Counter(physical_captures=manifest["physical_capture_count"])
    frame_ids=set()
    previous=None
    for capture in manifest["captures"]:
        payload=json.loads(contained(root,f"captures/{capture}.json").read_text())
        if payload["capture_id"]!=capture or payload["episode_id"]!=manifest["episode_id"]:
            raise ValueError("capture identity mismatch")
        timestamp=payload["timestamp_s"]
        if previous is not None and timestamp<=previous:
            raise ValueError("non-increasing physical capture clock")
        previous=timestamp
        oracle=payload["oracle_only"]
        if oracle["mapping"]["render_frame_id"]!=payload["render_frame_id"] or oracle["mapping"]["timestamp_s"]!=timestamp:
            raise ValueError("mask render barrier mismatch")
        with np.load(contained(root,oracle["mask_path"]),allow_pickle=False) as archive:
            if set(archive.files) not in ({"instance_id"},{"instance_id","raw_depth_to_image_plane_m"}):
                raise ValueError("mask archive fields mismatch")
            mask=archive["instance_id"]
            raw_depth=archive["raw_depth_to_image_plane_m"] if "raw_depth_to_image_plane_m" in archive.files else None
        if mask.dtype!=np.uint32 or mask.ndim!=2:
            raise ValueError("invalid integer instance map")
        if not {str(int(k)) for k in np.unique(mask)}<=set(oracle["mapping"]["instances"]):
            raise ValueError("unmapped instance pixels")
        verify_instance_evidence(mask,oracle["mapping"],raw_depth,manifest["contract"])
        for row in payload["records"]:
            if row["schema_version"]!=3 or row["capture_id"]!=capture:
                raise ValueError("V3 envelope mismatch")
            r=TargetStateFrameRecord.from_dict(row["record"])
            status=row["association"]["status"]
            if status not in {"matched_cube","known_non_cube","background","unresolved","missed_cube"}:
                raise ValueError("unknown association status")
            if r.frame_id in frame_ids or r.episode_id!=manifest["episode_id"] or r.timestamp_s!=timestamp:
                raise ValueError("record identity mismatch")
            if r.association_review_required != (status=="unresolved"):
                raise ValueError("unresolved record was made supervised")
            if (status in {"matched_cube","missed_cube"}) != (r.training_label is not None):
                raise ValueError("association/label mismatch")
            if r.sensor_input.instance_mask_path is not None:
                raise ValueError("oracle instance map leaked into sensor namespace")
            for relative in (r.sensor_input.rgb_path,r.sensor_input.depth_path):
                if relative not in listed:
                    raise ValueError("missing RGB-D asset")
            if mask.shape!=tuple(reversed(r.sensor_input.camera.resolution_wh_px)):
                raise ValueError("mask/camera shape mismatch")
            frame_ids.add(r.frame_id)
            stats[status]+=1
            stats["records"]+=1
    return dict(stats)


def make_archive(root,destination):
    """Deterministic archive; never replace different committed data."""
    verify_episode(root)
    temporary=destination.with_suffix(".tar.tmp")
    if temporary.exists():
        raise ValueError("unfinished archive exists; preserve and review it before retry")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(temporary,"x") as tar:
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            info=tar.gettarinfo(str(p),arcname=str(p.relative_to(root)))
            info.uid=info.gid=0
            info.uname=info.gname=""
            info.mtime=0
            info.mode=0o600
            with p.open("rb") as f:
                tar.addfile(info,f)
    durable(temporary)
    if destination.exists():
        raise ValueError("archive already exists")
    os.replace(temporary,destination)
    durable(destination)
    from target_state_v3.verify_tar import verify_tar
    verify_tar(destination,expected_manifest_sha=sha256_file(root/"episode_manifest.json"))
    return {"filename":destination.name,"sha256":sha256_file(destination),"size_bytes":destination.stat().st_size}


def publish_archive(archive,*,pc_root,pc_config,bridge):
    """Transport-only publication. Pilot originals are ALWAYS retained."""
    ready=bridge/"collection_spool"/"ready"/archive.name
    writing=bridge/"collection_spool"/"writing"/(archive.name+".tmp")
    if ready.exists():
        if sha256_file(ready)!=sha256_file(archive):
            raise ValueError("ready archive collision")
        return
    if writing.exists():
        if sha256_file(writing)!=sha256_file(archive):
            # Preserve interrupted copies for inspection instead of deleting.
            recovery=writing.with_name(writing.name+".interrupted")
            if recovery.exists():
                raise ValueError("multiple interrupted publications; manual review required")
            os.replace(writing,recovery)
    if not writing.exists():
        with archive.open("rb") as source,writing.open("xb") as output:
            shutil.copyfileobj(source,output)
            output.flush()
            os.fsync(output.fileno())
    if sha256_file(writing)!=sha256_file(archive):
        raise ValueError("publication copy failed verification")
    subprocess.run([sys.executable,"-m","pc_trans.cli","--config",str(pc_config),"seal","--src",str(writing)],
                   cwd=pc_root,check=True)


def safe_run_name(value):
    if not re.fullmatch(r"v3_[a-z0-9_]{1,32}",value):
        raise ValueError("run ID must match v3_[a-z0-9_]{1,32}")
    return value
