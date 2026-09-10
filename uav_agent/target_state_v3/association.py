"""Privileged OFFLINE labels: lossless integer instances, explicit unknowns.

This module must never be imported by production measurement/runtime code.
Renderer IDs and prim paths are evidence, never candidate IDs or neural inputs.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np

from datasets.target_state.schema import (CameraFrameInput, DetectorPrediction,
    SensorInput, TargetStateFrameRecord, TargetTrainingLabel)
from datasets.target_state.projection import project_label_center, center_in_image
from training.target_state.isaac_capture import DetectorCandidateLinker, TargetStateFrameAssembler

PROTOCOL = "instance_supported_association_v3"
INSTANCE_EVIDENCE_PROTOCOL = "renderer_zero_raw_positive_infinity_v1"


@dataclass(frozen=True)
class AssociationPolicy:
    minimum_pixels: int = 6
    positive_fraction: float = 0.6
    positive_margin: float = 0.2
    negative_fraction: float = 0.8
    maximum_cube_fraction_for_negative: float = 0.05
    maximum_unknown_fraction: float = 0.05

    def to_dict(self):
        return {"protocol": PROTOCOL, **asdict(self)}


def normalize_instances(payload, *, shape_hw, catalog, raw_depth_m=None):
    """Match exact leaf prims; infer no-hit zero only from raw same-frame depth.

    Replicator's renderer-ID mapping can omit zero. Never fabricate a prim
    entry, interpret other reserved numbers, or use clipped/NaN sensor depth
    as evidence of background. An all-zero/uninitialized render fails closed.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        raise ValueError("instance annotator must return data and info.idToLabels")
    mask = np.asarray(payload.get("data"))
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[...,0]
    if mask.shape != shape_hw or mask.dtype != np.uint32:
        raise ValueError("instance mask must be uncolorized uint32 at RGB-D resolution")
    raw = payload["info"].get("idToLabels")
    if not isinstance(raw, dict):
        raise ValueError("missing per-frame instance ID mapping")
    roots = [v["prim_path"] for v in catalog]
    if len(set(roots)) != len(roots) or len({v["object_id"] for v in catalog}) != len(catalog):
        raise ValueError("duplicate object catalog identity")
    for root in roots:
        if not root.startswith("/World/CubeV1Collection/") or ".." in root.split("/"):
            raise ValueError("object catalog escapes collection scope")
    mapping = {}
    # Preserve supplied entries verbatim, including off-screen instances.
    raw = {str(k): v for k,v in raw.items()}
    rendered = {str(int(k)) for k in np.unique(mask)}
    missing = rendered-set(raw)
    if missing-{"0"}:
        raise ValueError(f"instance {sorted(missing-{'0'})[0]} has no same-frame mapping")
    depth = None
    evidence = {}
    if raw_depth_m is not None:
        depth = np.asarray(raw_depth_m)
        if depth.shape != shape_hw or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError("raw instance depth must be floating point at mask resolution")
    zero_background = "0" in rendered and ("0" not in raw or
        (isinstance(raw["0"], str) and raw["0"].upper() == "BACKGROUND"))
    if "0" in missing and depth is None:
        raise ValueError("instance 0 has no same-frame mapping or raw depth evidence")
    if zero_background and depth is not None:
        zero = mask == 0
        if not np.isposinf(depth[zero]).all():
            raise ValueError("instance 0 background requires raw positive-infinity depth; not NaN/clipped/finite depth")
        mapped_ids = [int(k) for k in rendered-{"0"}
                      if isinstance(raw.get(k), str) and raw[k].startswith("/")]
        finite_surface = np.isin(mask, mapped_ids) & np.isfinite(depth) & (depth > 0)
        if not finite_surface.any():
            raise ValueError("instance background has no mapped finite surface; render may be uninitialized")
        if "0" in missing:
            evidence["0"] = {"rule": INSTANCE_EVIDENCE_PROTOCOL,
                "pixel_count": int(zero.sum()), "raw_positive_infinity_pixels": int(zero.sum()),
                "mapped_finite_surface_pixels": int(finite_surface.sum())}
    for instance_id in np.unique(mask):
        key = str(int(instance_id))
        if key not in raw:
            # Only zero can reach here, after checking raw depth and readiness.
            mapping[key] = {"prim_path": None, "object_id": None, "shape": None,
                            "kind": "background", "source": INSTANCE_EVIDENCE_PROTOCOL}
            continue
        path = raw[key]
        if not isinstance(path, str):
            raise ValueError("instance_id_segmentation must map IDs to prim-path strings")
        matches = [v for v in catalog if path == v["prim_path"] or path.startswith(v["prim_path"]+"/")]
        if len(matches) > 1:
            raise ValueError("ambiguous prim mapping")
        obj = matches[0] if matches else None
        mapping[key] = {"prim_path": path, "object_id": None if obj is None else obj["object_id"],
            "shape": None if obj is None else obj["shape"],
            "kind": ("cube" if obj["shape"]=="cube" else "known_non_cube") if obj else
                    ("background" if int(instance_id)==0 and path.upper()=="BACKGROUND" else "unknown")}
    result = {"id_to_prim": raw, "instances": mapping, "objects": catalog}
    if depth is not None:
        result.update(depth_evidence_protocol=INSTANCE_EVIDENCE_PROTOCOL, background_evidence=evidence)
    return np.ascontiguousarray(mask).copy(), result


def associate(detections, mask, mapping, policy=AssociationPolicy()):
    h,w = mask.shape
    decisions = []
    for det in detections:
        if det.class_id != 0 or det.class_name.casefold() != "cube":
            raise ValueError("unexpected detector class")
        x1,y1,x2,y2 = det.bbox_xyxy_normalized
        x1,y1 = int(math.floor(x1*w)),int(math.floor(y1*h))
        x2,y2 = int(math.ceil(x2*w)),int(math.ceil(y2*h))
        dx,dy=x2-x1,y2-y1
        roi=mask[y1+int(.1*dy):y2-int(.1*dy), x1+int(.1*dx):x2-int(.1*dx)]
        if not roi.size:
            decisions.append(dict(status="unresolved",object_id=None,reason="empty_roi",support={}))
            continue
        counts=Counter()
        cube_pixels=unknown_pixels=background_pixels=non_cube_pixels=0
        for k,n in zip(*np.unique(roi,return_counts=True)):
            info=mapping["instances"][str(int(k))]
            n=int(n)
            counts[info["object_id"] or info["kind"]]+=n
            cube_pixels+=n if info["kind"]=="cube" else 0
            non_cube_pixels+=n if info["kind"]=="known_non_cube" else 0
            unknown_pixels+=n if info["kind"]=="unknown" else 0
            background_pixels+=n if info["kind"]=="background" else 0
        ordered=counts.most_common()
        object_id,pixels=ordered[0]
        fraction=pixels/roi.size
        runner=ordered[1][1]/roi.size if len(ordered)>1 else 0.0
        catalog={v["object_id"]:v for v in mapping["objects"]}
        status,reason="unresolved","mixed_or_unknown_instances"
        if (pixels>=policy.minimum_pixels and unknown_pixels/roi.size<=policy.maximum_unknown_fraction):
            if (object_id in catalog and catalog[object_id]["shape"]=="cube"
                    and fraction>=policy.positive_fraction and fraction-runner>=policy.positive_margin):
                status,reason="matched_cube","visible_instance_support"
            elif (cube_pixels/roi.size <= policy.maximum_cube_fraction_for_negative and
                  non_cube_pixels/roi.size >= policy.negative_fraction):
                status,reason="known_non_cube","known_distractor_support"
            elif background_pixels/roi.size>=.95:
                status,reason="background","mapped_background"
        decisions.append(dict(status=status,object_id=object_id if status=="matched_cube" else None,
            reason=reason, support=dict(counts), roi_pixels=int(roi.size),
            cube_fraction=cube_pixels/roi.size,unknown_fraction=unknown_pixels/roi.size))
    duplicate=Counter(d["object_id"] for d in decisions if d["status"]=="matched_cube")
    for d in decisions:
        if d["object_id"] is not None and duplicate[d["object_id"]]>1:
            d.update(status="unresolved",object_id=None,reason="duplicate_detection_for_instance")
    return decisions


class InstanceFrameAssembler:
    def __init__(self):
        self.linker=DetectorCandidateLinker(maximum_gap_s=2.0,minimum_iou=.1)

    def assemble(self, *, capture_id, episode_id, truth, response, uav, mask, mapping):
        sample=truth.camera_sample
        if response.frame_id != capture_id or abs(response.timestamp_s-sample.timestamp_s)>1e-9:
            raise ValueError("YOLO response/frame barrier mismatch")
        dets=tuple(response.detections)
        candidates=self.linker.assign_frame(dets,timestamp_s=sample.timestamp_s,
            appearance_keys=tuple(TargetStateFrameAssembler._appearance_key(sample.rgb,d) for d in dets))
        decisions=associate(dets,mask,mapping)
        objects={obj.object_id:obj for obj in truth.objects}
        k=sample.intrinsics
        camera=CameraFrameInput(k.fx,k.fy,k.cx,k.cy,sample.camera_position_world_m,
                                sample.camera_orientation_world_wxyz,(k.width,k.height))
        sensor=SensorInput(camera,uav,f"rgb/{capture_id}.png",f"depth/{capture_id}.npy")
        records=[]
        matched=set()

        def label_for(obj):
            center,_=project_label_center(obj.position_world_m,camera.position_world_m,
                camera.orientation_world_wxyz,(k.fx,k.fy,k.cx,k.cy))
            ids=[int(n) for n,v in mapping["instances"].items() if v["object_id"]==obj.object_id]
            visible=bool(ids and np.isin(mask,ids).any() and center is not None)
            return TargetTrainingLabel(obj.position_world_m,obj.velocity_world_mps,center if visible else None,visible,
                float(obj.occlusion_ratio or 0.0),obj.color_name,obj.object_id,
                center_in_image=center_in_image(center,(k.width,k.height)))

        for i,(det,candidate,decision) in enumerate(zip(dets,candidates,decisions)):
            obj=objects.get(decision["object_id"])
            if obj is not None:
                matched.add(obj.object_id)
            record=TargetStateFrameRecord(f"{capture_id}_d{i}",episode_id,episode_id,"uav_1",
                sample.timestamp_s,sensor,
                DetectorPrediction(True,det.bbox_xyxy_normalized,det.confidence,f"track_{det.track_id}",candidate),
                None if obj is None else label_for(obj),association_review_required=decision["status"]=="unresolved")
            records.append({"schema_version":3,"record":record.to_dict(),"association":decision,
                            "capture_id":capture_id})
        # Preserve detector misses in raw observations, without deriving a
        # runtime candidate from object truth. These are not invented tracks.
        for i,obj in enumerate(truth.objects):
            if obj.shape!="cube" or obj.object_id in matched:
                continue
            record=TargetStateFrameRecord(f"{capture_id}_m{i}",episode_id,episode_id,"uav_1",
                sample.timestamp_s,sensor,DetectorPrediction(False,None,None,None,None),label_for(obj))
            records.append({"schema_version":3,"record":record.to_dict(),"capture_id":capture_id,
                "association":{"status":"missed_cube","object_id":obj.object_id,"reason":"no_resolved_detection"}})
        if not records:
            record=TargetStateFrameRecord(f"{capture_id}_empty",episode_id,episode_id,"uav_1",
                sample.timestamp_s,sensor,DetectorPrediction(False,None,None,None,None),None)
            records.append({"schema_version":3,"record":record.to_dict(),"capture_id":capture_id,
                "association":{"status":"background","object_id":None,"reason":"no_detection"}})
        return records
