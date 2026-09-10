"""Opt-in Stage-B loader; mask/catalog are verification/labels ONLY.

Legacy trainers cannot discover V3 capture envelopes as frames.jsonl. New
training integration must instantiate this loader and pin its policy in the
artifact. No old checkpoint is silently treated as a V3 checkpoint.
"""
from dataclasses import replace
import json

import numpy as np
from PIL import Image
import torch

from datasets.target_state.schema import TargetStateFrameRecord
from perception.rgbd_consistency import SensorObservation
from training.target_state.config import TrainingStage
from training.target_state.data import TargetStateTorchDataset
from target_state_v3.measurement import MeasurementPolicy, measure_window
from target_state_v3.storage import verify_episode,contained


def sensor_observations(records,root):
    """Explicit sensor allowlist. Neither training_label nor oracle_only is read."""
    observations=[]
    for r in records:
        s,d=r.sensor_input,r.detector_prediction
        c=s.camera
        with Image.open(contained(root,s.rgb_path)) as image:
            rgb=np.asarray(image.convert("RGB"),dtype=np.uint8)
        depth=np.load(contained(root,s.depth_path),allow_pickle=False)
        observations.append(SensorObservation(rgb,depth,d.bbox_xyxy_normalized if d.detected else None,
            (c.fx,c.fy,c.cx,c.cy),c.position_world_m,c.orientation_world_wxyz,r.timestamp_s,d.tracker_id))
    return tuple(observations)


class TargetStateV3Dataset(TargetStateTorchDataset):
    def __init__(self,config,*,episode_root,split,policy=MeasurementPolicy()):
        if config.stage is not TrainingStage.YOLO_DEPLOYMENT:
            raise ValueError("V3 pilot is sensor-only Stage B, not an oracle-clean Stage A loader")
        if (config.minimum_depth_m,config.maximum_depth_m,config.max_history_age_s)!=(
                policy.minimum_depth_m,policy.maximum_depth_m,policy.maximum_history_age_s):
            raise ValueError("training and runtime measurement policies differ")
        verify_episode(episode_root)
        manifest=json.loads((episode_root/"episode_manifest.json").read_text())
        if manifest["contract"]["measurement_preprocessing"]!=policy.contract():
            raise ValueError("episode preprocessing differs from training")
        deployment=manifest["contract"]["detector_deployment"]
        if deployment["model_sha256"]!=config.expected_yolo_model_sha256 or deployment["preflight_verified"] is not True:
            raise ValueError("unverified/different YOLO producer")
        records=[]
        for name in manifest["captures"]:
            payload=json.loads(contained(episode_root,f"captures/{name}.json").read_text())
            records.extend(TargetStateFrameRecord.from_dict(row["record"]) for row in payload["records"])
        self.measurement_policy=policy
        # Keep uncertain observations in sequence construction. The existing
        # builder rejects ANY window containing one; never remove rows first.
        super().__init__(replace(config,dataset_root=episode_root,require_dataset_manifest=False,
            supervision_protocol="projected_center_v2",reference_guard_protocol="none"),split=split,records=records)

    @property
    def artifact_preprocessing(self):
        return self.measurement_policy.contract()

    def __getitem__(self,index):
        batch=super().__getitem__(index)
        sequence=self.sequences[index]
        observations=sensor_observations((*sequence.history,sequence.reference),self.root)
        decision=measure_window(observations,self.measurement_policy)[-1]
        s=decision.surface
        batch["raw_depth_m"]=torch.tensor(0.0 if s is None else s.depth_m,dtype=torch.float32)
        batch["anchor_uv_px"]=torch.tensor((0.,0.) if s is None else s.uv_px,dtype=torch.float32)
        batch["valid_depth_mask"]=torch.tensor(decision.accepted)
        batch["reference_sensor_consistent"]=torch.tensor(decision.accepted)
        r=sequence.reference
        label=r.training_label
        positive=bool(label is not None and label.visible and r.detector_prediction.detected and decision.accepted
            and batch["reference_center_in_image"]
            and self.measurement_policy.minimum_depth_m<=float(batch["target_depth_m"])<=self.measurement_policy.maximum_depth_m)
        batch["measurement_valid"]=torch.tensor(positive)
        # Association has been established from same-frame integer instances,
        # not the legacy target-depth heuristic. Unknown windows were excluded.
        batch["validity_supervision_mask"]=torch.tensor(True)
        return batch
