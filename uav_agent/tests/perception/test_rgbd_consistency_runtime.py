import json
from hashlib import sha256
import numpy as np
import pytest
import torch

from perception.temporal_ray_depth import TemporalRayDepthResolver, TemporalMeasurementRejected
from runtime.frame_store import FrameStore
from tests.perception.test_temporal_ray_depth import _artifact, _frames


def artifact(root):
    checkpoint,_ = _artifact(root)
    payload=torch.load(checkpoint,weights_only=True)
    payload['reference_guard_protocol']='rgbd_consistency_v1'
    torch.save(payload,checkpoint)
    digest=sha256(checkpoint.read_bytes()).hexdigest()
    path=root/'model_manifest.json'
    manifest=json.loads(path.read_text())
    manifest['checkpoint_sha256']=digest
    manifest['preprocessing']['reference_guard_protocol']='rgbd_consistency_v1'
    path.write_text(json.dumps(manifest))
    return checkpoint,digest


def test_clear_v2_runtime_measurement_matches_legacy_depth(tmp_path):
    checkpoint,digest=artifact(tmp_path)
    store=FrameStore(max_frames=16,max_bytes=2_000_000,max_age_s=10)
    resolver=TemporalRayDepthResolver(store,checkpoint_path=checkpoint,expected_sha256=digest,
        history_size=4,roi_size_px=32,deterministic_fallback=True)
    resolver.reset(uav_id='uav_1',assignment_id='assignment_v2')
    _,candidate=_frames(5,store=store)
    result=resolver.resolve(candidate,timestamp_s=.8)
    assert result.corrected_depth_m==pytest.approx(4)
    assert resolver.statistics.fallback_total==0


def test_guard_rejects_ambiguity_before_fallback_even_with_short_history(tmp_path,monkeypatch):
    checkpoint,digest=artifact(tmp_path)
    store=FrameStore(max_frames=16,max_bytes=2_000_000,max_age_s=10)
    resolver=TemporalRayDepthResolver(store,checkpoint_path=checkpoint,expected_sha256=digest,
        history_size=4,roi_size_px=32,deterministic_fallback=True)
    resolver.reset(uav_id='uav_1',assignment_id='assignment_v2')
    _,candidate=_frames(1,store=store)
    get=store.get_temporal_inputs
    def occluded(ref):
        rgb,depth,camera,motion=get(ref)
        depth=depth.copy();depth[:,16:]=9
        return rgb,depth,camera,motion
    monkeypatch.setattr(store,'get_temporal_inputs',occluded)
    with pytest.raises(TemporalMeasurementRejected,match='multiple_depth'):
        resolver.resolve(candidate,timestamp_s=0)
    assert resolver.statistics.fallback_total==0


def test_guard_cannot_be_disabled_by_editing_only_manifest(tmp_path):
    checkpoint,digest=artifact(tmp_path)
    path=tmp_path/'model_manifest.json'
    manifest=json.loads(path.read_text());del manifest['preprocessing']['reference_guard_protocol']
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='reference guard'):
        TemporalRayDepthResolver(FrameStore(max_frames=16,max_bytes=2_000_000,max_age_s=10),
            checkpoint_path=checkpoint,expected_sha256=digest,history_size=4,roi_size_px=32)
