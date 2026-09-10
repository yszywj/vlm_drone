"""Opt-in FrameStore adapter. Never imports the V3 oracle/association module.

Not installed in SEARCH/TRACK by this pilot. Surface-only measurements are
explicit diagnostics; a future residual artifact must declare V3 preprocessing.
"""
import math

from perception.grounding import CandidateResolutionUnavailable
from perception.measurement import TargetMeasurement
from perception.rgbd_consistency import SensorObservation
from perception.depth_geometry import (backproject_pixel_to_camera_optical,
    optical_to_camera_flu)
from target_state_v3.measurement import (MeasurementPolicy, measure_window, geometry,
    world_point, require_v3_artifact, PROTOCOL)


class MultiClusterMeasurementResolver:
    def __init__(self, frame_store, *, policy=MeasurementPolicy(), history_size=6):
        if not 4<=history_size<=8:
            raise ValueError("history_size must be 4..8")
        self.frame_store,self.policy,self.history_size=frame_store,policy,history_size

    def observations(self,candidate):
        refs=candidate.frame_history[-(self.history_size+1):]
        boxes=candidate.bbox_history[-len(refs):]
        trackers=candidate.tracker_id_history[-len(refs):]
        if len(refs)!=self.history_size+1 or len(boxes)!=len(refs) or len(trackers)!=len(refs):
            raise CandidateResolutionUnavailable("V3 requires a full bounded history")
        result=[]
        for ref,box,tracker in zip(refs,boxes,trackers):
            sample=self.frame_store.get_camera_sample(ref)
            if sample is None or sample.depth_to_image_plane_m is None:
                raise CandidateResolutionUnavailable("missing synchronized V3 history")
            k=sample.intrinsics
            result.append(SensorObservation(sample.rgb,sample.depth_to_image_plane_m,box,
                (k.fx,k.fy,k.cx,k.cy),sample.camera_position_world_m,sample.camera_orientation_world_wxyz,
                ref.timestamp_s,tracker))
        return tuple(result)

    def resolve(self,candidate,*,timestamp_s,diagnostic_surface_only=False,
                artifact_manifest=None,prediction=None):
        if not math.isfinite(timestamp_s) or not 0<=timestamp_s-candidate.last_seen_timestamp_s<=self.policy.maximum_history_age_s:
            raise CandidateResolutionUnavailable("stale or invalid candidate timestamp")
        observations=self.observations(candidate)
        decision=measure_window(observations,self.policy)[-1]
        if not decision.accepted:
            raise CandidateResolutionUnavailable(decision.reason)
        s=decision.surface
        obs=observations[-1]
        uv,z=s.uv_px,s.depth_m
        if diagnostic_surface_only:
            if prediction is not None or artifact_manifest is not None:
                raise ValueError("surface diagnostics cannot bypass an artifact contract")
            # Conservative heuristic, not calibrated uncertainty or centre accuracy.
            variance=(max(.5,3*s.sigma_depth_m)**2,)*3
        else:
            require_v3_artifact(artifact_manifest or {},self.policy)
            if prediction is None or set(prediction)!={"delta_uv_px","depth_residual_m","variance_world_m2","validity_probability"}:
                raise ValueError("V3 residual prediction contract mismatch")
            probability=prediction["validity_probability"]
            if not math.isfinite(probability) or not 0<=probability<=1:
                raise ValueError("invalid predicted validity probability")
            if len(prediction["delta_uv_px"])!=2 or any(not math.isfinite(v) for v in prediction["delta_uv_px"]):
                raise ValueError("invalid predicted pixel correction")
            if not math.isfinite(prediction["depth_residual_m"]):
                raise ValueError("invalid predicted depth residual")
            if probability<.5:
                raise CandidateResolutionUnavailable("V3 validity rejected")
            uv=tuple(a+b for a,b in zip(uv,prediction["delta_uv_px"]))
            z+=prediction["depth_residual_m"]
            variance=tuple(prediction["variance_world_m2"])
            if len(variance)!=3 or any(not math.isfinite(v) or v<=0 for v in variance):
                raise ValueError("invalid predicted variance")
        if not self.policy.minimum_depth_m<=z<=self.policy.maximum_depth_m:
            raise CandidateResolutionUnavailable("corrected depth outside range")
        g=geometry(obs)
        optical=backproject_pixel_to_camera_optical(u_px=uv[0],v_px=uv[1],depth_m=z,intrinsics=g.intrinsics)
        return TargetMeasurement(obs.timestamp_s,candidate.candidate_id,obs.tracker_id,uv,s.depth_m,z,
            optical_to_camera_flu(optical),world_point(obs,uv,z),
            tuple(tuple(variance[i] if i==j else 0.0 for j in range(3)) for i in range(3)),
            min(.8,s.fraction),PROTOCOL+("_surface_diagnostic" if diagnostic_surface_only else "_residual"))
