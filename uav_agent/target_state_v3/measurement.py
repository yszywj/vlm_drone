"""Versioned sensor-only measurement layer shared by training and runtime.

This estimates an observed SURFACE, not object identity or its geometric centre.
All policy defaults are engineering hypotheses, not thresholds fitted on TEST.
No masks, object IDs, target poses, assignments or evaluator objects are inputs.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import numpy as np

from env.camera_types import CameraIntrinsics
from perception.depth_geometry import (backproject_pixel_to_camera_optical,
    optical_to_camera_flu, camera_flu_to_world)
from perception.rgbd_consistency import SensorObservation
from runtime.frame_store import FrameCameraGeometry

PROTOCOL = "multicluster_temporal_surface_v3"


@dataclass(frozen=True)
class MeasurementPolicy:
    minimum_depth_m: float = 0.2
    maximum_depth_m: float = 200.0
    minimum_pixels: int = 6
    minimum_valid_fraction: float = 0.2
    minimum_cluster_fraction: float = 0.08
    single_surface_fraction: float = 0.8
    depth_gap_m: float = 0.15
    depth_gap_fraction: float = 0.02
    maximum_clusters: int = 6
    maximum_history_age_s: float = 2.0
    motion_slack_m: float = 0.4
    maximum_surface_speed_mps: float = 5.0
    maximum_color_distance: float = 0.12

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid measurement policy: {key}")
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("invalid depth range")
        if any(not 0 < getattr(self, f) <= 1 for f in
               ("minimum_valid_fraction", "minimum_cluster_fraction", "single_surface_fraction")):
            raise ValueError("invalid fraction")
        if type(self.minimum_pixels) is not int or type(self.maximum_clusters) is not int:
            raise ValueError("pixel/cluster limits must be integers")

    def contract(self):
        return {"protocol": PROTOCOL, "policy": asdict(self)}

    @property
    def digest(self):
        return sha256(json.dumps(self.contract(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Surface:
    depth_m: float
    uv_px: tuple
    fraction: float
    sigma_depth_m: float
    color: tuple
    world_m: tuple


@dataclass(frozen=True)
class MeasurementDecision:
    accepted: bool
    reason: str
    surfaces: tuple = ()
    selected: int | None = None

    @property
    def surface(self):
        return None if self.selected is None else self.surfaces[self.selected]


def geometry(observation):
    h, w = observation.depth.shape
    fx, fy, cx, cy = observation.intrinsics
    return FrameCameraGeometry(observation.timestamp_s, CameraIntrinsics(fx, fy, cx, cy, w, h),
                               tuple(observation.position), tuple(observation.orientation))


def world_point(observation, uv, depth):
    g = geometry(observation)
    optical = backproject_pixel_to_camera_optical(u_px=uv[0], v_px=uv[1],
                                                  depth_m=depth, intrinsics=g.intrinsics)
    return camera_flu_to_world(optical_to_camera_flu(optical), g)


def extract_surfaces(observation, policy=MeasurementPolicy()):
    if not isinstance(observation, SensorObservation):
        raise TypeError("only SensorObservation is accepted")
    rgb, depth = np.asarray(observation.rgb), np.asarray(observation.depth)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError("RGB-D shape/dtype mismatch")
    if not math.isfinite(observation.timestamp_s) or observation.timestamp_s < 0:
        raise ValueError("invalid sensor timestamp")
    # Validate calibration/pose even if the detector is missing.
    world_point(observation, (0.0, 0.0), policy.minimum_depth_m)
    if observation.bbox is None:
        return MeasurementDecision(False, "detector_miss")
    box = np.asarray(observation.bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > 1):
        raise ValueError("invalid normalized bbox")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("empty bbox")
    h, w = depth.shape
    x1, y1 = np.floor(box[:2] * (w, h)).astype(int)
    x2, y2 = np.ceil(box[2:] * (w, h)).astype(int)
    dx, dy = x2-x1, y2-y1
    # Retain the legacy ground-exclusion intent, but never seed at bbox centre.
    x1, x2 = x1 + int(.1*dx), x2-int(.1*dx)
    y1, y2 = y1 + int(.1*dy), y2-int(.25*dy)
    roi = depth[y1:y2, x1:x2]
    valid = np.isfinite(roi) & (roi >= policy.minimum_depth_m) & (roi <= policy.maximum_depth_m)
    yy, xx = np.nonzero(valid)
    if len(xx) < policy.minimum_pixels or len(xx) / max(1, roi.size) < policy.minimum_valid_fraction:
        return MeasurementDecision(False, "insufficient_depth")
    z = roi[yy, xx].astype(float)
    order = np.argsort(z, kind="stable")
    sorted_z = z[order]
    breaks = np.flatnonzero(np.diff(sorted_z) > np.maximum(
        policy.depth_gap_m, policy.depth_gap_fraction*sorted_z[:-1])) + 1
    groups = np.split(order, breaks)
    surfaces = []
    for group in groups:
        fraction = len(group)/len(z)
        if len(group) < policy.minimum_pixels or fraction < policy.minimum_cluster_fraction:
            continue
        xs, ys, zs = xx[group]+x1, yy[group]+y1, z[group]
        median_z = float(np.median(zs))
        # Choose an ACTUAL cluster pixel near its spatial/depth medians. The
        # independent coordinate medians might otherwise land on an occluder.
        distance = ((xs-np.median(xs))/max(1,dx))**2 + ((ys-np.median(ys))/max(1,dy))**2
        distance += ((zs-median_z)/max(.15,.02*median_z))**2
        chosen = int(np.argmin(distance))
        uv = (float(xs[chosen]), float(ys[chosen]))
        depth_z = float(zs[chosen])
        pixels = rgb[ys, xs].astype(float)
        chroma = np.median(pixels/np.maximum(pixels.sum(axis=1, keepdims=True), 1), axis=0)
        surfaces.append(Surface(depth_z, uv, fraction,
            max(.02, 1.4826*float(np.median(abs(zs-median_z)))),
            tuple(chroma), world_point(observation, uv, depth_z)))
    if not surfaces or len(surfaces) > policy.maximum_clusters:
        return MeasurementDecision(False, "no_bounded_surface_hypothesis", tuple(surfaces))
    return MeasurementDecision(False, "unselected", tuple(surfaces))


def measure_window(observations, policy=MeasurementPolicy()):
    """Bounded, ordered window. Only a previously unambiguous surface can seed tracking."""
    observations = tuple(observations)
    if not 1 <= len(observations) <= 9:
        raise ValueError("measurement requires 1..9 sensor observations")
    if any(b.timestamp_s <= a.timestamp_s for a,b in zip(observations, observations[1:])):
        raise ValueError("timestamps must be strictly increasing")
    if observations[-1].timestamp_s-observations[0].timestamp_s > policy.maximum_history_age_s+1e-9:
        raise ValueError("window exceeds history age")
    previous = None
    decisions = []
    for observation in observations:
        result = extract_surfaces(observation, policy)
        surfaces = result.surfaces
        if result.reason == "unselected":
            compatible = []
            if previous is not None:
                old, surface = previous
                dt = observation.timestamp_s-old.timestamp_s
                if observation.tracker_id is not None and observation.tracker_id == old.tracker_id:
                    compatible = [i for i,s in enumerate(surfaces)
                        if np.linalg.norm(np.subtract(s.world_m, surface.world_m)) <=
                        policy.motion_slack_m+policy.maximum_surface_speed_mps*dt
                        and np.linalg.norm(np.subtract(s.color, surface.color)) <= policy.maximum_color_distance]
                    # A failed continuation is not silently reinitialized on another object.
                    result = MeasurementDecision(len(compatible)==1,
                        "temporal_surface_match" if len(compatible)==1 else "ambiguous_or_discontinuous_surface",
                        surfaces, compatible[0] if len(compatible)==1 else None)
                else:
                    result = MeasurementDecision(False, "tracker_discontinuity", surfaces)
            elif len(surfaces)==1 and surfaces[0].fraction >= policy.single_surface_fraction:
                result = MeasurementDecision(True, "single_surface", surfaces, 0)
            else:
                result = MeasurementDecision(False, "ambiguous_depth_surfaces", surfaces)
        decisions.append(result)
        previous = (observation,result.surface) if result.accepted else None
    return tuple(decisions)


def require_v3_artifact(manifest, policy=MeasurementPolicy()):
    if manifest.get("measurement_preprocessing") != policy.contract():
        raise ValueError("artifact preprocessing mismatch; legacy best.pt cannot use V3 sampling")
    if manifest.get("production_approved") is not True:
        raise ValueError("V3 artifact has not passed runtime/closed-loop promotion")
