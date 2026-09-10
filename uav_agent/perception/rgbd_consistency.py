"""Sensor-only, bounded-window RGB-D ambiguity guard (no target truth inputs).

The guard does not pick a far/near object or alter the legacy depth sampler.
Competing surfaces and unsupported seed rays cause abstention. Training and
runtime use this exact implementation, pinned by artifact protocol.
"""
from dataclasses import dataclass
import math
import numpy as np

REFERENCE_GUARD_PROTOCOL = "rgbd_consistency_v1"


@dataclass(frozen=True)
class SensorObservation:
    rgb: np.ndarray
    depth: np.ndarray
    bbox: tuple | None
    intrinsics: tuple
    position: tuple
    orientation: tuple
    timestamp_s: float
    tracker_id: str | None


@dataclass(frozen=True)
class ConsistencyDecision:
    accepted: bool
    reason: str
    seed_fraction: float = 0.0
    competing_surfaces: int = 0


def _chromaticity(pixels):
    rgb = np.asarray(pixels, dtype=np.float64)
    normalized = rgb / np.maximum(rgb.sum(axis=-1, keepdims=True), 1.0)
    return np.median(normalized, axis=0)


def _world(observation, u, v, depth):
    fx, fy, cx, cy = observation.intrinsics
    q = np.asarray(observation.orientation, dtype=np.float64)
    norm = np.linalg.norm(q)
    if (not np.isfinite((fx,fy,cx,cy)).all() or fx <= 0 or fy <= 0
            or not np.isfinite(q).all() or norm <= 1e-12):
        return np.full(3,np.nan)
    vector = np.array([depth, -(u-cx)*depth/fx, -(v-cy)*depth/fy])
    q = q / norm
    return (np.asarray(observation.position) + vector
            + 2*np.cross(q[1:], np.cross(q[1:], vector) + q[0]*vector))


def _inspect(observation, minimum_depth_m, maximum_depth_m):
    rgb, depth, bbox = observation.rgb, observation.depth, observation.bbox
    if bbox is None:
        return ConsistencyDecision(False, "missing_detection"), None
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("RGB-D guard requires aligned HxW depth and HxWx3 RGB")
    height, width = depth.shape
    x1, y1 = int(np.floor(bbox[0]*width)), int(np.floor(bbox[1]*height))
    x2, y2 = int(np.ceil(bbox[2]*width)), int(np.ceil(bbox[3]*height))
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(width,x2), min(height,y2)
    if x2 <= x1 or y2 <= y1:
        return ConsistencyDecision(False, "empty_bbox"), None
    u, v = min(width-1, round((x1+x2-1)/2)), min(height-1, round((y1+y2-1)/2))
    dx, dy = x2-x1, y2-y1
    left, right = x1+round(.1*dx), x2-round(.1*dx)
    top, bottom = y1+round(.1*dy), y2-round(.1*dy)-round(.15*dy)
    if right <= left or bottom <= top:
        return ConsistencyDecision(False, "empty_interior"), None
    # Deterministic bounded cost, independent of model RNG/DataLoader workers.
    stride = max(1, math.ceil(math.sqrt((right-left)*(bottom-top)/4096)))
    patch = depth[top:bottom:stride,left:right:stride]
    colors = rgb[top:bottom:stride,left:right:stride]
    valid = np.isfinite(patch) & (patch >= minimum_depth_m) & (patch <= maximum_depth_m)
    values, colors = patch[valid], colors[valid]
    if len(values) < 3 or len(values)/patch.size < .2:
        return ConsistencyDecision(False, "insufficient_depth"), None
    seed = float(depth[v,u])
    if not np.isfinite(seed) or not minimum_depth_m <= seed <= maximum_depth_m:
        return ConsistencyDecision(False, "invalid_center_seed"), None
    # Separate surfaces by empty depth intervals, never by true target depth.
    order = np.argsort(values)
    gaps = np.diff(values[order])
    threshold = max(.25, .025*float(np.median(values)))
    groups = np.split(order, np.flatnonzero(gaps > threshold)+1)
    seed_group = min(groups, key=lambda g: float(np.min(np.abs(values[g]-seed))))
    fraction = len(seed_group)/len(values)
    significant = [g for g in groups if len(g) >= max(3, .15*len(values))]
    seed_color = _chromaticity(colors[seed_group])
    competitors = [g for g in significant if g is not seed_group and
        abs(float(np.median(values[g]))-seed) > max(.75,.08*seed)]
    if competitors:
        different_color = any(np.linalg.norm(_chromaticity(colors[g])-seed_color) > .12
                              for g in competitors)
        reason = "multiple_depth_rgb_surfaces" if different_color else "multiple_depth_same_color_surfaces"
        return ConsistencyDecision(False, reason, fraction, len(competitors)), None
    if fraction < .5:
        return ConsistencyDecision(False, "minority_seed_surface", fraction), None
    world = _world(observation, u, v, seed)
    if not np.isfinite(world).all():
        return ConsistencyDecision(False, "invalid_camera_geometry"), None
    return ConsistencyDecision(True, "consistent", fraction), (world, seed_color)


def guard_sequence(observations, *, minimum_depth_m=.2, maximum_depth_m=200.0,
                   max_history_age_s=2.0):
    """Rebuild state from this window only; no persistent global/target state."""
    observations = tuple(observations)
    if not observations or len(observations) > 9:
        raise ValueError("RGB-D consistency requires 1 to 9 observations")
    if any(a.timestamp_s >= b.timestamp_s for a,b in zip(observations,observations[1:])):
        raise ValueError("RGB-D history must be strictly time ordered")
    previous = None
    decisions = []
    for observation in observations:
        if not np.isfinite(observation.timestamp_s):
            raise ValueError("non-finite observation timestamp")
        decision, evidence = _inspect(observation, minimum_depth_m, maximum_depth_m)
        if decision.accepted and previous is not None:
            old_observation, (old_world, old_color) = previous
            dt = observation.timestamp_s-old_observation.timestamp_s
            if dt <= 0:
                raise ValueError("RGB-D history must be strictly time ordered")
            if dt <= max_history_age_s and observation.tracker_id == old_observation.tracker_id:
                world, color = evidence
                displacement = float(np.linalg.norm(world-old_world))
                if (displacement > max(2.0,10.0*dt) or
                    displacement > max(.75,5.0*dt) and np.linalg.norm(color-old_color) > .12):
                    decision = ConsistencyDecision(False, "temporal_surface_jump", decision.seed_fraction)
        previous = (observation,evidence) if decision.accepted else None
        decisions.append(decision)
    return tuple(decisions)
