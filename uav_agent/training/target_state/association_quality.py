"""OFFLINE label review only. Never import truth-based gates into perception.

Depth agreement is necessary evidence, not proof of object identity. Occluded,
sparse-depth and competing associations are retained for review, not negatives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

ASSOCIATION_PROTOCOL = "offline_depth_association_review_v1"


@dataclass(frozen=True)
class AssociationPolicy:
    # Fixed conservative screening defaults, not selected using test errors.
    minimum_pixels: int = 3
    minimum_valid_fraction: float = 0.2
    minimum_support_fraction: float = 0.5
    surface_tolerance_m: float = 0.15
    surface_tolerance_fraction: float = 0.02
    # Old archives have centres but no object dimensions/corners. This wide
    # interval is a screening prior, NOT recovered object geometry.
    legacy_half_extent_m: float = 1.0
    legacy_half_extent_fraction: float = 0.1
    sampled_minimum_depth_m: float = 0.2
    sampled_maximum_depth_m: float = 200.0

    def to_dict(self):
        return asdict(self)


def depth_support(depth, bbox, near_m, far_m, *, policy=AssociationPolicy()):
    """Check the detection's interior, not the entire oracle projection box."""
    array = np.asarray(depth)
    if array.ndim != 2:
        raise ValueError("association depth must be a 2-D optical-z array")
    height, width = array.shape
    if not (math.isfinite(near_m) and math.isfinite(far_m) and 0 < near_m <= far_m):
        return {"status": "invalid_target_depth", "supported": False}
    x1, y1, x2, y2 = bbox
    left, right = max(0, int(math.floor(x1 * width))), min(width, int(math.ceil(x2 * width)))
    top, bottom = max(0, int(math.floor(y1 * height))), min(height, int(math.ceil(y2 * height)))
    dx, dy = right - left, bottom - top
    # Match the general interior used by the RGB-D sampler; do not require
    # the projected target centre to be visible during partial occlusion.
    left += int(0.1 * dx)
    right -= int(0.1 * dx)
    top += int(0.1 * dy)
    bottom -= int(0.15 * dy)
    patch = array[top:bottom, left:right] if right > left and bottom > top else np.array([])
    valid = patch[np.isfinite(patch) & (patch > 0)]
    valid_fraction = float(valid.size / patch.size) if patch.size else 0.0
    tolerance = max(policy.surface_tolerance_m, policy.surface_tolerance_fraction * (near_m + far_m) / 2)
    lower, upper = max(0.0, near_m - tolerance), far_m + tolerance
    count = int(np.count_nonzero((valid >= lower) & (valid <= upper)))
    support = float(count / valid.size) if valid.size else 0.0
    foreground = float(np.count_nonzero(valid < lower) / valid.size) if valid.size else 0.0
    if valid.size < policy.minimum_pixels or valid_fraction < policy.minimum_valid_fraction:
        status = "insufficient_depth"
    elif count >= policy.minimum_pixels and support >= policy.minimum_support_fraction:
        status = "depth_consistent"
    elif foreground >= policy.minimum_support_fraction:
        status = "foreground_conflict_or_occlusion"
    else:
        status = "insufficient_target_surface"
    return {"status": status, "supported": status == "depth_consistent",
            "depth_interval_m": [lower, upper], "valid_pixels": int(valid.size),
            "valid_fraction": valid_fraction, "support_pixels": count,
            "support_fraction": support, "foreground_fraction": foreground,
            "median_depth_m": float(np.median(valid)) if valid.size else None}
