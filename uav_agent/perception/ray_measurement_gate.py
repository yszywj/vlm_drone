"""Sensor-only acceptance contract shared by scalar runtime and tensor eval.

No labels, simulator state, torch import, or estimator/evaluator dependency.
Arguments may be scalars or tensors; tensor callers supply their isfinite.
"""
from math import isfinite
from typing import NamedTuple

MEASUREMENT_PROTOCOL = "reference_rgbd_gate_v2"


class RayGateResult(NamedTuple):
    input_valid: object
    geometry_valid: object
    accepted: object


def reference_depth_valid(*, reference_detected, raw_depth_m, minimum_depth_m,
                          maximum_depth_m, finite=isfinite):
    """Reference availability subset, also usable on older saved audit rows."""
    return (reference_detected & finite(raw_depth_m) & finite(minimum_depth_m)
            & finite(maximum_depth_m) & (minimum_depth_m > 0)
            & (maximum_depth_m > minimum_depth_m)
            & (raw_depth_m >= minimum_depth_m) & (raw_depth_m <= maximum_depth_m))


def ray_measurement_gate(*, reference_detected, raw_depth_m, anchor_uv_px,
                         corrected_uv_px, corrected_depth_m, image_size_wh,
                         validity_probability, minimum_depth_m, maximum_depth_m,
                         finite=isfinite):
    """A residual cannot create a new measurement from a missing reference.

    This gate controls *learned measurements*, not Kalman predictions or a
    separately identified fallback used when temporal history is unavailable.
    The probability threshold stays at 0.5; no truth-based gate is allowed.
    """
    width, height = image_size_wh
    u, v = anchor_uv_px
    corrected_u, corrected_v = corrected_uv_px
    image_valid = finite(width) & finite(height) & (width > 0) & (height > 0)
    depth_range_valid = (finite(minimum_depth_m) & finite(maximum_depth_m)
                         & (minimum_depth_m > 0) & (maximum_depth_m > minimum_depth_m))
    input_valid = (reference_depth_valid(
                   reference_detected=reference_detected, raw_depth_m=raw_depth_m,
                   minimum_depth_m=minimum_depth_m, maximum_depth_m=maximum_depth_m, finite=finite)
                   & image_valid
                   & finite(u) & finite(v) & (u >= 0) & (v >= 0)
                   & (u < width) & (v < height))
    geometry_valid = (image_valid & depth_range_valid & finite(corrected_depth_m)
                      & (corrected_depth_m >= minimum_depth_m)
                      & (corrected_depth_m <= maximum_depth_m)
                      & finite(corrected_u) & finite(corrected_v)
                      & (corrected_u >= 0) & (corrected_v >= 0)
                      & (corrected_u < width) & (corrected_v < height))
    accepted = (input_valid & geometry_valid & finite(validity_probability)
                & (validity_probability >= 0.5) & (validity_probability <= 1.0))
    return RayGateResult(input_valid, geometry_valid, accepted)
