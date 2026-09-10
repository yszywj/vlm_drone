"""Tensor adapter for the production sensor-only ray acceptance contract."""
import torch

from perception.ray_measurement_gate import ray_measurement_gate


def gate_batch(batch, *, corrected_depth_m, delta_uv_px, validity_probability,
               maximum_depth_m):
    # Intentionally do not use measurement_valid/visible/target_present:
    # those are privileged training labels and must not gate predictions.
    return ray_measurement_gate(
        reference_detected=(~batch["missing_mask"][:, -1].bool()
                            & batch.get("reference_sensor_consistent", True)),
        raw_depth_m=batch["raw_depth_m"],
        anchor_uv_px=batch["anchor_uv_px"].unbind(-1),
        corrected_uv_px=(batch["anchor_uv_px"] + delta_uv_px).unbind(-1),
        corrected_depth_m=corrected_depth_m,
        image_size_wh=batch["image_size_wh"].unbind(-1),
        validity_probability=validity_probability,
        minimum_depth_m=batch["depth_range_m"][:, 0],
        maximum_depth_m=batch["depth_range_m"][:, 1].clamp_max(maximum_depth_m),
        finite=torch.isfinite,
    )
