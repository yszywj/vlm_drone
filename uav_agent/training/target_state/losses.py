"""Masked objectives for temporal ray-depth residual training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from training.target_state.config import LossWeights
from training.target_state.geometry import corrected_ray_to_world, project_world_to_pixel
from training.target_state.model import TemporalRayDepthOutput


@dataclass(frozen=True)
class TargetStateLossResult:
    total: Tensor
    depth_huber: Tensor
    position_3d_huber: Tensor
    reprojection_huber: Tensor
    gaussian_nll: Tensor
    validity_bce: Tensor
    predicted_position_world_m: Tensor
    corrected_depth_m: Tensor
    ray_valid_mask: Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "loss/total": float(self.total.detach().cpu()),
            "loss/depth_huber": float(self.depth_huber.detach().cpu()),
            "loss/position_3d_huber": float(self.position_3d_huber.detach().cpu()),
            "loss/reprojection_huber": float(self.reprojection_huber.detach().cpu()),
            "loss/gaussian_nll": float(self.gaussian_nll.detach().cpu()),
            "loss/validity_bce": float(self.validity_bce.detach().cpu()),
        }


def _masked_mean(
    values: Tensor,
    mask: Tensor,
    *,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Return a finite mean over a boolean mask and optional soft weights.

    ``sample_weight`` is deliberately separate from ``mask``: detector misses
    and invalid depth are hard exclusions, while partially occluded examples
    contribute in proportion to their visible fraction.  Expanding both here
    also keeps vector-valued position/reprojection losses normalized per
    coordinate exactly as the original boolean-only implementation did.
    """

    expanded = mask.to(dtype=values.dtype)
    if sample_weight is not None:
        weight = sample_weight.to(device=values.device, dtype=values.dtype)
        while weight.ndim < values.ndim:
            weight = weight.unsqueeze(-1)
        weight = weight.expand_as(values)
        # The dataset schema already restricts weights to [0, 1], but retain a
        # fail-safe at the numeric boundary so a corrupt batch cannot turn one
        # training step into NaN/Inf.
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        while expanded.ndim < values.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(values) * weight
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    finite_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    safe = torch.where(
        expanded > 0.0,
        finite_values * expanded,
        torch.zeros_like(values),
    )
    return safe.sum() / expanded.sum().clamp_min(1.0)


def _occlusion_visibility_weight(occlusion_ratio: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Convert privileged occlusion labels to stable visible-fraction weights.

    Invalid labels are treated as fully occluded (zero regression weight), not
    as clean supervision.  Valid schema-produced values map exactly as
    ``0 -> 1``, ``partial -> 1-partial``, and ``1 -> 0``.
    """

    occlusion = occlusion_ratio.to(dtype=dtype)
    occlusion = torch.nan_to_num(occlusion, nan=1.0, posinf=1.0, neginf=1.0)
    return (1.0 - occlusion.clamp(0.0, 1.0)).clamp(0.0, 1.0)


def compute_target_state_losses(
    output: TemporalRayDepthOutput,
    batch: dict[str, Tensor],
    *,
    weights: LossWeights = LossWeights(),
) -> TargetStateLossResult:
    """Compute all required losses from deployable inputs and offline labels.

    Label tensors are supplied by the training caller.  This module is not
    imported by the production perception runtime.
    """

    required = {
        "anchor_uv_px", "raw_depth_m", "intrinsics_fx_fy_cx_cy",
        "camera_position_world_m", "camera_orientation_world_wxyz",
        "target_position_world_m", "target_depth_m", "measurement_valid",
        "target_present_mask", "label_valid_mask", "valid_depth_mask",
        "occlusion_ratio", "history_occlusion_ratio", "history_label_valid_mask",
        "history_intrinsics_fx_fy_cx_cy",
        "history_camera_position_world_m", "history_camera_orientation_world_wxyz",
        "history_center_uv_px", "history_visible_mask",
    }
    missing = required - set(batch)
    if missing:
        raise KeyError(f"target-state loss batch is missing fields: {sorted(missing)}")
    predicted_world, corrected_depth, ray_valid = corrected_ray_to_world(
        anchor_uv_px=batch["anchor_uv_px"],
        raw_depth_m=batch["raw_depth_m"],
        delta_uv_px=output.delta_uv_px,
        depth_residual_m=output.depth_residual_m,
        intrinsics_fx_fy_cx_cy=batch["intrinsics_fx_fy_cx_cy"],
        camera_position_world_m=batch["camera_position_world_m"],
        camera_orientation_world_wxyz=batch["camera_orientation_world_wxyz"],
    )
    label_valid = batch["label_valid_mask"].to(dtype=torch.bool)
    target_present = batch["target_present_mask"].to(dtype=torch.bool)
    if not torch.equal(label_valid, target_present):
        raise ValueError("target_present_mask and label_valid_mask disagree")
    measurement_valid = batch["measurement_valid"].to(dtype=torch.bool)
    if torch.any(measurement_valid & ~label_valid):
        raise ValueError("no-target samples cannot be marked measurement_valid")
    depth_mask = (
        measurement_valid
        & label_valid
        & batch["valid_depth_mask"].to(dtype=torch.bool)
        & ray_valid
    )
    occlusion_weight = _occlusion_visibility_weight(
        batch["occlusion_ratio"], dtype=corrected_depth.dtype
    )
    finite_target = torch.isfinite(batch["target_position_world_m"]).all(dim=-1)
    position_mask = depth_mask & finite_target
    depth_values = F.smooth_l1_loss(corrected_depth, batch["target_depth_m"], reduction="none")
    depth_loss = _masked_mean(
        depth_values,
        depth_mask,
        sample_weight=occlusion_weight,
    )
    position_values = F.smooth_l1_loss(
        predicted_world,
        batch["target_position_world_m"],
        reduction="none",
    )
    position_loss = _masked_mean(
        position_values,
        position_mask,
        sample_weight=occlusion_weight,
    )

    variance_log = output.position_log_variance.clamp(-12.0, 8.0)
    squared_error = (predicted_world - batch["target_position_world_m"]).square()
    nll_values = 0.5 * (torch.exp(-variance_log) * squared_error + variance_log)
    nll_loss = _masked_mean(
        nll_values,
        position_mask,
        sample_weight=occlusion_weight,
    )

    history_position = predicted_world.unsqueeze(1).expand_as(batch["history_camera_position_world_m"])
    projected_uv, _, projection_valid = project_world_to_pixel(
        position_world_m=history_position,
        intrinsics_fx_fy_cx_cy=batch["history_intrinsics_fx_fy_cx_cy"],
        camera_position_world_m=batch["history_camera_position_world_m"],
        camera_orientation_world_wxyz=batch["history_camera_orientation_world_wxyz"],
    )
    reprojection_mask = (
        position_mask.unsqueeze(1)
        & projection_valid
        & batch["history_label_valid_mask"].to(dtype=torch.bool)
        & batch["history_visible_mask"].to(dtype=torch.bool)
        & torch.isfinite(batch["history_center_uv_px"]).all(dim=-1)
    )
    history_occlusion_weight = _occlusion_visibility_weight(
        batch["history_occlusion_ratio"], dtype=projected_uv.dtype
    )
    reprojection_weight = occlusion_weight.unsqueeze(1) * history_occlusion_weight
    reprojection_values = F.smooth_l1_loss(
        projected_uv,
        batch["history_center_uv_px"],
        reduction="none",
    )
    # Normalize pixels by focal length so camera resolution does not dominate.
    focal_scale = batch["history_intrinsics_fx_fy_cx_cy"][..., :2].clamp_min(1.0)
    reprojection_loss = _masked_mean(
        reprojection_values / focal_scale,
        reprojection_mask,
        sample_weight=reprojection_weight,
    )

    validity_loss = F.binary_cross_entropy_with_logits(
        output.measurement_valid_logit,
        measurement_valid.to(dtype=output.measurement_valid_logit.dtype),
    )
    total = (
        float(weights.depth) * depth_loss
        + float(weights.position_3d) * position_loss
        + float(weights.reprojection) * reprojection_loss
        + float(weights.gaussian_nll) * nll_loss
        + float(weights.validity_bce) * validity_loss
    )
    return TargetStateLossResult(
        total=total,
        depth_huber=depth_loss,
        position_3d_huber=position_loss,
        reprojection_huber=reprojection_loss,
        gaussian_nll=nll_loss,
        validity_bce=validity_loss,
        predicted_position_world_m=predicted_world,
        corrected_depth_m=corrected_depth,
        ray_valid_mask=ray_valid,
    )


__all__ = ["TargetStateLossResult", "compute_target_state_losses"]
