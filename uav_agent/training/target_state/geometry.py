"""Differentiable, convention-explicit ray projection used by training.

The network supplies only pixel/depth corrections.  These functions retain the
auditable optical-camera -> FLU-camera -> world transform used in production.
"""

from __future__ import annotations

import torch
from torch import Tensor


CAMERA_CONVENTION = "camera_optical_x_right_y_down_z_forward"
WORLD_CONVENTION = "world_flu_x_forward_y_left_z_up"


def _normalize_quaternion_wxyz(quaternion: Tensor) -> Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in four wxyz components")
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)


def quaternion_rotate_wxyz(quaternion: Tensor, vector: Tensor) -> Tensor:
    """Rotate vector by an active unit quaternion, supporting batch dimensions."""

    q = _normalize_quaternion_wxyz(quaternion)
    scalar = q[..., :1]
    axis = q[..., 1:]
    return vector + 2.0 * torch.cross(axis, torch.cross(axis, vector, dim=-1) + scalar * vector, dim=-1)


def quaternion_inverse_rotate_wxyz(quaternion: Tensor, vector: Tensor) -> Tensor:
    q = _normalize_quaternion_wxyz(quaternion)
    inverse = torch.cat((q[..., :1], -q[..., 1:]), dim=-1)
    return quaternion_rotate_wxyz(inverse, vector)


def corrected_ray_to_world(
    *,
    anchor_uv_px: Tensor,
    raw_depth_m: Tensor,
    delta_uv_px: Tensor,
    depth_residual_m: Tensor,
    intrinsics_fx_fy_cx_cy: Tensor,
    camera_position_world_m: Tensor,
    camera_orientation_world_wxyz: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return world point, corrected optical-z depth, and a strict valid mask."""

    if anchor_uv_px.shape[-1] != 2 or delta_uv_px.shape[-1] != 2:
        raise ValueError("anchor_uv_px and delta_uv_px must end in two components")
    if intrinsics_fx_fy_cx_cy.shape[-1] != 4:
        raise ValueError("intrinsics must be [fx, fy, cx, cy]")
    if camera_position_world_m.shape[-1] != 3:
        raise ValueError("camera position must end in three components")
    corrected_uv = anchor_uv_px + delta_uv_px
    corrected_depth = raw_depth_m + depth_residual_m
    fx, fy, cx, cy = intrinsics_fx_fy_cx_cy.unbind(dim=-1)
    finite = (
        torch.isfinite(corrected_uv).all(dim=-1)
        & torch.isfinite(corrected_depth)
        & torch.isfinite(intrinsics_fx_fy_cx_cy).all(dim=-1)
        & torch.isfinite(camera_position_world_m).all(dim=-1)
        & torch.isfinite(camera_orientation_world_wxyz).all(dim=-1)
    )
    valid = finite & (corrected_depth > 0.0) & (fx > 0.0) & (fy > 0.0)
    safe_depth = corrected_depth.clamp_min(1e-6)
    optical = torch.stack(
        (
            (corrected_uv[..., 0] - cx) / fx.clamp_min(1e-6) * safe_depth,
            (corrected_uv[..., 1] - cy) / fy.clamp_min(1e-6) * safe_depth,
            safe_depth,
        ),
        dim=-1,
    )
    camera_flu = torch.stack((optical[..., 2], -optical[..., 0], -optical[..., 1]), dim=-1)
    world = camera_position_world_m + quaternion_rotate_wxyz(
        camera_orientation_world_wxyz,
        camera_flu,
    )
    return world, corrected_depth, valid


def project_world_to_pixel(
    *,
    position_world_m: Tensor,
    intrinsics_fx_fy_cx_cy: Tensor,
    camera_position_world_m: Tensor,
    camera_orientation_world_wxyz: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project world points to pixels; returns uv, optical-z depth, valid mask."""

    relative_world = position_world_m - camera_position_world_m
    camera_flu = quaternion_inverse_rotate_wxyz(camera_orientation_world_wxyz, relative_world)
    optical = torch.stack((-camera_flu[..., 1], -camera_flu[..., 2], camera_flu[..., 0]), dim=-1)
    depth = optical[..., 2]
    fx, fy, cx, cy = intrinsics_fx_fy_cx_cy.unbind(dim=-1)
    safe_depth = depth.clamp_min(1e-6)
    uv = torch.stack(
        (fx * optical[..., 0] / safe_depth + cx, fy * optical[..., 1] / safe_depth + cy),
        dim=-1,
    )
    valid = (
        torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(depth)
        & (depth > 0.0)
        & (fx > 0.0)
        & (fy > 0.0)
    )
    return uv, depth, valid


def diagonal_covariance(position_log_variance: Tensor) -> Tensor:
    if position_log_variance.shape[-1] != 3:
        raise ValueError("position_log_variance must end in three components")
    variance = torch.exp(position_log_variance.clamp(-12.0, 8.0)).clamp(1e-6, 1e4)
    return torch.diag_embed(variance)


__all__ = [
    "CAMERA_CONVENTION",
    "WORLD_CONVENTION",
    "corrected_ray_to_world",
    "diagonal_covariance",
    "project_world_to_pixel",
    "quaternion_inverse_rotate_wxyz",
    "quaternion_rotate_wxyz",
]
