"""Lightweight ROI + geometry + GRU temporal ray-depth residual network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TemporalRayDepthOutput:
    delta_uv_px: Tensor
    depth_residual_m: Tensor
    position_log_variance: Tensor
    measurement_valid_logit: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "delta_uv_px": self.delta_uv_px,
            "depth_residual_m": self.depth_residual_m,
            "position_log_variance": self.position_log_variance,
            "measurement_valid_logit": self.measurement_valid_logit,
        }


class TemporalRayDepthNet(nn.Module):
    """Predict corrections and uncertainty, never a black-box world position."""

    def __init__(
        self,
        *,
        geometry_input_dim: int = 25,
        roi_feature_dim: int = 96,
        geometry_feature_dim: int = 64,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        roi_channels: int = 4,
    ) -> None:
        super().__init__()
        if min(geometry_input_dim, roi_feature_dim, geometry_feature_dim, hidden_dim, gru_layers, roi_channels) <= 0:
            raise ValueError("model dimensions must be positive")
        self.geometry_input_dim = int(geometry_input_dim)
        self.roi_channels = int(roi_channels)
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(roi_channels, 24, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, roi_feature_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(roi_feature_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_input_dim, geometry_feature_dim),
            nn.LayerNorm(geometry_feature_dim),
            nn.SiLU(inplace=True),
            nn.Linear(geometry_feature_dim, geometry_feature_dim),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.GRU(
            roi_feature_dim + geometry_feature_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1 if gru_layers > 1 else 0.0,
        )
        self.delta_uv_head = nn.Linear(hidden_dim, 2)
        self.depth_head = nn.Linear(hidden_dim, 1)
        self.log_variance_head = nn.Linear(hidden_dim, 3)
        self.validity_head = nn.Linear(hidden_dim, 1)

    def forward(self, roi_rgbd: Tensor, geometry: Tensor, missing_mask: Tensor) -> TemporalRayDepthOutput:
        if roi_rgbd.ndim != 5:
            raise ValueError("roi_rgbd must have shape [batch, time, channels, height, width]")
        if geometry.ndim != 3 or missing_mask.ndim != 2:
            raise ValueError("geometry/missing_mask must have shapes [B,T,F] and [B,T]")
        batch, steps, channels, height, width = roi_rgbd.shape
        if geometry.shape[:2] != (batch, steps) or missing_mask.shape != (batch, steps):
            raise ValueError("temporal input batch/time dimensions do not match")
        if channels != self.roi_channels or geometry.shape[-1] != self.geometry_input_dim:
            raise ValueError("temporal input channel/geometry dimensions do not match model")
        if height < 16 or width < 16:
            raise ValueError("ROI dimensions are too small")
        if not torch.isfinite(roi_rgbd).all() or not torch.isfinite(geometry).all():
            raise ValueError("temporal model inputs must be finite")
        missing = missing_mask.to(dtype=torch.bool)
        roi_features = self.roi_encoder(roi_rgbd.reshape(batch * steps, channels, height, width))
        roi_features = roi_features.reshape(batch, steps, -1)
        geometry_features = self.geometry_encoder(geometry)
        valid = (~missing).unsqueeze(-1).to(dtype=roi_features.dtype)
        fused = torch.cat((roi_features * valid, geometry_features * valid), dim=-1)
        temporal, _ = self.temporal(fused)
        step_indices = torch.arange(steps, device=temporal.device).unsqueeze(0).expand(batch, -1)
        valid_indices = step_indices.masked_fill(missing, -1).max(dim=1).values.clamp(min=0)
        state = temporal[torch.arange(batch, device=temporal.device), valid_indices]
        return TemporalRayDepthOutput(
            delta_uv_px=self.delta_uv_head(state),
            depth_residual_m=self.depth_head(state).squeeze(-1),
            position_log_variance=self.log_variance_head(state).clamp(-12.0, 8.0),
            measurement_valid_logit=self.validity_head(state).squeeze(-1),
        )


__all__ = ["TemporalRayDepthNet", "TemporalRayDepthOutput"]
