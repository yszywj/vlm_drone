"""Validated, uncertainty-aware target measurements for production vision.

This structure is deliberately limited to sensor-derived geometry.  It has no
field for simulator identities, target truth, velocity, prim paths, motion
seeds, or evaluator frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Sequence

import numpy as np

from common.ids import validate_routing_id


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _vector(
    value: object,
    *,
    field_name: str,
    length: int,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must contain {length} finite numbers")
    if len(value) != length:
        raise ValueError(f"{field_name} must contain {length} finite numbers")
    return tuple(
        _finite(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )


def _covariance(value: object) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("covariance_world_m2 must be a 3-by-3 matrix")
    if len(value) != 3:
        raise ValueError("covariance_world_m2 must be a 3-by-3 matrix")
    rows: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(value):
        normalized = _vector(
            row,
            field_name=f"covariance_world_m2[{row_index}]",
            length=3,
        )
        rows.append((normalized[0], normalized[1], normalized[2]))
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-9):
        raise ValueError("covariance_world_m2 must be symmetric")
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    if minimum_eigenvalue < -1e-9:
        raise ValueError("covariance_world_m2 must be positive semidefinite")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class TargetMeasurement:
    """One synchronized sensor-derived three-dimensional target measurement."""

    timestamp_s: float
    candidate_id: str
    tracker_id: str | None
    pixel_uv: tuple[float, float]
    raw_depth_m: float | None
    corrected_depth_m: float
    position_camera_flu_m: tuple[float, float, float]
    position_world_m: tuple[float, float, float]
    covariance_world_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    measurement_quality: float
    source: str

    def __post_init__(self) -> None:
        timestamp = _finite(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(
            self,
            "candidate_id",
            validate_routing_id(self.candidate_id, "candidate_id"),
        )
        if self.tracker_id is not None:
            object.__setattr__(
                self,
                "tracker_id",
                validate_routing_id(self.tracker_id, "tracker_id"),
            )

        pixel = _vector(self.pixel_uv, field_name="pixel_uv", length=2)
        if pixel[0] < 0.0 or pixel[1] < 0.0:
            raise ValueError("pixel_uv components must be non-negative")
        object.__setattr__(self, "pixel_uv", (pixel[0], pixel[1]))

        if self.raw_depth_m is not None:
            raw_depth = _finite(self.raw_depth_m, "raw_depth_m")
            if raw_depth <= 0.0:
                raise ValueError("raw_depth_m must be greater than zero")
            object.__setattr__(self, "raw_depth_m", raw_depth)
        corrected_depth = _finite(self.corrected_depth_m, "corrected_depth_m")
        if corrected_depth <= 0.0:
            raise ValueError("corrected_depth_m must be greater than zero")
        object.__setattr__(self, "corrected_depth_m", corrected_depth)

        camera = _vector(
            self.position_camera_flu_m,
            field_name="position_camera_flu_m",
            length=3,
        )
        world = _vector(
            self.position_world_m,
            field_name="position_world_m",
            length=3,
        )
        object.__setattr__(
            self,
            "position_camera_flu_m",
            (camera[0], camera[1], camera[2]),
        )
        object.__setattr__(
            self,
            "position_world_m",
            (world[0], world[1], world[2]),
        )
        object.__setattr__(
            self,
            "covariance_world_m2",
            _covariance(self.covariance_world_m2),
        )

        quality = _finite(self.measurement_quality, "measurement_quality")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("measurement_quality must be within [0, 1]")
        object.__setattr__(self, "measurement_quality", quality)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        object.__setattr__(self, "source", self.source.strip())

    @property
    def position_xyz_m(self) -> tuple[float, float, float]:
        """Temporary compatibility alias during resolver migration."""

        return self.position_world_m

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_s": self.timestamp_s,
            "candidate_id": self.candidate_id,
            "tracker_id": self.tracker_id,
            "pixel_uv": list(self.pixel_uv),
            "raw_depth_m": self.raw_depth_m,
            "corrected_depth_m": self.corrected_depth_m,
            "position_camera_flu_m": list(self.position_camera_flu_m),
            "position_world_m": list(self.position_world_m),
            "covariance_world_m2": [
                list(row) for row in self.covariance_world_m2
            ],
            "measurement_quality": self.measurement_quality,
            "source": self.source,
        }


__all__ = ["TargetMeasurement"]
