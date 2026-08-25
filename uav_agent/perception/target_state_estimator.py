"""Constant-velocity Kalman filter for metric three-dimensional target state.

This state is intentionally separate from BoT-SORT's image-plane Kalman
state.  Only trusted world-position measurements enter this estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real

import numpy as np

from perception.measurement import TargetMeasurement


class TargetStateMeasurementRejected(RuntimeError):
    """Raised when innovation gating rejects a discontinuous measurement."""


class TargetStateUpdateOutcome(str, Enum):
    """Most recent externally visible estimator outcome."""

    MEASUREMENT_ACCEPTED = "measurement_accepted"
    MEASUREMENT_REJECTED = "measurement_rejected"
    PREDICTED_ONLY = "predicted_only"


@dataclass(frozen=True, slots=True)
class TargetStateEstimatorStatistics:
    measurements_accepted: int
    measurements_rejected: int
    predicted_only_outputs: int


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _position(value: object) -> np.ndarray:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("position_world_m must be a three-number tuple")
    return np.asarray(
        [
            _finite(component, f"position_world_m[{index}]")
            for index, component in enumerate(value)
        ],
        dtype=np.float64,
    )


def _covariance_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


@dataclass(frozen=True, slots=True)
class FilteredTargetState:
    """Small immutable snapshot returned by measurement or prediction."""

    timestamp_s: float
    position_world_m: tuple[float, float, float]
    velocity_world_mps: tuple[float, float, float]
    measurement_age_s: float
    predicted_only: bool
    covariance_6x6: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        timestamp = _finite(self.timestamp_s, "timestamp_s")
        age = _finite(self.measurement_age_s, "measurement_age_s")
        if timestamp < 0.0 or age < 0.0:
            raise ValueError("timestamps and measurement age must be non-negative")
        position = _position(self.position_world_m)
        velocity = _position(self.velocity_world_mps)
        if not isinstance(self.predicted_only, bool):
            raise TypeError("predicted_only must be a bool")
        covariance = self.covariance_6x6
        if (
            not isinstance(covariance, tuple)
            or len(covariance) != 6
            or any(not isinstance(row, tuple) or len(row) != 6 for row in covariance)
        ):
            raise TypeError("covariance_6x6 must be a 6-by-6 tuple")
        normalized_covariance = tuple(
            tuple(
                _finite(value, f"covariance_6x6[{row_index}][{column_index}]")
                for column_index, value in enumerate(row)
            )
            for row_index, row in enumerate(covariance)
        )
        covariance_array = np.asarray(normalized_covariance, dtype=np.float64)
        if not np.allclose(covariance_array, covariance_array.T, atol=1e-9):
            raise ValueError("covariance_6x6 must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance_array))) < -1e-8:
            raise ValueError("covariance_6x6 must be positive semidefinite")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "measurement_age_s", age)
        object.__setattr__(
            self,
            "position_world_m",
            tuple(float(value) for value in position),
        )
        object.__setattr__(
            self,
            "velocity_world_mps",
            tuple(float(value) for value in velocity),
        )
        object.__setattr__(self, "covariance_6x6", normalized_covariance)

    @property
    def covariance(self) -> tuple[tuple[float, ...], ...]:
        """Compatibility alias for consumers that do not encode dimensions."""

        return self.covariance_6x6

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_s": self.timestamp_s,
            "position_world_m": list(self.position_world_m),
            "velocity_world_mps": list(self.velocity_world_mps),
            "measurement_age_s": self.measurement_age_s,
            "predicted_only": self.predicted_only,
            "covariance_6x6": [list(row) for row in self.covariance_6x6],
        }


class TargetStateEstimator:
    """Timestamp-driven CV Kalman filter with bounded lost-target prediction."""

    _H = np.concatenate(
        [np.eye(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64)],
        axis=1,
    )

    def __init__(
        self,
        *,
        max_prediction_age_s: float = 2.0,
        max_position_jump_m: float = 10.0,
        process_noise: float = 1.0,
        measurement_noise: float = 0.5,
        innovation_gate_sigma: float = 5.0,
        initial_velocity_variance: float = 25.0,
    ) -> None:
        max_age = _finite(max_prediction_age_s, "max_prediction_age_s")
        max_jump = _finite(max_position_jump_m, "max_position_jump_m")
        process = _finite(process_noise, "process_noise")
        measurement = _finite(measurement_noise, "measurement_noise")
        gate = _finite(innovation_gate_sigma, "innovation_gate_sigma")
        initial_velocity = _finite(
            initial_velocity_variance,
            "initial_velocity_variance",
        )
        if any(
            value <= 0.0
            for value in (max_age, max_jump, process, measurement, gate, initial_velocity)
        ):
            raise ValueError("estimator limits and noise values must be greater than zero")
        self._max_prediction_age_s = max_age
        self._max_position_jump_m = max_jump
        self._process_noise = process
        self._measurement_noise = measurement
        self._innovation_gate_sigma = gate
        self._initial_velocity_variance = initial_velocity
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._timestamp_s: float | None = None
        self._measurement_timestamp_s: float | None = None
        self._measurements_accepted = 0
        self._measurements_rejected = 0
        self._predicted_only_outputs = 0
        self._last_outcome: TargetStateUpdateOutcome | None = None

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    @property
    def max_prediction_age_s(self) -> float:
        return self._max_prediction_age_s

    @property
    def last_outcome(self) -> TargetStateUpdateOutcome | None:
        return self._last_outcome

    @property
    def statistics(self) -> TargetStateEstimatorStatistics:
        return TargetStateEstimatorStatistics(
            measurements_accepted=self._measurements_accepted,
            measurements_rejected=self._measurements_rejected,
            predicted_only_outputs=self._predicted_only_outputs,
        )

    def reset(self) -> None:
        """Discard state on task/target/tracker identity changes."""

        self._state = None
        self._covariance = None
        self._timestamp_s = None
        self._measurement_timestamp_s = None
        self._last_outcome = None

    def update(
        self,
        measurement: TargetMeasurement | None = None,
        *,
        timestamp_s: float | None = None,
        position_world_m: tuple[float, float, float] | None = None,
        confidence: float | None = None,
    ) -> FilteredTargetState:
        """Accept a typed measurement or the temporary legacy keyword form.

        The typed form consumes its full world-frame covariance.  The legacy
        form remains available for callers not yet migrated and derives the
        old isotropic covariance from ``measurement_noise / confidence``.
        """

        if measurement is not None:
            if not isinstance(measurement, TargetMeasurement):
                raise TypeError("measurement must be a TargetMeasurement")
            if any(
                value is not None
                for value in (timestamp_s, position_world_m, confidence)
            ):
                raise TypeError(
                    "typed measurement cannot be combined with legacy update fields"
                )
            timestamp = measurement.timestamp_s
            measurement_position = _position(measurement.position_world_m)
            measurement_covariance = np.asarray(
                measurement.covariance_world_m2,
                dtype=np.float64,
            )
            if not np.all(np.isfinite(measurement_covariance)):
                raise ValueError("measurement covariance must be finite")
            if not np.allclose(
                measurement_covariance,
                measurement_covariance.T,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("measurement covariance must be symmetric")
            if float(np.min(np.linalg.eigvalsh(measurement_covariance))) < -1e-9:
                raise ValueError(
                    "measurement covariance must be positive semidefinite"
                )
            if float(np.trace(measurement_covariance)) <= 1e-12:
                raise ValueError(
                    "measurement covariance must contain positive uncertainty"
                )
        else:
            if timestamp_s is None or position_world_m is None or confidence is None:
                raise TypeError(
                    "update requires TargetMeasurement or all legacy fields"
                )
            timestamp = _finite(timestamp_s, "timestamp_s")
            measurement_position = _position(position_world_m)
            normalized_confidence = _finite(confidence, "confidence")
            if not 0.0 <= normalized_confidence <= 1.0:
                raise ValueError("confidence must be between zero and one")
            effective_confidence = max(normalized_confidence, 0.05)
            measurement_covariance = (
                np.eye(3, dtype=np.float64)
                * self._measurement_noise
                / effective_confidence
            )
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")

        if self._state is None:
            self._state = np.concatenate(
                [measurement_position, np.zeros(3, dtype=np.float64)]
            )
            self._covariance = np.zeros((6, 6), dtype=np.float64)
            self._covariance[:3, :3] = measurement_covariance
            self._covariance[3:, 3:] = (
                np.eye(3, dtype=np.float64)
                * self._initial_velocity_variance
            )
            self._timestamp_s = timestamp
            self._measurement_timestamp_s = timestamp
            self._measurements_accepted += 1
            self._last_outcome = TargetStateUpdateOutcome.MEASUREMENT_ACCEPTED
            return self._snapshot(
                state=self._state,
                covariance=self._covariance,
                timestamp_s=timestamp,
                predicted_only=False,
            )

        assert self._covariance is not None
        assert self._timestamp_s is not None
        if timestamp <= self._timestamp_s:
            raise ValueError("measurement timestamp must increase strictly")
        dt = timestamp - self._timestamp_s
        predicted_state, predicted_covariance = self._predict_arrays(
            self._state,
            self._covariance,
            dt,
        )
        innovation = measurement_position - predicted_state[:3]
        euclidean_jump = float(np.linalg.norm(innovation))
        innovation_covariance = (
            self._H @ predicted_covariance @ self._H.T
            + measurement_covariance
        )
        mahalanobis_squared = float(
            innovation.T
            @ np.linalg.solve(innovation_covariance, innovation)
        )
        if (
            euclidean_jump > self._max_position_jump_m
            or mahalanobis_squared > self._innovation_gate_sigma**2
        ):
            self._measurements_rejected += 1
            self._last_outcome = TargetStateUpdateOutcome.MEASUREMENT_REJECTED
            raise TargetStateMeasurementRejected(
                "world-position innovation rejected "
                f"(jump_m={euclidean_jump:.3f}, "
                f"mahalanobis_squared={mahalanobis_squared:.3f})"
            )

        kalman_gain = (
            predicted_covariance
            @ self._H.T
            @ np.linalg.inv(innovation_covariance)
        )
        updated_state = predicted_state + kalman_gain @ innovation
        identity = np.eye(6, dtype=np.float64)
        # Joseph form preserves symmetry/positive semidefiniteness under
        # floating-point roundoff.
        residual_transform = identity - kalman_gain @ self._H
        updated_covariance = (
            residual_transform
            @ predicted_covariance
            @ residual_transform.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        updated_covariance = (
            updated_covariance + updated_covariance.T
        ) / 2.0
        self._state = updated_state
        self._covariance = updated_covariance
        self._timestamp_s = timestamp
        self._measurement_timestamp_s = timestamp
        self._measurements_accepted += 1
        self._last_outcome = TargetStateUpdateOutcome.MEASUREMENT_ACCEPTED
        return self._snapshot(
            state=updated_state,
            covariance=updated_covariance,
            timestamp_s=timestamp,
            predicted_only=False,
        )

    def predict(self, timestamp_s: float) -> FilteredTargetState | None:
        """Return a non-mutating lost-target prediction within the age bound."""

        timestamp = _finite(timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        if self._state is None:
            return None
        assert self._covariance is not None
        assert self._timestamp_s is not None
        assert self._measurement_timestamp_s is not None
        if timestamp < self._timestamp_s:
            raise ValueError("prediction timestamp cannot move backwards")
        measurement_age = timestamp - self._measurement_timestamp_s
        if measurement_age > self._max_prediction_age_s:
            return None
        dt = timestamp - self._timestamp_s
        state, covariance = self._predict_arrays(
            self._state,
            self._covariance,
            dt,
        )
        snapshot = self._snapshot(
            state=state,
            covariance=covariance,
            timestamp_s=timestamp,
            predicted_only=measurement_age > 0.0,
        )
        if snapshot.predicted_only:
            self._predicted_only_outputs += 1
            self._last_outcome = TargetStateUpdateOutcome.PREDICTED_ONLY
        return snapshot

    def _predict_arrays(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3, dtype=np.float64) * dt
        process_covariance = np.zeros((6, 6), dtype=np.float64)
        process_covariance[:3, :3] = (
            np.eye(3, dtype=np.float64) * (dt**4 / 4.0)
        )
        process_covariance[:3, 3:] = (
            np.eye(3, dtype=np.float64) * (dt**3 / 2.0)
        )
        process_covariance[3:, :3] = process_covariance[:3, 3:]
        process_covariance[3:, 3:] = (
            np.eye(3, dtype=np.float64) * dt**2
        )
        process_covariance *= self._process_noise
        predicted_state = transition @ state
        predicted_covariance = (
            transition @ covariance @ transition.T + process_covariance
        )
        predicted_covariance = (
            predicted_covariance + predicted_covariance.T
        ) / 2.0
        return predicted_state, predicted_covariance

    def _snapshot(
        self,
        *,
        state: np.ndarray,
        covariance: np.ndarray,
        timestamp_s: float,
        predicted_only: bool,
    ) -> FilteredTargetState:
        assert self._measurement_timestamp_s is not None
        return FilteredTargetState(
            timestamp_s=timestamp_s,
            position_world_m=(
                float(state[0]),
                float(state[1]),
                float(state[2]),
            ),
            velocity_world_mps=(
                float(state[3]),
                float(state[4]),
                float(state[5]),
            ),
            measurement_age_s=timestamp_s - self._measurement_timestamp_s,
            predicted_only=predicted_only,
            covariance_6x6=_covariance_tuple(covariance),
        )


__all__ = [
    "FilteredTargetState",
    "TargetStateEstimator",
    "TargetStateEstimatorStatistics",
    "TargetStateMeasurementRejected",
    "TargetStateUpdateOutcome",
]
