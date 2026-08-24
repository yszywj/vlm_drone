"""Evaluator-only target-state error side channel.

This module intentionally has no dependency on Isaac Sim or Skill/Agent
types.  Simulator truth is copied into :class:`TargetGroundTruth` only by an
explicit evaluator launch path.  :meth:`TargetEstimateEvaluator.evaluate`
writes metrics and returns ``None``, so it cannot be substituted for a
control observation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from numbers import Integral, Real
from typing import Protocol

from common.target_estimate import TargetEstimate


ISAAC_EVALUATOR_GROUND_TRUTH = "isaac_evaluator_ground_truth"


class TargetEvaluationMode(str, Enum):
    """Explicit authority required to consume privileged target truth."""

    ORACLE_GROUND_TRUTH = "oracle_ground_truth"


class TargetEvaluationError(ValueError):
    """Raised when evaluator input is malformed or has untrusted provenance."""


class TargetErrorMetricSink(Protocol):
    def record_evaluator_error(
        self,
        *,
        position_error_m: float | None,
        velocity_error_mps: float | None,
        evaluator_mode: bool,
    ) -> None: ...


def _finite_nonnegative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TargetEvaluationError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise TargetEvaluationError(
            f"{field_name} must be finite and non-negative"
        )
    return normalized


def _vector3(value: object, field_name: str) -> tuple[float, float, float]:
    # Tuple-only input keeps numpy arrays, tensors and simulator objects out of
    # this small cross-boundary value type.
    if not isinstance(value, tuple) or len(value) != 3:
        raise TargetEvaluationError(f"{field_name} must be a three-number tuple")
    normalized: list[float] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, Real):
            raise TargetEvaluationError(
                f"{field_name}[{index}] must be a finite number"
            )
        number = float(component)
        if not isfinite(number):
            raise TargetEvaluationError(
                f"{field_name}[{index}] must be a finite number"
            )
        normalized.append(number)
    return normalized[0], normalized[1], normalized[2]


@dataclass(frozen=True, slots=True)
class TargetGroundTruth:
    """Minimal synchronized evaluator label, never an Agent observation."""

    timestamp_s: float
    position_world_m: tuple[float, float, float]
    velocity_world_mps: tuple[float, float, float]
    target_id: str = "target"
    provenance: str = ISAAC_EVALUATOR_GROUND_TRUTH

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_s",
            _finite_nonnegative(self.timestamp_s, "timestamp_s"),
        )
        object.__setattr__(
            self,
            "position_world_m",
            _vector3(self.position_world_m, "position_world_m"),
        )
        object.__setattr__(
            self,
            "velocity_world_mps",
            _vector3(self.velocity_world_mps, "velocity_world_mps"),
        )
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise TargetEvaluationError("target_id must be a non-empty string")
        if self.target_id != self.target_id.strip():
            raise TargetEvaluationError("target_id must not contain surrounding whitespace")
        if self.provenance != ISAAC_EVALUATOR_GROUND_TRUTH:
            raise TargetEvaluationError(
                "ground truth provenance must be isaac_evaluator_ground_truth"
            )


class TargetEstimateEvaluator:
    """Accumulate synchronized position/velocity errors without returning GT.

    Detector inference is asynchronous.  The evaluator therefore retains a
    small GT timestamp history and compares a delayed estimate with truth from
    the estimate's own timestamp, not with the newest simulator pose.
    """

    def __init__(
        self,
        metrics: TargetErrorMetricSink,
        *,
        mode: TargetEvaluationMode,
        allowed_estimate_sources: frozenset[str],
        max_history: int = 256,
        timestamp_tolerance_s: float = 1e-6,
    ) -> None:
        if mode is not TargetEvaluationMode.ORACLE_GROUND_TRUTH:
            raise PermissionError(
                "TargetEstimateEvaluator requires explicit oracle-ground-truth mode"
            )
        if not callable(getattr(metrics, "record_evaluator_error", None)):
            raise TypeError("metrics must provide record_evaluator_error()")
        if not isinstance(allowed_estimate_sources, frozenset) or not allowed_estimate_sources:
            raise TypeError("allowed_estimate_sources must be a non-empty frozenset")
        for source in allowed_estimate_sources:
            if not isinstance(source, str) or not source or source != source.strip():
                raise TargetEvaluationError(
                    "allowed estimate sources must be non-empty stripped strings"
                )
        if isinstance(max_history, bool) or not isinstance(max_history, Integral):
            raise TypeError("max_history must be a positive integer")
        if int(max_history) <= 0:
            raise ValueError("max_history must be greater than zero")
        tolerance = _finite_nonnegative(
            timestamp_tolerance_s,
            "timestamp_tolerance_s",
        )
        self._metrics = metrics
        self._allowed_sources = allowed_estimate_sources
        self._max_history = int(max_history)
        self._timestamp_tolerance_s = tolerance
        self._truth: deque[TargetGroundTruth] = deque(maxlen=self._max_history)
        self._evaluated: deque[TargetEstimate] = deque(maxlen=self._max_history)
        self._evaluated_set: set[TargetEstimate] = set()
        self._matched_samples = 0

    @property
    def matched_samples(self) -> int:
        return self._matched_samples

    def evaluate(
        self,
        estimate: TargetEstimate | None,
        ground_truth: TargetGroundTruth,
    ) -> None:
        """Record errors for one estimate and return no control-consumable value."""

        if not isinstance(ground_truth, TargetGroundTruth):
            raise TypeError("ground_truth must be TargetGroundTruth")
        self._truth.append(ground_truth)
        if estimate is None:
            return None
        if not isinstance(estimate, TargetEstimate):
            raise TypeError("estimate must be TargetEstimate or None")
        if estimate.source not in self._allowed_sources:
            raise PermissionError(
                f"target estimate source {estimate.source!r} is not authorized for this evaluator"
            )
        if estimate in self._evaluated_set:
            return None
        truth = self._truth_for_timestamp(estimate.timestamp_s)
        if truth is None:
            return None

        position_error = _euclidean_error(
            estimate.position_world_m,
            truth.position_world_m,
        )
        velocity_error = _euclidean_error(
            estimate.velocity_world_mps,
            truth.velocity_world_mps,
        )
        if position_error is None and velocity_error is None:
            return None
        self._metrics.record_evaluator_error(
            position_error_m=position_error,
            velocity_error_mps=velocity_error,
            evaluator_mode=True,
        )
        self._remember_evaluated(estimate)
        self._matched_samples += 1
        return None

    def _truth_for_timestamp(self, timestamp_s: float) -> TargetGroundTruth | None:
        for truth in reversed(self._truth):
            if abs(truth.timestamp_s - timestamp_s) <= self._timestamp_tolerance_s:
                return truth
        return None

    def _remember_evaluated(self, estimate: TargetEstimate) -> None:
        if len(self._evaluated) == self._max_history:
            expired = self._evaluated.popleft()
            self._evaluated_set.discard(expired)
        self._evaluated.append(estimate)
        self._evaluated_set.add(estimate)


def _euclidean_error(
    estimate: tuple[float, float, float] | None,
    truth: tuple[float, float, float],
) -> float | None:
    if estimate is None:
        return None
    return sqrt(sum((estimated - actual) ** 2 for estimated, actual in zip(estimate, truth)))


__all__ = [
    "ISAAC_EVALUATOR_GROUND_TRUTH",
    "TargetEvaluationError",
    "TargetEvaluationMode",
    "TargetEstimateEvaluator",
    "TargetGroundTruth",
]
