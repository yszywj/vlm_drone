from __future__ import annotations

import numpy as np
import pytest

from perception.measurement import TargetMeasurement
from perception.target_state_estimator import (
    TargetStateEstimator,
    TargetStateMeasurementRejected,
    TargetStateUpdateOutcome,
)


def _measurement(
    timestamp_s: float,
    position: tuple[float, float, float],
    variance: float,
) -> TargetMeasurement:
    covariance = tuple(
        tuple(variance if row == column else 0.0 for column in range(3))
        for row in range(3)
    )
    return TargetMeasurement(
        timestamp_s=timestamp_s,
        candidate_id="candidate_1",
        tracker_id="track_1",
        pixel_uv=(20.0, 20.0),
        raw_depth_m=5.0,
        corrected_depth_m=5.0,
        position_camera_flu_m=(5.0, 0.0, 0.0),
        position_world_m=position,
        covariance_world_m2=covariance,  # type: ignore[arg-type]
        measurement_quality=0.9,
        source="rgbd_depth_geometry",
    )


def test_filter_uses_measurement_covariance_and_records_acceptance() -> None:
    estimator = TargetStateEstimator(process_noise=0.01)
    state = estimator.update(_measurement(0.0, (1.0, 2.0, 3.0), 0.04))
    np.testing.assert_allclose(
        np.asarray(state.covariance)[:3, :3],
        np.eye(3) * 0.04,
    )
    assert estimator.last_outcome is TargetStateUpdateOutcome.MEASUREMENT_ACCEPTED
    assert estimator.statistics.measurements_accepted == 1


def test_lower_covariance_measurement_has_more_influence() -> None:
    precise = TargetStateEstimator(process_noise=0.01, max_position_jump_m=20.0)
    noisy = TargetStateEstimator(process_noise=0.01, max_position_jump_m=20.0)
    initial = _measurement(0.0, (0.0, 0.0, 0.0), 0.1)
    precise.update(initial)
    noisy.update(initial)
    precise_state = precise.update(_measurement(1.0, (1.0, 0.0, 0.0), 0.01))
    noisy_state = noisy.update(_measurement(1.0, (1.0, 0.0, 0.0), 10.0))
    assert precise_state.position_world_m[0] > noisy_state.position_world_m[0]


def test_innovation_rejection_and_predicted_only_are_distinct() -> None:
    estimator = TargetStateEstimator(
        max_position_jump_m=1.0,
        max_prediction_age_s=2.0,
    )
    estimator.update(_measurement(0.0, (0.0, 0.0, 0.0), 0.1))
    with pytest.raises(TargetStateMeasurementRejected):
        estimator.update(_measurement(1.0, (100.0, 0.0, 0.0), 0.1))
    assert estimator.last_outcome is TargetStateUpdateOutcome.MEASUREMENT_REJECTED
    assert estimator.statistics.measurements_rejected == 1

    predicted = estimator.predict(1.0)
    assert predicted is not None and predicted.predicted_only
    assert estimator.last_outcome is TargetStateUpdateOutcome.PREDICTED_ONLY
    assert estimator.statistics.predicted_only_outputs == 1


def test_zero_covariance_is_never_silently_treated_as_zero_noise() -> None:
    estimator = TargetStateEstimator()
    with pytest.raises(ValueError, match="positive uncertainty"):
        estimator.update(_measurement(0.0, (0.0, 0.0, 0.0), 0.0))

