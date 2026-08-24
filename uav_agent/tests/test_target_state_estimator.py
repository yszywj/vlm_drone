from __future__ import annotations

import unittest

import numpy as np

from perception.target_state_estimator import (
    TargetStateEstimator,
    TargetStateMeasurementRejected,
)


class TargetStateEstimatorTest(unittest.TestCase):
    def test_updates_velocity_and_bounds_lost_target_prediction(self) -> None:
        estimator = TargetStateEstimator(
            max_prediction_age_s=1.0,
            max_position_jump_m=5.0,
            process_noise=0.1,
            measurement_noise=0.05,
        )
        initial = estimator.update(
            timestamp_s=0.0,
            position_world_m=(0.0, 0.0, 0.0),
            confidence=1.0,
        )
        measured = estimator.update(
            timestamp_s=1.0,
            position_world_m=(1.0, 0.0, 0.0),
            confidence=1.0,
        )
        predicted = estimator.predict(1.5)
        assert predicted is not None

        self.assertFalse(initial.predicted_only)
        self.assertFalse(measured.predicted_only)
        self.assertGreater(measured.velocity_world_mps[0], 0.5)
        self.assertTrue(predicted.predicted_only)
        self.assertEqual(predicted.measurement_age_s, 0.5)
        self.assertGreater(predicted.position_world_m[0], measured.position_world_m[0])
        self.assertIsNone(estimator.predict(2.01))

    def test_time_backwards_and_duplicate_measurements_are_rejected(self) -> None:
        estimator = TargetStateEstimator()
        estimator.update(
            timestamp_s=2.0,
            position_world_m=(0.0, 0.0, 0.0),
            confidence=1.0,
        )
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            estimator.update(
                timestamp_s=2.0,
                position_world_m=(0.0, 0.0, 0.0),
                confidence=1.0,
            )
        with self.assertRaisesRegex(ValueError, "backwards"):
            estimator.predict(1.0)

    def test_position_jump_is_rejected_without_corrupting_filter(self) -> None:
        estimator = TargetStateEstimator(max_position_jump_m=2.0)
        estimator.update(
            timestamp_s=0.0,
            position_world_m=(0.0, 0.0, 0.0),
            confidence=1.0,
        )
        before = estimator.predict(1.0)
        with self.assertRaises(TargetStateMeasurementRejected):
            estimator.update(
                timestamp_s=1.0,
                position_world_m=(100.0, 0.0, 0.0),
                confidence=1.0,
            )
        after = estimator.predict(1.0)
        self.assertEqual(before, after)

    def test_covariance_is_finite_symmetric_and_reset_clears_state(self) -> None:
        estimator = TargetStateEstimator()
        state = estimator.update(
            timestamp_s=0.0,
            position_world_m=(1.0, 2.0, 3.0),
            confidence=0.5,
        )
        covariance = np.asarray(state.covariance)
        self.assertEqual(covariance.shape, (6, 6))
        self.assertTrue(np.all(np.isfinite(covariance)))
        np.testing.assert_allclose(covariance, covariance.T)
        estimator.reset()
        self.assertFalse(estimator.is_initialized)
        self.assertIsNone(estimator.predict(1.0))


if __name__ == "__main__":
    unittest.main()
