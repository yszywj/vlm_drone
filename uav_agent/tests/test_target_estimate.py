from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from common.target_estimate import TargetEstimate
from env.uav_controller import UAVState
from perception.runtime import (
    PerceptionBoundaryError,
    PerceptionRuntimeProfile,
    validate_observation_access,
)
from skills.types import Observation


def estimate(**overrides: object) -> TargetEstimate:
    values: dict[str, object] = {
        "timestamp_s": 1.0,
        "target_id": "target_1",
        "candidate_id": "candidate_1",
        "tracker_id": "track_7",
        "visible": True,
        "confirmed": True,
        "predicted_only": False,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "bbox_xyxy_normalized": (0.1, 0.2, 0.3, 0.8),
        "position_world_m": (1.0, 2.0, 3.0),
        "velocity_world_mps": (0.1, 0.0, 0.0),
        "measurement_age_s": 0.0,
        "source": "yolo26_botsort",
    }
    values.update(overrides)
    return TargetEstimate(**values)  # type: ignore[arg-type]


class TargetEstimateTest(unittest.TestCase):
    def test_round_trip_is_model_object_free_and_strict(self) -> None:
        value = estimate()
        self.assertEqual(TargetEstimate.from_dict(value.to_dict()), value)
        with self.assertRaises(ValueError):
            TargetEstimate.from_dict({**value.to_dict(), "tensor": object()})

    def test_visibility_confirmation_and_prediction_invariants(self) -> None:
        with self.assertRaises(ValueError):
            estimate(bbox_xyxy_normalized=None)
        with self.assertRaises(ValueError):
            estimate(target_id=None)
        with self.assertRaises(ValueError):
            estimate(predicted_only=True)
        with self.assertRaises(ValueError):
            estimate(confidence=float("nan"))

    def test_production_rejects_all_oracle_source_aliases_but_accepts_vision(
        self,
    ) -> None:
        base = Observation(
            uav_id="uav_1",
            timestamp=1.0,
            uav_pose=UAVState(0.0, 0.0, 1.0, 0.0),
            uav_velocity=np.zeros(3),
            camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            target_estimate=estimate(),
        )
        validate_observation_access(base)
        for source in ("oracle", "oracle_truth", "OrAcLe_EvAlUaTiOn"):
            privileged = replace(
                base,
                target_estimate=replace(estimate(), source=source),
            )
            with self.subTest(source=source), self.assertRaises(
                PerceptionBoundaryError
            ):
                validate_observation_access(privileged)

            # Evaluators remain able to consume explicitly privileged data;
            # only the production Agent Runtime must fail closed.
            validate_observation_access(
                privileged,
                PerceptionRuntimeProfile.ORACLE_EVALUATION,
            )


if __name__ == "__main__":
    unittest.main()
