"""Pure tests for the privileged/perception runtime boundary."""

from __future__ import annotations

import unittest

import numpy as np

from env.kinematic_uav import UAVState
from perception import (
    GuardedPerceptionBackend,
    OraclePerception,
    PerceptionBoundaryError,
    PerceptionCapability,
    PerceptionRuntimeProfile,
    observation_contains_oracle_data,
    validate_observation_access,
)
from skills.types import Observation


def observation(*, oracle: bool = False) -> Observation:
    return Observation(
        timestamp=1.0,
        uav_pose=UAVState(0.0, 0.0, 1.0, 0.0),
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        oracle_target_visible=False if oracle else None,
    )


class FakeBackend:
    capability = PerceptionCapability.VISION

    def __init__(self, value: Observation) -> None:
        self.value = value

    def observe(self, frame: object) -> Observation:
        del frame
        return self.value


class PerceptionRuntimeTests(unittest.TestCase):
    def test_production_is_default_and_accepts_unprivileged_observation(self) -> None:
        guarded = GuardedPerceptionBackend(FakeBackend(observation()))

        result = guarded.observe(object())

        self.assertFalse(observation_contains_oracle_data(result))
        self.assertIs(guarded.profile, PerceptionRuntimeProfile.PRODUCTION)

    def test_production_rejects_oracle_backend_before_observe(self) -> None:
        with self.assertRaisesRegex(PerceptionBoundaryError, "forbidden"):
            GuardedPerceptionBackend(OraclePerception())

    def test_oracle_evaluation_requires_explicit_acknowledgement(self) -> None:
        with self.assertRaisesRegex(PerceptionBoundaryError, "explicit"):
            GuardedPerceptionBackend(
                OraclePerception(),
                profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            )

        guarded = GuardedPerceptionBackend(
            OraclePerception(),
            profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            acknowledge_privileged_oracle=True,
        )
        self.assertIs(
            guarded.capability,
            PerceptionCapability.PRIVILEGED_ORACLE,
        )

    def test_mislabelled_vision_backend_cannot_smuggle_oracle_into_production(self) -> None:
        guarded = GuardedPerceptionBackend(FakeBackend(observation(oracle=True)))

        with self.assertRaisesRegex(PerceptionBoundaryError, "Oracle fields"):
            guarded.observe(object())

    def test_mislabelled_backend_cannot_emit_oracle_in_evaluation_profile(self) -> None:
        guarded = GuardedPerceptionBackend(
            FakeBackend(observation(oracle=True)),
            profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
        )

        with self.assertRaisesRegex(PerceptionBoundaryError, "declaring"):
            guarded.observe(object())

    def test_direct_access_validation_is_strict(self) -> None:
        validate_observation_access(observation())
        with self.assertRaisesRegex(PerceptionBoundaryError, "oracle_target_visible"):
            validate_observation_access(observation(oracle=True))
        validate_observation_access(
            observation(oracle=True),
            PerceptionRuntimeProfile.ORACLE_EVALUATION,
        )

    def test_acknowledgement_is_rejected_for_vision_backend(self) -> None:
        with self.assertRaisesRegex(PerceptionBoundaryError, "vision backend"):
            GuardedPerceptionBackend(
                FakeBackend(observation()),
                acknowledge_privileged_oracle=True,
            )

    def test_backend_cannot_change_capability_after_policy_is_set(self) -> None:
        backend = FakeBackend(observation(oracle=True))
        guarded = GuardedPerceptionBackend(
            backend,
            profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
        )
        backend.capability = PerceptionCapability.PRIVILEGED_ORACLE

        with self.assertRaisesRegex(PerceptionBoundaryError, "changed"):
            guarded.observe(object())


if __name__ == "__main__":
    unittest.main()
