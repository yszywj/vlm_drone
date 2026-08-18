"""Pure-Python routing regression for every Oracle Isaac entry point."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from env.kinematic_uav import UAVState
from env.moving_target import TargetState
from perception import GuardedPerceptionBackend, OraclePerception
from scripts.run_dynamic_visual_mission import (
    _build_oracle_evaluation_backend as build_dynamic_visual_backend,
)
from scripts.run_llm_oracle_pipeline import (
    _build_oracle_evaluation_backend as build_llm_oracle_backend,
)
from scripts.run_oracle_pipeline import (
    _build_oracle_evaluation_backend as build_oracle_backend,
)


class OracleScriptRoutingTest(unittest.TestCase):
    def test_every_oracle_entry_point_binds_nondefault_uav(self) -> None:
        for name, builder in (
            ("oracle", build_oracle_backend),
            ("llm_oracle", build_llm_oracle_backend),
            ("dynamic_visual", build_dynamic_visual_backend),
        ):
            with self.subTest(script=name):
                guarded = builder("uav_2")
                self.assertIsInstance(guarded, GuardedPerceptionBackend)
                self.assertIsInstance(guarded.backend, OraclePerception)
                self.assertEqual(guarded.backend.uav_id, "uav_2")

    def test_nondefault_route_reaches_the_emitted_observation(self) -> None:
        guarded = build_dynamic_visual_backend("uav_2")
        frame = SimpleNamespace(
            observation=SimpleNamespace(
                camera_timestamp_s=1.5,
                uav_state=UAVState(0.0, 0.0, 1.0, 0.0),
                uav_velocity_mps=np.zeros(3),
                rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                camera_position_m=np.array((0.0, 0.0, 1.0)),
                camera_orientation_wxyz=np.array((1.0, 0.0, 0.0, 0.0)),
            ),
            target_projection=SimpleNamespace(
                visible=np.array((True,), dtype=np.bool_),
            ),
            target_state=TargetState(2.0, 1.0, 0.0, 0.0),
            target_velocity_mps=np.zeros(3),
        )

        observation = guarded.observe(frame)

        self.assertEqual(observation.uav_id, "uav_2")
        observation.validate()

    def test_invalid_route_is_rejected_before_isaac_startup(self) -> None:
        for builder in (
            build_oracle_backend,
            build_llm_oracle_backend,
            build_dynamic_visual_backend,
        ):
            with self.assertRaises((TypeError, ValueError)):
                builder("wrong uav")


if __name__ == "__main__":
    unittest.main()
