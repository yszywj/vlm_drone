from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from common.target_estimate import TargetEstimate
from configs.loader import load_config
from env.uav_controller import UAVState
from perception.factory import (
    TargetPerceptionConfigurationError,
    TargetPerceptionUnavailableError,
    build_target_candidate_resolver,
    build_target_perception_backend,
    validate_target_perception_preflight,
)
from perception.depth_geometry import DepthCandidateResolver
from perception.runtime import GuardedPerceptionBackend, PerceptionRuntimeProfile
from perception.vision_backend import (
    DisabledTargetPerceptionBackend,
    VisionPerceptionBackend,
)
from skills.types import SkillName
from skills.types import Observation
from runtime.frame_store import FrameStore


ROOT = Path(__file__).resolve().parents[1]


class TargetPerceptionFactoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs/default.yaml")

    def with_backend(self, value: str):
        return replace(
            self.config,
            target_perception=replace(self.config.target_perception, backend=value),
        )

    def test_only_audited_profile_backend_combinations_construct(self) -> None:
        oracle = build_target_perception_backend(
            self.with_backend("oracle_evaluation"),
            runtime_profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            acknowledge_privileged_oracle=True,
            uav_id="uav_1",
        )
        self.assertIsInstance(oracle, GuardedPerceptionBackend)
        vision = build_target_perception_backend(
            self.with_backend("ultralytics_service"),
            runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
            acknowledge_privileged_oracle=False,
            uav_id="uav_1",
        )
        self.assertIsInstance(vision, VisionPerceptionBackend)
        disabled = build_target_perception_backend(
            self.with_backend("disabled"),
            runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
            acknowledge_privileged_oracle=False,
            uav_id="uav_1",
        )
        self.assertIsInstance(disabled, DisabledTargetPerceptionBackend)

    def test_invalid_combinations_fail_without_fallback(self) -> None:
        for config, profile, ack in (
            (self.with_backend("oracle_evaluation"), PerceptionRuntimeProfile.PRODUCTION, False),
            (self.with_backend("ultralytics_service"), PerceptionRuntimeProfile.PRODUCTION, True),
            (self.with_backend("oracle_evaluation"), PerceptionRuntimeProfile.ORACLE_EVALUATION, False),
        ):
            with self.subTest(profile=profile, ack=ack), self.assertRaises(
                TargetPerceptionConfigurationError
            ):
                build_target_perception_backend(
                    config,
                    runtime_profile=profile,
                    acknowledge_privileged_oracle=ack,
                    uav_id="uav_1",
                )

    def test_disabled_preflight_rejects_target_skills_only(self) -> None:
        validate_target_perception_preflight("disabled", (SkillName.TAKEOFF, SkillName.LAND))
        with self.assertRaises(TargetPerceptionUnavailableError):
            validate_target_perception_preflight("disabled", (SkillName.SEARCH,))

    def test_geometry_resolver_factory_honors_isaac_depth_mode(self) -> None:
        store = FrameStore(max_frames=2, max_bytes=4096, max_age_s=1.0)
        resolver = build_target_candidate_resolver(
            self.config.target_perception,
            frame_store=store,
        )
        self.assertIsInstance(resolver, DepthCandidateResolver)
        self.assertIs(resolver._frame_store, store)

    def test_vision_backend_rejects_oracle_source_aliases(self) -> None:
        backend = VisionPerceptionBackend(
            self.with_backend("ultralytics_service").target_perception,
            uav_id="uav_1",
        )
        base_estimate = TargetEstimate(
            timestamp_s=1.0,
            target_id="target_1",
            candidate_id="candidate_1",
            tracker_id="track_1",
            visible=True,
            confirmed=True,
            predicted_only=False,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox_xyxy_normalized=(0.1, 0.2, 0.3, 0.8),
            position_world_m=(1.0, 2.0, 3.0),
            velocity_world_mps=(0.0, 0.0, 0.0),
            measurement_age_s=0.0,
            source="yolo26_botsort",
        )
        base = Observation(
            uav_id="uav_1",
            timestamp=1.0,
            uav_pose=UAVState(0.0, 0.0, 1.0, 0.0),
            uav_velocity=np.zeros(3),
            camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            target_estimate=base_estimate,
        )

        self.assertIs(backend.observe(base), base)
        for source in ("oracle", "oracle_truth", "OrAcLe_BrIdGe"):
            with self.subTest(source=source), self.assertRaisesRegex(
                RuntimeError,
                "Oracle TargetEstimate",
            ):
                backend.observe(
                    replace(
                        base,
                        target_estimate=replace(base_estimate, source=source),
                    )
                )


if __name__ == "__main__":
    unittest.main()
