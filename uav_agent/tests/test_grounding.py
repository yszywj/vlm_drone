from __future__ import annotations

import unittest

import numpy as np

from perception.candidate_bank import CandidateLifecycle, CandidateSnapshot
from perception.grounding import (
    CandidateResolutionUnavailable,
    CandidateResolver,
    GroundingBackend,
    GroundingBackendUnavailable,
    GroundingProposal,
    LearnedGrounder,
    OracleEvaluationCandidateResolver,
    OracleEvaluationGrounder,
    ProductionCandidateResolver,
    QwenVLGrounder,
    YOLOEGrounder,
)
from perception.runtime import (
    PerceptionBoundaryError,
    PerceptionRuntimeProfile,
)
from runtime.frame_store import FrameRef


def _candidate() -> CandidateSnapshot:
    frame = FrameRef("uav_1", "frame_1", 1, 20, 10)
    return CandidateSnapshot(
        uav_id="uav_1",
        candidate_id="candidate_1",
        first_seen_timestamp_s=1,
        last_seen_timestamp_s=1,
        bbox_history=((0.1, 0.2, 0.5, 0.8),),
        frame_history=(frame,),
        source="qwen_vl",
        lifecycle=CandidateLifecycle.PROVISIONAL,
        review_history=(),
    )


class GroundingBoundaryTest(unittest.TestCase):
    def test_oracle_resolver_requires_profile_and_explicit_acknowledgement(self) -> None:
        provider = lambda uav_id, candidate_id, timestamp_s: (1.0, 2.0, 0.0)
        with self.assertRaisesRegex(PerceptionBoundaryError, "PRODUCTION"):
            OracleEvaluationCandidateResolver(
                provider,
                profile=PerceptionRuntimeProfile.PRODUCTION,
                acknowledge_privileged_oracle=True,
            )
        with self.assertRaisesRegex(PerceptionBoundaryError, "acknowledge"):
            OracleEvaluationCandidateResolver(
                provider,
                profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
                acknowledge_privileged_oracle=False,
            )

    def test_oracle_resolver_returns_explicitly_privileged_internal_geometry(self) -> None:
        calls: list[tuple[str, str, float]] = []

        def provider(
            uav_id: str,
            candidate_id: str,
            timestamp_s: float,
        ) -> tuple[float, float, float]:
            calls.append((uav_id, candidate_id, timestamp_s))
            return (3.0, 4.0, 0.5)

        resolver = OracleEvaluationCandidateResolver(
            provider,
            profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            acknowledge_privileged_oracle=True,
        )
        self.assertIsInstance(resolver, CandidateResolver)
        resolved = resolver.resolve(_candidate(), timestamp_s=2)
        self.assertEqual(calls, [("uav_1", "candidate_1", 2.0)])
        self.assertEqual(resolved.uav_id, "uav_1")
        self.assertEqual(resolved.candidate_id, "candidate_1")
        self.assertEqual(resolved.position_xyz_m, (3.0, 4.0, 0.5))
        self.assertEqual(resolved.source, "oracle_evaluation")

    def test_production_resolver_is_explicitly_unimplemented(self) -> None:
        resolver = ProductionCandidateResolver()
        self.assertIsInstance(resolver, CandidateResolver)
        self.assertIs(resolver.profile, PerceptionRuntimeProfile.PRODUCTION)
        with self.assertRaisesRegex(
            CandidateResolutionUnavailable,
            "not implemented",
        ):
            resolver.resolve(_candidate(), timestamp_s=2)

    def test_invalid_oracle_geometry_fails_closed(self) -> None:
        invalid_positions = (
            (1.0, 2.0),
            (1.0, float("nan"), 3.0),
            np.asarray(1.0),
            object(),
        )
        for position in invalid_positions:
            with self.subTest(position=position):
                resolver = OracleEvaluationCandidateResolver(
                    lambda *args, value=position: value,
                    profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
                    acknowledge_privileged_oracle=True,
                )
                with self.assertRaises((TypeError, ValueError)):
                    resolver.resolve(_candidate(), timestamp_s=2)

    def test_grounding_protocol_is_image_space_only(self) -> None:
        class FakeGrounder:
            def propose(self, **kwargs: object):
                return ()

        self.assertIsInstance(FakeGrounder(), GroundingBackend)
        frame = FrameRef("uav_1", "frame_1", 1, 20, 10)
        proposal = GroundingProposal(
            uav_id="uav_1",
            candidate_id="candidate_1",
            frame_ref=frame,
            bbox_xyxy_normalized=(0.1, 0.2, 0.5, 0.8),
            source="qwen_vl",
            confidence=0.7,
        )
        encoded = proposal.to_dict()
        self.assertEqual(encoded["confidence"], 0.7)
        for forbidden in ("position", "velocity", "controller"):
            self.assertNotIn(forbidden, encoded)

    def test_named_future_grounders_are_explicit_non_fake_slots(self) -> None:
        with self.assertRaisesRegex(PerceptionBoundaryError, "PRODUCTION"):
            OracleEvaluationGrounder(
                profile=PerceptionRuntimeProfile.PRODUCTION,
                acknowledge_privileged_oracle=True,
            )
        grounders = (
            OracleEvaluationGrounder(
                profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
                acknowledge_privileged_oracle=True,
            ),
            QwenVLGrounder(),
            LearnedGrounder(),
            YOLOEGrounder(),
        )
        for grounder in grounders:
            with self.subTest(grounder=type(grounder).__name__):
                self.assertIsInstance(grounder, GroundingBackend)
                with self.assertRaises(GroundingBackendUnavailable):
                    grounder.propose(
                        uav_id="uav_1",
                        frame_ref=FrameRef("uav_1", "frame_1", 1, 20, 10),
                        rgb=np.zeros((10, 20, 3), dtype=np.uint8),
                        target_spec=object(),  # type: ignore[arg-type]
                        prompt_bundle=object(),  # type: ignore[arg-type]
                    )


if __name__ == "__main__":
    unittest.main()
