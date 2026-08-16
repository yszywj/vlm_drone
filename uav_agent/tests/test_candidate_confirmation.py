"""Pure tests for detector -> candidate -> confirmed target coordination."""

from __future__ import annotations

import math
import unittest

from perception import (
    CandidateConfirmationCoordinator,
    CandidateConfirmationError,
    ConfirmationDecision,
    ConfirmationPolicy,
    DetectionCandidate,
    IdentityConsistencyEvidence,
    SemanticVerification,
    ShortTrackEvidence,
)
from target import TargetLifecycle, TargetManager, TargetSpec, TargetStateError


def searching_manager() -> TargetManager:
    manager = TargetManager()
    manager.start_search(TargetSpec("red moving vehicle"), timestamp_s=0.0)
    return manager


def reacquiring_manager() -> TargetManager:
    manager = searching_manager()
    manager.lock_oracle_from_search(
        "target_7",
        timestamp_s=0.2,
        last_seen_position=(2.0, 3.0, 0.0),
        last_seen_velocity=(0.1, 0.0, 0.0),
    )
    manager.start_tracking(timestamp_s=0.3)
    manager.mark_lost(timestamp_s=0.4)
    manager.start_reacquiring(timestamp_s=0.5)
    return manager


def candidate() -> DetectionCandidate:
    return DetectionCandidate(
        candidate_id="tracklet_7",
        timestamp_s=1.0,
        confidence=0.8,
        estimated_position=(1.0, 2.0, 0.0),
    )


def evidence(
    *,
    stable: bool = True,
    matches: bool = True,
    reidentified: bool = True,
    consistent: bool = True,
    count: int = 4,
    duration: float = 0.8,
    confidence: float = 0.9,
) -> tuple[ShortTrackEvidence, SemanticVerification, IdentityConsistencyEvidence]:
    return (
        ShortTrackEvidence(
            "tracklet_7",
            timestamp_s=1.8,
            observation_count=count,
            duration_s=duration,
            stable=stable,
            confidence=confidence,
        ),
        SemanticVerification(
            "tracklet_7",
            timestamp_s=1.9,
            target_description="red moving vehicle",
            matches=matches,
            confidence=confidence,
            verifier="qwen-vl",
        ),
        IdentityConsistencyEvidence(
            "tracklet_7",
            target_id="target_7",
            timestamp_s=2.0,
            reidentified=reidentified,
            temporally_consistent=consistent,
            consistent_observations=count,
            confidence=confidence,
        ),
    )


class CandidateConfirmationTests(unittest.TestCase):
    def test_register_candidate_uses_real_candidate_lifecycle(self) -> None:
        manager = searching_manager()
        result = CandidateConfirmationCoordinator().register_candidate(
            candidate(), manager
        )

        self.assertIs(result.decision, ConfirmationDecision.PENDING)
        self.assertIs(manager.lifecycle, TargetLifecycle.CANDIDATE)
        self.assertEqual(manager.snapshot().target_id, "tracklet_7")
        self.assertEqual(manager.snapshot().source, "detector")

    def test_complete_evidence_locks_only_after_all_three_stages(self) -> None:
        manager = searching_manager()
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(candidate(), manager)
        track, semantic, identity = evidence()

        result = coordinator.evaluate(
            target_manager=manager,
            track=track,
            semantic=semantic,
            identity=identity,
        )

        self.assertIs(result.decision, ConfirmationDecision.CONFIRMED)
        self.assertIs(manager.lifecycle, TargetLifecycle.LOCKED)
        self.assertEqual(manager.snapshot().target_id, "target_7")
        self.assertEqual(manager.snapshot().source, "confirmed_vision")
        self.assertEqual(
            [event.new_state for event in manager.events()],
            [
                TargetLifecycle.SEARCHING,
                TargetLifecycle.CANDIDATE,
                TargetLifecycle.LOCKED,
            ],
        )

    def test_reacquire_uses_same_candidate_confirmation_chain(self) -> None:
        manager = reacquiring_manager()
        before = manager.snapshot()
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(candidate(), manager)
        self.assertIs(manager.lifecycle, TargetLifecycle.CANDIDATE)
        track, semantic, identity = evidence()

        result = coordinator.evaluate(
            target_manager=manager,
            track=track,
            semantic=semantic,
            identity=identity,
        )

        self.assertIs(result.decision, ConfirmationDecision.CONFIRMED)
        self.assertIs(manager.lifecycle, TargetLifecycle.LOCKED)
        self.assertEqual(manager.snapshot().source, "confirmed_vision")
        self.assertEqual(before.description, manager.snapshot().description)

    def test_rejected_reacquire_candidate_restores_last_seen_target(self) -> None:
        manager = reacquiring_manager()
        previous = manager.snapshot()
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(candidate(), manager)
        track, semantic, identity = evidence(matches=False)

        result = coordinator.evaluate(
            target_manager=manager,
            track=track,
            semantic=semantic,
            identity=identity,
        )

        self.assertIs(result.decision, ConfirmationDecision.REJECTED)
        restored = manager.snapshot()
        self.assertIs(restored.lifecycle, TargetLifecycle.REACQUIRING)
        self.assertEqual(restored.target_id, previous.target_id)
        self.assertEqual(restored.last_seen_position, previous.last_seen_position)
        self.assertEqual(restored.last_seen_velocity, previous.last_seen_velocity)
        self.assertEqual(restored.last_seen_time_s, previous.last_seen_time_s)

    def test_reacquire_cannot_switch_to_a_different_target_identity(self) -> None:
        manager = reacquiring_manager()
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(candidate(), manager)
        track, semantic, identity = evidence()
        wrong_identity = IdentityConsistencyEvidence(
            identity.candidate_id,
            target_id="another_target",
            timestamp_s=identity.timestamp_s,
            reidentified=True,
            temporally_consistent=True,
            consistent_observations=identity.consistent_observations,
            confidence=identity.confidence,
        )
        before = manager.snapshot()
        events_before = manager.events()

        with self.assertRaisesRegex(TargetStateError, "previously tracked"):
            coordinator.evaluate(
                target_manager=manager,
                track=track,
                semantic=semantic,
                identity=wrong_identity,
            )

        self.assertEqual(manager.snapshot(), before)
        self.assertEqual(manager.events(), events_before)

    def test_positive_but_insufficient_short_track_remains_candidate(self) -> None:
        manager = searching_manager()
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(candidate(), manager)
        track, semantic, identity = evidence(count=2, duration=0.1)

        result = coordinator.evaluate(
            target_manager=manager,
            track=track,
            semantic=semantic,
            identity=identity,
        )

        self.assertIs(result.decision, ConfirmationDecision.PENDING)
        self.assertIn("track_observation_count", result.reason)
        self.assertIs(manager.lifecycle, TargetLifecycle.CANDIDATE)

    def test_negative_stage_rejects_and_clears_candidate_data(self) -> None:
        for field in ("stable", "matches", "reidentified", "consistent"):
            with self.subTest(field=field):
                manager = searching_manager()
                coordinator = CandidateConfirmationCoordinator()
                coordinator.register_candidate(candidate(), manager)
                kwargs = {field: False}
                track, semantic, identity = evidence(**kwargs)

                result = coordinator.evaluate(
                    target_manager=manager,
                    track=track,
                    semantic=semantic,
                    identity=identity,
                )

                self.assertIs(result.decision, ConfirmationDecision.REJECTED)
                self.assertIs(manager.lifecycle, TargetLifecycle.SEARCHING)
                snapshot = manager.snapshot()
                self.assertIsNone(snapshot.target_id)
                self.assertIsNone(snapshot.last_seen_position)
                self.assertIsNone(snapshot.source)

    def test_mismatched_candidate_evidence_is_rejected_without_mutation(self) -> None:
        manager = searching_manager()
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(candidate(), manager)
        track, semantic, identity = evidence()
        wrong_track = ShortTrackEvidence(
            "another",
            timestamp_s=track.timestamp_s,
            observation_count=track.observation_count,
            duration_s=track.duration_s,
            stable=track.stable,
            confidence=track.confidence,
        )

        with self.assertRaisesRegex(CandidateConfirmationError, "match"):
            coordinator.evaluate(
                target_manager=manager,
                track=wrong_track,
                semantic=semantic,
                identity=identity,
            )
        self.assertIs(manager.lifecycle, TargetLifecycle.CANDIDATE)

    def test_semantics_must_match_mission_description(self) -> None:
        manager = searching_manager()
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(candidate(), manager)
        track, _, identity = evidence()
        semantic = SemanticVerification(
            "tracklet_7",
            timestamp_s=1.6,
            target_description="wrong object",
            matches=True,
            confidence=0.9,
            verifier="qwen-vl",
        )

        with self.assertRaisesRegex(CandidateConfirmationError, "description"):
            coordinator.evaluate(
                target_manager=manager,
                track=track,
                semantic=semantic,
                identity=identity,
            )

    def test_evidence_cannot_predate_candidate(self) -> None:
        manager = searching_manager()
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(candidate(), manager)
        _, semantic, identity = evidence()
        early_track = ShortTrackEvidence(
            "tracklet_7", 0.9, 4, 0.8, True, 0.9
        )

        with self.assertRaisesRegex(CandidateConfirmationError, "predate"):
            coordinator.evaluate(
                target_manager=manager,
                track=early_track,
                semantic=semantic,
                identity=identity,
            )

    def test_evidence_must_be_temporally_possible_and_ordered(self) -> None:
        manager = searching_manager()
        coordinator = CandidateConfirmationCoordinator()
        coordinator.register_candidate(candidate(), manager)
        track, semantic, identity = evidence()

        reversed_semantic = SemanticVerification(
            semantic.candidate_id,
            timestamp_s=track.timestamp_s - 0.1,
            target_description=semantic.target_description,
            matches=True,
            confidence=semantic.confidence,
            verifier=semantic.verifier,
        )
        with self.assertRaisesRegex(CandidateConfirmationError, "ordered"):
            coordinator.evaluate(
                target_manager=manager,
                track=track,
                semantic=reversed_semantic,
                identity=identity,
            )

        impossible_track = ShortTrackEvidence(
            track.candidate_id,
            timestamp_s=1.2,
            observation_count=track.observation_count,
            duration_s=0.5,
            stable=True,
            confidence=track.confidence,
        )
        with self.assertRaisesRegex(CandidateConfirmationError, "duration"):
            coordinator.evaluate(
                target_manager=manager,
                track=impossible_track,
                semantic=semantic,
                identity=identity,
            )

        impossible_identity = IdentityConsistencyEvidence(
            identity.candidate_id,
            identity.target_id,
            timestamp_s=identity.timestamp_s,
            reidentified=True,
            temporally_consistent=True,
            consistent_observations=track.observation_count + 1,
            confidence=identity.confidence,
        )
        with self.assertRaisesRegex(
            CandidateConfirmationError,
            "consistent_observations",
        ):
            coordinator.evaluate(
                target_manager=manager,
                track=track,
                semantic=semantic,
                identity=impossible_identity,
            )

    def test_types_and_policy_reject_nonfinite_or_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            DetectionCandidate("candidate", math.nan, 0.5)
        with self.assertRaises(ValueError):
            DetectionCandidate("candidate", 1.0, 1.1)
        with self.assertRaises(ValueError):
            SemanticVerification(
                "candidate", 1.0, "target", True, 0.5, "oracle"
            )
        with self.assertRaises(ValueError):
            ConfirmationPolicy(min_track_duration_s=-0.1)


if __name__ == "__main__":
    unittest.main()
