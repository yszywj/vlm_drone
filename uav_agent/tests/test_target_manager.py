from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import unittest

from target import (
    TargetEvent,
    TargetLifecycle,
    TargetManager,
    TargetSnapshot,
    TargetSpec,
    TargetStateError,
)


def searching_manager() -> TargetManager:
    manager = TargetManager()
    manager.start_search(TargetSpec("moving red target"), timestamp_s=1.0)
    return manager


def tracking_manager() -> TargetManager:
    manager = searching_manager()
    manager.lock_oracle_from_search(
        "target_0",
        timestamp_s=2.0,
        confidence=1.0,
        last_seen_position=(4, 5, 0),
        last_seen_velocity=(0.5, 0, 0),
    )
    manager.start_tracking(timestamp_s=3.0)
    return manager


class TargetTypesTest(unittest.TestCase):
    def test_initial_snapshot_uses_non_empty_semantic_placeholder(self) -> None:
        snapshot = TargetManager().snapshot()

        self.assertEqual(snapshot.lifecycle, TargetLifecycle.UNINITIALIZED)
        self.assertEqual(snapshot.description, "uninitialized")
        self.assertIsNone(snapshot.target_id)

    def test_spec_normalizes_description_and_is_json_compatible(self) -> None:
        spec = TargetSpec("  moving target  ")

        self.assertEqual(spec.description, "moving target")
        self.assertEqual(json.loads(json.dumps(spec.to_dict())), spec.to_dict())
        with self.assertRaises(FrozenInstanceError):
            spec.description = "changed"  # type: ignore[misc]

    def test_string_fields_must_be_non_empty(self) -> None:
        with self.assertRaises(ValueError):
            TargetSpec(" \t ")
        with self.assertRaises(ValueError):
            TargetSnapshot(
                target_id=" ",
                description="target",
                lifecycle=TargetLifecycle.LOCKED,
                confidence=None,
                last_seen_position=None,
                last_seen_velocity=None,
                last_seen_time_s=None,
                source="oracle",
            )
        with self.assertRaises(ValueError):
            TargetEvent(
                timestamp_s=0,
                old_state=TargetLifecycle.SEARCHING,
                new_state=TargetLifecycle.LOCKED,
                reason=" ",
            )
        with self.assertRaises(ValueError):
            TargetSnapshot(
                target_id=None,
                description="",
                lifecycle=TargetLifecycle.UNINITIALIZED,
                confidence=None,
                last_seen_position=None,
                last_seen_velocity=None,
                last_seen_time_s=None,
                source=None,
            )

    def test_snapshot_normalizes_vectors_and_json_representation(self) -> None:
        snapshot = TargetSnapshot(
            target_id="target_0",
            description="target",
            lifecycle=TargetLifecycle.LOCKED,
            confidence=0.75,
            last_seen_position=[1, 2, 3],  # type: ignore[arg-type]
            last_seen_velocity=[0.1, 0.2, 0.3],  # type: ignore[arg-type]
            last_seen_time_s=4,
            source="detector",
        )

        self.assertEqual(snapshot.last_seen_position, (1.0, 2.0, 3.0))
        self.assertEqual(snapshot.last_seen_velocity, (0.1, 0.2, 0.3))
        encoded = json.loads(json.dumps(snapshot.to_dict()))
        self.assertEqual(encoded["lifecycle"], "LOCKED")
        self.assertEqual(encoded["last_seen_position"], [1.0, 2.0, 3.0])

    def test_numeric_fields_reject_bool_nan_inf_and_bad_vectors(self) -> None:
        valid = dict(
            target_id="target_0",
            description="target",
            lifecycle=TargetLifecycle.LOCKED,
            confidence=0.5,
            last_seen_position=(1, 2, 3),
            last_seen_velocity=(0, 0, 0),
            last_seen_time_s=1,
            source="oracle",
        )
        for field_name, value in (
            ("confidence", True),
            ("confidence", math.nan),
            ("confidence", math.inf),
            ("last_seen_time_s", True),
            ("last_seen_time_s", math.nan),
            ("last_seen_position", (1, 2)),
            ("last_seen_position", (1, math.inf, 3)),
            ("last_seen_velocity", (0, True, 0)),
        ):
            values = dict(valid)
            values[field_name] = value
            with self.subTest(field=field_name, value=value), self.assertRaises(
                (TypeError, ValueError)
            ):
                TargetSnapshot(**values)

    def test_confidence_range_is_inclusive(self) -> None:
        for confidence in (0, 1):
            manager = searching_manager()
            manager.lock_oracle_from_search(
                "target", timestamp_s=2, confidence=confidence
            )
            self.assertEqual(manager.snapshot().confidence, float(confidence))

        for confidence in (-0.01, 1.01):
            manager = searching_manager()
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                manager.lock_oracle_from_search(
                    "target", timestamp_s=2, confidence=confidence
                )
            self.assertEqual(manager.lifecycle, TargetLifecycle.SEARCHING)


class TargetManagerTest(unittest.TestCase):
    def test_oracle_search_lock_tracking_path(self) -> None:
        manager = searching_manager()

        manager.lock_oracle_from_search(
            "target_0", timestamp_s=2, confidence=1.0
        )
        manager.start_tracking(timestamp_s=3)

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.lifecycle, TargetLifecycle.TRACKING)
        self.assertEqual(snapshot.target_id, "target_0")
        self.assertEqual(snapshot.confidence, 1.0)
        self.assertEqual(snapshot.source, "oracle")
        self.assertEqual(snapshot.last_seen_time_s, 2.0)
        self.assertEqual(
            [event.new_state for event in manager.events()],
            [
                TargetLifecycle.SEARCHING,
                TargetLifecycle.LOCKED,
                TargetLifecycle.TRACKING,
            ],
        )

    def test_candidate_path(self) -> None:
        manager = searching_manager()
        manager.set_candidate(
            "candidate_0",
            timestamp_s=1.5,
            confidence=0.6,
            source="detector",
            last_seen_position=(1, 2, 0),
        )
        candidate = manager.snapshot()

        self.assertEqual(candidate.lifecycle, TargetLifecycle.CANDIDATE)
        self.assertEqual(candidate.source, "detector")
        with self.assertRaisesRegex(TargetStateError, "ConfirmationCoordinator"):
            manager.lock(
                "candidate_0",
                timestamp_s=2,
                confidence=0.9,
                source="vlm_verifier",
            )
        self.assertEqual(manager.lifecycle, TargetLifecycle.CANDIDATE)

    def test_visual_lock_cannot_bypass_candidate_confirmation(self) -> None:
        manager = searching_manager()

        with self.assertRaisesRegex(TargetStateError, "ConfirmationCoordinator"):
            manager.lock(
                "target_0",
                timestamp_s=2,
                confidence=0.9,
                source="detector",
            )

        self.assertIs(manager.lifecycle, TargetLifecycle.SEARCHING)
        self.assertEqual(len(manager.events()), 1)

        manager.set_candidate(
            "candidate_0",
            timestamp_s=2.1,
            confidence=0.8,
            source="detector",
        )
        with self.assertRaisesRegex(TargetStateError, "ConfirmationCoordinator"):
            manager.lock(
                "target_0",
                timestamp_s=2.2,
                confidence=0.9,
                source="confirmed_vision",
            )
        self.assertIs(manager.lifecycle, TargetLifecycle.CANDIDATE)

    def test_lost_reacquire_lock_track_path(self) -> None:
        manager = tracking_manager()

        manager.mark_lost(
            timestamp_s=5,
            last_seen_position=(6, 5, 0),
            last_seen_velocity=(0.5, 0, 0),
            last_seen_time_s=4.5,
        )
        manager.start_reacquiring(timestamp_s=5.1)
        manager.mark_reacquired_oracle(
            timestamp_s=6,
            confidence=1.0,
            last_seen_position=(6.75, 5, 0),
        )
        manager.start_tracking(timestamp_s=6.1)

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.lifecycle, TargetLifecycle.TRACKING)
        self.assertEqual(snapshot.target_id, "target_0")
        self.assertEqual(snapshot.last_seen_position, (6.75, 5.0, 0.0))
        self.assertEqual(snapshot.last_seen_velocity, (0.5, 0.0, 0.0))
        self.assertEqual(snapshot.last_seen_time_s, 6.0)

    def test_mark_lost_keeps_existing_last_seen_values_when_omitted(self) -> None:
        manager = tracking_manager()

        manager.mark_lost(timestamp_s=4)

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.last_seen_position, (4.0, 5.0, 0.0))
        self.assertEqual(snapshot.last_seen_velocity, (0.5, 0.0, 0.0))
        self.assertEqual(snapshot.last_seen_time_s, 2.0)

    def test_mark_lost_new_vectors_without_sample_time_use_event_time(self) -> None:
        manager = tracking_manager()

        manager.mark_lost(
            timestamp_s=4,
            last_seen_position=(5, 5, 0),
            last_seen_velocity=(1, 0, 0),
        )

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.last_seen_position, (5.0, 5.0, 0.0))
        self.assertEqual(snapshot.last_seen_velocity, (1.0, 0.0, 0.0))
        self.assertEqual(snapshot.last_seen_time_s, 4.0)

    def test_every_invalid_transition_raises_clear_state_error(self) -> None:
        manager = TargetManager()
        with self.assertRaisesRegex(TargetStateError, "UNINITIALIZED -> TRACKING"):
            manager.start_tracking(timestamp_s=0)

        manager.start_search(TargetSpec("target"), timestamp_s=1)
        with self.assertRaisesRegex(TargetStateError, "SEARCHING -> TRACKING"):
            manager.start_tracking(timestamp_s=2)
        with self.assertRaisesRegex(TargetStateError, "SEARCHING -> REACQUIRING"):
            manager.start_reacquiring(timestamp_s=2)
        with self.assertRaisesRegex(TargetStateError, "SEARCHING -> LOST"):
            manager.mark_lost(timestamp_s=2)

    def test_timestamps_are_finite_monotonic_and_equal_is_allowed(self) -> None:
        manager = searching_manager()
        manager.lock_oracle_from_search("target", timestamp_s=1)
        self.assertEqual(manager.lifecycle, TargetLifecycle.LOCKED)

        with self.assertRaisesRegex(TargetStateError, "cannot move backward"):
            manager.start_tracking(timestamp_s=0.9)
        self.assertEqual(manager.lifecycle, TargetLifecycle.LOCKED)

        for timestamp in (True, math.nan, math.inf):
            fresh = searching_manager()
            with self.subTest(timestamp=timestamp), self.assertRaises(
                (TypeError, ValueError)
            ):
                fresh.lock_oracle_from_search("target", timestamp_s=timestamp)
            self.assertEqual(fresh.lifecycle, TargetLifecycle.SEARCHING)

    def test_invalid_inputs_do_not_partially_mutate_state(self) -> None:
        manager = searching_manager()
        before = manager.snapshot()
        events_before = manager.events()

        with self.assertRaises(ValueError):
            manager.set_candidate(
                "target",
                timestamp_s=1.5,
                confidence=0.5,
                source=" ",
            )

        self.assertEqual(manager.snapshot(), before)
        self.assertEqual(manager.events(), events_before)

    def test_last_seen_time_cannot_be_after_loss_event(self) -> None:
        manager = tracking_manager()

        with self.assertRaises(ValueError):
            manager.mark_lost(timestamp_s=4, last_seen_time_s=4.1)

        self.assertEqual(manager.lifecycle, TargetLifecycle.TRACKING)

    def test_last_seen_time_cannot_move_backward_and_failure_is_atomic(self) -> None:
        manager = tracking_manager()
        before = manager.snapshot()
        events_before = manager.events()

        with self.assertRaisesRegex(TargetStateError, "cannot move backward"):
            manager.mark_lost(
                timestamp_s=4,
                last_seen_position=(99, 99, 99),
                last_seen_velocity=(9, 9, 9),
                last_seen_time_s=1.5,
            )

        self.assertEqual(manager.snapshot(), before)
        self.assertEqual(manager.events(), events_before)

    def test_terminate_then_reset(self) -> None:
        manager = tracking_manager()
        manager.terminate(timestamp_s=4, reason="mission complete")

        terminated = manager.snapshot()
        self.assertEqual(terminated.lifecycle, TargetLifecycle.TERMINATED)
        self.assertEqual(manager.events()[-1].reason, "mission complete")
        with self.assertRaises(TargetStateError):
            manager.start_tracking(timestamp_s=5)

        manager.reset()

        self.assertEqual(
            manager.snapshot(),
            TargetSnapshot(
                target_id=None,
                description="uninitialized",
                lifecycle=TargetLifecycle.UNINITIALIZED,
                confidence=None,
                last_seen_position=None,
                last_seen_velocity=None,
                last_seen_time_s=None,
                source=None,
            ),
        )
        self.assertEqual(manager.events(), ())
        manager.start_search(TargetSpec("new target"), timestamp_s=0)
        self.assertEqual(manager.lifecycle, TargetLifecycle.SEARCHING)

    def test_uninitialized_mission_can_terminate_without_fake_search(self) -> None:
        fresh = TargetManager()
        fresh.terminate(timestamp_s=0, reason="task canceled before search")

        snapshot = fresh.snapshot()
        self.assertEqual(snapshot.lifecycle, TargetLifecycle.TERMINATED)
        self.assertEqual(snapshot.description, "uninitialized")
        self.assertIsNone(snapshot.target_id)
        self.assertEqual(
            [event.old_state for event in fresh.events()],
            [TargetLifecycle.UNINITIALIZED],
        )
        self.assertNotIn(
            TargetLifecycle.SEARCHING,
            [event.new_state for event in fresh.events()],
        )

    def test_terminate_requires_non_empty_reason(self) -> None:

        manager = searching_manager()
        with self.assertRaises(ValueError):
            manager.terminate(timestamp_s=2, reason=" ")
        self.assertEqual(manager.lifecycle, TargetLifecycle.SEARCHING)

    def test_reset_requires_terminated_state(self) -> None:
        with self.assertRaisesRegex(TargetStateError, "reset requires"):
            TargetManager().reset()

    def test_snapshot_and_event_history_are_defensive_readonly_snapshots(self) -> None:
        manager = searching_manager()
        snapshot_a = manager.snapshot()
        events_a = manager.events()

        manager.lock_oracle_from_search(
            "target_0", timestamp_s=2, confidence=1.0
        )

        self.assertEqual(snapshot_a.lifecycle, TargetLifecycle.SEARCHING)
        self.assertEqual(len(events_a), 1)
        self.assertEqual(len(manager.events()), 2)
        with self.assertRaises(FrozenInstanceError):
            snapshot_a.lifecycle = TargetLifecycle.LOCKED  # type: ignore[misc]
        with self.assertRaises(TypeError):
            events_a[0] = manager.events()[0]  # type: ignore[index]

        payload = {
            "snapshot": manager.snapshot().to_dict(),
            "events": [event.to_dict() for event in manager.events()],
        }
        self.assertEqual(json.loads(json.dumps(payload)), payload)


if __name__ == "__main__":
    unittest.main()
