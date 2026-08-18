from __future__ import annotations

import json
from queue import Queue
from threading import Thread
import unittest

import numpy as np

from runtime.events import (
    EventSeverity,
    MissionEvent,
    MissionEventType,
)
from runtime.frame_store import FrameRef
from runtime.world_belief import (
    CandidateSummary,
    QwenRequestState,
    QwenRequestStatus,
    WorldBelief,
    WorldBeliefStore,
    WorldBeliefThreadError,
)
from target.types import TargetLifecycle, TargetSnapshot, TargetSpec


def _event(*, uav_id: str = "uav_1", plan_version: int = 1) -> MissionEvent:
    return MissionEvent(
        event_id="event_1",
        mission_id="mission_1",
        uav_id=uav_id,
        plan_version=plan_version,
        timestamp_s=2,
        event_type=MissionEventType.TRACK_CONFIDENCE_DROP,
        severity=EventSeverity.WARNING,
        payload={"confidence": 0.35},
    )


def _target_snapshot() -> TargetSnapshot:
    return TargetSnapshot(
        target_id="target_1",
        description="red vehicle",
        lifecycle=TargetLifecycle.TRACKING,
        confidence=0.8,
        last_seen_position=None,
        last_seen_velocity=None,
        last_seen_time_s=2,
        source="detector",
    )


def _belief(**overrides: object) -> WorldBelief:
    values: dict[str, object] = {
        "mission_id": "mission_1",
        "uav_id": "uav_1",
        "plan_id": "plan_1",
        "plan_version": 1,
        "current_step_id": "track_1",
        "current_skill": "TRACK",
        "skill_feedback": {"target_visible": True, "distance_error": 0.4},
        "target_spec": TargetSpec("red vehicle"),
        "target_snapshot": _target_snapshot(),
        "candidate_summaries": (
            CandidateSummary("candidate_1", 0.7, 1.5, "qwen_vl", 3),
        ),
        "recent_events": (_event(),),
        "qwen_request_status": QwenRequestStatus(
            state=QwenRequestState.IN_FLIGHT,
            request_id="request_1",
            review_id="review_1",
            blocking=False,
            submitted_timestamp_s=2,
        ),
        "latest_frame_ref": FrameRef("uav_1", "frame_1", 2, 640, 480),
        "mission_elapsed_s": 2.5,
    }
    values.update(overrides)
    return WorldBelief(**values)


class WorldBeliefTest(unittest.TestCase):
    def test_snapshot_contains_only_values_and_frame_metadata(self) -> None:
        belief = _belief()
        encoded = belief.to_dict()
        self.assertEqual(encoded["mission_id"], "mission_1")
        self.assertEqual(encoded["uav_id"], "uav_1")
        self.assertEqual(encoded["plan_version"], 1)
        self.assertEqual(encoded["current_step_id"], "track_1")
        self.assertEqual(encoded["current_skill"], "TRACK")
        self.assertEqual(
            encoded["target_spec"]["original_description"],
            "red vehicle",
        )
        self.assertEqual(encoded["latest_frame_ref"]["frame_id"], "frame_1")
        serialized = json.dumps(encoded, allow_nan=False)
        for forbidden in ("controller", "environment", "model_client", "rgb"):
            self.assertNotIn(forbidden, serialized.casefold())

    def test_feedback_is_defensive_and_rejects_runtime_objects_or_arrays(self) -> None:
        feedback = {"nested": {"count": 1}}
        belief = _belief(skill_feedback=feedback)
        feedback["nested"]["count"] = 99
        self.assertEqual(
            belief.to_dict()["skill_feedback"],
            {"nested": {"count": 1}},
        )

        class FakeController:
            pass

        for value in (
            {"controller": FakeController()},
            {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
        ):
            with self.subTest(value=value), self.assertRaises(TypeError):
                _belief(skill_feedback=value)

    def test_cross_uav_and_future_event_routes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "event routing"):
            _belief(recent_events=(_event(uav_id="uav_2"),))
        with self.assertRaisesRegex(ValueError, "future plan"):
            _belief(recent_events=(_event(plan_version=2),))
        with self.assertRaisesRegex(ValueError, "FrameRef"):
            _belief(latest_frame_ref=FrameRef("uav_2", "frame_1", 2, 640, 480))

    def test_target_identity_description_cannot_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "match TargetSpec"):
            _belief(target_spec=TargetSpec("blue vehicle"))

    def test_request_status_is_strict_and_idle_contains_no_request(self) -> None:
        idle = QwenRequestStatus()
        self.assertEqual(idle.state, QwenRequestState.IDLE)
        with self.assertRaises(ValueError):
            QwenRequestStatus(state=QwenRequestState.IDLE, request_id="request_1")
        with self.assertRaises((TypeError, ValueError)):
            QwenRequestStatus(state=QwenRequestState.IN_FLIGHT)

    def test_candidate_and_event_histories_are_bounded(self) -> None:
        candidates = tuple(
            CandidateSummary(f"candidate_{index}", None, 1, "qwen_vl")
            for index in range(WorldBelief.MAX_CANDIDATE_SUMMARIES + 1)
        )
        with self.assertRaisesRegex(ValueError, "bounded"):
            _belief(candidate_summaries=candidates)
        events = tuple(
            MissionEvent(
                event_id=f"event_{index}",
                mission_id="mission_1",
                uav_id="uav_1",
                plan_version=1,
                timestamp_s=index,
                event_type=MissionEventType.LOW_VISIBILITY,
                severity=EventSeverity.INFO,
                payload={},
            )
            for index in range(WorldBelief.MAX_RECENT_EVENTS + 1)
        )
        with self.assertRaisesRegex(ValueError, "bounded"):
            _belief(recent_events=events)

    def test_candidate_summary_requires_explicit_nonempty_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "source"):
            CandidateSummary("candidate_1", None, 1.0, "")


class WorldBeliefStoreTest(unittest.TestCase):
    def test_owner_thread_updates_are_serial_and_plan_version_is_monotonic(self) -> None:
        store = WorldBeliefStore(_belief())
        updated = store.update(mission_elapsed_s=3, plan_version=2)
        self.assertEqual(updated.plan_version, 2)
        self.assertEqual(store.snapshot().mission_elapsed_s, 3.0)
        with self.assertRaisesRegex(ValueError, "must not decrease"):
            store.update(plan_version=1)
        with self.assertRaisesRegex(ValueError, "routing IDs are immutable"):
            store.update(uav_id="uav_2")

    def test_worker_thread_cannot_modify_belief(self) -> None:
        store = WorldBeliefStore(_belief())
        results: Queue[BaseException | None] = Queue()

        def worker() -> None:
            try:
                store.update(mission_elapsed_s=5)
            except BaseException as exc:
                results.put(exc)
            else:
                results.put(None)

        thread = Thread(target=worker)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        error = results.get_nowait()
        self.assertIsInstance(error, WorldBeliefThreadError)
        self.assertEqual(store.snapshot().mission_elapsed_s, 2.5)


if __name__ == "__main__":
    unittest.main()
