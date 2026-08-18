from __future__ import annotations

import json
import math
import unittest

import numpy as np

from runtime.events import (
    EventSeverity,
    MissionEvent,
    MissionEventBus,
    MissionEventType,
)


def _event(
    index: int,
    *,
    uav_id: str = "uav_1",
    mission_id: str = "mission_1",
    event_type: MissionEventType = MissionEventType.LOW_VISIBILITY,
    payload: object = None,
) -> MissionEvent:
    return MissionEvent(
        event_id=f"event_{index}",
        mission_id=mission_id,
        uav_id=uav_id,
        plan_version=1,
        timestamp_s=float(index),
        event_type=event_type,
        severity=EventSeverity.WARNING,
        payload={} if payload is None else payload,
    )


class MissionEventTest(unittest.TestCase):
    def test_event_has_stable_routing_and_json_representation(self) -> None:
        event = _event(
            1,
            payload={
                "candidate_ids": ["candidate_1", "candidate_2"],
                "confidence": 0.4,
                "persistent": True,
            },
        )
        encoded = event.to_dict()
        self.assertEqual(encoded["event_id"], "event_1")
        self.assertEqual(encoded["mission_id"], "mission_1")
        self.assertEqual(encoded["uav_id"], "uav_1")
        self.assertEqual(encoded["plan_version"], 1)
        self.assertEqual(encoded["event_type"], "LOW_VISIBILITY")
        self.assertEqual(encoded["severity"], "WARNING")
        json.dumps(encoded, allow_nan=False)

    def test_payload_is_defensive_bounded_json_and_never_an_image(self) -> None:
        source = {"nested": {"values": [1, 2]}}
        event = _event(1, payload=source)
        source["nested"]["values"].append(3)
        self.assertEqual(
            event.to_dict()["payload"],
            {"nested": {"values": [1, 2]}},
        )
        with self.assertRaises(TypeError):
            event.payload["new"] = "value"  # type: ignore[index]

        invalid_payloads = (
            np.zeros((2, 2, 3), dtype=np.uint8),
            {"rgb": np.zeros((1, 1, 3), dtype=np.uint8)},
            {"bytes": b"raw pixels"},
            {"nan": math.nan},
            {1: "non-string key"},
            {"object": object()},
        )
        for index, payload in enumerate(invalid_payloads):
            with self.subTest(index=index), self.assertRaises(
                (TypeError, ValueError)
            ):
                _event(index + 10, payload=payload)

        with self.assertRaises(ValueError):
            _event(20, payload={"text": "x" * 70_000})

    def test_all_required_event_types_are_supported(self) -> None:
        required = {
            "PERIODIC_REVIEW_DUE",
            "CANDIDATE_PERSISTENT",
            "MULTIPLE_CANDIDATES",
            "TARGET_IDENTITY_UNCERTAIN",
            "TRACK_CONFIDENCE_DROP",
            "TRACK_LOST",
            "PATH_BLOCKED",
            "SKILL_PROGRESS_STALLED",
            "LOW_VISIBILITY",
            "MODEL_REVIEW_STARTED",
            "MODEL_REVIEW_COMPLETED",
            "MODEL_REVIEW_TIMEOUT",
            "MODEL_RESPONSE_STALE",
            "PLAN_REVISION_REQUESTED",
            "PLAN_REVISION_ACCEPTED",
            "PLAN_REVISION_REJECTED",
        }
        self.assertTrue(required <= {item.value for item in MissionEventType})

    def test_invalid_routing_time_and_payload_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _event(1, uav_id="uav bad")
        with self.assertRaises(ValueError):
            MissionEvent(
                "event_1",
                "mission_1",
                "uav_1",
                0,
                1,
                MissionEventType.TRACK_LOST,
                EventSeverity.ERROR,
                {},
            )
        with self.assertRaises(TypeError):
            _event(1, payload=[])


class MissionEventBusTest(unittest.TestCase):
    def test_bus_is_bounded_and_filtering_preserves_uav_routes(self) -> None:
        bus = MissionEventBus(max_events=2)
        first = _event(1, uav_id="uav_1")
        second = _event(2, uav_id="uav_2")
        third = _event(3, uav_id="uav_1")
        for event in (first, second, third):
            bus.publish(event)

        self.assertEqual(len(bus), 2)
        self.assertEqual(bus.recent(), (second, third))
        self.assertEqual(bus.recent(uav_id="uav_1"), (third,))
        self.assertEqual(bus.recent(uav_id="uav_2"), (second,))
        self.assertEqual(bus.recent(limit=1), (third,))

    def test_bus_rejects_duplicate_ids_until_evicted(self) -> None:
        bus = MissionEventBus(max_events=1)
        first = _event(1)
        bus.publish(first)
        with self.assertRaisesRegex(ValueError, "already"):
            bus.publish(first)
        bus.publish(_event(2))
        bus.publish(first)  # its prior occurrence has left the bounded history
        self.assertEqual(bus.recent(), (first,))


if __name__ == "__main__":
    unittest.main()
