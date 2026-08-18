from __future__ import annotations

import unittest

from runtime.events import (
    EventSeverity,
    MissionEvent,
    MissionEventType,
)
from runtime.review_scheduler import (
    ReviewScheduleReason,
    ReviewScheduler,
    ReviewTrigger,
)


def _event(
    index: int,
    event_type: MissionEventType,
    *,
    uav_id: str = "uav_1",
    mission_id: str = "mission_1",
    plan_version: int = 1,
    timestamp_s: float | None = None,
) -> MissionEvent:
    return MissionEvent(
        event_id=f"event_{index}",
        mission_id=mission_id,
        uav_id=uav_id,
        plan_version=plan_version,
        timestamp_s=float(index) if timestamp_s is None else timestamp_s,
        event_type=event_type,
        severity=EventSeverity.WARNING,
        payload={},
    )


def _schedule(
    scheduler: ReviewScheduler,
    *,
    timestamp_s: float,
    uav_id: str = "uav_1",
    event: MissionEvent | None = None,
    skill_name: str = "GOTO",
):
    return scheduler.schedule(
        mission_id="mission_1",
        uav_id=uav_id,
        plan_version=1,
        skill_name=skill_name,
        timestamp_s=timestamp_s,
        event=event,
        request_id=f"request_{uav_id}_{timestamp_s:g}".replace(".", "_"),
        review_id=f"review_{uav_id}_{timestamp_s:g}".replace(".", "_"),
    )


class ReviewSchedulerTest(unittest.TestCase):
    def test_default_periods_and_periodic_review_are_non_blocking(self) -> None:
        scheduler = ReviewScheduler(cooldown_s=0)
        self.assertEqual(
            scheduler.intervals_s,
            {"GOTO": 5.0, "SEARCH": 2.0, "INSPECT": 1.0, "TRACK": 5.0},
        )
        self.assertEqual(
            _schedule(scheduler, timestamp_s=0).reason,
            ReviewScheduleReason.NOT_DUE,
        )
        self.assertEqual(
            _schedule(scheduler, timestamp_s=4.99).reason,
            ReviewScheduleReason.NOT_DUE,
        )
        decision = _schedule(scheduler, timestamp_s=5)
        self.assertTrue(decision.should_submit)
        self.assertFalse(decision.blocking)
        self.assertFalse(decision.hover_required)
        assert decision.ticket is not None
        self.assertEqual(decision.ticket.trigger, ReviewTrigger.PERIODIC)

    def test_blocking_event_requires_hover_but_normal_event_does_not(self) -> None:
        blocking_scheduler = ReviewScheduler(cooldown_s=0)
        blocking = _schedule(
            blocking_scheduler,
            timestamp_s=0,
            event=_event(
                1,
                MissionEventType.MULTIPLE_CANDIDATES,
                timestamp_s=0,
            ),
            skill_name="SEARCH",
        )
        self.assertTrue(blocking.should_submit)
        self.assertTrue(blocking.blocking)
        self.assertTrue(blocking.hover_required)
        assert blocking.ticket is not None
        self.assertEqual(blocking.ticket.trigger, ReviewTrigger.EVENT)

        nonblocking_scheduler = ReviewScheduler(cooldown_s=0)
        nonblocking = _schedule(
            nonblocking_scheduler,
            timestamp_s=0,
            event=_event(
                2,
                MissionEventType.LOW_VISIBILITY,
                timestamp_s=0,
            ),
            skill_name="TRACK",
        )
        self.assertTrue(nonblocking.should_submit)
        self.assertFalse(nonblocking.blocking)
        self.assertFalse(nonblocking.hover_required)

    def test_default_blocking_rules_cover_trusted_pause_cases(self) -> None:
        for index, event_type in enumerate(
            (
                MissionEventType.CANDIDATE_PERSISTENT,
                MissionEventType.MULTIPLE_CANDIDATES,
                MissionEventType.TARGET_CONFIRMATION_REQUIRED,
                MissionEventType.PATH_BLOCKED,
                MissionEventType.TARGET_IDENTITY_UNCERTAIN,
                MissionEventType.TASK_COMPLETION_UNCERTAIN,
            )
        ):
            with self.subTest(event_type=event_type):
                scheduler = ReviewScheduler(cooldown_s=0)
                event = _event(index + 1, event_type, timestamp_s=0)
                self.assertTrue(
                    _schedule(
                        scheduler,
                        timestamp_s=0,
                        event=event,
                        skill_name="SEARCH",
                    ).hover_required
                )

    def test_one_inflight_request_per_uav_and_different_uavs_are_isolated(self) -> None:
        scheduler = ReviewScheduler(cooldown_s=0)
        first_event = _event(
            1,
            MissionEventType.LOW_VISIBILITY,
            uav_id="uav_1",
            timestamp_s=0,
        )
        first = _schedule(scheduler, timestamp_s=0, event=first_event)
        self.assertTrue(first.should_submit)
        second = _schedule(
            scheduler,
            timestamp_s=0,
            event=_event(
                2,
                MissionEventType.TRACK_LOST,
                uav_id="uav_1",
                timestamp_s=0,
            ),
        )
        self.assertEqual(second.reason, ReviewScheduleReason.IN_FLIGHT)

        other = _schedule(
            scheduler,
            timestamp_s=0,
            uav_id="uav_2",
            event=_event(
                3,
                MissionEventType.LOW_VISIBILITY,
                uav_id="uav_2",
                timestamp_s=0,
            ),
        )
        self.assertTrue(other.should_submit)
        assert first.ticket is not None and other.ticket is not None
        self.assertEqual(scheduler.inflight(uav_id="uav_1"), first.ticket)
        self.assertEqual(scheduler.inflight(uav_id="uav_2"), other.ticket)
        self.assertNotEqual(first.ticket.request_id, other.ticket.request_id)

    def test_completion_ids_cannot_cross_uav_or_request(self) -> None:
        scheduler = ReviewScheduler(cooldown_s=0)
        decision = _schedule(
            scheduler,
            timestamp_s=0,
            event=_event(
                1,
                MissionEventType.LOW_VISIBILITY,
                timestamp_s=0,
            ),
        )
        assert decision.ticket is not None
        ticket = decision.ticket
        with self.assertRaises(ValueError):
            scheduler.mark_completed(
                uav_id="uav_2",
                request_id=ticket.request_id,
                review_id=ticket.review_id,
                timestamp_s=1,
            )
        with self.assertRaisesRegex(ValueError, "do not match"):
            scheduler.mark_completed(
                uav_id="uav_1",
                request_id="request_wrong",
                review_id=ticket.review_id,
                timestamp_s=1,
            )
        self.assertEqual(scheduler.inflight(uav_id="uav_1"), ticket)
        completed = scheduler.mark_completed(
            uav_id="uav_1",
            request_id=ticket.request_id,
            review_id=ticket.review_id,
            timestamp_s=1,
        )
        self.assertEqual(completed, ticket)
        self.assertIsNone(scheduler.inflight(uav_id="uav_1"))

    def test_cooldown_suppresses_review_storms(self) -> None:
        scheduler = ReviewScheduler(cooldown_s=2)
        first = _schedule(
            scheduler,
            timestamp_s=0,
            event=_event(
                1,
                MissionEventType.LOW_VISIBILITY,
                timestamp_s=0,
            ),
        )
        assert first.ticket is not None
        scheduler.mark_completed(
            uav_id="uav_1",
            request_id=first.ticket.request_id,
            review_id=first.ticket.review_id,
            timestamp_s=0.1,
        )
        retry_event = _event(
            2,
            MissionEventType.TRACK_CONFIDENCE_DROP,
            timestamp_s=0.2,
        )
        self.assertEqual(
            _schedule(
                scheduler,
                timestamp_s=0.2,
                event=retry_event,
            ).reason,
            ReviewScheduleReason.COOLDOWN,
        )
        self.assertTrue(
            _schedule(
                scheduler,
                timestamp_s=2.1,
                event=retry_event,
            ).should_submit
        )

    def test_event_routing_and_time_are_fail_closed(self) -> None:
        scheduler = ReviewScheduler(cooldown_s=0)
        with self.assertRaisesRegex(ValueError, "routing IDs"):
            _schedule(
                scheduler,
                timestamp_s=0,
                event=_event(
                    1,
                    MissionEventType.LOW_VISIBILITY,
                    uav_id="uav_2",
                    timestamp_s=0,
                ),
            )
        with self.assertRaisesRegex(ValueError, "plan_version"):
            _schedule(
                scheduler,
                timestamp_s=0,
                event=_event(
                    2,
                    MissionEventType.LOW_VISIBILITY,
                    plan_version=2,
                    timestamp_s=0,
                ),
            )

        self.assertEqual(
            _schedule(
                scheduler,
                timestamp_s=0,
                event=_event(
                    3,
                    MissionEventType.MODEL_REVIEW_COMPLETED,
                    timestamp_s=0,
                ),
            ).reason,
            ReviewScheduleReason.NOT_DUE,
        )
        _schedule(scheduler, timestamp_s=1)
        with self.assertRaisesRegex(ValueError, "backwards"):
            _schedule(scheduler, timestamp_s=0.5)

    def test_skill_switch_restarts_periodic_interval(self) -> None:
        scheduler = ReviewScheduler(cooldown_s=0)
        self.assertFalse(_schedule(scheduler, timestamp_s=0).should_submit)
        self.assertFalse(
            _schedule(
                scheduler,
                timestamp_s=4,
                skill_name="SEARCH",
            ).should_submit
        )
        self.assertFalse(
            _schedule(
                scheduler,
                timestamp_s=5,
                skill_name="TRACK",
            ).should_submit
        )
        self.assertTrue(
            _schedule(
                scheduler,
                timestamp_s=10,
                skill_name="TRACK",
            ).should_submit
        )


if __name__ == "__main__":
    unittest.main()
