from __future__ import annotations

from dataclasses import replace
import json
import unittest
from unittest import mock

from agents.visual_review_coordinator import VisualReviewCoordinator
from models import AsyncModelResult, ModelResponse
from perception.visual_review import VisualReviewGate, VisualReviewMode
from runtime.events import MissionEventBus, MissionEventType
from runtime.frame_store import FrameStore
from runtime.review_scheduler import ReviewScheduler
from runtime.safety_supervisor import SafetyAction, SafetyDecision
from skills.types import SkillName
from target.types import TargetSpec
from tests.test_visual_review_coordinator import (
    FakeAsyncWorker,
    event,
    make_runtime,
    observation,
    target_snapshot,
    tick_coordinator,
)


def _coordinator(
    manager,
    worker: FakeAsyncWorker,
    *,
    intervals: dict[str, float] | None = None,
    frame_store: FrameStore | None = None,
    max_result_age_s: float = 10.0,
    debug_model_responses: bool | None = False,
    event_bus: MissionEventBus | None = None,
) -> VisualReviewCoordinator:
    return VisualReviewCoordinator(
        uav_id="uav_1",
        scheduler=ReviewScheduler(
            intervals_s=intervals or {"GOTO": 5.0, "TRACK": 5.0},
            cooldown_s=0.0,
        ),
        frame_store=(
            frame_store
            if frame_store is not None
            else FrameStore(max_frames=16, max_bytes=1_000_000, max_age_s=30.0)
        ),
        worker=worker,
        gate=VisualReviewGate(mode=VisualReviewMode.SHADOW),
        skill_manager=manager,
        event_bus=event_bus,
        review_timeout_s=20.0,
        max_result_age_s=max_result_age_s,
        debug_model_responses=debug_model_responses,
    )


def _valid_payload(request) -> dict[str, object]:
    return {
        "schema_version": 1,
        "review_id": request.review_id,
        "mission_id": request.mission_id,
        "uav_id": request.uav_id,
        "plan_version": request.plan_version,
        "observation_timestamp_s": request.observation_timestamp_s,
        "frame_id": request.frame_id,
        "decision": "NO_RELEVANT_CHANGE",
        "candidate": {
            "present": False,
            "bbox_xyxy_normalized": None,
            "description": None,
            "self_reported_confidence": None,
        },
        "scene_observations": [],
        "reason_codes": ["test"],
        "recommended_action": "CONTINUE",
    }


def _result(request, content: str) -> AsyncModelResult:
    return AsyncModelResult(
        request_id=request.request_id,
        review_id=request.review_id,
        mission_id=request.mission_id,
        uav_id=request.uav_id,
        plan_version=request.plan_version,
        observation_timestamp_s=request.observation_timestamp_s,
        frame_id=request.frame_id,
        response=ModelResponse(content, "fake-qwen", "stop", {}),
        error_code=None,
        error_message=None,
    )


class VisualReviewStaleReasonsTest(unittest.TestCase):
    def test_hold_established_transition_publishes_typed_event(self) -> None:
        manager, clock, uav = make_runtime()
        bus = MissionEventBus()
        coordinator = _coordinator(manager, FakeAsyncWorker(), event_bus=bus)
        manager.interrupt_with_hover(
            "review_path_blocked",
            defer_observation_timestamp_s=0.0,
        )
        manager.tick(observation(clock, uav))
        self.assertFalse(
            any(item.reason == "HOLD_ESTABLISHED" for item in manager.transition_log)
        )
        clock.time_s = 0.1
        manager.tick(observation(clock, uav))
        hold = next(
            item
            for item in reversed(manager.transition_log)
            if item.reason == "HOLD_ESTABLISHED"
        )

        published = coordinator.observe_skill_transition(hold)

        self.assertIsNotNone(published)
        assert published is not None
        self.assertIs(published.event_type, MissionEventType.HOLD_ESTABLISHED)
        self.assertEqual(bus.recent()[-1], published)

    def test_track_review_with_three_second_latency_is_valid(self) -> None:
        manager, clock, uav = make_runtime(SkillName.TRACK)
        worker = FakeAsyncWorker()
        coordinator = _coordinator(
            manager,
            worker,
            intervals={"TRACK": 5.0},
        )
        tick_coordinator(coordinator, manager, clock, uav)
        clock.time_s = 5.0
        tick_coordinator(coordinator, manager, clock, uav)
        self.assertEqual(len(worker.requests), 1)
        worker.complete_latest(decision="TARGET_MATCH")

        clock.time_s = 8.0
        tick_coordinator(coordinator, manager, clock, uav)

        record = coordinator.records[-1]
        self.assertFalse(record.stale)
        self.assertEqual(record.stale_reasons, ())
        self.assertEqual(record.decision, "TARGET_MATCH")
        self.assertIsNone(record.error_code)

    def test_cross_step_review_is_stale(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = _coordinator(manager, worker)
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest()
        clock.time_s = 0.1
        coordinator.tick(
            observation(clock, uav),
            mission_id="mission_1",
            plan_version=1,
            active_skill=SkillName.GOTO,
            active_step_id="search_next",
            target_spec=TargetSpec("moving target"),
            target_snapshot=target_snapshot(),
            safety_decision=SafetyDecision(SafetyAction.CONTINUE, "test"),
            mission_elapsed_s=clock.time_s,
        )

        record = coordinator.records[-1]
        self.assertTrue(record.stale)
        self.assertEqual(record.error_code, "STALE")
        self.assertEqual(record.stale_reasons, ("step_id_changed",))

    def test_plan_revision_invalidates_old_review(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = _coordinator(manager, worker)
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest()
        clock.time_s = 0.1
        coordinator.tick(
            observation(clock, uav),
            mission_id="mission_1",
            plan_version=2,
            active_skill=manager.active_name,
            active_step_id=manager.active_planned_step_id,
            target_spec=TargetSpec("moving target"),
            target_snapshot=target_snapshot(),
            safety_decision=SafetyDecision(SafetyAction.CONTINUE, "test"),
            mission_elapsed_s=clock.time_s,
        )

        self.assertEqual(
            coordinator.records[-1].stale_reasons,
            ("plan_version_changed",),
        )

    def test_frame_eviction_reports_specific_reason(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        frame_store = FrameStore(
            max_frames=1,
            max_bytes=100_000,
            max_age_s=30.0,
        )
        coordinator = _coordinator(
            manager,
            worker,
            frame_store=frame_store,
        )
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest()
        # Explicit invalidation remains observable even though ordinary ring
        # pressure may no longer evict an in-flight request's exact frame.
        frame_store.clear(uav_id="uav_1")
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)

        self.assertEqual(
            coordinator.records[-1].stale_reasons,
            ("frame_evicted",),
        )

    def test_request_and_worker_staleness_are_reported_independently(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = _coordinator(manager, worker)
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.complete_latest(stale=True)
        worker.results[0] = replace(worker.results[0], request_id="request_other")
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)

        self.assertEqual(
            coordinator.records[-1].stale_reasons,
            ("worker_marked_stale", "request_id_mismatch"),
        )

    def _record_for_content(
        self,
        content: str,
        *,
        debug: bool | None = False,
    ):
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = _coordinator(
            manager,
            worker,
            debug_model_responses=debug,
        )
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(coordinator, manager, clock, uav)
        worker.results.append(_result(worker.requests[-1], content))
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)
        return coordinator.records[-1]

    def test_model_and_parse_errors_have_specific_codes(self) -> None:
        manager, clock, uav = make_runtime()
        worker = FakeAsyncWorker()
        coordinator = _coordinator(manager, worker)
        coordinator.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(coordinator, manager, clock, uav)
        request = worker.requests[-1]
        worker.results.append(
            AsyncModelResult(
                request_id=request.request_id,
                review_id=request.review_id,
                mission_id=request.mission_id,
                uav_id=request.uav_id,
                plan_version=request.plan_version,
                observation_timestamp_s=request.observation_timestamp_s,
                frame_id=request.frame_id,
                response=None,
                error_code="MODEL_REQUEST_FAILED",
                error_message="ModelHTTPError",
            )
        )
        clock.time_s = 0.1
        tick_coordinator(coordinator, manager, clock, uav)
        self.assertEqual(coordinator.records[-1].error_code, "MODEL_REQUEST_FAILED")

        self.assertEqual(
            self._record_for_content("not JSON").error_code,
            "INVALID_JSON",
        )

        manager, clock2, uav2 = make_runtime()
        request_worker = FakeAsyncWorker()
        temporary = _coordinator(manager, request_worker)
        # Build a routed payload using a real submitted request.
        temporary.submit_event(event(MissionEventType.LOW_VISIBILITY))
        tick_coordinator(temporary, manager, clock2, uav2)
        valid = _valid_payload(request_worker.requests[-1])

        unsupported = dict(valid)
        unsupported["decision"] = "NOT_A_DECISION"
        self.assertEqual(
            self._record_for_content(json.dumps(unsupported)).error_code,
            "UNSUPPORTED_ENUM",
        )

        schema_invalid = dict(valid)
        schema_invalid.pop("candidate")
        self.assertEqual(
            self._record_for_content(json.dumps(schema_invalid)).error_code,
            "SCHEMA_INVALID",
        )

        duplicate = json.dumps(valid).replace(
            '"schema_version": 1,',
            '"schema_version": 1, "schema_version": 1,',
            1,
        )
        self.assertEqual(
            self._record_for_content(duplicate).error_code,
            "DUPLICATE_FIELD",
        )

        routing = dict(valid)
        routing["review_id"] = "review_wrong"
        self.assertEqual(
            self._record_for_content(json.dumps(routing)).error_code,
            "ROUTING_MISMATCH",
        )

    def test_debug_response_keeps_only_sanitized_bounded_tail(self) -> None:
        content = json.dumps(
            {
                "api_key": "TOP_SECRET",
                "authorization": "Bearer SECRET_TOKEN",
                "image": "data:image/jpeg;base64," + "A" * 2_000,
            }
        )
        record = self._record_for_content(content, debug=True)

        self.assertEqual(record.response_text_length, len(content))
        self.assertLessEqual(len(record.response_text_tail or ""), 500)
        serialized = json.dumps(record.to_dict()).casefold()
        for forbidden in (
            "top_secret",
            "secret_token",
            "base64,",
            "data:image",
            "api_key",
            "authorization",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_debug_response_can_be_enabled_by_environment(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"UAV_AGENT_DEBUG_VISUAL_REVIEW": "1"},
            clear=False,
        ):
            record = self._record_for_content("not JSON", debug=None)
        self.assertEqual(record.response_text_length, len("not JSON"))
        self.assertEqual(record.response_text_tail, "not JSON")


if __name__ == "__main__":
    unittest.main()
