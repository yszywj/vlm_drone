"""Pure-Python scenario F/G coverage for asynchronous plan revision."""

from __future__ import annotations

import json
import unittest

from agents.plan_revision_coordinator import (
    PlanRevisionCoordinator,
    PlanRevisionFallback,
    PlanRevisionState,
)
from models import AsyncModelRequest, AsyncModelResult, ModelResponse
from perception.candidate_bank import CandidateBank
from planner.policy import PlannerLimits
from planner.revision import (
    QwenPlanRevisionPlanner,
    RevisionLimits,
    RevisionValidator,
)
from planner.schemas import (
    LandingZoneSpec,
    NavigationPointSpec,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraftV2,
)
from planner.skill_catalog import build_default_skill_catalog
from runtime.events import EventSeverity, MissionEvent, MissionEventType
from runtime.frame_store import FrameRef
from runtime.plan_validator import PlanValidator
from runtime.safety_supervisor import SafetySupervisor
from runtime.world_belief import CandidateSummary, QwenRequestStatus, WorldBelief


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def now(self) -> float:
        return self.value


class _FakeWorker:
    def __init__(self, uav_id: str = "uav_1") -> None:
        self.uav_id = uav_id
        self.requests: list[AsyncModelRequest] = []
        self.ready: AsyncModelResult | None = None
        self.poll_count = 0

    def submit(self, request: AsyncModelRequest) -> None:
        self.requests.append(request)

    def poll(self, **kwargs: object) -> AsyncModelResult | None:
        self.poll_count += 1
        result = self.ready
        self.ready = None
        return result

    def complete(self, payload: dict[str, object], **route: object) -> None:
        request = self.requests[-1]
        values: dict[str, object] = {
            "request_id": request.request_id,
            "review_id": request.review_id,
            "mission_id": request.mission_id,
            "uav_id": request.uav_id,
            "plan_version": request.plan_version,
            "observation_timestamp_s": request.observation_timestamp_s,
            "frame_id": request.frame_id,
        }
        values.update(route)
        self.ready = AsyncModelResult(
            request_id=values["request_id"],  # type: ignore[arg-type]
            review_id=values["review_id"],  # type: ignore[arg-type]
            mission_id=values["mission_id"],  # type: ignore[arg-type]
            uav_id=values["uav_id"],  # type: ignore[arg-type]
            plan_version=values["plan_version"],  # type: ignore[arg-type]
            observation_timestamp_s=values["observation_timestamp_s"],  # type: ignore[arg-type]
            frame_id=values["frame_id"],  # type: ignore[arg-type]
            response=ModelResponse(
                content=json.dumps(payload),
                model="fake-qwen",
                finish_reason="stop",
                usage={"total_tokens": 42},
            ),
            error_code=None,
            error_message=None,
        )


class _FakeManager:
    def __init__(self, plan: object) -> None:
        self.uav_id = "uav_1"
        self.task_plan = plan
        self.task_status = "RUNNING"
        self.pending_task_result = None
        self.active_planned_step_id = "goto_search"
        self.active_name = "GOTO"
        self.is_supervisory_paused = False
        self.step_outputs = {
            "takeoff": {
                "status": "SUCCEEDED",
                "oracle_target_pose": [99.0, 88.0, 77.0],
            }
        }
        self.interrupt_count = 0
        self.resume_count = 0
        self.replace_count = 0
        self.candidate_handoff_count = 0
        self.cancel_count = 0
        self.hover_kwargs: dict[str, object] = {}
        self.last_candidate_handoff: tuple[str, str] | None = None

    def interrupt_with_hover(self, reason_code: str, **kwargs: object) -> None:
        if self.is_supervisory_paused:
            raise RuntimeError("already paused")
        self.interrupt_count += 1
        self.hover_kwargs = {"reason_code": reason_code, **kwargs}
        self.is_supervisory_paused = True
        self.active_name = "HOVER"

    def resume_interrupted_step(self) -> None:
        if not self.is_supervisory_paused:
            raise RuntimeError("not paused")
        self.resume_count += 1
        self.is_supervisory_paused = False
        current = next(
            step
            for step in self.task_plan.steps
            if step.step_id == self.active_planned_step_id
        )
        self.active_name = current.skill.value

    def replace_interrupted_step_and_suffix(self, plan: object) -> None:
        if not self.is_supervisory_paused:
            raise RuntimeError("replacement requires supervisory HOVER")
        self.replace_count += 1
        old_index = tuple(step.step_id for step in self.task_plan.steps).index(
            self.active_planned_step_id
        )
        self.task_plan = plan
        self.active_planned_step_id = plan.steps[old_index].step_id
        self.active_name = plan.steps[old_index].skill.value
        self.is_supervisory_paused = False

    def handoff_interrupted_search_candidate_to_inspect(
        self,
        plan: object,
        *,
        candidate_id: str,
        source: str,
    ) -> None:
        if not self.is_supervisory_paused:
            raise RuntimeError("candidate handoff requires supervisory HOVER")
        old_index = tuple(step.step_id for step in self.task_plan.steps).index(
            self.active_planned_step_id
        )
        if self.task_plan.steps[old_index].skill.value != "SEARCH":
            raise RuntimeError("candidate handoff requires SEARCH")
        if plan.steps[old_index].to_dict() != self.task_plan.steps[old_index].to_dict():
            raise RuntimeError("candidate handoff modified SEARCH")
        if old_index + 1 >= len(plan.steps):
            raise RuntimeError("candidate handoff omitted INSPECT")
        inspect = plan.steps[old_index + 1]
        if inspect.skill.value != "INSPECT" or inspect.params.get("candidate_id") != candidate_id:
            raise RuntimeError("candidate handoff INSPECT mismatch")
        self.candidate_handoff_count += 1
        self.last_candidate_handoff = (candidate_id, source)
        self.task_plan = plan
        self.active_planned_step_id = inspect.step_id
        self.active_name = "INSPECT"
        self.is_supervisory_paused = False

    def cancel_task(self) -> None:
        self.cancel_count += 1
        self.pending_task_result = "CANCELED"
        self.active_name = "LAND"
        self.is_supervisory_paused = False


def _world() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={
            "search_area": SearchRegionSpec(
                "search_area",
                (20.0, 20.0, 0.0),
                10.0,
                (20.0, 10.0, 10.0),
            )
        },
        landing_zones={"home": LandingZoneSpec("home", (0.0, 0.0))},
        navigation_points={
            "observation_point": NavigationPointSpec(
                "observation_point",
                (8.0, -4.0, 10.0),
            )
        },
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=20.0,
        search_timeout_s=60.0,
    )


def _step(
    step_id: str,
    skill: str,
    args: dict[str, object],
    *,
    uav_id: str = "uav_1",
) -> dict[str, object]:
    return {"id": step_id, "uav_id": uav_id, "skill": skill, "args": args}


def _original_plan() -> SkillPlanDraftV2:
    return SkillPlanDraftV2.from_dict(
        {
            "schema_version": 2,
            "mission_id": "mission_1",
            "uav_id": "uav_1",
            "plan_version": 1,
            "steps": [
                _step("takeoff", "TAKEOFF", {"altitude_m": 10.0}),
                _step(
                    "goto_search",
                    "GOTO",
                    {"destination": "search_area", "altitude_m": 10.0},
                ),
                _step(
                    "search",
                    "SEARCH",
                    {
                        "region": "search_area",
                        "target_description": "red moving target",
                        "altitude_m": 10.0,
                    },
                ),
                _step(
                    "goto_home",
                    "GOTO",
                    {"destination": "home", "altitude_m": 10.0},
                ),
                _step("land", "LAND", {"zone": "home"}),
            ],
        }
    )


def _revision_payload(
    *,
    mission_id: str = "mission_1",
    uav_id: str = "uav_1",
    base_version: int = 1,
    new_version: int = 2,
    invalid_land_order: bool = False,
) -> dict[str, object]:
    steps = [
        _step(
            "goto_detour",
            "GOTO",
            {"destination": "observation_point", "altitude_m": 10.0},
            uav_id=uav_id,
        ),
        _step(
            "goto_search_v2",
            "GOTO",
            {"destination": "search_area", "altitude_m": 10.0},
            uav_id=uav_id,
        ),
        _step(
            "search_v2",
            "SEARCH",
            {
                "region": "search_area",
                "target_description": "red moving target",
                "altitude_m": 10.0,
            },
            uav_id=uav_id,
        ),
        _step(
            "goto_home_v2",
            "GOTO",
            {"destination": "home", "altitude_m": 10.0},
            uav_id=uav_id,
        ),
        _step("land_v2", "LAND", {"zone": "home"}, uav_id=uav_id),
    ]
    if invalid_land_order:
        steps[-2], steps[-1] = steps[-1], steps[-2]
    return {
        "schema_version": 2,
        "mission_id": mission_id,
        "uav_id": uav_id,
        "base_plan_version": base_version,
        "new_plan_version": new_version,
        "replace_from_step_id": "goto_search",
        "steps": steps,
        "reason_codes": ["PATH_BLOCKED"],
    }


def _inspect_revision_payload(
    *,
    candidate_id: str = "candidate_1",
    mutate_search: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "mission_id": "mission_1",
        "uav_id": "uav_1",
        "base_plan_version": 1,
        "new_plan_version": 2,
        "replace_from_step_id": "search",
        "steps": [
            _step(
                "search",
                "SEARCH",
                {
                    "region": "search_area",
                    "target_description": "red moving target",
                    "altitude_m": 9.0 if mutate_search else 10.0,
                },
            ),
            _step(
                "inspect_candidate",
                "INSPECT",
                {"candidate_id": candidate_id},
            ),
            _step(
                "goto_home",
                "GOTO",
                {"destination": "home", "altitude_m": 10.0},
            ),
            _step("land", "LAND", {"zone": "home"}),
        ],
        "reason_codes": ["CANDIDATE_CONFIRMATION_REQUIRED"],
    }


def _event(
    plan: SkillPlanDraftV2,
    *,
    event_type: MissionEventType = MissionEventType.PLAN_REVISION_REQUESTED,
    uav_id: str | None = None,
    timestamp_s: float = 10.0,
) -> MissionEvent:
    return MissionEvent(
        event_id=f"event_revision_{plan.plan_version}",
        mission_id=plan.mission_id,
        uav_id=plan.uav_id if uav_id is None else uav_id,
        plan_version=plan.plan_version,
        timestamp_s=timestamp_s,
        event_type=event_type,
        severity=EventSeverity.WARNING,
        payload={"source": "test_injection", "oracle_target_pose": [1, 2, 3]},
    )


def _inspect_event(
    plan: SkillPlanDraftV2,
    *,
    candidate_id: str | None = "candidate_1",
    source: str = "qwen_vl",
) -> MissionEvent:
    return MissionEvent(
        event_id=f"event_inspect_{plan.plan_version}",
        mission_id=plan.mission_id,
        uav_id=plan.uav_id,
        plan_version=plan.plan_version,
        timestamp_s=10.0,
        event_type=MissionEventType.PLAN_REVISION_REQUESTED,
        severity=EventSeverity.INFO,
        payload={
            "action": "INSPECT",
            "candidate_id": candidate_id,
            "source": source,
        },
    )


def _candidate_bank(
    *,
    candidate_id: str = "candidate_1",
    source: str = "qwen_vl",
) -> CandidateBank:
    bank = CandidateBank(uav_id="uav_1")
    bank.propose(
        candidate_id=candidate_id,
        timestamp_s=10.0,
        bbox_xyxy_normalized=(0.1, 0.1, 0.4, 0.4),
        frame_ref=FrameRef("uav_1", "frame_candidate", 10.0, 32, 24),
        source=source,
    )
    return bank


def _belief(
    plan: SkillPlanDraftV2,
    event: MissionEvent,
    *,
    step_id: str = "goto_search",
    skill: str = "GOTO",
) -> WorldBelief:
    return WorldBelief(
        mission_id=plan.mission_id,
        uav_id=plan.uav_id,
        plan_version=plan.plan_version,
        current_step_id=step_id,
        current_skill=skill,
        skill_feedback={"progress": 0.25},
        target_spec=plan.target_spec,
        target_snapshot=None,
        candidate_summaries=(),
        recent_events=(event,),
        qwen_request_status=QwenRequestStatus(),
        latest_frame_ref=None,
        mission_elapsed_s=event.timestamp_s,
        plan_id=f"plan_{plan.plan_version}",
    )


class PlanRevisionCoordinatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = _world()
        self.plan = _original_plan()
        self.plan_validator = PlanValidator()
        compiled = self.plan_validator.validate_and_compile(
            self.plan,
            self.world,
            source="dynamic_llm",
            mission_id=self.plan.mission_id,
            uav_id=self.plan.uav_id,
            plan_version=self.plan.plan_version,
        )
        self.manager = _FakeManager(compiled.task_plan)
        self.worker = _FakeWorker()
        self.clock = _Clock()
        self.safety = SafetySupervisor(
            self.world.scene_min_xyz_m,
            self.world.scene_max_xyz_m,
        )
        self.revision_validator = RevisionValidator(
            self.plan_validator,
            revision_limits=RevisionLimits(cooldown_s=0.0),
            safety_preflight=self.safety,
        )
        self.qwen_planner = QwenPlanRevisionPlanner(
            world_context=self.world,
            skill_catalog=build_default_skill_catalog(),
            limits=PlannerLimits(),
        )
        self.coordinator = PlanRevisionCoordinator(
            uav_id="uav_1",
            planner=self.qwen_planner,
            worker=self.worker,
            validator=self.revision_validator,
            skill_manager=self.manager,
            world_context=self.world,
            safety_preflight=self.safety,
            original_instruction="绕开阻塞，继续搜索红色移动目标，最后返回 home",
            clock=self.clock.now,
            request_timeout_s=5.0,
        )

    def _submit(self) -> tuple[MissionEvent, WorldBelief]:
        event = _event(self.plan)
        belief = _belief(self.plan, event)
        snapshot = self.coordinator.submit_event(
            event,
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertIs(snapshot.state, PlanRevisionState.IN_FLIGHT)
        return event, belief

    def _inspect_coordinator(
        self,
        candidate_bank: CandidateBank | None,
    ) -> PlanRevisionCoordinator:
        return PlanRevisionCoordinator(
            uav_id="uav_1",
            planner=self.qwen_planner,
            worker=self.worker,
            validator=self.revision_validator,
            skill_manager=self.manager,
            candidate_bank=candidate_bank,
            world_context=self.world,
            safety_preflight=self.safety,
            original_instruction="检查候选目标，必要时改变视角确认，然后返回 home",
            clock=self.clock.now,
            request_timeout_s=5.0,
        )

    def test_scenario_d_inspect_uses_trusted_candidate_handoff_not_generic_replace(self) -> None:
        coordinator = self._inspect_coordinator(_candidate_bank())
        self.manager.active_planned_step_id = "search"
        self.manager.active_name = "SEARCH"
        event = _inspect_event(self.plan)
        belief = _belief(self.plan, event, step_id="search", skill="SEARCH")

        submitted = coordinator.submit_event(
            event,
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertIs(submitted.state, PlanRevisionState.IN_FLIGHT)
        request_text = self.worker.requests[-1].messages[-1].content
        self.assertIn('"action":"INSPECT"', request_text)
        self.assertIn('"candidate_id":"candidate_1"', request_text)
        self.assertNotIn("bbox", request_text.lower())
        self.assertNotIn("oracle_target", request_text.lower())
        self.worker.complete(_inspect_revision_payload())
        accepted = coordinator.tick(
            current_plan=self.plan,
            world_belief=belief,
        )

        self.assertIs(accepted.state, PlanRevisionState.ACCEPTED)
        self.assertEqual(self.manager.candidate_handoff_count, 1)
        self.assertEqual(self.manager.replace_count, 0)
        self.assertEqual(
            self.manager.last_candidate_handoff,
            ("candidate_1", "qwen_vl"),
        )
        self.assertEqual(self.manager.active_name, "INSPECT")
        self.assertEqual(self.manager.task_plan.plan_version, 2)
        self.assertEqual(accepted.last_record.old_step_id, "search")  # type: ignore[union-attr]
        self.assertEqual(  # type: ignore[union-attr]
            accepted.last_record.new_step_id,
            "inspect_candidate",
        )

    def test_inspect_event_without_bank_or_known_candidate_is_rejected_before_hover(self) -> None:
        self.manager.active_planned_step_id = "search"
        self.manager.active_name = "SEARCH"
        event = _inspect_event(self.plan)
        belief = _belief(self.plan, event, step_id="search", skill="SEARCH")

        without_bank = self._inspect_coordinator(None).submit_event(
            event,
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertEqual(
            without_bank.last_error_code,
            "INSPECT_CANDIDATE_BANK_REQUIRED",
        )
        self.assertEqual(self.manager.interrupt_count, 0)

    def test_oracle_candidate_is_rejected_before_qwen_request(self) -> None:
        coordinator = self._inspect_coordinator(
            _candidate_bank(source="oracle_evaluation")
        )
        self.manager.active_planned_step_id = "search"
        self.manager.active_name = "SEARCH"
        event = _inspect_event(self.plan, source="oracle_evaluation")
        ordinary = _belief(self.plan, event, step_id="search", skill="SEARCH")
        belief = WorldBelief(
            mission_id=ordinary.mission_id,
            uav_id=ordinary.uav_id,
            plan_version=ordinary.plan_version,
            current_step_id=ordinary.current_step_id,
            current_skill=ordinary.current_skill,
            skill_feedback=ordinary.skill_feedback,
            target_spec=ordinary.target_spec,
            target_snapshot=ordinary.target_snapshot,
            candidate_summaries=(
                CandidateSummary(
                    "candidate_1",
                    1.0,
                    event.timestamp_s,
                    "oracle_evaluation",
                ),
            ),
            recent_events=ordinary.recent_events,
            qwen_request_status=ordinary.qwen_request_status,
            latest_frame_ref=ordinary.latest_frame_ref,
            mission_elapsed_s=ordinary.mission_elapsed_s,
            plan_id=ordinary.plan_id,
        )
        result = coordinator.submit_event(
            event,
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertIs(result.state, PlanRevisionState.REJECTED)
        self.assertEqual(
            result.last_error_code,
            "INSPECT_CANDIDATE_SOURCE_PRIVILEGED",
        )
        self.assertEqual(self.manager.interrupt_count, 0)
        self.assertEqual(self.worker.requests, [])
        self.assertEqual(len(self.worker.requests), 0)

        wrong_candidate = self._inspect_coordinator(
            _candidate_bank(candidate_id="candidate_other")
        ).submit_event(
            event,
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertEqual(
            wrong_candidate.last_error_code,
            "INSPECT_CANDIDATE_UNKNOWN",
        )
        self.assertEqual(self.manager.interrupt_count, 0)

        non_provisional_bank = _candidate_bank()
        non_provisional_bank.mark_under_inspection("candidate_1")
        non_provisional = self._inspect_coordinator(
            non_provisional_bank
        ).submit_event(
            event,
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertEqual(
            non_provisional.last_error_code,
            "INSPECT_CANDIDATE_NOT_PROVISIONAL",
        )

        invalid_source = self._inspect_coordinator(
            _candidate_bank(source="detector")
        ).submit_event(
            _inspect_event(self.plan, source="detector"),
            current_plan=self.plan,
            world_belief=_belief(
                self.plan,
                _inspect_event(self.plan, source="detector"),
                step_id="search",
                skill="SEARCH",
            ),
        )
        self.assertEqual(
            invalid_source.last_error_code,
            "INSPECT_CANDIDATE_SOURCE_INVALID",
        )
        self.assertEqual(self.manager.interrupt_count, 0)

    def test_inspect_revision_with_wrong_candidate_or_changed_search_is_rejected(self) -> None:
        invalid_payloads = (
            _inspect_revision_payload(candidate_id="candidate_other"),
            _inspect_revision_payload(mutate_search=True),
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self.setUp()
                coordinator = self._inspect_coordinator(_candidate_bank())
                self.manager.active_planned_step_id = "search"
                self.manager.active_name = "SEARCH"
                event = _inspect_event(self.plan)
                belief = _belief(
                    self.plan,
                    event,
                    step_id="search",
                    skill="SEARCH",
                )
                before = self.manager.task_plan.to_dict()
                coordinator.submit_event(
                    event,
                    current_plan=self.plan,
                    world_belief=belief,
                )
                self.worker.complete(payload)
                rejected = coordinator.tick(
                    current_plan=self.plan,
                    world_belief=belief,
                )
                self.assertIs(rejected.state, PlanRevisionState.REJECTED)
                self.assertEqual(self.manager.candidate_handoff_count, 0)
                self.assertEqual(self.manager.replace_count, 0)
                self.assertEqual(self.manager.task_plan.to_dict(), before)
                self.assertEqual(self.manager.resume_count, 1)

    def test_scenario_f_valid_suffix_stays_in_hover_then_applies_atomically(self) -> None:
        before_semantic = self.plan.to_dict()
        before_runtime = self.manager.task_plan.to_dict()
        _event_value, belief = self._submit()

        self.assertTrue(self.manager.is_supervisory_paused)
        self.assertEqual(self.manager.interrupt_count, 1)
        self.assertEqual(len(self.worker.requests), 1)
        request_text = self.worker.requests[0].messages[-1].content
        self.assertNotIn("oracle_target", request_text.lower())
        self.assertNotIn("99.0", request_text)

        # A poll with no result returns immediately and leaves HOVER/plan alone.
        waiting = self.coordinator.tick(
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertIs(waiting.state, PlanRevisionState.IN_FLIGHT)
        self.assertTrue(self.manager.is_supervisory_paused)
        self.assertEqual(self.manager.task_plan.to_dict(), before_runtime)

        self.worker.complete(_revision_payload())
        accepted = self.coordinator.tick(
            current_plan=self.plan,
            world_belief=belief,
        )

        self.assertIs(accepted.state, PlanRevisionState.ACCEPTED)
        self.assertEqual(accepted.revision_count, 1)
        self.assertEqual(self.manager.replace_count, 1)
        self.assertEqual(self.manager.task_plan.plan_version, 2)
        self.assertEqual(self.manager.task_plan.steps[0].to_dict(), before_runtime["steps"][0])
        self.assertEqual(self.plan.to_dict(), before_semantic)
        self.assertEqual(
            self.coordinator.latest_accepted_revision.revised_plan.plan_version,  # type: ignore[union-attr]
            2,
        )
        self.assertEqual(accepted.last_record.old_step_id, "goto_search")  # type: ignore[union-attr]
        self.assertEqual(accepted.last_record.new_step_id, "goto_detour")  # type: ignore[union-attr]
        self.assertEqual(accepted.last_record.reason_codes, ("PATH_BLOCKED",))  # type: ignore[union-attr]

    def test_scenario_g_wrong_uav_version_or_land_order_never_mutates_plan(self) -> None:
        invalid_payloads = (
            _revision_payload(uav_id="uav_2"),
            _revision_payload(base_version=2, new_version=3),
            _revision_payload(invalid_land_order=True),
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                # Fresh coordinator/manager for each fail-closed case.
                self.setUp()
                before = self.manager.task_plan.to_dict()
                _event_value, belief = self._submit()
                self.worker.complete(payload)
                rejected = self.coordinator.tick(
                    current_plan=self.plan,
                    world_belief=belief,
                )
                self.assertIs(rejected.state, PlanRevisionState.REJECTED)
                self.assertEqual(self.manager.replace_count, 0)
                self.assertEqual(self.manager.resume_count, 1)
                self.assertEqual(self.manager.task_plan.to_dict(), before)
                self.assertFalse(self.manager.is_supervisory_paused)

    def test_timeout_uses_trusted_resume_or_cancel_land_fallback(self) -> None:
        _event_value, belief = self._submit()
        self.clock.value = 15.0
        timed_out = self.coordinator.tick(
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertIs(timed_out.state, PlanRevisionState.TIMED_OUT)
        self.assertEqual(timed_out.last_error_code, "REVISION_TIMEOUT")
        self.assertEqual(
            timed_out.fallback_applied,
            PlanRevisionFallback.RESUME_INTERRUPTED.value,
        )
        self.assertEqual(self.manager.resume_count, 1)
        self.assertEqual(self.worker.poll_count, 0)

        # The alternative is an existing fail-safe cancel-and-land path, not
        # a model-generated controller command.
        self.setUp()
        cancel_coordinator = PlanRevisionCoordinator(
            uav_id="uav_1",
            planner=self.qwen_planner,
            worker=self.worker,
            validator=self.revision_validator,
            skill_manager=self.manager,
            world_context=self.world,
            safety_preflight=self.safety,
            original_instruction="continue safely",
            clock=self.clock.now,
            request_timeout_s=2.0,
            fallback=PlanRevisionFallback.CANCEL_AND_LAND,
        )
        event = _event(self.plan)
        belief = _belief(self.plan, event)
        cancel_coordinator.submit_event(
            event,
            current_plan=self.plan,
            world_belief=belief,
        )
        self.clock.value += 2.0
        result = cancel_coordinator.tick(
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertEqual(result.fallback_applied, "CANCEL_AND_LAND")
        self.assertEqual(self.manager.cancel_count, 1)
        self.assertEqual(self.manager.active_name, "LAND")

    def test_stale_current_step_is_rejected_before_poll_or_replace(self) -> None:
        _event_value, _belief_value = self._submit()
        stale_event = _event(self.plan, timestamp_s=11.0)
        stale_belief = _belief(
            self.plan,
            stale_event,
            step_id="search",
            skill="SEARCH",
        )
        self.manager.active_planned_step_id = "search"
        self.worker.complete(_revision_payload())
        result = self.coordinator.tick(
            current_plan=self.plan,
            world_belief=stale_belief,
        )
        self.assertIs(result.state, PlanRevisionState.REJECTED)
        self.assertEqual(result.last_error_code, "STALE_REVISION")
        self.assertEqual(self.worker.poll_count, 0)
        self.assertEqual(self.manager.replace_count, 0)

    def test_wrong_event_route_and_land_are_rejected_without_hover_or_worker(self) -> None:
        wrong = _event(self.plan, uav_id="uav_2")
        # WorldBelief itself is still routed to uav_1, so the coordinator can
        # reject the cross-UAV event without accepting any work.
        belief = _belief(self.plan, _event(self.plan))
        result = self.coordinator.submit_event(
            wrong,
            current_plan=self.plan,
            world_belief=belief,
        )
        self.assertEqual(result.last_error_code, "ROUTING_MISMATCH")
        self.assertEqual(self.manager.interrupt_count, 0)
        self.assertEqual(len(self.worker.requests), 0)

        self.manager.active_planned_step_id = "land"
        self.manager.active_name = "LAND"
        land_event = _event(self.plan, timestamp_s=12.0)
        land_belief = _belief(self.plan, land_event, step_id="land", skill="LAND")
        result = self.coordinator.submit_event(
            land_event,
            current_plan=self.plan,
            world_belief=land_belief,
        )
        self.assertEqual(result.last_error_code, "REVISION_DURING_LAND")
        self.assertEqual(self.manager.interrupt_count, 0)
        self.assertEqual(len(self.worker.requests), 0)

    def test_revision_budget_is_checked_before_hover_and_submit(self) -> None:
        limited_validator = RevisionValidator(
            self.plan_validator,
            revision_limits=RevisionLimits(
                max_plan_revisions=1,
                cooldown_s=0.0,
                max_total_plan_steps=10,
            ),
            safety_preflight=SafetySupervisor(
                self.world.scene_min_xyz_m,
                self.world.scene_max_xyz_m,
            ),
        )
        coordinator = PlanRevisionCoordinator(
            uav_id="uav_1",
            planner=self.qwen_planner,
            worker=self.worker,
            validator=limited_validator,
            skill_manager=self.manager,
            world_context=self.world,
            safety_preflight=self.safety,
            original_instruction="safe detour",
            clock=self.clock.now,
        )
        event = _event(self.plan)
        belief = _belief(self.plan, event)
        coordinator.submit_event(event, current_plan=self.plan, world_belief=belief)
        self.worker.complete(_revision_payload())
        accepted = coordinator.tick(current_plan=self.plan, world_belief=belief)
        revised = coordinator.latest_accepted_revision.revised_plan  # type: ignore[union-attr]
        self.assertIs(accepted.state, PlanRevisionState.ACCEPTED)

        self.manager.active_planned_step_id = "goto_detour"
        self.manager.active_name = "GOTO"
        second_event = _event(revised, timestamp_s=20.0)
        second_belief = _belief(
            revised,
            second_event,
            step_id="goto_detour",
            skill="GOTO",
        )
        request_count = len(self.worker.requests)
        rejected = coordinator.submit_event(
            second_event,
            current_plan=revised,
            world_belief=second_belief,
        )
        self.assertEqual(rejected.last_error_code, "REVISION_BUDGET_EXCEEDED")
        self.assertEqual(len(self.worker.requests), request_count)
        self.assertEqual(self.manager.interrupt_count, 1)

    def test_logger_failure_does_not_affect_revision_and_reset_is_defensive(self) -> None:
        coordinator = PlanRevisionCoordinator(
            uav_id="uav_1",
            planner=self.qwen_planner,
            worker=self.worker,
            validator=self.revision_validator,
            skill_manager=self.manager,
            world_context=self.world,
            safety_preflight=self.safety,
            original_instruction="safe detour",
            clock=self.clock.now,
            logger=lambda _value: (_ for _ in ()).throw(RuntimeError("logger")),
        )
        event = _event(self.plan)
        belief = _belief(self.plan, event)
        coordinator.submit_event(event, current_plan=self.plan, world_belief=belief)
        self.worker.complete(_revision_payload())
        self.assertIs(
            coordinator.tick(current_plan=self.plan, world_belief=belief).state,
            PlanRevisionState.ACCEPTED,
        )
        coordinator.reset()
        snapshot = coordinator.snapshot()
        self.assertIs(snapshot.state, PlanRevisionState.IDLE)
        self.assertEqual(snapshot.revision_count, 0)
        self.assertEqual(coordinator.records, ())


if __name__ == "__main__":
    unittest.main()
