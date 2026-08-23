from __future__ import annotations

from dataclasses import replace
import json
import unittest

import numpy as np

from env.obstacle_registry import ObstacleRegistry
from agents.obstacle_revision_coordinator import (
    ObstacleRevisionCoordinator,
    ObstacleRevisionCoordinatorState,
)
from models import AsyncModelResult, ModelResponse
from perception.qwen_vlm_verifier import VisualReviewFrame
from perception.runtime_visual_assessment import (
    RuntimeHazardAssessment, RuntimeTargetAssessment, RuntimeVisualAction,
    RuntimeVisualAssessmentV2, RuntimeVisualDecision, TargetAssessmentStatus,
    TaskProgressAssessment,
)
from planner.obstacle_revision import (
    GroundedObstacleGeometry, ObstacleAwareRevisionPlanner,
    ObstacleAwareRevisionRequest, ObstacleReplacementStep,
    ObstacleRevisionError, ObstacleRevisionSession,
    ObstacleRevisionSessionState, ObstacleRouteRevisionDraft,
    build_obstacle_route_revision_schema,
)
from planner.route_critic import RouteValidationContext
from planner.route_types import AvoidanceStrategy, AvoidanceStrategyType, RouteConstraints, RouteDraft, RouteWaypoint
from planner.spatial import CoordinateFrame, PointTarget
from planner.spatial_resolver import FramePose, SpatialResolver
from runtime.frame_store import FrameRef
from runtime.collision_supervisor import CollisionSupervisor
from runtime.hazard_fusion import HazardFusion
from runtime.route_registry import RouteRegistry
from runtime.safety_supervisor import SafetyAction, SafetyDecision
from skills.plan import TaskPlan, TaskStep
from skills.types import SkillName


def _assessment(
    *,
    review_id: str = "review_1",
    plan_version: int = 1,
    timestamp_s: float = 5.0,
    frame_id: str = "frame_1",
) -> RuntimeVisualAssessmentV2:
    return RuntimeVisualAssessmentV2(
        review_id, "mission_1", "uav_1", plan_version, timestamp_s, frame_id,
        RuntimeVisualDecision.PATH_BLOCKED, TaskProgressAssessment(True, True, True),
        RuntimeTargetAssessment(TargetAssessmentStatus.NO_TARGET, False, None),
        (RuntimeHazardAssessment("box_red", True, True, 0.9, True),),
        RuntimeVisualAction.REQUEST_REPLAN, ("ACTIVE_CORRIDOR_BLOCKED",),
    )


def _request(
    *,
    base_plan_version: int = 1,
    route_id: str = "route_1",
    review_id: str = "review_1",
    frame_id: str = "frame_1",
    timestamp_s: float = 5.0,
) -> ObstacleAwareRevisionRequest:
    frame = VisualReviewFrame(FrameRef("uav_1", frame_id, timestamp_s, 8, 8), np.zeros((8, 8, 3), dtype=np.uint8))
    return ObstacleAwareRevisionRequest(
        "mission_1", "uav_1", base_plan_version, base_plan_version + 1, route_id, "goto_search", "search then land",
        {"steps": ["GOTO", "SEARCH", "LAND"]}, (), {"id": "goto_search", "skill": "GOTO"},
        ({"skill": "SEARCH"}, {"skill": "LAND"}), (frame,), _assessment(
            review_id=review_id,
            plan_version=base_plan_version,
            timestamp_s=timestamp_s,
            frame_id=frame_id,
        ),
        GroundedObstacleGeometry("box_red", CoordinateFrame.UAV_HOLD_FLU, (2.7, -1.8, -2.4), (5.7, 1.2, 1.6)),
        PointTarget(CoordinateFrame.UAV_HOLD_FLU, (10.0, 0.0, 0.0)),
        RouteConstraints(minimum_clearance_m=1.5),
    )


def _draft(
    *points: tuple[float, float, float],
    route_id: str = "route_1",
    base_plan_version: int = 1,
) -> ObstacleRouteRevisionDraft:
    route = RouteDraft(route_id, CoordinateFrame.UAV_HOLD_FLU, tuple(RouteWaypoint(f"wp_{i}", p) for i, p in enumerate(points, 1)))
    return ObstacleRouteRevisionDraft(
        "mission_1", "uav_1", base_plan_version, base_plan_version + 1, "goto_search",
        AvoidanceStrategy(AvoidanceStrategyType.BYPASS_LEFT, "original_goto_target", ("LEFT_CLEARANCE_VISIBLE",)),
        route,
        (
            ObstacleReplacementStep(
                "follow_detour",
                "uav_1",
                "FOLLOW_ROUTE",
                {"route_ref": route_id},
            ),
            ObstacleReplacementStep("resume_search", "uav_1", "SEARCH", {}),
            ObstacleReplacementStep("land_home", "uav_1", "LAND", {}),
        ),
    )


def _context() -> RouteValidationContext:
    resolver = SpatialResolver(home_pose=FramePose((0, 0, 0)), uav_start_pose=FramePose((0, 0, 0)), uav_hold_pose=FramePose((0, 0, 10)))
    from common.obstacle_types import ObstacleSpec
    return RouteValidationContext(resolver, ObstacleRegistry((ObstacleSpec("box_red", (5, 0, 10), (2, 2, 4), (1, 0, 0)),)), (-20, -20, 0), (20, 20, 30), (0, 0, 10), (10, 0, 10), RouteConstraints(minimum_clearance_m=1.0))


class ObstacleAwareRevisionTest(unittest.TestCase):
    def test_multimodal_request_contains_relative_geometry_not_hold_world_pose(self) -> None:
        planner = ObstacleAwareRevisionPlanner(max_image_side_px=64)
        request = planner.build_async_request(_request(), request_id="request_route_1")
        payload = request.messages[1].content[0].text
        self.assertIn("UAV_HOLD_FLU", payload)
        self.assertIn("relative_aabb_min_m", payload)
        self.assertIn("active_corridor_rejoin_target", payload)
        self.assertIn('"route_start_xyz_m":[0.0,0.0,0.0]', payload)
        self.assertIn('"required_terminal_suffix":[{"args":{},"skill":"SEARCH"},{"args":{},"skill":"LAND"}]', payload)
        self.assertIn('"xyz_m":[10.0,0.0,0.0]', payload)
        self.assertNotIn("hold_world", payload)
        self.assertEqual(len(request.messages[1].content), 2)

    def test_parse_preserves_model_waypoints_and_routing(self) -> None:
        planner, request, draft = ObstacleAwareRevisionPlanner(), _request(), _draft((2, 3, 0), (8, 3, 0), (10, 0, 0))
        result = AsyncModelResult("request_route_1", "review_route_route_1", "mission_1", "uav_1", 1, 5.0, "frame_1", ModelResponse(json.dumps(draft.to_dict()), None, None, {}), None, None)
        parsed = planner.parse_async_result(result, request=request)
        self.assertEqual(parsed.route_draft.waypoints[0].xyz_m, (2.0, 3.0, 0.0))

    def test_schema_exposes_trusted_empty_args_for_goto_and_land(self) -> None:
        schema = build_obstacle_route_revision_schema(_request())
        variants = schema["properties"]["replacement_steps"]["items"]["oneOf"]
        by_skill = {
            variant["properties"]["skill"]["const"]: variant
            for variant in variants
        }
        goto_args = by_skill["GOTO"]["properties"]["args"]
        self.assertEqual(goto_args["properties"], {})
        self.assertFalse(goto_args["additionalProperties"])
        self.assertNotIn("target", goto_args.get("properties", {}))
        land_args = by_skill["LAND"]["properties"]["args"]
        self.assertEqual(land_args["properties"], {})
        self.assertFalse(land_args["additionalProperties"])
        self.assertNotIn("zone", land_args.get("properties", {}))

    def test_track_context_schema_exposes_actions_without_top_level_reacquire(self) -> None:
        request = replace(
            _request(),
            replace_from_step_id="track_1",
            current_step_summary={"id": "track_1", "skill": "TRACK"},
            remaining_plan_summary=({"skill": "GOTO"}, {"skill": "LAND"}),
        )
        schema = build_obstacle_route_revision_schema(request)
        variants = schema["properties"]["replacement_steps"]["items"]["oneOf"]
        by_skill = {
            variant["properties"]["skill"]["const"]: variant
            for variant in variants
        }

        self.assertNotIn("REACQUIRE", by_skill)
        search_args = by_skill["SEARCH"]["properties"]["args"]
        self.assertEqual(
            search_args["oneOf"][1]["properties"]["target_continuation"]["const"],
            "RESTART_SEARCH",
        )
        track_args = by_skill["TRACK"]["properties"]["args"]
        self.assertEqual(
            track_args["oneOf"][1]["properties"]["target_continuation"]["enum"],
            ["CONTINUE_TRACK", "REACQUIRE"],
        )
        with self.assertRaisesRegex(
            ObstacleRevisionError,
            "invalid for replacement SEARCH",
        ):
            ObstacleReplacementStep(
                "invalid_action",
                "uav_1",
                "SEARCH",
                {"target_continuation": "REACQUIRE"},
            )

    def test_parse_rejects_missing_trusted_terminal_suffix_for_repair(self) -> None:
        planner, request = ObstacleAwareRevisionPlanner(), _request()
        invalid = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0)).to_dict()
        invalid["replacement_steps"] = invalid["replacement_steps"][:1] + [
            invalid["replacement_steps"][-1]
        ]
        result = AsyncModelResult(
            "request_route_1",
            "review_route_route_1",
            "mission_1",
            "uav_1",
            1,
            5.0,
            "frame_1",
            ModelResponse(json.dumps(invalid), None, None, {}),
            None,
            None,
        )
        with self.assertRaisesRegex(ObstacleRevisionError, "terminal suffix"):
            planner.parse_async_result(result, request=request)

    def test_critic_counterexample_then_qwen_repair_succeeds(self) -> None:
        session = ObstacleRevisionSession(mode="critic_sim", max_proposals=3)
        first = _draft((4, 0, 0), (6, 0, 0), (10, 0, 0))
        self.assertEqual(session.evaluate(first, _context()).status.value, "REVISE")
        self.assertEqual(session.state, ObstacleRevisionSessionState.AWAITING_REPAIR)
        second = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0))
        self.assertEqual(session.evaluate(second, _context()).status.value, "ACCEPT")
        self.assertEqual(session.state, ObstacleRevisionSessionState.ACCEPTED)
        history = session.history_dict()
        self.assertIn("proposal_0", history)
        self.assertIn("critique_0", history)
        self.assertEqual(history["proposal_0"]["route_draft"]["waypoints"][0]["xyz_m"], [4.0, 0.0, 0.0])

    def test_repair_budget_exhaustion_is_explicit(self) -> None:
        session = ObstacleRevisionSession(mode="strict", max_proposals=2)
        session.evaluate(_draft((4, 0, 0), (6, 0, 0), (10, 0, 0)), _context())
        session.evaluate(_draft((4, 0.1, 0), (6, 0.1, 0), (10, 0, 0)), _context())
        self.assertEqual(session.state, ObstacleRevisionSessionState.EXHAUSTED)
        self.assertIsNone(session.accepted_proposal)

    def test_wrong_uav_or_route_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ObstacleRevisionError, "uav_id"):
            ObstacleRouteRevisionDraft(
                "mission_1", "uav_1", 1, 2, "goto_search",
                AvoidanceStrategy(AvoidanceStrategyType.BYPASS_LEFT, "original_goto_target", ("TEST",)),
                _draft((1, 2, 0), (10, 0, 0)).route_draft,
                (ObstacleReplacementStep("follow", "uav_2", "FOLLOW_ROUTE", {"route_ref": "route_1"}),),
            )


class ObstacleRevisionCoordinatorTest(unittest.TestCase):
    class _Worker:
        uav_id = "uav_1"

        def __init__(self, proposals: tuple[ObstacleRouteRevisionDraft, ...]) -> None:
            self._proposals = list(proposals)
            self._request = None
            self.requests = []

        def submit(self, request: object) -> None:
            self._request = request
            self.requests.append(request)

        def poll(self, **kwargs: object) -> AsyncModelResult | None:
            del kwargs
            if self._request is None or not self._proposals:
                return None
            request, self._request = self._request, None
            proposal = self._proposals.pop(0)
            return AsyncModelResult(
                request.request_id,
                request.review_id,
                request.mission_id,
                request.uav_id,
                request.plan_version,
                request.observation_timestamp_s,
                request.frame_id,
                ModelResponse(json.dumps(proposal.to_dict()), None, None, {}),
                None,
                None,
            )

    class _RawWorker:
        uav_id = "uav_1"

        def __init__(self, content: str | tuple[str, ...]) -> None:
            self._contents = list(content if isinstance(content, tuple) else (content,))
            self._request = None
            self.requests = []

        def submit(self, request: object) -> None:
            self._request = request
            self.requests.append(request)

        def poll(self, **kwargs: object) -> AsyncModelResult | None:
            del kwargs
            if self._request is None or not self._contents:
                return None
            request, self._request = self._request, None
            content = self._contents.pop(0)
            return AsyncModelResult(
                request.request_id,
                request.review_id,
                request.mission_id,
                request.uav_id,
                request.plan_version,
                request.observation_timestamp_s,
                request.frame_id,
                ModelResponse(content, None, "stop", {}),
                None,
                None,
            )

    class _Manager:
        is_supervisory_paused = True

        def __init__(self) -> None:
            self.replacement = None
            self.replacements = []
            self.cancel_count = 0

        def replace_interrupted_step_and_suffix(self, plan: TaskPlan) -> None:
            self.replacement = plan
            self.replacements.append(plan)

        def cancel_task(self) -> None:
            self.cancel_count += 1

    @staticmethod
    def _grounded_supervisor(
        *,
        supervisor: CollisionSupervisor | None = None,
        plan_version: int = 1,
        timestamp_s: float = 5.0,
    ) -> CollisionSupervisor:
        fusion = HazardFusion()
        report = fusion.low_level_report(
            hazard_detected=True,
            geometry_grounded=True,
            obstacle_ids=("box_red",),
        )
        result = fusion.fuse(
            (report,),
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=plan_version,
            timestamp_s=timestamp_s,
        )
        if supervisor is None:
            supervisor = CollisionSupervisor()
        supervisor.evaluate(result)
        supervisor.mark_hold_established(timestamp_s=timestamp_s + 0.1)
        supervisor.mark_geometry_grounded()
        return supervisor

    @staticmethod
    def _compile(proposal: ObstacleRouteRevisionDraft) -> TaskPlan:
        return TaskPlan(
            (
                TaskStep(
                    "follow_detour",
                    SkillName.FOLLOW_ROUTE,
                    {"route_ref": proposal.route_draft.route_id},
                ),
                TaskStep(
                    "land_home",
                    SkillName.LAND,
                    {"ground_altitude": 0.0},
                ),
            ),
            mission_id=proposal.mission_id,
            uav_id=proposal.uav_id,
            plan_version=proposal.new_plan_version,
        )

    def test_counterexample_repair_registers_exact_final_route_and_replaces(self) -> None:
        first = _draft((4, 0, 0), (6, 0, 0), (10, 0, 0))
        second = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0))
        manager = self._Manager()
        registry = RouteRegistry()
        worker = self._Worker((first, second))
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=worker,
            route_registry=registry,
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(
                SafetyAction.CONTINUE,
                "safe",
            ),
            route_validation_mode="critic_sim",
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )
        self.assertEqual(
            coordinator.tick(timestamp_s=5.2).state,
            ObstacleRevisionCoordinatorState.AWAITING_MODEL,
        )
        final = coordinator.tick(timestamp_s=5.3)
        self.assertEqual(final.state, ObstacleRevisionCoordinatorState.ACCEPTED)
        self.assertEqual(final.proposal_count, 2)
        self.assertEqual(manager.replacement.plan_version, 2)
        self.assertEqual(len(registry.records), 1)
        self.assertEqual(
            registry.get("route_1").raw_proposal["route_draft"]["waypoints"][0]["xyz_m"],
            (2.0, 3.0, 0.0),
        )
        self.assertEqual(
            [record.proposal_index for record in coordinator.records],
            [0, 1],
        )
        critic_repair_system = worker.requests[1].messages[0].content
        self.assertIn("CRITIC REPAIR", critic_repair_system)
        self.assertIn("PATH_INTERSECTS_OBSTACLE", critic_repair_system)
        self.assertIn("grounded_obstacle_geometry", critic_repair_system)
        self.assertEqual(
            coordinator.records[0].proposal["route_draft"]["waypoints"][0]["xyz_m"],
            [4.0, 0.0, 0.0],
        )
        history = coordinator.history_dict
        self.assertIn("proposal_0", history)
        self.assertIn("critique_1", history)

    def test_open_sim_accept_keeps_independent_shadow_strict_rejection(self) -> None:
        intersecting = _draft((4, 0, 0), (6, 0, 0), (10, 0, 0))
        manager = self._Manager()
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=self._Worker((intersecting,)),
            route_registry=RouteRegistry(),
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(
                SafetyAction.CONTINUE,
                "safe",
            ),
            route_validation_mode="open_sim",
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )

        result = coordinator.tick(timestamp_s=5.2)

        self.assertEqual(result.state, ObstacleRevisionCoordinatorState.ACCEPTED)
        record = coordinator.records[0]
        self.assertEqual(record.critique["status"], "ACCEPT")
        self.assertEqual(record.shadow_strict_critique["status"], "REVISE")
        self.assertIn(
            "PATH_INTERSECTS_OBSTACLE",
            {
                item["type"]
                for item in record.shadow_strict_critique["violations"]
            },
        )

    def test_invalid_duplicate_waypoints_preserve_bounded_raw_audit_and_code(self) -> None:
        raw = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0)).to_dict()
        raw["route_draft"]["waypoints"][1]["xyz_m"] = list(
            raw["route_draft"]["waypoints"][0]["xyz_m"]
        )
        content = json.dumps(raw)
        manager = self._Manager()
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=self._RawWorker(content),
            route_registry=RouteRegistry(),
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(SafetyAction.CONTINUE, "safe"),
            route_validation_mode="critic_sim",
            max_proposals=1,
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )

        final = coordinator.tick(timestamp_s=5.2)

        self.assertEqual(final.state, ObstacleRevisionCoordinatorState.EXHAUSTED)
        self.assertEqual(final.error_code, "ROUTE_REPAIR_BUDGET_EXHAUSTED")
        self.assertEqual(manager.cancel_count, 1)
        record = coordinator.records[0]
        self.assertEqual(record.error_code, "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE")
        audit = record.proposal["raw_model_response_audit"]
        self.assertEqual(audit["response_text_length"], len(content))
        self.assertLessEqual(len(audit["response_text_tail"]), 500)
        self.assertEqual(audit["structured_payload_status"], "PRESERVED")
        self.assertEqual(audit["raw_json_object"], raw)

    def test_structural_failure_is_returned_to_qwen_then_second_proposal_succeeds(self) -> None:
        invalid = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0)).to_dict()
        invalid["route_draft"]["waypoints"][1]["xyz_m"] = list(
            invalid["route_draft"]["waypoints"][0]["xyz_m"]
        )
        valid = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0)).to_dict()
        worker = self._RawWorker((json.dumps(invalid), json.dumps(valid)))
        manager = self._Manager()
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=worker,
            route_registry=RouteRegistry(),
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(SafetyAction.CONTINUE, "safe"),
            route_validation_mode="critic_sim",
            max_proposals=2,
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )

        first = coordinator.tick(timestamp_s=5.2)

        self.assertEqual(first.state, ObstacleRevisionCoordinatorState.AWAITING_MODEL)
        self.assertEqual(first.proposal_count, 1)
        self.assertEqual(manager.cancel_count, 0)
        self.assertEqual(len(worker.requests), 2)
        repair_payload = json.loads(worker.requests[1].messages[1].content[0].text)
        counterexample = repair_payload["counterexample"]
        self.assertEqual(
            counterexample["parse_error_code"],
            "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE",
        )
        self.assertEqual(
            counterexample["rejected_output_kind"],
            "OMITTED_REPETITION_RISK",
        )
        self.assertIsNone(counterexample["rejected_model_output"])
        self.assertEqual(counterexample["repair_attempt_index"], 1)
        self.assertFalse(counterexample["repeated_unchanged"])
        repair_system = worker.requests[1].messages[0].content
        self.assertIn("STRUCTURAL REPAIR ATTEMPT 1", repair_system)
        self.assertIn("active_corridor_rejoin_target may appear only once", repair_system)
        self.assertIn("outside the obstacle plus minimum_clearance_m", repair_system)
        self.assertEqual(
            coordinator.records[0].proposal["raw_model_response_audit"][
                "raw_json_object"
            ],
            invalid,
        )
        corrections = counterexample["required_corrections"]
        self.assertTrue(
            any("adjacent waypoint xyz_m must differ" in item for item in corrections)
        )
        self.assertTrue(
            any("active_corridor_rejoin_target" in item for item in corrections)
        )
        repair_text = worker.requests[1].messages[1].content[0].text.casefold()
        self.assertNotIn("data:image/", repair_text)
        self.assertNotIn("base64,", repair_text)
        self.assertNotIn("request_id", repair_text)
        self.assertNotIn("api_key", repair_text)

        final = coordinator.tick(timestamp_s=5.3)

        self.assertEqual(final.state, ObstacleRevisionCoordinatorState.ACCEPTED)
        self.assertEqual(final.proposal_count, 2)
        self.assertEqual(manager.cancel_count, 0)
        self.assertIsNotNone(manager.replacement)
        self.assertEqual(
            [record.outcome for record in coordinator.records],
            ["REVISE_STRUCTURE", "ACCEPTED"],
        )
        self.assertEqual(
            coordinator.records[0].error_code,
            "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE",
        )

    def test_structural_repair_budget_exhaustion_cancels_only_after_last_attempt(self) -> None:
        invalid = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0)).to_dict()
        invalid["route_draft"]["waypoints"][1]["xyz_m"] = list(
            invalid["route_draft"]["waypoints"][0]["xyz_m"]
        )
        content = json.dumps(invalid)
        worker = self._RawWorker((content, content))
        manager = self._Manager()
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=worker,
            route_registry=RouteRegistry(),
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(SafetyAction.CONTINUE, "safe"),
            route_validation_mode="strict",
            max_proposals=2,
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )

        first = coordinator.tick(timestamp_s=5.2)
        self.assertEqual(first.state, ObstacleRevisionCoordinatorState.AWAITING_MODEL)
        self.assertEqual(manager.cancel_count, 0)

        final = coordinator.tick(timestamp_s=5.3)

        self.assertEqual(final.state, ObstacleRevisionCoordinatorState.EXHAUSTED)
        self.assertEqual(final.error_code, "ROUTE_REPAIR_BUDGET_EXHAUSTED")
        self.assertEqual(final.proposal_count, 2)
        self.assertEqual(manager.cancel_count, 1)
        self.assertEqual(len(worker.requests), 2)
        self.assertEqual(len(coordinator.records), 2)
        second_payload = json.loads(worker.requests[1].messages[1].content[0].text)
        self.assertEqual(second_payload["counterexample"]["repair_attempt_index"], 1)
        self.assertEqual(
            [record.error_code for record in coordinator.records],
            [
                "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE",
                "ROUTE_ADJACENT_WAYPOINTS_DUPLICATE",
            ],
        )
        self.assertEqual(
            [record.outcome for record in coordinator.records],
            ["REVISE_STRUCTURE", "EXHAUSTED_STRUCTURE"],
        )
        coordinator.tick(timestamp_s=5.4)
        self.assertEqual(manager.cancel_count, 1)

    def test_repeated_invalid_output_escalates_third_repair_prompt(self) -> None:
        invalid = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0)).to_dict()
        invalid["route_draft"]["waypoints"][1]["xyz_m"] = list(
            invalid["route_draft"]["waypoints"][0]["xyz_m"]
        )
        content = json.dumps(invalid)
        worker = self._RawWorker((content, content, content))
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=worker,
            route_registry=RouteRegistry(),
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=self._Manager(),
            safety_preflight=lambda plan: SafetyDecision(SafetyAction.CONTINUE, "safe"),
            route_validation_mode="critic_sim",
            max_proposals=3,
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )

        coordinator.tick(timestamp_s=5.2)
        second_prompt = worker.requests[1].messages[1].content[0].text
        coordinator.tick(timestamp_s=5.3)
        third_prompt = worker.requests[2].messages[1].content[0].text
        third_feedback = json.loads(third_prompt)["counterexample"]

        self.assertNotEqual(second_prompt, third_prompt)
        self.assertEqual(third_feedback["repair_attempt_index"], 2)
        self.assertTrue(third_feedback["repeated_unchanged"])
        self.assertIn(
            "immediately preceding repair repeated the same rejected JSON",
            worker.requests[2].messages[0].content,
        )
        self.assertEqual(
            third_feedback["rejected_output_kind"],
            "OMITTED_REPETITION_RISK",
        )
        self.assertIsNone(third_feedback["rejected_model_output"])

    def test_invalid_sensitive_response_audit_drops_image_and_credentials(self) -> None:
        content = json.dumps(
            {
                "schema_version": 3,
                "image_url": "data:image/jpeg;base64," + "A" * 900,
                "authorization": "Bearer should-not-be-retained",
            }
        )
        manager = self._Manager()
        worker = self._RawWorker(content)
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=worker,
            route_registry=RouteRegistry(),
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(SafetyAction.CONTINUE, "safe"),
            route_validation_mode="critic_sim",
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )

        coordinator.tick(timestamp_s=5.2)

        audit = coordinator.records[0].proposal["raw_model_response_audit"]
        serialized = json.dumps(audit).casefold()
        self.assertEqual(audit["response_text_length"], len(content))
        self.assertTrue(audit["response_text_truncated"])
        self.assertEqual(
            audit["structured_payload_status"],
            "REDACTED_SENSITIVE_CONTENT",
        )
        self.assertNotIn("raw_json_object", audit)
        self.assertNotIn("data:image/", serialized)
        self.assertNotIn("base64,", serialized)
        self.assertNotIn("bearer should-not-be-retained", serialized)
        self.assertEqual(len(worker.requests), 2)
        repair_text = worker.requests[1].messages[1].content[0].text.casefold()
        repair_payload = json.loads(repair_text)
        self.assertEqual(
            repair_payload["counterexample"]["rejected_output_kind"],
            "omitted_sensitive",
        )
        self.assertIsNone(
            repair_payload["counterexample"]["rejected_model_output"]
        )
        self.assertNotIn("data:image/", repair_text)
        self.assertNotIn("base64,", repair_text)
        self.assertNotIn("bearer should-not-be-retained", repair_text)

    def test_preserved_reset_allows_second_obstacle_revision_in_same_mission(self) -> None:
        first = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0))
        second = _draft(
            (2, -3, 0),
            (8, -3, 0),
            (10, 0, 0),
            route_id="route_2",
            base_plan_version=2,
        )
        manager = self._Manager()
        registry = RouteRegistry()
        supervisor = self._grounded_supervisor()
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=self._Worker((first, second)),
            route_registry=registry,
            collision_supervisor=supervisor,
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(SafetyAction.CONTINUE, "safe"),
            route_validation_mode="critic_sim",
        )

        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )
        self.assertEqual(
            coordinator.tick(timestamp_s=5.2).state,
            ObstacleRevisionCoordinatorState.ACCEPTED,
        )
        coordinator.reset(preserve_records=True)
        self.assertEqual(coordinator.snapshot().state, ObstacleRevisionCoordinatorState.IDLE)
        self.assertEqual(len(coordinator.records), 1)
        self.assertEqual(len(coordinator.history_dict["rounds"]), 1)

        self._grounded_supervisor(
            supervisor=supervisor,
            plan_version=2,
            timestamp_s=6.0,
        )
        coordinator.begin(
            _request(
                base_plan_version=2,
                route_id="route_2",
                review_id="review_2",
                frame_id="frame_2",
                timestamp_s=6.0,
            ),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=6.1,
        )
        final = coordinator.tick(timestamp_s=6.2)

        self.assertEqual(final.state, ObstacleRevisionCoordinatorState.ACCEPTED)
        self.assertEqual([plan.plan_version for plan in manager.replacements], [2, 3])
        self.assertEqual(
            [(record.proposal_index, record.round_index) for record in coordinator.records],
            [(0, 0), (1, 1)],
        )
        self.assertEqual(
            [(record.route_id, record.plan_version) for record in registry.records],
            [("route_1", 2), ("route_2", 3)],
        )
        rounds = coordinator.history_dict["rounds"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["proposal_0"]["route_draft"]["route_id"], "route_1")
        self.assertEqual(rounds[1]["proposal_0"]["route_draft"]["route_id"], "route_2")

    def test_reset_is_forbidden_while_model_request_is_active(self) -> None:
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=self._Worker((_draft((2, 3, 0), (8, 3, 0), (10, 0, 0)),)),
            route_registry=RouteRegistry(),
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=self._Manager(),
            safety_preflight=lambda plan: SafetyDecision(SafetyAction.CONTINUE, "safe"),
            route_validation_mode="critic_sim",
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )
        with self.assertRaisesRegex(RuntimeError, "model request is active"):
            coordinator.reset(preserve_records=True)

    def test_strict_exhaustion_uses_trusted_cancel_and_land_fallback(self) -> None:
        bad = (
            _draft((4, 0, 0), (6, 0, 0), (10, 0, 0)),
            _draft((4, 0.1, 0), (6, 0.1, 0), (10, 0, 0)),
        )
        manager = self._Manager()
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=self._Worker(bad),
            route_registry=RouteRegistry(),
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(SafetyAction.CONTINUE, "safe"),
            route_validation_mode="strict",
            max_proposals=2,
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )
        coordinator.tick(timestamp_s=5.2)
        result = coordinator.tick(timestamp_s=5.3)
        self.assertEqual(result.state, ObstacleRevisionCoordinatorState.EXHAUSTED)
        self.assertEqual(manager.cancel_count, 1)

    def test_safety_publication_failure_cancels_in_all_modes(self) -> None:
        proposal = _draft((2, 3, 0), (8, 3, 0), (10, 0, 0))
        for mode in ("open_sim", "critic_sim", "strict"):
            with self.subTest(mode=mode):
                manager = self._Manager()
                registry = RouteRegistry()
                coordinator = ObstacleRevisionCoordinator(
                    uav_id="uav_1",
                    planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
                    worker=self._Worker((proposal,)),
                    route_registry=registry,
                    collision_supervisor=self._grounded_supervisor(),
                    skill_manager=manager,
                    safety_preflight=lambda plan: SafetyDecision(
                        SafetyAction.CANCEL_AND_LAND,
                        "unsafe publication",
                    ),
                    route_validation_mode=mode,
                )
                coordinator.begin(
                    _request(),
                    validation_context=_context(),
                    frame_snapshot=FramePose((0, 0, 10)),
                    compile_replacement=self._compile,
                    timestamp_s=5.1,
                )

                result = coordinator.tick(timestamp_s=5.2)

                self.assertEqual(
                    result.state,
                    ObstacleRevisionCoordinatorState.FAILED,
                )
                self.assertEqual(
                    result.error_code,
                    "ACCEPTED_ROUTE_PUBLICATION_FAILED",
                )
                self.assertEqual(manager.cancel_count, 1)
                self.assertIsNone(manager.replacement)
                # Safety rejection happens before registry publication, so no
                # externally visible route can remain spuriously ACCEPTED.
                self.assertEqual(registry.records, ())

    def test_manager_publication_failure_marks_registered_route_rejected(self) -> None:
        class FailingManager(self._Manager):
            def replace_interrupted_step_and_suffix(self, plan: TaskPlan) -> None:
                del plan
                raise RuntimeError("injected manager publication failure")

        manager = FailingManager()
        registry = RouteRegistry()
        coordinator = ObstacleRevisionCoordinator(
            uav_id="uav_1",
            planner=ObstacleAwareRevisionPlanner(max_image_side_px=64),
            worker=self._Worker(
                (_draft((2, 3, 0), (8, 3, 0), (10, 0, 0)),)
            ),
            route_registry=registry,
            collision_supervisor=self._grounded_supervisor(),
            skill_manager=manager,
            safety_preflight=lambda plan: SafetyDecision(
                SafetyAction.CONTINUE,
                "safe",
            ),
            route_validation_mode="critic_sim",
        )
        coordinator.begin(
            _request(),
            validation_context=_context(),
            frame_snapshot=FramePose((0, 0, 10)),
            compile_replacement=self._compile,
            timestamp_s=5.1,
        )

        result = coordinator.tick(timestamp_s=5.2)

        self.assertEqual(result.state, ObstacleRevisionCoordinatorState.FAILED)
        self.assertEqual(manager.cancel_count, 1)
        self.assertEqual(registry.get("route_1").state.value, "REJECTED")


if __name__ == "__main__":
    unittest.main()
