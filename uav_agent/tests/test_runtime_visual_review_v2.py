from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

import numpy as np

from common.obstacle_types import ObstacleMotionState, ObstacleObservation, VisibleObstacle
from perception.qwen_vlm_verifier import VisualReviewFrame
from perception.runtime_visual_assessment import (
    CompletedStepSummary,
    CurrentStepSummary,
    PlanProgressSummary,
    RemainingStepSummary,
    RuntimeHazardAssessment,
    RuntimeSafetyState,
    RuntimeTargetAssessment,
    RuntimeVisualAction,
    RuntimeVisualAssessmentV2,
    RuntimeVisualDecision,
    RuntimeVisualProtocolError,
    RuntimeVisualReviewInputV2,
    QwenRuntimeVisualVerifierV2,
    TargetAssessmentStatus,
    TaskProgressAssessment,
    build_runtime_visual_assessment_v2_schema,
)
from runtime.frame_store import FrameRef
from target.types import TargetSpec
from models import AsyncModelResult, ModelResponse
from agents.runtime_visual_assessment_coordinator import (
    RuntimeVisualAssessmentCoordinator,
)
from skills.plan import TaskPlan
from skills.types import (
    SkillExecutionReport,
    SkillName,
    SkillResultCode,
    SkillStatus,
)


def _frame() -> VisualReviewFrame:
    return VisualReviewFrame(
        FrameRef("uav_1", "frame_1", 10.0, 8, 8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    )


def _observation() -> ObstacleObservation:
    return ObstacleObservation(
        "observation_1", "frame_1", "uav_1", 10.0,
        (
            VisibleObstacle(
                "box_red", (0.2, 0.1, 0.7, 0.9), (4.2, -0.3, -0.4),
                (3, 3, 2), 4.2, 0.0, ObstacleMotionState.STATIC, True, 2.1,
            ),
        ),
    )


def _progress() -> PlanProgressSummary:
    return PlanProgressSummary(
        (CompletedStepSummary("takeoff_1", "TAKEOFF", "TAKEOFF_COMPLETE"),),
        CurrentStepSummary("search_1", "SEARCH", 0.35, 8.2),
        (RemainingStepSummary("TRACK", 15.0), RemainingStepSummary("GOTO"), RemainingStepSummary("LAND")),
    )


def _input() -> RuntimeVisualReviewInputV2:
    return RuntimeVisualReviewInputV2(
        "review_1", "mission_1", "uav_1", 1, 10.0, "frame_1",
        "search north of home", TargetSpec("moving target"), _progress(),
        (_frame(),), (_observation(),), RuntimeSafetyState.HAZARD_SUSPECTED, 10.0,
    )


class RuntimeVisualReviewV2Test(unittest.TestCase):
    def test_input_exposes_only_task_summary_and_camera_relative_obstacle(self) -> None:
        value = _input()
        payload = value.text_payload()
        encoded = json.dumps(payload, sort_keys=True)
        self.assertIn("CAMERA_FLU", encoded)
        self.assertNotIn("oracle_target", encoded)
        self.assertNotIn("camera_rgb", encoded)
        self.assertNotIn("compiled_task_plan", encoded)

    def test_grounded_path_blocked_can_request_replan(self) -> None:
        assessment = RuntimeVisualAssessmentV2(
            "review_1", "mission_1", "uav_1", 1, 10.0, "frame_1",
            RuntimeVisualDecision.PATH_BLOCKED,
            TaskProgressAssessment(True, True, True),
            RuntimeTargetAssessment(TargetAssessmentStatus.NO_TARGET, False, None),
            (RuntimeHazardAssessment("box_red", True, True, 0.91, True),),
            RuntimeVisualAction.REQUEST_REPLAN,
            ("VISIBLE_OBSTACLE", "ACTIVE_CORRIDOR_BLOCKED"),
        )
        self.assertEqual(assessment.to_dict()["schema_version"], 2)

    def test_ungrounded_hazard_may_hold_but_not_request_route(self) -> None:
        with self.assertRaisesRegex(RuntimeVisualProtocolError, "grounded"):
            RuntimeVisualAssessmentV2(
                "review_1", "mission_1", "uav_1", 1, 10.0, "frame_1",
                RuntimeVisualDecision.PATH_MAY_BE_BLOCKED,
                TaskProgressAssessment(True, True, True),
                RuntimeTargetAssessment(TargetAssessmentStatus.NO_TARGET, False, None),
                (RuntimeHazardAssessment("unknown_box", True, True, 0.7, False),),
                RuntimeVisualAction.REQUEST_REPLAN,
                ("VISIBLE_OBSTACLE",),
            )
        valid = RuntimeVisualAssessmentV2(
            "review_1", "mission_1", "uav_1", 1, 10.0, "frame_1",
            RuntimeVisualDecision.PATH_MAY_BE_BLOCKED,
            TaskProgressAssessment(True, True, True),
            RuntimeTargetAssessment(TargetAssessmentStatus.NO_TARGET, False, None),
            (RuntimeHazardAssessment("unknown_box", True, True, 0.7, False),),
            RuntimeVisualAction.HOLD_AND_INSPECT,
            ("VISIBLE_OBSTACLE",),
        )
        self.assertEqual(valid.recommended_action, RuntimeVisualAction.HOLD_AND_INSPECT)

    def test_routing_mismatch_and_too_many_frames_fail_closed(self) -> None:
        wrong = VisualReviewFrame(
            FrameRef("uav_2", "frame_1", 10.0, 8, 8),
            np.zeros((8, 8, 3), dtype=np.uint8),
        )
        with self.assertRaisesRegex(RuntimeVisualProtocolError, "routing"):
            RuntimeVisualReviewInputV2(
                "review_1", "mission_1", "uav_1", 1, 10.0, "frame_1", "task",
                TargetSpec("target"), _progress(), (wrong,), (), RuntimeSafetyState.CLEAR, 10.0,
            )

    def test_json_schema_is_routed_and_strict(self) -> None:
        schema = build_runtime_visual_assessment_v2_schema(
            review_id="review_1", mission_id="mission_1", uav_id="uav_1",
            plan_version=2, frame_id="frame_1", observation_timestamp_s=1.5,
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["uav_id"]["const"], "uav_1")
        self.assertEqual(schema["properties"]["schema_version"]["const"], 2)

    def test_async_request_is_multimodal_and_parse_is_strict(self) -> None:
        verifier = QwenRuntimeVisualVerifierV2(max_image_side_px=64)
        request = verifier.build_async_request(_input(), request_id="request_1")
        self.assertEqual(request.options.response_format.name, "runtime_visual_assessment_v2")
        self.assertEqual(len(request.messages[1].content), 2)
        assessment = RuntimeVisualAssessmentV2(
            "review_1", "mission_1", "uav_1", 1, 10.0, "frame_1",
            RuntimeVisualDecision.PATH_BLOCKED,
            TaskProgressAssessment(True, True, True),
            RuntimeTargetAssessment(TargetAssessmentStatus.NO_TARGET, False, None),
            (RuntimeHazardAssessment("box_red", True, True, 0.9, True),),
            RuntimeVisualAction.REQUEST_REPLAN,
            ("ACTIVE_CORRIDOR_BLOCKED",),
        )
        result = AsyncModelResult(
            request_id="request_1", review_id="review_1", mission_id="mission_1",
            uav_id="uav_1", plan_version=1, observation_timestamp_s=10.0,
            frame_id="frame_1",
            response=ModelResponse(json.dumps(assessment.to_dict()), None, None, {}),
            error_code=None,
            error_message=None,
        )
        self.assertEqual(verifier.parse_async_result(result, expectation=_input()), assessment)


class RuntimeVisualAssessmentCoordinatorTest(unittest.TestCase):
    class Worker:
        uav_id = "uav_1"

        def __init__(self, *, obstacle_id: str) -> None:
            self.request = None
            self.obstacle_id = obstacle_id

        def submit(self, request: object) -> None:
            self.request = request

        def poll(self, **kwargs: object) -> AsyncModelResult | None:
            del kwargs
            if self.request is None:
                return None
            request, self.request = self.request, None
            assessment = RuntimeVisualAssessmentV2(
                request.review_id,
                request.mission_id,
                request.uav_id,
                request.plan_version,
                request.observation_timestamp_s,
                request.frame_id,
                RuntimeVisualDecision.PATH_MAY_BE_BLOCKED,
                TaskProgressAssessment(True, True, True),
                RuntimeTargetAssessment(
                    TargetAssessmentStatus.NO_TARGET, False, None
                ),
                (
                    RuntimeHazardAssessment(
                        self.obstacle_id,
                        True,
                        True,
                        0.8,
                        False,
                    ),
                ),
                RuntimeVisualAction.HOLD_AND_INSPECT,
                ("VISIBLE_OBSTACLE",),
            )
            return AsyncModelResult(
                request.request_id,
                request.review_id,
                request.mission_id,
                request.uav_id,
                request.plan_version,
                request.observation_timestamp_s,
                request.frame_id,
                ModelResponse(json.dumps(assessment.to_dict()), None, None, {}),
                None,
                None,
            )

    class ObstacleRuntime:
        def __init__(self) -> None:
            self.state = RuntimeSafetyState.CLEAR
            self.reports = []

        def add_qwen_hazard(self, report, **kwargs) -> None:
            self.reports.append((report, kwargs))
            self.state = RuntimeSafetyState.HAZARD_SUSPECTED

    @staticmethod
    def manager() -> object:
        plan = TaskPlan.from_dicts(
            [
                {"id": "takeoff", "skill": "TAKEOFF", "target_altitude": 10.0},
                {
                    "id": "goto",
                    "skill": "GOTO",
                    "position": [10.0, 0.0, 10.0],
                    "tolerance": 0.5,
                },
                {"id": "land", "skill": "LAND"},
            ],
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
        )
        feedback = SimpleNamespace(
            to_dict=lambda: {
                "progress": 0.25,
                "data": {"elapsed_time": 2.0},
            }
        )
        return SimpleNamespace(
            task_plan=plan,
            active_name=SkillName.GOTO,
            active_planned_step_id="goto",
            execution_reports=(),
            get_feedback=lambda: feedback,
        )

    def _run(
        self,
        *,
        obstacle_id: str,
        observation: ObstacleObservation | None,
        apply_to_control: bool = True,
    ):
        worker = self.Worker(obstacle_id=obstacle_id)
        runtime = self.ObstacleRuntime()
        coordinator = RuntimeVisualAssessmentCoordinator(
            uav_id="uav_1",
            worker=worker,
            verifier=QwenRuntimeVisualVerifierV2(max_image_side_px=64),
            original_instruction="fly to the search region",
            target_spec=TargetSpec("moving target"),
            intervals_s={"GOTO": 1.0},
            max_result_age_s=5.0,
            apply_to_control=apply_to_control,
        )
        kwargs = dict(
            manager=self.manager(),
            obstacle_runtime=runtime,
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            frame_id="frame_runtime_1",
            mission_elapsed_s=2.0,
            obstacle_observation=observation,
            safety_state=RuntimeSafetyState.CLEAR,
            uav_speed_mps=1.0,
        )
        coordinator.tick(timestamp_s=10.0, **kwargs)
        coordinator.tick(timestamp_s=10.1, **kwargs)
        return coordinator, runtime

    def test_completed_skill_result_uses_symbolic_name_in_model_summary(self) -> None:
        worker = self.Worker(obstacle_id="box_red")
        runtime = self.ObstacleRuntime()
        coordinator = RuntimeVisualAssessmentCoordinator(
            uav_id="uav_1",
            worker=worker,
            verifier=QwenRuntimeVisualVerifierV2(max_image_side_px=64),
            original_instruction="fly to the search region",
            target_spec=TargetSpec("moving target"),
            intervals_s={"GOTO": 1.0},
            max_result_age_s=5.0,
            apply_to_control=False,
        )
        manager = self.manager()
        manager.execution_reports = (
            SkillExecutionReport(
                mission_id="mission_1",
                uav_id="uav_1",
                plan_version=1,
                step_id="takeoff",
                invocation_id="invocation_1",
                skill_name=SkillName.TAKEOFF,
                status=SkillStatus.SUCCEEDED,
                result_code=SkillResultCode.TAKEOFF_COMPLETE,
                feedback_or_result={},
                timestamp_s=9.0,
            ),
        )

        coordinator.tick(
            manager=manager,
            obstacle_runtime=runtime,
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            frame_id="frame_runtime_1",
            timestamp_s=10.0,
            mission_elapsed_s=10.0,
            obstacle_observation=None,
            safety_state=RuntimeSafetyState.CLEAR,
            uav_speed_mps=1.0,
        )

        self.assertIsNotNone(worker.request)
        self.assertEqual(coordinator.records, ())

    def test_qwen_only_ungrounded_hazard_holds_but_cannot_route(self) -> None:
        coordinator, runtime = self._run(
            obstacle_id="unknown_box",
            observation=None,
        )
        self.assertEqual(len(runtime.reports), 1)
        self.assertFalse(runtime.reports[0][0].geometry_grounded)
        self.assertTrue(coordinator.records[0].applied_to_control)

    def test_matching_camera_observation_grounds_qwen_report(self) -> None:
        _, runtime = self._run(
            obstacle_id="box_red",
            observation=_observation(),
        )
        self.assertTrue(runtime.reports[0][0].geometry_grounded)
        self.assertEqual(runtime.reports[0][0].obstacle_ids, ("box_red",))

    def test_shadow_assessment_never_changes_control(self) -> None:
        coordinator, runtime = self._run(
            obstacle_id="unknown_box",
            observation=None,
            apply_to_control=False,
        )
        self.assertEqual(runtime.reports, [])
        self.assertFalse(coordinator.records[0].applied_to_control)


if __name__ == "__main__":
    unittest.main()
