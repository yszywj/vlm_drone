"""Non-blocking V2 mission/hazard visual assessment coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np

from common.ids import generate_routing_id
from common.obstacle_types import ObstacleObservation
from perception.qwen_vlm_verifier import VisualReviewFrame
from perception.runtime_visual_assessment import (
    CompletedStepSummary,
    CurrentStepSummary,
    PlanProgressSummary,
    QwenRuntimeVisualVerifierV2,
    RemainingStepSummary,
    RuntimeSafetyState,
    RuntimeVisualAssessmentV2,
    RuntimeVisualDecision,
    RuntimeVisualReviewInputV2,
)
from runtime.frame_store import FrameRef
from runtime.hazard_fusion import HazardFusion
from skills.plan import TaskPlan
from skills.types import SkillName
from target.types import TargetSpec


class _Worker(Protocol):
    uav_id: str

    def submit(self, request: object) -> None: ...

    def poll(self, **kwargs: object) -> object | None: ...


@dataclass(frozen=True, slots=True)
class RuntimeVisualAssessmentRecord:
    request_id: str
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    frame_id: str
    observation_timestamp_s: float
    submitted_timestamp_s: float
    completed_timestamp_s: float | None
    assessment: RuntimeVisualAssessmentV2 | None
    stale: bool
    applied_to_control: bool
    error_code: str | None


class RuntimeVisualAssessmentCoordinator:
    """Run V2 review independently from target confirmation and route Qwen."""

    def __init__(
        self,
        *,
        uav_id: str,
        worker: _Worker,
        verifier: QwenRuntimeVisualVerifierV2,
        original_instruction: str,
        target_spec: TargetSpec,
        intervals_s: Mapping[str, float],
        max_result_age_s: float,
        apply_to_control: bool,
    ) -> None:
        if getattr(worker, "uav_id", None) != uav_id:
            raise ValueError("worker.uav_id must match coordinator uav_id")
        if not isinstance(verifier, QwenRuntimeVisualVerifierV2):
            raise TypeError("verifier must be QwenRuntimeVisualVerifierV2")
        if not isinstance(original_instruction, str) or not original_instruction.strip():
            raise ValueError("original_instruction must be non-empty")
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        normalized: dict[str, float] = {}
        for skill, interval in intervals_s.items():
            value = float(interval)
            if not isfinite(value) or value <= 0.0:
                raise ValueError("runtime visual intervals must be positive")
            normalized[str(skill)] = value
        if not normalized:
            raise ValueError("at least one runtime visual interval is required")
        age = float(max_result_age_s)
        if not isfinite(age) or age <= 0.0:
            raise ValueError("max_result_age_s must be positive")
        if not isinstance(apply_to_control, bool):
            raise TypeError("apply_to_control must be bool")
        self._uav_id = uav_id
        self._worker = worker
        self._verifier = verifier
        self._instruction = original_instruction.strip()
        self._target_spec = target_spec
        self._intervals = normalized
        self._max_result_age_s = age
        self._apply_to_control = apply_to_control
        self._pending: RuntimeVisualReviewInputV2 | None = None
        self._pending_request_id: str | None = None
        self._submitted_timestamp_s: float | None = None
        self._last_submitted_by_skill: dict[str, float] = {}
        self._records: list[RuntimeVisualAssessmentRecord] = []

    @property
    def records(self) -> tuple[RuntimeVisualAssessmentRecord, ...]:
        return tuple(self._records)

    def tick(
        self,
        *,
        manager: object,
        obstacle_runtime: object,
        rgb: np.ndarray,
        frame_id: str,
        timestamp_s: float,
        mission_elapsed_s: float,
        obstacle_observation: ObstacleObservation | None,
        safety_state: object,
        uav_speed_mps: float,
    ) -> None:
        now = float(timestamp_s)
        plan = getattr(manager, "task_plan", None)
        if not isinstance(plan, TaskPlan):
            return
        self._poll(
            plan=plan,
            obstacle_runtime=obstacle_runtime,
            now=now,
            uav_speed_mps=uav_speed_mps,
        )
        safety_state = getattr(obstacle_runtime, "state", safety_state)
        skill = getattr(getattr(manager, "active_name", None), "value", None)
        interval = self._intervals.get(str(skill))
        if (
            interval is None
            or self._pending is not None
            or _state_value(safety_state) != RuntimeSafetyState.CLEAR.value
        ):
            return
        last = self._last_submitted_by_skill.get(str(skill))
        if last is not None and now - last < interval:
            return
        review_input = self._build_input(
            manager=manager,
            plan=plan,
            rgb=rgb,
            frame_id=frame_id,
            timestamp_s=now,
            mission_elapsed_s=mission_elapsed_s,
            obstacle_observation=obstacle_observation,
            safety_state=safety_state,
        )
        request_id = generate_routing_id("request_runtime_visual")
        self._worker.submit(
            self._verifier.build_async_request(
                review_input,
                request_id=request_id,
            )
        )
        self._pending = review_input
        self._pending_request_id = request_id
        self._submitted_timestamp_s = now
        self._last_submitted_by_skill[str(skill)] = now

    def _poll(
        self,
        *,
        plan: TaskPlan,
        obstacle_runtime: object,
        now: float,
        uav_speed_mps: float,
    ) -> None:
        expectation = self._pending
        request_id = self._pending_request_id
        if expectation is None or request_id is None:
            return
        result = self._worker.poll(
            expected_request_id=request_id,
            include_stale=True,
        )
        if result is None:
            return
        submitted = self._submitted_timestamp_s
        stale = bool(getattr(result, "stale", False)) or (
            plan.plan_version != expectation.plan_version
            or now - expectation.observation_timestamp_s > self._max_result_age_s
        )
        assessment: RuntimeVisualAssessmentV2 | None = None
        error_code: str | None = None
        applied = False
        if stale:
            error_code = "STALE"
        else:
            try:
                assessment = self._verifier.parse_async_result(
                    result,
                    expectation=expectation,
                )
                applied = self._apply_hazard(
                    assessment,
                    expectation=expectation,
                    obstacle_runtime=obstacle_runtime,
                    uav_speed_mps=uav_speed_mps,
                )
            except Exception as exc:
                error_code = type(exc).__name__
        self._records.append(
            RuntimeVisualAssessmentRecord(
                request_id=request_id,
                review_id=expectation.review_id,
                mission_id=expectation.mission_id,
                uav_id=expectation.uav_id,
                plan_version=expectation.plan_version,
                frame_id=expectation.frame_id,
                observation_timestamp_s=expectation.observation_timestamp_s,
                submitted_timestamp_s=(
                    expectation.observation_timestamp_s
                    if submitted is None
                    else submitted
                ),
                completed_timestamp_s=now,
                assessment=assessment,
                stale=stale,
                applied_to_control=applied,
                error_code=error_code,
            )
        )
        self._pending = None
        self._pending_request_id = None
        self._submitted_timestamp_s = None

    def _apply_hazard(
        self,
        assessment: RuntimeVisualAssessmentV2,
        *,
        expectation: RuntimeVisualReviewInputV2,
        obstacle_runtime: object,
        uav_speed_mps: float,
    ) -> bool:
        if not self._apply_to_control:
            return False
        hazards = tuple(
            item
            for item in assessment.hazards
            if item.present and item.blocks_active_corridor
        )
        hazard_detected = bool(hazards) or assessment.decision in {
            RuntimeVisualDecision.PATH_MAY_BE_BLOCKED,
            RuntimeVisualDecision.PATH_BLOCKED,
        }
        if not hazard_detected:
            return False
        grounded_ids = {
            item.obstacle_id
            for observation in expectation.obstacle_observations
            for item in observation.visible_obstacles
        }
        identified = tuple(dict.fromkeys(item.obstacle_id for item in hazards))
        geometry_grounded = bool(identified) and all(
            item in grounded_ids for item in identified
        )
        report = HazardFusion.qwen_report(
            hazard_detected=True,
            geometry_grounded=geometry_grounded,
            obstacle_ids=identified,
        )
        obstacle_runtime.add_qwen_hazard(
            report,
            mission_id=assessment.mission_id,
            uav_id=assessment.uav_id,
            plan_version=assessment.plan_version,
            timestamp_s=assessment.observation_timestamp_s,
            uav_speed_mps=uav_speed_mps,
        )
        return True

    def _build_input(
        self,
        *,
        manager: object,
        plan: TaskPlan,
        rgb: np.ndarray,
        frame_id: str,
        timestamp_s: float,
        mission_elapsed_s: float,
        obstacle_observation: ObstacleObservation | None,
        safety_state: object,
    ) -> RuntimeVisualReviewInputV2:
        step_id = getattr(manager, "active_planned_step_id", None)
        active = getattr(getattr(manager, "active_name", None), "value", None)
        if not isinstance(step_id, str) or not isinstance(active, str):
            raise RuntimeError("runtime visual review requires an active routed step")
        try:
            index = next(i for i, step in enumerate(plan.steps) if step.step_id == step_id)
        except StopIteration:
            raise RuntimeError("active step is absent from TaskPlan") from None
        feedback = manager.get_feedback()
        feedback_data = feedback.to_dict()
        nested = feedback_data.get("data", {})
        progress = feedback_data.get("progress")
        elapsed = nested.get("elapsed_time", mission_elapsed_s) if isinstance(nested, dict) else mission_elapsed_s
        reports = tuple(getattr(manager, "execution_reports", ()))
        completed = tuple(
            CompletedStepSummary(
                item.step_id,
                item.skill_name.value,
                # SkillResultCode uses ``auto()`` and therefore exposes an
                # integer through ``.value``.  The model-facing summary is a
                # symbolic protocol field, matching SkillExecutionReport's
                # public serialization, so it must use the enum name.
                "UNKNOWN" if item.result_code is None else item.result_code.name,
            )
            for item in reports[-100:]
        )
        remaining = tuple(
            RemainingStepSummary(
                step.skill.value,
                _duration_summary(step.params),
            )
            for step in plan.steps[index + 1 :]
        )
        height, width = rgb.shape[:2]
        frame = VisualReviewFrame(
            FrameRef(self._uav_id, frame_id, timestamp_s, width, height),
            rgb,
        )
        return RuntimeVisualReviewInputV2(
            review_id=generate_routing_id("review_runtime_visual"),
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            plan_version=plan.plan_version,
            observation_timestamp_s=timestamp_s,
            frame_id=frame_id,
            original_instruction=self._instruction,
            target_spec=self._target_spec,
            plan_progress=PlanProgressSummary(
                completed,
                CurrentStepSummary(
                    step_id,
                    active,
                    float(progress) if isinstance(progress, (int, float)) else None,
                    float(elapsed),
                ),
                remaining,
            ),
            frames=(frame,),
            obstacle_observations=(
                () if obstacle_observation is None else (obstacle_observation,)
            ),
            safety_state=RuntimeSafetyState(_state_value(safety_state)),
            mission_elapsed_s=mission_elapsed_s,
        )


def _duration_summary(params: Mapping[str, object]) -> float | None:
    for name in ("duration_s", "track_duration", "timeout_s", "timeout"):
        value = params.get(name)
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and isfinite(float(value))
            and float(value) >= 0.0
        ):
            return float(value)
    return None


def _state_value(value: object) -> str:
    return str(getattr(value, "value", value))


__all__ = [
    "RuntimeVisualAssessmentCoordinator",
    "RuntimeVisualAssessmentRecord",
]
