"""Pure-Python integration layer for one high-level UAV mission.

The Agent owns orchestration only.  It receives narrow planner/runtime
dependencies and deliberately has no environment, scene, target-truth, image
model, or Isaac Sim reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from numbers import Real

from perception.runtime import (
    PerceptionBoundaryError,
    PerceptionRuntimeProfile,
    observation_contains_oracle_data,
)
from planner.base import MissionPlanner
from planner.schemas import CompiledMission, PlannerRequest, PlannerWorldContext
from runtime.plan_validator import PlanValidator
from runtime.safety_supervisor import (
    SafetyAction,
    SafetyDecision,
    SafetySupervisor,
)
from skills.manager import SkillManager, TaskStatus, TransitionRecord
from skills.plan import TaskPlan, TaskStep
from skills.types import (
    Observation,
    SkillClock,
    SkillName,
    SkillResultCode,
    SkillStatus,
)
from target.target_manager import TargetManager
from target.types import TargetLifecycle, TargetSnapshot, TargetSpec


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


@dataclass(frozen=True, slots=True)
class MissionAgentSnapshot:
    """Immutable public view without environment or perception internals."""

    status: AgentStatus
    task_status: str
    active_skill: str | None
    target: TargetSnapshot
    feedback: dict[str, object] | None
    last_error: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, AgentStatus):
            raise TypeError("status must be an AgentStatus")
        if not isinstance(self.task_status, str) or not self.task_status:
            raise ValueError("task_status must be a non-empty string")
        if self.active_skill is not None and (
            not isinstance(self.active_skill, str) or not self.active_skill
        ):
            raise ValueError("active_skill must be a non-empty string or None")
        if not isinstance(self.target, TargetSnapshot):
            raise TypeError("target must be a TargetSnapshot")
        if self.feedback is not None:
            if not isinstance(self.feedback, dict):
                raise TypeError("feedback must be a dict or None")
            object.__setattr__(self, "feedback", deepcopy(self.feedback))
        if self.last_error is not None and not isinstance(self.last_error, str):
            raise TypeError("last_error must be a string or None")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible view with stable enum strings."""

        return {
            "status": self.status.value,
            "task_status": self.task_status,
            "active_skill": self.active_skill,
            "target": self.target.to_dict(),
            "feedback": (
                None if self.feedback is None else deepcopy(self.feedback)
            ),
            "last_error": self.last_error,
        }


class MissionAgentError(RuntimeError):
    """Raised for an invalid Agent lifecycle or integration failure."""


_TARGET_TERMINATABLE_STATES = frozenset(
    {
        TargetLifecycle.UNINITIALIZED,
        TargetLifecycle.SEARCHING,
        TargetLifecycle.CANDIDATE,
        TargetLifecycle.LOCKED,
        TargetLifecycle.TRACKING,
        TargetLifecycle.LOST,
        TargetLifecycle.REACQUIRING,
    }
)


class MissionAgent:
    """Connect planning, validation, safety, Skills, and target lifecycle.

    Planning occurs exactly once in :meth:`start`.  Runtime :meth:`tick` calls
    only deterministic safety and Skill components.  A safety shutdown is
    latched so later observations advance fail-safe LAND instead of repeatedly
    canceling it.
    """

    def __init__(
        self,
        planner: MissionPlanner,
        validator: PlanValidator,
        safety: SafetySupervisor,
        skill_manager: SkillManager,
        target_manager: TargetManager,
        clock: SkillClock,
        logger: Callable[[str], object] | None = None,
        perception_runtime_profile: PerceptionRuntimeProfile = (
            PerceptionRuntimeProfile.PRODUCTION
        ),
        acknowledge_privileged_oracle: bool = False,
    ) -> None:
        if not isinstance(planner, MissionPlanner):
            raise TypeError("planner must be a MissionPlanner")
        if not isinstance(validator, PlanValidator):
            raise TypeError("validator must be a PlanValidator")
        if not isinstance(safety, SafetySupervisor):
            raise TypeError("safety must be a SafetySupervisor")
        if not isinstance(skill_manager, SkillManager):
            raise TypeError("skill_manager must be a SkillManager")
        if not isinstance(target_manager, TargetManager):
            raise TypeError("target_manager must be a TargetManager")
        if not isinstance(clock, SkillClock):
            raise TypeError("clock must satisfy SkillClock")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable or None")
        if not isinstance(perception_runtime_profile, PerceptionRuntimeProfile):
            raise TypeError(
                "perception_runtime_profile must be a PerceptionRuntimeProfile"
            )
        if not isinstance(acknowledge_privileged_oracle, bool):
            raise TypeError("acknowledge_privileged_oracle must be bool")
        if (
            perception_runtime_profile
            is PerceptionRuntimeProfile.ORACLE_EVALUATION
            and not acknowledge_privileged_oracle
        ):
            raise MissionAgentError(
                "ORACLE_EVALUATION requires explicit "
                "acknowledge_privileged_oracle=True"
            )
        if (
            perception_runtime_profile is PerceptionRuntimeProfile.PRODUCTION
            and acknowledge_privileged_oracle
        ):
            raise MissionAgentError(
                "privileged Oracle acknowledgement is invalid in PRODUCTION"
            )
        if skill_manager.task_status is not TaskStatus.IDLE:
            raise MissionAgentError("skill_manager must be IDLE at construction")
        if target_manager.lifecycle is not TargetLifecycle.UNINITIALIZED:
            raise MissionAgentError("target_manager must be UNINITIALIZED")

        self._planner = planner
        self._validator = validator
        self._safety = safety
        self._skill_manager = skill_manager
        self._target_manager = target_manager
        self._clock = clock
        self._logger = logger
        self._perception_runtime_profile = perception_runtime_profile

        self._status = AgentStatus.IDLE
        self._compiled_mission: CompiledMission | None = None
        self._transition_cursor = 0
        self._last_observation_timestamp: float | None = None
        self._mission_start_time_s: float | None = None
        self._last_error: str | None = None
        # A runtime safety ABORT must finish LAND as FAILED; an ordinary cancel
        # or CANCEL_AND_LAND finishes as CANCELED.
        self._shutdown_outcome: AgentStatus | None = None

    def start(
        self,
        instruction: str,
        world_context: PlannerWorldContext,
    ) -> CompiledMission:
        """Plan once, validate, preflight, and start the first Skill."""

        if self._status is not AgentStatus.IDLE:
            raise MissionAgentError(
                f"start requires IDLE, current status is {self._status.value}"
            )
        if self._skill_manager.task_status is not TaskStatus.IDLE:
            raise MissionAgentError("skill_manager is not IDLE")
        if self._target_manager.lifecycle is not TargetLifecycle.UNINITIALIZED:
            raise MissionAgentError("target_manager is not UNINITIALIZED")

        self._status = AgentStatus.PLANNING
        self._last_error = None
        self._safe_log("[MissionAgent] planner_started")
        try:
            request = PlannerRequest(
                instruction=instruction,
                world_context=world_context,
            )
            planner_output = self._planner.plan(request)
        except Exception as exc:
            self._safe_log("[MissionAgent] planner_finished status=FAILED")
            self._raise_start_failure("planner failed", exc)
        self._safe_log("[MissionAgent] planner_finished status=SUCCEEDED")

        try:
            compiled = self._validator.validate_and_compile(
                planner_output,
                world_context,
                source=self._planner_source(),
            )
            owned_compiled = _copy_compiled_mission(compiled)
        except Exception as exc:
            self._raise_start_failure("plan validation failed", exc)

        try:
            decision = self._safety.preflight(owned_compiled)
        except Exception as exc:
            self._raise_start_failure("safety preflight failed", exc)
        if not isinstance(decision, SafetyDecision):
            self._raise_start_failure(
                "safety preflight failed",
                TypeError("SafetySupervisor returned an invalid decision"),
            )
        self._log_safety(decision, phase="preflight")
        if decision.action is not SafetyAction.CONTINUE:
            self._raise_start_failure(
                "safety preflight rejected mission",
                MissionAgentError(
                    f"{decision.action.value}: {decision.reason}"
                ),
            )

        try:
            mission_start_time = self._read_clock()
            task_status = self._skill_manager.start_task(owned_compiled.task_plan)
        except Exception as exc:
            self._raise_start_failure("SkillManager start failed", exc)
        if task_status is not TaskStatus.RUNNING:
            self._raise_start_failure(
                "SkillManager start failed",
                MissionAgentError(
                    "SkillManager did not enter RUNNING after start_task"
                ),
            )

        # Keep a private plan snapshot.  The returned CompiledMission is a
        # separate copy so callers cannot mutate future target metadata or
        # control-flow queries after Safety preflight.
        self._compiled_mission = owned_compiled
        self._mission_start_time_s = mission_start_time
        self._last_observation_timestamp = None
        self._transition_cursor = 0
        self._shutdown_outcome = None
        self._status = AgentStatus.RUNNING
        try:
            # Consume and log the initial NONE -> TAKEOFF record.  It never
            # changes TargetManager, which remains UNINITIALIZED until SEARCH.
            self._consume_transitions()
        except Exception as exc:
            self._last_error = self._error_text("transition processing failed", exc)
            self._begin_shutdown(AgentStatus.FAILED, self._last_error)
            raise MissionAgentError(self._last_error) from exc
        return _copy_compiled_mission(owned_compiled)

    def tick(self, observation: Observation) -> MissionAgentSnapshot:
        """Advance safety and at most one Skill tick for a new frame."""

        if self._status is not AgentStatus.RUNNING:
            raise MissionAgentError(
                f"tick requires RUNNING, current status is {self._status.value}"
            )

        # Enforce the information boundary before timestamp de-duplication,
        # SafetySupervisor, SkillManager, or transition handling sees the
        # observation.  Production is the constructor default; the legacy
        # ideal Oracle pipeline requires a conspicuous two-part opt-in.
        boundary_violation = (
            isinstance(observation, Observation)
            and self._perception_runtime_profile
            is PerceptionRuntimeProfile.PRODUCTION
            and observation_contains_oracle_data(observation)
        )
        if boundary_violation and self._shutdown_outcome is None:
            exc = PerceptionBoundaryError(
                "production MissionAgent rejects oracle_target_* fields"
            )
            message = self._error_text("perception boundary rejected observation", exc)
            self._last_error = message
            self._begin_shutdown(AgentStatus.FAILED, message)
            self._consume_transitions()
            self._sync_status()
            rejected_timestamp = _valid_observation_timestamp(observation)
            if rejected_timestamp is not None and (
                self._last_observation_timestamp is None
                or rejected_timestamp > self._last_observation_timestamp
            ):
                self._last_observation_timestamp = rejected_timestamp
            return self.snapshot()
        if boundary_violation:
            # Never pass privileged values into Safety or LAND.  Once shutdown
            # is latched, stripping those fields lets fail-safe LAND continue
            # even if a misconfigured backend keeps emitting Oracle data.
            observation = replace(
                observation,
                oracle_target_id=None,
                oracle_target_visible=None,
                oracle_target_pose=None,
                oracle_target_velocity=None,
            )

        timestamp = _valid_observation_timestamp(observation)
        if (
            timestamp is not None
            and self._last_observation_timestamp is not None
            and timestamp == self._last_observation_timestamp
        ):
            return self.snapshot()

        # During shutdown Safety still validates every distinct frame.  A
        # repeated CANCEL_AND_LAND must not cancel LAND again, but a newly
        # corrupted or time-reversed frame must also never reach the Skill.
        if self._shutdown_outcome is not None:
            decision = self._runtime_safety_decision(observation)
            self._log_safety(decision, phase="shutdown")
            if decision.action is SafetyAction.ABORT:
                self._last_error = (
                    f"safety {decision.action.value}: {decision.reason}"
                )
                self._begin_shutdown(AgentStatus.FAILED, self._last_error)
            else:
                # CONTINUE and an already-latched CANCEL_AND_LAND both allow
                # the fail-safe LAND Skill to consume this trusted frame.
                self._tick_manager_or_abort(observation)
        else:
            decision = self._runtime_safety_decision(observation)
            self._log_safety(decision, phase="runtime")
            if decision.action is SafetyAction.CONTINUE:
                self._tick_manager_or_abort(observation)
            else:
                forced_outcome = (
                    AgentStatus.FAILED
                    if decision.action is SafetyAction.ABORT
                    else AgentStatus.CANCELED
                )
                self._last_error = (
                    f"safety {decision.action.value}: {decision.reason}"
                )
                self._begin_shutdown(forced_outcome, self._last_error)

        if timestamp is not None and (
            self._last_observation_timestamp is None
            or timestamp > self._last_observation_timestamp
        ):
            self._last_observation_timestamp = timestamp
        try:
            self._consume_transitions()
        except Exception as exc:
            message = self._error_text("transition processing failed", exc)
            self._last_error = message
            self._begin_shutdown(AgentStatus.FAILED, message)
            self._consume_transitions()
        self._sync_status()
        return self.snapshot()

    def cancel(self) -> MissionAgentSnapshot:
        """Request cancellation while allowing SkillManager to complete LAND."""

        if self._status is not AgentStatus.RUNNING:
            raise MissionAgentError(
                f"cancel requires RUNNING, current status is {self._status.value}"
            )
        if self._shutdown_outcome is not None:
            raise MissionAgentError("mission shutdown is already in progress")
        self._begin_shutdown(AgentStatus.CANCELED, "user_requested_cancel")
        self._consume_transitions()
        self._sync_status()
        return self.snapshot()

    def reset(self) -> MissionAgentSnapshot:
        """Reset a terminal Agent and the managers it actually started."""

        if self._status not in {
            AgentStatus.SUCCEEDED,
            AgentStatus.FAILED,
            AgentStatus.CANCELED,
        }:
            raise MissionAgentError(
                "reset requires SUCCEEDED, FAILED, or CANCELED; "
                f"current status is {self._status.value}"
            )

        manager_status = self._skill_manager.task_status
        if manager_status in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELED,
        }:
            self._skill_manager.reset_task()
        elif manager_status is not TaskStatus.IDLE:
            raise MissionAgentError(
                "cannot reset while SkillManager task is still RUNNING"
            )

        target_status = self._target_manager.lifecycle
        if target_status is TargetLifecycle.TERMINATED:
            self._target_manager.reset()
        elif target_status is not TargetLifecycle.UNINITIALIZED:
            raise MissionAgentError(
                "cannot reset a non-terminal active target lifecycle"
            )

        self._safety.reset()
        self._compiled_mission = None
        self._transition_cursor = 0
        self._last_observation_timestamp = None
        self._mission_start_time_s = None
        self._last_error = None
        self._shutdown_outcome = None
        self._status = AgentStatus.IDLE
        self._safe_log("[MissionAgent] reset status=IDLE")
        return self.snapshot()

    def snapshot(self) -> MissionAgentSnapshot:
        """Return fresh, defensive high-level state only."""

        task_status = self._skill_manager.task_status
        active_name = self._skill_manager.active_name
        feedback: dict[str, object] | None = None
        if active_name is not None:
            try:
                feedback = self._skill_manager.get_feedback().to_dict()
            except Exception:
                # Feedback is observational; losing it must not alter control.
                feedback = None
        return MissionAgentSnapshot(
            status=self._status,
            task_status=_enum_text(task_status),
            active_skill=(
                None if active_name is None else _enum_text(active_name)
            ),
            target=self._target_manager.snapshot(),
            feedback=None if feedback is None else deepcopy(feedback),
            last_error=self._last_error,
        )

    def _planner_source(self) -> str:
        explicit = getattr(self._planner, "source", None)
        if explicit is not None:
            if explicit not in {
                "scripted",
                "llm",
                "dynamic_scripted",
                "dynamic_llm",
            }:
                raise MissionAgentError(
                    "planner.source must be scripted, llm, dynamic_scripted, "
                    "or dynamic_llm"
                )
            return explicit

        # Avoid importing the concrete LLM client stack into this integration
        # layer.  Built-in LLMPlanner subclasses are still identified reliably.
        if any(
            cls.__name__ == "LLMPlanner"
            and cls.__module__ == "planner.llm_planner"
            for cls in type(self._planner).__mro__
        ):
            return "llm"
        if any(
            cls.__name__ == "DynamicLLMPlanner"
            and cls.__module__ == "planner.dynamic_llm_planner"
            for cls in type(self._planner).__mro__
        ):
            return "dynamic_llm"
        if any(
            cls.__name__ == "ScriptedDynamicPlanner"
            and cls.__module__ == "planner.scripted_dynamic_planner"
            for cls in type(self._planner).__mro__
        ):
            return "dynamic_scripted"
        return "scripted"

    def _runtime_safety_decision(
        self,
        observation: Observation,
    ) -> SafetyDecision:
        try:
            now = self._read_clock()
            if self._mission_start_time_s is None:
                raise MissionAgentError("mission start time is missing")
            elapsed = now - self._mission_start_time_s
        except Exception:
            # SafetySupervisor maps a non-finite elapsed value to ABORT.
            elapsed = float("nan")
        try:
            decision = self._safety.evaluate(
                observation,
                mission_elapsed_s=elapsed,
            )
        except Exception as exc:
            return SafetyDecision(
                SafetyAction.ABORT,
                self._error_text("SafetySupervisor.evaluate failed", exc),
            )
        if not isinstance(decision, SafetyDecision):
            return SafetyDecision(
                SafetyAction.ABORT,
                "SafetySupervisor returned an invalid decision",
            )
        return decision

    def _tick_manager_or_abort(self, observation: Observation) -> None:
        try:
            self._skill_manager.tick(observation)
        except Exception as exc:
            message = self._error_text("SkillManager tick failed", exc)
            self._last_error = message
            self._begin_shutdown(AgentStatus.FAILED, message)

    def _begin_shutdown(self, outcome: AgentStatus, reason: str) -> None:
        if outcome not in {AgentStatus.FAILED, AgentStatus.CANCELED}:
            raise ValueError("shutdown outcome must be FAILED or CANCELED")
        if self._shutdown_outcome is not None:
            # ABORT has precedence if a later internal error occurs while an
            # ordinary cancellation is already landing.
            if outcome is AgentStatus.FAILED:
                self._shutdown_outcome = AgentStatus.FAILED
            return
        self._shutdown_outcome = outcome
        self._safe_log(
            f"[MissionAgent] shutdown_requested outcome={outcome.value} "
            f"reason={reason}"
        )
        try:
            self._skill_manager.cancel_task()
        except Exception as exc:
            self._shutdown_outcome = None
            message = self._error_text("SkillManager cancel failed", exc)
            self._last_error = message
            raise MissionAgentError(message) from exc

    def _consume_transitions(self) -> None:
        records = self._skill_manager.transition_log
        if self._transition_cursor > len(records):
            raise MissionAgentError("SkillManager transition log moved backwards")
        while self._transition_cursor < len(records):
            record = records[self._transition_cursor]
            # Advance before processing so a failing mapping is never applied a
            # second time after a partial TargetManager transition.
            self._transition_cursor += 1
            self._log_skill_transition(record)
            event_count = len(self._target_manager.events())
            try:
                self._apply_target_transition(record)
            finally:
                self._log_new_target_events(event_count)

    def _apply_target_transition(self, record: TransitionRecord) -> None:
        if not isinstance(record, TransitionRecord):
            raise MissionAgentError("SkillManager emitted an invalid transition")
        compiled = self._compiled_mission
        if compiled is None:
            raise MissionAgentError("compiled mission is missing")

        if record.new_skill is SkillName.SEARCH:
            if self._target_manager.lifecycle is not TargetLifecycle.UNINITIALIZED:
                raise MissionAgentError("SEARCH entered with an active target")
            target_description = self._search_target_description(
                getattr(record, "new_step_id", None)
            )
            self._target_manager.start_search(
                TargetSpec(target_description), record.timestamp
            )
            return

        if (
            record.old_skill is SkillName.SEARCH
            and record.old_status is SkillStatus.SUCCEEDED
            and record.result_code is SkillResultCode.TARGET_FOUND
        ):
            target_id = self._skill_manager.active_target_id
            if not isinstance(target_id, str) or not target_id.strip():
                raise MissionAgentError("SEARCH transition has no active target_id")
            target_id = target_id.strip()
            if (
                self._perception_runtime_profile
                is PerceptionRuntimeProfile.ORACLE_EVALUATION
            ):
                # Explicit Stage-0 expert/upper-bound bypass.  Production must
                # arrive here only after the visual confirmation coordinator
                # has already advanced CANDIDATE -> LOCKED.
                self._target_manager.lock_oracle_from_search(
                    target_id,
                    timestamp_s=record.timestamp,
                    confidence=1.0,
                )
            else:
                self._require_production_visual_lock(target_id, "SEARCH")
            # SEARCH may be followed by navigation or may be the last target
            # operation.  Lock immediately, but enter TRACKING only when the
            # planned successor is actually TRACK.
            if record.new_skill is SkillName.TRACK:
                self._target_manager.start_tracking(record.timestamp)
            return

        if (
            record.old_skill is SkillName.TRACK
            and record.old_status is SkillStatus.FAILED
            and record.result_code is SkillResultCode.TARGET_LOST
            and record.new_skill is SkillName.REACQUIRE
        ):
            data = self._matching_last_result_data(record)
            self._target_manager.mark_lost(
                timestamp_s=record.timestamp,
                last_seen_position=_optional_vector3(data, "last_seen_position"),
                last_seen_velocity=_optional_vector3(data, "last_seen_velocity"),
                last_seen_time_s=_optional_finite(data, "last_seen_time"),
            )
            self._target_manager.start_reacquiring(record.timestamp)
            return

        if (
            record.old_skill is SkillName.REACQUIRE
            and record.old_status is SkillStatus.SUCCEEDED
            and record.result_code is SkillResultCode.TARGET_FOUND
            and record.new_skill is SkillName.TRACK
        ):
            target_id = self._skill_manager.active_target_id
            if not isinstance(target_id, str) or not target_id.strip():
                raise MissionAgentError("REACQUIRE transition has no target_id")
            target_id = target_id.strip()
            if (
                self._perception_runtime_profile
                is PerceptionRuntimeProfile.ORACLE_EVALUATION
            ):
                self._target_manager.mark_reacquired_oracle(
                    target_id,
                    timestamp_s=record.timestamp,
                    confidence=1.0,
                )
            else:
                self._require_production_visual_lock(target_id, "REACQUIRE")
            self._target_manager.start_tracking(record.timestamp)
            return

        if record.new_skill is SkillName.TRACK:
            # Dynamic plans may insert one or more GOTO steps between SEARCH
            # and TRACK.  The target remains LOCKED during that navigation.
            lifecycle = self._target_manager.lifecycle
            if lifecycle is TargetLifecycle.LOCKED:
                self._target_manager.start_tracking(record.timestamp)
            elif lifecycle is not TargetLifecycle.TRACKING:
                raise MissionAgentError(
                    "TRACK entered without a locked or already-tracked target"
                )
            return

        if (
            record.old_skill is SkillName.TRACK
            and record.old_status is SkillStatus.SUCCEEDED
            and record.result_code is SkillResultCode.TRACK_COMPLETE
            and record.new_skill in {SkillName.GOTO, SkillName.LAND}
        ):
            # A dynamic plan may contain a second bounded TRACK call.  Keep the
            # identity locked, but do not claim active tracking while an
            # intervening navigation step is running.
            if self._has_future_track(getattr(record, "old_step_id", None)):
                self._target_manager.finish_tracking_segment(record.timestamp)
            else:
                self._terminate_target_if_needed(
                    record.timestamp,
                    "tracking_complete",
                )
            return

        # Failure/cancel transitions start fail-safe LAND.  A mission canceled
        # before SEARCH terminates directly without fabricating SEARCH or an ID.
        if record.new_skill is SkillName.LAND:
            # A navigation-only mission has no target lifecycle to end yet;
            # keep it UNINITIALIZED until a normal planned LAND completes.  A
            # failure/cancel landing still records task termination at the
            # fail-safe boundary, preserving the MissionAgent contract.
            if (
                self._target_manager.lifecycle is not TargetLifecycle.UNINITIALIZED
                or self._skill_manager.pending_task_result
                in {TaskStatus.FAILED, TaskStatus.CANCELED}
            ):
                self._terminate_target_if_needed(record.timestamp, record.reason)
            return

        # The final LAND transition records the actual task outcome.  This is a
        # defensive fallback when an unusual Manager path skipped the prior
        # LAND-start record.
        if record.new_skill is None:
            self._terminate_target_if_needed(record.timestamp, record.reason)

    def _matching_last_result_data(
        self,
        record: TransitionRecord,
    ) -> Mapping[str, object]:
        result = self._skill_manager.last_result
        if (
            result is None
            or result.status is not record.old_status
            or result.code is not record.result_code
        ):
            return {}
        return result.data

    def _search_target_description(self, step_id: str | None) -> str:
        compiled = self._compiled_mission
        if compiled is None:
            raise MissionAgentError("compiled mission is missing")
        search_steps = [
            step
            for step in compiled.task_plan.steps
            if step.skill is SkillName.SEARCH
        ]
        step = None
        if isinstance(step_id, str):
            step = next(
                (item for item in search_steps if item.step_id == step_id),
                None,
            )
        elif len(search_steps) == 1:
            # Compatibility for transition records produced before planned
            # step IDs were added.
            step = search_steps[0]
        if step is None:
            raise MissionAgentError("SEARCH transition step_id is not in TaskPlan")
        description = step.params.get("target_description")
        if not isinstance(description, str) or not description.strip():
            raise MissionAgentError(
                "compiled SEARCH step has no target_description"
            )
        return description.strip()

    def _has_future_track(self, step_id: str | None) -> bool:
        if not isinstance(step_id, str) or self._compiled_mission is None:
            return False
        steps = self._compiled_mission.task_plan.steps
        for index, step in enumerate(steps):
            if step.step_id == step_id:
                return any(
                    later.skill is SkillName.TRACK
                    for later in steps[index + 1 :]
                )
        raise MissionAgentError("TRACK transition step_id is not in TaskPlan")

    def _require_production_visual_lock(
        self,
        target_id: str,
        skill_name: str,
    ) -> None:
        snapshot = self._target_manager.snapshot()
        if snapshot.lifecycle is not TargetLifecycle.LOCKED:
            raise MissionAgentError(
                f"production {skill_name}->TRACK requires prior visual "
                "CANDIDATE confirmation and LOCKED state"
            )
        if snapshot.target_id != target_id:
            raise MissionAgentError(
                f"production {skill_name}->TRACK target_id does not match "
                "the confirmed visual target"
            )
        if snapshot.source is None or snapshot.source.casefold() == "oracle":
            raise MissionAgentError(
                f"production {skill_name}->TRACK rejects an Oracle target lock"
            )

    def _terminate_target_if_needed(self, timestamp: float, reason: str) -> None:
        if self._target_manager.lifecycle in _TARGET_TERMINATABLE_STATES:
            self._target_manager.terminate(timestamp, reason)

    def _sync_status(self) -> None:
        task_status = self._skill_manager.task_status
        if task_status is TaskStatus.RUNNING:
            self._status = AgentStatus.RUNNING
            return
        if task_status is TaskStatus.SUCCEEDED:
            self._status = AgentStatus.SUCCEEDED
            return
        if task_status is TaskStatus.FAILED:
            self._status = AgentStatus.FAILED
            if self._last_error is None:
                result = self._skill_manager.task_failure_result
                self._last_error = (
                    "task failed"
                    if result is None
                    else f"task failed: {result.message}"
                )
            return
        if task_status is TaskStatus.CANCELED:
            self._status = self._shutdown_outcome or AgentStatus.CANCELED
            return
        # RUNNING Agent and IDLE manager is an integration invariant violation.
        self._status = AgentStatus.FAILED
        self._last_error = "SkillManager unexpectedly became IDLE"

    def _raise_start_failure(self, prefix: str, exc: Exception) -> None:
        message = self._error_text(prefix, exc)
        self._last_error = message
        self._status = AgentStatus.FAILED
        self._compiled_mission = None
        self._mission_start_time_s = None
        self._safe_log(f"[MissionAgent] start_failed stage={prefix}")
        raise MissionAgentError(message) from exc

    def _read_clock(self) -> float:
        value = self._clock.now()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise MissionAgentError("clock.now() must return a finite number")
        parsed = float(value)
        if not isfinite(parsed):
            raise MissionAgentError("clock.now() must return a finite number")
        return parsed

    def _log_safety(self, decision: SafetyDecision, *, phase: str) -> None:
        self._safe_log(
            f"[MissionAgent] safety phase={phase} "
            f"action={decision.action.value} reason={decision.reason}"
        )

    def _log_skill_transition(self, record: TransitionRecord) -> None:
        self._safe_log(
            "[MissionAgent] skill_transition "
            f"old={_optional_enum_text(record.old_skill)} "
            f"old_step_id={getattr(record, 'old_step_id', None)} "
            f"status={_optional_enum_text(record.old_status)} "
            f"code={_optional_enum_text(record.result_code)} "
            f"new={_optional_enum_text(record.new_skill)} "
            f"new_step_id={getattr(record, 'new_step_id', None)} "
            f"recovery_attempt={getattr(record, 'recovery_attempt', None)} "
            f"reason={record.reason}"
        )

    def _log_new_target_events(self, start_index: int) -> None:
        for event in self._target_manager.events()[start_index:]:
            self._safe_log(
                "[MissionAgent] target_transition "
                f"old={event.old_state.value} new={event.new_state.value} "
                f"reason={event.reason}"
            )

    def _safe_log(self, message: str) -> None:
        if self._logger is None:
            return
        try:
            self._logger(str(message))
        except Exception:
            # Logging is observational and never changes mission execution.
            pass

    @staticmethod
    def _error_text(prefix: str, exc: BaseException) -> str:
        detail = str(exc).strip() or type(exc).__name__
        return f"{prefix}: {detail}"


def _copy_compiled_mission(compiled: CompiledMission) -> CompiledMission:
    """Return a CompiledMission with an independently owned executable plan."""

    plan = TaskPlan(
        tuple(
            TaskStep(
                step.step_id,
                step.skill,
                step.params,
                step.recovery,
            )
            for step in compiled.task_plan.steps
        )
    )
    return CompiledMission(
        planner_output=compiled.planner_output,
        task_plan=plan,
        source=compiled.source,
        compiler_notes=compiled.compiler_notes,
    )


def _valid_observation_timestamp(observation: object) -> float | None:
    if not isinstance(observation, Observation):
        return None
    try:
        value = observation.timestamp
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if isfinite(parsed) else None


def _optional_vector3(
    data: Mapping[str, object],
    key: str,
) -> tuple[float, float, float] | None:
    value = data.get(key)
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if not isinstance(value, Sequence) or len(value) != 3:
        return None
    parsed: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, Real):
            return None
        number = float(component)
        if not isfinite(number):
            return None
        parsed.append(number)
    return parsed[0], parsed[1], parsed[2]


def _optional_finite(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    parsed = float(value)
    return parsed if isfinite(parsed) else None


def _enum_text(value: object) -> str:
    if isinstance(value, Enum):
        raw = value.value
        return raw if isinstance(raw, str) else value.name
    return str(value)


def _optional_enum_text(value: object | None) -> str:
    return "NONE" if value is None else _enum_text(value)


__all__ = [
    "AgentStatus",
    "MissionAgent",
    "MissionAgentError",
    "MissionAgentSnapshot",
]
