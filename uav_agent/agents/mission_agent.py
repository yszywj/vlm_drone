"""Pure-Python integration layer for one high-level UAV mission.

The Agent owns orchestration only.  It receives narrow planner/runtime
dependencies and deliberately has no environment, scene, target-truth, image
model, or Isaac Sim reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from numbers import Real
import inspect

from common.ids import generate_routing_id, validate_mission_id, validate_uav_id
from common.provenance import is_privileged_oracle_source
from perception.runtime import (
    PerceptionBoundaryError,
    PerceptionRuntimeProfile,
    observation_contains_oracle_data,
)
from planner.base import MissionPlanner
from planner.schemas import (
    CompiledMission,
    PlannerRequest,
    PlannerWorldContext,
    SkillPlanDraftV2,
)
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
from agents.visual_review_coordinator import (
    RevisionCompletionAction,
    VisualReviewCoordinator,
)
from agents.plan_revision_coordinator import (
    PlanRevisionCoordinator,
    PlanRevisionState,
)
from agents.program_patch_coordinator import (
    ProgramPatchCoordinator,
    ProgramPatchCoordinatorState,
)
from runtime.events import MissionEvent, MissionEventType
from runtime.world_belief import (
    CandidateSummary,
    QwenRequestState,
    QwenRequestStatus,
    WorldBelief,
)


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
    # Public state must always carry the Agent's explicit routing identity.
    uav_id: str = field(kw_only=True)
    mission_id: str | None = None
    plan_version: int | None = None
    skill_report: dict[str, object] | None = None
    visual_review: dict[str, object] | None = None
    plan_revision: dict[str, object] | None = None
    program_patch: dict[str, object] | None = None

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
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if self.mission_id is not None:
            validate_mission_id(self.mission_id)
        if self.plan_version is not None and (
            isinstance(self.plan_version, bool)
            or not isinstance(self.plan_version, int)
            or self.plan_version <= 0
        ):
            raise ValueError("plan_version must be a positive integer or None")
        if self.skill_report is not None:
            if not isinstance(self.skill_report, dict):
                raise TypeError("skill_report must be a dict or None")
            object.__setattr__(self, "skill_report", deepcopy(self.skill_report))
        if self.visual_review is not None:
            if not isinstance(self.visual_review, dict):
                raise TypeError("visual_review must be a dict or None")
            object.__setattr__(self, "visual_review", deepcopy(self.visual_review))
        if self.plan_revision is not None:
            if not isinstance(self.plan_revision, dict):
                raise TypeError("plan_revision must be a dict or None")
            object.__setattr__(self, "plan_revision", deepcopy(self.plan_revision))
        if self.program_patch is not None:
            if not isinstance(self.program_patch, dict):
                raise TypeError("program_patch must be a dict or None")
            object.__setattr__(self, "program_patch", deepcopy(self.program_patch))

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
            "uav_id": self.uav_id,
            "mission_id": self.mission_id,
            "plan_version": self.plan_version,
            "skill_report": (
                None if self.skill_report is None else deepcopy(self.skill_report)
            ),
            "visual_review": (
                None if self.visual_review is None else deepcopy(self.visual_review)
            ),
            "plan_revision": (
                None if self.plan_revision is None else deepcopy(self.plan_revision)
            ),
            "program_patch": (
                None if self.program_patch is None else deepcopy(self.program_patch)
            ),
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
        uav_id: str | None = None,
        visual_review_coordinator: VisualReviewCoordinator | None = None,
        plan_revision_coordinator: PlanRevisionCoordinator | None = None,
        runtime_program: str = "linear",
        program_patch_coordinator: ProgramPatchCoordinator | None = None,
        target_perception_backend: str | None = None,
    ) -> None:
        if not isinstance(planner, MissionPlanner):
            raise TypeError("planner must be a MissionPlanner")
        allow_trusted_safety_completion = getattr(
            planner,
            "allow_trusted_safety_completion",
            False,
        )
        if not isinstance(allow_trusted_safety_completion, bool):
            raise TypeError(
                "planner.allow_trusted_safety_completion must be bool"
            )
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
        if runtime_program not in {"linear", "graph"}:
            raise ValueError("runtime_program must be linear or graph")
        if target_perception_backend not in {
            None,
            "disabled",
            "oracle_evaluation",
            "ultralytics_service",
        }:
            raise ValueError("unsupported target_perception_backend")
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
        bound_uav_id = (
            skill_manager.uav_id
            if uav_id is None
            else validate_uav_id(uav_id)
        )
        if bound_uav_id != skill_manager.uav_id:
            raise MissionAgentError(
                "MissionAgent.uav_id must match SkillManager.uav_id"
            )
        if visual_review_coordinator is not None:
            if not isinstance(visual_review_coordinator, VisualReviewCoordinator):
                raise TypeError(
                    "visual_review_coordinator must be a VisualReviewCoordinator or None"
                )
            if visual_review_coordinator.uav_id != bound_uav_id:
                raise MissionAgentError(
                    "VisualReviewCoordinator.uav_id must match MissionAgent.uav_id"
                )
            visual_review_coordinator.validate_agent_bindings(
                skill_manager,
                target_manager,
            )
        if plan_revision_coordinator is not None:
            if not isinstance(
                plan_revision_coordinator,
                PlanRevisionCoordinator,
            ):
                raise TypeError(
                    "plan_revision_coordinator must be a "
                    "PlanRevisionCoordinator or None"
                )
            if plan_revision_coordinator.uav_id != bound_uav_id:
                raise MissionAgentError(
                    "PlanRevisionCoordinator.uav_id must match MissionAgent.uav_id"
                )
            plan_revision_coordinator.validate_agent_bindings(
                skill_manager,
                safety,
            )
            if visual_review_coordinator is None:
                raise MissionAgentError(
                    "runtime plan revision requires a VisualReviewCoordinator"
                )
            if runtime_program == "graph":
                raise MissionAgentError(
                    "graph runtime requires ProgramPatch revisions; the "
                    "TaskPlan PlanRevisionCoordinator is unsupported"
                )
        if program_patch_coordinator is not None:
            if not isinstance(
                program_patch_coordinator, ProgramPatchCoordinator
            ):
                raise TypeError(
                    "program_patch_coordinator must be a "
                    "ProgramPatchCoordinator or None"
                )
            if program_patch_coordinator.uav_id != bound_uav_id:
                raise MissionAgentError(
                    "ProgramPatchCoordinator.uav_id must match MissionAgent.uav_id"
                )
            if runtime_program != "graph":
                raise MissionAgentError(
                    "ProgramPatchCoordinator requires runtime_program=graph"
                )

        self._planner = planner
        self._allow_trusted_safety_completion = (
            allow_trusted_safety_completion
        )
        self._validator = validator
        self._safety = safety
        self._skill_manager = skill_manager
        self._target_manager = target_manager
        self._clock = clock
        self._logger = logger
        self._perception_runtime_profile = perception_runtime_profile
        self._uav_id = bound_uav_id
        self._visual_review_coordinator = visual_review_coordinator
        self._plan_revision_coordinator = plan_revision_coordinator
        self._program_patch_coordinator = program_patch_coordinator
        self._runtime_program = runtime_program
        self._target_perception_backend = target_perception_backend

        self._status = AgentStatus.IDLE
        self._compiled_mission: CompiledMission | None = None
        self._transition_cursor = 0
        self._last_observation_timestamp: float | None = None
        self._mission_start_time_s: float | None = None
        self._last_error: str | None = None
        # A runtime safety ABORT must finish LAND as FAILED; an ordinary cancel
        # or CANCEL_AND_LAND finishes as CANCELED.
        self._shutdown_outcome: AgentStatus | None = None
        self._mission_id: str | None = None
        self._plan_version: int | None = None
        self._original_instruction: str | None = None
        self._world_context: PlannerWorldContext | None = None

    @property
    def uav_id(self) -> str:
        return self._uav_id

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
        if self._plan_revision_coordinator is not None:
            try:
                self._plan_revision_coordinator.validate_mission_start(
                    instruction,
                    world_context,
                )
            except Exception as exc:
                raise MissionAgentError(
                    self._error_text("revision coordinator context rejected", exc)
                ) from exc

        self._status = AgentStatus.PLANNING
        self._last_error = None
        self._mission_id = generate_routing_id("mission")
        self._plan_version = 1
        self._safe_log("[MissionAgent] planner_started")
        try:
            request = PlannerRequest(
                instruction=instruction,
                world_context=world_context,
                mission_id=self._mission_id,
                uav_id=self._uav_id,
                plan_version=self._plan_version,
                allow_trusted_safety_completion=(
                    self._allow_trusted_safety_completion
                ),
            )
            planner_output = self._planner.plan(request)
        except Exception as exc:
            self._safe_log("[MissionAgent] planner_finished status=FAILED")
            self._raise_start_failure("planner failed", exc)
        self._safe_log("[MissionAgent] planner_finished status=SUCCEEDED")

        try:
            validator_method = self._validator.validate_and_compile
            validator_parameters = inspect.signature(validator_method).parameters
            if {"mission_id", "uav_id", "plan_version"}.issubset(
                validator_parameters
            ):
                validator_kwargs: dict[str, object] = {
                    "source": self._planner_source(),
                    "mission_id": self._mission_id,
                    "uav_id": self._uav_id,
                    "plan_version": self._plan_version,
                }
                if "allow_trusted_safety_completion" in validator_parameters:
                    validator_kwargs["allow_trusted_safety_completion"] = (
                        self._allow_trusted_safety_completion
                    )
                compiled = validator_method(
                    planner_output,
                    world_context,
                    **validator_kwargs,
                )
                if (
                    compiled.task_plan.mission_id != self._mission_id
                    or compiled.task_plan.uav_id != self._uav_id
                    or compiled.task_plan.plan_version != self._plan_version
                ):
                    raise MissionAgentError(
                        "validator returned mismatched task routing IDs"
                    )
            else:
                # Compatibility for old test/adapter subclasses.  The trusted
                # Agent, never the planner output, binds their compiled plan.
                compiled = validator_method(
                    planner_output,
                    world_context,
                    source=self._planner_source(),
                )
                compiled = _rebind_compiled_mission(
                    compiled,
                    mission_id=self._mission_id,
                    uav_id=self._uav_id,
                    plan_version=self._plan_version,
                )
            owned_compiled = _copy_compiled_mission(compiled)
        except Exception as exc:
            self._raise_start_failure("plan validation failed", exc)

        if self._target_perception_backend == "disabled":
            try:
                from perception.factory import validate_target_perception_preflight

                validate_target_perception_preflight(
                    "disabled",
                    (step.skill for step in owned_compiled.task_plan.steps),
                )
            except Exception as exc:
                self._raise_start_failure("target perception preflight failed", exc)

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
            if self._runtime_program == "graph":
                from planner.mission_program import (
                    ProgramAction,
                    ProgramActionOp,
                    ProgramEvent,
                    ProgramEventHandler,
                    linear_plan_to_mission_program,
                )

                handlers = ()
                if self._program_patch_coordinator is not None:
                    handlers = (
                        ProgramEventHandler(
                            ProgramEvent.PATH_BLOCKED,
                            (
                                ProgramAction(ProgramActionOp.HOLD),
                                ProgramAction(
                                    ProgramActionOp.REPLAN_CURRENT_ROUTE,
                                    planner="QWEN_VL",
                                    allow_model_waypoints=True,
                                ),
                            ),
                        ),
                    )

                task_status = self._skill_manager.start_program(
                    linear_plan_to_mission_program(
                        owned_compiled.task_plan,
                        event_handlers=handlers,
                    )
                )
            else:
                task_status = self._skill_manager.start_task(
                    owned_compiled.task_plan
                )
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
        self._original_instruction = instruction.strip()
        self._world_context = world_context
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
        observation_uav_id = (
            getattr(observation, "uav_id", None)
            if isinstance(observation, Observation)
            else None
        )
        if observation_uav_id is not None and observation_uav_id != self._uav_id:
            raise MissionAgentError(
                "Observation.uav_id does not match this MissionAgent"
            )

        if self._runtime_program == "graph" and self._shutdown_outcome is None:
            try:
                published = self._skill_manager.graph_task_plan_for_adoption()
                synchronized = (
                    self._mission_id is not None
                    and self._plan_version is not None
                    and published.mission_id == self._mission_id
                    and published.uav_id == self._uav_id
                    and published.plan_version == self._plan_version
                )
            except Exception as exc:
                synchronized = False
                sync_error = self._error_text(
                    "graph runtime publication is inconsistent",
                    exc,
                )
            else:
                sync_error = (
                    "graph runtime plan version changed without "
                    "MissionAgent adoption"
                )
            if not synchronized:
                # A ProgramPatch successor is started but never ticked on its
                # publication frame.  This check therefore gives a trusted
                # coordinator exactly one inter-frame adoption window and
                # cancels before an unadopted plan can command motion.
                self._last_error = sync_error
                self._begin_shutdown(AgentStatus.FAILED, sync_error)
                self._consume_transitions()
                self._sync_status()
                return self.snapshot()

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
                "production MissionAgent rejects Oracle fields and "
                "target_estimate.source=oracle_evaluation"
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
                target_estimate=None,
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
                patch_owned_tick = (
                    self._program_patch_coordinator is not None
                    and self._program_patch_coordinator.is_inflight
                )
                if patch_owned_tick:
                    self._tick_program_patch(timestamp)
                revision = self._plan_revision_coordinator
                if patch_owned_tick:
                    # ProgramPatch owns supervisory HOVER and its own routed
                    # worker result for this frame.
                    pass
                elif revision is not None and revision.is_inflight:
                    # The visual and revision planners intentionally share one
                    # per-UAV AsyncModelWorker. While revision owns it, do not
                    # let the visual coordinator drain that routed result as
                    # an orphan. Both branches remain non-blocking.
                    self._tick_plan_revision()
                else:
                    self._tick_visual_review(observation, decision)
                    self._handoff_visual_plan_revision()
                    if revision is not None and revision.is_inflight:
                        self._tick_plan_revision()
                self._tick_manager_or_abort(observation)
                self._adopt_published_program_patch()
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
        if self._plan_revision_coordinator is not None:
            self._plan_revision_coordinator.reset()
        if self._visual_review_coordinator is not None:
            self._visual_review_coordinator.reset()
        if self._program_patch_coordinator is not None:
            self._program_patch_coordinator.reset()
        self._compiled_mission = None
        self._transition_cursor = 0
        self._last_observation_timestamp = None
        self._mission_start_time_s = None
        self._last_error = None
        self._shutdown_outcome = None
        self._mission_id = None
        self._plan_version = None
        self._original_instruction = None
        self._world_context = None
        self._status = AgentStatus.IDLE
        self._safe_log("[MissionAgent] reset status=IDLE")
        return self.snapshot()

    def snapshot(self) -> MissionAgentSnapshot:
        """Return fresh, defensive high-level state only."""

        task_status = self._skill_manager.task_status
        active_name = self._skill_manager.active_name
        feedback: dict[str, object] | None = None
        skill_report: dict[str, object] | None = None
        visual_review: dict[str, object] | None = None
        plan_revision: dict[str, object] | None = None
        program_patch: dict[str, object] | None = None
        if active_name is not None:
            try:
                feedback = self._skill_manager.get_feedback().to_dict()
            except Exception:
                # Feedback is observational; losing it must not alter control.
                feedback = None
        try:
            report = self._skill_manager.get_execution_report()
            skill_report = None if report is None else report.to_dict()
        except Exception:
            skill_report = None
        if self._visual_review_coordinator is not None:
            try:
                visual_review = self._visual_review_coordinator.snapshot().to_dict()
            except Exception:
                visual_review = None
        if self._plan_revision_coordinator is not None:
            try:
                plan_revision = (
                    self._plan_revision_coordinator.snapshot().to_dict()
                )
            except Exception:
                plan_revision = None
        if self._program_patch_coordinator is not None:
            try:
                program_patch = (
                    self._program_patch_coordinator.snapshot().to_dict()
                )
            except Exception:
                program_patch = None
        return MissionAgentSnapshot(
            status=self._status,
            task_status=_enum_text(task_status),
            active_skill=(
                None if active_name is None else _enum_text(active_name)
            ),
            target=self._target_manager.snapshot(),
            feedback=None if feedback is None else deepcopy(feedback),
            last_error=self._last_error,
            uav_id=self._uav_id,
            mission_id=self._mission_id,
            plan_version=self._plan_version,
            skill_report=skill_report,
            visual_review=visual_review,
            plan_revision=plan_revision,
            program_patch=program_patch,
        )

    def submit_review_event(self, event: MissionEvent) -> None:
        """Queue a routed semantic event without exposing the environment.

        Event submission does not call a model and does not change controller
        state. The next distinct, Safety-approved observation lets the review
        coordinator decide whether an asynchronous request is due.
        """

        if self._status is not AgentStatus.RUNNING:
            raise MissionAgentError(
                "submit_review_event requires a RUNNING MissionAgent"
            )
        if self._visual_review_coordinator is None:
            raise MissionAgentError("visual review is not configured")
        if not isinstance(event, MissionEvent):
            raise TypeError("event must be a MissionEvent")
        if (
            event.uav_id != self._uav_id
            or event.mission_id != self._mission_id
            or event.plan_version != self._plan_version
        ):
            raise MissionAgentError("review event routing IDs do not match mission")
        try:
            self._visual_review_coordinator.submit_event(event)
        except Exception as exc:
            raise MissionAgentError(
                self._error_text("review event rejected", exc)
            ) from exc

    def submit_program_event(
        self,
        event: MissionEvent,
        *,
        frame_id: str,
    ) -> None:
        """Start the graph ``PATH_BLOCKED`` ProgramPatch chain.

        This is a trusted runtime boundary for collision/obstacle integration;
        arbitrary review events cannot enter graph control flow.
        """

        coordinator = self._program_patch_coordinator
        if self._status is not AgentStatus.RUNNING or self._shutdown_outcome is not None:
            raise MissionAgentError("program event requires a RUNNING mission")
        if coordinator is None:
            raise MissionAgentError("ProgramPatch runtime is not configured")
        if not isinstance(event, MissionEvent):
            raise TypeError("event must be a MissionEvent")
        if event.event_type is not MissionEventType.PATH_BLOCKED:
            raise MissionAgentError("only PATH_BLOCKED is a graph program event")
        if (
            event.mission_id != self._mission_id
            or event.uav_id != self._uav_id
            or event.plan_version != self._plan_version
        ):
            raise MissionAgentError("program event routing/version mismatch")
        try:
            coordinator.begin(
                expected_plan_version=event.plan_version,
                observation_timestamp_s=event.timestamp_s,
                frame_id=frame_id,
                defer_observation_timestamp_s=event.timestamp_s,
            )
        except Exception as exc:
            raise MissionAgentError(
                self._error_text("program event rejected", exc)
            ) from exc

    def complete_visual_plan_revision(
        self,
        action: RevisionCompletionAction | str,
        *,
        replacement_plan: TaskPlan | None = None,
    ) -> MissionAgentSnapshot:
        """Finish a deferred blocking review after trusted revision work.

        A replacement must already be compiled and validated by the separate
        revision pipeline. The Agent performs a final Safety preflight before
        asking SkillManager to atomically replace the interrupted suffix.
        """

        if self._status is not AgentStatus.RUNNING:
            raise MissionAgentError(
                "complete_visual_plan_revision requires a RUNNING MissionAgent"
            )
        if self._plan_revision_coordinator is not None:
            raise MissionAgentError(
                "automatic PlanRevisionCoordinator owns revision completion"
            )
        coordinator = self._visual_review_coordinator
        compiled = self._compiled_mission
        if coordinator is None or compiled is None:
            raise MissionAgentError("visual review is not configured")
        try:
            selected = RevisionCompletionAction(action)
        except (TypeError, ValueError):
            raise MissionAgentError("revision action must be RESUME or REPLACE") from None

        replacement_compiled: CompiledMission | None = None
        if selected is RevisionCompletionAction.REPLACE:
            if not isinstance(replacement_plan, TaskPlan):
                raise TypeError("REPLACE requires a validated replacement TaskPlan")
            if (
                replacement_plan.mission_id != self._mission_id
                or replacement_plan.uav_id != self._uav_id
                or self._plan_version is None
                or replacement_plan.plan_version != self._plan_version + 1
            ):
                raise MissionAgentError("replacement TaskPlan routing/version mismatch")
            replacement_compiled = CompiledMission(
                planner_output=compiled.planner_output,
                task_plan=replacement_plan,
                source=compiled.source,
                compiler_notes=(
                    *compiled.compiler_notes,
                    "runtime suffix revision preflight-approved",
                ),
            )
            try:
                decision = self._safety.preflight(replacement_compiled)
            except Exception as exc:
                raise MissionAgentError(
                    self._error_text("revision safety preflight failed", exc)
                ) from exc
            if not isinstance(decision, SafetyDecision):
                raise MissionAgentError(
                    "revision safety preflight returned an invalid decision"
                )
            self._log_safety(decision, phase="revision_preflight")
            if decision.action is not SafetyAction.CONTINUE:
                raise MissionAgentError(
                    "revision safety preflight rejected replacement: "
                    f"{decision.action.value}: {decision.reason}"
                )
        elif replacement_plan is not None:
            raise ValueError("RESUME does not accept replacement_plan")

        try:
            coordinator.complete_revision(
                selected,
                replacement_plan=replacement_plan,
            )
        except Exception as exc:
            raise MissionAgentError(
                self._error_text("revision completion failed", exc)
            ) from exc
        if replacement_compiled is not None:
            self._compiled_mission = _copy_compiled_mission(replacement_compiled)
            self._plan_version = replacement_plan.plan_version
        self._safe_log(
            "[MissionAgent] visual_revision_completed "
            f"action={selected.value} plan_version={self._plan_version}"
        )
        return self.snapshot()

    def adopt_runtime_task_plan(self, task_plan: TaskPlan) -> MissionAgentSnapshot:
        """Synchronize Agent routing after a trusted Manager-side replacement.

        Obstacle route publication and experimental ProgramPatch publication
        are intentionally owned by their trusted coordinators.  Once
        ``SkillManager`` has atomically installed either replacement, this
        method advances the Agent's public plan version and compiled-plan view.
        It never dispatches a plan itself and repeats Safety preflight before
        accepting the metadata hand-off.  Graph callers must invoke it in the
        one-tick window after HOVER publishes the replacement and before the
        newly started successor consumes an observation.
        """

        if self._status is not AgentStatus.RUNNING:
            raise MissionAgentError(
                "adopt_runtime_task_plan requires a RUNNING MissionAgent"
            )
        compiled = self._compiled_mission
        if compiled is None or self._mission_id is None or self._plan_version is None:
            raise MissionAgentError("there is no active compiled mission")
        if not isinstance(task_plan, TaskPlan):
            raise TypeError("task_plan must be a TaskPlan")
        if (
            task_plan.mission_id != self._mission_id
            or task_plan.uav_id != self._uav_id
            or task_plan.plan_version != self._plan_version + 1
        ):
            raise MissionAgentError("runtime TaskPlan routing/version mismatch")
        manager_plan = (
            self._skill_manager.graph_task_plan_for_adoption()
            if self._runtime_program == "graph"
            else self._skill_manager.task_plan
        )
        if manager_plan is None or manager_plan.to_dict() != task_plan.to_dict():
            raise MissionAgentError(
                "SkillManager has not atomically installed this runtime TaskPlan"
            )
        replacement = CompiledMission(
            planner_output=compiled.planner_output,
            task_plan=task_plan,
            source=compiled.source,
            compiler_notes=(
                *compiled.compiler_notes,
                "runtime replacement adopted after trusted publication",
            ),
        )
        decision = self._safety.preflight(replacement)
        if not isinstance(decision, SafetyDecision):
            raise MissionAgentError(
                "runtime replacement safety preflight returned an invalid decision"
            )
        self._log_safety(decision, phase="runtime_replacement_preflight")
        if decision.action is not SafetyAction.CONTINUE:
            raise MissionAgentError(
                "runtime replacement safety preflight rejected TaskPlan: "
                f"{decision.action.value}: {decision.reason}"
            )
        self._compiled_mission = _copy_compiled_mission(replacement)
        self._plan_version = task_plan.plan_version
        self._safe_log(
            "[MissionAgent] runtime_replacement_adopted "
            f"plan_version={self._plan_version}"
        )
        return self.snapshot()

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

    def _tick_visual_review(
        self,
        observation: Observation,
        safety_decision: SafetyDecision,
    ) -> None:
        coordinator = self._visual_review_coordinator
        compiled = self._compiled_mission
        if coordinator is None or compiled is None:
            return
        if self._mission_id is None or self._plan_version is None:
            raise MissionAgentError("visual review route is missing")
        active_skill = self._skill_manager.active_name
        active_step_id = self._skill_manager.active_planned_step_id
        feedback: Mapping[str, object] | None = None
        if active_skill is not None:
            try:
                feedback = self._skill_manager.get_feedback().to_dict()
            except Exception:
                feedback = None
        try:
            now = self._read_clock()
            start = self._mission_start_time_s
            elapsed = 0.0 if start is None else max(0.0, now - start)
            coordinator.tick(
                observation,
                mission_id=self._mission_id,
                plan_version=self._plan_version,
                active_skill=active_skill,
                active_step_id=active_step_id,
                target_spec=compiled.target_spec,
                target_snapshot=self._target_manager.snapshot(),
                safety_decision=safety_decision,
                skill_feedback=feedback,
                mission_elapsed_s=elapsed,
            )
        except Exception as exc:
            # A shadow/non-blocking review is observational and can never make
            # an otherwise safe flight fail. If a gate-mode request had already
            # entered HOVER, its trusted manager timeout policy remains active.
            self._safe_log(
                "[MissionAgent] visual_review_error "
                + self._error_text("review tick failed", exc)
            )

    def _handoff_visual_plan_revision(self) -> None:
        """Transfer one visual request while preserving its blocking HOVER."""

        visual = self._visual_review_coordinator
        revision = self._plan_revision_coordinator
        compiled = self._compiled_mission
        if visual is None or revision is None or compiled is None:
            return
        pending = visual.pending_revision
        if pending is None:
            return
        semantic_plan = compiled.planner_output
        if not isinstance(semantic_plan, SkillPlanDraftV2):
            # Legacy plans have no routed/versioned suffix schema.  Do not
            # weaken that boundary merely to satisfy a model recommendation.
            try:
                visual.complete_revision(RevisionCompletionAction.RESUME)
            except Exception as exc:
                self._safe_log(
                    "[MissionAgent] revision_handoff_error "
                    + self._error_text("legacy plan resume failed", exc)
                )
            return
        try:
            belief = self._build_revision_world_belief(
                semantic_plan,
                recent_event=pending.event,
                candidate_id=pending.candidate_id,
            )
            snapshot = revision.submit_event(
                pending.event,
                current_plan=semantic_plan,
                world_belief=belief,
            )
            if snapshot.state is PlanRevisionState.IN_FLIGHT:
                # This is a clear-only ownership transfer. The independent
                # coordinator has actually accepted the request and is now the
                # sole component allowed to resume/replace Manager. A rejected
                # submission leaves the visual coordinator owning its wait.
                visual.acknowledge_revision_handoff(
                    event_id=pending.event.event_id,
                )
            self._safe_log(
                "[MissionAgent] revision_handoff "
                f"state={snapshot.state.value} "
                f"request_id={snapshot.request_id}"
            )
        except Exception as exc:
            # The visual coordinator still owns the wait if acknowledgement
            # did not happen; its trusted HOVER timeout remains authoritative.
            self._safe_log(
                "[MissionAgent] revision_handoff_error "
                + self._error_text("revision request rejected", exc)
            )

    def _tick_plan_revision(self) -> None:
        """Poll the second-stage worker once and synchronize accepted version."""

        coordinator = self._plan_revision_coordinator
        compiled = self._compiled_mission
        if coordinator is None or compiled is None:
            return
        semantic_plan = compiled.planner_output
        if not isinstance(semantic_plan, SkillPlanDraftV2):
            return
        try:
            belief = self._build_revision_world_belief(semantic_plan)
            snapshot = coordinator.tick(
                current_plan=semantic_plan,
                world_belief=belief,
            )
            if snapshot.state is not PlanRevisionState.ACCEPTED:
                return
            accepted = coordinator.latest_accepted_revision
            if accepted is None:
                raise MissionAgentError(
                    "accepted revision has no validated compiled mission"
                )
            revised = accepted.compiled_mission
            if (
                self._mission_id is None
                or self._plan_version is None
                or revised.task_plan.mission_id != self._mission_id
                or revised.task_plan.uav_id != self._uav_id
            ):
                raise MissionAgentError("accepted revision routing mismatch")
            if revised.task_plan.plan_version == self._plan_version:
                return
            if revised.task_plan.plan_version != self._plan_version + 1:
                raise MissionAgentError("accepted revision version jump")
            self._compiled_mission = _copy_compiled_mission(revised)
            self._plan_version = revised.task_plan.plan_version
            self._safe_log(
                "[MissionAgent] revision_adopted "
                f"request_id={snapshot.request_id} "
                f"plan_version={self._plan_version}"
            )
        except Exception as exc:
            # The coordinator owns a trusted HOVER timeout/fallback. A model
            # or parsing failure must not crash the simulation tick.
            self._safe_log(
                "[MissionAgent] plan_revision_error "
                + self._error_text("revision tick failed", exc)
            )

    def _tick_program_patch(self, timestamp_s: float | None) -> None:
        coordinator = self._program_patch_coordinator
        if coordinator is None or not coordinator.is_inflight:
            return
        now = self._read_clock() if timestamp_s is None else timestamp_s
        try:
            coordinator.tick(timestamp_s=now)
        except Exception as exc:
            self._safe_log(
                "[MissionAgent] program_patch_error "
                + self._error_text("ProgramPatch tick failed", exc)
            )

    def _adopt_published_program_patch(self) -> None:
        coordinator = self._program_patch_coordinator
        if (
            coordinator is None
            or coordinator.snapshot().state
            is not ProgramPatchCoordinatorState.ACCEPTED
            or self._plan_version is None
        ):
            return
        try:
            published = self._skill_manager.graph_task_plan_for_adoption()
            if published.plan_version == self._plan_version:
                return
            self.adopt_runtime_task_plan(published)
            coordinator.reset()
        except Exception as exc:
            message = self._error_text("ProgramPatch adoption failed", exc)
            self._last_error = message
            self._begin_shutdown(AgentStatus.FAILED, message)

    def _build_revision_world_belief(
        self,
        semantic_plan: SkillPlanDraftV2,
        *,
        recent_event: MissionEvent | None = None,
        candidate_id: str | None = None,
    ) -> WorldBelief:
        """Create a bounded, image-free main-thread snapshot for revision."""

        step_id = self._skill_manager.active_planned_step_id
        active_name = self._skill_manager.active_name
        if step_id is None or active_name is None:
            raise MissionAgentError("revision requires an active planned step")
        feedback: Mapping[str, object] | None
        try:
            feedback = self._skill_manager.get_feedback().to_dict()
        except Exception:
            feedback = None
        runtime_spec = self._target_manager.target_spec
        target_spec = (
            semantic_plan.target_spec if runtime_spec is None else runtime_spec
        )
        target_snapshot = (
            None if runtime_spec is None else self._target_manager.snapshot()
        )
        revision_snapshot = (
            None
            if self._plan_revision_coordinator is None
            else self._plan_revision_coordinator.snapshot()
        )
        if (
            revision_snapshot is not None
            and revision_snapshot.state is PlanRevisionState.IN_FLIGHT
            and revision_snapshot.request_id is not None
            and revision_snapshot.review_id is not None
            and revision_snapshot.submitted_timestamp_s is not None
        ):
            qwen_status = QwenRequestStatus(
                state=QwenRequestState.IN_FLIGHT,
                request_id=revision_snapshot.request_id,
                review_id=revision_snapshot.review_id,
                blocking=True,
                submitted_timestamp_s=(
                    revision_snapshot.submitted_timestamp_s
                ),
            )
        else:
            qwen_status = QwenRequestStatus()
        latest_frame = None
        if self._visual_review_coordinator is not None:
            try:
                latest_frame = (
                    self._visual_review_coordinator.snapshot().latest_frame_ref
                )
            except Exception:
                latest_frame = None
        now = self._read_clock()
        start = self._mission_start_time_s
        elapsed = 0.0 if start is None else max(0.0, now - start)
        candidate_summaries = (
            ()
            if (
                candidate_id is None
                or recent_event is None
                or not isinstance(recent_event.payload.get("source"), str)
                or not recent_event.payload["source"].strip()
            )
            else (
                CandidateSummary(
                    candidate_id=candidate_id,
                    confidence=None,
                    last_seen_timestamp_s=recent_event.timestamp_s,
                    source=recent_event.payload["source"],
                    observation_count=1,
                ),
            )
        )
        return WorldBelief(
            mission_id=semantic_plan.mission_id,
            uav_id=semantic_plan.uav_id,
            plan_version=semantic_plan.plan_version,
            current_step_id=step_id,
            current_skill=_enum_text(active_name),
            skill_feedback=feedback,
            target_spec=target_spec,
            target_snapshot=target_snapshot,
            candidate_summaries=candidate_summaries,
            recent_events=(() if recent_event is None else (recent_event,)),
            qwen_request_status=qwen_status,
            latest_frame_ref=latest_frame,
            mission_elapsed_s=elapsed,
            plan_id=f"plan_{semantic_plan.plan_version}",
        )

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
        if self._visual_review_coordinator is not None:
            try:
                self._visual_review_coordinator.abort_pending(
                    reason_code="MISSION_SHUTDOWN"
                )
            except Exception as exc:
                # Releasing auxiliary model work must never prevent the
                # trusted cancel-and-land path.
                self._safe_log(
                    "[MissionAgent] visual_review_abort_error "
                    + self._error_text("could not abort visual review", exc)
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
            if self._visual_review_coordinator is not None:
                try:
                    self._visual_review_coordinator.observe_skill_transition(record)
                except Exception as exc:
                    # Event publication is diagnostic and may never turn an
                    # otherwise stable supervisory HOVER into a failed flight.
                    self._safe_log(
                        "[MissionAgent] hold_event_error "
                        + self._error_text(
                            "could not publish Skill transition event",
                            exc,
                        )
                    )
            event_count = len(self._target_manager.events())
            try:
                self._apply_target_transition(record)
            finally:
                self._log_new_target_events(event_count)

    def _apply_target_transition(self, record: TransitionRecord) -> None:
        if not isinstance(record, TransitionRecord):
            raise MissionAgentError("SkillManager emitted an invalid transition")
        if record.uav_id != self._uav_id:
            raise MissionAgentError(
                "SkillManager transition uav_id does not match MissionAgent"
            )
        compiled = self._compiled_mission
        if compiled is None:
            raise MissionAgentError("compiled mission is missing")

        if record.new_skill is SkillName.SEARCH:
            if (
                record.old_skill is SkillName.SEARCH
                and self._target_manager.lifecycle is TargetLifecycle.SEARCHING
            ):
                # Consecutive SEARCH steps are one bounded fallback chain for
                # the same immutable TargetSpec.  Exhausting a region does not
                # create a second target lifecycle.
                return
            if (
                record.old_skill is SkillName.INSPECT
                and self._target_manager.lifecycle
                in {TargetLifecycle.SEARCHING, TargetLifecycle.CANDIDATE}
            ):
                # A runtime suffix revision may temporarily replace an
                # interrupted SEARCH with INSPECT and then return to the same
                # semantic search. Preserve the existing TargetSpec/candidate
                # lifecycle instead of fabricating a second search target.
                return
            if self._target_manager.lifecycle is not TargetLifecycle.UNINITIALIZED:
                raise MissionAgentError("SEARCH entered with an active target")
            # The initial planner identity is immutable. SEARCH's compiled
            # string remains useful for legacy plan compatibility, but a routed
            # schema-v2 plan must use its structured TargetSpec throughout the
            # target lifecycle.
            self._search_target_description(getattr(record, "new_step_id", None))
            self._target_manager.start_search(
                compiled.target_spec, record.timestamp
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
        if (
            snapshot.source is None
            or is_privileged_oracle_source(snapshot.source)
        ):
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
            f"uav_id={record.uav_id} mission_id={record.mission_id} "
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
            mission_id = self._mission_id or "NONE"
            plan_version = self._plan_version or 0
            self._logger(
                f"[Routing] uav_id={self._uav_id} "
                f"mission_id={mission_id} plan_version={plan_version} "
                f"{message}"
            )
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
        ),
        mission_id=compiled.task_plan.mission_id,
        uav_id=compiled.task_plan.uav_id,
        plan_version=compiled.task_plan.plan_version,
    )
    return CompiledMission(
        planner_output=compiled.planner_output,
        task_plan=plan,
        source=compiled.source,
        compiler_notes=compiled.compiler_notes,
    )


def _rebind_compiled_mission(
    compiled: CompiledMission,
    *,
    mission_id: str,
    uav_id: str,
    plan_version: int,
) -> CompiledMission:
    """Trusted compatibility adapter for validators predating routing IDs."""

    plan = TaskPlan(
        tuple(
            TaskStep(step.step_id, step.skill, step.params, step.recovery)
            for step in compiled.task_plan.steps
        ),
        mission_id=mission_id,
        uav_id=uav_id,
        plan_version=plan_version,
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
