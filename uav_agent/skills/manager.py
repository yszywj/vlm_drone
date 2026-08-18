"""Single-Skill dispatch and generic bounded linear task execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, dataclass, fields, replace
from enum import Enum, auto
from math import isfinite
from numbers import Real

from common.ids import (
    validate_invocation_id,
    validate_mission_id,
    validate_routing_id,
    validate_uav_id,
)
from skills.base import Skill, SkillLifecycleError
from skills.goto import GotoGoal, GotoSkill
from skills.hover import (
    HoverGoal,
    HoverMode,
    HoverSkill,
    HoverTimeoutFallback,
)
from skills.land import LandGoal, LandSkill
from skills.motion_types import MotionPolicy, YawMode
from skills.plan import RecoveryPolicy, StepOutputRef, TaskPlan, TaskPlanError, TaskStep
from skills.reacquire import ReacquireGoal, ReacquireSkill
from skills.search import SearchGoal, SearchSkill
from skills.takeoff import TakeoffGoal, TakeoffSkill
from skills.track import TrackGoal, TrackSkill
from skills.types import (
    Observation,
    SkillContext,
    SkillFeedback,
    SkillExecutionReport,
    SkillGoal,
    SkillInvocation,
    SkillName,
    SkillResult,
    SkillResultCode,
    SkillStatus,
)


class SkillManagerError(RuntimeError):
    """Base class for Manager registration, plan, and active-Skill errors."""


class SkillNotRegisteredError(SkillManagerError):
    """Raised when a requested tool name has no registered Skill."""


class TaskStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELED = auto()


class ExecutionKind(str, Enum):
    """Whether the active Skill is planned work or internal recovery."""

    PLANNED = "PLANNED"
    RECOVERY = "RECOVERY"
    EMERGENCY = "EMERGENCY"
    SUPERVISORY = "SUPERVISORY"


@dataclass(frozen=True, slots=True)
class _InterruptedExecution:
    """Owned snapshot of trusted runtime state at a soft interruption."""

    plan_index: int
    step: TaskStep
    resolved_goal: SkillGoal
    plan: TaskPlan
    step_outputs: dict[str, dict[str, object]]
    active_target_id: str | None
    recovery_attempts: dict[str, int]
    saved_track_goals: dict[str, TrackGoal]
    timeout_fallback: HoverTimeoutFallback


class _SupervisoryContinuation(str, Enum):
    RESUME = "RESUME"
    REPLACE = "REPLACE"
    SEARCH_CANDIDATE_HANDOFF = "SEARCH_CANDIDATE_HANDOFF"


@dataclass(frozen=True, slots=True)
class _PendingSearchCandidateHandoff:
    """Trusted metadata for completing SEARCH without claiming a target lock."""

    candidate_id: str
    source: str


@dataclass(frozen=True, slots=True)
class _SearchInspectionDetour:
    """One INSPECT detour that must return to the exact saved SEARCH.

    The validated plan remains ``SEARCH -> INSPECT -> suffix``. Runtime visits
    INSPECT first, restarts the interrupted SEARCH, and skips the already-run
    INSPECT only after SEARCH produces its real ``target_id`` output.
    """

    search_index: int
    inspect_index: int
    search_step: TaskStep
    search_goal: SearchGoal
    inspect_step_id: str
    candidate_id: str


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One task transition, including its planned-step identity."""

    timestamp: float
    old_skill: SkillName | None
    old_status: SkillStatus | None
    result_code: SkillResultCode | None
    new_skill: SkillName | None
    reason: str
    old_step_id: str | None = None
    new_step_id: str | None = None
    recovery_attempt: int | None = None
    uav_id: str = "uav_1"
    mission_id: str = "mission_legacy"
    plan_version: int = 1
    invocation_id: str | None = None

    def __post_init__(self) -> None:
        validate_uav_id(self.uav_id)
        validate_mission_id(self.mission_id)
        if isinstance(self.plan_version, bool) or not isinstance(
            self.plan_version, int
        ) or self.plan_version <= 0:
            raise ValueError("plan_version must be a positive integer")
        if self.invocation_id is not None:
            validate_invocation_id(self.invocation_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "old_skill": None if self.old_skill is None else self.old_skill.value,
            "old_status": None if self.old_status is None else self.old_status.name,
            "result_code": None if self.result_code is None else self.result_code.name,
            "new_skill": None if self.new_skill is None else self.new_skill.value,
            "reason": self.reason,
            "old_step_id": self.old_step_id,
            "new_step_id": self.new_step_id,
            "recovery_attempt": self.recovery_attempt,
            "uav_id": self.uav_id,
            "mission_id": self.mission_id,
            "plan_version": self.plan_version,
            "invocation_id": self.invocation_id,
        }


def create_default_skill_registry(
    *,
    transit_yaw_mode: YawMode | str = YawMode.FACE_POINT,
    inspect_skill: Skill | None = None,
) -> dict[SkillName, Skill]:
    registry: dict[SkillName, Skill] = {
        SkillName.TAKEOFF: TakeoffSkill(),
        SkillName.GOTO: GotoSkill(),
        SkillName.HOVER: HoverSkill(),
        SkillName.SEARCH: SearchSkill(transit_yaw_mode=transit_yaw_mode),
        SkillName.TRACK: TrackSkill(),
        SkillName.REACQUIRE: ReacquireSkill(),
        SkillName.LAND: LandSkill(),
    }
    # INSPECT owns runtime-specific CandidateBank/Resolver/FrameStore
    # dependencies, so callers must inject the correctly routed instance.
    if inspect_skill is not None:
        from skills.inspect import InspectSkill

        if not isinstance(inspect_skill, InspectSkill):
            raise TypeError("inspect_skill must be an InspectSkill or None")
        registry[SkillName.INSPECT] = inspect_skill
    return registry


class SkillManager:
    """Dispatch a Skill manually or execute an arbitrary validated linear plan.

    In task mode every external :meth:`tick` invokes at most one active
    ``Skill.tick``.  It may start a successor, but never ticks that successor
    until the next external sample.
    """

    def __init__(
        self,
        context: SkillContext,
        *,
        registry: Mapping[SkillName | str, Skill] | None = None,
        reacquire_search_radius: float = 10.0,
        reacquire_timeout: float = 30.0,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(context, SkillContext):
            raise TypeError("context must be a SkillContext")
        context.validate()
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable or None")
        self._context = context
        self._uav_id = validate_uav_id(context.uav_id)
        self._logger = logger
        self._reacquire_search_radius = _positive_number(
            reacquire_search_radius, "reacquire_search_radius"
        )
        self._reacquire_timeout = _positive_number(
            reacquire_timeout, "reacquire_timeout"
        )
        self._skills: dict[SkillName, Skill] = {}
        self._active_name: SkillName | None = None
        self._last_result: SkillResult | None = None

        self._task_status = TaskStatus.IDLE
        self._task_plan: TaskPlan | None = None
        self._plan_index: int | None = None
        self._pending_task_result: TaskStatus | None = None
        self._active_target_id: str | None = None
        self._task_failure_result: SkillResult | None = None
        self._transition_log: list[TransitionRecord] = []
        self._last_transition_time = 0.0

        self._step_outputs: dict[str, dict[str, object]] = {}
        self._recovery_attempts: dict[str, int] = {}
        self._active_planned_step_id: str | None = None
        self._active_execution_kind: ExecutionKind | None = None
        self._saved_track_goal_by_step: dict[str, TrackGoal] = {}
        self._active_invocation: SkillInvocation | None = None
        self._last_invocation: SkillInvocation | None = None
        self._execution_reports: list[SkillExecutionReport] = []
        self._invocation_counter = 0
        self._interrupted_execution: _InterruptedExecution | None = None
        self._supervisory_continuation: _SupervisoryContinuation | None = None
        self._pending_replacement_plan: TaskPlan | None = None
        self._pending_search_candidate_handoff: (
            _PendingSearchCandidateHandoff | None
        ) = None
        self._search_inspection_detour: _SearchInspectionDetour | None = None
        self._supervisory_waiting = False

        if registry is not None:
            if not isinstance(registry, Mapping):
                raise TypeError("registry must be a mapping or None")
            for name, skill in registry.items():
                self.register(name, skill)

    @property
    def active_name(self) -> SkillName | None:
        return self._active_name

    @property
    def uav_id(self) -> str:
        return self._uav_id

    @property
    def active_invocation(self) -> SkillInvocation | None:
        return self._active_invocation

    @property
    def execution_reports(self) -> tuple[SkillExecutionReport, ...]:
        return tuple(_copy_report(report) for report in self._execution_reports)

    @property
    def active_status(self) -> SkillStatus | None:
        skill = self._active_skill_or_none()
        return None if skill is None else skill.status

    @property
    def last_result(self) -> SkillResult | None:
        return _copy_result(self._last_result)

    @property
    def task_status(self) -> TaskStatus:
        return self._task_status

    @property
    def task_plan(self) -> TaskPlan | None:
        return None if self._task_plan is None else _copy_task_plan(self._task_plan)

    @property
    def pending_task_result(self) -> TaskStatus | None:
        return self._pending_task_result

    @property
    def active_target_id(self) -> str | None:
        return self._active_target_id

    @property
    def active_planned_step_id(self) -> str | None:
        return self._active_planned_step_id

    @property
    def active_execution_kind(self) -> ExecutionKind | None:
        return self._active_execution_kind

    @property
    def is_supervisory_paused(self) -> bool:
        """Whether a task owns an interrupted step awaiting/residing in HOVER."""

        return self._interrupted_execution is not None

    @property
    def step_outputs(self) -> dict[str, dict[str, object]]:
        return deepcopy(self._step_outputs)

    @property
    def recovery_attempts(self) -> dict[str, int]:
        return dict(self._recovery_attempts)

    @property
    def task_failure_result(self) -> SkillResult | None:
        return _copy_result(self._task_failure_result)

    @property
    def transition_log(self) -> tuple[TransitionRecord, ...]:
        return tuple(self._transition_log)

    @property
    def skill_registry(self) -> dict[SkillName, Skill]:
        return dict(self._skills)

    def register(self, name: SkillName | str, skill: Skill) -> None:
        normalized = _manager_skill_name(name)
        if not isinstance(skill, Skill):
            raise TypeError("skill must be a Skill instance")
        if skill.status is not SkillStatus.IDLE:
            raise SkillManagerError("only an IDLE Skill can be registered")
        if normalized in self._skills:
            raise SkillManagerError(f"Skill {normalized.value} is already registered")
        if any(registered is skill for registered in self._skills.values()):
            raise SkillManagerError("the same Skill instance cannot be registered twice")
        self._skills[normalized] = skill

    def available_skills(self) -> tuple[SkillName, ...]:
        return tuple(self._skills)

    def report_candidate_pending(
        self,
        candidate_id: str,
        *,
        source: str,
    ) -> None:
        """Forward trusted provisional evidence to the active SEARCH only.

        This narrow main-thread API does not confirm a target, start another
        Skill, or change the plan. It exists so asynchronous perception
        orchestration never reaches into Manager/Skill private state.
        """

        if self._task_status is not TaskStatus.RUNNING:
            raise SkillManagerError("candidate reporting requires a RUNNING task")
        if (
            self._active_name is not SkillName.SEARCH
            or self._active_execution_kind is not ExecutionKind.PLANNED
            or self.active_status is not SkillStatus.RUNNING
        ):
            raise SkillManagerError(
                "candidate reporting requires the active planned SEARCH"
            )
        skill = self._active_skill()
        if not isinstance(skill, SearchSkill):
            raise SkillManagerError("registered SEARCH does not support candidate reports")
        skill.report_candidate_pending(candidate_id, source=source)

    def report_search_candidate_pending(
        self,
        candidate_id: str,
        *,
        source: str,
    ) -> None:
        """Backward-compatible, explicit alias for SEARCH candidate reports."""

        self.report_candidate_pending(candidate_id, source=source)

    # Manual single-Skill compatibility API.
    def start(self, name: SkillName | str, goal: SkillGoal) -> SkillStatus:
        if self._task_status is not TaskStatus.IDLE:
            raise SkillManagerError("reset the task before using manual Skill dispatch")
        return self._start_registered(_manager_skill_name(name), goal)

    def invoke(self, invocation: SkillInvocation) -> SkillStatus:
        """Start one manually dispatched routed invocation."""

        if not isinstance(invocation, SkillInvocation):
            raise TypeError("invocation must be a SkillInvocation")
        if invocation.uav_id != self._uav_id:
            raise SkillManagerError(
                "SkillInvocation.uav_id does not match this SkillManager"
            )
        if self._task_status is not TaskStatus.IDLE:
            raise SkillManagerError("cannot invoke manually while a task is active")
        return self._start_registered(
            invocation.skill_name,
            invocation.goal,
            invocation=invocation,
        )

    def tick(self, observation: Observation) -> SkillStatus | TaskStatus:
        if (
            isinstance(observation, Observation)
            and getattr(observation, "uav_id", self._uav_id) != self._uav_id
        ):
            raise SkillManagerError(
                "Observation.uav_id does not match this SkillManager"
            )
        if self._task_status is TaskStatus.RUNNING:
            return self._tick_task(observation)
        if self._task_status is not TaskStatus.IDLE:
            return self._task_status
        return self._active_skill().tick(observation)

    def cancel_active(self) -> None:
        if self._task_status is TaskStatus.RUNNING:
            raise SkillManagerError("use cancel_task() while task mode is active")
        self._active_skill().cancel()

    def get_feedback(self) -> SkillFeedback:
        return self._active_skill().get_feedback()

    def get_result(self) -> SkillResult | None:
        return self._active_skill().get_result()

    def get_execution_report(self) -> SkillExecutionReport | None:
        """Return routed feedback/result without exposing an unbound payload."""

        if self._active_name is not None and self._active_invocation is not None:
            skill = self._active_skill()
            result = skill.get_result()
            if result is None:
                payload = skill.get_feedback().to_dict()
                result_code = None
            else:
                payload = result.to_dict()
                result_code = result.code
            return SkillExecutionReport(
                mission_id=self._active_invocation.mission_id,
                uav_id=self._active_invocation.uav_id,
                plan_version=self._active_invocation.plan_version,
                step_id=self._active_invocation.step_id,
                invocation_id=self._active_invocation.invocation_id,
                skill_name=self._active_invocation.skill_name,
                status=skill.status,
                result_code=result_code,
                feedback_or_result=payload,
                timestamp_s=self._read_transition_time(),
            )
        if self._execution_reports:
            return _copy_report(self._execution_reports[-1])
        return None

    def reset_active(self) -> None:
        if self._task_status is TaskStatus.RUNNING:
            raise SkillManagerError(
                "active Skill lifecycle is Manager-owned while task mode is active"
            )
        self._reset_active_internal()

    def _reset_active_internal(self) -> None:
        skill = self._active_skill()
        result = skill.get_result()
        if result is None:
            raise SkillLifecycleError("active Skill has no terminal result to reset")
        invocation = self._active_invocation
        if invocation is not None:
            report = SkillExecutionReport(
                mission_id=invocation.mission_id,
                uav_id=invocation.uav_id,
                plan_version=invocation.plan_version,
                step_id=invocation.step_id,
                invocation_id=invocation.invocation_id,
                skill_name=invocation.skill_name,
                status=result.status,
                result_code=result.code,
                feedback_or_result=result.to_dict(),
                timestamp_s=self._read_transition_time(),
            )
            self._execution_reports.append(report)
        try:
            skill.reset()
        finally:
            self._last_result = result
            self._active_name = None
            self._last_invocation = invocation
            self._active_invocation = None

    def start_task(self, plan: TaskPlan) -> TaskStatus:
        if not isinstance(plan, TaskPlan):
            raise TypeError("plan must be a TaskPlan")
        if plan.uav_id != self._uav_id:
            raise SkillManagerError(
                "TaskPlan.uav_id does not match this SkillManager"
            )
        if self._task_status is not TaskStatus.IDLE:
            raise SkillManagerError("reset_task() is required before starting another task")
        if self._active_name is not None:
            raise SkillManagerError("reset the manually active Skill before starting a task")
        # TaskStep is frozen, but its parameter Mapping intentionally remains
        # mutable for legacy compatibility and safety corruption tests.  Take
        # one owned snapshot here so caller mutations after preflight/start
        # cannot alter a future Goal or output reference (TOCTOU boundary).
        owned_plan = _copy_task_plan(plan)
        required = {step.skill for step in owned_plan.steps} | {SkillName.LAND}
        if any(
            self._effective_recovery_policy(step) is not None
            for step in owned_plan.steps
        ):
            required.add(SkillName.REACQUIRE)
        missing = sorted(name.value for name in required if name not in self._skills)
        if missing:
            raise SkillNotRegisteredError(
                "task registry is missing: " + ", ".join(missing)
            )

        # Validate every Goal shape before mutating task state. Runtime output
        # references are checked structurally but resolved only when reached.
        for step in owned_plan.steps:
            _reject_unknown_goal_fields(step.skill, step.params)
            self._goal_from_step(step, plan=owned_plan, validation_only=True)

        self._task_plan = owned_plan
        self._last_result = None
        self._plan_index = 0
        self._pending_task_result = None
        self._active_target_id = None
        self._task_failure_result = None
        self._transition_log = []
        self._last_transition_time = 0.0
        self._step_outputs = {}
        self._recovery_attempts = {
            step.step_id: 0
            for step in owned_plan.steps
            if self._effective_recovery_policy(step) is not None
        }
        self._saved_track_goal_by_step = {}
        self._active_invocation = None
        self._last_invocation = None
        self._execution_reports = []
        self._invocation_counter = 0
        self._discard_supervisory_state()
        self._active_planned_step_id = owned_plan.steps[0].step_id
        self._active_execution_kind = ExecutionKind.PLANNED
        self._task_status = TaskStatus.RUNNING

        first = owned_plan.steps[0]
        goal = self._goal_from_step(first)
        try:
            self._start_registered(first.skill, goal)
        except Exception:
            self._clear_task_to_idle()
            raise
        if first.skill is SkillName.TRACK and isinstance(goal, TrackGoal):
            self._saved_track_goal_by_step[first.step_id] = goal
        self._record_transition(
            old_skill=None,
            old_status=None,
            result_code=None,
            new_skill=first.skill,
            reason="task_started",
            old_step_id=None,
            new_step_id=first.step_id,
        )
        return self._task_status

    def cancel_task(self) -> TaskStatus:
        if self._task_status is not TaskStatus.RUNNING:
            raise SkillManagerError("there is no RUNNING task to cancel")
        old_name = self._active_name
        old_step_id = self._active_planned_step_id
        if old_name is SkillName.LAND:
            if self._pending_task_result is not TaskStatus.FAILED:
                self._pending_task_result = TaskStatus.CANCELED
            return self._task_status
        old_status: SkillStatus | None = None
        result: SkillResult | None = None
        if old_name is not None:
            skill = self._active_skill()
            if skill.status is SkillStatus.RUNNING:
                skill.cancel()
            old_status = skill.status
            result = skill.get_result()
            try:
                self._reset_active_internal()
            except Exception as exc:
                failure = _internal_result(
                    f"could not reset canceled {old_name.value}: {exc}",
                    {"reset_error": str(exc)},
                )
                self._last_result = failure
                self._task_failure_result = failure
                self._pending_task_result = TaskStatus.FAILED
                self._begin_landing(
                    old_name,
                    old_status,
                    failure,
                    "canceled_skill_reset_failed",
                    old_step_id=old_step_id,
                )
                return self._task_status
        self._pending_task_result = TaskStatus.CANCELED
        self._begin_landing(
            old_name,
            old_status,
            result,
            "task_canceled",
            old_step_id=old_step_id,
        )
        return self._task_status

    def interrupt_with_hover(
        self,
        reason_code: str,
        *,
        max_wait_s: float = 20.0,
        position_tolerance_m: float = 0.25,
        max_correction_speed_mps: float = 0.5,
        timeout_fallback: HoverTimeoutFallback | str = (
            HoverTimeoutFallback.CANCEL_AND_LAND
        ),
        motion_policy: MotionPolicy | None = None,
    ) -> TaskStatus:
        """Soft-pause one idempotent planned Skill in continuously commanded HOVER.

        This API is a trusted-runtime boundary.  In particular, its hold
        tolerance, correction limit, timeout, and fallback are not sourced
        from model output.
        """

        if self._task_status is not TaskStatus.RUNNING:
            raise SkillManagerError("supervisory HOVER requires a RUNNING task")
        if self._interrupted_execution is not None:
            raise SkillManagerError("a supervisory interruption is already active")
        if self._active_name is None or self._active_name.value not in {
            "GOTO",
            "SEARCH",
            "INSPECT",
            "TRACK",
        }:
            active = "NONE" if self._active_name is None else self._active_name.value
            raise SkillManagerError(
                f"Skill {active} cannot be interrupted by supervisory HOVER"
            )
        if self._active_execution_kind is not ExecutionKind.PLANNED:
            raise SkillManagerError(
                "only a planned Skill can be interrupted by supervisory HOVER"
            )
        if self.active_status is not SkillStatus.RUNNING:
            raise SkillManagerError("only a RUNNING Skill can be interrupted")
        if (
            self._task_plan is None
            or self._plan_index is None
            or self._active_planned_step_id is None
            or self._active_invocation is None
        ):
            raise SkillManagerError("task state is incomplete at interruption boundary")
        step = self._task_plan.steps[self._plan_index]
        if (
            step.step_id != self._active_planned_step_id
            or step.skill is not self._active_name
        ):
            raise SkillManagerError("active Skill does not match the current plan step")

        fallback = _hover_timeout_fallback(timeout_fallback)
        normalized_reason = validate_routing_id(reason_code, "reason_code")
        policy = (
            MotionPolicy(yaw_mode=YawMode.KEEP_CURRENT)
            if motion_policy is None
            else motion_policy
        )
        if not isinstance(policy, MotionPolicy):
            raise TypeError("motion_policy must be a MotionPolicy or None")
        policy.validate()
        goal = HoverGoal(
            mode=HoverMode.UNTIL_RELEASED,
            duration_s=None,
            max_wait_s=_positive_number(max_wait_s, "max_wait_s"),
            position_tolerance_m=_positive_number(
                position_tolerance_m, "position_tolerance_m"
            ),
            max_correction_speed_mps=_positive_number(
                max_correction_speed_mps, "max_correction_speed_mps"
            ),
            reason_code=normalized_reason,
            motion_policy=policy,
        )
        # Validate all trusted values before canceling the active Skill.
        _validate_hover_goal_for_manager(goal, supervisory=True)

        resolved_goal = deepcopy(self._active_invocation.goal)
        if self._active_name is SkillName.TRACK:
            if not isinstance(resolved_goal, TrackGoal):
                raise SkillManagerError("active TRACK invocation has an invalid Goal")
            trusted_target = self._active_target_id
            if trusted_target is not None and resolved_goal.target_id != trusted_target:
                raise SkillManagerError(
                    "active TRACK target_id disagrees with trusted target state"
                )
        interrupted = _InterruptedExecution(
            plan_index=self._plan_index,
            step=TaskStep(step.step_id, step.skill, step.params, step.recovery),
            resolved_goal=resolved_goal,
            plan=_copy_task_plan(self._task_plan),
            step_outputs=deepcopy(self._step_outputs),
            active_target_id=self._active_target_id,
            recovery_attempts=dict(self._recovery_attempts),
            saved_track_goals=deepcopy(self._saved_track_goal_by_step),
            timeout_fallback=fallback,
        )

        old_name = self._active_name
        old_step_id = self._active_planned_step_id
        skill = self._active_skill()
        skill.cancel()
        canceled = skill.get_result()
        self._reset_active_internal()
        self._interrupted_execution = interrupted
        self._supervisory_continuation = None
        self._pending_replacement_plan = None
        self._supervisory_waiting = False
        self._start_transition(
            old_name,
            SkillStatus.CANCELED,
            None if canceled is None else canceled.code,
            SkillName.HOVER,
            goal,
            "supervisory_hover_started",
            old_step_id=old_step_id,
            new_step_id=old_step_id,
            execution_kind=ExecutionKind.SUPERVISORY,
        )
        return self._task_status

    def release_supervisory_hover(self) -> TaskStatus:
        """Thread-safe release request; HOVER completes on a later task tick."""

        self._require_supervisory_interruption()
        if self._active_name is not SkillName.HOVER:
            raise SkillManagerError("supervisory HOVER is not currently RUNNING")
        skill = self._active_skill()
        if not isinstance(skill, HoverSkill) or skill.status is not SkillStatus.RUNNING:
            raise SkillManagerError("supervisory HOVER is not releasable")
        skill.request_release()
        self._record_transition(
            old_skill=SkillName.HOVER,
            old_status=SkillStatus.RUNNING,
            result_code=None,
            new_skill=SkillName.HOVER,
            reason="supervisory_hover_release_requested",
            old_step_id=self._active_planned_step_id,
            new_step_id=self._active_planned_step_id,
        )
        return self._task_status

    def resume_interrupted_step(self) -> TaskStatus:
        """Resume the exact owned Goal saved at the interruption boundary."""

        self._require_supervisory_interruption()
        if self._supervisory_continuation is not None:
            raise SkillManagerError("a supervisory continuation is already selected")
        self._supervisory_continuation = _SupervisoryContinuation.RESUME
        if self._active_name is SkillName.HOVER:
            return self.release_supervisory_hover()
        if not self._supervisory_waiting or self._active_name is not None:
            raise SkillManagerError("supervisory HOVER has not reached a resumable state")
        self._resume_saved_interruption(
            old_status=SkillStatus.SUCCEEDED,
            result_code=SkillResultCode.HOVER_COMPLETE,
            reason="interrupted_step_resumed",
        )
        return self._task_status

    def replace_interrupted_step_and_suffix(self, plan: TaskPlan) -> TaskStatus:
        """Atomically select a validated current-step/suffix replacement.

        The caller supplies a complete routed plan.  The already-completed
        prefix must be byte-for-byte equivalent at the TaskStep data level,
        and the plan version must advance by exactly one.
        """

        interrupted = self._require_supervisory_interruption()
        if self._supervisory_continuation is not None:
            raise SkillManagerError("a supervisory continuation is already selected")
        owned = self._validate_replacement_plan(plan, interrupted)
        self._pending_replacement_plan = owned
        self._supervisory_continuation = _SupervisoryContinuation.REPLACE
        self._record_transition(
            old_skill=self._active_name,
            old_status=self.active_status,
            result_code=None,
            new_skill=self._active_name,
            reason="plan_suffix_replacement_accepted",
            old_step_id=self._active_planned_step_id,
            new_step_id=self._active_planned_step_id,
        )
        if self._active_name is SkillName.HOVER:
            return self.release_supervisory_hover()
        if not self._supervisory_waiting or self._active_name is not None:
            raise SkillManagerError("supervisory HOVER has not reached a replaceable state")
        self._start_replacement_after_interruption(
            old_status=SkillStatus.SUCCEEDED,
            result_code=SkillResultCode.HOVER_COMPLETE,
        )
        return self._task_status

    def handoff_interrupted_search_candidate_to_inspect(
        self,
        plan: TaskPlan,
        *,
        candidate_id: str,
        source: str,
    ) -> TaskStatus:
        """Run INSPECT as a detour, then restart the interrupted SEARCH.

        This is deliberately narrower than ordinary suffix replacement.  The
        replacement must retain the interrupted SEARCH byte-for-byte at the
        same index and put an INSPECT for the trusted candidate immediately
        after it. SEARCH is not completed here and receives no synthetic
        output; only the restarted SEARCH may later publish ``target_id``.

        All validation completes before continuation state is selected, so a
        rejected call leaves the active HOVER and authoritative plan intact.
        """

        interrupted = self._require_supervisory_interruption()
        if self._supervisory_continuation is not None:
            raise SkillManagerError("a supervisory continuation is already selected")
        normalized_candidate_id = validate_routing_id(
            candidate_id,
            "candidate_id",
        )
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        normalized_source = source.strip()
        if normalized_source not in {"qwen_vl", "oracle_evaluation"}:
            raise SkillManagerError(
                "candidate handoff source must be qwen_vl or oracle_evaluation"
            )
        if interrupted.step.skill is not SkillName.SEARCH:
            raise SkillManagerError(
                "candidate handoff requires an interrupted planned SEARCH"
            )
        if interrupted.active_target_id is not None:
            raise SkillManagerError(
                "candidate handoff cannot inherit an already locked target"
            )
        if interrupted.step.step_id in interrupted.step_outputs:
            raise SkillManagerError(
                "interrupted SEARCH already has a completed step output"
            )
        if self._search_inspection_detour is not None:
            raise SkillManagerError(
                "the interrupted SEARCH already consumed an INSPECT detour"
            )

        # This validates route, exact +1 version, completed prefix, registry,
        # and every Goal shape without changing Manager state.
        owned = self._validate_replacement_plan(plan, interrupted)
        index = interrupted.plan_index
        if owned.steps[index].to_dict() != interrupted.step.to_dict():
            raise SkillManagerError(
                "candidate handoff must preserve the interrupted SEARCH step"
            )
        inspect_index = index + 1
        if inspect_index >= len(owned.steps):
            raise SkillManagerError(
                "candidate handoff requires INSPECT immediately after SEARCH"
            )
        inspect_step = owned.steps[inspect_index]
        if inspect_step.skill is not SkillName.INSPECT:
            raise SkillManagerError(
                "candidate handoff requires INSPECT immediately after SEARCH"
            )
        if inspect_step.params.get("candidate_id") != normalized_candidate_id:
            raise SkillManagerError(
                "INSPECT candidate_id does not match the trusted candidate"
            )
        if not isinstance(interrupted.resolved_goal, SearchGoal):
            raise SkillManagerError(
                "interrupted SEARCH does not own a validated SearchGoal"
            )

        self._pending_replacement_plan = owned
        self._pending_search_candidate_handoff = _PendingSearchCandidateHandoff(
            candidate_id=normalized_candidate_id,
            source=normalized_source,
        )
        self._supervisory_continuation = (
            _SupervisoryContinuation.SEARCH_CANDIDATE_HANDOFF
        )
        self._record_transition(
            old_skill=self._active_name,
            old_status=self.active_status,
            result_code=None,
            new_skill=self._active_name,
            reason="search_candidate_handoff_accepted",
            old_step_id=interrupted.step.step_id,
            new_step_id=inspect_step.step_id,
        )
        if self._active_name is SkillName.HOVER:
            return self.release_supervisory_hover()
        if not self._supervisory_waiting or self._active_name is not None:
            raise SkillManagerError(
                "supervisory HOVER has not reached a replaceable state"
            )
        self._start_search_candidate_handoff_after_interruption(
            old_status=SkillStatus.SUCCEEDED,
            result_code=SkillResultCode.HOVER_COMPLETE,
        )
        return self._task_status

    def reset_task(self) -> None:
        if self._task_status not in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELED,
        }:
            raise SkillManagerError("only a terminal task can be reset")
        if self._active_name is not None:
            raise SkillManagerError("terminal task unexpectedly still owns a Skill")
        self._clear_task_to_idle()

    def _start_registered(
        self,
        name: SkillName,
        goal: SkillGoal,
        *,
        invocation: SkillInvocation | None = None,
    ) -> SkillStatus:
        if self._active_name is not None:
            raise SkillManagerError("reset the active Skill before starting another")
        try:
            skill = self._skills[name]
        except KeyError as exc:
            raise SkillNotRegisteredError(f"Skill {name.value} is not registered") from exc
        if skill.status is not SkillStatus.IDLE:
            raise SkillManagerError(f"registered Skill {name.value} is not IDLE")
        if invocation is None:
            invocation = self._make_invocation(name, goal)
        elif invocation.uav_id != self._uav_id:
            raise SkillManagerError(
                "SkillInvocation.uav_id does not match this SkillManager"
            )
        if invocation.skill_name is not name:
            raise SkillManagerError(
                "SkillInvocation.skill_name does not match requested Skill"
            )
        self._active_name = name
        self._active_invocation = invocation
        try:
            skill.start(goal, self._context)
        except BaseException:
            if skill.status is SkillStatus.IDLE:
                self._active_name = None
                self._active_invocation = None
            raise
        return skill.status

    def _make_invocation(
        self,
        name: SkillName,
        goal: SkillGoal,
    ) -> SkillInvocation:
        self._invocation_counter += 1
        task_plan = self._task_plan
        mission_id = (
            "mission_manual" if task_plan is None else task_plan.mission_id
        )
        plan_version = 1 if task_plan is None else task_plan.plan_version
        step_id = self._active_planned_step_id or f"manual_{name.value.lower()}"
        return SkillInvocation(
            mission_id=mission_id,
            uav_id=self._uav_id,
            plan_version=plan_version,
            step_id=step_id,
            invocation_id=f"invocation_{self._invocation_counter:08d}",
            skill_name=name,
            goal=goal,
        )

    def _tick_task(self, observation: Observation) -> TaskStatus:
        if self._active_name is None:
            if self._supervisory_waiting and self._interrupted_execution is not None:
                return self._task_status
            self._fail_without_landing("task has no active Skill")
            return self._task_status
        skill = self._active_skill()
        # No successor is ticked after the active Skill terminates below.
        if skill.status is SkillStatus.RUNNING:
            skill.tick(observation)
        if skill.status is SkillStatus.RUNNING:
            return self._task_status

        old_name = self._active_name
        old_status = skill.status
        old_step_id = self._active_planned_step_id
        old_kind = self._active_execution_kind
        result = skill.get_result()
        if result is None:
            self._fail_without_landing("terminal Skill has no result")
            return self._task_status
        try:
            self._reset_active_internal()
        except Exception as exc:
            failure = _internal_result(
                f"could not reset {old_name.value}: {exc}",
                {"reset_error": str(exc), "previous_result": result.to_dict()},
            )
            self._last_result = failure
            self._task_failure_result = failure
            self._pending_task_result = TaskStatus.FAILED
            if old_name is SkillName.LAND:
                self._finish_task(
                    TaskStatus.FAILED,
                    old_name,
                    old_status,
                    SkillResultCode.INTERNAL_ERROR,
                    "landing_reset_failed",
                    old_step_id=old_step_id,
                )
            else:
                self._begin_landing(
                    old_name,
                    old_status,
                    failure,
                    "skill_reset_failed",
                    old_step_id=old_step_id,
                )
            return self._task_status

        self._transition_from_result(
            old_name, old_status, result, old_step_id, old_kind
        )
        return self._task_status

    def _transition_from_result(
        self,
        old_name: SkillName,
        old_status: SkillStatus,
        result: SkillResult,
        old_step_id: str | None,
        old_kind: ExecutionKind | None,
    ) -> None:
        if old_name is SkillName.LAND:
            if (
                old_status is SkillStatus.SUCCEEDED
                and result.code is SkillResultCode.LAND_COMPLETE
            ):
                if old_step_id is not None:
                    self._step_outputs[old_step_id] = deepcopy(result.data)
                final = (
                    self._pending_task_result
                    if self._pending_task_result
                    in {TaskStatus.FAILED, TaskStatus.CANCELED}
                    else self._land_completion_status(old_step_id)
                )
                reason = {
                    TaskStatus.SUCCEEDED: "task_completed",
                    TaskStatus.FAILED: "failure_landing_complete",
                    TaskStatus.CANCELED: "cancel_landing_complete",
                }[final]
                self._finish_task(
                    final,
                    old_name,
                    old_status,
                    result.code,
                    reason,
                    old_step_id=old_step_id,
                )
            else:
                self._task_failure_result = self._task_failure_result or result
                if old_kind is ExecutionKind.PLANNED:
                    self._pending_task_result = TaskStatus.FAILED
                    self._start_emergency_landing(
                        old_name,
                        old_status,
                        result,
                        "planned_landing_failed",
                        old_step_id=old_step_id,
                    )
                else:
                    # An emergency LAND is the single terminal fallback.  It
                    # must never recurse into another landing attempt.
                    self._finish_task(
                        TaskStatus.FAILED,
                        old_name,
                        old_status,
                        result.code,
                        "emergency_landing_failed",
                        old_step_id=old_step_id,
                    )
            return

        if old_name is SkillName.HOVER and old_kind is ExecutionKind.SUPERVISORY:
            self._transition_from_supervisory_hover(
                old_status,
                result,
                old_step_id,
            )
            return

        if (
            old_name is SkillName.INSPECT
            and old_kind is ExecutionKind.PLANNED
            and self._search_inspection_detour is not None
        ):
            if (
                old_status is SkillStatus.SUCCEEDED
                and result.code is SkillResultCode.GOAL_REACHED
            ):
                if old_step_id != self._search_inspection_detour.inspect_step_id:
                    self._fail_inspection_detour(
                        old_status,
                        old_step_id,
                        "inspection_detour_step_mismatch",
                    )
                    return
                self._step_outputs[old_step_id] = deepcopy(result.data)
                self._resume_search_after_inspection(
                    old_status,
                    result,
                    reason="inspection_evidence_collected_search_resumed",
                )
                return
            if (
                old_status is SkillStatus.FAILED
                and result.code
                in {SkillResultCode.TIMEOUT, SkillResultCode.INVALID_STATE}
            ):
                self._resume_search_after_inspection(
                    old_status,
                    result,
                    reason="inspection_rejected_search_resumed",
                )
                return
            # INVALID_GOAL/INTERNAL_ERROR and cancellation are trusted-boundary
            # failures and follow the normal fail-safe LAND path below.

        if old_status is SkillStatus.CANCELED:
            self._pending_task_result = TaskStatus.CANCELED
            self._begin_landing(
                old_name,
                old_status,
                result,
                "skill_canceled",
                old_step_id=old_step_id,
            )
            return

        if old_status is SkillStatus.FAILED:
            if (
                old_kind is ExecutionKind.PLANNED
                and old_name is SkillName.TRACK
                and result.code is SkillResultCode.TARGET_LOST
                and self._try_start_recovery(result, old_status, old_step_id)
            ):
                return
            self._task_failure_result = result
            self._pending_task_result = TaskStatus.FAILED
            recovery_attempt = (
                self._recovery_attempts.get(old_step_id)
                if old_name in {SkillName.TRACK, SkillName.REACQUIRE}
                and old_step_id is not None
                else None
            )
            self._begin_landing(
                old_name,
                old_status,
                result,
                _failure_reason(old_name, result.code),
                old_step_id=old_step_id,
                recovery_attempt=recovery_attempt or None,
            )
            return

        if old_status is not SkillStatus.SUCCEEDED:
            self._task_failure_result = result
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                old_name,
                old_status,
                result,
                "invalid_skill_status",
                old_step_id=old_step_id,
            )
            return

        if result.code is not _EXPECTED_SUCCESS_CODES.get(old_name):
            self._task_failure_result = result
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                old_name,
                old_status,
                result,
                "unexpected_result_code",
                old_step_id=old_step_id,
            )
            return

        if old_kind is ExecutionKind.RECOVERY:
            self._resume_track_after_recovery(result, old_status, old_step_id)
            return

        if old_step_id is None:
            self._fail_without_landing("planned Skill is missing its step id")
            return
        self._step_outputs[old_step_id] = deepcopy(result.data)
        if old_name is SkillName.SEARCH:
            target_id = result.data.get("target_id")
            if not isinstance(target_id, str) or not target_id.strip():
                self._step_outputs.pop(old_step_id, None)
                self._task_failure_result = result
                self._pending_task_result = TaskStatus.FAILED
                self._begin_landing(
                    old_name,
                    old_status,
                    result,
                    "search_result_missing_target_id",
                    old_step_id=old_step_id,
                )
                return
            self._active_target_id = target_id.strip()
        if old_name is SkillName.TRACK:
            # Compatibility-visible provisional result.  A later step failure
            # or cancel replaces it before LAND commits the terminal status.
            self._pending_task_result = TaskStatus.SUCCEEDED
        self._start_next_planned(old_name, old_status, result, old_step_id)

    def _transition_from_supervisory_hover(
        self,
        old_status: SkillStatus,
        result: SkillResult,
        old_step_id: str | None,
    ) -> None:
        interrupted = self._interrupted_execution
        if interrupted is None:
            self._task_failure_result = _internal_result(
                "supervisory HOVER completed without interrupted state"
            )
            self._last_result = self._task_failure_result
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                SkillName.HOVER,
                old_status,
                self._task_failure_result,
                "supervisory_state_missing",
                old_step_id=old_step_id,
            )
            return

        if (
            old_status is SkillStatus.SUCCEEDED
            and result.code is SkillResultCode.HOVER_COMPLETE
        ):
            if self._supervisory_continuation is _SupervisoryContinuation.RESUME:
                self._resume_saved_interruption(
                    old_status=old_status,
                    result_code=result.code,
                    reason="interrupted_step_resumed",
                )
                return
            if self._supervisory_continuation is _SupervisoryContinuation.REPLACE:
                self._start_replacement_after_interruption(
                    old_status=old_status,
                    result_code=result.code,
                )
                return
            if (
                self._supervisory_continuation
                is _SupervisoryContinuation.SEARCH_CANDIDATE_HANDOFF
            ):
                self._start_search_candidate_handoff_after_interruption(
                    old_status=old_status,
                    result_code=result.code,
                )
                return
            self._supervisory_waiting = True
            self._record_transition(
                old_skill=SkillName.HOVER,
                old_status=old_status,
                result_code=result.code,
                new_skill=None,
                reason="supervisory_hover_released",
                old_step_id=old_step_id,
                new_step_id=old_step_id,
            )
            return

        if (
            old_status is SkillStatus.FAILED
            and result.code is SkillResultCode.TIMEOUT
            and interrupted.timeout_fallback
            is HoverTimeoutFallback.RESUME_PREVIOUS
        ):
            self._resume_saved_interruption(
                old_status=old_status,
                result_code=result.code,
                reason="supervisory_hover_timeout_resume_previous",
            )
            return

        self._task_failure_result = result
        self._pending_task_result = TaskStatus.FAILED
        reason = (
            "supervisory_hover_timeout_cancel_and_land"
            if result.code is SkillResultCode.TIMEOUT
            else "supervisory_hover_failed"
        )
        self._begin_landing(
            SkillName.HOVER,
            old_status,
            result,
            reason,
            old_step_id=old_step_id,
        )

    def _require_supervisory_interruption(self) -> _InterruptedExecution:
        if self._task_status is not TaskStatus.RUNNING:
            raise SkillManagerError("there is no RUNNING supervisory interruption")
        interrupted = self._interrupted_execution
        if interrupted is None:
            raise SkillManagerError("there is no interrupted step")
        return interrupted

    def _resume_saved_interruption(
        self,
        *,
        old_status: SkillStatus,
        result_code: SkillResultCode,
        reason: str,
    ) -> None:
        interrupted = self._require_supervisory_interruption()
        current_plan = self._task_plan
        if (
            current_plan is None
            or current_plan.mission_id != interrupted.plan.mission_id
            or current_plan.uav_id != interrupted.plan.uav_id
            or current_plan.plan_version != interrupted.plan.plan_version
        ):
            raise SkillManagerError(
                "task routing/version changed during supervisory HOVER"
            )
        goal = deepcopy(interrupted.resolved_goal)
        if interrupted.step.skill is SkillName.TRACK:
            if not isinstance(goal, TrackGoal):
                raise SkillManagerError("saved TRACK Goal has invalid type")
            if (
                interrupted.active_target_id is not None
                and goal.target_id != interrupted.active_target_id
            ):
                raise SkillManagerError(
                    "saved TRACK target_id changed during supervisory HOVER"
                )

        self._task_plan = _copy_task_plan(interrupted.plan)
        self._plan_index = interrupted.plan_index
        self._step_outputs = deepcopy(interrupted.step_outputs)
        self._active_target_id = interrupted.active_target_id
        self._recovery_attempts = dict(interrupted.recovery_attempts)
        self._saved_track_goal_by_step = deepcopy(interrupted.saved_track_goals)
        step = interrupted.step
        self._discard_supervisory_state()
        self._start_transition(
            SkillName.HOVER,
            old_status,
            result_code,
            step.skill,
            goal,
            reason,
            old_step_id=step.step_id,
            new_step_id=step.step_id,
            recovery_attempt=(
                self._recovery_attempts.get(step.step_id)
                if step.skill is SkillName.TRACK
                else None
            ),
            execution_kind=ExecutionKind.PLANNED,
        )

    def _validate_replacement_plan(
        self,
        plan: TaskPlan,
        interrupted: _InterruptedExecution,
    ) -> TaskPlan:
        if not isinstance(plan, TaskPlan):
            raise TypeError("plan must be a TaskPlan")
        owned = _copy_task_plan(plan)
        original = interrupted.plan
        if owned.uav_id != self._uav_id or owned.uav_id != original.uav_id:
            raise SkillManagerError("replacement TaskPlan.uav_id mismatch")
        if owned.mission_id != original.mission_id:
            raise SkillManagerError("replacement TaskPlan.mission_id mismatch")
        if owned.plan_version != original.plan_version + 1:
            raise SkillManagerError(
                "replacement plan_version must advance by exactly one"
            )
        index = interrupted.plan_index
        if index >= len(owned.steps):
            raise SkillManagerError("replacement removed the interrupted step slot")
        if [step.to_dict() for step in owned.steps[:index]] != [
            step.to_dict() for step in original.steps[:index]
        ]:
            raise SkillManagerError("replacement modified the completed plan prefix")

        required = {step.skill for step in owned.steps[index:]} | {SkillName.LAND}
        if any(
            self._effective_recovery_policy(step) is not None
            for step in owned.steps[index:]
        ):
            required.add(SkillName.REACQUIRE)
        missing = sorted(name.value for name in required if name not in self._skills)
        if missing:
            raise SkillNotRegisteredError(
                "replacement registry is missing: " + ", ".join(missing)
            )
        for candidate_index, step in enumerate(owned.steps[index:], start=index):
            _reject_unknown_goal_fields(step.skill, step.params)
            candidate_goal = self._goal_from_step(
                step,
                plan=owned,
                validation_only=candidate_index != index,
            )
            if (
                candidate_index == index
                and step.skill is SkillName.TRACK
                and interrupted.active_target_id is not None
            ):
                if (
                    not isinstance(candidate_goal, TrackGoal)
                    or candidate_goal.target_id != interrupted.active_target_id
                ):
                    raise SkillManagerError(
                        "replacement TRACK cannot change the active target identity"
                    )
        return owned

    def _start_replacement_after_interruption(
        self,
        *,
        old_status: SkillStatus,
        result_code: SkillResultCode,
    ) -> None:
        interrupted = self._require_supervisory_interruption()
        replacement = self._pending_replacement_plan
        if replacement is None:
            raise SkillManagerError("replacement continuation has no TaskPlan")
        index = interrupted.plan_index
        step = replacement.steps[index]

        self._task_plan = _copy_task_plan(replacement)
        self._plan_index = index
        self._step_outputs = deepcopy(interrupted.step_outputs)
        self._active_target_id = interrupted.active_target_id
        self._recovery_attempts = {
            candidate.step_id: interrupted.recovery_attempts.get(
                candidate.step_id, 0
            )
            for candidate in replacement.steps
            if self._effective_recovery_policy(candidate) is not None
        }
        self._saved_track_goal_by_step = {
            step_id: deepcopy(goal)
            for step_id, goal in interrupted.saved_track_goals.items()
            if any(candidate.step_id == step_id for candidate in replacement.steps)
        }
        try:
            goal = self._goal_from_step(step)
        except Exception as exc:
            # Validation above makes this an internal state/routing failure,
            # never a partially accepted revision that continues unchecked.
            failure = _internal_result(
                f"could not resolve replacement step {step.step_id}: {exc}",
                {"step_id": step.step_id, "error": str(exc)},
            )
            self._last_result = failure
            self._task_failure_result = failure
            self._pending_task_result = TaskStatus.FAILED
            self._discard_supervisory_state()
            self._begin_landing(
                SkillName.HOVER,
                old_status,
                failure,
                "replacement_goal_invalid",
                old_step_id=interrupted.step.step_id,
            )
            return
        if step.skill is SkillName.TRACK:
            if not isinstance(goal, TrackGoal):
                raise SkillManagerError("replacement TRACK compiled to invalid Goal")
            if (
                interrupted.active_target_id is not None
                and goal.target_id != interrupted.active_target_id
            ):
                raise SkillManagerError(
                    "replacement TRACK cannot change the active target identity"
                )
            self._saved_track_goal_by_step[step.step_id] = goal

        old_step_id = interrupted.step.step_id
        self._discard_supervisory_state()
        self._start_transition(
            SkillName.HOVER,
            old_status,
            result_code,
            step.skill,
            goal,
            "interrupted_step_and_suffix_replaced",
            old_step_id=old_step_id,
            new_step_id=step.step_id,
            recovery_attempt=(
                self._recovery_attempts.get(step.step_id)
                if step.skill is SkillName.TRACK
                else None
            ),
            execution_kind=ExecutionKind.PLANNED,
        )

    def _start_search_candidate_handoff_after_interruption(
        self,
        *,
        old_status: SkillStatus,
        result_code: SkillResultCode,
    ) -> None:
        interrupted = self._require_supervisory_interruption()
        replacement = self._pending_replacement_plan
        handoff = self._pending_search_candidate_handoff
        if replacement is None or handoff is None:
            raise SkillManagerError(
                "candidate handoff continuation is missing trusted state"
            )
        index = interrupted.plan_index
        inspect_index = index + 1
        # The public API already validated these invariants.  Re-check before
        # publication because this is the exact state-changing boundary.
        if (
            interrupted.step.skill is not SkillName.SEARCH
            or inspect_index >= len(replacement.steps)
            or replacement.steps[index].to_dict() != interrupted.step.to_dict()
            or replacement.steps[inspect_index].skill is not SkillName.INSPECT
            or replacement.steps[inspect_index].params.get("candidate_id")
            != handoff.candidate_id
        ):
            raise SkillManagerError("candidate handoff state changed before release")

        inspect_step = replacement.steps[inspect_index]
        self._task_plan = _copy_task_plan(replacement)
        self._plan_index = inspect_index
        self._step_outputs = deepcopy(interrupted.step_outputs)
        self._active_target_id = interrupted.active_target_id
        self._recovery_attempts = {
            candidate.step_id: interrupted.recovery_attempts.get(
                candidate.step_id,
                0,
            )
            for candidate in replacement.steps
            if self._effective_recovery_policy(candidate) is not None
        }
        self._saved_track_goal_by_step = {
            step_id: deepcopy(goal)
            for step_id, goal in interrupted.saved_track_goals.items()
            if any(candidate.step_id == step_id for candidate in replacement.steps)
        }
        try:
            goal = self._goal_from_step(inspect_step)
        except Exception as exc:
            failure = _internal_result(
                f"could not resolve handoff INSPECT {inspect_step.step_id}: {exc}",
                {"step_id": inspect_step.step_id, "error": str(exc)},
            )
            self._last_result = failure
            self._task_failure_result = failure
            self._pending_task_result = TaskStatus.FAILED
            self._discard_supervisory_state()
            self._begin_landing(
                SkillName.HOVER,
                old_status,
                failure,
                "candidate_handoff_goal_invalid",
                old_step_id=interrupted.step.step_id,
            )
            return

        old_step_id = interrupted.step.step_id
        search_goal = interrupted.resolved_goal
        if not isinstance(search_goal, SearchGoal):
            raise SkillManagerError(
                "candidate handoff saved an invalid SEARCH goal"
            )
        self._search_inspection_detour = _SearchInspectionDetour(
            search_index=index,
            inspect_index=inspect_index,
            search_step=TaskStep(
                interrupted.step.step_id,
                interrupted.step.skill,
                interrupted.step.params,
                interrupted.step.recovery,
            ),
            search_goal=deepcopy(search_goal),
            inspect_step_id=inspect_step.step_id,
            candidate_id=handoff.candidate_id,
        )
        self._discard_supervisory_state()
        self._start_transition(
            SkillName.HOVER,
            old_status,
            result_code,
            SkillName.INSPECT,
            goal,
            "search_candidate_handoff_to_inspect",
            old_step_id=old_step_id,
            new_step_id=inspect_step.step_id,
            execution_kind=ExecutionKind.PLANNED,
        )

    def _discard_supervisory_state(self) -> None:
        self._interrupted_execution = None
        self._supervisory_continuation = None
        self._pending_replacement_plan = None
        self._pending_search_candidate_handoff = None
        self._supervisory_waiting = False

    def _try_start_recovery(
        self,
        result: SkillResult,
        old_status: SkillStatus,
        step_id: str | None,
    ) -> bool:
        step = self._step_by_id(step_id)
        if step is None or step.skill is not SkillName.TRACK:
            return False
        policy = self._effective_recovery_policy(step)
        attempts = self._recovery_attempts.get(step.step_id, 0)
        if policy is None or attempts >= policy.max_attempts:
            return False
        target_id, identity_error = self._validated_lost_target_id(
            result, step.step_id
        )
        if identity_error is not None:
            failure = _internal_result(
                identity_error,
                {
                    "step_id": step.step_id,
                    "active_target_id": self._active_target_id,
                    "reported_target_id": result.data.get("target_id"),
                },
            )
            self._last_result = failure
            self._task_failure_result = failure
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                SkillName.TRACK,
                old_status,
                failure,
                "track_lost_target_mismatch",
                old_step_id=step.step_id,
                recovery_attempt=attempts or None,
            )
            return True
        goal = self._reacquire_goal_from_result(
            result, step.step_id, policy, target_id
        )
        if goal is None:
            return False
        attempt = attempts + 1
        self._recovery_attempts[step.step_id] = attempt
        self._start_transition(
            SkillName.TRACK,
            old_status,
            result.code,
            SkillName.REACQUIRE,
            goal,
            "target_lost",
            old_step_id=step.step_id,
            new_step_id=step.step_id,
            recovery_attempt=attempt,
            execution_kind=ExecutionKind.RECOVERY,
        )
        # _start_transition also owns a start failure and its fail-safe LAND.
        return True

    def _resume_track_after_recovery(
        self,
        result: SkillResult,
        old_status: SkillStatus,
        step_id: str | None,
    ) -> None:
        if step_id is None:
            self._fail_without_landing("recovery is missing its TRACK step id")
            return
        saved = self._saved_track_goal_by_step.get(step_id)
        returned_target_id = result.data.get("target_id")
        if (
            not isinstance(returned_target_id, str)
            or not returned_target_id.strip()
        ):
            self._fail_recovery_resume(
                old_status,
                step_id,
                "REACQUIRE success result must explicitly contain a non-empty target_id",
                "reacquire_target_invalid",
            )
            return
        if (
            saved is None
            or self._active_target_id is None
            or not self._active_target_id.strip()
            or not saved.target_id.strip()
        ):
            self._fail_recovery_resume(
                old_status,
                step_id,
                "REACQUIRE cannot resume because trusted TRACK identity state is missing",
                "recovery_state_missing",
            )
            return
        normalized_target_id = returned_target_id.strip()
        active_target_id = self._active_target_id.strip()
        saved_target_id = saved.target_id.strip()
        if not (
            normalized_target_id == active_target_id == saved_target_id
        ):
            self._fail_recovery_resume(
                old_status,
                step_id,
                "REACQUIRE returned a target_id inconsistent with trusted TRACK identity",
                "reacquire_target_mismatch",
                {
                    "active_target_id": active_target_id,
                    "saved_track_target_id": saved_target_id,
                    "returned_target_id": normalized_target_id,
                },
            )
            return
        goal = replace(saved, target_id=normalized_target_id)
        self._saved_track_goal_by_step[step_id] = goal
        self._start_transition(
            SkillName.REACQUIRE,
            old_status,
            result.code,
            SkillName.TRACK,
            goal,
            "target_reacquired",
            old_step_id=step_id,
            new_step_id=step_id,
            recovery_attempt=self._recovery_attempts.get(step_id),
            execution_kind=ExecutionKind.PLANNED,
        )

    def _fail_recovery_resume(
        self,
        old_status: SkillStatus,
        step_id: str,
        message: str,
        reason: str,
        data: dict[str, object] | None = None,
    ) -> None:
        failure = _internal_result(message, data)
        self._last_result = failure
        self._task_failure_result = failure
        self._pending_task_result = TaskStatus.FAILED
        self._begin_landing(
            SkillName.REACQUIRE,
            old_status,
            failure,
            reason,
            old_step_id=step_id,
            recovery_attempt=self._recovery_attempts.get(step_id),
        )

    def _start_next_planned(
        self,
        old_name: SkillName,
        old_status: SkillStatus,
        result: SkillResult,
        old_step_id: str,
    ) -> None:
        if self._task_plan is None or self._plan_index is None:
            self._fail_without_landing("task plan state is missing")
            return
        next_index = self._plan_index + 1
        transition_reason = _normal_transition_reason(old_name)
        detour = self._search_inspection_detour
        if (
            detour is not None
            and old_name is SkillName.SEARCH
            and old_step_id == detour.search_step.step_id
            and self._plan_index == detour.search_index
        ):
            if next_index != detour.inspect_index:
                self._fail_without_landing(
                    "inspection detour no longer matches the authoritative plan"
                )
                return
            next_index += 1
            transition_reason = "target_found_after_inspection_detour"
            self._search_inspection_detour = None
        if next_index >= len(self._task_plan.steps):
            failure = _internal_result(
                "validated plan ended without a terminal LAND",
                {"last_step_id": old_step_id},
            )
            self._last_result = failure
            self._task_failure_result = failure
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                old_name,
                old_status,
                failure,
                "plan_ended_without_land",
                old_step_id=old_step_id,
            )
            return
        step = self._task_plan.steps[next_index]
        try:
            goal = self._goal_from_step(step)
        except Exception as exc:
            failure = _internal_result(
                f"could not resolve step {step.step_id}: {exc}",
                {"step_id": step.step_id, "error": str(exc)},
            )
            self._last_result = failure
            self._task_failure_result = failure
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                old_name,
                old_status,
                failure,
                f"next_goal_invalid:{exc}",
                old_step_id=old_step_id,
            )
            return

        self._plan_index = next_index
        if step.skill is SkillName.TRACK:
            if not isinstance(goal, TrackGoal):
                self._fail_without_landing("TRACK plan did not compile to TrackGoal")
                return
            self._saved_track_goal_by_step[step.step_id] = goal
        self._start_transition(
            old_name,
            old_status,
            result.code,
            step.skill,
            goal,
            transition_reason,
            old_step_id=old_step_id,
            new_step_id=step.step_id,
            execution_kind=ExecutionKind.PLANNED,
        )

    def _resume_search_after_inspection(
        self,
        old_status: SkillStatus,
        result: SkillResult,
        *,
        reason: str,
    ) -> None:
        """Return a completed/rejected INSPECT detour to the exact SEARCH."""

        detour = self._search_inspection_detour
        plan = self._task_plan
        if detour is None or plan is None:
            self._fail_inspection_detour(
                old_status,
                self._active_planned_step_id,
                "inspection_detour_state_missing",
            )
            return
        if (
            detour.search_index >= len(plan.steps)
            or detour.inspect_index >= len(plan.steps)
            or plan.steps[detour.search_index].to_dict()
            != detour.search_step.to_dict()
            or plan.steps[detour.inspect_index].step_id != detour.inspect_step_id
            or plan.steps[detour.inspect_index].skill is not SkillName.INSPECT
        ):
            self._fail_inspection_detour(
                old_status,
                detour.inspect_step_id,
                "inspection_detour_plan_mismatch",
            )
            return
        self._plan_index = detour.search_index
        self._start_transition(
            SkillName.INSPECT,
            old_status,
            result.code,
            SkillName.SEARCH,
            deepcopy(detour.search_goal),
            reason,
            old_step_id=detour.inspect_step_id,
            new_step_id=detour.search_step.step_id,
            execution_kind=ExecutionKind.PLANNED,
        )

    def _fail_inspection_detour(
        self,
        old_status: SkillStatus,
        old_step_id: str | None,
        reason: str,
    ) -> None:
        failure = _internal_result(
            "INSPECT detour could not return to its interrupted SEARCH",
            {"reason": reason},
        )
        self._last_result = failure
        self._task_failure_result = failure
        self._pending_task_result = TaskStatus.FAILED
        self._search_inspection_detour = None
        self._begin_landing(
            SkillName.INSPECT,
            old_status,
            failure,
            reason,
            old_step_id=old_step_id,
        )

    def _start_transition(
        self,
        old_name: SkillName | None,
        old_status: SkillStatus | None,
        result_code: SkillResultCode | None,
        new_name: SkillName,
        goal: SkillGoal,
        reason: str,
        *,
        old_step_id: str | None,
        new_step_id: str | None,
        recovery_attempt: int | None = None,
        execution_kind: ExecutionKind = ExecutionKind.PLANNED,
    ) -> None:
        self._active_planned_step_id = new_step_id
        self._active_execution_kind = execution_kind
        try:
            self._start_registered(new_name, goal)
        except Exception as exc:
            self._context.uav.stop()
            failure = _internal_result(
                f"could not start {new_name.value}: {exc}",
                {"step_id": new_step_id, "error": str(exc)},
            )
            self._last_result = failure
            self._task_failure_result = failure
            if new_name is SkillName.LAND:
                self._finish_task(
                    TaskStatus.FAILED,
                    old_name,
                    old_status,
                    result_code,
                    f"landing_start_failed:{exc}",
                    old_step_id=old_step_id,
                )
                return
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                old_name,
                old_status,
                failure,
                f"skill_start_failed:{new_name.value}:{exc}",
                old_step_id=old_step_id,
                recovery_attempt=(
                    recovery_attempt
                    if execution_kind is ExecutionKind.RECOVERY
                    else None
                ),
            )
            return
        self._record_transition(
            old_skill=old_name,
            old_status=old_status,
            result_code=result_code,
            new_skill=new_name,
            reason=reason,
            old_step_id=old_step_id,
            new_step_id=new_step_id,
            recovery_attempt=recovery_attempt,
        )

    def _begin_landing(
        self,
        old_name: SkillName | None,
        old_status: SkillStatus | None,
        result: SkillResult | None,
        reason: str,
        *,
        old_step_id: str | None,
        recovery_attempt: int | None = None,
    ) -> None:
        self._discard_supervisory_state()
        self._search_inspection_detour = None
        if old_name is SkillName.LAND:
            self._finish_task(
                TaskStatus.FAILED,
                old_name,
                old_status,
                None if result is None else result.code,
                "landing_failed",
                old_step_id=old_step_id,
            )
            return
        if self._pending_task_result is None:
            self._pending_task_result = TaskStatus.FAILED
        if SkillName.LAND not in self._skills:
            self._finish_task(
                TaskStatus.FAILED,
                old_name,
                old_status,
                None if result is None else result.code,
                "landing_not_registered",
                old_step_id=old_step_id,
            )
            return

        self._start_emergency_landing(
            old_name,
            old_status,
            result,
            reason,
            old_step_id=old_step_id,
            recovery_attempt=recovery_attempt,
        )

    def _start_emergency_landing(
        self,
        old_name: SkillName | None,
        old_status: SkillStatus | None,
        result: SkillResult | None,
        reason: str,
        *,
        old_step_id: str | None,
        recovery_attempt: int | None = None,
    ) -> None:
        land_step = self._planned_land_step()
        new_step_id = None if land_step is None else land_step.step_id
        land_goal = self._emergency_land_goal(land_step)
        if self._task_plan is not None and land_step is not None:
            self._plan_index = self._task_plan.steps.index(land_step)
        self._start_transition(
            old_name,
            old_status,
            None if result is None else result.code,
            SkillName.LAND,
            land_goal,
            reason,
            old_step_id=old_step_id,
            new_step_id=new_step_id,
            execution_kind=ExecutionKind.EMERGENCY,
            recovery_attempt=recovery_attempt,
        )

    def _reacquire_goal_from_result(
        self,
        result: SkillResult,
        step_id: str,
        policy: RecoveryPolicy,
        target_id: str | None,
    ) -> ReacquireGoal | None:
        position = _finite_vector3_or_none(result.data.get("last_seen_position"))
        velocity = _finite_vector3_or_none(result.data.get("last_seen_velocity"))
        last_seen_time = _nonnegative_number_or_none(result.data.get("last_seen_time"))
        if (
            not isinstance(target_id, str)
            or not target_id.strip()
            or position is None
            or velocity is None
            or last_seen_time is None
        ):
            return None
        self._active_target_id = target_id.strip()
        self._reduce_remaining_track_duration(step_id, result)
        return ReacquireGoal(
            target_id=self._active_target_id,
            last_seen_position=position,
            last_seen_velocity=velocity,
            last_seen_time=last_seen_time,
            search_radius=policy.search_radius_m,
            timeout=policy.timeout_s,
        )

    def _validated_lost_target_id(
        self,
        result: SkillResult,
        step_id: str,
    ) -> tuple[str | None, str | None]:
        """Validate TRACK loss identity before changing any recovery state."""

        saved = self._saved_track_goal_by_step.get(step_id)
        saved_target_id = None if saved is None else saved.target_id.strip()
        active_target_id = (
            None
            if self._active_target_id is None
            else self._active_target_id.strip()
        )
        if (
            active_target_id is not None
            and saved_target_id is not None
            and active_target_id != saved_target_id
        ):
            return None, "TRACK recovery identity state is inconsistent"

        reported = result.data.get("target_id")
        if reported is None:
            resolved = active_target_id or saved_target_id
            if resolved is None:
                return None, "TRACK TARGET_LOST has no trusted target identity"
            return resolved, None
        if not isinstance(reported, str) or not reported.strip():
            return None, "TRACK TARGET_LOST returned an invalid target_id"
        normalized = reported.strip()
        expected = active_target_id or saved_target_id
        if expected is not None and normalized != expected:
            return None, "TRACK TARGET_LOST returned a different target_id"
        return normalized, None

    def _reduce_remaining_track_duration(
        self, step_id: str, result: SkillResult
    ) -> None:
        goal = self._saved_track_goal_by_step.get(step_id)
        if goal is None or goal.track_duration is None:
            return
        elapsed = _nonnegative_number_or_none(result.data.get("tracking_duration"))
        if elapsed is None:
            return
        self._saved_track_goal_by_step[step_id] = replace(
            goal,
            track_duration=max(1e-9, float(goal.track_duration) - elapsed),
        )

    def _effective_recovery_policy(self, step: TaskStep) -> RecoveryPolicy | None:
        if step.recovery is not None:
            return step.recovery if step.recovery.max_attempts > 0 else None
        # Compatibility boundary for the fixed MissionIntent compiler.  Its
        # historical placeholder implied bounded Manager recovery.  A dynamic
        # StepOutputRef with no policy intentionally disables recovery.
        if (
            step.skill is SkillName.TRACK
            and step.params.get("target_id") == "$SEARCH.result.target_id"
        ):
            return RecoveryPolicy(
                skill=SkillName.REACQUIRE,
                max_attempts=2,
                search_radius_m=self._reacquire_search_radius,
                timeout_s=self._reacquire_timeout,
            )
        return None

    def _planned_land_step(self) -> TaskStep | None:
        if self._task_plan is None:
            return None
        return next(
            (
                step
                for step in reversed(self._task_plan.steps)
                if step.skill is SkillName.LAND
            ),
            None,
        )

    def _planned_land_goal(self, step: TaskStep | None) -> LandGoal:
        if step is not None:
            try:
                goal = self._goal_from_step(step)
                if isinstance(goal, LandGoal):
                    return goal
            except Exception:
                pass
        return LandGoal()

    def _emergency_land_goal(self, step: TaskStep | None) -> LandGoal:
        """Copy trusted LAND dynamics but remove named-zone enforcement."""

        planned = self._planned_land_goal(step)
        return replace(planned, expected_position_xy=None)

    def _land_completion_status(self, step_id: str | None) -> TaskStatus:
        if self._task_plan is None or self._plan_index is None or step_id is None:
            return TaskStatus.FAILED
        if (
            self._plan_index == len(self._task_plan.steps) - 1
            and self._task_plan.steps[-1].step_id == step_id
        ):
            return TaskStatus.SUCCEEDED
        return TaskStatus.FAILED

    def _goal_from_step(
        self,
        step: TaskStep,
        *,
        plan: TaskPlan | None = None,
        validation_only: bool = False,
    ) -> SkillGoal:
        params = deepcopy(dict(step.params))
        source_plan = plan or self._task_plan
        if step.skill is SkillName.TRACK:
            params["target_id"] = self._resolve_track_target(
                params.get("target_id", "$SEARCH.result.target_id"),
                step,
                source_plan,
                validation_only=validation_only,
            )
            params.setdefault("track_duration", 30.0)
        if step.skill is SkillName.SEARCH and "search_altitude" not in params:
            params["search_altitude"] = self._default_search_altitude(source_plan)
        if step.skill is SkillName.HOVER:
            if "mode" in params:
                params["mode"] = _hover_mode(params["mode"])
            if params.get("mode", HoverMode.TIMED) is not HoverMode.TIMED:
                raise TaskPlanError(
                    "TaskPlan HOVER must use TIMED mode; UNTIL_RELEASED is trusted-runtime only"
                )
        if "yaw_mode" in params:
            params["yaw_mode"] = _yaw_mode(params["yaw_mode"])
        if step.skill in {SkillName.GOTO, SkillName.HOVER} and "motion_policy" in params:
            params["motion_policy"] = _motion_policy(params["motion_policy"])
        for key in {
            "position",
            "center",
            "look_at_point",
            "last_seen_position",
            "last_seen_velocity",
            "expected_position_xy",
        } & params.keys():
            if isinstance(params[key], list):
                params[key] = tuple(params[key])
        goal_type = _goal_type_for_name(step.skill)
        if goal_type is None:
            raise TaskPlanError(f"{step.skill.value} is not a planned task step")
        try:
            return goal_type(**params)
        except (TypeError, ValueError) as exc:
            raise TaskPlanError(f"invalid {step.skill.value} parameters: {exc}") from exc

    def _resolve_track_target(
        self,
        raw_target: object,
        step: TaskStep,
        plan: TaskPlan | None,
        *,
        validation_only: bool,
    ) -> str:
        reference: StepOutputRef | None = None
        if isinstance(raw_target, StepOutputRef):
            reference = raw_target
        elif raw_target == "$SEARCH.result.target_id":
            reference = self._nearest_prior_search_ref(step, plan)
        elif isinstance(raw_target, str) and raw_target.startswith("$"):
            reference = StepOutputRef.from_string(raw_target)

        if reference is not None:
            self._validate_prior_search_reference(reference, step, plan)
            if validation_only:
                return "__validated_search_target__"
            output = self._step_outputs.get(reference.step_id)
            if output is None:
                raise TaskPlanError(
                    f"TRACK reference step {reference.step_id!r} has no output"
                )
            target_id = output.get(reference.field)
            if not isinstance(target_id, str) or not target_id.strip():
                raise TaskPlanError(
                    f"TRACK reference {reference.to_string()} is not a non-empty target_id"
                )
            return target_id.strip()
        if not isinstance(raw_target, str) or not raw_target.strip():
            raise TaskPlanError("TRACK target_id must be a non-empty string or reference")
        return raw_target.strip()

    def _nearest_prior_search_ref(
        self, step: TaskStep, plan: TaskPlan | None
    ) -> StepOutputRef:
        if plan is None:
            raise TaskPlanError("legacy SEARCH reference requires a task plan")
        index = self._index_of_step(plan, step)
        for candidate in reversed(plan.steps[:index]):
            if candidate.skill is SkillName.SEARCH:
                return StepOutputRef(candidate.step_id)
        raise TaskPlanError("legacy TRACK reference has no prior SEARCH step")

    def _validate_prior_search_reference(
        self,
        reference: StepOutputRef,
        step: TaskStep,
        plan: TaskPlan | None,
    ) -> None:
        if plan is None:
            raise TaskPlanError("TRACK output reference requires a task plan")
        current_index = self._index_of_step(plan, step)
        source_index = next(
            (
                index
                for index, candidate in enumerate(plan.steps)
                if candidate.step_id == reference.step_id
            ),
            None,
        )
        if source_index is None:
            raise TaskPlanError(
                f"TRACK reference step {reference.step_id!r} does not exist"
            )
        if source_index >= current_index:
            raise TaskPlanError("TRACK output reference must point to a prior step")
        if plan.steps[source_index].skill is not SkillName.SEARCH:
            raise TaskPlanError("TRACK target_id reference must point to SEARCH")

    @staticmethod
    def _index_of_step(plan: TaskPlan, step: TaskStep) -> int:
        for index, candidate in enumerate(plan.steps):
            if candidate.step_id == step.step_id:
                return index
        raise TaskPlanError(f"step {step.step_id!r} is not in the task plan")

    def _default_search_altitude(self, plan: TaskPlan | None) -> float:
        if plan is not None:
            for step in plan.steps:
                if step.skill is SkillName.TAKEOFF:
                    value = _positive_number_or_none(
                        step.params.get("target_altitude")
                    )
                    if value is not None:
                        return value
        altitude = float(self._context.uav.get_pose().z)
        if not isfinite(altitude) or altitude <= 0.0:
            raise TaskPlanError(
                "SEARCH search_altitude is required when no positive TAKEOFF altitude exists"
            )
        return altitude

    def _step_by_id(self, step_id: str | None) -> TaskStep | None:
        if self._task_plan is None or step_id is None:
            return None
        return next(
            (step for step in self._task_plan.steps if step.step_id == step_id),
            None,
        )

    def _finish_task(
        self,
        status: TaskStatus,
        old_name: SkillName | None,
        old_status: SkillStatus | None,
        result_code: SkillResultCode | None,
        reason: str,
        *,
        old_step_id: str | None,
    ) -> None:
        self._context.uav.stop()
        self._task_status = status
        self._active_planned_step_id = None
        self._active_execution_kind = None
        self._search_inspection_detour = None
        self._record_transition(
            old_skill=old_name,
            old_status=old_status,
            result_code=result_code,
            new_skill=None,
            reason=reason,
            old_step_id=old_step_id,
            new_step_id=None,
        )

    def _fail_without_landing(self, reason: str) -> None:
        old_name = self._active_name
        old_status = self.active_status
        old_step_id = self._active_planned_step_id
        if old_name is not None:
            skill = self._active_skill()
            try:
                if skill.status is SkillStatus.RUNNING:
                    skill.cancel()
                result = skill.get_result()
                if result is not None:
                    self._reset_active_internal()
                else:
                    try:
                        skill.reset()
                    finally:
                        self._active_name = None
            except Exception:
                self._active_name = None
        failure = _internal_result(reason)
        self._last_result = failure
        self._task_failure_result = failure
        self._pending_task_result = TaskStatus.FAILED
        if old_name is not SkillName.LAND and SkillName.LAND in self._skills:
            self._begin_landing(
                old_name,
                old_status,
                failure,
                reason,
                old_step_id=old_step_id,
            )
            return
        self._finish_task(
            TaskStatus.FAILED,
            old_name,
            old_status,
            failure.code,
            reason,
            old_step_id=old_step_id,
        )

    def _record_transition(
        self,
        *,
        old_skill: SkillName | None,
        old_status: SkillStatus | None,
        result_code: SkillResultCode | None,
        new_skill: SkillName | None,
        reason: str,
        old_step_id: str | None,
        new_step_id: str | None,
        recovery_attempt: int | None = None,
    ) -> None:
        timestamp = self._read_transition_time()
        record = TransitionRecord(
            timestamp=timestamp,
            old_skill=old_skill,
            old_status=old_status,
            result_code=result_code,
            new_skill=new_skill,
            reason=str(reason),
            old_step_id=old_step_id,
            new_step_id=new_step_id,
            recovery_attempt=recovery_attempt,
            uav_id=self._uav_id,
            mission_id=(
                self._task_plan.mission_id
                if self._task_plan is not None
                else "mission_manual"
            ),
            plan_version=(
                self._task_plan.plan_version
                if self._task_plan is not None
                else 1
            ),
            invocation_id=(
                self._active_invocation.invocation_id
                if self._active_invocation is not None
                else (
                    None
                    if self._last_invocation is None
                    else self._last_invocation.invocation_id
                )
            ),
        )
        self._transition_log.append(record)
        if self._logger is not None:
            try:
                attempt = (
                    ""
                    if recovery_attempt is None
                    else f" recovery_attempt={recovery_attempt}"
                )
                self._logger(
                    f"[SkillManager] t={timestamp:.3f} "
                    f"uav_id={self._uav_id} mission_id={record.mission_id} "
                    f"{_name_or_none(old_skill)}[{old_step_id or '-'}] -> "
                    f"{_name_or_none(new_skill)}[{new_step_id or '-'}] "
                    f"reason={record.reason}{attempt}"
                )
            except Exception:
                pass

    def _read_transition_time(self) -> float:
        try:
            value = self._context.clock.now()
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError
            timestamp = float(value)
        except Exception:
            timestamp = self._last_transition_time
        timestamp = max(self._last_transition_time, timestamp)
        self._last_transition_time = timestamp
        return timestamp

    def _clear_task_to_idle(self) -> None:
        self._task_status = TaskStatus.IDLE
        self._task_plan = None
        self._plan_index = None
        self._pending_task_result = None
        self._active_target_id = None
        self._active_planned_step_id = None
        self._active_execution_kind = None
        self._task_failure_result = None
        self._transition_log = []
        self._step_outputs = {}
        self._recovery_attempts = {}
        self._saved_track_goal_by_step = {}
        self._last_result = None
        self._active_invocation = None
        self._last_invocation = None
        self._execution_reports = []
        self._invocation_counter = 0
        self._search_inspection_detour = None
        self._discard_supervisory_state()

    def _active_skill_or_none(self) -> Skill | None:
        if self._active_name is None:
            return None
        return self._skills[self._active_name]

    def _active_skill(self) -> Skill:
        skill = self._active_skill_or_none()
        if skill is None:
            raise SkillManagerError("there is no active Skill")
        return skill


_EXPECTED_SUCCESS_CODES: dict[SkillName, SkillResultCode] = {
    SkillName.TAKEOFF: SkillResultCode.TAKEOFF_COMPLETE,
    SkillName.GOTO: SkillResultCode.GOAL_REACHED,
    SkillName.HOVER: SkillResultCode.HOVER_COMPLETE,
    SkillName.SEARCH: SkillResultCode.TARGET_FOUND,
    SkillName.INSPECT: SkillResultCode.GOAL_REACHED,
    SkillName.TRACK: SkillResultCode.TRACK_COMPLETE,
    SkillName.REACQUIRE: SkillResultCode.TARGET_FOUND,
    SkillName.LAND: SkillResultCode.LAND_COMPLETE,
}

_GOAL_TYPES: dict[SkillName, type[SkillGoal]] = {
    SkillName.TAKEOFF: TakeoffGoal,
    SkillName.GOTO: GotoGoal,
    SkillName.HOVER: HoverGoal,
    SkillName.SEARCH: SearchGoal,
    SkillName.TRACK: TrackGoal,
    SkillName.LAND: LandGoal,
}


def _reject_unknown_goal_fields(
    name: SkillName, params: Mapping[str, object]
) -> None:
    goal_type = _goal_type_for_name(name)
    if goal_type is None:
        raise TaskPlanError(f"{name.value} cannot appear in TaskPlan")
    allowed = {field.name for field in fields(goal_type)}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise TaskPlanError(f"unknown {name.value} parameter(s): {', '.join(unknown)}")
    required = {
        field.name
        for field in fields(goal_type)
        if field.default is MISSING and field.default_factory is MISSING
    }
    if name is SkillName.TRACK:
        required.discard("target_id")
    if name is SkillName.SEARCH:
        required.discard("search_altitude")
    missing = sorted(required - set(params))
    if missing:
        raise TaskPlanError(f"missing {name.value} parameter(s): {', '.join(missing)}")


def _goal_type_for_name(name: SkillName) -> type[SkillGoal] | None:
    if name is SkillName.INSPECT:
        # INSPECT's grounding dependencies traverse perception/runtime
        # packages.  Import lazily so importing the foundational skills.plan
        # module cannot create a planner.schemas -> runtime cycle.
        from skills.inspect import InspectGoal

        return InspectGoal
    return _GOAL_TYPES.get(name)


def _manager_skill_name(value: SkillName | str | object) -> SkillName:
    if isinstance(value, SkillName):
        return value
    if isinstance(value, str):
        try:
            return SkillName(value.upper())
        except ValueError as exc:
            raise SkillManagerError(f"unknown Skill name: {value}") from exc
    raise TypeError("Skill name must be SkillName or str")


def _yaw_mode(value: object) -> YawMode:
    if isinstance(value, YawMode):
        return value
    if isinstance(value, str):
        try:
            return YawMode[value.upper()]
        except KeyError as exc:
            raise TaskPlanError(f"unknown yaw_mode: {value}") from exc
    raise TaskPlanError("yaw_mode must be a YawMode or string")


def _hover_mode(value: object) -> HoverMode:
    if isinstance(value, HoverMode):
        return value
    if isinstance(value, str):
        try:
            return HoverMode(value.upper())
        except ValueError as exc:
            raise TaskPlanError(f"unknown HOVER mode: {value}") from exc
    raise TaskPlanError("HOVER mode must be a HoverMode or string")


def _hover_timeout_fallback(
    value: HoverTimeoutFallback | str | object,
) -> HoverTimeoutFallback:
    if isinstance(value, HoverTimeoutFallback):
        return value
    if isinstance(value, str):
        try:
            return HoverTimeoutFallback(value.upper())
        except ValueError as exc:
            raise SkillManagerError(
                f"unknown supervisory HOVER timeout fallback: {value}"
            ) from exc
    raise TypeError("timeout_fallback must be a HoverTimeoutFallback or string")


def _validate_hover_goal_for_manager(
    goal: HoverGoal,
    *,
    supervisory: bool,
) -> None:
    if supervisory:
        if goal.mode is not HoverMode.UNTIL_RELEASED or goal.duration_s is not None:
            raise SkillManagerError(
                "supervisory HOVER must be UNTIL_RELEASED with duration_s=None"
            )
    elif goal.mode is not HoverMode.TIMED:
        raise TaskPlanError("planned HOVER must use TIMED mode")
    _positive_number(goal.max_wait_s, "max_wait_s")
    _positive_number(goal.position_tolerance_m, "position_tolerance_m")
    _positive_number(
        goal.max_correction_speed_mps, "max_correction_speed_mps"
    )
    if goal.mode is HoverMode.TIMED:
        duration = _positive_number(goal.duration_s, "duration_s")
        if duration > float(goal.max_wait_s):
            raise TaskPlanError("HOVER duration_s must not exceed max_wait_s")
    validate_routing_id(goal.reason_code, "reason_code")
    if not isinstance(goal.motion_policy, MotionPolicy):
        raise TypeError("HOVER motion_policy must be a MotionPolicy")
    goal.motion_policy.validate()


def _motion_policy(value: object) -> MotionPolicy:
    if isinstance(value, MotionPolicy):
        return value
    if not isinstance(value, Mapping):
        raise TaskPlanError("motion_policy must be a MotionPolicy or mapping")
    params = deepcopy(dict(value))
    allowed = {field.name for field in fields(MotionPolicy)}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise TaskPlanError("unknown MotionPolicy parameter(s): " + ", ".join(unknown))
    if "yaw_mode" in params:
        params["yaw_mode"] = _yaw_mode(params["yaw_mode"])
    if isinstance(params.get("look_at_point"), list):
        params["look_at_point"] = tuple(params["look_at_point"])
    try:
        return MotionPolicy(**params)
    except (TypeError, ValueError) as exc:
        raise TaskPlanError(f"invalid MotionPolicy: {exc}") from exc


def _positive_number(value: object, name: str) -> float:
    parsed = _positive_number_or_none(value)
    if parsed is None:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return parsed


def _positive_number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    parsed = float(value)
    return parsed if isfinite(parsed) and parsed > 0.0 else None


def _nonnegative_number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    parsed = float(value)
    return parsed if isfinite(parsed) and parsed >= 0.0 else None


def _finite_vector3_or_none(value: object) -> tuple[float, float, float] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        return None
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real) or not isfinite(item):
            return None
        normalized.append(float(item))
    return normalized[0], normalized[1], normalized[2]


def _copy_result(result: SkillResult | None) -> SkillResult | None:
    if result is None:
        return None
    return SkillResult(
        status=result.status,
        code=result.code,
        message=result.message,
        data=deepcopy(result.data),
    )


def _copy_task_plan(plan: TaskPlan) -> TaskPlan:
    """Rebuild a plan so no mutable params object is shared with its caller."""

    return TaskPlan(
        tuple(
            TaskStep(
                step.step_id,
                step.skill,
                step.params,
                step.recovery,
            )
            for step in plan.steps
        ),
        mission_id=plan.mission_id,
        uav_id=plan.uav_id,
        plan_version=plan.plan_version,
    )


def _copy_report(report: SkillExecutionReport) -> SkillExecutionReport:
    return SkillExecutionReport(
        mission_id=report.mission_id,
        uav_id=report.uav_id,
        plan_version=report.plan_version,
        step_id=report.step_id,
        invocation_id=report.invocation_id,
        skill_name=report.skill_name,
        status=report.status,
        result_code=report.result_code,
        feedback_or_result=deepcopy(report.feedback_or_result),
        timestamp_s=report.timestamp_s,
    )


def _internal_result(
    message: str, data: dict[str, object] | None = None
) -> SkillResult:
    return SkillResult(
        status=SkillStatus.FAILED,
        code=SkillResultCode.INTERNAL_ERROR,
        message=message,
        data={} if data is None else data,
    )


def _failure_reason(name: SkillName, code: SkillResultCode) -> str:
    if name is SkillName.SEARCH and code is SkillResultCode.SEARCH_EXHAUSTED:
        return "search_exhausted"
    if name is SkillName.SEARCH and code is SkillResultCode.TIMEOUT:
        return "search_timeout"
    if name is SkillName.REACQUIRE and code is SkillResultCode.TIMEOUT:
        return "reacquire_timeout"
    if name is SkillName.TRACK and code is SkillResultCode.TARGET_LOST:
        return "target_lost_recovery_unavailable"
    return f"{name.value.lower()}_{code.name.lower()}"


def _normal_transition_reason(name: SkillName) -> str:
    return {
        SkillName.TAKEOFF: "takeoff_complete",
        SkillName.GOTO: "goal_reached",
        SkillName.SEARCH: "target_found",
        SkillName.TRACK: "track_complete",
    }.get(name, "step_complete")


def _name_or_none(name: SkillName | None) -> str:
    return "NONE" if name is None else name.value


__all__ = [
    "ExecutionKind",
    "HoverTimeoutFallback",
    "RecoveryPolicy",
    "SkillManager",
    "SkillManagerError",
    "SkillNotRegisteredError",
    "StepOutputRef",
    "TaskPlan",
    "TaskPlanError",
    "TaskStatus",
    "TaskStep",
    "TransitionRecord",
    "create_default_skill_registry",
]
