"""Single-Skill dispatch and the Stage-0 Oracle task state machine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, dataclass, fields, replace
from enum import Enum, auto
from math import isfinite
from numbers import Real

from skills.base import Skill, SkillLifecycleError
from skills.goto import GotoGoal, GotoSkill
from skills.land import LandGoal, LandSkill
from skills.motion_types import MotionPolicy, YawMode
from skills.reacquire import ReacquireGoal, ReacquireSkill
from skills.search import SearchGoal, SearchSkill
from skills.takeoff import TakeoffGoal, TakeoffSkill
from skills.track import TrackGoal, TrackSkill
from skills.types import (
    Observation,
    SkillContext,
    SkillFeedback,
    SkillGoal,
    SkillName,
    SkillResult,
    SkillResultCode,
    SkillStatus,
)


class SkillManagerError(RuntimeError):
    """Base class for Manager registration, plan, and active-Skill errors."""


class SkillNotRegisteredError(SkillManagerError):
    """Raised when a requested Qwen tool name has no registered Skill."""


class TaskPlanError(SkillManagerError):
    """Raised when a Stage-0 task plan is structurally invalid."""


class TaskStatus(Enum):
    """Lifecycle of the complete multi-Skill task, independent of SkillStatus."""

    IDLE = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELED = auto()


@dataclass(frozen=True, slots=True)
class TaskStep:
    """One orchestration-layer step before conversion to a typed SkillGoal."""

    skill: SkillName
    params: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill", _skill_name(self.skill))
        if not isinstance(self.params, Mapping):
            raise TaskPlanError("TaskStep.params must be a mapping")
        normalized = deepcopy(dict(self.params))
        if "skill" in normalized:
            raise TaskPlanError("TaskStep.params must not contain a skill key")
        object.__setattr__(self, "params", normalized)

    def to_dict(self) -> dict[str, object]:
        return {"skill": self.skill.value, **deepcopy(dict(self.params))}


_STANDARD_TASK_SEQUENCE = (
    SkillName.TAKEOFF,
    SkillName.GOTO,
    SkillName.SEARCH,
    SkillName.TRACK,
    SkillName.LAND,
)


@dataclass(frozen=True, slots=True)
class TaskPlan:
    """Hand-written Stage-0 plan; recovery steps are injected by the Manager."""

    steps: tuple[TaskStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple) or not all(
            isinstance(step, TaskStep) for step in self.steps
        ):
            raise TaskPlanError("TaskPlan.steps must be a tuple of TaskStep values")
        names = tuple(step.skill for step in self.steps)
        if names != _STANDARD_TASK_SEQUENCE:
            expected = " -> ".join(name.value for name in _STANDARD_TASK_SEQUENCE)
            actual = " -> ".join(name.value for name in names) or "<empty>"
            raise TaskPlanError(
                f"Stage-0 TaskPlan must be {expected}; received {actual}"
            )

    @classmethod
    def from_dicts(cls, entries: Sequence[Mapping[str, object]]) -> TaskPlan:
        """Parse the user-facing list format without passing dict Goals to Skills."""

        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise TaskPlanError("task plan must be a sequence of mappings")
        parsed: list[TaskStep] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise TaskPlanError(f"task plan entry {index} must be a mapping")
            if "skill" not in entry:
                raise TaskPlanError(f"task plan entry {index} is missing skill")
            if any(not isinstance(key, str) for key in entry):
                raise TaskPlanError(
                    f"task plan entry {index} keys must be strings"
                )
            name = _skill_name(entry["skill"])
            params = {key: deepcopy(value) for key, value in entry.items() if key != "skill"}
            _reject_unknown_goal_fields(name, params)
            parsed.append(TaskStep(name, params))
        return cls(tuple(parsed))

    def to_dicts(self) -> list[dict[str, object]]:
        return [step.to_dict() for step in self.steps]


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One externally inspectable task transition."""

    timestamp: float
    old_skill: SkillName | None
    old_status: SkillStatus | None
    result_code: SkillResultCode | None
    new_skill: SkillName | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "old_skill": None if self.old_skill is None else self.old_skill.value,
            "old_status": None if self.old_status is None else self.old_status.name,
            "result_code": None if self.result_code is None else self.result_code.name,
            "new_skill": None if self.new_skill is None else self.new_skill.value,
            "reason": self.reason,
        }


def create_default_skill_registry(
    *,
    transit_yaw_mode: YawMode | str = YawMode.FACE_POINT,
) -> dict[SkillName, Skill]:
    """Create exactly one fresh instance of every Stage-0 callable Skill."""

    return {
        SkillName.TAKEOFF: TakeoffSkill(),
        SkillName.GOTO: GotoSkill(),
        SkillName.SEARCH: SearchSkill(transit_yaw_mode=transit_yaw_mode),
        SkillName.TRACK: TrackSkill(),
        SkillName.REACQUIRE: ReacquireSkill(),
        SkillName.LAND: LandSkill(),
    }


class SkillManager:
    """Dispatch one Skill at a time and optionally run the complete task plan.

    The original manual ``start/tick/reset_active`` API remains available.  A
    caller enters task mode explicitly with :meth:`start_task`; in that mode a
    tick returns :class:`TaskStatus` and performs at most one Skill tick.
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
        self._context = context
        self._skills: dict[SkillName, Skill] = {}
        self._active_name: SkillName | None = None
        self._last_result: SkillResult | None = None

        self._reacquire_search_radius = _positive_number(
            reacquire_search_radius,
            "reacquire_search_radius",
        )
        self._reacquire_timeout = _positive_number(
            reacquire_timeout,
            "reacquire_timeout",
        )
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable or None")
        self._logger = logger

        self._task_status = TaskStatus.IDLE
        self._task_plan: TaskPlan | None = None
        self._plan_index: int | None = None
        self._pending_task_result: TaskStatus | None = None
        self._active_target_id: str | None = None
        self._saved_track_goal: TrackGoal | None = None
        self._task_failure_result: SkillResult | None = None
        self._transition_log: list[TransitionRecord] = []
        self._last_transition_time = 0.0

        if registry is not None:
            if not isinstance(registry, Mapping):
                raise TypeError("registry must be a mapping or None")
            for name, skill in registry.items():
                self.register(name, skill)

    @property
    def active_name(self) -> SkillName | None:
        return self._active_name

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
    def pending_task_result(self) -> TaskStatus | None:
        return self._pending_task_result

    @property
    def active_target_id(self) -> str | None:
        return self._active_target_id

    @property
    def task_failure_result(self) -> SkillResult | None:
        return _copy_result(self._task_failure_result)

    @property
    def transition_log(self) -> tuple[TransitionRecord, ...]:
        return tuple(self._transition_log)

    @property
    def skill_registry(self) -> dict[SkillName, Skill]:
        """Return a shallow registry snapshot; Skill instances remain Manager-owned."""

        return dict(self._skills)

    def register(self, name: SkillName | str, skill: Skill) -> None:
        normalized = _skill_name(name)
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
        return tuple(self._skills.keys())

    def start(self, name: SkillName | str, goal: SkillGoal) -> SkillStatus:
        """Start one Skill manually; unavailable while a task plan is active."""

        if self._task_status is not TaskStatus.IDLE:
            raise SkillManagerError("reset the task before using manual Skill dispatch")
        return self._start_registered(_skill_name(name), goal)

    def tick(self, observation: Observation) -> SkillStatus | TaskStatus:
        if self._task_status is TaskStatus.RUNNING:
            return self._tick_task(observation)
        if self._task_status is not TaskStatus.IDLE:
            # Task terminal states are stable and safe to poll from a runtime
            # loop while it is shutting down.
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
        try:
            skill.reset()
        finally:
            # Base reset is exception-safe and reaches IDLE even if a subclass
            # cleanup hook fails. Never leave Manager pointing at an IDLE Skill.
            self._last_result = result
            self._active_name = None

    def start_task(self, plan: TaskPlan) -> TaskStatus:
        """Atomically validate and start the standard Stage-0 task."""

        if not isinstance(plan, TaskPlan):
            raise TypeError("plan must be a TaskPlan")
        if self._task_status is not TaskStatus.IDLE:
            raise SkillManagerError("reset_task() is required before starting another task")
        if self._active_name is not None:
            raise SkillManagerError("reset the manually active Skill before starting a task")
        missing = [name.value for name in set(_STANDARD_TASK_SEQUENCE) | {SkillName.REACQUIRE} if name not in self._skills]
        if missing:
            raise SkillNotRegisteredError(
                "task registry is missing: " + ", ".join(sorted(missing))
            )

        # Compile every planned Goal before mutating task state. TRACK uses a
        # harmless validation id until SEARCH supplies the real result.
        for step in plan.steps:
            self._goal_from_step(step, plan=plan, validation_only=True)

        self._task_plan = plan
        self._plan_index = 0
        self._pending_task_result = None
        self._active_target_id = None
        self._saved_track_goal = None
        self._task_failure_result = None
        self._transition_log = []
        self._last_transition_time = 0.0
        self._task_status = TaskStatus.RUNNING
        first = plan.steps[0]
        goal = self._goal_from_step(first)
        try:
            self._start_registered(first.skill, goal)
        except Exception:
            self._clear_task_to_idle()
            raise
        self._record_transition(
            old_skill=None,
            old_status=None,
            result_code=None,
            new_skill=first.skill,
            reason="task_started",
        )
        return self._task_status

    def cancel_task(self) -> TaskStatus:
        """Cancel the active work, then LAND before committing CANCELED."""

        if self._task_status is not TaskStatus.RUNNING:
            raise SkillManagerError("there is no RUNNING task to cancel")
        old_name = self._active_name
        if old_name is SkillName.LAND:
            # LAND is already the fail-safe action. Do not cancel it in mid-air;
            # only change the Task result committed after LAND_COMPLETE. An
            # existing failure always has precedence over a later cancel.
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
                reset_failure = SkillResult(
                    status=SkillStatus.FAILED,
                    code=SkillResultCode.INTERNAL_ERROR,
                    message=f"could not reset canceled {old_name.value}: {exc}",
                    data={"reset_error": str(exc)},
                )
                self._last_result = reset_failure
                self._task_failure_result = reset_failure
                self._pending_task_result = TaskStatus.FAILED
                self._begin_landing(
                    old_name,
                    old_status,
                    reset_failure,
                    "canceled_skill_reset_failed",
                )
                return self._task_status
        self._pending_task_result = TaskStatus.CANCELED
        self._begin_landing(
            old_name,
            old_status,
            result,
            "task_canceled",
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

    def _start_registered(self, name: SkillName, goal: SkillGoal) -> SkillStatus:
        if self._active_name is not None:
            raise SkillManagerError("reset the active Skill before starting another")
        try:
            skill = self._skills[name]
        except KeyError as exc:
            raise SkillNotRegisteredError(f"Skill {name.value} is not registered") from exc
        if skill.status is not SkillStatus.IDLE:
            raise SkillManagerError(f"registered Skill {name.value} is not IDLE")
        self._active_name = name
        try:
            skill.start(goal, self._context)
        except BaseException:
            if skill.status is SkillStatus.IDLE:
                self._active_name = None
            raise
        return skill.status

    def _tick_task(self, observation: Observation) -> TaskStatus:
        if self._active_name is None:
            self._fail_without_landing("task has no active Skill")
            return self._task_status

        skill = self._active_skill()
        if skill.status is SkillStatus.RUNNING:
            skill.tick(observation)
        if skill.status is SkillStatus.RUNNING:
            return self._task_status

        old_name = self._active_name
        old_status = skill.status
        result = skill.get_result()
        if result is None:
            self._fail_without_landing("terminal Skill has no result")
            return self._task_status
        try:
            self._reset_active_internal()
        except Exception as exc:
            reset_failure = SkillResult(
                status=SkillStatus.FAILED,
                code=SkillResultCode.INTERNAL_ERROR,
                message=f"could not reset {old_name.value}: {exc}",
                data={
                    "reset_error": str(exc),
                    "previous_result": result.to_dict(),
                },
            )
            self._last_result = reset_failure
            self._task_failure_result = reset_failure
            self._pending_task_result = TaskStatus.FAILED
            if old_name is SkillName.LAND:
                self._finish_task(
                    TaskStatus.FAILED,
                    old_name,
                    old_status,
                    SkillResultCode.INTERNAL_ERROR,
                    "landing_reset_failed",
                )
            else:
                self._begin_landing(
                    old_name,
                    old_status,
                    reset_failure,
                    "skill_reset_failed",
                )
            return self._task_status

        self._transition_from_result(old_name, old_status, result)
        return self._task_status

    def _transition_from_result(
        self,
        old_name: SkillName,
        old_status: SkillStatus,
        result: SkillResult,
    ) -> None:
        if old_name is SkillName.LAND:
            if (
                old_status is SkillStatus.SUCCEEDED
                and result.code is SkillResultCode.LAND_COMPLETE
            ):
                final = self._pending_task_result or TaskStatus.SUCCEEDED
                reason = {
                    TaskStatus.SUCCEEDED: "task_completed",
                    TaskStatus.FAILED: "failure_landing_complete",
                    TaskStatus.CANCELED: "cancel_landing_complete",
                }[final]
                self._finish_task(final, old_name, old_status, result.code, reason)
            else:
                self._task_failure_result = self._task_failure_result or result
                self._finish_task(
                    TaskStatus.FAILED,
                    old_name,
                    old_status,
                    result.code,
                    "landing_failed",
                )
            return

        if old_status is SkillStatus.CANCELED:
            self._pending_task_result = TaskStatus.CANCELED
            self._begin_landing(old_name, old_status, result, "skill_canceled")
            return

        if old_status is SkillStatus.FAILED:
            if old_name is SkillName.TRACK and result.code is SkillResultCode.TARGET_LOST:
                recovery_goal = self._reacquire_goal_from_result(result)
                if recovery_goal is not None:
                    self._start_transition(
                        old_name,
                        old_status,
                        result.code,
                        SkillName.REACQUIRE,
                        recovery_goal,
                        "target_lost",
                    )
                    return
            self._task_failure_result = result
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                old_name,
                old_status,
                result,
                _failure_reason(old_name, result.code),
            )
            return

        if old_status is not SkillStatus.SUCCEEDED:
            self._task_failure_result = result
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(old_name, old_status, result, "invalid_skill_status")
            return

        expected_codes = {
            SkillName.TAKEOFF: SkillResultCode.TAKEOFF_COMPLETE,
            SkillName.GOTO: SkillResultCode.GOAL_REACHED,
            SkillName.SEARCH: SkillResultCode.TARGET_FOUND,
            SkillName.REACQUIRE: SkillResultCode.TARGET_FOUND,
            SkillName.TRACK: SkillResultCode.TRACK_COMPLETE,
        }
        if result.code is not expected_codes.get(old_name):
            self._task_failure_result = result
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(old_name, old_status, result, "unexpected_result_code")
            return

        if old_name is SkillName.SEARCH:
            target_id = result.data.get("target_id")
            if not isinstance(target_id, str) or not target_id.strip():
                self._task_failure_result = result
                self._pending_task_result = TaskStatus.FAILED
                self._begin_landing(
                    old_name,
                    old_status,
                    result,
                    "search_result_missing_target_id",
                )
                return
            self._active_target_id = target_id.strip()

        if old_name is SkillName.REACQUIRE:
            target_id = result.data.get("target_id", self._active_target_id)
            if isinstance(target_id, str) and target_id.strip():
                self._active_target_id = target_id.strip()
            if self._saved_track_goal is None or self._active_target_id is None:
                self._task_failure_result = result
                self._pending_task_result = TaskStatus.FAILED
                self._begin_landing(
                    old_name,
                    old_status,
                    result,
                    "recovery_state_missing",
                )
                return
            track_goal = replace(
                self._saved_track_goal,
                target_id=self._active_target_id,
            )
            self._start_transition(
                old_name,
                old_status,
                result.code,
                SkillName.TRACK,
                track_goal,
                "target_reacquired",
            )
            return

        if old_name is SkillName.TRACK:
            self._pending_task_result = TaskStatus.SUCCEEDED
            self._begin_landing(
                old_name,
                old_status,
                result,
                "track_complete",
            )
            return

        self._start_next_planned(old_name, old_status, result)

    def _start_next_planned(
        self,
        old_name: SkillName,
        old_status: SkillStatus,
        result: SkillResult,
    ) -> None:
        if self._task_plan is None or self._plan_index is None:
            self._fail_without_landing("task plan state is missing")
            return
        next_index = self._plan_index + 1
        if next_index >= len(self._task_plan.steps):
            self._task_failure_result = result
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(old_name, old_status, result, "plan_ended_early")
            return
        step = self._task_plan.steps[next_index]
        try:
            goal = self._goal_from_step(step)
        except Exception as exc:
            self._task_failure_result = result
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                old_name,
                old_status,
                result,
                f"next_goal_invalid:{exc}",
            )
            return
        self._plan_index = next_index
        if step.skill is SkillName.TRACK:
            if not isinstance(goal, TrackGoal):
                self._fail_without_landing("TRACK plan did not compile to TrackGoal")
                return
            self._saved_track_goal = goal
        self._start_transition(
            old_name,
            old_status,
            result.code,
            step.skill,
            goal,
            _normal_transition_reason(old_name),
        )

    def _start_transition(
        self,
        old_name: SkillName | None,
        old_status: SkillStatus | None,
        result_code: SkillResultCode | None,
        new_name: SkillName,
        goal: SkillGoal,
        reason: str,
    ) -> None:
        try:
            self._start_registered(new_name, goal)
        except Exception as exc:
            self._context.uav.stop()
            if new_name is SkillName.LAND:
                self._finish_task(
                    TaskStatus.FAILED,
                    old_name,
                    old_status,
                    result_code,
                    f"landing_start_failed:{exc}",
                )
                return
            self._pending_task_result = TaskStatus.FAILED
            self._begin_landing(
                old_name,
                old_status,
                self._last_result,
                f"skill_start_failed:{new_name.value}:{exc}",
            )
            return
        self._record_transition(
            old_skill=old_name,
            old_status=old_status,
            result_code=result_code,
            new_skill=new_name,
            reason=reason,
        )

    def _begin_landing(
        self,
        old_name: SkillName | None,
        old_status: SkillStatus | None,
        result: SkillResult | None,
        reason: str,
    ) -> None:
        if old_name is SkillName.LAND:
            self._finish_task(
                TaskStatus.FAILED,
                old_name,
                old_status,
                None if result is None else result.code,
                "landing_failed",
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
            )
            return
        land_goal = self._planned_land_goal()
        if self._task_plan is not None:
            self._plan_index = len(self._task_plan.steps) - 1
        self._start_transition(
            old_name,
            old_status,
            None if result is None else result.code,
            SkillName.LAND,
            land_goal,
            reason,
        )

    def _reacquire_goal_from_result(
        self,
        result: SkillResult,
    ) -> ReacquireGoal | None:
        target_id = result.data.get("target_id", self._active_target_id)
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
        self._reduce_remaining_track_duration(result)
        return ReacquireGoal(
            target_id=self._active_target_id,
            last_seen_position=position,
            last_seen_velocity=velocity,
            last_seen_time=last_seen_time,
            search_radius=self._reacquire_search_radius,
            timeout=self._reacquire_timeout,
        )

    def _reduce_remaining_track_duration(self, result: SkillResult) -> None:
        """Carry cumulative TRACK time across a successful recovery."""

        goal = self._saved_track_goal
        if goal is None or goal.track_duration is None:
            return
        elapsed = _nonnegative_number_or_none(result.data.get("tracking_duration"))
        if elapsed is None:
            return
        remaining = max(1e-9, float(goal.track_duration) - elapsed)
        self._saved_track_goal = replace(goal, track_duration=remaining)

    def _planned_land_goal(self) -> LandGoal:
        if self._task_plan is not None:
            step = self._task_plan.steps[-1]
            try:
                goal = self._goal_from_step(step)
                if isinstance(goal, LandGoal):
                    return goal
            except Exception:
                pass
        return LandGoal()

    def _goal_from_step(
        self,
        step: TaskStep,
        *,
        plan: TaskPlan | None = None,
        validation_only: bool = False,
    ) -> SkillGoal:
        params = deepcopy(dict(step.params))
        if step.skill is SkillName.TRACK:
            raw_target = params.get("target_id", "$SEARCH.result.target_id")
            if raw_target == "$SEARCH.result.target_id":
                target_id = "__search_target__" if validation_only else self._active_target_id
                if not isinstance(target_id, str) or not target_id:
                    raise TaskPlanError("TRACK target_id requires SEARCH TARGET_FOUND")
                params["target_id"] = target_id
            elif not isinstance(raw_target, str) or not raw_target.strip():
                raise TaskPlanError("TRACK target_id must be a non-empty string")
            params.setdefault("track_duration", 30.0)
        if step.skill is SkillName.SEARCH and "search_altitude" not in params:
            params["search_altitude"] = self._default_search_altitude(plan)

        if "yaw_mode" in params:
            params["yaw_mode"] = _yaw_mode(params["yaw_mode"])
        if step.skill is SkillName.GOTO and "motion_policy" in params:
            params["motion_policy"] = _motion_policy(params["motion_policy"])

        vector_fields = {
            "position",
            "center",
            "look_at_point",
            "last_seen_position",
            "last_seen_velocity",
        }
        for key in vector_fields & params.keys():
            value = params[key]
            if isinstance(value, list):
                params[key] = tuple(value)

        goal_types: dict[SkillName, type[SkillGoal]] = {
            SkillName.TAKEOFF: TakeoffGoal,
            SkillName.GOTO: GotoGoal,
            SkillName.SEARCH: SearchGoal,
            SkillName.TRACK: TrackGoal,
            SkillName.LAND: LandGoal,
        }
        goal_type = goal_types.get(step.skill)
        if goal_type is None:
            raise TaskPlanError(f"{step.skill.value} is not a planned task step")
        try:
            return goal_type(**params)
        except (TypeError, ValueError) as exc:
            raise TaskPlanError(f"invalid {step.skill.value} parameters: {exc}") from exc

    def _default_search_altitude(self, plan: TaskPlan | None) -> float:
        source = plan or self._task_plan
        if source is not None:
            takeoff_value = source.steps[0].params.get("target_altitude")
            value = _positive_number_or_none(takeoff_value)
            if value is not None:
                return value
        altitude = float(self._context.uav.get_pose().z)
        if not isfinite(altitude) or altitude <= 0.0:
            raise TaskPlanError(
                "SEARCH search_altitude is required when no positive TAKEOFF altitude exists"
            )
        return altitude

    def _finish_task(
        self,
        status: TaskStatus,
        old_name: SkillName | None,
        old_status: SkillStatus | None,
        result_code: SkillResultCode | None,
        reason: str,
    ) -> None:
        self._context.uav.stop()
        self._task_status = status
        self._record_transition(
            old_skill=old_name,
            old_status=old_status,
            result_code=result_code,
            new_skill=None,
            reason=reason,
        )

    def _fail_without_landing(self, reason: str) -> None:
        old_name = self._active_name
        old_status = self.active_status
        result: SkillResult | None = None
        if old_name is not None:
            skill = self._active_skill()
            try:
                if skill.status is SkillStatus.RUNNING:
                    skill.cancel()
                result = skill.get_result()
                if result is not None:
                    self._reset_active_internal()
                else:
                    # A corrupted terminal hook may have lost its Result. Base
                    # reset still returns the instance to IDLE, so release it
                    # before trying the fail-safe LAND.
                    try:
                        skill.reset()
                    finally:
                        self._active_name = None
            except Exception:
                self._active_name = None
        internal_result = SkillResult(
            status=SkillStatus.FAILED,
            code=SkillResultCode.INTERNAL_ERROR,
            message=reason,
            data={},
        )
        self._task_failure_result = internal_result
        self._pending_task_result = TaskStatus.FAILED
        if old_name is not SkillName.LAND and SkillName.LAND in self._skills:
            self._begin_landing(old_name, old_status, internal_result, reason)
            return
        self._finish_task(
            TaskStatus.FAILED,
            old_name,
            old_status,
            None if result is None else result.code,
            reason,
        )

    def _record_transition(
        self,
        *,
        old_skill: SkillName | None,
        old_status: SkillStatus | None,
        result_code: SkillResultCode | None,
        new_skill: SkillName | None,
        reason: str,
    ) -> None:
        timestamp = self._read_transition_time()
        record = TransitionRecord(
            timestamp=timestamp,
            old_skill=old_skill,
            old_status=old_status,
            result_code=result_code,
            new_skill=new_skill,
            reason=str(reason),
        )
        self._transition_log.append(record)
        if self._logger is not None:
            try:
                self._logger(
                    f"[SkillManager] t={timestamp:.3f} "
                    f"{_name_or_none(old_skill)} -> {_name_or_none(new_skill)} "
                    f"reason={record.reason}"
                )
            except Exception:
                # Logging is observational and must never alter flight safety.
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
        self._saved_track_goal = None
        self._task_failure_result = None
        self._transition_log = []

    def _active_skill_or_none(self) -> Skill | None:
        if self._active_name is None:
            return None
        return self._skills[self._active_name]

    def _active_skill(self) -> Skill:
        skill = self._active_skill_or_none()
        if skill is None:
            raise SkillManagerError("there is no active Skill")
        return skill


_GOAL_TYPES: dict[SkillName, type[SkillGoal]] = {
    SkillName.TAKEOFF: TakeoffGoal,
    SkillName.GOTO: GotoGoal,
    SkillName.SEARCH: SearchGoal,
    SkillName.TRACK: TrackGoal,
    SkillName.LAND: LandGoal,
}


def _reject_unknown_goal_fields(
    name: SkillName,
    params: Mapping[str, object],
) -> None:
    goal_type = _GOAL_TYPES.get(name)
    if goal_type is None:
        raise TaskPlanError(f"{name.value} cannot appear in the standard TaskPlan")
    allowed = {field.name for field in fields(goal_type)}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise TaskPlanError(
            f"unknown {name.value} parameter(s): {', '.join(unknown)}"
        )
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
        raise TaskPlanError(
            f"missing {name.value} parameter(s): {', '.join(missing)}"
        )


def _skill_name(value: SkillName | str | object) -> SkillName:
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


def _motion_policy(value: object) -> MotionPolicy:
    if isinstance(value, MotionPolicy):
        return value
    if not isinstance(value, Mapping):
        raise TaskPlanError("motion_policy must be a MotionPolicy or mapping")
    params = deepcopy(dict(value))
    allowed = {field.name for field in fields(MotionPolicy)}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise TaskPlanError(
            "unknown MotionPolicy parameter(s): " + ", ".join(unknown)
        )
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


def _failure_reason(name: SkillName, code: SkillResultCode) -> str:
    if name is SkillName.SEARCH and code is SkillResultCode.SEARCH_EXHAUSTED:
        return "search_exhausted"
    if name is SkillName.SEARCH and code is SkillResultCode.TIMEOUT:
        return "search_timeout"
    if name is SkillName.REACQUIRE and code is SkillResultCode.TIMEOUT:
        return "reacquire_timeout"
    return f"{name.value.lower()}_{code.name.lower()}"


def _normal_transition_reason(name: SkillName) -> str:
    return {
        SkillName.TAKEOFF: "takeoff_complete",
        SkillName.GOTO: "goal_reached",
        SkillName.SEARCH: "target_found",
    }.get(name, "step_complete")


def _name_or_none(name: SkillName | None) -> str:
    return "NONE" if name is None else name.value


__all__ = [
    "SkillManager",
    "SkillManagerError",
    "SkillNotRegisteredError",
    "TaskPlan",
    "TaskPlanError",
    "TaskStatus",
    "TaskStep",
    "TransitionRecord",
    "create_default_skill_registry",
]
