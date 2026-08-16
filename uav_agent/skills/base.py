"""Lifecycle-enforcing base class for every runtime-callable Skill."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence as SequenceABC
from copy import deepcopy
from math import isfinite
from numbers import Real
from typing import ClassVar, Sequence, final

import numpy as np

from skills.motion_types import (
    MotionPolicy,
    MotionPolicyValidationError,
    apply_motion_policy,
    move_toward_with_policy,
)
from skills.types import (
    Observation,
    SkillContext,
    SkillFeedback,
    SkillGoal,
    SkillResult,
    SkillResultCode,
    SkillStatus,
    validate_skill_result,
)


class SkillLifecycleError(RuntimeError):
    """Raised for an illegal public lifecycle transition."""


class SkillGoalValidationError(ValueError):
    """Raised by typed Goal validation and mapped to ``INVALID_GOAL``."""


class SkillExecutionStateError(RuntimeError):
    """Raised when a valid Goal cannot run in the current environment state."""


class Skill(ABC):
    """Template-method state machine shared by all Skills.

    Subclasses implement protected hooks and must use ``_succeed`` / ``_fail``
    to enter a terminal state. A Skill never owns or invokes another Skill.
    """

    goal_type: ClassVar[type[SkillGoal]] = SkillGoal

    def __init__(self) -> None:
        self._status = SkillStatus.IDLE
        self._goal: SkillGoal | None = None
        self._context: SkillContext | None = None
        self._initial_yaw: float | None = None
        self._motion_policy = MotionPolicy()
        self._feedback = SkillFeedback()
        self._result: SkillResult | None = None

    @property
    def status(self) -> SkillStatus:
        return self._status

    @property
    def initial_yaw(self) -> float | None:
        return self._initial_yaw

    @property
    def motion_policy(self) -> MotionPolicy:
        return self._motion_policy

    @final
    def start(self, goal: SkillGoal, context: SkillContext) -> None:
        self._require_status(SkillStatus.IDLE, "start")
        self._status = SkillStatus.RUNNING
        self._feedback = SkillFeedback(message="Skill started")
        self._result = None

        if not isinstance(context, SkillContext):
            self._complete(
                SkillStatus.FAILED,
                SkillResultCode.INVALID_STATE,
                "context must be a SkillContext",
            )
            return
        try:
            context.validate()
        except (TypeError, ValueError) as exc:
            self._complete(
                SkillStatus.FAILED,
                SkillResultCode.INVALID_STATE,
                str(exc),
            )
            return
        self._context = context

        if not isinstance(goal, self.goal_type):
            self._complete(
                SkillStatus.FAILED,
                SkillResultCode.INVALID_GOAL,
                f"expected {self.goal_type.__name__}, got {type(goal).__name__}",
            )
            return
        self._goal = goal

        try:
            self._initial_yaw = context.uav.get_pose().yaw
            candidate_policy = getattr(goal, "motion_policy", MotionPolicy())
            if not isinstance(candidate_policy, MotionPolicy):
                raise SkillGoalValidationError("goal.motion_policy must be a MotionPolicy")
            candidate_policy.validate()
            self._motion_policy = candidate_policy
            self._validate_goal(goal)
        except (MotionPolicyValidationError, SkillGoalValidationError) as exc:
            self._record_hook_failure(
                SkillResultCode.INVALID_GOAL,
                str(exc),
            )
            return
        except Exception as exc:
            self._record_hook_failure(
                SkillResultCode.INTERNAL_ERROR,
                f"goal validation failed unexpectedly: {exc}",
            )
            return
        if self._status is not SkillStatus.RUNNING:
            self._record_hook_failure(
                SkillResultCode.INTERNAL_ERROR,
                "goal validation hook attempted a lifecycle transition",
            )
            return

        try:
            self._on_start(goal, context)
        except SkillExecutionStateError as exc:
            self._record_hook_failure(SkillResultCode.INVALID_STATE, str(exc))
            return
        except Exception as exc:
            self._record_hook_failure(
                SkillResultCode.INTERNAL_ERROR,
                f"Skill start failed: {exc}",
            )
            return
        if self._status is not SkillStatus.RUNNING:
            self._record_hook_failure(
                SkillResultCode.INTERNAL_ERROR,
                "a valid Skill must remain RUNNING until its first tick",
            )

    @final
    def tick(self, observation: Observation) -> SkillStatus:
        self._require_status(SkillStatus.RUNNING, "tick")
        if not isinstance(observation, Observation):
            self._complete(
                SkillStatus.FAILED,
                SkillResultCode.INVALID_STATE,
                "observation must be an Observation",
            )
            return self._status
        try:
            observation.validate()
        except (TypeError, ValueError) as exc:
            self._complete(
                SkillStatus.FAILED,
                SkillResultCode.INVALID_STATE,
                str(exc),
            )
            return self._status
        try:
            self._on_tick(observation)
        except SkillExecutionStateError as exc:
            self._record_hook_failure(SkillResultCode.INVALID_STATE, str(exc))
        except Exception as exc:
            self._record_hook_failure(
                SkillResultCode.INTERNAL_ERROR,
                f"Skill tick failed: {exc}",
            )
        valid_post_tick_state = (
            self._status is SkillStatus.RUNNING
            and self._result is None
        ) or (
            (
                self._status is SkillStatus.SUCCEEDED
                or self._status is SkillStatus.FAILED
            )
            and isinstance(self._result, SkillResult)
            and self._result.status is self._status
        )
        if not valid_post_tick_state:
            self._record_hook_failure(
                SkillResultCode.INTERNAL_ERROR,
                "tick produced an invalid lifecycle/result transition",
            )
        return self._status

    @final
    def cancel(self) -> None:
        self._require_status(SkillStatus.RUNNING, "cancel")
        cleanup_errors: list[str] = []
        try:
            self._on_cancel()
        except Exception as exc:
            cleanup_errors.append(f"cancel hook: {exc}")
        if self._status is not SkillStatus.RUNNING:
            cleanup_errors.append("cancel hook attempted a lifecycle transition")
        if self._context is not None:
            try:
                self._context.uav.stop()
            except Exception as exc:
                cleanup_errors.append(f"UAV stop: {exc}")

        message = "Skill canceled"
        data: dict[str, object] = {}
        if cleanup_errors:
            message += "; cleanup reported errors"
            data["cleanup_errors"] = tuple(cleanup_errors)
        # Cancellation is a guaranteed RUNNING -> CANCELED transition. Cleanup
        # failures are diagnostic data and cannot silently turn it into FAILED.
        result = SkillResult(
            status=SkillStatus.CANCELED,
            code=SkillResultCode.CANCELED,
            message=message,
            data=_copy_data(data),
        )
        self._result, self._status = result, SkillStatus.CANCELED

    @final
    def get_feedback(self) -> SkillFeedback:
        return SkillFeedback(
            progress=self._feedback.progress,
            message=self._feedback.message,
            data=_copy_data(self._feedback.data),
        )

    @final
    def get_result(self) -> SkillResult | None:
        if self._result is None:
            return None
        return SkillResult(
            status=self._result.status,
            code=self._result.code,
            message=self._result.message,
            data=_copy_data(self._result.data),
        )

    @final
    def reset(self) -> None:
        if self._status not in {
            SkillStatus.SUCCEEDED,
            SkillStatus.FAILED,
            SkillStatus.CANCELED,
        }:
            raise SkillLifecycleError(f"reset is illegal while Skill is {self._status.name}")
        cleanup_error: Exception | None = None
        try:
            self._on_reset()
        except Exception as exc:
            cleanup_error = exc
        finally:
            self._status = SkillStatus.IDLE
            self._goal = None
            self._context = None
            self._initial_yaw = None
            self._motion_policy = MotionPolicy()
            self._feedback = SkillFeedback()
            self._result = None
        if cleanup_error is not None:
            raise SkillLifecycleError(
                f"Skill reset cleanup failed after returning to IDLE: {cleanup_error}"
            ) from cleanup_error

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        """Optional subclass initialization hook."""

    @abstractmethod
    def _on_tick(self, observation: Observation) -> None:
        """Advance behavior once; call ``_succeed`` or ``_fail`` when done."""

    def _on_cancel(self) -> None:
        """Optional subclass cancellation hook; base completion stops the UAV."""

    def _on_reset(self) -> None:
        """Optional subclass cleanup hook."""

    def _validate_goal(self, goal: SkillGoal) -> None:
        """Optional typed validation hook, executed while status is RUNNING."""

    def _set_feedback(
        self,
        progress: float | None,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        self._require_status(SkillStatus.RUNNING, "set feedback")
        normalized_progress: float | None = None
        if progress is not None:
            if (
                isinstance(progress, bool)
                or not isinstance(progress, Real)
                or not isfinite(progress)
                or not 0.0 <= float(progress) <= 1.0
            ):
                raise ValueError("feedback progress must be within [0, 1]")
            normalized_progress = float(progress)
        self._feedback = SkillFeedback(
            progress=normalized_progress,
            message=str(message),
            data=_copy_data(data),
        )

    def _apply_motion_policy(self, velocity_xyz_mps: Sequence[float]) -> np.ndarray:
        """Apply this run's policy, including the yaw captured by ``start``."""

        self._require_status(SkillStatus.RUNNING, "apply motion policy")
        return apply_motion_policy(
            self._active_context.uav,
            velocity_xyz_mps,
            self._motion_policy,
            initial_yaw=self._initial_yaw,
        )

    def _move_toward_with_motion_policy(
        self,
        goal_xyz_m: Sequence[float],
        speed_mps: float,
        tolerance_m: float,
    ) -> np.ndarray:
        """Queue bounded target motion using this run's captured yaw policy."""

        self._require_status(SkillStatus.RUNNING, "move with motion policy")
        return move_toward_with_policy(
            self._active_context.uav,
            goal_xyz_m,
            speed_mps,
            tolerance_m,
            self._motion_policy,
            initial_yaw=self._initial_yaw,
        )

    def _succeed(
        self,
        code: SkillResultCode,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        self._complete(SkillStatus.SUCCEEDED, code, message, data)

    def _fail(
        self,
        code: SkillResultCode,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        self._complete(SkillStatus.FAILED, code, message, data)

    def _complete(
        self,
        status: SkillStatus,
        code: SkillResultCode,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        self._require_status(SkillStatus.RUNNING, "complete")
        if status not in {SkillStatus.SUCCEEDED, SkillStatus.FAILED}:
            raise ValueError("tick completion status must be SUCCEEDED or FAILED")
        if not isinstance(code, SkillResultCode):
            raise TypeError("code must be a SkillResultCode")
        validate_skill_result(status, code)
        # Copy before changing lifecycle state so an invalid payload is mapped by
        # start/tick to INTERNAL_ERROR instead of leaving a terminal Skill with
        # no result object.
        result_data = _copy_data(data)
        result_message = str(message)
        if self._context is not None:
            try:
                self._context.uav.stop()
            except Exception as exc:
                status = SkillStatus.FAILED
                code = SkillResultCode.INTERNAL_ERROR
                result_message = f"could not stop UAV while completing Skill: {exc}"
                result_data = {"cleanup_error": str(exc)}
        result = SkillResult(
            status=status,
            code=code,
            message=result_message,
            data=result_data,
        )
        self._result, self._status = result, status

    def _record_hook_failure(self, code: SkillResultCode, message: str) -> None:
        """Override any partial/hook-created terminal state with a safe failure."""

        if code not in {
            SkillResultCode.INVALID_GOAL,
            SkillResultCode.INVALID_STATE,
            SkillResultCode.INTERNAL_ERROR,
        }:
            raise ValueError("hook failures require a validation/state/internal code")
        result_message = str(message)
        result_data: dict[str, object] = {}
        if self._context is not None:
            try:
                self._context.uav.stop()
            except Exception as exc:
                code = SkillResultCode.INTERNAL_ERROR
                result_message = f"{result_message}; could not stop UAV: {exc}"
                result_data["cleanup_error"] = str(exc)
        result = SkillResult(
            status=SkillStatus.FAILED,
            code=code,
            message=result_message,
            data=result_data,
        )
        self._result, self._status = result, SkillStatus.FAILED

    def _require_status(self, expected: SkillStatus, operation: str) -> None:
        if self._status is not expected:
            raise SkillLifecycleError(
                f"{operation} requires {expected.name}, current status is {self._status.name}"
            )

    @property
    def _active_goal(self) -> SkillGoal:
        if self._goal is None:
            raise SkillLifecycleError("Skill has no active goal")
        return self._goal

    @property
    def _active_context(self) -> SkillContext:
        if self._context is None:
            raise SkillLifecycleError("Skill has no active context")
        return self._context


def require_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise SkillGoalValidationError(f"{name} must be finite")
    return float(value)


def require_positive(value: float, name: str) -> float:
    result = require_finite(value, name)
    if result <= 0.0:
        raise SkillGoalValidationError(f"{name} must be greater than 0")
    return result


def require_vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (SequenceABC, np.ndarray))
    ):
        raise SkillGoalValidationError(f"{name} must contain three finite values")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise SkillGoalValidationError(f"{name} must contain three finite values") from exc
    if len(items) != 3 or any(
        isinstance(item, bool) or not isinstance(item, Real) or not isfinite(item)
        for item in items
    ):
        raise SkillGoalValidationError(f"{name} must contain three finite values")
    return float(items[0]), float(items[1]), float(items[2])


def _copy_data(data: dict[str, object] | None) -> dict[str, object]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("Skill data must be a dict")
    return deepcopy(data)
