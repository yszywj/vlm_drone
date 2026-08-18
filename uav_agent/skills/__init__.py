"""Callable navigation, search, confirmation, and tracking Skill API."""

from skills.base import (
    Skill,
    SkillExecutionStateError,
    SkillGoalValidationError,
    SkillLifecycleError,
)
from skills.goto import GotoGoal, GotoSkill
from skills.hover import HoverGoal, HoverMode, HoverSkill, HoverTimeoutFallback
from skills.land import LandGoal, LandSkill
from skills.manager import (
    ExecutionKind,
    SkillManager,
    SkillManagerError,
    SkillNotRegisteredError,
    TaskStatus,
    TransitionRecord,
    create_default_skill_registry,
)
from skills.motion_types import (
    MotionPolicy,
    MotionPolicyValidationError,
    YawMode,
    apply_motion_policy,
    move_toward_with_policy,
)
from skills.reacquire import ReacquireGoal, ReacquireSkill
from skills.plan import (
    RecoveryPolicy,
    StepOutputRef,
    TaskPlan,
    TaskPlanError,
    TaskStep,
)
from skills.search import SearchGoal, SearchPhase, SearchSkill
from skills.takeoff import TakeoffGoal, TakeoffSkill
from skills.track import TrackGoal, TrackSkill
from skills.types import (
    CameraSensor,
    Observation,
    SkillClock,
    SkillContext,
    SkillFeedback,
    SkillExecutionReport,
    SkillGoal,
    SkillInvocation,
    SkillName,
    SkillResult,
    SkillResultCode,
    SkillStatus,
    UAVController,
)

__all__ = [
    "CameraSensor",
    "ExecutionKind",
    "GotoGoal",
    "GotoSkill",
    "HoverGoal",
    "HoverMode",
    "HoverSkill",
    "HoverTimeoutFallback",
    "InspectApproachPolicy",
    "InspectGoal",
    "InspectPhase",
    "InspectPolicy",
    "InspectSkill",
    "InspectionEvidenceHandle",
    "LandGoal",
    "LandSkill",
    "MotionPolicy",
    "MotionPolicyValidationError",
    "Observation",
    "ReacquireGoal",
    "ReacquireSkill",
    "RecoveryPolicy",
    "SearchGoal",
    "SearchPhase",
    "SearchSkill",
    "Skill",
    "SkillClock",
    "SkillContext",
    "SkillExecutionStateError",
    "SkillExecutionReport",
    "SkillFeedback",
    "SkillGoal",
    "SkillInvocation",
    "SkillGoalValidationError",
    "SkillLifecycleError",
    "SkillManager",
    "SkillManagerError",
    "SkillName",
    "SkillNotRegisteredError",
    "SkillResult",
    "SkillResultCode",
    "SkillStatus",
    "StepOutputRef",
    "TakeoffGoal",
    "TakeoffSkill",
    "TaskPlan",
    "TaskPlanError",
    "TaskStatus",
    "TaskStep",
    "TrackGoal",
    "TrackSkill",
    "TransitionRecord",
    "UAVController",
    "YawMode",
    "apply_motion_policy",
    "create_default_skill_registry",
    "move_toward_with_policy",
]


_LAZY_INSPECT_EXPORTS = frozenset(
    {
        "InspectApproachPolicy",
        "InspectGoal",
        "InspectPhase",
        "InspectPolicy",
        "InspectSkill",
        "InspectionEvidenceHandle",
    }
)


def __getattr__(name: str) -> object:
    """Load INSPECT's perception-heavy symbols without import cycles."""

    if name in _LAZY_INSPECT_EXPORTS:
        from skills import inspect as inspect_module

        value = getattr(inspect_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
