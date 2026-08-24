"""Pure-Python runtime validation, compilation, and safety supervision."""

from runtime.collision_supervisor import (
    CollisionAction,
    CollisionDecision,
    CollisionState,
    CollisionSupervisor,
    CollisionSupervisorAction,
    CollisionSupervisorDecision,
    CollisionSupervisorState,
)

from runtime.events import (
    EventBus,
    EventSeverity,
    MissionEvent,
    MissionEventBus,
    MissionEventType,
)
from runtime.frame_store import BoundedFrameStore, FrameRef, FrameStore
from runtime.hazard_fusion import (
    HazardFusion,
    HazardFusionResult,
    HazardReport,
    HazardSource,
)
from runtime.plan_validator import PlanValidationError, PlannerLimits, PlanValidator
from runtime.obstacle_runtime import (
    ObstacleHazardRuntime,
    ObstacleRuntimeError,
    ObstacleRuntimeSnapshot,
)
from runtime.program_executor import (
    ProgramEventDispatch,
    ProgramExecutor,
    ProgramExecutorSnapshot,
)
from runtime.route_registry import RouteRecord, RouteRegistry, RouteRegistryError
from runtime.route_collision_monitor import (
    ROUTE_COLLISION_SOURCE,
    RouteCollision,
    RouteCollisionMonitor,
    RouteCollisionMonitorError,
)
from runtime.review_scheduler import (
    DEFAULT_BLOCKING_EVENT_TYPES,
    DEFAULT_EVENT_TRIGGERS,
    DEFAULT_REVIEW_INTERVALS_S,
    ReviewScheduleDecision,
    ReviewScheduleReason,
    ReviewScheduler,
    ReviewTicket,
    ReviewTrigger,
)
from runtime.safety_supervisor import SafetyAction, SafetyDecision, SafetySupervisor
from runtime.validation_codes import ValidationCode
from runtime.validation_report import (
    RecoveryRecommendation,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
)
from runtime.world_belief import (
    CandidateSummary,
    is_privileged_oracle_source,
    QwenRequestState,
    QwenRequestStatus,
    WorldBelief,
    WorldBeliefSnapshot,
    WorldBeliefStore,
    WorldBeliefThreadError,
)
from runtime.world_context_builder import (
    LANDING_ZONE_NAME,
    SEARCH_REGION_NAME,
    WorldContextBuildError,
    build_planner_world_context,
)

__all__ = [
    "BoundedFrameStore",
    "CandidateSummary",
    "CollisionAction",
    "CollisionDecision",
    "CollisionState",
    "CollisionSupervisor",
    "CollisionSupervisorAction",
    "CollisionSupervisorDecision",
    "CollisionSupervisorState",
    "is_privileged_oracle_source",
    "DEFAULT_BLOCKING_EVENT_TYPES",
    "DEFAULT_EVENT_TRIGGERS",
    "DEFAULT_REVIEW_INTERVALS_S",
    "EventBus",
    "EventSeverity",
    "FrameRef",
    "FrameStore",
    "HazardFusion",
    "HazardFusionResult",
    "HazardReport",
    "HazardSource",
    "MissionEvent",
    "MissionEventBus",
    "MissionEventType",
    "ObstacleHazardRuntime",
    "ObstacleRuntimeError",
    "ObstacleRuntimeSnapshot",
    "build_obstacle_revision_request",
    "grounded_runtime_assessment",
    "hold_relative_obstacle_geometry",
    "hold_relative_point_target",
    "PlanValidationError",
    "PlannerLimits",
    "PlanValidator",
    "ProgramExecutor",
    "ProgramExecutorSnapshot",
    "ProgramEventDispatch",
    "QwenRequestState",
    "QwenRequestStatus",
    "ReviewScheduleDecision",
    "ReviewScheduleReason",
    "ReviewScheduler",
    "ReviewTicket",
    "ReviewTrigger",
    "RouteRecord",
    "ROUTE_COLLISION_SOURCE",
    "RouteCollision",
    "RouteCollisionMonitor",
    "RouteCollisionMonitorError",
    "RouteRegistry",
    "RouteRegistryError",
    "SafetyAction",
    "SafetyDecision",
    "SafetySupervisor",
    "RecoveryRecommendation",
    "ValidationCode",
    "ValidationFinding",
    "ValidationReport",
    "ValidationSeverity",
    "WorldBelief",
    "WorldBeliefSnapshot",
    "WorldBeliefStore",
    "WorldBeliefThreadError",
    "LANDING_ZONE_NAME",
    "SEARCH_REGION_NAME",
    "WorldContextBuildError",
    "build_planner_world_context",
]


_LAZY_OBSTACLE_REVISION_CONTEXT_EXPORTS = frozenset(
    {
        "build_obstacle_revision_request",
        "grounded_runtime_assessment",
        "hold_relative_obstacle_geometry",
        "hold_relative_point_target",
    }
)


def __getattr__(name: str) -> object:
    """Keep the obstacle model contract out of foundational runtime imports."""

    if name in _LAZY_OBSTACLE_REVISION_CONTEXT_EXPORTS:
        from runtime import obstacle_revision_context as module

        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
