"""High-level mission schemas plus scripted and text-only LLM planners."""

from planner.base import MissionPlanner, PlannerError, PlannerOutputError
from planner.dynamic_llm_planner import DynamicLLMPlanner, PlanningContract
from planner.diagnostics import PlannerDiagnostics, PlannerExecution
from planner.json_schema import (
    build_skill_plan_draft_json_schema,
    build_skill_plan_v2_json_schema,
)
from planner.json_schema_v3 import (
    build_skill_plan_draft_v3_json_schema,
    build_skill_plan_v3_json_schema,
    region_spec_json_schema,
    search_strategy_json_schema,
    spatial_target_json_schema,
)
from planner.llm_planner import LLMPlanner
from planner.prompt_builder import (
    build_dynamic_skill_planner_messages,
    build_mission_planner_messages,
    build_spatial_v3_skill_planner_messages,
)
from planner.policy import PlannerLimits, PlannerPolicy, TargetLostAction
from planner.critic_protocol import RouteCriticProtocol
from planner.mission_program import (
    MissionEdge,
    MissionNode,
    MissionProgram,
    MissionProgramError,
    ProgramAction,
    ProgramActionOp,
    ProgramEvent,
    ProgramEventHandler,
    SpatialEntity,
    linear_plan_to_mission_program,
)
from planner.mission_program_schema import build_mission_program_json_schema
from planner.qwen_next_best_view import (
    NextBestViewProposalRecord,
    NextBestViewRouting,
    QwenNextBestViewProvider,
    build_next_best_view_json_schema,
)
from planner.program_patch import ProgramPatch, apply_program_patch
from planner.program_patch_planner import (
    ProgramPatchPlannerError,
    ProgramPatchRequest,
    QwenProgramPatchPlanner,
)
from planner.program_patch_schema import build_program_patch_json_schema
from planner.route_critic import (
    RouteCritic,
    RouteCriticStatus,
    RouteCritique,
    RouteValidationContext,
    RouteValidationMode,
    RouteViolation,
    RouteViolationType,
)
from planner.route_types import (
    AvoidanceStrategy,
    AvoidanceStrategyType,
    RouteConstraints,
    RouteContractError,
    RouteDraft,
    RouteState,
    RouteWaypoint,
)
from planner.revision import (
    PlanRevisionDraft,
    PlanRevisionRequest,
    QwenPlanRevisionPlanner,
    RevisionErrorCode,
    RevisionLimits,
    RevisionValidationError,
    RevisionValidator,
    ValidatedPlanRevision,
    apply_plan_revision_atomically,
    build_plan_revision_json_schema,
    replace_plan_suffix,
)
from planner.schemas import (
    CompiledMission,
    LandingZoneSpec,
    MissionIntent,
    NavigationPointSpec,
    PlannerOutput,
    PlannerRequest,
    PlannerWorldContext,
    PlanStepDraft,
    PlanStepDraftV2,
    RecoveryDraft,
    SearchRegionSpec,
    SkillPlanDraft,
    SkillPlanDraftV2,
    migrate_plan_v1_to_v2,
)
from planner.schemas_v3 import (
    PlanStepDraftV3,
    SkillPlanDraftV3,
    SpatialPlanValidationError,
)
from planner.spatial import (
    CircleRegion,
    CoordinateFrame,
    CorridorRegion,
    NamedLocationTarget,
    PointTarget,
    PolygonRegion,
    RectangleAnchor,
    RectangleRegion,
    RegionSpec,
    RelationalPointTarget,
    RelationalRegion,
    RouteTarget,
    SectorRegion,
    SpatialAssumption,
    SpatialContractError,
    SpatialRelation,
    SpatialTarget,
    region_spec_from_dict,
    spatial_target_from_dict,
)
from planner.spatial_resolver import (
    FramePose,
    MissingFramePoseError,
    SpatialResolutionError,
    SpatialResolver,
    UnresolvedSpatialReferenceError,
)
from planner.region_compiler import (
    CompiledSearchGeometry,
    RegionCompilationError,
    RegionCompiler,
)
from planner.scripted_dynamic_planner import ScriptedDynamicPlanner
from planner.scripted_planner import ScriptedPlanner
from planner.skill_catalog import (
    SkillArgumentSpec,
    SkillCatalog,
    SkillContract,
    build_default_skill_catalog,
    build_spatial_v3_skill_catalog,
    initial_planner_catalog,
    revision_planner_catalog,
)
from planner.symbolic_checker import (
    PlanIssue,
    PlanIssueCode,
    SymbolicCheckResult,
    SymbolicPlanChecker,
)

__all__ = [
    "CompiledMission",
    "DynamicLLMPlanner",
    "PlanningContract",
    "LandingZoneSpec",
    "LLMPlanner",
    "MissionIntent",
    "MissionEdge",
    "MissionNode",
    "MissionProgram",
    "MissionProgramError",
    "MissionPlanner",
    "NavigationPointSpec",
    "PlannerError",
    "PlannerDiagnostics",
    "PlannerExecution",
    "PlannerOutput",
    "PlannerOutputError",
    "PlannerLimits",
    "PlannerPolicy",
    "PlannerRequest",
    "PlannerWorldContext",
    "PlanStepDraft",
    "ProgramAction",
    "ProgramActionOp",
    "ProgramEvent",
    "ProgramEventHandler",
    "ProgramPatch",
    "ProgramPatchPlannerError",
    "ProgramPatchRequest",
    "QwenProgramPatchPlanner",
    "PlanStepDraftV2",
    "PlanRevisionDraft",
    "PlanRevisionRequest",
    "QwenPlanRevisionPlanner",
    "RecoveryDraft",
    "RevisionErrorCode",
    "RevisionLimits",
    "RevisionValidationError",
    "RevisionValidator",
    "RouteCriticProtocol",
    "RouteCritic",
    "RouteCriticStatus",
    "RouteCritique",
    "RouteValidationContext",
    "RouteValidationMode",
    "RouteViolation",
    "RouteViolationType",
    "RouteConstraints",
    "RouteContractError",
    "RouteDraft",
    "RouteState",
    "RouteWaypoint",
    "AvoidanceStrategy",
    "AvoidanceStrategyType",
    "ValidatedPlanRevision",
    "ScriptedDynamicPlanner",
    "ScriptedPlanner",
    "SearchRegionSpec",
    "SkillArgumentSpec",
    "SkillCatalog",
    "SkillContract",
    "SkillPlanDraft",
    "SkillPlanDraftV2",
    "PlanIssue",
    "PlanIssueCode",
    "SymbolicCheckResult",
    "SymbolicPlanChecker",
    "TargetLostAction",
    "build_default_skill_catalog",
    "initial_planner_catalog",
    "revision_planner_catalog",
    "build_dynamic_skill_planner_messages",
    "build_mission_planner_messages",
    "build_spatial_v3_skill_planner_messages",
    "build_skill_plan_draft_json_schema",
    "build_skill_plan_v2_json_schema",
    "build_plan_revision_json_schema",
    "build_program_patch_json_schema",
    "apply_plan_revision_atomically",
    "migrate_plan_v1_to_v2",
    "replace_plan_suffix",
    "CircleRegion",
    "CompiledSearchGeometry",
    "CoordinateFrame",
    "CorridorRegion",
    "FramePose",
    "MissingFramePoseError",
    "NamedLocationTarget",
    "PlanStepDraftV3",
    "PointTarget",
    "PolygonRegion",
    "RectangleAnchor",
    "RectangleRegion",
    "RegionCompilationError",
    "RegionCompiler",
    "RegionSpec",
    "RelationalPointTarget",
    "RelationalRegion",
    "RouteTarget",
    "SectorRegion",
    "SkillPlanDraftV3",
    "SpatialAssumption",
    "SpatialContractError",
    "SpatialPlanValidationError",
    "SpatialRelation",
    "SpatialResolutionError",
    "SpatialResolver",
    "SpatialTarget",
    "SpatialEntity",
    "UnresolvedSpatialReferenceError",
    "build_skill_plan_draft_v3_json_schema",
    "build_mission_program_json_schema",
    "NextBestViewProposalRecord",
    "NextBestViewRouting",
    "QwenNextBestViewProvider",
    "build_next_best_view_json_schema",
    "build_obstacle_route_revision_schema",
    "build_skill_plan_v3_json_schema",
    "build_spatial_v3_skill_catalog",
    "region_spec_from_dict",
    "region_spec_json_schema",
    "search_strategy_json_schema",
    "spatial_target_from_dict",
    "spatial_target_json_schema",
    "apply_program_patch",
    "linear_plan_to_mission_program",
    "GroundedObstacleGeometry",
    "ObstacleAwareRevisionPlanner",
    "ObstacleAwareRevisionRequest",
    "ObstacleParseRepairFeedback",
    "ObstacleReplacementStep",
    "ObstacleReplacementCompilationError",
    "ObstacleReplacementCompiler",
    "ObstacleRevisionAttempt",
    "ObstacleRevisionError",
    "ObstacleRevisionSession",
    "ObstacleRevisionSessionState",
    "ObstacleRouteRevisionDraft",
    "is_repairable_obstacle_parse_error",
    "TrustedFollowRouteDefaults",
    "compile_obstacle_replacement",
]


_LAZY_OBSTACLE_REVISION_EXPORTS = frozenset(
    {
        "GroundedObstacleGeometry",
        "ObstacleAwareRevisionPlanner",
        "ObstacleAwareRevisionRequest",
        "ObstacleParseRepairFeedback",
        "ObstacleReplacementStep",
        "ObstacleRevisionAttempt",
        "ObstacleRevisionError",
        "ObstacleRevisionSession",
        "ObstacleRevisionSessionState",
        "ObstacleRouteRevisionDraft",
        "build_obstacle_route_revision_schema",
        "is_repairable_obstacle_parse_error",
    }
)

_LAZY_OBSTACLE_REPLACEMENT_COMPILER_EXPORTS = frozenset(
    {
        "ObstacleReplacementCompilationError",
        "ObstacleReplacementCompiler",
        "TrustedFollowRouteDefaults",
        "compile_obstacle_replacement",
    }
)


def __getattr__(name: str) -> object:
    """Avoid pulling perception/runtime packages into foundational imports."""

    if name in _LAZY_OBSTACLE_REVISION_EXPORTS:
        from planner import obstacle_revision as module

        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _LAZY_OBSTACLE_REPLACEMENT_COMPILER_EXPORTS:
        from planner import obstacle_replacement_compiler as module

        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
