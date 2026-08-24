#!/usr/bin/env python3
"""Run one prevalidated multi-UAV Fleet mission in Isaac Sim.

Fleet decomposition, every local Spatial V3 model call, strict compilation,
target-perception preflight, and Safety preflight all finish before this module
performs its first Isaac import.  Isaac runtime execution then replays the
request-bound plans through the normal FleetMissionRuntime and MissionAgent
validation boundaries without calling either planner a second time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from math import cos, dist, radians, sin
from pathlib import Path
import re
import sys
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Every import in this section is pure Python.  In particular, importing this
# entry point for argument or planning tests must never import ``isaacsim``.
from configs.loader import load_config  # noqa: E402
from common.ids import generate_routing_id, validate_routing_id  # noqa: E402
from experiments.fleet_report import generate_fleet_report  # noqa: E402
from experiments.fleet_result_recorder import FleetResultRecorder  # noqa: E402
from experiments.planning_audit_logger import (  # noqa: E402
    PlanningAuditLogger,
    prompt_sha256,
    sanitize_text_tail,
)
from experiments.run_manager import RunManager  # noqa: E402
from experiments.schemas import (  # noqa: E402
    AgentMetricRecord,
    GoalResultRecord,
    PlanningAttemptRecord,
    RecoveryActionRecord,
    StateSampleRecord,
    ValidationFindingRecord,
)
from experiments.terminal_logger import TerminalLogger  # noqa: E402
from fleet.compiler import FleetAssignmentCompiler  # noqa: E402
from fleet.llm_planner import LLMFleetPlanner  # noqa: E402
from fleet.llm_planner_v2 import LLMFleetPlannerV2  # noqa: E402
from fleet.llm_task_interpreter import LLMFleetTaskInterpreter  # noqa: E402
from fleet.logging import FleetMissionLogger  # noqa: E402
from fleet.local_spatial_planner import (  # noqa: E402
    RoutedPreplannedSpatialPlanner,
    ScriptedAssignmentSpatialPlanner,
)
from fleet.preplanned_planner import RoutedPreplannedFleetPlanner  # noqa: E402
from fleet.request_builder import (  # noqa: E402
    FleetRequestBuildError,
    build_agent_world_contexts,
    build_agent_world_contexts_v2,
    build_fleet_mission_request,
    build_fleet_mission_request_v2,
    build_target_catalog,
    parse_explicit_assignment_instruction,
)
from fleet.schemas import validate_fleet_mission_plan  # noqa: E402
from fleet.scripted_planner import ScriptedFleetPlanner  # noqa: E402
from fleet.types import (  # noqa: E402
    AssignmentCompilation,
    FleetAssignment,
    FleetMissionPlan,
    FleetMissionRequest,
    FleetTargetRequest,
)
from fleet.schemas_v2 import validate_fleet_mission_plan_v2  # noqa: E402
from fleet.task_spec import (  # noqa: E402
    AssignmentConstraint,
    ConstraintStrength,
    FleetTaskSpecV1,
    GoalType,
    MissionGoal,
)
from fleet.types_v2 import (  # noqa: E402
    FleetAssignmentV2,
    FleetStateEvidenceType,
    FleetMissionPlanV2,
    FleetMissionRequestV2,
    TrustedFleetStateEvidence,
)
from fleet.runtime import (  # noqa: E402
    AssignmentStatus,
    FleetReplanPublication,
    FleetStatus,
    ReplannedAssignment,
)
from models.adapter_registry import (  # noqa: E402
    AdapterRegistry,
    AdapterSelection,
    DEFAULT_ADAPTER_CONFIG,
    ModelCallRole,
)
from models.model_client_factory import ModelClientFactory  # noqa: E402
from perception.factory import (  # noqa: E402
    TargetPerceptionConfigurationError,
    validate_target_perception_preflight,
)
from planner.dynamic_llm_planner import DynamicLLMPlanner  # noqa: E402
from planner.policy import PlannerLimits, PlannerPolicy  # noqa: E402
from planner.schemas import PlannerWorldContext  # noqa: E402
from planner.spatial import CircleRegion, CoordinateFrame, RegionSpec  # noqa: E402
from runtime.plan_validator import PlanValidator  # noqa: E402
from runtime.safety_supervisor import SafetyAction, SafetySupervisor  # noqa: E402
from target.types import TargetSpec  # noqa: E402


DEFAULT_INSTRUCTION = (
    "无人机A前往世界坐标二十、三十附近十五米范围搜索并跟踪目标i二十秒；"
    "无人机B前往世界坐标负二十五、十附近十二米范围搜索并跟踪目标j二十秒；"
    "完成后分别返回各自起点降落"
)


class FleetLaunchConfigurationError(ValueError):
    """Raised before Isaac startup when a launch is incomplete or unsafe."""


_LOCAL_PROPOSAL_REPAIR_MESSAGES: Mapping[str, str] = MappingProxyType(
    {
        "INVALID_JSON": (
            "Emit one complete strict JSON object matching the response schema."
        ),
        "SCHEMA_INVALID": (
            "Correct all required fields, value types, and trusted routing values "
            "to match the response schema."
        ),
        "CATALOG_CONTRACT_VIOLATION": (
            "Use only top-level Skills and arguments exposed by the active catalog."
        ),
        "V3_CONTRACT_VIOLATION": (
            "Correct Skill ordering, references, call limits, and Spatial V3 "
            "contracts."
        ),
        "MODEL_CLIENT_ERROR": (
            "Return one complete schema-valid proposal for this bounded retry."
        ),
        "ROUTING_IDS_REQUIRED": (
            "Echo every trusted mission, UAV, and plan-version routing value exactly."
        ),
        "UNKNOWN_FIELD": "Remove fields not present in the response schema.",
        "UNKNOWN_SKILL": "Use only Skills exposed by the active catalog.",
        "UNKNOWN_ENTITY": "Use only entities present in the trusted request.",
        "NON_FINITE_NUMBER": "Replace non-finite values with finite bounded numbers.",
        "ROUTING_MISMATCH": "Echo every trusted routing value exactly.",
        "PLAN_VERSION_MISMATCH": "Echo the trusted plan_version exactly.",
        "STEP_REFERENCE_INVALID": (
            "Repair cross-step references so they point backward to compatible outputs."
        ),
        "CALL_LIMIT_EXCEEDED": "Reduce Skill calls to the advertised bounded limits.",
        "LOW_LEVEL_CONTROL_FORBIDDEN": (
            "Remove low-level control commands and use high-level Skills only."
        ),
        "ORACLE_FIELD_FORBIDDEN": (
            "Remove privileged ground-truth fields and use only trusted request data."
        ),
        "OUT_OF_BOUNDS_GOTO": "Move every destination inside the trusted scene bounds.",
        "INVALID_LANDING_ZONE": "Use the supplied own_home landing zone exactly.",
        "UNSAFE_ACTION": "Replace the rejected action with a preflight-safe Skill plan.",
        "INTERNAL_ERROR": "Return one complete schema-valid bounded Skill plan.",
    }
)


def _local_proposal_repair_findings(
    *,
    planner: object | None = None,
    validation_findings: Sequence[object] = (),
) -> tuple[Mapping[str, object], ...]:
    """Return bounded structural feedback without exception/model contents."""

    result: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for finding in validation_findings:
        severity = getattr(
            getattr(finding, "severity", None),
            "value",
            getattr(finding, "severity", None),
        )
        if severity != "HARD_ACTION_BLOCK":
            continue
        code = getattr(
            getattr(finding, "code", None),
            "value",
            getattr(finding, "code", None),
        )
        if not isinstance(code, str) or code not in _LOCAL_PROPOSAL_REPAIR_MESSAGES:
            code = "SCHEMA_INVALID"
        if code in seen:
            continue
        seen.add(code)
        result.append(
            {"code": code, "message": _LOCAL_PROPOSAL_REPAIR_MESSAGES[code]}
        )
        if len(result) >= 32:
            return tuple(result)
    diagnostics = getattr(planner, "last_diagnostics", None)
    diagnostic_code = getattr(diagnostics, "initial_error_code", None)
    if not result:
        code = (
            diagnostic_code
            if isinstance(diagnostic_code, str)
            and diagnostic_code in _LOCAL_PROPOSAL_REPAIR_MESSAGES
            else "SCHEMA_INVALID"
        )
        result.append(
            {"code": code, "message": _LOCAL_PROPOSAL_REPAIR_MESSAGES[code]}
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RuntimeEnvelopeMetadata:
    """Semantic routing metadata kept outside the target-bound V1 envelope.

    A targetless V2 Assignment still needs a structurally valid V1 value while
    the Isaac runtime is migrated away from its legacy target-bound contract.
    The compatibility alias is therefore explicitly quarantined here.  It is
    never an environment assignment, perception target, TargetRegistry claim,
    or semantic/metric target.
    """

    non_target_assignment_ids: frozenset[str] = frozenset()
    semantic_target_by_assignment: Mapping[str, str | None] = field(
        default_factory=dict
    )
    required_by_assignment: Mapping[str, bool] = field(default_factory=dict)
    compatibility_anchor_by_assignment: Mapping[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        non_target = frozenset(
            validate_routing_id(value, "non_target_assignment_id")
            for value in self.non_target_assignment_ids
        )
        semantic = {
            validate_routing_id(key, "assignment_id"): (
                None
                if value is None
                else validate_routing_id(value, "semantic_target_alias")
            )
            for key, value in self.semantic_target_by_assignment.items()
        }
        required = {
            validate_routing_id(key, "assignment_id"): value
            for key, value in self.required_by_assignment.items()
        }
        anchors = {
            validate_routing_id(key, "assignment_id"): validate_routing_id(
                value, "compatibility_anchor_alias"
            )
            for key, value in self.compatibility_anchor_by_assignment.items()
        }
        if set(anchors) != set(non_target):
            raise ValueError(
                "compatibility anchors must exactly cover non-target assignments"
            )
        if set(semantic) != set(required):
            raise ValueError(
                "semantic target and requiredness metadata must cover the same assignments"
            )
        if not set(non_target) <= set(semantic):
            raise ValueError("non-target assignments are missing semantic metadata")
        for assignment_id, target_alias in semantic.items():
            if assignment_id in non_target:
                if target_alias is not None:
                    raise ValueError(
                        "non-target assignment cannot declare a semantic target"
                    )
            elif target_alias is None:
                raise ValueError(
                    "target-bound assignment must declare its semantic target"
                )
        if any(not isinstance(value, bool) for value in required.values()):
            raise TypeError("assignment requiredness values must be bool")
        object.__setattr__(self, "non_target_assignment_ids", non_target)
        object.__setattr__(
            self, "semantic_target_by_assignment", MappingProxyType(semantic)
        )
        object.__setattr__(
            self, "required_by_assignment", MappingProxyType(required)
        )
        object.__setattr__(
            self, "compatibility_anchor_by_assignment", MappingProxyType(anchors)
        )


@dataclass(frozen=True, slots=True)
class PreparedFleetMission:
    """Pure-Python result that is safe to hand across the Isaac boundary."""

    config: object
    request: FleetMissionRequest
    plan: FleetMissionPlan
    world_contexts: Mapping[str, PlannerWorldContext]
    compilations: Mapping[str, AssignmentCompilation]
    planning_failures: Mapping[str, str]
    initial_assignment_states: Mapping[str, str]
    compiler: FleetAssignmentCompiler
    planner_limits: PlannerLimits
    planner_policy: PlannerPolicy
    fleet_planner_source: str
    local_planner_source: str
    adapter_selections: tuple[Mapping[str, object], ...]
    model_call_records: tuple[Mapping[str, object], ...]
    visual_clients: Mapping[str, object]
    runtime_visual_selection: AdapterSelection | None
    planned_routes: Mapping[str, tuple[tuple[float, float, float], ...]]
    headless: bool
    vision_review_mode: str
    task_spec: FleetTaskSpecV1 | None = None
    fleet_request_v2: FleetMissionRequestV2 | None = None
    fleet_plan_v2: FleetMissionPlanV2 | None = None
    mission_interpreter_source: str = "scripted_fixed_parser"
    mission_interpreter_diagnostics: Mapping[str, object] | None = None
    mission_interpreter_proposals: tuple[Mapping[str, object], ...] = ()
    fleet_planner_proposals: tuple[Mapping[str, object], ...] = ()
    fleet_semantic_findings: tuple[Mapping[str, object], ...] = ()
    preparation_context: dict[str, object] | None = None
    local_planner_proposals: Mapping[
        str, tuple[Mapping[str, object], ...]
    ] = field(default_factory=dict)
    model_call_records_persisted_live: bool = False
    model_client_factory: object | None = None
    runtime_envelope_metadata: RuntimeEnvelopeMetadata = field(
        default_factory=RuntimeEnvelopeMetadata
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "world_contexts",
            MappingProxyType(dict(self.world_contexts)),
        )
        object.__setattr__(
            self,
            "compilations",
            MappingProxyType(dict(self.compilations)),
        )
        object.__setattr__(
            self,
            "planning_failures",
            MappingProxyType(dict(self.planning_failures)),
        )
        object.__setattr__(
            self,
            "initial_assignment_states",
            MappingProxyType(dict(self.initial_assignment_states)),
        )
        object.__setattr__(
            self,
            "visual_clients",
            MappingProxyType(dict(self.visual_clients)),
        )
        object.__setattr__(
            self,
            "planned_routes",
            MappingProxyType(
                {
                    uav_id: tuple(tuple(point) for point in route)
                    for uav_id, route in self.planned_routes.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "adapter_selections",
            tuple(MappingProxyType(dict(item)) for item in self.adapter_selections),
        )
        object.__setattr__(
            self,
            "model_call_records",
            tuple(MappingProxyType(dict(item)) for item in self.model_call_records),
        )
        if self.task_spec is not None and not isinstance(
            self.task_spec, FleetTaskSpecV1
        ):
            raise TypeError("task_spec must be FleetTaskSpecV1 or None")
        if self.fleet_request_v2 is not None and not isinstance(
            self.fleet_request_v2, FleetMissionRequestV2
        ):
            raise TypeError("fleet_request_v2 must be FleetMissionRequestV2 or None")
        if self.fleet_plan_v2 is not None and not isinstance(
            self.fleet_plan_v2, FleetMissionPlanV2
        ):
            raise TypeError("fleet_plan_v2 must be FleetMissionPlanV2 or None")
        if not isinstance(self.mission_interpreter_source, str) or not (
            self.mission_interpreter_source.strip()
        ):
            raise ValueError("mission_interpreter_source must be non-empty")
        if self.mission_interpreter_diagnostics is not None:
            object.__setattr__(
                self,
                "mission_interpreter_diagnostics",
                MappingProxyType(dict(self.mission_interpreter_diagnostics)),
            )
        for field_name in (
            "mission_interpreter_proposals",
            "fleet_planner_proposals",
            "fleet_semantic_findings",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(
                    MappingProxyType(dict(item))
                    for item in getattr(self, field_name)
                ),
            )
        object.__setattr__(
            self,
            "local_planner_proposals",
            MappingProxyType(
                {
                    str(uav_id): tuple(
                        MappingProxyType(dict(item)) for item in proposals
                    )
                    for uav_id, proposals in self.local_planner_proposals.items()
                }
            ),
        )
        if not isinstance(self.model_call_records_persisted_live, bool):
            raise TypeError("model_call_records_persisted_live must be bool")
        if self.model_client_factory is not None and not callable(
            getattr(self.model_client_factory, "for_role", None)
        ):
            raise TypeError("model_client_factory must provide for_role() or be None")
        if not isinstance(self.runtime_envelope_metadata, RuntimeEnvelopeMetadata):
            raise TypeError(
                "runtime_envelope_metadata must be RuntimeEnvelopeMetadata"
            )
        metadata_ids = set(
            self.runtime_envelope_metadata.semantic_target_by_assignment
        )
        if metadata_ids and metadata_ids != {
            assignment.assignment_id for assignment in self.plan.assignments
        }:
            raise ValueError(
                "runtime envelope metadata must exactly cover plan assignments"
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs/multi_uav_demo.yaml",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--fleet-planner",
        choices=("scripted", "llm"),
        default="scripted",
    )
    parser.add_argument(
        "--mission-interpreter",
        choices=("scripted", "llm"),
        default=None,
        help=(
            "semantic interpreter; defaults to llm with --fleet-planner llm "
            "and scripted with --fleet-planner scripted"
        ),
    )
    parser.add_argument(
        "--local-planner",
        choices=("dynamic_scripted", "dynamic_llm"),
        default=None,
        help=(
            "per-UAV planner; defaults to dynamic_llm for an LLM Fleet and "
            "dynamic_scripted for the scripted baseline"
        ),
    )
    parser.add_argument("--planning-contract", choices=("v3",), default="v3")
    parser.add_argument(
        "--runtime-program",
        choices=("linear", "graph"),
        default="linear",
    )
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=DEFAULT_ADAPTER_CONFIG,
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--enable-qwen-vision", action="store_true")
    parser.add_argument(
        "--vision-review-mode",
        choices=("shadow", "gate"),
        default=None,
    )
    parser.add_argument(
        "--perception-runtime-profile",
        choices=("production", "oracle_evaluation"),
        default="production",
    )
    parser.add_argument(
        "--acknowledge-privileged-oracle",
        action="store_true",
    )
    parser.add_argument("--debug-visualization", action="store_true")
    headless = parser.add_mutually_exclusive_group()
    headless.add_argument("--headless", dest="headless", action="store_true")
    headless.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=None)
    parser.add_argument("--max-sim-time", type=float, default=300.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_PROJECT_ROOT / "logs/fleet_missions",
    )
    parser.add_argument("--no-summary-figures", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def _resolved_mission_interpreter(args: argparse.Namespace) -> str:
    selected = args.mission_interpreter or args.fleet_planner
    if selected != args.fleet_planner:
        raise FleetLaunchConfigurationError(
            "--mission-interpreter and --fleet-planner must both be scripted "
            "or both be llm; cross-version fallback is forbidden"
        )
    return str(selected)


def _mission_alias_catalogs(
    config: object,
) -> tuple[dict[str, str], dict[str, str]]:
    """Expose only trusted semantic aliases to the Interpreter."""

    uav_aliases: dict[str, str] = {}
    for uav in config.uavs:
        uav_aliases[uav.id] = uav.id
        if uav.display_name:
            uav_aliases[uav.display_name] = uav.id
    target_aliases: dict[str, str] = {}
    for target in config.targets:
        target_aliases[target.id] = target.id
        if target.semantic_alias:
            target_aliases[target.semantic_alias] = target.id
    return uav_aliases, target_aliases


def _semantic_issue_dict(value: object) -> dict[str, object]:
    return {
        "code": getattr(value, "code", type(value).__name__),
        "message": str(getattr(value, "message", value))[:2048],
        "constraint_id": getattr(value, "constraint_id", None),
        "goal_id": getattr(value, "goal_id", None),
        "assignment_id": getattr(value, "assignment_id", None),
    }


def _lower_v2_target_plan_for_runtime(
    config: object,
    request: FleetMissionRequestV2,
    plan: FleetMissionPlanV2,
    *,
    include_metadata: bool = False,
) -> (
    tuple[FleetMissionRequest, FleetMissionPlan]
    | tuple[FleetMissionRequest, FleetMissionPlan, RuntimeEnvelopeMetadata]
):
    """Create an explicit v1 execution envelope for the current Isaac backend.

    The semantic and persisted plan remains V2.  The current Isaac environment
    is target-bound.  Assignments with exactly one semantic target retain that
    target in the envelope.  A strictly bounded targetless subset uses a
    synthetic structural alias that is quarantined by RuntimeEnvelopeMetadata;
    it never becomes target/perception authority.  Multi-target and unsupported
    targetless Assignments fail closed.
    """

    if not isinstance(include_metadata, bool):
        raise TypeError("include_metadata must be bool")
    validate_fleet_mission_plan_v2(plan, request)
    target_catalog = build_target_catalog(config)
    configured_uavs = {item.id: item for item in config.uavs}
    runtime_targets: list[FleetTargetRequest] = []
    runtime_assignments: list[FleetAssignment] = []
    non_target_assignment_ids: set[str] = set()
    semantic_target_by_assignment: dict[str, str | None] = {}
    required_by_assignment: dict[str, bool] = {}
    compatibility_anchor_by_assignment: dict[str, str] = {}
    targetless_goal_types = frozenset(
        {
            GoalType.NAVIGATE,
            GoalType.WAIT,
            GoalType.LAND,
            GoalType.RETURN_HOME,
            GoalType.RETURN_HOME_AND_LAND,
        }
    )
    for assignment in plan.assignments:
        goals = tuple(request.task_spec.goal(item) for item in assignment.goal_ids)
        target_aliases = {
            goal.target_alias
            for goal in goals
            if isinstance(goal, MissionGoal) and goal.target_alias is not None
        }
        if len(target_aliases) > 1:
            raise FleetLaunchConfigurationError(
                "the current Isaac compatibility envelope cannot safely lower "
                "multiple semantic targets in one V2 Assignment "
                f"({assignment.assignment_id})"
            )
        semantic_target: str | None
        if not target_aliases:
            unsupported = sorted(
                {
                    goal.goal_type.value
                    for goal in goals
                    if goal.goal_type not in targetless_goal_types
                }
            )
            if unsupported:
                raise FleetLaunchConfigurationError(
                    "targetless V2 Assignment is not executable for GoalType(s): "
                    + ", ".join(unsupported)
                    + f" ({assignment.assignment_id})"
                )
            semantic_target = None
            digest = sha256(
                (
                    request.fleet_mission_id
                    + ":"
                    + assignment.assignment_id
                ).encode("utf-8")
            ).hexdigest()[:16]
            target_alias = f"compat_non_target_{digest}"
            target_spec = TargetSpec(
                "runtime-only non-target compatibility envelope",
                category="runtime_compatibility_non_target",
                immutable_identity_summary=(
                    "not a semantic target; no observation or claim authority"
                ),
            )
            non_target_assignment_ids.add(assignment.assignment_id)
            compatibility_anchor_by_assignment[
                assignment.assignment_id
            ] = target_alias
        else:
            semantic_target = next(iter(target_aliases))
            target_alias = semantic_target
            try:
                target_spec = target_catalog[target_alias]
            except KeyError:
                raise FleetLaunchConfigurationError(
                    "V2 runtime envelope references an unknown configured "
                    f"target: {target_alias}"
                ) from None
        try:
            configured_uav = configured_uavs[assignment.uav_id]
        except KeyError as exc:
            raise FleetLaunchConfigurationError(
                f"V2 runtime envelope references an unknown configured entity: {exc.args[0]}"
            ) from None
        regions = tuple(
            goal.spatial_constraint
            for goal in goals
            if (
                isinstance(goal, MissionGoal)
                and goal.goal_type is GoalType.SEARCH_TARGET
                and goal.spatial_constraint is not None
                and getattr(goal.spatial_constraint, "shape", None) is not None
            )
        )
        if len(regions) > 1 and any(region != regions[0] for region in regions[1:]):
            raise FleetLaunchConfigurationError(
                f"Assignment {assignment.assignment_id} has incompatible SEARCH regions"
            )
        if regions:
            search_region = regions[0]
        else:
            # Runtime metadata only: a Local plan with no SEARCH never consumes
            # this conservative launch-centred region.
            x, y, _ = configured_uav.initial_position_xyz_m
            radius = min(
                float(config.search.radius_m),
                float(config.scene.size_xyz_m[0]) / 4.0,
                float(config.scene.size_xyz_m[1]) / 4.0,
            )
            search_region = CircleRegion(
                CoordinateFrame.WORLD_ENU,
                (float(x), float(y), 0.0),
                max(0.5, radius),
            )
        track_durations = tuple(
            float(goal.duration_s)
            for goal in goals
            if (
                isinstance(goal, MissionGoal)
                and goal.goal_type is GoalType.TRACK_TARGET
                and goal.duration_s is not None
            )
        )
        track_duration_s = (
            max(track_durations)
            if track_durations
            else max(1.0, float(config.planner.min_track_duration_s))
        )
        required = any(goal.strength is ConstraintStrength.MUST for goal in goals)
        semantic_target_by_assignment[assignment.assignment_id] = semantic_target
        required_by_assignment[assignment.assignment_id] = required
        runtime_targets.append(
            FleetTargetRequest(
                target_alias=target_alias,
                target_spec=target_spec,
                requested_uav_id=assignment.uav_id,
                search_region=search_region,
                track_duration_s=track_duration_s,
                priority=assignment.priority,
                start_policy=assignment.start_policy,
                required=required,
            )
        )
        runtime_assignments.append(
            FleetAssignment(
                assignment_id=assignment.assignment_id,
                uav_id=assignment.uav_id,
                target_alias=target_alias,
                target_spec=target_spec,
                search_region=search_region,
                track_duration_s=track_duration_s,
                priority=assignment.priority,
                start_policy=assignment.start_policy,
            )
        )
    if not runtime_targets:
        raise FleetLaunchConfigurationError(
            "Fleet Planner returned no target-bound executable assignments"
        )
    runtime_request = FleetMissionRequest(
        fleet_mission_id=request.fleet_mission_id,
        fleet_plan_version=request.fleet_plan_version,
        original_instruction=request.task_spec.source_text,
        uav_inventory=request.uav_inventory,
        target_requests=tuple(runtime_targets),
        coordination_policy=request.coordination_policy,
        assumptions=(
            (
                "runtime-only compatibility aliases carry no semantic target, "
                "perception, Oracle, claim, or target-metric authority"
            ),
        )
        if non_target_assignment_ids
        else (),
    )
    runtime_plan = FleetMissionPlan(
        fleet_mission_id=plan.fleet_mission_id,
        fleet_plan_version=plan.fleet_plan_version,
        assignments=tuple(runtime_assignments),
        coordination_policy=plan.coordination_policy,
        assumptions=plan.assumptions,
        unassigned_requirements=plan.unassigned_goal_ids,
    )
    validated_plan = validate_fleet_mission_plan(runtime_plan, runtime_request)
    if not include_metadata:
        return runtime_request, validated_plan
    return (
        runtime_request,
        validated_plan,
        RuntimeEnvelopeMetadata(
            non_target_assignment_ids=frozenset(non_target_assignment_ids),
            semantic_target_by_assignment=semantic_target_by_assignment,
            required_by_assignment=required_by_assignment,
            compatibility_anchor_by_assignment=compatibility_anchor_by_assignment,
        ),
    )


def _subset_task_spec_for_replan(
    task_spec: FleetTaskSpecV1,
    goal_ids: Sequence[str],
) -> FleetTaskSpecV1:
    """Retain exactly the unfinished Goals and their still-applicable constraints."""

    selected = tuple(dict.fromkeys(str(value) for value in goal_ids))
    known = set(task_spec.all_goal_ids)
    if not selected or set(selected) - known:
        raise FleetLaunchConfigurationError(
            "Fleet replan requires a non-empty set of known unfinished Goals"
        )
    selected_set = set(selected)
    assignment_constraints = tuple(
        replace(
            constraint,
            goal_ids=tuple(
                goal_id
                for goal_id in constraint.goal_ids
                if goal_id in selected_set
            ),
        )
        for constraint in task_spec.assignment_constraints
        if selected_set.intersection(constraint.goal_ids)
    )
    return replace(
        task_spec,
        goals=tuple(
            goal for goal in task_spec.goals if goal.goal_id in selected_set
        ),
        termination_goals=tuple(
            goal
            for goal in task_spec.termination_goals
            if goal.goal_id in selected_set
        ),
        assignment_constraints=assignment_constraints,
        ordering_constraints=tuple(
            constraint
            for constraint in task_spec.ordering_constraints
            if constraint.before_goal_id in selected_set
            and constraint.after_goal_id in selected_set
        ),
        ambiguities=tuple(
            ambiguity
            for ambiguity in task_spec.ambiguities
            if any(goal_id in ambiguity.field_path for goal_id in selected_set)
        ),
    )


def _build_runtime_replan_request_v2(
    prepared: PreparedFleetMission,
    *,
    goal_ids: Sequence[str],
    available_uav_ids: Sequence[str],
    world_belief: object,
) -> FleetMissionRequestV2:
    if prepared.task_spec is None or prepared.fleet_request_v2 is None:
        raise FleetLaunchConfigurationError(
            "V2 Fleet replan requires the original TaskSpec and Fleet request"
        )
    available = frozenset(str(value) for value in available_uav_ids)
    if not available:
        raise FleetLaunchConfigurationError(
            "Fleet replan has no trusted available UAV"
        )
    base_version = getattr(world_belief, "fleet_plan_version", None)
    if isinstance(base_version, bool) or not isinstance(base_version, int):
        raise FleetLaunchConfigurationError(
            "Fleet replan world belief lacks a trusted plan version"
        )
    raw_agents = getattr(world_belief, "agents", {})
    agents = raw_agents if isinstance(raw_agents, Mapping) else {}
    evidence: list[TrustedFleetStateEvidence] = []
    for capability in prepared.fleet_request_v2.uav_inventory:
        if capability.uav_id in available:
            continue
        summary = agents.get(capability.uav_id)
        status = str(getattr(summary, "status", "UNAVAILABLE"))
        evidence.append(
            TrustedFleetStateEvidence(
                evidence_id=f"evidence_{capability.uav_id}_unavailable",
                evidence_type=(
                    FleetStateEvidenceType.UAV_UNAVAILABLE
                    if status
                    in {
                        "FAILED",
                        "CANCELED",
                        "WAITING_REASSIGNMENT",
                        "REASSIGNMENT_REQUIRED",
                    }
                    else FleetStateEvidenceType.CURRENT_ASSIGNMENT_CONFLICT
                ),
                summary=(
                    f"{capability.uav_id} is not available for reassignment; "
                    f"trusted runtime status={status}"
                ),
                uav_id=capability.uav_id,
            )
        )
    return replace(
        prepared.fleet_request_v2,
        fleet_plan_version=base_version + 1,
        task_spec=_subset_task_spec_for_replan(prepared.task_spec, goal_ids),
        uav_inventory=tuple(
            replace(item, available=item.uav_id in available)
            for item in prepared.fleet_request_v2.uav_inventory
        ),
        trusted_fleet_state=tuple(evidence),
    )


def _build_runtime_fleet_replan_handler(
    prepared: PreparedFleetMission,
    *,
    audit: PlanningAuditLogger,
    agent_factory: Callable[
        [
            object,
            FleetAssignmentV2,
            object,
            PlannerWorldContext,
            FleetAssignment,
            tuple[tuple[float, float, float], ...],
        ],
        ReplannedAssignment,
    ],
    fleet_planner_factory: Callable[[object], object] | None = None,
    local_planner_factory: Callable[[object, str], object] | None = None,
    compiler_factory: Callable[[Mapping[str, object]], object] | None = None,
) -> Callable[[object, object], FleetReplanPublication | None] | None:
    """Build the bounded production V2 Fleet-reassignment boundary.

    The returned handler performs model planning and compilation but never
    executes a controller action.  A replacement is returned only after the
    Fleet proposal, exact unfinished-Goal coverage, local compiler report, and
    Safety preflight have all passed.  Runtime owns the final atomic publish.
    """

    if (
        prepared.task_spec is None
        or prepared.fleet_request_v2 is None
        or prepared.fleet_plan_v2 is None
        or prepared.model_client_factory is None
        or prepared.local_planner_source != "dynamic_llm"
    ):
        return None
    if not callable(agent_factory):
        raise TypeError("agent_factory must be callable")
    client_factory = prepared.model_client_factory
    fleet_planner_factory = fleet_planner_factory or (
        lambda client: LLMFleetPlannerV2(client, repair_budget=2)
    )
    local_planner_factory = local_planner_factory or (
        lambda client, _uav_id: DynamicLLMPlanner(
            client,
            system_prompt_path=(
                _PROJECT_ROOT / "prompts/dynamic_skill_planner_v3_system.txt"
            ),
            planner_limits=prepared.planner_limits,
            planner_policy=prepared.planner_policy,
            planning_contract="v3",
            repair_budget=0,
        )
    )
    compiler_factory = compiler_factory or (
        lambda planners: FleetAssignmentCompiler(
            planners,
            validator=PlanValidator(
                prepared.planner_limits,
                prepared.planner_policy,
            ),
        )
    )
    for name, factory in (
        ("fleet_planner_factory", fleet_planner_factory),
        ("local_planner_factory", local_planner_factory),
        ("compiler_factory", compiler_factory),
    ):
        if not callable(factory):
            raise TypeError(f"{name} must be callable")

    active_v2: dict[str, FleetAssignmentV2] = {
        assignment.assignment_id: assignment
        for assignment in prepared.fleet_plan_v2.assignments
    }
    attempted_sources: set[str] = set()
    runtime_context = prepared.preparation_context
    if runtime_context is None:
        runtime_context = {}
        object.__setattr__(prepared, "preparation_context", runtime_context)
    raw_history = runtime_context.setdefault("runtime_reassignments", [])
    if not isinstance(raw_history, list):
        raise TypeError("runtime_reassignments audit context must be a list")

    def safe_audit(operation: Callable[[], object]) -> None:
        try:
            operation()
        except Exception:
            # Result logging is observational and must not grant or revoke
            # flight authority.
            pass

    def handler(record: object, world_belief: object) -> FleetReplanPublication:
        source_runtime = getattr(record, "assignment", None)
        source_id = getattr(source_runtime, "assignment_id", None)
        if not isinstance(source_runtime, FleetAssignment) or not isinstance(
            source_id, str
        ):
            raise FleetLaunchConfigurationError(
                "Fleet replan received an invalid runtime assignment"
            )
        if source_id in attempted_sources:
            raise FleetLaunchConfigurationError(
                "Fleet replan budget exhausted for assignment " + source_id
            )
        attempted_sources.add(source_id)
        source_v2 = active_v2.get(source_id)
        if source_v2 is None:
            raise FleetLaunchConfigurationError(
                "Fleet replan lacks the source Goal mapping: " + source_id
            )
        raw_agents = getattr(world_belief, "agents", {})
        belief_agents = raw_agents if isinstance(raw_agents, Mapping) else {}
        occupied_uavs = {
            str(getattr(summary, "uav_id", uav_id))
            for uav_id, summary in belief_agents.items()
        }
        available_uavs = tuple(
            capability.uav_id
            for capability in prepared.fleet_request_v2.uav_inventory
            if capability.available and capability.uav_id not in occupied_uavs
        )
        request_v2 = _build_runtime_replan_request_v2(
            prepared,
            goal_ids=source_v2.goal_ids,
            available_uav_ids=available_uavs,
            world_belief=world_belief,
        )
        base_version = request_v2.fleet_plan_version - 1
        prompt_material = json.dumps(
            request_v2.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        planner = fleet_planner_factory(
            client_factory.for_role(
                ModelCallRole.FLEET_REPLAN,
                fleet_mission_id=request_v2.fleet_mission_id,
                assignment_id=source_id,
                uav_id=source_runtime.uav_id,
            )
        )
        try:
            proposed_plan = validate_fleet_mission_plan_v2(
                planner.plan(request_v2), request_v2
            )
        finally:
            proposals = tuple(getattr(planner, "model_proposals", ()))
            safe_audit(
                lambda: _record_proposal_stage(
                    audit,
                    mission_id=request_v2.fleet_mission_id,
                    stage="FLEET_REPLAN",
                    role=ModelCallRole.FLEET_REPLAN.value,
                    schema="fleet_mission_plan_v2",
                    prompt_material=prompt_material,
                    proposals=proposals,
                    assignment_id=source_id,
                    uav_id=source_runtime.uav_id,
                )
            )
        if proposed_plan.unassigned_goal_ids:
            raise FleetLaunchConfigurationError(
                "Fleet replan left unfinished Goals unassigned: "
                + ", ".join(proposed_plan.unassigned_goal_ids)
            )
        fleet_findings = proposed_plan.semantic_findings(request_v2)
        unexplained = tuple(
            item
            for item in fleet_findings
            if item.code != "EXPLAINED_ASSIGNMENT_DEVIATION"
        )
        for finding_index, finding in enumerate(fleet_findings):
            safe_audit(
                lambda finding=finding, finding_index=finding_index: audit.log_finding(
                    ValidationFindingRecord(
                        finding_id=(
                            f"finding_runtime_fleet_replan_{base_version}_"
                            f"{finding_index}"
                        ),
                        timestamp_s=float(base_version),
                        stage="FLEET_REPLAN",
                        scope="ASSIGNMENT",
                        severity="RECOVERABLE_SEMANTIC_ERROR",
                        code=finding.code,
                        message=finding.message,
                        mission_id=request_v2.fleet_mission_id,
                        assignment_id=finding.assignment_id,
                        uav_id=None,
                        goal_id=finding.goal_id,
                        evidence_refs=(),
                        recommended_action="REPAIR_FLEET_PLAN",
                    )
                )
            )
        if unexplained:
            raise FleetLaunchConfigurationError(
                "Fleet replan contains unexplained semantic deviations: "
                + ", ".join(item.code for item in unexplained)
            )
        if len(proposed_plan.assignments) != 1:
            raise FleetLaunchConfigurationError(
                "one failed runtime assignment requires exactly one replacement"
            )
        assignment_v2 = proposed_plan.assignments[0]
        if set(assignment_v2.goal_ids) != set(source_v2.goal_ids):
            raise FleetLaunchConfigurationError(
                "Fleet replan replacement changed unfinished Goal coverage"
            )
        if assignment_v2.uav_id not in available_uavs:
            raise FleetLaunchConfigurationError(
                "Fleet replan selected a non-idle UAV"
            )
        if assignment_v2.assignment_id == source_id:
            assignment_v2 = replace(
                assignment_v2,
                assignment_id=generate_routing_id("assignment_replan"),
            )
            proposed_plan = replace(
                proposed_plan,
                assignments=(assignment_v2,),
            )
            proposed_plan = validate_fleet_mission_plan_v2(
                proposed_plan, request_v2
            )

        contexts = build_agent_world_contexts_v2(
            prepared.config, request_v2, proposed_plan
        )
        uav_id = assignment_v2.uav_id
        local_planner = local_planner_factory(
            client_factory.for_role(
                ModelCallRole.AGENT_SPATIAL_PLAN,
                fleet_mission_id=request_v2.fleet_mission_id,
                assignment_id=assignment_v2.assignment_id,
                uav_id=uav_id,
            ),
            uav_id,
        )
        compiler = compiler_factory({uav_id: local_planner})
        capability = next(
            item for item in request_v2.uav_inventory if item.uav_id == uav_id
        )
        semantic_findings: tuple[Mapping[str, object], ...] = ()
        proposal_findings: tuple[Mapping[str, object], ...] = ()
        compilation = None
        local_proposals: list[Mapping[str, object]] = []
        last_local_error: Exception | None = None
        for local_attempt in range(3):
            try:
                compilation = compiler.compile_assignment_v2(
                    request_v2,
                    proposed_plan,
                    assignment_v2,
                    contexts[uav_id],
                    local_plan_version=(
                        getattr(record, "local_plan_version", 1)
                        + local_attempt
                        + 1
                    ),
                    target_catalog=build_target_catalog(prepared.config),
                    spatial_resolver=_spatial_resolver(
                        contexts[uav_id], capability.home_name
                    ),
                    proposal_id=(
                        f"proposal_fleet_replan_local_{uav_id}_{local_attempt}"
                    ),
                    semantic_repair_findings=semantic_findings,
                    proposal_repair_findings=proposal_findings,
                )
            except Exception as exc:
                last_local_error = exc
                local_proposals.extend(
                    dict(item)
                    for item in getattr(local_planner, "model_proposals", ())
                )
                if local_attempt >= 2:
                    raise FleetLaunchConfigurationError(
                        "Fleet replan local repair budget exhausted: "
                        + f"{type(exc).__name__}: {exc}"
                    ) from None
                # Structural/parser failures authorize a fresh bounded model
                # call, but are not mislabeled as semantic Goal findings.
                semantic_findings = ()
                proposal_findings = _local_proposal_repair_findings(
                    planner=local_planner
                )
                continue
            local_proposals.extend(
                dict(item)
                for item in getattr(local_planner, "model_proposals", ())
            )
            if not compilation.executable:
                findings = tuple(compilation.validation_report.findings)
                last_local_error = FleetLaunchConfigurationError(
                    "Fleet replan local compilation is hard-blocked"
                )
                if local_attempt >= 2:
                    raise FleetLaunchConfigurationError(
                        "Fleet replan local hard-block repair budget exhausted"
                    )
                semantic_findings = ()
                proposal_findings = _local_proposal_repair_findings(
                    planner=local_planner,
                    validation_findings=findings,
                )
                continue
            if compilation.semantically_valid:
                break
            if local_attempt >= 2:
                raise FleetLaunchConfigurationError(
                    "Fleet replan local semantic repair budget exhausted"
                )
            semantic_findings = tuple(
                {
                    "code": getattr(item.code, "value", str(item.code)),
                    "goal_id": item.goal_id,
                    "message": item.message[:512],
                }
                for item in compilation.goal_coverage.findings
                if getattr(item.severity, "value", item.severity)
                == "RECOVERABLE_SEMANTIC_ERROR"
            )[:32]
            proposal_findings = ()
            last_local_error = FleetLaunchConfigurationError(
                "Fleet replan local semantic coverage requires repair"
            )
        if compilation is None:
            assert last_local_error is not None
            raise FleetLaunchConfigurationError(str(last_local_error))
        assert compilation is not None and compilation.compiled_mission is not None
        safety = SafetySupervisor(
            contexts[uav_id].scene_min_xyz_m,
            contexts[uav_id].scene_max_xyz_m,
            max_mission_time_s=86_520.0,
            position_margin_m=0.25,
            max_safe_altitude_m=contexts[uav_id].scene_max_xyz_m[2],
            planner_limits=prepared.planner_limits,
        )
        decision = safety.preflight(compilation.compiled_mission)
        if decision.action is not SafetyAction.CONTINUE:
            raise FleetLaunchConfigurationError(
                "Fleet replan Safety preflight rejected replacement: "
                + decision.reason
            )

        safe_audit(
            lambda: _record_proposal_stage(
                audit,
                mission_id=request_v2.fleet_mission_id,
                stage="LOCAL_REPLAN",
                role=ModelCallRole.AGENT_SPATIAL_PLAN.value,
                schema="spatial_skill_plan_v3",
                prompt_material=compilation.planner_request.instruction,
                proposals=tuple(local_proposals),
                assignment_id=assignment_v2.assignment_id,
                uav_id=uav_id,
            )
        )
        safe_audit(
            lambda: audit.write_final_plan(
                stage="FLEET_REPLAN",
                mission_id=request_v2.fleet_mission_id,
                plan_version=request_v2.fleet_plan_version,
                plan=proposed_plan.to_dict(),
            )
        )
        safe_audit(
            lambda: audit.write_final_plan(
                stage="LOCAL_REPLAN",
                mission_id=request_v2.fleet_mission_id,
                assignment_id=assignment_v2.assignment_id,
                uav_id=uav_id,
                plan_version=compilation.agent_request.local_plan_version,
                plan=compilation.planner_output.to_dict(),
            )
        )
        for finding in compilation.validation_report.findings:
            safe_audit(
                lambda finding=finding: audit.log_finding(
                    ValidationFindingRecord(
                        finding_id=finding.finding_id,
                        timestamp_s=float(finding.timestamp),
                        stage=finding.stage,
                        scope=finding.scope,
                        severity=finding.severity.value,
                        code=finding.code.value,
                        message=finding.message,
                        mission_id=finding.mission_id,
                        assignment_id=finding.assignment_id,
                        uav_id=finding.uav_id,
                        goal_id=finding.goal_id,
                        step_id=finding.step_id,
                        proposal_id=finding.proposal_id,
                        evidence_refs=finding.evidence_refs,
                        recommended_action=finding.recommended_action.value,
                    )
                )
            )

        runtime_assignment = replace(
            source_runtime,
            assignment_id=assignment_v2.assignment_id,
            uav_id=uav_id,
        )
        route = _build_planned_routes(
            contexts,
            {uav_id: compilation},
        )[uav_id]
        replacement = agent_factory(
            record,
            assignment_v2,
            compilation,
            contexts[uav_id],
            runtime_assignment,
            route,
        )
        if (
            not isinstance(replacement, ReplannedAssignment)
            or replacement.assignment_id != source_id
            or replacement.replacement_assignment != runtime_assignment
        ):
            raise FleetLaunchConfigurationError(
                "agent_factory returned an invalid Fleet replan handoff"
            )
        active_v2.pop(source_id, None)
        active_v2[runtime_assignment.assignment_id] = assignment_v2
        if len(raw_history) < 64:
            raw_history.append(
                {
                    "schema_version": 1,
                    "base_fleet_plan_version": base_version,
                    "new_fleet_plan_version": request_v2.fleet_plan_version,
                    "source_assignment_id": source_id,
                    "replacement_assignment_id": runtime_assignment.assignment_id,
                    "uav_id": uav_id,
                    "goal_ids": list(assignment_v2.goal_ids),
                    "semantically_valid": compilation.semantically_valid,
                }
            )
        safe_audit(
            lambda: audit.log_recovery(
                RecoveryActionRecord(
                    recovery_action_id=(
                        f"recovery_fleet_replan_{source_id}_{base_version}"
                    ),
                    timestamp_s=float(base_version),
                    mission_id=request_v2.fleet_mission_id,
                    stage="FLEET_REPLAN",
                    action="REASSIGN_GOALS",
                    outcome="PUBLISHED_FOR_RUNTIME_VALIDATION",
                    assignment_id=source_id,
                    uav_id=uav_id,
                    resulting_plan_version=request_v2.fleet_plan_version,
                )
            )
        )
        return FleetReplanPublication(
            base_fleet_plan_version=base_version,
            new_fleet_plan_version=request_v2.fleet_plan_version,
            replacements=(replacement,),
        )

    def audited_handler(
        record: object,
        world_belief: object,
    ) -> FleetReplanPublication:
        try:
            return handler(record, world_belief)
        except Exception:
            assignment = getattr(record, "assignment", None)
            assignment_id = getattr(assignment, "assignment_id", None)
            uav_id = getattr(assignment, "uav_id", None)
            version = getattr(world_belief, "fleet_plan_version", 0)
            if not isinstance(version, int) or isinstance(version, bool):
                version = 0
            if isinstance(assignment_id, str):
                safe_audit(
                    lambda: audit.log_recovery(
                        RecoveryActionRecord(
                            recovery_action_id=(
                                f"recovery_fleet_replan_failed_{assignment_id}_"
                                f"{max(0, version)}"
                            ),
                            timestamp_s=float(max(0, version)),
                            mission_id=prepared.request.fleet_mission_id,
                            stage="FLEET_REPLAN",
                            action="REASSIGN_GOALS",
                            outcome="REJECTED_FAIL_CLOSED",
                            assignment_id=assignment_id,
                            uav_id=uav_id if isinstance(uav_id, str) else None,
                            resulting_plan_version=None,
                        )
                    )
                )
            raise

    return audited_handler


def prepare_fleet_mission(args: argparse.Namespace) -> PreparedFleetMission:
    """Plan, validate, compile, and preflight without importing Isaac."""

    if not isinstance(args.instruction, str) or not args.instruction.strip():
        raise FleetLaunchConfigurationError("--instruction must not be empty")
    if not isinstance(args.max_sim_time, (int, float)) or isinstance(
        args.max_sim_time, bool
    ):
        raise FleetLaunchConfigurationError("--max-sim-time must be a number")
    if not 0.0 < float(args.max_sim_time) <= 86_400.0:
        raise FleetLaunchConfigurationError(
            "--max-sim-time must be within (0, 86400] seconds"
        )

    config = load_config(args.config)
    resolved_headless = (
        bool(config.simulation.headless)
        if args.headless is None
        else bool(args.headless)
    )
    if args.debug_visualization and resolved_headless:
        raise FleetLaunchConfigurationError(
            "--debug-visualization requires --no-headless"
        )
    is_oracle = args.perception_runtime_profile == "oracle_evaluation"
    if is_oracle and not args.acknowledge_privileged_oracle:
        raise FleetLaunchConfigurationError(
            "oracle_evaluation requires --acknowledge-privileged-oracle"
        )
    if not is_oracle and args.acknowledge_privileged_oracle:
        raise FleetLaunchConfigurationError(
            "Oracle acknowledgement is forbidden in production"
        )
    review_mode = args.vision_review_mode or config.qwen_visual_review.mode
    if review_mode not in {"shadow", "gate"}:
        raise FleetLaunchConfigurationError(
            "vision review mode must be shadow or gate"
        )

    registry = AdapterRegistry(args.adapter_config)
    if args.model is not None and args.model != registry.base_model_name:
        raise FleetLaunchConfigurationError(
            "--model must match adapters.json base_model.served_model_name"
        )
    selections: list[dict[str, object]] = []
    model_call_records: list[dict[str, object]] = []
    live_logger_holder: list[FleetMissionLogger | None] = [None]

    def record_model_call(value: Mapping[str, object]) -> None:
        """Retain the bounded row and, when managed, persist it immediately.

        The immediate write is what makes a failed first Interpreter call
        auditable.  Logging is observational: a storage/logging defect is
        recorded for the terminal summary but cannot broaden model authority.
        """

        row = dict(value)
        model_call_records.append(row)
        live_logger = live_logger_holder[0]
        if live_logger is None:
            return
        try:
            live_logger.log_model_call(row)
        except Exception as exc:
            setattr(
                args,
                "_preparation_logging_error",
                _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
            )

    client_factory = ModelClientFactory(
        registry,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_s=config.model_worker.request_timeout_s,
        selection_logger=lambda value: selections.append(dict(value)),
        call_logger=record_model_call,
    )

    interpreter_mode = _resolved_mission_interpreter(args)
    local_planner_mode = args.local_planner or (
        "dynamic_llm" if args.fleet_planner == "llm" else "dynamic_scripted"
    )
    if (args.fleet_planner == "llm") != (local_planner_mode == "dynamic_llm"):
        raise FleetLaunchConfigurationError(
            "LLM Fleet planning requires independent dynamic_llm Local Planners; "
            "the fixed local skeleton is reserved for the scripted baseline"
        )
    task_spec: FleetTaskSpecV1 | None = None
    request_v2: FleetMissionRequestV2 | None = None
    plan_v2: FleetMissionPlanV2 | None = None
    interpreter_diagnostics: Mapping[str, object] | None = None
    interpreter_proposals: tuple[Mapping[str, object], ...] = ()
    fleet_planner_proposals: tuple[Mapping[str, object], ...] = ()
    fleet_semantic_findings: tuple[Mapping[str, object], ...] = ()
    preparation_context: dict[str, object] | None = None
    runtime_envelope_metadata: RuntimeEnvelopeMetadata | None = None

    if args.fleet_planner == "scripted":
        # The fixed grammar is intentionally confined to this deterministic
        # compatibility baseline.  The LLM branch below never calls it, even
        # after an Interpreter or Planner failure.
        directives = parse_explicit_assignment_instruction(
            args.instruction,
            config,
        )
        request = build_fleet_mission_request(
            config,
            args.instruction.strip(),
            directives=directives,
        )
        interpreter_selection = {
            **registry.resolve(ModelCallRole.MISSION_INTERPRETATION).to_dict(),
            "used": False,
        }
        selections.append(interpreter_selection)
        fleet_planner = ScriptedFleetPlanner()
        selection = {
            **registry.resolve(ModelCallRole.FLEET_PLAN).to_dict(),
            "used": False,
        }
        selections.append(selection)
        model_call_records.append(
            _not_called_model_record(
                selection,
                fleet_mission_id=request.fleet_mission_id,
                call_id="not_called_fleet_plan",
            )
        )
        plan = validate_fleet_mission_plan(fleet_planner.plan(request), request)
        required_by_alias = {
            item.target_alias: item.required for item in request.target_requests
        }
        runtime_envelope_metadata = RuntimeEnvelopeMetadata(
            semantic_target_by_assignment={
                assignment.assignment_id: assignment.target_alias
                for assignment in plan.assignments
            },
            required_by_assignment={
                assignment.assignment_id: required_by_alias.get(
                    assignment.target_alias, True
                )
                for assignment in plan.assignments
            },
        )
    else:
        if interpreter_mode != "llm":  # guarded by _resolved_, kept explicit
            raise FleetLaunchConfigurationError(
                "LLM Fleet planning requires the LLM Mission Interpreter"
            )
        fleet_mission_id = generate_routing_id("fleet_mission")
        setattr(args, "_preparation_fleet_mission_id", fleet_mission_id)
        preparation_context = {
            "fleet_mission_id": fleet_mission_id,
            "source_text": args.instruction.strip(),
            # Keep the same bounded metadata-only buffer alive after pure
            # planning.  Runtime FLEET_REPLAN/LOCAL_PLAN calls use the same
            # ModelClientFactory and append here, so terminal token/latency
            # metrics include every role instead of stopping at Isaac launch.
            "model_call_records": model_call_records,
            "interpreter_proposals": (),
            "fleet_planner_proposals": (),
            "local_planner_proposals": {},
            "compilations": {},
        }
        setattr(args, "_preparation_audit_context", preparation_context)
        managed_run_dir = getattr(args, "_managed_run_dir", None)
        if managed_run_dir is not None:
            live_logger_holder[0] = FleetMissionLogger.attach_run_dir(
                managed_run_dir,
                fleet_mission_id,
                uav_ids=tuple(item.id for item in config.uavs),
                max_record_bytes=config.results.max_record_bytes,
                max_stream_bytes=config.results.max_stream_bytes,
                max_run_bytes=config.results.max_run_bytes,
            )
            setattr(args, "_preparation_fleet_logger", live_logger_holder[0])
        uav_aliases, target_aliases = _mission_alias_catalogs(config)
        interpreter = LLMFleetTaskInterpreter(
            client_factory.for_role(
                ModelCallRole.MISSION_INTERPRETATION,
                fleet_mission_id=fleet_mission_id,
            ),
            uav_alias_catalog=uav_aliases,
            target_alias_catalog=target_aliases,
            repair_budget=1,
        )
        try:
            task_spec = interpreter.interpret(args.instruction.strip())
        except Exception:
            preparation_context["interpreter_proposals"] = (
                interpreter.model_proposals
            )
            diagnostics = interpreter.last_diagnostics
            preparation_context["interpreter_diagnostics"] = (
                None if diagnostics is None else diagnostics.to_dict()
            )
            raise
        diagnostics = interpreter.last_diagnostics
        interpreter_diagnostics = (
            None if diagnostics is None else diagnostics.to_dict()
        )
        interpreter_proposals = interpreter.model_proposals
        preparation_context.update(
            {
                "task_spec": task_spec,
                "interpreter_proposals": interpreter_proposals,
                "interpreter_diagnostics": interpreter_diagnostics,
            }
        )
        request_v2 = build_fleet_mission_request_v2(
            config,
            task_spec,
            fleet_mission_id=fleet_mission_id,
        )
        preparation_context["request_v2"] = request_v2
        fleet_planner = LLMFleetPlannerV2(
            client_factory.for_role(
                ModelCallRole.FLEET_PLAN,
                fleet_mission_id=fleet_mission_id,
            ),
            repair_budget=2,
        )
        try:
            plan_v2 = validate_fleet_mission_plan_v2(
                fleet_planner.plan(request_v2), request_v2
            )
        except Exception:
            preparation_context["fleet_planner_proposals"] = (
                fleet_planner.model_proposals
            )
            raise
        fleet_planner_proposals = fleet_planner.model_proposals
        fleet_semantic_findings = tuple(
            _semantic_issue_dict(item)
            for item in fleet_planner.last_semantic_findings
        )
        preparation_context.update(
            {
                "plan_v2": plan_v2,
                "fleet_planner_proposals": fleet_planner_proposals,
                "fleet_semantic_findings": fleet_semantic_findings,
            }
        )
        request, plan, runtime_envelope_metadata = _lower_v2_target_plan_for_runtime(
            config,
            request_v2,
            plan_v2,
            include_metadata=True,
        )
    assert runtime_envelope_metadata is not None
    if not plan.assignments:
        raise FleetLaunchConfigurationError(
            "Fleet Planner returned no executable assignments"
        )

    limits = PlannerLimits.from_config(config.planner)
    policy = PlannerPolicy.from_config(config.planner, limits)
    semantic_assignments = (
        plan.assignments if plan_v2 is None else plan_v2.assignments
    )
    contexts = (
        build_agent_world_contexts(config, plan)
        if request_v2 is None or plan_v2 is None
        else build_agent_world_contexts_v2(config, request_v2, plan_v2)
    )
    local_planners: dict[str, object] = {}
    if local_planner_mode == "dynamic_scripted":
        for assignment in semantic_assignments:
            local_planners[assignment.uav_id] = ScriptedAssignmentSpatialPlanner()
        selection = {
            **registry.resolve(ModelCallRole.AGENT_SPATIAL_PLAN).to_dict(),
            "used": False,
        }
        selections.append(selection)
        model_call_records.append(
            _not_called_model_record(
                selection,
                fleet_mission_id=request.fleet_mission_id,
                call_id="not_called_agent_spatial_plan",
            )
        )
    else:
        prompt = _PROJECT_ROOT / "prompts/dynamic_skill_planner_v3_system.txt"
        for assignment in semantic_assignments:
            local_planners[assignment.uav_id] = DynamicLLMPlanner(
                client_factory.for_role(
                    ModelCallRole.AGENT_SPATIAL_PLAN,
                    fleet_mission_id=request.fleet_mission_id,
                    assignment_id=assignment.assignment_id,
                    uav_id=assignment.uav_id,
                ),
                system_prompt_path=prompt,
                planner_limits=limits,
                planner_policy=policy,
                planning_contract="v3",
                # Fleet owns the two bounded retries and increments the
                # trusted local plan version for each one.  Disable the
                # planner-internal retry so the total remains initial + 2.
                repair_budget=0,
            )
    compiler = FleetAssignmentCompiler(
        local_planners,
        validator=PlanValidator(limits, policy),
    )
    compilations: dict[str, object] = {}
    planning_failures: dict[str, str] = {}
    initial_assignment_states: dict[str, str] = {}
    local_proposal_history: dict[str, list[Mapping[str, object]]] = {
        assignment.uav_id: [] for assignment in semantic_assignments
    }
    if preparation_context is not None:
        preparation_context["local_planner_proposals"] = local_proposal_history
        preparation_context["compilations"] = compilations

    # Every assignment owns its compile/repair budget.  A broken local proposal
    # cannot erase a different UAV's READY plan.  Dynamic LLM planning disables
    # planner-internal repair and uses exactly one initial call plus at most two
    # Fleet-owned focused retries; scripted planning remains single-attempt.
    max_local_repairs = 2 if local_planner_mode == "dynamic_llm" else 0
    for assignment in semantic_assignments:
        last_error: Exception | None = None
        semantic_repair_findings: tuple[Mapping[str, object], ...] = ()
        proposal_repair_findings: tuple[Mapping[str, object], ...] = ()
        for repair_attempt in range(max_local_repairs + 1):
            proposals_captured = False
            current_proposal_findings: tuple[Mapping[str, object], ...] = ()
            try:
                if request_v2 is not None and plan_v2 is not None:
                    result = compiler.compile_assignment_v2(
                        request_v2,
                        plan_v2,
                        assignment,
                        contexts[assignment.uav_id],
                        local_plan_version=repair_attempt + 1,
                        target_catalog=build_target_catalog(config),
                        semantic_repair_findings=semantic_repair_findings,
                        proposal_repair_findings=proposal_repair_findings,
                    )
                else:
                    result = compiler.compile_assignment(
                        request,
                        plan,
                        assignment,
                        contexts[assignment.uav_id],
                        local_plan_version=repair_attempt + 1,
                    )
                local_proposal_history[assignment.uav_id].extend(
                    dict(item)
                    for item in getattr(
                        local_planners[assignment.uav_id],
                        "model_proposals",
                        (),
                    )
                )
                proposals_captured = True
                compiled = result.compiled_mission
                if compiled is None:
                    current_proposal_findings = _local_proposal_repair_findings(
                        planner=local_planners[assignment.uav_id],
                        validation_findings=getattr(
                            getattr(result, "validation_report", None),
                            "findings",
                            (),
                        ),
                    )
                    raise FleetLaunchConfigurationError(
                        f"local plan for {assignment.uav_id} was not compiled"
                    )
                if (
                    not bool(getattr(result, "semantically_valid", True))
                    and repair_attempt < max_local_repairs
                ):
                    coverage = getattr(result, "goal_coverage", None)
                    findings = getattr(coverage, "findings", ())
                    semantic_repair_findings = tuple(
                        {
                            "code": getattr(item.code, "value", str(item.code)),
                            "goal_id": item.goal_id,
                            "message": item.message[:512],
                        }
                        for item in findings
                        if getattr(item.severity, "value", item.severity)
                        == "RECOVERABLE_SEMANTIC_ERROR"
                    )[:32]
                    proposal_repair_findings = ()
                    last_error = FleetLaunchConfigurationError(
                        "local semantic Goal coverage requires repair: "
                        + ", ".join(getattr(result, "uncovered_goal_ids", ()))
                    )
                    continue
                if not is_oracle:
                    validate_target_perception_preflight(
                        config.target_perception.backend,
                        (step.skill for step in compiled.task_plan.steps),
                    )
                context = contexts[assignment.uav_id]
                safety = SafetySupervisor(
                    context.scene_min_xyz_m,
                    context.scene_max_xyz_m,
                    max_mission_time_s=float(args.max_sim_time) + 120.0,
                    position_margin_m=0.25,
                    max_safe_altitude_m=context.scene_max_xyz_m[2],
                    planner_limits=limits,
                )
                decision = safety.preflight(compiled)
                if decision.action is not SafetyAction.CONTINUE:
                    current_proposal_findings = (
                        {
                            "code": "UNSAFE_ACTION",
                            "message": _LOCAL_PROPOSAL_REPAIR_MESSAGES[
                                "UNSAFE_ACTION"
                            ],
                        },
                    )
                    raise FleetLaunchConfigurationError(
                        f"Safety preflight rejected {assignment.uav_id}: "
                        f"{decision.reason}"
                    )
            except Exception as exc:
                if not proposals_captured:
                    local_proposal_history[assignment.uav_id].extend(
                        dict(item)
                        for item in getattr(
                            local_planners[assignment.uav_id],
                            "model_proposals",
                            (),
                        )
                    )
                if isinstance(exc, TargetPerceptionConfigurationError):
                    # This is a Fleet-wide launch configuration defect, not an
                    # assignment-local model proposal that another UAV can
                    # safely work around.
                    raise
                last_error = exc
                semantic_repair_findings = ()
                proposal_repair_findings = (
                    current_proposal_findings
                    or _local_proposal_repair_findings(
                        planner=local_planners[assignment.uav_id]
                    )
                )
                continue
            compilations[assignment.uav_id] = result
            if bool(getattr(result, "semantically_valid", True)):
                initial_assignment_states[assignment.assignment_id] = (
                    AssignmentStatus.READY.value
                )
            else:
                uncovered = tuple(getattr(result, "uncovered_goal_ids", ()))
                initial_assignment_states[assignment.assignment_id] = (
                    AssignmentStatus.DEGRADED_EXECUTABLE.value
                )
                planning_failures[assignment.assignment_id] = (
                    "recoverable semantic coverage exhausted; uncovered goals: "
                    + ", ".join(uncovered)
                )
            break
        else:
            assert last_error is not None
            planning_failures[assignment.assignment_id] = _redact_terminal_error(
                f"{type(last_error).__name__}: {last_error}"
            )
            initial_assignment_states[assignment.assignment_id] = (
                AssignmentStatus.REASSIGNMENT_REQUIRED.value
            )

    visual_clients: dict[str, object] = {}
    runtime_visual_selection: AdapterSelection | None = None
    if args.enable_qwen_vision:
        # Construct role-bound clients during pure-Python preparation so URL,
        # credential shape, Adapter routing and per-UAV isolation all fail
        # before the first Isaac import.  The clients do not own threads and
        # cannot execute until the runtime wraps them behind the shared Broker
        # dispatcher below.
        visual_factory = ModelClientFactory(
            registry,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=config.model_worker.request_timeout_s,
        )
        assignment_by_uav = {
            assignment.uav_id: assignment for assignment in plan.assignments
        }
        for capability in request.uav_inventory:
            assignment = assignment_by_uav.get(capability.uav_id)
            visual_clients[capability.uav_id] = visual_factory.for_role(
                ModelCallRole.RUNTIME_VISUAL_REVIEW,
                fleet_mission_id=request.fleet_mission_id,
                assignment_id=(
                    None if assignment is None else assignment.assignment_id
                ),
                uav_id=capability.uav_id,
            )
        runtime_visual_selection = registry.resolve(
            ModelCallRole.RUNTIME_VISUAL_REVIEW
        )
        selections.append(
            {
                **runtime_visual_selection.to_dict(),
                "used": True,
                "brokered": True,
            }
        )

    planned_routes = _build_planned_routes(contexts, compilations)

    return PreparedFleetMission(
        config=config,
        request=request,
        plan=plan,
        world_contexts=contexts,
        compilations=compilations,
        planning_failures=planning_failures,
        initial_assignment_states=initial_assignment_states,
        compiler=compiler,
        planner_limits=limits,
        planner_policy=policy,
        fleet_planner_source=fleet_planner.source,
        local_planner_source=local_planner_mode,
        adapter_selections=tuple(selections),
        model_call_records=tuple(model_call_records),
        visual_clients=visual_clients,
        runtime_visual_selection=runtime_visual_selection,
        planned_routes=planned_routes,
        headless=resolved_headless,
        vision_review_mode=review_mode,
        task_spec=task_spec,
        fleet_request_v2=request_v2,
        fleet_plan_v2=plan_v2,
        mission_interpreter_source=(
            "qwen_task_spec_v1"
            if task_spec is not None
            else "scripted_fixed_parser"
        ),
        mission_interpreter_diagnostics=interpreter_diagnostics,
        mission_interpreter_proposals=interpreter_proposals,
        fleet_planner_proposals=fleet_planner_proposals,
        fleet_semantic_findings=fleet_semantic_findings,
        preparation_context=preparation_context,
        local_planner_proposals={
            uav_id: tuple(proposals)
            for uav_id, proposals in local_proposal_history.items()
        },
        model_call_records_persisted_live=live_logger_holder[0] is not None,
        model_client_factory=(
            client_factory if args.fleet_planner == "llm" else None
        ),
        runtime_envelope_metadata=runtime_envelope_metadata,
    )


def _region_representative_world(region: object) -> tuple[float, float, float]:
    """Return a conservative representative point from a compiled WORLD region."""

    frame = getattr(region, "frame", None)
    if frame is not None and getattr(frame, "value", frame) != "WORLD_ENU":
        raise FleetLaunchConfigurationError(
            "compiled SEARCH region must use WORLD_ENU before route extraction"
        )
    if hasattr(region, "entry_point_xyz_m") and region.entry_point_xyz_m is not None:
        return tuple(float(value) for value in region.entry_point_xyz_m)
    if hasattr(region, "center_xyz_m"):
        return tuple(float(value) for value in region.center_xyz_m)
    if hasattr(region, "origin_xyz_m"):
        distance_range = region.distance_range_m
        distance_m = (float(distance_range[0]) + float(distance_range[1])) / 2.0
        angle = radians(float(region.azimuth_center_deg))
        origin = region.origin_xyz_m
        return (
            float(origin[0]) + distance_m * cos(angle),
            float(origin[1]) + distance_m * sin(angle),
            float(origin[2]),
        )
    if hasattr(region, "vertices_xyz_m"):
        vertices = tuple(region.vertices_xyz_m)
        return tuple(
            sum(float(point[index]) for point in vertices) / len(vertices)
            for index in range(3)
        )
    if hasattr(region, "centerline_xyz_m"):
        return tuple(float(value) for value in region.centerline_xyz_m[0])
    raise FleetLaunchConfigurationError(
        "compiled SEARCH region has no trusted route representative"
    )


def _build_planned_routes(
    contexts: Mapping[str, PlannerWorldContext],
    compilations: Mapping[str, AssignmentCompilation],
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    """Extract image/Oracle-free high-level routes from compiled TaskPlans."""

    routes: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for uav_id, result in compilations.items():
        context = contexts[uav_id]
        points: list[tuple[float, float, float]] = [
            tuple(float(value) for value in context.initial_uav_xyz_m)
        ]
        compiled = result.compiled_mission
        if compiled is None:
            raise FleetLaunchConfigurationError(
                f"cannot extract route from uncompiled assignment {uav_id}"
            )
        for step in compiled.task_plan.steps:
            if step.skill.value == "TAKEOFF":
                altitude = float(step.params["target_altitude"])
                points.append((points[-1][0], points[-1][1], altitude))
            elif step.skill.value == "SEARCH":
                representative = _region_representative_world(step.params["region"])
                points.append(
                    (
                        representative[0],
                        representative[1],
                        float(step.params["search_altitude_m"]),
                    )
                )
            elif step.skill.value == "GOTO":
                points.append(
                    tuple(float(value) for value in step.params["position"])
                )
            elif step.skill.value == "LAND":
                xy = step.params.get("expected_position_xy")
                if xy is not None:
                    points.append(
                        (
                            float(xy[0]),
                            float(xy[1]),
                            float(step.params["ground_altitude"]),
                        )
                    )
        deduplicated: list[tuple[float, float, float]] = []
        for point in points:
            if not deduplicated or point != deduplicated[-1]:
                deduplicated.append(point)
        if len(deduplicated) < 2:
            raise FleetLaunchConfigurationError(
                f"planned route for {uav_id} contains fewer than two points"
            )
        routes[uav_id] = tuple(deduplicated)
    return routes


class _FleetSimulationClock:
    def __init__(self, environment: object) -> None:
        self._environment = environment

    def now(self) -> float:
        world = getattr(self._environment, "world", None)
        if world is None:
            raise RuntimeError("Fleet environment World is unavailable")
        return float(world.current_time)


def _spatial_resolver(context: PlannerWorldContext, home_name: str) -> object:
    from planner.spatial_resolver import FramePose, SpatialResolver

    home = context.landing_zones[home_name]
    home_xyz = (
        home.position_xy_m[0],
        home.position_xy_m[1],
        home.ground_altitude_m,
    )
    start = FramePose(context.initial_uav_xyz_m, 0.0)
    return SpatialResolver(
        home_pose=FramePose(home_xyz, 0.0),
        uav_start_pose=start,
        named_locations={home_name: home_xyz},
    )


def _build_visual_review_coordinator(
    *,
    prepared: PreparedFleetMission,
    uav_id: str,
    manager: object,
    target_manager: object,
    worker: object,
) -> object:
    from agents.visual_review_coordinator import VisualReviewCoordinator
    from perception import CandidateBank, QwenVLMVerifier, VisualReviewGate
    from runtime import FrameStore, MissionEventBus, ReviewScheduler

    config = prepared.config
    scheduler = ReviewScheduler(
        intervals_s={
            "GOTO": config.qwen_visual_review.goto_interval_s,
            "SEARCH": config.qwen_visual_review.search_interval_s,
            "INSPECT": config.qwen_visual_review.inspect_interval_s,
            "TRACK": config.qwen_visual_review.track_interval_s,
        }
    )
    frame_store = FrameStore(
        max_frames=config.frame_store.max_frames,
        max_bytes=config.frame_store.max_bytes,
        max_age_s=config.frame_store.max_age_s,
    )
    return VisualReviewCoordinator(
        uav_id=uav_id,
        scheduler=scheduler,
        frame_store=frame_store,
        worker=worker,
        verifier=QwenVLMVerifier(
            max_image_side_px=config.qwen_visual_review.max_image_side_px,
            jpeg_quality=config.qwen_visual_review.jpeg_quality,
        ),
        gate=VisualReviewGate(
            mode=prepared.vision_review_mode,
            min_consistent_matches=2,
        ),
        skill_manager=manager,
        target_manager=target_manager,
        candidate_bank=CandidateBank(uav_id=uav_id),
        event_bus=MissionEventBus(max_events=256),
        review_timeout_s=config.qwen_visual_review.blocking_hover_timeout_s,
        max_result_age_s=config.frame_store.max_age_s,
        hover_position_tolerance_m=(
            config.qwen_visual_review.hover_position_tolerance_m
        ),
        hover_max_correction_speed_mps=(
            config.qwen_visual_review.hover_max_correction_speed_mps
        ),
        blocking_timeout_fallback=(
            config.qwen_visual_review.blocking_timeout_fallback
        ),
        max_recent_frames=config.qwen_visual_review.max_recent_frames,
        require_stable_search_candidate=True,
        min_search_candidate_observations=(
            config.target_perception.tracker.min_track_observations
        ),
        min_search_candidate_duration_s=(
            config.target_perception.tracker.min_track_duration_s
        ),
    )


def _build_brokered_visual_workers(
    prepared: PreparedFleetMission,
    broker: object,
    logger: object,
    *,
    worker_factory: object | None = None,
    dispatcher_factory: object | None = None,
) -> tuple[object, Mapping[str, object]]:
    """Create per-UAV clients behind one shared Broker acquire owner.

    This helper is deliberately pure Python and injectable so the entrypoint's
    exact ownership wiring can be tested without importing Isaac Sim.
    """

    if prepared.runtime_visual_selection is None:
        raise FleetLaunchConfigurationError(
            "runtime visual Adapter selection was not preflighted"
        )
    expected_uav_ids = set(prepared.request.available_uav_ids)
    if set(prepared.visual_clients) != expected_uav_ids:
        raise FleetLaunchConfigurationError(
            "runtime visual clients do not exactly cover available Fleet UAVs"
        )
    record_logger = getattr(logger, "log_model_call", None)
    if not callable(record_logger):
        raise TypeError("logger must provide log_model_call()")
    if worker_factory is None:
        from models import AsyncModelWorker

        worker_factory = AsyncModelWorker
    if dispatcher_factory is None:
        from fleet.model_request_dispatcher import ModelRequestDispatcher

        dispatcher_factory = ModelRequestDispatcher
    if not callable(worker_factory) or not callable(dispatcher_factory):
        raise TypeError("visual worker and dispatcher factories must be callable")

    raw_workers: dict[str, object] = {}
    dispatcher: object | None = None
    close_timeout_s = float(prepared.config.model_worker.request_timeout_s)
    try:
        for uav_id in sorted(expected_uav_ids):
            raw_workers[uav_id] = worker_factory(
                prepared.visual_clients[uav_id],
                uav_id=uav_id,
            )
        dispatcher = dispatcher_factory(
            broker,
            raw_workers,
            adapter_selection=prepared.runtime_visual_selection,
            record_logger=record_logger,
        )
        worker_for = getattr(dispatcher, "worker_for", None)
        if not callable(worker_for):
            raise TypeError("dispatcher must provide worker_for()")
        facades = {
            assignment.uav_id: worker_for(
                assignment.uav_id,
                assignment_id=assignment.assignment_id,
            )
            for assignment in prepared.plan.assignments
            if assignment.uav_id in expected_uav_ids
        }
        return dispatcher, MappingProxyType(facades)
    except BaseException:
        owners = (dispatcher,) if dispatcher is not None else tuple(raw_workers.values())
        for owner in owners:
            close = getattr(owner, "close", None)
            if callable(close):
                try:
                    close(timeout_s=close_timeout_s)
                except BaseException:
                    pass
        raise


def _not_called_model_record(
    selection: Mapping[str, object],
    *,
    fleet_mission_id: str,
    call_id: str,
) -> dict[str, object]:
    """Represent a Scripted role selection without pretending a call ran."""

    return {
        "call_id": call_id,
        "call_role": selection["call_role"],
        "fleet_mission_id": fleet_mission_id,
        "assignment_id": None,
        "uav_id": None,
        "requested_adapter": selection["requested_adapter"],
        "adapter_status": selection["adapter_status"],
        "effective_model": selection["effective_model"],
        "fallback_used": selection["fallback_used"],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_s": 0.0,
        "finish_reason": "not_called",
        "error_code": None,
        "stale_reasons": [],
    }


def _record_proposal_stage(
    audit: PlanningAuditLogger,
    *,
    mission_id: str,
    stage: str,
    role: str,
    schema: str,
    prompt_material: str,
    proposals: Sequence[Mapping[str, object]],
    assignment_id: str | None = None,
    uav_id: str | None = None,
) -> None:
    """Persist one bounded proposal chain without retaining its prompt."""

    previous_id: str | None = None
    for ordinal, raw in enumerate(proposals):
        attempt_id = f"attempt_{stage.casefold()}_{uav_id or 'fleet'}_{ordinal}"
        accepted = bool(raw.get("accepted", False))
        error = raw.get("error_code")
        proposal = raw.get("proposal", raw.get("raw_proposal"))
        raw_length = raw.get(
            "response_length", raw.get("response_text_length", 0)
        )
        length: int | None = None
        tail: str | None = None
        if proposal is None:
            length = int(raw_length) if isinstance(raw_length, int) else 0
            _, tail = sanitize_text_tail(
                str(raw.get("response_text_tail", ""))
            )
        audit.log_attempt(
            PlanningAttemptRecord(
                attempt_id=attempt_id,
                timestamp_s=float(ordinal),
                stage=stage,
                mission_id=mission_id,
                assignment_id=assignment_id,
                uav_id=uav_id,
                model_role=role,
                prompt_sha256=prompt_sha256(prompt_material),
                prompt_schema_version=schema,
                accepted=accepted,
                proposal_id=(
                    f"proposal_{stage.casefold()}_{uav_id or 'fleet'}_{ordinal}"
                ),
                repaired_from_attempt_id=(
                    previous_id if bool(raw.get("repair", False)) else None
                ),
                error_codes=(() if error is None else (str(error),)),
                proposal=(proposal if isinstance(proposal, Mapping) else None),
                raw_text_length=length,
                raw_text_tail=tail,
            )
        )
        if bool(raw.get("repair", False)):
            audit.log_recovery(
                RecoveryActionRecord(
                    recovery_action_id=(
                        f"recovery_{stage.casefold()}_{uav_id or 'fleet'}_{ordinal}"
                    ),
                    timestamp_s=float(ordinal),
                    mission_id=mission_id,
                    stage=stage,
                    action="REPAIR_PROPOSAL",
                    outcome="ACCEPTED" if accepted else "REJECTED",
                    assignment_id=assignment_id,
                    uav_id=uav_id,
                    source_attempt_id=previous_id,
                    resulting_plan_version=ordinal + 1,
                )
            )
        previous_id = attempt_id


def _record_planning_attempts(
    recorder: FleetResultRecorder,
    prepared: PreparedFleetMission,
) -> None:
    """Persist bounded structured proposals without retaining full prompts."""

    audit = recorder.planning_audit
    mission_id = prepared.request.fleet_mission_id

    def record_stage(
        *,
        stage: str,
        role: str,
        schema: str,
        prompt_material: str,
        proposals: Sequence[Mapping[str, object]],
        assignment_id: str | None = None,
        uav_id: str | None = None,
    ) -> None:
        _record_proposal_stage(
            audit,
            mission_id=mission_id,
            stage=stage,
            role=role,
            schema=schema,
            prompt_material=prompt_material,
            proposals=proposals,
            assignment_id=assignment_id,
            uav_id=uav_id,
        )

    if prepared.task_spec is not None:
        record_stage(
            stage="MISSION_INTERPRETATION",
            role=ModelCallRole.MISSION_INTERPRETATION.value,
            schema="fleet_task_spec_v1",
            prompt_material=prepared.task_spec.source_text,
            proposals=prepared.mission_interpreter_proposals,
        )
        audit.write_final_plan(
            stage="MISSION_INTERPRETATION",
            mission_id=mission_id,
            plan_version=1,
            plan=prepared.task_spec.to_dict(),
        )
    if prepared.fleet_request_v2 is not None and prepared.fleet_plan_v2 is not None:
        fleet_prompt_material = json.dumps(
            prepared.fleet_request_v2.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        record_stage(
            stage="FLEET_PLANNING",
            role=ModelCallRole.FLEET_PLAN.value,
            schema="fleet_mission_plan_v2",
            prompt_material=fleet_prompt_material,
            proposals=prepared.fleet_planner_proposals,
        )
        audit.write_final_plan(
            stage="FLEET_PLANNING",
            mission_id=mission_id,
            plan_version=prepared.fleet_plan_v2.fleet_plan_version,
            plan=prepared.fleet_plan_v2.to_dict(),
        )
    runtime_assignment_by_uav = {
        item.uav_id: item for item in prepared.plan.assignments
    }
    for uav_id, result in sorted(prepared.compilations.items()):
        assignment = runtime_assignment_by_uav[uav_id]
        record_stage(
            stage="LOCAL_PLANNING",
            role=ModelCallRole.AGENT_SPATIAL_PLAN.value,
            schema="spatial_skill_plan_v3",
            prompt_material=result.planner_request.instruction,
            proposals=prepared.local_planner_proposals.get(uav_id, ()),
            assignment_id=assignment.assignment_id,
            uav_id=uav_id,
        )
        audit.write_final_plan(
            stage="LOCAL_PLANNING",
            mission_id=mission_id,
            assignment_id=assignment.assignment_id,
            uav_id=uav_id,
            plan_version=result.agent_request.local_plan_version,
            plan=result.planner_output.to_dict(),
        )
        report = getattr(result, "validation_report", None)
        for finding in getattr(report, "findings", ()):
            audit.log_finding(
                ValidationFindingRecord(
                    finding_id=finding.finding_id,
                    timestamp_s=float(finding.timestamp),
                    stage=finding.stage,
                    scope=finding.scope,
                    severity=finding.severity.value,
                    code=finding.code.value,
                    message=finding.message,
                    mission_id=finding.mission_id,
                    assignment_id=finding.assignment_id,
                    uav_id=finding.uav_id,
                    goal_id=finding.goal_id,
                    step_id=finding.step_id,
                    proposal_id=finding.proposal_id,
                    evidence_refs=finding.evidence_refs,
                    recommended_action=finding.recommended_action.value,
                )
            )
    for index, finding in enumerate(prepared.fleet_semantic_findings):
        audit.log_finding(
            ValidationFindingRecord(
                finding_id=f"finding_fleet_semantic_{index}",
                timestamp_s=0.0,
                stage="FLEET_PLANNING",
                scope="ASSIGNMENT",
                severity="RECOVERABLE_SEMANTIC_ERROR",
                code=str(finding.get("code", "ASSIGNMENT_CONSTRAINT_DEVIATION")),
                message=str(finding.get("message", "Fleet semantic deviation")),
                mission_id=mission_id,
                assignment_id=(
                    None
                    if finding.get("assignment_id") is None
                    else str(finding["assignment_id"])
                ),
                goal_id=(
                    None
                    if finding.get("goal_id") is None
                    else str(finding["goal_id"])
                ),
                evidence_refs=(),
                recommended_action="REPAIR_LOCAL_PLAN",
            )
        )


def _write_prepared_logs(
    logger: object,
    prepared: PreparedFleetMission,
    args: argparse.Namespace,
    *,
    result_recorder: FleetResultRecorder | None = None,
) -> None:
    logger.write_run_manifest(
        {
            "schema_version": 1,
            "fleet_mission_id": prepared.request.fleet_mission_id,
            "original_instruction": prepared.request.original_instruction,
            "config_path": str(Path(args.config).expanduser().resolve()),
            "fleet_planner": args.fleet_planner,
            "mission_interpreter": prepared.mission_interpreter_source,
            "fleet_planner_source": prepared.fleet_planner_source,
            "local_planner": prepared.local_planner_source,
            "planning_contract": args.planning_contract,
            "runtime_program": args.runtime_program,
            "perception_runtime_profile": args.perception_runtime_profile,
            "acknowledge_privileged_oracle": bool(
                args.acknowledge_privileged_oracle
            ),
            "qwen_visual_review_enabled": bool(args.enable_qwen_vision),
            "vision_review_mode": prepared.vision_review_mode,
            "headless": prepared.headless,
            "debug_visualization": bool(args.debug_visualization),
            "max_sim_time_s": float(args.max_sim_time),
            "runtime_envelope": {
                "non_target_assignment_ids": sorted(
                    prepared.runtime_envelope_metadata.non_target_assignment_ids
                ),
                "semantic_target_by_assignment": dict(
                    prepared.runtime_envelope_metadata.semantic_target_by_assignment
                ),
                "required_by_assignment": dict(
                    prepared.runtime_envelope_metadata.required_by_assignment
                ),
                "compatibility_anchor_by_assignment": dict(
                    prepared.runtime_envelope_metadata.compatibility_anchor_by_assignment
                ),
            },
            "adapter_selections": [
                dict(selection) for selection in prepared.adapter_selections
            ],
        }
    )
    if not prepared.model_call_records_persisted_live:
        for record in prepared.model_call_records:
            logger.log_model_call(record)
    if prepared.task_spec is not None:
        logger.write_task_spec(prepared.task_spec)
    logger.write_fleet_plan(
        prepared.fleet_plan_v2
        if prepared.fleet_plan_v2 is not None
        else prepared.plan
    )
    if prepared.fleet_plan_v2 is not None:
        logger.write_runtime_execution_plan(prepared.plan)
    logger.write_assignments(_prepared_assignment_rows(prepared, status="PENDING"))
    for uav_id, result in prepared.compilations.items():
        logger.write_local_plan(
            uav_id,
            result.agent_request.local_plan_version,
            {
                "agent_planner_request": result.agent_request.to_dict(),
                "spatial_plan_draft_v3": result.planner_output.to_dict(),
                "compiled_task_plan": result.compiled_mission.task_plan.to_dict(),
                "goal_coverage": (
                    result.goal_coverage.to_dict()
                    if hasattr(result, "goal_coverage")
                    else None
                ),
                "validation_report": (
                    result.validation_report.to_dict()
                    if hasattr(result, "validation_report")
                    else None
                ),
            },
        )
    if result_recorder is not None:
        _record_planning_attempts(result_recorder, prepared)


def _prepared_assignment_rows(
    prepared: PreparedFleetMission,
    *,
    status: str,
) -> tuple[dict[str, object], ...]:
    metadata = prepared.runtime_envelope_metadata
    return tuple(
        {
            "assignment_id": assignment.assignment_id,
            "uav_id": assignment.uav_id,
            # The structural V1 compatibility alias is deliberately omitted
            # from semantic logs and target metrics.
            "target_alias": metadata.semantic_target_by_assignment.get(
                assignment.assignment_id,
                assignment.target_alias,
            ),
            "required": metadata.required_by_assignment.get(
                assignment.assignment_id,
                True,
            ),
            "non_target_assignment": (
                assignment.assignment_id in metadata.non_target_assignment_ids
            ),
            "priority": assignment.priority,
            "status": (
                status
                if status != "PENDING"
                else (
                    AssignmentStatus.DEGRADED_EXECUTABLE.value
                    if prepared.initial_assignment_states.get(
                        assignment.assignment_id
                    )
                    == AssignmentStatus.DEGRADED_EXECUTABLE.value
                    else (
                        AssignmentStatus.PENDING.value
                        if assignment.uav_id in prepared.compilations
                        else prepared.initial_assignment_states.get(
                            assignment.assignment_id,
                            AssignmentStatus.REASSIGNMENT_REQUIRED.value,
                        )
                    )
                )
            ),
            "local_plan_version": (
                prepared.compilations[
                    assignment.uav_id
                ].agent_request.local_plan_version
                if assignment.uav_id in prepared.compilations
                else 1
            ),
        }
        for assignment in prepared.plan.assignments
    )


_TERMINAL_SECRET_PATTERNS = (
    re.compile(r"(?i)(--api-key(?:=|\s+))\S+"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|access[_-]?token|password|passwd|"
        r"client[_-]?secret)\b\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@"),
)


def _redact_terminal_error(value: object) -> str:
    """Remove credential-shaped text before stderr or persistent logging."""

    result = str(value)
    result = _TERMINAL_SECRET_PATTERNS[0].sub(r"\1[REDACTED]", result)
    result = _TERMINAL_SECRET_PATTERNS[1].sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        result,
    )
    result = _TERMINAL_SECRET_PATTERNS[2].sub("Bearer [REDACTED]", result)
    result = _TERMINAL_SECRET_PATTERNS[3].sub("sk-[REDACTED]", result)
    result = _TERMINAL_SECRET_PATTERNS[4].sub(r"\1[REDACTED]@", result)
    return result


def _append_terminal_error(current: str | None, value: object) -> str:
    addition = _redact_terminal_error(value)
    return addition if current is None else f"{current}; {addition}"


def _apply_terminal_outcome(
    summary: dict[str, object],
    *,
    exit_code: int,
    terminal_error: str | None,
    interrupted: bool,
) -> None:
    if terminal_error is not None:
        runtime_status = str(summary.get("status", "UNKNOWN"))
        summary.setdefault("shutdown_runtime_status", runtime_status)
        summary["status"] = "CANCELED" if interrupted else "FAILED"
        summary["last_error"] = _redact_terminal_error(terminal_error)
    summary["exit_code"] = exit_code
    summary["interrupted"] = interrupted


def _terminal_log_payload(
    prepared: PreparedFleetMission,
    runtime: object | None,
    *,
    exit_code: int,
    terminal_error: str | None,
    interrupted: bool,
) -> tuple[tuple[Mapping[str, object], ...], dict[str, object]]:
    """Build one bounded summary for success, failure, or interruption."""

    terminal_error = (
        None
        if terminal_error is None
        else _redact_terminal_error(terminal_error)
    )
    prepared_rows = {
        str(row["assignment_id"]): dict(row)
        for row in _prepared_assignment_rows(prepared, status="PENDING")
    }
    if runtime is None:
        terminal_status = "CANCELED" if interrupted else "FAILED"
        rows = tuple(
            {**row, "status": terminal_status}
            for row in prepared_rows.values()
        )
        summary: dict[str, object] = {
            "status": terminal_status,
            "fleet_mission_id": prepared.request.fleet_mission_id,
            "fleet_plan_version": prepared.plan.fleet_plan_version,
            "agent_plan_versions": {
                uav_id: result.agent_request.local_plan_version
                for uav_id, result in prepared.compilations.items()
            },
            "agent_statuses": {
                uav_id: "NOT_STARTED" for uav_id in prepared.compilations
            },
            "assignments": {
                str(row["assignment_id"]): dict(row) for row in rows
            },
            "last_airspace_decision": None,
            "event_count": 0,
            "last_error": terminal_error,
            "target_registry": {
                "claim_policy": (
                    prepared.plan.coordination_policy.target_claim_policy.value
                ),
                "targets": {},
                "event_count": 0,
            },
            "model_broker": {
                "pending": [],
                "inflight": [],
                "log_count": 0,
            },
        }
    else:
        snapshot = runtime.snapshot()
        runtime_rows = {
            str(assignment_id): dict(row)
            for assignment_id, row in snapshot.assignments.items()
        }
        missing_status = (
            "CANCELED"
            if interrupted
            else "FAILED"
            if terminal_error is not None
            else "PENDING"
        )
        merged_rows = [
            (
                {**prepared_row, **runtime_rows[assignment_id]}
                if assignment_id in runtime_rows
                else {**prepared_row, "status": missing_status}
            )
            for assignment_id, prepared_row in prepared_rows.items()
        ]
        merged_rows.extend(
            row
            for assignment_id, row in runtime_rows.items()
            if assignment_id not in prepared_rows
        )
        rows = tuple(merged_rows)
        event_types = tuple(
            str(event.get("event_type", "")) for event in snapshot.events
        )
        summary = {
            **snapshot.to_summary_dict(),
            "fleet_mission_id": prepared.request.fleet_mission_id,
            "fleet_plan_version": (
                prepared.plan.fleet_plan_version
                if snapshot.fleet_plan_version is None
                else snapshot.fleet_plan_version
            ),
            "agent_plan_versions": {
                **{
                    uav_id: result.agent_request.local_plan_version
                    for uav_id, result in prepared.compilations.items()
                },
                **dict(snapshot.agent_plan_versions),
            },
            "agent_statuses": {
                **{
                    uav_id: "NOT_STARTED" for uav_id in prepared.compilations
                },
                **dict(snapshot.agent_statuses),
            },
            "reassignment_count": sum(
                value == "FLEET_REPLAN_REQUESTED" for value in event_types
            ),
            "reassignments_succeeded": sum(
                value == "FLEET_PLAN_VERSION_PUBLISHED" for value in event_types
            ),
            "assignments": {
                str(row["assignment_id"]): dict(row) for row in rows
            },
            "target_registry": runtime.targets.summary_snapshot(),
            "model_broker": runtime.model_broker.summary_snapshot(),
        }
    _apply_terminal_outcome(
        summary,
        exit_code=exit_code,
        terminal_error=terminal_error,
        interrupted=interrupted,
    )
    summary["original_instruction"] = prepared.request.original_instruction
    summary["planning_failures"] = dict(prepared.planning_failures)
    return tuple(rows), summary


def _drain_runtime_logs(
    logger: object,
    visual_coordinators: Mapping[str, object],
    visual_log_cursors: dict[str, int],
    managers: Mapping[str, object],
    transition_log_cursors: dict[str, int],
) -> None:
    """Persist every record exactly once, including records emitted at shutdown."""

    for uav_id in sorted(visual_coordinators):
        records = tuple(visual_coordinators[uav_id].records)
        cursor = visual_log_cursors.get(uav_id, 0)
        if cursor > len(records):
            raise RuntimeError(f"visual log moved backwards for {uav_id}")
        while cursor < len(records):
            logger.log_visual_review(uav_id, records[cursor])
            cursor += 1
            visual_log_cursors[uav_id] = cursor

    for uav_id in sorted(managers):
        records = tuple(managers[uav_id].transition_log)
        cursor = transition_log_cursors.get(uav_id, 0)
        if cursor > len(records):
            raise RuntimeError(f"transition log moved backwards for {uav_id}")
        while cursor < len(records):
            logger.log_agent_transition(uav_id, records[cursor])
            cursor += 1
            transition_log_cursors[uav_id] = cursor


def _best_effort_cancel_and_land(
    runtime: object,
    simulation_app: object,
    clock: _FleetSimulationClock,
    *,
    guard_s: float = 120.0,
) -> None:
    """Keep advancing an interrupted Fleet until fail-safe LAND terminates."""

    from fleet.runtime import FleetStatus

    if getattr(runtime, "status", None) is not FleetStatus.RUNNING:
        return
    if not bool(getattr(runtime, "cancel_requested", False)):
        runtime.cancel()
    deadline_s = clock.now() + guard_s
    while (
        simulation_app.is_running()
        and runtime.status is FleetStatus.RUNNING
        and clock.now() < deadline_s
    ):
        runtime.tick()
    if runtime.status is FleetStatus.RUNNING:
        raise RuntimeError("Fleet cancel-and-land exceeded shutdown guard")


def _close_execution_resources(
    runtime: object | None,
    environment: object | None,
    simulation_app: object | None,
) -> None:
    """Close the runtime owner and always release ``SimulationApp`` last."""

    first_error: BaseException | None = None
    owner = runtime if runtime is not None else environment
    close_owner = getattr(owner, "close", None)
    if callable(close_owner):
        try:
            close_owner()
        except BaseException as exc:
            first_error = exc
            # FleetMissionRuntime normally owns Environment.close().  If an
            # earlier Agent.close() failed, explicitly attempt the environment
            # so World/Camera resources are still released before Isaac exits.
            if runtime is not None and environment is not None:
                close_environment = getattr(environment, "close", None)
                if callable(close_environment):
                    try:
                        close_environment()
                    except BaseException:
                        # Preserve the first failure; SimulationApp still must
                        # be closed after this best-effort fallback.
                        pass
    if simulation_app is not None:
        try:
            simulation_app.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _create_fleet_debug_visualizer(
    plan: FleetMissionPlan,
    agent_plan_versions: Mapping[str, int | None],
    *,
    headless: bool,
    draw_factory: object | None = None,
    overlay_factory: object | None = None,
) -> object:
    """Build viewport-only Fleet geometry and GUI text after Isaac startup."""

    if not isinstance(headless, bool):
        raise TypeError("headless must be bool")
    if draw_factory is None:
        from visualization import FleetDebugDraw

        draw_factory = FleetDebugDraw
    if not callable(draw_factory):
        raise TypeError("draw_factory must be callable")
    status_overlay = None
    if not headless:
        if overlay_factory is None:
            from visualization import FleetStatusOverlay

            overlay_factory = FleetStatusOverlay
        if not callable(overlay_factory):
            raise TypeError("overlay_factory must be callable")
        status_overlay = overlay_factory()
    try:
        visualizer = draw_factory(status_overlay=status_overlay)
    except BaseException:
        # Ownership transfers to FleetDebugDraw only after its constructor
        # returns.  Do not leak the omni.ui window on constructor failure.
        close_overlay = getattr(status_overlay, "close", None)
        if callable(close_overlay):
            close_overlay()
        raise
    try:
        set_plan = getattr(visualizer, "set_plan", None)
        if not callable(set_plan):
            raise TypeError("draw_factory must return an object with set_plan()")
        set_plan(plan, agent_plan_versions=agent_plan_versions)
    except BaseException:
        close = getattr(visualizer, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                # Preserve the construction/set_plan failure that explains
                # why viewport setup could not complete.
                pass
        else:
            close_overlay = getattr(status_overlay, "close", None)
            if callable(close_overlay):
                close_overlay()
        raise
    return visualizer


def _finalize_bounded_results(
    recorder: FleetResultRecorder,
    prepared: PreparedFleetMission,
    summary: Mapping[str, object],
    *,
    wall_time_s: float,
) -> dict[str, object]:
    """Derive consistent Fleet/Goal scalars from the trusted terminal state."""

    enriched = dict(summary)
    assignment_rows = enriched.get("assignments", {})
    assignments = (
        assignment_rows if isinstance(assignment_rows, Mapping) else {}
    )
    statuses = {
        str(key): str(value.get("status", "UNKNOWN"))
        for key, value in assignments.items()
        if isinstance(value, Mapping)
    }
    v2_assignment_for_goal = {
        goal_id: assignment
        for assignment in (
            ()
            if prepared.fleet_plan_v2 is None
            else prepared.fleet_plan_v2.assignments
        )
        for goal_id in assignment.goal_ids
    }
    compilation_by_assignment = {
        result.agent_request.assignment_id: result
        for result in prepared.compilations.values()
    }
    runtime_goal_routes: dict[str, tuple[str, str, bool]] = {}
    runtime_reassignments: tuple[Mapping[str, object], ...] = ()
    if prepared.preparation_context is not None:
        raw_reassignments = prepared.preparation_context.get(
            "runtime_reassignments", ()
        )
        if isinstance(raw_reassignments, Sequence) and not isinstance(
            raw_reassignments, (str, bytes)
        ):
            runtime_reassignments = tuple(
                item for item in raw_reassignments if isinstance(item, Mapping)
            )[:64]
    for item in runtime_reassignments:
        assignment_id = item.get("replacement_assignment_id")
        uav_id = item.get("uav_id")
        goal_ids = item.get("goal_ids", ())
        if (
            isinstance(assignment_id, str)
            and isinstance(uav_id, str)
            and isinstance(goal_ids, Sequence)
            and not isinstance(goal_ids, (str, bytes))
        ):
            for goal_id in goal_ids[:32]:
                if isinstance(goal_id, str):
                    runtime_goal_routes[goal_id] = (
                        assignment_id,
                        uav_id,
                        bool(item.get("semantically_valid", False)),
                    )
    completed_goals = 0
    goal_count = 0
    mission_sim_time_s = recorder.latest_state_timestamp_s
    if prepared.task_spec is not None:
        for goal_id in prepared.task_spec.all_goal_ids:
            goal_count += 1
            goal = prepared.task_spec.goal(goal_id)
            assignment = v2_assignment_for_goal.get(goal_id)
            runtime_route = runtime_goal_routes.get(goal_id)
            assignment_id = (
                runtime_route[0]
                if runtime_route is not None
                else None if assignment is None else assignment.assignment_id
            )
            assigned_uav_id = (
                runtime_route[1]
                if runtime_route is not None
                else None if assignment is None else assignment.uav_id
            )
            result = (
                None
                if assignment_id is None
                else compilation_by_assignment.get(assignment_id)
            )
            coverage = None
            if result is not None:
                report = getattr(result, "goal_coverage", None)
                coverage = next(
                    (
                        item
                        for item in getattr(report, "coverages", ())
                        if item.goal_id == goal_id
                    ),
                    None,
                )
            completed = bool(
                assignment_id is not None
                and statuses.get(assignment_id) == AssignmentStatus.SUCCEEDED.value
                and (
                    runtime_route is not None and runtime_route[2]
                    or coverage is not None and coverage.covered
                )
            )
            completed_goals += int(completed)
            deviation = None
            if assignment is not None:
                related = tuple(
                    item.constraint_id for item in assignment.deviations
                )
                deviation = "|".join(related) or None
            recorder.record_goal_result(
                GoalResultRecord(
                    fleet_mission_id=prepared.request.fleet_mission_id,
                    goal_id=goal_id,
                    goal_type=goal.goal_type.value,
                    completed=completed,
                    assignment_id=assignment_id,
                    uav_id=assigned_uav_id,
                    completion_time_s=(mission_sim_time_s if completed else None),
                    evidence_source=(
                        None
                        if coverage is None or not coverage.evidence_step_ids
                        else "|".join(coverage.evidence_step_ids)
                    ),
                    unmet_reason=(
                        None
                        if completed
                        else (
                            "UNASSIGNED"
                            if assignment is None and runtime_route is None
                            else (
                                coverage.message
                                if coverage is not None and not coverage.covered
                                else statuses.get(assignment_id, "NOT_EXECUTED")
                            )
                        )
                    ),
                    constraint_deviation=deviation,
                )
            )
    status = str(enriched.get("status", "UNKNOWN"))
    repair_records = tuple(
        item
        for item in (
            *prepared.mission_interpreter_proposals,
            *prepared.fleet_planner_proposals,
            *(
                proposal
                for proposals in prepared.local_planner_proposals.values()
                for proposal in proposals
            ),
        )
        if bool(item.get("repair", False))
    )
    live_model_records = (
        None
        if prepared.preparation_context is None
        else prepared.preparation_context.get("model_call_records")
    )
    model_records = tuple(
        item
        for item in (
            live_model_records
            if isinstance(live_model_records, Sequence)
            and not isinstance(live_model_records, (str, bytes))
            else prepared.model_call_records
        )
        if isinstance(item, Mapping)
    )
    prompt_tokens = sum(
        int(item.get("prompt_tokens", 0) or 0) for item in model_records
    )
    completion_tokens = sum(
        int(item.get("completion_tokens", 0) or 0) for item in model_records
    )
    model_latency_s = sum(
        float(item.get("latency_s", 0.0) or 0.0) for item in model_records
    )
    assignments_succeeded = sum(
        value == AssignmentStatus.SUCCEEDED.value for value in statuses.values()
    )
    assignments_failed = sum(
        value
        in {
            AssignmentStatus.FAILED.value,
            AssignmentStatus.CANCELED.value,
            AssignmentStatus.REASSIGNMENT_REQUIRED.value,
        }
        for value in statuses.values()
    )
    semantic_success = goal_count == completed_goals and not (
        prepared.fleet_semantic_findings
    )
    strict_success = status == FleetStatus.SUCCEEDED.value and semantic_success
    collision_count = recorder.collision_count
    out_of_bounds_count = recorder.out_of_bounds_count
    emergency_landing_count = recorder.emergency_landing_count
    enriched.update(
        {
            "strict_success": strict_success,
            "semantic_success": semantic_success,
            "execution_success": status == FleetStatus.SUCCEEDED.value,
            "safety_success": (
                status not in {"CANCELED", "FATAL_SAFETY"}
                and collision_count == 0
                and out_of_bounds_count == 0
            ),
            "partial_success": 0 < completed_goals < goal_count,
            "goal_count": goal_count,
            "goals_completed": completed_goals,
            "goal_completion_rate": (
                completed_goals / goal_count if goal_count else 0.0
            ),
            "assignment_count": len(statuses),
            "assignments_succeeded": assignments_succeeded,
            "assignments_failed": assignments_failed,
            "reassignment_count": len(runtime_reassignments),
            "reassignments_succeeded": int(
                sum(
                    statuses.get(str(item.get("replacement_assignment_id")))
                    == AssignmentStatus.SUCCEEDED.value
                    for item in runtime_reassignments
                )
            ),
            "repair_count": len(repair_records),
            "repairs_succeeded": sum(
                bool(item.get("accepted", False)) for item in repair_records
            ),
            "repair_success_rate": (
                sum(bool(item.get("accepted", False)) for item in repair_records)
                / len(repair_records)
                if repair_records
                else 0.0
            ),
            "validation_finding_count": len(prepared.fleet_semantic_findings)
            + sum(
                len(getattr(getattr(item, "validation_report", None), "findings", ()))
                for item in prepared.compilations.values()
            ),
            "collision_count": collision_count,
            "out_of_bounds_count": out_of_bounds_count,
            "emergency_landing_count": emergency_landing_count,
            "minimum_inter_uav_distance_m": (
                recorder.minimum_inter_uav_distance_m
            ),
            "mission_sim_time_s": mission_sim_time_s,
            "interpreter_schema_success": prepared.task_spec is not None
            or prepared.mission_interpreter_source == "scripted_fixed_parser",
            "fleet_plan_success": bool(prepared.plan.assignments),
            "local_plan_success": bool(prepared.compilations),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model_latency_s": model_latency_s,
            "mission_wall_time_s": max(0.0, float(wall_time_s)),
        }
    )
    return recorder.finalize(enriched)


def _record_fleet_state_samples(
    recorder: FleetResultRecorder,
    prepared: PreparedFleetMission,
    runtime: object,
    environment: object,
) -> None:
    """Feed synchronized state at simulation rate; the recorder keeps 1 Hz."""

    from fleet.airspace_manager import coerce_fleet_pose_snapshot

    pose_snapshot = coerce_fleet_pose_snapshot(
        environment.get_fleet_pose_snapshot()
    )
    positions = tuple(
        pose.position_xyz_m for pose in pose_snapshot.poses.values()
    )
    minimum_separation = (
        min(
            dist(first, second)
            for index, first in enumerate(positions)
            for second in positions[index + 1 :]
        )
        if len(positions) >= 2
        else None
    )
    out_of_bounds_uavs = tuple(
        uav_id
        for uav_id, pose in pose_snapshot.poses.items()
        if uav_id in prepared.world_contexts
        and any(
            coordinate < lower or coordinate > upper
            for coordinate, lower, upper in zip(
                pose.position_xyz_m,
                prepared.world_contexts[uav_id].scene_min_xyz_m,
                prepared.world_contexts[uav_id].scene_max_xyz_m,
            )
        )
    )
    recorder.observe_safety_snapshot(
        collision=(
            minimum_separation is not None
            and minimum_separation
            < prepared.plan.coordination_policy.minimum_uav_separation_m
        ),
        out_of_bounds_uav_ids=out_of_bounds_uavs,
    )
    runtime_snapshot = runtime.snapshot()
    assignment_by_uav = {
        item.uav_id: item.assignment_id for item in prepared.plan.assignments
    }
    home_altitude_by_uav = {
        item.id: float(item.initial_position_xyz_m[2]) for item in prepared.config.uavs
    }
    for uav_id, pose in pose_snapshot.poses.items():
        agent = getattr(runtime, "agents", {}).get(uav_id)
        agent_snapshot = None
        if agent is not None:
            try:
                agent_snapshot = agent.snapshot()
            except Exception:
                agent_snapshot = None

        def snapshot_value(value: object, name: str, default: object = None) -> object:
            if isinstance(value, Mapping):
                return value.get(name, default)
            return getattr(value, name, default)

        target = snapshot_value(agent_snapshot, "target")
        lifecycle_raw = snapshot_value(target, "lifecycle", "UNINITIALIZED")
        lifecycle = str(getattr(lifecycle_raw, "value", lifecycle_raw)).upper()
        detected = lifecycle not in {"UNINITIALIZED", "SEARCHING"}
        locked = lifecycle in {
            "LOCKED",
            "TRACKING",
            "LOST",
            "REACQUIRING",
            "TERMINATED",
        }
        feedback = snapshot_value(agent_snapshot, "feedback")
        step_id = snapshot_value(feedback, "step_id")
        active_skill_raw = snapshot_value(agent_snapshot, "active_skill")
        active_skill = (
            None
            if active_skill_raw is None
            else str(getattr(active_skill_raw, "value", active_skill_raw))
        )
        home_z = home_altitude_by_uav.get(uav_id, 0.0)
        on_ground = (
            abs(float(pose.position_xyz_m[2]) - home_z) <= 0.15
            and abs(float(pose.velocity_xyz_mps[2])) <= 0.15
        )
        recorder.record_state_sample(
            StateSampleRecord(
                fleet_mission_id=prepared.request.fleet_mission_id,
                uav_id=uav_id,
                timestamp_s=pose_snapshot.timestamp_s,
                position_xyz_m=pose.position_xyz_m,
                velocity_xyz_mps=pose.velocity_xyz_mps,
                mode=(
                    "GROUND"
                    if on_ground
                    else active_skill
                    or runtime_snapshot.agent_statuses.get(uav_id, "UNKNOWN")
                ),
                assignment_id=assignment_by_uav.get(uav_id),
                step_id=(step_id if isinstance(step_id, str) else None),
                target_detected=detected,
                target_locked=locked,
                minimum_inter_uav_distance_m=minimum_separation,
            )
        )


def _record_skill_execution_metrics(
    recorder: FleetResultRecorder,
    prepared: PreparedFleetMission,
    managers: Mapping[str, object],
    *,
    status_by_uav: Mapping[str, str] | None = None,
) -> None:
    assignment_by_uav = {
        item.uav_id: item.assignment_id for item in prepared.plan.assignments
    }
    base_metrics = {
        item.uav_id: item
        for item in recorder.agent_metric_snapshots(
            status_by_uav=status_by_uav,
        )
    }
    for uav_id, manager in sorted(managers.items()):
        starts: dict[str, float] = {}
        valid_track_duration_s = 0.0
        target_lost_count = 0
        target_reacquired_count = 0
        returned_home = False
        landed = False
        world_context = prepared.world_contexts.get(uav_id)
        home_position = (
            world_context.initial_uav_xyz_m
            if world_context is not None
            else next(
                item.initial_position_xyz_m
                for item in prepared.config.uavs
                if item.id == uav_id
            )
        )
        compilation = prepared.compilations.get(uav_id)
        compiled_steps = (
            ()
            if compilation is None
            else compilation.compiled_mission.task_plan.steps
        )
        return_step_ids = {
            step.step_id
            for step in compiled_steps
            if step.skill.value == "GOTO"
            and isinstance(step.params.get("position"), tuple)
            and dist(step.params["position"][:2], home_position[:2]) <= 1.0
        }
        home_land_step_ids = {
            step.step_id
            for step in compiled_steps
            if step.skill.value == "LAND"
            and isinstance(step.params.get("expected_position_xy"), tuple)
            and dist(
                step.params["expected_position_xy"], home_position[:2]
            ) <= 1.0
        }
        for transition in getattr(manager, "transition_log", ()):
            timestamp = float(transition.timestamp)
            if transition.old_skill is not None and transition.old_step_id is not None:
                start = starts.pop(transition.old_step_id, timestamp)
                result = transition.result_code
                recorder.record_skill_execution(
                    {
                        "schema_version": 1,
                        "fleet_mission_id": prepared.request.fleet_mission_id,
                        "assignment_id": assignment_by_uav.get(uav_id),
                        "uav_id": uav_id,
                        "step_id": transition.old_step_id,
                        "skill_name": transition.old_skill.value,
                        "start_time_s": start,
                        "end_time_s": max(start, timestamp),
                        "result_code": (
                            transition.reason.upper()
                            if result is None
                            else result.name
                        ),
                        "attempt": transition.recovery_attempt or 1,
                        "recovery_action_id": (
                            None
                            if transition.recovery_attempt is None
                            else f"runtime_recovery_{uav_id}_{transition.recovery_attempt}"
                        ),
                    }
                )
                result_name = (
                    "" if result is None else str(getattr(result, "name", result))
                )
                skill_name = transition.old_skill.value
                duration_s = max(0.0, timestamp - start)
                if skill_name == "TRACK" and result_name == "TRACK_COMPLETE":
                    valid_track_duration_s += duration_s
                if result_name == "TARGET_LOST":
                    target_lost_count += 1
                if skill_name == "REACQUIRE" and result_name in {
                    "TARGET_FOUND",
                    "TRACK_COMPLETE",
                }:
                    target_reacquired_count += 1
                if (
                    skill_name == "GOTO"
                    and result_name == "GOAL_REACHED"
                    and transition.old_step_id in return_step_ids
                ):
                    returned_home = True
                if (
                    skill_name == "LAND"
                    and result_name == "LAND_COMPLETE"
                    and transition.old_step_id in home_land_step_ids
                ):
                    landed = True
                    returned_home = True
            if transition.new_skill is not None and transition.new_step_id is not None:
                starts[transition.new_step_id] = timestamp
        base = base_metrics.get(
            uav_id,
            AgentMetricRecord(
                fleet_mission_id=prepared.request.fleet_mission_id,
                assignment_id=assignment_by_uav.get(uav_id),
                uav_id=uav_id,
                status=(status_by_uav or {}).get(uav_id, "UNKNOWN"),
            ),
        )
        recorder.record_agent_metrics(
            replace(
                base,
                assignment_id=assignment_by_uav.get(uav_id),
                valid_track_duration_s=valid_track_duration_s,
                target_lost_count=target_lost_count,
                target_reacquired_count=target_reacquired_count,
                returned_home=returned_home,
                landed=landed,
            )
        )


def _finalize_fleet_execution(
    prepared: PreparedFleetMission,
    logger: object,
    *,
    runtime: object | None,
    simulation_app: object | None,
    environment: object | None,
    clock: _FleetSimulationClock | None,
    debug_draw: object | None,
    workers: Sequence[object],
    visual_coordinators: Mapping[str, object],
    visual_log_cursors: dict[str, int],
    managers: Mapping[str, object],
    transition_log_cursors: dict[str, int],
    exit_code: int,
    terminal_error: str | None,
    interrupted: bool,
    print_terminal_summary: bool,
    primary_error_present: bool,
    result_recorder: FleetResultRecorder | None = None,
    result_wall_time_s: float = 0.0,
    no_summary_figures: bool = False,
) -> int:
    """Finish one run without allowing cleanup/log failures to mask its cause."""

    def record_finalization_error(label: str, exc: BaseException) -> None:
        nonlocal exit_code, interrupted, terminal_error
        detail = _redact_terminal_error(f"{label}: {type(exc).__name__}: {exc}")
        terminal_error = _append_terminal_error(terminal_error, detail)
        if isinstance(exc, KeyboardInterrupt) and not primary_error_present:
            interrupted = True
            exit_code = 130
        elif not primary_error_present and not interrupted:
            exit_code = 1
        print(f"[Fleet] {detail}", file=sys.stderr)

    if runtime is not None and clock is not None and simulation_app is not None:
        try:
            _best_effort_cancel_and_land(runtime, simulation_app, clock)
        except BaseException as exc:
            record_finalization_error("best-effort cancel-and-land failed", exc)

    try:
        _drain_runtime_logs(
            logger,
            visual_coordinators,
            visual_log_cursors,
            managers,
            transition_log_cursors,
        )
    except BaseException as exc:
        record_finalization_error("terminal runtime log drain failed", exc)

    assignment_rows: tuple[Mapping[str, object], ...] | None = None
    final_summary: dict[str, object] | None = None
    try:
        assignment_rows, final_summary = _terminal_log_payload(
            prepared,
            runtime,
            exit_code=exit_code,
            terminal_error=terminal_error,
            interrupted=interrupted,
        )
    except BaseException as exc:
        record_finalization_error("terminal snapshot failed", exc)
        try:
            assignment_rows, final_summary = _terminal_log_payload(
                prepared,
                None,
                exit_code=exit_code,
                terminal_error=terminal_error,
                interrupted=interrupted,
            )
        except BaseException as fallback_exc:
            record_finalization_error(
                "prepared terminal snapshot fallback failed",
                fallback_exc,
            )

    if debug_draw is not None:
        close_debug_draw = getattr(debug_draw, "close", None)
        if callable(close_debug_draw):
            try:
                close_debug_draw()
            except BaseException as exc:
                record_finalization_error("debug visualization cleanup failed", exc)
    for worker in workers:
        close_worker = getattr(worker, "close", None)
        if callable(close_worker):
            try:
                close_worker()
            except BaseException as exc:
                record_finalization_error("visual worker cleanup failed", exc)
    try:
        _close_execution_resources(runtime, environment, simulation_app)
    except BaseException as exc:
        record_finalization_error("execution resource cleanup failed", exc)

    # Dispatcher.close() can complete or stale outstanding model work after
    # the first terminal snapshot was built.  Refresh only the bounded Broker
    # scheduler summary so pending/inflight/log_count describe durable final
    # model_calls.csv state rather than the pre-cleanup instant.
    if final_summary is not None and runtime is not None:
        try:
            final_summary["model_broker"] = (
                runtime.model_broker.summary_snapshot()
            )
        except BaseException as exc:
            record_finalization_error("terminal Broker summary refresh failed", exc)

    if final_summary is not None:
        _apply_terminal_outcome(
            final_summary,
            exit_code=exit_code,
            terminal_error=terminal_error,
            interrupted=interrupted,
        )

    if assignment_rows is not None:
        try:
            logger.write_assignments(assignment_rows)
        except BaseException as exc:
            record_finalization_error("terminal assignments write failed", exc)
            if final_summary is not None:
                _apply_terminal_outcome(
                    final_summary,
                    exit_code=exit_code,
                    terminal_error=terminal_error,
                    interrupted=interrupted,
                )

    if final_summary is not None and result_recorder is not None:
        try:
            _record_skill_execution_metrics(
                result_recorder,
                prepared,
                managers,
                status_by_uav=(
                    final_summary.get("agent_statuses", {})
                    if isinstance(final_summary.get("agent_statuses"), Mapping)
                    else {}
                ),
            )
            final_summary = _finalize_bounded_results(
                result_recorder,
                prepared,
                final_summary,
                wall_time_s=result_wall_time_s,
            )
        except BaseException as exc:
            # Result-output defects must not alter flight outcome or prevent
            # the sparse legacy summary from being written.
            print(
                "[Fleet] bounded result finalization failed: "
                + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
                file=sys.stderr,
            )

    summary_written = False
    if final_summary is not None:
        try:
            logger.write_summary(final_summary)
            summary_written = True
        except BaseException as exc:
            record_finalization_error("terminal summary write failed", exc)

    if print_terminal_summary and summary_written:
        try:
            print(json.dumps(final_summary, ensure_ascii=False, indent=2))
        except Exception as exc:
            # The durable summary is authoritative; a closed stdout pipe must
            # not retroactively change a fully recorded mission outcome.
            print(
                "[Fleet] terminal summary display failed: "
                + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
                file=sys.stderr,
            )
    if result_recorder is not None:
        try:
            generate_fleet_report(
                result_recorder.run_dir,
                no_summary_figures=no_summary_figures,
                max_run_bytes=prepared.config.results.max_run_bytes,
            )
            if final_summary is not None:
                # report.md and the optional scalar figures are part of the
                # same hard-bounded run, so refresh the persisted byte count
                # only after those artifacts have been generated.
                final_summary = result_recorder.finalize(final_summary)
                logger.write_summary(final_summary)
        except BaseException as exc:
            print(
                "[Fleet] report generation failed: "
                + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
                file=sys.stderr,
            )
        finally:
            result_recorder.close()
    return exit_code


def run_prepared_fleet_mission(
    prepared: PreparedFleetMission,
    args: argparse.Namespace,
) -> int:
    """Cross the Isaac boundary and execute an already prepared mission."""

    # Create the sparse run record before crossing the Isaac boundary so an
    # import, SimulationApp, or environment setup failure is still auditable.
    managed_run_dir = getattr(args, "_managed_run_dir", None)
    uav_ids = tuple(
        sorted({assignment.uav_id for assignment in prepared.plan.assignments})
    )
    logger = (
        FleetMissionLogger.attach_run_dir(
            managed_run_dir,
            prepared.request.fleet_mission_id,
            uav_ids=uav_ids,
            max_record_bytes=prepared.config.results.max_record_bytes,
            max_stream_bytes=prepared.config.results.max_stream_bytes,
            max_run_bytes=prepared.config.results.max_run_bytes,
        )
        if managed_run_dir is not None
        else FleetMissionLogger(
            args.output_root,
            prepared.request.fleet_mission_id,
            uav_ids=uav_ids,
            max_record_bytes=prepared.config.results.max_record_bytes,
            max_stream_bytes=prepared.config.results.max_stream_bytes,
            max_run_bytes=prepared.config.results.max_run_bytes,
        )
    )
    result_recorder = FleetResultRecorder(
        logger.run_dir,
        config=prepared.config,
        fleet_mission_id=prepared.request.fleet_mission_id,
    )
    run_wall_started = time.monotonic()
    _write_prepared_logs(
        logger,
        prepared,
        args,
        result_recorder=result_recorder,
    )

    if not prepared.compilations:
        rows = _prepared_assignment_rows(prepared, status="FAILED")
        logger.write_assignments(rows)
        no_plan_summary = {
                "status": "FAILED_NO_EXECUTABLE_PLAN",
                "fleet_mission_id": prepared.request.fleet_mission_id,
                "fleet_plan_version": prepared.plan.fleet_plan_version,
                "assignments": {
                    str(row["assignment_id"]): dict(row) for row in rows
                },
                "planning_failures": dict(prepared.planning_failures),
                "agent_plan_versions": {},
                "agent_statuses": {
                    assignment.uav_id: "NOT_STARTED"
                    for assignment in prepared.plan.assignments
                },
                "event_count": 0,
                "last_airspace_decision": None,
                "last_error": "all assignments lack a safe executable local plan",
                "exit_code": 1,
                "interrupted": False,
            }
        logger.write_summary(no_plan_summary)
        result_recorder.record_failure(
            {
                "stage": "LOCAL_PLANNING",
                "code": "FAILED_NO_EXECUTABLE_PLAN",
                "severity": "HARD_ACTION_BLOCK",
                "status": "FAILED_NO_EXECUTABLE_PLAN",
                "message": no_plan_summary["last_error"],
            }
        )
        no_plan_summary = _finalize_bounded_results(
            result_recorder,
            prepared,
            no_plan_summary,
            wall_time_s=time.monotonic() - run_wall_started,
        )
        logger.write_summary(no_plan_summary)
        try:
            generate_fleet_report(
                logger.run_dir,
                no_summary_figures=bool(args.no_summary_figures),
                max_run_bytes=prepared.config.results.max_run_bytes,
            )
            # Refresh final_run_bytes after report/figures exist.
            no_plan_summary = result_recorder.finalize(no_plan_summary)
            logger.write_summary(no_plan_summary)
        except Exception as report_exc:
            print(
                "[Fleet] no-plan report generation failed: "
                + _redact_terminal_error(
                    f"{type(report_exc).__name__}: {report_exc}"
                ),
                file=sys.stderr,
            )
        finally:
            result_recorder.close()
        print(
            "[Fleet] no safe executable local plan; Isaac Sim was not started",
            file=sys.stderr,
        )
        return 1

    simulation_app = None
    environment = None
    runtime = None
    clock = None
    debug_draw = None
    workers: list[object] = []
    visual_coordinators: dict[str, object] = {}
    visual_log_cursors: dict[str, int] = {}
    managers: dict[str, object] = {}
    transition_log_cursors: dict[str, int] = {}
    exit_code = 1
    terminal_error: str | None = None
    interrupted = False
    print_terminal_summary = False
    primary_error: BaseException | None = None
    primary_traceback = None
    try:
        # FIRST ISAAC IMPORT.  All planning, validation, compilation, model
        # calls, perception preflight, Safety preflight, and initial logging
        # have completed above this line.
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": prepared.headless})
        if args.debug_visualization:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.util.debug_draw")
            simulation_app.update()
        from agents.mission_agent import MissionAgent
        from env.fleet_uav_search_env import FleetUavSearchEnv
        from fleet.airspace_manager import (
            FleetAirspaceManager,
            coerce_fleet_pose_snapshot,
        )
        from fleet.model_request_broker import GlobalModelRequestBroker
        from fleet.runtime import FleetMissionRuntime, FleetStatus
        from fleet.target_registry import SharedTargetRegistry
        from perception.factory import build_target_perception_backend
        from perception.vision_backend import DisabledTargetPerceptionBackend
        from perception.runtime import (
            GuardedPerceptionBackend,
            PerceptionRuntimeProfile,
        )
        from runtime.route_registry import RouteRegistry
        from skills.manager import SkillManager, create_default_skill_registry
        from target.target_manager import TargetManager

        non_target_assignment_ids = (
            prepared.runtime_envelope_metadata.non_target_assignment_ids
        )
        assignments = {
            assignment.uav_id: assignment.target_alias
            for assignment in prepared.plan.assignments
            if assignment.assignment_id not in non_target_assignment_ids
        }
        environment = FleetUavSearchEnv(
            prepared.config,
            assignments=assignments,
        )
        environment.setup()
        if not prepared.headless and environment.scene is not None:
            environment.scene.configure_overview_viewport()

        profile = (
            PerceptionRuntimeProfile.ORACLE_EVALUATION
            if args.perception_runtime_profile == "oracle_evaluation"
            else PerceptionRuntimeProfile.PRODUCTION
        )
        agents: dict[str, object] = {}
        perceptions: dict[str, object] = {}
        clock = _FleetSimulationClock(environment)

        # One Broker owns admission for the entire Fleet runtime.  Visual
        # clients remain isolated per UAV, but no AsyncModelWorker is exposed
        # directly to a coordinator: each receives only its Broker facade.
        broker = GlobalModelRequestBroker(
            max_inflight_global=(
                prepared.config.model_broker.max_inflight_global
            ),
            max_inflight_per_uav=(
                prepared.config.model_broker.max_inflight_per_uav
            ),
            max_pending_per_uav=(
                prepared.config.model_broker.max_pending_per_uav
            ),
            starvation_timeout_s=(
                prepared.config.model_broker.starvation_timeout_s
            ),
        )
        brokered_visual_workers: dict[str, object] = {}
        visual_dispatcher: object | None = None
        if args.enable_qwen_vision:
            dispatcher, initial_visual_workers = (
                _build_brokered_visual_workers(prepared, broker, logger)
            )
            visual_dispatcher = dispatcher
            brokered_visual_workers.update(initial_visual_workers)
            workers.append(dispatcher)
        for assignment in prepared.plan.assignments:
            uav_id = assignment.uav_id
            if uav_id not in prepared.compilations:
                continue
            is_non_target_assignment = (
                assignment.assignment_id in non_target_assignment_ids
            )
            if is_non_target_assignment:
                # Explicit no-target input: never bind the synthetic envelope
                # alias to Oracle, detector/tracker, or target metrics.
                runtime_perception = DisabledTargetPerceptionBackend(uav_id=uav_id)
                backend_name = "disabled_non_target_assignment"
            elif profile is PerceptionRuntimeProfile.ORACLE_EVALUATION:
                raw_perception = environment.make_oracle_perception(uav_id)
                if (
                    raw_perception.uav_id != uav_id
                    or raw_perception.target_id != assignment.target_alias
                ):
                    raise RuntimeError("Oracle backend is outside its assignment")
                runtime_perception = GuardedPerceptionBackend(
                    raw_perception,
                    profile=profile,
                    acknowledge_privileged_oracle=True,
                )
                backend_name = "oracle_evaluation"
            else:
                raw_perception = build_target_perception_backend(
                    prepared.config,
                    runtime_profile=profile,
                    acknowledge_privileged_oracle=False,
                    uav_id=uav_id,
                )
                runtime_perception = raw_perception
                backend_name = prepared.config.target_perception.backend
            context = environment.make_skill_context(
                uav_id,
                clock,
                # Apply the same information-policy boundary to direct
                # SkillContext access and to FleetRuntime observations.
                perception=runtime_perception,
            )
            manager = SkillManager(
                context,
                registry=create_default_skill_registry(
                    transit_yaw_mode=prepared.config.search.transit_yaw_mode,
                ),
                route_registry=RouteRegistry(),
            )
            managers[uav_id] = manager
            transition_log_cursors[uav_id] = 0
            target_manager = TargetManager()
            visual_coordinator = None
            if args.enable_qwen_vision and not is_non_target_assignment:
                visual_coordinator = _build_visual_review_coordinator(
                    prepared=prepared,
                    uav_id=uav_id,
                    manager=manager,
                    target_manager=target_manager,
                    worker=brokered_visual_workers[uav_id],
                )
                visual_coordinators[uav_id] = visual_coordinator
                visual_log_cursors[uav_id] = 0
            world_context = prepared.world_contexts[uav_id]
            home_name = prepared.request.uav(uav_id).home_name
            validator = PlanValidator(
                prepared.planner_limits,
                prepared.planner_policy,
                spatial_resolver=_spatial_resolver(world_context, home_name),
            )
            safety = SafetySupervisor(
                world_context.scene_min_xyz_m,
                world_context.scene_max_xyz_m,
                max_mission_time_s=float(args.max_sim_time) + 120.0,
                position_margin_m=0.25,
                max_safe_altitude_m=world_context.scene_max_xyz_m[2],
                planner_limits=prepared.planner_limits,
            )
            replay = RoutedPreplannedSpatialPlanner(
                prepared.compilations[uav_id].planner_output,
                source=prepared.local_planner_source,
                expected_instruction=(
                    prepared.compilations[uav_id].planner_request.instruction
                    if prepared.fleet_plan_v2 is not None
                    else None
                ),
            )
            agents[uav_id] = MissionAgent(
                replay,
                validator,
                safety,
                manager,
                target_manager,
                clock,
                perception_runtime_profile=profile,
                acknowledge_privileged_oracle=bool(
                    args.acknowledge_privileged_oracle
                ),
                uav_id=uav_id,
                visual_review_coordinator=visual_coordinator,
                runtime_program=args.runtime_program,
                target_perception_backend=backend_name,
            )
            perceptions[uav_id] = runtime_perception

        def build_replanned_agent(
            record: object,
            assignment_v2: FleetAssignmentV2,
            compilation: object,
            world_context: PlannerWorldContext,
            runtime_assignment: FleetAssignment,
            route: tuple[tuple[float, float, float], ...],
        ) -> ReplannedAssignment:
            """Construct an idle-UAV Agent without starting or ticking it."""

            uav_id = runtime_assignment.uav_id
            source_assignment = getattr(record, "assignment")
            is_non_target_assignment = (
                source_assignment.assignment_id in non_target_assignment_ids
            )
            previous_assignments = dict(environment.assignments)
            provisional_assignments = dict(previous_assignments)
            provisional_assignments.pop(source_assignment.uav_id, None)
            if not is_non_target_assignment:
                provisional_assignments[uav_id] = runtime_assignment.target_alias
            environment.set_assignments(provisional_assignments)
            try:
                if is_non_target_assignment:
                    runtime_perception = DisabledTargetPerceptionBackend(
                        uav_id=uav_id
                    )
                    backend_name = "disabled_non_target_assignment"
                elif profile is PerceptionRuntimeProfile.ORACLE_EVALUATION:
                    raw_perception = environment.make_oracle_perception(uav_id)
                    runtime_perception = GuardedPerceptionBackend(
                        raw_perception,
                        profile=profile,
                        acknowledge_privileged_oracle=True,
                    )
                    backend_name = "oracle_evaluation"
                else:
                    raw_perception = build_target_perception_backend(
                        prepared.config,
                        runtime_profile=profile,
                        acknowledge_privileged_oracle=False,
                        uav_id=uav_id,
                    )
                    runtime_perception = raw_perception
                    backend_name = prepared.config.target_perception.backend
                context = environment.make_skill_context(
                    uav_id,
                    clock,
                    perception=runtime_perception,
                )
            finally:
                # No controller action occurs while the provisional Oracle
                # routing is visible.  Runtime publishes the same mapping only
                # after its own atomic validation gate succeeds.
                environment.set_assignments(previous_assignments)

            manager = SkillManager(
                context,
                registry=create_default_skill_registry(
                    transit_yaw_mode=prepared.config.search.transit_yaw_mode,
                ),
                route_registry=RouteRegistry(),
            )
            target_manager = TargetManager()
            visual_coordinator = None
            if args.enable_qwen_vision and not is_non_target_assignment:
                worker = brokered_visual_workers.get(uav_id)
                if worker is None:
                    worker_for = getattr(visual_dispatcher, "worker_for", None)
                    if not callable(worker_for):
                        raise FleetLaunchConfigurationError(
                            "runtime visual dispatcher cannot bind idle UAV"
                        )
                    worker = worker_for(
                        uav_id,
                        assignment_id=runtime_assignment.assignment_id,
                    )
                    brokered_visual_workers[uav_id] = worker
                visual_coordinator = _build_visual_review_coordinator(
                    prepared=prepared,
                    uav_id=uav_id,
                    manager=manager,
                    target_manager=target_manager,
                    worker=worker,
                )
                visual_coordinators[uav_id] = visual_coordinator
                visual_log_cursors[uav_id] = 0
            home_name = next(
                item.home_name
                for item in prepared.fleet_request_v2.uav_inventory
                if item.uav_id == uav_id
            )
            validator = PlanValidator(
                prepared.planner_limits,
                prepared.planner_policy,
                spatial_resolver=_spatial_resolver(world_context, home_name),
            )
            safety = SafetySupervisor(
                world_context.scene_min_xyz_m,
                world_context.scene_max_xyz_m,
                max_mission_time_s=float(args.max_sim_time) + 120.0,
                position_margin_m=0.25,
                max_safe_altitude_m=world_context.scene_max_xyz_m[2],
                planner_limits=prepared.planner_limits,
            )
            replay = RoutedPreplannedSpatialPlanner(
                compilation.planner_output,
                source="dynamic_llm",
                expected_instruction=compilation.planner_request.instruction,
            )
            agent = MissionAgent(
                replay,
                validator,
                safety,
                manager,
                target_manager,
                clock,
                perception_runtime_profile=profile,
                acknowledge_privileged_oracle=bool(
                    args.acknowledge_privileged_oracle
                ),
                uav_id=uav_id,
                visual_review_coordinator=visual_coordinator,
                runtime_program=args.runtime_program,
                target_perception_backend=backend_name,
            )
            # These are the same mutable containers drained by terminal logs
            # and metric collection, so a dynamically assigned UAV is visible
            # immediately after Runtime accepts the publication.
            managers[uav_id] = manager
            transition_log_cursors[uav_id] = 0
            perceptions[uav_id] = runtime_perception
            return ReplannedAssignment(
                assignment_id=source_assignment.assignment_id,
                replacement_assignment=runtime_assignment,
                agent=agent,
                start_input=(
                    compilation.planner_request.instruction,
                    compilation.planner_request.world_context,
                ),
                perception=runtime_perception,
                planned_route=route,
                degraded=not compilation.semantically_valid,
                uncovered_goal_ids=compilation.uncovered_goal_ids,
            )

        replan_handler = _build_runtime_fleet_replan_handler(
            prepared,
            audit=result_recorder.planning_audit,
            agent_factory=build_replanned_agent,
        )

        targets = SharedTargetRegistry(
            prepared.plan.coordination_policy.target_claim_policy.value
        )
        airspace = FleetAirspaceManager(
            prepared.plan.coordination_policy.minimum_uav_separation_m,
            policy=prepared.plan.coordination_policy.route_conflict_policy.value,
        )
        runtime = FleetMissionRuntime(
            environment,
            RoutedPreplannedFleetPlanner(
                prepared.request,
                prepared.plan,
                source=prepared.fleet_planner_source,
            ),
            agents,
            assignment_compiler=prepared.compiler,
            world_contexts=prepared.world_contexts,
            precomputed_start_inputs=(
                {
                    uav_id: (
                        result.planner_request.instruction,
                        result.planner_request.world_context,
                    )
                    for uav_id, result in prepared.compilations.items()
                }
                if prepared.fleet_plan_v2 is not None
                else None
            ),
            perceptions=perceptions,
            planned_routes=prepared.planned_routes,
            targets=targets,
            airspace=airspace,
            model_broker=broker,
            logger=logger,
            initial_assignment_states=prepared.initial_assignment_states,
            non_target_assignment_ids=tuple(non_target_assignment_ids),
            assignment_requiredness=(
                prepared.runtime_envelope_metadata.required_by_assignment
            ),
            replan_handler=replan_handler,
        )
        runtime.start(args.instruction.strip(), request=prepared.request)

        # Persist the real NONE -> TAKEOFF records produced by MissionAgent;
        # later drains append only newly emitted records.
        _drain_runtime_logs(
            logger,
            visual_coordinators,
            visual_log_cursors,
            managers,
            transition_log_cursors,
        )

        if args.debug_visualization:
            debug_draw = _create_fleet_debug_visualizer(
                prepared.plan,
                runtime.snapshot().agent_plan_versions,
                headless=prepared.headless,
            )

        cancel_requested = False
        shutdown_deadline_s = float(args.max_sim_time) + 120.0
        last_printed_second = -1
        while simulation_app.is_running() and runtime.status is FleetStatus.RUNNING:
            snapshot = runtime.tick()
            now_s = clock.now()
            try:
                _record_fleet_state_samples(
                    result_recorder,
                    prepared,
                    runtime,
                    environment,
                )
            except (RuntimeError, TypeError, ValueError):
                # Warm-up snapshots can be incomplete.  State logging is
                # observational and never controls mission execution.
                pass
            whole_second = int(now_s)
            if whole_second != last_printed_second:
                last_printed_second = whole_second
                statuses = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(snapshot.agent_statuses.items())
                )
                print(
                    f"[Fleet] t={now_s:.1f}s status={snapshot.status.value} {statuses}"
                )
            if (
                not cancel_requested
                and runtime.status is FleetStatus.RUNNING
                and now_s >= float(args.max_sim_time)
            ):
                print(
                    "[Fleet] max-sim-time reached; requesting cancel-and-land",
                    file=sys.stderr,
                )
                runtime.cancel()
                cancel_requested = True
            if cancel_requested and now_s >= shutdown_deadline_s:
                raise RuntimeError("Fleet cancel-and-land exceeded shutdown guard")

            _drain_runtime_logs(
                logger,
                visual_coordinators,
                visual_log_cursors,
                managers,
                transition_log_cursors,
            )

            if debug_draw is not None:
                try:
                    pose_snapshot = coerce_fleet_pose_snapshot(
                        environment.get_fleet_pose_snapshot()
                    )
                    fleet_pose = environment.get_fleet_pose_snapshot()
                    target_positions = {
                        target_id: (
                            state.x,
                            state.y,
                            state.z,
                        )
                        for target_id, state in fleet_pose.target_states.items()
                    }
                    debug_draw.update(
                        poses=pose_snapshot,
                        target_records=runtime.targets.records,
                        target_positions_world_m=target_positions,
                        airspace_decision=runtime.airspace.last_decision,
                        agent_plan_versions=snapshot.agent_plan_versions,
                    )
                except RuntimeError:
                    # Camera/pose warm-up may leave no complete first snapshot.
                    pass

        final_snapshot = runtime.snapshot()
        exit_code = 0 if final_snapshot.status is FleetStatus.SUCCEEDED else 1
        print_terminal_summary = True
    except KeyboardInterrupt:
        print("[Fleet] interrupted", file=sys.stderr)
        terminal_error = "KeyboardInterrupt: fleet mission interrupted"
        interrupted = True
        exit_code = 130
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
        terminal_error = _redact_terminal_error(
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        try:
            exit_code = _finalize_fleet_execution(
                prepared,
                logger,
                runtime=runtime,
                simulation_app=simulation_app,
                environment=environment,
                clock=clock,
                debug_draw=debug_draw,
                workers=workers,
                visual_coordinators=visual_coordinators,
                visual_log_cursors=visual_log_cursors,
                managers=managers,
                transition_log_cursors=transition_log_cursors,
                exit_code=exit_code,
                terminal_error=terminal_error,
                interrupted=interrupted,
                print_terminal_summary=print_terminal_summary,
                primary_error_present=primary_error is not None,
                result_recorder=result_recorder,
                result_wall_time_s=time.monotonic() - run_wall_started,
                no_summary_figures=bool(args.no_summary_figures),
            )
        except BaseException as exc:
            # This is a final containment boundary.  The helper handles every
            # expected stage independently; an internal bug here still cannot
            # replace the original mission exception or an earlier Ctrl+C.
            print(
                "[Fleet] terminal finalization failed unexpectedly: "
                + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
                file=sys.stderr,
            )
            if primary_error is None and not interrupted:
                exit_code = 1
    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)
    return exit_code


def _sanitized_command(argv: Sequence[str] | None) -> list[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    safe: list[str] = []
    index = 0
    while index < len(raw):
        item = str(raw[index])
        if item == "--api-key":
            index += 2
            continue
        if item.casefold().startswith("--api-key="):
            index += 1
            continue
        safe.append(_redact_terminal_error(item))
        index += 1
    return [sys.executable, "-u", str(Path(__file__).resolve()), *safe]


def _initial_run_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "instruction_length": len(args.instruction),
        "fleet_planner": args.fleet_planner,
        "mission_interpreter": args.mission_interpreter,
        "local_planner": args.local_planner,
        "planning_contract": args.planning_contract,
        "perception_runtime_profile": args.perception_runtime_profile,
        "headless": args.headless,
    }


def _persist_failed_preparation_audit(
    recorder: FleetResultRecorder,
    args: argparse.Namespace,
) -> None:
    """Flush whatever structured planning state existed before preparation failed."""

    raw_context = getattr(args, "_preparation_audit_context", None)
    if not isinstance(raw_context, Mapping):
        return
    mission_id = raw_context.get("fleet_mission_id")
    if not isinstance(mission_id, str):
        return
    source_text = raw_context.get("source_text", args.instruction)
    if not isinstance(source_text, str):
        source_text = str(source_text)
    audit = recorder.planning_audit

    interpreter_proposals = raw_context.get("interpreter_proposals", ())
    if isinstance(interpreter_proposals, Sequence) and not isinstance(
        interpreter_proposals, (str, bytes)
    ):
        _record_proposal_stage(
            audit,
            mission_id=mission_id,
            stage="MISSION_INTERPRETATION",
            role=ModelCallRole.MISSION_INTERPRETATION.value,
            schema="fleet_task_spec_v1",
            prompt_material=source_text,
            proposals=tuple(
                item for item in interpreter_proposals if isinstance(item, Mapping)
            ),
        )

    task_spec = raw_context.get("task_spec")
    live_logger = getattr(args, "_preparation_fleet_logger", None)
    if isinstance(task_spec, FleetTaskSpecV1):
        audit.write_final_plan(
            stage="MISSION_INTERPRETATION",
            mission_id=mission_id,
            plan_version=1,
            plan=task_spec.to_dict(),
        )
        if live_logger is not None:
            live_logger.write_task_spec(task_spec)

    request_v2 = raw_context.get("request_v2")
    fleet_proposals = raw_context.get("fleet_planner_proposals", ())
    if isinstance(request_v2, FleetMissionRequestV2):
        prompt_material = json.dumps(
            request_v2.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if isinstance(fleet_proposals, Sequence) and not isinstance(
            fleet_proposals, (str, bytes)
        ):
            _record_proposal_stage(
                audit,
                mission_id=mission_id,
                stage="FLEET_PLANNING",
                role=ModelCallRole.FLEET_PLAN.value,
                schema="fleet_mission_plan_v2",
                prompt_material=prompt_material,
                proposals=tuple(
                    item for item in fleet_proposals if isinstance(item, Mapping)
                ),
            )

    plan_v2 = raw_context.get("plan_v2")
    if isinstance(plan_v2, FleetMissionPlanV2):
        audit.write_final_plan(
            stage="FLEET_PLANNING",
            mission_id=mission_id,
            plan_version=plan_v2.fleet_plan_version,
            plan=plan_v2.to_dict(),
        )
        if live_logger is not None:
            live_logger.write_fleet_plan(plan_v2)

    assignment_by_uav = {
        item.uav_id: item
        for item in (() if not isinstance(plan_v2, FleetMissionPlanV2) else plan_v2.assignments)
    }
    raw_compilations = raw_context.get("compilations", {})
    compilations = (
        raw_compilations if isinstance(raw_compilations, Mapping) else {}
    )
    raw_local = raw_context.get("local_planner_proposals", {})
    local = raw_local if isinstance(raw_local, Mapping) else {}
    for raw_uav_id, raw_proposals in sorted(local.items(), key=lambda item: str(item[0])):
        uav_id = str(raw_uav_id)
        assignment = assignment_by_uav.get(uav_id)
        result = compilations.get(uav_id)
        prompt_material = (
            result.planner_request.instruction
            if result is not None and hasattr(result, "planner_request")
            else json.dumps(
                {
                    "uav_id": uav_id,
                    "goal_ids": (
                        [] if assignment is None else list(assignment.goal_ids)
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        proposals = (
            tuple(item for item in raw_proposals if isinstance(item, Mapping))
            if isinstance(raw_proposals, Sequence)
            and not isinstance(raw_proposals, (str, bytes))
            else ()
        )
        _record_proposal_stage(
            audit,
            mission_id=mission_id,
            stage="LOCAL_PLANNING",
            role=ModelCallRole.AGENT_SPATIAL_PLAN.value,
            schema="spatial_skill_plan_v3",
            prompt_material=prompt_material,
            proposals=proposals,
            assignment_id=(None if assignment is None else assignment.assignment_id),
            uav_id=uav_id,
        )
        if result is not None and hasattr(result, "planner_output"):
            audit.write_final_plan(
                stage="LOCAL_PLANNING",
                mission_id=mission_id,
                assignment_id=(
                    None if assignment is None else assignment.assignment_id
                ),
                uav_id=uav_id,
                plan_version=result.agent_request.local_plan_version,
                plan=result.planner_output.to_dict(),
            )


def _record_preparation_failure(
    run_manager: RunManager,
    args: argparse.Namespace,
    exc: BaseException,
    *,
    interrupted: bool,
) -> int:
    exit_code = 130 if interrupted else 2
    error = _redact_terminal_error(f"{type(exc).__name__}: {exc}")
    name = type(exc).__name__
    stage = (
        "MISSION_INTERPRETATION"
        if "Interpret" in name or "TaskSpec" in name
        else "FLEET_PLANNING"
        if "FleetPlanner" in name
        else "PREFLIGHT"
    )
    status = "CANCELED" if interrupted else "FAILED_PREPARATION"
    recorder = FleetResultRecorder(run_manager.paths.run_dir)
    preparation_audit_error: str | None = None
    try:
        _persist_failed_preparation_audit(recorder, args)
    except Exception as audit_exc:
        preparation_audit_error = _redact_terminal_error(
            f"{type(audit_exc).__name__}: {audit_exc}"
        )
        print(
            "[Fleet] preparation audit persistence failed: "
            + preparation_audit_error,
            file=sys.stderr,
        )
    recorder.record_failure(
        {
            "run_id": run_manager.run_id,
            "stage": stage,
            "code": name.upper(),
            "severity": "HARD_ACTION_BLOCK",
            "status": status,
            "message": error,
        }
    )
    recorder.finalize(
        {
            "schema_version": 1,
            "status": status,
            "stage": stage,
            "last_error": error,
            "exit_code": exit_code,
            "interrupted": interrupted,
            "isaac_started": False,
            "goal_count": 0,
            "goals_completed": 0,
            "preparation_audit_error": preparation_audit_error,
        }
    )
    try:
        try:
            generate_fleet_report(
                run_manager.paths.run_dir,
                no_summary_figures=bool(args.no_summary_figures),
            )
        except Exception as report_exc:
            print(
                "[Fleet] preparation failure report generation failed: "
                + _redact_terminal_error(
                    f"{type(report_exc).__name__}: {report_exc}"
                ),
                file=sys.stderr,
            )
    finally:
        recorder.close()
    if interrupted:
        run_manager.interrupt(exit_code=exit_code)
    else:
        run_manager.fail(exit_code=exit_code, failure_reason=name)
    print(
        ("fleet mission interrupted: " if interrupted else "fleet launch configuration error: ")
        + error,
        file=sys.stderr,
    )
    return exit_code


def _ensure_runtime_failure_result(
    run_manager: RunManager,
    prepared: object,
    args: argparse.Namespace,
    exc: BaseException,
    *,
    interrupted: bool,
) -> None:
    """Best-effort outer containment for failures before runtime's own try block."""

    summary_path = run_manager.paths.run_dir / "summary.json"
    if summary_path.is_file():
        return
    exit_code = 130 if interrupted else 1
    status = "CANCELED" if interrupted else "FAILED"
    error = _redact_terminal_error(f"{type(exc).__name__}: {exc}")
    request = getattr(prepared, "request", None)
    mission_id = getattr(request, "fleet_mission_id", run_manager.run_id)
    summary = {
        "schema_version": 1,
        "fleet_mission_id": mission_id,
        "status": status,
        "stage": "RUNTIME",
        "last_error": error,
        "exit_code": exit_code,
        "interrupted": interrupted,
        # This boundary can be reached either just before or during the first
        # Isaac import, so it must not invent a boolean launch fact.
        "isaac_started": None,
        "goal_count": 0,
        "goals_completed": 0,
    }
    recorder: FleetResultRecorder | None = None
    try:
        recorder = FleetResultRecorder(
            run_manager.paths.run_dir,
            fleet_mission_id=(mission_id if isinstance(mission_id, str) else None),
        )
        recorder.record_failure(
            {
                "run_id": run_manager.run_id,
                "stage": "RUNTIME",
                "code": type(exc).__name__.upper(),
                "severity": "HARD_ACTION_BLOCK",
                "status": status,
                "message": error,
            }
        )
        recorder.finalize(summary)
    except Exception as result_exc:
        # The minimal atomic JSON fallback has no dependency on logger layout
        # or CSV schemas.  It is used only when the bounded recorder itself is
        # the component that failed.
        fallback = {
            **summary,
            "result_output_error": _redact_terminal_error(
                f"{type(result_exc).__name__}: {result_exc}"
            ),
        }
        temporary = summary_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    fallback,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(summary_path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        if recorder is not None:
            recorder.close()
    try:
        generate_fleet_report(
            run_manager.paths.run_dir,
            no_summary_figures=bool(args.no_summary_figures),
        )
    except Exception as report_exc:
        print(
            "[Fleet] runtime failure report generation failed: "
            + _redact_terminal_error(
                f"{type(report_exc).__name__}: {report_exc}"
            ),
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_manager = RunManager.create(
            experiment_name="fleet_mission",
            seed=0,
            resolved_config=_initial_run_config(args),
            output_root=args.output_root,
            project_root=_PROJECT_ROOT,
            stage="fleet_mission",
            command=_sanitized_command(argv),
            min_free_space_gb_before_start=0.0,
            # The user explicitly allows Git work to be skipped for this task.
            git_metadata={"commit": None, "branch": None, "dirty": None},
        )
    except Exception as exc:
        print(
            "fleet result directory creation failed: "
            + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
            file=sys.stderr,
        )
        return 2

    setattr(args, "_managed_run_dir", run_manager.paths.run_dir)
    with TerminalLogger(
        run_manager.paths.terminal_log,
        max_log_bytes=1_048_576,
    ):
        print(f"[Fleet] result_dir={run_manager.paths.run_dir}")
        try:
            prepared = prepare_fleet_mission(args)
            if hasattr(prepared, "config"):
                run_manager.update_resolved_config(prepared.config)
        except KeyboardInterrupt as exc:
            return _record_preparation_failure(
                run_manager, args, exc, interrupted=True
            )
        except Exception as exc:
            return _record_preparation_failure(
                run_manager, args, exc, interrupted=False
            )
        try:
            exit_code = run_prepared_fleet_mission(prepared, args)
        except KeyboardInterrupt as exc:
            error = _redact_terminal_error(f"{type(exc).__name__}: {exc}")
            print("fleet mission interrupted: " + error, file=sys.stderr)
            _ensure_runtime_failure_result(
                run_manager,
                prepared,
                args,
                exc,
                interrupted=True,
            )
            run_manager.interrupt(exit_code=130)
            return 130
        except Exception as exc:
            print(
                "fleet mission failed: "
                + _redact_terminal_error(f"{type(exc).__name__}: {exc}"),
                file=sys.stderr,
            )
            _ensure_runtime_failure_result(
                run_manager,
                prepared,
                args,
                exc,
                interrupted=False,
            )
            run_manager.fail(
                exit_code=1,
                failure_reason=type(exc).__name__,
            )
            return 1
        if exit_code == 0:
            run_manager.complete(exit_code=0)
        elif exit_code == 130:
            run_manager.interrupt(exit_code=130)
        else:
            run_manager.fail(
                exit_code=exit_code,
                failure_reason="FLEET_MISSION_FAILED",
            )
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
