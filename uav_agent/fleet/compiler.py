"""Compile independent Fleet assignments through existing Spatial V3 planners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json

from common.ids import validate_routing_id
from planner.goal_checker import GoalSatisfactionChecker
from planner.schemas import PlannerRequest, PlannerWorldContext
from planner.schemas_v3 import SkillPlanDraftV3
from planner.spatial import NamedLocationTarget
from runtime.plan_validator import PlanValidator
from runtime.validation_codes import ValidationCode
from runtime.validation_report import (
    RecoveryRecommendation,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
)
from target.types import TargetSpec

from fleet.planner_base import FleetPlannerError
from fleet.schemas import validate_fleet_mission_plan
from fleet.schemas_v2 import validate_fleet_mission_plan_v2
from fleet.types import (
    AgentPlannerRequest,
    AssignmentCompilation,
    FleetAssignment,
    FleetMissionPlan,
    FleetMissionRequest,
    FleetUavCapability,
)
from fleet.types_v2 import (
    AgentPlannerRequestV2,
    AssignmentCompilationV2,
    FleetAssignmentV2,
    FleetMissionPlanV2,
    FleetMissionRequestV2,
    FleetSafetySummaryEntry,
)
from fleet.world_belief import AgentFleetSummary


_MAX_SEMANTIC_REPAIR_FINDINGS = 32
_MAX_SEMANTIC_REPAIR_MESSAGE_CHARS = 512
_MAX_PROPOSAL_REPAIR_FINDINGS = 32
_MAX_PROPOSAL_REPAIR_MESSAGE_CHARS = 512
_SEMANTIC_REPAIR_CODES = frozenset(
    {
        ValidationCode.GOAL_NOT_COVERED,
        ValidationCode.GOAL_PATH_INFEASIBLE,
        ValidationCode.TRACK_DURATION_UNDERSHOOT,
        ValidationCode.WAIT_DURATION_UNDERSHOOT,
        ValidationCode.RETURN_HOME_NOT_COVERED,
        ValidationCode.LAND_NOT_COVERED,
        ValidationCode.UNSUPPORTED_GOAL_TYPE,
        ValidationCode.AMBIGUOUS_GOAL,
    }
)
_PROPOSAL_REPAIR_CODES = frozenset(
    {
        ValidationCode.INVALID_JSON.value,
        ValidationCode.SCHEMA_INVALID.value,
        ValidationCode.UNKNOWN_FIELD.value,
        ValidationCode.UNKNOWN_SKILL.value,
        ValidationCode.UNKNOWN_ENTITY.value,
        ValidationCode.NON_FINITE_NUMBER.value,
        ValidationCode.ROUTING_MISMATCH.value,
        ValidationCode.PLAN_VERSION_MISMATCH.value,
        ValidationCode.STEP_REFERENCE_INVALID.value,
        ValidationCode.CALL_LIMIT_EXCEEDED.value,
        ValidationCode.LOW_LEVEL_CONTROL_FORBIDDEN.value,
        ValidationCode.ORACLE_FIELD_FORBIDDEN.value,
        ValidationCode.OUT_OF_BOUNDS_GOTO.value,
        ValidationCode.INVALID_LANDING_ZONE.value,
        ValidationCode.UNSAFE_ACTION.value,
        ValidationCode.INTERNAL_ERROR.value,
        # Stable parser/catalog diagnostics emitted by DynamicLLMPlanner
        # before a ValidationReport exists.
        "CATALOG_CONTRACT_VIOLATION",
        "V3_CONTRACT_VIOLATION",
        "MODEL_CLIENT_ERROR",
        "ROUTING_IDS_REQUIRED",
    }
)
_PROPOSAL_REPAIR_FORBIDDEN_MESSAGE_TOKENS = (
    "oracle",
    "base64",
    "data:image",
    "data:video",
    "prompt",
    "raw_output",
    "raw proposal",
    "observation",
    "private",
    "secret",
    "api_key",
    "authorization",
    "bearer",
    "password",
    "hidden reasoning",
    "-----begin",
)


class AssignmentCompilerError(FleetPlannerError):
    """Raised when one local assignment cannot be isolated or compiled."""


class FleetAssignmentCompiler:
    """Create one routed local PlannerRequest per assignment.

    The local planner receives a focused instruction and a world context that
    contains only its UAV's home.  It never receives another assignment,
    target semantics, image, or Oracle state.
    """

    def __init__(
        self,
        local_planner: object | Mapping[str, object],
        *,
        validator: PlanValidator | None = None,
        goal_checker: GoalSatisfactionChecker | None = None,
    ) -> None:
        if isinstance(local_planner, Mapping):
            planners = dict(local_planner)
            if not planners or any(
                not isinstance(key, str)
                or not callable(getattr(value, "plan", None))
                for key, value in planners.items()
            ):
                raise TypeError(
                    "local_planner mapping must contain planners with plan()"
                )
            self._planners: dict[str, object] | None = planners
            self._planner: object | None = None
        else:
            if not callable(getattr(local_planner, "plan", None)):
                raise TypeError("local_planner must provide plan()")
            self._planner = local_planner
            self._planners = None
        if validator is not None and not isinstance(validator, PlanValidator):
            raise TypeError("validator must be a PlanValidator or None")
        if goal_checker is not None and not isinstance(
            goal_checker, GoalSatisfactionChecker
        ):
            raise TypeError(
                "goal_checker must be a GoalSatisfactionChecker or None"
            )
        self._validator = validator
        self._goal_checker = goal_checker or GoalSatisfactionChecker()

    def build_agent_request(
        self,
        fleet_request: FleetMissionRequest,
        fleet_plan: FleetMissionPlan,
        assignment: FleetAssignment,
        *,
        local_plan_version: int = 1,
    ) -> AgentPlannerRequest:
        if not isinstance(fleet_request, FleetMissionRequest):
            raise TypeError("fleet_request must be a FleetMissionRequest")
        if not isinstance(fleet_plan, FleetMissionPlan):
            raise TypeError("fleet_plan must be a FleetMissionPlan")
        if not isinstance(assignment, FleetAssignment):
            raise TypeError("assignment must be a FleetAssignment")
        validate_fleet_mission_plan(fleet_plan, fleet_request)
        selected = next(
            (
                item
                for item in fleet_plan.assignments
                if item.assignment_id == assignment.assignment_id
            ),
            None,
        )
        if selected is None or selected != assignment:
            raise AssignmentCompilerError(
                "assignment is not an exact member of FleetMissionPlan"
            )
        safety_summary = tuple(
            AgentFleetSummary(
                uav_id=other.uav_id,
                assignment_id=other.assignment_id,
                status="PLANNED",
                plan_version=1,
                current_region=getattr(
                    other.search_region.shape,
                    "value",
                    str(other.search_region.shape),
                ),
                altitude_layer=None,
            )
            for other in fleet_plan.assignments
            if other.uav_id != assignment.uav_id
        )
        return AgentPlannerRequest(
            fleet_mission_id=fleet_plan.fleet_mission_id,
            assignment_id=assignment.assignment_id,
            uav_id=assignment.uav_id,
            original_instruction=fleet_request.original_instruction,
            assignment=assignment,
            target_spec=assignment.target_spec,
            search_region=assignment.search_region,
            track_duration_s=assignment.track_duration_s,
            local_plan_version=local_plan_version,
            fleet_safety_summary=safety_summary,
        )

    def build_agent_request_v2(
        self,
        fleet_request: FleetMissionRequestV2,
        fleet_plan: FleetMissionPlanV2,
        assignment: FleetAssignmentV2,
        *,
        local_plan_version: int = 1,
        target_catalog: Mapping[str, TargetSpec] | None = None,
    ) -> AgentPlannerRequestV2:
        """Project one V2 Assignment onto only its referenced semantic Goals."""

        if not isinstance(fleet_request, FleetMissionRequestV2):
            raise TypeError("fleet_request must be a FleetMissionRequestV2")
        if not isinstance(fleet_plan, FleetMissionPlanV2):
            raise TypeError("fleet_plan must be a FleetMissionPlanV2")
        if not isinstance(assignment, FleetAssignmentV2):
            raise TypeError("assignment must be a FleetAssignmentV2")
        validate_fleet_mission_plan_v2(fleet_plan, fleet_request)
        selected = next(
            (
                item
                for item in fleet_plan.assignments
                if item.assignment_id == assignment.assignment_id
            ),
            None,
        )
        if selected is None or selected != assignment:
            raise AssignmentCompilerError(
                "assignment is not an exact member of FleetMissionPlanV2"
            )
        trusted_targets = self._assignment_target_specs(
            fleet_request,
            assignment,
            target_catalog,
        )
        safety_summary = tuple(
            FleetSafetySummaryEntry(
                uav_id=other.uav_id,
                assignment_id=other.assignment_id,
                status="PLANNED",
            )
            for other in fleet_plan.assignments
            if other.uav_id != assignment.uav_id
        )
        return AgentPlannerRequestV2.for_assignment(
            fleet_request,
            assignment,
            local_plan_version=local_plan_version,
            fleet_safety_summary=safety_summary,
            trusted_target_specs=trusted_targets,
        )

    def build_planner_request(
        self,
        agent_request: AgentPlannerRequest,
        world_context: PlannerWorldContext,
        capability: FleetUavCapability,
    ) -> PlannerRequest:
        if not isinstance(agent_request, AgentPlannerRequest):
            raise TypeError("agent_request must be an AgentPlannerRequest")
        if not isinstance(world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")
        if not isinstance(capability, FleetUavCapability):
            raise TypeError("capability must be a FleetUavCapability")
        if capability.uav_id != agent_request.uav_id:
            raise AssignmentCompilerError(
                "capability.uav_id does not match AgentPlannerRequest"
            )
        try:
            home = world_context.landing_zones[capability.home_name]
        except KeyError:
            raise AssignmentCompilerError(
                f"local world context is missing home: {capability.home_name}"
            ) from None
        isolated_context = PlannerWorldContext(
            scene_min_xyz_m=world_context.scene_min_xyz_m,
            scene_max_xyz_m=world_context.scene_max_xyz_m,
            initial_uav_xyz_m=world_context.initial_uav_xyz_m,
            # The assignment's RegionSpec is explicit in the focused request;
            # named regions belonging to other assignments stay hidden.
            search_regions={},
            landing_zones={capability.home_name: home},
            navigation_points={
                key: value
                for key, value in world_context.navigation_points.items()
                if key == capability.home_name
            },
            default_takeoff_altitude_m=world_context.default_takeoff_altitude_m,
            default_track_duration_s=agent_request.track_duration_s,
            search_timeout_s=world_context.search_timeout_s,
            goto_timeout_s=world_context.goto_timeout_s,
            land_timeout_s=world_context.land_timeout_s,
        )
        focused_payload = {
            "task": "Create one independent Spatial SkillPlanDraftV3.",
            "assignment_id": agent_request.assignment_id,
            "uav_id": agent_request.uav_id,
            "target_alias": agent_request.assignment.target_alias,
            "target_spec": agent_request.target_spec.to_dict(),
            "search_region": agent_request.search_region.to_dict(),
            "track_duration_s": agent_request.track_duration_s,
            "return_home": capability.home_name,
            "requirements": [
                "Plan only this assignment.",
                (
                    "Copy the input target_spec object byte-for-field into the "
                    "top-level output field target_spec. It is required: do not "
                    "omit, summarize, translate, or modify it."
                ),
                "Preserve RegionSpec and track duration exactly.",
                "Take off, search, track, return to the listed home, and land.",
                (
                    "Set top-level assumptions to []. This request already supplies "
                    "trusted structured RegionSpec geometry and contains no natural-"
                    "language spatial ambiguity to reinterpret. Do not turn JSON "
                    "field names or values into SpatialAssumption.source_text."
                ),
            ],
            "required_spatial_assumptions": [],
            "fleet_safety_summary": [
                item.to_dict() for item in agent_request.fleet_safety_summary
            ],
        }
        instruction = json.dumps(
            focused_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return PlannerRequest(
            instruction=instruction,
            world_context=isolated_context,
            mission_id=agent_request.agent_mission_id,
            uav_id=agent_request.uav_id,
            plan_version=agent_request.local_plan_version,
            trusted_target_spec=agent_request.target_spec,
            require_empty_spatial_assumptions=True,
        )

    def build_planner_request_v2(
        self,
        agent_request: AgentPlannerRequestV2,
        world_context: PlannerWorldContext,
        capability: FleetUavCapability,
        *,
        trusted_target_id: str | None = None,
        semantic_repair_findings: Sequence[Mapping[str, object]] = (),
        proposal_repair_findings: Sequence[Mapping[str, object]] = (),
    ) -> PlannerRequest:
        """Create a focused local prompt without prescribing a Skill chain."""

        if not isinstance(agent_request, AgentPlannerRequestV2):
            raise TypeError("agent_request must be an AgentPlannerRequestV2")
        if not isinstance(world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")
        if not isinstance(capability, FleetUavCapability):
            raise TypeError("capability must be a FleetUavCapability")
        if capability.uav_id != agent_request.uav_id:
            raise AssignmentCompilerError(
                "capability.uav_id does not match AgentPlannerRequestV2"
            )
        try:
            home = world_context.landing_zones[capability.home_name]
        except KeyError:
            raise AssignmentCompilerError(
                f"local world context is missing home: {capability.home_name}"
            ) from None

        relevant_names = {capability.home_name}
        for goal in agent_request.goals:
            spatial = getattr(goal, "spatial_constraint", None)
            if isinstance(spatial, NamedLocationTarget):
                relevant_names.add(spatial.name)
            reference_id = getattr(spatial, "reference_id", None)
            if isinstance(reference_id, str):
                relevant_names.add(reference_id)
        isolated_context = PlannerWorldContext(
            scene_min_xyz_m=world_context.scene_min_xyz_m,
            scene_max_xyz_m=world_context.scene_max_xyz_m,
            initial_uav_xyz_m=world_context.initial_uav_xyz_m,
            # V2 Goals carry their own explicit Spatial V3 values.  Named
            # regions belonging to another Assignment remain invisible.
            search_regions={},
            landing_zones={capability.home_name: home},
            navigation_points={
                name: value
                for name, value in world_context.navigation_points.items()
                if name in relevant_names
            },
            default_takeoff_altitude_m=(
                world_context.default_takeoff_altitude_m
            ),
            default_track_duration_s=world_context.default_track_duration_s,
            search_timeout_s=world_context.search_timeout_s,
            goto_timeout_s=world_context.goto_timeout_s,
            land_timeout_s=world_context.land_timeout_s,
        )
        repair_findings = self._semantic_repair_findings(
            semantic_repair_findings
        )
        structural_repair_findings = self._proposal_repair_findings(
            proposal_repair_findings
        )
        focused_payload: dict[str, object] = {
            "schema_version": 2,
            "task": (
                "Create one independent Spatial SkillPlanDraftV3 that covers "
                "the assigned semantic Goals."
            ),
            "assignment_id": agent_request.assignment_id,
            "uav_id": agent_request.uav_id,
            "assigned_goals": [goal.to_dict() for goal in agent_request.goals],
            "trusted_target_specs": {
                alias: spec.to_dict()
                for alias, spec in agent_request.trusted_target_specs.items()
            },
            "own_home": capability.home_name,
            # The model owns only assigned semantic Goals.  The trusted
            # compiler, not Qwen, closes an otherwise-airborne plan with a
            # bounded return/LAND epilogue after Goal coverage is measured.
            "trusted_runtime_safety_completion": True,
            "requirements": [
                "Plan only the assigned_goals in this request.",
                (
                    "Choose legal Skill call counts, order, and arguments "
                    "autonomously within the supplied catalog and limits."
                ),
                (
                    "Do not invent unassigned semantic Goals or change trusted "
                    "routing, Goal fields, or target specifications."
                ),
            ],
            "fleet_safety_summary": [
                item.to_dict() for item in agent_request.fleet_safety_summary
            ],
        }
        if repair_findings:
            # Only the bounded verdict is fed back.  The rejected proposal and
            # prior prompt are deliberately absent from the retry request.
            focused_payload["semantic_repair_findings"] = repair_findings
        if structural_repair_findings:
            # This channel is deliberately separate from Goal coverage.  It
            # contains only a bounded code/message verdict: never the prior
            # prompt, raw model response, media, or private/Oracle state.
            focused_payload["proposal_repair_findings"] = (
                structural_repair_findings
            )
        instruction = json.dumps(
            focused_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        target_specs = tuple(agent_request.trusted_target_specs.values())
        return PlannerRequest(
            instruction=instruction,
            world_context=isolated_context,
            mission_id=agent_request.agent_mission_id,
            uav_id=agent_request.uav_id,
            plan_version=agent_request.local_plan_version,
            # Spatial V3 currently exposes one optional top-level TargetSpec.
            # Bind it only when the Assignment has exactly one trusted target;
            # the focused prompt still carries the bounded alias map otherwise.
            trusted_target_spec=(
                target_specs[0] if len(target_specs) == 1 else None
            ),
            trusted_target_id=trusted_target_id,
            require_empty_spatial_assumptions=False,
            allow_trusted_safety_completion=True,
        )

    def compile_assignment(
        self,
        fleet_request: FleetMissionRequest,
        fleet_plan: FleetMissionPlan,
        assignment: FleetAssignment,
        world_context: PlannerWorldContext,
        *,
        local_plan_version: int = 1,
        spatial_resolver: object | None = None,
    ) -> AssignmentCompilation:
        agent_request = self.build_agent_request(
            fleet_request,
            fleet_plan,
            assignment,
            local_plan_version=local_plan_version,
        )
        capability = fleet_request.uav(assignment.uav_id)
        planner_request = self.build_planner_request(
            agent_request, world_context, capability
        )
        planner = self._planner_for(assignment.uav_id)
        output = planner.plan(planner_request)
        self._validate_local_output(output, agent_request)
        compiled = None
        if self._validator is not None:
            source = getattr(planner, "source", None)
            if source not in {"dynamic_scripted", "dynamic_llm"}:
                raise AssignmentCompilerError(
                    "Spatial V3 validation requires local planner source "
                    "dynamic_scripted or dynamic_llm"
                )
            kwargs: dict[str, object] = {}
            if spatial_resolver is not None:
                kwargs["spatial_resolver"] = spatial_resolver
            compiled = self._validator.validate_and_compile(
                output,
                planner_request.world_context,
                source=source,
                mission_id=planner_request.mission_id,
                uav_id=planner_request.uav_id,
                plan_version=planner_request.plan_version,
                allow_trusted_safety_completion=(
                    planner_request.allow_trusted_safety_completion
                ),
                **kwargs,
            )
        return AssignmentCompilation(
            agent_request=agent_request,
            planner_request=planner_request,
            planner_output=output,
            compiled_mission=compiled,
        )

    def compile_assignment_v2(
        self,
        fleet_request: FleetMissionRequestV2,
        fleet_plan: FleetMissionPlanV2,
        assignment: FleetAssignmentV2,
        world_context: PlannerWorldContext,
        *,
        local_plan_version: int = 1,
        target_catalog: Mapping[str, TargetSpec] | None = None,
        spatial_resolver: object | None = None,
        trusted_target_id: str | None = None,
        timestamp: float = 0.0,
        proposal_id: str | None = None,
        semantic_repair_findings: Sequence[Mapping[str, object]] = (),
        proposal_repair_findings: Sequence[Mapping[str, object]] = (),
    ) -> AssignmentCompilationV2:
        """Compile one Goal-based Assignment through tiered validation.

        Missing semantic coverage is recoverable and therefore retains the
        safe compiled mission.  Structural/action findings are hard blocks and
        always clear it before this method returns.
        """

        agent_request = self.build_agent_request_v2(
            fleet_request,
            fleet_plan,
            assignment,
            local_plan_version=local_plan_version,
            target_catalog=target_catalog,
        )
        capability = self._v2_capability(fleet_request, assignment.uav_id)
        planner_request = self.build_planner_request_v2(
            agent_request,
            world_context,
            capability,
            trusted_target_id=trusted_target_id,
            semantic_repair_findings=semantic_repair_findings,
            proposal_repair_findings=proposal_repair_findings,
        )
        planner = self._planner_for(assignment.uav_id)
        output = planner.plan(planner_request)
        if not isinstance(output, SkillPlanDraftV3):
            raise AssignmentCompilerError(
                "V2 local planner must return SkillPlanDraftV3"
            )

        coverage = self._goal_checker.check(
            agent_request.goals,
            output,
            mission_id=agent_request.agent_mission_id,
            assignment_id=agent_request.assignment_id,
            uav_id=agent_request.uav_id,
            timestamp=timestamp,
            proposal_id=proposal_id,
            trusted_target_locked=trusted_target_id is not None,
            valid_landing_zone=bool(
                planner_request.world_context.landing_zones
            ),
            home_name=capability.home_name,
        )
        source = getattr(planner, "source", None)
        if source not in {"dynamic_scripted", "dynamic_llm"}:
            raise AssignmentCompilerError(
                "Spatial V3 validation requires local planner source "
                "dynamic_scripted or dynamic_llm"
            )
        validator = self._validator or PlanValidator()
        compiled, raw_report = validator.validate_and_compile_with_report(
            output,
            planner_request.world_context,
            source=source,
            mission_id=planner_request.mission_id,
            uav_id=planner_request.uav_id,
            plan_version=planner_request.plan_version,
            assignment_id=agent_request.assignment_id,
            proposal_id=proposal_id,
            timestamp=timestamp,
            goals=agent_request.goals,
            trusted_target_id=trusted_target_id,
            spatial_resolver=spatial_resolver,
            allow_trusted_safety_completion=(
                planner_request.allow_trusted_safety_completion
            ),
        )
        if not isinstance(raw_report, ValidationReport):
            raise AssignmentCompilerError(
                "PlanValidator returned an invalid validation report"
            )
        report = raw_report
        target_error = self._trusted_target_output_error(output, agent_request)
        if target_error is not None:
            report = self._with_compiler_hard_block(
                report,
                code=ValidationCode.UNKNOWN_ENTITY,
                message=target_error,
                timestamp=timestamp,
                proposal_id=proposal_id,
            )
            compiled = None
        if report.hard_blocked:
            compiled = None
        return AssignmentCompilationV2(
            agent_request=agent_request,
            planner_request=planner_request,
            planner_output=output,
            compiled_mission=compiled,
            goal_coverage=coverage,
            validation_report=report,
        )

    def compile_assignments(
        self,
        fleet_request: FleetMissionRequest,
        fleet_plan: FleetMissionPlan,
        world_contexts: Mapping[str, PlannerWorldContext],
        *,
        local_plan_versions: Mapping[str, int] | None = None,
    ) -> tuple[AssignmentCompilation, ...]:
        if not isinstance(world_contexts, Mapping):
            raise TypeError("world_contexts must be a mapping")
        versions = {} if local_plan_versions is None else dict(local_plan_versions)
        results: list[AssignmentCompilation] = []
        for assignment in fleet_plan.assignments:
            try:
                context = world_contexts[assignment.uav_id]
            except KeyError:
                raise AssignmentCompilerError(
                    f"missing world context for {assignment.uav_id}"
                ) from None
            results.append(
                self.compile_assignment(
                    fleet_request,
                    fleet_plan,
                    assignment,
                    context,
                    local_plan_version=versions.get(assignment.uav_id, 1),
                )
            )
        return tuple(results)

    def compile_assignments_v2(
        self,
        fleet_request: FleetMissionRequestV2,
        fleet_plan: FleetMissionPlanV2,
        world_contexts: Mapping[str, PlannerWorldContext],
        *,
        local_plan_versions: Mapping[str, int] | None = None,
        target_catalog: Mapping[str, TargetSpec] | None = None,
        trusted_target_ids: Mapping[str, str] | None = None,
        timestamp: float = 0.0,
    ) -> tuple[AssignmentCompilationV2, ...]:
        if not isinstance(world_contexts, Mapping):
            raise TypeError("world_contexts must be a mapping")
        versions = {} if local_plan_versions is None else dict(local_plan_versions)
        locked = {} if trusted_target_ids is None else dict(trusted_target_ids)
        results: list[AssignmentCompilationV2] = []
        for assignment in fleet_plan.assignments:
            try:
                context = world_contexts[assignment.uav_id]
            except KeyError:
                raise AssignmentCompilerError(
                    f"missing world context for {assignment.uav_id}"
                ) from None
            results.append(
                self.compile_assignment_v2(
                    fleet_request,
                    fleet_plan,
                    assignment,
                    context,
                    local_plan_version=versions.get(assignment.uav_id, 1),
                    target_catalog=target_catalog,
                    trusted_target_id=locked.get(assignment.assignment_id),
                    timestamp=timestamp,
                )
            )
        return tuple(results)

    def compile(
        self,
        fleet_request: FleetMissionRequest,
        fleet_plan: FleetMissionPlan,
        world_contexts: Mapping[str, PlannerWorldContext],
        *,
        local_plan_versions: Mapping[str, int] | None = None,
    ) -> dict[str, AssignmentCompilation]:
        results = self.compile_assignments(
            fleet_request,
            fleet_plan,
            world_contexts,
            local_plan_versions=local_plan_versions,
        )
        by_uav: dict[str, AssignmentCompilation] = {}
        for result in results:
            uav_id = result.agent_request.uav_id
            if uav_id in by_uav:
                raise AssignmentCompilerError(
                    "compile() cannot flatten sequential assignments by UAV; "
                    "use compile_assignments()"
                )
            by_uav[uav_id] = result
        return by_uav

    def compile_v2(
        self,
        fleet_request: FleetMissionRequestV2,
        fleet_plan: FleetMissionPlanV2,
        world_contexts: Mapping[str, PlannerWorldContext],
        *,
        local_plan_versions: Mapping[str, int] | None = None,
        target_catalog: Mapping[str, TargetSpec] | None = None,
        trusted_target_ids: Mapping[str, str] | None = None,
        timestamp: float = 0.0,
    ) -> dict[str, AssignmentCompilationV2]:
        results = self.compile_assignments_v2(
            fleet_request,
            fleet_plan,
            world_contexts,
            local_plan_versions=local_plan_versions,
            target_catalog=target_catalog,
            trusted_target_ids=trusted_target_ids,
            timestamp=timestamp,
        )
        by_uav: dict[str, AssignmentCompilationV2] = {}
        for result in results:
            uav_id = result.agent_request.uav_id
            if uav_id in by_uav:
                raise AssignmentCompilerError(
                    "compile_v2() cannot flatten sequential assignments by UAV"
                )
            by_uav[uav_id] = result
        return by_uav

    @staticmethod
    def _v2_capability(
        fleet_request: FleetMissionRequestV2,
        uav_id: str,
    ) -> FleetUavCapability:
        for capability in fleet_request.uav_inventory:
            if capability.uav_id == uav_id:
                return capability
        raise AssignmentCompilerError(f"unknown V2 UAV capability: {uav_id}")

    @staticmethod
    def _assignment_target_specs(
        fleet_request: FleetMissionRequestV2,
        assignment: FleetAssignmentV2,
        target_catalog: Mapping[str, TargetSpec] | None,
    ) -> dict[str, TargetSpec]:
        if target_catalog is None:
            return {}
        if not isinstance(target_catalog, Mapping) or any(
            not isinstance(alias, str) for alias in target_catalog
        ):
            raise TypeError("target_catalog must map target aliases to TargetSpec")
        if any(not isinstance(spec, TargetSpec) for spec in target_catalog.values()):
            raise TypeError("target_catalog must map target aliases to TargetSpec")
        assigned_aliases = {
            getattr(fleet_request.task_spec.goal(goal_id), "target_alias", None)
            for goal_id in assignment.goal_ids
        }
        return {
            alias: spec
            for alias, spec in target_catalog.items()
            if alias in assigned_aliases
        }

    @staticmethod
    def _trusted_target_output_error(
        output: SkillPlanDraftV3,
        request: AgentPlannerRequestV2,
    ) -> str | None:
        trusted = tuple(request.trusted_target_specs.values())
        if not trusted:
            if output.target_spec is not None:
                return "local Spatial V3 plan introduced an untrusted target_spec"
            return None
        if len(trusted) == 1 and output.target_spec != trusted[0]:
            return "local Spatial V3 plan changed the trusted target_spec"
        if output.target_spec is not None and output.target_spec not in trusted:
            return "local Spatial V3 plan introduced an untrusted target_spec"
        return None

    @staticmethod
    def _semantic_repair_findings(
        findings: Sequence[Mapping[str, object]],
    ) -> list[dict[str, str]]:
        if isinstance(findings, (str, bytes)) or not isinstance(
            findings, Sequence
        ):
            raise TypeError("semantic_repair_findings must be an array")
        if len(findings) > _MAX_SEMANTIC_REPAIR_FINDINGS:
            raise AssignmentCompilerError(
                "semantic_repair_findings must contain at most 32 items"
            )
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for index, finding in enumerate(findings):
            if not isinstance(finding, Mapping):
                raise TypeError(
                    f"semantic_repair_findings[{index}] must be an object"
                )
            goal_id = validate_routing_id(
                finding.get("goal_id"),
                f"semantic_repair_findings[{index}].goal_id",
            )
            raw_code = getattr(finding.get("code"), "value", finding.get("code"))
            try:
                code = ValidationCode(raw_code)
            except (TypeError, ValueError):
                raise AssignmentCompilerError(
                    f"semantic_repair_findings[{index}].code is invalid"
                ) from None
            if code not in _SEMANTIC_REPAIR_CODES:
                raise AssignmentCompilerError(
                    "semantic_repair_findings accepts only recoverable Goal codes"
                )
            raw_message = finding.get("message")
            if not isinstance(raw_message, str):
                raise TypeError(
                    f"semantic_repair_findings[{index}].message must be a string"
                )
            message = raw_message.strip()
            if not message or len(message) > _MAX_SEMANTIC_REPAIR_MESSAGE_CHARS:
                raise AssignmentCompilerError(
                    "semantic repair messages must contain 1..512 characters"
                )
            folded = message.casefold()
            if any(
                token in folded
                for token in ("oracle", "base64", "data:image", "hidden reasoning")
            ):
                raise AssignmentCompilerError(
                    "semantic repair messages contain forbidden private/media data"
                )
            key = (goal_id, code.value)
            if key in seen:
                raise AssignmentCompilerError(
                    "semantic_repair_findings must not repeat goal/code pairs"
                )
            seen.add(key)
            result.append(
                {"goal_id": goal_id, "code": code.value, "message": message}
            )
        return result

    @staticmethod
    def _proposal_repair_findings(
        findings: Sequence[Mapping[str, object]],
    ) -> list[dict[str, str]]:
        """Project structural retry feedback onto a small, private-safe shape.

        Callers may pass a richer trusted ValidationFinding mapping, but only
        ``code`` and ``message`` cross the model boundary.  Goal identifiers
        intentionally stay in ``semantic_repair_findings`` so structural
        failures cannot masquerade as Goal-coverage evidence.
        """

        if isinstance(findings, (str, bytes)) or not isinstance(
            findings, Sequence
        ):
            raise TypeError("proposal_repair_findings must be an array")
        if len(findings) > _MAX_PROPOSAL_REPAIR_FINDINGS:
            raise AssignmentCompilerError(
                "proposal_repair_findings must contain at most 32 items"
            )
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for index, finding in enumerate(findings):
            if not isinstance(finding, Mapping):
                raise TypeError(
                    f"proposal_repair_findings[{index}] must be an object"
                )
            raw_code = getattr(finding.get("code"), "value", finding.get("code"))
            if not isinstance(raw_code, str) or raw_code not in _PROPOSAL_REPAIR_CODES:
                raise AssignmentCompilerError(
                    f"proposal_repair_findings[{index}].code is invalid"
                )
            raw_message = finding.get("message")
            if not isinstance(raw_message, str):
                raise TypeError(
                    f"proposal_repair_findings[{index}].message must be a string"
                )
            message = " ".join(raw_message.split())
            if (
                not message
                or len(message) > _MAX_PROPOSAL_REPAIR_MESSAGE_CHARS
            ):
                raise AssignmentCompilerError(
                    "proposal repair messages must contain 1..512 characters"
                )
            folded = message.casefold()
            if any(
                token in folded
                for token in _PROPOSAL_REPAIR_FORBIDDEN_MESSAGE_TOKENS
            ):
                raise AssignmentCompilerError(
                    "proposal repair messages contain forbidden private/media data"
                )
            key = (raw_code, message)
            if key in seen:
                raise AssignmentCompilerError(
                    "proposal_repair_findings must not repeat code/message pairs"
                )
            seen.add(key)
            result.append({"code": raw_code, "message": message})
        return result

    @staticmethod
    def _with_compiler_hard_block(
        report: ValidationReport,
        *,
        code: ValidationCode,
        message: str,
        timestamp: float,
        proposal_id: str | None,
    ) -> ValidationReport:
        digest = sha256(
            (
                f"{report.mission_id}|{report.assignment_id}|"
                f"{proposal_id}|{code.value}|{message}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        finding = ValidationFinding(
            schema_version=1,
            finding_id=f"finding_{digest}",
            timestamp=timestamp,
            stage="LOCAL_COMPILATION",
            scope="ASSIGNMENT",
            severity=ValidationSeverity.HARD_ACTION_BLOCK,
            code=code,
            message=message,
            mission_id=report.mission_id,
            assignment_id=report.assignment_id,
            uav_id=report.uav_id,
            goal_id=None,
            step_id=None,
            proposal_id=proposal_id,
            evidence_refs=(),
            recommended_action=RecoveryRecommendation.REPAIR_LOCAL_PLAN,
        )
        report_digest = sha256(
            f"{report.report_id}|{finding.finding_id}".encode("utf-8")
        ).hexdigest()[:24]
        return ValidationReport(
            schema_version=1,
            report_id=f"report_{report_digest}",
            timestamp=timestamp,
            stage="LOCAL_COMPILATION",
            mission_id=report.mission_id,
            assignment_id=report.assignment_id,
            uav_id=report.uav_id,
            findings=report.findings + (finding,),
        )

    def _planner_for(self, uav_id: str) -> object:
        if self._planners is None:
            assert self._planner is not None
            return self._planner
        try:
            return self._planners[uav_id]
        except KeyError:
            raise AssignmentCompilerError(
                f"no local planner is configured for {uav_id}"
            ) from None

    @staticmethod
    def _validate_local_output(
        output: object,
        request: AgentPlannerRequest,
    ) -> None:
        if not isinstance(output, SkillPlanDraftV3):
            raise AssignmentCompilerError(
                "local planner must return SkillPlanDraftV3"
            )
        if (
            output.mission_id != request.agent_mission_id
            or output.uav_id != request.uav_id
            or output.plan_version != request.local_plan_version
        ):
            raise AssignmentCompilerError(
                "local Spatial V3 plan changed trusted routing/version"
            )
        if output.target_spec != request.target_spec:
            raise AssignmentCompilerError(
                "local Spatial V3 plan changed assignment target_spec"
            )
        searches = tuple(step for step in output.steps if step.skill == "SEARCH")
        tracks = tuple(step for step in output.steps if step.skill == "TRACK")
        if not searches or any(
            step.region != request.search_region for step in searches
        ):
            raise AssignmentCompilerError(
                "local Spatial V3 plan changed assignment RegionSpec"
            )
        if len(tracks) != 1 or tracks[0].args.get("duration_s") != request.track_duration_s:
            raise AssignmentCompilerError(
                "local Spatial V3 plan changed assignment track duration"
            )


AssignmentCompiler = FleetAssignmentCompiler


__all__ = [
    "AssignmentCompiler",
    "AssignmentCompilerError",
    "AssignmentCompilationV2",
    "FleetAssignmentCompiler",
]
