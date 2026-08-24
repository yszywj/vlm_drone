"""Compile independent Fleet assignments through existing Spatial V3 planners."""

from __future__ import annotations

from collections.abc import Mapping
import json

from planner.schemas import PlannerRequest, PlannerWorldContext
from planner.schemas_v3 import SkillPlanDraftV3
from runtime.plan_validator import PlanValidator

from fleet.planner_base import FleetPlannerError
from fleet.schemas import validate_fleet_mission_plan
from fleet.types import (
    AgentPlannerRequest,
    AssignmentCompilation,
    FleetAssignment,
    FleetMissionPlan,
    FleetMissionRequest,
    FleetUavCapability,
)
from fleet.world_belief import AgentFleetSummary


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
        self._validator = validator

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
                **kwargs,
            )
        return AssignmentCompilation(
            agent_request=agent_request,
            planner_request=planner_request,
            planner_output=output,
            compiled_mission=compiled,
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
    "FleetAssignmentCompiler",
]
