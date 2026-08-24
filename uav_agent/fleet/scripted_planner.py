"""Deterministic Fleet Planner for structured requests and baselines."""

from __future__ import annotations

from hashlib import sha256

from fleet.planner_base import FleetPlanner, FleetPlannerError
from fleet.schemas import validate_fleet_mission_plan
from fleet.types import (
    FleetAssignment,
    FleetMissionPlan,
    FleetMissionRequest,
    FleetStartPolicy,
)


class ScriptedFleetPlanner(FleetPlanner):
    """Map trusted structured target requests to deterministic assignments."""

    source = "fleet_scripted"

    def plan(self, request: FleetMissionRequest) -> FleetMissionPlan:
        if not isinstance(request, FleetMissionRequest):
            raise TypeError("request must be a FleetMissionRequest")
        sequential_targets = tuple(
            target.target_alias
            for target in request.target_requests
            if target.start_policy is FleetStartPolicy.SEQUENTIAL
        )
        if sequential_targets:
            raise FleetPlannerError(
                "ScriptedFleetPlanner v1 does not support SEQUENTIAL targets: "
                + ", ".join(sequential_targets)
            )
        available = {
            item.uav_id: item for item in request.uav_inventory if item.available
        }
        if not available:
            raise FleetPlannerError("no available UAV exists in trusted inventory")
        used: set[str] = set()
        assignments: list[FleetAssignment] = []
        for target in request.target_requests:
            if target.search_region is None or target.track_duration_s is None:
                raise FleetPlannerError(
                    "ScriptedFleetPlanner requires structured search_region and "
                    f"track_duration_s for {target.target_alias}"
                )
            start_policy = target.start_policy or FleetStartPolicy.PARALLEL
            priority = 100 if target.priority is None else target.priority
            if target.requested_uav_id is not None:
                uav_id = target.requested_uav_id
                if uav_id not in available:
                    raise FleetPlannerError(
                        f"requested UAV is unavailable: {uav_id}"
                    )
            else:
                unused = sorted(set(available) - used)
                if unused:
                    uav_id = unused[0]
                else:
                    raise FleetPlannerError(
                        "parallel target requirements exceed available UAV count"
                    )
            if uav_id in used:
                raise FleetPlannerError(
                    f"uav_id has more than one active assignment: {uav_id}"
                )
            assignment = FleetAssignment(
                assignment_id=_assignment_id(uav_id, target.target_alias),
                uav_id=uav_id,
                target_alias=target.target_alias,
                target_spec=target.target_spec,
                search_region=target.search_region,
                track_duration_s=target.track_duration_s,
                priority=priority,
                start_policy=start_policy,
            )
            assignments.append(assignment)
            used.add(uav_id)
        plan = FleetMissionPlan(
            fleet_mission_id=request.fleet_mission_id,
            fleet_plan_version=request.fleet_plan_version,
            assignments=tuple(assignments),
            coordination_policy=request.coordination_policy,
            assumptions=request.assumptions,
            unassigned_requirements=(),
        )
        return validate_fleet_mission_plan(plan, request)


def _assignment_id(uav_id: str, target_alias: str) -> str:
    candidate = f"assignment_{uav_id}_{target_alias}"
    if len(candidate) <= 64:
        return candidate
    digest = sha256(f"{uav_id}\0{target_alias}".encode("utf-8")).hexdigest()[:24]
    return f"assignment_{digest}"


__all__ = ["ScriptedFleetPlanner"]
