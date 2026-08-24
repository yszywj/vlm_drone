"""Exact parsers and request-bound validation for Fleet Planner V2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fleet.schemas import (
    parse_fleet_coordination_policy,
    parse_fleet_uav_capability,
)
from fleet.task_spec import ConstraintStrength, parse_fleet_task_spec
from fleet.task_spec import MissionGoal, TerminationGoal
from fleet.types import FleetMissionError
from fleet.types_v2 import (
    AssignmentDeviation,
    AgentPlannerRequestV2,
    FleetAssignmentV2,
    FleetMissionPlanV2,
    FleetMissionRequestV2,
    FleetSafetySummaryEntry,
    TrustedFleetStateEvidence,
)
from target.types import TargetSpec


def _exact(
    value: object,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    fields = frozenset(value)
    unknown = fields - required
    missing = required - fields
    if unknown:
        raise FleetMissionError(
            f"{name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise FleetMissionError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )
    return value


def _array(value: object, name: str, *, maximum: int) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    if len(value) > maximum:
        raise FleetMissionError(f"{name} must contain at most {maximum} items")
    return tuple(value)


def parse_trusted_fleet_state_evidence(
    value: object,
) -> TrustedFleetStateEvidence:
    data = _exact(
        value,
        name="TrustedFleetStateEvidence",
        required=frozenset(
            {"evidence_id", "evidence_type", "summary", "uav_id", "goal_id"}
        ),
    )
    return TrustedFleetStateEvidence(
        evidence_id=data["evidence_id"],  # type: ignore[arg-type]
        evidence_type=data["evidence_type"],  # type: ignore[arg-type]
        summary=data["summary"],  # type: ignore[arg-type]
        uav_id=data["uav_id"],  # type: ignore[arg-type]
        goal_id=data["goal_id"],  # type: ignore[arg-type]
    )


def parse_fleet_mission_request_v2(value: object) -> FleetMissionRequestV2:
    data = _exact(
        value,
        name="FleetMissionRequestV2",
        required=frozenset(
            {
                "schema_version",
                "fleet_mission_id",
                "fleet_plan_version",
                "task_spec",
                "uav_inventory",
                "trusted_fleet_state",
                "coordination_policy",
            }
        ),
    )
    raw_inventory = _array(data["uav_inventory"], "uav_inventory", maximum=64)
    inventory = tuple(parse_fleet_uav_capability(item) for item in raw_inventory)
    raw_task = data["task_spec"]
    if not isinstance(raw_task, Mapping):
        raise TypeError("task_spec must be an object")
    raw_evidence = _array(
        data["trusted_fleet_state"], "trusted_fleet_state", maximum=128
    )
    return FleetMissionRequestV2(
        schema_version=data["schema_version"],  # type: ignore[arg-type]
        fleet_mission_id=data["fleet_mission_id"],  # type: ignore[arg-type]
        fleet_plan_version=data["fleet_plan_version"],  # type: ignore[arg-type]
        task_spec=parse_fleet_task_spec(
            raw_task,
            trusted_uav_ids=tuple(item.uav_id for item in inventory),
        ),
        uav_inventory=inventory,
        trusted_fleet_state=tuple(
            parse_trusted_fleet_state_evidence(item) for item in raw_evidence
        ),
        coordination_policy=parse_fleet_coordination_policy(
            data["coordination_policy"]
        ),
    )


def parse_assignment_deviation(value: object) -> AssignmentDeviation:
    data = _exact(
        value,
        name="AssignmentDeviation",
        required=frozenset({"constraint_id", "reason_code", "evidence_refs"}),
    )
    return AssignmentDeviation(
        constraint_id=data["constraint_id"],  # type: ignore[arg-type]
        reason_code=data["reason_code"],  # type: ignore[arg-type]
        evidence_refs=_array(
            data["evidence_refs"], "evidence_refs", maximum=16
        ),  # type: ignore[arg-type]
    )


def parse_fleet_assignment_v2(value: object) -> FleetAssignmentV2:
    data = _exact(
        value,
        name="FleetAssignmentV2",
        required=frozenset(
            {"assignment_id", "uav_id", "goal_ids", "priority", "start_policy", "deviations"}
        ),
    )
    return FleetAssignmentV2(
        assignment_id=data["assignment_id"],  # type: ignore[arg-type]
        uav_id=data["uav_id"],  # type: ignore[arg-type]
        goal_ids=_array(data["goal_ids"], "goal_ids", maximum=32),  # type: ignore[arg-type]
        priority=data["priority"],  # type: ignore[arg-type]
        start_policy=data["start_policy"],  # type: ignore[arg-type]
        deviations=tuple(
            parse_assignment_deviation(item)
            for item in _array(data["deviations"], "deviations", maximum=32)
        ),
    )


def parse_fleet_mission_plan_v2(
    value: object,
    *,
    request: FleetMissionRequestV2 | None = None,
) -> FleetMissionPlanV2:
    data = _exact(
        value,
        name="FleetMissionPlanV2",
        required=frozenset(
            {
                "schema_version",
                "fleet_mission_id",
                "fleet_plan_version",
                "assignments",
                "coordination_policy",
                "assumptions",
                "unassigned_goal_ids",
            }
        ),
    )
    plan = FleetMissionPlanV2(
        schema_version=data["schema_version"],  # type: ignore[arg-type]
        fleet_mission_id=data["fleet_mission_id"],  # type: ignore[arg-type]
        fleet_plan_version=data["fleet_plan_version"],  # type: ignore[arg-type]
        assignments=tuple(
            parse_fleet_assignment_v2(item)
            for item in _array(data["assignments"], "assignments", maximum=64)
        ),
        coordination_policy=parse_fleet_coordination_policy(
            data["coordination_policy"]
        ),
        assumptions=_array(data["assumptions"], "assumptions", maximum=32),  # type: ignore[arg-type]
        unassigned_goal_ids=_array(
            data["unassigned_goal_ids"], "unassigned_goal_ids", maximum=32
        ),  # type: ignore[arg-type]
    )
    if request is not None:
        validate_fleet_mission_plan_v2(plan, request)
    return plan


def parse_fleet_safety_summary_entry(value: object) -> FleetSafetySummaryEntry:
    data = _exact(
        value,
        name="FleetSafetySummaryEntry",
        required=frozenset(
            {"uav_id", "assignment_id", "status", "current_region", "altitude_layer"}
        ),
    )
    return FleetSafetySummaryEntry(
        uav_id=data["uav_id"],  # type: ignore[arg-type]
        assignment_id=data["assignment_id"],  # type: ignore[arg-type]
        status=data["status"],  # type: ignore[arg-type]
        current_region=data["current_region"],  # type: ignore[arg-type]
        altitude_layer=data["altitude_layer"],  # type: ignore[arg-type]
    )


def parse_agent_planner_request_v2(value: object) -> AgentPlannerRequestV2:
    data = _exact(
        value,
        name="AgentPlannerRequestV2",
        required=frozenset(
            {
                "schema_version",
                "fleet_mission_id",
                "assignment_id",
                "uav_id",
                "goals",
                "local_plan_version",
                "fleet_safety_summary",
                "trusted_target_specs",
            }
        ),
    )
    raw_goals = _array(data["goals"], "goals", maximum=32)
    goals: list[MissionGoal | TerminationGoal] = []
    for item in raw_goals:
        if not isinstance(item, Mapping):
            raise TypeError("goals items must be objects")
        if "target_alias" in item:
            goals.append(MissionGoal.from_dict(item))
        elif "uav_id" in item:
            goals.append(TerminationGoal.from_dict(item))
        else:
            raise FleetMissionError("goal item is not a recognized TaskSpec Goal")
    raw_safety = _array(
        data["fleet_safety_summary"], "fleet_safety_summary", maximum=64
    )
    raw_targets = data["trusted_target_specs"]
    if not isinstance(raw_targets, Mapping):
        raise TypeError("trusted_target_specs must be an object")
    target_specs: dict[str, TargetSpec] = {}
    for alias, raw_target in raw_targets.items():
        if not isinstance(alias, str) or not isinstance(raw_target, Mapping):
            raise TypeError("trusted_target_specs must map strings to objects")
        target_specs[alias] = TargetSpec.from_dict(raw_target)
    return AgentPlannerRequestV2(
        schema_version=data["schema_version"],  # type: ignore[arg-type]
        fleet_mission_id=data["fleet_mission_id"],  # type: ignore[arg-type]
        assignment_id=data["assignment_id"],  # type: ignore[arg-type]
        uav_id=data["uav_id"],  # type: ignore[arg-type]
        goals=tuple(goals),
        local_plan_version=data["local_plan_version"],  # type: ignore[arg-type]
        fleet_safety_summary=tuple(
            parse_fleet_safety_summary_entry(item) for item in raw_safety
        ),
        trusted_target_specs=target_specs,
    )


def validate_fleet_mission_plan_v2(
    plan: FleetMissionPlanV2,
    request: FleetMissionRequestV2,
) -> FleetMissionPlanV2:
    """Block structural/entity/evidence errors, but keep semantics recoverable."""

    if not isinstance(plan, FleetMissionPlanV2):
        raise TypeError("plan must be a FleetMissionPlanV2")
    if not isinstance(request, FleetMissionRequestV2):
        raise TypeError("request must be a FleetMissionRequestV2")
    if (
        plan.fleet_mission_id != request.fleet_mission_id
        or plan.fleet_plan_version != request.fleet_plan_version
    ):
        raise FleetMissionError(
            "FleetMissionPlanV2 routing/version must exactly echo the request"
        )
    if plan.coordination_policy != request.coordination_policy:
        raise FleetMissionError(
            "FleetMissionPlanV2 cannot change trusted coordination policy"
        )
    available_uavs = set(request.available_uav_ids)
    known_goals = set(request.task_spec.all_goal_ids)
    known_constraints = {
        item.constraint_id for item in request.task_spec.assignment_constraints
    }
    trusted_evidence = set(request.trusted_evidence_ids)
    for assignment in plan.assignments:
        if assignment.uav_id not in available_uavs:
            raise FleetMissionError(
                f"assignment references unavailable or unknown UAV: {assignment.uav_id}"
            )
        unknown_goals = sorted(set(assignment.goal_ids) - known_goals)
        if unknown_goals:
            raise FleetMissionError(
                "assignment references unknown Goal(s): " + ", ".join(unknown_goals)
            )
        for deviation in assignment.deviations:
            if deviation.constraint_id not in known_constraints:
                raise FleetMissionError(
                    "deviation references unknown assignment constraint: "
                    + deviation.constraint_id
                )
            unknown_evidence = sorted(
                set(deviation.evidence_refs) - trusted_evidence
            )
            if unknown_evidence:
                raise FleetMissionError(
                    "deviation evidence must come from trusted Fleet state: "
                    + ", ".join(unknown_evidence)
                )
    unknown_unassigned = sorted(set(plan.unassigned_goal_ids) - known_goals)
    if unknown_unassigned:
        raise FleetMissionError(
            "unassigned_goal_ids contains unknown Goal(s): "
            + ", ".join(unknown_unassigned)
        )
    return plan


@dataclass(frozen=True, slots=True)
class FleetPlanSemanticIssue:
    """Recoverable semantic issue; never used as an action block by itself."""

    code: str
    message: str
    constraint_id: str | None = None
    goal_id: str | None = None
    assignment_id: str | None = None


def fleet_plan_v2_semantic_findings(
    plan: FleetMissionPlanV2,
    request: FleetMissionRequestV2,
) -> tuple[FleetPlanSemanticIssue, ...]:
    validate_fleet_mission_plan_v2(plan, request)
    findings: list[FleetPlanSemanticIssue] = []
    assignment_for_goal = {
        goal_id: assignment
        for assignment in plan.assignments
        for goal_id in assignment.goal_ids
    }
    accounted = set(assignment_for_goal) | set(plan.unassigned_goal_ids)
    for goal_id in request.task_spec.all_goal_ids:
        if goal_id not in accounted:
            findings.append(
                FleetPlanSemanticIssue(
                    code="UNACCOUNTED_GOAL",
                    message=f"Goal {goal_id} is neither assigned nor declared unassigned",
                    goal_id=goal_id,
                )
            )
    for constraint in request.task_spec.assignment_constraints:
        if constraint.strength is ConstraintStrength.OPEN:
            continue
        for goal_id in constraint.goal_ids:
            assignment = assignment_for_goal.get(goal_id)
            if assignment is None:
                findings.append(
                    FleetPlanSemanticIssue(
                        code="CONSTRAINED_GOAL_UNASSIGNED",
                        message=(
                            f"Goal {goal_id} has assignment constraint "
                            f"{constraint.constraint_id} but is unassigned"
                        ),
                        constraint_id=constraint.constraint_id,
                        goal_id=goal_id,
                    )
                )
                continue
            if assignment.uav_id == constraint.uav_id:
                continue
            deviation = next(
                (
                    item
                    for item in assignment.deviations
                    if item.constraint_id == constraint.constraint_id
                ),
                None,
            )
            findings.append(
                FleetPlanSemanticIssue(
                    code=(
                        "EXPLAINED_ASSIGNMENT_DEVIATION"
                        if deviation is not None
                        else "UNEXPLAINED_ASSIGNMENT_DEVIATION"
                    ),
                    message=(
                        f"Goal {goal_id} was assigned to {assignment.uav_id} instead "
                        f"of requested {constraint.uav_id}"
                    ),
                    constraint_id=constraint.constraint_id,
                    goal_id=goal_id,
                    assignment_id=assignment.assignment_id,
                )
            )
    return tuple(findings)


__all__ = [
    "parse_agent_planner_request_v2",
    "FleetPlanSemanticIssue",
    "fleet_plan_v2_semantic_findings",
    "parse_assignment_deviation",
    "parse_fleet_assignment_v2",
    "parse_fleet_mission_plan_v2",
    "parse_fleet_mission_request_v2",
    "parse_fleet_safety_summary_entry",
    "parse_trusted_fleet_state_evidence",
    "validate_fleet_mission_plan_v2",
]
