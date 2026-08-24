"""Fleet Planner V2 contracts based on semantic goal references."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id

from fleet.task_spec import (
    ConstraintStrength,
    FleetTaskSpecV1,
    MissionGoal,
    TerminationGoal,
)
from fleet.types import (
    FleetCoordinationPolicy,
    FleetMissionError,
    FleetStartPolicy,
    FleetUavCapability,
)
from target.types import TargetSpec

if TYPE_CHECKING:
    from fleet.schemas_v2 import FleetPlanSemanticIssue
    from planner.goal_checker import GoalCoverageReport
    from planner.schemas import CompiledMission, PlannerRequest
    from planner.schemas_v3 import SkillPlanDraftV3
    from runtime.validation_report import ValidationReport


MAX_TRUSTED_FLEET_EVIDENCE = 128
MAX_V2_ASSIGNMENTS = 64
MAX_ASSIGNMENT_GOALS = 32
MAX_ASSIGNMENT_DEVIATIONS = 32


class FleetStateEvidenceType(str, Enum):
    UAV_UNAVAILABLE = "UAV_UNAVAILABLE"
    INSUFFICIENT_ENERGY = "INSUFFICIENT_ENERGY"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    AIRSPACE_CONFLICT = "AIRSPACE_CONFLICT"
    TARGET_CLAIM_CONFLICT = "TARGET_CLAIM_CONFLICT"
    CURRENT_ASSIGNMENT_CONFLICT = "CURRENT_ASSIGNMENT_CONFLICT"
    OTHER_TRUSTED_STATE = "OTHER_TRUSTED_STATE"


class DeviationReasonCode(str, Enum):
    UAV_UNAVAILABLE = "UAV_UNAVAILABLE"
    INSUFFICIENT_ENERGY = "INSUFFICIENT_ENERGY"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    AIRSPACE_CONFLICT = "AIRSPACE_CONFLICT"
    TARGET_CLAIM_CONFLICT = "TARGET_CLAIM_CONFLICT"
    CURRENT_ASSIGNMENT_CONFLICT = "CURRENT_ASSIGNMENT_CONFLICT"
    BETTER_FLEET_UTILITY = "BETTER_FLEET_UTILITY"
    OTHER_TRUSTED_STATE = "OTHER_TRUSTED_STATE"


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    raise FleetMissionError(
        f"{name} must be one of: " + ", ".join(item.value for item in enum_type)
    )


def _text(value: object, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result or len(result) > maximum:
        raise FleetMissionError(
            f"{name} must contain between 1 and {maximum} characters"
        )
    if "oracle" in result.casefold() or "base64" in result.casefold():
        raise FleetMissionError(f"{name} must not contain privileged/image data")
    return result


def _id_tuple(
    value: object,
    name: str,
    *,
    maximum: int,
    minimum: int = 0,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array of routing IDs")
    if not minimum <= len(value) <= maximum:
        raise FleetMissionError(
            f"{name} must contain between {minimum} and {maximum} items"
        )
    result = tuple(
        validate_routing_id(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise FleetMissionError(f"{name} must not contain duplicates")
    return result


def _version(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FleetMissionError(f"{name} must be a positive integer")
    return value


def _priority(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("priority must be an integer")
    if not 0 <= value <= 1000:
        raise FleetMissionError("priority must be within [0, 1000]")
    return value


@dataclass(frozen=True, slots=True)
class TrustedFleetStateEvidence:
    evidence_id: str
    evidence_type: FleetStateEvidenceType
    summary: str
    uav_id: str | None = None
    goal_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            validate_routing_id(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(
            self,
            "evidence_type",
            _enum(self.evidence_type, FleetStateEvidenceType, "evidence_type"),
        )
        object.__setattr__(
            self, "summary", _text(self.summary, "summary", maximum=512)
        )
        if self.uav_id is not None:
            object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if self.goal_id is not None:
            object.__setattr__(
                self, "goal_id", validate_routing_id(self.goal_id, "goal_id")
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "summary": self.summary,
            "uav_id": self.uav_id,
            "goal_id": self.goal_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrustedFleetStateEvidence":
        from fleet.schemas_v2 import parse_trusted_fleet_state_evidence

        return parse_trusted_fleet_state_evidence(value)


@dataclass(frozen=True, slots=True)
class FleetMissionRequestV2:
    fleet_mission_id: str
    fleet_plan_version: int
    task_spec: FleetTaskSpecV1
    uav_inventory: tuple[FleetUavCapability, ...]
    trusted_fleet_state: tuple[TrustedFleetStateEvidence, ...]
    coordination_policy: FleetCoordinationPolicy
    schema_version: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            raise FleetMissionError("schema_version must equal integer 2")
        object.__setattr__(
            self, "fleet_mission_id", validate_mission_id(self.fleet_mission_id)
        )
        object.__setattr__(
            self,
            "fleet_plan_version",
            _version(self.fleet_plan_version, "fleet_plan_version"),
        )
        if not isinstance(self.task_spec, FleetTaskSpecV1):
            raise TypeError("task_spec must be a FleetTaskSpecV1")
        inventory = tuple(self.uav_inventory)
        if not inventory or len(inventory) > 64 or any(
            not isinstance(item, FleetUavCapability) for item in inventory
        ):
            raise FleetMissionError(
                "uav_inventory must contain 1..64 FleetUavCapability values"
            )
        if len({item.uav_id for item in inventory}) != len(inventory):
            raise FleetMissionError("uav_inventory contains duplicate uav_id")
        evidence = tuple(self.trusted_fleet_state)
        if len(evidence) > MAX_TRUSTED_FLEET_EVIDENCE or any(
            not isinstance(item, TrustedFleetStateEvidence) for item in evidence
        ):
            raise FleetMissionError(
                "trusted_fleet_state contains invalid or too many evidence values"
            )
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise FleetMissionError("trusted_fleet_state contains duplicate evidence_id")
        known_uavs = {item.uav_id for item in inventory}
        known_goals = set(self.task_spec.all_goal_ids)
        for item in evidence:
            if item.uav_id is not None and item.uav_id not in known_uavs:
                raise FleetMissionError(
                    f"trusted fleet evidence references unknown UAV: {item.uav_id}"
                )
            if item.goal_id is not None and item.goal_id not in known_goals:
                raise FleetMissionError(
                    f"trusted fleet evidence references unknown Goal: {item.goal_id}"
                )
        if not isinstance(self.coordination_policy, FleetCoordinationPolicy):
            raise TypeError("coordination_policy must be a FleetCoordinationPolicy")
        object.__setattr__(self, "uav_inventory", inventory)
        object.__setattr__(self, "trusted_fleet_state", evidence)

    @property
    def available_uav_ids(self) -> tuple[str, ...]:
        return tuple(item.uav_id for item in self.uav_inventory if item.available)

    @property
    def trusted_evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.trusted_fleet_state)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "fleet_mission_id": self.fleet_mission_id,
            "fleet_plan_version": self.fleet_plan_version,
            "task_spec": self.task_spec.to_dict(),
            "uav_inventory": [item.to_dict() for item in self.uav_inventory],
            "trusted_fleet_state": [
                item.to_dict() for item in self.trusted_fleet_state
            ],
            "coordination_policy": self.coordination_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FleetMissionRequestV2":
        from fleet.schemas_v2 import parse_fleet_mission_request_v2

        return parse_fleet_mission_request_v2(value)


@dataclass(frozen=True, slots=True)
class AssignmentDeviation:
    constraint_id: str
    reason_code: DeviationReasonCode
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constraint_id",
            validate_routing_id(self.constraint_id, "constraint_id"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(self.reason_code, DeviationReasonCode, "reason_code"),
        )
        evidence = _id_tuple(
            self.evidence_refs,
            "evidence_refs",
            maximum=16,
            minimum=1,
        )
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "reason_code": self.reason_code.value,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AssignmentDeviation":
        from fleet.schemas_v2 import parse_assignment_deviation

        return parse_assignment_deviation(value)


@dataclass(frozen=True, slots=True)
class FleetAssignmentV2:
    assignment_id: str
    uav_id: str
    goal_ids: tuple[str, ...]
    priority: int
    start_policy: FleetStartPolicy
    deviations: tuple[AssignmentDeviation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assignment_id",
            validate_routing_id(self.assignment_id, "assignment_id"),
        )
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "goal_ids",
            _id_tuple(
                self.goal_ids,
                "goal_ids",
                maximum=MAX_ASSIGNMENT_GOALS,
                minimum=1,
            ),
        )
        object.__setattr__(self, "priority", _priority(self.priority))
        object.__setattr__(
            self,
            "start_policy",
            _enum(self.start_policy, FleetStartPolicy, "start_policy"),
        )
        deviations = tuple(self.deviations)
        if len(deviations) > MAX_ASSIGNMENT_DEVIATIONS or any(
            not isinstance(item, AssignmentDeviation) for item in deviations
        ):
            raise FleetMissionError(
                "deviations must contain at most 32 AssignmentDeviation values"
            )
        if len({item.constraint_id for item in deviations}) != len(deviations):
            raise FleetMissionError(
                "deviations must not repeat a constraint_id within one assignment"
            )
        object.__setattr__(self, "deviations", deviations)

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "goal_ids": list(self.goal_ids),
            "priority": self.priority,
            "start_policy": self.start_policy.value,
            "deviations": [item.to_dict() for item in self.deviations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FleetAssignmentV2":
        from fleet.schemas_v2 import parse_fleet_assignment_v2

        return parse_fleet_assignment_v2(value)


@dataclass(frozen=True, slots=True)
class FleetMissionPlanV2:
    fleet_mission_id: str
    fleet_plan_version: int
    assignments: tuple[FleetAssignmentV2, ...]
    coordination_policy: FleetCoordinationPolicy
    assumptions: tuple[str, ...] = ()
    unassigned_goal_ids: tuple[str, ...] = ()
    schema_version: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            raise FleetMissionError("schema_version must equal integer 2")
        object.__setattr__(
            self, "fleet_mission_id", validate_mission_id(self.fleet_mission_id)
        )
        object.__setattr__(
            self,
            "fleet_plan_version",
            _version(self.fleet_plan_version, "fleet_plan_version"),
        )
        assignments = tuple(self.assignments)
        if len(assignments) > MAX_V2_ASSIGNMENTS or any(
            not isinstance(item, FleetAssignmentV2) for item in assignments
        ):
            raise FleetMissionError(
                "assignments must contain at most 64 FleetAssignmentV2 values"
            )
        if len({item.assignment_id for item in assignments}) != len(assignments):
            raise FleetMissionError("assignments contain duplicate assignment_id")
        if len({item.uav_id for item in assignments}) != len(assignments):
            raise FleetMissionError("one UAV cannot own multiple active V2 assignments")
        assigned_goals = [goal for item in assignments for goal in item.goal_ids]
        duplicates = sorted(
            goal for goal in set(assigned_goals) if assigned_goals.count(goal) > 1
        )
        if duplicates:
            raise FleetMissionError(
                "one Goal cannot be assigned to multiple active assignments: "
                + ", ".join(duplicates)
            )
        if not isinstance(self.coordination_policy, FleetCoordinationPolicy):
            raise TypeError("coordination_policy must be a FleetCoordinationPolicy")
        assumptions: list[str] = []
        if isinstance(self.assumptions, (str, bytes)) or not isinstance(
            self.assumptions, Sequence
        ):
            raise TypeError("assumptions must be an array of strings")
        if len(self.assumptions) > 32:
            raise FleetMissionError("assumptions must contain at most 32 items")
        for index, value in enumerate(self.assumptions):
            assumptions.append(_text(value, f"assumptions[{index}]", maximum=512))
        if len(assumptions) != len(set(assumptions)):
            raise FleetMissionError("assumptions must not contain duplicates")
        unassigned = _id_tuple(
            self.unassigned_goal_ids,
            "unassigned_goal_ids",
            maximum=MAX_ASSIGNMENT_GOALS,
        )
        overlap = sorted(set(assigned_goals) & set(unassigned))
        if overlap:
            raise FleetMissionError(
                "Goal cannot be both assigned and unassigned: " + ", ".join(overlap)
            )
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "assumptions", tuple(assumptions))
        object.__setattr__(self, "unassigned_goal_ids", unassigned)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "fleet_mission_id": self.fleet_mission_id,
            "fleet_plan_version": self.fleet_plan_version,
            "assignments": [item.to_dict() for item in self.assignments],
            "coordination_policy": self.coordination_policy.to_dict(),
            "assumptions": list(self.assumptions),
            "unassigned_goal_ids": list(self.unassigned_goal_ids),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        request: FleetMissionRequestV2 | None = None,
    ) -> "FleetMissionPlanV2":
        from fleet.schemas_v2 import parse_fleet_mission_plan_v2

        return parse_fleet_mission_plan_v2(value, request=request)

    def semantic_findings(
        self, request: FleetMissionRequestV2
    ) -> tuple["FleetPlanSemanticIssue", ...]:
        from fleet.schemas_v2 import fleet_plan_v2_semantic_findings

        return fleet_plan_v2_semantic_findings(self, request)


@dataclass(frozen=True, slots=True)
class FleetSafetySummaryEntry:
    """Minimal other-agent metadata safe for one local Planner prompt."""

    uav_id: str
    assignment_id: str
    status: str
    current_region: str | None = None
    altitude_layer: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "assignment_id",
            validate_routing_id(self.assignment_id, "assignment_id"),
        )
        object.__setattr__(self, "status", _text(self.status, "status", maximum=64))
        for name in ("current_region", "altitude_layer"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, _text(value, name, maximum=128)
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "uav_id": self.uav_id,
            "assignment_id": self.assignment_id,
            "status": self.status,
            "current_region": self.current_region,
            "altitude_layer": self.altitude_layer,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FleetSafetySummaryEntry":
        from fleet.schemas_v2 import parse_fleet_safety_summary_entry

        return parse_fleet_safety_summary_entry(value)


@dataclass(frozen=True, slots=True)
class AgentPlannerRequestV2:
    """Local-planner view containing only one Assignment's semantic Goals."""

    fleet_mission_id: str
    assignment_id: str
    uav_id: str
    goals: tuple[MissionGoal | TerminationGoal, ...]
    local_plan_version: int
    fleet_safety_summary: tuple[FleetSafetySummaryEntry, ...] = ()
    trusted_target_specs: Mapping[str, TargetSpec] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            raise FleetMissionError("schema_version must equal integer 2")
        object.__setattr__(
            self, "fleet_mission_id", validate_mission_id(self.fleet_mission_id)
        )
        object.__setattr__(
            self,
            "assignment_id",
            validate_routing_id(self.assignment_id, "assignment_id"),
        )
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        goals = tuple(self.goals)
        if not goals or len(goals) > MAX_ASSIGNMENT_GOALS or any(
            not isinstance(item, (MissionGoal, TerminationGoal)) for item in goals
        ):
            raise FleetMissionError(
                "goals must contain 1..32 MissionGoal/TerminationGoal values"
            )
        if len({item.goal_id for item in goals}) != len(goals):
            raise FleetMissionError("goals contains duplicate goal_id")
        object.__setattr__(
            self,
            "local_plan_version",
            _version(self.local_plan_version, "local_plan_version"),
        )
        safety = tuple(self.fleet_safety_summary)
        if len(safety) > 64 or any(
            not isinstance(item, FleetSafetySummaryEntry) for item in safety
        ):
            raise FleetMissionError(
                "fleet_safety_summary must contain at most 64 safe entries"
            )
        if len({item.uav_id for item in safety}) != len(safety):
            raise FleetMissionError("fleet_safety_summary contains duplicate uav_id")
        if any(item.uav_id == self.uav_id for item in safety):
            raise FleetMissionError(
                "fleet_safety_summary must contain only other UAVs"
            )
        if not isinstance(self.trusted_target_specs, Mapping):
            raise TypeError("trusted_target_specs must be a mapping")
        target_aliases = {
            item.target_alias
            for item in goals
            if isinstance(item, MissionGoal) and item.target_alias is not None
        }
        normalized_specs: dict[str, TargetSpec] = {}
        for raw_alias, target_spec in self.trusted_target_specs.items():
            alias = validate_routing_id(raw_alias, "target_alias")
            if alias not in target_aliases:
                raise FleetMissionError(
                    "trusted_target_specs contains a target outside this Assignment: "
                    + alias
                )
            if not isinstance(target_spec, TargetSpec):
                raise TypeError("trusted_target_specs values must be TargetSpec")
            if target_spec.mutable_appearance_notes:
                raise FleetMissionError(
                    "trusted target specs for planning must not contain mutable appearance"
                )
            normalized_specs[alias] = target_spec
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "fleet_safety_summary", safety)
        object.__setattr__(
            self,
            "trusted_target_specs",
            MappingProxyType(dict(sorted(normalized_specs.items()))),
        )

    @classmethod
    def for_assignment(
        cls,
        request: FleetMissionRequestV2,
        assignment: FleetAssignmentV2,
        *,
        local_plan_version: int,
        fleet_safety_summary: tuple[FleetSafetySummaryEntry, ...] = (),
        trusted_target_specs: Mapping[str, TargetSpec] | None = None,
    ) -> "AgentPlannerRequestV2":
        """Project exactly the Goals referenced by one validated Assignment."""

        from fleet.schemas_v2 import validate_fleet_mission_plan_v2

        # Reuse request-bound checks without requiring callers to construct a
        # complete fleet plan: a one-assignment plan is sufficient here.
        partial = FleetMissionPlanV2(
            fleet_mission_id=request.fleet_mission_id,
            fleet_plan_version=request.fleet_plan_version,
            assignments=(assignment,),
            coordination_policy=request.coordination_policy,
        )
        validate_fleet_mission_plan_v2(partial, request)
        return cls(
            fleet_mission_id=request.fleet_mission_id,
            assignment_id=assignment.assignment_id,
            uav_id=assignment.uav_id,
            goals=tuple(request.task_spec.goal(goal_id) for goal_id in assignment.goal_ids),
            local_plan_version=local_plan_version,
            fleet_safety_summary=fleet_safety_summary,
            trusted_target_specs=(
                {} if trusted_target_specs is None else trusted_target_specs
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "fleet_mission_id": self.fleet_mission_id,
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "goals": [item.to_dict() for item in self.goals],
            "local_plan_version": self.local_plan_version,
            "fleet_safety_summary": [
                item.to_dict() for item in self.fleet_safety_summary
            ],
            "trusted_target_specs": {
                alias: target.to_dict()
                for alias, target in self.trusted_target_specs.items()
            },
        }

    @property
    def agent_mission_id(self) -> str:
        """Return a stable local routing ID without extending the JSON schema."""

        digest = sha256(
            f"{self.fleet_mission_id}\0{self.assignment_id}".encode("utf-8")
        ).hexdigest()[:24]
        return validate_mission_id(f"mission_agent_{digest}")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AgentPlannerRequestV2":
        from fleet.schemas_v2 import parse_agent_planner_request_v2

        return parse_agent_planner_request_v2(value)


@dataclass(frozen=True, slots=True)
class AssignmentCompilationV2:
    """One local V2 Assignment proposal and its tiered admission result.

    Recoverable Goal-coverage findings retain ``compiled_mission``.  A hard
    action/safety finding must leave it ``None`` so a caller cannot dispatch a
    blocked plan accidentally.
    """

    agent_request: AgentPlannerRequestV2
    planner_request: "PlannerRequest"
    planner_output: "SkillPlanDraftV3"
    compiled_mission: "CompiledMission | None"
    goal_coverage: "GoalCoverageReport"
    validation_report: "ValidationReport"

    def __post_init__(self) -> None:
        from planner.goal_checker import GoalCoverageReport
        from planner.schemas import CompiledMission, PlannerRequest
        from planner.schemas_v3 import SkillPlanDraftV3
        from runtime.validation_report import ValidationReport

        if not isinstance(self.agent_request, AgentPlannerRequestV2):
            raise TypeError("agent_request must be an AgentPlannerRequestV2")
        if not isinstance(self.planner_request, PlannerRequest):
            raise TypeError("planner_request must be a PlannerRequest")
        if not isinstance(self.planner_output, SkillPlanDraftV3):
            raise TypeError("planner_output must be a SkillPlanDraftV3")
        if self.compiled_mission is not None and not isinstance(
            self.compiled_mission, CompiledMission
        ):
            raise TypeError("compiled_mission must be a CompiledMission or None")
        if not isinstance(self.goal_coverage, GoalCoverageReport):
            raise TypeError("goal_coverage must be a GoalCoverageReport")
        if not isinstance(self.validation_report, ValidationReport):
            raise TypeError("validation_report must be a ValidationReport")
        expected_route = (
            self.agent_request.agent_mission_id,
            self.agent_request.assignment_id,
            self.agent_request.uav_id,
        )
        coverage_route = (
            self.goal_coverage.mission_id,
            self.goal_coverage.assignment_id,
            self.goal_coverage.uav_id,
        )
        report_route = (
            self.validation_report.mission_id,
            self.validation_report.assignment_id,
            self.validation_report.uav_id,
        )
        if coverage_route != expected_route or report_route != expected_route:
            raise FleetMissionError(
                "V2 compilation reports changed trusted local routing"
            )
        if self.validation_report.hard_blocked:
            if self.compiled_mission is not None:
                raise FleetMissionError(
                    "hard-blocked V2 compilation cannot retain an executable plan"
                )
        elif self.compiled_mission is None:
            raise FleetMissionError(
                "executable V2 validation must retain its compiled mission"
            )

    @property
    def executable(self) -> bool:
        return self.compiled_mission is not None

    @property
    def semantically_valid(self) -> bool:
        return self.goal_coverage.complete and self.validation_report.semantically_valid

    @property
    def uncovered_goal_ids(self) -> tuple[str, ...]:
        return self.goal_coverage.uncovered_goal_ids


__all__ = [
    "AgentPlannerRequestV2",
    "AssignmentCompilationV2",
    "AssignmentDeviation",
    "DeviationReasonCode",
    "FleetAssignmentV2",
    "FleetMissionPlanV2",
    "FleetMissionRequestV2",
    "FleetSafetySummaryEntry",
    "FleetStateEvidenceType",
    "MAX_ASSIGNMENT_DEVIATIONS",
    "MAX_ASSIGNMENT_GOALS",
    "MAX_TRUSTED_FLEET_EVIDENCE",
    "MAX_V2_ASSIGNMENTS",
    "TrustedFleetStateEvidence",
]
