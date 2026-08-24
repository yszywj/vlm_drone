"""Immutable high-level contracts for fleet mission decomposition.

These values describe UAV/target assignment only.  They deliberately cannot
carry Skill steps, controller commands, camera images, PID values, or Oracle
target state; local Spatial V3 planning remains a separate compiler stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from planner.spatial import (
    CircleRegion,
    CorridorRegion,
    PolygonRegion,
    RectangleRegion,
    RegionSpec,
    RelationalRegion,
    SectorRegion,
)
from target.types import TargetSpec

if TYPE_CHECKING:
    from fleet.world_belief import AgentFleetSummary
    from planner.schemas import CompiledMission, PlannerOutput, PlannerRequest


_REGION_TYPES = (
    CircleRegion,
    RectangleRegion,
    SectorRegion,
    PolygonRegion,
    CorridorRegion,
    RelationalRegion,
)


class FleetMissionError(ValueError):
    """Raised when a fleet data contract is structurally invalid."""


class FleetStartPolicy(str, Enum):
    PARALLEL = "PARALLEL"
    SEQUENTIAL = "SEQUENTIAL"


class TargetClaimPolicy(str, Enum):
    EXCLUSIVE = "EXCLUSIVE"
    SHARED = "SHARED"


class RouteConflictPolicy(str, Enum):
    LOWER_PRIORITY_HOLDS = "LOWER_PRIORITY_HOLDS"
    REPORT_ONLY = "REPORT_ONLY"


class AssignmentFailurePolicy(str, Enum):
    REPORT_AND_REPLAN = "REPORT_AND_REPLAN"
    REPORT_ONLY = "REPORT_ONLY"
    CANCEL_FLEET = "CANCEL_FLEET"


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    raise FleetMissionError(
        f"{name} must be one of: "
        + ", ".join(item.value for item in enum_type)
    )


def _text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise FleetMissionError(
            f"{name} must contain between 1 and {maximum} characters"
        )
    return normalized


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise FleetMissionError(f"{name} must be a finite number")
    if positive and normalized <= 0.0:
        raise FleetMissionError(f"{name} must be greater than zero")
    return normalized


def _positive_version(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FleetMissionError(f"{name} must be a positive integer")
    return value


def _integer_range(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not low <= value <= high:
        raise FleetMissionError(f"{name} must be within [{low}, {high}]")
    return value


def _text_tuple(
    value: object,
    name: str,
    *,
    maximum_items: int = 32,
    maximum_chars: int = 128,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array of strings")
    if len(value) > maximum_items:
        raise FleetMissionError(
            f"{name} must contain at most {maximum_items} items"
        )
    result = tuple(
        _text(item, f"{name}[{index}]", maximum=maximum_chars)
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise FleetMissionError(f"{name} must not contain duplicates")
    return result


def _region(value: object, name: str, *, optional: bool = False) -> RegionSpec | None:
    if value is None and optional:
        return None
    if not isinstance(value, _REGION_TYPES):
        raise TypeError(f"{name} must be an existing RegionSpec")
    return value


def _target_spec(value: object, name: str) -> TargetSpec:
    if not isinstance(value, TargetSpec):
        raise TypeError(f"{name} must be a TargetSpec")
    if value.mutable_appearance_notes:
        raise FleetMissionError(
            f"{name}.mutable_appearance_notes must be empty for initial planning"
        )
    return value


@dataclass(frozen=True, slots=True)
class FleetUavCapability:
    uav_id: str
    display_name: str
    available: bool
    home_name: str
    max_speed_mps: float
    max_altitude_m: float
    camera_modalities: tuple[str, ...] = ("RGB",)
    payload_capabilities: tuple[str, ...] = ()
    remaining_energy_ratio: float = 1.0
    current_assignment_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self, "display_name", _text(self.display_name, "display_name", maximum=128)
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be bool")
        object.__setattr__(
            self, "home_name", validate_routing_id(self.home_name, "home_name")
        )
        object.__setattr__(
            self,
            "max_speed_mps",
            _finite(self.max_speed_mps, "max_speed_mps", positive=True),
        )
        object.__setattr__(
            self,
            "max_altitude_m",
            _finite(self.max_altitude_m, "max_altitude_m", positive=True),
        )
        modalities = _text_tuple(
            self.camera_modalities, "camera_modalities", maximum_items=16
        )
        if not modalities:
            raise FleetMissionError("camera_modalities must not be empty")
        object.__setattr__(self, "camera_modalities", modalities)
        object.__setattr__(
            self,
            "payload_capabilities",
            _text_tuple(
                self.payload_capabilities,
                "payload_capabilities",
                maximum_items=32,
            ),
        )
        energy = _finite(self.remaining_energy_ratio, "remaining_energy_ratio")
        if not 0.0 <= energy <= 1.0:
            raise FleetMissionError("remaining_energy_ratio must be within [0, 1]")
        object.__setattr__(self, "remaining_energy_ratio", energy)
        if self.current_assignment_id is not None:
            object.__setattr__(
                self,
                "current_assignment_id",
                validate_routing_id(
                    self.current_assignment_id, "current_assignment_id"
                ),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "uav_id": self.uav_id,
            "display_name": self.display_name,
            "available": self.available,
            "home_name": self.home_name,
            "max_speed_mps": self.max_speed_mps,
            "max_altitude_m": self.max_altitude_m,
            "camera_modalities": list(self.camera_modalities),
            "payload_capabilities": list(self.payload_capabilities),
            "remaining_energy_ratio": self.remaining_energy_ratio,
            "current_assignment_id": self.current_assignment_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FleetUavCapability:
        from fleet.schemas import parse_fleet_uav_capability

        return parse_fleet_uav_capability(value)


@dataclass(frozen=True, slots=True)
class FleetCoordinationPolicy:
    target_claim_policy: TargetClaimPolicy = TargetClaimPolicy.EXCLUSIVE
    minimum_uav_separation_m: float = 5.0
    route_conflict_policy: RouteConflictPolicy = (
        RouteConflictPolicy.LOWER_PRIORITY_HOLDS
    )
    assignment_failure_policy: AssignmentFailurePolicy = (
        AssignmentFailurePolicy.REPORT_AND_REPLAN
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_claim_policy",
            _enum(
                self.target_claim_policy,
                TargetClaimPolicy,
                "target_claim_policy",
            ),
        )
        separation = _finite(
            self.minimum_uav_separation_m,
            "minimum_uav_separation_m",
            positive=True,
        )
        if separation > 1000.0:
            raise FleetMissionError(
                "minimum_uav_separation_m must not exceed 1000"
            )
        object.__setattr__(self, "minimum_uav_separation_m", separation)
        object.__setattr__(
            self,
            "route_conflict_policy",
            _enum(
                self.route_conflict_policy,
                RouteConflictPolicy,
                "route_conflict_policy",
            ),
        )
        object.__setattr__(
            self,
            "assignment_failure_policy",
            _enum(
                self.assignment_failure_policy,
                AssignmentFailurePolicy,
                "assignment_failure_policy",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_claim_policy": self.target_claim_policy.value,
            "minimum_uav_separation_m": self.minimum_uav_separation_m,
            "route_conflict_policy": self.route_conflict_policy.value,
            "assignment_failure_policy": self.assignment_failure_policy.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FleetCoordinationPolicy:
        from fleet.schemas import parse_fleet_coordination_policy

        return parse_fleet_coordination_policy(value)


@dataclass(frozen=True, slots=True)
class FleetTargetRequest:
    target_alias: str
    target_spec: TargetSpec
    requested_uav_id: str | None = None
    search_region: RegionSpec | None = None
    track_duration_s: float | None = None
    # ``None`` leaves these decomposition choices to the Fleet Planner.  A
    # concrete value is a trusted relationship that model output must echo.
    priority: int | None = None
    start_policy: FleetStartPolicy | None = None
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_alias",
            validate_routing_id(self.target_alias, "target_alias"),
        )
        object.__setattr__(
            self, "target_spec", _target_spec(self.target_spec, "target_spec")
        )
        if self.requested_uav_id is not None:
            object.__setattr__(
                self,
                "requested_uav_id",
                validate_uav_id(self.requested_uav_id),
            )
        object.__setattr__(
            self,
            "search_region",
            _region(self.search_region, "search_region", optional=True),
        )
        if self.track_duration_s is not None:
            duration = _finite(
                self.track_duration_s, "track_duration_s", positive=True
            )
            if duration > 3600.0:
                raise FleetMissionError("track_duration_s must not exceed 3600")
            object.__setattr__(self, "track_duration_s", duration)
        if self.priority is not None:
            object.__setattr__(
                self,
                "priority",
                _integer_range(self.priority, "priority", 0, 1000),
            )
        if self.start_policy is not None:
            object.__setattr__(
                self,
                "start_policy",
                _enum(self.start_policy, FleetStartPolicy, "start_policy"),
            )
        if not isinstance(self.required, bool):
            raise TypeError("required must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_alias": self.target_alias,
            "target_spec": self.target_spec.to_dict(),
            "requested_uav_id": self.requested_uav_id,
            "search_region": (
                None if self.search_region is None else self.search_region.to_dict()
            ),
            "track_duration_s": self.track_duration_s,
            "priority": self.priority,
            "start_policy": (
                None if self.start_policy is None else self.start_policy.value
            ),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FleetTargetRequest:
        from fleet.schemas import parse_fleet_target_request

        return parse_fleet_target_request(value)


@dataclass(frozen=True, slots=True)
class FleetMissionRequest:
    fleet_mission_id: str
    fleet_plan_version: int
    original_instruction: str
    uav_inventory: tuple[FleetUavCapability, ...]
    target_requests: tuple[FleetTargetRequest, ...]
    coordination_policy: FleetCoordinationPolicy = FleetCoordinationPolicy()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fleet_mission_id", validate_mission_id(self.fleet_mission_id)
        )
        object.__setattr__(
            self,
            "fleet_plan_version",
            _positive_version(self.fleet_plan_version, "fleet_plan_version"),
        )
        object.__setattr__(
            self,
            "original_instruction",
            _text(
                self.original_instruction,
                "original_instruction",
                maximum=4096,
            ),
        )
        inventory = tuple(self.uav_inventory)
        if not inventory or len(inventory) > 64 or any(
            not isinstance(item, FleetUavCapability) for item in inventory
        ):
            raise FleetMissionError(
                "uav_inventory must contain 1..64 FleetUavCapability values"
            )
        if len({item.uav_id for item in inventory}) != len(inventory):
            raise FleetMissionError("uav_inventory contains duplicate uav_id")
        targets = tuple(self.target_requests)
        if not targets or len(targets) > 64 or any(
            not isinstance(item, FleetTargetRequest) for item in targets
        ):
            raise FleetMissionError(
                "target_requests must contain 1..64 FleetTargetRequest values"
            )
        if len({item.target_alias for item in targets}) != len(targets):
            raise FleetMissionError(
                "target_requests contains duplicate target_alias"
            )
        known_uavs = {item.uav_id for item in inventory}
        unknown_preferences = sorted(
            {
                item.requested_uav_id
                for item in targets
                if item.requested_uav_id is not None
                and item.requested_uav_id not in known_uavs
            }
        )
        if unknown_preferences:
            raise FleetMissionError(
                "target_requests reference unknown requested UAVs: "
                + ", ".join(unknown_preferences)
            )
        if not isinstance(self.coordination_policy, FleetCoordinationPolicy):
            raise TypeError(
                "coordination_policy must be a FleetCoordinationPolicy"
            )
        object.__setattr__(self, "uav_inventory", inventory)
        object.__setattr__(self, "target_requests", targets)
        object.__setattr__(
            self,
            "assumptions",
            _text_tuple(
                self.assumptions,
                "assumptions",
                maximum_items=32,
                maximum_chars=512,
            ),
        )

    @property
    def available_uav_ids(self) -> tuple[str, ...]:
        return tuple(item.uav_id for item in self.uav_inventory if item.available)

    @property
    def target_aliases(self) -> tuple[str, ...]:
        return tuple(item.target_alias for item in self.target_requests)

    def target_request(self, alias: str) -> FleetTargetRequest:
        normalized = validate_routing_id(alias, "target_alias")
        for item in self.target_requests:
            if item.target_alias == normalized:
                return item
        raise FleetMissionError(f"unknown target_alias: {normalized}")

    def uav(self, uav_id: str) -> FleetUavCapability:
        normalized = validate_uav_id(uav_id)
        for item in self.uav_inventory:
            if item.uav_id == normalized:
                return item
        raise FleetMissionError(f"unknown uav_id: {normalized}")

    def to_dict(self) -> dict[str, object]:
        return {
            "fleet_mission_id": self.fleet_mission_id,
            "fleet_plan_version": self.fleet_plan_version,
            "original_instruction": self.original_instruction,
            "uav_inventory": [item.to_dict() for item in self.uav_inventory],
            "target_requests": [item.to_dict() for item in self.target_requests],
            "coordination_policy": self.coordination_policy.to_dict(),
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FleetMissionRequest:
        from fleet.schemas import parse_fleet_mission_request

        return parse_fleet_mission_request(value)


@dataclass(frozen=True, slots=True)
class FleetAssignment:
    assignment_id: str
    uav_id: str
    target_alias: str
    target_spec: TargetSpec
    search_region: RegionSpec
    track_duration_s: float
    priority: int = 100
    start_policy: FleetStartPolicy = FleetStartPolicy.PARALLEL

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assignment_id",
            validate_routing_id(self.assignment_id, "assignment_id"),
        )
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "target_alias",
            validate_routing_id(self.target_alias, "target_alias"),
        )
        object.__setattr__(
            self, "target_spec", _target_spec(self.target_spec, "target_spec")
        )
        object.__setattr__(
            self, "search_region", _region(self.search_region, "search_region")
        )
        duration = _finite(
            self.track_duration_s, "track_duration_s", positive=True
        )
        if duration > 3600.0:
            raise FleetMissionError("track_duration_s must not exceed 3600")
        object.__setattr__(self, "track_duration_s", duration)
        object.__setattr__(
            self, "priority", _integer_range(self.priority, "priority", 0, 1000)
        )
        object.__setattr__(
            self,
            "start_policy",
            _enum(self.start_policy, FleetStartPolicy, "start_policy"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "target_alias": self.target_alias,
            "target_spec": self.target_spec.to_dict(),
            "search_region": self.search_region.to_dict(),
            "track_duration_s": self.track_duration_s,
            "priority": self.priority,
            "start_policy": self.start_policy.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FleetAssignment:
        from fleet.schemas import parse_fleet_assignment

        return parse_fleet_assignment(value)


def _validate_assignment_set(
    assignments: tuple[FleetAssignment, ...],
    policy: FleetCoordinationPolicy,
) -> None:
    if len({item.assignment_id for item in assignments}) != len(assignments):
        raise FleetMissionError("assignments contain duplicate assignment_id")
    by_uav: dict[str, list[FleetAssignment]] = {}
    for item in assignments:
        by_uav.setdefault(item.uav_id, []).append(item)
    invalid_uavs = sorted(
        uav_id
        for uav_id, items in by_uav.items()
        if len(items) > 1
        and any(item.start_policy is not FleetStartPolicy.SEQUENTIAL for item in items)
    )
    if invalid_uavs:
        raise FleetMissionError(
            "one UAV cannot own multiple active assignments: "
            + ", ".join(invalid_uavs)
        )
    if policy.target_claim_policy is TargetClaimPolicy.EXCLUSIVE:
        aliases = [item.target_alias for item in assignments]
        duplicates = sorted(
            alias for alias in set(aliases) if aliases.count(alias) > 1
        )
        if duplicates:
            raise FleetMissionError(
                "EXCLUSIVE target claims must be unique: "
                + ", ".join(duplicates)
            )


@dataclass(frozen=True, slots=True)
class FleetMissionPlan:
    fleet_mission_id: str
    fleet_plan_version: int
    assignments: tuple[FleetAssignment, ...]
    coordination_policy: FleetCoordinationPolicy
    assumptions: tuple[str, ...] = ()
    unassigned_requirements: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise FleetMissionError("schema_version must equal integer 1")
        object.__setattr__(
            self, "fleet_mission_id", validate_mission_id(self.fleet_mission_id)
        )
        object.__setattr__(
            self,
            "fleet_plan_version",
            _positive_version(self.fleet_plan_version, "fleet_plan_version"),
        )
        assignments = tuple(self.assignments)
        if len(assignments) > 64 or any(
            not isinstance(item, FleetAssignment) for item in assignments
        ):
            raise FleetMissionError(
                "assignments must contain at most 64 FleetAssignment values"
            )
        if not isinstance(self.coordination_policy, FleetCoordinationPolicy):
            raise TypeError(
                "coordination_policy must be a FleetCoordinationPolicy"
            )
        _validate_assignment_set(assignments, self.coordination_policy)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(
            self,
            "assumptions",
            _text_tuple(
                self.assumptions,
                "assumptions",
                maximum_items=32,
                maximum_chars=512,
            ),
        )
        object.__setattr__(
            self,
            "unassigned_requirements",
            _text_tuple(
                self.unassigned_requirements,
                "unassigned_requirements",
                maximum_items=64,
                maximum_chars=512,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "fleet_mission_id": self.fleet_mission_id,
            "fleet_plan_version": self.fleet_plan_version,
            "assignments": [item.to_dict() for item in self.assignments],
            "coordination_policy": self.coordination_policy.to_dict(),
            "assumptions": list(self.assumptions),
            "unassigned_requirements": list(self.unassigned_requirements),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        request: FleetMissionRequest | None = None,
    ) -> FleetMissionPlan:
        from fleet.schemas import parse_fleet_mission_plan

        return parse_fleet_mission_plan(value, request=request)


@dataclass(frozen=True, slots=True)
class FleetPlanPatch:
    fleet_mission_id: str
    base_fleet_plan_version: int
    new_fleet_plan_version: int
    replacement_assignments: tuple[FleetAssignment, ...]
    coordination_policy: FleetCoordinationPolicy
    reason_codes: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise FleetMissionError("schema_version must equal integer 1")
        object.__setattr__(
            self, "fleet_mission_id", validate_mission_id(self.fleet_mission_id)
        )
        base = _positive_version(
            self.base_fleet_plan_version, "base_fleet_plan_version"
        )
        new = _positive_version(
            self.new_fleet_plan_version, "new_fleet_plan_version"
        )
        if new != base + 1:
            raise FleetMissionError(
                "new_fleet_plan_version must equal base_fleet_plan_version + 1"
            )
        assignments = tuple(self.replacement_assignments)
        if not assignments or len(assignments) > 64 or any(
            not isinstance(item, FleetAssignment) for item in assignments
        ):
            raise FleetMissionError(
                "replacement_assignments must contain 1..64 assignments"
            )
        if not isinstance(self.coordination_policy, FleetCoordinationPolicy):
            raise TypeError(
                "coordination_policy must be a FleetCoordinationPolicy"
            )
        _validate_assignment_set(assignments, self.coordination_policy)
        reasons = _text_tuple(
            self.reason_codes,
            "reason_codes",
            maximum_items=16,
            maximum_chars=64,
        )
        if not reasons:
            raise FleetMissionError("reason_codes must not be empty")
        object.__setattr__(self, "replacement_assignments", assignments)
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "fleet_mission_id": self.fleet_mission_id,
            "base_fleet_plan_version": self.base_fleet_plan_version,
            "new_fleet_plan_version": self.new_fleet_plan_version,
            "replacement_assignments": [
                item.to_dict() for item in self.replacement_assignments
            ],
            "coordination_policy": self.coordination_policy.to_dict(),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class AgentPlannerRequest:
    fleet_mission_id: str
    assignment_id: str
    uav_id: str
    original_instruction: str
    assignment: FleetAssignment
    target_spec: TargetSpec
    search_region: RegionSpec
    track_duration_s: float
    local_plan_version: int = 1
    fleet_safety_summary: tuple["AgentFleetSummary", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fleet_mission_id", validate_mission_id(self.fleet_mission_id)
        )
        object.__setattr__(
            self,
            "assignment_id",
            validate_routing_id(self.assignment_id, "assignment_id"),
        )
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "original_instruction",
            _text(
                self.original_instruction,
                "original_instruction",
                maximum=4096,
            ),
        )
        if not isinstance(self.assignment, FleetAssignment):
            raise TypeError("assignment must be a FleetAssignment")
        if (
            self.assignment.assignment_id != self.assignment_id
            or self.assignment.uav_id != self.uav_id
        ):
            raise FleetMissionError(
                "AgentPlannerRequest routing must match its assignment"
            )
        target = _target_spec(self.target_spec, "target_spec")
        region = _region(self.search_region, "search_region")
        duration = _finite(
            self.track_duration_s, "track_duration_s", positive=True
        )
        if (
            target != self.assignment.target_spec
            or region != self.assignment.search_region
            or duration != self.assignment.track_duration_s
        ):
            raise FleetMissionError(
                "AgentPlannerRequest semantics must exactly match its assignment"
            )
        object.__setattr__(self, "target_spec", target)
        object.__setattr__(self, "search_region", region)
        object.__setattr__(self, "track_duration_s", duration)
        object.__setattr__(
            self,
            "local_plan_version",
            _positive_version(self.local_plan_version, "local_plan_version"),
        )
        from fleet.world_belief import AgentFleetSummary

        summaries = tuple(self.fleet_safety_summary)
        if any(not isinstance(item, AgentFleetSummary) for item in summaries):
            raise TypeError(
                "fleet_safety_summary must contain AgentFleetSummary values"
            )
        if any(item.uav_id == self.uav_id for item in summaries):
            raise FleetMissionError(
                "fleet_safety_summary must describe only other UAVs"
            )
        if len({item.uav_id for item in summaries}) != len(summaries):
            raise FleetMissionError(
                "fleet_safety_summary contains duplicate UAV IDs"
            )
        object.__setattr__(
            self,
            "fleet_safety_summary",
            tuple(sorted(summaries, key=lambda item: item.uav_id)),
        )

    @property
    def agent_mission_id(self) -> str:
        digest = sha256(
            f"{self.fleet_mission_id}\0{self.assignment_id}".encode("utf-8")
        ).hexdigest()[:24]
        return validate_mission_id(f"mission_agent_{digest}")

    def to_dict(self) -> dict[str, object]:
        return {
            "fleet_mission_id": self.fleet_mission_id,
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "original_instruction": self.original_instruction,
            "assignment": self.assignment.to_dict(),
            "target_spec": self.target_spec.to_dict(),
            "search_region": self.search_region.to_dict(),
            "track_duration_s": self.track_duration_s,
            "local_plan_version": self.local_plan_version,
            "fleet_safety_summary": [
                item.to_dict() for item in self.fleet_safety_summary
            ],
        }


@dataclass(frozen=True, slots=True)
class AssignmentCompilation:
    agent_request: AgentPlannerRequest
    planner_request: PlannerRequest
    planner_output: PlannerOutput
    compiled_mission: CompiledMission | None = None

    def __post_init__(self) -> None:
        from planner.schemas import CompiledMission, PlannerRequest

        if not isinstance(self.agent_request, AgentPlannerRequest):
            raise TypeError("agent_request must be an AgentPlannerRequest")
        if not isinstance(self.planner_request, PlannerRequest):
            raise TypeError("planner_request must be a PlannerRequest")
        if self.compiled_mission is not None and not isinstance(
            self.compiled_mission, CompiledMission
        ):
            raise TypeError("compiled_mission must be a CompiledMission or None")


__all__ = [
    "AgentPlannerRequest",
    "AssignmentCompilation",
    "AssignmentFailurePolicy",
    "FleetAssignment",
    "FleetCoordinationPolicy",
    "FleetMissionError",
    "FleetMissionPlan",
    "FleetMissionRequest",
    "FleetPlanPatch",
    "FleetStartPolicy",
    "FleetTargetRequest",
    "FleetUavCapability",
    "RouteConflictPolicy",
    "TargetClaimPolicy",
]
