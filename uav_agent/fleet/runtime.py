"""Fleet lifecycle orchestration over independent single-UAV MissionAgents.

Planning and strict assignment validation happen before the environment is
started.  Each simulation tick then consumes one synchronized fleet snapshot,
runs global airspace arbitration, and only afterwards ticks agents in sorted
UAV-ID order using observations captured at that same barrier.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import inspect
import json
from math import dist, isfinite
from types import MappingProxyType

from common.ids import generate_routing_id, validate_routing_id, validate_uav_id
from fleet.airspace_manager import (
    FleetAirspaceDecision,
    FleetAirspaceManager,
    FleetPoseSnapshot,
    FleetUavPose,
    coerce_fleet_pose_snapshot,
)
from fleet.model_request_broker import GlobalModelRequestBroker
from fleet.schemas import validate_fleet_mission_plan
from fleet.target_registry import (
    SharedTargetRegistry,
    TargetClaimError,
    TargetClaimState,
)
from fleet.types import (
    AssignmentFailurePolicy,
    FleetAssignment,
    FleetCoordinationPolicy,
    FleetMissionPlan,
    FleetMissionRequest,
    FleetStartPolicy,
    FleetTargetRequest,
    FleetUavCapability,
)
from fleet.world_belief import AgentFleetSummary, FleetWorldBelief


class FleetRuntimeError(RuntimeError):
    pass


class FleetStatus(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    FAILED_NO_EXECUTABLE_PLAN = "FAILED_NO_EXECUTABLE_PLAN"
    CANCELED = "CANCELED"


class AssignmentStatus(str, Enum):
    PENDING = "PENDING"
    INTERPRETING = "INTERPRETING"
    PLANNING = "PLANNING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    REPAIRING = "REPAIRING"
    DEGRADED_EXECUTABLE = "DEGRADED_EXECUTABLE"
    RUNNING = "RUNNING"
    HOLDING = "HOLDING"
    CANCELING = "CANCELING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    REASSIGNMENT_REQUIRED = "REASSIGNMENT_REQUIRED"
    WAITING_REASSIGNMENT = "WAITING_REASSIGNMENT"


_TERMINAL_ASSIGNMENT_STATES = frozenset(
    {
        AssignmentStatus.SUCCEEDED,
        AssignmentStatus.FAILED,
        AssignmentStatus.CANCELED,
        AssignmentStatus.REASSIGNMENT_REQUIRED,
    }
)

_STARTABLE_ASSIGNMENT_STATES = frozenset(
    {
        AssignmentStatus.PENDING,
        AssignmentStatus.READY,
        AssignmentStatus.DEGRADED_EXECUTABLE,
    }
)


@dataclass(frozen=True, slots=True)
class AssignmentRuntimeRecord:
    assignment: FleetAssignment
    required: bool = True
    status: AssignmentStatus = AssignmentStatus.PENDING
    local_plan_version: int = 1
    last_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, FleetAssignment):
            raise TypeError("assignment must be a FleetAssignment")
        if not isinstance(self.required, bool):
            raise TypeError("required must be bool")
        if not isinstance(self.status, AssignmentStatus):
            object.__setattr__(self, "status", AssignmentStatus(self.status))
        if (
            isinstance(self.local_plan_version, bool)
            or not isinstance(self.local_plan_version, int)
            or self.local_plan_version <= 0
        ):
            raise ValueError("local_plan_version must be a positive integer")
        if self.last_error is not None and not isinstance(self.last_error, str):
            raise TypeError("last_error must be a string or None")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.assignment.to_dict(),
            "required": self.required,
            "status": self.status.value,
            "local_plan_version": self.local_plan_version,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True)
class ReplannedAssignment:
    """Trusted handoff produced after Fleet planning and local compilation."""

    assignment_id: str
    agent: object
    start_input: tuple[str, object | None]
    perception: object | None = None
    planned_route: tuple[tuple[float, float, float], ...] = ()
    degraded: bool = False
    uncovered_goal_ids: tuple[str, ...] = ()
    schema_version: int = 1
    replacement_assignment: FleetAssignment | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("schema_version must equal integer 1")
        object.__setattr__(
            self,
            "assignment_id",
            validate_routing_id(self.assignment_id, "assignment_id"),
        )
        if not callable(getattr(self.agent, "snapshot", None)):
            raise TypeError("agent must provide snapshot()")
        if (
            not isinstance(self.start_input, tuple)
            or len(self.start_input) != 2
            or not isinstance(self.start_input[0], str)
            or not self.start_input[0].strip()
            or len(self.start_input[0]) > 4096
        ):
            raise TypeError(
                "start_input must be (instruction with 1..4096 characters, context)"
            )
        route = tuple(
            tuple(float(value) for value in point) for point in self.planned_route
        )
        if len(route) > 4096 or any(
            len(point) != 3 or any(not isfinite(value) for value in point)
            for point in route
        ):
            raise ValueError(
                "planned_route must contain at most 4096 finite 3D points"
            )
        object.__setattr__(self, "planned_route", route)
        if not isinstance(self.degraded, bool):
            raise TypeError("degraded must be bool")
        goals = tuple(
            validate_routing_id(value, "uncovered_goal_id")
            for value in self.uncovered_goal_ids
        )
        if len(goals) != len(set(goals)) or len(goals) > 64:
            raise ValueError("uncovered_goal_ids must be unique and bounded")
        object.__setattr__(self, "uncovered_goal_ids", goals)
        if self.replacement_assignment is not None:
            if not isinstance(self.replacement_assignment, FleetAssignment):
                raise TypeError(
                    "replacement_assignment must be a FleetAssignment or None"
                )


@dataclass(frozen=True, slots=True)
class FleetReplanPublication:
    """Atomic, trusted Fleet-plan publication returned by a replan handler.

    Each :class:`ReplannedAssignment` identifies the currently waiting
    assignment in ``assignment_id`` and carries its newly compiled
    ``replacement_assignment``.  The Runtime accepts the bundle only when the
    entire replacement set is valid; it never publishes a successful subset.
    """

    base_fleet_plan_version: int
    new_fleet_plan_version: int
    replacements: tuple[ReplannedAssignment, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("schema_version must equal integer 1")
        for name in ("base_fleet_plan_version", "new_fleet_plan_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.new_fleet_plan_version != self.base_fleet_plan_version + 1:
            raise ValueError(
                "new_fleet_plan_version must equal base_fleet_plan_version + 1"
            )
        replacements = tuple(self.replacements)
        if not replacements or len(replacements) > 64 or any(
            not isinstance(item, ReplannedAssignment) for item in replacements
        ):
            raise ValueError(
                "replacements must contain 1..64 ReplannedAssignment values"
            )
        source_ids = tuple(item.assignment_id for item in replacements)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("replacements contain duplicate source assignment_id")
        if any(item.replacement_assignment is None for item in replacements):
            raise ValueError(
                "Fleet publication replacements require replacement_assignment"
            )
        object.__setattr__(self, "replacements", replacements)


class AssignmentRegistry:
    """Mutable runtime state keyed independently by assignment and UAV."""

    def __init__(self) -> None:
        self._records: dict[str, AssignmentRuntimeRecord] = {}
        self._active_by_uav: dict[str, str] = {}

    def initialize(
        self,
        plan: FleetMissionPlan,
        request: FleetMissionRequest,
        *,
        assignment_requiredness: Mapping[str, bool] | None = None,
    ) -> None:
        if self._records:
            raise FleetRuntimeError("AssignmentRegistry is already initialized")
        required_by_alias = {
            item.target_alias: item.required for item in request.target_requests
        }
        required_by_assignment = (
            {} if assignment_requiredness is None else dict(assignment_requiredness)
        )
        if any(not isinstance(value, bool) for value in required_by_assignment.values()):
            raise TypeError("assignment_requiredness values must be bool")
        for assignment in plan.assignments:
            if assignment.assignment_id in self._records:
                raise FleetRuntimeError("duplicate assignment_id")
            if assignment.uav_id in self._active_by_uav:
                raise FleetRuntimeError(
                    "FleetMissionRuntime v1 supports one active assignment per UAV"
                )
            self._records[assignment.assignment_id] = AssignmentRuntimeRecord(
                assignment=assignment,
                required=required_by_assignment.get(
                    assignment.assignment_id,
                    required_by_alias.get(assignment.target_alias, True),
                ),
            )
            self._active_by_uav[assignment.uav_id] = assignment.assignment_id

    def for_uav(self, uav_id: str) -> AssignmentRuntimeRecord:
        uav_id = validate_uav_id(uav_id)
        try:
            return self._records[self._active_by_uav[uav_id]]
        except KeyError:
            raise FleetRuntimeError(f"no active assignment for {uav_id}") from None

    def by_id(self, assignment_id: str) -> AssignmentRuntimeRecord:
        assignment_id = validate_routing_id(assignment_id, "assignment_id")
        try:
            return self._records[assignment_id]
        except KeyError:
            raise FleetRuntimeError(f"unknown assignment_id: {assignment_id}") from None

    def update(
        self,
        assignment_id: str,
        status: AssignmentStatus | str,
        *,
        local_plan_version: int | None = None,
        last_error: str | None = None,
    ) -> AssignmentRuntimeRecord:
        record = self.by_id(assignment_id)
        status = AssignmentStatus(status)
        updated = replace(
            record,
            status=status,
            local_plan_version=(
                record.local_plan_version
                if local_plan_version is None
                else local_plan_version
            ),
            last_error=last_error,
        )
        self._records[assignment_id] = updated
        return updated

    def publish_replacements(
        self,
        replacements: Mapping[str, AssignmentRuntimeRecord],
    ) -> None:
        """Atomically replace source records after all validation is complete."""

        if not isinstance(replacements, Mapping) or not replacements:
            raise TypeError("replacements must be a non-empty mapping")
        staged_records = dict(self._records)
        staged_active = dict(self._active_by_uav)
        for raw_source_id, replacement in replacements.items():
            source_id = validate_routing_id(raw_source_id, "source_assignment_id")
            if source_id not in staged_records:
                raise FleetRuntimeError(
                    f"unknown source assignment_id: {source_id}"
                )
            if not isinstance(replacement, AssignmentRuntimeRecord):
                raise TypeError(
                    "replacement values must be AssignmentRuntimeRecord values"
                )
            source = staged_records.pop(source_id)
            if staged_active.get(source.assignment.uav_id) == source_id:
                staged_active.pop(source.assignment.uav_id)

        for replacement in replacements.values():
            new_id = replacement.assignment.assignment_id
            new_uav_id = replacement.assignment.uav_id
            if new_id in staged_records:
                raise FleetRuntimeError(
                    f"replacement assignment_id is already active: {new_id}"
                )
            if new_uav_id in staged_active:
                raise FleetRuntimeError(
                    f"replacement UAV is already routed: {new_uav_id}"
                )
            staged_records[new_id] = replacement
            staged_active[new_uav_id] = new_id

        # The only mutation point.  Validation failures above leave both
        # registry indexes untouched.
        self._records = staged_records
        self._active_by_uav = staged_active

    @property
    def records(self) -> tuple[AssignmentRuntimeRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {record.assignment.assignment_id: record.to_dict() for record in self.records}


@dataclass(frozen=True, slots=True)
class FleetRuntimeSnapshot:
    status: FleetStatus
    fleet_mission_id: str | None
    fleet_plan_version: int | None
    agent_plan_versions: Mapping[str, int | None]
    agent_statuses: Mapping[str, str]
    assignments: Mapping[str, Mapping[str, object]]
    last_airspace_decision: Mapping[str, object] | None
    events: tuple[Mapping[str, object], ...]
    last_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, FleetStatus):
            object.__setattr__(self, "status", FleetStatus(self.status))
        object.__setattr__(self, "agent_plan_versions", MappingProxyType(dict(self.agent_plan_versions)))
        object.__setattr__(self, "agent_statuses", MappingProxyType(dict(self.agent_statuses)))
        object.__setattr__(
            self,
            "assignments",
            MappingProxyType(
                {
                    key: MappingProxyType(deepcopy(dict(value)))
                    for key, value in self.assignments.items()
                }
            ),
        )
        if self.last_airspace_decision is not None:
            object.__setattr__(
                self,
                "last_airspace_decision",
                MappingProxyType(deepcopy(dict(self.last_airspace_decision))),
            )
        object.__setattr__(
            self,
            "events",
            tuple(MappingProxyType(deepcopy(dict(event))) for event in self.events),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "fleet_mission_id": self.fleet_mission_id,
            "fleet_plan_version": self.fleet_plan_version,
            "agent_plan_versions": dict(self.agent_plan_versions),
            "agent_statuses": dict(self.agent_statuses),
            "assignments": {
                key: deepcopy(dict(value)) for key, value in self.assignments.items()
            },
            "last_airspace_decision": (
                None
                if self.last_airspace_decision is None
                else deepcopy(dict(self.last_airspace_decision))
            ),
            "events": [deepcopy(dict(event)) for event in self.events],
            "last_error": self.last_error,
        }

    def to_summary_dict(self) -> dict[str, object]:
        """Return a bounded terminal summary without duplicating the event stream."""

        summary = self.to_dict()
        events = summary.pop("events")
        assert isinstance(events, list)
        summary["event_count"] = len(events)
        return summary


class FleetMissionRuntime:
    """Coordinate one MissionAgent per UAV without merging local plans."""

    def __init__(
        self,
        environment: object,
        fleet_planner: object,
        agents: Mapping[str, object],
        *,
        inventory: Sequence[FleetUavCapability] | None = None,
        target_requests: Sequence[FleetTargetRequest] | None = None,
        coordination_policy: FleetCoordinationPolicy | None = None,
        request_factory: Callable[[str], FleetMissionRequest] | None = None,
        assignment_compiler: object | None = None,
        world_contexts: Mapping[str, object] | None = None,
        precomputed_start_inputs: Mapping[
            str, tuple[str, object | None]
        ] | None = None,
        world_context_factory: Callable[[str, FleetAssignment], object] | None = None,
        perceptions: Mapping[str, object] | None = None,
        planned_routes: Mapping[str, Sequence[Sequence[float]]] | None = None,
        targets: SharedTargetRegistry | None = None,
        airspace: FleetAirspaceManager | None = None,
        model_broker: GlobalModelRequestBroker | None = None,
        logger: object | None = None,
        initial_assignment_states: Mapping[str, AssignmentStatus | str] | None = None,
        non_target_assignment_ids: Sequence[str] = (),
        assignment_requiredness: Mapping[str, bool] | None = None,
        replan_handler: Callable[
            [AssignmentRuntimeRecord, FleetWorldBelief | None],
            ReplannedAssignment | FleetReplanPublication | None,
        ]
        | None = None,
    ) -> None:
        if not callable(getattr(fleet_planner, "plan", None)):
            raise TypeError("fleet_planner must provide plan()")
        if not isinstance(agents, Mapping):
            raise TypeError("agents must be a mapping")
        normalized_agents: dict[str, object] = {}
        for raw_uav_id, agent in agents.items():
            uav_id = validate_uav_id(raw_uav_id)
            bound_id = getattr(agent, "uav_id", uav_id)
            if bound_id != uav_id:
                raise FleetRuntimeError("agent.uav_id does not match mapping key")
            if not callable(getattr(agent, "snapshot", None)):
                raise TypeError("each agent must provide snapshot()")
            normalized_agents[uav_id] = agent
        if request_factory is not None and not callable(request_factory):
            raise TypeError("request_factory must be callable")
        if assignment_compiler is not None and not all(
            callable(getattr(assignment_compiler, name, None))
            for name in ("build_agent_request", "build_planner_request")
        ):
            raise TypeError("assignment_compiler has an unsupported interface")
        if world_context_factory is not None and not callable(world_context_factory):
            raise TypeError("world_context_factory must be callable")
        self.environment = environment
        self.fleet_planner = fleet_planner
        self.agents = dict(sorted(normalized_agents.items()))
        self._inventory = None if inventory is None else tuple(inventory)
        self._target_requests = None if target_requests is None else tuple(target_requests)
        self._coordination_policy = coordination_policy or FleetCoordinationPolicy()
        self._request_factory = request_factory
        self._assignment_compiler = assignment_compiler
        self._world_contexts = {} if world_contexts is None else dict(world_contexts)
        if precomputed_start_inputs is None:
            self._precomputed_start_inputs: dict[
                str, tuple[str, object | None]
            ] = {}
        elif not isinstance(precomputed_start_inputs, Mapping):
            raise TypeError("precomputed_start_inputs must be a mapping")
        else:
            self._precomputed_start_inputs = {}
            for raw_uav_id, raw_value in precomputed_start_inputs.items():
                uav_id = validate_uav_id(raw_uav_id)
                if uav_id not in normalized_agents:
                    raise FleetRuntimeError(
                        "precomputed_start_inputs contains an unknown UAV"
                    )
                if (
                    not isinstance(raw_value, tuple)
                    or len(raw_value) != 2
                    or not isinstance(raw_value[0], str)
                    or not raw_value[0].strip()
                ):
                    raise TypeError(
                        "precomputed_start_inputs values must be "
                        "(non-empty instruction, context)"
                    )
                self._precomputed_start_inputs[uav_id] = raw_value
        self._world_context_factory = world_context_factory
        self._perceptions = {} if perceptions is None else dict(perceptions)
        if set(self._perceptions) - set(self.agents):
            raise FleetRuntimeError("perceptions contains an unknown UAV")
        self._planned_routes: dict[str, tuple[tuple[float, float, float], ...]] = {}
        self._route_progress: dict[str, int] = {}
        for raw_uav_id, raw_route in (planned_routes or {}).items():
            uav_id = validate_uav_id(raw_uav_id)
            if uav_id not in self.agents:
                raise FleetRuntimeError("planned_routes contains an unknown UAV")
            validated = FleetUavPose(
                uav_id=uav_id,
                position_xyz_m=(0.0, 0.0, 0.0),
                route_xyz_m=tuple(tuple(point) for point in raw_route),
            )
            self._planned_routes[uav_id] = validated.route_xyz_m
            self._route_progress[uav_id] = 0
        self.assignments = AssignmentRegistry()
        self.targets = targets or SharedTargetRegistry(
            self._coordination_policy.target_claim_policy.value
        )
        self.airspace = airspace or FleetAirspaceManager(
            self._coordination_policy.minimum_uav_separation_m
        )
        self.model_broker = model_broker or GlobalModelRequestBroker()
        self._logger = logger
        if isinstance(non_target_assignment_ids, (str, bytes)):
            raise TypeError("non_target_assignment_ids must be a sequence of IDs")
        raw_non_target_ids = tuple(non_target_assignment_ids)
        normalized_non_target_ids = frozenset(
            validate_routing_id(value, "non_target_assignment_id")
            for value in raw_non_target_ids
        )
        if len(normalized_non_target_ids) != len(raw_non_target_ids):
            raise FleetRuntimeError("non_target_assignment_ids contains duplicates")
        if assignment_requiredness is None:
            normalized_requiredness: dict[str, bool] = {}
        elif not isinstance(assignment_requiredness, Mapping):
            raise TypeError("assignment_requiredness must be a mapping")
        else:
            normalized_requiredness = {
                validate_routing_id(key, "assignment_id"): value
                for key, value in assignment_requiredness.items()
            }
            if any(
                not isinstance(value, bool)
                for value in normalized_requiredness.values()
            ):
                raise TypeError("assignment_requiredness values must be bool")
        self._non_target_assignment_ids = set(normalized_non_target_ids)
        self._assignment_requiredness = normalized_requiredness
        if replan_handler is not None and not callable(replan_handler):
            raise TypeError("replan_handler must be callable or None")
        self._replan_handler = replan_handler
        if initial_assignment_states is None:
            self._initial_assignment_states: dict[str, AssignmentStatus] = {}
        elif not isinstance(initial_assignment_states, Mapping):
            raise TypeError("initial_assignment_states must be a mapping")
        else:
            self._initial_assignment_states = {
                validate_routing_id(key, "assignment_id"): AssignmentStatus(value)
                for key, value in initial_assignment_states.items()
            }
        self._pending_ready_assignments: set[str] = set()
        self._pending_reassignments: set[str] = set()
        self._status = FleetStatus.IDLE
        self._request: FleetMissionRequest | None = None
        self._plan: FleetMissionPlan | None = None
        self._agent_start_inputs: dict[str, tuple[str, object | None]] = {}
        self._events: list[dict[str, object]] = []
        self._last_airspace_decision: FleetAirspaceDecision | None = None
        self._last_error: str | None = None
        self._world_belief: FleetWorldBelief | None = None
        self._closed = False
        self._cancel_requested = False
        self._cancel_airspace_override_logged: set[str] = set()
        self._local_failsafe_landings: set[str] = set()
        self._retired_failsafe_agents: dict[
            str, tuple[object, FleetAssignment, str | None]
        ] = {}

    @property
    def status(self) -> FleetStatus:
        return self._status

    @property
    def fleet_plan(self) -> FleetMissionPlan | None:
        return self._plan

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def world_belief(self) -> FleetWorldBelief | None:
        """Latest sanitized coordination snapshot, never evaluator/Camera data."""

        return self._world_belief

    def start(
        self,
        instruction: str,
        *,
        request: FleetMissionRequest | None = None,
    ) -> FleetMissionPlan:
        if self._closed:
            raise FleetRuntimeError("runtime is closed")
        if self._status is not FleetStatus.IDLE:
            raise FleetRuntimeError("start requires IDLE")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        self._status = FleetStatus.PLANNING
        self._last_error = None
        try:
            trusted_request = self._make_request(instruction.strip(), request)
            plan = self.fleet_planner.plan(trusted_request)
            if not isinstance(plan, FleetMissionPlan):
                raise FleetRuntimeError("fleet planner did not return FleetMissionPlan")
            plan = validate_fleet_mission_plan(plan, trusted_request)
            if not plan.assignments:
                raise FleetRuntimeError(
                    "Fleet plan has no executable assignments; required "
                    "unassigned targets cannot produce SUCCEEDED"
                )
            assignments_by_id = {
                assignment.assignment_id: assignment for assignment in plan.assignments
            }
            unknown_initial_states = sorted(
                set(self._initial_assignment_states) - set(assignments_by_id)
            )
            if unknown_initial_states:
                raise FleetRuntimeError(
                    "initial_assignment_states contains unknown assignments: "
                    + ", ".join(unknown_initial_states)
                )
            missing_assignments = tuple(
                assignment
                for assignment in plan.assignments
                if assignment.uav_id not in self.agents
            )
            undeclared_missing = sorted(
                assignment.uav_id
                for assignment in missing_assignments
                if assignment.assignment_id not in self._initial_assignment_states
            )
            if undeclared_missing:
                raise FleetRuntimeError(
                    "no MissionAgent exists for assigned UAVs without an explicit "
                    "planning state: " + ", ".join(undeclared_missing)
                )
            sequential = tuple(
                assignment.assignment_id
                for assignment in plan.assignments
                if assignment.start_policy is FleetStartPolicy.SEQUENTIAL
            )
            if sequential:
                raise FleetRuntimeError(
                    "FleetMissionRuntime v1 does not execute SEQUENTIAL "
                    "assignments; refusing to run them as PARALLEL: "
                    + ", ".join(sequential)
                )
            plan_assignment_ids = {
                assignment.assignment_id for assignment in plan.assignments
            }
            unknown_non_target = sorted(
                self._non_target_assignment_ids - plan_assignment_ids
            )
            unknown_requiredness = sorted(
                set(self._assignment_requiredness) - plan_assignment_ids
            )
            if unknown_non_target or unknown_requiredness:
                raise FleetRuntimeError(
                    "runtime envelope metadata references unknown assignments: "
                    + ", ".join(unknown_non_target + unknown_requiredness)
                )
            missing_requiredness = sorted(
                self._non_target_assignment_ids
                - set(self._assignment_requiredness)
            )
            if missing_requiredness:
                raise FleetRuntimeError(
                    "targetless assignments require explicit semantic requiredness: "
                    + ", ".join(missing_requiredness)
                )
            if (
                self.targets.claim_policy.value
                != plan.coordination_policy.target_claim_policy.value
            ):
                raise FleetRuntimeError(
                    "SharedTargetRegistry claim policy does not match the Fleet plan"
                )
            if plan.coordination_policy.route_conflict_policy.value != (
                "LOWER_PRIORITY_HOLDS"
            ):
                raise FleetRuntimeError(
                    "FleetMissionRuntime v1 only supports "
                    "LOWER_PRIORITY_HOLDS"
                )
            configured_separation = getattr(
                self.airspace,
                "minimum_separation_m",
                plan.coordination_policy.minimum_uav_separation_m,
            )
            if (
                not isinstance(configured_separation, (int, float))
                or isinstance(configured_separation, bool)
                or abs(
                    float(configured_separation)
                    - plan.coordination_policy.minimum_uav_separation_m
                )
                > 1e-9
            ):
                raise FleetRuntimeError(
                    "FleetAirspaceManager minimum separation does not match "
                    "the Fleet plan"
                )
            self.assignments.initialize(
                plan,
                trusted_request,
                assignment_requiredness=self._assignment_requiredness,
            )
            for assignment_id, state in self._initial_assignment_states.items():
                self.assignments.update(assignment_id, state)
            self._prepare_agent_start_inputs(trusted_request, plan)
            self._bind_target_claims(plan)
        except Exception as exc:
            self._status = FleetStatus.FAILED
            self._last_error = f"fleet planning failed: {type(exc).__name__}: {exc}"
            self._event("FLEET_PLANNING_FAILED", error=self._last_error)
            raise FleetRuntimeError(self._last_error) from exc

        self._request = trusted_request
        self._plan = plan
        executable_records = tuple(
            record
            for record in self.assignments.records
            if record.assignment.uav_id in self.agents
            and record.status in _STARTABLE_ASSIGNMENT_STATES
        )
        if not executable_records:
            for record in self.assignments.records:
                if record.status not in _TERMINAL_ASSIGNMENT_STATES:
                    self.assignments.update(
                        record.assignment.assignment_id,
                        AssignmentStatus.FAILED,
                        last_error="NO_EXECUTABLE_LOCAL_PLAN",
                    )
            self._status = FleetStatus.FAILED_NO_EXECUTABLE_PLAN
            self._last_error = "all assignments lack a safe executable local plan"
            self._event(
                "FLEET_NO_EXECUTABLE_PLAN",
                error=self._last_error,
            )
            self._refresh_world_belief()
            self._write_initial_logs()
            self._write_summary_log()
            return plan
        try:
            bind_environment = getattr(self.environment, "set_assignments", None)
            if callable(bind_environment):
                bind_environment(self._environment_assignments(plan))
            self._start_environment(plan)
        except Exception as exc:
            self._status = FleetStatus.FAILED
            self._last_error = (
                f"fleet environment start failed: {type(exc).__name__}: {exc}"
            )
            self._event("FLEET_ENVIRONMENT_START_FAILED", error=self._last_error)
            self._refresh_world_belief()
            raise FleetRuntimeError(self._last_error) from exc
        started = 0
        for uav_id in sorted({item.assignment.uav_id for item in executable_records}):
            record = self.assignments.for_uav(uav_id)
            if record.status not in _STARTABLE_ASSIGNMENT_STATES:
                continue
            try:
                self._start_agent(uav_id, record.assignment)
                snapshot = self._agent_snapshot(self.agents[uav_id])
                version = self._validated_agent_plan_version(
                    uav_id,
                    snapshot,
                    current_version=record.local_plan_version,
                    allow_initial_jump=True,
                )
                self.assignments.update(
                    record.assignment.assignment_id,
                    AssignmentStatus.RUNNING,
                    local_plan_version=version,
                )
            except Exception as exc:
                self._mark_local_failure(
                    record.assignment.assignment_id,
                    f"agent start failed: {type(exc).__name__}: {exc}",
                )
                continue
            started += 1
            self._event(
                "ASSIGNMENT_STARTED",
                uav_id=uav_id,
                assignment_id=record.assignment.assignment_id,
            )
        self._status = FleetStatus.RUNNING if started else FleetStatus.FAILED
        self._event("FLEET_STARTED", started_assignments=started)
        self._refresh_world_belief()
        self._write_initial_logs()
        return plan

    def tick(self) -> FleetRuntimeSnapshot:
        if self._closed:
            raise FleetRuntimeError("runtime is closed")
        if self._status is not FleetStatus.RUNNING:
            raise FleetRuntimeError(f"tick requires RUNNING, current={self._status.value}")
        self._start_pending_ready_assignments()
        barrier = self._advance_environment()
        observations_ready = barrier is not False
        pose_snapshot = self._fleet_pose_snapshot(barrier)
        held_by_airspace: set[str] = set()
        if pose_snapshot is not None:
            self._last_airspace_decision = self.airspace.evaluate(pose_snapshot)
            held_by_airspace.update(self._last_airspace_decision.hold_uav_ids)
            if not self._last_airspace_decision.clear:
                self._event(
                    "AIRSPACE_CONFLICT",
                    decision=self._last_airspace_decision.to_dict(),
                )
                self._log_airspace(self._last_airspace_decision)
            cancel_landing_overrides = self._cancel_landing_overrides(
                held_by_airspace
            )
            for uav_id in sorted(cancel_landing_overrides):
                if uav_id not in self._cancel_airspace_override_logged:
                    self._event(
                        "AIRSPACE_HOLD_OVERRIDDEN_FOR_FAILSAFE_LAND",
                        uav_id=uav_id,
                        reason=(
                            "fleet cancellation is already executing fail-safe "
                            "LAND; continuing to tick it prevents an indefinite "
                            "HOLD/LAND deadlock"
                        ),
                    )
                    self._cancel_airspace_override_logged.add(uav_id)
            held_by_airspace.difference_update(cancel_landing_overrides)
            for uav_id in sorted(held_by_airspace):
                self._hold_uav(uav_id, "AIRSPACE_CONFLICT")

        self._service_retired_failsafe_landings(
            barrier,
            observations_ready=observations_ready,
            held_uav_ids=held_by_airspace,
        )

        for uav_id in sorted(self.agents):
            try:
                record = self.assignments.for_uav(uav_id)
            except FleetRuntimeError:
                continue
            if record.status in _TERMINAL_ASSIGNMENT_STATES:
                continue
            if record.status is AssignmentStatus.WAITING_REASSIGNMENT:
                self._hold_uav(uav_id, "WAITING_FLEET_REPLAN")
                continue
            if uav_id in held_by_airspace:
                if not (
                    record.status is AssignmentStatus.HOLDING
                    and record.last_error not in {None, "AIRSPACE_CONFLICT"}
                ):
                    self.assignments.update(
                        record.assignment.assignment_id,
                        AssignmentStatus.HOLDING,
                        local_plan_version=record.local_plan_version,
                        last_error="AIRSPACE_CONFLICT",
                    )
                continue
            if not observations_ready:
                continue
            if record.status is AssignmentStatus.HOLDING:
                if record.last_error != "AIRSPACE_CONFLICT":
                    # Target-routing/claim holds are fail-closed and require a
                    # trusted replan or operator action.  They must not be
                    # mistaken for a transient airspace hold on the next tick.
                    continue
                self.assignments.update(
                    record.assignment.assignment_id,
                    AssignmentStatus.RUNNING,
                    local_plan_version=record.local_plan_version,
                    last_error=None,
                )
            try:
                observation = self._observation_for(uav_id, record.assignment, barrier)
                result = self.agents[uav_id].tick(observation)
                if (
                    record.assignment.assignment_id
                    in self._local_failsafe_landings
                ):
                    self._adopt_agent_snapshot(
                        uav_id, record.assignment, result
                    )
                    continue
                claim_accepted = self._synchronize_target_claim(
                    uav_id,
                    record.assignment,
                    result,
                )
                if claim_accepted:
                    self._adopt_agent_snapshot(uav_id, record.assignment, result)
            except Exception as exc:
                self._mark_local_failure(
                    record.assignment.assignment_id,
                    f"agent tick failed: {type(exc).__name__}: {exc}",
                )
                continue

        self._service_pending_reassignments()
        self._aggregate_status()
        self._refresh_world_belief()
        if self._status in {
            FleetStatus.SUCCEEDED,
            FleetStatus.PARTIAL_SUCCESS,
            FleetStatus.FAILED,
            FleetStatus.CANCELED,
        }:
            self._event(
                "FLEET_CANCELED"
                if self._status is FleetStatus.CANCELED
                else "FLEET_FINISHED",
                status=self._status.value,
            )
            self._write_summary_log()
        return self.snapshot()

    def register_ready_agent(
        self,
        assignment_id: str,
        agent: object,
        *,
        perception: object | None = None,
        start_input: tuple[str, object | None] | None = None,
        planned_route: Sequence[Sequence[float]] | None = None,
        degraded: bool = False,
        uncovered_goal_ids: Sequence[str] = (),
    ) -> AssignmentRuntimeRecord:
        """Queue a repaired local plan for start at the next Fleet barrier.

        The method changes no controller state.  It only validates routing and
        publishes READY; :meth:`tick` starts the agent before advancing the
        next synchronized environment barrier.
        """

        if self._closed:
            raise FleetRuntimeError("runtime is closed")
        if self._status is not FleetStatus.RUNNING:
            raise FleetRuntimeError("register_ready_agent requires RUNNING Fleet")
        record = self.assignments.by_id(assignment_id)
        if record.status not in {
            AssignmentStatus.INTERPRETING,
            AssignmentStatus.PLANNING,
            AssignmentStatus.VALIDATING,
            AssignmentStatus.REPAIRING,
            AssignmentStatus.WAITING_REASSIGNMENT,
            AssignmentStatus.REASSIGNMENT_REQUIRED,
        }:
            raise FleetRuntimeError(
                f"assignment {assignment_id} cannot accept a repaired plan from "
                f"state {record.status.value}"
            )
        uav_id = record.assignment.uav_id
        bound_id = getattr(agent, "uav_id", uav_id)
        if bound_id != uav_id or not callable(getattr(agent, "snapshot", None)):
            raise FleetRuntimeError("repaired MissionAgent routing is invalid")
        if start_input is not None:
            if (
                not isinstance(start_input, tuple)
                or len(start_input) != 2
                or not isinstance(start_input[0], str)
                or not start_input[0].strip()
            ):
                raise TypeError("start_input must be (non-empty instruction, context)")
            self._agent_start_inputs[uav_id] = start_input
        if uav_id not in self._agent_start_inputs:
            raise FleetRuntimeError("repaired assignment has no trusted start input")
        self.agents[uav_id] = agent
        if perception is not None:
            self._perceptions[uav_id] = perception
        if planned_route is not None:
            validated = FleetUavPose(
                uav_id=uav_id,
                position_xyz_m=(0.0, 0.0, 0.0),
                route_xyz_m=tuple(tuple(point) for point in planned_route),
            )
            self._planned_routes[uav_id] = validated.route_xyz_m
            self._route_progress[uav_id] = 0
        self._bind_assignment_claim(record.assignment)
        goals = tuple(
            validate_routing_id(value, "uncovered_goal_id")
            for value in uncovered_goal_ids
        )
        status = (
            AssignmentStatus.DEGRADED_EXECUTABLE
            if degraded
            else AssignmentStatus.READY
        )
        updated = self.assignments.update(
            assignment_id,
            status,
            local_plan_version=record.local_plan_version + 1,
            last_error=(
                None
                if not goals
                else "UNCOVERED_GOALS:" + ",".join(goals)
            ),
        )
        self._pending_ready_assignments.add(updated.assignment.assignment_id)
        self._event(
            "LOCAL_PLAN_READY",
            assignment_id=assignment_id,
            uav_id=uav_id,
            local_plan_version=updated.local_plan_version,
            degraded=degraded,
            uncovered_goal_ids=list(goals),
        )
        return updated

    def _start_pending_ready_assignments(self) -> None:
        for assignment_id in sorted(tuple(self._pending_ready_assignments)):
            record = self.assignments.by_id(assignment_id)
            if record.status not in {
                AssignmentStatus.READY,
                AssignmentStatus.DEGRADED_EXECUTABLE,
            }:
                self._pending_ready_assignments.discard(assignment_id)
                continue
            try:
                self._start_agent(record.assignment.uav_id, record.assignment)
                snapshot = self._agent_snapshot(
                    self.agents[record.assignment.uav_id]
                )
                version = self._validated_agent_plan_version(
                    record.assignment.uav_id,
                    snapshot,
                    current_version=record.local_plan_version,
                    allow_initial_jump=True,
                )
                self.assignments.update(
                    assignment_id,
                    AssignmentStatus.RUNNING,
                    local_plan_version=max(record.local_plan_version, version),
                    last_error=record.last_error,
                )
            except Exception as exc:
                self._mark_local_failure(
                    assignment_id,
                    f"repaired agent start failed: {type(exc).__name__}: {exc}",
                )
            else:
                self._event(
                    "REPAIRED_ASSIGNMENT_STARTED",
                    assignment_id=assignment_id,
                    uav_id=record.assignment.uav_id,
                )
            finally:
                self._pending_ready_assignments.discard(assignment_id)

    def _service_pending_reassignments(self) -> None:
        handler = self._replan_handler
        if handler is None:
            return
        for assignment_id in sorted(tuple(self._pending_reassignments)):
            record = self.assignments.by_id(assignment_id)
            if record.status is not AssignmentStatus.WAITING_REASSIGNMENT:
                self._pending_reassignments.discard(assignment_id)
                continue
            try:
                replacement = handler(record, self._world_belief)
                if replacement is None:
                    continue
                if isinstance(replacement, FleetReplanPublication):
                    published_ids = self._publish_fleet_replan(
                        assignment_id,
                        replacement,
                    )
                    try:
                        self._event(
                            "FLEET_REPLAN_ACCEPTED",
                            assignment_id=assignment_id,
                            replacement_assignment_ids=list(published_ids),
                            fleet_plan_version=replacement.new_fleet_plan_version,
                        )
                    except Exception:
                        pass
                    continue
                if not isinstance(replacement, ReplannedAssignment):
                    raise TypeError(
                        "replan_handler must return ReplannedAssignment, "
                        "FleetReplanPublication, or None"
                    )
                if replacement.replacement_assignment is not None:
                    raise FleetRuntimeError(
                        "a changed FleetAssignment requires FleetReplanPublication"
                    )
                if replacement.assignment_id != assignment_id:
                    raise FleetRuntimeError(
                        "replan_handler changed assignment routing identity"
                    )
                self.register_ready_agent(
                    assignment_id,
                    replacement.agent,
                    perception=replacement.perception,
                    start_input=replacement.start_input,
                    planned_route=(
                        None
                        if not replacement.planned_route
                        else replacement.planned_route
                    ),
                    degraded=replacement.degraded,
                    uncovered_goal_ids=replacement.uncovered_goal_ids,
                )
            except Exception as exc:
                error = f"Fleet replan failed: {type(exc).__name__}: {exc}"
                self._begin_local_failsafe_landing(
                    assignment_id,
                    error,
                )
                self._event(
                    "FLEET_REPLAN_FAILED",
                    assignment_id=assignment_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                self._event(
                    "FLEET_REPLAN_ACCEPTED",
                    assignment_id=assignment_id,
                )
            finally:
                try:
                    current = self.assignments.by_id(assignment_id)
                except FleetRuntimeError:
                    current = None
                if current is None or current.status is not (
                    AssignmentStatus.WAITING_REASSIGNMENT
                ):
                    self._pending_reassignments.discard(assignment_id)

    def _publish_fleet_replan(
        self,
        trigger_assignment_id: str,
        publication: FleetReplanPublication,
    ) -> tuple[str, ...]:
        """Validate and publish a trusted Fleet replacement bundle atomically."""

        plan = self._plan
        request = self._request
        if plan is None or request is None:
            raise FleetRuntimeError("Fleet replan requires an active trusted plan")
        if publication.base_fleet_plan_version != plan.fleet_plan_version:
            raise FleetRuntimeError("Fleet replan publication has a stale base version")
        if publication.new_fleet_plan_version != plan.fleet_plan_version + 1:
            raise FleetRuntimeError(
                "Fleet replan publication must increment fleet_plan_version by one"
            )

        trigger_assignment_id = validate_routing_id(
            trigger_assignment_id, "trigger_assignment_id"
        )
        replacements = publication.replacements
        source_ids = {item.assignment_id for item in replacements}
        if trigger_assignment_id not in source_ids:
            raise FleetRuntimeError(
                "Fleet replan publication does not replace its trigger assignment"
            )

        inventory = {item.uav_id: item for item in request.uav_inventory}
        current_records = {
            record.assignment.assignment_id: record
            for record in self.assignments.records
        }
        proposed_by_source: dict[str, FleetAssignment] = {}
        replacement_ids: set[str] = set()
        replacement_uavs: set[str] = set()
        replacement_aliases: set[str] = set()
        staged_records: dict[str, AssignmentRuntimeRecord] = {}
        claim_history = tuple(
            claim
            for target_record in self.targets.records
            for claim in target_record.claims
        )

        for item in replacements:
            source = current_records.get(item.assignment_id)
            if source is None:
                raise FleetRuntimeError(
                    f"Fleet replan references unknown source: {item.assignment_id}"
                )
            if source.status is not AssignmentStatus.WAITING_REASSIGNMENT:
                raise FleetRuntimeError(
                    f"Fleet replan source is not waiting: {item.assignment_id}"
                )
            assignment = item.replacement_assignment
            assert assignment is not None  # guaranteed by publication schema
            if assignment.assignment_id in replacement_ids:
                raise FleetRuntimeError(
                    "Fleet replan contains duplicate replacement assignment_id"
                )
            if assignment.uav_id in replacement_uavs:
                raise FleetRuntimeError(
                    "Fleet replan routes multiple replacements to one UAV"
                )
            if assignment.target_alias in replacement_aliases:
                raise FleetRuntimeError(
                    "Fleet replan contains duplicate replacement target_alias"
                )
            replacement_ids.add(assignment.assignment_id)
            replacement_uavs.add(assignment.uav_id)
            replacement_aliases.add(assignment.target_alias)

            # Replanning may transfer execution authority, but it may not
            # rewrite the unfinished target/region contract.
            if (
                assignment.target_alias != source.assignment.target_alias
                or assignment.target_spec != source.assignment.target_spec
                or assignment.search_region != source.assignment.search_region
                or assignment.track_duration_s
                != source.assignment.track_duration_s
                or assignment.priority != source.assignment.priority
                or assignment.start_policy is not source.assignment.start_policy
            ):
                raise FleetRuntimeError(
                    "Fleet replan replacement changed trusted goal semantics"
                )
            capability = inventory.get(assignment.uav_id)
            if capability is None or not capability.available:
                raise FleetRuntimeError(
                    f"Fleet replan routes to unavailable UAV: {assignment.uav_id}"
                )
            agent = item.agent
            if getattr(agent, "uav_id", assignment.uav_id) != assignment.uav_id:
                raise FleetRuntimeError("Fleet replan agent routing is invalid")
            if any(
                claim.assignment_id == assignment.assignment_id
                and (
                    claim.uav_id != assignment.uav_id
                    or claim.target_runtime_id != assignment.target_alias
                )
                for claim in claim_history
            ):
                raise FleetRuntimeError(
                    "Fleet replan replacement assignment_id has prior routing"
                )
            if any(
                claim.uav_id == assignment.uav_id
                and claim.assignment_id != assignment.assignment_id
                for claim in claim_history
            ):
                raise FleetRuntimeError(
                    "Fleet replan replacement UAV has prior assignment routing"
                )

            proposed_by_source[item.assignment_id] = assignment
            goals = tuple(item.uncovered_goal_ids)
            staged_records[item.assignment_id] = AssignmentRuntimeRecord(
                assignment=assignment,
                required=source.required,
                status=(
                    AssignmentStatus.DEGRADED_EXECUTABLE
                    if item.degraded
                    else AssignmentStatus.READY
                ),
                local_plan_version=source.local_plan_version + 1,
                last_error=(
                    None if not goals else "UNCOVERED_GOALS:" + ",".join(goals)
                ),
            )

        untouched = tuple(
            assignment
            for assignment in plan.assignments
            if assignment.assignment_id not in source_ids
        )
        untouched_ids = {item.assignment_id for item in untouched}
        untouched_uavs = {item.uav_id for item in untouched}
        if replacement_ids & untouched_ids:
            raise FleetRuntimeError(
                "Fleet replan replacement assignment_id conflicts with active plan"
            )
        if replacement_uavs & untouched_uavs:
            raise FleetRuntimeError(
                "Fleet replan replacement UAV is already routed by active plan"
            )
        if (
            plan.coordination_policy.target_claim_policy.value == "EXCLUSIVE"
            and replacement_aliases & {item.target_alias for item in untouched}
        ):
            raise FleetRuntimeError(
                "Fleet replan replacement conflicts with an active target claim"
            )

        ordered_replacements = tuple(
            proposed_by_source[source_id]
            for source_id in sorted(proposed_by_source)
        )
        new_plan = replace(
            plan,
            fleet_plan_version=publication.new_fleet_plan_version,
            assignments=untouched + ordered_replacements,
        )
        reassigned_by_alias = {
            assignment.target_alias: assignment.uav_id
            for assignment in ordered_replacements
        }
        new_request = replace(
            request,
            fleet_plan_version=publication.new_fleet_plan_version,
            target_requests=tuple(
                replace(
                    target,
                    requested_uav_id=reassigned_by_alias.get(
                        target.target_alias,
                        target.requested_uav_id,
                    ),
                )
                for target in request.target_requests
            ),
        )
        # This is the final validation gate and runs before any registry,
        # routing, claim, or Agent mutation.
        validate_fleet_mission_plan(new_plan, new_request)

        # Preserve targetless/required Goal semantics across an assignment-ID
        # and UAV reassignment.  The V1 compatibility alias is unchanged by
        # the semantic validation above, but still receives no target authority.
        staged_non_target_ids = set(self._non_target_assignment_ids)
        staged_requiredness = dict(self._assignment_requiredness)
        for item in replacements:
            source_required = staged_requiredness.pop(
                item.assignment_id,
                current_records[item.assignment_id].required,
            )
            replacement = proposed_by_source[item.assignment_id]
            staged_requiredness[replacement.assignment_id] = source_required
            if item.assignment_id in staged_non_target_ids:
                staged_non_target_ids.remove(item.assignment_id)
                staged_non_target_ids.add(replacement.assignment_id)

        # A cross-UAV reassignment transfers Goal authority, not permission to
        # leave the failed aircraft hovering forever.  Start its already
        # trusted MissionAgent cancel-and-land before removing old routing; if
        # LAND is asynchronous, retain and tick that retired Agent separately.
        for item in replacements:
            source = current_records[item.assignment_id]
            replacement_assignment = proposed_by_source[item.assignment_id]
            if replacement_assignment.uav_id == source.assignment.uav_id:
                continue
            source_uav_id = source.assignment.uav_id
            source_agent = self.agents.get(source_uav_id)
            if source_agent is None:
                raise FleetRuntimeError(
                    "failed source UAV has no MissionAgent for cancel-and-land"
                )
            try:
                result = source_agent.cancel()
                if result is None:
                    result = self._agent_snapshot(source_agent)
            except Exception as exc:
                raise FleetRuntimeError(
                    "failed source UAV could not enter cancel-and-land: "
                    + f"{type(exc).__name__}: {exc}"
                ) from exc
            source_status = _enum_text(
                _snapshot_value(result, "status", "RUNNING")
            )
            if source_status not in {"CANCELED", "FAILED", "SUCCEEDED"}:
                self._retired_failsafe_agents[source_uav_id] = (
                    source_agent,
                    source.assignment,
                    source.last_error,
                )

        set_environment_assignments = getattr(
            self.environment, "set_assignments", None
        )
        if callable(set_environment_assignments):
            set_environment_assignments(
                self._environment_assignments(
                    new_plan,
                    non_target_assignment_ids=staged_non_target_ids,
                )
            )

        self.assignments.publish_replacements(staged_records)
        self._non_target_assignment_ids = staged_non_target_ids
        self._assignment_requiredness = staged_requiredness
        self._plan = new_plan
        self._request = new_request
        for item in replacements:
            assignment = proposed_by_source[item.assignment_id]
            uav_id = assignment.uav_id
            self.agents[uav_id] = item.agent
            self._agent_start_inputs[uav_id] = item.start_input
            if item.perception is not None:
                self._perceptions[uav_id] = item.perception
            if item.planned_route:
                self._planned_routes[uav_id] = item.planned_route
                self._route_progress[uav_id] = 0
            self._bind_assignment_claim(assignment)
            self._pending_ready_assignments.add(assignment.assignment_id)
            self._pending_reassignments.discard(item.assignment_id)

        try:
            self._event(
                "FLEET_PLAN_VERSION_PUBLISHED",
                base_fleet_plan_version=publication.base_fleet_plan_version,
                fleet_plan_version=publication.new_fleet_plan_version,
                replaced_assignment_ids=sorted(source_ids),
                replacement_assignment_ids=sorted(replacement_ids),
            )
        except Exception:
            # Logging is not part of flight authority and cannot roll back an
            # already validated/published Fleet plan.
            pass
        write_plan = self._runtime_plan_log_method()
        if callable(write_plan):
            try:
                write_plan(new_plan)
            except Exception:
                pass
        return tuple(sorted(replacement_ids))

    def _cancel_landing_overrides(
        self,
        held_uav_ids: set[str],
    ) -> set[str]:
        """Let an already-requested fail-safe LAND advance through a HOLD.

        ``MissionAgent.cancel()`` transitions a non-terminal Agent into its
        trusted fail-safe LAND and reports RUNNING until that LAND completes.
        Skipping every subsequent Agent tick because the same UAV is held by
        airspace arbitration freezes LAND forever.  Cancellation is an explicit
        emergency action, so its bounded descent takes precedence while the
        conflict decision and one-shot override event remain fully auditable.
        """

        if not self._cancel_requested and not self._local_failsafe_landings:
            return set()
        overrides: set[str] = set()
        for uav_id in held_uav_ids:
            if uav_id in self._retired_failsafe_agents:
                overrides.add(uav_id)
                continue
            try:
                record = self.assignments.for_uav(uav_id)
            except FleetRuntimeError:
                continue
            if record.status is AssignmentStatus.CANCELING and (
                self._cancel_requested
                or record.assignment.assignment_id
                in self._local_failsafe_landings
            ):
                overrides.add(uav_id)
        return overrides

    def _service_retired_failsafe_landings(
        self,
        barrier: object | None,
        *,
        observations_ready: bool,
        held_uav_ids: set[str],
    ) -> None:
        """Advance cross-UAV source LANDs outside the new assignment registry."""

        if not observations_ready:
            return
        for uav_id in sorted(tuple(self._retired_failsafe_agents)):
            if uav_id in held_uav_ids:
                continue
            agent, assignment, error = self._retired_failsafe_agents[uav_id]
            try:
                observation = self._observation_for(uav_id, assignment, barrier)
                snapshot = agent.tick(observation)
            except Exception as exc:
                self._hold_uav(uav_id, "RETIRED_FAILSAFE_LAND_ERROR")
                self._event(
                    "RETIRED_FAILSAFE_LAND_TICK_FAILED",
                    assignment_id=assignment.assignment_id,
                    uav_id=uav_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            status = _enum_text(_snapshot_value(snapshot, "status", "RUNNING"))
            if status in {"CANCELED", "FAILED", "SUCCEEDED"}:
                self._retired_failsafe_agents.pop(uav_id, None)
                self._event(
                    "RETIRED_FAILSAFE_LAND_COMPLETED",
                    assignment_id=assignment.assignment_id,
                    uav_id=uav_id,
                    agent_terminal_status=status,
                    original_error=error,
                )

    def cancel(self) -> FleetRuntimeSnapshot:
        if self._status is not FleetStatus.RUNNING:
            raise FleetRuntimeError("cancel requires RUNNING")
        self._cancel_requested = True
        self._event("FLEET_CANCEL_REQUESTED")
        for uav_id, agent in self.agents.items():
            try:
                record = self.assignments.for_uav(uav_id)
            except FleetRuntimeError:
                continue
            if record.status in _TERMINAL_ASSIGNMENT_STATES:
                continue
            try:
                result = agent.cancel()
                if result is None:
                    result = self._agent_snapshot(agent)
            except Exception as exc:
                self.assignments.update(
                    record.assignment.assignment_id,
                    AssignmentStatus.FAILED,
                    last_error=f"cancel failed: {type(exc).__name__}: {exc}",
                )
                self._release_target_claim(record.assignment.assignment_id)
            else:
                status = _enum_text(_snapshot_value(result, "status", "RUNNING"))
                if status == "CANCELED":
                    assignment_status = AssignmentStatus.CANCELED
                elif status == "FAILED":
                    assignment_status = AssignmentStatus.FAILED
                elif status == "SUCCEEDED":
                    assignment_status = AssignmentStatus.SUCCEEDED
                else:
                    # MissionAgent keeps RUNNING while its fail-safe LAND is
                    # active.  Do not terminate the Fleet loop until LAND has
                    # produced the requested terminal outcome.
                    assignment_status = AssignmentStatus.CANCELING
                self.assignments.update(
                    record.assignment.assignment_id,
                    assignment_status,
                )
                if assignment_status in {
                    AssignmentStatus.CANCELED,
                    AssignmentStatus.FAILED,
                }:
                    self._release_target_claim(record.assignment.assignment_id)
        self._aggregate_status()
        self._refresh_world_belief()
        if self._status is FleetStatus.CANCELED:
            self._event("FLEET_CANCELED", status=self._status.value)
            self._write_summary_log()
        return self.snapshot()

    def close(self) -> None:
        if self._closed:
            return
        for agent in self.agents.values():
            close = getattr(agent, "close", None)
            if callable(close):
                close()
        close_environment = getattr(self.environment, "close", None)
        if callable(close_environment):
            close_environment()
        self._closed = True

    def snapshot(self) -> FleetRuntimeSnapshot:
        plan_versions: dict[str, int | None] = {}
        statuses: dict[str, str] = {}
        for uav_id, agent in self.agents.items():
            try:
                snapshot = self._agent_snapshot(agent)
            except Exception:
                plan_versions[uav_id] = None
                statuses[uav_id] = "UNKNOWN"
                continue
            plan_versions[uav_id] = _optional_int(_snapshot_value(snapshot, "plan_version"))
            statuses[uav_id] = _enum_text(_snapshot_value(snapshot, "status", "UNKNOWN"))
        return FleetRuntimeSnapshot(
            status=self._status,
            fleet_mission_id=(None if self._plan is None else self._plan.fleet_mission_id),
            fleet_plan_version=(None if self._plan is None else self._plan.fleet_plan_version),
            agent_plan_versions=plan_versions,
            agent_statuses=statuses,
            assignments={
                str(row["assignment_id"]): row
                for row in self._assignment_log_rows()
            },
            last_airspace_decision=(
                None
                if self._last_airspace_decision is None
                else self._last_airspace_decision.to_dict()
            ),
            events=tuple(self._events),
            last_error=self._last_error,
        )

    def _make_request(
        self,
        instruction: str,
        supplied: FleetMissionRequest | None,
    ) -> FleetMissionRequest:
        if supplied is not None:
            if not isinstance(supplied, FleetMissionRequest):
                raise TypeError("request must be a FleetMissionRequest")
            if supplied.original_instruction != instruction:
                raise FleetRuntimeError("request instruction does not match start instruction")
            return supplied
        if self._request_factory is not None:
            request = self._request_factory(instruction)
            if not isinstance(request, FleetMissionRequest):
                raise TypeError("request_factory must return FleetMissionRequest")
            if request.original_instruction != instruction:
                raise FleetRuntimeError("request_factory changed original instruction")
            return request
        environment_factory = getattr(self.environment, "build_fleet_mission_request", None)
        if callable(environment_factory):
            request = environment_factory(instruction)
            if not isinstance(request, FleetMissionRequest):
                raise TypeError("environment request builder returned invalid type")
            if request.original_instruction != instruction:
                raise FleetRuntimeError(
                    "environment request builder changed original instruction"
                )
            return request
        if self._inventory is None or self._target_requests is None:
            raise FleetRuntimeError(
                "start requires request=, request_factory, or configured inventory/target_requests"
            )
        return FleetMissionRequest(
            fleet_mission_id=generate_routing_id("fleet_mission"),
            fleet_plan_version=1,
            original_instruction=instruction,
            uav_inventory=self._inventory,
            target_requests=self._target_requests,
            coordination_policy=self._coordination_policy,
        )

    def _prepare_agent_start_inputs(
        self,
        request: FleetMissionRequest,
        plan: FleetMissionPlan,
    ) -> None:
        for assignment in plan.assignments:
            precomputed = self._precomputed_start_inputs.get(assignment.uav_id)
            if precomputed is not None:
                self._agent_start_inputs[assignment.uav_id] = precomputed
                continue
            context = self._context_for(assignment.uav_id, assignment)
            focused = _focused_assignment_instruction(request, assignment)
            if self._assignment_compiler is not None:
                if context is None:
                    raise FleetRuntimeError(
                        f"assignment compiler requires world context for {assignment.uav_id}"
                    )
                agent_request = self._assignment_compiler.build_agent_request(
                    request,
                    plan,
                    assignment,
                    local_plan_version=1,
                )
                planner_request = self._assignment_compiler.build_planner_request(
                    agent_request,
                    context,
                    request.uav(assignment.uav_id),
                )
                focused = planner_request.instruction
                context = planner_request.world_context
            self._agent_start_inputs[assignment.uav_id] = (focused, context)

    def _context_for(self, uav_id: str, assignment: FleetAssignment) -> object | None:
        if self._world_context_factory is not None:
            return self._world_context_factory(uav_id, assignment)
        if uav_id in self._world_contexts:
            return self._world_contexts[uav_id]
        getter = getattr(self.environment, "get_agent_world_context", None)
        if callable(getter):
            return getter(uav_id, assignment)
        return None

    def _bind_target_claims(self, plan: FleetMissionPlan) -> None:
        for assignment in plan.assignments:
            record = self.assignments.by_id(assignment.assignment_id)
            if (
                assignment.uav_id in self.agents
                and record.status in _STARTABLE_ASSIGNMENT_STATES
            ):
                self._bind_assignment_claim(assignment)

    def _bind_assignment_claim(self, assignment: FleetAssignment) -> None:
        if assignment.assignment_id in self._non_target_assignment_ids:
            return
        self.targets.bind_assignment(
            assignment_id=assignment.assignment_id,
            uav_id=assignment.uav_id,
            target_runtime_id=assignment.target_alias,
            semantic_alias=assignment.target_spec.original_description,
            priority=assignment.priority,
            timestamp_s=(
                0.0
                if self._last_airspace_decision is None
                else self._last_airspace_decision.timestamp_s
            ),
            provisional=True,
        )

    def _environment_assignments(
        self,
        plan: FleetMissionPlan,
        *,
        non_target_assignment_ids: set[str] | None = None,
    ) -> dict[str, str]:
        """Return only real semantic target routes for the environment.

        Compatibility aliases exist solely to satisfy the legacy V1 envelope;
        omitting them here prevents evaluator frames and Oracle routing from
        being manufactured for a targetless V2 Assignment.
        """

        excluded = (
            self._non_target_assignment_ids
            if non_target_assignment_ids is None
            else non_target_assignment_ids
        )
        return {
            assignment.uav_id: assignment.target_alias
            for assignment in plan.assignments
            if assignment.assignment_id not in excluded
        }

    def _start_environment(self, plan: FleetMissionPlan) -> None:
        method = getattr(self.environment, "start", None)
        if not callable(method):
            return
        parameters = inspect.signature(method).parameters
        if not parameters:
            method()
        else:
            method(plan)

    def _start_agent(self, uav_id: str, assignment: FleetAssignment) -> None:
        agent = self.agents[uav_id]
        focused, context = self._agent_start_inputs[uav_id]
        start_assignment = getattr(agent, "start_assignment", None)
        if callable(start_assignment):
            parameters = inspect.signature(start_assignment).parameters
            kwargs: dict[str, object] = {}
            if "instruction" in parameters:
                kwargs["instruction"] = focused
            if "world_context" in parameters:
                kwargs["world_context"] = context
            start_assignment(assignment, **kwargs)
            return
        start = getattr(agent, "start", None)
        if not callable(start):
            raise TypeError("MissionAgent must provide start()")
        parameters = inspect.signature(start).parameters
        if "world_context" in parameters:
            if context is None:
                raise FleetRuntimeError(f"world context is required for {uav_id}")
            start(focused, context)
        elif len(parameters) >= 2:
            start(focused, context)
        else:
            start(focused)

    def _advance_environment(self) -> object | None:
        barrier_method = getattr(self.environment, "step_fleet_barrier", None)
        if callable(barrier_method):
            return barrier_method()
        step = getattr(self.environment, "step", None)
        if callable(step):
            return step()
        tick = getattr(self.environment, "advance", None)
        if callable(tick):
            return tick()
        return None

    def _fleet_pose_snapshot(self, barrier: object | None) -> object | None:
        candidate = None
        if isinstance(barrier, Mapping):
            candidate = barrier.get("fleet_pose_snapshot")
        elif barrier is not None:
            candidate = getattr(barrier, "fleet_pose_snapshot", None)
        if candidate is None:
            getter = getattr(self.environment, "get_fleet_pose_snapshot", None)
            if callable(getter):
                candidate = getter()
        if candidate is None:
            return None
        snapshot = coerce_fleet_pose_snapshot(candidate)
        enriched = {}
        for uav_id, pose in snapshot.poses.items():
            try:
                record = self.assignments.for_uav(uav_id)
            except FleetRuntimeError:
                enriched[uav_id] = pose
                continue
            home_name = None
            if self._request is not None:
                home_name = self._request.uav(uav_id).home_name
            active_route = (
                ()
                if record.status in _TERMINAL_ASSIGNMENT_STATES
                else self._active_planned_route(
                    uav_id,
                    pose.position_xyz_m,
                    fallback=pose.route_xyz_m,
                )
            )
            enriched[uav_id] = replace(
                pose,
                priority=record.assignment.priority,
                assignment_id=record.assignment.assignment_id,
                route_xyz_m=active_route,
                landing_zone_id=home_name,
            )
        return FleetPoseSnapshot(snapshot.timestamp_s, enriched)

    def _active_planned_route(
        self,
        uav_id: str,
        current_position: tuple[float, float, float],
        *,
        fallback: tuple[tuple[float, float, float], ...],
    ) -> tuple[tuple[float, float, float], ...]:
        """Expose only the current route leg, advancing at reached waypoints.

        Feeding the complete outbound-and-return polyline into pairwise route
        checks would reserve every future crossing forever.  A current-position
        anchored leg lets a held UAV resume once the higher-priority vehicle
        has actually passed the crossing.
        """

        route = self._planned_routes.get(uav_id)
        if route is None:
            return fallback
        index = self._route_progress[uav_id]
        while index + 1 < len(route) and dist(current_position, route[index + 1]) <= 1.0:
            index += 1
        self._route_progress[uav_id] = index
        if index + 1 >= len(route):
            return ()
        return (current_position, route[index + 1])

    def _observation_for(
        self,
        uav_id: str,
        assignment: FleetAssignment,
        barrier: object | None,
    ) -> object:
        observations = None
        if isinstance(barrier, Mapping):
            observations = barrier.get("agent_observations")
        elif barrier is not None:
            observations = getattr(barrier, "agent_observations", None)
        if isinstance(observations, Mapping) and uav_id in observations:
            return observations[uav_id]
        perception = self._perceptions.get(uav_id)
        if perception is not None:
            inner_backend = getattr(perception, "backend", perception)
            if assignment.assignment_id in self._non_target_assignment_ids:
                if getattr(inner_backend, "target_id", None) is not None:
                    raise FleetRuntimeError(
                        "targetless assignment cannot use a target-bound perception backend"
                    )
                capability = _enum_text(
                    getattr(perception, "capability", "VISION")
                )
                if capability == "PRIVILEGED_ORACLE":
                    raise FleetRuntimeError(
                        "targetless assignment cannot use Oracle perception"
                    )
            bound_uav = getattr(inner_backend, "uav_id", uav_id)
            bound_target = getattr(inner_backend, "target_id", assignment.target_alias)
            if bound_uav != uav_id or bound_target != assignment.target_alias:
                raise FleetRuntimeError("perception backend routing does not match assignment")
            observe = getattr(perception, "observe", getattr(perception, "get_observation", None))
            if not callable(observe):
                raise TypeError("perception backend must provide observe()")
            capability = _enum_text(getattr(perception, "capability", "VISION"))
            if capability == "PRIVILEGED_ORACLE":
                frame_getter = getattr(self.environment, "get_evaluator_frame", None)
                if not callable(frame_getter):
                    raise FleetRuntimeError("Oracle perception requires evaluator frame API")
                source = frame_getter(uav_id, assignment.target_alias)
            else:
                raw_getter = getattr(self.environment, "get_skill_observation", None)
                if callable(raw_getter):
                    source = raw_getter(uav_id, include_oracle=False)
                    return observe(source)
                raw_getter = getattr(self.environment, "get_agent_observation", None)
                if not callable(raw_getter):
                    raise FleetRuntimeError("vision perception requires agent observation API")
                source = raw_getter(uav_id)
            return observe(source)
        getter = getattr(self.environment, "get_agent_observation", None)
        if callable(getter):
            return getter(uav_id)
        view = getattr(self.environment, "get_agent_view", None)
        if callable(view):
            return view(uav_id)
        raise FleetRuntimeError("environment does not provide per-agent observations")

    def _adopt_agent_snapshot(
        self,
        uav_id: str,
        assignment: FleetAssignment,
        snapshot: object,
    ) -> None:
        status = _enum_text(_snapshot_value(snapshot, "status", "RUNNING"))
        current = self.assignments.by_id(assignment.assignment_id)
        if assignment.assignment_id in self._local_failsafe_landings:
            if status in {"CANCELED", "FAILED", "SUCCEEDED"}:
                self._local_failsafe_landings.discard(assignment.assignment_id)
                self.assignments.update(
                    assignment.assignment_id,
                    AssignmentStatus.FAILED,
                    local_plan_version=current.local_plan_version,
                    last_error=current.last_error,
                )
                self._release_target_claim(
                    assignment.assignment_id,
                    timestamp_s=self._snapshot_timestamp(snapshot),
                )
                self._event(
                    "LOCAL_FAILSAFE_LAND_COMPLETED",
                    assignment_id=assignment.assignment_id,
                    uav_id=uav_id,
                    agent_terminal_status=status,
                )
            else:
                self.assignments.update(
                    assignment.assignment_id,
                    AssignmentStatus.CANCELING,
                    local_plan_version=current.local_plan_version,
                    last_error=current.last_error,
                )
            return
        version = self._validated_agent_plan_version(
            uav_id,
            snapshot,
            current_version=current.local_plan_version,
        )
        if status == "SUCCEEDED":
            assignment_status = AssignmentStatus.SUCCEEDED
            if assignment.assignment_id not in self._non_target_assignment_ids:
                try:
                    self.targets.terminate(
                        assignment.assignment_id,
                        timestamp_s=self._snapshot_timestamp(snapshot),
                    )
                except TargetClaimError:
                    pass
        elif status == "CANCELED" and self._cancel_requested:
            assignment_status = AssignmentStatus.CANCELED
            self._release_target_claim(
                assignment.assignment_id,
                timestamp_s=self._snapshot_timestamp(snapshot),
            )
        elif status in {"FAILED", "CANCELED"}:
            reason = _snapshot_value(snapshot, "last_error", status)
            self._mark_local_failure(assignment.assignment_id, str(reason))
            return
        else:
            assignment_status = (
                AssignmentStatus.CANCELING
                if self._cancel_requested
                else AssignmentStatus.RUNNING
            )
        self.assignments.update(
            assignment.assignment_id,
            assignment_status,
            local_plan_version=version,
        )

    def _synchronize_target_claim(
        self,
        uav_id: str,
        assignment: FleetAssignment,
        agent_snapshot: object,
    ) -> bool:
        if assignment.assignment_id in self._non_target_assignment_ids:
            return True
        target = _snapshot_value(agent_snapshot, "target")
        lifecycle = _enum_text(_snapshot_value(target, "lifecycle", "UNINITIALIZED"))
        if lifecycle not in {"LOCKED", "TRACKING"}:
            return True
        runtime_target_id = _snapshot_value(target, "target_id", assignment.target_alias)
        if not isinstance(runtime_target_id, str):
            runtime_target_id = assignment.target_alias
        if runtime_target_id != assignment.target_alias:
            event_type = "TARGET_ASSIGNMENT_MISMATCH"
            try:
                existing = self.targets.record(runtime_target_id)
            except TargetClaimError:
                existing = None
            if existing is not None and existing.active_claims:
                event_type = "TARGET_CLAIM_CONFLICT"
            self._event(
                event_type,
                uav_id=uav_id,
                assignment_id=assignment.assignment_id,
                observed_target_id=runtime_target_id,
                assigned_target_id=assignment.target_alias,
            )
            self._hold_uav(uav_id, event_type)
            self.assignments.update(
                assignment.assignment_id,
                AssignmentStatus.HOLDING,
                last_error=event_type,
            )
            return False
        confidence = _snapshot_value(target, "confidence", 1.0)
        timestamp = _snapshot_value(target, "last_seen_time_s", None)
        decision = self.targets.claim(
            assignment_id=assignment.assignment_id,
            uav_id=uav_id,
            target_runtime_id=runtime_target_id,
            confidence=1.0 if confidence is None else float(confidence),
            timestamp_s=(self._snapshot_timestamp(agent_snapshot) if timestamp is None else float(timestamp)),
            state=(
                TargetClaimState.EXCLUSIVE
                if self.targets.claim_policy.value == "EXCLUSIVE"
                else TargetClaimState.SHARED
            ),
        )
        if decision.conflict:
            self._event("TARGET_CLAIM_CONFLICT", decision=decision.to_dict())
            if decision.hold_uav_id is not None:
                self._hold_uav(decision.hold_uav_id, "TARGET_CLAIM_CONFLICT")
                try:
                    loser = self.assignments.for_uav(decision.hold_uav_id)
                except FleetRuntimeError:
                    loser = None
                if (
                    loser is not None
                    and loser.status not in _TERMINAL_ASSIGNMENT_STATES
                ):
                    self.assignments.update(
                        loser.assignment.assignment_id,
                        AssignmentStatus.HOLDING,
                        local_plan_version=loser.local_plan_version,
                        last_error="TARGET_CLAIM_CONFLICT",
                    )
        return decision.accepted

    def _mark_local_failure(self, assignment_id: str, reason: str) -> None:
        policy = (
            self._plan.coordination_policy.assignment_failure_policy
            if self._plan is not None
            else self._coordination_policy.assignment_failure_policy
        )
        status = (
            AssignmentStatus.WAITING_REASSIGNMENT
            if policy is AssignmentFailurePolicy.REPORT_AND_REPLAN
            and self._replan_handler is not None
            else AssignmentStatus.REASSIGNMENT_REQUIRED
            if policy is AssignmentFailurePolicy.REPORT_AND_REPLAN
            else AssignmentStatus.FAILED
        )
        self.assignments.update(assignment_id, status, last_error=reason)
        self._release_target_claim(assignment_id)
        record = self.assignments.by_id(assignment_id)
        self._hold_uav(record.assignment.uav_id, "LOCAL_AGENT_FAILURE")
        if status is AssignmentStatus.WAITING_REASSIGNMENT:
            self._pending_reassignments.add(assignment_id)
        self._event(
            "FLEET_REPLAN_REQUESTED"
            if status is AssignmentStatus.WAITING_REASSIGNMENT
            else "REASSIGNMENT_REQUIRED"
            if status is AssignmentStatus.REASSIGNMENT_REQUIRED
            else "ASSIGNMENT_FAILED",
            assignment_id=assignment_id,
            error=reason,
        )
        if status is AssignmentStatus.REASSIGNMENT_REQUIRED:
            self._begin_local_failsafe_landing(
                assignment_id,
                "no Fleet replan handler is available; " + reason,
            )
        if policy is AssignmentFailurePolicy.CANCEL_FLEET:
            self._cancel_requested = True
            self._event(
                "FLEET_CANCEL_REQUESTED",
                reason="assignment failure policy",
                failed_assignment_id=assignment_id,
            )
            for record in self.assignments.records:
                if record.assignment.assignment_id == assignment_id:
                    continue
                if record.status is AssignmentStatus.PENDING:
                    self.assignments.update(
                        record.assignment.assignment_id,
                        AssignmentStatus.CANCELED,
                        last_error="fleet failure policy",
                    )
                    self._release_target_claim(
                        record.assignment.assignment_id
                    )
                    continue
                if record.status in _TERMINAL_ASSIGNMENT_STATES:
                    continue
                agent = self.agents.get(record.assignment.uav_id)
                try:
                    result = agent.cancel() if agent is not None else None
                    if result is None and agent is not None:
                        result = self._agent_snapshot(agent)
                    agent_status = _enum_text(
                        _snapshot_value(result, "status", "RUNNING")
                    )
                except Exception as exc:
                    self.assignments.update(
                        record.assignment.assignment_id,
                        AssignmentStatus.FAILED,
                        last_error=(
                            "fleet failure cancel failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                    self._release_target_claim(
                        record.assignment.assignment_id
                    )
                    continue
                terminal = {
                    "CANCELED": AssignmentStatus.CANCELED,
                    "FAILED": AssignmentStatus.FAILED,
                    "SUCCEEDED": AssignmentStatus.SUCCEEDED,
                }.get(agent_status)
                self.assignments.update(
                    record.assignment.assignment_id,
                    terminal or AssignmentStatus.CANCELING,
                    last_error=(
                        "fleet failure policy"
                        if terminal is not AssignmentStatus.SUCCEEDED
                        else None
                    ),
                )
                if terminal in {
                    AssignmentStatus.CANCELED,
                    AssignmentStatus.FAILED,
                }:
                    self._release_target_claim(
                        record.assignment.assignment_id
                    )
                elif terminal is AssignmentStatus.SUCCEEDED:
                    try:
                        self.targets.terminate(
                            record.assignment.assignment_id,
                            timestamp_s=(
                                0.0
                                if self._last_airspace_decision is None
                                else self._last_airspace_decision.timestamp_s
                            ),
                        )
                    except TargetClaimError:
                        pass

    def _begin_local_failsafe_landing(
        self,
        assignment_id: str,
        reason: str,
    ) -> None:
        """Cancel-and-land one failed UAV without canceling its healthy peers."""

        record = self.assignments.by_id(assignment_id)
        uav_id = record.assignment.uav_id
        self._pending_reassignments.discard(assignment_id)
        self._local_failsafe_landings.add(assignment_id)
        agent = self.agents.get(uav_id)
        try:
            result = None if agent is None else agent.cancel()
            if result is None and agent is not None:
                result = self._agent_snapshot(agent)
        except Exception as exc:
            self._local_failsafe_landings.discard(assignment_id)
            self.assignments.update(
                assignment_id,
                AssignmentStatus.FAILED,
                last_error=(
                    reason
                    + "; local cancel-and-land failed: "
                    + f"{type(exc).__name__}: {exc}"
                ),
            )
            self._event(
                "LOCAL_FAILSAFE_LAND_FAILED",
                assignment_id=assignment_id,
                uav_id=uav_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        status = _enum_text(_snapshot_value(result, "status", "RUNNING"))
        if status in {"CANCELED", "FAILED", "SUCCEEDED"}:
            self._local_failsafe_landings.discard(assignment_id)
            assignment_status = AssignmentStatus.FAILED
        else:
            assignment_status = AssignmentStatus.CANCELING
        self.assignments.update(
            assignment_id,
            assignment_status,
            last_error=reason,
        )
        self._event(
            "LOCAL_FAILSAFE_LAND_REQUESTED",
            assignment_id=assignment_id,
            uav_id=uav_id,
            state=assignment_status.value,
            reason=reason,
        )

    def _aggregate_status(self) -> None:
        if self._plan is not None and self._request is not None:
            try:
                validate_fleet_mission_plan(self._plan, self._request)
            except (TypeError, ValueError) as exc:
                self._status = FleetStatus.FAILED
                error = (
                    "runtime required-target coverage is invalid: "
                    f"{type(exc).__name__}: {exc}"
                )
                if self._last_error != error:
                    self._last_error = error
                    self._event("FLEET_COVERAGE_INVALID", error=error)
                return
        if self._retired_failsafe_agents:
            self._status = FleetStatus.RUNNING
            return
        records = self.assignments.records
        if any(record.status not in _TERMINAL_ASSIGNMENT_STATES for record in records):
            self._status = FleetStatus.RUNNING
            return
        successes = sum(record.status is AssignmentStatus.SUCCEEDED for record in records)
        required = tuple(record for record in records if record.required)
        required_succeeded = bool(required) and all(
            record.status is AssignmentStatus.SUCCEEDED for record in required
        )
        failures = len(records) - successes
        unassigned = bool(self._plan and self._plan.unassigned_requirements)
        if self._cancel_requested:
            failed_cancel = any(
                record.status
                in {AssignmentStatus.FAILED, AssignmentStatus.REASSIGNMENT_REQUIRED}
                for record in records
            )
            if not failed_cancel:
                self._status = FleetStatus.CANCELED
            elif successes:
                self._status = FleetStatus.PARTIAL_SUCCESS
            else:
                self._status = FleetStatus.FAILED
            return
        if records and not unassigned and (
            failures == 0 or required_succeeded
        ):
            self._status = FleetStatus.SUCCEEDED
        elif successes:
            self._status = FleetStatus.PARTIAL_SUCCESS
        else:
            self._status = FleetStatus.FAILED

    def _validated_agent_plan_version(
        self,
        uav_id: str,
        snapshot: object,
        *,
        current_version: int,
        allow_initial_jump: bool = False,
    ) -> int:
        reported_uav_id = _snapshot_value(snapshot, "uav_id", None)
        if reported_uav_id is not None and reported_uav_id != uav_id:
            raise FleetRuntimeError(
                f"agent snapshot routing mismatch for {uav_id}: {reported_uav_id}"
            )
        raw_version = _snapshot_value(snapshot, "plan_version", None)
        if raw_version is None:
            return current_version
        version = _optional_int(raw_version)
        if version is None or version <= 0:
            raise FleetRuntimeError("agent plan_version must be a positive integer")
        if not allow_initial_jump and version < current_version:
            raise FleetRuntimeError(
                f"agent plan_version decreased for {uav_id}: "
                f"{current_version} -> {version}"
            )
        return version

    def _release_target_claim(
        self,
        assignment_id: str,
        *,
        timestamp_s: float | None = None,
    ) -> None:
        if assignment_id in self._non_target_assignment_ids:
            return
        timestamp = (
            0.0
            if timestamp_s is None and self._last_airspace_decision is None
            else self._last_airspace_decision.timestamp_s
            if timestamp_s is None
            else timestamp_s
        )
        try:
            self.targets.release(assignment_id, timestamp_s=timestamp)
        except TargetClaimError:
            pass

    def _hold_uav(self, uav_id: str, reason: str) -> None:
        uav_id = validate_uav_id(uav_id)
        for method_name in ("hold_uav", "emergency_hold"):
            method = getattr(self.environment, method_name, None)
            if callable(method):
                method(uav_id)
                break
        else:
            controllers = getattr(self.environment, "uav_controllers", None)
            if isinstance(controllers, Mapping) and uav_id in controllers:
                stop = getattr(controllers[uav_id], "stop", None)
                if callable(stop):
                    stop()
        self._event("UAV_HOLD", uav_id=uav_id, reason=reason)

    def _refresh_world_belief(self) -> None:
        plan = self._plan
        if plan is None:
            self._world_belief = None
            return
        timestamp_s = (
            0.0
            if self._last_airspace_decision is None
            else self._last_airspace_decision.timestamp_s
        )
        summaries: dict[str, AgentFleetSummary] = {}
        for record in self.assignments.records:
            shape = getattr(record.assignment.search_region, "shape", None)
            current_region = (
                None
                if record.assignment.assignment_id
                in self._non_target_assignment_ids
                or shape is None
                else _enum_text(shape)
            )
            summaries[record.assignment.uav_id] = AgentFleetSummary(
                uav_id=record.assignment.uav_id,
                assignment_id=record.assignment.assignment_id,
                status=record.status.value,
                plan_version=record.local_plan_version,
                current_region=current_region,
                altitude_layer=None,
            )
        self._world_belief = FleetWorldBelief(
            fleet_mission_id=plan.fleet_mission_id,
            fleet_plan_version=plan.fleet_plan_version,
            timestamp_s=timestamp_s,
            agents=summaries,
            target_claims=self.targets.snapshot(),
            airspace=(
                {}
                if self._last_airspace_decision is None
                else self._last_airspace_decision.to_dict()
            ),
            events=tuple(self._events[-64:]),
        )

    @staticmethod
    def _agent_snapshot(agent: object) -> object:
        return agent.snapshot()

    def _snapshot_timestamp(self, snapshot: object) -> float:
        target = _snapshot_value(snapshot, "target")
        value = _snapshot_value(target, "last_seen_time_s", None)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
        pose = self._last_airspace_decision
        return 0.0 if pose is None else pose.timestamp_s

    def _event(self, event_type: str, **payload: object) -> None:
        event = {"event_type": event_type, **deepcopy(payload)}
        self._events.append(event)
        if len(self._events) > 10_000:
            self._events.pop(0)
        method = getattr(self._logger, "log_fleet_event", None)
        if callable(method):
            method(event)

    def _log_airspace(self, decision: FleetAirspaceDecision) -> None:
        method = getattr(self._logger, "log_airspace_decision", None)
        if callable(method):
            method(decision)

    def _write_initial_logs(self) -> None:
        if self._logger is None or self._plan is None:
            return
        write_plan = self._runtime_plan_log_method()
        if callable(write_plan):
            write_plan(self._plan)
        write_assignments = getattr(self._logger, "write_assignments", None)
        if callable(write_assignments):
            write_assignments(self._assignment_log_rows())

    def _assignment_log_rows(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for record in self.assignments.records:
            row = record.to_dict()
            assignment_id = record.assignment.assignment_id
            if assignment_id in self._non_target_assignment_ids:
                # Never expose the structural V1 alias as a semantic target.
                row.update(
                    {
                        "target_alias": None,
                        "target_spec": None,
                        "search_region": None,
                        "track_duration_s": None,
                        "non_target_assignment": True,
                    }
                )
            else:
                row["non_target_assignment"] = False
            rows.append(row)
        return tuple(rows)

    def _runtime_plan_log_method(self) -> object:
        """Keep a semantic V2 artifact separate from its V1 execution envelope."""

        source = str(getattr(self.fleet_planner, "source", ""))
        if source == "fleet_llm_v2":
            method = getattr(self._logger, "write_runtime_execution_plan", None)
            if callable(method):
                return method
        return getattr(self._logger, "write_fleet_plan", None)

    def _write_summary_log(self) -> None:
        write_assignments = getattr(self._logger, "write_assignments", None)
        if callable(write_assignments):
            write_assignments(self._assignment_log_rows())
        method = getattr(self._logger, "write_summary", None)
        if callable(method):
            method(self.snapshot().to_summary_dict())


def _focused_assignment_instruction(
    request: FleetMissionRequest,
    assignment: FleetAssignment,
) -> str:
    return json.dumps(
        {
            "task": "Plan only this independent UAV assignment with Spatial Contract V3.",
            "original_instruction": request.original_instruction,
            "fleet_mission_id": request.fleet_mission_id,
            "assignment": assignment.to_dict(),
            "requirements": [
                "Do not inspect or plan another UAV assignment.",
                "Take off, search this RegionSpec, track for the exact duration, return home, and land.",
            ],
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _snapshot_value(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_text(value: object) -> str:
    return str(getattr(value, "value", value))


def _optional_int(value: object) -> int | None:
    return None if isinstance(value, bool) or not isinstance(value, int) else value


__all__ = [
    "AssignmentRegistry",
    "AssignmentRuntimeRecord",
    "AssignmentStatus",
    "FleetMissionRuntime",
    "FleetReplanPublication",
    "FleetRuntimeError",
    "FleetRuntimeSnapshot",
    "FleetStatus",
    "ReplannedAssignment",
]
