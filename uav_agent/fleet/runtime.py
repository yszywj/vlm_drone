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
from math import dist
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
    CANCELED = "CANCELED"


class AssignmentStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    HOLDING = "HOLDING"
    CANCELING = "CANCELING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    REASSIGNMENT_REQUIRED = "REASSIGNMENT_REQUIRED"


_TERMINAL_ASSIGNMENT_STATES = frozenset(
    {
        AssignmentStatus.SUCCEEDED,
        AssignmentStatus.FAILED,
        AssignmentStatus.CANCELED,
        AssignmentStatus.REASSIGNMENT_REQUIRED,
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


class AssignmentRegistry:
    """Mutable runtime state keyed independently by assignment and UAV."""

    def __init__(self) -> None:
        self._records: dict[str, AssignmentRuntimeRecord] = {}
        self._active_by_uav: dict[str, str] = {}

    def initialize(
        self,
        plan: FleetMissionPlan,
        request: FleetMissionRequest,
    ) -> None:
        if self._records:
            raise FleetRuntimeError("AssignmentRegistry is already initialized")
        required_by_alias = {
            item.target_alias: item.required for item in request.target_requests
        }
        for assignment in plan.assignments:
            if assignment.assignment_id in self._records:
                raise FleetRuntimeError("duplicate assignment_id")
            if assignment.uav_id in self._active_by_uav:
                raise FleetRuntimeError(
                    "FleetMissionRuntime v1 supports one active assignment per UAV"
                )
            self._records[assignment.assignment_id] = AssignmentRuntimeRecord(
                assignment=assignment,
                required=required_by_alias.get(assignment.target_alias, True),
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
        world_context_factory: Callable[[str, FleetAssignment], object] | None = None,
        perceptions: Mapping[str, object] | None = None,
        planned_routes: Mapping[str, Sequence[Sequence[float]]] | None = None,
        targets: SharedTargetRegistry | None = None,
        airspace: FleetAirspaceManager | None = None,
        model_broker: GlobalModelRequestBroker | None = None,
        logger: object | None = None,
    ) -> None:
        if not callable(getattr(fleet_planner, "plan", None)):
            raise TypeError("fleet_planner must provide plan()")
        if not isinstance(agents, Mapping) or not agents:
            raise TypeError("agents must be a non-empty mapping")
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
            unknown_agents = sorted(
                {assignment.uav_id for assignment in plan.assignments} - set(self.agents)
            )
            if unknown_agents:
                raise FleetRuntimeError(
                    "no MissionAgent exists for assigned UAVs: " + ", ".join(unknown_agents)
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
            self.assignments.initialize(plan, trusted_request)
            self._prepare_agent_start_inputs(trusted_request, plan)
            self._bind_target_claims(plan)
        except Exception as exc:
            self._status = FleetStatus.FAILED
            self._last_error = f"fleet planning failed: {type(exc).__name__}: {exc}"
            self._event("FLEET_PLANNING_FAILED", error=self._last_error)
            raise FleetRuntimeError(self._last_error) from exc

        self._request = trusted_request
        self._plan = plan
        try:
            bind_environment = getattr(self.environment, "set_assignments", None)
            if callable(bind_environment):
                bind_environment(
                    {
                        assignment.uav_id: assignment.target_alias
                        for assignment in plan.assignments
                    }
                )
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
        for uav_id in sorted({item.uav_id for item in plan.assignments}):
            record = self.assignments.for_uav(uav_id)
            if record.status in _TERMINAL_ASSIGNMENT_STATES:
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

        for uav_id in sorted(self.agents):
            try:
                record = self.assignments.for_uav(uav_id)
            except FleetRuntimeError:
                continue
            if record.status in _TERMINAL_ASSIGNMENT_STATES:
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

        if not self._cancel_requested:
            return set()
        overrides: set[str] = set()
        for uav_id in held_uav_ids:
            try:
                record = self.assignments.for_uav(uav_id)
            except FleetRuntimeError:
                continue
            if record.status is AssignmentStatus.CANCELING:
                overrides.add(uav_id)
        return overrides

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
            assignments=self.assignments.snapshot(),
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
            self.targets.bind_assignment(
                assignment_id=assignment.assignment_id,
                uav_id=assignment.uav_id,
                target_runtime_id=assignment.target_alias,
                semantic_alias=assignment.target_spec.original_description,
                priority=assignment.priority,
                provisional=True,
            )

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
        version = self._validated_agent_plan_version(
            uav_id,
            snapshot,
            current_version=current.local_plan_version,
        )
        if status == "SUCCEEDED":
            assignment_status = AssignmentStatus.SUCCEEDED
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
            AssignmentStatus.REASSIGNMENT_REQUIRED
            if policy is AssignmentFailurePolicy.REPORT_AND_REPLAN
            else AssignmentStatus.FAILED
        )
        self.assignments.update(assignment_id, status, last_error=reason)
        self._release_target_claim(assignment_id)
        self._event(
            "REASSIGNMENT_REQUIRED" if status is AssignmentStatus.REASSIGNMENT_REQUIRED else "ASSIGNMENT_FAILED",
            assignment_id=assignment_id,
            error=reason,
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
            current_region = None if shape is None else _enum_text(shape)
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
        write_plan = getattr(self._logger, "write_fleet_plan", None)
        if callable(write_plan):
            write_plan(self._plan)
        write_assignments = getattr(self._logger, "write_assignments", None)
        if callable(write_assignments):
            write_assignments(tuple(record.to_dict() for record in self.assignments.records))

    def _write_summary_log(self) -> None:
        write_assignments = getattr(self._logger, "write_assignments", None)
        if callable(write_assignments):
            write_assignments(
                tuple(record.to_dict() for record in self.assignments.records)
            )
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
    "FleetRuntimeError",
    "FleetRuntimeSnapshot",
    "FleetStatus",
]
