"""Owner-driven, opt-in Fleet recovery over the existing Broker and SkillManager.

Only immutable request values cross into model threads. Flight permission is
published by Fleet's single writer after a second live-state/deadline check.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from math import isfinite
from threading import get_ident
from time import monotonic
from typing import Callable

from common.ids import generate_routing_id
from configs.schema import FleetRecoveryConfig
from fleet.airspace_manager import FleetAirspaceManager, FleetPoseSnapshot, coerce_fleet_pose_snapshot
from fleet.local_repair import (
    DependencyVerdict, JointRepairRequest, LocalRepairCandidate, LocalRepairContextV3,
    LocalRepairDraftV3, LocalRepairError, RepairAnchor, SkillExecutionEvidence,
    SpatialReferenceVerdict, Transferability, build_joint_repair_json_schema,
    build_remaining_task_contract, check_cross_uav_dependencies, evaluate_spatial_reference,
    build_local_repair_json_schema, extract_external_dependencies, parse_joint_repair_drafts,
    reproject_world_route, validate_joint_repair, validate_local_repair,
)
from fleet.model_request_broker import ModelBrokerRequest, ModelRequestPriority
from fleet.model_request_dispatcher import BrokeredTextTaskRunner
from fleet.strict_json import strict_json_object_loads
from models.adapter_registry import ModelCallRole
from models.base import ChatMessage, GenerationOptions, JsonSchemaResponseFormat, ModelResponse
from planner.schemas_v3 import SkillPlanDraftV3
from planner.spatial_resolver import FramePose
from runtime.plan_validator import PlanValidator
from skills.types import SkillExecutionReport, SkillResult, SkillStatus


@dataclass
class RecoveryEpisode:
    assignment_id: str
    episode_id: str
    event_id: str | None
    generation: int
    opened_wall_s: float
    deadline_wall_s: float
    base_version: int
    request_id: str | None = None
    phase: str = "WAIT_HOLD"
    attempts: int = 0
    reassign_attempts: int = 0
    next_attempt_wall_s: float = 0.0
    context: LocalRepairContextV3 | None = None
    candidate: object | None = None
    queue_signature: tuple = ()
    submitted_wall_s: float | None = None
    external_dependencies: tuple = ()
    dependency_versions: object | None = None
    remaining_goal_ids: tuple[str, ...] = ()
    confirmed_goal_evidence: tuple = ()
    source_uav_id: str | None = None
    joint_scope: tuple[str, ...] = ()
    joint_contexts: object | None = None


class FleetRecoveryController:
    """A bounded recovery owner, called only at Fleet tick's serial event boundary."""

    def __init__(self, *, config: FleetRecoveryConfig, broker, task_spec,
                 assignments, compilations, world_contexts, home_names, client_factory,
                 geometry_provider: Callable, replan_boundary=None, planner_limits=None,
                 planner_policy=None, clock=monotonic, runner=None, goal_contracts=None):
        if not config.enabled:
            raise ValueError("FleetRecoveryController requires explicit enabled configuration")
        if not callable(geometry_provider) or not callable(clock):
            raise TypeError("recovery requires a trusted geometry provider and monotonic clock")
        self.config, self.clock = config, clock
        # Every completion semantic is registry-driven; adding task types does
        # not modify this controller.
        from fleet.contract_registry import DEFAULT_GOAL_CONTRACT_REGISTRY
        self._goal_contracts = DEFAULT_GOAL_CONTRACT_REGISTRY if goal_contracts is None else goal_contracts
        self.task_spec = task_spec
        # A handoff's WORLD-bound goal meanings must survive subsequent local
        # repairs and another handoff, independently of the spare's home/start.
        self.assignment_task_specs = {}
        self.confirmed_goal_evidence = {}
        self.assignments = {a.assignment_id: a for a in assignments}
        self.compilations = dict(compilations)
        self.world_contexts, self.home_names = dict(world_contexts), dict(home_names)
        self.client_factory, self.geometry_provider = client_factory, geometry_provider
        self.replan_boundary = replan_boundary
        self.limits, self.policy = planner_limits, planner_policy
        self.runner = runner or BrokeredTextTaskRunner(
            broker, clock=clock, max_workers=config.max_concurrent_requests)
        self.runtime = None
        self.episodes: dict[str, RecoveryEpisode] = {}
        self.finished_events: set[str] = set()
        self._owner = get_ident()
        self._closed = False
        self._epoch = 0
        self._last_error: dict[str, str] = {}
        # Resolve now: placeholders use exactly the registry's existing policy.
        self.local_selection = client_factory.selection_for_role(ModelCallRole.RUNTIME_REPLAN)
        self.fleet_selection = client_factory.selection_for_role(ModelCallRole.FLEET_REPLAN)
        if config.mode == "LOCAL_THEN_REASSIGN" and replan_boundary is None:
            raise ValueError("async reassignment boundary is required in LOCAL_THEN_REASSIGN mode")
        if replan_boundary is not None:
            for name in ("snapshot_request", "compute_candidate", "prepare_candidate", "committed"):
                if not callable(getattr(replan_boundary, name, None)):
                    raise TypeError("reassignment boundary must expose " + name)

    @property
    def supports_reassignment(self):
        return self.config.mode == "LOCAL_THEN_REASSIGN" and self.replan_boundary is not None

    def bind(self, runtime):
        self._assert_owner()
        self.runtime = runtime
        for uav, agent in runtime.agents.items():
            configure = getattr(agent, "configure_local_repair", None)
            if not callable(configure):
                raise TypeError("enabled Fleet recovery requires MissionAgent local repair support")
            configure(enabled=True, max_wait_s=max(300.0, self.config.episode_timeout_s * 2))
        self.geometry_provider()  # Fail before flight on a missing trusted provider.

    def _assert_owner(self):
        if get_ident() != self._owner:
            raise RuntimeError("Fleet recovery has a single execution-state writer")

    def _log(self, event, episode, **values):
        self.runtime._event(event, assignment_id=episode.assignment_id,
            episode_id=episode.episode_id, request_id=episode.request_id,
            generation=episode.generation, base_local_version=episode.base_version,
            uav_id=episode.source_uav_id,
            scope="ASSIGNMENT_HANDOFF" if episode.phase.startswith("REASSIGN") else "LOCAL_SUFFIX",
            replace_from_step_id=None if episode.context is None else episode.context.current_step_id,
            completed_prefix_length=0 if episode.context is None else len(episode.context.completed_step_ids),
            episode_deadline_wall_s=episode.deadline_wall_s,
            wall_timestamp_s=self.clock(), phase=episode.phase,
            anchor_id=None if episode.context is None else episode.context.anchor.anchor_id,
            **values)

    def _record(self, episode):
        return self.runtime.assignments.by_id(episode.assignment_id)

    def _task_spec_for(self, assignment_id):
        return self.assignment_task_specs.get(assignment_id, self.task_spec)

    def _state(self, episode):
        record = self._record(episode)
        agent = self.runtime.agents[record.assignment.uav_id]
        return record, agent, agent.local_repair_snapshot

    def _signature(self, episode):
        record, agent, state = self._state(episode)
        event = state.event
        return (record.assignment.assignment_id, record.assignment.uav_id,
                record.local_plan_version, self.runtime.fleet_plan.fleet_plan_version,
                None if event is None else event.event_id, state.execution_epoch,
                state.current_step_id, str(getattr(agent.snapshot().status, "value", agent.snapshot().status)))

    def _versions(self):
        # Snapshot the current owner versions; individual requests retain only dependencies.
        versions = {"fleet": self.runtime.fleet_plan.fleet_plan_version}
        for record in self.runtime.assignments.records:
            versions[record.assignment.assignment_id] = record.local_plan_version
        return versions

    def _evidence(self, state):
        if state.task_plan is None:
            return ()
        plan = state.task_plan
        planned = {step.step_id: step.skill for step in plan.steps}
        invocations = {item.invocation_id: item for item in state.started_invocations}
        values = []
        for report in state.execution_reports:
            if not isinstance(report, SkillExecutionReport):
                continue
            invocation = invocations.get(report.invocation_id)
            if (report.result_code is None
                    or report.status not in {SkillStatus.SUCCEEDED, SkillStatus.FAILED, SkillStatus.CANCELED}
                    or report.mission_id != plan.mission_id or report.uav_id != plan.uav_id
                    or report.plan_version > plan.plan_version
                    or planned.get(report.step_id) != report.skill_name
                    or invocation is None or invocation.step_id != report.step_id
                    or invocation.skill_name != report.skill_name
                    or invocation.plan_version != report.plan_version):
                continue
            # Manager terminal reports contain SkillResult.to_dict(); ordinary
            # feedback and temporary HOVER/REACQUIRE invocations earn no credit.
            payload = report.feedback_or_result
            data = payload.get("data", {})
            if not isinstance(data, Mapping):
                continue
            result = SkillResult(report.status, report.result_code,
                                 str(payload.get("message", "")), dict(data))
            values.append(SkillExecutionEvidence(report.step_id, report.invocation_id,
                                                 report.plan_version, result))
        return tuple(values)

    def _goal_state(self):
        states = {goal.goal_id: "UNKNOWN" for goal in self.task_spec.goals + self.task_spec.termination_goals}
        refs = dict(self.confirmed_goal_evidence)
        states.update({goal_id: "CONFIRMED" for goal_id in refs})
        for assignment_id, assignment in self.assignments.items():
            try:
                record = self.runtime.assignments.by_id(assignment_id)
                agent = self.runtime.agents[record.assignment.uav_id]
                state = agent.local_repair_snapshot
                original = state.compiled_mission
                if original is None or not isinstance(original.planner_output, SkillPlanDraftV3):
                    continue
                goals = tuple(self._task_spec_for(assignment_id).goal(goal) for goal in assignment.goal_ids)
                completed = tuple(step.step_id for step in state.task_plan.steps[:state.current_step_index])
                contract = build_remaining_task_contract(goals, original.planner_output, completed,
                    self._evidence(state), current_step_id=state.current_step_id,
                    current_step_started=True, home_name=self.home_names[record.assignment.uav_id],
                    consumer="FLEET_STATE")
                for goal in contract.pending_goals:
                    states[goal.goal_id] = "PENDING"
                    # Owner-produced assignment/version evidence, not a model
                    # completion claim. ACTIVE below removes this permission.
                    refs[goal.goal_id] = (f"pending_{assignment_id}_v{record.local_plan_version}",)
                for entry in contract.goals:
                    if entry.status != "CONFIRMED":
                        continue
                    goal_id = entry.goal_id
                    states[goal_id] = "CONFIRMED"
                    # Terminal invocation references only; a confirmed goal
                    # without them keeps no stable owner evidence downstream.
                    refs[goal_id] = entry.evidence_refs
                # Incomplete/active external successors must not be called pending.
                if state.current_step_id is not None:
                    from fleet.local_repair import matching_steps_for
                    for goal in goals:
                        if any(step.id == state.current_step_id for step in matching_steps_for(
                                goal, original.planner_output.steps, self.home_names[record.assignment.uav_id])):
                            if states[goal.goal_id] != "CONFIRMED":
                                states[goal.goal_id] = "ACTIVE"
                                refs.pop(goal.goal_id, None)
            except (AttributeError, KeyError, ValueError):
                # Missing evidence is UNKNOWN, never model-authorized completion.
                continue
        return states, refs

    def _dependencies(self, assignment_id, *, goal_ids=None):
        states, refs = self._goal_state()
        geometry = self.geometry_provider()
        versions = {"fleet": self.runtime.fleet_plan.fleet_plan_version,
                    assignment_id: self.runtime.assignments.by_id(assignment_id).local_plan_version}
        versions["reference"] = int(geometry["reference_version"])
        versions["map"] = int(geometry["map_version"])
        dependencies = extract_external_dependencies(
            self.task_spec, self.assignments[assignment_id].goal_ids if goal_ids is None else goal_ids,
            assignments=tuple(self.assignments.values()), goal_states=states,
            goal_evidence_refs=refs, versions=versions,
            shared_dependencies=tuple(geometry.get("shared_dependencies", ())))
        dependent_uavs = {uav for dependency in dependencies for uav in dependency.uav_ids}
        for record in self.runtime.assignments.records:
            if record.assignment.uav_id in dependent_uavs:
                versions[record.assignment.assignment_id] = record.local_plan_version
        return dependencies, versions

    def _new_episode(self, record, state=None):
        self._epoch += 1
        now = self.clock()
        event = None if state is None else state.event
        episode = RecoveryEpisode(record.assignment.assignment_id, generate_routing_id("recovery"),
            None if event is None else event.event_id,
            self._epoch if state is None else state.execution_epoch,
            now, now + self.config.episode_timeout_s, record.local_plan_version,
            source_uav_id=record.assignment.uav_id)
        self.episodes[episode.assignment_id] = episode
        self._log("RECOVERY_EPISODE_OPENED", episode,
                  route="LOCAL_REPAIR" if event is not None else "REASSIGNMENT")
        return episode

    def tick(self):
        self._assert_owner()
        if self._closed or self.runtime is None:
            return
        if self.runtime.cancel_requested:
            self.cancel_all("USER_CANCEL")
            return
        # Capture only actual Manager-originated recoverable events. Ordinary
        # feedback and deterministic REACQUIRE never enter this model path.
        for record in tuple(self.runtime.assignments.records):
            agent = self.runtime.agents.get(record.assignment.uav_id)
            state = getattr(agent, "local_repair_snapshot", None)
            if state is not None and state.event is not None:
                if record.assignment.assignment_id not in self.episodes and state.event.event_id not in self.finished_events:
                    self._new_episode(record, state)
        for episode in tuple(self.episodes.values()):
            try:
                if not self._owns_execution(episode):
                    self._finish(episode, "STALE_EXECUTION")
                    continue
                if self.clock() >= episode.deadline_wall_s:
                    self._exit(episode, "EPISODE_DEADLINE_EXPIRED")
                    continue
                if episode.phase.startswith("REASSIGN"):
                    self._tick_reassignment(episode)
                elif episode.phase.startswith("JOINT"):
                    self._tick_joint(episode)
                else:
                    self._tick_local(episode)
            except Exception as exc:
                self._handle_failure(episode, getattr(exc, "code", type(exc).__name__),
                                     affected=getattr(exc, "affected_uav_ids", ()))
        self.runner.pump()

    def service_reassignments(self):
        self._assert_owner()
        for assignment_id in tuple(self.runtime._pending_reassignments):
            if assignment_id not in self.episodes:
                record = self.runtime.assignments.by_id(assignment_id)
                agent = self.runtime.agents[record.assignment.uav_id]
                episode = self._new_episode(record, agent.local_repair_snapshot)
                episode.phase = "REASSIGN_QUEUED"
                self._safe_source_wait(episode)

    def _owns_execution(self, episode):
        if self.episodes.get(episode.assignment_id) is not episode:
            return False
        try:
            record, agent, state = self._state(episode)
        except (KeyError, AttributeError):
            return False
        if (record.assignment.uav_id != episode.source_uav_id
                or record.local_plan_version != episode.base_version):
            return False
        if episode.phase.startswith("REASSIGN"):
            return (record.status.value == "WAITING_REASSIGNMENT"
                    and state.execution_epoch == episode.generation)
        return (state.event is not None and state.event.event_id == episode.event_id
                and state.execution_epoch == episode.generation
                and state.current_step_id == state.event.step_id
                and state.task_plan is not None
                and state.task_plan.plan_version == episode.base_version)

    def _safe_source_wait(self, episode):
        record, agent, state = self._state(episode)
        if state.event is None and getattr(agent.snapshot().status, "value", agent.snapshot().status) == "RUNNING":
            # A non-repairable or terminal source must not wait indefinitely in
            # the air. Let the existing cancel-and-land controller run while
            # an idle replacement is computed.
            agent.cancel()

    def _tick_local(self, episode):
        record, agent, state = self._state(episode)
        if state.event is None or state.event.event_id != episode.event_id:
            self._finish(episode, "STALE_EVENT")
            return
        if state.execution_epoch != episode.generation or record.local_plan_version != episode.base_version:
            self._finish(episode, "STALE_EXECUTION")
            return
        if episode.request_id is None:
            if not state.stable_hold or self.clock() < episode.next_attempt_wall_s:
                return
            if episode.attempts >= self.config.max_local_attempts:
                self._escalate(episode, "LOCAL_RETRY_BUDGET_EXHAUSTED")
                return
            episode.attempts += 1
            episode.request_id = generate_routing_id("request")
            episode.phase = "LOCAL_QUEUED"
            episode.queue_signature = self._signature(episode)
            request = self._broker_request(episode, ModelCallRole.RUNTIME_REPLAN, self.local_selection)
            deadline = min(episode.deadline_wall_s, self.clock() + self.config.request_timeout_s)
            self.runner.submit(request, lambda: self._prepare_local(episode, deadline),
                               deadline_at_s=deadline, adapter_selection=self.local_selection)
            self._log("RECOVERY_QUEUED", episode)
            return
        result = self.runner.poll(episode.request_id)
        if result is None:
            return
        if result.exception is not None:
            if isinstance(result.exception, LocalRepairError):
                raise result.exception
            raise LocalRepairError("MODEL_REQUEST_REJECTED", type(result.exception).__name__)
        if result.stale:
            raise LocalRepairError("MODEL_REQUEST_REJECTED", result.reason or "STALE_RESULT")
        candidate = result.value
        if (not isinstance(candidate, LocalRepairCandidate) or episode.context is None
                or candidate.context_digest != episode.context.digest):
            raise LocalRepairError("ROUTING_MISMATCH", "candidate does not belong to the admitted request")
        episode.phase = "LOCAL_VALIDATING"
        self._log("RECOVERY_MODEL_COMPLETED", episode)
        # Agent repeats its Safety preflight, then invokes this final guard,
        # then performs its own event/version/hold comparison before publish.
        # A REPROJECTABLE reference reconnects the live position onto the
        # admitted WORLD route by trusted code only; no new model request.
        def guard():
            self._check_local_live(episode)
            route = self._admitted_route(episode, candidate.world_route)
            self._check_space(record.assignment.uav_id, route, episode.context)
            self._check_local_live(episode)
            return tuple(route)
        route = guard()
        self._log("RECOVERY_VALIDATION_COMPLETED", episode)
        with self.runtime._recovery_commit_lock:
            route = guard()
            try:
                agent.commit_local_repair(candidate.task_plan, expected_event_id=episode.event_id,
                    expected_plan_version=episode.base_version, compiled_mission=candidate.compiled_mission,
                    final_guard=guard, allow_goto_detour_prefix=True)
            finally:
                # A physical Skill start can fail after Manager publication.
                # Agent adopts that version in its own finally block; Fleet
                # must publish matching metadata even on that exception path.
                published = agent.local_repair_snapshot.task_plan
                if published is not None and published.to_dict() == candidate.task_plan.to_dict():
                    self.runtime._planned_routes[record.assignment.uav_id] = route
                    self.runtime._route_progress[record.assignment.uav_id] = 0
                    from fleet.runtime import AssignmentStatus
                    self.runtime.assignments.update(episode.assignment_id, AssignmentStatus.RUNNING,
                        local_plan_version=candidate.task_plan.plan_version, last_error=None)
                    self.compilations[record.assignment.uav_id] = candidate.compiled_mission
        self._log("RECOVERY_COMMITTED", episode, new_local_version=candidate.task_plan.plan_version)
        self._finish(episode, "LOCAL_REPAIR_SUCCEEDED")

    def _prepare_local(self, episode, deadline):
        if self.episodes.get(episode.assignment_id) is not episode or self._signature(episode) != episode.queue_signature:
            return None
        record, agent, state = self._state(episode)
        if not state.stable_hold or self.runtime.cancel_requested:
            return None
        geometry = self.geometry_provider()
        observation = state.latest_observation
        if observation is None:
            raise LocalRepairError("OBSERVATION_UNAVAILABLE", "no aligned safe waiting observation")
        stamp = float(observation.timestamp)
        raw_pose_stamp = getattr(observation, "pose_timestamp_s", None)
        pose_stamp = stamp if raw_pose_stamp is None else float(raw_pose_stamp)
        if not isfinite(stamp) or not isfinite(pose_stamp):
            raise LocalRepairError("OBSERVATION_TIME_MISMATCH", "observation timestamps must be finite")
        if abs(stamp - pose_stamp) > self.config.max_pose_time_error_s:
            raise LocalRepairError("OBSERVATION_TIME_MISMATCH", "image/telemetry and pose do not align")
        uav = record.assignment.uav_id
        world = self.world_contexts[uav]
        resolver = self._initial_resolver(uav)
        pose = observation.uav_pose
        anchor = RepairAnchor(generate_routing_id("anchor"), getattr(observation, "frame_id", None) or
            generate_routing_id("observation"), stamp, pose_stamp,
            getattr(observation, "time_domain", "simulation"), FramePose((pose.x, pose.y, pose.z), pose.yaw),
            resolver._home_pose, resolver._uav_start_pose, int(geometry["map_version"]),
            int(geometry["reference_version"]), resolver.named_locations, self.config.max_pose_time_error_s)
        deps, versions = self._dependencies(episode.assignment_id)
        event = state.event
        completed = tuple(step.step_id for step in state.task_plan.steps[:state.current_step_index])
        context = LocalRepairContextV3(
            fleet_mission_id=self.runtime.fleet_plan.fleet_mission_id, assignment_id=episode.assignment_id,
            request_id=episode.request_id, episode_id=episode.episode_id, execution_generation=state.execution_epoch,
            original=state.compiled_mission, current_step_id=event.step_id, completed_step_ids=completed,
            completed_step_outputs=state.completed_outputs,
            goals=tuple(self._task_spec_for(episode.assignment_id).goal(goal)
                        for goal in self.assignments[episode.assignment_id].goal_ids),
            evidence=self._evidence(state), anchor=anchor, submitted_wall_s=self.clock(), deadline_wall_s=deadline,
            external_dependencies=deps, dependency_versions=versions, home_name=self.home_names[uav],
            max_suffix_steps=min(self.config.max_suffix_steps, 10),
            ordering_constraints=self.task_spec.ordering_constraints)
        check = check_cross_uav_dependencies(deps, deps, versions, versions)
        if check.verdict is DependencyVerdict.INVALID:
            raise LocalRepairError("DEPENDENCY_INVALID", ";".join(check.reasons), affected_uav_ids=check.affected_uav_ids)
        if not check.allowed:
            raise LocalRepairError("COORDINATION_REQUIRED", ";".join(check.reasons), affected_uav_ids=check.affected_uav_ids)
        contract = context.remaining_task_contract
        if not contract.supported:
            raise LocalRepairError("EVIDENCE_INSUFFICIENT", ";".join(contract.assessment.reasons))
        episode.context = context
        episode.submitted_wall_s = self.clock()
        episode.phase = "LOCAL_INFLIGHT"
        self._log("RECOVERY_MODEL_SUBMITTED", episode, **self.local_selection.to_dict())
        client = self.client_factory.for_role(ModelCallRole.RUNTIME_REPLAN,
            fleet_mission_id=context.fleet_mission_id, assignment_id=context.assignment_id, uav_id=uav)
        from models.runtime_deadline import DeadlineModelClient
        client = DeadlineModelClient(client, deadline_wall_s=context.deadline_wall_s, clock=self.clock)
        limits, policy, clock = self.limits, self.policy, self.clock
        # All values used by this closure are frozen request data or isolated
        # model/compiler instances, never a controller or live Fleet object.
        def compute():
            draft = _generate_suffix(client, context)
            return validate_local_repair(draft, context, world, dependency_versions=versions,
                external_dependencies=deps, now_wall_s=clock(), plan_validator=PlanValidator(limits, policy))
        return compute

    def _initial_resolver(self, uav):
        from planner.spatial_resolver import SpatialResolver
        world = self.world_contexts[uav]
        home = world.landing_zones[self.home_names[uav]]
        home_pose = FramePose((*home.position_xy_m, home.ground_altitude_m))
        start = FramePose(tuple(world.initial_uav_xyz_m))
        named = {item.name: item.position_xyz_m for item in world.navigation_points.values()}
        named[self.home_names[uav]] = home_pose.xyz_m
        return SpatialResolver(home_pose=home_pose, uav_start_pose=start, named_locations=named)

    def _reference_validity(self, episode):
        """Code-only spatial-reference verdict for the episode's frozen anchor."""
        context = episode.context
        state = self._state(episode)[2]
        observation = state.latest_observation
        if observation is None:
            return SpatialReferenceValidity(SpatialReferenceVerdict.INVALID,
                                            ("OBSERVATION_UNAVAILABLE",), "OBSERVATION_TIME_MISMATCH")
        pose_stamp = getattr(observation, "pose_timestamp_s", None)
        pose_stamp = float(observation.timestamp) if pose_stamp is None else float(pose_stamp)
        pose = observation.uav_pose
        geometry = self.geometry_provider()
        return evaluate_spatial_reference(
            context.anchor, observation_time_s=float(observation.timestamp), pose_time_s=pose_stamp,
            time_domain=getattr(observation, "time_domain", "simulation"),
            current_pose_xyz_m=(pose.x, pose.y, pose.z),
            now_wall_s=self.clock(), submitted_wall_s=context.submitted_wall_s,
            max_anchor_age_s=self.config.max_anchor_age_s,
            max_pose_time_error_s=self.config.max_pose_time_error_s,
            max_hold_drift_m=self.config.max_hold_drift_m,
            valid_pose_tolerance_m=self.config.valid_pose_tolerance_m,
            map_version=int(geometry["map_version"]), reference_version=int(geometry["reference_version"]))

    def _admitted_route(self, episode, admitted_route):
        """VALID keeps the admitted route; REPROJECTABLE reconnects by trusted code."""
        validity = self._reference_validity(episode)
        if validity.verdict is SpatialReferenceVerdict.INVALID:
            raise LocalRepairError(validity.code or "REFERENCE_CHANGED", ";".join(validity.reasons))
        if validity.verdict is SpatialReferenceVerdict.VALID:
            return tuple(admitted_route)
        pose = self._state(episode)[2].latest_observation.uav_pose
        return reproject_world_route(admitted_route, (pose.x, pose.y, pose.z))

    def _check_local_live(self, episode):
        context = episode.context
        if context is None or self.episodes.get(episode.assignment_id) is not episode:
            raise LocalRepairError("STALE_EPISODE", "episode no longer owns this result")
        if self.runtime.cancel_requested or self.clock() >= min(episode.deadline_wall_s, context.deadline_wall_s):
            raise LocalRepairError("DEADLINE_OR_CANCEL", "candidate lost execution eligibility")
        record, agent, state = self._state(episode)
        if state.event is None or state.event.event_id != episode.event_id:
            raise LocalRepairError("STALE_EVENT", "safe wait no longer belongs to this fault")
        if not state.stable_hold:
            raise LocalRepairError("HOLD_UNSTABLE", "safe waiting state lost stability")
        if state.execution_epoch != context.execution_generation or record.local_plan_version != episode.base_version:
            raise LocalRepairError("STALE_VERSION", "execution generation or local version changed")
        if state.current_step_id != context.current_step_id or state.task_plan.plan_version != episode.base_version:
            raise LocalRepairError("STALE_STEP", "authorized step has changed")
        deps, versions = self._dependencies(episode.assignment_id)
        decision = check_cross_uav_dependencies(context.external_dependencies, deps, context.dependency_versions, versions)
        if decision.verdict is DependencyVerdict.INVALID:
            raise LocalRepairError("DEPENDENCY_INVALID", ";".join(decision.reasons), affected_uav_ids=decision.affected_uav_ids)
        if not decision.allowed:
            raise LocalRepairError("COORDINATION_REQUIRED", ";".join(decision.reasons), affected_uav_ids=decision.affected_uav_ids)
        validity = self._reference_validity(episode)
        if validity.verdict is SpatialReferenceVerdict.INVALID:
            raise LocalRepairError(validity.code or "REFERENCE_CHANGED", ";".join(validity.reasons))

    def _check_space(self, uav, route, context=None):
        geometry = self.geometry_provider()
        snapshot = coerce_fleet_pose_snapshot(geometry["fleet_pose_snapshot"])
        if uav not in snapshot.poses:
            raise LocalRepairError("POSE_UNAVAILABLE", "missing trusted current world pose")
        pose = snapshot.poses[uav]
        if context is not None and (geometry["map_version"] != context.anchor.map_version or
                                    geometry["reference_version"] != context.anchor.reference_version):
            raise LocalRepairError("REFERENCE_CHANGED", "map or coordinate reference changed")
        points = (pose.position_xyz_m,) + tuple(route[1:] if context is not None else route)
        # Every current collidable obstacle, including newly introduced IDs.
        for obstacle in geometry["obstacles"]:
            if not obstacle.collidable:
                continue
            aabb = obstacle.aabb.expanded(geometry.get("uav_half_extent_xyz_m", (0.25, 0.25, 0.25)))
            if any(aabb.segment_intersection_fraction(a, b) is not None for a,b in zip(points, points[1:])):
                raise LocalRepairError("UNSAFE_ENTRY_OR_ROUTE", "current route crosses an obstacle")
        poses = dict(snapshot.poses)
        for other, other_pose in poses.items():
            if other == uav:
                poses[other] = replace(other_pose, route_xyz_m=points if len(points)>1 else ())
            else:
                current = self.runtime._planned_routes.get(other, ())
                index = self.runtime._route_progress.get(other, 0)
                remainder = (other_pose.position_xyz_m,) + tuple(current[index+1:])
                poses[other] = replace(other_pose, route_xyz_m=remainder if len(remainder)>1 else ())
        checker = FleetAirspaceManager(self.runtime.fleet_plan.coordination_policy.minimum_uav_separation_m)
        decision = checker.evaluate(FleetPoseSnapshot(snapshot.timestamp_s, poses))
        involved = sorted({tuple(sorted((p.uav_a_id, p.uav_b_id))) for p in decision.conflicts
                           if p.is_conflict and uav in {p.uav_a_id, p.uav_b_id}})
        if involved:
            # Route contention is a cross-UAV dependency outcome, not a private
            # spatial failure; the loser must coordinate, never both commit.
            check = check_cross_uav_dependencies((), (), {}, {}, route_conflicts=involved)
            raise LocalRepairError("SHARED_SPACE_CONFLICT", ";".join(check.reasons), affected_uav_ids=check.affected_uav_ids)

    # ------------------------------------------------------------------
    # Bounded related-group joint repair (COORDINATION_REQUIRED follow-up)
    # ------------------------------------------------------------------

    def _record_for_uav(self, uav_id):
        for row in self.runtime.assignments.records:
            if row.assignment.uav_id == uav_id:
                return row
        return None

    def _require_joint_peer(self, uav_id, episode):
        """Trusted suitability gate for a healthy UAV joining a repair scope."""
        row = self._record_for_uav(uav_id)
        if row is None or row.status.value != "RUNNING":
            raise LocalRepairError("JOINT_PEER_NOT_RUNNING", "peer is not executing a plan")
        if row.assignment.assignment_id in self.episodes:
            raise LocalRepairError("JOINT_PEER_BUSY", "peer already owns a recovery episode")
        agent = self.runtime.agents.get(uav_id)
        state = getattr(agent, "local_repair_snapshot", None)
        if agent is None or state is None or state.event is not None:
            raise LocalRepairError("JOINT_PEER_NOT_HEALTHY", "peer is itself under repair")
        if state.compiled_mission is None or not isinstance(state.compiled_mission.planner_output, SkillPlanDraftV3):
            raise LocalRepairError("JOINT_PEER_NO_V3_PLAN", "peer lacks a linear V3 plan")
        if state.task_plan is None or state.current_step_id is None:
            raise LocalRepairError("JOINT_PEER_NO_V3_PLAN", "peer lacks an active step boundary")
        current = next((step for step in state.compiled_mission.planner_output.steps
                        if step.id == state.current_step_id), None)
        if current is None or current.skill != "GOTO":
            raise LocalRepairError("JOINT_PEER_NOT_AT_TRANSIT", "only a transit GOTO boundary may be coordinated")
        if state.task_plan.plan_version != row.local_plan_version:
            raise LocalRepairError("JOINT_PEER_VERSION_MISMATCH", "peer plan version disagrees with records")
        return row

    def _try_enter_joint_repair(self, episode, reason, affected):
        """Select a bounded repair scope from trusted conflict evidence only.

        The scope is {faulted UAV} plus the conflict-affected set: a sound
        over-approximation, not a proven minimal set. Anything above the
        configured bound escalates/exits instead of becoming a fleet replan.
        """
        if not self.config.joint_repair_enabled or episode.phase.startswith(("REASSIGN", "JOINT")):
            return False
        scope = {episode.source_uav_id, *affected}
        scope.discard(None)
        if not 2 <= len(scope) <= self.config.max_joint_repair_scope_uavs:
            self._log("RECOVERY_JOINT_SCOPE_REJECTED", episode, reason=reason,
                      scope_size=len(scope), max_scope=self.config.max_joint_repair_scope_uavs)
            return False
        for uav_id in sorted(scope):
            if uav_id == episode.source_uav_id:
                continue
            try:
                self._require_joint_peer(uav_id, episode)
            except LocalRepairError as exc:
                self._log("RECOVERY_JOINT_PEER_REJECTED", episode, uav_id=uav_id, reason=exc.code)
                return False
        if episode.request_id:
            self.runner.cancel(episode.request_id, reason=reason)
            self.runner.poll(episode.request_id)
        episode.request_id = None
        episode.joint_scope = tuple(sorted(scope))
        episode.joint_contexts = None
        episode.phase = "JOINT_QUEUED"
        episode.next_attempt_wall_s = self.clock()
        self._log("RECOVERY_JOINT_SCOPE_SELECTED", episode, reason=reason,
                  joint_scope=list(episode.joint_scope))
        return True

    def _joint_peer_context(self, row, episode, deadline):
        """Frozen per-peer trusted context for the joint model request."""
        agent = self.runtime.agents[row.assignment.uav_id]
        state = agent.local_repair_snapshot
        observation = state.latest_observation
        if observation is None:
            raise LocalRepairError("OBSERVATION_UNAVAILABLE", "peer has no aligned observation")
        stamp = float(observation.timestamp)
        raw_pose_stamp = getattr(observation, "pose_timestamp_s", None)
        pose_stamp = stamp if raw_pose_stamp is None else float(raw_pose_stamp)
        if not isfinite(stamp) or not isfinite(pose_stamp) or abs(stamp - pose_stamp) > self.config.max_pose_time_error_s:
            raise LocalRepairError("OBSERVATION_TIME_MISMATCH", "peer observation and pose do not align")
        geometry = self.geometry_provider()
        resolver = self._initial_resolver(row.assignment.uav_id)
        pose = observation.uav_pose
        anchor = RepairAnchor(generate_routing_id("anchor"), getattr(observation, "frame_id", None) or
            generate_routing_id("observation"), stamp, pose_stamp,
            getattr(observation, "time_domain", "simulation"), FramePose((pose.x, pose.y, pose.z), pose.yaw),
            resolver._home_pose, resolver._uav_start_pose, int(geometry["map_version"]),
            int(geometry["reference_version"]), resolver.named_locations, self.config.max_pose_time_error_s)
        deps, versions = self._dependencies(row.assignment.assignment_id)
        completed = tuple(step.step_id for step in state.task_plan.steps[:state.current_step_index])
        context = LocalRepairContextV3(
            fleet_mission_id=self.runtime.fleet_plan.fleet_mission_id,
            assignment_id=row.assignment.assignment_id,
            request_id=episode.request_id, episode_id=episode.episode_id,
            execution_generation=state.execution_epoch,
            original=state.compiled_mission, current_step_id=state.current_step_id,
            completed_step_ids=completed, completed_step_outputs=state.completed_outputs,
            goals=tuple(self._task_spec_for(row.assignment.assignment_id).goal(goal)
                        for goal in row.assignment.goal_ids),
            evidence=self._evidence(state), anchor=anchor,
            submitted_wall_s=self.clock(), deadline_wall_s=deadline,
            external_dependencies=deps, dependency_versions=versions,
            home_name=self.home_names[row.assignment.uav_id],
            max_suffix_steps=min(self.config.max_suffix_steps, 10),
            ordering_constraints=self.task_spec.ordering_constraints)
        return context, deps, versions

    def _prepare_joint(self, episode, deadline):
        if self.episodes.get(episode.assignment_id) is not episode or episode.phase != "JOINT_QUEUED":
            return None
        record, agent, state = self._state(episode)
        if not state.stable_hold or self.runtime.cancel_requested:
            return None
        if episode.context is None:
            raise LocalRepairError("STALE_EPISODE", "joint repair needs the admitted single-repair context")
        # The faulted UAV must still pass every live single-repair check.
        self._check_local_live(episode)
        contexts = {episode.source_uav_id: episode.context}
        deps_map = {episode.source_uav_id: (episode.context.external_dependencies,
                                            episode.context.dependency_versions)}
        for uav_id in episode.joint_scope:
            if uav_id == episode.source_uav_id:
                continue
            row = self._require_joint_peer(uav_id, episode)
            context, deps, versions = self._joint_peer_context(row, episode, deadline)
            check = check_cross_uav_dependencies(deps, deps, versions, versions)
            if check.verdict is DependencyVerdict.INVALID:
                raise LocalRepairError("DEPENDENCY_INVALID", ";".join(check.reasons), affected_uav_ids=check.affected_uav_ids)
            if not check.allowed:
                raise LocalRepairError("COORDINATION_REQUIRED", ";".join(check.reasons), affected_uav_ids=check.affected_uav_ids)
            if not context.remaining_task_contract.supported:
                raise LocalRepairError("EVIDENCE_INSUFFICIENT", "peer remaining goals lack trusted evidence")
            contexts[uav_id] = context
            deps_map[uav_id] = (deps, versions)
        geometry = self.geometry_provider()
        snapshot = coerce_fleet_pose_snapshot(geometry["fleet_pose_snapshot"])
        readonly_routes = {}
        for row in self.runtime.assignments.records:
            uav_id = row.assignment.uav_id
            if uav_id in contexts or uav_id not in snapshot.poses:
                continue
            current = self.runtime._planned_routes.get(uav_id, ())
            index = self.runtime._route_progress.get(uav_id, 0)
            remainder = (snapshot.poses[uav_id].position_xyz_m,) + tuple(current[index + 1:])
            if len(remainder) > 1:
                readonly_routes[uav_id] = remainder
        request = JointRepairRequest(request_id=episode.request_id, episode_id=episode.episode_id,
            faulted_uav_id=episode.source_uav_id, scope=tuple(contexts), contexts=contexts,
            readonly_uav_routes=readonly_routes)
        episode.joint_contexts = dict(contexts)
        episode.submitted_wall_s = self.clock()
        episode.phase = "JOINT_INFLIGHT"
        self._log("RECOVERY_JOINT_MODEL_SUBMITTED", episode, joint_scope=list(request.scope),
                  **self.local_selection.to_dict())
        from models.runtime_deadline import DeadlineModelClient
        client = DeadlineModelClient(self.client_factory.for_role(ModelCallRole.RUNTIME_REPLAN,
            fleet_mission_id=episode.context.fleet_mission_id, assignment_id=episode.context.assignment_id,
            uav_id=episode.source_uav_id), deadline_wall_s=deadline, clock=self.clock)
        worlds = {uav_id: self.world_contexts[uav_id] for uav_id in request.scope}
        versions_map = {uav_id: deps_map[uav_id][1] for uav_id in request.scope}
        deps_only = {uav_id: deps_map[uav_id][0] for uav_id in request.scope}
        limits, policy, clock = self.limits, self.policy, self.clock
        def compute():
            drafts = _generate_joint_suffix(client, request)
            return validate_joint_repair(drafts, request, worlds,
                dependency_versions=versions_map, external_dependencies=deps_only,
                now_wall_s=clock(), plan_validator=PlanValidator(limits, policy))
        return compute

    def _check_joint_space(self, routes, contexts):
        """Live obstacle/separation check for every scope candidate at once."""
        geometry = self.geometry_provider()
        snapshot = coerce_fleet_pose_snapshot(geometry["fleet_pose_snapshot"])
        scope = set(routes)
        full_points = {}
        for uav_id, route in routes.items():
            if uav_id not in snapshot.poses:
                raise LocalRepairError("POSE_UNAVAILABLE", "missing trusted current world pose")
            context = contexts.get(uav_id)
            if context is not None and (geometry["map_version"] != context.anchor.map_version or
                                        geometry["reference_version"] != context.anchor.reference_version):
                raise LocalRepairError("REFERENCE_CHANGED", "map or coordinate reference changed")
            points = (snapshot.poses[uav_id].position_xyz_m,) + tuple(route[1:])
            full_points[uav_id] = points
            for obstacle in geometry["obstacles"]:
                if not obstacle.collidable:
                    continue
                aabb = obstacle.aabb.expanded(geometry.get("uav_half_extent_xyz_m", (0.25, 0.25, 0.25)))
                if any(aabb.segment_intersection_fraction(a, b) is not None for a, b in zip(points, points[1:])):
                    raise LocalRepairError("UNSAFE_ENTRY_OR_ROUTE", "joint candidate crosses an obstacle")
        poses = dict(snapshot.poses)
        for other, other_pose in poses.items():
            if other in scope:
                points = full_points[other]
                poses[other] = replace(other_pose, route_xyz_m=points if len(points) > 1 else ())
            else:
                current = self.runtime._planned_routes.get(other, ())
                index = self.runtime._route_progress.get(other, 0)
                remainder = (other_pose.position_xyz_m,) + tuple(current[index + 1:])
                poses[other] = replace(other_pose, route_xyz_m=remainder if len(remainder) > 1 else ())
        checker = FleetAirspaceManager(self.runtime.fleet_plan.coordination_policy.minimum_uav_separation_m)
        decision = checker.evaluate(FleetPoseSnapshot(snapshot.timestamp_s, poses))
        involved = sorted({tuple(sorted((p.uav_a_id, p.uav_b_id))) for p in decision.conflicts
                           if p.is_conflict and scope & {p.uav_a_id, p.uav_b_id}})
        if involved:
            check = check_cross_uav_dependencies((), (), {}, {}, route_conflicts=involved)
            raise LocalRepairError("SHARED_SPACE_CONFLICT", ";".join(check.reasons), affected_uav_ids=check.affected_uav_ids)

    def _check_joint_live(self, episode, candidates, committed=()):
        """Full live guard for every scope UAV; returns the routes to publish.

        UAVs already published inside this coordinated commit are consistency
        checked against their candidate instead of their request-time state.
        """
        if self.runtime.cancel_requested or self.clock() >= episode.deadline_wall_s:
            raise LocalRepairError("DEADLINE_OR_CANCEL", "joint candidate lost execution eligibility")
        # Faulted UAV keeps every single-repair liveness requirement.
        self._check_local_live(episode)
        routes = {}
        for uav_id in episode.joint_scope:
            candidate = candidates[uav_id]
            if uav_id == episode.source_uav_id:
                routes[uav_id] = self._admitted_route(episode, candidate.world_route)
                continue
            row = self._record_for_uav(uav_id)
            context = episode.joint_contexts[uav_id]
            agent = self.runtime.agents[uav_id]
            state = agent.local_repair_snapshot
            if row is None or row.status.value != "RUNNING":
                raise LocalRepairError("STALE_VERSION", "peer left the running state")
            if uav_id in committed:
                # Already published by this coordinated commit: verify the
                # published plan is exactly the admitted candidate.
                if (state.task_plan is None
                        or state.task_plan.to_dict() != candidate.task_plan.to_dict()):
                    raise LocalRepairError("STALE_STEP", "published peer plan diverged from the candidate")
                routes[uav_id] = tuple(candidate.world_route)
                continue
            if row.local_plan_version != context.original.planner_output.plan_version:
                raise LocalRepairError("STALE_VERSION", "peer local plan version changed")
            if state.event is not None:
                raise LocalRepairError("STALE_EVENT", "peer entered its own repair")
            if (state.current_step_id != context.current_step_id
                    or state.task_plan is None or state.task_plan.plan_version != row.local_plan_version):
                raise LocalRepairError("STALE_STEP", "peer authorized step changed")
            observation = state.latest_observation
            if observation is None:
                raise LocalRepairError("OBSERVATION_TIME_MISMATCH", "peer observation disappeared")
            pose_stamp = getattr(observation, "pose_timestamp_s", None)
            pose_stamp = float(observation.timestamp) if pose_stamp is None else float(pose_stamp)
            pose = observation.uav_pose
            validity = evaluate_spatial_reference(
                context.anchor, observation_time_s=float(observation.timestamp), pose_time_s=pose_stamp,
                time_domain=getattr(observation, "time_domain", "simulation"),
                current_pose_xyz_m=(pose.x, pose.y, pose.z),
                now_wall_s=self.clock(), submitted_wall_s=context.submitted_wall_s,
                max_anchor_age_s=self.config.max_anchor_age_s,
                max_pose_time_error_s=self.config.max_pose_time_error_s,
                max_hold_drift_m=self.config.max_joint_peer_drift_m,
                valid_pose_tolerance_m=self.config.valid_pose_tolerance_m,
                map_version=int(self.geometry_provider()["map_version"]),
                reference_version=int(self.geometry_provider()["reference_version"]))
            if validity.verdict is SpatialReferenceVerdict.INVALID:
                raise LocalRepairError(validity.code or "REFERENCE_CHANGED", ";".join(validity.reasons))
            if validity.verdict is SpatialReferenceVerdict.VALID:
                routes[uav_id] = tuple(candidate.world_route)
            else:
                routes[uav_id] = reproject_world_route(candidate.world_route, (pose.x, pose.y, pose.z))
            deps, versions = self._dependencies(row.assignment.assignment_id)
            decision = check_cross_uav_dependencies(context.external_dependencies, deps,
                                                     context.dependency_versions, versions)
            if decision.verdict is DependencyVerdict.INVALID:
                raise LocalRepairError("DEPENDENCY_INVALID", ";".join(decision.reasons), affected_uav_ids=decision.affected_uav_ids)
            if not decision.allowed:
                raise LocalRepairError("COORDINATION_REQUIRED", ";".join(decision.reasons), affected_uav_ids=decision.affected_uav_ids)
        self._check_joint_space(routes, episode.joint_contexts)
        return routes

    def _tick_joint(self, episode):
        record, agent, state = self._state(episode)
        if not self._owns_execution(episode):
            self._finish(episode, "STALE_EXECUTION")
            return
        if episode.request_id is None:
            if not state.stable_hold or self.clock() < episode.next_attempt_wall_s:
                return
            episode.request_id = generate_routing_id("request")
            episode.queue_signature = self._signature(episode)
            request = self._broker_request(episode, ModelCallRole.RUNTIME_REPLAN, self.local_selection)
            deadline = min(episode.deadline_wall_s, self.clock() + self.config.request_timeout_s)
            self.runner.submit(request, lambda: self._prepare_joint(episode, deadline),
                               deadline_at_s=deadline, adapter_selection=self.local_selection)
            self._log("RECOVERY_JOINT_QUEUED", episode, joint_scope=list(episode.joint_scope))
            return
        result = self.runner.poll(episode.request_id)
        if result is None:
            return
        if result.exception is not None:
            if isinstance(result.exception, LocalRepairError):
                raise result.exception
            raise LocalRepairError("MODEL_REQUEST_REJECTED", type(result.exception).__name__)
        if result.stale:
            raise LocalRepairError("MODEL_REQUEST_REJECTED", result.reason or "STALE_RESULT")
        candidates = result.value
        if (not isinstance(candidates, dict) or episode.joint_contexts is None
                or set(candidates) != set(episode.joint_scope)
                or any(not isinstance(item, LocalRepairCandidate) for item in candidates.values())):
            raise LocalRepairError("ROUTING_MISMATCH", "joint candidate does not belong to the admitted request")
        episode.phase = "JOINT_VALIDATING"
        self._log("RECOVERY_MODEL_COMPLETED", episode)
        committed = set()
        def guard():
            return self._check_joint_live(episode, candidates, committed)
        # PREPARE phase: every fallible per-UAV check runs to completion for
        # the whole scope before any publication decision; nothing mutates.
        guard()
        prepared = self._prepare_joint_publications(episode, candidates)
        self._log("RECOVERY_VALIDATION_COMPLETED", episode)
        # FINAL GUARD + ATOMIC PUBLISH inside the single commit lock.
        with self.runtime._recovery_commit_lock:
            routes = guard()
            self._publish_joint(episode, candidates, prepared, routes, committed)
        self._log("RECOVERY_COMMITTED", episode, joint_scope=list(episode.joint_scope))
        self._finish(episode, "JOINT_REPAIR_SUCCEEDED")

    def _prepare_joint_publications(self, episode, candidates):
        """PREPARE: per-UAV plans, goals, prefixes, versions and contracts.

        Pure validation plus immutable object construction; no plan, version,
        route, assignment record or execution state changes for any UAV.
        """
        faulted = episode.source_uav_id
        prepared = {}
        # Faulted UAV first so its checks gate the peers' work; ordering is
        # safe because prepare never mutates anything.
        prepared[faulted] = self.runtime.agents[faulted].prepare_local_repair_suffix(
            candidates[faulted].task_plan,
            expected_event_id=episode.event_id,
            expected_plan_version=episode.base_version,
            compiled_mission=candidates[faulted].compiled_mission,
            allow_goto_detour_prefix=True)
        for uav_id in episode.joint_scope:
            if uav_id == faulted:
                continue
            prepared[uav_id] = self.runtime.agents[uav_id].prepare_coordinated_suffix(
                candidates[uav_id].task_plan,
                expected_plan_version=candidates[uav_id].task_plan.plan_version - 1,
                compiled_mission=candidates[uav_id].compiled_mission)
        return prepared

    def _verify_joint_candidate_binding(self, candidate, prepared):
        """Bind the current LocalRepairCandidate to the prepared publication.

        The candidate digest is recomputed from the live candidate TaskPlan;
        any drift between what was prepared and what is about to be committed
        is a ROUTING_MISMATCH, never a silent substitute.
        """
        from skills.manager import task_plan_digest
        digest = task_plan_digest(candidate.task_plan)
        if prepared.candidate_digest != digest or prepared.manager_prepared.plan_digest != digest:
            raise LocalRepairError("ROUTING_MISMATCH",
                                   "joint candidate changed after prepare",
                                   affected_uav_ids=(prepared.uav_id,))

    def _publish_joint(self, episode, candidates, prepared, routes, committed):
        """FINAL GUARD -> SOFTWARE COMMIT (whole scope) -> EXECUTION RELEASE.

        Inside the commit lock: the joint final guard runs once for the whole
        scope, then every prepared binding (agent, manager and recomputed
        candidate digest) is verified for every UAV BEFORE any state changes.
        Only then is the scope's software state committed in one pass --
        Manager TaskPlan, Agent plan state/versions, assignment records,
        routes, progress and compilations -- and only after all of it is
        consistent is each UAV's execution released. Software atomicity never
        means physical actions are reversible.
        """
        faulted = episode.source_uav_id
        order = tuple(uav_id for uav_id in episode.joint_scope if uav_id != faulted) + (faulted,)

        def joint_guard():
            return self._check_joint_live(episode, candidates, committed)

        # FINAL GUARD for the whole scope, then binding verification for the
        # whole scope: any failure here leaves every UAV untouched.
        joint_guard()
        for uav_id in order:
            self._verify_joint_candidate_binding(candidates[uav_id], prepared[uav_id])
            self.runtime.agents[uav_id].verify_prepared_binding(prepared[uav_id])

        # SOFTWARE COMMIT for the whole scope. Bindings for every UAV were
        # verified immediately above; a failure here is an internal error and
        # only the exceptional fallback below restores consistency.
        applied = []
        try:
            for uav_id in order:
                self.runtime.agents[uav_id].apply_prepared_suffix_state(prepared[uav_id])
                applied.append(uav_id)
                committed.add(uav_id)
        except Exception as exc:
            self._log("RECOVERY_JOINT_SOFTWARE_COMMIT_FAILED", episode,
                      failed_uav_id=None if not applied else applied[-1],
                      error_code=type(exc).__name__)
            self._rollback_joint_peers(episode, candidates, applied)
            raise
        for uav_id in order:
            row = self._record_for_uav(uav_id)
            self.runtime._planned_routes[uav_id] = routes[uav_id]
            self.runtime._route_progress[uav_id] = 0
            from fleet.runtime import AssignmentStatus
            self.runtime.assignments.update(row.assignment.assignment_id, AssignmentStatus.RUNNING,
                local_plan_version=candidates[uav_id].task_plan.plan_version, last_error=None)
            self.compilations[uav_id] = candidates[uav_id].compiled_mission

        # EXECUTION RELEASE only after every UAV's software state committed.
        # A release/Skill-start failure is a trusted execution failure (the
        # Manager's skill_start_failed path fails safe to LAND internally);
        # the committed software plan is never rolled back for it.
        for uav_id in order:
            try:
                self.runtime.agents[uav_id].release_prepared_execution(prepared[uav_id])
            except Exception as exc:
                self._log("RECOVERY_JOINT_RELEASE_FAILED", episode, uav_id=uav_id,
                          error_code=type(exc).__name__)

    def _rollback_joint_peers(self, episode, candidates, applied):
        """Exceptional fallback only; never the all-or-none mechanism.

        Unreachable on every normal path: PREPARE, the final guard and the
        whole-scope binding verification front-run every fallible step, so a
        SOFTWARE COMMIT failure means an internal invariant broke. The
        residual window is a truly unexpected exception between two
        in-memory state swaps (no physical action is rolled back); the
        already-committed peers are restored to their original suffix at the
        next version with records/compilations re-aligned so Agent, Manager,
        assignment record, route and compilation stay version-consistent.
        """
        if not applied:
            return
        self._log("RECOVERY_JOINT_ROLLBACK", episode, uav_ids=list(applied))
        from skills.plan import TaskPlan
        from fleet.runtime import AssignmentStatus
        for uav_id in reversed(applied):
            if uav_id == episode.source_uav_id:
                continue
            agent = self.runtime.agents[uav_id]
            row = self._record_for_uav(uav_id)
            try:
                current = agent.local_repair_snapshot.task_plan
                original = episode.joint_contexts[uav_id].original.task_plan
                restore = TaskPlan(original.steps, original.mission_id, original.uav_id,
                                   current.plan_version + 1)
                agent.commit_coordinated_suffix(restore,
                    expected_plan_version=current.plan_version, final_guard=None)
                # Re-align every layer with the restored (version-bumped) plan.
                record_version = agent.local_repair_snapshot.task_plan.plan_version
                self.runtime.assignments.update(row.assignment.assignment_id,
                    AssignmentStatus.RUNNING, local_plan_version=record_version, last_error=None)
                self.compilations[uav_id] = agent.local_repair_snapshot.compiled_mission
            except Exception as exc:
                self._log("RECOVERY_JOINT_ROLLBACK_FAILED", episode, uav_id=uav_id,
                          error_code=type(exc).__name__)

    def _broker_request(self, episode, role, selection):
        uav = self._record(episode).assignment.uav_id
        return ModelBrokerRequest(call_role=role.value,
            priority=ModelRequestPriority.P1_FLEET_REPLAN if role is ModelCallRole.FLEET_REPLAN else ModelRequestPriority.P2_AGENT_RUNTIME_REPLAN,
            uav_id=uav, assignment_id=episode.assignment_id, request_id=episode.request_id,
            submitted_at_s=self.clock(), requested_adapter=selection.requested_adapter,
            payload={"episode_id":episode.episode_id}, replaceable=False)

    def _tick_reassignment(self, episode):
        record = self._record(episode)
        if record.status.value != "WAITING_REASSIGNMENT":
            self._finish(episode, "STALE_REASSIGNMENT")
            return
        if episode.request_id is None:
            if self.clock() < episode.next_attempt_wall_s:
                return
            if not self.supports_reassignment or episode.reassign_attempts >= self.config.max_reassign_attempts:
                self._exit(episode, "NO_REASSIGNMENT_BUDGET")
                return
            episode.reassign_attempts += 1
            episode.request_id = generate_routing_id("request")
            episode.queue_signature = (record.local_plan_version, self.runtime.fleet_plan.fleet_plan_version)
            deadline = min(episode.deadline_wall_s, self.clock()+self.config.request_timeout_s)
            self.runner.submit(self._broker_request(episode,ModelCallRole.FLEET_REPLAN,self.fleet_selection),
                lambda: self._prepare_reassignment(episode,deadline), deadline_at_s=deadline,
                adapter_selection=self.fleet_selection)
            self._log("RECOVERY_QUEUED", episode)
            return
        result=self.runner.poll(episode.request_id)
        if result is None:
            return
        if result.exception is not None:
            raise result.exception
        if result.stale:
            self._exit(episode, result.reason or "STALE_RESULT")
            return
        candidate=result.value
        self._log("RECOVERY_MODEL_COMPLETED",episode)
        self._check_reassignment_live(episode)
        publication = None
        published = False
        try:
            publication = self.replan_boundary.prepare_candidate(candidate)
            if len(publication.replacements) != 1:
                raise LocalRepairError("REASSIGNMENT_SHAPE_UNSUPPORTED", "one failed assignment requires one replacement")
            item = publication.replacements[0]
            # Resolve all required metadata before any execution publication.
            assignment_v2 = candidate["assignment_v2"]
            world_context = candidate["world_context"]
            compilation = candidate["compilation"]
            task_spec = candidate["request_v2"].task_spec
            if assignment_v2.assignment_id != item.replacement_assignment.assignment_id:
                raise LocalRepairError("ROUTING_MISMATCH", "replacement metadata differs from prepared publication")
            def guard():
                self._check_reassignment_live(episode)
                self._check_space(item.replacement_assignment.uav_id, item.planned_route)
                self._check_reassignment_live(episode)
            guard()
            self._log("RECOVERY_VALIDATION_COMPLETED", episode)
            with self.runtime._recovery_commit_lock:
                guard()
                self.runtime._publish_fleet_replan(episode.assignment_id, publication, final_guard=guard)
                published = True
                self.assignments.pop(episode.assignment_id)
                self.assignments[assignment_v2.assignment_id] = assignment_v2
                self.assignment_task_specs.pop(episode.assignment_id, None)
                self.assignment_task_specs[assignment_v2.assignment_id] = task_spec
                self.confirmed_goal_evidence.update(dict(episode.confirmed_goal_evidence))
                self.world_contexts[item.replacement_assignment.uav_id] = world_context
                self.compilations[item.replacement_assignment.uav_id] = compilation
            # This is a post-publication bookkeeping notification. Failure may
            # be logged, but cannot discard a now-live Agent/perception or roll
            # the source aircraft back to an obsolete request's state.
            try:
                self.replan_boundary.committed(candidate, publication)
            except Exception as exc:
                self._log("RECOVERY_POST_COMMIT_ERROR", episode, error_code=type(exc).__name__)
            self._log("RECOVERY_COMMITTED", episode, new_fleet_version=publication.new_fleet_plan_version)
            self._finish(episode, "REASSIGNMENT_SUCCEEDED")
        except Exception:
            if publication is not None:
                published = published or any(
                    self.runtime.agents.get(item.replacement_assignment.uav_id) is item.agent
                    for item in publication.replacements
                )
                if not published:
                    for item in publication.replacements:
                        cleanup = getattr(item, "discard", None)
                        if callable(cleanup):
                            cleanup()
            if published:
                self._finish(episode, "REASSIGNMENT_PUBLISHED")
            raise

    def _prepare_reassignment(self,episode,deadline):
        record=self._record(episode)
        if not self._owns_execution(episode) or self.runtime.cancel_requested:
            return None
        if (record.local_plan_version,self.runtime.fleet_plan.fleet_plan_version)!=episode.queue_signature:
            raise LocalRepairError("STALE_VERSION", "Fleet changed before model admission")
        # Failed source.goal_ids are never silently treated as remaining goals.
        record,agent,state=self._state(episode)
        if state.compiled_mission is None:
            raise LocalRepairError("EVIDENCE_INSUFFICIENT","source lacks a trusted executable plan")
        goals=tuple(self._task_spec_for(episode.assignment_id).goal(g)
                    for g in self.assignments[episode.assignment_id].goal_ids)
        completed=tuple(step.step_id for step in state.task_plan.steps[:state.current_step_index])
        # Handoff consumes the SAME remaining-task computation as local repair;
        # it never re-derives completion amounts or goal identity itself.
        contract=build_remaining_task_contract(goals,state.compiled_mission.planner_output,completed,self._evidence(state),
            current_step_id=state.current_step_id,current_step_started=True,
            home_name=self.home_names[record.assignment.uav_id],consumer="HANDOFF")
        if not contract.supported:
            raise LocalRepairError("HANDOFF_EVIDENCE_UNSUPPORTED", "remaining execution evidence is insufficient")
        if not contract.pending_goal_ids:
            raise LocalRepairError("HANDOFF_EVIDENCE_UNSUPPORTED", "no independently executable remaining goal")
        # Handoff accepts only fully world-anchored TRANSFERABLE obligations.
        # There is no trusted shared-target evidence source yet, so both
        # SAME_UAV_ONLY and REQUIRES_SHARED_EVIDENCE must stop here, with or
        # without confirmed goals on the source.
        if any(entry.transferability is not Transferability.TRANSFERABLE for entry in contract.pending_entries):
            raise LocalRepairError("HANDOFF_EVIDENCE_UNSUPPORTED", "remaining goals require source-local outputs")
        if contract.confirmed_goal_ids:
            # Delegation anchors come from the contract registry, not from
            # goal_type branches in the controller.
            if not any(self._goal_contracts.is_delegation_anchor(goal.goal_type)
                       for goal in contract.pending_goals):
                raise LocalRepairError("HANDOFF_EVIDENCE_UNSUPPORTED", "source termination alone cannot be delegated")
        deps,versions=self._dependencies(episode.assignment_id, goal_ids=contract.pending_goal_ids)
        decision=check_cross_uav_dependencies(deps,deps,versions,versions)
        if decision.verdict is DependencyVerdict.INVALID:
            raise LocalRepairError("DEPENDENCY_INVALID",";".join(decision.reasons),affected_uav_ids=decision.affected_uav_ids)
        if not decision.allowed:
            raise LocalRepairError("COORDINATION_REQUIRED",";".join(decision.reasons),affected_uav_ids=decision.affected_uav_ids)
        states, refs = self._goal_state()
        episode.confirmed_goal_evidence = tuple(
            (goal_id, refs[goal_id]) for goal_id in contract.confirmed_goal_ids
            if states.get(goal_id) == "CONFIRMED" and refs.get(goal_id))
        if len(episode.confirmed_goal_evidence) != len(contract.confirmed_goal_ids):
            raise LocalRepairError("HANDOFF_EVIDENCE_UNSUPPORTED", "completed goals lack stable owner evidence")
        snapshot=self.replan_boundary.snapshot_request(record,self.runtime.world_belief,
                                                       remaining_goal_ids=contract.pending_goal_ids,
                                                       remaining_goals=contract.pending_goals,
                                                       source_compiled_mission=state.compiled_mission,
                                                       external_dependencies=deps,
                                                       recovery_request_id=episode.request_id,
                                                       deadline_wall_s=deadline, wall_clock=self.clock)
        episode.external_dependencies=deps
        episode.dependency_versions=versions
        episode.remaining_goal_ids=contract.pending_goal_ids
        episode.submitted_wall_s=self.clock()
        episode.candidate=deadline
        episode.phase="REASSIGN_INFLIGHT"
        self._log("RECOVERY_MODEL_SUBMITTED",episode,**self.fleet_selection.to_dict())
        compute=self.replan_boundary.compute_candidate
        return lambda:compute(snapshot)

    def _check_reassignment_live(self,episode):
        if not self._owns_execution(episode) or self.runtime.cancel_requested:
            raise LocalRepairError("STALE_EPISODE","request lost ownership")
        record=self._record(episode)
        if record.status.value!="WAITING_REASSIGNMENT" or (record.local_plan_version,self.runtime.fleet_plan.fleet_plan_version)!=episode.queue_signature:
            raise LocalRepairError("STALE_VERSION","Fleet changed while reassignment was computed")
        if self.clock()>=min(episode.deadline_wall_s,float(episode.candidate)):
            raise LocalRepairError("DEADLINE_EXPIRED","reassignment validation exceeded deadline")
        deps,versions=self._dependencies(episode.assignment_id, goal_ids=episode.remaining_goal_ids)
        decision=check_cross_uav_dependencies(episode.external_dependencies,deps,episode.dependency_versions,versions)
        if decision.verdict is DependencyVerdict.INVALID:
            raise LocalRepairError("DEPENDENCY_INVALID","dependency evidence is structurally invalid",affected_uav_ids=decision.affected_uav_ids)
        if not decision.allowed:
            raise LocalRepairError("COORDINATION_REQUIRED","external dependency changed",affected_uav_ids=decision.affected_uav_ids)

    def _handle_failure(self,episode,reason,*,affected=()):
        if self.episodes.get(episode.assignment_id) is not episode:
            return
        if not self._owns_execution(episode):
            self._finish(episode, "STALE_EXECUTION")
            return
        self._last_error[episode.assignment_id]=reason
        self._log("RECOVERY_REJECTED",episode,reason=reason,affected_uav_ids=list(affected))
        if reason in {"COORDINATION_REQUIRED","SHARED_SPACE_CONFLICT","REFERENCE_CHANGED","EVIDENCE_INSUFFICIENT",
                      "HANDOFF_EVIDENCE_UNSUPPORTED","DEPENDENCY_INVALID"}:
            if (reason in {"COORDINATION_REQUIRED", "SHARED_SPACE_CONFLICT"}
                    and self._try_enter_joint_repair(episode, reason, affected)):
                return
            if affected:
                self._log("RECOVERY_COORDINATION_REQUIRED",episode,reason=reason,affected_uav_ids=list(affected))
            self._exit(episode,reason)
        elif episode.phase.startswith("JOINT"):
            # Joint repair is one bounded attempt: a rejected candidate exits
            # safely instead of falling back into unbounded local retries.
            self._exit(episode,reason)
        elif episode.phase.startswith("REASSIGN"):
            # A concurrent handoff may change only the global Fleet version.
            # Discard this candidate and re-snapshot under the SAME episode's
            # bounded budget; never publish it against refreshed metadata.
            if (reason == "STALE_VERSION"
                    and episode.reassign_attempts < self.config.max_reassign_attempts
                    and self.clock() < episode.deadline_wall_s):
                if episode.request_id:
                    self.runner.cancel(episode.request_id, reason=reason)
                episode.request_id = None
                episode.context = None
                episode.candidate = None
                episode.phase = "REASSIGN_QUEUED"
                episode.next_attempt_wall_s = self.clock() + self.config.retry_cooldown_s
                self._log("RECOVERY_REASSIGNMENT_RETRY", episode, reason=reason)
            else:
                self._exit(episode,reason)
        elif reason in {"STALE_EPISODE","STALE_EVENT","STALE_EXECUTION","STALE_VERSION","STALE_STEP"}:
            # A late request cannot act on or fall back a newer execution.
            self._finish(episode,reason)
        else:
            if episode.request_id is not None:
                self.runner.cancel(episode.request_id,reason=reason)
            episode.request_id=None
            episode.phase="WAIT_HOLD"
            episode.next_attempt_wall_s=self.clock()+self.config.retry_cooldown_s
            if episode.attempts>=self.config.max_local_attempts:
                self._escalate(episode,reason)

    def _escalate(self,episode,reason):
        if not self._owns_execution(episode):
            self._finish(episode, "STALE_EXECUTION")
            return
        if not self.supports_reassignment:
            self._exit(episode,reason)
            return
        if episode.request_id:
            self.runner.cancel(episode.request_id,reason=reason)
        episode.request_id=None
        episode.phase="REASSIGN_QUEUED"
        episode.next_attempt_wall_s = self.clock()
        self.runtime._mark_local_failure(episode.assignment_id,reason)
        self._log("RECOVERY_ESCALATED",episode,reason=reason)

    def _exit(self,episode,reason):
        if self.episodes.get(episode.assignment_id) is not episode:
            return
        if not self._owns_execution(episode):
            self._finish(episode, "STALE_EXECUTION")
            return
        self._log("RECOVERY_SAFE_EXIT",episode,reason=reason)
        self.runtime._begin_local_failsafe_landing(episode.assignment_id,reason)
        self._finish(episode,reason)

    def _finish(self,episode,reason):
        if self.episodes.get(episode.assignment_id) is not episode:
            return
        self.episodes.pop(episode.assignment_id)
        if episode.event_id:
            self.finished_events.add(episode.event_id)
        if episode.request_id:
            self.runner.cancel(episode.request_id,reason=reason)
            # Revocation has its own result; retire it once so abandoned old
            # episodes cannot accumulate unconsumed completion slots.
            self.runner.poll(episode.request_id)

    def cancel_all(self,reason="CANCELED"):
        self._assert_owner()
        for episode in tuple(self.episodes.values()):
            self._finish(episode,reason)

    def close(self):
        self._assert_owner()
        if self._closed:
            return
        self.cancel_all("CLOSED")
        self.runner.close(timeout_s=self.config.shutdown_timeout_s)
        self._closed=True


def _generate_suffix(client, context):
    output=context.original.planner_output
    suffix_schema = build_local_repair_json_schema(context)
    constants = {key: value["const"] for key, value in suffix_schema["properties"].items()
                 if "const" in value}
    payload={"trusted_repair_context":context.to_dict(),"authorized_output":constants,
             "original_suffix":[step.to_dict() for step in output.steps[len(context.completed_step_ids):]]}
    response=client.chat((
        ChatMessage("system","Return one authorized Spatial V3 suffix JSON only. Preserve retained step IDs, target references, durations, goal conditions and termination. Never replay completed steps or TAKEOFF. Do not rewrite another UAV or reinterpret the original user instruction. Relative geometry uses only the supplied immutable anchor. Routing, versions, permissions and deadlines are trusted constants. The remaining_task_contract in trusted_repair_context is trusted read-only evidence computed from execution proof: never dispute, recompute, reduce or restate its completion amounts, goal identity, target bindings or completion conditions."),
        ChatMessage("user",json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")))),
        options=GenerationOptions(temperature=0.0,max_tokens=4096,
            response_format=JsonSchemaResponseFormat("local_repair_v3",suffix_schema)))
    if not isinstance(response,ModelResponse) or len(response.content.encode("utf-8"))>65536:
        raise LocalRepairError("INVALID_MODEL_RESPONSE","invalid or oversized repair response")
    return LocalRepairDraftV3.from_dict(strict_json_object_loads(response.content))


def _generate_joint_suffix(client, request):
    """One Qwen request for the whole editable scope; readonly UAVs are geometry."""
    joint_schema = build_joint_repair_json_schema(request)
    payload={"trusted_joint_context":request.to_dict(),
             "authorized_output":{uav:{key:value["const"] for key,value in joint_schema["properties"][uav]["properties"].items()
                                    if "const" in value} for uav in request.scope}}
    response=client.chat((
        ChatMessage("system","Return one JSON object whose keys are exactly the editable UAV IDs in editable_uavs. For each UAV return only its authorized Spatial V3 suffix. You may modify ONLY those UAVs; every readonly UAV, its route and its resources are fixed constraints that must be avoided, never rewritten or released. Preserve each UAV's retained step IDs, target references, durations, goal conditions and termination. Each UAV's remaining_task_contract is trusted read-only evidence computed from execution proof: never dispute, recompute, reduce or restate completion amounts, goal identity, target bindings or completion conditions. Routing, versions, permissions and deadlines are trusted constants."),
        ChatMessage("user",json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")))),
        options=GenerationOptions(temperature=0.0,max_tokens=8192,
            response_format=JsonSchemaResponseFormat("joint_repair_v3",joint_schema)))
    if not isinstance(response,ModelResponse) or len(response.content.encode("utf-8"))>262144:
        raise LocalRepairError("INVALID_MODEL_RESPONSE","invalid or oversized joint repair response")
    return parse_joint_repair_drafts(strict_json_object_loads(response.content), request)
