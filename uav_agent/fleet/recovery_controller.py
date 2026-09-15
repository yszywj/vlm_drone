"""Owner-driven, opt-in Fleet recovery over the existing Broker and SkillManager.

Only immutable request values cross into model threads. Flight permission is
published by Fleet's single writer after a second live-state/deadline check.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from math import dist, isfinite
from threading import get_ident
from time import monotonic
from typing import Callable

from common.ids import generate_routing_id
from configs.schema import FleetRecoveryConfig
from fleet.airspace_manager import FleetAirspaceManager, FleetPoseSnapshot, coerce_fleet_pose_snapshot
from fleet.local_repair import (
    LocalRepairCandidate, LocalRepairContextV3, LocalRepairDraftV3, LocalRepairError, RepairAnchor,
    SkillExecutionEvidence, assess_remaining_goals, check_external_dependencies,
    build_local_repair_json_schema, extract_external_dependencies, validate_local_repair,
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
    source_uav_id: str | None = None


class FleetRecoveryController:
    """A bounded recovery owner, called only at Fleet tick's serial event boundary."""

    def __init__(self, *, config: FleetRecoveryConfig, broker, task_spec,
                 assignments, compilations, world_contexts, home_names, client_factory,
                 geometry_provider: Callable, replan_boundary=None, planner_limits=None,
                 planner_policy=None, clock=monotonic, runner=None):
        if not config.enabled:
            raise ValueError("FleetRecoveryController requires explicit enabled configuration")
        if not callable(geometry_provider) or not callable(clock):
            raise TypeError("recovery requires a trusted geometry provider and monotonic clock")
        self.config, self.clock = config, clock
        self.task_spec = task_spec
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
        refs = {}
        for assignment_id, assignment in self.assignments.items():
            try:
                record = self.runtime.assignments.by_id(assignment_id)
                agent = self.runtime.agents[record.assignment.uav_id]
                state = agent.local_repair_snapshot
                original = state.compiled_mission
                if original is None or not isinstance(original.planner_output, SkillPlanDraftV3):
                    continue
                goals = tuple(self.task_spec.goal(goal) for goal in assignment.goal_ids)
                completed = tuple(step.step_id for step in state.task_plan.steps[:state.current_step_index])
                assessment = assess_remaining_goals(goals, original.planner_output, completed,
                    self._evidence(state), current_step_id=state.current_step_id,
                    current_step_started=True, home_name=self.home_names[record.assignment.uav_id])
                for goal in assessment.pending_goals:
                    states[goal.goal_id] = "PENDING"
                    # Owner-produced assignment/version evidence, not a model
                    # completion claim. ACTIVE below removes this permission.
                    refs[goal.goal_id] = (f"pending_{assignment_id}_v{record.local_plan_version}",)
                for goal_id in assessment.confirmed_goal_ids:
                    states[goal_id] = "CONFIRMED"
                    refs[goal_id] = tuple(e.invocation_id for e in self._evidence(state)
                                          if e.step_id in completed)
                # Incomplete/active external successors must not be called pending.
                if state.current_step_id is not None:
                    from fleet.local_repair import _goal_steps
                    for goal in goals:
                        if any(step.id == state.current_step_id for step in _goal_steps(
                                goal, original.planner_output.steps, self.home_names[record.assignment.uav_id])):
                            if states[goal.goal_id] != "CONFIRMED":
                                states[goal.goal_id] = "ACTIVE"
                                refs.pop(goal.goal_id, None)
            except (AttributeError, KeyError, ValueError):
                # Missing evidence is UNKNOWN, never model-authorized completion.
                continue
        return states, refs

    def _dependencies(self, assignment_id):
        states, refs = self._goal_state()
        geometry = self.geometry_provider()
        versions = {"fleet": self.runtime.fleet_plan.fleet_plan_version,
                    assignment_id: self.runtime.assignments.by_id(assignment_id).local_plan_version}
        versions["reference"] = int(geometry["reference_version"])
        versions["map"] = int(geometry["map_version"])
        dependencies = extract_external_dependencies(
            self.task_spec, self.assignments[assignment_id].goal_ids,
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
        self._check_local_live(episode)
        self._check_space(record.assignment.uav_id, candidate.world_route, episode.context)
        self._log("RECOVERY_VALIDATION_COMPLETED", episode)
        # Agent repeats its Safety preflight, then invokes this final guard,
        # then performs its own event/version/hold comparison before publish.
        def guard():
            self._check_local_live(episode)
            self._check_space(record.assignment.uav_id, candidate.world_route, episode.context)
            self._check_local_live(episode)
        with self.runtime._recovery_commit_lock:
            guard()
            try:
                agent.commit_local_repair(candidate.task_plan, expected_event_id=episode.event_id,
                    expected_plan_version=episode.base_version, compiled_mission=candidate.compiled_mission,
                    final_guard=guard)
            finally:
                # A physical Skill start can fail after Manager publication.
                # Agent adopts that version in its own finally block; Fleet
                # must publish matching metadata even on that exception path.
                published = agent.local_repair_snapshot.task_plan
                if published is not None and published.to_dict() == candidate.task_plan.to_dict():
                    self.runtime._planned_routes[record.assignment.uav_id] = candidate.world_route
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
            goals=tuple(self.task_spec.goal(goal) for goal in self.assignments[episode.assignment_id].goal_ids),
            evidence=self._evidence(state), anchor=anchor, submitted_wall_s=self.clock(), deadline_wall_s=deadline,
            external_dependencies=deps, dependency_versions=versions, home_name=self.home_names[uav],
            max_suffix_steps=min(self.config.max_suffix_steps, 10),
            ordering_constraints=self.task_spec.ordering_constraints)
        check = check_external_dependencies(deps, deps, versions, versions)
        if not check.allowed:
            raise LocalRepairError("COORDINATION_REQUIRED", ";".join(check.reasons), affected_uav_ids=check.affected_uav_ids)
        if not context.assessment.supported:
            raise LocalRepairError("EVIDENCE_INSUFFICIENT", ";".join(context.assessment.reasons))
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

    def _check_local_live(self, episode):
        context = episode.context
        if context is None or self.episodes.get(episode.assignment_id) is not episode:
            raise LocalRepairError("STALE_EPISODE", "episode no longer owns this result")
        if self.runtime.cancel_requested or self.clock() >= min(episode.deadline_wall_s, context.deadline_wall_s):
            raise LocalRepairError("DEADLINE_OR_CANCEL", "candidate lost execution eligibility")
        if self.clock() - context.submitted_wall_s > self.config.max_anchor_age_s:
            raise LocalRepairError("ANCHOR_EXPIRED", "bound reference is too old")
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
        decision = check_external_dependencies(context.external_dependencies, deps, context.dependency_versions, versions)
        if not decision.allowed:
            raise LocalRepairError("COORDINATION_REQUIRED", ";".join(decision.reasons), affected_uav_ids=decision.affected_uav_ids)
        observation = state.latest_observation
        if observation is None or float(observation.timestamp) < context.anchor.observation_time_s:
            raise LocalRepairError("OBSERVATION_TIME_MISMATCH", "observation time moved backwards")
        pose_stamp = getattr(observation, "pose_timestamp_s", None)
        pose_stamp = float(observation.timestamp) if pose_stamp is None else float(pose_stamp)
        if (not isfinite(pose_stamp) or not isfinite(float(observation.timestamp))
                or abs(pose_stamp - float(observation.timestamp)) > self.config.max_pose_time_error_s):
            raise LocalRepairError("OBSERVATION_TIME_MISMATCH", "live pose and observation are not aligned")
        pose = observation.uav_pose
        if dist((pose.x, pose.y, pose.z), context.anchor.pose.xyz_m) > self.config.max_hold_drift_m:
            raise LocalRepairError("HOLD_DRIFT", "vehicle left the admitted safe waiting region")

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
        involved = [p for p in decision.conflicts if uav in {p.uav_a_id,p.uav_b_id} and p.is_conflict]
        if involved:
            affected = tuple(sorted({x for p in involved for x in (p.uav_a_id,p.uav_b_id)}))
            raise LocalRepairError("SHARED_SPACE_CONFLICT", "replacement conflicts with shared world space", affected_uav_ids=affected)

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
        if not self._owns_execution(episode) or self.runtime.cancel_requested or (record.local_plan_version,self.runtime.fleet_plan.fleet_plan_version)!=episode.queue_signature:
            return None
        deps,versions=self._dependencies(episode.assignment_id)
        decision=check_external_dependencies(deps,deps,versions,versions)
        if not decision.allowed:
            raise LocalRepairError("COORDINATION_REQUIRED",";".join(decision.reasons),affected_uav_ids=decision.affected_uav_ids)
        # Failed source.goal_ids are never silently treated as remaining goals.
        record,agent,state=self._state(episode)
        if state.compiled_mission is None:
            raise LocalRepairError("EVIDENCE_INSUFFICIENT","source lacks a trusted executable plan")
        goals=tuple(self.task_spec.goal(g) for g in self.assignments[episode.assignment_id].goal_ids)
        completed=tuple(step.step_id for step in state.task_plan.steps[:state.current_step_index])
        assessment=assess_remaining_goals(goals,state.compiled_mission.planner_output,completed,self._evidence(state),
            current_step_id=state.current_step_id,current_step_started=True,home_name=self.home_names[record.assignment.uav_id])
        if not assessment.supported or assessment.confirmed_goal_ids:
            # First release cannot safely transfer retained output references
            # or partial temporal obligations to a fresh aircraft.
            raise LocalRepairError("HANDOFF_EVIDENCE_UNSUPPORTED","partial/completed goal handoff requires coordination")
        snapshot=self.replan_boundary.snapshot_request(record,self.runtime.world_belief,
                                                       remaining_goal_ids=assessment.pending_goal_ids,
                                                       external_dependencies=deps,
                                                       deadline_wall_s=deadline, wall_clock=self.clock)
        episode.external_dependencies=deps
        episode.dependency_versions=versions
        episode.remaining_goal_ids=assessment.pending_goal_ids
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
        deps,versions=self._dependencies(episode.assignment_id)
        decision=check_external_dependencies(episode.external_dependencies,deps,episode.dependency_versions,versions)
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
                      "HANDOFF_EVIDENCE_UNSUPPORTED"}:
            if affected:
                self._log("RECOVERY_COORDINATION_REQUIRED",episode,reason=reason,affected_uav_ids=list(affected))
            self._exit(episode,reason)
        elif episode.phase.startswith("REASSIGN"):
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
        ChatMessage("system","Return one authorized Spatial V3 suffix JSON only. Preserve retained step IDs, target references, durations, goal conditions and termination. Never replay completed steps or TAKEOFF. Do not rewrite another UAV or reinterpret the original user instruction. Relative geometry uses only the supplied immutable anchor. Routing, versions, permissions and deadlines are trusted constants."),
        ChatMessage("user",json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")))),
        options=GenerationOptions(temperature=0.0,max_tokens=4096,
            response_format=JsonSchemaResponseFormat("local_repair_v3",suffix_schema)))
    if not isinstance(response,ModelResponse) or len(response.content.encode("utf-8"))>65536:
        raise LocalRepairError("INVALID_MODEL_RESPONSE","invalid or oversized repair response")
    return LocalRepairDraftV3.from_dict(strict_json_object_loads(response.content))
