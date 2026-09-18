"""Recovery integration with real Agent/Manager/V3 contracts and fake model IO."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from threading import Event, RLock
from types import SimpleNamespace

import numpy as np
import pytest

from agents.mission_agent import AgentStatus, MissionAgent
from common.obstacle_types import ObstacleAABB
from configs.schema import FleetRecoveryConfig
from env.kinematic_uav import KinematicUAV, UAVState
from fleet.airspace_manager import FleetPoseSnapshot, FleetUavPose
from fleet.local_repair import (
    ExternalDependencySnapshot, LocalRepairError, Transferability, build_remaining_task_contract,
)
from fleet.local_spatial_planner import RoutedPreplannedSpatialPlanner
from fleet.model_request_broker import GlobalModelRequestBroker, ModelBrokerRequest, ModelRequestPriority
from fleet.recovery_controller import FleetRecoveryController
from scripts.run_fleet_mission import _build_fleet_recovery_controller
from fleet.runtime import AssignmentStatus
from fleet.task_spec import ConstraintStrength, FleetTaskSpecV1, GoalType, MissionGoal, OrderingConstraint
from models.adapter_registry import AdapterSelection, AdapterStatus, ModelCallRole
from models.base import ModelResponse
from planner.schemas import LandingZoneSpec, PlannerWorldContext
from planner.schemas_v3 import SkillPlanDraftV3
from planner.spatial import CoordinateFrame, NamedLocationTarget, PointTarget
from runtime.plan_validator import PlanValidator
from runtime.safety_supervisor import SafetySupervisor
from skills.hover import HoverSkill
from skills.manager import SkillManager
from skills.types import Observation, SkillContext, SkillName, SkillResultCode, SkillStatus
from target.target_manager import TargetManager
from tests.test_mission_agent import FakeCamera, FakeClock, ScriptedSkill, failed, running, succeeded


class WallClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


@dataclass
class Local:
    agent: MissionAgent
    manager: SkillManager
    clock: FakeClock
    uav: KinematicUAV
    world: PlannerWorldContext
    home: str
    goal: MissionGoal
    assignment: object
    goto: ScriptedSkill
    takeoff: ScriptedSkill
    instruction: str

    def tick(self, timestamp):
        self.clock.set(timestamp)
        self.agent.tick(Observation(uav_id=self.assignment.uav_id, timestamp=float(timestamp),
            uav_pose=self.uav.get_pose(), uav_velocity=np.zeros(3), camera_rgb=np.zeros((2, 2, 3), dtype=np.uint8)))

    def fail(self, *, hold=True):
        self.uav.set_pose(*self.world.initial_uav_xyz_m[:2], 10.0, 0.0)
        self.tick(1.0)
        self.tick(2.0)
        assert self.agent.local_repair_snapshot.event is not None
        if hold:
            self.tick(3.0)
            assert self.agent.local_repair_snapshot.stable_hold


def local(uav_id="uav_1", x=0.0):
    home = "home_" + uav_id
    world = PlannerWorldContext((-100, -100, 0), (100, 100, 30), (x, 0, 0), {},
        {home: LandingZoneSpec(home, (x, 0))}, 10, 10, 60)
    goal = MissionGoal("goal_" + uav_id, GoalType.NAVIGATE, None,
        PointTarget(CoordinateFrame.WORLD_ENU, (x + 10, 0, 10)), None, None, ConstraintStrength.MUST)
    steps = [
        {"id": "takeoff", "skill": "TAKEOFF", "args": {"altitude_m": 10}},
        {"id": "goto", "skill": "GOTO", "args": {"target": goal.spatial_constraint.to_dict()}},
        {"id": "home", "skill": "GOTO", "args": {"target": NamedLocationTarget(home).to_dict()}},
        {"id": "land", "skill": "LAND", "args": {"zone": home}},
    ]
    draft = SkillPlanDraftV3.from_dict({"schema_version": 3, "mission_id": "mission_template", "uav_id": uav_id,
        "plan_version": 1, "assumptions": [], "steps": [{**step, "uav_id": uav_id} for step in steps]})
    instruction = json.dumps({"schema_version": 2, "uav_id": uav_id, "assigned_goals": [goal.to_dict()]})
    planner = RoutedPreplannedSpatialPlanner(draft, source="dynamic_scripted", expected_instruction=instruction)
    clock = FakeClock()
    uav = KinematicUAV(UAVState(x, 0, 0, 0), max_speed_mps=5, max_yaw_rate_rad_s=2)
    takeoff = ScriptedSkill(succeeded(SkillResultCode.TAKEOFF_COMPLETE))
    goto = ScriptedSkill(failed(SkillResultCode.TIMEOUT), *[running() for _ in range(20)])
    manager = SkillManager(SkillContext(uav=uav, camera=FakeCamera(), perception=None, clock=clock, uav_id=uav_id),
        registry={SkillName.TAKEOFF: takeoff, SkillName.GOTO: goto, SkillName.HOVER: HoverSkill(),
                  SkillName.LAND: ScriptedSkill(succeeded(SkillResultCode.LAND_COMPLETE))})
    agent = MissionAgent(planner=planner, validator=PlanValidator(), skill_manager=manager,
        safety=SafetySupervisor(world.scene_min_xyz_m, world.scene_max_xyz_m,
            max_mission_time_s=300, max_safe_altitude_m=25), target_manager=TargetManager(), clock=clock)
    assignment = SimpleNamespace(assignment_id="assignment_" + uav_id, uav_id=uav_id, goal_ids=(goal.goal_id,))
    return Local(agent, manager, clock, uav, world, home, goal, assignment, goto, takeoff, instruction)


def track_local(uav_id="uav_1", x=0.0):
    """The assignment's only goal is a TRACK that never started; the injected
    fault sits on the first transit GOTO in front of it, so the TRACK remains
    a pending, unexecuted obligation."""
    home = "home_" + uav_id
    world = PlannerWorldContext((-100, -100, 0), (100, 100, 30), (x, 0, 0), {},
        {home: LandingZoneSpec(home, (x, 0))}, 10, 10, 60)
    goal = MissionGoal("goal_track_" + uav_id, GoalType.TRACK_TARGET, "target_a", None, 10, None, ConstraintStrength.MUST)
    steps = [
        {"id": "takeoff", "skill": "TAKEOFF", "args": {"altitude_m": 10}},
        {"id": "goto", "skill": "GOTO", "args": {"target": PointTarget(CoordinateFrame.WORLD_ENU, (x + 12, 0, 10)).to_dict()}},
        {"id": "search", "skill": "SEARCH", "args": {
            "region": {"shape": "RECTANGLE", "frame": "WORLD_ENU", "center_xyz_m": [x + 10, 0, 0],
                "width_m": 6, "height_m": 6},
            "strategy": {"kind": "LAWNMOWER", "spacing_m": 3},
            "entry_policy": "START_IN_PLACE_IF_INSIDE", "target_description": "a moving person",
            "search_altitude_m": 10, "timeout_s": 30}},
        {"id": "track", "skill": "TRACK", "args": {"target_ref": "$search.target_id", "duration_s": 10}},
        {"id": "home", "skill": "GOTO", "args": {"target": NamedLocationTarget(home).to_dict()}},
        {"id": "land", "skill": "LAND", "args": {"zone": home}},
    ]
    draft = SkillPlanDraftV3.from_dict({"schema_version": 3, "mission_id": "mission_template", "uav_id": uav_id,
        "plan_version": 1, "assumptions": [], "steps": [{**step, "uav_id": uav_id} for step in steps]})
    instruction = json.dumps({"schema_version": 2, "uav_id": uav_id, "assigned_goals": [goal.to_dict()]})
    planner = RoutedPreplannedSpatialPlanner(draft, source="dynamic_scripted", expected_instruction=instruction)
    clock = FakeClock()
    uav = KinematicUAV(UAVState(x, 0, 0, 0), max_speed_mps=5, max_yaw_rate_rad_s=2)
    takeoff = ScriptedSkill(succeeded(SkillResultCode.TAKEOFF_COMPLETE))
    goto = ScriptedSkill(failed(SkillResultCode.TIMEOUT), *[running() for _ in range(20)])
    manager = SkillManager(SkillContext(uav=uav, camera=FakeCamera(), perception=None, clock=clock, uav_id=uav_id),
        registry={SkillName.TAKEOFF: takeoff, SkillName.GOTO: goto,
                  SkillName.SEARCH: ScriptedSkill(succeeded(SkillResultCode.TARGET_FOUND, {"target_id": "target_1"})),
                  SkillName.TRACK: ScriptedSkill(succeeded(SkillResultCode.TRACK_COMPLETE)),
                  SkillName.REACQUIRE: ScriptedSkill(succeeded(SkillResultCode.TRACK_COMPLETE)),
                  SkillName.HOVER: HoverSkill(), SkillName.LAND: ScriptedSkill(succeeded(SkillResultCode.LAND_COMPLETE))})
    agent = MissionAgent(planner=planner, validator=PlanValidator(), skill_manager=manager,
        safety=SafetySupervisor(world.scene_min_xyz_m, world.scene_max_xyz_m,
            max_mission_time_s=300, max_safe_altitude_m=25), target_manager=TargetManager(), clock=clock)
    assignment = SimpleNamespace(assignment_id="assignment_" + uav_id, uav_id=uav_id, goal_ids=(goal.goal_id,))
    return Local(agent, manager, clock, uav, world, home, goal, assignment, goto, takeoff, instruction)


class Records:
    def __init__(self, locals):
        self.rows = {item.assignment.assignment_id: SimpleNamespace(assignment=item.assignment,
            local_plan_version=1, status=AssignmentStatus.RUNNING, last_error=None) for item in locals}

    @property
    def records(self):
        return tuple(self.rows.values())

    def by_id(self, assignment_id):
        return self.rows[assignment_id]

    def update(self, assignment_id, status, **kwargs):
        record = self.rows[assignment_id]
        record.status = status
        for name, value in kwargs.items():
            setattr(record, name, value)
        return record


class ModelFactory:
    def __init__(self, locals, *, blocked=False):
        self.started = {item.assignment.uav_id: Event() for item in locals}
        self.release = {uav: Event() for uav in self.started}
        if not blocked:
            for event in self.release.values():
                event.set()
        self.calls = []
        self.invalid = False
        self.transform = None

    def selection_for_role(self, role):
        return AdapterSelection(role, "runtime_replan" if role is ModelCallRole.RUNTIME_REPLAN else "fleet_planner",
            AdapterStatus.PLACEHOLDER, "base_test_model", True)

    def for_role(self, role, **routing):
        factory = self
        uav = routing["uav_id"]

        class Client:
            def chat(self, messages, *, options=None):
                factory.calls.append((role, dict(routing)))
                factory.started[uav].set()
                assert factory.release[uav].wait(2.0)
                payload = json.loads(messages[1].content)
                if "trusted_joint_context" in payload:
                    joint = payload["trusted_joint_context"]
                    data = {uav_id: {**payload["authorized_output"][uav_id],
                                     "steps": joint["original_suffixes"][uav_id]}
                            for uav_id in joint["editable_uavs"]}
                else:
                    data = {**payload["authorized_output"], "steps": payload["original_suffix"]}
                if factory.transform is not None:
                    data = factory.transform(data, payload)
                return ModelResponse(content="invalid JSON" if factory.invalid else json.dumps(data),
                    model="base_test_model", finish_reason="stop", usage={})
        return Client()


class Harness:
    def __init__(self, *, count=1, blocked=False, config=None, ordering=False, geometry_error=False):
        self.locals = [local("uav_" + str(index + 1), 30.0 * index) for index in range(count)]
        self.clock = WallClock()
        self.factory = ModelFactory(self.locals, blocked=blocked)
        self.broker = GlobalModelRequestBroker(max_inflight_global=count, clock=self.clock)
        self.geometry_error = geometry_error
        self.geometry_calls = 0
        self.geometry_hook = None
        self.obstacles = ()
        self.shared_dependencies = ()
        self.map_version = self.reference_version = 1
        self.events = []
        self.exits = []
        self.runtime = SimpleNamespace(agents={item.assignment.uav_id: item.agent for item in self.locals},
            assignments=Records(self.locals), fleet_plan=SimpleNamespace(fleet_mission_id="fleet_test", fleet_plan_version=1,
                coordination_policy=SimpleNamespace(minimum_uav_separation_m=2.0)), cancel_requested=False,
            _planned_routes={}, _route_progress={}, _recovery_commit_lock=RLock(), _pending_reassignments=set())
        self.runtime._event = lambda event, **values: self.events.append({"event": event, **values})

        def exit_local(assignment_id, reason):
            self.exits.append((assignment_id, reason))
            row = self.runtime.assignments.by_id(assignment_id)
            self.runtime.agents[row.assignment.uav_id].cancel()
            row.status = AssignmentStatus.CANCELING
        self.runtime._begin_local_failsafe_landing = exit_local
        constraints = ()
        if ordering:
            constraints = (OrderingConstraint("order_ab", self.locals[0].goal.goal_id,
                self.locals[1].goal.goal_id, ConstraintStrength.MUST),)
        self.spec = FleetTaskSpecV1(source_text="visit assigned places", goals=tuple(item.goal for item in self.locals),
            ordering_constraints=constraints)
        self.prepared = SimpleNamespace(
            config=SimpleNamespace(fleet_recovery=config or FleetRecoveryConfig(enabled=True, mode="LOCAL_ONLY")),
            task_spec=self.spec, fleet_plan_v2=SimpleNamespace(assignments=[item.assignment for item in self.locals]),
            compilations={}, world_contexts={item.assignment.uav_id: item.world for item in self.locals},
            request=SimpleNamespace(uav_inventory=[SimpleNamespace(uav_id=item.assignment.uav_id, home_name=item.home)
                                                  for item in self.locals]),
            model_client_factory=self.factory, planner_limits=None, planner_policy=None)
        self.controller = _build_fleet_recovery_controller(self.prepared, broker=self.broker,
            geometry_provider=self.geometry, replan_boundary=None, clock=self.clock)
        self.controller.bind(self.runtime)
        for item in self.locals:
            compiled = item.agent.start(item.instruction, item.world)
            self.controller.compilations[item.assignment.uav_id] = compiled

    def geometry(self):
        self.geometry_calls += 1
        if self.geometry_hook is not None:
            self.geometry_hook()
        if self.geometry_error and self.geometry_calls > 1:
            raise ValueError("trusted geometry temporarily unavailable")
        poses = {}
        for item in self.locals:
            pose = item.uav.get_pose()
            poses[item.assignment.uav_id] = FleetUavPose(item.assignment.uav_id, (pose.x, pose.y, pose.z))
        return {"fleet_pose_snapshot": FleetPoseSnapshot(max(item.clock.now() for item in self.locals), poses),
            "obstacles": self.obstacles, "map_version": self.map_version, "reference_version": self.reference_version,
            "shared_dependencies": self.shared_dependencies}

    def submit(self, index=0):
        self.locals[index].fail()
        self.controller.tick()
        episode = self.controller.episodes[self.locals[index].assignment.assignment_id]
        assert self.factory.started[self.locals[index].assignment.uav_id].wait(2.0)
        return episode

    def complete(self, episode):
        thread = self.controller.runner._active.get(episode.request_id)
        if thread is not None:
            thread.join(2.0)
            assert not thread.is_alive()
        self.controller.tick()

    def close(self):
        for event in self.factory.release.values():
            event.set()
        self.controller.close()


@pytest.fixture
def harness():
    values = []
    def build(**kwargs):
        result = Harness(**kwargs)
        values.append(result)
        return result
    yield build
    for result in values:
        result.close()


def test_real_event_waits_for_stable_hold_and_commits_suffix_without_replaying_takeoff(harness):
    h = harness()
    item = h.locals[0]
    item.fail(hold=False)
    h.controller.tick()
    assert h.factory.calls == []
    event_id = item.agent.local_repair_snapshot.event.event_id
    item.tick(3)
    h.controller.tick()
    episode = h.controller.episodes[item.assignment.assignment_id]
    assert h.factory.started[item.assignment.uav_id].wait(2)
    h.complete(episode)
    assert item.agent.snapshot().plan_version == item.manager.task_plan.plan_version == 2
    assert h.runtime.assignments.by_id(item.assignment.assignment_id).local_plan_version == 2
    assert len(item.takeoff.started_goals) == 1
    assert len(item.goto.started_goals) == 2
    assert event_id in h.controller.finished_events
    assert h.exits == []
    assert len(h.factory.calls) == 1
    assert h.controller._evidence(item.agent.local_repair_snapshot)[0].result.code is SkillResultCode.TAKEOFF_COMPLETE
    for _ in range(3):
        h.controller.tick()
    assert len(h.factory.calls) == 1


def test_two_independent_uavs_complete_out_of_order_without_crossing_results_or_versions(harness):
    h = harness(count=2, blocked=True)
    first = h.submit(0)
    second = h.submit(1)
    assert h.broker.inflight_count == 2
    h.factory.release["uav_2"].set()
    h.complete(second)
    assert h.locals[1].agent.snapshot().plan_version == 2
    assert h.locals[0].agent.snapshot().plan_version == 1
    h.factory.release["uav_1"].set()
    h.complete(first)
    assert [item.agent.snapshot().plan_version for item in h.locals] == [2, 2]
    assert h.exits == []


def test_preparation_errors_consume_attempt_budget_and_do_not_create_request_storm(harness):
    h = harness(geometry_error=True, config=FleetRecoveryConfig(enabled=True, mode="LOCAL_ONLY", max_local_attempts=2))
    h.locals[0].fail()
    h.controller.tick()
    episode = next(iter(h.controller.episodes.values()))
    assert episode.attempts == 1
    h.controller.tick()
    for _ in range(10):
        h.controller.tick()
    assert episode.attempts == 1
    h.clock.value += 1
    h.controller.tick()
    h.controller.tick()
    assert episode.attempts == 2
    assert len(h.exits) == 1
    assert h.factory.calls == []
    assert h.controller.episodes == {}


def test_expired_old_episode_cannot_land_a_newer_local_execution(harness):
    h = harness(blocked=True)
    episode = h.submit()
    item = h.locals[0]
    state = item.agent.local_repair_snapshot
    item.agent.commit_local_repair(replace(state.task_plan, plan_version=2),
        expected_event_id=state.event.event_id, expected_plan_version=1)
    h.runtime.assignments.by_id(item.assignment.assignment_id).local_plan_version = 2
    h.clock.value = episode.deadline_wall_s + 1
    h.controller.tick()
    assert h.exits == []
    assert item.agent.snapshot().plan_version == 2
    assert item.agent.snapshot().status is AgentStatus.RUNNING
    assert h.controller.episodes == {}
    assert h.broker.inflight_count == 1


def test_queued_request_is_rejected_before_model_start_when_execution_changes(harness):
    h = harness()
    blocker = ModelBrokerRequest(call_role=ModelCallRole.FLEET_REPLAN, priority=ModelRequestPriority.P1_FLEET_REPLAN,
        uav_id="other_uav", request_id="request_blocker", submitted_at_s=h.clock())
    h.broker.submit(blocker)
    assert h.broker.acquire_next() == blocker
    h.locals[0].fail()
    h.controller.tick()
    episode = next(iter(h.controller.episodes.values()))
    assert episode.phase == "LOCAL_QUEUED"
    assert h.factory.calls == []
    h.locals[0].agent.cancel()
    h.broker.complete(blocker.request_id)
    h.controller.tick()
    assert h.factory.calls == []
    assert h.exits == []
    assert h.controller.episodes == {}


def test_final_guard_rechecks_wall_deadline_after_safety_preflight(harness):
    h = harness()
    episode = h.submit()
    item = h.locals[0]
    original = item.agent._safety.preflight
    def expires(candidate):
        result = original(candidate)
        h.clock.value = episode.context.deadline_wall_s
        return result
    item.agent._safety.preflight = expires
    h.complete(episode)
    assert item.agent.snapshot().plan_version == item.manager.task_plan.plan_version == 1
    assert h.runtime.assignments.by_id(item.assignment.assignment_id).local_plan_version == 1
    assert len(item.goto.started_goals) == 1
    assert not any(event["event"] == "RECOVERY_COMMITTED" for event in h.events)


def test_current_obstacles_reject_unsafe_entry_connector_before_publication(harness):
    h = harness()
    episode = h.submit()
    h.obstacles = (SimpleNamespace(collidable=True, aabb=ObstacleAABB((4, -1, 9), (6, 1, 11))),)
    h.complete(episode)
    assert h.locals[0].agent.snapshot().plan_version == 1
    assert len(h.locals[0].goto.started_goals) == 1
    assert h.controller._last_error[episode.assignment_id] == "UNSAFE_ENTRY_OR_ROUTE"


def test_reference_change_and_shared_dependencies_are_explicit_coordination_failures(harness):
    h = harness()
    episode = h.submit()
    h.reference_version += 1
    h.complete(episode)
    assert h.locals[0].agent.snapshot().plan_version == 1
    assert h.exits[0][1] == "COORDINATION_REQUIRED"
    second = harness()
    second.shared_dependencies = (ExternalDependencySnapshot("shared_corridor", "SHARED_RESOURCE",
        (second.locals[0].goal.goal_id,), ("uav_1", "uav_other"), 1, "UNKNOWN"),)
    second.locals[0].fail()
    second.controller.tick()
    second.controller.tick()
    assert second.factory.calls == []
    assert second.exits[0][1] == "COORDINATION_REQUIRED"
    event = next(event for event in second.events if event["event"] == "RECOVERY_COORDINATION_REQUIRED")
    assert event["affected_uav_ids"] == ["uav_1", "uav_other"]


def test_external_predecessor_active_without_success_proof_cannot_be_dropped(harness):
    h = harness(count=2, ordering=True)
    h.locals[0].fail()
    h.locals[1].fail()
    h.controller.tick()
    h.controller.tick()
    assert h.factory.calls == []
    assert len(h.exits) == 2
    assert all(reason == "COORDINATION_REQUIRED" for _, reason in h.exits)



class RealFleetScenario:
    """Real production controller helper + Runtime/Agent, only environment/model IO is fake."""
    def __init__(self, items=None, *, spare=None, boundary_builder=None):
        from fleet.runtime import FleetMissionRuntime
        from fleet.scripted_planner import ScriptedFleetPlanner
        from fleet.types import FleetCoordinationPolicy, FleetMissionRequest, FleetTargetRequest, FleetUavCapability
        from fleet.types_v2 import FleetAssignmentV2, FleetMissionPlanV2, FleetMissionRequestV2
        from planner.spatial import CircleRegion
        from target.types import TargetSpec

        self.items = items or [local(f"uav_{i + 1}", (i - 2) * 30.0) for i in range(5)]
        self.fault_index = 2 if len(self.items) == 5 else 0
        for index, item in enumerate(self.items):
            if index != self.fault_index:
                item.goto._outcomes.popleft()  # healthy aircraft never emits the injected TIMEOUT
        all_items = self.items + ([] if spare is None else [spare])
        self.by_uav = {item.assignment.uav_id: item for item in all_items}
        self.wall = WallClock()
        self.factory = ModelFactory(all_items, blocked=True)
        self.broker = GlobalModelRequestBroker(max_inflight_global=2, clock=self.wall)
        inventory = tuple(FleetUavCapability(item.assignment.uav_id, item.assignment.uav_id,
            True, item.home, 5, 30) for item in all_items)
        self.request = FleetMissionRequest(fleet_mission_id="fleet_real_tick", fleet_plan_version=1,
            original_instruction="visit assigned world points", uav_inventory=inventory,
            target_requests=tuple(FleetTargetRequest("envelope_" + item.assignment.uav_id,
                TargetSpec("execution envelope"), requested_uav_id=item.assignment.uav_id,
                search_region=CircleRegion(CoordinateFrame.WORLD_ENU, (item.world.initial_uav_xyz_m[0]+10, 0, 0), 5),
                track_duration_s=5) for item in self.items), coordination_policy=FleetCoordinationPolicy())
        planner = ScriptedFleetPlanner()
        plan = planner.plan(self.request)
        semantics = []
        for item in self.items:
            runtime_assignment = next(a for a in plan.assignments if a.uav_id == item.assignment.uav_id)
            item.assignment = FleetAssignmentV2(runtime_assignment.assignment_id, runtime_assignment.uav_id,
                (item.goal.goal_id,), priority=100, start_policy="PARALLEL")
            semantics.append(item.assignment)
        spec = FleetTaskSpecV1(source_text="visit points", goals=tuple(item.goal for item in self.items))
        config = SimpleNamespace(fleet_recovery=FleetRecoveryConfig(enabled=True, mode="LOCAL_ONLY",
            max_local_attempts=1))
        self.prepared = SimpleNamespace(config=config,
            fleet_plan_v2=FleetMissionPlanV2(self.request.fleet_mission_id, 1, tuple(semantics),
                self.request.coordination_policy),
            fleet_request_v2=FleetMissionRequestV2(self.request.fleet_mission_id, 1, spec, inventory, (),
                self.request.coordination_policy),
            task_spec=spec, model_client_factory=self.factory, compilations={},
            world_contexts={uav: item.world for uav,item in self.by_uav.items()},
            request=self.request, plan=plan, planner_limits=None, planner_policy=None,
            preparation_context={}, local_planner_source="dynamic_llm")

        scenario = self
        class Environment:
            def __init__(self):
                self.steps = 0
                self.assignments = {}
            def start(self, plan):
                self.assignments = {a.uav_id: a.target_alias for a in plan.assignments}
            def step(self):
                self.steps += 1
                for item in all_items:
                    item.clock.set(float(self.steps))
                    if self.steps == 1 and item in scenario.items:
                        item.uav.set_pose(*item.world.initial_uav_xyz_m[:2], 10, 0)
            def get_agent_observation(self, uav_id):
                item = scenario.by_uav[uav_id]
                return Observation(uav_id=uav_id, timestamp=item.clock.now(), uav_pose=item.uav.get_pose(),
                    uav_velocity=np.zeros(3), camera_rgb=np.zeros((2,2,3), dtype=np.uint8))
            def get_fleet_pose_snapshot(self):
                poses = {}
                for uav,item in scenario.by_uav.items():
                    pose = item.uav.get_pose()
                    poses[uav] = FleetUavPose(uav, (pose.x,pose.y,pose.z))
                return FleetPoseSnapshot(float(self.steps), poses)
            def prepare_assignment_update(self, assignments):
                return dict(assignments)
            def commit_assignment_update(self, assignments):
                self.assignments = assignments
            def close(self):
                pass

        self.env = Environment()
        boundary = None if boundary_builder is None else boundary_builder(self)
        geometry = lambda: {"fleet_pose_snapshot":self.env.get_fleet_pose_snapshot(),
                           "obstacles":(), "map_version":1, "reference_version":1}
        self.controller = _build_fleet_recovery_controller(self.prepared, broker=self.broker,
            geometry_provider=geometry, replan_boundary=boundary, clock=self.wall)
        self.runtime = FleetMissionRuntime(self.env, planner, {i.assignment.uav_id:i.agent for i in self.items},
            inventory=inventory, target_requests=self.request.target_requests,
            coordination_policy=self.request.coordination_policy,
            precomputed_start_inputs={i.assignment.uav_id:(i.instruction,i.world) for i in self.items},
            non_target_assignment_ids=tuple(a.assignment_id for a in semantics),
            assignment_requiredness={a.assignment_id:True for a in semantics},
            model_broker=self.broker, recovery_controller=self.controller)
        self.runtime.start(self.request.original_instruction, request=self.request)
        self.original_plans = {i.assignment.uav_id:i.manager.task_plan for i in self.items}

    def submit_fault(self):
        for _ in range(3):
            self.runtime.tick()
        item = self.items[self.fault_index]
        assert self.factory.started[item.assignment.uav_id].wait(2)
        return self.controller.episodes[item.assignment.assignment_id]

    def finish_request(self, episode):
        thread = self.controller.runner._active.get(episode.request_id)
        if thread is not None:
            thread.join(2)
            assert not thread.is_alive()
        if self.runtime.status.value == "RUNNING":
            self.runtime.tick()
        else:
            self.controller.tick()

    def close(self):
        for event in self.factory.release.values():
            event.set()
        self.runtime.close()


@pytest.mark.parametrize("cancel_during_model", [False, True])
def test_five_uavs_only_third_repairs_while_healthy_safety_and_cancel_keep_progressing(cancel_during_model):
    from fleet.runtime import FleetStatus
    scenario = RealFleetScenario()
    runtime, controller = scenario.runtime, scenario.controller
    try:
        episode = scenario.submit_fault()
        third = scenario.items[2]
        healthy = [i for n,i in enumerate(scenario.items) if n != 2]
        before = [i.goto.tick_count for i in healthy]
        safety_checks = []
        original_check = third.agent._safety.evaluate
        def counted_check(*args, **kwargs):
            safety_checks.append(True)
            return original_check(*args, **kwargs)
        third.agent._safety.evaluate = counted_check
        for _ in range(3):
            runtime.tick()
        assert scenario.env.steps == 6
        assert [i.goto.tick_count for i in healthy] == [count + 3 for count in before]
        assert all(i.manager.task_plan.to_dict() == scenario.original_plans[i.assignment.uav_id].to_dict() for i in healthy)
        assert all(len(i.takeoff.started_goals) == len(i.goto.started_goals) == 1 for i in healthy)
        assert third.agent.local_repair_snapshot.stable_hold
        assert safety_checks and scenario.broker.inflight_count == 1
        if cancel_during_model:
            runtime.cancel()
            runtime.tick()  # cancellation and landing advance before the blocked model returns
            assert runtime.cancel_requested
            assert controller.episodes == {}
            assert scenario.broker.inflight_count == 1
        scenario.factory.release["uav_3"].set()
        scenario.finish_request(episode)
        if cancel_during_model:
            assert third.manager.task_plan.plan_version == 1
            assert len(third.goto.started_goals) == 1
            assert not any(e.get("event_type") == "RECOVERY_COMMITTED" for e in runtime._events)
        else:
            assert runtime.status is FleetStatus.RUNNING
            assert third.agent.snapshot().plan_version == third.manager.task_plan.plan_version == 2
            assert runtime.assignments.by_id(third.assignment.assignment_id).local_plan_version == 2
            assert len(third.takeoff.started_goals) == 1 and len(third.goto.started_goals) == 2
            assert all(i.agent.snapshot().plan_version == 1 for i in healthy)
        assert len(scenario.factory.calls) == 1
        assert scenario.factory.calls[0][0] is ModelCallRole.RUNTIME_REPLAN
        assert scenario.factory.calls[0][1]["uav_id"] == "uav_3"
    finally:
        scenario.close()


def production_boundary(scenario, tmp_path, *, spare=None):
    """Use the production split handler and actual compiler; fake only model outputs."""
    from configs.loader import load_config
    from experiments.planning_audit_logger import PlanningAuditLogger
    from fleet.runtime import ReplannedAssignment
    from fleet.types_v2 import FleetAssignmentV2, FleetMissionPlanV2
    from models.base import ChatMessage
    from scripts.run_fleet_mission import _build_runtime_fleet_replan_handler

    config = load_config("configs/multi_uav_demo.yaml")
    source = scenario.items[0]
    items = [source] + ([] if spare is None else [spare])
    uavs = tuple(replace(template, id=item.assignment.uav_id, home_name=item.home,
        initial_position_xyz_m=item.world.initial_uav_xyz_m) for template,item in zip(config.uavs,items))
    config = replace(config, uav=None, camera=None, target=None, uavs=uavs, fleet_recovery=FleetRecoveryConfig(enabled=True,
        mode="LOCAL_THEN_REASSIGN", max_local_attempts=1))
    scenario.prepared.config = config
    scenario.fleet_started = Event()
    scenario.fleet_release = Event()
    scenario.created = []
    original_for_role = scenario.factory.for_role
    def for_role(role, **routing):
        if role is not ModelCallRole.FLEET_REPLAN:
            return original_for_role(role, **routing)
        class Client:
            def chat(self, messages, *, options=None):
                scenario.factory.calls.append((role, dict(routing)))
                scenario.fleet_started.set()
                assert scenario.fleet_release.wait(2)
                return ModelResponse(content="{}", model="fake", finish_reason="stop", usage={})
        return Client()
    scenario.factory.for_role = for_role

    class FleetPlanner:
        model_proposals = ()
        def __init__(self, client):
            self.client = client
        def plan(self, request):
            self.client.chat((ChatMessage("user", "select idle standby from frozen request"),))
            available = next(uav for uav in request.uav_inventory if uav.available)
            return FleetMissionPlanV2(request.fleet_mission_id, request.fleet_plan_version,
                (FleetAssignmentV2("assignment_handoff", available.uav_id, request.task_spec.all_goal_ids,
                    priority=100, start_policy="PARALLEL"),), request.coordination_policy)

    class NavigationPlanner:
        source = "dynamic_llm"
        model_proposals = ()
        def plan(self, request):
            payload = json.loads(request.instruction)
            goal = payload["assigned_goals"][0]
            steps = [
                {"id":"takeoff", "skill":"TAKEOFF", "args":{"altitude_m":10}},
                {"id":"goto", "skill":"GOTO", "args":{"target":goal["spatial_constraint"]}},
                {"id":"home", "skill":"GOTO", "args":{"target":{"kind":"NAMED_LOCATION", "name":payload["own_home"]}}},
                {"id":"land", "skill":"LAND", "args":{"zone":payload["own_home"]}},
            ]
            return SkillPlanDraftV3.from_dict({"schema_version":3,"mission_id":request.mission_id,
                "uav_id":request.uav_id,"plan_version":request.plan_version,"assumptions":[],
                "steps":[{**step,"uav_id":request.uav_id} for step in steps]})

    def agent_factory(record, assignment, compilation, world, runtime_assignment, route):
        assert spare is not None
        assert scenario.runtime.agents.get(spare.assignment.uav_id) is None
        assert scenario.prepared.preparation_context["runtime_reassignments"] == []
        replay = RoutedPreplannedSpatialPlanner(compilation.planner_output,
            source="dynamic_llm", expected_instruction=compilation.planner_request.instruction)
        spare.agent = MissionAgent(planner=replay, validator=PlanValidator(), skill_manager=spare.manager,
            safety=SafetySupervisor(world.scene_min_xyz_m, world.scene_max_xyz_m,
                max_mission_time_s=300, max_safe_altitude_m=25),
            target_manager=TargetManager(), clock=spare.clock)
        spare.agent.configure_local_repair(enabled=True, max_wait_s=300)
        scenario.created.append(spare.agent)
        return ReplannedAssignment(record.assignment.assignment_id, spare.agent,
            (compilation.planner_request.instruction, world), planned_route=route,
            replacement_assignment=runtime_assignment)

    return _build_runtime_fleet_replan_handler(scenario.prepared, audit=PlanningAuditLogger(tmp_path),
        agent_factory=agent_factory, fleet_planner_factory=FleetPlanner,
        local_planner_factory=lambda client,uav: NavigationPlanner())


@pytest.mark.parametrize("local_success", [True, False])
def test_no_standby_still_attempts_local_then_exits_if_unrepairable(tmp_path, local_success):
    scenario = RealFleetScenario([local("uav_a", -30)],
        boundary_builder=lambda s: production_boundary(s, tmp_path))
    try:
        scenario.factory.invalid = not local_success
        episode = scenario.submit_fault()
        scenario.factory.release["uav_a"].set()
        scenario.finish_request(episode)
        if local_success:
            assert scenario.items[0].agent.snapshot().plan_version == 2
            assert scenario.controller.episodes == {}
        else:
            # The empty standby set fails owner preparation before any Fleet model request.
            scenario.runtime.tick()
            scenario.runtime.tick()
            assert scenario.controller.episodes == {}
            assert not scenario.runtime._pending_reassignments
            assert scenario.items[0].agent.snapshot().plan_version == 1
        assert [role for role,_ in scenario.factory.calls] == [ModelCallRole.RUNTIME_REPLAN]
        assert not scenario.fleet_started.is_set()
    finally:
        scenario.fleet_release.set()
        scenario.close()


def test_production_async_standby_handoff_then_new_agent_local_repair(tmp_path):
    source, spare = local("uav_a", -30), local("uav_b", 30)
    scenario = RealFleetScenario([source], spare=spare,
        boundary_builder=lambda s: production_boundary(s, tmp_path, spare=spare))
    try:
        scenario.factory.invalid = True
        episode = scenario.submit_fault()
        scenario.factory.release["uav_a"].set()
        scenario.finish_request(episode)
        scenario.runtime.tick()
        assert scenario.fleet_started.wait(2)
        assert source.agent.local_repair_snapshot.stable_hold
        before = scenario.env.steps
        for _ in range(3):
            scenario.runtime.tick()
        assert scenario.env.steps == before + 3
        assert scenario.created == []
        assert scenario.prepared.preparation_context["runtime_reassignments"] == []
        assert "uav_b" not in scenario.runtime.agents
        scenario.fleet_release.set()
        scenario.finish_request(episode)
        assert scenario.runtime.fleet_plan.fleet_plan_version == 2, scenario.runtime._events
        assert scenario.runtime.agents["uav_b"] is spare.agent
        assert len(scenario.created) == 1
        assert len(scenario.prepared.preparation_context["runtime_reassignments"]) == 1
        assert "uav_a" in scenario.runtime._retired_failsafe_agents
        # Advance the actual newly published Agent to its own recoverable GOTO failure.
        scenario.factory.invalid = False
        spare.uav.set_pose(30, 0, 10, 0)
        for _ in range(4):
            scenario.runtime.tick()
            assert spare.agent.snapshot().status is AgentStatus.RUNNING, json.dumps(scenario.runtime._events[-14:])
        assert scenario.factory.started["uav_b"].wait(2), scenario.runtime._events
        replacement = scenario.controller.episodes["assignment_handoff"]
        scenario.factory.release["uav_b"].set()
        scenario.finish_request(replacement)
        assert spare.agent.snapshot().plan_version == 3, scenario.runtime._events
        assert len(spare.takeoff.started_goals) == 1
        assert [role for role,_ in scenario.factory.calls] == [
            ModelCallRole.RUNTIME_REPLAN, ModelCallRole.FLEET_REPLAN, ModelCallRole.RUNTIME_REPLAN]
        assert scenario.runtime.assignments.by_id("assignment_handoff").local_plan_version == 3
    finally:
        scenario.fleet_release.set()
        scenario.close()


def test_live_hold_drift_consumes_existing_budget_and_exits_instead_of_abandoning_episode(harness):
    h = harness(config=FleetRecoveryConfig(enabled=True, mode="LOCAL_ONLY", max_local_attempts=1))
    episode = h.submit()
    item = h.locals[0]
    item.uav.set_pose(4, 0, 10, 0)
    item.tick(4)
    h.complete(episode)
    assert item.manager.task_plan.plan_version == 1
    assert len(item.goto.started_goals) == 1
    assert len(h.exits) == 1
    assert h.controller.episodes == {}


@pytest.mark.parametrize("change", ["step", "version"])
def test_change_during_agent_preflight_has_no_commit_or_old_fallback(harness, change):
    h = harness(config=FleetRecoveryConfig(enabled=True, mode="LOCAL_ONLY", max_local_attempts=1))
    episode = h.submit()
    item = h.locals[0]
    original = item.agent._safety.preflight
    def changed(candidate):
        result = original(candidate)
        if change == "step":
            item.manager._active_planned_step_id = "home"
        else:
            h.runtime.assignments.by_id(episode.assignment_id).local_plan_version = 99
        return result
    item.agent._safety.preflight = changed
    h.complete(episode)
    assert len(item.goto.started_goals) == 1
    assert item.manager.task_plan.plan_version == 1
    assert h.exits == []
    assert h.controller.episodes == {}


def test_wall_episode_deadline_exits_even_while_simulation_clock_and_http_are_frozen(harness):
    h = harness(blocked=True)
    episode = h.submit()
    item = h.locals[0]
    old_sim_time = item.clock.now()
    h.clock.value = episode.deadline_wall_s
    h.controller.tick()
    assert item.clock.now() == old_sim_time
    assert len(h.exits) == 1
    assert h.controller.episodes == {}
    assert h.broker.inflight_count == 1
    h.factory.release["uav_1"].set()
    h.complete(episode)
    assert len(h.exits) == 1
    assert item.manager.task_plan.plan_version == 1


def test_yaw_change_does_not_rebind_the_admitted_hold_reference(harness):
    from math import pi
    h = harness()
    episode = h.submit()
    anchor = episode.context.anchor
    item = h.locals[0]
    item.uav.set_pose(0, 0, 10, pi / 2)
    item.tick(4)
    h.complete(episode)
    assert episode.context.anchor is anchor
    assert anchor.pose.yaw_rad == 0
    assert item.manager.task_plan.plan_version == 2
    assert h.runtime._planned_routes["uav_1"][1] == (10, 0, 10)


def test_five_uavs_changed_detour_reaches_real_manager_commit():
    scenario = RealFleetScenario()
    third = scenario.items[2]
    def detour(data, payload):
        data["steps"].insert(0, {"id": "repair_detour", "uav_id": third.assignment.uav_id,
            "skill": "GOTO", "args": {"target": PointTarget(
                CoordinateFrame.WORLD_ENU, (2, 3, 10)).to_dict()}})
        return data
    scenario.factory.transform = detour
    try:
        episode = scenario.submit_fault()
        scenario.factory.release[third.assignment.uav_id].set()
        scenario.finish_request(episode)
        assert third.manager.task_plan.plan_version == 2, scenario.runtime._events
        assert third.manager.active_planned_step_id == "repair_detour"
        assert third.manager.task_plan.steps[0] == scenario.original_plans[third.assignment.uav_id].steps[0]
        assert len(third.takeoff.started_goals) == 1
        for item in scenario.items:
            if item is not third:
                assert item.manager.task_plan == scenario.original_plans[item.assignment.uav_id]
        assert scenario.runtime._planned_routes[third.assignment.uav_id][1] == (2, 3, 10)
    finally:
        scenario.close()


def test_time_domain_change_rejects_completed_candidate(harness):
    h = harness(config=FleetRecoveryConfig(enabled=True, mode="LOCAL_ONLY", max_local_attempts=1))
    episode = h.submit()
    item = h.locals[0]
    item.agent._local_repair_observation = replace(
        item.agent._local_repair_observation, time_domain="localization_epoch_2")
    h.complete(episode)
    assert item.manager.task_plan.plan_version == 1
    assert h.exits[0][1] == "REFERENCE_CHANGED"


@pytest.mark.parametrize("budget", [1, 2])
def test_stale_handoff_refreshes_snapshot_within_existing_budget(tmp_path, budget):
    source, spare = local("uav_a", -30), local("uav_b", 30)
    scenario = RealFleetScenario([source], spare=spare,
        boundary_builder=lambda s: production_boundary(s, tmp_path, spare=spare))
    scenario.controller.config = replace(scenario.controller.config,
        max_reassign_attempts=budget)
    try:
        scenario.factory.invalid = True
        episode = scenario.submit_fault()
        scenario.factory.release["uav_a"].set()
        scenario.finish_request(episode)
        scenario.runtime.tick()
        assert scenario.fleet_started.wait(2)
        first_request = episode.request_id
        # An independent owner publication advances the Fleet version while
        # this model is still computing. No source-local ownership changes.
        scenario.runtime._plan = replace(scenario.runtime.fleet_plan, fleet_plan_version=2)
        scenario.runtime._request = replace(scenario.runtime._request, fleet_plan_version=2)
        scenario.fleet_release.set()
        scenario.finish_request(episode)
        assert not scenario.created
        if budget == 1:
            assert not scenario.controller.episodes
            assert "uav_b" not in scenario.runtime.agents
        else:
            assert episode.phase == "REASSIGN_QUEUED"
            assert episode.reassign_attempts == 1
            scenario.wall.value += scenario.controller.config.retry_cooldown_s
            scenario.runtime.tick()
            assert episode.request_id != first_request
            scenario.finish_request(episode)
            assert scenario.runtime.fleet_plan.fleet_plan_version == 3, scenario.runtime._events
            assert scenario.runtime.agents["uav_b"] is spare.agent
            assert episode.reassign_attempts == 2
            assert len(scenario.created) == 1
            assert scenario.controller.episodes == {}
    finally:
        scenario.fleet_release.set()
        scenario.close()


def test_handoff_keeps_confirmed_navigation_and_transfers_only_independent_remainder(tmp_path):
    from collections import deque
    source, spare = local("uav_a", -30), local("uav_b", 30)
    second_goal = replace(source.goal, goal_id="goal_second",
                          spatial_constraint=NamedLocationTarget(source.home))
    def boundary(scenario):
        spec = replace(scenario.prepared.task_spec,
            goals=(source.goal, second_goal), ordering_constraints=(OrderingConstraint(
                "first_before_second", source.goal.goal_id, second_goal.goal_id, ConstraintStrength.MUST),))
        assignment = replace(scenario.prepared.fleet_plan_v2.assignments[0],
            goal_ids=(source.goal.goal_id, second_goal.goal_id))
        scenario.prepared.task_spec = spec
        scenario.prepared.fleet_request_v2 = replace(scenario.prepared.fleet_request_v2, task_spec=spec)
        scenario.prepared.fleet_plan_v2 = replace(scenario.prepared.fleet_plan_v2, assignments=(assignment,))
        return production_boundary(scenario, tmp_path, spare=spare)
    scenario = RealFleetScenario([source], spare=spare, boundary_builder=boundary)
    source.goto._outcomes = deque([succeeded(SkillResultCode.GOAL_REACHED),
                                   failed(SkillResultCode.TIMEOUT), *[running() for _ in range(20)]])
    try:
        scenario.factory.invalid = True
        for index in range(4):
            if index == 2:
                source.uav.set_pose(-30, 20, 10, 0)
            scenario.runtime.tick()
        assert scenario.factory.started["uav_a"].wait(2)
        episode = scenario.controller.episodes[source.assignment.assignment_id]
        scenario.factory.release["uav_a"].set()
        scenario.finish_request(episode)
        scenario.runtime.tick()
        assert scenario.fleet_started.wait(2), scenario.runtime._events
        assert episode.remaining_goal_ids == ("goal_second",)
        assert episode.external_dependencies[0].kind == "PREDECESSOR"
        scenario.fleet_release.set()
        scenario.finish_request(episode)
        assert scenario.runtime.fleet_plan.fleet_plan_version == 2, scenario.controller._last_error
        new_assignment = scenario.controller.assignments["assignment_handoff"]
        assert new_assignment.goal_ids == ("goal_second",)
        task = scenario.controller.assignment_task_specs["assignment_handoff"]
        assert task.goals[0].spatial_constraint == PointTarget(CoordinateFrame.WORLD_ENU, (-30, 0, 10))
        states, evidence = scenario.controller._goal_state()
        assert states[source.goal.goal_id] == "CONFIRMED"
        assert evidence[source.goal.goal_id]
        assert len(source.goto.started_goals) == 2  # original completed goal was never replayed
    finally:
        scenario.fleet_release.set()
        scenario.close()


def test_small_pose_change_reprojects_admitted_route_without_new_model_request(harness, monkeypatch):
    import fleet.recovery_controller as controller_module
    h = harness()
    item = h.locals[0]
    # Capture the candidate's original world_route inside the worker thread,
    # before the post-model pose change is applied.
    captured = {}
    original_validate = controller_module.validate_local_repair
    def spy(draft, context, world_context, **kwargs):
        candidate = original_validate(draft, context, world_context, **kwargs)
        captured["world_route"] = tuple(tuple(point) for point in candidate.world_route)
        return candidate
    monkeypatch.setattr(controller_module, "validate_local_repair", spy)
    episode = h.submit()
    # A deterministic small position change after model completion must be
    # reprojected by trusted code, not re-asked from the model. 0.2m stays
    # inside the hold tolerance yet exceeds the VALID reference tolerance.
    moved = (item.world.initial_uav_xyz_m[0] + 0.2, 0.0, 10.0)
    item.uav.set_pose(*moved, 0.0)
    item.tick(4.0)
    assert item.agent.local_repair_snapshot.stable_hold
    h.complete(episode)
    assert item.agent.snapshot().plan_version == 2
    assert h.exits == []
    assert len(h.factory.calls) == 1
    route = h.runtime._planned_routes[item.assignment.uav_id]
    admitted = captured["world_route"]
    assert len(route) == len(admitted)
    assert admitted[0] == pytest.approx((item.world.initial_uav_xyz_m[0], 0.0, 10.0))
    # Only the current access point changed; the admitted WORLD_ENU tail and
    # every later waypoint are preserved verbatim.
    assert route[0] == pytest.approx(moved)
    assert route[0] != pytest.approx(admitted[0])
    assert route[1:] == admitted[1:]
    for _ in range(2):
        h.controller.tick()
    assert len(h.factory.calls) == 1  # no second Qwen call after reproject


def test_pending_track_only_task_is_not_transferable_to_a_replacement(harness, monkeypatch):
    # pytest imports this file as uav_agent.tests.fleet.*; patch the running
    # module's globals so the Harness builder sees the track-only local.
    monkeypatch.setitem(globals(), "local", track_local)
    h = harness(blocked=True)
    item = h.locals[0]
    item.fail()  # TIMEOUT on the first transit GOTO; SEARCH/TRACK never start
    state = item.agent.local_repair_snapshot
    assert state.event is not None and state.stable_hold
    h.controller.tick()
    episode = h.controller.episodes[item.assignment.assignment_id]
    # The shared contract for this fault: one pending TRACK, zero confirmed.
    completed = tuple(step.step_id for step in state.task_plan.steps[:state.current_step_index])
    contract = build_remaining_task_contract((item.goal,), state.compiled_mission.planner_output, completed,
        h.controller._evidence(state), current_step_id=state.current_step_id,
        current_step_started=True, home_name=item.home, consumer="HANDOFF")
    assert contract.confirmed_goal_ids == ()
    assert [entry.status for entry in contract.goals] == ["PENDING"]
    assert [entry.transferability for entry in contract.pending_entries] == [Transferability.SAME_UAV_ONLY]
    # Escalate the same episode to the reassignment gate: a pending TRACK
    # with no completed goal must be refused even though nothing is confirmed.
    record = h.runtime.assignments.by_id(item.assignment.assignment_id)
    record.status = AssignmentStatus.WAITING_REASSIGNMENT
    episode.phase = "REASSIGN_QUEUED"
    episode.queue_signature = (record.local_plan_version, h.runtime.fleet_plan.fleet_plan_version)
    with pytest.raises(LocalRepairError) as error:
        h.controller._prepare_reassignment(episode, h.clock() + 10.0)
    assert error.value.code == "HANDOFF_EVIDENCE_UNSUPPORTED"


def test_unrelated_uav_version_change_does_not_discard_active_repair(harness):
    h = harness(count=2, blocked=True)
    episode = h.submit(0)
    # uav_2 shares no ordering, assignment or resource edge with uav_1: its
    # local version must not enter uav_1's dependency digest.
    h.runtime.assignments.by_id(h.locals[1].assignment.assignment_id).local_plan_version = 9
    h.factory.release["uav_1"].set()
    h.complete(episode)
    assert h.locals[0].agent.snapshot().plan_version == 2
    assert h.exits == []


def test_shared_channel_contention_lets_only_one_committer_through(harness):
    h = harness(count=2, blocked=True)
    h.shared_dependencies = (ExternalDependencySnapshot("channel", "SHARED_RESOURCE",
        (h.locals[0].goal.goal_id, h.locals[1].goal.goal_id), ("uav_1", "uav_2"), 1,
        "UNCHANGED", ("reservation_open",)),)
    first = h.submit(0)
    second = h.submit(1)
    h.factory.release["uav_1"].set()
    h.complete(first)
    assert h.locals[0].agent.snapshot().plan_version == 2
    # The winning publication flips the shared reservation; the loser's
    # compare-and-commit must observe the change and stop, not also commit.
    h.shared_dependencies = (ExternalDependencySnapshot("channel", "SHARED_RESOURCE",
        (h.locals[0].goal.goal_id, h.locals[1].goal.goal_id), ("uav_1", "uav_2"), 2,
        "UNCHANGED", ("reservation_uav_1",)),)
    h.factory.release["uav_2"].set()
    h.complete(second)
    assert h.locals[1].agent.snapshot().plan_version == 1
    assert ("assignment_uav_2", "COORDINATION_REQUIRED") in h.exits


# ---------------------------------------------------------------------------
# Bounded related-group joint repair
# ---------------------------------------------------------------------------

def joint_config():
    return FleetRecoveryConfig(enabled=True, mode="LOCAL_ONLY", joint_repair_enabled=True)


def healthy_peers(h, indexes):
    """Remove the injected TIMEOUT and advance peers onto their transit GOTO."""
    for index in indexes:
        item = h.locals[index]
        item.goto._outcomes.popleft()
        item.uav.set_pose(*item.world.initial_uav_xyz_m[:2], 10.0, 0.0)
        for ts in range(1, 4):
            item.tick(float(ts))
        assert item.agent.local_repair_snapshot.current_step_id == "goto"


def occupy_corridor(h, uav_id, waypoint):
    """Publish a committed route for a healthy UAV crossing uav_1's corridor."""
    h.runtime._planned_routes[uav_id] = (waypoint, (10.0, 0.0, 10.0))
    h.runtime._route_progress[uav_id] = 0


def test_corridor_conflict_routes_bounded_joint_repair_without_touching_outsiders(harness):
    h = harness(count=3, config=joint_config())
    healthy_peers(h, (1, 2))
    occupy_corridor(h, "uav_2", (30.0, 0.0, 10.0))
    outsiders_before = {item.assignment.uav_id: item.manager.task_plan.to_dict()
                        for item in h.locals[2:]}
    episode = h.submit(0)
    # The single-UAV candidate enters uav_2's occupied corridor and must be
    # rejected, then re-planned as a bounded {uav_1, uav_2} joint repair.
    h.complete(episode)
    assert episode.phase == "JOINT_QUEUED"
    scope_event = next(event for event in h.events if event["event"] == "RECOVERY_JOINT_SCOPE_SELECTED")
    assert scope_event["joint_scope"] == ["uav_1", "uav_2"]
    h.complete(episode)  # submit the joint request
    h.complete(episode)  # join, validate and coordinate-commit
    assert h.locals[0].agent.snapshot().plan_version == 2
    assert h.locals[1].agent.snapshot().plan_version == 2
    for item in h.locals[2:]:
        assert item.agent.snapshot().plan_version == 1
        assert item.manager.task_plan.to_dict() == outsiders_before[item.assignment.uav_id]
    assert h.exits == []
    assert episode.assignment_id not in h.controller.episodes
    assert h.runtime._planned_routes["uav_1"][1] == (10.0, 0.0, 10.0)
    assert h.runtime._planned_routes["uav_2"][1] == (40.0, 0.0, 10.0)
    assert [role for role, _ in h.factory.calls] == [ModelCallRole.RUNTIME_REPLAN] * 2
    assert h.runtime._pending_reassignments == set()


def test_local_ok_conflict_free_repair_stays_on_the_single_uav_path(harness):
    h = harness(count=3, config=joint_config())
    healthy_peers(h, (1, 2))
    episode = h.submit(0)
    h.complete(episode)
    assert h.locals[0].agent.snapshot().plan_version == 2
    assert h.exits == []
    assert len(h.factory.calls) == 1
    assert not any("JOINT" in event["event"] for event in h.events)


def test_joint_model_touching_a_readonly_uav_is_strictly_rejected(harness):
    h = harness(count=3, config=joint_config())
    healthy_peers(h, (1, 2))
    occupy_corridor(h, "uav_2", (30.0, 0.0, 10.0))
    episode = h.submit(0)
    h.complete(episode)
    assert episode.phase == "JOINT_QUEUED"

    def add_readonly_edit(data, payload):
        # The model tries to rewrite readonly uav_3 alongside the scope.
        return {**data, "uav_3": {**data["uav_1"]}}

    h.factory.transform = add_readonly_edit
    h.complete(episode)  # submit the joint request with the hostile output
    h.complete(episode)  # poll and reject
    assert h.locals[0].agent.snapshot().plan_version == 1
    assert h.locals[1].agent.snapshot().plan_version == 1
    assert h.exits == [(episode.assignment_id, "JOINT_SCOPE_MUTATED")]
    assert episode.assignment_id not in h.controller.episodes


def test_joint_candidates_colliding_as_a_combination_reject_the_whole_commit(harness):
    h = harness(count=3, config=joint_config())
    healthy_peers(h, (1, 2))
    occupy_corridor(h, "uav_2", (30.0, 0.0, 10.0))
    episode = h.submit(0)
    h.complete(episode)
    assert episode.phase == "JOINT_QUEUED"

    def collide(data, payload):
        # uav_1 accepts a detour waypoint that crosses uav_2's candidate
        # route: each candidate alone is fine, together they still collide.
        steps = [step for step in data["uav_1"]["steps"] if step["skill"] != "TAKEOFF"]
        detour = {"id": "detour", "uav_id": "uav_1", "skill": "GOTO",
                  "args": {"target": {"kind": "POINT", "frame": "WORLD_ENU",
                                      "xyz_m": [35.0, 0.0, 10.0]}}}
        data["uav_1"]["steps"] = [detour, *steps]
        return data

    h.factory.transform = collide
    h.complete(episode)  # submit the colliding joint candidates
    h.complete(episode)  # poll and reject
    assert h.locals[0].agent.snapshot().plan_version == 1
    assert h.locals[1].agent.snapshot().plan_version == 1
    assert h.exits == [(episode.assignment_id, "SHARED_SPACE_CONFLICT")]


def test_peer_plan_version_change_invalidates_the_whole_joint_candidate(harness):
    h = harness(count=3, config=joint_config())
    healthy_peers(h, (1, 2))
    occupy_corridor(h, "uav_2", (30.0, 0.0, 10.0))
    episode = h.submit(0)
    h.complete(episode)
    assert episode.phase == "JOINT_QUEUED"
    h.controller.tick()  # submit the joint request; worker completes
    # uav_2's local plan version moves while the joint candidate is in flight.
    h.runtime.assignments.by_id(h.locals[1].assignment.assignment_id).local_plan_version = 5
    h.complete(episode)
    assert h.locals[0].agent.snapshot().plan_version == 1
    assert h.locals[1].agent.snapshot().plan_version == 1
    assert h.exits and h.exits[0][1] == "STALE_VERSION"


def test_newly_occupied_shared_resource_rejects_the_joint_commit(harness):
    h = harness(count=3, config=joint_config())
    healthy_peers(h, (1, 2))
    h.shared_dependencies = (ExternalDependencySnapshot("channel", "SHARED_RESOURCE",
        (h.locals[0].goal.goal_id, h.locals[1].goal.goal_id), ("uav_1", "uav_2"), 1,
        "UNCHANGED", ("reservation_open",)),)
    occupy_corridor(h, "uav_2", (30.0, 0.0, 10.0))
    episode = h.submit(0)
    h.complete(episode)
    assert episode.phase == "JOINT_QUEUED"
    h.controller.tick()  # joint request submitted and computed
    # Another UAV takes the shared channel while the joint candidate waits.
    h.shared_dependencies = (ExternalDependencySnapshot("channel", "SHARED_RESOURCE",
        (h.locals[0].goal.goal_id, h.locals[1].goal.goal_id), ("uav_1", "uav_2"), 2,
        "UNCHANGED", ("reservation_other",)),)
    h.complete(episode)
    assert h.locals[0].agent.snapshot().plan_version == 1
    assert h.locals[1].agent.snapshot().plan_version == 1
    assert h.exits and h.exits[0][1] == "COORDINATION_REQUIRED"


def test_one_uav_validation_failure_rejects_the_whole_joint_submission(harness):
    h = harness(count=3, config=joint_config())
    healthy_peers(h, (1, 2))
    occupy_corridor(h, "uav_2", (30.0, 0.0, 10.0))
    episode = h.submit(0)
    h.complete(episode)
    assert episode.phase == "JOINT_QUEUED"

    def break_peer_draft(data, payload):
        # uav_2's suffix renames its LAND step: a protected-semantics attack
        # that only fails for one UAV, which must reject the whole submission.
        for step in data["uav_2"]["steps"]:
            if step["skill"] == "LAND":
                step["id"] = "renamed_land"
        return data

    h.factory.transform = break_peer_draft
    h.complete(episode)  # submit the broken joint candidates
    h.complete(episode)  # poll and reject
    assert h.locals[0].agent.snapshot().plan_version == 1
    assert h.locals[1].agent.snapshot().plan_version == 1
    assert h.exits and h.exits[0][1] in {"PROTECTED_STEP_MUTATION", "UNAUTHORIZED_NEW_EFFECT"}
    assert not any(event["event"] == "RECOVERY_COMMITTED" for event in h.events)


def test_oversized_conflict_scope_exits_instead_of_fleet_wide_replanning(harness):
    h = harness(count=3, config=FleetRecoveryConfig(enabled=True, mode="LOCAL_ONLY",
        joint_repair_enabled=True, max_joint_repair_scope_uavs=2))
    healthy_peers(h, (1, 2))
    # uav_2 and uav_3 both cross uav_1's corridor: a three-UAV conflict
    # exceeds the configured bound of two.
    occupy_corridor(h, "uav_2", (30.0, 0.0, 10.0))
    occupy_corridor(h, "uav_3", (60.0, 0.0, 10.0))
    episode = h.submit(0)
    h.complete(episode)
    # A four-UAV conflict exceeds the configured scope bound: safe exit, no
    # joint request and no fallback to a full-Fleet replan.
    assert h.exits == [(episode.assignment_id, "SHARED_SPACE_CONFLICT")]
    assert not any(event["event"].startswith("RECOVERY_JOINT")
                   and event["event"] != "RECOVERY_JOINT_SCOPE_REJECTED"
                   for event in h.events)
    assert len(h.factory.calls) == 1
    assert all(item.agent.snapshot().plan_version == 1 for item in h.locals)
    assert h.runtime._pending_reassignments == set()
