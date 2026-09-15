"""Recovery candidates remain inert until the final owner publication gate."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from configs.loader import load_config
from env.fleet_uav_search_env import FleetUavSearchEnv
from env.kinematic_uav import UAVState
from experiments.planning_audit_logger import PlanningAuditLogger
from fleet.airspace_manager import FleetPoseSnapshot, FleetUavPose
from fleet.runtime import (
    AssignmentStatus, FleetMissionRuntime, FleetReplanPublication,
    FleetRuntimeError, ReplannedAssignment,
)
from fleet.scripted_planner import ScriptedFleetPlanner
from fleet.target_registry import SharedTargetRegistry, TargetClaimError
from fleet.types import FleetUavCapability
from perception.factory import build_oracle_target_perception_runtime
from perception.mode import resolve_target_perception_mode
from scripts.run_fleet_mission import _build_runtime_fleet_replan_handler
from skills.types import Observation
from target import TargetManager, TargetSpec
from tests.fleet.test_compiler_v2 import _GoalDrivenSpatialPlanner
from tests.fleet.test_fleet_runtime import _FakeAgent, _FakeEnvironment, _request
from tests.fleet.test_runtime_replan_handler import _FleetReplanner, _prepared


ROOT = Path(__file__).resolve().parents[2]


class _PublicationEnvironment(_FakeEnvironment):
    def __init__(self):
        super().__init__()
        self.assignments = {}
        self.prepared_maps = []
        self.committed_maps = []
        self.fail_prepare = False
        self.conflict_after_publish = False

    def set_assignments(self, assignments):
        self.assignments = dict(assignments)

    def prepare_assignment_update(self, assignments):
        self.prepared_maps.append(dict(assignments))
        if self.fail_prepare:
            raise RuntimeError("injected environment prepare failure")
        return dict(assignments)

    def commit_assignment_update(self, assignments):
        self.committed_maps.append(dict(assignments))
        self.assignments = assignments

    def get_fleet_pose_snapshot(self):
        b_x = 0.5 if self.conflict_after_publish else 20.0
        return FleetPoseSnapshot(float(self.steps), {
            "uav_a": FleetUavPose("uav_a", (0.0, 0.0, 10.0), priority=0),
            "uav_b": FleetUavPose("uav_b", (b_x, 0.0, 10.0), priority=10),
            "uav_c": FleetUavPose("uav_c", (40.0, 0.0, 10.0), priority=100),
        })


class _LandingAgent(_FakeAgent):
    def __post_init__(self):
        super().__post_init__()
        self.landing_ticks = 0
        self.cancel_hook = None

    def cancel(self):
        self.cancels += 1
        self.status = "RUNNING"
        if self.cancel_hook is not None:
            self.cancel_hook()
        return self.snapshot()

    def tick(self, observation):
        self.ticks += 1
        if self.cancels:
            self.landing_ticks += 1
            self.status = "CANCELED" if self.landing_ticks >= 2 else "RUNNING"
        return self.snapshot()


def _runtime():
    request = _request()
    request = replace(request, uav_inventory=request.uav_inventory + (
        FleetUavCapability("uav_c", "standby", True, "home_c", 5.0, 30.0),
    ))
    env = _PublicationEnvironment()
    a = _LandingAgent("uav_a", terminal_status="RUNNING")
    b = _FakeAgent("uav_b", terminal_status="RUNNING")
    runtime = FleetMissionRuntime(
        env, ScriptedFleetPlanner(), {"uav_a": a, "uav_b": b},
        inventory=request.uav_inventory, target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
    )
    runtime.start(request.original_instruction)
    source = runtime.assignments.for_uav("uav_a").assignment
    runtime.assignments.update(source.assignment_id, AssignmentStatus.WAITING_REASSIGNMENT)
    runtime.targets.release(source.assignment_id, timestamp_s=1.0)
    return runtime, env, a, b, source


def _publication(runtime, source, metadata=None):
    candidate = _FakeAgent("uav_c", terminal_status="RUNNING", plan_version=2)
    assignment = replace(source, assignment_id=f"replacement_{source.uav_id}", uav_id="uav_c")
    item = ReplannedAssignment(
        assignment_id=source.assignment_id, replacement_assignment=assignment,
        agent=candidate, start_input=("continue assigned goal", None),
        metadata_updates=() if metadata is None else ((metadata, {"uav_c": candidate}),),
    )
    return FleetReplanPublication(1, 2, (item,)), candidate


def _published_state(runtime, env):
    return (
        runtime._plan, runtime.assignments.snapshot(), runtime.targets.snapshot(),
        dict(env.assignments), dict(runtime.agents), dict(runtime._agent_start_inputs),
    )


def test_candidate_agent_preparation_failure_preserves_active_metadata(tmp_path):
    prepared = _prepared()
    source = prepared.plan.assignments[0]
    record = SimpleNamespace(assignment=source, local_plan_version=1)
    belief = SimpleNamespace(fleet_plan_version=1, agents={
        "uav_a": SimpleNamespace(uav_id="uav_a", status="WAITING_REASSIGNMENT"),
    })
    original_plan = prepared.fleet_plan_v2
    original_compilations = dict(prepared.compilations)
    calls = []
    def fail_factory(*args):
        calls.append(args)
        raise RuntimeError("injected candidate construction failure")
    handler = _build_runtime_fleet_replan_handler(
        prepared, audit=PlanningAuditLogger(tmp_path), agent_factory=fail_factory,
        fleet_planner_factory=lambda client: _FleetReplanner(),
        local_planner_factory=lambda client, uav_id: _GoalDrivenSpatialPlanner(),
    )
    with pytest.raises(RuntimeError, match="candidate construction"):
        handler(record, belief)
    assert len(calls) == 1
    assert prepared.fleet_plan_v2 is original_plan
    assert prepared.compilations == original_compilations
    assert prepared.preparation_context["runtime_reassignments"] == []


def test_environment_prepare_failure_does_not_publish_or_cancel_source():
    runtime, env, old, _, source = _runtime()
    metadata = {"existing": "kept"}
    publication, candidate = _publication(runtime, source, metadata)
    before = _published_state(runtime, env)
    env.fail_prepare = True
    with pytest.raises(RuntimeError, match="environment prepare"):
        runtime._publish_fleet_replan(source.assignment_id, publication)
    assert _published_state(runtime, env) == before
    assert old.cancels == candidate.started == 0
    assert metadata == {"existing": "kept"}
    assert env.committed_maps == []


def test_target_bind_failure_stays_in_staging_registry(monkeypatch):
    runtime, env, old, _, source = _runtime()
    publication, candidate = _publication(runtime, source)
    before = _published_state(runtime, env)
    original = SharedTargetRegistry.bind_assignment
    def fail_after_bind(registry, **kwargs):
        result = original(registry, **kwargs)
        if kwargs["assignment_id"].startswith("replacement_"):
            raise TargetClaimError("injected candidate binding failure")
        return result
    monkeypatch.setattr(SharedTargetRegistry, "bind_assignment", fail_after_bind)
    with pytest.raises(TargetClaimError, match="candidate binding"):
        runtime._publish_fleet_replan(source.assignment_id, publication)
    assert _published_state(runtime, env) == before
    assert old.cancels == candidate.started == 0
    assert env.prepared_maps == []


def test_final_guard_expiry_keeps_metadata_and_authority_unpublished():
    runtime, env, old, _, source = _runtime()
    metadata = {"existing": "kept"}
    publication, candidate = _publication(runtime, source, metadata)
    before = _published_state(runtime, env)
    def expired():
        assert env.prepared_maps  # expiry is checked after fallible preparation
        assert metadata == {"existing": "kept"}
        raise TimeoutError("episode deadline")
    with pytest.raises(TimeoutError, match="episode deadline"):
        runtime._publish_fleet_replan(source.assignment_id, publication, final_guard=expired)
    assert _published_state(runtime, env) == before
    assert old.cancels == candidate.started == 0
    assert env.committed_maps == []


def test_guard_cancel_prevents_candidate_start_and_publication():
    runtime, env, _, _, source = _runtime()
    publication, candidate = _publication(runtime, source)
    plan = runtime._plan
    bindings = dict(env.assignments)
    def canceled():
        runtime.cancel()
    with pytest.raises(FleetRuntimeError, match="Fleet changed"):
        runtime._publish_fleet_replan(source.assignment_id, publication, final_guard=canceled)
    assert runtime._plan is plan
    assert env.assignments == bindings
    assert candidate.started == 0
    assert "uav_c" not in runtime.agents
    assert env.committed_maps == []


def test_second_guard_expiry_allows_only_irreversible_source_landing():
    runtime, env, old, _, source = _runtime()
    publication, candidate = _publication(runtime, source)
    before = _published_state(runtime, env)
    gates = []
    def expire_after_cancel():
        gates.append(old.cancels)
        if old.cancels:
            raise TimeoutError("deadline crossed by source cancel")
    with pytest.raises(TimeoutError, match="source cancel"):
        runtime._publish_fleet_replan(source.assignment_id, publication, final_guard=expire_after_cancel)
    assert gates == [0, 1]
    assert _published_state(runtime, env) == before
    assert old.cancels == 1 and candidate.started == 0
    assert "uav_a" in runtime._retired_failsafe_agents
    assert env.committed_maps == []


def test_reentrant_source_version_change_is_rejected_after_cancel():
    runtime, env, old, _, source = _runtime()
    publication, candidate = _publication(runtime, source)
    old.cancel_hook = lambda: runtime.assignments.update(
        source.assignment_id, AssignmentStatus.WAITING_REASSIGNMENT, local_plan_version=99)
    with pytest.raises(FleetRuntimeError, match="Fleet changed"):
        runtime._publish_fleet_replan(source.assignment_id, publication)
    assert runtime._plan.fleet_plan_version == 1
    assert runtime.assignments.by_id(source.assignment_id).local_plan_version == 99
    assert env.committed_maps == []
    assert candidate.started == 0


def test_two_ready_requests_cannot_publish_same_standby_uav():
    runtime, env, _, _, source_a = _runtime()
    source_b = runtime.assignments.for_uav("uav_b").assignment
    runtime.assignments.update(source_b.assignment_id, AssignmentStatus.WAITING_REASSIGNMENT)
    runtime.targets.release(source_b.assignment_id, timestamp_s=1.0)
    first, agent_first = _publication(runtime, source_a)
    second, agent_second = _publication(runtime, source_b)
    runtime._publish_fleet_replan(source_a.assignment_id, first)
    committed = _published_state(runtime, env)
    with pytest.raises(FleetRuntimeError, match="stale base version"):
        runtime._publish_fleet_replan(source_b.assignment_id, second)
    assert _published_state(runtime, env) == committed
    assert runtime.agents["uav_c"] is agent_first
    assert agent_second.started == 0
    assert len(env.committed_maps) == 1
    # Even an attempted rebase cannot steal the already assigned standby.
    rebased = FleetReplanPublication(2, 3, second.replacements)
    with pytest.raises(FleetRuntimeError, match="prior assignment routing"):
        runtime._publish_fleet_replan(source_b.assignment_id, rebased)
    assert _published_state(runtime, env) == committed


def test_metadata_publishes_only_after_guards_then_old_land_keeps_ticking():
    runtime, env, old, healthy, source = _runtime()
    metadata = {"existing": "kept"}
    publication, candidate = _publication(runtime, source, metadata)
    checks = []
    def observe_gate():
        checks.append(old.cancels)
        assert "uav_c" not in metadata
        assert "uav_c" not in env.assignments
        assert "uav_c" not in runtime.agents
        assert candidate.started == 0
    runtime._publish_fleet_replan(source.assignment_id, publication, final_guard=observe_gate)
    assert checks == [0, 1]
    assert metadata["uav_c"] is candidate
    assert env.assignments["uav_c"] == source.target_alias
    assert candidate.started == 0
    assert "uav_a" in runtime._retired_failsafe_agents
    # A low-priority retired LAND must advance even while airspace requests
    # HOLD. The replacement and unrelated healthy UAV continue ordinary ticks.
    env.conflict_after_publish = True
    runtime.tick()
    runtime.tick()
    assert old.landing_ticks == 2
    assert "uav_a" not in runtime._retired_failsafe_agents
    assert candidate.started == 1 and candidate.ticks >= 1
    assert healthy.ticks >= 1


def test_environment_staging_is_inert_and_invalid_maps_preserve_cache():
    config = load_config(ROOT / "configs/multi_uav_oracle.yaml")
    env = FleetUavSearchEnv(config, assignments={"uav_a": "target_i"})
    env.latest_evaluator_frames[("uav_a", "target_i")] = {"valid": "frame"}
    staged = env.prepare_assignment_update({"uav_b": "target_i"})
    assert dict(env.assignments) == {"uav_a": "target_i"}
    assert env.get_evaluator_frame("uav_a", "target_i") == {"valid": "frame"}
    with pytest.raises(ValueError, match="unknown target"):
        env.prepare_assignment_update({"uav_b": "target_unknown"})
    assert dict(env.assignments) == {"uav_a": "target_i"}
    assert env.latest_evaluator_frames
    env.commit_assignment_update(staged)
    assert dict(env.assignments) == {"uav_b": "target_i"}
    assert env.latest_evaluator_frames == {}
    with pytest.raises(PermissionError, match="restricted"):
        env.get_evaluator_frame("uav_a", "target_i")


def test_candidate_oracle_provider_cannot_read_before_live_binding():
    config = load_config(ROOT / "configs/multi_uav_oracle.yaml")
    env = FleetUavSearchEnv(config, assignments={"uav_a": "target_i"})
    mode = resolve_target_perception_mode("oracle", acknowledge_privileged_oracle=True)
    provider = build_oracle_target_perception_runtime(
        config, resolved_mode=mode, environment=env, uav_id="uav_b",
        candidate_target_alias="target_i",
    )
    provider.reset(mission_id="mission_candidate", assignment_id="assignment_candidate",
                   uav_id="uav_b", target_alias="target_i", target_spec=TargetSpec("red cube"))
    base = Observation(
        uav_id="uav_b", timestamp=1.0, uav_pose=UAVState(0.0, 0.0, 5.0, 0.0),
        uav_velocity=np.zeros(3), camera_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    target_manager = TargetManager()
    with pytest.raises(PermissionError, match="restricted"):
        provider.observe(base_observation=base, camera_sample=None, target_manager=target_manager)
    assert dict(env.assignments) == {"uav_a": "target_i"}
    prepared = env.prepare_assignment_update({"uav_b": "target_i"})
    with pytest.raises(PermissionError, match="restricted"):
        provider.observe(base_observation=base, camera_sample=None, target_manager=target_manager)
    env.commit_assignment_update(prepared)
    # Now routing is authorized, but a new synchronized frame is still needed.
    with pytest.raises(RuntimeError, match="no synchronized evaluator frame"):
        provider.observe(base_observation=base, camera_sample=None, target_manager=target_manager)


def test_candidate_context_does_not_change_oracle_frame_authority():
    from perception.oracle import OraclePerception
    from perception.runtime import GuardedPerceptionBackend, PerceptionRuntimeProfile
    from tests.test_skill_manager_hover import _context

    config = load_config(ROOT / "configs/multi_uav_oracle.yaml")
    env = FleetUavSearchEnv(config, assignments={"uav_a": "target_i"})
    context, clock = _context()
    env.uav_controllers["uav_b"] = context.uav
    env.camera_sensors["uav_b"] = context.camera
    guarded = GuardedPerceptionBackend(
        OraclePerception(uav_id="uav_b", target_id="target_i"),
        profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
        acknowledge_privileged_oracle=True,
    )
    with pytest.raises(PermissionError, match="assigned target"):
        env.make_skill_context("uav_b", clock, perception=guarded)
    candidate_context = env.make_skill_context(
        "uav_b", clock, perception=guarded, candidate_target_id="target_i")
    candidate_context.validate()
    assert dict(env.assignments) == {"uav_a": "target_i"}
    with pytest.raises(PermissionError, match="restricted"):
        env.get_evaluator_frame("uav_b", "target_i")
    with pytest.raises(PermissionError, match="assigned target"):
        env.make_skill_context("uav_b", clock, perception=guarded, candidate_target_id="target_j")


def test_recovery_geometry_omits_target_truth_and_tracks_obstacle_change():
    from common.obstacle_types import ObstacleSpec
    from env.obstacle_registry import ObstacleRegistry
    from env.fleet_uav_search_env import FleetPoseSnapshot as EnvironmentPoseSnapshot

    config = load_config(ROOT / "configs/multi_uav_oracle.yaml")
    env = FleetUavSearchEnv(config, assignments={"uav_a": "target_i"})
    obstacle = ObstacleSpec("wall", (5.0, 5.0, 5.0), (1.0, 8.0, 10.0), (0.2, 0.3, 0.4))
    env.scene = SimpleNamespace(obstacle_registry=ObstacleRegistry((obstacle,)))
    env._fleet_pose_snapshot = EnvironmentPoseSnapshot(
        1, 1.0, {"uav_a": UAVState(0.0, 0.0, 5.0, 0.0)},
        {"uav_a": np.zeros(3)}, {"target_i": {"privileged_target_truth": 99}},
    )
    first = env.recovery_geometry_snapshot()
    assert "target_states" not in first
    assert not hasattr(first["fleet_pose_snapshot"], "target_states")
    assert "privileged_target_truth" not in repr(first)
    env.scene.obstacle_registry = ObstacleRegistry((replace(obstacle, center_xyz_m=(8.0, 5.0, 5.0)),))
    second = env.recovery_geometry_snapshot()
    assert second["map_version"] != first["map_version"]
    assert first["obstacles"][0].center_xyz_m == (5.0, 5.0, 5.0)
