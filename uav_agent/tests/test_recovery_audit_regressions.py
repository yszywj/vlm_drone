"""Execution-level regressions from the asynchronous recovery code audit."""
from dataclasses import replace

import pytest

from fleet.task_spec import MissionGoal
from env.kinematic_uav import UAVState
from planner.goal_checker import GoalSatisfactionChecker
from planner.schemas_v3 import SkillPlanDraftV3
from runtime.plan_validator import PlanValidator
from skills.hover import HoverSkill
from skills.manager import SkillManager, TaskStatus
from skills.types import SkillName, SkillResultCode
from tests.fleet.test_local_repair import anchor, search_track_context, world
from tests.test_mission_agent import make_harness, succeeded, failed, running
from tests.test_skill_manager import (
    make_context, make_registry, standard_plan, tick_once,
    failed as manager_failed,
)


def test_track_success_does_not_disable_later_return_navigation_repair():
    h = make_harness(outcomes={SkillName.GOTO: [
        succeeded(SkillResultCode.GOAL_REACHED), failed(SkillResultCode.TIMEOUT), running()]})
    h.manager.register(SkillName.HOVER, HoverSkill())
    h.agent.configure_local_repair(enabled=True)
    h.start()
    for timestamp in (1.0, 2.0, 3.0, 4.0):
        h.tick(timestamp, pose=UAVState(0, 0, 0, 0))
    assert h.manager.pending_task_result is TaskStatus.SUCCEEDED
    assert h.manager.active_name is SkillName.GOTO
    h.tick(5.0, pose=UAVState(0, 0, 0, 0))
    assert h.manager.local_repair_event is not None
    assert h.manager.pending_task_result is None
    h.tick(6.0, pose=UAVState(0, 0, 0, 0))
    snapshot = h.agent.local_repair_snapshot
    assert snapshot.stable_hold
    completed = snapshot.task_plan.steps[:snapshot.current_step_index]
    h.agent.commit_local_repair(replace(snapshot.task_plan, plan_version=2),
        expected_event_id=snapshot.event.event_id, expected_plan_version=1)
    assert h.manager.task_plan.steps[:len(completed)] == completed
    assert h.manager.active_name is SkillName.GOTO
    assert len(h.skills[SkillName.TAKEOFF].started_goals) == 1
    assert len(h.skills[SkillName.TRACK].started_goals) == 1


@pytest.mark.parametrize("basis,ledger,remaining", [
    ("continuous", True, 5.0), ("valid_execution", False, 5.0),
    ("valid_execution", True, 3.0),
])
def test_reacquire_resumes_only_credible_progress_and_resets_continuous(basis, ledger, remaining):
    lost = {"target_id": "target_0", "last_seen_position": (1.0, 2.0, 0.5),
            "last_seen_velocity": (0.0, 0.0, 0.0), "last_seen_time": 3.5, "tracking_duration": 4.9}
    if ledger:
        lost.update(progress_schema="track_progress.v1", elapsed_s=4.9,
                    valid_execution_s=2.0, continuous_execution_s=0.0,
                    required_duration_s=5.0, completion_basis=basis)
    context, clock = make_context()
    scripted, registry = make_registry({SkillName.TRACK: [manager_failed(SkillResultCode.TARGET_LOST, lost)]})
    manager = SkillManager(context, registry=registry)
    plan = standard_plan()
    steps = tuple(replace(step, params={**step.params, "completion_basis": basis})
                  if step.skill is SkillName.TRACK else step for step in plan.steps)
    manager.start_task(replace(plan, steps=steps))
    for _ in range(5):
        tick_once(manager, clock)
    resumed = scripted[SkillName.TRACK].started_goals[-1]
    assert len(scripted[SkillName.TRACK].started_goals) == 2
    assert resumed.track_duration == remaining
    assert resumed.completion_basis == basis


def test_continuous_goal_survives_schema_compile_and_rejects_fragmented_plan():
    ctx = search_track_context()
    goal = replace(ctx.goals[0], completion_basis="continuous")
    assert MissionGoal.from_dict(goal.to_dict()) == goal
    payload = ctx.original.planner_output.to_dict()
    track = next(step for step in payload["steps"] if step["skill"] == "TRACK")
    track["args"]["completion_basis"] = "continuous"
    draft = SkillPlanDraftV3.from_dict(payload)
    compiled = PlanValidator().validate_and_compile(draft, world(), source="dynamic_scripted",
        spatial_resolver=anchor().resolver, mission_id=draft.mission_id,
        uav_id=draft.uav_id, plan_version=1)
    assert next(step for step in compiled.task_plan.steps if step.skill is SkillName.TRACK).params["completion_basis"] == "continuous"
    checker = GoalSatisfactionChecker()
    assert checker.check((goal,), draft, mission_id=draft.mission_id, uav_id=draft.uav_id).complete
    track["args"]["duration_s"] = 5
    index = payload["steps"].index(track)
    payload["steps"].insert(index + 1, {**track, "id": "track_second"})
    assert not checker.check((goal,), SkillPlanDraftV3.from_dict(payload),
                             mission_id=draft.mission_id, uav_id=draft.uav_id).complete
    payload["steps"].pop(index + 1)
    track["args"].update(duration_s=10, completion_basis="valid_execution")
    assert not checker.check((goal,), SkillPlanDraftV3.from_dict(payload),
                             mission_id=draft.mission_id, uav_id=draft.uav_id).complete
