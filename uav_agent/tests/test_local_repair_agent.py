"""Local recovery uses real Skill lifecycle evidence and protected publication."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from agents.mission_agent import AgentStatus, MissionAgentError
from env.kinematic_uav import UAVState
from skills.hover import HoverSkill
from skills.manager import SkillManager, SkillManagerError, TaskStatus
from skills.plan import TaskPlan, TaskStep
from skills.types import SkillName, SkillResultCode, SkillStatus
from tests.test_skill_manager_hover import (
    _context, _observation, _goto_plan, _ScriptedSkill, _Outcome, _ok,
)
from tests.test_mission_agent import (
    failed, make_harness, succeeded, running,
)


class ManagerLocalRepairTest(unittest.TestCase):
    def manager(self, *, enabled=True, outcomes=None):
        context, clock = _context()
        goto = _ScriptedSkill(*(outcomes or [
            _Outcome(SkillStatus.FAILED, SkillResultCode.TIMEOUT, {"distance": 2.0}),
            _ok(SkillResultCode.GOAL_REACHED),
        ]))
        takeoff = _ScriptedSkill(_ok(SkillResultCode.TAKEOFF_COMPLETE))
        manager = SkillManager(context, registry={
            SkillName.TAKEOFF: takeoff, SkillName.GOTO: goto,
            SkillName.HOVER: HoverSkill(),
            SkillName.LAND: _ScriptedSkill(_ok(SkillResultCode.LAND_COMPLETE)),
        })
        manager.configure_local_repair(enabled=enabled)
        return manager, clock, goto, takeoff

    def fail_and_hold(self, manager, clock):
        clock.value = 1.0
        manager.tick(_observation(clock))
        event = manager.local_repair_event
        self.assertIsNotNone(event)
        self.assertFalse(manager.local_repair_stable_hold)
        # Even an explicit repeated manager frame cannot establish the hold.
        manager.tick(_observation(clock))
        self.assertFalse(manager.local_repair_stable_hold)
        clock.value = 2.0
        manager.tick(_observation(clock))
        self.assertTrue(manager.local_repair_stable_hold)
        return event

    def test_default_off_preserves_failure_and_landing(self):
        manager, clock, _, _ = self.manager(enabled=False)
        manager.start_task(_goto_plan())
        manager.tick(_observation(clock))
        self.assertIsNone(manager.local_repair_event)
        self.assertEqual(manager.pending_task_result, TaskStatus.FAILED)
        self.assertEqual(manager.active_name, SkillName.LAND)

    def test_failure_is_evidence_not_completed_output_and_commit_is_once(self):
        manager, clock, goto, _ = self.manager()
        manager.start_task(_goto_plan())
        event = self.fail_and_hold(manager, clock)
        self.assertEqual(manager.task_status, TaskStatus.RUNNING)
        self.assertIsNone(manager.pending_task_result)
        self.assertNotIn("goto", manager.step_outputs)
        self.assertEqual(event.result.code, SkillResultCode.TIMEOUT)
        self.assertEqual(manager.execution_reports[0].invocation_id, event.invocation_id)
        self.assertEqual(len(manager.started_invocations), 2)  # GOTO, HOVER
        manager.commit_local_repair(
            _goto_plan(version=2, x=3.0), expected_event_id=event.event_id,
            expected_plan_version=1,
        )
        self.assertEqual(manager.task_plan.plan_version, 2)
        self.assertEqual(manager.active_name, SkillName.GOTO)
        self.assertEqual(len(goto.started_goals), 2)
        self.assertIsNone(manager.local_repair_event)
        self.assertEqual(manager.execution_reports[-1].status, SkillStatus.CANCELED)
        with self.assertRaises(SkillManagerError):
            manager.commit_local_repair(
                _goto_plan(version=2), expected_event_id=event.event_id,
                expected_plan_version=1,
            )
        self.assertFalse(manager.fail_local_repair(expected_event_id=event.event_id, reason="late"))
        self.assertEqual(len(goto.started_goals), 2)
        clock.value = 3.0
        manager.tick(_observation(clock))
        self.assertEqual(manager.active_name, SkillName.LAND)

    def test_drift_and_measured_velocity_block_stable_hold(self):
        manager, clock, _, _ = self.manager()
        manager.start_task(_goto_plan())
        self.fail_and_hold(manager, clock)
        clock.value = 3.0
        manager.tick(replace(_observation(clock), uav_velocity=np.array([0.8, 0.0, 0.0])))
        self.assertFalse(manager.local_repair_stable_hold)
        clock.value = 4.0
        manager.tick(replace(_observation(clock), uav_pose=UAVState(2.0, 0.0, 5.0, 0.0)))
        self.assertFalse(manager.local_repair_stable_hold)

    def test_completed_takeoff_cannot_change_or_replay(self):
        manager, clock, goto, takeoff = self.manager()
        plan = TaskPlan.from_dicts([
            {"id": "takeoff", "skill": "TAKEOFF", "target_altitude": 5.0},
            {"id": "goto", "skill": "GOTO", "position": [2.0, 0.0, 5.0]},
            {"id": "land", "skill": "LAND"},
        ], mission_id="mission_hover", uav_id="uav_1")
        manager.start_task(plan)
        manager.tick(_observation(clock))
        event = self.fail_and_hold(manager, clock)
        malicious = replace(plan, plan_version=2, steps=(
            TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 7.0}), *plan.steps[1:],
        ))
        with self.assertRaisesRegex(SkillManagerError, "completed plan prefix"):
            manager.commit_local_repair(malicious, expected_event_id=event.event_id, expected_plan_version=1)
        manager.commit_local_repair(replace(plan, plan_version=2), expected_event_id=event.event_id, expected_plan_version=1)
        self.assertEqual(len(takeoff.started_goals), 1)
        self.assertEqual(len(goto.started_goals), 2)
        self.assertEqual(manager.current_step_index, 1)

    def test_current_identity_cannot_be_removed_and_guard_preserves_hold(self):
        manager, clock, _, _ = self.manager()
        manager.start_task(_goto_plan())
        event = self.fail_and_hold(manager, clock)
        changed = replace(_goto_plan(version=2), steps=(
            TaskStep("different", SkillName.GOTO, {"position": [2.0, 0.0, 5.0]}),
            TaskStep("land", SkillName.LAND, {}),
        ))
        with self.assertRaisesRegex(SkillManagerError, "current step identity"):
            manager.commit_local_repair(changed, expected_event_id=event.event_id, expected_plan_version=1)
        def expired():
            raise TimeoutError("episode expired after validation")
        with self.assertRaises(TimeoutError):
            manager.commit_local_repair(_goto_plan(version=2), expected_event_id=event.event_id, expected_plan_version=1, final_guard=expired)
        self.assertEqual(manager.task_plan.plan_version, 1)
        self.assertTrue(manager.local_repair_stable_hold)
        self.assertTrue(manager.fail_local_repair(expected_event_id=event.event_id, reason="budget"))
        self.assertEqual(manager.pending_task_result, TaskStatus.FAILED)
        self.assertEqual(manager.active_name, SkillName.LAND)

    def test_cancel_during_final_guard_cannot_reactivate(self):
        manager, clock, goto, _ = self.manager()
        manager.start_task(_goto_plan())
        event = self.fail_and_hold(manager, clock)
        with self.assertRaises(SkillManagerError):
            manager.commit_local_repair(_goto_plan(version=2), expected_event_id=event.event_id, expected_plan_version=1, final_guard=manager.cancel_task)
        self.assertEqual(manager.task_plan.plan_version, 1)
        self.assertEqual(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.pending_task_result, TaskStatus.CANCELED)
        self.assertEqual(len(goto.started_goals), 1)
        self.assertFalse(manager.fail_local_repair(expected_event_id=event.event_id, reason="stale"))

    def test_task_epoch_survives_reset_and_old_event_is_inert(self):
        manager, clock, _, _ = self.manager(outcomes=[
            _Outcome(SkillStatus.FAILED, SkillResultCode.TIMEOUT, {}),
            _Outcome(SkillStatus.FAILED, SkillResultCode.TIMEOUT, {}),
        ])
        manager.start_task(_goto_plan())
        old = self.fail_and_hold(manager, clock)
        manager.cancel_task()
        clock.value = 3.0
        manager.tick(_observation(clock))
        manager.reset_task()
        manager.start_task(_goto_plan())
        clock.value = 4.0
        manager.tick(_observation(clock))
        new = manager.local_repair_event
        self.assertGreater(new.execution_epoch, old.execution_epoch)
        self.assertNotEqual(new.event_id, old.event_id)
        self.assertFalse(manager.fail_local_repair(expected_event_id=old.event_id, reason="stale"))
        self.assertEqual(manager.local_repair_event.event_id, new.event_id)

    def test_invalid_goal_does_not_enter_generic_repair(self):
        manager, clock, _, _ = self.manager(outcomes=[
            _Outcome(SkillStatus.FAILED, SkillResultCode.INVALID_GOAL, {}),
        ])
        manager.start_task(_goto_plan())
        manager.tick(_observation(clock))
        self.assertIsNone(manager.local_repair_event)
        self.assertEqual(manager.active_name, SkillName.LAND)


class AgentLocalRepairTest(unittest.TestCase):
    def harness(self, *, failing_skill=SkillName.GOTO):
        h = make_harness(outcomes={
            failing_skill: [failed(SkillResultCode.TIMEOUT), running()],
        })
        h.manager.register(SkillName.HOVER, HoverSkill())
        h.agent.configure_local_repair(enabled=True)
        h.start()
        # Scripted flight does not move the actual UAV. Hover captures z=0;
        # supply that same measured pose to avoid faking stable hold evidence.
        for ts in range(1, 8):
            h.tick(float(ts), pose=UAVState(0.0, 0.0, 0.0, 0.0))
            if h.manager.local_repair_event is not None:
                h.tick(float(ts + 1), pose=UAVState(0.0, 0.0, 0.0, 0.0))
                break
        self.assertTrue(h.agent.local_repair_snapshot.stable_hold)
        return h

    def test_agent_manager_adopt_same_version_without_start_replay(self):
        h = self.harness()
        snapshot = h.agent.local_repair_snapshot
        before_safety = h.safety.evaluate_calls
        h.tick(float(h.clock.now() + 1), pose=UAVState(0.0, 0.0, 0.0, 0.0))
        self.assertGreater(h.safety.evaluate_calls, before_safety)
        h.agent.commit_local_repair(
            replace(snapshot.task_plan, plan_version=2),
            expected_event_id=snapshot.event.event_id, expected_plan_version=1,
        )
        self.assertEqual(h.agent.snapshot().plan_version, 2)
        self.assertEqual(h.manager.task_plan.plan_version, 2)
        self.assertEqual(h.manager.start_task_calls, 1)
        self.assertEqual(h.manager.active_planned_step_id, snapshot.current_step_id)
        self.assertEqual(h.manager.active_invocation.plan_version, 2)

    def test_search_resume_preserves_active_target_lifecycle(self):
        h = self.harness(failing_skill=SkillName.SEARCH)
        snapshot = h.agent.local_repair_snapshot
        target_before = h.target.snapshot()
        h.agent.commit_local_repair(replace(snapshot.task_plan, plan_version=2), expected_event_id=snapshot.event.event_id, expected_plan_version=1)
        h.tick(float(h.clock.now() + 1), pose=UAVState(0.0, 0.0, 0.0, 0.0))
        self.assertEqual(h.agent.snapshot().status, AgentStatus.RUNNING)
        self.assertEqual(h.target.snapshot().description, target_before.description)

    def test_cancel_from_preflight_blocks_publication(self):
        h = self.harness()
        snapshot = h.agent.local_repair_snapshot
        original_preflight = h.safety.preflight
        def canceling_preflight(candidate):
            decision = original_preflight(candidate)
            h.agent.cancel()
            return decision
        h.safety.preflight = canceling_preflight
        with self.assertRaises(MissionAgentError):
            h.agent.commit_local_repair(replace(snapshot.task_plan, plan_version=2), expected_event_id=snapshot.event.event_id, expected_plan_version=1)
        self.assertEqual(h.manager.task_plan.plan_version, 1)
        self.assertEqual(h.agent.snapshot().plan_version, 1)
        self.assertIsNone(h.manager.local_repair_event)
        self.assertFalse(h.agent.fail_local_repair(expected_event_id=snapshot.event.event_id, reason="late"))

    def test_final_guard_after_preflight_preserves_wait_when_rejected(self):
        h = self.harness()
        snapshot = h.agent.local_repair_snapshot
        before = h.safety.preflight_calls
        def expired():
            self.assertGreater(h.safety.preflight_calls, before)
            raise TimeoutError("wall-clock deadline")
        with self.assertRaises(TimeoutError):
            h.agent.commit_local_repair(replace(snapshot.task_plan, plan_version=2), expected_event_id=snapshot.event.event_id, expected_plan_version=1, final_guard=expired)
        self.assertTrue(h.agent.local_repair_snapshot.stable_hold)
        self.assertEqual(h.agent.snapshot().plan_version, 1)

    def test_start_hook_failure_keeps_published_versions_synchronized(self):
        h = self.harness()
        snapshot = h.agent.local_repair_snapshot
        def broken_start(goal, context):
            raise RuntimeError("actuator start failed")
        h.skills[SkillName.GOTO]._on_start = broken_start
        h.agent.commit_local_repair(replace(snapshot.task_plan, plan_version=2), expected_event_id=snapshot.event.event_id, expected_plan_version=1)
        self.assertEqual(h.manager.task_plan.plan_version, 2)
        self.assertEqual(h.agent.snapshot().plan_version, 2)
        self.assertEqual(h.manager.active_status, SkillStatus.FAILED)
        self.assertIsNone(h.manager.local_repair_event)
        h.tick(float(h.clock.now() + 1), pose=UAVState(0.0, 0.0, 0.0, 0.0))
        self.assertEqual(h.manager.active_name, SkillName.LAND)
        self.assertEqual(h.manager.pending_task_result, TaskStatus.FAILED)
        self.assertFalse(h.agent.fail_local_repair(expected_event_id=snapshot.event.event_id, reason="old retry"))

    def test_deterministic_reacquire_precedes_any_high_level_repair(self):
        h = make_harness(outcomes={SkillName.TRACK: [failed(
            SkillResultCode.TARGET_LOST, {
                "target_id": "target_0", "last_seen_position": (7.0, 8.0, 0.0),
                "last_seen_velocity": (0.5, 0.0, 0.0), "last_seen_time": 3.5,
                "tracking_duration": 2.5,
                "progress_schema": "track_progress.v1", "elapsed_s": 2.5,
                "valid_execution_s": 0.5, "continuous_execution_s": 0.0,
                "completion_basis": "valid_execution", "required_duration_s": 30.0,
            }), running()]})
        h.manager.register(SkillName.HOVER, HoverSkill())
        h.agent.configure_local_repair(enabled=True)
        h.start()
        for timestamp in (1.0, 2.0, 3.0, 4.0):
            h.tick(timestamp)
        self.assertEqual(h.manager.active_name, SkillName.REACQUIRE)
        self.assertIsNone(h.manager.local_repair_event)
        h.tick(5.0)
        self.assertEqual(h.manager.active_name, SkillName.TRACK)
        self.assertEqual(h.skills[SkillName.TRACK].started_goals[-1].track_duration, 29.5)
        self.assertIsNone(h.manager.local_repair_event)

    def test_partial_track_and_takeoff_timeouts_cannot_enter_generic_repair(self):
        for name in (SkillName.TAKEOFF, SkillName.TRACK):
            with self.subTest(skill=name):
                h = make_harness(outcomes={name: [failed(SkillResultCode.TIMEOUT)]})
                h.manager.register(SkillName.HOVER, HoverSkill())
                h.agent.configure_local_repair(enabled=True)
                h.start()
                for timestamp in range(1, 6):
                    h.tick(float(timestamp))
                    self.assertIsNone(h.manager.local_repair_event)
                    if h.manager.pending_task_result is TaskStatus.FAILED:
                        break
                self.assertEqual(h.manager.pending_task_result, TaskStatus.FAILED)
                self.assertEqual(h.manager.active_name, SkillName.LAND)

    def test_event_and_snapshot_data_are_owned(self):
        h = self.harness()
        first = h.agent.local_repair_snapshot
        first.event.result.data["forged"] = True
        first.completed_outputs["forged_step"] = {"target_id": "other"}
        first.latest_observation.camera_rgb[:] = 100
        next_snapshot = h.agent.local_repair_snapshot
        self.assertNotIn("forged", next_snapshot.event.result.data)
        self.assertNotIn("forged_step", next_snapshot.completed_outputs)
        self.assertTrue(np.all(next_snapshot.latest_observation.camera_rgb == 0))

    def test_graph_config_rejects_local_repair(self):
        h = make_harness(runtime_program="graph")
        h.manager.register(SkillName.HOVER, HoverSkill())
        with self.assertRaisesRegex(MissionAgentError, "linear"):
            h.agent.configure_local_repair(enabled=True)


if __name__ == "__main__":
    unittest.main()
