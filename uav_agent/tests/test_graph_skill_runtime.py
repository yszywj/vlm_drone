from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from planner.mission_program import (
    MissionEdge,
    MissionNode,
    MissionProgram,
    MissionProgramError,
    ProgramAction,
    ProgramActionOp,
    ProgramEvent,
    ProgramEventHandler,
    linear_plan_to_mission_program,
)
from planner.program_patch import ProgramPatch
from skills.base import Skill
from skills.hover import HoverSkill
from skills.manager import SkillManager, SkillManagerError, TaskStatus
from skills.plan import TaskPlan, TaskStep
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillName,
    SkillResultCode,
    SkillStatus,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self) -> float:
        self.value += 1.0
        return self.value


class _Camera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))


@dataclass(frozen=True, slots=True)
class _Outcome:
    status: SkillStatus
    code: SkillResultCode
    data: dict[str, object] = field(default_factory=dict)


class _ScriptedSkill(Skill):
    goal_type = SkillGoal

    def __init__(self, *outcomes: _Outcome) -> None:
        super().__init__()
        self._outcomes = deque(outcomes)
        self.started_goals: list[SkillGoal] = []

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        del context
        self.started_goals.append(deepcopy(goal))

    def _on_tick(self, observation: Observation) -> None:
        del observation
        outcome = self._outcomes.popleft()
        if outcome.status is SkillStatus.SUCCEEDED:
            self._succeed(outcome.code, "scripted", outcome.data)
        else:
            self._fail(outcome.code, "scripted", outcome.data)


def _success(code: SkillResultCode, **data: object) -> _Outcome:
    return _Outcome(SkillStatus.SUCCEEDED, code, data)


def _failure(code: SkillResultCode, **data: object) -> _Outcome:
    return _Outcome(SkillStatus.FAILED, code, data)


def _context() -> tuple[SkillContext, _Clock]:
    clock = _Clock()
    return (
        SkillContext(
            uav=KinematicUAV(
                UAVState(0.0, 0.0, 0.0, 0.0),
                max_speed_mps=5.0,
                max_yaw_rate_rad_s=2.0,
            ),
            camera=_Camera(),
            perception=None,
            clock=clock,
            uav_id="uav_1",
        ),
        clock,
    )


def _observation(timestamp: float, *, altitude: float = 10.0) -> Observation:
    return Observation(
        uav_id="uav_1",
        timestamp=timestamp,
        uav_pose=UAVState(0.0, 0.0, altitude, 0.0),
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def _tick(manager: SkillManager, clock: _Clock, *, altitude: float = 10.0) -> TaskStatus:
    status = manager.tick(_observation(clock.advance(), altitude=altitude))
    assert isinstance(status, TaskStatus)
    return status


def _program(
    nodes: tuple[MissionNode, ...],
    edges: tuple[MissionEdge, ...],
    *,
    handlers: tuple[ProgramEventHandler, ...] = (),
) -> MissionProgram:
    return MissionProgram(
        mission_id="mission_graph",
        uav_id="uav_1",
        plan_version=1,
        entry_node_id=nodes[0].node_id,
        spatial_entities=(),
        nodes=nodes,
        edges=edges,
        event_handlers=handlers,
    )


class GraphSkillRuntimeTest(unittest.TestCase):
    def test_linear_adapter_preserves_consecutive_search_exhaustion_fallbacks(self) -> None:
        for result_code in (
            SkillResultCode.SEARCH_EXHAUSTED,
            SkillResultCode.TIMEOUT,
        ):
            with self.subTest(result_code=result_code.name):
                context, clock = _context()
                search = _ScriptedSkill(
                    _failure(result_code),
                    _success(
                        SkillResultCode.TARGET_FOUND,
                        target_id="target_1",
                    ),
                )
                manager = SkillManager(
                    context,
                    registry={
                        SkillName.SEARCH: search,
                        SkillName.LAND: _ScriptedSkill(
                            _success(SkillResultCode.LAND_COMPLETE)
                        ),
                    },
                )
                plan = TaskPlan(
                    (
                        TaskStep(
                            "search_a",
                            SkillName.SEARCH,
                            {
                                "center": (0.0, 0.0, 0.0),
                                "radius": 5.0,
                                "search_altitude": 10.0,
                                "target_description": "moving target",
                            },
                        ),
                        TaskStep(
                            "search_b",
                            SkillName.SEARCH,
                            {
                                "center": (10.0, 0.0, 0.0),
                                "radius": 5.0,
                                "search_altitude": 10.0,
                                "target_description": "moving target",
                            },
                        ),
                        TaskStep("land", SkillName.LAND, {}),
                    ),
                    mission_id="mission_search_fallback",
                    uav_id="uav_1",
                    plan_version=1,
                )
                program = linear_plan_to_mission_program(plan)

                manager.start_program(program)
                self.assertIs(_tick(manager, clock), TaskStatus.RUNNING)

                self.assertIs(manager.active_name, SkillName.SEARCH)
                self.assertEqual(manager.active_planned_step_id, "search_b")
                self.assertEqual(
                    manager.transition_log[-1].reason,
                    "program_event_timeout",
                )

    def test_graph_edge_really_selects_skill_manager_successor(self) -> None:
        context, clock = _context()
        takeoff = _ScriptedSkill(_success(SkillResultCode.TAKEOFF_COMPLETE))
        goto = _ScriptedSkill(_success(SkillResultCode.GOAL_REACHED))
        land = _ScriptedSkill(_success(SkillResultCode.LAND_COMPLETE))
        manager = SkillManager(
            context,
            registry={
                SkillName.TAKEOFF: takeoff,
                SkillName.GOTO: goto,
                SkillName.LAND: land,
            },
        )
        nodes = (
            MissionNode(
                "takeoff",
                TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 10.0}),
            ),
            MissionNode(
                "goto_skipped",
                TaskStep("goto_skipped", SkillName.GOTO, {"position": (1.0, 0.0, 10.0)}),
            ),
            MissionNode(
                "goto_selected",
                TaskStep("goto_selected", SkillName.GOTO, {"position": (9.0, 0.0, 10.0)}),
            ),
            MissionNode("land", TaskStep("land", SkillName.LAND, {})),
        )
        program = _program(
            nodes,
            (
                MissionEdge("takeoff", "goto_selected", ProgramEvent.SUCCESS),
                MissionEdge("takeoff", "goto_skipped", ProgramEvent.FAILURE),
                MissionEdge("goto_skipped", "goto_selected", ProgramEvent.SUCCESS),
                MissionEdge("goto_selected", "land", ProgramEvent.SUCCESS),
            ),
        )

        self.assertIs(manager.start_program(program), TaskStatus.RUNNING)
        self.assertTrue(manager.is_graph_runtime)
        self.assertIs(_tick(manager, clock), TaskStatus.RUNNING)
        self.assertEqual(manager.active_planned_step_id, "goto_selected")
        self.assertEqual(goto.started_goals[0].position, (9.0, 0.0, 10.0))
        self.assertIs(_tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(_tick(manager, clock), TaskStatus.SUCCEEDED)
        snapshot = manager.program_snapshot
        assert snapshot is not None
        self.assertTrue(snapshot.terminal)
        self.assertEqual(
            snapshot.completed_node_ids,
            ("takeoff", "goto_selected", "land"),
        )

    def test_target_confirmed_precedes_generic_success(self) -> None:
        context, clock = _context()
        search = _ScriptedSkill(
            _success(SkillResultCode.TARGET_FOUND, target_id="target_1")
        )
        track = _ScriptedSkill(_success(SkillResultCode.TRACK_COMPLETE))
        manager = SkillManager(
            context,
            registry={
                SkillName.TAKEOFF: _ScriptedSkill(
                    _success(SkillResultCode.TAKEOFF_COMPLETE)
                ),
                SkillName.SEARCH: search,
                SkillName.TRACK: track,
                SkillName.LAND: _ScriptedSkill(
                    _success(SkillResultCode.LAND_COMPLETE)
                ),
            },
        )
        nodes = (
            MissionNode("takeoff", TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 10.0})),
            MissionNode(
                "search",
                TaskStep(
                    "search",
                    SkillName.SEARCH,
                    {
                        "center": (0.0, 0.0, 0.0),
                        "radius": 5.0,
                        "search_altitude": 10.0,
                        "target_description": "moving target",
                    },
                ),
            ),
            MissionNode(
                "track",
                TaskStep(
                    "track",
                    SkillName.TRACK,
                    {"target_id": "$search.target_id", "track_duration": 2.0},
                ),
            ),
            MissionNode("land", TaskStep("land", SkillName.LAND, {})),
        )
        manager.start_program(
            _program(
                nodes,
                (
                    MissionEdge("takeoff", "search", ProgramEvent.SUCCESS),
                    MissionEdge("search", "track", ProgramEvent.TARGET_CONFIRMED),
                    MissionEdge("search", "land", ProgramEvent.SUCCESS),
                    MissionEdge("track", "land", ProgramEvent.SUCCESS),
                ),
            )
        )

        _tick(manager, clock)
        _tick(manager, clock)
        self.assertIs(manager.active_name, SkillName.TRACK)
        self.assertEqual(track.started_goals[0].target_id, "target_1")
        self.assertEqual(
            manager.transition_log[-1].reason,
            "program_event_target_confirmed",
        )

    def test_timeout_edge_can_drive_bounded_recovery_branch(self) -> None:
        context, clock = _context()
        goto = _ScriptedSkill(
            _failure(SkillResultCode.TIMEOUT),
            _success(SkillResultCode.GOAL_REACHED),
        )
        manager = SkillManager(
            context,
            registry={
                SkillName.TAKEOFF: _ScriptedSkill(
                    _success(SkillResultCode.TAKEOFF_COMPLETE)
                ),
                SkillName.GOTO: goto,
                SkillName.LAND: _ScriptedSkill(
                    _success(SkillResultCode.LAND_COMPLETE)
                ),
            },
        )
        nodes = (
            MissionNode("takeoff", TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 10.0})),
            MissionNode("goto_primary", TaskStep("goto_primary", SkillName.GOTO, {"position": (2.0, 0.0, 10.0)})),
            MissionNode("goto_recovery", TaskStep("goto_recovery", SkillName.GOTO, {"position": (0.0, 0.0, 10.0)})),
            MissionNode("land", TaskStep("land", SkillName.LAND, {})),
        )
        manager.start_program(
            _program(
                nodes,
                (
                    MissionEdge("takeoff", "goto_primary", ProgramEvent.SUCCESS),
                    MissionEdge("goto_primary", "goto_recovery", ProgramEvent.TIMEOUT),
                    MissionEdge("goto_primary", "land", ProgramEvent.FAILURE),
                    MissionEdge("goto_recovery", "land", ProgramEvent.SUCCESS),
                ),
            )
        )

        _tick(manager, clock)
        _tick(manager, clock)
        self.assertEqual(manager.active_planned_step_id, "goto_recovery")
        self.assertEqual(manager.transition_log[-1].reason, "program_event_timeout")
        _tick(manager, clock)
        self.assertIs(_tick(manager, clock), TaskStatus.SUCCEEDED)

    def test_unsafe_event_handlers_and_path_blocked_edges_fail_closed(self) -> None:
        nodes = (
            MissionNode("takeoff", TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 10.0})),
            MissionNode("land", TaskStep("land", SkillName.LAND, {})),
        )
        context, _ = _context()
        manager = SkillManager(
            context,
            registry={
                SkillName.TAKEOFF: _ScriptedSkill(
                    _success(SkillResultCode.TAKEOFF_COMPLETE)
                ),
                SkillName.LAND: _ScriptedSkill(
                    _success(SkillResultCode.LAND_COMPLETE)
                ),
            },
        )
        handler = ProgramEventHandler(
            ProgramEvent.PATH_BLOCKED,
            (ProgramAction(ProgramActionOp.HOLD),),
        )
        with self.assertRaisesRegex(SkillManagerError, "HOLD followed"):
            manager.start_program(
                _program(
                    nodes,
                    (MissionEdge("takeoff", "land", ProgramEvent.SUCCESS),),
                    handlers=(handler,),
                )
            )
        self.assertIs(manager.task_status, TaskStatus.IDLE)
        with self.assertRaisesRegex(SkillManagerError, "interruptible"):
            manager.start_program(
                _program(
                    nodes,
                    (
                        MissionEdge("takeoff", "land", ProgramEvent.SUCCESS),
                        MissionEdge("takeoff", "land", ProgramEvent.PATH_BLOCKED),
                    ),
                )
            )
        self.assertIs(manager.task_status, TaskStatus.IDLE)

    def test_program_patch_protects_completed_prefix_and_updates_current_suffix(self) -> None:
        context, clock = _context()
        goto = _ScriptedSkill(_success(SkillResultCode.GOAL_REACHED))
        manager = SkillManager(
            context,
            registry={
                SkillName.TAKEOFF: _ScriptedSkill(
                    _success(SkillResultCode.TAKEOFF_COMPLETE)
                ),
                SkillName.GOTO: goto,
                SkillName.HOVER: HoverSkill(),
                SkillName.LAND: _ScriptedSkill(
                    _success(SkillResultCode.LAND_COMPLETE)
                ),
            },
        )
        nodes = (
            MissionNode("takeoff", TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 10.0})),
            MissionNode("goto", TaskStep("goto", SkillName.GOTO, {"position": (2.0, 0.0, 10.0)})),
            MissionNode("land", TaskStep("land", SkillName.LAND, {})),
        )
        manager.start_program(
            _program(
                nodes,
                (
                    MissionEdge("takeoff", "goto", ProgramEvent.SUCCESS),
                    MissionEdge("goto", "land", ProgramEvent.SUCCESS),
                ),
            )
        )
        _tick(manager, clock)
        manager.interrupt_with_hover("PATH_BLOCKED")

        completed_patch = ProgramPatch(
            mission_id="mission_graph",
            uav_id="uav_1",
            base_plan_version=1,
            new_plan_version=2,
            replace_from_node_id="takeoff",
            replacement_nodes=(nodes[0],),
            replacement_edges=(),
            reason_codes=("PATH_BLOCKED",),
        )
        with self.assertRaisesRegex(MissionProgramError, "completed"):
            manager.replace_interrupted_program_suffix(completed_patch)

        replacement_goto = MissionNode(
            "goto",
            TaskStep("goto", SkillName.GOTO, {"position": (7.0, 1.0, 10.0)}),
        )
        replacement_land = MissionNode(
            "land_replanned",
            TaskStep("land_replanned", SkillName.LAND, {}),
        )
        patch = ProgramPatch(
            mission_id="mission_graph",
            uav_id="uav_1",
            base_plan_version=1,
            new_plan_version=2,
            replace_from_node_id="goto",
            replacement_nodes=(replacement_goto, replacement_land),
            replacement_edges=(
                MissionEdge("goto", "land_replanned", ProgramEvent.SUCCESS),
            ),
            reason_codes=("PATH_BLOCKED",),
        )
        self.assertIs(
            manager.replace_interrupted_program_suffix(patch),
            TaskStatus.RUNNING,
        )
        snapshot = manager.program_snapshot
        assert snapshot is not None
        self.assertEqual(snapshot.plan_version, 1)
        self.assertEqual(snapshot.completed_node_ids, ("takeoff",))
        assert manager.task_plan is not None
        self.assertEqual(manager.task_plan.plan_version, 1)
        self.assertEqual(manager.transition_log[-1].plan_version, 1)
        self.assertEqual(
            manager.graph_task_plan_for_adoption().plan_version,
            1,
        )
        self.assertIs(_tick(manager, clock, altitude=0.0), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertEqual(len(goto.started_goals), 2)
        self.assertEqual(goto.started_goals[-1].position, (7.0, 1.0, 10.0))
        snapshot = manager.program_snapshot
        assert snapshot is not None
        assert manager.task_plan is not None
        self.assertEqual(snapshot.plan_version, 2)
        self.assertEqual(manager.task_plan.plan_version, 2)
        self.assertEqual(manager.transition_log[-1].plan_version, 2)
        self.assertEqual(
            manager.graph_task_plan_for_adoption().plan_version,
            2,
        )

        _tick(manager, clock)
        self.assertEqual(manager.active_planned_step_id, "land_replanned")
        self.assertIs(_tick(manager, clock), TaskStatus.SUCCEEDED)

    def test_graph_mode_rejects_legacy_taskplan_suffix_replacement(self) -> None:
        context, clock = _context()
        manager = SkillManager(
            context,
            registry={
                SkillName.TAKEOFF: _ScriptedSkill(
                    _success(SkillResultCode.TAKEOFF_COMPLETE)
                ),
                SkillName.GOTO: _ScriptedSkill(
                    _success(SkillResultCode.GOAL_REACHED)
                ),
                SkillName.HOVER: HoverSkill(),
                SkillName.LAND: _ScriptedSkill(
                    _success(SkillResultCode.LAND_COMPLETE)
                ),
            },
        )
        nodes = (
            MissionNode("takeoff", TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 10.0})),
            MissionNode("goto", TaskStep("goto", SkillName.GOTO, {"position": (2.0, 0.0, 10.0)})),
            MissionNode("land", TaskStep("land", SkillName.LAND, {})),
        )
        manager.start_program(
            _program(
                nodes,
                (
                    MissionEdge("takeoff", "goto", ProgramEvent.SUCCESS),
                    MissionEdge("goto", "land", ProgramEvent.SUCCESS),
                ),
            )
        )
        _tick(manager, clock)
        manager.interrupt_with_hover("PATH_BLOCKED")
        replacement = TaskPlan(
            tuple(node.step for node in nodes),
            mission_id="mission_graph",
            uav_id="uav_1",
            plan_version=2,
        )
        with self.assertRaisesRegex(SkillManagerError, "ProgramPatch"):
            manager.replace_interrupted_step_and_suffix(replacement)


if __name__ == "__main__":
    unittest.main()
