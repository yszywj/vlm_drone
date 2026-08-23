from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import json
import unittest

import numpy as np

from agents.program_patch_coordinator import (
    ProgramPatchCoordinator,
    ProgramPatchCoordinatorState,
)
from env.kinematic_uav import KinematicUAV, UAVState
from models import AsyncModelRequest, AsyncModelResult, ModelResponse
from planner.mission_program import (
    MissionEdge,
    MissionNode,
    MissionProgram,
    ProgramAction,
    ProgramActionOp,
    ProgramEvent,
    ProgramEventHandler,
)
from planner.program_patch import ProgramPatch
from planner.program_patch_planner import QwenProgramPatchPlanner
from skills.base import Skill
from skills.hover import HoverSkill
from skills.manager import SkillManager, TaskStatus
from skills.plan import TaskStep
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
    data: dict[str, object]


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


class _Worker:
    def __init__(self) -> None:
        self.uav_id = "uav_1"
        self.submitted: AsyncModelRequest | None = None
        self.result: AsyncModelResult | None = None

    def submit(self, request: object) -> None:
        assert isinstance(request, AsyncModelRequest)
        self.submitted = request

    def complete(self, payload: dict[str, object], *, stale: bool = False) -> None:
        assert self.submitted is not None
        request = self.submitted
        self.result = AsyncModelResult(
            request.request_id,
            request.review_id,
            request.mission_id,
            request.uav_id,
            request.plan_version,
            request.observation_timestamp_s,
            request.frame_id,
            ModelResponse(json.dumps(payload), "fake-qwen", "stop", {}),
            None,
            None,
            stale=stale,
        )

    def poll(self, **kwargs: object) -> AsyncModelResult | None:
        del kwargs
        result, self.result = self.result, None
        return result


def _success(code: SkillResultCode, **data: object) -> _Outcome:
    return _Outcome(SkillStatus.SUCCEEDED, code, data)


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


def _observation(timestamp: float, altitude: float = 0.0) -> Observation:
    return Observation(
        uav_id="uav_1",
        timestamp=timestamp,
        uav_pose=UAVState(0.0, 0.0, altitude, 0.0),
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def _tick(manager: SkillManager, clock: _Clock, altitude: float = 0.0) -> TaskStatus:
    result = manager.tick(_observation(clock.advance(), altitude))
    assert isinstance(result, TaskStatus)
    return result


def _nodes() -> tuple[MissionNode, ...]:
    return (
        MissionNode(
            "takeoff",
            TaskStep("takeoff", SkillName.TAKEOFF, {"target_altitude": 10.0}),
        ),
        MissionNode(
            "goto",
            TaskStep("goto", SkillName.GOTO, {"position": (2.0, 0.0, 10.0)}),
        ),
        MissionNode("land", TaskStep("land", SkillName.LAND, {})),
    )


def _qwen_handler() -> ProgramEventHandler:
    return ProgramEventHandler(
        ProgramEvent.PATH_BLOCKED,
        (
            ProgramAction(ProgramActionOp.HOLD),
            ProgramAction(
                ProgramActionOp.REPLAN_CURRENT_ROUTE,
                planner="QWEN_VL",
                allow_model_waypoints=True,
            ),
        ),
    )


def _program(*, handler: bool = True) -> MissionProgram:
    nodes = _nodes()
    return MissionProgram(
        mission_id="mission_graph",
        uav_id="uav_1",
        plan_version=1,
        entry_node_id="takeoff",
        spatial_entities=(),
        nodes=nodes,
        edges=(
            MissionEdge("takeoff", "goto", ProgramEvent.SUCCESS),
            MissionEdge("goto", "land", ProgramEvent.SUCCESS),
        ),
        event_handlers=(_qwen_handler(),) if handler else (),
    )


def _manager() -> tuple[SkillManager, _Clock, _ScriptedSkill]:
    context, clock = _context()
    goto = _ScriptedSkill(
        _success(SkillResultCode.GOAL_REACHED),
        _success(SkillResultCode.GOAL_REACHED),
    )
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
    return manager, clock, goto


def _valid_patch() -> ProgramPatch:
    replacement_goto = MissionNode(
        "goto",
        TaskStep("goto", SkillName.GOTO, {"position": (7.0, 1.0, 10.0)}),
    )
    replacement_land = MissionNode(
        "land_replanned", TaskStep("land_replanned", SkillName.LAND, {})
    )
    return ProgramPatch(
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


class ProgramPatchRuntimeTest(unittest.TestCase):
    def test_qwen_patch_is_staged_then_atomically_published_after_hover(self) -> None:
        manager, clock, goto = _manager()
        manager.start_program(_program())
        _tick(manager, clock, altitude=10.0)
        self.assertEqual(manager.active_planned_step_id, "goto")

        worker = _Worker()
        coordinator = ProgramPatchCoordinator(
            uav_id="uav_1",
            planner=QwenProgramPatchPlanner(),
            worker=worker,
            skill_manager=manager,
            request_timeout_s=10.0,
        )
        started = coordinator.begin(
            expected_plan_version=1,
            observation_timestamp_s=clock.value,
            frame_id="frame_blocked",
        )
        self.assertIs(started.state, ProgramPatchCoordinatorState.AWAITING_MODEL)
        self.assertIs(manager.active_name, SkillName.HOVER)
        assert worker.submitted is not None
        schema = worker.submitted.options.response_format.to_dict()["schema"]  # type: ignore[union-attr]
        self.assertEqual(schema["properties"]["new_plan_version"]["const"], 2)  # type: ignore[index]

        worker.complete(_valid_patch().to_dict())
        accepted = coordinator.tick(timestamp_s=clock.value + 0.1)
        self.assertIs(accepted.state, ProgramPatchCoordinatorState.ACCEPTED)
        self.assertEqual(manager.program_snapshot.plan_version, 1)  # type: ignore[union-attr]
        self.assertEqual(manager.task_plan.plan_version, 1)  # type: ignore[union-attr]

        self.assertIs(_tick(manager, clock), TaskStatus.RUNNING)
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertEqual(manager.program_snapshot.plan_version, 2)  # type: ignore[union-attr]
        self.assertEqual(manager.task_plan.plan_version, 2)  # type: ignore[union-attr]
        self.assertEqual(goto.started_goals[-1].position, (7.0, 1.0, 10.0))

    def test_stale_model_result_never_resumes_and_cancels_to_land(self) -> None:
        manager, clock, _ = _manager()
        manager.start_program(_program())
        _tick(manager, clock, altitude=10.0)
        worker = _Worker()
        coordinator = ProgramPatchCoordinator(
            uav_id="uav_1",
            planner=QwenProgramPatchPlanner(),
            worker=worker,
            skill_manager=manager,
        )
        coordinator.begin(
            expected_plan_version=1,
            observation_timestamp_s=clock.value,
            frame_id="frame_blocked",
        )
        worker.complete(_valid_patch().to_dict(), stale=True)
        result = coordinator.tick(timestamp_s=clock.value + 0.1)
        self.assertIs(result.state, ProgramPatchCoordinatorState.FAILED)
        self.assertEqual(result.error_code, "PROGRAM_PATCH_REJECTED")
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertIs(manager.pending_task_result, TaskStatus.CANCELED)
        self.assertEqual(manager.program_snapshot.plan_version, 1)  # type: ignore[union-attr]

    def test_patch_without_causal_event_reason_fails_closed(self) -> None:
        manager, clock, _ = _manager()
        manager.start_program(_program())
        _tick(manager, clock, altitude=10.0)
        worker = _Worker()
        coordinator = ProgramPatchCoordinator(
            uav_id="uav_1",
            planner=QwenProgramPatchPlanner(),
            worker=worker,
            skill_manager=manager,
        )
        coordinator.begin(
            expected_plan_version=1,
            observation_timestamp_s=clock.value,
            frame_id="frame_blocked",
        )
        invalid = _valid_patch().to_dict()
        invalid["reason_codes"] = ["VISIBLE_OBSTACLE"]
        worker.complete(invalid)

        result = coordinator.tick(timestamp_s=clock.value + 0.1)

        self.assertIs(result.state, ProgramPatchCoordinatorState.FAILED)
        self.assertEqual(result.error_code, "PROGRAM_PATCH_REJECTED")
        self.assertIs(manager.active_name, SkillName.LAND)
        self.assertEqual(manager.program_snapshot.plan_version, 1)  # type: ignore[union-attr]

    def test_static_path_blocked_edge_commits_only_after_hover(self) -> None:
        manager, clock, _ = _manager()
        base = _nodes()
        detour = MissionNode(
            "goto_detour",
            TaskStep(
                "goto_detour",
                SkillName.GOTO,
                {"position": (0.0, 4.0, 10.0)},
            ),
        )
        program = MissionProgram(
            mission_id="mission_graph",
            uav_id="uav_1",
            plan_version=1,
            entry_node_id="takeoff",
            spatial_entities=(),
            nodes=(base[0], base[1], detour, base[2]),
            edges=(
                MissionEdge("takeoff", "goto", ProgramEvent.SUCCESS),
                MissionEdge("goto", "land", ProgramEvent.SUCCESS),
                MissionEdge("goto", "goto_detour", ProgramEvent.PATH_BLOCKED),
                MissionEdge("goto_detour", "land", ProgramEvent.SUCCESS),
            ),
        )
        manager.start_program(program)
        _tick(manager, clock, altitude=10.0)
        dispatch = manager.dispatch_program_event(
            ProgramEvent.PATH_BLOCKED,
            expected_plan_version=1,
        )
        self.assertEqual(dispatch.target_node_id, "goto_detour")
        self.assertEqual(manager.program_snapshot.current_node_id, "goto")  # type: ignore[union-attr]
        self.assertIs(manager.active_name, SkillName.HOVER)

        _tick(manager, clock)
        self.assertEqual(manager.active_planned_step_id, "goto_detour")
        self.assertEqual(manager.program_snapshot.current_node_id, "goto_detour")  # type: ignore[union-attr]
        self.assertEqual(
            manager.program_snapshot.completed_node_ids,  # type: ignore[union-attr]
            ("takeoff", "goto"),
        )

    def test_stale_external_event_is_rejected_before_hover(self) -> None:
        manager, clock, _ = _manager()
        manager.start_program(_program())
        _tick(manager, clock, altitude=10.0)
        with self.assertRaisesRegex(Exception, "stale"):
            manager.dispatch_program_event(
                ProgramEvent.PATH_BLOCKED,
                expected_plan_version=2,
            )
        self.assertIs(manager.active_name, SkillName.GOTO)
        self.assertFalse(manager.is_supervisory_paused)

    def test_program_patch_round_trip_does_not_rewrite_output(self) -> None:
        patch = _valid_patch()
        encoded = patch.to_dict()
        self.assertEqual(ProgramPatch.from_dict(encoded).to_dict(), encoded)
        invalid = patch.to_dict()
        invalid["replacement_nodes"][0]["step"]["extra"] = True  # type: ignore[index]
        with self.assertRaisesRegex(Exception, "unknown fields"):
            ProgramPatch.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
