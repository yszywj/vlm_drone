from __future__ import annotations

import unittest

from planner.mission_program import (
    MissionEdge,
    MissionNode,
    MissionProgramError,
    ProgramAction,
    ProgramActionOp,
    ProgramEvent,
    ProgramEventHandler,
    linear_plan_to_mission_program,
)
from planner.mission_program_schema import build_mission_program_json_schema
from planner.program_patch import ProgramPatch, apply_program_patch
from runtime.program_executor import ProgramExecutor
from skills.plan import TaskPlan, TaskStep
from skills.types import SkillName


def _plan() -> TaskPlan:
    return TaskPlan(
        (
            TaskStep("takeoff_1", SkillName.TAKEOFF, {"altitude": 10.0}),
            TaskStep("goto_1", SkillName.GOTO, {"position": (1.0, 0.0, 10.0)}),
            TaskStep("land_1", SkillName.LAND, {"ground_altitude": 0.0}),
        ),
        mission_id="mission_1",
        uav_id="uav_1",
        plan_version=1,
    )


class MissionProgramTest(unittest.TestCase):
    def test_linear_adapter_preserves_steps_and_routing(self) -> None:
        handler = ProgramEventHandler(
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
        program = linear_plan_to_mission_program(_plan(), event_handlers=(handler,))
        self.assertEqual(program.mission_id, "mission_1")
        self.assertEqual([node.node_id for node in program.nodes], ["takeoff_1", "goto_1", "land_1"])
        self.assertEqual([edge.on for edge in program.edges], [ProgramEvent.SUCCESS] * 2)
        self.assertEqual(program.event_handlers[0].to_dict()["actions"][1]["planner"], "QWEN_VL")

    def test_executor_advances_and_becomes_terminal(self) -> None:
        executor = ProgramExecutor(linear_plan_to_mission_program(_plan()))
        self.assertEqual(executor.current_step.step_id, "takeoff_1")
        self.assertEqual(executor.handle_event(ProgramEvent.SUCCESS).step_id, "goto_1")
        self.assertEqual(executor.handle_event("SUCCESS").step_id, "land_1")
        self.assertIsNone(executor.handle_event(ProgramEvent.SUCCESS))
        self.assertTrue(executor.snapshot().terminal)

    def test_patch_replaces_only_future_suffix_atomically(self) -> None:
        program = linear_plan_to_mission_program(_plan())
        replacement = (
            MissionNode("goto_1", TaskStep("goto_1", SkillName.GOTO, {"position": (2.0, 1.0, 10.0)})),
            MissionNode("land_2", TaskStep("land_2", SkillName.LAND, {"ground_altitude": 0.0})),
        )
        patch = ProgramPatch(
            mission_id="mission_1",
            uav_id="uav_1",
            base_plan_version=1,
            new_plan_version=2,
            replace_from_node_id="goto_1",
            replacement_nodes=replacement,
            replacement_edges=(MissionEdge("goto_1", "land_2", ProgramEvent.SUCCESS),),
            reason_codes=("PATH_BLOCKED",),
        )
        updated = apply_program_patch(
            program,
            patch,
            completed_node_ids=frozenset({"takeoff_1"}),
        )
        self.assertEqual(program.plan_version, 1)
        self.assertEqual(updated.plan_version, 2)
        self.assertEqual([node.node_id for node in updated.nodes], ["takeoff_1", "goto_1", "land_2"])
        self.assertIn(
            MissionEdge("takeoff_1", "goto_1", ProgramEvent.SUCCESS),
            updated.edges,
        )

    def test_patch_cannot_restate_or_modify_completed_prefix_edges(self) -> None:
        program = linear_plan_to_mission_program(_plan())
        patch = ProgramPatch(
            mission_id="mission_1",
            uav_id="uav_1",
            base_plan_version=1,
            new_plan_version=2,
            replace_from_node_id="goto_1",
            replacement_nodes=(
                MissionNode("goto_1", _plan().steps[1]),
                MissionNode("land_1", _plan().steps[2]),
            ),
            replacement_edges=(
                MissionEdge("takeoff_1", "land_1", ProgramEvent.FAILURE),
                MissionEdge("goto_1", "land_1", ProgramEvent.SUCCESS),
            ),
            reason_codes=("TEST",),
        )

        with self.assertRaisesRegex(MissionProgramError, "prefix control flow"):
            apply_program_patch(
                program,
                patch,
                completed_node_ids=frozenset({"takeoff_1"}),
            )

    def test_completed_node_cannot_be_replaced(self) -> None:
        program = linear_plan_to_mission_program(_plan())
        patch = ProgramPatch(
            mission_id="mission_1",
            uav_id="uav_1",
            base_plan_version=1,
            new_plan_version=2,
            replace_from_node_id="takeoff_1",
            replacement_nodes=(MissionNode("takeoff_1", _plan().steps[0]),),
            replacement_edges=(),
            reason_codes=("TEST",),
        )
        with self.assertRaisesRegex(MissionProgramError, "completed"):
            apply_program_patch(program, patch, completed_node_ids=frozenset({"takeoff_1"}))

    def test_patch_routing_or_version_mismatch_is_rejected(self) -> None:
        program = linear_plan_to_mission_program(_plan())
        patch = ProgramPatch(
            mission_id="mission_2",
            uav_id="uav_1",
            base_plan_version=1,
            new_plan_version=2,
            replace_from_node_id="goto_1",
            replacement_nodes=(MissionNode("goto_1", _plan().steps[1]),),
            replacement_edges=(),
            reason_codes=("TEST",),
        )
        with self.assertRaisesRegex(MissionProgramError, "routing"):
            apply_program_patch(program, patch)

    def test_schema_is_strict_and_bounded(self) -> None:
        schema = build_mission_program_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["nodes"]["maxItems"], 100)


if __name__ == "__main__":
    unittest.main()
