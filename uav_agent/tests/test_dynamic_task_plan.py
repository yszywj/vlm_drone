from __future__ import annotations

import unittest

from skills.plan import (
    RecoveryPolicy,
    StepOutputRef,
    TaskPlan,
    TaskPlanError,
    TaskStep,
)
from skills.motion_types import MotionPolicy, YawMode
from skills.types import SkillName


class DynamicTaskPlanTests(unittest.TestCase):
    def test_legacy_entries_receive_deterministic_ids(self) -> None:
        plan = TaskPlan.from_dicts(
            [
                {"skill": "TAKEOFF", "target_altitude": 8.0},
                {"skill": "GOTO", "position": [1.0, 2.0, 8.0]},
                {"skill": "LAND"},
            ]
        )
        self.assertEqual(
            [step.step_id for step in plan.steps],
            ["step_01", "step_02", "step_03"],
        )
        self.assertEqual(
            [entry["id"] for entry in plan.to_dicts()],
            ["step_01", "step_02", "step_03"],
        )

    def test_generic_plan_has_no_fixed_sequence_constraint(self) -> None:
        plan = TaskPlan.from_dicts(
            [
                {"id": "takeoff", "skill": "TAKEOFF", "target_altitude": 8.0},
                {"id": "goto_a", "skill": "GOTO", "position": [1, 2, 8]},
                {"id": "goto_b", "skill": "GOTO", "position": [3, 4, 8]},
                {"id": "land", "skill": "LAND"},
            ]
        )
        self.assertEqual(
            [step.skill for step in plan.steps],
            [SkillName.TAKEOFF, SkillName.GOTO, SkillName.GOTO, SkillName.LAND],
        )

    def test_step_output_reference_and_recovery_round_trip(self) -> None:
        recovery = RecoveryPolicy(
            SkillName.REACQUIRE,
            max_attempts=2,
            search_radius_m=9.0,
            timeout_s=20.0,
        )
        step = TaskStep(
            "track_1",
            SkillName.TRACK,
            {
                "target_id": StepOutputRef("search_1"),
                "track_duration": 12.0,
            },
            recovery,
        )
        encoded = step.to_dict()
        self.assertEqual(encoded["target_id"], "$search_1.target_id")
        self.assertEqual(encoded["recovery"]["skill"], "REACQUIRE")
        decoded = TaskPlan.from_dicts(
            [
                {"id": "search_1", "skill": "SEARCH"},
                encoded,
            ]
        )
        self.assertEqual(
            decoded.steps[1].params["target_id"],
            StepOutputRef("search_1"),
        )
        self.assertEqual(decoded.steps[1].recovery, recovery)

    def test_params_are_defensively_copied(self) -> None:
        params = {"position": [1.0, 2.0, 3.0]}
        step = TaskStep("goto_1", SkillName.GOTO, params)
        params["position"][0] = 99.0
        self.assertEqual(step.params["position"], [1.0, 2.0, 3.0])
        encoded = step.to_dict()
        encoded["position"][1] = 88.0
        self.assertEqual(step.params["position"], [1.0, 2.0, 3.0])

    def test_motion_policy_serializes_with_stable_enum_name(self) -> None:
        plan = TaskPlan(
            (
                TaskStep(
                    "goto_1",
                    SkillName.GOTO,
                    {
                        "position": (1.0, 2.0, 3.0),
                        "motion_policy": MotionPolicy(
                            max_speed=2.0,
                            yaw_mode=YawMode.COURSE_ALIGNED,
                        ),
                    },
                ),
            )
        )
        encoded = plan.to_dicts()
        self.assertEqual(
            encoded[0]["motion_policy"]["yaw_mode"],
            "COURSE_ALIGNED",
        )
        self.assertEqual(TaskPlan.from_dicts(encoded).to_dicts(), encoded)

    def test_structural_rejections(self) -> None:
        with self.assertRaisesRegex(TaskPlanError, "non-empty"):
            TaskPlan(())
        with self.assertRaisesRegex(TaskPlanError, "unique"):
            TaskPlan(
                (
                    TaskStep("same", SkillName.TAKEOFF, {}),
                    TaskStep("same", SkillName.LAND, {}),
                )
            )
        with self.assertRaisesRegex(TaskPlanError, "top-level"):
            TaskStep("recover", SkillName.REACQUIRE, {})
        with self.assertRaisesRegex(TaskPlanError, "only allowed"):
            TaskStep(
                "goto_1",
                SkillName.GOTO,
                {},
                RecoveryPolicy(SkillName.REACQUIRE, 1, 10.0, 30.0),
            )
        with self.assertRaisesRegex(TaskPlanError, "must match"):
            TaskStep("Bad-ID", SkillName.GOTO, {})

    def test_reference_and_recovery_bounds_are_strict(self) -> None:
        with self.assertRaisesRegex(TaskPlanError, "field"):
            StepOutputRef("search_1", "pose")
        with self.assertRaisesRegex(TaskPlanError, "between 0 and 2"):
            RecoveryPolicy(SkillName.REACQUIRE, 3, 10.0, 30.0)
        with self.assertRaisesRegex(TaskPlanError, "integer"):
            RecoveryPolicy(SkillName.REACQUIRE, True, 10.0, 30.0)
        with self.assertRaisesRegex(TaskPlanError, "between 3 and 20"):
            RecoveryPolicy(SkillName.REACQUIRE, 1, 2.0, 30.0)
        with self.assertRaisesRegex(TaskPlanError, "between 5 and 60"):
            RecoveryPolicy(SkillName.REACQUIRE, 1, 10.0, 61.0)


if __name__ == "__main__":
    unittest.main()
