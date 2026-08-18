"""Planner/compiler/safety closure for planned HOVER and INSPECT."""

from __future__ import annotations

from math import pi
import unittest

from planner.json_schema import build_skill_plan_draft_json_schema
from planner.policy import PlannerLimits, PlannerPolicy
from planner.schemas import (
    LandingZoneSpec,
    NavigationPointSpec,
    PlanStepDraft,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraft,
)
from planner.skill_catalog import build_default_skill_catalog
from planner.symbolic_checker import PlanIssueCode, SymbolicPlanChecker
from runtime.plan_validator import PlanValidationError, PlanValidator
from runtime.safety_supervisor import SafetyAction, SafetySupervisor
from skills.hover import HoverMode
from skills.inspect import InspectApproachPolicy
from skills.motion_types import YawMode
from skills.plan import TaskPlan, TaskStep
from skills.types import SkillName


def _context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={
            "search_area": SearchRegionSpec(
                "search_area",
                (20.0, 20.0, 1.0),
                8.0,
                (12.0, 20.0, 10.0),
            )
        },
        landing_zones={"home": LandingZoneSpec("home", (0.0, 0.0))},
        navigation_points={
            "checkpoint": NavigationPointSpec(
                "checkpoint",
                (5.0, -5.0, 10.0),
            )
        },
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=10.0,
        search_timeout_s=60.0,
    )


def _draft(steps: list[dict[str, object]]) -> SkillPlanDraft:
    return SkillPlanDraft.from_dict({"schema_version": 1, "steps": steps})


def _base_steps() -> list[dict[str, object]]:
    return [
        {
            "id": "takeoff",
            "skill": "TAKEOFF",
            "args": {"altitude_m": 10.0},
        },
        {
            "id": "hover",
            "skill": "HOVER",
            "args": {
                "duration_s": 5.0,
                "yaw_mode": "FIXED",
                "yaw_deg": 90.0,
            },
        },
        {
            "id": "search",
            "skill": "SEARCH",
            "args": {
                "region": "search_area",
                "target_description": "red moving vehicle",
            },
        },
        {
            "id": "inspect",
            "skill": "INSPECT",
            "args": {
                "candidate_id": "candidate_1",
                "desired_observation_distance_m": 5.0,
                "viewpoint_change_deg": -60.0,
                "max_duration_s": 20.0,
                "approach_policy": "MAINTAIN_ALTITUDE_ORBIT",
            },
        },
        {
            "id": "goto_home",
            "skill": "GOTO",
            "args": {"destination": "home"},
        },
        {"id": "land", "skill": "LAND", "args": {"zone": "home"}},
    ]


class HoverInspectPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = _context()
        self.validator = PlanValidator()
        self.safety = SafetySupervisor(
            self.context.scene_min_xyz_m,
            self.context.scene_max_xyz_m,
        )

    def test_catalog_and_schema_expose_only_bounded_semantics(self) -> None:
        catalog = build_default_skill_catalog()
        self.assertEqual(
            [argument.name for argument in catalog.get("HOVER").arguments],
            ["duration_s", "yaw_mode", "yaw_deg"],
        )
        self.assertEqual(
            [argument.name for argument in catalog.get("INSPECT").arguments],
            [
                "candidate_id",
                "desired_observation_distance_m",
                "viewpoint_change_deg",
                "max_duration_s",
                "approach_policy",
            ],
        )
        schema = build_skill_plan_draft_json_schema(
            world_context=self.context,
            skill_catalog=catalog,
            limits=PlannerLimits(),
        )
        variants = {
            item["properties"]["skill"]["const"]: item
            for item in schema["properties"]["steps"]["items"]["oneOf"]
        }
        hover_args = variants["HOVER"]["properties"]["args"]
        self.assertNotIn("INSPECT", variants)
        self.assertEqual(hover_args["required"], ["duration_s"])
        self.assertEqual(
            hover_args["properties"]["duration_s"]["maximum"],
            60.0,
        )
        for forbidden in ("position", "center", "mode", "max_wait_s"):
            self.assertNotIn(forbidden, hover_args["properties"])

    def test_schema_boundary_rejects_untrusted_hover_and_inspect_values(self) -> None:
        self.assertEqual(PlanStepDraft.from_dict(_base_steps()[1]).skill, "HOVER")
        self.assertEqual(
            PlanStepDraft.from_dict(_base_steps()[3]).args["candidate_id"],
            "candidate_1",
        )
        invalid_args = (
            ("HOVER", {"duration_s": 0.5}),
            ("HOVER", {"duration_s": 5.0, "mode": "UNTIL_RELEASED"}),
            ("INSPECT", {"candidate_id": "candidate 1"}),
            (
                "INSPECT",
                {"candidate_id": "candidate_1", "viewpoint_change_deg": 0},
            ),
            (
                "INSPECT",
                {"candidate_id": "candidate_1", "position": [1, 2, 3]},
            ),
        )
        for index, (skill, args) in enumerate(invalid_args):
            with self.subTest(skill=skill, args=args), self.assertRaises(
                (TypeError, ValueError)
            ):
                PlanStepDraft.from_dict(
                    {"id": f"invalid_{index}", "skill": skill, "args": args}
                )

    def test_inspect_requires_prior_search_symbolically(self) -> None:
        steps = _base_steps()
        steps.pop(2)
        result = SymbolicPlanChecker().check(
            _draft(steps),
            world_context=self.context,
            limits=PlannerLimits(),
            policy=PlannerPolicy(),
        )
        self.assertIn(
            PlanIssueCode.INSPECT_WITHOUT_SEARCH,
            {issue.code for issue in result.issues},
        )
        with self.assertRaisesRegex(
            PlanValidationError,
            "INSPECT_WITHOUT_SEARCH",
        ):
            self.validator.validate_and_compile(
                _draft(steps),
                self.context,
                source="dynamic_scripted",
                trusted_inspect_candidate_ids=("candidate_1",),
            )

    def test_compiler_supplies_trusted_hover_and_inspect_goal_fields(self) -> None:
        source = _draft(_base_steps())
        compiled = self.validator.validate_and_compile(
            source,
            self.context,
            source="dynamic_scripted",
            trusted_inspect_candidate_ids=("candidate_1",),
        )
        hover = compiled.task_plan.steps[1]
        inspect = compiled.task_plan.steps[3]

        self.assertIs(hover.skill, SkillName.HOVER)
        self.assertIs(hover.params["mode"], HoverMode.TIMED)
        self.assertEqual(hover.params["duration_s"], 5.0)
        self.assertEqual(hover.params["max_wait_s"], 5.0)
        self.assertEqual(hover.params["position_tolerance_m"], 0.25)
        self.assertEqual(hover.params["max_correction_speed_mps"], 0.5)
        self.assertEqual(hover.params["reason_code"], "PLANNED_HOVER")
        self.assertIs(hover.params["motion_policy"].yaw_mode, YawMode.FIXED)
        self.assertAlmostEqual(hover.params["motion_policy"].yaw_value, pi / 2)

        self.assertIs(inspect.skill, SkillName.INSPECT)
        self.assertEqual(inspect.params["candidate_id"], "candidate_1")
        self.assertEqual(inspect.params["desired_observation_distance_m"], 5.0)
        self.assertAlmostEqual(inspect.params["viewpoint_change_rad"], -pi / 3)
        self.assertEqual(inspect.params["max_duration_s"], 20.0)
        self.assertIs(
            inspect.params["approach_policy"],
            InspectApproachPolicy.MAINTAIN_ALTITUDE_ORBIT,
        )
        for forbidden in ("position", "center", "target_position"):
            self.assertNotIn(forbidden, inspect.params)
            self.assertNotIn(forbidden, source.steps[3].args)
        self.assertIs(
            self.safety.preflight(compiled).action,
            SafetyAction.CONTINUE,
        )

    def test_initial_compile_rejects_inspect_without_candidate_bank_authority(self) -> None:
        with self.assertRaisesRegex(
            PlanValidationError,
            "not authorized by trusted runtime CandidateBank",
        ):
            self.validator.validate_and_compile(
                _draft(_base_steps()),
                self.context,
                source="dynamic_scripted",
            )

        with self.assertRaisesRegex(
            PlanValidationError,
            "not authorized by trusted runtime CandidateBank",
        ):
            self.validator.validate_and_compile(
                _draft(_base_steps()),
                self.context,
                source="dynamic_scripted",
                trusted_inspect_candidate_ids=("candidate_other",),
            )

    def test_full_ten_step_order_is_accepted(self) -> None:
        steps = _base_steps()
        steps.insert(
            2,
            {
                "id": "goto_checkpoint",
                "skill": "GOTO",
                "args": {"destination": "checkpoint"},
            },
        )
        steps.insert(
            3,
            {"id": "hover_two", "skill": "HOVER", "args": {"duration_s": 1}},
        )
        steps.insert(
            6,
            {
                "id": "track",
                "skill": "TRACK",
                "args": {"target_ref": "$search.target_id", "duration_s": 5},
            },
        )
        steps.insert(
            7,
            {"id": "hover_three", "skill": "HOVER", "args": {"duration_s": 2}},
        )
        self.assertEqual(len(steps), 10)
        compiled = self.validator.validate_and_compile(
            _draft(steps),
            self.context,
            source="dynamic_scripted",
            trusted_inspect_candidate_ids=("candidate_1",),
        )
        self.assertEqual(len(compiled.task_plan.steps), 10)
        self.assertIs(
            self.safety.preflight(compiled).action,
            SafetyAction.CONTINUE,
        )

    def test_safety_rechecks_planned_mode_bounds_and_candidate_semantics(self) -> None:
        compiled = self.validator.validate_and_compile(
            _draft(_base_steps()),
            self.context,
            source="dynamic_scripted",
            trusted_inspect_candidate_ids=("candidate_1",),
        )

        def with_replaced(index: int, step: TaskStep) -> TaskPlan:
            steps = list(compiled.task_plan.steps)
            steps[index] = step
            return TaskPlan(tuple(steps))

        hover = compiled.task_plan.steps[1]
        bad_hover_params = dict(hover.params)
        bad_hover_params["mode"] = HoverMode.UNTIL_RELEASED
        bad_hover = TaskStep(hover.step_id, hover.skill, bad_hover_params)
        self.assertIs(
            self.safety.preflight(with_replaced(1, bad_hover)).action,
            SafetyAction.ABORT,
        )

        inspect = compiled.task_plan.steps[3]
        bad_inspect_params = dict(inspect.params)
        bad_inspect_params["candidate_id"] = "candidate with spaces"
        bad_inspect = TaskStep(inspect.step_id, inspect.skill, bad_inspect_params)
        self.assertIs(
            self.safety.preflight(with_replaced(3, bad_inspect)).action,
            SafetyAction.ABORT,
        )

        injected = dict(inspect.params)
        injected["position"] = (1.0, 2.0, 3.0)
        injected_step = TaskStep(inspect.step_id, inspect.skill, injected)
        self.assertIs(
            self.safety.preflight(with_replaced(3, injected_step)).action,
            SafetyAction.ABORT,
        )


if __name__ == "__main__":
    unittest.main()
