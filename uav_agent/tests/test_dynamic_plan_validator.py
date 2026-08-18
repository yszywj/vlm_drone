"""Pure-Python tests for trusted dynamic Skill-plan compilation."""

from __future__ import annotations

from dataclasses import replace
from math import pi
import unittest

from configs.schema import PlannerConfig
from planner.policy import PlannerPolicy
from planner.schemas import (
    LandingZoneSpec,
    NavigationPointSpec,
    PlanStepDraft,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraft,
)
from runtime.plan_validator import PlanValidationError, PlannerLimits, PlanValidator
from skills.motion_types import MotionPolicy, YawMode
from skills.plan import StepOutputRef


def _context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={
            "search_area": SearchRegionSpec(
                name="search_area",
                center_xyz_m=(20.0, 20.0, 0.5),
                radius_m=8.0,
                approach_xyz_m=(12.0, 20.0, 10.0),
            )
        },
        landing_zones={
            "home": LandingZoneSpec(
                name="home",
                position_xy_m=(0.0, 0.0),
                ground_altitude_m=0.0,
            )
        },
        navigation_points={
            "checkpoint": NavigationPointSpec(
                name="checkpoint",
                position_xyz_m=(5.0, -6.0, 3.0),
            )
        },
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=30.0,
        search_timeout_s=75.0,
        goto_timeout_s=120.0,
        land_timeout_s=60.0,
    )


def _draft(steps: list[dict[str, object]]) -> SkillPlanDraft:
    return SkillPlanDraft.from_dict({"schema_version": 1, "steps": steps})


def _takeoff(step_id: str = "takeoff_1") -> dict[str, object]:
    return {"id": step_id, "skill": "TAKEOFF", "args": {"altitude_m": 10.0}}


def _goto(
    step_id: str,
    destination: str,
    **args: object,
) -> dict[str, object]:
    return {
        "id": step_id,
        "skill": "GOTO",
        "args": {"destination": destination, **args},
    }


def _search(step_id: str = "search_1") -> dict[str, object]:
    return {
        "id": step_id,
        "skill": "SEARCH",
        "args": {
            "region": "search_area",
            "target_description": "moving target",
            "altitude_m": 10.0,
        },
    }


def _track(
    step_id: str = "track_1",
    *,
    search_id: str = "search_1",
    recovery: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": step_id,
        "skill": "TRACK",
        "args": {
            "target_ref": f"${search_id}.target_id",
            "duration_s": 10.0,
        },
    }
    if recovery is not None:
        result["recovery"] = recovery
    return result


def _land(step_id: str = "land_1") -> dict[str, object]:
    return {"id": step_id, "skill": "LAND", "args": {"zone": "home"}}


def _recovery(attempts: int = 2) -> dict[str, object]:
    return {
        "skill": "REACQUIRE",
        "max_attempts": attempts,
        "search_radius_m": 10.0,
        "timeout_s": 30.0,
    }


class DynamicPlanValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = _context()
        self.validator = PlanValidator()

    def compile(self, steps: list[dict[str, object]]):
        return self.validator.validate_and_compile(
            _draft(steps),
            self.context,
            source="dynamic_scripted",
        )

    def test_standard_search_track_plan_compiles_trusted_values(self) -> None:
        compiled = self.compile(
            [
                _takeoff(),
                _goto("goto_search", "search_area"),
                _search(),
                _track(recovery=_recovery()),
                _goto("goto_home", "home"),
                _land(),
            ]
        )
        self.assertIsNotNone(compiled.skill_plan_draft)
        self.assertIsNone(compiled.intent)
        self.assertEqual(compiled.source, "dynamic_scripted")
        self.assertEqual(len(compiled.task_plan.steps), 6)
        self.assertTrue(compiled.compiler_notes)

        goto = compiled.task_plan.steps[1]
        self.assertEqual(goto.params["position"], (12.0, 20.0, 10.0))
        self.assertEqual(goto.params["timeout"], 120.0)
        self.assertIsInstance(goto.params["motion_policy"], MotionPolicy)
        track = compiled.task_plan.steps[3]
        self.assertEqual(
            track.params["target_id"],
            StepOutputRef("search_1", "target_id"),
        )
        self.assertEqual(track.params["max_speed"], 2.0)
        self.assertEqual(track.recovery.max_attempts, 2)

    def test_recovery_geometry_and_timeout_receive_trusted_defaults(self) -> None:
        compiled = self.compile(
            [
                _takeoff(),
                _search(),
                _track(
                    recovery={"skill": "REACQUIRE", "max_attempts": 1}
                ),
                _goto("goto_home", "home"),
                _land(),
            ]
        )
        recovery = compiled.task_plan.steps[2].recovery
        self.assertEqual(recovery.search_radius_m, 10.0)
        self.assertEqual(recovery.timeout_s, 30.0)

    def test_lost_target_action_and_trusted_defaults_are_compiled(self) -> None:
        base = [_takeoff(), _search()]
        tail = [_goto("goto_home", "home"), _land()]

        inherited = self.compile([*base, _track(), *tail])
        inherited_recovery = inherited.task_plan.steps[2].recovery
        self.assertIsNotNone(inherited_recovery)
        self.assertEqual(inherited_recovery.max_attempts, 2)
        self.assertTrue(
            any(
                "recovery injected from trusted default policy" in note
                for note in inherited.compiler_notes
            )
        )

        explicit_fail = _track()
        explicit_fail["args"]["on_target_lost"] = "FAIL"
        failed = self.compile([*base, explicit_fail, *tail])
        self.assertIsNone(failed.task_plan.steps[2].recovery)
        self.assertTrue(
            any(
                "recovery explicitly disabled by on_target_lost=FAIL" in note
                for note in failed.compiler_notes
            )
        )

        explicit_reacquire = _track()
        explicit_reacquire["args"]["on_target_lost"] = "REACQUIRE"
        enabled = self.compile([*base, explicit_reacquire, *tail])
        self.assertEqual(enabled.task_plan.steps[2].recovery.max_attempts, 2)

        legacy_zero = _track(recovery=_recovery(0))
        deprecated = self.compile([*base, legacy_zero, *tail])
        self.assertIsNone(deprecated.task_plan.steps[2].recovery)
        self.assertTrue(
            any("deprecated max_attempts=0" in note for note in deprecated.compiler_notes)
        )

    def test_compiled_land_uses_only_trusted_zone_geometry(self) -> None:
        compiled = self.compile([_takeoff(), _goto("goto_home", "home"), _land()])
        landing_approach = compiled.task_plan.steps[-2]
        land = compiled.task_plan.steps[-1]
        self.assertEqual(landing_approach.params["tolerance"], 0.75)
        self.assertEqual(land.params["expected_position_xy"], (0.0, 0.0))
        self.assertEqual(land.params["zone_tolerance_m"], 0.75)
        self.assertNotIn("expected_position_xy", compiled.skill_plan_draft.steps[-1].args)
        self.assertTrue(
            any("trusted landing geometry attached" in note for note in compiled.compiler_notes)
        )

    def test_navigation_only_plan_is_not_forced_to_search(self) -> None:
        compiled = self.compile(
            [
                _takeoff(),
                _goto("goto_search", "search_area"),
                _goto("goto_home", "home"),
                _land(),
            ]
        )
        self.assertEqual(
            [step.skill.value for step in compiled.task_plan.steps],
            ["TAKEOFF", "GOTO", "GOTO", "LAND"],
        )

    def test_search_only_plan_compiles(self) -> None:
        compiled = self.compile(
            [
                _takeoff(),
                _goto("goto_search", "search_area"),
                _search(),
                _goto("goto_home", "home"),
                _land(),
            ]
        )
        self.assertNotIn(
            "TRACK", [step.skill.value for step in compiled.task_plan.steps]
        )

    def test_track_need_not_be_adjacent_to_search(self) -> None:
        compiled = self.compile(
            [
                _takeoff(),
                _goto("goto_search", "search_area"),
                _search(),
                _goto("goto_checkpoint", "checkpoint"),
                _track(),
                _goto("goto_home", "home"),
                _land(),
            ]
        )
        self.assertEqual(compiled.task_plan.steps[3].params["position"], (5.0, -6.0, 10.0))

    def test_takeoff_land_is_only_allowed_at_the_initial_zone(self) -> None:
        self.assertEqual(len(self.compile([_takeoff(), _land()]).task_plan.steps), 2)
        moved_home = replace(
            self.context,
            landing_zones={
                "home": LandingZoneSpec("home", (1.0, 0.0), 0.0)
            },
        )
        with self.assertRaisesRegex(PlanValidationError, "LAND_GOTO_MISSING"):
            self.validator.validate_and_compile(
                _draft([_takeoff(), _land()]),
                moved_home,
                source="dynamic_scripted",
            )

    def test_unknown_named_locations_are_rejected(self) -> None:
        cases = (
            [_takeoff(), _goto("goto_bad", "missing"), _goto("goto_home", "home"), _land()],
            [
                _takeoff(),
                {
                    "id": "search_1",
                    "skill": "SEARCH",
                    "args": {
                        "region": "missing",
                        "target_description": "moving target",
                    },
                },
                _goto("goto_home", "home"),
                _land(),
            ],
            [_takeoff(), _goto("goto_home", "home"), {"id": "land_1", "skill": "LAND", "args": {"zone": "missing"}}],
        )
        for steps in cases:
            with self.subTest(steps=steps):
                with self.assertRaises(PlanValidationError):
                    self.compile(steps)

    def test_ambiguous_named_location_is_rejected(self) -> None:
        context = replace(
            self.context,
            navigation_points={
                "home": NavigationPointSpec("home", (2.0, 2.0, 10.0))
            },
        )
        with self.assertRaisesRegex(PlanValidationError, "ambiguous"):
            self.validator.validate_and_compile(
                _draft([_takeoff(), _goto("goto_home", "home"), _land()]),
                context,
                source="dynamic_scripted",
            )

    def test_raw_low_level_arguments_are_rejected_before_compilation(self) -> None:
        for skill, args in (
            ("GOTO", {"destination": "home", "position": [1, 2, 3]}),
            ("SEARCH", {"region": "search_area", "target_description": "x", "center": [0, 0, 0]}),
            ("TRACK", {"target_ref": "$search_1.target_id", "duration_s": 2, "target_id": "truth"}),
        ):
            with self.subTest(skill=skill):
                with self.assertRaisesRegex(ValueError, "unknown"):
                    PlanStepDraft("bad_step", skill, args)

    def test_missing_takeoff_and_nonfinal_land_are_rejected(self) -> None:
        for steps in (
            [_goto("goto_home", "home"), _land()],
            [_takeoff(), _land(), _goto("goto_home", "home")],
        ):
            with self.subTest(steps=steps):
                with self.assertRaises(PlanValidationError):
                    self.compile(steps)

    def test_call_count_and_recovery_budgets_are_enforced(self) -> None:
        too_many_gotos = [_takeoff()]
        too_many_gotos.extend(
            _goto(f"goto_{index}", "checkpoint") for index in range(6)
        )
        too_many_gotos.extend([_goto("goto_home", "home"), _land()])
        with self.assertRaisesRegex(PlanValidationError, "GOTO_LIMIT_EXCEEDED"):
            self.compile(too_many_gotos)

        with self.assertRaisesRegex(PlanValidationError, "SEARCH_LIMIT_EXCEEDED"):
            self.compile(
                [
                    _takeoff(),
                    _search("search_1"),
                    _search("search_2"),
                    _goto("goto_home", "home"),
                    _land(),
                ]
            )

        with self.assertRaisesRegex(PlanValidationError, "TRACK_LIMIT_EXCEEDED"):
            self.compile(
                [
                    _takeoff(),
                    _search(),
                    _track("track_1"),
                    _track("track_2"),
                    _track("track_3"),
                    _goto("goto_home", "home"),
                    _land(),
                ]
            )

        budget_validator = PlanValidator(
            PlannerLimits(max_total_reacquire_attempts=3),
            PlannerPolicy(default_reacquire_max_attempts=1),
        )
        plan = _draft(
            [
                _takeoff(),
                _search(),
                _track("track_1", recovery=_recovery(2)),
                _track("track_2", recovery=_recovery(2)),
                _goto("goto_home", "home"),
                _land(),
            ]
        )
        with self.assertRaisesRegex(PlanValidationError, "RECOVERY_BUDGET_EXCEEDED"):
            budget_validator.validate_and_compile(
                plan, self.context, source="dynamic_scripted"
            )

    def test_altitude_bounds_and_landing_altitude_are_enforced(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "scene Z"):
            self.compile(
                [
                    {"id": "takeoff_1", "skill": "TAKEOFF", "args": {"altitude_m": 31}},
                    _goto("goto_home", "home"),
                    _land(),
                ]
            )

        raised_zone = replace(
            self.context,
            landing_zones={"home": LandingZoneSpec("home", (0, 0), 12.0)},
        )
        with self.assertRaisesRegex(PlanValidationError, "current flight altitude"):
            self.validator.validate_and_compile(
                _draft([_takeoff(), _goto("goto_home", "home"), _land()]),
                raised_zone,
                source="dynamic_scripted",
            )

    def test_track_distance_is_bounded_by_trusted_scene_scale(self) -> None:
        track = _track()
        track["args"]["desired_distance_m"] = 1e308  # type: ignore[index]
        with self.assertRaisesRegex(PlanValidationError, "scene scale"):
            self.compile(
                [
                    _takeoff(),
                    _search(),
                    track,
                    _goto("goto_home", "home"),
                    _land(),
                ]
            )

    def test_fixed_yaw_is_converted_from_degrees_and_missing_value_rejected(self) -> None:
        compiled = self.compile(
            [
                {
                    "id": "takeoff_1",
                    "skill": "TAKEOFF",
                    "args": {"altitude_m": 10, "yaw_mode": "FIXED", "yaw_deg": 180},
                },
                _goto("goto_home", "home", yaw_mode="FIXED", yaw_deg=90),
                _land(),
            ]
        )
        self.assertAlmostEqual(compiled.task_plan.steps[0].params["yaw_value"], pi)
        goto_policy = compiled.task_plan.steps[1].params["motion_policy"]
        self.assertIs(goto_policy.yaw_mode, YawMode.FIXED)
        self.assertAlmostEqual(goto_policy.yaw_value, pi / 2)
        with self.assertRaisesRegex(ValueError, "yaw_deg"):
            PlanStepDraft(
                "takeoff_1",
                "TAKEOFF",
                {"yaw_mode": "FIXED"},
            )

    def test_face_point_uses_trusted_semantic_geometry(self) -> None:
        compiled = self.compile(
            [
                _takeoff(),
                _goto("goto_search", "search_area", yaw_mode="FACE_POINT"),
                _goto("goto_home", "home"),
                _land(),
            ]
        )
        policy = compiled.task_plan.steps[1].params["motion_policy"]
        self.assertIs(policy.yaw_mode, YawMode.FACE_POINT)
        self.assertEqual(policy.look_at_point, (20.0, 20.0, 0.5))
        self.assertNotEqual(
            policy.look_at_point,
            compiled.task_plan.steps[1].params["position"],
        )

    def test_land_must_follow_goto_to_the_same_zone(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "LAND_ZONE_MISMATCH"):
            self.compile([_takeoff(), _goto("goto_search", "search_area"), _land()])
        with self.assertRaisesRegex(PlanValidationError, "LAND_ZONE_MISMATCH"):
            self.compile(
                [
                    _takeoff(),
                    _goto("goto_home", "home"),
                    _goto("goto_checkpoint", "checkpoint"),
                    _land(),
                ]
            )

    def test_source_and_output_types_are_not_cross_wired(self) -> None:
        draft = _draft([_takeoff(), _land()])
        with self.assertRaisesRegex(PlanValidationError, "dynamic"):
            self.validator.validate_and_compile(
                draft, self.context, source="scripted"
            )

    def test_planner_limits_are_strict_and_convert_from_config_shape(self) -> None:
        for kwargs in (
            {"max_plan_steps": True},
            {"max_plan_steps": 1},
            {"max_plan_steps": 11},
            {"max_search_calls": 2},
            {"min_track_duration_s": 20, "max_track_duration_s": 10},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PlannerLimits(**kwargs)
        self.assertEqual(
            PlannerLimits.from_config(PlannerConfig()),
            PlannerLimits(),
        )


if __name__ == "__main__":
    unittest.main()
