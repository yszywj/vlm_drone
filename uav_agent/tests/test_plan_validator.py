"""Tests for the high-level intent validation and six-step compiler boundary."""

from __future__ import annotations

from dataclasses import replace
import unittest

from planner.schemas import (
    CompiledMission,
    LandingZoneSpec,
    MissionIntent,
    PlannerWorldContext,
    SearchRegionSpec,
)
from runtime.plan_validator import PlanValidationError, PlanValidator


class PlanValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.search_region = SearchRegionSpec(
            name="search_area",
            center_xyz_m=(20.0, 30.0, 0.0),
            radius_m=15.0,
            approach_xyz_m=(20.0, 12.0, 10.0),
            description="north search sector",
        )
        self.landing_zone = LandingZoneSpec(
            name="home",
            position_xy_m=(1.0, -2.0),
            ground_altitude_m=0.0,
            description="launch pad",
        )
        self.context = PlannerWorldContext(
            scene_min_xyz_m=(-50.0, -50.0, 0.0),
            scene_max_xyz_m=(50.0, 50.0, 30.0),
            initial_uav_xyz_m=(0.0, 0.0, 0.0),
            search_regions={"search_area": self.search_region},
            landing_zones={"home": self.landing_zone},
            default_takeoff_altitude_m=10.0,
            default_track_duration_s=30.0,
            search_timeout_s=75.0,
            goto_timeout_s=120.0,
            land_timeout_s=60.0,
        )
        self.intent = MissionIntent(
            target_description="moving red target",
            search_region="search_area",
            track_duration_s=30.0,
            landing_zone="home",
        )
        self.validator = PlanValidator()

    def test_compiles_exact_deterministic_six_step_plan(self) -> None:
        compiled = self.validator.validate_and_compile(
            self.intent,
            self.context,
            source="scripted",
        )

        self.assertIsInstance(compiled, CompiledMission)
        self.assertEqual(compiled.intent, self.intent)
        self.assertEqual(compiled.source, "scripted")
        self.assertEqual(
            compiled.task_plan.to_dicts(),
            [
                {
                    "id": "step_01",
                    "skill": "TAKEOFF",
                    "target_altitude": 10.0,
                    # No separate takeoff timeout exists in the world schema;
                    # the trusted generic motion timeout is used deliberately.
                    "timeout": 120.0,
                },
                {
                    "id": "step_02",
                    "skill": "GOTO",
                    "position": [20.0, 12.0, 10.0],
                    "timeout": 120.0,
                },
                {
                    "id": "step_03",
                    "skill": "SEARCH",
                    "center": [20.0, 30.0, 0.0],
                    "radius": 15.0,
                    "target_description": "moving red target",
                    "search_altitude": 10.0,
                    "timeout": 75.0,
                },
                {
                    "id": "step_04",
                    "skill": "TRACK",
                    "target_id": "$SEARCH.result.target_id",
                    "desired_altitude": 10.0,
                    "track_duration": 30.0,
                },
                {
                    "id": "step_05",
                    "skill": "GOTO",
                    "position": [1.0, -2.0, 10.0],
                    "timeout": 120.0,
                },
                {
                    "id": "step_06",
                    "skill": "LAND",
                    "ground_altitude": 0.0,
                    "timeout": 60.0,
                },
            ],
        )

    def test_takeoff_override_is_used_for_all_flight_altitudes(self) -> None:
        intent = replace(self.intent, takeoff_altitude_m=12.5)
        entries = self.validator.validate_and_compile(
            intent,
            self.context,
            source="llm",
        ).task_plan.to_dicts()

        self.assertEqual(entries[0]["target_altitude"], 12.5)
        self.assertEqual(entries[2]["search_altitude"], 12.5)
        self.assertEqual(entries[3]["desired_altitude"], 12.5)
        self.assertEqual(entries[4]["position"], [1.0, -2.0, 12.5])

    def test_compiled_plan_never_contains_reacquire(self) -> None:
        compiled = self.validator.validate_and_compile(
            self.intent,
            self.context,
            source="scripted",
        )
        names = [entry["skill"] for entry in compiled.task_plan.to_dicts()]

        self.assertNotIn("REACQUIRE", names)

    def test_unknown_search_region_is_rejected(self) -> None:
        intent = replace(self.intent, search_region="missing")
        with self.assertRaisesRegex(PlanValidationError, "unknown search_region"):
            self.validator.validate_and_compile(intent, self.context, source="scripted")

    def test_unknown_landing_zone_is_rejected(self) -> None:
        intent = replace(self.intent, landing_zone="missing")
        with self.assertRaisesRegex(PlanValidationError, "unknown landing_zone"):
            self.validator.validate_and_compile(intent, self.context, source="scripted")

    def test_search_center_outside_scene_is_rejected(self) -> None:
        region = replace(self.search_region, center_xyz_m=(51.0, 0.0, 0.0))
        context = replace(self.context, search_regions={"search_area": region})
        with self.assertRaisesRegex(PlanValidationError, "search center"):
            self.validator.validate_and_compile(self.intent, context, source="scripted")

    def test_search_approach_outside_scene_is_rejected(self) -> None:
        region = replace(self.search_region, approach_xyz_m=(20.0, 12.0, 31.0))
        context = replace(self.context, search_regions={"search_area": region})
        with self.assertRaisesRegex(PlanValidationError, "search approach"):
            self.validator.validate_and_compile(self.intent, context, source="scripted")

    def test_landing_xy_outside_scene_is_rejected(self) -> None:
        zone = replace(self.landing_zone, position_xy_m=(-51.0, 0.0))
        context = replace(self.context, landing_zones={"home": zone})
        with self.assertRaisesRegex(PlanValidationError, "scene XY"):
            self.validator.validate_and_compile(self.intent, context, source="scripted")

    def test_landing_ground_altitude_outside_scene_is_rejected(self) -> None:
        zone = replace(self.landing_zone, ground_altitude_m=31.0)
        context = replace(self.context, landing_zones={"home": zone})
        with self.assertRaisesRegex(PlanValidationError, "ground altitude"):
            self.validator.validate_and_compile(self.intent, context, source="scripted")

    def test_landing_ground_altitude_above_flight_altitude_is_rejected(self) -> None:
        zone = replace(self.landing_zone, ground_altitude_m=12.0)
        context = replace(self.context, landing_zones={"home": zone})

        with self.assertRaisesRegex(PlanValidationError, "flight altitude"):
            self.validator.validate_and_compile(
                self.intent,
                context,
                source="scripted",
            )

    def test_takeoff_altitude_outside_scene_is_rejected(self) -> None:
        intent = replace(self.intent, takeoff_altitude_m=31.0)
        with self.assertRaisesRegex(PlanValidationError, "scene Z"):
            self.validator.validate_and_compile(intent, self.context, source="scripted")

    def test_takeoff_altitude_below_initial_uav_is_rejected(self) -> None:
        context = replace(
            self.context,
            initial_uav_xyz_m=(0.0, 0.0, 8.0),
        )
        intent = replace(self.intent, takeoff_altitude_m=7.0)

        with self.assertRaisesRegex(PlanValidationError, "initial UAV altitude"):
            self.validator.validate_and_compile(intent, context, source="llm")

    def test_initial_uav_outside_scene_is_rejected(self) -> None:
        context = replace(
            self.context,
            initial_uav_xyz_m=(51.0, 0.0, 0.0),
        )

        with self.assertRaisesRegex(PlanValidationError, "initial UAV position"):
            self.validator.validate_and_compile(
                self.intent,
                context,
                source="scripted",
            )

    def test_nonpositive_takeoff_altitude_is_rejected(self) -> None:
        intent = replace(self.intent, takeoff_altitude_m=0.0)
        with self.assertRaisesRegex(PlanValidationError, "greater than zero"):
            self.validator.validate_and_compile(intent, self.context, source="scripted")

    def test_track_duration_below_runtime_minimum_is_rejected(self) -> None:
        intent = replace(self.intent, track_duration_s=0.5)
        with self.assertRaisesRegex(PlanValidationError, "between 1 and 600"):
            self.validator.validate_and_compile(intent, self.context, source="scripted")

    def test_track_duration_above_runtime_maximum_is_rejected(self) -> None:
        intent = replace(self.intent, track_duration_s=600.1)
        with self.assertRaisesRegex(PlanValidationError, "between 1 and 600"):
            self.validator.validate_and_compile(intent, self.context, source="scripted")

    def test_overlong_target_description_is_rejected(self) -> None:
        intent = replace(self.intent, target_description="x" * 257)
        with self.assertRaisesRegex(PlanValidationError, "at most 256"):
            self.validator.validate_and_compile(intent, self.context, source="scripted")

    def test_invalid_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "source"):
            self.validator.validate_and_compile(self.intent, self.context, source="oracle")
        with self.assertRaisesRegex(PlanValidationError, "source"):
            self.validator.validate_and_compile(  # type: ignore[arg-type]
                self.intent,
                self.context,
                source=["scripted"],
            )

    def test_boundary_points_and_duration_are_inclusive(self) -> None:
        region = replace(
            self.search_region,
            center_xyz_m=(-50.0, 50.0, 0.0),
            approach_xyz_m=(50.0, -50.0, 30.0),
        )
        zone = replace(
            self.landing_zone,
            position_xy_m=(50.0, -50.0),
            ground_altitude_m=0.0,
        )
        context = replace(
            self.context,
            search_regions={"search_area": region},
            landing_zones={"home": zone},
        )
        intent = replace(
            self.intent,
            track_duration_s=600.0,
            takeoff_altitude_m=30.0,
        )

        compiled = self.validator.validate_and_compile(
            intent,
            context,
            source="scripted",
        )
        self.assertEqual(len(compiled.task_plan.steps), 6)


if __name__ == "__main__":
    unittest.main()
