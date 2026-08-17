from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from planner.base import MissionPlanner
from planner.schemas import (
    LandingZoneSpec,
    MissionIntent,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
)
from planner.scripted_planner import ScriptedPlanner


def request() -> PlannerRequest:
    region = SearchRegionSpec("search_area", (20, 30, 0), 15, (5, 10, 10))
    zone = LandingZoneSpec("home", (0, 0))
    context = PlannerWorldContext(
        scene_min_xyz_m=(-50, -50, 0),
        scene_max_xyz_m=(50, 50, 30),
        initial_uav_xyz_m=(0, 0, 0),
        search_regions={"search_area": region},
        landing_zones={"home": zone},
        default_takeoff_altitude_m=10,
        default_track_duration_s=30,
        search_timeout_s=60,
    )
    return PlannerRequest("find the moving target and return home", context)


class ScriptedPlannerTest(unittest.TestCase):
    def test_is_mission_planner_and_returns_stable_fresh_intents(self) -> None:
        expected = MissionIntent(
            target_description="moving target",
            search_region="search_area",
            track_duration_s=30,
            landing_zone="home",
        )
        planner = ScriptedPlanner(expected)

        first = planner.plan(request())
        second = planner.plan(request())

        self.assertIsInstance(planner, MissionPlanner)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertIsNot(first, second)
        execution = planner.plan_with_diagnostics(request())
        self.assertEqual(execution.output, expected)
        self.assertEqual(execution.diagnostics.model_calls, 0)
        self.assertFalse(execution.diagnostics.structured_output_enabled)

    def test_caller_cannot_mutate_or_pollute_planner_state(self) -> None:
        planner = ScriptedPlanner(
            MissionIntent("moving target", "search_area", 30, "home")
        )
        returned = planner.plan(request())
        serialized = returned.to_dict()
        serialized["target_description"] = "polluted"

        with self.assertRaises(FrozenInstanceError):
            returned.target_description = "polluted"  # type: ignore[misc]
        self.assertEqual(
            planner.plan(request()).target_description,
            "moving target",
        )

    def test_rejects_wrong_constructor_and_request_types(self) -> None:
        with self.assertRaises(TypeError):
            ScriptedPlanner({})  # type: ignore[arg-type]
        planner = ScriptedPlanner(
            MissionIntent("moving target", "search_area", 30, "home")
        )
        with self.assertRaises(TypeError):
            planner.plan(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
