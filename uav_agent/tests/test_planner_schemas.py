from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import unittest

from planner.schemas import (
    CompiledMission,
    LandingZoneSpec,
    MissionIntent,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
)
from skills.manager import TaskPlan


def search_region() -> SearchRegionSpec:
    return SearchRegionSpec(
        name="search_area",
        center_xyz_m=[20, 30, 0],
        radius_m=15,
        approach_xyz_m=[5, 10, 10],
        description="north search area",
    )


def landing_zone() -> LandingZoneSpec:
    return LandingZoneSpec(
        name="home",
        position_xy_m=[0, 0],
        ground_altitude_m=0,
    )


def world_context(
    *,
    regions: dict[str, SearchRegionSpec] | None = None,
    zones: dict[str, LandingZoneSpec] | None = None,
) -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=[-50, -50, 0],
        scene_max_xyz_m=[50, 50, 30],
        initial_uav_xyz_m=[0, 0, 0],
        search_regions=regions if regions is not None else {"search_area": search_region()},
        landing_zones=zones if zones is not None else {"home": landing_zone()},
        default_takeoff_altitude_m=10,
        default_track_duration_s=30,
        search_timeout_s=60,
    )


def mission_intent() -> MissionIntent:
    return MissionIntent(
        target_description="moving red target",
        search_region="search_area",
        track_duration_s=30,
        landing_zone="home",
    )


def five_step_plan() -> TaskPlan:
    return TaskPlan.from_dicts(
        [
            {"skill": "TAKEOFF", "target_altitude": 10},
            {"skill": "GOTO", "position": [5, 10, 10]},
            {
                "skill": "SEARCH",
                "center": [20, 30, 0],
                "radius": 15,
                "target_description": "moving red target",
                "search_altitude": 10,
            },
            {
                "skill": "TRACK",
                "target_id": "$SEARCH.result.target_id",
                "track_duration": 30,
            },
            {"skill": "LAND"},
        ]
    )


class RegionAndWorldSchemaTest(unittest.TestCase):
    def test_region_and_landing_inputs_are_normalized(self) -> None:
        region = search_region()
        zone = landing_zone()

        self.assertEqual(region.center_xyz_m, (20.0, 30.0, 0.0))
        self.assertEqual(region.approach_xyz_m, (5.0, 10.0, 10.0))
        self.assertEqual(region.radius_m, 15.0)
        self.assertEqual(zone.position_xy_m, (0.0, 0.0))
        self.assertEqual(zone.ground_altitude_m, 0.0)

    def test_region_rejects_invalid_name_vectors_and_radius(self) -> None:
        with self.assertRaises(ValueError):
            SearchRegionSpec(" ", (0, 0, 0), 1, (0, 0, 0))
        with self.assertRaises(ValueError):
            SearchRegionSpec("area", (0, 0), 1, (0, 0, 0))
        with self.assertRaises(TypeError):
            SearchRegionSpec("area", (0, True, 0), 1, (0, 0, 0))
        for radius in (0, -1, True, math.nan, math.inf):
            with self.subTest(radius=radius), self.assertRaises((TypeError, ValueError)):
                SearchRegionSpec("area", (0, 0, 0), radius, (0, 0, 0))

    def test_landing_zone_rejects_nonfinite_and_bool_values(self) -> None:
        for position in ((0,), (0, True), (0, math.inf)):
            with self.subTest(position=position), self.assertRaises((TypeError, ValueError)):
                LandingZoneSpec("home", position)
        for altitude in (True, math.nan, math.inf):
            with self.subTest(altitude=altitude), self.assertRaises((TypeError, ValueError)):
                LandingZoneSpec("home", (0, 0), altitude)

    def test_world_context_takes_readonly_mapping_snapshots(self) -> None:
        regions = {"search_area": search_region()}
        zones = {"home": landing_zone()}
        context = world_context(regions=regions, zones=zones)

        regions.clear()
        zones["alternate"] = LandingZoneSpec("alternate", (1, 1))

        self.assertEqual(tuple(context.search_regions), ("search_area",))
        self.assertEqual(tuple(context.landing_zones), ("home",))
        with self.assertRaises(TypeError):
            context.search_regions["new"] = search_region()  # type: ignore[index]

    def test_world_context_rejects_invalid_bounds_and_defaults(self) -> None:
        kwargs = dict(
            initial_uav_xyz_m=(0, 0, 0),
            search_regions={"search_area": search_region()},
            landing_zones={"home": landing_zone()},
            default_takeoff_altitude_m=10,
            default_track_duration_s=30,
            search_timeout_s=60,
        )
        with self.assertRaises(ValueError):
            PlannerWorldContext(
                scene_min_xyz_m=(0, 0, 0),
                scene_max_xyz_m=(0, 10, 10),
                **kwargs,
            )

        for field_name, bad_value in (
            ("default_takeoff_altitude_m", math.nan),
            ("default_takeoff_altitude_m", True),
            ("default_takeoff_altitude_m", 0),
            ("default_track_duration_s", 0),
            ("search_timeout_s", math.inf),
            ("goto_timeout_s", -1),
            ("land_timeout_s", True),
        ):
            values = dict(kwargs)
            values[field_name] = bad_value
            with self.subTest(field=field_name), self.assertRaises((TypeError, ValueError)):
                PlannerWorldContext(
                    scene_min_xyz_m=(-10, -10, 0),
                    scene_max_xyz_m=(10, 10, 20),
                    **values,
                )


class MissionIntentSchemaTest(unittest.TestCase):
    def test_from_dict_round_trip_is_strict_and_json_compatible(self) -> None:
        raw = {
            "target_description": "moving red target",
            "search_region": "search_area",
            "track_duration_s": 30,
            "landing_zone": "home",
            "takeoff_altitude_m": 12,
        }

        intent = MissionIntent.from_dict(raw)

        self.assertEqual(intent.track_duration_s, 30.0)
        self.assertEqual(intent.takeoff_altitude_m, 12.0)
        self.assertEqual(MissionIntent.from_dict(intent.to_dict()), intent)
        self.assertEqual(json.loads(json.dumps(intent.to_dict())), intent.to_dict())

    def test_unknown_and_oracle_fields_are_rejected(self) -> None:
        valid = mission_intent().to_dict()
        for field_name in ("unexpected", "target_position", "target_xyz", "oracle_target_pose"):
            raw = dict(valid)
            raw[field_name] = [1, 2, 3]
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                MissionIntent.from_dict(raw)

    def test_every_required_field_is_required(self) -> None:
        valid = mission_intent().to_dict()
        valid.pop("takeoff_altitude_m")
        for field_name in (
            "target_description",
            "search_region",
            "track_duration_s",
            "landing_zone",
        ):
            raw = dict(valid)
            raw.pop(field_name)
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                MissionIntent.from_dict(raw)

    def test_invalid_strings_numbers_nan_inf_and_bool_are_rejected(self) -> None:
        for field_name in ("target_description", "search_region", "landing_zone"):
            raw = mission_intent().to_dict()
            raw[field_name] = " "
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                MissionIntent.from_dict(raw)

        for duration in (0, -1, True, math.nan, math.inf):
            raw = mission_intent().to_dict()
            raw["track_duration_s"] = duration
            with self.subTest(duration=duration), self.assertRaises((TypeError, ValueError)):
                MissionIntent.from_dict(raw)

        for altitude in (True, math.nan, math.inf):
            raw = mission_intent().to_dict()
            raw["takeoff_altitude_m"] = altitude
            with self.subTest(altitude=altitude), self.assertRaises((TypeError, ValueError)):
                MissionIntent.from_dict(raw)

    def test_request_and_compiled_mission_enforce_boundary_types(self) -> None:
        request = PlannerRequest("search and return home", world_context())
        mission = CompiledMission(mission_intent(), five_step_plan(), "scripted")

        self.assertEqual(request.instruction, "search and return home")
        self.assertEqual(mission.source, "scripted")
        with self.assertRaises(FrozenInstanceError):
            request.instruction = "changed"  # type: ignore[misc]
        with self.assertRaises(ValueError):
            CompiledMission(mission_intent(), five_step_plan(), "oracle")


if __name__ == "__main__":
    unittest.main()
