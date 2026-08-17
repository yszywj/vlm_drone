from __future__ import annotations

import json
import unittest

from planner.json_schema import build_skill_plan_draft_json_schema
from planner.policy import PlannerLimits
from planner.schemas import (
    LandingZoneSpec,
    NavigationPointSpec,
    PlannerWorldContext,
    SearchRegionSpec,
)
from planner.skill_catalog import build_default_skill_catalog


def _world_context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-51.234567, -41.345678, 0.123456),
        scene_max_xyz_m=(61.456789, 71.567891, 31.678912),
        initial_uav_xyz_m=(7.123456, -8.234567, 1.345678),
        search_regions={
            "search_area": SearchRegionSpec(
                "search_area",
                center_xyz_m=(12.345678, 23.456789, 2.345679),
                radius_m=14.567891,
                approach_xyz_m=(16.789123, 27.891234, 9.876543),
                description="north sector",
            ),
            "east_area": SearchRegionSpec(
                "east_area",
                center_xyz_m=(-21.123456, 11.234567, 2.456789),
                radius_m=9.345678,
                approach_xyz_m=(-18.456789, 12.567891, 8.678912),
                description="east sector",
            ),
        },
        landing_zones={
            "home": LandingZoneSpec(
                "home",
                position_xy_m=(31.415926, -27.182818),
                ground_altitude_m=0.234567,
                description="launch pad",
            )
        },
        navigation_points={
            "checkpoint": NavigationPointSpec(
                "checkpoint",
                position_xyz_m=(-11.111111, 22.222222, 8.888888),
                description="visual checkpoint",
            )
        },
        default_takeoff_altitude_m=10,
        default_track_duration_s=30,
        search_timeout_s=60,
    )


def _variants(schema: dict[str, object]) -> dict[str, dict[str, object]]:
    variants = schema["properties"]["steps"]["items"]["oneOf"]
    return {
        variant["properties"]["skill"]["const"]: variant
        for variant in variants
    }


class SkillPlanDraftJsonSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = _world_context()
        self.limits = PlannerLimits(
            max_plan_steps=8,
            min_track_duration_s=2,
            max_track_duration_s=45,
        )
        self.schema = build_skill_plan_draft_json_schema(
            world_context=self.world,
            skill_catalog=build_default_skill_catalog(),
            limits=self.limits,
        )

    def test_top_level_and_step_variants_are_strict(self) -> None:
        self.assertEqual(self.schema["type"], "object")
        self.assertEqual(
            set(self.schema["properties"]),
            {"schema_version", "steps"},
        )
        self.assertEqual(self.schema["required"], ["schema_version", "steps"])
        self.assertFalse(self.schema["additionalProperties"])
        self.assertEqual(
            self.schema["properties"]["schema_version"],
            {"type": "integer", "const": 1},
        )
        steps = self.schema["properties"]["steps"]
        self.assertEqual(steps["minItems"], 2)
        self.assertEqual(steps["maxItems"], 8)

        variants = _variants(self.schema)
        self.assertEqual(
            list(variants),
            ["TAKEOFF", "GOTO", "SEARCH", "TRACK", "LAND"],
        )
        for variant in variants.values():
            self.assertEqual(variant["required"], ["id", "skill", "args"])
            self.assertFalse(variant["additionalProperties"])
            self.assertEqual(
                variant["properties"]["id"]["pattern"],
                "^[a-z][a-z0-9_]{0,31}$",
            )
            self.assertFalse(
                variant["properties"]["args"]["additionalProperties"]
            )

    def test_named_location_enums_have_no_geometry(self) -> None:
        variants = _variants(self.schema)
        goto_args = variants["GOTO"]["properties"]["args"]["properties"]
        search_args = variants["SEARCH"]["properties"]["args"]["properties"]
        land_args = variants["LAND"]["properties"]["args"]["properties"]

        self.assertEqual(
            goto_args["destination"]["enum"],
            ["checkpoint", "east_area", "home", "search_area"],
        )
        self.assertEqual(
            search_args["region"]["enum"],
            ["east_area", "search_area"],
        )
        self.assertEqual(land_args["zone"]["enum"], ["home"])

        serialized = json.dumps(self.schema, sort_keys=True, allow_nan=False)
        for forbidden in (
            "12.345678",
            "23.456789",
            "31.415926",
            "27.182818",
            "11.111111",
            "22.222222",
            "scene_min_xyz_m",
            "position_xyz_m",
            "oracle",
            "target_position",
            "target_velocity",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized.casefold())

    def test_track_and_recovery_constraints_match_policy_limits(self) -> None:
        variants = _variants(self.schema)
        track = variants["TRACK"]
        args = track["properties"]["args"]["properties"]
        self.assertEqual(args["duration_s"]["minimum"], 2.0)
        self.assertEqual(args["duration_s"]["maximum"], 45.0)
        self.assertEqual(
            args["target_ref"]["pattern"],
            r"^\$[a-z][a-z0-9_]{0,31}\.target_id$",
        )
        self.assertEqual(
            args["on_target_lost"]["enum"],
            ["REACQUIRE", "FAIL"],
        )

        recovery = track["properties"]["recovery"]
        self.assertEqual(recovery["properties"]["skill"]["const"], "REACQUIRE")
        attempts = recovery["properties"]["max_attempts"]
        self.assertEqual(attempts["minimum"], 1)
        self.assertEqual(attempts["maximum"], 2)
        self.assertFalse(recovery["additionalProperties"])

        for skill, variant in variants.items():
            if skill != "TRACK":
                self.assertNotIn("recovery", variant["properties"])

    def test_builder_returns_fresh_schema_and_validates_dependencies(self) -> None:
        first = build_skill_plan_draft_json_schema(
            world_context=self.world,
            skill_catalog=build_default_skill_catalog(),
            limits=self.limits,
        )
        second = build_skill_plan_draft_json_schema(
            world_context=self.world,
            skill_catalog=build_default_skill_catalog(),
            limits=self.limits,
        )
        first["properties"]["schema_version"]["const"] = 99
        self.assertEqual(second["properties"]["schema_version"]["const"], 1)

        with self.assertRaises(TypeError):
            build_skill_plan_draft_json_schema(
                world_context=object(),  # type: ignore[arg-type]
                skill_catalog=build_default_skill_catalog(),
                limits=self.limits,
            )


if __name__ == "__main__":
    unittest.main()
