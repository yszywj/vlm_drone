"""Spatial Contract V3 value and strict-schema tests."""

from __future__ import annotations

import json
import math
import unittest

from planner.json_schema_v3 import build_skill_plan_v3_json_schema
from planner.schemas_v3 import SkillPlanDraftV3, SpatialPlanValidationError
from planner.spatial import (
    CircleRegion,
    CoordinateFrame,
    CorridorRegion,
    NamedLocationTarget,
    PointTarget,
    PolygonRegion,
    RectangleRegion,
    RelationalPointTarget,
    RelationalRegion,
    RouteTarget,
    SectorRegion,
    SpatialAssumption,
    SpatialContractError,
    SpatialRelation,
    region_spec_from_dict,
    spatial_target_from_dict,
)


class SpatialTypesV3Tests(unittest.TestCase):
    def test_coordinate_frames_are_explicit_and_stable(self) -> None:
        self.assertEqual(
            [item.value for item in CoordinateFrame],
            ["WORLD_ENU", "HOME_ENU", "UAV_START_FLU", "UAV_HOLD_FLU", "CAMERA_FLU"],
        )

    def test_all_spatial_target_variants_round_trip(self) -> None:
        targets = (
            NamedLocationTarget("home"),
            PointTarget(CoordinateFrame.WORLD_ENU, (1, 2, 3)),
            RelationalPointTarget(SpatialRelation.LEFT_OF, "red_building", 10, CoordinateFrame.UAV_START_FLU),
            RouteTarget(CoordinateFrame.HOME_ENU, ((0, 0, 5), (10, 0, 5))),
        )
        for target in targets:
            with self.subTest(target=target):
                self.assertEqual(spatial_target_from_dict(target.to_dict()), target)

    def test_six_region_variants_round_trip(self) -> None:
        regions = (
            CircleRegion(CoordinateFrame.WORLD_ENU, (0, 0, 0), 5),
            RectangleRegion(CoordinateFrame.HOME_ENU, (2, 3, 0), 10, 6, 15),
            SectorRegion(CoordinateFrame.HOME_ENU, (0, 0, 0), 0, 60, (20, 50)),
            PolygonRegion(CoordinateFrame.WORLD_ENU, ((0, 0, 0), (4, 0, 0), (0, 3, 0))),
            CorridorRegion(CoordinateFrame.WORLD_ENU, ((0, 0, 0), (4, 0, 0), (8, 2, 0)), 2),
            RelationalRegion(SpatialRelation.RIGHT_OF, "tower", 12, (20, 10), CoordinateFrame.UAV_START_FLU),
        )
        for region in regions:
            with self.subTest(region=region):
                self.assertEqual(region_spec_from_dict(region.to_dict()), region)

    def test_rejects_bare_or_invalid_geometry(self) -> None:
        with self.assertRaises(SpatialContractError):
            spatial_target_from_dict({"kind": "POINT", "xyz_m": [1, 2, 3]})
        with self.assertRaises(SpatialContractError):
            PointTarget(CoordinateFrame.WORLD_ENU, (1, math.nan, 3))
        with self.assertRaises(SpatialContractError):
            PolygonRegion(CoordinateFrame.WORLD_ENU, ((0, 0, 0), (1, 0, 0), (2, 0, 0)))
        with self.assertRaises(SpatialContractError):
            region_spec_from_dict({
                "shape": "CIRCLE", "frame": "WORLD_ENU", "center_xyz_m": [0, 0, 0],
                "radius_m": 2, "silent_extra": True,
            })

    def test_assumption_is_strict_and_serializable(self) -> None:
        assumption = SpatialAssumption("左边", "UAV_START_FLU left", 0.71)
        self.assertEqual(SpatialAssumption.from_dict(assumption.to_dict()), assumption)
        with self.assertRaises(SpatialContractError):
            SpatialAssumption("left", "world west", 1.01)

    def test_plan_v3_round_trip_keeps_assumptions_and_routing(self) -> None:
        raw = {
            "schema_version": 3,
            "mission_id": "mission_spatial",
            "uav_id": "uav_1",
            "plan_version": 1,
            "assumptions": [
                {"source_text": "左边", "interpretation": "UAV_START_FLU left", "confidence": 0.71}
            ],
            "steps": [
                {"id": "takeoff_1", "uav_id": "uav_1", "skill": "TAKEOFF", "args": {"altitude_m": 10}},
                {
                    "id": "goto_1", "uav_id": "uav_1", "skill": "GOTO",
                    "args": {"target": {"kind": "POINT", "frame": "WORLD_ENU", "xyz_m": [25, 40, 10]}},
                },
            ],
        }
        parsed = SkillPlanDraftV3.from_dict(raw)
        self.assertEqual(parsed.assumptions[0].confidence, 0.71)
        self.assertEqual(SkillPlanDraftV3.from_dict(parsed.to_dict()), parsed)
        json.dumps(parsed.to_dict())

    def test_plan_v3_rejects_cross_uav_and_initial_inspect(self) -> None:
        base = {
            "schema_version": 3, "mission_id": "mission_spatial", "uav_id": "uav_1",
            "plan_version": 1, "assumptions": [],
        }
        with self.assertRaises(SpatialPlanValidationError):
            SkillPlanDraftV3.from_dict({**base, "steps": [
                {"id": "takeoff_1", "uav_id": "uav_2", "skill": "TAKEOFF", "args": {}},
                {"id": "land_1", "uav_id": "uav_1", "skill": "LAND", "args": {"zone": "home"}},
            ]})
        with self.assertRaises(SpatialPlanValidationError):
            SkillPlanDraftV3.from_dict({**base, "steps": [
                {"id": "takeoff_1", "uav_id": "uav_1", "skill": "TAKEOFF", "args": {}},
                {"id": "inspect_1", "uav_id": "uav_1", "skill": "INSPECT", "args": {"candidate_id": "candidate_1"}},
            ]})

    def test_json_schema_is_v3_routed_and_not_named_region_limited(self) -> None:
        schema = build_skill_plan_v3_json_schema(
            mission_id="mission_spatial", uav_id="uav_1", plan_version=2
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], 3)
        self.assertEqual(schema["properties"]["mission_id"]["const"], "mission_spatial")
        serialized = json.dumps(schema)
        self.assertIn('"POINT"', serialized)
        self.assertIn('"RECTANGLE"', serialized)
        self.assertIn('"CORRIDOR"', serialized)
        self.assertNotIn('"search_area"', serialized)
        self.assertNotIn("uniqueItems", serialized)


if __name__ == "__main__":
    unittest.main()
