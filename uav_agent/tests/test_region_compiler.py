"""SEARCH V3 region entry and macro-geometry tests."""

from __future__ import annotations

import unittest

from planner.region_compiler import RegionCompilationError, RegionCompiler
from planner.spatial import (
    CircleRegion,
    CoordinateFrame,
    CorridorRegion,
    PolygonRegion,
    RectangleAnchor,
    RectangleRegion,
    SectorRegion,
)
from skills.search_geometry import point_inside_region
from skills.search_strategy import (
    SearchEntryPolicy,
    SearchRuntimeCapabilities,
    SearchStrategySpec,
    SearchStrategyType,
)


class RegionCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = RegionCompiler()
        self.circle = CircleRegion(CoordinateFrame.WORLD_ENU, (0, 0, 0), 10)
        self.rectangle = RectangleRegion(CoordinateFrame.WORLD_ENU, (10, 5, 0), 20, 10)

    def compile(self, region, kind, **kwargs):
        return self.compiler.compile(
            region=region,
            strategy=SearchStrategySpec(kind, spacing_m=3, max_viewpoints=24, **kwargs),
            entry_policy=SearchEntryPolicy.NEAREST_POINT,
            current_uav_xyz_m=(-10, -10, 8),
            search_altitude_m=8,
        )

    def test_perimeter_v1_preserves_exact_six_point_baseline(self) -> None:
        result = self.compile(self.circle, SearchStrategyType.PERIMETER_V1)
        self.assertEqual(len(result.observation_waypoints_xyz_m), 6)
        self.assertAlmostEqual(result.observation_waypoints_xyz_m[0][0], 10 * 3**0.5 / 2)
        self.assertAlmostEqual(result.observation_waypoints_xyz_m[0][1], 5)

    def test_at_least_five_region_shapes_generate_geometry(self) -> None:
        examples = (
            (self.circle, SearchStrategyType.LAWNMOWER),
            (self.rectangle, SearchStrategyType.SPIRAL_OUT),
            (PolygonRegion(CoordinateFrame.WORLD_ENU, ((0, 0, 0), (12, 0, 0), (8, 8, 0), (0, 6, 0))), SearchStrategyType.PERIMETER),
            (SectorRegion(CoordinateFrame.WORLD_ENU, (0, 0, 0), 0, 60, (2, 15)), SearchStrategyType.SECTOR_SWEEP),
            (CorridorRegion(CoordinateFrame.WORLD_ENU, ((0, 0, 0), (10, 0, 0), (15, 5, 0)), 2), SearchStrategyType.CORRIDOR_FOLLOW),
        )
        for region, kind in examples:
            with self.subTest(shape=region.shape, strategy=kind):
                result = self.compile(region, kind)
                self.assertTrue(result.observation_waypoints_xyz_m)
                self.assertTrue(
                    all(point_inside_region(result.region, point) for point in result.observation_waypoints_xyz_m)
                )

    def test_start_in_place_inside_keeps_current_xy(self) -> None:
        result = self.compiler.compile(
            region=self.rectangle,
            strategy=SearchStrategySpec(SearchStrategyType.LAWNMOWER, spacing_m=2),
            entry_policy=SearchEntryPolicy.START_IN_PLACE_IF_INSIDE,
            current_uav_xyz_m=(10, 5, 4),
            search_altitude_m=8,
        )
        self.assertEqual(result.entry_point_xyz_m, (10.0, 5.0, 8.0))
        self.assertEqual(result.route_waypoints_xyz_m[0], (10.0, 5.0, 8.0))

    def test_rectangle_supports_all_named_anchors(self) -> None:
        for anchor in RectangleAnchor:
            if anchor is RectangleAnchor.ENTRY_POINT:
                continue
            with self.subTest(anchor=anchor):
                point = self.compiler.rectangle_anchor(
                    self.rectangle, anchor, search_altitude_m=8
                )
                self.assertTrue(point_inside_region(self.rectangle, point))

    def test_user_and_model_entry_must_be_explicit_and_inside(self) -> None:
        strategy = SearchStrategySpec(SearchStrategyType.PERIMETER)
        with self.assertRaises(RegionCompilationError):
            self.compiler.compile(
                region=self.circle, strategy=strategy,
                entry_policy=SearchEntryPolicy.USER_ANCHOR,
                current_uav_xyz_m=(0, 0, 8), search_altitude_m=8,
            )
        with self.assertRaises(RegionCompilationError):
            self.compiler.compile(
                region=self.circle, strategy=strategy,
                entry_policy=SearchEntryPolicy.MODEL_SELECTED,
                current_uav_xyz_m=(0, 0, 8), search_altitude_m=8,
                model_selected_entry_xyz_m=(100, 100, 8),
            )

    def test_model_waypoints_are_validated_against_region(self) -> None:
        with self.assertRaises(RegionCompilationError):
            self.compile(
                self.circle,
                SearchStrategyType.MODEL_WAYPOINTS,
                model_waypoints_xyz_m=((0, 0, 8), (100, 100, 8)),
            )

    def test_adaptive_strategy_requires_runtime_provider(self) -> None:
        with self.assertRaisesRegex(RegionCompilationError, "next-best-view"):
            self.compile(self.circle, SearchStrategyType.ADAPTIVE_NEXT_BEST_VIEW)

    def test_adaptive_strategy_compiles_seed_after_capability_negotiation(self) -> None:
        compiler = RegionCompiler(
            search_runtime_capabilities=SearchRuntimeCapabilities(
                adaptive_next_best_view=True
            )
        )
        result = compiler.compile(
            region=self.circle,
            strategy=SearchStrategySpec(
                SearchStrategyType.ADAPTIVE_NEXT_BEST_VIEW,
                max_viewpoints=4,
            ),
            entry_policy=SearchEntryPolicy.START_IN_PLACE_IF_INSIDE,
            current_uav_xyz_m=(1, 2, 8),
            search_altitude_m=8,
        )
        self.assertEqual(result.observation_waypoints_xyz_m, ((1.0, 2.0, 8.0),))
        self.assertEqual(result.route_waypoints_xyz_m, ((1.0, 2.0, 8.0),))
        with self.assertRaisesRegex(RegionCompilationError, "outside"):
            compiler.validate_adaptive_waypoint(
                result.region,
                (100, 100, 8),
                search_altitude_m=8,
            )


if __name__ == "__main__":
    unittest.main()
