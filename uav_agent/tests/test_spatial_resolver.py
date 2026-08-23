"""Coordinate-frame and relational grounding tests."""

from __future__ import annotations

import math
import unittest

from planner.spatial import (
    CircleRegion,
    CoordinateFrame,
    NamedLocationTarget,
    PointTarget,
    RectangleRegion,
    RelationalPointTarget,
    RelationalRegion,
    SpatialRelation,
)
from planner.spatial_resolver import (
    FramePose,
    MissingFramePoseError,
    SpatialResolutionError,
    SpatialResolver,
    UnresolvedSpatialReferenceError,
)


class SpatialResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = SpatialResolver(
            home_pose=FramePose((100, 200, 2), 1.2),
            uav_start_pose=FramePose((10, 20, 5), math.pi / 2),
            named_locations={"home": (100, 200, 2)},
            landmarks={"tower": (30, 40, 0)},
        )

    def test_world_and_home_enu_resolution(self) -> None:
        self.assertEqual(
            self.resolver.resolve_point(CoordinateFrame.WORLD_ENU, (1, 2, 3)),
            (1.0, 2.0, 3.0),
        )
        # HOME_ENU axes remain ENU; home yaw is intentionally ignored.
        self.assertEqual(
            self.resolver.resolve_point(CoordinateFrame.HOME_ENU, (1, 2, 3)),
            (101.0, 202.0, 5.0),
        )

    def test_uav_start_flu_uses_start_yaw(self) -> None:
        resolved = self.resolver.resolve_point(CoordinateFrame.UAV_START_FLU, (2, 3, 1))
        self.assertAlmostEqual(resolved[0], 7.0)
        self.assertAlmostEqual(resolved[1], 22.0)
        self.assertAlmostEqual(resolved[2], 6.0)

    def test_missing_runtime_frame_fails_closed(self) -> None:
        with self.assertRaises(MissingFramePoseError):
            self.resolver.resolve_point(CoordinateFrame.UAV_HOLD_FLU, (1, 0, 0))
        with self.assertRaises(MissingFramePoseError):
            self.resolver.resolve_point(CoordinateFrame.CAMERA_FLU, (1, 0, 0))

    def test_named_and_relational_targets_require_trusted_grounding(self) -> None:
        self.assertEqual(
            self.resolver.resolve_target(NamedLocationTarget("home")),
            PointTarget(CoordinateFrame.WORLD_ENU, (100, 200, 2)),
        )
        with self.assertRaises(UnresolvedSpatialReferenceError):
            self.resolver.resolve_target(NamedLocationTarget("unknown"))
        with self.assertRaises(UnresolvedSpatialReferenceError):
            self.resolver.resolve_target(
                RelationalPointTarget(SpatialRelation.NORTH_OF, "not_grounded", 5)
            )

    def test_ambiguous_left_without_frame_is_not_guessed(self) -> None:
        with self.assertRaises(SpatialResolutionError):
            self.resolver.resolve_target(
                RelationalPointTarget(SpatialRelation.LEFT_OF, "tower", 10)
            )
        resolved = self.resolver.resolve_target(
            RelationalPointTarget(
                SpatialRelation.LEFT_OF, "tower", 10, CoordinateFrame.UAV_START_FLU
            )
        )
        self.assertAlmostEqual(resolved.xyz_m[0], 20.0)
        self.assertAlmostEqual(resolved.xyz_m[1], 40.0)

    def test_resolves_rotated_regions_to_world(self) -> None:
        rectangle = self.resolver.resolve_region(
            RectangleRegion(CoordinateFrame.UAV_START_FLU, (5, 0, 0), 10, 4, 15)
        )
        self.assertIs(rectangle.frame, CoordinateFrame.WORLD_ENU)
        self.assertAlmostEqual(rectangle.center_xyz_m[0], 10.0)
        self.assertAlmostEqual(rectangle.center_xyz_m[1], 25.0)
        self.assertAlmostEqual(rectangle.yaw_deg, 105.0)
        circle = self.resolver.resolve_region(
            CircleRegion(CoordinateFrame.HOME_ENU, (5, -5, 0), 3)
        )
        self.assertEqual(circle.center_xyz_m, (105.0, 195.0, 2.0))

    def test_relational_region_only_executes_after_grounding(self) -> None:
        region = RelationalRegion(
            SpatialRelation.EAST_OF, "tower", 10, (20, 8)
        )
        resolved = self.resolver.resolve_region(region)
        self.assertIsInstance(resolved, RectangleRegion)
        self.assertEqual(resolved.center_xyz_m, (40.0, 40.0, 0.0))
        with self.assertRaises(UnresolvedSpatialReferenceError):
            self.resolver.resolve_region(
                RelationalRegion(SpatialRelation.EAST_OF, "car", 10, (20, 8))
            )


if __name__ == "__main__":
    unittest.main()
