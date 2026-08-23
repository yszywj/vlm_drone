from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from common.obstacle_types import ObstacleAABB, ObstacleMotionState, ObstacleSpec
from configs.loader import ConfigError, load_config
from env.moving_target import MovingTarget, TargetMotionMode
from env.obstacle_registry import ObstacleRegistry, obstacle_scene_prim_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _spec(
    obstacle_id: str = "box_red",
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.5),
) -> ObstacleSpec:
    return ObstacleSpec(
        obstacle_id=obstacle_id,
        center_xyz_m=center,
        size_xyz_m=(1.0, 1.0, 1.0),
        color_rgb=(0.8, 0.2, 0.1),
        collidable=True,
        motion_state=ObstacleMotionState.STATIC,
    )


class ObstacleTypesTest(unittest.TestCase):
    def test_spec_and_aabb_are_strict_immutable_shared_geometry(self) -> None:
        spec = _spec(center=(2.0, 3.0, 1.0))
        self.assertEqual(spec.aabb.min_xyz_m, (1.5, 2.5, 0.5))
        self.assertEqual(spec.aabb.max_xyz_m, (2.5, 3.5, 1.5))
        self.assertEqual(spec.aabb.center_xyz_m, spec.center_xyz_m)
        self.assertEqual(spec.aabb.size_xyz_m, spec.size_xyz_m)
        with self.assertRaises(FrozenInstanceError):
            spec.collidable = False  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            spec.aabb.min_xyz_m = (0.0, 0.0, 0.0)  # type: ignore[misc]

    def test_aabb_expansion_and_segment_intersection_are_exact(self) -> None:
        aabb = ObstacleAABB((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        expanded = aabb.expanded((0.2, 0.3, 0.4))
        self.assertEqual(expanded.min_xyz_m, (-1.2, -1.3, -1.4))
        self.assertEqual(expanded.max_xyz_m, (1.2, 1.3, 1.4))
        self.assertAlmostEqual(
            aabb.segment_intersection_fraction((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            0.25,
        )
        self.assertIsNone(
            aabb.segment_intersection_fraction((-2.0, 2.0, 0.0), (2.0, 2.0, 0.0))
        )

    def test_invalid_geometry_and_labels_are_rejected(self) -> None:
        for kwargs in (
            {"obstacle_id": "bad id"},
            {"size_xyz_m": (1.0, 0.0, 1.0)},
            {"color_rgb": (1.1, 0.0, 0.0)},
            {"collidable": 1},
            {"motion_state": "FLYING"},
        ):
            base = dict(
                obstacle_id="box_1",
                center_xyz_m=(0.0, 0.0, 1.0),
                size_xyz_m=(1.0, 1.0, 1.0),
                color_rgb=(1.0, 0.0, 0.0),
                collidable=True,
                motion_state="STATIC",
            )
            base.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                ObstacleSpec(**base)


class ObstacleRegistryTest(unittest.TestCase):
    def test_scene_prim_keys_are_stable_and_begin_with_a_usd_identifier(self) -> None:
        self.assertEqual(
            obstacle_scene_prim_key(0, "box_red"),
            "obstacle_000_box_red",
        )
        self.assertEqual(
            obstacle_scene_prim_key(12, "box-red.v2"),
            "obstacle_012_box-red.v2",
        )
        self.assertTrue(obstacle_scene_prim_key(0, "box_red")[0].isalpha())
        with self.assertRaises(TypeError):
            obstacle_scene_prim_key(True, "box_red")
        with self.assertRaises(ValueError):
            obstacle_scene_prim_key(-1, "box_red")

    def test_registry_preserves_order_and_supports_iteration_and_get(self) -> None:
        red = _spec("box_red")
        blue = _spec("box_blue", center=(3.0, 0.0, 0.5))
        registry = ObstacleRegistry((red, blue))
        self.assertEqual(tuple(registry), (red, blue))
        self.assertEqual(registry.specs, (red, blue))
        self.assertEqual(registry.ids, ("box_red", "box_blue"))
        self.assertEqual(registry.get("box_blue"), blue)
        self.assertEqual(registry.get_aabb("box_red"), red.aabb)
        self.assertIn("box_red", registry)
        self.assertEqual(len(registry.aabbs), 2)
        with self.assertRaises(KeyError):
            registry.get("box_missing")

    def test_duplicate_ids_and_non_specs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ObstacleRegistry((_spec(), _spec()))
        with self.assertRaises(TypeError):
            ObstacleRegistry((object(),))  # type: ignore[arg-type]

    def test_default_config_is_the_single_public_obstacle_source(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
        registry = ObstacleRegistry.from_scene_config(config.scene)
        self.assertEqual(registry.ids, ("box_red", "box_blue"))
        self.assertEqual(registry.get("box_red").center_xyz_m, (12.0, 4.0, 1.0))
        self.assertEqual(registry.get("box_blue").size_xyz_m, (3.0, 2.1, 3.0))
        self.assertEqual(config.obstacle_perception.mode, "disabled")
        self.assertEqual(config.obstacle_perception.max_distance_m, 40.0)
        self.assertEqual(config.obstacle_perception.min_bbox_area_px, 64)
        self.assertEqual(config.obstacle_perception.max_occlusion_ratio, 0.95)
        source = (PROJECT_ROOT / "env" / "scene.py").read_text(encoding="utf-8")
        self.assertIn("self.obstacle_registry", source)
        self.assertNotIn("Box_00", source)

    def test_loader_rejects_duplicate_out_of_scene_and_partial_obstacles(self) -> None:
        raw = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        )
        mutations = (
            lambda data: data["scene"]["obstacles"].append(
                dict(data["scene"]["obstacles"][0])
            ),
            lambda data: data["scene"]["obstacles"][0].__setitem__(
                "center_xyz_m", [1000.0, 0.0, 1.0]
            ),
            lambda data: data["scene"]["obstacles"][0].pop("color_rgb"),
            lambda data: data["scene"]["obstacles"][0].__setitem__("typo", 1),
        )
        for index, mutate in enumerate(mutations):
            candidate = yaml.safe_load(yaml.safe_dump(raw))
            mutate(candidate)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / f"bad_{index}.yaml"
                path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(ConfigError):
                    load_config(path)

    def test_obstacle_perception_limits_are_strictly_validated(self) -> None:
        raw = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        )
        mutations = (
            lambda data: data["obstacle_perception"].__setitem__("mode", "oracle"),
            lambda data: data["obstacle_perception"].__setitem__("max_distance_m", 0),
            lambda data: data["obstacle_perception"].__setitem__("min_bbox_area_px", True),
            lambda data: data["obstacle_perception"].__setitem__("max_occlusion_ratio", 1.0),
            lambda data: data["obstacle_perception"].__setitem__("unknown", 1),
        )
        for index, mutate in enumerate(mutations):
            candidate = yaml.safe_load(yaml.safe_dump(raw))
            mutate(candidate)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / f"bad_perception_{index}.yaml"
                path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(ConfigError):
                    load_config(path)


class MovingTargetObstacleTest(unittest.TestCase):
    def _make(self, *, mode: TargetMotionMode, seed: int = 17) -> MovingTarget:
        registry = ObstacleRegistry((_spec(center=(0.0, 0.0, 0.5)),))
        return MovingTarget(
            mode=mode,
            initial_position_xyz_m=(-2.0, 0.0, 0.5),
            bounds_min_xyz_m=(-5.0, -4.0, 0.5),
            bounds_max_xyz_m=(5.0, 4.0, 0.5),
            speed_mps=1.0,
            max_speed_mps=1.5,
            direction_change_interval_s=0.31,
            seed=seed,
            initial_heading_rad=0.0,
            obstacle_registry=registry,
            target_half_extent_xyz_m=(0.2, 0.2, 0.2),
        )

    def test_linear_target_reflects_without_tunnelling_through_expanded_aabb(self) -> None:
        target = self._make(mode=TargetMotionMode.LINEAR)
        pose = target.step(3.0)
        self.assertLess(pose.x, -0.7)
        self.assertLess(target.get_velocity()[0], 0.0)
        expanded = _spec().aabb.expanded((0.2, 0.2, 0.2))
        self.assertFalse(expanded.contains((pose.x, pose.y, pose.z), strict=True))

    def test_obstacle_avoiding_random_walk_is_reproducible_for_same_seed(self) -> None:
        first = self._make(mode=TargetMotionMode.RANDOM_WALK, seed=23)
        second = self._make(mode=TargetMotionMode.RANDOM_WALK, seed=23)
        expanded = _spec().aabb.expanded((0.2, 0.2, 0.2))
        for _ in range(2_000):
            first_pose = first.step(1.0 / 60.0)
            second_pose = second.step(1.0 / 60.0)
            self.assertEqual(first_pose, second_pose)
            self.assertFalse(
                expanded.contains(
                    (first_pose.x, first_pose.y, first_pose.z), strict=True
                )
            )

    def test_reset_into_expanded_obstacle_is_rejected_atomically(self) -> None:
        target = self._make(mode=TargetMotionMode.LINEAR)
        before = target.get_pose()
        with self.assertRaisesRegex(ValueError, "intersects"):
            target.reset(position_m=(0.0, 0.0, 0.5))
        self.assertEqual(target.get_pose(), before)


if __name__ == "__main__":
    unittest.main()
