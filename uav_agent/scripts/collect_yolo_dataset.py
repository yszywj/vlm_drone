#!/usr/bin/env python3
"""Collect a bounded YOLO dataset from Isaac truth (never for agent runtime)."""

from __future__ import annotations

import argparse
from dataclasses import replace
from math import ceil, cos, isfinite, pi, radians, sin, tan
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Protocol, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = Path(
    os.environ.get("UAV_AGENT_CONDA_ENV", "/home/amax/miniconda3/envs/r_isaac_sim")
).expanduser()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import ConfigError, load_config  # noqa: E402
from training.yolo.isaac_collector import (  # noqa: E402
    CollectionLimits,
    EpisodeRandomization,
    IsaacDatasetCollectionError,
    IsaacYoloDatasetCollector,
    OracleObjectTruth,
    OracleFrameTruth,
    RandomizationBounds,
    estimate_depth_visibility,
    require_oracle_label_acknowledgements,
)
from training.yolo.collection_scene import (  # noqa: E402
    CUBE_CLASS_ID,
    CUBE_CLASS_NAME,
    CollectionSceneObject,
    CubeCollectionProtocol,
    build_cube_v1_scene_inventory,
    load_cube_collection_protocol,
    transformed_local_bounds_corners,
    validate_scene_inventory,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="unified Isaac scene YAML")
    parser.add_argument(
        "--collection-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "yolo" / "collect_cube.yaml",
        help="strict public cube-v1 collection protocol",
    )
    parser.add_argument("--output", required=True, help="new/replaceable dataset directory")
    parser.add_argument("--scene-seed", type=_nonnegative_int, default=42)
    parser.add_argument("--max-samples", type=_positive_int, default=2_000)
    parser.add_argument("--max-episodes", type=_positive_int, default=100)
    parser.add_argument("--frames-per-episode", type=_positive_int, default=20)
    parser.add_argument("--sample-hz", type=_positive_float, default=2.0)
    parser.add_argument("--min-bbox-area-px", type=_positive_float, default=16.0)
    parser.add_argument("--class-id", type=_nonnegative_int, default=CUBE_CLASS_ID)
    parser.add_argument(
        "--class-name",
        default=CUBE_CLASS_NAME,
        help="closed-set detector class; cube-v1 requires exactly 'cube'",
    )
    parser.add_argument(
        "--oracle-label-generation",
        action="store_true",
        help="declare that Oracle truth is used only to generate labels",
    )
    parser.add_argument(
        "--acknowledge-privileged-oracle",
        action="store_true",
        help="acknowledge that simulator ground truth is privileged",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate flags/config without importing or starting Isaac Sim",
    )
    parser.add_argument(
        "--gpu-device",
        type=_nonnegative_int,
        default=0,
        help="physical GPU index for Isaac rendering and physics (default: 0)",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser.parse_args(argv)


class _CollectionSceneDriver(Protocol):
    """Late-bound rendered-scene operations used by the formal adapter."""

    def install(self, objects: Sequence[CollectionSceneObject]) -> None: ...

    def update_pose(self, obj: CollectionSceneObject) -> None: ...

    def rendered_geometry(
        self,
        obj: CollectionSceneObject,
    ) -> tuple[CollectionSceneObject, np.ndarray]: ...


class _UsdCubeV1SceneDriver:
    """Render the complete cube-v1 inventory under one reusable USD scope.

    All Isaac/USD imports are deliberately inside methods reached only after
    ``SimulationApp`` exists.  The legacy scene target is always hidden, so it
    can never become an unenumerated positive cube in a collected image.
    """

    _ROOT_PATH = "/World/CubeV1Collection"
    _COLOR_RGB = {
        "red": (0.92, 0.18, 0.08),
        "blue": (0.08, 0.28, 0.92),
        "green": (0.08, 0.78, 0.28),
        "yellow": (0.92, 0.72, 0.08),
        "gray": (0.48, 0.48, 0.48),
    }

    def __init__(self, environment: object) -> None:
        self._environment = environment
        self._roots: dict[str, object] = {}
        self._active_ids: set[str] = set()

    def install(self, objects: Sequence[CollectionSceneObject]) -> None:
        from pxr import UsdGeom

        validate_scene_inventory(objects)
        self._hide_legacy_target()
        self._hide_mission_obstacles()
        for root in self._roots.values():
            UsdGeom.Imageable(root).MakeInvisible()
        active: set[str] = set()
        for obj in objects:
            root = self._ensure_object(obj)
            self._set_pose_and_appearance(root, obj)
            UsdGeom.Imageable(root).MakeVisible()
            active.add(obj.object_id)
        self._active_ids = active

    def update_pose(self, obj: CollectionSceneObject) -> None:
        if obj.object_id not in self._active_ids:
            raise RuntimeError(f"collection object is not active: {obj.object_id}")
        root = self._roots[obj.object_id]
        self._set_pose_and_appearance(root, obj)

    def rendered_geometry(
        self,
        obj: CollectionSceneObject,
    ) -> tuple[CollectionSceneObject, np.ndarray]:
        """Read local USD bounds and transform their eight oriented corners."""

        from pxr import Gf, Usd, UsdGeom

        if obj.object_id not in self._active_ids:
            raise RuntimeError(f"collection object is not active: {obj.object_id}")
        root = self._roots[obj.object_id]
        body = root.GetStage().GetPrimAtPath(f"{root.GetPath()}/Body")
        if not body.IsValid() or not body.IsA(UsdGeom.Boundable):
            raise RuntimeError(
                f"collection object Body is not boundable: {obj.object_id}"
            )
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        # The root owns the randomized translate/rotate/scale while Body owns
        # the unit primitive.  ComputeLocalBound(root) already folds the root
        # transform into an aligned descendant box; multiplying that result by
        # root-to-world again applies the pose twice and loses orientation.
        # Start from Body's untransformed bound, then apply its complete world
        # transform exactly once.
        local_bound = cache.ComputeUntransformedBound(body)
        local_range = local_bound.GetRange()
        local_matrix = local_bound.GetMatrix()
        world_matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            body
        )

        def transform(point: tuple[float, float, float]) -> tuple[float, float, float]:
            local = local_matrix.Transform(Gf.Vec3d(*point))
            world = world_matrix.Transform(local)
            return tuple(float(value) for value in world)

        corners, dimensions = transformed_local_bounds_corners(
            tuple(float(value) for value in local_range.GetMin()),
            tuple(float(value) for value in local_range.GetMax()),
            transform,
        )
        center = tuple(float(value) for value in np.mean(corners, axis=0))
        rendered = replace(
            obj,
            position_world_m=center,
            dimensions_xyz_m=dimensions,
        )
        return rendered, corners

    def _stage(self) -> object:
        scene = getattr(self._environment, "scene", None)
        target = getattr(scene, "target", None)
        prim = getattr(target, "prim", None)
        if prim is None:
            raise RuntimeError("Isaac collection scene target prim is unavailable")
        return prim.GetStage()

    def _hide_legacy_target(self) -> None:
        from pxr import UsdGeom

        scene = getattr(self._environment, "scene", None)
        target = getattr(scene, "target", None)
        prim = getattr(target, "prim", None)
        if prim is None:
            raise RuntimeError("Isaac legacy target prim is unavailable")
        UsdGeom.Imageable(prim).MakeInvisible()

    def _hide_mission_obstacles(self) -> None:
        """Exclude non-protocol mission geometry from cube-v1 RGB frames."""

        from pxr import UsdGeom

        obstacle_root = self._stage().GetPrimAtPath("/World/Obstacles")
        if obstacle_root.IsValid():
            # cube-v1 supplies its own complete, metadata-enumerated hard
            # negatives.  Mission obstacles are neither part of that catalog
            # nor safe to leave as unrecorded occluders/near-cube examples.
            UsdGeom.Imageable(obstacle_root).MakeInvisible()

    def _ensure_object(self, obj: CollectionSceneObject) -> object:
        from pxr import UsdGeom

        existing = self._roots.get(obj.object_id)
        if existing is not None:
            return existing
        stage = self._stage()
        safe_name = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in obj.object_id
        )
        root_path = f"{self._ROOT_PATH}/{safe_name}"
        root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        body_path = f"{root_path}/Body"
        if obj.shape == "sphere":
            body = UsdGeom.Sphere.Define(stage, body_path)
            body.CreateRadiusAttr(0.5)
        elif obj.shape == "cylinder":
            body = UsdGeom.Cylinder.Define(stage, body_path)
            body.CreateRadiusAttr(0.5)
            body.CreateHeightAttr(1.0)
            body.CreateAxisAttr(UsdGeom.Tokens.z)
        else:
            body = UsdGeom.Cube.Define(stage, body_path)
            body.CreateSizeAttr(1.0)
        self._roots[obj.object_id] = root
        return root

    def _set_pose_and_appearance(
        self,
        root: object,
        obj: CollectionSceneObject,
    ) -> None:
        from math import atan2, degrees
        from pxr import Gf, UsdGeom

        xform = UsdGeom.XformCommonAPI(root)
        xform.SetTranslate(Gf.Vec3d(*obj.position_world_m))
        w, _x, _y, z = obj.orientation_world_wxyz
        yaw_deg = degrees(2.0 * atan2(float(z), float(w)))
        xform.SetRotate(
            Gf.Vec3f(0.0, 0.0, yaw_deg),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        xform.SetScale(Gf.Vec3f(*obj.dimensions_xyz_m))
        body = root.GetStage().GetPrimAtPath(f"{root.GetPath()}/Body")
        gprim = UsdGeom.Gprim(body)
        color = self._COLOR_RGB[obj.color_name]
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


class _SimpleSceneCollectionAdapter:
    """Formal cube-v1 adapter for the repository's simple Isaac scene.

    A reusable collection-only USD scope replaces the hidden legacy target.
    The inventory is protocol-planned (0--3 cubes plus seven hard negatives),
    and every rendered object is enumerated from its real oriented USD bounds.
    Cube motion is integrated at physics rate with a bounded turn rate;
    direction targets change only at the configured episode interval.
    """

    def __init__(
        self,
        environment: object,
        simulation_app: object,
        config: object,
        *,
        protocol: CubeCollectionProtocol | None = None,
        scene_driver: _CollectionSceneDriver | None = None,
        crossing_trajectories: bool = False,
    ) -> None:
        self._environment = environment
        self._simulation_app = simulation_app
        self._config = config
        self._protocol = protocol or load_cube_collection_protocol(
            PROJECT_ROOT / "configs" / "yolo" / "collect_cube.yaml"
        )
        self._scene_driver = scene_driver
        if not isinstance(crossing_trajectories, bool):
            raise TypeError("crossing_trajectories must be bool")
        self._crossing_trajectories = crossing_trajectories
        self._scene_objects: tuple[CollectionSceneObject, ...] = ()
        self._object_velocities_mps: dict[str, tuple[float, float, float]] = {}
        self._plan: EpisodeRandomization | None = None
        self._target_position = np.zeros(3, dtype=np.float64)
        self._target_heading_rad = 0.0
        self._desired_heading_rad = 0.0
        self._target_scale = 1.0
        self._elapsed_s = 0.0
        self._next_turn_s = float("inf")
        self._rng = random.Random(0)
        self._sample_barrier_timestamp_s: float | None = None

    def begin_episode(self, randomization: EpisodeRandomization) -> None:
        self._plan = randomization
        self._sample_barrier_timestamp_s = None
        self._rng = random.Random(
            (randomization.key.scene_seed << 32)
            ^ int(randomization.key.episode_id.rsplit("_", 1)[-1])
        )
        self._environment.reset(target_seed=randomization.key.scene_seed)
        uav_yaw = radians(randomization.uav_yaw_deg)
        self._environment.set_uav_pose(
            randomization.uav_position_world_m,
            (cos(uav_yaw / 2.0), 0.0, 0.0, sin(uav_yaw / 2.0)),
        )
        velocity_heading = uav_yaw
        self._environment.set_uav_velocity(
            (
                randomization.uav_speed_mps * cos(velocity_heading),
                randomization.uav_speed_mps * sin(velocity_heading),
                0.0,
            )
        )
        self._target_position = np.asarray(
            randomization.target_position_world_m,
            dtype=np.float64,
        )
        self._target_heading_rad = radians(randomization.target_heading_deg)
        self._desired_heading_rad = self._target_heading_rad
        self._target_scale = randomization.target_scale
        self._elapsed_s = 0.0
        self._next_turn_s = randomization.target_direction_change_interval_s

        # Freeze the scene's legacy random-walk controller.  This collection
        # adapter supplies the smooth, turn-rate-limited trajectory instead.
        # The legacy target is hidden and must remain at its configured safe
        # spawn; resetting it to the randomized collection anchor would apply
        # mission obstacle checks to an object that is not part of the dataset.
        target_motion = self._environment._require_target_motion()
        target_motion.reset(
            seed=randomization.key.scene_seed,
            mode="STATIC",
        )
        try:
            episode_index = int(randomization.key.episode_id.rsplit("_", 1)[-1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(
                "cube-v1 episode_id must end in a numeric episode index"
            ) from exc
        inventory = build_cube_v1_scene_inventory(
            self._protocol,
            scene_seed=randomization.key.scene_seed,
            episode_index=episode_index,
            sample_kind=randomization.sample_kind,
            anchor_position_world_m=self._target_position,
            target_scale=self._target_scale,
        )
        if randomization.sample_kind == "partial_occlusion":
            inventory = self._place_partial_noncube_occluder(inventory, randomization)
        self._scene_objects = inventory
        cubes = [item for item in inventory if item.shape == CUBE_CLASS_NAME]
        if self._crossing_trajectories and len(cubes) >= 2:
            crossing_axis = (
                np.asarray(cubes[1].position_world_m, dtype=np.float64)
                - np.asarray(cubes[0].position_world_m, dtype=np.float64)
            )
            crossing_axis[2] = 0.0
            crossing_axis /= max(float(np.linalg.norm(crossing_axis)), 1e-12)
            self._target_heading_rad = float(
                np.arctan2(crossing_axis[1], crossing_axis[0])
            )
            self._desired_heading_rad = self._target_heading_rad
            separation_m = float(
                np.linalg.norm(
                    np.asarray(cubes[1].position_world_m, dtype=np.float64)
                    - np.asarray(cubes[0].position_world_m, dtype=np.float64)
                )
            )
            crossing_time_s = separation_m / max(
                2.0 * randomization.target_speed_mps,
                1e-6,
            )
            self._next_turn_s = max(self._next_turn_s, crossing_time_s + 0.5)
        initial_velocity = randomization.target_speed_mps * np.asarray(
            [cos(self._target_heading_rad), sin(self._target_heading_rad), 0.0],
            dtype=np.float64,
        )
        self._object_velocities_mps = {
            item.object_id: (0.0, 0.0, 0.0) for item in inventory
        }
        for slot, cube in enumerate(cubes):
            velocity = initial_velocity.copy()
            if self._crossing_trajectories and slot == 1:
                velocity *= -1.0
            elif self._crossing_trajectories and slot >= 2:
                velocity = np.asarray(
                    [-initial_velocity[1], initial_velocity[0], 0.0],
                    dtype=np.float64,
                )
            self._object_velocities_mps[cube.object_id] = tuple(
                float(value) for value in velocity
            )
        self._require_scene_driver().install(inventory)
        self._apply_render_randomization(randomization)
        self._set_camera_view(
            pitch_deg=randomization.camera_pitch_deg,
            horizontal_fov_deg=randomization.camera_horizontal_fov_deg,
        )

    def advance_to_next_sample(self, sample_period_s: float) -> None:
        plan = self._require_plan()
        dt_s = float(self._config.simulation.physics_dt_s)
        steps = max(1, ceil(sample_period_s / dt_s))
        new_frame = False
        for _ in range(steps):
            self._advance_smooth_target(dt_s, plan)
            # A frame from an earlier step cannot be paired with the USD
            # geometry at the end of this sampling interval.  Only the final
            # dynamics barrier determines whether capture is ready.
            new_frame = bool(self._environment.step())
        # Annotators often need extra render frames after an episode reset.
        warmup_limit = ceil(
            (1.0 / float(self._config.camera.frequency_hz)) / dt_s
        ) + 4
        warmup_steps = 0
        while (
            not new_frame
            and self._simulation_app.is_running()
            and warmup_steps < warmup_limit
        ):
            self._advance_smooth_target(dt_s, plan)
            new_frame = bool(self._environment.step())
            warmup_steps += 1
        if not new_frame:
            raise RuntimeError("Isaac Camera did not produce a synchronized RGB-D sample")

        # Record the exact world/Camera publication barrier.  The fleet
        # environment already enforces this invariant; repeating the check at
        # the privileged-label boundary prevents stale RGB-D from ever being
        # combined with newer Oracle geometry after future adapter changes.
        sample = self._environment.get_camera_sample()
        world_timestamp_s = self._current_world_timestamp_s()
        self._validate_sample_world_barrier(sample, world_timestamp_s)
        self._sample_barrier_timestamp_s = world_timestamp_s

    def capture_oracle_frame(self, frame_id: str) -> OracleFrameTruth:
        del frame_id
        self._require_plan()
        sample = self._environment.get_camera_sample()
        world_timestamp_s = self._current_world_timestamp_s()
        expected_timestamp_s = getattr(self, "_sample_barrier_timestamp_s", None)
        if expected_timestamp_s is None:
            raise RuntimeError(
                "advance_to_next_sample() must publish a fresh Camera frame before capture"
            )
        tolerance_s = self._camera_barrier_tolerance_s()
        if abs(world_timestamp_s - expected_timestamp_s) > tolerance_s:
            raise RuntimeError(
                "Isaac world advanced after the synchronized Camera sample and before "
                "Oracle geometry capture"
            )
        self._validate_sample_world_barrier(sample, world_timestamp_s)
        driver = self._require_scene_driver()
        objects: list[OracleObjectTruth] = []
        for planned in self._scene_objects:
            rendered, corners = driver.rendered_geometry(planned)
            projection = self._environment.world_to_image(corners)
            center_projection = self._environment.world_to_image(
                np.asarray([rendered.position_world_m], dtype=np.float64)
            )
            center_pixels = np.asarray(center_projection.pixels_uv, dtype=np.float64)
            center_pixel = (
                tuple(float(value) for value in center_pixels[0])
                if center_pixels.shape == (1, 2)
                else tuple(
                    float(value)
                    for value in np.mean(projection.pixels_uv, axis=0)
                )
            )
            # RGB and depth come from this same atomic CameraSample.  A box
            # projection alone cannot prove visibility; depth fails closed.
            depth_visibility = estimate_depth_visibility(
                sample,
                projection.pixels_uv,
                projection.depth_m,
            )
            objects.append(
                OracleObjectTruth(
                    object_id=rendered.object_id,
                    shape=rendered.shape,
                    color_name=rendered.color_name,
                    position_world_m=rendered.position_world_m,
                    orientation_world_wxyz=rendered.orientation_world_wxyz,
                    dimensions_xyz_m=rendered.dimensions_xyz_m,
                    projected_pixels_uv=projection.pixels_uv,
                    projected_depth_m=projection.depth_m,
                    velocity_world_mps=self._object_velocities_mps.get(
                        rendered.object_id,
                        (0.0, 0.0, 0.0),
                    ),
                    center_pixel_uv=center_pixel,
                    occlusion_ratio=depth_visibility.occlusion_ratio,
                )
            )
        return OracleFrameTruth(
            camera_sample=sample,
            objects=tuple(objects),
        )

    def _current_world_timestamp_s(self) -> float:
        world = getattr(self._environment, "world", None)
        timestamp = getattr(world, "current_time", None)
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float, np.number))
            or not isfinite(float(timestamp))
            or float(timestamp) < 0.0
        ):
            raise RuntimeError(
                "Isaac collection requires a finite non-negative World.current_time"
            )
        return float(timestamp)

    def _camera_barrier_tolerance_s(self) -> float:
        rendering_dt_s = getattr(self._config.simulation, "rendering_dt_s", None)
        if (
            isinstance(rendering_dt_s, bool)
            or not isinstance(rendering_dt_s, (int, float, np.number))
            or not isfinite(float(rendering_dt_s))
            or float(rendering_dt_s) <= 0.0
        ):
            rendering_dt_s = float(self._config.simulation.physics_dt_s)
        return max(1e-7, float(rendering_dt_s) * 1e-5)

    def _validate_sample_world_barrier(
        self,
        sample: object,
        world_timestamp_s: float,
    ) -> None:
        sample_timestamp_s = getattr(sample, "timestamp_s", None)
        if (
            isinstance(sample_timestamp_s, bool)
            or not isinstance(sample_timestamp_s, (int, float, np.number))
            or not isfinite(float(sample_timestamp_s))
        ):
            raise RuntimeError("Isaac Camera sample has an invalid timestamp")
        lag_s = float(world_timestamp_s) - float(sample_timestamp_s)
        tolerance_s = self._camera_barrier_tolerance_s()
        if abs(lag_s) > tolerance_s:
            raise RuntimeError(
                "Isaac collection Camera sample is outside the Oracle geometry "
                "barrier: "
                f"world={world_timestamp_s!r}, camera={float(sample_timestamp_s)!r}, "
                f"lag={lag_s!r}, allowed_absolute_error={tolerance_s!r}"
            )

    def _advance_smooth_target(
        self,
        dt_s: float,
        plan: EpisodeRandomization,
    ) -> None:
        self._elapsed_s += dt_s
        if self._elapsed_s >= self._next_turn_s:
            self._desired_heading_rad = self._rng.uniform(-pi, pi)
            self._next_turn_s += plan.target_direction_change_interval_s
        heading_error = (self._desired_heading_rad - self._target_heading_rad + pi) % (
            2.0 * pi
        ) - pi
        maximum_turn = radians(plan.target_max_turn_rate_deg_s) * dt_s
        self._target_heading_rad += max(-maximum_turn, min(maximum_turn, heading_error))
        velocity = plan.target_speed_mps * np.asarray(
            [cos(self._target_heading_rad), sin(self._target_heading_rad), 0.0]
        )
        candidate = self._target_position + velocity * dt_s
        region = self._config.target.motion.region
        low = np.asarray(region.min_xyz_m, dtype=np.float64)
        high = np.asarray(region.max_xyz_m, dtype=np.float64)
        for axis in (0, 1):
            if candidate[axis] < low[axis] or candidate[axis] > high[axis]:
                self._target_heading_rad = (
                    pi - self._target_heading_rad
                    if axis == 0
                    else -self._target_heading_rad
                )
                candidate[axis] = np.clip(candidate[axis], low[axis], high[axis])
        candidate[2] = np.clip(candidate[2], low[2], high[2])
        delta = candidate - self._target_position
        orientation = (
            cos(self._target_heading_rad / 2.0),
            0.0,
            0.0,
            sin(self._target_heading_rad / 2.0),
        )
        updated: list[CollectionSceneObject] = []
        driver = self._require_scene_driver()
        cube_slot = 0
        for obj in self._scene_objects:
            if obj.shape != CUBE_CLASS_NAME:
                updated.append(obj)
                continue
            if not self._crossing_trajectories or cube_slot == 0:
                moved_position = (
                    np.asarray(obj.position_world_m, dtype=np.float64) + delta
                )
                moved_velocity = delta / dt_s
                moved_orientation = orientation
            else:
                moved_velocity = np.asarray(
                    self._object_velocities_mps.get(obj.object_id, (0.0, 0.0, 0.0)),
                    dtype=np.float64,
                )
                moved_position = (
                    np.asarray(obj.position_world_m, dtype=np.float64)
                    + moved_velocity * dt_s
                )
                for axis in (0, 1):
                    if moved_position[axis] < low[axis] or moved_position[axis] > high[axis]:
                        moved_velocity[axis] *= -1.0
                        moved_position[axis] = np.clip(
                            moved_position[axis], low[axis], high[axis]
                        )
                moved_position[2] = np.clip(moved_position[2], low[2], high[2])
                moved_heading = float(np.arctan2(moved_velocity[1], moved_velocity[0]))
                moved_orientation = (
                    cos(moved_heading / 2.0),
                    0.0,
                    0.0,
                    sin(moved_heading / 2.0),
                )
            moved = replace(
                obj,
                position_world_m=tuple(
                    float(value) for value in moved_position
                ),
                orientation_world_wxyz=moved_orientation,
            )
            driver.update_pose(moved)
            updated.append(moved)
            self._object_velocities_mps[obj.object_id] = tuple(
                float(value) for value in moved_velocity
            )
            cube_slot += 1
        self._scene_objects = tuple(updated)
        self._target_position = candidate

    def _set_camera_view(self, *, pitch_deg: float, horizontal_fov_deg: float) -> None:
        sensor = self._environment.scene.camera_sensor
        camera = sensor.camera
        pitch_rotation_rad = radians(-pitch_deg)
        camera.set_local_pose(
            orientation=np.asarray(
                [cos(pitch_rotation_rad / 2.0), 0.0, sin(pitch_rotation_rad / 2.0), 0.0]
            ),
            camera_axes="world",
        )
        aperture = float(camera.get_horizontal_aperture())
        camera.set_focal_length(aperture / (2.0 * tan(radians(horizontal_fov_deg) / 2.0)))
        sensor.invalidate_frame()

    def _place_partial_noncube_occluder(
        self,
        objects: Sequence[CollectionSceneObject],
        plan: EpisodeRandomization,
    ) -> tuple[CollectionSceneObject, ...]:
        """Move the catalogued non-cube screen between Camera and a cube."""

        cubes = tuple(obj for obj in objects if obj.shape == CUBE_CLASS_NAME)
        if not cubes:
            raise RuntimeError("partial_occlusion scene must contain a cube")
        uav = np.asarray(plan.uav_position_world_m, dtype=np.float64)
        target = np.asarray(cubes[0].position_world_m, dtype=np.float64)
        direction = target - uav
        norm_xy = float(np.linalg.norm(direction[:2]))
        if norm_xy <= 1e-6:
            direction[:2] = (1.0, 0.0)
            norm_xy = 1.0
        unit_xy = direction[:2] / norm_xy
        center = target.copy()
        center[:2] -= 0.8 * unit_xy
        yaw = float(np.arctan2(unit_xy[1], unit_xy[0]))
        adjusted = tuple(
            replace(
                obj,
                position_world_m=tuple(float(value) for value in center),
                orientation_world_wxyz=(cos(yaw / 2.0), 0.0, 0.0, sin(yaw / 2.0)),
                dimensions_xyz_m=(
                    0.12,
                    float(0.60 * self._target_scale),
                    float(1.15 * self._target_scale),
                ),
            )
            if obj.shape == "partial_noncube"
            else obj
            for obj in objects
        )
        validate_scene_inventory(adjusted)
        return adjusted

    def _apply_render_randomization(self, plan: EpisodeRandomization) -> None:
        """Apply bounded lighting, background/material, and blur variants."""

        import carb
        import carb.settings
        from pxr import Gf, UsdLux

        stage = self._environment.scene.target.prim.GetStage()
        dome = UsdLux.DomeLight.Get(stage, "/World/Lights/Dome")
        sun = UsdLux.DistantLight.Get(stage, "/World/Lights/Sun")
        if dome:
            dome.GetIntensityAttr().Set(300.0 * plan.light_intensity_scale)
        if sun:
            sun.GetIntensityAttr().Set(1500.0 * plan.light_intensity_scale)
        palette = (
            (0.92, 0.18, 0.08),
            (0.08, 0.28, 0.92),
            (0.08, 0.78, 0.28),
            (0.92, 0.72, 0.08),
            (0.48, 0.48, 0.48),
            (0.08, 0.72, 0.78),
        )
        if dome:
            background_color = palette[plan.background_variant % len(palette)]
            # Keep illumination desaturated so appearance changes without
            # turning the target into an unphysical emissive object.
            dome.GetColorAttr().Set(
                Gf.Vec3f(*[0.65 + 0.25 * component for component in background_color])
            )
        settings = carb.settings.get_settings()
        settings.set_bool(
            "/rtx/post/motionblur/enabled",
            plan.motion_blur_strength > 0.0,
        )
        settings.set_float(
            "/rtx/post/motionblur/maxBlurDiameterFraction",
            float(plan.motion_blur_strength),
        )

    def _require_scene_driver(self) -> _CollectionSceneDriver:
        if self._scene_driver is None:
            self._scene_driver = _UsdCubeV1SceneDriver(self._environment)
        return self._scene_driver

    def _require_plan(self) -> EpisodeRandomization:
        if self._plan is None:
            raise RuntimeError("begin_episode must be called before sampling")
        return self._plan


def _randomization_bounds(config: object) -> RandomizationBounds:
    scene_x, scene_y, scene_z = config.scene.size_xyz_m
    region = config.target.motion.region
    return RandomizationBounds(
        uav_x_m=(-0.35 * scene_x, 0.35 * scene_x),
        uav_y_m=(-0.35 * scene_y, 0.35 * scene_y),
        uav_altitude_m=(max(2.0, 0.1 * scene_z), max(2.1, 0.55 * scene_z)),
        uav_speed_mps=(0.0, min(3.0, config.uav.max_speed_mps)),
        camera_pitch_deg=(-50.0, -15.0),
        camera_horizontal_fov_deg=(55.0, 85.0),
        target_x_m=(region.min_xyz_m[0], region.max_xyz_m[0]),
        target_y_m=(region.min_xyz_m[1], region.max_xyz_m[1]),
        target_altitude_m=(region.min_xyz_m[2], region.max_xyz_m[2]),
        target_speed_mps=(0.1, min(2.5, config.target.max_speed_mps)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_oracle_label_acknowledgements(
            oracle_label_generation=args.oracle_label_generation,
            acknowledge_privileged_oracle=args.acknowledge_privileged_oracle,
        )
    except IsaacDatasetCollectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        collection_protocol = load_cube_collection_protocol(args.collection_config)
    except (OSError, ValueError) as exc:
        print(f"collection protocol error: {exc}", file=sys.stderr)
        return 2
    if args.class_name.strip() != CUBE_CLASS_NAME:
        print(
            "error: cube-v1 is a single closed-set class; --class-name must be "
            "exactly cube (colour belongs only in metadata)",
            file=sys.stderr,
        )
        return 2
    if args.class_id != CUBE_CLASS_ID:
        print("error: cube-v1 requires --class-id 0", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        limits = CollectionLimits(
            max_samples=args.max_samples,
            max_episodes=args.max_episodes,
            frames_per_episode=args.frames_per_episode,
            sample_hz=args.sample_hz,
            min_bbox_area_px=args.min_bbox_area_px,
        )
        bounds = _randomization_bounds(config)
    except (TypeError, ValueError) as exc:
        print(f"collection configuration error: {exc}", file=sys.stderr)
        return 2
    if args.validate_only:
        print(
            "Isaac YOLO collection configuration valid: "
            f"max_samples={limits.max_samples}, max_episodes={limits.max_episodes}, "
            f"sample_hz={limits.sample_hz}, privileged_oracle=acknowledged"
        )
        return 0
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve():
        print(
            "error: run this collector with ./python.sh or set UAV_AGENT_CONDA_ENV",
            file=sys.stderr,
        )
        return 2

    # No Isaac-backed module is imported before this point.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "active_gpu": args.gpu_device,
            "physics_gpu": args.gpu_device,
            # DLSS adds a multi-frame post-processing queue.  Isaac Camera
            # annotators then report pixels from several renderer ticks behind
            # World.current_time, which cannot be paired atomically with the
            # Oracle geometry used for labels.  Synthetic training images need
            # the raw renderer output, so disable temporal anti-aliasing here.
            "anti_aliasing": 0,
            # Isaac 5.1 Camera RenderProducts inherit a USD schema fallback of
            # ``dlss`` which otherwise overrides ``anti_aliasing=0``.  These
            # bounded startup settings mirror NVIDIA's SyntheticData sensor
            # tests and make raw, non-temporal rendering effective per product.
            "extra_args": [
                "--/rtx/post/aa/op=0",
                "--/rtx-defaults/post/aa/op=0",
                "--/rtx-transient/post/aa/limitedOps=false",
                "--/app/hydra/renderSettings/useUsdAttributes=false",
                "--/app/hydra/renderSettings/useFabricAttributes=false",
                # Keep Isaac's known non-fatal Camera/Timeline warnings out
                # of the terminal while preserving error-level diagnostics.
                "--/log/channels/isaacsim.core.simulation_manager.plugin=error",
                "--/log/channels/isaacsim.sensors.camera.camera=error",
            ],
            # A 640x480 collection render does not benefit from spanning every
            # server GPU, and doing so can interfere with model/training jobs.
            "multi_gpu": False,
            # Full extension teardown can hang for minutes in Isaac Sim 5.1;
            # failures are flushed above before the bounded fast shutdown.
            "fast_shutdown": True,
        }
    )
    environment = None
    try:
        from env.simple_uav_search_env import SimpleUavSearchEnv

        # Dataset collection is evaluator-only regardless of mission runtime
        # profile.  The flag cannot be routed into MissionAgent.
        environment = SimpleUavSearchEnv(config)
        environment.setup()
        collector = IsaacYoloDatasetCollector(
            output_dir=args.output,
            class_names=(args.class_name.strip(),),
            class_id=args.class_id,
            limits=limits,
            bounds=bounds,
            scene_seed=args.scene_seed,
            oracle_label_generation=args.oracle_label_generation,
            acknowledge_privileged_oracle=args.acknowledge_privileged_oracle,
            cube_protocol=True,
        )
        summary = collector.collect(
            _SimpleSceneCollectionAdapter(
                environment,
                simulation_app,
                config,
                protocol=collection_protocol,
            )
        )
        print(
            "Isaac YOLO collection complete: "
            f"samples={summary.total_samples}, labels={summary.positive_labels}, "
            f"positives={summary.positive_samples}, "
            f"partial_occlusions={summary.partial_occlusion_samples}, "
            f"negatives={summary.negative_samples}, output={summary.output_dir}"
        )
        print(f"split_counts={dict(summary.split_counts)}")
        print(f"manifest={summary.manifest_path}")
        return 0
    except IsaacDatasetCollectionError as exc:
        print(f"Isaac YOLO collection failed closed: {exc}", file=sys.stderr)
        sys.stderr.flush()
        return 2
    except KeyboardInterrupt:
        print("Isaac YOLO collection interrupted", file=sys.stderr)
        sys.stderr.flush()
        return 130
    except Exception as exc:
        # Print the actionable Python failure *before* SimulationApp.close().
        # Kit's native shutdown can terminate/redirect logging before the
        # interpreter gets a chance to render an uncaught traceback, which used
        # to leave users with only unrelated Camera/Fabric/DLSS warnings.
        print(
            "Isaac YOLO collection crashed before completion: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 2
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
