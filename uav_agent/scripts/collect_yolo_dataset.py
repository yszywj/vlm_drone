#!/usr/bin/env python3
"""Collect a bounded YOLO dataset from Isaac truth (never for agent runtime)."""

from __future__ import annotations

import argparse
from math import ceil, cos, isfinite, pi, radians, sin, tan
import os
from pathlib import Path
import random
import sys
from typing import Sequence

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
    OracleFrameTruth,
    RandomizationBounds,
    estimate_depth_visibility,
    require_oracle_label_acknowledgements,
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
    parser.add_argument("--output", required=True, help="new/replaceable dataset directory")
    parser.add_argument("--scene-seed", type=_nonnegative_int, default=42)
    parser.add_argument("--max-samples", type=_positive_int, default=2_000)
    parser.add_argument("--max-episodes", type=_positive_int, default=100)
    parser.add_argument("--frames-per-episode", type=_positive_int, default=20)
    parser.add_argument("--sample-hz", type=_positive_float, default=2.0)
    parser.add_argument("--min-bbox-area-px", type=_positive_float, default=16.0)
    parser.add_argument("--class-id", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--class-name",
        default="red_cube",
        help="fixed simple-scene target semantic; currently only red_cube is valid",
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
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser.parse_args(argv)


class _SimpleSceneCollectionAdapter:
    """Late-bound adapter for the repository's simple Isaac scene.

    The adapter uses simulator projection of the eight rendered Target-box
    corners.  Target motion is integrated at physics rate with a bounded turn
    rate; direction targets change only at the configured episode interval.
    """

    def __init__(self, environment: object, simulation_app: object, config: object) -> None:
        self._environment = environment
        self._simulation_app = simulation_app
        self._config = config
        self._plan: EpisodeRandomization | None = None
        self._target_position = np.zeros(3, dtype=np.float64)
        self._target_heading_rad = 0.0
        self._desired_heading_rad = 0.0
        self._target_scale = 1.0
        self._elapsed_s = 0.0
        self._next_turn_s = float("inf")
        self._rng = random.Random(0)

    def begin_episode(self, randomization: EpisodeRandomization) -> None:
        self._plan = randomization
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
        target_motion = self._environment._require_target_motion()
        target_motion.reset(
            position_m=self._target_position,
            yaw_rad=self._target_heading_rad,
            seed=randomization.key.scene_seed,
            mode="STATIC",
        )
        self._set_target_visibility(randomization.sample_kind != "negative")
        self._set_target_scale(self._target_scale)
        self._configure_occluder(randomization.sample_kind == "partial_occlusion")
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
            new_frame = bool(self._environment.step()) or new_frame
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

    def capture_oracle_frame(self, frame_id: str) -> OracleFrameTruth:
        del frame_id
        plan = self._require_plan()
        sample = self._environment.get_camera_sample()
        if plan.sample_kind == "negative":
            return OracleFrameTruth(
                camera_sample=sample,
                target_position_world_m=None,
                target_orientation_world_wxyz=None,
                projected_target_pixels_uv=None,
                projected_target_depth_m=None,
                occlusion_ratio=None,
            )
        evaluator = self._environment.get_evaluator_frame()
        target_position = np.asarray(evaluator.target_position_m, dtype=np.float64)
        target_orientation = np.asarray(
            evaluator.target_orientation_wxyz,
            dtype=np.float64,
        )
        corners = self._target_world_corners(
            target_position,
            target_orientation,
            self._target_scale,
        )
        projection = self._environment.world_to_image(corners)
        # This is the same atomic CameraSample that supplied RGB.  A projected
        # box alone cannot prove visibility: nearer depth pixels identify the
        # collection-only screen, while absent/invalid depth fails closed.
        depth_visibility = estimate_depth_visibility(
            sample,
            projection.pixels_uv,
            projection.depth_m,
        )
        occlusion_ratio = depth_visibility.occlusion_ratio
        return OracleFrameTruth(
            camera_sample=sample,
            target_position_world_m=tuple(float(value) for value in target_position),
            target_orientation_world_wxyz=tuple(
                float(value) for value in target_orientation
            ),
            projected_target_pixels_uv=projection.pixels_uv,
            projected_target_depth_m=projection.depth_m,
            occlusion_ratio=occlusion_ratio,
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
        try:
            self._environment.set_target_pose(
                candidate,
                (
                    cos(self._target_heading_rad / 2.0),
                    0.0,
                    0.0,
                    sin(self._target_heading_rad / 2.0),
                ),
            )
            self._target_position = candidate
        except ValueError:
            # A reflected smooth turn avoids an obstacle without teleporting.
            self._target_heading_rad += pi

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

    def _set_target_visibility(self, visible: bool) -> None:
        # Imported only after SimulationApp in main().
        from pxr import UsdGeom

        imageable = UsdGeom.Imageable(self._environment.scene.target.prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()

    def _set_target_scale(self, scale: float) -> None:
        # Scale the rendered body; the projected GT corners use the same value.
        from pxr import Gf, UsdGeom

        stage = self._environment.scene.target.prim.GetStage()
        target_root_path = str(self._environment.scene.target.prim.GetPath())
        body = stage.GetPrimAtPath(f"{target_root_path}/Body")
        xformable = UsdGeom.Xformable(body)
        for operation in xformable.GetOrderedXformOps():
            if operation.GetOpType() == UsdGeom.XformOp.TypeScale:
                operation.Set(Gf.Vec3d(0.6 * scale, 0.6 * scale, 1.0 * scale))
                return
        xformable.AddScaleOp().Set(
            Gf.Vec3d(0.6 * scale, 0.6 * scale, 1.0 * scale)
        )

    def _configure_occluder(self, enabled: bool) -> None:
        """Place a non-colliding screen between Camera and Target when requested."""

        from math import atan2, degrees
        from pxr import Gf, UsdGeom

        stage = self._environment.scene.target.prim.GetStage()
        cube = UsdGeom.Cube.Get(stage, "/World/CollectionOnlyOccluder")
        if not cube:
            cube = UsdGeom.Cube.Define(stage, "/World/CollectionOnlyOccluder")
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.12)])
        imageable = UsdGeom.Imageable(cube.GetPrim())
        if not enabled:
            imageable.MakeInvisible()
            return
        plan = self._require_plan()
        uav = np.asarray(plan.uav_position_world_m, dtype=np.float64)
        target = self._target_position
        direction = target - uav
        norm_xy = float(np.linalg.norm(direction[:2]))
        if norm_xy <= 1e-6:
            direction[:2] = (1.0, 0.0)
            norm_xy = 1.0
        unit_xy = direction[:2] / norm_xy
        center = target.copy()
        center[:2] -= 0.8 * unit_xy
        xform = UsdGeom.XformCommonAPI(cube.GetPrim())
        xform.SetTranslate(Gf.Vec3d(*[float(value) for value in center]))
        xform.SetRotate(
            Gf.Vec3f(0.0, 0.0, degrees(atan2(unit_xy[1], unit_xy[0]))),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        xform.SetScale(
            Gf.Vec3f(
                0.08,
                float(0.28 * self._target_scale),
                float(0.70 * self._target_scale),
            )
        )
        imageable.MakeVisible()

    def _apply_render_randomization(self, plan: EpisodeRandomization) -> None:
        """Apply bounded lighting, background/material, and blur variants."""

        import carb
        from pxr import Gf, UsdGeom, UsdLux

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
            (0.72, 0.12, 0.82),
            (0.08, 0.72, 0.78),
        )
        target_color = palette[plan.material_variant % len(palette)]
        target_root_path = str(self._environment.scene.target.prim.GetPath())
        target_body = UsdGeom.Gprim.Get(stage, f"{target_root_path}/Body")
        if target_body:
            target_body.GetDisplayColorAttr().Set([Gf.Vec3f(*target_color)])
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

    @staticmethod
    def _target_world_corners(
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        half_extent = np.asarray([0.3, 0.3, 0.5], dtype=np.float64) * scale
        local = np.asarray(
            [
                [sx * half_extent[0], sy * half_extent[1], sz * half_extent[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        )
        w, x, y, z = orientation_wxyz / np.linalg.norm(orientation_wxyz)
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return local @ rotation.T + position

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
    if args.class_name.strip() != "red_cube":
        print(
            "error: this collector renders a fixed red cube; --class-name must be "
            "exactly red_cube to prevent semantic mislabelling",
            file=sys.stderr,
        )
        return 2
    if args.class_id != 0:
        print("error: one --class-name is configured, so --class-id must be 0", file=sys.stderr)
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

    simulation_app = SimulationApp({"headless": bool(args.headless)})
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
        )
        summary = collector.collect(
            _SimpleSceneCollectionAdapter(environment, simulation_app, config)
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
    except KeyboardInterrupt:
        print("Isaac YOLO collection interrupted", file=sys.stderr)
        return 130
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
