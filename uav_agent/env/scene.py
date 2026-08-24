"""Isaac Sim scene construction for UAV search.

This module must only be imported after ``SimulationApp`` is created.

World frame (right-handed, Z-up, meters):
    +X: UAV forward when yaw is zero.
    +Y: UAV left when yaw is zero.
    +Z: up.
    Positive yaw: right-hand rotation about +Z, from +X toward +Y
    (counter-clockwise when viewed from above).

All orientations use scalar-first quaternions ``[w, x, y, z]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Tf, UsdGeom, UsdLux

from configs.schema import AppConfig
from env.camera_sensor import ImageProjection, RGBCameraSensor
from env.camera_types import CameraSample
from env.fleet_uav_search_env import camera_prim_path, target_prim_path, uav_prim_path
from env.obstacle_registry import ObstacleRegistry, obstacle_scene_prim_key


@dataclass(frozen=True)
class ScenePoseState:
    """Current root poses in the world frame; quaternion order is wxyz."""

    uav_position: np.ndarray
    uav_orientation: np.ndarray
    target_position: np.ndarray
    target_orientation: np.ndarray


@dataclass(frozen=True)
class FleetScenePoseState:
    """One atomic read of every fleet entity pose."""

    uav_positions: dict[str, np.ndarray]
    uav_orientations: dict[str, np.ndarray]
    target_positions: dict[str, np.ndarray]
    target_orientations: dict[str, np.ndarray]


class UavSearchScene:
    """Build one or more UAVs/targets while retaining singleton conveniences."""

    GROUND_PRIM_PATH = "/World/Ground"
    UAV_ROOT_PRIM_PATH = "/World/UAVs"
    TARGET_ROOT_PRIM_PATH = "/World/Targets"

    def __init__(
        self,
        world: World,
        config: AppConfig,
        *,
        obstacle_registry: ObstacleRegistry | None = None,
    ) -> None:
        self.world = world
        self.config = config
        self.obstacle_registry = (
            ObstacleRegistry.from_scene_config(config.scene)
            if obstacle_registry is None
            else obstacle_registry
        )
        if not isinstance(self.obstacle_registry, ObstacleRegistry):
            raise TypeError("obstacle_registry must be an ObstacleRegistry")
        self.ground = None
        self.uav_prims: dict[str, SingleXFormPrim] = {}
        self.camera_sensors: dict[str, RGBCameraSensor] = {}
        self.target_prims: dict[str, SingleXFormPrim] = {}
        self.uav_prim_paths = {
            item.id: uav_prim_path(item.id) for item in config.uavs
        }
        self.camera_prim_paths = {
            item.id: camera_prim_path(item.id) for item in config.uavs
        }
        self.target_prim_paths = {
            item.id: target_prim_path(item.id) for item in config.targets
        }
        if len(set(self.uav_prim_paths.values())) != len(self.uav_prim_paths):
            raise ValueError("UAV IDs must resolve to unique Prim paths")
        if len(set(self.target_prim_paths.values())) != len(self.target_prim_paths):
            raise ValueError("target IDs must resolve to unique Prim paths")
        # Singleton aliases used by the existing SimpleUavSearchEnv.
        self.uav: SingleXFormPrim | None = None
        self.camera_sensor: RGBCameraSensor | None = None
        self.target: SingleXFormPrim | None = None
        self._built = False

    def build(self) -> None:
        """Create all local primitives; ``World.reset`` remains the caller's job."""

        if self._built:
            raise RuntimeError("scene has already been built")
        self._add_ground_and_lights()
        self._add_obstacles()
        self._add_uavs()
        self._add_targets()
        if len(self.config.uavs) == 1:
            sole_uav_id = self.config.uavs[0].id
            self.uav = self.uav_prims[sole_uav_id]
            self.camera_sensor = self.camera_sensors[sole_uav_id]
        if len(self.config.targets) == 1:
            self.target = self.target_prims[self.config.targets[0].id]
        self._built = True

    def _add_ground_and_lights(self) -> None:
        size_x, size_y, _ = self.config.scene.size_xyz_m
        self.ground = self.world.scene.add_ground_plane(
            size=max(size_x, size_y),
            z_position=0.0,
            name="ground",
            prim_path=self.GROUND_PRIM_PATH,
            color=np.asarray([1.0, 1.0, 1.0]),
        )

        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, "/World/Lights")
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(300.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
        sun.CreateIntensityAttr(1500.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
        UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(
            Gf.Vec3f(-45.0, 30.0, 0.0),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )

    def _add_obstacles(self) -> None:
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, "/World/Obstacles")
        for index, spec in enumerate(self.obstacle_registry):
            prim_name = Tf.MakeValidIdentifier(
                obstacle_scene_prim_key(index, spec.obstacle_id)
            )
            obstacle_type = FixedCuboid if spec.collidable else VisualCuboid
            self.world.scene.add(
                obstacle_type(
                    prim_path=f"/World/Obstacles/{prim_name}",
                    name=prim_name,
                    position=np.asarray(spec.center_xyz_m, dtype=np.float64),
                    scale=np.asarray(spec.size_xyz_m, dtype=np.float64),
                    size=1.0,
                    color=np.asarray(spec.color_rgb, dtype=np.float64),
                )
            )

    def _add_uavs(self) -> None:
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, self.UAV_ROOT_PRIM_PATH)
        components = (
            ("Body", [0.0, 0.0, 0.0], [0.90, 0.55, 0.22], [0.08, 0.30, 0.90]),
            ("ArmX", [0.0, 0.0, 0.04], [1.70, 0.12, 0.08], [0.18, 0.38, 0.75]),
            ("ArmY", [0.0, 0.0, 0.04], [0.12, 1.70, 0.08], [0.18, 0.38, 0.75]),
            ("MotorFront", [0.75, 0.0, 0.08], [0.24, 0.24, 0.12], [0.10, 0.10, 0.12]),
            ("MotorRear", [-0.75, 0.0, 0.08], [0.24, 0.24, 0.12], [0.10, 0.10, 0.12]),
            ("MotorLeft", [0.0, 0.75, 0.08], [0.24, 0.24, 0.12], [0.10, 0.10, 0.12]),
            ("MotorRight", [0.0, -0.75, 0.08], [0.24, 0.24, 0.12], [0.10, 0.10, 0.12]),
            ("Nose", [0.48, 0.0, 0.08], [0.14, 0.28, 0.10], [1.00, 0.45, 0.05]),
        )
        for uav_config in self.config.uavs:
            prim_path = self.uav_prim_paths[uav_config.id]
            prim_name = Tf.MakeValidIdentifier(f"uav_{uav_config.id}")
            self.uav_prims[uav_config.id] = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=prim_path,
                    name=prim_name,
                    position=np.asarray(
                        uav_config.initial_position_xyz_m, dtype=np.float64
                    ),
                )
            )
            for component_name, translation, scale, color in components:
                self.world.scene.add(
                    VisualCuboid(
                        prim_path=f"{prim_path}/{component_name}",
                        name=Tf.MakeValidIdentifier(
                            f"{prim_name}_{component_name.lower()}"
                        ),
                        translation=np.asarray(translation),
                        scale=np.asarray(scale),
                        size=1.0,
                        color=np.asarray(color),
                    )
                )
            sensor = RGBCameraSensor(
                self.world,
                self.config.camera_profiles[uav_config.camera_profile],
                camera_prim_path=self.camera_prim_paths[uav_config.id],
                housing_prim_path=f"{prim_path}/CameraHousing",
                sensor_name=f"uav_rgb_camera_{uav_config.id}",
                acquisition_frequency_hz=round(
                    1.0 / self.config.simulation.rendering_dt_s
                ),
            )
            sensor.build()
            self.camera_sensors[uav_config.id] = sensor

    def _add_targets(self) -> None:
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, self.TARGET_ROOT_PRIM_PATH)
        for target_config in self.config.targets:
            region = target_config.initial_region
            target_position = (
                np.asarray(region.min_xyz_m, dtype=np.float64)
                + np.asarray(region.max_xyz_m, dtype=np.float64)
            ) / 2.0
            prim_path = self.target_prim_paths[target_config.id]
            prim_name = Tf.MakeValidIdentifier(f"target_{target_config.id}")
            self.target_prims[target_config.id] = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=prim_path,
                    name=prim_name,
                    position=target_position,
                )
            )
            self.world.scene.add(
                VisualCuboid(
                    prim_path=f"{prim_path}/Body",
                    name=f"{prim_name}_body",
                    translation=np.zeros(3),
                    scale=np.asarray(target_config.appearance.size_xyz_m),
                    size=1.0,
                    color=np.asarray(target_config.appearance.color_rgb),
                )
            )

    def configure_overview_viewport(self) -> None:
        """Aim the GUI viewport so Ground, UAV, CameraHousing, and Target share the view."""

        poses = self.read_fleet_poses()
        all_positions = tuple(poses.uav_positions.values()) + tuple(
            poses.target_positions.values()
        )
        focus = np.mean(np.stack(all_positions), axis=0)
        span = max(12.0, min(25.0, min(self.config.scene.size_xyz_m[:2]) * 0.20))
        eye = focus + np.asarray([0.70 * span, -0.80 * span, 0.50 * span])
        set_camera_view(
            eye=eye,
            target=focus,
            camera_prim_path="/OmniverseKit_Persp",
        )

    def read_poses(self) -> ScenePoseState:
        """Read current UAV and Target root poses from the USD world frame."""

        uav = self._require_uav()
        target = self._require_target()
        uav_position, uav_orientation = uav.get_world_pose()
        target_position, target_orientation = target.get_world_pose()
        return ScenePoseState(
            uav_position=np.asarray(uav_position).copy(),
            uav_orientation=np.asarray(uav_orientation).copy(),
            target_position=np.asarray(target_position).copy(),
            target_orientation=np.asarray(target_orientation).copy(),
        )

    def read_fleet_poses(self) -> FleetScenePoseState:
        """Read every root pose before any agent is ticked."""

        uav_positions: dict[str, np.ndarray] = {}
        uav_orientations: dict[str, np.ndarray] = {}
        target_positions: dict[str, np.ndarray] = {}
        target_orientations: dict[str, np.ndarray] = {}
        for uav_id in sorted(self.uav_prims):
            position, orientation = self.uav_prims[uav_id].get_world_pose()
            uav_positions[uav_id] = np.asarray(position).copy()
            uav_orientations[uav_id] = np.asarray(orientation).copy()
        for target_id in sorted(self.target_prims):
            position, orientation = self.target_prims[target_id].get_world_pose()
            target_positions[target_id] = np.asarray(position).copy()
            target_orientations[target_id] = np.asarray(orientation).copy()
        if len(uav_positions) != len(self.config.uavs) or len(target_positions) != len(
            self.config.targets
        ):
            raise RuntimeError("scene must be built before reading fleet poses")
        return FleetScenePoseState(
            uav_positions=uav_positions,
            uav_orientations=uav_orientations,
            target_positions=target_positions,
            target_orientations=target_orientations,
        )

    @property
    def uav_position(self) -> np.ndarray:
        return self.read_poses().uav_position

    @property
    def uav_orientation(self) -> np.ndarray:
        return self.read_poses().uav_orientation

    @property
    def target_position(self) -> np.ndarray:
        return self.read_poses().target_position

    @property
    def target_orientation(self) -> np.ndarray:
        return self.read_poses().target_orientation

    def set_uav_pose(
        self,
        position_m: Sequence[float] | None = None,
        orientation_wxyz: Sequence[float] | None = None,
    ) -> None:
        """Set the UAV root pose; every child, including Camera, follows it."""

        self.set_uav_pose_for(
            self.config.uav.id,
            position_m=position_m,
            orientation_wxyz=orientation_wxyz,
        )

    def set_uav_pose_for(
        self,
        uav_id: str,
        position_m: Sequence[float] | None = None,
        orientation_wxyz: Sequence[float] | None = None,
    ) -> None:
        self._require_uav(uav_id).set_world_pose(
            position=_position(position_m) if position_m is not None else None,
            orientation=_unit_quaternion(orientation_wxyz) if orientation_wxyz is not None else None,
        )

    def set_target_pose(
        self,
        position_m: Sequence[float] | None = None,
        orientation_wxyz: Sequence[float] | None = None,
    ) -> None:
        """Set the Target root pose in meters and scalar-first quaternion form."""

        self.set_target_pose_for(
            self.config.target.id,
            position_m=position_m,
            orientation_wxyz=orientation_wxyz,
        )

    def set_target_pose_for(
        self,
        target_id: str,
        position_m: Sequence[float] | None = None,
        orientation_wxyz: Sequence[float] | None = None,
    ) -> None:
        self._require_target(target_id).set_world_pose(
            position=_position(position_m) if position_m is not None else None,
            orientation=_unit_quaternion(orientation_wxyz) if orientation_wxyz is not None else None,
        )

    def get_camera_rgb(self) -> np.ndarray:
        """Return the latest RGB image; render simulation steps first."""

        return self._require_camera_sensor().get_rgb()

    def get_camera_sample(self) -> CameraSample:
        """Return one synchronized RGB-D Camera sample."""

        return self._require_camera_sensor().get_sample()

    def save_camera_rgb(self, path: str, image: np.ndarray | None = None) -> str:
        return str(self._require_camera_sensor().save_rgb(path, image=image))

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._require_camera_sensor().get_camera_pose()

    def world_to_image(
        self,
        points_xyz_m: Sequence[float] | Sequence[Sequence[float]],
    ) -> ImageProjection:
        return self._require_camera_sensor().world_to_image(points_xyz_m)

    def get_camera_sensor(self, uav_id: str) -> RGBCameraSensor:
        return self._require_camera_sensor(uav_id)

    def world_to_image_for(
        self,
        uav_id: str,
        points_xyz_m: Sequence[float] | Sequence[Sequence[float]],
    ) -> ImageProjection:
        return self._require_camera_sensor(uav_id).world_to_image(points_xyz_m)

    def _require_uav(self, uav_id: str | None = None) -> SingleXFormPrim:
        if uav_id is not None:
            try:
                return self.uav_prims[uav_id]
            except KeyError as exc:
                raise KeyError(f"unknown uav_id: {uav_id}") from exc
        if self.uav is None:
            raise RuntimeError("scene must be built before accessing UAV pose")
        return self.uav

    def _require_target(self, target_id: str | None = None) -> SingleXFormPrim:
        if target_id is not None:
            try:
                return self.target_prims[target_id]
            except KeyError as exc:
                raise KeyError(f"unknown target_id: {target_id}") from exc
        if self.target is None:
            raise RuntimeError("scene must be built before accessing Target pose")
        return self.target

    def _require_camera_sensor(self, uav_id: str | None = None) -> RGBCameraSensor:
        if uav_id is not None:
            try:
                return self.camera_sensors[uav_id]
            except KeyError as exc:
                raise KeyError(f"unknown uav_id: {uav_id}") from exc
        if self.camera_sensor is None:
            raise RuntimeError("scene must be built before accessing the camera")
        return self.camera_sensor


def _position(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("position_m must contain three finite values")
    return result


def _unit_quaternion(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError("orientation_wxyz must contain four finite values")
    norm = np.linalg.norm(result)
    if norm <= 1e-12:
        raise ValueError("orientation_wxyz must have non-zero norm")
    return result / norm
