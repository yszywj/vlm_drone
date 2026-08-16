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
from pxr import Gf, UsdGeom, UsdLux

from configs.loader import AppConfig
from env.camera_sensor import ImageProjection, RGBCameraSensor


@dataclass(frozen=True)
class ScenePoseState:
    """Current root poses in the world frame; quaternion order is wxyz."""

    uav_position: np.ndarray
    uav_orientation: np.ndarray
    target_position: np.ndarray
    target_orientation: np.ndarray


class UavSearchScene:
    """Build and expose the objects in the first UAV-search scene."""

    GROUND_PRIM_PATH = "/World/Ground"
    UAV_PRIM_PATH = "/World/UAV"
    CAMERA_PRIM_PATH = "/World/UAV/Camera"
    TARGET_PRIM_PATH = "/World/Target"

    def __init__(self, world: World, config: AppConfig) -> None:
        self.world = world
        self.config = config
        self.ground = None
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
        self._add_uav()
        self._add_target()
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
        size_x, size_y, size_z = self.config.scene.size_xyz_m
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, "/World/Obstacles")

        footprint = max(0.5, min(3.0, min(size_x, size_y) * 0.06))
        heights = (
            max(0.5, min(2.0, size_z * 0.20)),
            max(0.5, min(3.0, size_z * 0.25)),
            max(0.5, min(4.0, size_z * 0.30)),
        )
        obstacle_specs = (
            ("Box_00", [0.12 * size_x, 0.08 * size_y], [footprint, footprint, heights[0]], [0.75, 0.25, 0.20]),
            ("Box_01", [-0.15 * size_x, 0.10 * size_y], [footprint, 0.7 * footprint, heights[1]], [0.25, 0.60, 0.85]),
            ("Box_02", [0.10 * size_x, -0.14 * size_y], [0.7 * footprint, footprint, heights[2]], [0.55, 0.35, 0.75]),
        )
        for name, xy, scale, color in obstacle_specs:
            position = np.asarray([xy[0], xy[1], scale[2] / 2.0])
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Obstacles/{name}",
                    name=f"obstacle_{name.lower()}",
                    position=position,
                    scale=np.asarray(scale),
                    size=1.0,
                    color=np.asarray(color),
                )
            )

    def _add_uav(self) -> None:
        uav_position = np.asarray(self.config.uav.initial_position_xyz_m, dtype=np.float64)
        self.uav = self.world.scene.add(
            SingleXFormPrim(
                prim_path=self.UAV_PRIM_PATH,
                name="uav",
                position=uav_position,
            )
        )

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
        for name, translation, scale, color in components:
            self.world.scene.add(
                VisualCuboid(
                    prim_path=f"{self.UAV_PRIM_PATH}/{name}",
                    name=f"uav_{name.lower()}",
                    translation=np.asarray(translation),
                    scale=np.asarray(scale),
                    size=1.0,
                    color=np.asarray(color),
                )
            )

        self.camera_sensor = RGBCameraSensor(self.world, self.config.camera)
        self.camera_sensor.build()

    def _add_target(self) -> None:
        region = self.config.target.initial_region
        target_position = (
            np.asarray(region.min_xyz_m, dtype=np.float64)
            + np.asarray(region.max_xyz_m, dtype=np.float64)
        ) / 2.0
        self.target = self.world.scene.add(
            SingleXFormPrim(
                prim_path=self.TARGET_PRIM_PATH,
                name="target",
                position=target_position,
            )
        )
        self.world.scene.add(
            VisualCuboid(
                prim_path=f"{self.TARGET_PRIM_PATH}/Body",
                name="target_body",
                translation=np.zeros(3),
                scale=np.asarray([0.6, 0.6, 1.0]),
                size=1.0,
                color=np.asarray([1.0, 0.12, 0.05]),
            )
        )

    def configure_overview_viewport(self) -> None:
        """Aim the GUI viewport so Ground, UAV, CameraHousing, and Target share the view."""

        poses = self.read_poses()
        focus = (poses.uav_position + poses.target_position) / 2.0
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

        self._require_uav().set_world_pose(
            position=_position(position_m) if position_m is not None else None,
            orientation=_unit_quaternion(orientation_wxyz) if orientation_wxyz is not None else None,
        )

    def set_target_pose(
        self,
        position_m: Sequence[float] | None = None,
        orientation_wxyz: Sequence[float] | None = None,
    ) -> None:
        """Set the Target root pose in meters and scalar-first quaternion form."""

        self._require_target().set_world_pose(
            position=_position(position_m) if position_m is not None else None,
            orientation=_unit_quaternion(orientation_wxyz) if orientation_wxyz is not None else None,
        )

    def get_camera_rgb(self) -> np.ndarray:
        """Return the latest RGB image; render simulation steps first."""

        return self._require_camera_sensor().get_rgb()

    def save_camera_rgb(self, path: str, image: np.ndarray | None = None) -> str:
        return str(self._require_camera_sensor().save_rgb(path, image=image))

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._require_camera_sensor().get_camera_pose()

    def world_to_image(
        self,
        points_xyz_m: Sequence[float] | Sequence[Sequence[float]],
    ) -> ImageProjection:
        return self._require_camera_sensor().world_to_image(points_xyz_m)

    def _require_uav(self) -> SingleXFormPrim:
        if self.uav is None:
            raise RuntimeError("scene must be built before accessing UAV pose")
        return self.uav

    def _require_target(self) -> SingleXFormPrim:
        if self.target is None:
            raise RuntimeError("scene must be built before accessing Target pose")
        return self.target

    def _require_camera_sensor(self) -> RGBCameraSensor:
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
