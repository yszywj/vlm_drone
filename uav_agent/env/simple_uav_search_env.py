"""Lifecycle wrapper around :mod:`env.scene` for the standalone demo.

This module must only be imported after ``SimulationApp`` has been created.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, radians
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from isaacsim.core.api import World

from configs.loader import AppConfig
from env.camera_sensor import CameraFrameNotReady, ImageProjection
from env.kinematic_uav import KinematicUAV, UAVState
from env.moving_target import MovingTarget, TargetState
from env.scene import ScenePoseState, UavSearchScene
from skills.types import Observation, SkillClock, SkillContext


@dataclass(frozen=True)
class AgentObservation:
    """Default agent-facing observation; deliberately excludes Target truth."""

    rgb: np.ndarray
    uav_state: UAVState
    uav_velocity_mps: np.ndarray
    camera_position_m: np.ndarray
    camera_orientation_wxyz: np.ndarray
    camera_timestamp_s: float


@dataclass(frozen=True)
class EvaluatorFrame:
    """A synchronized Camera frame plus privileged Target truth."""

    observation: AgentObservation
    target_position_m: np.ndarray
    target_orientation_wxyz: np.ndarray
    target_state: TargetState
    target_velocity_mps: np.ndarray
    target_projection: ImageProjection


class AgentView:
    """Narrow Planner capability that cannot read evaluator Target APIs."""

    __slots__ = (
        "__observe",
        "__move_toward",
        "__set_velocity",
        "__rotate_yaw",
        "__stop",
        "__distance_to_goal",
        "__heading_error",
        "__goal_reached",
    )

    def __init__(
        self,
        *,
        observe: Callable[[], AgentObservation],
        move_toward: Callable[..., None],
        set_velocity: Callable[..., None],
        rotate_yaw: Callable[..., None],
        stop: Callable[[], None],
        distance_to_goal: Callable[..., float],
        heading_error: Callable[..., float],
        goal_reached: Callable[..., bool],
    ) -> None:
        self.__observe = observe
        self.__move_toward = move_toward
        self.__set_velocity = set_velocity
        self.__rotate_yaw = rotate_yaw
        self.__stop = stop
        self.__distance_to_goal = distance_to_goal
        self.__heading_error = heading_error
        self.__goal_reached = goal_reached

    def observe(self) -> AgentObservation:
        return self.__observe()

    def move_toward(
        self,
        goal_xyz_m: Sequence[float],
        speed_mps: float | None = None,
        *,
        face_goal: bool = True,
        tolerance_m: float | None = None,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        self.__move_toward(
            goal_xyz_m,
            speed_mps,
            face_goal=face_goal,
            tolerance_m=tolerance_m,
            max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        )

    def set_velocity(
        self,
        velocity_xyz_mps: Sequence[float],
        yaw_rate_rad_s: float = 0.0,
    ) -> None:
        self.__set_velocity(velocity_xyz_mps, yaw_rate_rad_s)

    def rotate_yaw(
        self,
        target_yaw_rad: float,
        *,
        relative: bool = False,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        self.__rotate_yaw(
            target_yaw_rad,
            relative=relative,
            max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        )

    def stop(self) -> None:
        self.__stop()

    def distance_to_goal(self, goal_xyz_m: Sequence[float] | None = None) -> float:
        return self.__distance_to_goal(goal_xyz_m)

    def heading_error(self, goal_xyz_m: Sequence[float] | None = None) -> float:
        return self.__heading_error(goal_xyz_m)

    def goal_reached(
        self,
        goal_xyz_m: Sequence[float] | None = None,
        tolerance_m: float | None = None,
    ) -> bool:
        return self.__goal_reached(goal_xyz_m, tolerance_m)


class SimpleUavSearchEnv:
    """Own the World lifecycle while delegating construction to UavSearchScene."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.world: World | None = None
        self.scene: UavSearchScene | None = None
        self.uav_controller: KinematicUAV | None = None
        self._target_motion: MovingTarget | None = None
        self._latest_agent_observation: AgentObservation | None = None
        self._latest_evaluator_frame: EvaluatorFrame | None = None
        self._last_camera_timestamp_s: float | None = None

    def setup(self) -> None:
        if self.world is not None:
            raise RuntimeError("environment has already been set up")
        simulation = self.config.simulation
        self.world = World(
            physics_dt=simulation.physics_dt_s,
            rendering_dt=simulation.rendering_dt_s,
            stage_units_in_meters=simulation.stage_units_in_meters,
        )
        self.scene = UavSearchScene(self.world, self.config)
        self.scene.build()
        self.world.reset()

        initial_uav = self.config.uav.initial_position_xyz_m
        self.uav_controller = KinematicUAV(
            initial_state=UAVState(*initial_uav, yaw=0.0),
            max_speed_mps=self.config.uav.max_speed_mps,
            max_yaw_rate_rad_s=radians(self.config.uav.max_yaw_rate_deg_s),
            pose_writer=self.scene.set_uav_pose,
        )
        motion = self.config.target.motion
        self._target_motion = MovingTarget(
            mode=motion.mode,
            initial_position_xyz_m=self._initial_target_position(),
            bounds_min_xyz_m=motion.region.min_xyz_m,
            bounds_max_xyz_m=motion.region.max_xyz_m,
            speed_mps=motion.speed_mps,
            max_speed_mps=self.config.target.max_speed_mps,
            direction_change_interval_s=motion.direction_change_interval_s,
            seed=motion.seed,
            initial_heading_rad=radians(motion.initial_heading_deg),
            pose_writer=self.scene.set_target_pose,
        )
        self._invalidate_camera_observation()

    def reset(self, *, target_seed: int | None = None) -> None:
        """Coordinately reset World, motion truth sources, and Camera cache."""

        if self.world is None or self.scene is None:
            raise RuntimeError("environment must be set up before reset")
        if target_seed is not None and (
            isinstance(target_seed, bool) or not isinstance(target_seed, int) or target_seed < 0
        ):
            raise ValueError("target_seed must be a non-negative integer")
        self._invalidate_camera_observation()
        self.world.reset()
        initial_uav = self.config.uav.initial_position_xyz_m
        self._require_uav_controller().set_pose(*initial_uav, yaw=0.0)
        self._require_target_motion().reset(
            position_m=self._initial_target_position(),
            seed=target_seed,
        )
        self._invalidate_camera_observation()

    def get_agent_view(self) -> AgentView:
        """Return the narrow interface that should be injected into Planner."""

        return AgentView(
            observe=self.get_agent_observation,
            move_toward=self.move_uav_toward,
            set_velocity=self.set_uav_velocity,
            rotate_yaw=self.rotate_uav_yaw,
            stop=self.stop_uav,
            distance_to_goal=self.distance_to_goal,
            heading_error=self.heading_error,
            goal_reached=self.goal_reached,
        )

    def make_skill_context(
        self,
        clock: SkillClock,
        *,
        perception: object | None = None,
    ) -> SkillContext:
        """Inject only the dependencies allowed by the unified Skill API."""

        if self.scene is None or self.scene.camera_sensor is None:
            raise RuntimeError("environment must be set up before creating SkillContext")
        return SkillContext(
            uav=self._require_uav_controller(),
            camera=self.scene.camera_sensor,
            perception=perception,
            clock=clock,
        )

    def configure_overview_viewport(self) -> None:
        if self.scene is None:
            raise RuntimeError("environment must be set up before configuring the viewport")
        self.scene.configure_overview_viewport()

    def read_poses(self) -> ScenePoseState:
        """Return evaluator/debug ground truth, including the hidden Target pose."""

        if self.scene is None:
            raise RuntimeError("environment must be set up before reading poses")
        return self.scene.read_poses()

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
        """Teleport the kinematic state for reset/debug, never for navigation."""

        if self.uav_controller is None:
            raise RuntimeError("environment must be set up before setting UAV pose")
        current = self.uav_controller.get_pose()
        position = (
            np.asarray([current.x, current.y, current.z])
            if position_m is None
            else _position(position_m)
        )
        yaw = current.yaw if orientation_wxyz is None else _yaw_from_wxyz(orientation_wxyz)
        try:
            self.uav_controller.set_pose(*position.tolist(), yaw)
        finally:
            self._invalidate_camera_observation()

    def set_target_pose(
        self,
        position_m: Sequence[float] | None = None,
        orientation_wxyz: Sequence[float] | None = None,
    ) -> None:
        """Reset Target truth for evaluator/debug setup."""

        if self._target_motion is None:
            raise RuntimeError("environment must be set up before setting Target pose")
        current = self._target_motion.get_pose()
        position = (
            np.asarray([current.x, current.y, current.z])
            if position_m is None
            else _position(position_m)
        )
        yaw = current.yaw if orientation_wxyz is None else _yaw_from_wxyz(orientation_wxyz)
        try:
            self._target_motion.reset(position_m=position, yaw_rad=yaw)
        finally:
            self._invalidate_camera_observation()

    def set_uav_velocity(
        self,
        velocity_xyz_mps: Sequence[float],
        yaw_rate_rad_s: float = 0.0,
    ) -> None:
        self._require_uav_controller().set_velocity(velocity_xyz_mps, yaw_rate_rad_s)

    def move_uav_toward(
        self,
        goal_xyz_m: Sequence[float],
        speed_mps: float | None = None,
        *,
        face_goal: bool = True,
        tolerance_m: float | None = None,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        self._require_uav_controller().move_toward(
            goal_xyz_m,
            speed_mps,
            face_goal=face_goal,
            tolerance_m=tolerance_m,
            max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        )

    def rotate_uav_yaw(
        self,
        target_yaw_rad: float,
        *,
        relative: bool = False,
        max_yaw_rate_rad_s: float | None = None,
    ) -> None:
        self._require_uav_controller().rotate_yaw(
            target_yaw_rad,
            relative=relative,
            max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        )

    def stop_uav(self) -> None:
        self._require_uav_controller().stop()

    def distance_to_goal(self, goal_xyz_m: Sequence[float] | None = None) -> float:
        return self._require_uav_controller().distance_to_goal(goal_xyz_m)

    def heading_error(self, goal_xyz_m: Sequence[float] | None = None) -> float:
        return self._require_uav_controller().heading_error(goal_xyz_m)

    def goal_reached(
        self,
        goal_xyz_m: Sequence[float] | None = None,
        tolerance_m: float | None = None,
    ) -> bool:
        return self._require_uav_controller().goal_reached(goal_xyz_m, tolerance_m)

    def get_camera_rgb(self) -> np.ndarray:
        return self.get_agent_observation().rgb

    def get_rgb(self) -> np.ndarray:
        return self.get_camera_rgb()

    def save_rgb(self, path: str | Path) -> Path:
        if self.scene is None:
            raise RuntimeError("environment must be set up before saving the camera image")
        observation = self.get_agent_observation()
        return Path(self.scene.save_camera_rgb(str(path), image=observation.rgb))

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.scene is None:
            raise RuntimeError("environment must be set up before reading the camera pose")
        return self.scene.get_camera_pose()

    def world_to_image(
        self,
        points_xyz_m: Sequence[float] | Sequence[Sequence[float]],
    ) -> ImageProjection:
        if self.scene is None:
            raise RuntimeError("environment must be set up before projecting world points")
        return self.scene.world_to_image(points_xyz_m)

    def get_agent_observation(self) -> AgentObservation:
        """Return one time-consistent sampled observation without Target truth."""

        if self._latest_agent_observation is None:
            raise RuntimeError("no synchronized RGB observation yet; step until a Camera frame arrives")
        return _copy_agent_observation(self._latest_agent_observation)

    def get_evaluator_frame(self) -> EvaluatorFrame:
        """Return synchronized privileged truth for tests/debug, never for Planner."""

        if self._latest_evaluator_frame is None:
            raise RuntimeError("no synchronized evaluator frame yet; step until a Camera frame arrives")
        frame = self._latest_evaluator_frame
        return EvaluatorFrame(
            observation=_copy_agent_observation(frame.observation),
            target_position_m=frame.target_position_m.copy(),
            target_orientation_wxyz=frame.target_orientation_wxyz.copy(),
            target_state=frame.target_state,
            target_velocity_mps=frame.target_velocity_mps.copy(),
            target_projection=_copy_projection(frame.target_projection),
        )

    def get_skill_observation(self, *, include_oracle: bool = False) -> Observation:
        """Adapt a synchronized Camera sample to the unified Skill Observation."""

        if not include_oracle:
            agent = self.get_agent_observation()
            return Observation(
                timestamp=agent.camera_timestamp_s,
                uav_pose=agent.uav_state,
                uav_velocity=agent.uav_velocity_mps.copy(),
                camera_rgb=agent.rgb.copy(),
                camera_position_m=agent.camera_position_m.copy(),
                camera_orientation_wxyz=agent.camera_orientation_wxyz.copy(),
            )
        evaluator = self.get_evaluator_frame()
        # Use the AgentObservation embedded in this exact evaluator snapshot;
        # do not independently fetch two caches when constructing Oracle data.
        agent = evaluator.observation
        return Observation(
            timestamp=agent.camera_timestamp_s,
            uav_pose=agent.uav_state,
            uav_velocity=agent.uav_velocity_mps.copy(),
            camera_rgb=agent.rgb.copy(),
            camera_position_m=agent.camera_position_m.copy(),
            camera_orientation_wxyz=agent.camera_orientation_wxyz.copy(),
            oracle_target_id="target",
            oracle_target_visible=bool(evaluator.target_projection.visible[0]),
            oracle_target_pose=evaluator.target_state,
            oracle_target_velocity=evaluator.target_velocity_mps.copy(),
        )

    def _refresh_camera_observation(self) -> bool:
        if self.scene is None or self.scene.camera_sensor is None:
            return False
        try:
            rgb, timestamp_s = self.scene.camera_sensor.get_sample()
        except CameraFrameNotReady:
            return False
        if self._last_camera_timestamp_s is not None and timestamp_s == self._last_camera_timestamp_s:
            return False

        camera_position, camera_orientation = self.scene.get_camera_pose()
        poses = self.scene.read_poses()
        observation = AgentObservation(
            rgb=rgb,
            uav_state=self._require_uav_controller().get_pose(),
            uav_velocity_mps=self._require_uav_controller().get_velocity(),
            camera_position_m=camera_position,
            camera_orientation_wxyz=camera_orientation,
            camera_timestamp_s=timestamp_s,
        )
        self._latest_agent_observation = observation
        self._latest_evaluator_frame = EvaluatorFrame(
            observation=observation,
            target_position_m=poses.target_position,
            target_orientation_wxyz=poses.target_orientation,
            target_state=self._require_target_motion().get_pose(),
            target_velocity_mps=self._require_target_motion().get_velocity(),
            target_projection=self.scene.world_to_image(poses.target_position),
        )
        self._last_camera_timestamp_s = timestamp_s
        return True

    def _invalidate_camera_observation(self) -> None:
        self._latest_agent_observation = None
        self._latest_evaluator_frame = None
        self._last_camera_timestamp_s = None
        if self.scene is not None and self.scene.camera_sensor is not None:
            self.scene.camera_sensor.invalidate_frame()

    def _initial_target_position(self) -> np.ndarray:
        region = self.config.target.initial_region
        return (
            np.asarray(region.min_xyz_m, dtype=np.float64)
            + np.asarray(region.max_xyz_m, dtype=np.float64)
        ) / 2.0

    def step(self) -> bool:
        """Advance one physics step and report whether a new RGB sample arrived."""

        if self.world is None:
            raise RuntimeError("environment must be set up before stepping")
        dt_s = self.config.simulation.physics_dt_s
        self._require_uav_controller().step(dt_s)
        self._require_target_motion().step(dt_s)
        self.world.step(render=True)
        return self._refresh_camera_observation()

    def close(self) -> None:
        if self.world is not None:
            try:
                self.world.stop()
            finally:
                try:
                    if self.scene is not None and self.scene.camera_sensor is not None:
                        self.scene.camera_sensor.destroy()
                finally:
                    try:
                        World.clear_instance()
                    finally:
                        self.world = None
                        self.scene = None
                        self.uav_controller = None
                        self._target_motion = None
                        self._latest_agent_observation = None
                        self._latest_evaluator_frame = None
                        self._last_camera_timestamp_s = None

    def _require_uav_controller(self) -> KinematicUAV:
        if self.uav_controller is None:
            raise RuntimeError("environment must be set up before controlling the UAV")
        return self.uav_controller

    def _require_target_motion(self) -> MovingTarget:
        if self._target_motion is None:
            raise RuntimeError("environment must be set up before stepping the Target")
        return self._target_motion


def _position(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("position_m must contain three finite values")
    return result


def _yaw_from_wxyz(value: Sequence[float]) -> float:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("orientation_wxyz must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("orientation_wxyz must have non-zero norm")
    w, x, y, z = quaternion / norm
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _copy_agent_observation(observation: AgentObservation) -> AgentObservation:
    return AgentObservation(
        rgb=observation.rgb.copy(),
        uav_state=observation.uav_state,
        uav_velocity_mps=observation.uav_velocity_mps.copy(),
        camera_position_m=observation.camera_position_m.copy(),
        camera_orientation_wxyz=observation.camera_orientation_wxyz.copy(),
        camera_timestamp_s=observation.camera_timestamp_s,
    )


def _copy_projection(projection: ImageProjection) -> ImageProjection:
    return ImageProjection(
        pixels_uv=projection.pixels_uv.copy(),
        depth_m=projection.depth_m.copy(),
        visible=projection.visible.copy(),
    )
