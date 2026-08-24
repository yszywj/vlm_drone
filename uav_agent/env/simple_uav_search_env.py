"""Backward-compatible singleton adapter over the fleet environment.

This module deliberately owns no Isaac ``World``, scene entities, Camera
cache, or simulation tick.  :class:`FleetUavSearchEnv` is the sole lifecycle
owner; this class only supplies the historical no-ID API for one UAV and one
Target.  The module is safe to import before ``SimulationApp`` because the
fleet coordinator delays all Isaac imports until ``setup()``.
"""

from __future__ import annotations

from math import atan2
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from configs.schema import AppConfig
from env.fleet_uav_search_env import FleetUavSearchEnv
from env.observation_types import AgentObservation, AgentView, EvaluatorFrame

if TYPE_CHECKING:
    from env.camera_sensor import ImageProjection
    from env.camera_types import CameraSample
    from env.kinematic_uav import KinematicUAV
    from env.moving_target import MovingTarget
    from env.scene import ScenePoseState, UavSearchScene
    from skills.types import Observation, SkillClock, SkillContext


class SimpleUavSearchEnv:
    """Expose the legacy singleton API as a thin fleet-environment adapter."""

    def __init__(self, config: AppConfig) -> None:
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        if len(config.uavs) != 1 or len(config.targets) != 1:
            raise ValueError(
                "SimpleUavSearchEnv requires exactly one UAV and one target; "
                "use FleetUavSearchEnv for plural inventories"
            )
        self.config = config
        self._uav_id = config.uavs[0].id
        self._target_id = config.targets[0].id
        self._fleet_env = FleetUavSearchEnv(
            config,
            assignments={self._uav_id: self._target_id},
        )

    @property
    def world(self) -> object | None:
        return self._fleet_env.world

    @world.setter
    def world(self, value: object | None) -> None:
        self._fleet_env.world = value

    @property
    def scene(self) -> UavSearchScene | None:
        return self._fleet_env.scene  # type: ignore[return-value]

    @scene.setter
    def scene(self, value: UavSearchScene | None) -> None:
        self._fleet_env.scene = value

    @property
    def uav_controller(self) -> KinematicUAV | None:
        return self._fleet_env.uav_controllers.get(self._uav_id)  # type: ignore[return-value]

    @uav_controller.setter
    def uav_controller(self, value: KinematicUAV | None) -> None:
        if value is None:
            self._fleet_env.uav_controllers.pop(self._uav_id, None)
        else:
            self._fleet_env.uav_controllers[self._uav_id] = value

    @property
    def _target_motion(self) -> MovingTarget | None:
        return self._fleet_env.target_motions.get(self._target_id)  # type: ignore[return-value]

    @_target_motion.setter
    def _target_motion(self, value: MovingTarget | None) -> None:
        if value is None:
            self._fleet_env.target_motions.pop(self._target_id, None)
        else:
            self._fleet_env.target_motions[self._target_id] = value

    @property
    def _latest_agent_observation(self) -> AgentObservation | None:
        return self._fleet_env.latest_agent_observations.get(self._uav_id)  # type: ignore[return-value]

    @_latest_agent_observation.setter
    def _latest_agent_observation(self, value: AgentObservation | None) -> None:
        if value is None:
            self._fleet_env.latest_agent_observations.pop(self._uav_id, None)
        else:
            self._fleet_env.latest_agent_observations[self._uav_id] = value

    @property
    def _latest_evaluator_frame(self) -> EvaluatorFrame | None:
        return self._fleet_env.latest_evaluator_frames.get(
            (self._uav_id, self._target_id)
        )  # type: ignore[return-value]

    @_latest_evaluator_frame.setter
    def _latest_evaluator_frame(self, value: EvaluatorFrame | None) -> None:
        key = (self._uav_id, self._target_id)
        if value is None:
            self._fleet_env.latest_evaluator_frames.pop(key, None)
        else:
            self._fleet_env.latest_evaluator_frames[key] = value

    @property
    def _last_camera_timestamp_s(self) -> float | None:
        return self._fleet_env._last_camera_timestamps_s.get(self._uav_id)

    @_last_camera_timestamp_s.setter
    def _last_camera_timestamp_s(self, value: float | None) -> None:
        if value is None:
            self._fleet_env._last_camera_timestamps_s.pop(self._uav_id, None)
        else:
            self._fleet_env._last_camera_timestamps_s[self._uav_id] = value

    def setup(self) -> None:
        self._fleet_env.setup()

    def reset(self, *, target_seed: int | None = None) -> None:
        seeds = None if target_seed is None else {self._target_id: target_seed}
        self._fleet_env.reset(target_seeds=seeds)

    def get_agent_view(self) -> AgentView:
        return self._fleet_env.get_agent_view(self._uav_id)  # type: ignore[return-value]

    def make_skill_context(
        self,
        clock: SkillClock,
        *,
        perception: object | None = None,
    ) -> SkillContext:
        return self._fleet_env.make_skill_context(
            self._uav_id,
            clock,
            perception=perception,
        )  # type: ignore[return-value]

    def configure_overview_viewport(self) -> None:
        self._require_scene().configure_overview_viewport()

    def read_poses(self) -> ScenePoseState:
        return self._require_scene().read_poses()

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
        controller = self._require_uav_controller()
        current = controller.get_pose()
        position = (
            np.asarray([current.x, current.y, current.z], dtype=np.float64)
            if position_m is None
            else _position(position_m)
        )
        yaw = current.yaw if orientation_wxyz is None else _yaw_from_wxyz(orientation_wxyz)
        try:
            controller.set_pose(*position.tolist(), yaw)
        finally:
            self._invalidate_camera_observation()

    def set_target_pose(
        self,
        position_m: Sequence[float] | None = None,
        orientation_wxyz: Sequence[float] | None = None,
    ) -> None:
        motion = self._require_target_motion()
        current = motion.get_pose()
        position = (
            np.asarray([current.x, current.y, current.z], dtype=np.float64)
            if position_m is None
            else _position(position_m)
        )
        yaw = current.yaw if orientation_wxyz is None else _yaw_from_wxyz(orientation_wxyz)
        try:
            motion.reset(position_m=position, yaw_rad=yaw)
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

    def get_camera_sample(self) -> CameraSample:
        sample = self.get_agent_observation().camera_sample
        if sample is None:
            raise RuntimeError("latest observation does not contain an RGB-D Camera sample")
        return sample

    def get_rgb(self) -> np.ndarray:
        return self.get_camera_rgb()

    def save_rgb(self, path: str | Path) -> Path:
        observation = self.get_agent_observation()
        return Path(
            self._require_scene().save_camera_rgb(str(path), image=observation.rgb)
        )

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._require_scene().get_camera_pose()

    def world_to_image(
        self,
        points_xyz_m: Sequence[float] | Sequence[Sequence[float]],
    ) -> ImageProjection:
        return self._require_scene().world_to_image(points_xyz_m)

    def get_agent_observation(self) -> AgentObservation:
        return self._fleet_env.get_agent_observation(self._uav_id)  # type: ignore[return-value]

    def get_evaluator_frame(self) -> EvaluatorFrame:
        return self._fleet_env.get_evaluator_frame(
            self._uav_id,
            self._target_id,
        )  # type: ignore[return-value]

    def get_skill_observation(self, *, include_oracle: bool = False) -> Observation:
        if not isinstance(include_oracle, bool):
            raise TypeError("include_oracle must be bool")
        if not include_oracle:
            return self._fleet_env.get_skill_observation(
                self._uav_id,
                include_oracle=False,
            )  # type: ignore[return-value]

        # Privileged truth remains an explicit singleton compatibility path.
        # Fleet callers must use assignment-scoped EvaluatorFrame access.
        from skills.types import Observation

        evaluator = self.get_evaluator_frame()
        agent = evaluator.observation
        return Observation(
            uav_id=self._uav_id,
            timestamp=agent.camera_timestamp_s,
            uav_pose=agent.uav_state,
            uav_velocity=agent.uav_velocity_mps.copy(),
            camera_rgb=agent.rgb.copy(),
            camera_position_m=agent.camera_position_m.copy(),
            camera_orientation_wxyz=agent.camera_orientation_wxyz.copy(),
            oracle_target_id=self._target_id,
            oracle_target_visible=bool(evaluator.target_projection.visible[0]),
            oracle_target_pose=evaluator.target_state,
            oracle_target_velocity=evaluator.target_velocity_mps.copy(),
        )

    def _refresh_camera_observation(self) -> bool:
        snapshot = self._fleet_env.get_fleet_pose_snapshot()
        return self._fleet_env._refresh_all_observations(snapshot) > 0

    def _invalidate_camera_observation(self) -> None:
        self._fleet_env._invalidate_observation_caches()

    def _initial_target_position(self) -> np.ndarray:
        return self._fleet_env._initial_target_position(self._target_id)

    def step(self) -> bool:
        return self._fleet_env.step()

    def close(self) -> None:
        self._fleet_env.close()

    def _require_scene(self) -> UavSearchScene:
        return self._fleet_env._require_scene()  # type: ignore[return-value]

    def _require_uav_controller(self) -> KinematicUAV:
        return self._fleet_env._require_uav_controller(self._uav_id)  # type: ignore[return-value]

    def _require_target_motion(self) -> MovingTarget:
        return self._fleet_env._require_target_motion(self._target_id)  # type: ignore[return-value]


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


__all__ = [
    "AgentObservation",
    "AgentView",
    "EvaluatorFrame",
    "SimpleUavSearchEnv",
]
