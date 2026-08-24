from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from configs.loader import load_config
from env.fleet_uav_search_env import FleetUavSearchEnv
from env.moving_target import TargetState
from env.observation_types import AgentObservation, AgentView, EvaluatorFrame
from env.simple_uav_search_env import SimpleUavSearchEnv
from env.uav_controller import UAVState
from perception import (
    GuardedPerceptionBackend,
    OraclePerception,
    PerceptionRuntimeProfile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _environment() -> SimpleUavSearchEnv:
    return SimpleUavSearchEnv(load_config(PROJECT_ROOT / "configs" / "default.yaml"))


def _observation() -> AgentObservation:
    return AgentObservation(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        uav_state=UAVState(1.0, 2.0, 3.0, 0.25),
        uav_velocity_mps=np.asarray([0.1, 0.2, 0.3]),
        camera_position_m=np.asarray([1.0, 2.0, 2.8]),
        camera_orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        camera_timestamp_s=1.5,
        camera_sample=SimpleNamespace(sample_id="rgbd"),
    )


def _frame(observation: AgentObservation) -> EvaluatorFrame:
    return EvaluatorFrame(
        observation=observation,
        target_position_m=np.asarray([4.0, 5.0, 0.5]),
        target_orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        target_state=TargetState(4.0, 5.0, 0.5, 0.0),
        target_velocity_mps=np.asarray([0.2, 0.0, 0.0]),
        target_projection=SimpleNamespace(visible=np.asarray([True])),
    )


def test_singleton_is_composed_from_one_fleet_environment() -> None:
    environment = _environment()

    assert isinstance(environment._fleet_env, FleetUavSearchEnv)
    assert environment._fleet_env.uav_ids == (environment.config.uav.id,)
    assert environment._fleet_env.target_ids == (environment.config.target.id,)
    assert dict(environment._fleet_env.assignments) == {
        environment.config.uav.id: environment.config.target.id
    }
    assert environment.world is None
    assert environment.scene is None
    assert environment.uav_controller is None
    assert environment._target_motion is None


def test_lifecycle_and_single_seed_are_adapted_without_a_second_owner() -> None:
    environment = _environment()
    fleet = environment._fleet_env
    fleet.setup = Mock()
    fleet.reset = Mock()
    fleet.step = Mock(return_value=True)
    fleet.close = Mock()

    environment.setup()
    environment.reset(target_seed=17)
    assert environment.step() is True
    environment.close()

    fleet.setup.assert_called_once_with()
    fleet.reset.assert_called_once_with(
        target_seeds={environment.config.target.id: 17}
    )
    fleet.step.assert_called_once_with()
    fleet.close.assert_called_once_with()


def test_legacy_world_entity_cache_and_timestamp_attributes_proxy_fleet_state() -> None:
    environment = _environment()
    world = object()
    scene = object()
    uav = object()
    target = object()
    observation = _observation()
    frame = _frame(observation)

    environment.world = world
    environment.scene = scene
    environment.uav_controller = uav
    environment._target_motion = target
    environment._latest_agent_observation = observation
    environment._latest_evaluator_frame = frame
    environment._last_camera_timestamp_s = 2.5

    assert environment._fleet_env.world is world
    assert environment._fleet_env.scene is scene
    assert environment._fleet_env.uav_controllers[environment.config.uav.id] is uav
    assert environment._fleet_env.target_motions[environment.config.target.id] is target
    assert environment._latest_agent_observation is observation
    assert environment._latest_evaluator_frame is frame
    assert environment._last_camera_timestamp_s == 2.5


def test_control_scene_camera_and_pose_calls_are_forwarded_with_singleton_ids(
    tmp_path: Path,
) -> None:
    environment = _environment()
    pose_state = SimpleNamespace(
        uav_position=np.asarray([1.0, 2.0, 3.0]),
        uav_orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
        target_position=np.asarray([4.0, 5.0, 0.5]),
        target_orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    scene = SimpleNamespace(
        configure_overview_viewport=Mock(),
        read_poses=Mock(return_value=pose_state),
        save_camera_rgb=Mock(return_value=str(tmp_path / "frame.png")),
        get_camera_pose=Mock(return_value=(np.zeros(3), np.asarray([1, 0, 0, 0]))),
        world_to_image=Mock(return_value="projection"),
    )
    controller = SimpleNamespace(
        get_pose=Mock(return_value=UAVState(1.0, 2.0, 3.0, 0.25)),
        set_pose=Mock(),
        set_velocity=Mock(),
        move_toward=Mock(),
        rotate_yaw=Mock(),
        stop=Mock(),
        distance_to_goal=Mock(return_value=3.0),
        heading_error=Mock(return_value=0.2),
        goal_reached=Mock(return_value=False),
    )
    motion = SimpleNamespace(
        get_pose=Mock(return_value=TargetState(4.0, 5.0, 0.5, 0.4)),
        reset=Mock(),
    )
    observation = _observation()
    environment.scene = scene
    environment.uav_controller = controller
    environment._target_motion = motion
    environment._latest_agent_observation = observation

    environment.configure_overview_viewport()
    assert environment.read_poses() is pose_state
    np.testing.assert_array_equal(environment.uav_position, pose_state.uav_position)
    np.testing.assert_array_equal(environment.target_position, pose_state.target_position)
    environment.set_uav_velocity([1.0, 0.0, 0.0], 0.1)
    environment.move_uav_toward([2.0, 3.0, 4.0], 2.0, face_goal=False)
    environment.rotate_uav_yaw(0.8, relative=True)
    environment.stop_uav()
    assert environment.distance_to_goal() == 3.0
    assert environment.heading_error() == 0.2
    assert environment.goal_reached() is False
    assert environment.get_camera_rgb().shape == (2, 3, 3)
    assert environment.get_camera_sample().sample_id == "rgbd"
    assert environment.save_rgb(tmp_path / "frame.png") == tmp_path / "frame.png"
    environment.get_camera_pose()
    assert environment.world_to_image([1.0, 2.0, 3.0]) == "projection"

    controller.set_velocity.assert_called_once_with([1.0, 0.0, 0.0], 0.1)
    controller.move_toward.assert_called_once_with(
        [2.0, 3.0, 4.0],
        2.0,
        face_goal=False,
        tolerance_m=None,
        max_yaw_rate_rad_s=None,
    )
    controller.rotate_yaw.assert_called_once_with(
        0.8, relative=True, max_yaw_rate_rad_s=None
    )
    scene.world_to_image.assert_called_once_with([1.0, 2.0, 3.0])


def test_pose_mutation_and_observations_use_fleet_caches() -> None:
    environment = _environment()
    controller = SimpleNamespace(
        get_pose=Mock(return_value=UAVState(1.0, 2.0, 3.0, 0.25)),
        set_pose=Mock(),
    )
    motion = SimpleNamespace(
        get_pose=Mock(return_value=TargetState(4.0, 5.0, 0.5, 0.4)),
        reset=Mock(),
    )
    environment.uav_controller = controller
    environment._target_motion = motion
    observation = _observation()
    frame = _frame(observation)
    environment._latest_agent_observation = observation
    environment._latest_evaluator_frame = frame

    copied_observation = environment.get_agent_observation()
    copied_frame = environment.get_evaluator_frame()
    assert copied_observation is not observation
    assert copied_frame is not frame
    assert environment.get_skill_observation().oracle_target_id is None
    oracle = environment.get_skill_observation(include_oracle=True)
    assert oracle.oracle_target_id == environment.config.target.id
    assert oracle.oracle_target_visible is True

    environment.set_uav_pose([7.0, 8.0, 9.0])
    controller.set_pose.assert_called_once_with(7.0, 8.0, 9.0, 0.25)
    assert environment._fleet_env.latest_agent_observations == {}
    assert environment._fleet_env.latest_evaluator_frames == {}

    environment._latest_agent_observation = observation
    environment._latest_evaluator_frame = frame
    environment.set_target_pose(orientation_wxyz=[1.0, 0.0, 0.0, 0.0])
    reset_kwargs = motion.reset.call_args.kwargs
    np.testing.assert_array_equal(reset_kwargs["position_m"], [4.0, 5.0, 0.5])
    assert reset_kwargs["yaw_rad"] == 0.0
    assert environment._fleet_env.latest_agent_observations == {}


def test_agent_view_and_skill_context_delegate_with_the_single_uav_id() -> None:
    environment = _environment()
    fleet = environment._fleet_env
    view = AgentView(
        observe=Mock(),
        move_toward=Mock(),
        set_velocity=Mock(),
        rotate_yaw=Mock(),
        stop=Mock(),
        distance_to_goal=Mock(),
        heading_error=Mock(),
        goal_reached=Mock(),
    )
    context = object()
    clock = object()
    perception = object()
    fleet.get_agent_view = Mock(return_value=view)
    fleet.make_skill_context = Mock(return_value=context)

    assert environment.get_agent_view() is view
    assert environment.make_skill_context(clock, perception=perception) is context
    fleet.get_agent_view.assert_called_once_with(environment.config.uav.id)
    fleet.make_skill_context.assert_called_once_with(
        environment.config.uav.id,
        clock,
        perception=perception,
    )


def test_guarded_oracle_remains_compatible_with_fleet_assignment_validation() -> None:
    environment = _environment()
    controller = object()
    sensor = object()
    environment.uav_controller = controller
    environment._fleet_env.camera_sensors[environment.config.uav.id] = sensor
    guarded = GuardedPerceptionBackend(
        OraclePerception(
            uav_id=environment.config.uav.id,
            target_id=environment.config.target.id,
        ),
        profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
        acknowledge_privileged_oracle=True,
    )

    context = environment.make_skill_context(object(), perception=guarded)

    assert context.uav is controller
    assert context.camera is sensor
    assert context.perception is guarded
