from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from env.moving_target import TargetState
from skills.motion_types import YawMode, move_toward_with_policy
from skills.reacquire import ReacquireGoal, ReacquireSkill
from skills.types import Observation, SkillContext, SkillResultCode, SkillStatus


class ManualClock:
    def __init__(self, time_s: float = 0.0) -> None:
        self.time_s = time_s

    def now(self) -> float:
        return self.time_s

    def advance(self, dt_s: float) -> None:
        self.time_s += dt_s


def camera_pose(
    state: UAVState,
    *,
    camera_pitch_deg: float = -45.0,
) -> tuple[np.ndarray, np.ndarray]:
    half_yaw = state.yaw / 2.0
    half_pitch_rotation = -math.radians(camera_pitch_deg) / 2.0
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    cp = math.cos(half_pitch_rotation)
    sp = math.sin(half_pitch_rotation)
    return (
        np.asarray([state.x, state.y, state.z], dtype=np.float64),
        np.asarray([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float64),
    )


class FakeCamera:
    def __init__(self, uav: KinematicUAV) -> None:
        self.uav = uav

    def get_rgb(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return camera_pose(self.uav.get_pose())


def make_uav(
    position_xyz_m: tuple[float, float, float],
    *,
    yaw: float = 0.0,
    max_speed: float = 2.0,
    max_yaw_rate: float = 2.0,
) -> KinematicUAV:
    return KinematicUAV(
        UAVState(*position_xyz_m, yaw),
        max_speed_mps=max_speed,
        max_yaw_rate_rad_s=max_yaw_rate,
    )


def make_context(uav: KinematicUAV, clock: ManualClock) -> SkillContext:
    return SkillContext(
        uav=uav,
        camera=FakeCamera(uav),
        perception=None,
        clock=clock,
        uav_id="uav_1",
    )


def make_observation(
    uav: KinematicUAV,
    clock: ManualClock,
    *,
    visible: bool,
    target_pose: TargetState | None = None,
    target_velocity: np.ndarray | None = None,
    target_id: str | None = "target",
) -> Observation:
    camera_position, camera_orientation = camera_pose(uav.get_pose())
    return Observation(
        uav_id="uav_1",
        timestamp=clock.now(),
        uav_pose=uav.get_pose(),
        uav_velocity=uav.get_velocity(),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        camera_position_m=camera_position,
        camera_orientation_wxyz=camera_orientation,
        oracle_target_id=target_id if target_pose is not None else None,
        oracle_target_visible=visible,
        oracle_target_pose=target_pose,
        oracle_target_velocity=(
            target_velocity
            if target_velocity is not None
            else np.zeros(3, dtype=np.float64)
            if target_pose is not None
            else None
        ),
    )


def target_in_camera_fov(
    uav_state: UAVState,
    target_state: TargetState,
    *,
    camera_pitch_deg: float = -45.0,
    horizontal_fov_deg: float = 90.0,
    vertical_fov_deg: float = 90.0,
) -> bool:
    yaw = uav_state.yaw
    pitch = math.radians(camera_pitch_deg)
    forward = np.asarray(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ]
    )
    left = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.cross(forward, left)
    delta = np.asarray(
        [
            target_state.x - uav_state.x,
            target_state.y - uav_state.y,
            target_state.z - uav_state.z,
        ]
    )
    camera_forward = float(np.dot(delta, forward))
    if camera_forward <= 0.0:
        return False
    horizontal_angle = math.atan2(float(np.dot(delta, left)), camera_forward)
    vertical_angle = math.atan2(float(np.dot(delta, up)), camera_forward)
    return (
        abs(horizontal_angle) <= math.radians(horizontal_fov_deg) / 2.0
        and abs(vertical_angle) <= math.radians(vertical_fov_deg) / 2.0
    )


def wrapped_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


def reacquire_goal(**overrides: object) -> ReacquireGoal:
    values: dict[str, object] = {
        "target_id": "target",
        "last_seen_position": (0.0, 0.0, 0.5),
        "last_seen_velocity": (0.0, 0.0, 0.0),
        "last_seen_time": 0.0,
        "search_radius": 1.0,
        "timeout": 100.0,
    }
    values.update(overrides)
    return ReacquireGoal(**values)  # type: ignore[arg-type]


def advance_physics(
    uav: KinematicUAV,
    clock: ManualClock,
    duration_s: float,
    *,
    physics_dt_s: float = 0.05,
) -> None:
    steps = round(duration_s / physics_dt_s)
    for _ in range(steps):
        uav.step(physics_dt_s)
        clock.advance(physics_dt_s)


class ReacquireSkillTest(unittest.TestCase):
    def test_goal_defaults_and_validation(self) -> None:
        defaults = ReacquireGoal(
            target_id="target",
            last_seen_position=(1.0, 2.0, 0.5),
            last_seen_velocity=(0.1, 0.2, 0.0),
            last_seen_time=3.0,
        )
        self.assertEqual(defaults.search_radius, 10.0)
        self.assertEqual(defaults.timeout, 30.0)

        invalid_goals = (
            reacquire_goal(target_id=""),
            reacquire_goal(last_seen_position=(1.0, float("inf"), 0.5)),
            reacquire_goal(last_seen_velocity=(0.0, True, 0.0)),
            reacquire_goal(last_seen_time=-1.0),
            reacquire_goal(search_radius=0.0),
            reacquire_goal(timeout=0.0),
        )
        for invalid_goal in invalid_goals:
            with self.subTest(goal=invalid_goal):
                uav = make_uav((0.0, 0.0, 5.0))
                skill = ReacquireSkill()
                skill.start(invalid_goal, make_context(uav, ManualClock()))
                self.assertIs(skill.status, SkillStatus.FAILED)
                self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)

    def test_future_last_seen_time_is_invalid_state(self) -> None:
        uav = make_uav((0.0, 0.0, 5.0))
        skill = ReacquireSkill()
        skill.start(
            reacquire_goal(last_seen_time=1.0),
            make_context(uav, ManualClock(0.5)),
        )
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_STATE)

    def test_static_last_seen_position_is_approached_without_changing_altitude(self) -> None:
        uav = make_uav((-2.0, 0.0, 5.0), max_speed=1.0)
        clock = ManualClock()
        skill = ReacquireSkill()
        skill.start(
            reacquire_goal(search_radius=0.2),
            make_context(uav, clock),
        )
        self.assertEqual(
            skill.get_feedback().data["predicted_position"],
            (0.0, 0.0, 0.5),
        )
        start_pose = uav.get_pose()
        with patch(
            "skills.reacquire.move_toward_with_policy",
            wraps=move_toward_with_policy,
        ) as mocked_move:
            self.assertIs(
                skill.tick(make_observation(uav, clock, visible=False)),
                SkillStatus.RUNNING,
            )
        policy = mocked_move.call_args.args[4]
        self.assertIs(policy.yaw_mode, YawMode.FACE_POINT)
        self.assertEqual(policy.look_at_point, (0.0, 0.0, 0.5))
        self.assertEqual(uav.get_pose(), start_pose)

        previous = start_pose
        for _ in range(30):
            current = uav.step(0.1)
            clock.advance(0.1)
            self.assertAlmostEqual(current.z, start_pose.z)
            self.assertGreaterEqual(current.x + 1e-12, previous.x)
            self.assertLessEqual(current.x - previous.x, 0.1 + 1e-12)
            previous = current
            self.assertIs(
                skill.tick(make_observation(uav, clock, visible=False)),
                SkillStatus.RUNNING,
            )
            if skill.get_feedback().data["phase"] == "SCANNING":
                break
        else:
            self.fail("REACQUIRE never entered its local scan")
        self.assertAlmostEqual(uav.get_pose().z, 5.0)
        self.assertLessEqual(abs(uav.get_pose().x), 0.2 + 1e-12)

    def test_nonzero_velocity_prediction_is_frozen_and_faced_during_transit(self) -> None:
        uav = make_uav(
            (-10.0, -5.0, 5.0),
            yaw=-1.0,
            max_speed=1.0,
            max_yaw_rate=20.0,
        )
        clock = ManualClock(10.0)
        skill = ReacquireSkill()
        goal = reacquire_goal(
            last_seen_position=(1.0, 2.0, 0.5),
            last_seen_velocity=(0.5, -0.25, 0.0),
            last_seen_time=4.0,
            search_radius=0.1,
        )
        skill.start(goal, make_context(uav, clock))
        predicted = (4.0, 0.5, 0.5)
        self.assertEqual(skill.get_feedback().data["predicted_position"], predicted)

        self.assertIs(
            skill.tick(make_observation(uav, clock, visible=False)),
            SkillStatus.RUNNING,
        )
        before = uav.get_pose()
        after = uav.step(0.1)
        clock.advance(0.1)
        self.assertNotEqual((after.x, after.y), (before.x, before.y))
        self.assertNotEqual(after.yaw, before.yaw)
        expected_yaw = math.atan2(predicted[1] - after.y, predicted[0] - after.x)
        self.assertAlmostEqual(wrapped_delta(after.yaw, expected_yaw), 0.0, places=12)
        self.assertAlmostEqual(after.z, before.z)

        self.assertIs(
            skill.tick(make_observation(uav, clock, visible=False)),
            SkillStatus.RUNNING,
        )
        self.assertEqual(skill.get_feedback().data["predicted_position"], predicted)

    def test_hidden_truth_and_wrong_target_id_do_not_change_recovery_goal(self) -> None:
        uav = make_uav((-5.0, 0.0, 5.0), max_speed=1.0)
        clock = ManualClock()
        skill = ReacquireSkill()
        skill.start(reacquire_goal(search_radius=0.1), make_context(uav, clock))
        hidden_pose = TargetState(100.0, 100.0, 0.5, 0.0)
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    visible=False,
                    target_pose=hidden_pose,
                    target_velocity=np.asarray([50.0, 50.0, 0.0]),
                )
            ),
            SkillStatus.RUNNING,
        )
        self.assertEqual(
            skill.get_feedback().data["predicted_position"],
            (0.0, 0.0, 0.5),
        )
        np.testing.assert_allclose(uav.get_velocity(), (1.0, 0.0, 0.0))

        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    visible=True,
                    target_pose=hidden_pose,
                    target_id="other-target",
                )
            ),
            SkillStatus.RUNNING,
        )
        self.assertIsNone(skill.get_result())
        np.testing.assert_allclose(uav.get_velocity(), (1.0, 0.0, 0.0))

    def test_full_yaw_scan_is_continuous_stationary_and_repeats(self) -> None:
        uav = make_uav((0.0, 0.0, 5.0), yaw=0.2, max_yaw_rate=1.0)
        clock = ManualClock()
        skill = ReacquireSkill()
        skill.start(reacquire_goal(timeout=100.0), make_context(uav, clock))
        cached = make_observation(uav, clock, visible=False)
        self.assertIs(skill.tick(cached), SkillStatus.RUNNING)
        self.assertEqual(skill.get_feedback().data["phase"], "SCANNING")
        fixed_position = np.asarray([uav.get_pose().x, uav.get_pose().y, uav.get_pose().z])
        previous_yaw = uav.get_pose().yaw
        advance_physics(uav, clock, 0.05)
        accumulated_yaw = wrapped_delta(uav.get_pose().yaw, previous_yaw)
        previous_yaw = uav.get_pose().yaw
        self.assertIs(skill.tick(cached), SkillStatus.RUNNING)
        self.assertEqual(skill.get_feedback().data["scan_angle_rad"], 0.0)

        camera_period_s = 0.2
        for _ in range(150):
            advance_physics(uav, clock, camera_period_s)
            current = uav.get_pose()
            accumulated_yaw += wrapped_delta(current.yaw, previous_yaw)
            previous_yaw = current.yaw
            np.testing.assert_allclose(
                [current.x, current.y, current.z],
                fixed_position,
                atol=1e-12,
            )
            self.assertIs(
                skill.tick(make_observation(uav, clock, visible=False)),
                SkillStatus.RUNNING,
            )
            if skill.get_feedback().data["completed_scans"] >= 2:
                break
        else:
            self.fail("REACQUIRE did not continue into a second 360 degree scan")

        self.assertGreaterEqual(accumulated_yaw, 4.0 * math.pi - 1e-9)
        self.assertLessEqual(
            accumulated_yaw,
            4.0 * math.pi + ReacquireSkill.SCAN_YAW_RATE_RAD_S * camera_period_s,
        )
        self.assertIsNone(skill.get_result())

    def test_target_is_found_only_after_entering_camera_fov(self) -> None:
        target = TargetState(5.0, 0.0, 0.5, 0.0)
        uav = make_uav((0.0, 0.0, 5.0), yaw=-math.pi, max_yaw_rate=1.0)
        clock = ManualClock()
        skill = ReacquireSkill()
        skill.start(
            reacquire_goal(
                last_seen_position=(5.0, 0.0, 0.5),
                search_radius=10.0,
                timeout=20.0,
            ),
            make_context(uav, clock),
        )

        visibility_history: list[bool] = []
        for _ in range(100):
            visible = target_in_camera_fov(uav.get_pose(), target)
            visibility_history.append(visible)
            status = skill.tick(
                make_observation(
                    uav,
                    clock,
                    visible=visible,
                    target_pose=target,
                )
            )
            if status is not SkillStatus.RUNNING:
                break
            advance_physics(uav, clock, 0.1)
        else:
            self.fail("REACQUIRE did not find a target entering the FOV")

        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertIs(skill.get_result().code, SkillResultCode.TARGET_FOUND)
        self.assertTrue(visibility_history[-1])
        self.assertTrue(all(not visible for visible in visibility_history[:-1]))
        result = skill.get_result().data
        self.assertEqual(result["target_id"], "target")
        self.assertEqual(result["found_timestamp"], clock.now())
        self.assertEqual(result["predicted_position"], (5.0, 0.0, 0.5))
        self.assertIn("camera_pose", result)
        self.assertIn("oracle_target_pose", result)

    def test_timeout_is_the_only_normal_failure_and_stops_scan(self) -> None:
        uav = make_uav((0.0, 0.0, 5.0), max_yaw_rate=1.0)
        clock = ManualClock()
        skill = ReacquireSkill()
        skill.start(reacquire_goal(timeout=1.0), make_context(uav, clock))
        self.assertIs(
            skill.tick(make_observation(uav, clock, visible=False)),
            SkillStatus.RUNNING,
        )
        uav.step(0.5)
        clock.time_s = 0.5
        self.assertIs(
            skill.tick(make_observation(uav, clock, visible=False)),
            SkillStatus.RUNNING,
        )
        uav.step(0.5)
        clock.time_s = 1.0
        self.assertIs(
            skill.tick(make_observation(uav, clock, visible=False)),
            SkillStatus.FAILED,
        )
        self.assertIs(skill.get_result().code, SkillResultCode.TIMEOUT)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        stopped_pose = uav.get_pose()
        self.assertEqual(uav.step(1.0), stopped_pose)

    def test_matching_visible_frame_at_timeout_deadline_succeeds(self) -> None:
        target = TargetState(5.0, 0.0, 0.5, 0.0)
        uav = make_uav((0.0, 0.0, 5.0))
        clock = ManualClock()
        skill = ReacquireSkill()
        skill.start(
            reacquire_goal(
                last_seen_position=(5.0, 0.0, 0.5),
                timeout=1.0,
            ),
            make_context(uav, clock),
        )
        clock.time_s = 1.0
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    visible=True,
                    target_pose=target,
                )
            ),
            SkillStatus.SUCCEEDED,
        )
        self.assertIs(skill.get_result().code, SkillResultCode.TARGET_FOUND)


if __name__ == "__main__":
    unittest.main()
