from __future__ import annotations

import math
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from env.moving_target import MovingTarget, TargetState
from skills.track import TrackGoal, TrackSkill
from skills.types import Observation, SkillContext, SkillResultCode, SkillStatus


class ManualClock:
    def __init__(self, time_s: float = 0.0) -> None:
        self.time_s = time_s

    def now(self) -> float:
        return self.time_s

    def advance(self, dt_s: float) -> None:
        self.time_s += dt_s


class FakeCamera:
    def __init__(self, uav: KinematicUAV) -> None:
        self.uav = uav

    def get_rgb(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.uav.get_pose()
        half_yaw = state.yaw / 2.0
        # Match the fixed -45 deg pitch used by target_in_camera_fov().
        half_pitch_rotation = math.radians(45.0) / 2.0
        cy = math.cos(half_yaw)
        sy = math.sin(half_yaw)
        cp = math.cos(half_pitch_rotation)
        sp = math.sin(half_pitch_rotation)
        return (
            np.asarray([state.x, state.y, state.z], dtype=np.float64),
            np.asarray(
                [cy * cp, -sy * sp, cy * sp, sy * cp],
                dtype=np.float64,
            ),
        )


def make_uav(
    position_xyz_m: tuple[float, float, float],
    *,
    yaw: float = 0.0,
    max_speed: float = 5.0,
    max_yaw_rate: float = 10.0,
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
    )


def make_target(
    mode: str,
    *,
    speed_mps: float = 0.0,
    initial_heading_rad: float = 0.0,
    seed: int = 17,
    direction_change_interval_s: float = 0.5,
    bounds_xy: float = 20.0,
) -> MovingTarget:
    return MovingTarget(
        mode=mode,
        initial_position_xyz_m=(0.0, 0.0, 0.5),
        bounds_min_xyz_m=(-bounds_xy, -bounds_xy, 0.5),
        bounds_max_xyz_m=(bounds_xy, bounds_xy, 0.5),
        speed_mps=speed_mps,
        max_speed_mps=1.0,
        direction_change_interval_s=direction_change_interval_s,
        seed=seed,
        initial_heading_rad=initial_heading_rad,
    )


def wrapped_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def target_in_camera_fov(
    uav_state: UAVState,
    target_state: TargetState,
    *,
    horizontal_fov_deg: float = 90.0,
    vertical_fov_deg: float = 90.0,
    camera_pitch_deg: float = -45.0,
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
    forward_distance = float(np.dot(delta, forward))
    if forward_distance <= 0.0:
        return False
    horizontal_angle = math.atan2(float(np.dot(delta, left)), forward_distance)
    vertical_angle = math.atan2(float(np.dot(delta, up)), forward_distance)
    return (
        abs(horizontal_angle) <= math.radians(horizontal_fov_deg) / 2.0
        and abs(vertical_angle) <= math.radians(vertical_fov_deg) / 2.0
    )


def make_observation(
    uav: KinematicUAV,
    clock: ManualClock,
    target_pose: TargetState,
    target_velocity: np.ndarray,
    *,
    visible: bool | None = None,
    target_id: str = "target",
    timestamp: float | None = None,
) -> Observation:
    state = uav.get_pose()
    camera_position, camera_orientation = FakeCamera(uav).get_camera_pose()
    oracle_visible = (
        target_in_camera_fov(state, target_pose) if visible is None else visible
    )
    return Observation(
        timestamp=clock.now() if timestamp is None else timestamp,
        uav_pose=state,
        uav_velocity=uav.get_velocity(),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        camera_position_m=camera_position,
        camera_orientation_wxyz=camera_orientation,
        oracle_target_id=target_id,
        oracle_target_visible=oracle_visible,
        oracle_target_pose=target_pose,
        oracle_target_velocity=np.asarray(target_velocity, dtype=np.float64).copy(),
    )


def horizontal_distance(uav_state: UAVState, target_state: TargetState) -> float:
    return math.hypot(
        target_state.x - uav_state.x,
        target_state.y - uav_state.y,
    )


def run_tracking_steps(
    skill: TrackSkill,
    uav: KinematicUAV,
    target: MovingTarget,
    clock: ManualClock,
    *,
    steps: int,
    dt_s: float = 0.1,
) -> list[dict[str, object]]:
    feedback_history: list[dict[str, object]] = []
    for _ in range(steps):
        target_pose = target.get_pose()
        status = skill.tick(
            make_observation(
                uav,
                clock,
                target_pose,
                target.get_velocity(),
            )
        )
        if status is not SkillStatus.RUNNING:
            raise AssertionError(f"TRACK terminated unexpectedly with {status.name}")
        feedback_history.append(skill.get_feedback().data)
        uav.step(dt_s)
        target.step(dt_s)
        clock.advance(dt_s)
    return feedback_history


class TrackSkillTest(unittest.TestCase):
    def test_goal_defaults_and_validation(self) -> None:
        defaults = TrackGoal(target_id="target")
        self.assertEqual(defaults.desired_distance, 6.0)
        self.assertEqual(defaults.desired_altitude, 8.0)
        self.assertEqual(defaults.max_speed, 2.0)
        self.assertEqual(defaults.max_target_lost_time, 2.0)
        self.assertIsNone(defaults.timeout)
        self.assertIsNone(defaults.track_duration)

        invalid_goals = (
            TrackGoal(target_id=""),
            TrackGoal(target_id="target", desired_distance=0.0),
            TrackGoal(target_id="target", desired_altitude=0.0),
            TrackGoal(target_id="target", max_speed=0.0),
            TrackGoal(target_id="target", max_target_lost_time=0.0),
            TrackGoal(target_id="target", timeout=0.0),
            TrackGoal(target_id="target", track_duration=0.0),
        )
        for invalid_goal in invalid_goals:
            with self.subTest(goal=invalid_goal):
                uav = make_uav((-6.0, 0.0, 8.0))
                skill = TrackSkill()
                skill.start(invalid_goal, make_context(uav, ManualClock()))
                self.assertIs(skill.status, SkillStatus.FAILED)
                self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)

    def test_static_target_converges_and_holds_distance_and_altitude(self) -> None:
        target = make_target("STATIC")
        uav = make_uav((-10.0, 0.0, 5.0))
        clock = ManualClock()
        skill = TrackSkill()
        goal = TrackGoal(target_id="target")
        skill.start(goal, make_context(uav, clock))
        positions: list[np.ndarray] = []

        for _ in range(50):
            before = uav.get_pose()
            feedback = run_tracking_steps(skill, uav, target, clock, steps=1)[0]
            after = uav.get_pose()
            displacement = np.linalg.norm(
                np.asarray([after.x - before.x, after.y - before.y, after.z - before.z])
            )
            self.assertLessEqual(displacement, goal.max_speed * 0.1 + 1e-12)
            positions.append(np.asarray([after.x, after.y, after.z]))

        final_target = target.get_pose()
        self.assertAlmostEqual(
            horizontal_distance(uav.get_pose(), final_target),
            goal.desired_distance,
            delta=0.11,
        )
        self.assertAlmostEqual(uav.get_pose().z, goal.desired_altitude, delta=0.11)
        for position in positions[-10:]:
            np.testing.assert_allclose(position, positions[-1], atol=1e-12)
        required_feedback = {
            "target_distance",
            "distance_error",
            "target_visible",
            "target_relative_bearing",
            "last_seen_age",
            "tracking_duration",
        }
        self.assertTrue(required_feedback.issubset(feedback))
        self.assertTrue(feedback["target_visible"])

    def test_linear_target_is_followed(self) -> None:
        target = make_target(
            "LINEAR",
            speed_mps=0.4,
            initial_heading_rad=math.pi / 2.0,
        )
        uav = make_uav((-6.0, 0.0, 8.0))
        clock = ManualClock()
        skill = TrackSkill()
        goal = TrackGoal(target_id="target")
        skill.start(goal, make_context(uav, clock))

        run_tracking_steps(skill, uav, target, clock, steps=80)
        target_state = target.get_pose()
        self.assertGreater(uav.get_pose().y, 2.8)
        self.assertAlmostEqual(
            horizontal_distance(uav.get_pose(), target_state),
            goal.desired_distance,
            delta=0.2,
        )
        self.assertAlmostEqual(uav.get_pose().z, goal.desired_altitude, delta=0.1)

    def test_seeded_random_walk_target_is_followed(self) -> None:
        target = make_target(
            "RANDOM_WALK",
            speed_mps=0.25,
            seed=17,
            direction_change_interval_s=0.5,
            bounds_xy=3.0,
        )
        uav = make_uav((-6.0, 0.0, 8.0))
        clock = ManualClock()
        skill = TrackSkill()
        goal = TrackGoal(target_id="target")
        skill.start(goal, make_context(uav, clock))

        run_tracking_steps(skill, uav, target, clock, steps=150)
        self.assertAlmostEqual(
            horizontal_distance(uav.get_pose(), target.get_pose()),
            goal.desired_distance,
            delta=0.3,
        )
        self.assertAlmostEqual(uav.get_pose().z, goal.desired_altitude, delta=0.1)

    def test_yaw_continuously_faces_target(self) -> None:
        target = make_target(
            "LINEAR",
            speed_mps=0.4,
            initial_heading_rad=math.pi / 2.0,
        )
        uav = make_uav((-6.0, 0.0, 8.0), yaw=-1.0, max_yaw_rate=5.0)
        clock = ManualClock()
        skill = TrackSkill()
        skill.start(TrackGoal(target_id="target"), make_context(uav, clock))
        bearing_errors: list[float] = []

        for _ in range(60):
            target_state = target.get_pose()
            self.assertIs(
                skill.tick(
                    make_observation(
                        uav,
                        clock,
                        target_state,
                        target.get_velocity(),
                    )
                ),
                SkillStatus.RUNNING,
            )
            uav.step(0.1)
            target.step(0.1)
            clock.advance(0.1)
            current_target = target.get_pose()
            bearing = math.atan2(
                current_target.y - uav.get_pose().y,
                current_target.x - uav.get_pose().x,
            )
            bearing_errors.append(abs(wrapped_angle(bearing - uav.get_pose().yaw)))

        self.assertGreater(bearing_errors[0], 0.1)
        self.assertLess(max(bearing_errors[-20:]), 0.05)

    def test_uav_can_side_fly_while_facing_target(self) -> None:
        target = make_target(
            "LINEAR",
            speed_mps=0.4,
            initial_heading_rad=math.pi / 2.0,
        )
        uav = make_uav((-6.0, 0.0, 8.0), yaw=-0.5, max_yaw_rate=5.0)
        clock = ManualClock()
        skill = TrackSkill()
        skill.start(TrackGoal(target_id="target"), make_context(uav, clock))

        initial_target = target.get_pose()
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    initial_target,
                    target.get_velocity(),
                )
            ),
            SkillStatus.RUNNING,
        )
        target.step(0.5)
        clock.advance(0.5)
        moved_target = target.get_pose()
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    moved_target,
                    target.get_velocity(),
                )
            ),
            SkillStatus.RUNNING,
        )
        velocity = uav.get_velocity()
        course = math.atan2(velocity[1], velocity[0])
        target_bearing = math.atan2(
            moved_target.y - uav.get_pose().y,
            moved_target.x - uav.get_pose().x,
        )
        self.assertGreater(np.linalg.norm(velocity[:2]), 0.0)
        self.assertGreater(abs(wrapped_angle(course - target_bearing)), 1.0)

        before = uav.get_pose()
        after = uav.step(0.1)
        self.assertGreater(after.y, before.y)
        self.assertNotEqual(after.yaw, before.yaw)

    def test_short_visibility_loss_does_not_fail_and_recovery_updates_seen_time(self) -> None:
        target = make_target("STATIC")
        uav = make_uav((-6.0, 0.0, 8.0))
        clock = ManualClock()
        skill = TrackSkill()
        goal = TrackGoal(target_id="target", max_target_lost_time=0.5)
        skill.start(goal, make_context(uav, clock))
        target_pose = target.get_pose()

        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    target_pose,
                    target.get_velocity(),
                    visible=True,
                )
            ),
            SkillStatus.RUNNING,
        )
        for timestamp in (0.2, 0.4):
            clock.time_s = timestamp
            self.assertIs(
                skill.tick(
                    make_observation(
                        uav,
                        clock,
                        target_pose,
                        target.get_velocity(),
                        visible=False,
                    )
                ),
                SkillStatus.RUNNING,
            )
            self.assertAlmostEqual(skill.get_feedback().data["last_seen_age"], timestamp)

        clock.time_s = 0.5
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    target_pose,
                    target.get_velocity(),
                    visible=True,
                )
            ),
            SkillStatus.RUNNING,
        )
        self.assertTrue(skill.get_feedback().data["target_visible"])
        self.assertAlmostEqual(skill.get_feedback().data["last_seen_age"], 0.0)

    def test_long_visibility_loss_returns_last_seen_state(self) -> None:
        uav = make_uav((-6.0, 0.0, 8.0))
        clock = ManualClock()
        skill = TrackSkill()
        goal = TrackGoal(target_id="target", max_target_lost_time=0.5)
        skill.start(goal, make_context(uav, clock))
        seen_pose = TargetState(1.0, 2.0, 0.5, 0.3)
        seen_velocity = np.asarray([0.1, 0.2, 0.0])

        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    seen_pose,
                    seen_velocity,
                    visible=True,
                )
            ),
            SkillStatus.RUNNING,
        )
        hidden_pose = TargetState(4.0, 5.0, 0.5, 1.0)
        hidden_velocity = np.asarray([0.8, 0.7, 0.0])
        clock.time_s = 0.5
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    hidden_pose,
                    hidden_velocity,
                    visible=False,
                )
            ),
            SkillStatus.RUNNING,
        )
        clock.time_s = 0.6
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    hidden_pose,
                    hidden_velocity,
                    visible=False,
                )
            ),
            SkillStatus.FAILED,
        )

        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.TARGET_LOST)
        self.assertEqual(result.data["target_id"], "target")
        np.testing.assert_allclose(result.data["last_seen_position"], (1.0, 2.0, 0.5))
        np.testing.assert_allclose(result.data["last_seen_velocity"], (0.1, 0.2, 0.0))
        self.assertEqual(result.data["last_seen_time"], 0.0)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        stopped_pose = uav.get_pose()
        self.assertEqual(uav.step(1.0), stopped_pose)

    def test_late_visible_frame_cannot_revive_an_already_lost_track(self) -> None:
        target = make_target("STATIC")
        uav = make_uav((-6.0, 0.0, 8.0))
        clock = ManualClock()
        skill = TrackSkill()
        skill.start(
            TrackGoal(target_id="target", max_target_lost_time=0.5),
            make_context(uav, clock),
        )
        first_pose = target.get_pose()
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    first_pose,
                    target.get_velocity(),
                    visible=True,
                )
            ),
            SkillStatus.RUNNING,
        )

        clock.time_s = 0.6
        late_pose = TargetState(3.0, 4.0, 0.5, 0.0)
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    late_pose,
                    np.asarray([0.8, 0.7, 0.0]),
                    visible=True,
                )
            ),
            SkillStatus.FAILED,
        )
        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.TARGET_LOST)
        np.testing.assert_allclose(
            result.data["last_seen_position"],
            (first_pose.x, first_pose.y, first_pose.z),
        )
        self.assertEqual(result.data["last_seen_time"], 0.0)

    def test_delayed_visible_frame_ages_from_capture_timestamp(self) -> None:
        target = make_target("STATIC")
        uav = make_uav((-6.0, 0.0, 8.0))
        clock = ManualClock()
        skill = TrackSkill()
        skill.start(
            TrackGoal(target_id="target", max_target_lost_time=1.0),
            make_context(uav, clock),
        )

        clock.time_s = 1.0
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    target.get_pose(),
                    target.get_velocity(),
                    visible=True,
                    timestamp=0.5,
                )
            ),
            SkillStatus.RUNNING,
        )
        self.assertAlmostEqual(skill.get_feedback().data["last_seen_age"], 0.5)

        clock.time_s = 1.6
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    target.get_pose(),
                    target.get_velocity(),
                    visible=False,
                )
            ),
            SkillStatus.FAILED,
        )
        self.assertIs(skill.get_result().code, SkillResultCode.TARGET_LOST)
        self.assertEqual(skill.get_result().data["last_seen_time"], 0.5)

    def test_track_duration_succeeds_and_stops(self) -> None:
        target = make_target("STATIC")
        uav = make_uav((-6.0, 0.0, 8.0))
        clock = ManualClock()
        skill = TrackSkill()
        skill.start(
            TrackGoal(target_id="target", track_duration=0.3),
            make_context(uav, clock),
        )

        for timestamp in (0.0, 0.1, 0.2):
            clock.time_s = timestamp
            self.assertIs(
                skill.tick(
                    make_observation(
                        uav,
                        clock,
                        target.get_pose(),
                        target.get_velocity(),
                        visible=True,
                    )
                ),
                SkillStatus.RUNNING,
            )

        clock.time_s = 0.3
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    target.get_pose(),
                    target.get_velocity(),
                    visible=True,
                )
            ),
            SkillStatus.SUCCEEDED,
        )
        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.TRACK_COMPLETE)
        self.assertEqual(result.data["target_id"], "target")
        self.assertAlmostEqual(result.data["tracking_duration"], 0.3)
        self.assertAlmostEqual(result.data["track_duration"], 0.3)
        self.assertEqual(skill.get_feedback().progress, 1.0)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))

    def test_terminal_code_uses_the_earliest_absolute_deadline(self) -> None:
        cases = (
            (
                "completion first",
                2.0,
                2.0,
                1.0,
                3.0,
                SkillStatus.SUCCEEDED,
                SkillResultCode.TRACK_COMPLETE,
            ),
            (
                "timeout first",
                2.0,
                1.0,
                3.0,
                3.0,
                SkillStatus.FAILED,
                SkillResultCode.TIMEOUT,
            ),
            (
                "loss first",
                1.0,
                3.0,
                2.0,
                3.0,
                SkillStatus.FAILED,
                SkillResultCode.TARGET_LOST,
            ),
            (
                "timeout and loss tie",
                1.0,
                1.0,
                None,
                1.1,
                SkillStatus.FAILED,
                SkillResultCode.TIMEOUT,
            ),
            (
                "completion and timeout tie",
                2.0,
                1.0,
                1.0,
                1.0,
                SkillStatus.SUCCEEDED,
                SkillResultCode.TRACK_COMPLETE,
            ),
            (
                "completion and loss tie",
                1.0,
                2.0,
                1.0,
                1.0,
                SkillStatus.SUCCEEDED,
                SkillResultCode.TRACK_COMPLETE,
            ),
        )
        for (
            name,
            lost_time,
            timeout,
            track_duration,
            final_time,
            expected_status,
            expected_code,
        ) in cases:
            with self.subTest(name=name):
                target = make_target("STATIC")
                uav = make_uav((-6.0, 0.0, 8.0))
                clock = ManualClock()
                skill = TrackSkill()
                skill.start(
                    TrackGoal(
                        target_id="target",
                        max_target_lost_time=lost_time,
                        timeout=timeout,
                        track_duration=track_duration,
                    ),
                    make_context(uav, clock),
                )
                self.assertIs(
                    skill.tick(
                        make_observation(
                            uav,
                            clock,
                            target.get_pose(),
                            target.get_velocity(),
                            visible=True,
                        )
                    ),
                    SkillStatus.RUNNING,
                )

                clock.time_s = final_time
                self.assertIs(
                    skill.tick(
                        make_observation(
                            uav,
                            clock,
                            target.get_pose(),
                            target.get_velocity(),
                            visible=False,
                        )
                    ),
                    expected_status,
                )
                self.assertIs(skill.get_result().code, expected_code)

    def test_optional_timeout_fails_and_stops(self) -> None:
        target = make_target("STATIC")
        uav = make_uav((-6.0, 0.0, 8.0))
        clock = ManualClock()
        skill = TrackSkill()
        skill.start(
            TrackGoal(target_id="target", timeout=0.3),
            make_context(uav, clock),
        )

        for timestamp in (0.0, 0.1, 0.2):
            clock.time_s = timestamp
            self.assertIs(
                skill.tick(
                    make_observation(
                        uav,
                        clock,
                        target.get_pose(),
                        target.get_velocity(),
                        visible=True,
                    )
                ),
                SkillStatus.RUNNING,
            )
        clock.time_s = 0.3
        self.assertIs(
            skill.tick(
                make_observation(
                    uav,
                    clock,
                    target.get_pose(),
                    target.get_velocity(),
                    visible=True,
                )
            ),
            SkillStatus.FAILED,
        )
        self.assertIs(skill.get_result().code, SkillResultCode.TIMEOUT)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
