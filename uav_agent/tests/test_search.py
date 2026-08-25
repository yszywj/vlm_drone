from __future__ import annotations

import math
import unittest

import numpy as np

from env.kinematic_uav import KinematicUAV, UAVState
from env.moving_target import MovingTarget, TargetState
from skills.motion_types import YawMode
from skills.search import SearchGoal, SearchPhase, SearchSkill
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
        return camera_pose(state)


def make_uav(
    position_xyz_m: tuple[float, float, float],
    *,
    yaw: float = 0.0,
    max_speed: float = 5.0,
    max_yaw_rate: float = 5.0,
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


def camera_pose(
    state: UAVState,
    *,
    camera_pitch_deg: float = -45.0,
) -> tuple[np.ndarray, np.ndarray]:
    half_yaw = state.yaw / 2.0
    # Rz(yaw) * Ry(-pitch) maps the camera's +X axis to the same
    # forward/down direction used by target_in_camera_fov().
    half_pitch_rotation = -math.radians(camera_pitch_deg) / 2.0
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


def make_observation(
    uav: KinematicUAV,
    clock: ManualClock,
    *,
    target_pose: TargetState | None,
    target_velocity: np.ndarray | None = None,
    visible: bool | None,
    target_id: str | None = "target",
) -> Observation:
    state = uav.get_pose()
    camera_position, camera_orientation = camera_pose(state)
    return Observation(
        uav_id="uav_1",
        timestamp=clock.now(),
        uav_pose=state,
        uav_velocity=uav.get_velocity(),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        camera_position_m=camera_position,
        camera_orientation_wxyz=camera_orientation,
        oracle_target_id=target_id if target_pose is not None else None,
        oracle_target_visible=visible,
        oracle_target_pose=target_pose,
        oracle_target_velocity=(
            np.zeros(3, dtype=np.float64)
            if target_pose is not None and target_velocity is None
            else target_velocity
        ),
    )


def target_in_camera_fov(
    uav_state: UAVState,
    target_state: TargetState,
    *,
    horizontal_fov_deg: float = 90.0,
    vertical_fov_deg: float = 90.0,
    camera_pitch_deg: float = -45.0,
) -> bool:
    """Pure-math oracle matching the project's forward/down Camera convention."""

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


def run_with_oracle(
    skill: SearchSkill,
    uav: KinematicUAV,
    clock: ManualClock,
    *,
    static_target: TargetState | None = None,
    moving_target: MovingTarget | None = None,
    force_visible: bool | None = None,
    dt_s: float = 0.1,
    max_steps: int = 5000,
) -> tuple[SkillStatus, list[tuple[Observation, SkillStatus]]]:
    history: list[tuple[Observation, SkillStatus]] = []
    for _ in range(max_steps):
        target_pose = (
            moving_target.get_pose() if moving_target is not None else static_target
        )
        target_velocity = (
            moving_target.get_velocity() if moving_target is not None else None
        )
        visible = (
            force_visible
            if force_visible is not None
            else False
            if target_pose is None
            else target_in_camera_fov(uav.get_pose(), target_pose)
        )
        observation = make_observation(
            uav,
            clock,
            target_pose=target_pose,
            target_velocity=target_velocity,
            visible=visible,
        )
        status = skill.tick(observation)
        history.append((observation, status))
        if status is not SkillStatus.RUNNING:
            return status, history
        uav.step(dt_s)
        if moving_target is not None:
            moving_target.step(dt_s)
        clock.advance(dt_s)
    raise AssertionError("SEARCH did not reach a terminal state")


def search_goal(**overrides: object) -> SearchGoal:
    values: dict[str, object] = {
        "center": (0.0, 0.0, 0.5),
        "radius": 2.0,
        "target_description": "person wearing a red jacket",
        "search_altitude": 5.0,
        "transit_speed": 2.0,
        "scan_yaw_rate": 2.0,
        "timeout": 100.0,
    }
    values.update(overrides)
    return SearchGoal(**values)  # type: ignore[arg-type]


class SearchSkillTest(unittest.TestCase):
    def test_provisional_candidate_never_directly_completes_search(self) -> None:
        uav = make_uav((0.0, 0.0, 5.0))
        skill = SearchSkill()
        skill.start(search_goal(), make_context(uav, ManualClock()))

        skill.report_candidate_pending("candidate_1", source="qwen_vl")

        self.assertIs(skill.status, SkillStatus.RUNNING)
        self.assertIsNone(skill.get_result())
        self.assertIs(skill.phase, SearchPhase.CANDIDATE_PENDING)
        feedback = skill.get_feedback().data
        self.assertEqual(feedback["phase"], "CANDIDATE_PENDING")
        self.assertEqual(feedback["candidate_id"], "candidate_1")
        skill.mark_waiting_for_review("candidate_1")
        self.assertEqual(skill.get_feedback().data["phase"], "WAITING_FOR_REVIEW")
        skill.reject_candidate("candidate_1")
        self.assertIs(skill.phase, SearchPhase.TRANSIT)
        self.assertNotIn("candidate_id", skill.get_feedback().data)

    def test_goal_defaults_and_validation(self) -> None:
        defaults = SearchGoal(
            center=(1.0, 2.0, 0.5),
            radius=10.0,
            target_description="vehicle",
            search_altitude=8.0,
        )
        self.assertEqual(defaults.transit_speed, 1.5)
        self.assertEqual(defaults.scan_yaw_rate, 0.5)
        self.assertEqual(defaults.timeout, 60.0)
        self.assertIs(SearchSkill().transit_yaw_mode, YawMode.FACE_POINT)

        invalid_goals = (
            search_goal(radius=0.0),
            search_goal(target_description="  "),
            search_goal(search_altitude=0.0),
            search_goal(transit_speed=0.0),
            search_goal(scan_yaw_rate=0.0),
            search_goal(timeout=0.0),
        )
        for invalid_goal in invalid_goals:
            with self.subTest(goal=invalid_goal):
                uav = make_uav((0.0, 0.0, 5.0))
                skill = SearchSkill()
                skill.start(invalid_goal, make_context(uav, ManualClock()))
                self.assertIs(skill.status, SkillStatus.FAILED)
                self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)

    def test_center_target_enters_fov_and_returns_synchronized_snapshot(self) -> None:
        uav = make_uav((-6.0, -6.0, 5.0), yaw=-math.pi / 2.0)
        clock = ManualClock()
        skill = SearchSkill()
        skill.start(search_goal(), make_context(uav, clock))
        target = TargetState(0.0, 0.0, 0.5, 0.0)

        status, history = run_with_oracle(
            skill,
            uav,
            clock,
            static_target=target,
        )
        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertFalse(history[0][0].oracle_target_visible)
        self.assertTrue(
            all(
                not observation.oracle_target_visible
                for observation, _ in history[:-1]
            )
        )
        found_observation = history[-1][0]
        self.assertTrue(found_observation.oracle_target_visible)
        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.TARGET_FOUND)
        self.assertEqual(result.data["target_id"], "target")
        self.assertEqual(result.data["found_timestamp"], found_observation.timestamp)
        self.assertEqual(
            result.data["uav_pose"],
            {
                "x": found_observation.uav_pose.x,
                "y": found_observation.uav_pose.y,
                "z": found_observation.uav_pose.z,
                "yaw": found_observation.uav_pose.yaw,
            },
        )
        self.assertEqual(
            result.data["camera_pose"]["position_m"],
            tuple(found_observation.camera_position_m),
        )
        self.assertEqual(
            result.data["camera_pose"]["orientation_wxyz"],
            tuple(found_observation.camera_orientation_wxyz),
        )
        self.assertEqual(
            result.data["target_position_world_m"],
            (0.0, 0.0, 0.5),
        )
        self.assertEqual(
            result.data["target_velocity_world_mps"],
            (0.0, 0.0, 0.0),
        )
        self.assertNotIn("oracle_target_pose", result.data)
        self.assertNotIn("oracle_target_velocity_mps", result.data)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))

    def test_target_on_search_region_edge_is_found(self) -> None:
        uav = make_uav((-6.0, -6.0, 5.0), yaw=-math.pi / 2.0)
        clock = ManualClock()
        skill = SearchSkill()
        skill.start(search_goal(), make_context(uav, clock))
        edge_target = TargetState(2.0, 0.0, 0.5, 0.0)

        status, _ = run_with_oracle(
            skill,
            uav,
            clock,
            static_target=edge_target,
        )
        self.assertIs(status, SkillStatus.SUCCEEDED)
        self.assertIs(skill.get_result().code, SkillResultCode.TARGET_FOUND)

    def test_seeded_random_walk_target_is_found(self) -> None:
        target = MovingTarget(
            mode="RANDOM_WALK",
            initial_position_xyz_m=(0.0, 0.0, 0.5),
            bounds_min_xyz_m=(-1.5, -1.5, 0.5),
            bounds_max_xyz_m=(1.5, 1.5, 0.5),
            speed_mps=0.2,
            max_speed_mps=1.0,
            direction_change_interval_s=0.7,
            seed=17,
            initial_heading_rad=0.0,
        )
        initial_target = target.get_pose()
        uav = make_uav((-6.0, -6.0, 5.0), yaw=-math.pi / 2.0)
        clock = ManualClock()
        skill = SearchSkill()
        skill.start(search_goal(), make_context(uav, clock))

        status, _ = run_with_oracle(
            skill,
            uav,
            clock,
            moving_target=target,
        )
        self.assertIs(status, SkillStatus.SUCCEEDED)
        final_target = target.get_pose()
        self.assertGreater(
            math.hypot(final_target.x - initial_target.x, final_target.y - initial_target.y),
            0.0,
        )
        self.assertIs(skill.get_result().code, SkillResultCode.TARGET_FOUND)

    def test_default_face_point_transit_moves_and_turns_together(self) -> None:
        uav = make_uav((8.0, 0.0, 5.0), yaw=0.0, max_yaw_rate=10.0)
        clock = ManualClock()
        skill = SearchSkill()
        skill.start(search_goal(), make_context(uav, clock))
        initial = uav.get_pose()
        observation = make_observation(
            uav,
            clock,
            target_pose=None,
            visible=False,
        )
        self.assertIs(skill.tick(observation), SkillStatus.RUNNING)
        self.assertEqual(uav.get_pose(), initial)
        self.assertGreater(np.linalg.norm(uav.get_velocity()), 0.0)

        before_error = abs(wrapped_delta(math.pi, initial.yaw))
        updated = uav.step(0.1)
        center_bearing = math.atan2(-updated.y, -updated.x)
        after_error = abs(wrapped_delta(center_bearing, updated.yaw))
        self.assertNotEqual((updated.x, updated.y), (initial.x, initial.y))
        self.assertNotEqual(updated.yaw, initial.yaw)
        self.assertLess(after_error, before_error)

    def test_six_continuous_full_scans_end_in_search_exhausted(self) -> None:
        uav = make_uav((0.0, 0.0, 5.0), max_speed=10.0, max_yaw_rate=5.0)
        clock = ManualClock()
        goal = search_goal(transit_speed=10.0, scan_yaw_rate=2.0, timeout=100.0)
        skill = SearchSkill(transit_yaw_mode=YawMode.KEEP_CURRENT)
        skill.start(goal, make_context(uav, clock))
        accumulated_by_waypoint = {index: 0.0 for index in range(1, 7)}
        dt_s = 0.05

        for _ in range(5000):
            observation = make_observation(
                uav,
                clock,
                target_pose=None,
                visible=False,
            )
            status = skill.tick(observation)
            if status is not SkillStatus.RUNNING:
                break
            feedback = skill.get_feedback().data
            before = uav.get_pose()
            after = uav.step(dt_s)
            if feedback["phase"] == "SCANNING":
                delta = wrapped_delta(after.yaw, before.yaw)
                self.assertGreaterEqual(delta, -1e-12)
                self.assertAlmostEqual(delta, goal.scan_yaw_rate * dt_s, places=10)
                accumulated_by_waypoint[int(feedback["waypoint_index"])] += max(0.0, delta)
                self.assertAlmostEqual(after.x, before.x)
                self.assertAlmostEqual(after.y, before.y)
                self.assertAlmostEqual(after.z, before.z)
            clock.advance(dt_s)
        else:
            self.fail("SEARCH did not exhaust its fixed waypoints")

        self.assertIs(status, SkillStatus.FAILED)
        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.SEARCH_EXHAUSTED)
        self.assertEqual(result.data["completed_scans"], 6)
        for waypoint_index, angle in accumulated_by_waypoint.items():
            with self.subTest(waypoint=waypoint_index):
                self.assertGreaterEqual(angle, 2.0 * math.pi - 1e-9)
                self.assertLessEqual(angle, 2.0 * math.pi + goal.scan_yaw_rate * dt_s)

    def test_hidden_oracle_pose_never_changes_route_or_causes_detection(self) -> None:
        goal = search_goal(
            radius=1.0,
            transit_speed=5.0,
            scan_yaw_rate=3.0,
            timeout=100.0,
        )
        first = make_uav((0.0, 0.0, 5.0), max_speed=5.0, max_yaw_rate=5.0)
        second = make_uav((0.0, 0.0, 5.0), max_speed=5.0, max_yaw_rate=5.0)
        first_clock = ManualClock()
        second_clock = ManualClock()
        first_skill = SearchSkill(transit_yaw_mode=YawMode.KEEP_CURRENT)
        second_skill = SearchSkill(transit_yaw_mode=YawMode.KEEP_CURRENT)
        first_skill.start(goal, make_context(first, first_clock))
        second_skill.start(goal, make_context(second, second_clock))
        dt_s = 0.1

        for step_index in range(3000):
            first_status = first_skill.tick(
                make_observation(
                    first,
                    first_clock,
                    target_pose=None,
                    visible=False,
                )
            )
            decoy = TargetState(
                0.1 * math.cos(step_index * 0.2),
                0.1 * math.sin(step_index * 0.2),
                0.5,
                step_index * 0.01,
            )
            second_status = second_skill.tick(
                make_observation(
                    second,
                    second_clock,
                    target_pose=decoy,
                    target_velocity=np.asarray([0.1, 0.1, 0.0]),
                    visible=False,
                    target_id="decoy",
                )
            )
            self.assertIs(first_status, second_status)
            self.assertEqual(first.get_pose(), second.get_pose())
            np.testing.assert_array_equal(first.get_velocity(), second.get_velocity())
            if first_status is not SkillStatus.RUNNING:
                break
            self.assertEqual(first.step(dt_s), second.step(dt_s))
            first_clock.advance(dt_s)
            second_clock.advance(dt_s)
        else:
            self.fail("paired SEARCH runs did not terminate")

        self.assertIs(first_status, SkillStatus.FAILED)
        self.assertIs(first_skill.get_result().code, SkillResultCode.SEARCH_EXHAUSTED)
        self.assertIs(second_skill.get_result().code, SkillResultCode.SEARCH_EXHAUSTED)

    def test_timeout_fails_and_stops(self) -> None:
        uav = make_uav((0.0, 0.0, 5.0), max_speed=1.0)
        clock = ManualClock()
        skill = SearchSkill()
        skill.start(
            search_goal(
                radius=20.0,
                transit_speed=0.1,
                timeout=0.2,
            ),
            make_context(uav, clock),
        )
        for _ in range(2):
            status = skill.tick(
                make_observation(
                    uav,
                    clock,
                    target_pose=None,
                    visible=False,
                )
            )
            self.assertIs(status, SkillStatus.RUNNING)
            uav.step(0.1)
            clock.advance(0.1)

        status = skill.tick(
            make_observation(
                uav,
                clock,
                target_pose=None,
                visible=False,
            )
        )
        self.assertIs(status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.TIMEOUT)
        np.testing.assert_array_equal(uav.get_velocity(), np.zeros(3))
        stopped_pose = uav.get_pose()
        self.assertEqual(uav.step(1.0), stopped_pose)

    def test_target_first_visible_after_deadline_is_still_timeout(self) -> None:
        uav = make_uav((0.0, 0.0, 5.0))
        clock = ManualClock()
        skill = SearchSkill()
        skill.start(search_goal(timeout=0.2), make_context(uav, clock))
        clock.time_s = 1.0
        late_target = TargetState(1.0, 0.0, 0.5, 0.0)

        status = skill.tick(
            make_observation(
                uav,
                clock,
                target_pose=late_target,
                visible=True,
            )
        )
        self.assertIs(status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.TIMEOUT)


if __name__ == "__main__":
    unittest.main()
