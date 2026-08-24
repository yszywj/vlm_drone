from __future__ import annotations

import numpy as np

from common.target_estimate import TargetEstimate
from env.kinematic_uav import KinematicUAV
from env.uav_controller import UAVState
from skills.reacquire import ReacquireGoal, ReacquireSkill
from skills.search import SearchGoal, SearchPhase, SearchSkill
from skills.track import TrackGoal, TrackSkill
from skills.types import Observation, SkillContext, SkillResultCode, SkillStatus


class _Clock:
    def __init__(self, timestamp_s: float = 0.0) -> None:
        self.timestamp_s = timestamp_s

    def now(self) -> float:
        return self.timestamp_s


class _Camera:
    def __init__(self, uav: KinematicUAV) -> None:
        self._uav = uav

    def get_rgb(self) -> np.ndarray:
        return np.zeros((12, 16, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pose = self._uav.get_pose()
        return (
            np.asarray((pose.x, pose.y, pose.z), dtype=np.float64),
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        )


def _runtime() -> tuple[KinematicUAV, _Clock, SkillContext]:
    uav = KinematicUAV(
        UAVState(0.0, 0.0, 5.0, 0.0),
        max_speed_mps=3.0,
        max_yaw_rate_rad_s=2.0,
    )
    clock = _Clock()
    context = SkillContext(
        uav=uav,
        camera=_Camera(uav),
        perception=None,
        clock=clock,
        uav_id="uav_1",
    )
    return uav, clock, context


def _estimate(
    *,
    target_id: str | None = "target_1",
    candidate_id: str | None = "candidate_1",
    tracker_id: str | None = "track_7",
    visible: bool = True,
    confirmed: bool = True,
    predicted_only: bool = False,
    position_world_m: tuple[float, float, float] | None = (5.0, 0.0, 0.5),
) -> TargetEstimate:
    return TargetEstimate(
        timestamp_s=0.0,
        target_id=target_id,
        candidate_id=candidate_id,
        tracker_id=tracker_id,
        visible=visible,
        confirmed=confirmed,
        predicted_only=predicted_only,
        class_id=0,
        class_name="person",
        confidence=0.91,
        bbox_xyxy_normalized=(0.25, 0.20, 0.55, 0.85) if visible else None,
        position_world_m=position_world_m,
        velocity_world_mps=(0.2, 0.0, 0.0),
        measurement_age_s=0.0,
        source="yolo26_botsort",
    )


def _observation(
    uav: KinematicUAV,
    clock: _Clock,
    estimate: TargetEstimate | None,
) -> Observation:
    pose = uav.get_pose()
    return Observation(
        uav_id="uav_1",
        timestamp=clock.now(),
        uav_pose=pose,
        uav_velocity=uav.get_velocity(),
        camera_rgb=np.zeros((12, 16, 3), dtype=np.uint8),
        camera_position_m=np.asarray((pose.x, pose.y, pose.z), dtype=np.float64),
        camera_orientation_wxyz=np.asarray(
            (1.0, 0.0, 0.0, 0.0), dtype=np.float64
        ),
        target_estimate=estimate,
    )


def _search_goal() -> SearchGoal:
    return SearchGoal(
        center=(0.0, 0.0, 0.5),
        radius=2.0,
        target_description="person",
        search_altitude=5.0,
        transit_speed=1.0,
        scan_yaw_rate=0.5,
        timeout=30.0,
    )


def test_search_succeeds_from_confirmed_production_target_estimate() -> None:
    uav, clock, context = _runtime()
    skill = SearchSkill()
    skill.start(_search_goal(), context)

    assert skill.tick(_observation(uav, clock, _estimate())) is SkillStatus.SUCCEEDED
    result = skill.get_result()
    assert result is not None
    assert result.code is SkillResultCode.TARGET_FOUND
    assert result.data["target_id"] == "target_1"
    assert result.data["perception_source"] == "yolo26_botsort"
    assert "oracle_target_pose" not in result.data


def test_search_keeps_unconfirmed_detection_in_candidate_state() -> None:
    uav, clock, context = _runtime()
    skill = SearchSkill()
    skill.start(_search_goal(), context)
    candidate = _estimate(target_id=None, confirmed=False)

    assert skill.tick(_observation(uav, clock, candidate)) is SkillStatus.RUNNING
    assert skill.phase is SearchPhase.CANDIDATE_PENDING
    feedback = skill.get_feedback().data
    assert feedback["candidate_id"] == "candidate_1"
    assert feedback["candidate_source"] == "yolo26_botsort"


def test_track_consumes_neutral_estimate_and_never_requires_oracle_fields() -> None:
    uav, clock, context = _runtime()
    skill = TrackSkill()
    skill.start(
        TrackGoal(
            target_id="target_1",
            desired_distance=3.0,
            desired_altitude=5.0,
            max_speed=1.0,
            max_target_lost_time=2.0,
            timeout=20.0,
        ),
        context,
    )

    assert skill.tick(_observation(uav, clock, _estimate())) is SkillStatus.RUNNING
    feedback = skill.get_feedback().data
    assert feedback["target_visible"] is True
    assert feedback["perception_source"] == "yolo26_botsort"
    assert feedback["tracker_id"] == "track_7"
    assert feedback["position_available"] is True


def test_track_does_not_fabricate_position_when_depth_is_unavailable() -> None:
    uav, clock, context = _runtime()
    skill = TrackSkill()
    skill.start(TrackGoal(target_id="target_1"), context)

    status = skill.tick(
        _observation(uav, clock, _estimate(position_world_m=None))
    )
    assert status is SkillStatus.FAILED
    result = skill.get_result()
    assert result is not None
    assert result.code is SkillResultCode.INVALID_STATE
    assert result.data["position_available"] is False


def test_reacquire_succeeds_only_for_matching_confirmed_target_estimate() -> None:
    uav, clock, context = _runtime()
    skill = ReacquireSkill()
    skill.start(
        ReacquireGoal(
            target_id="target_1",
            last_seen_position=(4.0, 0.0, 0.5),
            last_seen_velocity=(0.0, 0.0, 0.0),
            last_seen_time=0.0,
            search_radius=2.0,
            timeout=20.0,
        ),
        context,
    )

    assert skill.tick(_observation(uav, clock, _estimate())) is SkillStatus.SUCCEEDED
    result = skill.get_result()
    assert result is not None
    assert result.code is SkillResultCode.TARGET_FOUND
    assert result.data["perception_source"] == "yolo26_botsort"
    assert result.data["tracker_id"] == "track_7"
    assert "oracle_target_pose" not in result.data

