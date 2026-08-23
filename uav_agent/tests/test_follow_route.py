from __future__ import annotations

import unittest

import numpy as np

from env.uav_controller import UAVState
from skills.follow_route import FollowRouteGoal, FollowRouteSkill
from skills.types import Observation, SkillResultCode, SkillStatus
from tests.test_goto import ManualClock, make_context, make_uav


def _observation(
    clock: ManualClock,
    position: tuple[float, float, float],
) -> Observation:
    pose = UAVState(*position, 0.0)
    return Observation(
        timestamp=clock.now(),
        uav_pose=pose,
        uav_velocity=np.zeros(3),
        camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        uav_id="uav_1",
    )


class FollowRouteTest(unittest.TestCase):
    def test_visits_bounded_waypoints_and_completes(self) -> None:
        uav, clock = make_uav((0.0, 0.0, 0.0)), ManualClock()
        ctx = make_context(uav, clock)
        skill = FollowRouteSkill()
        goal = FollowRouteGoal(
            "route_1",
            ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            tolerance_m=0.1,
            timeout_s=20.0,
        )
        skill.start(goal, ctx)
        for x in (0.0, 1.0, 2.0):
            clock.advance(1.0)
            skill.tick(_observation(clock, (x, 0.0, 0.0)))
            if skill.status is SkillStatus.SUCCEEDED:
                break
        self.assertEqual(skill.status, SkillStatus.SUCCEEDED)
        self.assertEqual(skill.get_result().code, SkillResultCode.ROUTE_COMPLETE)
        self.assertEqual(skill.get_result().data["visited_waypoints"], 2)

    def test_timeout_does_not_skip_to_unvisited_waypoint(self) -> None:
        uav, clock = make_uav((0.0, 0.0, 0.0)), ManualClock()
        ctx = make_context(uav, clock)
        skill = FollowRouteSkill()
        skill.start(
            FollowRouteGoal(
                "route_2",
                ((10.0, 0.0, 0.0), (20.0, 0.0, 0.0)),
                timeout_s=1.0,
            ),
            ctx,
        )
        clock.advance(1.0)
        skill.tick(_observation(clock, (0.0, 0.0, 0.0)))
        self.assertEqual(skill.status, SkillStatus.FAILED)
        self.assertEqual(skill.get_result().code, SkillResultCode.TIMEOUT)
        self.assertEqual(skill.get_result().data["waypoint_index"], 0)

    def test_goal_rejects_duplicate_or_unbounded_route(self) -> None:
        uav, clock = make_uav((0.0, 0.0, 0.0)), ManualClock()
        ctx = make_context(uav, clock)
        skill = FollowRouteSkill()
        skill.start(
            FollowRouteGoal("route_3", ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            ctx,
        )
        self.assertEqual(skill.status, SkillStatus.FAILED)
        self.assertEqual(skill.get_result().code, SkillResultCode.INVALID_GOAL)
        with self.assertRaisesRegex(ValueError, "between 2 and 16"):
            FollowRouteGoal("route_4", ((1.0, 0.0, 0.0),))


if __name__ == "__main__":
    unittest.main()
