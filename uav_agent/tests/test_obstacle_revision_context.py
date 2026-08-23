from __future__ import annotations

from math import pi
import unittest

import numpy as np

from common.obstacle_types import ObstacleAABB
from perception.qwen_vlm_verifier import VisualReviewFrame
from planner.route_types import RouteConstraints
from planner.spatial import CoordinateFrame, PointTarget
from planner.spatial_resolver import FramePose
from runtime.frame_store import FrameRef
from runtime.obstacle_revision_context import (
    build_obstacle_revision_request,
    hold_relative_obstacle_geometry,
    hold_relative_point_target,
)
from skills.plan import TaskPlan, TaskStep
from skills.types import SkillName


class ObstacleRevisionContextTest(unittest.TestCase):
    def test_world_rejoin_point_is_rotated_into_uav_hold_flu(self) -> None:
        target = hold_relative_point_target(
            world_xyz_m=(10.0, 3.0, 12.0),
            hold_pose=FramePose((10.0, 0.0, 10.0), pi / 2.0),
        )
        self.assertEqual(target.frame, CoordinateFrame.UAV_HOLD_FLU)
        for actual, expected in zip(target.xyz_m, (3.0, 0.0, 2.0)):
            self.assertAlmostEqual(actual, expected)

    def test_world_aabb_is_rotated_into_uav_hold_flu(self) -> None:
        geometry = hold_relative_obstacle_geometry(
            obstacle_id="box_red",
            world_aabb=ObstacleAABB((9.0, 1.0, 8.0), (11.0, 3.0, 12.0)),
            hold_pose=FramePose((10.0, 0.0, 10.0), pi / 2.0),
        )
        for actual, expected in zip(
            geometry.relative_aabb_min_m,
            (1.0, -1.0, -2.0),
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            geometry.relative_aabb_max_m,
            (3.0, 1.0, 2.0),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_request_preserves_prefix_and_records_elapsed_time(self) -> None:
        plan = TaskPlan(
            (
                TaskStep("takeoff_1", SkillName.TAKEOFF, {"target_altitude": 10.0}),
                TaskStep("goto_1", SkillName.GOTO, {"position": (10.0, 0.0, 10.0)}),
                TaskStep("land_1", SkillName.LAND, {"ground_altitude": 0.0}),
            ),
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
        )
        ref = FrameRef("uav_1", "frame_1", 4.0, 8, 6)
        frame = VisualReviewFrame(ref, np.zeros((6, 8, 3), dtype=np.uint8))
        geometry = hold_relative_obstacle_geometry(
            obstacle_id="box_red",
            world_aabb=ObstacleAABB((2, -1, 0), (4, 1, 2)),
            hold_pose=FramePose((0, 0, 10)),
        )
        request = build_obstacle_revision_request(
            original_instruction="avoid and continue",
            original_plan_summary=plan.to_dict(),
            active_plan=plan,
            replace_from_step_id="goto_1",
            route_id="route_1",
            frame=frame,
            grounded_geometry=geometry,
            active_corridor_rejoin_target=PointTarget(
                CoordinateFrame.UAV_HOLD_FLU,
                (10.0, 0.0, 0.0),
            ),
            mission_elapsed_s=4.0,
            route_constraints=RouteConstraints(),
        )
        self.assertEqual(request.completed_prefix_summary[0]["id"], "takeoff_1")
        self.assertEqual(request.current_step_summary["status"], "INTERRUPTED_BY_OBSTACLE")
        self.assertEqual(request.remaining_plan_summary[0]["id"], "land_1")
        self.assertEqual(request.mission_elapsed_s, 4.0)
        self.assertEqual(request.visual_assessment.hazards[0].obstacle_id, "box_red")


if __name__ == "__main__":
    unittest.main()
