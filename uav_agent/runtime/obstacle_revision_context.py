"""Trusted context construction for obstacle-aware route replanning.

This module converts only camera-visible, registry-backed obstacle geometry
into the hold-relative contract.  It never selects a route or changes model
waypoints; those responsibilities remain with Qwen and the Route Critic.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, sin

from common.obstacle_types import ObstacleAABB
from perception.qwen_vlm_verifier import VisualReviewFrame
from perception.runtime_visual_assessment import (
    RuntimeHazardAssessment,
    RuntimeTargetAssessment,
    RuntimeVisualAction,
    RuntimeVisualAssessmentV2,
    RuntimeVisualDecision,
    TargetAssessmentStatus,
    TaskProgressAssessment,
)
from planner.obstacle_revision import (
    GroundedObstacleGeometry,
    ObstacleAwareRevisionRequest,
)
from planner.route_types import RouteConstraints
from planner.spatial import CoordinateFrame, PointTarget
from planner.spatial_resolver import FramePose
from skills.plan import TaskPlan


def hold_relative_obstacle_geometry(
    *,
    obstacle_id: str,
    world_aabb: ObstacleAABB,
    hold_pose: FramePose,
) -> GroundedObstacleGeometry:
    """Return the tight yaw-only FLU AABB of one visible world AABB."""

    if not isinstance(world_aabb, ObstacleAABB):
        raise TypeError("world_aabb must be an ObstacleAABB")
    if not isinstance(hold_pose, FramePose):
        raise TypeError("hold_pose must be a FramePose")
    c, s = cos(hold_pose.yaw_rad), sin(hold_pose.yaw_rad)
    local: list[tuple[float, float, float]] = []
    for point in world_aabb.corners_xyz_m():
        dx = point[0] - hold_pose.xyz_m[0]
        dy = point[1] - hold_pose.xyz_m[1]
        local.append(
            (
                c * dx + s * dy,
                -s * dx + c * dy,
                point[2] - hold_pose.xyz_m[2],
            )
        )
    minimum = tuple(min(point[index] for point in local) for index in range(3))
    maximum = tuple(max(point[index] for point in local) for index in range(3))
    return GroundedObstacleGeometry(
        obstacle_id=obstacle_id,
        frame=CoordinateFrame.UAV_HOLD_FLU,
        relative_aabb_min_m=minimum,
        relative_aabb_max_m=maximum,
    )


def hold_relative_point_target(
    *,
    world_xyz_m: tuple[float, float, float],
    hold_pose: FramePose,
) -> PointTarget:
    """Express one trusted world point in the anchored UAV HOLD frame."""

    if not isinstance(hold_pose, FramePose):
        raise TypeError("hold_pose must be a FramePose")
    world = PointTarget(CoordinateFrame.WORLD_ENU, world_xyz_m)
    dx = world.xyz_m[0] - hold_pose.xyz_m[0]
    dy = world.xyz_m[1] - hold_pose.xyz_m[1]
    c, s = cos(hold_pose.yaw_rad), sin(hold_pose.yaw_rad)
    return PointTarget(
        CoordinateFrame.UAV_HOLD_FLU,
        (
            c * dx + s * dy,
            -s * dx + c * dy,
            world.xyz_m[2] - hold_pose.xyz_m[2],
        ),
    )


def grounded_runtime_assessment(
    *,
    review_id: str,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    frame: VisualReviewFrame,
    obstacle_id: str,
) -> RuntimeVisualAssessmentV2:
    """Describe a trusted low-level PATH_BLOCKED observation for route Qwen."""

    if not isinstance(frame, VisualReviewFrame):
        raise TypeError("frame must be a VisualReviewFrame")
    return RuntimeVisualAssessmentV2(
        review_id=review_id,
        mission_id=mission_id,
        uav_id=uav_id,
        plan_version=plan_version,
        observation_timestamp_s=frame.ref.timestamp_s,
        frame_id=frame.ref.frame_id,
        decision=RuntimeVisualDecision.PATH_BLOCKED,
        task_progress_assessment=TaskProgressAssessment(
            current_step_consistent=True,
            current_step_blocked=True,
            original_mission_still_achievable=True,
        ),
        target_assessment=RuntimeTargetAssessment(
            TargetAssessmentStatus.NO_TARGET,
            False,
            None,
        ),
        hazards=(
            RuntimeHazardAssessment(
                obstacle_id=obstacle_id,
                present=True,
                blocks_active_corridor=True,
                visual_confidence=1.0,
                geometry_grounded=True,
            ),
        ),
        recommended_action=RuntimeVisualAction.REQUEST_REPLAN,
        reason_codes=(
            "VISIBLE_OBSTACLE",
            "ACTIVE_CORRIDOR_BLOCKED",
            "LOW_LEVEL_GEOMETRY_GROUNDED",
        ),
    )


def build_obstacle_revision_request(
    *,
    original_instruction: str,
    original_plan_summary: Mapping[str, object],
    active_plan: TaskPlan,
    replace_from_step_id: str,
    route_id: str,
    frame: VisualReviewFrame,
    grounded_geometry: GroundedObstacleGeometry,
    active_corridor_rejoin_target: PointTarget,
    mission_elapsed_s: float,
    route_constraints: RouteConstraints = RouteConstraints(),
) -> ObstacleAwareRevisionRequest:
    """Build a routed request while preserving the completed plan prefix."""

    if not isinstance(active_plan, TaskPlan):
        raise TypeError("active_plan must be a TaskPlan")
    try:
        index = next(
            index
            for index, step in enumerate(active_plan.steps)
            if step.step_id == replace_from_step_id
        )
    except StopIteration:
        raise ValueError("replace_from_step_id is not present in active_plan") from None
    assessment = grounded_runtime_assessment(
        review_id=f"review_route_{route_id}",
        mission_id=active_plan.mission_id,
        uav_id=active_plan.uav_id,
        plan_version=active_plan.plan_version,
        frame=frame,
        obstacle_id=grounded_geometry.obstacle_id,
    )
    interrupted = active_plan.steps[index]
    return ObstacleAwareRevisionRequest(
        mission_id=active_plan.mission_id,
        uav_id=active_plan.uav_id,
        base_plan_version=active_plan.plan_version,
        new_plan_version=active_plan.plan_version + 1,
        route_id=route_id,
        replace_from_step_id=replace_from_step_id,
        original_instruction=original_instruction,
        original_plan_summary=original_plan_summary,
        completed_prefix_summary=tuple(
            step.to_dict() for step in active_plan.steps[:index]
        ),
        current_step_summary={
            "id": interrupted.step_id,
            "skill": interrupted.skill.value,
            "status": "INTERRUPTED_BY_OBSTACLE",
        },
        remaining_plan_summary=tuple(
            step.to_dict() for step in active_plan.steps[index + 1 :]
        ),
        frames=(frame,),
        visual_assessment=assessment,
        grounded_obstacle_geometry=grounded_geometry,
        active_corridor_rejoin_target=active_corridor_rejoin_target,
        route_constraints=route_constraints,
        mission_elapsed_s=mission_elapsed_s,
    )


__all__ = [
    "build_obstacle_revision_request",
    "grounded_runtime_assessment",
    "hold_relative_obstacle_geometry",
    "hold_relative_point_target",
]
