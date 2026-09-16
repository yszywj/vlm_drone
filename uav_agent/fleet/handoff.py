"""Trusted spatial binding when a semantic task changes aircraft.

Mission goals retain their original world meaning.  The receiving aircraft's
own home remains its own home for the return/landing safety epilogue.
"""

from dataclasses import replace

from fleet.task_spec import FleetTaskSpecV1
from planner.schemas import PlannerWorldContext
from planner.spatial import CoordinateFrame, NamedLocationTarget, PointTarget, RegionSpec
from planner.spatial_resolver import FramePose, SpatialResolver


def task_spatial_resolver(
    context: PlannerWorldContext, home_name: str,
) -> SpatialResolver:
    """Recreate the same immutable launch references used at initial compile.

    HOLD/camera poses and grounded landmarks are deliberately absent: a task
    referring to those cannot be transferred by guessing a newer reference.
    """
    if not isinstance(context, PlannerWorldContext):
        raise TypeError("handoff requires the trusted source PlannerWorldContext")
    home = context.landing_zones[home_name]
    home_xyz = (*home.position_xy_m, home.ground_altitude_m)
    return SpatialResolver(
        home_pose=FramePose(home_xyz, 0.0),
        uav_start_pose=FramePose(context.initial_uav_xyz_m, 0.0),
        named_locations={home_name: home_xyz},
    )


def freeze_handoff_task(
    task_spec: FleetTaskSpecV1, resolver: SpatialResolver, *, compiled_mission=None,
) -> FleetTaskSpecV1:
    """Resolve mission geometry once, before choosing a replacement UAV.

    This changes the representation, never the location or extent. Goal IDs,
    target identity, temporal requirements and ordering/assignment constraints
    survive unchanged. Termination goals have no spatial payload and retain
    their receiving-aircraft semantics.
    """
    goals = []
    for goal in task_spec.goals:
        spatial = goal.spatial_constraint
        if spatial is not None:
            if isinstance(spatial, NamedLocationTarget) and compiled_mission is not None:
                # GOTO(home) keeps the source's planned flight altitude; the
                # landing-zone ground z is not an executable GOTO altitude.
                semantic = compiled_mission.planner_output.steps
                matching_ids = {step.id for step in semantic if step.skill == "GOTO"
                                and step.spatial_target == spatial}
                positions = {tuple(step.params["position"]) for step in compiled_mission.task_plan.steps
                             if step.step_id in matching_ids and "position" in step.params}
                if len(positions) != 1:
                    raise ValueError("named handoff goal lacks one trusted executable position")
                spatial = PointTarget(CoordinateFrame.WORLD_ENU, positions.pop())
            else:
                spatial = (
                    resolver.resolve_region(spatial)
                    if isinstance(spatial, RegionSpec)
                    else resolver.resolve_target(spatial)
                )
        goals.append(replace(goal, spatial_constraint=spatial))
    return replace(task_spec, goals=tuple(goals))
