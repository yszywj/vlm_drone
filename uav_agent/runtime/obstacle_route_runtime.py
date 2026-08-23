"""Pure-Python bridge from an established obstacle HOLD to route replanning.

The collision runtime owns the immediate stop.  This bridge starts a model
request only after that stop is established and camera-visible obstacle
geometry is grounded.  It preserves the interrupted :class:`TaskPlan`, binds
``UAV_HOLD_FLU`` to the actual hold pose, and delegates proposal/critique and
atomic publication to :class:`ObstacleRevisionCoordinator`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from common.ids import generate_routing_id
from common.obstacle_types import FlightCorridor
from env.obstacle_registry import ObstacleRegistry
from perception.qwen_vlm_verifier import VisualReviewFrame
from planner.obstacle_replacement_compiler import ObstacleReplacementCompiler
from planner.obstacle_revision import ObstacleRouteRevisionDraft
from planner.route_critic import RouteValidationContext
from planner.route_types import RouteConstraints
from planner.spatial_resolver import FramePose, SpatialResolver
from runtime.frame_store import FrameRef
from runtime.obstacle_revision_context import (
    build_obstacle_revision_request,
    hold_relative_obstacle_geometry,
    hold_relative_point_target,
)
from skills.plan import TaskPlan


class _Coordinator(Protocol):
    records: tuple[object, ...]

    def begin(self, request: object, **kwargs: object) -> object: ...

    def tick(self, *, timestamp_s: float) -> object: ...

    def snapshot(self) -> object: ...

    def reset(self, **kwargs: object) -> None: ...


class _PausedManager(Protocol):
    is_supervisory_paused: bool
    active_planned_step_id: str | None
    task_plan: TaskPlan | None


@dataclass(frozen=True, slots=True)
class ObstacleRouteRuntimeSnapshot:
    coordinator_state: str
    revision_started: bool
    request_id: str | None
    accepted_route_id: str | None
    error_code: str | None


class ObstacleRouteReplanRuntime:
    """Start and advance one non-blocking Qwen detour at a time."""

    _TERMINAL = frozenset({"ACCEPTED", "EXHAUSTED", "FAILED"})

    def __init__(
        self,
        *,
        coordinator: _Coordinator,
        initial_resolver: SpatialResolver,
        obstacles: ObstacleRegistry,
        scene_min_xyz_m: tuple[float, float, float],
        scene_max_xyz_m: tuple[float, float, float],
        original_instruction: str,
        original_plan_summary: Mapping[str, object],
        route_constraints: RouteConstraints = RouteConstraints(),
        compiler: ObstacleReplacementCompiler | None = None,
    ) -> None:
        for method in ("begin", "tick", "snapshot", "reset"):
            if not callable(getattr(coordinator, method, None)):
                raise TypeError(f"coordinator must provide {method}")
        if not isinstance(initial_resolver, SpatialResolver):
            raise TypeError("initial_resolver must be a SpatialResolver")
        if not isinstance(obstacles, ObstacleRegistry):
            raise TypeError("obstacles must be an ObstacleRegistry")
        if not isinstance(original_instruction, str) or not original_instruction.strip():
            raise ValueError("original_instruction must be non-empty")
        if not isinstance(original_plan_summary, Mapping):
            raise TypeError("original_plan_summary must be a mapping")
        if not isinstance(route_constraints, RouteConstraints):
            raise TypeError("route_constraints must be RouteConstraints")
        self._coordinator = coordinator
        self._initial_resolver = initial_resolver
        self._obstacles = obstacles
        self._scene_min = _xyz(scene_min_xyz_m, "scene_min_xyz_m")
        self._scene_max = _xyz(scene_max_xyz_m, "scene_max_xyz_m")
        if any(low >= high for low, high in zip(self._scene_min, self._scene_max)):
            raise ValueError("scene bounds must be strictly ordered")
        self._instruction = original_instruction.strip()
        self._original_plan_summary = dict(original_plan_summary)
        self._constraints = route_constraints
        self._compiler = compiler or ObstacleReplacementCompiler()
        self._last_clear_corridor: FlightCorridor | None = None
        self._active_hold_pose: FramePose | None = None

    @property
    def coordinator(self) -> _Coordinator:
        return self._coordinator

    @property
    def records(self) -> tuple[object, ...]:
        return tuple(self._coordinator.records)

    @property
    def active_hold_pose(self) -> FramePose | None:
        return self._active_hold_pose

    def observe_active_corridor(
        self,
        corridor: FlightCorridor | None,
        *,
        collision_state: object,
    ) -> None:
        """Remember the trusted pre-HOLD goal while supervision is clear."""

        if corridor is not None and not isinstance(corridor, FlightCorridor):
            raise TypeError("corridor must be a FlightCorridor or None")
        if corridor is not None and _state_value(collision_state) == "CLEAR":
            self._last_clear_corridor = corridor

    def tick(
        self,
        *,
        obstacle_snapshot: object,
        manager: _PausedManager,
        rgb: np.ndarray,
        frame_id: str,
        timestamp_s: float,
        mission_elapsed_s: float,
        hold_pose: FramePose,
    ) -> ObstacleRouteRuntimeSnapshot:
        """Poll the worker or start a request after grounded HOLD geometry."""

        current = self._coordinator.snapshot()
        state = _state_value(getattr(current, "state", current))
        if state == "AWAITING_MODEL":
            current = self._coordinator.tick(timestamp_s=float(timestamp_s))
            state = _state_value(getattr(current, "state", current))

        # Publication releases HOVER asynchronously on the next Manager tick.
        # Once that hand-off has completed, retain proposal history and make
        # the coordinator available for a later independent obstacle.
        if state == "ACCEPTED" and not bool(manager.is_supervisory_paused):
            self._coordinator.reset(preserve_records=True)
            current = self._coordinator.snapshot()
            state = _state_value(getattr(current, "state", current))

        started = False
        obstacle_state = _state_value(getattr(obstacle_snapshot, "state", None))
        if state == "IDLE" and obstacle_state == "GEOMETRY_GROUNDED":
            current = self._begin(
                obstacle_snapshot=obstacle_snapshot,
                manager=manager,
                rgb=rgb,
                frame_id=frame_id,
                timestamp_s=float(timestamp_s),
                mission_elapsed_s=float(mission_elapsed_s),
                hold_pose=hold_pose,
            )
            state = _state_value(getattr(current, "state", current))
            started = True

        return ObstacleRouteRuntimeSnapshot(
            coordinator_state=state,
            revision_started=started,
            request_id=getattr(current, "request_id", None),
            accepted_route_id=getattr(current, "accepted_route_id", None),
            error_code=getattr(current, "error_code", None),
        )

    def _begin(
        self,
        *,
        obstacle_snapshot: object,
        manager: _PausedManager,
        rgb: np.ndarray,
        frame_id: str,
        timestamp_s: float,
        mission_elapsed_s: float,
        hold_pose: FramePose,
    ) -> object:
        if not bool(manager.is_supervisory_paused):
            raise RuntimeError("grounded route revision requires supervisory HOVER")
        if not isinstance(hold_pose, FramePose):
            raise TypeError("hold_pose must be a FramePose")
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
        ):
            raise ValueError("rgb must be a uint8 HxWx3 image")
        plan = manager.task_plan
        step_id = manager.active_planned_step_id
        if not isinstance(plan, TaskPlan) or step_id is None:
            raise RuntimeError("interrupted routed TaskPlan is unavailable")
        try:
            interrupted_index = next(
                index for index, step in enumerate(plan.steps) if step.step_id == step_id
            )
        except StopIteration:
            raise RuntimeError("active step is absent from interrupted TaskPlan") from None
        fusion = getattr(obstacle_snapshot, "fusion", None)
        obstacle_ids = tuple(getattr(fusion, "obstacle_ids", ()))
        if not obstacle_ids:
            raise RuntimeError("grounded hazard has no obstacle ID")
        obstacle_id = obstacle_ids[0]
        world_aabb = self._obstacles.get_aabb(obstacle_id)
        corridor = self._last_clear_corridor
        if corridor is None:
            raise RuntimeError("pre-HOLD active corridor is unavailable")

        height, width = rgb.shape[:2]
        frame = VisualReviewFrame(
            FrameRef(
                uav_id=plan.uav_id,
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                width=width,
                height=height,
            ),
            rgb,
        )
        geometry = hold_relative_obstacle_geometry(
            obstacle_id=obstacle_id,
            world_aabb=world_aabb,
            hold_pose=hold_pose,
        )
        route_id = generate_routing_id("route")
        request = build_obstacle_revision_request(
            original_instruction=self._instruction,
            original_plan_summary=self._original_plan_summary,
            active_plan=plan,
            replace_from_step_id=step_id,
            route_id=route_id,
            frame=frame,
            grounded_geometry=geometry,
            active_corridor_rejoin_target=hold_relative_point_target(
                world_xyz_m=corridor.end_world_m,
                hold_pose=hold_pose,
            ),
            mission_elapsed_s=mission_elapsed_s,
            route_constraints=self._constraints,
        )
        resolver = self._initial_resolver.with_uav_hold_pose(hold_pose)
        validation = RouteValidationContext(
            resolver=resolver,
            obstacles=self._obstacles,
            scene_min_xyz_m=self._scene_min,
            scene_max_xyz_m=self._scene_max,
            route_start_world_m=hold_pose.xyz_m,
            original_goal_world_m=corridor.end_world_m,
            constraints=self._constraints,
        )

        def compile_replacement(draft: ObstacleRouteRevisionDraft) -> TaskPlan:
            return self._compiler.compile(
                draft,
                plan,
                interrupted_index,
                spatial_resolver=resolver,
            )

        result = self._coordinator.begin(
            request,
            validation_context=validation,
            frame_snapshot=hold_pose,
            compile_replacement=compile_replacement,
            timestamp_s=timestamp_s,
        )
        self._active_hold_pose = hold_pose
        self._last_clear_corridor = None
        return result


def _state_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return "NONE" if raw is None else str(raw)


def _xyz(value: object, name: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError(f"{name} must contain three finite numbers") from None
    if len(result) != 3 or any(not np.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain three finite numbers")
    return result  # type: ignore[return-value]


__all__ = ["ObstacleRouteReplanRuntime", "ObstacleRouteRuntimeSnapshot"]
