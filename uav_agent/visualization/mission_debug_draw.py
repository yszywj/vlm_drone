"""Viewport-only mission visualization.

This module uses a lazy Isaac import so importing the module itself does not
violate the project's SimulationApp-before-Isaac ordering rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass
from math import cos, pi, sin, tan
from typing import Sequence

import numpy as np

from common.obstacle_types import FlightCorridor, ObstacleAABB, ObstacleSpec
from planner.spatial import (
    CircleRegion,
    CorridorRegion,
    PolygonRegion,
    RectangleRegion,
    RegionSpec,
    RelationalRegion,
    SectorRegion,
    region_spec_from_dict,
)
from planner.route_types import RouteDraft
from skills.search import build_search_waypoints
from skills.search_geometry import generate_search_waypoints
from skills.search_strategy import (
    SearchStrategyError,
    SearchStrategySpec,
    SearchStrategyType,
)


Point3 = tuple[float, float, float]
Color = tuple[float, float, float, float]
Line = tuple[Point3, Point3, Color, float]
Point = tuple[Point3, Color, float]


_SKILL_COLORS: dict[str, Color] = {
    "TAKEOFF": (0.20, 0.55, 1.00, 1.00),
    "GOTO": (0.10, 0.90, 0.95, 1.00),
    "HOVER": (1.00, 1.00, 1.00, 1.00),
    "SEARCH": (1.00, 0.75, 0.05, 1.00),
    "INSPECT": (1.00, 0.40, 0.05, 1.00),
    "TRACK": (0.10, 1.00, 0.25, 1.00),
    "REACQUIRE": (1.00, 0.15, 0.10, 1.00),
    "LAND": (0.75, 0.20, 1.00, 1.00),
    "FOLLOW_ROUTE": (0.00, 1.00, 1.00, 1.00),
    "NONE": (0.70, 0.70, 0.70, 1.00),
}

_PLAN_COLOR: Color = (0.65, 0.65, 0.65, 0.65)
_SEARCH_COLOR: Color = (1.00, 0.75, 0.05, 0.95)
_SEARCH_GROUND_COLOR: Color = (1.00, 0.75, 0.05, 0.35)
_HOME_COLOR: Color = (0.75, 0.20, 1.00, 0.95)
_ACTIVE_GOAL_COLOR: Color = (1.00, 1.00, 1.00, 0.90)
_MODEL_WAYPOINT_COLOR: Color = (1.00, 0.40, 0.05, 1.00)
_OBSTACLE_COLOR: Color = (1.00, 0.20, 0.10, 0.90)
_CORRIDOR_COLOR: Color = (0.20, 0.80, 1.00, 0.75)
_HOLD_COLOR: Color = (1.00, 0.05, 0.05, 1.00)

_ROUTE_STATE_COLORS: dict[str, Color] = {
    "PROPOSED": (1.00, 0.40, 0.05, 0.95),
    "REJECTED": (1.00, 0.05, 0.05, 0.95),
    "ACCEPTED": (0.10, 1.00, 0.25, 0.95),
    "EXECUTING": (0.00, 1.00, 1.00, 1.00),
    "COMPLETED": (0.10, 0.75, 0.75, 0.55),
    "COLLIDED": (1.00, 0.05, 0.05, 1.00),
}


@dataclass(frozen=True, slots=True)
class DebugDrawOptions:
    circle_segments: int = 72
    trajectory_min_spacing_m: float = 0.08
    max_trajectory_segments: int = 3000
    frustum_near_m: float = 0.30
    frustum_far_m: float = 15.0
    static_line_width: float = 2.0
    trajectory_line_width: float = 3.0
    frustum_line_width: float = 2.5
    corridor_segments: int = 16


@dataclass(frozen=True, slots=True)
class DebugDrawSnapshot:
    """Small GUI/status view; it never contains camera pixels."""

    plan_version: int | None
    current_skill: str
    search_region_shapes: tuple[str, ...]
    search_strategies: tuple[str, ...]
    search_waypoint_count: int
    obstacle_count: int
    route_states: tuple[str, ...]
    hold_active: bool
    trajectory_segment_count: int
    viewport_only: bool = True


class MissionStatusOverlay:
    """Small viewport GUI window for state that geometry alone cannot convey.

    ``omni.ui`` is imported lazily, so constructing ``SimulationApp`` remains
    the hard ordering boundary. Tests may inject a tiny compatible UI module.
    The overlay receives only :class:`DebugDrawSnapshot`; camera pixels and
    model prompts can never enter it.
    """

    def __init__(
        self,
        *,
        ui_module: object | None = None,
        title: str = "VLM Drone Mission Status",
        width: int = 390,
        height: int = 190,
        initial_text: str = "Mission status unavailable",
    ) -> None:
        if ui_module is None:
            import omni.ui as ui_module  # type: ignore[no-redef]
        window_factory = getattr(ui_module, "Window", None)
        label_factory = getattr(ui_module, "Label", None)
        if not callable(window_factory) or not callable(label_factory):
            raise TypeError("ui_module must provide Window and Label")
        self._window = window_factory(
            title,
            width=width,
            height=height,
        )
        frame = getattr(self._window, "frame", None)
        if frame is None or not callable(getattr(frame, "__enter__", None)):
            raise TypeError("ui Window must expose a context-managed frame")
        with frame:
            self._label = label_factory(
                initial_text,
                word_wrap=True,
            )
        self._text = initial_text

    @property
    def text(self) -> str:
        return self._text

    def update(self, snapshot: DebugDrawSnapshot) -> None:
        if not isinstance(snapshot, DebugDrawSnapshot):
            raise TypeError("snapshot must be a DebugDrawSnapshot")
        regions = ", ".join(snapshot.search_region_shapes) or "none"
        strategies = ", ".join(snapshot.search_strategies) or "none"
        routes = ", ".join(snapshot.route_states) or "none"
        version = "none" if snapshot.plan_version is None else str(snapshot.plan_version)
        self._set_text(
            f"Plan version: {version}\n"
            f"Current Skill: {snapshot.current_skill}\n"
            f"Search: {regions} / {strategies} "
            f"({snapshot.search_waypoint_count} viewpoints)\n"
            f"Obstacles: {snapshot.obstacle_count} | Routes: {routes}\n"
            f"HOLD: {'ACTIVE' if snapshot.hold_active else 'clear'} | "
            f"Trajectory segments: {snapshot.trajectory_segment_count}"
        )

    def _set_text(self, text: str) -> None:
        """Update the shared viewport text shell without touching Camera data."""

        if not isinstance(text, str):
            raise TypeError("overlay text must be a string")
        self._text = text
        setattr(self._label, "text", text)

    def close(self) -> None:
        window = self._window
        if window is None:
            return
        if hasattr(window, "visible"):
            setattr(window, "visible", False)
        destroy = getattr(window, "destroy", None)
        if callable(destroy):
            destroy()
        self._window = None


class MissionDebugDraw:
    def __init__(
        self,
        *,
        world_context: object,
        camera_config: object,
        options: DebugDrawOptions | None = None,
        draw_interface: object | None = None,
        debug_draw_module: object | None = None,
        status_overlay: object | None = None,
    ) -> None:
        if draw_interface is None:
            # This import must occur only after SimulationApp exists.
            from isaacsim.util.debug_draw import _debug_draw

            self._debug_draw_module = _debug_draw
            self._draw = _debug_draw.acquire_debug_draw_interface()
            self._owns_draw_interface = True
        else:
            # Pure-Python tests and alternate viewport adapters may inject the
            # four-method debug-draw protocol.  It is never released by us.
            self._debug_draw_module = debug_draw_module
            self._draw = draw_interface
            self._owns_draw_interface = False
        self._world_context = world_context
        self._camera_config = camera_config
        self._options = options or DebugDrawOptions()
        if status_overlay is not None and (
            not callable(getattr(status_overlay, "update", None))
            or not callable(getattr(status_overlay, "close", None))
        ):
            raise TypeError("status_overlay must provide update and close")
        self._status_overlay = status_overlay

        self._static_lines: list[Line] = []
        self._static_points: list[Point] = []
        self._trajectory: deque[Line] = deque(
            maxlen=self._options.max_trajectory_segments
        )
        self._frustum_lines: list[Line] = []
        self._search_lines: list[Line] = []
        self._search_points: list[Point] = []
        self._route_lines: list[Line] = []
        self._route_points: list[Point] = []
        self._obstacle_lines: list[Line] = []
        self._obstacle_points: list[Point] = []
        self._corridor_lines: list[Line] = []
        self._hold_lines: list[Line] = []
        self._hold_point: Point | None = None
        self._active_goal_line: Line | None = None
        self._current_uav_point: Point | None = None
        self._step_goals: dict[str, Point3] = {}
        self._follow_route_step_refs: dict[str, str] = {}

        self._last_uav_position: np.ndarray | None = None
        self._last_skill = "NONE"
        self._plan_version: int | None = None
        self._search_region_shapes: list[str] = []
        self._search_strategies: list[str] = []
        self._search_waypoint_count = 0
        self._obstacle_count = 0
        self._route_states: list[str] = []

    @property
    def plan_version(self) -> int | None:
        return self._plan_version

    @property
    def viewport_only(self) -> bool:
        """Debug primitives are submitted to the viewport, never RGB sensors."""

        return True

    def snapshot(self) -> DebugDrawSnapshot:
        """Return bounded status fields suitable for a GUI text overlay."""

        return DebugDrawSnapshot(
            plan_version=self._plan_version,
            current_skill=self._last_skill,
            search_region_shapes=tuple(self._search_region_shapes),
            search_strategies=tuple(self._search_strategies),
            search_waypoint_count=self._search_waypoint_count,
            obstacle_count=self._obstacle_count,
            route_states=tuple(self._route_states),
            hold_active=self._hold_point is not None,
            trajectory_segment_count=len(self._trajectory),
        )

    def set_search_region(
        self,
        region: RegionSpec | Mapping[str, object],
        *,
        strategy: SearchStrategySpec | Mapping[str, object] | None = None,
        altitude_m: float | None = None,
        waypoints_xyz_m: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Replace standalone SEARCH overlays with one resolved V3 region."""

        self._search_lines.clear()
        self._search_points.clear()
        self._search_region_shapes.clear()
        self._search_strategies.clear()
        self._search_waypoint_count = 0
        self._append_search_region(
            region,
            strategy=strategy,
            altitude_m=altitude_m,
            waypoints_xyz_m=waypoints_xyz_m,
        )
        self._redraw()

    def set_obstacles(
        self,
        obstacles: Sequence[ObstacleSpec | ObstacleAABB | object],
    ) -> None:
        """Replace world-space obstacle AABB overlays."""

        self._obstacle_lines.clear()
        self._obstacle_points.clear()
        normalized: list[ObstacleAABB] = []
        for obstacle in tuple(obstacles):
            if isinstance(obstacle, ObstacleAABB):
                box = obstacle
            elif isinstance(obstacle, ObstacleSpec):
                box = obstacle.aabb
            else:
                box = getattr(obstacle, "aabb", None)
                if not isinstance(box, ObstacleAABB):
                    raise TypeError(
                        "obstacles must contain ObstacleSpec or ObstacleAABB values"
                    )
            normalized.append(box)
            self._obstacle_lines.extend(
                _aabb_lines(box, _OBSTACLE_COLOR, self._options.static_line_width)
            )
            self._obstacle_points.append(
                (box.center_xyz_m, _OBSTACLE_COLOR, 8.0)
            )
        self._obstacle_count = len(normalized)
        self._redraw()

    def set_safety_corridor(self, corridor: FlightCorridor | None) -> None:
        """Replace the active swept flight corridor, or clear it with ``None``."""

        if corridor is not None and not isinstance(corridor, FlightCorridor):
            raise TypeError("corridor must be a FlightCorridor or None")
        self._corridor_lines.clear()
        if corridor is not None:
            self._corridor_lines.extend(
                _tube_lines(
                    corridor.start_world_m,
                    corridor.end_world_m,
                    corridor.radius_m,
                    _CORRIDOR_COLOR,
                    self._options.static_line_width,
                    self._options.corridor_segments,
                )
            )
        self._redraw()

    def set_hold_point(self, position_world_m: Sequence[float] | None) -> None:
        """Set or clear the supervisory HOLD marker."""

        self._hold_lines.clear()
        self._hold_point = None
        if position_world_m is not None:
            center = _point(position_world_m)
            radius = 0.40
            for axis in range(3):
                start = list(center)
                end = list(center)
                start[axis] -= radius
                end[axis] += radius
                self._hold_lines.append(
                    (_point(start), _point(end), _HOLD_COLOR, 4.0)
                )
            self._hold_point = (center, _HOLD_COLOR, 15.0)
        self._redraw()

    def set_route_records(self, records: Sequence[object]) -> None:
        """Draw raw coordinator proposals and registry lifecycle records.

        Coordinator records intentionally retain proposals that a Critic
        rejected and therefore never entered RouteRegistry.  Their raw path is
        drawn orange first, followed by a red rejection overlay.  The accepted
        registry record is drawn separately in green/cyan/completed colors, so
        no model proposal is silently lost from the GUI.
        """

        self._route_lines.clear()
        self._route_points.clear()
        self._route_states.clear()
        route_goals: dict[str, Point3] = {}
        for record in tuple(records):
            route = getattr(record, "route", None)
            coordinator_proposal = False
            if route is None:
                proposal = getattr(record, "proposal", None)
                route_payload = (
                    proposal.get("route_draft")
                    if isinstance(proposal, Mapping)
                    else None
                )
                if route_payload is None:
                    # Model/schema failures contain no drawable geometry.
                    continue
                if not isinstance(route_payload, Mapping):
                    raise TypeError("proposal route_draft must be a mapping")
                route = RouteDraft.from_dict(route_payload)
                coordinator_proposal = True
            state = _enum_text(getattr(record, "state", "PROPOSED"))
            route_id = str(getattr(record, "route_id", getattr(route, "route_id", "")))
            if route is None or not route_id:
                raise TypeError("records must contain route registry records")
            waypoints = _route_record_world_waypoints(record, route=route)
            if len(waypoints) < 2:
                raise ValueError("route records must contain at least two waypoints")
            if coordinator_proposal:
                self._append_route_geometry(
                    waypoints,
                    route_state_color("PROPOSED"),
                )
                self._route_states.append("PROPOSED")
                outcome = _enum_text(getattr(record, "outcome", ""))
                if outcome == "REVISE":
                    self._append_route_geometry(
                        waypoints,
                        route_state_color("REJECTED"),
                    )
                    self._route_states.append("REJECTED")
                continue
            color = route_state_color(state)
            self._append_route_geometry(waypoints, color)
            self._route_states.append(state)
            route_goals[route_id] = waypoints[-1]
        for step_id, route_ref in self._follow_route_step_refs.items():
            goal = route_goals.get(route_ref)
            if goal is not None:
                self._step_goals[step_id] = goal
        self._redraw()

    def set_plan(self, compiled_mission: object) -> None:
        """Replace static plan geometry while preserving executed trajectory."""

        self._static_lines.clear()
        self._static_points.clear()
        self._search_lines.clear()
        self._search_points.clear()
        self._step_goals.clear()
        self._follow_route_step_refs.clear()
        self._search_region_shapes.clear()
        self._search_strategies.clear()
        self._search_waypoint_count = 0

        task_plan = compiled_mission.task_plan
        self._plan_version = int(task_plan.plan_version)
        steps = task_plan.to_dicts()

        # Landing-zone footprint.
        for zone in self._world_context.landing_zones.values():
            center = (
                float(zone.position_xy_m[0]),
                float(zone.position_xy_m[1]),
                float(zone.ground_altitude_m) + 0.05,
            )
            self._add_circle(
                center,
                float(zone.horizontal_tolerance_m),
                _HOME_COLOR,
                self._options.static_line_width,
            )
            self._static_points.append((center, _HOME_COLOR, 11.0))

        current = np.asarray(
            self._world_context.initial_uav_xyz_m,
            dtype=np.float64,
        )
        current_known = True

        for step in steps:
            step_id = str(step["id"])
            skill = _enum_text(step["skill"])

            if skill == "TAKEOFF":
                target = np.asarray(
                    [
                        current[0],
                        current[1],
                        float(step["target_altitude"]),
                    ],
                    dtype=np.float64,
                )
                self._add_static_line(current, target, _SKILL_COLORS["TAKEOFF"])
                self._step_goals[step_id] = _point(target)
                current = target
                current_known = True

            elif skill == "GOTO":
                target = np.asarray(step["position"], dtype=np.float64)
                if current_known:
                    self._add_static_line(current, target, _PLAN_COLOR)
                self._static_points.append(
                    (_point(target), _SKILL_COLORS["GOTO"], 9.0)
                )
                self._step_goals[step_id] = _point(target)
                current = target
                current_known = True

            elif skill == "SEARCH":
                if "region" in step:
                    altitude = float(
                        step.get(
                            "search_altitude_m",
                            step.get("search_altitude", step.get("altitude_m", 0.0)),
                        )
                    )
                    self._append_search_region(
                        step["region"],
                        strategy=step.get("strategy"),
                        altitude_m=altitude,
                    )
                else:
                    center_raw = np.asarray(step["center"], dtype=np.float64)
                    radius = float(step["radius"])
                    altitude = float(step["search_altitude"])
                    self._append_legacy_search(center_raw, radius, altitude)

                # SEARCH/TRACK end positions are not known at compile time.
                current_known = False

            elif skill in {"INSPECT", "TRACK", "REACQUIRE"}:
                current_known = False

            elif skill == "FOLLOW_ROUTE":
                route_ref = step.get("route_ref")
                if isinstance(route_ref, str) and route_ref:
                    self._follow_route_step_refs[step_id] = route_ref
                raw_waypoints = step.get("waypoints")
                if isinstance(raw_waypoints, Sequence) and not isinstance(
                    raw_waypoints, (str, bytes)
                ):
                    waypoints = tuple(_point(value) for value in raw_waypoints)
                    if len(waypoints) >= 2:
                        self._append_route_geometry(
                            waypoints,
                            route_state_color("ACCEPTED"),
                        )
                        self._step_goals[step_id] = waypoints[-1]
                current_known = False

            elif skill == "LAND":
                target = np.asarray(
                    [
                        float(step["expected_position_xy"][0]),
                        float(step["expected_position_xy"][1]),
                        float(step["ground_altitude"]),
                    ],
                    dtype=np.float64,
                )
                if current_known:
                    self._add_static_line(
                        current,
                        target,
                        _SKILL_COLORS["LAND"],
                    )
                self._step_goals[step_id] = _point(target)
                current = target
                current_known = True

        self._redraw()

    def _append_legacy_search(
        self,
        center_raw: Sequence[float],
        radius: float,
        altitude: float,
    ) -> None:
        center = _point(center_raw)
        flight_center = (center[0], center[1], float(altitude))
        ground_center = (center[0], center[1], center[2] + 0.05)
        _append_circle(
            self._search_lines,
            flight_center,
            radius,
            _SEARCH_COLOR,
            self._options.static_line_width,
            self._options.circle_segments,
        )
        _append_circle(
            self._search_lines,
            ground_center,
            radius,
            _SEARCH_GROUND_COLOR,
            1.5,
            self._options.circle_segments,
        )
        self._search_lines.append(
            (ground_center, flight_center, _SEARCH_GROUND_COLOR, 1.5)
        )
        self._search_points.append((flight_center, _SEARCH_COLOR, 12.0))
        waypoints = build_search_waypoints(center, radius, altitude)
        self._search_points.extend(
            (waypoint, _SEARCH_COLOR, 8.0) for waypoint in waypoints
        )
        self._search_region_shapes.append("CIRCLE")
        self._search_strategies.append("PERIMETER_V1")
        self._search_waypoint_count += len(waypoints)

    def _append_search_region(
        self,
        region: RegionSpec | Mapping[str, object] | object,
        *,
        strategy: SearchStrategySpec | Mapping[str, object] | object | None,
        altitude_m: float | None,
        waypoints_xyz_m: Sequence[Sequence[float]] | None = None,
    ) -> None:
        normalized_region = _normalize_region(region)
        if isinstance(normalized_region, RelationalRegion):
            raise ValueError("RELATIONAL search regions must be resolved before drawing")
        normalized_strategy = _normalize_strategy(strategy)
        altitude = (
            _region_reference_altitude(normalized_region)
            if altitude_m is None
            else float(altitude_m)
        )
        if not np.isfinite(altitude):
            raise ValueError("altitude_m must be finite")

        flight_segments = _region_lines(
            normalized_region,
            altitude,
            _SEARCH_COLOR,
            self._options.static_line_width,
            self._options.circle_segments,
        )
        ground_altitude = _region_reference_altitude(normalized_region) + 0.05
        ground_segments = _region_lines(
            normalized_region,
            ground_altitude,
            _SEARCH_GROUND_COLOR,
            1.5,
            self._options.circle_segments,
        )
        self._search_lines.extend(flight_segments)
        self._search_lines.extend(ground_segments)
        center = _region_center_for_draw(normalized_region, altitude)
        ground_center = _region_center_for_draw(normalized_region, ground_altitude)
        self._search_lines.append(
            (ground_center, center, _SEARCH_GROUND_COLOR, 1.5)
        )
        self._search_points.append((center, _SEARCH_COLOR, 12.0))

        if waypoints_xyz_m is not None:
            waypoints = tuple(_point(value) for value in waypoints_xyz_m)
        elif normalized_strategy is not None:
            try:
                waypoints = generate_search_waypoints(
                    normalized_region,
                    normalized_strategy,
                    altitude_m=altitude,
                )
            except SearchStrategyError:
                # Runtime-only adaptive strategies have no complete path yet.
                waypoints = ()
        else:
            waypoints = ()
        waypoint_color = (
            _MODEL_WAYPOINT_COLOR
            if normalized_strategy is not None
            and normalized_strategy.kind is SearchStrategyType.MODEL_WAYPOINTS
            else _SEARCH_COLOR
        )
        self._search_points.extend(
            (waypoint, waypoint_color, 9.0) for waypoint in waypoints
        )
        self._search_region_shapes.append(normalized_region.shape)
        self._search_strategies.append(
            "NONE"
            if normalized_strategy is None
            else normalized_strategy.kind.value
        )
        self._search_waypoint_count += len(waypoints)

    def _append_route_geometry(
        self,
        waypoints: Sequence[Point3],
        color: Color,
    ) -> None:
        for start, end in zip(waypoints, waypoints[1:]):
            self._route_lines.append(
                (start, end, color, self._options.static_line_width + 1.0)
            )
        self._route_points.extend((waypoint, color, 10.0) for waypoint in waypoints)

    def update(
        self,
        *,
        uav_pose: object,
        camera_position_m: Sequence[float] | None,
        camera_orientation_wxyz: Sequence[float] | None,
        active_skill: object,
        active_step_id: str | None,
        target_lifecycle: object,
        hazard_active: bool = False,
        hold_active: bool = False,
    ) -> None:
        if not isinstance(hazard_active, bool) or not isinstance(hold_active, bool):
            raise TypeError("hazard_active and hold_active must be booleans")
        current = np.asarray(
            [uav_pose.x, uav_pose.y, uav_pose.z],
            dtype=np.float64,
        )
        skill = _enum_text(active_skill)

        if self._last_uav_position is None:
            self._last_uav_position = current.copy()
        else:
            distance = float(
                np.linalg.norm(current - self._last_uav_position)
            )
            if distance >= self._options.trajectory_min_spacing_m:
                color = _SKILL_COLORS.get(
                    skill,
                    _SKILL_COLORS["NONE"],
                )
                self._trajectory.append(
                    (
                        _point(self._last_uav_position),
                        _point(current),
                        color,
                        self._options.trajectory_line_width,
                    )
                )
                self._last_uav_position = current.copy()

        self._last_skill = skill
        self._current_uav_point = (
            _point(current),
            _SKILL_COLORS.get(skill, _SKILL_COLORS["NONE"]),
            11.0,
        )

        goal = (
            None
            if active_step_id is None
            else self._step_goals.get(active_step_id)
        )
        self._active_goal_line = (
            None
            if goal is None
            else (
                _point(current),
                goal,
                _ACTIVE_GOAL_COLOR,
                1.5,
            )
        )

        self._frustum_lines = []
        if (
            camera_position_m is not None
            and camera_orientation_wxyz is not None
        ):
            self._frustum_lines = self._build_frustum(
                camera_position_m,
                camera_orientation_wxyz,
                _frustum_color(
                    target_lifecycle,
                    active_skill=skill,
                    hazard_active=hazard_active,
                    hold_active=hold_active,
                ),
            )

        self._redraw()
        if self._status_overlay is not None:
            self._status_overlay.update(self.snapshot())

    def close(self) -> None:
        if self._draw is not None:
            try:
                self._draw.clear_lines()
                self._draw.clear_points()
            finally:
                if self._owns_draw_interface:
                    self._debug_draw_module.release_debug_draw_interface(
                        self._draw
                    )
                self._draw = None
        if self._status_overlay is not None:
            self._status_overlay.close()
            self._status_overlay = None

    def _add_static_line(
        self,
        start: Sequence[float],
        end: Sequence[float],
        color: Color,
    ) -> None:
        self._static_lines.append(
            (
                _point(start),
                _point(end),
                color,
                self._options.static_line_width,
            )
        )

    def _add_circle(
        self,
        center: Point3,
        radius: float,
        color: Color,
        width: float,
    ) -> None:
        _append_circle(
            self._static_lines,
            center,
            radius,
            color,
            width,
            self._options.circle_segments,
        )

    def _build_frustum(
        self,
        camera_position_m: Sequence[float],
        camera_orientation_wxyz: Sequence[float],
        color: Color,
    ) -> list[Line]:
        position = np.asarray(camera_position_m, dtype=np.float64)
        rotation = _rotation_matrix_wxyz(camera_orientation_wxyz)

        width_px, height_px = self._camera_config.resolution_wh_px
        half_horizontal = np.deg2rad(
            float(self._camera_config.horizontal_fov_deg)
        ) / 2.0
        horizontal_tangent = tan(half_horizontal)
        vertical_tangent = (
            horizontal_tangent * float(height_px) / float(width_px)
        )

        def plane(depth: float) -> list[Point3]:
            half_y = depth * horizontal_tangent
            half_z = depth * vertical_tangent

            # Project camera convention: +X forward, +Y left, +Z up.
            local_corners = (
                (depth, +half_y, +half_z),
                (depth, -half_y, +half_z),
                (depth, -half_y, -half_z),
                (depth, +half_y, -half_z),
            )
            return [
                _point(position + rotation @ np.asarray(corner))
                for corner in local_corners
            ]

        near = plane(self._options.frustum_near_m)
        far = plane(self._options.frustum_far_m)

        result: list[Line] = []
        for rectangle in (near, far):
            for index, start in enumerate(rectangle):
                result.append(
                    (
                        start,
                        rectangle[(index + 1) % 4],
                        color,
                        self._options.frustum_line_width,
                    )
                )
        for index in range(4):
            result.append(
                (
                    near[index],
                    far[index],
                    color,
                    self._options.frustum_line_width,
                )
            )
        return result

    def _redraw(self) -> None:
        if self._draw is None:
            return

        self._draw.clear_lines()
        self._draw.clear_points()

        lines = [
            *self._static_lines,
            *self._search_lines,
            *self._route_lines,
            *self._obstacle_lines,
            *self._corridor_lines,
            *self._hold_lines,
            *tuple(self._trajectory),
            *self._frustum_lines,
        ]
        if self._active_goal_line is not None:
            lines.append(self._active_goal_line)

        if lines:
            self._draw.draw_lines(
                [line[0] for line in lines],
                [line[1] for line in lines],
                [line[2] for line in lines],
                [line[3] for line in lines],
            )

        points = [
            *self._static_points,
            *self._search_points,
            *self._route_points,
            *self._obstacle_points,
        ]
        if self._hold_point is not None:
            points.append(self._hold_point)
        if self._current_uav_point is not None:
            points.append(self._current_uav_point)

        if points:
            self._draw.draw_points(
                [point[0] for point in points],
                [point[1] for point in points],
                [point[2] for point in points],
            )


def route_state_color(state: object) -> Color:
    """Return the stable viewport color for a route lifecycle state."""

    normalized = _enum_text(state).upper()
    try:
        return _ROUTE_STATE_COLORS[normalized]
    except KeyError:
        raise ValueError(f"unsupported route state: {normalized}") from None


def _normalize_region(region: object) -> RegionSpec:
    types = (
        CircleRegion,
        RectangleRegion,
        SectorRegion,
        PolygonRegion,
        CorridorRegion,
        RelationalRegion,
    )
    if isinstance(region, types):
        return region
    if not isinstance(region, Mapping):
        raise TypeError("region must be a RegionSpec or mapping")
    data = dict(region)
    if "shape" not in data:
        keys = set(data)
        if {"center_xyz_m", "radius_m"} <= keys:
            data["shape"] = "CIRCLE"
        elif {"center_xyz_m", "width_m", "height_m"} <= keys:
            data["shape"] = "RECTANGLE"
        elif {"origin_xyz_m", "azimuth_center_deg", "azimuth_span_deg"} <= keys:
            data["shape"] = "SECTOR"
        elif "vertices_xyz_m" in keys:
            data["shape"] = "POLYGON"
        elif "centerline_xyz_m" in keys:
            data["shape"] = "CORRIDOR"
        elif {"relation", "reference_id"} <= keys:
            data["shape"] = "RELATIONAL"
        else:
            raise ValueError("could not infer search region shape")
    return region_spec_from_dict(data)


def _normalize_strategy(strategy: object | None) -> SearchStrategySpec | None:
    if strategy is None:
        return None
    if isinstance(strategy, SearchStrategySpec):
        return strategy
    if isinstance(strategy, Mapping):
        return SearchStrategySpec.from_dict(strategy)
    raise TypeError("strategy must be a SearchStrategySpec, mapping, or None")


def _region_reference_altitude(region: RegionSpec) -> float:
    if isinstance(region, (CircleRegion, RectangleRegion)):
        return float(region.center_xyz_m[2])
    if isinstance(region, SectorRegion):
        return float(region.origin_xyz_m[2])
    if isinstance(region, PolygonRegion):
        return float(
            sum(point[2] for point in region.vertices_xyz_m)
            / len(region.vertices_xyz_m)
        )
    if isinstance(region, CorridorRegion):
        return float(
            sum(point[2] for point in region.centerline_xyz_m)
            / len(region.centerline_xyz_m)
        )
    raise ValueError("RELATIONAL search regions have no drawable altitude")


def _region_center_for_draw(region: RegionSpec, altitude: float) -> Point3:
    if isinstance(region, (CircleRegion, RectangleRegion)):
        center = region.center_xyz_m
    elif isinstance(region, SectorRegion):
        distance = sum(region.distance_range_m) / 2.0
        angle = np.deg2rad(region.azimuth_center_deg)
        center = (
            region.origin_xyz_m[0] + distance * cos(angle),
            region.origin_xyz_m[1] + distance * sin(angle),
            altitude,
        )
    elif isinstance(region, PolygonRegion):
        center = (
            sum(point[0] for point in region.vertices_xyz_m)
            / len(region.vertices_xyz_m),
            sum(point[1] for point in region.vertices_xyz_m)
            / len(region.vertices_xyz_m),
            altitude,
        )
    elif isinstance(region, CorridorRegion):
        center = region.centerline_xyz_m[len(region.centerline_xyz_m) // 2]
    else:
        raise ValueError("RELATIONAL search regions cannot be drawn")
    return float(center[0]), float(center[1]), float(altitude)


def _region_lines(
    region: RegionSpec,
    altitude: float,
    color: Color,
    width: float,
    circle_segments: int,
) -> list[Line]:
    lines: list[Line] = []
    if isinstance(region, CircleRegion):
        _append_circle(
            lines,
            (region.center_xyz_m[0], region.center_xyz_m[1], altitude),
            region.radius_m,
            color,
            width,
            circle_segments,
        )
        return lines
    if isinstance(region, RectangleRegion):
        yaw = np.deg2rad(region.yaw_deg)
        cosine, sine = cos(yaw), sin(yaw)
        half_x, half_y = region.width_m / 2.0, region.height_m / 2.0
        corners = tuple(
            (
                region.center_xyz_m[0] + local_x * cosine - local_y * sine,
                region.center_xyz_m[1] + local_x * sine + local_y * cosine,
                altitude,
            )
            for local_x, local_y in (
                (-half_x, -half_y),
                (half_x, -half_y),
                (half_x, half_y),
                (-half_x, half_y),
            )
        )
        return _polyline_lines(corners, color, width, closed=True)
    if isinstance(region, SectorRegion):
        count = max(
            4,
            int(round(circle_segments * region.azimuth_span_deg / 360.0)),
        )
        start_angle = region.azimuth_center_deg - region.azimuth_span_deg / 2.0
        angles = tuple(
            np.deg2rad(start_angle + region.azimuth_span_deg * index / count)
            for index in range(count + 1)
        )
        near, far = region.distance_range_m

        def arc(radius: float) -> tuple[Point3, ...]:
            return tuple(
                (
                    region.origin_xyz_m[0] + radius * cos(angle),
                    region.origin_xyz_m[1] + radius * sin(angle),
                    altitude,
                )
                for angle in angles
            )

        inner, outer = arc(near), arc(far)
        lines.extend(_polyline_lines(inner, color, width))
        lines.extend(_polyline_lines(outer, color, width))
        lines.append((inner[0], outer[0], color, width))
        lines.append((inner[-1], outer[-1], color, width))
        return lines
    if isinstance(region, PolygonRegion):
        points = tuple((point[0], point[1], altitude) for point in region.vertices_xyz_m)
        return _polyline_lines(points, color, width, closed=True)
    if isinstance(region, CorridorRegion):
        centerline = tuple(
            (point[0], point[1], altitude) for point in region.centerline_xyz_m
        )
        lines.extend(_polyline_lines(centerline, color, width))
        for start, end in zip(centerline, centerline[1:]):
            delta_x, delta_y = end[0] - start[0], end[1] - start[1]
            length = float(np.hypot(delta_x, delta_y))
            if length <= 1e-12:
                continue
            offset_x = -delta_y * region.half_width_m / length
            offset_y = delta_x * region.half_width_m / length
            left_start = (start[0] + offset_x, start[1] + offset_y, altitude)
            left_end = (end[0] + offset_x, end[1] + offset_y, altitude)
            right_start = (start[0] - offset_x, start[1] - offset_y, altitude)
            right_end = (end[0] - offset_x, end[1] - offset_y, altitude)
            lines.extend(
                (
                    (left_start, left_end, color, width),
                    (right_start, right_end, color, width),
                    (left_start, right_start, color, width),
                    (left_end, right_end, color, width),
                )
            )
        return lines
    raise ValueError("RELATIONAL search regions cannot be drawn")


def _append_circle(
    lines: list[Line],
    center: Point3,
    radius: float,
    color: Color,
    width: float,
    segments: int,
) -> None:
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("debug circle radius must be finite and positive")
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 4:
        raise ValueError("debug circle segments must be an integer of at least four")
    points = tuple(
        (
            center[0] + radius * cos(2.0 * pi * index / segments),
            center[1] + radius * sin(2.0 * pi * index / segments),
            center[2],
        )
        for index in range(segments)
    )
    lines.extend(
        (
            start,
            points[(index + 1) % len(points)],
            color,
            width,
        )
        for index, start in enumerate(points)
    )


def _polyline_lines(
    points: Sequence[Point3],
    color: Color,
    width: float,
    *,
    closed: bool = False,
) -> list[Line]:
    pairs = list(zip(points, points[1:]))
    if closed and len(points) >= 2:
        pairs.append((points[-1], points[0]))
    return [(start, end, color, width) for start, end in pairs]


def _aabb_lines(box: ObstacleAABB, color: Color, width: float) -> list[Line]:
    low, high = box.min_xyz_m, box.max_xyz_m
    corners: tuple[Point3, ...] = (
        (low[0], low[1], low[2]),
        (high[0], low[1], low[2]),
        (high[0], high[1], low[2]),
        (low[0], high[1], low[2]),
        (low[0], low[1], high[2]),
        (high[0], low[1], high[2]),
        (high[0], high[1], high[2]),
        (low[0], high[1], high[2]),
    )
    indexes = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return [(corners[a], corners[b], color, width) for a, b in indexes]


def _tube_lines(
    start: Sequence[float],
    end: Sequence[float],
    radius: float,
    color: Color,
    width: float,
    segments: int,
) -> list[Line]:
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 4:
        raise ValueError("corridor_segments must be an integer of at least four")
    start_array = np.asarray(_point(start), dtype=np.float64)
    end_array = np.asarray(_point(end), dtype=np.float64)
    direction = end_array - start_array
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12 or radius <= 1e-12:
        return [(_point(start_array), _point(end_array), color, width)]
    axis = direction / norm
    reference = (
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if abs(float(axis[2])) < 0.90
        else np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    )
    basis_a = np.cross(axis, reference)
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(axis, basis_a)
    start_ring = tuple(
        _point(
            start_array
            + radius
            * (cos(2.0 * pi * index / segments) * basis_a
               + sin(2.0 * pi * index / segments) * basis_b)
        )
        for index in range(segments)
    )
    end_ring = tuple(
        _point(
            end_array
            + radius
            * (cos(2.0 * pi * index / segments) * basis_a
               + sin(2.0 * pi * index / segments) * basis_b)
        )
        for index in range(segments)
    )
    lines = _polyline_lines(start_ring, color, width, closed=True)
    lines.extend(_polyline_lines(end_ring, color, width, closed=True))
    lines.extend(
        (start_ring[index], end_ring[index], color, width)
        for index in range(segments)
    )
    lines.append((_point(start_array), _point(end_array), color, width))
    return lines


def _route_record_world_waypoints(
    record: object,
    *,
    route: object | None = None,
) -> tuple[Point3, ...]:
    if route is None:
        route = getattr(record, "route", None)
    raw_waypoints = getattr(route, "waypoints", ())
    points = tuple(_point(getattr(item, "xyz_m", item)) for item in raw_waypoints)
    frame = _enum_text(getattr(route, "frame", "WORLD_ENU"))
    if frame == "WORLD_ENU":
        return points
    snapshot = getattr(record, "frame_snapshot", None)
    origin = _point(getattr(snapshot, "xyz_m", (0.0, 0.0, 0.0)))
    yaw = float(getattr(snapshot, "yaw_rad", 0.0))
    cosine, sine = cos(yaw), sin(yaw)
    return tuple(
        (
            origin[0] + cosine * point[0] - sine * point[1],
            origin[1] + sine * point[0] + cosine * point[1],
            origin[2] + point[2],
        )
        for point in points
    )


def _point(value: Sequence[float]) -> Point3:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("debug point must contain three finite values")
    return float(array[0]), float(array[1]), float(array[2])


def _enum_text(value: object) -> str:
    if value is None:
        return "NONE"
    return str(getattr(value, "value", value))


def _frustum_color(
    target_lifecycle: object,
    *,
    active_skill: object = "NONE",
    hazard_active: bool = False,
    hold_active: bool = False,
) -> Color:
    state = _enum_text(target_lifecycle)
    skill = _enum_text(active_skill)
    if hazard_active or hold_active or state in {
        "HAZARD",
        "HOLD",
        "IMMINENT_COLLISION",
        "LOST",
        "REACQUIRING",
    }:
        return 1.00, 0.10, 0.10, 0.95
    if state in {"LOCKED", "TRACKING"}:
        return 0.10, 1.00, 0.20, 0.95
    if state in {"SEARCHING", "CANDIDATE"} or skill == "SEARCH":
        return 1.00, 0.75, 0.05, 0.95
    return 0.10, 0.85, 1.00, 0.80


def _rotation_matrix_wxyz(
    quaternion_wxyz: Sequence[float],
) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("camera quaternion must contain four finite values")

    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("camera quaternion must be non-zero")

    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


__all__ = [
    "DebugDrawOptions",
    "DebugDrawSnapshot",
    "MissionDebugDraw",
    "MissionStatusOverlay",
    "route_state_color",
]
