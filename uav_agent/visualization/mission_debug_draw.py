"""Viewport-only mission visualization.

This module uses a lazy Isaac import so importing the module itself does not
violate the project's SimulationApp-before-Isaac ordering rule.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, pi, sin, tan
from typing import Sequence

import numpy as np

from skills.search import build_search_waypoints


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
    "NONE": (0.70, 0.70, 0.70, 1.00),
}

_PLAN_COLOR: Color = (0.65, 0.65, 0.65, 0.65)
_SEARCH_COLOR: Color = (1.00, 0.75, 0.05, 0.95)
_SEARCH_GROUND_COLOR: Color = (1.00, 0.75, 0.05, 0.35)
_HOME_COLOR: Color = (0.75, 0.20, 1.00, 0.95)
_ACTIVE_GOAL_COLOR: Color = (1.00, 1.00, 1.00, 0.90)


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


class MissionDebugDraw:
    def __init__(
        self,
        *,
        world_context: object,
        camera_config: object,
        options: DebugDrawOptions | None = None,
    ) -> None:
        # This import must occur only after SimulationApp exists.
        from isaacsim.util.debug_draw import _debug_draw

        self._debug_draw_module = _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._world_context = world_context
        self._camera_config = camera_config
        self._options = options or DebugDrawOptions()

        self._static_lines: list[Line] = []
        self._static_points: list[Point] = []
        self._trajectory: deque[Line] = deque(
            maxlen=self._options.max_trajectory_segments
        )
        self._frustum_lines: list[Line] = []
        self._active_goal_line: Line | None = None
        self._current_uav_point: Point | None = None
        self._step_goals: dict[str, Point3] = {}

        self._last_uav_position: np.ndarray | None = None
        self._last_skill = "NONE"
        self._plan_version: int | None = None

    @property
    def plan_version(self) -> int | None:
        return self._plan_version

    def set_plan(self, compiled_mission: object) -> None:
        """Replace static plan geometry while preserving executed trajectory."""

        self._static_lines.clear()
        self._static_points.clear()
        self._step_goals.clear()

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
                center_raw = np.asarray(step["center"], dtype=np.float64)
                radius = float(step["radius"])
                altitude = float(step["search_altitude"])

                flight_center = (
                    float(center_raw[0]),
                    float(center_raw[1]),
                    altitude,
                )
                ground_center = (
                    float(center_raw[0]),
                    float(center_raw[1]),
                    float(center_raw[2]) + 0.05,
                )

                self._add_circle(
                    flight_center,
                    radius,
                    _SEARCH_COLOR,
                    self._options.static_line_width,
                )
                self._add_circle(
                    ground_center,
                    radius,
                    _SEARCH_GROUND_COLOR,
                    1.5,
                )
                self._static_lines.append(
                    (
                        ground_center,
                        flight_center,
                        _SEARCH_GROUND_COLOR,
                        1.5,
                    )
                )
                self._static_points.append(
                    (flight_center, _SEARCH_COLOR, 12.0)
                )

                for waypoint in build_search_waypoints(
                    tuple(float(value) for value in center_raw),
                    radius,
                    altitude,
                ):
                    self._static_points.append(
                        (waypoint, _SEARCH_COLOR, 8.0)
                    )

                # SEARCH/TRACK end positions are not known at compile time.
                current_known = False

            elif skill in {"INSPECT", "TRACK", "REACQUIRE"}:
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

    def update(
        self,
        *,
        uav_pose: object,
        camera_position_m: Sequence[float] | None,
        camera_orientation_wxyz: Sequence[float] | None,
        active_skill: object,
        active_step_id: str | None,
        target_lifecycle: object,
    ) -> None:
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
                _frustum_color(target_lifecycle),
            )

        self._redraw()

    def close(self) -> None:
        if self._draw is None:
            return
        try:
            self._draw.clear_lines()
            self._draw.clear_points()
        finally:
            self._debug_draw_module.release_debug_draw_interface(
                self._draw
            )
            self._draw = None

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
        points = [
            (
                center[0] + radius * cos(2.0 * pi * index / self._options.circle_segments),
                center[1] + radius * sin(2.0 * pi * index / self._options.circle_segments),
                center[2],
            )
            for index in range(self._options.circle_segments)
        ]
        for index, start in enumerate(points):
            self._static_lines.append(
                (
                    start,
                    points[(index + 1) % len(points)],
                    color,
                    width,
                )
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

        points = list(self._static_points)
        if self._current_uav_point is not None:
            points.append(self._current_uav_point)

        if points:
            self._draw.draw_points(
                [point[0] for point in points],
                [point[1] for point in points],
                [point[2] for point in points],
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


def _frustum_color(target_lifecycle: object) -> Color:
    state = _enum_text(target_lifecycle)
    if state in {"LOCKED", "TRACKING"}:
        return 0.10, 1.00, 0.20, 0.95
    if state == "CANDIDATE":
        return 1.00, 0.55, 0.05, 0.95
    if state in {"LOST", "REACQUIRING"}:
        return 1.00, 0.10, 0.10, 0.95
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


__all__ = ["DebugDrawOptions", "MissionDebugDraw"]