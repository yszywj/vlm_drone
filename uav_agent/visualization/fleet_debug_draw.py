"""Viewport-only fleet visualization over trusted world-space state.

The class deliberately accepts no RGB or Camera sample.  Geometry is submitted
only to an Isaac debug-draw compatible interface, while identifiers, semantic
aliases, claim states, and plan versions remain in a small typed snapshot for a
text overlay or test harness.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import cos, dist, isfinite, pi, sin
from numbers import Integral, Real

from common.ids import validate_routing_id, validate_uav_id
from fleet.airspace_manager import (
    FleetAirspaceDecision,
    FleetPoseSnapshot,
    FleetUavPose,
)
from fleet.target_registry import SharedTargetRecord
from fleet.types import FleetMissionPlan
from visualization.mission_debug_draw import (
    Color,
    Line,
    MissionStatusOverlay,
    Point,
    _region_center_for_draw,
    _region_lines,
    _region_reference_altitude,
)


Point3 = tuple[float, float, float]

_CONFLICT_COLOR: Color = (1.0, 0.05, 0.05, 1.0)
_TARGET_COLOR: Color = (1.0, 0.15, 0.85, 1.0)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _point(value: object, name: str = "position") -> Point3:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must contain exactly three finite numbers")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three finite numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{name}[{index}] must be a finite number")
        number = float(item)
        if not isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return result[0], result[1], result[2]


def _agent_versions(
    value: Mapping[str, int | None],
    *,
    known_uavs: frozenset[str],
) -> dict[str, int | None]:
    if not isinstance(value, Mapping):
        raise TypeError("agent_plan_versions must be a mapping")
    result: dict[str, int | None] = {}
    for raw_uav_id, raw_version in value.items():
        uav_id = validate_uav_id(raw_uav_id)
        if uav_id not in known_uavs:
            raise ValueError(f"agent_plan_versions contains unknown UAV {uav_id}")
        if raw_version is not None:
            raw_version = _positive_int(raw_version, f"agent_plan_versions[{uav_id}]")
        result[uav_id] = raw_version
    return result


def _uav_colors(uav_ids: Sequence[str]) -> dict[str, Color]:
    """Return deterministic, visually distinct colors for one fleet plan."""

    ordered = tuple(sorted(uav_ids))
    count = max(1, len(ordered))
    colors: dict[str, Color] = {}
    for index, uav_id in enumerate(ordered):
        hue = index / count
        # Three phase-shifted cosines provide an HSV-like bright palette with
        # no dependency on UI or image libraries.
        colors[uav_id] = (
            0.55 + 0.40 * cos(2.0 * pi * hue),
            0.55 + 0.40 * cos(2.0 * pi * (hue - 1.0 / 3.0)),
            0.55 + 0.40 * cos(2.0 * pi * (hue - 2.0 / 3.0)),
            1.0,
        )
    return colors


def _with_alpha(color: Color, alpha: float) -> Color:
    return color[0], color[1], color[2], alpha


def _circle_lines(
    center: Point3,
    radius: float,
    color: Color,
    width: float,
    segments: int,
) -> tuple[Line, ...]:
    points = tuple(
        (
            center[0] + radius * cos(2.0 * pi * index / segments),
            center[1] + radius * sin(2.0 * pi * index / segments),
            center[2],
        )
        for index in range(segments)
    )
    return tuple(
        (points[index], points[(index + 1) % segments], color, width)
        for index in range(segments)
    )


@dataclass(frozen=True, slots=True)
class FleetDebugDrawOptions:
    circle_segments: int = 48
    trajectory_min_spacing_m: float = 0.08
    max_trajectory_segments_per_uav: int = 1000
    static_line_width: float = 2.0
    trajectory_line_width: float = 3.0
    conflict_line_width: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "circle_segments", _positive_int(self.circle_segments, "circle_segments")
        )
        if self.circle_segments < 8:
            raise ValueError("circle_segments must be at least 8")
        object.__setattr__(
            self,
            "max_trajectory_segments_per_uav",
            _positive_int(
                self.max_trajectory_segments_per_uav,
                "max_trajectory_segments_per_uav",
            ),
        )
        for name in (
            "trajectory_min_spacing_m",
            "static_line_width",
            "trajectory_line_width",
            "conflict_line_width",
        ):
            object.__setattr__(self, name, _positive_number(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class FleetAssignmentOverlay:
    assignment_id: str
    uav_id: str
    target_alias: str
    region_shape: str
    color: Color


@dataclass(frozen=True, slots=True)
class FleetUavOverlay:
    uav_id: str
    color: Color
    assignment_id: str | None
    target_alias: str | None
    agent_plan_version: int | None
    trajectory_segment_count: int


@dataclass(frozen=True, slots=True)
class FleetTargetOverlay:
    target_runtime_id: str
    semantic_alias: str
    claim_states: tuple[str, ...]
    claiming_uav_ids: tuple[str, ...]
    assignment_ids: tuple[str, ...]
    position_world_m: Point3 | None


@dataclass(frozen=True, slots=True)
class FleetConflictOverlay:
    uav_a_id: str
    uav_b_id: str
    risk: str
    hold_uav_id: str | None


@dataclass(frozen=True, slots=True)
class FleetDebugDrawSnapshot:
    """Bounded overlay metadata; camera pixels cannot be represented here."""

    fleet_mission_id: str | None
    fleet_plan_version: int | None
    minimum_uav_separation_m: float | None
    assignments: tuple[FleetAssignmentOverlay, ...]
    uavs: tuple[FleetUavOverlay, ...]
    targets: tuple[FleetTargetOverlay, ...]
    conflicts: tuple[FleetConflictOverlay, ...]
    viewport_only: bool = True


class FleetStatusOverlay(MissionStatusOverlay):
    """Viewport text for bounded Fleet metadata, never Camera RGB.

    The window/label lifecycle is shared with :class:`MissionStatusOverlay`;
    only the formatter differs.  ``omni.ui`` remains a lazy constructor-time
    import, so importing Fleet visualization is safe before ``SimulationApp``.
    """

    def __init__(self, *, ui_module: object | None = None) -> None:
        super().__init__(
            ui_module=ui_module,
            title="VLM Drone Fleet Status",
            width=680,
            height=460,
            initial_text="Fleet status unavailable",
        )

    def update(self, snapshot: FleetDebugDrawSnapshot) -> None:
        if not isinstance(snapshot, FleetDebugDrawSnapshot):
            raise TypeError("snapshot must be a FleetDebugDrawSnapshot")
        fleet_version = (
            "none"
            if snapshot.fleet_plan_version is None
            else str(snapshot.fleet_plan_version)
        )
        separation = (
            "none"
            if snapshot.minimum_uav_separation_m is None
            else f"{snapshot.minimum_uav_separation_m:g} m"
        )
        lines = [
            f"Fleet mission: {snapshot.fleet_mission_id or 'none'}",
            f"Fleet plan version: {fleet_version} | Minimum separation: {separation}",
            "Assignments:",
        ]
        if snapshot.assignments:
            lines.extend(
                "  "
                f"{item.uav_id} | assignment_id={item.assignment_id} | "
                f"target_alias={item.target_alias} | region={item.region_shape}"
                for item in snapshot.assignments
            )
        else:
            lines.append("  none")
        lines.append("UAVs:")
        if snapshot.uavs:
            lines.extend(
                "  "
                f"{item.uav_id} | assignment={item.assignment_id or 'none'} | "
                f"target={item.target_alias or 'none'} | "
                "local plan version="
                f"{item.agent_plan_version if item.agent_plan_version is not None else 'none'}"
                for item in snapshot.uavs
            )
        else:
            lines.append("  none")
        lines.append("Targets:")
        if snapshot.targets:
            lines.extend(
                "  "
                f"{item.target_runtime_id} | semantic_alias={item.semantic_alias} | "
                f"claims={','.join(item.claim_states) or 'none'} | "
                f"UAVs={','.join(item.claiming_uav_ids) or 'none'}"
                for item in snapshot.targets
            )
        else:
            lines.append("  none")
        lines.append("Conflicts:")
        if snapshot.conflicts:
            lines.extend(
                "  "
                f"{item.uav_a_id}/{item.uav_b_id}: {item.risk} | "
                f"HOLD={item.hold_uav_id or 'none'}"
                for item in snapshot.conflicts
            )
        else:
            lines.append("  clear")
        self._set_text("\n".join(lines))


class FleetDebugDraw:
    """One batched fleet overlay that does not alter single-UAV drawing."""

    def __init__(
        self,
        *,
        options: FleetDebugDrawOptions | None = None,
        draw_interface: object | None = None,
        debug_draw_module: object | None = None,
        status_overlay: object | None = None,
    ) -> None:
        if draw_interface is None:
            # Lazy import preserves SimulationApp-before-Isaac ordering.
            from isaacsim.util.debug_draw import _debug_draw

            self._debug_draw_module = _debug_draw
            self._draw = _debug_draw.acquire_debug_draw_interface()
            self._owns_draw_interface = True
        else:
            for method in ("clear_lines", "clear_points", "draw_lines", "draw_points"):
                if not callable(getattr(draw_interface, method, None)):
                    raise TypeError(f"draw_interface must provide {method}()")
            self._debug_draw_module = debug_draw_module
            self._draw = draw_interface
            self._owns_draw_interface = False
        if status_overlay is not None and (
            not callable(getattr(status_overlay, "update", None))
            or not callable(getattr(status_overlay, "close", None))
        ):
            raise TypeError("status_overlay must provide update() and close()")
        self._status_overlay = status_overlay
        self._options = options or FleetDebugDrawOptions()
        self._plan: FleetMissionPlan | None = None
        self._colors: dict[str, Color] = {}
        self._agent_plan_versions: dict[str, int | None] = {}
        self._assignment_lines: list[Line] = []
        self._assignment_points: list[Point] = []
        self._assignment_overlays: tuple[FleetAssignmentOverlay, ...] = ()
        self._trajectories: dict[str, deque[Line]] = {}
        self._last_positions: dict[str, Point3] = {}
        self._uav_points: list[Point] = []
        self._spacing_lines: list[Line] = []
        self._claim_lines: list[Line] = []
        self._target_points: list[Point] = []
        self._conflict_lines: list[Line] = []
        self._uav_overlays: tuple[FleetUavOverlay, ...] = ()
        self._target_overlays: tuple[FleetTargetOverlay, ...] = ()
        self._conflict_overlays: tuple[FleetConflictOverlay, ...] = ()

    @property
    def viewport_only(self) -> bool:
        return True

    def set_plan(
        self,
        plan: FleetMissionPlan,
        *,
        agent_plan_versions: Mapping[str, int | None] | None = None,
    ) -> None:
        if not isinstance(plan, FleetMissionPlan):
            raise TypeError("plan must be a FleetMissionPlan")
        uav_ids = frozenset(assignment.uav_id for assignment in plan.assignments)
        versions = _agent_versions(agent_plan_versions or {}, known_uavs=uav_ids)
        mission_changed = (
            self._plan is not None
            and self._plan.fleet_mission_id != plan.fleet_mission_id
        )
        if mission_changed:
            self._trajectories.clear()
            self._last_positions.clear()
        self._plan = plan
        self._colors = _uav_colors(tuple(uav_ids))
        self._agent_plan_versions = versions
        self._assignment_lines.clear()
        self._assignment_points.clear()
        overlays: list[FleetAssignmentOverlay] = []
        for assignment in plan.assignments:
            color = self._colors[assignment.uav_id]
            region = assignment.search_region
            altitude = _region_reference_altitude(region) + 0.05
            self._assignment_lines.extend(
                _region_lines(
                    region,
                    altitude,
                    _with_alpha(color, 0.70),
                    self._options.static_line_width,
                    self._options.circle_segments,
                )
            )
            self._assignment_points.append(
                (_region_center_for_draw(region, altitude), color, 10.0)
            )
            overlays.append(
                FleetAssignmentOverlay(
                    assignment.assignment_id,
                    assignment.uav_id,
                    assignment.target_alias,
                    region.shape,
                    color,
                )
            )
            self._trajectories.setdefault(
                assignment.uav_id,
                deque(maxlen=self._options.max_trajectory_segments_per_uav),
            )
        self._assignment_overlays = tuple(overlays)
        self._rebuild_uav_overlays(())
        self._redraw()
        self._update_overlay()

    def update(
        self,
        *,
        poses: FleetPoseSnapshot,
        target_records: Sequence[SharedTargetRecord] = (),
        target_positions_world_m: Mapping[str, Sequence[float]] | None = None,
        airspace_decision: FleetAirspaceDecision | None = None,
        agent_plan_versions: Mapping[str, int | None] | None = None,
    ) -> None:
        if self._plan is None:
            raise RuntimeError("set_plan() must be called before update()")
        if not isinstance(poses, FleetPoseSnapshot):
            raise TypeError("poses must be a FleetPoseSnapshot")
        known_uavs = frozenset(self._colors)
        unknown = set(poses.poses) - known_uavs
        if unknown:
            raise ValueError("poses contains unplanned UAVs: " + ", ".join(sorted(unknown)))
        if agent_plan_versions is not None:
            self._agent_plan_versions = _agent_versions(
                agent_plan_versions,
                known_uavs=known_uavs,
            )
        if airspace_decision is not None and not isinstance(
            airspace_decision, FleetAirspaceDecision
        ):
            raise TypeError("airspace_decision must be a FleetAirspaceDecision or None")
        records = tuple(target_records)
        if any(not isinstance(record, SharedTargetRecord) for record in records):
            raise TypeError("target_records must contain SharedTargetRecord values")
        positions = self._target_positions(target_positions_world_m or {}, records)

        self._uav_points.clear()
        self._spacing_lines.clear()
        minimum = self._plan.coordination_policy.minimum_uav_separation_m
        for uav_id, pose in poses.poses.items():
            self._update_uav(pose, minimum)
        self._rebuild_uav_overlays(tuple(poses.poses.values()))
        self._rebuild_targets(records, positions, poses)
        self._rebuild_conflicts(airspace_decision, poses)
        self._redraw()
        self._update_overlay()

    def snapshot(self) -> FleetDebugDrawSnapshot:
        plan = self._plan
        return FleetDebugDrawSnapshot(
            fleet_mission_id=None if plan is None else plan.fleet_mission_id,
            fleet_plan_version=None if plan is None else plan.fleet_plan_version,
            minimum_uav_separation_m=(
                None
                if plan is None
                else plan.coordination_policy.minimum_uav_separation_m
            ),
            assignments=self._assignment_overlays,
            uavs=self._uav_overlays,
            targets=self._target_overlays,
            conflicts=self._conflict_overlays,
        )

    def close(self) -> None:
        if self._draw is not None:
            try:
                self._draw.clear_lines()
                self._draw.clear_points()
            finally:
                if self._owns_draw_interface:
                    self._debug_draw_module.release_debug_draw_interface(self._draw)
                self._draw = None
        if self._status_overlay is not None:
            self._status_overlay.close()
            self._status_overlay = None

    def _update_uav(self, pose: FleetUavPose, minimum_separation_m: float) -> None:
        color = self._colors[pose.uav_id]
        current = pose.position_xyz_m
        previous = self._last_positions.get(pose.uav_id)
        if previous is None:
            self._last_positions[pose.uav_id] = current
        elif dist(previous, current) >= self._options.trajectory_min_spacing_m:
            self._trajectories[pose.uav_id].append(
                (previous, current, color, self._options.trajectory_line_width)
            )
            self._last_positions[pose.uav_id] = current
        self._uav_points.append((current, color, 13.0))
        self._spacing_lines.extend(
            _circle_lines(
                current,
                minimum_separation_m,
                _with_alpha(color, 0.35),
                self._options.static_line_width,
                self._options.circle_segments,
            )
        )

    def _rebuild_uav_overlays(self, poses: Sequence[FleetUavPose]) -> None:
        pose_by_uav = {pose.uav_id: pose for pose in poses}
        assignment_by_uav = {
            assignment.uav_id: assignment for assignment in self._plan.assignments
        } if self._plan is not None else {}
        overlays: list[FleetUavOverlay] = []
        for uav_id in sorted(self._colors):
            assignment = assignment_by_uav.get(uav_id)
            overlays.append(
                FleetUavOverlay(
                    uav_id=uav_id,
                    color=self._colors[uav_id],
                    assignment_id=(
                        pose_by_uav[uav_id].assignment_id
                        if uav_id in pose_by_uav
                        and pose_by_uav[uav_id].assignment_id is not None
                        else None if assignment is None else assignment.assignment_id
                    ),
                    target_alias=None if assignment is None else assignment.target_alias,
                    agent_plan_version=self._agent_plan_versions.get(uav_id),
                    trajectory_segment_count=len(self._trajectories.get(uav_id, ())),
                )
            )
        self._uav_overlays = tuple(overlays)

    def _target_positions(
        self,
        values: Mapping[str, Sequence[float]],
        records: Sequence[SharedTargetRecord],
    ) -> dict[str, Point3]:
        if not isinstance(values, Mapping):
            raise TypeError("target_positions_world_m must be a mapping")
        known = {record.target_runtime_id for record in records}
        result: dict[str, Point3] = {}
        for raw_target_id, raw_position in values.items():
            target_id = validate_routing_id(raw_target_id, "target_runtime_id")
            if target_id not in known:
                raise ValueError(f"target position has no matching target record: {target_id}")
            result[target_id] = _point(raw_position, f"target_positions_world_m[{target_id}]")
        return result

    def _rebuild_targets(
        self,
        records: Sequence[SharedTargetRecord],
        positions: Mapping[str, Point3],
        poses: FleetPoseSnapshot,
    ) -> None:
        self._claim_lines.clear()
        self._target_points.clear()
        overlays: list[FleetTargetOverlay] = []
        for record in sorted(records, key=lambda item: item.target_runtime_id):
            active = tuple(claim for claim in record.claims if claim.active)
            position = positions.get(record.target_runtime_id)
            if position is not None:
                self._target_points.append((position, _TARGET_COLOR, 14.0))
                for claim in active:
                    pose = poses.poses.get(claim.uav_id)
                    if pose is not None:
                        self._claim_lines.append(
                            (
                                pose.position_xyz_m,
                                position,
                                _with_alpha(self._colors[claim.uav_id], 0.85),
                                self._options.static_line_width + 1.0,
                            )
                        )
            overlays.append(
                FleetTargetOverlay(
                    target_runtime_id=record.target_runtime_id,
                    semantic_alias=record.semantic_alias,
                    claim_states=tuple(claim.state.value for claim in record.claims),
                    claiming_uav_ids=tuple(claim.uav_id for claim in active),
                    assignment_ids=tuple(claim.assignment_id for claim in active),
                    position_world_m=position,
                )
            )
        self._target_overlays = tuple(overlays)

    def _rebuild_conflicts(
        self,
        decision: FleetAirspaceDecision | None,
        poses: FleetPoseSnapshot,
    ) -> None:
        self._conflict_lines.clear()
        overlays: list[FleetConflictOverlay] = []
        if decision is not None:
            for conflict in decision.conflicts:
                if not conflict.is_conflict:
                    continue
                a = poses.poses.get(conflict.uav_a_id)
                b = poses.poses.get(conflict.uav_b_id)
                if a is not None and b is not None:
                    self._conflict_lines.append(
                        (
                            a.position_xyz_m,
                            b.position_xyz_m,
                            _CONFLICT_COLOR,
                            self._options.conflict_line_width,
                        )
                    )
                overlays.append(
                    FleetConflictOverlay(
                        conflict.uav_a_id,
                        conflict.uav_b_id,
                        conflict.risk.value,
                        conflict.hold_uav_id,
                    )
                )
        self._conflict_overlays = tuple(overlays)

    def _redraw(self) -> None:
        if self._draw is None:
            return
        self._draw.clear_lines()
        self._draw.clear_points()
        trajectory_lines = tuple(
            line
            for uav_id in sorted(self._trajectories)
            for line in self._trajectories[uav_id]
        )
        lines = (
            *self._assignment_lines,
            *self._spacing_lines,
            *self._claim_lines,
            *self._conflict_lines,
            *trajectory_lines,
        )
        if lines:
            self._draw.draw_lines(
                [line[0] for line in lines],
                [line[1] for line in lines],
                [line[2] for line in lines],
                [line[3] for line in lines],
            )
        points = (*self._assignment_points, *self._uav_points, *self._target_points)
        if points:
            self._draw.draw_points(
                [point[0] for point in points],
                [point[1] for point in points],
                [point[2] for point in points],
            )

    def _update_overlay(self) -> None:
        if self._status_overlay is not None:
            self._status_overlay.update(self.snapshot())


__all__ = [
    "FleetAssignmentOverlay",
    "FleetConflictOverlay",
    "FleetDebugDraw",
    "FleetDebugDrawOptions",
    "FleetDebugDrawSnapshot",
    "FleetStatusOverlay",
    "FleetTargetOverlay",
    "FleetUavOverlay",
]
