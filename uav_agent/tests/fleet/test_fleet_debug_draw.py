from __future__ import annotations

from dataclasses import fields
from inspect import signature

import pytest

from fleet.airspace_manager import FleetAirspaceManager, FleetPoseSnapshot, FleetUavPose
from fleet.target_registry import SharedTargetRegistry, TargetClaimState
from fleet.types import FleetAssignment, FleetCoordinationPolicy, FleetMissionPlan
from planner.spatial import CircleRegion, CoordinateFrame, RectangleRegion
from target.types import TargetSpec
from visualization.fleet_debug_draw import (
    FleetDebugDraw,
    FleetDebugDrawOptions,
    FleetDebugDrawSnapshot,
    FleetStatusOverlay,
)


class _FakeDraw:
    def __init__(self) -> None:
        self.clear_lines_count = 0
        self.clear_points_count = 0
        self.line_batches: list[tuple[list[object], list[object], list[object], list[object]]] = []
        self.point_batches: list[tuple[list[object], list[object], list[object]]] = []

    def clear_lines(self) -> None:
        self.clear_lines_count += 1

    def clear_points(self) -> None:
        self.clear_points_count += 1

    def draw_lines(self, starts, ends, colors, widths) -> None:
        self.line_batches.append((list(starts), list(ends), list(colors), list(widths)))

    def draw_points(self, positions, colors, sizes) -> None:
        self.point_batches.append((list(positions), list(colors), list(sizes)))


class _FakeOverlay:
    def __init__(self) -> None:
        self.snapshots: list[FleetDebugDrawSnapshot] = []
        self.closed = False

    def update(self, snapshot: FleetDebugDrawSnapshot) -> None:
        self.snapshots.append(snapshot)

    def close(self) -> None:
        self.closed = True


class _FakeFrame:
    def __enter__(self) -> "_FakeFrame":
        return self

    def __exit__(self, *args: object) -> None:
        del args


class _FakeWindow:
    def __init__(self, title: str, **options: object) -> None:
        self.title = title
        self.options = options
        self.frame = _FakeFrame()
        self.visible = True
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class _FakeLabel:
    def __init__(self, text: str, **options: object) -> None:
        self.text = text
        self.options = options


class _FakeUI:
    Window = _FakeWindow
    Label = _FakeLabel


def _plan() -> FleetMissionPlan:
    return FleetMissionPlan(
        fleet_mission_id="fleet_visual_test",
        fleet_plan_version=4,
        assignments=(
            FleetAssignment(
                "assignment_a_i",
                "uav_a",
                "target_i",
                TargetSpec("red person i", category="person"),
                CircleRegion(CoordinateFrame.WORLD_ENU, (10.0, 0.0, 0.0), 4.0),
                10.0,
                priority=100,
            ),
            FleetAssignment(
                "assignment_b_j",
                "uav_b",
                "target_j",
                TargetSpec("blue car j", category="car"),
                RectangleRegion(
                    CoordinateFrame.WORLD_ENU,
                    (20.0, 5.0, 0.0),
                    8.0,
                    6.0,
                ),
                12.0,
                priority=50,
            ),
        ),
        coordination_policy=FleetCoordinationPolicy(
            minimum_uav_separation_m=5.0
        ),
    )


def _poses(timestamp_s: float, *, offset: float = 0.0) -> FleetPoseSnapshot:
    return FleetPoseSnapshot(
        timestamp_s,
        {
            "uav_a": FleetUavPose(
                "uav_a",
                (0.0 + offset, 0.0, 10.0),
                (1.0, 0.0, 0.0),
                priority=100,
                assignment_id="assignment_a_i",
            ),
            "uav_b": FleetUavPose(
                "uav_b",
                (8.0 - offset, 0.0, 10.0),
                (-1.0, 0.0, 0.0),
                priority=50,
                assignment_id="assignment_b_j",
            ),
        },
    )


def _target_records():
    registry = SharedTargetRegistry()
    registry.bind_assignment(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="runtime_target_i",
        semantic_alias="target_i",
        priority=100,
    )
    registry.claim(
        assignment_id="assignment_a_i",
        uav_id="uav_a",
        target_runtime_id="runtime_target_i",
        confidence=0.9,
        timestamp_s=1.0,
        state=TargetClaimState.EXCLUSIVE,
    )
    registry.bind_assignment(
        assignment_id="assignment_b_j",
        uav_id="uav_b",
        target_runtime_id="runtime_target_j",
        semantic_alias="target_j",
        priority=50,
    )
    return (
        registry.record("runtime_target_i"),
        registry.record("runtime_target_j"),
    )


def test_fleet_draw_renders_distinct_uavs_regions_claims_spacing_and_conflict() -> None:
    draw = _FakeDraw()
    overlay = _FakeOverlay()
    visualizer = FleetDebugDraw(draw_interface=draw, status_overlay=overlay)
    plan = _plan()
    visualizer.set_plan(
        plan,
        agent_plan_versions={"uav_a": 7, "uav_b": 3},
    )

    initial = _poses(1.0)
    visualizer.update(
        poses=initial,
        target_records=_target_records(),
        target_positions_world_m={
            "runtime_target_i": (12.0, 1.0, 0.5),
            "runtime_target_j": (22.0, 5.0, 0.5),
        },
        airspace_decision=FleetAirspaceManager(5.0).evaluate(initial),
    )
    moved = _poses(2.0, offset=1.0)
    conflict = FleetAirspaceManager(5.0).evaluate(moved)
    visualizer.update(
        poses=moved,
        target_records=_target_records(),
        target_positions_world_m={
            "runtime_target_i": (12.0, 1.0, 0.5),
            "runtime_target_j": (22.0, 5.0, 0.5),
        },
        airspace_decision=conflict,
        agent_plan_versions={"uav_a": 8, "uav_b": 3},
    )

    snapshot = visualizer.snapshot()
    assert snapshot.viewport_only
    assert snapshot.fleet_mission_id == "fleet_visual_test"
    assert snapshot.fleet_plan_version == 4
    assert snapshot.minimum_uav_separation_m == 5.0
    assert [(item.assignment_id, item.region_shape) for item in snapshot.assignments] == [
        ("assignment_a_i", "CIRCLE"),
        ("assignment_b_j", "RECTANGLE"),
    ]
    assert snapshot.assignments[0].color != snapshot.assignments[1].color
    assert [item.agent_plan_version for item in snapshot.uavs] == [8, 3]
    assert [item.trajectory_segment_count for item in snapshot.uavs] == [1, 1]
    assert [(item.semantic_alias, item.claim_states) for item in snapshot.targets] == [
        ("target_i", ("EXCLUSIVE",)),
        ("target_j", ("PROVISIONAL",)),
    ]
    assert snapshot.targets[0].claiming_uav_ids == ("uav_a",)
    assert snapshot.targets[1].claiming_uav_ids == ("uav_b",)
    assert len(snapshot.conflicts) == 1
    assert snapshot.conflicts[0].risk == "COLLISION_PREDICTED"
    assert snapshot.conflicts[0].hold_uav_id == "uav_b"
    assert overlay.snapshots[-1] == snapshot

    starts, ends, colors, widths = draw.line_batches[-1]
    # Exact conflict segment is present and uses the widest red line.
    conflict_index = next(
        index
        for index, (start, end) in enumerate(zip(starts, ends))
        if start == moved.poses["uav_a"].position_xyz_m
        and end == moved.poses["uav_b"].position_xyz_m
    )
    assert colors[conflict_index] == (1.0, 0.05, 0.05, 1.0)
    assert widths[conflict_index] == 5.0
    # Two minimum-separation circles contribute 48 segments each.
    assert sum(width == 2.0 for width in widths) >= 96


def test_fleet_draw_is_rgb_free_bounded_and_does_not_own_injected_interface() -> None:
    draw = _FakeDraw()
    overlay = _FakeOverlay()
    visualizer = FleetDebugDraw(
        draw_interface=draw,
        status_overlay=overlay,
        options=FleetDebugDrawOptions(
            trajectory_min_spacing_m=0.01,
            max_trajectory_segments_per_uav=2,
        ),
    )
    visualizer.set_plan(_plan())
    for index in range(5):
        visualizer.update(poses=_poses(float(index), offset=float(index) * 0.1))

    snapshot = visualizer.snapshot()
    assert [item.trajectory_segment_count for item in snapshot.uavs] == [2, 2]
    assert "rgb" not in signature(FleetDebugDraw.update).parameters
    assert "camera" not in signature(FleetDebugDraw.update).parameters
    assert all(
        "rgb" not in field.name.casefold() and "camera" not in field.name.casefold()
        for field in fields(FleetDebugDrawSnapshot)
    )

    clears_before = (draw.clear_lines_count, draw.clear_points_count)
    visualizer.close()
    assert overlay.closed
    assert draw.clear_lines_count == clears_before[0] + 1
    assert draw.clear_points_count == clears_before[1] + 1


def test_fleet_status_overlay_displays_ids_aliases_versions_and_hold() -> None:
    draw = _FakeDraw()
    overlay = FleetStatusOverlay(ui_module=_FakeUI)
    visualizer = FleetDebugDraw(draw_interface=draw, status_overlay=overlay)
    visualizer.set_plan(
        _plan(),
        agent_plan_versions={"uav_a": 7, "uav_b": 3},
    )
    poses = _poses(2.0, offset=1.0)
    visualizer.update(
        poses=poses,
        target_records=_target_records(),
        target_positions_world_m={
            "runtime_target_i": (12.0, 1.0, 0.5),
            "runtime_target_j": (22.0, 5.0, 0.5),
        },
        airspace_decision=FleetAirspaceManager(5.0).evaluate(poses),
    )

    text = overlay.text
    assert "Fleet mission: fleet_visual_test" in text
    assert "Fleet plan version: 4" in text
    assert "assignment_id=assignment_a_i" in text
    assert "target_alias=target_i" in text
    assert "local plan version=7" in text
    assert "runtime_target_i | semantic_alias=target_i" in text
    assert "HOLD=uav_b" in text
    assert "rgb" not in text.casefold()
    assert "camera" not in text.casefold()
    window = overlay._window
    assert window.title == "VLM Drone Fleet Status"
    assert window.options == {"width": 680, "height": 460}

    visualizer.close()
    assert window.destroyed
    assert not window.visible


def test_fleet_draw_rejects_unplanned_pose_and_unknown_target_position() -> None:
    visualizer = FleetDebugDraw(draw_interface=_FakeDraw())
    visualizer.set_plan(_plan())
    with pytest.raises(ValueError, match="unplanned UAV"):
        visualizer.update(
            poses=FleetPoseSnapshot(
                1.0,
                {"uav_c": FleetUavPose("uav_c", (0.0, 0.0, 0.0))},
            )
        )
    with pytest.raises(ValueError, match="no matching target record"):
        visualizer.update(
            poses=_poses(1.0),
            target_positions_world_m={"runtime_unknown": (1.0, 2.0, 3.0)},
        )
