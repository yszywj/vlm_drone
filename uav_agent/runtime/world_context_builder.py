"""Build the planner's non-privileged world context from trusted config.

This module is intentionally pure Python.  It derives only named mission
regions and static safety geometry; it has no access to a live environment,
per-frame observations, or a spawned target state.
"""

from __future__ import annotations

from math import isfinite
from numbers import Real

from configs.schema import AppConfig
from planner.schemas import (
    LandingZoneSpec,
    PlannerWorldContext,
    SearchRegionSpec,
)


SEARCH_REGION_NAME = "search_area"
LANDING_ZONE_NAME = "home"


class WorldContextBuildError(ValueError):
    """Raised when config geometry cannot form a safe planner context."""


def build_planner_world_context(
    config: AppConfig,
    *,
    takeoff_altitude_m: float,
    track_duration_s: float,
    start_altitude_m: float | None = None,
    goto_timeout_s: float = 120.0,
    land_timeout_s: float = 60.0,
) -> PlannerWorldContext:
    """Return the fixed ``search_area``/``home`` planner context.

    ``start_altitude_m`` represents the effective reset altitude used by a
    runtime CLI.  When omitted, the configured UAV altitude is retained.  All
    other geometry comes from static configuration: the centered scene bounds,
    UAV home XY, target *initial region*, and configured search radius.
    """

    if not isinstance(config, AppConfig):
        raise TypeError("config must be a configs.schema.AppConfig")

    size = _vector3(config.scene.size_xyz_m, "scene.size_xyz_m")
    if any(component <= 0.0 for component in size):
        raise WorldContextBuildError(
            "scene.size_xyz_m components must be greater than zero"
        )
    scene_min = (-size[0] / 2.0, -size[1] / 2.0, 0.0)
    scene_max = (size[0] / 2.0, size[1] / 2.0, size[2])

    configured_uav = _vector3(
        config.uav.initial_position_xyz_m,
        "uav.initial_position_xyz_m",
    )
    effective_start_z = (
        configured_uav[2]
        if start_altitude_m is None
        else _finite_number(start_altitude_m, "start_altitude_m")
    )
    initial_uav = (configured_uav[0], configured_uav[1], effective_start_z)
    _require_point_in_bounds(initial_uav, scene_min, scene_max, "initial UAV position")

    takeoff_altitude = _positive_number(
        takeoff_altitude_m,
        "takeoff_altitude_m",
    )
    if not scene_min[2] < takeoff_altitude <= scene_max[2]:
        raise WorldContextBuildError(
            "takeoff_altitude_m is outside valid scene flight bounds"
        )
    if effective_start_z > takeoff_altitude:
        raise WorldContextBuildError(
            "start_altitude_m must not exceed takeoff_altitude_m"
        )
    track_duration = _positive_number(track_duration_s, "track_duration_s")
    if not 1.0 <= track_duration <= 600.0:
        raise WorldContextBuildError(
            "track_duration_s must be between 1 and 600 seconds"
        )
    goto_timeout = _positive_number(goto_timeout_s, "goto_timeout_s")
    land_timeout = _positive_number(land_timeout_s, "land_timeout_s")
    search_timeout = _positive_number(
        config.search.timeout_s,
        "search.timeout_s",
    )
    radius = _positive_number(config.search.radius_m, "search.radius_m")

    region_min = _vector3(
        config.target.initial_region.min_xyz_m,
        "target.initial_region.min_xyz_m",
    )
    region_max = _vector3(
        config.target.initial_region.max_xyz_m,
        "target.initial_region.max_xyz_m",
    )
    if any(lower > upper for lower, upper in zip(region_min, region_max)):
        raise WorldContextBuildError(
            "target.initial_region minimum must not exceed maximum"
        )
    _require_point_in_bounds(
        region_min,
        scene_min,
        scene_max,
        "target.initial_region minimum",
    )
    _require_point_in_bounds(
        region_max,
        scene_min,
        scene_max,
        "target.initial_region maximum",
    )
    center = tuple(
        (lower + upper) / 2.0
        for lower, upper in zip(region_min, region_max)
    )
    center_xyz = (center[0], center[1], center[2])

    # SafetySupervisor requires the complete horizontal search disk to remain
    # in bounds.  Reject incompatible configuration rather than silently
    # shrinking the operator-selected search radius.
    if (
        center_xyz[0] - radius < scene_min[0]
        or center_xyz[0] + radius > scene_max[0]
        or center_xyz[1] - radius < scene_min[1]
        or center_xyz[1] + radius > scene_max[1]
    ):
        raise WorldContextBuildError(
            "configured search_area radius leaves the scene XY bounds"
        )

    # Approach from the west edge while remaining at the requested flight
    # altitude.  This keeps the first search transit aimed into the region.
    approach = (
        center_xyz[0] - radius,
        center_xyz[1],
        takeoff_altitude,
    )
    _require_point_in_bounds(approach, scene_min, scene_max, "search approach")

    home_xy = (initial_uav[0], initial_uav[1])
    ground_altitude = 0.0
    if not scene_min[2] <= ground_altitude <= scene_max[2]:
        raise WorldContextBuildError("home ground altitude is outside scene bounds")

    try:
        return PlannerWorldContext(
            scene_min_xyz_m=scene_min,
            scene_max_xyz_m=scene_max,
            initial_uav_xyz_m=initial_uav,
            search_regions={
                SEARCH_REGION_NAME: SearchRegionSpec(
                    name=SEARCH_REGION_NAME,
                    center_xyz_m=center_xyz,
                    radius_m=radius,
                    approach_xyz_m=approach,
                    description="configured semantic search area",
                )
            },
            landing_zones={
                LANDING_ZONE_NAME: LandingZoneSpec(
                    name=LANDING_ZONE_NAME,
                    position_xy_m=home_xy,
                    ground_altitude_m=ground_altitude,
                    description="configured UAV home landing zone",
                )
            },
            default_takeoff_altitude_m=takeoff_altitude,
            default_track_duration_s=track_duration,
            search_timeout_s=search_timeout,
            goto_timeout_s=goto_timeout,
            land_timeout_s=land_timeout,
        )
    except (TypeError, ValueError) as exc:
        raise WorldContextBuildError(
            f"could not construct PlannerWorldContext: {exc}"
        ) from exc


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise WorldContextBuildError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise WorldContextBuildError(f"{name} must be a finite number") from exc
    if not isfinite(parsed):
        raise WorldContextBuildError(f"{name} must be a finite number")
    return parsed


def _positive_number(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed <= 0.0:
        raise WorldContextBuildError(f"{name} must be greater than zero")
    return parsed


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise WorldContextBuildError(f"{name} must contain exactly three numbers")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise WorldContextBuildError(
            f"{name} must contain exactly three numbers"
        ) from exc
    if len(items) != 3:
        raise WorldContextBuildError(f"{name} must contain exactly three numbers")
    parsed = tuple(
        _finite_number(component, f"{name}[{index}]")
        for index, component in enumerate(items)
    )
    return parsed[0], parsed[1], parsed[2]


def _require_point_in_bounds(
    point: tuple[float, float, float],
    scene_min: tuple[float, float, float],
    scene_max: tuple[float, float, float],
    name: str,
) -> None:
    if any(
        coordinate < lower or coordinate > upper
        for coordinate, lower, upper in zip(point, scene_min, scene_max)
    ):
        raise WorldContextBuildError(f"{name} is outside the scene bounds")


__all__ = [
    "LANDING_ZONE_NAME",
    "SEARCH_REGION_NAME",
    "WorldContextBuildError",
    "build_planner_world_context",
]
