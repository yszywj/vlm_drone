"""Validate a high-level mission intent and compile a deterministic TaskPlan."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from planner.schemas import (
    CompiledMission,
    LandingZoneSpec,
    MissionIntent,
    PlannerWorldContext,
    SearchRegionSpec,
)
from skills.manager import TaskPlan, TaskPlanError

if TYPE_CHECKING:
    from collections.abc import Sequence


class PlanValidationError(ValueError):
    """Raised when an intent cannot safely be compiled in its world context."""


class PlanValidator:
    """Compile planner output into the only supported six-step mission shape.

    The planner names semantic regions.  This boundary resolves those names to
    trusted world coordinates and adds all low-level timeouts.  Consequently a
    scripted or future LLM planner never needs access to target truth or motion
    controller parameters.
    """

    MIN_TRACK_DURATION_S = 1.0
    MAX_TRACK_DURATION_S = 600.0
    MAX_TARGET_DESCRIPTION_CHARS = 256
    ALLOWED_SOURCES = frozenset({"scripted", "llm"})

    def validate_and_compile(
        self,
        intent: MissionIntent,
        context: PlannerWorldContext,
        *,
        source: str,
    ) -> CompiledMission:
        """Return a validated six-step plan with no statically planned recovery."""

        if not isinstance(intent, MissionIntent):
            raise TypeError("intent must be a MissionIntent")
        if not isinstance(context, PlannerWorldContext):
            raise TypeError("context must be a PlannerWorldContext")
        if not isinstance(source, str) or source not in self.ALLOWED_SOURCES:
            raise PlanValidationError("source must be 'scripted' or 'llm'")

        description = intent.target_description
        if not isinstance(description, str) or not description.strip():
            raise PlanValidationError("target_description must be non-empty")
        if len(description) > self.MAX_TARGET_DESCRIPTION_CHARS:
            raise PlanValidationError(
                "target_description must contain at most "
                f"{self.MAX_TARGET_DESCRIPTION_CHARS} characters"
            )

        try:
            search_region = context.search_regions[intent.search_region]
        except KeyError as exc:
            raise PlanValidationError(
                f"unknown search_region: {intent.search_region}"
            ) from exc
        if not isinstance(search_region, SearchRegionSpec):
            raise PlanValidationError(
                f"search region {intent.search_region!r} has an invalid specification"
            )

        try:
            landing_zone = context.landing_zones[intent.landing_zone]
        except KeyError as exc:
            raise PlanValidationError(
                f"unknown landing_zone: {intent.landing_zone}"
            ) from exc
        if not isinstance(landing_zone, LandingZoneSpec):
            raise PlanValidationError(
                f"landing zone {intent.landing_zone!r} has an invalid specification"
            )

        scene_min = _finite_vector3(context.scene_min_xyz_m, "scene_min_xyz_m")
        scene_max = _finite_vector3(context.scene_max_xyz_m, "scene_max_xyz_m")
        if any(lower >= upper for lower, upper in zip(scene_min, scene_max)):
            raise PlanValidationError(
                "each scene_min_xyz_m component must be smaller than scene_max_xyz_m"
            )

        # Recheck every scalar used in the generated plan at this trust boundary,
        # even though the immutable schemas normally catch malformed values first.
        takeoff_altitude = _positive_finite_number(
            context.default_takeoff_altitude_m
            if intent.takeoff_altitude_m is None
            else intent.takeoff_altitude_m,
            "takeoff_altitude_m",
        )
        track_duration = _finite_number(
            intent.track_duration_s,
            "track_duration_s",
        )
        search_timeout = _positive_finite_number(
            context.search_timeout_s,
            "search_timeout_s",
        )
        goto_timeout = _positive_finite_number(
            context.goto_timeout_s,
            "goto_timeout_s",
        )
        land_timeout = _positive_finite_number(
            context.land_timeout_s,
            "land_timeout_s",
        )
        radius = _positive_finite_number(search_region.radius_m, "search radius")
        ground_altitude = _finite_number(
            landing_zone.ground_altitude_m,
            "landing ground_altitude_m",
        )

        if not self.MIN_TRACK_DURATION_S <= track_duration <= self.MAX_TRACK_DURATION_S:
            raise PlanValidationError(
                "track_duration_s must be between "
                f"{self.MIN_TRACK_DURATION_S:g} and "
                f"{self.MAX_TRACK_DURATION_S:g} seconds"
            )
        if not scene_min[2] <= takeoff_altitude <= scene_max[2]:
            raise PlanValidationError("takeoff altitude is outside the scene Z bounds")

        initial_uav = _finite_vector3(
            context.initial_uav_xyz_m,
            "initial_uav_xyz_m",
        )
        _require_point_in_bounds(
            initial_uav,
            scene_min,
            scene_max,
            "initial UAV position",
        )
        if takeoff_altitude < initial_uav[2]:
            raise PlanValidationError(
                "takeoff altitude must not be below the initial UAV altitude"
            )

        center = _finite_vector3(search_region.center_xyz_m, "search center")
        approach = _finite_vector3(search_region.approach_xyz_m, "search approach")
        _require_point_in_bounds(center, scene_min, scene_max, "search center")
        _require_point_in_bounds(approach, scene_min, scene_max, "search approach")

        landing_xy = _finite_vector2(
            landing_zone.position_xy_m,
            "landing position_xy_m",
        )
        if not (
            scene_min[0] <= landing_xy[0] <= scene_max[0]
            and scene_min[1] <= landing_xy[1] <= scene_max[1]
        ):
            raise PlanValidationError("landing position is outside the scene XY bounds")
        if not scene_min[2] <= ground_altitude <= scene_max[2]:
            raise PlanValidationError(
                "landing ground altitude is outside the scene Z bounds"
            )
        if ground_altitude > takeoff_altitude:
            raise PlanValidationError(
                "landing ground altitude must not exceed the flight altitude"
            )

        # PlannerWorldContext deliberately has no separate TAKEOFF timeout.  The
        # trusted GOTO timeout is also used for this initial bounded motion.
        raw_plan: list[dict[str, object]] = [
            {
                "skill": "TAKEOFF",
                "target_altitude": takeoff_altitude,
                "timeout": goto_timeout,
            },
            {
                "skill": "GOTO",
                "position": list(approach),
                "timeout": goto_timeout,
            },
            {
                "skill": "SEARCH",
                "center": list(center),
                "radius": radius,
                "target_description": description,
                "search_altitude": takeoff_altitude,
                "timeout": search_timeout,
            },
            {
                "skill": "TRACK",
                "target_id": "$SEARCH.result.target_id",
                "desired_altitude": takeoff_altitude,
                "track_duration": track_duration,
            },
            {
                "skill": "GOTO",
                "position": [landing_xy[0], landing_xy[1], takeoff_altitude],
                "timeout": goto_timeout,
            },
            {
                "skill": "LAND",
                "ground_altitude": ground_altitude,
                "timeout": land_timeout,
            },
        ]
        try:
            task_plan = TaskPlan.from_dicts(raw_plan)
        except TaskPlanError as exc:
            raise PlanValidationError(
                f"compiled TaskPlan is invalid: {exc}"
            ) from exc
        return CompiledMission(intent=intent, task_plan=task_plan, source=source)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PlanValidationError(f"{name} must be a finite number")
    parsed = float(value)
    if not isfinite(parsed):
        raise PlanValidationError(f"{name} must be a finite number")
    return parsed


def _positive_finite_number(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed <= 0.0:
        raise PlanValidationError(f"{name} must be greater than zero")
    return parsed


def _finite_vector(
    value: object,
    size: int,
    name: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise PlanValidationError(f"{name} must contain exactly {size} numbers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise PlanValidationError(
            f"{name} must contain exactly {size} numbers"
        ) from exc
    if len(values) != size:
        raise PlanValidationError(f"{name} must contain exactly {size} numbers")
    return tuple(_finite_number(component, name) for component in values)


def _finite_vector2(value: object, name: str) -> tuple[float, float]:
    parsed = _finite_vector(value, 2, name)
    return parsed[0], parsed[1]


def _finite_vector3(value: object, name: str) -> tuple[float, float, float]:
    parsed = _finite_vector(value, 3, name)
    return parsed[0], parsed[1], parsed[2]


def _require_point_in_bounds(
    point: "Sequence[float]",
    scene_min: "Sequence[float]",
    scene_max: "Sequence[float]",
    name: str,
) -> None:
    if any(
        coordinate < lower or coordinate > upper
        for coordinate, lower, upper in zip(point, scene_min, scene_max)
    ):
        raise PlanValidationError(f"{name} is outside the scene bounds")


__all__ = ["PlanValidationError", "PlanValidator"]
