"""Trusted compilation from a V3 RegionSpec to executable SEARCH geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from numbers import Real
from collections.abc import Sequence

from planner.spatial import CoordinateFrame, RectangleAnchor, RectangleRegion, RegionSpec, RelationalRegion
from planner.spatial_resolver import SpatialResolver
from skills.search_geometry import (
    Point3,
    generate_search_waypoints,
    nearest_boundary_point,
    point_inside_region,
    rectangle_anchor_point,
    region_center,
)
from skills.search_strategy import (
    SearchEntryPolicy,
    SearchRuntimeCapabilities,
    SearchStrategyError,
    SearchStrategySpec,
    SearchStrategyType,
)


class RegionCompilationError(ValueError):
    """Raised when a region/strategy cannot produce safe bounded geometry."""


def _point(value: object, name: str) -> Point3:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise RegionCompilationError(f"{name} must contain exactly three numbers")
    result: list[float] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, Real):
            raise TypeError(f"{name}[{index}] must be a finite number")
        number = float(component)
        if not isfinite(number):
            raise RegionCompilationError(f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CompiledSearchGeometry:
    """Resolved macro waypoints; no 60 Hz controller commands are included."""

    region: RegionSpec
    strategy: SearchStrategySpec
    entry_policy: SearchEntryPolicy
    entry_point_xyz_m: Point3
    observation_waypoints_xyz_m: tuple[Point3, ...]
    route_waypoints_xyz_m: tuple[Point3, ...]

    def __post_init__(self) -> None:
        if isinstance(self.region, RelationalRegion) or self.region.frame is not CoordinateFrame.WORLD_ENU:
            raise RegionCompilationError("compiled region must be resolved WORLD_ENU geometry")
        if not isinstance(self.strategy, SearchStrategySpec):
            raise TypeError("strategy must be a SearchStrategySpec")
        if not isinstance(self.entry_policy, SearchEntryPolicy):
            raise TypeError("entry_policy must be a SearchEntryPolicy")
        if not self.observation_waypoints_xyz_m or not self.route_waypoints_xyz_m:
            raise RegionCompilationError("compiled SEARCH must contain waypoints")


class RegionCompiler:
    """Resolve a region, generate coverage points and apply an entry policy."""

    def __init__(
        self,
        resolver: SpatialResolver | None = None,
        *,
        search_runtime_capabilities: SearchRuntimeCapabilities | None = None,
    ) -> None:
        if resolver is not None and not isinstance(resolver, SpatialResolver):
            raise TypeError("resolver must be a SpatialResolver or None")
        if search_runtime_capabilities is None:
            search_runtime_capabilities = SearchRuntimeCapabilities()
        if not isinstance(search_runtime_capabilities, SearchRuntimeCapabilities):
            raise TypeError(
                "search_runtime_capabilities must be a "
                "SearchRuntimeCapabilities or None"
            )
        self._resolver = resolver
        self._search_runtime_capabilities = search_runtime_capabilities

    def compile(
        self,
        *,
        region: RegionSpec,
        strategy: SearchStrategySpec,
        entry_policy: SearchEntryPolicy,
        current_uav_xyz_m: Sequence[float],
        search_altitude_m: float,
        user_anchor_xyz_m: Sequence[float] | None = None,
        model_selected_entry_xyz_m: Sequence[float] | None = None,
    ) -> CompiledSearchGeometry:
        if not isinstance(entry_policy, SearchEntryPolicy):
            try:
                entry_policy = SearchEntryPolicy(entry_policy)
            except (TypeError, ValueError) as exc:
                raise RegionCompilationError("unknown SEARCH entry policy") from exc
        current = _point(current_uav_xyz_m, "current_uav_xyz_m")
        altitude = _positive(search_altitude_m, "search_altitude_m")
        original_frame = None if isinstance(region, RelationalRegion) else region.frame
        resolved = self._resolved_region(region)

        effective_strategy = strategy
        if strategy.model_waypoints_xyz_m and original_frame is not CoordinateFrame.WORLD_ENU:
            if self._resolver is None or original_frame is None:
                raise RegionCompilationError("relative MODEL_WAYPOINTS require a SpatialResolver")
            effective_strategy = SearchStrategySpec(
                kind=strategy.kind,
                spacing_m=strategy.spacing_m,
                max_viewpoints=strategy.max_viewpoints,
                model_waypoints_xyz_m=tuple(
                    self._resolver.resolve_point(original_frame, point)
                    for point in strategy.model_waypoints_xyz_m
                ),
                random_seed=strategy.random_seed,
            )
        user_anchor = self._resolve_optional_point(user_anchor_xyz_m, original_frame, "user_anchor_xyz_m")
        model_entry = self._resolve_optional_point(model_selected_entry_xyz_m, original_frame, "model_selected_entry_xyz_m")
        if effective_strategy.kind is SearchStrategyType.ADAPTIVE_NEXT_BEST_VIEW:
            if not self._search_runtime_capabilities.adaptive_next_best_view:
                raise RegionCompilationError(
                    "ADAPTIVE_NEXT_BEST_VIEW requires a negotiated runtime "
                    "next-best-view provider"
                )
            # The first observation point is deterministic and uses the entry
            # policy. Every later point is requested only after a fresh frame
            # has been acquired and scanned by SearchSkill.
            entry = self._select_adaptive_entry(
                resolved,
                entry_policy,
                current,
                altitude,
                user_anchor,
                model_entry,
            )
            observations = (entry,)
            route = observations
        else:
            try:
                observations = generate_search_waypoints(
                    resolved,
                    effective_strategy,
                    altitude_m=altitude,
                )
            except SearchStrategyError as exc:
                raise RegionCompilationError(str(exc)) from exc
            entry = self._select_entry(
                resolved,
                observations,
                entry_policy,
                current,
                altitude,
                user_anchor,
                model_entry,
            )
            route = (
                observations
                if _same_xy(entry, observations[0])
                else (entry, *observations)
            )
        return CompiledSearchGeometry(
            region=resolved,
            strategy=effective_strategy,
            entry_policy=entry_policy,
            entry_point_xyz_m=entry,
            observation_waypoints_xyz_m=observations,
            route_waypoints_xyz_m=route,
        )

    @staticmethod
    def validate_adaptive_waypoint(
        region: RegionSpec,
        point_xyz_m: Sequence[float],
        *,
        search_altitude_m: float,
    ) -> Point3:
        """Validate one provider-selected WORLD_ENU observation point.

        Coordinates are rejected rather than clamped. This keeps a learned or
        Qwen provider on the same explicit trusted boundary as initial V3
        geometry.
        """

        point = _point(point_xyz_m, "adaptive waypoint")
        altitude = _positive(search_altitude_m, "search_altitude_m")
        if isinstance(region, RelationalRegion) or region.frame is not CoordinateFrame.WORLD_ENU:
            raise RegionCompilationError(
                "adaptive waypoint validation requires a resolved WORLD_ENU region"
            )
        if abs(point[2] - altitude) > 1e-6:
            raise RegionCompilationError(
                "adaptive waypoint altitude must equal search_altitude_m"
            )
        if not point_inside_region(region, point):
            raise RegionCompilationError("adaptive waypoint lies outside the region")
        return point

    def rectangle_anchor(
        self,
        region: RectangleRegion,
        anchor: RectangleAnchor,
        *,
        search_altitude_m: float,
    ) -> Point3:
        resolved = self._resolved_region(region)
        if not isinstance(resolved, RectangleRegion):  # pragma: no cover
            raise TypeError("region must resolve to RectangleRegion")
        return rectangle_anchor_point(
            resolved,
            anchor,
            altitude_m=_positive(search_altitude_m, "search_altitude_m"),
        )

    def _resolved_region(self, region: RegionSpec) -> RegionSpec:
        if isinstance(region, RelationalRegion) or region.frame is not CoordinateFrame.WORLD_ENU:
            if self._resolver is None:
                raise RegionCompilationError(
                    "relative or unresolved region requires a SpatialResolver"
                )
            try:
                return self._resolver.resolve_region(region)
            except (TypeError, ValueError) as exc:
                raise RegionCompilationError(str(exc)) from exc
        return region

    def _resolve_optional_point(
        self,
        value: Sequence[float] | None,
        frame: CoordinateFrame | None,
        name: str,
    ) -> Point3 | None:
        if value is None:
            return None
        point = _point(value, name)
        if frame in {None, CoordinateFrame.WORLD_ENU}:
            return point
        if self._resolver is None:
            raise RegionCompilationError(f"relative {name} requires a SpatialResolver")
        return self._resolver.resolve_point(frame, point)

    @staticmethod
    def _select_entry(
        region: RegionSpec,
        observations: tuple[Point3, ...],
        policy: SearchEntryPolicy,
        current: Point3,
        altitude: float,
        user_anchor: Point3 | None,
        model_entry: Point3 | None,
    ) -> Point3:
        current_at_altitude = (current[0], current[1], altitude)
        if policy is SearchEntryPolicy.START_IN_PLACE_IF_INSIDE:
            if point_inside_region(region, current):
                # Preserve current XY: do not insert a west-edge detour.
                return current_at_altitude
            return min(observations, key=lambda point: _distance_xy(point, current))
        if policy is SearchEntryPolicy.NEAREST_POINT:
            return min(observations, key=lambda point: _distance_xy(point, current))
        if policy is SearchEntryPolicy.NEAREST_BOUNDARY:
            return nearest_boundary_point(region, current, altitude)
        if policy is SearchEntryPolicy.CENTER:
            return region_center(region, altitude)
        if policy is SearchEntryPolicy.USER_ANCHOR:
            return RegionCompiler._validated_anchor(region, user_anchor, "USER_ANCHOR", altitude)
        if policy is SearchEntryPolicy.MODEL_SELECTED:
            return RegionCompiler._validated_anchor(region, model_entry, "MODEL_SELECTED", altitude)
        raise RegionCompilationError(f"unsupported entry policy: {policy.value}")

    @staticmethod
    def _select_adaptive_entry(
        region: RegionSpec,
        policy: SearchEntryPolicy,
        current: Point3,
        altitude: float,
        user_anchor: Point3 | None,
        model_entry: Point3 | None,
    ) -> Point3:
        current_at_altitude = (current[0], current[1], altitude)
        if policy is SearchEntryPolicy.START_IN_PLACE_IF_INSIDE:
            if point_inside_region(region, current):
                return current_at_altitude
            return nearest_boundary_point(region, current, altitude)
        if policy in {
            SearchEntryPolicy.NEAREST_POINT,
            SearchEntryPolicy.NEAREST_BOUNDARY,
        }:
            if point_inside_region(region, current):
                return current_at_altitude
            return nearest_boundary_point(region, current, altitude)
        if policy is SearchEntryPolicy.CENTER:
            return region_center(region, altitude)
        if policy is SearchEntryPolicy.USER_ANCHOR:
            return RegionCompiler._validated_anchor(
                region,
                user_anchor,
                "USER_ANCHOR",
                altitude,
            )
        if policy is SearchEntryPolicy.MODEL_SELECTED:
            return RegionCompiler._validated_anchor(
                region,
                model_entry,
                "MODEL_SELECTED",
                altitude,
            )
        raise RegionCompilationError(f"unsupported entry policy: {policy.value}")

    @staticmethod
    def _validated_anchor(
        region: RegionSpec,
        point: Point3 | None,
        policy: str,
        altitude: float,
    ) -> Point3:
        if point is None:
            raise RegionCompilationError(f"{policy} requires an explicit entry point")
        if not point_inside_region(region, point):
            raise RegionCompilationError(f"{policy} entry point lies outside the region")
        return (point[0], point[1], altitude)


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise RegionCompilationError(f"{name} must be finite and greater than zero")
    return result


def _distance_xy(a: Point3, b: Point3) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _same_xy(a: Point3, b: Point3) -> bool:
    return _distance_xy(a, b) <= 1e-9


__all__ = ["CompiledSearchGeometry", "RegionCompilationError", "RegionCompiler"]
