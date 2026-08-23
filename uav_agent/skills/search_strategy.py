"""Model-visible SEARCH V3 entry and coverage strategy contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from planner.spatial import RegionSpec


class SearchStrategyError(ValueError):
    """Raised for an invalid or unsupported SEARCH V3 strategy."""


class SearchEntryPolicy(str, Enum):
    START_IN_PLACE_IF_INSIDE = "START_IN_PLACE_IF_INSIDE"
    NEAREST_POINT = "NEAREST_POINT"
    NEAREST_BOUNDARY = "NEAREST_BOUNDARY"
    CENTER = "CENTER"
    USER_ANCHOR = "USER_ANCHOR"
    MODEL_SELECTED = "MODEL_SELECTED"


class SearchStrategyType(str, Enum):
    # Exact six-point circle baseline retained for experiment comparability.
    PERIMETER_V1 = "PERIMETER_V1"
    PERIMETER = "PERIMETER"
    LAWNMOWER = "LAWNMOWER"
    SPIRAL_IN = "SPIRAL_IN"
    SPIRAL_OUT = "SPIRAL_OUT"
    SECTOR_SWEEP = "SECTOR_SWEEP"
    CORRIDOR_FOLLOW = "CORRIDOR_FOLLOW"
    RANDOM_COVERAGE = "RANDOM_COVERAGE"
    MODEL_WAYPOINTS = "MODEL_WAYPOINTS"
    ADAPTIVE_NEXT_BEST_VIEW = "ADAPTIVE_NEXT_BEST_VIEW"


# Concise compatibility spelling for callers that prefer an enum noun.
SearchStrategy = SearchStrategyType


@dataclass(frozen=True, slots=True)
class SearchRuntimeCapabilities:
    """Runtime SEARCH features that may safely be advertised to a planner.

    Static geometry strategies are always executable. Adaptive next-best-view
    planning is opt-in because it requires a runtime provider that consumes a
    fresh observation after every completed macro viewpoint.
    """

    adaptive_next_best_view: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.adaptive_next_best_view, bool):
            raise TypeError("adaptive_next_best_view must be a bool")

    @property
    def supported_strategies(self) -> tuple[SearchStrategyType, ...]:
        return tuple(
            strategy
            for strategy in SearchStrategyType
            if (
                strategy is not SearchStrategyType.ADAPTIVE_NEXT_BEST_VIEW
                or self.adaptive_next_best_view
            )
        )

    def supports(self, strategy: SearchStrategyType | str) -> bool:
        try:
            normalized = (
                strategy
                if isinstance(strategy, SearchStrategyType)
                else SearchStrategyType(strategy)
            )
        except (TypeError, ValueError):
            return False
        return normalized in self.supported_strategies

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "adaptive_next_best_view": self.adaptive_next_best_view,
            "supported_search_strategies": [
                strategy.value for strategy in self.supported_strategies
            ],
        }


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise SearchStrategyError(f"{name} must be finite and greater than zero")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point(value: object, name: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise SearchStrategyError(f"{name} must contain exactly three numbers")
    result: list[float] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, Real):
            raise TypeError(f"{name}[{index}] must be a finite number")
        number = float(component)
        if not isfinite(number):
            raise SearchStrategyError(f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class NextBestViewRequest:
    """One bounded macro-viewpoint request issued after a fresh frame.

    Only prompt-safe frame and UAV pose fields are projected. In particular,
    no ``oracle_target_*`` field crosses this provider boundary. Providers
    choose one WORLD_ENU observation point or return ``None`` to declare that
    no useful next view remains; they never emit controller-rate commands.
    """

    region: "RegionSpec"
    target_description: str
    observation_timestamp_s: float
    uav_position_xyz_m: tuple[float, float, float]
    uav_yaw_rad: float
    camera_rgb: np.ndarray
    camera_position_m: tuple[float, float, float] | None
    camera_orientation_wxyz: tuple[float, float, float, float] | None
    visited_viewpoints_xyz_m: tuple[tuple[float, float, float], ...]
    coverage_ratio: float
    max_viewpoints: int
    search_altitude_m: float

    def __post_init__(self) -> None:
        from planner.spatial import (
            CircleRegion,
            CorridorRegion,
            PolygonRegion,
            RectangleRegion,
            SectorRegion,
        )
        if not isinstance(
            self.region,
            (
                CircleRegion,
                RectangleRegion,
                SectorRegion,
                PolygonRegion,
                CorridorRegion,
            ),
        ):
            raise TypeError("region must be resolved executable RegionSpec geometry")
        if (
            not isinstance(self.target_description, str)
            or not self.target_description.strip()
        ):
            raise ValueError("target_description must be a non-empty string")
        object.__setattr__(
            self,
            "target_description",
            self.target_description.strip(),
        )
        timestamp = _finite(self.observation_timestamp_s, "observation_timestamp_s")
        if timestamp < 0.0:
            raise ValueError("observation_timestamp_s must be non-negative")
        object.__setattr__(self, "observation_timestamp_s", timestamp)
        object.__setattr__(
            self,
            "uav_position_xyz_m",
            _point(self.uav_position_xyz_m, "uav_position_xyz_m"),
        )
        object.__setattr__(
            self,
            "uav_yaw_rad",
            _finite(self.uav_yaw_rad, "uav_yaw_rad"),
        )
        if (
            not isinstance(self.camera_rgb, np.ndarray)
            or self.camera_rgb.ndim != 3
            or self.camera_rgb.shape[0] <= 0
            or self.camera_rgb.shape[1] <= 0
            or self.camera_rgb.shape[2] != 3
        ):
            raise ValueError("camera_rgb must have shape (height, width, 3)")
        owned_rgb = np.array(self.camera_rgb, copy=True)
        owned_rgb.setflags(write=False)
        object.__setattr__(self, "camera_rgb", owned_rgb)
        if (self.camera_position_m is None) != (
            self.camera_orientation_wxyz is None
        ):
            raise ValueError(
                "camera position and orientation must be provided together"
            )
        if self.camera_position_m is not None:
            object.__setattr__(
                self,
                "camera_position_m",
                _point(self.camera_position_m, "camera_position_m"),
            )
            orientation = self.camera_orientation_wxyz
            assert orientation is not None
            if len(orientation) != 4:
                raise ValueError(
                    "camera_orientation_wxyz must contain exactly four numbers"
                )
            object.__setattr__(
                self,
                "camera_orientation_wxyz",
                tuple(
                    _finite(value, f"camera_orientation_wxyz[{index}]")
                    for index, value in enumerate(orientation)
                ),
            )
        if not isinstance(self.visited_viewpoints_xyz_m, tuple):
            object.__setattr__(
                self,
                "visited_viewpoints_xyz_m",
                tuple(self.visited_viewpoints_xyz_m),
            )
        visited = tuple(
            _point(point, f"visited_viewpoints_xyz_m[{index}]")
            for index, point in enumerate(self.visited_viewpoints_xyz_m)
        )
        object.__setattr__(self, "visited_viewpoints_xyz_m", visited)
        if isinstance(self.coverage_ratio, bool) or not isinstance(
            self.coverage_ratio,
            Real,
        ):
            raise TypeError("coverage_ratio must be a finite number")
        coverage = float(self.coverage_ratio)
        if not isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage_ratio must be between zero and one")
        object.__setattr__(self, "coverage_ratio", coverage)
        if isinstance(self.max_viewpoints, bool) or not isinstance(
            self.max_viewpoints,
            int,
        ):
            raise TypeError("max_viewpoints must be an integer")
        if not 1 <= self.max_viewpoints <= 128:
            raise ValueError("max_viewpoints must be between 1 and 128")
        if len(visited) > self.max_viewpoints:
            raise ValueError("visited viewpoints exceed max_viewpoints")
        object.__setattr__(
            self,
            "search_altitude_m",
            _positive(self.search_altitude_m, "search_altitude_m"),
        )


@dataclass(frozen=True, slots=True)
class NextBestViewPollResult:
    """One non-blocking provider poll.

    ``completed=False`` means model work is still in flight.  A completed
    result with ``viewpoint_xyz_m=None`` is the provider's explicit coverage
    exhaustion decision.  This small value object avoids using ``None`` for
    both "not ready" and "no useful view remains".
    """

    completed: bool
    viewpoint_xyz_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a boolean")
        if not self.completed and self.viewpoint_xyz_m is not None:
            raise ValueError("an incomplete poll cannot contain a viewpoint")
        if self.viewpoint_xyz_m is not None:
            object.__setattr__(
                self,
                "viewpoint_xyz_m",
                _point(self.viewpoint_xyz_m, "viewpoint_xyz_m"),
            )


@runtime_checkable
class NextBestViewProvider(Protocol):
    """Runtime provider for one adaptive SEARCH macro viewpoint at a time."""

    def next_best_view(
        self,
        request: NextBestViewRequest,
    ) -> Sequence[float] | None:
        """Return one WORLD_ENU point, or ``None`` when coverage is exhausted."""


@runtime_checkable
class AsyncNextBestViewProvider(Protocol):
    """Non-blocking macro-view provider used by network-backed planners.

    Submission occurs once after a completed viewpoint scan.  SearchSkill
    then polls at Camera cadence while holding position; it never waits for an
    HTTP response in the controller thread.
    """

    def submit_next_best_view(self, request: NextBestViewRequest) -> None:
        """Submit one fresh-frame request without waiting for model I/O."""

    def poll_next_best_view(self) -> NextBestViewPollResult:
        """Return immediately with pending, a macro point, or exhaustion."""


@dataclass(frozen=True, slots=True)
class SearchStrategySpec:
    kind: SearchStrategyType
    spacing_m: float = 5.0
    max_viewpoints: int = 32
    model_waypoints_xyz_m: tuple[tuple[float, float, float], ...] = ()
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SearchStrategyType):
            try:
                object.__setattr__(self, "kind", SearchStrategyType(self.kind))
            except (TypeError, ValueError) as exc:
                raise SearchStrategyError("unknown SEARCH strategy") from exc
        object.__setattr__(self, "spacing_m", _positive(self.spacing_m, "spacing_m"))
        if isinstance(self.max_viewpoints, bool) or not isinstance(self.max_viewpoints, int):
            raise TypeError("max_viewpoints must be an integer")
        if not 1 <= self.max_viewpoints <= 128:
            raise SearchStrategyError("max_viewpoints must be between 1 and 128")
        if (
            isinstance(self.model_waypoints_xyz_m, (str, bytes))
            or not isinstance(self.model_waypoints_xyz_m, Sequence)
        ):
            raise TypeError("model_waypoints_xyz_m must be an array")
        points = tuple(
            _point(item, f"model_waypoints_xyz_m[{index}]")
            for index, item in enumerate(self.model_waypoints_xyz_m)
        )
        if len(points) > self.max_viewpoints:
            raise SearchStrategyError("model waypoints exceed max_viewpoints")
        if self.kind is SearchStrategyType.MODEL_WAYPOINTS and not points:
            raise SearchStrategyError("MODEL_WAYPOINTS requires at least one waypoint")
        if self.kind is not SearchStrategyType.MODEL_WAYPOINTS and points:
            raise SearchStrategyError(
                "model_waypoints_xyz_m is only allowed for MODEL_WAYPOINTS"
            )
        object.__setattr__(self, "model_waypoints_xyz_m", points)
        if self.random_seed is not None:
            if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
                raise TypeError("random_seed must be an integer or None")
            if not -(2**31) <= self.random_seed < 2**31:
                raise SearchStrategyError("random_seed must fit a signed 32-bit integer")
        if self.kind is not SearchStrategyType.RANDOM_COVERAGE and self.random_seed is not None:
            raise SearchStrategyError("random_seed is only allowed for RANDOM_COVERAGE")

    @classmethod
    def from_dict(cls, value: object) -> SearchStrategySpec:
        if not isinstance(value, Mapping):
            raise TypeError("SearchStrategySpec must be an object")
        allowed = {
            "kind", "spacing_m", "max_viewpoints", "model_waypoints_xyz_m", "random_seed"
        }
        unknown = set(value) - allowed
        if unknown:
            raise SearchStrategyError(
                "SearchStrategySpec contains unknown fields: " + ", ".join(sorted(unknown))
            )
        if "kind" not in value:
            raise SearchStrategyError("SearchStrategySpec is missing field: kind")
        return cls(
            kind=value["kind"],  # type: ignore[arg-type]
            spacing_m=value.get("spacing_m", 5.0),  # type: ignore[arg-type]
            max_viewpoints=value.get("max_viewpoints", 32),  # type: ignore[arg-type]
            model_waypoints_xyz_m=value.get("model_waypoints_xyz_m", ()),  # type: ignore[arg-type]
            random_seed=value.get("random_seed"),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind.value,
            "spacing_m": self.spacing_m,
            "max_viewpoints": self.max_viewpoints,
        }
        if self.model_waypoints_xyz_m:
            result["model_waypoints_xyz_m"] = [
                list(point) for point in self.model_waypoints_xyz_m
            ]
        if self.random_seed is not None:
            result["random_seed"] = self.random_seed
        return result


__all__ = [
    "AsyncNextBestViewProvider",
    "NextBestViewProvider",
    "NextBestViewPollResult",
    "NextBestViewRequest",
    "SearchEntryPolicy",
    "SearchRuntimeCapabilities",
    "SearchStrategy",
    "SearchStrategyError",
    "SearchStrategySpec",
    "SearchStrategyType",
]
