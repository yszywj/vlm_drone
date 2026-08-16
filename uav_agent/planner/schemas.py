"""Pure-Python schemas at the boundary between planning and Skill execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType

from skills.manager import TaskPlan


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _positive_number(value: object, field_name: str) -> float:
    normalized = _finite_number(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return normalized


def _finite_vector(
    value: object,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ValueError(f"{field_name} must contain exactly {length} numbers")
    return tuple(
        _finite_number(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class SearchRegionSpec:
    """Named search geometry supplied by trusted world configuration."""

    name: str
    center_xyz_m: tuple[float, float, float]
    radius_m: float
    approach_xyz_m: tuple[float, float, float]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(
            self,
            "center_xyz_m",
            _finite_vector(self.center_xyz_m, 3, "center_xyz_m"),
        )
        object.__setattr__(self, "radius_m", _positive_number(self.radius_m, "radius_m"))
        object.__setattr__(
            self,
            "approach_xyz_m",
            _finite_vector(self.approach_xyz_m, 3, "approach_xyz_m"),
        )
        object.__setattr__(
            self,
            "description",
            _string(self.description, "description"),
        )


@dataclass(frozen=True, slots=True)
class LandingZoneSpec:
    """Named landing geometry supplied by trusted world configuration."""

    name: str
    position_xy_m: tuple[float, float]
    ground_altitude_m: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(
            self,
            "position_xy_m",
            _finite_vector(self.position_xy_m, 2, "position_xy_m"),
        )
        object.__setattr__(
            self,
            "ground_altitude_m",
            _finite_number(self.ground_altitude_m, "ground_altitude_m"),
        )
        object.__setattr__(
            self,
            "description",
            _string(self.description, "description"),
        )


@dataclass(frozen=True, slots=True)
class PlannerWorldContext:
    """Trusted, non-oracle world facts available to a mission planner."""

    scene_min_xyz_m: tuple[float, float, float]
    scene_max_xyz_m: tuple[float, float, float]
    initial_uav_xyz_m: tuple[float, float, float]
    search_regions: Mapping[str, SearchRegionSpec]
    landing_zones: Mapping[str, LandingZoneSpec]
    default_takeoff_altitude_m: float
    default_track_duration_s: float
    search_timeout_s: float
    goto_timeout_s: float = 120.0
    land_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        scene_min = _finite_vector(self.scene_min_xyz_m, 3, "scene_min_xyz_m")
        scene_max = _finite_vector(self.scene_max_xyz_m, 3, "scene_max_xyz_m")
        if any(lower >= upper for lower, upper in zip(scene_min, scene_max)):
            raise ValueError(
                "scene_min_xyz_m must be strictly less than scene_max_xyz_m "
                "on every axis"
            )

        object.__setattr__(self, "scene_min_xyz_m", scene_min)
        object.__setattr__(self, "scene_max_xyz_m", scene_max)
        object.__setattr__(
            self,
            "initial_uav_xyz_m",
            _finite_vector(self.initial_uav_xyz_m, 3, "initial_uav_xyz_m"),
        )
        object.__setattr__(
            self,
            "search_regions",
            _readonly_spec_mapping(
                self.search_regions,
                SearchRegionSpec,
                "search_regions",
            ),
        )
        object.__setattr__(
            self,
            "landing_zones",
            _readonly_spec_mapping(
                self.landing_zones,
                LandingZoneSpec,
                "landing_zones",
            ),
        )
        object.__setattr__(
            self,
            "default_takeoff_altitude_m",
            _positive_number(
                self.default_takeoff_altitude_m,
                "default_takeoff_altitude_m",
            ),
        )
        object.__setattr__(
            self,
            "default_track_duration_s",
            _positive_number(
                self.default_track_duration_s,
                "default_track_duration_s",
            ),
        )
        object.__setattr__(
            self,
            "search_timeout_s",
            _positive_number(self.search_timeout_s, "search_timeout_s"),
        )
        object.__setattr__(
            self,
            "goto_timeout_s",
            _positive_number(self.goto_timeout_s, "goto_timeout_s"),
        )
        object.__setattr__(
            self,
            "land_timeout_s",
            _positive_number(self.land_timeout_s, "land_timeout_s"),
        )


@dataclass(frozen=True, slots=True)
class MissionIntent:
    """Model-facing task intent without low-level coordinates or timeouts."""

    target_description: str
    search_region: str
    track_duration_s: float
    landing_zone: str
    takeoff_altitude_m: float | None = None

    _REQUIRED_FIELDS = frozenset(
        {
            "target_description",
            "search_region",
            "track_duration_s",
            "landing_zone",
        }
    )
    _OPTIONAL_FIELDS = frozenset({"takeoff_altitude_m"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_description",
            _non_empty_string(self.target_description, "target_description"),
        )
        object.__setattr__(
            self,
            "search_region",
            _non_empty_string(self.search_region, "search_region"),
        )
        object.__setattr__(
            self,
            "track_duration_s",
            _positive_number(self.track_duration_s, "track_duration_s"),
        )
        object.__setattr__(
            self,
            "landing_zone",
            _non_empty_string(self.landing_zone, "landing_zone"),
        )
        if self.takeoff_altitude_m is not None:
            object.__setattr__(
                self,
                "takeoff_altitude_m",
                _finite_number(self.takeoff_altitude_m, "takeoff_altitude_m"),
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MissionIntent:
        """Parse an exact, JSON-like planner response without ignoring fields."""

        if not isinstance(data, Mapping):
            raise TypeError("MissionIntent input must be a mapping")
        if any(not isinstance(key, str) for key in data):
            raise TypeError("MissionIntent field names must be strings")

        keys = frozenset(data)
        allowed = cls._REQUIRED_FIELDS | cls._OPTIONAL_FIELDS
        unknown = keys - allowed
        if unknown:
            raise ValueError(
                "MissionIntent contains unknown fields: " + ", ".join(sorted(unknown))
            )
        missing = cls._REQUIRED_FIELDS - keys
        if missing:
            raise ValueError(
                "MissionIntent is missing required fields: "
                + ", ".join(sorted(missing))
            )

        return cls(
            target_description=data["target_description"],
            search_region=data["search_region"],
            track_duration_s=data["track_duration_s"],
            landing_zone=data["landing_zone"],
            takeoff_altitude_m=data.get("takeoff_altitude_m"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible mapping."""

        return {
            "target_description": self.target_description,
            "search_region": self.search_region,
            "track_duration_s": self.track_duration_s,
            "landing_zone": self.landing_zone,
            "takeoff_altitude_m": self.takeoff_altitude_m,
        }


@dataclass(frozen=True, slots=True)
class PlannerRequest:
    instruction: str
    world_context: PlannerWorldContext

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instruction",
            _non_empty_string(self.instruction, "instruction"),
        )
        if not isinstance(self.world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")


@dataclass(frozen=True, slots=True)
class CompiledMission:
    """Validated high-level intent paired with its executable Skill plan."""

    intent: MissionIntent
    task_plan: TaskPlan
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.intent, MissionIntent):
            raise TypeError("intent must be a MissionIntent")
        if not isinstance(self.task_plan, TaskPlan):
            raise TypeError("task_plan must be a skills.manager.TaskPlan")
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        if self.source not in {"scripted", "llm"}:
            raise ValueError("source must be either 'scripted' or 'llm'")


def _readonly_spec_mapping(
    value: object,
    expected_type: type[SearchRegionSpec] | type[LandingZoneSpec],
    field_name: str,
) -> Mapping[str, SearchRegionSpec] | Mapping[str, LandingZoneSpec]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    snapshot: dict[str, SearchRegionSpec] | dict[str, LandingZoneSpec] = {}
    for key, spec in value.items():
        normalized_key = _non_empty_string(key, f"{field_name} key")
        if not isinstance(spec, expected_type):
            raise TypeError(
                f"{field_name}[{normalized_key!r}] must be a "
                f"{expected_type.__name__}"
            )
        snapshot[normalized_key] = spec
    return MappingProxyType(snapshot)
