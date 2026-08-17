"""Immutable, model-independent schemas for Gold planner tasks.

The classes in this module deliberately do not accept ``MissionIntent`` as an
input.  Gold specifications are authored or generated before a planner runs;
the only supported bridge points from Gold data toward runtime planning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import ClassVar


GOLD_INTENT_FIELDS = frozenset(
    {
        "target_description",
        "search_region",
        "track_duration_s",
        "landing_zone",
        "takeoff_altitude_m",
    }
)


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


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


def _finite_xyz(value: object, field_name: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ValueError(f"{field_name} must contain exactly three numbers")
    return (
        _finite_number(value[0], f"{field_name}[0]"),
        _finite_number(value[1], f"{field_name}[1]"),
        _finite_number(value[2], f"{field_name}[2]"),
    )


def _readonly_string_mapping(value: object, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    snapshot: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _non_empty_string(raw_key, f"{field_name} key")
        description = _non_empty_string(raw_value, f"{field_name}[{key!r}]")
        if key in snapshot:
            raise ValueError(f"{field_name} contains duplicate key {key!r}")
        snapshot[key] = description
    return MappingProxyType(snapshot)


@dataclass(frozen=True, slots=True)
class TargetConcept:
    """One closed-set semantic target, without instance identity or geometry."""

    concept_id: str
    category: str
    attributes: Mapping[str, str]
    canonical_description: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "concept_id", _non_empty_string(self.concept_id, "concept_id")
        )
        object.__setattr__(
            self, "category", _non_empty_string(self.category, "category")
        )
        if not isinstance(self.attributes, Mapping):
            raise TypeError("attributes must be a mapping")
        attributes: dict[str, str] = {}
        for raw_key, raw_value in self.attributes.items():
            key = _non_empty_string(raw_key, "attribute key")
            value = _non_empty_string(raw_value, f"attributes[{key!r}]")
            if key in attributes:
                raise ValueError(f"duplicate target attribute {key!r}")
            attributes[key] = value
        if not attributes:
            raise ValueError("attributes must contain at least one attribute")
        object.__setattr__(self, "attributes", MappingProxyType(attributes))
        object.__setattr__(
            self,
            "canonical_description",
            _non_empty_string(
                self.canonical_description,
                "canonical_description",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "category": self.category,
            "attributes": dict(self.attributes),
            "canonical_description": self.canonical_description,
        }


@dataclass(frozen=True, slots=True)
class GoldPlannerSpec:
    """Trusted expected planner intent fixed before any model invocation."""

    spec_id: str
    target_concept_id: str
    target_description: str
    search_region: str
    track_duration_s: float
    landing_zone: str
    takeoff_altitude_m: float | None
    explicit_fields: frozenset[str]

    _FIELDS: ClassVar[frozenset[str]] = GOLD_INTENT_FIELDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_id", _non_empty_string(self.spec_id, "spec_id"))
        object.__setattr__(
            self,
            "target_concept_id",
            _non_empty_string(self.target_concept_id, "target_concept_id"),
        )
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
                _positive_number(self.takeoff_altitude_m, "takeoff_altitude_m"),
            )

        if isinstance(self.explicit_fields, (str, bytes)):
            raise TypeError("explicit_fields must be an iterable of field names")
        try:
            explicit = frozenset(
                _non_empty_string(value, "explicit_fields item")
                for value in self.explicit_fields
            )
        except TypeError:
            raise TypeError(
                "explicit_fields must be an iterable of field names"
            ) from None
        unknown = explicit - self._FIELDS
        if unknown:
            raise ValueError(
                "explicit_fields contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if "takeoff_altitude_m" in explicit and self.takeoff_altitude_m is None:
            raise ValueError(
                "an explicitly specified takeoff_altitude_m cannot be None"
            )
        if "takeoff_altitude_m" not in explicit and self.takeoff_altitude_m is not None:
            raise ValueError(
                "takeoff_altitude_m must be None when the field was not explicit"
            )
        object.__setattr__(self, "explicit_fields", explicit)

    def to_expected_intent(self):
        """Convert Gold data to the only permitted runtime-facing direction.

        The local import keeps this module independent of planner execution and
        avoids exposing any inverse ``MissionIntent -> GoldPlannerSpec`` API.
        """

        from planner.schemas import MissionIntent

        return MissionIntent(
            target_description=self.target_description,
            search_region=self.search_region,
            track_duration_s=self.track_duration_s,
            landing_zone=self.landing_zone,
            takeoff_altitude_m=self.takeoff_altitude_m,
        )

    # A descriptive alias used by dataset code.
    expected_intent = to_expected_intent

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "target_concept_id": self.target_concept_id,
            "target_description": self.target_description,
            "search_region": self.search_region,
            "track_duration_s": self.track_duration_s,
            "landing_zone": self.landing_zone,
            "takeoff_altitude_m": self.takeoff_altitude_m,
            "explicit_fields": sorted(self.explicit_fields),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "GoldPlannerSpec":
        if not isinstance(data, Mapping):
            raise TypeError("GoldPlannerSpec input must be a mapping")
        required = {
            "spec_id",
            "target_concept_id",
            "target_description",
            "search_region",
            "track_duration_s",
            "landing_zone",
            "takeoff_altitude_m",
            "explicit_fields",
        }
        keys = set(data)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("GoldPlannerSpec field names must be strings")
        unknown = keys - required
        missing = required - keys
        if unknown:
            raise ValueError(
                "GoldPlannerSpec contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if missing:
            raise ValueError(
                "GoldPlannerSpec is missing required fields: "
                + ", ".join(sorted(missing))
            )
        return cls(**{key: data[key] for key in required})


@dataclass(frozen=True, slots=True)
class PlannerWorldCase:
    """Dataset-visible world facts with no target instance or oracle state."""

    context_id: str
    search_regions: Mapping[str, str]
    landing_zones: Mapping[str, str]
    default_takeoff_altitude_m: float
    default_track_duration_s: float
    scene_min_xyz_m: tuple[float, float, float]
    scene_max_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "context_id", _non_empty_string(self.context_id, "context_id")
        )
        search_regions = _readonly_string_mapping(
            self.search_regions, "search_regions"
        )
        landing_zones = _readonly_string_mapping(
            self.landing_zones, "landing_zones"
        )
        if len(search_regions) < 2:
            raise ValueError("search_regions must contain at least two choices")
        if len(landing_zones) < 2:
            raise ValueError("landing_zones must contain at least two choices")
        object.__setattr__(self, "search_regions", search_regions)
        object.__setattr__(self, "landing_zones", landing_zones)
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
        lower = _finite_xyz(self.scene_min_xyz_m, "scene_min_xyz_m")
        upper = _finite_xyz(self.scene_max_xyz_m, "scene_max_xyz_m")
        if any(minimum >= maximum for minimum, maximum in zip(lower, upper)):
            raise ValueError(
                "scene_min_xyz_m must be strictly less than scene_max_xyz_m "
                "on every axis"
            )
        object.__setattr__(self, "scene_min_xyz_m", lower)
        object.__setattr__(self, "scene_max_xyz_m", upper)

    def to_dict(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "search_regions": dict(self.search_regions),
            "landing_zones": dict(self.landing_zones),
            "default_takeoff_altitude_m": self.default_takeoff_altitude_m,
            "default_track_duration_s": self.default_track_duration_s,
            "scene_min_xyz_m": list(self.scene_min_xyz_m),
            "scene_max_xyz_m": list(self.scene_max_xyz_m),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PlannerWorldCase":
        if not isinstance(data, Mapping):
            raise TypeError("PlannerWorldCase input must be a mapping")
        required = {
            "context_id",
            "search_regions",
            "landing_zones",
            "default_takeoff_altitude_m",
            "default_track_duration_s",
            "scene_min_xyz_m",
            "scene_max_xyz_m",
        }
        unknown = set(data) - required
        missing = required - set(data)
        if unknown:
            raise ValueError(
                "PlannerWorldCase contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if missing:
            raise ValueError(
                "PlannerWorldCase is missing required fields: "
                + ", ".join(sorted(missing))
            )
        return cls(**{key: data[key] for key in required})


__all__ = [
    "GOLD_INTENT_FIELDS",
    "GoldPlannerSpec",
    "PlannerWorldCase",
    "TargetConcept",
]
