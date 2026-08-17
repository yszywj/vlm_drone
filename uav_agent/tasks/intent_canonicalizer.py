"""Deterministic expansion of Gold and predicted mission defaults."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real

from planner.schemas import MissionIntent

from .schemas import GoldPlannerSpec, PlannerWorldCase
from .target_ontology import TargetOntology


DEFAULT_NUMERIC_TOLERANCE = 1e-6


class IntentCanonicalizationError(ValueError):
    """Raised when trusted Gold/world inputs are inconsistent."""


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return number


@dataclass(frozen=True, slots=True)
class EffectiveMissionIntent:
    """A task after defaults and exact ontology aliases have been resolved."""

    target_concept_id: str | None
    target_description: str
    search_region: str
    track_duration_s: float
    landing_zone: str
    takeoff_altitude_m: float

    def __post_init__(self) -> None:
        if self.target_concept_id is not None:
            if not isinstance(self.target_concept_id, str) or not self.target_concept_id:
                raise ValueError("target_concept_id must be a non-empty string or None")
        for field_name in ("target_description", "search_region", "landing_zone"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        duration = _finite(self.track_duration_s, "track_duration_s")
        if duration <= 0.0:
            raise ValueError("track_duration_s must be greater than zero")
        object.__setattr__(self, "track_duration_s", duration)
        object.__setattr__(
            self,
            "takeoff_altitude_m",
            _finite(self.takeoff_altitude_m, "takeoff_altitude_m"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_concept_id": self.target_concept_id,
            "target_description": self.target_description,
            "search_region": self.search_region,
            "track_duration_s": self.track_duration_s,
            "landing_zone": self.landing_zone,
            "takeoff_altitude_m": self.takeoff_altitude_m,
        }


def _validate_world_choice(world: PlannerWorldCase, name: str, field_name: str) -> None:
    choices = world.search_regions if field_name == "search_region" else world.landing_zones
    if name not in choices:
        raise IntentCanonicalizationError(
            f"Gold {field_name} {name!r} is not available in world {world.context_id!r}"
        )


def canonicalize_gold(
    gold: GoldPlannerSpec,
    world: PlannerWorldCase,
    ontology: TargetOntology,
) -> EffectiveMissionIntent:
    """Convert trusted Gold into its effective task without consulting a model."""

    if not isinstance(gold, GoldPlannerSpec):
        raise TypeError("gold must be a GoldPlannerSpec")
    if not isinstance(world, PlannerWorldCase):
        raise TypeError("world must be a PlannerWorldCase")
    if not isinstance(ontology, TargetOntology):
        raise TypeError("ontology must be a TargetOntology")
    ontology.validate_gold_spec(gold)
    _validate_world_choice(world, gold.search_region, "search_region")
    _validate_world_choice(world, gold.landing_zone, "landing_zone")
    if (
        "track_duration_s" not in gold.explicit_fields
        and not finite_numbers_match(
            gold.track_duration_s,
            world.default_track_duration_s,
        )
    ):
        raise IntentCanonicalizationError(
            "an omitted Gold track_duration_s must equal the trusted world default"
        )

    duration = (
        gold.track_duration_s
        if "track_duration_s" in gold.explicit_fields
        else world.default_track_duration_s
    )
    altitude = (
        gold.takeoff_altitude_m
        if gold.takeoff_altitude_m is not None
        else world.default_takeoff_altitude_m
    )
    concept = ontology.require_concept(gold.target_concept_id)
    return EffectiveMissionIntent(
        target_concept_id=concept.concept_id,
        target_description=concept.canonical_description,
        search_region=gold.search_region,
        track_duration_s=duration,
        landing_zone=gold.landing_zone,
        takeoff_altitude_m=altitude,
    )


def canonicalize_prediction(
    predicted: MissionIntent,
    world: PlannerWorldCase,
    ontology: TargetOntology,
) -> EffectiveMissionIntent:
    """Expand planner defaults and resolve only explicitly registered aliases.

    An unknown description is deliberately retained with ``concept_id=None``;
    no nearest-neighbour, embedding, or language-model guess is attempted.
    Unknown world choices are likewise retained so the Judge can report the
    corresponding field-level mismatch.
    """

    if not isinstance(predicted, MissionIntent):
        raise TypeError("predicted must be a MissionIntent")
    if not isinstance(world, PlannerWorldCase):
        raise TypeError("world must be a PlannerWorldCase")
    if not isinstance(ontology, TargetOntology):
        raise TypeError("ontology must be a TargetOntology")

    concept = ontology.resolve_description(predicted.target_description)
    altitude = (
        predicted.takeoff_altitude_m
        if predicted.takeoff_altitude_m is not None
        else world.default_takeoff_altitude_m
    )
    return EffectiveMissionIntent(
        target_concept_id=concept.concept_id if concept is not None else None,
        target_description=(
            concept.canonical_description
            if concept is not None
            else predicted.target_description
        ),
        search_region=predicted.search_region,
        track_duration_s=predicted.track_duration_s,
        landing_zone=predicted.landing_zone,
        takeoff_altitude_m=altitude,
    )


def finite_numbers_match(
    left: object,
    right: object,
    *,
    tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> bool:
    """Compare two finite scalar values using only a small absolute tolerance."""

    tolerance = _finite(tolerance, "tolerance")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    try:
        left_number = _finite(left, "left")
        right_number = _finite(right, "right")
    except (TypeError, ValueError):
        return False
    return isclose(left_number, right_number, rel_tol=0.0, abs_tol=tolerance)


__all__ = [
    "DEFAULT_NUMERIC_TOLERANCE",
    "EffectiveMissionIntent",
    "IntentCanonicalizationError",
    "canonicalize_gold",
    "canonicalize_prediction",
    "finite_numbers_match",
]
