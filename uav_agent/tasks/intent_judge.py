"""Field-level exact and semantic comparison against independent Gold data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from planner.schemas import MissionIntent

from .intent_canonicalizer import (
    DEFAULT_NUMERIC_TOLERANCE,
    canonicalize_gold,
    canonicalize_prediction,
    finite_numbers_match,
)
from .schemas import GoldPlannerSpec, PlannerWorldCase
from .target_ontology import TargetOntology


class IntentErrorCode(str, Enum):
    PLANNER_OUTPUT_INVALID = "PLANNER_OUTPUT_INVALID"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    SEARCH_REGION_MISMATCH = "SEARCH_REGION_MISMATCH"
    TRACK_DURATION_MISMATCH = "TRACK_DURATION_MISMATCH"
    LANDING_ZONE_MISMATCH = "LANDING_ZONE_MISMATCH"
    TAKEOFF_ALTITUDE_MISMATCH = "TAKEOFF_ALTITUDE_MISMATCH"
    UNKNOWN_TARGET_DESCRIPTION = "UNKNOWN_TARGET_DESCRIPTION"


@dataclass(frozen=True, slots=True)
class IntentJudgeResult:
    output_valid: bool
    exact_match: bool
    semantic_match: bool

    target_match: bool
    search_region_match: bool
    track_duration_match: bool
    landing_zone_match: bool
    takeoff_altitude_match: bool

    track_duration_error_s: float | None
    takeoff_altitude_error_m: float | None
    error_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "output_valid",
            "exact_match",
            "semantic_match",
            "target_match",
            "search_region_match",
            "track_duration_match",
            "landing_zone_match",
            "takeoff_altitude_match",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        for field_name in (
            "track_duration_error_s",
            "takeoff_altitude_error_m",
        ):
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{field_name} must be a finite number or None")
                normalized = float(value)
                if not isfinite(normalized) or normalized < 0.0:
                    raise ValueError(
                        f"{field_name} must be a non-negative finite number"
                    )
                object.__setattr__(self, field_name, normalized)
        if isinstance(self.error_codes, (str, bytes)):
            raise TypeError("error_codes must be an iterable of strings")
        codes = tuple(self.error_codes)
        if any(not isinstance(code, str) or not code for code in codes):
            raise TypeError("error_codes must contain non-empty strings")
        if len(set(codes)) != len(codes):
            raise ValueError("error_codes must not contain duplicates")
        object.__setattr__(self, "error_codes", codes)

    def to_dict(self) -> dict[str, object]:
        """Return a fresh object suitable for JSON serialization."""

        return {
            "output_valid": self.output_valid,
            "exact_match": self.exact_match,
            "semantic_match": self.semantic_match,
            "target_match": self.target_match,
            "search_region_match": self.search_region_match,
            "track_duration_match": self.track_duration_match,
            "landing_zone_match": self.landing_zone_match,
            "takeoff_altitude_match": self.takeoff_altitude_match,
            "track_duration_error_s": self.track_duration_error_s,
            "takeoff_altitude_error_m": self.takeoff_altitude_error_m,
            "error_codes": list(self.error_codes),
        }


def _invalid_result() -> IntentJudgeResult:
    return IntentJudgeResult(
        output_valid=False,
        exact_match=False,
        semantic_match=False,
        target_match=False,
        search_region_match=False,
        track_duration_match=False,
        landing_zone_match=False,
        takeoff_altitude_match=False,
        track_duration_error_s=None,
        takeoff_altitude_error_m=None,
        error_codes=(IntentErrorCode.PLANNER_OUTPUT_INVALID.value,),
    )


class IntentJudge:
    """Compare a parsed model intent to independently authored Gold intent."""

    def __init__(
        self,
        ontology: TargetOntology | None = None,
        *,
        numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
    ) -> None:
        self._ontology = ontology or TargetOntology.load_default()
        if not isinstance(self._ontology, TargetOntology):
            raise TypeError("ontology must be a TargetOntology")
        if isinstance(numeric_tolerance, bool) or not isinstance(
            numeric_tolerance, (int, float)
        ):
            raise TypeError("numeric_tolerance must be a finite number")
        self._tolerance = float(numeric_tolerance)
        if not isfinite(self._tolerance) or self._tolerance < 0.0:
            raise ValueError("numeric_tolerance must be non-negative and finite")

    @property
    def ontology(self) -> TargetOntology:
        return self._ontology

    def judge(
        self,
        *,
        gold: GoldPlannerSpec,
        predicted: MissionIntent | None,
        world: PlannerWorldCase,
        parse_error: Exception | None = None,
    ) -> IntentJudgeResult:
        if not isinstance(gold, GoldPlannerSpec):
            raise TypeError("gold must be a GoldPlannerSpec")
        if not isinstance(world, PlannerWorldCase):
            raise TypeError("world must be a PlannerWorldCase")
        if parse_error is not None and not isinstance(parse_error, Exception):
            raise TypeError("parse_error must be an Exception or None")
        # Gold/world consistency is checked even for invalid model output.  A
        # broken benchmark must not be silently counted as a planner failure.
        effective_gold = canonicalize_gold(gold, world, self._ontology)
        if parse_error is not None or predicted is None:
            return _invalid_result()
        if not isinstance(predicted, MissionIntent):
            raise TypeError("predicted must be a MissionIntent or None")

        effective_prediction = canonicalize_prediction(
            predicted,
            world,
            self._ontology,
        )
        target_match = (
            effective_prediction.target_concept_id is not None
            and effective_prediction.target_concept_id
            == effective_gold.target_concept_id
        )
        search_region_match = (
            effective_prediction.search_region == effective_gold.search_region
        )
        track_duration_match = finite_numbers_match(
            effective_prediction.track_duration_s,
            effective_gold.track_duration_s,
            tolerance=self._tolerance,
        )
        landing_zone_match = (
            effective_prediction.landing_zone == effective_gold.landing_zone
        )
        takeoff_altitude_match = finite_numbers_match(
            effective_prediction.takeoff_altitude_m,
            effective_gold.takeoff_altitude_m,
            tolerance=self._tolerance,
        )
        semantic_match = all(
            (
                target_match,
                search_region_match,
                track_duration_match,
                landing_zone_match,
                takeoff_altitude_match,
            )
        )

        exact_altitude_match = (
            gold.takeoff_altitude_m is None
            and predicted.takeoff_altitude_m is None
        ) or (
            gold.takeoff_altitude_m is not None
            and predicted.takeoff_altitude_m is not None
            and finite_numbers_match(
                gold.takeoff_altitude_m,
                predicted.takeoff_altitude_m,
                tolerance=self._tolerance,
            )
        )
        exact_match = all(
            (
                predicted.target_description == gold.target_description,
                predicted.search_region == gold.search_region,
                finite_numbers_match(
                    predicted.track_duration_s,
                    gold.track_duration_s,
                    tolerance=self._tolerance,
                ),
                predicted.landing_zone == gold.landing_zone,
                exact_altitude_match,
            )
        )

        errors: list[str] = []
        if effective_prediction.target_concept_id is None:
            errors.append(IntentErrorCode.UNKNOWN_TARGET_DESCRIPTION.value)
        if not target_match:
            errors.append(IntentErrorCode.TARGET_MISMATCH.value)
        if not search_region_match:
            errors.append(IntentErrorCode.SEARCH_REGION_MISMATCH.value)
        if not track_duration_match:
            errors.append(IntentErrorCode.TRACK_DURATION_MISMATCH.value)
        if not landing_zone_match:
            errors.append(IntentErrorCode.LANDING_ZONE_MISMATCH.value)
        if not takeoff_altitude_match:
            errors.append(IntentErrorCode.TAKEOFF_ALTITUDE_MISMATCH.value)

        return IntentJudgeResult(
            output_valid=True,
            exact_match=exact_match,
            semantic_match=semantic_match,
            target_match=target_match,
            search_region_match=search_region_match,
            track_duration_match=track_duration_match,
            landing_zone_match=landing_zone_match,
            takeoff_altitude_match=takeoff_altitude_match,
            track_duration_error_s=abs(
                effective_prediction.track_duration_s
                - effective_gold.track_duration_s
            ),
            takeoff_altitude_error_m=abs(
                effective_prediction.takeoff_altitude_m
                - effective_gold.takeoff_altitude_m
            ),
            error_codes=tuple(errors),
        )


__all__ = ["IntentErrorCode", "IntentJudge", "IntentJudgeResult"]
