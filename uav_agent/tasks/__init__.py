"""Independent Gold task definitions and deterministic intent evaluation."""

from .intent_canonicalizer import (
    DEFAULT_NUMERIC_TOLERANCE,
    EffectiveMissionIntent,
    IntentCanonicalizationError,
    canonicalize_gold,
    canonicalize_prediction,
    finite_numbers_match,
)
from .intent_judge import IntentErrorCode, IntentJudge, IntentJudgeResult
from .schemas import (
    GOLD_INTENT_FIELDS,
    GoldPlannerSpec,
    PlannerWorldCase,
    TargetConcept,
)
from .target_ontology import (
    DEFAULT_ONTOLOGY_PATH,
    TargetOntology,
    TargetOntologyError,
    load_target_ontology,
    render_canonical_target_description,
)

__all__ = [
    "DEFAULT_NUMERIC_TOLERANCE",
    "DEFAULT_ONTOLOGY_PATH",
    "EffectiveMissionIntent",
    "GOLD_INTENT_FIELDS",
    "GoldPlannerSpec",
    "IntentCanonicalizationError",
    "IntentErrorCode",
    "IntentJudge",
    "IntentJudgeResult",
    "PlannerWorldCase",
    "TargetConcept",
    "TargetOntology",
    "TargetOntologyError",
    "canonicalize_gold",
    "canonicalize_prediction",
    "finite_numbers_match",
    "load_target_ontology",
    "render_canonical_target_description",
]
