"""Strict, geometry-limited Qwen visual-review value types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Integral, Real

from common.ids import (
    validate_mission_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)


class VisualReviewDecision(str, Enum):
    NO_RELEVANT_CHANGE = "NO_RELEVANT_CHANGE"
    NO_TARGET = "NO_TARGET"
    POSSIBLE_TARGET = "POSSIBLE_TARGET"
    TARGET_MATCH = "TARGET_MATCH"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    AMBIGUOUS = "AMBIGUOUS"
    PATH_MAY_BE_BLOCKED = "PATH_MAY_BE_BLOCKED"
    TASK_REVIEW_REQUIRED = "TASK_REVIEW_REQUIRED"


class VisualReviewAction(str, Enum):
    CONTINUE = "CONTINUE"
    HOVER = "HOVER"
    INSPECT = "INSPECT"
    REACQUIRE = "REACQUIRE"
    REQUEST_REPLAN = "REQUEST_REPLAN"
    REQUEST_LAND = "REQUEST_LAND"


class VisualReviewMode(str, Enum):
    SHADOW = "shadow"
    GATE = "gate"


class ReviewDisposition(str, Enum):
    SHADOW_RECORDED = "SHADOW_RECORDED"
    PENDING = "PENDING"
    CONSENSUS_REACHED = "CONSENSUS_REACHED"
    STALE = "STALE"


class VisualReviewProtocolError(RuntimeError):
    """Raised when a model review violates trusted routing metadata."""


def _strict_keys(
    data: Mapping[str, object],
    required: frozenset[str],
    name: str,
) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - required
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return normalized


def _text(value: object, name: str, *, max_length: int = 512) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > max_length:
        raise ValueError(f"{name} must contain at most {max_length} characters")
    return normalized


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _confidence(value: object) -> float | None:
    if value is None:
        return None
    normalized = _finite(value, "self_reported_confidence")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError("self_reported_confidence must be within [0, 1]")
    return normalized


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 4
    ):
        raise ValueError("bbox_xyxy_normalized must contain four numbers or null")
    values = tuple(
        _finite(component, f"bbox_xyxy_normalized[{index}]")
        for index, component in enumerate(value)
    )
    if any(component < 0.0 or component > 1.0 for component in values):
        raise ValueError("bbox_xyxy_normalized values must be within [0, 1]")
    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox_xyxy_normalized must satisfy x1 < x2 and y1 < y2")
    return x1, y1, x2, y2


def _text_tuple(value: object, name: str, *, max_items: int = 32) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if len(value) > max_items:
        raise ValueError(f"{name} must contain at most {max_items} items")
    return tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class VisualReviewCandidate:
    present: bool
    bbox_xyxy_normalized: tuple[float, float, float, float] | None
    description: str | None
    self_reported_confidence: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.present, bool):
            raise TypeError("candidate.present must be bool")
        object.__setattr__(
            self,
            "bbox_xyxy_normalized",
            _bbox(self.bbox_xyxy_normalized),
        )
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, "candidate.description"),
        )
        object.__setattr__(
            self,
            "self_reported_confidence",
            _confidence(self.self_reported_confidence),
        )
        values = (
            self.bbox_xyxy_normalized,
            self.description,
            self.self_reported_confidence,
        )
        if self.present and any(value is None for value in values):
            raise ValueError("a present candidate requires bbox, description, and confidence")
        if not self.present and any(value is not None for value in values):
            raise ValueError("an absent candidate must not contain candidate evidence")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "VisualReviewCandidate":
        if not isinstance(data, Mapping):
            raise TypeError("candidate must be an object")
        required = frozenset(
            {
                "present",
                "bbox_xyxy_normalized",
                "description",
                "self_reported_confidence",
            }
        )
        _strict_keys(data, required, "candidate")
        return cls(
            present=data["present"],  # type: ignore[arg-type]
            bbox_xyxy_normalized=data["bbox_xyxy_normalized"],  # type: ignore[arg-type]
            description=data["description"],  # type: ignore[arg-type]
            self_reported_confidence=data["self_reported_confidence"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "bbox_xyxy_normalized": (
                None
                if self.bbox_xyxy_normalized is None
                else list(self.bbox_xyxy_normalized)
            ),
            "description": self.description,
            "self_reported_confidence": self.self_reported_confidence,
        }


@dataclass(frozen=True, slots=True)
class QwenVisualReview:
    schema_version: int
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    observation_timestamp_s: float
    frame_id: str
    decision: VisualReviewDecision
    candidate: VisualReviewCandidate
    scene_observations: tuple[str, ...]
    reason_codes: tuple[str, ...]
    recommended_action: VisualReviewAction

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("QwenVisualReview.schema_version must equal 1")
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "plan_version",
            _positive_int(self.plan_version, "plan_version"),
        )
        timestamp = _finite(
            self.observation_timestamp_s,
            "observation_timestamp_s",
        )
        if timestamp < 0.0:
            raise ValueError("observation_timestamp_s must be non-negative")
        object.__setattr__(self, "observation_timestamp_s", timestamp)
        object.__setattr__(
            self,
            "frame_id",
            validate_routing_id(self.frame_id, "frame_id"),
        )
        if not isinstance(self.decision, VisualReviewDecision):
            raise TypeError("decision must be a VisualReviewDecision")
        if not isinstance(self.candidate, VisualReviewCandidate):
            raise TypeError("candidate must be a VisualReviewCandidate")
        object.__setattr__(
            self,
            "scene_observations",
            _text_tuple(self.scene_observations, "scene_observations"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _text_tuple(self.reason_codes, "reason_codes"),
        )
        if not isinstance(self.recommended_action, VisualReviewAction):
            raise TypeError("recommended_action must be a VisualReviewAction")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "QwenVisualReview":
        if not isinstance(data, Mapping):
            raise TypeError("Qwen visual review must be an object")
        required = frozenset(
            {
                "schema_version",
                "review_id",
                "mission_id",
                "uav_id",
                "plan_version",
                "observation_timestamp_s",
                "frame_id",
                "decision",
                "candidate",
                "scene_observations",
                "reason_codes",
                "recommended_action",
            }
        )
        _strict_keys(data, required, "Qwen visual review")
        try:
            decision = VisualReviewDecision(data["decision"])
        except (TypeError, ValueError) as exc:
            raise ValueError("decision is not supported") from exc
        try:
            action = VisualReviewAction(data["recommended_action"])
        except (TypeError, ValueError) as exc:
            raise ValueError("recommended_action is not supported") from exc
        return cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            review_id=data["review_id"],  # type: ignore[arg-type]
            mission_id=data["mission_id"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            plan_version=data["plan_version"],  # type: ignore[arg-type]
            observation_timestamp_s=data["observation_timestamp_s"],  # type: ignore[arg-type]
            frame_id=data["frame_id"],  # type: ignore[arg-type]
            decision=decision,
            candidate=VisualReviewCandidate.from_dict(data["candidate"]),  # type: ignore[arg-type]
            scene_observations=data["scene_observations"],  # type: ignore[arg-type]
            reason_codes=data["reason_codes"],  # type: ignore[arg-type]
            recommended_action=action,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "review_id": self.review_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "observation_timestamp_s": self.observation_timestamp_s,
            "frame_id": self.frame_id,
            "decision": self.decision.value,
            "candidate": self.candidate.to_dict(),
            "scene_observations": list(self.scene_observations),
            "reason_codes": list(self.reason_codes),
            "recommended_action": self.recommended_action.value,
        }


@dataclass(frozen=True, slots=True)
class VisualReviewExpectation:
    """Trusted routing/time tuple against which a worker response is checked."""

    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    observation_timestamp_s: float
    frame_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "plan_version",
            _positive_int(self.plan_version, "plan_version"),
        )
        timestamp = _finite(
            self.observation_timestamp_s,
            "observation_timestamp_s",
        )
        if timestamp < 0.0:
            raise ValueError("observation_timestamp_s must be non-negative")
        object.__setattr__(self, "observation_timestamp_s", timestamp)
        object.__setattr__(
            self,
            "frame_id",
            validate_routing_id(self.frame_id, "frame_id"),
        )


@dataclass(frozen=True, slots=True)
class VisualReviewAcceptance:
    disposition: ReviewDisposition
    accepted_for_control: bool
    consistent_match_count: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, ReviewDisposition):
            raise TypeError("disposition must be a ReviewDisposition")
        if not isinstance(self.accepted_for_control, bool):
            raise TypeError("accepted_for_control must be bool")
        if (
            isinstance(self.consistent_match_count, bool)
            or not isinstance(self.consistent_match_count, Integral)
            or self.consistent_match_count < 0
        ):
            raise ValueError("consistent_match_count must be a non-negative integer")
        object.__setattr__(self, "reason", _text(self.reason, "reason"))


class VisualReviewGate:
    """Route reviews and require temporal semantic consensus in gate mode.

    This class deliberately emits no ``IdentityConsistencyEvidence`` and never
    mutates ``TargetManager``.  A future real ReID/confirmation pipeline must
    consume the consensus separately.
    """

    def __init__(
        self,
        *,
        mode: VisualReviewMode | str = VisualReviewMode.SHADOW,
        min_consistent_matches: int = 2,
    ) -> None:
        try:
            self._mode = VisualReviewMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("mode must be shadow or gate") from exc
        self._min_consistent_matches = _positive_int(
            min_consistent_matches,
            "min_consistent_matches",
        )
        self._match_count = 0
        self._route_key: tuple[str, str, int, str | None] | None = None
        self._last_review_id: str | None = None
        self._last_observation_timestamp_s: float | None = None

    @property
    def mode(self) -> VisualReviewMode:
        return self._mode

    def reset(self) -> None:
        self._match_count = 0
        self._route_key = None
        self._last_review_id = None
        self._last_observation_timestamp_s = None

    def evaluate(
        self,
        review: QwenVisualReview,
        expectation: VisualReviewExpectation,
        *,
        consensus_key: str | None = None,
    ) -> VisualReviewAcceptance:
        if not isinstance(review, QwenVisualReview):
            raise TypeError("review must be a QwenVisualReview")
        if not isinstance(expectation, VisualReviewExpectation):
            raise TypeError("expectation must be a VisualReviewExpectation")
        identity_fields = (
            ("review_id", review.review_id, expectation.review_id),
            ("mission_id", review.mission_id, expectation.mission_id),
            ("uav_id", review.uav_id, expectation.uav_id),
        )
        mismatched = [name for name, actual, expected in identity_fields if actual != expected]
        if mismatched:
            raise VisualReviewProtocolError(
                "visual review routing mismatch: " + ", ".join(mismatched)
            )
        stale_fields: list[str] = []
        if review.plan_version != expectation.plan_version:
            stale_fields.append("plan_version")
        if review.frame_id != expectation.frame_id:
            stale_fields.append("frame_id")
        if (
            abs(
                review.observation_timestamp_s
                - expectation.observation_timestamp_s
            )
            > 1e-9
        ):
            stale_fields.append("observation_timestamp_s")
        if stale_fields:
            return VisualReviewAcceptance(
                ReviewDisposition.STALE,
                False,
                self._match_count,
                "stale:" + ",".join(stale_fields),
            )
        if self._mode is VisualReviewMode.SHADOW:
            return VisualReviewAcceptance(
                ReviewDisposition.SHADOW_RECORDED,
                False,
                self._match_count,
                "shadow_mode_has_no_control_effect",
            )

        normalized_consensus_key = (
            None
            if consensus_key is None
            else validate_routing_id(consensus_key, "consensus_key")
        )
        route_key = (
            review.mission_id,
            review.uav_id,
            review.plan_version,
            normalized_consensus_key,
        )
        if self._route_key != route_key:
            # Evidence from an old mission or plan revision must never be
            # combined with the new control decision.
            self._match_count = 0
            self._last_review_id = None
            self._last_observation_timestamp_s = None
            self._route_key = route_key
        if (
            review.review_id == self._last_review_id
            or (
                self._last_observation_timestamp_s is not None
                and review.observation_timestamp_s
                <= self._last_observation_timestamp_s
            )
        ):
            return VisualReviewAcceptance(
                ReviewDisposition.STALE,
                False,
                self._match_count,
                "duplicate_or_non_increasing_review",
            )
        self._last_review_id = review.review_id
        self._last_observation_timestamp_s = review.observation_timestamp_s
        if normalized_consensus_key is None:
            # Route equality is not identity equality. SEARCH/INSPECT reviews
            # without a trusted CandidateBank association must fail closed
            # instead of pooling arbitrary image regions into one consensus.
            self._match_count = 0
            return VisualReviewAcceptance(
                ReviewDisposition.PENDING,
                False,
                0,
                "trusted_consensus_identity_required",
            )
        if (
            review.decision
            in {
                VisualReviewDecision.TARGET_MATCH,
                VisualReviewDecision.POSSIBLE_TARGET,
            }
            and review.candidate.present
        ):
            # POSSIBLE_TARGET is evidence only for repeated review of the
            # same trusted consensus_key. Reaching this threshold does not
            # create ReID evidence or lock/switch a TargetManager identity.
            self._match_count += 1
        elif review.decision is VisualReviewDecision.TARGET_MISMATCH:
            # One semantic mismatch is never sufficient to discard identity.
            self._match_count = 0
        else:
            self._match_count = 0
        if self._match_count >= self._min_consistent_matches:
            return VisualReviewAcceptance(
                ReviewDisposition.CONSENSUS_REACHED,
                True,
                self._match_count,
                "temporal_semantic_consensus",
            )
        return VisualReviewAcceptance(
            ReviewDisposition.PENDING,
            False,
            self._match_count,
            "more_consistent_reviews_required",
        )


def build_qwen_visual_review_json_schema(
    *,
    review_id: str,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    frame_id: str,
    observation_timestamp_s: float,
) -> dict[str, object]:
    """Build a strict response schema pinned to trusted routing metadata."""

    review_id = validate_review_id(review_id)
    mission_id = validate_mission_id(mission_id)
    uav_id = validate_uav_id(uav_id)
    plan_version = _positive_int(plan_version, "plan_version")
    frame_id = validate_routing_id(frame_id, "frame_id")
    timestamp = _finite(observation_timestamp_s, "observation_timestamp_s")
    if timestamp < 0.0:
        raise ValueError("observation_timestamp_s must be non-negative")
    nullable_string = {"type": ["string", "null"]}
    nullable_number = {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "review_id",
            "mission_id",
            "uav_id",
            "plan_version",
            "observation_timestamp_s",
            "frame_id",
            "decision",
            "candidate",
            "scene_observations",
            "reason_codes",
            "recommended_action",
        ],
        "properties": {
            "schema_version": {"const": 1},
            "review_id": {"const": review_id},
            "mission_id": {"const": mission_id},
            "uav_id": {"const": uav_id},
            "plan_version": {"const": plan_version},
            "observation_timestamp_s": {"const": timestamp},
            "frame_id": {"const": frame_id},
            "decision": {"enum": [item.value for item in VisualReviewDecision]},
            "candidate": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "present",
                    "bbox_xyxy_normalized",
                    "description",
                    "self_reported_confidence",
                ],
                "properties": {
                    "present": {"type": "boolean"},
                    "bbox_xyxy_normalized": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                        ]
                    },
                    "description": nullable_string,
                    "self_reported_confidence": nullable_number,
                },
            },
            "scene_observations": {
                "type": "array",
                "maxItems": 32,
                "items": {"type": "string"},
            },
            "reason_codes": {
                "type": "array",
                "maxItems": 32,
                "items": {"type": "string"},
            },
            "recommended_action": {
                "enum": [item.value for item in VisualReviewAction]
            },
        },
    }


__all__ = [
    "QwenVisualReview",
    "ReviewDisposition",
    "VisualReviewAcceptance",
    "VisualReviewAction",
    "VisualReviewCandidate",
    "VisualReviewDecision",
    "VisualReviewExpectation",
    "VisualReviewGate",
    "VisualReviewMode",
    "VisualReviewProtocolError",
    "build_qwen_visual_review_json_schema",
]
