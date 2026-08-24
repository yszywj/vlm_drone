"""Bounded candidate lifecycle and negative-memory store."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from numbers import Real

from common.ids import validate_review_id, validate_routing_id, validate_uav_id
from common.provenance import is_privileged_oracle_source
from runtime.frame_store import FrameRef


class CandidateLifecycle(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    UNDER_INSPECTION = "UNDER_INSPECTION"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    STALE = "STALE"


def _timestamp(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("bbox must contain four normalized coordinates")
    result = tuple(_timestamp(component, "bbox component") for component in value)
    if any(component > 1.0 for component in result):
        raise ValueError("bbox coordinates must be within [0, 1]")
    x1, y1, x2, y2 = result
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
    return result  # type: ignore[return-value]


def _source(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source must be a non-empty string")
    normalized = value.strip()
    if (
        is_privileged_oracle_source(normalized)
        and normalized != "oracle_evaluation"
    ):
        raise ValueError(
            "privileged candidates must use the explicit canonical source "
            "'oracle_evaluation'; production boundaries still reject it"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class CandidateReviewRef:
    review_id: str
    timestamp_s: float
    decision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(
            self,
            "timestamp_s",
            _timestamp(self.timestamp_s, "timestamp_s"),
        )
        if not isinstance(self.decision, str) or not self.decision.strip():
            raise ValueError("decision must be a non-empty string")
        object.__setattr__(self, "decision", self.decision.strip())


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    uav_id: str
    candidate_id: str
    first_seen_timestamp_s: float
    last_seen_timestamp_s: float
    bbox_history: tuple[tuple[float, float, float, float], ...]
    frame_history: tuple[FrameRef, ...]
    source: str
    lifecycle: CandidateLifecycle
    review_history: tuple[CandidateReviewRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self,
            "candidate_id",
            validate_routing_id(self.candidate_id, "candidate_id"),
        )
        first = _timestamp(self.first_seen_timestamp_s, "first_seen_timestamp_s")
        last = _timestamp(self.last_seen_timestamp_s, "last_seen_timestamp_s")
        if last < first:
            raise ValueError("last_seen_timestamp_s cannot predate first_seen_timestamp_s")
        object.__setattr__(self, "first_seen_timestamp_s", first)
        object.__setattr__(self, "last_seen_timestamp_s", last)
        object.__setattr__(
            self,
            "bbox_history",
            tuple(_bbox(value) for value in self.bbox_history),
        )
        if not self.bbox_history:
            raise ValueError("bbox_history must be non-empty")
        frame_history = tuple(self.frame_history)
        if not frame_history or any(
            not isinstance(frame, FrameRef) or frame.uav_id != self.uav_id
            for frame in frame_history
        ):
            raise ValueError("frame_history must contain matching FrameRef values")
        object.__setattr__(self, "frame_history", frame_history)
        object.__setattr__(self, "source", _source(self.source))
        if not isinstance(self.lifecycle, CandidateLifecycle):
            raise TypeError("lifecycle must be a CandidateLifecycle")
        reviews = tuple(self.review_history)
        if any(not isinstance(review, CandidateReviewRef) for review in reviews):
            raise TypeError("review_history must contain CandidateReviewRef values")
        object.__setattr__(self, "review_history", reviews)

    def to_dict(self) -> dict[str, object]:
        return {
            "uav_id": self.uav_id,
            "candidate_id": self.candidate_id,
            "first_seen_timestamp_s": self.first_seen_timestamp_s,
            "last_seen_timestamp_s": self.last_seen_timestamp_s,
            "bbox_history": [list(value) for value in self.bbox_history],
            "frame_history": [frame.to_dict() for frame in self.frame_history],
            "source": self.source,
            "lifecycle": self.lifecycle.value,
            "review_history": [
                {
                    "review_id": review.review_id,
                    "timestamp_s": review.timestamp_s,
                    "decision": review.decision,
                }
                for review in self.review_history
            ],
        }


class CandidateBank:
    """One-UAV bounded candidate store with rejection cooldown memory."""

    def __init__(
        self,
        *,
        uav_id: str,
        max_candidates: int = 32,
        max_history_per_candidate: int = 8,
        max_review_history: int = 8,
        rejected_cooldown_s: float = 10.0,
        stale_after_s: float = 20.0,
    ) -> None:
        self._uav_id = validate_uav_id(uav_id)
        self._max_candidates = _positive_int(max_candidates, "max_candidates")
        self._max_history = _positive_int(
            max_history_per_candidate,
            "max_history_per_candidate",
        )
        self._max_review_history = _positive_int(
            max_review_history,
            "max_review_history",
        )
        self._rejected_cooldown_s = _timestamp(
            rejected_cooldown_s,
            "rejected_cooldown_s",
        )
        self._stale_after_s = _timestamp(stale_after_s, "stale_after_s")
        if self._rejected_cooldown_s == 0.0 or self._stale_after_s == 0.0:
            raise ValueError("candidate time bounds must be greater than zero")
        self._candidates: dict[str, CandidateSnapshot] = {}

    @property
    def uav_id(self) -> str:
        return self._uav_id

    def propose(
        self,
        *,
        candidate_id: str,
        timestamp_s: float,
        bbox_xyxy_normalized: tuple[float, float, float, float],
        frame_ref: FrameRef,
        source: str,
    ) -> CandidateSnapshot | None:
        candidate_id = validate_routing_id(candidate_id, "candidate_id")
        timestamp = _timestamp(timestamp_s, "timestamp_s")
        bbox = _bbox(bbox_xyxy_normalized)
        if not isinstance(frame_ref, FrameRef) or frame_ref.uav_id != self._uav_id:
            raise ValueError("frame_ref must belong to this CandidateBank uav_id")
        if abs(frame_ref.timestamp_s - timestamp) > 1e-9:
            raise ValueError("frame_ref timestamp must match candidate timestamp")
        normalized_source = _source(source)
        existing = self._candidates.get(candidate_id)
        if existing is not None:
            if timestamp < existing.last_seen_timestamp_s:
                raise ValueError("candidate timestamp cannot move backwards")
            if (
                existing.lifecycle is CandidateLifecycle.REJECTED
                and timestamp - existing.last_seen_timestamp_s
                < self._rejected_cooldown_s
            ):
                return None
            starts_new_evidence_epoch = existing.lifecycle in {
                CandidateLifecycle.REJECTED,
                CandidateLifecycle.STALE,
            }
            first = (
                timestamp
                if starts_new_evidence_epoch
                else existing.first_seen_timestamp_s
            )
            # Cooldown retains only negative-memory identity.  Once recovery
            # is allowed, the reused candidate_id begins a new evidence epoch:
            # old images, boxes, and semantic reviews cannot count again.
            bboxes = (
                (bbox,)
                if starts_new_evidence_epoch
                else (*existing.bbox_history, bbox)[-self._max_history :]
            )
            frames = (
                (frame_ref,)
                if starts_new_evidence_epoch
                else (*existing.frame_history, frame_ref)[-self._max_history :]
            )
            lifecycle = (
                CandidateLifecycle.PROVISIONAL
                if starts_new_evidence_epoch
                else existing.lifecycle
            )
            reviews = () if starts_new_evidence_epoch else existing.review_history
            snapshot = CandidateSnapshot(
                self._uav_id,
                candidate_id,
                first,
                timestamp,
                bboxes,
                frames,
                normalized_source,
                lifecycle,
                reviews,
            )
        else:
            snapshot = CandidateSnapshot(
                self._uav_id,
                candidate_id,
                timestamp,
                timestamp,
                (bbox,),
                (frame_ref,),
                normalized_source,
                CandidateLifecycle.PROVISIONAL,
                (),
            )
        self._candidates[candidate_id] = snapshot
        self._evict_oldest()
        return snapshot

    def find_matching(
        self,
        *,
        timestamp_s: float,
        bbox_xyxy_normalized: tuple[float, float, float, float],
        min_iou: float = 0.5,
        source: str | None = None,
    ) -> CandidateSnapshot | None:
        """Find one recent image-space candidate by latest-box overlap.

        This is deliberately only a bounded association heuristic. It is not
        a tracker, ReID signal, identity proof, or verification transition.
        Including ``REJECTED`` candidates is intentional: callers can reuse
        their ID when proposing, allowing the existing rejection cooldown to
        suppress immediate repeated reviews instead of minting a new ID.
        """

        timestamp = _timestamp(timestamp_s, "timestamp_s")
        bbox = _bbox(bbox_xyxy_normalized)
        if isinstance(min_iou, bool) or not isinstance(min_iou, Real):
            raise TypeError("min_iou must be a finite number within [0, 1]")
        threshold = float(min_iou)
        if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("min_iou must be a finite number within [0, 1]")
        normalized_source = None if source is None else _source(source)
        self.expire(timestamp)
        matches: list[tuple[float, float, str, CandidateSnapshot]] = []
        for candidate in self._candidates.values():
            if candidate.lifecycle is CandidateLifecycle.STALE:
                continue
            if normalized_source is not None and candidate.source != normalized_source:
                continue
            # A model review is asynchronous: by completion time the same
            # detector track may already have newer observations.  Associate
            # against the most recent retained box at (or before) the review
            # observation time instead of rejecting the candidate merely
            # because its live ``last_seen`` has advanced.
            historical = [
                (frame.timestamp_s, candidate_bbox)
                for frame, candidate_bbox in zip(
                    candidate.frame_history,
                    candidate.bbox_history,
                )
                if frame.timestamp_s <= timestamp + 1e-9
            ]
            if not historical:
                continue
            association_time, association_bbox = max(
                historical,
                key=lambda item: item[0],
            )
            overlap = _bbox_iou(association_bbox, bbox)
            if overlap >= threshold:
                matches.append(
                    (
                        overlap,
                        association_time,
                        candidate.candidate_id,
                        candidate,
                    )
                )
        if not matches:
            return None
        # Highest overlap wins; ties prefer the most recently seen candidate,
        # then stable candidate_id order for reproducibility.
        matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return matches[0][3]

    def associate_proposal(
        self,
        *,
        timestamp_s: float,
        bbox_xyxy_normalized: tuple[float, float, float, float],
        proposed_candidate_id: str,
        min_iou: float = 0.5,
        source: str | None = None,
    ) -> str | None:
        """Resolve an image-space proposal to a bounded candidate ID.

        The method only performs the same deterministic IoU association as
        :meth:`find_matching`; it does not add evidence or claim tracking,
        ReID, or identity verification.  ``None`` means a matching rejected
        candidate is still inside its negative-memory cooldown and therefore
        must not contribute to semantic consensus.
        """

        timestamp = _timestamp(timestamp_s, "timestamp_s")
        proposed = validate_routing_id(
            proposed_candidate_id,
            "proposed_candidate_id",
        )
        existing = self.find_matching(
            timestamp_s=timestamp,
            bbox_xyxy_normalized=bbox_xyxy_normalized,
            min_iou=min_iou,
            source=source,
        )
        if existing is None:
            return proposed
        if (
            existing.lifecycle is CandidateLifecycle.REJECTED
            and timestamp - existing.last_seen_timestamp_s
            < self._rejected_cooldown_s
        ):
            return None
        return existing.candidate_id

    def mark_under_inspection(self, candidate_id: str) -> CandidateSnapshot:
        return self._transition(
            candidate_id,
            CandidateLifecycle.UNDER_INSPECTION,
            allowed={CandidateLifecycle.PROVISIONAL},
        )

    def release_inspection_pending_review(
        self,
        candidate_id: str,
    ) -> CandidateSnapshot:
        """Release INSPECT ownership without claiming semantic verification.

        Collecting better views is evidence acquisition, not identity proof.
        The candidate therefore returns to ``PROVISIONAL`` until an
        independent review gate explicitly verifies or rejects it.
        """

        return self._transition(
            candidate_id,
            CandidateLifecycle.PROVISIONAL,
            allowed={CandidateLifecycle.UNDER_INSPECTION},
        )

    def verify(self, candidate_id: str) -> CandidateSnapshot:
        return self._transition(
            candidate_id,
            CandidateLifecycle.VERIFIED,
            allowed={
                CandidateLifecycle.PROVISIONAL,
                CandidateLifecycle.UNDER_INSPECTION,
            },
        )

    def reject(self, candidate_id: str, *, timestamp_s: float) -> CandidateSnapshot:
        candidate_id = validate_routing_id(candidate_id, "candidate_id")
        current = self._require(candidate_id)
        timestamp = _timestamp(timestamp_s, "timestamp_s")
        if timestamp < current.last_seen_timestamp_s:
            raise ValueError("rejection timestamp cannot move backwards")
        updated = replace(
            current,
            last_seen_timestamp_s=timestamp,
            lifecycle=CandidateLifecycle.REJECTED,
        )
        self._candidates[candidate_id] = updated
        return updated

    def add_review(
        self,
        candidate_id: str,
        review: CandidateReviewRef,
    ) -> CandidateSnapshot:
        if not isinstance(review, CandidateReviewRef):
            raise TypeError("review must be a CandidateReviewRef")
        current = self._require(candidate_id)
        if review.timestamp_s < current.first_seen_timestamp_s:
            raise ValueError("review cannot predate the candidate")
        updated = replace(
            current,
            review_history=(*current.review_history, review)[
                -self._max_review_history :
            ],
        )
        self._candidates[current.candidate_id] = updated
        return updated

    def expire(self, now_s: float) -> tuple[str, ...]:
        now = _timestamp(now_s, "now_s")
        expired: list[str] = []
        for candidate_id, candidate in tuple(self._candidates.items()):
            if (
                candidate.lifecycle
                not in {CandidateLifecycle.REJECTED, CandidateLifecycle.VERIFIED}
                and now - candidate.last_seen_timestamp_s >= self._stale_after_s
            ):
                self._candidates[candidate_id] = replace(
                    candidate,
                    lifecycle=CandidateLifecycle.STALE,
                )
                expired.append(candidate_id)
        return tuple(expired)

    def get(self, candidate_id: str) -> CandidateSnapshot | None:
        candidate_id = validate_routing_id(candidate_id, "candidate_id")
        return self._candidates.get(candidate_id)

    def snapshots(self) -> tuple[CandidateSnapshot, ...]:
        return tuple(
            sorted(
                self._candidates.values(),
                key=lambda item: (item.first_seen_timestamp_s, item.candidate_id),
            )
        )

    def clear(self) -> int:
        """Reset mission-scoped candidate and negative-memory state."""

        removed = len(self._candidates)
        self._candidates.clear()
        return removed

    def _require(self, candidate_id: str) -> CandidateSnapshot:
        normalized = validate_routing_id(candidate_id, "candidate_id")
        try:
            return self._candidates[normalized]
        except KeyError as exc:
            raise KeyError(f"unknown candidate_id: {normalized}") from exc

    def _transition(
        self,
        candidate_id: str,
        lifecycle: CandidateLifecycle,
        *,
        allowed: set[CandidateLifecycle],
    ) -> CandidateSnapshot:
        current = self._require(candidate_id)
        if current.lifecycle not in allowed:
            raise ValueError(
                f"candidate transition {current.lifecycle.value} -> "
                f"{lifecycle.value} is not allowed"
            )
        updated = replace(current, lifecycle=lifecycle)
        self._candidates[current.candidate_id] = updated
        return updated

    def _evict_oldest(self) -> None:
        while len(self._candidates) > self._max_candidates:
            victim = min(
                self._candidates.values(),
                key=lambda item: (item.last_seen_timestamp_s, item.candidate_id),
            )
            del self._candidates[victim.candidate_id]


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


__all__ = [
    "CandidateBank",
    "CandidateLifecycle",
    "CandidateReviewRef",
    "CandidateSnapshot",
]
