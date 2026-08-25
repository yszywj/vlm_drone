"""Leakage-safe temporal sequence construction for target-state training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from datasets.target_state.schema import TargetStateFrameRecord


@dataclass(frozen=True, slots=True)
class TargetStateSequence:
    sequence_id: str
    history: tuple[TargetStateFrameRecord, ...]
    reference: TargetStateFrameRecord
    delta_t_s: tuple[float, ...]
    missing_mask: tuple[bool, ...]
    tracker_id_changed: tuple[bool, ...]
    target_present_mask: tuple[bool, ...] = ()
    sequence_group_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        history = tuple(self.history)
        if not 4 <= len(history) <= 8:
            raise ValueError("history must contain 4 to 8 observations")
        if not isinstance(self.reference, TargetStateFrameRecord):
            raise TypeError("reference must be TargetStateFrameRecord")
        if any(not isinstance(item, TargetStateFrameRecord) for item in history):
            raise TypeError("history items must be TargetStateFrameRecord")
        frames = (*history, self.reference)
        if any(left.timestamp_s >= right.timestamp_s for left, right in zip(frames, frames[1:])):
            raise ValueError("sequence frames must be strictly time ordered")
        for field in ("uav_id", "assignment_id", "episode_id"):
            if len({getattr(item, field) for item in frames}) != 1:
                raise ValueError(f"sequence cannot cross {field}")
        candidate_ids = {
            item.detector_prediction.candidate_id
            for item in frames
            if item.detector_prediction.candidate_id is not None
        }
        if len(candidate_ids) > 1:
            raise ValueError("sequence cannot mix candidates")
        instance_ids = {
            item.training_label.instance_id
            for item in frames
            if item.training_label is not None
            and item.training_label.instance_id is not None
        }
        if len(instance_ids) > 1:
            raise ValueError("sequence cannot mix target instances")
        expected = len(frames)
        if len(self.delta_t_s) != expected or len(self.missing_mask) != expected or len(self.tracker_id_changed) != expected:
            raise ValueError("delta_t_s/masks must have history_size + 1 entries")
        deltas = tuple(float(value) for value in self.delta_t_s)
        if any(not isfinite(value) or value < 0.0 for value in deltas):
            raise ValueError("delta_t_s values must be finite and non-negative")
        if abs(deltas[-1]) > 1e-9:
            raise ValueError("reference delta_t_s must be zero")
        expected_deltas = tuple(self.reference.timestamp_s - item.timestamp_s for item in frames)
        if any(abs(left - right) > 1e-6 for left, right in zip(deltas, expected_deltas)):
            raise ValueError("delta_t_s must be relative to the reference timestamp")
        if any(not isinstance(value, bool) for value in (*self.missing_mask, *self.tracker_id_changed)):
            raise TypeError("sequence masks must contain bool values")
        target_present = tuple(self.target_present_mask)
        if not target_present:
            target_present = tuple(item.training_label is not None for item in frames)
        if len(target_present) != expected or any(
            not isinstance(value, bool) for value in target_present
        ):
            raise ValueError("target_present_mask must contain history_size + 1 bools")
        expected_target_present = tuple(
            item.training_label is not None for item in frames
        )
        if target_present != expected_target_present:
            raise ValueError("target_present_mask does not match training labels")
        group_id = self.sequence_group_id
        if group_id is None:
            if candidate_ids:
                group_id = next(iter(candidate_ids))
            elif not any(target_present):
                group_id = "negative_background"
            else:
                raise ValueError(
                    "target-bearing sequences require one detector candidate_id"
                )
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("sequence_group_id must be non-empty")
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "delta_t_s", deltas)
        object.__setattr__(self, "target_present_mask", target_present)
        object.__setattr__(self, "sequence_group_id", group_id)

    @property
    def max_history_age_s(self) -> float:
        return max(self.delta_t_s)

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "history_frame_ids": [item.frame_id for item in self.history],
            "reference_frame_id": self.reference.frame_id,
            "uav_id": self.reference.uav_id,
            "assignment_id": self.reference.assignment_id,
            "episode_id": self.reference.episode_id,
            "candidate_id": self.reference.detector_prediction.candidate_id,
            "delta_t_s": list(self.delta_t_s),
            "missing_mask": list(self.missing_mask),
            "tracker_id_changed": list(self.tracker_id_changed),
            "target_present_mask": list(self.target_present_mask),
            "sequence_group_id": self.sequence_group_id,
        }


def build_sequences(
    records: Sequence[TargetStateFrameRecord],
    *,
    history_size: int = 6,
    max_history_age_s: float = 2.0,
) -> tuple[TargetStateSequence, ...]:
    if not 4 <= history_size <= 8:
        raise ValueError("history_size must be within [4, 8]")
    if not isfinite(max_history_age_s) or max_history_age_s <= 0.0:
        raise ValueError("max_history_age_s must be finite and positive")
    groups: dict[tuple[str, str, str, str], list[TargetStateFrameRecord]] = {}
    for record in records:
        if not isinstance(record, TargetStateFrameRecord):
            raise TypeError("records must contain TargetStateFrameRecord")
        candidate_id = record.detector_prediction.candidate_id
        if candidate_id is None:
            if record.training_label is not None:
                # No deployable candidate exists yet.  Keep this honest frame
                # in frames.jsonl, but do not invent a target identity merely
                # to make it trainable.
                continue
            group_id = "negative_background"
        else:
            group_id = candidate_id
        key = (record.uav_id, record.assignment_id, record.episode_id, group_id)
        groups.setdefault(key, []).append(record)
    result: list[TargetStateSequence] = []
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda item: (item.timestamp_s, item.frame_id))
        for reference_index in range(history_size, len(ordered)):
            reference = ordered[reference_index]
            history = tuple(ordered[reference_index - history_size : reference_index])
            frames = (*history, reference)
            deltas = tuple(reference.timestamp_s - item.timestamp_s for item in frames)
            if max(deltas) > max_history_age_s + 1e-9:
                continue
            # A sensor-only candidate can attach to a different physical
            # object during a crossing.  That association error is useful
            # deployment data, but one supervised temporal window must never
            # contain labels from two target instances.  Skip only windows
            # spanning such a switch: later windows remain eligible once the
            # new association has a complete history.  Null labels remain in
            # the window as explicit target-absent/false-positive examples.
            instance_ids = {
                item.training_label.instance_id
                for item in frames
                if item.training_label is not None
                and item.training_label.instance_id is not None
            }
            if len(instance_ids) > 1:
                continue
            tracker_ids = [item.detector_prediction.tracker_id for item in frames]
            changed = tuple(
                False if index == 0 else tracker_ids[index] != tracker_ids[index - 1]
                for index in range(len(frames))
            )
            missing = tuple(not item.detector_prediction.detected for item in frames)
            result.append(
                TargetStateSequence(
                    sequence_id=f"seq_{key[0]}_{key[1]}_{reference.frame_id}",
                    history=history,
                    reference=reference,
                    delta_t_s=deltas,
                    missing_mask=missing,
                    tracker_id_changed=changed,
                    target_present_mask=tuple(
                        item.training_label is not None for item in frames
                    ),
                    sequence_group_id=key[3],
                )
            )
    return tuple(result)


__all__ = ["TargetStateSequence", "build_sequences"]
