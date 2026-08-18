"""Bounded, image-free logs for dynamic visual missions.

The logger intentionally exposes typed records instead of accepting arbitrary
payload mappings.  This keeps base64 images, prompts, model responses and
environment objects out of persistent logs by construction.  Every append is
flushed immediately and every stream has both a record and byte budget.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import IO

from common.ids import (
    validate_mission_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)


DEFAULT_MAX_RECORDS_PER_STREAM = 10_000
DEFAULT_MAX_BYTES_PER_STREAM = 16 * 1024 * 1024
MAX_RECORD_BYTES = 8 * 1024


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _positive_plan_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("plan_version must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("plan_version must be a positive integer")
    return normalized


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _bounded_text(value: object, name: str, *, maximum: int = 256) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    if any(character in normalized for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{name} must be a single line")
    # This catches image data URLs early even if somebody tries to disguise
    # one as a reason or event label.
    if "base64," in normalized.casefold() or normalized.casefold().startswith("data:image/"):
        raise ValueError(f"{name} must not contain image/base64 data")
    return normalized


def _optional_routing_id(value: str | None, name: str) -> str | None:
    return None if value is None else validate_routing_id(value, name)


def _bbox(
    value: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("bbox_xyxy_normalized must be a four-number tuple or None")
    normalized = tuple(
        _finite_nonnegative(component, f"bbox_xyxy_normalized[{index}]")
        for index, component in enumerate(value)
    )
    if any(component > 1.0 for component in normalized):
        raise ValueError("bbox_xyxy_normalized values must be within [0, 1]")
    x1, y1, x2, y2 = normalized
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox_xyxy_normalized must satisfy x1 < x2 and y1 < y2")
    return x1, y1, x2, y2


@dataclass(frozen=True, slots=True)
class QwenReviewLogRecord:
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    frame_id: str
    observation_timestamp_s: float
    decision: str
    bbox_xyxy_normalized: tuple[float, float, float, float] | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    stale: bool = False
    accepted: bool = False
    timeout: bool = False
    request_id: str | None = None
    step_id: str | None = None
    semantic_source: str = "qwen_vl"
    geometry_source: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _positive_plan_version(self.plan_version))
        object.__setattr__(self, "frame_id", validate_routing_id(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "observation_timestamp_s",
            _finite_nonnegative(self.observation_timestamp_s, "observation_timestamp_s"),
        )
        object.__setattr__(self, "decision", _bounded_text(self.decision, "decision", maximum=64))
        object.__setattr__(self, "bbox_xyxy_normalized", _bbox(self.bbox_xyxy_normalized))
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(self, "latency_s", _finite_nonnegative(self.latency_s, "latency_s"))
        for name in ("stale", "accepted", "timeout"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(self, "request_id", _optional_routing_id(self.request_id, "request_id"))
        object.__setattr__(self, "step_id", _optional_routing_id(self.step_id, "step_id"))
        semantic_source = _bounded_text(
            self.semantic_source,
            "semantic_source",
            maximum=64,
        )
        if semantic_source != "qwen_vl":
            raise ValueError("semantic_source must be qwen_vl")
        geometry_source = _bounded_text(
            self.geometry_source,
            "geometry_source",
            maximum=64,
        )
        if geometry_source not in {"none", "oracle_evaluation"}:
            raise ValueError(
                "geometry_source must be none or oracle_evaluation"
            )
        object.__setattr__(self, "semantic_source", semantic_source)
        object.__setattr__(self, "geometry_source", geometry_source)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "request_id": self.request_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "step_id": self.step_id,
            "frame_id": self.frame_id,
            "observation_timestamp_s": self.observation_timestamp_s,
            "decision": self.decision,
            "semantic_source": self.semantic_source,
            "geometry_source": self.geometry_source,
            "bbox_xyxy_normalized": (
                None if self.bbox_xyxy_normalized is None else list(self.bbox_xyxy_normalized)
            ),
            "token_usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "latency_s": self.latency_s,
            "stale": self.stale,
            "accepted": self.accepted,
            "timeout": self.timeout,
        }


@dataclass(frozen=True, slots=True)
class MissionEventLogRecord:
    timestamp_s: float
    mission_id: str
    uav_id: str
    plan_version: int
    step_id: str
    skill: str
    event: str
    status: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite_nonnegative(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _positive_plan_version(self.plan_version))
        object.__setattr__(self, "step_id", validate_routing_id(self.step_id, "step_id"))
        for name in ("skill", "event", "status", "reason"):
            object.__setattr__(self, name, _bounded_text(getattr(self, name), name))

    def to_csv_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _MISSION_EVENT_FIELDS}

    def to_terminal_line(self) -> str:
        """Render the mandatory routing context for one compact terminal line."""

        return (
            f"[MISSION] mission_id={self.mission_id} uav_id={self.uav_id} "
            f"plan_version={self.plan_version} step_id={self.step_id} "
            f"skill={self.skill} event={self.event} status={self.status} "
            f"reason={self.reason}"
        )


@dataclass(frozen=True, slots=True)
class SkillTransitionLogRecord:
    timestamp_s: float
    mission_id: str
    uav_id: str
    plan_version: int
    step_id: str
    old_skill: str
    new_skill: str
    old_status: str
    result_code: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite_nonnegative(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _positive_plan_version(self.plan_version))
        object.__setattr__(self, "step_id", validate_routing_id(self.step_id, "step_id"))
        for name in ("old_skill", "new_skill", "old_status", "result_code", "reason"):
            object.__setattr__(self, name, _bounded_text(getattr(self, name), name))

    def to_csv_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _SKILL_TRANSITION_FIELDS}


@dataclass(frozen=True, slots=True)
class VisualRunStats:
    review_count: int
    accepted_count: int
    stale_count: int
    timeout_count: int
    plan_revision_count: int
    hover_count: int
    hover_total_time_s: float
    dropped_log_record_count: int

    def to_manifest_dict(
        self,
        *,
        debug_images_count: int = 0,
        debug_images_bytes: int = 0,
    ) -> dict[str, object]:
        return {
            "qwen_visual_reviews": {
                "count": self.review_count,
                "accepted": self.accepted_count,
                "stale": self.stale_count,
                "timeout": self.timeout_count,
            },
            "plan_revisions": self.plan_revision_count,
            "supervisory_hover": {
                "count": self.hover_count,
                "total_time_s": self.hover_total_time_s,
            },
            "debug_images": {
                "count": _nonnegative_int(debug_images_count, "debug_images_count"),
                "bytes": _nonnegative_int(debug_images_bytes, "debug_images_bytes"),
            },
            "dropped_log_records": self.dropped_log_record_count,
        }


_MISSION_EVENT_FIELDS = (
    "timestamp_s",
    "mission_id",
    "uav_id",
    "plan_version",
    "step_id",
    "skill",
    "event",
    "status",
    "reason",
)
_SKILL_TRANSITION_FIELDS = (
    "timestamp_s",
    "mission_id",
    "uav_id",
    "plan_version",
    "step_id",
    "old_skill",
    "new_skill",
    "old_status",
    "result_code",
    "reason",
)


@dataclass(slots=True)
class _StreamBudget:
    path: Path
    stream: IO[str]
    record_count: int
    byte_count: int


class SparseMissionLogger:
    """Write three fixed-schema, immediately flushed and bounded streams."""

    def __init__(
        self,
        logs_dir: str | Path,
        *,
        max_records_per_stream: int = DEFAULT_MAX_RECORDS_PER_STREAM,
        max_bytes_per_stream: int = DEFAULT_MAX_BYTES_PER_STREAM,
    ) -> None:
        self._max_records = _nonnegative_int(max_records_per_stream, "max_records_per_stream")
        self._max_bytes = _nonnegative_int(max_bytes_per_stream, "max_bytes_per_stream")
        if self._max_records == 0 or self._max_bytes < MAX_RECORD_BYTES:
            raise ValueError(
                f"max_records_per_stream must be positive and max_bytes_per_stream at least {MAX_RECORD_BYTES}"
            )
        if self._max_records > 1_000_000 or self._max_bytes > 1_073_741_824:
            raise ValueError("log budgets exceed the trusted hard cap")
        directory = Path(logs_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise ValueError("logs_dir must be a directory")
        self._lock = RLock()
        self._closed = False
        self._review_count = 0
        self._accepted_count = 0
        self._stale_count = 0
        self._timeout_count = 0
        self._plan_revision_count = 0
        self._hover_count = 0
        self._hover_total_time_s = 0.0
        self._dropped = 0
        opened: list[_StreamBudget] = []
        try:
            self._qwen = self._open_stream(directory / "qwen_reviews.jsonl")
            opened.append(self._qwen)
            if self._qwen.record_count > self._max_records:
                raise ValueError(
                    f"existing log exceeds configured record budget: {self._qwen.path}"
                )
            self._mission = self._open_stream(directory / "mission_events.csv")
            opened.append(self._mission)
            self._transitions = self._open_stream(directory / "skill_transitions.csv")
            opened.append(self._transitions)
        except Exception:
            for budget in opened:
                budget.stream.close()
            raise
        self._mission_writer = csv.DictWriter(self._mission.stream, fieldnames=_MISSION_EVENT_FIELDS)
        self._transition_writer = csv.DictWriter(
            self._transitions.stream, fieldnames=_SKILL_TRANSITION_FIELDS
        )
        self._ensure_csv_header(self._mission, self._mission_writer)
        self._ensure_csv_header(self._transitions, self._transition_writer)

    @staticmethod
    def _count_existing_records(path: Path) -> int:
        if not path.exists() or path.stat().st_size == 0:
            return 0
        with path.open("r", encoding="utf-8", newline="") as stream:
            return sum(1 for _ in stream)

    def _open_stream(self, path: Path) -> _StreamBudget:
        byte_count = path.stat().st_size if path.exists() else 0
        if byte_count > self._max_bytes:
            raise ValueError(f"existing log exceeds configured byte budget: {path}")
        lines = self._count_existing_records(path)
        if lines > self._max_records + 1:
            raise ValueError(f"existing log exceeds configured record budget: {path}")
        stream = path.open("a", encoding="utf-8", newline="")
        return _StreamBudget(path, stream, lines, byte_count)

    def _ensure_csv_header(self, budget: _StreamBudget, writer: csv.DictWriter) -> None:
        if budget.byte_count != 0:
            # Existing header is validated to prevent schema mixing on resume.
            with budget.path.open("r", encoding="utf-8", newline="") as stream:
                first = stream.readline().rstrip("\r\n")
            if first != ",".join(writer.fieldnames):
                self.close()
                raise ValueError(f"existing CSV header does not match schema: {budget.path}")
            budget.record_count = max(0, budget.record_count - 1)
            return
        rendered = ",".join(writer.fieldnames) + "\r\n"
        encoded_size = len(rendered.encode("utf-8"))
        if encoded_size > self._max_bytes:
            self.close()
            raise ValueError("CSV header exceeds byte budget")
        writer.writeheader()
        budget.stream.flush()
        budget.byte_count += encoded_size
        budget.record_count = 0

    def _can_append(self, budget: _StreamBudget, encoded_size: int) -> bool:
        if encoded_size > MAX_RECORD_BYTES:
            raise ValueError(f"log record exceeds {MAX_RECORD_BYTES} byte limit")
        if (
            budget.record_count >= self._max_records
            or budget.byte_count + encoded_size > self._max_bytes
        ):
            self._dropped += 1
            return False
        return True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SparseMissionLogger is closed")

    def log_qwen_review(self, record: QwenReviewLogRecord) -> bool:
        if not isinstance(record, QwenReviewLogRecord):
            raise TypeError("record must be QwenReviewLogRecord")
        rendered = json.dumps(
            record.to_json_dict(), ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ) + "\n"
        size = len(rendered.encode("utf-8"))
        with self._lock:
            self._require_open()
            # Manifest counters describe reviews that occurred, including a
            # record dropped by the bounded persistence budget.
            self._review_count += 1
            self._accepted_count += int(record.accepted)
            self._stale_count += int(record.stale)
            self._timeout_count += int(record.timeout)
            if not self._can_append(self._qwen, size):
                return False
            self._qwen.stream.write(rendered)
            self._qwen.stream.flush()
            self._qwen.record_count += 1
            self._qwen.byte_count += size
            return True

    def _log_csv(
        self,
        budget: _StreamBudget,
        writer: csv.DictWriter,
        row: dict[str, object],
    ) -> bool:
        # Render once into an in-memory buffer so the byte budget is checked
        # before touching the persistent file.
        import io

        buffer = io.StringIO(newline="")
        csv.DictWriter(buffer, fieldnames=writer.fieldnames).writerow(row)
        rendered = buffer.getvalue()
        size = len(rendered.encode("utf-8"))
        if not self._can_append(budget, size):
            return False
        budget.stream.write(rendered)
        budget.stream.flush()
        budget.record_count += 1
        budget.byte_count += size
        return True

    def log_mission_event(self, record: MissionEventLogRecord) -> bool:
        if not isinstance(record, MissionEventLogRecord):
            raise TypeError("record must be MissionEventLogRecord")
        with self._lock:
            self._require_open()
            return self._log_csv(
                self._mission, self._mission_writer, record.to_csv_dict()
            )

    def log_skill_transition(self, record: SkillTransitionLogRecord) -> bool:
        if not isinstance(record, SkillTransitionLogRecord):
            raise TypeError("record must be SkillTransitionLogRecord")
        with self._lock:
            self._require_open()
            return self._log_csv(
                self._transitions,
                self._transition_writer,
                record.to_csv_dict(),
            )

    def record_plan_revision(self) -> None:
        with self._lock:
            self._require_open()
            self._plan_revision_count += 1

    def record_supervisory_hover(self, duration_s: float) -> None:
        """Compatibility helper for one already-completed HOVER interval."""

        duration = _finite_nonnegative(duration_s, "duration_s")
        with self._lock:
            self._require_open()
            self._hover_count += 1
            self._hover_total_time_s += duration

    def record_supervisory_hover_started(self) -> None:
        """Count HOVER immediately so an interrupted run is still truthful."""

        with self._lock:
            self._require_open()
            self._hover_count += 1

    def record_supervisory_hover_duration(self, duration_s: float) -> None:
        """Add elapsed time when a previously counted HOVER interval exits."""

        duration = _finite_nonnegative(duration_s, "duration_s")
        with self._lock:
            self._require_open()
            self._hover_total_time_s += duration

    def snapshot(self) -> VisualRunStats:
        with self._lock:
            return VisualRunStats(
                review_count=self._review_count,
                accepted_count=self._accepted_count,
                stale_count=self._stale_count,
                timeout_count=self._timeout_count,
                plan_revision_count=self._plan_revision_count,
                hover_count=self._hover_count,
                hover_total_time_s=self._hover_total_time_s,
                dropped_log_record_count=self._dropped,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for budget in (self._qwen, self._mission, self._transitions):
                budget.stream.flush()
                budget.stream.close()
            self._closed = True

    def __enter__(self) -> "SparseMissionLogger":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "DEFAULT_MAX_BYTES_PER_STREAM",
    "DEFAULT_MAX_RECORDS_PER_STREAM",
    "MAX_RECORD_BYTES",
    "MissionEventLogRecord",
    "QwenReviewLogRecord",
    "SkillTransitionLogRecord",
    "SparseMissionLogger",
    "VisualRunStats",
]
