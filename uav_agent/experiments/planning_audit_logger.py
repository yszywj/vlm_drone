"""Bounded, prompt-free audit records for the three planning stages.

The public logger accepts typed records and intentionally has no API for image
or prompt persistence.  It can operate on its own, or share the byte-budget
store owned by :class:`experiments.fleet_result_recorder.FleetResultRecorder`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from threading import RLock
from typing import Protocol

from .schemas import (
    PlanningAttemptRecord,
    RecoveryActionRecord,
    ValidationFindingRecord,
)


AUDIT_STREAMS = (
    "planning_attempts.jsonl",
    "validation_findings.jsonl",
    "recovery_actions.jsonl",
    "final_plans.jsonl",
)

_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "bearer_token",
        "camera_image",
        "camera_images",
        "camera_rgb",
        "rgb",
        "image",
        "images",
        "image_url",
        "video",
        "videos",
        "frames",
        "raw_frame",
        "raw_frames",
        "observation",
        "observations",
        "pixels",
        "prompt",
        "full_prompt",
        "chain_of_thought",
        "hidden_reasoning",
        "reasoning_content",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)((?:api[_-]?key|authorization)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


class AuditSink(Protocol):
    def append_jsonl(
        self,
        stream_name: str,
        value: Mapping[str, object],
        *,
        high_priority: bool = False,
    ) -> bool: ...


def prompt_sha256(prompt: str) -> str:
    """Return a stable prompt digest without retaining the prompt itself."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def sanitize_text_tail(text: str, *, maximum: int = 500) -> tuple[int, str]:
    """Return only total length and a credential/base64-redacted suffix."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum <= 500:
        raise ValueError("maximum must be an integer within [0, 500]")
    tail = text[-maximum:] if maximum else ""
    for pattern in _SECRET_PATTERNS:
        tail = pattern.sub("[REDACTED_CREDENTIAL]", tail)
    # A data URL can be enormous.  Its suffix usually no longer contains the
    # prefix, so redact base64-looking runs as well as explicit data URLs.
    tail = re.sub(r"(?is)data:(?:image|video)/[^,;]+(?:;base64)?,[^\s]+", "[REDACTED_MEDIA]", tail)
    tail = re.sub(r"(?i)base64,[A-Za-z0-9+/=]+", "base64,[REDACTED]", tail)
    tail = re.sub(r"[A-Za-z0-9+/]{160,}={0,2}", "[REDACTED_BLOB]", tail)
    return len(text), tail


def sanitize_persisted_payload(value: object, path: str = "record") -> object:
    """Convert one result payload to finite JSON and reject forbidden data."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{path} keys must be strings")
            key = raw_key.casefold().strip()
            if key in _FORBIDDEN_KEYS:
                raise ValueError(f"{path}.{raw_key} is forbidden in persisted results")
            result[raw_key] = sanitize_persisted_payload(nested, f"{path}.{raw_key}")
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_persisted_payload(item, f"{path}[]") for item in value]
    if isinstance(value, Enum):
        return sanitize_persisted_payload(value.value, path)
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, str):
            lowered = value.casefold()
            if "data:image/" in lowered or "data:video/" in lowered or "base64," in lowered:
                raise ValueError(f"{path} must not contain encoded media")
            if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
                raise ValueError(f"{path} must not contain credentials")
        return value
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


class _StandaloneAuditSink:
    """Small fallback store used when no FleetResultRecorder owns the budget."""

    def __init__(self, run_dir: Path, max_record_bytes: int, max_stream_bytes: int) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.max_record_bytes = max_record_bytes
        self.max_stream_bytes = max_stream_bytes
        self.dropped_records: dict[str, int] = {}
        self.truncated_streams: set[str] = set()
        self._lock = RLock()

    def append_jsonl(
        self,
        stream_name: str,
        value: Mapping[str, object],
        *,
        high_priority: bool = False,
    ) -> bool:
        del high_priority
        payload = sanitize_persisted_payload(value)
        assert isinstance(payload, Mapping)
        encoded = (
            json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        path = self.run_dir / stream_name
        with self._lock:
            current_size = path.stat().st_size if path.exists() else 0
            if len(encoded) > self.max_record_bytes or current_size + len(encoded) > self.max_stream_bytes:
                self.dropped_records[stream_name] = self.dropped_records.get(stream_name, 0) + 1
                self.truncated_streams.add(stream_name)
                return False
            with path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
        return True


class PlanningAuditLogger:
    """Persist accepted/repaired planner proposals without prompts or media."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        sink: AuditSink | None = None,
        max_record_bytes: int = 32_768,
        max_stream_bytes: int = 8_388_608,
    ) -> None:
        if max_record_bytes <= 0 or max_stream_bytes < max_record_bytes:
            raise ValueError("audit byte limits must be positive and ordered")
        self.run_dir = Path(run_dir).expanduser().resolve()
        self._sink: AuditSink = sink or _StandaloneAuditSink(
            self.run_dir,
            max_record_bytes,
            max_stream_bytes,
        )

    def log_attempt(self, record: PlanningAttemptRecord) -> bool:
        if not isinstance(record, PlanningAttemptRecord):
            raise TypeError("record must be PlanningAttemptRecord")
        return self._sink.append_jsonl("planning_attempts.jsonl", record.to_dict())

    # Alias used by planning coordinators that call every output an event.
    record_attempt = log_attempt

    def log_invalid_attempt(
        self,
        *,
        attempt_id: str,
        timestamp_s: float,
        stage: str,
        mission_id: str,
        model_role: str,
        prompt: str,
        prompt_schema_version: str,
        raw_text: str,
        error_codes: tuple[str, ...] = (),
        assignment_id: str | None = None,
        uav_id: str | None = None,
        repaired_from_attempt_id: str | None = None,
    ) -> bool:
        length, tail = sanitize_text_tail(raw_text)
        return self.log_attempt(
            PlanningAttemptRecord(
                attempt_id=attempt_id,
                timestamp_s=timestamp_s,
                stage=stage,
                mission_id=mission_id,
                assignment_id=assignment_id,
                uav_id=uav_id,
                model_role=model_role,
                prompt_sha256=prompt_sha256(prompt),
                prompt_schema_version=prompt_schema_version,
                accepted=False,
                repaired_from_attempt_id=repaired_from_attempt_id,
                error_codes=error_codes,
                raw_text_length=length,
                raw_text_tail=tail,
            )
        )

    def log_finding(self, record: ValidationFindingRecord) -> bool:
        if not isinstance(record, ValidationFindingRecord):
            raise TypeError("record must be ValidationFindingRecord")
        return self._sink.append_jsonl("validation_findings.jsonl", record.to_dict())

    record_finding = log_finding

    def log_recovery(self, record: RecoveryActionRecord) -> bool:
        if not isinstance(record, RecoveryActionRecord):
            raise TypeError("record must be RecoveryActionRecord")
        return self._sink.append_jsonl("recovery_actions.jsonl", record.to_dict())

    record_recovery = log_recovery

    def write_final_plan(
        self,
        *,
        stage: str,
        mission_id: str,
        plan_version: int,
        plan: Mapping[str, object],
        assignment_id: str | None = None,
        uav_id: str | None = None,
    ) -> bool:
        if isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version <= 0:
            raise ValueError("plan_version must be a positive integer")
        record: dict[str, object] = {
            "schema_version": 1,
            "stage": stage,
            "mission_id": mission_id,
            "assignment_id": assignment_id,
            "uav_id": uav_id,
            "plan_version": plan_version,
            "plan": plan,
        }
        return self._sink.append_jsonl("final_plans.jsonl", record, high_priority=True)


__all__ = [
    "AUDIT_STREAMS",
    "AuditSink",
    "PlanningAttemptRecord",
    "PlanningAuditLogger",
    "RecoveryActionRecord",
    "ValidationFindingRecord",
    "prompt_sha256",
    "sanitize_persisted_payload",
    "sanitize_text_tail",
]
