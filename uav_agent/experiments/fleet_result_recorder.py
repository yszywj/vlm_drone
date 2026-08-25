"""Complete but hard-bounded scalar results for one fleet mission.

The recorder is intentionally independent of Isaac Sim.  Callers may feed it
60 Hz trusted state, but it persists only the configured (1 Hz by default)
samples and updates path/time metrics with streaming accumulators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field, is_dataclass
import io
import json
from math import dist, isfinite
import os
from pathlib import Path
from threading import RLock
from typing import Any

from common.ids import validate_uav_id
from fleet.strict_json import strict_json_object_loads
from .planning_audit_logger import PlanningAuditLogger, sanitize_persisted_payload
from .schemas import (
    AGENT_METRIC_FIELDS,
    ATTRIBUTE_EVIDENCE_FIELDS,
    FAILURE_CASE_FIELDS,
    FLEET_METRIC_FIELDS,
    GOAL_METRIC_FIELDS,
    SKILL_EXECUTION_FIELDS,
    STATE_SAMPLE_FIELDS,
    TARGET_PERCEPTION_TRANSITION_FIELDS,
    AgentMetricRecord,
    AttributeEvidenceRecord,
    GoalResultRecord,
    SkillExecutionRecord,
    StateSampleRecord,
    TargetPerceptionTransitionRecord,
)


DEFAULT_STATE_SAMPLE_HZ = 1.0
DEFAULT_MAX_RECORD_BYTES = 32_768
DEFAULT_MAX_STREAM_BYTES = 8_388_608
DEFAULT_MAX_RUN_BYTES = 33_554_432
_SUMMARY_RESERVE_BYTES = 32_768
_FLEET_LOG_STORAGE_SIDECAR = ".fleet_logger_storage.json"
_MAX_FLEET_LOG_SIDECAR_BYTES = 4096

FORBIDDEN_RESULT_DIRECTORIES = frozenset(
    {
        "camera_images",
        "images",
        "videos",
        "raw_frames",
        "frames",
        "observation_dumps",
        "observations",
    }
)
FORBIDDEN_RESULT_SUFFIXES = frozenset(
    {".avi", ".bmp", ".gif", ".jpeg", ".jpg", ".mkv", ".mov", ".mp4", ".webm"}
)


class ResultRecorderError(RuntimeError):
    """Base class for result-contract errors (not storage exhaustion)."""


@dataclass(frozen=True, slots=True)
class ResultStorageSnapshot:
    dropped_records: Mapping[str, int]
    truncated_streams: tuple[str, ...]
    final_run_bytes: int
    state_samples_skipped_by_cadence: int = 0
    fleet_logger_dropped_records: Mapping[str, int] = field(default_factory=dict)
    fleet_logger_truncated_streams: tuple[str, ...] = ()
    fleet_logger_drop_generation: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "dropped_records": dict(sorted(self.dropped_records.items())),
            "dropped_record_count": sum(self.dropped_records.values()),
            "truncated_streams": list(self.truncated_streams),
            "final_run_bytes": self.final_run_bytes,
            "state_samples_skipped_by_cadence": self.state_samples_skipped_by_cadence,
            "fleet_logger_dropped_records": dict(
                sorted(self.fleet_logger_dropped_records.items())
            ),
            "fleet_logger_truncated_streams": list(
                self.fleet_logger_truncated_streams
            ),
            "fleet_logger_drop_generation": self.fleet_logger_drop_generation,
        }


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except FileNotFoundError:  # another logger atomically replaced it
                continue
    return total


def _read_fleet_logger_storage_state(
    run_dir: Path,
) -> tuple[dict[str, int], tuple[str, ...], int]:
    """Read the small scalar-only sidecar shared across logger re-attaches."""

    path = run_dir / _FLEET_LOG_STORAGE_SIDECAR
    if not path.exists() or path.is_symlink() or not path.is_file():
        return {}, (), 0
    try:
        if path.stat().st_size > _MAX_FLEET_LOG_SIDECAR_BYTES:
            return {}, (), 0
        payload = strict_json_object_loads(path.read_text(encoding="utf-8"))
        if set(payload) != {
            "schema_version",
            "generation",
            "dropped_records",
            "untracked_dropped_record_count",
            "truncated_streams",
        } or payload.get("schema_version") != 1:
            return {}, (), 0
        dropped_raw = payload.get("dropped_records", {})
        truncated_raw = payload.get("truncated_streams", [])
        generation = payload.get("generation", 0)
        untracked = payload.get("untracked_dropped_record_count", 0)
        if (
            not isinstance(dropped_raw, Mapping)
            or not isinstance(truncated_raw, list)
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
            or isinstance(untracked, bool)
            or not isinstance(untracked, int)
            or untracked < 0
        ):
            return {}, (), 0
        dropped = {
            str(name): count
            for name, count in dropped_raw.items()
            if isinstance(name, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
        }
        if untracked:
            dropped["__additional_fleet_log_streams__"] = (
                dropped.get("__additional_fleet_log_streams__", 0) + untracked
            )
        truncated = tuple(sorted(str(item) for item in truncated_raw if isinstance(item, str)))
        return dropped, truncated, generation
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # Result finalization remains best effort; a malformed diagnostic
        # sidecar cannot become a flight-control failure.
        return {}, (), 0


def _validate_limit(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized) or (nonnegative and normalized < 0.0):
        raise ValueError(f"{name} must be finite" + (" and non-negative" if nonnegative else ""))
    return normalized


def _mapping(value: Mapping[str, object] | object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        candidate = value.to_dict() if hasattr(value, "to_dict") else vars(value)
        return dict(candidate)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        candidate = value.to_dict()
        if isinstance(candidate, Mapping):
            return dict(candidate)
    try:
        return dict(vars(value))
    except TypeError as exc:
        raise TypeError("record must be a mapping or typed result record") from exc


def _csv_scalar(value: object) -> object:
    safe = sanitize_persisted_payload(value)
    if safe is None:
        return ""
    if isinstance(safe, (str, bool, int, float)):
        return safe
    if isinstance(safe, list) and all(isinstance(item, (str, bool, int, float)) for item in safe):
        return "|".join(str(item) for item in safe)
    raise TypeError("fleet CSV values must be scalar or a sequence of scalars")


class _BoundedResultStore:
    """Atomic JSON and append-only CSV/JSONL writes under three byte caps."""

    def __init__(
        self,
        run_dir: Path,
        *,
        max_record_bytes: int,
        max_stream_bytes: int,
        max_run_bytes: int,
    ) -> None:
        self.run_dir = run_dir
        self.max_record_bytes = _validate_limit(max_record_bytes, "max_record_bytes")
        self.max_stream_bytes = _validate_limit(max_stream_bytes, "max_stream_bytes")
        self.max_run_bytes = _validate_limit(max_run_bytes, "max_run_bytes")
        if self.max_record_bytes > self.max_stream_bytes:
            raise ValueError("max_record_bytes must not exceed max_stream_bytes")
        if self.max_stream_bytes > self.max_run_bytes:
            raise ValueError("max_stream_bytes must not exceed max_run_bytes")
        self.dropped_records: dict[str, int] = {}
        self.truncated_streams: set[str] = set()
        # No other "high priority" result may consume the final summary's
        # space.  The summary is the bounded record that explains every drop.
        self.summary_reserve_bytes = min(
            _SUMMARY_RESERVE_BYTES,
            max(1, self.max_run_bytes // 3),
        )
        self._lock = RLock()

    @staticmethod
    def _stream_name(value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("result stream must be a relative path below run_dir")
        if any(part.casefold() in FORBIDDEN_RESULT_DIRECTORIES for part in path.parts):
            raise ValueError("forbidden image/video/observation result stream")
        if path.suffix.casefold() in FORBIDDEN_RESULT_SUFFIXES:
            raise ValueError("image and video result files are forbidden")
        return path.as_posix()

    def _drop(self, stream_name: str) -> bool:
        self.dropped_records[stream_name] = self.dropped_records.get(stream_name, 0) + 1
        self.truncated_streams.add(stream_name)
        return False

    def _can_write(
        self,
        stream_name: str,
        encoded_size: int,
        *,
        replacing_bytes: int = 0,
        high_priority: bool,
    ) -> bool:
        is_summary = stream_name == "summary.json"
        record_limit = self.summary_reserve_bytes if is_summary else self.max_record_bytes
        if encoded_size > record_limit:
            return self._drop(stream_name)
        path = self.run_dir / stream_name
        current_stream_bytes = path.stat().st_size if path.exists() else 0
        projected_stream = encoded_size if replacing_bytes else current_stream_bytes + encoded_size
        stream_limit = max(self.max_stream_bytes, self.summary_reserve_bytes) if is_summary else self.max_stream_bytes
        if projected_stream > stream_limit:
            return self._drop(stream_name)
        # ``high_priority`` controls stream semantics, not permission to steal
        # the terminal summary reserve. Only summary.json may use that space.
        del high_priority
        reserve = 0 if is_summary else self.summary_reserve_bytes
        projected_run = _directory_size(self.run_dir) - replacing_bytes + encoded_size
        if projected_run > self.max_run_bytes - reserve:
            return self._drop(stream_name)
        return True

    def ensure_csv(self, stream_name: str, fields: Sequence[str]) -> bool:
        name = self._stream_name(stream_name)
        path = self.run_dir / name
        with self._lock:
            if path.exists() and path.stat().st_size:
                with path.open("r", encoding="utf-8", newline="") as stream:
                    header = tuple(next(csv.reader(stream), ()))
                if header != tuple(fields):
                    raise ResultRecorderError(f"existing CSV schema mismatch: {path}")
                return True
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\n").writerow(tuple(fields))
            encoded = buffer.getvalue().encode("utf-8")
            if not self._can_write(name, len(encoded), high_priority=True):
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
            return True

    def append_csv(
        self,
        stream_name: str,
        fields: Sequence[str],
        value: Mapping[str, object],
        *,
        high_priority: bool = False,
    ) -> bool:
        name = self._stream_name(stream_name)
        with self._lock:
            if not self.ensure_csv(name, fields):
                return self._drop(name)
            row = {field: _csv_scalar(value.get(field)) for field in fields}
            buffer = io.StringIO(newline="")
            csv.DictWriter(buffer, fieldnames=tuple(fields), lineterminator="\n").writerow(row)
            encoded = buffer.getvalue().encode("utf-8")
            if not self._can_write(name, len(encoded), high_priority=high_priority):
                return False
            path = self.run_dir / name
            with path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
            return True

    def append_jsonl(
        self,
        stream_name: str,
        value: Mapping[str, object],
        *,
        high_priority: bool = False,
    ) -> bool:
        name = self._stream_name(stream_name)
        safe = sanitize_persisted_payload(value)
        assert isinstance(safe, Mapping)
        encoded = (
            json.dumps(safe, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with self._lock:
            if not self._can_write(name, len(encoded), high_priority=high_priority):
                return False
            path = self.run_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
            return True

    def write_json(
        self,
        stream_name: str,
        value: Mapping[str, object],
        *,
        high_priority: bool = True,
    ) -> bool:
        name = self._stream_name(stream_name)
        safe = sanitize_persisted_payload(value)
        assert isinstance(safe, Mapping)
        encoded = (
            json.dumps(safe, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        path = self.run_dir / name
        with self._lock:
            old_size = path.stat().st_size if path.exists() else 0
            if not self._can_write(
                name,
                len(encoded),
                replacing_bytes=old_size,
                high_priority=high_priority,
            ):
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                with temporary.open("wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return True


@dataclass(slots=True)
class _AgentAccumulator:
    last_timestamp_s: float | None = None
    last_position_xyz_m: tuple[float, float, float] | None = None
    last_mode: str = "UNKNOWN"
    path_length_m: float = 0.0
    airborne_time_s: float = 0.0
    hover_time_s: float = 0.0
    hold_time_s: float = 0.0
    time_to_first_detection_s: float | None = None
    time_to_first_lock_s: float | None = None
    minimum_inter_uav_distance_m: float | None = None

    def update(self, sample: StateSampleRecord) -> None:
        timestamp = _finite(sample.timestamp_s, "timestamp_s", nonnegative=True)
        position = tuple(_finite(item, "position_xyz_m") for item in sample.position_xyz_m)
        if len(position) != 3:
            raise ValueError("position_xyz_m must contain exactly three numbers")
        if self.last_timestamp_s is not None:
            if timestamp < self.last_timestamp_s:
                raise ValueError("state timestamps must be monotonic per UAV")
            elapsed = timestamp - self.last_timestamp_s
            mode = self.last_mode.upper()
            if mode not in {"LANDED", "GROUND", "PENDING", "READY"}:
                self.airborne_time_s += elapsed
            if "HOVER" in mode:
                self.hover_time_s += elapsed
            if "HOLD" in mode:
                self.hold_time_s += elapsed
        if self.last_position_xyz_m is not None:
            self.path_length_m += dist(self.last_position_xyz_m, position)
        if sample.target_detected and self.time_to_first_detection_s is None:
            self.time_to_first_detection_s = timestamp
        if sample.target_locked and self.time_to_first_lock_s is None:
            self.time_to_first_lock_s = timestamp
        if sample.minimum_inter_uav_distance_m is not None:
            separation = _finite(
                sample.minimum_inter_uav_distance_m,
                "minimum_inter_uav_distance_m",
                nonnegative=True,
            )
            self.minimum_inter_uav_distance_m = (
                separation
                if self.minimum_inter_uav_distance_m is None
                else min(self.minimum_inter_uav_distance_m, separation)
            )
        self.last_timestamp_s = timestamp
        self.last_position_xyz_m = position  # type: ignore[assignment]
        self.last_mode = str(sample.mode)


class FleetResultRecorder:
    """Own all bounded result streams for one mission.

    Storage exhaustion is expressed as ``False`` from a logging method and in
    :meth:`storage_snapshot`; it never raises a mission-control exception.
    Contract violations (for example an RGB payload) still fail closed.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        config: object | None = None,
        fleet_mission_id: str | None = None,
        state_sample_hz: float | None = None,
        max_record_bytes: int | None = None,
        max_stream_bytes: int | None = None,
        max_run_bytes: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.run_dir / "metrics"
        self._reject_forbidden_existing_artifacts()
        results_config = getattr(config, "results", config)

        def configured(name: str, explicit: object, default: object) -> object:
            if explicit is not None:
                return explicit
            return getattr(results_config, name, default) if results_config is not None else default

        self.state_sample_hz = _finite(
            configured("state_sample_hz", state_sample_hz, DEFAULT_STATE_SAMPLE_HZ),
            "state_sample_hz",
        )
        if self.state_sample_hz <= 0.0 or self.state_sample_hz > 10.0:
            raise ValueError("state_sample_hz must be within (0, 10]")
        self.fleet_mission_id = fleet_mission_id
        self._store = _BoundedResultStore(
            self.run_dir,
            max_record_bytes=configured("max_record_bytes", max_record_bytes, DEFAULT_MAX_RECORD_BYTES),
            max_stream_bytes=configured("max_stream_bytes", max_stream_bytes, DEFAULT_MAX_STREAM_BYTES),
            max_run_bytes=configured("max_run_bytes", max_run_bytes, DEFAULT_MAX_RUN_BYTES),
        )
        self.planning_audit = PlanningAuditLogger(self.run_dir, sink=self._store)
        self._last_persisted_state_s: dict[str, float] = {}
        self._agent_accumulators: dict[str, _AgentAccumulator] = {}
        self._state_skipped = 0
        self._collision_active = False
        self._collision_count = 0
        self._out_of_bounds_uavs: set[str] = set()
        self._emergency_landing_uavs: set[str] = set()
        self._fleet_rows: list[dict[str, object]] = []
        self._agent_rows: list[dict[str, object]] = []
        self._goal_rows: list[dict[str, object]] = []
        self._skill_rows: list[dict[str, object]] = []
        self._explicit_agent_ids: set[str] = set()
        self._closed = False
        for stream, fields in (
            ("metrics/fleet_metrics.csv", FLEET_METRIC_FIELDS),
            ("metrics/agent_metrics.csv", AGENT_METRIC_FIELDS),
            ("metrics/goal_metrics.csv", GOAL_METRIC_FIELDS),
            ("metrics/skill_executions.csv", SKILL_EXECUTION_FIELDS),
            ("metrics/state_samples_1hz.csv", STATE_SAMPLE_FIELDS),
            ("metrics/failure_cases.csv", FAILURE_CASE_FIELDS),
        ):
            self._store.ensure_csv(stream, fields)

    def _reject_forbidden_existing_artifacts(self) -> None:
        for path in self.run_dir.rglob("*"):
            if path.is_dir() and path.name.casefold() in FORBIDDEN_RESULT_DIRECTORIES:
                raise ResultRecorderError(f"forbidden result directory exists: {path.name}")
            if path.is_file() and path.suffix.casefold() in FORBIDDEN_RESULT_SUFFIXES:
                raise ResultRecorderError(f"forbidden image/video result exists: {path.name}")

    def _ensure_open(self) -> None:
        if self._closed:
            raise ResultRecorderError("FleetResultRecorder is closed")

    def record_fleet_metrics(self, metrics: Mapping[str, object] | object) -> bool:
        self._ensure_open()
        values = _mapping(metrics)
        values.setdefault("schema_version", 1)
        if self.fleet_mission_id:
            values.setdefault("fleet_mission_id", self.fleet_mission_id)
        persisted = self._store.append_csv(
            "metrics/fleet_metrics.csv", FLEET_METRIC_FIELDS, values, high_priority=True
        )
        if persisted:
            self._fleet_rows.append(values)
        return persisted

    log_fleet_metrics = record_fleet_metrics

    def record_agent_metrics(self, record: AgentMetricRecord | Mapping[str, object]) -> bool:
        self._ensure_open()
        values = _mapping(record)
        values.setdefault("schema_version", 1)
        persisted = self._store.append_csv("metrics/agent_metrics.csv", AGENT_METRIC_FIELDS, values)
        if persisted and values.get("uav_id"):
            self._explicit_agent_ids.add(str(values["uav_id"]))
            self._agent_rows.append(values)
        return persisted

    log_agent_metrics = record_agent_metrics

    def latest_agent_metric_rows(self) -> Mapping[str, Mapping[str, object]]:
        """Return the newest scalar row for each UAV."""

        latest: dict[str, Mapping[str, object]] = {}
        for row in self._agent_rows:
            uav_id = row.get("uav_id")
            if isinstance(uav_id, str) and uav_id:
                latest[uav_id] = dict(row)
        return latest

    def record_goal_result(self, record: GoalResultRecord | Mapping[str, object]) -> bool:
        self._ensure_open()
        values = _mapping(record)
        values.setdefault("schema_version", 1)
        persisted = self._store.append_csv("metrics/goal_metrics.csv", GOAL_METRIC_FIELDS, values)
        if persisted:
            self._goal_rows.append(values)
        return persisted

    log_goal_result = record_goal_result

    def record_skill_execution(
        self, record: SkillExecutionRecord | Mapping[str, object]
    ) -> bool:
        self._ensure_open()
        values = _mapping(record)
        values.setdefault("schema_version", 1)
        if "duration_s" not in values and values.get("start_time_s") is not None and values.get("end_time_s") is not None:
            values["duration_s"] = max(
                0.0,
                _finite(values["end_time_s"], "end_time_s")
                - _finite(values["start_time_s"], "start_time_s"),
            )
        persisted = self._store.append_csv(
            "metrics/skill_executions.csv", SKILL_EXECUTION_FIELDS, values
        )
        if persisted:
            self._skill_rows.append(values)
        return persisted

    log_skill_execution = record_skill_execution

    def record_state_sample(self, record: StateSampleRecord) -> bool:
        self._ensure_open()
        if not isinstance(record, StateSampleRecord):
            raise TypeError("record must be StateSampleRecord")
        accumulator = self._agent_accumulators.setdefault(record.uav_id, _AgentAccumulator())
        accumulator.update(record)
        timestamp = float(record.timestamp_s)
        previous = self._last_persisted_state_s.get(record.uav_id)
        period = 1.0 / self.state_sample_hz
        if previous is not None and timestamp + 1e-9 < previous + period:
            self._state_skipped += 1
            return False
        persisted = self._store.append_csv(
            "metrics/state_samples_1hz.csv", STATE_SAMPLE_FIELDS, record.to_dict()
        )
        if persisted:
            self._last_persisted_state_s[record.uav_id] = timestamp
        return persisted

    sample_state = record_state_sample
    log_state_sample = record_state_sample

    def observe_safety_snapshot(
        self,
        *,
        collision: bool = False,
        out_of_bounds_uav_ids: Sequence[str] = (),
        emergency_landing_uav_ids: Sequence[str] = (),
    ) -> None:
        """Accumulate trusted scalar safety facts without persisting raw state."""

        self._ensure_open()
        if not isinstance(collision, bool):
            raise TypeError("collision must be bool")
        # Count collision episodes, not frames.  The Fleet loop samples a
        # separation breach every simulation tick, so summing ``True`` values
        # would turn one sustained contact into hundreds of collisions while a
        # latched boolean could never represent a later, distinct contact.
        if collision and not self._collision_active:
            self._collision_count += 1
        self._collision_active = collision
        for values, destination, name in (
            (out_of_bounds_uav_ids, self._out_of_bounds_uavs, "out_of_bounds_uav_ids"),
            (
                emergency_landing_uav_ids,
                self._emergency_landing_uavs,
                "emergency_landing_uav_ids",
            ),
        ):
            if isinstance(values, (str, bytes)):
                raise TypeError(f"{name} must be a sequence of UAV IDs")
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must contain non-empty strings")
                destination.add(value.strip())

    @property
    def collision_count(self) -> int:
        return self._collision_count

    @property
    def out_of_bounds_count(self) -> int:
        return len(self._out_of_bounds_uavs)

    @property
    def emergency_landing_count(self) -> int:
        return len(self._emergency_landing_uavs)

    def record_failure(self, failure: Mapping[str, object] | object) -> bool:
        self._ensure_open()
        values = _mapping(failure)
        if self.fleet_mission_id:
            values.setdefault("fleet_mission_id", self.fleet_mission_id)
            values.setdefault("run_id", self.fleet_mission_id)
        return self._store.append_csv(
            "metrics/failure_cases.csv", FAILURE_CASE_FIELDS, values, high_priority=True
        )

    log_failure = record_failure

    def record_attribute_evidence(
        self,
        uav_id: str,
        evidence: Mapping[str, object] | object,
        *,
        target_perception_mode: str,
    ) -> bool:
        """Persist one scalar-only YOLO attribute decision.

        Oracle callers are rejected rather than producing a misleading empty
        production stream.  Unknown fields are also rejected so image/crop or
        depth-plane payloads cannot be smuggled into this dedicated record.
        """

        self._ensure_open()
        normalized_uav = validate_uav_id(uav_id)
        if target_perception_mode != "yolo":
            raise ResultRecorderError(
                "attribute evidence is applicable only to target_perception_mode=yolo"
            )
        values = _mapping(evidence)
        values.setdefault("schema_version", 1)
        record = AttributeEvidenceRecord.from_mapping(values)
        if record.uav_id != normalized_uav:
            raise ValueError("attribute evidence uav_id does not match stream route")
        values = record.to_dict()
        ordered = {
            name: values[name]
            for name in ATTRIBUTE_EVIDENCE_FIELDS
            if name in values and values[name] is not None
        }
        return self._store.append_jsonl(
            f"agents/{normalized_uav}/attribute_evidence.jsonl",
            ordered,
        )

    log_attribute_evidence = record_attribute_evidence

    def record_target_perception_transition(
        self,
        uav_id: str,
        transition: Mapping[str, object] | object,
        *,
        target_perception_mode: str,
    ) -> bool:
        """Persist one throttled, scalar-only YOLO candidate lifecycle edge."""

        self._ensure_open()
        normalized_uav = validate_uav_id(uav_id)
        if target_perception_mode != "yolo":
            raise ResultRecorderError(
                "target perception transitions are applicable only to "
                "target_perception_mode=yolo"
            )
        values = _mapping(transition)
        values.setdefault("schema_version", 1)
        record = TargetPerceptionTransitionRecord.from_mapping(values)
        if record.uav_id != normalized_uav:
            raise ValueError(
                "target perception transition uav_id does not match stream route"
            )
        serialized = record.to_dict()
        ordered = {
            name: serialized[name]
            for name in TARGET_PERCEPTION_TRANSITION_FIELDS
        }
        return self._store.append_jsonl(
            f"agents/{normalized_uav}/target_perception_transitions.jsonl",
            ordered,
        )

    log_target_perception_transition = record_target_perception_transition

    def agent_metric_snapshots(
        self,
        *,
        status_by_uav: Mapping[str, str] | None = None,
    ) -> tuple[AgentMetricRecord, ...]:
        statuses = status_by_uav or {}
        records = []
        for uav_id, item in sorted(self._agent_accumulators.items()):
            records.append(
                AgentMetricRecord(
                    fleet_mission_id=self.fleet_mission_id or "unknown_mission",
                    uav_id=uav_id,
                    status=statuses.get(uav_id, "UNKNOWN"),
                    path_length_m=item.path_length_m,
                    airborne_time_s=item.airborne_time_s,
                    hover_time_s=item.hover_time_s,
                    hold_time_s=item.hold_time_s,
                    time_to_first_detection_s=item.time_to_first_detection_s,
                    time_to_first_lock_s=item.time_to_first_lock_s,
                )
            )
        return tuple(records)

    def storage_snapshot(self) -> ResultStorageSnapshot:
        fleet_dropped, fleet_truncated, fleet_generation = (
            _read_fleet_logger_storage_state(self.run_dir)
        )
        dropped = dict(self._store.dropped_records)
        for stream, count in fleet_dropped.items():
            dropped[stream] = dropped.get(stream, 0) + count
        return ResultStorageSnapshot(
            dropped_records=dropped,
            truncated_streams=tuple(
                sorted(self._store.truncated_streams.union(fleet_truncated))
            ),
            final_run_bytes=_directory_size(self.run_dir),
            state_samples_skipped_by_cadence=self._state_skipped,
            fleet_logger_dropped_records=fleet_dropped,
            fleet_logger_truncated_streams=fleet_truncated,
            fleet_logger_drop_generation=fleet_generation,
        )

    @property
    def latest_state_timestamp_s(self) -> float | None:
        values = (
            item.last_timestamp_s for item in self._agent_accumulators.values()
            if item.last_timestamp_s is not None
        )
        return max(values, default=None)

    @property
    def minimum_inter_uav_distance_m(self) -> float | None:
        values = (
            item.minimum_inter_uav_distance_m
            for item in self._agent_accumulators.values()
            if item.minimum_inter_uav_distance_m is not None
        )
        return min(values, default=None)

    def finalize(self, summary: Mapping[str, object]) -> dict[str, object]:
        """Write a consistent summary and terminal fleet-metric row.

        This method remains safe after any number of budget drops.  If a caller
        supplied no Fleet metric, a compact one is derived from the summary.
        """

        self._ensure_open()
        values = _mapping(summary)
        status = values.get("status", values.get("final_status", "UNKNOWN"))
        if not isinstance(status, str) or not status.strip():
            raise ValueError("summary requires a non-empty status")
        mission_id = values.get("fleet_mission_id", self.fleet_mission_id)
        if mission_id is not None:
            values["fleet_mission_id"] = mission_id
        latest_status = self._fleet_rows[-1].get("status") if self._fleet_rows else None
        if latest_status != status:
            goal_count = values.get("goal_count", len(self._goal_rows))
            goals_completed = values.get(
                "goals_completed",
                sum(bool(row.get("completed")) for row in self._goal_rows),
            )
            self.record_fleet_metrics(
                {
                    **values,
                    "status": status,
                    "goal_count": goal_count,
                    "goals_completed": goals_completed,
                    "goal_completion_rate": (
                        float(goals_completed) / float(goal_count)
                        if isinstance(goal_count, (int, float)) and goal_count
                        else 0.0
                    ),
                }
            )
        statuses = values.get("agent_statuses")
        status_by_uav = statuses if isinstance(statuses, Mapping) else {}
        for record in self.agent_metric_snapshots(status_by_uav=status_by_uav):
            if record.uav_id not in self._explicit_agent_ids:
                self.record_agent_metrics(record)

        # Iterate because the byte count embedded in summary.json can change
        # its own digit width.  Three passes are sufficient even at GiB sizes.
        for _ in range(3):
            storage = self.storage_snapshot().to_dict()
            values["result_storage"] = storage
            before = _directory_size(self.run_dir)
            previous_size = (self.run_dir / "summary.json").stat().st_size if (self.run_dir / "summary.json").exists() else 0
            rendered = (
                json.dumps(
                    sanitize_persisted_payload(values),
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            projected = before - previous_size + len(rendered)
            values["result_storage"]["final_run_bytes"] = projected  # type: ignore[index]
            if not self._store.write_json("summary.json", values, high_priority=True):
                # Capacity exhaustion must not escape into mission execution.
                # A compact summary is reserved for this exact condition.
                compact = {
                    "schema_version": 1,
                    "fleet_mission_id": mission_id,
                    "status": status,
                }
                for _ in range(3):
                    compact["result_storage"] = self.storage_snapshot().to_dict()
                    previous_size = (
                        (self.run_dir / "summary.json").stat().st_size
                        if (self.run_dir / "summary.json").exists()
                        else 0
                    )
                    rendered = (
                        json.dumps(
                            sanitize_persisted_payload(compact),
                            ensure_ascii=False,
                            allow_nan=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode("utf-8")
                    projected = _directory_size(self.run_dir) - previous_size + len(rendered)
                    compact["result_storage"]["final_run_bytes"] = projected  # type: ignore[index]
                    if not self._store.write_json("summary.json", compact, high_priority=True):
                        break
                    if compact["result_storage"]["final_run_bytes"] == _directory_size(  # type: ignore[index]
                        self.run_dir
                    ):
                        break
                values = compact
                break
        return values

    write_summary = finalize

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "FleetResultRecorder":
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_MAX_RECORD_BYTES",
    "DEFAULT_MAX_RUN_BYTES",
    "DEFAULT_MAX_STREAM_BYTES",
    "DEFAULT_STATE_SAMPLE_HZ",
    "FORBIDDEN_RESULT_DIRECTORIES",
    "FORBIDDEN_RESULT_SUFFIXES",
    "FleetResultRecorder",
    "ResultRecorderError",
    "ResultStorageSnapshot",
]
