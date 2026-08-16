"""Fixed-schema CSV metrics and a dependency-free scalar TensorBoard writer."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import math
import os
from pathlib import Path
import socket
import struct
import time
from typing import Mapping, Sequence

from .evaluator import compute_mission_success_strict, parse_metric_bool
from .schemas import (
    EPISODE_METRIC_FIELDS,
    EVAL_METRIC_FIELDS,
    FAILURE_CASE_FIELDS,
    FINAL_METRIC_FIELDS,
    TRAIN_METRIC_FIELDS,
    FailureReason,
    MetricPhase,
)


class MetricLoggerError(RuntimeError):
    """Base class for metric persistence errors."""


class MetricSchemaError(MetricLoggerError, ValueError):
    """Raised when a record cannot be represented by a fixed CSV schema."""


class DuplicateEpisodeError(MetricLoggerError, ValueError):
    """Raised when one run/phase/episode tuple would be written twice."""


class FinalMetricsAlreadyLoggedError(MetricLoggerError, ValueError):
    """Raised when final metrics already contain their single allowed data row."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Mapping[str, object] | object | None) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    try:
        return dict(vars(value))
    except TypeError as exc:
        raise MetricSchemaError("metrics must be a mapping or dataclass") from exc


def _csv_scalar(value: object) -> object:
    """Normalize one CSV cell without manufacturing zero-valued measurements."""

    if value is None:
        return ""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MetricSchemaError("NaN and infinity are not valid metric values")
        return value
    if isinstance(value, (str, Path)):
        return str(value)
    raise MetricSchemaError(
        f"only scalar CSV values are allowed, got {type(value).__name__}"
    )


def _normalize_row(fields: Sequence[str], values: Mapping[str, object]) -> dict[str, object]:
    return {field: _csv_scalar(values.get(field)) for field in fields}


class _AppendCsv:
    """Append-only CSV with one validated header and explicit flushing."""

    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        self.path = path
        self.fields = tuple(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_existing_header()
        self._stream = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._stream,
            fieldnames=self.fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._stream.flush()

    def _validate_existing_header(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("r", encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), None)
        if tuple(header or ()) != self.fields:
            raise MetricSchemaError(
                f"existing CSV header does not match fixed schema: {self.path}"
            )

    def append(self, values: Mapping[str, object]) -> None:
        self._writer.writerow(_normalize_row(self.fields, values))

    def rows(self) -> list[dict[str, str]]:
        self._stream.flush()
        with self.path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()


# TensorFlow Event files are TFRecord streams of protobuf Event messages.  The
# tiny encoder below implements only the wire fields needed for file_version and
# Summary.Value(simple_value).  Therefore no image/video/histogram/graph API is
# present and tensorboard/protobuf are not runtime dependencies.


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("protobuf varint must be non-negative")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _length_delimited(field_number: int, payload: bytes) -> bytes:
    return _varint((field_number << 3) | 2) + _varint(len(payload)) + payload


def _event_file_version(wall_time: float) -> bytes:
    return (
        _varint((1 << 3) | 1)
        + struct.pack("<d", wall_time)
        + _length_delimited(3, b"brain.Event:2")
    )


def _event_scalar(wall_time: float, step: int, tag: str, value: float) -> bytes:
    summary_value = (
        _length_delimited(1, tag.encode("utf-8"))
        + _varint((2 << 3) | 5)
        + struct.pack("<f", value)
    )
    summary = _length_delimited(1, summary_value)
    return (
        _varint((1 << 3) | 1)
        + struct.pack("<d", wall_time)
        + _varint((2 << 3) | 0)
        + _varint(step)
        + _length_delimited(5, summary)
    )


_CRC32C_TABLE: tuple[int, ...] | None = None


def _crc32c(data: bytes) -> int:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        polynomial = 0x82F63B78
        table: list[int] = []
        for initial in range(256):
            value = initial
            for _ in range(8):
                value = (value >> 1) ^ polynomial if value & 1 else value >> 1
            table.append(value)
        _CRC32C_TABLE = tuple(table)
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


class ScalarEventWriter:
    """Minimal TensorBoard event writer exposing scalar operations only."""

    def __init__(self, log_dir: str | Path, *, flush_interval_s: float = 30.0) -> None:
        if flush_interval_s <= 0 or not math.isfinite(flush_interval_s):
            raise ValueError("flush_interval_s must be finite and positive")
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = (
            f"events.out.tfevents.{int(time.time())}.{socket.gethostname()}."
            f"{os.getpid()}.{time.time_ns()}"
        )
        self.path = directory / filename
        self._stream = self.path.open("xb")
        self._flush_interval_s = float(flush_interval_s)
        self._last_flush = time.monotonic()
        self._closed = False
        self._last_step_by_tag: dict[str, int] = {}
        self._write_record(_event_file_version(time.time()))
        self.flush()

    def _write_record(self, payload: bytes) -> None:
        length = struct.pack("<Q", len(payload))
        self._stream.write(length)
        self._stream.write(struct.pack("<I", _masked_crc32c(length)))
        self._stream.write(payload)
        self._stream.write(struct.pack("<I", _masked_crc32c(payload)))

    def add_scalar(
        self,
        tag: str,
        scalar_value: int | float,
        global_step: int,
        walltime: float | None = None,
    ) -> None:
        if self._closed:
            raise MetricLoggerError("TensorBoard writer is closed")
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("TensorBoard scalar tag must be non-empty")
        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        previous_step = self._last_step_by_tag.get(tag)
        if previous_step is not None and global_step <= previous_step:
            raise ValueError(
                f"global_step for scalar tag {tag!r} must increase beyond {previous_step}"
            )
        if isinstance(scalar_value, bool) or not isinstance(scalar_value, (int, float)):
            raise TypeError("TensorBoard scalar value must be numeric")
        numeric = float(scalar_value)
        if not math.isfinite(numeric):
            raise ValueError("TensorBoard scalar value must be finite")
        event_time = time.time() if walltime is None else float(walltime)
        if not math.isfinite(event_time):
            raise ValueError("walltime must be finite")
        self._write_record(_event_scalar(event_time, global_step, tag, numeric))
        self._last_step_by_tag[tag] = global_step
        if time.monotonic() - self._last_flush >= self._flush_interval_s:
            self.flush()

    def flush(self) -> None:
        if not self._closed:
            self._stream.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        if not self._closed:
            self.flush()
            self._stream.close()
            self._closed = True

    def __enter__(self) -> "ScalarEventWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_TRAIN_SCALARS = {
    "episode_return_mean": "train/episode_return",
    "mission_success_rate_100": "train/mission_success_rate_100",
    "episode_length_mean": "train/episode_length",
    "learning_rate": "train/learning_rate",
    "fps": "train/fps",
    "policy_loss": "train/policy_loss",
    "value_loss": "train/value_loss",
    "entropy": "train/entropy",
    "approx_kl": "train/approx_kl",
    "train_loss": "train/loss",
    "validation_loss": "eval/loss",
    "token_accuracy": "eval/accuracy",
}

_EVAL_SCALARS = {
    "mission_success_rate": "eval/mission_success_rate",
    "search_success_rate": "eval/search_success_rate",
    "correct_lock_rate": "eval/correct_lock_rate",
    "false_lock_rate": "eval/false_lock_rate",
    "track_success_rate": "eval/track_success_rate",
    "reacquire_success_rate": "eval/reacquire_success_rate",
    "landing_success_rate": "eval/landing_success_rate",
    "collision_rate": "eval/collision_rate",
    "mean_mission_time_s": "eval/mean_mission_time_s",
}


class MetricLogger:
    """Own the five fixed CSV files and optional scalar TensorBoard stream."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str | None = None,
        tensorboard_enabled: bool = True,
        tensorboard_flush_interval_s: float = 30.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.metrics_dir = self.run_dir / "metrics"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        if tensorboard_enabled:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        self._train = _AppendCsv(self.metrics_dir / "train_metrics.csv", TRAIN_METRIC_FIELDS)
        self._eval = _AppendCsv(self.metrics_dir / "eval_metrics.csv", EVAL_METRIC_FIELDS)
        self._episode = _AppendCsv(
            self.metrics_dir / "episode_metrics.csv", EPISODE_METRIC_FIELDS
        )
        self._failure = _AppendCsv(
            self.metrics_dir / "failure_cases.csv", FAILURE_CASE_FIELDS
        )
        self._final = _AppendCsv(self.metrics_dir / "final_metrics.csv", FINAL_METRIC_FIELDS)
        self._tables = (self._train, self._eval, self._episode, self._failure, self._final)
        self._episode_keys = self._read_keys(self._episode)
        self._failure_keys = self._read_keys(self._failure)
        self._final_logged = bool(self._final.rows())
        self._last_train_global_step = self._read_last_global_step(self._train, "train")
        self._last_eval_global_step = self._read_last_global_step(self._eval, "eval")
        self._closed = False
        self.tensorboard = (
            ScalarEventWriter(
                self.tensorboard_dir,
                flush_interval_s=tensorboard_flush_interval_s,
            )
            if tensorboard_enabled
            else None
        )

    @staticmethod
    def _read_keys(table: _AppendCsv) -> set[tuple[str, str, str]]:
        keys: set[tuple[str, str, str]] = set()
        for row in table.rows():
            key = (row.get("run_id", ""), row.get("phase", ""), row.get("episode_id", ""))
            if key in keys:
                raise DuplicateEpisodeError(
                    f"existing CSV contains duplicate episode key {key}: {table.path}"
                )
            keys.add(key)
        return keys

    @staticmethod
    def _read_last_global_step(table: _AppendCsv, label: str) -> int | None:
        last: int | None = None
        for row in table.rows():
            raw = row.get("global_step", "")
            try:
                step = int(raw)
            except (TypeError, ValueError) as exc:
                raise MetricSchemaError(
                    f"existing {label} global_step is not an integer: {raw!r}"
                ) from exc
            if step < 0 or (last is not None and step <= last):
                raise MetricSchemaError(
                    f"existing {label} global_step values must be strictly increasing"
                )
            last = step
        return last

    @staticmethod
    def _validate_next_step(value: object, previous: int | None, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MetricSchemaError(f"{label} global_step must be a non-negative integer")
        if previous is not None and value <= previous:
            raise MetricSchemaError(
                f"{label} global_step must resume above {previous}, got {value}"
            )
        return value

    def _ensure_open(self) -> None:
        if self._closed:
            raise MetricLoggerError("MetricLogger is closed")

    @staticmethod
    def _merge(
        metrics: Mapping[str, object] | object | None,
        **explicit: object,
    ) -> dict[str, object]:
        values = _mapping(metrics)
        for key, value in explicit.items():
            if value is not None:
                values[key] = value
        return values

    def log_train(
        self,
        global_step: int,
        update: int,
        metrics: Mapping[str, object] | object | None = None,
        *,
        timestamp: str | None = None,
    ) -> None:
        self._ensure_open()
        global_step = self._validate_next_step(
            global_step, self._last_train_global_step, "train"
        )
        values = self._merge(
            metrics,
            global_step=global_step,
            update=update,
            timestamp=timestamp or _utc_timestamp(),
        )
        self._train.append(values)
        self._log_scalars(_TRAIN_SCALARS, values, global_step)
        self._last_train_global_step = global_step

    def log_eval(
        self,
        global_step: int,
        metrics: Mapping[str, object] | object,
        *,
        checkpoint_step: int | None = None,
        timestamp: str | None = None,
    ) -> None:
        self._ensure_open()
        global_step = self._validate_next_step(
            global_step, self._last_eval_global_step, "eval"
        )
        values = self._merge(
            metrics,
            global_step=global_step,
            checkpoint_step=checkpoint_step,
            timestamp=timestamp or _utc_timestamp(),
        )
        self._eval.append(values)
        self._log_scalars(_EVAL_SCALARS, values, global_step)
        self._last_eval_global_step = global_step

    def log_episode(self, metrics: Mapping[str, object] | object) -> bool:
        """Write one episode and, if failed, one canonical failure row.

        Returns the strict mission success decision written to the CSV.
        """

        self._ensure_open()
        values = _mapping(metrics)
        if not values.get("run_id") and self.run_id:
            values["run_id"] = self.run_id
        phase = self._validate_phase(values.get("phase"))
        values["phase"] = phase
        key = self._episode_key(values)
        if key in self._episode_keys:
            raise DuplicateEpisodeError(f"episode already logged: {key}")

        success = compute_mission_success_strict(values)
        values["mission_success_strict"] = success
        failure_reason = values.get("failure_reason")
        if success and failure_reason not in (None, ""):
            raise MetricSchemaError("successful episode cannot have a failure_reason")
        if not success and failure_reason in (None, ""):
            values["failure_reason"] = FailureReason.UNKNOWN_ERROR.value
        elif not success:
            values["failure_reason"] = self._validate_failure_reason(failure_reason)

        # Validate both records before mutating either append-only table.  In
        # particular, a failure-only field such as ``message`` must not leave
        # an episode row behind when it cannot be represented in the failure
        # CSV.
        _normalize_row(EPISODE_METRIC_FIELDS, values)
        prepared_failure: tuple[dict[str, object], tuple[str, str, str]] | None = None
        if not success:
            prepared_failure = self._prepare_failure(values, strict_success=False)

        self._episode.append(values)
        self._episode_keys.add(key)
        if prepared_failure is not None:
            failure_row, failure_key = prepared_failure
            self._failure.append(failure_row)
            self._failure_keys.add(failure_key)
        return success

    def log_failure(
        self,
        metrics: Mapping[str, object] | object,
        *,
        mission_success_strict: bool | None = None,
    ) -> None:
        """Write an explicitly failed episode to ``failure_cases.csv`` only."""

        self._ensure_open()
        values = _mapping(metrics)
        declared = values.get("mission_success_strict", mission_success_strict)
        if (
            parse_metric_bool(declared) is True
            or compute_mission_success_strict(values)
        ):
            raise MetricSchemaError("failure_cases.csv may contain only failed episodes")
        row, key = self._prepare_failure(values, strict_success=False)
        self._failure.append(row)
        self._failure_keys.add(key)

    def _prepare_failure(
        self,
        values: Mapping[str, object],
        *,
        strict_success: bool,
    ) -> tuple[dict[str, object], tuple[str, str, str]]:
        if strict_success:
            raise MetricSchemaError("failure_cases.csv may contain only failed episodes")
        row = dict(values)
        if not row.get("run_id") and self.run_id:
            row["run_id"] = self.run_id
        row["phase"] = self._validate_phase(row.get("phase"))
        reason = row.get("failure_reason")
        if reason in (None, ""):
            reason = FailureReason.UNKNOWN_ERROR.value
        row["failure_reason"] = self._validate_failure_reason(reason)
        key = self._episode_key(row)
        if key in self._failure_keys:
            raise DuplicateEpisodeError(f"failure episode already logged: {key}")
        _normalize_row(FAILURE_CASE_FIELDS, row)
        return row, key

    def log_final(self, metrics: Mapping[str, object] | object) -> None:
        self._ensure_open()
        if self._final_logged:
            raise FinalMetricsAlreadyLoggedError("final_metrics.csv already has its one data row")
        values = _mapping(metrics)
        if not values.get("run_id") and self.run_id:
            values["run_id"] = self.run_id
        self._validate_final_metrics(values)
        self._final.append(values)
        self._final_logged = True

    @staticmethod
    def _validate_final_metrics(values: Mapping[str, object]) -> None:
        run_id = values.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise MetricSchemaError("final metrics require a non-empty run_id")
        step = values.get("best_checkpoint_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise MetricSchemaError(
                "final metrics require a non-negative best_checkpoint_step"
            )
        episodes = values.get("num_test_episodes")
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
            raise MetricSchemaError("final metrics require positive num_test_episodes")

        rate_names = (
            "mission_success_rate",
            "mission_success_ci95_low",
            "mission_success_ci95_high",
        )
        rates: list[float] = []
        for name in rate_names:
            value = values.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MetricSchemaError(f"final metrics require numeric {name}")
            rate = float(value)
            if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise MetricSchemaError(f"final metric {name} must be within [0, 1]")
            rates.append(rate)
        mission_rate, ci_low, ci_high = rates
        if not ci_low <= mission_rate <= ci_high:
            raise MetricSchemaError(
                "mission_success_rate must lie inside its 95% confidence interval"
            )

    @staticmethod
    def _validate_phase(value: object) -> str:
        if isinstance(value, MetricPhase):
            return value.value
        try:
            return MetricPhase(str(value)).value
        except ValueError as exc:
            raise MetricSchemaError("phase must be train, validation, or test") from exc

    @staticmethod
    def _validate_failure_reason(value: object) -> str:
        if isinstance(value, FailureReason):
            return value.value
        try:
            return FailureReason(str(value)).value
        except ValueError as exc:
            raise MetricSchemaError(f"unknown failure_reason: {value!r}") from exc

    @staticmethod
    def _episode_key(values: Mapping[str, object]) -> tuple[str, str, str]:
        run_id = values.get("run_id")
        phase = values.get("phase")
        episode_id = values.get("episode_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise MetricSchemaError("run_id is required for episode identity")
        if episode_id is None or isinstance(episode_id, bool):
            raise MetricSchemaError("episode_id is required for episode identity")
        if isinstance(episode_id, float) and not episode_id.is_integer():
            raise MetricSchemaError("episode_id must be a stable scalar identifier")
        normalized_id = str(episode_id)
        if not normalized_id:
            raise MetricSchemaError("episode_id must not be empty")
        return run_id, str(phase), normalized_id

    def _log_scalars(
        self,
        tags: Mapping[str, str],
        values: Mapping[str, object],
        global_step: int,
    ) -> None:
        if self.tensorboard is None:
            return
        for field, tag in tags.items():
            value = values.get(field)
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                self.tensorboard.add_scalar(tag, value, global_step)

    def flush(self) -> None:
        self._ensure_open()
        for table in self._tables:
            table.flush()
        if self.tensorboard is not None:
            self.tensorboard.flush()

    def close(self) -> None:
        if self._closed:
            return
        for table in self._tables:
            table.close()
        if self.tensorboard is not None:
            self.tensorboard.close()
        self._closed = True

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "DuplicateEpisodeError",
    "FinalMetricsAlreadyLoggedError",
    "MetricLogger",
    "MetricLoggerError",
    "MetricSchemaError",
    "ScalarEventWriter",
]
