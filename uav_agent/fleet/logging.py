"""Sparse, image-free output layout for fleet missions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import io
import json
from math import isfinite
from pathlib import Path
import re
from threading import RLock

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from fleet.strict_json import strict_json_object_loads


_FORBIDDEN_LOG_KEYS = frozenset(
    {
        "image",
        "images",
        "image_url",
        "camera_rgb",
        "rgb",
        "pixels",
        "video",
        "videos",
        "raw_frame",
        "raw_frames",
        "observation",
        "observations",
        "observation_dump",
        "observation_dumps",
        "prompt",
        "full_prompt",
        "raw_prompt",
        "api_key",
        "authorization",
    }
)

_DEFAULT_MAX_RECORD_BYTES = 32_768
_DEFAULT_MAX_STREAM_BYTES = 8_388_608
_DEFAULT_MAX_RUN_BYTES = 33_554_432
_SUMMARY_RESERVE_BYTES = 32_768
FLEET_LOG_STORAGE_SIDECAR = ".fleet_logger_storage.json"
_MAX_SIDECAR_BYTES = 4096
_MAX_TRACKED_DROPPED_STREAMS = 32


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _positive_limit(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _safe_json(value: object, path: str = "record") -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if key.casefold() in _FORBIDDEN_LOG_KEYS:
                raise ValueError(f"{path}.{key} must not be persisted")
            result[key] = _safe_json(nested, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_json(item, f"{path}[]") for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{path} must be a finite JSON number")
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and (
            "base64," in value.casefold()
            or value.casefold().startswith("data:image/")
            or value.casefold().startswith("data:video/")
            or re.fullmatch(r"[A-Za-z0-9+/]{160,}={0,2}", value) is not None
        ):
            raise ValueError(f"{path} must not contain encoded media or base64 blobs")
        return value
    if hasattr(value, "to_dict"):
        return _safe_json(value.to_dict(), path)
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


class FleetMissionLogger:
    """Create the checklist's deterministic directory and sparse log streams."""

    MODEL_FIELDS = (
        "call_id",
        "call_role",
        "fleet_mission_id",
        "assignment_id",
        "uav_id",
        "priority",
        "requested_adapter",
        "adapter_status",
        "effective_model",
        "fallback_used",
        "prompt_tokens",
        "completion_tokens",
        "latency_s",
        "finish_reason",
        "error_code",
        "stale_reasons",
    )
    ASSIGNMENT_FIELDS = (
        "assignment_id",
        "uav_id",
        "target_alias",
        "priority",
        "status",
        "local_plan_version",
    )
    AIRSPACE_FIELDS = (
        "timestamp_s",
        "uav_a_id",
        "uav_b_id",
        "risk",
        "current_distance_m",
        "predicted_closest_distance_m",
        "time_to_closest_s",
        "hold_uav_id",
    )

    def __init__(
        self,
        root_dir: str | Path,
        fleet_mission_id: str,
        *,
        uav_ids: Sequence[str] = (),
        max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES,
        max_stream_bytes: int = _DEFAULT_MAX_STREAM_BYTES,
        max_run_bytes: int = _DEFAULT_MAX_RUN_BYTES,
    ) -> None:
        self.fleet_mission_id = validate_mission_id(fleet_mission_id)
        self._configure_storage_limits(
            max_record_bytes=max_record_bytes,
            max_stream_bytes=max_stream_bytes,
            max_run_bytes=max_run_bytes,
        )
        normalized_uav_ids = tuple(validate_uav_id(uav_id) for uav_id in uav_ids)
        self.run_dir = Path(root_dir).expanduser().resolve() / self.fleet_mission_id
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise FileExistsError(
                "fleet mission log directory already exists; refusing to mix "
                f"records from another run: {self.run_dir}"
            ) from None
        self._initialize(uav_ids=normalized_uav_ids, preserve_existing=False)

    @classmethod
    def attach_run_dir(
        cls,
        run_dir: str | Path,
        fleet_mission_id: str,
        *,
        uav_ids: Sequence[str] = (),
        max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES,
        max_stream_bytes: int = _DEFAULT_MAX_STREAM_BYTES,
        max_run_bytes: int = _DEFAULT_MAX_RUN_BYTES,
    ) -> "FleetMissionLogger":
        """Attach fleet logging to an existing experiment run directory.

        This is the bridge used when :class:`experiments.run_manager.RunManager`
        already owns the run directory.  Unlike the legacy constructor, this
        entry point never creates the run directory and never truncates an
        existing fleet stream.  Files owned by RunManager (for example
        ``manifest.yaml``, ``logs/terminal.log``, and ``metrics/``) are not
        inspected or modified.
        """

        mission_id = validate_mission_id(fleet_mission_id)
        normalized_uav_ids = tuple(validate_uav_id(uav_id) for uav_id in uav_ids)
        requested_path = Path(run_dir).expanduser()
        if requested_path.is_symlink():
            raise ValueError("run_dir must not be a symbolic link")
        try:
            resolved_path = requested_path.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"fleet mission run directory does not exist: {requested_path}"
            ) from None
        if not resolved_path.is_dir():
            raise NotADirectoryError(
                f"fleet mission run path is not a directory: {resolved_path}"
            )

        logger = cls.__new__(cls)
        logger.fleet_mission_id = mission_id
        logger.run_dir = resolved_path
        logger._configure_storage_limits(
            max_record_bytes=max_record_bytes,
            max_stream_bytes=max_stream_bytes,
            max_run_bytes=max_run_bytes,
        )
        logger._initialize(uav_ids=normalized_uav_ids, preserve_existing=True)
        return logger

    def _configure_storage_limits(
        self,
        *,
        max_record_bytes: int,
        max_stream_bytes: int,
        max_run_bytes: int,
    ) -> None:
        self.max_record_bytes = _positive_limit(max_record_bytes, "max_record_bytes")
        self.max_stream_bytes = _positive_limit(max_stream_bytes, "max_stream_bytes")
        self.max_run_bytes = _positive_limit(max_run_bytes, "max_run_bytes")
        if not self.max_record_bytes <= self.max_stream_bytes <= self.max_run_bytes:
            raise ValueError("fleet log byte limits must satisfy record <= stream <= run")
        self._summary_reserve_bytes = min(
            _SUMMARY_RESERVE_BYTES,
            max(1, self.max_run_bytes // 3),
        )
        self._sidecar_reserve_bytes = min(
            _MAX_SIDECAR_BYTES,
            max(128, self.max_run_bytes // 8),
        )
        self._dropped_records: dict[str, int] = {}
        self._truncated_streams: set[str] = set()
        self._untracked_dropped_record_count = 0
        self._drop_generation = 0

    def _initialize(
        self,
        *,
        uav_ids: Sequence[str],
        preserve_existing: bool,
    ) -> None:
        """Initialize logger-owned paths after the run directory is selected."""

        self.agents_dir = self.run_dir / "agents"
        if preserve_existing:
            self._validate_existing_layout(uav_ids)
        self.agents_dir.mkdir(exist_ok=True)
        self._lock = RLock()
        self._initialize_storage_state()
        self._registered_uavs: set[str] = set()
        for uav_id in uav_ids:
            self.register_agent(uav_id)
        # Stable headers make a zero-call/zero-conflict run self-describing.
        (self.run_dir / "fleet_events.jsonl").touch(exist_ok=True)
        streams = (
            ("assignments.csv", self.ASSIGNMENT_FIELDS),
            ("model_calls.csv", self.MODEL_FIELDS),
            ("airspace_conflicts.csv", self.AIRSPACE_FIELDS),
        )
        for filename, fields in streams:
            path = self.run_dir / filename
            if not preserve_existing or not path.exists() or path.stat().st_size == 0:
                self._write_csv(filename, fields, (), append=False)

    def _validate_existing_layout(self, uav_ids: Sequence[str]) -> None:
        """Fail before mutation when an attached fleet layout is incompatible."""

        if self.agents_dir.is_symlink():
            raise ValueError("fleet agents directory must not be a symbolic link")
        if self.agents_dir.exists() and not self.agents_dir.is_dir():
            raise NotADirectoryError(f"fleet agents path is not a directory: {self.agents_dir}")

        jsonl_path = self.run_dir / "fleet_events.jsonl"
        self._validate_regular_stream(jsonl_path)
        for filename, expected_fields in (
            ("assignments.csv", self.ASSIGNMENT_FIELDS),
            ("model_calls.csv", self.MODEL_FIELDS),
            ("airspace_conflicts.csv", self.AIRSPACE_FIELDS),
        ):
            path = self.run_dir / filename
            self._validate_regular_stream(path)
            if path.exists() and path.stat().st_size:
                with path.open(newline="", encoding="utf-8") as stream:
                    header = tuple(next(csv.reader(stream), ()))
                if header != tuple(expected_fields):
                    raise ValueError(
                        f"existing fleet CSV has an incompatible header: {path}"
                    )

        for uav_id in uav_ids:
            agent_dir = self.agents_dir / uav_id
            if agent_dir.is_symlink():
                raise ValueError(f"agent log directory must not be a symbolic link: {agent_dir}")
            if agent_dir.exists() and not agent_dir.is_dir():
                raise NotADirectoryError(f"agent log path is not a directory: {agent_dir}")
            for filename in (
                "transitions.jsonl",
                "visual_reviews.jsonl",
                "revisions.jsonl",
            ):
                self._validate_regular_stream(agent_dir / filename)

    @staticmethod
    def _validate_regular_stream(path: Path) -> None:
        if path.is_symlink():
            raise ValueError(f"fleet log stream must not be a symbolic link: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"fleet log stream must be a regular file: {path}")

    def _stream_name(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.run_dir)
        except ValueError as exc:
            raise ValueError("fleet log stream must be below run_dir") from exc
        return relative.as_posix()

    def _drop(self, path: Path) -> bool:
        name = self._stream_name(path)
        if name in self._dropped_records or len(self._dropped_records) < _MAX_TRACKED_DROPPED_STREAMS:
            self._dropped_records[name] = self._dropped_records.get(name, 0) + 1
            self._truncated_streams.add(name)
        else:
            self._untracked_dropped_record_count += 1
            self._truncated_streams.add("__additional_fleet_log_streams__")
        self._drop_generation += 1
        self._persist_storage_state()
        return False

    def _initialize_storage_state(self) -> None:
        path = self.run_dir / FLEET_LOG_STORAGE_SIDECAR
        if path.exists():
            self._validate_regular_stream(path)
            try:
                payload = strict_json_object_loads(path.read_text(encoding="utf-8"))
                if set(payload) != {
                    "schema_version",
                    "generation",
                    "dropped_records",
                    "untracked_dropped_record_count",
                    "truncated_streams",
                } or payload.get("schema_version") != 1:
                    raise ValueError("invalid fleet logger storage sidecar schema")
                dropped = payload.get("dropped_records", {})
                truncated = payload.get("truncated_streams", [])
                if not isinstance(dropped, dict) or not isinstance(truncated, list):
                    raise ValueError("invalid fleet logger storage sidecar")
                self._dropped_records = {
                    str(name): int(count)
                    for name, count in dropped.items()
                    if isinstance(name, str)
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                }
                self._truncated_streams = {
                    str(name) for name in truncated if isinstance(name, str)
                }
                self._untracked_dropped_record_count = int(
                    payload.get("untracked_dropped_record_count", 0)
                )
                self._drop_generation = int(payload.get("generation", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid fleet logger storage sidecar") from exc
            return
        self._persist_storage_state()

    def _persist_storage_state(self) -> None:
        path = self.run_dir / FLEET_LOG_STORAGE_SIDECAR
        payload = {
            "schema_version": 1,
            "generation": self._drop_generation,
            "dropped_records": dict(sorted(self._dropped_records.items())),
            "untracked_dropped_record_count": self._untracked_dropped_record_count,
            "truncated_streams": sorted(self._truncated_streams),
        }
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        if len(encoded) > self._sidecar_reserve_bytes:
            total = sum(self._dropped_records.values()) + self._untracked_dropped_record_count
            payload = {
                "schema_version": 1,
                "generation": self._drop_generation,
                "dropped_records": {"__all_fleet_log_streams__": total},
                "untracked_dropped_record_count": 0,
                "truncated_streams": ["__all_fleet_log_streams__"],
            }
            encoded = (
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        previous_size = path.stat().st_size if path.exists() else 0
        projected = _directory_size(self.run_dir) - previous_size + len(encoded)
        if len(encoded) > self._sidecar_reserve_bytes or projected > self.max_run_bytes - self._summary_reserve_bytes:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_bytes(encoded)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _can_write(
        self,
        path: Path,
        encoded_size: int,
        *,
        replacing_bytes: int = 0,
        is_summary: bool = False,
    ) -> bool:
        record_limit = self._summary_reserve_bytes if is_summary else self.max_record_bytes
        if encoded_size > record_limit:
            return self._drop(path)
        current_stream_bytes = path.stat().st_size if path.exists() else 0
        projected_stream = encoded_size if replacing_bytes else current_stream_bytes + encoded_size
        stream_limit = (
            max(self.max_stream_bytes, self._summary_reserve_bytes)
            if is_summary
            else self.max_stream_bytes
        )
        if projected_stream > stream_limit:
            return self._drop(path)
        reserve = (
            0
            if is_summary
            else self._summary_reserve_bytes + self._sidecar_reserve_bytes
        )
        projected_run = _directory_size(self.run_dir) - replacing_bytes + encoded_size
        if projected_run > self.max_run_bytes - reserve:
            return self._drop(path)
        return True

    def storage_snapshot(self) -> dict[str, object]:
        """Return bounded logger drops for merging into mission summary."""

        with self._lock:
            dropped = dict(sorted(self._dropped_records.items()))
            if self._untracked_dropped_record_count:
                dropped["__additional_fleet_log_streams__"] = (
                    dropped.get("__additional_fleet_log_streams__", 0)
                    + self._untracked_dropped_record_count
                )
            return {
                "dropped_records": dropped,
                "dropped_record_count": sum(dropped.values()),
                "truncated_streams": sorted(self._truncated_streams),
                "final_run_bytes": _directory_size(self.run_dir),
                "generation": self._drop_generation,
            }

    def register_agent(self, uav_id: str) -> Path:
        uav_id = validate_uav_id(uav_id)
        with self._lock:
            agent_dir = self.agents_dir / uav_id
            if agent_dir.is_symlink():
                raise ValueError(f"agent log directory must not be a symbolic link: {agent_dir}")
            if agent_dir.exists() and not agent_dir.is_dir():
                raise NotADirectoryError(f"agent log path is not a directory: {agent_dir}")
            agent_dir.mkdir(parents=True, exist_ok=True)
            for filename in (
                "transitions.jsonl",
                "visual_reviews.jsonl",
                "revisions.jsonl",
            ):
                stream_path = agent_dir / filename
                self._validate_regular_stream(stream_path)
                stream_path.touch(exist_ok=True)
            self._registered_uavs.add(uav_id)
            return agent_dir

    def write_run_manifest(self, manifest: Mapping[str, object]) -> None:
        self._write_json("run_manifest.json", manifest)

    def write_fleet_plan(self, plan: object) -> None:
        self._write_json("fleet_plan.json", plan)

    def write_task_spec(self, task_spec: object) -> None:
        self._write_json("fleet_task_spec.json", task_spec)

    def write_runtime_execution_plan(self, plan: object) -> None:
        """Persist the explicit compatibility envelope, never model it as V2."""

        self._write_json("runtime_execution_plan_v1.json", plan)

    def write_summary(self, summary: Mapping[str, object]) -> None:
        payload = _safe_json(summary)
        assert isinstance(payload, dict)
        incoming_storage = payload.get("result_storage")
        base_storage = dict(incoming_storage) if isinstance(incoming_storage, Mapping) else {}
        incoming_dropped = base_storage.get("dropped_records")
        base_dropped = (
            {str(key): int(value) for key, value in incoming_dropped.items()}
            if isinstance(incoming_dropped, Mapping)
            else {}
        )
        previous_fleet_dropped = base_storage.get("fleet_logger_dropped_records")
        if isinstance(previous_fleet_dropped, Mapping):
            for stream, count in previous_fleet_dropped.items():
                if isinstance(stream, str) and isinstance(count, int) and not isinstance(count, bool):
                    remaining = base_dropped.get(stream, 0) - count
                    if remaining > 0:
                        base_dropped[stream] = remaining
                    else:
                        base_dropped.pop(stream, None)
        incoming_truncated = base_storage.get("truncated_streams")
        base_truncated = {
            str(item) for item in incoming_truncated
        } if isinstance(incoming_truncated, (list, tuple, set)) else set()

        def merged_storage() -> dict[str, object]:
            storage = dict(base_storage)
            dropped = dict(base_dropped)
            for stream, count in self._dropped_records.items():
                dropped[stream] = dropped.get(stream, 0) + count
            if self._untracked_dropped_record_count:
                aggregate = "__additional_fleet_log_streams__"
                dropped[aggregate] = (
                    dropped.get(aggregate, 0) + self._untracked_dropped_record_count
                )
            truncated = set(base_truncated)
            truncated.update(self._truncated_streams)
            storage.update({
                "dropped_records": dict(sorted(dropped.items())),
                "dropped_record_count": sum(dropped.values()),
                "truncated_streams": sorted(truncated),
                "fleet_logger_dropped_records": self.storage_snapshot()[
                    "dropped_records"
                ],
                "fleet_logger_truncated_streams": sorted(self._truncated_streams),
                "fleet_logger_drop_generation": self._drop_generation,
            })
            return storage

        storage = merged_storage()
        payload["result_storage"] = storage
        terminal_status = payload.get("status", payload.get("final_status", "UNKNOWN"))
        path = self.run_dir / "summary.json"
        for _ in range(3):
            previous_size = path.stat().st_size if path.exists() else 0
            rendered = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            storage["final_run_bytes"] = (
                _directory_size(self.run_dir) - previous_size + len(rendered)
            )
            if not self._write_json_path(path, payload, is_summary=True):
                compact = {
                    "schema_version": 1,
                    "fleet_mission_id": self.fleet_mission_id,
                    "status": terminal_status,
                    "result_storage": merged_storage(),
                }
                payload = compact
                storage = compact["result_storage"]  # type: ignore[assignment]
                continue
            if storage["final_run_bytes"] == _directory_size(self.run_dir):
                break

    def write_assignments(self, rows: Sequence[Mapping[str, object]]) -> None:
        self._write_csv("assignments.csv", self.ASSIGNMENT_FIELDS, rows, append=False)

    def log_fleet_event(self, event: Mapping[str, object] | object) -> None:
        self._append_jsonl("fleet_events.jsonl", event)

    def log_model_call(self, record: Mapping[str, object] | object) -> None:
        row = _safe_json(record)
        assert isinstance(row, dict)
        row.setdefault("fleet_mission_id", self.fleet_mission_id)
        if isinstance(row.get("stale_reasons"), list):
            row["stale_reasons"] = "|".join(str(item) for item in row["stale_reasons"])
        self._write_csv("model_calls.csv", self.MODEL_FIELDS, (row,), append=True)

    def log_airspace_decision(self, decision: object) -> None:
        data = _safe_json(decision)
        assert isinstance(data, dict)
        timestamp = data.get("timestamp_s")
        conflicts = data.get("conflicts", [])
        if not isinstance(conflicts, list):
            raise TypeError("airspace decision conflicts must be an array")
        rows = []
        for conflict in conflicts:
            if not isinstance(conflict, dict):
                raise TypeError("airspace conflict must be an object")
            rows.append({"timestamp_s": timestamp, **conflict})
        if rows:
            self._write_csv(
                "airspace_conflicts.csv",
                self.AIRSPACE_FIELDS,
                rows,
                append=True,
            )

    def write_local_plan(self, uav_id: str, plan_version: int, plan: object) -> None:
        uav_id = validate_uav_id(uav_id)
        if isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version <= 0:
            raise ValueError("plan_version must be a positive integer")
        agent_dir = self.register_agent(uav_id)
        self._write_json_path(agent_dir / f"local_plan_v{plan_version}.json", plan)

    def log_agent_transition(self, uav_id: str, record: object) -> None:
        self._agent_jsonl(uav_id, "transitions.jsonl", record)

    def log_visual_review(self, uav_id: str, record: object) -> None:
        self._agent_jsonl(uav_id, "visual_reviews.jsonl", record)

    def log_revision(self, uav_id: str, record: object) -> None:
        self._agent_jsonl(uav_id, "revisions.jsonl", record)

    def _agent_jsonl(self, uav_id: str, filename: str, record: object) -> None:
        agent_dir = self.register_agent(uav_id)
        self._append_jsonl_path(agent_dir / filename, record)

    def _write_json(self, filename: str, value: object) -> bool:
        return self._write_json_path(self.run_dir / filename, value)

    def _write_json_path(
        self,
        path: Path,
        value: object,
        *,
        is_summary: bool = False,
    ) -> bool:
        payload = _safe_json(value)
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        temporary = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            old_size = path.stat().st_size if path.exists() else 0
            if not self._can_write(
                path,
                len(encoded),
                replacing_bytes=old_size,
                is_summary=is_summary,
            ):
                return False
            try:
                temporary.write_bytes(encoded)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return True

    def _append_jsonl(self, filename: str, value: object) -> bool:
        return self._append_jsonl_path(self.run_dir / filename, value)

    def _append_jsonl_path(self, path: Path, value: object) -> bool:
        payload = _safe_json(value)
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        with self._lock:
            if not self._can_write(path, len(encoded)):
                return False
            with path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
        return True

    def _write_csv(
        self,
        filename: str,
        fields: Sequence[str],
        rows: Sequence[Mapping[str, object]],
        *,
        append: bool,
    ) -> bool:
        path = self.run_dir / filename
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=tuple(fields),
            extrasaction="ignore",
        )
        write_header = not append or not path.exists() or path.stat().st_size == 0
        if write_header:
            writer.writeheader()
        for raw_row in rows:
            row = _safe_json(raw_row)
            if not isinstance(row, dict):
                raise TypeError("CSV row must be an object")
            writer.writerow({field: row.get(field) for field in fields})
        encoded = buffer.getvalue().encode("utf-8")
        with self._lock:
            destination = path if append else path.with_suffix(path.suffix + ".tmp")
            old_size = path.stat().st_size if path.exists() and not append else 0
            if not self._can_write(path, len(encoded), replacing_bytes=old_size):
                return False
            try:
                mode = "ab" if append else "wb"
                with destination.open(mode) as stream:
                    stream.write(encoded)
                    stream.flush()
                if not append:
                    destination.replace(path)
            except BaseException:
                if not append:
                    destination.unlink(missing_ok=True)
                raise
        return True


__all__ = ["FleetMissionLogger"]
