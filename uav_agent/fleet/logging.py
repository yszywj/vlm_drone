"""Sparse, image-free output layout for fleet missions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from math import isfinite
from pathlib import Path
from threading import RLock

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id


_FORBIDDEN_LOG_KEYS = frozenset(
    {
        "image",
        "images",
        "image_url",
        "camera_rgb",
        "rgb",
        "pixels",
        "api_key",
        "authorization",
    }
)


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
            "base64," in value.casefold() or value.casefold().startswith("data:image/")
        ):
            raise ValueError(f"{path} must not contain image data")
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
    ) -> None:
        self.fleet_mission_id = validate_mission_id(fleet_mission_id)
        normalized_uav_ids = tuple(validate_uav_id(uav_id) for uav_id in uav_ids)
        self.run_dir = Path(root_dir).expanduser().resolve() / self.fleet_mission_id
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise FileExistsError(
                "fleet mission log directory already exists; refusing to mix "
                f"records from another run: {self.run_dir}"
            ) from None
        self.agents_dir = self.run_dir / "agents"
        self.agents_dir.mkdir(exist_ok=True)
        self._lock = RLock()
        self._registered_uavs: set[str] = set()
        for uav_id in normalized_uav_ids:
            self.register_agent(uav_id)
        # Stable headers make a zero-call/zero-conflict run self-describing.
        (self.run_dir / "fleet_events.jsonl").touch(exist_ok=True)
        self._write_csv(
            "assignments.csv",
            self.ASSIGNMENT_FIELDS,
            (),
            append=False,
        )
        self._write_csv("model_calls.csv", self.MODEL_FIELDS, (), append=False)
        self._write_csv(
            "airspace_conflicts.csv",
            self.AIRSPACE_FIELDS,
            (),
            append=False,
        )

    def register_agent(self, uav_id: str) -> Path:
        uav_id = validate_uav_id(uav_id)
        with self._lock:
            agent_dir = self.agents_dir / uav_id
            agent_dir.mkdir(parents=True, exist_ok=True)
            for filename in (
                "transitions.jsonl",
                "visual_reviews.jsonl",
                "revisions.jsonl",
            ):
                (agent_dir / filename).touch(exist_ok=True)
            self._registered_uavs.add(uav_id)
            return agent_dir

    def write_run_manifest(self, manifest: Mapping[str, object]) -> None:
        self._write_json("run_manifest.json", manifest)

    def write_fleet_plan(self, plan: object) -> None:
        self._write_json("fleet_plan.json", plan)

    def write_summary(self, summary: Mapping[str, object]) -> None:
        self._write_json("summary.json", summary)

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

    def _write_json(self, filename: str, value: object) -> None:
        self._write_json_path(self.run_dir / filename, value)

    def _write_json_path(self, path: Path, value: object) -> None:
        payload = _safe_json(value)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)

    def _append_jsonl(self, filename: str, value: object) -> None:
        self._append_jsonl_path(self.run_dir / filename, value)

    def _append_jsonl_path(self, path: Path, value: object) -> None:
        payload = _safe_json(value)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode("utf-8")) > 64 * 1024:
            raise ValueError("fleet log record exceeds 64 KiB")
        with self._lock, path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()

    def _write_csv(
        self,
        filename: str,
        fields: Sequence[str],
        rows: Sequence[Mapping[str, object]],
        *,
        append: bool,
    ) -> None:
        path = self.run_dir / filename
        with self._lock:
            write_header = not append or not path.exists() or path.stat().st_size == 0
            destination = path if append else path.with_suffix(path.suffix + ".tmp")
            mode = "a" if append else "w"
            try:
                with destination.open(mode, encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=tuple(fields),
                        extrasaction="ignore",
                    )
                    if write_header:
                        writer.writeheader()
                    for raw_row in rows:
                        row = _safe_json(raw_row)
                        if not isinstance(row, dict):
                            raise TypeError("CSV row must be an object")
                        writer.writerow(
                            {field: row.get(field) for field in fields}
                        )
                    stream.flush()
                if not append:
                    destination.replace(path)
            except BaseException:
                if not append:
                    destination.unlink(missing_ok=True)
                raise


__all__ = ["FleetMissionLogger"]
