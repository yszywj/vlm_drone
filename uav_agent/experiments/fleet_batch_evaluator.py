"""Bounded batch aggregation and deterministic detailed-run retention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from math import isfinite
from pathlib import Path
import shutil
from typing import Any

from .evaluator import aggregate_fleet_episode_metrics, parse_metric_bool
from .schemas import BATCH_EPISODE_METRIC_FIELDS, FAILURE_CASE_FIELDS
from fleet.strict_json import strict_json_object_loads


MAX_RETAINED_RUNS = 25
DEFAULT_RETAINED_SUCCESSES = 5
DEFAULT_RETAINED_PER_FAILURE = 3
_DETAIL_FILE_LIMIT_BYTES = 8 * 1024 * 1024
_ALLOWED_DETAIL_NAMES = frozenset(
    {
        "summary.json",
        "run_manifest.json",
        "manifest.yaml",
        "resolved_config.yaml",
        "fleet_task_spec.json",
        "fleet_plan.json",
        "planning_attempts.jsonl",
        "validation_findings.jsonl",
        "recovery_actions.jsonl",
        "final_plans.jsonl",
        "model_calls.csv",
        "report.md",
    }
)
_ALLOWED_METRIC_NAMES = frozenset(
    {
        "fleet_metrics.csv",
        "agent_metrics.csv",
        "goal_metrics.csv",
        "skill_executions.csv",
        "state_samples_1hz.csv",
        "failure_cases.csv",
    }
)
_FORBIDDEN_BYTES = (
    b"base64,",
    b"data:image/",
    b"data:video/",
    b'"camera_rgb"',
    b'"raw_frames"',
    b'"observations"',
    b'"api_key"',
    b'"prompt"',
)


class FleetBatchEvaluationError(RuntimeError):
    """Raised for malformed batch inputs, never for an episode failure."""


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = strict_json_object_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FleetBatchEvaluationError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FleetBatchEvaluationError(f"{path} must contain an object")
    return value


def _read_last_csv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: object, *, integer: bool = False) -> int | float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(normalized):
        return None
    return int(normalized) if integer else normalized


def _bool(value: object, default: bool | None = None) -> bool | None:
    parsed = parse_metric_bool(value)
    return default if parsed is None else parsed


def _error_codes(run_dir: Path, summary: Mapping[str, object]) -> tuple[str, ...]:
    codes: set[str] = set()
    raw = summary.get("error_codes", ())
    if isinstance(raw, str):
        codes.update(item for item in raw.split("|") if item)
    elif isinstance(raw, Sequence):
        codes.update(str(item) for item in raw if item)
    for filename in ("validation_findings.jsonl", "recovery_actions.jsonl"):
        path = run_dir / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = strict_json_object_loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(record, Mapping) and record.get("code"):
                    codes.add(str(record["code"]))
    return tuple(sorted(codes))


def load_fleet_episode_metrics(run_dir: str | Path) -> dict[str, object]:
    """Normalize one result directory into the fixed batch episode schema."""

    run = Path(run_dir).expanduser().resolve()
    summary_path = run / "summary.json"
    if not summary_path.is_file():
        raise FleetBatchEvaluationError(f"missing summary.json: {run}")
    summary = _read_json(summary_path)
    metrics_dir = run / "metrics" if (run / "metrics").is_dir() else run
    fleet = _read_last_csv(metrics_dir / "fleet_metrics.csv")
    source: dict[str, object] = {**summary, **fleet}
    goals = _read_csv(metrics_dir / "goal_metrics.csv")
    if "goal_count" not in source and goals:
        source["goal_count"] = len(goals)
        source["goals_completed"] = sum(
            _bool(row.get("completed"), False) is True for row in goals
        )
    model_calls = _read_csv(run / "model_calls.csv") or _read_csv(
        metrics_dir / "model_calls.csv"
    )
    if model_calls:
        if "prompt_tokens" not in source:
            source["prompt_tokens"] = sum(
                int(_number(row.get("prompt_tokens"), integer=True) or 0)
                for row in model_calls
            )
        if "completion_tokens" not in source:
            source["completion_tokens"] = sum(
                int(_number(row.get("completion_tokens"), integer=True) or 0)
                for row in model_calls
            )
        latencies = [
            value
            for row in model_calls
            if (value := _number(row.get("latency_s"))) is not None
        ]
        if "model_latency_s" not in source and latencies:
            source["model_latency_s"] = sum(float(value) for value in latencies) / len(latencies)
    source.setdefault("fleet_plan_success", (run / "fleet_plan.json").is_file())
    source.setdefault(
        "local_plan_success",
        any((run / "agents").glob("*/local_plan_v*.json")),
    )
    status = str(source.get("status", source.get("final_status", "UNKNOWN")))
    strict_default = status == "SUCCEEDED"
    strict = _bool(source.get("strict_success"), strict_default)
    semantic = _bool(source.get("semantic_success"), strict)
    execution = _bool(source.get("execution_success"), strict)
    collision_count = _number(source.get("collision_count"), integer=True) or 0
    out_of_bounds_count = _number(source.get("out_of_bounds_count"), integer=True) or 0
    safety = _bool(
        source.get("safety_success"),
        collision_count == 0 and out_of_bounds_count == 0,
    )
    goals_completed_value = _number(source.get("goals_completed"), integer=True) or 0
    partial = _bool(
        source.get("partial_success"),
        goals_completed_value > 0 and not strict,
    )
    failure_reason = source.get("failure_reason") or source.get("last_error")
    if not strict and not failure_reason:
        failure_reason = status if status not in {"", "UNKNOWN"} else "UNKNOWN_ERROR"
    codes = _error_codes(run, summary)
    record = {
        "run_id": str(source.get("fleet_mission_id") or source.get("run_id") or run.name),
        "status": status,
        "failure_reason": "" if strict else str(failure_reason),
        "strict_success": bool(strict),
        "semantic_success": bool(semantic),
        "execution_success": bool(execution),
        "safety_success": bool(safety),
        "partial_success": bool(partial),
        "interpreter_schema_success": _bool(source.get("interpreter_schema_success")),
        "fleet_plan_success": _bool(source.get("fleet_plan_success")),
        "local_plan_success": _bool(source.get("local_plan_success")),
        "repair_count": _number(source.get("repair_count"), integer=True) or 0,
        "repairs_succeeded": _number(source.get("repairs_succeeded"), integer=True) or 0,
        "reassignment_count": _number(source.get("reassignment_count"), integer=True) or 0,
        "reassignments_succeeded": _number(source.get("reassignments_succeeded"), integer=True) or 0,
        "goal_count": _number(source.get("goal_count"), integer=True) or 0,
        "goals_completed": goals_completed_value,
        "goal_completion_rate": _number(source.get("goal_completion_rate")),
        "prompt_tokens": _number(source.get("prompt_tokens"), integer=True) or 0,
        "completion_tokens": _number(source.get("completion_tokens"), integer=True) or 0,
        "model_latency_s": _number(source.get("model_latency_s")),
        "mission_sim_time_s": _number(source.get("mission_sim_time_s")),
        "mission_wall_time_s": _number(source.get("mission_wall_time_s")),
        "error_codes": "|".join(codes),
        "details_retained": False,
        "_source_run_dir": run,
        "_error_code_tuple": codes,
    }
    return record


class FleetBatchEvaluator:
    """Aggregate every episode while retaining at most 25 sanitized details."""

    def __init__(
        self,
        evaluation_root: str | Path,
        *,
        retain_successes: int = DEFAULT_RETAINED_SUCCESSES,
        retain_per_failure_type: int = DEFAULT_RETAINED_PER_FAILURE,
        max_retained_runs: int = MAX_RETAINED_RUNS,
        save_summary_figures: bool = True,
    ) -> None:
        self.evaluation_root = Path(evaluation_root).expanduser().resolve()
        self.evaluation_root.mkdir(parents=True, exist_ok=True)
        for value, name in (
            (retain_successes, "retain_successes"),
            (retain_per_failure_type, "retain_per_failure_type"),
            (max_retained_runs, "max_retained_runs"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if max_retained_runs > MAX_RETAINED_RUNS:
            raise ValueError(f"max_retained_runs must not exceed {MAX_RETAINED_RUNS}")
        self.retain_successes = retain_successes
        self.retain_per_failure_type = retain_per_failure_type
        self.max_retained_runs = max_retained_runs
        self.save_summary_figures = bool(save_summary_figures)
        self._episodes: list[dict[str, object]] = []

    @property
    def episodes(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._episodes)

    def add_run(self, run_dir: str | Path) -> dict[str, object]:
        record = load_fleet_episode_metrics(run_dir)
        if any(existing["run_id"] == record["run_id"] for existing in self._episodes):
            raise FleetBatchEvaluationError(f"duplicate run_id: {record['run_id']}")
        self._episodes.append(record)
        return dict(record)

    def add_episode(
        self,
        metrics: Mapping[str, object],
        *,
        source_run_dir: str | Path | None = None,
    ) -> None:
        record = dict(metrics)
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("episode metrics require run_id")
        if any(existing.get("run_id") == run_id for existing in self._episodes):
            raise FleetBatchEvaluationError(f"duplicate run_id: {run_id}")
        record.setdefault("details_retained", False)
        record["_source_run_dir"] = None if source_run_dir is None else Path(source_run_dir).resolve()
        raw_codes = record.get("error_codes", "")
        record["_error_code_tuple"] = tuple(
            str(item) for item in (raw_codes.split("|") if isinstance(raw_codes, str) else raw_codes)
            if item
        )
        self._episodes.append(record)

    def _select_retained(self) -> list[dict[str, object]]:
        selected: list[dict[str, object]] = []
        selected_ids: set[str] = set()
        successes = 0
        failures: dict[str, int] = {}
        seen_error_codes: set[str] = set()
        for record in self._episodes:
            if len(selected) >= self.max_retained_runs:
                break
            run_id = str(record["run_id"])
            strict = _bool(record.get("strict_success"), False) is True
            reason = str(record.get("failure_reason") or "UNKNOWN_ERROR")
            codes = tuple(record.get("_error_code_tuple", ()))
            new_code = any(code not in seen_error_codes for code in codes)
            retain = False
            if strict and successes < self.retain_successes:
                successes += 1
                retain = True
            elif not strict and failures.get(reason, 0) < self.retain_per_failure_type:
                failures[reason] = failures.get(reason, 0) + 1
                retain = True
            elif new_code:
                retain = True
            seen_error_codes.update(codes)
            if retain and run_id not in selected_ids:
                selected.append(record)
                selected_ids.add(run_id)
        return selected

    def _copy_details(self, record: Mapping[str, object]) -> None:
        source = record.get("_source_run_dir")
        if not isinstance(source, Path) or not source.is_dir():
            return
        destination = self.evaluation_root / "retained_runs" / str(record["run_id"])
        destination.mkdir(parents=True, exist_ok=True)
        candidates = [source / name for name in _ALLOWED_DETAIL_NAMES]
        metrics_dir = source / "metrics" if (source / "metrics").is_dir() else source
        candidates.extend(metrics_dir / name for name in _ALLOWED_METRIC_NAMES)
        for path in candidates:
            if not path.is_file() or path.stat().st_size > _DETAIL_FILE_LIMIT_BYTES:
                continue
            data = path.read_bytes()
            lowered = data.lower()
            if any(marker in lowered for marker in _FORBIDDEN_BYTES):
                continue
            relative = Path("metrics") / path.name if path.parent == metrics_dir and metrics_dir != source else Path(path.name)
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)

    @staticmethod
    def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(fields), extrasaction="ignore", lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in fields})
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _plot_summary(self, summary: Mapping[str, object]) -> tuple[Path, ...]:
        if not self.save_summary_figures or not self._episodes:
            return ()
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError:
            return ()
        paths = []
        stages = summary.get("stage_success_rate", {})
        if isinstance(stages, Mapping):
            measured = [(str(name), float(value)) for name, value in stages.items() if value is not None]
            if measured:
                fig, axis = plt.subplots(figsize=(7.2, 4.0))
                axis.bar([item[0] for item in measured], [item[1] for item in measured])
                axis.set(ylim=(0.0, 1.0), ylabel="Success rate", title="Fleet stage success")
                axis.tick_params(axis="x", rotation=25)
                path = self.evaluation_root / "stage_success_rate.png"
                fig.savefig(path, dpi=110, bbox_inches="tight")
                plt.close(fig)
                paths.append(path)
        breakdown = summary.get("failure_breakdown", {})
        if isinstance(breakdown, Mapping) and breakdown:
            ordered = sorted(((str(key), int(value)) for key, value in breakdown.items()), key=lambda item: (-item[1], item[0]))
            fig, axis = plt.subplots(figsize=(7.5, max(3.2, 0.35 * len(ordered))))
            axis.barh([item[0] for item in reversed(ordered)], [item[1] for item in reversed(ordered)])
            axis.set(xlabel="Episodes", title="Fleet failure breakdown")
            path = self.evaluation_root / "failure_breakdown.png"
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        return tuple(paths)

    def finalize(self) -> dict[str, object]:
        selected = self._select_retained()
        retained_ids = {str(record["run_id"]) for record in selected}
        for record in self._episodes:
            record["details_retained"] = str(record["run_id"]) in retained_ids
        for record in selected:
            self._copy_details(record)

        public_rows = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in self._episodes
        ]
        failures = []
        for record in public_rows:
            if _bool(record.get("strict_success"), False) is True:
                continue
            failures.append(
                {
                    "run_id": record.get("run_id"),
                    "failure_reason": record.get("failure_reason") or "UNKNOWN_ERROR",
                    "status": record.get("status"),
                    "code": record.get("error_codes"),
                    "message": "fleet batch episode failed",
                }
            )
        self._write_csv(
            self.evaluation_root / "episode_metrics.csv",
            BATCH_EPISODE_METRIC_FIELDS,
            public_rows,
        )
        self._write_csv(
            self.evaluation_root / "failure_cases.csv",
            FAILURE_CASE_FIELDS,
            failures,
        )
        summary = aggregate_fleet_episode_metrics(public_rows)
        summary["retained_run_count"] = len(selected)
        summary["retained_run_ids"] = sorted(retained_ids)
        summary["retention_policy"] = {
            "first_successes": self.retain_successes,
            "per_failure_type": self.retain_per_failure_type,
            "first_new_error_code": True,
            "max_total": self.max_retained_runs,
        }
        (self.evaluation_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._plot_summary(summary)
        return summary


def evaluate_fleet_run_directories(
    run_dirs: Sequence[str | Path],
    evaluation_root: str | Path,
    *,
    save_summary_figures: bool = True,
) -> dict[str, object]:
    evaluator = FleetBatchEvaluator(
        evaluation_root,
        save_summary_figures=save_summary_figures,
    )
    for run_dir in run_dirs:
        evaluator.add_run(run_dir)
    return evaluator.finalize()


__all__ = [
    "DEFAULT_RETAINED_PER_FAILURE",
    "DEFAULT_RETAINED_SUCCESSES",
    "MAX_RETAINED_RUNS",
    "FleetBatchEvaluationError",
    "FleetBatchEvaluator",
    "evaluate_fleet_run_directories",
    "load_fleet_episode_metrics",
]
