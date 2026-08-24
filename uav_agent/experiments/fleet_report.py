"""Human-readable, scalar-only report for one fleet mission run."""

from __future__ import annotations

import csv
import json
from math import isfinite
import os
from pathlib import Path
from typing import Mapping

from .fleet_plotter import FleetPlotter
from .fleet_result_recorder import DEFAULT_MAX_RUN_BYTES
from fleet.strict_json import strict_json_object_loads


class FleetReportError(RuntimeError):
    """Raised when persisted result files are malformed or inconsistent."""


class ResultStatusMismatchError(FleetReportError):
    """Raised when summary.json and the terminal fleet metric disagree."""


def _json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    if path.stat().st_size > 2 * 1024 * 1024:
        raise FleetReportError(f"refusing oversized JSON input: {path.name}")
    try:
        value = strict_json_object_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FleetReportError(f"could not parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FleetReportError(f"{path.name} must contain a JSON object")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    if path.stat().st_size > 16 * 1024 * 1024:
        raise FleetReportError(f"refusing oversized CSV input: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _jsonl(path: Path, *, maximum_rows: int = 10_000) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= maximum_rows:
                break
            try:
                value = strict_json_object_loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _text(value: object, *, maximum: int = 500) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}" if isfinite(value) else "—"
    rendered = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return rendered[:maximum] + ("…" if len(rendered) > maximum else "")


def _truth(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _local_plan_steps(value: Mapping[str, object]) -> list[object]:
    """Return the authoritative bounded step list from a persisted local plan."""

    sources: tuple[object, ...] = (
        value.get("spatial_plan_draft_v3"),
        value,
        value.get("compiled_task_plan"),
    )
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        steps = source.get("steps")
        if isinstance(steps, list):
            return steps[:20]
    return []


class FleetReportGenerator:
    """Build ``report.md`` plus, optionally, the approved four figures."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        save_summary_figures: bool = True,
        max_run_bytes: int = DEFAULT_MAX_RUN_BYTES,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.metrics_dir = self.run_dir / "metrics" if (self.run_dir / "metrics").is_dir() else self.run_dir
        self.save_summary_figures = bool(save_summary_figures)
        if isinstance(max_run_bytes, bool) or not isinstance(max_run_bytes, int) or max_run_bytes <= 0:
            raise ValueError("max_run_bytes must be a positive integer")
        self.max_run_bytes = max_run_bytes

    def _load_context(self) -> dict[str, object]:
        summary = _json(self.run_dir / "summary.json")
        manifest = _json(self.run_dir / "run_manifest.json")
        if not manifest:
            # RunManager uses YAML; loading it here is safe and remains optional.
            manifest_path = self.run_dir / "manifest.yaml"
            if manifest_path.is_file() and manifest_path.stat().st_size <= 1024 * 1024:
                try:
                    import yaml

                    candidate = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(candidate, dict):
                        manifest = candidate
                except (ImportError, OSError, ValueError):
                    pass
        fleet_plan = _json(self.run_dir / "fleet_plan.json")
        task_spec = _json(self.run_dir / "fleet_task_spec.json") or _json(self.run_dir / "task_spec.json")
        fleet_metrics = _csv(self.metrics_dir / "fleet_metrics.csv")
        if summary and fleet_metrics:
            summary_status = summary.get("status", summary.get("final_status"))
            metric_status = fleet_metrics[-1].get("status")
            if summary_status and metric_status and str(summary_status) != str(metric_status):
                raise ResultStatusMismatchError(
                    f"summary status {summary_status!r} != fleet_metrics status {metric_status!r}"
                )
        return {
            "summary": summary,
            "manifest": manifest,
            "fleet_plan": fleet_plan,
            "task_spec": task_spec,
            "fleet_metrics": fleet_metrics,
            "agent_metrics": _csv(self.metrics_dir / "agent_metrics.csv"),
            "goal_metrics": _csv(self.metrics_dir / "goal_metrics.csv"),
            "skills": _csv(self.metrics_dir / "skill_executions.csv"),
            "failures": _csv(self.metrics_dir / "failure_cases.csv"),
            "attempts": _jsonl(self.run_dir / "planning_attempts.jsonl"),
            "findings": _jsonl(self.run_dir / "validation_findings.jsonl"),
            "recoveries": _jsonl(self.run_dir / "recovery_actions.jsonl"),
            "model_calls": _csv(self.run_dir / "model_calls.csv") or _csv(self.metrics_dir / "model_calls.csv"),
        }

    @staticmethod
    def _instruction(context: Mapping[str, object]) -> object:
        for source_name, field_name in (
            ("summary", "original_instruction"),
            ("task_spec", "source_text"),
            ("manifest", "original_instruction"),
            ("manifest", "instruction"),
        ):
            source = context.get(source_name)
            if isinstance(source, Mapping) and source.get(field_name):
                return source[field_name]
        return "not recorded"

    @staticmethod
    def _assignments(context: Mapping[str, object]) -> list[Mapping[str, object]]:
        plan = context.get("fleet_plan")
        raw = plan.get("assignments") if isinstance(plan, Mapping) else None
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, Mapping)]
        summary = context.get("summary")
        raw = summary.get("assignments") if isinstance(summary, Mapping) else None
        if isinstance(raw, Mapping):
            return [item for item in raw.values() if isinstance(item, Mapping)]
        return []

    def _local_plan_rows(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for path in sorted(self.run_dir.glob("agents/*/local_plan_v*.json")):
            value = _json(path)
            names = []
            for step in _local_plan_steps(value):
                if isinstance(step, Mapping):
                    names.append(str(step.get("skill", step.get("skill_name", step.get("type", "?")))))
            rows.append((path.parent.name, path.stem.rsplit("_v", 1)[-1], " → ".join(names) or "—"))
        return rows

    def _render(self, context: Mapping[str, object], figures: tuple[Path, ...]) -> str:
        summary = context["summary"] if isinstance(context["summary"], Mapping) else {}
        status = summary.get("status", summary.get("final_status", "UNKNOWN"))
        mission_id = summary.get("fleet_mission_id", self.run_dir.name)
        lines = [
            "# Fleet mission report",
            "",
            f"- Mission: `{_text(mission_id)}`",
            f"- Final status: **{_text(status)}**",
            "- Result contract: scalar/structured data only; no camera images, video, raw frames, prompts, or full observations.",
            "",
            "## Original instruction",
            "",
            _text(self._instruction(context), maximum=2_000),
            "",
            "## Qwen task interpretation",
            "",
        ]
        task_spec = context.get("task_spec")
        goals = task_spec.get("goals", []) if isinstance(task_spec, Mapping) else []
        ambiguities = task_spec.get("ambiguities", []) if isinstance(task_spec, Mapping) else []
        lines.append(f"Goals interpreted: {len(goals) if isinstance(goals, list) else 0}; ambiguities: {len(ambiguities) if isinstance(ambiguities, list) else 0}.")
        if isinstance(goals, list):
            for goal in goals[:50]:
                if isinstance(goal, Mapping):
                    lines.append(f"- `{_text(goal.get('goal_id'))}`: {_text(goal.get('goal_type', goal.get('type')))}")

        lines.extend(["", "## Final assignments", "", "| Assignment | UAV | Goals/target | Priority | Status |", "|---|---|---|---:|---|"])
        assignments = self._assignments(context)
        statuses = summary.get("assignments", {}) if isinstance(summary, Mapping) else {}
        for item in assignments[:100]:
            assignment_id = item.get("assignment_id")
            detail = statuses.get(assignment_id, {}) if isinstance(statuses, Mapping) else {}
            referenced = item.get("goal_ids", item.get("target_alias", "—"))
            if isinstance(referenced, list):
                referenced = ", ".join(str(value) for value in referenced)
            lines.append(
                f"| {_text(assignment_id)} | {_text(item.get('uav_id'))} | {_text(referenced)} | {_text(item.get('priority'))} | {_text(detail.get('status') if isinstance(detail, Mapping) else None)} |"
            )
        if not assignments:
            lines.append("| — | — | — | — | — |")

        lines.extend(["", "## Local plans", "", "| UAV | Version | Skill sequence |", "|---|---:|---|"])
        local_rows = self._local_plan_rows()
        for uav_id, version, sequence in local_rows:
            lines.append(f"| {_text(uav_id)} | {_text(version)} | {_text(sequence)} |")
        if not local_rows:
            lines.append("| — | — | No persisted final local plan |")

        goal_rows = context.get("goal_metrics", [])
        lines.extend(["", "## Goal results", "", "| Goal | Type | UAV | Completed | Evidence / unmet reason |", "|---|---|---|---|---|"])
        if isinstance(goal_rows, list):
            for row in goal_rows[:200]:
                evidence = row.get("evidence_source") if _truth(row.get("completed")) else row.get("unmet_reason")
                lines.append(f"| {_text(row.get('goal_id'))} | {_text(row.get('goal_type'))} | {_text(row.get('uav_id'))} | {_text(_truth(row.get('completed')))} | {_text(evidence)} |")
        if not goal_rows:
            lines.append("| — | — | — | — | No goal metrics recorded |")

        recoveries = context.get("recoveries", [])
        attempts = context.get("attempts", [])
        reassignments = [row for row in recoveries if "REASSIGN" in str(row.get("action", "")).upper()] if isinstance(recoveries, list) else []
        lines.extend([
            "",
            "## Repair and reassignment",
            "",
            f"- Planning attempts: {len(attempts) if isinstance(attempts, list) else 0}",
            f"- Recovery actions: {len(recoveries) if isinstance(recoveries, list) else 0}",
            f"- Reassignments: {len(reassignments)}",
        ])

        findings = context.get("findings", [])
        failures = context.get("failures", [])
        safety = [row for row in findings if str(row.get("severity", "")).upper() in {"FATAL_SAFETY", "HARD_ACTION_BLOCK"}] if isinstance(findings, list) else []
        lines.extend([
            "",
            "## Safety and failures",
            "",
            f"- Safety/blocking findings: {len(safety)}",
            f"- Failure cases: {len(failures) if isinstance(failures, list) else 0}",
        ])

        calls = context.get("model_calls", [])
        def numeric_rows(field: str) -> list[float]:
            result: list[float] = []
            if not isinstance(calls, list):
                return result
            for row in calls:
                try:
                    number = float(row.get(field, ""))
                except (TypeError, ValueError):
                    continue
                if isfinite(number):
                    result.append(number)
            return result

        prompt_tokens = int(sum(numeric_rows("prompt_tokens")))
        completion_tokens = int(sum(numeric_rows("completion_tokens")))
        latencies = numeric_rows("latency_s")
        lines.extend([
            "",
            "## Model calls",
            "",
            f"- Calls: {len(calls) if isinstance(calls, list) else 0}",
            f"- Prompt/completion tokens: {prompt_tokens} / {completion_tokens}",
            f"- Mean latency: {sum(latencies) / len(latencies):.3f} s" if latencies else "- Mean latency: —",
        ])

        agents = context.get("agent_metrics", [])
        lines.extend(["", "## Agent results", "", "| UAV | Status | Path (m) | Airborne (s) | HOVER (s) | HOLD (s) | Landed |", "|---|---|---:|---:|---:|---:|---|"])
        if isinstance(agents, list):
            for row in agents[:100]:
                lines.append(f"| {_text(row.get('uav_id'))} | {_text(row.get('status'))} | {_text(row.get('path_length_m'))} | {_text(row.get('airborne_time_s'))} | {_text(row.get('hover_time_s'))} | {_text(row.get('hold_time_s'))} | {_text(row.get('landed'))} |")
        if not agents:
            lines.append("| — | — | — | — | — | — | — |")

        if figures:
            lines.extend(["", "## Summary figures", ""])
            for figure in figures:
                lines.append(f"- [{figure.name}](figures/{figure.name})")

        lines.extend(["", "## Key files", ""])
        candidates = (
            "summary.json",
            "fleet_task_spec.json",
            "fleet_plan.json",
            "planning_attempts.jsonl",
            "validation_findings.jsonl",
            "recovery_actions.jsonl",
            "metrics/fleet_metrics.csv",
            "metrics/agent_metrics.csv",
            "metrics/goal_metrics.csv",
            "metrics/skill_executions.csv",
            "metrics/state_samples_1hz.csv",
            "metrics/failure_cases.csv",
            "model_calls.csv",
        )
        for relative in candidates:
            if (self.run_dir / relative).is_file():
                lines.append(f"- [{relative}]({relative})")
        return "\n".join(lines) + "\n"

    def generate(self, *, no_summary_figures: bool = False) -> Path:
        context = self._load_context()
        figures = FleetPlotter(
            self.run_dir,
            enabled=self.save_summary_figures and not no_summary_figures,
            max_run_bytes=self.max_run_bytes,
        ).generate_all()
        content = self._render(context, figures)
        encoded = content.encode("utf-8")
        if len(encoded) > 128 * 1024:
            encoded = encoded[: 128 * 1024 - 32] + b"\n\n[report truncated]\n"
        path = self.run_dir / "report.md"
        previous = path.stat().st_size if path.exists() else 0
        current_size = sum(item.stat().st_size for item in self.run_dir.rglob("*") if item.is_file())
        if current_size - previous + len(encoded) > self.max_run_bytes:
            encoded = (
                f"# Fleet mission report\n\n- Final status: **{_text(context['summary'].get('status', 'UNKNOWN') if isinstance(context['summary'], Mapping) else 'UNKNOWN')}**\n"
                "- Detail omitted because the bounded run storage limit was reached.\n"
            ).encode("utf-8")
        temporary = path.with_suffix(".md.tmp")
        try:
            with temporary.open("wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


FleetReport = FleetReportGenerator


def generate_fleet_report(
    run_dir: str | Path,
    *,
    save_summary_figures: bool = True,
    no_summary_figures: bool = False,
    max_run_bytes: int = DEFAULT_MAX_RUN_BYTES,
) -> Path:
    return FleetReportGenerator(
        run_dir,
        save_summary_figures=save_summary_figures,
        max_run_bytes=max_run_bytes,
    ).generate(no_summary_figures=no_summary_figures)


__all__ = [
    "FleetReport",
    "FleetReportError",
    "FleetReportGenerator",
    "ResultStatusMismatchError",
    "generate_fleet_report",
]
