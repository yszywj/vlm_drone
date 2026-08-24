from __future__ import annotations

import csv
import json

import pytest

from experiments.fleet_batch_evaluator import FleetBatchEvaluator
from experiments.fleet_report import (
    ResultStatusMismatchError,
    generate_fleet_report,
)
from experiments.fleet_result_recorder import FleetResultRecorder
from experiments.schemas import AgentMetricRecord, GoalResultRecord, SkillExecutionRecord


def _make_run(root, run_id: str, *, status: str, failure_reason: str = ""):
    run = root / run_id
    recorder = FleetResultRecorder(run, fleet_mission_id=run_id)
    recorder.record_goal_result(
        GoalResultRecord(
            run_id,
            "goal_1",
            "TRACK_TARGET",
            status == "SUCCEEDED",
            uav_id="uav_a",
            evidence_source="runtime" if status == "SUCCEEDED" else None,
            unmet_reason=None if status == "SUCCEEDED" else failure_reason,
        )
    )
    recorder.record_agent_metrics(
        AgentMetricRecord(run_id, "uav_a", status, path_length_m=10.0, landed=True)
    )
    recorder.record_skill_execution(
        SkillExecutionRecord(run_id, "uav_a", "step_1", "TRACK", 1, 3, "TRACK_COMPLETE")
    )
    recorder.finalize(
        {
            "fleet_mission_id": run_id,
            "status": status,
            "original_instruction": "搜索目标并报告",
            "strict_success": status == "SUCCEEDED",
            "semantic_success": status == "SUCCEEDED",
            "execution_success": status == "SUCCEEDED",
            "safety_success": True,
            "partial_success": status != "SUCCEEDED",
            "goal_count": 1,
            "goals_completed": int(status == "SUCCEEDED"),
            "failure_reason": failure_reason,
        }
    )
    return run


def test_report_status_and_tables_match_summary_without_figures(tmp_path) -> None:
    run = _make_run(tmp_path, "fleet_report_ok", status="SUCCEEDED")
    path = generate_fleet_report(run, no_summary_figures=True)
    report = path.read_text(encoding="utf-8")
    assert "Final status: **SUCCEEDED**" in report
    assert "搜索目标并报告" in report
    assert "goal_1" in report
    assert "uav_a" in report
    assert not (run / "figures").exists()
    assert "camera image" in report


def test_report_reads_real_nested_local_plan_shape_with_bounded_precedence(tmp_path) -> None:
    run = _make_run(tmp_path, "fleet_report_nested_plans", status="SUCCEEDED")

    spatial_path = run / "agents/uav_spatial/local_plan_v7.json"
    spatial_path.parent.mkdir(parents=True)
    spatial_path.write_text(
        json.dumps(
            {
                "steps": [{"skill": "WRONG_TOP_LEVEL"}],
                "compiled_task_plan": {"steps": [{"skill": "WRONG_COMPILED"}]},
                "spatial_plan_draft_v3": {
                    "schema_version": 3,
                    "steps": [
                        {"id": "takeoff_1", "skill": "TAKEOFF", "args": {"altitude_m": 10.0}},
                        {"id": "orbit_1", "skill": "ORBIT", "args": {"radius_m": 8.0}},
                        {"id": "land_1", "skill": "LAND", "args": {}},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    compiled_path = run / "agents/uav_compiled/local_plan_v2.json"
    compiled_path.parent.mkdir(parents=True)
    compiled_path.write_text(
        json.dumps({"compiled_task_plan": {"steps": [{"skill": "SEARCH"}, {"skill": "TRACK"}]}}),
        encoding="utf-8",
    )

    top_level_path = run / "agents/uav_legacy/local_plan_v1.json"
    top_level_path.parent.mkdir(parents=True)
    top_level_path.write_text(
        json.dumps({"steps": [{"skill_name": "GOTO"}, {"type": "LAND"}]}),
        encoding="utf-8",
    )

    report = generate_fleet_report(run, no_summary_figures=True).read_text(encoding="utf-8")
    assert "| uav_spatial | 7 | TAKEOFF → ORBIT → LAND |" in report
    assert "| uav_compiled | 2 | SEARCH → TRACK |" in report
    assert "| uav_legacy | 1 | GOTO → LAND |" in report
    assert "WRONG_TOP_LEVEL" not in report
    assert "WRONG_COMPILED" not in report


def test_report_rejects_summary_csv_final_status_mismatch(tmp_path) -> None:
    run = _make_run(tmp_path, "fleet_report_mismatch", status="SUCCEEDED")
    path = run / "metrics/fleet_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = tuple(rows[0])
    rows[-1]["status"] = "FAILED"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ResultStatusMismatchError):
        generate_fleet_report(run, no_summary_figures=True)


def test_batch_writes_every_episode_and_bounds_detailed_retention(tmp_path) -> None:
    runs = tmp_path / "runs"
    successes = [_make_run(runs, f"success_{i}", status="SUCCEEDED") for i in range(8)]
    failures = [
        _make_run(runs, f"failed_{i}", status="FAILED", failure_reason="LOCAL_PLAN_FAILED")
        for i in range(8)
    ]
    evaluator = FleetBatchEvaluator(tmp_path / "evaluation", save_summary_figures=False)
    for run in successes + failures:
        evaluator.add_run(run)
    summary = evaluator.finalize()
    with (tmp_path / "evaluation/episode_metrics.csv").open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    with (tmp_path / "evaluation/failure_cases.csv").open(newline="") as stream:
        failure_rows = list(csv.DictReader(stream))
    assert len(episodes) == 16
    assert len(failure_rows) == 8
    assert summary["strict_success_rate"] == pytest.approx(0.5)
    assert summary["retained_run_count"] == 8  # first five success + first three failure
    assert len(list((tmp_path / "evaluation/retained_runs").iterdir())) == 8
    assert not list((tmp_path / "evaluation/retained_runs").rglob("*.png"))
