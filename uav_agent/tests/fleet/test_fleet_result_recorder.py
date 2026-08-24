from __future__ import annotations

import csv
import json

import pytest

from experiments.fleet_result_recorder import FleetResultRecorder
from experiments.planning_audit_logger import prompt_sha256
from experiments.schemas import (
    GoalResultRecord,
    PlanningAttemptRecord,
    SkillExecutionRecord,
    StateSampleRecord,
)


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_recorder_persists_typed_scalar_results_and_exact_1hz_samples(tmp_path) -> None:
    recorder = FleetResultRecorder(tmp_path, fleet_mission_id="fleet_result_test")
    for tick in range(101):
        timestamp = tick / 20.0
        recorder.record_state_sample(
            StateSampleRecord(
                "fleet_result_test",
                "uav_a",
                timestamp,
                (timestamp, 0.0, 10.0),
                mode="TRACK",
                target_detected=timestamp >= 1.0,
                target_locked=timestamp >= 2.0,
            )
        )
    recorder.record_goal_result(
        GoalResultRecord(
            "fleet_result_test",
            "goal_track",
            "TRACK_TARGET",
            True,
            uav_id="uav_a",
            completion_time_s=5.0,
            evidence_source="trusted_runtime",
        )
    )
    recorder.record_skill_execution(
        SkillExecutionRecord(
            "fleet_result_test",
            "uav_a",
            "step_track",
            "TRACK",
            1.0,
            5.0,
            "TRACK_COMPLETE",
        )
    )
    recorder.planning_audit.log_attempt(
        PlanningAttemptRecord(
            attempt_id="attempt_1",
            timestamp_s=0.0,
            stage="LOCAL_PLANNER",
            mission_id="fleet_result_test",
            model_role="AGENT_SPATIAL_PLAN",
            prompt_sha256=prompt_sha256("private prompt text"),
            prompt_schema_version="3",
            accepted=True,
            proposal={"schema_version": 3, "steps": []},
        )
    )
    summary = recorder.finalize(
        {
            "fleet_mission_id": "fleet_result_test",
            "status": "SUCCEEDED",
            "strict_success": True,
            "agent_statuses": {"uav_a": "SUCCEEDED"},
        }
    )

    samples = _rows(tmp_path / "metrics/state_samples_1hz.csv")
    assert len(samples) == 6
    assert [float(row["timestamp_s"]) for row in samples] == pytest.approx(
        [0, 1, 2, 3, 4, 5]
    )
    agents = _rows(tmp_path / "metrics/agent_metrics.csv")
    assert float(agents[-1]["path_length_m"]) == pytest.approx(5.0)
    assert summary["result_storage"]["state_samples_skipped_by_cadence"] == 95
    assert _rows(tmp_path / "metrics/fleet_metrics.csv")[-1]["status"] == "SUCCEEDED"
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "SUCCEEDED"

    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert "private prompt text" not in persisted
    assert "camera_rgb" not in persisted
    assert "base64," not in persisted


def test_audit_invalid_text_retains_only_redacted_bounded_tail(tmp_path) -> None:
    recorder = FleetResultRecorder(tmp_path, fleet_mission_id="fleet_invalid")
    assert recorder.planning_audit.log_invalid_attempt(
        attempt_id="attempt_invalid",
        timestamp_s=0.0,
        stage="INTERPRETER",
        mission_id="fleet_invalid",
        model_role="MISSION_INTERPRETATION",
        prompt="never persist this prompt",
        prompt_schema_version="1",
        raw_text="x" * 800 + " api_key=super-secret sk-abcdefghijk",
        error_codes=("INVALID_JSON",),
    )
    record = json.loads((tmp_path / "planning_attempts.jsonl").read_text())
    assert record["raw_text_length"] > 800
    assert len(record["raw_text_tail"]) <= 500
    assert "super-secret" not in record["raw_text_tail"]
    assert "sk-abcdefghijk" not in record["raw_text_tail"]
    assert "never persist this prompt" not in (tmp_path / "planning_attempts.jsonl").read_text()


@pytest.mark.parametrize(
    "payload",
    (
        {"camera_rgb": [[1, 2, 3]]},
        {"image_url": "data:image/png;base64,AAAA"},
        {"api_key": "secret"},
        {"prompt": "full hidden prompt"},
        {"observation": {"pose": [0, 0, 0]}},
    ),
)
def test_result_payload_contract_rejects_forbidden_data(tmp_path, payload) -> None:
    recorder = FleetResultRecorder(tmp_path, fleet_mission_id="fleet_forbidden")
    with pytest.raises(ValueError, match="forbidden|encoded media|credentials"):
        recorder.planning_audit.write_final_plan(
            stage="LOCAL_PLANNER",
            mission_id="fleet_forbidden",
            plan_version=1,
            plan=payload,
        )
