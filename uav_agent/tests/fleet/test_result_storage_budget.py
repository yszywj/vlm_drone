from __future__ import annotations

import csv
from dataclasses import replace
import json

import pytest
import yaml

from configs.loader import ConfigError, load_config
from configs.schema import ResultsConfig
from experiments.fleet_result_recorder import (
    DEFAULT_MAX_RUN_BYTES,
    FleetResultRecorder,
    ResultRecorderError,
)
from experiments.schemas import StateSampleRecord
from fleet.logging import FleetMissionLogger


def test_stream_and_record_limits_drop_only_that_stream_without_interrupting(tmp_path) -> None:
    recorder = FleetResultRecorder(
        tmp_path,
        fleet_mission_id="fleet_budget",
        max_record_bytes=1024,
        max_stream_bytes=1500,
        max_run_bytes=12_000,
    )
    proposal = {"schema_version": 1, "items": ["x" * 700]}
    results = [
        recorder.planning_audit.write_final_plan(
            stage="LOCAL_PLANNER",
            mission_id="fleet_budget",
            plan_version=index + 1,
            plan=proposal,
        )
        for index in range(10)
    ]
    assert any(results)
    assert not all(results)

    # A truncated planning stream does not prevent independent state/summary streams.
    assert recorder.record_state_sample(
        StateSampleRecord("fleet_budget", "uav_a", 0.0, (0.0, 0.0, 1.0))
    )
    summary = recorder.finalize({"status": "FAILED", "failure_reason": "TEST"})
    assert (tmp_path / "summary.json").is_file()
    assert "final_plans.jsonl" in summary["result_storage"]["truncated_streams"]
    assert summary["result_storage"]["dropped_record_count"] > 0
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) <= 12_000


def test_five_minute_60hz_feed_persists_only_1hz_rows(tmp_path) -> None:
    recorder = FleetResultRecorder(tmp_path, fleet_mission_id="fleet_five_minutes")
    for tick in range(60 * 300 + 1):
        timestamp = tick / 60.0
        recorder.record_state_sample(
            StateSampleRecord(
                "fleet_five_minutes",
                "uav_a",
                timestamp,
                (timestamp * 0.01, 0.0, 10.0),
            )
        )
    recorder.finalize({"status": "SUCCEEDED"})
    with (tmp_path / "metrics/state_samples_1hz.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 301
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) < 2 * 1024 * 1024


def test_results_config_is_strict_and_forbids_heavy_artifacts(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    source = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    source["results"]["save_camera_images"] = True
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="must not be persisted"):
        load_config(config_path)

    source["results"]["save_camera_images"] = False
    source["results"]["unknown"] = 1
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(config_path)

    with pytest.raises(ValueError, match="must not exceed 33554432"):
        replace(ResultsConfig(), max_run_bytes=33_554_433)


def test_existing_forbidden_artifact_directory_is_rejected(tmp_path) -> None:
    (tmp_path / "camera_images").mkdir()
    with pytest.raises(ResultRecorderError, match="forbidden result directory"):
        FleetResultRecorder(tmp_path)


def test_standard_default_is_a_32_mib_hard_limit() -> None:
    assert DEFAULT_MAX_RUN_BYTES == 32 * 1024 * 1024
    assert ResultsConfig().max_run_bytes == DEFAULT_MAX_RUN_BYTES


def test_summary_reserve_cannot_be_consumed_by_high_priority_stream(tmp_path) -> None:
    recorder = FleetResultRecorder(
        tmp_path,
        fleet_mission_id="fleet_summary_reserve",
        max_record_bytes=256,
        max_stream_bytes=1000,
        max_run_bytes=4000,
    )
    # Final plans are high priority, but they must not consume summary space.
    for plan_version in range(1, 40):
        recorder.planning_audit.write_final_plan(
            stage="LOCAL_PLANNER",
            mission_id="fleet_summary_reserve",
            plan_version=plan_version,
            plan={"items": ["bounded plan content " * 20]},
        )

    summary = recorder.finalize({"status": "FAILED"})
    actual_bytes = sum(
        path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()
    )
    persisted = yaml.safe_load((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert actual_bytes <= 4000
    assert summary["result_storage"]["final_run_bytes"] == actual_bytes
    assert persisted["result_storage"]["final_run_bytes"] == actual_bytes
    assert persisted["result_storage"]["dropped_record_count"] > 0
    assert "final_plans.jsonl" in persisted["result_storage"]["truncated_streams"]


def test_fleet_logger_stream_cap_drops_records_and_merges_summary(tmp_path) -> None:
    logger = FleetMissionLogger(
        tmp_path,
        "fleet_logger_budget",
        max_record_bytes=512,
        max_stream_bytes=1000,
        max_run_bytes=5000,
    )
    for sequence in range(30):
        # Legal scalar diagnostics overfill only fleet_events.jsonl. Logging
        # remains best effort and must never become a mission exception.
        logger.log_fleet_event(
            {"sequence": sequence, "detail": "bounded event text " * 10}
        )
    logger.log_model_call(
        {
            "call_id": "call_after_event_cap",
            "call_role": "FLEET_PLAN",
            "fleet_mission_id": "fleet_logger_budget",
        }
    )
    logger.write_summary({"status": "SUCCEEDED"})

    run_dir = logger.run_dir
    actual_bytes = sum(
        path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert actual_bytes <= 5000
    assert summary["result_storage"]["final_run_bytes"] == actual_bytes
    assert summary["result_storage"]["dropped_record_count"] > 0
    assert "fleet_events.jsonl" in summary["result_storage"]["truncated_streams"]
    assert "call_after_event_cap" in (run_dir / "model_calls.csv").read_text()
    forbidden = {"videos", "camera_images", "raw_frames", "observation_dumps"}
    assert not forbidden.intersection(path.name for path in run_dir.rglob("*") if path.is_dir())


@pytest.mark.parametrize(
    "payload",
    [
        {"observation": {"pose": [1, 2, 3]}},
        {"value": "data:video/mp4;base64,AAAA"},
        {"value": "A" * 200},
    ],
)
def test_fleet_logger_rejects_observations_and_encoded_media(tmp_path, payload) -> None:
    logger = FleetMissionLogger(tmp_path, "fleet_no_media")
    with pytest.raises(ValueError, match="must not"):
        logger.log_fleet_event(payload)


def test_fleet_logger_budget_survives_reattach_and_recorder_summary(tmp_path) -> None:
    run_dir = tmp_path / "shared_run"
    run_dir.mkdir()
    first = FleetMissionLogger.attach_run_dir(
        run_dir,
        "fleet_shared_budget",
        max_record_bytes=512,
        max_stream_bytes=900,
        max_run_bytes=6000,
    )
    for sequence in range(20):
        first.log_fleet_event(
            {"sequence": sequence, "detail": "fleet bounded record " * 10}
        )
    first_snapshot = first.storage_snapshot()
    assert first_snapshot["dropped_record_count"] > 0

    second = FleetMissionLogger.attach_run_dir(
        run_dir,
        "fleet_shared_budget",
        max_record_bytes=512,
        max_stream_bytes=900,
        max_run_bytes=6000,
    )
    assert second.storage_snapshot()["dropped_records"] == first_snapshot["dropped_records"]

    recorder = FleetResultRecorder(
        run_dir,
        fleet_mission_id="fleet_shared_budget",
        max_record_bytes=512,
        max_stream_bytes=1000,
        max_run_bytes=6000,
    )
    summary = recorder.finalize({"status": "SUCCEEDED"})
    assert summary["result_storage"]["fleet_logger_dropped_records"] == first_snapshot[
        "dropped_records"
    ]
    second.write_summary(summary)
    persisted = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert persisted["result_storage"]["fleet_logger_dropped_records"] == first_snapshot[
        "dropped_records"
    ]
    assert persisted["result_storage"]["dropped_record_count"] == summary[
        "result_storage"
    ]["dropped_record_count"]
    sidecar = run_dir / ".fleet_logger_storage.json"
    assert sidecar.stat().st_size <= 4096
    assert "detail" not in sidecar.read_text(encoding="utf-8")


def test_prompt_hash_schema_and_token_metadata_are_allowed_but_raw_prompt_is_not(
    tmp_path,
) -> None:
    logger = FleetMissionLogger(tmp_path, "fleet_prompt_metadata")
    logger.log_fleet_event(
        {
            "prompt_sha256": "a" * 64,
            "prompt_schema_version": "fleet-v2",
            "prompt_tokens": 42,
            "completion_tokens": 17,
        }
    )
    persisted = (logger.run_dir / "fleet_events.jsonl").read_text(encoding="utf-8")
    assert '"prompt_sha256"' in persisted
    assert '"prompt_schema_version"' in persisted
    assert '"prompt_tokens": 42' in persisted
    with pytest.raises(ValueError, match="must not be persisted"):
        logger.log_fleet_event({"prompt": "raw model prompt"})
