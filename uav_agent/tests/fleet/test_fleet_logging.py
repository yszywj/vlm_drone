from __future__ import annotations

import csv
import json

import pytest

from fleet.airspace_manager import FleetAirspaceManager, FleetPoseSnapshot, FleetUavPose
from fleet.logging import FleetMissionLogger
from fleet.world_belief import AgentFleetSummary, FleetWorldBelief


def test_logger_creates_expected_sparse_layout(tmp_path) -> None:
    logger = FleetMissionLogger(
        tmp_path,
        "fleet_mission_test",
        uav_ids=("uav_a", "uav_b"),
    )
    logger.write_run_manifest({"fleet_plan_version": 1})
    logger.write_fleet_plan({"schema_version": 1, "assignments": []})
    logger.write_assignments(
        (
            {
                "assignment_id": "assignment_a_i",
                "uav_id": "uav_a",
                "target_alias": "target_i",
                "priority": 100,
                "status": "RUNNING",
                "local_plan_version": 1,
            },
        )
    )
    logger.write_local_plan("uav_a", 1, {"steps": []})
    logger.log_fleet_event({"event_type": "FLEET_STARTED"})
    logger.log_agent_transition("uav_a", {"from": "TAKEOFF", "to": "SEARCH"})
    logger.log_model_call(
        {
            "call_id": "request_1",
            "call_role": "FLEET_PLAN",
            "uav_id": None,
            "priority": "P2_AGENT_RUNTIME_REPLAN",
            "requested_adapter": "fleet_planner",
            "adapter_status": "placeholder",
            "effective_model": "Qwen3-VL-4B-Instruct",
            "fallback_used": True,
            "stale_reasons": [],
        }
    )
    logger.write_summary({"status": "SUCCEEDED"})

    run_dir = tmp_path / "fleet_mission_test"
    assert json.loads((run_dir / "run_manifest.json").read_text())["fleet_plan_version"] == 1
    assert (run_dir / "fleet_events.jsonl").exists()
    assert (run_dir / "agents/uav_a/local_plan_v1.json").exists()
    assert (run_dir / "agents/uav_b").is_dir()
    for uav_id in ("uav_a", "uav_b"):
        for filename in (
            "transitions.jsonl",
            "visual_reviews.jsonl",
            "revisions.jsonl",
        ):
            assert (run_dir / "agents" / uav_id / filename).is_file()
    assert (run_dir / "airspace_conflicts.csv").read_text().startswith(
        "timestamp_s,"
    )
    with (run_dir / "model_calls.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["requested_adapter"] == "fleet_planner"
    assert row["effective_model"] == "Qwen3-VL-4B-Instruct"
    assert row["priority"] == "P2_AGENT_RUNTIME_REPLAN"


def test_zero_event_logger_has_stable_streams_and_exact_model_fields(tmp_path) -> None:
    logger = FleetMissionLogger(
        tmp_path,
        "fleet_mission_empty",
        uav_ids=("uav_a", "uav_b"),
    )
    run_dir = logger.run_dir
    assert (run_dir / "fleet_events.jsonl").read_text() == ""
    with (run_dir / "assignments.csv").open(newline="") as stream:
        assert tuple(next(csv.reader(stream))) == logger.ASSIGNMENT_FIELDS
    with (run_dir / "model_calls.csv").open(newline="") as stream:
        assert tuple(next(csv.reader(stream))) == logger.MODEL_FIELDS
    with (run_dir / "airspace_conflicts.csv").open(newline="") as stream:
        assert tuple(next(csv.reader(stream))) == logger.AIRSPACE_FIELDS


def test_existing_mission_directory_is_rejected_without_touching_old_logs(
    tmp_path,
) -> None:
    logger = FleetMissionLogger(tmp_path, "fleet_mission_existing")
    logger.log_fleet_event({"event_type": "OLD_EVENT"})
    logger.write_assignments(
        (
            {
                "assignment_id": "assignment_old",
                "uav_id": "uav_a",
                "target_alias": "target_old",
                "priority": 1,
                "status": "RUNNING",
                "local_plan_version": 1,
            },
        )
    )
    event_path = logger.run_dir / "fleet_events.jsonl"
    assignment_path = logger.run_dir / "assignments.csv"
    before = (event_path.read_bytes(), assignment_path.read_bytes())

    with pytest.raises(FileExistsError, match="refusing to mix records"):
        FleetMissionLogger(tmp_path, "fleet_mission_existing")

    assert (event_path.read_bytes(), assignment_path.read_bytes()) == before


def test_attach_run_dir_preserves_run_manager_and_existing_fleet_artifacts(
    tmp_path,
) -> None:
    run_dir = tmp_path / "runs/mission/20260824-120000"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    preserved = {
        run_dir / "manifest.yaml": b"status: running\n",
        run_dir / "logs/terminal.log": b"interpreter starting\n",
        run_dir / "metrics/episode_metrics.csv": b"episode_id,status\n1,ok\n",
        run_dir / "fleet_events.jsonl": b'{"event_type":"INTERPRETER_STARTED"}\n',
    }
    for path, payload in preserved.items():
        path.write_bytes(payload)

    logger = FleetMissionLogger.attach_run_dir(
        run_dir,
        "fleet_mission_attached",
        uav_ids=("uav_a",),
    )

    assert logger.run_dir == run_dir.resolve()
    assert logger.fleet_mission_id == "fleet_mission_attached"
    for path, payload in preserved.items():
        assert path.read_bytes() == payload
    assert (run_dir / "agents/uav_a/transitions.jsonl").is_file()
    with (run_dir / "assignments.csv").open(newline="") as stream:
        assert tuple(next(csv.reader(stream))) == logger.ASSIGNMENT_FIELDS
    logger.log_fleet_event({"event_type": "INTERPRETER_SUCCEEDED"})
    assert "INTERPRETER_STARTED" in (run_dir / "fleet_events.jsonl").read_text()


def test_attach_run_dir_keeps_existing_compatible_csv_and_rejects_bad_header(
    tmp_path,
) -> None:
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    assignments = run_dir / "assignments.csv"
    original = (
        ",".join(FleetMissionLogger.ASSIGNMENT_FIELDS)
        + "\nassignment_old,uav_a,target_old,1,RUNNING,1\n"
    ).encode()
    assignments.write_bytes(original)

    FleetMissionLogger.attach_run_dir(run_dir, "fleet_mission_existing")
    assert assignments.read_bytes() == original

    bad_run_dir = tmp_path / "bad"
    bad_run_dir.mkdir()
    bad_csv = bad_run_dir / "model_calls.csv"
    bad_csv.write_text("wrong,header\nold,data\n", encoding="utf-8")
    before = bad_csv.read_bytes()
    with pytest.raises(ValueError, match="incompatible header"):
        FleetMissionLogger.attach_run_dir(bad_run_dir, "fleet_mission_bad")
    assert bad_csv.read_bytes() == before
    assert not (bad_run_dir / "agents").exists()


def test_assignment_csv_overwrite_is_atomic_on_validation_failure(tmp_path) -> None:
    logger = FleetMissionLogger(tmp_path, "fleet_mission_atomic")
    original = {
        "assignment_id": "assignment_original",
        "uav_id": "uav_a",
        "target_alias": "target_i",
        "priority": 10,
        "status": "RUNNING",
        "local_plan_version": 1,
    }
    logger.write_assignments((original,))
    path = logger.run_dir / "assignments.csv"
    before = path.read_bytes()

    invalid = {**original, "assignment_id": "assignment_invalid", "priority": float("nan")}
    with pytest.raises(ValueError, match="finite JSON number"):
        logger.write_assignments((original, invalid))

    assert path.read_bytes() == before
    assert not path.with_suffix(".csv.tmp").exists()


def test_logger_rejects_non_finite_json_numbers(tmp_path) -> None:
    logger = FleetMissionLogger(tmp_path, "fleet_mission_finite")
    with pytest.raises(ValueError, match="finite JSON number"):
        logger.log_fleet_event({"timestamp_s": float("nan")})


def test_logger_rejects_images_and_credentials(tmp_path) -> None:
    logger = FleetMissionLogger(tmp_path, "fleet_mission_test")
    with pytest.raises(ValueError, match="must not be persisted"):
        logger.log_fleet_event({"camera_rgb": [[1, 2, 3]]})
    with pytest.raises(ValueError, match="must not be persisted"):
        logger.write_run_manifest({"api_key": "secret"})


def test_world_belief_exposes_only_sanitized_other_uav_summary() -> None:
    belief = FleetWorldBelief(
        fleet_mission_id="fleet_mission_test",
        fleet_plan_version=2,
        timestamp_s=5.0,
        agents={
            "uav_a": AgentFleetSummary(
                "uav_a", "assignment_a_i", "RUNNING", 3, "region_a", "HIGH"
            ),
            "uav_b": AgentFleetSummary(
                "uav_b", "assignment_b_j", "RUNNING", 1, "region_b", "LOW"
            ),
        },
    )
    assert belief.local_safety_summary("uav_a") == (
        {
            "uav_id": "uav_b",
            "assignment_id": "assignment_b_j",
            "current_region": "region_b",
            "altitude_layer": "LOW",
            "status": "RUNNING",
        },
    )
    with pytest.raises(ValueError, match="not allowed"):
        FleetWorldBelief(
            "fleet_mission_test",
            1,
            0.0,
            belief.agents,
            target_claims={"oracle_target_pose": [1, 2, 3]},
        )


def test_airspace_csv_is_written_from_decision(tmp_path) -> None:
    logger = FleetMissionLogger(tmp_path, "fleet_mission_test")
    decision = FleetAirspaceManager(5.0).evaluate(
        FleetPoseSnapshot(
            1.0,
            {
                "uav_a": FleetUavPose("uav_a", (0.0, 0.0, 10.0), priority=100),
                "uav_b": FleetUavPose("uav_b", (1.0, 0.0, 10.0), priority=10),
            },
        )
    )
    logger.log_airspace_decision(decision)
    assert (tmp_path / "fleet_mission_test/airspace_conflicts.csv").exists()
