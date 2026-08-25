from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from configs.loader import load_config
from experiments.yolo_fixed_seed_eval import (
    DEFAULT_FIXED_SEEDS,
    aggregate_episode_records,
    build_runtime_command,
    materialize_seed_config,
    parse_fixed_seeds,
    run_fixed_seed_evaluation,
    summarize_episode_run,
    YoloFixedSeedEvaluationError,
    _run_bounded_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIG = PROJECT_ROOT / "configs/yolo/runtime_yolo26.yaml"


def _write_success_run(run: Path, *, seed: int) -> None:
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "fleet_mission_id": f"mission_seed_{seed}",
                "strict_success": True,
                "search_success": True,
                "time_to_first_detection_s": 2.0,
                "time_to_lock_s": 3.0,
                "valid_track_duration_s": 20.0,
                "return_success": True,
                "landing_success": True,
                "perception_by_uav": {
                    "uav_1": {
                        "yolo_requests": 10,
                        "yolo_successful_responses": 10,
                        "detections_total": 8,
                        "candidates_total": 8,
                        "candidates_confirmed": 1,
                        "color_observations": 4,
                        "color_matches": 3,
                        "color_mismatches": 0,
                        "color_pending": 1,
                        "depth_resolution_attempts": 8,
                        "depth_resolution_successes": 8,
                        "depth_resolution_failures": 0,
                        "measurement_created": 8,
                        "measurement_rejected": 0,
                        "track_id_switches": 0,
                    }
                },
                "yolo_services_by_uav": {
                    "uav_1": {"model_sha256": "a" * 64}
                },
            },
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_preflight_failure_run(run: Path, *, seed: int) -> None:
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": "FAILED_PREPARATION",
                "fleet_mission_id": f"mission_seed_{seed}",
                "strict_success": False,
                "stage": "PREFLIGHT",
                "last_error": "TargetPerceptionConfigurationError: worker rejected model",
                "exit_code": 2,
            },
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = run / "metrics"
    metrics.mkdir()
    (metrics / "failure_cases.csv").write_text(
        "run_id,phase,global_step,episode_id,scenario_id,failure_reason,"
        "terminal_skill,mission_sim_time_s,message,fleet_mission_id,"
        "assignment_id,uav_id,goal_id,stage,code,severity,status\n"
        f"mission_seed_{seed},,,,,,,,worker rejected model,mission_seed_{seed},"
        ",,,PREFLIGHT,MODEL_IDENTITY_MISMATCH,HARD_ACTION_BLOCK,FAILED_PREPARATION\n",
        encoding="utf-8",
    )


def test_fixed_seed_parser_requires_five_unique_nonnegative_values() -> None:
    assert parse_fixed_seeds("101,211,307,401,503") == DEFAULT_FIXED_SEEDS
    for invalid in ("1,2,3,4", "1,2,3,4,4", "1,2,3,4,-1", "1,,2,3,4,5"):
        try:
            parse_fixed_seeds(invalid)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion message is clearer than parametrization here
            raise AssertionError(f"invalid seed list was accepted: {invalid}")


def test_materialized_config_is_complete_loadable_and_changes_only_seeds(
    tmp_path: Path,
) -> None:
    output = tmp_path / "seed_211.yaml"
    original = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    materialize_seed_config(RUNTIME_CONFIG, output, seed=211)
    generated = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert generated["target"]["motion"]["seed"] == 211
    assert generated["experiment"]["seed"] == 211
    original["target"]["motion"]["seed"] = 211
    original["experiment"]["seed"] = 211
    assert generated == original
    config = load_config(output)
    assert config.targets[0].motion.seed == 211
    assert config.experiment.seed == 211


def test_runtime_command_is_production_yolo_and_never_enables_oracle(
    tmp_path: Path,
) -> None:
    command = build_runtime_command(
        project_root=PROJECT_ROOT,
        config=RUNTIME_CONFIG,
        output_root=tmp_path,
        instruction="search red cube and track twenty seconds",
        max_sim_time_s=300.0,
    )
    joined = " ".join(command)
    assert "scripts/run_fleet_mission.py" in command
    assert "--target-perception-mode yolo" in joined
    assert "--perception-runtime-profile production" in joined
    assert "--headless" in command
    assert "oracle" not in joined.casefold()
    assert "--acknowledge-privileged-oracle" not in command


def test_episode_summary_reports_detector_color_geometry_and_mission_stages(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _write_success_run(run, seed=101)
    record = summarize_episode_run(run, seed=101, process_exit_code=0)

    assert record["detection"]["success"] is True
    assert record["color"]["confirmed"] is True
    assert record["color"]["sensor_match_rate"] == 1.0
    assert record["geometry_3d"]["success"] is True
    assert record["search"]["success"] is True
    assert record["track"]["success"] is True
    assert record["land"]["success"] is True
    assert record["failure"]["stage"] is None
    assert record["oracle_metrics_used"] is False
    assert record["model_sha256"] == ["a" * 64]


def test_aggregate_keeps_failed_seed_and_identifies_first_failed_stage() -> None:
    successful = {
        "seed": 1,
        "strict_success": True,
        "detection": {"success": True},
        "color": {"confirmed": True, "matches": 2, "mismatches": 0},
        "geometry_3d": {"success": True},
        "search": {"success": True},
        "track": {"success": True},
        "land": {"success": True},
        "failure": {"stage": None},
        "model_sha256": ["a" * 64],
    }
    failed = {
        **successful,
        "seed": 2,
        "strict_success": False,
        "geometry_3d": {"success": False},
        "search": {"success": False},
        "track": {"success": False},
        "land": {"success": True},
        "failure": {"stage": "GEOMETRY_3D", "reason": "NO_VALID_DEPTH"},
    }
    summary = aggregate_episode_records((successful, failed))
    assert summary["episode_count"] == 2
    assert summary["strict_success_rate"] == 0.5
    assert summary["stage_success_rate"]["geometry_3d"] == 0.5
    assert summary["failure_stage_counts"] == {"GEOMETRY_3D": 1}
    assert summary["oracle_metrics_used"] is False


@dataclass
class _Completed:
    returncode: int = 0
    timed_out: bool = False


def test_five_seed_driver_reuses_runtime_command_and_writes_structured_summary(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...], *, cwd: Path, check: bool) -> _Completed:
        assert cwd == PROJECT_ROOT
        assert check is False
        calls.append(command)
        output_index = command.index("--output-root") + 1
        config_index = command.index("--config") + 1
        config = load_config(command[config_index])
        seed = config.targets[0].motion.seed
        run = Path(command[output_index]) / "runs/fleet_mission" / f"run_seed_{seed}"
        _write_success_run(run, seed=seed)
        return _Completed()

    evaluation_root = tmp_path / "evaluation"
    result = run_fixed_seed_evaluation(
        project_root=PROJECT_ROOT,
        source_config=RUNTIME_CONFIG,
        evaluation_root=evaluation_root,
        seeds=DEFAULT_FIXED_SEEDS,
        instruction="search red cube and track twenty seconds",
        max_sim_time_s=300.0,
        command_runner=fake_runner,
    )

    assert result.exit_code == 0
    assert len(calls) == 5
    assert result.summary["episode_count"] == 5
    assert result.summary["strict_success_rate"] == 1.0
    assert result.summary["stage_success_rate"] == {
        "detection": 1.0,
        "color_confirmation": 1.0,
        "geometry_3d": 1.0,
        "search": 1.0,
        "track": 1.0,
        "land": 1.0,
    }
    persisted = json.loads((evaluation_root / "summary.json").read_text(encoding="utf-8"))
    assert persisted["fixed_seeds"] == list(DEFAULT_FIXED_SEEDS)
    assert persisted["oracle_metrics_used"] is False
    assert not (evaluation_root / "summary.partial.json").exists()


def test_existing_output_root_is_rejected_without_overwrite_or_launch(
    tmp_path: Path,
) -> None:
    evaluation_root = tmp_path / "existing"
    evaluation_root.mkdir()
    marker = evaluation_root / "keep.txt"
    marker.write_text("owned by prior evaluation\n", encoding="utf-8")
    called = False

    def forbidden_runner(*args: object, **kwargs: object) -> _Completed:
        nonlocal called
        called = True
        return _Completed()

    try:
        run_fixed_seed_evaluation(
            project_root=PROJECT_ROOT,
            source_config=RUNTIME_CONFIG,
            evaluation_root=evaluation_root,
            seeds=DEFAULT_FIXED_SEEDS,
            instruction="search red cube and track twenty seconds",
            max_sim_time_s=300.0,
            command_runner=forbidden_runner,
        )
    except YoloFixedSeedEvaluationError as exc:
        assert "refusing to overwrite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("existing output root was accepted")
    assert called is False
    assert marker.read_text(encoding="utf-8") == "owned by prior evaluation\n"
    assert tuple(evaluation_root.iterdir()) == (marker,)


def test_nonzero_hard_exit_code_is_recorded_and_later_seeds_still_run(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def fake_runner(command: tuple[str, ...], *, cwd: Path, check: bool) -> _Completed:
        del cwd, check
        config = load_config(command[command.index("--config") + 1])
        seed = config.targets[0].motion.seed
        calls.append(seed)
        output = Path(command[command.index("--output-root") + 1])
        run = output / "runs/fleet_mission" / f"run_seed_{seed}"
        if seed == DEFAULT_FIXED_SEEDS[0]:
            _write_preflight_failure_run(run, seed=seed)
            return _Completed(returncode=2)
        _write_success_run(run, seed=seed)
        return _Completed(returncode=0)

    result = run_fixed_seed_evaluation(
        project_root=PROJECT_ROOT,
        source_config=RUNTIME_CONFIG,
        evaluation_root=tmp_path / "hard_exit_eval",
        seeds=DEFAULT_FIXED_SEEDS,
        instruction="search red cube and track twenty seconds",
        max_sim_time_s=300.0,
        command_runner=fake_runner,
    )

    assert calls == list(DEFAULT_FIXED_SEEDS)
    assert result.exit_code == 1
    failed = result.records[0]
    assert failed["process_exit_code"] == 2
    assert failed["status"] == "FAILED_PREPARATION"
    assert failed["failure"]["stage"] == "PREFLIGHT"
    assert failed["failure"]["reason"] == "MODEL_IDENTITY_MISMATCH"
    assert result.summary["episode_count"] == 5
    assert result.summary["strict_success_rate"] == 0.8


def test_actual_os_hard_exit_returncode_overrides_stale_success_summary(
    tmp_path: Path,
) -> None:
    run = tmp_path / "durable_success_before_wrapper_exit"
    _write_success_run(run, seed=101)
    completed = subprocess.run(
        [sys.executable, "-c", "import os; os._exit(7)"],
        check=False,
    )
    assert completed.returncode == 7

    record = summarize_episode_run(
        run,
        seed=101,
        process_exit_code=completed.returncode,
    )
    assert record["process_exit_code"] == 7
    assert record["strict_success"] is False
    assert record["failure"] == {
        "stage": "PROCESS_EXIT",
        "reason": "PROCESS_EXIT_7",
        "message": "runtime subprocess exited with code 7",
    }


def test_bounded_runner_terminates_a_hung_process_tree(tmp_path: Path) -> None:
    started = time.monotonic()
    result = _run_bounded_command(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        timeout_s=0.05,
    )
    assert result.returncode == 124
    assert result.timed_out is True
    assert time.monotonic() - started < 5.0


def test_episode_timeout_is_classified_and_remaining_seeds_continue(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def fake_runner(command: tuple[str, ...], *, cwd: Path, check: bool) -> _Completed:
        del cwd, check
        config = load_config(command[command.index("--config") + 1])
        seed = config.targets[0].motion.seed
        calls.append(seed)
        if seed == DEFAULT_FIXED_SEEDS[0]:
            return _Completed(returncode=124, timed_out=True)
        output = Path(command[command.index("--output-root") + 1])
        _write_success_run(
            output / "runs/fleet_mission" / f"run_seed_{seed}",
            seed=seed,
        )
        return _Completed()

    result = run_fixed_seed_evaluation(
        project_root=PROJECT_ROOT,
        source_config=RUNTIME_CONFIG,
        evaluation_root=tmp_path / "timeout_eval",
        seeds=DEFAULT_FIXED_SEEDS,
        instruction="search red cube and track twenty seconds",
        max_sim_time_s=300.0,
        episode_timeout_s=1.0,
        command_runner=fake_runner,
    )

    assert calls == list(DEFAULT_FIXED_SEEDS)
    assert result.exit_code == 1
    assert result.records[0]["failure"] == {
        "stage": "EPISODE_TIMEOUT",
        "reason": "episode exceeded wall-clock timeout of 1s",
        "message": "episode exceeded wall-clock timeout of 1s",
    }
    assert result.summary["failure_stage_counts"] == {"EPISODE_TIMEOUT": 1}
    assert result.summary["strict_success_rate"] == 0.8
