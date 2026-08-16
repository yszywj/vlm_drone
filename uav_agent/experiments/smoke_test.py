"""Pure-Python end-to-end smoke test for lightweight experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

from .checkpoint_manager import CheckpointManager
from .evaluator import Evaluator, compute_mission_success_strict
from .metric_logger import MetricLogger
from .plotter import ExperimentPlotter
from .run_manager import FORBIDDEN_ARTIFACT_DIRECTORIES, RunManager, RunStatus
from .runtime import ExperimentRuntime
from .schemas import FailureReason
from .terminal_logger import TerminalLogger


_FIGURES = (
    "train_success_rate.png",
    "eval_success_rate.png",
    "final_success_rate.png",
    "stage_success_rate.png",
    "failure_breakdown.png",
    "training_curve.png",
)


class _FakeEpisodeRunner:
    """Deterministic fake with successes and several meaningful failures."""

    def __init__(self) -> None:
        self.loaded_checkpoint: Path | None = None

    def load_checkpoint(self, path: Path) -> None:
        self.loaded_checkpoint = Path(path)

    def run_episode(self, *, seed: int, phase: str, deterministic: bool) -> Mapping[str, object]:
        if not deterministic:
            raise AssertionError("smoke evaluation must disable exploration")
        case = seed % 5
        row: dict[str, object] = {
            "phase": phase,
            "seed": seed,
            "scenario_id": f"scenario_{seed}",
            "takeoff_success": True,
            "goto_search_success": True,
            "search_success": True,
            "correct_target_locked": True,
            "false_target_lock": False,
            "track_success": True,
            "reacquire_triggered": seed % 3 == 0,
            "reacquire_success": seed % 3 == 0,
            "return_success": True,
            "landing_success": True,
            "collision": False,
            "out_of_bounds": False,
            "safety_abort": False,
            "timeout": False,
            "time_to_first_detection_s": 2.0 + (seed % 4) * 0.1,
            "time_to_correct_lock_s": 3.0 + (seed % 4) * 0.1,
            "valid_track_duration_s": 30.0,
            "mission_sim_time_s": 48.0 + (seed % 7),
            "mission_wall_time_s": 0.02,
            "path_length_m": 38.0 + (seed % 5),
            "episode_return": 15.0,
        }
        if case == 0:
            row.update(
                search_success=False,
                correct_target_locked=False,
                track_success=False,
                failure_reason=FailureReason.TARGET_NOT_FOUND.value,
                episode_return=2.0,
            )
        elif case == 1:
            row.update(
                correct_target_locked=False,
                false_target_lock=True,
                track_success=False,
                failure_reason=FailureReason.FALSE_TARGET_LOCK.value,
                episode_return=-2.0,
            )
        elif case == 2:
            row.update(
                collision=True,
                return_success=False,
                landing_success=False,
                failure_reason=FailureReason.COLLISION.value,
                episode_return=-5.0,
            )
        elif case == 3:
            row.update(
                landing_success=False,
                timeout=True,
                failure_reason=FailureReason.LAND_TIMEOUT.value,
                episode_return=5.0,
            )
        return row


def _resolved_config(output_root: str | Path | None) -> dict[str, object]:
    return {
        "experiment": {"name": "output_smoke", "seed": 42, "output_root": None if output_root is None else str(output_root)},
        "logging": {"terminal": True, "console_log_interval_updates": 10, "print_every_episode": False, "debug_logging": False, "csv": True},
        "tensorboard": {"enabled": True, "scalars_only": True, "log_interval_updates": 1, "flush_interval_s": 30},
        "checkpoint": {"save_best": True, "save_latest": True, "latest_interval_steps": 5, "save_periodic": False, "save_full_base_model": False, "save_adapter_only": True, "save_optimizer_in_latest_only": True},
        "evaluation": {"enabled": True, "interval_steps": 20_000, "num_validation_episodes": 10, "num_test_episodes": 20, "deterministic": True, "fixed_validation_seeds": True, "fixed_test_seeds": True},
        "artifacts": {"save_images": False, "save_videos": False, "save_trajectories": False, "save_observations": False, "save_raw_frames": False},
        "figures": {"enabled": True, "format": "png", "save_pdf": False},
        "storage": {"min_free_space_gb_before_start": 20, "min_free_space_gb_during_run": 10, "warning_run_size_gb": 3, "max_run_size_gb": 5},
    }


def _failure_reason(row: Mapping[str, object]) -> str | None:
    value = row.get("failure_reason")
    return str(value) if value not in (None, "") else None


def _log_evaluation_episodes(
    logger: MetricLogger,
    episodes: tuple[dict[str, object], ...],
    *,
    run_id: str,
    global_step: int,
) -> None:
    for index, raw in enumerate(episodes):
        row = dict(raw)
        row.update(
            run_id=run_id,
            global_step=global_step,
            episode_id=index,
            scenario_id=row.get("scenario_id", f"{row['phase']}_{row['seed']}"),
        )
        if not compute_mission_success_strict(row) and not _failure_reason(row):
            row["failure_reason"] = FailureReason.UNKNOWN_ERROR.value
        logger.log_episode(row)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def verify_smoke_run(run_dir: str | Path, *, max_size_mb: float = 20.0) -> dict[str, object]:
    """Validate the exact compact output contract and return a size summary."""

    run = Path(run_dir)
    required = (
        "manifest.yaml",
        "resolved_config.yaml",
        "command.sh",
        "exit_code.txt",
        "logs/terminal.log",
        "metrics/train_metrics.csv",
        "metrics/eval_metrics.csv",
        "metrics/episode_metrics.csv",
        "metrics/failure_cases.csv",
        "metrics/final_metrics.csv",
        "checkpoints/best/checkpoint_meta.json",
        "checkpoints/latest/checkpoint_meta.json",
        *tuple(f"figures/{name}" for name in _FIGURES),
    )
    missing = [relative for relative in required if not (run / relative).is_file()]
    if missing:
        raise RuntimeError(f"smoke output is missing required files: {missing}")
    event_files = tuple((run / "tensorboard").glob("events.out.tfevents.*"))
    if len(event_files) != 1 or event_files[0].stat().st_size == 0:
        raise RuntimeError("smoke output must contain one non-empty scalar TFEvent file")
    forbidden = [
        str(path.relative_to(run))
        for path in run.rglob("*")
        if path.is_dir() and path.name in FORBIDDEN_ARTIFACT_DIRECTORIES
    ]
    forbidden.extend(
        str(path.relative_to(run))
        for path in run.rglob("step_*")
        if path.is_dir()
    )
    if forbidden:
        raise RuntimeError(f"smoke output contains forbidden artifact directories: {forbidden}")
    raw_extensions = {
        ".jpg", ".jpeg", ".bmp", ".gif", ".webp",
        ".mp4", ".mov", ".avi", ".mkv", ".webm",
        ".npy", ".npz",
    }
    unexpected = [
        str(path.relative_to(run))
        for path in run.rglob("*")
        if path.is_file() and path.suffix.lower() in raw_extensions
    ]
    if unexpected:
        raise RuntimeError(f"smoke output contains raw media/array artifacts: {unexpected}")
    checkpoint_children = sorted(path.name for path in (run / "checkpoints").iterdir())
    if checkpoint_children != ["best", "latest"]:
        raise RuntimeError(f"only best/latest checkpoints are allowed: {checkpoint_children}")
    allowed_directories = {
        "logs",
        "metrics",
        "tensorboard",
        "checkpoints",
        "checkpoints/best",
        "checkpoints/latest",
        "figures",
    }
    actual_directories = {
        str(path.relative_to(run)) for path in run.rglob("*") if path.is_dir()
    }
    if actual_directories != allowed_directories:
        raise RuntimeError(
            "smoke output directory set differs from the compact contract: "
            f"{sorted(actual_directories)}"
        )
    allowed_files = {
        *required,
        "checkpoints/best/checkpoint_state.pkl",
        "checkpoints/latest/checkpoint_state.pkl",
        str(event_files[0].relative_to(run)),
    }
    actual_files = {
        str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()
    }
    if actual_files != allowed_files:
        extras = sorted(actual_files - allowed_files)
        missing_allowed = sorted(allowed_files - actual_files)
        raise RuntimeError(
            f"smoke output file set differs from contract; extras={extras}, missing={missing_allowed}"
        )
    size_bytes = _directory_size(run)
    if size_bytes > max_size_mb * 1024 * 1024:
        raise RuntimeError(
            f"smoke run is unexpectedly large: {size_bytes / (1024 * 1024):.2f} MiB"
        )
    return {
        "run_dir": str(run),
        "size_bytes": size_bytes,
        "size_mib": size_bytes / (1024 * 1024),
        "tensorboard_event": str(event_files[0]),
    }


def run_smoke(
    output_root: str | Path | None = None,
    *,
    min_free_space_gb_before_start: float = 20.0,
) -> tuple[Path, dict[str, object]]:
    """Create one complete fake experiment without Isaac, images, or a model."""

    config = _resolved_config(output_root)
    run = RunManager.create(
        experiment_name="output_smoke",
        seed=42,
        resolved_config=config,
        output_root=output_root,
        command=[sys.executable, "-m", "uav_agent.experiments.smoke_test", "--output-root", str(output_root or "outputs")],
        model={"base_model": "Qwen3-VL-4B-Instruct", "base_model_path": "models/initial_model/Qwen3-VL-4B-Instruct", "adapter_path": None},
        pipeline={"planner_mode": "fake", "perception_mode": "fake", "control_backend": "fake", "search_policy": "fixed_hexagon", "tracker": "fake", "reacquire_policy": "constant_velocity"},
        min_free_space_gb_before_start=min_free_space_gb_before_start,
    )
    runner = _FakeEpisodeRunner()
    checkpoints: CheckpointManager | None = None
    last_step = 0
    last_update = 0
    latest_payload: dict[str, object] = {"adapter": b"fake-lora-interrupted"}
    summary: dict[str, object] | None = None
    final: dict[str, object] | None = None
    try:
        with TerminalLogger(run.paths.terminal_log) as terminal:
            terminal.emit("RUN", f"run_id={run.run_id} output_dir={run.paths.run_dir}")
            evaluator = Evaluator(
                runner,
                training_seeds=tuple(range(1, 21)),
                validation_seeds=tuple(range(10_000, 10_010)),
                test_seeds=tuple(range(20_000, 20_020)),
                interval_steps=20_000,
                checkpoint_loader=runner.load_checkpoint,
            )
            checkpoints = CheckpointManager(run.paths.checkpoints_dir, latest_interval_steps=5)
            with MetricLogger(run.paths.run_dir, run_id=run.run_id) as metrics:
                runtime = ExperimentRuntime(
                    run,
                    checkpoints,
                    metrics,
                    terminal,
                    min_free_space_gb_during_run=10.0,
                    warning_run_size_gb=3.0,
                    max_run_size_gb=5.0,
                )
                strict_history: list[bool] = []
                for update in range(1, 21):
                    last_step = update * 1_000
                    last_update = update
                    episode = dict(runner.run_episode(seed=update, phase="train", deterministic=True))
                    episode.update(run_id=run.run_id, global_step=update * 1_000, episode_id=update - 1)
                    success = metrics.log_episode(episode)
                    strict_history.append(success)
                    metrics.log_train(
                        update * 1_000,
                        update,
                        {
                            "episodes_completed": update,
                            "episode_return_mean": episode["episode_return"],
                            "episode_length_mean": 250 + update,
                            "mission_success_rate_100": sum(strict_history[-100:]) / len(strict_history[-100:]),
                            "learning_rate": 3.0e-4,
                            "fps": 300.0,
                            "wall_time_s": update * 0.02,
                            "policy_loss": 0.8 / update,
                            "value_loss": 1.2 / update,
                            "entropy": 0.5 / update,
                            "approx_kl": 0.01,
                            "clip_fraction": 0.1,
                        },
                    )
                    if update % 10 == 0:
                        terminal.emit(
                            "TRAIN",
                            f"update={update} step={update * 1000} episodes={update} return={float(episode['episode_return']):.2f} success_100={sum(strict_history[-100:]) / len(strict_history[-100:]):.2f} fps=300",
                        )
                        runtime.periodic_storage_check(
                            global_step=last_step,
                            update=last_update,
                            payload=latest_payload,
                        )

                checkpoints.save_latest(
                    global_step=20_000,
                    update=20,
                    payload={"adapter": b"fake-lora-latest", "optimizer_state": {"learning_rate": 3.0e-4}, "scheduler_state": {"step": 20}, "rng_state": (42,)},
                )
                validation = evaluator.evaluate(global_step=20_000, checkpoint_step=20_000)
                metrics.log_eval(20_000, validation, checkpoint_step=20_000)
                _log_evaluation_episodes(
                    metrics,
                    evaluator.last_validation_episodes,
                    run_id=run.run_id,
                    global_step=20_000,
                )
                checkpoints.maybe_save_best(
                    global_step=20_000,
                    update=20,
                    metrics=validation,
                    payload={"adapter": b"fake-lora-best", "optimizer_state": {"not": "persisted"}},
                )
                terminal.emit(
                    "EVAL",
                    f"step=20000 episodes={validation['num_episodes']} mission_success={float(validation['mission_success_rate']):.2f} false_lock={float(validation['false_lock_rate']):.2f}",
                )
                terminal.emit("CHECKPOINT", f"best={checkpoints.best_path} latest={checkpoints.latest_path}")

                final = evaluator.run_final_test(checkpoints.best_path, run_id=run.run_id)
                if runner.loaded_checkpoint != checkpoints.best_path:
                    raise RuntimeError("final evaluator did not load the best checkpoint")
                _log_evaluation_episodes(
                    metrics,
                    evaluator.last_test_episodes,
                    run_id=run.run_id,
                    global_step=20_000,
                )
                metrics.log_final(final)
                metrics.flush()

            generated = ExperimentPlotter(run.paths.run_dir).generate_all()
            if len(generated) != len(_FIGURES):
                raise RuntimeError(f"expected {len(_FIGURES)} figures, generated {len(generated)}")
            terminal.emit(
                "FINISHED",
                f"status=COMPLETED best_checkpoint={checkpoints.best_path} final_test_success={float(final['mission_success_rate']):.2f}",
            )
            terminal.emit("SMOKE", f"run_dir={run.paths.run_dir}")
            # Validate every generated artifact while the run is still
            # RUNNING. Completion is the final fallible persistence action.
            summary = verify_smoke_run(run.paths.run_dir)
        assert final is not None
        run.complete(final)
    except KeyboardInterrupt:
        if run.status is RunStatus.RUNNING and checkpoints is not None:
            checkpoints.try_save_latest(
                global_step=last_step,
                update=last_update,
                payload=latest_payload,
            )
        if run.status is RunStatus.RUNNING:
            run.interrupt()
        raise
    except BaseException:
        if run.status is RunStatus.RUNNING and checkpoints is not None:
            checkpoints.try_save_latest(
                global_step=last_step,
                update=last_update,
                payload=latest_payload,
            )
        if run.status is RunStatus.RUNNING:
            run.fail(failure_reason=FailureReason.UNKNOWN_ERROR)
        raise

    assert summary is not None
    return run.paths.run_dir, summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None, help="CLI override for the output root")
    parser.add_argument(
        "--min-free-space-gb",
        type=float,
        default=20.0,
        help="preflight free-space requirement (default: 20 GiB)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    run_smoke(
        arguments.output_root,
        min_free_space_gb_before_start=arguments.min_free_space_gb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
