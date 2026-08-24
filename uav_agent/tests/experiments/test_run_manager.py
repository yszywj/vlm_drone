from __future__ import annotations

import io
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import yaml

from configs.loader import load_config
from experiments.run_manager import (
    FORBIDDEN_ARTIFACT_DIRECTORIES,
    InsufficientDiskSpaceError,
    InvalidRunStateError,
    OutputRootError,
    RunManager,
    RunStorageLimitError,
    SensitiveDataError,
    resolve_output_root,
)
from experiments.schemas import RunStatus, StorageStatus
from experiments.terminal_logger import (
    RUN_RESUMED_SEPARATOR,
    TerminalLogger,
    render_bash_tee_launcher,
    run_command_with_tee,
    run_python_with_tee,
)


FIXED_TIME = datetime(2026, 8, 16, 21, 35, 0, tzinfo=timezone(timedelta(hours=8)))
GIT_METADATA = {
    "commit": "a1b2c3def4567890",
    "branch": "main",
    "dirty": False,
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RunManagerTest(unittest.TestCase):
    def create_run(self, root: Path, **overrides: object) -> RunManager:
        arguments: dict[str, object] = {
            "experiment_name": "oracle_search",
            "seed": 42,
            "resolved_config": {
                "experiment": {"name": "oracle_search", "seed": 42},
                "model": {"base_model": "Qwen3-VL-4B-Instruct"},
            },
            "output_root": root,
            "command": ["python", "-u", "train.py", "--seed", "42"],
            "model": {
                "base_model": "Qwen3-VL-4B-Instruct",
                "base_model_path": "/models/Qwen3-VL-4B-Instruct",
                "adapter_path": None,
            },
            "pipeline": {
                "planner_mode": "qwen_text",
                "perception_mode": "oracle",
                "control_backend": "kinematic",
            },
            "min_free_space_gb_before_start": 0.0,
            "now": FIXED_TIME,
            "git_metadata": GIT_METADATA,
        }
        arguments.update(overrides)
        return RunManager.create(**arguments)  # type: ignore[arg-type]

    def test_output_root_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            explicit = root / "explicit"
            environment = root / "environment"
            project = root / "project"
            environ = {"VLM_DRONE_OUTPUT_ROOT": str(environment)}
            self.assertEqual(
                resolve_output_root(explicit, environ=environ, project_root=project),
                explicit.resolve(),
            )
            self.assertEqual(
                resolve_output_root(None, environ=environ, project_root=project),
                environment.resolve(),
            )
            self.assertEqual(
                resolve_output_root(None, environ={}, project_root=project),
                (project / "outputs").resolve(),
            )

    def test_create_writes_fixed_minimal_layout_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self.create_run(Path(temporary))
            self.assertEqual(
                manager.run_id,
                "20260816-213500_oracle_search_seed42_a1b2c3d",
            )
            expected_directories = {
                "logs",
                "metrics",
                "tensorboard",
                "checkpoints",
                "checkpoints/best",
                "checkpoints/latest",
                "figures",
            }
            actual_directories = {
                str(path.relative_to(manager.paths.run_dir))
                for path in manager.paths.run_dir.rglob("*")
                if path.is_dir()
            }
            self.assertEqual(actual_directories, expected_directories)
            self.assertTrue(manager.paths.terminal_log.is_file())
            self.assertTrue(manager.paths.exit_code.is_file())
            self.assertFalse(manager.paths.exit_code.read_text(encoding="utf-8"))
            for forbidden in FORBIDDEN_ARTIFACT_DIRECTORIES:
                self.assertFalse((manager.paths.run_dir / forbidden).exists())
            self.assertFalse((manager.paths.run_dir / "artifacts").exists())

            manifest = yaml.safe_load(manager.paths.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(manifest["git"], GIT_METADATA)
            self.assertEqual(manifest["output_dir"], str(manager.paths.run_dir))
            self.assertIsNone(manifest["end_time"])
            config = yaml.safe_load(manager.paths.resolved_config.read_text(encoding="utf-8"))
            self.assertEqual(config["experiment"]["seed"], 42)
            command = manager.paths.command.read_text(encoding="utf-8")
            self.assertIn("python -u train.py --seed 42", command)
            self.assertTrue(manager.paths.command.stat().st_mode & 0o100)
            self.assertFalse(list(manager.paths.run_dir.glob(".*.tmp")))

    def test_collision_adds_suffix_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.create_run(root)
            marker = first.paths.run_dir / "do_not_overwrite.txt"
            marker.write_text("original", encoding="utf-8")
            second = self.create_run(root)
            self.assertEqual(second.run_id, first.run_id + "_01")
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")

    def test_invalid_experiment_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for invalid in ("", "../escape", "has space", "中文"):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        self.create_run(Path(temporary), experiment_name=invalid)

    def test_negative_seed_and_multiline_command_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                self.create_run(root, seed=-1)
            with self.assertRaisesRegex(ValueError, "line breaks"):
                self.create_run(root, command="python train.py\necho injected")
            self.assertFalse((root / "runs").exists())

    def test_nested_sensitive_fields_and_commands_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for secret_key in (
                "openai_api_key",
                "token",
                "hf_token",
                "authorization",
            ):
                with self.subTest(secret_key=secret_key):
                    with self.assertRaises(SensitiveDataError):
                        self.create_run(
                            root,
                            resolved_config={"service": {secret_key: "secret-value"}},
                        )
            for command in (
                ["python", "train.py", "--api-key", "secret-value"],
                ["python", "train.py", "--token=secret-value"],
                ["env", "HF_TOKEN=secret-value", "python", "train.py"],
            ):
                with self.subTest(command=command):
                    with self.assertRaises(SensitiveDataError):
                        self.create_run(root, command=command)
            self.assertFalse((root / "runs").exists())

    def test_recursive_or_excessively_deep_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recursive: dict[str, object] = {}
            recursive["self"] = recursive
            with self.assertRaisesRegex(ValueError, "recursive"):
                self.create_run(root, resolved_config=recursive)

            deep: dict[str, object] = {}
            cursor = deep
            for _ in range(70):
                child: dict[str, object] = {}
                cursor["child"] = child
                cursor = child
            with self.assertRaisesRegex(ValueError, "nesting depth"):
                self.create_run(root, resolved_config=deep)
            self.assertFalse((root / "runs").exists())

    def test_metric_names_containing_token_are_not_false_positive_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self.create_run(
                Path(temporary),
                resolved_config={"metrics": {"token_accuracy": 0.75}},
            )
            self.assertTrue(manager.paths.resolved_config.is_file())

    def test_update_resolved_config_is_validated_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self.create_run(Path(temporary))
            manager.update_resolved_config(
                {
                    "mission": {"interpreter": "llm", "goal_count": 3},
                    "planner": {"mode": "dynamic_llm"},
                }
            )
            updated = yaml.safe_load(
                manager.paths.resolved_config.read_text(encoding="utf-8")
            )
            self.assertEqual(updated["mission"]["goal_count"], 3)
            self.assertEqual(updated["planner"]["mode"], "dynamic_llm")
            before = manager.paths.resolved_config.read_bytes()

            with self.assertRaises(SensitiveDataError):
                manager.update_resolved_config(
                    {"mission": {"api_key": "must-not-be-persisted"}}
                )
            self.assertEqual(manager.paths.resolved_config.read_bytes(), before)
            self.assertFalse(list(manager.paths.run_dir.rglob("*.tmp")))

            manager.complete()
            with self.assertRaises(InvalidRunStateError):
                manager.update_resolved_config({"mission": {"goal_count": 4}})
            self.assertEqual(manager.paths.resolved_config.read_bytes(), before)

    def test_update_resolved_config_uses_canonical_single_and_multi_inventories(
        self,
    ) -> None:
        config_paths = (
            PROJECT_ROOT / "configs" / "default.yaml",
            PROJECT_ROOT / "configs" / "multi_uav_demo.yaml",
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, config_path in enumerate(config_paths):
                with self.subTest(config_path=config_path.name):
                    manager = self.create_run(
                        Path(temporary) / str(index),
                        experiment_name=f"resolved_config_{index}",
                    )
                    config = load_config(config_path)

                    manager.update_resolved_config(config)

                    persisted = yaml.safe_load(
                        manager.paths.resolved_config.read_text(encoding="utf-8")
                    )
                    self.assertNotIn("uav", persisted)
                    self.assertNotIn("target", persisted)
                    self.assertNotIn("camera", persisted)
                    self.assertEqual(
                        [item["id"] for item in persisted["uavs"]],
                        [item.id for item in config.uavs],
                    )
                    self.assertEqual(
                        [item["id"] for item in persisted["targets"]],
                        [item.id for item in config.targets],
                    )
                    self.assertEqual(
                        set(persisted["camera_profiles"]),
                        set(config.camera_profiles),
                    )
                    reloaded = load_config(manager.paths.resolved_config)
                    self.assertEqual(
                        [item.id for item in reloaded.uavs],
                        [item.id for item in config.uavs],
                    )
                    self.assertEqual(
                        [item.id for item in reloaded.targets],
                        [item.id for item in config.targets],
                    )

    def test_lifecycle_updates_manifest_and_exit_code_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self.create_run(Path(temporary))
            manager.complete(
                {"mission_success_rate": 0.8},
                now=FIXED_TIME + timedelta(hours=1),
            )
            self.assertIs(manager.status, RunStatus.COMPLETED)
            self.assertEqual(manager.paths.exit_code.read_text(encoding="utf-8"), "0\n")
            manifest = yaml.safe_load(manager.paths.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["exit_code"], 0)
            self.assertEqual(manifest["final_metrics"]["mission_success_rate"], 0.8)
            self.assertIsNotNone(manifest["end_time"])
            with self.assertRaises(InvalidRunStateError):
                manager.fail()
            self.assertFalse(list(manager.paths.run_dir.rglob("*.tmp")))

    def test_failed_and_interrupted_runs_require_nonzero_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            failed = self.create_run(Path(temporary), experiment_name="failed")
            with self.assertRaises(ValueError):
                failed.fail(exit_code=0)
            failed.fail(exit_code=2, failure_reason="PROCESS_CRASH")
            self.assertEqual(failed.paths.exit_code.read_text(encoding="utf-8"), "2\n")

            interrupted = self.create_run(Path(temporary), experiment_name="interrupted")
            interrupted.interrupt()
            self.assertIs(interrupted.status, RunStatus.INTERRUPTED)
            self.assertEqual(interrupted.paths.exit_code.read_text(encoding="utf-8"), "130\n")

    def test_resume_reuses_run_id_and_appends_separator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self.create_run(Path(temporary))
            manager.paths.terminal_log.write_text("old output\n", encoding="utf-8")
            manager.interrupt()
            resumed = RunManager.resume(
                manager.paths.run_dir,
                min_free_space_gb_before_start=0.0,
                now=FIXED_TIME + timedelta(hours=2),
            )
            self.assertEqual(resumed.run_id, manager.run_id)
            self.assertIs(resumed.status, RunStatus.RUNNING)
            log = resumed.paths.terminal_log.read_text(encoding="utf-8")
            self.assertIn("old output", log)
            self.assertIn(RUN_RESUMED_SEPARATOR, log)
            self.assertEqual(resumed.manifest["resume_count"], 1)
            self.assertIsNone(resumed.manifest["end_time"])
            self.assertEqual(resumed.paths.exit_code.read_text(encoding="utf-8"), "")
            code = run_command_with_tee(
                [sys.executable, "-u", "-c", "raise SystemExit(7)"],
                terminal_log=resumed.paths.terminal_log,
                exit_code_path=resumed.paths.exit_code,
                console=io.StringIO(),
            )
            self.assertEqual(code, 7)
            self.assertEqual(resumed.paths.exit_code.read_text(encoding="utf-8"), "7\n")

    def test_completed_run_cannot_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self.create_run(Path(temporary))
            manager.complete()
            with self.assertRaises(InvalidRunStateError):
                RunManager.resume(
                    manager.paths.run_dir,
                    min_free_space_gb_before_start=0.0,
                )

    def test_unwritable_root_fails_before_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            with mock.patch("experiments.run_manager.os.access", return_value=False):
                with self.assertRaises(OutputRootError):
                    self.create_run(root)
            self.assertFalse((root / "runs").exists())

    def test_insufficient_disk_fails_before_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            disk_usage = mock.Mock(free=1024)
            with mock.patch("experiments.run_manager.shutil.disk_usage", return_value=disk_usage):
                with self.assertRaises(InsufficientDiskSpaceError):
                    self.create_run(root, min_free_space_gb_before_start=1.0)
            self.assertFalse((root / "runs").exists())

    def test_periodic_storage_check_returns_warning_and_stop_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self.create_run(Path(temporary))
            one_gib = 1024**3
            with mock.patch(
                "experiments.run_manager.shutil.disk_usage",
                return_value=mock.Mock(free=15 * one_gib),
            ), mock.patch(
                "experiments.run_manager._directory_size_bytes",
                return_value=4 * one_gib,
            ):
                warning = manager.check_storage(
                    min_free_space_gb_during_run=10,
                    warning_run_size_gb=3,
                    max_run_size_gb=5,
                )
            self.assertIs(warning.status, StorageStatus.WARNING)
            self.assertFalse(warning.should_stop)

            with mock.patch(
                "experiments.run_manager.shutil.disk_usage",
                return_value=mock.Mock(free=9 * one_gib),
            ), mock.patch(
                "experiments.run_manager._directory_size_bytes",
                return_value=6 * one_gib,
            ):
                stopped = manager.check_storage(
                    min_free_space_gb_during_run=10,
                    warning_run_size_gb=3,
                    max_run_size_gb=5,
                )
                with self.assertRaises(RunStorageLimitError) as raised:
                    manager.check_storage(
                        min_free_space_gb_during_run=10,
                        warning_run_size_gb=3,
                        max_run_size_gb=5,
                        raise_on_stop=True,
                    )
            self.assertIs(stopped.status, StorageStatus.STOP_REQUIRED)
            self.assertTrue(stopped.should_stop)
            self.assertIs(raised.exception.check.status, StorageStatus.STOP_REQUIRED)


class TerminalLoggerTest(unittest.TestCase):
    def test_context_tees_stdout_stderr_and_exception_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "logs" / "terminal.log"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with TerminalLogger(log_path, stdout=stdout, stderr=stderr):
                    print("standard output")
                    print("standard error", file=sys.stderr)
                    raise RuntimeError("boom")
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("standard output", log)
            self.assertIn("standard error", log)
            self.assertIn("Traceback", log)
            self.assertIn("RuntimeError: boom", log)
            self.assertIn("standard output", stdout.getvalue())
            self.assertIn("standard error", stderr.getvalue())

    def test_unbuffered_child_tee_preserves_real_exit_code_and_resume_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "logs" / "terminal.log"
            exit_code_path = root / "exit_code.txt"
            console = io.StringIO()
            code = run_python_with_tee(
                "-c",
                (
                    "import sys; print('child out'); "
                    "print('child err', file=sys.stderr); sys.exit(7)",
                ),
                terminal_log=log_path,
                exit_code_path=exit_code_path,
                module=False,
                python_executable=sys.executable,
                console=console,
            )
            self.assertEqual(code, 7)

            code = run_command_with_tee(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys; print('child out'); print('child err', file=sys.stderr); sys.exit(7)",
                ],
                terminal_log=log_path,
                exit_code_path=exit_code_path,
                resume=True,
                console=console,
            )
            self.assertEqual(code, 7)
            self.assertEqual(exit_code_path.read_text(encoding="utf-8"), "7\n")
            log = log_path.read_text(encoding="utf-8")
            self.assertIn(RUN_RESUMED_SEPARATOR, log)
            self.assertIn("child out", log)
            self.assertIn("child err", log)
            self.assertIn("child out", console.getvalue())

    def test_bash_launcher_uses_pipefail_python_u_and_python_exit_code(self) -> None:
        launcher = render_bash_tee_launcher(["train.py", "--seed", "42"])
        self.assertIn("set -o pipefail", launcher)
        self.assertIn("python -u train.py --seed 42", launcher)
        self.assertIn("2>&1 | tee -a", launcher)
        self.assertIn("EXIT_CODE=${PIPESTATUS[0]}", launcher)
        self.assertIn("exit \"${EXIT_CODE}\"", launcher)


if __name__ == "__main__":
    unittest.main()
