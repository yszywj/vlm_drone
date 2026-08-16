from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.runtime import (
    ExperimentInterrupted,
    ExperimentRuntime,
    InterruptState,
    STORAGE_FAILURE_REASON,
    SignalHandlerInstallationError,
    StorageShutdownRequired,
    installed_interrupt_handlers,
)
from experiments.run_manager import StorageCheck
from experiments.schemas import FailureReason, StorageStatus


class _FakeRun:
    def __init__(self, check: StorageCheck, calls: list[object], *, fail_transition: bool = False) -> None:
        self.check = check
        self.calls = calls
        self.fail_transition = fail_transition
        self.check_kwargs: dict[str, object] | None = None

    def check_storage(self, **kwargs: object) -> StorageCheck:
        self.check_kwargs = dict(kwargs)
        self.calls.append("check")
        return self.check

    def fail(self, **kwargs: object) -> None:
        self.calls.append(("fail", kwargs))
        if self.fail_transition:
            raise RuntimeError("manifest unavailable")

    def interrupt(self, **kwargs: object) -> None:
        self.calls.append(("interrupt", kwargs))
        if self.fail_transition:
            raise RuntimeError("manifest unavailable")


class _FakeCheckpoints:
    def __init__(self, calls: list[object], *, saved: bool = True, raises: bool = False) -> None:
        self.calls = calls
        self.saved = saved
        self.raises = raises

    def try_save_latest(self, **kwargs: object) -> bool:
        self.calls.append(("latest", kwargs))
        if self.raises:
            raise RuntimeError("checkpoint unavailable")
        return self.saved


class _FakeMetrics:
    def __init__(self, calls: list[object], *, raises: bool = False) -> None:
        self.calls = calls
        self.raises = raises

    def flush(self) -> None:
        self.calls.append("flush")
        if self.raises:
            raise RuntimeError("flush unavailable")


class _FakeTerminal:
    def __init__(self, calls: list[object], *, raises: bool = False) -> None:
        self.calls = calls
        self.raises = raises

    def emit(self, section: str, message: str) -> None:
        self.calls.append(("terminal", section, message))
        if self.raises:
            raise RuntimeError("terminal unavailable")


def _check(status: StorageStatus) -> StorageCheck:
    reasons = () if status is StorageStatus.OK else ("disk threshold reached\ncompactly",)
    return StorageCheck(
        status=status,
        free_space_gb=9.25,
        run_size_gb=3.5,
        reasons=reasons,
    )


class ExperimentRuntimeTest(unittest.TestCase):
    def _runtime(
        self,
        status: StorageStatus,
        calls: list[object],
        *,
        checkpoint_raises: bool = False,
        flush_raises: bool = False,
        transition_raises: bool = False,
        terminal_raises: bool = False,
    ) -> ExperimentRuntime:
        return ExperimentRuntime(
            _FakeRun(_check(status), calls, fail_transition=transition_raises),
            _FakeCheckpoints(calls, raises=checkpoint_raises),
            _FakeMetrics(calls, raises=flush_raises),
            _FakeTerminal(calls, raises=terminal_raises),
            min_free_space_gb_during_run=10.0,
            warning_run_size_gb=3.0,
            max_run_size_gb=5.0,
        )

    def test_ok_check_has_no_persistence_side_effects(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(StorageStatus.OK, calls)
        result = runtime.check_storage(global_step=100, update=4, payload={"adapter": b"x"})
        self.assertIs(result.status, StorageStatus.OK)
        self.assertEqual(calls, ["check"])
        self.assertEqual(runtime.run_manager.check_kwargs["raise_on_stop"], False)

    def test_warning_emits_one_compact_terminal_line_only(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(StorageStatus.WARNING, calls)
        result = runtime.periodic_storage_check(global_step=100)
        self.assertIs(result.status, StorageStatus.WARNING)
        self.assertEqual(calls[0], "check")
        _, section, message = calls[1]
        self.assertEqual(section, "STORAGE")
        self.assertIn("WARNING free_gb=9.25 run_gb=3.50", message)
        self.assertNotIn("\n", message)
        self.assertEqual(len(calls), 2)

    def test_stop_saves_then_flushes_then_fails_manifest_and_raises(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(StorageStatus.STOP_REQUIRED, calls)
        payload = {"adapter": b"latest"}
        with self.assertRaises(StorageShutdownRequired) as captured:
            runtime.check_storage(global_step=200, update=8, payload=payload)

        operational = [
            call[0] if isinstance(call, tuple) else call
            for call in calls
            if call != "check" and not (isinstance(call, tuple) and call[0] == "terminal")
        ]
        self.assertEqual(operational, ["latest", "flush", "fail"])
        latest = next(call for call in calls if isinstance(call, tuple) and call[0] == "latest")
        self.assertEqual(
            latest[1],
            {"global_step": 200, "update": 8, "payload": payload},
        )
        failed = next(call for call in calls if isinstance(call, tuple) and call[0] == "fail")
        self.assertEqual(failed[1]["failure_reason"], STORAGE_FAILURE_REASON)
        self.assertEqual(failed[1]["exit_code"], 75)
        self.assertTrue(captured.exception.report.checkpoint_saved)
        self.assertEqual(captured.exception.report.errors, ())

    def test_stop_still_flushes_and_finalizes_when_latest_raises(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(
            StorageStatus.STOP_REQUIRED,
            calls,
            checkpoint_raises=True,
            flush_raises=True,
            transition_raises=True,
            terminal_raises=True,
        )
        with self.assertRaises(StorageShutdownRequired) as captured:
            runtime.check_storage(global_step=1)
        names = [call[0] if isinstance(call, tuple) else call for call in calls]
        self.assertLess(names.index("latest"), names.index("flush"))
        self.assertLess(names.index("flush"), names.index("fail"))
        self.assertFalse(captured.exception.report.checkpoint_saved)
        self.assertEqual(len(captured.exception.report.errors), 3)

    def test_interrupt_is_explicit_best_effort_and_installs_no_signal_handler(self) -> None:
        calls: list[object] = []
        with patch.object(signal, "signal") as signal_mock:
            runtime = self._runtime(StorageStatus.OK, calls)
            report = runtime.handle_interrupt(
                global_step=33,
                update=7,
                payload={"adapter": "state"},
            )
        signal_mock.assert_not_called()
        self.assertTrue(report.completed_without_errors)
        names = [call[0] if isinstance(call, tuple) else call for call in calls]
        self.assertEqual(names, ["latest", "flush", "interrupt"])
        self.assertEqual(calls[-1][1], {"exit_code": 130})

    def test_failure_is_best_effort_and_preserves_actual_reason(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(
            StorageStatus.OK,
            calls,
            checkpoint_raises=True,
            flush_raises=True,
        )
        report = runtime.handle_failure(
            global_step=44,
            failure_reason=FailureReason.CUDA_OUT_OF_MEMORY,
            exit_code=9,
        )
        names = [call[0] if isinstance(call, tuple) else call for call in calls]
        self.assertEqual(names, ["latest", "flush", "fail"])
        self.assertFalse(report.checkpoint_saved)
        self.assertEqual(len(report.errors), 2)
        self.assertEqual(calls[-1][1]["failure_reason"], FailureReason.CUDA_OUT_OF_MEMORY)
        self.assertEqual(calls[-1][1]["exit_code"], 9)

    def test_runtime_never_deletes_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / "other_users_result.bin"
            sentinel.write_bytes(b"keep")
            calls: list[object] = []
            runtime = self._runtime(StorageStatus.STOP_REQUIRED, calls)
            with self.assertRaises(StorageShutdownRequired):
                runtime.check_storage(global_step=1)
            self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_signal_context_is_opt_in_and_restores_original_handlers(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(StorageStatus.OK, calls)
        original = {
            signal.SIGINT: object(),
            signal.SIGTERM: object(),
        }
        active = dict(original)
        signal_calls: list[tuple[int, object]] = []

        def fake_getsignal(signum: int) -> object:
            return active[signum]

        def fake_signal(signum: int, handler: object) -> object:
            previous = active[signum]
            active[signum] = handler
            signal_calls.append((signum, handler))
            return previous

        state_calls = 0

        def state_provider() -> InterruptState:
            nonlocal state_calls
            state_calls += 1
            # A repeated signal while cleanup is in progress is ignored, so it
            # cannot interrupt latest -> flush -> manifest persistence.
            active[signal.SIGTERM](signal.SIGTERM, None)
            return InterruptState(
                global_step=91,
                update=6,
                payload={"adapter": "latest"},
            )

        with patch.object(signal, "getsignal", side_effect=fake_getsignal), patch.object(
            signal, "signal", side_effect=fake_signal
        ):
            with self.assertRaises(ExperimentInterrupted) as captured:
                with installed_interrupt_handlers(runtime, state_provider):
                    self.assertEqual(calls, [])
                    handler = active[signal.SIGTERM]
                    self.assertTrue(callable(handler))
                    handler(signal.SIGTERM, None)

        self.assertEqual(state_calls, 1)
        self.assertEqual(captured.exception.signum, signal.SIGTERM)
        self.assertEqual(captured.exception.exit_code, 143)
        self.assertEqual(captured.exception.code, 143)
        self.assertTrue(captured.exception.report.completed_without_errors)
        self.assertEqual(active, original)
        self.assertEqual(len(signal_calls), 4)
        self.assertEqual(
            calls,
            [
                (
                    "latest",
                    {
                        "global_step": 91,
                        "update": 6,
                        "payload": {"adapter": "latest"},
                    },
                ),
                "flush",
                ("interrupt", {"exit_code": 143}),
            ],
        )

    def test_signal_context_normal_exit_restores_without_persistence(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(StorageStatus.OK, calls)
        original = {signal.SIGINT: object(), signal.SIGTERM: object()}
        active = dict(original)

        def fake_signal(signum: int, handler: object) -> object:
            previous = active[signum]
            active[signum] = handler
            return previous

        with patch.object(signal, "getsignal", side_effect=active.__getitem__), patch.object(
            signal, "signal", side_effect=fake_signal
        ):
            with runtime.interrupt_handlers(lambda: InterruptState(global_step=1)):
                pass

        self.assertEqual(active, original)
        self.assertEqual(calls, [])

    def test_explicit_keyboard_interrupt_uses_same_safe_shutdown(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(StorageStatus.OK, calls)
        original = {signal.SIGINT: object()}
        active = dict(original)

        def fake_signal(signum: int, handler: object) -> object:
            previous = active[signum]
            active[signum] = handler
            return previous

        with patch.object(signal, "getsignal", side_effect=active.__getitem__), patch.object(
            signal, "signal", side_effect=fake_signal
        ):
            with self.assertRaises(ExperimentInterrupted) as captured:
                with runtime.interrupt_handlers(
                    lambda: InterruptState(global_step=7),
                    signals=(signal.SIGINT,),
                ):
                    raise KeyboardInterrupt

        self.assertEqual(captured.exception.signum, signal.SIGINT)
        self.assertEqual(captured.exception.code, 130)
        self.assertEqual(active, original)
        self.assertEqual(
            [item[0] if isinstance(item, tuple) else item for item in calls],
            ["latest", "flush", "interrupt"],
        )

    def test_nested_signal_context_is_rejected_and_outer_remains_active(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(StorageStatus.OK, calls)
        original = {signal.SIGINT: object()}
        active = dict(original)

        def fake_signal(signum: int, handler: object) -> object:
            previous = active[signum]
            active[signum] = handler
            return previous

        with patch.object(signal, "getsignal", side_effect=active.__getitem__), patch.object(
            signal, "signal", side_effect=fake_signal
        ):
            with runtime.interrupt_handlers(
                lambda: InterruptState(global_step=1), signals=(signal.SIGINT,)
            ):
                outer_handler = active[signal.SIGINT]
                with self.assertRaisesRegex(
                    SignalHandlerInstallationError, "nested"
                ):
                    with runtime.interrupt_handlers(
                        lambda: InterruptState(global_step=2),
                        signals=(signal.SIGINT,),
                    ):
                        pass
                self.assertIs(active[signal.SIGINT], outer_handler)

        self.assertEqual(active, original)
        self.assertEqual(calls, [])

    def test_signal_context_rejects_non_main_thread_before_installing(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(StorageStatus.OK, calls)
        errors: list[BaseException] = []

        def enter_context() -> None:
            try:
                with runtime.interrupt_handlers(
                    lambda: InterruptState(global_step=1)
                ):
                    pass
            except BaseException as exc:
                errors.append(exc)

        with patch.object(signal, "signal") as signal_mock:
            worker = threading.Thread(target=enter_context)
            worker.start()
            worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        signal_mock.assert_not_called()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], SignalHandlerInstallationError)
        self.assertIn("main thread", str(errors[0]))

    def test_state_provider_failure_still_flushes_and_finalizes(self) -> None:
        calls: list[object] = []
        runtime = self._runtime(StorageStatus.OK, calls)
        original = {signal.SIGTERM: object()}
        active = dict(original)

        def fake_signal(signum: int, handler: object) -> object:
            previous = active[signum]
            active[signum] = handler
            return previous

        def broken_state_provider() -> InterruptState:
            raise RuntimeError("state snapshot unavailable")

        with patch.object(signal, "getsignal", side_effect=active.__getitem__), patch.object(
            signal, "signal", side_effect=fake_signal
        ):
            with self.assertRaises(ExperimentInterrupted) as captured:
                with runtime.interrupt_handlers(
                    broken_state_provider,
                    signals=(signal.SIGTERM,),
                ):
                    active[signal.SIGTERM](signal.SIGTERM, None)

        self.assertEqual(active, original)
        self.assertEqual(captured.exception.code, 143)
        self.assertIn("state_provider: RuntimeError", captured.exception.report.errors)
        self.assertEqual(
            [item[0] if isinstance(item, tuple) else item for item in calls],
            ["latest", "flush", "interrupt"],
        )
        self.assertEqual(calls[0][1]["global_step"], 0)

    def test_unhandled_sigterm_translation_exits_subprocess_with_143(self) -> None:
        script = textwrap.dedent(
            """
            import signal
            from experiments.runtime import ExperimentRuntime, InterruptState

            class Run:
                def interrupt(self, **kwargs):
                    assert kwargs["exit_code"] == 143

            class Checkpoints:
                def try_save_latest(self, **kwargs):
                    return True

            class Metrics:
                def flush(self):
                    pass

            runtime = ExperimentRuntime(Run(), Checkpoints(), Metrics())
            with runtime.interrupt_handlers(
                lambda: InterruptState(global_step=12),
                signals=(signal.SIGTERM,),
            ):
                installed = signal.getsignal(signal.SIGTERM)
                installed(signal.SIGTERM, None)
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        self.assertEqual(completed.returncode, 143, completed.stderr)


if __name__ == "__main__":
    unittest.main()
