"""Small, dependency-injected runtime coordinator for safe output shutdowns.

The coordinator does not install signal handlers during construction, own a
training loop, or delete files.  A caller periodically invokes
:meth:`check_storage` and explicitly calls :meth:`handle_interrupt` /
:meth:`handle_failure` from its own exception policy.  Training entry points
which want process-signal integration may opt in through
:func:`installed_interrupt_handlers`.
"""

from __future__ import annotations

import signal
import threading
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from types import FrameType
from typing import Callable, Iterator, Mapping, Sequence

from .schemas import FailureReason, StorageStatus


STORAGE_FAILURE_REASON = "STORAGE_LIMIT"


@dataclass(frozen=True, slots=True)
class CleanupReport:
    """Best-effort persistence outcome without retaining exception objects."""

    checkpoint_saved: bool
    errors: tuple[str, ...] = ()

    @property
    def completed_without_errors(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class InterruptState:
    """Training state captured outside the low-level signal handler."""

    global_step: int
    update: int | None = None
    payload: Mapping[str, object] | None = None


class ExperimentInterrupted(SystemExit):
    """Raised after an opted-in signal has been persisted safely.

    Subclassing :class:`SystemExit` makes an unhandled SIGINT exit with 130 and
    an unhandled SIGTERM exit with 143, matching the persisted run metadata.
    Entry points may still catch this exception to inspect the cleanup report.
    """

    def __init__(self, signum: int, report: CleanupReport) -> None:
        self.signum = signum
        self.report = report
        try:
            self.signal_name = signal.Signals(signum).name
        except ValueError:
            self.signal_name = f"SIGNAL_{signum}"
        self.exit_code = 128 + signum
        super().__init__(self.exit_code)


class SignalHandlerInstallationError(RuntimeError):
    """Raised when process-global interrupt handlers cannot be installed."""


class _SignalRequest(BaseException):
    """Minimal control-flow exception raised directly by a signal handler."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(signum)


_signal_install_lock = threading.Lock()
_signal_handlers_active = False


class StorageShutdownRequired(RuntimeError):
    """Raised after STOP_REQUIRED persistence and manifest finalization."""

    def __init__(self, check: object, report: CleanupReport) -> None:
        self.check = check
        self.report = report
        reasons = getattr(check, "reasons", ())
        reason_text = "; ".join(str(reason) for reason in reasons) or "storage limit reached"
        if report.errors:
            reason_text += "; cleanup errors: " + "; ".join(report.errors)
        super().__init__(reason_text)


class ExperimentRuntime:
    """Coordinate storage checks, checkpoints, metrics, terminal, and run state.

    Dependencies are intentionally duck typed.  Production objects may be
    :class:`RunManager`, :class:`CheckpointManager`, :class:`MetricLogger`, and
    :class:`TerminalLogger`; unit tests and small training jobs can use fakes.
    """

    def __init__(
        self,
        run_manager: object,
        checkpoint_manager: object,
        metric_logger: object,
        terminal_logger: object | None = None,
        *,
        min_free_space_gb_during_run: float = 10.0,
        warning_run_size_gb: float = 3.0,
        max_run_size_gb: float = 5.0,
        storage_failure_exit_code: int = 75,
    ) -> None:
        if isinstance(storage_failure_exit_code, bool) or not isinstance(
            storage_failure_exit_code, int
        ) or storage_failure_exit_code == 0:
            raise ValueError("storage_failure_exit_code must be a non-zero integer")
        self.run_manager = run_manager
        self.checkpoint_manager = checkpoint_manager
        self.metric_logger = metric_logger
        self.terminal_logger = terminal_logger
        self.min_free_space_gb_during_run = min_free_space_gb_during_run
        self.warning_run_size_gb = warning_run_size_gb
        self.max_run_size_gb = max_run_size_gb
        self.storage_failure_exit_code = storage_failure_exit_code

    def check_storage(
        self,
        *,
        global_step: int,
        update: int | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> object:
        """Perform one periodic check and safely terminate on STOP_REQUIRED."""

        check_method = self._required_callable(self.run_manager, "check_storage")
        check = check_method(
            min_free_space_gb_during_run=self.min_free_space_gb_during_run,
            warning_run_size_gb=self.warning_run_size_gb,
            max_run_size_gb=self.max_run_size_gb,
            raise_on_stop=False,
        )
        status = self._storage_status(check)
        if status is StorageStatus.WARNING:
            self._emit_storage("WARNING", check)
            return check
        if status is not StorageStatus.STOP_REQUIRED:
            return check

        self._emit_storage("STOP_REQUIRED", check)
        # Ordering is deliberate and tested: preserve resumable state first,
        # then flush all CSV/TensorBoard buffers, then finalize the manifest.
        errors: list[str] = []
        checkpoint_saved = self._try_latest(global_step, update, payload, errors)
        self._try_flush(errors)
        self._try_run_transition(
            "fail",
            errors,
            exit_code=self.storage_failure_exit_code,
            failure_reason=STORAGE_FAILURE_REASON,
        )
        raise StorageShutdownRequired(
            check,
            CleanupReport(checkpoint_saved=checkpoint_saved, errors=tuple(errors)),
        )

    # A descriptive alias for training loops that call this every N updates.
    periodic_storage_check = check_storage

    def handle_interrupt(
        self,
        *,
        global_step: int,
        update: int | None = None,
        payload: Mapping[str, object] | None = None,
        exit_code: int = 130,
    ) -> CleanupReport:
        """Best-effort latest + flush, then persist an interrupted run state."""

        errors: list[str] = []
        checkpoint_saved = self._try_latest(global_step, update, payload, errors)
        self._try_flush(errors)
        self._try_run_transition("interrupt", errors, exit_code=exit_code)
        return CleanupReport(checkpoint_saved, tuple(errors))

    def handle_failure(
        self,
        *,
        global_step: int,
        update: int | None = None,
        payload: Mapping[str, object] | None = None,
        failure_reason: str | FailureReason = FailureReason.UNKNOWN_ERROR,
        exit_code: int = 1,
    ) -> CleanupReport:
        """Best-effort latest + flush, then persist a failed run state."""

        errors: list[str] = []
        checkpoint_saved = self._try_latest(global_step, update, payload, errors)
        self._try_flush(errors)
        self._try_run_transition(
            "fail",
            errors,
            exit_code=exit_code,
            failure_reason=failure_reason,
        )
        return CleanupReport(checkpoint_saved, tuple(errors))

    def interrupt_handlers(
        self,
        state_provider: Callable[[], InterruptState],
        *,
        signals: Sequence[int] = (signal.SIGINT, signal.SIGTERM),
    ) -> AbstractContextManager[None]:
        """Return the explicit signal-handling context for this runtime.

        This is a convenience spelling of :func:`installed_interrupt_handlers`.
        Merely constructing a runtime still has no process-global side effects.
        """

        return installed_interrupt_handlers(self, state_provider, signals=signals)

    def _try_latest(
        self,
        global_step: int,
        update: int | None,
        payload: Mapping[str, object] | None,
        errors: list[str],
    ) -> bool:
        try:
            save = self._required_callable(self.checkpoint_manager, "try_save_latest")
            return bool(save(global_step=global_step, update=update, payload=payload))
        except Exception as exc:  # best-effort cleanup must continue to flush/manifest
            errors.append(self._error_text("latest", exc))
            return False

    def _try_flush(self, errors: list[str]) -> None:
        try:
            self._required_callable(self.metric_logger, "flush")()
        except Exception as exc:  # manifest transition must still be attempted
            errors.append(self._error_text("flush", exc))

    def _try_run_transition(self, name: str, errors: list[str], **kwargs: object) -> None:
        try:
            self._required_callable(self.run_manager, name)(**kwargs)
        except Exception as exc:
            errors.append(self._error_text(f"run.{name}", exc))

    def _emit_storage(self, level: str, check: object) -> None:
        if self.terminal_logger is None:
            return
        free = getattr(check, "free_space_gb", None)
        size = getattr(check, "run_size_gb", None)
        reasons = getattr(check, "reasons", ())
        reason = " | ".join(" ".join(str(item).split()) for item in reasons)
        message = f"{level} free_gb={self._compact_number(free)} run_gb={self._compact_number(size)}"
        if reason:
            message += f" reason={reason[:512]}"
        try:
            self._required_callable(self.terminal_logger, "emit")("STORAGE", message)
        except Exception:
            # Diagnostics are never allowed to block checkpoint/flush/shutdown.
            return

    @staticmethod
    def _compact_number(value: object) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{float(value):.2f}"
        return "unknown"

    @staticmethod
    def _storage_status(check: object) -> StorageStatus:
        raw = getattr(check, "status", None)
        if isinstance(raw, StorageStatus):
            return raw
        try:
            return StorageStatus(str(raw))
        except ValueError as exc:
            raise TypeError("storage check has an invalid status") from exc

    @staticmethod
    def _required_callable(owner: object, name: str):
        value = getattr(owner, name, None)
        if not callable(value):
            raise TypeError(f"{type(owner).__name__} must provide callable {name}()")
        return value

    @staticmethod
    def _error_text(stage: str, error: Exception) -> str:
        # Cleanup callbacks may wrap clients whose exception text contains a
        # credential-bearing URL or header. Persist only the failure class.
        return f"{stage}: {type(error).__name__}"


@contextmanager
def installed_interrupt_handlers(
    runtime: ExperimentRuntime,
    state_provider: Callable[[], InterruptState],
    *,
    signals: Sequence[int] = (signal.SIGINT, signal.SIGTERM),
) -> Iterator[None]:
    """Temporarily translate SIGINT/SIGTERM into a persisted interruption.

    Installation is explicit because Python signal handlers are process-global.
    It is therefore restricted to the main thread and nested installations are
    rejected.  The installed handler performs no checkpoint or file I/O: it
    raises a private control-flow exception which this context catches.  Only
    then is ``state_provider`` called and :meth:`ExperimentRuntime.handle_interrupt`
    run in ``latest -> flush -> manifest`` order.

    The original handlers are restored on normal exit and on every exceptional
    exit.  The resulting :class:`ExperimentInterrupted` can be handled by the
    training entry point, or allowed to terminate with the conventional
    ``128 + signal number`` exit code.
    """

    if threading.current_thread() is not threading.main_thread():
        raise SignalHandlerInstallationError(
            "interrupt handlers may only be installed from the main thread"
        )
    if not callable(state_provider):
        raise TypeError("state_provider must be callable")

    normalized_signals = _normalize_signals(signals)
    global _signal_handlers_active
    with _signal_install_lock:
        if _signal_handlers_active:
            raise SignalHandlerInstallationError(
                "nested interrupt-handler installations are not supported"
            )
        _signal_handlers_active = True

    previous: dict[int, object] = {}
    interrupt_requested = False

    def request_interrupt(signum: int, _frame: FrameType | None) -> None:
        # Keep the asynchronous handler intentionally tiny. State capture and
        # all persistence happen in the context's exception path below. Ignore
        # repeated signals while the first request is already being persisted.
        nonlocal interrupt_requested
        if interrupt_requested:
            return
        interrupt_requested = True
        raise _SignalRequest(signum)

    try:
        for signum in normalized_signals:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_interrupt)

        try:
            yield
        except _SignalRequest as request:
            raise _persist_signal_interrupt(runtime, state_provider, request.signum) from None
        except KeyboardInterrupt:
            # Also make an explicitly raised KeyboardInterrupt safe while the
            # opt-in context is active (useful for portable training wrappers).
            raise _persist_signal_interrupt(runtime, state_provider, signal.SIGINT) from None
    finally:
        try:
            _restore_signal_handlers(previous)
        finally:
            with _signal_install_lock:
                _signal_handlers_active = False


def _normalize_signals(signals: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for raw in signals:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError("signals must contain integer signal numbers")
        if raw not in (signal.SIGINT, signal.SIGTERM):
            raise ValueError("only SIGINT and SIGTERM are supported")
        if raw not in normalized:
            normalized.append(raw)
    if not normalized:
        raise ValueError("at least one signal must be installed")
    return tuple(normalized)


def _persist_signal_interrupt(
    runtime: ExperimentRuntime,
    state_provider: Callable[[], InterruptState],
    signum: int,
) -> ExperimentInterrupted:
    try:
        state = state_provider()
        if not isinstance(state, InterruptState):
            raise TypeError("state_provider must return InterruptState")
    except Exception as error:
        # We cannot manufacture a current adapter payload, but flushing metrics
        # and finalizing the manifest is still safer than abandoning cleanup.
        report = runtime.handle_interrupt(
            global_step=0,
            update=None,
            payload=None,
            exit_code=128 + signum,
        )
        report = CleanupReport(
            report.checkpoint_saved,
            (f"state_provider: {type(error).__name__}", *report.errors),
        )
        return ExperimentInterrupted(signum, report)
    report = runtime.handle_interrupt(
        global_step=state.global_step,
        update=state.update,
        payload=state.payload,
        exit_code=128 + signum,
    )
    return ExperimentInterrupted(signum, report)


def _restore_signal_handlers(previous: Mapping[int, object]) -> None:
    for signum, handler in reversed(tuple(previous.items())):
        signal.signal(signum, handler)


# Alternate descriptive spelling kept as the same implementation, not a second
# policy surface.
ExperimentRuntimeCoordinator = ExperimentRuntime


__all__ = [
    "CleanupReport",
    "ExperimentInterrupted",
    "ExperimentRuntime",
    "ExperimentRuntimeCoordinator",
    "InterruptState",
    "STORAGE_FAILURE_REASON",
    "SignalHandlerInstallationError",
    "StorageShutdownRequired",
    "installed_interrupt_handlers",
]
