"""Small terminal tee utilities with append-safe resume behaviour."""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Mapping, Sequence, TextIO


RUN_RESUMED_SEPARATOR = "================ RUN RESUMED ================"


def _atomic_write_exit_code(path: Path, exit_code: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(f"{exit_code}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def append_resume_separator(path: str | Path, timestamp: datetime | None = None) -> None:
    """Append a conspicuous boundary without truncating the previous log."""

    log_path = Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n{RUN_RESUMED_SEPARATOR}\n")
        if timestamp is not None:
            stream.write(f"resume_time={timestamp.isoformat()}\n")
        stream.flush()


class _TeeTextIO(io.TextIOBase):
    def __init__(self, console: TextIO, log: TextIO, lock: threading.RLock) -> None:
        self._console = console
        self._log = log
        self._lock = lock

    @property
    def encoding(self) -> str | None:
        return getattr(self._console, "encoding", "utf-8")

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return bool(getattr(self._console, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._console.fileno()

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("terminal output must be text")
        with self._lock:
            self._console.write(text)
            self._log.write(text)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._console.flush()
            self._log.flush()


class TerminalLogger:
    """Tee current-process stdout and stderr to one append-only log.

    For a child training process, prefer :func:`run_command_with_tee`; it
    mirrors ``set -o pipefail; python -u ... 2>&1 | tee`` while returning and
    persisting the Python process's real exit code.
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        resume: bool = False,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self.log_path = Path(log_path).expanduser()
        self.resume = resume
        self._requested_stdout = stdout
        self._requested_stderr = stderr
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None
        self._log_stream: TextIO | None = None
        self._lock = threading.RLock()
        self._active = False

    def __enter__(self) -> "TerminalLogger":
        if self._active:
            raise RuntimeError("TerminalLogger is already active")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume:
            append_resume_separator(self.log_path)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        console_stdout = self._requested_stdout or self._original_stdout
        console_stderr = self._requested_stderr or self._original_stderr
        self._log_stream = self.log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeTextIO(console_stdout, self._log_stream, self._lock)
        sys.stderr = _TeeTextIO(console_stderr, self._log_stream, self._lock)
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> bool:
        if not self._active:
            return False
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            # If the caller does not print an exception, the full traceback is
            # still retained in terminal.log. It is not printed a second time
            # to the terminal here; normal exception propagation handles that.
            if exc_type is not None and self._log_stream is not None:
                traceback.print_exception(
                    exc_type,
                    exc_value,
                    exc_traceback,
                    file=self._log_stream,
                )
                self._log_stream.flush()
        finally:
            if self._original_stdout is not None:
                sys.stdout = self._original_stdout
            if self._original_stderr is not None:
                sys.stderr = self._original_stderr
            if self._log_stream is not None:
                self._log_stream.close()
            self._active = False
        return False

    def emit(self, section: str, message: str) -> None:
        """Print one compact, stable terminal record."""

        if not section or any(character in section for character in "\r\n[]"):
            raise ValueError("section must be a non-empty single-line label")
        if any(character in message for character in "\r\n"):
            raise ValueError("message must fit on one line")
        print(f"[{section}] {message}", flush=True)


def run_command_with_tee(
    command: Sequence[str],
    *,
    terminal_log: str | Path,
    exit_code_path: str | Path,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    resume: bool = False,
    console: TextIO | None = None,
) -> int:
    """Run a command, tee merged stdout/stderr, and preserve its exit code."""

    arguments = list(command)
    if not arguments or any(not isinstance(argument, str) for argument in arguments):
        raise TypeError("command must be a non-empty sequence of strings")
    log_path = Path(terminal_log).expanduser()
    code_path = Path(exit_code_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        append_resume_separator(log_path)
    output_console = sys.stdout if console is None else console
    with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
        process = subprocess.Popen(
            arguments,
            cwd=None if cwd is None else str(cwd),
            env=None if env is None else dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for chunk in iter(process.stdout.readline, ""):
            output_console.write(chunk)
            output_console.flush()
            log_stream.write(chunk)
            log_stream.flush()
        process.stdout.close()
        exit_code = process.wait()
    _atomic_write_exit_code(code_path, exit_code)
    return exit_code


def run_python_with_tee(
    entrypoint: str,
    arguments: Sequence[str] = (),
    *,
    terminal_log: str | Path,
    exit_code_path: str | Path,
    module: bool = False,
    python_executable: str | Path = sys.executable,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    resume: bool = False,
    console: TextIO | None = None,
) -> int:
    """Run Python unbuffered so terminal output is visible and logged live."""

    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("entrypoint must be a non-empty string")
    command = [str(python_executable), "-u"]
    if module:
        command.extend(["-m", entrypoint])
    else:
        command.append(entrypoint)
    command.extend(arguments)
    return run_command_with_tee(
        command,
        terminal_log=terminal_log,
        exit_code_path=exit_code_path,
        cwd=cwd,
        env=env,
        resume=resume,
        console=console,
    )


def render_bash_tee_launcher(
    python_arguments: Sequence[str],
    *,
    run_dir_expression: str = '"${RUN_DIR}"',
) -> str:
    """Render the equivalent Bash launcher for external schedulers.

    ``run_dir_expression`` is intentionally an explicit shell expression; no
    path is hard-coded by this module.
    """

    arguments = list(python_arguments)
    if not arguments or any(not isinstance(argument, str) for argument in arguments):
        raise TypeError("python_arguments must be a non-empty sequence of strings")
    quoted = " ".join(shlex.quote(argument) for argument in arguments)
    return (
        "#!/usr/bin/env bash\n"
        "set -o pipefail\n\n"
        f"python -u {quoted} 2>&1 | tee -a {run_dir_expression}/logs/terminal.log\n"
        "EXIT_CODE=${PIPESTATUS[0]}\n"
        f"echo \"${{EXIT_CODE}}\" > {run_dir_expression}/exit_code.txt\n"
        "exit \"${EXIT_CODE}\"\n"
    )


__all__ = [
    "RUN_RESUMED_SEPARATOR",
    "TerminalLogger",
    "append_resume_separator",
    "render_bash_tee_launcher",
    "run_command_with_tee",
    "run_python_with_tee",
]
