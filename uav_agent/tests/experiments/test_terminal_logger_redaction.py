from __future__ import annotations

import io
import sys
from pathlib import Path

from experiments.terminal_logger import (
    TERMINAL_LOG_TRUNCATION_MARKER,
    TerminalLogger,
)


def test_persistent_log_redacts_sensitive_values_split_across_writes(tmp_path: Path) -> None:
    log_path = tmp_path / "terminal.log"
    console = io.StringIO()
    secret = "sk-this-must-never-reach-disk"
    media = "A" * 240
    observation = "private-pixel-observation-123"

    with TerminalLogger(log_path, stdout=console, stderr=console):
        sys.stdout.write("Authorization: Bea")
        sys.stdout.write(f"rer {secret}\n")
        sys.stdout.write("--api-")
        sys.stdout.write(f"key {secret}\n")
        sys.stdout.write("frame=data:image/png;base")
        sys.stdout.write(f"64,{media}\n")
        sys.stdout.write("Observation: private-pixel-")
        sys.stdout.write("observation-123")

    persisted = log_path.read_text(encoding="utf-8")
    displayed = console.getvalue()

    assert secret not in persisted
    assert media not in persisted
    assert observation not in persisted
    assert "[REDACTED]" in persisted
    assert "[REDACTED MEDIA DATA URI]" in persisted
    assert f"Bearer {secret}" in displayed
    assert f"--api-key {secret}" in displayed
    assert f"data:image/png;base64,{media}" in displayed
    assert observation in displayed


def test_max_log_bytes_adds_marker_without_limiting_console(tmp_path: Path) -> None:
    log_path = tmp_path / "terminal.log"
    console = io.StringIO()
    first = "one two three four five " * 20
    second = "still-visible-after-file-cap\n"
    max_bytes = 192

    with TerminalLogger(
        log_path,
        stdout=console,
        stderr=console,
        max_log_bytes=max_bytes,
    ):
        sys.stdout.write(first + "\n")
        sys.stderr.write(second)

    persisted_bytes = log_path.read_bytes()
    assert len(persisted_bytes) <= max_bytes
    assert TERMINAL_LOG_TRUNCATION_MARKER.encode("utf-8") in persisted_bytes
    assert console.getvalue() == first + "\n" + second
    assert second not in persisted_bytes.decode("utf-8")


def test_long_base64_like_blob_is_not_persisted(tmp_path: Path) -> None:
    log_path = tmp_path / "terminal.log"
    blob = "QWxhZGRpbjpvcGVuIHNlc2FtZQ==" * 8

    with TerminalLogger(log_path, stdout=io.StringIO(), stderr=io.StringIO()):
        print(f"encoded={blob}")

    persisted = log_path.read_text(encoding="utf-8")
    assert blob not in persisted
    assert "[REDACTED LONG BLOB]" in persisted


def test_oversized_sensitive_line_drops_all_continuation_chunks(tmp_path: Path) -> None:
    log_path = tmp_path / "terminal.log"
    console = io.StringIO()
    # Dot-separated short components deliberately avoid looking like one
    # generic base64 token. The Authorization context must protect the entire
    # logical line even after the bounded pending buffer is drained.
    huge_secret = "part." * 4000

    with TerminalLogger(log_path, stdout=console, stderr=console):
        sys.stdout.write("Authorization: Bearer ")
        sys.stdout.write(huge_secret[:9000])
        sys.stdout.write(huge_secret[9000:])
        sys.stdout.write("\nordinary diagnostic\n")

    persisted = log_path.read_text(encoding="utf-8")
    assert huge_secret not in persisted
    assert "part.part.part" not in persisted
    assert "ordinary diagnostic" in persisted
    assert huge_secret in console.getvalue()
