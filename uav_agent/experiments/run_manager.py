"""Create and maintain one small, reproducible experiment run directory."""

from __future__ import annotations

import copy
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .schemas import RunStatus, StorageStatus


DEFAULT_MIN_FREE_SPACE_GB = 20.0
DEFAULT_MIN_FREE_SPACE_GB_DURING_RUN = 10.0
DEFAULT_WARNING_RUN_SIZE_GB = 3.0
DEFAULT_MAX_RUN_SIZE_GB = 5.0
OUTPUT_ROOT_ENV = "VLM_DRONE_OUTPUT_ROOT"

_EXPERIMENT_NAME_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "auth_token",
        "bearer_token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "private_key",
        "credential",
        "credentials",
        "token",
        "hf_token",
        "authorization",
        "authorization_header",
    }
)
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_access_token",
    "_refresh_token",
    "_auth_token",
    "_bearer_token",
    "_password",
    "_passwd",
    "_secret",
    "_private_key",
    "_credential",
    "_credentials",
    "_token",
)
_COMMAND_SECRET_RE = re.compile(
    r"(?ix)(?:"
    r"--?(?:token|hf[-_]?token|authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|auth[-_]?token|"
    r"bearer[-_]?token|password|passwd|client[-_]?secret|private[-_]?key|"
    r"credential|credentials)(?:\s|=|$)"
    r"|(?:TOKEN|HF_TOKEN|AUTHORIZATION|API_KEY|ACCESS_TOKEN|REFRESH_TOKEN|AUTH_TOKEN|BEARER_TOKEN|PASSWORD|"
    r"PASSWD|CLIENT_SECRET|PRIVATE_KEY|CREDENTIALS?)\s*="
    r")"
)

_REQUIRED_DIRECTORIES = (
    "logs",
    "metrics",
    "tensorboard",
    "checkpoints",
    "checkpoints/best",
    "checkpoints/latest",
    "figures",
)

FORBIDDEN_ARTIFACT_DIRECTORIES = frozenset(
    {
        "videos",
        "images",
        "frames",
        "trajectories",
        "observations",
        "debug_dumps",
        "periodic_checkpoints",
    }
)


class RunManagerError(RuntimeError):
    """Base class for experiment-output failures."""


class OutputRootError(RunManagerError):
    """Raised when the selected output root cannot be used safely."""


class InsufficientDiskSpaceError(OutputRootError):
    """Raised before a run starts when its output partition is too full."""


class SensitiveDataError(RunManagerError):
    """Raised instead of persisting a credential-bearing configuration."""


class InvalidRunStateError(RunManagerError):
    """Raised for an invalid persistent run-state transition."""


class RunStorageLimitError(RunManagerError):
    """Raised when the training layer must flush, save latest, and stop."""

    def __init__(self, check: "StorageCheck") -> None:
        self.check = check
        super().__init__("; ".join(check.reasons))


@dataclass(frozen=True, slots=True)
class StorageCheck:
    """Result of a read-only periodic storage check."""

    status: StorageStatus
    free_space_gb: float
    run_size_gb: float
    reasons: tuple[str, ...] = ()

    @property
    def should_stop(self) -> bool:
        return self.status is StorageStatus.STOP_REQUIRED


@dataclass(frozen=True, slots=True)
class RunPaths:
    """The complete, deliberately small path surface of one run."""

    output_root: Path
    run_dir: Path
    manifest: Path
    resolved_config: Path
    command: Path
    exit_code: Path
    logs_dir: Path
    terminal_log: Path
    metrics_dir: Path
    train_metrics: Path
    eval_metrics: Path
    episode_metrics: Path
    failure_cases: Path
    final_metrics: Path
    tensorboard_dir: Path
    checkpoints_dir: Path
    best_checkpoint_dir: Path
    latest_checkpoint_dir: Path
    figures_dir: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "RunPaths":
        directory = Path(run_dir).expanduser().resolve()
        # <output_root>/runs/<experiment_name>/<run_id>
        if directory.parent.parent.name != "runs" or not _EXPERIMENT_NAME_RE.fullmatch(
            directory.parent.name
        ):
            raise OutputRootError(
                "run_dir must have the form <output_root>/runs/<experiment_name>/<run_id>"
            )
        try:
            output_root = directory.parents[2]
        except IndexError as exc:  # pragma: no cover - a Path always has root parents
            raise OutputRootError(f"invalid run directory: {directory}") from exc
        metrics_dir = directory / "metrics"
        checkpoints_dir = directory / "checkpoints"
        logs_dir = directory / "logs"
        return cls(
            output_root=output_root,
            run_dir=directory,
            manifest=directory / "manifest.yaml",
            resolved_config=directory / "resolved_config.yaml",
            command=directory / "command.sh",
            exit_code=directory / "exit_code.txt",
            logs_dir=logs_dir,
            terminal_log=logs_dir / "terminal.log",
            metrics_dir=metrics_dir,
            train_metrics=metrics_dir / "train_metrics.csv",
            eval_metrics=metrics_dir / "eval_metrics.csv",
            episode_metrics=metrics_dir / "episode_metrics.csv",
            failure_cases=metrics_dir / "failure_cases.csv",
            final_metrics=metrics_dir / "final_metrics.csv",
            tensorboard_dir=directory / "tensorboard",
            checkpoints_dir=checkpoints_dir,
            best_checkpoint_dir=checkpoints_dir / "best",
            latest_checkpoint_dir=checkpoints_dir / "latest",
            figures_dir=directory / "figures",
        )


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the checkout root without embedding a server-specific path."""

    candidate = Path(start).expanduser().resolve() if start is not None else Path(__file__).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    # Source layout fallback: <project>/uav_agent/experiments/run_manager.py.
    return Path(__file__).resolve().parents[2]


def resolve_output_root(
    output_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve CLI > environment > ``<project_root>/outputs`` precedence."""

    environment = os.environ if environ is None else environ
    if output_root is not None and str(output_root).strip():
        selected = Path(output_root)
    elif environment.get(OUTPUT_ROOT_ENV, "").strip():
        selected = Path(environment[OUTPUT_ROOT_ENV])
    else:
        root = Path(project_root).expanduser() if project_root is not None else find_project_root()
        selected = root / "outputs"
    return selected.expanduser().resolve()


def _validate_experiment_name(experiment_name: str) -> str:
    if not isinstance(experiment_name, str) or not _EXPERIMENT_NAME_RE.fullmatch(experiment_name):
        raise ValueError(
            "experiment_name must contain only ASCII letters, digits, underscores, and hyphens"
        )
    return experiment_name


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalized_key(key)
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def _to_plain_data(
    value: Any,
    path: str = "value",
    *,
    _active_ids: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Convert small config values to deterministic YAML-safe primitives."""

    if _depth > 64:
        raise ValueError(f"{path} exceeds the maximum configuration nesting depth")
    active_ids = set() if _active_ids is None else _active_ids
    is_container = (
        (is_dataclass(value) and not isinstance(value, type))
        or isinstance(value, Mapping)
        or isinstance(value, (list, tuple))
    )
    container_id = id(value)
    if is_container:
        if container_id in active_ids:
            raise ValueError(f"{path} contains a recursive configuration reference")
        active_ids.add(container_id)
    try:
        return _to_plain_data_inner(value, path, active_ids, _depth)
    finally:
        if is_container:
            active_ids.remove(container_id)


def _to_plain_data_inner(
    value: Any,
    path: str,
    active_ids: set[int],
    depth: int,
) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for field in fields(value):
            if _is_sensitive_key(field.name):
                raise SensitiveDataError(f"refusing to persist sensitive field: {path}.{field.name}")
            result[field.name] = _to_plain_data(
                getattr(value, field.name),
                f"{path}.{field.name}",
                _active_ids=active_ids,
                _depth=depth + 1,
            )
        return result
    if isinstance(value, Mapping):
        result = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            if _is_sensitive_key(raw_key):
                raise SensitiveDataError(f"refusing to persist sensitive field: {path}.{raw_key}")
            result[raw_key] = _to_plain_data(
                item,
                f"{path}.{raw_key}",
                _active_ids=active_ids,
                _depth=depth + 1,
            )
        return result
    if isinstance(value, Enum):
        return _to_plain_data(
            value.value,
            path,
            _active_ids=active_ids,
            _depth=depth + 1,
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [
            _to_plain_data(
                item,
                f"{path}[{index}]",
                _active_ids=active_ids,
                _depth=depth + 1,
            )
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and "-----BEGIN PRIVATE KEY-----" in value:
            raise SensitiveDataError(f"refusing to persist private key material at {path}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    raise TypeError(f"{path} contains unsupported value type: {type(value).__name__}")


def _validate_command(command: str) -> None:
    if not command.strip():
        raise ValueError("command must not be empty")
    if "\x00" in command:
        raise ValueError("command must not contain NUL bytes")
    if "\n" in command or "\r" in command:
        raise ValueError("command must be a single shell command without line breaks")
    if _COMMAND_SECRET_RE.search(command):
        raise SensitiveDataError("refusing to save a command containing a credential option")


def _render_command(command: str | Sequence[str] | None) -> str:
    if command is None:
        rendered = shlex.join([sys.executable, "-u", *sys.argv])
    elif isinstance(command, str):
        rendered = command
    else:
        arguments = list(command)
        if not arguments or any(not isinstance(item, str) for item in arguments):
            raise TypeError("command must be a non-empty string or sequence of strings")
        rendered = shlex.join(arguments)
    _validate_command(rendered)
    return rendered


def _atomic_write_text(path: Path, text: str, *, mode: int = 0o640) -> None:
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
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    serialized = yaml.safe_dump(
        dict(data),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    _atomic_write_text(path, serialized)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OutputRootError(f"missing run metadata: {path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise OutputRootError(f"could not read run metadata {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise OutputRootError(f"run metadata must be a mapping: {path}")
    return dict(_to_plain_data(value, path.name))


def _probe_output_root(output_root: Path, min_free_space_gb: float) -> None:
    if isinstance(min_free_space_gb, bool) or not isinstance(min_free_space_gb, (int, float)):
        raise ValueError("min_free_space_gb must be a non-negative finite number")
    threshold = float(min_free_space_gb)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("min_free_space_gb must be a non-negative finite number")
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputRootError(f"could not create output root {output_root}: {exc}") from exc
    if not output_root.is_dir():
        raise OutputRootError(f"output root is not a directory: {output_root}")
    if not os.access(output_root, os.W_OK | os.X_OK):
        raise OutputRootError(f"output root is not writable: {output_root}")
    try:
        with tempfile.NamedTemporaryFile(dir=output_root, prefix=".write-probe-", delete=True):
            pass
    except OSError as exc:
        raise OutputRootError(f"output root is not writable: {output_root}: {exc}") from exc
    try:
        free_bytes = shutil.disk_usage(output_root).free
    except OSError as exc:
        raise OutputRootError(f"could not inspect free space for {output_root}: {exc}") from exc
    required_bytes = threshold * (1024**3)
    if free_bytes < required_bytes:
        free_gb = free_bytes / (1024**3)
        raise InsufficientDiskSpaceError(
            f"output root has {free_gb:.2f} GiB free; {threshold:.2f} GiB is required before start"
        )


def _nonnegative_finite_threshold(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _directory_size_bytes(directory: Path) -> int:
    total = 0
    pending = [directory]
    try:
        while pending:
            current = pending.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
    except OSError as exc:
        raise OutputRootError(f"could not inspect run size for {directory}: {exc}") from exc
    return total


def _git_command(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _detect_git(project_root: Path) -> dict[str, Any]:
    commit = _git_command(project_root, "rev-parse", "HEAD")
    if not commit or not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        return {"commit": "nogit", "branch": None, "dirty": None}
    branch = _git_command(project_root, "branch", "--show-current") or "DETACHED"
    porcelain = _git_command(project_root, "status", "--porcelain")
    return {
        "commit": commit.lower(),
        "branch": branch,
        "dirty": None if porcelain is None else bool(porcelain),
    }


def _git_short_sha(git: Mapping[str, Any]) -> str:
    commit = git.get("commit")
    if isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        return commit[:7].lower()
    return "nogit"


def _aware_now(now: datetime | None = None) -> datetime:
    value = datetime.now().astimezone() if now is None else now
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _create_unique_run_dir(
    output_root: Path,
    experiment_name: str,
    timestamp: datetime,
    seed: int,
    git_short_sha: str,
) -> tuple[str, Path]:
    experiment_dir = output_root / "runs" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    base_id = (
        f"{timestamp.strftime('%Y%m%d-%H%M%S')}_"
        f"{experiment_name}_seed{seed}_{git_short_sha}"
    )
    for suffix_index in range(0, 10_000):
        suffix = "" if suffix_index == 0 else f"_{suffix_index:02d}"
        run_id = f"{base_id}{suffix}"
        run_dir = experiment_dir / run_id
        try:
            run_dir.mkdir(mode=0o750)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise OutputRootError(f"could not allocate a unique run directory under {experiment_dir}")


class RunManager:
    """Own metadata and lifecycle transitions for one experiment run."""

    def __init__(self, paths: RunPaths, manifest: Mapping[str, Any]) -> None:
        self.paths = paths
        self._manifest = dict(_to_plain_data(manifest, "manifest"))

    @classmethod
    def create(
        cls,
        *,
        experiment_name: str,
        seed: int,
        resolved_config: Any,
        output_root: str | Path | None = None,
        project_root: str | Path | None = None,
        stage: str = "training",
        command: str | Sequence[str] | None = None,
        model: Mapping[str, Any] | None = None,
        pipeline: Mapping[str, Any] | None = None,
        min_free_space_gb_before_start: float = DEFAULT_MIN_FREE_SPACE_GB,
        now: datetime | None = None,
        git_metadata: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "RunManager":
        name = _validate_experiment_name(experiment_name)
        validated_seed = _validate_seed(seed)
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")

        # Reject secrets before creating a run directory, so a failed start
        # cannot leave behind a credential-bearing partial run.
        plain_config = _to_plain_data(resolved_config, "resolved_config")
        if not isinstance(plain_config, Mapping):
            raise TypeError("resolved_config must be a mapping or dataclass configuration")
        plain_model = _to_plain_data(model or {}, "model")
        plain_pipeline = _to_plain_data(pipeline or {}, "pipeline")
        rendered_command = _render_command(command)

        checkout_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else find_project_root()
        )
        selected_root = resolve_output_root(
            output_root,
            environ=environ,
            project_root=checkout_root,
        )
        _probe_output_root(selected_root, min_free_space_gb_before_start)

        git = (
            dict(_to_plain_data(git_metadata, "git"))
            if git_metadata is not None
            else _detect_git(checkout_root)
        )
        started_at = _aware_now(now)
        run_id, run_dir = _create_unique_run_dir(
            selected_root,
            name,
            started_at,
            validated_seed,
            _git_short_sha(git),
        )
        paths = RunPaths.from_run_dir(run_dir)
        try:
            for relative_path in _REQUIRED_DIRECTORIES:
                (run_dir / relative_path).mkdir(parents=True, exist_ok=True)

            manifest = {
                "run_id": run_id,
                "experiment_name": name,
                "stage": stage,
                "status": RunStatus.RUNNING.value,
                "start_time": started_at.isoformat(),
                "end_time": None,
                "seed": validated_seed,
                "git": git,
                "model": plain_model,
                "pipeline": plain_pipeline,
                "output_dir": str(run_dir),
            }
            _atomic_write_yaml(paths.manifest, manifest)
            _atomic_write_yaml(paths.resolved_config, plain_config)
            _atomic_write_text(
                paths.command,
                "#!/usr/bin/env bash\nset -euo pipefail\nexec " + rendered_command + "\n",
                mode=0o750,
            )
            # The terminal launcher overwrites this atomically with the real
            # process result. Its presence makes the minimal layout stable.
            _atomic_write_text(paths.exit_code, "")
            paths.terminal_log.touch(mode=0o640, exist_ok=True)
        except Exception:
            # Do not recursively remove anything: retaining a small partial
            # directory is safer than risking deletion outside our allocation.
            raise
        return cls(paths, manifest)

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        min_free_space_gb_before_start: float = DEFAULT_MIN_FREE_SPACE_GB,
        now: datetime | None = None,
    ) -> "RunManager":
        """Resume an incomplete run in place.

        A manifest left as ``running`` is treated as an abruptly terminated
        process. The caller must ensure that no original writer is still alive
        before invoking this method; RunManager never kills or probes another
        user's process.
        """

        paths = RunPaths.from_run_dir(run_dir)
        _probe_output_root(paths.output_root, min_free_space_gb_before_start)
        manifest = _read_yaml_mapping(paths.manifest)
        if manifest.get("run_id") != paths.run_dir.name:
            raise OutputRootError("manifest run_id does not match its run directory")
        try:
            previous_status = RunStatus(manifest.get("status"))
        except (TypeError, ValueError) as exc:
            raise OutputRootError("manifest contains an unknown run status") from exc
        if previous_status is RunStatus.COMPLETED:
            raise InvalidRunStateError("a completed run cannot be resumed")
        for relative_path in _REQUIRED_DIRECTORIES:
            (paths.run_dir / relative_path).mkdir(parents=True, exist_ok=True)
        if not paths.resolved_config.is_file() or not paths.command.is_file():
            raise OutputRootError("run is missing resolved_config.yaml or command.sh")

        resumed_at = _aware_now(now)
        manifest["status"] = RunStatus.RUNNING.value
        manifest["end_time"] = None
        manifest["last_resume_time"] = resumed_at.isoformat()
        resume_count = manifest.get("resume_count", 0)
        if isinstance(resume_count, bool) or not isinstance(resume_count, int) or resume_count < 0:
            raise OutputRootError("manifest resume_count must be a non-negative integer")
        manifest["resume_count"] = resume_count + 1
        # An exit code describes a finished process segment. Clear it before
        # publishing RUNNING again so observers never pair a resumed RUNNING
        # manifest with the previous segment's 130/error result.
        _atomic_write_text(paths.exit_code, "")
        _atomic_write_yaml(paths.manifest, manifest)

        from .terminal_logger import append_resume_separator

        append_resume_separator(paths.terminal_log, resumed_at)
        return cls(paths, manifest)

    @property
    def run_id(self) -> str:
        return str(self._manifest["run_id"])

    @property
    def status(self) -> RunStatus:
        return RunStatus(self._manifest["status"])

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a defensive copy of the current in-memory manifest."""

        return copy.deepcopy(self._manifest)

    def reload_manifest(self) -> dict[str, Any]:
        self._manifest = _read_yaml_mapping(self.paths.manifest)
        return self.manifest

    def record_exit_code(self, exit_code: int) -> None:
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise TypeError("exit_code must be an integer")
        _atomic_write_text(self.paths.exit_code, f"{exit_code}\n")

    def check_storage(
        self,
        *,
        min_free_space_gb_during_run: float = DEFAULT_MIN_FREE_SPACE_GB_DURING_RUN,
        warning_run_size_gb: float = DEFAULT_WARNING_RUN_SIZE_GB,
        max_run_size_gb: float = DEFAULT_MAX_RUN_SIZE_GB,
        raise_on_stop: bool = False,
    ) -> StorageCheck:
        """Inspect storage without deleting or creating experiment artifacts.

        ``STOP_REQUIRED`` is a signal to the training layer: save ``latest``,
        flush CSV/TensorBoard, mark the run, and terminate safely. RunManager
        deliberately performs none of those policy-specific operations here.
        """

        min_free = _nonnegative_finite_threshold(
            min_free_space_gb_during_run,
            "min_free_space_gb_during_run",
        )
        warning_size = _nonnegative_finite_threshold(
            warning_run_size_gb,
            "warning_run_size_gb",
        )
        max_size = _nonnegative_finite_threshold(max_run_size_gb, "max_run_size_gb")
        if max_size <= 0.0:
            raise ValueError("max_run_size_gb must be greater than 0")
        if warning_size > max_size:
            raise ValueError("warning_run_size_gb must not exceed max_run_size_gb")
        if not isinstance(raise_on_stop, bool):
            raise TypeError("raise_on_stop must be a boolean")

        try:
            free_bytes = shutil.disk_usage(self.paths.output_root).free
        except OSError as exc:
            raise OutputRootError(
                f"could not inspect free space for {self.paths.output_root}: {exc}"
            ) from exc
        run_bytes = _directory_size_bytes(self.paths.run_dir)
        free_gb = free_bytes / (1024**3)
        run_size_gb = run_bytes / (1024**3)
        stop_reasons: list[str] = []
        warning_reasons: list[str] = []
        if free_gb < min_free:
            stop_reasons.append(
                f"free space {free_gb:.2f} GiB is below the during-run minimum {min_free:.2f} GiB"
            )
        if run_size_gb >= max_size:
            stop_reasons.append(
                f"run size {run_size_gb:.2f} GiB reached the maximum {max_size:.2f} GiB"
            )
        elif run_size_gb >= warning_size:
            warning_reasons.append(
                f"run size {run_size_gb:.2f} GiB reached the warning threshold {warning_size:.2f} GiB"
            )
        if stop_reasons:
            status = StorageStatus.STOP_REQUIRED
            reasons = tuple(stop_reasons + warning_reasons)
        elif warning_reasons:
            status = StorageStatus.WARNING
            reasons = tuple(warning_reasons)
        else:
            status = StorageStatus.OK
            reasons = ()
        check = StorageCheck(
            status=status,
            free_space_gb=free_gb,
            run_size_gb=run_size_gb,
            reasons=reasons,
        )
        if check.should_stop and raise_on_stop:
            raise RunStorageLimitError(check)
        return check

    def complete(
        self,
        final_metrics: Mapping[str, Any] | None = None,
        *,
        exit_code: int = 0,
        now: datetime | None = None,
    ) -> None:
        updates: dict[str, Any] = {}
        if final_metrics is not None:
            updates["final_metrics"] = _to_plain_data(final_metrics, "final_metrics")
        self._finish(RunStatus.COMPLETED, exit_code, now=now, updates=updates)

    def fail(
        self,
        *,
        exit_code: int = 1,
        failure_reason: str | Enum | None = None,
        now: datetime | None = None,
    ) -> None:
        updates: dict[str, Any] = {}
        if failure_reason is not None:
            updates["failure_reason"] = _to_plain_data(failure_reason, "failure_reason")
        self._finish(RunStatus.FAILED, exit_code, now=now, updates=updates)

    def interrupt(self, *, exit_code: int = 130, now: datetime | None = None) -> None:
        self._finish(RunStatus.INTERRUPTED, exit_code, now=now, updates={})

    def _finish(
        self,
        status: RunStatus,
        exit_code: int,
        *,
        now: datetime | None,
        updates: Mapping[str, Any],
    ) -> None:
        if self.status is not RunStatus.RUNNING:
            raise InvalidRunStateError(
                f"cannot transition run from {self.status.value} to {status.value}"
            )
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise TypeError("exit_code must be an integer")
        if status is RunStatus.COMPLETED and exit_code != 0:
            raise ValueError("a completed run must have exit code 0")
        if status in (RunStatus.FAILED, RunStatus.INTERRUPTED) and exit_code == 0:
            raise ValueError(f"a {status.value} run must have a non-zero exit code")
        safe_updates = _to_plain_data(updates, "manifest_update")
        finished_at = _aware_now(now)
        next_manifest = copy.deepcopy(self._manifest)
        next_manifest.update(safe_updates)
        next_manifest["status"] = status.value
        next_manifest["end_time"] = finished_at.isoformat()
        next_manifest["exit_code"] = exit_code
        _atomic_write_yaml(self.paths.manifest, next_manifest)
        self.record_exit_code(exit_code)
        self._manifest = next_manifest


__all__ = [
    "DEFAULT_MIN_FREE_SPACE_GB",
    "DEFAULT_MIN_FREE_SPACE_GB_DURING_RUN",
    "DEFAULT_WARNING_RUN_SIZE_GB",
    "DEFAULT_MAX_RUN_SIZE_GB",
    "FORBIDDEN_ARTIFACT_DIRECTORIES",
    "InsufficientDiskSpaceError",
    "InvalidRunStateError",
    "OUTPUT_ROOT_ENV",
    "OutputRootError",
    "RunManager",
    "RunManagerError",
    "RunPaths",
    "RunStorageLimitError",
    "RunStatus",
    "StorageCheck",
    "StorageStatus",
    "SensitiveDataError",
    "find_project_root",
    "resolve_output_root",
]
