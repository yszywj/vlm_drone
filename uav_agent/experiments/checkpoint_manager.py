"""Bounded, resumable checkpoint storage for lightweight experiments.

The manager deliberately exposes only two checkpoint slots:

``latest``
    A resumable training snapshot.  Optimizer, scheduler and RNG state may be
    included here.

``best``
    The best validation snapshot.  Training-only state is removed from
    mapping payloads by default so this slot remains suitable for inference.

Payloads are serialized with :mod:`pickle` because Python and NumPy RNG states
are not generally JSON-compatible.  Consequently, callers must only load
checkpoints they trust.  No PyTorch import is required.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Final
from uuid import uuid4


class CheckpointError(RuntimeError):
    """Base error raised by checkpoint operations."""


class CheckpointSecurityError(CheckpointError):
    """Raised when a payload attempts to store prohibited base-model data."""


class CheckpointFormatError(CheckpointError):
    """Raised when checkpoint metadata or payload data is malformed."""


# A callback writes model/adapter-specific files into the staging directory.
# The mapping is already filtered for the selected checkpoint kind.
CheckpointSaveCallback = Callable[[Path, Mapping[str, Any]], None]
CheckpointLoadCallback = Callable[[Path], Mapping[str, Any]]


_CHECKPOINT_KINDS: Final = frozenset({"best", "latest"})
_META_FILENAME: Final = "checkpoint_meta.json"
_STATE_FILENAME: Final = "checkpoint_state.pkl"

# Common training-state names.  These are retained in ``latest`` but removed
# from mapping payloads written to ``best``.
_TRAINING_STATE_KEYS: Final = frozenset(
    {
        "optimizer",
        "optimizer_state",
        "optimizer_state_dict",
        "scheduler",
        "scheduler_state",
        "scheduler_state_dict",
        "lr_scheduler",
        "lr_scheduler_state",
        "rng_state",
        "random_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "normalizer_state",
        "scaler",
        "scaler_state",
        "grad_scaler_state",
    }
)
_OPTIMIZER_KEYS: Final = frozenset(
    {"optimizer", "optimizer_state", "optimizer_state_dict"}
)
_PROHIBITED_FULL_MODEL_KEYS: Final = frozenset(
    {
        "base_model_state",
        "base_model_state_dict",
        "full_base_model",
        "full_base_model_state",
        "full_model",
        "full_model_state_dict",
        "save_full_base_model",
    }
)
_QWEN_SHARD_RE: Final = re.compile(
    r"^(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:safetensors|bin)$",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validated_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_metric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointFormatError(f"metric {name!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CheckpointFormatError(f"metric {name!r} must be finite")
    return result


class CheckpointManager:
    """Maintain exactly one ``best`` and one ``latest`` checkpoint.

    Args:
        checkpoints_dir: Directory containing the two bounded slots.
        save_callback: Optional model-specific writer.  It receives a staging
            directory and a read-only mapping payload.  When omitted, the
            manager serializes the mapping itself.
        load_callback: Optional counterpart for callback-written checkpoints.
        latest_interval_steps: Minimum step distance between automatic latest
            saves.  Direct :meth:`save_latest` calls are never rate-limited.

    ``save_full_base_model=True`` is intentionally rejected.  A Qwen base
    model belongs in the shared model cache; experiment checkpoints should
    contain only inference weights or adapters.
    """

    def __init__(
        self,
        checkpoints_dir: str | os.PathLike[str],
        *,
        save_callback: CheckpointSaveCallback | None = None,
        load_callback: CheckpointLoadCallback | None = None,
        latest_interval_steps: int = 50_000,
        save_best: bool = True,
        save_latest: bool = True,
        save_full_base_model: bool = False,
        save_adapter_only: bool = True,
        save_optimizer_in_latest_only: bool = True,
    ) -> None:
        if save_full_base_model:
            raise CheckpointSecurityError(
                "save_full_base_model is prohibited; save an adapter or "
                "inference-only payload instead"
            )
        self._checkpoints_dir = Path(checkpoints_dir).expanduser().resolve()
        self._latest_interval_steps = _validated_nonnegative_int(
            latest_interval_steps, "latest_interval_steps"
        )
        if self._latest_interval_steps == 0:
            raise ValueError("latest_interval_steps must be greater than zero")
        if not isinstance(save_best, bool) or not isinstance(save_latest, bool):
            raise TypeError("save_best and save_latest must be bool")
        if not isinstance(save_adapter_only, bool):
            raise TypeError("save_adapter_only must be bool")
        if not isinstance(save_optimizer_in_latest_only, bool):
            raise TypeError("save_optimizer_in_latest_only must be bool")

        self._save_callback = save_callback
        self._load_callback = load_callback
        self._save_best_enabled = save_best
        self._save_latest_enabled = save_latest
        self._save_adapter_only = save_adapter_only
        self._optimizer_latest_only = save_optimizer_in_latest_only
        self._lock = threading.RLock()
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted_replacements()

    @property
    def checkpoints_dir(self) -> Path:
        return self._checkpoints_dir

    @property
    def best_path(self) -> Path:
        return self._checkpoints_dir / "best"

    @property
    def latest_path(self) -> Path:
        return self._checkpoints_dir / "latest"

    @property
    def best_meta(self) -> dict[str, Any] | None:
        return self.meta("best")

    @property
    def latest_meta(self) -> dict[str, Any] | None:
        return self.meta("latest")

    def meta(self, checkpoint: str | os.PathLike[str]) -> dict[str, Any] | None:
        """Return a defensive metadata copy, or ``None`` if the slot is empty."""

        path = self._resolve_checkpoint(checkpoint)
        meta_path = path / _META_FILENAME
        if not meta_path.is_file():
            return None
        try:
            value = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointFormatError(
                f"cannot read checkpoint metadata: {meta_path}"
            ) from exc
        if not isinstance(value, dict):
            raise CheckpointFormatError("checkpoint metadata must be a JSON object")
        return value

    def load(
        self,
        checkpoint: str | os.PathLike[str] = "latest",
        *,
        load_callback: CheckpointLoadCallback | None = None,
    ) -> dict[str, Any]:
        """Load a trusted mapping payload from ``best`` or ``latest``.

        Callback-written formats can be loaded by supplying ``load_callback``
        here or in the constructor.  Metadata remains available separately via
        :meth:`meta`.
        """

        path = self._resolve_checkpoint(checkpoint)
        if self.meta(path) is None:
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        callback = load_callback or self._load_callback
        if callback is not None:
            value = callback(path)
        else:
            state_path = path / _STATE_FILENAME
            if not state_path.is_file():
                raise CheckpointFormatError(
                    "checkpoint uses a custom format but no load_callback was supplied"
                )
            try:
                with state_path.open("rb") as stream:
                    value = pickle.load(stream)  # noqa: S301 - trusted local checkpoint
            except (OSError, pickle.PickleError, EOFError) as exc:
                raise CheckpointFormatError(
                    f"cannot load checkpoint payload: {state_path}"
                ) from exc
        if not isinstance(value, Mapping):
            raise CheckpointFormatError("loaded checkpoint payload must be a mapping")
        return dict(value)

    def save_latest(
        self,
        global_step: int,
        payload: Mapping[str, Any] | None = None,
        *,
        update: int | None = None,
    ) -> Path | None:
        """Overwrite ``latest`` with a resumable snapshot."""

        if not self._save_latest_enabled:
            return None
        with self._lock:
            step, effective_update, state = self._prepare_payload(
                global_step=global_step,
                update=update,
                payload=payload,
                kind="latest",
            )
            current = self.latest_meta
            if current is not None:
                current_step = current.get("global_step")
                if isinstance(current_step, int) and step < current_step:
                    raise CheckpointFormatError(
                        "latest global_step cannot move backwards"
                    )
            metadata = self._make_metadata(
                kind="latest",
                global_step=step,
                update=effective_update,
                metrics=None,
                state=state,
            )
            self._write_slot("latest", state, metadata)
        return self.latest_path

    def try_save_latest(self, **kwargs: Any) -> bool:
        """Best-effort latest save for exception/signal cleanup paths."""

        try:
            return self.save_latest(**kwargs) is not None
        except Exception:
            return False

    def maybe_save_latest(
        self,
        global_step: int,
        payload: Mapping[str, Any] | None = None,
        *,
        update: int | None = None,
        force: bool = False,
    ) -> Path | None:
        """Save latest only when its bounded interval has elapsed."""

        with self._lock:
            step = _validated_nonnegative_int(global_step, "global_step")
            if not self._save_latest_enabled:
                return None
            previous = self.latest_meta
            previous_step = None if previous is None else previous.get("global_step")
            due = (
                force
                or (previous_step is None and step >= self._latest_interval_steps)
                or (
                    isinstance(previous_step, int)
                    and step - previous_step >= self._latest_interval_steps
                )
            )
            if not due:
                return None
            return self.save_latest(global_step=step, update=update, payload=payload)

    def save_best(
        self,
        global_step: int,
        metrics: Mapping[str, Any],
        payload: Mapping[str, Any] | None = None,
        *,
        update: int | None = None,
    ) -> Path | None:
        """Unconditionally replace ``best`` with the supplied validation result."""

        if not self._save_best_enabled:
            return None
        canonical_metrics = self._validated_selection_metrics(metrics)
        step, effective_update, state = self._prepare_payload(
            global_step=global_step,
            update=update,
            payload=payload,
            kind="best",
        )
        metadata = self._make_metadata(
            kind="best",
            global_step=step,
            update=effective_update,
            metrics=canonical_metrics,
            state=state,
        )
        self._write_slot("best", state, metadata)
        return self.best_path

    def maybe_save_best(
        self,
        global_step: int,
        metrics: Mapping[str, Any],
        payload: Mapping[str, Any] | None = None,
        *,
        update: int | None = None,
    ) -> Path | None:
        """Save when validation metrics improve by the documented ordering.

        Mission success rate is maximized.  Ties minimize false-lock rate,
        collision rate, safety-abort rate, then mean mission time.
        """

        with self._lock:
            if not self._save_best_enabled:
                return None
            candidate = self._validated_selection_metrics(metrics)
            current_meta = self.best_meta
            if current_meta is not None:
                current_raw = current_meta.get("selection_metrics")
                if not isinstance(current_raw, Mapping):
                    raise CheckpointFormatError("best checkpoint has no selection metrics")
                current = self._validated_selection_metrics(current_raw)
                if self._selection_key(candidate) >= self._selection_key(current):
                    return None
            return self.save_best(
                global_step=global_step,
                update=update,
                payload=payload,
                metrics=candidate,
            )

    def maybe_save(
        self,
        checkpoint: str,
        *,
        global_step: int,
        update: int | None = None,
        payload: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> Path | None:
        """Dispatch to a bounded ``best`` or ``latest`` maybe-save operation."""

        if checkpoint == "latest":
            return self.maybe_save_latest(
                global_step=global_step,
                update=update,
                payload=payload,
                force=force,
            )
        if checkpoint == "best":
            if metrics is None:
                raise ValueError("metrics are required when saving best")
            return self.maybe_save_best(
                global_step=global_step,
                update=update,
                payload=payload,
                metrics=metrics,
            )
        raise ValueError("checkpoint must be 'best' or 'latest'")

    def _resolve_checkpoint(self, value: str | os.PathLike[str]) -> Path:
        if isinstance(value, str) and value in _CHECKPOINT_KINDS:
            return self._checkpoints_dir / value
        candidate = Path(value).expanduser().resolve()
        if candidate not in {self.best_path, self.latest_path}:
            raise ValueError("only the best and latest checkpoint slots are supported")
        return candidate

    def _prepare_payload(
        self,
        *,
        global_step: int,
        update: int | None,
        payload: Mapping[str, Any] | None,
        kind: str,
    ) -> tuple[int, int, dict[str, Any]]:
        step = _validated_nonnegative_int(global_step, "global_step")
        if payload is None:
            state: dict[str, Any] = {}
        elif not isinstance(payload, Mapping):
            raise TypeError("checkpoint payload must be a mapping")
        else:
            state = dict(payload)
        self._reject_full_base_model_payload(state)

        existing_step = state.get("global_step")
        if existing_step is not None and existing_step != step:
            raise CheckpointFormatError(
                "payload global_step does not match save global_step"
            )
        if update is None:
            update_value = state.get("update", step)
        else:
            update_value = update
            existing_update = state.get("update")
            if existing_update is not None and existing_update != update:
                raise CheckpointFormatError("payload update does not match save update")
        effective_update = _validated_nonnegative_int(update_value, "update")

        if kind == "best" and self._optimizer_latest_only:
            state = self._without_training_state(state)
        state["global_step"] = step
        state["update"] = effective_update
        return step, effective_update, state

    def _reject_full_base_model_payload(self, payload: Mapping[str, Any]) -> None:
        """Reject prohibited base-model material at any mapping depth."""

        ancestors: set[int] = set()

        def visit(value: Any, path: str, depth: int) -> None:
            if depth > 32:
                raise CheckpointSecurityError("checkpoint payload nesting is too deep")
            if isinstance(value, Mapping):
                identity = id(value)
                if identity in ancestors:
                    raise CheckpointSecurityError("checkpoint payload must not be cyclic")
                ancestors.add(identity)
                try:
                    for key, item in value.items():
                        normalized = str(key).strip().lower()
                        name = Path(str(key)).name
                        adapter_only_model_keys = {
                            "model",
                            "weights",
                            "weight",
                            "state_dict",
                            "model_weights",
                            "model_state",
                            "model_state_dict",
                            "inference_weights",
                            "pretrained_model_state_dict",
                            "qwen_state_dict",
                        }
                        if normalized in _PROHIBITED_FULL_MODEL_KEYS or (
                            self._save_adapter_only and normalized in adapter_only_model_keys
                        ):
                            if normalized != "save_full_base_model" or bool(item):
                                raise CheckpointSecurityError(
                                    f"full base-model payload field is prohibited: {path}.{key}"
                                )
                        if _QWEN_SHARD_RE.fullmatch(name) or name.lower() in {
                            "model.safetensors.index.json",
                            "pytorch_model.bin.index.json",
                        }:
                            raise CheckpointSecurityError(
                                f"Qwen base-model file is prohibited: {path}.{key}"
                            )
                        visit(item, f"{path}.{key}", depth + 1)
                finally:
                    ancestors.remove(identity)
                return
            if isinstance(value, (list, tuple)):
                identity = id(value)
                if identity in ancestors:
                    raise CheckpointSecurityError("checkpoint payload must not be cyclic")
                ancestors.add(identity)
                try:
                    for index, item in enumerate(value):
                        visit(item, f"{path}[{index}]", depth + 1)
                finally:
                    ancestors.remove(identity)
                return
            if isinstance(value, (str, Path)):
                name = Path(str(value)).name
                if _QWEN_SHARD_RE.fullmatch(name) or name.lower() in {
                    "model.safetensors.index.json",
                    "pytorch_model.bin.index.json",
                }:
                    raise CheckpointSecurityError(
                        f"Qwen base-model file reference is prohibited in checkpoint payload: {path}"
                    )

        visit(payload, "payload", 0)

    @classmethod
    def _without_training_state(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Recursively remove optimizer/scheduler/RNG state from ``best``."""

        def filtered(value: Any, depth: int) -> Any:
            if depth > 32:
                raise CheckpointSecurityError("checkpoint payload nesting is too deep")
            if isinstance(value, Mapping):
                return {
                    key: filtered(item, depth + 1)
                    for key, item in value.items()
                    if str(key).strip().lower() not in _TRAINING_STATE_KEYS
                }
            if isinstance(value, list):
                return [filtered(item, depth + 1) for item in value]
            if isinstance(value, tuple):
                return tuple(filtered(item, depth + 1) for item in value)
            return value

        return filtered(payload, 0)

    @classmethod
    def _contains_optimizer_state(cls, value: Any, depth: int = 0) -> bool:
        if depth > 32:
            return False
        if isinstance(value, Mapping):
            return any(
                str(key).strip().lower() in _OPTIMIZER_KEYS
                or cls._contains_optimizer_state(item, depth + 1)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(cls._contains_optimizer_state(item, depth + 1) for item in value)
        return False

    @staticmethod
    def _validated_selection_metrics(
        metrics: Mapping[str, Any],
    ) -> dict[str, float]:
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if "mission_success_rate" not in metrics:
            raise CheckpointFormatError("mission_success_rate is required")

        def optional_rate(name: str) -> float:
            return _finite_metric(metrics.get(name, 1.0), name)

        mission_time_key = next(
            (
                name
                for name in (
                    "mean_mission_time_s",
                    "mission_time_s",
                    "mission_time",
                )
                if name in metrics
            ),
            None,
        )
        mission_time = (
            _finite_metric(metrics[mission_time_key], mission_time_key)
            if mission_time_key is not None
            else 1.0e300
        )
        canonical = {
            "mission_success_rate": _finite_metric(
                metrics["mission_success_rate"], "mission_success_rate"
            ),
            "false_lock_rate": optional_rate("false_lock_rate"),
            "collision_rate": optional_rate("collision_rate"),
            "safety_abort_rate": optional_rate("safety_abort_rate"),
            "mean_mission_time_s": mission_time,
        }
        for name in (
            "mission_success_rate",
            "false_lock_rate",
            "collision_rate",
            "safety_abort_rate",
        ):
            if not 0.0 <= canonical[name] <= 1.0:
                raise CheckpointFormatError(f"metric {name!r} must be in [0, 1]")
        if canonical["mean_mission_time_s"] < 0.0:
            raise CheckpointFormatError("mean_mission_time_s must be non-negative")
        return canonical

    @staticmethod
    def _selection_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
        return (
            -metrics["mission_success_rate"],
            metrics["false_lock_rate"],
            metrics["collision_rate"],
            metrics["safety_abort_rate"],
            metrics["mean_mission_time_s"],
        )

    @staticmethod
    def _make_metadata(
        *,
        kind: str,
        global_step: int,
        update: int,
        metrics: Mapping[str, float] | None,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "kind": kind,
            "created_at": _utc_now(),
            "global_step": global_step,
            "update": update,
            "has_optimizer_state": CheckpointManager._contains_optimizer_state(state),
        }
        if metrics is not None:
            metadata["selection_metrics"] = dict(metrics)
        return metadata

    def _write_slot(
        self,
        kind: str,
        state: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        if kind not in _CHECKPOINT_KINDS:
            raise ValueError("only best and latest checkpoint slots are supported")
        destination = self._checkpoints_dir / kind
        with self._lock:
            staging = Path(
                tempfile.mkdtemp(prefix=f".{kind}.tmp-", dir=self._checkpoints_dir)
            )
            try:
                read_only_state = MappingProxyType(dict(state))
                if self._save_callback is None:
                    with (staging / _STATE_FILENAME).open("wb") as stream:
                        pickle.dump(dict(state), stream, protocol=pickle.HIGHEST_PROTOCOL)
                        stream.flush()
                        os.fsync(stream.fileno())
                else:
                    self._save_callback(staging, read_only_state)
                self._validate_staged_files(staging, kind)
                meta_path = staging / _META_FILENAME
                meta_path.write_text(
                    json.dumps(
                        dict(metadata), ensure_ascii=False, indent=2, sort_keys=True
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._fsync_path(meta_path)
                self._fsync_directory(staging)
                self._replace_directory(staging, destination)
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                raise

    def _validate_staged_files(self, staging: Path, kind: str) -> None:
        for path in staging.rglob("*"):
            if path.is_symlink():
                raise CheckpointSecurityError(
                    "checkpoint callbacks may not create symbolic links"
                )
            if not path.is_file():
                continue
            name = path.name.lower()
            if _QWEN_SHARD_RE.fullmatch(name) or name in {
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            }:
                raise CheckpointSecurityError(
                    f"Qwen base-model file is prohibited: {path.name}"
                )
            if kind == "best" and self._optimizer_latest_only:
                stem = path.stem.lower()
                if any(
                    marker in stem
                    for marker in ("optimizer", "scheduler", "rng", "random_state", "normalizer", "scaler")
                ):
                    raise CheckpointSecurityError(
                        f"training-state file is prohibited in best: {path.name}"
                    )
            if self._save_adapter_only and path.suffix.lower() in {
                ".safetensors",
                ".bin",
                ".pt",
                ".pth",
                ".ckpt",
            }:
                if "adapter" not in name:
                    raise CheckpointSecurityError(
                        f"save_adapter_only prohibits non-adapter weight file: {path.name}"
                    )

    def _recover_interrupted_replacements(self) -> None:
        """Restore a hidden old slot after a crash between the two renames."""

        for kind in _CHECKPOINT_KINDS:
            destination = self._checkpoints_dir / kind
            backups = sorted(
                self._checkpoints_dir.glob(f".{kind}.old-*"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if not destination.exists() and backups:
                os.replace(backups.pop(0), destination)
            for stale in backups:
                if stale.is_dir() and not stale.is_symlink():
                    shutil.rmtree(stale)
            for staging in self._checkpoints_dir.glob(f".{kind}.tmp-*"):
                if staging.is_dir() and not staging.is_symlink():
                    shutil.rmtree(staging)

    @staticmethod
    def _fsync_path(path: Path) -> None:
        try:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        except OSError:
            # Some network filesystems do not support fsync; os.replace below
            # still prevents readers from observing a partially written tree.
            pass

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass

    def _replace_directory(self, staging: Path, destination: Path) -> None:
        """Commit a staged tree while retaining the old slot on failure."""

        backup = self._checkpoints_dir / f".{destination.name}.old-{uuid4().hex}"
        moved_old = False
        try:
            if destination.exists():
                os.replace(destination, backup)
                moved_old = True
            os.replace(staging, destination)
            self._fsync_directory(self._checkpoints_dir)
        except Exception:
            if moved_old and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        finally:
            # Preserve the backup for manual recovery if both commit and
            # rollback fail.  Under normal success/rollback paths destination
            # exists and the obsolete tree can be removed safely.
            if backup.exists() and destination.exists():
                shutil.rmtree(backup)


__all__ = [
    "CheckpointError",
    "CheckpointFormatError",
    "CheckpointLoadCallback",
    "CheckpointManager",
    "CheckpointSaveCallback",
    "CheckpointSecurityError",
]
