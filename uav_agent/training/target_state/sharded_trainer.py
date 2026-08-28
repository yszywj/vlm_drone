"""Episode-sharded Target State training over the :mod:`pc_trans` CLI.

The model and optimizer are continuous across shards.  Transport/cache state
is deliberately accessed only through the public ``python -m pc_trans.cli``
boundary; this module never imports the independent ``pc_trans`` package.
Progress is committed at shard boundaries in ``latest.pt`` before any active
archive is consumed.
"""

from __future__ import annotations

import csv
import cmath
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
from hashlib import sha256
import json
import math
import numbers
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.target_state.dataset import split_for_episode
from experiments.metric_logger import ScalarEventWriter
from training.target_state.config import TargetStateTrainingConfig, TrainingStage
from training.target_state.data import GEOMETRY_INPUT_FIELDS, TargetStateTorchDataset
from training.target_state.losses import compute_target_state_losses
from training.target_state.model import TemporalRayDepthNet
from training.target_state.shard_runtime import (
    MaterializedShard,
    cleanup_materialized_path,
    cleanup_materialized_shard,
    materialize_shard,
)
from training.target_state.shards import (
    ShardIndexEntry,
    TargetStateShardIndex,
    load_shard_index,
)
from training.target_state.trainer import (
    MODEL_SCHEMA_VERSION,
    MODEL_TYPE,
    OUTPUT_FIELDS,
    TargetStateEvaluationAccumulator,
    TargetStateTrainingError,
    _device,
    _json_safe,
    _seed_everything,
    _to_device,
    _write_figures,
    accumulate_evaluation,
    evaluate_promotion,
    sha256_file,
    validate_initial_checkpoint,
)


TRAINING_PROTOCOL = "episode_sharded_v1"
RESUME_PROTOCOL = "shard_boundary"
SHARD_RNG_PROTOCOL = "per_shard_stable_seed_v1"
_RUN_LOCK_FILENAME = ".sharded_training.lock"
_PHASES = {"train", "validation", "final_test", "complete"}
_METRIC_FIELDS = (
    "global_epoch",
    "global_step",
    "train_total",
    "train_depth",
    "train_position_3d",
    "train_reprojection",
    "train_gaussian_nll",
    "train_validity_bce",
    "validation_loss",
    "validation_position_median_error_m",
    "validation_position_p95_error_m",
    "validation_measurement_failure_rate",
    "validation_no_target_false_positive_rate",
    "validation_covariance_error_spearman",
)
_LOSS_NAMES = (
    "total",
    "depth",
    "position_3d",
    "reprojection",
    "gaussian_nll",
    "validity_bce",
)


class ShardedTrainingError(TargetStateTrainingError):
    """Raised when sharded training cannot continue without guessing."""


class ShardLifecycle(Protocol):
    """Narrow transport boundary used by production and offline tests."""

    def request(self, run_id: str, filenames: Sequence[str]) -> None: ...

    def wait_shard(self, run_id: str, filename: str, timeout_s: float) -> None: ...

    def shard_state(self, run_id: str, filename: str) -> Mapping[str, object]: ...

    def activate(self, run_id: str, filename: str) -> Path: ...

    def consume(self, run_id: str, filename: str, *, delete: bool) -> None: ...


class ShardMaterializer(Protocol):
    def materialize(
        self, archive_path: Path, index: TargetStateShardIndex
    ) -> MaterializedShard: ...

    def cleanup(self, materialized: MaterializedShard) -> None: ...

    def cleanup_path(self, path: Path, *, materialized_root: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class DefaultShardMaterializer:
    """Adapter around the validated target-state shard runtime."""

    def materialize(
        self, archive_path: Path, index: TargetStateShardIndex
    ) -> MaterializedShard:
        return materialize_shard(archive_path, index)

    def cleanup(self, materialized: MaterializedShard) -> None:
        cleanup_materialized_shard(materialized)

    def cleanup_path(self, path: Path, *, materialized_root: Path) -> None:
        cleanup_materialized_path(path, materialized_root=materialized_root)


@dataclass(frozen=True, slots=True)
class PCTransCLI:
    """Stable subprocess adapter for the independent pc_trans repository."""

    project_root: Path
    config_path: Path
    python_executable: Path = Path(sys.executable)

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).expanduser().resolve())
        object.__setattr__(self, "config_path", Path(self.config_path).expanduser().resolve())
        object.__setattr__(
            self, "python_executable", Path(self.python_executable).expanduser().resolve()
        )
        if not (self.project_root / "pc_trans" / "cli.py").is_file():
            raise ShardedTrainingError(
                f"pc_trans project root has no pc_trans/cli.py: {self.project_root}"
            )
        if not self.config_path.is_file():
            raise ShardedTrainingError(f"pc_trans config does not exist: {self.config_path}")

    def _run(
        self,
        command: str,
        *arguments: str,
        accepted_returncodes: tuple[int, ...] = (0,),
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        invocation = [
            str(self.python_executable),
            "-m",
            "pc_trans.cli",
            "--config",
            str(self.config_path),
            command,
            *arguments,
        ]
        try:
            result = subprocess.run(
                invocation,
                cwd=self.project_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ShardedTrainingError(
                f"pc_trans {command} could not run: {exc}"
            ) from exc
        if result.returncode not in accepted_returncodes:
            detail = (result.stderr or result.stdout).strip()
            raise ShardedTrainingError(
                f"pc_trans {command} failed with exit {result.returncode}: {detail}"
            )
        return result

    def request(self, run_id: str, filenames: Sequence[str]) -> None:
        if not filenames:
            raise ShardedTrainingError("pc_trans request cannot be empty")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{run_id}.", suffix=".shards.list"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(filenames) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._run(
                "request",
                "--run-id",
                run_id,
                "--files",
                temporary_name,
                "--replace",
            )
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass

    def wait_shard(self, run_id: str, filename: str, timeout_s: float) -> None:
        self._run(
            "wait-shard",
            "--run-id",
            run_id,
            "--shard",
            filename,
            "--timeout",
            str(timeout_s),
            timeout_s=max(timeout_s + 30.0, 30.0),
        )

    def shard_state(self, run_id: str, filename: str) -> Mapping[str, object]:
        result = self._run(
            "shard-state",
            "--run-id",
            run_id,
            "--shard",
            filename,
            "--json",
        )
        duplicate_fields: list[str] = []

        def reject_duplicate_fields(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            decoded: dict[str, object] = {}
            for key, value in pairs:
                if key in decoded:
                    duplicate_fields.append(key)
                decoded[key] = value
            return decoded

        try:
            payload = json.loads(
                result.stdout, object_pairs_hook=reject_duplicate_fields
            )
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ShardedTrainingError(
                f"pc_trans shard-state returned invalid JSON: {result.stdout!r}"
            ) from exc
        if duplicate_fields:
            raise ShardedTrainingError(
                "pc_trans shard-state JSON contains duplicate fields: "
                f"{sorted(set(duplicate_fields))}"
            )
        if not isinstance(payload, dict):
            raise ShardedTrainingError("pc_trans shard-state JSON must be an object")
        expected_fields = {
            "run_id",
            "shard",
            "requested",
            "ready",
            "active",
            "consumed_record",
            "deleted",
        }
        if set(payload) != expected_fields:
            missing = sorted(expected_fields - set(payload))
            unexpected = sorted(set(payload) - expected_fields)
            raise ShardedTrainingError(
                "pc_trans shard-state JSON fields do not match the protocol: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if payload["run_id"] != run_id or payload["shard"] != filename:
            raise ShardedTrainingError(
                "pc_trans shard-state identity mismatch: "
                f"expected={run_id}/{filename}, "
                f"actual={payload['run_id']!r}/{payload['shard']!r}"
            )
        for field_name in ("requested", "ready", "active", "consumed_record"):
            if type(payload[field_name]) is not bool:
                raise ShardedTrainingError(
                    f"pc_trans shard-state {field_name} must be bool"
                )
        deleted = payload["deleted"]
        if deleted is not None and type(deleted) is not bool:
            raise ShardedTrainingError(
                "pc_trans shard-state deleted must be bool or null"
            )
        if payload["requested"] is not True:
            raise ShardedTrainingError(
                "pc_trans shard-state must identify the shard as requested"
            )

        ready = payload["ready"]
        active = payload["active"]
        consumed = payload["consumed_record"]
        contradictions: list[str] = []
        if ready and active:
            contradictions.append("shard is simultaneously ready and active")
        if not consumed and deleted is not None:
            contradictions.append("deleted must be null without a consumed record")
        if consumed and deleted is None:
            contradictions.append("a consumed record must declare deleted as bool")
        if consumed and deleted is True and (ready or active):
            contradictions.append(
                "deleted=true conflicts with a ready or active shard"
            )
        if consumed and deleted is False and (ready or not active):
            contradictions.append(
                "deleted=false requires one active shard and no ready shard"
            )
        if contradictions:
            raise ShardedTrainingError(
                "pc_trans shard-state is contradictory: "
                + "; ".join(contradictions)
            )
        # pc_trans deliberately exposes lifecycle facts, not dataset semantics.
        # Derive the local orchestration label without changing that public API.
        result = dict(payload)
        result["state"] = _state_name(result)
        return result

    def activate(self, run_id: str, filename: str) -> Path:
        result = self._run(
            "activate", "--run-id", run_id, "--shard", filename
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise ShardedTrainingError("pc_trans activate returned no active path")
        path = Path(lines[-1]).expanduser().resolve()
        if path.name != filename or not path.is_file():
            raise ShardedTrainingError(
                f"pc_trans activate returned an invalid active shard path: {path}"
            )
        return path

    def consume(self, run_id: str, filename: str, *, delete: bool) -> None:
        arguments = ["--run-id", run_id, "--shard", filename]
        if delete:
            arguments.append("--delete")
        self._run("consume", *arguments)


@dataclass(frozen=True, slots=True)
class ShardedTrainingOptions:
    shard_index_path: Path
    pc_trans_root: Path
    pc_trans_config: Path
    bridge_root: Path
    run_id_prefix: str
    resume_checkpoint: Path | None = None
    wait_timeout_s: float = 86400.0
    pc_trans_python: Path = Path(sys.executable)

    def __post_init__(self) -> None:
        for name in (
            "shard_index_path",
            "pc_trans_root",
            "pc_trans_config",
            "bridge_root",
            "pc_trans_python",
        ):
            value = Path(getattr(self, name)).expanduser().resolve()
            object.__setattr__(self, name, value)
        if self.resume_checkpoint is not None:
            object.__setattr__(
                self,
                "resume_checkpoint",
                Path(self.resume_checkpoint).expanduser().resolve(),
            )
        if (
            not self.run_id_prefix
            or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for character in self.run_id_prefix)
            or self.run_id_prefix in {".", ".."}
        ):
            raise ValueError("run_id_prefix must be a non-empty pc_trans-safe identifier")
        if not math.isfinite(self.wait_timeout_s) or self.wait_timeout_s < 0.0:
            raise ValueError("wait_timeout_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ShardedTrainingHooks:
    """Fault-injection/observability hooks used by offline tests."""

    after_checkpoint: Callable[[Path, Mapping[str, object]], None] | None = None
    after_consume: Callable[[str, str], None] | None = None


@dataclass(frozen=True, slots=True)
class ShardedTrainingResult:
    run_dir: Path
    best_checkpoint: Path
    latest_checkpoint: Path
    model_manifest: Path
    validation_metrics: Mapping[str, object]
    test_metrics: Mapping[str, object]
    promoted: bool
    elapsed_s: float
    global_step: int

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "best_checkpoint": str(self.best_checkpoint),
            "latest_checkpoint": str(self.latest_checkpoint),
            "model_manifest": str(self.model_manifest),
            "validation_metrics": dict(self.validation_metrics),
            "test_metrics": dict(self.test_metrics),
            "promoted": self.promoted,
            "elapsed_s": self.elapsed_s,
            "global_step": self.global_step,
        }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_finite_checkpoint_state(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        candidate = torch.load(temporary, map_location="cpu", weights_only=False)
        if not isinstance(candidate, Mapping):
            raise ShardedTrainingError(
                f"temporary checkpoint did not reload as a mapping: {path}"
            )
        _assert_finite_checkpoint_state(candidate)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(_json_safe(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _stable_seed(base_seed: int, *parts: object) -> int:
    message = ":".join((str(base_seed), *(str(part) for part in parts)))
    return int.from_bytes(sha256(message.encode("utf-8")).digest()[:8], "big")


def deterministic_shard_order(
    shards: Sequence[ShardIndexEntry], *, base_seed: int, global_epoch: int
) -> tuple[str, ...]:
    """Return a process-independent order for one complete global epoch."""

    filenames = sorted(entry.filename for entry in shards)
    random.Random(_stable_seed(base_seed, "train_shards", global_epoch)).shuffle(filenames)
    return tuple(filenames)


def deterministic_batch_seed(
    *, base_seed: int, global_epoch: int, filename: str
) -> int:
    return _stable_seed(base_seed, "batches", global_epoch, filename) % (2**63 - 1)


def deterministic_shard_execution_seed(
    *, base_seed: int, global_epoch: int, filename: str
) -> int:
    """Seed model-side RNG independently at every retryable shard boundary."""

    return _stable_seed(base_seed, "execution", global_epoch, filename) % (2**32)


def _model_config(config: TargetStateTrainingConfig) -> dict[str, int]:
    return {
        "geometry_input_dim": config.geometry_input_dim,
        "roi_channels": 4,
        "roi_size_px": config.roi_size_px,
        "roi_feature_dim": config.roi_feature_dim,
        "geometry_feature_dim": config.geometry_feature_dim,
        "hidden_dim": config.hidden_dim,
        "gru_layers": config.gru_layers,
        "time_steps": config.history_size + 1,
    }


def _training_contract_sha256(config: TargetStateTrainingConfig) -> str:
    """Hash semantic settings that may not change during same-run resume."""

    payload = {
        "stage": config.stage.value,
        "history_size": config.history_size,
        "max_history_age_s": config.max_history_age_s,
        "roi_size_px": config.roi_size_px,
        "geometry_input_dim": config.geometry_input_dim,
        "roi_feature_dim": config.roi_feature_dim,
        "geometry_feature_dim": config.geometry_feature_dim,
        "hidden_dim": config.hidden_dim,
        "gru_layers": config.gru_layers,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "camera_convention": config.camera_convention,
        "coordinate_convention": config.coordinate_convention,
        "maximum_depth_m": config.maximum_depth_m,
        "minimum_depth_m": config.minimum_depth_m,
        "require_dataset_manifest": config.require_dataset_manifest,
        "expected_yolo_model_sha256": config.expected_yolo_model_sha256,
        "loss_weights": asdict(config.loss_weights),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _new_model(config: TargetStateTrainingConfig, device: torch.device) -> TemporalRayDepthNet:
    return TemporalRayDepthNet(
        geometry_input_dim=config.geometry_input_dim,
        roi_feature_dim=config.roi_feature_dim,
        geometry_feature_dim=config.geometry_feature_dim,
        hidden_dim=config.hidden_dim,
        gru_layers=config.gru_layers,
    ).to(device)


def validate_shard_index_for_training(
    index: TargetStateShardIndex, config: TargetStateTrainingConfig
) -> None:
    if index.history_size != config.history_size:
        raise ShardedTrainingError(
            f"shard index history_size={index.history_size} does not match config={config.history_size}"
        )
    if abs(index.max_history_age_s - config.max_history_age_s) > 1e-12:
        raise ShardedTrainingError(
            "shard index max_history_age_s does not match training config"
        )
    for split in ("train", "validation", "test"):
        entries = index.shards_for_split(split)
        if not entries:
            raise ShardedTrainingError(f"shard index has no {split} shards")
        for entry in entries:
            for episode_id in entry.episode_ids:
                if split_for_episode(episode_id, seed=index.split_seed) != split:
                    raise ShardedTrainingError(
                        f"shard index assigns episode {episode_id!r} to wrong split {split}"
                    )


def _new_progress(
    *, config: TargetStateTrainingConfig, index: TargetStateShardIndex
) -> dict[str, object]:
    return {
        "global_epoch": 1,
        "global_step": 0,
        "phase": "train",
        "current_pc_trans_run_id": f"pending:{config.run_name}",
        "train_shard_order": list(
            deterministic_shard_order(
                index.shards_for_split("train"),
                base_seed=config.seed,
                global_epoch=1,
            )
        ),
        "next_train_shard_index": 0,
        "validation_shard_index": 0,
        "test_shard_index": 0,
        "last_completed_shard": None,
        "last_completed_phase": None,
        "last_completed_run_id": None,
        "last_materialized_path": None,
        "best_validation_loss": float("inf"),
        "best_epoch": None,
        "last_validation_metrics": {},
        "last_test_metrics": {},
        "training_protocol": TRAINING_PROTOCOL,
        "train_loss_sums": {name: 0.0 for name in _LOSS_NAMES},
        "train_batch_count": 0,
        "evaluation_accumulator": None,
        "metric_history": [],
        "per_epoch_run_ids": [],
        "final_test_run_id": None,
        "completed_training_shards": 0,
        "run_id_prefix": None,
        "initial_checkpoint_path": (
            None
            if config.initial_checkpoint_path is None
            else str(config.initial_checkpoint_path)
        ),
    }


def _accumulator_state(
    accumulator: TargetStateEvaluationAccumulator | None,
) -> Mapping[str, object] | None:
    if accumulator is None:
        return None
    state_dict = getattr(accumulator, "state_dict", None)
    if callable(state_dict):
        state = state_dict()
        if not isinstance(state, Mapping):
            raise ShardedTrainingError("evaluation accumulator state_dict must return a mapping")
        return state
    # Compatibility while keeping the checkpoint free of class instances.
    try:
        values = vars(accumulator)
    except TypeError as exc:  # pragma: no cover - guarded by accumulator tests
        raise ShardedTrainingError("evaluation accumulator is not serializable") from exc
    return {
        key: value.cpu() if isinstance(value, Tensor) else value
        for key, value in values.items()
    }


def _restore_accumulator(
    state: object, *, maximum_depth_m: float
) -> TargetStateEvaluationAccumulator:
    if state is None:
        return TargetStateEvaluationAccumulator(maximum_depth_m=maximum_depth_m)
    if isinstance(state, TargetStateEvaluationAccumulator):
        if state.maximum_depth_m != float(maximum_depth_m):
            raise ShardedTrainingError(
                "in-memory evaluation accumulator maximum_depth_m mismatch"
            )
        return state
    if not isinstance(state, Mapping):
        raise ShardedTrainingError("checkpoint evaluation_accumulator must be a mapping or null")
    class_loader = getattr(TargetStateEvaluationAccumulator, "from_state_dict", None)
    if callable(class_loader):
        return class_loader(state, maximum_depth_m=maximum_depth_m)
    accumulator = TargetStateEvaluationAccumulator(maximum_depth_m=maximum_depth_m)
    for name, value in state.items():
        if not hasattr(accumulator, name):
            raise ShardedTrainingError(
                f"checkpoint evaluation accumulator has unknown field: {name}"
            )
        setattr(accumulator, name, value)
    return accumulator


def _checkpoint_payload(
    *,
    model: TemporalRayDepthNet,
    optimizer: AdamW,
    config: TargetStateTrainingConfig,
    index: TargetStateShardIndex,
    model_config: Mapping[str, int],
    progress: Mapping[str, object],
    initial_checkpoint_sha256: str | None,
) -> dict[str, object]:
    payload = {
        "model_type": MODEL_TYPE,
        "schema_version": MODEL_SCHEMA_VERSION,
        "training_stage": config.stage.value,
        "model_config": dict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dataset_sha256": index.parent_dataset_sha256,
        "parent_dataset_sha256": index.parent_dataset_sha256,
        "shard_index_sha256": index.index_sha256,
        "dataset_provenance": _json_safe(index.parent_dataset_provenance),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "checkpoint_role": "latest",
        "resume_protocol": RESUME_PROTOCOL,
        "shard_rng_protocol": SHARD_RNG_PROTOCOL,
        "training_contract_sha256": _training_contract_sha256(config),
        "epoch": int(progress["global_epoch"]),
        "metrics": _json_safe(progress.get("last_validation_metrics", {})),
    }
    payload.update(progress)
    payload["evaluation_accumulator"] = _accumulator_state(
        progress.get("evaluation_accumulator")  # type: ignore[arg-type]
    )
    return payload


def _human_training_state(checkpoint: Mapping[str, object]) -> dict[str, object]:
    return {
        key: _json_safe(checkpoint.get(key))
        for key in (
            "training_protocol",
            "resume_protocol",
            "shard_rng_protocol",
            "checkpoint_role",
            "training_stage",
            "parent_dataset_sha256",
            "shard_index_sha256",
            "training_contract_sha256",
            "global_epoch",
            "global_step",
            "phase",
            "current_pc_trans_run_id",
            "train_shard_order",
            "next_train_shard_index",
            "validation_shard_index",
            "test_shard_index",
            "last_completed_shard",
            "last_completed_phase",
            "last_completed_run_id",
            "best_validation_loss",
            "best_epoch",
            "last_validation_metrics",
            "last_test_metrics",
            "completed_training_shards",
            "per_epoch_run_ids",
            "final_test_run_id",
            "run_id_prefix",
            "initial_checkpoint_path",
            "initial_checkpoint_sha256",
        )
    }


def _commit_checkpoint(
    path: Path,
    payload: Mapping[str, object],
    *,
    training_state_path: Path | None = None,
    expected_last_completed_shard: str | None = None,
) -> Mapping[str, object]:
    _atomic_torch_save(path, payload)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(reloaded, Mapping):
        raise ShardedTrainingError(f"committed checkpoint is not a mapping: {path}")
    _assert_finite_checkpoint_state(reloaded)
    if reloaded.get("training_protocol") != TRAINING_PROTOCOL:
        raise ShardedTrainingError(f"committed checkpoint protocol is invalid: {path}")
    if reloaded.get("last_completed_shard") != expected_last_completed_shard:
        raise ShardedTrainingError(
            "committed checkpoint did not preserve last_completed_shard: "
            f"expected={expected_last_completed_shard!r}, "
            f"actual={reloaded.get('last_completed_shard')!r}"
        )
    if training_state_path is not None:
        _atomic_write_json(training_state_path, _human_training_state(reloaded))
    return reloaded


def _write_metrics_csv(path: Path, history: object) -> None:
    rows = history if isinstance(history, list) else []
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_METRIC_FIELDS)
            writer.writeheader()
            for row in rows:
                if isinstance(row, Mapping):
                    writer.writerow({name: row.get(name) for name in _METRIC_FIELDS})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _checkpoint_integer(
    checkpoint: Mapping[str, object], name: str, *, minimum: int = 0
) -> int:
    value = checkpoint.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ShardedTrainingError(
            f"resume checkpoint {name} must be an integer >= {minimum}"
        )
    return value


def _checkpoint_mapping(
    checkpoint: Mapping[str, object], name: str
) -> Mapping[str, object]:
    value = checkpoint.get(name)
    if not isinstance(value, Mapping):
        raise ShardedTrainingError(f"resume checkpoint {name} must be a mapping")
    return value


def _assert_finite_checkpoint_state(checkpoint: Mapping[str, object]) -> None:
    """Reject a model/optimizer artifact that could poison a safe boundary."""

    def visit(value: object, path: str) -> None:
        if isinstance(value, Tensor):
            if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all().item()
            ):
                raise ShardedTrainingError(
                    f"checkpoint contains a non-finite tensor at {path}"
                )
            return
        if isinstance(value, numbers.Integral):
            return
        if isinstance(value, numbers.Real):
            if not math.isfinite(float(value)):
                raise ShardedTrainingError(
                    f"checkpoint contains a non-finite scalar at {path}"
                )
            return
        if isinstance(value, numbers.Complex):
            if not cmath.isfinite(complex(value)):
                raise ShardedTrainingError(
                    f"checkpoint contains a non-finite scalar at {path}"
                )
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(_checkpoint_mapping(checkpoint, "model_state_dict"), "model_state_dict")
    visit(
        _checkpoint_mapping(checkpoint, "optimizer_state_dict"),
        "optimizer_state_dict",
    )


def _validate_resume_progress(
    checkpoint: Mapping[str, object],
    *,
    config: TargetStateTrainingConfig,
    index: TargetStateShardIndex,
) -> None:
    """Prove that a latest checkpoint represents one exact shard boundary."""

    phase = checkpoint.get("phase")
    if not isinstance(phase, str) or phase not in _PHASES:
        raise ShardedTrainingError(
            f"resume checkpoint phase is invalid: {phase!r}"
        )
    global_epoch = _checkpoint_integer(checkpoint, "global_epoch", minimum=1)
    if global_epoch > config.epochs:
        raise ShardedTrainingError(
            "resume checkpoint global_epoch is outside configured epochs"
        )
    if checkpoint.get("epoch") != global_epoch:
        raise ShardedTrainingError(
            "resume checkpoint epoch alias does not match global_epoch"
        )

    train_entries = index.shards_for_split("train")
    validation_entries = index.shards_for_split("validation")
    test_entries = index.shards_for_split("test")
    entries = _entry_map(index)
    expected_train_order = deterministic_shard_order(
        train_entries, base_seed=config.seed, global_epoch=global_epoch
    )
    raw_order = checkpoint.get("train_shard_order")
    if not isinstance(raw_order, list) or any(
        not isinstance(item, str) for item in raw_order
    ):
        raise ShardedTrainingError(
            "resume checkpoint train_shard_order must be a list of filenames"
        )
    train_order = tuple(raw_order)
    if train_order != expected_train_order:
        raise ShardedTrainingError(
            "resume checkpoint train_shard_order is not the deterministic full order"
        )
    validation_order = tuple(sorted(item.filename for item in validation_entries))
    test_order = tuple(sorted(item.filename for item in test_entries))

    next_train = _checkpoint_integer(checkpoint, "next_train_shard_index")
    validation_index = _checkpoint_integer(checkpoint, "validation_shard_index")
    test_index = _checkpoint_integer(checkpoint, "test_shard_index")
    if next_train > len(train_order):
        raise ShardedTrainingError("next_train_shard_index is outside shard order")
    if validation_index > len(validation_order):
        raise ShardedTrainingError("validation_shard_index is outside shard order")
    if test_index > len(test_order):
        raise ShardedTrainingError("test_shard_index is outside shard order")

    if phase == "train":
        if validation_index != 0 or test_index != 0:
            raise ShardedTrainingError("train phase must have zero validation/test index")
    elif phase == "validation":
        if next_train != len(train_order) or test_index != 0:
            raise ShardedTrainingError(
                "validation phase requires a complete train prefix and zero test index"
            )
    else:
        if (
            global_epoch != config.epochs
            or next_train != len(train_order)
            or validation_index != len(validation_order)
        ):
            raise ShardedTrainingError(
                f"{phase} phase requires final epoch and complete train/validation prefixes"
            )
        if phase == "complete" and test_index != len(test_order):
            raise ShardedTrainingError("complete phase requires the full test prefix")

    batches_per_epoch = sum(
        math.ceil(entry.sequence_count / config.batch_size) for entry in train_entries
    )
    current_train_batches = sum(
        math.ceil(entries[filename].sequence_count / config.batch_size)
        for filename in train_order[:next_train]
    )
    expected_global_step = (global_epoch - 1) * batches_per_epoch + current_train_batches
    global_step = _checkpoint_integer(checkpoint, "global_step")
    if global_step != expected_global_step:
        raise ShardedTrainingError(
            "resume checkpoint global_step does not match completed train shards: "
            f"expected={expected_global_step}, actual={global_step}"
        )
    completed_training_shards = _checkpoint_integer(
        checkpoint, "completed_training_shards"
    )
    expected_completed = (global_epoch - 1) * len(train_order) + next_train
    if completed_training_shards != expected_completed:
        raise ShardedTrainingError(
            "resume checkpoint completed_training_shards is inconsistent: "
            f"expected={expected_completed}, actual={completed_training_shards}"
        )
    train_batch_count = _checkpoint_integer(checkpoint, "train_batch_count")
    if train_batch_count != current_train_batches:
        raise ShardedTrainingError(
            "resume checkpoint train_batch_count does not match the current epoch prefix"
        )
    loss_sums = _checkpoint_mapping(checkpoint, "train_loss_sums")
    if set(loss_sums) != set(_LOSS_NAMES):
        raise ShardedTrainingError("resume checkpoint train_loss_sums fields are invalid")
    for name in _LOSS_NAMES:
        value = loss_sums[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ShardedTrainingError(
                f"resume checkpoint train_loss_sums.{name} must be finite"
            )

    optimizer_state = _checkpoint_mapping(checkpoint, "optimizer_state_dict")
    parameter_groups = optimizer_state.get("param_groups")
    optimizer_slots = optimizer_state.get("state")
    if (
        not isinstance(parameter_groups, list)
        or len(parameter_groups) != 1
        or not isinstance(parameter_groups[0], Mapping)
        or not isinstance(optimizer_slots, Mapping)
    ):
        raise ShardedTrainingError(
            "resume checkpoint optimizer_state_dict must contain one AdamW group"
        )
    group = parameter_groups[0]
    expected_optimizer_values: Mapping[str, object] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    }
    for name, expected in expected_optimizer_values.items():
        actual = group.get(name)
        if name == "betas" and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected:
            raise ShardedTrainingError(
                f"resume checkpoint optimizer {name} mismatch: "
                f"expected={expected!r}, actual={actual!r}"
            )
    parameter_ids = group.get("params")
    if (
        not isinstance(parameter_ids, list)
        or not parameter_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in parameter_ids
        )
        or len(set(parameter_ids)) != len(parameter_ids)
        or any(key not in set(parameter_ids) for key in optimizer_slots)
    ):
        raise ShardedTrainingError(
            "resume checkpoint optimizer parameter identities are invalid"
        )

    prefix = checkpoint.get("run_id_prefix")
    if (
        not isinstance(prefix, str)
        or not prefix
        or prefix in {".", ".."}
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
            for character in prefix
        )
    ):
        raise ShardedTrainingError("resume checkpoint run_id_prefix is invalid")
    epoch_run = f"{prefix}.e{global_epoch:04d}"
    previous_epoch_run = f"{prefix}.e{global_epoch - 1:04d}"
    final_test_run = f"{prefix}.finaltest"
    current_run = checkpoint.get("current_pc_trans_run_id")
    if not isinstance(current_run, str):
        raise ShardedTrainingError(
            "resume checkpoint current_pc_trans_run_id must be a string"
        )
    if phase == "train":
        allowed_current_runs = {epoch_run}
        if next_train == 0:
            allowed_current_runs.add(
                f"pending:{config.run_name}" if global_epoch == 1 else previous_epoch_run
            )
        if current_run not in allowed_current_runs:
            raise ShardedTrainingError(
                "resume checkpoint current run does not match train progress"
            )
    elif phase == "validation":
        if current_run != epoch_run:
            raise ShardedTrainingError(
                "resume checkpoint validation run_id is inconsistent"
            )
    elif phase == "final_test":
        if current_run not in {epoch_run, final_test_run}:
            raise ShardedTrainingError(
                "resume checkpoint final-test run_id is inconsistent"
            )
    elif current_run != final_test_run:
        raise ShardedTrainingError("complete checkpoint must reference final-test run_id")

    expected_last: tuple[str | None, str | None, str | None]
    if phase == "train":
        if next_train:
            expected_last = (train_order[next_train - 1], "train", epoch_run)
        elif global_epoch == 1:
            expected_last = (None, None, None)
        else:
            expected_last = (
                validation_order[-1],
                "validation",
                previous_epoch_run,
            )
    elif phase == "validation":
        expected_last = (
            (validation_order[validation_index - 1] if validation_index else train_order[-1]),
            ("validation" if validation_index else "train"),
            epoch_run,
        )
    elif phase == "final_test":
        expected_last = (
            (test_order[test_index - 1] if test_index else validation_order[-1]),
            ("test" if test_index else "validation"),
            (final_test_run if test_index else epoch_run),
        )
    else:
        expected_last = (test_order[-1], "test", final_test_run)
    actual_last = (
        checkpoint.get("last_completed_shard"),
        checkpoint.get("last_completed_phase"),
        checkpoint.get("last_completed_run_id"),
    )
    if actual_last != expected_last:
        raise ShardedTrainingError(
            "resume checkpoint last-completed identity is inconsistent: "
            f"expected={expected_last!r}, actual={actual_last!r}"
        )
    last_materialized_path = checkpoint.get("last_materialized_path")
    if expected_last[0] is None:
        if last_materialized_path is not None:
            raise ShardedTrainingError(
                "resume checkpoint without a completed shard must not name materialized data"
            )
    else:
        if not isinstance(last_materialized_path, str) or not last_materialized_path:
            raise ShardedTrainingError(
                "resume checkpoint completed shard must name its materialized path"
            )
        materialized_path = Path(last_materialized_path)
        if (
            not materialized_path.is_absolute()
            or materialized_path.name != Path(expected_last[0]).stem
            or materialized_path.parent.name != ".materialized"
            or materialized_path.parent.parent.name != expected_last[2]
        ):
            raise ShardedTrainingError(
                "resume checkpoint last_materialized_path is inconsistent with the completed shard"
            )

    accumulator_state = checkpoint.get("evaluation_accumulator")
    if phase in {"validation", "final_test"}:
        if not isinstance(accumulator_state, Mapping):
            raise ShardedTrainingError(
                f"{phase} resume checkpoint must contain an evaluation accumulator"
            )
        try:
            accumulator = _restore_accumulator(
                accumulator_state, maximum_depth_m=config.maximum_depth_m
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ShardedTrainingError(
                f"resume checkpoint evaluation accumulator is invalid: {exc}"
            ) from exc
        completed_eval_order = (
            validation_order[:validation_index]
            if phase == "validation"
            else test_order[:test_index]
        )
        expected_eval_batches = sum(
            math.ceil(entries[filename].sequence_count / config.batch_size)
            for filename in completed_eval_order
        )
        if accumulator.batch_count != expected_eval_batches:
            raise ShardedTrainingError(
                "resume checkpoint accumulator does not match completed eval shards: "
                f"expected_batches={expected_eval_batches}, "
                f"actual={accumulator.batch_count}"
            )
    elif accumulator_state is not None:
        raise ShardedTrainingError(
            f"{phase} resume checkpoint must not retain an evaluation accumulator"
        )

    last_validation_metrics = _checkpoint_mapping(
        checkpoint, "last_validation_metrics"
    )
    last_test_metrics = _checkpoint_mapping(checkpoint, "last_test_metrics")
    best_epoch = checkpoint.get("best_epoch")
    best_loss = checkpoint.get("best_validation_loss")
    if best_epoch is None:
        if best_loss != float("inf"):
            raise ShardedTrainingError(
                "resume checkpoint without best_epoch must retain infinite best loss"
            )
    else:
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or not 1 <= best_epoch <= global_epoch
            or isinstance(best_loss, bool)
            or not isinstance(best_loss, (int, float))
            or not math.isfinite(float(best_loss))
        ):
            raise ShardedTrainingError("resume checkpoint best validation state is invalid")
    if phase in {"final_test", "complete"} and (
        best_epoch is None or not last_validation_metrics
    ):
        raise ShardedTrainingError(
            f"{phase} resume checkpoint requires completed validation metrics"
        )
    if phase == "complete" and not last_test_metrics:
        raise ShardedTrainingError(
            "complete resume checkpoint requires completed test metrics"
        )

    metric_history = checkpoint.get("metric_history")
    if not isinstance(metric_history, list):
        raise ShardedTrainingError("resume checkpoint metric_history must be a list")
    per_epoch_run_ids = checkpoint.get("per_epoch_run_ids")
    if not isinstance(per_epoch_run_ids, list) or any(
        not isinstance(item, str) for item in per_epoch_run_ids
    ):
        raise ShardedTrainingError("resume checkpoint per_epoch_run_ids must be a string list")
    completed_request_epochs = global_epoch
    if phase == "train" and current_run != epoch_run:
        completed_request_epochs -= 1
    expected_epoch_run_ids = [
        f"{prefix}.e{epoch:04d}" for epoch in range(1, completed_request_epochs + 1)
    ]
    if per_epoch_run_ids != expected_epoch_run_ids:
        raise ShardedTrainingError(
            "resume checkpoint per_epoch_run_ids is inconsistent with progress"
        )
    final_test_run_id = checkpoint.get("final_test_run_id")
    if phase in {"train", "validation"}:
        if final_test_run_id is not None:
            raise ShardedTrainingError(
                "resume checkpoint declares final_test_run_id before final test"
            )
    elif phase == "final_test":
        expected_final_id = final_test_run if current_run == final_test_run else None
        if final_test_run_id != expected_final_id:
            raise ShardedTrainingError(
                "resume checkpoint final_test_run_id is inconsistent"
            )
    elif final_test_run_id != final_test_run:
        raise ShardedTrainingError(
            "complete checkpoint final_test_run_id is inconsistent"
        )

    initial_path = checkpoint.get("initial_checkpoint_path")
    initial_sha = checkpoint.get("initial_checkpoint_sha256")
    if initial_path is not None and not isinstance(initial_path, str):
        raise ShardedTrainingError(
            "resume initial_checkpoint_path must be a string or null"
        )
    if initial_sha is not None and (
        not isinstance(initial_sha, str)
        or len(initial_sha) != 64
        or any(character not in "0123456789abcdef" for character in initial_sha)
    ):
        raise ShardedTrainingError(
            "resume initial_checkpoint_sha256 must be a lowercase SHA256 or null"
        )
    if (initial_path is None) != (initial_sha is None):
        raise ShardedTrainingError(
            "resume initial checkpoint path and SHA256 must be present together"
        )

    _assert_finite_checkpoint_state(checkpoint)


def _load_resume(
    path: Path,
    *,
    config: TargetStateTrainingConfig,
    index: TargetStateShardIndex,
    model_config: Mapping[str, int],
) -> Mapping[str, object]:
    if not path.is_file():
        raise ShardedTrainingError(f"resume checkpoint does not exist: {path}")
    # Evaluation accumulator state is deliberately CPU-only.  Model weights
    # are copied into the already-placed model below, and optimizer.load_state_dict
    # migrates parameter state to its parameter devices.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ShardedTrainingError(
            f"resume checkpoint cannot be decoded safely: {path}: {exc}"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise ShardedTrainingError("resume checkpoint payload must be a mapping")
    expected = {
        "model_type": MODEL_TYPE,
        "schema_version": MODEL_SCHEMA_VERSION,
        "training_stage": config.stage.value,
        "training_protocol": TRAINING_PROTOCOL,
        "resume_protocol": RESUME_PROTOCOL,
        "shard_rng_protocol": SHARD_RNG_PROTOCOL,
        "checkpoint_role": "latest",
        "parent_dataset_sha256": index.parent_dataset_sha256,
        "shard_index_sha256": index.index_sha256,
        "training_contract_sha256": _training_contract_sha256(config),
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ShardedTrainingError(
                f"resume checkpoint {name} mismatch: expected={value!r}, actual={checkpoint.get(name)!r}"
            )
    if checkpoint.get("model_config") != dict(model_config):
        raise ShardedTrainingError("resume checkpoint model architecture is incompatible")
    for required in (
        "model_state_dict",
        "optimizer_state_dict",
        "global_epoch",
        "global_step",
        "phase",
        "train_shard_order",
        "next_train_shard_index",
        "validation_shard_index",
        "test_shard_index",
        "last_completed_shard",
        "best_validation_loss",
        "best_epoch",
        "last_validation_metrics",
        "last_test_metrics",
        "last_completed_phase",
        "last_completed_run_id",
        "last_materialized_path",
        "current_pc_trans_run_id",
        "train_loss_sums",
        "train_batch_count",
        "evaluation_accumulator",
        "metric_history",
        "per_epoch_run_ids",
        "final_test_run_id",
        "completed_training_shards",
        "run_id_prefix",
        "initial_checkpoint_path",
        "initial_checkpoint_sha256",
    ):
        if required not in checkpoint:
            raise ShardedTrainingError(f"resume checkpoint is missing {required}")
    _validate_resume_progress(checkpoint, config=config, index=index)
    return checkpoint


def validate_resume_checkpoint(
    path: str | Path,
    *,
    config: TargetStateTrainingConfig,
    index: TargetStateShardIndex,
) -> Mapping[str, object]:
    """Validate a same-run resume artifact without constructing a model."""

    return _load_resume(
        Path(path).expanduser().resolve(),
        config=config,
        index=index,
        model_config=_model_config(config),
    )


def _progress_from_checkpoint(checkpoint: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "global_epoch",
        "global_step",
        "phase",
        "current_pc_trans_run_id",
        "train_shard_order",
        "next_train_shard_index",
        "validation_shard_index",
        "test_shard_index",
        "last_completed_shard",
        "last_completed_phase",
        "last_completed_run_id",
        "last_materialized_path",
        "best_validation_loss",
        "best_epoch",
        "last_validation_metrics",
        "last_test_metrics",
        "training_protocol",
        "train_loss_sums",
        "train_batch_count",
        "evaluation_accumulator",
        "metric_history",
        "per_epoch_run_ids",
        "final_test_run_id",
        "completed_training_shards",
        "run_id_prefix",
        "initial_checkpoint_path",
    )
    progress = {key: checkpoint.get(key) for key in keys}
    defaults = _new_progress_dummy()
    for key, value in defaults.items():
        if progress.get(key) is None and key not in {
            "last_completed_shard",
            "last_completed_phase",
            "last_completed_run_id",
            "last_materialized_path",
            "best_epoch",
            "evaluation_accumulator",
        }:
            progress[key] = value
    return progress


def _new_progress_dummy() -> dict[str, object]:
    return {
        "current_pc_trans_run_id": "unknown",
        "last_validation_metrics": {},
        "last_test_metrics": {},
        "training_protocol": TRAINING_PROTOCOL,
        "train_loss_sums": {name: 0.0 for name in _LOSS_NAMES},
        "train_batch_count": 0,
        "metric_history": [],
        "per_epoch_run_ids": [],
        "final_test_run_id": None,
        "completed_training_shards": 0,
        "run_id_prefix": None,
    }


def _state_name(payload: Mapping[str, object]) -> str:
    requested = payload.get("requested")
    ready = payload.get("ready")
    active = payload.get("active")
    consumed = payload.get("consumed_record", payload.get("consumed"))
    deleted = payload.get("deleted")
    if not all(type(item) is bool for item in (requested, ready, active, consumed)):
        raise ShardedTrainingError("shard lifecycle facts must be booleans")
    if deleted is not None and type(deleted) is not bool:
        raise ShardedTrainingError("shard lifecycle deleted must be bool or null")
    if ready and active:
        raise ShardedTrainingError("shard lifecycle cannot be ready and active")
    if not consumed and deleted is not None:
        raise ShardedTrainingError(
            "shard lifecycle deleted must be null without a consumed record"
        )
    if consumed and deleted is None:
        raise ShardedTrainingError(
            "shard lifecycle consumed record must declare deleted"
        )
    if consumed and deleted is True and (ready or active):
        raise ShardedTrainingError(
            "deleted consumed shard cannot remain ready or active"
        )
    if consumed and deleted is False and (ready or not active):
        raise ShardedTrainingError(
            "retained consumed shard must be active and not ready"
        )
    if not requested:
        derived = "invalid"
    elif consumed:
        derived = "consumed"
    elif active:
        derived = "active"
    elif ready:
        derived = "ready"
    else:
        derived = "missing"
    declared = payload.get("state")
    if declared is not None and declared != derived:
        raise ShardedTrainingError(
            "derived lifecycle state disagrees with the declared orchestration state"
        )
    return derived


def _active_archive_path(
    *,
    options: ShardedTrainingOptions,
    run_id: str,
    filename: str,
    state: Mapping[str, object],
) -> Path:
    for key in ("active_path", "path", "shard_path"):
        raw = state.get(key)
        if isinstance(raw, str) and raw:
            candidate = Path(raw).expanduser().resolve()
            if candidate.name == filename and candidate.is_file():
                return candidate
    candidate = (
        options.bridge_root / "train_cache" / "active" / run_id / filename
    ).resolve()
    if not candidate.is_file():
        raise ShardedTrainingError(
            f"lifecycle says active but archive does not exist: {candidate}"
        )
    return candidate


def _prepare_active_shard(
    *,
    lifecycle: ShardLifecycle,
    options: ShardedTrainingOptions,
    run_id: str,
    entry: ShardIndexEntry,
) -> Path:
    expected_active = (
        options.bridge_root / "train_cache" / "active" / run_id / entry.filename
    )
    try:
        active_info = expected_active.lstat()
    except FileNotFoundError:
        active_info = None
    except OSError as exc:
        raise ShardedTrainingError(
            f"cannot inspect expected active shard: {expected_active}: {exc}"
        ) from exc
    if active_info is not None:
        if stat.S_ISLNK(active_info.st_mode) or not stat.S_ISREG(active_info.st_mode):
            raise ShardedTrainingError(
                f"expected active shard is not a non-symlink regular file: {expected_active}"
            )
        # Explicit crash-recovery exception: wait-shard only accepts ready, so
        # an already-active exact cache file must be reconciled through state.
        state = lifecycle.shard_state(run_id, entry.filename)
        name = _state_name(state)
    else:
        # Normal lifecycle is exactly wait -> state -> activate.
        lifecycle.wait_shard(run_id, entry.filename, options.wait_timeout_s)
        state = lifecycle.shard_state(run_id, entry.filename)
        name = _state_name(state)
    if name == "ready":
        active = lifecycle.activate(run_id, entry.filename)
    elif name == "active":
        active = _active_archive_path(
            options=options,
            run_id=run_id,
            filename=entry.filename,
            state=state,
        )
    elif name == "consumed":
        raise ShardedTrainingError(
            "pc_trans reports consumed but authoritative checkpoint has not "
            f"completed this shard: run_id={run_id}, shard={entry.filename}"
        )
    elif name in {"missing", "invalid"}:
        raise ShardedTrainingError(
            f"pc_trans shard is not usable after wait: state={name}, "
            f"run_id={run_id}, shard={entry.filename}"
        )
    else:
        raise ShardedTrainingError(
            f"unsupported pc_trans shard state {name!r} for {entry.filename}"
        )
    active = Path(active).expanduser().resolve()
    if active.name != entry.filename or not active.is_file():
        raise ShardedTrainingError(f"active archive path is invalid: {active}")
    return active


def _materialize_uncompleted_shard(
    *,
    materializer: ShardMaterializer,
    archive_path: Path,
    index: TargetStateShardIndex,
) -> MaterializedShard:
    """Clear only this uncommitted shard's interrupted final before retry."""

    materialized_root = archive_path.parent / ".materialized"
    dataset_root = materialized_root / archive_path.stem
    materializer.cleanup_path(dataset_root, materialized_root=materialized_root)
    return materializer.materialize(archive_path, index)


def _release_loader(loader: DataLoader[dict[str, Tensor]] | None) -> None:
    """Ensure non-persistent DataLoader workers have exited before deletion."""

    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    # All loaders here use persistent_workers=False.  Exhausted iterators have
    # already joined their workers; the explicit shutdown above covers failure.


def _dataset_loader(
    *,
    config: TargetStateTrainingConfig,
    dataset_root: Path,
    split: str,
    device: torch.device,
    shuffle: bool,
    generator_seed: int | None,
    split_seed: int,
) -> tuple[TargetStateTorchDataset, DataLoader[dict[str, Tensor]]]:
    # The index owns split assignment.  Training RNG may use a different base
    # seed, so only the shard-local Dataset view receives split_seed.
    shard_config = replace(config, dataset_root=dataset_root, seed=split_seed)
    dataset = TargetStateTorchDataset(shard_config, split=split)
    if len(dataset) == 0:
        raise ShardedTrainingError(
            f"materialized {split} shard contains no temporal sequences: {dataset_root}"
        )
    generator = None
    if generator_seed is not None:
        generator = torch.Generator().manual_seed(generator_seed)
    loader: DataLoader[dict[str, Tensor]] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=False,
    )
    return dataset, loader


def _train_materialized_shard(
    *,
    model: TemporalRayDepthNet,
    optimizer: AdamW,
    config: TargetStateTrainingConfig,
    device: torch.device,
    dataset_root: Path,
    global_epoch: int,
    filename: str,
    split_seed: int,
) -> tuple[dict[str, float], int]:
    dataset: TargetStateTorchDataset | None = None
    loader: DataLoader[dict[str, Tensor]] | None = None
    try:
        # A mid-shard retry starts from the previous checkpoint and receives
        # the same model-side RNG stream (including multi-layer GRU dropout).
        _seed_everything(
            deterministic_shard_execution_seed(
                base_seed=config.seed,
                global_epoch=global_epoch,
                filename=filename,
            )
        )
        dataset, loader = _dataset_loader(
            config=config,
            dataset_root=dataset_root,
            split="train",
            device=device,
            shuffle=True,
            generator_seed=deterministic_batch_seed(
                base_seed=config.seed,
                global_epoch=global_epoch,
                filename=filename,
            ),
            split_seed=split_seed,
        )
        model.train()
        totals = {name: 0.0 for name in _LOSS_NAMES}
        count = 0
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
            losses = compute_target_state_losses(
                output, batch, weights=config.loss_weights
            )
            if not torch.isfinite(losses.total):
                raise ShardedTrainingError(
                    f"non-finite training loss in global epoch {global_epoch}, shard {filename}"
                )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0, error_if_nonfinite=True
            )
            optimizer.step()
            totals["total"] += float(losses.total.detach().cpu())
            totals["depth"] += float(losses.depth_huber.detach().cpu())
            totals["position_3d"] += float(losses.position_3d_huber.detach().cpu())
            totals["reprojection"] += float(losses.reprojection_huber.detach().cpu())
            totals["gaussian_nll"] += float(losses.gaussian_nll.detach().cpu())
            totals["validity_bce"] += float(losses.validity_bce.detach().cpu())
            count += 1
        if count == 0:
            raise ShardedTrainingError(f"training shard produced no batches: {filename}")
        return totals, count
    finally:
        _release_loader(loader)
        del loader
        del dataset


def _evaluate_materialized_shard(
    *,
    model: TemporalRayDepthNet,
    accumulator: TargetStateEvaluationAccumulator,
    config: TargetStateTrainingConfig,
    device: torch.device,
    dataset_root: Path,
    split: str,
    split_seed: int,
) -> TargetStateEvaluationAccumulator:
    dataset: TargetStateTorchDataset | None = None
    loader: DataLoader[dict[str, Tensor]] | None = None
    try:
        dataset, loader = _dataset_loader(
            config=config,
            dataset_root=dataset_root,
            split=split,
            device=device,
            shuffle=False,
            generator_seed=None,
            split_seed=split_seed,
        )
        return accumulate_evaluation(
            model,
            loader,
            device=device,
            maximum_depth_m=config.maximum_depth_m,
            accumulator=accumulator,
        )
    finally:
        _release_loader(loader)
        del loader
        del dataset


def _log_line(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()


def _commit_latest(
    *,
    latest_path: Path,
    state_path: Path,
    model: TemporalRayDepthNet,
    optimizer: AdamW,
    config: TargetStateTrainingConfig,
    index: TargetStateShardIndex,
    model_config: Mapping[str, int],
    progress: Mapping[str, object],
    initial_checkpoint_sha256: str | None,
    hooks: ShardedTrainingHooks,
) -> Mapping[str, object]:
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        index=index,
        model_config=model_config,
        progress=progress,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
    )
    _validate_resume_progress(payload, config=config, index=index)
    committed = _commit_checkpoint(
        latest_path,
        payload,
        training_state_path=state_path,
        expected_last_completed_shard=progress.get("last_completed_shard")  # type: ignore[arg-type]
    )
    _validate_resume_progress(committed, config=config, index=index)
    if hooks.after_checkpoint is not None:
        hooks.after_checkpoint(latest_path, committed)
    return committed


def _consume_after_checkpoint(
    *,
    materializer: ShardMaterializer,
    materialized: MaterializedShard,
    lifecycle: ShardLifecycle,
    run_id: str,
    filename: str,
    completed_phase: str,
    completed_index: int,
    committed_checkpoint: Mapping[str, object],
    hooks: ShardedTrainingHooks,
) -> None:
    # No Dataset/DataLoader object reaches this function.  The verified latest
    # checkpoint has already been committed and reloaded by the caller.
    index_field = {
        "train": "next_train_shard_index",
        "validation": "validation_shard_index",
        "test": "test_shard_index",
    }.get(completed_phase)
    checkpoint_phase = "final_test" if completed_phase == "test" else completed_phase
    if (
        index_field is None
        or committed_checkpoint.get("checkpoint_role") != "latest"
        or committed_checkpoint.get("phase") != checkpoint_phase
        or committed_checkpoint.get("last_completed_phase") != completed_phase
        or committed_checkpoint.get("last_completed_shard") != filename
        or committed_checkpoint.get("last_completed_run_id") != run_id
        or committed_checkpoint.get(index_field) != completed_index
    ):
        raise ShardedTrainingError(
            "committed checkpoint does not authorize shard deletion: "
            f"run_id={run_id}, shard={filename}, phase={completed_phase}, "
            f"index={completed_index}"
        )
    if completed_phase in {"validation", "test"} and not isinstance(
        committed_checkpoint.get("evaluation_accumulator"), Mapping
    ):
        raise ShardedTrainingError(
            "committed eval checkpoint does not contain persisted accumulator state"
        )
    materializer.cleanup(materialized)
    lifecycle.consume(run_id, filename, delete=True)
    if hooks.after_consume is not None:
        hooks.after_consume(run_id, filename)


def _reconcile_last_completed(
    *,
    progress: Mapping[str, object],
    config: TargetStateTrainingConfig,
    index: TargetStateShardIndex,
    lifecycle: ShardLifecycle,
    materializer: ShardMaterializer,
    options: ShardedTrainingOptions,
    latest_path: Path,
    log_path: Path,
) -> None:
    filename = progress.get("last_completed_shard")
    run_id = progress.get("last_completed_run_id")
    if not isinstance(filename, str) or not isinstance(run_id, str):
        return
    # Re-read the authoritative checkpoint before using it to authorize any
    # cleanup/deletion during startup recovery.
    checkpoint = _load_resume(
        latest_path,
        config=config,
        index=index,
        model_config=_model_config(config),
    )
    if checkpoint.get("last_completed_shard") != filename:
        raise ShardedTrainingError("latest checkpoint failed startup progress verification")
    if checkpoint.get("last_completed_run_id") != run_id:
        raise ShardedTrainingError(
            "latest checkpoint run identity changed during startup verification"
        )
    state = lifecycle.shard_state(run_id, filename)
    name = _state_name(state)
    if name == "consumed":
        if state.get("deleted") is False:
            lifecycle.consume(run_id, filename, delete=True)
        return
    if name == "ready":
        # A completed archive may have been recovered active->ready after the
        # checkpoint commit.  Move it back to active solely for safe consume.
        lifecycle.activate(run_id, filename)
        name = "active"
    if name == "active":
        raw_path = progress.get("last_materialized_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ShardedTrainingError(
                "completed active shard has no checkpointed materialized path"
            )
        materialized_path = Path(raw_path).expanduser().resolve()
        expected_root = (
            options.bridge_root
            / "train_cache"
            / "active"
            / run_id
            / ".materialized"
        ).resolve()
        expected_materialized_path = expected_root / Path(filename).stem
        if materialized_path != expected_materialized_path:
            raise ShardedTrainingError(
                "checkpointed materialized path is outside the exact active run/shard"
            )
        materializer.cleanup_path(
            materialized_path, materialized_root=expected_root
        )
        lifecycle.consume(run_id, filename, delete=True)
        _log_line(log_path, f"RECONCILED checkpointed shard run_id={run_id} shard={filename}")
        return
    raise ShardedTrainingError(
        "checkpoint says shard completed but pc_trans cannot prove an active/consumed "
        f"copy: run_id={run_id}, shard={filename}, state={name}"
    )


def _entry_map(index: TargetStateShardIndex) -> dict[str, ShardIndexEntry]:
    return {entry.filename: entry for entry in index.shards}


def _validation_row(
    *,
    progress: Mapping[str, object],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    loss_sums = progress.get("train_loss_sums")
    loss_sums = loss_sums if isinstance(loss_sums, Mapping) else {}
    batch_count = max(int(progress.get("train_batch_count", 0)), 1)
    model_metrics = metrics.get("model")
    model_metrics = model_metrics if isinstance(model_metrics, Mapping) else {}
    return {
        "global_epoch": int(progress["global_epoch"]),
        "global_step": int(progress["global_step"]),
        "train_total": float(loss_sums.get("total", 0.0)) / batch_count,
        "train_depth": float(loss_sums.get("depth", 0.0)) / batch_count,
        "train_position_3d": float(loss_sums.get("position_3d", 0.0)) / batch_count,
        "train_reprojection": float(loss_sums.get("reprojection", 0.0)) / batch_count,
        "train_gaussian_nll": float(loss_sums.get("gaussian_nll", 0.0)) / batch_count,
        "train_validity_bce": float(loss_sums.get("validity_bce", 0.0)) / batch_count,
        "validation_loss": metrics.get("mean_loss"),
        "validation_position_median_error_m": model_metrics.get("position_median_error_m"),
        "validation_position_p95_error_m": model_metrics.get("position_p95_error_m"),
        "validation_measurement_failure_rate": model_metrics.get("measurement_failure_rate"),
        "validation_no_target_false_positive_rate": model_metrics.get("no_target_false_positive_rate"),
        "validation_covariance_error_spearman": model_metrics.get("covariance_error_spearman"),
    }


@contextmanager
def _exclusive_run_lock(run_dir: Path):
    """Hold one non-blocking process lock for checkpoint/consume serialization."""

    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / _RUN_LOCK_FILENAME
    descriptor = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ShardedTrainingError(
                f"another sharded trainer already owns this run: {run_dir}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def train_target_state_sharded(
    config: TargetStateTrainingConfig,
    options: ShardedTrainingOptions,
    *,
    lifecycle: ShardLifecycle | None = None,
    materializer: ShardMaterializer | None = None,
    hooks: ShardedTrainingHooks = ShardedTrainingHooks(),
) -> ShardedTrainingResult:
    """Run under an exclusive run-directory lock."""

    run_dir = config.output_dir / config.run_name
    if options.resume_checkpoint is not None and not run_dir.is_dir():
        raise ShardedTrainingError(
            f"resume output directory does not exist: {run_dir}"
        )
    if options.resume_checkpoint is not None:
        canonical_latest = (run_dir / "latest.pt").resolve()
        if options.resume_checkpoint != canonical_latest:
            raise ShardedTrainingError(
                "--resume-checkpoint must be this run's canonical latest.pt; "
                f"expected={canonical_latest}, actual={options.resume_checkpoint}"
            )
    with _exclusive_run_lock(run_dir):
        return _train_target_state_sharded_locked(
            config,
            options,
            lifecycle=lifecycle,
            materializer=materializer,
            hooks=hooks,
        )


def _train_target_state_sharded_locked(
    config: TargetStateTrainingConfig,
    options: ShardedTrainingOptions,
    *,
    lifecycle: ShardLifecycle | None = None,
    materializer: ShardMaterializer | None = None,
    hooks: ShardedTrainingHooks = ShardedTrainingHooks(),
) -> ShardedTrainingResult:
    """Run continuous model/optimizer training over episode-atomic shards.

    ``latest.pt`` is the sole recovery authority.  The function may be tested
    completely offline by injecting a lifecycle that presents local tar files;
    production uses :class:`PCTransCLI` and never connects from server to PC.
    """

    started = time.monotonic()
    _seed_everything(config.seed)
    device = _device(config.device)
    index = load_shard_index(options.shard_index_path)
    validate_shard_index_for_training(index, config)
    entries = _entry_map(index)
    train_entries = index.shards_for_split("train")
    validation_entries = index.shards_for_split("validation")
    test_entries = index.shards_for_split("test")
    validation_order = tuple(sorted(item.filename for item in validation_entries))
    test_order = tuple(sorted(item.filename for item in test_entries))

    run_dir = config.output_dir / config.run_name
    if options.resume_checkpoint is None:
        non_lock_entries = (
            [item for item in run_dir.iterdir() if item.name != _RUN_LOCK_FILENAME]
            if run_dir.is_dir()
            else []
        )
        if run_dir.exists() and (not run_dir.is_dir() or non_lock_entries):
            raise ShardedTrainingError(
                f"new sharded training output directory must be new or empty: {run_dir}"
            )
    else:
        if not run_dir.is_dir():
            raise ShardedTrainingError(
                f"resume output directory does not exist: {run_dir}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    state_path = run_dir / "training_state.json"
    metrics_path = run_dir / "metrics.csv"
    log_path = run_dir / "terminal.log"
    model_manifest_path = run_dir / "model_manifest.json"

    if lifecycle is None:
        try:
            pc_config_payload = json.loads(
                options.pc_trans_config.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ShardedTrainingError(f"cannot read pc_trans config: {exc}") from exc
        if not isinstance(pc_config_payload, Mapping):
            raise ShardedTrainingError("pc_trans config must be a JSON object")
        configured_bridge = pc_config_payload.get("bridge_root")
        if not isinstance(configured_bridge, str) or (
            Path(configured_bridge).expanduser().resolve() != options.bridge_root
        ):
            raise ShardedTrainingError(
                "--bridge-root must exactly match pc_trans config bridge_root"
            )
        lifecycle = PCTransCLI(
            options.pc_trans_root,
            options.pc_trans_config,
            options.pc_trans_python,
        )
    if materializer is None:
        materializer = DefaultShardMaterializer()

    model_config = _model_config(config)
    model = _new_model(config, device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    if options.resume_checkpoint is not None:
        resume = _load_resume(
            options.resume_checkpoint,
            config=config,
            index=index,
            model_config=model_config,
        )
        try:
            model.load_state_dict(resume["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ShardedTrainingError(
                f"resume model/optimizer state is incompatible: {exc}"
            ) from exc
        progress = _progress_from_checkpoint(resume)
        if progress.get("run_id_prefix") != options.run_id_prefix:
            raise ShardedTrainingError(
                "resume run_id_prefix does not match the authoritative checkpoint"
            )
        initial_checkpoint_sha256 = resume.get("initial_checkpoint_sha256")
        if initial_checkpoint_sha256 is not None and not isinstance(
            initial_checkpoint_sha256, str
        ):
            raise ShardedTrainingError(
                "resume initial_checkpoint_sha256 must be a string or null"
            )
        initial_checkpoint_path = resume.get("initial_checkpoint_path")
        if initial_checkpoint_path is not None and not isinstance(
            initial_checkpoint_path, str
        ):
            raise ShardedTrainingError(
                "resume initial_checkpoint_path must be a string or null"
            )
        # Publish the explicitly selected resume checkpoint as this run's
        # canonical latest before lifecycle reconciliation.
        _commit_checkpoint(
            latest_path,
            resume,
            training_state_path=state_path,
            expected_last_completed_shard=progress.get("last_completed_shard"),  # type: ignore[arg-type]
        )
        _write_metrics_csv(metrics_path, progress.get("metric_history"))
        _reconcile_last_completed(
            progress=progress,
            config=config,
            index=index,
            lifecycle=lifecycle,
            materializer=materializer,
            options=options,
            latest_path=latest_path,
            log_path=log_path,
        )
        _log_line(
            log_path,
            f"RESUME checkpoint={options.resume_checkpoint} phase={progress['phase']} "
            f"global_epoch={progress['global_epoch']} global_step={progress['global_step']}",
        )
    else:
        initial, initial_checkpoint_sha256 = validate_initial_checkpoint(
            config, map_location=device
        )
        initial_checkpoint_path = (
            None
            if config.initial_checkpoint_path is None
            else str(config.initial_checkpoint_path)
        )
        if initial is not None:
            try:
                model.load_state_dict(initial["model_state_dict"], strict=True)
            except (KeyError, RuntimeError) as exc:
                raise ShardedTrainingError(
                    f"initial checkpoint architecture is incompatible: {exc}"
                ) from exc
        progress = _new_progress(config=config, index=index)
        progress["run_id_prefix"] = options.run_id_prefix
        _commit_latest(
            latest_path=latest_path,
            state_path=state_path,
            model=model,
            optimizer=optimizer,
            config=config,
            index=index,
            model_config=model_config,
            progress=progress,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
            hooks=hooks,
        )
        _write_metrics_csv(metrics_path, [])
        _log_line(
            log_path,
            f"START protocol={TRAINING_PROTOCOL} parent_dataset_sha256={index.parent_dataset_sha256} "
            f"shard_index_sha256={index.index_sha256}",
        )

    if int(progress["global_epoch"]) < 1 or int(progress["global_epoch"]) > config.epochs:
        raise ShardedTrainingError("checkpoint global_epoch is outside configured epochs")
    if progress["phase"] == "complete":
        if not best_path.is_file():
            raise ShardedTrainingError(
                "checkpoint is complete but final best.pt is missing"
            )
        if model_manifest_path.is_file():
            manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
            return ShardedTrainingResult(
                run_dir=run_dir,
                best_checkpoint=best_path,
                latest_checkpoint=latest_path,
                model_manifest=model_manifest_path,
                validation_metrics=(
                    manifest.get("validation_metrics", {})
                    if isinstance(manifest, Mapping)
                    else {}
                ),  # type: ignore[arg-type]
                test_metrics=(
                    manifest.get("test_metrics", {})
                    if isinstance(manifest, Mapping)
                    else {}
                ),  # type: ignore[arg-type]
                promoted=bool(
                    manifest.get("promotion", {}).get("passed", False)
                    if isinstance(manifest, Mapping)
                    and isinstance(manifest.get("promotion"), Mapping)
                    else False
                ),
                elapsed_s=time.monotonic() - started,
                global_step=int(progress["global_step"]),
            )
        # A crash after committing phase=complete but before the atomic model
        # manifest publication needs no data replay; rebuild final metadata.

    with ScalarEventWriter(run_dir / "tensorboard") as tensorboard:
        while progress["phase"] != "complete":
            phase = str(progress["phase"])
            global_epoch = int(progress["global_epoch"])

            if phase == "train":
                expected_order = deterministic_shard_order(
                    train_entries,
                    base_seed=config.seed,
                    global_epoch=global_epoch,
                )
                stored_order = tuple(progress["train_shard_order"])  # type: ignore[arg-type]
                if stored_order != expected_order:
                    raise ShardedTrainingError(
                        "checkpoint train_shard_order is not deterministic for this epoch"
                    )
                run_id = f"{options.run_id_prefix}.e{global_epoch:04d}"
                request_files = (*stored_order, *validation_order)
                lifecycle.request(run_id, request_files)
                previous_run_id = progress.get("current_pc_trans_run_id")
                progress["current_pc_trans_run_id"] = run_id
                run_ids = list(progress.get("per_epoch_run_ids", []))
                if run_id not in run_ids:
                    run_ids.append(run_id)
                    progress["per_epoch_run_ids"] = run_ids
                _log_line(log_path, f"RUN_ID={run_id} phase=train global_epoch={global_epoch}")
                if previous_run_id != run_id:
                    _commit_latest(
                        latest_path=latest_path,
                        state_path=state_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        index=index,
                        model_config=model_config,
                        progress=progress,
                        initial_checkpoint_sha256=initial_checkpoint_sha256,
                        hooks=hooks,
                    )

                start_index = int(progress["next_train_shard_index"])
                if not 0 <= start_index <= len(stored_order):
                    raise ShardedTrainingError("next_train_shard_index is outside shard order")
                for shard_index in range(start_index, len(stored_order)):
                    filename = stored_order[shard_index]
                    entry = entries[filename]
                    if entry.split != "train":
                        raise ShardedTrainingError(f"train order contains {entry.split} shard")
                    materialized: MaterializedShard | None = None
                    try:
                        active = _prepare_active_shard(
                            lifecycle=lifecycle,
                            options=options,
                            run_id=run_id,
                            entry=entry,
                        )
                        materialized = _materialize_uncompleted_shard(
                            materializer=materializer,
                            archive_path=active,
                            index=index,
                        )
                        totals, batches = _train_materialized_shard(
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            device=device,
                            dataset_root=materialized.dataset_root,
                            global_epoch=global_epoch,
                            filename=filename,
                            split_seed=index.split_seed,
                        )
                        loss_sums = dict(progress["train_loss_sums"])  # type: ignore[arg-type]
                        for name in _LOSS_NAMES:
                            loss_sums[name] = float(loss_sums.get(name, 0.0)) + totals[name]
                        progress["train_loss_sums"] = loss_sums
                        progress["train_batch_count"] = int(progress["train_batch_count"]) + batches
                        progress["global_step"] = int(progress["global_step"]) + batches
                        progress["next_train_shard_index"] = shard_index + 1
                        progress["last_completed_shard"] = filename
                        progress["last_completed_phase"] = "train"
                        progress["last_completed_run_id"] = run_id
                        progress["last_materialized_path"] = str(materialized.dataset_root)
                        progress["completed_training_shards"] = int(
                            progress["completed_training_shards"]
                        ) + 1
                        committed = _commit_latest(
                            latest_path=latest_path,
                            state_path=state_path,
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            index=index,
                            model_config=model_config,
                            progress=progress,
                            initial_checkpoint_sha256=initial_checkpoint_sha256,
                            hooks=hooks,
                        )
                        _consume_after_checkpoint(
                            materializer=materializer,
                            materialized=materialized,
                            lifecycle=lifecycle,
                            run_id=run_id,
                            filename=filename,
                            completed_phase="train",
                            completed_index=shard_index + 1,
                            committed_checkpoint=committed,
                            hooks=hooks,
                        )
                        materialized = None
                        _log_line(
                            log_path,
                            f"COMPLETED_SHARD phase=train epoch={global_epoch} "
                            f"index={shard_index + 1}/{len(stored_order)} shard={filename} "
                            f"global_step={progress['global_step']}",
                        )
                    except Exception:
                        if materialized is not None:
                            try:
                                materializer.cleanup(materialized)
                            except Exception as cleanup_error:
                                _log_line(log_path, f"MATERIALIZED_CLEANUP_FAILED {cleanup_error}")
                        _log_line(log_path, f"FAILED_SHARD={filename}")
                        _log_line(log_path, f"RUN_ID={run_id}")
                        _log_line(log_path, f"SHARD={filename}")
                        _log_line(log_path, f"LATEST_SAFE_CHECKPOINT={latest_path}")
                        raise

                progress["phase"] = "validation"
                progress["validation_shard_index"] = 0
                progress["evaluation_accumulator"] = TargetStateEvaluationAccumulator(
                    config.maximum_depth_m
                )
                _commit_latest(
                    latest_path=latest_path,
                    state_path=state_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    index=index,
                    model_config=model_config,
                    progress=progress,
                    initial_checkpoint_sha256=initial_checkpoint_sha256,
                    hooks=hooks,
                )
                phase = "validation"

            if phase == "validation":
                run_id = f"{options.run_id_prefix}.e{global_epoch:04d}"
                train_order = tuple(progress["train_shard_order"])  # type: ignore[arg-type]
                lifecycle.request(run_id, (*train_order, *validation_order))
                progress["current_pc_trans_run_id"] = run_id
                accumulator = _restore_accumulator(
                    progress.get("evaluation_accumulator"),
                    maximum_depth_m=config.maximum_depth_m,
                )
                progress["evaluation_accumulator"] = accumulator
                start_index = int(progress["validation_shard_index"])
                if not 0 <= start_index <= len(validation_order):
                    raise ShardedTrainingError("validation_shard_index is outside shard order")
                _log_line(log_path, f"RUN_ID={run_id} phase=validation global_epoch={global_epoch}")
                for shard_index in range(start_index, len(validation_order)):
                    filename = validation_order[shard_index]
                    entry = entries[filename]
                    materialized = None
                    try:
                        active = _prepare_active_shard(
                            lifecycle=lifecycle,
                            options=options,
                            run_id=run_id,
                            entry=entry,
                        )
                        materialized = _materialize_uncompleted_shard(
                            materializer=materializer,
                            archive_path=active,
                            index=index,
                        )
                        accumulator = _evaluate_materialized_shard(
                            model=model,
                            accumulator=accumulator,
                            config=config,
                            device=device,
                            dataset_root=materialized.dataset_root,
                            split="validation",
                            split_seed=index.split_seed,
                        )
                        progress["evaluation_accumulator"] = accumulator
                        progress["validation_shard_index"] = shard_index + 1
                        progress["last_completed_shard"] = filename
                        progress["last_completed_phase"] = "validation"
                        progress["last_completed_run_id"] = run_id
                        progress["last_materialized_path"] = str(materialized.dataset_root)
                        committed = _commit_latest(
                            latest_path=latest_path,
                            state_path=state_path,
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            index=index,
                            model_config=model_config,
                            progress=progress,
                            initial_checkpoint_sha256=initial_checkpoint_sha256,
                            hooks=hooks,
                        )
                        _consume_after_checkpoint(
                            materializer=materializer,
                            materialized=materialized,
                            lifecycle=lifecycle,
                            run_id=run_id,
                            filename=filename,
                            completed_phase="validation",
                            completed_index=shard_index + 1,
                            committed_checkpoint=committed,
                            hooks=hooks,
                        )
                        materialized = None
                        _log_line(
                            log_path,
                            f"COMPLETED_SHARD phase=validation epoch={global_epoch} "
                            f"index={shard_index + 1}/{len(validation_order)} shard={filename}",
                        )
                    except Exception:
                        if materialized is not None:
                            try:
                                materializer.cleanup(materialized)
                            except Exception as cleanup_error:
                                _log_line(log_path, f"MATERIALIZED_CLEANUP_FAILED {cleanup_error}")
                        _log_line(log_path, f"FAILED_SHARD={filename}")
                        _log_line(log_path, f"RUN_ID={run_id}")
                        _log_line(log_path, f"SHARD={filename}")
                        _log_line(log_path, f"LATEST_SAFE_CHECKPOINT={latest_path}")
                        raise

                validation_metrics = accumulator.finalize()
                progress["last_validation_metrics"] = validation_metrics
                progress["evaluation_accumulator"] = None
                validation_loss = float(validation_metrics["mean_loss"])
                if not math.isfinite(validation_loss):
                    raise ShardedTrainingError("global validation loss is not finite")
                row = _validation_row(progress=progress, metrics=validation_metrics)
                history = list(progress.get("metric_history", []))
                history = [
                    item
                    for item in history
                    if not (
                        isinstance(item, Mapping)
                        and int(item.get("global_epoch", -1)) == global_epoch
                    )
                ]
                history.append(row)
                progress["metric_history"] = history

                if validation_loss < float(progress["best_validation_loss"]):
                    progress["best_validation_loss"] = validation_loss
                    progress["best_epoch"] = global_epoch
                    best_payload = _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        index=index,
                        model_config=model_config,
                        progress=progress,
                        initial_checkpoint_sha256=initial_checkpoint_sha256,
                    )
                    best_payload["checkpoint_role"] = "best"
                    _commit_checkpoint(
                        best_path,
                        best_payload,
                        expected_last_completed_shard=progress.get("last_completed_shard"),  # type: ignore[arg-type]
                    )
                    _log_line(
                        log_path,
                        f"BEST_COMMITTED global_epoch={global_epoch} validation_loss={validation_loss}",
                    )
                _write_metrics_csv(metrics_path, history)
                for name in _LOSS_NAMES:
                    value = row.get(f"train_{name}")
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        tensorboard.add_scalar(f"train/{name}", float(value), global_epoch)
                tensorboard.add_scalar("validation/loss", validation_loss, global_epoch)
                model_metrics = validation_metrics.get("model")
                if isinstance(model_metrics, Mapping):
                    for name in (
                        "position_median_error_m",
                        "position_p95_error_m",
                        "measurement_failure_rate",
                        "no_target_false_positive_rate",
                        "covariance_error_spearman",
                    ):
                        value = model_metrics.get(name)
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            tensorboard.add_scalar(
                                f"validation/{name}", float(value), global_epoch
                            )

                if global_epoch < config.epochs:
                    next_epoch = global_epoch + 1
                    progress["global_epoch"] = next_epoch
                    progress["phase"] = "train"
                    progress["train_shard_order"] = list(
                        deterministic_shard_order(
                            train_entries,
                            base_seed=config.seed,
                            global_epoch=next_epoch,
                        )
                    )
                    progress["next_train_shard_index"] = 0
                    progress["validation_shard_index"] = 0
                    progress["train_loss_sums"] = {name: 0.0 for name in _LOSS_NAMES}
                    progress["train_batch_count"] = 0
                else:
                    if not best_path.is_file():
                        raise ShardedTrainingError("full validation never produced best.pt")
                    best_checkpoint = torch.load(
                        best_path, map_location=device, weights_only=False
                    )
                    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
                    progress["phase"] = "final_test"
                    progress["test_shard_index"] = 0
                    progress["evaluation_accumulator"] = TargetStateEvaluationAccumulator(
                        config.maximum_depth_m
                    )
                _commit_latest(
                    latest_path=latest_path,
                    state_path=state_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    index=index,
                    model_config=model_config,
                    progress=progress,
                    initial_checkpoint_sha256=initial_checkpoint_sha256,
                    hooks=hooks,
                )
                continue

            if phase == "final_test":
                run_id = f"{options.run_id_prefix}.finaltest"
                lifecycle.request(run_id, test_order)
                previous_run_id = progress.get("current_pc_trans_run_id")
                progress["current_pc_trans_run_id"] = run_id
                progress["final_test_run_id"] = run_id
                accumulator = _restore_accumulator(
                    progress.get("evaluation_accumulator"),
                    maximum_depth_m=config.maximum_depth_m,
                )
                progress["evaluation_accumulator"] = accumulator
                if previous_run_id != run_id:
                    _commit_latest(
                        latest_path=latest_path,
                        state_path=state_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        index=index,
                        model_config=model_config,
                        progress=progress,
                        initial_checkpoint_sha256=initial_checkpoint_sha256,
                        hooks=hooks,
                    )
                start_index = int(progress["test_shard_index"])
                if not 0 <= start_index <= len(test_order):
                    raise ShardedTrainingError("test_shard_index is outside shard order")
                _log_line(log_path, f"RUN_ID={run_id} phase=final_test best_epoch={progress['best_epoch']}")
                for shard_index in range(start_index, len(test_order)):
                    filename = test_order[shard_index]
                    entry = entries[filename]
                    materialized = None
                    try:
                        active = _prepare_active_shard(
                            lifecycle=lifecycle,
                            options=options,
                            run_id=run_id,
                            entry=entry,
                        )
                        materialized = _materialize_uncompleted_shard(
                            materializer=materializer,
                            archive_path=active,
                            index=index,
                        )
                        accumulator = _evaluate_materialized_shard(
                            model=model,
                            accumulator=accumulator,
                            config=config,
                            device=device,
                            dataset_root=materialized.dataset_root,
                            split="test",
                            split_seed=index.split_seed,
                        )
                        progress["evaluation_accumulator"] = accumulator
                        progress["test_shard_index"] = shard_index + 1
                        progress["last_completed_shard"] = filename
                        progress["last_completed_phase"] = "test"
                        progress["last_completed_run_id"] = run_id
                        progress["last_materialized_path"] = str(materialized.dataset_root)
                        committed = _commit_latest(
                            latest_path=latest_path,
                            state_path=state_path,
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            index=index,
                            model_config=model_config,
                            progress=progress,
                            initial_checkpoint_sha256=initial_checkpoint_sha256,
                            hooks=hooks,
                        )
                        _consume_after_checkpoint(
                            materializer=materializer,
                            materialized=materialized,
                            lifecycle=lifecycle,
                            run_id=run_id,
                            filename=filename,
                            completed_phase="test",
                            completed_index=shard_index + 1,
                            committed_checkpoint=committed,
                            hooks=hooks,
                        )
                        materialized = None
                        _log_line(
                            log_path,
                            f"COMPLETED_SHARD phase=test index={shard_index + 1}/{len(test_order)} "
                            f"shard={filename}",
                        )
                    except Exception:
                        if materialized is not None:
                            try:
                                materializer.cleanup(materialized)
                            except Exception as cleanup_error:
                                _log_line(log_path, f"MATERIALIZED_CLEANUP_FAILED {cleanup_error}")
                        _log_line(log_path, f"FAILED_SHARD={filename}")
                        _log_line(log_path, f"RUN_ID={run_id}")
                        _log_line(log_path, f"SHARD={filename}")
                        _log_line(log_path, f"LATEST_SAFE_CHECKPOINT={latest_path}")
                        raise
                progress["last_test_metrics"] = accumulator.finalize()
                progress["evaluation_accumulator"] = None
                progress["phase"] = "complete"
                _commit_latest(
                    latest_path=latest_path,
                    state_path=state_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    index=index,
                    model_config=model_config,
                    progress=progress,
                    initial_checkpoint_sha256=initial_checkpoint_sha256,
                    hooks=hooks,
                )
                continue

            if phase not in _PHASES:
                raise ShardedTrainingError(f"unknown sharded training phase: {phase}")

    best_for_metrics = torch.load(best_path, map_location="cpu", weights_only=False)
    if not isinstance(best_for_metrics, Mapping):
        raise ShardedTrainingError("best.pt payload must be a mapping")
    validation_metrics = best_for_metrics.get("last_validation_metrics")
    test_metrics = progress["last_test_metrics"]
    if not isinstance(validation_metrics, Mapping) or not isinstance(test_metrics, Mapping):
        raise ShardedTrainingError("final validation/test metrics are missing")
    validation_gate = evaluate_promotion(
        validation_metrics,
        p95_max_ratio=config.promotion_p95_max_ratio,
        minimum_covariance_correlation=config.promotion_min_covariance_correlation,
    )
    test_gate = evaluate_promotion(
        test_metrics,
        p95_max_ratio=config.promotion_p95_max_ratio,
        minimum_covariance_correlation=config.promotion_min_covariance_correlation,
    )
    stage_ok = config.stage is TrainingStage.YOLO_DEPLOYMENT
    stage_a_initialized = initial_checkpoint_sha256 is not None
    promoted = (
        stage_ok
        and stage_a_initialized
        and config.require_dataset_manifest
        and bool(validation_gate["passed"])
        and bool(test_gate["passed"])
    )
    shard_counts = {
        split: len(index.shards_for_split(split))
        for split in ("train", "validation", "test")
    }
    episode_counts = {
        split: sum(entry.episode_count for entry in index.shards_for_split(split))
        for split in ("train", "validation", "test")
    }
    manifest = {
        "model_type": MODEL_TYPE,
        "schema_version": MODEL_SCHEMA_VERSION,
        "training_stage": config.stage.value,
        "training_protocol": TRAINING_PROTOCOL,
        "resume_protocol": RESUME_PROTOCOL,
        "checkpoint_path": str(best_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "dataset_sha256": index.parent_dataset_sha256,
        "parent_dataset_sha256": index.parent_dataset_sha256,
        "shard_index_sha256": index.index_sha256,
        "dataset_provenance": _json_safe(index.parent_dataset_provenance),
        "global_epochs": config.epochs,
        "global_step": int(progress["global_step"]),
        "shard_counts": shard_counts,
        "episode_counts": episode_counts,
        "completed_training_shards": int(progress["completed_training_shards"]),
        "pc_trans": {
            "used": True,
            "per_epoch_run_ids": list(progress.get("per_epoch_run_ids", [])),
            "final_test_run_id": progress.get("final_test_run_id"),
        },
        "artifacts": {
            "best_checkpoint": str(best_path.resolve()),
            "latest_checkpoint": str(latest_path.resolve()),
            "training_state": str(state_path.resolve()),
            "metrics_csv": str(metrics_path.resolve()),
            "terminal_log": str(log_path.resolve()),
            "tensorboard_dir": str((run_dir / "tensorboard").resolve()),
            "videos_saved": False,
        },
        "input_fields": {
            "roi_rgbd": ["red", "green", "blue", "normalized_depth"],
            "geometry_25d": list(GEOMETRY_INPUT_FIELDS),
            "missing_mask": True,
        },
        "output_fields": list(OUTPUT_FIELDS),
        "model_config": model_config,
        "history_size": config.history_size,
        "max_history_age_s": config.max_history_age_s,
        "camera_convention": config.camera_convention,
        "coordinate_convention": config.coordinate_convention,
        "initial_checkpoint": (
            None
            if initial_checkpoint_path is None
            else {
                "path": initial_checkpoint_path,
                "sha256": initial_checkpoint_sha256,
            }
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "promotion": {
            "passed": promoted,
            "requires_yolo_deployment_stage": True,
            "stage_satisfied": stage_ok,
            "requires_stage_a_initialization": True,
            "stage_a_initialization_satisfied": stage_a_initialized,
            "requires_verified_dataset_manifest": True,
            "dataset_manifest_satisfied": config.require_dataset_manifest,
            "validation": validation_gate,
            "test": test_gate,
        },
        "config": _json_safe(asdict(config)),
        "training_commit_sha": os.environ.get("UAV_AGENT_TRAINING_COMMIT_SHA", "nogit"),
        "torch_version": torch.__version__,
    }
    _atomic_write_json(model_manifest_path, manifest)
    _write_figures(run_dir, test_metrics, config.save_figures)
    _log_line(
        log_path,
        f"COMPLETE global_epochs={config.epochs} global_step={progress['global_step']} "
        f"best_epoch={progress['best_epoch']} promoted={str(promoted).lower()}",
    )
    return ShardedTrainingResult(
        run_dir=run_dir,
        best_checkpoint=best_path,
        latest_checkpoint=latest_path,
        model_manifest=model_manifest_path,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        promoted=promoted,
        elapsed_s=time.monotonic() - started,
        global_step=int(progress["global_step"]),
    )


__all__ = [
    "DefaultShardMaterializer",
    "PCTransCLI",
    "RESUME_PROTOCOL",
    "ShardLifecycle",
    "ShardMaterializer",
    "ShardedTrainingError",
    "ShardedTrainingHooks",
    "ShardedTrainingOptions",
    "ShardedTrainingResult",
    "TRAINING_PROTOCOL",
    "deterministic_batch_seed",
    "deterministic_shard_order",
    "train_target_state_sharded",
    "validate_shard_index_for_training",
    "validate_resume_checkpoint",
]
