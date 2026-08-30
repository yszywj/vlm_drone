"""Transformers Trainer integration that persists PEFT adapters, never base weights.

The Qwen dependencies intentionally remain lazy imports.  Importing this module in
the Isaac/Python test environment must not require ``transformers`` or ``peft``;
the active training path gives an actionable error when the dedicated
``qwen_lora`` environment is not active.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping


class LoraTrainerError(RuntimeError):
    """Raised when adapter-only Trainer invariants cannot be guaranteed."""


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    """Filesystem layout for one immutable LoRA training run."""

    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    tensorboard_dir: Path
    final_adapter_dir: Path


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_BASE_WEIGHT_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "tf_model.h5",
    "flax_model.msgpack",
}
_BASE_WEIGHT_SHARD_RE = re.compile(
    r"^(?:model-\d{5}-of-\d{5}\.safetensors|pytorch_model-\d{5}-of-\d{5}\.bin)$"
)


def validate_run_id(run_id: str) -> str:
    """Return a safe single-component run id or fail closed."""

    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise LoraTrainerError(
            "run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}"
        )
    if run_id in {".", ".."}:
        raise LoraTrainerError("run_id cannot be a traversal component")
    return run_id


def build_training_paths(config: object, run_id: str) -> TrainingPaths:
    """Build, but do not create, the run and deployment directories."""

    safe_id = validate_run_id(run_id)
    output_root = Path(getattr(config, "output_dir")).expanduser().resolve()
    adapter_root_raw = getattr(config, "adapter_output_dir", None)
    if adapter_root_raw is None:
        raise LoraTrainerError("active config requires adapter_output_dir")
    adapter_root = Path(adapter_root_raw).expanduser().resolve()
    run_dir = output_root / safe_id
    final_adapter = adapter_root / safe_id
    return TrainingPaths(
        run_id=safe_id,
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        metrics_dir=run_dir / "metrics",
        tensorboard_dir=run_dir / "tensorboard",
        final_adapter_dir=final_adapter,
    )


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on the dedicated env
        raise LoraTrainerError(
            "transformers is required for active LoRA training; activate the "
            "dedicated qwen_lora environment"
        ) from exc
    return transformers


def _accepted_keyword(callable_object: object, preferred: str, legacy: str) -> str:
    """Choose a Transformers keyword across adjacent supported releases."""

    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return preferred
    if preferred in parameters:
        return preferred
    if legacy in parameters:
        return legacy
    return preferred


def build_training_arguments(
    config: object,
    paths: TrainingPaths,
    *,
    arguments_class: type[Any] | None = None,
    has_validation: bool = True,
) -> Any:
    """Translate the reviewed config into ``TrainingArguments``.

    Keeping this mapping here prevents training hyperparameters from being
    scattered through the entrypoint.  An injectable class makes the CPU unit
    tests independent of a Transformers installation.
    """

    if arguments_class is None:
        arguments_class = _import_transformers().TrainingArguments
    eval_key = _accepted_keyword(arguments_class.__init__, "eval_strategy", "evaluation_strategy")
    max_steps = getattr(config, "max_steps")
    kwargs: dict[str, object] = {
        "output_dir": str(paths.checkpoints_dir),
        "logging_dir": str(paths.tensorboard_dir),
        "num_train_epochs": float(getattr(config, "num_train_epochs")),
        "max_steps": -1 if max_steps is None else int(max_steps),
        "per_device_train_batch_size": int(getattr(config, "per_device_train_batch_size")),
        "per_device_eval_batch_size": int(getattr(config, "per_device_eval_batch_size")),
        "gradient_accumulation_steps": int(getattr(config, "gradient_accumulation_steps")),
        "learning_rate": float(getattr(config, "learning_rate")),
        "weight_decay": float(getattr(config, "weight_decay")),
        "warmup_ratio": float(getattr(config, "warmup_ratio")),
        "lr_scheduler_type": str(getattr(config, "lr_scheduler_type")),
        "max_grad_norm": float(getattr(config, "max_grad_norm")),
        "bf16": bool(getattr(config, "bf16")),
        "gradient_checkpointing": bool(getattr(config, "gradient_checkpointing")),
        "logging_strategy": "steps",
        "logging_steps": int(getattr(config, "logging_steps")),
        eval_key: "steps" if has_validation else "no",
        "eval_steps": int(getattr(config, "eval_steps")),
        "save_strategy": "steps",
        "save_steps": int(getattr(config, "save_steps")),
        "save_total_limit": int(getattr(config, "save_total_limit")),
        "dataloader_num_workers": int(getattr(config, "dataloader_num_workers")),
        "seed": int(getattr(config, "seed")),
        "data_seed": int(getattr(config, "seed")),
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
        "save_safetensors": True,
        "ddp_find_unused_parameters": False,
    }
    return arguments_class(**kwargs)


def _unwrap_adapter_model(model: object) -> object:
    """Unwrap common DDP containers without importing torch/accelerate."""

    current = model
    seen: set[int] = set()
    while hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        current = getattr(current, "module")
    return current


def _require_peft_adapter_model(model: object) -> object:
    unwrapped = _unwrap_adapter_model(model)
    peft_config = getattr(unwrapped, "peft_config", None)
    if not isinstance(peft_config, Mapping) or not peft_config:
        raise LoraTrainerError(
            "refusing to save a checkpoint from a non-PEFT model; this could write "
            "the complete base model"
        )
    if not callable(getattr(unwrapped, "save_pretrained", None)):
        raise LoraTrainerError("PEFT adapter model does not implement save_pretrained")
    return unwrapped


def assert_no_base_model_weights(directory: str | Path) -> None:
    """Fail if a checkpoint/final directory contains recognized base weights."""

    root = Path(directory)
    if not root.is_dir():
        raise LoraTrainerError(f"adapter output directory does not exist: {root}")
    offenders = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (path.name in _BASE_WEIGHT_NAMES or _BASE_WEIGHT_SHARD_RE.fullmatch(path.name))
    )
    if offenders:
        raise LoraTrainerError(
            "base-model weight files are forbidden in LoRA outputs: " + ", ".join(offenders)
        )


def require_adapter_artifacts(directory: str | Path) -> tuple[Path, tuple[Path, ...]]:
    """Require the minimum deployable PEFT adapter artifacts."""

    root = Path(directory)
    assert_no_base_model_weights(root)
    config_path = root / "adapter_config.json"
    weights = tuple(sorted(root.glob("adapter_model*.safetensors")))
    if not config_path.is_file() or config_path.is_symlink():
        raise LoraTrainerError("adapter_config.json was not produced as a regular file")
    if not weights or any(path.is_symlink() or not path.is_file() or path.stat().st_size == 0 for path in weights):
        raise LoraTrainerError("non-empty adapter_model*.safetensors output is required")
    return config_path, weights


def get_adapter_only_trainer_class(trainer_base: type[Any] | None = None) -> type[Any]:
    """Create a minimal Trainer subclass whose checkpoints are adapter-only."""

    if trainer_base is None:
        trainer_base = _import_transformers().Trainer

    class AdapterOnlyTrainer(trainer_base):  # type: ignore[misc, valid-type]
        """Delegate optimization/DDP to Transformers while constraining saves."""

        def _save(self, output_dir: str | None = None, state_dict: object | None = None) -> None:
            # Intentionally ignore ``state_dict``: Trainer may provide a full base
            # state dict.  PeftModel.save_pretrained performs its own adapter-only
            # filtering and is the sole authorized model writer here.
            del state_dict
            destination = Path(output_dir or self.args.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            adapter_model = _require_peft_adapter_model(self.model)
            adapter_model.save_pretrained(str(destination), safe_serialization=True)
            require_adapter_artifacts(destination)

    AdapterOnlyTrainer.__name__ = "AdapterOnlyTrainer"
    AdapterOnlyTrainer.__qualname__ = "AdapterOnlyTrainer"
    return AdapterOnlyTrainer


def build_trainer(
    *,
    model: object,
    processor: object,
    config: object,
    paths: TrainingPaths,
    train_dataset: object,
    validation_dataset: object | None,
    data_collator: Callable[..., object],
    trainer_class: type[Any] | None = None,
    arguments_class: type[Any] | None = None,
) -> Any:
    """Construct a real Transformers Trainer without custom optimizer/DDP code."""

    if trainer_class is None:
        trainer_class = get_adapter_only_trainer_class()
    args = build_training_arguments(
        config,
        paths,
        arguments_class=arguments_class,
        has_validation=validation_dataset is not None and len(validation_dataset) > 0,  # type: ignore[arg-type]
    )
    init_kwargs: dict[str, object] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": data_collator,
    }
    processor_key = _accepted_keyword(
        trainer_class.__init__, "processing_class", "tokenizer"
    )
    init_kwargs[processor_key] = processor
    return trainer_class(**init_kwargs)


def save_final_adapter(model: object, destination: str | Path) -> Path:
    """Atomically publish a final adapter while refusing overwrite/base weights."""

    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise LoraTrainerError(f"final adapter output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    adapter_model = _require_peft_adapter_model(model)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        adapter_model.save_pretrained(str(temporary), safe_serialization=True)
        require_adapter_artifacts(temporary)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    require_adapter_artifacts(target)
    return target


def write_metrics(directory: str | Path, name: str, metrics: Mapping[str, object]) -> Path:
    """Write finite JSON metrics atomically under the run metrics directory."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise LoraTrainerError("metric name must be lowercase snake_case")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    payload = json.dumps(
        dict(metrics), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    return path


__all__ = [
    "LoraTrainerError",
    "TrainingPaths",
    "assert_no_base_model_weights",
    "build_trainer",
    "build_training_arguments",
    "build_training_paths",
    "get_adapter_only_trainer_class",
    "require_adapter_artifacts",
    "save_final_adapter",
    "validate_run_id",
    "write_metrics",
]
