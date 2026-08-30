#!/usr/bin/env python3
"""Validate or train the Qwen3-VL Fleet Planner PEFT LoRA adapter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fleet_data.validator import validate_dataset  # noqa: E402
from training.lora.config import (  # noqa: E402
    LoraScaffoldConfig,
    LoraScaffoldError,
    load_lora_config,
)
from training.lora.export_adapter_manifest import build_manifest  # noqa: E402
from training.lora.trainer import (  # noqa: E402
    LoraTrainerError,
    TrainingPaths,
    build_trainer,
    build_training_paths,
    require_adapter_artifacts,
    save_final_adapter,
    validate_run_id,
    write_metrics,
)


DEFAULT_CONFIG = _ROOT / "configs/lora/fleet_planner_lora.json"


@dataclass(frozen=True, slots=True)
class ActiveTrainingComponents:
    """Injectable active-path components for dependency-free unit tests."""

    dataset_class: type[Any]
    collator_class: type[Any]
    model_loader: Callable[..., object]
    model_validator: Callable[[str | Path], Mapping[str, object]] | None = None
    trainer_builder: Callable[..., object] = build_trainer
    final_adapter_saver: Callable[[object, str | Path], Path] = save_final_adapter


def _active_components() -> ActiveTrainingComponents:
    try:
        from training.lora.collator import AssistantOnlyDataCollator
        from training.lora.dataset import FleetPlannerSFTDataset
        from training.lora.modeling import (
            load_qwen_lora_model,
            validate_local_model_directory,
        )
    except ImportError as exc:
        raise LoraTrainerError(
            "active LoRA training dependencies are unavailable; activate qwen_lora"
        ) from exc
    return ActiveTrainingComponents(
        dataset_class=FleetPlannerSFTDataset,
        collator_class=AssistantOnlyDataCollator,
        model_loader=load_qwen_lora_model,
        model_validator=validate_local_model_directory,
    )


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise LoraTrainerError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LoraTrainerError(f"invalid {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LoraTrainerError(f"{description} must be a JSON object: {path}")
    return payload


def _value(source: object, *names: str, default: object = None) -> object:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _world_rank() -> tuple[int, int]:
    try:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise LoraTrainerError("RANK and WORLD_SIZE must be integers") from exc
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise LoraTrainerError("invalid RANK/WORLD_SIZE environment")
    return rank, world_size


def _is_cli_primary_process() -> bool:
    try:
        return _world_rank()[0] == 0
    except LoraTrainerError:
        return True


def _shared_torchrun_id() -> str:
    token = os.environ.get("TORCHELASTIC_RUN_ID", "").strip()
    if token and token.lower() != "none":
        return "torchrun_" + sha256(token.encode("utf-8")).hexdigest()[:16]
    # Single-node torchrun workers share their elastic agent parent.  This
    # avoids independently generated timestamps/UUIDs even when no rdzv id was
    # supplied.  Multi-node jobs should provide --run-id or TORCHELASTIC_RUN_ID.
    local_world = os.environ.get("LOCAL_WORLD_SIZE", "").strip()
    world = os.environ.get("WORLD_SIZE", "1").strip()
    if local_world and local_world == world:
        return f"torchrun_{os.getppid()}"
    raise LoraTrainerError(
        "multi-node torchrun requires --run-id, UAV_AGENT_LORA_RUN_ID, or a "
        "shared TORCHELASTIC_RUN_ID"
    )


def resolve_run_id(
    config: LoraScaffoldConfig,
    requested: str | None = None,
    *,
    now: datetime | None = None,
) -> str:
    """Resolve one run id shared by every rank, including resume jobs."""

    resume = getattr(config, "resume_from_checkpoint", None)
    if resume is not None:
        checkpoint = Path(resume).expanduser().resolve()
        if checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent.parent != Path(
            config.output_dir
        ).resolve():
            raise LoraTrainerError(
                "resume_from_checkpoint must be <output_dir>/<run_id>/checkpoints/checkpoint-*"
            )
        inferred = validate_run_id(checkpoint.parent.parent.name)
        if requested is not None and validate_run_id(requested) != inferred:
            raise LoraTrainerError("--run-id does not match the resumed checkpoint run")
        return inferred
    if requested is not None:
        return validate_run_id(requested)
    environment_id = os.environ.get("UAV_AGENT_LORA_RUN_ID", "").strip()
    if environment_id:
        return validate_run_id(environment_id)
    _, world_size = _world_rank()
    if world_size > 1:
        return validate_run_id(_shared_torchrun_id())
    current = now or datetime.now(timezone.utc)
    return validate_run_id(
        f"{current.strftime('%Y%m%d-%H%M%SZ')}_{uuid4().hex[:8]}"
    )


def validate_placeholder(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Preserve the original no-model/no-write placeholder behavior."""

    config = load_lora_config(config_path)
    if config.status != "placeholder":
        raise LoraScaffoldError(
            "validate_placeholder only accepts status=placeholder; use --validate-only "
            "or the active training path for a reviewed active config"
        )
    report = validate_dataset(config.dataset_dir)
    if not report.valid:
        raise LoraScaffoldError("Fleet Planner dataset is invalid: " + "; ".join(report.errors))
    return {
        "status": "placeholder",
        "training_started": False,
        "weights_created": False,
        "base_model_path": str(config.base_model_path),
        "output_dir": str(config.output_dir),
        "dataset": report.to_dict(),
        "target_modules": None,
        "note": "configuration and data validated; no model was loaded and no training ran",
    }


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _validate_output_paths(config: LoraScaffoldConfig) -> dict[str, str]:
    output = Path(config.output_dir).expanduser().resolve()
    adapter_raw = getattr(config, "adapter_output_dir", None)
    if adapter_raw is None:
        raise LoraScaffoldError("active config requires adapter_output_dir")
    adapter = Path(adapter_raw).expanduser().resolve()
    base = Path(config.base_model_path).expanduser().resolve()
    for name, path in (("output_dir", output), ("adapter_output_dir", adapter)):
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise LoraScaffoldError(f"{name} must be a real directory when it exists: {path}")
        ancestor = _nearest_existing_parent(path)
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
            raise LoraScaffoldError(f"{name} has no writable parent: {path}")
        if path == base or base in path.parents:
            raise LoraScaffoldError(f"{name} cannot be inside the base model directory")
    if output == adapter or output in adapter.parents or adapter in output.parents:
        raise LoraScaffoldError("output_dir and adapter_output_dir must be separate trees")
    return {"output_dir": str(output), "adapter_output_dir": str(adapter)}


def _validate_local_model_path(
    config: LoraScaffoldConfig,
    validator: Callable[[str | Path], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    root = Path(config.base_model_path).expanduser().resolve()
    if validator is not None:
        result = dict(validator(root))
        if not bool(result.get("complete")):
            raw_errors = result.get("errors", [])
            errors = raw_errors if isinstance(raw_errors, list) else [raw_errors]
            raise LoraScaffoldError(
                "local base model directory is incomplete: "
                + "; ".join(str(item) for item in errors)
            )
        return result
    config_json = root / "config.json"
    if not root.is_dir() or not config_json.is_file():
        raise LoraScaffoldError(
            f"local base model must be a directory containing config.json: {root}"
        )
    weights = sorted(
        path.name
        for pattern in ("*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin")
        for path in root.glob(pattern)
        if path.is_file()
    )
    if not weights:
        raise LoraScaffoldError(f"local base model has no local weight files: {root}")
    return {"path": str(root), "config": str(config_json), "weight_files": weights}


def _build_datasets(
    config: LoraScaffoldConfig, components: ActiveTrainingComponents
) -> tuple[object, object]:
    train = components.dataset_class(
        config.dataset_dir,
        split=str(getattr(config, "train_split")),
        max_samples=getattr(config, "max_train_samples", None),
    )
    validation = components.dataset_class(
        config.dataset_dir,
        split=str(getattr(config, "validation_split")),
        max_samples=getattr(config, "max_validation_samples", None),
    )
    if len(train) <= 0 or len(validation) <= 0:  # type: ignore[arg-type]
        raise LoraScaffoldError("active training requires non-empty train and validation splits")
    return train, validation


def validate_active(
    config: LoraScaffoldConfig,
    *,
    components: ActiveTrainingComponents | None = None,
) -> tuple[dict[str, object], tuple[object, object]]:
    """Validate active inputs without loading Qwen, allocating CUDA, or writing."""

    if config.status != "active":
        raise LoraScaffoldError("active validation requires status=active")
    resolved_components = components or _active_components()
    train, validation = _build_datasets(config, resolved_components)
    paths = _validate_output_paths(config)
    model = _validate_local_model_path(config, resolved_components.model_validator)
    requested_targets = list(getattr(config, "target_modules") or ())
    if not requested_targets:
        raise LoraScaffoldError("active config requires reviewed target_modules")
    result: dict[str, object] = {
        "status": "active",
        "training_started": False,
        "weights_created": False,
        "dataset_dir": str(config.dataset_dir),
        "dataset_manifest_sha256": str(_value(train, "manifest_sha256")),
        "train_split": str(getattr(config, "train_split")),
        "validation_split": str(getattr(config, "validation_split")),
        "train_count": len(train),  # type: ignore[arg-type]
        "validation_count": len(validation),  # type: ignore[arg-type]
        "model": model,
        **paths,
        "target_modules": requested_targets,
        "target_module_runtime_match_required": True,
        "note": (
            "config, complete dataset, local model files, target patterns, and output "
            "paths validated; model was not loaded and no training ran"
        ),
    }
    return result, (train, validation)


def _prepare_run(config: LoraScaffoldConfig, paths: TrainingPaths) -> Path:
    """Create one run layout on rank zero and join it from other ranks."""

    rank, world_size = _world_rank()
    run_config = paths.run_dir / "config.json"
    resume = getattr(config, "resume_from_checkpoint", None)
    if rank == 0:
        if resume is None:
            if paths.run_dir.exists() and any(paths.run_dir.iterdir()):
                raise LoraTrainerError(f"fresh run directory is not empty: {paths.run_dir}")
            if paths.final_adapter_dir.exists():
                raise LoraTrainerError(
                    f"final adapter output already exists: {paths.final_adapter_dir}"
                )
            paths.run_dir.mkdir(parents=True, exist_ok=True)
            paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            paths.metrics_dir.mkdir(parents=True, exist_ok=True)
            paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            source_payload = _read_json_object(config.config_path, "training config")
            _atomic_json(run_config, source_payload)
        else:
            if not paths.run_dir.is_dir() or not run_config.is_file():
                raise LoraTrainerError("resumed run is missing its immutable config.json")
            checkpoint = Path(resume).expanduser().resolve()
            if not checkpoint.is_dir():
                raise LoraTrainerError(f"resume checkpoint does not exist: {checkpoint}")
            require_adapter_artifacts(checkpoint)
            missing_state = [
                filename
                for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt")
                if not (checkpoint / filename).is_file()
            ]
            if missing_state:
                raise LoraTrainerError(
                    "resume checkpoint is missing Trainer state: "
                    + ", ".join(missing_state)
                )
            original = _read_json_object(run_config, "stored training config")
            current = _read_json_object(config.config_path, "resume training config")
            original.pop("resume_from_checkpoint", None)
            current.pop("resume_from_checkpoint", None)
            if original != current:
                raise LoraTrainerError(
                    "resume config differs from the stored run config outside "
                    "resume_from_checkpoint"
                )
    elif world_size > 1:
        deadline = time.monotonic() + 30.0
        while not run_config.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not run_config.is_file():
            raise LoraTrainerError("rank zero did not publish the shared run config")
    return run_config


def _wait_for_everyone(trainer: object) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    wait = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait):
        wait()
        return
    try:
        import torch.distributed as distributed

        if distributed.is_available() and distributed.is_initialized():
            distributed.barrier()
    except ImportError:
        return


def _is_world_process_zero(trainer: object) -> bool:
    predicate = getattr(trainer, "is_world_process_zero", None)
    if callable(predicate):
        return bool(predicate())
    return _world_rank()[0] == 0


def _json_metrics(result: object) -> dict[str, object]:
    raw = result if isinstance(result, Mapping) else _value(result, "metrics", default={})
    if not isinstance(raw, Mapping):
        return {}
    # A JSON encode/decode round trip normalizes numpy scalar subclasses without
    # allowing NaN/Infinity into reproducibility artifacts.
    normalized: dict[str, object] = {}
    for key, value in raw.items():
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        normalized[str(key)] = value
    json.dumps(normalized, allow_nan=False)
    return normalized


def _module_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().lower()
    return value if len(value) == 40 and all(char in "0123456789abcdef" for char in value) else None


def _gpu_metadata() -> tuple[str | None, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, 0
        _, world_size = _world_rank()
        return torch.cuda.get_device_name(torch.cuda.current_device()), max(
            torch.cuda.device_count(), world_size
        )
    except (ImportError, RuntimeError):
        return None, 0


def _adapter_rank_and_expression(adapter_dir: Path) -> tuple[int, str | list[str]]:
    payload = _read_json_object(adapter_dir / "adapter_config.json", "adapter config")
    rank = payload.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise LoraTrainerError("saved adapter config has invalid rank")
    expression = payload.get("target_modules")
    if isinstance(expression, str):
        if not expression.startswith("^") or not expression.endswith("$"):
            raise LoraTrainerError("saved adapter target regex must be anchored")
    elif isinstance(expression, list):
        if not expression or any(not isinstance(item, str) or not item for item in expression):
            raise LoraTrainerError("saved adapter target_modules list is invalid")
    else:
        raise LoraTrainerError("saved adapter target_modules must be a list or regex")
    return rank, expression


def _build_run_manifest(
    *,
    config: LoraScaffoldConfig,
    paths: TrainingPaths,
    train_dataset: object,
    validation_dataset: object,
    model_bundle: object,
    trainer: object,
    run_config: Path,
    final_adapter: Path,
) -> dict[str, object]:
    rank, adapter_target_expression = _adapter_rank_and_expression(final_adapter)
    target_report = _value(model_bundle, "target_report", default={})
    raw_targets = _value(target_report, "matched_language_modules", default=())
    if not isinstance(raw_targets, (list, tuple)) or not raw_targets or any(
        not isinstance(item, str) or not item for item in raw_targets
    ):
        raise LoraTrainerError("model loader returned invalid matched target modules")
    actual_targets = sorted(raw_targets)
    stats = _value(model_bundle, "parameter_stats", default={})
    total = int(_value(stats, "total_parameter_count", "total_parameters", default=0))
    trainable = int(
        _value(stats, "trainable_parameter_count", "trainable_parameters", default=0)
    )
    if total <= 0 or trainable <= 0 or trainable > total:
        raise LoraTrainerError("model loader returned invalid parameter statistics")
    state = getattr(trainer, "state", None)
    best_checkpoint = _value(state, "best_model_checkpoint")
    global_step = _value(state, "global_step", default=0)
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise LoraTrainerError(
            "Trainer reported no completed optimizer step; refusing success manifest"
        )
    gpu_name, gpu_count = _gpu_metadata()
    base_config = Path(config.base_model_path) / "config.json"
    manifest_sha = str(_value(train_dataset, "manifest_sha256", default=""))
    if len(manifest_sha) != 64:
        raise LoraTrainerError("dataset adapter did not expose a valid manifest SHA-256")
    return {
        "schema_version": 1,
        "run_id": paths.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(),
        "base_model_path": str(Path(config.base_model_path).resolve()),
        "base_model_name": Path(config.base_model_path).name,
        "base_model_config_sha256": _hash_file(base_config),
        "dataset_dir": str(Path(config.dataset_dir).resolve()),
        "dataset_manifest_sha256": manifest_sha,
        "train_count": len(train_dataset),  # type: ignore[arg-type]
        "validation_count": len(validation_dataset),  # type: ignore[arg-type]
        "rank": rank,
        "alpha": float(getattr(config, "lora_alpha")),
        "dropout": float(getattr(config, "lora_dropout")),
        "target_modules": actual_targets,
        "adapter_target_expression": adapter_target_expression,
        "requested_target_modules": list(getattr(config, "target_modules") or ()),
        "learning_rate": float(getattr(config, "learning_rate")),
        "batch_size": int(getattr(config, "per_device_train_batch_size")),
        "per_device_train_batch_size": int(getattr(config, "per_device_train_batch_size")),
        "gradient_accumulation_steps": int(getattr(config, "gradient_accumulation_steps")),
        "epochs": float(getattr(config, "num_train_epochs")),
        "max_steps": getattr(config, "max_steps"),
        "seed": int(getattr(config, "seed")),
        "torch_version": _module_version("torch"),
        "transformers_version": _module_version("transformers"),
        "peft_version": _module_version("peft"),
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "trainable_parameter_count": trainable,
        "total_parameter_count": total,
        "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint),
        "global_step": global_step,
        "final_adapter_path": str(final_adapter),
        "training_config_sha256": _hash_file(run_config),
        "training_started": True,
        "weights_created": True,
    }


def run_active_training(
    config: LoraScaffoldConfig,
    *,
    run_id: str | None = None,
    components: ActiveTrainingComponents | None = None,
) -> dict[str, object]:
    """Execute active Qwen/PEFT SFT through Transformers Trainer."""

    resolved_components = components or _active_components()
    _, datasets = validate_active(config, components=resolved_components)
    train_dataset, validation_dataset = datasets
    resolved_run_id = resolve_run_id(config, run_id)
    paths = build_training_paths(config, resolved_run_id)
    run_config = _prepare_run(config, paths)

    bundle = resolved_components.model_loader(config, reporter=print)
    model = _value(bundle, "model")
    processor = _value(bundle, "processor")
    if model is None or processor is None:
        raise LoraTrainerError("model loader did not return model and processor")
    collator = resolved_components.collator_class(
        processor, model_max_length=int(getattr(config, "model_max_length"))
    )
    trainer = resolved_components.trainer_builder(
        model=model,
        processor=processor,
        config=config,
        paths=paths,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        data_collator=collator,
    )

    resume = getattr(config, "resume_from_checkpoint", None)
    train_result = trainer.train(
        resume_from_checkpoint=None if resume is None else str(Path(resume).resolve())
    )
    train_metrics = _json_metrics(train_result)
    evaluate = getattr(trainer, "evaluate", None)
    validation_metrics = _json_metrics(evaluate()) if callable(evaluate) else {}
    save_state = getattr(trainer, "save_state", None)
    if not callable(save_state):
        raise LoraTrainerError("Transformers Trainer does not expose save_state")
    save_state()
    _wait_for_everyone(trainer)

    manifest_path = paths.run_dir / "run_manifest.json"
    publish_error: Exception | None = None
    if _is_world_process_zero(trainer):
        try:
            write_metrics(paths.metrics_dir, "train_metrics", train_metrics)
            write_metrics(paths.metrics_dir, "validation_metrics", validation_metrics)
            final_adapter = resolved_components.final_adapter_saver(
                getattr(trainer, "model", model), paths.final_adapter_dir
            )
            run_manifest = _build_run_manifest(
                config=config,
                paths=paths,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                model_bundle=bundle,
                trainer=trainer,
                run_config=run_config,
                final_adapter=final_adapter,
            )
            _atomic_json(manifest_path, run_manifest)
            adapter_manifest = build_manifest(
                final_adapter,
                Path(config.base_model_path).name,
                run_manifest=manifest_path,
                training_config=run_config,
                base_model_config=Path(config.base_model_path) / "config.json",
            )
            _atomic_json(final_adapter / "adapter_manifest.json", adapter_manifest)
        except Exception as exc:  # all ranks must still reach the final barrier
            publish_error = exc
    _wait_for_everyone(trainer)
    if publish_error is not None:
        raise LoraTrainerError(
            f"rank zero final adapter publication failed: {publish_error}"
        ) from publish_error
    if not manifest_path.is_file():
        raise LoraTrainerError("rank zero did not publish run_manifest.json")
    result = _read_json_object(manifest_path, "run manifest")
    final_adapter_value = result.get("final_adapter_path")
    if not isinstance(final_adapter_value, str) or not final_adapter_value.strip():
        raise LoraTrainerError("run manifest has no final_adapter_path")
    final_adapter_path = Path(final_adapter_value).expanduser().resolve()
    if not (final_adapter_path / "adapter_manifest.json").is_file():
        raise LoraTrainerError("rank zero did not publish adapter_manifest.json")
    result["status"] = "active"
    result["train_metrics"] = train_metrics
    result["validation_metrics"] = validation_metrics
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        config = load_lora_config(args.config)
        if config.status == "placeholder":
            if args.run_id is not None:
                raise LoraScaffoldError("--run-id is invalid for placeholder mode")
            result = validate_placeholder(args.config)
        elif args.validate_only:
            if args.run_id is not None:
                validate_run_id(args.run_id)
            result, _ = validate_active(config)
        else:
            result = run_active_training(config, run_id=args.run_id)
        if _is_cli_primary_process():
            print(
                json.dumps(
                    result, ensure_ascii=False, allow_nan=False, sort_keys=True
                )
            )
        return 0
    except (
        ImportError,
        LoraScaffoldError,
        LoraTrainerError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        if _is_cli_primary_process():
            print(f"Fleet Planner LoRA error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
