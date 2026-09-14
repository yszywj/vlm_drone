#!/usr/bin/env python3
"""Train one planning role with complete, prevalidated assistant-only labels."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import gc
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.lora.collator import AssistantOnlyDataCollator
from training.lora.config import load_lora_config
from training.lora.modeling import load_qwen_lora_model, validate_local_model_directory
from training.lora.roles_dataset import PLANNING_ROLES, PlanningRoleSFTDataset
from training.lora.train_fleet_planner_lora import (
    ActiveTrainingComponents, _atomic_json, resolve_run_id, run_active_training, validate_active,
)
from training.lora.trainer import build_trainer, build_training_paths


def import_huggingface_datasets():
    """Bind the installed distribution before Trainer sees the local namesake.

    The repository's datasets/target_state package is unrelated to Hugging Face
    datasets. Resolve the latter through package metadata in this training-only
    process, without renaming the repository package or altering sys.path.
    """
    distribution = importlib.metadata.distribution("datasets")
    package_file = Path(distribution.locate_file("datasets/__init__.py")).resolve()
    existing = sys.modules.get("datasets")
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != package_file:
            raise RuntimeError("the local datasets package was imported before the training environment bootstrap")
        return existing
    spec = importlib.util.spec_from_file_location("datasets", package_file,
                                                 submodule_search_locations=[str(package_file.parent)])
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load installed Hugging Face datasets")
    module = importlib.util.module_from_spec(spec)
    sys.modules["datasets"] = module
    try:
        spec.loader.exec_module(module)
        if not hasattr(module, "Dataset"):
            raise RuntimeError("installed datasets distribution lacks Dataset")
    except BaseException:
        sys.modules.pop("datasets", None)
        raise
    return module


class EncodedRoleDataset(Sequence):
    """Token IDs cached once, with their source manifest identity retained."""
    def __init__(self, source, collator):
        self.manifest_sha256 = source.manifest_sha256
        self.role = source.role
        self.split = source.split
        self.rows = []
        self.sample_ids = []
        for index in range(len(source)):
            row = source[index]
            self.rows.append(collator.encode_feature(row))
            self.sample_ids.append(row["sample_id"])
            if (index + 1) % 500 == 0:
                print(f"Pre-tokenized {self.role}/{self.split}: {index + 1}/{len(source)}", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def summary(self):
        return {
            "count": len(self),
            "max_full_tokens": max(len(row["input_ids"]) for row in self.rows),
            "total_full_tokens": sum(len(row["input_ids"]) for row in self.rows),
            "supervised_tokens": sum(sum(label != -100 for label in row["labels"]) for row in self.rows),
        }


def prepare_role(config, role):
    """Check all files and encode train/validation before loading any weights."""
    if role not in PLANNING_ROLES:
        raise ValueError(f"unsupported role: {role}")
    if config.train_split != "train" or config.validation_split != "validation":
        raise ValueError("planning role training requires train/validation; test is held out")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config.base_model_path), local_files_only=True)
    collator = AssistantOnlyDataCollator(tokenizer, model_max_length=config.model_max_length)
    raw_components = ActiveTrainingComponents(
        dataset_class=lambda root, **kwargs: PlanningRoleSFTDataset(root, role=role, **kwargs),
        collator_class=AssistantOnlyDataCollator,
        model_loader=load_qwen_lora_model,
        model_validator=validate_local_model_directory,
    )
    report, raw_datasets = validate_active(config, components=raw_components)
    prepared = {source.split: EncodedRoleDataset(source, collator) for source in raw_datasets}
    report.update({
        "role": role, "model_max_length": config.model_max_length,
        "assistant_only": True, "truncation": "reject", "test_used_for_training": False,
        "encoded_splits": {split: data.summary() for split, data in prepared.items()},
    })
    return report, prepared, collator


def longest_training_step_probe(model, dataset, collator, config):
    """Exercise peak-length forward/backward/AdamW, then restore initial LoRA.

    Only the training split is used. All trainable tensors and RNG are restored
    before Trainer begins; the probe does not add an untracked training step.
    """
    import torch
    from transformers import set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("training requires a CUDA device supporting BF16")
    index = max(range(len(dataset)), key=lambda item: len(dataset[item]["input_ids"]))
    model.to("cuda")
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    originals = [parameter.detach().cpu().clone() for parameter in parameters]
    batch = {key: tensor.to("cuda") for key, tensor in collator([dataset[index]]).items()}
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    torch.cuda.reset_peak_memory_stats()
    print(f"Longest training sample probe: {dataset.sample_ids[index]}, {batch['input_ids'].shape[-1]} tokens", flush=True)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(**batch, use_cache=False)
            loss = result.loss
            del result
        if not torch.isfinite(loss):
            raise RuntimeError("longest-sample loss is non-finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
        if not torch.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError("longest-sample LoRA gradients are absent, zero or non-finite")
        optimizer.step()
        torch.cuda.synchronize()
        report = {
            "passed": True, "sample_id": dataset.sample_ids[index],
            "full_tokens": int(batch["input_ids"].shape[-1]),
            "loss": float(loss.detach()), "grad_norm": float(grad_norm),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "optimizer_step_completed": True, "initial_lora_weights_restored": True,
        }
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)
        model.zero_grad(set_to_none=True)
        del optimizer, batch, originals
        gc.collect()
        torch.cuda.empty_cache()
        set_seed(config.seed)
    print(json.dumps({"longest_sample_probe": report}), flush=True)
    return report


def _versions():
    return {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "peft", "accelerate", "tensorboard", "datasets",
    )}


def run_role(config, role, run_id):
    import_huggingface_datasets()
    from transformers import TrainerCallback, set_seed

    paths = build_training_paths(config, run_id)
    if config.resume_from_checkpoint is not None:
        raise ValueError("this initial role launcher does not yet support resume; use a new run")
    if paths.run_dir.exists() and any(paths.run_dir.iterdir()):
        raise ValueError(f"refusing to overwrite an existing role run: {paths.run_dir}")
    started = time.monotonic()
    status = {"role": role, "run_id": run_id, "pid": os.getpid(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "global_step": 0}

    def update(phase, **values):
        status.update(values, phase=phase, elapsed_seconds=time.monotonic()-started,
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(paths.run_dir / "status.json", status)

    report, prepared, _ = prepare_role(config, role)
    # The legacy entry prepares immutable run directories before model_loader.
    # Do not create the run directory during preflight.
    class Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            update("training", global_step=state.global_step, max_steps=state.max_steps)

        def on_log(self, args, state, control, logs=None, **kwargs):
            metrics = dict(logs or {})
            for key in ("loss", "grad_norm", "eval_loss"):
                if key in metrics and not math.isfinite(float(metrics[key])):
                    raise RuntimeError(f"non-finite training metric {key}: {metrics[key]}")
            update("training", global_step=state.global_step, max_steps=state.max_steps,
                   epoch=state.epoch, latest_metrics=metrics)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in (1, 5):
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            update("training", global_step=state.global_step,
                   latest_checkpoint=str(paths.checkpoints_dir / f"checkpoint-{state.global_step}"))

    def loader(active_config, reporter=print):
        _atomic_json(paths.run_dir / "preflight.json", report)
        source_paths = sorted((ROOT / "training/lora").glob("*.py"))
        _atomic_json(paths.run_dir / "training_sources.json", {
            "sha256": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in source_paths},
            "packages": _versions(), "python": sys.version, "executable": sys.executable,
        })
        update("loading_model")
        set_seed(active_config.seed)
        bundle = load_qwen_lora_model(active_config, reporter=reporter)
        update("longest_sample_probe")
        collator = AssistantOnlyDataCollator(bundle.processor, model_max_length=active_config.model_max_length)
        probe = longest_training_step_probe(bundle.model, prepared["train"], collator, active_config)
        _atomic_json(paths.run_dir / "longest_sample_probe.json", probe)
        return bundle

    def trainer_builder(**kwargs):
        trainer = build_trainer(**kwargs)
        trainer.args.logging_nan_inf_filter = False
        trainer.add_callback(Progress())
        return trainer

    components = ActiveTrainingComponents(
        dataset_class=lambda root, split, max_samples=None: prepared[split],
        collator_class=AssistantOnlyDataCollator, model_loader=loader,
        model_validator=validate_local_model_directory, trainer_builder=trainer_builder,
    )
    try:
        result = run_active_training(config, run_id=run_id, components=components)
        for path in (paths.run_dir / "run_manifest.json", paths.final_adapter_dir / "adapter_manifest.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata.update(planning_role=role, training_data_contract="planning_roles_v1_messages")
            _atomic_json(path, metadata)
        update("completed", global_step=result["global_step"], final_adapter_path=str(paths.final_adapter_dir))
        result["planning_role"] = role
        return result
    except BaseException as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=PLANNING_ROLES, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        config = load_lora_config(args.config).require_active()
        if args.validate_only:
            result, _, _ = prepare_role(config, args.role)
        else:
            result = run_role(config, args.role, resolve_run_id(config, args.run_id))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
