from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import struct

import pytest

from training.lora.config import LoraScaffoldConfig
from training.lora.export_adapter_manifest import build_manifest
from training.lora.train_fleet_planner_lora import (
    ActiveTrainingComponents,
    resolve_run_id,
    run_active_training,
)
from training.lora.trainer import (
    LoraTrainerError,
    build_training_arguments,
    build_training_paths,
    get_adapter_only_trainer_class,
)


TARGET = "model.language_model.layers.0.self_attn.q_proj"
TARGET_REGEX = r"^(?:model\.language_model\.layers\.0\.self_attn\.q_proj)$"


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {"base_model.model.lora_A.weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"x"


class _FakePeftModel:
    peft_config = {"default": object()}

    def __init__(self, base_model_name: str = "Qwen3-VL-4B-Instruct", *, leak_base: bool = False):
        self.base_model_name = base_model_name
        self.leak_base = leak_base

    def save_pretrained(self, destination: str, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": 16,
                    "target_modules": TARGET_REGEX,
                    "base_model_name_or_path": self.base_model_name,
                }
            ),
            encoding="utf-8",
        )
        (root / "adapter_model.safetensors").write_bytes(_safetensors_bytes())
        if self.leak_base:
            (root / "model.safetensors").write_bytes(b"forbidden")


def _active_config(tmp_path: Path) -> LoraScaffoldConfig:
    model = tmp_path / "base" / "Qwen3-VL-4B-Instruct"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"local-base")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = tmp_path / "active.json"
    source.write_text('{"fixture": true}\n', encoding="utf-8")
    return LoraScaffoldConfig(
        config_path=source,
        schema_version=1,
        status="active",
        base_model_path=model,
        output_dir=tmp_path / "runs",
        dataset_dir=dataset,
        target_modules=(TARGET,),
        rank=16,
        lora_alpha=32.0,
        lora_dropout=0.05,
        adapter_output_dir=tmp_path / "adapters",
        seed=7,
        model_max_length=512,
        num_train_epochs=1.0,
        max_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        save_total_limit=2,
        dataloader_num_workers=0,
        train_split="train",
        validation_split="validation",
        resume_from_checkpoint=None,
        max_train_samples=2,
        max_validation_samples=1,
        notes="test",
    )


def test_training_arguments_map_config_and_checkpoint_root(tmp_path: Path) -> None:
    config = _active_config(tmp_path)
    paths = build_training_paths(config, "run_1")

    class Arguments:
        def __init__(self, evaluation_strategy: str | None = None, **kwargs: object):
            self.values = {"evaluation_strategy": evaluation_strategy, **kwargs}

    args = build_training_arguments(config, paths, arguments_class=Arguments)
    assert args.values["output_dir"] == str(paths.checkpoints_dir)
    assert args.values["evaluation_strategy"] == "steps"
    assert args.values["gradient_accumulation_steps"] == 2
    assert args.values["save_total_limit"] == 2
    assert args.values["remove_unused_columns"] is False
    assert args.values["ddp_find_unused_parameters"] is False


def test_adapter_only_trainer_ignores_full_state_dict_and_rejects_base_leak(
    tmp_path: Path,
) -> None:
    class BaseTrainer:
        def __init__(self, model: object, output_dir: Path):
            self.model = model
            self.args = SimpleNamespace(output_dir=str(output_dir))

    cls = get_adapter_only_trainer_class(BaseTrainer)
    output = tmp_path / "checkpoint-1"
    trainer = cls(_FakePeftModel(), output)
    trainer._save(state_dict={"complete.base.weight": object()})
    assert (output / "adapter_config.json").is_file()
    assert (output / "adapter_model.safetensors").is_file()
    assert not (output / "model.safetensors").exists()

    leaking = cls(_FakePeftModel(leak_base=True), tmp_path / "checkpoint-2")
    with pytest.raises(LoraTrainerError, match="base-model weight files"):
        leaking._save()


def test_adapter_only_trainer_refuses_non_peft_model(tmp_path: Path) -> None:
    class BaseTrainer:
        def __init__(self) -> None:
            self.model = SimpleNamespace(save_pretrained=lambda *_args, **_kwargs: None)
            self.args = SimpleNamespace(output_dir=str(tmp_path / "checkpoint"))

    trainer = get_adapter_only_trainer_class(BaseTrainer)()
    with pytest.raises(LoraTrainerError, match="non-PEFT"):
        trainer._save()


def test_manifest_accepts_legacy_call_and_strongly_checks_safetensors(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "target_modules": [TARGET],
                "base_model_name_or_path": "/models/Qwen3-VL-4B-Instruct",
            }
        ),
        encoding="utf-8",
    )
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(_safetensors_bytes())
    payload = build_manifest(adapter, "Qwen3-VL-4B-Instruct")
    assert payload["schema_version"] == 1
    assert payload["rank"] == 16
    assert payload["target_modules"] == [TARGET]
    assert set(payload["files"]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }

    weights.write_bytes(b"not-a-safetensors-file")
    with pytest.raises(ValueError, match="safetensors"):
        build_manifest(adapter, "Qwen3-VL-4B-Instruct")


def test_manifest_rejects_sharded_base_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "target_modules": [TARGET],
                "base_model_name_or_path": "Qwen3-VL-4B-Instruct",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(_safetensors_bytes())
    leaked = adapter / "nested"
    leaked.mkdir()
    (leaked / "model-00001-of-00002.safetensors").write_bytes(b"base")

    with pytest.raises(ValueError, match="forbidden base weights"):
        build_manifest(adapter, "Qwen3-VL-4B-Instruct")


def test_manifest_rejects_broad_or_non_language_target_expression(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config_path = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(_safetensors_bytes())
    base = {
        "peft_type": "LORA",
        "r": 16,
        "base_model_name_or_path": "Qwen3-VL-4B-Instruct",
    }
    config_path.write_text(
        json.dumps({**base, "target_modules": r"^.*$"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exact alternation"):
        build_manifest(adapter, "Qwen3-VL-4B-Instruct")

    config_path.write_text(
        json.dumps({**base, "target_modules": ["visual.blocks.0.attn.q_proj"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="language attention/MLP"):
        build_manifest(adapter, "Qwen3-VL-4B-Instruct")


def test_resolve_run_id_uses_resumed_run_and_shared_torchrun_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _active_config(tmp_path)
    checkpoint = config.output_dir / "existing" / "checkpoints" / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    resumed = replace(config, resume_from_checkpoint=checkpoint)
    assert resolve_run_id(resumed) == "existing"

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "shared-rendezvous")
    first = resolve_run_id(config)
    assert first == resolve_run_id(config)
    assert first.startswith("torchrun_")


def test_mock_active_training_writes_run_final_adapter_and_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _active_config(tmp_path)

    class Dataset:
        manifest_sha256 = "a" * 64

        def __init__(
            self,
            dataset_root: Path,
            *,
            split: str,
            max_samples: int | None,
        ) -> None:
            self.dataset_root = dataset_root
            self.split = split
            available = 4 if split == "train" else 3
            self.count = min(available, max_samples or available)

        def __len__(self) -> int:
            return self.count

    class Collator:
        def __init__(self, processor: object, *, model_max_length: int) -> None:
            assert processor == "processor"
            assert model_max_length == 512

    model = _FakePeftModel()
    bundle = SimpleNamespace(
        model=model,
        processor="processor",
        target_report=SimpleNamespace(matched_language_modules=(TARGET,)),
        parameter_stats=SimpleNamespace(total_parameters=1000, trainable_parameters=16),
    )

    class Accelerator:
        def wait_for_everyone(self) -> None:
            return None

    class Trainer:
        def __init__(self) -> None:
            self.model = model
            self.accelerator = Accelerator()
            self.state = SimpleNamespace(best_model_checkpoint=None, global_step=1)
            self.resume: str | None = "unset"
            self.saved_state = False

        def train(self, *, resume_from_checkpoint: str | None) -> object:
            self.resume = resume_from_checkpoint
            return SimpleNamespace(metrics={"train_loss": 0.5})

        def evaluate(self) -> dict[str, float]:
            return {"eval_loss": 0.4}

        def save_state(self) -> None:
            self.saved_state = True

        def is_world_process_zero(self) -> bool:
            return True

    trainer = Trainer()

    def trainer_builder(**kwargs: object) -> Trainer:
        assert len(kwargs["train_dataset"]) == 2  # type: ignore[arg-type]
        assert len(kwargs["validation_dataset"]) == 1  # type: ignore[arg-type]
        return trainer

    components = ActiveTrainingComponents(
        dataset_class=Dataset,
        collator_class=Collator,
        model_loader=lambda *_args, **_kwargs: bundle,
        trainer_builder=trainer_builder,
    )
    monkeypatch.setattr(
        "training.lora.train_fleet_planner_lora._git_commit", lambda: None
    )
    monkeypatch.setattr(
        "training.lora.train_fleet_planner_lora._gpu_metadata", lambda: (None, 0)
    )

    result = run_active_training(config, run_id="mock_run", components=components)
    run = config.output_dir / "mock_run"
    adapter = config.adapter_output_dir / "mock_run"  # type: ignore[operator]
    assert trainer.resume is None
    assert trainer.saved_state is True
    assert result["training_started"] is True
    assert result["train_count"] == 2
    assert result["validation_count"] == 1
    assert result["global_step"] == 1
    assert result["target_modules"] == [TARGET]
    assert result["adapter_target_expression"] == TARGET_REGEX
    assert (run / "config.json").is_file()
    assert (run / "run_manifest.json").is_file()
    assert json.loads((run / "metrics/train_metrics.json").read_text())["train_loss"] == 0.5
    assert json.loads((run / "metrics/validation_metrics.json").read_text())["eval_loss"] == 0.4
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()
    exported = json.loads((adapter / "adapter_manifest.json").read_text())
    assert exported["training_run_id"] == "mock_run"
    assert exported["target_modules"] == [TARGET]
