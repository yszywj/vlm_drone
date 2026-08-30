from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from training.lora.config import LoraScaffoldError, load_lora_config
from training.lora.inspect_qwen_lora_targets import inspect_loaded_model
from training.lora.modeling import (
    LoraModelError,
    exact_target_regex,
    freeze_all_parameters,
    inspect_model_linear_modules,
    validate_local_model_directory,
    validate_lora_target_modules,
    verify_peft_targeted_modules,
    verify_lora_trainable_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLACEHOLDER = PROJECT_ROOT / "configs/lora/fleet_planner_lora.json"
ACTIVE_EXAMPLE = (
    PROJECT_ROOT / "configs/lora/fleet_planner_lora_train.example.json"
)


class Linear:
    pass


class FakeModel:
    def __init__(self, names: list[str], parameters: list[tuple[str, object]] | None = None):
        self._modules = [(name, Linear()) for name in names]
        self._parameters = parameters or []

    def named_modules(self):
        return iter(self._modules)

    def named_parameters(self):
        return iter(self._parameters)

    def parameters(self):
        return (parameter for _, parameter in self._parameters)


class FakeParameter:
    def __init__(self, count: int, requires_grad: bool):
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


def _active_payload() -> dict[str, object]:
    return json.loads(ACTIVE_EXAMPLE.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return path


def test_legacy_placeholder_is_safe_and_has_no_active_values() -> None:
    config = load_lora_config(PLACEHOLDER)
    assert config.is_placeholder
    assert config.target_modules is None
    assert config.max_train_samples is None
    assert config.max_validation_samples is None
    with pytest.raises(LoraScaffoldError, match="cannot start training"):
        config.require_active()


def test_active_example_has_all_strict_fields_and_safe_full_patterns() -> None:
    config = load_lora_config(ACTIVE_EXAMPLE)
    assert config.require_active() is config
    assert config.rank == 16
    assert config.max_steps is None
    assert config.resume_from_checkpoint is None
    assert config.max_train_samples is None
    assert config.max_validation_samples is None
    assert all(
        pattern.startswith("model.language_model.")
        for pattern in config.target_modules or ()
    )


def test_active_missing_any_field_is_rejected(tmp_path: Path) -> None:
    payload = _active_payload()
    del payload["save_total_limit"]
    with pytest.raises(LoraScaffoldError, match="missing=.*save_total_limit"):
        load_lora_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rank", 0, "rank"),
        ("lora_alpha", -1, "lora_alpha"),
        ("lora_dropout", 1.0, "lora_dropout"),
        ("max_train_samples", 0, "max_train_samples"),
        ("max_validation_samples", True, "max_validation_samples"),
        ("learning_rate", float("inf"), "non-finite"),
    ],
)
def test_active_invalid_numeric_values_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _active_payload()
    payload[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LoraScaffoldError, match=message):
        load_lora_config(path)


def test_config_rejects_duplicate_keys_and_nan(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1,"status":"placeholder"}',
        encoding="utf-8",
    )
    with pytest.raises(LoraScaffoldError, match="duplicate JSON key"):
        load_lora_config(duplicate)

    payload = _active_payload()
    payload["learning_rate"] = float("nan")
    encoded = json.dumps(payload)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(encoded, encoding="utf-8")
    with pytest.raises(LoraScaffoldError, match="non-finite JSON constant"):
        load_lora_config(nonfinite)


def test_config_rejects_unknown_split_and_protected_output_paths(tmp_path: Path) -> None:
    payload = _active_payload()
    payload["train_split"] = "made_up"
    with pytest.raises(LoraScaffoldError, match="train_split must be one of"):
        load_lora_config(_write_config(tmp_path, payload))

    payload = _active_payload()
    payload["output_dir"] = str(Path(str(payload["base_model_path"])) / "bad")
    with pytest.raises(LoraScaffoldError, match="base_model_path"):
        load_lora_config(_write_config(tmp_path, payload))

    payload = _active_payload()
    payload["adapter_output_dir"] = str(Path(str(payload["output_dir"])) / "nested")
    with pytest.raises(LoraScaffoldError, match="non-nested"):
        load_lora_config(_write_config(tmp_path, payload))


def test_inventory_and_inspector_keep_language_vision_connector_separate() -> None:
    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.visual.blocks.0.attn.q_proj",
        "model.visual.merger.mlp.0",
        "model.visual.deepstack_merger_list.0.linear_fc1",
        "lm_head",
    ]
    model = FakeModel(names)
    inventory = inspect_model_linear_modules(model)
    assert inventory.language_candidates == tuple(sorted(names[:2]))
    assert inventory.vision_modules == (names[2],)
    assert inventory.connector_modules == tuple(sorted((names[3], names[4])))
    payload = inspect_loaded_model(model, model_path="/model", language_only=True)
    assert payload["language_candidates"] == sorted(names[:2])
    assert payload["vision_modules"] == [names[2]]
    assert payload["connector_modules"] == sorted((names[3], names[4]))
    assert payload["selected_modules"] == sorted(names[:2])


def test_target_globs_expand_only_to_actual_language_linear_modules() -> None:
    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.1.self_attn.q_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.visual.blocks.0.attn.q_proj",
        "model.visual.merger.mlp.0",
    ]
    report = validate_lora_target_modules(
        FakeModel(names),
        [
            "model.language_model.layers.*.self_attn.q_proj",
            "model.language_model.layers.*.mlp.up_proj",
        ],
    )
    assert report.target_module_count == 3
    assert report.matched_vision_modules == ()
    assert report.matched_connector_modules == ()

    regex = exact_target_regex(report.matched_language_modules)
    assert all(re.fullmatch(regex, name) for name in names[:3])
    assert re.fullmatch(regex, names[3]) is None
    assert re.fullmatch(regex, "prefix." + names[0]) is None


@pytest.mark.parametrize(
    "patterns",
    [
        ["q_proj"],
        ["*.q_proj"],
        ["model.visual.blocks.*.attn.q_proj"],
        ["model.language_model.layers.*.self_attn.does_not_exist"],
    ],
)
def test_unsafe_or_zero_target_modules_are_rejected(patterns: list[str]) -> None:
    model = FakeModel(
        [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.visual.blocks.0.attn.q_proj",
        ]
    )
    with pytest.raises(LoraModelError):
        validate_lora_target_modules(model, patterns)


def test_only_language_lora_parameters_may_be_trainable() -> None:
    safe = FakeModel(
        [],
        [
            ("base_model.model.model.language_model.layers.0.self_attn.q_proj.weight", FakeParameter(20, False)),
            ("base_model.model.model.visual.blocks.0.attn.q_proj.weight", FakeParameter(10, False)),
            ("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight", FakeParameter(4, True)),
            ("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight", FakeParameter(4, True)),
        ],
    )
    stats = verify_lora_trainable_parameters(safe)
    assert stats.total_parameters == 38
    assert stats.trainable_parameters == 8

    unsafe_base = FakeModel(
        [],
        [("model.language_model.layers.0.mlp.up_proj.weight", FakeParameter(4, True))],
    )
    with pytest.raises(LoraModelError, match="non-LoRA base parameter"):
        verify_lora_trainable_parameters(unsafe_base)

    unsafe_vision = FakeModel(
        [],
        [("base_model.model.model.visual.blocks.0.attn.q_proj.lora_A.weight", FakeParameter(4, True))],
    )
    with pytest.raises(LoraModelError, match="vision parameter"):
        verify_lora_trainable_parameters(unsafe_vision)


def test_peft_target_audit_rejects_extra_or_missing_modules() -> None:
    expected = ("model.language_model.layers.0.self_attn.q_proj",)

    class PeftAudit:
        targeted_module_names = [expected[0]]

    assert verify_peft_targeted_modules(PeftAudit(), expected) == expected

    class VisionLeak:
        targeted_module_names = [expected[0], "model.visual.blocks.0.attn.q_proj"]

    with pytest.raises(LoraModelError, match="unexpected"):
        verify_peft_targeted_modules(VisionLeak(), expected)

    class Missing:
        targeted_module_names: list[str] = []

    with pytest.raises(LoraModelError, match="zero targeted"):
        verify_peft_targeted_modules(Missing(), expected)


def test_freeze_all_parameters_disables_every_parameter() -> None:
    parameters = [FakeParameter(1, True), FakeParameter(2, True)]
    model = FakeModel([], [("a", parameters[0]), ("b", parameters[1])])
    freeze_all_parameters(model)
    assert not any(parameter.requires_grad for parameter in parameters)


def test_local_checkpoint_validation_does_not_load_weights(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3VLForConditionalGeneration"]}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"not loaded by validation")
    report = validate_local_model_directory(model_dir)
    assert report["complete"] is True
    assert report["weight_files"] == ["model.safetensors"]

    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["WrongModel"]}), encoding="utf-8"
    )
    report = validate_local_model_directory(model_dir)
    assert report["complete"] is False
    assert "Qwen3VLForConditionalGeneration" in "; ".join(report["errors"])


@pytest.mark.parametrize("unsafe_name", [None, 7, "../outside.safetensors", "/abs/model.safetensors"])
def test_local_checkpoint_index_rejects_non_string_and_unsafe_shards(
    tmp_path: Path, unsafe_name: object
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3VLForConditionalGeneration"]}),
        encoding="utf-8",
    )
    for filename in ("tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
        (model_dir / filename).write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": unsafe_name}}),
        encoding="utf-8",
    )
    report = validate_local_model_directory(model_dir)
    assert report["complete"] is False
    assert "invalid or unsafe" in "; ".join(report["errors"])
