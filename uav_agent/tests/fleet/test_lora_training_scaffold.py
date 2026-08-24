from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.lora.config import LoraScaffoldError, load_lora_config
from training.lora.inspect_qwen_lora_targets import _category
from training.lora.train_fleet_planner_lora import validate_placeholder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs/lora/fleet_planner_lora.json"


def test_committed_placeholder_validates_data_without_model_or_weights() -> None:
    config = load_lora_config(CONFIG)
    before = (
        {path.relative_to(config.output_dir) for path in config.output_dir.rglob("*")}
        if config.output_dir.exists()
        else None
    )
    result = validate_placeholder(CONFIG)
    assert result["status"] == "placeholder"
    assert result["training_started"] is False
    assert result["weights_created"] is False
    assert result["target_modules"] is None
    after = (
        {path.relative_to(config.output_dir) for path in config.output_dir.rglob("*")}
        if config.output_dir.exists()
        else None
    )
    assert after == before


def test_placeholder_refuses_guessed_target_modules(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["target_modules"] = ["q_proj"]
    path = tmp_path / "lora.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(LoraScaffoldError, match="do not guess"):
        load_lora_config(path)


def test_target_inspection_keeps_visual_connector_separate_from_tower() -> None:
    assert _category("visual.blocks.0.attn.q_proj") == "vision_tower"
    assert _category("visual.merger.mlp.0") == "connector"
    assert _category("model.layers.0.self_attn.q_proj") == (
        "language_attention_projection"
    )
    assert _category("model.layers.0.mlp.gate_proj") == (
        "language_mlp_projection"
    )
