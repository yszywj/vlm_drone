from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.adapter_registry import AdapterRegistry, AdapterRegistryError
from scripts.build_vllm_lora_args import build_vllm_lora_args
from scripts.check_qwen_server import _redact, _validate_served_models
from models.base import ModelProtocolError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _placeholder_payload() -> dict[str, object]:
    payload = json.loads((PROJECT_ROOT / "configs/adapters.json").read_text())
    for adapter in payload["adapters"].values():
        adapter.update(status="placeholder", path=None, rank=None)
        adapter.pop("generation", None)
    payload["fallback_to_base"] = True
    return payload


@pytest.fixture
def placeholder_registry(tmp_path: Path) -> AdapterRegistry:
    config = tmp_path / "adapters.json"
    config.write_text(json.dumps(_placeholder_payload()), encoding="utf-8")
    return AdapterRegistry(config)


def _active_registry(tmp_path: Path) -> AdapterRegistry:
    payload = _placeholder_payload()
    for name, rank in (("fleet_planner", 8), ("runtime_visual", 32)):
        path = tmp_path / name
        path.mkdir()
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"test-weights")
        payload["adapters"][name].update(status="active", path=str(path), rank=rank)
    config = tmp_path / "adapters.json"
    config.write_text(json.dumps(payload))
    return AdapterRegistry(config)


def test_vllm_args_include_only_active_and_bounded_configured_rank(tmp_path: Path) -> None:
    args = build_vllm_lora_args(_active_registry(tmp_path))
    assert args[0] == "--enable-lora"
    assert "Qwen3-VL-4B-Fleet-Planner-LoRA=" in " ".join(args)
    assert "Qwen3-VL-4B-Runtime-Visual-LoRA=" in " ".join(args)
    assert "Spatial-Mission" not in " ".join(args)
    assert args[args.index("--max-lora-rank") + 1] == "32"
    assert args[args.index("--max-loras") + 1] == "2"
    assert args[args.index("--max-cpu-loras") + 1] == "2"


def test_placeholder_only_registry_keeps_base_vllm_command_unchanged(placeholder_registry: AdapterRegistry) -> None:
    assert build_vllm_lora_args(placeholder_registry) == ()


def test_vllm_args_reject_wrong_served_base_lineage(placeholder_registry: AdapterRegistry) -> None:
    with pytest.raises(AdapterRegistryError, match="base model name"):
        build_vllm_lora_args(
            placeholder_registry,
            expected_base_model_name="wrong-base",
        )


def test_models_check_requires_base_and_all_active_but_not_placeholders(tmp_path: Path) -> None:
    registry = _active_registry(tmp_path)
    served = frozenset(
        {
            "Qwen3-VL-4B-Instruct",
            "Qwen3-VL-4B-Fleet-Planner-LoRA",
            "Qwen3-VL-4B-Runtime-Visual-LoRA",
        }
    )
    _validate_served_models(
        served,
        requested_base_model="Qwen3-VL-4B-Instruct",
        registry=registry,
    )
    with pytest.raises(ModelProtocolError, match="Runtime-Visual"):
        _validate_served_models(
            served - {"Qwen3-VL-4B-Runtime-Visual-LoRA"},
            requested_base_model="Qwen3-VL-4B-Instruct",
            registry=registry,
        )


def test_healthcheck_error_redaction_never_echoes_api_key() -> None:
    secret = "TOP-SECRET-QWEN-KEY"
    assert secret not in _redact(f"Authorization failed for {secret}", secret)
