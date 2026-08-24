from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.adapter_registry import (
    AdapterRegistry,
    AdapterRegistryError,
    AdapterStatus,
    ModelCallRole,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _payload() -> dict[str, object]:
    return json.loads((PROJECT_ROOT / "configs/adapters.json").read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "adapters.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_committed_four_placeholders_fall_back_and_fleet_roles_share_slot() -> None:
    registry = AdapterRegistry(PROJECT_ROOT / "configs/adapters.json")
    assert set(registry.adapters) == {
        "fleet_planner",
        "spatial_mission",
        "runtime_visual",
        "runtime_replanner",
    }
    assert all(spec.status is AdapterStatus.PLACEHOLDER for spec in registry.adapters.values())
    plan = registry.resolve(ModelCallRole.FLEET_PLAN)
    replan = registry.resolve(ModelCallRole.FLEET_REPLAN)
    assert plan.requested_adapter == replan.requested_adapter == "fleet_planner"
    assert plan.effective_model == "Qwen3-VL-4B-Instruct"
    assert plan.fallback_used
    assert plan.to_dict()["adapter_status"] == "placeholder"


def test_active_uses_served_model_and_requires_existing_path_and_rank(tmp_path: Path) -> None:
    payload = _payload()
    adapter = payload["adapters"]["fleet_planner"]
    adapter.update(
        status="active",
        path=str(tmp_path / "real_adapter"),
        rank=8,
    )
    with pytest.raises(AdapterRegistryError, match="path does not exist"):
        AdapterRegistry(_write(tmp_path, payload))
    (tmp_path / "real_adapter").mkdir()
    with pytest.raises(AdapterRegistryError, match="adapter_config.json"):
        AdapterRegistry(_write(tmp_path, payload))
    (tmp_path / "real_adapter/adapter_config.json").write_text("{}")
    with pytest.raises(AdapterRegistryError, match="safetensors"):
        AdapterRegistry(_write(tmp_path, payload))
    (tmp_path / "real_adapter/adapter_model.safetensors").write_bytes(b"weights")
    selection = AdapterRegistry(_write(tmp_path, payload)).resolve(ModelCallRole.FLEET_PLAN)
    assert selection.effective_model == "Qwen3-VL-4B-Fleet-Planner-LoRA"
    assert not selection.fallback_used


def test_active_path_must_be_an_adapter_directory(tmp_path: Path) -> None:
    payload = _payload()
    adapter_path = tmp_path / "adapter.safetensors"
    adapter_path.write_bytes(b"not an adapter directory")
    payload["adapters"]["fleet_planner"].update(
        status="active",
        path=str(adapter_path),
        rank=8,
    )
    with pytest.raises(AdapterRegistryError, match="not a directory"):
        AdapterRegistry(_write(tmp_path, payload))


def test_active_adapter_rejects_symlink_config_and_weights(tmp_path: Path) -> None:
    payload = _payload()
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    payload["adapters"]["fleet_planner"].update(
        status="active",
        path=str(adapter_path),
        rank=8,
    )

    real_config = tmp_path / "real_adapter_config.json"
    real_config.write_text("{}", encoding="utf-8")
    (adapter_path / "adapter_config.json").symlink_to(real_config)
    (adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
    with pytest.raises(AdapterRegistryError, match="non-symlink adapter_config"):
        AdapterRegistry(_write(tmp_path, payload))

    (adapter_path / "adapter_config.json").unlink()
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_path / "adapter_model.safetensors").unlink()
    real_weights = tmp_path / "real_adapter_model.safetensors"
    real_weights.write_bytes(b"weights")
    (adapter_path / "adapter_model.safetensors").symlink_to(real_weights)
    with pytest.raises(AdapterRegistryError, match="non-symlink .safetensors"):
        AdapterRegistry(_write(tmp_path, payload))


def test_placeholder_cannot_claim_a_speculative_rank(tmp_path: Path) -> None:
    payload = _payload()
    payload["adapters"]["fleet_planner"]["rank"] = 8
    with pytest.raises(AdapterRegistryError, match="rank=null"):
        AdapterRegistry(_write(tmp_path, payload))


def test_adapter_served_names_are_unambiguous(tmp_path: Path) -> None:
    payload = _payload()
    payload["adapters"]["runtime_visual"]["served_model_name"] = payload[
        "adapters"
    ]["fleet_planner"]["served_model_name"]
    with pytest.raises(AdapterRegistryError, match="must be unique"):
        AdapterRegistry(_write(tmp_path, payload))


def test_disabled_adapter_cannot_be_selected(tmp_path: Path) -> None:
    payload = _payload()
    payload["adapters"]["fleet_planner"]["status"] = "disabled"
    registry = AdapterRegistry(_write(tmp_path, payload))
    with pytest.raises(AdapterRegistryError, match="disabled"):
        registry.resolve(ModelCallRole.FLEET_PLAN)


def test_base_model_lineage_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["adapters"]["fleet_planner"]["base_model_name"] = "wrong-base"
    with pytest.raises(AdapterRegistryError, match="lineage"):
        AdapterRegistry(_write(tmp_path, payload))
