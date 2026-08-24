from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from models.adapter_registry import AdapterRegistry, ModelCallRole
from models.base import ChatMessage, ModelResponse
from models.model_client_factory import ModelClientFactory


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Client:
    def __init__(self, **kwargs: object) -> None:
        self.model = kwargs["model"]
        self.kwargs = kwargs


class _ChatClient(_Client):
    def healthcheck(self) -> None:
        return None

    def chat(self, messages, *, options=None) -> ModelResponse:
        return ModelResponse(
            content='{"ok":true}',
            model=str(self.model),
            finish_reason="stop",
            usage={"prompt_tokens": 17, "completion_tokens": 4},
        )


def test_factory_returns_new_role_bound_clients_and_logs_selection(tmp_path: Path) -> None:
    payload = json.loads((PROJECT_ROOT / "configs/adapters.json").read_text())
    for name, rank in (("fleet_planner", 8), ("spatial_mission", 16)):
        path = tmp_path / name
        path.mkdir()
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"test-weights")
        payload["adapters"][name].update(status="active", path=str(path), rank=rank)
    config = tmp_path / "adapters.json"
    config.write_text(json.dumps(payload))
    logs: list[dict[str, object]] = []
    factory = ModelClientFactory(
        AdapterRegistry(config),
        base_url="http://127.0.0.1:8000/v1",
        api_key="secret-not-logged",
        client_factory=_Client,
        selection_logger=logs.append,
    )

    roles = [ModelCallRole.FLEET_PLAN, ModelCallRole.AGENT_SPATIAL_PLAN] * 10
    with ThreadPoolExecutor(max_workers=4) as pool:
        clients = list(pool.map(factory.for_role, roles))

    assert len({id(client) for client in clients}) == len(clients)
    for role, client in zip(roles, clients, strict=True):
        expected = (
            "Qwen3-VL-4B-Fleet-Planner-LoRA"
            if role is ModelCallRole.FLEET_PLAN
            else "Qwen3-VL-4B-Spatial-Mission-LoRA"
        )
        assert client.model == expected
    assert all("effective_model" in row and "requested_adapter" in row for row in logs)
    assert all("secret-not-logged" not in json.dumps(row) for row in logs)


def test_factory_records_real_call_usage_latency_and_trusted_routing() -> None:
    calls: list[dict[str, object]] = []
    factory = ModelClientFactory(
        AdapterRegistry(PROJECT_ROOT / "configs/adapters.json"),
        base_url="http://127.0.0.1:8000/v1",
        api_key="secret-not-logged",
        client_factory=_ChatClient,
        call_logger=calls.append,
    )
    client = factory.for_role(
        ModelCallRole.AGENT_SPATIAL_PLAN,
        fleet_mission_id="fleet_mission_test",
        assignment_id="assignment_uav_a_target_i",
        uav_id="uav_a",
    )

    response = client.chat((ChatMessage("user", "plan"),))

    assert response.finish_reason == "stop"
    assert len(calls) == 1
    record = calls[0]
    assert record["call_id"] == "model_call_00000001"
    assert record["call_role"] == "AGENT_SPATIAL_PLAN"
    assert record["fleet_mission_id"] == "fleet_mission_test"
    assert record["assignment_id"] == "assignment_uav_a_target_i"
    assert record["uav_id"] == "uav_a"
    assert record["requested_adapter"] == "spatial_mission"
    assert record["adapter_status"] == "placeholder"
    assert record["effective_model"] == "Qwen3-VL-4B-Instruct"
    assert record["fallback_used"] is True
    assert record["prompt_tokens"] == 17
    assert record["completion_tokens"] == 4
    assert record["finish_reason"] == "stop"
    assert record["error_code"] is None
    assert record["latency_s"] >= 0.0
    assert "secret-not-logged" not in json.dumps(record)


def test_factory_supports_distinct_trusted_call_id_namespace() -> None:
    calls: list[dict[str, object]] = []
    factory = ModelClientFactory(
        AdapterRegistry(PROJECT_ROOT / "configs/adapters.json"),
        client_factory=_ChatClient,
        call_logger=calls.append,
        call_id_prefix="runtime_model_call",
    )
    client = factory.for_role(ModelCallRole.RUNTIME_VISUAL_REVIEW)

    client.chat((ChatMessage("user", "review"),))

    assert calls[0]["call_id"] == "runtime_model_call_00000001"


@pytest.mark.parametrize("prefix", ("", "bad prefix", "x" * 56, 1))
def test_factory_rejects_untrusted_call_id_prefix(prefix: object) -> None:
    with pytest.raises((TypeError, ValueError), match="call_id_prefix"):
        ModelClientFactory(
            AdapterRegistry(PROJECT_ROOT / "configs/adapters.json"),
            client_factory=_Client,
            call_id_prefix=prefix,  # type: ignore[arg-type]
        )
