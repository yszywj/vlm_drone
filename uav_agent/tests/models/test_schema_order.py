from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from models.adapter_registry import AdapterRegistry, AdapterRegistryError, ModelCallRole
from models.base import ChatMessage, GenerationOptions, JsonSchemaResponseFormat, ModelResponse
from models.model_client_factory import ModelClientFactory
from models.schema_order import JsonSchemaPropertyOrder, apply_json_schema_property_order


def _schema() -> dict[str, object]:
    nested = {
        "type": "object",
        "properties": {"z": {"type": "string"}, "a": {"type": "integer"}},
        "required": ["z", "a"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "z": {"type": "array", "items": nested},
            "properties": {"oneOf": [nested, {"type": "null"}]},
            "a": {"type": "string", "enum": ["z", "a"]},
        },
        "required": ["z", "properties", "a"],
        "$defs": {"z": nested, "a": {"type": "string"}},
        "default": {"properties": {"z": 1, "a": 2}},
        "const": {"properties": {"z": 1, "a": 2}},
    }


def _options() -> GenerationOptions:
    return GenerationOptions(
        temperature=0.2, max_tokens=2048, top_p=0.9,
        response_format=JsonSchemaResponseFormat("planning", _schema()),
    )


def _payload() -> dict[str, object]:
    role_to_slot = {
        ModelCallRole.MISSION_INTERPRETATION: "mission_interpreter",
        ModelCallRole.FLEET_PLAN: "fleet_planner",
        ModelCallRole.FLEET_REPLAN: "fleet_planner",
        ModelCallRole.AGENT_SPATIAL_PLAN: "spatial_mission",
        ModelCallRole.RUNTIME_VISUAL_REVIEW: "runtime_visual",
        ModelCallRole.RUNTIME_REPLAN: "runtime_replanner",
    }
    return {
        "schema_version": 1,
        "base_model": {"served_model_name": "base"},
        "fallback_to_base": True,
        "adapters": {
            slot: {"status": "placeholder", "served_model_name": slot,
                   "path": None, "base_model_name": "base", "rank": None}
            for slot in dict.fromkeys(role_to_slot.values())
        },
        "routing": {role.value: slot for role, slot in role_to_slot.items()},
    }


def _registry(tmp_path: Path, payload: dict[str, object]) -> AdapterRegistry:
    path = tmp_path / "adapters.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return AdapterRegistry(path)


def _activate(tmp_path: Path, payload: dict[str, object], slot: str) -> None:
    directory = tmp_path / slot
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"fake test weights")
    payload["adapters"][slot].update(status="active", path=str(directory), rank=16)


def test_recursive_order_preserves_semantics_and_significant_sequences() -> None:
    original = _options()
    original_bytes = json.dumps(original.response_format.to_dict())
    actual = apply_json_schema_property_order(original, JsonSchemaPropertyOrder.ALPHABETICAL)
    original_schema = original.response_format.to_dict()["schema"]
    schema = actual.response_format.to_dict()["schema"]

    assert actual is not original
    assert (actual.temperature, actual.top_p, actual.max_tokens) == (0.2, 0.9, 2048)
    assert actual.response_format.name == original.response_format.name
    assert schema == original_schema
    assert json.dumps(schema, sort_keys=True) == json.dumps(original_schema, sort_keys=True)
    assert list(schema) == list(original_schema)
    assert list(schema["properties"]) == ["a", "properties", "z"]
    assert list(schema["properties"]["z"]["items"]["properties"]) == ["a", "z"]
    alternatives = schema["properties"]["properties"]["oneOf"]
    assert list(alternatives[0]["properties"]) == ["a", "z"]
    assert alternatives[1] == {"type": "null"}
    assert schema["required"] == ["z", "properties", "a"]
    assert alternatives[0]["required"] == ["z", "a"]
    assert schema["properties"]["a"]["enum"] == ["z", "a"]
    assert list(schema["$defs"]) == ["z", "a"]
    assert list(schema["$defs"]["z"]["properties"]) == ["a", "z"]
    for literal in ("default", "const"):
        assert json.dumps(schema[literal]) == json.dumps(original_schema[literal])
    assert json.dumps(original.response_format.to_dict()) == original_bytes
    schema["properties"].clear()
    assert len(actual.response_format.to_dict()["schema"]["properties"]) == 3


@pytest.mark.parametrize("options", [None, GenerationOptions(), _options()])
def test_preserve_returns_original_options(options: GenerationOptions | None) -> None:
    assert apply_json_schema_property_order(options, JsonSchemaPropertyOrder.PRESERVE) is options


def test_alphabetical_free_text_keeps_options_unchanged() -> None:
    options = GenerationOptions(max_tokens=17)
    assert apply_json_schema_property_order(options, JsonSchemaPropertyOrder.ALPHABETICAL) is options
    assert apply_json_schema_property_order(None, JsonSchemaPropertyOrder.ALPHABETICAL) is None


@pytest.mark.parametrize("generation", [None, [], "alphabetical", {}, {"unknown": "preserve"},
    {"json_schema_property_order": None}, {"json_schema_property_order": True},
    {"json_schema_property_order": 1}, {"json_schema_property_order": []},
    {"json_schema_property_order": {}}, {"json_schema_property_order": "sorted"},
    {"json_schema_property_order": " alphabetical"},
    {"json_schema_property_order": "preserve", "unknown": True}])
def test_registry_rejects_invalid_generation_configuration(tmp_path: Path, generation: object) -> None:
    payload = _payload()
    payload["adapters"]["fleet_planner"]["generation"] = generation
    with pytest.raises(AdapterRegistryError, match="generation"):
        _registry(tmp_path, payload)


def test_omitted_generation_keeps_existing_configs_valid(tmp_path: Path) -> None:
    registry = _registry(tmp_path, _payload())
    assert all(adapter.json_schema_property_order is JsonSchemaPropertyOrder.PRESERVE
               for adapter in registry.adapters.values())


@pytest.mark.parametrize("status", ["placeholder", "disabled"])
def test_non_active_adapter_never_applies_specific_generation(tmp_path: Path, status: str) -> None:
    payload = _payload()
    payload["adapters"]["fleet_planner"].update(
        status=status, generation={"json_schema_property_order": "alphabetical"})
    registry = _registry(tmp_path, payload)
    assert registry.adapters["fleet_planner"].json_schema_property_order is JsonSchemaPropertyOrder.ALPHABETICAL
    if status == "disabled":
        with pytest.raises(AdapterRegistryError, match="disabled"):
            registry.resolve(ModelCallRole.FLEET_PLAN)
    else:
        selection = registry.resolve(ModelCallRole.FLEET_PLAN)
        assert selection.effective_model == "base"
        assert selection.fallback_used
        assert selection.json_schema_property_order is JsonSchemaPropertyOrder.PRESERVE
        assert selection.to_dict()["json_schema_property_order"] == "preserve"


class _CaptureClient:
    def __init__(self, **kwargs: object) -> None:
        self.model = kwargs["model"]
        self.calls = []

    def healthcheck(self) -> None:
        return None

    def chat(self, messages, *, options=None) -> ModelResponse:
        self.calls.append((messages, options))
        return ModelResponse("{}", str(self.model), "stop", {})


@pytest.mark.parametrize("logging_enabled", [False, True])
def test_active_policy_is_applied_per_role_without_mutating_shared_options(
    tmp_path: Path, logging_enabled: bool,
) -> None:
    payload = _payload()
    for slot in ("fleet_planner", "spatial_mission"):
        _activate(tmp_path, payload, slot)
    for slot in ("fleet_planner", "mission_interpreter"):
        payload["adapters"][slot]["generation"] = {"json_schema_property_order": "alphabetical"}
    created = []
    def make_client(**kwargs):
        client = _CaptureClient(**kwargs)
        created.append(client)
        return client
    records = []
    factory = ModelClientFactory(
        _registry(tmp_path, payload), client_factory=make_client,
        call_logger=records.append if logging_enabled else None,
    )
    roles = [ModelCallRole.FLEET_PLAN, ModelCallRole.AGENT_SPATIAL_PLAN,
             ModelCallRole.RUNTIME_VISUAL_REVIEW, ModelCallRole.MISSION_INTERPRETATION] * 5
    clients = [factory.for_role(role) for role in roles]
    messages = (ChatMessage("user", "原提示不变"),)
    options = _options()
    original_bytes = json.dumps(options.response_format.to_dict())
    prepared = clients[0].prepare_options(options)
    assert list(prepared.response_format.to_dict()["schema"]["properties"]) == ["a", "properties", "z"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda client: client.chat(messages, options=options), clients))
    for role, client in zip(roles, created, strict=True):
        actual_messages, actual_options = client.calls[0]
        assert actual_messages is messages
        actual_order = list(actual_options.response_format.to_dict()["schema"]["properties"])
        if role is ModelCallRole.FLEET_PLAN:
            assert actual_order == ["a", "properties", "z"]
            assert actual_options is not options
        else:
            assert actual_order == ["z", "properties", "a"]
            assert actual_options is options
    assert json.dumps(options.response_format.to_dict()) == original_bytes
    assert len(records) == (len(roles) if logging_enabled else 0)
    for record in records:
        expected = "alphabetical" if record["call_role"] == "FLEET_PLAN" else "preserve"
        assert record["json_schema_property_order"] == expected
        effective = record["generation_options"]
        assert effective["max_tokens"] == options.max_tokens
        assert effective["temperature"] == options.temperature
        assert effective["top_p"] == options.top_p
        properties = list(effective["response_format"]["schema"]["properties"])
        assert properties == (["a", "properties", "z"] if expected == "alphabetical"
                              else ["z", "properties", "a"])
        saved = json.loads(json.dumps(record, sort_keys=True))
        saved_options = json.loads(saved["generation_options_json"])
        assert saved_options == effective
        assert list(saved_options["response_format"]["schema"]["properties"]) == properties


def test_failed_call_audits_effective_order_and_options(tmp_path: Path) -> None:
    payload = _payload()
    _activate(tmp_path, payload, "fleet_planner")
    payload["adapters"]["fleet_planner"]["generation"] = {"json_schema_property_order": "alphabetical"}
    class FailingClient(_CaptureClient):
        def chat(self, messages, *, options=None):
            raise TimeoutError("test timeout")
    records = []
    factory = ModelClientFactory(_registry(tmp_path, payload), client_factory=FailingClient,
                                 call_logger=records.append)
    with pytest.raises(TimeoutError, match="test timeout"):
        factory.for_role(ModelCallRole.FLEET_PLAN).chat((ChatMessage("user", "plan"),), options=_options())
    assert records[0]["error_code"] == "TimeoutError"
    assert records[0]["json_schema_property_order"] == "alphabetical"
    assert list(records[0]["generation_options"]["response_format"]["schema"]["properties"]) == ["a", "properties", "z"]
