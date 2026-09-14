"""Offline checks of evaluation aggregation and gold-free inference metadata."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from models.adapter_registry import AdapterRegistry, AdapterRegistryError
from models.base import ChatMessage, GenerationOptions, JsonSchemaResponseFormat, ModelResponse
from models.openai_compatible_client import OpenAICompatibleClient
from scripts import evaluate_planning_roles_lora as runner
from training.lora.planning_chain_eval import RecordingEvaluationClient, generation_options_audit


def test_summary_counts_service_failures_and_blocked_locals_in_denominators():
    roles = [
        {"role": "mission_interpreter", "family": "search", "scale": 2,
         "passed": False, "structural_pass": False, "service_error": {"type": "Timeout"}},
        {"role": "spatial_mission", "family": "search", "scale": 2,
         "passed": True, "structural_pass": True},
        {"role": "spatial_mission", "family": "search", "scale": 2,
         "passed": False, "structural_pass": True, "findings": [{"code": "DURATION_MISMATCH"}],
         "response": {"finish_reason": "length"}},
    ]
    chains = [
        {"family": "search", "scale": 2, "strict_entire_task_pass": False,
         "entire_task_pass_with_runtime_completion": False,
         "local_pass_count": 0, "expected_local_count": 2, "local_blocked_count": 2,
         "client_error_count": 1, "truncated_call_count": 0, "model_call_count": 1},
        {"family": "search", "scale": 2, "strict_entire_task_pass": False,
         "entire_task_pass_with_runtime_completion": True,
         "local_pass_count": 2, "expected_local_count": 2, "local_blocked_count": 0,
         "client_error_count": 0, "truncated_call_count": 0, "model_call_count": 4},
    ]
    summary = runner.summarize(roles, chains)
    assert summary["isolated"]["by_role"]["mission_interpreter"] == {"passed": 0, "total": 1, "rate": 0}
    assert summary["isolated"]["by_role_scale"]["spatial_mission/2"] == {"passed": 1, "total": 2, "rate": 0.5}
    assert summary["isolated"]["structural_by_role"]["spatial_mission"]["passed"] == 2
    assert summary["isolated"]["service_errors"] == 1
    assert summary["isolated"]["truncated"] == 1
    assert summary["chain"]["strict_passed"] == 0
    assert summary["chain"]["passed_with_runtime_completion"] == 1
    assert summary["chain"]["local_expected"] == 4
    assert summary["chain"]["local_passed"] == 2
    assert summary["chain"]["local_blocked"] == 2
    assert summary["chain"]["model_calls"] == 5


def test_empty_summary_is_valid_before_any_result_arrives():
    summary = runner.summarize([], [])
    assert summary["isolated"]["completed"] == 0
    assert summary["isolated"]["by_role"] == {}
    assert summary["chain"]["completed"] == 0
    assert summary["chain"]["local_expected"] == 0


@pytest.fixture(params=[2, 3])
def prepared_fixture(monkeypatch, tmp_path, request):
    """Different complete split sizes; production counts must come from data."""
    tasks = [{"task_id": f"task_{index}", "split": "test", "scale": 2 + 2 * index,
              "uavs": [{"uav_id": f"uav_{i}"} for i in range(2 + 2 * index)]}
             for index in range(request.param)]
    messages = [{"role": "system", "content": "system input"},
                {"role": "user", "content": "trusted user input"},
                {"role": "assistant", "content": '{"gold":"SECRET_GOLD_ANSWER"}'}]
    role_rows = {}
    for role in runner.ROLES:
        role_rows[role] = []
        for task in tasks:
            count = len(task["uavs"]) if role == "spatial_mission" else 1
            for index in range(count):
                uav_id = f"uav_{index}" if role == "spatial_mission" else None
                row = {
                    "dataset_schema_version": 1, "task_id": task["task_id"],
                    "sample_id": f"{task['task_id']}/{role}/{index}",
                    "split": "test", "role": role, "family": "search", "scale": task["scale"],
                    "uav_id": uav_id, "semantic_hash": "task_hash",
                    "instruction_semantic_hash": "instruction_hash",
                    "response_schema_sha256": "schema_hash", "messages": deepcopy(messages),
                    "checks": {"blueprint_semantics": True},
                    "label_origin": "deterministic_program_gold",
                    "future_gold_helper": "SECRET_GOLD_ANSWER",
                }
                if role == "mission_interpreter":
                    # These are added by PlanningRoleSFTDataset.__getitem__.
                    row.update(assistant_json=messages[-1]["content"],
                               target={"gold": "SECRET_GOLD_ANSWER"},
                               request={"input": "trusted user input"},
                               input_json='{"input":"trusted user input"}', output_kind=role)
                role_rows[role].append(row)

    manifest = {
        "task_split_counts": {"test": len(tasks)},
        "role_split_counts": {role: {"test": len(rows)} for role, rows in role_rows.items()},
    }
    class CheckedDataset(list):
        def __init__(self):
            super().__init__(deepcopy(role_rows["mission_interpreter"]))
            self.manifest = deepcopy(manifest)
    monkeypatch.setattr(runner, "PlanningRoleSFTDataset", lambda *args, **kwargs: CheckedDataset())

    def read(path):
        return tasks if path.name == "tasks.jsonl" else deepcopy(role_rows[path.parent.name])

    monkeypatch.setattr(runner, "read_jsonl", read)
    monkeypatch.setattr(runner, "capture_role_request", lambda *args, **kwargs: {
        "messages": [ChatMessage(**message) for message in messages[:-1]],
        "options": _options(), "response_schema_sha256": "schema_hash",
    })
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def test_prepare_retains_only_identity_and_integrity_metadata(prepared_fixture):
    prepared, tasks = runner.prepare(prepared_fixture)
    allowed = {
        "dataset_schema_version", "sample_id", "task_id", "split", "role", "family", "scale",
        "semantic_hash", "instruction_semantic_hash", "uav_id", "response_schema_sha256",
    }
    assert len(tasks) in (2, 3)
    assert len(prepared) == sum(2 + len(task["uavs"]) for task in tasks)
    for _, metadata, captured in prepared:
        assert set(metadata) <= allowed, "gold or loader-helper fields crossed the inference work-item metadata boundary"
        assert "SECRET_GOLD_ANSWER" not in str(metadata)
        assert [message.role for message in captured["messages"]] == ["system", "user"]
        assert "SECRET_GOLD_ANSWER" not in str([message.to_dict() for message in captured["messages"]])
        assert captured["options"].max_tokens == runner.DEFAULT_MAX_TOKENS[metadata["role"]]
        assert captured["options"].temperature == 0


@pytest.mark.parametrize("drift", ("messages", "response_schema_sha256"))
def test_prepare_refuses_prompt_or_schema_drift_before_generation(prepared_fixture, monkeypatch, drift):
    original = runner.capture_role_request

    def changed(*args, **kwargs):
        captured = original(*args, **kwargs)
        captured[drift] = [ChatMessage("system", "changed")] if drift == "messages" else "different_hash"
        return captured

    monkeypatch.setattr(runner, "capture_role_request", changed)
    with pytest.raises(AssertionError, match="production .* drift"):
        runner.prepare(prepared_fixture)


def _options():
    return GenerationOptions(max_tokens=256, response_format=JsonSchemaResponseFormat(
        "test_schema", {"type": "object", "properties": {
            "z": {"type": "integer"}, "a": {"type": "string", "enum": ["甲", "乙"]},
        }, "required": ["z", "a"]},
    ))


@pytest.fixture
def active_registry(tmp_path):
    payload = json.loads((runner.ROOT / "configs/adapters.json").read_text())
    payload["base_model"]["served_model_name"] = "fixture_base"
    for slot, adapter in payload["adapters"].items():
        adapter.update(status="placeholder", path=None, rank=None, base_model_name="fixture_base")
        adapter.pop("generation", None)
        if slot in runner.ROLES:
            directory = tmp_path / slot
            directory.mkdir()
            (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
            (directory / "adapter_model.safetensors").write_bytes(b"test weights")
            adapter.update(status="active", path=str(directory), rank=16,
                           served_model_name=f"fixture_{slot}",
                           generation={"json_schema_property_order": "alphabetical"})
    config = tmp_path / "adapters.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    return AdapterRegistry(config)


@pytest.fixture
def capture_transport(monkeypatch):
    calls = []
    class Transport:
        def __init__(self, base_url=None, model=None, **kwargs):
            self.model = model
            self.base_url = base_url
        def healthcheck(self):
            return None
        def chat(self, messages, *, options):
            calls.append({"model": self.model, "url": self.base_url,
                          "messages": messages, "options": options})
            return ModelResponse("{}", self.model, "stop", {})
    monkeypatch.setattr(runner, "OpenAICompatibleClient", Transport)
    monkeypatch.setattr(runner, "score_role_output", lambda *args, **kwargs: {
        "passed": True, "structural_pass": True, "semantic_pass": True,
    })
    return calls


def test_verified_manifest_counts_are_enforced(prepared_fixture, monkeypatch):
    original = runner.PlanningRoleSFTDataset
    def mismatched(*args, **kwargs):
        checked = original(*args, **kwargs)
        checked.manifest["role_split_counts"]["fleet_planner"]["test"] += 1
        return checked
    monkeypatch.setattr(runner, "PlanningRoleSFTDataset", mismatched)
    with pytest.raises(AssertionError, match="complete test split counts"):
        runner.prepare(prepared_fixture)


def test_production_factory_is_used_and_effective_schema_is_audited(
    prepared_fixture, active_registry, capture_transport, monkeypatch,
):
    prepared, _ = runner.prepare(prepared_fixture)
    factory_calls = []
    original_factory = runner.ModelClientFactory
    def observed_factory(*args, **kwargs):
        factory_calls.append((args, kwargs))
        return original_factory(*args, **kwargs)
    monkeypatch.setattr(runner, "ModelClientFactory", observed_factory)
    original_options = prepared[0][2]["options"]
    results = {}
    for variant in ("base", "lora"):
        clients = runner.build_clients(variant, "http://127.0.0.1:18080/v1", active_registry)
        results[variant] = runner.evaluate_role(prepared[0], clients)
    assert len(factory_calls) == 1
    assert capture_transport[0]["model"] == "fixture_base"
    assert capture_transport[1]["model"] == "fixture_mission_interpreter"
    assert capture_transport[0]["options"] is original_options
    assert capture_transport[0]["url"] == capture_transport[1]["url"]
    for variant, call in zip(("base", "lora"), capture_transport, strict=True):
        generation = results[variant]["generation"]
        expected = ["z", "a"] if variant == "base" else ["a", "z"]
        assert list(generation["response_format"]["schema"]["properties"]) == expected
        assert generation["response_format"] == call["options"].response_format.to_dict()
        saved_options = json.loads(json.loads(json.dumps(results[variant], sort_keys=True))["generation"]["generation_options_json"])
        assert list(saved_options["response_format"]["schema"]["properties"]) == expected
        wire = json.dumps(saved_options["response_format"], ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        assert sha256(wire.encode()).hexdigest() == generation["response_schema_wire_sha256"]
        assert "SECRET_GOLD_ANSWER" not in str(call["messages"])
    assert results["base"]["generation"]["response_schema_sha256"] == results["lora"]["generation"]["response_schema_sha256"]
    assert results["base"]["generation"]["response_schema_wire_sha256"] != results["lora"]["generation"]["response_schema_wire_sha256"]
    assert list(original_options.response_format.to_dict()["schema"]["properties"]) == ["z", "a"]


def test_chain_recorder_logs_options_actually_sent_by_factory(active_registry, capture_transport):
    client = runner.build_clients("lora", "http://127.0.0.1:18081/v1", active_registry)["mission_interpreter"]
    calls = []
    recording = RecordingEvaluationClient(client, "mission_interpreter", 6144, calls)
    recording.chat((ChatMessage("user", "trusted input"),), options=_options())
    assert capture_transport[0]["options"].max_tokens == 6144
    assert calls[0]["options"] == generation_options_audit(capture_transport[0]["options"])
    assert list(calls[0]["options"]["response_format"]["schema"]["properties"]) == ["a", "z"]
    assert calls[0]["response"]["model"] == "fixture_mission_interpreter"


def test_wire_hash_matches_production_http_schema_serialization(monkeypatch):
    client = OpenAICompatibleClient("http://127.0.0.1:18080/v1", "fixture_base", max_retries=0)
    requests = []
    def send(request):
        requests.append(request.data.decode("utf-8"))
        return json.dumps({"model": "fixture_base", "choices": [{
            "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop",
        }], "usage": {}}).encode(), 200
    monkeypatch.setattr(client, "_send", send)
    options = _options()
    client.chat((ChatMessage("user", "plan"),), options=options)
    audit = generation_options_audit(options)
    subtree = json.dumps(audit["response_format"], ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    assert '"json_schema":' + subtree in requests[0]
    assert audit["response_schema_wire_sha256"] == sha256(subtree.encode()).hexdigest()
    assert "\\u7532" in requests[0]


@pytest.mark.parametrize("status", ["placeholder", "disabled"])
def test_lora_group_refuses_missing_active_role(active_registry, status, monkeypatch):
    payload = json.loads(active_registry.config_path.read_text())
    payload["adapters"]["mission_interpreter"].update(status=status, rank=None, path=None)
    active_registry.config_path.write_text(json.dumps(payload))
    registry = AdapterRegistry(active_registry.config_path)
    monkeypatch.setattr(runner, "ModelClientFactory", lambda *args, **kwargs: pytest.fail("factory called before role validation"))
    with pytest.raises(AdapterRegistryError, match="active|disabled"):
        runner.build_clients("lora", "http://127.0.0.1:18081/v1", registry)


def test_role_evaluation_rejects_wrong_response_model(prepared_fixture):
    class WrongModel:
        model = "requested_model"
        def chat(self, messages, *, options):
            return ModelResponse("{}", "different_model", "stop", {})
    item = runner.prepare(prepared_fixture)[0][0]
    with pytest.raises(AssertionError, match="unexpected model routing"):
        runner.evaluate_role(item, {item[1]["role"]: WrongModel()})


def test_main_records_dynamic_counts_and_refuses_changed_resume(
    prepared_fixture, active_registry, tmp_path, monkeypatch,
):
    output = tmp_path / "evaluation"
    invocations = []
    def evaluate(variant, url, destination, prepared, tasks, workers, registry):
        invocations.append((variant, url, len(prepared), len(tasks), registry.config_path))
        return runner.summarize([], [])
    monkeypatch.setattr(runner, "evaluate_variant", evaluate)
    args = ["--dataset", str(prepared_fixture), "--output", str(output),
            "--adapter-config", str(active_registry.config_path), "--base-url", "http://localhost:18080/v1",
            "--lora-url", "http://localhost:18080/v1", "--workers", "2"]
    assert runner.main(args) == 0
    manifest_path = output / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prepared, tasks = runner.prepare(prepared_fixture)
    assert manifest["expected_isolated_samples_per_variant"] == len(prepared)
    assert manifest["expected_chain_tasks_per_variant"] == len(tasks)
    assert manifest["adapter_config_sha256"] == runner.digest(active_registry.config_path)
    assert manifest["base_model"] == "fixture_base"
    assert manifest["lora_models"]["mission_interpreter"] == "fixture_mission_interpreter"
    assert manifest["role_routing"]["base"]["mission_interpreter"]["json_schema_property_order"] == "preserve"
    assert manifest["role_routing"]["lora"]["mission_interpreter"]["json_schema_property_order"] == "alphabetical"
    assert manifest["comparison_scope"] == "base decoding versus configured production LoRA decoding; not weights-only"
    assert {"models/adapter_registry.py", "models/model_client_factory.py", "models/schema_order.py"} <= set(manifest["source_sha256"])
    assert {row[0] for row in invocations} == {"base", "lora"}
    saved_manifest = manifest_path.read_bytes()
    comparison = (output / "comparison.json").read_bytes()
    payload = json.loads(active_registry.config_path.read_text())
    payload["adapters"]["mission_interpreter"]["generation"]["json_schema_property_order"] = "preserve"
    active_registry.config_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="resume refused"):
        runner.main(args)
    assert manifest_path.read_bytes() == saved_manifest
    assert (output / "comparison.json").read_bytes() == comparison
    assert len(invocations) == 2


def test_main_refuses_existing_non_evaluation_output(tmp_path):
    output = tmp_path / "old_report"
    output.mkdir()
    old_file = output / "report.md"
    old_file.write_text("old report", encoding="utf-8")
    with pytest.raises(RuntimeError, match="choose a fresh directory"):
        runner.main(["--dataset", str(tmp_path / "unused"), "--output", str(output)])
    assert old_file.read_text() == "old report"
