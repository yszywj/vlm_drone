from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from scripts import plan_fleet


def _install_fake_planning(monkeypatch, *, status="proposal_ready", during_invoke=None):
    captured: dict[str, object] = {}

    class FakeFactory:
        def __init__(self, registry, **kwargs):
            captured["factory"] = kwargs

    class FakeService:
        def __init__(self, config, factory, **kwargs):
            captured["service"] = kwargs
            captured["uav_ids"] = [uav.id for uav in config.uavs]

    class FakeTool:
        def __init__(self, service):
            pass

        def invoke(self, arguments, *, audit_context):
            captured["arguments"] = arguments
            audit_context["interpreter_proposals"] = [{"accepted": True}]
            captured["factory"]["selection_logger"]({"effective_model": "fake-planning-adapter"})
            captured["factory"]["call_logger"]({"call_role": "MISSION_INTERPRETATION", "prompt_tokens": 17})
            if during_invoke is not None:
                during_invoke()
            return {"status": status, "execution_started": False, "executable": False}

    monkeypatch.setattr(plan_fleet, "ModelClientFactory", FakeFactory)
    monkeypatch.setattr(plan_fleet, "FleetPlanningService", FakeService)
    monkeypatch.setattr(plan_fleet, "FleetPlanningTool", FakeTool)
    return captured


def _forbid_planning(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local-only mode constructed a model planning component")

    monkeypatch.setattr(plan_fleet, "ModelClientFactory", forbidden)
    monkeypatch.setattr(plan_fleet, "FleetPlanningService", forbidden)


def test_validate_only_needs_no_instruction_model_or_isaac(monkeypatch, capsys):
    _forbid_planning(monkeypatch)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "isaacsim" and not name.startswith(("isaacsim.", "omni."))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert plan_fleet.main(["--validate-only"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["result"]["status"] == "validated"
    assert record["model_call_records"] == []
    assert record["result"]["execution_started"] is False


def test_tool_schema_does_not_require_configuration_or_model(monkeypatch, capsys):
    _forbid_planning(monkeypatch)
    monkeypatch.setattr(plan_fleet, "load_config", lambda *args: pytest.fail("schema mode loaded a scene"))
    assert plan_fleet.main(["--tool-schema", "--config", "/missing/config.yaml"]) == 0
    schema = json.loads(capsys.readouterr().out)
    function = schema["function"]
    assert function["name"] == "plan_fleet"
    assert set(function["parameters"]["properties"]) == {"instruction"}
    assert function["parameters"]["additionalProperties"] is False


def test_instruction_file_preserves_text_and_records_audit_and_budgets(monkeypatch, tmp_path, capsys):
    captured = _install_fake_planning(monkeypatch)
    instruction = "无人机A搜索目标i。\n完成后返回各自起点降落。\n"
    source = tmp_path / "mission.txt"
    source.write_text(instruction, encoding="utf-8")
    assert plan_fleet.main([
        "--instruction-file", str(source), "--interpreter-max-tokens", "6144",
        "--fleet-max-tokens", "4096", "--timeout-s", "45",
    ]) == 0
    assert captured["arguments"] == {"instruction": instruction}
    assert captured["service"] == {"interpreter_max_tokens": 6144, "fleet_max_tokens": 4096}
    assert captured["factory"]["timeout_s"] == 45.0
    record = json.loads(capsys.readouterr().out)
    assert record["audit_context"]["interpreter_proposals"] == [{"accepted": True}]
    assert record["adapter_selections"] == [{"effective_model": "fake-planning-adapter"}]
    assert record["model_call_records"][0]["prompt_tokens"] == 17
    assert "api_key" not in record


@pytest.mark.parametrize("status", ["incomplete_assignment", "planning_failed"])
def test_incomplete_or_failed_proposal_returns_two_and_retains_artifact(monkeypatch, tmp_path, status, capsys):
    _install_fake_planning(monkeypatch, status=status)
    output = tmp_path / "new" / "proposal.json"
    assert plan_fleet.main(["--instruction", "搜索目标i", "--output", str(output)]) == 2
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["result"]["status"] == status
    assert record["result"]["executable"] is False
    assert record["audit_context"]["interpreter_proposals"]
    assert capsys.readouterr().out == ""
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_output_is_rejected_before_model_call(monkeypatch, tmp_path, capsys, symlink):
    _forbid_planning(monkeypatch)
    output = tmp_path / "existing.json"
    if symlink:
        output.symlink_to(tmp_path / "missing.json")
    else:
        output.write_text("preserve this", encoding="utf-8")
    assert plan_fleet.main(["--instruction", "搜索目标i", "--output", str(output)]) == 1
    assert "already exists" in capsys.readouterr().err
    if symlink:
        assert output.is_symlink()
    else:
        assert output.read_text(encoding="utf-8") == "preserve this"


def test_concurrent_output_creation_is_not_overwritten(monkeypatch, tmp_path, capsys):
    output = tmp_path / "proposal.json"
    _install_fake_planning(
        monkeypatch, during_invoke=lambda: output.write_text("other writer", encoding="utf-8")
    )
    assert plan_fleet.main(["--instruction", "搜索目标i", "--output", str(output)]) == 1
    assert output.read_text(encoding="utf-8") == "other writer"
    assert list(tmp_path.iterdir()) == [output]
    assert capsys.readouterr().out == ""


def test_real_service_and_tool_write_typed_audit_without_local_planning(monkeypatch, tmp_path):
    from models.base import ModelResponse
    from models.model_client_factory import ModelClientFactory
    from tests.fleet.test_run_fleet_mission_llm_path import (
        INSTRUCTION, _fleet_plan_payload, _task_spec_payload,
    )

    response_names = []

    class FakeClient:
        def __init__(self, *, model, **kwargs):
            self.model = model

        def chat(self, messages, *, options):
            response_names.append(options.response_format.name)
            request = json.loads(messages[1].content)
            if options.response_format.name == "fleet_task_spec_v1":
                payload = _task_spec_payload(request["source_text"])
            elif options.response_format.name == "fleet_mission_plan_v2":
                payload = _fleet_plan_payload(request["trusted_request"])
            else:
                pytest.fail("planning-only CLI invoked a local or visual planner")
            return ModelResponse(json.dumps(payload), self.model, "stop", {"prompt_tokens": 12})

    monkeypatch.setattr(
        plan_fleet, "ModelClientFactory",
        lambda registry, **kwargs: ModelClientFactory(registry, client_factory=FakeClient, **kwargs),
    )
    output = tmp_path / "proposal.json"
    assert plan_fleet.main(["--instruction", INSTRUCTION, "--output", str(output)]) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert response_names == ["fleet_task_spec_v1", "fleet_mission_plan_v2"]
    assert record["result"]["status"] == "proposal_ready"
    assert record["result"]["local_compilation_performed"] is False
    assert record["result"]["execution_started"] is False
    assert record["result"]["completeness"]["source_intent_verified"] is False
    assert record["audit_context"]["task_spec"]["source_text"] == INSTRUCTION
    assert record["audit_context"]["request_v2"]["fleet_mission_id"] == record["result"]["fleet_mission_id"]
    assert [row["call_role"] for row in record["model_call_records"]] == [
        "MISSION_INTERPRETATION", "FLEET_PLAN",
    ]


@pytest.mark.parametrize("argv", [
    [], ["--instruction", "  "],
    ["--validate-only", "--instruction", "任务\x00"],
    ["--instruction", "a", "--instruction-file", "mission.txt"],
    ["--validate-only", "--fleet-max-tokens", "255"],
    ["--validate-only", "--interpreter-max-tokens", "8193"],
    ["--validate-only", "--timeout-s", "nan"],
    ["--validate-only", "--timeout-s", "0"],
    ["--validate-only", "--config", "/missing/config.yaml"],
    ["--validate-only", "--adapter-config", "/missing/adapters.json"],
])
def test_invalid_input_fails_before_model_call(monkeypatch, argv, capsys):
    _forbid_planning(monkeypatch)
    assert plan_fleet.main(argv) == 1
    assert capsys.readouterr().err.startswith("plan_fleet:")
