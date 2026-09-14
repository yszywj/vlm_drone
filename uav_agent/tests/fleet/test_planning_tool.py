from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from fleet.planning_tool import FleetPlanningTool, PlanningToolInputError


class _Service:
    def __init__(self, *, complete=True, error=None):
        self.calls = []
        self.complete = complete
        self.error = error

    def plan(self, instruction, **kwargs):
        self.calls.append((instruction, kwargs))
        if self.error is not None:
            raise self.error
        payload = {
            "task_spec": {"source_text": instruction},
            "plan": {"fleet_mission_id": kwargs["fleet_mission_id"]},
            "semantic_findings": [],
            "completeness": {
                "assignment_complete": self.complete,
                "scope": "interpreted_task_spec_assignment",
                "source_intent_verified": False,
                "local_plans_validated": False,
            },
            "interpreter_proposals": [{"raw_response": "host audit only"}],
            "request": {"trusted_state": "host audit only"},
        }
        return SimpleNamespace(to_dict=lambda: deepcopy(payload))


@pytest.mark.parametrize("extra", [
    "config", "config_path", "adapter", "model", "base_url", "fleet_state",
    "fleet_mission_id", "fleet_plan_version", "execute",
])
def test_tool_caller_cannot_override_host_configuration(extra):
    service = _Service()
    tool = FleetPlanningTool(service)
    with pytest.raises(PlanningToolInputError):
        tool.invoke({"instruction": "搜索目标后返航", extra: "caller supplied"})
    assert service.calls == []


@pytest.mark.parametrize("arguments", [
    '{"instruction":"first","instruction":"second"}',
    '{"instruction":NaN}', "[]", "null", {}, None,
    {"instruction": " "}, {"instruction": "x" * 8193}, {"instruction": "x\x00"},
])
def test_invalid_arguments_fail_before_any_planning_call(arguments):
    service = _Service()
    with pytest.raises(PlanningToolInputError):
        FleetPlanningTool(service).invoke(arguments)
    assert not service.calls


@pytest.mark.parametrize("arguments", [
    '{"instruction":' + '[' * 2000 + '0' + ']' * 2000 + '}',
    '\ud800',
])
def test_malformed_unicode_and_deep_json_return_invalid_arguments(arguments):
    service = _Service()
    reply = FleetPlanningTool(service).handle_tool_call({
        "id": "call_bad", "type": "function",
        "function": {"name": "plan_fleet", "arguments": arguments},
    })
    assert json.loads(reply["content"])["status"] == "invalid_arguments"
    assert service.calls == []


def test_tool_uses_original_instruction_and_host_generated_unique_mission_ids():
    service = _Service()
    tool = FleetPlanningTool(service)
    instruction = "无人机B跟踪目标A 20秒；无人机A跟踪目标B 30秒，各自返航降落。"
    first = tool.invoke({"instruction": instruction})
    second = tool.invoke(json.dumps({"instruction": instruction}))
    assert [call[0] for call in service.calls] == [instruction, instruction]
    assert first["fleet_mission_id"] != second["fleet_mission_id"]
    assert first["fleet_plan"]["fleet_mission_id"] == first["fleet_mission_id"]
    assert first["status"] == "proposal_ready"
    assert first["execution_started"] is False
    assert first["executable"] is False
    assert first["local_compilation_performed"] is False
    assert first["completeness"]["source_intent_verified"] is False
    assert "request" not in first
    assert "interpreter_proposals" not in first


def test_tool_does_not_label_partial_goal_coverage_ready():
    result = FleetPlanningTool(_Service(complete=False)).invoke({"instruction": "搜索全部目标"})
    assert result["status"] == "incomplete_assignment"
    assert result["executable"] is False


def test_model_failure_is_returned_without_exception_secrets():
    service = _Service(error=RuntimeError("Bearer private-key at http://private-endpoint"))
    result = FleetPlanningTool(service).invoke({"instruction": "搜索目标"})
    assert result["status"] == "planning_failed"
    assert result["failed_stage"] == "mission_interpretation"
    assert result["error_type"] == "RuntimeError"
    assert "private-key" not in json.dumps(result)


def test_tool_reports_explicit_service_stage_and_preserves_original_text():
    class _FailingFleet(_Service):
        def plan(self, instruction, **kwargs):
            kwargs["audit_context"]["planning_stage"] = "fleet_assignment"
            raise ValueError("invalid proposal")

    audit = {}
    result = FleetPlanningTool(_FailingFleet()).invoke(
        {"instruction": "  搜索全部目标后返航\n"}, audit_context=audit,
    )
    assert result["failed_stage"] == "fleet_assignment"
    assert audit["raw_instruction"] == "  搜索全部目标后返航\n"
    assert audit["source_text"] == "搜索全部目标后返航"


def test_chat_tool_reply_is_correlated_and_invalid_arguments_are_repairable():
    service = _Service()
    tool = FleetPlanningTool(service)
    call = {
        "type": "function", "id": "call_001",
        "function": {"name": "plan_fleet", "arguments": '{"instruction":"搜索目标"}'},
    }
    reply = tool.handle_tool_call(call)
    assert reply["role"] == "tool"
    assert reply["tool_call_id"] == "call_001"
    assert json.loads(reply["content"])["status"] == "proposal_ready"
    call["function"]["arguments"] = '{"instruction":"搜索目标","execute":true}'
    reply = tool.handle_tool_call(call)
    assert json.loads(reply["content"])["status"] == "invalid_arguments"
    assert len(service.calls) == 1


def test_unknown_tool_never_dispatches_planning():
    service = _Service()
    with pytest.raises(PlanningToolInputError):
        FleetPlanningTool(service).handle_tool_call({
            "id": "call_001", "type": "function",
            "function": {"name": "execute_fleet", "arguments": "{}"},
        })
    assert service.calls == []


def test_tool_definition_is_instruction_only_and_returned_independently():
    definition = FleetPlanningTool.tool_definition()
    parameters = definition["function"]["parameters"]
    assert set(parameters["properties"]) == {"instruction"}
    assert parameters["additionalProperties"] is False
    parameters["properties"]["execute"] = {"type": "boolean"}
    assert "execute" not in FleetPlanningTool.tool_definition()["function"]["parameters"]["properties"]
