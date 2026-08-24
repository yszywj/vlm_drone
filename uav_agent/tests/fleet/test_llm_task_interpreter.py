from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fleet.llm_task_interpreter import (
    FleetTaskInterpretationError,
    LLMFleetTaskInterpreter,
    MAX_PROPOSAL_BYTES,
)
from models.base import ModelResponse

from tests.fleet.test_task_spec import SOURCE, _payload


class _QueuedClient:
    def __init__(self, contents: list[str | object]) -> None:
        self.contents = list(contents)
        self.calls: list[tuple[object, object]] = []

    def chat(self, messages, *, options=None):
        self.calls.append((messages, options))
        item = self.contents.pop(0)
        if isinstance(item, str):
            return ModelResponse(item, "fake", "stop", {})
        return item


def _interpreter(client: object, *, repair_budget: int = 1) -> LLMFleetTaskInterpreter:
    return LLMFleetTaskInterpreter(
        client,  # type: ignore[arg-type]
        uav_alias_catalog={"无人机A": "uav_a", "无人机B": "uav_b"},
        target_alias_catalog={"红色目标i": "target_i", "蓝色目标j": "target_j"},
        supported_coordinate_frames=("WORLD_ENU", "HOME_ENU"),
        repair_budget=repair_budget,
    )


def test_interpreter_uses_temperature_zero_schema_and_does_not_assign_work() -> None:
    client = _QueuedClient([json.dumps(_payload(), ensure_ascii=False)])
    interpreter = _interpreter(client)

    spec = interpreter.interpret(SOURCE)

    assert spec.to_dict() == _payload()
    assert len(client.calls) == 1
    messages, options = client.calls[0]
    assert options.temperature == 0.0
    assert options.response_format.name == "fleet_task_spec_v1"
    assert "do not assign work" in messages[1].content
    assert '("search and track") requires one SEARCH_TARGET' in messages[0].content
    assert "AssignmentConstraint listing every bound goal ID" in messages[0].content
    assert "as a CIRCLE RegionSpec" in messages[0].content
    assert "Put them only in termination_goals" in messages[0].content
    assert interpreter.last_diagnostics.model_calls == 1
    assert not interpreter.last_diagnostics.repair_used
    assert interpreter.model_proposals[0]["accepted"] is True
    assert len(json.dumps(interpreter.model_proposals[0]).encode()) < MAX_PROPOSAL_BYTES


def test_interpreter_repairs_once_and_records_sanitized_diagnostics() -> None:
    invalid = deepcopy(_payload())
    invalid["unknown"] = "field"
    client = _QueuedClient(
        [
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(_payload(), ensure_ascii=False),
        ]
    )
    interpreter = _interpreter(client)

    assert interpreter.interpret(SOURCE).source_text == SOURCE
    diagnostics = interpreter.last_diagnostics
    assert diagnostics.model_calls == 2
    assert diagnostics.repair_used and diagnostics.repair_succeeded
    assert diagnostics.initial_error_code == "TASK_SPEC_VALIDATION_ERROR"
    assert [item["accepted"] for item in interpreter.model_proposals] == [False, True]
    assert interpreter.model_proposals[1]["repair"] is True


def test_interpreter_repairs_termination_action_misplaced_in_goals() -> None:
    invalid = deepcopy(_payload())
    invalid["goals"][0]["goal_type"] = "LAND"
    invalid["goals"][0]["target_alias"] = None
    client = _QueuedClient(
        [
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(_payload(), ensure_ascii=False),
        ]
    )

    spec = _interpreter(client).interpret(SOURCE)

    assert spec.to_dict() == _payload()
    repair_message = client.calls[1][0][-1].content
    assert "termination_goals" in repair_message
    assert "TASK_SPEC_VALIDATION_ERROR" in repair_message


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":NaN}',
        '{"schema_version":Infinity}',
        "not json",
    ],
)
def test_interpreter_strict_json_rejects_duplicates_nan_inf_and_free_text(raw: str) -> None:
    interpreter = _interpreter(_QueuedClient([raw]), repair_budget=0)
    with pytest.raises(FleetTaskInterpretationError):
        interpreter.interpret(SOURCE)
    assert interpreter.last_diagnostics.model_calls == 1
    assert interpreter.model_proposals[0]["accepted"] is False


def test_interpreter_forbidden_proposal_is_not_retained() -> None:
    raw = json.dumps(
        {"schema_version": 1, "oracle_target_pose": [1, 2, 3]},
        ensure_ascii=False,
    )
    interpreter = _interpreter(_QueuedClient([raw]), repair_budget=0)
    with pytest.raises(FleetTaskInterpretationError):
        interpreter.interpret(SOURCE)
    assert interpreter.model_proposals[0]["proposal"] is None
