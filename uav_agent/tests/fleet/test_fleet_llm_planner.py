from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import json

import pytest

from fleet.llm_planner import LLMFleetPlanner
from fleet.planner_base import FleetPlannerOutputError
from fleet.scripted_planner import ScriptedFleetPlanner
from fleet.types import (
    FleetMissionRequest,
    FleetTargetRequest,
    FleetUavCapability,
)
from models import ChatMessage, GenerationOptions, ModelResponse
from planner.spatial import CircleRegion, CoordinateFrame
from target.types import TargetSpec


class _FakeModelClient:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[
            tuple[tuple[ChatMessage, ...], GenerationOptions | None]
        ] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> object:
        self.calls.append((tuple(messages), options))
        if not self.responses:
            raise AssertionError("unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, ModelResponse):
            return response
        if isinstance(response, str):
            content = response
        else:
            content = json.dumps(response, ensure_ascii=False)
        return ModelResponse(content, "fake-qwen", "stop", {})


class _InvalidResponseClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> object:
        del messages, options
        self.calls += 1
        return object()


def _request() -> FleetMissionRequest:
    return FleetMissionRequest(
        "fleet_mission_llm",
        4,
        "uav_a searches target_i; uav_b searches target_j",
        (
            FleetUavCapability("uav_a", "A", True, "home_a", 5, 30),
            FleetUavCapability("uav_b", "B", True, "home_b", 5, 30),
        ),
        (
            FleetTargetRequest(
                "target_i",
                TargetSpec("target i", immutable_identity_summary="target i"),
                "uav_a",
                CircleRegion(CoordinateFrame.WORLD_ENU, (10, 20, 0), 8),
                20,
            ),
            FleetTargetRequest(
                "target_j",
                TargetSpec("target j", immutable_identity_summary="target j"),
                "uav_b",
                CircleRegion(CoordinateFrame.WORLD_ENU, (30, 40, 0), 9),
                15,
            ),
        ),
    )


def _valid_payload(request: FleetMissionRequest) -> dict[str, object]:
    return ScriptedFleetPlanner().plan(request).to_dict()


def test_llm_planner_uses_one_strict_json_schema_call_and_accepts_valid_plan() -> None:
    request = _request()
    client = _FakeModelClient(_valid_payload(request))
    planner = LLMFleetPlanner(client)
    plan = planner.plan(request)
    assert [(item.uav_id, item.target_alias) for item in plan.assignments] == [
        ("uav_a", "target_i"),
        ("uav_b", "target_j"),
    ]
    assert len(client.calls) == 1
    messages, options = client.calls[0]
    assert messages[0].role == "system"
    assert "Do not generate TAKEOFF" in messages[0].content
    prompt = json.loads(messages[1].content)
    assert prompt["trusted_routing"] == {
        "fleet_mission_id": "fleet_mission_llm",
        "fleet_plan_version": 4,
    }
    assert "images" not in prompt
    assert options is not None and options.temperature == 0.0
    assert options.response_format is not None
    assert options.response_format.name == "fleet_mission_plan_v1"
    schema = options.response_format.schema
    assert schema["properties"]["fleet_mission_id"]["const"] == request.fleet_mission_id
    assert "steps" not in repr(schema).lower()


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (
            lambda payload: payload.update(fleet_plan_version=99),
            "routing/version",
        ),
        (
            lambda payload: payload["assignments"][0].update(uav_id="uav_unknown"),
            "unknown uav_id",
        ),
        (
            lambda payload: payload["assignments"][0].update(
                target_alias="target_unknown"
            ),
            "target allowlist",
        ),
        (
            lambda payload: payload["assignments"][0].update(
                track_duration_s=123
            ),
            "track duration",
        ),
    ),
)
def test_llm_planner_rejects_model_changes_to_trusted_values(
    mutator: object, message: str
) -> None:
    request = _request()
    payload = deepcopy(_valid_payload(request))
    mutator(payload)  # type: ignore[operator]
    with pytest.raises(FleetPlannerOutputError, match=message):
        LLMFleetPlanner(_FakeModelClient(payload)).plan(request)


def test_llm_planner_rejects_duplicates_non_json_and_invalid_response_object() -> None:
    request = _request()
    duplicate_routing = (
        '{"schema_version":1,"fleet_mission_id":"fleet_mission_llm",'
        '"fleet_mission_id":"fleet_mission_changed"}'
    )
    with pytest.raises(FleetPlannerOutputError, match="duplicate JSON field"):
        LLMFleetPlanner(_FakeModelClient(duplicate_routing)).plan(request)
    with pytest.raises(FleetPlannerOutputError, match="invalid FleetMissionPlan"):
        LLMFleetPlanner(_FakeModelClient("```json\n{}\n```" )).plan(request)


def test_llm_planner_rejects_non_model_response_without_recovery_call() -> None:
    request = _request()
    client = _InvalidResponseClient()
    with pytest.raises(FleetPlannerOutputError, match="invalid response object"):
        LLMFleetPlanner(client).plan(request)
    assert client.calls == 1
