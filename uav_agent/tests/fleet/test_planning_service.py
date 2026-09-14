from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from configs.loader import load_config
from fleet.llm_task_interpreter import FleetTaskInterpretationError
from fleet.planner_base import FleetPlannerOutputError
from fleet.planning_service import FleetPlanningService
from fleet.request_builder import build_fleet_mission_request_v2
from fleet.task_spec import (
    AssignmentConstraint,
    FleetTaskSpecV1,
    MissionGoal,
    OrderingConstraint,
    TerminationGoal,
)
from fleet.types_v2 import TrustedFleetStateEvidence
from models.adapter_registry import ModelCallRole
from models.base import ModelResponse
from planner.spatial import CircleRegion


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "无人机A在世界坐标(20,30)半径15米搜索目标i，跟踪10秒后返航降落。"


def _task_spec() -> FleetTaskSpecV1:
    return FleetTaskSpecV1(
        source_text=SOURCE,
        goals=(
            MissionGoal(
                "goal_search_i", "SEARCH_TARGET", "target_i",
                CircleRegion("WORLD_ENU", (20.0, 30.0, 0.0), 15.0),
                None, None, "MUST",
            ),
            MissionGoal(
                "goal_track_i", "TRACK_TARGET", "target_i", None, 10.0, None, "MUST",
            ),
        ),
        assignment_constraints=(
            AssignmentConstraint(
                "constraint_a", "uav_a", ("goal_search_i", "goal_track_i", "goal_home"),
                "MUST",
            ),
        ),
        ordering_constraints=(
            OrderingConstraint("order_search_track", "goal_search_i", "goal_track_i", "MUST"),
            OrderingConstraint("order_track_home", "goal_track_i", "goal_home", "MUST"),
        ),
        termination_goals=(
            TerminationGoal("goal_home", "RETURN_HOME_AND_LAND", "uav_a", None, "MUST"),
        ),
    )


class _Client:
    def __init__(self, response: Callable) -> None:
        self.response = response
        self.calls = []

    def chat(self, messages, *, options=None):
        self.calls.append((messages, options))
        value = self.response(messages)
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return ModelResponse(content, "fake", "stop", {})


def _fleet_response(messages):
    request = json.loads(messages[1].content)["trusted_request"]
    return {
        "schema_version": 2,
        "fleet_mission_id": request["fleet_mission_id"],
        "fleet_plan_version": request["fleet_plan_version"],
        "assignments": [{
            "assignment_id": "assignment_a",
            "uav_id": "uav_a",
            "goal_ids": ["goal_search_i", "goal_track_i", "goal_home"],
            "priority": 100,
            "start_policy": "PARALLEL",
            "deviations": [],
        }],
        "coordination_policy": request["coordination_policy"],
        "assumptions": [],
        "unassigned_goal_ids": [],
    }


class _Clients:
    def __init__(self, *, interpreter_response=None, fleet_response=None) -> None:
        self.interpreter = _Client(interpreter_response or (lambda _: _task_spec().to_dict()))
        self.fleet = _Client(fleet_response or _fleet_response)
        self.selections = []

    def for_role(self, role, **routing):
        self.selections.append((role, routing))
        return self.interpreter if role is ModelCallRole.MISSION_INTERPRETATION else self.fleet


def _config():
    return load_config(ROOT / "configs/multi_uav_demo.yaml")


def test_service_routes_two_phases_and_preserves_instruction_and_audit() -> None:
    config = _config()
    clients = _Clients()
    audit = {"model_call_records": [], "local_planner_proposals": {}}
    service = FleetPlanningService(
        config, clients, interpreter_max_tokens=6144, fleet_max_tokens=4096
    )

    result = service.plan(f"  {SOURCE}\n", fleet_mission_id="mission_service", audit_context=audit)

    assert result.task_spec == _task_spec()
    assert result.request.task_spec is result.task_spec
    assert result.plan.fleet_mission_id == "mission_service"
    assert clients.selections == [
        (ModelCallRole.MISSION_INTERPRETATION, {"fleet_mission_id": "mission_service"}),
        (ModelCallRole.FLEET_PLAN, {"fleet_mission_id": "mission_service"}),
    ]
    assert clients.interpreter.calls[0][1].max_tokens == 6144
    assert clients.fleet.calls[0][1].max_tokens == 4096
    interpreter_prompt = clients.interpreter.calls[0][0][1].content
    assert config.uavs[0].display_name in interpreter_prompt
    assert config.targets[0].semantic_alias in interpreter_prompt
    assert "initial_position_xyz_m" not in interpreter_prompt
    assert "initial_position_xyz_m" not in clients.fleet.calls[0][0][1].content
    assert audit["task_spec"] is result.task_spec
    assert audit["request_v2"] is result.request
    assert audit["plan_v2"] is result.plan
    assert audit["interpreter_proposals"][0]["accepted"] is True
    assert audit["fleet_planner_proposals"][0]["accepted"] is True
    assert audit["interpreter_diagnostics"]["model_calls"] == 1
    assert audit["fleet_planner_diagnostics"]["model_calls"] == 1
    assert audit["planning_stage"] == "completed"
    assert audit["model_call_records"] == []
    encoded = json.loads(json.dumps(result.to_dict()))
    assert encoded["completeness"]["assignment_complete"] is True
    assert encoded["completeness"]["source_intent_verified"] is False
    assert encoded["completeness"]["local_plans_validated"] is False


@pytest.mark.parametrize("declared_unassigned", [False, True])
def test_partial_assignment_is_returned_with_uncovered_goals(declared_unassigned) -> None:
    def partial(messages):
        payload = _fleet_response(messages)
        payload["assignments"][0]["goal_ids"].remove("goal_home")
        if declared_unassigned:
            payload["unassigned_goal_ids"] = ["goal_home"]
        return payload

    clients = _Clients(fleet_response=partial)
    result = FleetPlanningService(_config(), clients).plan(
        SOURCE, fleet_mission_id="mission_partial"
    )

    assert result.uncovered_goal_ids == ("goal_home",)
    assert not result.assignment_complete
    assert result.plan.assignments[0].goal_ids == ("goal_search_i", "goal_track_i")
    assert len(clients.fleet.calls) == 1  # Semantic findings remain recoverable.
    codes = {row["code"] for row in result.semantic_findings}
    assert "CONSTRAINED_GOAL_UNASSIGNED" in codes
    assert ("UNACCOUNTED_GOAL" in codes) is not declared_unassigned


def test_declared_unassigned_open_goal_is_incomplete_even_without_findings() -> None:
    task_spec = replace(_task_spec(), assignment_constraints=())

    def partial(messages):
        payload = _fleet_response(messages)
        payload["assignments"][0]["goal_ids"].remove("goal_home")
        payload["unassigned_goal_ids"] = ["goal_home"]
        return payload

    result = FleetPlanningService(
        _config(),
        _Clients(interpreter_response=lambda _: task_spec.to_dict(), fleet_response=partial),
    ).plan(SOURCE, fleet_mission_id="mission_open_unassigned")

    assert result.semantic_findings == ()
    assert result.uncovered_goal_ids == ("goal_home",)
    assert not result.assignment_complete


def test_wrong_owner_is_recoverable_but_not_a_complete_assignment() -> None:
    def deviated(messages):
        payload = _fleet_response(messages)
        payload["assignments"][0]["uav_id"] = "uav_b"
        return payload

    result = FleetPlanningService(_config(), _Clients(fleet_response=deviated)).plan(
        SOURCE, fleet_mission_id="mission_wrong_owner"
    )

    assert result.uncovered_goal_ids == ()
    assert not result.assignment_complete
    assert {row["code"] for row in result.semantic_findings} == {
        "UNEXPLAINED_ASSIGNMENT_DEVIATION"
    }


def test_interpreter_exhaustion_keeps_all_failed_proposals_without_fleet_call() -> None:
    clients = _Clients(interpreter_response=lambda _: "{")
    audit = {}

    with pytest.raises(FleetTaskInterpretationError):
        FleetPlanningService(_config(), clients).plan(
            SOURCE, fleet_mission_id="mission_bad_interpretation", audit_context=audit
        )

    assert len(audit["interpreter_proposals"]) == 2
    assert not any(row["accepted"] for row in audit["interpreter_proposals"])
    assert audit["interpreter_diagnostics"]["final_output_valid"] is False
    assert audit["planning_stage"] == "mission_interpretation"
    assert "task_spec" not in audit
    assert not clients.fleet.calls
    assert [role for role, _ in clients.selections] == [ModelCallRole.MISSION_INTERPRETATION]


def test_fleet_exhaustion_keeps_interpretation_request_and_failed_proposals() -> None:
    clients = _Clients(fleet_response=lambda _: "{")
    audit = {}

    with pytest.raises(FleetPlannerOutputError):
        FleetPlanningService(_config(), clients).plan(
            SOURCE, fleet_mission_id="mission_bad_fleet", audit_context=audit
        )

    assert audit["task_spec"] == _task_spec()
    assert audit["request_v2"].task_spec == _task_spec()
    assert len(audit["fleet_planner_proposals"]) == 3
    assert not any(row["accepted"] for row in audit["fleet_planner_proposals"])
    assert audit["fleet_planner_diagnostics"]["final_output_valid"] is False
    assert audit["planning_stage"] == "fleet_assignment"
    assert "plan_v2" not in audit


def test_model_transport_exception_propagates_unchanged_and_audits_prior_phase() -> None:
    failure = RuntimeError("model transport failed")

    def unavailable(_):
        raise failure

    clients = _Clients(fleet_response=unavailable)
    audit = {}
    with pytest.raises(RuntimeError) as caught:
        FleetPlanningService(_config(), clients).plan(
            SOURCE, fleet_mission_id="mission_unavailable", audit_context=audit
        )

    assert caught.value is failure
    assert audit["interpreter_proposals"][0]["accepted"] is True
    assert audit["fleet_planner_proposals"] == ()
    assert audit["fleet_planner_diagnostics"] is None


@pytest.mark.parametrize("replan", [False, True])
def test_plan_request_preserves_trusted_state_and_skips_interpretation(replan) -> None:
    config = _config()
    evidence = TrustedFleetStateEvidence(
        "evidence_unavailable", "UAV_UNAVAILABLE", "uav_b unavailable", "uav_b", None
    )
    request = build_fleet_mission_request_v2(
        config, _task_spec(), fleet_mission_id="mission_runtime", fleet_plan_version=4,
        trusted_fleet_state=(evidence,),
    )
    clients = _Clients()
    result = FleetPlanningService(config, clients).plan_request(
        request, replan=replan, assignment_id="assignment_origin", uav_id="uav_a"
    )

    assert result.request is request
    assert result.plan.fleet_plan_version == 4
    assert result.interpreter_diagnostics is None
    assert result.interpreter_proposals == ()
    assert not clients.interpreter.calls
    assert clients.selections == [(
        ModelCallRole.FLEET_REPLAN if replan else ModelCallRole.FLEET_PLAN,
        {
            "fleet_mission_id": "mission_runtime",
            "assignment_id": "assignment_origin",
            "uav_id": "uav_a",
        },
    )]
    sent_request = json.loads(clients.fleet.calls[0][0][1].content)["trusted_request"]
    assert sent_request["trusted_fleet_state"] == [evidence.to_dict()]


def test_serialized_audit_is_detached_from_result() -> None:
    result = FleetPlanningService(_config(), _Clients()).plan(
        SOURCE, fleet_mission_id="mission_serialization"
    )
    original = deepcopy(result.to_dict())
    serialized = result.to_dict()
    serialized["interpreter_proposals"][0]["accepted"] = False
    serialized["fleet_planner_proposals"][0]["accepted"] = False
    serialized["plan"]["assignments"].clear()
    assert result.to_dict() == original


def test_reused_audit_drops_success_artifacts_on_new_interpreter_failure() -> None:
    clients = _Clients()
    service = FleetPlanningService(_config(), clients)
    model_calls = []
    local_proposals = {}
    audit = {"model_call_records": model_calls, "local_planner_proposals": local_proposals}
    service.plan(SOURCE, fleet_mission_id="mission_old_success", audit_context=audit)
    clients.interpreter.response = lambda _: "{"

    with pytest.raises(FleetTaskInterpretationError):
        service.plan(SOURCE, fleet_mission_id="mission_new_failure", audit_context=audit)

    assert audit["planning_stage"] == "mission_interpretation"
    assert audit["fleet_mission_id"] == "mission_new_failure"
    assert audit["model_call_records"] is model_calls
    assert audit["local_planner_proposals"] is local_proposals
    assert not {"task_spec", "request_v2", "plan_v2", "fleet_semantic_findings"} & audit.keys()
    assert not any(key.startswith("fleet_planner_") for key in audit)
    assert len(audit["interpreter_proposals"]) == 2
    assert not any(row["accepted"] for row in audit["interpreter_proposals"])


def test_reused_request_audit_drops_old_plan_and_interpretation_on_failure() -> None:
    config = _config()
    clients = _Clients()
    service = FleetPlanningService(config, clients)
    audit = {}
    initial = service.plan(SOURCE, fleet_mission_id="mission_reused_request", audit_context=audit)
    service.plan_request(initial.request, audit_context=audit)
    assert audit["planning_stage"] == "completed"
    assert not any(key.startswith("interpreter_") for key in audit)
    clients.fleet.response = lambda _: "{"
    next_request = replace(initial.request, fleet_plan_version=2)

    with pytest.raises(FleetPlannerOutputError):
        service.plan_request(next_request, replan=True, audit_context=audit)

    assert audit["planning_stage"] == "fleet_assignment"
    assert audit["request_v2"] is next_request
    assert audit["task_spec"] is next_request.task_spec
    assert "plan_v2" not in audit
    assert "fleet_semantic_findings" not in audit
    assert not any(key.startswith("interpreter_") for key in audit)
    assert len(audit["fleet_planner_proposals"]) == 3
    assert not any(row["accepted"] for row in audit["fleet_planner_proposals"])


@pytest.mark.parametrize("instruction", ["  \n", "x" * 8193, "instruction\x00"])
def test_invalid_instruction_is_rejected_before_creating_model_clients(instruction) -> None:
    clients = _Clients()
    audit = {}
    with pytest.raises(ValueError):
        FleetPlanningService(_config(), clients).plan(
            instruction, fleet_mission_id="mission_invalid_instruction", audit_context=audit
        )
    assert clients.selections == []
    assert audit["planning_stage"] == "mission_interpretation"
