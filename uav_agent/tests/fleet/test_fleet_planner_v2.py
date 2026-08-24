from __future__ import annotations

from copy import deepcopy
import json

import pytest

from configs.loader import load_config
from fleet.json_schema_v2 import build_fleet_mission_plan_v2_json_schema
from fleet.llm_planner_v2 import LLMFleetPlannerV2
from fleet.planner_base import FleetPlannerOutputError
from fleet.request_builder import build_fleet_mission_request_v2
from fleet.schemas_v2 import validate_fleet_mission_plan_v2
from fleet.task_spec import (
    AssignmentConstraint,
    FleetTaskSpecV1,
    MissionGoal,
    TerminationGoal,
)
from fleet.types import FleetCoordinationPolicy, FleetStartPolicy, FleetUavCapability
from fleet.types_v2 import (
    AssignmentDeviation,
    AgentPlannerRequestV2,
    FleetAssignmentV2,
    FleetMissionPlanV2,
    FleetMissionRequestV2,
    FleetSafetySummaryEntry,
    TrustedFleetStateEvidence,
)
from models.base import ModelResponse
from planner.spatial import CircleRegion
from target.types import TargetSpec
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


class _QueuedClient:
    def __init__(self, payloads: list[str]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[object, object]] = []

    def chat(self, messages, *, options=None):
        self.calls.append((messages, options))
        return ModelResponse(self.payloads.pop(0), "fake", "stop", {})


def _request(*, with_evidence: bool = True) -> FleetMissionRequestV2:
    task = FleetTaskSpecV1(
        source_text="让无人机A搜索并跟踪红色目标i，最后降落",
        goals=(
            MissionGoal(
                "goal_search_i",
                "SEARCH_TARGET",
                "target_i",
                CircleRegion("WORLD_ENU", (20.0, 30.0, 0.0), 15.0),
                None,
                None,
                "MUST",
            ),
            MissionGoal(
                "goal_track_i",
                "TRACK_TARGET",
                "target_i",
                None,
                10.0,
                None,
                "MUST",
            ),
        ),
        assignment_constraints=(
            AssignmentConstraint(
                "constraint_a_i",
                "uav_a",
                ("goal_search_i", "goal_track_i"),
                "MUST",
            ),
        ),
        termination_goals=(
            TerminationGoal("goal_land", "LAND", None, None, "MUST"),
        ),
    )
    inventory = (
        FleetUavCapability("uav_a", "A", True, "home_a", 5.0, 50.0),
        FleetUavCapability("uav_b", "B", True, "home_b", 5.0, 50.0),
    )
    evidence = (
        TrustedFleetStateEvidence(
            "fleet_ev_a_unavailable",
            "UAV_UNAVAILABLE",
            "uav_a is unavailable",
            uav_id="uav_a",
        ),
    ) if with_evidence else ()
    return FleetMissionRequestV2(
        fleet_mission_id="fleet_mission_v2_test",
        fleet_plan_version=1,
        task_spec=task,
        uav_inventory=inventory,
        trusted_fleet_state=evidence,
        coordination_policy=FleetCoordinationPolicy(),
    )


def _payload(request: FleetMissionRequestV2, *, uav_id: str = "uav_a") -> dict[str, object]:
    deviations = []
    if uav_id != "uav_a":
        deviations = [
            {
                "constraint_id": "constraint_a_i",
                "reason_code": "UAV_UNAVAILABLE",
                "evidence_refs": ["fleet_ev_a_unavailable"],
            }
        ]
    return {
        "schema_version": 2,
        "fleet_mission_id": request.fleet_mission_id,
        "fleet_plan_version": request.fleet_plan_version,
        "assignments": [
            {
                "assignment_id": "assignment_goal_i",
                "uav_id": uav_id,
                "goal_ids": ["goal_search_i", "goal_track_i", "goal_land"],
                "priority": 100,
                "start_policy": "PARALLEL",
                "deviations": deviations,
            }
        ],
        "coordination_policy": request.coordination_policy.to_dict(),
        "assumptions": [],
        "unassigned_goal_ids": [],
    }


def test_v2_round_trip_and_schema_do_not_const_lock_user_uav_constraint() -> None:
    request = _request()
    assert FleetMissionRequestV2.from_dict(request.to_dict()) == request
    schema = build_fleet_mission_plan_v2_json_schema(request)
    uav_schema = schema["properties"]["assignments"]["items"]["properties"]["uav_id"]
    assert "const" not in uav_schema
    assert uav_schema["enum"] == ["uav_a", "uav_b"]

    plan = FleetMissionPlanV2.from_dict(_payload(request, uav_id="uav_b"), request=request)
    assert plan.assignments[0].uav_id == "uav_b"
    findings = plan.semantic_findings(request)
    assert {item.code for item in findings} == {"EXPLAINED_ASSIGNMENT_DEVIATION"}


def test_request_builder_consumes_task_spec_without_fixed_instruction_parser() -> None:
    semantic_request = _request()
    config = load_config(_ROOT / "configs/multi_uav_demo.yaml")

    request = build_fleet_mission_request_v2(
        config,
        semantic_request.task_spec,
        fleet_mission_id="fleet_mission_builder_v2",
    )

    assert request.schema_version == 2
    assert request.task_spec is semantic_request.task_spec
    assert request.available_uav_ids == ("uav_a", "uav_b")
    assert request.task_spec.source_text == semantic_request.task_spec.source_text


def test_agent_planner_request_v2_projects_only_its_assignment_goals() -> None:
    request = _request()
    assignment = FleetAssignmentV2.from_dict(_payload(request)["assignments"][0])
    target_spec = TargetSpec(
        original_description="red target i",
        category="target",
        immutable_identity_summary="red target i",
    )
    local_request = AgentPlannerRequestV2.for_assignment(
        request,
        assignment,
        local_plan_version=3,
        fleet_safety_summary=(
            FleetSafetySummaryEntry(
                "uav_b", "assignment_other", "READY", "region_b", "layer_b"
            ),
        ),
        trusted_target_specs={"target_i": target_spec},
    )

    assert local_request.assignment_id == "assignment_goal_i"
    assert local_request.local_plan_version == 3
    assert tuple(goal.goal_id for goal in local_request.goals) == assignment.goal_ids
    assert AgentPlannerRequestV2.from_dict(local_request.to_dict()) == local_request
    serialized = json.dumps(local_request.to_dict()).casefold()
    assert "oracle" not in serialized and "velocity" not in serialized

    with pytest.raises(ValueError, match="outside this Assignment"):
        AgentPlannerRequestV2(
            fleet_mission_id=request.fleet_mission_id,
            assignment_id=assignment.assignment_id,
            uav_id=assignment.uav_id,
            goals=local_request.goals,
            local_plan_version=1,
            trusted_target_specs={"target_j": target_spec},
        )


def test_unexplained_assignment_deviation_is_finding_not_schema_block() -> None:
    request = _request()
    payload = _payload(request, uav_id="uav_b")
    payload["assignments"][0]["deviations"] = []
    plan = FleetMissionPlanV2.from_dict(payload, request=request)
    assert {item.code for item in plan.semantic_findings(request)} == {
        "UNEXPLAINED_ASSIGNMENT_DEVIATION"
    }


def test_deviation_evidence_must_come_from_trusted_fleet_state() -> None:
    request = _request()
    payload = _payload(request, uav_id="uav_b")
    payload["assignments"][0]["deviations"][0]["evidence_refs"] = ["invented_ev"]
    with pytest.raises(ValueError, match="trusted Fleet state"):
        FleetMissionPlanV2.from_dict(payload, request=request)


def test_llm_v2_repairs_structural_unknown_goal_then_accepts() -> None:
    request = _request()
    invalid = deepcopy(_payload(request))
    invalid["assignments"][0]["goal_ids"] = ["goal_invented"]
    client = _QueuedClient(
        [json.dumps(invalid), json.dumps(_payload(request), ensure_ascii=False)]
    )
    planner = LLMFleetPlannerV2(client)

    plan = planner.plan(request)

    assert plan.assignments[0].uav_id == "uav_a"
    assert len(client.calls) == 2
    assert all(call[1].temperature == 0.0 for call in client.calls)
    assert client.calls[0][1].response_format.name == "fleet_mission_plan_v2"
    assert planner.last_diagnostics.repair_used
    assert [item["accepted"] for item in planner.model_proposals] == [False, True]


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":2,"schema_version":2}',
        '{"schema_version":NaN}',
        '{"schema_version":Infinity}',
        '{"schema_version":2,"unknown":true}',
    ],
)
def test_llm_v2_strict_parser_rejects_duplicate_nan_inf_unknown(raw: str) -> None:
    planner = LLMFleetPlannerV2(_QueuedClient([raw]), repair_budget=0)
    with pytest.raises(FleetPlannerOutputError):
        planner.plan(_request())
    assert planner.last_diagnostics.final_output_valid is False


def test_direct_v2_types_reject_unknown_uav_goal_and_nonfinite_is_impossible() -> None:
    request = _request()
    assignment = FleetAssignmentV2(
        "assignment_bad",
        "uav_unknown",
        ("goal_search_i",),
        100,
        FleetStartPolicy.PARALLEL,
        (),
    )
    plan = FleetMissionPlanV2(
        request.fleet_mission_id,
        request.fleet_plan_version,
        (assignment,),
        request.coordination_policy,
    )
    with pytest.raises(ValueError, match="unknown UAV"):
        validate_fleet_mission_plan_v2(plan, request)

    with pytest.raises((TypeError, ValueError), match="priority"):
        FleetAssignmentV2(
            "assignment_nan",
            "uav_a",
            ("goal_search_i",),
            float("nan"),  # type: ignore[arg-type]
            "PARALLEL",
        )
