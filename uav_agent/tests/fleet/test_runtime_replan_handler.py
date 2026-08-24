from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from configs.loader import load_config
from experiments.planning_audit_logger import PlanningAuditLogger
from fleet.runtime import FleetReplanPublication, ReplannedAssignment
from fleet.types_v2 import (
    AssignmentDeviation,
    FleetAssignmentV2,
    FleetMissionPlanV2,
)
from models.adapter_registry import ModelCallRole
from planner.policy import PlannerLimits, PlannerPolicy
from scripts.run_fleet_mission import (
    _build_runtime_fleet_replan_handler,
    _finalize_bounded_results,
    _lower_v2_target_plan_for_runtime,
)
from tests.fleet.test_compiler_v2 import _GoalDrivenSpatialPlanner
from tests.fleet.test_fleet_planner_v2 import _request


ROOT = Path(__file__).resolve().parents[2]


class _ClientFactory:
    def __init__(self) -> None:
        self.roles: list[ModelCallRole] = []

    def for_role(self, role, **routing):
        self.roles.append(role)
        return SimpleNamespace(role=role, routing=routing)


class _FleetReplanner:
    def __init__(self) -> None:
        self.model_proposals = (
            {
                "attempt_index": 0,
                "repair": False,
                "accepted": True,
                "error_code": None,
                "response_length": 256,
                "proposal": None,
            },
        )

    def plan(self, request):
        return FleetMissionPlanV2(
            fleet_mission_id=request.fleet_mission_id,
            fleet_plan_version=request.fleet_plan_version,
            assignments=(
                FleetAssignmentV2(
                    assignment_id="assignment_runtime_replanned",
                    uav_id="uav_b",
                    goal_ids=request.task_spec.all_goal_ids,
                    priority=100,
                    start_policy="PARALLEL",
                    deviations=(
                        AssignmentDeviation(
                            "constraint_a_i",
                            "UAV_UNAVAILABLE",
                            ("evidence_uav_a_unavailable",),
                        ),
                    ),
                ),
            ),
            coordination_policy=request.coordination_policy,
        )


class _ReplacementAgent:
    uav_id = "uav_b"

    def snapshot(self):
        return {"status": "IDLE", "plan_version": None}


def _prepared():
    config = load_config(ROOT / "configs/multi_uav_demo.yaml")
    original = _request(with_evidence=False)
    request_v2 = replace(
        original,
        fleet_mission_id="fleet_mission_runtime_replan",
        trusted_fleet_state=(),
    )
    assignment_v2 = FleetAssignmentV2(
        assignment_id="assignment_source",
        uav_id="uav_a",
        goal_ids=request_v2.task_spec.all_goal_ids,
        priority=100,
        start_policy="PARALLEL",
    )
    plan_v2 = FleetMissionPlanV2(
        fleet_mission_id=request_v2.fleet_mission_id,
        fleet_plan_version=1,
        assignments=(assignment_v2,),
        coordination_policy=request_v2.coordination_policy,
    )
    runtime_request, runtime_plan = _lower_v2_target_plan_for_runtime(
        config, request_v2, plan_v2
    )
    limits = PlannerLimits.from_config(config.planner)
    return SimpleNamespace(
        config=config,
        task_spec=request_v2.task_spec,
        fleet_request_v2=request_v2,
        fleet_plan_v2=plan_v2,
        request=runtime_request,
        plan=runtime_plan,
        model_client_factory=_ClientFactory(),
        local_planner_source="dynamic_llm",
        planner_limits=limits,
        planner_policy=PlannerPolicy.from_config(config.planner, limits),
        preparation_context={},
        compilations={},
        fleet_semantic_findings=(),
        mission_interpreter_proposals=(),
        fleet_planner_proposals=(),
        local_planner_proposals={},
        mission_interpreter_source="qwen_task_spec_v1",
        model_call_records=(),
    )


def test_production_handler_replans_compiles_audits_and_preserves_goal_mapping(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    source_assignment = prepared.plan.assignments[0]
    record = SimpleNamespace(
        assignment=source_assignment,
        local_plan_version=1,
    )
    belief = SimpleNamespace(
        fleet_plan_version=1,
        agents={
            "uav_a": SimpleNamespace(
                uav_id="uav_a", status="WAITING_REASSIGNMENT"
            )
        },
    )
    factory_calls: list[tuple[object, ...]] = []

    def agent_factory(
        runtime_record,
        assignment_v2,
        compilation,
        context,
        runtime_assignment,
        route,
    ):
        factory_calls.append(
            (
                runtime_record,
                assignment_v2,
                compilation,
                context,
                runtime_assignment,
                route,
            )
        )
        return ReplannedAssignment(
            assignment_id=runtime_record.assignment.assignment_id,
            replacement_assignment=runtime_assignment,
            agent=_ReplacementAgent(),
            start_input=(
                compilation.planner_request.instruction,
                compilation.planner_request.world_context,
            ),
            planned_route=route,
        )

    handler = _build_runtime_fleet_replan_handler(
        prepared,
        audit=PlanningAuditLogger(tmp_path),
        agent_factory=agent_factory,
        fleet_planner_factory=lambda client: _FleetReplanner(),
        local_planner_factory=lambda client, uav_id: _GoalDrivenSpatialPlanner(),
    )
    assert handler is not None

    publication = handler(record, belief)

    assert isinstance(publication, FleetReplanPublication)
    assert publication.base_fleet_plan_version == 1
    assert publication.new_fleet_plan_version == 2
    handoff = publication.replacements[0]
    assert handoff.assignment_id == source_assignment.assignment_id
    assert handoff.replacement_assignment is not None
    assert handoff.replacement_assignment.uav_id == "uav_b"
    assert handoff.replacement_assignment.assignment_id == (
        "assignment_runtime_replanned"
    )
    assert factory_calls and factory_calls[0][1].goal_ids == (
        "goal_search_i",
        "goal_track_i",
        "goal_land",
    )
    history = prepared.preparation_context["runtime_reassignments"]
    assert history == [
        {
            "schema_version": 1,
            "base_fleet_plan_version": 1,
            "new_fleet_plan_version": 2,
            "source_assignment_id": "assignment_source",
            "replacement_assignment_id": "assignment_runtime_replanned",
            "uav_id": "uav_b",
            "goal_ids": ["goal_search_i", "goal_track_i", "goal_land"],
            "semantically_valid": True,
        }
    ]
    assert prepared.model_client_factory.roles == [
        ModelCallRole.FLEET_REPLAN,
        ModelCallRole.AGENT_SPATIAL_PLAN,
    ]
    assert (tmp_path / "planning_attempts.jsonl").is_file()
    assert (tmp_path / "validation_findings.jsonl").is_file()
    assert (tmp_path / "recovery_actions.jsonl").is_file()
    assert (tmp_path / "final_plans.jsonl").is_file()


def test_production_handler_has_one_attempt_per_failed_assignment(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    record = SimpleNamespace(
        assignment=prepared.plan.assignments[0],
        local_plan_version=1,
    )
    belief = SimpleNamespace(
        fleet_plan_version=1,
        agents={
            "uav_a": SimpleNamespace(
                uav_id="uav_a", status="WAITING_REASSIGNMENT"
            ),
            "uav_b": SimpleNamespace(uav_id="uav_b", status="RUNNING"),
        },
    )
    handler = _build_runtime_fleet_replan_handler(
        prepared,
        audit=PlanningAuditLogger(tmp_path),
        agent_factory=lambda *args: None,
        fleet_planner_factory=lambda client: _FleetReplanner(),
        local_planner_factory=lambda client, uav_id: _GoalDrivenSpatialPlanner(),
    )
    assert handler is not None

    # Both configured UAVs are occupied, so no model call is authorized.
    try:
        handler(record, belief)
    except Exception as exc:
        assert "no trusted available UAV" in str(exc)
    else:
        raise AssertionError("handler accepted a replan without an idle UAV")
    try:
        handler(record, belief)
    except Exception as exc:
        assert "budget exhausted" in str(exc)
    else:
        raise AssertionError("handler exceeded its per-assignment budget")
    assert prepared.model_client_factory.roles == []


def test_local_replan_repair_uses_independent_incrementing_plan_versions(
    tmp_path: Path,
) -> None:
    class _RepairingLocalPlanner:
        source = "dynamic_llm"

        def __init__(self) -> None:
            self.calls = 0
            self.delegate = _GoalDrivenSpatialPlanner()
            self.model_proposals = ()
            self.requests = []

        def plan(self, request):
            self.calls += 1
            self.requests.append(request)
            if self.calls == 1:
                raise ValueError(
                    "rejected local structure with private prompt/raw output"
                )
            return self.delegate.plan(request)

    prepared = _prepared()
    record = SimpleNamespace(
        assignment=prepared.plan.assignments[0],
        local_plan_version=1,
    )
    belief = SimpleNamespace(
        fleet_plan_version=1,
        agents={
            "uav_a": SimpleNamespace(
                uav_id="uav_a", status="WAITING_REASSIGNMENT"
            )
        },
    )
    local = _RepairingLocalPlanner()
    accepted_versions: list[int] = []

    def agent_factory(
        runtime_record,
        assignment_v2,
        compilation,
        context,
        runtime_assignment,
        route,
    ):
        accepted_versions.append(compilation.agent_request.local_plan_version)
        return ReplannedAssignment(
            assignment_id=runtime_record.assignment.assignment_id,
            replacement_assignment=runtime_assignment,
            agent=_ReplacementAgent(),
            start_input=(
                compilation.planner_request.instruction,
                compilation.planner_request.world_context,
            ),
            planned_route=route,
        )

    handler = _build_runtime_fleet_replan_handler(
        prepared,
        audit=PlanningAuditLogger(tmp_path),
        agent_factory=agent_factory,
        fleet_planner_factory=lambda client: _FleetReplanner(),
        local_planner_factory=lambda client, uav_id: local,
    )
    assert handler is not None

    publication = handler(record, belief)

    assert publication.new_fleet_plan_version == 2
    assert local.calls == 2
    assert accepted_versions == [3]
    initial_focused = json.loads(local.requests[0].instruction)
    repaired_focused = json.loads(local.requests[1].instruction)
    assert "proposal_repair_findings" not in initial_focused
    assert repaired_focused["proposal_repair_findings"] == [
        {
            "code": "SCHEMA_INVALID",
            "message": (
                "Correct all required fields, value types, and trusted routing "
                "values to match the response schema."
            ),
        }
    ]
    assert "semantic_repair_findings" not in repaired_focused
    assert "private prompt" not in local.requests[1].instruction
    assert "raw output" not in local.requests[1].instruction


def test_terminal_goal_metrics_follow_successful_runtime_reassignment() -> None:
    class _Recorder:
        latest_state_timestamp_s = 12.0
        collision_count = 0
        out_of_bounds_count = 0
        emergency_landing_count = 0
        minimum_inter_uav_distance_m = 8.0

        def __init__(self) -> None:
            self.goals = []

        def record_goal_result(self, record):
            self.goals.append(record)

        def finalize(self, summary):
            return dict(summary)

    prepared = _prepared()
    prepared.preparation_context["runtime_reassignments"] = [
        {
            "schema_version": 1,
            "base_fleet_plan_version": 1,
            "new_fleet_plan_version": 2,
            "source_assignment_id": "assignment_source",
            "replacement_assignment_id": "assignment_runtime_replanned",
            "uav_id": "uav_b",
            "goal_ids": list(prepared.task_spec.all_goal_ids),
            "semantically_valid": True,
        }
    ]
    recorder = _Recorder()

    summary = _finalize_bounded_results(
        recorder,
        prepared,
        {
            "status": "SUCCEEDED",
            "assignments": {
                "assignment_runtime_replanned": {
                    "status": "SUCCEEDED",
                    "uav_id": "uav_b",
                }
            },
        },
        wall_time_s=2.0,
    )

    assert summary["goal_count"] == 3
    assert summary["goals_completed"] == 3
    assert summary["reassignment_count"] == 1
    assert summary["reassignments_succeeded"] == 1
    assert {item.assignment_id for item in recorder.goals} == {
        "assignment_runtime_replanned"
    }
    assert {item.uav_id for item in recorder.goals} == {"uav_b"}
    assert all(item.completed for item in recorder.goals)
