from copy import deepcopy
from dataclasses import replace

import pytest

from fleet.task_spec import (
    AssignmentConstraint, FleetTaskSpecV1, MissionGoal, OrderingConstraint,
    SourceEvidence, TerminationGoal,
)
from fleet.types import FleetCoordinationPolicy
from fleet.types_v2 import FleetAssignmentV2, FleetMissionPlanV2
from planner.spatial import CircleRegion, PointTarget
from planning_data.validation import validate_semantic_gold


SOURCE = (
    "各无人机并行执行。无人机A飞到(-30,20,8)后降落；"
    "无人机B飞到(20,-20,8)后悬停12秒；"
    "无人机C在(10,12,0)半径7米搜索目标C后返航降落；"
    "无人机D在(-10,12,0)半径8米搜索目标D并跟踪25秒后返航。"
)
EVIDENCE = ("source",)


@pytest.fixture
def labels():
    task = {
        "task_id": "sample_independent", "split": "train", "family": "mixed",
        "scale": 4, "instruction": SOURCE, "language_variant": "explicit",
        "assignments": [
            {"uav_id": "uav_a", "kind": "navigate", "destination_xyz_m": [-30, 20, 8], "terminal_goal_type": "LAND"},
            {"uav_id": "uav_b", "kind": "hover", "destination_xyz_m": [20, -20, 8], "duration_s": 12, "terminal_goal_type": "WAIT"},
            {"uav_id": "uav_c", "kind": "search", "target_alias": "target_c", "search_center_xyz_m": [10, 12, 0], "search_radius_m": 7, "terminal_goal_type": "RETURN_HOME_AND_LAND"},
            {"uav_id": "uav_d", "kind": "search_track", "target_alias": "target_d", "search_center_xyz_m": [-10, 12, 0], "search_radius_m": 8, "duration_s": 25, "terminal_goal_type": "RETURN_HOME"},
        ],
    }
    goals = (
        MissionGoal("nav_a", "NAVIGATE", None, PointTarget("WORLD_ENU", (-30, 20, 8)), None, None, "MUST", EVIDENCE),
        MissionGoal("nav_b", "NAVIGATE", None, PointTarget("WORLD_ENU", (20, -20, 8)), None, None, "MUST", EVIDENCE),
        MissionGoal("search_c", "SEARCH_TARGET", "target_c", CircleRegion("WORLD_ENU", (10, 12, 0), 7), None, None, "MUST", EVIDENCE),
        MissionGoal("search_d", "SEARCH_TARGET", "target_d", CircleRegion("WORLD_ENU", (-10, 12, 0), 8), None, None, "MUST", EVIDENCE),
        MissionGoal("track_d", "TRACK_TARGET", "target_d", None, 25, None, "MUST", EVIDENCE),
    )
    terminals = (
        TerminationGoal("finish_a", "LAND", "uav_a", None, "MUST", EVIDENCE),
        TerminationGoal("wait_b", "WAIT", "uav_b", 12, "MUST", EVIDENCE),
        TerminationGoal("finish_c", "RETURN_HOME_AND_LAND", "uav_c", None, "MUST", EVIDENCE),
        TerminationGoal("finish_d", "RETURN_HOME", "uav_d", None, "MUST", EVIDENCE),
    )
    chain_ids = (
        ("nav_a", "finish_a"), ("nav_b", "wait_b"),
        ("search_c", "finish_c"), ("search_d", "track_d", "finish_d"),
    )
    constraints, ordering, assignments = [], [], []
    for suffix, ids in zip("abcd", chain_ids):
        constraints.append(AssignmentConstraint(f"bind_{suffix}", f"uav_{suffix}", ids, "MUST", EVIDENCE))
        assignments.append(FleetAssignmentV2(f"assignment_{suffix}", f"uav_{suffix}", ids, 100, "PARALLEL"))
        for index, (before, after) in enumerate(zip(ids, ids[1:])):
            ordering.append(OrderingConstraint(f"order_{suffix}_{index}", before, after, "MUST", EVIDENCE))
    spec = FleetTaskSpecV1(
        SOURCE, goals, tuple(constraints), tuple(ordering), terminals,
        source_evidence=(SourceEvidence("source", SOURCE),),
    )
    plan = FleetMissionPlanV2("mission_independent", 1, tuple(assignments), FleetCoordinationPolicy())
    return task, spec, plan


def _codes(result):
    return {item["code"] for item in result["findings"]}


def _change_goal(spec, goal_id, **updates):
    return replace(spec, goals=tuple(replace(goal, **updates) if goal.goal_id == goal_id else goal for goal in spec.goals))


def test_all_four_task_kinds_pass_independent_blueprint_check(labels):
    assert validate_semantic_gold(*labels) == {"passed": True, "findings": []}


def test_ten_hover_uavs_require_only_explicit_wait_semantics():
    source = "十架无人机并行起飞原地悬停15秒，计时结束即完成本轮任务。"
    owners = [f"uav_{chr(97 + index)}" for index in range(10)]
    task = {"instruction": source, "assignments": [
        {"uav_id": owner, "kind": "hover", "destination_xyz_m": None,
         "duration_s": 15, "terminal_goal_type": "WAIT",
         "closure_policy": "runtime_contract_home_and_land"}
        for owner in owners
    ]}
    terminal = tuple(TerminationGoal(f"wait_{index}", "WAIT", owner, 15, "MUST", EVIDENCE) for index, owner in enumerate(owners))
    constraints = tuple(AssignmentConstraint(f"bind_{index}", owner, (terminal[index].goal_id,), "MUST", EVIDENCE) for index, owner in enumerate(owners))
    spec = FleetTaskSpecV1(source, (), constraints, (), terminal, source_evidence=(SourceEvidence("source", source),))
    assignments = tuple(FleetAssignmentV2(f"assignment_{index}", owner, (terminal[index].goal_id,), 100, "PARALLEL") for index, owner in enumerate(owners))
    plan = FleetMissionPlanV2("mission_hover", 1, assignments, FleetCoordinationPolicy())
    assert validate_semantic_gold(task, spec, plan)["passed"]
    # Local controller closure must not become an invented semantic goal.
    with_extra_return = replace(spec, termination_goals=(*spec.termination_goals, TerminationGoal("invented_home", "RETURN_HOME_AND_LAND", "uav_a", None, "MUST", EVIDENCE)))
    assert "UNEXPECTED_GOAL" in _codes(validate_semantic_gold(task, with_extra_return, plan))


def test_goal_and_constraint_names_and_array_order_are_not_gold(labels):
    task, spec, plan = labels
    ids = {goal_id: f"renamed_{index}" for index, goal_id in enumerate(spec.all_goal_ids)}
    spec = replace(
        spec,
        goals=tuple(replace(g, goal_id=ids[g.goal_id]) for g in reversed(spec.goals)),
        termination_goals=tuple(replace(g, goal_id=ids[g.goal_id]) for g in reversed(spec.termination_goals)),
        assignment_constraints=tuple(replace(c, goal_ids=tuple(ids[g] for g in c.goal_ids)) for c in spec.assignment_constraints),
        ordering_constraints=tuple(replace(c, before_goal_id=ids[c.before_goal_id], after_goal_id=ids[c.after_goal_id]) for c in spec.ordering_constraints),
    )
    plan = replace(plan, assignments=tuple(replace(a, goal_ids=tuple(ids[g] for g in reversed(a.goal_ids))) for a in reversed(plan.assignments)))
    assert validate_semantic_gold(task, spec, plan)["passed"]


@pytest.mark.parametrize("updates,code", [
    ({"target_alias": "target_wrong"}, "TARGET_ALIAS_MISMATCH"),
    ({"duration_s": 3}, "DURATION_MISMATCH"),
    ({"strength": "PREFER"}, "GOAL_STRENGTH_MISMATCH"),
    ({"distance_m": 12}, "INVENTED_DISTANCE"),
    ({"evidence_refs": ()}, "MISSING_OR_INVALID_EVIDENCE"),
    ({"spatial_constraint": CircleRegion("WORLD_ENU", (99, 12, 0), 7)}, "SPATIAL_CONSTRAINT_MISMATCH"),
    ({"spatial_constraint": CircleRegion("HOME_ENU", (10, 12, 0), 7)}, "SPATIAL_CONSTRAINT_MISMATCH"),
    ({"spatial_constraint": CircleRegion("WORLD_ENU", (10, 12, 0), 9)}, "SPATIAL_CONSTRAINT_MISMATCH"),
])
def test_mutated_search_facts_are_rejected(labels, updates, code):
    task, spec, plan = labels
    result = validate_semantic_gold(task, _change_goal(spec, "search_c", **updates), plan)
    assert not result["passed"]
    assert code in _codes(result)


def test_tracking_duration_is_checked_against_original_task(labels):
    task, spec, plan = labels
    result = validate_semantic_gold(task, _change_goal(spec, "track_d", duration_s=20), plan)
    assert "DURATION_MISMATCH" in _codes(result)


@pytest.mark.parametrize("change", ["missing", "weakened", "wrong_owner"])
def test_must_binding_is_required_independently_of_correct_fleet_output(labels, change):
    task, spec, plan = labels
    original = spec.assignment_constraints[2]
    if change == "missing":
        constraints = tuple(c for c in spec.assignment_constraints if c != original)
    else:
        altered = replace(original, **({"strength": "PREFER"} if change == "weakened" else {"uav_id": "uav_b"}))
        constraints = tuple(altered if c == original else c for c in spec.assignment_constraints)
    result = validate_semantic_gold(task, replace(spec, assignment_constraints=constraints), plan)
    assert "MUST_BINDING_MISMATCH" in _codes(result)


def test_terminal_ownership_and_type_are_checked(labels):
    task, spec, plan = labels
    wrong_owner = replace(spec, termination_goals=tuple(replace(g, uav_id="uav_a") if g.goal_id == "finish_c" else g for g in spec.termination_goals))
    assert "TERMINATION_OWNER_MISMATCH" in _codes(validate_semantic_gold(task, wrong_owner, plan))
    wrong_type = replace(spec, termination_goals=tuple(replace(g, goal_type="LAND") if g.goal_id == "finish_c" else g for g in spec.termination_goals))
    assert "MISSING_GOAL" in _codes(validate_semantic_gold(task, wrong_type, plan))


@pytest.mark.parametrize("change", ["missing", "reversed", "cross_uav", "weakened"])
def test_execution_order_must_preserve_each_parallel_chain(labels, change):
    task, spec, plan = labels
    ordering = list(spec.ordering_constraints)
    if change == "missing":
        ordering.pop()
    elif change == "reversed":
        ordering[-1] = replace(ordering[-1], before_goal_id=ordering[-1].after_goal_id, after_goal_id=ordering[-1].before_goal_id)
    elif change == "cross_uav":
        ordering.append(OrderingConstraint("cross", "finish_c", "search_d", "MUST", EVIDENCE))
    else:
        ordering[-1] = replace(ordering[-1], strength="PREFER")
    result = validate_semantic_gold(task, replace(spec, ordering_constraints=tuple(ordering)), plan)
    assert not result["passed"]
    assert _codes(result) & {"MISSING_ORDERING", "UNEXPECTED_ORDERING", "ORDERING_STRENGTH_MISMATCH"}


def test_redundant_forward_order_does_not_fail(labels):
    task, spec, plan = labels
    spec = replace(spec, ordering_constraints=(*spec.ordering_constraints, OrderingConstraint("redundant", "search_d", "finish_d", "MUST", EVIDENCE)))
    assert validate_semantic_gold(task, spec, plan)["passed"]


def test_extra_goal_is_rejected_even_when_fleet_assigns_it(labels):
    task, spec, plan = labels
    extra = MissionGoal("extra", "NAVIGATE", None, PointTarget("WORLD_ENU", (1, 2, 8)), None, None, "MUST", EVIDENCE)
    spec = replace(spec, goals=(*spec.goals, extra))
    plan = replace(plan, assignments=(replace(plan.assignments[0], goal_ids=(*plan.assignments[0].goal_ids, "extra")), *plan.assignments[1:]))
    assert "UNEXPECTED_GOAL" in _codes(validate_semantic_gold(task, spec, plan))


def test_fleet_owner_is_checked_against_blueprint(labels):
    task, spec, plan = labels
    first, second, *remaining = plan.assignments
    plan = replace(plan, assignments=(replace(first, uav_id=second.uav_id), replace(second, uav_id=first.uav_id), *remaining))
    assert "FLEET_OWNER_MISMATCH" in _codes(validate_semantic_gold(task, spec, plan))


@pytest.mark.parametrize("declared", [False, True])
def test_fleet_omission_and_declared_unassigned_both_fail(labels, declared):
    task, spec, plan = labels
    last = plan.assignments[-1]
    plan = replace(plan, assignments=(*plan.assignments[:-1], replace(last, goal_ids=last.goal_ids[:-1])), unassigned_goal_ids=("finish_d",) if declared else ())
    result = validate_semantic_gold(task, spec, plan)
    assert "FLEET_GOAL_COVERAGE" in _codes(result)
    if declared:
        assert "FLEET_UNASSIGNED_GOALS" in _codes(result)


def test_original_instruction_is_not_replaced_by_interpreter_text(labels):
    task, spec, plan = labels
    spec = replace(spec, source_text=" " + spec.source_text)
    assert "SOURCE_TEXT_MISMATCH" in _codes(validate_semantic_gold(task, spec, plan))


def test_blueprint_target_change_cannot_be_hidden_by_consistent_labels(labels):
    task, spec, plan = labels
    changed = deepcopy(task)
    changed["assignments"][2]["target_alias"] = "different_target"
    assert "TARGET_ALIAS_MISMATCH" in _codes(validate_semantic_gold(changed, spec, plan))


def test_unknown_blueprint_kind_is_an_error_not_a_false_pass(labels):
    task, spec, plan = labels
    changed = deepcopy(task)
    changed["assignments"][0]["kind"] = "unimplemented"
    with pytest.raises(ValueError, match="unsupported blueprint"):
        validate_semantic_gold(changed, spec, plan)
