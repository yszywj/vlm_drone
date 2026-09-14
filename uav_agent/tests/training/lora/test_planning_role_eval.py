"""Semantic mutation tests for first-response, real-generation evaluation."""

from copy import deepcopy
import json

import pytest

from planning_data.generator import render_task
from planning_data.tasks import generate_task_blueprints
from training.lora.planning_role_eval import capture_role_request, score_role_output


@pytest.fixture(scope="module")
def cases():
    return [(task, render_task(task)) for task in generate_task_blueprints(25)]


def _case(cases, family, role, scale=2):
    task, rows = next((task, rows) for task, rows in cases if task["family"] == family and task["scale"] == scale)
    row = next(row for row in rows if row["role"] == role)
    return task, row, json.loads(row["messages"][-1]["content"])


def _score(task, row, answer):
    raw = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    return score_role_output(task, row["role"], raw, uav_id=row["uav_id"])


def _codes(score):
    return {finding["code"] for finding in score["findings"]}


@pytest.mark.parametrize("case_index", range(25))
def test_gold_passes_and_captured_inputs_match_persisted_messages_and_schema(cases, case_index):
    task, rows = cases[case_index]
    for row in rows:
        score = _score(task, row, row["messages"][-1]["content"])
        assert score["passed"], score
        assert score["replay_calls"] == 1
        assert not score["repair_used"]
        captured = capture_role_request(task, row["role"], uav_id=row["uav_id"])
        assert [message.to_dict() for message in captured["messages"]] == row["messages"][:-1]
        assert captured["response_schema_sha256"] == row["response_schema_sha256"]
        assert set(captured) == {"messages", "options", "response_schema_sha256"}
        assert all(message.role != "assistant" for message in captured["messages"])


@pytest.mark.parametrize("role", ("mission_interpreter", "fleet_planner", "spatial_mission"))
@pytest.mark.parametrize("invalid", ("{", '{"schema_version":1,"schema_version":3}'))
def test_invalid_or_duplicate_json_never_repaired(cases, role, invalid):
    task, row, _ = _case(cases, "search_track", role)
    score = _score(task, row, invalid)
    assert not score["passed"]
    assert not score["structural_pass"]
    assert score["replay_calls"] == 1
    assert not score["repair_used"]
    assert score["findings"]


def test_interpreter_arbitrary_consistent_goal_ids_are_semantically_accepted(cases):
    task, row, answer = _case(cases, "search_track", "mission_interpreter", 10)
    renamed = {goal["goal_id"]: f"independent_{index}" for index, goal in enumerate(answer["goals"] + answer["termination_goals"])}
    for goal in answer["goals"] + answer["termination_goals"]:
        goal["goal_id"] = renamed[goal["goal_id"]]
    for constraint in answer["assignment_constraints"]:
        constraint["goal_ids"] = [renamed[goal_id] for goal_id in constraint["goal_ids"]]
    for constraint in answer["ordering_constraints"]:
        for key in ("before_goal_id", "after_goal_id"):
            constraint[key] = renamed[constraint[key]]
    answer["goals"].reverse()
    answer["termination_goals"].reverse()
    score = _score(task, row, answer)
    assert score["passed"], score


def test_interpreter_missing_return_is_detected_even_when_valid_schema(cases):
    task, row, answer = _case(cases, "navigate", "mission_interpreter")
    removed = answer["termination_goals"].pop()["goal_id"]
    for constraint in answer["assignment_constraints"]:
        constraint["goal_ids"] = [goal_id for goal_id in constraint["goal_ids"] if goal_id != removed]
    answer["ordering_constraints"] = [constraint for constraint in answer["ordering_constraints"] if constraint["after_goal_id"] != removed]
    score = _score(task, row, answer)
    assert score["structural_pass"]
    assert not score["semantic_pass"]
    assert "MISSING_GOAL" in _codes(score)
    assert not any(code.startswith("FLEET_") for code in _codes(score))


@pytest.mark.parametrize("field", ("coordinate", "duration", "binding", "ordering"))
def test_interpreter_semantic_mutations_fail(cases, field):
    family = "navigate" if field == "coordinate" else "search_track"
    task, row, answer = _case(cases, family, "mission_interpreter")
    if field == "coordinate":
        answer["goals"][0]["spatial_constraint"]["xyz_m"][0] += 1
    elif field == "duration":
        next(goal for goal in answer["goals"] if goal["goal_type"] == "TRACK_TARGET")["duration_s"] += 1
    elif field == "binding":
        answer["assignment_constraints"][0]["uav_id"] = answer["assignment_constraints"][1]["uav_id"]
    else:
        answer["ordering_constraints"] = answer["ordering_constraints"][1:]
    score = _score(task, row, answer)
    assert score["structural_pass"], score
    assert not score["semantic_pass"]


def test_fleet_reordered_assignments_and_arbitrary_assignment_ids_pass(cases):
    task, row, answer = _case(cases, "search", "fleet_planner", 10)
    answer["assignments"].reverse()
    for index, assignment in enumerate(answer["assignments"]):
        assignment["assignment_id"] = f"chosen_assignment_{index}"
    assert _score(task, row, answer)["passed"]


def test_fleet_wrong_owners_fail_even_if_every_goal_covered(cases):
    task, row, answer = _case(cases, "search", "fleet_planner")
    first, second = answer["assignments"][:2]
    first["uav_id"], second["uav_id"] = second["uav_id"], first["uav_id"]
    score = _score(task, row, answer)
    assert score["structural_pass"], score
    assert not score["semantic_pass"]
    assert "FLEET_OWNER_MISMATCH" in _codes(score)


def test_fleet_declared_unassigned_goal_fails_semantics(cases):
    task, row, answer = _case(cases, "search", "fleet_planner")
    removed = answer["assignments"][0]["goal_ids"].pop()
    answer["unassigned_goal_ids"] = [removed]
    score = _score(task, row, answer)
    assert score["structural_pass"], score
    assert not score["semantic_pass"]
    assert "FLEET_UNASSIGNED_GOALS" in _codes(score)


def test_local_consistent_step_renaming_does_not_require_exact_gold_string(cases):
    task, row, answer = _case(cases, "search_track", "spatial_mission")
    renamed = {step["id"]: f"generated_step_{index}" for index, step in enumerate(answer["steps"])}
    for step in answer["steps"]:
        step["id"] = renamed[step["id"]]
        if step["skill"] == "TRACK":
            old = step["args"]["target_ref"][1:].split(".", 1)[0]
            step["args"]["target_ref"] = f"${renamed[old]}.target_id"
    assert _score(task, row, answer)["passed"]


def test_local_excess_duration_rejected_by_independent_blueprint(cases):
    task, row, answer = _case(cases, "search_track", "spatial_mission")
    next(step for step in answer["steps"] if step["skill"] == "TRACK")["args"]["duration_s"] += 5
    score = _score(task, row, answer)
    assert score["structural_pass"] and score["compilation_pass"]
    assert score["goal_coverage_pass"]
    assert not score["blueprint_semantic_pass"]
    assert not score["passed"]


def test_local_wrong_coordinate_rejected_despite_successful_compilation(cases):
    task, row, answer = _case(cases, "navigate", "spatial_mission")
    destination = next(step for step in answer["steps"] if step["skill"] == "GOTO" and step["args"]["target"]["kind"] == "POINT")
    destination["args"]["target"]["xyz_m"][0] += 1
    score = _score(task, row, answer)
    assert score["structural_pass"]
    assert not score["blueprint_semantic_pass"]
    assert not score["passed"]


@pytest.mark.parametrize("mutation", ("remove_land", "wrong_owner", "missing_search_reference"))
def test_local_invalid_raw_contract_never_silently_completed(cases, mutation):
    task, row, answer = _case(cases, "search_track", "spatial_mission")
    if mutation == "remove_land":
        answer["steps"].pop()
    elif mutation == "wrong_owner":
        answer["uav_id"] = "uav_b" if row["uav_id"] != "uav_b" else "uav_a"
    else:
        next(step for step in answer["steps"] if step["skill"] == "TRACK")["args"]["target_ref"] = "$not_a_search.target_id"
    score = _score(task, row, answer)
    assert not score["structural_pass"]
    assert not score["passed"]
    assert score["replay_calls"] == 1


def test_bad_evaluation_inputs_raise_instead_of_inflating_model_failure_count(cases):
    task, _, _ = _case(cases, "navigate", "mission_interpreter")
    with pytest.raises(ValueError, match="unknown planning role"):
        score_role_output(task, "other", "{}")
    with pytest.raises(ValueError, match="requires a UAV"):
        score_role_output(task, "spatial_mission", "{}")
    with pytest.raises(TypeError, match="generated response string"):
        score_role_output(task, "mission_interpreter", {})


def test_scoring_does_not_mutate_blueprint(cases):
    task, row, answer = _case(cases, "search_track", "spatial_mission")
    before = deepcopy(task)
    _score(task, row, answer)
    assert task == before
