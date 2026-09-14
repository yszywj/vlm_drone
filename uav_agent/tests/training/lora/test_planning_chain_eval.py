import json

import pytest

from models.base import ModelResponse
from planning_data.generator import render_task
from planning_data.tasks import generate_task_blueprints
from training.lora.planning_chain_eval import DEFAULT_MAX_TOKENS, run_chain


class Client:
    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def chat(self, messages, *, options):
        self.requests.append((messages, options))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
        return ModelResponse(answer, "fixture_client", "stop", {
            "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
        })


def fixture(family="navigate"):
    task = next(task for task in generate_task_blueprints(25) if task["family"] == family and task["scale"] == 2)
    rows = render_task(task)
    answers = {
        role: [json.loads(row["messages"][-1]["content"]) for row in rows if row["role"] == role]
        for role in DEFAULT_MAX_TOKENS
    }
    return task, answers


def clients(answers):
    return {role: Client(values) for role, values in answers.items()}


@pytest.mark.parametrize("family", ["navigate", "hover", "search", "search_track", "mixed_parallel"])
def test_valid_real_production_chain(family):
    task, answers = fixture(family)
    models = clients(answers)
    result = run_chain(task, models)
    assert result["strict_entire_task_pass"], result
    assert result["local_pass_count"] == result["expected_local_count"] == 2
    assert result["model_call_count"] == 4
    assert result["usage"] == {"prompt_tokens": 40, "completion_tokens": 80, "total_tokens": 120}
    assert len(result["calls"]) == 4
    assert all(row["production_accepted"] and row["blueprint_pass"] for row in result["local_results"])
    for role, client in models.items():
        assert all(options.max_tokens == DEFAULT_MAX_TOKENS[role] for _, options in client.requests)
        assert all(options.response_format is not None for _, options in client.requests)
    json.dumps(result, allow_nan=False)


def test_unparseable_interpretation_blocks_dependent_stages():
    task, answers = fixture()
    answers["mission_interpreter"] = ["broken JSON"]
    models = clients(answers)
    result = run_chain(task, models)
    assert not result["strict_entire_task_pass"]
    assert result["stages"]["interpreter"]["status"] == "failed"
    assert result["stages"]["fleet"]["status"] == "blocked"
    assert result["local_blocked_count"] == 2
    assert result["local_pass_count"] == 0
    assert result["model_call_count"] == 1
    assert models["fleet_planner"].requests == models["spatial_mission"].requests == []
    assert result["calls"][0]["response"]["content"] == "broken JSON"


def test_upstream_semantic_error_propagates_without_gold_replacement():
    task, answers = fixture()
    goal = answers["mission_interpreter"][0]["goals"][0]
    old_x = goal["spatial_constraint"]["xyz_m"][0]
    goal["spatial_constraint"]["xyz_m"][0] += 1.0
    goal_id = goal["goal_id"]
    fleet = answers["fleet_planner"][0]
    owner = next(a["uav_id"] for a in fleet["assignments"] if goal_id in a["goal_ids"])
    raw = next(answer for answer in answers["spatial_mission"] if answer["uav_id"] == owner)
    goto = next(step for step in raw["steps"] if step["skill"] == "GOTO")
    goto["args"]["target"]["xyz_m"][0] = old_x + 1.0
    result = run_chain(task, clients(answers))
    assert not result["strict_entire_task_pass"]
    assert result["model_call_count"] == 4
    assert result["stages"]["interpreter"]["production_accepted"]
    assert not result["stages"]["interpreter"]["blueprint_pass"]
    assert result["stages"]["fleet"]["production_accepted"]
    transmitted = json.loads(result["calls"][1]["messages"][1]["content"])
    actual = next(g for g in transmitted["trusted_request"]["task_spec"]["goals"] if g["goal_id"] == goal_id)
    assert actual["spatial_constraint"]["xyz_m"][0] == old_x + 1.0
    row = next(row for row in result["local_results"] if row["uav_id"] == owner)
    assert row["assigned_semantics_pass"]
    assert not row["blueprint_pass"]
    assert any(item["code"] == "NAVIGATION_COORDINATES_MISMATCH" for item in row["findings"])


def test_failed_fleet_output_does_not_count_unattempted_locals_as_success():
    task, answers = fixture()
    answers["fleet_planner"] = ["not a plan"]
    result = run_chain(task, clients(answers))
    assert result["stages"]["interpreter"]["blueprint_pass"]
    assert result["stages"]["fleet"]["status"] == "failed"
    assert result["local_blocked_count"] == 2
    assert result["model_call_count"] == 2
    assert not result["strict_entire_task_pass"]


def test_locals_continue_after_one_model_failure():
    task, answers = fixture()
    answers["spatial_mission"][0] = RuntimeError("fixture network failure")
    result = run_chain(task, clients(answers))
    assert result["model_call_count"] == 4
    assert result["client_error_count"] == 1
    assert result["local_pass_count"] == 1
    assert result["local_blocked_count"] == 0
    assert not result["strict_entire_task_pass"]
    assert result["calls"][2]["error"]["type"] == "RuntimeError"


def test_complete_but_longer_tracking_is_not_accepted_as_original_intent():
    task, answers = fixture("search_track")
    raw = answers["spatial_mission"][0]
    step = next(step for step in raw["steps"] if step["skill"] == "TRACK")
    step["args"]["duration_s"] += 5
    result = run_chain(task, clients(answers))
    row = next(row for row in result["local_results"] if row["uav_id"] == raw["uav_id"])
    assert row["production_accepted"]
    assert not row["blueprint_pass"]
    assert any(item["code"] == "DURATION_MISMATCH" for item in row["findings"])
    assert not result["strict_entire_task_pass"]


def test_semantically_wrong_fleet_is_a_scored_failure_not_evaluator_crash():
    task, answers = fixture()
    first, second = answers["fleet_planner"][0]["assignments"]
    first["goal_ids"], second["goal_ids"] = second["goal_ids"], first["goal_ids"]
    result = run_chain(task, clients(answers))
    assert result["stages"]["fleet"]["production_accepted"]
    assert result["stages"]["fleet"]["production_semantic_findings"]
    assert not result["stages"]["fleet"]["blueprint_pass"]
    assert not result["strict_entire_task_pass"]


def test_no_assignment_keeps_original_expected_uav_denominator():
    task, answers = fixture()
    fleet = answers["fleet_planner"][0]
    removed = fleet["assignments"].pop()
    fleet["unassigned_goal_ids"] = removed["goal_ids"]
    result = run_chain(task, clients(answers))
    assert result["expected_local_count"] == 2
    assert result["local_pass_count"] == 1
    assert result["local_blocked_count"] == 1
    assert result["model_call_count"] == 3
    assert not result["strict_entire_task_pass"]
    row = next(row for row in result["local_results"] if row["uav_id"] == removed["uav_id"])
    assert row["blocked_reason"] == "no_assignment_for_expected_uav"


def test_grouping_two_targets_never_passes_from_one_targets_local_plan():
    task, answers = fixture("search_track")
    fleet = answers["fleet_planner"][0]
    first, second = fleet["assignments"]
    first["goal_ids"].extend(second["goal_ids"])
    fleet["assignments"] = [first]
    result = run_chain(task, clients(answers))
    assert result["stages"]["fleet"]["production_accepted"]
    assert result["model_call_count"] == 3
    assert result["local_blocked_count"] == 1
    assert not result["strict_entire_task_pass"]
    row = next(row for row in result["local_results"] if row["uav_id"] == first["uav_id"])
    assert not row["passed"]
