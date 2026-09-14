from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from math import dist

import pytest

from fleet.task_spec import FleetTaskSpecV1, MAX_TERMINATION_GOALS
from fleet.types_v2 import FleetMissionPlanV2, FleetMissionRequestV2
from planning_data.tasks import (
    FAMILIES,
    SCALES,
    build_fleet_plan,
    build_fleet_request,
    build_task_spec,
    generate_task_blueprints,
    instruction_semantic_hash,
    semantic_hash,
)


@pytest.fixture(scope="module")
def tasks():
    return generate_task_blueprints()


def test_default_corpus_is_balanced_unique_and_split_by_task(tasks) -> None:
    assert len(tasks) == 1000
    assert len({task["task_id"] for task in tasks}) == 1000
    assert len({task["semantic_hash"] for task in tasks}) == 1000
    assert len({task["instruction_semantic_hash"] for task in tasks}) == 1000
    assert all(instruction_semantic_hash(task) == task["instruction_semantic_hash"] for task in tasks)
    assert Counter(task["family"] for task in tasks) == {family: 200 for family in FAMILIES}
    assert Counter(task["scale"] for task in tasks) == {scale: 200 for scale in SCALES}
    assert Counter(task["split"] for task in tasks) == {"train": 800, "validation": 100, "test": 100}
    strata = Counter((task["family"], task["scale"], task["split"]) for task in tasks)
    for family in FAMILIES:
        for scale in SCALES:
            assert strata[family, scale, "train"] == 32
            assert strata[family, scale, "validation"] == 4
            assert strata[family, scale, "test"] == 4
    by_split = {
        split: {task["semantic_hash"] for task in tasks if task["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert not by_split["train"] & by_split["validation"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["validation"] & by_split["test"]


def test_seed_is_reproducible_and_does_not_reset_global_random_state(tasks) -> None:
    import random

    state = random.getstate()
    assert generate_task_blueprints() == tasks
    assert random.getstate() == state
    other = generate_task_blueprints(seed=43)
    assert {task["semantic_hash"] for task in tasks}.isdisjoint(
        task["semantic_hash"] for task in other
    )


def test_semantic_digest_ignores_presentation_but_tracks_mission_changes(tasks) -> None:
    original = next(task for task in tasks if task["family"] == "search_track")
    variant = deepcopy(original)
    variant.update(task_id="changed_task_id", split="other_split", language_variant=99, instruction="a paraphrase")
    variant["clause_order"].reverse()
    variant["assignments"].reverse()
    variant["uavs"].reverse()
    variant["source_quotes"] = {}
    assert semantic_hash(variant) == original["semantic_hash"]
    variant["assignments"][0]["duration_s"] += 5
    assert semantic_hash(variant) != original["semantic_hash"]
    bound = deepcopy(original)
    bound["assignments"][0]["target_alias"], bound["assignments"][1]["target_alias"] = (
        bound["assignments"][1]["target_alias"], bound["assignments"][0]["target_alias"]
    )
    assert semantic_hash(bound) != original["semantic_hash"]


@pytest.mark.parametrize("family", ["hover", "search", "search_track"])
def test_instruction_digest_ignores_trusted_geometry_not_expressed_as_goals(tasks, family) -> None:
    original = next(task for task in tasks if task["family"] == family)
    changed = deepcopy(original)
    changed["uavs"][0]["home_xyz_m"][0] += 10
    changed["world"]["flight_altitude_m"] += 2
    changed["world"]["scene_max_xyz_m"][0] += 100
    changed["family"] = "presentation_group"
    changed["scale"] = 999
    changed["closure_policy"] = "a different host policy"
    changed["instruction"] = "a paraphrase"
    changed["assignments"].reverse()
    assert instruction_semantic_hash(changed) == original["instruction_semantic_hash"]
    assert semantic_hash(changed) != original["semantic_hash"]


def test_instruction_digest_keeps_explicit_destination_altitude_and_wait_duration(tasks) -> None:
    navigate = deepcopy(next(task for task in tasks if task["family"] == "navigate"))
    before = instruction_semantic_hash(navigate)
    navigate["assignments"][0]["destination_xyz_m"][2] += 2
    assert instruction_semantic_hash(navigate) != before
    hover = deepcopy(next(task for task in tasks if task["family"] == "hover"))
    before = instruction_semantic_hash(hover)
    hover["assignments"][0]["duration_s"] += 5
    assert instruction_semantic_hash(hover) != before


def test_all_gold_records_pass_contracts_and_match_independent_blueprints(tasks) -> None:
    for task in tasks:
        used_targets = {
            assignment["target_alias"] for assignment in task["assignments"]
            if assignment["target_alias"] is not None
        }
        assert set(task["target_catalog"]) == used_targets
        assert set(task["target_aliases"].values()) == used_targets
        spec = build_task_spec(task)
        request = build_fleet_request(task, spec)
        plan = build_fleet_plan(task, request)
        assert FleetTaskSpecV1.from_dict(spec.to_dict()) == spec
        assert FleetMissionRequestV2.from_dict(request.to_dict()) == request
        assert FleetMissionPlanV2.from_dict(plan.to_dict(), request=request) == plan
        assert plan.semantic_findings(request) == ()
        assert not spec.ambiguities
        assert len(spec.termination_goals) == task["scale"] <= MAX_TERMINATION_GOALS
        assert len({goal.goal_id for goal in (*spec.goals, *spec.termination_goals)}) == len(spec.all_goal_ids)
        all_assigned = [goal_id for assignment in plan.assignments for goal_id in assignment.goal_ids]
        assert len(all_assigned) == len(set(all_assigned)) == len(spec.all_goal_ids)
        assert set(all_assigned) == set(spec.all_goal_ids)
        assert plan.unassigned_goal_ids == ()
        assert all(item.quote in task["instruction"] for item in spec.source_evidence)
        known_evidence = {item.evidence_id for item in spec.source_evidence}
        expected = {assignment["uav_id"]: assignment for assignment in task["assignments"]}
        for fleet_assignment in plan.assignments:
            blueprint = expected[fleet_assignment.uav_id]
            owned_goals = [spec.goal(goal_id) for goal_id in fleet_assignment.goal_ids]
            expected_types = {
                "navigate": ["NAVIGATE", "RETURN_HOME_AND_LAND"],
                "hover": ["WAIT"],
                "search": ["SEARCH_TARGET", "RETURN_HOME_AND_LAND"],
                "search_track": ["SEARCH_TARGET", "TRACK_TARGET", "RETURN_HOME_AND_LAND"],
            }[blueprint["kind"]]
            assert [goal.goal_type.value for goal in owned_goals] == expected_types
            binding = next(c for c in spec.assignment_constraints if c.uav_id == fleet_assignment.uav_id)
            assert binding.strength.value == "MUST"
            assert set(binding.goal_ids) == set(fleet_assignment.goal_ids)
            for goal in owned_goals:
                assert goal.strength.value == "MUST"
                assert goal.evidence_refs and set(goal.evidence_refs) <= known_evidence
                if goal.goal_type.value == "SEARCH_TARGET":
                    assert goal.target_alias == blueprint["target_alias"]
                    assert goal.duration_s is None
                    assert list(goal.spatial_constraint.center_xyz_m) == blueprint["search_center_xyz_m"]
                    assert goal.spatial_constraint.radius_m == blueprint["radius_m"]
                elif goal.goal_type.value == "TRACK_TARGET":
                    assert goal.target_alias == blueprint["target_alias"]
                    assert goal.duration_s == blueprint["duration_s"]
                elif goal.goal_type.value == "NAVIGATE":
                    assert goal.target_alias is None
                    assert list(goal.spatial_constraint.xyz_m) == blueprint["destination_xyz_m"]
                elif goal.goal_type.value == "WAIT":
                    assert goal.duration_s == blueprint["duration_s"]
                    assert goal.uav_id == blueprint["uav_id"]
                else:
                    assert goal.uav_id == blueprint["uav_id"]
            for before, after in zip(owned_goals, owned_goals[1:]):
                assert any(
                    order.before_goal_id == before.goal_id and order.after_goal_id == after.goal_id
                    and order.strength.value == "MUST"
                    for order in spec.ordering_constraints
                )
        assert len(json.dumps(spec.to_dict(), ensure_ascii=False, separators=(",", ":")).encode()) <= 32768


def test_hover_recovery_is_execution_policy_not_an_invented_user_goal(tasks) -> None:
    for task in tasks:
        for assignment in task["assignments"]:
            if assignment["kind"] != "hover":
                assert assignment["closure_policy"] == "explicit_return_home_and_land"
                continue
            assert assignment["terminal_goal_type"] == "WAIT"
            assert assignment["closure_policy"] == "runtime_contract_home_and_land"
            quote = task["source_quotes"][assignment["uav_id"]]
            assert "起飞" in quote and "悬停" in quote
            assert not any(word in quote for word in ("返航", "降落", "调度器", "收尾"))


def test_corpus_varies_units_owners_clause_order_and_mixed_task_kinds(tasks) -> None:
    assert {task["language_variant"] for task in tasks} == {0, 1, 2, 3}
    assert any("厘米" in task["instruction"] for task in tasks)
    assert any("分钟" in task["instruction"] for task in tasks)
    assert any(task["clause_order"] != [uav["uav_id"] for uav in task["uavs"]] for task in tasks)
    assert any(
        assignment["target_alias"] is not None
        and assignment["uav_id"].removeprefix("uav_") != assignment["target_alias"].removeprefix("target_")
        for task in tasks for assignment in task["assignments"]
    )
    for task in tasks:
        if task["family"] == "mixed_parallel":
            assert len({assignment["kind"] for assignment in task["assignments"]}) == min(4, task["scale"])


def test_blueprint_geometry_stays_inside_world_and_separated_home_cells(tasks) -> None:
    for task in tasks:
        low = task["world"]["scene_min_xyz_m"]
        high = task["world"]["scene_max_xyz_m"]
        homes = [uav["home_xyz_m"] for uav in task["uavs"]]
        assert all(dist(a, b) >= 25 for i, a in enumerate(homes) for b in homes[i + 1:])
        circles = []
        for assignment in task["assignments"]:
            point = assignment["destination_xyz_m"]
            if point is not None:
                assert all(low[i] <= point[i] <= high[i] for i in range(3))
            center = assignment["search_center_xyz_m"]
            if center is not None:
                radius = assignment["radius_m"]
                assert all(low[i] <= center[i] - radius <= center[i] + radius <= high[i] for i in range(2))
                circles.append((center, radius))
        assert all(
            dist(center_a, center_b) - radius_a - radius_b >= 5
            for i, (center_a, radius_a) in enumerate(circles)
            for center_b, radius_b in circles[i + 1:]
        )


@pytest.mark.parametrize("count", [1, 25, 53])
def test_small_corpora_keep_exact_global_split_counts(count) -> None:
    generated = generate_task_blueprints(count)
    counts = Counter(task["split"] for task in generated)
    assert len(generated) == count
    assert counts["train"] == count * 8 // 10
    assert counts["validation"] == count // 10
    assert counts["test"] == count - count * 8 // 10 - count // 10


def test_gold_builder_rejects_nonverbatim_evidence_and_wrong_mission(tasks) -> None:
    task = deepcopy(tasks[0])
    task["source_quotes"]["uav_a"] = "不存在于原始指令中的证据"
    with pytest.raises(ValueError, match="not verbatim"):
        build_task_spec(task)
    request = build_fleet_request(tasks[0])
    with pytest.raises(ValueError, match="different blueprint mission"):
        build_fleet_plan(tasks[0], replace(request, fleet_mission_id="wrong_mission"))


@pytest.mark.parametrize("kwargs", [{"count": 0}, {"count": True}, {"seed": False}])
def test_generator_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        generate_task_blueprints(**kwargs)
