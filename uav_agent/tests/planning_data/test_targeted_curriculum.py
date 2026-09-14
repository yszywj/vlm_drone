from collections import Counter
from copy import deepcopy
import json
import random

import pytest

from planning_data.generator import _source_hashes, render_task
from planning_data.targeted_curriculum import FOCI, audit_targeted_task, build_targeted_tasks
from planning_data.tasks import SCALES, build_task_spec, generate_task_blueprints


@pytest.fixture(scope="module")
def corpus():
    result = []
    for partition in ("train", "validation", "test"):
        result.extend(build_targeted_tasks(3, partition=partition))
    for index, task in enumerate(result):
        task.update(task_id=f"curriculum_test_{index:06d}", split=task["curriculum_partition"])
    return result


def test_balanced_fresh_partitions_and_reproducibility(corpus):
    expected = {
        (focus, scale, partition): 3
        for focus in FOCI for scale in SCALES
        for partition in ("train", "validation", "test")
    }
    assert Counter((t["curriculum_focus"], t["scale"], t["split"]) for t in corpus) == expected
    for field in ("semantic_hash", "instruction_semantic_hash"):
        assert len({task[field] for task in corpus}) == len(corpus)
        assert {task[field] for task in corpus}.isdisjoint(task[field] for task in generate_task_blueprints())
    templates = {
        split: {t["curriculum_template_id"] for t in corpus if t["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert templates["train"].isdisjoint(templates["validation"] | templates["test"])
    assert templates["validation"].isdisjoint(templates["test"])
    state = random.getstate()
    one = build_targeted_tasks(2)
    assert one == build_targeted_tasks(2)
    assert random.getstate() == state
    assert all("task_id" not in task and "split" not in task for task in one)
    excluded = build_targeted_tasks(
        2,
        exclude_semantic_hashes=[task["semantic_hash"] for task in one],
        exclude_instruction_semantic_hashes=[task["instruction_semantic_hash"] for task in one],
    )
    for field in ("semantic_hash", "instruction_semantic_hash"):
        assert {t[field] for t in one}.isdisjoint(t[field] for t in excluded)


def test_every_focus_scale_partition_passes_all_production_labels(corpus):
    for task in corpus:
        audit_targeted_task(task)
        rows = render_task(task)
        assert len(rows) == task["scale"] + 2
        assert {row["role"] for row in rows} == {"mission_interpreter", "fleet_planner", "spatial_mission"}
        assert all(row["checks"]["blueprint_semantics"] for row in rows)
        assert all(json.loads(row["messages"][-1]["content"]) for row in rows)


def test_foci_contain_the_intended_counterexamples(corpus):
    for task in corpus:
        focus = task["curriculum_focus"]
        spec = build_task_spec(task)
        if focus == "mixed_wait":
            wait_ids = {goal.goal_id for goal in spec.termination_goals if goal.goal_type.value == "WAIT"}
            assert wait_ids and spec.goals
            assert not any(order.before_goal_id in wait_ids or order.after_goal_id in wait_ids for order in spec.ordering_constraints)
        if focus == "axes_zero":
            for assignment in task["assignments"]:
                point = assignment["destination_xyz_m"] or assignment["search_center_xyz_m"]
                assert (point[0] == 0) != (point[1] == 0)
        if focus == "mixed_ownership":
            assert len({a["kind"] for a in task["assignments"]}) == min(4, task["scale"])
        if focus == "duration_contrast":
            durations = [a["duration_s"] for a in task["assignments"] if a["kind"] == "search_track"]
            assert 20 in durations and 120 in durations
        if focus == "minute_conversion":
            assert "分钟" in task["instruction"] and "秒" in task["instruction"]
        assert task["clause_order"] != [uav["uav_id"] for uav in task["uavs"]]
        for assignment in task["assignments"]:
            if assignment["target_alias"]:
                assert assignment["target_alias"].removeprefix("target_") != assignment["uav_id"].removeprefix("uav_")


def _mutate_quote(task, owner, old, new):
    quote = task["source_quotes"][owner]
    assert old in quote
    changed = quote.replace(old, new, 1)
    task["source_quotes"][owner] = changed
    task["instruction"] = task["instruction"].replace(quote, changed, 1)


def test_independent_text_audit_rejects_axis_duration_owner_and_action_corruption(corpus):
    task = deepcopy(next(t for t in corpus if t["curriculum_focus"] == "axes_zero"))
    assignment = task["assignments"][0]
    x, y, _ = assignment["search_center_xyz_m"]
    _mutate_quote(task, assignment["uav_id"], f"({x:g},{y:g})", f"({y:g},{x:g})")
    with pytest.raises(ValueError, match="coordinate axes"):
        audit_targeted_task(task)

    task = deepcopy(next(t for t in corpus if t["curriculum_focus"] == "duration_contrast"))
    assignment = next(a for a in task["assignments"] if a["kind"] == "search_track" and a["duration_s"] == 20)
    _mutate_quote(task, assignment["uav_id"], "20秒", "120秒")
    with pytest.raises(ValueError, match="duration or units"):
        audit_targeted_task(task)

    task = deepcopy(corpus[0])
    _mutate_quote(task, "uav_a", "无人机A", "无人机B")
    with pytest.raises(ValueError, match="unique owner"):
        audit_targeted_task(task)

    task = deepcopy(next(t for t in corpus if t["curriculum_focus"] == "duration_contrast"))
    assignment = next(a for a in task["assignments"] if a["kind"] == "search_track")
    _mutate_quote(task, assignment["uav_id"], "跟踪", "观察")
    with pytest.raises(ValueError, match="action types"):
        audit_targeted_task(task)


def test_partition_reassignment_and_novel_source_hash_are_visible(corpus):
    changed = deepcopy(corpus[0])
    changed["split"] = "test"
    with pytest.raises(ValueError, match="split differs"):
        audit_targeted_task(changed)
    assert "planning_data/targeted_curriculum.py" in _source_hashes()


@pytest.mark.parametrize("kwargs", [
    {"count_per_focus_scale": 0}, {"count_per_focus_scale": True},
    {"seed": True}, {"partition": "dev"},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        build_targeted_tasks(**kwargs)
