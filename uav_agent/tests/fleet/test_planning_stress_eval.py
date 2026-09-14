from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from fleet.task_spec import AssignmentConstraint, FleetTaskSpecV1, MissionGoal, OrderingConstraint, TerminationGoal
from models import ModelResponse
from models.adapter_registry import ModelCallRole
from planner.spatial import CircleRegion
from scripts import evaluate_fleet_planning_stress as evaluation


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/benchmarks/fleet_open_4x4.json"


def _task(case):
    goals, constraints, endings, ordering = [], [], [], []
    for i, t in enumerate(case["gold"]["targets"]):
        uav_id = t["uav_id"] or f"uav_{chr(97 + i)}"
        search_id, track_id, end_id = f"search_{i}", f"track_{i}", f"end_{i}"
        goals += [MissionGoal(search_id, "SEARCH_TARGET", t["target_alias"],
            CircleRegion("WORLD_ENU", tuple(t["search_center_xyz_m"]), t["search_radius_m"]), None, None, "MUST"),
            MissionGoal(track_id, "TRACK_TARGET", t["target_alias"], None, t["track_duration_s"], None, "MUST")]
        if t["uav_id"]:
            constraints.append(AssignmentConstraint(f"constraint_{i}", uav_id, (search_id, track_id, end_id), "MUST"))
        endings.append(TerminationGoal(end_id, "RETURN_HOME_AND_LAND", uav_id, None, "MUST"))
        ordering.append(OrderingConstraint(f"order_{i}", search_id, track_id, "MUST"))
    return FleetTaskSpecV1(case["instruction"], tuple(goals), tuple(constraints),
                          ordering_constraints=tuple(ordering), termination_goals=tuple(endings)).to_dict()


def _plan(request, case):
    return {"schema_version": 2, "fleet_mission_id": request["fleet_mission_id"],
        "fleet_plan_version": request["fleet_plan_version"], "coordination_policy": request["coordination_policy"],
        "assumptions": [], "unassigned_goal_ids": [], "assignments": [
            {"assignment_id": f"assignment_{i}", "uav_id": t["uav_id"] or f"uav_{chr(97+i)}",
             "goal_ids": [f"search_{i}", f"track_{i}", f"end_{i}"], "priority": 100,
             "start_policy": "PARALLEL", "deviations": []} for i, t in enumerate(case["gold"]["targets"])]}


def _local(messages):
    outer = json.loads(messages[1].content)
    routing = outer["trusted_routing"]
    focused = json.loads(outer["user_instruction"])
    search = next(g for g in focused["assigned_goals"] if g["goal_type"] == "SEARCH_TARGET")
    track = next(g for g in focused["assigned_goals"] if g["goal_type"] == "TRACK_TARGET")
    target = next(iter(focused["trusted_target_specs"].values()))
    home, uav = focused["own_home"], routing["uav_id"]
    steps = [
        ("takeoff_1", "TAKEOFF", {"altitude_m": 10.0}),
        ("search_1", "SEARCH", {"region": search["spatial_constraint"],
            "strategy": {"kind": "SPIRAL_OUT", "spacing_m": 4.0}, "entry_policy": "START_IN_PLACE_IF_INSIDE",
            "target_description": target["original_description"], "search_altitude_m": 10.0, "timeout_s": 60.0}),
        ("track_1", "TRACK", {"target_ref": "$search_1.target_id", "duration_s": track["duration_s"]}),
        ("goto_home", "GOTO", {"target": {"kind": "NAMED_LOCATION", "name": home}}),
        ("land_1", "LAND", {"zone": home}),
    ]
    return {"schema_version": 3, "mission_id": routing["mission_id"], "uav_id": uav,
            "plan_version": routing["plan_version"], "target_spec": target, "assumptions": [],
            "steps": [{"id": step, "uav_id": uav, "skill": skill, "args": args} for step, skill, args in steps]}


def _fake_factory(monkeypatch, case, *, repair_interpreter=False, fail_fleet=False, wrong_duration=False, fail_local=None):
    class Client:
        def __init__(self, role, uav_id):
            self.role, self.uav_id, self.calls = role, uav_id, 0

        def chat(self, messages, *, options=None):
            self.calls += 1
            if self.role is ModelCallRole.MISSION_INTERPRETATION:
                if repair_interpreter and self.calls == 1:
                    return ModelResponse("{", "fake", "length", {"prompt_tokens": 10, "completion_tokens": 1})
                payload = _task(case)
                if wrong_duration:
                    payload["goals"][1]["duration_s"] = 1.0
            elif self.role is ModelCallRole.FLEET_PLAN:
                if fail_fleet:
                    raise ConnectionError("offline fake failure")
                payload = _plan(json.loads(messages[1].content)["trusted_request"], case)
            else:
                if self.uav_id == fail_local:
                    return ModelResponse("{}", "fake", "stop", {})
                payload = _local(messages)
            return ModelResponse(json.dumps(payload, ensure_ascii=False), "fake", "stop", {"prompt_tokens": 10, "completion_tokens": 20})

    class Factory:
        def __init__(self, *args, **kwargs):
            pass

        def for_role(self, role, **routing):
            return Client(role, routing.get("uav_id"))

    monkeypatch.setattr(evaluation, "ModelClientFactory", Factory)


def _run(monkeypatch, tmp_path, **kwargs):
    case = evaluation.load_suite(SUITE)[0]
    _fake_factory(monkeypatch, case, **kwargs)
    args = evaluation.build_parser().parse_args(["--suite", str(SUITE)])
    before = {x for x in sys.modules if x.startswith(("isaacsim", "omni"))}
    report = evaluation.run_case(case, args, tmp_path / "case")
    assert {x for x in sys.modules if x.startswith(("isaacsim", "omni"))} == before
    return report


def test_validate_only_does_not_construct_model_client(monkeypatch, capsys):
    monkeypatch.setattr(evaluation, "ModelClientFactory", lambda *a, **k: pytest.fail("model client used"))
    assert evaluation.main(["--suite", str(SUITE), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0


def test_gold_accepts_reordered_goals_and_split_termination():
    case = evaluation.load_suite(SUITE)[0]
    spec = _task(case)
    spec["goals"].reverse()
    terminal = spec["termination_goals"].pop()
    spec["termination_goals"].extend([{**terminal, "goal_type": "RETURN_HOME"},
        {**terminal, "goal_id": "land_split", "goal_type": "LAND"}])
    assert evaluation.score_interpretation(spec, case["gold"], [u.id for u in case["loaded_config"].uavs])["passed"]


def test_gold_rejects_wrong_duration_and_swapped_uav():
    case = evaluation.load_suite(SUITE)[0]
    spec = _task(case)
    spec["goals"][1]["duration_s"] = 10
    score = evaluation.score_interpretation(spec, case["gold"], [u.id for u in case["loaded_config"].uavs])
    assert "target_i:track_duration" in score["findings"]
    plan = _plan({"fleet_mission_id": "fake", "fleet_plan_version": 1, "coordination_policy": {}}, case)
    plan["assignments"][0]["uav_id"] = "uav_b"
    assert not evaluation.score_assignment(plan, spec, case["gold"])["passed"]


def test_gold_rejects_missing_or_reversed_search_track_order():
    case = evaluation.load_suite(SUITE)[0]
    spec = _task(case)
    order = spec["ordering_constraints"][0]
    order["before_goal_id"], order["after_goal_id"] = order["after_goal_id"], order["before_goal_id"]
    score = evaluation.score_interpretation(spec, case["gold"], [u.id for u in case["loaded_config"].uavs])
    assert "target_i:search_before_track_order" in score["findings"]


def test_gold_rejects_cross_target_serialization():
    case = evaluation.load_suite(SUITE)[0]
    spec = _task(case)
    spec["ordering_constraints"].append(OrderingConstraint("cross_order", "track_0", "search_1", "MUST").to_dict())
    score = evaluation.score_interpretation(spec, case["gold"], [u.id for u in case["loaded_config"].uavs])
    assert "cross_target_serialization" in score["findings"]


def test_real_offline_pipeline_scores_all_four_local_plans(monkeypatch, tmp_path):
    report = _run(monkeypatch, tmp_path)
    assert report["first_pass"] and report["final_pass"], report
    assert report["totals"]["model_calls"] == 6
    assert len(report["stages"]["local"]) == 4
    assert all(c["response_text"] and c["usage"] for c in report["model_calls"])
    assert json.loads((tmp_path / "case/result.json").read_text())["final_pass"]


def test_repair_success_does_not_hide_failed_first_attempt(monkeypatch, tmp_path):
    report = _run(monkeypatch, tmp_path, repair_interpreter=True)
    assert report["final_pass"] and not report["first_pass"]
    assert report["stages"]["interpreter"]["diagnostics"]["repair_used"]
    assert report["output_token_limit_hit"]
    assert report["model_calls"][0]["response_text"] == "{"


def test_schema_valid_semantic_error_cannot_pass_end_to_end(monkeypatch, tmp_path):
    report = _run(monkeypatch, tmp_path, wrong_duration=True)
    assert report["stages"]["interpreter"]["diagnostics"]["final_output_valid"]
    assert all(s["final_pass"] for s in report["stages"]["local"].values())
    assert not report["final_pass"]


def test_failed_later_stage_keeps_first_stage_and_call_audit(monkeypatch, tmp_path):
    report = _run(monkeypatch, tmp_path, fail_fleet=True)
    saved = json.loads((tmp_path / "case/result.json").read_text())
    assert saved["stages"]["interpreter"]["output"]
    assert saved["stages"]["interpreter"]["model_proposals"]
    assert saved["service_failure"]
    assert len(saved["model_calls"]) == 2
    assert report["status"] == "failed" and not report["final_pass"]


def test_rescore_saved_report_without_model_or_mutation(monkeypatch, tmp_path):
    report = _run(monkeypatch, tmp_path)
    report["stages"]["interpreter"]["output"]["ordering_constraints"] = []
    monkeypatch.setattr(evaluation, "ModelClientFactory", lambda *a, **k: pytest.fail("model client used"))
    updated = evaluation.rescore_report(report)
    assert report["final_pass"]
    assert not updated["final_pass"]
    assert updated["grading_version"] == 2
    assert updated["model_calls"] == report["model_calls"]


def test_one_local_failure_preserves_other_uavs(monkeypatch, tmp_path):
    report = _run(monkeypatch, tmp_path, fail_local="uav_a")
    local = report["stages"]["local"]
    assert len(local["uav_a"]["attempts"]) == 3
    assert not local["uav_a"]["final_pass"]
    assert all(local[u]["final_pass"] for u in ("uav_b", "uav_c", "uav_d"))
    assert not report["final_pass"]
