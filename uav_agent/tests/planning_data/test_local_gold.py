from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from models.base import ModelResponse
from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.schemas import LandingZoneSpec, PlannerRequest, PlannerWorldContext
from planning_data.local_gold import build_local_gold, validate_local_against_blueprint
from runtime.plan_validator import PlanValidator
from target.types import TargetSpec


ROOT = Path(__file__).resolve().parents[2]
TARGET = TargetSpec(
    "red cube", category="cube", hard_attributes=("color=red",),
    immutable_identity_summary="red cube",
)
HOME = (-20.0, -20.0, 0.0)


def _navigate(x=8.0, y=3.0, z=12.0):
    return {"goal_type": "NAVIGATE", "spatial_constraint": {
        "kind": "POINT", "frame": "WORLD_ENU", "xyz_m": [x, y, z],
    }}


def _search():
    return {"goal_type": "SEARCH_TARGET", "target_alias": "target_i", "spatial_constraint": {
        "shape": "CIRCLE", "frame": "WORLD_ENU", "center_xyz_m": [10.0, 15.0, 0.0], "radius_m": 5.0,
    }}


def _track(duration=20.0):
    return {"goal_type": "TRACK_TARGET", "target_alias": "target_i", "duration_s": duration}


def _wait(duration=5.0):
    return {"goal_type": "WAIT", "duration_s": duration}


def _return():
    return {"goal_type": "RETURN_HOME_AND_LAND"}


def _request(*goals, completion=False, lock=None):
    has_target = any(goal.get("target_alias") for goal in goals)
    payload = {
        "schema_version": 2, "uav_id": "uav_a", "own_home": "home_a",
        "assigned_goals": list(goals),
        "trusted_target_specs": {"target_i": TARGET.to_dict()} if has_target else {},
        "trusted_runtime_safety_completion": completion,
    }
    world = PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0), scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=HOME, search_regions={},
        landing_zones={"home_a": LandingZoneSpec("home_a", HOME[:2], HOME[2])},
        default_takeoff_altitude_m=10.0, default_track_duration_s=20.0, search_timeout_s=60.0,
    )
    return PlannerRequest(
        instruction=json.dumps(payload), world_context=world,
        mission_id="mission_local_gold", uav_id="uav_a", plan_version=1,
        trusted_target_spec=TARGET if has_target else None,
        trusted_target_id=lock, require_empty_spatial_assumptions=True,
        allow_trusted_safety_completion=completion,
    )


def _validate_production(request, draft):
    class Client:
        def chat(self, messages, *, options):
            return ModelResponse(json.dumps(draft.to_dict()), "gold", "stop", {})

    # A valid dataclass alone is insufficient: production planner semantics
    # enforce TAKEOFF, target references and the terminal named-home GOTO.
    planner = DynamicLLMPlanner(
        Client(), ROOT / "prompts/dynamic_skill_planner_v3_system.txt",
        planning_contract="v3", repair_budget=0,
    )
    accepted = planner.plan(request)
    compiled, report = PlanValidator().validate_and_compile_with_report(
        accepted, request.world_context, source="dynamic_llm",
        mission_id=request.mission_id, uav_id=request.uav_id, plan_version=request.plan_version,
        trusted_target_id=request.trusted_target_id,
        allow_trusted_safety_completion=request.allow_trusted_safety_completion,
    )
    assert report.accepted, report.to_dict()
    assert compiled is not None
    return compiled.task_plan


@pytest.mark.parametrize("goals,expected", [
    ((_navigate(), _return()), ["TAKEOFF", "GOTO", "GOTO", "LAND"]),
    ((_wait(), _return()), ["TAKEOFF", "HOVER", "GOTO", "LAND"]),
    ((_navigate(), _wait(), _return()), ["TAKEOFF", "GOTO", "HOVER", "GOTO", "LAND"]),
    ((_navigate(), _navigate(-5, 8, 15), _return()), ["TAKEOFF", "GOTO", "GOTO", "GOTO", "LAND"]),
    ((_search(), _return()), ["TAKEOFF", "SEARCH", "GOTO", "LAND"]),
    ((_search(), _track(), _return()), ["TAKEOFF", "SEARCH", "TRACK", "GOTO", "LAND"]),
])
def test_local_labels_pass_real_dynamic_planner_and_compiler(goals, expected):
    request = _request(*goals)
    draft = build_local_gold(request)
    assert [step.skill for step in draft.steps] == expected
    compiled = _validate_production(request, draft)
    assert [step.skill.value for step in compiled.steps] == expected
    assert compiled.steps[-2].params["position"][:2] == HOME[:2]
    assert compiled.steps[-1].params["expected_position_xy"] == HOME[:2]


def test_no_termination_label_omits_compiler_safety_completion():
    request = _request(_navigate(), completion=True)
    draft = build_local_gold(request)
    assert [step.skill for step in draft.steps] == ["TAKEOFF", "GOTO"]
    compiled = _validate_production(request, draft)
    assert [step.skill.value for step in compiled.steps] == ["TAKEOFF", "GOTO", "GOTO", "LAND"]


def test_track_without_search_requires_real_trusted_target_state():
    with pytest.raises(ValueError, match="prior SEARCH or trusted target lock"):
        build_local_gold(_request(_track(), _return()))
    request = _request(_track(), _return(), lock="confirmed_i")
    draft = build_local_gold(request)
    assert draft.steps[1].args["target_ref"] == "$trusted_target.target_id"
    assert all(step.skill != "SEARCH" for step in draft.steps)
    _validate_production(request, draft)


@pytest.mark.parametrize("goals,message", [
    ((_navigate(),), "lack an explicit terminal LAND"),
    ((_wait(61), _return()), "WAIT duration"),
    ((_search(), _search(), _return()), "one SEARCH"),
    (({"goal_type": "REPORT"},), "unsupported local pilot"),
    ((_return(), _navigate()), "follows LAND"),
])
def test_unsupported_or_incomplete_gold_is_rejected(goals, message):
    with pytest.raises(ValueError, match=message):
        build_local_gold(_request(*goals))


def _blueprint(kind, *, terminal="RETURN_HOME_AND_LAND", closure_policy=None):
    row = {"uav_id": "uav_a", "kind": kind, "home_name": "home_a", "terminal_goal_type": terminal}
    if kind == "navigate":
        row["destination_xyz_m"] = [8.0, 3.0, 12.0]
    if kind in {"search", "search_track"}:
        row.update({"target_alias": "target_i", "search_center_xyz_m": [10.0, 15.0, 0.0], "radius_m": 5.0})
    if kind == "hover":
        row["duration_s"] = 5.0
    if kind == "search_track":
        row["duration_s"] = 20.0
    if closure_policy is not None:
        row["closure_policy"] = closure_policy
    return row


@pytest.mark.parametrize("kind,goals", [
    ("navigate", (_navigate(), _return())),
    ("hover", (_wait(), _return())),
    ("search", (_search(), _return())),
    ("search_track", (_search(), _track(), _return())),
])
def test_blueprint_audit_accepts_correct_raw_and_compiled_plans(kind, goals):
    request = _request(*goals)
    draft = build_local_gold(request)
    compiled = _validate_production(request, draft)
    report = validate_local_against_blueprint(
        _blueprint(kind), draft, compiled, HOME,
        target_spec=TARGET if kind.startswith("search") else None,
    )
    assert report["passed"], report
    assert report["runtime_safety_completion_added"] is False


def test_wait_only_closes_actual_runtime_contract_without_inventing_return_goal():
    request = _request(_wait())
    before = request.instruction
    draft = build_local_gold(request)
    compiled = _validate_production(request, draft)
    assert [step.skill for step in draft.steps] == ["TAKEOFF", "HOVER", "GOTO", "LAND"]
    assert request.instruction == before
    row = _blueprint("hover", terminal="WAIT", closure_policy="runtime_contract_home_and_land")
    report = validate_local_against_blueprint(row, draft, compiled, HOME)
    assert report["passed"], report
    assert report["runtime_contract_closure"] is True
    assert report["runtime_safety_completion_added"] is False


def test_blueprint_distinguishes_compiler_safety_completion_from_model_steps():
    request = _request(_navigate(), completion=True)
    draft = build_local_gold(request)
    compiled = _validate_production(request, draft)
    report = validate_local_against_blueprint(
        _blueprint("navigate", terminal=None), draft, compiled, HOME, allow_safety_completion=True,
    )
    assert report["passed"], report
    assert report["runtime_safety_completion_added"] is True
    # A compiler-added epilogue does not satisfy an explicitly requested model
    # return/landing sequence, even though the final aircraft path is safe.
    failed = validate_local_against_blueprint(_blueprint("navigate"), draft, compiled, HOME)
    assert failed["passed"] is False


@pytest.mark.parametrize("mutation,code", [
    (lambda d, c: d.update(uav_id="uav_b"), "UAV_ROUTING_MISMATCH"),
    (lambda d, c: d["steps"][2]["args"].update(duration_s=21), "DURATION_MISMATCH"),
    (lambda d, c: d["steps"][2]["args"].update(target_ref="$unrelated.target_id"), "TRACK_REFERENCE_MISMATCH"),
    (lambda d, c: d["steps"][1]["args"]["region"].update(radius_m=7), "SEARCH_REGION_MISMATCH"),
    (lambda d, c: d["steps"][1]["args"].update(target_description="blue cube"), "SEARCH_TARGET_MISMATCH"),
    (lambda d, c: c["steps"][-2].update(position=[20.0, 20.0, 10.0]), "HOME_COORDINATES_MISMATCH"),
    (lambda d, c: c["steps"][-1].update(expected_position_xy=[20.0, 20.0]), "LAND_GEOMETRY_MISMATCH"),
    (lambda d, c: d["steps"].reverse(), "STEP_SEQUENCE_MISMATCH"),
    (lambda d, c: c["steps"][2].update(track_duration=3), "DURATION_MISMATCH"),
])
def test_blueprint_audit_catches_changes_that_typed_or_coverage_checks_may_miss(mutation, code):
    request = _request(_search(), _track(), _return())
    draft = build_local_gold(request)
    compiled = _validate_production(request, draft)
    raw, final = deepcopy(draft.to_dict()), deepcopy(compiled.to_dict())
    mutation(raw, final)
    report = validate_local_against_blueprint(_blueprint("search_track"), raw, final, HOME, target_spec=TARGET)
    assert report["passed"] is False
    assert code in {finding["code"] for finding in report["findings"]}


def test_blueprint_catches_navigation_coordinate_error_and_unrequested_search():
    request = _request(_navigate(), _return())
    draft = build_local_gold(request)
    compiled = _validate_production(request, draft).to_dict()
    compiled["steps"][1]["position"] = [9, 3, 12]
    report = validate_local_against_blueprint(_blueprint("navigate"), draft, compiled, HOME)
    assert not report["passed"]
    assert "NAVIGATION_COORDINATES_MISMATCH" in {finding["code"] for finding in report["findings"]}
    raw = draft.to_dict()
    raw["steps"].insert(1, {"id": "unrequested", "uav_id": "uav_a", "skill": "SEARCH", "args": {}})
    report = validate_local_against_blueprint(_blueprint("navigate"), raw, compiled, HOME)
    assert not report["passed"]
    assert "STEP_SEQUENCE_MISMATCH" in {finding["code"] for finding in report["findings"]}
