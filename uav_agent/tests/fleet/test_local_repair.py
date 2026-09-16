from dataclasses import FrozenInstanceError, replace
from math import pi
from types import SimpleNamespace

import pytest

from fleet.local_repair import (
    ExternalDependencySnapshot, LocalRepairContextV3, LocalRepairDraftV3,
    LocalRepairError, RepairAnchor, SkillExecutionEvidence,
    assess_remaining_goals, build_local_repair_json_schema, check_external_dependencies, extract_external_dependencies,
    validate_local_repair,
)
from fleet.task_spec import AssignmentConstraint, ConstraintStrength, FleetTaskSpecV1, GoalType, MissionGoal, OrderingConstraint, TerminationGoal, SourceEvidence
from planner.schemas import LandingZoneSpec, PlannerWorldContext
from planner.schemas_v3 import SkillPlanDraftV3
from planner.spatial import CoordinateFrame, NamedLocationTarget, PointTarget
from planner.spatial_resolver import FramePose
from runtime.plan_validator import PlanValidator
from skills.types import SkillResult, SkillResultCode, SkillStatus


def world():
    return PlannerWorldContext(scene_min_xyz_m=(-50, -50, 0), scene_max_xyz_m=(50, 50, 30),
        initial_uav_xyz_m=(0, 0, 0), search_regions={}, landing_zones={"home": LandingZoneSpec("home", (0, 0), 0)},
        default_takeoff_altitude_m=10, default_track_duration_s=10, search_timeout_s=75, goto_timeout_s=120, land_timeout_s=60)


def step(id, skill, **args):
    return {"id": id, "uav_id": "uav_3", "skill": skill, "args": args}


def plan(steps):
    return SkillPlanDraftV3.from_dict({"schema_version": 3, "mission_id": "mission_3", "uav_id": "uav_3",
        "plan_version": 1, "assumptions": [], "steps": steps})


def nav_goal(id="goal_nav", xyz=(10, 0, 10)):
    return MissionGoal(id, GoalType.NAVIGATE, None, PointTarget(CoordinateFrame.WORLD_ENU, xyz), None, None, ConstraintStrength.MUST)


def evidence(id, code, **data):
    return SkillExecutionEvidence(id, "invoke_" + id, 1, SkillResult(SkillStatus.SUCCEEDED, code, "trusted", data))


def anchor(**kwargs):
    data = dict(anchor_id="anchor_1", frame_id="frame_1", observation_time_s=20, pose_time_s=20,
        time_domain="simulation", pose=FramePose((1, 0, 10), pi/2), home_pose=FramePose((0, 0, 0)),
        start_pose=FramePose((0, 0, 0)), map_version=1, reference_version=1, named_locations={"home": (0, 0, 0)})
    return RepairAnchor(**{**data, **kwargs})


def context(**kwargs):
    raw = plan([step("takeoff", "TAKEOFF", altitude_m=10), step("goto", "GOTO", target=nav_goal().spatial_constraint.to_dict()),
                step("home", "GOTO", target=NamedLocationTarget("home").to_dict()), step("land", "LAND", zone="home")])
    compiled = PlanValidator().validate_and_compile(raw, world(), source="dynamic_scripted", spatial_resolver=anchor().resolver,
        mission_id=raw.mission_id, uav_id=raw.uav_id, plan_version=raw.plan_version)
    data = dict(fleet_mission_id="fleet_1", assignment_id="assignment_3", request_id="request_1", episode_id="episode_1",
        execution_generation=1, original=compiled, current_step_id="goto", completed_step_ids=("takeoff",),
        completed_step_outputs={"takeoff": {"altitude": 10}}, goals=(nav_goal(),),
        evidence=(evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),), anchor=anchor(), submitted_wall_s=100,
        deadline_wall_s=120, dependency_versions={"fleet": 1})
    return LocalRepairContextV3(**{**data, **kwargs})


def draft(ctx, steps=None, **kwargs):
    data = dict(schema_version=3, request_id=ctx.request_id, episode_id=ctx.episode_id,
        mission_id="mission_3", uav_id="uav_3", base_plan_version=1, new_plan_version=2,
        replace_from_step_id=ctx.current_step_id, anchor_id=ctx.anchor.anchor_id,
        steps=[item.to_dict() for item in ctx.original.planner_output.steps[len(ctx.completed_step_ids):]] if steps is None else steps)
    return LocalRepairDraftV3.from_dict({**data, **kwargs})


def validate(ctx, proposal=None, **kwargs):
    return validate_local_repair(proposal or draft(ctx), ctx, world(),
        dependency_versions=kwargs.pop("dependency_versions", ctx.dependency_versions),
        external_dependencies=kwargs.pop("external_dependencies", ctx.external_dependencies),
        now_wall_s=kwargs.pop("now_wall_s", 101), **kwargs)


def test_compiles_only_authorized_suffix_and_preserves_typed_prefix():
    ctx = context()
    new = [step("detour", "GOTO", target=PointTarget(CoordinateFrame.UAV_HOLD_FLU, (2, 0, 0)).to_dict()),
           *draft(ctx).to_dict()["steps"]]
    candidate = validate(ctx, draft(ctx, new))
    assert candidate.task_plan.plan_version == 2
    assert candidate.replace_from_index == 1
    assert candidate.task_plan.steps[0].to_dict() == ctx.original.task_plan.steps[0].to_dict()
    assert candidate.world_route[1] == pytest.approx((1, 2, 10))
    assert candidate.world_route[-1] == (0, 0, 0)
    assert candidate.compiled_mission.planner_output.schema_version == 3
    assert candidate.completed_step_outputs == ctx.completed_step_outputs


@pytest.mark.parametrize("changed", [{"uav_id": "uav_4"}, {"request_id": "request_old"}, {"episode_id": "episode_old"},
    {"anchor_id": "anchor_old"}, {"base_plan_version": 2, "new_plan_version": 3}, {"replace_from_step_id": "home"}])
def test_route_and_authorization_mutation_is_rejected(changed):
    ctx = context()
    with pytest.raises((LocalRepairError, ValueError)):
        validate(ctx, draft(ctx, **changed))


def test_strict_v3_does_not_ignore_unknown_keys_or_accept_v2():
    data = draft(context()).to_dict()
    with pytest.raises(LocalRepairError):
        LocalRepairDraftV3.from_dict({**data, "other_uav_plan": []})
    with pytest.raises(LocalRepairError):
        LocalRepairDraftV3.from_dict({**data, "schema_version": 2})


def test_context_and_evidence_are_deeply_detached_and_frozen():
    result = SkillResult(SkillStatus.SUCCEEDED, SkillResultCode.TAKEOFF_COMPLETE, "", {"nested": [1]})
    proof = SkillExecutionEvidence("takeoff", "invoke_1", 1, result)
    ctx = context(evidence=(proof,))
    digest = ctx.digest
    result.data["nested"].append(2)
    assert ctx.digest == digest
    with pytest.raises(TypeError):
        ctx.original.task_plan.steps[0].params["altitude"] = 999
    with pytest.raises(TypeError):
        ctx.completed_step_outputs["takeoff"]["altitude"] = 999
    with pytest.raises(FrozenInstanceError):
        ctx.anchor.pose = FramePose((9, 9, 9))
    assert proof.result.data["nested"] == (1,)


def test_anchor_timestamp_mismatch_and_graph_mode_fail_closed():
    with pytest.raises(LocalRepairError, match="aligned"):
        anchor(pose_time_s=22)
    with pytest.raises(LocalRepairError, match="graph"):
        context(mode="GRAPH")


def test_confirmed_navigation_is_removed_and_cannot_be_replayed():
    base = context()
    ctx = replace(base, current_step_id="home", completed_step_ids=("takeoff", "goto"),
        evidence=(*base.evidence, evidence("goto", SkillResultCode.GOAL_REACHED)))
    assert ctx.assessment.confirmed_goal_ids == ("goal_nav",)
    assert ctx.assessment.pending_goal_ids == ()
    validate(ctx)
    proposal = draft(ctx, [step("repeat", "GOTO", target=nav_goal().spatial_constraint.to_dict()), *draft(ctx).to_dict()["steps"]])
    with pytest.raises(LocalRepairError) as error:
        validate(ctx, proposal)
    assert error.value.code == "COMPLETED_GOAL_REPLAY"


def test_completed_prefix_without_success_evidence_is_not_goal_credit():
    base = context()
    ctx = replace(base, current_step_id="home", completed_step_ids=("takeoff", "goto"))
    assert ctx.assessment.insufficient_goal_ids == ("goal_nav",)
    with pytest.raises(LocalRepairError) as error:
        validate(ctx)
    assert error.value.code == "EVIDENCE_INSUFFICIENT"


def test_partial_track_never_uses_reported_elapsed_time_as_remaining_credit():
    raw = plan([step("takeoff", "TAKEOFF", altitude_m=10), step("track", "TRACK", target_ref="$trusted_target.target_id", duration_s=10),
                step("home", "GOTO", target=NamedLocationTarget("home").to_dict()), step("land", "LAND", zone="home")])
    goal = MissionGoal("goal_track", GoalType.TRACK_TARGET, "target_a", None, 10, None, ConstraintStrength.MUST)
    failed = SkillExecutionEvidence("track", "invoke_track", 1,
        SkillResult(SkillStatus.FAILED, SkillResultCode.TARGET_LOST, "", {"tracking_duration": 9, "target_id": "target_1"}))
    result = assess_remaining_goals((goal,), raw, ("takeoff",), (failed,), current_step_id="track")
    assert result.insufficient_goal_ids == ("goal_track",)
    assert result.pending_goals == ()
    assert "PARTIAL_DURATION_EVIDENCE_INSUFFICIENT" in result.reasons


def test_prefix_and_land_mutation_cannot_pass():
    ctx = context()
    data = draft(ctx).to_dict()["steps"]
    data[-1]["id"] = "new_land"
    with pytest.raises(LocalRepairError):
        validate(ctx, draft(ctx, data))
    with pytest.raises(LocalRepairError):
        validate(ctx, draft(ctx, [step("takeoff", "GOTO", target=nav_goal().spatial_constraint.to_dict()), *draft(ctx).to_dict()["steps"]]))


def test_missing_goal_or_changed_world_target_is_rejected():
    ctx = context()
    data = draft(ctx).to_dict()["steps"]
    data[0]["args"]["target"] = PointTarget(CoordinateFrame.WORLD_ENU, (20, 0, 10)).to_dict()
    with pytest.raises(LocalRepairError) as error:
        validate(ctx, draft(ctx, data))
    assert error.value.code == "GOAL_NOT_COVERED"


def test_cross_subset_ordering_and_shared_dependency_are_preserved_and_versioned():
    a, b = nav_goal("goal_a"), nav_goal("goal_b", (20, 0, 10))
    spec = FleetTaskSpecV1(source_text="A before B", goals=(a, b),
        ordering_constraints=(OrderingConstraint("order_ab", "goal_a", "goal_b", ConstraintStrength.MUST),))
    assignments = (SimpleNamespace(goal_ids=("goal_a",), uav_id="uav_1"), SimpleNamespace(goal_ids=("goal_b",), uav_id="uav_3"))
    shared = ExternalDependencySnapshot("channel", "SHARED_RESOURCE", ("goal_b",), ("uav_3", "uav_4"), 1, "UNCHANGED", ("reservation_1",))
    dependencies = extract_external_dependencies(spec, ("goal_b",), assignments=assignments,
        goal_states={"goal_a": "CONFIRMED"}, goal_evidence_refs={"goal_a": ("result_a",)}, versions={"order_ab": 1}, shared_dependencies=(shared,))
    assert {item.dependency_id for item in dependencies} == {"order_ab", "channel"}
    assert check_external_dependencies(dependencies, dependencies, {"fleet": 1}, {"fleet": 1}).allowed
    changed = (replace(shared, version=2), dependencies[1])
    verdict = check_external_dependencies(dependencies, changed, {"fleet": 1}, {"fleet": 1})
    assert not verdict.allowed
    assert verdict.affected_uav_ids == ("uav_1", "uav_3", "uav_4")
    assert not check_external_dependencies(dependencies, dependencies[1:], {}, {}).allowed


def test_external_predecessor_missing_proof_demands_specific_coordination():
    dependency = ExternalDependencySnapshot("order_ab", "PREDECESSOR", ("goal_a", "goal_nav"), ("uav_1", "uav_3"), 1, "UNKNOWN")
    ctx = context(external_dependencies=(dependency,))
    with pytest.raises(LocalRepairError) as error:
        validate(ctx)
    assert error.value.code == "COORDINATION_REQUIRED"
    assert error.value.affected_uav_ids == ("uav_1", "uav_3")


def test_expiry_and_second_dependency_check_can_reject_after_validation():
    ctx = context()
    candidate = validate(ctx)
    assert candidate.task_plan.plan_version == 2
    assert not check_external_dependencies(ctx.external_dependencies, ctx.external_dependencies, ctx.dependency_versions, {"fleet": 2}).allowed
    with pytest.raises(LocalRepairError) as error:
        validate(ctx, now_wall_s=120)
    assert error.value.code == "DEADLINE_EXPIRED"


def test_old_hold_prefix_is_not_reinterpreted_at_new_heading():
    base = context()
    old = base.original.planner_output.to_dict()
    old["steps"][1]["args"]["target"] = PointTarget(CoordinateFrame.UAV_HOLD_FLU, (2, 0, 0)).to_dict()
    original = PlanValidator().validate_and_compile(SkillPlanDraftV3.from_dict(old), world(), source="dynamic_scripted", spatial_resolver=anchor().resolver,
        mission_id="mission_3", uav_id="uav_3", plan_version=1)
    ctx = context(original=original, current_step_id="home", completed_step_ids=("takeoff", "goto"), goals=(),
        anchor=anchor(pose=FramePose((20, 20, 10), 0)))
    candidate = validate(ctx)
    assert candidate.task_plan.steps[1].params["position"] == original.task_plan.steps[1].params["position"]
    assert candidate.task_plan.steps[1].params["position"] == pytest.approx((1, 2, 10))


def search_track_context():
    search = step("search", "SEARCH", region={"shape": "RECTANGLE", "frame": "WORLD_ENU", "center_xyz_m": [10, 0, 0],
        "width_m": 6, "height_m": 6}, strategy={"kind": "LAWNMOWER", "spacing_m": 3},
        entry_policy="START_IN_PLACE_IF_INSIDE", target_description="a moving person", search_altitude_m=10, timeout_s=30)
    raw = plan([step("takeoff", "TAKEOFF", altitude_m=10), search,
        step("goto", "GOTO", target=PointTarget(CoordinateFrame.WORLD_ENU, (10, 0, 10)).to_dict()),
        step("track", "TRACK", target_ref="$search.target_id", duration_s=10, desired_distance_m=5),
        step("home", "GOTO", target=NamedLocationTarget("home").to_dict()), step("land", "LAND", zone="home")])
    compiled = PlanValidator().validate_and_compile(raw, world(), source="dynamic_scripted", spatial_resolver=anchor().resolver,
        mission_id=raw.mission_id, uav_id=raw.uav_id, plan_version=1)
    goal = MissionGoal("goal_track", GoalType.TRACK_TARGET, "target_a", None, 10, 5, ConstraintStrength.MUST)
    return context(original=compiled, completed_step_ids=("takeoff", "search"), goals=(goal,),
        completed_step_outputs={"search": {"target_id": "target_1"}},
        evidence=(evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE), evidence("search", SkillResultCode.TARGET_FOUND, target_id="target_1")),
        trusted_target_id="target_1")


def test_retained_search_reference_requires_same_trusted_output():
    ctx = search_track_context()
    candidate = validate(ctx)
    assert candidate.task_plan.steps[3].params["target_id"].step_id == "search"
    with pytest.raises(LocalRepairError):
        replace(ctx, completed_step_outputs={"search": {"target_id": "other"}})
    with pytest.raises(LocalRepairError) as error:
        validate(replace(ctx, trusted_target_id="other"))
    assert error.value.code == "TARGET_IDENTITY_MISMATCH"


@pytest.mark.parametrize("field,value", [("duration_s", 1), ("desired_distance_m", 2), ("target_ref", "$trusted_target.target_id")])
def test_target_duration_and_distance_cannot_be_weakened(field, value):
    ctx = search_track_context()
    data = draft(ctx).to_dict()["steps"]
    next(item for item in data if item["skill"] == "TRACK")["args"][field] = value
    with pytest.raises(LocalRepairError) as error:
        validate(ctx, draft(ctx, data))
    assert error.value.code == "PROTECTED_STEP_MUTATION"


def test_completed_track_credit_requires_valid_execution_and_caps_at_step_requirement():
    ctx = search_track_context()
    raw = ctx.original.planner_output
    goal = ctx.goals[0]
    proof = track_evidence("invoke_track", SkillStatus.SUCCEEDED, valid=10, elapsed=999)
    assessment = assess_remaining_goals((replace(goal, duration_s=15),), raw,
        ("takeoff", "search", "goto", "track"), (*ctx.evidence, proof), current_step_id="home")
    assert assessment.pending_goals[0].duration_s == 5
    assert assessment.confirmed_goal_ids == ()


def track_evidence(invocation, status, *, valid, elapsed=None, continuous=None,
                   required=10, basis="valid_execution", **changes):
    data = dict(progress_schema="track_progress.v1", target_id="target_1",
        elapsed_s=valid if elapsed is None else elapsed, valid_execution_s=valid,
        continuous_execution_s=valid if continuous is None else continuous,
        required_duration_s=required, completion_basis=basis)
    return SkillExecutionEvidence("track", invocation, 1,
        SkillResult(status, SkillResultCode.TRACK_COMPLETE if status is SkillStatus.SUCCEEDED
                    else SkillResultCode.TARGET_LOST, "", {**data, **changes}))


def track_assessment(proofs, *, basis="valid_execution"):
    ctx = search_track_context()
    payload = ctx.original.planner_output.to_dict()
    next(step for step in payload["steps"] if step["skill"] == "TRACK")["args"]["completion_basis"] = basis
    original = SkillPlanDraftV3.from_dict(payload)
    goals = tuple(replace(goal, completion_basis=basis) for goal in ctx.goals)
    return assess_remaining_goals(goals, original,
        ("takeoff", "search", "goto", "track"), (*ctx.evidence, *proofs), current_step_id="home")


def test_legacy_elapsed_only_track_success_cannot_confirm_goal():
    assessment = track_assessment((evidence("track", SkillResultCode.TRACK_COMPLETE,
        target_id="target_1", tracking_duration=100),))
    assert assessment.insufficient_goal_ids == ("goal_track",)
    assert not assessment.confirmed_goal_ids


@pytest.mark.parametrize("changes", [
    {"valid_execution_s": 2}, {"elapsed_s": 9}, {"continuous_execution_s": 11},
    {"valid_execution_s": float("nan")}, {"required_duration_s": 0},
    {"progress_schema": "model_claim"}, {"completion_basis": "elapsed"},
])
def test_invalid_or_incomplete_track_ledger_is_not_terminal_goal_evidence(changes):
    # Nonfinite evidence itself is rejected before assessment.
    if changes.get("valid_execution_s") != changes.get("valid_execution_s"):
        with pytest.raises(ValueError):
            track_evidence("done", SkillStatus.SUCCEEDED, valid=10, **changes)
    else:
        assessment = track_assessment((track_evidence("done", SkillStatus.SUCCEEDED, valid=10, **changes),))
        assert assessment.insufficient_goal_ids == ("goal_track",)


def test_reacquire_progress_uses_each_trusted_invocation_once():
    first = track_evidence("lost", SkillStatus.FAILED, valid=3, elapsed=8)
    last = track_evidence("done", SkillStatus.SUCCEEDED, valid=7, required=7)
    assessment = track_assessment((first, first, last))
    assert assessment.confirmed_goal_ids == ("goal_track",)
    assert assessment.supported
    assert not track_assessment((last,)).supported


def test_continuous_track_cannot_join_intervals_across_loss():
    first = track_evidence("lost", SkillStatus.FAILED, valid=6, continuous=0, basis="continuous")
    last = track_evidence("done", SkillStatus.SUCCEEDED, valid=4, basis="continuous")
    assert not track_assessment((first, last), basis="continuous").supported
    completed = track_evidence("done", SkillStatus.SUCCEEDED, valid=10, basis="continuous")
    assert track_assessment((first, completed), basis="continuous").confirmed_goal_ids == ("goal_track",)


def test_reacquire_credit_cannot_change_target_identity():
    first = track_evidence("lost", SkillStatus.FAILED, valid=3, target_id="another_target")
    last = track_evidence("done", SkillStatus.SUCCEEDED, valid=7, required=7)
    assert not track_assessment((first, last)).supported


@pytest.mark.parametrize("strength", tuple(ConstraintStrength))
def test_local_assignment_uses_live_owner_evidence_not_optional_source_citation(strength):
    spec = FleetTaskSpecV1(source_text="navigate", goals=(nav_goal(),),
        assignment_constraints=(AssignmentConstraint("owner", "uav_3", ("goal_nav",), strength),))
    assignments = (SimpleNamespace(assignment_id="assignment_3", goal_ids=("goal_nav",), uav_id="uav_3"),)
    deps = extract_external_dependencies(spec, ("goal_nav",), assignments=assignments,
        goal_states={}, goal_evidence_refs={}, versions={})
    assert deps[0].scope == "LOCAL"
    assert deps[0].strength == strength.value
    assert deps[0].evidence_refs
    assert deps[0].source_evidence_refs == ()
    assert check_external_dependencies(deps, deps, {}, {}).allowed
    validate(context(external_dependencies=deps))


@pytest.mark.parametrize("strength,allowed", [(ConstraintStrength.MUST, False),
    (ConstraintStrength.PREFER, True), (ConstraintStrength.OPEN, True)])
def test_assignment_strength_respects_unchanged_accepted_owners(strength, allowed):
    spec = FleetTaskSpecV1(source_text="navigate", goals=(nav_goal(),),
        source_evidence=(SourceEvidence("source_1", "navigate"),),
        assignment_constraints=(AssignmentConstraint("owner", "uav_1", ("goal_nav",), strength,
            evidence_refs=("source_1",)),))
    deps = extract_external_dependencies(spec, ("goal_nav",),
        assignments=(SimpleNamespace(goal_ids=("goal_nav",), uav_id="uav_3"),),
        goal_states={}, goal_evidence_refs={}, versions={})
    assert deps[0].source_evidence_refs == ("source_1",)
    assert check_external_dependencies(deps, deps, {}, {}).allowed is allowed


@pytest.mark.parametrize("assignments", [(),
    (SimpleNamespace(goal_ids=("goal_nav",), uav_id="uav_3"),
     SimpleNamespace(goal_ids=("goal_nav",), uav_id="uav_3"))])
def test_missing_or_duplicate_live_owners_are_not_authorized_by_source_citation(assignments):
    spec = FleetTaskSpecV1(source_text="navigate", goals=(nav_goal(),),
        source_evidence=(SourceEvidence("source_1", "navigate"),),
        assignment_constraints=(AssignmentConstraint("owner", "uav_3", ("goal_nav",),
            ConstraintStrength.OPEN, evidence_refs=("source_1",)),))
    deps = extract_external_dependencies(spec, ("goal_nav",), assignments=assignments,
        goal_states={}, goal_evidence_refs={}, versions={})
    assert deps[0].state == "UNKNOWN"
    assert not check_external_dependencies(deps, deps, {}, {}).allowed


def test_cross_assignment_binding_is_retained_and_detects_owner_swaps():
    a, b = nav_goal("goal_a"), nav_goal("goal_b", (20, 0, 10))
    spec = FleetTaskSpecV1(source_text="navigate", goals=(a, b),
        assignment_constraints=(AssignmentConstraint("owner", "uav_3", ("goal_a", "goal_b"), ConstraintStrength.PREFER),))
    def dependencies(owners):
        return extract_external_dependencies(spec, ("goal_a",),
            assignments=tuple(SimpleNamespace(goal_ids=(goal,), uav_id=uav) for goal, uav in owners),
            goal_states={}, goal_evidence_refs={}, versions={})
    before = dependencies((("goal_a", "uav_3"), ("goal_b", "uav_4")))
    after = dependencies((("goal_a", "uav_4"), ("goal_b", "uav_3")))
    assert before[0].scope == "EXTERNAL"
    assert before[0].uav_ids == after[0].uav_ids
    assert not check_external_dependencies(before, after, {}, {}).allowed


@pytest.mark.parametrize("change", ("remove_current", "rename_current", "move_existing_step"))
def test_detour_must_retain_interrupted_boundary_without_moving_existing_effects(change):
    ctx = context()
    steps = draft(ctx).to_dict()["steps"]
    if change == "remove_current":
        steps.pop(0)
    elif change == "rename_current":
        steps[0]["id"] = "different_boundary"
    else:
        steps[0], steps[1] = steps[1], steps[0]
    with pytest.raises(LocalRepairError) as error:
        validate(ctx, draft(ctx, steps))
    assert error.value.code in {"CURRENT_STEP_MUTATION", "INVALID_DETOUR_PREFIX"}


def test_suffix_wire_schema_binds_scope_without_disguising_v3_as_v2():
    ctx = context()
    schema = build_local_repair_json_schema(ctx)
    assert schema["properties"]["schema_version"] == {"const": 3}
    assert schema["properties"]["request_id"] == {"const": ctx.request_id}
    assert set(schema["properties"]) == set(draft(ctx).to_dict())
    assert schema["additionalProperties"] is False
    variants = schema["properties"]["steps"]["items"]["oneOf"]
    assert all(item["properties"]["skill"]["const"] != "TAKEOFF" for item in variants)


def test_terminal_external_predecessor_can_be_confirmed_from_skill_result():
    ctx = context()
    assessment = assess_remaining_goals(ctx.goals, ctx.original.planner_output,
        tuple(step.id for step in ctx.original.planner_output.steps),
        (evidence("goto", SkillResultCode.GOAL_REACHED),), current_step_id=None)
    assert assessment.confirmed_goal_ids == ("goal_nav",)
    assert assessment.supported


def test_prior_failed_attempt_does_not_erase_later_unique_success():
    ctx = context()
    failed = SkillExecutionEvidence("goto", "invoke_failed", 1,
        SkillResult(SkillStatus.FAILED, SkillResultCode.TIMEOUT, "", {}))
    success = evidence("goto", SkillResultCode.GOAL_REACHED)
    assessment = assess_remaining_goals(ctx.goals, ctx.original.planner_output, ("takeoff", "goto"),
        (failed, success), current_step_id="home")
    assert assessment.confirmed_goal_ids == ("goal_nav",)
    with pytest.raises(LocalRepairError) as error:
        assess_remaining_goals(ctx.goals, ctx.original.planner_output, ("takeoff", "goto"),
            (success, replace(success, invocation_id="invoke_duplicate_effect")), current_step_id="home")
    assert error.value.code == "AMBIGUOUS_EXECUTION_EVIDENCE"


def test_context_refuses_silently_cropped_cross_subset_edge():
    edge = OrderingConstraint("order_ab", "goal_a", "goal_nav", ConstraintStrength.MUST)
    with pytest.raises(LocalRepairError) as error:
        context(ordering_constraints=(edge,))
    assert error.value.code == "EXTERNAL_DEPENDENCY_MISSING"


def test_anchor_digest_includes_original_home_and_start_reference():
    ctx = context()
    changed = replace(ctx, anchor=replace(ctx.anchor, start_pose=FramePose((8, 8, 0))))
    assert ctx.digest != changed.digest
    with pytest.raises(LocalRepairError) as error:
        validate(ctx, now_wall_s=99)
    assert error.value.code == "WALL_CLOCK_REGRESSION"


def test_search_entry_is_frozen_to_same_world_geometry_used_for_admission():
    from planner.region_compiler import RegionCompiler
    from skills.search_strategy import SearchEntryPolicy
    base = search_track_context()
    ctx = replace(base, current_step_id="search", completed_step_ids=("takeoff",),
        completed_step_outputs={}, evidence=(base.evidence[0],), trusted_target_id=None)
    candidate = validate(ctx)
    params = candidate.task_plan.steps[1].params
    assert params["entry_policy"] is SearchEntryPolicy.MODEL_SELECTED
    admitted_entry = params["model_selected_entry_xyz_m"]
    runtime_geometry = RegionCompiler(ctx.anchor.resolver).compile(region=params["region"], strategy=params["strategy"],
        entry_policy=params["entry_policy"], current_uav_xyz_m=(9, 1, 10), search_altitude_m=params["search_altitude_m"],
        model_selected_entry_xyz_m=admitted_entry)
    assert runtime_geometry.entry_point_xyz_m == admitted_entry
    assert candidate.world_route[1] == admitted_entry


def test_local_ordering_cannot_be_reversed_while_preserving_both_goals():
    base = context()
    second = nav_goal("goal_second", (20, 0, 10))
    data = base.original.planner_output.to_dict()
    data["steps"].insert(2, step("second", "GOTO", target=second.spatial_constraint.to_dict()))
    raw = SkillPlanDraftV3.from_dict(data)
    compiled = PlanValidator().validate_and_compile(raw, world(), source="dynamic_scripted", spatial_resolver=anchor().resolver,
        mission_id=raw.mission_id, uav_id=raw.uav_id, plan_version=1)
    edge = OrderingConstraint("order_two", "goal_nav", "goal_second", ConstraintStrength.MUST)
    ctx = context(original=compiled, goals=(nav_goal(), second), ordering_constraints=(edge,))
    values = draft(ctx).to_dict()["steps"]
    values[0]["args"]["target"], values[1]["args"]["target"] = values[1]["args"]["target"], values[0]["args"]["target"]
    with pytest.raises(LocalRepairError) as error:
        validate(ctx, draft(ctx, values))
    assert error.value.code == "ORDERING_CONSTRAINT_VIOLATION"
