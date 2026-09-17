"""Unified remaining-task, spatial-reference and cross-UAV dependency contracts."""
from dataclasses import replace
from math import pi
from types import SimpleNamespace

import pytest

from fleet.local_repair import (
    DependencyVerdict, ExternalDependencySnapshot, LocalRepairDraftV3, LocalRepairError,
    RestartPolicy, SpatialReferenceVerdict, Transferability, RepairAnchor, SkillExecutionEvidence,
    assess_remaining_goals, build_remaining_task_contract, check_cross_uav_dependencies,
    evaluate_spatial_reference, reproject_world_route,
)
from fleet.task_spec import AssignmentConstraint, ConstraintStrength, FleetTaskSpecV1, GoalType, MissionGoal, OrderingConstraint
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


def track_goal(duration, *, basis="valid_execution", alias="target_a"):
    return MissionGoal("goal_track", GoalType.TRACK_TARGET, alias, None, duration, None,
                       ConstraintStrength.MUST, completion_basis=basis)


def evidence(id, code, **data):
    return SkillExecutionEvidence(id, "invoke_" + id, 1, SkillResult(SkillStatus.SUCCEEDED, code, "trusted", data))


def track_evidence(invocation, status, *, valid, elapsed=None, continuous=None, required=10,
                   basis="valid_execution", step_id="track", **changes):
    data = dict(progress_schema="track_progress.v1", target_id="target_1",
                elapsed_s=valid if elapsed is None else elapsed, valid_execution_s=valid,
                continuous_execution_s=valid if continuous is None else continuous,
                required_duration_s=required, completion_basis=basis)
    return SkillExecutionEvidence(step_id, invocation, 1,
        SkillResult(status, SkillResultCode.TRACK_COMPLETE if status is SkillStatus.SUCCEEDED
                    else SkillResultCode.TARGET_LOST, "", {**data, **changes}))


def anchor_pose():
    return FramePose((1, 0, 10), pi / 2)


def anchor(**kwargs):
    data = dict(anchor_id="anchor_1", frame_id="frame_1", observation_time_s=20, pose_time_s=20,
        time_domain="simulation", pose=anchor_pose(), home_pose=FramePose((0, 0, 0)),
        start_pose=FramePose((0, 0, 0)), map_version=1, reference_version=1, named_locations={"home": (0, 0, 0)})
    return RepairAnchor(**{**data, **kwargs})


def reference_arguments(**overrides):
    data = dict(observation_time_s=20, pose_time_s=20, time_domain="simulation",
        current_pose_xyz_m=(1, 0, 10), now_wall_s=110, submitted_wall_s=100,
        max_anchor_age_s=45, max_pose_time_error_s=0.05, max_hold_drift_m=2.0, valid_pose_tolerance_m=0.05)
    return {**data, **overrides}


def compile_plan(raw):
    return PlanValidator().validate_and_compile(raw, world(), source="dynamic_scripted",
        spatial_resolver=anchor().resolver, mission_id=raw.mission_id, uav_id=raw.uav_id, plan_version=raw.plan_version)


# ---------------------------------------------------------------------------
# 1/2. RemainingTaskContract duration arithmetic
# ---------------------------------------------------------------------------

def test_track_30s_with_20s_valid_credit_leaves_exact_10s_remaining():
    raw = plan([step("takeoff", "TAKEOFF", altitude_m=10),
                step("track", "TRACK", target_ref="$trusted_target.target_id", duration_s=20),
                step("home", "GOTO", target=NamedLocationTarget("home").to_dict()),
                step("land", "LAND", zone="home")])
    goal = track_goal(30)
    proofs = (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),
              track_evidence("done", SkillStatus.SUCCEEDED, valid=20, elapsed=25, required=20))
    contract = build_remaining_task_contract((goal,), raw, ("takeoff", "track"), proofs, current_step_id="home")
    entry = contract.entry("goal_track")
    assert contract.supported
    assert entry.status == "PENDING"
    assert entry.confirmed_amount == pytest.approx(20.0)
    assert entry.remaining_amount == pytest.approx(10.0)
    assert entry.restart_policy is RestartPolicy.CONTINUE
    assert entry.completion_basis == "valid_execution"
    assert entry.evidence_refs == ("done",)  # only the credited TRACK invocation
    assert contract.pending_goals[0].duration_s == pytest.approx(10.0)
    assert contract.to_dict()["computed_by"] == "skill_terminal_evidence"
    assert contract.to_dict()["read_only"] is True


def test_continuous_track_loss_cannot_inherit_pre_loss_duration():
    raw = plan([step("takeoff", "TAKEOFF", altitude_m=10),
                step("track_1", "TRACK", target_ref="$trusted_target.target_id", duration_s=20, completion_basis="continuous"),
                step("track_2", "TRACK", target_ref="$trusted_target.target_id", duration_s=10, completion_basis="continuous")])
    goal = track_goal(30, basis="continuous")
    proofs = (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),
              track_evidence("done", SkillStatus.SUCCEEDED, valid=20, continuous=20, required=20, basis="continuous", step_id="track_1"))
    contract = build_remaining_task_contract((goal,), raw, ("takeoff", "track_1"), proofs,
                                             current_step_id="track_2", current_step_started=False)
    entry = contract.entry("goal_track")
    # A lost lock resets continuous credit: the full 30s remain even though
    # 20 continuous seconds were executed in the completed first interval.
    assert entry.status == "PENDING"
    assert entry.confirmed_amount == pytest.approx(0.0)
    assert entry.remaining_amount == pytest.approx(30.0)
    assert entry.restart_policy is RestartPolicy.RESTART
    assert contract.pending_goals[0].duration_s == pytest.approx(30.0)


def test_valid_execution_joins_reacquire_intervals_but_continuous_does_not():
    lost = track_evidence("lost", SkillStatus.FAILED, valid=6)
    done = track_evidence("done", SkillStatus.SUCCEEDED, valid=4, required=4)
    joined = track_evidence("done_full", SkillStatus.SUCCEEDED, valid=10, required=10)
    continuous_lost = track_evidence("lost_c", SkillStatus.FAILED, valid=6, basis="continuous")
    continuous_done = track_evidence("done_c", SkillStatus.SUCCEEDED, valid=4, required=4, basis="continuous")
    continuous_full = track_evidence("done_full_c", SkillStatus.SUCCEEDED, valid=10, required=10, basis="continuous")
    raw = plan([step("takeoff", "TAKEOFF", altitude_m=10),
                step("track", "TRACK", target_ref="$trusted_target.target_id", duration_s=10)])
    continuous_raw = plan([step("takeoff", "TAKEOFF", altitude_m=10),
                step("track", "TRACK", target_ref="$trusted_target.target_id", duration_s=10, completion_basis="continuous")])
    base = (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),)
    # valid_execution banks every trusted interval: 6 + 4 confirms the goal.
    contract = build_remaining_task_contract((track_goal(10),), raw, ("takeoff", "track"),
        base + (lost, done), current_step_id=None)
    assert contract.entry("goal_track").status == "CONFIRMED"
    assert contract.entry("goal_track").confirmed_amount == pytest.approx(10.0)
    # Continuous cannot join across loss: a 4s terminal interval is not enough
    # and the lost 6s cannot be inherited, so the goal fails closed.
    continuous = build_remaining_task_contract((track_goal(10, basis="continuous"),), continuous_raw,
        ("takeoff", "track"), base + (continuous_lost, continuous_done), current_step_id=None)
    assert continuous.entry("goal_track").status == "INSUFFICIENT"
    assert continuous.entry("goal_track").restart_policy is RestartPolicy.CANNOT_RESUME
    completed = build_remaining_task_contract((track_goal(10, basis="continuous"),), continuous_raw,
        ("takeoff", "track"), base + (continuous_lost, continuous_full), current_step_id=None)
    assert completed.entry("goal_track").status == "CONFIRMED"


# ---------------------------------------------------------------------------
# 3/4. Confirmed goals never re-execute; insufficient evidence fails closed
# ---------------------------------------------------------------------------

def confirmed_nav_context():
    raw = plan([step("takeoff", "TAKEOFF", altitude_m=10), step("goto", "GOTO", target=nav_goal().spatial_constraint.to_dict()),
                step("home", "GOTO", target=NamedLocationTarget("home").to_dict()), step("land", "LAND", zone="home")])
    compiled = compile_plan(raw)
    proofs = (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE), evidence("goto", SkillResultCode.GOAL_REACHED))
    contract = build_remaining_task_contract((nav_goal(),), compiled.planner_output, ("takeoff", "goto"),
                                             proofs, current_step_id="home")
    return compiled, contract


def test_confirmed_goal_is_never_reexecuted_by_repair_or_handoff():
    compiled, contract = confirmed_nav_context()
    entry = contract.entry("goal_nav")
    assert entry.status == "CONFIRMED"
    assert entry.restart_policy is RestartPolicy.COMPLETED
    assert entry.remaining_goal is None
    assert contract.pending_goal_ids == ()
    assert contract.handoff_entries == ()
    # The same single computation feeds both consumers unchanged.
    again = build_remaining_task_contract((nav_goal(),), compiled.planner_output, ("takeoff", "goto"),
        (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE), evidence("goto", SkillResultCode.GOAL_REACHED)),
        current_step_id="home", consumer="HANDOFF")
    assert again.to_dict() == contract.to_dict()


def test_insufficient_evidence_yields_cannot_resume_and_fail_closed_validation():
    raw = plan([step("takeoff", "TAKEOFF", altitude_m=10), step("goto", "GOTO", target=nav_goal().spatial_constraint.to_dict()),
                step("home", "GOTO", target=NamedLocationTarget("home").to_dict()), step("land", "LAND", zone="home")])
    compiled = compile_plan(raw)
    proofs = (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),)
    contract = build_remaining_task_contract((nav_goal(),), compiled.planner_output, ("takeoff", "goto"),
                                             proofs, current_step_id="home")
    entry = contract.entry("goal_nav")
    assert entry.status == "INSUFFICIENT"
    assert entry.restart_policy is RestartPolicy.CANNOT_RESUME
    assert entry.reasons == ("INSUFFICIENT_COMPLETION_EVIDENCE",)
    assert not contract.supported
    assert contract.pending_goals == ()
    # Conflicting terminal evidence for one completed step is ambiguous, and
    # ambiguity can never be resolved by picking the more convenient claim.
    with pytest.raises(LocalRepairError) as error:
        assess_remaining_goals((nav_goal(),), compiled.planner_output, ("takeoff", "goto"),
            (SkillExecutionEvidence("goto", "invoke_1", 1, SkillResult(SkillStatus.SUCCEEDED, SkillResultCode.GOAL_REACHED, "", {})),
             SkillExecutionEvidence("goto", "invoke_2", 1, SkillResult(SkillStatus.SUCCEEDED, SkillResultCode.GOAL_REACHED, "", {}))),
            current_step_id="home")
    assert error.value.code == "AMBIGUOUS_EXECUTION_EVIDENCE"


def test_transferability_marks_what_a_replacement_uav_may_own():
    # One target alias per assignment: mixed aliases are ambiguous by design.
    track_raw = plan([step("takeoff", "TAKEOFF", altitude_m=10),
                      step("goto", "GOTO", target=nav_goal().spatial_constraint.to_dict()),
                      step("track", "TRACK", target_ref="$trusted_target.target_id", duration_s=10)])
    search_raw = plan([step("takeoff", "TAKEOFF", altitude_m=10),
                       step("goto", "GOTO", target=nav_goal().spatial_constraint.to_dict()),
                       step("search", "SEARCH", region={"shape": "RECTANGLE", "frame": "WORLD_ENU", "center_xyz_m": [10, 0, 0],
                           "width_m": 6, "height_m": 6}, strategy={"kind": "LAWNMOWER", "spacing_m": 3},
                           entry_policy="START_IN_PLACE_IF_INSIDE", target_description="a person", search_altitude_m=10, timeout_s=30)])
    nav_raw = plan([step("takeoff", "TAKEOFF", altitude_m=10),
                    step("goto", "GOTO", target=nav_goal().spatial_constraint.to_dict())])
    takeoff_proof = (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),)
    track = build_remaining_task_contract((track_goal(10),), track_raw, ("takeoff",), takeoff_proof, current_step_id="goto")
    search = build_remaining_task_contract(
        (MissionGoal("goal_search", GoalType.SEARCH_TARGET, "target_b", None, None, None, ConstraintStrength.MUST),),
        search_raw, ("takeoff",), takeoff_proof, current_step_id="goto")
    navigate = build_remaining_task_contract((nav_goal(),), nav_raw, ("takeoff",), takeoff_proof, current_step_id="goto")
    assert track.entry("goal_track").status == "PENDING"
    assert track.entry("goal_track").transferability is Transferability.SAME_UAV_ONLY
    assert search.entry("goal_search").transferability is Transferability.SAME_UAV_ONLY
    assert navigate.entry("goal_nav").transferability is Transferability.TRANSFERABLE
    shared = build_remaining_task_contract((track_goal(10),), track_raw, ("takeoff",), takeoff_proof,
                                           current_step_id="goto", shared_target_ids=("target_a",))
    assert shared.entry("goal_track").transferability is Transferability.REQUIRES_SHARED_EVIDENCE
    assert shared.entry("goal_track").target_binding == "target_a"
    assert [item.goal_id for item in shared.handoff_entries] == []


# ---------------------------------------------------------------------------
# 5/6. Spatial reference validity: VALID / REPROJECTABLE / INVALID
# ---------------------------------------------------------------------------

def test_unchanged_reference_is_valid_and_small_pose_change_is_reprojectable():
    valid = evaluate_spatial_reference(anchor(), **reference_arguments())
    assert valid.verdict is SpatialReferenceVerdict.VALID
    assert valid.reasons == ()
    reprojectable = evaluate_spatial_reference(anchor(), **reference_arguments(current_pose_xyz_m=(1.5, 0, 10)))
    assert reprojectable.verdict is SpatialReferenceVerdict.REPROJECTABLE
    assert reprojectable.code is None
    assert reprojectable.position_delta_m == pytest.approx(0.5)
    # Pure heading change cannot move a WORLD_ENU route: position unchanged.
    assert evaluate_spatial_reference(anchor(), **reference_arguments()).verdict is SpatialReferenceVerdict.VALID


@pytest.mark.parametrize("overrides,code", [
    ({"map_version": 2}, "REFERENCE_CHANGED"),
    ({"reference_version": 5}, "REFERENCE_CHANGED"),
    ({"time_domain": "monotonic"}, "REFERENCE_CHANGED"),
    ({"observation_time_s": 19.9}, "OBSERVATION_TIME_MISMATCH"),
    ({"pose_time_s": 20.2}, "OBSERVATION_TIME_MISMATCH"),
    ({"observation_time_s": float("nan")}, "OBSERVATION_TIME_MISMATCH"),
    ({"current_pose_xyz_m": (3.5, 0, 10)}, "HOLD_DRIFT"),
    ({"now_wall_s": 160.0}, "ANCHOR_EXPIRED"),
])
def test_version_time_or_drift_changes_invalidate_old_candidates(overrides, code):
    verdict = evaluate_spatial_reference(anchor(), **reference_arguments(**overrides))
    assert verdict.verdict is SpatialReferenceVerdict.INVALID
    assert verdict.code == code
    assert verdict.reasons


def test_new_obstacle_or_cross_uav_route_conflict_invalidates_candidate():
    blocked = evaluate_spatial_reference(anchor(), **reference_arguments(
        route=((1, 0, 10), (10, 0, 10)), segment_blocked=lambda a, b: True))
    assert blocked.verdict is SpatialReferenceVerdict.INVALID
    assert blocked.code == "UNSAFE_ENTRY_OR_ROUTE"
    conflict = evaluate_spatial_reference(anchor(), **reference_arguments(route_conflicts=(("uav_3", "uav_4"),)))
    assert conflict.verdict is SpatialReferenceVerdict.INVALID
    assert conflict.code == "SHARED_SPACE_CONFLICT"
    free = evaluate_spatial_reference(anchor(), **reference_arguments(
        route=((1, 0, 10), (10, 0, 10)), segment_blocked=lambda a, b: False))
    assert free.verdict is SpatialReferenceVerdict.VALID


def test_reprojection_reconnects_position_without_reinterpreting_old_geometry():
    admitted = ((1, 0, 10), (1, 2, 10), (0, 0, 0))
    reprojected = reproject_world_route(admitted, (1.5, 0.3, 10))
    assert reprojected == ((1.5, 0.3, 10), (1, 2, 10), (0, 0, 0))
    # Historical resolved geometry beyond the access point is untouched.
    assert reprojected[1:] == admitted[1:]
    with pytest.raises(LocalRepairError):
        reproject_world_route((), (0, 0, 10))


# ---------------------------------------------------------------------------
# 7/8/9. Unified cross-UAV dependency outcomes
# ---------------------------------------------------------------------------

def cross_subset_dependencies(goal_states):
    a, b = nav_goal("goal_a"), nav_goal("goal_nav")
    spec = FleetTaskSpecV1(source_text="A before B", goals=(a, b),
        ordering_constraints=(OrderingConstraint("order_ab", "goal_a", "goal_nav", ConstraintStrength.MUST),))
    assignments = (SimpleNamespace(assignment_id="assignment_1", goal_ids=("goal_a",), uav_id="uav_1"),
                   SimpleNamespace(assignment_id="assignment_3", goal_ids=("goal_nav",), uav_id="uav_3"))
    return extract_for(spec, ("goal_nav",), assignments, goal_states)


def extract_for(spec, local_goal_ids, assignments, goal_states):
    from fleet.local_repair import extract_external_dependencies
    return extract_external_dependencies(spec, local_goal_ids, assignments=assignments,
        goal_states=goal_states, goal_evidence_refs={"goal_a": ("result_a",)}, versions={"order_ab": 1})


def test_unfinished_external_predecessor_returns_coordination_required_with_affected_set():
    dependencies = cross_subset_dependencies({"goal_a": "PENDING"})
    versions = {"fleet": 1, "assignment_3": 1, "assignment_1": 1}
    check = check_cross_uav_dependencies(dependencies, dependencies, versions, versions)
    assert check.verdict is DependencyVerdict.COORDINATION_REQUIRED
    assert check.affected_uav_ids == ("uav_1", "uav_3")
    assert check.dependency_ids == ("order_ab",)
    assert any(reason.startswith("EXTERNAL_DEPENDENCY_REQUIRES_COORDINATION") for reason in check.reasons)
    satisfied = cross_subset_dependencies({"goal_a": "CONFIRMED"})
    ok = check_cross_uav_dependencies(satisfied, satisfied, versions, versions)
    assert ok.verdict is DependencyVerdict.LOCAL_OK
    assert ok.dependency_ids == ()


def test_shared_channel_contention_cannot_be_committed_twice():
    open_channel = ExternalDependencySnapshot("channel", "SHARED_RESOURCE", ("goal_b", "goal_c"),
        ("uav_3", "uav_4"), 1, "UNCHANGED", ("reservation_open",))
    versions = {"fleet": 1}
    first = check_cross_uav_dependencies((open_channel,), (open_channel,), versions, versions)
    assert first.verdict is DependencyVerdict.LOCAL_OK
    # The first committer's publication flips the shared reservation; the
    # compare-and-commit of the second UAV must observe the change and stop.
    reserved = replace(open_channel, version=2, evidence_refs=("reservation_uav_3",))
    second = check_cross_uav_dependencies((open_channel,), (reserved,), versions, versions,
        shared_resource_ids=("channel",))
    assert second.verdict is DependencyVerdict.COORDINATION_REQUIRED
    assert "DEPENDENCY_SNAPSHOT_CHANGED" in second.reasons
    assert "channel" in second.dependency_ids
    assert second.affected_uav_ids == ("uav_3", "uav_4")
    gone = check_cross_uav_dependencies((open_channel,), (), versions, versions)
    assert gone.verdict is DependencyVerdict.INVALID
    assert gone.dependency_ids == ("channel",)


def test_unrelated_uav_version_change_does_not_discard_current_repair():
    dependencies = cross_subset_dependencies({"goal_a": "CONFIRMED"})
    versions = {"fleet": 1, "assignment_3": 1, "assignment_1": 1}
    # An unrelated UAV's version never enters the dependency-scoped version
    # set, so its progress cannot invalidate this repair.
    check = check_cross_uav_dependencies(dependencies, dependencies, versions, versions)
    assert check.verdict is DependencyVerdict.LOCAL_OK
    # A DEPENDENT UAV's version change must still block the commit.
    blocked = check_cross_uav_dependencies(dependencies, dependencies, versions,
                                           {**versions, "assignment_1": 2})
    assert blocked.verdict is DependencyVerdict.COORDINATION_REQUIRED
    assert "DEPENDENCY_SNAPSHOT_CHANGED" in blocked.reasons


def test_route_conflict_is_a_coordination_outcome_naming_both_uavs():
    check = check_cross_uav_dependencies((), (), {}, {}, route_conflicts=(("uav_4", "uav_3"),))
    assert check.verdict is DependencyVerdict.COORDINATION_REQUIRED
    assert check.affected_uav_ids == ("uav_3", "uav_4")
    assert check.reasons == ("ROUTE_CONFLICT:uav_4:uav_3",)
    with pytest.raises(ValueError):
        check_cross_uav_dependencies((), (), {}, {}, route_conflicts=(("uav_3", "uav_3"),))


# ---------------------------------------------------------------------------
# 10. Qwen cannot rewrite trusted context through the suffix contract
# ---------------------------------------------------------------------------

def test_model_output_cannot_rewrite_contract_anchor_goal_or_other_uav():
    from tests.fleet.test_local_repair import context, draft, search_track_context, validate
    # Anchor or routing rewrite: the envelope is compared against the frozen
    # request before any semantic checking.
    ctx = context()
    with pytest.raises(LocalRepairError) as error:
        validate(ctx, draft(ctx, anchor_id="anchor_forged"))
    assert error.value.code == "ROUTING_MISMATCH"
    # Steps flown by another UAV never enter this UAV's plan.
    other_uav = [step("goto", "GOTO", target=PointTarget(CoordinateFrame.WORLD_ENU, (10, 0, 10)).to_dict())]
    with pytest.raises(LocalRepairError) as error:
        LocalRepairDraftV3.from_dict({**draft(ctx).to_dict(),
            "steps": [{**item, "uav_id": "uav_4"} for item in draft(ctx).to_dict()["steps"]]})
    assert error.value.code == "ROUTING_MISMATCH"
    # The wire schema has no field through which a model could restate the
    # remaining contract; unknown keys are rejected outright.
    with pytest.raises(LocalRepairError) as error:
        LocalRepairDraftV3.from_dict({**draft(ctx).to_dict(), "remaining_task_contract": {"goals": []}})
    assert error.value.code == "INVALID_SUFFIX_FIELDS"
    # Shrinking a trusted remaining duration is a protected-semantics attack.
    track_ctx = search_track_context()
    data = draft(track_ctx).to_dict()["steps"]
    next(item for item in data if item["skill"] == "TRACK")["args"]["duration_s"] = 1
    with pytest.raises(LocalRepairError) as error:
        validate(track_ctx, draft(track_ctx, data))
    assert error.value.code == "PROTECTED_STEP_MUTATION"
    # The context the model actually receives exposes the contract read-only.
    view = track_ctx.to_dict()["remaining_task_contract"]
    assert view["read_only"] is True
    assert view["computed_by"] == "skill_terminal_evidence"
    assert track_ctx.remaining_task_contract.to_dict() == view
