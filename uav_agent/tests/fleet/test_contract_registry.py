"""Contract Registry: registration-driven completion semantics, fail-closed."""
import inspect

import pytest

from fleet.contract_registry import (
    DEFAULT_GOAL_CONTRACT_REGISTRY, GoalContractRegistry, GoalContractEvaluator,
    InspectGoalContractEvaluator, Transferability, build_default_goal_contract_registry,
)
from fleet.local_repair import (
    SkillExecutionEvidence, build_remaining_task_contract,
)
from fleet.recovery_controller import FleetRecoveryController
from fleet.task_spec import ConstraintStrength, GoalType, MissionGoal, TerminationGoal
from planner.schemas_v3 import PlanStepDraftV3, SkillPlanDraftV3
from skills.types import SkillResult, SkillResultCode, SkillStatus


def step(id, skill, **args):
    return {"id": id, "uav_id": "uav_3", "skill": skill, "args": args}


def plan(steps):
    return SkillPlanDraftV3.from_dict({"schema_version": 3, "mission_id": "mission_3", "uav_id": "uav_3",
        "plan_version": 1, "assumptions": [], "steps": steps})


def evidence(id, code, **data):
    return SkillExecutionEvidence(id, "invoke_" + id, 1, SkillResult(SkillStatus.SUCCEEDED, code, "trusted", data))


INSPECT_STEPS = [
    step("takeoff", "TAKEOFF", altitude_m=10),
    step("goto", "GOTO", target={"kind": "POINT", "frame": "WORLD_ENU", "xyz_m": [10, 0, 10]}),
    step("inspect", "INSPECT", candidate_id="candidate_1", desired_observation_distance_m=4.0,
         viewpoint_change_deg=30.0, max_duration_s=20.0, approach_policy="MAINTAIN_ALTITUDE_ORBIT"),
    step("home", "GOTO", target={"kind": "NAMED_LOCATION", "name": "home"}),
    step("land", "LAND", zone="home"),
]


def inspect_plan():
    """A V3 plan whose INSPECT step arrived via a trusted runtime revision.

    Initial plans never contain INSPECT (from_dict rejects it); constructing
    the dataclass directly models the post-revision view the registry sees.
    """
    steps = tuple(PlanStepDraftV3.from_dict(item) for item in INSPECT_STEPS)
    return SkillPlanDraftV3(3, "mission_3", "uav_3", 1, (), steps, None)


def inspect_goal():
    return MissionGoal("goal_inspect", GoalType.INSPECT_TARGET, "target_a", None, None, None,
                       ConstraintStrength.MUST)


def test_inspect_evaluator_is_registered_and_supervises_without_framework_changes():
    # INSPECT supervision exists purely as a registry entry; the recovery
    # controller itself carries no goal-type branching to extend.
    assert DEFAULT_GOAL_CONTRACT_REGISTRY.is_registered(GoalType.INSPECT_TARGET)
    assert "GoalType" not in inspect.getsource(FleetRecoveryController)
    assert not any("goal_type ==" in line for line in inspect.getsource(FleetRecoveryController).splitlines())
    raw = inspect_plan()
    contract = build_remaining_task_contract((inspect_goal(),), raw, ("takeoff", "goto"),
        (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),), current_step_id="inspect")
    entry = contract.entry("goal_inspect")
    assert entry.status == "PENDING"
    assert entry.transferability is Transferability.SAME_UAV_ONLY
    assert entry.required_resources == ("target:target_a",)
    assert entry.remaining_goal is not None
    # Confirmed INSPECT needs the terminal success code AND target identity.
    done = build_remaining_task_contract((inspect_goal(),), raw,
        ("takeoff", "goto", "inspect"),
        (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),
         evidence("inspect", SkillResultCode.INSPECT_COMPLETE, target_id="target_1")),
        current_step_id="home")
    assert done.entry("goal_inspect").status == "CONFIRMED"
    assert done.entry("goal_inspect").evidence_refs == ("invoke_inspect",)
    identity_missing = build_remaining_task_contract((inspect_goal(),), raw,
        ("takeoff", "goto", "inspect"),
        (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),
         evidence("inspect", SkillResultCode.INSPECT_COMPLETE)),
        current_step_id="home")
    assert identity_missing.entry("goal_inspect").status == "INSUFFICIENT"


def test_unregistered_goal_type_fails_closed():
    report = TerminationGoal("goal_report", GoalType.REPORT, None, None, ConstraintStrength.MUST)
    raw = inspect_plan()
    contract = build_remaining_task_contract((report,), raw, ("takeoff",),
        (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),), current_step_id="goto")
    entry = contract.entry("goal_report")
    assert entry.status == "INSUFFICIENT"
    assert entry.restart_policy.value == "CANNOT_RESUME"
    assert "UNREGISTERED_GOAL_CONTRACT:REPORT" in contract.assessment.reasons
    assert not contract.supported


def test_removing_an_evalator_fail_closes_only_that_type():
    registry = DEFAULT_GOAL_CONTRACT_REGISTRY.without_goal_evaluator(GoalType.TRACK_TARGET)
    assert not registry.is_registered(GoalType.TRACK_TARGET)
    track = MissionGoal("goal_track", GoalType.TRACK_TARGET, "target_a", None, 10, None, ConstraintStrength.MUST)
    raw = plan([step("takeoff", "TAKEOFF", altitude_m=10),
                step("track", "TRACK", target_ref="$trusted_target.target_id", duration_s=10)])
    contract = build_remaining_task_contract((track,), raw, ("takeoff",),
        (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),),
        current_step_id="track", current_step_started=False, registry=registry)
    assert contract.entry("goal_track").status == "INSUFFICIENT"
    # The default registry still supervises the same goal.
    default = build_remaining_task_contract((track,), raw, ("takeoff",),
        (evidence("takeoff", SkillResultCode.TAKEOFF_COMPLETE),),
        current_step_id="track", current_step_started=False)
    assert default.entry("goal_track").status == "PENDING"


def test_registry_extension_only_requires_registration():
    class FakedEvaluator(GoalContractEvaluator):
        goal_type = GoalType.WAIT

    registry = DEFAULT_GOAL_CONTRACT_REGISTRY.with_goal_evaluator(FakedEvaluator())
    assert isinstance(registry.evaluator_for(GoalType.WAIT), FakedEvaluator)
    # Mismatched registration is a configuration error, not silent supervision.
    with pytest.raises(ValueError):
        GoalContractRegistry({GoalType.WAIT: InspectGoalContractEvaluator()})


def test_track_ledger_semantics_live_in_the_registry():
    track = DEFAULT_GOAL_CONTRACT_REGISTRY.skill_contract("TRACK")
    assert track.quantifies_duration and track.requires_target_identity
    from fleet.contract_registry import DEFAULT_SKILL_CONTRACTS
    assert DEFAULT_SKILL_CONTRACTS["SEARCH"].success_code is SkillResultCode.TARGET_FOUND
    assert DEFAULT_SKILL_CONTRACTS["GOTO"].success_code is SkillResultCode.GOAL_REACHED
    assert DEFAULT_SKILL_CONTRACTS["INSPECT"].success_code is SkillResultCode.INSPECT_COMPLETE
    assert build_default_goal_contract_registry().delegation_anchor_types == (GoalType.NAVIGATE,)
