"""Generic goal/skill completion contracts for recovery supervision.

The registry owns every task-completion semantic used by remaining-task
computation and repair admission. SkillManager/executor terminal evidence
remains the only trusted input; Qwen output can never assert completion,
remaining amounts, restart policy or transferability. Unregistered goal
types fail closed: no completion state is guessed.

Adding a task type means adding/reusing evaluators and registering them
here; the recovery controller and remaining-task pipeline stay unchanged:

    Execution Evidence -> Contract Registry -> RemainingTaskContract
    -> Recovery/Repair
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite

from fleet.task_spec import GoalType, MissionGoal, TerminationGoal
from planner.schemas_v3 import PlanStepDraftV3
from planner.spatial import NamedLocationTarget
from skills.types import SkillResultCode, SkillStatus

Goal = MissionGoal | TerminationGoal


class RestartPolicy(str, Enum):
    COMPLETED = "COMPLETED"
    CONTINUE = "CONTINUE"
    RESTART = "RESTART"
    CANNOT_RESUME = "CANNOT_RESUME"


class Transferability(str, Enum):
    SAME_UAV_ONLY = "SAME_UAV_ONLY"
    TRANSFERABLE = "TRANSFERABLE"
    REQUIRES_SHARED_EVIDENCE = "REQUIRES_SHARED_EVIDENCE"


def _goal_basis(goal: Goal) -> str:
    return getattr(goal, "completion_basis", "valid_execution")


def _target_alias(goal: Goal):
    return getattr(goal, "target_alias", None)


@dataclass(frozen=True, slots=True)
class SkillContractEvaluator:
    """Per-Skill terminal-evidence semantics; one row of the supervision table."""

    skill: str
    success_code: SkillResultCode
    requires_target_identity: bool = False
    quantifies_duration: bool = False

    def validate_completion(self, step, proof) -> bool:
        """One terminal Skill result qualifies as step completion evidence."""
        if proof is None or proof.result.status is not SkillStatus.SUCCEEDED:
            return False
        if proof.result.code is not self.success_code:
            return False
        if self.requires_target_identity and not proof.result.data.get("target_id"):
            return False
        return True

    def duration_credit(self, step, proof, invocations, *, completion_basis: str):
        """Optional partial-duration ledger; None means no partial credit."""
        return None


class TrackSkillContractEvaluator(SkillContractEvaluator):
    """TRACK's trusted duration ledger, ported unchanged from local repair.

    Credit comes only from the execution owner's per-invocation ledger, never
    elapsed time. Deterministic REACQUIRE may split one semantic TRACK into
    several invocations; each counts once and continuous completion never
    concatenates intervals across lost-target invocations.
    """

    def duration_credit(self, step, proof, invocations, *, completion_basis: str):
        required = float(step.args.get("duration_s", 0))
        if required <= 0:
            return None
        mode = proof.result.data.get("completion_basis")
        if (mode not in {"valid_execution", "continuous"} or mode != completion_basis
                or step.args.get("completion_basis", "valid_execution") != completion_basis):
            return None
        valid = 0.0
        final_continuous = 0.0
        for item in invocations:
            data = item.result.data
            if (data.get("progress_schema") != "track_progress.v1"
                    or data.get("completion_basis") != mode
                    or data.get("target_id") != proof.result.data.get("target_id")):
                return None
            try:
                elapsed, execution, continuous, invocation_required = (
                    float(data.get(key)) for key in
                    ("elapsed_s", "valid_execution_s", "continuous_execution_s", "required_duration_s")
                )
            except (TypeError, ValueError):
                return None
            if not all(isfinite(value) and value >= 0 for value in
                       (elapsed, execution, continuous, invocation_required)):
                return None
            if (invocation_required <= 0 or execution > elapsed + 1e-9
                    or continuous > execution + 1e-9):
                return None
            valid += min(execution, invocation_required)
            if item.invocation_id == proof.invocation_id:
                final_continuous = continuous
        credited = final_continuous if mode == "continuous" else valid
        # A success code alone cannot turn an incomplete Skill into goal credit.
        return required if credited + 1e-9 >= required else None


DEFAULT_SKILL_CONTRACTS: Mapping[str, SkillContractEvaluator] = {
    "TAKEOFF": SkillContractEvaluator("TAKEOFF", SkillResultCode.TAKEOFF_COMPLETE),
    "GOTO": SkillContractEvaluator("GOTO", SkillResultCode.GOAL_REACHED),
    "FOLLOW_ROUTE": SkillContractEvaluator("FOLLOW_ROUTE", SkillResultCode.ROUTE_COMPLETE),
    "HOVER": SkillContractEvaluator("HOVER", SkillResultCode.HOVER_COMPLETE),
    "SEARCH": SkillContractEvaluator("SEARCH", SkillResultCode.TARGET_FOUND,
                                     requires_target_identity=True),
    "TRACK": TrackSkillContractEvaluator("TRACK", SkillResultCode.TRACK_COMPLETE,
                                         requires_target_identity=True, quantifies_duration=True),
    "INSPECT": SkillContractEvaluator("INSPECT", SkillResultCode.INSPECT_COMPLETE,
                                      requires_target_identity=True),
    "LAND": SkillContractEvaluator("LAND", SkillResultCode.LAND_COMPLETE),
}


class GoalContractEvaluator:
    """Base class: completion semantics for one GoalType.

    Default behaviour covers binary skills with no duration ledger; richer
    task types override only what differs. Nothing here reads a model, agent
    or simulator: every input is trusted terminal evidence.
    """

    goal_type: GoalType
    duration_quantified: bool = False
    # Multiple goals of this type in one assignment cannot be told apart by
    # step matching alone, so they fail closed instead of guessing.
    ambiguous_with_sibling: bool = False
    pending_transferability: Transferability = Transferability.SAME_UAV_ONLY
    shared_evidence_transfer: bool = False
    is_delegation_anchor: bool = False

    def matching_steps(self, goal: Goal, steps: Sequence[PlanStepDraftV3], home_name: str):
        return ()

    def evaluate_evidence(self, goal, matches, completed, by_step, grouped, skills):
        """Classify matched completed steps: (done, per-step credits, unknown)."""
        done, credits, unknown = [], {}, False
        for step in matches:
            if step.id not in completed:
                continue
            contract = skills.get(step.skill)
            proof = by_step.get(step.id)
            if contract is None or not contract.validate_completion(step, proof):
                unknown = True
                continue
            if contract.quantifies_duration:
                credit = contract.duration_credit(step, proof, grouped[step.id],
                                                  completion_basis=_goal_basis(goal))
                if credit is None:
                    unknown = True
                    continue
                credits[step.id] = credit
            done.append(step)
        return done, credits, unknown

    def compute_progress(self, goal, done, credits, skills):
        """Per-done-step progress amounts (seconds) for quantified goals."""
        amounts = []
        for step in done:
            if step.id in credits:
                amounts.append(credits[step.id])
            else:
                skill = skills.get(step.skill)
                amounts.append(float(step.args.get("duration_s", 0))
                               if skill is not None and skill.quantifies_duration else 0.0)
        return amounts

    def compute_remaining(self, goal, done, credits, skills):
        """Map step evidence to a (verdict, remaining goal) pair."""
        if done:
            return "CONFIRMED", goal
        return "PENDING", goal

    def restart_policy(self, status: str, confirmed_amount):
        if status == "CONFIRMED":
            return RestartPolicy.COMPLETED
        if status == "INSUFFICIENT":
            return RestartPolicy.CANNOT_RESUME
        if confirmed_amount is not None and confirmed_amount > 1e-9:
            return RestartPolicy.CONTINUE
        return RestartPolicy.RESTART

    def transferability(self, goal, status: str, shared_target_ids) -> Transferability:
        if status == "PENDING":
            if self.pending_transferability is Transferability.TRANSFERABLE:
                return Transferability.TRANSFERABLE
            if self.shared_evidence_transfer and _target_alias(goal) in frozenset(shared_target_ids):
                return Transferability.REQUIRES_SHARED_EVIDENCE
        return Transferability.SAME_UAV_ONLY

    def required_resources(self, goal):
        return ()


class DurationGoalContractEvaluator(GoalContractEvaluator):
    """Goals whose obligation is a trusted accumulated duration."""

    duration_quantified = True
    ambiguous_with_sibling = True

    def compute_remaining(self, goal, done, credits, skills):
        requested = goal.duration_s
        if requested is None:
            return "INSUFFICIENT", None
        amounts = self.compute_progress(goal, done, credits, skills)
        continuous = _goal_basis(goal) == "continuous"
        total = max(amounts, default=0.0) if continuous else sum(amounts)
        if total >= requested:
            return "CONFIRMED", goal
        # Continuous completion cannot bank partial intervals: the remaining
        # obligation keeps the full original duration after any loss.
        remaining = requested if continuous else requested - total
        from dataclasses import replace
        return "PENDING", replace(goal, duration_s=remaining)


class NavigateGoalContractEvaluator(GoalContractEvaluator):
    goal_type = GoalType.NAVIGATE
    pending_transferability = Transferability.TRANSFERABLE
    is_delegation_anchor = True

    def matching_steps(self, goal, steps, home_name):
        expected = goal.spatial_constraint.to_dict()
        return tuple(step for step in steps if step.skill == "GOTO"
                     and step.to_dict()["args"].get("target") == expected)

    def required_resources(self, goal):
        constraint = goal.spatial_constraint
        return () if constraint is None else ("destination:" + goal.goal_id,)


class SearchGoalContractEvaluator(GoalContractEvaluator):
    goal_type = GoalType.SEARCH_TARGET

    def matching_steps(self, goal, steps, home_name):
        expected = None if goal.spatial_constraint is None else goal.spatial_constraint.to_dict()
        return tuple(step for step in steps if step.skill == "SEARCH" and
                     (expected is None or step.to_dict()["args"].get("region") == expected))

    def required_resources(self, goal):
        constraint = goal.spatial_constraint
        return () if constraint is None else ("search_region:" + goal.goal_id,)


class TrackGoalContractEvaluator(DurationGoalContractEvaluator):
    goal_type = GoalType.TRACK_TARGET
    shared_evidence_transfer = True

    def matching_steps(self, goal, steps, home_name):
        return tuple(step for step in steps if step.skill == "TRACK")

    def required_resources(self, goal):
        alias = _target_alias(goal)
        return () if alias is None else ("target:" + alias,)


class InspectGoalContractEvaluator(GoalContractEvaluator):
    """Example extension: target inspection supervised only by registration.

    INSPECT steps reference a trusted CandidateBank candidate; completion
    requires the terminal INSPECT success code plus the target identity in
    the execution evidence. No recovery-framework change is needed.
    """

    goal_type = GoalType.INSPECT_TARGET
    shared_evidence_transfer = True

    def matching_steps(self, goal, steps, home_name):
        return tuple(step for step in steps if step.skill == "INSPECT")

    def required_resources(self, goal):
        alias = _target_alias(goal)
        return () if alias is None else ("target:" + alias,)


class WaitGoalContractEvaluator(DurationGoalContractEvaluator):
    goal_type = GoalType.WAIT
    pending_transferability = Transferability.TRANSFERABLE

    def matching_steps(self, goal, steps, home_name):
        return tuple(step for step in steps if step.skill == "HOVER")


class ReturnHomeGoalContractEvaluator(GoalContractEvaluator):
    goal_type = GoalType.RETURN_HOME
    pending_transferability = Transferability.TRANSFERABLE

    def matching_steps(self, goal, steps, home_name):
        return tuple(step for step in steps if step.skill == "GOTO"
                     and isinstance(step.spatial_target, NamedLocationTarget)
                     and step.spatial_target.name == home_name)


class LandGoalContractEvaluator(GoalContractEvaluator):
    goal_type = GoalType.LAND
    pending_transferability = Transferability.TRANSFERABLE

    def matching_steps(self, goal, steps, home_name):
        return tuple(step for step in steps if step.skill == "LAND")


class ReturnHomeAndLandGoalContractEvaluator(GoalContractEvaluator):
    goal_type = GoalType.RETURN_HOME_AND_LAND
    pending_transferability = Transferability.TRANSFERABLE

    def matching_steps(self, goal, steps, home_name):
        returns = tuple(step for step in steps if step.skill == "GOTO"
                        and isinstance(step.spatial_target, NamedLocationTarget)
                        and step.spatial_target.name == home_name)
        if not returns:
            return ()
        last_index = steps.index(returns[-1])
        return returns + tuple(step for step in steps[last_index + 1:] if step.skill == "LAND")

    def compute_remaining(self, goal, done, credits, skills):
        if done and any(step.skill == "LAND" for step in done) and any(step.skill == "GOTO" for step in done):
            return "CONFIRMED", goal
        if done:
            # A completed return must not be reissued as a new user action.
            return "INSUFFICIENT", None
        return "PENDING", goal


class GoalContractRegistry:
    """The single table recovery consults for completion semantics."""

    def __init__(self, goal_evaluators: Mapping[GoalType, GoalContractEvaluator], *,
                 skill_contracts: Mapping[str, SkillContractEvaluator] | None = None):
        evaluators = dict(goal_evaluators)
        for goal_type, evaluator in evaluators.items():
            if not isinstance(goal_type, GoalType):
                raise TypeError("goal evaluator keys must be GoalType values")
            if not isinstance(evaluator, GoalContractEvaluator) or evaluator.goal_type is not goal_type:
                raise ValueError("goal evaluator does not match its registered GoalType")
        self._goal_evaluators = evaluators
        self._skill_contracts: dict[str, SkillContractEvaluator] = (
            dict(DEFAULT_SKILL_CONTRACTS) if skill_contracts is None else dict(skill_contracts))

    def evaluator_for(self, goal_type) -> GoalContractEvaluator | None:
        return self._goal_evaluators.get(goal_type)

    def is_registered(self, goal_type) -> bool:
        return goal_type in self._goal_evaluators

    def skill_contract(self, skill_name) -> SkillContractEvaluator | None:
        return self._skill_contracts.get(skill_name)

    @property
    def skill_contracts(self):
        return dict(self._skill_contracts)

    @property
    def delegation_anchor_types(self):
        return tuple(sorted((etype for etype, evaluator in self._goal_evaluators.items()
                             if evaluator.is_delegation_anchor), key=lambda item: item.value))

    def is_delegation_anchor(self, goal_type) -> bool:
        evaluator = self._goal_evaluators.get(goal_type)
        return evaluator is not None and evaluator.is_delegation_anchor

    def matching_steps(self, goal, steps, home_name):
        evaluator = self._goal_evaluators.get(goal.goal_type)
        return () if evaluator is None else evaluator.matching_steps(goal, steps, home_name)

    def with_goal_evaluator(self, evaluator: GoalContractEvaluator) -> "GoalContractRegistry":
        """Extend by registration; used by tests and new task types."""
        return GoalContractRegistry({**self._goal_evaluators, evaluator.goal_type: evaluator},
                                    skill_contracts=self._skill_contracts)

    def without_goal_evaluator(self, goal_type) -> "GoalContractRegistry":
        return GoalContractRegistry({etype: evaluator for etype, evaluator in self._goal_evaluators.items()
                                     if etype is not goal_type}, skill_contracts=self._skill_contracts)


def build_default_goal_contract_registry() -> GoalContractRegistry:
    """Every task type with a trusted completion semantic, or fail-closed."""
    evaluators = (NavigateGoalContractEvaluator(), SearchGoalContractEvaluator(),
                  TrackGoalContractEvaluator(), InspectGoalContractEvaluator(),
                  WaitGoalContractEvaluator(), ReturnHomeGoalContractEvaluator(),
                  LandGoalContractEvaluator(), ReturnHomeAndLandGoalContractEvaluator())
    # REPORT stays deliberately unregistered: its completion semantics are not
    # trusted yet, so it must fail closed rather than be guessed.
    return GoalContractRegistry({evaluator.goal_type: evaluator for evaluator in evaluators})


DEFAULT_GOAL_CONTRACT_REGISTRY = build_default_goal_contract_registry()
