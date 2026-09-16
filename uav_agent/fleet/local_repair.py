"""Pure, fail-closed contracts for owner-authorized Spatial V3 suffix repair.

This module never reads an Agent, controller, simulator, or model. Evidence and
dependency snapshots must be supplied by their runtime owners. A compiled
candidate still requires current-world entry/collision checks and an owner-side
compare-and-commit; compilation is not permission to execute.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from fleet.task_spec import ConstraintStrength, FleetTaskSpecV1, GoalType, MissionGoal, OrderingConstraint, TerminationGoal
from planner.goal_checker import GoalSatisfactionChecker
from planner.schemas import CompiledMission, PlannerWorldContext
from planner.schemas_v3 import PlanStepDraftV3, SkillPlanDraftV3
from planner.spatial import CoordinateFrame, NamedLocationTarget, PointTarget
from planner.spatial_resolver import FramePose, SpatialResolver
from skills.plan import TaskPlan, TaskStep
from skills.types import SkillName, SkillResult, SkillResultCode, SkillStatus

Goal = MissionGoal | TerminationGoal


class LocalRepairError(ValueError):
    def __init__(self, code: str, message: str, *, affected_uav_ids=(), goal_ids=()):
        self.code = code
        self.affected_uav_ids = tuple(sorted(set(affected_uav_ids)))
        self.goal_ids = tuple(sorted(set(goal_ids)))
        super().__init__(message)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _version(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("snapshot mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("snapshot contains a nonfinite value")
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _runtime_copy(value):
    if isinstance(value, Mapping):
        return {key: _runtime_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_runtime_copy(item) for item in value)
    return value


def _json(value) -> str:
    return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return sha256(_json(value).encode()).hexdigest()


def _frozen_step(step: TaskStep) -> TaskStep:
    result = TaskStep(step.step_id, step.skill, _runtime_copy(step.params), step.recovery)
    object.__setattr__(result, "params", _freeze(result.params))
    return result


def _frozen_compiled(original: CompiledMission) -> CompiledMission:
    if not isinstance(original, CompiledMission) or not isinstance(original.planner_output, SkillPlanDraftV3):
        raise LocalRepairError("UNSUPPORTED_CONTRACT", "local repair requires a compiled linear Spatial V3 plan")
    semantic = SkillPlanDraftV3.from_dict(original.planner_output.to_dict())
    task = original.task_plan
    snapshot = TaskPlan(tuple(_frozen_step(step) for step in task.steps), task.mission_id, task.uav_id, task.plan_version)
    return CompiledMission(semantic, snapshot, original.source, original.compiler_notes)


@dataclass(frozen=True, slots=True)
class RepairAnchor:
    anchor_id: str
    frame_id: str
    observation_time_s: float
    pose_time_s: float
    time_domain: str
    pose: FramePose
    home_pose: FramePose
    start_pose: FramePose
    map_version: int
    reference_version: int
    named_locations: Mapping[str, PointTarget | Sequence[float]] = field(default_factory=dict)
    max_pose_time_error_s: float = 0.05

    def __post_init__(self):
        validate_routing_id(self.anchor_id, "anchor_id")
        validate_routing_id(self.frame_id, "frame_id")
        if self.time_domain not in {"simulation", "observation", "monotonic"}:
            raise ValueError("anchor must name its observation time domain")
        for name in ("observation_time_s", "pose_time_s", "max_pose_time_error_s"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if abs(self.observation_time_s - self.pose_time_s) > self.max_pose_time_error_s:
            raise LocalRepairError("OBSERVATION_POSE_MISMATCH", "image/frame and pose timestamps are not aligned")
        for name in ("pose", "home_pose", "start_pose"):
            if not isinstance(getattr(self, name), FramePose):
                raise TypeError(f"{name} must be a FramePose")
        _version(self.map_version, "map_version")
        _version(self.reference_version, "reference_version")
        resolver = SpatialResolver(home_pose=self.home_pose, uav_start_pose=self.start_pose, named_locations=self.named_locations)
        object.__setattr__(self, "named_locations", resolver.named_locations)

    @property
    def resolver(self) -> SpatialResolver:
        # A fresh resolver cannot silently re-anchor an in-flight request.
        return SpatialResolver(home_pose=self.home_pose, uav_start_pose=self.start_pose,
                               uav_hold_pose=self.pose, named_locations=self.named_locations)

    def to_dict(self):
        return {"anchor_id": self.anchor_id, "frame_id": self.frame_id,
                "observation_time_s": self.observation_time_s, "pose_time_s": self.pose_time_s,
                "time_domain": self.time_domain, "frame": "UAV_HOLD_FLU",
                "position_world_m": list(self.pose.xyz_m), "yaw_rad": self.pose.yaw_rad,
                "map_version": self.map_version, "reference_version": self.reference_version,
                "home_pose": {"xyz_m": list(self.home_pose.xyz_m), "yaw_rad": self.home_pose.yaw_rad},
                "start_pose": {"xyz_m": list(self.start_pose.xyz_m), "yaw_rad": self.start_pose.yaw_rad},
                "named_locations": {key: value.to_dict() for key, value in self.named_locations.items()},
                "max_pose_time_error_s": self.max_pose_time_error_s}


@dataclass(frozen=True, slots=True)
class SkillExecutionEvidence:
    """A copied terminal result supplied by the Skill execution owner only."""
    step_id: str
    invocation_id: str
    plan_version: int
    result: SkillResult

    def __post_init__(self):
        validate_routing_id(self.step_id, "step_id")
        validate_routing_id(self.invocation_id, "invocation_id")
        _version(self.plan_version, "plan_version")
        if not isinstance(self.result, SkillResult):
            raise TypeError("execution evidence requires a SkillResult")
        copied = SkillResult(self.result.status, self.result.code, self.result.message, _thaw(self.result.data))
        # SkillResult itself is frozen but its default data mapping is not.
        object.__setattr__(copied, "data", _freeze(copied.data))
        _json(copied.data)
        object.__setattr__(self, "result", copied)

    def to_dict(self):
        return {"step_id": self.step_id, "invocation_id": self.invocation_id,
                "plan_version": self.plan_version, "status": self.result.status.name,
                "code": self.result.code.name, "data": _thaw(self.result.data)}


@dataclass(frozen=True, slots=True)
class RemainingGoalAssessment:
    confirmed_goal_ids: tuple[str, ...]
    pending_goals: tuple[Goal, ...]
    insufficient_goal_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def pending_goal_ids(self):
        return tuple(goal.goal_id for goal in self.pending_goals)

    @property
    def supported(self):
        return not self.insufficient_goal_ids and not self.reasons


_SUCCESS = {"TAKEOFF": SkillResultCode.TAKEOFF_COMPLETE, "GOTO": SkillResultCode.GOAL_REACHED,
            "FOLLOW_ROUTE": SkillResultCode.ROUTE_COMPLETE, "HOVER": SkillResultCode.HOVER_COMPLETE,
            "SEARCH": SkillResultCode.TARGET_FOUND, "TRACK": SkillResultCode.TRACK_COMPLETE,
            "LAND": SkillResultCode.LAND_COMPLETE}


def _track_completion_credit(step, proof, invocations, *, completion_basis):
    """Credit only the execution owner's duration ledger, never elapsed time.

    Deterministic REACQUIRE can split one semantic TRACK into several Skill
    invocations. Its terminal invocation may therefore request only the
    remainder. Each invocation is counted once; continuous completion cannot
    concatenate intervals across lost-target invocations.
    """
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
                _number(data.get(key), key) for key in
                ("elapsed_s", "valid_execution_s", "continuous_execution_s", "required_duration_s")
            )
        except (ValueError, TypeError):
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


def _goal_steps(goal: Goal, steps: Sequence[PlanStepDraftV3], home_name: str):
    kind = goal.goal_type
    if kind is GoalType.NAVIGATE:
        expected = goal.spatial_constraint.to_dict()
        return tuple(step for step in steps if step.skill == "GOTO" and step.to_dict()["args"].get("target") == expected)
    if kind is GoalType.SEARCH_TARGET:
        expected = None if goal.spatial_constraint is None else goal.spatial_constraint.to_dict()
        return tuple(step for step in steps if step.skill == "SEARCH" and
                     (expected is None or step.to_dict()["args"].get("region") == expected))
    if kind in {GoalType.TRACK_TARGET, GoalType.WAIT, GoalType.LAND}:
        skill = {GoalType.TRACK_TARGET: "TRACK", GoalType.WAIT: "HOVER", GoalType.LAND: "LAND"}[kind]
        return tuple(step for step in steps if step.skill == skill)
    if kind in {GoalType.RETURN_HOME, GoalType.RETURN_HOME_AND_LAND}:
        returns = tuple(step for step in steps if step.skill == "GOTO" and isinstance(step.spatial_target, NamedLocationTarget)
                        and step.spatial_target.name == home_name)
        if kind is GoalType.RETURN_HOME:
            return returns
        if returns:
            last_index = steps.index(returns[-1])
            return returns + tuple(step for step in steps[last_index + 1:] if step.skill == "LAND")
    return ()


def assess_remaining_goals(goals: Sequence[Goal], original: SkillPlanDraftV3,
                           completed_step_ids: Sequence[str], evidence: Sequence[SkillExecutionEvidence],
                           *, current_step_id: str | None, current_step_started: bool = True,
                           home_name: str = "home") -> RemainingGoalAssessment:
    """Subtract only terminal Skill evidence; elapsed/feedback time is not credit."""
    goals, evidence = tuple(goals), tuple(evidence)
    completed = set(completed_step_ids)
    by_step, grouped, invocations = {}, {}, {}
    for item in evidence:
        prior = invocations.get(item.invocation_id)
        if prior is not None:
            if prior.to_dict() != item.to_dict():
                raise LocalRepairError("AMBIGUOUS_EXECUTION_EVIDENCE", "one invocation has conflicting terminal evidence")
            continue
        invocations[item.invocation_id] = item
        if item.plan_version > original.plan_version:
            raise LocalRepairError("EVIDENCE_VERSION_MISMATCH", "execution evidence is newer than base plan")
        grouped.setdefault(item.step_id, []).append(item)
    for step_id, items in grouped.items():
        successes = [item for item in items if item.result.status is SkillStatus.SUCCEEDED]
        if step_id in completed and (len(successes) > 1 or (successes and any(item.plan_version > successes[0].plan_version for item in items))):
            raise LocalRepairError("AMBIGUOUS_EXECUTION_EVIDENCE", "completed step has repeated or superseded success evidence")
        by_step[step_id] = successes[0] if successes else items[-1]
    current = next((step for step in original.steps if step.id == current_step_id), None)
    confirmed, pending, insufficient, reasons = [], [], [], []
    terminal_complete = current_step_id is None and set(step.id for step in original.steps).issubset(completed)
    if current is None and not terminal_complete:
        return RemainingGoalAssessment((), (), tuple(goal.goal_id for goal in goals), ("CURRENT_STEP_NOT_IN_V3",))
    if current is not None and current.skill not in {"GOTO", "SEARCH", "TRACK", "HOVER"}:
        reasons.append("CURRENT_SKILL_NOT_RESTARTABLE")
    if current is not None and current_step_started and current.skill in {"TRACK", "HOVER"}:
        reasons.append("PARTIAL_DURATION_EVIDENCE_INSUFFICIENT")
    target_aliases = {getattr(goal, "target_alias", None) for goal in goals} - {None}
    for goal in goals:
        matches = _goal_steps(goal, original.steps, home_name)
        ambiguous = (goal.goal_type in {GoalType.TRACK_TARGET, GoalType.WAIT} and
                     sum(other.goal_type is goal.goal_type for other in goals) > 1)
        if not matches or ambiguous or (len(target_aliases) > 1 and getattr(goal, "target_alias", None)):
            insufficient.append(goal.goal_id)
            continue
        if current is not None and current_step_started and current.skill in {"TRACK", "HOVER"} and current in matches:
            insufficient.append(goal.goal_id)
            continue
        done = []
        duration_credit = {}
        unknown = False
        for step in matches:
            if step.id not in completed:
                continue
            proof = by_step.get(step.id)
            if proof is None or proof.result.status is not SkillStatus.SUCCEEDED or proof.result.code is not _SUCCESS.get(step.skill):
                unknown = True
            elif step.skill in {"SEARCH", "TRACK"} and not proof.result.data.get("target_id"):
                unknown = True
            elif step.skill == "TRACK":
                credit = _track_completion_credit(step, proof, grouped[step.id],
                    completion_basis=goal.completion_basis)
                if credit is None:
                    unknown = True
                else:
                    duration_credit[step.id] = credit
                    done.append(step)
            else:
                done.append(step)
        if unknown:
            insufficient.append(goal.goal_id)
            continue
        if goal.goal_type in {GoalType.TRACK_TARGET, GoalType.WAIT}:
            requested = goal.duration_s
            if requested is None:
                insufficient.append(goal.goal_id)
                continue
            credits = [duration_credit.get(step.id, float(step.args.get("duration_s", 0))) for step in done]
            continuous = getattr(goal, "completion_basis", None) == "continuous"
            credit = max(credits, default=0.0) if continuous else sum(credits)
            if credit >= requested:
                confirmed.append(goal.goal_id)
            else:
                pending.append(replace(goal, duration_s=requested if continuous else requested - credit))
        elif goal.goal_type is GoalType.RETURN_HOME_AND_LAND:
            if done and any(step.skill == "LAND" for step in done) and any(step.skill == "GOTO" for step in done):
                confirmed.append(goal.goal_id)
            elif done:
                # A completed return must not be reissued as a new user action.
                insufficient.append(goal.goal_id)
            else:
                pending.append(goal)
        elif done:
            confirmed.append(goal.goal_id)
        else:
            pending.append(goal)
    return RemainingGoalAssessment(tuple(confirmed), tuple(pending), tuple(insufficient), tuple(reasons))


@dataclass(frozen=True, slots=True)
class ExternalDependencySnapshot:
    dependency_id: str
    kind: str
    goal_ids: tuple[str, ...]
    uav_ids: tuple[str, ...]
    version: int
    state: str
    evidence_refs: tuple[str, ...] = ()
    before_goal_id: str | None = None
    after_goal_id: str | None = None
    scope: str = "EXTERNAL"
    strength: str = "MUST"
    source_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        validate_routing_id(self.dependency_id, "dependency_id")
        if self.kind not in {"PREDECESSOR", "SUCCESSOR", "ASSIGNMENT", "SHARED_RESOURCE"}:
            raise ValueError("unknown external dependency kind")
        _version(self.version, "dependency.version")
        for name in ("goal_ids", "uav_ids", "evidence_refs"):
            values = tuple(getattr(self, name))
            for value in values:
                validate_routing_id(value, name)
            object.__setattr__(self, name, values)
        if self.state not in {"SATISFIED", "UNCHANGED", "BLOCKED", "UNKNOWN"}:
            raise ValueError("unknown dependency state")
        if self.scope not in {"LOCAL", "EXTERNAL"}:
            raise ValueError("unknown dependency scope")
        object.__setattr__(self, "strength", ConstraintStrength(self.strength).value)
        source_refs = tuple(self.source_evidence_refs)
        if any(not isinstance(ref, str) or not ref or len(ref) > 64 for ref in source_refs):
            raise ValueError("source evidence references must be nonempty strings of at most 64 characters")
        object.__setattr__(self, "source_evidence_refs", source_refs)

    def to_dict(self):
        return {name: _thaw(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    allowed: bool
    reasons: tuple[str, ...]
    affected_uav_ids: tuple[str, ...]
    goal_ids: tuple[str, ...]
    digest: str


def dependency_digest(dependencies, versions):
    return _digest({"dependencies": sorted((item.to_dict() for item in dependencies), key=lambda item: item["dependency_id"]),
                    "versions": dict(versions)})


def check_external_dependencies(expected_dependencies, current_dependencies, expected_versions, current_versions) -> DependencyCheck:
    expected, current = tuple(expected_dependencies), tuple(current_dependencies)
    digest = dependency_digest(current, current_versions)
    reasons = []
    if len({item.dependency_id for item in current}) != len(current):
        reasons.append("DUPLICATE_DEPENDENCY")
    if digest != dependency_digest(expected, expected_versions):
        reasons.append("DEPENDENCY_SNAPSHOT_CHANGED")
    for item in current:
        if item.state not in {"SATISFIED", "UNCHANGED"} or not item.evidence_refs:
            reasons.append("EXTERNAL_DEPENDENCY_REQUIRES_COORDINATION:" + item.dependency_id)
    return DependencyCheck(not reasons, tuple(reasons), tuple(sorted({uav for item in (*expected, *current) for uav in item.uav_ids})),
                           tuple(sorted({goal for item in (*expected, *current) for goal in item.goal_ids})), digest)


def extract_external_dependencies(task_spec: FleetTaskSpecV1, local_goal_ids: Sequence[str], *,
                                  assignments: Sequence[object], goal_states: Mapping[str, str],
                                  goal_evidence_refs: Mapping[str, Sequence[str]], versions: Mapping[str, int],
                                  shared_dependencies: Sequence[ExternalDependencySnapshot] = ()):
    """Preserve every crossing edge; UNKNOWN evidence mandates coordination.

    goal_states are owner-produced CONFIRMED/PENDING/ACTIVE/UNKNOWN values.
    Assignment bindings come from the runtime owner, not source-text citations.
    Local bindings are still versioned admission conditions, but are explicitly
    labelled LOCAL. PREFER/OPEN do not turn an already accepted owner mapping
    into a hard assignment restriction. A MUST mismatch remains blocked.
    """
    local = set(local_goal_ids)
    assignments_by_goal = {}
    for assignment in assignments:
        for goal in assignment.goal_ids:
            assignments_by_goal.setdefault(goal, []).append(assignment)
    owner = {goal: items[0].uav_id for goal, items in assignments_by_goal.items() if len(items) == 1}
    result = list(shared_dependencies)
    for edge in task_spec.ordering_constraints:
        before_local, after_local = edge.before_goal_id in local, edge.after_goal_id in local
        if before_local == after_local:
            continue
        outside = edge.after_goal_id if before_local else edge.before_goal_id
        external_state = goal_states.get(outside, "UNKNOWN")
        # A future external successor is safe only while it remains pending;
        # a local successor cannot start until its external predecessor is done.
        satisfied = external_state == ("PENDING" if before_local else "CONFIRMED")
        result.append(ExternalDependencySnapshot(edge.constraint_id,
            "SUCCESSOR" if before_local else "PREDECESSOR", (edge.before_goal_id, edge.after_goal_id),
            tuple(sorted({owner[goal] for goal in (edge.before_goal_id, edge.after_goal_id) if goal in owner})),
            versions.get(edge.constraint_id, 0), "SATISFIED" if satisfied else "BLOCKED",
            tuple(goal_evidence_refs.get(outside, ())), edge.before_goal_id, edge.after_goal_id))
    for constraint in task_spec.assignment_constraints:
        if not local.intersection(constraint.goal_ids):
            continue
        known = all(goal in owner for goal in constraint.goal_ids)
        correct = known and all(owner[goal] == constraint.uav_id for goal in constraint.goal_ids)
        state = ("UNKNOWN" if not known else "BLOCKED"
                 if constraint.strength is ConstraintStrength.MUST and not correct else "UNCHANGED")
        bindings = [{"goal_id": goal, "uav_id": owner[goal],
                     "assignment_id": getattr(assignments_by_goal[goal][0], "assignment_id", None)}
                    for goal in sorted(constraint.goal_ids) if goal in owner]
        # Owner snapshots are reconstructed at final admission. The digest
        # changes even if two goals exchange owners inside the same UAV set.
        owner_proof = ("assignment_" + _digest(bindings)[:48],) if known else ()
        result.append(ExternalDependencySnapshot(constraint.constraint_id, "ASSIGNMENT", tuple(constraint.goal_ids),
            tuple(sorted({constraint.uav_id, *(owner[goal] for goal in constraint.goal_ids if goal in owner)})),
            versions.get(constraint.constraint_id, 0), state, owner_proof,
            scope="LOCAL" if set(constraint.goal_ids).issubset(local) else "EXTERNAL",
            strength=constraint.strength.value, source_evidence_refs=constraint.evidence_refs))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class LocalRepairContextV3:
    fleet_mission_id: str
    assignment_id: str
    request_id: str
    episode_id: str
    execution_generation: int
    original: CompiledMission
    current_step_id: str
    completed_step_ids: tuple[str, ...]
    completed_step_outputs: Mapping[str, object]
    goals: tuple[Goal, ...]
    evidence: tuple[SkillExecutionEvidence, ...]
    anchor: RepairAnchor
    submitted_wall_s: float
    deadline_wall_s: float
    external_dependencies: tuple[ExternalDependencySnapshot, ...] = ()
    dependency_versions: Mapping[str, int] = field(default_factory=dict)
    mode: str = "LINEAR"
    trusted_target_id: str | None = None
    home_name: str = "home"
    max_suffix_steps: int = 8
    current_step_started: bool = True
    ordering_constraints: tuple[OrderingConstraint, ...] = ()

    def __post_init__(self):
        for name in ("fleet_mission_id", "assignment_id", "request_id", "episode_id", "current_step_id"):
            validate_routing_id(getattr(self, name), name)
        if self.mode != "LINEAR":
            raise LocalRepairError("UNSUPPORTED_MODE", "graph plans cannot enter the linear V3 repair path")
        _version(self.execution_generation, "execution_generation")
        if not isinstance(self.current_step_started, bool):
            raise TypeError("current_step_started must be bool")
        if type(self.max_suffix_steps) is not int or not 1 <= self.max_suffix_steps <= 10:
            raise ValueError("max_suffix_steps must be in 1..10")
        object.__setattr__(self, "original", _frozen_compiled(self.original))
        original = self.original.planner_output
        task = self.original.task_plan
        if (original.mission_id, original.uav_id, original.plan_version) != (task.mission_id, task.uav_id, task.plan_version):
            raise LocalRepairError("ROUTING_MISMATCH", "semantic and executable base routes differ")
        ids = tuple(step.id for step in original.steps)
        if self.current_step_id not in ids:
            raise LocalRepairError("CURRENT_STEP_NOT_IN_V3", "current step has no V3 semantic source")
        index = ids.index(self.current_step_id)
        completed = tuple(self.completed_step_ids)
        if completed != ids[:index] or tuple(step.step_id for step in task.steps[:index]) != completed:
            raise LocalRepairError("COMPLETED_PREFIX_MISMATCH", "completed steps must be exactly the trusted prefix")
        object.__setattr__(self, "completed_step_ids", completed)
        if set(self.completed_step_outputs) - set(completed):
            raise LocalRepairError("COMPLETED_OUTPUT_INVALID", "outputs may name completed prefix steps only")
        _json(self.completed_step_outputs)
        object.__setattr__(self, "completed_step_outputs", _freeze(self.completed_step_outputs))
        for name in ("goals", "evidence", "external_dependencies", "ordering_constraints"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if any(not isinstance(goal, (MissionGoal, TerminationGoal)) for goal in self.goals):
            raise TypeError("goals must be trusted semantic goal objects")
        if len({goal.goal_id for goal in self.goals}) != len(self.goals):
            raise ValueError("duplicate goal ID")
        if any(not isinstance(item, SkillExecutionEvidence) for item in self.evidence):
            raise TypeError("evidence must be SkillExecutionEvidence")
        for proof in self.evidence:
            if proof.step_id in completed:
                output = self.completed_step_outputs.get(proof.step_id)
                if isinstance(output, Mapping) and "target_id" in output and output["target_id"] != proof.result.data.get("target_id"):
                    raise LocalRepairError("COMPLETED_OUTPUT_INVALID", "completed target output disagrees with the Skill result evidence")
        if any(not isinstance(item, ExternalDependencySnapshot) for item in self.external_dependencies):
            raise TypeError("external dependencies must be snapshots")
        if any(not isinstance(item, OrderingConstraint) for item in self.ordering_constraints):
            raise TypeError("ordering constraints must be immutable trusted OrderingConstraint values")
        external = {item.dependency_id: item for item in self.external_dependencies}
        if len(external) != len(self.external_dependencies):
            raise LocalRepairError("DUPLICATE_DEPENDENCY", "dependency IDs must be unique")
        local_goals = {goal.goal_id for goal in self.goals}
        for edge in self.ordering_constraints:
            if (edge.before_goal_id in local_goals) != (edge.after_goal_id in local_goals):
                dependency = external.get(edge.constraint_id)
                if dependency is None or (dependency.before_goal_id, dependency.after_goal_id) != (edge.before_goal_id, edge.after_goal_id):
                    raise LocalRepairError("EXTERNAL_DEPENDENCY_MISSING", "cross-subset ordering must retain both endpoints and its versioned evidence",
                        goal_ids=(edge.before_goal_id, edge.after_goal_id))
        if not isinstance(self.anchor, RepairAnchor):
            raise TypeError("anchor must be RepairAnchor")
        for value in self.dependency_versions.values():
            _version(value, "dependency version")
        object.__setattr__(self, "dependency_versions", _freeze(self.dependency_versions))
        for name in ("submitted_wall_s", "deadline_wall_s"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.deadline_wall_s <= self.submitted_wall_s:
            raise ValueError("deadline must follow submission on the monotonic wall clock")
        if self.trusted_target_id is not None:
            validate_routing_id(self.trusted_target_id, "trusted_target_id")

    @property
    def assessment(self):
        return assess_remaining_goals(self.goals, self.original.planner_output, self.completed_step_ids, self.evidence,
            current_step_id=self.current_step_id, current_step_started=self.current_step_started, home_name=self.home_name)

    @property
    def digest(self):
        return _digest(self.to_dict())

    def to_dict(self):
        original = self.original.planner_output
        return {"schema_version": 3, "fleet_mission_id": self.fleet_mission_id, "assignment_id": self.assignment_id,
                "request_id": self.request_id, "episode_id": self.episode_id, "execution_generation": self.execution_generation,
                "mission_id": original.mission_id, "uav_id": original.uav_id, "base_plan_version": original.plan_version,
                "new_plan_version": original.plan_version + 1, "replace_from_step_id": self.current_step_id,
                "completed_step_ids": list(self.completed_step_ids), "completed_step_outputs": _thaw(self.completed_step_outputs),
                "original_plan": original.to_dict(), "original_task_plan": self.original.task_plan.to_dict(),
                "goals": [goal.to_dict() for goal in self.goals], "evidence": [item.to_dict() for item in self.evidence],
                "external_dependencies": [item.to_dict() for item in self.external_dependencies],
                "dependency_versions": dict(self.dependency_versions), "anchor": self.anchor.to_dict(),
                "submitted_wall_s": self.submitted_wall_s, "deadline_wall_s": self.deadline_wall_s,
                "trusted_target_id": self.trusted_target_id, "max_suffix_steps": self.max_suffix_steps,
                "current_step_started": self.current_step_started,
                "ordering_constraints": [item.to_dict() for item in self.ordering_constraints]}


@dataclass(frozen=True, slots=True)
class LocalRepairDraftV3:
    schema_version: int
    request_id: str
    episode_id: str
    mission_id: str
    uav_id: str
    base_plan_version: int
    new_plan_version: int
    replace_from_step_id: str
    anchor_id: str
    steps: tuple[PlanStepDraftV3, ...]

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 3:
            raise LocalRepairError("UNSUPPORTED_CONTRACT", "suffix draft must use schema 3")
        for name in ("request_id", "episode_id", "replace_from_step_id", "anchor_id"):
            validate_routing_id(getattr(self, name), name)
        validate_mission_id(self.mission_id)
        validate_uav_id(self.uav_id)
        _version(self.base_plan_version, "base_plan_version")
        _version(self.new_plan_version, "new_plan_version")
        if self.base_plan_version < 1 or self.new_plan_version != self.base_plan_version + 1:
            raise LocalRepairError("VERSION_MISMATCH", "new version must increment the base version exactly once")
        steps = tuple(self.steps)
        if not 1 <= len(steps) <= 10 or any(not isinstance(step, PlanStepDraftV3) for step in steps):
            raise LocalRepairError("INVALID_SUFFIX", "suffix must contain 1..10 V3 steps")
        if len({step.id for step in steps}) != len(steps) or any(step.uav_id != self.uav_id for step in steps):
            raise LocalRepairError("ROUTING_MISMATCH", "suffix step IDs must be unique and belong to this UAV")
        object.__setattr__(self, "steps", steps)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise LocalRepairError("INVALID_SUFFIX_FIELDS", "suffix must contain exactly the authorized V3 output fields")
        if not isinstance(value["steps"], (list, tuple)):
            raise LocalRepairError("INVALID_SUFFIX", "steps must be an array")
        return cls(**{**value, "steps": tuple(PlanStepDraftV3.from_dict(step) for step in value["steps"])})

    def to_dict(self):
        return {**{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "steps"},
                "steps": [step.to_dict() for step in self.steps]}


@dataclass(frozen=True, slots=True)
class LocalRepairCandidate:
    compiled_mission: CompiledMission
    world_route: tuple[tuple[float, float, float], ...]
    context_digest: str
    dependency_digest: str
    completed_step_outputs: Mapping[str, object]
    replace_from_index: int

    @property
    def task_plan(self):
        return self.compiled_mission.task_plan


def build_local_repair_json_schema(context: LocalRepairContextV3) -> dict[str, object]:
    """Reuse V3 argument grammar while exposing only an authorized suffix."""
    from planner.json_schema_v3 import build_skill_plan_v3_json_schema
    original = context.original.planner_output
    full = build_skill_plan_v3_json_schema(mission_id=original.mission_id, uav_id=original.uav_id,
        plan_version=original.plan_version + 1, trusted_target_locked=context.trusted_target_id is not None)
    steps = full["properties"]["steps"]
    steps["minItems"] = 1
    steps["maxItems"] = min(context.max_suffix_steps, 10 - len(context.completed_step_ids))
    steps["items"]["oneOf"] = [variant for variant in steps["items"]["oneOf"]
                                if variant["properties"]["skill"].get("const") != "TAKEOFF"]
    envelope = {"schema_version": 3, "request_id": context.request_id, "episode_id": context.episode_id,
        "mission_id": original.mission_id, "uav_id": original.uav_id, "base_plan_version": original.plan_version,
        "new_plan_version": original.plan_version + 1, "replace_from_step_id": context.current_step_id,
        "anchor_id": context.anchor.anchor_id}
    properties = {name: {"const": value} for name, value in envelope.items()}
    properties["steps"] = steps
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _check_protected_semantics(context, draft):
    original = context.original.planner_output
    index = len(context.completed_step_ids)
    old_suffix = original.steps[index:]
    new_by_id = {step.id: step for step in draft.steps}
    old_by_id = {step.id: step for step in old_suffix}
    prefix_ids = set(context.completed_step_ids)
    if prefix_ids.intersection(new_by_id):
        raise LocalRepairError("COMPLETED_PREFIX_MUTATION", "suffix reuses a completed step ID")
    if any(step.skill in {"TAKEOFF", "INSPECT", "FOLLOW_ROUTE"} for step in draft.steps):
        raise LocalRepairError("UNSUPPORTED_SUFFIX_SKILL", "suffix cannot replay takeoff or use unsupported runtime skill contracts")
    if any(step.skill != "GOTO" and step.id not in old_by_id for step in draft.steps):
        raise LocalRepairError("UNAUTHORIZED_NEW_EFFECT", "new suffix steps may only be bounded transit GOTO waypoints")
    current = old_suffix[0]
    retained = new_by_id.get(current.id)
    if retained is None or retained.skill != current.skill:
        raise LocalRepairError("CURRENT_STEP_MUTATION", "repair must retain the interrupted step identity and Skill")
    current_index = next(i for i, step in enumerate(draft.steps) if step.id == current.id)
    if any(step.skill != "GOTO" or step.id in old_by_id for step in draft.steps[:current_index]):
        raise LocalRepairError("INVALID_DETOUR_PREFIX", "only new transit GOTO steps may precede the interrupted step")
    for old in old_suffix:
        if old.skill in {"TRACK", "HOVER", "LAND"}:
            if old.id not in new_by_id or new_by_id[old.id].to_dict() != old.to_dict():
                raise LocalRepairError("PROTECTED_STEP_MUTATION", "target references, durations and termination steps are immutable")
        if old.skill == "SEARCH":
            new = new_by_id.get(old.id)
            if new is None or new.skill != "SEARCH" or any(new.to_dict()["args"].get(key) != old.to_dict()["args"].get(key)
                    for key in ("target_description", "region", "search_altitude_m")):
                raise LocalRepairError("TARGET_CONDITION_MUTATION", "SEARCH identity, area and altitude must be retained")
    old_refs = {step.args.get("target_ref") for step in old_suffix if step.skill == "TRACK"}
    for reference in old_refs:
        if isinstance(reference, str) and reference.startswith("$") and reference.endswith(".target_id"):
            source_id = reference[1:-10]
            if source_id in prefix_ids:
                output = context.completed_step_outputs.get(source_id)
                if not isinstance(output, Mapping) or not output.get("target_id"):
                    raise LocalRepairError("COMPLETED_OUTPUT_INVALID", "a retained target reference has no trusted prefix output")
                if context.trusted_target_id is not None and output["target_id"] != context.trusted_target_id:
                    raise LocalRepairError("TARGET_IDENTITY_MISMATCH", "prefix output and current trusted target differ")
    confirmed = set(context.assessment.confirmed_goal_ids)
    for goal in context.goals:
        if goal.goal_id in confirmed and _goal_steps(goal, draft.steps, context.home_name):
            raise LocalRepairError("COMPLETED_GOAL_REPLAY", "suffix repeats a confirmed user goal", goal_ids=(goal.goal_id,))


def _world_prefix(context):
    """Materialize historical geometry for compilation without re-anchoring it.

    Only the compiler's state-reconstruction view changes. The published V3
    prefix and executable prefix remain byte-for-byte semantically unchanged.
    """
    prefix = []
    for semantic, runtime in zip(context.original.planner_output.steps, context.original.task_plan.steps):
        if semantic.id not in context.completed_step_ids:
            break
        data = semantic.to_dict()
        if semantic.skill == "GOTO":
            data["args"]["target"] = PointTarget(CoordinateFrame.WORLD_ENU, tuple(runtime.params["position"])).to_dict()
            data["args"].pop("altitude_m", None)
        elif semantic.skill == "SEARCH":
            data["args"]["region"] = runtime.params["region"].to_dict()
            for name in ("user_anchor_xyz_m", "model_selected_entry_xyz_m"):
                if name in runtime.params:
                    data["args"][name] = list(runtime.params[name])
        prefix.append(PlanStepDraftV3.from_dict(data))
    return tuple(prefix)


def _world_route(task_plan: TaskPlan, replace_index: int, anchor: RepairAnchor):
    """Conservative WORLD_ENU polyline; dynamic target motion remains runtime-owned."""
    from planner.region_compiler import RegionCompiler
    points = [anchor.pose.xyz_m]
    for step in task_plan.steps[replace_index:]:
        if step.skill is SkillName.GOTO:
            points.append(tuple(float(value) for value in step.params["position"]))
        elif step.skill is SkillName.SEARCH:
            params = step.params
            geometry = RegionCompiler(anchor.resolver).compile(region=params["region"], strategy=params["strategy"],
                entry_policy=params["entry_policy"], current_uav_xyz_m=points[-1], search_altitude_m=params["search_altitude_m"],
                user_anchor_xyz_m=params.get("user_anchor_xyz_m"), model_selected_entry_xyz_m=params.get("model_selected_entry_xyz_m"))
            points.extend(geometry.route_waypoints_xyz_m)
        elif step.skill is SkillName.LAND:
            xy = step.params["expected_position_xy"]
            points.append((float(xy[0]), float(xy[1]), float(step.params.get("ground_altitude", 0.0))))
    return tuple(points)


def _bind_search_entries(task_plan: TaskPlan, replace_index: int, anchor: RepairAnchor):
    """Keep runtime entry selection identical to the world polyline admission.

    START_IN_PLACE/nearest policies depend on position at execution time. A
    repair specializes that choice using its frozen HOLD anchor and preceding
    transit points; current-to-entry safety remains the owner's final check.
    """
    from planner.region_compiler import RegionCompiler
    from skills.search_strategy import SearchEntryPolicy
    current = anchor.pose.xyz_m
    steps = list(task_plan.steps)
    for index in range(replace_index, len(steps)):
        step = steps[index]
        if step.skill is SkillName.GOTO:
            current = tuple(step.params["position"])
        elif step.skill is SkillName.SEARCH:
            params = _runtime_copy(step.params)
            geometry = RegionCompiler(anchor.resolver).compile(region=params["region"], strategy=params["strategy"],
                entry_policy=params["entry_policy"], current_uav_xyz_m=current, search_altitude_m=params["search_altitude_m"],
                user_anchor_xyz_m=params.get("user_anchor_xyz_m"), model_selected_entry_xyz_m=params.get("model_selected_entry_xyz_m"))
            params["entry_policy"] = SearchEntryPolicy.MODEL_SELECTED
            params["model_selected_entry_xyz_m"] = geometry.entry_point_xyz_m
            params.pop("user_anchor_xyz_m", None)
            steps[index] = TaskStep(step.step_id, step.skill, params, step.recovery)
            current = geometry.route_waypoints_xyz_m[-1]
    return TaskPlan(tuple(steps), task_plan.mission_id, task_plan.uav_id, task_plan.plan_version)


def validate_local_repair(draft: LocalRepairDraftV3, context: LocalRepairContextV3,
                          world_context: PlannerWorldContext, *, dependency_versions: Mapping[str, int],
                          external_dependencies: Sequence[ExternalDependencySnapshot], now_wall_s: float,
                          plan_validator=None) -> LocalRepairCandidate:
    """Pure preparation. The owner must repeat live checks AFTER this returns."""
    if not isinstance(draft, LocalRepairDraftV3) or not isinstance(context, LocalRepairContextV3):
        raise TypeError("expected V3 repair draft and context")
    original = context.original.planner_output
    expected = (context.request_id, context.episode_id, original.mission_id, original.uav_id,
                original.plan_version, original.plan_version + 1, context.current_step_id, context.anchor.anchor_id)
    actual = (draft.request_id, draft.episode_id, draft.mission_id, draft.uav_id, draft.base_plan_version,
              draft.new_plan_version, draft.replace_from_step_id, draft.anchor_id)
    if actual != expected:
        raise LocalRepairError("ROUTING_MISMATCH", "model changed request, episode, route, version, anchor or authorized suffix")
    now = _number(now_wall_s, "now_wall_s")
    if now < context.submitted_wall_s:
        raise LocalRepairError("WALL_CLOCK_REGRESSION", "validation wall clock predates submission")
    if now >= context.deadline_wall_s:
        raise LocalRepairError("DEADLINE_EXPIRED", "repair candidate expired on the monotonic wall clock")
    deps = check_external_dependencies(context.external_dependencies, external_dependencies,
                                       context.dependency_versions, dependency_versions)
    if not deps.allowed:
        raise LocalRepairError("COORDINATION_REQUIRED", ";".join(deps.reasons), affected_uav_ids=deps.affected_uav_ids, goal_ids=deps.goal_ids)
    assessment = context.assessment
    if not assessment.supported:
        raise LocalRepairError("EVIDENCE_INSUFFICIENT", ";".join(assessment.reasons) or "remaining goals lack trustworthy completion evidence",
                               affected_uav_ids=(original.uav_id,), goal_ids=assessment.insufficient_goal_ids)
    if len(draft.steps) > context.max_suffix_steps:
        raise LocalRepairError("SUFFIX_BUDGET_EXCEEDED", "repair exceeds authorized suffix step budget")
    _check_protected_semantics(context, draft)
    index = len(context.completed_step_ids)
    revised = SkillPlanDraftV3(3, original.mission_id, original.uav_id, draft.new_plan_version,
                              original.assumptions, original.steps[:index] + draft.steps, original.target_spec)
    if plan_validator is None:
        from runtime.plan_validator import PlanValidator
        plan_validator = PlanValidator()
    # Check all original obligations against the union of immutable prefix and
    # candidate suffix. Prefix completion is independently justified above.
    coverage = GoalSatisfactionChecker().check(context.goals, revised,
        mission_id=original.mission_id, assignment_id=context.assignment_id, uav_id=original.uav_id,
        home_name=context.home_name, trusted_target_locked=context.trusted_target_id is not None)
    if not coverage.complete or coverage.validation_report.hard_blocked:
        raise LocalRepairError("GOAL_NOT_COVERED", "replacement does not preserve all original task obligations",
                               goal_ids=coverage.uncovered_goal_ids)
    by_goal = {item.goal_id: item.evidence_step_ids for item in coverage.coverages}
    positions = {step.id: i for i, step in enumerate(revised.steps)}
    for edge in context.ordering_constraints:
        before, after = by_goal.get(edge.before_goal_id), by_goal.get(edge.after_goal_id)
        if before and after and max(positions[item] for item in before) >= min(positions[item] for item in after):
            raise LocalRepairError("ORDERING_CONSTRAINT_VIOLATION", "local suffix changes a trusted ordering constraint")
    compiler_view = SkillPlanDraftV3(3, original.mission_id, original.uav_id, draft.new_plan_version,
                                    original.assumptions, _world_prefix(context) + draft.steps, original.target_spec)
    compiled = plan_validator.validate_and_compile(compiler_view, world_context, source="dynamic_llm",
        mission_id=original.mission_id, uav_id=original.uav_id, plan_version=draft.new_plan_version,
        trusted_target_id=context.trusted_target_id, spatial_resolver=context.anchor.resolver,
        allow_trusted_safety_completion=True)
    if tuple(step.step_id for step in compiled.task_plan.steps[:index]) != context.completed_step_ids:
        raise LocalRepairError("COMPLETED_PREFIX_MUTATION", "compiler changed completed prefix identity")
    # Never regenerate historical geometry or control arguments under a new
    # resolver. Keep exact compiled prefix objects/values and its output ledger.
    prefix = tuple(TaskStep(step.step_id, step.skill, _runtime_copy(step.params), step.recovery)
                   for step in context.original.task_plan.steps[:index])
    task = TaskPlan(prefix + compiled.task_plan.steps[index:],
                    original.mission_id, original.uav_id, draft.new_plan_version)
    task = _bind_search_entries(task, index, context.anchor)
    for goal in context.goals:
        if goal.goal_type is GoalType.TRACK_TARGET and goal.distance_m is not None:
            matched_ids = {step.id for step in _goal_steps(goal, revised.steps, context.home_name)}
            if any(abs(float(step.params["desired_distance"]) - goal.distance_m) > 1e-9
                   for step in task.steps if step.step_id in matched_ids):
                raise LocalRepairError("TARGET_CONDITION_MUTATION", "compiled TRACK distance does not satisfy the trusted goal")
    compiled = CompiledMission(revised, task, "dynamic_llm", (*compiled.compiler_notes, "repair SEARCH entries bound to admitted WORLD_ENU geometry"))
    return LocalRepairCandidate(compiled, _world_route(task, index, context.anchor), context.digest,
                                deps.digest, context.completed_step_outputs, index)
