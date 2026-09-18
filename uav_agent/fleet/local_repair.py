"""Pure, fail-closed contracts for owner-authorized Spatial V3 suffix repair.

This module never reads an Agent, controller, simulator, or model. Evidence and
dependency snapshots must be supplied by their runtime owners. A compiled
candidate still requires current-world entry/collision checks and an owner-side
compare-and-commit; compilation is not permission to execute.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json
from math import dist, isfinite
from types import MappingProxyType

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from fleet.contract_registry import (
    DEFAULT_GOAL_CONTRACT_REGISTRY, DEFAULT_SKILL_CONTRACTS, GoalContractRegistry,
    RestartPolicy, Transferability,
)
from fleet.task_spec import ConstraintStrength, FleetTaskSpecV1, GoalType, MissionGoal, OrderingConstraint, TerminationGoal
from planner.goal_checker import GoalSatisfactionChecker
from planner.schemas import CompiledMission, PlannerWorldContext
from planner.schemas_v3 import PlanStepDraftV3, SkillPlanDraftV3
from planner.spatial import CoordinateFrame, PointTarget
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


class SpatialReferenceVerdict(str, Enum):
    VALID = "VALID"
    REPROJECTABLE = "REPROJECTABLE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class SpatialReferenceValidity:
    """Code-only verdict on whether an admitted reference still carries meaning."""
    verdict: SpatialReferenceVerdict
    reasons: tuple[str, ...]
    code: str | None = None
    position_delta_m: float | None = None

    def __post_init__(self):
        if not isinstance(self.verdict, SpatialReferenceVerdict):
            raise TypeError("verdict must be a SpatialReferenceVerdict")
        reasons = tuple(dict.fromkeys(self.reasons))
        if any(not isinstance(item, str) or not item for item in reasons):
            raise ValueError("reasons must be nonempty strings")
        object.__setattr__(self, "reasons", reasons)
        if self.verdict is SpatialReferenceVerdict.VALID and (reasons or self.code is not None):
            raise ValueError("VALID carries no rejection reason")
        if self.verdict is not SpatialReferenceVerdict.VALID and not reasons:
            raise ValueError("non-VALID verdicts must state reasons")
        if self.position_delta_m is not None:
            object.__setattr__(self, "position_delta_m", _number(self.position_delta_m, "position_delta_m"))


def evaluate_spatial_reference(anchor: RepairAnchor, *, observation_time_s: float, pose_time_s: float,
                               time_domain: str, current_pose_xyz_m: Sequence[float], now_wall_s: float,
                               submitted_wall_s: float, max_anchor_age_s: float,
                               max_pose_time_error_s: float, max_hold_drift_m: float,
                               valid_pose_tolerance_m: float, map_version: int | None = None,
                               reference_version: int | None = None, route: Sequence[Sequence[float]] = (),
                               segment_blocked=None, route_conflicts: Sequence[Sequence[str]] = ()) \
        -> SpatialReferenceValidity:
    """Classify an admitted anchor against current trusted state, by code only.

    VALID: nothing meaning-bearing changed; the candidate proceeds to final checks.
    REPROJECTABLE: only a deterministically known pose change occurred; trusted
    code may reconnect the current position onto the existing admitted WORLD
    polyline (reproject_world_route). No target is re-resolved, so a later
    heading change can never reinterpret historical HOLD-relative geometry.
    INVALID: version, time-domain, alignment, drift, obstacle or cross-UAV
    changes destroyed the reference; the candidate must be discarded.
    """
    def reject(code, reason, delta=None):
        return SpatialReferenceValidity(SpatialReferenceVerdict.INVALID, (reason,), code, delta)

    try:
        observation = _number(float(observation_time_s), "observation_time_s")
        pose_stamp = _number(float(pose_time_s), "pose_time_s")
    except (TypeError, ValueError):
        return reject("OBSERVATION_TIME_MISMATCH", "OBSERVATION_NOT_FINITE")
    if not all(isinstance(value, (int, float)) and isfinite(value) for value in current_pose_xyz_m) \
            or len(tuple(current_pose_xyz_m)) != 3:
        return reject("OBSERVATION_TIME_MISMATCH", "POSE_NOT_FINITE")
    if time_domain != anchor.time_domain:
        return reject("REFERENCE_CHANGED", "TIME_DOMAIN_CHANGED")
    if observation < anchor.observation_time_s:
        return reject("OBSERVATION_TIME_MISMATCH", "OBSERVATION_TIME_REGRESSED")
    if abs(observation - pose_stamp) > max(max_pose_time_error_s, anchor.max_pose_time_error_s):
        return reject("OBSERVATION_TIME_MISMATCH", "OBSERVATION_POSE_MISALIGNED")
    if map_version is not None and _version(map_version, "map_version") != anchor.map_version:
        return reject("REFERENCE_CHANGED", "MAP_VERSION_CHANGED")
    if reference_version is not None and _version(reference_version, "reference_version") != anchor.reference_version:
        return reject("REFERENCE_CHANGED", "REFERENCE_VERSION_CHANGED")
    now = _number(float(now_wall_s), "now_wall_s")
    if now < submitted_wall_s:
        return reject("WALL_CLOCK_REGRESSION", "NOW_PREDATES_SUBMISSION")
    if now - submitted_wall_s > max_anchor_age_s:
        return reject("ANCHOR_EXPIRED", "ANCHOR_OLDER_THAN_LIMIT")
    current = tuple(float(value) for value in current_pose_xyz_m)
    delta = dist(current, tuple(anchor.pose.xyz_m))
    if delta > max_hold_drift_m:
        return reject("HOLD_DRIFT", "VEHICLE_LEFT_ADMITTED_WAITING_REGION", delta)
    points = tuple(tuple(float(value) for value in point) for point in route)
    if segment_blocked is not None:
        for a, b in zip(points, points[1:]):
            if segment_blocked(a, b):
                return reject("UNSAFE_ENTRY_OR_ROUTE", "ROUTE_CROSSES_CURRENT_OBSTACLE", delta)
    conflicts = tuple((str(pair[0]), str(pair[1])) for pair in route_conflicts if len(pair) == 2)
    if conflicts:
        return reject("SHARED_SPACE_CONFLICT", "ROUTE_CONFLICTS_WITH_ANOTHER_UAV", delta)
    if delta <= valid_pose_tolerance_m:
        return SpatialReferenceValidity(SpatialReferenceVerdict.VALID, (), None, delta)
    return SpatialReferenceValidity(SpatialReferenceVerdict.REPROJECTABLE,
                                    ("POSE_WITHIN_DRIFT_LIMIT",), None, delta)


def reproject_world_route(route: Sequence[Sequence[float]], current_xyz_m: Sequence[float]):
    """Reconnect current position onto the admitted WORLD polyline, by code only.

    The admitted polyline beyond its first point is preserved byte-for-byte:
    no anchor reinterpretation, no HOLD-relative re-resolution and no new model
    request. Obstacle/separation validity of the new access segment remains
    the owner's live check.
    """
    points = tuple(tuple(float(value) for value in point) for point in route)
    if not points or not all(len(point) == 3 for point in points):
        raise LocalRepairError("UNSUPPORTED_CONTRACT", "world route must be a nonempty 3D polyline")
    if len(tuple(current_xyz_m)) != 3 or not all(isfinite(float(value)) for value in current_xyz_m):
        raise LocalRepairError("UNSUPPORTED_CONTRACT", "reprojection requires a finite current position")
    return (tuple(float(value) for value in current_xyz_m),) + points[1:]


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


_SUCCESS = {name: contract.success_code for name, contract in DEFAULT_SKILL_CONTRACTS.items()}


def _goal_steps(goal: Goal, steps: Sequence[PlanStepDraftV3], home_name: str, *,
                registry: GoalContractRegistry | None = None):
    """Compatibility wrapper: step matching now lives in the contract registry."""
    return matching_steps_for(goal, steps, home_name, registry=registry)


def matching_steps_for(goal: Goal, steps: Sequence[PlanStepDraftV3], home_name: str, *,
                       registry: GoalContractRegistry | None = None):
    registry = DEFAULT_GOAL_CONTRACT_REGISTRY if registry is None else registry
    return registry.matching_steps(goal, steps, home_name)


def _terminal_evidence_index(evidence: Sequence[SkillExecutionEvidence], original: SkillPlanDraftV3,
                             completed: set[str]):
    """Single trusted grouping of terminal evidence; shared by every consumer."""
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
    return by_step, grouped


def assess_remaining_goals(goals: Sequence[Goal], original: SkillPlanDraftV3,
                           completed_step_ids: Sequence[str], evidence: Sequence[SkillExecutionEvidence],
                           *, current_step_id: str | None, current_step_started: bool = True,
                           home_name: str = "home",
                           registry: GoalContractRegistry | None = None) -> RemainingGoalAssessment:
    """Subtract only terminal Skill evidence; elapsed/feedback time is not credit.

    Completion semantics are looked up in the contract registry; an
    unregistered goal type is recorded as insufficient with an explicit
    UNREGISTERED_GOAL_CONTRACT reason instead of guessing a completion state.
    """
    registry = DEFAULT_GOAL_CONTRACT_REGISTRY if registry is None else registry
    skills = registry.skill_contracts
    goals, evidence = tuple(goals), tuple(evidence)
    completed = set(completed_step_ids)
    by_step, grouped = _terminal_evidence_index(evidence, original, completed)
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
        evaluator = registry.evaluator_for(goal.goal_type)
        if evaluator is None:
            insufficient.append(goal.goal_id)
            reason = "UNREGISTERED_GOAL_CONTRACT:" + goal.goal_type.value
            if reason not in reasons:
                reasons.append(reason)
            continue
        matches = evaluator.matching_steps(goal, original.steps, home_name)
        ambiguous = (evaluator.ambiguous_with_sibling and
                     sum(other.goal_type is goal.goal_type for other in goals) > 1)
        if not matches or ambiguous or (len(target_aliases) > 1 and getattr(goal, "target_alias", None)):
            insufficient.append(goal.goal_id)
            continue
        if current is not None and current_step_started and current.skill in {"TRACK", "HOVER"} and current in matches:
            insufficient.append(goal.goal_id)
            continue
        done, credits, unknown = evaluator.evaluate_evidence(goal, matches, completed,
                                                             by_step, grouped, skills)
        if unknown:
            insufficient.append(goal.goal_id)
            continue
        verdict, remaining_goal = evaluator.compute_remaining(goal, done, credits, skills)
        if verdict == "CONFIRMED":
            confirmed.append(goal.goal_id)
        elif verdict == "PENDING":
            pending.append(remaining_goal)
        else:
            insufficient.append(goal.goal_id)
    return RemainingGoalAssessment(tuple(confirmed), tuple(pending), tuple(insufficient), tuple(reasons))


# RestartPolicy/Transferability are re-exported from the contract registry so
# existing imports keep working; the registry owns every completion semantic.


def _original_condition(goal: Goal) -> Mapping[str, object]:
    if isinstance(goal, TerminationGoal):
        return _freeze({"uav_id": goal.uav_id, "duration_s": goal.duration_s, "strength": goal.strength.value})
    return _freeze({"target_alias": goal.target_alias,
                    "spatial_constraint": None if goal.spatial_constraint is None else goal.spatial_constraint.to_dict(),
                    "duration_s": goal.duration_s, "distance_m": goal.distance_m,
                    "strength": goal.strength.value, "completion_basis": goal.completion_basis})


@dataclass(frozen=True, slots=True)
class GoalTaskContract:
    """One goal's trusted remaining obligation; never model-generated or editable."""
    goal_id: str
    goal_type: GoalType
    status: str  # CONFIRMED / PENDING / INSUFFICIENT
    target_binding: str | None
    original_condition: Mapping[str, object]
    completion_basis: str
    confirmed_amount: float | None
    remaining_amount: float | None
    evidence_refs: tuple[str, ...]
    remaining_goal: Goal | None
    restart_policy: RestartPolicy
    transferability: Transferability
    reasons: tuple[str, ...]
    required_resources: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in {"CONFIRMED", "PENDING", "INSUFFICIENT"}:
            raise ValueError("unknown goal contract status")
        if not isinstance(self.restart_policy, RestartPolicy) or not isinstance(self.transferability, Transferability):
            raise TypeError("restart_policy and transferability are trusted enum verdicts")
        if self.goal_id in {"", None} or not isinstance(self.goal_id, str):
            raise ValueError("goal contract requires a goal_id")
        object.__setattr__(self, "original_condition", _freeze(dict(self.original_condition)))
        _json(self.original_condition)
        for name in ("confirmed_amount", "remaining_amount"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _number(value, name))
        refs = tuple(dict.fromkeys(self.evidence_refs))
        if any(not isinstance(ref, str) or not ref for ref in refs):
            raise ValueError("evidence refs must be nonempty strings")
        object.__setattr__(self, "evidence_refs", refs)
        resources = tuple(dict.fromkeys(self.required_resources))
        if any(not isinstance(item, str) or not item for item in resources):
            raise ValueError("required resources must be nonempty strings")
        object.__setattr__(self, "required_resources", resources)

    def to_dict(self):
        return {"goal_id": self.goal_id, "goal_type": self.goal_type.value, "status": self.status,
                "target_binding": self.target_binding, "original_condition": _thaw(self.original_condition),
                "completion_basis": self.completion_basis,
                "confirmed_amount": self.confirmed_amount, "remaining_amount": self.remaining_amount,
                "evidence_refs": list(self.evidence_refs),
                "remaining_obligation": None if self.remaining_goal is None else self.remaining_goal.to_dict(),
                "restart_policy": self.restart_policy.value, "transferability": self.transferability.value,
                "required_resources": list(self.required_resources),
                "reasons": list(self.reasons)}


@dataclass(frozen=True, slots=True)
class RemainingTaskContract:
    """The single remaining-task computation shared by local repair and takeover.

    Built only from SkillManager terminal evidence via assess_remaining_goals.
    Elapsed time, model claims and feedback never enter this contract. Qwen
    consumers may read it; no consumer may write completion amounts, goal
    identity or completion conditions back through it.
    """
    schema_version: int
    mission_id: str
    uav_id: str
    plan_version: int
    goals: tuple[GoalTaskContract, ...]
    assessment: RemainingGoalAssessment
    shared_target_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("remaining task contract must use schema 1")
        validate_mission_id(self.mission_id)
        validate_uav_id(self.uav_id)
        _version(self.plan_version, "plan_version")
        goals = tuple(self.goals)
        if any(not isinstance(item, GoalTaskContract) for item in goals):
            raise TypeError("contract goals must be GoalTaskContract entries")
        if len({item.goal_id for item in goals}) != len(goals):
            raise ValueError("duplicate goal ID in remaining task contract")
        object.__setattr__(self, "goals", goals)
        if not isinstance(self.assessment, RemainingGoalAssessment):
            raise TypeError("contract wraps exactly one RemainingGoalAssessment")
        object.__setattr__(self, "shared_target_ids", tuple(sorted(set(self.shared_target_ids))))

    @property
    def computed_by(self):
        return "skill_terminal_evidence"

    @property
    def supported(self):
        return self.assessment.supported

    @property
    def confirmed_goal_ids(self):
        return self.assessment.confirmed_goal_ids

    @property
    def pending_goals(self):
        return self.assessment.pending_goals

    @property
    def pending_goal_ids(self):
        return self.assessment.pending_goal_ids

    @property
    def insufficient_goal_ids(self):
        return self.assessment.insufficient_goal_ids

    @property
    def pending_entries(self):
        return tuple(item for item in self.goals if item.status == "PENDING")

    @property
    def handoff_entries(self):
        """Pending obligations a replacement aircraft may own without shared evidence."""
        return tuple(item for item in self.goals
                     if item.status == "PENDING" and item.transferability is Transferability.TRANSFERABLE)

    @property
    def digest(self):
        return _digest(self.to_dict())

    def entry(self, goal_id):
        return next((item for item in self.goals if item.goal_id == goal_id), None)

    def to_dict(self):
        return {"schema_version": self.schema_version, "computed_by": self.computed_by,
                "mission_id": self.mission_id, "uav_id": self.uav_id, "plan_version": self.plan_version,
                "read_only": True, "shared_target_ids": list(self.shared_target_ids),
                "goals": [item.to_dict() for item in self.goals],
                "global_reasons": list(self.assessment.reasons)}


def build_remaining_task_contract(goals: Sequence[Goal], original: SkillPlanDraftV3,
                                  completed_step_ids: Sequence[str], evidence: Sequence[SkillExecutionEvidence], *,
                                  current_step_id: str | None, current_step_started: bool = True,
                                  home_name: str = "home", shared_target_ids: Sequence[str] = (),
                                  consumer: str = "LOCAL_REPAIR",
                                  registry: GoalContractRegistry | None = None) -> RemainingTaskContract:
    """The only remaining-task computation; both repair and handoff consume this.

    consumer is logging context only: it never changes amounts, identity or
    transferability. Insufficient evidence fails closed to CANNOT_RESUME.
    Completion semantics come from the contract registry; unregistered goal
    types stay INSUFFICIENT with CANNOT_RESUME.
    """
    if consumer not in {"LOCAL_REPAIR", "HANDOFF", "FLEET_STATE", "JOINT_REPAIR"}:
        raise ValueError("unknown remaining task contract consumer")
    registry = DEFAULT_GOAL_CONTRACT_REGISTRY if registry is None else registry
    skills = registry.skill_contracts
    goals = tuple(goals)
    assessment = assess_remaining_goals(goals, original, completed_step_ids, evidence,
        current_step_id=current_step_id, current_step_started=current_step_started,
        home_name=home_name, registry=registry)
    completed = set(completed_step_ids)
    by_step, grouped = _terminal_evidence_index(evidence, original, completed)
    shared = frozenset(shared_target_ids)
    pending_by_id = {goal.goal_id: goal for goal in assessment.pending_goals}
    confirmed, insufficient = set(assessment.confirmed_goal_ids), set(assessment.insufficient_goal_ids)
    entries = []
    for goal in goals:
        evaluator = registry.evaluator_for(goal.goal_type)
        matched = () if evaluator is None else evaluator.matching_steps(goal, original.steps, home_name)
        refs = []
        for step in matched:
            if step.id not in completed:
                continue
            contract = skills.get(step.skill)
            proof = by_step.get(step.id)
            if contract is not None and contract.validate_completion(step, proof):
                refs.extend(item.invocation_id for item in grouped[step.id])
        status = ("CONFIRMED" if goal.goal_id in confirmed
                  else "PENDING" if goal.goal_id in pending_by_id else "INSUFFICIENT")
        confirmed_amount = remaining_amount = None
        requested = goal.duration_s if isinstance(goal, (MissionGoal, TerminationGoal)) else None
        if status == "CONFIRMED" and requested is not None:
            confirmed_amount, remaining_amount = float(requested), 0.0
        elif status == "PENDING" and requested is not None:
            remaining_amount = float(pending_by_id[goal.goal_id].duration_s)
            confirmed_amount = max(0.0, float(requested) - remaining_amount)
        if evaluator is None:
            restart, transferable = RestartPolicy.CANNOT_RESUME, Transferability.SAME_UAV_ONLY
        else:
            restart = evaluator.restart_policy(status, confirmed_amount)
            transferable = evaluator.transferability(goal, status, shared)
        entries.append(GoalTaskContract(
            goal_id=goal.goal_id, goal_type=goal.goal_type, status=status,
            target_binding=getattr(goal, "target_alias", None),
            original_condition=_original_condition(goal),
            completion_basis=getattr(goal, "completion_basis", "valid_execution"),
            confirmed_amount=confirmed_amount, remaining_amount=remaining_amount,
            evidence_refs=tuple(refs),
            remaining_goal=pending_by_id.get(goal.goal_id),
            restart_policy=restart, transferability=transferable,
            required_resources=() if evaluator is None else evaluator.required_resources(goal),
            reasons=() if status != "INSUFFICIENT" else ("INSUFFICIENT_COMPLETION_EVIDENCE",)))
    return RemainingTaskContract(1, original.mission_id, original.uav_id, original.plan_version,
                                 tuple(entries), assessment, tuple(sorted(shared)))


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


class DependencyVerdict(str, Enum):
    LOCAL_OK = "LOCAL_OK"
    COORDINATION_REQUIRED = "COORDINATION_REQUIRED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class CrossUAVDependencyCheck:
    """The unified cross-UAV dependency outcome for local repair and handoff.

    LOCAL_OK: the faulted UAV may proceed alone. COORDINATION_REQUIRED names
    the affected UAVs and dependencies that must be resolved first; it is an
    admission to stop, not an automatic joint replan, and the affected set is
    a sound over-approximation rather than a proven minimal set. INVALID means
    the dependency evidence itself is structurally unusable.
    """
    verdict: DependencyVerdict
    reasons: tuple[str, ...]
    affected_uav_ids: tuple[str, ...]
    goal_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    digest: str

    @property
    def allowed(self):
        return self.verdict is DependencyVerdict.LOCAL_OK


def check_cross_uav_dependencies(expected_dependencies, current_dependencies, expected_versions,
                                 current_versions, *, route_conflicts: Sequence[Sequence[str]] = (),
                                 shared_resource_ids: Sequence[str] = ()) -> CrossUAVDependencyCheck:
    """One dependency decision; structural corruption is INVALID, everything
    blocking is COORDINATION_REQUIRED, otherwise LOCAL_OK."""
    expected, current = tuple(expected_dependencies), tuple(current_dependencies)
    digest = dependency_digest(current, current_versions)
    hard, coordination, cited = [], [], set()
    seen = set()
    for item in current:
        if item.dependency_id in seen:
            hard.append("DUPLICATE_DEPENDENCY")
        seen.add(item.dependency_id)
    expected_by_id = {item.dependency_id: item for item in expected}
    current_by_id = {item.dependency_id: item for item in current}
    for dependency_id in sorted(expected_by_id):
        expected_version = _version(dict(expected_versions).get(dependency_id, expected_by_id[dependency_id].version), "version")
        present = current_by_id.get(dependency_id)
        if present is None:
            hard.append("MISSING_DEPENDENCY:" + dependency_id)
            cited.add(dependency_id)
            continue
        current_version = _version(dict(current_versions).get(dependency_id, present.version), "version")
        if current_version < expected_version:
            hard.append("VERSION_REGRESSION:" + dependency_id)
            cited.add(dependency_id)
    changed_digest = digest != dependency_digest(expected, expected_versions)
    if changed_digest:
        coordination.append("DEPENDENCY_SNAPSHOT_CHANGED")
        for dependency_id in sorted(set(expected_by_id) | set(current_by_id)):
            before, after = expected_by_id.get(dependency_id), current_by_id.get(dependency_id)
            if dependency_id in shared_resource_ids or (before is not None and after is not None
                    and (before.to_dict() != after.to_dict()
                         or dict(expected_versions).get(dependency_id) != dict(current_versions).get(dependency_id))):
                cited.add(dependency_id)
    for item in current:
        if item.state not in {"SATISFIED", "UNCHANGED"} or not item.evidence_refs:
            coordination.append("EXTERNAL_DEPENDENCY_REQUIRES_COORDINATION:" + item.dependency_id)
            cited.add(item.dependency_id)
    participants = set()
    for pair in route_conflicts:
        pair = tuple(str(value) for value in pair)
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("route conflicts must name two distinct UAVs")
        coordination.append("ROUTE_CONFLICT:" + pair[0] + ":" + pair[1])
        participants.update(pair)
    verdict = (DependencyVerdict.INVALID if hard
               else DependencyVerdict.COORDINATION_REQUIRED if coordination else DependencyVerdict.LOCAL_OK)
    affected = tuple(sorted({uav for item in (*expected, *current) for uav in item.uav_ids} | participants))
    goals = tuple(sorted({goal for item in (*expected, *current) for goal in item.goal_ids}))
    return CrossUAVDependencyCheck(verdict, tuple(dict.fromkeys((*hard, *coordination))), affected, goals,
                                   tuple(sorted(cited)) if verdict is not DependencyVerdict.LOCAL_OK else (), digest)


def check_external_dependencies(expected_dependencies, current_dependencies, expected_versions,
                                current_versions) -> DependencyCheck:
    """Compatibility view over the unified cross-UAV dependency check."""
    check = check_cross_uav_dependencies(expected_dependencies, current_dependencies,
                                         expected_versions, current_versions)
    return DependencyCheck(check.allowed, check.reasons, check.affected_uav_ids, check.goal_ids, check.digest)


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
    def remaining_task_contract(self) -> RemainingTaskContract:
        # The single shared computation; local repair never re-derives amounts.
        # shared_target_ids stays empty here: a locally tracked target is not
        # automatically fleet-shared evidence. Transfer decisions belong to the
        # handoff consumer with its own trusted shared-evidence source.
        return build_remaining_task_contract(self.goals, self.original.planner_output,
            self.completed_step_ids, self.evidence, current_step_id=self.current_step_id,
            current_step_started=self.current_step_started, home_name=self.home_name,
            consumer="LOCAL_REPAIR")

    @property
    def assessment(self):
        return self.remaining_task_contract.assessment

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
                "remaining_task_contract": self.remaining_task_contract.to_dict(),
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


@dataclass(frozen=True, slots=True)
class JointRepairRequest:
    """One bounded multi-UAV repair request; editable scope is trusted-only.

    The scope and readonly routes are decided by trusted code from dependency
    snapshots, route conflicts and resource ownership. The model may propose
    suffixes for scope UAVs only; every other UAV is read-only geometry.
    """
    request_id: str
    episode_id: str
    faulted_uav_id: str
    scope: tuple[str, ...]
    contexts: Mapping[str, LocalRepairContextV3]
    readonly_uav_routes: Mapping[str, tuple[tuple[float, float, float], ...]] = field(default_factory=dict)

    def __post_init__(self):
        validate_routing_id(self.request_id, "request_id")
        validate_routing_id(self.episode_id, "episode_id")
        validate_uav_id(self.faulted_uav_id)
        scope = tuple(sorted(set(self.scope)))
        if self.faulted_uav_id not in scope:
            raise ValueError("joint scope must contain the faulted UAV")
        object.__setattr__(self, "scope", scope)
        contexts = dict(self.contexts)
        if set(contexts) != set(scope) or not scope:
            raise LocalRepairError("JOINT_SCOPE_INVALID", "joint request needs one context per scope UAV")
        for uav, context in contexts.items():
            if not isinstance(context, LocalRepairContextV3) or context.original.planner_output.uav_id != uav:
                raise LocalRepairError("JOINT_SCOPE_INVALID", "joint context routing mismatch")
        object.__setattr__(self, "contexts", _freeze(contexts))
        routes = {}
        for uav, route in dict(self.readonly_uav_routes).items():
            validate_uav_id(uav)
            if uav in scope:
                raise ValueError("readonly routes must not name an editable scope UAV")
            points = tuple(tuple(float(value) for value in point) for point in route)
            if not points or not all(len(point) == 3 for point in points):
                raise ValueError("readonly route must be a nonempty 3D polyline")
            routes[uav] = points
        object.__setattr__(self, "readonly_uav_routes", _freeze(routes))

    @property
    def digest(self):
        return _digest({"request_id": self.request_id, "episode_id": self.episode_id,
                        "faulted_uav_id": self.faulted_uav_id, "scope": list(self.scope),
                        "contexts": {uav: context.digest for uav, context in self.contexts.items()},
                        "readonly": {uav: [list(p) for p in route] for uav, route in self.readonly_uav_routes.items()}})

    def to_dict(self):
        return {"request_id": self.request_id, "episode_id": self.episode_id,
                "faulted_uav_id": self.faulted_uav_id, "editable_uavs": list(self.scope),
                "readonly_uavs": sorted(self.readonly_uav_routes),
                "readonly_routes": {uav: [list(point) for point in route]
                                    for uav, route in self.readonly_uav_routes.items()},
                "per_uav_trusted_context": {uav: context.to_dict() for uav, context in self.contexts.items()},
                "original_suffixes": {uav: [step.to_dict() for step in context.original.planner_output.steps[len(context.completed_step_ids):]]
                                      for uav, context in self.contexts.items()}}


def build_joint_repair_json_schema(request: JointRepairRequest) -> dict[str, object]:
    """One authorized suffix grammar per editable UAV; readonly UAVs have no key."""
    properties = {uav: build_local_repair_json_schema(request.contexts[uav]) for uav in request.scope}
    return {"type": "object", "properties": properties, "required": list(request.scope),
            "additionalProperties": False,
            "properties_order_note": "keys are exactly the editable UAV IDs"}


def parse_joint_repair_drafts(value, request: JointRepairRequest) -> Mapping[str, LocalRepairDraftV3]:
    """Strictly bind the model response to the trusted repair scope."""
    if not isinstance(value, Mapping):
        raise LocalRepairError("INVALID_MODEL_RESPONSE", "joint response must be a JSON object keyed by UAV")
    if set(value) != set(request.scope):
        raise LocalRepairError("JOINT_SCOPE_MUTATED",
                               "model returned edits outside the authorized repair scope",
                               affected_uav_ids=tuple(sorted(set(value) - set(request.scope))) or request.scope)
    return {uav: LocalRepairDraftV3.from_dict(value[uav]) for uav in request.scope}


def validate_joint_repair(drafts: Mapping[str, LocalRepairDraftV3], request: JointRepairRequest,
                          world_contexts: Mapping[str, PlannerWorldContext], *,
                          dependency_versions: Mapping[str, Mapping[str, int]],
                          external_dependencies: Mapping[str, Sequence[ExternalDependencySnapshot]],
                          now_wall_s: float, plan_validator=None) -> Mapping[str, LocalRepairCandidate]:
    """Per-UAV validation of a joint proposal; pure preparation only.

    Each editable UAV passes the full single-repair pipeline against its own
    frozen context. Cross-UAV space/resource freshness remains the owner's
    live compare-and-commit guard; this never grants execution permission.
    """
    if set(drafts) != set(request.scope):
        raise LocalRepairError("JOINT_SCOPE_MUTATED", "draft set does not match the repair scope")
    candidates = {}
    for uav in request.scope:
        context = request.contexts[uav]
        candidates[uav] = validate_local_repair(drafts[uav], context, world_contexts[uav],
            dependency_versions=dependency_versions[uav], external_dependencies=external_dependencies[uav],
            now_wall_s=now_wall_s, plan_validator=plan_validator)
    return candidates


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
    confirmed = set(context.remaining_task_contract.confirmed_goal_ids)
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
    deps = check_cross_uav_dependencies(context.external_dependencies, external_dependencies,
                                        context.dependency_versions, dependency_versions)
    if deps.verdict is DependencyVerdict.INVALID:
        raise LocalRepairError("DEPENDENCY_INVALID", ";".join(deps.reasons), affected_uav_ids=deps.affected_uav_ids, goal_ids=deps.goal_ids)
    if deps.verdict is DependencyVerdict.COORDINATION_REQUIRED:
        raise LocalRepairError("COORDINATION_REQUIRED", ";".join(deps.reasons), affected_uav_ids=deps.affected_uav_ids, goal_ids=deps.goal_ids)
    contract = context.remaining_task_contract
    if not contract.supported:
        raise LocalRepairError("EVIDENCE_INSUFFICIENT", ";".join(contract.assessment.reasons) or "remaining goals lack trustworthy completion evidence",
                               affected_uav_ids=(original.uav_id,), goal_ids=contract.insufficient_goal_ids)
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
