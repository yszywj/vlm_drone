"""Synchronous publication coordinator for the explicit classical baseline.

The coordinator implements the small protocol consumed by
``ObstacleRouteReplanRuntime``.  It intentionally has no model worker and is
not referenced by ``ObstacleRevisionCoordinator``: selecting this baseline is
an experiment-level choice, never an implicit fallback from Qwen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from numbers import Real

from agents.obstacle_revision_coordinator import (
    ObstacleRevisionCoordinatorRecord,
    ObstacleRevisionCoordinatorSnapshot,
    ObstacleRevisionCoordinatorState,
)
from common.ids import generate_routing_id, validate_uav_id
from planner.classical_route_planner import (
    ClassicalNoFeasibleRoute,
    ClassicalRoutePlanner,
    ClassicalRouteSolution,
)
from planner.obstacle_revision import (
    ObstacleAwareRevisionRequest,
    ObstacleReplacementStep,
    ObstacleRouteRevisionDraft,
)
from planner.route_critic import RouteCritique, RouteValidationContext
from planner.route_types import (
    AvoidanceStrategy,
    AvoidanceStrategyType,
    RouteDraft,
)
from planner.spatial_resolver import FramePose
from runtime.collision_supervisor import CollisionSupervisor
from runtime.events import MissionEvent
from runtime.route_registry import RouteRegistry
from runtime.safety_supervisor import SafetyAction, SafetyDecision
from skills.plan import TaskPlan


_PLANNER_KIND = "CLASSICAL_VISIBILITY_GRAPH_V1"
_ROUTE_SUBSTITUTED_SKILLS = frozenset({"GOTO", "FOLLOW_ROUTE"})
_REUSABLE_SUFFIX_SKILLS = frozenset(
    {"GOTO", "HOVER", "SEARCH", "TRACK", "LAND"}
)


class ClassicalObstacleRevisionCoordinator:
    """Plan once, run STRICT checks, and publish through trusted runtime APIs."""

    def __init__(
        self,
        *,
        uav_id: str,
        planner: ClassicalRoutePlanner,
        route_registry: RouteRegistry,
        collision_supervisor: CollisionSupervisor,
        skill_manager: object,
        safety_preflight: Callable[[TaskPlan], SafetyDecision],
        event_sink: Callable[[MissionEvent], object] | None = None,
    ) -> None:
        self._uav_id = validate_uav_id(uav_id)
        if not isinstance(planner, ClassicalRoutePlanner):
            raise TypeError("planner must be ClassicalRoutePlanner")
        if not isinstance(route_registry, RouteRegistry):
            raise TypeError("route_registry must be RouteRegistry")
        if not isinstance(collision_supervisor, CollisionSupervisor):
            raise TypeError("collision_supervisor must be CollisionSupervisor")
        for method in ("replace_interrupted_step_and_suffix", "cancel_task"):
            if not callable(getattr(skill_manager, method, None)):
                raise TypeError(f"skill_manager must provide {method}")
        if not callable(safety_preflight):
            raise TypeError("safety_preflight must be callable")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self._planner = planner
        self._registry = route_registry
        self._supervisor = collision_supervisor
        self._manager = skill_manager
        self._safety_preflight = safety_preflight
        self._event_sink = event_sink
        self._state = ObstacleRevisionCoordinatorState.IDLE
        self._request_id: str | None = None
        self._accepted_route_id: str | None = None
        self._error_code: str | None = None
        self._frame_snapshot: FramePose | None = None
        self._records: list[ObstacleRevisionCoordinatorRecord] = []
        self._round_index = 0
        self._next_proposal_index = 0
        self._current_round_record_index: int | None = None

    @property
    def records(self) -> tuple[ObstacleRevisionCoordinatorRecord, ...]:
        return tuple(self._records)

    @property
    def history_dict(self) -> dict[str, object]:
        rounds: list[dict[str, object]] = []
        by_round: dict[int, list[ObstacleRevisionCoordinatorRecord]] = {}
        for record in self._records:
            by_round.setdefault(record.round_index, []).append(record)
        for round_index in sorted(by_round):
            record = by_round[round_index][-1]
            rounds.append(
                {
                    "round_index": round_index,
                    "planner_kind": _PLANNER_KIND,
                    "state": record.outcome,
                    "proposal": record.proposal,
                    "critique": record.critique,
                    "error_code": record.error_code,
                }
            )
        return {
            "state": self._state.value,
            "round_index": self._round_index,
            "planner_kind": _PLANNER_KIND,
            "rounds": rounds,
        }

    def begin(
        self,
        request: ObstacleAwareRevisionRequest,
        *,
        validation_context: RouteValidationContext,
        frame_snapshot: FramePose,
        compile_replacement: Callable[[ObstacleRouteRevisionDraft], TaskPlan],
        timestamp_s: float,
    ) -> ObstacleRevisionCoordinatorSnapshot:
        """Synchronously plan and atomically publish one classical detour."""

        if self._state is not ObstacleRevisionCoordinatorState.IDLE:
            raise RuntimeError("classical obstacle coordinator is already active")
        if not isinstance(request, ObstacleAwareRevisionRequest):
            raise TypeError("request must be ObstacleAwareRevisionRequest")
        if request.uav_id != self._uav_id:
            raise ValueError("request.uav_id does not match coordinator")
        if not isinstance(validation_context, RouteValidationContext):
            raise TypeError("validation_context must be RouteValidationContext")
        if validation_context.constraints != request.route_constraints:
            raise ValueError("request and validation route constraints must match")
        if not isinstance(frame_snapshot, FramePose):
            raise TypeError("frame_snapshot must be FramePose")
        if not callable(compile_replacement):
            raise TypeError("compile_replacement must be callable")
        if not bool(getattr(self._manager, "is_supervisory_paused", False)):
            raise RuntimeError("classical route revision requires supervisory pause")
        timestamp = _timestamp(timestamp_s)
        self._request_id = generate_routing_id("request_classical_route")
        self._accepted_route_id = None
        self._error_code = None
        self._frame_snapshot = frame_snapshot
        self._publish_events(self._supervisor.begin_replanning().events)

        try:
            result = self._planner.plan(
                route_id=request.route_id,
                rejoin_target=request.active_corridor_rejoin_target,
                grounded_obstacles=(request.grounded_obstacle_geometry,),
                validation_context=validation_context,
            )
        except Exception:
            # The baseline stays fail-closed even for an implementation/input
            # error after HOLD.  No Qwen request or alternate route is tried.
            self._publish_rejection_without_candidate(
                route_id=request.route_id,
                reason_code="CLASSICAL_PLANNER_ERROR",
                timestamp_s=timestamp,
            )
            return self.snapshot()

        self._publish_events(
            self._supervisor.route_proposed(
                request.route_id,
                timestamp_s=timestamp,
            ).events
        )
        if isinstance(result, ClassicalNoFeasibleRoute):
            self._publish_no_feasible(result, timestamp_s=timestamp)
            return self.snapshot()

        assert isinstance(result, ClassicalRouteSolution)
        try:
            proposal = _revision_draft(request, result.route)
        except Exception:
            self._finish_rejected_candidate(
                route=result.route,
                critique=result.critique,
                reason_code="CLASSICAL_TRUSTED_SUFFIX_UNSUPPORTED",
                timestamp_s=timestamp,
            )
            return self.snapshot()
        self._publish_solution(
            request=request,
            proposal=proposal,
            critique=result.critique,
            frame_snapshot=frame_snapshot,
            compile_replacement=compile_replacement,
            timestamp_s=timestamp,
        )
        return self.snapshot()

    def tick(self, *, timestamp_s: float) -> ObstacleRevisionCoordinatorSnapshot:
        # The classical planner is synchronous; accepting the timestamp keeps
        # this object substitutable for the non-blocking model coordinator.
        _timestamp(timestamp_s)
        return self.snapshot()

    def snapshot(self) -> ObstacleRevisionCoordinatorSnapshot:
        proposal_count = 0 if self._current_round_record_index is None else 1
        return ObstacleRevisionCoordinatorSnapshot(
            self._state,
            self._request_id,
            proposal_count,
            self._accepted_route_id,
            self._error_code,
        )

    def reset(self, *, preserve_records: bool = False) -> None:
        if not isinstance(preserve_records, bool):
            raise TypeError("preserve_records must be bool")
        if preserve_records:
            if self._current_round_record_index is not None:
                self._round_index += 1
        else:
            self._records.clear()
            self._round_index = 0
            self._next_proposal_index = 0
        self._state = ObstacleRevisionCoordinatorState.IDLE
        self._request_id = None
        self._accepted_route_id = None
        self._error_code = None
        self._frame_snapshot = None
        self._current_round_record_index = None

    def _publish_solution(
        self,
        *,
        request: ObstacleAwareRevisionRequest,
        proposal: ObstacleRouteRevisionDraft,
        critique: RouteCritique,
        frame_snapshot: FramePose,
        compile_replacement: Callable[[ObstacleRouteRevisionDraft], TaskPlan],
        timestamp_s: float,
    ) -> None:
        registered_route_id: str | None = None
        try:
            plan = compile_replacement(proposal)
            if not isinstance(plan, TaskPlan):
                raise TypeError("compile_replacement must return TaskPlan")
            if (
                plan.mission_id != proposal.mission_id
                or plan.uav_id != proposal.uav_id
                or plan.plan_version != proposal.new_plan_version
            ):
                raise ValueError("compiled classical replacement routing mismatch")
            safety = self._safety_preflight(plan)
            if (
                not isinstance(safety, SafetyDecision)
                or safety.action is not SafetyAction.CONTINUE
            ):
                raise ValueError("classical replacement failed safety preflight")
            record = self._registry.register(
                proposal.route_draft,
                frame_snapshot=frame_snapshot,
                raw_proposal=proposal.to_dict(),
                plan_version=proposal.new_plan_version,
                proposal_timestamp_s=request.frames[-1].ref.timestamp_s,
            )
            self._registry.record_critique(record.route_id, critique)
            registered_route_id = record.route_id
            self._manager.replace_interrupted_step_and_suffix(plan)
            self._publish_events(
                self._supervisor.route_accepted(
                    validation_mode="strict",
                    required_checks_passed=True,
                    timestamp_s=timestamp_s,
                ).events
            )
            self._supervisor.resume(required_checks_passed=True)
        except Exception:
            if registered_route_id is not None:
                try:
                    self._registry.transition(registered_route_id, "REJECTED")
                except Exception:
                    pass
            self._append_record(
                timestamp_s=timestamp_s,
                outcome="PUBLICATION_FAILED",
                proposal=proposal.to_dict(),
                critique=critique.to_dict(),
                error_code="CLASSICAL_ROUTE_PUBLICATION_FAILED",
            )
            self._state = ObstacleRevisionCoordinatorState.FAILED
            self._error_code = "CLASSICAL_ROUTE_PUBLICATION_FAILED"
            self._manager.cancel_task()
            return
        self._append_record(
            timestamp_s=timestamp_s,
            outcome="ACCEPTED",
            proposal=proposal.to_dict(),
            critique=critique.to_dict(),
            error_code=None,
        )
        self._state = ObstacleRevisionCoordinatorState.ACCEPTED
        self._accepted_route_id = proposal.route_draft.route_id

    def _publish_no_feasible(
        self,
        result: ClassicalNoFeasibleRoute,
        *,
        timestamp_s: float,
    ) -> None:
        reason = result.reason_code.value
        proposal: dict[str, object] = {
            "planner_kind": _PLANNER_KIND,
            "status": "NO_FEASIBLE_ROUTE",
            "reason_code": reason,
            "grounded_obstacle_ids": list(result.grounded_obstacle_ids),
        }
        if result.candidate_route is not None:
            proposal["route_draft"] = result.candidate_route.to_dict()
        self._publish_events(
            self._supervisor.route_rejected(
                reason_codes=(reason,),
                timestamp_s=timestamp_s,
            ).events
        )
        self._append_record(
            timestamp_s=timestamp_s,
            outcome="NO_FEASIBLE_ROUTE",
            proposal=proposal,
            critique=(
                None if result.critique is None else result.critique.to_dict()
            ),
            error_code=reason,
        )
        self._state = ObstacleRevisionCoordinatorState.EXHAUSTED
        self._error_code = reason
        self._manager.cancel_task()

    def _finish_rejected_candidate(
        self,
        *,
        route: RouteDraft,
        critique: RouteCritique,
        reason_code: str,
        timestamp_s: float,
    ) -> None:
        self._publish_events(
            self._supervisor.route_rejected(
                reason_codes=(reason_code,),
                timestamp_s=timestamp_s,
            ).events
        )
        self._append_record(
            timestamp_s=timestamp_s,
            outcome="NO_FEASIBLE_ROUTE",
            proposal={
                "planner_kind": _PLANNER_KIND,
                "status": "NO_FEASIBLE_ROUTE",
                "reason_code": reason_code,
                "route_draft": route.to_dict(),
            },
            critique=critique.to_dict(),
            error_code=reason_code,
        )
        self._state = ObstacleRevisionCoordinatorState.EXHAUSTED
        self._error_code = reason_code
        self._manager.cancel_task()

    def _publish_rejection_without_candidate(
        self,
        *,
        route_id: str,
        reason_code: str,
        timestamp_s: float,
    ) -> None:
        self._publish_events(
            self._supervisor.route_proposed(
                route_id,
                timestamp_s=timestamp_s,
            ).events
        )
        self._publish_events(
            self._supervisor.route_rejected(
                reason_codes=(reason_code,),
                timestamp_s=timestamp_s,
            ).events
        )
        self._append_record(
            timestamp_s=timestamp_s,
            outcome="PLANNER_ERROR",
            proposal={
                "planner_kind": _PLANNER_KIND,
                "status": "NO_FEASIBLE_ROUTE",
                "reason_code": reason_code,
            },
            critique=None,
            error_code=reason_code,
        )
        self._state = ObstacleRevisionCoordinatorState.FAILED
        self._error_code = reason_code
        self._manager.cancel_task()

    def _append_record(
        self,
        *,
        timestamp_s: float,
        outcome: str,
        proposal: dict[str, object] | None,
        critique: dict[str, object] | None,
        error_code: str | None,
    ) -> None:
        if self._request_id is None:
            raise RuntimeError("classical request_id is unavailable")
        record = ObstacleRevisionCoordinatorRecord(
            request_id=self._request_id,
            proposal_index=self._next_proposal_index,
            submitted_timestamp_s=timestamp_s,
            completed_timestamp_s=timestamp_s,
            outcome=outcome,
            proposal=proposal,
            critique=critique,
            error_code=error_code,
            round_index=self._round_index,
            frame_snapshot=self._frame_snapshot,
            # The classical planner always evaluates with STRICT; retain the
            # same result in the benchmark-only shadow channel as well.
            shadow_strict_critique=critique,
        )
        self._records.append(record)
        self._current_round_record_index = len(self._records) - 1
        self._next_proposal_index += 1

    def _publish_events(self, events: tuple[MissionEvent, ...]) -> None:
        if self._event_sink is None:
            return
        for event in events:
            self._event_sink(event)


def _revision_draft(
    request: ObstacleAwareRevisionRequest,
    route: RouteDraft,
) -> ObstacleRouteRevisionDraft:
    replacement_steps = _replacement_steps(request, route.route_id)
    return ObstacleRouteRevisionDraft(
        mission_id=request.mission_id,
        uav_id=request.uav_id,
        base_plan_version=request.base_plan_version,
        new_plan_version=request.new_plan_version,
        replace_from_step_id=request.replace_from_step_id,
        avoidance_strategy=AvoidanceStrategy(
            _classify_strategy(route),
            "original_goto_target",
            (_PLANNER_KIND,),
        ),
        route_draft=route,
        replacement_steps=replacement_steps,
    )


def _replacement_steps(
    request: ObstacleAwareRevisionRequest,
    route_id: str,
) -> tuple[ObstacleReplacementStep, ...]:
    steps: list[ObstacleReplacementStep] = [
        ObstacleReplacementStep(
            "classical_follow_route",
            request.uav_id,
            "FOLLOW_ROUTE",
            {"route_ref": route_id},
        )
    ]
    summaries: list[Mapping[str, object]] = []
    current_skill = request.current_step_summary.get("skill")
    if current_skill not in _ROUTE_SUBSTITUTED_SKILLS:
        summaries.append(request.current_step_summary)
    summaries.extend(request.remaining_plan_summary)
    for index, summary in enumerate(summaries, 1):
        skill = summary.get("skill")
        if not isinstance(skill, str) or skill not in _REUSABLE_SUFFIX_SKILLS:
            raise ValueError("trusted suffix contains an unsupported Skill")
        steps.append(
            ObstacleReplacementStep(
                f"classical_{index:02d}_{skill.lower()}",
                request.uav_id,
                skill,
                {},
            )
        )
    if (
        steps[-1].skill != "LAND"
        or sum(step.skill == "LAND" for step in steps) != 1
        or len(steps) > 16
    ):
        raise ValueError("trusted suffix must terminate with exactly one LAND")
    return tuple(steps)


def _classify_strategy(route: RouteDraft) -> AvoidanceStrategyType:
    first = route.waypoints[0].xyz_m
    goal = route.waypoints[-1].xyz_m
    direct_scale = 0.0 if abs(goal[0]) <= 1e-9 else first[0] / goal[0]
    expected_y = direct_scale * goal[1]
    expected_z = direct_scale * goal[2]
    lateral = first[1] - expected_y
    vertical = first[2] - expected_z
    if abs(lateral) >= abs(vertical) and abs(lateral) > 1e-6:
        return (
            AvoidanceStrategyType.BYPASS_LEFT
            if lateral > 0.0
            else AvoidanceStrategyType.BYPASS_RIGHT
        )
    if vertical > 1e-6:
        return AvoidanceStrategyType.BYPASS_ABOVE
    if first[0] < -1e-6:
        return AvoidanceStrategyType.BACKTRACK
    return AvoidanceStrategyType.HOLD_POSITION


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timestamp_s must be finite and non-negative")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError("timestamp_s must be finite and non-negative")
    return normalized


__all__ = ["ClassicalObstacleRevisionCoordinator"]
