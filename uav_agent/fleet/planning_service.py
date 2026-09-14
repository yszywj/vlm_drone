"""Program-directed global planning, independent of perception and execution.

The service owns the interpretation and Fleet assignment calls.  The caller
owns trusted configuration/state, local compilation, and execution.  Recoverable
semantic findings are returned unchanged; this layer never invents missing
goals or silently substitutes a scripted plan.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from common.ids import validate_routing_id
from fleet.llm_planner_v2 import LLMFleetPlannerV2
from fleet.llm_task_interpreter import LLMFleetTaskInterpreter
from fleet.request_builder import build_fleet_mission_request_v2
from fleet.schemas_v2 import validate_fleet_mission_plan_v2
from fleet.task_spec import FleetTaskSpecV1
from fleet.types_v2 import FleetMissionPlanV2, FleetMissionRequestV2
from models.adapter_registry import ModelCallRole

if TYPE_CHECKING:
    from configs.schema import AppConfig
    from models.model_client_factory import ModelClientFactory


@dataclass(frozen=True, slots=True)
class GlobalPlanningResult:
    """Validated global assignment and its audit, not an executable flight plan."""

    task_spec: FleetTaskSpecV1
    request: FleetMissionRequestV2
    plan: FleetMissionPlanV2
    interpreter_diagnostics: Mapping[str, object] | None
    interpreter_proposals: tuple[Mapping[str, object], ...]
    fleet_planner_proposals: tuple[Mapping[str, object], ...]
    semantic_findings: tuple[Mapping[str, object], ...]
    fleet_planner_source: str
    fleet_planner_diagnostics: Mapping[str, object] | None = None

    @property
    def uncovered_goal_ids(self) -> tuple[str, ...]:
        assigned = {
            goal_id
            for assignment in self.plan.assignments
            for goal_id in assignment.goal_ids
        }
        # Declaring a goal unassigned accounts for it in the schema, but does
        # not fulfil it.  Include both omissions and declared unassigned goals.
        return tuple(
            goal_id for goal_id in self.task_spec.all_goal_ids if goal_id not in assigned
        )

    @property
    def assignment_complete(self) -> bool:
        """Coverage and existing semantic checks, relative to interpreted goals."""

        return not (
            self.uncovered_goal_ids or self.semantic_findings or self.task_spec.ambiguities
        )

    def to_dict(self) -> dict[str, object]:
        """Expose bounded planning artifacts without implying execution readiness."""

        return {
            "task_spec": self.task_spec.to_dict(),
            "request": self.request.to_dict(),
            "plan": self.plan.to_dict(),
            "interpreter_diagnostics": deepcopy(self.interpreter_diagnostics),
            "interpreter_proposals": deepcopy(list(self.interpreter_proposals)),
            "fleet_planner_diagnostics": deepcopy(self.fleet_planner_diagnostics),
            "fleet_planner_proposals": deepcopy(list(self.fleet_planner_proposals)),
            "semantic_findings": deepcopy(list(self.semantic_findings)),
            "fleet_planner_source": self.fleet_planner_source,
            "completeness": {
                "scope": "interpreted_task_spec_assignment",
                "all_goals_assigned": not self.uncovered_goal_ids,
                "uncovered_goal_ids": list(self.uncovered_goal_ids),
                "semantic_findings_clear": not self.semantic_findings,
                "interpretation_ambiguities_clear": not self.task_spec.ambiguities,
                "assignment_complete": self.assignment_complete,
                "source_intent_verified": False,
                "local_plans_validated": False,
            },
        }


def _mission_alias_catalogs(config: "AppConfig") -> tuple[dict[str, str], dict[str, str]]:
    uavs: dict[str, str] = {}
    for uav in config.uavs:
        uavs[uav.id] = uav.id
        if uav.display_name:
            uavs[uav.display_name] = uav.id
    targets: dict[str, str] = {}
    for target in config.targets:
        targets[target.id] = target.id
        if target.semantic_alias:
            targets[target.semantic_alias] = target.id
    return uavs, targets


def _diagnostics(planner: object) -> Mapping[str, object] | None:
    value = getattr(planner, "last_diagnostics", None)
    return None if value is None else value.to_dict()


def _proposals(planner: object) -> tuple[Mapping[str, object], ...]:
    return tuple(getattr(planner, "model_proposals", ()))


def _clear_planning_audit(
    audit: MutableMapping[str, object], *, include_interpreter: bool
) -> None:
    """Discard old stage artifacts while retaining caller-owned audit streams."""

    keys = {"request_v2", "plan_v2", "fleet_semantic_findings"}
    if include_interpreter:
        keys.add("task_spec")
    for key in tuple(audit):
        if (
            key in keys
            or key.startswith("fleet_planner_")
            or (include_interpreter and key.startswith("interpreter_"))
        ):
            del audit[key]


class FleetPlanningService:
    """Route each planning phase to its configured model/LoRA role.

    Factories are optional test/integration seams, resolved at construction time.
    Each invocation creates fresh role clients and planners, so no mutable model
    selection or repair history is shared between missions.
    """

    def __init__(
        self,
        config: "AppConfig",
        client_factory: "ModelClientFactory",
        *,
        interpreter_max_tokens: int = 6144,
        fleet_max_tokens: int = 2048,
        interpreter_factory: Callable[..., LLMFleetTaskInterpreter] | None = None,
        fleet_planner_factory: Callable[..., LLMFleetPlannerV2] | None = None,
    ) -> None:
        if not callable(getattr(client_factory, "for_role", None)):
            raise TypeError("client_factory must provide for_role()")
        for name, value in (
            ("interpreter_max_tokens", interpreter_max_tokens),
            ("fleet_max_tokens", fleet_max_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 8192:
                raise ValueError(f"{name} must be within [256, 8192]")
        self._config = config
        self._clients = client_factory
        self._interpreter_max_tokens = interpreter_max_tokens
        self._fleet_max_tokens = fleet_max_tokens
        self._interpreter_factory = interpreter_factory or LLMFleetTaskInterpreter
        self._fleet_planner_factory = fleet_planner_factory or LLMFleetPlannerV2

    def plan(
        self,
        instruction: str,
        *,
        fleet_mission_id: str,
        audit_context: MutableMapping[str, object] | None = None,
    ) -> GlobalPlanningResult:
        """Interpret an original instruction and assign its goals to the fleet.

        Audit keys match the mission entry point. They are written in ``finally``
        after each model phase, including failures, before the original exception
        propagates.  Caller-owned model-call logging remains in the client factory.
        """

        audit = audit_context if audit_context is not None else {}
        _clear_planning_audit(audit, include_interpreter=True)
        audit["planning_stage"] = "mission_interpretation"
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction must not be empty")
        if len(instruction) > 8192:
            raise ValueError("instruction must contain at most 8192 characters")
        if "\x00" in instruction:
            raise ValueError("instruction must not contain NUL characters")
        fleet_mission_id = validate_routing_id(fleet_mission_id, "fleet_mission_id")
        audit.update({
            "fleet_mission_id": fleet_mission_id,
            "source_text": instruction,
            "interpreter_proposals": (),
            "interpreter_diagnostics": None,
        })
        uav_aliases, target_aliases = _mission_alias_catalogs(self._config)
        interpreter = self._interpreter_factory(
            self._clients.for_role(
                ModelCallRole.MISSION_INTERPRETATION,
                fleet_mission_id=fleet_mission_id,
            ),
            uav_alias_catalog=uav_aliases,
            target_alias_catalog=target_aliases,
            repair_budget=1,
            max_tokens=self._interpreter_max_tokens,
        )
        try:
            task_spec = interpreter.interpret(instruction)
        finally:
            audit["interpreter_proposals"] = _proposals(interpreter)
            audit["interpreter_diagnostics"] = _diagnostics(interpreter)
        audit["task_spec"] = task_spec
        audit["planning_stage"] = "fleet_assignment"
        request = build_fleet_mission_request_v2(
            self._config, task_spec, fleet_mission_id=fleet_mission_id
        )
        result = self._plan_request(request, audit_context=audit)
        return replace(
            result,
            interpreter_diagnostics=_diagnostics(interpreter),
            interpreter_proposals=_proposals(interpreter),
        )

    def plan_request(
        self,
        request: FleetMissionRequestV2,
        *,
        replan: bool = False,
        assignment_id: str | None = None,
        uav_id: str | None = None,
        audit_context: MutableMapping[str, object] | None = None,
    ) -> GlobalPlanningResult:
        """Plan trusted state directly, without reinterpreting the user text.

        Runtime orchestration supplies the versioned request and trusted evidence;
        only the program chooses whether to use the FLEET_REPLAN adapter role.
        """

        audit = audit_context if audit_context is not None else {}
        _clear_planning_audit(audit, include_interpreter=True)
        return self._plan_request(
            request,
            replan=replan,
            assignment_id=assignment_id,
            uav_id=uav_id,
            audit_context=audit,
        )

    def _plan_request(
        self,
        request: FleetMissionRequestV2,
        *,
        replan: bool = False,
        assignment_id: str | None = None,
        uav_id: str | None = None,
        audit_context: MutableMapping[str, object],
    ) -> GlobalPlanningResult:
        # The initial pipeline uses this core directly to keep its fresh
        # Interpreter audit. Public plan_request starts a separate invocation.
        audit = audit_context
        _clear_planning_audit(audit, include_interpreter=False)
        audit["planning_stage"] = "fleet_assignment"
        if not isinstance(request, FleetMissionRequestV2):
            raise TypeError("request must be a FleetMissionRequestV2")
        if not isinstance(replan, bool):
            raise TypeError("replan must be a bool")
        routing = {"fleet_mission_id": request.fleet_mission_id}
        for name, value in (("assignment_id", assignment_id), ("uav_id", uav_id)):
            if value is not None:
                routing[name] = validate_routing_id(value, name)
        audit.update({
            "fleet_mission_id": request.fleet_mission_id,
            "source_text": request.task_spec.source_text,
            "task_spec": request.task_spec,
            "request_v2": request,
            "fleet_planner_proposals": (),
            "fleet_planner_diagnostics": None,
        })
        planner = self._fleet_planner_factory(
            self._clients.for_role(
                ModelCallRole.FLEET_REPLAN if replan else ModelCallRole.FLEET_PLAN,
                **routing,
            ),
            repair_budget=2,
            max_tokens=self._fleet_max_tokens,
        )
        try:
            plan = validate_fleet_mission_plan_v2(planner.plan(request), request)
        finally:
            audit["fleet_planner_proposals"] = _proposals(planner)
            audit["fleet_planner_diagnostics"] = _diagnostics(planner)
        findings = tuple(
            {
                "code": item.code,
                "message": item.message[:2048],
                "constraint_id": item.constraint_id,
                "goal_id": item.goal_id,
                "assignment_id": item.assignment_id,
            }
            for item in plan.semantic_findings(request)
        )
        audit.update({
            "plan_v2": plan,
            "fleet_semantic_findings": findings,
            "planning_stage": "completed",
        })
        return GlobalPlanningResult(
            task_spec=request.task_spec,
            request=request,
            plan=plan,
            interpreter_diagnostics=None,
            interpreter_proposals=(),
            fleet_planner_proposals=_proposals(planner),
            semantic_findings=findings,
            fleet_planner_source=getattr(planner, "source", "fleet_llm_v2"),
            fleet_planner_diagnostics=_diagnostics(planner),
        )


__all__ = ["FleetPlanningService", "GlobalPlanningResult"]
