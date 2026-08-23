"""Immutable experimental mission-program graph types.

The graph is deliberately a control-flow representation, not a flight
planner.  Nodes contain already validated :class:`~skills.plan.TaskStep`
values and edges react to a small closed event vocabulary.  Geometry and
controller ownership remain outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from skills.plan import TaskPlan, TaskStep
from skills.types import SkillName


class MissionProgramError(ValueError):
    """Raised when a mission graph violates its structural contract."""


class ProgramEvent(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    TARGET_CONFIRMED = "TARGET_CONFIRMED"
    TARGET_LOST = "TARGET_LOST"
    PATH_BLOCKED = "PATH_BLOCKED"
    TIMEOUT = "TIMEOUT"


class ProgramActionOp(str, Enum):
    HOLD = "HOLD"
    RESUME = "RESUME"
    REPLAN_CURRENT_ROUTE = "REPLAN_CURRENT_ROUTE"
    CANCEL_AND_LAND = "CANCEL_AND_LAND"


def _positive_version(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MissionProgramError(f"{name} must be a positive integer")
    return value


def _bounded_text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MissionProgramError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise MissionProgramError(f"{name} must contain at most {maximum} characters")
    return normalized


def _json_value(value: object, name: str, *, depth: int = 0) -> object:
    if depth > 16:
        raise MissionProgramError(f"{name} exceeds the maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return deepcopy(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not isfinite(normalized):
            raise MissionProgramError(f"{name} contains a non-finite number")
        return normalized
    if isinstance(value, Mapping):
        if len(value) > 64 or any(not isinstance(key, str) for key in value):
            raise MissionProgramError(f"{name} must be a bounded string-key mapping")
        return {
            key: _json_value(item, f"{name}.{key}", depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 64:
            raise MissionProgramError(f"{name} exceeds the maximum item count")
        return tuple(
            _json_value(item, f"{name}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise MissionProgramError(f"{name} contains unsupported data")


@dataclass(frozen=True, slots=True)
class SpatialEntity:
    entity_id: str
    kind: str
    value: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entity_id", validate_routing_id(self.entity_id, "entity_id")
        )
        object.__setattr__(self, "kind", _bounded_text(self.kind, "kind", maximum=64))
        normalized = _json_value(self.value, "value")
        if not isinstance(normalized, dict):
            raise MissionProgramError("value must be a mapping")
        object.__setattr__(self, "value", normalized)

    def to_dict(self) -> dict[str, object]:
        return {"entity_id": self.entity_id, "kind": self.kind, "value": deepcopy(dict(self.value))}


@dataclass(frozen=True, slots=True)
class MissionNode:
    node_id: str
    step: TaskStep

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", validate_routing_id(self.node_id, "node_id"))
        if not isinstance(self.step, TaskStep):
            raise TypeError("step must be a TaskStep")
        if self.step.step_id != self.node_id:
            raise MissionProgramError("node_id must equal TaskStep.step_id")

    def to_dict(self) -> dict[str, object]:
        return {"node_id": self.node_id, "step": self.step.to_dict()}


@dataclass(frozen=True, slots=True)
class MissionEdge:
    source_node_id: str
    target_node_id: str
    on: ProgramEvent

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_node_id",
            validate_routing_id(self.source_node_id, "source_node_id"),
        )
        object.__setattr__(
            self,
            "target_node_id",
            validate_routing_id(self.target_node_id, "target_node_id"),
        )
        if not isinstance(self.on, ProgramEvent):
            try:
                object.__setattr__(self, "on", ProgramEvent(self.on))
            except (TypeError, ValueError):
                raise MissionProgramError("on must be a supported ProgramEvent") from None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "on": self.on.value,
        }


@dataclass(frozen=True, slots=True)
class ProgramAction:
    op: ProgramActionOp
    planner: str | None = None
    allow_model_waypoints: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.op, ProgramActionOp):
            try:
                object.__setattr__(self, "op", ProgramActionOp(self.op))
            except (TypeError, ValueError):
                raise MissionProgramError("op must be a supported ProgramActionOp") from None
        if self.planner is not None:
            object.__setattr__(
                self, "planner", _bounded_text(self.planner, "planner", maximum=64)
            )
        if self.allow_model_waypoints is not None and not isinstance(
            self.allow_model_waypoints, bool
        ):
            raise TypeError("allow_model_waypoints must be bool or None")
        if self.op is not ProgramActionOp.REPLAN_CURRENT_ROUTE and (
            self.planner is not None or self.allow_model_waypoints is not None
        ):
            raise MissionProgramError("planner fields are only valid for REPLAN_CURRENT_ROUTE")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"op": self.op.value}
        if self.planner is not None:
            result["planner"] = self.planner
        if self.allow_model_waypoints is not None:
            result["allow_model_waypoints"] = self.allow_model_waypoints
        return result


@dataclass(frozen=True, slots=True)
class ProgramEventHandler:
    on: ProgramEvent
    actions: tuple[ProgramAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.on, ProgramEvent):
            try:
                object.__setattr__(self, "on", ProgramEvent(self.on))
            except (TypeError, ValueError):
                raise MissionProgramError("on must be a supported ProgramEvent") from None
        actions = tuple(self.actions)
        if not actions or len(actions) > 8 or any(
            not isinstance(action, ProgramAction) for action in actions
        ):
            raise MissionProgramError("actions must contain between 1 and 8 ProgramAction values")
        object.__setattr__(self, "actions", actions)

    def to_dict(self) -> dict[str, object]:
        return {"on": self.on.value, "actions": [action.to_dict() for action in self.actions]}


@dataclass(frozen=True, slots=True)
class MissionProgram:
    mission_id: str
    uav_id: str
    plan_version: int
    entry_node_id: str
    spatial_entities: tuple[SpatialEntity, ...]
    nodes: tuple[MissionNode, ...]
    edges: tuple[MissionEdge, ...]
    event_handlers: tuple[ProgramEventHandler, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise MissionProgramError("MissionProgram.schema_version must equal 1")
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(
            self, "plan_version", _positive_version(self.plan_version, "plan_version")
        )
        object.__setattr__(
            self,
            "entry_node_id",
            validate_routing_id(self.entry_node_id, "entry_node_id"),
        )
        entities, nodes, edges, handlers = (
            tuple(self.spatial_entities),
            tuple(self.nodes),
            tuple(self.edges),
            tuple(self.event_handlers),
        )
        if len(entities) > 128 or any(not isinstance(item, SpatialEntity) for item in entities):
            raise MissionProgramError("spatial_entities must be bounded SpatialEntity values")
        if not nodes or len(nodes) > 100 or any(not isinstance(item, MissionNode) for item in nodes):
            raise MissionProgramError("nodes must contain between 1 and 100 MissionNode values")
        if len(edges) > 256 or any(not isinstance(item, MissionEdge) for item in edges):
            raise MissionProgramError("edges must be bounded MissionEdge values")
        if len(handlers) > len(ProgramEvent) or any(
            not isinstance(item, ProgramEventHandler) for item in handlers
        ):
            raise MissionProgramError("event_handlers contains invalid values")
        entity_ids = [item.entity_id for item in entities]
        node_ids = [item.node_id for item in nodes]
        if len(entity_ids) != len(set(entity_ids)):
            raise MissionProgramError("spatial entity IDs must be unique")
        if len(node_ids) != len(set(node_ids)):
            raise MissionProgramError("node IDs must be unique")
        if self.entry_node_id not in node_ids:
            raise MissionProgramError("entry_node_id must reference a node")
        known = set(node_ids)
        edge_keys: set[tuple[str, ProgramEvent]] = set()
        for edge in edges:
            if edge.source_node_id not in known or edge.target_node_id not in known:
                raise MissionProgramError("every edge endpoint must reference a node")
            key = (edge.source_node_id, edge.on)
            if key in edge_keys:
                raise MissionProgramError("a node may have at most one edge per event")
            edge_keys.add(key)
        events = [item.on for item in handlers]
        if len(events) != len(set(events)):
            raise MissionProgramError("event handler events must be unique")
        object.__setattr__(self, "spatial_entities", entities)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "event_handlers", handlers)

    def node(self, node_id: str) -> MissionNode:
        normalized = validate_routing_id(node_id, "node_id")
        for node in self.nodes:
            if node.node_id == normalized:
                return node
        raise MissionProgramError(f"unknown node_id: {normalized}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "entry_node_id": self.entry_node_id,
            "spatial_entities": [item.to_dict() for item in self.spatial_entities],
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "event_handlers": [item.to_dict() for item in self.event_handlers],
        }


def linear_plan_to_mission_program(
    plan: TaskPlan,
    *,
    event_handlers: Sequence[ProgramEventHandler] = (),
) -> MissionProgram:
    """Adapt an existing linear plan without changing any step or output ID."""

    if not isinstance(plan, TaskPlan):
        raise TypeError("plan must be a TaskPlan")
    nodes = tuple(MissionNode(step.step_id, step) for step in plan.steps)
    edges: list[MissionEdge] = []
    for left, right in zip(nodes, nodes[1:]):
        edges.append(
            MissionEdge(left.node_id, right.node_id, ProgramEvent.SUCCESS)
        )
        if (
            left.step.skill is SkillName.SEARCH
            and right.step.skill is SkillName.SEARCH
        ):
            # A linear TaskPlan treats SEARCH_EXHAUSTED/TIMEOUT as exhaustion
            # of one bounded fallback region, not as a mission failure.  The
            # graph event vocabulary intentionally folds both outcomes into
            # TIMEOUT, so the adapter preserves that historical behavior
            # without making every FAILURE (for example INVALID_GOAL) retry.
            edges.append(
                MissionEdge(left.node_id, right.node_id, ProgramEvent.TIMEOUT)
            )
    return MissionProgram(
        mission_id=plan.mission_id,
        uav_id=plan.uav_id,
        plan_version=plan.plan_version,
        entry_node_id=nodes[0].node_id,
        spatial_entities=(),
        nodes=nodes,
        edges=tuple(edges),
        event_handlers=tuple(event_handlers),
    )


__all__ = [
    "MissionEdge",
    "MissionNode",
    "MissionProgram",
    "MissionProgramError",
    "ProgramAction",
    "ProgramActionOp",
    "ProgramEvent",
    "ProgramEventHandler",
    "SpatialEntity",
    "linear_plan_to_mission_program",
]
