"""Atomic, suffix-only patches for experimental mission programs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from planner.mission_program import (
    MissionEdge,
    MissionNode,
    MissionProgram,
    MissionProgramError,
)
from skills.plan import TaskStep


@dataclass(frozen=True, slots=True)
class ProgramPatch:
    mission_id: str
    uav_id: str
    base_plan_version: int
    new_plan_version: int
    replace_from_node_id: str
    replacement_nodes: tuple[MissionNode, ...]
    replacement_edges: tuple[MissionEdge, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        for name in ("base_plan_version", "new_plan_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MissionProgramError(f"{name} must be a positive integer")
        if self.new_plan_version != self.base_plan_version + 1:
            raise MissionProgramError("new_plan_version must equal base_plan_version + 1")
        object.__setattr__(
            self,
            "replace_from_node_id",
            validate_routing_id(self.replace_from_node_id, "replace_from_node_id"),
        )
        nodes, edges, reasons = (
            tuple(self.replacement_nodes),
            tuple(self.replacement_edges),
            tuple(self.reason_codes),
        )
        if not nodes or len(nodes) > 32 or any(not isinstance(node, MissionNode) for node in nodes):
            raise MissionProgramError("replacement_nodes must contain 1..32 MissionNode values")
        if nodes[0].node_id != self.replace_from_node_id:
            raise MissionProgramError("the first replacement node must match replace_from_node_id")
        if len(edges) > 64 or any(not isinstance(edge, MissionEdge) for edge in edges):
            raise MissionProgramError("replacement_edges contains invalid values")
        if not reasons or len(reasons) > 16:
            raise MissionProgramError("reason_codes must contain 1..16 values")
        normalized_reasons: list[str] = []
        for reason in reasons:
            if not isinstance(reason, str) or not reason or len(reason) > 64:
                raise MissionProgramError("reason_codes must be bounded non-empty strings")
            normalized_reasons.append(reason)
        if len(normalized_reasons) != len(set(normalized_reasons)):
            raise MissionProgramError("reason_codes must be unique")
        object.__setattr__(self, "replacement_nodes", nodes)
        object.__setattr__(self, "replacement_edges", edges)
        object.__setattr__(self, "reason_codes", tuple(normalized_reasons))

    def to_dict(self) -> dict[str, object]:
        """Return the strict model-transport representation."""

        replacement_nodes: list[dict[str, object]] = []
        for node in self.replacement_nodes:
            step = node.step.to_dict()
            recovery = step.pop("recovery", None)
            step_id = step.pop("id")
            skill = step.pop("skill")
            transport_step: dict[str, object] = {
                "id": step_id,
                "skill": skill,
                "args": deepcopy(step),
            }
            if recovery is not None:
                transport_step["recovery"] = deepcopy(recovery)
            replacement_nodes.append(
                {"node_id": node.node_id, "step": transport_step}
            )
        return {
            "schema_version": 1,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "base_plan_version": self.base_plan_version,
            "new_plan_version": self.new_plan_version,
            "replace_from_node_id": self.replace_from_node_id,
            "replacement_nodes": replacement_nodes,
            "replacement_edges": [
                edge.to_dict() for edge in self.replacement_edges
            ],
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProgramPatch:
        """Parse without repairing, filling, or normalizing model fields."""

        data = _strict_mapping(
            value,
            {
                "schema_version",
                "mission_id",
                "uav_id",
                "base_plan_version",
                "new_plan_version",
                "replace_from_node_id",
                "replacement_nodes",
                "replacement_edges",
                "reason_codes",
            },
            "ProgramPatch",
        )
        if data["schema_version"] != 1 or isinstance(
            data["schema_version"], bool
        ):
            raise MissionProgramError("ProgramPatch.schema_version must equal 1")
        raw_nodes = _sequence(data["replacement_nodes"], "replacement_nodes")
        nodes: list[MissionNode] = []
        for index, raw_node in enumerate(raw_nodes):
            node_data = _strict_mapping(
                raw_node,
                {"node_id", "step"},
                f"replacement_nodes[{index}]",
            )
            step_data = _strict_mapping_optional(
                node_data["step"],
                required={"id", "skill", "args"},
                optional={"recovery"},
                name=f"replacement_nodes[{index}].step",
            )
            args = step_data["args"]
            if not isinstance(args, Mapping) or any(
                not isinstance(key, str) for key in args
            ):
                raise MissionProgramError(
                    f"replacement_nodes[{index}].step.args must be an object"
                )
            step = TaskStep(
                step_data["id"],  # type: ignore[arg-type]
                step_data["skill"],  # type: ignore[arg-type]
                deepcopy(dict(args)),
                deepcopy(step_data.get("recovery")),  # type: ignore[arg-type]
            )
            nodes.append(MissionNode(node_data["node_id"], step))  # type: ignore[arg-type]

        raw_edges = _sequence(data["replacement_edges"], "replacement_edges")
        edges: list[MissionEdge] = []
        for index, raw_edge in enumerate(raw_edges):
            edge_data = _strict_mapping(
                raw_edge,
                {"source_node_id", "target_node_id", "on"},
                f"replacement_edges[{index}]",
            )
            edges.append(
                MissionEdge(
                    edge_data["source_node_id"],  # type: ignore[arg-type]
                    edge_data["target_node_id"],  # type: ignore[arg-type]
                    edge_data["on"],  # type: ignore[arg-type]
                )
            )
        reasons = _sequence(data["reason_codes"], "reason_codes")
        return cls(
            mission_id=data["mission_id"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            base_plan_version=data["base_plan_version"],  # type: ignore[arg-type]
            new_plan_version=data["new_plan_version"],  # type: ignore[arg-type]
            replace_from_node_id=data["replace_from_node_id"],  # type: ignore[arg-type]
            replacement_nodes=tuple(nodes),
            replacement_edges=tuple(edges),
            reason_codes=tuple(reasons),  # type: ignore[arg-type]
        )


def _strict_mapping(
    value: object,
    required: set[str],
    name: str,
) -> dict[str, object]:
    return _strict_mapping_optional(value, required=required, optional=set(), name=name)


def _strict_mapping_optional(
    value: object,
    *,
    required: set[str],
    optional: set[str],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise MissionProgramError(f"{name} must be an object")
    data = dict(value)
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required - optional)
    if missing:
        raise MissionProgramError(
            f"{name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise MissionProgramError(
            f"{name} has unknown fields: {', '.join(unknown)}"
        )
    return data


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise MissionProgramError(f"{name} must be an array")
    return tuple(value)


def apply_program_patch(
    program: MissionProgram,
    patch: ProgramPatch,
    *,
    completed_node_ids: frozenset[str] = frozenset(),
) -> MissionProgram:
    """Validate completely and return a new program; never mutate ``program``."""

    if not isinstance(program, MissionProgram) or not isinstance(patch, ProgramPatch):
        raise TypeError("program and patch must use their typed contracts")
    if patch.mission_id != program.mission_id or patch.uav_id != program.uav_id:
        raise MissionProgramError("ProgramPatch routing does not match MissionProgram")
    if patch.base_plan_version != program.plan_version:
        raise MissionProgramError("ProgramPatch base_plan_version is stale")
    known = {node.node_id for node in program.nodes}
    completed = frozenset(
        validate_routing_id(item, "completed_node_id") for item in completed_node_ids
    )
    if not completed <= known:
        raise MissionProgramError("completed_node_ids contains an unknown node")
    if patch.replace_from_node_id in completed:
        raise MissionProgramError("completed nodes cannot be replaced")
    try:
        start_index = next(
            index
            for index, node in enumerate(program.nodes)
            if node.node_id == patch.replace_from_node_id
        )
    except StopIteration:
        raise MissionProgramError("replace_from_node_id does not exist") from None
    prefix = program.nodes[:start_index]
    if any(node.node_id not in {item.node_id for item in prefix} for node in program.nodes if node.node_id in completed):
        raise MissionProgramError("patch would modify a completed node")
    new_nodes = prefix + patch.replacement_nodes
    kept_ids = {node.node_id for node in prefix}
    replacement_ids = {node.node_id for node in patch.replacement_nodes}
    invalid_sources = sorted(
        {
            edge.source_node_id
            for edge in patch.replacement_edges
            if edge.source_node_id not in replacement_ids
        }
    )
    if invalid_sources:
        raise MissionProgramError(
            "replacement_edges may originate only from replacement nodes; "
            "completed/prefix control flow is immutable"
        )

    # Prefix-internal edges and the trusted boundary edge into the unchanged
    # current node are runtime-owned history.  Preserve them automatically so
    # a model patch neither restates nor modifies completed control flow.
    kept_edges = tuple(
        edge
        for edge in program.edges
        if edge.source_node_id in kept_ids
        and (
            edge.target_node_id in kept_ids
            or edge.target_node_id == patch.replace_from_node_id
        )
    )
    candidate = replace(
        program,
        plan_version=patch.new_plan_version,
        nodes=new_nodes,
        edges=kept_edges + patch.replacement_edges,
    )
    # The dataclass constructor above performs all edge/reference/duplicate
    # checks before this candidate can escape.
    return candidate


__all__ = ["ProgramPatch", "apply_program_patch"]
