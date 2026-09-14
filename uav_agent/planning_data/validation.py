"""Independent blueprint-to-label checks for planning SFT data.

Expected facts are read directly from the task blueprint.  This module never
calls the task/answer builders: a shared generator mistake must not become its
own gold standard.  Production dataclasses still enforce the structural schema.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose
from typing import Any

from fleet.task_spec import FleetTaskSpecV1, MissionGoal, TerminationGoal
from fleet.types_v2 import FleetMissionPlanV2


@dataclass(frozen=True)
class _ExpectedGoal:
    owner: str
    goal_type: str
    target_alias: str | None = None
    spatial: Mapping[str, object] | None = None
    duration_s: float | None = None
    termination: bool = False


def _value(value: object) -> object:
    return getattr(value, "value", value)


def _same(actual: object, expected: object) -> bool:
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
        )
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            _same(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (tuple, list)):
        return isinstance(actual, (tuple, list)) and len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def _expected_chains(task: Mapping[str, object]) -> list[list[_ExpectedGoal]]:
    rows = task.get("assignments")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("task blueprint requires non-empty assignments")
    chains: list[list[_ExpectedGoal]] = []
    owners: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("task assignment must be an object")
        owner = row.get("uav_id")
        if not isinstance(owner, str) or not owner or owner in owners:
            raise ValueError("blueprint assignments require distinct UAV owners")
        owners.add(owner)
        kind = row.get("kind")
        if kind not in {"navigate", "hover", "search", "search_track"}:
            raise ValueError(f"unsupported blueprint assignment kind: {kind!r}")
        chain: list[_ExpectedGoal] = []
        if kind == "navigate" or (kind == "hover" and row.get("destination_xyz_m") is not None):
            chain.append(_ExpectedGoal(owner, "NAVIGATE", spatial={
                "kind": "POINT", "frame": "WORLD_ENU",
                "xyz_m": row["destination_xyz_m"],
            }))
        if kind in {"search", "search_track"}:
            alias = row["target_alias"]
            if not isinstance(alias, str) or not alias:
                raise ValueError("search blueprint requires a target_alias")
            radius = row.get("radius_m", row.get("search_radius_m"))
            if radius is None:
                raise ValueError("search blueprint requires radius_m")
            if "radius_m" in row and "search_radius_m" in row and not _same(row["radius_m"], row["search_radius_m"]):
                raise ValueError("blueprint radius_m and search_radius_m disagree")
            chain.append(_ExpectedGoal(owner, "SEARCH_TARGET", alias, {
                "shape": "CIRCLE", "frame": "WORLD_ENU",
                "center_xyz_m": row["search_center_xyz_m"],
                "radius_m": radius,
            }))
            if kind == "search_track":
                chain.append(_ExpectedGoal(owner, "TRACK_TARGET", alias, duration_s=row["duration_s"]))
        if kind == "hover":
            chain.append(_ExpectedGoal(owner, "WAIT", duration_s=row["duration_s"], termination=True))
        terminal = row.get("terminal_goal_type")
        if terminal is not None and not (kind == "hover" and terminal == "WAIT"):
            if terminal not in {"RETURN_HOME", "LAND", "RETURN_HOME_AND_LAND", "WAIT", "REPORT"}:
                raise ValueError(f"unsupported blueprint terminal goal: {terminal!r}")
            chain.append(_ExpectedGoal(
                owner, terminal,
                duration_s=row.get("duration_s") if terminal == "WAIT" else None,
                termination=True,
            ))
        chains.append(chain)
    return chains


def validate_semantic_gold(
    task: dict[str, Any],
    spec: FleetTaskSpecV1,
    plan: FleetMissionPlanV2,
) -> dict[str, object]:
    """Check complete labels against their independent task blueprint.

    Goal and constraint identifiers may be renamed freely.  Semantic matching
    uses goal type, target, spatial facts and explicit ownership, then checks
    every expected fact, constraint and assignment.  Findings are JSON-safe.
    Invalid blueprints/types raise; semantically incorrect labels return failed.
    """
    if not isinstance(task, Mapping):
        raise TypeError("task must be a blueprint object")
    if not isinstance(spec, FleetTaskSpecV1) or not isinstance(plan, FleetMissionPlanV2):
        raise TypeError("spec and plan must be production planning values")
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task blueprint requires an instruction")
    chains = _expected_chains(task)
    findings: list[dict[str, object]] = []

    def issue(code: str, message: str, *, goal_id: str | None = None, uav_id: str | None = None) -> None:
        findings.append({"code": code, "message": message, "goal_id": goal_id, "uav_id": uav_id})

    if spec.source_text != instruction:
        issue("SOURCE_TEXT_MISMATCH", "task_spec must retain the original blueprint instruction")
    if spec.ambiguities:
        issue("UNEXPECTED_AMBIGUITY", "fully specified blueprint has no unresolved ambiguities")
    evidence = {item.evidence_id: item.quote for item in spec.source_evidence}
    for evidence_id, quote in evidence.items():
        if not quote or quote not in instruction:
            issue("INVALID_SOURCE_EVIDENCE", f"evidence {evidence_id} is not verbatim original instruction")
    all_goals = (*spec.goals, *spec.termination_goals)
    for item in (*all_goals, *spec.assignment_constraints, *spec.ordering_constraints):
        refs = item.evidence_refs
        if not refs or any(ref not in evidence or evidence[ref] not in instruction for ref in refs):
            issue("MISSING_OR_INVALID_EVIDENCE", "each goal and constraint must cite original instruction evidence", goal_id=getattr(item, "goal_id", None))

    constraints_by_goal: dict[str, list[object]] = {goal.goal_id: [] for goal in all_goals}
    for constraint in spec.assignment_constraints:
        for goal_id in constraint.goal_ids:
            constraints_by_goal.setdefault(goal_id, []).append(constraint)
    unused = {goal.goal_id: goal for goal in all_goals}
    expected_by_id: dict[str, _ExpectedGoal] = {}
    matched_chains: list[list[str | None]] = []
    for chain in chains:
        matched: list[str | None] = []
        for expected in chain:
            candidates = [goal for goal in unused.values() if (
                _value(goal.goal_type) == expected.goal_type
                and isinstance(goal, TerminationGoal) == expected.termination
            )]
            if not candidates:
                issue("MISSING_GOAL", f"missing {expected.goal_type} for {expected.owner}", uav_id=expected.owner)
                matched.append(None)
                continue

            def score(goal: MissionGoal | TerminationGoal) -> int:
                candidate_owners = {c.uav_id for c in constraints_by_goal[goal.goal_id]}
                if isinstance(goal, TerminationGoal):
                    candidate_owners.add(goal.uav_id)
                spatial = getattr(goal, "spatial_constraint", None)
                return (
                    8 * (expected.owner in candidate_owners)
                    + 4 * (getattr(goal, "target_alias", None) == expected.target_alias)
                    + 2 * _same(None if spatial is None else spatial.to_dict(), expected.spatial)
                    + _same(goal.duration_s, expected.duration_s)
                )

            goal = max(candidates, key=score)
            del unused[goal.goal_id]
            expected_by_id[goal.goal_id] = expected
            matched.append(goal.goal_id)
            detail = {"goal_id": goal.goal_id, "uav_id": expected.owner}
            if _value(goal.strength) != "MUST":
                issue("GOAL_STRENGTH_MISMATCH", "explicit task goals must retain MUST strength", **detail)
            if getattr(goal, "target_alias", None) != expected.target_alias:
                issue("TARGET_ALIAS_MISMATCH", "goal target differs from the blueprint", **detail)
            spatial = getattr(goal, "spatial_constraint", None)
            if not _same(None if spatial is None else spatial.to_dict(), expected.spatial):
                issue("SPATIAL_CONSTRAINT_MISMATCH", "goal coordinates, frame, shape or radius differ from blueprint", **detail)
            if not _same(goal.duration_s, expected.duration_s):
                issue("DURATION_MISMATCH", "goal duration differs from blueprint (including invented duration)", **detail)
            if getattr(goal, "distance_m", None) is not None:
                issue("INVENTED_DISTANCE", "blueprint does not specify a separate distance goal", **detail)
            constraints = constraints_by_goal[goal.goal_id]
            must_owners = {c.uav_id for c in constraints if _value(c.strength) == "MUST"}
            if must_owners != {expected.owner}:
                issue("MUST_BINDING_MISMATCH", "goal lacks its exact explicit MUST UAV binding", **detail)
            if any(c.uav_id != expected.owner or _value(c.strength) != "MUST" for c in constraints):
                issue("UNEXPECTED_BINDING", "assignment constraint weakens or conflicts with blueprint binding", **detail)
            if isinstance(goal, TerminationGoal) and goal.uav_id != expected.owner:
                issue("TERMINATION_OWNER_MISMATCH", "termination must refer to the owning UAV", **detail)
        matched_chains.append(matched)
    for goal_id, goal in unused.items():
        issue("UNEXPECTED_GOAL", f"extra or duplicate {_value(goal.goal_type)}", goal_id=goal_id)

    positions = {
        goal_id: (chain_index, offset)
        for chain_index, chain in enumerate(matched_chains)
        for offset, goal_id in enumerate(chain) if goal_id is not None
    }
    edges: dict[str, set[str]] = {goal.goal_id: set() for goal in all_goals}
    for ordering in spec.ordering_constraints:
        before, after = ordering.before_goal_id, ordering.after_goal_id
        if _value(ordering.strength) != "MUST":
            issue("ORDERING_STRENGTH_MISMATCH", "explicit execution order must retain MUST strength")
        else:
            edges.setdefault(before, set()).add(after)
        if (
            before not in positions or after not in positions
            or positions[before][0] != positions[after][0]
            or positions[before][1] >= positions[after][1]
        ):
            issue("UNEXPECTED_ORDERING", "ordering reverses a task chain or serializes parallel UAVs", goal_id=after)

    def reachable(start: str, target: str) -> bool:
        pending, visited = [start], set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if target in edges.get(current, ()):
                return True
            pending.extend(edges.get(current, ()))
        return False

    for chain in matched_chains:
        for before, after in zip(chain, chain[1:]):
            if before is not None and after is not None and not reachable(before, after):
                issue("MISSING_ORDERING", f"missing required order {before} before {after}", goal_id=after)

    counts = Counter(goal_id for assignment in plan.assignments for goal_id in assignment.goal_ids)
    known = set(spec.all_goal_ids)
    for goal_id in known:
        if counts[goal_id] != 1:
            issue("FLEET_GOAL_COVERAGE", "every interpreted goal must be assigned exactly once", goal_id=goal_id)
    for goal_id in counts.keys() - known:
        issue("FLEET_UNKNOWN_GOAL", "assignment references a goal absent from the task spec", goal_id=goal_id)
    expected_owners = {chain[0].owner for chain in chains}
    actual_owners = {assignment.uav_id for assignment in plan.assignments}
    if actual_owners != expected_owners:
        issue("FLEET_UAV_COVERAGE", "fleet assignment owners differ from blueprint UAVs")
    for assignment in plan.assignments:
        if _value(assignment.start_policy) != "PARALLEL":
            issue("FLEET_START_POLICY", "blueprint UAV tasks must start in parallel", uav_id=assignment.uav_id)
        if assignment.deviations:
            issue("UNEXPECTED_DEVIATION", "fully available explicit blueprint needs no assignment deviation", uav_id=assignment.uav_id)
        for goal_id in assignment.goal_ids:
            expected = expected_by_id.get(goal_id)
            if expected is not None and assignment.uav_id != expected.owner:
                issue("FLEET_OWNER_MISMATCH", "goal is assigned to a UAV different from blueprint owner", goal_id=goal_id, uav_id=assignment.uav_id)
    if plan.unassigned_goal_ids:
        issue("FLEET_UNASSIGNED_GOALS", "complete blueprint must not leave goals unassigned")
    if plan.assumptions:
        issue("UNEXPECTED_FLEET_ASSUMPTION", "fully specified blueprint needs no invented assumptions")
    if task.get("fleet_mission_id") is not None and plan.fleet_mission_id != task["fleet_mission_id"]:
        issue("FLEET_MISSION_ID_MISMATCH", "plan routing differs from blueprint mission")
    return {"passed": not findings, "findings": findings}


__all__ = ["validate_semantic_gold"]
