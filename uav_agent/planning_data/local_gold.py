"""Deterministic local labels and an independent blueprint-based audit.

Labels consume the production Fleet compiler's focused request.  The audit
consumes the original task blueprint, never a model-interpreted TaskSpec.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from math import isfinite
from numbers import Real

from planner.schemas import PlannerRequest
from planner.schemas_v3 import SkillPlanDraftV3


def _mapping(value: object, name: str) -> dict[str, object]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _number(value: object, name: str, *, minimum: float = 0.0, maximum: float = 86400.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
    return number


def build_local_gold(planner_request: PlannerRequest) -> SkillPlanDraftV3:
    """Render one bounded label in exactly the assigned semantic goal order.

    TAKEOFF is an execution prerequisite. No SEARCH or TRACK is inferred from
    another task. WAIT-only requests require a model-written home/LAND closure
    under the current runtime contract; the blueprint audit records that
    separately from user goals. Compiler-added safety completion is deliberately
    absent from the model label.
    """
    if not isinstance(planner_request, PlannerRequest):
        raise TypeError("planner_request must be a PlannerRequest")
    payload = _mapping(json.loads(planner_request.instruction), "focused request")
    goals = payload.get("assigned_goals")
    if not isinstance(goals, list) or not goals:
        raise ValueError("focused request must contain assigned_goals")
    targets = _mapping(payload.get("trusted_target_specs", {}), "trusted_target_specs")
    aliases = {
        goal.get("target_alias") for goal in goals
        if isinstance(goal, Mapping) and goal.get("target_alias") is not None
    }
    if len(aliases) > 1:
        raise ValueError("local gold supports at most one semantic target")
    target = None
    if aliases:
        alias = next(iter(aliases))
        if alias not in targets:
            raise ValueError("assigned target lacks a trusted specification")
        target = _mapping(targets[alias], "trusted target specification")
    own_home = payload.get("own_home")
    if not isinstance(own_home, str) or not own_home:
        raise ValueError("focused request requires own_home")
    if payload.get("uav_id") != planner_request.uav_id:
        raise ValueError("focused request UAV differs from trusted routing")
    steps: list[dict[str, object]] = []

    def append(skill: str, args: Mapping[str, object]) -> str:
        step_id = f"gold_{len(steps) + 1}_{skill.lower()}"
        steps.append({"id": step_id, "uav_id": planner_request.uav_id, "skill": skill, "args": dict(args)})
        return step_id

    altitude = planner_request.world_context.default_takeoff_altitude_m
    append("TAKEOFF", {"altitude_m": altitude})
    prior_search: str | None = None
    landed = False
    for raw_goal in goals:
        goal = _mapping(raw_goal, "assigned goal")
        kind = goal.get("goal_type")
        if landed:
            raise ValueError("a semantic goal follows LAND")
        if kind == "NAVIGATE":
            spatial = _mapping(goal.get("spatial_constraint"), "NAVIGATE target")
            if spatial.get("kind") != "POINT" or spatial.get("frame") != "WORLD_ENU":
                raise ValueError("local pilot NAVIGATE requires a WORLD_ENU POINT")
            append("GOTO", {"target": spatial})
        elif kind == "WAIT":
            duration = _number(goal.get("duration_s"), "WAIT duration", minimum=1.0, maximum=60.0)
            append("HOVER", {"duration_s": duration})
        elif kind == "SEARCH_TARGET":
            if prior_search is not None:
                raise ValueError("local pilot supports one SEARCH per assignment")
            region = _mapping(goal.get("spatial_constraint"), "SEARCH region")
            if region.get("shape") != "CIRCLE" or region.get("frame") != "WORLD_ENU":
                raise ValueError("local pilot SEARCH requires a WORLD_ENU CIRCLE")
            if target is None:
                raise ValueError("SEARCH requires a trusted target specification")
            timeout = goal.get("duration_s")
            if timeout is None:
                timeout = planner_request.world_context.search_timeout_s
            timeout = _number(timeout, "SEARCH timeout", minimum=1.0, maximum=planner_request.world_context.search_timeout_s)
            prior_search = append("SEARCH", {
                "region": region,
                "strategy": {"kind": "SPIRAL_OUT", "spacing_m": 4.0},
                "entry_policy": "START_IN_PLACE_IF_INSIDE",
                "target_description": target["original_description"],
                "search_altitude_m": altitude,
                "timeout_s": timeout,
            })
        elif kind == "TRACK_TARGET":
            if prior_search is not None:
                reference = f"${prior_search}.target_id"
            elif planner_request.trusted_target_id is not None:
                reference = "$trusted_target.target_id"
            else:
                raise ValueError("TRACK requires a prior SEARCH or trusted target lock")
            duration = _number(goal.get("duration_s"), "TRACK duration", minimum=1.0, maximum=600.0)
            append("TRACK", {"target_ref": reference, "duration_s": duration})
        elif kind in {"RETURN_HOME", "RETURN_HOME_AND_LAND"}:
            append("GOTO", {"target": {"kind": "NAMED_LOCATION", "name": own_home}})
            if kind == "RETURN_HOME_AND_LAND":
                append("LAND", {"zone": own_home})
                landed = True
        elif kind == "LAND":
            last = steps[-1]
            if last.get("skill") != "GOTO" or last.get("args", {}).get("target") != {"kind": "NAMED_LOCATION", "name": own_home}:
                raise ValueError("LAND gold requires a previously assigned RETURN_HOME goal")
            append("LAND", {"zone": own_home})
            landed = True
        else:
            raise ValueError(f"unsupported local pilot goal type: {kind}")
    if planner_request.allow_trusted_safety_completion:
        if landed:
            raise ValueError("safety-completed model labels must omit LAND")
    elif not landed:
        if all(goal.get("goal_type") == "WAIT" for goal in goals):
            # WAIT is classified as a TerminationGoal by the current compiler,
            # disabling automatic completion. Its actual model contract still
            # requires a named-home GOTO and LAND. This closes execution, not
            # an invented RETURN_HOME semantic goal in the task specification.
            append("GOTO", {"target": {"kind": "NAMED_LOCATION", "name": own_home}})
            append("LAND", {"zone": own_home})
        else:
            raise ValueError("assigned goals lack an explicit terminal LAND and safety completion is disabled")
    raw: dict[str, object] = {
        "schema_version": 3,
        "mission_id": planner_request.mission_id,
        "uav_id": planner_request.uav_id,
        "plan_version": planner_request.plan_version,
        "assumptions": [],
        "steps": steps,
    }
    if target is not None:
        raw["target_spec"] = target
    return SkillPlanDraftV3.from_dict(raw)


def _same(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, Real) and isinstance(expected, Real):
        return isfinite(float(actual)) and isfinite(float(expected)) and abs(float(actual) - float(expected)) <= 1e-6
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(_same(actual[key], value) for key, value in expected.items())
    if isinstance(expected, (tuple, list)):
        return isinstance(actual, (tuple, list)) and len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_local_against_blueprint(
    task_assignment: Mapping[str, object],
    draft: object,
    compiled_task_plan: object,
    home_xyz_m: Sequence[float],
    *,
    allow_safety_completion: bool = False,
    target_spec: object | None = None,
) -> dict[str, object]:
    """Check original task facts and ordering in raw AND compiled local plans.

    ``task_assignment`` uses the pilot blueprint's independent fields (kind,
    destination_xyz_m, duration_s, search_center_xyz_m, radius_m,
    target_alias, terminal_goal_type, closure_policy).  It is not a TaskSpec.
    Search identities must be supplied independently through ``target_spec``.
    This strict pilot audit expects one step per requested navigation, search,
    track or hover, avoiding the generic checker's cumulative-duration shortcut.
    """
    row = _mapping(task_assignment, "task_assignment")
    if not isinstance(allow_safety_completion, bool):
        raise TypeError("allow_safety_completion must be a boolean")
    if isinstance(home_xyz_m, (str, bytes)) or len(home_xyz_m) != 3:
        raise ValueError("home_xyz_m must contain three numbers")
    home = tuple(_number(value, "home coordinate", minimum=-100000.0, maximum=100000.0) for value in home_xyz_m)
    owner, kind = row.get("uav_id"), row.get("kind")
    if not isinstance(owner, str) or not owner:
        raise ValueError("blueprint assignment requires uav_id")
    if kind not in {"navigate", "hover", "search", "search_track"}:
        raise ValueError(f"unsupported blueprint assignment kind: {kind}")
    expected: list[tuple[str, str, object]] = [("TAKEOFF", "takeoff", None)]
    if kind == "navigate" or (kind == "hover" and row.get("destination_xyz_m") is not None):
        expected.append(("GOTO", "navigate", {"kind": "POINT", "frame": "WORLD_ENU", "xyz_m": row["destination_xyz_m"]}))
    if kind in {"search", "search_track"}:
        expected.append(("SEARCH", "search", {
            "shape": "CIRCLE", "frame": "WORLD_ENU",
            "center_xyz_m": row["search_center_xyz_m"], "radius_m": row["radius_m"],
        }))
    if kind == "search_track":
        expected.append(("TRACK", "track", row["duration_s"]))
    if kind == "hover":
        expected.append(("HOVER", "hover", row["duration_s"]))
    terminal = row.get("terminal_goal_type")
    if terminal not in {None, "WAIT", "RETURN_HOME_AND_LAND"}:
        raise ValueError(f"unsupported pilot terminal goal: {terminal}")
    contract_closure = row.get("closure_policy") == "runtime_contract_home_and_land"
    explicit_closure = terminal == "RETURN_HOME_AND_LAND"
    model_closure = explicit_closure or contract_closure
    closure = [("GOTO", "home", None), ("LAND", "land", None)]
    if model_closure:
        expected.extend(closure)
    compiled_expected = list(expected)
    automatic_closure = allow_safety_completion and not model_closure
    if automatic_closure:
        compiled_expected.extend(closure)
    findings: list[dict[str, object]] = []

    def issue(code: str, stage: str, **details: object) -> None:
        findings.append({"code": code, "stage": stage, "uav_id": owner, **details})

    if model_closure and allow_safety_completion:
        issue("INCONSISTENT_CLOSURE_POLICY", "blueprint")
    if not model_closure and not allow_safety_completion:
        issue("MISSING_TERMINAL_CLOSURE_POLICY", "blueprint")
    if contract_closure and (kind != "hover" or explicit_closure):
        issue("INVALID_RUNTIME_CONTRACT_CLOSURE", "blueprint")
    trusted_target = None if target_spec is None else _mapping(target_spec, "target_spec")
    if kind in {"search", "search_track"} and trusted_target is None:
        issue("MISSING_INDEPENDENT_TARGET_SPEC", "blueprint")

    raw_plans: dict[str, dict[str, object]] = {}
    for stage, value in (("draft", draft), ("compiled", compiled_task_plan)):
        try:
            raw_plans[stage] = _mapping(value, stage)
        except (TypeError, ValueError):
            issue("MISSING_OR_INVALID_PLAN", stage)
    draft_steps: list[Mapping[str, object]] = []
    if "draft" in raw_plans:
        raw = raw_plans["draft"]
        if raw.get("assumptions") != []:
            issue("UNEXPECTED_ASSUMPTIONS", "draft")
        if trusted_target is not None:
            if not _same(raw.get("target_spec"), trusted_target):
                issue("TARGET_SPEC_MISMATCH", "draft")
        elif kind not in {"search", "search_track"} and raw.get("target_spec") is not None:
            issue("UNEXPECTED_TARGET_SPEC", "draft")
        value = raw.get("steps", [])
        if isinstance(value, list):
            draft_steps = [step for step in value if isinstance(step, Mapping)]
    for stage, expected_steps in (("draft", expected), ("compiled", compiled_expected)):
        if stage not in raw_plans:
            continue
        raw = raw_plans[stage]
        if raw.get("uav_id") != owner:
            issue("UAV_ROUTING_MISMATCH", stage)
        steps = raw.get("steps")
        if not isinstance(steps, list) or any(not isinstance(step, Mapping) for step in steps):
            issue("INVALID_STEPS", stage)
            continue
        skills = [step.get("skill") for step in steps]
        if skills != [item[0] for item in expected_steps]:
            issue("STEP_SEQUENCE_MISMATCH", stage, actual_skills=skills, expected_skills=[item[0] for item in expected_steps])
        ids = [step.get("id") for step in steps]
        if any(not isinstance(step_id, str) for step_id in ids) or len(set(ids)) != len(ids):
            issue("INVALID_STEP_IDENTITIES", stage)
        prior_search: str | None = None
        return_name: str | None = None
        for index, ((skill, action, wanted), step) in enumerate(zip(expected_steps, steps)):
            if step.get("skill") != skill:
                continue
            args = step.get("args", {}) if stage == "draft" else step
            if not isinstance(args, Mapping):
                issue("INVALID_STEP_ARGUMENTS", stage, step_index=index)
                continue
            if stage == "draft" and step.get("uav_id") != owner:
                issue("STEP_UAV_ROUTING_MISMATCH", stage, step_index=index)
            if stage == "compiled" and index < len(draft_steps) and step.get("id") != draft_steps[index].get("id"):
                issue("COMPILED_STEP_ID_MISMATCH", stage, step_index=index)
            if action == "navigate":
                actual = args.get("target") if stage == "draft" else args.get("position")
                desired = wanted if stage == "draft" else wanted["xyz_m"]
                if not _same(actual, desired):
                    issue("NAVIGATION_COORDINATES_MISMATCH", stage, step_index=index)
            elif action == "search":
                region = args.get("region")
                # TaskPlan serializes RegionSpec dataclass fields; CircleRegion
                # stores shape as a ClassVar, so its compiled JSON omits it.
                # The remaining exact field set still distinguishes geometry.
                expected_region = wanted
                if stage == "compiled" and isinstance(region, Mapping) and "shape" not in region:
                    expected_region = {key: value for key, value in wanted.items() if key != "shape"}
                if not _same(region, expected_region):
                    issue("SEARCH_REGION_MISMATCH", stage, step_index=index)
                if trusted_target is not None and args.get("target_description") != trusted_target.get("original_description"):
                    issue("SEARCH_TARGET_MISMATCH", stage, step_index=index)
                prior_search = step.get("id")
            elif action in {"hover", "track"}:
                duration_key = "track_duration" if stage == "compiled" and action == "track" else "duration_s"
                if not _same(args.get(duration_key), wanted):
                    issue("DURATION_MISMATCH", stage, step_index=index)
                if action == "track":
                    reference = args.get("target_ref" if stage == "draft" else "target_id")
                    if prior_search is None or reference != f"${prior_search}.target_id":
                        issue("TRACK_REFERENCE_MISMATCH", stage, step_index=index)
            elif action == "home":
                if stage == "draft":
                    target = args.get("target")
                    if not isinstance(target, Mapping) or target.get("kind") != "NAMED_LOCATION" or not isinstance(target.get("name"), str):
                        issue("HOME_TARGET_MISMATCH", stage, step_index=index)
                    else:
                        return_name = target["name"]
                        if row.get("home_name") is not None and return_name != row["home_name"]:
                            issue("HOME_TARGET_MISMATCH", stage, step_index=index)
                else:
                    position = args.get("position")
                    if not isinstance(position, (tuple, list)) or len(position) != 3 or not _same(position[:2], home[:2]):
                        issue("HOME_COORDINATES_MISMATCH", stage, step_index=index)
            elif action == "land":
                if stage == "draft":
                    if return_name is None or args.get("zone") != return_name:
                        issue("LAND_ZONE_MISMATCH", stage, step_index=index)
                elif not _same(args.get("expected_position_xy"), home[:2]) or not _same(args.get("ground_altitude"), home[2]):
                    issue("LAND_GEOMETRY_MISMATCH", stage, step_index=index)
    if "draft" in raw_plans and "compiled" in raw_plans:
        for field in ("mission_id", "plan_version"):
            if raw_plans["draft"].get(field) != raw_plans["compiled"].get(field):
                issue("PLAN_ROUTING_MISMATCH", "compiled", field=field)
    compiled_steps = raw_plans.get("compiled", {}).get("steps", [])
    completion_observed = (
        automatic_closure
        and isinstance(compiled_steps, list)
        and len(compiled_steps) == len(draft_steps) + 2
        and all(isinstance(step, Mapping) for step in compiled_steps[-2:])
        and [step.get("skill") for step in compiled_steps[-2:]] == ["GOTO", "LAND"]
    )
    return {
        "passed": not findings,
        "findings": findings,
        "runtime_contract_closure": contract_closure,
        "runtime_safety_completion_added": completion_observed,
        "source": "independent_task_blueprint",
    }


__all__ = ["build_local_gold", "validate_local_against_blueprint"]
