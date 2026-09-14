#!/usr/bin/env python3
"""Bounded, simulator-free Qwen Fleet evaluation against independent task gold.

This measures text/structured planning, not visual grounding or flight success.
Each case uses the production Interpreter, Fleet V2 and Spatial V3 planners.
Gold is used only by the evaluator and is never supplied to the model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from math import isclose, isfinite
from pathlib import Path
import re
import sys
from time import perf_counter
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.loader import load_config
from fleet.compiler import FleetAssignmentCompiler
from fleet.llm_planner_v2 import LLMFleetPlannerV2
from fleet.llm_task_interpreter import LLMFleetTaskInterpreter
from fleet.request_builder import (
    build_agent_world_contexts_v2,
    build_fleet_mission_request_v2,
    build_target_catalog,
)
from models import ModelResponse
from models.adapter_registry import AdapterRegistry, DEFAULT_ADAPTER_CONFIG, ModelCallRole
from models.model_client_factory import ModelClientFactory
from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.policy import PlannerLimits, PlannerPolicy
from runtime.plan_validator import PlanValidator

_MAX_SUITE_BYTES = 262_144
_MAX_RESPONSE_BYTES = 65_536
_ID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def _read_json(path: Path) -> object:
    if path.stat().st_size > _MAX_SUITE_BYTES:
        raise ValueError("suite exceeds 256 KiB")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=reject)


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("gold coordinates, radius and duration must be finite numbers")
    if positive and value <= 0:
        raise ValueError("gold radius and duration must be positive")
    return float(value)


def load_suite(path: Path) -> list[dict]:
    """Validate the bounded search/track benchmark contract without model calls."""
    path = path.expanduser().resolve()
    suite = _read_json(path)
    if not isinstance(suite, dict) or suite.get("schema_version") != 1:
        raise ValueError("suite schema_version must be 1")
    if set(suite) - {"schema_version", "description", "cases"}:
        raise ValueError("unknown suite field")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("suite must contain 1..32 cases")
    seen = set()
    result = []
    for case in cases:
        required = {"id", "config", "instruction", "gold"}
        if not isinstance(case, dict) or not required <= set(case) or set(case) - required - {"description"}:
            raise ValueError("each case requires id, config, instruction, gold; description is optional")
        case_id = case["id"]
        if not isinstance(case_id, str) or not _ID.fullmatch(case_id) or case_id in seen:
            raise ValueError("case id must be a unique lowercase routing identifier")
        seen.add(case_id)
        if not isinstance(case["instruction"], str) or not 1 <= len(case["instruction"].strip()) <= 8192:
            raise ValueError(f"{case_id}: instruction must contain 1..8192 characters")
        if not isinstance(case["config"], str) or not case["config"]:
            raise ValueError(f"{case_id}: config must be a path")
        config_path = (path.parent / case["config"]).resolve()
        config = load_config(config_path)
        if config.scene.obstacles:
            raise ValueError(f"{case_id}: this open-field benchmark requires obstacles: []")
        gold = case["gold"]
        if not isinstance(gold, dict) or set(gold) != {"targets", "return_home_and_land"}:
            raise ValueError(f"{case_id}: gold requires targets and return_home_and_land")
        if not isinstance(gold["return_home_and_land"], bool):
            raise ValueError(f"{case_id}: return_home_and_land must be boolean")
        targets = gold["targets"]
        if not isinstance(targets, list) or not 1 <= len(targets) <= 16:
            raise ValueError(f"{case_id}: gold must contain 1..16 targets")
        uavs = {u.id for u in config.uavs}
        aliases = {t.id for t in config.targets}
        used_aliases, used_uavs = set(), set()
        for target in targets:
            fields = {"target_alias", "uav_id", "search_center_xyz_m", "search_radius_m", "track_duration_s"}
            if not isinstance(target, dict) or set(target) != fields:
                raise ValueError(f"{case_id}: invalid gold target fields")
            alias, uav_id = target["target_alias"], target["uav_id"]
            if alias not in aliases or alias in used_aliases:
                raise ValueError(f"{case_id}: duplicate or unknown gold target")
            used_aliases.add(alias)
            if uav_id is not None:
                if uav_id not in uavs or uav_id in used_uavs:
                    raise ValueError(f"{case_id}: duplicate or unknown gold UAV")
                used_uavs.add(uav_id)
            center = target["search_center_xyz_m"]
            if not isinstance(center, list) or len(center) != 3:
                raise ValueError(f"{case_id}: search center must be a 3-vector")
            center = [_number(x) for x in center]
            radius = _number(target["search_radius_m"], positive=True)
            _number(target["track_duration_s"], positive=True)
            sx, sy, sz = config.scene.size_xyz_m
            if abs(center[0]) + radius > sx / 2 or abs(center[1]) + radius > sy / 2 or not 0 <= center[2] <= sz:
                raise ValueError(f"{case_id}: gold search circle is outside the scene")
        if len(targets) > len(uavs):
            raise ValueError(f"{case_id}: benchmark supports at most one target per UAV")
        result.append({**case, "config_path": str(config_path), "loaded_config": config})
    return result


def _equal(a, b):
    return (not isinstance(a, bool) and isinstance(a, (int, float))
            and isclose(a, b, rel_tol=0.0, abs_tol=1e-5))


def _termination_kinds(goals):
    kinds = {g.get("goal_type") for g in goals}
    return "RETURN_HOME_AND_LAND" in kinds or {"RETURN_HOME", "LAND"} <= kinds


def _precedes(spec, before, after):
    edges = {}
    for constraint in spec.get("ordering_constraints", []):
        if constraint.get("strength") == "MUST":
            edges.setdefault(constraint.get("before_goal_id"), []).append(constraint.get("after_goal_id"))
    pending, visited = list(edges.get(before, [])), set()
    while pending:
        value = pending.pop()
        if value == after:
            return True
        if value not in visited:
            visited.add(value)
            pending.extend(edges.get(value, []))
    return False


def score_interpretation(spec: dict | None, gold: dict, uav_ids) -> dict:
    """Align by target semantics; model-created goal IDs never serve as gold IDs."""
    failures = []
    if not isinstance(spec, dict):
        return {"passed": False, "findings": ["no_valid_task_spec"]}
    goals = spec.get("goals", [])
    expected = {t["target_alias"]: t for t in gold["targets"]}
    if Counter(g.get("target_alias") for g in goals) != Counter({a: 2 for a in expected}):
        failures.append("missing_duplicate_or_extra_target_goals")
    for alias, target in expected.items():
        selected = [g for g in goals if g.get("target_alias") == alias]
        if Counter(g.get("goal_type") for g in selected) != Counter({"SEARCH_TARGET": 1, "TRACK_TARGET": 1}):
            failures.append(f"{alias}:search_track_goals")
        else:
            search = next(g for g in selected if g["goal_type"] == "SEARCH_TARGET")
            track = next(g for g in selected if g["goal_type"] == "TRACK_TARGET")
            if (not _precedes(spec, search["goal_id"], track["goal_id"])
                or _precedes(spec, track["goal_id"], search["goal_id"])):
                failures.append(f"{alias}:search_before_track_order")
        for goal in selected:
            if goal.get("strength") != "MUST":
                failures.append(f"{alias}:required_goal_weakened")
            if goal.get("goal_type") == "SEARCH_TARGET":
                if goal.get("duration_s") is not None:
                    failures.append(f"{alias}:invented_search_duration")
                region = goal.get("spatial_constraint") or {}
                center = region.get("center_xyz_m", [])
                if (region.get("shape") != "CIRCLE" or region.get("frame") != "WORLD_ENU"
                    or len(center) != 3 or not all(_equal(a, b) for a, b in zip(center, target["search_center_xyz_m"]))
                    or not _equal(region.get("radius_m"), target["search_radius_m"])):
                    failures.append(f"{alias}:search_region")
            if goal.get("goal_type") == "TRACK_TARGET" and not _equal(goal.get("duration_s"), target["track_duration_s"]):
                failures.append(f"{alias}:track_duration")
            constraints = [c for c in spec.get("assignment_constraints", [])
                           if goal.get("goal_id") in c.get("goal_ids", []) and c.get("strength") != "OPEN"]
            if target["uav_id"] is not None and (
                not constraints or any(c.get("uav_id") != target["uav_id"] for c in constraints)
                or not any(c.get("strength") == "MUST" for c in constraints)
            ):
                failures.append(f"{alias}:explicit_uav_constraint")
            if target["uav_id"] is None and constraints:
                failures.append(f"{alias}:invented_uav_constraint")
    if spec.get("ambiguities"):
        failures.append("unexpected_ambiguities")
    if any(before.get("target_alias") != after.get("target_alias")
           and _precedes(spec, before["goal_id"], after["goal_id"])
           for before in goals for after in goals):
        failures.append("cross_target_serialization")
    if any(g.get("goal_type") not in {"RETURN_HOME", "LAND", "RETURN_HOME_AND_LAND"}
           for g in spec.get("termination_goals", [])):
        failures.append("unexpected_termination_action")
    if gold["return_home_and_land"]:
        for uav_id in uav_ids:
            terminal = [g for g in spec.get("termination_goals", [])
                        if g.get("uav_id") in (None, uav_id) and g.get("strength") == "MUST"]
            if not _termination_kinds(terminal):
                failures.append(f"{uav_id}:return_home_and_land")
    return {"passed": not failures, "findings": sorted(set(failures))}


def score_assignment(plan: dict | None, spec: dict, gold: dict) -> dict:
    failures = []
    if not isinstance(plan, dict):
        return {"passed": False, "findings": ["no_valid_fleet_plan"]}
    assignments = plan.get("assignments", [])
    goals = {g["goal_id"]: g for g in spec.get("goals", []) + spec.get("termination_goals", [])}
    owners = {}
    for assignment in assignments:
        if assignment.get("start_policy") != "PARALLEL":
            failures.append("nonparallel_assignment")
        for goal_id in assignment.get("goal_ids", []):
            owners.setdefault(goal_id, []).append(assignment.get("uav_id"))
    if plan.get("unassigned_goal_ids"):
        failures.append("unassigned_goals")
    if any(len(owners.get(goal_id, [])) != 1 for goal_id in goals):
        failures.append("missing_or_duplicate_goal_ownership")
    routed_uavs = []
    for target in gold["targets"]:
        alias = target["target_alias"]
        relevant = [g for g in goals.values() if g.get("target_alias") == alias]
        routed = {u for g in relevant for u in owners.get(g["goal_id"], [])}
        if len(relevant) != 2 or len(routed) != 1:
            failures.append(f"{alias}:split_or_missing_target_assignment")
            continue
        uav_id = next(iter(routed))
        routed_uavs.append(uav_id)
        if target["uav_id"] is not None and uav_id != target["uav_id"]:
            failures.append(f"{alias}:wrong_uav")
        if gold["return_home_and_land"]:
            terminal = [g for gid, g in goals.items() if "target_alias" not in g
                        and uav_id in owners.get(gid, []) and g.get("uav_id") in (None, uav_id)]
            if not _termination_kinds(terminal):
                failures.append(f"{uav_id}:termination_not_assigned")
    if len(set(routed_uavs)) != len(gold["targets"]):
        failures.append("targets_not_one_per_uav")
    if len(assignments) != len(gold["targets"]):
        failures.append("assignment_count")
    return {"passed": not failures, "findings": sorted(set(failures))}


def _write(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class _TraceClient:
    def __init__(self, client, role, records, persist, *, max_tokens=None, uav_id=None):
        self.client, self.role, self.records, self.persist = client, role, records, persist
        self.max_tokens, self.uav_id = max_tokens, uav_id
        self.attempt = 0

    def chat(self, messages, *, options=None):
        if self.max_tokens is not None:
            options = replace(options, max_tokens=self.max_tokens)
        prepare_options = getattr(self.client, "prepare_options", None)
        if callable(prepare_options):
            options = prepare_options(options)
        row = {"role": self.role, "uav_id": self.uav_id, "attempt_index": self.attempt,
               "max_tokens": options.max_tokens, "temperature": options.temperature,
               "top_p": options.top_p,
               "structured_output_enabled": options.response_format is not None,
               "requested_model": getattr(self.client, "model", None),
               "generation_options_json": json.dumps({
                   "max_tokens": options.max_tokens, "temperature": options.temperature,
                   "top_p": options.top_p, "response_format": (
                       options.response_format.to_dict() if options.response_format else None
                   ),
               }, ensure_ascii=False, separators=(",", ":")),
               "messages": [message.to_dict() for message in messages]}
        self.attempt += 1
        started = perf_counter()
        try:
            response = self.client.chat(messages, options=options)
            if not isinstance(response, ModelResponse):
                raise TypeError("invalid model response")
            encoded = response.content.encode("utf-8")
            row.update({"model": response.model, "usage": response.usage,
                        "finish_reason": response.finish_reason,
                        "response_text": encoded[:_MAX_RESPONSE_BYTES].decode("utf-8", errors="ignore"),
                        "response_bytes": len(encoded), "response_truncated_in_log": len(encoded) > _MAX_RESPONSE_BYTES})
            return response
        except Exception as exc:
            row.update({"error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)})
            raise
        finally:
            row["latency_s"] = perf_counter() - started
            self.records.append(row)
            self.persist()


def _capture(stage, planner):
    diagnostics = getattr(planner, "last_diagnostics", None)
    stage["diagnostics"] = None if diagnostics is None else diagnostics.to_dict()
    stage["model_proposals"] = list(getattr(planner, "model_proposals", ()))


def _stage_scores(stage, final_score, first_score):
    diagnostic = stage.get("diagnostics") or {}
    stage["gold_score"] = final_score
    stage["first_proposal_gold_score"] = first_score
    stage["first_pass"] = bool(diagnostic.get("initial_output_valid") and first_score["passed"])
    stage["final_pass"] = bool(diagnostic.get("final_output_valid") and final_score["passed"])


def _first_proposal(stage):
    proposals = stage.get("model_proposals", [])
    return proposals[0].get("proposal") if proposals and proposals[0].get("accepted") else None


def _overall_scores(report):
    interpreter_stage = report["stages"].get("interpreter", {})
    fleet_stage = report["stages"].get("fleet", {})
    local_stages = report["stages"].get("local", {})
    enough_local = len(local_stages) == len(report["gold"]["targets"])
    for key in ("first_pass", "final_pass"):
        report[key] = bool(interpreter_stage.get(key) and fleet_stage.get(key) and enough_local
                           and all(s.get(key) for s in local_stages.values()))


def rescore_report(report: dict, uav_ids=None) -> dict:
    """Return a regraded copy of a saved result; no model calls or file writes.

    The first proposal and final output are rescored independently. Local
    compiler reports are retained because they already checked actual skills.
    """
    result = deepcopy(report)
    if uav_ids is None:
        uav_ids = result.get("uav_ids") or [u.id for u in load_config(Path(result["config"])).uavs]
    interpreter = result["stages"].get("interpreter")
    if interpreter is not None:
        _stage_scores(interpreter, score_interpretation(interpreter.get("output"), result["gold"], uav_ids),
                      score_interpretation(_first_proposal(interpreter), result["gold"], uav_ids))
        fleet = result["stages"].get("fleet")
        if fleet is not None and interpreter.get("output") is not None:
            _stage_scores(fleet, score_assignment(fleet.get("output"), interpreter["output"], result["gold"]),
                          score_assignment(_first_proposal(fleet), interpreter["output"], result["gold"]))
    _overall_scores(result)
    result["grading_version"] = 2
    return result


def run_case(case, args, output_dir: Path, registry=None):
    """Persist every actual model call and retain partial results after failures."""
    output_dir.mkdir(parents=True, exist_ok=False)
    config, gold = case["loaded_config"], case["gold"]
    report = {"schema_version": 1, "grading_version": 2, "case_id": case["id"], "instruction": case["instruction"],
              "config": case["config_path"], "gold": gold, "stages": {}, "model_calls": [],
              "uav_ids": [u.id for u in config.uavs],
              "evaluation_scope": "text_structured_planning_without_simulation",
              "budgets": {"interpreter_max_tokens": args.interpreter_max_tokens,
                          "fleet_max_tokens": args.fleet_max_tokens, "local_max_tokens": args.local_max_tokens,
                          "interpreter_repairs": args.interpreter_repairs, "fleet_repairs": args.fleet_repairs,
                          "local_repairs": args.local_repairs},
              "first_pass": False, "final_pass": False, "status": "running"}
    persist = lambda: _write(output_dir / "result.json", report)
    persist()
    registry = registry or AdapterRegistry(args.adapter_config)
    factory = ModelClientFactory(registry, base_url=args.base_url, timeout_s=args.timeout_s,
                                 max_retries=0, selection_logger=lambda row: report.setdefault("adapter_selection", []).append(row))
    mission_id = "stress_" + uuid4().hex

    def client(role, *, uav_id=None, assignment_id=None, max_tokens=None):
        raw = factory.for_role(role, fleet_mission_id=mission_id, uav_id=uav_id, assignment_id=assignment_id)
        return _TraceClient(raw, role.value, report["model_calls"], persist, max_tokens=max_tokens, uav_id=uav_id)

    uav_ids = [u.id for u in config.uavs]
    stage = {}
    try:
        uav_aliases = {alias: u.id for u in config.uavs for alias in (u.id, u.display_name) if alias}
        target_aliases = {alias: t.id for t in config.targets for alias in (t.id, t.semantic_alias) if alias}
        interpreter = LLMFleetTaskInterpreter(client(ModelCallRole.MISSION_INTERPRETATION),
            uav_alias_catalog=uav_aliases, target_alias_catalog=target_aliases,
            max_tokens=args.interpreter_max_tokens, repair_budget=args.interpreter_repairs)
        stage = report["stages"]["interpreter"] = {}
        try:
            task_spec = interpreter.interpret(case["instruction"])
            stage["output"] = task_spec.to_dict()
        finally:
            _capture(stage, interpreter)
            _stage_scores(stage, score_interpretation(stage.get("output"), gold, uav_ids),
                          score_interpretation(_first_proposal(stage), gold, uav_ids))
            persist()
        request = build_fleet_mission_request_v2(config, task_spec, fleet_mission_id=mission_id)
        fleet = LLMFleetPlannerV2(client(ModelCallRole.FLEET_PLAN), max_tokens=args.fleet_max_tokens,
                                  repair_budget=args.fleet_repairs)
        stage = report["stages"]["fleet"] = {}
        try:
            plan = fleet.plan(request)
            stage["output"] = plan.to_dict()
            stage["semantic_findings"] = [{"code": f.code, "message": f.message} for f in fleet.last_semantic_findings]
        finally:
            _capture(stage, fleet)
            _stage_scores(stage, score_assignment(stage.get("output"), task_spec.to_dict(), gold),
                          score_assignment(_first_proposal(stage), task_spec.to_dict(), gold))
            persist()
        contexts = build_agent_world_contexts_v2(config, request, plan)
        limits = PlannerLimits.from_config(config.planner)
        policy = PlannerPolicy.from_config(config.planner, limits)
        locals_report = report["stages"]["local"] = {}
        # Reuse the production bounded, sanitized repair findings, never gold feedback.
        from scripts.run_fleet_mission import _local_proposal_repair_findings
        for assignment in plan.assignments:
            stage = locals_report[assignment.uav_id] = {"attempts": [], "first_pass": False, "final_pass": False}
            local = DynamicLLMPlanner(client(ModelCallRole.AGENT_SPATIAL_PLAN, uav_id=assignment.uav_id,
                assignment_id=assignment.assignment_id, max_tokens=args.local_max_tokens),
                _ROOT / "prompts/dynamic_skill_planner_v3_system.txt", planner_limits=limits,
                planner_policy=policy, planning_contract="v3", repair_budget=0)
            compiler = FleetAssignmentCompiler({assignment.uav_id: local}, validator=PlanValidator(limits, policy))
            semantic_findings, proposal_findings = (), ()
            for attempt_index in range(args.local_repairs + 1):
                attempt = {"attempt_index": attempt_index, "repair": attempt_index > 0}
                stage["attempts"].append(attempt)
                try:
                    result = compiler.compile_assignment_v2(request, plan, assignment, contexts[assignment.uav_id],
                        local_plan_version=attempt_index + 1, target_catalog=build_target_catalog(config),
                        semantic_repair_findings=semantic_findings, proposal_repair_findings=proposal_findings)
                    attempt.update({"output": result.planner_output.to_dict(),
                        "goal_coverage": result.goal_coverage.to_dict(), "validation_report": result.validation_report.to_dict(),
                        "compiled_task_plan": None if result.compiled_mission is None else result.compiled_mission.task_plan.to_dict(),
                        "passed": result.compiled_mission is not None and result.goal_coverage.complete and result.semantically_valid})
                    semantic_findings = tuple({"code": f.code.value, "goal_id": f.goal_id, "message": f.message[:512]}
                        for f in result.goal_coverage.findings if f.severity.value == "RECOVERABLE_SEMANTIC_ERROR")[:32]
                    proposal_findings = () if result.compiled_mission is not None else _local_proposal_repair_findings(
                        planner=local, validation_findings=result.validation_report.findings)
                    if proposal_findings:
                        semantic_findings = ()
                except Exception as exc:
                    attempt.update({"error_type": type(exc).__name__, "passed": False})
                    semantic_findings, proposal_findings = (), _local_proposal_repair_findings(planner=local)
                finally:
                    _capture(attempt, local)
                    stage["first_pass"] = bool(stage["attempts"][0].get("passed"))
                    stage["final_pass"] = bool(attempt.get("passed"))
                    persist()
                if attempt.get("passed"):
                    break
        report["status"] = "completed"
    except Exception as exc:
        stage["error_type"] = type(exc).__name__
        report["error_type"] = type(exc).__name__
        report["status"] = "failed"
    finally:
        _overall_scores(report)
        calls = report["model_calls"]
        report["service_failure"] = any(c.get("error_type") for c in calls)
        report["output_token_limit_hit"] = any(c.get("finish_reason") == "length" for c in calls)
        report["totals"] = {"model_calls": len(calls), "latency_s": sum(c["latency_s"] for c in calls),
            "prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in calls),
            "completion_tokens": sum(c.get("usage", {}).get("completion_tokens", 0) for c in calls)}
        persist()
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="case_ids", help="repeat to select cases")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=_ROOT.parent / "outputs/fleet_planning_stress")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--interpreter-max-tokens", type=int, default=6144)
    parser.add_argument("--fleet-max-tokens", type=int, default=2048)
    parser.add_argument("--local-max-tokens", type=int, default=1024)
    parser.add_argument("--interpreter-repairs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--fleet-repairs", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--local-repairs", type=int, choices=(0, 1, 2), default=2)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if not 1 <= args.repeat <= 20:
            raise ValueError("repeat must be within 1..20")
        if not isfinite(args.timeout_s) or not 0 < args.timeout_s <= 600:
            raise ValueError("timeout-s must be within (0,600]")
        if any(not 256 <= value <= 8192 for value in
               (args.interpreter_max_tokens, args.fleet_max_tokens, args.local_max_tokens)):
            raise ValueError("token budgets must be within 256..8192")
        cases = load_suite(args.suite)
        if args.case_ids:
            unknown = set(args.case_ids) - {c["id"] for c in cases}
            if unknown:
                raise ValueError("unknown case IDs: " + ", ".join(sorted(unknown)))
            cases = [c for c in cases if c["id"] in args.case_ids]
        if len(cases) * args.repeat > 100:
            raise ValueError("at most 100 case runs are allowed per invocation")
        registry = AdapterRegistry(args.adapter_config)
        if args.model is not None and args.model != registry.base_model_name:
            raise ValueError("--model must match adapter config base_model.served_model_name")
        if args.validate_only:
            print(json.dumps({"validated": True, "case_ids": [c["id"] for c in cases],
                "model_calls": 0, "model": registry.base_model_name}, ensure_ascii=False))
            return 0
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        root = args.output_root.expanduser().resolve() / run_id
        root.mkdir(parents=True, exist_ok=False)
        summaries = []
        for case in cases:
            for repeat_index in range(args.repeat):
                directory = root / f"{case['id']}_{repeat_index + 1:02d}"
                report = run_case(case, args, directory, registry=registry)
                row = {k: report[k] for k in ("case_id", "status", "first_pass", "final_pass", "service_failure", "output_token_limit_hit", "totals")}
                row["result_file"] = str(directory / "result.json")
                summaries.append(row)
                summary = {"run_count": len(summaries), "first_pass_rate": sum(x["first_pass"] for x in summaries) / len(summaries),
                    "final_pass_rate": sum(x["final_pass"] for x in summaries) / len(summaries), "runs": summaries,
                    "scope": "text/structured planning; no visual evaluation, simulation or flight guarantee",
                    "repeats": "temperature=0 repeats measure repeatability, not pass@k; repair calls are scored separately"}
                _write(root / "summary.json", summary)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        print(json.dumps({"summary_file": str(root / "summary.json")}, ensure_ascii=False))
        return 0 if all(row["final_pass"] for row in summaries) else 2
    except (OSError, TypeError, ValueError) as exc:
        print(f"planning stress evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
