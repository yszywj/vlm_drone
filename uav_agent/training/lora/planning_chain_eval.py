"""Real-generation evaluation of the complete three-role planning chain.

Only the original instruction and trusted inventory enter the chain. Each
downstream prompt is built from the preceding model's actual accepted output.
Blueprint auditors run out of band: their findings never repair model inputs.
No simulator is started and no compiled plan is dispatched.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from fleet.compiler import FleetAssignmentCompiler
from fleet.llm_planner_v2 import LLMFleetPlannerV2
from fleet.llm_task_interpreter import LLMFleetTaskInterpreter
from models.base import GenerationOptions, ModelResponse
from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.schemas import LandingZoneSpec, PlannerWorldContext
from planning_data.local_gold import validate_local_against_blueprint
from planning_data.tasks import build_fleet_request
from planning_data.validation import validate_semantic_gold
from target.types import TargetSpec


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_TOKENS = {
    "mission_interpreter": 6144,
    "fleet_planner": 2048,
    "spatial_mission": 1024,
}
_FLEET_ONLY_CODES = {"UNEXPECTED_DEVIATION", "UNEXPECTED_FLEET_ASSUMPTION"}


def _fleet_finding(finding: Mapping) -> bool:
    code = str(finding["code"])
    return code.startswith("FLEET_") or code in _FLEET_ONLY_CODES


def _error(exc: Exception) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)}


def prepare_evaluation_options(client, options: GenerationOptions) -> GenerationOptions:
    """Observe the production client's effective options before recording them."""
    prepare = getattr(client, "prepare_options", None)
    effective = prepare(options) if callable(prepare) else options
    if not isinstance(effective, GenerationOptions):
        raise TypeError("effective evaluation options must be GenerationOptions")
    return effective


def generation_options_audit(options: GenerationOptions) -> dict:
    """Keep semantic and order-sensitive schema identities in generated results.

    The wire identity hashes the response_format.json_schema HTTP subtree,
    using the production client's compact ASCII-escaped JSON in insertion
    order. It excludes messages and the rest of the HTTP request. The JSON
    string survives downstream sort_keys serialization.
    """
    schema = options.response_format
    response_format = None if schema is None else schema.to_dict()
    generation = {
        "temperature": options.temperature,
        "top_p": options.top_p,
        "max_tokens": options.max_tokens,
        "response_format": response_format,
    }
    schema_json = None if response_format is None else json.dumps(
        response_format, ensure_ascii=True, allow_nan=False, separators=(",", ":"),
    )
    return {
        **generation,
        "response_schema_name": None if schema is None else schema.name,
        "response_schema_sha256": None if response_format is None else sha256(
            json.dumps(response_format, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "response_schema_wire_sha256": None if schema_json is None else sha256(
            schema_json.encode("utf-8")
        ).hexdigest(),
        "generation_options_json": json.dumps(
            generation, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ),
    }


class RecordingEvaluationClient:
    """Observe actual calls and apply the common comparison output budget."""

    def __init__(self, client, role: str, max_tokens: int, calls: list, *, uav_id=None):
        self.client = client
        self.role = role
        self.max_tokens = max_tokens
        self.calls = calls
        self.uav_id = uav_id

    def chat(self, messages, *, options):
        actual_options = prepare_evaluation_options(
            self.client, replace(options, max_tokens=self.max_tokens)
        )
        call = {
            "call_index": len(self.calls), "role": self.role,
            "uav_id": self.uav_id,
            "messages": [message.to_dict() for message in messages],
            "options": generation_options_audit(actual_options),
            "response": None, "error": None,
        }
        self.calls.append(call)
        start = perf_counter()
        try:
            response = self.client.chat(messages, options=actual_options)
            if not isinstance(response, ModelResponse):
                raise TypeError("model client did not return ModelResponse")
            call["response"] = {
                "content": response.content, "model": response.model,
                "finish_reason": response.finish_reason,
                "usage": dict(response.usage),
            }
            return response
        except Exception as exc:
            call["error"] = _error(exc)
            raise
        finally:
            call["elapsed_s"] = perf_counter() - start


def _world(task: dict, uav: dict) -> PlannerWorldContext:
    world = task["world"]
    home = tuple(uav["home_xyz_m"])
    name = uav["home_name"]
    return PlannerWorldContext(
        scene_min_xyz_m=tuple(world["scene_min_xyz_m"]),
        scene_max_xyz_m=tuple(world["scene_max_xyz_m"]),
        initial_uav_xyz_m=home, search_regions={},
        landing_zones={name: LandingZoneSpec(name, home[:2], home[2])},
        default_takeoff_altitude_m=world["flight_altitude_m"],
        default_track_duration_s=30.0, search_timeout_s=75.0,
    )


def _empty_stage() -> dict:
    return {
        "status": "blocked", "production_accepted": False,
        "blueprint_pass": False, "findings": [], "error": None,
        "output": None, "diagnostics": None,
    }


def run_chain(
    task: dict,
    clients: Mapping,
    *,
    max_tokens: Mapping | None = None,
    repair_budget: int = 0,
) -> dict:
    """Generate and independently score one original mission end to end.

    ``clients`` maps the three role names in ``DEFAULT_MAX_TOKENS`` to model
    clients. A structurally accepted but semantically wrong interpretation or
    assignment still continues down the real chain. Unparseable output blocks
    dependent stages; blocked UAVs remain in the expected-UAV denominator.
    ``strict_entire_task_pass`` requires original intent, Fleet ownership and
    every UAV's raw and compiled plan to pass, without compiler-added closure.
    """
    # Share exactly the interpreter auditor used by the role-isolation run.
    from training.lora.planning_role_eval import audit_interpreted_spec

    if not isinstance(clients, Mapping) or any(
        not callable(getattr(clients.get(role), "chat", None))
        for role in DEFAULT_MAX_TOKENS
    ):
        raise TypeError("clients must contain a model client for all three planning roles")
    budgets = dict(DEFAULT_MAX_TOKENS)
    if max_tokens is not None:
        if not isinstance(max_tokens, Mapping) or set(max_tokens) - set(budgets):
            raise ValueError("max_tokens contains an unknown planning role")
        budgets.update(max_tokens)
    for role, value in budgets.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 8192:
            raise ValueError(f"max_tokens for {role} must be an integer in [256, 8192]")
    if isinstance(repair_budget, bool) or repair_budget not in (0, 1):
        raise ValueError("repair_budget must be 0 or 1 for all three production roles")

    start = perf_counter()
    calls: list[dict] = []
    events: list[dict] = []
    stages = {"interpreter": _empty_stage(), "fleet": _empty_stage()}
    expected = {assignment["uav_id"]: assignment for assignment in task["assignments"]}
    uavs = {uav["uav_id"]: uav for uav in task["uavs"]}
    local_results = {
        owner: {
            **_empty_stage(), "uav_id": owner, "assignment_id": None,
            "raw_blueprint_pass": False, "compiled": False,
            "goal_coverage_complete": False, "passed": False,
            "runtime_safety_completion_allowed": None,
            "runtime_safety_completion_added": False,
            "runtime_contract_closure": assignment.get("closure_policy") == "runtime_contract_home_and_land",
        }
        for owner, assignment in expected.items()
    }

    def event(stage, status, **details):
        events.append({"stage": stage, "status": status, "elapsed_s": perf_counter() - start, **details})

    def finish():
        usage = Counter()
        usage_by_role = {role: Counter() for role in budgets}
        for call in calls:
            response = call["response"]
            if response:
                usage.update(response["usage"])
                usage_by_role[call["role"]].update(response["usage"])
        rows = list(local_results.values())
        for row in rows:
            if row["status"] == "blocked" and "blocked_reason" not in row:
                row["blocked_reason"] = (
                    "interpreter_output_unavailable" if not stages["interpreter"]["production_accepted"]
                    else "fleet_output_unavailable"
                )
        pass_with_completion = (
            stages["interpreter"]["blueprint_pass"]
            and stages["fleet"]["blueprint_pass"]
            and set(local_results) == set(expected)
            and all(row["passed"] for row in rows)
        )
        return {
            "evaluation_schema_version": 1, "task_id": task["task_id"],
            "family": task["family"], "scale": task["scale"],
            "split": task.get("split"), "semantic_hash": task.get("semantic_hash"),
            "instruction": task["instruction"], "max_tokens": budgets,
            "repair_budget": repair_budget, "stages": stages,
            "local_results": rows, "expected_local_count": len(expected),
            "local_pass_count": sum(row["passed"] for row in rows),
            "local_blocked_count": sum(row["status"] == "blocked" for row in rows),
            "strict_entire_task_pass": bool(pass_with_completion and not any(
                row["runtime_safety_completion_added"] for row in rows
            )),
            "entire_task_pass_with_runtime_completion": bool(pass_with_completion),
            "calls": calls, "events": events, "usage": dict(usage),
            "usage_by_role": {role: dict(values) for role, values in usage_by_role.items()},
            "model_call_count": len(calls),
            "truncated_call_count": sum(
                call["response"] is not None and call["response"]["finish_reason"] == "length"
                for call in calls
            ),
            "client_error_count": sum(call["error"] is not None for call in calls),
            "elapsed_s": perf_counter() - start,
        }

    interpreter = LLMFleetTaskInterpreter(
        RecordingEvaluationClient(clients["mission_interpreter"], "mission_interpreter", budgets["mission_interpreter"], calls),
        uav_alias_catalog=task["uav_aliases"], target_alias_catalog=task["target_aliases"],
        max_tokens=budgets["mission_interpreter"], repair_budget=repair_budget,
    )
    event("interpreter", "started")
    try:
        spec = interpreter.interpret(task["instruction"])
        stages["interpreter"].update(status="completed", production_accepted=True, output=spec.to_dict())
    except Exception as exc:
        stages["interpreter"].update(status="failed", error=_error(exc))
        event("interpreter", "failed", error=_error(exc))
        return finish()
    finally:
        diagnostics = interpreter.last_diagnostics
        stages["interpreter"]["diagnostics"] = None if diagnostics is None else diagnostics.to_dict()
    audit = audit_interpreted_spec(task, spec)
    stages["interpreter"].update(blueprint_pass=audit["passed"], findings=audit["findings"])
    event("interpreter", "completed", blueprint_pass=audit["passed"])

    # Explicitly pass the MODEL spec; omitting this argument would use gold.
    try:
        request = build_fleet_request(task, spec)
    except Exception as exc:
        stages["fleet"].update(error=_error(exc), blocked_reason="actual_interpretation_cannot_form_trusted_request")
        event("fleet", "blocked", error=_error(exc))
        return finish()
    fleet = LLMFleetPlannerV2(
        RecordingEvaluationClient(clients["fleet_planner"], "fleet_planner", budgets["fleet_planner"], calls),
        max_tokens=budgets["fleet_planner"], repair_budget=repair_budget,
    )
    event("fleet", "started")
    try:
        plan = fleet.plan(request)
        stages["fleet"].update(status="completed", production_accepted=True, output=plan.to_dict())
    except Exception as exc:
        stages["fleet"].update(status="failed", error=_error(exc))
        event("fleet", "failed", error=_error(exc))
        return finish()
    finally:
        diagnostics = fleet.last_diagnostics
        stages["fleet"]["diagnostics"] = None if diagnostics is None else diagnostics.to_dict()
    audit = validate_semantic_gold(task, spec, plan)
    fleet_findings = [finding for finding in audit["findings"] if _fleet_finding(finding)]
    production_findings = [asdict(finding) for finding in fleet.last_semantic_findings]
    stages["fleet"].update(
        blueprint_pass=not fleet_findings and not production_findings,
        findings=fleet_findings, production_semantic_findings=production_findings,
        whole_chain_semantics=audit,
    )
    event("fleet", "completed", blueprint_pass=stages["fleet"]["blueprint_pass"])
    catalog = {alias: TargetSpec.from_dict(value) for alias, value in task["target_catalog"].items()}
    for assignment in plan.assignments:
        owner = assignment.uav_id
        row = local_results[owner]
        row["assignment_id"] = assignment.assignment_id
        client = RecordingEvaluationClient(clients["spatial_mission"], "spatial_mission", budgets["spatial_mission"], calls, uav_id=owner)
        local = DynamicLLMPlanner(
            client, ROOT / "prompts/dynamic_skill_planner_v3_system.txt",
            planning_contract="v3", repair_budget=repair_budget,
        )
        event("local", "started", uav_id=owner)
        calls_before_local = len(calls)
        try:
            compiled = FleetAssignmentCompiler(local).compile_assignment_v2(
                request, plan, assignment, _world(task, uavs[owner]), target_catalog=catalog,
            )
            raw = compiled.planner_output.to_dict()
            executable = None if compiled.compiled_mission is None else compiled.compiled_mission.task_plan.to_dict()
            allow_completion = compiled.planner_request.allow_trusted_safety_completion
            # Independently observe append behavior even when a wrong upstream
            # spec incorrectly enabled completion for an explicitly requested return.
            steps = [] if executable is None else executable["steps"]
            added = bool(allow_completion and len(steps) == len(raw["steps"]) + 2
                         and [step["skill"] for step in steps[-2:]] == ["GOTO", "LAND"])
            audit = validate_local_against_blueprint(
                {**expected[owner], "home_name": uavs[owner]["home_name"]},
                compiled.planner_output,
                None if compiled.compiled_mission is None else compiled.compiled_mission.task_plan,
                tuple(uavs[owner]["home_xyz_m"]),
                allow_safety_completion=allow_completion,
                target_spec=catalog.get(expected[owner].get("target_alias")),
            )
            row.update(
                status="completed", production_accepted=True, output=raw,
                compiled=compiled.compiled_mission is not None,
                compiled_output=executable,
                goal_coverage_complete=compiled.goal_coverage.complete,
                goal_coverage=compiled.goal_coverage.to_dict(),
                validation_report=compiled.validation_report.to_dict(),
                assigned_semantics_pass=compiled.semantically_valid,
                blueprint_pass=audit["passed"], findings=audit["findings"],
                raw_blueprint_pass=not any(item.get("stage") in {"draft", "blueprint"} for item in audit["findings"]),
                runtime_safety_completion_allowed=allow_completion,
                runtime_safety_completion_added=added,
                passed=bool(compiled.compiled_mission is not None and compiled.semantically_valid and audit["passed"]),
            )
            event("local", "completed", uav_id=owner, passed=row["passed"])
        except Exception as exc:
            # An invalid assignment or missing trusted world value may be
            # rejected while constructing the focused request, before the
            # spatial model was called. Do not attribute that to its output.
            status = "blocked" if len(calls) == calls_before_local else "failed"
            row.update(status=status, error=_error(exc))
            if status == "blocked":
                row["blocked_reason"] = "assignment_cannot_form_local_request"
            event("local", status, uav_id=owner, error=_error(exc))
        finally:
            diagnostics = local.last_diagnostics
            row["diagnostics"] = None if diagnostics is None else diagnostics.to_dict()
    for owner, row in local_results.items():
        if row["status"] == "blocked" and "blocked_reason" not in row:
            row["blocked_reason"] = "no_assignment_for_expected_uav"
            event("local", "blocked", uav_id=owner, reason=row["blocked_reason"])
    return finish()


__all__ = [
    "DEFAULT_MAX_TOKENS", "RecordingEvaluationClient", "generation_options_audit",
    "prepare_evaluation_options", "run_chain",
]
