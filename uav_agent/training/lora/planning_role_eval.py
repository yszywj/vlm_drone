"""Score generated planning-role answers through production contracts.

This module performs no model inference, text repair, or label-string matching.
Each raw answer is replayed exactly once. Upstream roles use trusted gold inputs
when scoring a downstream role, so these are isolated role scores, not an
end-to-end mission success estimate. Semantic checks use the original blueprint.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from fleet.compiler import FleetAssignmentCompiler
from fleet.llm_planner_v2 import LLMFleetPlannerV2
from fleet.llm_task_interpreter import LLMFleetTaskInterpreter, FleetTaskInterpretationError
from fleet.planner_base import FleetPlannerOutputError
from fleet.task_spec import FleetTaskSpecV1
from fleet.types import FleetCoordinationPolicy
from fleet.types_v2 import FleetMissionPlanV2
from models.base import ModelResponse
from planner.base import PlannerOutputError
from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.schemas import LandingZoneSpec, PlannerWorldContext
from planning_data.local_gold import build_local_gold, validate_local_against_blueprint
from planning_data.tasks import build_fleet_plan, build_fleet_request, build_task_spec
from planning_data.validation import validate_semantic_gold
from target.types import TargetSpec


ROOT = Path(__file__).resolve().parents[2]
ROLES = ("mission_interpreter", "fleet_planner", "spatial_mission")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


class _ReplayClient:
    """Preserve the exact raw response and capture only its actual request."""

    def __init__(self, raw: str | None):
        self.raw = raw
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, *, options):
        if self.calls:
            raise AssertionError("evaluation must not request a repair response")
        if self.raw is None:
            raise AssertionError("capture requires an internal gold replay answer")
        self.calls.append({"messages": list(messages), "options": options})
        return ModelResponse(self.raw, "offline_evaluation_replay", "stop", {})


class _LocalReplayPlanner:
    source = "dynamic_llm"

    def __init__(self, client: _ReplayClient):
        self.client = client
        self.planner = DynamicLLMPlanner(
            client, ROOT / "prompts/dynamic_skill_planner_v3_system.txt",
            planning_contract="v3", repair_budget=0,
        )

    @property
    def last_diagnostics(self):
        return self.planner.last_diagnostics

    def plan(self, request):
        if self.client.raw is None:
            self.client.raw = _canonical(build_local_gold(request).to_dict())
        return self.planner.plan(request)


def world_context_for_task(task: dict, uav: dict) -> PlannerWorldContext:
    """Reconstruct the trusted, obstacle-free dataset world for one UAV."""
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


def audit_interpreted_spec(task: dict, spec: FleetTaskSpecV1) -> dict[str, object]:
    """Audit interpreter semantics without assuming canonical goal identifiers.

    The independent validator accepts a spec and Fleet plan together. An empty
    carrier lets it audit generated goals, bindings, evidence and ordering using
    its semantic matcher. Findings about this artificial carrier are excluded;
    no actual Fleet assignment quality is attributed to the interpreter.
    """
    carrier = FleetMissionPlanV2(
        fleet_mission_id=f"fleet_{task['task_id']}", fleet_plan_version=1,
        assignments=(),
        coordination_policy=FleetCoordinationPolicy(minimum_uav_separation_m=5.0),
    )
    audit = validate_semantic_gold(task, spec, carrier)
    findings = [finding for finding in audit["findings"] if not (
        finding["code"].startswith("FLEET_")
        or finding["code"] in {"UNEXPECTED_DEVIATION", "UNEXPECTED_FLEET_ASSUMPTION"}
    )]
    return {"passed": not findings, "findings": findings}


def _validate_role_args(task: dict, role: str, uav_id: str | None) -> None:
    if role not in ROLES:
        raise ValueError(f"unknown planning role: {role!r}")
    if not isinstance(task, dict):
        raise TypeError("task must be a blueprint dict")
    if role == "spatial_mission":
        if uav_id not in {uav["uav_id"] for uav in task["uavs"]}:
            raise ValueError("spatial_mission requires a UAV present in the task")
    elif uav_id is not None:
        raise ValueError("uav_id is only valid for spatial_mission")


def _evaluate(task: dict, role: str, raw: str | None, uav_id: str | None):
    _validate_role_args(task, role, uav_id)
    # Match the persisted dataset's stable alias-directory ordering.
    task = json.loads(_canonical(task))
    client = _ReplayClient(raw)
    report: dict[str, Any] = {
        "passed": False, "structural_pass": False, "semantic_pass": False,
        "findings": [], "role": role, "uav_id": uav_id,
        "replay_calls": 0, "repair_used": False,
        "conditioning": "original_instruction" if role == "mission_interpreter" else "gold_upstream",
    }
    planner = None
    try:
        if role == "mission_interpreter":
            if raw is None:
                client.raw = _canonical(build_task_spec(task).to_dict())
            planner = LLMFleetTaskInterpreter(
                client, uav_alias_catalog=task["uav_aliases"],
                target_alias_catalog=task["target_aliases"], repair_budget=0,
            )
            spec = planner.interpret(task["instruction"])
            report["structural_pass"] = True
            audit = audit_interpreted_spec(task, spec)
            report["semantic_pass"] = audit["passed"]
            report["findings"].extend(audit["findings"])
        elif role == "fleet_planner":
            request = build_fleet_request(task)
            if raw is None:
                client.raw = _canonical(build_fleet_plan(task, request).to_dict())
            planner = LLMFleetPlannerV2(client, repair_budget=0)
            plan = planner.plan(request)
            report["structural_pass"] = True
            audit = validate_semantic_gold(task, request.task_spec, plan)
            production_findings = [asdict(finding) for finding in planner.last_semantic_findings]
            report["semantic_pass"] = audit["passed"] and not production_findings
            report["findings"].extend(audit["findings"])
            report["production_semantic_findings"] = production_findings
            report["findings"].extend(production_findings)
        else:
            request = build_fleet_request(task)
            plan = build_fleet_plan(task, request)
            assignment = next(item for item in plan.assignments if item.uav_id == uav_id)
            uav = next(item for item in task["uavs"] if item["uav_id"] == uav_id)
            blueprint = next(item for item in task["assignments"] if item["uav_id"] == uav_id)
            targets = {alias: TargetSpec.from_dict(value) for alias, value in task["target_catalog"].items()}
            planner = _LocalReplayPlanner(client)
            result = FleetAssignmentCompiler(planner).compile_assignment_v2(
                request, plan, assignment, world_context_for_task(task, uav),
                target_catalog=targets,
            )
            report["structural_pass"] = True
            report["compilation_pass"] = result.compiled_mission is not None
            report["goal_coverage_pass"] = result.goal_coverage.complete
            report["production_semantic_pass"] = result.semantically_valid
            report["production_validation_report"] = result.validation_report.to_dict()
            report["findings"].extend(finding.to_dict() for finding in result.validation_report.findings)
            if result.compiled_mission is not None:
                audit = validate_local_against_blueprint(
                    blueprint, result.planner_output, result.compiled_mission.task_plan,
                    tuple(uav["home_xyz_m"]),
                    allow_safety_completion=result.planner_request.allow_trusted_safety_completion,
                    target_spec=targets.get(blueprint.get("target_alias")),
                )
                report["blueprint_semantic_pass"] = audit["passed"]
                report["findings"].extend(audit["findings"])
                for key in ("runtime_safety_completion_added", "runtime_contract_closure"):
                    report[key] = audit[key]
                report["semantic_pass"] = (
                    result.semantically_valid and result.goal_coverage.complete and audit["passed"]
                )
            else:
                report["blueprint_semantic_pass"] = False
    except (FleetTaskInterpretationError, FleetPlannerOutputError, PlannerOutputError) as exc:
        diagnostic = getattr(planner, "last_diagnostics", None)
        report["findings"].append({
            "code": getattr(diagnostic, "initial_error_code", None) or type(exc).__name__,
            "message": str(exc)[:1500],
        })
    report["replay_calls"] = len(client.calls)
    diagnostic = getattr(planner, "last_diagnostics", None)
    if diagnostic is not None:
        report["diagnostics"] = diagnostic.to_dict()
        report["repair_used"] = diagnostic.repair_used
    if len(client.calls) != 1 or report["repair_used"]:
        raise AssertionError("scoring must replay exactly one unmodified response without repair")
    report["passed"] = report["structural_pass"] and report["semantic_pass"]
    return report, client


def score_role_output(task: dict, role: str, raw: str, *, uav_id: str | None = None) -> dict[str, Any]:
    """Score one actual generated string; never substitute or repair its text."""
    if not isinstance(raw, str):
        raise TypeError("raw must be the generated response string")
    return _evaluate(task, role, raw, uav_id)[0]


def capture_role_request(task: dict, role: str, *, uav_id: str | None = None) -> dict[str, Any]:
    """Return production messages and schema without exposing an answer.

    Gold is replayed internally only to capture the real request path. The
    returned messages contain system/user inputs; the caller performs actual
    inference independently and passes its output to ``score_role_output``.
    """
    report, client = _evaluate(task, role, None, uav_id)
    if not report["passed"]:
        raise ValueError(f"internal gold request capture failed: {report}")
    captured = dict(client.calls[0])
    options = captured["options"]
    captured["response_schema_sha256"] = (
        sha256(_canonical(options.response_format.to_dict()).encode("utf-8")).hexdigest()
        if options.response_format is not None else None
    )
    return captured


__all__ = [
    "ROLES", "audit_interpreted_spec", "capture_role_request",
    "score_role_output", "world_context_for_task",
]
