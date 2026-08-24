#!/usr/bin/env python3
"""Pure-Python Fleet Planner -> per-agent Spatial V3 planning demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.ids import generate_routing_id  # noqa: E402
from configs.loader import load_config  # noqa: E402
from fleet.compiler import FleetAssignmentCompiler  # noqa: E402
from fleet.llm_planner_v2 import LLMFleetPlannerV2  # noqa: E402
from fleet.llm_task_interpreter import LLMFleetTaskInterpreter  # noqa: E402
from fleet.local_spatial_planner import ScriptedAssignmentSpatialPlanner  # noqa: E402
from fleet.request_builder import (  # noqa: E402
    build_agent_world_contexts,
    build_agent_world_contexts_v2,
    build_fleet_mission_request,
    build_fleet_mission_request_v2,
    build_target_catalog,
    parse_explicit_assignment_instruction,
)
from fleet.scripted_planner import ScriptedFleetPlanner  # noqa: E402
from models.adapter_registry import (  # noqa: E402
    AdapterRegistry,
    DEFAULT_ADAPTER_CONFIG,
    ModelCallRole,
)
from models.model_client_factory import ModelClientFactory  # noqa: E402
from runtime.plan_validator import PlanValidator  # noqa: E402


DEFAULT_INSTRUCTION = (
    "无人机A前往世界坐标二十、三十附近十五米范围搜索并跟踪目标i二十秒；"
    "无人机B前往世界坐标负二十五、十附近十二米范围搜索并跟踪目标j二十秒；"
    "完成后分别返回各自起点降落"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_PROJECT_ROOT / "configs/multi_uav_demo.yaml")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--fleet-planner", choices=("scripted", "llm"), default="scripted")
    parser.add_argument(
        "--mission-interpreter",
        choices=("scripted", "llm"),
        default=None,
        help="defaults to the selected Fleet Planner contract",
    )
    parser.add_argument(
        "--local-planner",
        choices=("dynamic_scripted", "dynamic_llm"),
        default=None,
        help="defaults to dynamic_llm for Fleet llm, otherwise dynamic_scripted",
    )
    parser.add_argument("--planning-contract", choices=("v3",), default="v3")
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--json-output", action="store_true")
    return parser


def _effective_interpreter(args: argparse.Namespace) -> str:
    selected = args.mission_interpreter or args.fleet_planner
    if selected != args.fleet_planner:
        raise ValueError(
            "--mission-interpreter must match --fleet-planner: the scripted "
            "baseline uses Fleet v1 while the LLM pipeline uses Fleet v2"
        )
    return selected


def _effective_local_planner(args: argparse.Namespace) -> str:
    expected = (
        "dynamic_llm" if args.fleet_planner == "llm" else "dynamic_scripted"
    )
    selected = args.local_planner or expected
    if selected != expected:
        raise ValueError(
            f"--fleet-planner {args.fleet_planner} requires "
            f"--local-planner {expected}; cross-contract fallback is disabled"
        )
    return selected


def _alias_catalogs(config: object) -> tuple[dict[str, str], dict[str, str]]:
    uav_aliases: dict[str, str] = {}
    for uav in config.uavs:
        uav_aliases[uav.id] = uav.id
        if uav.display_name:
            uav_aliases[uav.display_name] = uav.id
    target_aliases: dict[str, str] = {}
    for target in config.targets:
        target_aliases[target.id] = target.id
        if target.semantic_alias:
            target_aliases[target.semantic_alias] = target.id
    return uav_aliases, target_aliases


def _diagnostics_payload(planner: object) -> dict[str, object]:
    source = getattr(planner, "source", type(planner).__name__)
    diagnostics = getattr(planner, "last_diagnostics", None)
    if diagnostics is None:
        return {
            "source": source,
            "model_calls": 0,
            "structured_output_enabled": False,
            "final_output_valid": True,
        }
    return {"source": source, **diagnostics.to_dict()}


def _semantic_finding_payload(finding: object) -> dict[str, object]:
    return {
        "code": finding.code,
        "message": finding.message,
        "constraint_id": finding.constraint_id,
        "goal_id": finding.goal_id,
        "assignment_id": finding.assignment_id,
    }


def _run_scripted_v1(
    args: argparse.Namespace,
    config: object,
    registry: AdapterRegistry,
    selection_records: list[dict[str, object]],
) -> dict[str, object]:
    directives = parse_explicit_assignment_instruction(args.instruction, config)
    request = build_fleet_mission_request(
        config,
        args.instruction,
        directives=directives,
    )
    interpreter_selection = registry.resolve(ModelCallRole.MISSION_INTERPRETATION)
    selection_records.append({**interpreter_selection.to_dict(), "used": False})

    fleet_planner = ScriptedFleetPlanner()
    fleet_selection = registry.resolve(ModelCallRole.FLEET_PLAN)
    selection_records.append({**fleet_selection.to_dict(), "used": False})
    fleet_plan = fleet_planner.plan(request)

    contexts = build_agent_world_contexts(config, fleet_plan)
    local_planners: dict[str, object] = {}
    if args.local_planner == "dynamic_scripted":
        for assignment in fleet_plan.assignments:
            local_planners[assignment.uav_id] = ScriptedAssignmentSpatialPlanner()
        local_selection = registry.resolve(ModelCallRole.AGENT_SPATIAL_PLAN)
        selection_records.append({**local_selection.to_dict(), "used": False})
    else:
        raise ValueError(
            "the scripted Fleet v1 baseline requires "
            "--local-planner dynamic_scripted"
        )
    compiler = FleetAssignmentCompiler(local_planners, validator=PlanValidator())
    compilations = compiler.compile(request, fleet_plan, contexts)

    local_plans: dict[str, object] = {}
    local_diagnostics: dict[str, object] = {}
    for uav_id, result in sorted(compilations.items()):
        compiled = result.compiled_mission
        local_plans[uav_id] = {
            "agent_planner_request": result.agent_request.to_dict(),
            "spatial_plan_draft_v3": result.planner_output.to_dict(),
            "compiled_task_plan": None if compiled is None else compiled.task_plan.to_dict(),
        }
        local_diagnostics[uav_id] = _diagnostics_payload(local_planners[uav_id])

    return {
        "fleet_task_spec": None,
        "fleet_mission_plan": fleet_plan.to_dict(),
        "assignment_summary": [
            {
                "assignment_id": item.assignment_id,
                "uav_id": item.uav_id,
                "target_alias": item.target_alias,
                "search_region": item.search_region.to_dict(),
                "track_duration_s": item.track_duration_s,
                "priority": item.priority,
                "start_policy": item.start_policy.value,
            }
            for item in fleet_plan.assignments
        ],
        "per_agent_local_plan": local_plans,
        "adapter_selection": selection_records,
        "mission_interpreter_diagnostics": {
            "source": "fixed_explicit_assignment_parser",
            "model_calls": 0,
            "structured_output_enabled": False,
            "final_output_valid": True,
        },
        "fleet_planner_diagnostics": _diagnostics_payload(fleet_planner),
        "fleet_plan_semantic_findings": [],
        "per_agent_planner_diagnostics": local_diagnostics,
    }


def _run_llm_v2(
    args: argparse.Namespace,
    config: object,
    client_factory: ModelClientFactory,
    selection_records: list[dict[str, object]],
) -> dict[str, object]:
    if args.local_planner != "dynamic_llm":
        raise ValueError(
            "the LLM Fleet v2 pipeline requires --local-planner dynamic_llm; "
            "dynamic_scripted is the fixed Fleet v1 baseline"
        )
    uav_aliases, target_aliases = _alias_catalogs(config)
    fleet_mission_id = generate_routing_id("fleet_mission")
    interpreter = LLMFleetTaskInterpreter(
        client_factory.for_role(
            ModelCallRole.MISSION_INTERPRETATION,
            fleet_mission_id=fleet_mission_id,
        ),
        uav_alias_catalog=uav_aliases,
        target_alias_catalog=target_aliases,
    )
    task_spec = interpreter.interpret(args.instruction)
    request = build_fleet_mission_request_v2(
        config,
        task_spec,
        fleet_mission_id=fleet_mission_id,
    )

    fleet_planner = LLMFleetPlannerV2(
        client_factory.for_role(
            ModelCallRole.FLEET_PLAN,
            fleet_mission_id=request.fleet_mission_id,
        )
    )
    fleet_plan = fleet_planner.plan(request)
    semantic_findings = fleet_plan.semantic_findings(request)
    contexts = build_agent_world_contexts_v2(config, request, fleet_plan)

    from planner.dynamic_llm_planner import DynamicLLMPlanner

    prompt = _PROJECT_ROOT / "prompts/dynamic_skill_planner_v3_system.txt"
    local_planners: dict[str, object] = {}
    for assignment in fleet_plan.assignments:
        local_planners[assignment.uav_id] = DynamicLLMPlanner(
            client_factory.for_role(
                ModelCallRole.AGENT_SPATIAL_PLAN,
                fleet_mission_id=request.fleet_mission_id,
                assignment_id=assignment.assignment_id,
                uav_id=assignment.uav_id,
            ),
            system_prompt_path=prompt,
            planning_contract="v3",
        )

    compilations: dict[str, object] = {}
    if local_planners:
        compiler = FleetAssignmentCompiler(local_planners, validator=PlanValidator())
        compilations = compiler.compile_v2(
            request,
            fleet_plan,
            contexts,
            target_catalog=build_target_catalog(config),
        )

    local_plans: dict[str, object] = {}
    local_diagnostics: dict[str, object] = {}
    for uav_id, result in sorted(compilations.items()):
        compiled = result.compiled_mission
        local_plans[uav_id] = {
            "agent_planner_request": result.agent_request.to_dict(),
            "spatial_plan_draft_v3": result.planner_output.to_dict(),
            "compiled_task_plan": None if compiled is None else compiled.task_plan.to_dict(),
            "goal_coverage": result.goal_coverage.to_dict(),
            "validation_report": result.validation_report.to_dict(),
        }
        local_diagnostics[uav_id] = _diagnostics_payload(local_planners[uav_id])

    interpreter_diagnostics = _diagnostics_payload(interpreter)
    interpreter_diagnostics["model_proposals"] = list(interpreter.model_proposals)
    fleet_diagnostics = _diagnostics_payload(fleet_planner)
    fleet_diagnostics["model_proposals"] = list(fleet_planner.model_proposals)
    return {
        "fleet_task_spec": task_spec.to_dict(),
        "fleet_mission_plan": fleet_plan.to_dict(),
        "assignment_summary": [
            {
                "assignment_id": item.assignment_id,
                "uav_id": item.uav_id,
                "goal_ids": list(item.goal_ids),
                "priority": item.priority,
                "start_policy": item.start_policy.value,
                "deviations": [value.to_dict() for value in item.deviations],
            }
            for item in fleet_plan.assignments
        ],
        "per_agent_local_plan": local_plans,
        "adapter_selection": selection_records,
        "mission_interpreter_diagnostics": interpreter_diagnostics,
        "fleet_planner_diagnostics": fleet_diagnostics,
        "fleet_plan_semantic_findings": [
            _semantic_finding_payload(item) for item in semantic_findings
        ],
        "per_agent_planner_diagnostics": local_diagnostics,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    interpreter_mode = _effective_interpreter(args)
    args.local_planner = _effective_local_planner(args)
    registry = AdapterRegistry(args.adapter_config)
    if args.model is not None and args.model != registry.base_model_name:
        raise ValueError(
            "--model must match configs/adapters.json base_model.served_model_name"
        )
    selection_records: list[dict[str, object]] = []
    client_factory = ModelClientFactory(
        registry,
        base_url=args.base_url,
        api_key=args.api_key,
        selection_logger=lambda value: selection_records.append(dict(value)),
    )
    if interpreter_mode == "scripted":
        return _run_scripted_v1(args, config, registry, selection_records)
    return _run_llm_v2(
        args,
        config,
        client_factory,
        selection_records,
    )


def _print_sections(result: dict[str, object]) -> None:
    labels = (
        ("FleetMissionPlan", "fleet_mission_plan"),
        ("Assignment summary", "assignment_summary"),
        ("Per-agent local plan", "per_agent_local_plan"),
        ("Adapter selection", "adapter_selection"),
        ("Fleet Planner diagnostics", "fleet_planner_diagnostics"),
        ("Per-agent planner diagnostics", "per_agent_planner_diagnostics"),
        ("FleetTaskSpec", "fleet_task_spec"),
        ("Mission Interpreter diagnostics", "mission_interpreter_diagnostics"),
        ("Fleet Plan semantic findings", "fleet_plan_semantic_findings"),
    )
    for label, key in labels:
        print(f"=== {label} ===")
        print(json.dumps(result[key], ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(
            f"fleet planner demo failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        _print_sections(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
