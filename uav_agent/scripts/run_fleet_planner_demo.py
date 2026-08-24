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

from configs.loader import load_config  # noqa: E402
from fleet.compiler import FleetAssignmentCompiler  # noqa: E402
from fleet.llm_planner import LLMFleetPlanner  # noqa: E402
from fleet.local_spatial_planner import ScriptedAssignmentSpatialPlanner  # noqa: E402
from fleet.request_builder import (  # noqa: E402
    FleetRequestBuildError,
    build_agent_world_contexts,
    build_fleet_mission_request,
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
        "--local-planner",
        choices=("dynamic_scripted", "dynamic_llm"),
        default="dynamic_scripted",
    )
    parser.add_argument("--planning-contract", choices=("v3",), default="v3")
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--json-output", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
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

    directives = None
    try:
        directives = parse_explicit_assignment_instruction(args.instruction, config)
    except FleetRequestBuildError:
        if args.fleet_planner == "scripted":
            raise
    request = build_fleet_mission_request(
        config,
        args.instruction,
        directives=directives,
    )

    if args.fleet_planner == "scripted":
        fleet_planner = ScriptedFleetPlanner()
        fleet_selection = registry.resolve(ModelCallRole.FLEET_PLAN)
        selection_records.append({**fleet_selection.to_dict(), "used": False})
        fleet_diagnostics: dict[str, object] = {
            "source": fleet_planner.source,
            "model_calls": 0,
            "structured_output": False,
            "final_output_valid": True,
        }
    else:
        fleet_planner = LLMFleetPlanner(
            client_factory.for_role(ModelCallRole.FLEET_PLAN)
        )
        fleet_diagnostics = {
            "source": fleet_planner.source,
            "model_calls": 1,
            "structured_output": True,
        }
    fleet_plan = fleet_planner.plan(request)
    if args.fleet_planner == "llm":
        fleet_diagnostics.update(
            {
                "response_text_length": fleet_planner.last_response_text_length,
                "final_output_valid": True,
            }
        )

    contexts = build_agent_world_contexts(config, fleet_plan)
    local_planners: dict[str, object] = {}
    if args.local_planner == "dynamic_scripted":
        for assignment in fleet_plan.assignments:
            local_planners[assignment.uav_id] = ScriptedAssignmentSpatialPlanner()
        local_selection = registry.resolve(ModelCallRole.AGENT_SPATIAL_PLAN)
        selection_records.append({**local_selection.to_dict(), "used": False})
    else:
        from planner.dynamic_llm_planner import DynamicLLMPlanner

        prompt = _PROJECT_ROOT / "prompts/dynamic_skill_planner_v3_system.txt"
        for assignment in fleet_plan.assignments:
            local_planners[assignment.uav_id] = DynamicLLMPlanner(
                client_factory.for_role(ModelCallRole.AGENT_SPATIAL_PLAN),
                system_prompt_path=prompt,
                planning_contract="v3",
            )
    compiler = FleetAssignmentCompiler(
        local_planners,
        validator=PlanValidator(),
    )
    compilations = compiler.compile(request, fleet_plan, contexts)

    local_plans: dict[str, object] = {}
    local_diagnostics: dict[str, object] = {}
    for uav_id, result in sorted(compilations.items()):
        compiled = result.compiled_mission
        local_plans[uav_id] = {
            "agent_planner_request": result.agent_request.to_dict(),
            "spatial_plan_draft_v3": result.planner_output.to_dict(),
            "compiled_task_plan": (
                None if compiled is None else compiled.task_plan.to_dict()
            ),
        }
        planner = local_planners[uav_id]
        diagnostics = getattr(planner, "last_diagnostics", None)
        if diagnostics is not None:
            local_diagnostics[uav_id] = diagnostics.to_dict()
        else:
            local_diagnostics[uav_id] = {
                "source": getattr(planner, "source", "unknown"),
                "model_calls": 0,
                "final_output_valid": True,
            }

    return {
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
        "fleet_planner_diagnostics": fleet_diagnostics,
        "per_agent_planner_diagnostics": local_diagnostics,
    }


def _print_sections(result: dict[str, object]) -> None:
    labels = (
        ("FleetMissionPlan", "fleet_mission_plan"),
        ("Assignment summary", "assignment_summary"),
        ("Per-agent local plan", "per_agent_local_plan"),
        ("Adapter selection", "adapter_selection"),
        ("Fleet Planner diagnostics", "fleet_planner_diagnostics"),
        ("Per-agent planner diagnostics", "per_agent_planner_diagnostics"),
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
