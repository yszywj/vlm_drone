#!/usr/bin/env python3
"""Run the pure planning pipeline without importing or starting Isaac Sim."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


# Direct script execution places ``scripts/`` on sys.path.  Add the project
# package root so the documented ``./python.sh scripts/...`` command works.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planner.schemas import (  # noqa: E402
    CompiledMission,
    LandingZoneSpec,
    MissionIntent,
    PlannerOutput,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraft,
)
from planner.diagnostics import PlannerDiagnostics  # noqa: E402
from planner.scripted_planner import ScriptedPlanner  # noqa: E402
from runtime.plan_validator import PlanValidator  # noqa: E402


DEFAULT_INSTRUCTION = (
    "起飞后前往 search_area 搜寻移动目标，找到后跟踪三十秒，然后返回 home 降落"
)
DEFAULT_TAKEOFF_ALTITUDE_M = 10.0
DEFAULT_TRACK_DURATION_S = 30.0
_SYSTEM_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "mission_planner_system.txt"
_DYNAMIC_SYSTEM_PROMPT_PATH = (
    _PROJECT_ROOT / "prompts" / "dynamic_skill_planner_system.txt"
)


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than zero"
        )
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one text instruction into either the legacy MissionIntent "
            "or a constrained dynamic SkillPlanDraft, then compile it with "
            "trusted world geometry. Isaac Sim is not used."
        )
    )
    parser.add_argument(
        "--planner",
        choices=("scripted", "llm", "dynamic_scripted", "dynamic_llm"),
        default="scripted",
        help="planner backend (default: %(default)s)",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="natural-language mission instruction",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base; llm mode defaults to QWEN_API_BASE",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="served model name; llm mode defaults to QWEN_MODEL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key; llm mode defaults to QWEN_API_KEY (never printed)",
    )
    parser.add_argument(
        "--takeoff-altitude",
        type=_positive_finite_float,
        default=DEFAULT_TAKEOFF_ALTITUDE_M,
        metavar="METERS",
        help="trusted default takeoff altitude (default: %(default)s)",
    )
    parser.add_argument(
        "--track-duration",
        type=_positive_finite_float,
        default=DEFAULT_TRACK_DURATION_S,
        metavar="SECONDS",
        help="trusted default tracking duration (default: %(default)s)",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="write exactly one JSON object to stdout",
    )
    return parser


def _build_world_context(
    *,
    takeoff_altitude_m: float,
    track_duration_s: float,
) -> PlannerWorldContext:
    """Return the small trusted world used only by this planning demo."""

    search_region = SearchRegionSpec(
        name="search_area",
        center_xyz_m=(20.0, 30.0, 0.0),
        radius_m=15.0,
        approach_xyz_m=(20.0, 12.0, takeoff_altitude_m),
        description="the designated outdoor area in which to search",
    )
    landing_zone = LandingZoneSpec(
        name="home",
        position_xy_m=(0.0, 0.0),
        ground_altitude_m=0.0,
        horizontal_tolerance_m=0.75,
        description="the UAV launch and recovery zone",
    )
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={search_region.name: search_region},
        landing_zones={landing_zone.name: landing_zone},
        default_takeoff_altitude_m=takeoff_altitude_m,
        default_track_duration_s=track_duration_s,
        search_timeout_s=75.0,
        goto_timeout_s=120.0,
        land_timeout_s=60.0,
    )


def _standard_dynamic_draft(args: argparse.Namespace) -> SkillPlanDraft:
    """Return the model-free dynamic search/track baseline."""

    return SkillPlanDraft.from_dict(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "takeoff_1",
                    "skill": "TAKEOFF",
                    "args": {"altitude_m": args.takeoff_altitude},
                },
                {
                    "id": "goto_search",
                    "skill": "GOTO",
                    "args": {
                        "destination": "search_area",
                        "altitude_m": args.takeoff_altitude,
                        "yaw_mode": "COURSE_ALIGNED",
                    },
                },
                {
                    "id": "search_1",
                    "skill": "SEARCH",
                    "args": {
                        "region": "search_area",
                        "target_description": "moving target",
                        "altitude_m": args.takeoff_altitude,
                    },
                },
                {
                    "id": "track_1",
                    "skill": "TRACK",
                    "args": {
                        "target_ref": "$search_1.target_id",
                        "duration_s": args.track_duration,
                        "desired_altitude_m": args.takeoff_altitude,
                        "desired_distance_m": 6.0,
                    },
                },
                {
                    "id": "goto_home",
                    "skill": "GOTO",
                    "args": {
                        "destination": "home",
                        "altitude_m": args.takeoff_altitude,
                        "yaw_mode": "COURSE_ALIGNED",
                    },
                },
                {
                    "id": "land_1",
                    "skill": "LAND",
                    "args": {"zone": "home"},
                },
            ],
        }
    )


class _PlannerDemoFailure(RuntimeError):
    def __init__(
        self,
        cause: Exception,
        diagnostics: PlannerDiagnostics | None,
    ) -> None:
        super().__init__(str(cause))
        self.cause_type = type(cause).__name__
        self.diagnostics = diagnostics


def _scripted_diagnostics() -> PlannerDiagnostics:
    return PlannerDiagnostics(
        model_calls=0,
        repair_used=False,
        repair_succeeded=False,
        initial_output_valid=True,
        final_output_valid=True,
        initial_error_code=None,
        initial_error_message=None,
        structured_output_enabled=False,
    )


def _plan(
    args: argparse.Namespace,
) -> tuple[PlannerOutput, CompiledMission, PlannerDiagnostics]:
    context = _build_world_context(
        takeoff_altitude_m=args.takeoff_altitude,
        track_duration_s=args.track_duration,
    )
    request = PlannerRequest(
        instruction=args.instruction,
        world_context=context,
    )

    if args.planner == "scripted":
        supplied_intent = MissionIntent(
            target_description="moving target",
            search_region="search_area",
            track_duration_s=args.track_duration,
            landing_zone="home",
            takeoff_altitude_m=args.takeoff_altitude,
        )
        planner = ScriptedPlanner(supplied_intent)
    elif args.planner == "dynamic_scripted":
        from planner.scripted_dynamic_planner import ScriptedDynamicPlanner

        planner = ScriptedDynamicPlanner(_standard_dynamic_draft(args))
    else:
        # Keep the model-specific stack out of the deterministic mode.  This
        # also makes ``scripted`` useful as an offline smoke test.
        from models.openai_compatible_client import OpenAICompatibleClient
        from planner.dynamic_llm_planner import DynamicLLMPlanner
        from planner.llm_planner import LLMPlanner

        client = OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )
        if args.planner == "llm":
            planner = LLMPlanner(client, system_prompt_path=_SYSTEM_PROMPT_PATH)
        else:
            planner = DynamicLLMPlanner(
                client,
                system_prompt_path=_DYNAMIC_SYSTEM_PROMPT_PATH,
            )

    diagnostics: PlannerDiagnostics | None = None
    try:
        plan_with_diagnostics = getattr(planner, "plan_with_diagnostics", None)
        if callable(plan_with_diagnostics):
            execution = plan_with_diagnostics(request)
            planner_output = execution.output
            diagnostics = execution.diagnostics
        else:
            planner_output = planner.plan(request)
            diagnostics = _scripted_diagnostics()
        compiled = PlanValidator().validate_and_compile(
            planner_output,
            context,
            source=args.planner,
        )
    except Exception as exc:
        observed = getattr(planner, "last_diagnostics", None)
        if isinstance(observed, PlannerDiagnostics):
            diagnostics = observed
        raise _PlannerDemoFailure(exc, diagnostics) from None
    return planner_output, compiled, diagnostics


def _render(
    planner_output: PlannerOutput,
    compiled: CompiledMission,
    diagnostics: PlannerDiagnostics,
    *,
    json_output: bool,
) -> None:
    dynamic = isinstance(planner_output, SkillPlanDraft)
    output_key = "skill_plan_draft" if dynamic else "mission_intent"
    payload = {
        output_key: planner_output.to_dict(),
        "compiled_task_plan": compiled.task_plan.to_dicts(),
        "planner_diagnostics": diagnostics.to_dict(),
    }
    if dynamic:
        payload["planner_output_type"] = type(planner_output).__name__
        payload["compiler_notes"] = list(compiled.compiler_notes)
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return

    print("SkillPlanDraft (planner-selected)" if dynamic else "MissionIntent")
    print(json.dumps(payload[output_key], ensure_ascii=False, indent=2))
    print("\nCompiled TaskPlan (trusted coordinates, policies, and timeouts)")
    print(json.dumps(payload["compiled_task_plan"], ensure_ascii=False, indent=2))
    if dynamic:
        print("\nCompiler Notes")
        print(json.dumps(payload["compiler_notes"], ensure_ascii=False, indent=2))
    print("\nPlanner Diagnostics")
    print(json.dumps(payload["planner_diagnostics"], ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        planner_output, compiled, diagnostics = _plan(args)
        _render(
            planner_output,
            compiled,
            diagnostics,
            json_output=args.json_output,
        )
        return 0
    except Exception as exc:
        # Model client errors intentionally do not contain credentials.  Keep
        # normal CLI failures concise and leave stdout clean for JSON callers.
        if isinstance(exc, _PlannerDemoFailure):
            error_type = exc.cause_type
            diagnostics = exc.diagnostics
        else:
            error_type = type(exc).__name__
            diagnostics = None
        failure = {
            "status": "FAILED",
            "error": {"type": error_type, "message": str(exc)},
            "planner_diagnostics": (
                None if diagnostics is None else diagnostics.to_dict()
            ),
        }
        if args.json_output:
            print(json.dumps(failure, ensure_ascii=False, separators=(",", ":")))
        else:
            print(
                f"[Planner demo] FAILED ({error_type}): {exc}",
                file=sys.stderr,
            )
            if diagnostics is not None:
                print(
                    "Planner Diagnostics: "
                    + json.dumps(diagnostics.to_dict(), ensure_ascii=False),
                    file=sys.stderr,
                )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
