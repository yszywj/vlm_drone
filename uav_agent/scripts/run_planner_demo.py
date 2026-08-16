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
    LandingZoneSpec,
    MissionIntent,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
)
from planner.scripted_planner import ScriptedPlanner  # noqa: E402
from runtime.plan_validator import PlanValidator  # noqa: E402


DEFAULT_INSTRUCTION = (
    "起飞后前往 search_area 搜寻移动目标，找到后跟踪三十秒，然后返回 home 降落"
)
DEFAULT_TAKEOFF_ALTITUDE_M = 10.0
DEFAULT_TRACK_DURATION_S = 30.0
_SYSTEM_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "mission_planner_system.txt"


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
            "Parse one text instruction into MissionIntent and compile the "
            "trusted six-step UAV TaskPlan. Isaac Sim is not used."
        )
    )
    parser.add_argument(
        "--planner",
        choices=("scripted", "llm"),
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


def _plan(args: argparse.Namespace) -> tuple[MissionIntent, list[dict[str, object]]]:
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
    else:
        # Keep the model-specific stack out of the deterministic mode.  This
        # also makes ``scripted`` useful as an offline smoke test.
        from models.openai_compatible_client import OpenAICompatibleClient
        from planner.llm_planner import LLMPlanner

        client = OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )
        planner = LLMPlanner(client, system_prompt_path=_SYSTEM_PROMPT_PATH)

    intent = planner.plan(request)
    compiled = PlanValidator().validate_and_compile(
        intent,
        context,
        source=args.planner,
    )
    return intent, compiled.task_plan.to_dicts()


def _render(
    intent: MissionIntent,
    task_plan: list[dict[str, object]],
    *,
    json_output: bool,
) -> None:
    payload = {
        "mission_intent": intent.to_dict(),
        "compiled_task_plan": task_plan,
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return

    print("MissionIntent")
    print(json.dumps(payload["mission_intent"], ensure_ascii=False, indent=2))
    print("\nCompiled TaskPlan")
    print(json.dumps(payload["compiled_task_plan"], ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        intent, task_plan = _plan(args)
        _render(intent, task_plan, json_output=args.json_output)
        return 0
    except Exception as exc:
        # Model client errors intentionally do not contain credentials.  Keep
        # normal CLI failures concise and leave stdout clean for JSON callers.
        print(
            f"[Planner demo] FAILED ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
