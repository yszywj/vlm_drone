#!/usr/bin/env python3
"""Evaluate one Planner dataset split without Isaac Sim or Skill execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _PACKAGE_ROOT.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from models.openai_compatible_client import OpenAICompatibleClient  # noqa: E402
from common.ids import validate_uav_id  # noqa: E402
from planner_data.evaluator import (  # noqa: E402
    DEFAULT_DYNAMIC_SYSTEM_PROMPT_PATH,
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_WORLD_CONTEXTS_PATH,
    PlannerDatasetEvaluator,
    PlannerEvaluationError,
    load_planner_dataset_split,
    load_planner_world_cases,
)
from planner_data.schemas import PLANNER_DATASET_SPLITS  # noqa: E402
from tasks.target_ontology import (  # noqa: E402
    DEFAULT_ONTOLOGY_PATH,
    TargetOntology,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gold-grounded, field-level evaluation of the text-only mission "
            "Planner. This command never starts Isaac Sim."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_REPOSITORY_ROOT / "datasets" / "planner_v1",
        help="directory containing planner_v1 JSONL splits",
    )
    parser.add_argument(
        "--split",
        choices=PLANNER_DATASET_SPLITS,
        default="test_iid",
    )
    parser.add_argument(
        "--planner",
        choices=("scripted", "llm", "dynamic_scripted", "dynamic_llm"),
        default="scripted",
        help=(
            "scripted/dynamic_scripted validate the evaluator; "
            "llm/dynamic_llm call the configured text-model service"
        ),
    )
    parser.add_argument(
        "--uav-id",
        type=validate_uav_id,
        default="uav_1",
        help="trusted routing ID for schema-v2 dynamic_llm outputs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPOSITORY_ROOT / "outputs" / "planner_eval",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="resume this run directory and skip completed sample IDs",
    )
    parser.add_argument("--start-index", type=_nonnegative_int, default=0)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument(
        "--world-contexts",
        type=Path,
        default=DEFAULT_WORLD_CONTEXTS_PATH,
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=DEFAULT_ONTOLOGY_PATH,
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
        help=(
            "override the mode-specific system prompt (legacy defaults to "
            f"{DEFAULT_SYSTEM_PROMPT_PATH}; dynamic defaults to "
            f"{DEFAULT_DYNAMIC_SYSTEM_PROMPT_PATH})"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL; defaults to QWEN_API_BASE",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="served model; defaults to QWEN_MODEL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key; defaults to QWEN_API_KEY and is never persisted",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-request timeout in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=_nonnegative_int,
        default=2,
        help="transport retries inside each model call",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="print one compact JSON result object",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        samples = load_planner_dataset_split(args.dataset_root, args.split)
        worlds = load_planner_world_cases(args.world_contexts)
        ontology = TargetOntology.from_file(args.ontology)
        model_client = None
        if args.planner in {"llm", "dynamic_llm"}:
            model_client = OpenAICompatibleClient(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                timeout_s=args.timeout,
                max_retries=args.max_retries,
            )
        evaluator = PlannerDatasetEvaluator(
            planner=args.planner,
            world_cases=worlds,
            ontology=ontology,
            model_client=model_client,
            system_prompt_path=args.system_prompt,
            uav_id=args.uav_id,
        )
        run = evaluator.evaluate(
            samples,
            output_root=args.output_root,
            run_dir=args.resume,
            start_index=args.start_index,
            limit=args.limit,
            resume=args.resume is not None,
        )
    except (PlannerEvaluationError, OSError, TypeError, ValueError) as exc:
        # Errors never include or echo the API key supplied above.
        parser.error(str(exc))

    result = {"run_dir": str(run.run_dir), "summary": dict(run.summary)}
    if args.json_output:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    else:
        print(f"Planner evaluation output: {run.run_dir}")
        print(json.dumps(run.summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
