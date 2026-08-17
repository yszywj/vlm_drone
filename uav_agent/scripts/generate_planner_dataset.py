#!/usr/bin/env python3
"""Generate the deterministic, Gold-first Planner dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from planner_data.generator import (  # noqa: E402
    DEFAULT_DATASET_CONFIG_PATH,
    DatasetGenerationError,
    PlannerDatasetGenerator,
    generate_and_write_dataset,
    stage_external_candidates,
)


def _strict_json_loads(text: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant {value!r} is forbidden")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Planner v1 text-only MissionIntent data without Isaac or Qwen."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_DATASET_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--paraphraser", choices=("none", "external"), default="none")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument(
        "--candidate-input",
        type=Path,
        help=(
            "explicit external JSONL containing only sample_id and "
            "candidate_instruction; never used as an assistant label"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.seed < 0:
        print("generation error: --seed must be non-negative", file=sys.stderr)
        return 2
    if arguments.paraphraser == "external" and not arguments.candidate_only:
        print(
            "generation error: external rewrites are candidate-only and cannot "
            "be written into official splits",
            file=sys.stderr,
        )
        return 2
    if arguments.candidate_only and (
        arguments.paraphraser != "external" or arguments.candidate_input is None
    ):
        print(
            "generation error: --candidate-only requires --paraphraser external "
            "and --candidate-input; this CLI never calls an external API itself",
            file=sys.stderr,
        )
        return 2
    try:
        if arguments.candidate_only:
            rows = []
            with arguments.candidate_input.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        raise DatasetGenerationError(
                            f"candidate input line {line_number} is blank"
                        )
                    row = _strict_json_loads(line)
                    if not isinstance(row, dict):
                        raise DatasetGenerationError(
                            f"candidate input line {line_number} is not an object"
                        )
                    rows.append(row)
            generator = PlannerDatasetGenerator(arguments.config)
            generated = generator.generate(seed=arguments.seed, profile=arguments.profile)
            candidate_path = stage_external_candidates(
                generated=generated,
                candidates=rows,
                output_root=arguments.output_root,
                ontology=generator.ontology,
                lexicon=generator.lexicon,
            )
            print(
                json.dumps(
                    {"candidate_path": str(candidate_path), "num_candidates": len(rows)},
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
            )
            return 0
        manifest = generate_and_write_dataset(
            config_path=arguments.config,
            output_root=arguments.output_root,
            seed=arguments.seed,
            profile=arguments.profile,
            overwrite=arguments.overwrite,
        )
    except (DatasetGenerationError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"generation error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
