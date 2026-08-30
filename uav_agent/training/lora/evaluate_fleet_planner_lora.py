#!/usr/bin/env python3
"""Evaluate Fleet Planner model diagnostics with production Fleet contracts.

Model loading deliberately lives in ``generate_fleet_planner_predictions``.
This module consumes its JSONL diagnostics, re-validates parsed payloads with
the production schema, and delegates field-level scoring to
``fleet_data.evaluator``. A legacy strict prediction JSONL remains supported.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys
import tempfile

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fleet_data.evaluator import evaluate_predictions  # noqa: E402
from fleet_data.validator import (  # noqa: E402
    PATCH_OUTPUT_KIND,
    PLAN_OUTPUT_KIND,
    FleetDatasetValidationError,
    load_split,
    parse_input_request,
    validate_fleet_output,
    validate_fleet_plan_patch,
    validate_sample,
)


class LoraEvaluationError(ValueError):
    """Raised when prediction diagnostics are ambiguous or inconsistent."""


_DIAGNOSTIC_REQUIRED_KEYS = frozenset(
    {
        "sample_id",
        "output_kind",
        "raw_model_output",
        "parsed_output",
        "parse_success",
        "schema_valid",
    }
)
_CONFLICT_SCENARIOS = frozenset(
    {"duplicate_target_request", "overlapping_regions_auto_assignment"}
)
_REPLAN_SCENARIOS = frozenset(
    {"unavailable_uav", "failed_assignment_reassignment"}
)


def _finite_float(raw: str, context: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise LoraEvaluationError(f"{context} contains a non-finite number {raw!r}")
    return value


def _strict_json(line: str, *, path: Path, line_number: int) -> Mapping[str, object]:
    def reject_constant(value: str) -> object:
        raise LoraEvaluationError(f"non-standard JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise LoraEvaluationError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            line,
            parse_constant=reject_constant,
            parse_float=lambda raw: _finite_float(raw, "diagnostic JSON"),
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, LoraEvaluationError) as exc:
        raise LoraEvaluationError(f"{path}:{line_number}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise LoraEvaluationError(f"{path}:{line_number}: row must be an object")
    return value


def _strict_model_output(text: str) -> Mapping[str, object]:
    """Reparse raw model text so diagnostics cannot claim a repaired payload."""

    def reject_constant(value: str) -> object:
        raise LoraEvaluationError(f"non-standard JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise LoraEvaluationError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            parse_float=lambda raw: _finite_float(raw, "raw model output"),
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise LoraEvaluationError(str(exc)) from exc
    if not isinstance(value, Mapping):
        raise LoraEvaluationError("raw model output JSON root must be an object")
    return value


def _load_diagnostics(path: str | Path) -> dict[str, Mapping[str, object]]:
    source = Path(path).expanduser().resolve()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LoraEvaluationError(f"could not read predictions {source}: {exc}") from exc
    result: dict[str, Mapping[str, object]] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise LoraEvaluationError(f"{source}:{line_number}: blank line")
        row = _strict_json(line, path=source, line_number=line_number)
        missing = sorted(_DIAGNOSTIC_REQUIRED_KEYS - set(row))
        if missing:
            raise LoraEvaluationError(
                f"{source}:{line_number}: diagnostic fields missing: {missing}"
            )
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise LoraEvaluationError(
                f"{source}:{line_number}: sample_id must be non-empty text"
            )
        if sample_id in result:
            raise LoraEvaluationError(f"duplicate prediction sample_id {sample_id!r}")
        if row["output_kind"] not in {PLAN_OUTPUT_KIND, PATCH_OUTPUT_KIND}:
            raise LoraEvaluationError(
                f"{source}:{line_number}: output_kind is invalid"
            )
        if not isinstance(row["raw_model_output"], str):
            raise LoraEvaluationError(
                f"{source}:{line_number}: raw_model_output must be text"
            )
        if not isinstance(row["parse_success"], bool) or not isinstance(
            row["schema_valid"], bool
        ):
            raise LoraEvaluationError(
                f"{source}:{line_number}: parse_success/schema_valid must be booleans"
            )
        if not row["parse_success"] and row["parsed_output"] is not None:
            raise LoraEvaluationError(
                f"{source}:{line_number}: failed parse must use parsed_output=null"
            )
        if row["schema_valid"] and not row["parse_success"]:
            raise LoraEvaluationError(
                f"{source}:{line_number}: schema cannot be valid after parse failure"
            )
        raw_payload: Mapping[str, object] | None = None
        try:
            raw_payload = _strict_model_output(row["raw_model_output"])
        except (LoraEvaluationError, TypeError, ValueError):
            pass
        if row["parse_success"] is not (raw_payload is not None):
            raise LoraEvaluationError(
                f"{source}:{line_number}: parse_success disagrees with strict raw output parsing"
            )
        if raw_payload is not None:
            parsed = row["parsed_output"]
            if not isinstance(parsed, Mapping) or dict(parsed) != dict(raw_payload):
                raise LoraEvaluationError(
                    f"{source}:{line_number}: parsed_output disagrees with raw_model_output"
                )
        result[sample_id] = row
    return result


def _is_diagnostic_file(path: str | Path) -> bool:
    source = Path(path).expanduser().resolve()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LoraEvaluationError(f"could not read predictions {source}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if line.strip():
            row = _strict_json(line, path=source, line_number=line_number)
            return _DIAGNOSTIC_REQUIRED_KEYS <= set(row)
    return False


def _payload(sample: Mapping[str, object]) -> Mapping[str, object]:
    key = "output" if sample["output_kind"] == PLAN_OUTPUT_KIND else "fleet_plan_patch"
    value = sample.get(key)
    if not isinstance(value, Mapping):
        raise LoraEvaluationError(f"gold sample {sample['sample_id']} has invalid {key}")
    return value


def _scenario_rate(successes: int, count: int) -> float | None:
    return None if count == 0 else successes / count


def evaluate_prediction_diagnostics(
    gold_split: str | Path,
    predictions: str | Path,
) -> dict[str, float | int | None]:
    """Score generator diagnostics without repairing or substituting Gold."""

    samples = load_split(gold_split)
    rows = _load_diagnostics(predictions)
    gold_by_id = {str(sample["sample_id"]): sample for sample in samples}
    extras = sorted(set(rows) - set(gold_by_id))
    if extras:
        raise LoraEvaluationError(
            "predictions contain IDs outside the gold split: " + ", ".join(extras)
        )

    parse_successes = 0
    schema_successes = 0
    semantic_successes = 0
    exact_successes = 0
    unassigned_successes = 0
    conflict_count = conflict_exact = conflict_semantic = 0
    replan_count = replan_exact = replan_semantic = 0
    evaluator_rows: list[dict[str, object]] = []

    for sample_id, sample in gold_by_id.items():
        row = rows.get(sample_id)
        scenario = str(sample["metadata"]["scenario_type"])  # type: ignore[index]
        is_conflict = scenario in _CONFLICT_SCENARIOS
        is_replan = scenario in _REPLAN_SCENARIOS
        conflict_count += int(is_conflict)
        replan_count += int(is_replan)
        if row is None:
            continue
        expected_kind = str(sample["output_kind"])
        if row["output_kind"] != expected_kind:
            raise LoraEvaluationError(
                f"prediction {sample_id} output_kind disagrees with dataset contract"
            )
        if row["parse_success"]:
            parse_successes += 1
        parsed = row["parsed_output"]
        if not isinstance(parsed, Mapping):
            if row["schema_valid"]:
                raise LoraEvaluationError(
                    f"prediction {sample_id} claims non-object output is schema valid"
                )
            continue

        request = parse_input_request(sample["input"])
        recomputed_schema = True
        try:
            if expected_kind == PLAN_OUTPUT_KIND:
                validate_fleet_output(parsed, request=request)
            else:
                validate_fleet_plan_patch(parsed, request=request)
        except FleetDatasetValidationError:
            recomputed_schema = False
        if row["schema_valid"] is not recomputed_schema:
            raise LoraEvaluationError(
                f"prediction {sample_id} schema_valid disagrees with production schema"
            )
        schema_successes += int(recomputed_schema)

        prediction_sample = dict(sample)
        prediction_sample.pop("output", None)
        prediction_sample.pop("fleet_plan_patch", None)
        prediction_sample[
            "output" if expected_kind == PLAN_OUTPUT_KIND else "fleet_plan_patch"
        ] = parsed
        semantic_valid = True
        try:
            validate_sample(prediction_sample)
        except FleetDatasetValidationError:
            semantic_valid = False
        semantic_successes += int(semantic_valid)

        gold_payload = _payload(sample)
        exact = dict(parsed) == dict(gold_payload)
        exact_successes += int(exact)
        notes_key = (
            "unassigned_requirements"
            if expected_kind == PLAN_OUTPUT_KIND
            else "reason_codes"
        )
        unassigned_successes += int(parsed.get(notes_key) == gold_payload.get(notes_key))
        if is_conflict:
            conflict_exact += int(exact)
            conflict_semantic += int(semantic_valid)
        if is_replan:
            replan_exact += int(exact)
            replan_semantic += int(semantic_valid)

        evaluator_row: dict[str, object] = {
            "sample_id": sample_id,
            "output_kind": expected_kind,
        }
        evaluator_row[
            "output" if expected_kind == PLAN_OUTPUT_KIND else "fleet_plan_patch"
        ] = parsed
        evaluator_rows.append(evaluator_row)

    with tempfile.TemporaryDirectory(prefix="fleet_lora_eval_") as temporary:
        strict_path = Path(temporary) / "parsed_predictions.jsonl"
        strict_path.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row in evaluator_rows
            ),
            encoding="utf-8",
        )
        field_metrics = evaluate_predictions(gold_split, strict_path)

    count = len(samples)
    result: dict[str, float | int | None] = {
        "sample_count": count,
        "prediction_record_count": len(rows),
        "json_parse_rate": parse_successes / count,
        "schema_validity_rate": schema_successes / count,
        "semantic_evaluator_success_rate": semantic_successes / count,
        "exact_output_accuracy": exact_successes / count,
        "unassigned_requirement_accuracy": unassigned_successes / count,
        "conflict_scenario_count": conflict_count,
        "conflict_scenario_exact_accuracy": _scenario_rate(
            conflict_exact, conflict_count
        ),
        "conflict_scenario_semantic_success_rate": _scenario_rate(
            conflict_semantic, conflict_count
        ),
        "reassignment_replan_count": replan_count,
        "reassignment_replan_exact_accuracy": _scenario_rate(
            replan_exact, replan_count
        ),
        "reassignment_replan_semantic_success_rate": _scenario_rate(
            replan_semantic, replan_count
        ),
    }
    for name, value in field_metrics.items():
        if name != "sample_count":
            result[name] = value
    return result


def compare_base_and_lora(
    gold_split: str | Path,
    base_predictions: str | Path,
    lora_predictions: str | Path,
) -> dict[str, object]:
    base = evaluate_prediction_diagnostics(gold_split, base_predictions)
    lora = evaluate_prediction_diagnostics(gold_split, lora_predictions)
    delta: dict[str, float] = {}
    for key in sorted(set(base) & set(lora)):
        left = base[key]
        right = lora[key]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and not key.endswith("_count")
        ):
            delta[key] = float(right) - float(left)
    return {"base": base, "lora": lora, "delta_lora_minus_base": delta}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--predictions", type=Path)
    selection.add_argument("--base-predictions", type=Path)
    parser.add_argument("--lora-predictions", type=Path)
    parser.add_argument("--legacy-strict-predictions", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.base_predictions is not None:
            if args.lora_predictions is None:
                raise LoraEvaluationError(
                    "--base-predictions requires --lora-predictions"
                )
            payload: object = compare_base_and_lora(
                args.gold,
                args.base_predictions,
                args.lora_predictions,
            )
        elif args.lora_predictions is not None:
            raise LoraEvaluationError(
                "--lora-predictions is valid only with --base-predictions"
            )
        elif args.legacy_strict_predictions or not _is_diagnostic_file(
            args.predictions
        ):
            payload = evaluate_predictions(args.gold, args.predictions)
        else:
            payload = evaluate_prediction_diagnostics(args.gold, args.predictions)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except (
        FleetDatasetValidationError,
        LoraEvaluationError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"LoRA evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
