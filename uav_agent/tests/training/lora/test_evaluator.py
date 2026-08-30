from __future__ import annotations

import json
from pathlib import Path

import pytest

from fleet_data.validator import load_split
from training.lora.evaluate_fleet_planner_lora import (
    LoraEvaluationError,
    compare_base_and_lora,
    evaluate_prediction_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLD = PROJECT_ROOT / "datasets/fleet_planner_v1/test_reassignment.jsonl"


def _payload(sample: dict[str, object]) -> object:
    return sample[
        "output"
        if sample["output_kind"] == "fleet_mission_plan"
        else "fleet_plan_patch"
    ]


def _write_diagnostics(path: Path, *, malformed_second: bool = False) -> None:
    samples = load_split(GOLD)
    rows = []
    for index, raw_sample in enumerate(samples):
        sample = dict(raw_sample)
        parsed = _payload(sample)
        failed = malformed_second and index == 1
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "output_kind": sample["output_kind"],
                "raw_model_output": "not-json" if failed else json.dumps(parsed),
                "parsed_output": None if failed else parsed,
                "parse_success": not failed,
                "parse_error": "invalid JSON" if failed else None,
                "schema_valid": not failed,
                "schema_error": "parse failed" if failed else None,
            }
        )
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_diagnostics_distinguish_parse_schema_semantic_and_exact(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write_diagnostics(predictions, malformed_second=True)
    metrics = evaluate_prediction_diagnostics(GOLD, predictions)
    assert metrics["sample_count"] == 2
    assert metrics["json_parse_rate"] == 0.5
    assert metrics["schema_validity_rate"] == 0.5
    assert metrics["semantic_evaluator_success_rate"] == 0.5
    assert metrics["exact_output_accuracy"] == 0.5
    assert metrics["uav_assignment_accuracy"] < 1.0
    assert metrics["reassignment_replan_count"] == 2


def test_base_lora_comparison_reports_numeric_delta(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    lora = tmp_path / "lora.jsonl"
    _write_diagnostics(base, malformed_second=True)
    _write_diagnostics(lora)
    result = compare_base_and_lora(GOLD, base, lora)
    assert result["base"]["json_parse_rate"] == 0.5
    assert result["lora"]["json_parse_rate"] == 1.0
    assert result["delta_lora_minus_base"]["json_parse_rate"] == 0.5


def test_claimed_schema_validity_is_recomputed_fail_closed(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write_diagnostics(predictions, malformed_second=True)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[1]["schema_valid"] = True
    predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(LoraEvaluationError, match="schema cannot be valid"):
        evaluate_prediction_diagnostics(GOLD, predictions)


def test_raw_output_and_parsed_diagnostics_must_agree(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write_diagnostics(predictions)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]

    rows[0]["raw_model_output"] = "not-json"
    predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(LoraEvaluationError, match="parse_success disagrees"):
        evaluate_prediction_diagnostics(GOLD, predictions)

    _write_diagnostics(predictions)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[0]["parsed_output"] = {"schema_version": 1}
    predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(LoraEvaluationError, match="parsed_output disagrees"):
        evaluate_prediction_diagnostics(GOLD, predictions)
