from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from fleet.schemas import (
    parse_fleet_mission_plan,
    parse_fleet_mission_request,
    validate_fleet_mission_plan,
)
from fleet_data.evaluator import evaluate_predictions
from fleet_data.generator import FleetDatasetGenerator
from fleet_data.validator import (
    FLEET_DATASET_SPLITS,
    PATCH_OUTPUT_KIND,
    PLAN_OUTPUT_KIND,
    FleetDatasetValidationError,
    load_split,
    parse_fleet_plan_patch,
    validate_dataset,
    validate_sample,
)
from scripts.generate_fleet_planner_dataset import main as generate_main
from scripts.validate_fleet_planner_dataset import main as validate_main


def test_generated_dataset_has_all_splits_scenarios_and_v3_regions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output, seed=42)
    report = validate_dataset(output)
    assert report.valid, report.errors
    assert set(report.split_counts) == set(FLEET_DATASET_SPLITS)
    assert all(count == 2 for count in report.split_counts.values())
    assert len(report.scenario_types) == 14
    assert {
        "surplus_uav",
        "more_tasks_than_uavs",
        "unavailable_uav",
        "capability_mismatch",
        "duplicate_target_request",
        "failed_assignment_reassignment",
    }.issubset(report.scenario_types)
    kinds = {
        target["search_region"]["shape"]
        for split in FLEET_DATASET_SPLITS
        for sample in load_split(output / f"{split}.jsonl")
        for target in sample["input"]["target_requests"]
    }
    assert kinds == {"CIRCLE", "RECTANGLE", "SECTOR", "POLYGON", "CORRIDOR"}


def test_every_input_and_output_reuses_production_fleet_contracts() -> None:
    generated = FleetDatasetGenerator().generate(seed=42)
    for split, samples in generated.items():
        for sample in samples:
            request = parse_fleet_mission_request(sample["input"])
            assert request.to_dict() == sample["input"]
            if sample["output_kind"] == PLAN_OUTPUT_KIND:
                plan = parse_fleet_mission_plan(
                    sample["output"],
                    request=request,
                )
                assert validate_fleet_mission_plan(plan, request) is plan
                assert plan.to_dict() == sample["output"]
                assert (
                    sample["metadata"]["scenario_type"]
                    != "failed_assignment_reassignment"
                )
            else:
                patch = parse_fleet_plan_patch(
                    sample["fleet_plan_patch"],
                    request=request,
                )
                assert patch.to_dict() == sample["fleet_plan_patch"]
                assert split == "test_reassignment"
                assert (
                    sample["metadata"]["scenario_type"]
                    == "failed_assignment_reassignment"
                )


def test_contract_has_no_training_only_region_or_coordination_schema() -> None:
    generated = FleetDatasetGenerator().generate(seed=42)
    serialized = repr(generated)
    assert "target_claim_mode" not in serialized
    assert "airspace_conflict_policy" not in serialized
    assert "'type': 'circle'" not in serialized
    assert "'geometry':" not in serialized
    for sample in generated["train"]:
        policy = sample["input"]["coordination_policy"]
        assert policy["target_claim_policy"] == "EXCLUSIVE"
        assert policy["route_conflict_policy"] == "LOWER_PRIORITY_HOLDS"


def test_validator_rejects_unknown_uav_via_production_plan_validator() -> None:
    generated = FleetDatasetGenerator().generate(seed=1)
    sample = copy.deepcopy(generated["train"][0])
    sample["output"]["assignments"][0]["uav_id"] = "uav_unknown"
    with pytest.raises(FleetDatasetValidationError, match="unknown uav_id"):
        validate_sample(sample)


def test_validator_rejects_changed_v3_region_via_request_binding() -> None:
    sample = copy.deepcopy(FleetDatasetGenerator().generate(seed=1)["train"][0])
    region = sample["output"]["assignments"][0]["search_region"]
    if region["shape"] == "CIRCLE":
        region["center_xyz_m"][0] += 1.0
    else:
        region["center_xyz_m"][0] += 1.0
    with pytest.raises(FleetDatasetValidationError, match="changed requested RegionSpec"):
        validate_sample(sample)


def test_capability_requirement_uses_bound_target_spec_and_dataset_eligibility() -> None:
    sample = copy.deepcopy(
        next(
            item
            for item in FleetDatasetGenerator().generate(seed=42)[
                "test_compositional"
            ]
            if item["metadata"]["scenario_type"] == "capability_mismatch"
        )
    )
    request = parse_fleet_mission_request(sample["input"])
    assert request.target_request("target_i").target_spec.hard_attributes == (
        "required_payload:high_resolution_camera",
    )

    # The formal Fleet validator binds the complete TargetSpec, so a model may
    # not erase the requirement to escape dataset-level capability evaluation.
    changed_spec = copy.deepcopy(sample)
    changed_spec["output"]["assignments"][0]["target_spec"][
        "hard_attributes"
    ] = []
    with pytest.raises(FleetDatasetValidationError, match="changed for target_i"):
        validate_sample(changed_spec)

    # Generic payload eligibility is intentionally a Fleet-dataset-v1 semantic
    # on top of the formal hard_attributes field; the production contract does
    # not yet define a general capability-expression solver.
    ineligible = copy.deepcopy(sample)
    ineligible["output"]["assignments"][0]["uav_id"] = "uav_a"
    with pytest.raises(FleetDatasetValidationError, match="lacks required payload"):
        validate_sample(ineligible)


def test_reassignment_cannot_masquerade_as_normal_plan() -> None:
    sample = copy.deepcopy(
        next(
            item
            for item in FleetDatasetGenerator().generate(seed=42)[
                "test_reassignment"
            ]
            if item["metadata"]["scenario_type"]
            == "failed_assignment_reassignment"
        )
    )
    assert sample["output_kind"] == PATCH_OUTPUT_KIND
    assert "fleet_plan_patch" in sample and "output" not in sample
    sample["output_kind"] = PLAN_OUTPUT_KIND
    with pytest.raises(FleetDatasetValidationError, match="keys invalid"):
        validate_sample(sample)


def test_unavailable_uav_is_an_ordinary_preplanning_plan() -> None:
    sample = next(
        item
        for item in FleetDatasetGenerator().generate(seed=42)["test_reassignment"]
        if item["metadata"]["scenario_type"] == "unavailable_uav"
    )
    assert sample["output_kind"] == PLAN_OUTPUT_KIND
    assert "output" in sample and "fleet_plan_patch" not in sample
    inventory = {item["uav_id"]: item for item in sample["input"]["uav_inventory"]}
    assert inventory["uav_a"]["available"] is False
    assert inventory["uav_a"]["current_assignment_id"] is None
    assert all(
        target["requested_uav_id"] is None
        for target in sample["input"]["target_requests"]
    )
    assert sample["input"]["assumptions"] == [
        "uav_a is unavailable before planning"
    ]


def test_gold_self_evaluation_exposes_all_required_metrics(tmp_path: Path) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output)
    for split in ("test_iid", "test_reassignment"):
        metrics = evaluate_predictions(output / f"{split}.jsonl")
        assert metrics["schema_valid_rate"] == 1.0
        assert metrics["output_kind_accuracy"] == 1.0
        assert metrics["fleet_routing_exact_match"] == 1.0
        assert metrics["final_plan_valid_rate"] == 1.0
        assert metrics["duplicate_target_claim_rate"] == 0.0
        assert metrics["missing_task_rate"] == 0.0
        assert metrics["invalid_uav_rate"] == 0.0


@pytest.mark.parametrize("split", ("test_iid", "test_reassignment"))
def test_evaluator_aligns_semantics_when_model_chooses_new_assignment_ids(
    tmp_path: Path,
    split: str,
) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output)
    samples = load_split(output / f"{split}.jsonl")
    predictions = tmp_path / f"{split}_predictions.jsonl"
    rows = []
    for sample_index, sample in enumerate(samples, 1):
        kind = sample["output_kind"]
        payload_key = "output" if kind == PLAN_OUTPUT_KIND else "fleet_plan_patch"
        payload = copy.deepcopy(sample[payload_key])
        assignments_key = (
            "assignments"
            if kind == PLAN_OUTPUT_KIND
            else "replacement_assignments"
        )
        for assignment_index, assignment in enumerate(
            payload[assignments_key],
            1,
        ):
            assignment["assignment_id"] = (
                f"model_route_{sample_index}_{assignment_index}"
            )
        payload[assignments_key].reverse()
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "output_kind": kind,
                payload_key: payload,
            }
        )
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    metrics = evaluate_predictions(output / f"{split}.jsonl", predictions)

    for metric in (
        "schema_valid_rate",
        "output_kind_accuracy",
        "fleet_routing_exact_match",
        "assignment_count_accuracy",
        "uav_assignment_accuracy",
        "target_assignment_accuracy",
        "region_accuracy",
        "track_duration_accuracy",
        "coordination_policy_accuracy",
        "final_plan_valid_rate",
    ):
        assert metrics[metric] == 1.0
    assert metrics["duplicate_target_claim_rate"] == 0.0
    assert metrics["missing_task_rate"] == 0.0
    assert metrics["invalid_uav_rate"] == 0.0


def test_evaluator_does_not_cherry_pick_an_ambiguous_duplicate_target(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output)
    samples = load_split(output / "test_iid.jsonl")
    predictions = tmp_path / "ambiguous_predictions.jsonl"
    rows = []
    for sample_index, sample in enumerate(samples):
        payload = copy.deepcopy(sample["output"])
        if sample_index == 0:
            assignments = payload["assignments"]
            assignments[1]["target_alias"] = assignments[0]["target_alias"]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "output_kind": PLAN_OUTPUT_KIND,
                "output": payload,
            }
        )
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    metrics = evaluate_predictions(output / "test_iid.jsonl", predictions)

    assert metrics["duplicate_target_claim_rate"] == 0.5
    assert metrics["schema_valid_rate"] == 0.5
    assert metrics["final_plan_valid_rate"] == 0.5
    # The ambiguous sample contributes no field-level matches; the other sample
    # remains perfect.  In particular, the evaluator must not select the better
    # of two duplicate target claims after the fact.
    assert metrics["uav_assignment_accuracy"] == 0.5
    assert metrics["target_assignment_accuracy"] == 0.5
    assert metrics["region_accuracy"] == 0.5
    assert metrics["track_duration_accuracy"] == 0.5


def test_evaluator_rejects_prediction_ids_outside_gold_split(tmp_path: Path) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output)
    sample = load_split(output / "test_iid.jsonl")[0]
    predictions = tmp_path / "extra_predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "sample_id": "fleet_sample_outside_gold",
                "output_kind": PLAN_OUTPUT_KIND,
                "output": sample["output"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FleetDatasetValidationError,
        match="outside the gold split",
    ):
        evaluate_predictions(output / "test_iid.jsonl", predictions)


def test_evaluator_rates_are_bounded_and_noise_cannot_hide_missing_tasks(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output)
    samples = load_split(output / "test_iid.jsonl")

    def write_predictions(path: Path, *, noisy: bool) -> None:
        rows = []
        for sample_index, sample in enumerate(samples):
            payload = copy.deepcopy(sample["output"])
            if sample_index == 0:
                retained = payload["assignments"][0]
                payload["assignments"] = [retained]
                payload["unassigned_requirements"] = []
                if noisy:
                    duplicate = copy.deepcopy(retained)
                    duplicate["assignment_id"] = "noise_duplicate_assignment"
                    payload["assignments"].append(duplicate)
                    payload["unassigned_requirements"] = [
                        "noise without a target alias",
                        "noise without a target alias",
                        f"{retained['target_alias']}: duplicate coverage noise",
                    ]
                    for index in range(12):
                        invalid = copy.deepcopy(retained)
                        invalid["assignment_id"] = f"noise_invalid_{index}"
                        invalid["uav_id"] = f"uav_unknown_{index}"
                        invalid["target_alias"] = f"target_unknown_{index}"
                        payload["assignments"].append(invalid)
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "output_kind": PLAN_OUTPUT_KIND,
                    "output": payload,
                }
            )
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    baseline_path = tmp_path / "baseline.jsonl"
    noisy_path = tmp_path / "noisy.jsonl"
    write_predictions(baseline_path, noisy=False)
    write_predictions(noisy_path, noisy=True)
    baseline = evaluate_predictions(output / "test_iid.jsonl", baseline_path)
    noisy = evaluate_predictions(output / "test_iid.jsonl", noisy_path)

    assert noisy["missing_task_rate"] == baseline["missing_task_rate"]
    assert noisy["invalid_uav_rate"] > 0.0
    assert noisy["duplicate_target_claim_rate"] > 0.0
    for name, value in noisy.items():
        if name == "sample_count":
            continue
        assert 0.0 <= value <= 1.0, name


def test_real_instructions_cover_all_required_natural_uav_aliases() -> None:
    generated = FleetDatasetGenerator().generate(seed=42)
    samples = [sample for rows in generated.values() for sample in rows]
    instructions = tuple(sample["input"]["original_instruction"] for sample in samples)
    for alias in (
        "第一架无人机",
        "左边那架无人机",
        "速度较快的无人机",
        "带高分辨率相机的无人机",
    ):
        assert any(alias in instruction for instruction in instructions)

    natural_alias = next(
        sample
        for sample in samples
        if sample["metadata"]["scenario_type"] == "natural_alias"
    )
    assert all(
        target["requested_uav_id"] is None
        for target in natural_alias["input"]["target_requests"]
    )
    assert {
        assignment["target_alias"]: assignment["uav_id"]
        for assignment in natural_alias["output"]["assignments"]
    } == {"target_i": "uav_a", "target_j": "uav_b"}


def test_generator_is_deterministic_and_seed_only_changes_order() -> None:
    generator = FleetDatasetGenerator()
    first = generator.generate(seed=42)
    second = generator.generate(seed=42)
    different_order = generator.generate(seed=43)
    assert first == second
    assert {
        sample["sample_id"]
        for rows in first.values()
        for sample in rows
    } == {
        sample["sample_id"]
        for rows in different_order.values()
        for sample in rows
    }


def test_failed_overwrite_generation_preserves_previous_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fleet"
    generator = FleetDatasetGenerator()
    generator.write(output, seed=42)
    previous_manifest = (output / "manifest.json").read_bytes()

    def fail_generation(*, seed: int) -> object:
        raise RuntimeError(f"synthetic failure for seed {seed}")

    monkeypatch.setattr(generator, "generate", fail_generation)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        generator.write(output, seed=43, overwrite=True)
    assert (output / "manifest.json").read_bytes() == previous_manifest
    assert validate_dataset(output).valid


@pytest.mark.parametrize(
    ("field", "replacement", "error_fragment"),
    (
        ("schema_version", 1, "schema_version"),
        ("seed", -1, "seed"),
        ("dataset_contract", "training-only-contract", "dataset_contract"),
        ("generation_source", "unknown-generator", "generation_source"),
    ),
)
def test_validator_rejects_invalid_manifest_contract_fields(
    tmp_path: Path,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_dataset(output)
    assert not report.valid
    assert any(error_fragment in error for error in report.errors)


def test_validator_rejects_manifest_count_and_hash_mismatches(tmp_path: Path) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_counts"]["train"] = 3
    manifest["sha256"]["validation.jsonl"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_dataset(output)
    assert not report.valid
    assert any("split_counts.train" in error for error in report.errors)


def test_validator_rejects_manifest_seed_that_does_not_reproduce_order(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output, seed=42)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 43
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_dataset(output)
    assert not report.valid
    assert any("manifest.seed" in error for error in report.errors)


def test_validator_rejects_tampered_split_even_when_json_is_valid(tmp_path: Path) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output)
    split_path = output / "train.jsonl"
    rows = [json.loads(line) for line in split_path.read_text().splitlines()]
    rows[0]["input"]["original_instruction"] += "（被篡改）"
    split_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = validate_dataset(output)
    assert not report.valid
    assert any("SHA-256 mismatch" in error for error in report.errors)


def test_validator_requires_all_fourteen_scenarios_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fleet"
    FleetDatasetGenerator().write(output)
    split_path = output / "train.jsonl"
    rows = [json.loads(line) for line in split_path.read_text().splitlines()]
    replaced = rows[0]["metadata"]["scenario_type"]
    rows[0]["metadata"]["scenario_type"] = rows[1]["metadata"]["scenario_type"]
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    split_path.write_text(encoded, encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"]["train.jsonl"] = sha256(encoded.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_dataset(output)
    assert not report.valid
    assert any(replaced in error and "missing" in error for error in report.errors)


def test_overwrite_refuses_an_unrelated_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "unrelated"
    output.mkdir()
    (output / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="previously created"):
        FleetDatasetGenerator().write(output, overwrite=True)
    assert (output / "user.txt").read_text() == "keep"


def test_documented_dataset_cli_names_and_compatibility_aliases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical = tmp_path / "canonical"
    assert generate_main(["--output", str(canonical), "--seed", "42"]) == 0
    assert validate_main(["--dataset-root", str(canonical)]) == 0
    assert '"valid": true' in capsys.readouterr().out

    aliases = tmp_path / "aliases"
    assert generate_main(["--output-dir", str(aliases), "--seed", "42"]) == 0
    assert validate_main(["--dataset-dir", str(aliases)]) == 0
    assert '"valid": true' in capsys.readouterr().out
