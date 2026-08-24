"""Fleet Planner JSONL generation, validation, and offline evaluation."""

from fleet_data.generator import FleetDatasetGenerator, generate_fleet_dataset
from fleet_data.validator import (
    FLEET_DATASET_CONTRACT,
    FLEET_DATASET_SCENARIOS,
    FLEET_DATASET_SPLIT_SCENARIOS,
    FLEET_DATASET_SPLITS,
    PATCH_OUTPUT_KIND,
    PLAN_OUTPUT_KIND,
    FleetDatasetValidationError,
    FleetDatasetValidationReport,
    parse_fleet_plan_patch,
    parse_input_request,
    validate_dataset,
    validate_fleet_output,
    validate_fleet_plan_patch,
    validate_sample,
)
from fleet_data.evaluator import evaluate_predictions

__all__ = [
    "FLEET_DATASET_CONTRACT",
    "FLEET_DATASET_SCENARIOS",
    "FLEET_DATASET_SPLIT_SCENARIOS",
    "FLEET_DATASET_SPLITS",
    "PATCH_OUTPUT_KIND",
    "PLAN_OUTPUT_KIND",
    "FleetDatasetGenerator",
    "FleetDatasetValidationError",
    "FleetDatasetValidationReport",
    "evaluate_predictions",
    "generate_fleet_dataset",
    "parse_fleet_plan_patch",
    "parse_input_request",
    "validate_dataset",
    "validate_fleet_output",
    "validate_fleet_plan_patch",
    "validate_sample",
]
