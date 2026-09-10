from dataclasses import replace

import pytest
import torch

from tests.training.target_state.test_data_and_trainer import (
    _missing_depth_evaluation_batch, _FixedEvaluationModel,
)
from training.target_state.trainer import TargetStateEvaluationAccumulator


def test_positive_residual_cannot_resurrect_missing_zero_depth_reference():
    batch = _missing_depth_evaluation_batch()
    output = _FixedEvaluationModel(validity_logit=8.0)(batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
    output = replace(output, depth_residual_m=torch.tensor([0.46742684]))
    accumulator = TargetStateEvaluationAccumulator(200.0)
    accumulator.add_batch(batch=batch, output=output)
    metrics = accumulator.finalize()["model"]
    assert metrics["measurement_failure_rate"] == 1.0
    assert metrics["accepted_measurement_count"] == 0
    assert metrics["position_p95_error_m"] is None


def test_acceptance_does_not_consult_truth_labels():
    batch = _missing_depth_evaluation_batch()
    batch["missing_mask"][:, -1] = False
    batch["raw_depth_m"].fill_(4.0)
    batch["valid_depth_mask"].fill_(True)
    batch["label_valid_mask"].fill_(False)
    batch["target_present_mask"].fill_(False)
    batch["target_position_world_m"].zero_()
    # measurement_valid remains False: it is a training label, not a sensor gate.
    output = _FixedEvaluationModel(validity_logit=8.0)(batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
    accumulator = TargetStateEvaluationAccumulator(200.0)
    accumulator.add_batch(batch=batch, output=output)
    assert accumulator.finalize()["model"]["no_target_false_positive_rate"] == 1.0


def test_old_evaluation_state_cannot_mix_with_new_acceptance_contract():
    state = TargetStateEvaluationAccumulator(200.0).state_dict()
    state["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        TargetStateEvaluationAccumulator.from_state_dict(state, maximum_depth_m=200.0)
