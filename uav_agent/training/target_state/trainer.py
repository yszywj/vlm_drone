"""Trainer and auditable promotion gate for temporal ray-depth models."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from datasets.target_state.dataset import compute_dataset_sha256, read_frame_records
from experiments.metric_logger import ScalarEventWriter
from training.target_state.config import TargetStateTrainingConfig, TrainingStage
from training.target_state.data import GEOMETRY_INPUT_FIELDS, TargetStateTorchDataset
from training.target_state.geometry import corrected_ray_to_world
from training.target_state.losses import compute_target_state_losses
from training.target_state.model import TemporalRayDepthNet, TemporalRayDepthOutput


MODEL_SCHEMA_VERSION = 1
MODEL_TYPE = "temporal_ray_depth_residual"
OUTPUT_FIELDS = (
    "delta_u_px", "delta_v_px", "depth_residual_m",
    "position_log_variance_xyz", "measurement_valid_logit",
)


class TargetStateTrainingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TargetStateTrainingResult:
    run_dir: Path
    best_checkpoint: Path
    latest_checkpoint: Path
    model_manifest: Path
    validation_metrics: Mapping[str, object]
    test_metrics: Mapping[str, object]
    promoted: bool
    elapsed_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "best_checkpoint": str(self.best_checkpoint),
            "latest_checkpoint": str(self.latest_checkpoint),
            "model_manifest": str(self.model_manifest),
            "validation_metrics": dict(self.validation_metrics),
            "test_metrics": dict(self.test_metrics),
            "promoted": self.promoted,
            "elapsed_s": self.elapsed_s,
        }


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_initial_checkpoint(
    config: TargetStateTrainingConfig,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[Mapping[str, object] | None, str | None]:
    """Validate the stage handoff without starting a training epoch.

    Stage B is intentionally not permitted to start from random weights or a
    previous stage-B artifact: its initialization must be auditable as the
    output of the clean stage-A run.
    """

    path = config.initial_checkpoint_path
    if path is None:
        if config.stage is TrainingStage.YOLO_DEPLOYMENT:
            raise TargetStateTrainingError(
                "yolo_deployment is stage B and requires --initial-checkpoint pointing "
                "to a stage-A oracle_clean best.pt"
            )
        return None, None
    if not path.is_file():
        raise TargetStateTrainingError(f"initial checkpoint does not exist: {path}")
    initial = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(initial, Mapping):
        raise TargetStateTrainingError("initial checkpoint payload must be a mapping")
    if (
        initial.get("model_type") != MODEL_TYPE
        or initial.get("schema_version") != MODEL_SCHEMA_VERSION
    ):
        raise TargetStateTrainingError("initial checkpoint model type/schema is incompatible")
    if (
        config.stage is TrainingStage.YOLO_DEPLOYMENT
        and initial.get("training_stage") != TrainingStage.ORACLE_CLEAN.value
    ):
        raise TargetStateTrainingError(
            "stage-B initial checkpoint must declare training_stage=oracle_clean"
        )
    return initial, sha256_file(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise TargetStateTrainingError(f"invalid training device {value!r}: {exc}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TargetStateTrainingError(f"CUDA device requested but unavailable: {value}")
    return device


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _quantile(values: Tensor, q: float) -> float | None:
    finite = values[torch.isfinite(values)]
    return float(torch.quantile(finite, q).cpu()) if finite.numel() else None


def _rank_correlation(left: Tensor, right: Tensor) -> float | None:
    mask = torch.isfinite(left) & torch.isfinite(right)
    left, right = left[mask], right[mask]
    if left.numel() < 3:
        return None
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = torch.linalg.vector_norm(left_rank) * torch.linalg.vector_norm(right_rank)
    if denominator <= 1e-12:
        return None
    return float((left_rank @ right_rank / denominator).cpu())


def _invalid_model_output_mask(
    output: TemporalRayDepthOutput,
    *,
    predicted_position_world_m: Tensor,
    corrected_depth_m: Tensor,
    model_claims_valid: Tensor,
    maximum_depth_m: float,
) -> Tensor:
    """Return samples whose deployable output violates the model contract.

    Network heads are always outputs, so NaN/Inf in any head is invalid even
    when the validity head rejects the measurement.  World position and
    corrected depth are conditional outputs, however: production constructs
    and publishes them only after ``measurement_valid_logit`` accepts the
    measurement.  A rejected sample with no usable raw depth is therefore a
    measurement failure, not an out-of-contract numeric output.
    """

    head_invalid = (
        ~torch.isfinite(output.delta_uv_px).all(dim=-1)
        | ~torch.isfinite(output.depth_residual_m)
        | ~torch.isfinite(output.position_log_variance).all(dim=-1)
        | ~torch.isfinite(output.measurement_valid_logit)
    )
    claimed_geometry_invalid = model_claims_valid & (
        ~torch.isfinite(predicted_position_world_m).all(dim=-1)
        | ~torch.isfinite(corrected_depth_m)
        | (corrected_depth_m <= 0.0)
        | (corrected_depth_m > maximum_depth_m)
    )
    return head_invalid | claimed_geometry_invalid


def _metric_block(
    errors: Tensor,
    *,
    failures: Tensor,
    occlusion: Tensor,
    jitter: Tensor,
    uncertainty: Tensor | None,
    no_target_valid_claims: Tensor,
    invalid_output_count: int,
) -> dict[str, object]:
    occluded = errors[occlusion >= 0.25]
    jittered = errors[jitter >= 0.01]
    return {
        "position_median_error_m": _quantile(errors, 0.5),
        "position_p95_error_m": _quantile(errors, 0.95),
        "measurement_failure_rate": float(failures.float().mean().cpu()) if failures.numel() else None,
        "no_target_false_positive_rate": (
            float(no_target_valid_claims.float().mean().cpu())
            if no_target_valid_claims.numel()
            else None
        ),
        "occluded_position_median_error_m": _quantile(occluded, 0.5),
        "jittered_position_median_error_m": _quantile(jittered, 0.5),
        "covariance_error_spearman": (
            _rank_correlation(uncertainty, errors) if uncertainty is not None else None
        ),
        "invalid_output_count": int(invalid_output_count),
        "evaluated_measurement_count": int(errors.numel()),
        "evaluated_no_target_count": int(no_target_valid_claims.numel()),
    }


_EVALUATION_ACCUMULATOR_SCHEMA_VERSION = 1
_EVALUATION_TENSOR_FIELDS = (
    "model_errors",
    "baseline_errors",
    "model_failures",
    "baseline_failures",
    "model_no_target_claims",
    "baseline_no_target_claims",
    "uncertainties",
    "occlusions",
    "jitters",
)
_EVALUATION_BOOLEAN_TENSOR_FIELDS = frozenset(
    {
        "model_failures",
        "baseline_failures",
        "model_no_target_claims",
        "baseline_no_target_claims",
    }
)


@dataclass(slots=True)
class TargetStateEvaluationAccumulator:
    """Mergeable sufficient statistics for target-state evaluation.

    Every tensor retained here is a detached CPU snapshot.  Final shard
    metrics must not be averaged: callers accumulate batches per shard, merge
    accumulators in dataset order, and call :meth:`finalize` once for the full
    validation or test split.
    """

    maximum_depth_m: float
    model_errors: list[Tensor] = field(default_factory=list)
    baseline_errors: list[Tensor] = field(default_factory=list)
    model_failures: list[Tensor] = field(default_factory=list)
    baseline_failures: list[Tensor] = field(default_factory=list)
    model_no_target_claims: list[Tensor] = field(default_factory=list)
    baseline_no_target_claims: list[Tensor] = field(default_factory=list)
    uncertainties: list[Tensor] = field(default_factory=list)
    occlusions: list[Tensor] = field(default_factory=list)
    jitters: list[Tensor] = field(default_factory=list)
    invalid_model_output_count: int = 0
    invalid_baseline_output_count: int = 0
    loss_sum: float = 0.0
    batch_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.maximum_depth_m, bool):
            raise ValueError("maximum_depth_m must be finite and positive")
        self.maximum_depth_m = float(self.maximum_depth_m)
        if not math.isfinite(self.maximum_depth_m) or self.maximum_depth_m <= 0.0:
            raise ValueError("maximum_depth_m must be finite and positive")

    @staticmethod
    def _cpu_snapshot(value: Tensor) -> Tensor:
        return value.detach().cpu().clone()

    @torch.no_grad()
    def add_batch(
        self,
        *,
        batch: Mapping[str, Tensor],
        output: TemporalRayDepthOutput,
    ) -> None:
        """Accumulate one already-inferred batch without retaining GPU state."""

        network_heads = (
            output.delta_uv_px,
            output.depth_residual_m,
            output.position_log_variance,
            output.measurement_valid_logit,
        )
        if any(not bool(torch.isfinite(value).all().item()) for value in network_heads):
            raise TargetStateTrainingError(
                "evaluation batch contains a non-finite model output"
            )
        loss = compute_target_state_losses(output, batch)  # type: ignore[arg-type]
        baseline_world, baseline_depth, baseline_ray_valid = corrected_ray_to_world(
            anchor_uv_px=batch["anchor_uv_px"],
            raw_depth_m=batch["raw_depth_m"],
            delta_uv_px=torch.zeros_like(output.delta_uv_px),
            depth_residual_m=torch.zeros_like(output.depth_residual_m),
            intrinsics_fx_fy_cx_cy=batch["intrinsics_fx_fy_cx_cy"],
            camera_position_world_m=batch["camera_position_world_m"],
            camera_orientation_world_wxyz=batch["camera_orientation_world_wxyz"],
        )
        target = batch["target_position_world_m"]
        # Keep the original non-sharded semantics: every visible target is an
        # evaluation sample, including a real detector miss.
        target_present = batch["target_present_mask"].bool()
        label_valid = (
            batch["label_valid_mask"].bool()
            & target_present
            & batch["history_visible_mask"][:, -1].bool()
            & torch.isfinite(target).all(dim=-1)
        )
        model_claims_valid = torch.sigmoid(output.measurement_valid_logit) >= 0.5
        model_output_valid = (
            loss.ray_valid_mask
            & torch.isfinite(loss.predicted_position_world_m).all(dim=-1)
            & torch.isfinite(loss.corrected_depth_m)
            & (loss.corrected_depth_m <= self.maximum_depth_m)
            & model_claims_valid
        )
        baseline_output_valid = (
            baseline_ray_valid
            & torch.isfinite(baseline_world).all(dim=-1)
            & torch.isfinite(baseline_depth)
            & (baseline_depth <= self.maximum_depth_m)
        )
        model_error = torch.linalg.vector_norm(
            loss.predicted_position_world_m - target, dim=-1
        )
        baseline_error = torch.linalg.vector_norm(baseline_world - target, dim=-1)
        no_target = ~target_present

        # Compute all snapshots/counts before mutating self, so a malformed
        # batch cannot leave a partially-added accumulator entry.
        values = {
            "model_errors": self._cpu_snapshot(model_error[label_valid]),
            "baseline_errors": self._cpu_snapshot(baseline_error[label_valid]),
            "model_failures": self._cpu_snapshot(~model_output_valid[label_valid]),
            "baseline_failures": self._cpu_snapshot(~baseline_output_valid[label_valid]),
            "model_no_target_claims": self._cpu_snapshot(model_output_valid[no_target]),
            "baseline_no_target_claims": self._cpu_snapshot(baseline_output_valid[no_target]),
            "uncertainties": self._cpu_snapshot(
                torch.exp(output.position_log_variance).sum(dim=-1)[label_valid]
            ),
            "occlusions": self._cpu_snapshot(batch["occlusion_ratio"][label_valid]),
            "jitters": self._cpu_snapshot(batch["bbox_jitter_score"][label_valid]),
        }
        invalid_model = int(
            _invalid_model_output_mask(
                output,
                predicted_position_world_m=loss.predicted_position_world_m,
                corrected_depth_m=loss.corrected_depth_m,
                model_claims_valid=model_claims_valid,
                maximum_depth_m=self.maximum_depth_m,
            ).sum().cpu()
        )
        invalid_baseline = int(
            (
                ~torch.isfinite(baseline_world).all(dim=-1)
                | ~torch.isfinite(baseline_depth)
            ).sum().cpu()
        )
        loss_value = float(loss.total.detach().cpu())
        if not math.isfinite(loss_value):
            raise TargetStateTrainingError(
                "evaluation batch produced a non-finite loss"
            )
        floating_metric_fields = (
            "model_errors",
            "baseline_errors",
            "uncertainties",
            "occlusions",
            "jitters",
        )
        for name in floating_metric_fields:
            if not bool(torch.isfinite(values[name]).all().item()):
                raise TargetStateTrainingError(
                    f"evaluation batch produced non-finite metric values: {name}"
                )
        next_loss_sum = self.loss_sum + loss_value
        if not math.isfinite(self.loss_sum) or not math.isfinite(next_loss_sum):
            raise TargetStateTrainingError(
                "evaluation accumulated loss is non-finite"
            )

        for name, value in values.items():
            getattr(self, name).append(value)
        self.invalid_model_output_count += invalid_model
        self.invalid_baseline_output_count += invalid_baseline
        self.loss_sum = next_loss_sum
        self.batch_count += 1

    def merge(
        self, other: "TargetStateEvaluationAccumulator"
    ) -> "TargetStateEvaluationAccumulator":
        """Merge *other* into this accumulator in caller-defined shard order."""

        if not isinstance(other, TargetStateEvaluationAccumulator):
            raise TypeError("other must be a TargetStateEvaluationAccumulator")
        if other is self:
            raise ValueError("an evaluation accumulator cannot merge itself")
        if self.maximum_depth_m != other.maximum_depth_m:
            raise ValueError(
                "cannot merge evaluation accumulators with different maximum_depth_m"
            )
        if not math.isfinite(self.loss_sum) or not math.isfinite(other.loss_sum):
            raise TargetStateTrainingError(
                "cannot merge an accumulator with non-finite loss_sum"
            )
        merged_loss_sum = self.loss_sum + other.loss_sum
        if not math.isfinite(merged_loss_sum):
            raise TargetStateTrainingError(
                "merged evaluation loss_sum would be non-finite"
            )
        for name in _EVALUATION_TENSOR_FIELDS:
            getattr(self, name).extend(value.clone() for value in getattr(other, name))
        self.invalid_model_output_count += other.invalid_model_output_count
        self.invalid_baseline_output_count += other.invalid_baseline_output_count
        self.loss_sum = merged_loss_sum
        self.batch_count += other.batch_count
        return self

    def state_dict(self) -> dict[str, object]:
        """Return a validated, checkpoint-safe CPU-only accumulator state."""

        state: dict[str, object] = {
            "schema_version": _EVALUATION_ACCUMULATOR_SCHEMA_VERSION,
            "maximum_depth_m": self.maximum_depth_m,
            "invalid_model_output_count": self.invalid_model_output_count,
            "invalid_baseline_output_count": self.invalid_baseline_output_count,
            "loss_sum": self.loss_sum,
            "batch_count": self.batch_count,
        }
        for name in _EVALUATION_TENSOR_FIELDS:
            state[name] = [value.detach().cpu().clone() for value in getattr(self, name)]
        # Public fields are intentionally inspectable.  Validate them before
        # persisting so accidental caller mutation cannot create a corrupt
        # eval-boundary checkpoint.
        self.from_state_dict(state, maximum_depth_m=self.maximum_depth_m)
        return state

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        maximum_depth_m: float,
    ) -> "TargetStateEvaluationAccumulator":
        """Restore a strictly validated CPU-only accumulator checkpoint."""

        if not isinstance(state, Mapping):
            raise TypeError("evaluation accumulator state must be a mapping")
        expected_keys = {
            "schema_version",
            "maximum_depth_m",
            "invalid_model_output_count",
            "invalid_baseline_output_count",
            "loss_sum",
            "batch_count",
            *_EVALUATION_TENSOR_FIELDS,
        }
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            extra = sorted(repr(value) for value in set(state) - expected_keys)
            raise ValueError(
                f"invalid evaluation accumulator state keys; missing={missing}, extra={extra}"
            )
        if (
            isinstance(state["schema_version"], bool)
            or not isinstance(state["schema_version"], int)
            or state["schema_version"] != _EVALUATION_ACCUMULATOR_SCHEMA_VERSION
        ):
            raise ValueError("unsupported evaluation accumulator schema_version")

        if isinstance(maximum_depth_m, bool):
            raise ValueError("maximum_depth_m must be finite and positive")
        requested_maximum = float(maximum_depth_m)
        if not math.isfinite(requested_maximum) or requested_maximum <= 0.0:
            raise ValueError("maximum_depth_m must be finite and positive")
        stored_maximum = state["maximum_depth_m"]
        if isinstance(stored_maximum, bool) or not isinstance(stored_maximum, (int, float)):
            raise TypeError("state maximum_depth_m must be numeric")
        if float(stored_maximum) != requested_maximum:
            raise ValueError("state maximum_depth_m does not match requested evaluation")

        def nonnegative_integer(name: str) -> int:
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"state {name} must be a non-negative integer")
            return value

        invalid_model_output_count = nonnegative_integer(
            "invalid_model_output_count"
        )
        invalid_baseline_output_count = nonnegative_integer(
            "invalid_baseline_output_count"
        )
        batch_count = nonnegative_integer("batch_count")
        loss_sum = state["loss_sum"]
        if isinstance(loss_sum, bool) or not isinstance(loss_sum, (int, float)):
            raise TypeError("state loss_sum must be numeric")
        restored_loss_sum = float(loss_sum)
        if not math.isfinite(restored_loss_sum):
            raise ValueError("state loss_sum must be finite")

        restored: dict[str, list[Tensor]] = {}
        for name in _EVALUATION_TENSOR_FIELDS:
            values = state[name]
            if not isinstance(values, list) or len(values) != batch_count:
                raise ValueError(
                    f"state {name} must be a list with batch_count entries"
                )
            snapshots: list[Tensor] = []
            for value in values:
                if (
                    not isinstance(value, Tensor)
                    or value.device.type != "cpu"
                    or value.layout is not torch.strided
                    or value.ndim != 1
                    or value.requires_grad
                ):
                    raise ValueError(
                        f"state {name} entries must be detached 1-D strided CPU tensors"
                    )
                if name in _EVALUATION_BOOLEAN_TENSOR_FIELDS:
                    if value.dtype != torch.bool:
                        raise ValueError(f"state {name} entries must use bool dtype")
                else:
                    if not value.is_floating_point():
                        raise ValueError(
                            f"state {name} entries must be floating-point tensors"
                        )
                    if not bool(torch.isfinite(value).all().item()):
                        raise ValueError(
                            f"state {name} entries must contain only finite values"
                        )
                snapshots.append(value.clone())
            restored[name] = snapshots

        visible_groups = (
            "model_errors",
            "baseline_errors",
            "model_failures",
            "baseline_failures",
            "uncertainties",
            "occlusions",
            "jitters",
        )
        for index in range(batch_count):
            visible_lengths = {restored[name][index].numel() for name in visible_groups}
            if len(visible_lengths) != 1:
                raise ValueError(
                    "state visible-target tensor lengths disagree within a batch"
                )
            no_target_lengths = {
                restored[name][index].numel()
                for name in ("model_no_target_claims", "baseline_no_target_claims")
            }
            if len(no_target_lengths) != 1:
                raise ValueError(
                    "state no-target tensor lengths disagree within a batch"
                )

        # Construct and populate the accumulator only after the complete state
        # has passed validation.  A corrupt checkpoint therefore cannot expose
        # a partially restored object.
        result = cls(requested_maximum)
        result.invalid_model_output_count = invalid_model_output_count
        result.invalid_baseline_output_count = invalid_baseline_output_count
        result.batch_count = batch_count
        result.loss_sum = restored_loss_sum
        for name, values in restored.items():
            setattr(result, name, values)
        return result

    def finalize(self) -> dict[str, object]:
        """Compute full-split metrics from all accumulated raw observations."""

        if self.batch_count == 0:
            raise TargetStateTrainingError("evaluation loader produced no batches")
        model_error_values = torch.cat(self.model_errors)
        baseline_error_values = torch.cat(self.baseline_errors)
        common = {
            "occlusion": torch.cat(self.occlusions),
            "jitter": torch.cat(self.jitters),
        }
        return {
            "mean_loss": self.loss_sum / self.batch_count,
            "model": _metric_block(
                model_error_values,
                failures=torch.cat(self.model_failures),
                no_target_valid_claims=torch.cat(self.model_no_target_claims),
                uncertainty=torch.cat(self.uncertainties),
                invalid_output_count=self.invalid_model_output_count,
                **common,
            ),
            "deterministic_rgbd_baseline": _metric_block(
                baseline_error_values,
                failures=torch.cat(self.baseline_failures),
                no_target_valid_claims=torch.cat(self.baseline_no_target_claims),
                uncertainty=None,
                invalid_output_count=self.invalid_baseline_output_count,
                **common,
            ),
        }


@torch.no_grad()
def accumulate_evaluation(
    model: TemporalRayDepthNet,
    loader: DataLoader[dict[str, Tensor]],
    *,
    device: torch.device,
    maximum_depth_m: float,
    accumulator: TargetStateEvaluationAccumulator | None = None,
) -> TargetStateEvaluationAccumulator:
    """Evaluate one loader into a new or existing mergeable accumulator."""

    if accumulator is None:
        accumulator = TargetStateEvaluationAccumulator(maximum_depth_m)
    elif accumulator.maximum_depth_m != float(maximum_depth_m):
        raise ValueError("accumulator maximum_depth_m does not match evaluation")
    model.eval()
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        output = model(batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
        accumulator.add_batch(batch=batch, output=output)
    return accumulator


@torch.no_grad()
def evaluate_model(
    model: TemporalRayDepthNet,
    loader: DataLoader[dict[str, Tensor]],
    *,
    device: torch.device,
    maximum_depth_m: float,
) -> dict[str, object]:
    return accumulate_evaluation(
        model,
        loader,
        device=device,
        maximum_depth_m=maximum_depth_m,
    ).finalize()


def evaluate_promotion(
    metrics: Mapping[str, object],
    *,
    p95_max_ratio: float,
    minimum_covariance_correlation: float,
) -> dict[str, object]:
    model = metrics.get("model")
    baseline = metrics.get("deterministic_rgbd_baseline")
    if not isinstance(model, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError("metrics must contain model and deterministic_rgbd_baseline mappings")
    reasons: list[str] = []

    def require_better(field: str, label: str) -> None:
        candidate, reference = model.get(field), baseline.get(field)
        if candidate is None or reference is None:
            reasons.append(f"{label}: insufficient evaluation samples")
        elif float(candidate) >= float(reference):
            reasons.append(f"{label}: {candidate} is not better than baseline {reference}")

    require_better("position_median_error_m", "median position error")
    candidate_p95, baseline_p95 = model.get("position_p95_error_m"), baseline.get("position_p95_error_m")
    if candidate_p95 is None or baseline_p95 is None or float(candidate_p95) > float(baseline_p95) * p95_max_ratio:
        reasons.append("position p95 error exceeds allowed baseline ratio")
    candidate_failure, baseline_failure = model.get("measurement_failure_rate"), baseline.get("measurement_failure_rate")
    if candidate_failure is None or baseline_failure is None or float(candidate_failure) > float(baseline_failure) + 1e-12:
        reasons.append("measurement failure rate is above baseline")
    candidate_false_positive = model.get("no_target_false_positive_rate")
    baseline_false_positive = baseline.get("no_target_false_positive_rate")
    if candidate_false_positive is None or baseline_false_positive is None:
        reasons.append("no-target false-positive rate: insufficient evaluation samples")
    elif float(candidate_false_positive) > float(baseline_false_positive) + 1e-12:
        reasons.append("no-target false-positive rate is above baseline")
    require_better("occluded_position_median_error_m", "occluded position error")
    require_better("jittered_position_median_error_m", "bbox-jitter position error")
    correlation = model.get("covariance_error_spearman")
    if correlation is None or float(correlation) < minimum_covariance_correlation:
        reasons.append("covariance/error correlation is below threshold")
    if int(model.get("invalid_output_count", 1)) != 0:
        reasons.append("model produced NaN, Inf, or out-of-contract outputs")
    return {"passed": not reasons, "reasons": reasons}


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, TrainingStage):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_checkpoint(
    path: Path,
    model: TemporalRayDepthNet,
    optimizer: AdamW,
    epoch: int,
    metrics: Mapping[str, object],
    *,
    training_stage: TrainingStage,
    model_config: Mapping[str, int],
    dataset_sha256: str,
    dataset_provenance: Mapping[str, object],
) -> None:
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "schema_version": MODEL_SCHEMA_VERSION,
            "training_stage": training_stage.value,
            "model_config": dict(model_config),
            "dataset_sha256": dataset_sha256,
            "dataset_provenance": _json_safe(dataset_provenance),
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": _json_safe(metrics),
        },
        path,
    )


def _writer(log_dir: Path):
    # The repository's scalar-only TFEvent writer has no optional TensorBoard
    # runtime dependency and intentionally exposes no image/video API.
    return ScalarEventWriter(log_dir)


def _write_figures(run_dir: Path, metrics: Mapping[str, object], count: int) -> None:
    if count <= 0:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    model = metrics["model"]
    baseline = metrics["deterministic_rgbd_baseline"]
    fields = ("position_median_error_m", "position_p95_error_m", "measurement_failure_rate")
    for index, field in enumerate(fields[:count]):
        values = [model.get(field), baseline.get(field)]
        if any(value is None for value in values):
            continue
        figure, axis = plt.subplots(figsize=(4, 3))
        axis.bar(("temporal", "deterministic"), values)
        axis.set_title(field)
        figure.tight_layout()
        figure.savefig(run_dir / f"evaluation_{index + 1}_{field}.png", dpi=120)
        plt.close(figure)


def train_target_state(
    config: TargetStateTrainingConfig,
    *,
    datasets: Mapping[str, Dataset[dict[str, Tensor]]] | None = None,
    dataset_sha256: str | None = None,
) -> TargetStateTrainingResult:
    started = time.monotonic()
    _seed_everything(config.seed)
    device = _device(config.device)
    initial, initial_checkpoint_sha256 = validate_initial_checkpoint(
        config, map_location=device
    )
    if datasets is None:
        datasets = {
            split: TargetStateTorchDataset(config, split=split)
            for split in ("train", "validation", "test")
        }
    for split in ("train", "validation", "test"):
        if split not in datasets or len(datasets[split]) == 0:
            raise TargetStateTrainingError(f"{split} split has no temporal sequences")
    if dataset_sha256 is None:
        records = read_frame_records(config.dataset_root / "frames.jsonl")
        dataset_sha256 = compute_dataset_sha256(config.dataset_root, records)
    if len(dataset_sha256) != 64 or any(character not in "0123456789abcdef" for character in dataset_sha256.lower()):
        raise TargetStateTrainingError("dataset_sha256 must contain 64 hexadecimal characters")
    dataset_sha256 = dataset_sha256.lower()
    dataset_manifest: Mapping[str, object] = {}
    if config.require_dataset_manifest:
        raw_dataset_manifest = json.loads(
            (config.dataset_root / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        if not isinstance(raw_dataset_manifest, Mapping):
            raise TargetStateTrainingError("dataset_manifest.json must contain a JSON object")
        dataset_manifest = raw_dataset_manifest
        if dataset_manifest.get("dataset_sha256") != dataset_sha256:
            raise TargetStateTrainingError(
                "dataset content SHA256 does not match dataset_manifest.json"
            )
        if config.stage is TrainingStage.YOLO_DEPLOYMENT:
            source = dataset_manifest.get("detector_prediction_source")
            actual_yolo_sha = dataset_manifest.get("yolo_model_sha256")
            if source != "real_yolo_deployment_output":
                raise TargetStateTrainingError(
                    "stage-B dataset detector_prediction_source must be "
                    "real_yolo_deployment_output"
                )
            if actual_yolo_sha != config.expected_yolo_model_sha256:
                raise TargetStateTrainingError(
                    "stage-B dataset YOLO identity mismatch: "
                    f"expected={config.expected_yolo_model_sha256}, "
                    f"actual={actual_yolo_sha}"
                )
            deployment = dataset_manifest.get("detector_deployment")
            expected_names = {"0": "cube"}
            if not isinstance(deployment, Mapping) or not (
                deployment.get("preflight_verified") is True
                and deployment.get("model_family") == "yolo"
                and deployment.get("model_names") == expected_names
                and deployment.get("model_sha256") == config.expected_yolo_model_sha256
            ):
                raise TargetStateTrainingError(
                    "stage-B dataset has no valid preflight-verified YOLO deployment receipt"
                )
    dataset_provenance = {
        "manifest_verified": config.require_dataset_manifest,
        "schema_version": dataset_manifest.get("schema_version"),
        "detector_prediction_source": dataset_manifest.get("detector_prediction_source"),
        "yolo_model_sha256": dataset_manifest.get("yolo_model_sha256"),
        "oracle_usage": dataset_manifest.get("oracle_usage"),
        "generation_commit_sha": dataset_manifest.get("generation_commit_sha"),
        "detector_deployment": dataset_manifest.get("detector_deployment"),
    }
    run_dir = config.output_dir / config.run_name
    if run_dir.exists():
        if not run_dir.is_dir() or any(run_dir.iterdir()):
            raise TargetStateTrainingError(
                f"training output directory must be new or empty: {run_dir}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    train_generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            generator=train_generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    model_config = {
        "geometry_input_dim": config.geometry_input_dim,
        "roi_channels": 4,
        "roi_size_px": config.roi_size_px,
        "roi_feature_dim": config.roi_feature_dim,
        "geometry_feature_dim": config.geometry_feature_dim,
        "hidden_dim": config.hidden_dim,
        "gru_layers": config.gru_layers,
        "time_steps": config.history_size + 1,
    }
    model = TemporalRayDepthNet(
        geometry_input_dim=config.geometry_input_dim,
        roi_feature_dim=config.roi_feature_dim,
        geometry_feature_dim=config.geometry_feature_dim,
        hidden_dim=config.hidden_dim,
        gru_layers=config.gru_layers,
    ).to(device)
    if initial is not None:
        try:
            model.load_state_dict(initial["model_state_dict"], strict=True)
        except (KeyError, RuntimeError) as exc:
            raise TargetStateTrainingError(f"initial checkpoint architecture is incompatible: {exc}") from exc
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_path, latest_path = run_dir / "best.pt", run_dir / "latest.pt"
    csv_path = run_dir / "metrics.csv"
    best_loss = float("inf")
    last_validation: dict[str, object] = {}
    with csv_path.open("w", newline="", encoding="utf-8") as csv_stream, _writer(run_dir / "tensorboard") as tensorboard:
        fieldnames = [
            "epoch", "train_total", "train_depth", "train_position_3d",
            "train_reprojection", "train_gaussian_nll", "train_validity_bce",
            "validation_loss", "validation_position_median_error_m",
            "validation_position_p95_error_m", "validation_measurement_failure_rate",
            "validation_no_target_false_positive_rate",
            "validation_covariance_error_spearman",
        ]
        csv_writer = csv.DictWriter(csv_stream, fieldnames=fieldnames)
        csv_writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            model.train()
            accumulators = {name: 0.0 for name in ("total", "depth", "position_3d", "reprojection", "gaussian_nll", "validity_bce")}
            batch_count = 0
            for raw_batch in loaders["train"]:
                batch = _to_device(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
                losses = compute_target_state_losses(output, batch, weights=config.loss_weights)
                if not torch.isfinite(losses.total):
                    raise TargetStateTrainingError(f"non-finite training loss at epoch {epoch}")
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                accumulators["total"] += float(losses.total.detach().cpu())
                accumulators["depth"] += float(losses.depth_huber.detach().cpu())
                accumulators["position_3d"] += float(losses.position_3d_huber.detach().cpu())
                accumulators["reprojection"] += float(losses.reprojection_huber.detach().cpu())
                accumulators["gaussian_nll"] += float(losses.gaussian_nll.detach().cpu())
                accumulators["validity_bce"] += float(losses.validity_bce.detach().cpu())
                batch_count += 1
            averages = {name: value / max(batch_count, 1) for name, value in accumulators.items()}
            if epoch % config.validation_interval == 0 or epoch == config.epochs:
                last_validation = evaluate_model(
                    model, loaders["validation"], device=device, maximum_depth_m=config.maximum_depth_m
                )
            validation_loss = float(last_validation.get("mean_loss", float("inf")))
            _save_checkpoint(
                latest_path,
                model,
                optimizer,
                epoch,
                last_validation,
                training_stage=config.stage,
                model_config=model_config,
                dataset_sha256=dataset_sha256,
                dataset_provenance=dataset_provenance,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                _save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    last_validation,
                    training_stage=config.stage,
                    model_config=model_config,
                    dataset_sha256=dataset_sha256,
                    dataset_provenance=dataset_provenance,
                )
            validation_model = last_validation.get("model")
            validation_model = validation_model if isinstance(validation_model, Mapping) else {}
            csv_writer.writerow({
                "epoch": epoch,
                "train_total": averages["total"],
                "train_depth": averages["depth"],
                "train_position_3d": averages["position_3d"],
                "train_reprojection": averages["reprojection"],
                "train_gaussian_nll": averages["gaussian_nll"],
                "train_validity_bce": averages["validity_bce"],
                "validation_loss": last_validation.get("mean_loss"),
                "validation_position_median_error_m": validation_model.get("position_median_error_m"),
                "validation_position_p95_error_m": validation_model.get("position_p95_error_m"),
                "validation_measurement_failure_rate": validation_model.get("measurement_failure_rate"),
                "validation_no_target_false_positive_rate": validation_model.get("no_target_false_positive_rate"),
                "validation_covariance_error_spearman": validation_model.get("covariance_error_spearman"),
            })
            csv_stream.flush()
            for name, value in averages.items():
                tensorboard.add_scalar(f"train/{name}", value, epoch)
            tensorboard.add_scalar("validation/loss", validation_loss, epoch)
            if validation_model:
                for field in (
                    "position_median_error_m",
                    "position_p95_error_m",
                    "measurement_failure_rate",
                    "no_target_false_positive_rate",
                    "covariance_error_spearman",
                ):
                    value = validation_model.get(field)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        tensorboard.add_scalar(f"validation/{field}", float(value), epoch)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics = evaluate_model(model, loaders["validation"], device=device, maximum_depth_m=config.maximum_depth_m)
    test_metrics = evaluate_model(model, loaders["test"], device=device, maximum_depth_m=config.maximum_depth_m)
    validation_gate = evaluate_promotion(
        validation_metrics,
        p95_max_ratio=config.promotion_p95_max_ratio,
        minimum_covariance_correlation=config.promotion_min_covariance_correlation,
    )
    test_gate = evaluate_promotion(
        test_metrics,
        p95_max_ratio=config.promotion_p95_max_ratio,
        minimum_covariance_correlation=config.promotion_min_covariance_correlation,
    )
    stage_ok = config.stage is TrainingStage.YOLO_DEPLOYMENT
    stage_a_initialized = (
        config.initial_checkpoint_path is not None
        and initial_checkpoint_sha256 is not None
    )
    protocol_ok = stage_ok and stage_a_initialized and config.require_dataset_manifest
    promoted = protocol_ok and bool(validation_gate["passed"]) and bool(test_gate["passed"])
    checkpoint_digest = sha256_file(best_path)
    config_snapshot = _json_safe(asdict(config))
    manifest = {
        "model_type": MODEL_TYPE,
        "schema_version": MODEL_SCHEMA_VERSION,
        "checkpoint_path": str(best_path.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "dataset_sha256": dataset_sha256,
        "dataset_provenance": dataset_provenance,
        "training_commit_sha": os.environ.get("UAV_AGENT_TRAINING_COMMIT_SHA", "nogit"),
        "artifacts": {
            "best_checkpoint": str(best_path.resolve()),
            "latest_checkpoint": str(latest_path.resolve()),
            "metrics_csv": str(csv_path.resolve()),
            "tensorboard_dir": str((run_dir / "tensorboard").resolve()),
            "maximum_result_figures": config.save_figures,
            "videos_saved": False,
        },
        "input_fields": {
            "roi_rgbd": ["red", "green", "blue", "normalized_depth"],
            "geometry_25d": list(GEOMETRY_INPUT_FIELDS),
            "missing_mask": True,
        },
        "input_semantics": {
            "roi_source": "detector_bbox_crop",
            "camera_relative_pose_source": "synchronized_camera_pose_including_extrinsics",
            "uav_linear_velocity_frame": "world",
            "uav_angular_velocity_frame": "body",
            "delta_t_reference": "reference_timestamp_minus_frame_timestamp",
            "tracker_continuity_source": "previous_non_missing_tracker_id_match",
        },
        "output_fields": list(OUTPUT_FIELDS),
        "model_config": model_config,
        "preprocessing": {
            "rgb_scale": 255.0,
            "depth_scale_m": config.maximum_depth_m,
            "minimum_depth_m": config.minimum_depth_m,
            "maximum_depth_m": config.maximum_depth_m,
            "roi_interpolation": "bilinear_align_corners_false",
            "baseline_depth_sampling": "foreground_cluster_median",
            "foreground_inset_ratio": 0.1,
            "foreground_bottom_exclusion_ratio": 0.15,
            "foreground_min_valid_samples": 3,
            "foreground_seed_patch_radius_px": 4,
        },
        "history_size": config.history_size,
        "max_history_age_s": config.max_history_age_s,
        "camera_convention": config.camera_convention,
        "coordinate_convention": config.coordinate_convention,
        "training_stage": config.stage.value,
        "initial_checkpoint": (
            None
            if config.initial_checkpoint_path is None
            else {
                "path": str(config.initial_checkpoint_path),
                "sha256": initial_checkpoint_sha256,
                "training_stage": (
                    initial.get("training_stage") if initial is not None else None
                ),
            }
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "promotion": {
            "passed": promoted,
            "requires_yolo_deployment_stage": True,
            "stage_satisfied": stage_ok,
            "requires_stage_a_initialization": True,
            "stage_a_initialization_satisfied": stage_a_initialized,
            "requires_verified_dataset_manifest": True,
            "dataset_manifest_satisfied": config.require_dataset_manifest,
            "validation": validation_gate,
            "test": test_gate,
        },
        "config": config_snapshot,
        "torch_version": torch.__version__,
    }
    manifest_path = run_dir / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_figures(run_dir, test_metrics, config.save_figures)
    return TargetStateTrainingResult(
        run_dir=run_dir,
        best_checkpoint=best_path,
        latest_checkpoint=latest_path,
        model_manifest=manifest_path,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        promoted=promoted,
        elapsed_s=time.monotonic() - started,
    )


__all__ = [
    "MODEL_SCHEMA_VERSION", "MODEL_TYPE", "OUTPUT_FIELDS", "TargetStateTrainingError",
    "TargetStateEvaluationAccumulator", "TargetStateTrainingResult",
    "accumulate_evaluation", "evaluate_model", "evaluate_promotion", "sha256_file",
    "train_target_state", "validate_initial_checkpoint",
]
