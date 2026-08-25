"""Production temporal ray-depth measurement resolver.

This module consumes only bounded RGB-D frames, detector history, camera
calibration/poses, and synchronized UAV self-motion.  It never imports the
offline target-state label schema or an evaluator capability.  The learned
network predicts pixel/depth residuals and covariance; deterministic geometry
still creates the world-space :class:`TargetMeasurement`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import acos, isfinite
from pathlib import Path
from threading import RLock
from typing import Mapping, Sequence

import numpy as np

from common.ids import validate_routing_id, validate_uav_id
from perception.candidate_bank import CandidateSnapshot
from perception.depth_geometry import (
    DepthCandidateResolver,
    DepthSamplingStrategy,
    backproject_pixel_to_camera_optical,
    camera_flu_to_world,
    optical_to_camera_flu,
)
from perception.grounding import CandidateResolutionUnavailable
from perception.measurement import TargetMeasurement
from perception.runtime import PerceptionRuntimeProfile
from runtime.frame_store import (
    FrameCameraGeometry,
    FrameRef,
    FrameStore,
    FrameUavSelfMotion,
)


TEMPORAL_GEOMETRY_FIELDS = (
    "bbox_x1_normalized", "bbox_y1_normalized", "bbox_x2_normalized",
    "bbox_y2_normalized", "detector_confidence", "tracker_continuity",
    "fx_normalized", "fy_normalized", "cx_normalized", "cy_normalized",
    "camera_relative_x_m", "camera_relative_y_m", "camera_relative_z_m",
    "camera_relative_orientation_w", "camera_relative_orientation_x",
    "camera_relative_orientation_y", "camera_relative_orientation_z",
    "uav_velocity_x_mps", "uav_velocity_y_mps", "uav_velocity_z_mps",
    "uav_angular_x_radps", "uav_angular_y_radps", "uav_angular_z_radps",
    "delta_t_s", "missing",
)
TEMPORAL_OUTPUT_FIELDS = (
    "delta_u_px",
    "delta_v_px",
    "depth_residual_m",
    "position_log_variance_xyz",
    "measurement_valid_logit",
)
CAMERA_CONVENTION = "camera_optical_x_right_y_down_z_forward"
COORDINATE_CONVENTION = "world_flu_x_forward_y_left_z_up"
INPUT_SEMANTICS = {
    "roi_source": "detector_bbox_crop",
    "camera_relative_pose_source": (
        "synchronized_camera_pose_including_extrinsics"
    ),
    "uav_linear_velocity_frame": "world",
    "uav_angular_velocity_frame": "body",
    "tracker_continuity_source": "previous_non_missing_tracker_id_match",
    "delta_t_reference": "reference_timestamp_minus_frame_timestamp",
}
_MAX_REASON_KEYS = 16


@dataclass(frozen=True, slots=True)
class TemporalRayDepthArtifactInfo:
    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    schema_version: int
    history_size: int
    max_history_age_s: float
    roi_size_px: int
    model_config: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TemporalRayDepthStatistics:
    attempts: int
    successes: int
    fallback_total: int
    unavailable_total: int
    invalid_output_total: int
    reset_total: int
    fallback_reasons: Mapping[str, int]
    unavailable_reasons: Mapping[str, int]
    last_fallback_reason: str | None
    last_unavailable_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "temporal_resolution_attempts": self.attempts,
            "temporal_resolution_successes": self.successes,
            "temporal_fallback_total": self.fallback_total,
            "temporal_unavailable_total": self.unavailable_total,
            "temporal_invalid_output_total": self.invalid_output_total,
            "temporal_reset_total": self.reset_total,
            "temporal_fallback_reasons": dict(self.fallback_reasons),
            "temporal_unavailable_reasons": dict(self.unavailable_reasons),
            "temporal_last_fallback_reason": self.last_fallback_reason,
            "temporal_last_unavailable_reason": self.last_unavailable_reason,
        }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite positive number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return result


def _expected_digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    return value.casefold()


def _manifest_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"temporal model manifest {field} must be a mapping")
    return value


def _validate_json_finite(value: object, field: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not isfinite(float(value)):
            raise ValueError(f"temporal model manifest {field} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"temporal model manifest {field} keys must be strings"
                )
            _validate_json_finite(item, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_finite(item, f"{field}[{index}]")
        return
    raise TypeError(f"temporal model manifest {field} is not JSON-compatible")


def _positive_manifest_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"temporal model manifest {field} must be a positive integer")
    return value


def _validate_no_target_receipt(
    metrics: Mapping[str, object],
    *,
    field: str,
) -> None:
    model = _manifest_mapping(metrics.get("model"), f"{field}.model")
    baseline = _manifest_mapping(
        metrics.get("deterministic_rgbd_baseline"),
        f"{field}.deterministic_rgbd_baseline",
    )
    model_count = model.get("evaluated_no_target_count")
    baseline_count = baseline.get("evaluated_no_target_count")
    if (
        isinstance(model_count, bool)
        or not isinstance(model_count, int)
        or model_count <= 0
        or isinstance(baseline_count, bool)
        or not isinstance(baseline_count, int)
        or baseline_count <= 0
    ):
        raise ValueError(
            f"temporal model manifest {field} has no verified no-target samples"
        )
    model_rate = model.get("no_target_false_positive_rate")
    baseline_rate = baseline.get("no_target_false_positive_rate")
    if (
        isinstance(model_rate, bool)
        or not isinstance(model_rate, (int, float))
        or not isfinite(float(model_rate))
        or not 0.0 <= float(model_rate) <= 1.0
        or isinstance(baseline_rate, bool)
        or not isinstance(baseline_rate, (int, float))
        or not isfinite(float(baseline_rate))
        or not 0.0 <= float(baseline_rate) <= 1.0
    ):
        raise ValueError(
            f"temporal model manifest {field} has invalid no-target rates"
        )
    if float(model_rate) > float(baseline_rate) + 1e-12:
        raise ValueError(
            f"temporal model manifest {field} no-target rate exceeds baseline"
        )


def _reason_code(value: object) -> str:
    text = str(value).splitlines()[0].strip().casefold()
    if text.startswith("temporal_ray_depth_unavailable:"):
        text = text.split(":", 1)[1]
    # Variable numeric suffixes (for example a validity probability) must not
    # create an unbounded cardinality metric.  Keep the stable reason prefix.
    text = text.split(":", 1)[0]
    normalized = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in text
    ).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return (normalized or type(value).__name__.casefold())[:96]


def _bounded_counter_increment(counter: Counter[str], reason: str) -> None:
    key = reason
    if key not in counter and len(counter) >= _MAX_REASON_KEYS:
        key = "other_temporal_failure"
    counter[key] += 1


def _load_artifact(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    manifest_path: str | Path | None,
    device: str,
    min_depth_m: float,
    max_depth_m: float,
    sampling_strategy: DepthSamplingStrategy,
    patch_radius_px: int,
) -> tuple[TemporalRayDepthArtifactInfo, object]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    expected = _expected_digest(expected_sha256)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"temporal checkpoint does not exist: {checkpoint}")
    actual = _file_sha256(checkpoint)
    if actual != expected:
        raise ValueError(
            "temporal checkpoint SHA256 mismatch: "
            f"expected={expected}, actual={actual}, path={checkpoint}"
        )
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else checkpoint.parent / "model_manifest.json"
    )
    if not manifest.is_file():
        raise FileNotFoundError(f"temporal model manifest does not exist: {manifest}")
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read temporal model manifest: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TypeError("temporal model manifest root must be a mapping")
    required = {
        "model_type", "schema_version", "checkpoint_path", "checkpoint_sha256",
        "dataset_sha256", "training_commit_sha", "input_fields", "output_fields",
        "history_size", "max_history_age_s", "camera_convention",
        "coordinate_convention", "validation_metrics", "test_metrics",
        "model_config", "preprocessing", "input_semantics", "training_stage",
        "promotion",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"temporal model manifest is missing fields: {missing}")
    if raw["model_type"] != "temporal_ray_depth_residual":
        raise ValueError("temporal model manifest has an unsupported model_type")
    if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
        raise ValueError("temporal model manifest schema_version must be 1")
    if _expected_digest(raw["checkpoint_sha256"]) != actual:
        raise ValueError("temporal model manifest checkpoint_sha256 does not match checkpoint")
    if not isinstance(raw["checkpoint_path"], str) or not raw["checkpoint_path"].strip():
        raise ValueError("temporal model manifest checkpoint_path must be a non-empty string")
    try:
        manifest_checkpoint = Path(raw["checkpoint_path"]).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        raise ValueError("temporal model manifest checkpoint_path is invalid") from exc
    if manifest_checkpoint != checkpoint:
        raise ValueError(
            "temporal model manifest checkpoint_path does not match configured checkpoint"
        )
    _expected_digest(raw["dataset_sha256"])
    training_commit = raw["training_commit_sha"]
    if not isinstance(training_commit, str) or not training_commit.strip():
        raise ValueError("temporal model manifest training_commit_sha must be non-empty")
    expected_input_fields = {
        "roi_rgbd": ["red", "green", "blue", "normalized_depth"],
        "geometry_25d": list(TEMPORAL_GEOMETRY_FIELDS),
        "missing_mask": True,
    }
    if raw["input_fields"] != expected_input_fields:
        raise ValueError("temporal model manifest input_fields do not match runtime schema")
    if tuple(raw["output_fields"]) != TEMPORAL_OUTPUT_FIELDS:
        raise ValueError("temporal model manifest output_fields do not match runtime schema")
    if raw["camera_convention"] != CAMERA_CONVENTION:
        raise ValueError("temporal model camera coordinate convention mismatch")
    if raw["coordinate_convention"] != COORDINATE_CONVENTION:
        raise ValueError("temporal model world coordinate convention mismatch")
    if raw["input_semantics"] != INPUT_SEMANTICS:
        raise ValueError("temporal model manifest input_semantics do not match runtime")
    if raw["training_stage"] != "yolo_deployment":
        raise ValueError(
            "production temporal model must come from training_stage=yolo_deployment"
        )
    promotion = _manifest_mapping(raw["promotion"], "promotion")
    required_promotion_truths = (
        "passed",
        "requires_yolo_deployment_stage",
        "stage_satisfied",
        "requires_stage_a_initialization",
        "stage_a_initialization_satisfied",
        "requires_verified_dataset_manifest",
        "dataset_manifest_satisfied",
    )
    if any(promotion.get(field) is not True for field in required_promotion_truths):
        raise ValueError(
            "production temporal model manifest has not passed every promotion gate"
        )
    validation_metrics = _manifest_mapping(
        raw["validation_metrics"], "validation_metrics"
    )
    test_metrics = _manifest_mapping(raw["test_metrics"], "test_metrics")
    _validate_json_finite(validation_metrics, "validation_metrics")
    _validate_json_finite(test_metrics, "test_metrics")
    _validate_no_target_receipt(
        validation_metrics,
        field="validation_metrics",
    )
    _validate_no_target_receipt(test_metrics, field="test_metrics")
    for split in ("validation", "test"):
        gate = _manifest_mapping(promotion.get(split), f"promotion.{split}")
        if gate.get("passed") is not True or gate.get("reasons") != []:
            raise ValueError(
                f"temporal model manifest promotion.{split} did not pass"
            )
    history_size = raw["history_size"]
    if isinstance(history_size, bool) or not isinstance(history_size, int) or not 4 <= history_size <= 8:
        raise ValueError("temporal model history_size must be within [4, 8]")
    max_age = _finite_positive(raw["max_history_age_s"], "manifest.max_history_age_s")
    model_config_raw = _manifest_mapping(raw["model_config"], "model_config")
    allowed_config = {
        "geometry_input_dim", "roi_feature_dim", "geometry_feature_dim",
        "hidden_dim", "gru_layers", "roi_channels", "roi_size_px", "time_steps",
    }
    if set(model_config_raw) - allowed_config:
        raise ValueError("temporal model manifest contains unknown model_config fields")
    required_model_config = allowed_config
    missing_model_config = required_model_config - set(model_config_raw)
    if missing_model_config:
        raise ValueError(
            "temporal model manifest model_config is missing fields: "
            f"{sorted(missing_model_config)}"
        )
    normalized_model_config = {
        field: _positive_manifest_int(model_config_raw[field], f"model_config.{field}")
        for field in allowed_config
    }
    model_config = {
        field: normalized_model_config[field]
        for field in (
            "geometry_input_dim", "roi_feature_dim", "geometry_feature_dim",
            "hidden_dim", "gru_layers", "roi_channels",
        )
    }
    roi_size = normalized_model_config["roi_size_px"]
    if model_config["geometry_input_dim"] != 25 or model_config["roi_channels"] != 4:
        raise ValueError("temporal model deployable input dimensions must be geometry=25, ROI channels=4")
    if roi_size < 32 or roi_size > 512:
        raise ValueError("temporal model roi_size_px must be within [32, 512]")
    if normalized_model_config["time_steps"] != history_size + 1:
        raise ValueError("temporal model time_steps must equal history_size + 1")

    preprocessing = _manifest_mapping(raw["preprocessing"], "preprocessing")
    expected_preprocessing = {
        "rgb_scale": 255.0,
        "depth_scale_m": max_depth_m,
        "minimum_depth_m": min_depth_m,
        "maximum_depth_m": max_depth_m,
        "roi_interpolation": "bilinear_align_corners_false",
        "baseline_depth_sampling": "foreground_cluster_median",
        "foreground_inset_ratio": 0.1,
        "foreground_bottom_exclusion_ratio": 0.15,
        "foreground_min_valid_samples": 3,
        "foreground_seed_patch_radius_px": 4,
    }
    if dict(preprocessing) != expected_preprocessing:
        raise ValueError(
            "temporal model manifest preprocessing does not match runtime"
        )
    if sampling_strategy is not DepthSamplingStrategy.FOREGROUND_CLUSTER_MEDIAN:
        raise ValueError(
            "temporal_ray_depth requires depth_anchor=foreground_cluster_median"
        )
    if patch_radius_px != 4:
        raise ValueError(
            "temporal_ray_depth requires depth_patch_radius_px=4 to match training"
        )

    try:
        import torch
        from training.target_state.model import TemporalRayDepthNet
    except Exception as exc:  # pragma: no cover - depends on deployment env
        raise RuntimeError(f"temporal model runtime dependencies are unavailable: {exc}") from exc
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint payload must be a mapping")
        if payload.get("model_type") != "temporal_ray_depth_residual":
            raise ValueError("checkpoint model_type does not match manifest")
        checkpoint_schema = payload.get("schema_version")
        if isinstance(checkpoint_schema, bool) or checkpoint_schema != 1:
            raise ValueError("checkpoint schema_version does not match manifest")
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint has no model_state_dict")
        model = TemporalRayDepthNet(**model_config)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        with torch.inference_mode():
            output = model(
                torch.zeros(1, history_size + 1, 4, roi_size, roi_size, device=device),
                torch.zeros(1, history_size + 1, 25, device=device),
                torch.zeros(1, history_size + 1, dtype=torch.bool, device=device),
            )
        output_values = output.as_dict()
        expected_shapes = {
            "delta_uv_px": (1, 2),
            "depth_residual_m": (1,),
            "position_log_variance": (1, 3),
            "measurement_valid_logit": (1,),
        }
        if {
            name: tuple(value.shape) for name, value in output_values.items()
        } != expected_shapes:
            raise ValueError("temporal model dry-run output shapes are incompatible")
        if not all(torch.isfinite(value).all() for value in output_values.values()):
            raise ValueError("temporal model dry-run produced a non-finite output")
    except Exception as exc:
        raise RuntimeError(f"temporal model dry-run failed: {type(exc).__name__}: {exc}") from exc
    return (
        TemporalRayDepthArtifactInfo(
            checkpoint_path=checkpoint,
            checkpoint_sha256=actual,
            manifest_path=manifest,
            schema_version=1,
            history_size=history_size,
            max_history_age_s=max_age,
            roi_size_px=roi_size,
            model_config=model_config,
        ),
        model,
    )


class TemporalRayDepthResolver:
    """Candidate-isolated temporal inference with explicit RGB-D fallback."""

    def __init__(
        self,
        frame_store: FrameStore,
        *,
        checkpoint_path: str | Path,
        expected_sha256: str,
        manifest_path: str | Path | None = None,
        history_size: int = 6,
        max_history_age_s: float = 2.0,
        roi_size_px: int = 128,
        use_rgb: bool = True,
        use_depth: bool = True,
        deterministic_fallback: bool = True,
        device: str = "cpu",
        min_depth_m: float = 0.2,
        max_depth_m: float = 200.0,
        sampling_strategy: DepthSamplingStrategy | str = (
            DepthSamplingStrategy.FOREGROUND_CLUSTER_MEDIAN
        ),
        patch_radius_px: int = 4,
        fallback_resolver: DepthCandidateResolver | None = None,
    ) -> None:
        if not isinstance(frame_store, FrameStore):
            raise TypeError("frame_store must be a FrameStore")
        if isinstance(history_size, bool) or not isinstance(history_size, int) or not 4 <= history_size <= 8:
            raise ValueError("history_size must be within [4, 8]")
        max_age = _finite_positive(max_history_age_s, "max_history_age_s")
        if isinstance(roi_size_px, bool) or not isinstance(roi_size_px, int) or not 32 <= roi_size_px <= 512:
            raise ValueError("roi_size_px must be within [32, 512]")
        if not isinstance(use_rgb, bool) or not isinstance(use_depth, bool) or not isinstance(deterministic_fallback, bool):
            raise TypeError("use_rgb/use_depth/deterministic_fallback must be bool")
        if not use_rgb and not use_depth:
            raise ValueError("at least one of use_rgb/use_depth must be enabled")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be non-empty")
        minimum = _finite_positive(min_depth_m, "min_depth_m")
        maximum = _finite_positive(max_depth_m, "max_depth_m")
        if maximum <= minimum:
            raise ValueError("max_depth_m must be greater than min_depth_m")
        try:
            normalized_strategy = (
                sampling_strategy
                if isinstance(sampling_strategy, DepthSamplingStrategy)
                else DepthSamplingStrategy(str(sampling_strategy).strip().lower())
            )
        except ValueError as exc:
            raise ValueError("unsupported temporal fallback sampling_strategy") from exc
        if isinstance(patch_radius_px, bool) or not isinstance(patch_radius_px, int):
            raise TypeError("patch_radius_px must be an integer")
        if patch_radius_px <= 0:
            raise ValueError("patch_radius_px must be positive")
        artifact, model = _load_artifact(
            checkpoint_path,
            expected_sha256=expected_sha256,
            manifest_path=manifest_path,
            device=device.strip(),
            min_depth_m=minimum,
            max_depth_m=maximum,
            sampling_strategy=normalized_strategy,
            patch_radius_px=patch_radius_px,
        )
        if artifact.history_size != history_size:
            raise ValueError(
                f"configured history_size={history_size} does not match manifest={artifact.history_size}"
            )
        if abs(artifact.max_history_age_s - max_age) > 1e-9:
            raise ValueError("configured max_history_age_s does not match manifest")
        if artifact.roi_size_px != roi_size_px:
            raise ValueError("configured roi_size_px does not match manifest")
        self._frame_store = frame_store
        self._artifact = artifact
        self._model = model
        self._history_size = history_size
        self._sequence_length = history_size + 1
        self._max_history_age_s = max_age
        self._roi_size_px = roi_size_px
        self._use_rgb = use_rgb
        self._use_depth = use_depth
        self._deterministic_fallback = deterministic_fallback
        self._device = device.strip()
        self._min_depth_m = minimum
        self._max_depth_m = maximum
        self._fallback = fallback_resolver or DepthCandidateResolver(
            frame_store,
            sampling_strategy=normalized_strategy,
            patch_radius_px=patch_radius_px,
            min_depth_m=minimum,
            max_depth_m=maximum,
            source="rgbd_depth_geometry_fallback",
        )
        self._active_uav_id: str | None = None
        self._assignment_id: str | None = None
        self._attempts = 0
        self._successes = 0
        self._fallback_total = 0
        self._unavailable_total = 0
        self._invalid_total = 0
        self._reset_total = 0
        self._fallback_reasons: Counter[str] = Counter()
        self._unavailable_reasons: Counter[str] = Counter()
        self._last_fallback_reason: str | None = None
        self._last_unavailable_reason: str | None = None
        self._lock = RLock()

    @property
    def profile(self) -> PerceptionRuntimeProfile:
        return PerceptionRuntimeProfile.PRODUCTION

    @property
    def artifact_info(self) -> TemporalRayDepthArtifactInfo:
        return self._artifact

    @property
    def statistics(self) -> TemporalRayDepthStatistics:
        with self._lock:
            return TemporalRayDepthStatistics(
                self._attempts,
                self._successes,
                self._fallback_total,
                self._unavailable_total,
                self._invalid_total,
                self._reset_total,
                dict(self._fallback_reasons),
                dict(self._unavailable_reasons),
                self._last_fallback_reason,
                self._last_unavailable_reason,
            )

    def reset(self, *, uav_id: str, assignment_id: str | None = None) -> None:
        normalized_uav = validate_uav_id(uav_id)
        if assignment_id is None:
            raise ValueError("temporal_ray_depth reset requires assignment_id")
        normalized_assignment = validate_routing_id(
            assignment_id,
            "assignment_id",
        )
        with self._lock:
            # FrameStore is the sole bounded history owner.  Clearing this
            # UAV on every Assignment reset prevents previous-Assignment
            # RGB-D or self-motion from being reused even when frame/track
            # identifiers restart.
            previous_uav = self._active_uav_id
            if previous_uav is not None and previous_uav != normalized_uav:
                self._frame_store.clear(uav_id=previous_uav)
            self._frame_store.clear(uav_id=normalized_uav)
            self._active_uav_id = normalized_uav
            self._assignment_id = normalized_assignment
            self._reset_total += 1

    def resolve(
        self,
        candidate: CandidateSnapshot,
        *,
        timestamp_s: float,
    ) -> TargetMeasurement:
        if not isinstance(candidate, CandidateSnapshot):
            raise TypeError("candidate must be a CandidateSnapshot")
        timestamp = float(timestamp_s)
        if not isfinite(timestamp) or timestamp < candidate.last_seen_timestamp_s:
            raise ValueError("timestamp_s must be finite and cannot predate candidate")
        with self._lock:
            self._attempts += 1
            if self._active_uav_id is None or self._assignment_id is None:
                self._record_unavailable("assignment_reset_required")
                raise CandidateResolutionUnavailable(
                    "temporal resolver must be reset for an Assignment before use"
                )
            if candidate.uav_id != self._active_uav_id:
                self._record_unavailable("uav_binding_mismatch")
                raise CandidateResolutionUnavailable(
                    "temporal history is bound to another UAV; reset is required"
                )
        if timestamp - candidate.last_seen_timestamp_s > self._max_history_age_s:
            self._record_unavailable("candidate_history_expired")
            raise CandidateResolutionUnavailable(
                "temporal_ray_depth_unavailable:candidate_history_expired"
            )
        try:
            baseline = self._fallback.resolve(candidate, timestamp_s=timestamp)
        except CandidateResolutionUnavailable as exc:
            reason = "deterministic_rgbd_unavailable_" + _reason_code(exc)
            self._record_unavailable(reason)
            raise CandidateResolutionUnavailable(reason) from exc
        try:
            roi, geometry_features, missing_mask = self._build_model_inputs(candidate)
            measurement = self._infer_measurement(candidate, baseline, roi, geometry_features, missing_mask)
        except (
            CandidateResolutionUnavailable,
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
        ) as exc:
            reason = _reason_code(exc)
            with self._lock:
                self._invalid_total += int(not isinstance(exc, CandidateResolutionUnavailable))
            self._record_unavailable(reason)
            return self._fallback_or_raise(candidate, timestamp, reason, baseline=baseline)
        with self._lock:
            self._successes += 1
        return measurement

    def _fallback_or_raise(
        self,
        candidate: CandidateSnapshot,
        timestamp_s: float,
        reason: str,
        *,
        baseline: TargetMeasurement | None = None,
    ) -> TargetMeasurement:
        normalized_reason = _reason_code(reason)
        if not self._deterministic_fallback:
            raise CandidateResolutionUnavailable(
                f"temporal_ray_depth_unavailable:{normalized_reason}"
            )
        try:
            measurement = baseline or self._fallback.resolve(
                candidate,
                timestamp_s=timestamp_s,
            )
        except CandidateResolutionUnavailable as exc:
            fallback_reason = "deterministic_rgbd_unavailable_" + _reason_code(exc)
            self._record_unavailable(fallback_reason)
            raise CandidateResolutionUnavailable(fallback_reason) from exc
        with self._lock:
            self._fallback_total += 1
            _bounded_counter_increment(self._fallback_reasons, normalized_reason)
            self._last_fallback_reason = normalized_reason
        return replace(measurement, source="rgbd_depth_geometry_fallback")

    def _record_unavailable(self, reason: str) -> None:
        normalized = _reason_code(reason)
        with self._lock:
            self._unavailable_total += 1
            _bounded_counter_increment(self._unavailable_reasons, normalized)
            self._last_unavailable_reason = normalized

    def _build_model_inputs(self, candidate: CandidateSnapshot):
        try:
            import torch
            from torch.nn import functional as F
        except Exception as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError(f"torch unavailable: {exc}") from exc
        if not (
            len(candidate.frame_history)
            == len(candidate.bbox_history)
            == len(candidate.confidence_history)
            == len(candidate.tracker_id_history)
        ):
            raise CandidateResolutionUnavailable(
                "candidate_history_fields_are_not_aligned"
            )
        candidate_refs = candidate.frame_history
        if any(
            later.timestamp_s < earlier.timestamp_s
            for earlier, later in zip(candidate_refs, candidate_refs[1:])
        ):
            raise CandidateResolutionUnavailable(
                "candidate_frame_history_is_not_monotonic"
            )
        last_ref = candidate_refs[-1]
        if (
            abs(last_ref.timestamp_s - candidate.last_seen_timestamp_s) > 1e-9
            or candidate_refs[0].timestamp_s
            < candidate.first_seen_timestamp_s - 1e-9
        ):
            raise CandidateResolutionUnavailable(
                "candidate_frame_history_timestamp_mismatch"
            )
        refs = tuple(
            ref
            for ref in self._frame_store.refs(uav_id=candidate.uav_id)
            if candidate.first_seen_timestamp_s - 1e-9 <= ref.timestamp_s <= last_ref.timestamp_s + 1e-9
            and last_ref.timestamp_s - ref.timestamp_s <= self._max_history_age_s + 1e-9
        )
        if len(refs) < self._sequence_length:
            raise CandidateResolutionUnavailable(
                f"insufficient_temporal_history:{len(refs)}/{self._sequence_length}"
            )
        refs = refs[-self._sequence_length :]
        detected = {
            ref.frame_id: (bbox, confidence, tracker_id)
            for ref, bbox, confidence, tracker_id in zip(
                candidate.frame_history,
                candidate.bbox_history,
                candidate.confidence_history,
                candidate.tracker_id_history,
            )
        }
        samples: list[
            tuple[
                FrameRef,
                np.ndarray,
                np.ndarray,
                FrameCameraGeometry,
                FrameUavSelfMotion,
                object,
            ]
        ] = []
        for ref in refs:
            synchronized = self._frame_store.get_temporal_inputs(ref)
            if synchronized is None:
                raise CandidateResolutionUnavailable("temporal_history_frame_evicted_or_incomplete")
            rgb, depth, camera, uav_self_motion = synchronized
            samples.append(
                (
                    ref,
                    rgb,
                    depth,
                    camera,
                    uav_self_motion,
                    detected.get(ref.frame_id),
                )
            )
        reference_camera = samples[-1][3]
        reference_time = refs[-1].timestamp_s
        roi_tensors = []
        feature_rows = []
        missing_values = []
        previous_tracker: str | None = None
        for ref, rgb, depth, camera, uav_self_motion, detection in samples:
            missing = detection is None
            if detection is None:
                bbox = (0.0, 0.0, 0.0, 0.0)
                confidence = 0.0
                tracker_id = None
                roi_tensor = torch.zeros(4, self._roi_size_px, self._roi_size_px)
            else:
                bbox, raw_confidence, tracker_id = detection
                confidence = 0.0 if raw_confidence is None else float(raw_confidence)
                height, width = depth.shape
                x1 = max(0, min(width - 1, int(np.floor(bbox[0] * width))))
                y1 = max(0, min(height - 1, int(np.floor(bbox[1] * height))))
                x2 = max(x1 + 1, min(width, int(np.ceil(bbox[2] * width))))
                y2 = max(y1 + 1, min(height, int(np.ceil(bbox[3] * height))))
                rgb_values = rgb[y1:y2, x1:x2].astype(np.float32) / 255.0
                depth_values = depth[y1:y2, x1:x2].astype(np.float32)
                depth_values = np.where(
                    np.isfinite(depth_values)
                    & (depth_values >= self._min_depth_m)
                    & (depth_values <= self._max_depth_m),
                    depth_values / self._max_depth_m,
                    0.0,
                )
                if not self._use_rgb:
                    rgb_values.fill(0.0)
                if not self._use_depth:
                    depth_values.fill(0.0)
                channels = np.concatenate((rgb_values, depth_values[..., None]), axis=-1)
                roi_tensor = torch.from_numpy(channels).permute(2, 0, 1).unsqueeze(0)
                roi_tensor = F.interpolate(
                    roi_tensor,
                    size=(self._roi_size_px, self._roi_size_px),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            relative_position, relative_quaternion = _relative_camera_pose(camera, reference_camera)
            intrinsics = camera.intrinsics
            normalized_intrinsics = (
                intrinsics.fx / intrinsics.width,
                intrinsics.fy / intrinsics.height,
                intrinsics.cx / intrinsics.width,
                intrinsics.cy / intrinsics.height,
            )
            tracker_continuity = float(
                tracker_id is not None
                and previous_tracker is not None
                and tracker_id == previous_tracker
            )
            features = (
                *bbox,
                confidence,
                tracker_continuity,
                *normalized_intrinsics,
                *relative_position,
                *relative_quaternion,
                *uav_self_motion.linear_velocity_world_mps,
                *uav_self_motion.angular_velocity_body_radps,
                reference_time - ref.timestamp_s,
                float(missing),
            )
            if len(features) != 25 or not np.all(np.isfinite(features)):
                raise ValueError("temporal geometry feature construction failed")
            roi_tensors.append(roi_tensor)
            feature_rows.append(features)
            missing_values.append(missing)
            if tracker_id is not None:
                previous_tracker = tracker_id
        return (
            torch.stack(roi_tensors).unsqueeze(0).to(self._device),
            torch.tensor(feature_rows, dtype=torch.float32, device=self._device).unsqueeze(0),
            torch.tensor(missing_values, dtype=torch.bool, device=self._device).unsqueeze(0),
        )

    def _infer_measurement(self, candidate, baseline, roi, features, missing):
        import torch

        with torch.inference_mode():
            output = self._model(roi, features, missing)
        values = output.as_dict()
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise ValueError("temporal_network_non_finite_output")
        valid_probability = float(torch.sigmoid(output.measurement_valid_logit[0]).cpu())
        if valid_probability < 0.5:
            raise CandidateResolutionUnavailable(
                f"temporal_measurement_invalid_probability:{valid_probability:.4f}"
            )
        delta_uv = output.delta_uv_px[0].detach().cpu().numpy().astype(np.float64)
        residual = float(output.depth_residual_m[0].detach().cpu())
        corrected_uv = np.asarray(baseline.pixel_uv, dtype=np.float64) + delta_uv
        corrected_depth = float(baseline.corrected_depth_m + residual)
        last_ref = candidate.frame_history[-1]
        geometry = self._frame_store.get_camera_geometry(last_ref)
        if geometry is None:
            raise CandidateResolutionUnavailable("reference_camera_geometry_evicted")
        if (
            corrected_uv[0] < 0.0
            or corrected_uv[1] < 0.0
            or corrected_uv[0] >= geometry.intrinsics.width
            or corrected_uv[1] >= geometry.intrinsics.height
        ):
            raise ValueError("temporal_corrected_pixel_out_of_bounds")
        if not self._min_depth_m <= corrected_depth <= self._max_depth_m:
            raise ValueError("temporal_corrected_depth_out_of_bounds")
        optical = backproject_pixel_to_camera_optical(
            u_px=float(corrected_uv[0]),
            v_px=float(corrected_uv[1]),
            depth_m=corrected_depth,
            intrinsics=geometry.intrinsics,
        )
        camera_flu = optical_to_camera_flu(optical)
        world = camera_flu_to_world(camera_flu, geometry)
        variance = np.exp(
            np.clip(output.position_log_variance[0].detach().cpu().numpy(), -12.0, 8.0)
        )
        variance = np.clip(variance, 1e-6, 1e4)
        covariance = tuple(
            tuple(float(value) for value in row)
            for row in np.diag(variance)
        )
        return TargetMeasurement(
            timestamp_s=last_ref.timestamp_s,
            candidate_id=candidate.candidate_id,
            tracker_id=candidate.tracker_id_history[-1],
            pixel_uv=(float(corrected_uv[0]), float(corrected_uv[1])),
            raw_depth_m=baseline.raw_depth_m,
            corrected_depth_m=corrected_depth,
            position_camera_flu_m=camera_flu,
            position_world_m=world,
            covariance_world_m2=covariance,
            measurement_quality=max(0.0, min(1.0, baseline.measurement_quality * valid_probability)),
            source="temporal_ray_depth",
        )


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _quaternion_inverse(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion * np.asarray((1.0, -1.0, -1.0, -1.0))


def _rotate_inverse(quaternion_wxyz: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return rotation.T @ np.asarray(vector, dtype=np.float64)


def _relative_camera_pose(
    camera: FrameCameraGeometry,
    reference: FrameCameraGeometry,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    relative_position = _rotate_inverse(
        reference.camera_orientation_world_wxyz,
        np.asarray(camera.camera_position_world_m) - np.asarray(reference.camera_position_world_m),
    )
    relative_quaternion = _quaternion_multiply(
        _quaternion_inverse(reference.camera_orientation_world_wxyz),
        camera.camera_orientation_world_wxyz,
    )
    relative_quaternion /= np.linalg.norm(relative_quaternion)
    return (
        tuple(float(value) for value in relative_position),
        tuple(float(value) for value in relative_quaternion),
    )


def _camera_velocity(
    previous: FrameCameraGeometry | None,
    current: FrameCameraGeometry,
    previous_time: float | None,
    current_time: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if previous is None or previous_time is None or current_time <= previous_time + 1e-9:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    delta_t = current_time - previous_time
    linear = (
        np.asarray(current.camera_position_world_m, dtype=np.float64)
        - np.asarray(previous.camera_position_world_m, dtype=np.float64)
    ) / delta_t
    delta_q = _quaternion_multiply(
        _quaternion_inverse(previous.camera_orientation_world_wxyz),
        current.camera_orientation_world_wxyz,
    )
    delta_q /= np.linalg.norm(delta_q)
    if delta_q[0] < 0.0:
        delta_q = -delta_q
    angle = 2.0 * acos(float(np.clip(delta_q[0], -1.0, 1.0)))
    axis_norm = float(np.linalg.norm(delta_q[1:]))
    angular = (
        np.zeros(3, dtype=np.float64)
        if axis_norm <= 1e-9
        else delta_q[1:] / axis_norm * angle / delta_t
    )
    return (
        tuple(float(value) for value in linear),
        tuple(float(value) for value in angular),
    )


__all__ = [
    "CAMERA_CONVENTION",
    "COORDINATE_CONVENTION",
    "INPUT_SEMANTICS",
    "TEMPORAL_GEOMETRY_FIELDS",
    "TEMPORAL_OUTPUT_FIELDS",
    "TemporalRayDepthArtifactInfo",
    "TemporalRayDepthResolver",
    "TemporalRayDepthStatistics",
]
