from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
import torch

from env.camera_types import CameraIntrinsics
from configs.loader import load_config
from configs.schema import TemporalRayDepthConfig
from perception.factory import preflight_temporal_ray_depth
from perception.candidate_bank import CandidateBank, CandidateSnapshot
from perception.temporal_ray_depth import (
    CAMERA_CONVENTION,
    COORDINATE_CONVENTION,
    INPUT_SEMANTICS,
    TEMPORAL_GEOMETRY_FIELDS,
    TEMPORAL_OUTPUT_FIELDS,
    TemporalRayDepthResolver,
)
from runtime.frame_store import FrameStore
from training.target_state.model import TemporalRayDepthNet


def _artifact(root: Path, *, history_size: int = 4) -> tuple[Path, str]:
    model = TemporalRayDepthNet(
        geometry_input_dim=25,
        roi_feature_dim=16,
        geometry_feature_dim=12,
        hidden_dim=20,
        gru_layers=1,
        roi_channels=4,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.validity_head.bias.fill_(10.0)
    checkpoint = root / "best.pt"
    torch.save(
        {
            "model_type": "temporal_ray_depth_residual",
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "model_type": "temporal_ray_depth_residual",
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "dataset_sha256": "0" * 64,
        "training_commit_sha": "nogit",
        "input_fields": {
            "roi_rgbd": ["red", "green", "blue", "normalized_depth"],
            "geometry_25d": list(TEMPORAL_GEOMETRY_FIELDS),
            "missing_mask": True,
        },
        "input_semantics": dict(INPUT_SEMANTICS),
        "output_fields": list(TEMPORAL_OUTPUT_FIELDS),
        "model_config": {
            "geometry_input_dim": 25,
            "roi_channels": 4,
            "roi_size_px": 32,
            "roi_feature_dim": 16,
            "geometry_feature_dim": 12,
            "hidden_dim": 20,
            "gru_layers": 1,
            "time_steps": history_size + 1,
        },
        "history_size": history_size,
        "max_history_age_s": 2.0,
        "camera_convention": CAMERA_CONVENTION,
        "coordinate_convention": COORDINATE_CONVENTION,
        "validation_metrics": {
            "model": {
                "evaluated_no_target_count": 2,
                "no_target_false_positive_rate": 0.0,
            },
            "deterministic_rgbd_baseline": {
                "evaluated_no_target_count": 2,
                "no_target_false_positive_rate": 0.0,
            },
        },
        "test_metrics": {
            "model": {
                "evaluated_no_target_count": 2,
                "no_target_false_positive_rate": 0.0,
            },
            "deterministic_rgbd_baseline": {
                "evaluated_no_target_count": 2,
                "no_target_false_positive_rate": 0.0,
            },
        },
        "preprocessing": {
            "rgb_scale": 255.0,
            "depth_scale_m": 200.0,
            "minimum_depth_m": 0.2,
            "maximum_depth_m": 200.0,
            "roi_interpolation": "bilinear_align_corners_false",
            "baseline_depth_sampling": "foreground_cluster_median",
            "foreground_inset_ratio": 0.1,
            "foreground_bottom_exclusion_ratio": 0.15,
            "foreground_min_valid_samples": 3,
            "foreground_seed_patch_radius_px": 4,
        },
        "training_stage": "yolo_deployment",
        "promotion": {
            "passed": True,
            "requires_yolo_deployment_stage": True,
            "stage_satisfied": True,
            "requires_stage_a_initialization": True,
            "stage_a_initialization_satisfied": True,
            "requires_verified_dataset_manifest": True,
            "dataset_manifest_satisfied": True,
            "validation": {"passed": True, "reasons": []},
            "test": {"passed": True, "reasons": []},
        },
    }
    (root / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, digest


def _frames(
    count: int,
    *,
    store: FrameStore | None = None,
    uav_id: str = "uav_1",
) -> tuple[FrameStore, CandidateSnapshot]:
    if store is None:
        store = FrameStore(max_frames=16, max_bytes=2_000_000, max_age_s=10.0)
    bank = CandidateBank(uav_id=uav_id, max_history_per_candidate=8)
    intrinsics = CameraIntrinsics(fx=80.0, fy=80.0, cx=15.5, cy=15.5, width=32, height=32)
    candidate = None
    for index in range(count):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[8:24, 8:24, 0] = 255
        depth = np.full((32, 32), 8.0, dtype=np.float32)
        depth[8:24, 8:24] = 4.0
        ref = store.add_frame(
            uav_id=uav_id,
            frame_id=f"frame_{index}",
            timestamp_s=index * 0.2,
            rgb=rgb,
            depth_to_image_plane_m=depth,
            intrinsics=intrinsics,
            camera_position_world_m=(index * 0.01, 0.0, 0.0),
            camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            uav_linear_velocity_world_mps=(1.0 + index, 2.0, 3.0),
            uav_angular_velocity_body_radps=(0.1, 0.2, 0.3 + index),
        )
        candidate = bank.propose(
            candidate_id="candidate_1",
            timestamp_s=ref.timestamp_s,
            bbox_xyxy_normalized=(0.25, 0.25, 0.75, 0.75),
            frame_ref=ref,
            source="ultralytics_service",
            confidence=0.9,
            tracker_id="track_1",
        )
    assert candidate is not None
    return store, candidate


def test_temporal_resolver_outputs_corrected_geometry_and_covariance(tmp_path: Path) -> None:
    checkpoint, digest = _artifact(tmp_path)
    store = FrameStore(max_frames=16, max_bytes=2_000_000, max_age_s=10.0)
    resolver = TemporalRayDepthResolver(
        store,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        history_size=4,
        max_history_age_s=2.0,
        roi_size_px=32,
    )
    resolver.reset(uav_id="uav_1", assignment_id="assignment_1")
    _, candidate = _frames(5, store=store)

    measurement = resolver.resolve(candidate, timestamp_s=0.8)

    assert measurement.source == "temporal_ray_depth"
    assert measurement.corrected_depth_m == pytest.approx(4.0)
    assert np.all(np.isfinite(measurement.position_world_m))
    assert np.linalg.eigvalsh(np.asarray(measurement.covariance_world_m2)).min() > 0.0
    assert resolver.statistics.successes == 1
    assert resolver.statistics.fallback_total == 0


def test_temporal_resolver_uses_only_configured_rgbd_fallback(tmp_path: Path) -> None:
    checkpoint, digest = _artifact(tmp_path)
    store = FrameStore(max_frames=16, max_bytes=2_000_000, max_age_s=10.0)
    resolver = TemporalRayDepthResolver(
        store,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        history_size=4,
        max_history_age_s=2.0,
        roi_size_px=32,
        deterministic_fallback=True,
    )
    resolver.reset(uav_id="uav_1", assignment_id="assignment_fallback")
    _, candidate = _frames(1, store=store)

    measurement = resolver.resolve(candidate, timestamp_s=0.0)

    assert measurement.source == "rgbd_depth_geometry_fallback"
    assert resolver.statistics.fallback_total == 1
    assert "oracle" not in json.dumps(resolver.statistics.to_dict()).casefold()


def test_temporal_preflight_fails_on_sha_or_manifest_contract(tmp_path: Path) -> None:
    checkpoint, digest = _artifact(tmp_path)
    store, _ = _frames(1)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        TemporalRayDepthResolver(
            store,
            checkpoint_path=checkpoint,
            expected_sha256="f" * 64,
            history_size=4,
            max_history_age_s=2.0,
            roi_size_px=32,
        )

    manifest_path = tmp_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_fields"] = ["world_position_black_box"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="output_fields"):
        TemporalRayDepthResolver(
            store,
            checkpoint_path=checkpoint,
            expected_sha256=digest,
            history_size=4,
            max_history_age_s=2.0,
            roi_size_px=32,
        )


def test_temporal_history_is_uav_bound_and_reset_explicit(tmp_path: Path) -> None:
    checkpoint, digest = _artifact(tmp_path)
    store = FrameStore(max_frames=16, max_bytes=2_000_000, max_age_s=10.0)
    resolver = TemporalRayDepthResolver(
        store,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        history_size=4,
        max_history_age_s=2.0,
        roi_size_px=32,
    )
    resolver.reset(uav_id="uav_2", assignment_id="assignment_2")
    _, candidate = _frames(5, store=store, uav_id="uav_1")
    with pytest.raises(Exception, match="another UAV"):
        resolver.resolve(candidate, timestamp_s=0.8)


def test_temporal_resolver_refuses_use_without_assignment_reset(tmp_path: Path) -> None:
    checkpoint, digest = _artifact(tmp_path)
    store, candidate = _frames(1)
    resolver = TemporalRayDepthResolver(
        store,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        history_size=4,
        max_history_age_s=2.0,
        roi_size_px=32,
    )

    with pytest.raises(Exception, match="reset for an Assignment"):
        resolver.resolve(candidate, timestamp_s=0.0)
    with pytest.raises(ValueError, match="requires assignment_id"):
        resolver.reset(uav_id="uav_1")


def test_factory_preflight_dry_runs_artifact_without_isaac(tmp_path: Path) -> None:
    checkpoint, digest = _artifact(tmp_path)
    config = load_config(Path(__file__).resolve().parents[2] / "configs/yolo/runtime_yolo26.yaml")
    temporal = TemporalRayDepthConfig(
        checkpoint_path=str(checkpoint),
        expected_sha256=digest,
        manifest_path=str(tmp_path / "model_manifest.json"),
        history_size=4,
        max_history_age_s=2.0,
        roi_size_px=32,
        device="cpu",
    )
    geometry = replace(
        config.target_perception.geometry,
        mode="temporal_ray_depth",
        temporal_ray_depth=temporal,
    )
    target_perception = replace(config.target_perception, geometry=geometry)
    config = replace(config, target_perception=target_perception)

    metadata = preflight_temporal_ray_depth(config)

    assert metadata is not None
    assert metadata["checkpoint_sha256"] == digest
    assert metadata["dry_run"] == "passed"
    assert metadata["training_stage"] == "yolo_deployment"
    assert metadata["promotion"] == "passed"


def test_temporal_features_use_synchronized_uav_self_motion_not_camera_delta(
    tmp_path: Path,
) -> None:
    checkpoint, digest = _artifact(tmp_path)
    store = FrameStore(max_frames=16, max_bytes=2_000_000, max_age_s=10.0)
    resolver = TemporalRayDepthResolver(
        store,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        history_size=4,
        max_history_age_s=2.0,
        roi_size_px=32,
    )
    resolver.reset(uav_id="uav_1", assignment_id="assignment_motion")
    _, candidate = _frames(5, store=store)

    _roi, geometry, _missing = resolver._build_model_inputs(candidate)  # noqa: SLF001
    last = geometry.detach().cpu().numpy()[0, -1]

    np.testing.assert_allclose(last[17:20], (5.0, 2.0, 3.0))
    np.testing.assert_allclose(last[20:23], (0.1, 0.2, 4.3))
    # The camera itself moved only 0.01 m per frame. A finite-difference
    # camera velocity would therefore be about 0.05 m/s, not 5 m/s.
    assert last[17] != pytest.approx(0.05)


def test_assignment_reset_clears_the_only_rgbd_history_owner(tmp_path: Path) -> None:
    checkpoint, digest = _artifact(tmp_path)
    store, _candidate = _frames(5)
    resolver = TemporalRayDepthResolver(
        store,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        history_size=4,
        max_history_age_s=2.0,
        roi_size_px=32,
    )
    assert len(store) == 5

    resolver.reset(uav_id="uav_1", assignment_id="assignment_new")

    assert len(store) == 0
    assert resolver.statistics.reset_total == 1


def test_expired_candidate_is_rejected_without_rgbd_fallback(tmp_path: Path) -> None:
    checkpoint, digest = _artifact(tmp_path)
    store = FrameStore(max_frames=16, max_bytes=2_000_000, max_age_s=10.0)
    resolver = TemporalRayDepthResolver(
        store,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        history_size=4,
        max_history_age_s=2.0,
        roi_size_px=32,
        deterministic_fallback=True,
    )
    resolver.reset(uav_id="uav_1", assignment_id="assignment_stale")
    _, candidate = _frames(5, store=store)

    with pytest.raises(Exception, match="candidate_history_expired"):
        resolver.resolve(candidate, timestamp_s=2.81)

    statistics = resolver.statistics
    assert statistics.fallback_total == 0
    assert statistics.unavailable_reasons["candidate_history_expired"] == 1
    assert statistics.last_unavailable_reason == "candidate_history_expired"


def test_fallback_reason_is_bounded_and_source_is_explicit(tmp_path: Path) -> None:
    checkpoint, digest = _artifact(tmp_path)
    store = FrameStore(max_frames=16, max_bytes=2_000_000, max_age_s=10.0)
    resolver = TemporalRayDepthResolver(
        store,
        checkpoint_path=checkpoint,
        expected_sha256=digest,
        history_size=4,
        max_history_age_s=2.0,
        roi_size_px=32,
        deterministic_fallback=True,
    )
    resolver.reset(uav_id="uav_1", assignment_id="assignment_fallback")
    _, candidate = _frames(1, store=store)

    measurement = resolver.resolve(candidate, timestamp_s=0.0)
    statistics = resolver.statistics

    assert measurement.source == "rgbd_depth_geometry_fallback"
    assert statistics.fallback_total == 1
    assert statistics.last_fallback_reason == "insufficient_temporal_history"
    assert statistics.fallback_reasons == {"insufficient_temporal_history": 1}
    assert len(statistics.fallback_reasons) <= 16


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["promotion"].__setitem__("passed", False), "promotion gate"),
        (lambda value: value.__setitem__("training_stage", "oracle_clean"), "training_stage"),
        (
            lambda value: value["preprocessing"].__setitem__("depth_scale_m", 100.0),
            "preprocessing",
        ),
        (
            lambda value: value["input_semantics"].__setitem__(
                "uav_linear_velocity_frame", "camera"
            ),
            "input_semantics",
        ),
        (
            lambda value: value["validation_metrics"]["model"].__setitem__(
                "evaluated_no_target_count", 0
            ),
            "no verified no-target samples",
        ),
        (
            lambda value: value["test_metrics"]["model"].__setitem__(
                "no_target_false_positive_rate", 0.5
            ),
            "no-target rate exceeds baseline",
        ),
    ),
)
def test_temporal_production_preflight_rejects_unpromoted_or_mismatched_manifest(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    checkpoint, digest = _artifact(tmp_path)
    manifest_path = tmp_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        TemporalRayDepthResolver(
            FrameStore(),
            checkpoint_path=checkpoint,
            expected_sha256=digest,
            history_size=4,
            max_history_age_s=2.0,
            roi_size_px=32,
        )


def test_temporal_runtime_module_has_no_truth_or_evaluator_input_surface() -> None:
    source = (Path(__file__).resolve().parents[2] / "perception/temporal_ray_depth.py").read_text(
        encoding="utf-8"
    ).casefold()
    for forbidden in (
        "training_label",
        "oracle_target",
        "get_evaluator_frame",
        "target_truth",
        "motion_seed",
        "prim_path",
    ):
        assert forbidden not in source
