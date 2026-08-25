from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from configs.loader import load_config
from perception.factory import build_yolo_target_perception_runtime
from perception.mode import resolve_target_perception_mode
from perception.production_boundary import (
    build_target_perception_startup_audit,
    validate_production_estimate_source,
)
from perception.runtime import (
    PerceptionBoundaryError,
    PerceptionRuntimeProfile,
    validate_observation_access,
)
from perception.runtime_provider import YoloTargetPerceptionRuntime
from tests.perception.test_yolo_runtime_provider import _base, _estimate, _sample


def test_yolo_builder_and_reset_expose_no_environment_or_target_spec() -> None:
    builder_parameters = inspect.signature(
        build_yolo_target_perception_runtime
    ).parameters
    assert "environment" not in builder_parameters
    assert "frame_provider" not in builder_parameters
    assert "get_evaluator_frame" not in builder_parameters
    assert "make_oracle_perception" not in builder_parameters
    builder_source = inspect.getsource(build_yolo_target_perception_runtime)
    assert "training." not in builder_source
    assert "datasets." not in builder_source

    reset_parameters = inspect.signature(YoloTargetPerceptionRuntime.reset).parameters
    assert "target_query" in reset_parameters
    assert "target_spec" not in reset_parameters
    assert "target_alias" not in reset_parameters


def test_constructed_yolo_runtime_holds_no_evaluator_capability() -> None:
    config = load_config("configs/yolo/runtime_yolo26.yaml")
    runtime = build_yolo_target_perception_runtime(
        config,
        resolved_mode=resolve_target_perception_mode(
            "yolo",
            runtime_profile="production",
            backend="ultralytics_service",
            acknowledge_privileged_oracle=False,
        ),
        uav_id="uav_1",
    )
    try:
        objects = (runtime, runtime._bridge, runtime.coordinator)  # noqa: SLF001
        forbidden = {"get_evaluator_frame", "make_oracle_perception"}
        for value in objects:
            attributes = vars(value)
            assert forbidden.isdisjoint(attributes)
            assert all(
                getattr(candidate, "__name__", None) not in forbidden
                for candidate in attributes.values()
            )
    finally:
        runtime.close()


def test_startup_audit_is_bounded_and_contains_no_query_values() -> None:
    digest = "89" * 32
    audit = build_target_perception_startup_audit(
        runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
        target_perception_mode="yolo",
        backend_by_uav={"uav_a": "ultralytics_service"},
        privileged=False,
        yolo_model_sha256_by_uav={"uav_a": digest},
    )
    assert audit == {
        "runtime_profile": "production",
        "target_perception_mode": "yolo",
        "backend_by_uav": {"uav_a": "ultralytics_service"},
        "privileged": False,
        "allowed_target_query_fields": [
            "target_alias",
            "detector_class_id",
            "detector_class_name",
            "hard_attributes",
            "soft_description",
        ],
        "yolo_model_sha256": digest,
        "temporal_model_sha256": None,
    }
    serialized = repr(audit).casefold()
    assert "position_world" not in serialized
    assert "motion_seed" not in serialized
    assert "prim_path" not in serialized


@pytest.mark.parametrize(
    "source",
    ("oracle", "oracle_evaluation", "ground_truth", "sim_truth", "unknown"),
)
def test_production_estimate_rejects_non_visual_sources(source: str) -> None:
    with pytest.raises(PermissionError, match="allowlist"):
        validate_production_estimate_source(source)


@pytest.mark.parametrize(
    "source",
    (
        "ultralytics_service",
        "yolo26_botsort",
        "rgbd_depth_geometry",
        "temporal_ray_depth",
        "kalman_prediction",
    ),
)
def test_production_estimate_accepts_only_declared_visual_sources(source: str) -> None:
    assert validate_production_estimate_source(source) == source


def test_production_audit_rejects_oracle_ack_and_disabled_backend() -> None:
    digest = "ab" * 32
    with pytest.raises(PermissionError, match="privileged=false"):
        build_target_perception_startup_audit(
            runtime_profile="production",
            target_perception_mode="yolo",
            backend_by_uav={"uav_a": "ultralytics_service"},
            privileged=True,
            yolo_model_sha256_by_uav={"uav_a": digest},
        )
    with pytest.raises(PermissionError, match="ultralytics_service"):
        build_target_perception_startup_audit(
            runtime_profile="production",
            target_perception_mode="yolo",
            backend_by_uav={"uav_a": "disabled"},
            privileged=False,
            yolo_model_sha256_by_uav={"uav_a": digest},
        )


@pytest.mark.parametrize(
    "source",
    ("oracle_evaluation", "ground_truth", "sim_truth", "simulator_truth"),
)
def test_production_observation_rejects_privileged_estimate_immediately(
    source: str,
) -> None:
    sample = _sample()
    observation = replace(
        _base(sample),
        target_estimate=_estimate(sample.timestamp_s, source=source),
    )
    with pytest.raises(PerceptionBoundaryError, match="production Agent Runtime"):
        validate_observation_access(observation)
