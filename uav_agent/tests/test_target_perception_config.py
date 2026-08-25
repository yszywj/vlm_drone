from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from configs.loader import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"
YOLO26_CONFIG = PROJECT_ROOT / "configs/yolo/runtime_yolo26.yaml"
YOLOE_CONFIG = PROJECT_ROOT / "configs/yolo/runtime_yoloe.yaml"
SINGLE_YOLO_LAUNCHER = PROJECT_ROOT / "scripts/run_single_uav_yolo_e2e.sh"


def _read_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_yaml(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _set_nested(payload: dict[str, object], dotted_path: str, value: object) -> None:
    keys = dotted_path.split(".")
    current: dict[str, object] = payload
    for key in keys[:-1]:
        child = current[key]
        assert isinstance(child, dict)
        current = child
    current[keys[-1]] = value


def test_target_perception_block_is_optional_and_defaults_to_disabled(
    tmp_path: Path,
) -> None:
    committed = load_config(DEFAULT_CONFIG)
    assert committed.target_perception.backend == "disabled"

    legacy_payload = _read_yaml(DEFAULT_CONFIG)
    del legacy_payload["target_perception"]
    legacy = load_config(_write_yaml(tmp_path, legacy_payload))
    assert legacy.target_perception.backend == "disabled"
    assert legacy.target_perception.yolo_service.max_inflight_per_uav == 1
    assert legacy.target_perception.detector.model_family == "yolo"


def test_committed_yolo_runtime_configs_are_complete_and_consistent() -> None:
    yolo = load_config(YOLO26_CONFIG).target_perception
    assert yolo.backend == "ultralytics_service"
    assert (yolo.detector.model_family, yolo.detector.proposal_mode) == (
        "yolo",
        "closed_set",
    )
    assert yolo.geometry.mode == "temporal_ray_depth"
    temporal = yolo.geometry.temporal_ray_depth
    assert temporal.checkpoint_path == (
        "/home/amax/ry/vlm_drones/outputs/trained_models/target_state/"
        "temporal_ray_depth_yolo_deployment_v2/best.pt"
    )
    assert temporal.manifest_path == (
        "/home/amax/ry/vlm_drones/outputs/trained_models/target_state/"
        "temporal_ray_depth_yolo_deployment_v2/model_manifest.json"
    )
    assert temporal.expected_sha256 == (
        "2a8acd63347a568a4e5588cfc335979709958f7859617405841062af679e023c"
    )
    assert temporal.history_size == 6
    assert temporal.deterministic_fallback is True
    assert yolo.tracker.type == "botsort"
    assert yolo.yolo_service.url == "http://127.0.0.1:8011"
    assert yolo.yolo_service.request_timeout_s == 5.0
    assert yolo.yolo_service.max_result_age_s == 2.0
    assert yolo.confirmation.mode == "class_track_attribute_or_qwen"
    assert yolo.attributes.enabled is True
    assert yolo.attributes.color.supported_values == ("red", "blue")
    assert yolo.detector.expected_model_family == "yolo"
    assert dict(yolo.detector.expected_model_names) == {0: "cube"}
    assert yolo.detector.expected_model_sha256 == (
        "895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07"
    )

    yoloe = load_config(YOLOE_CONFIG).target_perception
    assert yoloe.backend == "ultralytics_service"
    assert (yoloe.detector.model_family, yoloe.detector.proposal_mode) == (
        "yoloe",
        "open_vocabulary",
    )
    assert yoloe.yolo_service.request_timeout_s == 5.0
    assert yoloe.yolo_service.max_result_age_s == 2.0


def test_temporal_ray_depth_config_has_strict_artifact_contract(
    tmp_path: Path,
) -> None:
    payload = deepcopy(_read_yaml(YOLO26_CONFIG))
    _set_nested(payload, "target_perception.geometry.mode", "temporal_ray_depth")
    geometry = payload["target_perception"]["geometry"]  # type: ignore[index]
    assert isinstance(geometry, dict)
    geometry["temporal_ray_depth"] = {
        "checkpoint_path": "/tmp/temporal/best.pt",
        "expected_sha256": "a" * 64,
        "manifest_path": "/tmp/temporal/model_manifest.json",
        "history_size": 6,
        "max_history_age_s": 2.0,
        "roi_size_px": 128,
        "use_rgb": True,
        "use_depth": True,
        "deterministic_fallback": True,
        "device": "cpu",
    }

    config = load_config(_write_yaml(tmp_path, payload))

    temporal = config.target_perception.geometry.temporal_ray_depth
    assert config.target_perception.geometry.mode == "temporal_ray_depth"
    assert temporal.expected_sha256 == "a" * 64
    assert temporal.history_size == 6
    assert temporal.deterministic_fallback is True


def test_temporal_ray_depth_mode_without_complete_block_is_rejected(
    tmp_path: Path,
) -> None:
    payload = deepcopy(_read_yaml(YOLO26_CONFIG))
    _set_nested(payload, "target_perception.geometry.mode", "temporal_ray_depth")
    geometry = payload["target_perception"]["geometry"]  # type: ignore[index]
    assert isinstance(geometry, dict)
    geometry.pop("temporal_ray_depth", None)
    with pytest.raises(ConfigError, match="complete temporal_ray_depth block"):
        load_config(_write_yaml(tmp_path, payload))


def test_temporal_ray_depth_manifest_and_device_have_safe_defaults(
    tmp_path: Path,
) -> None:
    payload = deepcopy(_read_yaml(YOLO26_CONFIG))
    _set_nested(payload, "target_perception.geometry.mode", "temporal_ray_depth")
    geometry = payload["target_perception"]["geometry"]  # type: ignore[index]
    assert isinstance(geometry, dict)
    geometry["temporal_ray_depth"] = {
        "checkpoint_path": "/tmp/temporal/best.pt",
        "expected_sha256": "b" * 64,
        "history_size": 6,
        "max_history_age_s": 2.0,
        "roi_size_px": 128,
        "use_rgb": True,
        "use_depth": True,
        "deterministic_fallback": True,
    }

    config = load_config(_write_yaml(tmp_path, payload))

    temporal = config.target_perception.geometry.temporal_ray_depth
    assert temporal.manifest_path is None
    assert temporal.device == "cpu"


def test_single_uav_yolo_launcher_uses_formal_fail_closed_runtime() -> None:
    assert SINGLE_YOLO_LAUNCHER.stat().st_mode & 0o111
    launcher = SINGLE_YOLO_LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "set -euo pipefail",
        "weights/best.pt",
        "scripts/serve_yolo.py",
        "http://127.0.0.1:8011",
        "scripts/check_fleet_yolo_services.py",
        "scripts/run_fleet_mission.py",
        "--target-perception-mode yolo",
        "--perception-runtime-profile production",
        "trap cleanup EXIT",
    ):
        assert required in launcher
    assert "oracle_evaluation" not in launcher
    assert "--target-perception-mode disabled" not in launcher


def test_target_perception_config_is_immutable() -> None:
    config = load_config(DEFAULT_CONFIG).target_perception
    with pytest.raises(FrozenInstanceError):
        config.backend = "oracle_evaluation"  # type: ignore[misc]


def test_unknown_target_perception_key_is_rejected(tmp_path: Path) -> None:
    payload = _read_yaml(DEFAULT_CONFIG)
    target = payload["target_perception"]
    assert isinstance(target, dict)
    target["oracle_fallback"] = True
    with pytest.raises(ConfigError, match="unknown keys: oracle_fallback"):
        load_config(_write_yaml(tmp_path, payload))


def test_partial_nested_target_perception_block_is_rejected(tmp_path: Path) -> None:
    payload = _read_yaml(DEFAULT_CONFIG)
    target = payload["target_perception"]
    assert isinstance(target, dict)
    target["detector"] = {"model_family": "yolo"}
    with pytest.raises(ConfigError, match="detector is missing required keys"):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("dotted_path", "value", "message"),
    (
        (
            "target_perception.yolo_service.url",
            "http://0.0.0.0:8011",
            "must use loopback HTTP",
        ),
        (
            "target_perception.yolo_service.url",
            "http://127.0.0.1:8011@evil.example:80",
            "must not contain user information",
        ),
        (
            "target_perception.yolo_service.max_inflight_per_uav",
            2,
            "must be exactly 1",
        ),
        (
            "target_perception.yolo_service.jpeg_quality",
            96,
            "must not exceed 95",
        ),
        (
            "target_perception.detector.confidence_threshold",
            0.0,
            "must be in",
        ),
        (
            "target_perception.geometry.depth_patch_radius_px",
            0,
            "positive integer",
        ),
        (
            "target_perception.state_estimator.max_prediction_age_s",
            0.0,
            "greater than 0",
        ),
        (
            "target_perception.confirmation.require_qwen_for_attributes",
            "yes",
            "must be true or false",
        ),
    ),
)
def test_target_perception_numeric_and_type_bounds_are_strict(
    tmp_path: Path,
    dotted_path: str,
    value: object,
    message: str,
) -> None:
    payload = deepcopy(_read_yaml(YOLO26_CONFIG))
    _set_nested(payload, dotted_path, value)
    with pytest.raises(ConfigError, match=message):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("family", "proposal"),
    (("yolo", "open_vocabulary"), ("yoloe", "closed_set")),
)
def test_model_family_and_proposal_mode_must_match(
    tmp_path: Path,
    family: str,
    proposal: str,
) -> None:
    payload = deepcopy(_read_yaml(YOLO26_CONFIG))
    _set_nested(payload, "target_perception.detector.model_family", family)
    _set_nested(payload, "target_perception.detector.proposal_mode", proposal)
    with pytest.raises(ConfigError, match="requires proposal_mode"):
        load_config(_write_yaml(tmp_path, payload))


def test_geometry_depth_range_must_be_ordered(tmp_path: Path) -> None:
    payload = deepcopy(_read_yaml(YOLO26_CONFIG))
    _set_nested(payload, "target_perception.geometry.min_depth_m", 2.0)
    _set_nested(payload, "target_perception.geometry.max_depth_m", 2.0)
    with pytest.raises(ConfigError, match="max_depth_m must exceed min_depth_m"):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("expected_model_sha256", "abc", "64-character"),
        ("expected_model_names", {"0": "cube"}, "non-negative integers"),
        ("expected_model_names", {0: ""}, "non-empty string"),
        ("expected_model_names", {}, "requires expected_model_names"),
        ("expected_model_family", "other", "must be yolo or yoloe"),
        ("expected_model_family", "yoloe", "must match model_family"),
        ("expected_model_family", None, "requires expected_model_family"),
    ),
)
def test_detector_model_identity_contract_is_strict(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = deepcopy(_read_yaml(YOLO26_CONFIG))
    _set_nested(payload, f"target_perception.detector.{field}", value)
    with pytest.raises(ConfigError, match=message):
        load_config(_write_yaml(tmp_path, payload))
