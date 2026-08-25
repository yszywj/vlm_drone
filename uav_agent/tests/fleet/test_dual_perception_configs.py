from __future__ import annotations

import builtins
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from configs.loader import ConfigError, load_config
from fleet.request_builder import build_target_catalog
from perception.mode import TargetPerceptionMode
from scripts import run_fleet_mission


ROOT = Path(__file__).resolve().parents[2]
ORACLE_CONFIG = ROOT / "configs/multi_uav_oracle.yaml"
YOLO_CONFIG = ROOT / "configs/multi_uav_cube_yolo.yaml"
LEGACY_CONFIG = ROOT / "configs/multi_uav_demo.yaml"


def _yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_dual_configs_only_change_perception_and_experiment_name() -> None:
    oracle = _yaml(ORACLE_CONFIG)
    yolo = _yaml(YOLO_CONFIG)
    assert set(oracle) == set(yolo)
    for key in oracle:
        if key not in {"target_perception", "experiment"}:
            assert oracle[key] == yolo[key], key

    oracle_experiment = deepcopy(oracle["experiment"])
    yolo_experiment = deepcopy(yolo["experiment"])
    assert isinstance(oracle_experiment, dict)
    assert isinstance(yolo_experiment, dict)
    assert oracle_experiment.pop("name") != yolo_experiment.pop("name")
    assert oracle_experiment == yolo_experiment

    for key in (
        "scene",
        "uavs",
        "camera_profiles",
        "targets",
        "fleet",
        "search",
        "planner",
    ):
        assert oracle[key] == yolo[key]


def test_oracle_config_has_no_yolo_or_attribute_runtime_settings() -> None:
    raw = _yaml(ORACLE_CONFIG)
    assert raw["target_perception"] == {"backend": "oracle_evaluation"}
    serialized = yaml.safe_dump(raw["target_perception"])
    for forbidden in ("url", "model_path", "tracker", "attributes", "threshold"):
        assert forbidden not in serialized

    config = load_config(ORACLE_CONFIG)
    assert config.target_perception.backend == "oracle_evaluation"
    assert dict(config.target_perception.yolo_service.per_uav_urls) == {}


def test_cube_yolo_config_has_complete_isolated_per_uav_settings() -> None:
    config = load_config(YOLO_CONFIG)
    target = config.target_perception
    assert target.backend == "ultralytics_service"
    assert dict(target.yolo_service.per_uav_urls) == {
        "uav_a": "http://127.0.0.1:8011",
        "uav_b": "http://127.0.0.1:8012",
    }
    assert target.detector.model_family == "yolo"
    assert target.detector.proposal_mode == "closed_set"
    assert target.tracker.type == "botsort"
    assert target.confirmation.mode == "class_track_attribute_or_qwen"
    assert target.attributes.mode == "deterministic_then_qwen"
    assert target.attributes.color.supported_values == ("red", "blue")
    assert set(target.attributes.color.hue_ranges_deg) == {"red", "blue"}


@pytest.mark.parametrize("config_path", (ORACLE_CONFIG, YOLO_CONFIG))
def test_both_modes_use_cube_class_and_color_only_for_identity(
    config_path: Path,
) -> None:
    catalog = build_target_catalog(load_config(config_path))
    assert catalog["target_i"].to_dict() == {
        "original_description": "red cube",
        "category": "cube",
        "hard_attributes": ["color=red"],
        "soft_attributes": [],
        "negative_constraints": [],
        "relation_constraints": [],
        "query_ladder": [],
        "inspection_questions": [],
        "immutable_identity_summary": "red cube",
        "mutable_appearance_notes": [],
    }
    assert catalog["target_j"].category == "cube"
    assert catalog["target_j"].hard_attributes == ("color=blue",)
    serialized = str(
        {
            key: value.to_dict()
            for key, value in sorted(catalog.items())
        }
    )
    assert "shape=CUBE" not in serialized
    assert "target_i" not in catalog["target_i"].original_description
    assert "target_j" not in catalog["target_j"].original_description


def test_multi_uav_yolo_urls_must_cover_uavs_and_be_distinct(
    tmp_path: Path,
) -> None:
    payload = _yaml(YOLO_CONFIG)
    target = payload["target_perception"]
    assert isinstance(target, dict)
    service = target["yolo_service"]
    assert isinstance(service, dict)

    missing = deepcopy(payload)
    missing_target = missing["target_perception"]
    assert isinstance(missing_target, dict)
    missing_service = missing_target["yolo_service"]
    assert isinstance(missing_service, dict)
    missing_urls = missing_service["per_uav_urls"]
    assert isinstance(missing_urls, dict)
    del missing_urls["uav_b"]
    missing_path = tmp_path / "missing.yaml"
    missing_path.write_text(yaml.safe_dump(missing, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="every active UAV"):
        load_config(missing_path)

    duplicate = deepcopy(payload)
    duplicate_target = duplicate["target_perception"]
    assert isinstance(duplicate_target, dict)
    duplicate_service = duplicate_target["yolo_service"]
    assert isinstance(duplicate_service, dict)
    duplicate_urls = duplicate_service["per_uav_urls"]
    assert isinstance(duplicate_urls, dict)
    duplicate_urls["uav_b"] = duplicate_urls["uav_a"]
    duplicate_path = tmp_path / "duplicate.yaml"
    duplicate_path.write_text(
        yaml.safe_dump(duplicate, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="distinct"):
        load_config(duplicate_path)


@pytest.mark.parametrize(
    ("config_path", "mode", "extra", "message"),
    (
        (ORACLE_CONFIG, "yolo", (), "requires YAML"),
        (YOLO_CONFIG, "oracle", ("--acknowledge-privileged-oracle",), "requires YAML"),
        (ORACLE_CONFIG, "oracle", (), "acknowledge"),
        (YOLO_CONFIG, "yolo", ("--acknowledge-privileged-oracle",), "forbidden"),
        (
            ORACLE_CONFIG,
            "oracle",
            (
                "--perception-runtime-profile",
                "production",
                "--acknowledge-privileged-oracle",
            ),
            "runtime profile",
        ),
    ),
)
def test_explicit_mode_mismatch_fails_before_isaac(
    config_path: Path,
    mode: str,
    extra: tuple[str, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "isaacsim" or name.startswith("isaacsim."):
            attempted.append(name)
            raise AssertionError("mode mismatch crossed the Isaac boundary")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(config_path),
            "--target-perception-mode",
            mode,
            *extra,
        ]
    )
    with pytest.raises(
        run_fleet_mission.FleetLaunchConfigurationError,
        match=message,
    ):
        run_fleet_mission.prepare_fleet_mission(args)
    assert attempted == []


@pytest.mark.parametrize(
    ("config_path", "mode", "extra", "profile"),
    (
        (
            ORACLE_CONFIG,
            "oracle",
            ("--acknowledge-privileged-oracle",),
            "oracle_evaluation",
        ),
        (YOLO_CONFIG, "yolo", (), "production"),
    ),
)
def test_explicit_modes_complete_pure_preparation(
    config_path: Path,
    mode: str,
    extra: tuple[str, ...],
    profile: str,
) -> None:
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(config_path),
            "--target-perception-mode",
            mode,
            *extra,
        ]
    )
    prepared = run_fleet_mission.prepare_fleet_mission(args)
    assert args.perception_runtime_profile == profile
    assert prepared.resolved_target_perception_mode is not None
    assert prepared.resolved_target_perception_mode.mode is TargetPerceptionMode(mode)
    assert sorted(prepared.compilations) == ["uav_a", "uav_b"]


def test_oracle_and_yolo_prepare_identical_tasks_plans_and_local_goals() -> None:
    oracle = run_fleet_mission.prepare_fleet_mission(
        run_fleet_mission.parse_args(
            [
                "--config",
                str(ORACLE_CONFIG),
                "--target-perception-mode",
                "oracle",
                "--acknowledge-privileged-oracle",
            ]
        )
    )
    yolo = run_fleet_mission.prepare_fleet_mission(
        run_fleet_mission.parse_args(
            [
                "--config",
                str(YOLO_CONFIG),
                "--target-perception-mode",
                "yolo",
            ]
        )
    )

    assert oracle.task_spec == yolo.task_spec
    oracle_plan = oracle.plan.to_dict()
    yolo_plan = yolo.plan.to_dict()
    oracle_plan.pop("fleet_mission_id")
    yolo_plan.pop("fleet_mission_id")
    assert oracle_plan == yolo_plan

    def local_goals(prepared) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for uav_id, compilation in prepared.compilations.items():
            assert compilation.compiled_mission is not None
            plan = compilation.compiled_mission.task_plan.to_dict()
            plan.pop("mission_id")
            result[uav_id] = plan
        return result

    assert local_goals(oracle) == local_goals(yolo)


def test_disabled_target_plan_fails_before_isaac_and_recommends_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "isaacsim" or name.startswith("isaacsim."):
            attempted.append(name)
            raise AssertionError("disabled target task crossed the Isaac boundary")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = run_fleet_mission.parse_args(["--config", str(LEGACY_CONFIG)])
    with pytest.raises(Exception) as captured:
        run_fleet_mission.prepare_fleet_mission(args)
    message = str(captured.value)
    assert "disabled cannot execute target Skills" in message
    assert "--target-perception-mode oracle" in message
    assert "--target-perception-mode yolo" in message
    assert attempted == []


def test_legacy_yolo_gate_still_requires_explicit_gate_acknowledgement() -> None:
    common = [
        "--config",
        str(YOLO_CONFIG),
        "--enable-qwen-vision",
        "--vision-review-mode",
        "gate",
    ]
    with pytest.raises(
        run_fleet_mission.FleetLaunchConfigurationError,
        match="acknowledge-vision-gate",
    ):
        run_fleet_mission.prepare_fleet_mission(
            run_fleet_mission.parse_args(common)
        )

    prepared = run_fleet_mission.prepare_fleet_mission(
        run_fleet_mission.parse_args(
            [*common, "--acknowledge-vision-gate"]
        )
    )
    assert prepared.resolved_target_perception_mode is None
    assert prepared.vision_review_mode == "gate"


def test_failed_oracle_preparation_still_labels_every_result_surface(
    tmp_path: Path,
) -> None:
    exit_code = run_fleet_mission.main(
        [
            "--config",
            str(ORACLE_CONFIG),
            "--target-perception-mode",
            "oracle",
            "--output-root",
            str(tmp_path),
            "--no-summary-figures",
        ]
    )
    assert exit_code == 2
    run_dirs = tuple((tmp_path / "runs/fleet_mission").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    manifest = yaml.safe_load(
        (run_dir / "manifest.yaml").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    required = {
        "target_perception_mode": "oracle",
        "runtime_profile": "oracle_evaluation",
        "backend_by_uav": {
            "uav_a": "oracle_evaluation",
            "uav_b": "oracle_evaluation",
        },
        "privileged_perception": True,
        "oracle_acknowledged": False,
        "qwen_vision_mode": "disabled",
    }
    for field, expected in required.items():
        assert manifest[field] == expected
        assert summary[field] == expected
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## WARNING:" in report
    assert "privileged Oracle target perception" in report
