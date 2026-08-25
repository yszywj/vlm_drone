from __future__ import annotations

from pathlib import Path

import pytest

from configs.loader import load_config
from perception.factory import (
    TargetPerceptionConfigurationError,
    build_target_perception_runtime,
    preflight_fleet_yolo_services,
    validate_target_perception_preflight,
)
from perception.mode import TargetPerceptionModeError, resolve_target_perception_mode
from skills.types import SkillName


ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "configs/multi_uav_oracle.yaml"
YOLO = ROOT / "configs/multi_uav_cube_yolo.yaml"


@pytest.mark.parametrize(
    ("config_path", "mode", "acknowledged"),
    (
        (ORACLE, "yolo", False),
        (YOLO, "oracle", True),
    ),
)
def test_mode_and_yaml_backend_cannot_be_crossed(
    config_path: Path,
    mode: str,
    acknowledged: bool,
) -> None:
    config = load_config(config_path)
    with pytest.raises(TargetPerceptionModeError, match="requires YAML"):
        resolve_target_perception_mode(
            mode,
            backend=config.target_perception.backend,
            acknowledge_privileged_oracle=acknowledged,
        )


def test_factory_rechecks_resolved_mode_against_direct_config() -> None:
    config = load_config(ORACLE)

    class UntouchableEnvironment:
        def __getattribute__(self, name: str):
            if name.startswith("make_") or name.startswith("get_"):
                raise AssertionError("mismatch reached the environment boundary")
            return object.__getattribute__(self, name)

    with pytest.raises(TargetPerceptionConfigurationError, match="does not match"):
        build_target_perception_runtime(
            config,
            resolved_mode=resolve_target_perception_mode("yolo"),
            environment=UntouchableEnvironment(),
            uav_id="uav_a",
        )


def test_oracle_config_is_not_a_valid_yolo_preflight_input() -> None:
    config = load_config(ORACLE)
    factory_calls = 0

    def forbidden_factory(**kwargs):
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError(kwargs)

    with pytest.raises(
        TargetPerceptionConfigurationError,
        match="requires backend=ultralytics_service",
    ):
        preflight_fleet_yolo_services(
            config,
            ("uav_a", "uav_b"),
            client_factory=forbidden_factory,
        )
    assert factory_calls == 0


def test_disabled_backend_rejects_target_skills_without_fallback() -> None:
    with pytest.raises(Exception, match="disabled cannot execute target Skills"):
        validate_target_perception_preflight(
            "disabled",
            (SkillName.SEARCH, SkillName.TRACK),
        )


def test_dual_modes_share_fleet_task_inputs_but_not_perception_backend() -> None:
    oracle = load_config(ORACLE)
    yolo = load_config(YOLO)

    assert oracle.uavs == yolo.uavs
    assert oracle.targets == yolo.targets
    assert oracle.scene == yolo.scene
    assert oracle.fleet == yolo.fleet
    assert oracle.search == yolo.search
    assert oracle.planner == yolo.planner
    assert oracle.target_perception.backend == "oracle_evaluation"
    assert yolo.target_perception.backend == "ultralytics_service"

