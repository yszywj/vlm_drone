from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from training.yolo.config import (
    YoloTrainConfig,
    YoloTrainingConfigError,
    load_yolo_train_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CONFIG = PROJECT_ROOT / "configs/yolo/train_yolo26s.yaml"


def _write_config(tmp_path: Path, **updates: object) -> Path:
    payload: dict[str, object] = {
        "model_family": "yolo",
        "task": "detect",
        "epochs": 10,
        "imgsz": 640,
        "batch": 4,
        "device": "cpu",
        "workers": 0,
        "patience": 2,
        "seed": 7,
        "deterministic": True,
        "amp": False,
        "cache": False,
        "resume": False,
        "project_dir": str(tmp_path / "runs"),
        "run_name": "yaml-run",
    }
    payload.update(updates)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_committed_training_config_parses_without_loading_a_model() -> None:
    config = load_yolo_train_config(COMMITTED_CONFIG, environ={})
    assert config.model_family == "yolo"
    assert config.task == "detect"
    assert config.base_model_path is None
    assert config.dataset_yaml is None
    assert config.resume is None
    assert config.run_name == "yolo26s_uav_target"


def test_cli_overrides_environment_which_overrides_yaml(tmp_path: Path) -> None:
    config = load_yolo_train_config(
        _write_config(tmp_path),
        environ={
            "UAV_AGENT_YOLO_EPOCHS": "20",
            "UAV_AGENT_YOLO_IMGSZ": "800",
            "UAV_AGENT_YOLO_RUN_NAME": "env-run",
        },
        overrides={"epochs": 30, "run_name": "cli-run"},
    )
    assert config.epochs == 30
    assert config.imgsz == 800
    assert config.run_name == "cli-run"


def test_empty_environment_value_does_not_erase_yaml(tmp_path: Path) -> None:
    config = load_yolo_train_config(
        _write_config(tmp_path, epochs=11),
        environ={"UAV_AGENT_YOLO_EPOCHS": ""},
    )
    assert config.epochs == 11


def test_unknown_yaml_and_override_keys_are_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path, surprise_download=True)
    with pytest.raises(YoloTrainingConfigError, match="unknown keys"):
        load_yolo_train_config(path, environ={})

    clean = _write_config(tmp_path)
    with pytest.raises(YoloTrainingConfigError, match="overrides.*unknown keys"):
        load_yolo_train_config(
            clean,
            environ={},
            overrides={"surprise_download": True},
        )


@pytest.mark.parametrize(
    "updates",
    (
        {"model_family": "unknown"},
        {"task": "classify"},
        {"epochs": 0},
        {"imgsz": True},
        {"workers": -1},
        {"device": "cuda:any"},
        {"run_name": "contains spaces"},
        {"deterministic": "true"},
    ),
)
def test_training_values_are_strictly_validated(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    with pytest.raises(YoloTrainingConfigError):
        load_yolo_train_config(_write_config(tmp_path, **updates), environ={})


def test_resume_requires_explicit_last_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(YoloTrainingConfigError, match="explicit last.pt"):
        YoloTrainConfig(resume=tmp_path / "best.pt")
    assert YoloTrainConfig(resume=tmp_path / "last.pt").resume == tmp_path / "last.pt"


def test_runtime_paths_must_be_existing_local_files(tmp_path: Path) -> None:
    model = tmp_path / "base.pt"
    dataset = tmp_path / "data.yaml"
    model.write_bytes(b"local checkpoint seam")
    dataset.write_text("path: .\ntrain: images\nval: images\nnames: [person]\n")
    config = YoloTrainConfig(base_model_path=model, dataset_yaml=dataset)
    config.require_runtime_paths()

    with pytest.raises(YoloTrainingConfigError, match="automatic downloads are disabled"):
        YoloTrainConfig(
            base_model_path=tmp_path / "missing.pt",
            dataset_yaml=dataset,
        ).require_runtime_paths()


def test_export_and_tracker_dependencies_cover_all_yolo_environment_entries() -> None:
    environment = (PROJECT_ROOT / "environment-yolo.yml").read_text(encoding="utf-8")
    requirements_in = (PROJECT_ROOT / "requirements/yolo.in").read_text(
        encoding="utf-8"
    )
    direct_lock = (PROJECT_ROOT / "requirements/yolo.lock").read_text(
        encoding="utf-8"
    )

    assert "lap==0.5.13" in environment
    assert "onnx==1.22.0" in environment
    assert "ultralytics==8.4.0" in environment
    assert "\nlap\n" in f"\n{requirements_in}"
    assert "\nonnx\n" in f"\n{requirements_in}"
    assert "ultralytics>=8.4.0,<8.5" in requirements_in
    assert "lap==0.5.13" in direct_lock
    assert "onnx==1.22.0" in direct_lock
    assert "ultralytics==8.4.0" in direct_lock
    assert "ultralytics==8.3.222" not in environment + direct_lock
    assert "does not claim to be a hash-complete transitive lock" in direct_lock
    dependency_lines = [
        line.strip().removeprefix("- ").lower()
        for text in (environment, requirements_in, direct_lock)
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(line.startswith("tensorrt") for line in dependency_lines)
