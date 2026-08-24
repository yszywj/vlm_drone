from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile

import pytest
import yaml

from configs.loader import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "configs" / "default.yaml"
MULTI_PATH = PROJECT_ROOT / "configs" / "multi_uav_demo.yaml"


def _raw(path: Path = MULTI_PATH) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_raw(raw: dict[str, object]):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "config.yaml"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        return load_config(path)


def test_default_is_canonical_singleton_with_read_only_compatibility_accessors() -> None:
    config = load_config(DEFAULT_PATH)

    assert tuple(item.id for item in config.uavs) == ("uav_1",)
    assert tuple(item.id for item in config.targets) == ("target",)
    assert tuple(config.camera_profiles) == ("default_rgb",)
    assert config.uav is config.uavs[0]
    assert config.target is config.targets[0]
    assert config.camera is config.camera_profiles[config.uav.camera_profile]
    with pytest.raises(TypeError):
        config.camera_profiles["other"] = config.camera  # type: ignore[index]


def test_legacy_layout_normalizes_to_plural_inventory() -> None:
    raw = _raw(DEFAULT_PATH)
    uav = raw.pop("uavs")[0]  # type: ignore[index,union-attr]
    uav.pop("camera_profile")
    raw["uav"] = uav
    profiles = raw.pop("camera_profiles")
    raw["camera"] = profiles["default_rgb"]  # type: ignore[index]
    raw["target"] = raw.pop("targets")[0]  # type: ignore[index,union-attr]
    config = _load_raw(raw)

    assert tuple(item.id for item in config.uavs) == ("uav_1",)
    assert tuple(config.camera_profiles) == ("default",)
    assert config.uav.camera_profile == "default"
    assert config.target.id == "target"


def test_multi_demo_loads_two_independent_inventories() -> None:
    config = load_config(MULTI_PATH)

    assert tuple(item.id for item in config.uavs) == ("uav_a", "uav_b")
    assert tuple(item.id for item in config.targets) == ("target_i", "target_j")
    assert {item.semantic_alias for item in config.targets} == {"目标i", "目标j"}
    assert all(item.camera_profile == "default_rgb" for item in config.uavs)
    assert config.model_broker.max_inflight_global == 4
    assert config.model_broker.max_inflight_per_uav == 1
    assert config.model_broker.max_pending_per_uav == 2
    assert config.model_broker.starvation_timeout_s == 15.0
    for legacy_name in ("uav", "target", "camera"):
        with pytest.raises(ValueError, match="ambiguous"):
            getattr(config, legacy_name)


@pytest.mark.parametrize(
    ("legacy_key", "canonical_key"),
    (("uav", "uavs"), ("target", "targets"), ("camera", "camera_profiles")),
)
def test_legacy_and_canonical_keys_cannot_be_mixed(
    legacy_key: str, canonical_key: str
) -> None:
    raw = _raw()
    raw[legacy_key] = deepcopy(raw[canonical_key])
    with pytest.raises(ConfigError, match="cannot be mixed"):
        _load_raw(raw)


def test_duplicate_ids_unknown_profile_and_second_entity_bounds_are_rejected() -> None:
    mutations = []

    duplicate_uav = _raw()
    duplicate_uav["uavs"][1]["id"] = "uav_a"  # type: ignore[index]
    mutations.append((duplicate_uav, "unique IDs"))

    duplicate_target = _raw()
    duplicate_target["targets"][1]["id"] = "target_i"  # type: ignore[index]
    mutations.append((duplicate_target, "unique IDs"))

    unknown_profile = _raw()
    unknown_profile["uavs"][1]["camera_profile"] = "missing"  # type: ignore[index]
    mutations.append((unknown_profile, "unknown camera_profile"))

    out_of_bounds_uav = _raw()
    out_of_bounds_uav["uavs"][1]["initial_position_xyz_m"] = [60, 0, 1]  # type: ignore[index]
    mutations.append((out_of_bounds_uav, "outside the scene"))

    out_of_bounds_target = _raw()
    out_of_bounds_target["targets"][1]["initial_region"]["max_xyz_m"] = [  # type: ignore[index]
        60,
        12,
        0.5,
    ]
    mutations.append((out_of_bounds_target, "outside the scene"))

    for raw, message in mutations:
        with pytest.raises(ConfigError, match=message):
            _load_raw(raw)


def test_camera_profiles_and_fleet_policy_are_strictly_validated() -> None:
    different_frequency = _raw()
    different_frequency["camera_profiles"]["fast"] = deepcopy(  # type: ignore[index]
        different_frequency["camera_profiles"]["default_rgb"]  # type: ignore[index]
    )
    different_frequency["camera_profiles"]["fast"]["frequency_hz"] = 12  # type: ignore[index]
    with pytest.raises(ConfigError, match="same frequency"):
        _load_raw(different_frequency)

    bad_policy = _raw()
    bad_policy["fleet"]["target_claim_policy"] = "SHARED"  # type: ignore[index]
    with pytest.raises(ConfigError, match="EXCLUSIVE"):
        _load_raw(bad_policy)

    bad_broker = _raw()
    bad_broker["model_broker"]["max_inflight_per_uav"] = 2  # type: ignore[index]
    with pytest.raises(ConfigError, match="exactly 1"):
        _load_raw(bad_broker)


def test_duplicate_yaml_camera_profile_key_is_rejected_before_normalization() -> None:
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    marker = "camera_profiles:\n  default_rgb:"
    duplicate = text.replace(
        marker,
        "camera_profiles:\n  default_rgb: &camera_profile",
        1,
    ).replace(
        "\nobstacle_perception:",
        "\n  default_rgb: *camera_profile\n\nobstacle_perception:",
        1,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "duplicate.yaml"
        path.write_text(duplicate, encoding="utf-8")
        with pytest.raises(ConfigError, match="duplicate key"):
            load_config(path)
