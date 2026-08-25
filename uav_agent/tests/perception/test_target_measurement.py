from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from perception.measurement import TargetMeasurement


def _measurement(**overrides: object) -> TargetMeasurement:
    values: dict[str, object] = {
        "timestamp_s": 1.0,
        "candidate_id": "candidate_1",
        "tracker_id": "track_7",
        "pixel_uv": (320.0, 240.0),
        "raw_depth_m": 8.0,
        "corrected_depth_m": 8.0,
        "position_camera_flu_m": (8.0, 0.0, 0.0),
        "position_world_m": (9.0, 2.0, 3.0),
        "covariance_world_m2": (
            (0.04, 0.0, 0.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.06),
        ),
        "measurement_quality": 0.8,
        "source": "rgbd_depth_geometry",
    }
    values.update(overrides)
    return TargetMeasurement(**values)  # type: ignore[arg-type]


def test_measurement_is_finite_immutable_and_contains_no_truth_fields() -> None:
    measurement = _measurement(raw_depth_m=None)
    assert measurement.position_xyz_m == measurement.position_world_m
    assert np.all(np.isfinite(np.asarray(measurement.covariance_world_m2)))
    assert set(measurement.to_dict()) == {item.name for item in fields(measurement)}
    forbidden_fragments = (
        "truth",
        "oracle",
        "velocity",
        "prim",
        "motion_seed",
        "instance_id",
        "evaluator",
    )
    assert not any(
        fragment in item.name
        for item in fields(measurement)
        for fragment in forbidden_fragments
    )
    with pytest.raises(FrozenInstanceError):
        measurement.measurement_quality = 0.1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("timestamp_s", float("nan"), "finite"),
        ("pixel_uv", (float("inf"), 2.0), "finite"),
        ("raw_depth_m", 0.0, "greater than zero"),
        ("corrected_depth_m", -1.0, "greater than zero"),
        ("position_world_m", (1.0, 2.0, float("nan")), "finite"),
        ("measurement_quality", 1.01, "within"),
    ),
)
def test_invalid_numeric_measurement_fields_fail_closed(
    field_name: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _measurement(**{field_name: value})


def test_covariance_must_be_symmetric_positive_semidefinite() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        _measurement(
            covariance_world_m2=(
                (1.0, 0.2, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        _measurement(
            covariance_world_m2=(
                (1.0, 0.0, 0.0),
                (0.0, -0.01, 0.0),
                (0.0, 0.0, 1.0),
            )
        )

