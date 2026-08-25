from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from perception.target_query import ALLOWED_TARGET_QUERY_FIELDS, TargetQuerySpec
from target import TargetSpec


def _query(**overrides: object) -> TargetQuerySpec:
    values: dict[str, object] = {
        "target_alias": "target_red",
        "detector_class_id": 0,
        "detector_class_name": "cube",
        "hard_attributes": {"color": "red", "shape": "cube"},
        "soft_description": "moving red cube",
    }
    values.update(overrides)
    return TargetQuerySpec(**values)  # type: ignore[arg-type]


def test_query_has_exact_immutable_production_whitelist() -> None:
    query = _query()
    assert tuple(field.name for field in fields(query)) == ALLOWED_TARGET_QUERY_FIELDS
    assert not hasattr(query, "__dict__")
    assert not hasattr(query, "position_world_m")
    assert not hasattr(query, "velocity_world_mps")
    assert not hasattr(query, "motion_seed")
    assert not hasattr(query, "prim_path")
    with pytest.raises(FrozenInstanceError):
        query.target_alias = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        query.hard_attributes["color"] = "blue"  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("position_world_m", "1,2,3"),
        ("target_velocity", "1,0,0"),
        ("initial_region", "secret"),
        ("motion_seed", "42"),
        ("prim_path", "/World/Target"),
        ("instance_id", "77"),
        ("future_trajectory", "next pose"),
    ),
)
def test_nested_truth_fields_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="truth (marker|field)"):
        _query(hard_attributes={field: value})


def test_unknown_top_level_truth_field_cannot_be_constructed() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        TargetQuerySpec(  # type: ignore[call-arg]
            target_alias="target_red",
            detector_class_id=0,
            detector_class_name="cube",
            hard_attributes={},
            soft_description="red cube",
            position_world_m=(1.0, 2.0, 3.0),
        )


@pytest.mark.parametrize("name", ("position", "velocity", "seed", "trajectory"))
def test_generic_geometry_cannot_hide_inside_attributes(name: str) -> None:
    with pytest.raises(ValueError, match="forbidden truth field"):
        _query(hard_attributes={name: "secret"})


def test_assignment_projection_contains_semantics_but_not_search_geometry() -> None:
    assignment_semantics = TargetSpec(
        "moving red cube",
        category="cube",
        hard_attributes=("color=red", "shape=cube"),
    )
    query = TargetQuerySpec.from_assignment_semantics(
        target_alias="target_red",
        target_spec=assignment_semantics,
        detector_class_id=0,
        detector_class_name="cube",
    )
    assert dict(query.hard_attributes) == {"color": "red", "shape": "cube"}
    assert query.soft_description == "moving red cube"
    assert query.to_semantic_target_spec().category == "cube"
    assert query.to_audit_dict() == {
        "allowed_target_query_fields": list(ALLOWED_TARGET_QUERY_FIELDS)
    }
