from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fleet.task_spec import (
    FleetTaskSpecError,
    FleetTaskSpecV1,
    GoalType,
    MAX_GOALS,
    MISSION_GOAL_TYPES,
    MissionGoal,
    TERMINATION_GOAL_TYPES,
)
from fleet.task_spec_json_schema import build_fleet_task_spec_json_schema
from planner.spatial import CircleRegion, PointTarget


SOURCE = "无人机A到世界坐标二十、三十附近搜索红色目标i，找到后跟踪十秒，最后降落"


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_text": SOURCE,
        "goals": [
            {
                "goal_id": "goal_search_i",
                "goal_type": "SEARCH_TARGET",
                "target_alias": "target_i",
                "spatial_constraint": {
                    "shape": "CIRCLE",
                    "frame": "WORLD_ENU",
                    "center_xyz_m": [20.0, 30.0, 0.0],
                    "radius_m": 15.0,
                },
                "duration_s": None,
                "distance_m": None,
                "strength": "MUST",
                "evidence_refs": ["ev_search"],
            },
            {
                "goal_id": "goal_track_i",
                "goal_type": "TRACK_TARGET",
                "target_alias": "target_i",
                "spatial_constraint": None,
                "duration_s": 10.0,
                "distance_m": None,
                "strength": "MUST",
                "evidence_refs": ["ev_track"],
            },
        ],
        "assignment_constraints": [
            {
                "constraint_id": "constraint_a_i",
                "uav_id": "uav_a",
                "goal_ids": ["goal_search_i", "goal_track_i"],
                "strength": "MUST",
                "evidence_refs": ["ev_uav"],
            }
        ],
        "ordering_constraints": [
            {
                "constraint_id": "order_search_track",
                "before_goal_id": "goal_search_i",
                "after_goal_id": "goal_track_i",
                "strength": "MUST",
                "evidence_refs": [],
            }
        ],
        "termination_goals": [
            {
                "goal_id": "goal_land",
                "goal_type": "LAND",
                "uav_id": None,
                "duration_s": None,
                "strength": "MUST",
                "evidence_refs": ["ev_land"],
            }
        ],
        "ambiguities": [],
        "source_evidence": [
            {"evidence_id": "ev_search", "quote": "搜索红色目标i"},
            {"evidence_id": "ev_track", "quote": "跟踪十秒"},
            {"evidence_id": "ev_uav", "quote": "无人机A"},
            {"evidence_id": "ev_land", "quote": "降落"},
        ],
    }


def _parse(payload: dict[str, object]) -> FleetTaskSpecV1:
    return FleetTaskSpecV1.from_dict(
        payload,
        trusted_uav_ids=("uav_a", "uav_b"),
        trusted_target_aliases=("target_i", "target_j"),
        supported_coordinate_frames=("WORLD_ENU", "HOME_ENU"),
        expected_source_text=SOURCE,
    )


def test_task_spec_round_trip_preserves_source_and_spatial_v3() -> None:
    spec = _parse(_payload())

    assert spec.source_text == SOURCE
    assert spec.to_dict() == _payload()
    assert spec.all_goal_ids == ("goal_search_i", "goal_track_i", "goal_land")
    assert spec.goal("goal_track_i").goal_type is GoalType.TRACK_TARGET


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.update(extra=True), "unknown fields"),
        (
            lambda p: p["goals"][0].update(velocity_mps=5.0),
            "forbidden",
        ),
        (
            lambda p: p["goals"][0].update(target_alias="target_unknown"),
            "unknown target",
        ),
        (
            lambda p: p["goals"][0]["spatial_constraint"].update(
                frame="CAMERA_FLU"
            ),
            "unsupported coordinate frame",
        ),
        (
            lambda p: p["goals"][0]["spatial_constraint"].update(
                radius_m=float("nan")
            ),
            "NaN or Infinity",
        ),
        (
            lambda p: p["source_evidence"][0].update(quote="model invented this"),
            "not verbatim",
        ),
    ],
)
def test_task_spec_rejects_unknown_privileged_untrusted_and_nonfinite(
    mutate, match: str
) -> None:
    payload = deepcopy(_payload())
    mutate(payload)
    with pytest.raises((FleetTaskSpecError, ValueError), match=match):
        _parse(payload)


def test_task_spec_rejects_dangling_and_excess_goals() -> None:
    payload = _payload()
    payload["assignment_constraints"][0]["goal_ids"] = ["goal_missing"]
    with pytest.raises(FleetTaskSpecError, match="unknown assignment goal"):
        _parse(payload)

    payload = _payload()
    payload["goals"] = [payload["goals"][0]] * (MAX_GOALS + 1)
    with pytest.raises(FleetTaskSpecError, match="at most"):
        _parse(payload)


@pytest.mark.parametrize("goal_type", TERMINATION_GOAL_TYPES)
def test_task_spec_rejects_termination_types_in_mission_goals(
    goal_type: GoalType,
) -> None:
    payload = _payload()
    payload["goals"][0]["goal_type"] = goal_type.value
    payload["goals"][0]["target_alias"] = None

    with pytest.raises(FleetTaskSpecError, match="must be placed.*termination_goals"):
        _parse(payload)


def test_structured_output_schema_uses_only_trusted_entity_and_frame_enums() -> None:
    schema = build_fleet_task_spec_json_schema(
        source_text=SOURCE,
        trusted_uav_ids=("uav_a", "uav_b"),
        trusted_target_aliases=("target_i", "target_j"),
        supported_coordinate_frames=("WORLD_ENU",),
    )
    properties = schema["properties"]
    assignment = properties["assignment_constraints"]["items"]
    assert assignment["additionalProperties"] is False
    assert assignment["properties"]["uav_id"]["enum"] == ["uav_a", "uav_b"]
    goal_variants = properties["goals"]["items"]["oneOf"]
    variants_by_type = {
        item["properties"]["goal_type"]["const"]: item
        for item in goal_variants
    }
    assert set(variants_by_type) == {item.value for item in MISSION_GOAL_TYPES}
    target = variants_by_type["SEARCH_TARGET"]["properties"]["target_alias"]
    assert target["oneOf"][0]["enum"] == ["target_i", "target_j"]
    assert properties["termination_goals"]["items"]["properties"]["goal_type"][
        "enum"
    ] == [item.value for item in TERMINATION_GOAL_TYPES]
    assert properties["goals"]["maxItems"] == MAX_GOALS
    assert properties["source_text"]["const"] == SOURCE
    # Duplicate rejection stays in the strict parser because this keyword
    # crashes the deployed vLLM structured-output grammar.
    assert "uniqueItems" not in json.dumps(schema, sort_keys=True)


def test_mission_goal_wire_schema_binds_spatial_shape_to_goal_type() -> None:
    schema = build_fleet_task_spec_json_schema(
        source_text=SOURCE,
        trusted_uav_ids=("uav_a",),
        trusted_target_aliases=("target_i",),
        supported_coordinate_frames=("WORLD_ENU",),
    )
    variants = schema["properties"]["goals"]["items"]["oneOf"]
    by_type = {
        item["properties"]["goal_type"]["const"]: item["properties"][
            "spatial_constraint"
        ]
        for item in variants
    }

    search = by_type["SEARCH_TARGET"]
    assert search["oneOf"][1] == {"type": "null"}
    assert all(
        "shape" in variant["properties"]
        for variant in search["oneOf"][0]["oneOf"]
    )

    for goal_type in ("TRACK_TARGET", "INSPECT_TARGET"):
        spatial = by_type[goal_type]
        assert spatial["oneOf"][0] == {"type": "null"}
        assert all(
            "shape" in item["properties"] for item in spatial["oneOf"][1]["oneOf"]
        )
        assert all(
            "kind" in item["properties"] for item in spatial["oneOf"][2]["oneOf"]
        )

    navigate = by_type["NAVIGATE"]
    assert all("kind" in variant["properties"] for variant in navigate["oneOf"])
    wire_schema = json.dumps(schema, sort_keys=True)
    assert '"enum": []' not in wire_schema
    assert "uniqueItems" not in wire_schema
    assert "(?!" not in wire_schema


def test_mission_goal_rejects_search_point_and_navigate_region() -> None:
    common = {
        "target_alias": "target_i",
        "duration_s": None,
        "distance_m": None,
        "strength": "MUST",
    }
    point = PointTarget("WORLD_ENU", (20.0, 30.0, 0.0))
    region = CircleRegion("WORLD_ENU", (20.0, 30.0, 0.0), 15.0)

    with pytest.raises(FleetTaskSpecError, match="SEARCH_TARGET.*RegionSpec"):
        MissionGoal("goal_search", "SEARCH_TARGET", spatial_constraint=point, **common)

    with pytest.raises(FleetTaskSpecError, match="NAVIGATE.*SpatialTarget"):
        MissionGoal(
            "goal_navigate",
            "NAVIGATE",
            spatial_constraint=region,
            **{**common, "target_alias": None},
        )


def test_mission_goal_track_and_inspect_keep_full_spatial_union() -> None:
    point = PointTarget("WORLD_ENU", (20.0, 30.0, 0.0))
    region = CircleRegion("WORLD_ENU", (20.0, 30.0, 0.0), 15.0)
    for goal_type in ("TRACK_TARGET", "INSPECT_TARGET"):
        for spatial_constraint in (None, point, region):
            goal = MissionGoal(
                f"goal_{goal_type.casefold()}",
                goal_type,
                "target_i",
                spatial_constraint,
                None,
                None,
                "MUST",
            )
            assert goal.spatial_constraint is spatial_constraint
