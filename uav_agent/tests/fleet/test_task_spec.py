from __future__ import annotations

from copy import deepcopy

import pytest

from fleet.task_spec import (
    FleetTaskSpecError,
    FleetTaskSpecV1,
    GoalType,
    MAX_GOALS,
)
from fleet.task_spec_json_schema import build_fleet_task_spec_json_schema


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
    target = properties["goals"]["items"]["properties"]["target_alias"]
    assert target["oneOf"][0]["enum"] == ["target_i", "target_j"]
    assert properties["goals"]["maxItems"] == MAX_GOALS
    assert properties["source_text"]["const"] == SOURCE

