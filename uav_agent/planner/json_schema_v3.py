"""Strict JSON Schema builder for the opt-in Spatial Contract V3."""

from __future__ import annotations

from common.ids import ROUTING_ID_PATTERN_TEXT, validate_mission_id, validate_uav_id
from planner.spatial import CoordinateFrame, SpatialRelation
from target.types import TargetSpec
from skills.search_strategy import (
    SearchEntryPolicy,
    SearchRuntimeCapabilities,
    SearchStrategyType,
)


_STEP_ID_PATTERN = "^[a-z][a-z0-9_]{0,31}$"
_TARGET_REF_PATTERN = r"^\$[a-z][a-z0-9_]{0,31}\.target_id$"


def _object(
    properties: dict[str, object],
    required: list[str],
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _number(*, minimum: float | None = None, exclusive_minimum: float | None = None) -> dict[str, object]:
    result: dict[str, object] = {"type": "number"}
    if minimum is not None:
        result["minimum"] = minimum
    if exclusive_minimum is not None:
        result["exclusiveMinimum"] = exclusive_minimum
    return result


def _point_schema() -> dict[str, object]:
    return {"type": "array", "minItems": 3, "maxItems": 3, "items": _number()}


def _points_schema(*, minimum: int, maximum: int) -> dict[str, object]:
    return {"type": "array", "minItems": minimum, "maxItems": maximum, "items": _point_schema()}


def spatial_target_json_schema(*, include_route: bool = True) -> dict[str, object]:
    frame = {"type": "string", "enum": [item.value for item in CoordinateFrame]}
    relation = {"type": "string", "enum": [item.value for item in SpatialRelation]}
    text = {"type": "string", "minLength": 1, "maxLength": 128}
    variants = [
            _object({"kind": {"const": "NAMED_LOCATION"}, "name": text}, ["kind", "name"]),
            _object({"kind": {"const": "POINT"}, "frame": frame, "xyz_m": _point_schema()}, ["kind", "frame", "xyz_m"]),
            _object(
                {
                    "kind": {"const": "RELATIONAL_POINT"},
                    "relation": relation,
                    "reference_id": text,
                    "distance_m": _number(exclusive_minimum=0.0),
                    "frame": frame,
                },
                ["kind", "relation", "reference_id", "distance_m"],
            ),
    ]
    if include_route:
        variants.append(
            _object(
                {"kind": {"const": "ROUTE"}, "frame": frame, "waypoints_xyz_m": _points_schema(minimum=2, maximum=64)},
                ["kind", "frame", "waypoints_xyz_m"],
            )
        )
    return {"oneOf": variants}


def region_spec_json_schema() -> dict[str, object]:
    frame = {"type": "string", "enum": [item.value for item in CoordinateFrame]}
    relation = {"type": "string", "enum": [item.value for item in SpatialRelation]}
    text = {"type": "string", "minLength": 1, "maxLength": 128}
    return {
        "oneOf": [
            _object(
                {"shape": {"const": "CIRCLE"}, "frame": frame, "center_xyz_m": _point_schema(), "radius_m": _number(exclusive_minimum=0.0)},
                ["shape", "frame", "center_xyz_m", "radius_m"],
            ),
            _object(
                {
                    "shape": {"const": "RECTANGLE"}, "frame": frame,
                    "center_xyz_m": _point_schema(), "width_m": _number(exclusive_minimum=0.0),
                    "height_m": _number(exclusive_minimum=0.0), "yaw_deg": _number(),
                    "entry_point_xyz_m": _point_schema(),
                },
                ["shape", "frame", "center_xyz_m", "width_m", "height_m"],
            ),
            _object(
                {
                    "shape": {"const": "SECTOR"}, "frame": frame,
                    "origin_xyz_m": _point_schema(), "azimuth_center_deg": _number(),
                    "azimuth_span_deg": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 360.0},
                    "distance_range_m": {"type": "array", "minItems": 2, "maxItems": 2, "items": _number(minimum=0.0)},
                },
                ["shape", "frame", "origin_xyz_m", "azimuth_center_deg", "azimuth_span_deg", "distance_range_m"],
            ),
            _object(
                {"shape": {"const": "POLYGON"}, "frame": frame, "vertices_xyz_m": _points_schema(minimum=3, maximum=64)},
                ["shape", "frame", "vertices_xyz_m"],
            ),
            _object(
                {"shape": {"const": "CORRIDOR"}, "frame": frame, "centerline_xyz_m": _points_schema(minimum=2, maximum=64), "half_width_m": _number(exclusive_minimum=0.0)},
                ["shape", "frame", "centerline_xyz_m", "half_width_m"],
            ),
            _object(
                {
                    "shape": {"const": "RELATIONAL"}, "relation": relation,
                    "reference_id": text, "distance_m": _number(exclusive_minimum=0.0),
                    "extent_m": {"type": "array", "minItems": 2, "maxItems": 2, "items": _number(exclusive_minimum=0.0)},
                    "frame": frame,
                },
                ["shape", "relation", "reference_id", "distance_m", "extent_m"],
            ),
        ]
    }


def search_strategy_json_schema(
    *,
    search_runtime_capabilities: SearchRuntimeCapabilities | None = None,
) -> dict[str, object]:
    capabilities = (
        SearchRuntimeCapabilities()
        if search_runtime_capabilities is None
        else search_runtime_capabilities
    )
    if not isinstance(capabilities, SearchRuntimeCapabilities):
        raise TypeError(
            "search_runtime_capabilities must be a SearchRuntimeCapabilities or None"
        )
    variants: list[dict[str, object]] = []
    for kind in capabilities.supported_strategies:
        properties: dict[str, object] = {
            "kind": {"const": kind.value},
            "spacing_m": _number(exclusive_minimum=0.0),
            "max_viewpoints": {"type": "integer", "minimum": 1, "maximum": 128},
        }
        required = ["kind"]
        if kind is SearchStrategyType.MODEL_WAYPOINTS:
            properties["model_waypoints_xyz_m"] = _points_schema(minimum=1, maximum=128)
            required.append("model_waypoints_xyz_m")
        if kind is SearchStrategyType.RANDOM_COVERAGE:
            properties["random_seed"] = {"type": "integer", "minimum": -(2**31), "maximum": 2**31 - 1}
        variants.append(_object(properties, required))
    return {"oneOf": variants}


def _yaw_properties() -> dict[str, object]:
    return {
        "altitude_m": _number(exclusive_minimum=0.0),
        "yaw_mode": {"type": "string", "enum": ["KEEP_CURRENT", "COURSE_ALIGNED", "FACE_POINT", "FIXED"]},
        "yaw_deg": _number(),
    }


def _search_args_schema(
    search_runtime_capabilities: SearchRuntimeCapabilities,
) -> dict[str, object]:
    common: dict[str, object] = {
        "region": region_spec_json_schema(),
        "strategy": search_strategy_json_schema(
            search_runtime_capabilities=search_runtime_capabilities
        ),
        "target_description": {"type": "string", "minLength": 1, "maxLength": 256},
        "search_altitude_m": _number(exclusive_minimum=0.0),
        "timeout_s": _number(exclusive_minimum=0.0),
        "transit_speed_mps": _number(exclusive_minimum=0.0),
        "scan_yaw_rate_rad_s": _number(exclusive_minimum=0.0),
    }
    required = [
        "region", "strategy", "entry_policy", "target_description",
        "search_altitude_m", "timeout_s",
    ]
    variants: list[dict[str, object]] = []
    for policy in SearchEntryPolicy:
        properties = {**common, "entry_policy": {"const": policy.value}}
        policy_required = list(required)
        if policy is SearchEntryPolicy.USER_ANCHOR:
            properties["user_anchor_xyz_m"] = _point_schema()
            policy_required.append("user_anchor_xyz_m")
        elif policy is SearchEntryPolicy.MODEL_SELECTED:
            properties["model_selected_entry_xyz_m"] = _point_schema()
            policy_required.append("model_selected_entry_xyz_m")
        variants.append(_object(properties, policy_required))
    return {"oneOf": variants}


def _recovery_schema() -> dict[str, object]:
    return _object(
        {
            "skill": {"const": "REACQUIRE"},
            "max_attempts": {"type": "integer", "minimum": 0, "maximum": 2},
            "search_radius_m": {"type": "number", "minimum": 3.0, "maximum": 20.0},
            "timeout_s": {"type": "number", "minimum": 5.0, "maximum": 60.0},
        },
        ["skill", "max_attempts"],
    )


def _step(skill: str, uav_id: str, args: dict[str, object], *, recovery: bool = False) -> dict[str, object]:
    properties: dict[str, object] = {
        "id": {"type": "string", "pattern": _STEP_ID_PATTERN},
        "uav_id": {"const": uav_id},
        "skill": {"const": skill},
        "args": args,
    }
    if recovery:
        properties["recovery"] = _recovery_schema()
    return _object(properties, ["id", "uav_id", "skill", "args"])


def _target_spec_schema() -> dict[str, object]:
    text = {"type": "string", "minLength": 1, "maxLength": 512}
    texts = {"type": "array", "maxItems": 32, "items": dict(text)}
    return _object(
        {
            "original_description": dict(text), "category": dict(text),
            "hard_attributes": dict(texts), "soft_attributes": dict(texts),
            "negative_constraints": dict(texts), "relation_constraints": dict(texts),
            "query_ladder": dict(texts), "inspection_questions": dict(texts),
            "immutable_identity_summary": dict(text),
            "mutable_appearance_notes": {"type": "array", "maxItems": 0, "items": dict(text)},
        },
        ["original_description", "category", "hard_attributes", "soft_attributes", "negative_constraints", "relation_constraints", "query_ladder", "inspection_questions", "immutable_identity_summary", "mutable_appearance_notes"],
    )


def build_skill_plan_v3_json_schema(
    *,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    search_runtime_capabilities: SearchRuntimeCapabilities | None = None,
    trusted_target_spec: TargetSpec | None = None,
    require_empty_assumptions: bool = False,
) -> dict[str, object]:
    """Return the initial-plan V3 schema bound to trusted routing values."""

    mission = validate_mission_id(mission_id)
    uav = validate_uav_id(uav_id)
    if isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version <= 0:
        raise ValueError("plan_version must be a positive integer")
    capabilities = (
        SearchRuntimeCapabilities()
        if search_runtime_capabilities is None
        else search_runtime_capabilities
    )
    if not isinstance(capabilities, SearchRuntimeCapabilities):
        raise TypeError(
            "search_runtime_capabilities must be a SearchRuntimeCapabilities or None"
        )
    if trusted_target_spec is not None and not isinstance(
        trusted_target_spec, TargetSpec
    ):
        raise TypeError("trusted_target_spec must be a TargetSpec or None")
    if not isinstance(require_empty_assumptions, bool):
        raise TypeError("require_empty_assumptions must be bool")
    empty_args = _object({}, [])
    variants = [
        _step("TAKEOFF", uav, _object(_yaw_properties(), [])),
        _step("GOTO", uav, _object({"target": spatial_target_json_schema(include_route=False), **_yaw_properties()}, ["target"])),
        _step("HOVER", uav, _object({"duration_s": {"type": "number", "minimum": 1.0, "maximum": 60.0}, "yaw_mode": {"type": "string", "enum": ["KEEP_CURRENT", "FIXED"]}, "yaw_deg": _number()}, ["duration_s"])),
        _step("SEARCH", uav, _search_args_schema(capabilities)),
        _step("TRACK", uav, _object({"target_ref": {"type": "string", "pattern": _TARGET_REF_PATTERN}, "duration_s": _number(exclusive_minimum=0.0), "desired_altitude_m": _number(exclusive_minimum=0.0), "desired_distance_m": _number(exclusive_minimum=0.0), "on_target_lost": {"type": "string", "enum": ["REACQUIRE", "FAIL"]}}, ["target_ref", "duration_s"]), recovery=True),
        _step("LAND", uav, _object({"zone": {"type": "string", "minLength": 1, "maxLength": 128}, "yaw_mode": {"type": "string", "enum": ["KEEP_CURRENT", "FIXED"]}, "yaw_deg": _number()}, ["zone"])),
    ]
    del empty_args
    target_spec_schema: dict[str, object]
    required = [
        "schema_version",
        "mission_id",
        "uav_id",
        "plan_version",
        "assumptions",
        "steps",
    ]
    if trusted_target_spec is None:
        target_spec_schema = _target_spec_schema()
    else:
        target_spec_value = trusted_target_spec.to_dict()
        target_spec_schema = _object(
            {
                key: {"const": value}
                for key, value in target_spec_value.items()
            },
            list(target_spec_value),
        )
        required.append("target_spec")
    return _object(
        {
            "schema_version": {"const": 3},
            "mission_id": {"type": "string", "const": mission, "pattern": ROUTING_ID_PATTERN_TEXT},
            "uav_id": {"type": "string", "const": uav, "pattern": ROUTING_ID_PATTERN_TEXT},
            "plan_version": {"type": "integer", "const": plan_version},
            "assumptions": {
                "type": "array",
                "maxItems": 0 if require_empty_assumptions else 32,
                "items": _object(
                    {
                        "source_text": {"type": "string", "minLength": 1, "maxLength": 256},
                        "interpretation": {"type": "string", "minLength": 1, "maxLength": 512},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    ["source_text", "interpretation", "confidence"],
                ),
            },
            "target_spec": target_spec_schema,
            "steps": {
                "type": "array",
                "minItems": 2,
                "maxItems": 10,
                "items": {"oneOf": variants},
            },
        },
        required,
    )


# Verbose spelling mirrors the V2 builder and helps migration code stay clear.
build_skill_plan_draft_v3_json_schema = build_skill_plan_v3_json_schema


__all__ = [
    "build_skill_plan_draft_v3_json_schema", "build_skill_plan_v3_json_schema",
    "region_spec_json_schema", "search_strategy_json_schema",
    "spatial_target_json_schema",
]
