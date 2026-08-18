"""JSON Schema generation for constrained dynamic Skill planning.

The builder exposes only model-visible semantic arguments.  Trusted geometry
in :class:`PlannerWorldContext` is used solely to obtain named-location enums;
coordinates, Oracle state, and runtime controller parameters never enter the
returned schema.
"""

from __future__ import annotations

from collections.abc import Mapping

from common.ids import (
    ROUTING_ID_PATTERN_TEXT,
    validate_mission_id,
    validate_routing_id,
    validate_uav_id,
)
from planner.policy import PlannerLimits
from planner.schemas import PlannerWorldContext
from planner.skill_catalog import SkillArgumentSpec, SkillCatalog, SkillContract


_STEP_ID_PATTERN = "^[a-z][a-z0-9_]{0,31}$"
_TARGET_REF_PATTERN = r"^\$[a-z][a-z0-9_]{0,31}\.target_id$"
_TOP_LEVEL_SKILL_ORDER = (
    "TAKEOFF",
    "GOTO",
    "HOVER",
    "SEARCH",
    "TRACK",
    "LAND",
)
_REVISION_ONLY_SKILL_ORDER = ("INSPECT",)


def _target_spec_schema() -> dict[str, object]:
    text = {"type": "string", "minLength": 1, "maxLength": 512}
    text_list = {
        "type": "array",
        "maxItems": 32,
        "items": dict(text),
    }
    # vLLM's structured-output backend currently returns HTTP 500 for
    # ``uniqueItems`` on string arrays.  TargetSpec performs the same
    # duplicate rejection after parsing, so keep the transport grammar
    # portable and retain the invariant at the trusted Python boundary.
    return {
        "type": "object",
        "properties": {
            "original_description": dict(text),
            "category": dict(text),
            "hard_attributes": dict(text_list),
            "soft_attributes": dict(text_list),
            "negative_constraints": dict(text_list),
            "relation_constraints": dict(text_list),
            "query_ladder": dict(text_list),
            "inspection_questions": dict(text_list),
            "immutable_identity_summary": dict(text),
            # Appearance observations are runtime evidence, not something an
            # initial planner may invent.
            "mutable_appearance_notes": {
                "type": "array",
                "maxItems": 0,
                "items": dict(text),
            },
        },
        "required": [
            "original_description",
            "category",
            "hard_attributes",
            "soft_attributes",
            "negative_constraints",
            "relation_constraints",
            "query_ladder",
            "inspection_questions",
            "immutable_identity_summary",
            "mutable_appearance_notes",
        ],
        "additionalProperties": False,
    }


def _argument_schema(
    argument: SkillArgumentSpec,
    *,
    world_context: PlannerWorldContext,
    limits: PlannerLimits,
) -> dict[str, object]:
    result: dict[str, object] = {"type": argument.value_type}

    if argument.allowed_values:
        result["enum"] = list(argument.allowed_values)
    if argument.minimum is not None:
        result["minimum"] = argument.minimum
    if argument.maximum is not None:
        result["maximum"] = argument.maximum

    if argument.name == "destination":
        result["enum"] = sorted(
            {
                *world_context.search_regions,
                *world_context.landing_zones,
                *world_context.navigation_points,
            }
        )
    elif argument.name == "region":
        result["enum"] = sorted(world_context.search_regions)
    elif argument.name == "zone":
        result["enum"] = sorted(world_context.landing_zones)
    elif argument.name == "target_ref":
        result["pattern"] = _TARGET_REF_PATTERN
    elif argument.name == "target_description":
        result["minLength"] = 1
        result["maxLength"] = 256
    elif argument.name == "duration_s" and "minimum" not in result:
        result["minimum"] = limits.min_track_duration_s
        result["maximum"] = limits.max_track_duration_s
    elif argument.name == "candidate_id":
        result["pattern"] = ROUTING_ID_PATTERN_TEXT
    elif argument.name == "viewpoint_change_deg":
        result["not"] = {"const": 0}
    elif argument.name == "on_target_lost":
        # This enum is a protocol invariant even if a caller supplies a
        # narrower catalog description.
        result["enum"] = ["REACQUIRE", "FAIL"]
    elif argument.name in {
        "altitude_m",
        "desired_altitude_m",
        "desired_distance_m",
    } and "minimum" not in result:
        result["exclusiveMinimum"] = 0.0

    return result


def _args_schema(
    contract: SkillContract,
    *,
    world_context: PlannerWorldContext,
    limits: PlannerLimits,
) -> dict[str, object]:
    properties = {
        argument.name: _argument_schema(
            argument,
            world_context=world_context,
            limits=limits,
        )
        for argument in contract.arguments
    }
    required = [
        argument.name for argument in contract.arguments if argument.required
    ]
    result: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def _recovery_schema(
    contract: SkillContract,
    *,
    world_context: PlannerWorldContext,
    limits: PlannerLimits,
) -> dict[str, object]:
    argument_properties = {
        argument.name: _argument_schema(
            argument,
            world_context=world_context,
            limits=limits,
        )
        for argument in contract.arguments
    }

    attempts = argument_properties.get("max_attempts")
    if attempts is not None:
        catalog_minimum = float(attempts.get("minimum", 1))
        catalog_maximum = float(
            attempts.get(
                "maximum",
                limits.max_reacquire_attempts_per_track,
            )
        )
        minimum = max(1.0, catalog_minimum)
        maximum = min(
            limits.max_reacquire_attempts_per_track,
            catalog_maximum,
        )
        if minimum > maximum:
            raise ValueError(
                "REACQUIRE max_attempts catalog range conflicts with limits"
            )
        attempts["minimum"] = minimum
        attempts["maximum"] = maximum

    properties: dict[str, object] = {
        "skill": {"type": "string", "const": "REACQUIRE"},
        **argument_properties,
    }
    required = ["skill"] + [
        argument.name for argument in contract.arguments if argument.required
    ]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _step_variant(
    contract: SkillContract,
    *,
    world_context: PlannerWorldContext,
    limits: PlannerLimits,
    recovery_contract: SkillContract | None,
) -> dict[str, object]:
    properties: dict[str, object] = {
        "id": {"type": "string", "pattern": _STEP_ID_PATTERN},
        "skill": {"type": "string", "const": contract.name},
        "args": _args_schema(
            contract,
            world_context=world_context,
            limits=limits,
        ),
    }
    if contract.name == "TRACK" and recovery_contract is not None:
        properties["recovery"] = _recovery_schema(
            recovery_contract,
            world_context=world_context,
            limits=limits,
        )
    return {
        "type": "object",
        "properties": properties,
        "required": ["id", "skill", "args"],
        "additionalProperties": False,
    }


def build_skill_plan_draft_json_schema(
    *,
    world_context: PlannerWorldContext,
    skill_catalog: SkillCatalog,
    limits: PlannerLimits,
    _trusted_inspect_candidate_id: str | None = None,
) -> dict[str, object]:
    """Build the strict, model-visible JSON Schema for ``SkillPlanDraft``.

    Cross-step semantics such as unique ids, call budgets, reference ordering,
    LAND/GOTO agreement, and recovery conflicts deliberately remain outside
    this structural schema and belong to the shared symbolic checker.
    """

    if not isinstance(world_context, PlannerWorldContext):
        raise TypeError("world_context must be a PlannerWorldContext")
    if not isinstance(skill_catalog, SkillCatalog):
        raise TypeError("skill_catalog must be a SkillCatalog")
    if not isinstance(limits, PlannerLimits):
        raise TypeError("limits must be a PlannerLimits")

    contracts: Mapping[str, SkillContract] = {
        contract.name: contract for contract in skill_catalog
    }
    recovery_contract = contracts.get("REACQUIRE")
    top_level_order = _TOP_LEVEL_SKILL_ORDER
    trusted_inspect_candidate_id: str | None = None
    if _trusted_inspect_candidate_id is not None:
        trusted_inspect_candidate_id = validate_routing_id(
            _trusted_inspect_candidate_id,
            "trusted_inspect_candidate_id",
        )
        top_level_order += _REVISION_ONLY_SKILL_ORDER
    variants = [
        _step_variant(
            contracts[name],
            world_context=world_context,
            limits=limits,
            recovery_contract=recovery_contract,
        )
        for name in top_level_order
        if name in contracts
        and contracts[name].top_level_allowed
        and not contracts[name].recovery_only
    ]
    if not variants:
        raise ValueError("skill_catalog has no top-level Skill variants")
    if trusted_inspect_candidate_id is not None:
        inspect_variant = next(
            (
                variant
                for variant in variants
                if variant["properties"]["skill"]["const"] == "INSPECT"
            ),
            None,
        )
        if inspect_variant is None:
            raise ValueError(
                "trusted INSPECT candidate requires INSPECT in skill_catalog"
            )
        inspect_variant["properties"]["args"]["properties"]["candidate_id"] = {
            "type": "string",
            "const": trusted_inspect_candidate_id,
        }

    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "steps": {
                "type": "array",
                "minItems": 2,
                "maxItems": limits.max_plan_steps,
                "items": {"oneOf": variants},
            },
        },
        "required": ["schema_version", "steps"],
        "additionalProperties": False,
    }


def build_skill_plan_v2_json_schema(
    *,
    world_context: PlannerWorldContext,
    skill_catalog: SkillCatalog,
    limits: PlannerLimits,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    _trusted_inspect_candidate_id: str | None = None,
) -> dict[str, object]:
    """Build the strict routed Qwen plan schema.

    By default this is the initial-plan schema and therefore omits INSPECT.
    The private candidate argument exists solely for the revision-schema
    builder, which binds INSPECT to an identifier already resolved by trusted
    CandidateBank code.
    """

    trusted_mission_id = validate_mission_id(mission_id)
    trusted_uav_id = validate_uav_id(uav_id)
    if isinstance(plan_version, bool) or not isinstance(
        plan_version, int
    ) or plan_version <= 0:
        raise ValueError("plan_version must be a positive integer")
    legacy = build_skill_plan_draft_json_schema(
        world_context=world_context,
        skill_catalog=skill_catalog,
        limits=limits,
        _trusted_inspect_candidate_id=_trusted_inspect_candidate_id,
    )
    legacy_steps = legacy["properties"]["steps"]  # type: ignore[index]
    variants = legacy_steps["items"]["oneOf"]  # type: ignore[index]
    for variant in variants:
        variant["properties"]["uav_id"] = {  # type: ignore[index]
            "type": "string",
            "const": trusted_uav_id,
        }
        variant["required"] = ["id", "uav_id", "skill", "args"]
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "const": 2},
            "mission_id": {"type": "string", "const": trusted_mission_id},
            "uav_id": {"type": "string", "const": trusted_uav_id},
            "plan_version": {"type": "integer", "const": plan_version},
            "target_spec": _target_spec_schema(),
            "steps": legacy_steps,
        },
        "required": [
            "schema_version",
            "mission_id",
            "uav_id",
            "plan_version",
            "target_spec",
            "steps",
        ],
        "additionalProperties": False,
    }


__all__ = [
    "build_skill_plan_draft_json_schema",
    "build_skill_plan_v2_json_schema",
]
