"""JSON Schema generation for constrained dynamic Skill planning.

The builder exposes only model-visible semantic arguments.  Trusted geometry
in :class:`PlannerWorldContext` is used solely to obtain named-location enums;
coordinates, Oracle state, and runtime controller parameters never enter the
returned schema.
"""

from __future__ import annotations

from collections.abc import Mapping

from planner.policy import PlannerLimits
from planner.schemas import PlannerWorldContext
from planner.skill_catalog import SkillArgumentSpec, SkillCatalog, SkillContract


_STEP_ID_PATTERN = "^[a-z][a-z0-9_]{0,31}$"
_TARGET_REF_PATTERN = r"^\$[a-z][a-z0-9_]{0,31}\.target_id$"
_TOP_LEVEL_SKILL_ORDER = ("TAKEOFF", "GOTO", "SEARCH", "TRACK", "LAND")


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
    elif argument.name == "duration_s":
        result["minimum"] = limits.min_track_duration_s
        result["maximum"] = limits.max_track_duration_s
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
    variants = [
        _step_variant(
            contracts[name],
            world_context=world_context,
            limits=limits,
            recovery_contract=recovery_contract,
        )
        for name in _TOP_LEVEL_SKILL_ORDER
        if name in contracts
        and contracts[name].top_level_allowed
        and not contracts[name].recovery_only
    ]
    if not variants:
        raise ValueError("skill_catalog has no top-level Skill variants")

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


__all__ = ["build_skill_plan_draft_json_schema"]
