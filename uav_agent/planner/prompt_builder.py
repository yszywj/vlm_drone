"""Shared, text-only prompt construction for mission-intent planning.

This module is the single source of truth for the messages used by the
runtime planner and Planner dataset generation.  It deliberately projects a
``PlannerWorldContext`` onto the small public subset the model is allowed to
see; Skill geometry and other runtime-only state remain unavailable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import isfinite
from numbers import Real
import re
import unicodedata

from models.base import ChatMessage
from common.ids import validate_mission_id, validate_uav_id
from planner.schemas import PlannerWorldContext
from planner.policy import PlannerLimits, PlannerPolicy
from planner.skill_catalog import SkillCatalog, initial_planner_catalog
from planner.text_safety import reject_forbidden_planner_text
from skills.search_strategy import SearchRuntimeCapabilities


# Planner-visible world metadata is configuration, not user prose.  Reject
# hidden-state and media labels here instead of trying to redact them: a
# redaction could silently turn one configured name into another or leave a
# semantically equivalent leak behind.  Both token and compact forms are
# checked so common spelling variants such as ``target_position``,
# ``target-position`` and ``targetPosition`` fail closed.
_FORBIDDEN_CONTEXT_ASCII_TOKENS = frozenset(
    {
        "oracle",
        "evaluator",
        "image",
        "video",
        "frame",
        "camera",
        "spawn",
    }
)
_FORBIDDEN_CONTEXT_ASCII_COMPACT = (
    "oracle",
    "targetspawn",
    "targetsspawn",
    "targetpose",
    "targetposition",
    "targetvelocity",
    "targetcoordinate",
    "targetcoordinates",
    "targettruth",
    "groundtruth",
    "trueposition",
    "realposition",
    "evaluatorframe",
    "evaluator",
    "image",
    "video",
    "camera",
)
_FORBIDDEN_CONTEXT_UNICODE = (
    "出生点",
    "真实位置",
    "目标位姿",
    "目标的位姿",
    "目标位置",
    "目标的位置",
    "目标坐标",
    "目标的坐标",
    "目标速度",
    "目标的速度",
    "目标真值",
    "目标的真值",
    "真值",
    "评估器",
    "评价器",
    "评测器",
    "图像",
    "影像",
    "视频",
    "相机",
)
_ASCII_WORD = re.compile(r"[a-z0-9]+")


def build_mission_planner_messages(
    instruction: str,
    world_context: PlannerWorldContext,
    system_prompt: str,
) -> tuple[ChatMessage, ...]:
    """Build the canonical system/user messages for ``MissionIntent`` parsing.

    Only the same trusted public context used by the production
    :class:`planner.llm_planner.LLMPlanner` is serialized.  In particular, the
    projection contains no initial UAV pose, region geometry, landing
    coordinates, timeouts, target truth, evaluator state, or image data.

    The compact, sorted JSON representation is intentional: callers creating
    Planner examples must receive byte-for-byte identical message contents to
    runtime inference for the same inputs.
    """

    normalized_instruction = _non_empty_text(instruction, "instruction")
    normalized_system_prompt = _non_empty_text(system_prompt, "system_prompt")
    if not isinstance(world_context, PlannerWorldContext):
        raise TypeError("world_context must be a PlannerWorldContext")
    _validate_planner_visible_world_text(world_context)

    safe_context = {
        "scene_bounds_m": {
            "minimum": list(world_context.scene_min_xyz_m),
            "maximum": list(world_context.scene_max_xyz_m),
        },
        "search_regions": [
            {"name": name, "description": spec.description}
            for name, spec in sorted(world_context.search_regions.items())
        ],
        "landing_zones": [
            {"name": name, "description": spec.description}
            for name, spec in sorted(world_context.landing_zones.items())
        ],
        "default_takeoff_altitude_m": (
            world_context.default_takeoff_altitude_m
        ),
        "default_track_duration_s": world_context.default_track_duration_s,
    }
    payload = {
        "task": "Parse the user instruction into one MissionIntent JSON object.",
        "trusted_world_context": safe_context,
        "user_instruction": normalized_instruction,
    }
    user_prompt = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        ChatMessage(role="system", content=normalized_system_prompt),
        ChatMessage(role="user", content=user_prompt),
    )


_DYNAMIC_LIMIT_DEFAULTS: dict[str, int | float] = {
    "max_plan_steps": 10,
    "max_goto_calls": 5,
    "max_search_calls": 1,
    "max_track_calls": 2,
    "max_reacquire_attempts_per_track": 2,
    "max_total_reacquire_attempts": 4,
    "min_track_duration_s": 1.0,
    "max_track_duration_s": 600.0,
}
_INTEGER_DYNAMIC_LIMITS = frozenset(
    {
        "max_plan_steps",
        "max_goto_calls",
        "max_search_calls",
        "max_track_calls",
        "max_reacquire_attempts_per_track",
        "max_total_reacquire_attempts",
    }
)


def build_dynamic_skill_planner_messages(
    instruction: str,
    world_context: PlannerWorldContext,
    skill_catalog: SkillCatalog,
    planner_limits: object,
    system_prompt: str,
    planner_policy: object = None,
    *,
    mission_id: str | None = None,
    uav_id: str | None = None,
    plan_version: int | None = None,
) -> tuple[ChatMessage, ...]:
    """Build the deterministic, text-only prompt for ``SkillPlanDraft``.

    The projection exposes names and prose descriptions but never the geometry
    behind a search region, landing zone, or navigation point.  The model gets
    only the model-facing catalog, not the runtime Goal dataclasses.
    """

    normalized_instruction = _non_empty_text(instruction, "instruction")
    normalized_system_prompt = _non_empty_text(system_prompt, "system_prompt")
    if not isinstance(world_context, PlannerWorldContext):
        raise TypeError("world_context must be a PlannerWorldContext")
    if not isinstance(skill_catalog, SkillCatalog):
        raise TypeError("skill_catalog must be a SkillCatalog")

    _validate_planner_visible_world_text(world_context)
    for index, (mapping_name, spec) in enumerate(
        sorted(world_context.navigation_points.items())
    ):
        _reject_forbidden_context_text(
            mapping_name,
            f"navigation_points key at index {index}",
        )
        _reject_forbidden_context_text(
            spec.name,
            f"navigation_points spec name at index {index}",
        )
        _reject_forbidden_context_text(
            spec.description,
            f"navigation_points description at index {index}",
        )

    safe_context = {
        "scene_bounds_m": {
            "minimum": list(world_context.scene_min_xyz_m),
            "maximum": list(world_context.scene_max_xyz_m),
        },
        "search_regions": [
            {"name": name, "description": spec.description}
            for name, spec in sorted(world_context.search_regions.items())
        ],
        "landing_zones": [
            {"name": name, "description": spec.description}
            for name, spec in sorted(world_context.landing_zones.items())
        ],
        "navigation_points": [
            {"name": name, "description": spec.description}
            for name, spec in sorted(world_context.navigation_points.items())
        ],
        "default_takeoff_altitude_m": world_context.default_takeoff_altitude_m,
        "default_track_duration_s": world_context.default_track_duration_s,
    }
    projected_limits = _project_dynamic_limits(planner_limits)
    trusted_limits = PlannerLimits(**projected_limits)
    # Initial mission planning has no trusted CandidateBank entry.  INSPECT is
    # a runtime-revision-only capability and must never be advertised here.
    model_catalog = initial_planner_catalog(skill_catalog)
    payload = {
        "task": "Create one constrained SkillPlanDraft JSON object.",
        "trusted_world_context": safe_context,
        "skill_catalog": model_catalog.to_prompt_dict(),
        "planner_limits": projected_limits,
        "trusted_planner_policy": _project_dynamic_policy(
            planner_policy,
            limits=trusted_limits,
        ),
        "user_instruction": normalized_instruction,
    }
    routing_supplied = (
        mission_id is not None,
        uav_id is not None,
        plan_version is not None,
    )
    if any(routing_supplied) and not all(routing_supplied):
        raise ValueError(
            "mission_id, uav_id, and plan_version must be supplied together"
        )
    if all(routing_supplied):
        trusted_mission_id = validate_mission_id(mission_id)
        trusted_uav_id = validate_uav_id(uav_id)
        if isinstance(plan_version, bool) or not isinstance(
            plan_version, int
        ) or plan_version <= 0:
            raise ValueError("plan_version must be a positive integer")
        payload["task"] = (
            "Create one constrained SkillPlanDraft schema-v2 JSON object, "
            "extract one immutable TargetSpec from the user instruction, and "
            "echo every trusted routing field exactly."
        )
        payload["trusted_routing"] = {
            "schema_version": 2,
            "mission_id": trusted_mission_id,
            "uav_id": trusted_uav_id,
            "plan_version": plan_version,
            "step_uav_id_must_equal": trusted_uav_id,
        }
    user_prompt = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        ChatMessage(role="system", content=normalized_system_prompt),
        ChatMessage(role="user", content=user_prompt),
    )


def build_spatial_v3_skill_planner_messages(
    instruction: str,
    world_context: PlannerWorldContext,
    skill_catalog: SkillCatalog,
    planner_limits: object,
    system_prompt: str,
    planner_policy: object = None,
    *,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    search_runtime_capabilities: SearchRuntimeCapabilities | None = None,
    trusted_target_locked: bool = False,
    allow_trusted_safety_completion: bool = False,
) -> tuple[ChatMessage, ...]:
    """Build the independent coordinate-capable Spatial Contract V3 prompt.

    This intentionally does not call the V2 prompt builder: in V3, explicit
    framed geometry is model output rather than trusted-compiler-only state.
    Hidden target truth and media are still excluded from the public context.
    """

    normalized_instruction = _non_empty_text(instruction, "instruction")
    normalized_system_prompt = _non_empty_text(system_prompt, "system_prompt")
    if not isinstance(world_context, PlannerWorldContext):
        raise TypeError("world_context must be a PlannerWorldContext")
    if not isinstance(skill_catalog, SkillCatalog):
        raise TypeError("skill_catalog must be a SkillCatalog")
    if not isinstance(trusted_target_locked, bool):
        raise TypeError("trusted_target_locked must be bool")
    if not isinstance(allow_trusted_safety_completion, bool):
        raise TypeError("allow_trusted_safety_completion must be bool")
    capabilities = (
        SearchRuntimeCapabilities()
        if search_runtime_capabilities is None
        else search_runtime_capabilities
    )
    if not isinstance(capabilities, SearchRuntimeCapabilities):
        raise TypeError(
            "search_runtime_capabilities must be a SearchRuntimeCapabilities or None"
        )
    trusted_mission_id = validate_mission_id(mission_id)
    trusted_uav_id = validate_uav_id(uav_id)
    if (
        isinstance(plan_version, bool)
        or not isinstance(plan_version, int)
        or plan_version <= 0
    ):
        raise ValueError("plan_version must be a positive integer")

    _validate_v3_planner_visible_world_text(world_context)
    projected_limits = _project_dynamic_limits(planner_limits)
    trusted_limits = PlannerLimits(**projected_limits)
    # V2 keeps its single-SEARCH baseline. Spatial V3 bounds repeated SEARCH
    # with the total step budget, reserving TAKEOFF and LAND slots.
    projected_limits["max_search_calls"] = max(
        1,
        int(projected_limits["max_plan_steps"]) - 2,
    )
    model_catalog = initial_planner_catalog(skill_catalog)
    if allow_trusted_safety_completion:
        model_catalog = SkillCatalog(
            tuple(
                contract
                for contract in model_catalog
                if contract.name != "LAND"
            )
        )
    safe_context = {
        "scene_bounds_world_enu_m": {
            "minimum": list(world_context.scene_min_xyz_m),
            "maximum": list(world_context.scene_max_xyz_m),
        },
        "named_search_regions": [
            {"name": name, "description": spec.description}
            for name, spec in sorted(world_context.search_regions.items())
        ],
        "named_landing_zones": [
            {"name": name, "description": spec.description}
            for name, spec in sorted(world_context.landing_zones.items())
        ],
        "named_navigation_points": [
            {"name": name, "description": spec.description}
            for name, spec in sorted(world_context.navigation_points.items())
        ],
        "default_takeoff_altitude_m": world_context.default_takeoff_altitude_m,
        "default_track_duration_s": world_context.default_track_duration_s,
        # V2 historically allowed exactly one SEARCH.  V3 reuses the same
        # trusted budget as an aggregate across every SEARCH timeout so
        # repeated searches do not silently multiply mission time.
        "total_search_time_budget_s": world_context.search_timeout_s,
    }
    payload = {
        "task": (
            "Create one routed Spatial Contract schema-v3 SkillPlanDraft JSON "
            "object with explicit frames and recorded spatial assumptions."
        ),
        "trusted_routing": {
            "schema_version": 3,
            "mission_id": trusted_mission_id,
            "uav_id": trusted_uav_id,
            "plan_version": plan_version,
            "step_uav_id_must_equal": trusted_uav_id,
        },
        "coordinate_frames": {
            "WORLD_ENU": "world origin; +x east, +y north, +z up",
            "HOME_ENU": "home origin; +x east, +y north, +z up",
            "UAV_START_FLU": "mission-start UAV origin; +x forward, +y left, +z up",
            "UAV_HOLD_FLU": "supervisory-hold UAV origin; +x forward, +y left, +z up",
            "CAMERA_FLU": "current camera origin; +x forward, +y left, +z up",
        },
        "spatial_output_policy": {
            "framed_coordinates_allowed": True,
            "bare_spatial_objects_forbidden": True,
            "ambiguous_reference_requires_assumption": True,
            "compiler_must_not_silently_choose_reference_frame": True,
        },
        "mission_completeness_contract": {
            "emit_the_complete_mission_not_a_prefix": True,
            "represent_every_assigned_goal": True,
            "cover_only_goals_assigned_to_this_uav": True,
            "do_not_invent_search_or_track_goals": True,
            "trusted_runtime_safety_completion": (
                allow_trusted_safety_completion
            ),
            "omit_unrequested_return_or_land": (
                allow_trusted_safety_completion
            ),
            "model_land_allowed": not allow_trusted_safety_completion,
            "when_return_and_land_are_requested": (
                "trusted Python owns the bounded home/LAND epilogue; do not "
                "emit LAND"
                if allow_trusted_safety_completion
                else "emit a safe matching return and landing path"
            ),
            "runtime_rule": (
                "When trusted_runtime_safety_completion is true, emit only "
                "the assigned semantic Goals; trusted Python may append a "
                "bounded return/LAND epilogue after Goal coverage is checked."
                if allow_trusted_safety_completion
                else "The model plan itself must contain the complete safe "
                "return and terminal LAND path."
            ),
        },
        "trusted_target_state": {
            "confirmed_target_available": trusted_target_locked,
            "confirmed_target_ref": (
                "$trusted_target.target_id" if trusted_target_locked else None
            ),
            "rule": (
                "TRACK may omit SEARCH only when confirmed_target_available "
                "is true; this flag is trusted runtime state"
            ),
        },
        "runtime_search_capabilities": capabilities.to_prompt_dict(),
        "trusted_world_context": safe_context,
        # The response JSON Schema already carries exact field types, bounds,
        # and enum values.  Repeating every argument paragraph here consumed
        # most of the 4096-token local Qwen context and truncated otherwise
        # valid plans before their final LAND.  Keep the semantic Skill tags,
        # argument names, required flags, concise descriptions, and outputs;
        # trusted Python/schema validation remains authoritative.
        "skill_catalog": _compact_v3_skill_catalog(model_catalog),
        "planner_limits": projected_limits,
        "trusted_planner_policy": _project_dynamic_policy(
            planner_policy,
            limits=trusted_limits,
        ),
        "user_instruction": normalized_instruction,
    }
    return (
        ChatMessage(role="system", content=normalized_system_prompt),
        ChatMessage(
            role="user",
            content=json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )


def _compact_v3_skill_catalog(skill_catalog: SkillCatalog) -> dict[str, object]:
    """Project V3 semantic affordances without duplicating JSON Schema prose."""

    if not isinstance(skill_catalog, SkillCatalog):
        raise TypeError("skill_catalog must be a SkillCatalog")
    return {
        "skills": [
            {
                "name": skill.name,
                "description": skill.description,
                "arguments": [
                    {
                        "name": argument.name,
                        "required": argument.required,
                    }
                    for argument in skill.arguments
                ],
                "outputs": list(skill.outputs),
            }
            for skill in skill_catalog.skills
        ]
    }


def _project_dynamic_limits(value: object) -> dict[str, int | float]:
    if value is None:
        raw: dict[str, object] = dict(_DYNAMIC_LIMIT_DEFAULTS)
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("planner_limits keys must be strings")
        unknown = set(value) - set(_DYNAMIC_LIMIT_DEFAULTS)
        if unknown:
            raise ValueError(
                "planner_limits contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        raw = dict(_DYNAMIC_LIMIT_DEFAULTS)
        raw.update(value)
    else:
        raw = {}
        for name in _DYNAMIC_LIMIT_DEFAULTS:
            try:
                raw[name] = getattr(value, name)
            except AttributeError as exc:
                raise TypeError(
                    "planner_limits object must expose every trusted limit"
                ) from exc

    result: dict[str, int | float] = {}
    for name in _DYNAMIC_LIMIT_DEFAULTS:
        item = raw[name]
        if name in _INTEGER_DYNAMIC_LIMITS:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"planner_limits.{name} must be an integer")
            if item <= 0:
                raise ValueError(f"planner_limits.{name} must be positive")
            result[name] = item
        else:
            if isinstance(item, bool) or not isinstance(item, Real):
                raise TypeError(f"planner_limits.{name} must be a finite number")
            number = float(item)
            if not isfinite(number) or number <= 0.0:
                raise ValueError(
                    f"planner_limits.{name} must be a positive finite number"
                )
            result[name] = number
    if result["max_plan_steps"] > 10:
        raise ValueError("planner_limits.max_plan_steps cannot exceed 10 in v1")
    if result["max_plan_steps"] < 2:
        raise ValueError("planner_limits.max_plan_steps must be at least 2")
    if result["max_search_calls"] != 1:
        raise ValueError("planner_limits.max_search_calls must equal 1 in v1")
    hard_caps = {
        "max_goto_calls": 5,
        "max_track_calls": 2,
        "max_reacquire_attempts_per_track": 2,
        "max_total_reacquire_attempts": 4,
    }
    for name, maximum in hard_caps.items():
        if result[name] > maximum:
            raise ValueError(
                f"planner_limits.{name} cannot exceed {maximum} in v1"
            )
    if (
        result["max_reacquire_attempts_per_track"]
        > result["max_total_reacquire_attempts"]
    ):
        raise ValueError(
            "planner_limits per-TRACK recovery budget cannot exceed total budget"
        )
    if result["min_track_duration_s"] > result["max_track_duration_s"]:
        raise ValueError(
            "planner_limits.min_track_duration_s must not exceed the maximum"
        )
    return result


def _project_dynamic_policy(
    value: object,
    *,
    limits: PlannerLimits,
) -> dict[str, object]:
    """Validate trusted defaults while exposing only model-relevant actions."""

    if value is None:
        policy = PlannerPolicy()
    elif isinstance(value, PlannerPolicy):
        policy = value
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("planner_policy keys must be strings")
        allowed = set(PlannerPolicy.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "planner_policy contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        policy = PlannerPolicy(**dict(value))
    else:
        try:
            policy = PlannerPolicy.from_config(value)
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "planner_policy object must expose every trusted policy field"
            ) from exc
    policy.validate_against(limits)
    return policy.to_prompt_dict()


def _non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _validate_planner_visible_world_text(
    world_context: PlannerWorldContext,
) -> None:
    """Reject hidden-truth/media markers in planner-visible configuration.

    Skill geometry remains hidden by projection, while this guard protects the
    textual fields which are intentionally public.  The exception identifies
    only the structural field, never the rejected value.
    """

    for index, (mapping_name, spec) in enumerate(
        sorted(world_context.search_regions.items())
    ):
        _reject_forbidden_context_text(
            mapping_name,
            f"search_regions key at index {index}",
        )
        _reject_forbidden_context_text(
            spec.name,
            f"search_regions spec name at index {index}",
        )
        _reject_forbidden_context_text(
            spec.description,
            f"search_regions description at index {index}",
        )

    for index, (mapping_name, spec) in enumerate(
        sorted(world_context.landing_zones.items())
    ):
        _reject_forbidden_context_text(
            mapping_name,
            f"landing_zones key at index {index}",
        )
        _reject_forbidden_context_text(
            spec.name,
            f"landing_zones spec name at index {index}",
        )
        _reject_forbidden_context_text(
            spec.description,
            f"landing_zones description at index {index}",
        )


def _validate_v3_planner_visible_world_text(
    world_context: PlannerWorldContext,
) -> None:
    """Reject hidden/media provenance without applying V2's geometry ban."""

    collections = (
        ("search_regions", world_context.search_regions),
        ("landing_zones", world_context.landing_zones),
        ("navigation_points", world_context.navigation_points),
    )
    for collection_name, values in collections:
        for index, (mapping_name, spec) in enumerate(sorted(values.items())):
            for field_name, value in (
                ("key", mapping_name),
                ("name", spec.name),
                ("description", spec.description),
            ):
                _reject_v3_forbidden_context_text(
                    value,
                    f"{collection_name} {field_name} at index {index}",
                )


def _reject_v3_forbidden_context_text(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    nfkc = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", nfkc).casefold()
    words = tuple(_ASCII_WORD.findall(normalized))
    compact = "".join(words)
    unicode_compact = "".join(
        character for character in normalized if character.isalnum()
    )
    if (
        any(word in _FORBIDDEN_CONTEXT_ASCII_TOKENS for word in words)
        or any(marker in compact for marker in _FORBIDDEN_CONTEXT_ASCII_COMPACT)
        or any(marker in unicode_compact for marker in _FORBIDDEN_CONTEXT_UNICODE)
    ):
        raise ValueError(
            f"{field_name} contains a forbidden hidden-state or media marker"
        )


def _reject_forbidden_context_text(value: str, field_name: str) -> None:
    # Shared semantic boundary additionally rejects raw coordinate triples,
    # XYZ assignments, PID/motor/waypoint text, and similar low-level data.
    try:
        reject_forbidden_planner_text(value, field_name)
    except ValueError:
        # Preserve the established planner_v1 error contract; the value itself
        # is intentionally omitted from the exception and ordinary logs.
        raise ValueError(
            f"{field_name} contains a forbidden hidden-state or media marker"
        ) from None
    nfkc = unicodedata.normalize("NFKC", value)
    # Preserve camel-case boundaries before case folding so ``cameraFrame``
    # cannot hide two otherwise obvious forbidden tokens.
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", nfkc).casefold()
    words = tuple(_ASCII_WORD.findall(normalized))
    compact = "".join(words)
    unicode_compact = "".join(
        character for character in normalized if character.isalnum()
    )
    if (
        any(word in _FORBIDDEN_CONTEXT_ASCII_TOKENS for word in words)
        or any(marker in compact for marker in _FORBIDDEN_CONTEXT_ASCII_COMPACT)
        or any(marker in unicode_compact for marker in _FORBIDDEN_CONTEXT_UNICODE)
    ):
        raise ValueError(
            f"{field_name} contains a forbidden hidden-state or media marker"
        )


__all__ = [
    "build_dynamic_skill_planner_messages",
    "build_mission_planner_messages",
    "build_spatial_v3_skill_planner_messages",
]
