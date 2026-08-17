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
from planner.schemas import PlannerWorldContext
from planner.skill_catalog import SkillCatalog
from planner.text_safety import reject_forbidden_planner_text


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
    payload = {
        "task": "Create one constrained SkillPlanDraft JSON object.",
        "trusted_world_context": safe_context,
        "skill_catalog": skill_catalog.to_prompt_dict(),
        "planner_limits": _project_dynamic_limits(planner_limits),
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
]
