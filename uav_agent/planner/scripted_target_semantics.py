"""Conservative target semantics for the model-free dynamic planner.

The scripted planner is intentionally not a general natural-language model.
This module therefore recognizes only a small, auditable search-command
grammar and exact target-category aliases.  Unknown noun phrases are retained
as ``category=unspecified`` so a closed-set detector can reject them instead of
silently searching for a different class.
"""

from __future__ import annotations

import re
from typing import Final

from target.types import TargetSpec


_MAX_SEARCH_TARGET_LENGTH: Final = 256

# These aliases are deliberately finite and exact.  Each value compiles to the
# canonical class name used by the public YOLO alias configuration.  There is
# no substring, edit-distance, embedding, or fuzzy fallback.
_CATEGORY_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "person": (
        "person",
        "pedestrian",
        "human",
        "人",
        "行人",
    ),
    "car": (
        "car",
        "automobile",
        "汽车",
        "轿车",
    ),
    "bicycle": (
        "bicycle",
        "bike",
        "自行车",
    ),
    "motorcycle": (
        "motorcycle",
        "motorbike",
        "摩托车",
    ),
    "bus": (
        "bus",
        "公交车",
        "巴士",
    ),
    "truck": (
        "truck",
        "lorry",
        "卡车",
    ),
}

_ALIAS_TO_CATEGORY: Final[dict[str, str]] = {
    alias.casefold(): category
    for category, aliases in _CATEGORY_ALIASES.items()
    for alias in aliases
}
_ALIASES_LONGEST_FIRST: Final[tuple[str, ...]] = tuple(
    sorted(_ALIAS_TO_CATEGORY, key=len, reverse=True)
)

_ENGLISH_DETERMINER = re.compile(r"^(?:a|an|one|the)\s+", re.IGNORECASE)
_CHINESE_DETERMINERS: Final[tuple[str, ...]] = (
    "一个",
    "一名",
    "一位",
    "那个",
    "那名",
    "那位",
)

_CHINESE_SEARCH = re.compile(
    r"(?:搜寻|寻找|搜索|查找|找到|找)"
    r"(?P<target>.*?)"
    r"(?=，|。|；|,|;|(?:确认|发现|找到)?(?:以后|之后)|跟踪|追踪|然后|最后|返回|降落|$)"
)
_ENGLISH_SEARCH = re.compile(
    r"\b(?:search\s+for|search|look\s+for|find|locate)\s+"
    r"(?P<target>.*?)"
    r"(?=,|;|\.|\bthen\b|\bafter(?:wards)?\b|"
    r"\band\s+(?:confirm|track|follow|return|land)\b|"
    r"\b(?:confirm|track|follow|return|land)\b|$)",
    re.IGNORECASE,
)

_BARE_GENERIC_TARGETS: Final[frozenset[str]] = frozenset(
    {
        "target",
        "a target",
        "one target",
        "the target",
        "目标",
        "一个目标",
        "任务目标",
    }
)
_MOVING_GENERIC_TARGETS: Final[frozenset[str]] = frozenset(
    {
        "moving target",
        "a moving target",
        "one moving target",
        "移动目标",
        "一个移动目标",
        "正在移动的目标",
        "一个正在移动的目标",
        "运动中的目标",
        "一个运动中的目标",
    }
)
_MOVEMENT_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
        "moving",
        "in motion",
        "that is moving",
        "which is moving",
        "移动",
        "正在移动",
        "运动中",
        "正在运动",
    }
)


def _normalized_spaces(value: str) -> str:
    return " ".join(value.strip().split())


def _strip_determiner(value: str) -> str:
    normalized = _ENGLISH_DETERMINER.sub("", value, count=1)
    for determiner in _CHINESE_DETERMINERS:
        if normalized.startswith(determiner):
            return normalized[len(determiner) :].strip()
    return normalized.strip()


def _category_for_exact_alias(value: str) -> str | None:
    return _ALIAS_TO_CATEGORY.get(value.casefold())


def _attribute_tuple(value: str) -> tuple[str, ...]:
    """Canonicalize only explicit movement syntax; preserve everything else."""

    attribute = _normalized_spaces(value).strip(" 的")
    if not attribute:
        return ()
    folded = attribute.casefold()
    if folded in _MOVEMENT_ATTRIBUTES:
        return ("moving",)

    # Recognize movement only as a complete, delimited conjunct.  This avoids
    # treating values such as ``not moving`` or arbitrary substring matches as
    # a positive movement requirement.
    chinese_suffix = re.fullmatch(
        r"(?P<other>.+?)(?:且|并且)(?:正在)?(?:移动|运动中)",
        attribute,
    )
    if chinese_suffix is not None:
        other = chinese_suffix.group("other").strip()
        return tuple(item for item in (other, "moving") if item)
    chinese_prefix = re.fullmatch(
        r"(?:正在)?(?:移动|运动中)(?:且|并且)(?P<other>.+)",
        attribute,
    )
    if chinese_prefix is not None:
        other = chinese_prefix.group("other").strip()
        return tuple(item for item in ("moving", other) if item)
    english_suffix = re.fullmatch(
        r"(?P<other>.+?)\s+and\s+(?:moving|in motion)",
        attribute,
        re.IGNORECASE,
    )
    if english_suffix is not None:
        other = _normalized_spaces(english_suffix.group("other"))
        return tuple(item for item in (other, "moving") if item)
    english_prefix = re.fullmatch(
        r"(?:moving|in motion)\s+and\s+(?P<other>.+)",
        attribute,
        re.IGNORECASE,
    )
    if english_prefix is not None:
        other = _normalized_spaces(english_prefix.group("other"))
        return tuple(item for item in ("moving", other) if item)
    return (attribute,)


def _structured_category(value: str) -> tuple[str, tuple[str, ...]] | None:
    """Parse a whole noun phrase using exact, anchored category aliases."""

    direct = _category_for_exact_alias(value)
    if direct is not None:
        return direct, ()

    folded = value.casefold()
    for alias in _ALIASES_LONGEST_FIRST:
        category = _ALIAS_TO_CATEGORY[alias]

        # English adjective/modifier before the exact category token, such as
        # ``moving person`` or ``red car``.
        suffix = f" {alias}"
        if folded.endswith(suffix):
            attribute = value[: -len(suffix)].strip()
            if attribute:
                return category, _attribute_tuple(attribute)

        # English post-modifier, such as ``person wearing red`` or
        # ``person that is moving``.
        prefix = f"{alias} "
        if folded.startswith(prefix):
            attribute = value[len(prefix) :].strip()
            if attribute:
                return category, _attribute_tuple(attribute)

        # Chinese has no whitespace token boundary. Require an explicit ``的``
        # before an attributed category so unrelated complete words such as
        # ``机器人`` can never match the exact alias ``人`` by suffix alone.
        if any("\u4e00" <= character <= "\u9fff" for character in alias):
            attributed_suffix = f"的{alias}"
            if value.endswith(attributed_suffix) and len(value) > len(
                attributed_suffix
            ):
                attribute = value[: -len(attributed_suffix)].strip()
                if attribute:
                    return category, _attribute_tuple(attribute)
            if value.startswith(alias) and len(value) > len(alias):
                attribute = value[len(alias) :].strip()
                if attribute.casefold() in _MOVEMENT_ATTRIBUTES:
                    return category, ("moving",)
    return None


def compile_scripted_target_description(description: str) -> TargetSpec:
    """Compile one already-isolated target phrase without semantic guessing."""

    if not isinstance(description, str):
        raise TypeError("description must be a string")
    original = _normalized_spaces(description).strip(" \t\r\n\"'“”‘’")
    if not original:
        raise ValueError("description must be non-empty")
    if len(original) > _MAX_SEARCH_TARGET_LENGTH:
        raise ValueError(
            "scripted target description must contain at most "
            f"{_MAX_SEARCH_TARGET_LENGTH} characters"
        )

    folded = original.casefold()
    if folded in _MOVING_GENERIC_TARGETS:
        return TargetSpec(
            original_description=original,
            category="unspecified",
            hard_attributes=("moving",),
            immutable_identity_summary=original,
        )

    noun_phrase = _strip_determiner(original)
    structured = _structured_category(noun_phrase)
    if structured is None:
        category = "unspecified"
        hard_attributes: tuple[str, ...] = ()
    else:
        category, hard_attributes = structured
    return TargetSpec(
        original_description=original,
        category=category,
        hard_attributes=hard_attributes,
        immutable_identity_summary=original,
    )


def target_spec_from_scripted_instruction(instruction: str) -> TargetSpec | None:
    """Return explicit target semantics from a bounded search-command grammar.

    ``None`` means that the instruction supplied no target more specific than
    bare ``target``/``目标``.  In that case a constructor-supplied scripted
    baseline remains authoritative.
    """

    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    for pattern in (_CHINESE_SEARCH, _ENGLISH_SEARCH):
        for match in pattern.finditer(instruction):
            phrase = _normalized_spaces(match.group("target")).strip()
            if not phrase:
                continue
            if phrase.casefold() in _BARE_GENERIC_TARGETS:
                continue
            return compile_scripted_target_description(phrase)
    return None


__all__ = [
    "compile_scripted_target_description",
    "target_spec_from_scripted_instruction",
]
