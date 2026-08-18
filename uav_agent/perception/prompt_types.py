"""Stable prompt boundary for current and future visual grounding backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from target import TargetSpec


_MAX_PROMPT_PHRASES = 32
_MAX_PROMPT_TEXT_LENGTH = 512


def _phrase(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > _MAX_PROMPT_TEXT_LENGTH:
        raise ValueError(
            f"{name} must contain at most {_MAX_PROMPT_TEXT_LENGTH} characters"
        )
    return normalized


def _phrases(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if len(value) > _MAX_PROMPT_PHRASES:
        raise ValueError(f"{name} must contain at most {_MAX_PROMPT_PHRASES} items")
    normalized = tuple(
        _phrase(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class PromptBundle:
    """Pure prompt phrases and opaque frame handles; never image arrays."""

    positive_phrases: tuple[str, ...]
    fallback_phrases: tuple[str, ...]
    negative_phrases: tuple[str, ...]
    reference_handles: tuple[str, ...]
    immutable_target_identity: str

    def __post_init__(self) -> None:
        for name in (
            "positive_phrases",
            "fallback_phrases",
            "negative_phrases",
            "reference_handles",
        ):
            object.__setattr__(self, name, _phrases(getattr(self, name), name))
        object.__setattr__(
            self,
            "immutable_target_identity",
            _phrase(
                self.immutable_target_identity,
                "immutable_target_identity",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "positive_phrases": list(self.positive_phrases),
            "fallback_phrases": list(self.fallback_phrases),
            "negative_phrases": list(self.negative_phrases),
            "reference_handles": list(self.reference_handles),
            "immutable_target_identity": self.immutable_target_identity,
        }


@runtime_checkable
class TargetPromptAdapter(Protocol):
    """Future learned adapters must preserve this pure value boundary."""

    def compile(
        self,
        target_spec: TargetSpec,
        *,
        reference_handles: Sequence[str] = (),
    ) -> PromptBundle: ...


class DeterministicPromptCompiler:
    """Deterministically project TargetSpec v2 without neural state."""

    def compile(
        self,
        target_spec: TargetSpec,
        *,
        reference_handles: Sequence[str] = (),
    ) -> PromptBundle:
        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        positives = _deduplicate(
            (
                target_spec.original_description,
                target_spec.immutable_identity_summary,
                *target_spec.hard_attributes,
                *target_spec.query_ladder,
            )
        )
        fallbacks = _deduplicate(
            (
                target_spec.category,
                *target_spec.soft_attributes,
            )
        )
        return PromptBundle(
            positive_phrases=positives,
            fallback_phrases=fallbacks,
            negative_phrases=_deduplicate(target_spec.negative_constraints),
            reference_handles=tuple(reference_handles),
            immutable_target_identity=target_spec.immutable_identity_summary,
        )


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _phrase(value, "prompt phrase")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


__all__ = [
    "DeterministicPromptCompiler",
    "PromptBundle",
    "TargetPromptAdapter",
]
