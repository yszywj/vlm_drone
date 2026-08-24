"""Audited target-category compilation for YOLO and YOLOE.

Ordinary YOLO is a closed-set detector.  This module therefore performs only
exact alias lookup and verifies the selected canonical class against the
*loaded model's* ``model.names``.  It deliberately contains no built-in COCO
class list and no fuzzy or substring matching.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from perception.prompt_types import DeterministicPromptCompiler
from target import TargetSpec
from yolo_service.protocol import TargetQuery


UNSUPPORTED_TARGET_CATEGORY = "UNSUPPORTED_TARGET_CATEGORY"


class ClassAliasConfigError(ValueError):
    """Raised when the auditable alias file is malformed or ambiguous."""


class UnsupportedTargetCategory(ValueError):
    """A target cannot be represented by an ordinary closed-set model."""

    code = UNSUPPORTED_TARGET_CATEGORY

    def __init__(self, category: object, reason: str | None = None) -> None:
        self.category = category
        message = f"{self.code}: target category {category!r} is not supported"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ResolvedClassAlias:
    """One exact, auditable alias resolution aligned to ``model.names``."""

    requested_category: str
    matched_alias: str
    canonical_name: str
    class_id: int
    class_name: str


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Local loader which rejects YAML duplicate keys instead of overwriting."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _exact_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ClassAliasConfigError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ClassAliasConfigError(
            f"{name} must be non-empty and have no surrounding whitespace"
        )
    if len(value) > 256:
        raise ClassAliasConfigError(f"{name} must contain at most 256 characters")
    if any(ord(character) < 32 for character in value):
        raise ClassAliasConfigError(f"{name} contains control characters")
    return value


def _lookup_key(value: str) -> str:
    # Case folding is the sole normalization.  Equality is still exact after
    # that normalization; no tokenization, substring, edit distance, or model
    # inference is permitted at this safety boundary.
    return value.casefold()


def _model_names(value: object) -> dict[int, str]:
    if isinstance(value, Mapping):
        raw_items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_items = enumerate(value)
    else:
        raise TypeError("model_names must be an ID-to-name mapping or name sequence")
    names: dict[int, str] = {}
    for raw_id, raw_name in raw_items:
        if isinstance(raw_id, bool):
            raise ValueError("model_names IDs must be non-negative integers")
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("model_names IDs must be non-negative integers") from exc
        if class_id < 0 or class_id in names:
            raise ValueError("model_names IDs must be unique non-negative integers")
        names[class_id] = _exact_text(raw_name, f"model_names[{class_id}]")
    if not names:
        raise ValueError("model_names must not be empty")
    if len(set(names.values())) != len(names):
        raise ValueError("model_names class names must be unique")
    return names


class ClassAliasMapper:
    """Immutable exact alias map loaded from a strict public YAML file."""

    def __init__(self, aliases: Mapping[str, Sequence[str]]) -> None:
        if not isinstance(aliases, Mapping) or not aliases:
            raise ClassAliasConfigError("class alias root must be a non-empty mapping")
        canonical_to_aliases: dict[str, tuple[str, ...]] = {}
        alias_to_canonical: dict[str, tuple[str, str]] = {}
        for raw_canonical, raw_aliases in aliases.items():
            canonical = _exact_text(raw_canonical, "canonical class name")
            if isinstance(raw_aliases, (str, bytes)) or not isinstance(
                raw_aliases, Sequence
            ):
                raise ClassAliasConfigError(
                    f"aliases for {canonical!r} must be a non-empty sequence"
                )
            if not raw_aliases:
                raise ClassAliasConfigError(
                    f"aliases for {canonical!r} must not be empty"
                )
            if len(raw_aliases) > 128:
                raise ClassAliasConfigError(
                    f"aliases for {canonical!r} exceeds 128 entries"
                )
            audited = tuple(
                _exact_text(item, f"aliases[{canonical!r}]")
                for item in raw_aliases
            )
            # The visible canonical YAML key is itself an audited exact alias.
            searchable = (canonical, *audited)
            local_seen: set[str] = set()
            for alias in searchable:
                key = _lookup_key(alias)
                if key in local_seen:
                    # Repeating the canonical name in ``aliases`` is harmless
                    # and common in human-readable configuration examples.
                    if key == _lookup_key(canonical):
                        continue
                    raise ClassAliasConfigError(
                        f"duplicate alias {alias!r} for {canonical!r}"
                    )
                local_seen.add(key)
                previous = alias_to_canonical.get(key)
                if previous is not None and previous[0] != canonical:
                    raise ClassAliasConfigError(
                        f"alias {alias!r} is ambiguous between "
                        f"{previous[0]!r} and {canonical!r}"
                    )
                alias_to_canonical[key] = (canonical, alias)
            canonical_to_aliases[canonical] = audited
        self._canonical_to_aliases = canonical_to_aliases
        self._alias_to_canonical = alias_to_canonical

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClassAliasMapper":
        config_path = Path(path)
        if not config_path.is_file():
            raise ClassAliasConfigError(
                f"class alias file does not exist: {config_path}"
            )
        try:
            raw: Any = yaml.load(
                config_path.read_text(encoding="utf-8"),
                Loader=_UniqueKeySafeLoader,
            )
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ClassAliasConfigError(
                f"failed to load class alias file {config_path}: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise ClassAliasConfigError("class alias YAML root must be a mapping")
        parsed: dict[str, Sequence[str]] = {}
        for canonical, block in raw.items():
            canonical_name = _exact_text(canonical, "canonical class name")
            if not isinstance(block, Mapping):
                raise ClassAliasConfigError(
                    f"class alias entry {canonical_name!r} must be a mapping"
                )
            if set(block) != {"aliases"}:
                missing = {"aliases"} - set(block)
                unknown = set(block) - {"aliases"}
                details: list[str] = []
                if missing:
                    details.append("missing aliases")
                if unknown:
                    details.append(
                        "unknown fields: " + ", ".join(sorted(map(str, unknown)))
                    )
                raise ClassAliasConfigError(
                    f"class alias entry {canonical_name!r} is invalid: "
                    + "; ".join(details)
                )
            parsed[canonical_name] = block["aliases"]  # type: ignore[assignment]
        return cls(parsed)

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(self._canonical_to_aliases)

    def aliases_for(self, canonical_name: str) -> tuple[str, ...]:
        canonical = _exact_text(canonical_name, "canonical_name")
        try:
            return self._canonical_to_aliases[canonical]
        except KeyError:
            raise UnsupportedTargetCategory(canonical) from None

    def resolve(
        self,
        category: str,
        model_names: Mapping[int, str] | Sequence[str],
    ) -> ResolvedClassAlias:
        requested = _exact_text(category, "target category")
        alias = self._alias_to_canonical.get(_lookup_key(requested))
        if alias is None:
            raise UnsupportedTargetCategory(requested, "no exact configured alias")
        canonical, audited_alias = alias
        names = _model_names(model_names)
        matches = [
            (class_id, class_name)
            for class_id, class_name in names.items()
            if class_name == canonical
        ]
        if not matches:
            raise UnsupportedTargetCategory(
                requested,
                f"canonical class {canonical!r} is absent from loaded model.names",
            )
        class_id, class_name = matches[0]
        return ResolvedClassAlias(
            requested_category=requested,
            matched_alias=audited_alias,
            canonical_name=canonical,
            class_id=class_id,
            class_name=class_name,
        )

    def category_is_exact_alias(self, value: str, canonical_name: str) -> bool:
        """Return whether *value* is an exact alias for one canonical class."""

        text = _exact_text(value, "category value")
        matched = self._alias_to_canonical.get(_lookup_key(text))
        return matched is not None and matched[0] == canonical_name

    def compile_target_query(
        self,
        target_spec: TargetSpec,
        model_family: str,
        model_names: Mapping[int, str] | Sequence[str],
        *,
        prompt_compiler: DeterministicPromptCompiler | None = None,
    ) -> TargetQuery:
        """Convenience form of the module-level compiler using this map."""

        return compile_target_query(
            target_spec,
            model_family,
            model_names,
            self,
            prompt_compiler=prompt_compiler,
        )


def compile_target_query(
    target_spec: TargetSpec,
    model_family: str,
    model_names: Mapping[int, str] | Sequence[str],
    mapper: ClassAliasMapper,
    *,
    prompt_compiler: DeterministicPromptCompiler | None = None,
) -> TargetQuery:
    """Compile one TargetSpec for ordinary YOLO or dynamic-prompt YOLOE."""

    if not isinstance(target_spec, TargetSpec):
        raise TypeError("target_spec must be a TargetSpec")
    if not isinstance(mapper, ClassAliasMapper):
        raise TypeError("mapper must be a ClassAliasMapper")
    if model_family == "yolo":
        resolved = mapper.resolve(target_spec.category, model_names)
        return TargetQuery(class_ids=(resolved.class_id,), text_prompts=())
    if model_family == "yoloe":
        # Validate the runtime model metadata even though open-vocabulary
        # prompts do not select one of its preloaded class IDs.
        _model_names(model_names)
        compiler = prompt_compiler or DeterministicPromptCompiler()
        if not isinstance(compiler, DeterministicPromptCompiler):
            raise TypeError("prompt_compiler must be DeterministicPromptCompiler")
        bundle = compiler.compile(target_spec)
        prompts = tuple(dict.fromkeys((*bundle.positive_phrases, *bundle.fallback_phrases)))
        return TargetQuery(class_ids=(), text_prompts=prompts)
    raise ValueError("model_family must be exactly 'yolo' or 'yoloe'")


__all__ = [
    "ClassAliasConfigError",
    "ClassAliasMapper",
    "ResolvedClassAlias",
    "UNSUPPORTED_TARGET_CATEGORY",
    "UnsupportedTargetCategory",
    "compile_target_query",
]
