"""Deterministic Chinese instruction rendering for Planner dataset v1.

The renderer is deliberately Gold-first: it receives an already validated
``GoldPlannerSpec`` and only verbalizes those semantics.  It never invents a
label, calls a model, or sees target instance state.  Split-specific language
pools keep held-out aliases and prompt-injection cases out of ordinary data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import cos, isfinite, pi, sin
from pathlib import Path
import random
import re
import string
from types import MappingProxyType
from typing import Final
import unicodedata

import yaml

from planner.schemas import (
    LandingZoneSpec,
    PlannerWorldContext,
    SearchRegionSpec,
)
from tasks.schemas import GoldPlannerSpec, PlannerWorldCase
from tasks.target_ontology import TargetOntology


RESOURCE_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[1] / "resources" / "planner_v1"
)
DEFAULT_LEXICON_PATH: Final[Path] = RESOURCE_ROOT / "language_lexicon_zh.yaml"
DEFAULT_WORLD_CONTEXTS_PATH: Final[Path] = RESOURCE_ROOT / "world_contexts.yaml"
DEFAULT_TEMPLATES_PATH: Final[Path] = RESOURCE_ROOT / "templates_zh.yaml"

DATASET_SPLITS: Final[tuple[str, ...]] = (
    "train",
    "validation",
    "test_iid",
    "test_compositional",
    "test_language",
    "test_robustness",
)
_SPLIT_SET = frozenset(DATASET_SPLITS)
_LANGUAGE_SPLIT = "test_language"
_ROBUSTNESS_SPLIT = "test_robustness"
_MODES = frozenset({"explicit", "default"})
_PLACEHOLDERS = frozenset(
    {
        "prefix",
        "region",
        "target",
        "duration",
        "landing",
        "altitude",
        "suffix",
        "injection",
    }
)
_CORE_PLACEHOLDERS = frozenset({"region", "target", "duration", "landing"})


class RendererConfigError(ValueError):
    """Raised when language resources or a render request are inconsistent."""


class StrictYamlError(ValueError):
    """Raised when a Planner resource is unreadable or has duplicate keys."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects YAML's silent last-key-wins behavior."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
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


def load_yaml_strict(
    path: str | Path,
    resource_name: str = "Planner YAML resource",
) -> object:
    """Load safe YAML while rejecting duplicate mapping keys at every depth."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise StrictYamlError(f"could not read {resource_name}") from exc
    try:
        return yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise StrictYamlError(f"invalid {resource_name}: {exc}") from exc


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise RendererConfigError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise RendererConfigError(f"{field_name} must be non-empty")
    return normalized


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RendererConfigError(f"{field_name} must be a list")
    return value


def _mapping(value: object, field_name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise RendererConfigError(f"{field_name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[object, object],
    expected: set[str],
    field_name: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise RendererConfigError(f"{field_name} keys must be strings")
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise RendererConfigError(
            f"{field_name} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise RendererConfigError(
            f"{field_name} contains unknown fields: {', '.join(sorted(unknown))}"
        )


def _normalized_alias(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@dataclass(frozen=True, slots=True)
class AliasPool:
    """Disjoint ordinary-training and held-out-language aliases."""

    train_aliases: tuple[str, ...]
    heldout_aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        train = tuple(_text(value, "train alias") for value in self.train_aliases)
        heldout = tuple(
            _text(value, "heldout alias") for value in self.heldout_aliases
        )
        if not train:
            raise RendererConfigError("train_aliases must not be empty")
        if not heldout:
            raise RendererConfigError("heldout_aliases must not be empty")
        train_keys = tuple(_normalized_alias(value) for value in train)
        heldout_keys = tuple(_normalized_alias(value) for value in heldout)
        if len(set(train_keys)) != len(train_keys):
            raise RendererConfigError("train_aliases contains a duplicate")
        if len(set(heldout_keys)) != len(heldout_keys):
            raise RendererConfigError("heldout_aliases contains a duplicate")
        overlap = set(train_keys) & set(heldout_keys)
        if overlap:
            raise RendererConfigError(
                "train_aliases and heldout_aliases must be disjoint"
            )
        object.__setattr__(self, "train_aliases", train)
        object.__setattr__(self, "heldout_aliases", heldout)

    def for_split(self, split: str) -> tuple[str, ...]:
        _validate_split(split)
        if split == _LANGUAGE_SPLIT:
            return self.heldout_aliases
        return self.train_aliases


def _parse_alias_pool(value: object, field_name: str) -> AliasPool:
    raw = _mapping(value, field_name)
    _exact_keys(raw, {"train_aliases", "heldout_aliases"}, field_name)
    return AliasPool(
        train_aliases=tuple(
            _text(item, f"{field_name}.train_aliases")
            for item in _sequence(raw["train_aliases"], f"{field_name}.train_aliases")
        ),
        heldout_aliases=tuple(
            _text(item, f"{field_name}.heldout_aliases")
            for item in _sequence(
                raw["heldout_aliases"], f"{field_name}.heldout_aliases"
            )
        ),
    )


def _parse_named_aliases(
    value: object,
    field_name: str,
) -> Mapping[str, AliasPool]:
    raw = _mapping(value, field_name)
    if not raw:
        raise RendererConfigError(f"{field_name} must not be empty")
    result: dict[str, AliasPool] = {}
    seen_aliases: dict[str, str] = {}
    for raw_name, raw_pool in raw.items():
        name = _text(raw_name, f"{field_name} name")
        if name in result:
            raise RendererConfigError(f"duplicate {field_name} name {name!r}")
        pool = _parse_alias_pool(raw_pool, f"{field_name}.{name}")
        for alias in (*pool.train_aliases, *pool.heldout_aliases):
            key = _normalized_alias(alias)
            previous = seen_aliases.get(key)
            if previous is not None and previous != name:
                raise RendererConfigError(
                    f"alias {alias!r} maps to both {previous!r} and {name!r}"
                )
            seen_aliases[key] = name
        result[name] = pool
    return MappingProxyType(result)


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise RendererConfigError(f"{field_name} must be a positive number")
    try:
        number = float(value)
    except ValueError as exc:
        raise RendererConfigError(f"{field_name} must be a positive number") from exc
    if not isfinite(number) or number <= 0.0:
        raise RendererConfigError(f"{field_name} must be a positive finite number")
    return number


def _parse_numeric_aliases(
    value: object,
    field_name: str,
) -> Mapping[float, AliasPool]:
    raw = _mapping(value, field_name)
    if not raw:
        raise RendererConfigError(f"{field_name} must not be empty")
    result: dict[float, AliasPool] = {}
    seen_aliases: dict[str, float] = {}
    for raw_number, raw_pool in raw.items():
        number = _positive_float(raw_number, f"{field_name} key")
        if number in result:
            raise RendererConfigError(f"duplicate numeric key {number} in {field_name}")
        pool = _parse_alias_pool(raw_pool, f"{field_name}.{raw_number}")
        for alias in (*pool.train_aliases, *pool.heldout_aliases):
            key = _normalized_alias(alias)
            previous = seen_aliases.get(key)
            if previous is not None and previous != number:
                raise RendererConfigError(
                    f"alias {alias!r} maps to both {previous:g} and {number:g} "
                    f"in {field_name}"
                )
            seen_aliases[key] = number
        result[number] = pool
    return MappingProxyType(result)


def _validate_semantic_alias_registry(
    *,
    search_regions: Mapping[str, AliasPool],
    landing_zones: Mapping[str, AliasPool],
    durations: Mapping[float, AliasPool],
    default_duration: AliasPool,
    altitudes: Mapping[float, AliasPool],
) -> None:
    registry: dict[str, tuple[str, object]] = {}

    def register(text: str, semantic: tuple[str, object]) -> None:
        key = _normalized_alias(text)
        previous = registry.get(key)
        if previous is not None and previous != semantic:
            raise RendererConfigError(
                f"normalized alias {text!r} is ambiguous between "
                f"{previous!r} and {semantic!r}"
            )
        registry[key] = semantic

    for kind, named in (
        ("search_region", search_regions),
        ("landing_zone", landing_zones),
    ):
        for name, pool in named.items():
            semantic = (kind, name)
            register(name, semantic)
            for alias in (*pool.train_aliases, *pool.heldout_aliases):
                register(alias, semantic)
    for kind, numeric in (("duration", durations), ("altitude", altitudes)):
        for number, pool in numeric.items():
            semantic = (kind, number)
            for alias in (*pool.train_aliases, *pool.heldout_aliases):
                register(alias, semantic)
    for alias in (
        *default_duration.train_aliases,
        *default_duration.heldout_aliases,
    ):
        register(alias, ("duration_default", None))


@dataclass(frozen=True, slots=True)
class LanguageLexicon:
    schema_version: str
    language: str
    search_regions: Mapping[str, AliasPool]
    landing_zones: Mapping[str, AliasPool]
    duration_expressions: Mapping[float, AliasPool]
    default_duration_expressions: AliasPool
    altitude_expressions: Mapping[float, AliasPool]
    polite_prefixes: AliasPool
    neutral_suffixes: AliasPool
    robustness_injections: tuple["RobustnessInjection", ...]


@dataclass(frozen=True, slots=True)
class RobustnessInjection:
    """One audited adversarial-language category and its literal text."""

    category: str
    text: str

    def __post_init__(self) -> None:
        category = _text(self.category, "robustness category")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", category):
            raise RendererConfigError(
                "robustness category must use lowercase snake_case"
            )
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "text", _text(self.text, "robustness text"))


def _read_yaml(path: str | Path, resource_name: str) -> object:
    try:
        return load_yaml_strict(path, resource_name)
    except StrictYamlError as exc:
        raise RendererConfigError(str(exc)) from exc


def load_language_lexicon(
    path: str | Path = DEFAULT_LEXICON_PATH,
) -> LanguageLexicon:
    raw = _mapping(_read_yaml(path, "language lexicon"), "language lexicon")
    expected = {
        "schema_version",
        "language",
        "search_regions",
        "landing_zones",
        "duration_expressions",
        "default_duration_expressions",
        "altitude_expressions",
        "polite_prefixes",
        "neutral_suffixes",
        "robustness_injections",
    }
    _exact_keys(raw, expected, "language lexicon")
    injections_list: list[RobustnessInjection] = []
    for index, value in enumerate(
        _sequence(raw["robustness_injections"], "robustness_injections")
    ):
        entry = _mapping(value, f"robustness_injections[{index}]")
        _exact_keys(
            entry,
            {"category", "text"},
            f"robustness_injections[{index}]",
        )
        injections_list.append(
            RobustnessInjection(category=entry["category"], text=entry["text"])
        )
    injections = tuple(injections_list)
    if not injections:
        raise RendererConfigError("robustness_injections must not be empty")
    if len({_normalized_alias(item.text) for item in injections}) != len(injections):
        raise RendererConfigError("robustness_injections contains a duplicate")
    required_robustness_categories = {
        "prompt_injection",
        "extra_field",
        "format_interference",
        "irrelevant_text",
        "long_instruction",
        "repeated_requirement",
    }
    categories = {item.category for item in injections}
    missing_categories = required_robustness_categories - categories
    if missing_categories:
        raise RendererConfigError(
            "robustness_injections is missing categories: "
            + ", ".join(sorted(missing_categories))
        )
    search_regions = _parse_named_aliases(raw["search_regions"], "search_regions")
    landing_zones = _parse_named_aliases(raw["landing_zones"], "landing_zones")
    durations = _parse_numeric_aliases(
        raw["duration_expressions"], "duration_expressions"
    )
    default_duration = _parse_alias_pool(
        raw["default_duration_expressions"], "default_duration_expressions"
    )
    altitudes = _parse_numeric_aliases(
        raw["altitude_expressions"], "altitude_expressions"
    )
    _validate_semantic_alias_registry(
        search_regions=search_regions,
        landing_zones=landing_zones,
        durations=durations,
        default_duration=default_duration,
        altitudes=altitudes,
    )
    return LanguageLexicon(
        schema_version=_text(raw["schema_version"], "schema_version"),
        language=_text(raw["language"], "language"),
        search_regions=search_regions,
        landing_zones=landing_zones,
        duration_expressions=durations,
        default_duration_expressions=default_duration,
        altitude_expressions=altitudes,
        polite_prefixes=_parse_alias_pool(raw["polite_prefixes"], "polite_prefixes"),
        neutral_suffixes=_parse_alias_pool(
            raw["neutral_suffixes"], "neutral_suffixes"
        ),
        robustness_injections=injections,
    )


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    template_id: str
    template_family: str
    paraphrase_family: str
    splits: tuple[str, ...]
    duration_mode: str
    altitude_mode: str
    difficulty: str
    text: str
    robustness: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "template_id",
            "template_family",
            "paraphrase_family",
            "duration_mode",
            "altitude_mode",
            "difficulty",
            "text",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        splits = tuple(_text(split, "template split") for split in self.splits)
        if not splits or len(set(splits)) != len(splits):
            raise RendererConfigError("template splits must be non-empty and unique")
        unknown_splits = set(splits) - _SPLIT_SET
        if unknown_splits:
            raise RendererConfigError(
                "template contains unknown splits: " + ", ".join(sorted(unknown_splits))
            )
        if self.duration_mode not in _MODES:
            raise RendererConfigError("duration_mode must be explicit or default")
        if self.altitude_mode not in _MODES:
            raise RendererConfigError("altitude_mode must be explicit or default")
        if not isinstance(self.robustness, bool):
            raise RendererConfigError("robustness must be boolean")

        fields: set[str] = set()
        for _, field_name, format_spec, conversion in string.Formatter().parse(self.text):
            if field_name is None:
                continue
            if format_spec or conversion:
                raise RendererConfigError("template placeholders cannot use formatting")
            fields.add(field_name)
        unknown_fields = fields - _PLACEHOLDERS
        if unknown_fields:
            raise RendererConfigError(
                "template contains unknown placeholders: "
                + ", ".join(sorted(unknown_fields))
            )
        missing_core = _CORE_PLACEHOLDERS - fields
        if missing_core:
            raise RendererConfigError(
                "template omits task semantics: " + ", ".join(sorted(missing_core))
            )
        if ("altitude" in fields) != (self.altitude_mode == "explicit"):
            raise RendererConfigError(
                "altitude placeholder must exactly match explicit altitude_mode"
            )
        if self.robustness:
            if splits != (_ROBUSTNESS_SPLIT,) or "injection" not in fields:
                raise RendererConfigError(
                    "robustness templates must be test_robustness-only and use injection"
                )
        elif _ROBUSTNESS_SPLIT in splits or "injection" in fields:
            raise RendererConfigError(
                "ordinary templates cannot use the robustness split or injection"
            )
        object.__setattr__(self, "splits", splits)


@dataclass(frozen=True, slots=True)
class TemplateCatalog:
    schema_version: str
    language: str
    templates: tuple[TemplateSpec, ...]

    def for_split(
        self,
        split: str,
        *,
        duration_mode: str,
        altitude_mode: str,
    ) -> tuple[TemplateSpec, ...]:
        _validate_split(split)
        if duration_mode not in _MODES or altitude_mode not in _MODES:
            raise RendererConfigError("render modes must be explicit or default")
        return tuple(
            template
            for template in self.templates
            if split in template.splits
            and template.duration_mode == duration_mode
            and template.altitude_mode == altitude_mode
        )


def load_template_catalog(
    path: str | Path = DEFAULT_TEMPLATES_PATH,
) -> TemplateCatalog:
    raw = _mapping(_read_yaml(path, "template catalog"), "template catalog")
    _exact_keys(raw, {"schema_version", "language", "templates"}, "template catalog")
    parsed: list[TemplateSpec] = []
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(_sequence(raw["templates"], "templates")):
        entry = _mapping(raw_entry, f"templates[{index}]")
        required = {
            "template_id",
            "template_family",
            "paraphrase_family",
            "splits",
            "duration_mode",
            "altitude_mode",
            "difficulty",
            "text",
        }
        allowed = required | {"robustness"}
        if any(not isinstance(key, str) for key in entry):
            raise RendererConfigError(f"templates[{index}] keys must be strings")
        missing = required - set(entry)
        unknown = set(entry) - allowed
        if missing:
            raise RendererConfigError(
                f"templates[{index}] is missing fields: {', '.join(sorted(missing))}"
            )
        if unknown:
            raise RendererConfigError(
                f"templates[{index}] contains unknown fields: {', '.join(sorted(unknown))}"
            )
        template = TemplateSpec(
            template_id=entry["template_id"],
            template_family=entry["template_family"],
            paraphrase_family=entry["paraphrase_family"],
            splits=tuple(_sequence(entry["splits"], f"templates[{index}].splits")),
            duration_mode=entry["duration_mode"],
            altitude_mode=entry["altitude_mode"],
            difficulty=entry["difficulty"],
            text=entry["text"],
            robustness=entry.get("robustness", False),
        )
        if template.template_id in seen_ids:
            raise RendererConfigError(
                f"duplicate template_id {template.template_id!r}"
            )
        seen_ids.add(template.template_id)
        parsed.append(template)
    if not parsed:
        raise RendererConfigError("template catalog must not be empty")
    ordinary_splits = {
        "train",
        "validation",
        "test_iid",
        "test_compositional",
    }
    family_owner: dict[str, str] = {}
    skeleton_owner: dict[str, str] = {}
    for template in parsed:
        owners = ordinary_splits & set(template.splits)
        if not owners:
            continue
        if len(owners) != 1 or len(template.splits) != 1:
            raise RendererConfigError(
                "ordinary templates must belong to exactly one split"
            )
        owner = next(iter(owners))
        previous_family_owner = family_owner.get(template.template_family)
        if previous_family_owner is not None and previous_family_owner != owner:
            raise RendererConfigError(
                f"template family {template.template_family!r} crosses ordinary splits"
            )
        family_owner[template.template_family] = owner
        skeleton = _normalized_alias(template.text)
        previous_skeleton_owner = skeleton_owner.get(skeleton)
        if previous_skeleton_owner is not None and previous_skeleton_owner != owner:
            raise RendererConfigError(
                "an exact template skeleton crosses ordinary splits"
            )
        skeleton_owner[skeleton] = owner
    catalog = TemplateCatalog(
        schema_version=_text(raw["schema_version"], "schema_version"),
        language=_text(raw["language"], "language"),
        templates=tuple(parsed),
    )
    # Every split must support all four explicit/default combinations.  This is
    # what lets the generator sample Gold first without silently changing it.
    for split in DATASET_SPLITS:
        for duration_mode in _MODES:
            for altitude_mode in _MODES:
                if not catalog.for_split(
                    split,
                    duration_mode=duration_mode,
                    altitude_mode=altitude_mode,
                ):
                    raise RendererConfigError(
                        f"no template for {split}/{duration_mode}/{altitude_mode}"
                    )
    return catalog


def load_world_cases(
    path: str | Path = DEFAULT_WORLD_CONTEXTS_PATH,
) -> Mapping[str, PlannerWorldCase]:
    raw = _mapping(_read_yaml(path, "world contexts"), "world contexts")
    _exact_keys(raw, {"schema_version", "worlds"}, "world contexts")
    _text(raw["schema_version"], "schema_version")
    result: dict[str, PlannerWorldCase] = {}
    for index, raw_world in enumerate(_sequence(raw["worlds"], "worlds")):
        try:
            world = PlannerWorldCase.from_dict(_mapping(raw_world, f"worlds[{index}]"))
        except (TypeError, ValueError) as exc:
            raise RendererConfigError(f"invalid worlds[{index}]: {exc}") from exc
        if world.context_id in result:
            raise RendererConfigError(f"duplicate context_id {world.context_id!r}")
        if len(world.search_regions) < 4:
            raise RendererConfigError(
                f"world {world.context_id!r} must expose at least four search regions"
            )
        if len(world.landing_zones) < 3:
            raise RendererConfigError(
                f"world {world.context_id!r} must expose at least three landing zones"
            )
        result[world.context_id] = world
    if not result:
        raise RendererConfigError("world contexts must contain at least one world")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class RenderedInstruction:
    instruction: str
    template_family: str
    paraphrase_family: str
    difficulty: str
    template_id: str
    aliases: Mapping[str, str]
    robustness_category: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "instruction",
            "template_family",
            "paraphrase_family",
            "difficulty",
            "template_id",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        if not isinstance(self.aliases, Mapping):
            raise RendererConfigError("aliases must be a mapping")
        aliases = {
            _text(key, "alias role"): _text(value, f"aliases[{key!r}]")
            for key, value in self.aliases.items()
        }
        object.__setattr__(self, "aliases", MappingProxyType(aliases))
        has_injection = "robustness_injection" in aliases
        if self.robustness_category is None:
            if has_injection:
                raise RendererConfigError(
                    "robustness injection requires robustness_category"
                )
        else:
            category = _text(self.robustness_category, "robustness_category")
            if not re.fullmatch(r"[a-z][a-z0-9_]*", category):
                raise RendererConfigError(
                    "robustness_category must use lowercase snake_case"
                )
            if not has_injection:
                raise RendererConfigError(
                    "robustness_category requires robustness injection text"
                )
            object.__setattr__(self, "robustness_category", category)

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "template_family": self.template_family,
            "paraphrase_family": self.paraphrase_family,
            "difficulty": self.difficulty,
            "template_id": self.template_id,
            "aliases": dict(self.aliases),
            "robustness_category": self.robustness_category,
        }

    @property
    def aliases_used(self) -> tuple[str, ...]:
        """Stable alias values for language-feature metadata."""

        return tuple(self.aliases[key] for key in sorted(self.aliases))


def _validate_split(split: object) -> str:
    normalized = _text(split, "split")
    if normalized not in _SPLIT_SET:
        raise RendererConfigError(f"unknown dataset split {normalized!r}")
    return normalized


def _stable_random(
    *,
    seed: int,
    gold: GoldPlannerSpec,
    world: PlannerWorldCase,
    split: str,
    variant_index: int,
) -> random.Random:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(variant_index, bool) or not isinstance(variant_index, int):
        raise TypeError("variant_index must be an integer")
    if variant_index < 0:
        raise ValueError("variant_index must be non-negative")
    material = "\x1f".join(
        (
            str(seed),
            gold.spec_id,
            world.context_id,
            split,
            str(variant_index),
        )
    ).encode("utf-8")
    return random.Random(int.from_bytes(sha256(material).digest()[:16], "big"))


def _numeric_pool(
    pools: Mapping[float, AliasPool],
    value: float,
    field_name: str,
) -> AliasPool:
    for configured, pool in pools.items():
        if abs(configured - value) <= 1e-9:
            return pool
    raise RendererConfigError(
        f"{field_name} {value:g} has no approved language expression"
    )


class InstructionRenderer:
    """Render one immutable Gold task without changing any semantic field."""

    def __init__(
        self,
        ontology: TargetOntology | LanguageLexicon,
        lexicon: LanguageLexicon | TemplateCatalog | None = None,
        templates: TemplateCatalog | TargetOntology | None = None,
    ) -> None:
        # Prefer ``(ontology, lexicon?, templates?)``.  The explicit
        # ``(lexicon, templates, ontology)`` form is retained for the generator
        # while the module boundary is being introduced.
        if isinstance(ontology, LanguageLexicon):
            supplied_lexicon = ontology
            supplied_templates = lexicon
            supplied_ontology = templates
            if not isinstance(supplied_templates, TemplateCatalog) or not isinstance(
                supplied_ontology, TargetOntology
            ):
                raise TypeError(
                    "legacy constructor order requires "
                    "(LanguageLexicon, TemplateCatalog, TargetOntology)"
                )
            ontology = supplied_ontology
            lexicon = supplied_lexicon
            templates = supplied_templates
        if not isinstance(ontology, TargetOntology):
            raise TypeError("ontology must be a TargetOntology")
        if lexicon is not None and not isinstance(lexicon, LanguageLexicon):
            raise TypeError("lexicon must be a LanguageLexicon")
        if templates is not None and not isinstance(templates, TemplateCatalog):
            raise TypeError("templates must be a TemplateCatalog")
        self._ontology = ontology
        self._lexicon = lexicon or load_language_lexicon()
        self._templates = templates or load_template_catalog()
        if self._lexicon.language != self._templates.language:
            raise RendererConfigError("lexicon and template languages differ")

    @property
    def lexicon(self) -> LanguageLexicon:
        return self._lexicon

    @property
    def templates(self) -> TemplateCatalog:
        return self._templates

    def render(
        self,
        gold: GoldPlannerSpec,
        world: PlannerWorldCase,
        *,
        split: str,
        seed: int,
        variant_index: int = 0,
    ) -> RenderedInstruction:
        if not isinstance(gold, GoldPlannerSpec):
            raise TypeError("gold must be a GoldPlannerSpec")
        if not isinstance(world, PlannerWorldCase):
            raise TypeError("world must be a PlannerWorldCase")
        split = _validate_split(split)
        self._ontology.validate_gold_spec(gold)
        if gold.search_region not in world.search_regions:
            raise RendererConfigError(
                f"Gold search region {gold.search_region!r} is absent from world"
            )
        if gold.landing_zone not in world.landing_zones:
            raise RendererConfigError(
                f"Gold landing zone {gold.landing_zone!r} is absent from world"
            )
        if gold.search_region not in self._lexicon.search_regions:
            raise RendererConfigError(
                f"no aliases for search region {gold.search_region!r}"
            )
        if gold.landing_zone not in self._lexicon.landing_zones:
            raise RendererConfigError(
                f"no aliases for landing zone {gold.landing_zone!r}"
            )

        duration_mode = (
            "explicit" if "track_duration_s" in gold.explicit_fields else "default"
        )
        altitude_mode = (
            "explicit" if "takeoff_altitude_m" in gold.explicit_fields else "default"
        )
        if duration_mode == "default" and abs(
            gold.track_duration_s - world.default_track_duration_s
        ) > 1e-9:
            raise RendererConfigError(
                "an implicit track duration must equal the world default"
            )
        if altitude_mode == "explicit" and gold.takeoff_altitude_m is None:
            raise RendererConfigError("explicit takeoff altitude cannot be null")
        if altitude_mode == "default" and gold.takeoff_altitude_m is not None:
            raise RendererConfigError("implicit takeoff altitude must be null")

        candidates = self._templates.for_split(
            split,
            duration_mode=duration_mode,
            altitude_mode=altitude_mode,
        )
        if not candidates:
            raise RendererConfigError(
                f"no template can express {split}/{duration_mode}/{altitude_mode}"
            )
        rng = _stable_random(
            seed=seed,
            gold=gold,
            world=world,
            split=split,
            variant_index=variant_index,
        )
        template = rng.choice(candidates)

        region = rng.choice(
            self._lexicon.search_regions[gold.search_region].for_split(split)
        )
        landing = rng.choice(
            self._lexicon.landing_zones[gold.landing_zone].for_split(split)
        )
        concept = self._ontology.require_concept(gold.target_concept_id)
        ontology_aliases = self._ontology.aliases_for(gold.target_concept_id)
        if split == _LANGUAGE_SPLIT and ontology_aliases:
            # The final ontology alias is a strict language holdout.  A concept
            # with only one alias therefore uses canonical text in every
            # ordinary split and reserves that sole alias for test_language.
            target_choices = ontology_aliases[-1:]
        else:
            ordinary_aliases = (
                ontology_aliases[:-1] if len(ontology_aliases) >= 2 else ()
            )
            target_choices = (concept.canonical_description, *ordinary_aliases)
        target = rng.choice(target_choices)

        if duration_mode == "explicit":
            duration_pool = _numeric_pool(
                self._lexicon.duration_expressions,
                gold.track_duration_s,
                "track_duration_s",
            )
        else:
            duration_pool = self._lexicon.default_duration_expressions
        duration = rng.choice(duration_pool.for_split(split))

        altitude = ""
        if altitude_mode == "explicit":
            assert gold.takeoff_altitude_m is not None
            altitude_pool = _numeric_pool(
                self._lexicon.altitude_expressions,
                gold.takeoff_altitude_m,
                "takeoff_altitude_m",
            )
            altitude = rng.choice(altitude_pool.for_split(split))

        # Low-rate neutral language noise creates variety without changing any
        # task field.  It is derived from the per-sample RNG and is reproducible.
        prefix = ""
        if rng.random() < 0.55:
            prefix = rng.choice(self._lexicon.polite_prefixes.for_split(split))
        suffix = ""
        if rng.random() < 0.35:
            suffix = rng.choice(self._lexicon.neutral_suffixes.for_split(split)) + "。"
        injection = ""
        robustness_category: str | None = None
        if split == _ROBUSTNESS_SPLIT:
            selected_injection = rng.choice(self._lexicon.robustness_injections)
            injection = selected_injection.text
            robustness_category = selected_injection.category

        aliases = {
            "target": target,
            "search_region": region,
            "track_duration": duration,
            "landing_zone": landing,
        }
        if altitude:
            aliases["takeoff_altitude"] = altitude
        if injection:
            aliases["robustness_injection"] = injection

        instruction = template.text.format(
            prefix=prefix,
            region=region,
            target=target,
            duration=duration,
            landing=landing,
            altitude=altitude,
            suffix=suffix,
            injection=injection,
        )
        instruction = _normalize_rendered_text(instruction)
        for role in ("target", "search_region", "track_duration", "landing_zone"):
            if aliases[role] not in instruction:
                raise RendererConfigError(
                    f"template failed to express required semantic role {role}"
                )
        if altitude and altitude not in instruction:
            raise RendererConfigError("template failed to express takeoff altitude")
        if split != _ROBUSTNESS_SPLIT and any(
            injection.text in instruction
            for injection in self._lexicon.robustness_injections
        ):
            raise RendererConfigError(
                "robustness injection leaked into an ordinary dataset split"
            )
        return RenderedInstruction(
            instruction=instruction,
            template_family=template.template_family,
            paraphrase_family=template.paraphrase_family,
            difficulty=template.difficulty,
            template_id=template.template_id,
            aliases=aliases,
            robustness_category=robustness_category,
        )


def _normalize_rendered_text(value: str) -> str:
    # Collapse accidental whitespace while retaining intentional forms such as
    # ``30 秒`` and ``0.5 分钟`` from the approved language lexicon.
    value = re.sub(r"[ \t\r\n]+", " ", value).strip()
    value = re.sub(r"\s+([，。；：！？])", r"\1", value)
    if not value:
        raise RendererConfigError("rendered instruction is empty")
    return value


def render_instruction(
    gold: GoldPlannerSpec,
    world: PlannerWorldCase,
    ontology: TargetOntology,
    *,
    split: str,
    seed: int,
    variant_index: int = 0,
    lexicon: LanguageLexicon | None = None,
    templates: TemplateCatalog | None = None,
) -> RenderedInstruction:
    """Convenience wrapper around :class:`InstructionRenderer`."""

    return InstructionRenderer(ontology, lexicon, templates).render(
        gold,
        world,
        split=split,
        seed=seed,
        variant_index=variant_index,
    )


def world_case_to_runtime_context(world: PlannerWorldCase) -> PlannerWorldContext:
    """Project a public dataset world into the runtime prompt-builder schema.

    Runtime schemas require navigation geometry even though the common Prompt
    Builder intentionally hides it.  The deterministic synthetic geometry here
    is merely schema scaffolding: it contains no target location or motion and
    cannot leak through ``build_mission_planner_messages``.
    """

    if not isinstance(world, PlannerWorldCase):
        raise TypeError("world must be a PlannerWorldCase")
    lower = world.scene_min_xyz_m
    upper = world.scene_max_xyz_m
    center_x = (lower[0] + upper[0]) / 2.0
    center_y = (lower[1] + upper[1]) / 2.0
    span_x = upper[0] - lower[0]
    span_y = upper[1] - lower[1]
    radius = min(span_x, span_y) / 20.0
    flight_z = min(
        max(world.default_takeoff_altitude_m, lower[2]),
        upper[2],
    )
    search_regions: dict[str, SearchRegionSpec] = {}
    names = sorted(world.search_regions)
    for index, name in enumerate(names):
        angle = (2.0 * pi * index) / len(names)
        x = center_x + cos(angle) * span_x * 0.25
        y = center_y + sin(angle) * span_y * 0.25
        search_regions[name] = SearchRegionSpec(
            name=name,
            center_xyz_m=(x, y, lower[2]),
            radius_m=radius,
            approach_xyz_m=(x, y, flight_z),
            description=world.search_regions[name],
        )
    landing_zones: dict[str, LandingZoneSpec] = {}
    landing_names = sorted(world.landing_zones)
    for index, name in enumerate(landing_names):
        fraction = (index + 1.0) / (len(landing_names) + 1.0)
        landing_zones[name] = LandingZoneSpec(
            name=name,
            position_xy_m=(
                lower[0] + span_x * fraction,
                center_y,
            ),
            ground_altitude_m=lower[2],
            description=world.landing_zones[name],
        )
    return PlannerWorldContext(
        scene_min_xyz_m=lower,
        scene_max_xyz_m=upper,
        initial_uav_xyz_m=(center_x, center_y, lower[2]),
        search_regions=search_regions,
        landing_zones=landing_zones,
        default_takeoff_altitude_m=world.default_takeoff_altitude_m,
        default_track_duration_s=world.default_track_duration_s,
        search_timeout_s=60.0,
        goto_timeout_s=120.0,
        land_timeout_s=60.0,
    )


__all__ = [
    "AliasPool",
    "DATASET_SPLITS",
    "DEFAULT_LEXICON_PATH",
    "DEFAULT_TEMPLATES_PATH",
    "DEFAULT_WORLD_CONTEXTS_PATH",
    "InstructionRenderer",
    "LanguageLexicon",
    "RenderedInstruction",
    "RendererConfigError",
    "RobustnessInjection",
    "StrictYamlError",
    "TemplateCatalog",
    "TemplateSpec",
    "load_language_lexicon",
    "load_yaml_strict",
    "load_template_catalog",
    "load_world_cases",
    "render_instruction",
    "world_case_to_runtime_context",
]
