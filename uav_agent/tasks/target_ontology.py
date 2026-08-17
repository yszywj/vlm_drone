"""Closed-set, deterministic target ontology for Planner data v1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import unicodedata

import yaml
from yaml.constructor import ConstructorError

from .schemas import GoldPlannerSpec, TargetConcept


DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "planner_v1"
    / "target_ontology.yaml"
)


class TargetOntologyError(ValueError):
    """Raised when an ontology or a concept violates the closed vocabulary."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects conflicting/duplicate mapping keys."""


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
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
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


def _name(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise TargetOntologyError(f"{field_name} must be non-empty")
    return normalized


def _alias_key(value: object, field_name: str = "target alias") -> str:
    text = _name(value, field_name)
    # This normalization is deterministic spelling normalization, not fuzzy
    # matching: only explicitly registered aliases can resolve a concept.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


_COLOR_LABELS = MappingProxyType(
    {
        "red": "红色",
        "blue": "蓝色",
        "black": "黑色",
        "white": "白色",
        "yellow": "黄色",
        "gray": "灰色",
    }
)
_PERSON_ATTRIBUTE_ORDER = (
    "upper_clothing_color",
    "lower_clothing_color",
    "backpack_color",
)


def _person_attribute_phrase(attribute: str, value: str) -> str:
    if attribute == "upper_clothing_color":
        try:
            return f"穿{_COLOR_LABELS[value]}上衣"
        except KeyError as exc:
            raise TargetOntologyError(
                f"cannot render unknown upper clothing color {value!r}"
            ) from exc
    if attribute == "lower_clothing_color":
        try:
            return f"穿{_COLOR_LABELS[value]}下装"
        except KeyError as exc:
            raise TargetOntologyError(
                f"cannot render unknown lower clothing color {value!r}"
            ) from exc
    if attribute == "backpack_color":
        if value == "none":
            return "未背背包"
        try:
            return f"背{_COLOR_LABELS[value]}背包"
        except KeyError as exc:
            raise TargetOntologyError(
                f"cannot render unknown backpack color {value!r}"
            ) from exc
    raise TargetOntologyError(f"cannot render unknown person attribute {attribute!r}")


def render_canonical_target_description(
    category: str,
    attributes: Mapping[str, str],
) -> str:
    """Render a canonical description using a fixed semantic field order."""

    category = _name(category, "category")
    if category != "person":
        raise TargetOntologyError(
            f"no canonical description renderer for category {category!r}"
        )
    if not isinstance(attributes, Mapping) or not attributes:
        raise TargetOntologyError("attributes must be a non-empty mapping")
    unknown = set(attributes) - set(_PERSON_ATTRIBUTE_ORDER)
    if unknown:
        raise TargetOntologyError(
            "unknown person attributes: " + ", ".join(sorted(unknown))
        )
    phrases = [
        _person_attribute_phrase(attribute, _name(attributes[attribute], attribute))
        for attribute in _PERSON_ATTRIBUTE_ORDER
        if attribute in attributes
    ]
    if len(phrases) == 1:
        body = phrases[0]
    elif len(phrases) == 2:
        body = f"{phrases[0]}并{phrases[1]}"
    else:
        body = "、".join(phrases[:-1]) + f"并{phrases[-1]}"
    return f"{body}的人"


def _readonly_categories(
    raw: object,
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    if not isinstance(raw, Mapping) or not raw:
        raise TargetOntologyError("categories must be a non-empty mapping")
    categories: dict[str, Mapping[str, tuple[str, ...]]] = {}
    for raw_category, raw_attributes in raw.items():
        category = _name(raw_category, "category name")
        if not isinstance(raw_attributes, Mapping) or not raw_attributes:
            raise TargetOntologyError(
                f"category {category!r} must define a non-empty attribute mapping"
            )
        attributes: dict[str, tuple[str, ...]] = {}
        for raw_attribute, raw_values in raw_attributes.items():
            attribute = _name(raw_attribute, f"{category} attribute name")
            if (
                isinstance(raw_values, (str, bytes))
                or not isinstance(raw_values, Sequence)
                or not raw_values
            ):
                raise TargetOntologyError(
                    f"{category}.{attribute} must define a non-empty value list"
                )
            values = tuple(
                _name(value, f"{category}.{attribute} value")
                for value in raw_values
            )
            if len(set(values)) != len(values):
                raise TargetOntologyError(
                    f"{category}.{attribute} contains duplicate values"
                )
            attributes[attribute] = values
        categories[category] = MappingProxyType(attributes)
    return MappingProxyType(categories)


@dataclass(frozen=True, slots=True)
class TargetOntology:
    """Validated concepts and a deterministic, exact alias lookup table."""

    schema_version: str
    categories: Mapping[str, Mapping[str, tuple[str, ...]]]
    concepts: Mapping[str, TargetConcept]
    aliases_by_concept: Mapping[str, tuple[str, ...]]
    _alias_to_concept: Mapping[str, str]
    _attribute_signature_to_concept: Mapping[
        tuple[str, tuple[tuple[str, str], ...]], str
    ]

    @classmethod
    def from_file(cls, path: str | Path) -> "TargetOntology":
        try:
            raw = yaml.load(
                Path(path).read_text(encoding="utf-8"),
                Loader=_UniqueKeySafeLoader,
            )
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise TargetOntologyError("could not read target ontology") from exc
        return cls.from_mapping(raw)

    @classmethod
    def load_default(cls) -> "TargetOntology":
        return cls.from_file(DEFAULT_ONTOLOGY_PATH)

    @classmethod
    def from_mapping(cls, raw: object) -> "TargetOntology":
        if not isinstance(raw, Mapping):
            raise TargetOntologyError("target ontology root must be a mapping")
        allowed_root = {"schema_version", "categories", "concepts"}
        unknown_root = set(raw) - allowed_root
        if unknown_root:
            raise TargetOntologyError(
                "target ontology contains unknown fields: "
                + ", ".join(sorted(str(value) for value in unknown_root))
            )
        missing_root = allowed_root - set(raw)
        if missing_root:
            raise TargetOntologyError(
                "target ontology is missing fields: "
                + ", ".join(sorted(missing_root))
            )

        schema_version = _name(raw["schema_version"], "schema_version")
        categories = _readonly_categories(raw["categories"])
        raw_concepts = raw["concepts"]
        if (
            isinstance(raw_concepts, (str, bytes, Mapping))
            or not isinstance(raw_concepts, Sequence)
            or not raw_concepts
        ):
            raise TargetOntologyError("concepts must be a non-empty list")

        concepts: dict[str, TargetConcept] = {}
        concept_aliases: dict[str, tuple[str, ...]] = {}
        aliases: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        signatures: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}
        for index, entry in enumerate(raw_concepts):
            if not isinstance(entry, Mapping):
                raise TargetOntologyError(f"concepts[{index}] must be a mapping")
            allowed_concept = {
                "concept_id",
                "category",
                "attributes",
                "canonical_description",
                "aliases",
            }
            unknown = set(entry) - allowed_concept
            required = {"concept_id", "category", "attributes"}
            if unknown:
                raise TargetOntologyError(
                    f"concepts[{index}] contains unknown fields: "
                    + ", ".join(sorted(str(value) for value in unknown))
                )
            missing = required - set(entry)
            if missing:
                raise TargetOntologyError(
                    f"concepts[{index}] is missing fields: "
                    + ", ".join(sorted(missing))
                )
            concept_id = _name(entry["concept_id"], f"concepts[{index}].concept_id")
            if concept_id in concepts:
                raise TargetOntologyError(f"duplicate concept_id {concept_id!r}")
            category = _name(entry["category"], f"concepts[{index}].category")
            attributes = entry["attributes"]
            cls._validate_attributes(categories, category, attributes)
            canonical = render_canonical_target_description(category, attributes)
            supplied_canonical = entry.get("canonical_description")
            if supplied_canonical is not None and supplied_canonical != canonical:
                raise TargetOntologyError(
                    f"concept {concept_id!r} has non-canonical target description"
                )
            if canonical in descriptions:
                raise TargetOntologyError(
                    f"duplicate canonical description for {concept_id!r} and "
                    f"{descriptions[canonical]!r}"
                )
            descriptions[canonical] = concept_id
            concept = TargetConcept(
                concept_id=concept_id,
                category=category,
                attributes=attributes,
                canonical_description=canonical,
            )
            signature = (
                category,
                tuple(sorted(concept.attributes.items())),
            )
            if signature in signatures:
                raise TargetOntologyError(
                    f"concept {concept_id!r} duplicates the attributes of "
                    f"{signatures[signature]!r}"
                )
            signatures[signature] = concept_id
            concepts[concept_id] = concept

            raw_aliases = entry.get("aliases", ())
            if isinstance(raw_aliases, (str, bytes)) or not isinstance(
                raw_aliases, Sequence
            ):
                raise TargetOntologyError(
                    f"concepts[{index}].aliases must be a list of strings"
                )
            display_aliases = tuple(
                _name(value, f"concepts[{index}].aliases item")
                for value in raw_aliases
            )
            if len({_alias_key(value) for value in display_aliases}) != len(
                display_aliases
            ):
                raise TargetOntologyError(
                    f"concept {concept_id!r} contains duplicate aliases"
                )
            concept_aliases[concept_id] = display_aliases
            for alias in (canonical, *display_aliases):
                key = _alias_key(alias)
                existing = aliases.get(key)
                if existing is not None and existing != concept_id:
                    raise TargetOntologyError(
                        f"target alias {alias!r} maps to both {existing!r} "
                        f"and {concept_id!r}"
                    )
                aliases[key] = concept_id

        return cls(
            schema_version=schema_version,
            categories=categories,
            concepts=MappingProxyType(concepts),
            aliases_by_concept=MappingProxyType(concept_aliases),
            _alias_to_concept=MappingProxyType(aliases),
            _attribute_signature_to_concept=MappingProxyType(signatures),
        )

    @staticmethod
    def _validate_attributes(
        categories: Mapping[str, Mapping[str, tuple[str, ...]]],
        category: object,
        attributes: object,
    ) -> None:
        category_name = _name(category, "category")
        if category_name not in categories:
            raise TargetOntologyError(f"unknown target category {category_name!r}")
        if not isinstance(attributes, Mapping) or not attributes:
            raise TargetOntologyError("target attributes must be a non-empty mapping")
        if any(not isinstance(key, str) for key in attributes):
            raise TargetOntologyError("target attribute names must be strings")
        allowed_attributes = categories[category_name]
        unknown = set(attributes) - set(allowed_attributes)
        if unknown:
            raise TargetOntologyError(
                f"unknown attributes for {category_name!r}: "
                + ", ".join(sorted(unknown))
            )
        for attribute, raw_value in attributes.items():
            value = _name(raw_value, f"attribute {attribute}")
            if value not in allowed_attributes[attribute]:
                raise TargetOntologyError(
                    f"unknown value {value!r} for target attribute {attribute!r}"
                )

    def render_description(
        self,
        category: str,
        attributes: Mapping[str, str],
    ) -> str:
        self._validate_attributes(self.categories, category, attributes)
        return render_canonical_target_description(category, attributes)

    def concept_for_attributes(
        self,
        category: str,
        attributes: Mapping[str, str],
    ) -> TargetConcept | None:
        self._validate_attributes(self.categories, category, attributes)
        key = (category.strip(), tuple(sorted(attributes.items())))
        concept_id = self._attribute_signature_to_concept.get(key)
        return self.concepts.get(concept_id) if concept_id is not None else None

    def require_concept(self, concept_id: str) -> TargetConcept:
        normalized = _name(concept_id, "concept_id")
        try:
            return self.concepts[normalized]
        except KeyError as exc:
            raise TargetOntologyError(f"unknown target concept {normalized!r}") from exc

    def resolve_description(self, description: str) -> TargetConcept | None:
        concept_id = self._alias_to_concept.get(_alias_key(description))
        return self.concepts.get(concept_id) if concept_id is not None else None

    def resolve_concept_id(self, description: str) -> str | None:
        concept = self.resolve_description(description)
        return concept.concept_id if concept is not None else None

    def aliases_for(self, concept_id: str) -> tuple[str, ...]:
        self.require_concept(concept_id)
        return self.aliases_by_concept[concept_id]

    def validate_gold_spec(self, gold: GoldPlannerSpec) -> None:
        if not isinstance(gold, GoldPlannerSpec):
            raise TypeError("gold must be a GoldPlannerSpec")
        concept = self.require_concept(gold.target_concept_id)
        if gold.target_description != concept.canonical_description:
            raise TargetOntologyError(
                "Gold target_description must equal the concept's canonical description"
            )

    def validate_concept(self, concept: TargetConcept) -> None:
        """Validate a standalone concept against this ontology's vocabulary."""

        if not isinstance(concept, TargetConcept):
            raise TypeError("concept must be a TargetConcept")
        self._validate_attributes(
            self.categories,
            concept.category,
            concept.attributes,
        )
        expected = self.render_description(concept.category, concept.attributes)
        if concept.canonical_description != expected:
            raise TargetOntologyError(
                "TargetConcept canonical_description is not canonical for its attributes"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "categories": {
                category: {
                    attribute: list(values)
                    for attribute, values in attributes.items()
                }
                for category, attributes in self.categories.items()
            },
            "concepts": [
                {
                    **concept.to_dict(),
                    "aliases": list(self.aliases_by_concept[concept.concept_id]),
                }
                for concept in self.concepts.values()
            ],
        }


def load_target_ontology(
    path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> TargetOntology:
    return TargetOntology.from_file(path)


__all__ = [
    "DEFAULT_ONTOLOGY_PATH",
    "TargetOntology",
    "TargetOntologyError",
    "load_target_ontology",
    "render_canonical_target_description",
]
