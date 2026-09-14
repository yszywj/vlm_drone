"""Adapter-specific JSON Schema serialization without changing constraints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from enum import Enum

from models.base import GenerationOptions, JsonSchemaResponseFormat


class JsonSchemaPropertyOrder(str, Enum):
    PRESERVE = "preserve"
    ALPHABETICAL = "alphabetical"


_SCHEMA_MAPS = {"$defs", "definitions", "patternProperties", "dependentSchemas"}
_SCHEMA_ARRAYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_VALUES = {
    "additionalProperties", "unevaluatedProperties", "propertyNames",
    "contains", "additionalItems", "unevaluatedItems", "not", "if", "then",
    "else", "contentSchema",
}


def _alphabetical_properties(schema: object) -> object:
    """Visit schema locations, keeping literal defaults/enum/const untouched.

    Property names may themselves be JSON Schema keywords.  Handle schema
    maps explicitly so such names never get mistaken for another keyword.
    Array order (including required, enum, and alternatives) is significant
    to generation and is always preserved.
    """

    if not isinstance(schema, dict):
        return deepcopy(schema)
    result: dict[str, object] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            result[key] = {
                name: _alphabetical_properties(value[name]) for name in sorted(value)
            }
        elif key in _SCHEMA_MAPS and isinstance(value, dict):
            result[key] = {
                name: _alphabetical_properties(item) for name, item in value.items()
            }
        elif key in _SCHEMA_ARRAYS and isinstance(value, list):
            result[key] = [_alphabetical_properties(item) for item in value]
        elif key == "items":
            result[key] = (
                [_alphabetical_properties(item) for item in value]
                if isinstance(value, list)
                else _alphabetical_properties(value)
            )
        elif key in _SCHEMA_VALUES:
            result[key] = _alphabetical_properties(value)
        elif key == "dependencies" and isinstance(value, dict):
            result[key] = {
                name: (
                    _alphabetical_properties(item)
                    if isinstance(item, (dict, bool)) else deepcopy(item)
                )
                for name, item in value.items()
            }
        else:
            result[key] = deepcopy(value)
    return result


def apply_json_schema_property_order(
    options: GenerationOptions | None,
    order: JsonSchemaPropertyOrder,
) -> GenerationOptions | None:
    """Return independently frozen options for an adapter's declared order."""

    if options is not None and not isinstance(options, GenerationOptions):
        raise TypeError("options must be GenerationOptions or None")
    if not isinstance(order, JsonSchemaPropertyOrder):
        raise TypeError("order must be a JsonSchemaPropertyOrder")
    if (
        options is None
        or options.response_format is None
        or order is JsonSchemaPropertyOrder.PRESERVE
    ):
        return options
    response_format = options.response_format
    schema = _alphabetical_properties(response_format.to_dict()["schema"])
    assert isinstance(schema, dict)
    return replace(
        options,
        response_format=JsonSchemaResponseFormat(response_format.name, schema),
    )


__all__ = ["JsonSchemaPropertyOrder", "apply_json_schema_property_order"]
