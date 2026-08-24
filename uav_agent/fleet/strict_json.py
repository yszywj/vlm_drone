"""Small strict JSON boundary shared by model-facing Fleet parsers."""

from __future__ import annotations

import json


class DuplicateJSONKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def strict_json_object_loads(raw: str) -> dict[str, object]:
    if not isinstance(raw, str):
        raise TypeError("raw JSON must be a string")
    decoded = json.loads(
        raw,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
        object_pairs_hook=_strict_object,
    )
    if not isinstance(decoded, dict):
        raise TypeError("top-level response must be an object")
    return decoded


__all__ = ["DuplicateJSONKeyError", "strict_json_object_loads"]
