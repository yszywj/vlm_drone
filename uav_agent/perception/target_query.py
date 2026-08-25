"""Whitelisted production target-query values.

This module is intentionally dependency-light and contains no simulator
types.  A :class:`TargetQuerySpec` is the only target-description object that
may cross from a Fleet Assignment into the production perception runtime.
Search geometry remains in the SEARCH Goal and simulator target configuration
remains on the scene/evaluator side of the information boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral
from types import MappingProxyType

from common.ids import validate_routing_id
from target.types import TargetSpec


ALLOWED_TARGET_QUERY_FIELDS = (
    "target_alias",
    "detector_class_id",
    "detector_class_name",
    "hard_attributes",
    "soft_description",
)

# Match normalized field-like text, not innocent words such as "moving" in a
# user's semantic description.  These names denote simulator/evaluator
# capabilities and can never be target attributes in production.
_FORBIDDEN_TRUTH_MARKERS = (
    "oracle_target",
    "ground_truth",
    "sim_truth",
    "position_world",
    "positionworld",
    "target_position",
    "targetposition",
    "velocity_world",
    "velocityworld",
    "target_velocity",
    "targetvelocity",
    "initial_region",
    "initialregion",
    "motion_region",
    "motionregion",
    "motion_seed",
    "motionseed",
    "prim_path",
    "primpath",
    "object_id",
    "objectid",
    "instance_id",
    "instanceid",
    "instance_segmentation",
    "future_trajectory",
    "futuretrajectory",
    "world_coordinate",
    "worldcoordinate",
    "world_coord",
    "search_region",
    "searchregion",
    "search_area",
    "世界坐标",
    "搜索区域",
)
_FORBIDDEN_ATTRIBUTE_NAMES = frozenset(
    {
        "position",
        "pose",
        "velocity",
        "speed",
        "coordinate",
        "coordinates",
        "location",
        "region",
        "seed",
        "trajectory",
        "path",
        "object_id",
        "instance_id",
        "segmentation_id",
    }
)


def _bounded_text(
    value: object,
    field_name: str,
    *,
    maximum: int,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} characters")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} contains control characters")
    return normalized


def _truth_key(value: str) -> str:
    return "_".join(value.casefold().replace("/", " ").replace(".", " ").split())


def _reject_truth_marker(value: str, field_name: str) -> None:
    folded = _truth_key(value)
    if any(marker in folded for marker in _FORBIDDEN_TRUTH_MARKERS):
        raise ValueError(
            f"{field_name} contains a forbidden simulator/evaluator truth marker"
        )


def _parse_hard_attributes(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, raw_value in enumerate(values):
        text = _bounded_text(raw_value, f"hard_attributes[{index}]", maximum=256)
        if "=" in text:
            raw_name, raw_semantic_value = text.split("=", 1)
        elif ":" in text:
            raw_name, raw_semantic_value = text.split(":", 1)
        else:
            raw_name, raw_semantic_value = f"attribute_{index}", text
        name = _bounded_text(raw_name, f"hard_attributes[{index}].name", maximum=64)
        semantic_value = _bounded_text(
            raw_semantic_value,
            f"hard_attributes[{index}].value",
            maximum=256,
        )
        if name in result:
            raise ValueError(f"duplicate hard attribute name: {name!r}")
        result[name] = semantic_value
    return result


@dataclass(frozen=True, slots=True)
class TargetQuerySpec:
    """Assignment semantics permitted to enter production target perception.

    The exact top-level schema is deliberately closed.  ``MappingProxyType``
    also prevents a caller from mutating nested attributes after validation.
    """

    target_alias: str
    detector_class_id: int
    detector_class_name: str
    hard_attributes: Mapping[str, str]
    soft_description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_alias",
            validate_routing_id(self.target_alias, "target_alias"),
        )
        if (
            isinstance(self.detector_class_id, bool)
            or not isinstance(self.detector_class_id, Integral)
            or int(self.detector_class_id) < 0
        ):
            raise ValueError("detector_class_id must be a non-negative integer")
        object.__setattr__(self, "detector_class_id", int(self.detector_class_id))
        class_name = _bounded_text(
            self.detector_class_name,
            "detector_class_name",
            maximum=128,
        )
        _reject_truth_marker(class_name, "detector_class_name")
        object.__setattr__(self, "detector_class_name", class_name)

        if not isinstance(self.hard_attributes, Mapping):
            raise TypeError("hard_attributes must be a mapping of semantic strings")
        if len(self.hard_attributes) > 32:
            raise ValueError("hard_attributes must contain at most 32 entries")
        normalized: dict[str, str] = {}
        for raw_name, raw_value in self.hard_attributes.items():
            name = _bounded_text(raw_name, "hard attribute name", maximum=64)
            value = _bounded_text(raw_value, f"hard_attributes[{name!r}]", maximum=256)
            if _truth_key(name) in _FORBIDDEN_ATTRIBUTE_NAMES:
                raise ValueError(
                    f"hard_attributes[{name!r}] is a forbidden truth field"
                )
            _reject_truth_marker(name, f"hard_attributes[{name!r}]")
            _reject_truth_marker(value, f"hard_attributes[{name!r}]")
            if name in normalized:
                raise ValueError(f"duplicate hard attribute name: {name!r}")
            normalized[name] = value
        object.__setattr__(
            self,
            "hard_attributes",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

        if self.soft_description is not None:
            description = _bounded_text(
                self.soft_description,
                "soft_description",
                maximum=512,
            )
            _reject_truth_marker(description, "soft_description")
            object.__setattr__(self, "soft_description", description)

        # A code change that adds another dataclass field must update the
        # public audit constant in the same review.
        actual_fields = tuple(field.name for field in fields(self))
        if actual_fields != ALLOWED_TARGET_QUERY_FIELDS:
            raise RuntimeError("TargetQuerySpec field whitelist is out of sync")

    @classmethod
    def from_assignment_semantics(
        cls,
        *,
        target_alias: str,
        target_spec: TargetSpec,
        detector_class_id: int,
        detector_class_name: str,
    ) -> "TargetQuerySpec":
        """Project a trusted Assignment TargetSpec onto the production schema."""

        if not isinstance(target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec")
        return cls(
            target_alias=target_alias,
            detector_class_id=detector_class_id,
            detector_class_name=detector_class_name,
            hard_attributes=_parse_hard_attributes(target_spec.hard_attributes),
            soft_description=target_spec.original_description,
        )

    def to_semantic_target_spec(self) -> TargetSpec:
        """Create the legacy semantic value needed by verifier adapters.

        This is a one-way projection from already-audited fields; it cannot
        recreate simulator geometry, IDs, motion state, or future trajectory.
        """

        description = self.soft_description or self.detector_class_name
        return TargetSpec(
            description,
            category=self.detector_class_name,
            hard_attributes=tuple(
                f"{name}={value}" for name, value in self.hard_attributes.items()
            ),
            immutable_identity_summary=(
                f"{self.target_alias}:{self.detector_class_name}:"
                + ",".join(
                    f"{name}={value}" for name, value in self.hard_attributes.items()
                )
            ),
        )

    def to_audit_dict(self) -> dict[str, object]:
        """Return schema metadata only, never the mission's query contents."""

        return {"allowed_target_query_fields": list(ALLOWED_TARGET_QUERY_FIELDS)}


__all__ = ["ALLOWED_TARGET_QUERY_FIELDS", "TargetQuerySpec"]
