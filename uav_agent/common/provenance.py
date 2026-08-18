"""Dependency-free provenance classification shared across trust boundaries."""

from __future__ import annotations


def is_privileged_oracle_source(value: object) -> bool:
    """Return whether a provenance label denotes privileged Oracle data.

    Any label containing ``oracle`` is treated as privileged.  New evaluator
    adapters therefore fail closed until their values are explicitly projected
    through a non-Oracle runtime boundary.
    """

    return isinstance(value, str) and "oracle" in value.strip().casefold()


__all__ = ["is_privileged_oracle_source"]
