"""Dependency-free provenance classification shared across trust boundaries."""

from __future__ import annotations


def is_privileged_oracle_source(value: object) -> bool:
    """Return whether a provenance label denotes privileged Oracle data.

    Any label denoting Oracle, ground truth, or simulator truth is treated as
    privileged.  New evaluator adapters therefore fail closed until their
    values are explicitly projected through a non-privileged runtime boundary.
    """

    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    return any(
        marker in normalized
        for marker in (
            "oracle",
            "ground_truth",
            "groundtruth",
            "sim_truth",
            "simulator_truth",
        )
    )


__all__ = ["is_privileged_oracle_source"]
