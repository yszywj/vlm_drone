"""Protocol boundary for route critics that never generate replacement paths."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from planner.route_types import RouteDraft


@runtime_checkable
class RouteCriticProtocol(Protocol):
    def evaluate(self, route: RouteDraft, context: object) -> object: ...


__all__ = ["RouteCriticProtocol"]
