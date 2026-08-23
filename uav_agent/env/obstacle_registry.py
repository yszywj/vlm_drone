"""Immutable shared registry for configured scene obstacles.

This module is safe to import in ordinary CPython.  It intentionally does not
import Isaac Sim; :mod:`env.scene` is the adapter that materialises these same
specifications as simulator prims.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import MappingProxyType
from typing import Protocol

from common.ids import validate_routing_id
from common.obstacle_types import ObstacleAABB, ObstacleSpec


class _SceneObstacleConfig(Protocol):
    obstacles: tuple[ObstacleSpec, ...]


class ObstacleRegistry:
    """Read-only ordered obstacle collection with unique stable IDs."""

    __slots__ = ("_aabbs", "_by_id", "_specs")

    def __init__(self, obstacles: Iterable[ObstacleSpec] = ()) -> None:
        specs = tuple(obstacles)
        if any(not isinstance(item, ObstacleSpec) for item in specs):
            raise TypeError("obstacles must contain ObstacleSpec values")
        by_id: dict[str, ObstacleSpec] = {}
        for spec in specs:
            if spec.obstacle_id in by_id:
                raise ValueError(f"duplicate obstacle_id: {spec.obstacle_id}")
            by_id[spec.obstacle_id] = spec
        self._specs = specs
        self._by_id = MappingProxyType(by_id)
        self._aabbs = MappingProxyType(
            {obstacle_id: spec.aabb for obstacle_id, spec in by_id.items()}
        )

    @classmethod
    def from_scene_config(cls, scene: _SceneObstacleConfig) -> "ObstacleRegistry":
        if not hasattr(scene, "obstacles"):
            raise TypeError("scene must expose an obstacles tuple")
        return cls(scene.obstacles)

    @property
    def specs(self) -> tuple[ObstacleSpec, ...]:
        return self._specs

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(spec.obstacle_id for spec in self._specs)

    @property
    def aabbs(self) -> tuple[ObstacleAABB, ...]:
        return tuple(self._aabbs[spec.obstacle_id] for spec in self._specs)

    @property
    def collidable_specs(self) -> tuple[ObstacleSpec, ...]:
        return tuple(spec for spec in self._specs if spec.collidable)

    @property
    def collidable_aabbs(self) -> tuple[ObstacleAABB, ...]:
        return tuple(spec.aabb for spec in self._specs if spec.collidable)

    def get(self, obstacle_id: str) -> ObstacleSpec:
        normalized = validate_routing_id(obstacle_id, "obstacle_id")
        try:
            return self._by_id[normalized]
        except KeyError:
            raise KeyError(f"unknown obstacle_id: {normalized}") from None

    def get_aabb(self, obstacle_id: str) -> ObstacleAABB:
        normalized = validate_routing_id(obstacle_id, "obstacle_id")
        try:
            return self._aabbs[normalized]
        except KeyError:
            raise KeyError(f"unknown obstacle_id: {normalized}") from None

    def __iter__(self) -> Iterator[ObstacleSpec]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, obstacle_id: object) -> bool:
        return isinstance(obstacle_id, str) and obstacle_id in self._by_id


def obstacle_scene_prim_key(index: int, obstacle_id: str) -> str:
    """Return a stable alphabetic USD-name seed for one registry entry.

    The Isaac adapter still passes this through ``Tf.MakeValidIdentifier`` to
    replace punctuation allowed by routing IDs.  Keeping the numeric ordering
    after an alphabetic prefix is essential: a USD prim component such as
    ``000_box_red`` is invalid even though ``box_red`` itself is valid.
    """

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("obstacle index must be an integer")
    if index < 0:
        raise ValueError("obstacle index must be non-negative")
    normalized = validate_routing_id(obstacle_id, "obstacle_id")
    return f"obstacle_{index:03d}_{normalized}"


__all__ = ["ObstacleRegistry", "obstacle_scene_prim_key"]
