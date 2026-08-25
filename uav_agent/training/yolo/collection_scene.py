"""Pure-Python contracts for the single-class cube collection scene.

The types in this module describe what an Isaac adapter must render and report;
they do not import Isaac Sim.  Shape and colour are deliberately separate:
every cube maps to detector class ``0`` while non-cube objects are hard
negatives and therefore have no detector class.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml


CUBE_CLASS_ID = 0
CUBE_CLASS_NAME = "cube"
CUBE_COLORS = ("red", "blue", "green", "yellow", "gray")
HARD_NEGATIVE_KINDS = (
    "red_sphere",
    "blue_sphere",
    "red_cylinder",
    "blue_cylinder",
    "cuboid",
    "colored_background_block",
    "partial_noncube",
)
HARD_NEGATIVE_SEMANTICS: Mapping[str, tuple[str, str]] = {
    "red_sphere": ("sphere", "red"),
    "blue_sphere": ("sphere", "blue"),
    "red_cylinder": ("cylinder", "red"),
    "blue_cylinder": ("cylinder", "blue"),
    "cuboid": ("cuboid", "gray"),
    "colored_background_block": ("colored_background_block", "yellow"),
    "partial_noncube": ("partial_noncube", "green"),
}
MAX_CUBES_PER_IMAGE = 3


class CubeCollectionConfigError(ValueError):
    """Raised when the public cube collection contract is not exact."""


def _non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CubeCollectionConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _strict_keys(
    raw: Mapping[str, Any],
    *,
    required: set[str],
    field_name: str,
) -> None:
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise CubeCollectionConfigError(f"{field_name}: " + "; ".join(details))


@dataclass(frozen=True, slots=True)
class CubeCollectionProtocol:
    """Auditable, bounded semantics loaded from ``collect_cube.yaml``."""

    schema_version: int
    cube_count_min: int
    cube_count_max: int
    cube_colors: tuple[str, ...]
    hard_negatives: tuple[str, ...]
    metadata_directory: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise CubeCollectionConfigError("schema_version must be exactly 1")
        if (
            type(self.cube_count_min) is not int
            or type(self.cube_count_max) is not int
            or self.cube_count_min != 0
            or self.cube_count_max != MAX_CUBES_PER_IMAGE
        ):
            raise CubeCollectionConfigError("cube_count range must be exactly [0, 3]")
        if self.cube_colors != CUBE_COLORS:
            raise CubeCollectionConfigError(
                "cube_colors must be exactly red, blue, green, yellow, gray"
            )
        if self.hard_negatives != HARD_NEGATIVE_KINDS:
            raise CubeCollectionConfigError("hard-negative catalog is incomplete or reordered")
        if self.metadata_directory != "metadata":
            raise CubeCollectionConfigError("metadata.directory must be exactly 'metadata'")


def load_cube_collection_protocol(path: str | Path) -> CubeCollectionProtocol:
    """Load the strict public collection contract without starting Isaac."""

    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CubeCollectionConfigError(f"cannot load cube collection config {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise CubeCollectionConfigError("collection config root must be a mapping")
    _strict_keys(
        raw,
        required={
            "schema_version",
            "detector_class",
            "scene_inventory",
            "labels",
            "metadata",
            "split",
        },
        field_name="collection config",
    )
    detector = raw["detector_class"]
    inventory = raw["scene_inventory"]
    labels = raw["labels"]
    metadata = raw["metadata"]
    split = raw["split"]
    for name, block in (
        ("detector_class", detector),
        ("scene_inventory", inventory),
        ("labels", labels),
        ("metadata", metadata),
        ("split", split),
    ):
        if not isinstance(block, Mapping):
            raise CubeCollectionConfigError(f"{name} must be a mapping")
    _strict_keys(detector, required={"id", "name"}, field_name="detector_class")
    if (
        type(detector["id"]) is not int
        or detector["id"] != CUBE_CLASS_ID
        or detector["name"] != CUBE_CLASS_NAME
    ):
        raise CubeCollectionConfigError("detector class must be exactly 0: cube")
    _strict_keys(
        inventory,
        required={"cube_count", "cube_colors", "hard_negatives"},
        field_name="scene_inventory",
    )
    count = inventory["cube_count"]
    if not isinstance(count, Mapping):
        raise CubeCollectionConfigError("scene_inventory.cube_count must be a mapping")
    _strict_keys(count, required={"min", "max"}, field_name="cube_count")
    _strict_keys(
        labels,
        required={
            "format",
            "label_every_visible_cube",
            "non_cube_class_id",
            "use_rendered_object_dimensions",
        },
        field_name="labels",
    )
    if labels != {
        "format": "yolo_detect",
        "label_every_visible_cube": True,
        "non_cube_class_id": None,
        "use_rendered_object_dimensions": True,
    }:
        raise CubeCollectionConfigError("labels must preserve the strict cube-v1 semantics")
    _strict_keys(
        metadata,
        required={"schema_version", "directory", "store_rgb", "store_depth"},
        field_name="metadata",
    )
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != 1
        or metadata["store_rgb"] is not False
        or metadata["store_depth"] is not False
    ):
        raise CubeCollectionConfigError("metadata must be schema 1 and must not store RGB/depth")
    _strict_keys(
        split,
        required={"grouping_keys", "require_train_val_test"},
        field_name="split",
    )
    if (
        split["grouping_keys"] != ["episode_id", "trajectory_id"]
        or split["require_train_val_test"] is not True
    ):
        raise CubeCollectionConfigError("split must group by episode_id and trajectory_id")
    colors = inventory["cube_colors"]
    negatives = inventory["hard_negatives"]
    if not isinstance(colors, Sequence) or isinstance(colors, (str, bytes)):
        raise CubeCollectionConfigError("cube_colors must be a sequence")
    if not isinstance(negatives, Sequence) or isinstance(negatives, (str, bytes)):
        raise CubeCollectionConfigError("hard_negatives must be a sequence")
    return CubeCollectionProtocol(
        schema_version=raw["schema_version"],
        cube_count_min=count["min"],
        cube_count_max=count["max"],
        cube_colors=tuple(colors),
        hard_negatives=tuple(negatives),
        metadata_directory=_non_empty_text(metadata["directory"], "metadata.directory"),
    )


@dataclass(frozen=True, slots=True)
class CollectionSceneObject:
    """One rendered object; dimensions are the adapter's real scene dimensions."""

    object_id: str
    shape: str
    color_name: str
    position_world_m: tuple[float, float, float]
    orientation_world_wxyz: tuple[float, float, float, float]
    dimensions_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _non_empty_text(self.object_id, "object_id"))
        shape = _non_empty_text(self.shape, "shape").lower()
        color = _non_empty_text(self.color_name, "color_name").lower()
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "color_name", color)
        if shape == CUBE_CLASS_NAME and color not in CUBE_COLORS:
            raise CubeCollectionConfigError(
                "cube color_name must be red, blue, green, yellow, or gray"
            )
        if (
            len(self.position_world_m) != 3
            or len(self.orientation_world_wxyz) != 4
            or len(self.dimensions_xyz_m) != 3
        ):
            raise CubeCollectionConfigError("object pose/dimensions have invalid arity")
        numeric = (*self.position_world_m, *self.orientation_world_wxyz, *self.dimensions_xyz_m)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in numeric
        ):
            raise CubeCollectionConfigError("object pose/dimensions must be finite numbers")
        if any(float(value) <= 0.0 for value in self.dimensions_xyz_m):
            raise CubeCollectionConfigError("object dimensions must be positive")
        norm = float(np.linalg.norm(np.asarray(self.orientation_world_wxyz, dtype=np.float64)))
        if norm <= 1e-9:
            raise CubeCollectionConfigError("object orientation quaternion cannot be zero")

    @property
    def detector_class_id(self) -> int | None:
        return CUBE_CLASS_ID if self.shape == CUBE_CLASS_NAME else None

    @property
    def detector_class_name(self) -> str | None:
        return CUBE_CLASS_NAME if self.shape == CUBE_CLASS_NAME else None


def oriented_box_corners_world(obj: CollectionSceneObject) -> np.ndarray:
    """Return eight corners using the object's supplied real dimensions."""

    half = np.asarray(obj.dimensions_xyz_m, dtype=np.float64) * 0.5
    local = np.asarray(
        [
            (sx * half[0], sy * half[1], sz * half[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    quaternion = np.asarray(obj.orientation_world_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    rotation = np.asarray(
        [
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ],
        dtype=np.float64,
    )
    return local @ rotation.T + np.asarray(obj.position_world_m, dtype=np.float64)


def transformed_local_bounds_corners(
    minimum_xyz: Sequence[float],
    maximum_xyz: Sequence[float],
    transform_point: Callable[[tuple[float, float, float]], Sequence[float]],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Transform an oriented local bound without collapsing it to a world AABB.

    Isaac's ``ComputeAlignedBox`` loses object orientation.  The late-bound USD
    adapter instead supplies the local range and a point transform here.  Edge
    lengths are measured after the transform, so metadata dimensions reflect
    the actual rendered USD scale rather than a legacy hard-coded target size.
    This helper remains pure Python and is directly testable without Isaac.
    """

    minimum = np.asarray(minimum_xyz, dtype=np.float64)
    maximum = np.asarray(maximum_xyz, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise CubeCollectionConfigError("local bounds must have three coordinates")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise CubeCollectionConfigError("local bounds must be finite")
    if np.any(maximum <= minimum):
        raise CubeCollectionConfigError("local bounds must have positive extent")
    if not callable(transform_point):
        raise TypeError("transform_point must be callable")
    local = tuple(
        (float(x), float(y), float(z))
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    )
    corners = np.asarray([transform_point(point) for point in local], dtype=np.float64)
    if corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
        raise CubeCollectionConfigError(
            "transform_point must return eight finite world xyz coordinates"
        )
    origin = corners[0]
    dimensions = (
        float(np.linalg.norm(corners[4] - origin)),
        float(np.linalg.norm(corners[2] - origin)),
        float(np.linalg.norm(corners[1] - origin)),
    )
    if any(value <= 0.0 or not isfinite(value) for value in dimensions):
        raise CubeCollectionConfigError("transformed bounds have invalid dimensions")
    return corners, dimensions


def _yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    return (cos(yaw_rad * 0.5), 0.0, 0.0, sin(yaw_rad * 0.5))


def build_cube_v1_scene_inventory(
    protocol: CubeCollectionProtocol,
    *,
    scene_seed: int,
    episode_index: int,
    sample_kind: str,
    anchor_position_world_m: Sequence[float],
    target_scale: float,
) -> tuple[CollectionSceneObject, ...]:
    """Build the deterministic inventory rendered by the formal Isaac adapter.

    Positive episodes exercise one through three cubes, partial-occlusion
    episodes use three cubes, and negative episodes use zero.  Every inventory
    also contains the complete seven-item hard-negative catalog.  Cube colour
    is metadata-only and cycles through the five protocol colours.
    """

    if not isinstance(protocol, CubeCollectionProtocol):
        raise TypeError("protocol must be a CubeCollectionProtocol")
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int) or scene_seed < 0:
        raise CubeCollectionConfigError("scene_seed must be a non-negative integer")
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        raise CubeCollectionConfigError("episode_index must be a non-negative integer")
    normalized_kind = _non_empty_text(sample_kind, "sample_kind").lower()
    if normalized_kind not in {"positive", "partial_occlusion", "negative"}:
        raise CubeCollectionConfigError("sample_kind is invalid")
    anchor = np.asarray(anchor_position_world_m, dtype=np.float64)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise CubeCollectionConfigError("anchor_position_world_m must be finite xyz")
    if (
        isinstance(target_scale, bool)
        or not isinstance(target_scale, (int, float))
        or not isfinite(float(target_scale))
        or float(target_scale) <= 0.0
    ):
        raise CubeCollectionConfigError("target_scale must be finite and positive")

    if normalized_kind == "negative":
        cube_count = 0
    elif normalized_kind == "partial_occlusion":
        cube_count = 3
    else:
        cube_count = 1 + (episode_index // 3) % MAX_CUBES_PER_IMAGE
    rng = random.Random((scene_seed << 32) ^ episode_index ^ 0xC0BE_0001)
    objects: list[CollectionSceneObject] = []
    cube_offsets = ((0.0, 0.0), (1.45, 0.65), (-1.35, 0.85))
    for slot in range(cube_count):
        if cube_count >= 2 and slot < 2:
            color = ("red", "blue")[slot]
        elif normalized_kind == "partial_occlusion" and slot == 2:
            color = ("green", "yellow", "gray")[(episode_index // 3) % 3]
        else:
            color = protocol.cube_colors[(episode_index + slot) % len(protocol.cube_colors)]
        side = float(target_scale) * rng.uniform(0.65, 1.25)
        offset_x, offset_y = cube_offsets[slot]
        position = anchor + np.asarray((offset_x, offset_y, 0.0), dtype=np.float64)
        objects.append(
            CollectionSceneObject(
                object_id=f"cube_{slot}",
                shape=CUBE_CLASS_NAME,
                color_name=color,
                position_world_m=tuple(float(value) for value in position),
                orientation_world_wxyz=_yaw_quaternion(rng.uniform(-pi, pi)),
                dimensions_xyz_m=(side, side, side),
            )
        )

    hard_dimensions = {
        "red_sphere": (0.85, 0.85, 0.85),
        "blue_sphere": (1.05, 1.05, 1.05),
        "red_cylinder": (0.80, 0.80, 1.35),
        "blue_cylinder": (1.00, 1.00, 1.10),
        "cuboid": (1.85, 0.55, 0.70),
        "colored_background_block": (2.40, 0.22, 1.55),
        "partial_noncube": (0.18, 0.75, 1.30),
    }
    for index, kind in enumerate(protocol.hard_negatives):
        shape, color = HARD_NEGATIVE_SEMANTICS[kind]
        angle = (2.0 * pi * index / len(protocol.hard_negatives)) + rng.uniform(-0.10, 0.10)
        radius = 3.0 + 0.35 * (index % 3)
        position = anchor + np.asarray(
            (radius * cos(angle), radius * sin(angle), 0.15 * (index % 2)),
            dtype=np.float64,
        )
        dimensions = tuple(
            float(target_scale) * value for value in hard_dimensions[kind]
        )
        objects.append(
            CollectionSceneObject(
                object_id=kind,
                shape=shape,
                color_name=color,
                position_world_m=tuple(float(value) for value in position),
                orientation_world_wxyz=_yaw_quaternion(rng.uniform(-pi, pi)),
                dimensions_xyz_m=dimensions,
            )
        )
    validate_scene_inventory(objects)
    return tuple(objects)


def validate_scene_inventory(objects: Sequence[CollectionSceneObject]) -> None:
    """Reject duplicate IDs or scenes outside the documented 0--3 cube bound."""

    if len(objects) > 32:
        raise CubeCollectionConfigError("scene inventory exceeds the bounded object limit")
    identifiers = [obj.object_id for obj in objects]
    if len(set(identifiers)) != len(identifiers):
        raise CubeCollectionConfigError("scene object_id values must be unique")
    cube_count = sum(obj.shape == CUBE_CLASS_NAME for obj in objects)
    if not 0 <= cube_count <= MAX_CUBES_PER_IMAGE:
        raise CubeCollectionConfigError("a frame must contain between 0 and 3 cubes")


__all__ = [
    "CUBE_CLASS_ID",
    "CUBE_CLASS_NAME",
    "CUBE_COLORS",
    "HARD_NEGATIVE_SEMANTICS",
    "HARD_NEGATIVE_KINDS",
    "MAX_CUBES_PER_IMAGE",
    "CollectionSceneObject",
    "CubeCollectionConfigError",
    "CubeCollectionProtocol",
    "load_cube_collection_protocol",
    "build_cube_v1_scene_inventory",
    "oriented_box_corners_world",
    "transformed_local_bounds_corners",
    "validate_scene_inventory",
]
