"""Read-only validation of Ultralytics-format YOLO datasets."""

from __future__ import annotations

import base64
import binascii
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

import yaml

from fleet.strict_json import strict_json_object_loads


_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_SPLITS = ("train", "val", "test")
_CUBE_PROTOCOL = "cube-v1"
_CUBE_CLASS_NAME = "cube"
_CUBE_COLORS = ("red", "blue", "green", "yellow", "gray")
_HARD_NEGATIVE_KINDS = (
    "red_sphere",
    "blue_sphere",
    "red_cylinder",
    "blue_cylinder",
    "cuboid",
    "colored_background_block",
    "partial_noncube",
)
_VISIBLE_POSITIVE_STATES = frozenset(
    {"visible", "edge_clipped", "partially_occluded"}
)
_CUBE_VISIBILITY_STATES = frozenset(
    {
        "visible",
        "edge_clipped",
        "partially_occluded",
        "fully_occluded",
        "behind_camera",
        "out_of_frame",
        "too_small",
        "depth_unavailable",
        "invalid_projection",
        "invalid_depth",
        "depth_unresolved",
        "negative",
    }
)
_PRIVILEGED_METADATA_TOKENS = frozenset(
    {"image", "depth", "rgb", "crop", "base64", "prompt", "payload"}
)
_DATA_URI = re.compile(r"^data:[^,]{0,256},", re.IGNORECASE)
_BASE64_TEXT = re.compile(r"^[A-Za-z0-9+/_-]+={0,2}$")
_CUBE_COVERAGE_KEYS = (
    "positive",
    "negative",
    "red_cube",
    "blue_cube",
    "partial_occlusion",
    "multi_cube",
)
_CUBE_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "split",
        "scene_seed",
        "episode_id",
        "trajectory_id",
        "frame_id",
        "objects",
    }
)
_CUBE_METADATA_OBJECT_FIELDS = frozenset(
    {
        "object_id",
        "shape",
        "color_name",
        "detector_class_id",
        "detector_class_name",
        "bbox",
        "visibility",
        "occlusion_ratio",
    }
)


class YoloDatasetError(ValueError):
    """Raised when the dataset descriptor itself cannot be parsed."""


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    severity: str
    code: str
    message: str
    split: str | None = None
    path: str | None = None
    line: int | None = None

    def __post_init__(self) -> None:
        if self.severity not in {"error", "warning"}:
            raise ValueError("issue severity must be error or warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("severity", self.severity),
                ("code", self.code),
                ("message", self.message),
                ("split", self.split),
                ("path", self.path),
                ("line", self.line),
            )
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    data_yaml: Path
    dataset_root: Path
    class_names: tuple[str, ...]
    split_image_counts: Mapping[str, int]
    split_annotation_counts: Mapping[str, int]
    class_counts: Mapping[str, int]
    target_area_px: Mapping[str, float | int | None]
    target_area_px_by_split: Mapping[
        str, Mapping[str, float | int | None]
    ]
    image_hashes: Mapping[str, tuple[str, ...]]
    empty_label_files: int
    metadata_counts: Mapping[str, int]
    cube_protocol_coverage: Mapping[str, Mapping[str, bool]]
    issues: tuple[DatasetIssue, ...]

    @property
    def errors(self) -> tuple[DatasetIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[DatasetIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def total_images(self) -> int:
        return sum(self.split_image_counts.values())

    @property
    def total_annotations(self) -> int:
        return sum(self.split_annotation_counts.values())

    def to_statistics_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "data_yaml": str(self.data_yaml),
            "dataset_root": str(self.dataset_root),
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "classes": list(self.class_names),
            "split_image_counts": dict(self.split_image_counts),
            "split_annotation_counts": dict(self.split_annotation_counts),
            "class_counts": dict(self.class_counts),
            "target_area_px": dict(self.target_area_px),
            "target_area_px_by_split": {
                split: dict(summary)
                for split, summary in self.target_area_px_by_split.items()
            },
            "empty_label_files": self.empty_label_files,
            "metadata_counts": dict(self.metadata_counts),
            "cube_protocol_coverage": {
                split: dict(coverage)
                for split, coverage in self.cube_protocol_coverage.items()
            },
            "image_hash_summary": {
                split: {"total": len(hashes), "unique": len(set(hashes))}
                for split, hashes in self.image_hashes.items()
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def write_statistics(self, path: str | Path | None = None) -> Path:
        destination = self.dataset_root / "statistics.json" if path is None else Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_statistics_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise YoloDatasetError(f"dataset YAML does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise YoloDatasetError(f"cannot parse dataset YAML {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise YoloDatasetError("dataset YAML root must be a mapping")
    return raw


def _class_names(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        names = tuple(raw)
    elif isinstance(raw, Mapping):
        normalized: dict[int, Any] = {}
        for key, value in raw.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise YoloDatasetError("class name keys must be integer IDs") from exc
            if index < 0 or index in normalized:
                raise YoloDatasetError("class name IDs must be unique and non-negative")
            normalized[index] = value
        if sorted(normalized) != list(range(len(normalized))):
            raise YoloDatasetError("class name IDs must be contiguous from zero")
        names = tuple(normalized[index] for index in range(len(normalized)))
    else:
        raise YoloDatasetError("dataset YAML names must be a list or ID-to-name mapping")
    if not names:
        raise YoloDatasetError("dataset must declare at least one class")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise YoloDatasetError("every class name must be a non-empty string")
    stripped = tuple(name.strip() for name in names)
    if len(set(stripped)) != len(stripped):
        raise YoloDatasetError("class names must be unique")
    return stripped


def _resolve_dataset_root(data_yaml: Path, raw_path: Any) -> Path:
    if raw_path is None:
        return data_yaml.parent.resolve()
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise YoloDatasetError("dataset YAML path must be a non-empty path")
    root = Path(raw_path).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def _resolve_reference(reference: str, dataset_root: Path) -> Path:
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def _images_from_reference(reference: Any, dataset_root: Path) -> list[Path]:
    if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)):
        images: list[Path] = []
        for item in reference:
            images.extend(_images_from_reference(item, dataset_root))
        return images
    if not isinstance(reference, str) or not reference.strip():
        raise YoloDatasetError("split references must be paths or lists of paths")
    path = _resolve_reference(reference, dataset_root)
    if path.is_dir():
        return sorted(
            item.resolve()
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
        )
    if path.is_file() and path.suffix.lower() == ".txt":
        images: list[Path] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise YoloDatasetError(f"cannot read image-list file {path}: {exc}") from exc
        for line in lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            listed = Path(value).expanduser()
            if not listed.is_absolute():
                # Ultralytics resolves image lists relative to the dataset root.
                listed = dataset_root / listed
            images.append(listed.resolve())
        return images
    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
        return [path]
    # Preserve a missing reference as a sentinel so the report can explain it.
    return [path]


def _label_path(image_path: Path, dataset_root: Path, split: str) -> Path:
    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        return dataset_root / "labels" / split / f"{image_path.stem}.txt"
    parts = list(relative.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return (dataset_root / Path(*parts)).with_suffix(".txt")
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def _metadata_path(image_path: Path, dataset_root: Path, split: str) -> Path:
    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        return dataset_root / "metadata" / split / f"{image_path.stem}.json"
    parts = list(relative.parts)
    if "images" in parts:
        parts[parts.index("images")] = "metadata"
        return (dataset_root / Path(*parts)).with_suffix(".json")
    return dataset_root / "metadata" / split / f"{image_path.stem}.json"


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - package is in both supported envs
        raise RuntimeError("Pillow is required to validate image contents") from exc
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(str(exc)) from exc
    if width <= 0 or height <= 0:
        raise ValueError("image has non-positive dimensions")
    return int(width), int(height)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _area_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "small_count": 0,
            "medium_count": 0,
            "large_count": 0,
            "small_fraction": None,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
        }
    ordered = sorted(values)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    small_count = sum(area < 32.0**2 for area in ordered)
    medium_count = sum(32.0**2 <= area < 96.0**2 for area in ordered)
    large_count = len(ordered) - small_count - medium_count
    return {
        "count": len(ordered),
        "small_count": small_count,
        "medium_count": medium_count,
        "large_count": large_count,
        "small_fraction": float(small_count / len(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p90": float(ordered[p90_index]),
    }


class YoloDatasetValidator:
    """Validate images, annotations, distributions, and split leakage."""

    def __init__(
        self,
        *,
        task: str = "detect",
        protocol: str = "generic",
        cube_only: bool | None = None,
    ) -> None:
        normalized = str(task).strip().lower()
        if normalized not in {"detect", "segment"}:
            raise ValueError("task must be detect or segment")
        normalized_protocol = str(protocol).strip().lower()
        if cube_only is not None:
            if not isinstance(cube_only, bool):
                raise TypeError("cube_only must be a bool or None")
            normalized_protocol = _CUBE_PROTOCOL if cube_only else "generic"
        if normalized_protocol not in {"generic", _CUBE_PROTOCOL}:
            raise ValueError("protocol must be generic or cube-v1")
        if normalized_protocol == _CUBE_PROTOCOL and normalized != "detect":
            raise ValueError("cube-v1 supports detection labels only")
        self._task = normalized
        self._protocol = normalized_protocol

    def validate(self, data_yaml: str | Path) -> DatasetValidationReport:
        descriptor = Path(data_yaml).expanduser().resolve()
        raw = _load_yaml(descriptor)
        names = _class_names(raw.get("names"))
        root = _resolve_dataset_root(descriptor, raw.get("path"))
        issues: list[DatasetIssue] = []
        if self._protocol == _CUBE_PROTOCOL and names != (_CUBE_CLASS_NAME,):
            issues.append(
                DatasetIssue(
                    "error",
                    "cube_class_schema",
                    "cube-v1 data.yaml must declare exactly names: {0: cube}",
                    path=str(descriptor),
                )
            )
        split_images: dict[str, list[Path]] = {}
        for split in _SPLITS:
            reference = raw.get(split)
            if reference is None:
                issues.append(
                    DatasetIssue(
                        "error",
                        "missing_split",
                        f"dataset YAML does not declare {split}",
                        split=split,
                        path=str(descriptor),
                    )
                )
                split_images[split] = []
                continue
            try:
                images = _images_from_reference(reference, root)
            except YoloDatasetError as exc:
                issues.append(
                    DatasetIssue(
                        "error",
                        "invalid_split_reference",
                        str(exc),
                        split=split,
                        path=str(descriptor),
                    )
                )
                images = []
            split_images[split] = images
            if not images:
                issues.append(
                    DatasetIssue(
                        "error",
                        "empty_split",
                        f"split {split} contains no images",
                        split=split,
                    )
                )

        split_annotation_counts = {split: 0 for split in _SPLITS}
        class_counts: Counter[int] = Counter()
        target_areas: dict[str, list[float]] = {split: [] for split in _SPLITS}
        image_hashes: dict[str, list[str]] = {split: [] for split in _SPLITS}
        hash_locations: defaultdict[str, list[tuple[str, Path]]] = defaultdict(list)
        empty_labels = 0
        expected_labels: dict[str, set[Path]] = {split: set() for split in _SPLITS}
        image_annotation_counts: dict[tuple[str, Path], int] = {}
        image_annotation_boxes: defaultdict[
            tuple[str, Path], list[tuple[float, float, float, float]]
        ] = defaultdict(list)
        image_dimensions: dict[tuple[str, Path], tuple[int, int]] = {}

        for split, images in split_images.items():
            seen_paths: set[Path] = set()
            for image_path in images:
                image_annotation_counts[(split, image_path.resolve())] = 0
                if image_path in seen_paths:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "duplicate_image_path",
                            "the same image path is listed more than once",
                            split=split,
                            path=str(image_path),
                        )
                    )
                    continue
                seen_paths.add(image_path)
                if not image_path.is_file():
                    issues.append(
                        DatasetIssue(
                            "error",
                            "missing_image",
                            "image file does not exist",
                            split=split,
                            path=str(image_path),
                        )
                    )
                    continue
                if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "unsupported_image_extension",
                            f"unsupported image extension {image_path.suffix!r}",
                            split=split,
                            path=str(image_path),
                        )
                    )
                    continue
                try:
                    width, height = _image_size(image_path)
                    image_hash = _sha256(image_path)
                except (OSError, RuntimeError, ValueError) as exc:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "unreadable_image",
                            f"image cannot be decoded: {exc}",
                            split=split,
                            path=str(image_path),
                        )
                    )
                    continue
                image_hashes[split].append(image_hash)
                hash_locations[image_hash].append((split, image_path))
                image_dimensions[(split, image_path.resolve())] = (width, height)

                label_path = _label_path(image_path, root, split).resolve()
                expected_labels[split].add(label_path)
                if not label_path.is_file():
                    issues.append(
                        DatasetIssue(
                            "error",
                            "missing_label",
                            "image has no matching label file",
                            split=split,
                            path=str(label_path),
                        )
                    )
                    continue
                try:
                    text = label_path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "unreadable_label",
                            f"label cannot be read as UTF-8: {exc}",
                            split=split,
                            path=str(label_path),
                        )
                    )
                    continue
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if not lines:
                    empty_labels += 1
                    issues.append(
                        DatasetIssue(
                            "warning",
                            "empty_label",
                            "empty label represents a background-only image",
                            split=split,
                            path=str(label_path),
                        )
                    )
                    continue
                seen_annotations: set[str] = set()
                for line_number, line in enumerate(lines, start=1):
                    if line in seen_annotations:
                        issues.append(
                            DatasetIssue(
                                "warning",
                                "duplicate_annotation",
                                "duplicate annotation line",
                                split=split,
                                path=str(label_path),
                                line=line_number,
                            )
                        )
                    seen_annotations.add(line)
                    area = self._validate_annotation(
                        line,
                        class_count=len(names),
                        image_width=width,
                        image_height=height,
                        split=split,
                        label_path=label_path,
                        line_number=line_number,
                        issues=issues,
                        class_counts=class_counts,
                    )
                    if area is not None:
                        target_areas[split].append(area)
                        split_annotation_counts[split] += 1
                        image_annotation_counts[(split, image_path.resolve())] += 1
                        if self._protocol == _CUBE_PROTOCOL:
                            center_x, center_y, box_width, box_height = (
                                float(value) for value in line.split()[1:]
                            )
                            image_annotation_boxes[
                                (split, image_path.resolve())
                            ].append(
                                (
                                    (center_x - box_width / 2.0) * width,
                                    (center_y - box_height / 2.0) * height,
                                    (center_x + box_width / 2.0) * width,
                                    (center_y + box_height / 2.0) * height,
                                )
                            )

        self._check_orphan_labels(root, expected_labels, issues)
        self._check_hash_duplicates(hash_locations, issues)
        metadata_counts = {split: 0 for split in _SPLITS}
        cube_protocol_coverage = {
            split: {key: False for key in _CUBE_COVERAGE_KEYS}
            for split in _SPLITS
        }
        if self._protocol == _CUBE_PROTOCOL:
            metadata_counts, cube_protocol_coverage = self._check_cube_metadata(
                root=root,
                split_images=split_images,
                image_annotation_counts=image_annotation_counts,
                image_annotation_boxes=image_annotation_boxes,
                image_dimensions=image_dimensions,
                issues=issues,
            )
        if sum(split_annotation_counts.values()) == 0:
            issues.append(
                DatasetIssue(
                    "error",
                    "no_positive_annotations",
                    "dataset contains no valid positive target annotations",
                    path=str(root),
                )
            )
        named_counts = {
            names[class_id]: int(class_counts.get(class_id, 0))
            for class_id in range(len(names))
        }
        return DatasetValidationReport(
            data_yaml=descriptor,
            dataset_root=root,
            class_names=names,
            split_image_counts={
                split: len({path for path in split_images[split]}) for split in _SPLITS
            },
            split_annotation_counts=split_annotation_counts,
            class_counts=named_counts,
            target_area_px=_area_summary(
                [area for values in target_areas.values() for area in values]
            ),
            target_area_px_by_split={
                split: _area_summary(target_areas[split]) for split in _SPLITS
            },
            image_hashes={
                split: tuple(sorted(hashes)) for split, hashes in image_hashes.items()
            },
            empty_label_files=empty_labels,
            metadata_counts=metadata_counts,
            cube_protocol_coverage=cube_protocol_coverage,
            issues=tuple(issues),
        )

    def _validate_annotation(
        self,
        line: str,
        *,
        class_count: int,
        image_width: int,
        image_height: int,
        split: str,
        label_path: Path,
        line_number: int,
        issues: list[DatasetIssue],
        class_counts: Counter[int],
    ) -> float | None:
        fields = line.split()
        minimum = 5 if self._task == "detect" else 7
        valid_length = len(fields) == 5 if self._task == "detect" else (
            len(fields) >= minimum and (len(fields) - 1) % 2 == 0
        )
        if not valid_length:
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_label_columns",
                    (
                        "detection labels require class_id x_center y_center width height"
                        if self._task == "detect"
                        else (
                            "segmentation labels require class_id followed by at least "
                            "three xy points"
                        )
                    ),
                    split=split,
                    path=str(label_path),
                    line=line_number,
                )
            )
            return None
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            issues.append(
                DatasetIssue(
                    "error",
                    "non_numeric_label",
                    "class ID and coordinates must be numeric",
                    split=split,
                    path=str(label_path),
                    line=line_number,
                )
            )
            return None
        if not math.isfinite(class_value) or any(
            not math.isfinite(value) for value in coordinates
        ):
            issues.append(
                DatasetIssue(
                    "error",
                    "non_finite_label",
                    "class ID and coordinates must be finite",
                    split=split,
                    path=str(label_path),
                    line=line_number,
                )
            )
            return None
        class_id = int(class_value)
        if self._protocol == _CUBE_PROTOCOL and class_value != 0.0:
            issues.append(
                DatasetIssue(
                    "error",
                    "cube_class_id_not_zero",
                    "cube-v1 labels must use detector class ID 0 only",
                    split=split,
                    path=str(label_path),
                    line=line_number,
                )
            )
            return None
        if class_value != class_id or not 0 <= class_id < class_count:
            issues.append(
                DatasetIssue(
                    "error",
                    "class_id_out_of_range",
                    f"class ID must be an integer in [0, {class_count - 1}]",
                    split=split,
                    path=str(label_path),
                    line=line_number,
                )
            )
            return None
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            issues.append(
                DatasetIssue(
                    "error",
                    "coordinate_out_of_range",
                    "all coordinates must be normalized to [0, 1]",
                    split=split,
                    path=str(label_path),
                    line=line_number,
                )
            )
            return None
        if self._task == "detect":
            center_x, center_y, width, height = coordinates
            if width <= 0.0 or height <= 0.0:
                issues.append(
                    DatasetIssue(
                        "error",
                        "non_positive_box",
                        "box width and height must be greater than zero",
                        split=split,
                        path=str(label_path),
                        line=line_number,
                    )
                )
                return None
            if (
                center_x - width / 2.0 < 0.0
                or center_x + width / 2.0 > 1.0
                or center_y - height / 2.0 < 0.0
                or center_y + height / 2.0 > 1.0
            ):
                issues.append(
                    DatasetIssue(
                        "error",
                        "box_outside_image",
                        "normalized box extends outside the image",
                        split=split,
                        path=str(label_path),
                        line=line_number,
                    )
                )
                return None
            area = width * image_width * height * image_height
        else:
            xs = coordinates[0::2]
            ys = coordinates[1::2]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            if width <= 0.0 or height <= 0.0:
                issues.append(
                    DatasetIssue(
                        "error",
                        "non_positive_polygon",
                        "segmentation polygon must span positive width and height",
                        split=split,
                        path=str(label_path),
                        line=line_number,
                    )
                )
                return None
            area = width * image_width * height * image_height
        class_counts[class_id] += 1
        return float(area)

    @staticmethod
    def _metadata_name_has_privileged_token(value: str) -> bool:
        collapsed = re.sub(r"[^a-z0-9]+", "", value.casefold())
        return any(token in collapsed for token in _PRIVILEGED_METADATA_TOKENS)

    @staticmethod
    def _looks_like_encoded_payload(value: str) -> bool:
        stripped = value.strip()
        if _DATA_URI.match(stripped):
            return True
        compact = "".join(stripped.split())
        if len(compact) < 128:
            return False
        if _BASE64_TEXT.fullmatch(compact) is None:
            return False
        padded = compact + "=" * ((4 - len(compact) % 4) % 4)
        try:
            decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
        except (binascii.Error, ValueError):
            return False
        return len(decoded) >= 64

    @classmethod
    def _metadata_has_privileged_pixels(
        cls,
        value: Any,
        *,
        parent_key: str | None = None,
    ) -> bool:
        """Reject pixel, encoded, or model-prompt payloads at any nesting depth."""

        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                if cls._metadata_name_has_privileged_token(key_text):
                    return True
                if cls._metadata_has_privileged_pixels(
                    item,
                    parent_key=key_text.strip().casefold(),
                ):
                    return True
            return False
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            # These are scalar status values emitted by the collector, not
            # embedded depth arrays.  The key remains part of the allow-list.
            if parent_key == "visibility" and normalized in _CUBE_VISIBILITY_STATES:
                return False
            return (
                cls._metadata_name_has_privileged_token(value)
                or cls._looks_like_encoded_payload(value)
            )
        if isinstance(value, Sequence):
            return any(
                cls._metadata_has_privileged_pixels(item, parent_key=parent_key)
                for item in value
            )
        return False

    @staticmethod
    def _valid_metadata_bbox(value: Any) -> bool:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
            return False
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        ):
            return False
        x1, y1, x2, y2 = (float(item) for item in value)
        return x1 >= 0.0 and y1 >= 0.0 and x2 > x1 and y2 > y1

    @classmethod
    def _validate_metadata_object(
        cls,
        raw: Any,
        *,
        split: str,
        metadata_path: Path,
        object_index: int,
        image_width: int,
        image_height: int,
        issues: list[DatasetIssue],
    ) -> tuple[
        bool,
        str | None,
        bool,
        bool,
        tuple[float, float, float, float] | None,
    ]:
        """Return visible-cube, colour, partial, validity, and pixel bbox."""

        if not isinstance(raw, Mapping):
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_cube_metadata_object",
                    f"objects[{object_index}] must be a mapping",
                    split=split,
                    path=str(metadata_path),
                )
            )
            return False, None, False, False, None
        required = set(_CUBE_METADATA_OBJECT_FIELDS)
        missing = sorted(required - set(raw))
        if missing:
            issues.append(
                DatasetIssue(
                    "error",
                    "incomplete_cube_metadata_object",
                    f"objects[{object_index}] is missing: {', '.join(missing)}",
                    split=split,
                    path=str(metadata_path),
                )
            )
            return False, None, False, False, None
        unknown = sorted(set(raw) - _CUBE_METADATA_OBJECT_FIELDS)
        has_unknown_fields = bool(unknown)
        if unknown:
            issues.append(
                DatasetIssue(
                    "error",
                    "unknown_cube_metadata_object_fields",
                    (
                        f"objects[{object_index}] contains unknown fields: "
                        + ", ".join(unknown)
                    ),
                    split=split,
                    path=str(metadata_path),
                )
            )
        object_id = raw.get("object_id")
        shape = raw.get("shape")
        color = raw.get("color_name")
        visibility = raw.get("visibility")
        if not isinstance(object_id, str) or not object_id.strip():
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_metadata_object_id",
                    f"objects[{object_index}].object_id must be non-empty",
                    split=split,
                    path=str(metadata_path),
                )
            )
        if not isinstance(shape, str) or not shape.strip():
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_metadata_shape",
                    f"objects[{object_index}].shape must be non-empty",
                    split=split,
                    path=str(metadata_path),
                )
            )
            return False, None, False, False, None
        shape = shape.strip().lower()
        semantic_valid = not has_unknown_fields
        if not isinstance(color, str) or not color.strip():
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_metadata_color",
                    f"objects[{object_index}].color_name must be non-empty",
                    split=split,
                    path=str(metadata_path),
                )
            )
            color_name: str | None = None
        else:
            color_name = color.strip().lower()
        if not isinstance(visibility, str) or not visibility.strip():
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_metadata_visibility",
                    f"objects[{object_index}].visibility must be non-empty",
                    split=split,
                    path=str(metadata_path),
                )
            )
            visibility_name = ""
        else:
            visibility_name = visibility.strip().lower()
            if visibility_name not in _CUBE_VISIBILITY_STATES:
                issues.append(
                    DatasetIssue(
                        "error",
                        "invalid_metadata_visibility",
                        (
                            f"objects[{object_index}].visibility must be one of: "
                            + ", ".join(sorted(_CUBE_VISIBILITY_STATES))
                        ),
                        split=split,
                        path=str(metadata_path),
                    )
                )
                semantic_valid = False
        occlusion = raw.get("occlusion_ratio")
        if occlusion is not None and (
            isinstance(occlusion, bool)
            or not isinstance(occlusion, (int, float))
            or not math.isfinite(float(occlusion))
            or not 0.0 <= float(occlusion) <= 1.0
        ):
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_metadata_occlusion",
                    f"objects[{object_index}].occlusion_ratio must be null or in [0, 1]",
                    split=split,
                    path=str(metadata_path),
                )
            )
        bbox = raw.get("bbox")
        bbox_xyxy: tuple[float, float, float, float] | None = None
        if bbox is not None:
            if not cls._valid_metadata_bbox(bbox):
                issues.append(
                    DatasetIssue(
                        "error",
                        "invalid_metadata_bbox",
                        f"objects[{object_index}].bbox must be null or valid xyxy pixels",
                        split=split,
                        path=str(metadata_path),
                    )
                )
            else:
                candidate = tuple(float(item) for item in bbox)
                if candidate[2] > image_width or candidate[3] > image_height:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "metadata_bbox_outside_image",
                            (
                                f"objects[{object_index}].bbox must fit inside "
                                f"the {image_width}x{image_height} image"
                            ),
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                else:
                    bbox_xyxy = candidate
        bbox_valid = bbox_xyxy is not None

        is_cube = shape == _CUBE_CLASS_NAME
        visible_cube = (
            is_cube
            and semantic_valid
            and visibility_name in _VISIBLE_POSITIVE_STATES
        )
        if is_cube:
            if color_name not in _CUBE_COLORS:
                issues.append(
                    DatasetIssue(
                        "error",
                        "cube_color_out_of_catalog",
                        (
                            f"objects[{object_index}] cube color must be one of: "
                            + ", ".join(_CUBE_COLORS)
                        ),
                        split=split,
                        path=str(metadata_path),
                    )
                )
                semantic_valid = False
            if (
                raw.get("detector_class_id") != 0
                or raw.get("detector_class_name") != _CUBE_CLASS_NAME
            ):
                issues.append(
                    DatasetIssue(
                        "error",
                        "invalid_cube_metadata_class",
                        f"objects[{object_index}] cube must map exactly to class 0/cube",
                        split=split,
                        path=str(metadata_path),
                    )
                )
                semantic_valid = False
            if visible_cube and not bbox_valid:
                issues.append(
                    DatasetIssue(
                        "error",
                        "visible_cube_without_bbox",
                        f"objects[{object_index}] is a visible cube without a real bbox",
                        split=split,
                        path=str(metadata_path),
                    )
                )
                semantic_valid = False
        elif (
            raw.get("detector_class_id") is not None
            or raw.get("detector_class_name") is not None
        ):
            issues.append(
                DatasetIssue(
                    "error",
                    "non_cube_has_detector_class",
                    f"objects[{object_index}] is a hard negative and must remain unlabelled",
                    split=split,
                    path=str(metadata_path),
                )
            )
            semantic_valid = False
        return (
            visible_cube and bbox_valid and semantic_valid,
            color_name,
            visible_cube and visibility_name == "partially_occluded",
            semantic_valid,
            bbox_xyxy,
        )

    @staticmethod
    def _bboxes_match_one_to_one(
        metadata_boxes: Sequence[tuple[float, float, float, float]],
        label_boxes: Sequence[tuple[float, float, float, float]],
        *,
        image_width: int,
        image_height: int,
    ) -> bool:
        if len(metadata_boxes) != len(label_boxes):
            return False
        tolerance = max(1e-3, max(image_width, image_height) * 1e-7)
        adjacency = [
            [
                label_index
                for label_index, label_box in enumerate(label_boxes)
                if all(
                    abs(metadata_value - label_value) <= tolerance
                    for metadata_value, label_value in zip(metadata_box, label_box)
                )
            ]
            for metadata_box in metadata_boxes
        ]
        matched_metadata_by_label: dict[int, int] = {}

        def assign(metadata_index: int, seen_labels: set[int]) -> bool:
            for label_index in adjacency[metadata_index]:
                if label_index in seen_labels:
                    continue
                seen_labels.add(label_index)
                previous = matched_metadata_by_label.get(label_index)
                if previous is None or assign(previous, seen_labels):
                    matched_metadata_by_label[label_index] = metadata_index
                    return True
            return False

        return all(assign(index, set()) for index in range(len(metadata_boxes)))

    @staticmethod
    def _hard_negative_kind(shape: str, color: str | None) -> str | None:
        if shape == "sphere" and color in {"red", "blue"}:
            return f"{color}_sphere"
        if shape == "cylinder" and color in {"red", "blue"}:
            return f"{color}_cylinder"
        if shape in {"cuboid", "colored_background_block", "partial_noncube"}:
            return shape
        return None

    @classmethod
    def _check_cube_metadata(
        cls,
        *,
        root: Path,
        split_images: Mapping[str, Sequence[Path]],
        image_annotation_counts: Mapping[tuple[str, Path], int],
        image_annotation_boxes: Mapping[
            tuple[str, Path], Sequence[tuple[float, float, float, float]]
        ],
        image_dimensions: Mapping[tuple[str, Path], tuple[int, int]],
        issues: list[DatasetIssue],
    ) -> tuple[dict[str, int], dict[str, dict[str, bool]]]:
        metadata_counts = {split: 0 for split in _SPLITS}
        coverage = {
            split: {key: False for key in _CUBE_COVERAGE_KEYS}
            for split in _SPLITS
        }
        episode_splits: defaultdict[str, set[str]] = defaultdict(set)
        trajectory_splits: defaultdict[str, set[str]] = defaultdict(set)
        expected_metadata: dict[str, set[Path]] = {split: set() for split in _SPLITS}
        cube_colors_seen: set[str] = set()
        hard_negatives_seen: set[str] = set()

        for split in _SPLITS:
            for image_path in dict.fromkeys(split_images.get(split, ())):
                image_path = image_path.resolve()
                if not image_path.is_file():
                    continue
                metadata_path = _metadata_path(image_path, root, split).resolve()
                expected_metadata[split].add(metadata_path)
                dimensions = image_dimensions.get((split, image_path))
                if dimensions is None:
                    # The image decoder has already emitted the authoritative
                    # unreadable-image issue; metadata cannot be checked
                    # against dimensions that were not proven.
                    continue
                image_width, image_height = dimensions
                if not metadata_path.is_file():
                    issues.append(
                        DatasetIssue(
                            "error",
                            "missing_cube_metadata",
                            "cube-v1 image has no matching metadata JSON",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                    continue
                try:
                    raw = strict_json_object_loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, ValueError, TypeError) as exc:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "invalid_cube_metadata",
                            f"metadata is not valid UTF-8 JSON: {exc}",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                    continue
                metadata_counts[split] += 1
                if cls._metadata_has_privileged_pixels(raw):
                    issues.append(
                        DatasetIssue(
                            "error",
                            "metadata_contains_pixels",
                            (
                                "metadata must not embed RGB/depth/image/crop, "
                                "base64, prompt, or opaque payload data"
                            ),
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                required_root = {"schema_version", "episode_id", "trajectory_id", "objects"}
                missing_root = sorted(required_root - set(raw))
                if missing_root:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "incomplete_cube_metadata",
                            "metadata is missing: " + ", ".join(missing_root),
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                    continue
                unknown_root = sorted(set(raw) - _CUBE_METADATA_FIELDS)
                if unknown_root:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "unknown_cube_metadata_fields",
                            "metadata contains unknown fields: "
                            + ", ".join(unknown_root),
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                if raw.get("schema_version") != 1:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "unsupported_cube_metadata_schema",
                            "metadata schema_version must be exactly 1",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                if "split" in raw and raw.get("split") != split:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "metadata_split_mismatch",
                            "metadata split does not match its directory",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                if "frame_id" in raw and raw.get("frame_id") != image_path.stem:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "metadata_frame_mismatch",
                            "metadata frame_id does not match image stem",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                episode_id = raw.get("episode_id")
                trajectory_id = raw.get("trajectory_id")
                if not isinstance(episode_id, str) or not episode_id.strip():
                    issues.append(
                        DatasetIssue(
                            "error",
                            "invalid_metadata_episode",
                            "episode_id must be a non-empty string",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                else:
                    episode_splits[episode_id.strip()].add(split)
                if not isinstance(trajectory_id, str) or not trajectory_id.strip():
                    issues.append(
                        DatasetIssue(
                            "error",
                            "invalid_metadata_trajectory",
                            "trajectory_id must be a non-empty string",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                else:
                    trajectory_splits[trajectory_id.strip()].add(split)
                objects = raw.get("objects")
                if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
                    issues.append(
                        DatasetIssue(
                            "error",
                            "invalid_cube_metadata_objects",
                            "metadata objects must be a list",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                    continue
                object_ids = [
                    item.get("object_id")
                    for item in objects
                    if isinstance(item, Mapping)
                ]
                if len(object_ids) != len(set(str(item) for item in object_ids)):
                    issues.append(
                        DatasetIssue(
                            "error",
                            "duplicate_metadata_object_id",
                            "metadata object_id values must be unique per frame",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                visible_count = 0
                visible_boxes: list[tuple[float, float, float, float]] = []
                cube_count = 0
                has_red = has_blue = has_partial = False
                for index, item in enumerate(objects):
                    if (
                        isinstance(item, Mapping)
                        and str(item.get("shape", "")).strip().lower()
                        == _CUBE_CLASS_NAME
                    ):
                        cube_count += 1
                    (
                        visible,
                        color,
                        partial,
                        semantic_valid,
                        bbox_xyxy,
                    ) = cls._validate_metadata_object(
                        item,
                        split=split,
                        metadata_path=metadata_path,
                        object_index=index,
                        image_width=image_width,
                        image_height=image_height,
                        issues=issues,
                    )
                    visible_count += int(visible)
                    if visible and bbox_xyxy is not None:
                        visible_boxes.append(bbox_xyxy)
                    if semantic_valid and isinstance(item, Mapping):
                        shape = str(item.get("shape", "")).strip().lower()
                        if shape == _CUBE_CLASS_NAME and color in _CUBE_COLORS:
                            assert color is not None
                            cube_colors_seen.add(color)
                        elif shape != _CUBE_CLASS_NAME:
                            kind = cls._hard_negative_kind(shape, color)
                            if kind is not None:
                                hard_negatives_seen.add(kind)
                    has_red = has_red or (visible and color == "red")
                    has_blue = has_blue or (visible and color == "blue")
                    has_partial = has_partial or partial
                if cube_count > 3:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "cube_count_out_of_range",
                            "metadata may contain at most three cubes per image",
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                label_count = image_annotation_counts.get((split, image_path), 0)
                if label_count != visible_count:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "metadata_label_count_mismatch",
                            (
                                f"label count {label_count} does not equal visible "
                                f"cube count {visible_count}"
                            ),
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                elif not cls._bboxes_match_one_to_one(
                    visible_boxes,
                    image_annotation_boxes.get((split, image_path), ()),
                    image_width=image_width,
                    image_height=image_height,
                ):
                    issues.append(
                        DatasetIssue(
                            "error",
                            "metadata_label_bbox_mismatch",
                            (
                                "visible cube metadata bboxes do not match class-0 "
                                "YOLO label bboxes one-to-one"
                            ),
                            split=split,
                            path=str(metadata_path),
                        )
                    )
                coverage[split]["positive"] |= visible_count > 0
                coverage[split]["negative"] |= (
                    cube_count == 0 and visible_count == 0 and label_count == 0
                )
                coverage[split]["red_cube"] |= has_red
                coverage[split]["blue_cube"] |= has_blue
                coverage[split]["partial_occlusion"] |= has_partial
                coverage[split]["multi_cube"] |= visible_count >= 2

            metadata_dir = root / "metadata" / split
            if metadata_dir.is_dir():
                for metadata_path in metadata_dir.rglob("*.json"):
                    if metadata_path.resolve() not in expected_metadata[split]:
                        issues.append(
                            DatasetIssue(
                                "error",
                                "orphan_cube_metadata",
                                "metadata JSON has no matching image in this split",
                                split=split,
                                path=str(metadata_path.resolve()),
                            )
                        )

        missing_colors = [color for color in _CUBE_COLORS if color not in cube_colors_seen]
        if missing_colors:
            issues.append(
                DatasetIssue(
                    "error",
                    "missing_cube_color_coverage",
                    "cube-v1 metadata lacks cube color(s): " + ", ".join(missing_colors),
                    path=str(root),
                )
            )
        missing_hard_negatives = [
            kind for kind in _HARD_NEGATIVE_KINDS if kind not in hard_negatives_seen
        ]
        if missing_hard_negatives:
            issues.append(
                DatasetIssue(
                    "error",
                    "missing_hard_negative_coverage",
                    (
                        "cube-v1 metadata lacks hard-negative kind(s): "
                        + ", ".join(missing_hard_negatives)
                    ),
                    path=str(root),
                )
            )

        for episode_id, splits in episode_splits.items():
            if len(splits) > 1:
                issues.append(
                    DatasetIssue(
                        "error",
                        "episode_split_leakage",
                        f"episode_id {episode_id!r} occurs across {sorted(splits)}",
                        path=episode_id,
                    )
                )
        for trajectory_id, splits in trajectory_splits.items():
            if len(splits) > 1:
                issues.append(
                    DatasetIssue(
                        "error",
                        "trajectory_split_leakage",
                        f"trajectory_id {trajectory_id!r} occurs across {sorted(splits)}",
                        path=trajectory_id,
                    )
                )
        for split, split_coverage in coverage.items():
            missing = [key for key in _CUBE_COVERAGE_KEYS if not split_coverage[key]]
            if missing:
                issues.append(
                    DatasetIssue(
                        "error",
                        "missing_cube_split_coverage",
                        f"split {split} lacks required cube-v1 coverage: {', '.join(missing)}",
                        split=split,
                    )
                )
        return metadata_counts, coverage

    @staticmethod
    def _check_orphan_labels(
        root: Path,
        expected: Mapping[str, set[Path]],
        issues: list[DatasetIssue],
    ) -> None:
        for split in _SPLITS:
            labels_dir = root / "labels" / split
            if not labels_dir.is_dir():
                continue
            for label in labels_dir.rglob("*.txt"):
                if label.resolve() not in expected[split]:
                    issues.append(
                        DatasetIssue(
                            "error",
                            "orphan_label",
                            "label has no matching image in this split",
                            split=split,
                            path=str(label.resolve()),
                        )
                    )

    @staticmethod
    def _check_hash_duplicates(
        locations: Mapping[str, list[tuple[str, Path]]],
        issues: list[DatasetIssue],
    ) -> None:
        for image_hash, entries in locations.items():
            if len(entries) < 2:
                continue
            splits = {split for split, _ in entries}
            paths = ", ".join(str(path) for _, path in entries)
            if len(splits) > 1:
                issues.append(
                    DatasetIssue(
                        "error",
                        "split_hash_leakage",
                        f"identical image bytes occur across {sorted(splits)}: {paths}",
                        path=image_hash,
                    )
                )
            else:
                issues.append(
                    DatasetIssue(
                        "error",
                        "duplicate_image_content",
                        f"duplicate image bytes occur within split {next(iter(splits))}: {paths}",
                        split=next(iter(splits)),
                        path=image_hash,
                    )
                )


def validate_yolo_dataset(
    data_yaml: str | Path,
    *,
    task: str = "detect",
    protocol: str = "generic",
) -> DatasetValidationReport:
    """Convenience wrapper used by CLIs and training backends."""

    return YoloDatasetValidator(task=task, protocol=protocol).validate(data_yaml)


class CubeDatasetValidator(YoloDatasetValidator):
    """Strict validator for the public single-class ``cube-v1`` protocol."""

    def __init__(self) -> None:
        super().__init__(task="detect", protocol=_CUBE_PROTOCOL)


# Descriptive compatibility alias for callers that include the format name.
CubeYoloDatasetValidator = CubeDatasetValidator


def validate_cube_yolo_dataset(data_yaml: str | Path) -> DatasetValidationReport:
    return CubeDatasetValidator().validate(data_yaml)
