"""Read-only validation of Ultralytics-format YOLO datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import yaml


_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_SPLITS = ("train", "val", "test")


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

    def __init__(self, *, task: str = "detect") -> None:
        normalized = str(task).strip().lower()
        if normalized not in {"detect", "segment"}:
            raise ValueError("task must be detect or segment")
        self._task = normalized

    def validate(self, data_yaml: str | Path) -> DatasetValidationReport:
        descriptor = Path(data_yaml).expanduser().resolve()
        raw = _load_yaml(descriptor)
        names = _class_names(raw.get("names"))
        root = _resolve_dataset_root(descriptor, raw.get("path"))
        issues: list[DatasetIssue] = []
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

        for split, images in split_images.items():
            seen_paths: set[Path] = set()
            for image_path in images:
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

        self._check_orphan_labels(root, expected_labels, issues)
        self._check_hash_duplicates(hash_locations, issues)
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
) -> DatasetValidationReport:
    """Convenience wrapper used by CLIs and training backends."""

    return YoloDatasetValidator(task=task).validate(data_yaml)
