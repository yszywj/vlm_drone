from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest
import yaml

from training.yolo.dataset import CubeDatasetValidator, YoloDatasetValidator


_VISIBLE_STATES = {"visible", "edge_clipped", "partially_occluded"}


def _object(
    object_id: str,
    *,
    shape: str = "cube",
    color: str = "red",
    visibility: str = "visible",
    bbox: list[float] | None = None,
) -> dict[str, object]:
    positive = shape == "cube"
    return {
        "object_id": object_id,
        "shape": shape,
        "color_name": color,
        "detector_class_id": 0 if positive else None,
        "detector_class_name": "cube" if positive else None,
        "bbox": [2.0, 2.0, 12.0, 12.0] if bbox is None else bbox,
        "visibility": visibility,
        "occlusion_ratio": 0.4 if visibility == "partially_occluded" else 0.0,
    }


def _write_frame(
    root: Path,
    split: str,
    stem: str,
    *,
    color: tuple[int, int, int],
    objects: list[dict[str, object]],
    labels: int,
) -> None:
    width, height = 24, 20
    Image.new("RGB", (width, height), color=color).save(
        root / "images" / split / f"{stem}.png"
    )
    visible_cubes = [
        item
        for item in objects
        if item["shape"] == "cube" and item["visibility"] in _VISIBLE_STATES
    ]
    assert len(visible_cubes) >= labels
    label_lines: list[str] = []
    for item in visible_cubes[:labels]:
        x1, y1, x2, y2 = (float(value) for value in item["bbox"])
        label_lines.append(
            (
                f"0 {(x1 + x2) / (2.0 * width):.8f} "
                f"{(y1 + y2) / (2.0 * height):.8f} "
                f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}\n"
            )
        )
    (root / "labels" / split / f"{stem}.txt").write_text(
        "".join(label_lines),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "split": split,
        "scene_seed": {"train": 1, "val": 2, "test": 3}[split],
        "episode_id": f"{split}_episode_{stem}",
        "trajectory_id": f"{split}_trajectory_{stem}",
        "frame_id": stem,
        "objects": objects,
    }
    (root / "metadata" / split / f"{stem}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels", "metadata"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "cube"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    color_index = 0
    for split in ("train", "val", "test"):
        scenarios = (
            ("red", [_object(f"{split}_red", color="red")], 1),
            ("blue", [_object(f"{split}_blue", color="blue")], 1),
            (
                "negative",
                [
                    _object(f"{split}_red_sphere", shape="sphere", color="red"),
                    _object(f"{split}_blue_sphere", shape="sphere", color="blue"),
                    _object(f"{split}_red_cylinder", shape="cylinder", color="red"),
                    _object(f"{split}_blue_cylinder", shape="cylinder", color="blue"),
                    _object(f"{split}_cuboid", shape="cuboid", color="gray"),
                    _object(
                        f"{split}_background",
                        shape="colored_background_block",
                        color="yellow",
                    ),
                    _object(
                        f"{split}_partial_noncube",
                        shape="partial_noncube",
                        color="green",
                    ),
                ],
                0,
            ),
            (
                "partial",
                [
                    _object(
                        f"{split}_partial",
                        color="green",
                        visibility="partially_occluded",
                    )
                ],
                1,
            ),
            (
                "multi",
                [
                    _object(
                        f"{split}_multi_yellow",
                        color="yellow",
                        bbox=[1.0, 2.0, 9.0, 12.0],
                    ),
                    _object(
                        f"{split}_multi_gray",
                        color="gray",
                        bbox=[13.0, 3.0, 22.0, 15.0],
                    ),
                ],
                2,
            ),
        )
        for stem, objects, labels in scenarios:
            color_index += 1
            _write_frame(
                root,
                split,
                stem,
                color=(color_index, 50 + color_index, 100 + color_index),
                objects=objects,
                labels=labels,
            )
    return data_yaml


def _codes(report) -> set[str]:
    return {issue.code for issue in report.errors}


def test_cube_validator_accepts_complete_single_class_protocol(tmp_path: Path) -> None:
    report = CubeDatasetValidator().validate(_dataset(tmp_path))

    assert report.ok, report.errors
    assert report.class_names == ("cube",)
    assert report.metadata_counts == {"train": 5, "val": 5, "test": 5}
    assert all(all(coverage.values()) for coverage in report.cube_protocol_coverage.values())
    assert report.class_counts == {"cube": 15}


def test_cube_validator_rejects_non_cube_schema_and_nonzero_class(tmp_path: Path) -> None:
    data_yaml = _dataset(tmp_path)
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    payload["names"] = {0: "cube", 1: "red_cube"}
    data_yaml.write_text(yaml.safe_dump(payload), encoding="utf-8")
    (tmp_path / "labels/train/red.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    codes = _codes(CubeDatasetValidator().validate(data_yaml))
    assert "cube_class_schema" in codes
    assert "cube_class_id_not_zero" in codes


def test_cube_validator_detects_unlabelled_cube_and_embedded_depth(tmp_path: Path) -> None:
    data_yaml = _dataset(tmp_path)
    (tmp_path / "labels/train/red.txt").write_text("", encoding="utf-8")
    metadata_path = tmp_path / "metadata/train/red.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["depth"] = [[1.0]]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    codes = _codes(CubeDatasetValidator().validate(data_yaml))
    assert "metadata_label_count_mismatch" in codes
    assert "metadata_contains_pixels" in codes


def test_cube_validator_rejects_unknown_visibility(tmp_path: Path) -> None:
    data_yaml = _dataset(tmp_path)
    metadata_path = tmp_path / "metadata/train/red.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["objects"][0]["visibility"] = "probably_visible"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    codes = _codes(CubeDatasetValidator().validate(data_yaml))
    assert "invalid_metadata_visibility" in codes


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_cube_validator_rejects_non_finite_json_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    data_yaml = _dataset(tmp_path)
    metadata_path = tmp_path / "metadata/train/red.json"
    raw = metadata_path.read_text(encoding="utf-8")
    raw = raw.replace('"scene_seed": 1', f'"scene_seed": {constant}', 1)
    metadata_path.write_text(raw, encoding="utf-8")

    assert "invalid_cube_metadata" in _codes(
        CubeDatasetValidator().validate(data_yaml)
    )


def test_cube_validator_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    data_yaml = _dataset(tmp_path)
    metadata_path = tmp_path / "metadata/train/red.json"
    raw = metadata_path.read_text(encoding="utf-8")
    raw = raw.replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )
    metadata_path.write_text(raw, encoding="utf-8")

    assert "invalid_cube_metadata" in _codes(
        CubeDatasetValidator().validate(data_yaml)
    )


def test_cube_validator_rejects_unknown_root_and_object_fields(
    tmp_path: Path,
) -> None:
    data_yaml = _dataset(tmp_path)

    root_path = tmp_path / "metadata/train/red.json"
    root_payload = json.loads(root_path.read_text(encoding="utf-8"))
    root_payload["unexpected"] = "value"
    root_path.write_text(json.dumps(root_payload), encoding="utf-8")

    object_path = tmp_path / "metadata/val/red.json"
    object_payload = json.loads(object_path.read_text(encoding="utf-8"))
    object_payload["objects"][0]["unexpected"] = "value"
    object_path.write_text(json.dumps(object_payload), encoding="utf-8")

    codes = _codes(CubeDatasetValidator().validate(data_yaml))
    assert "unknown_cube_metadata_fields" in codes
    assert "unknown_cube_metadata_object_fields" in codes


def test_cube_validator_rejects_metadata_bbox_outside_image(tmp_path: Path) -> None:
    data_yaml = _dataset(tmp_path)
    metadata_path = tmp_path / "metadata/train/red.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["objects"][0]["bbox"] = [2.0, 2.0, 25.0, 12.0]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    codes = _codes(CubeDatasetValidator().validate(data_yaml))
    assert "metadata_bbox_outside_image" in codes


def test_cube_validator_rejects_equal_count_but_mismatched_bboxes(
    tmp_path: Path,
) -> None:
    data_yaml = _dataset(tmp_path)
    metadata_path = tmp_path / "metadata/train/red.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["objects"][0]["bbox"] = [10.0, 3.0, 18.0, 14.0]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    codes = _codes(CubeDatasetValidator().validate(data_yaml))
    assert "metadata_label_count_mismatch" not in codes
    assert "metadata_label_bbox_mismatch" in codes


def test_cube_validator_bbox_matching_is_order_independent(tmp_path: Path) -> None:
    data_yaml = _dataset(tmp_path)
    label_path = tmp_path / "labels/train/multi.txt"
    lines = label_path.read_text(encoding="utf-8").splitlines(keepends=True)
    label_path.write_text("".join(reversed(lines)), encoding="utf-8")

    report = CubeDatasetValidator().validate(data_yaml)
    assert report.ok, report.errors


@pytest.mark.parametrize(
    "nested_payload",
    [
        {"audit": {"camera_depth": [[1.0]]}},
        {"audit": {"rgbPayload": "AA=="}},
        {"audit": {"description": "rgbPayload"}},
        {"audit": {"description": "prompt"}},
        {"audit": {"encoded": "data:image/png;base64,AA=="}},
        {
            "audit": {
                "encoded": base64.b64encode(b"pixel payload" * 20).decode("ascii")
            }
        },
        {
            "audit": {
                "encoded": base64.urlsafe_b64encode(b"raw crop" * 24)
                .decode("ascii")
                .rstrip("=")
            }
        },
    ],
)
def test_cube_validator_recursively_rejects_privileged_payloads(
    tmp_path: Path,
    nested_payload: dict[str, object],
) -> None:
    data_yaml = _dataset(tmp_path)
    metadata_path = tmp_path / "metadata/train/red.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload.update(nested_payload)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    assert "metadata_contains_pixels" in _codes(
        CubeDatasetValidator().validate(data_yaml)
    )


def test_generic_validator_does_not_apply_cube_metadata_payload_policy(
    tmp_path: Path,
) -> None:
    data_yaml = _dataset(tmp_path)
    metadata_path = tmp_path / "metadata/train/red.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["camera_depth"] = [[1.0]]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    report = YoloDatasetValidator().validate(data_yaml)
    assert report.ok, report.errors


def test_cube_validator_requires_all_colors_and_hard_negative_kinds(
    tmp_path: Path,
) -> None:
    data_yaml = _dataset(tmp_path)
    for split in ("train", "val", "test"):
        partial_path = tmp_path / f"metadata/{split}/partial.json"
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        partial["objects"][0]["color_name"] = "red"
        partial_path.write_text(json.dumps(partial), encoding="utf-8")

        multi_path = tmp_path / f"metadata/{split}/multi.json"
        multi = json.loads(multi_path.read_text(encoding="utf-8"))
        multi["objects"][0]["color_name"] = "red"
        multi["objects"][1]["color_name"] = "blue"
        multi_path.write_text(json.dumps(multi), encoding="utf-8")

        negative_path = tmp_path / f"metadata/{split}/negative.json"
        negative = json.loads(negative_path.read_text(encoding="utf-8"))
        negative["objects"] = [negative["objects"][0]]
        negative_path.write_text(json.dumps(negative), encoding="utf-8")

    codes = _codes(CubeDatasetValidator().validate(data_yaml))
    assert "missing_cube_color_coverage" in codes
    assert "missing_hard_negative_coverage" in codes


def test_cube_validator_detects_hash_episode_and_trajectory_leakage(tmp_path: Path) -> None:
    data_yaml = _dataset(tmp_path)
    shutil.copyfile(
        tmp_path / "images/train/red.png",
        tmp_path / "images/val/red.png",
    )
    train_metadata = json.loads(
        (tmp_path / "metadata/train/red.json").read_text(encoding="utf-8")
    )
    val_path = tmp_path / "metadata/val/red.json"
    val_metadata = json.loads(val_path.read_text(encoding="utf-8"))
    val_metadata["episode_id"] = train_metadata["episode_id"]
    val_metadata["trajectory_id"] = train_metadata["trajectory_id"]
    val_path.write_text(json.dumps(val_metadata), encoding="utf-8")

    codes = _codes(CubeDatasetValidator().validate(data_yaml))
    assert "split_hash_leakage" in codes
    assert "episode_split_leakage" in codes
    assert "trajectory_split_leakage" in codes
