from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from env.camera_types import CameraIntrinsics, CameraSample
from perception.class_aliases import ClassAliasMapper, UnsupportedTargetCategory, compile_target_query
from target import TargetSpec
from training.yolo.collection_scene import (
    CUBE_COLORS,
    HARD_NEGATIVE_KINDS,
    CollectionSceneObject,
    load_cube_collection_protocol,
    oriented_box_corners_world,
    validate_scene_inventory,
)
from training.yolo.config import load_yolo_train_config
from training.yolo.isaac_collector import (
    CollectionLimits,
    IsaacYoloDatasetCollector,
    OracleFrameTruth,
    OracleObjectTruth,
    RandomizationBounds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _sample(value: int = 80) -> CameraSample:
    return CameraSample(
        timestamp_s=1.0,
        rgb=np.full((64, 96, 3), value, dtype=np.uint8),
        depth_to_image_plane_m=np.full((64, 96), 5.0, dtype=np.float32),
        camera_position_world_m=(0.0, 0.0, 4.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=CameraIntrinsics(
            fx=80.0,
            fy=80.0,
            cx=48.0,
            cy=32.0,
            width=96,
            height=64,
        ),
    )


def _truth(
    object_id: str,
    shape: str,
    color: str,
    pixels: tuple[tuple[float, float], ...],
    *,
    occlusion: float = 0.0,
    dimensions: tuple[float, float, float] = (0.8, 0.8, 0.8),
) -> OracleObjectTruth:
    return OracleObjectTruth(
        object_id=object_id,
        shape=shape,
        color_name=color,
        position_world_m=(1.0, 2.0, 0.5),
        orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        dimensions_xyz_m=dimensions,
        projected_pixels_uv=np.asarray(pixels, dtype=np.float64),
        projected_depth_m=np.full(len(pixels), 5.0, dtype=np.float64),
        occlusion_ratio=occlusion,
    )


class _MultiObjectAdapter:
    def __init__(self) -> None:
        self.plan = None

    def begin_episode(self, randomization) -> None:
        self.plan = randomization

    def advance_to_next_sample(self, sample_period_s: float) -> None:
        assert sample_period_s > 0.0

    def capture_oracle_frame(self, frame_id: str) -> OracleFrameTruth:
        assert frame_id
        assert self.plan is not None
        episode_index = int(self.plan.key.episode_id.rsplit("_", 1)[-1])
        sample = _sample(20 + episode_index)
        if self.plan.sample_kind == "negative":
            return OracleFrameTruth(
                camera_sample=sample,
                objects=(
                    _truth(
                        "red_sphere_0",
                        "sphere",
                        "red",
                        ((8.0, 8.0), (20.0, 20.0)),
                    ),
                    _truth("blue_sphere_0", "sphere", "blue", ((9.0, 9.0), (21.0, 21.0))),
                    _truth("red_cylinder_0", "cylinder", "red", ((10.0, 10.0), (22.0, 22.0))),
                    _truth("blue_cylinder_0", "cylinder", "blue", ((11.0, 11.0), (23.0, 23.0))),
                    _truth("cuboid_0", "cuboid", "gray", ((12.0, 12.0), (24.0, 24.0))),
                    _truth(
                        "background_0",
                        "colored_background_block",
                        "yellow",
                        ((13.0, 13.0), (25.0, 25.0)),
                    ),
                    _truth(
                        "partial_noncube_0",
                        "partial_noncube",
                        "green",
                        ((14.0, 14.0), (26.0, 26.0)),
                    ),
                ),
            )
        occlusion = 0.4 if self.plan.sample_kind == "partial_occlusion" else 0.0
        third_color = ("green", "yellow", "gray")[(episode_index // 3) % 3]
        return OracleFrameTruth(
            camera_sample=sample,
            objects=(
                _truth(
                    "routing_target_i",
                    "cube",
                    "red",
                    ((8.0, 8.0), (28.0, 28.0)),
                    occlusion=occlusion,
                    dimensions=(0.6, 0.6, 0.6),
                ),
                _truth(
                    "routing_target_j",
                    "cube",
                    "blue",
                    ((55.0, 30.0), (80.0, 55.0)),
                    dimensions=(1.1, 1.1, 1.1),
                ),
                _truth(
                    "routing_target_k",
                    "cube",
                    third_color,
                    ((34.0, 12.0), (50.0, 28.0)),
                    occlusion=occlusion,
                    dimensions=(0.9, 0.9, 0.9),
                ),
            ),
        )


def test_committed_cube_contract_and_exact_aliases_are_closed_set() -> None:
    protocol = load_cube_collection_protocol(
        PROJECT_ROOT / "configs/yolo/collect_cube.yaml"
    )
    assert protocol.cube_count_min == 0
    assert protocol.cube_count_max == 3
    assert protocol.cube_colors == CUBE_COLORS
    assert protocol.hard_negatives == HARD_NEGATIVE_KINDS
    train_config = load_yolo_train_config(
        PROJECT_ROOT / "configs/yolo/train_yolo26s_cube.yaml",
        environ={},
    )
    assert train_config.dataset_protocol == "cube-v1"
    assert train_config.run_name == "yolo26s_cube"

    mapper = ClassAliasMapper.from_yaml(
        PROJECT_ROOT / "configs/yolo/class_aliases.yaml"
    )
    red = compile_target_query(
        TargetSpec("red cube", category="cube", hard_attributes=("color=red",)),
        "yolo",
        {7: "cube"},
        mapper,
    )
    blue = compile_target_query(
        TargetSpec("blue cube", category="cube", hard_attributes=("color=blue",)),
        "yolo",
        {7: "cube"},
        mapper,
    )
    assert red.class_ids == blue.class_ids == (7,)
    assert red.text_prompts == blue.text_prompts == ()
    with np.testing.assert_raises(UnsupportedTargetCategory):
        mapper.resolve("cube", {0: "person"})
    with np.testing.assert_raises(UnsupportedTargetCategory):
        mapper.resolve("red cube", {0: "cube"})


def test_real_object_dimensions_drive_box_corners_and_inventory_is_bounded() -> None:
    small = CollectionSceneObject(
        "cube_0",
        "cube",
        "red",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.4, 0.8, 1.2),
    )
    corners = oriented_box_corners_world(small)
    assert np.allclose(np.ptp(corners, axis=0), (0.4, 0.8, 1.2))
    validate_scene_inventory(
        (
            small,
            CollectionSceneObject(
                "cube_1", "cube", "blue", (2.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
            ),
            CollectionSceneObject(
                "sphere_0", "sphere", "red", (4.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
            ),
        )
    )


def test_collector_labels_every_cube_as_class_zero_and_writes_safe_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cube_dataset"
    collector = IsaacYoloDatasetCollector(
        output_dir=output,
        class_names=("cube",),
        class_id=0,
        limits=CollectionLimits(
            max_samples=22,
            max_episodes=22,
            frames_per_episode=1,
        ),
        bounds=RandomizationBounds(),
        scene_seed=42,
        oracle_label_generation=True,
        acknowledge_privileged_oracle=True,
        cube_protocol=True,
    )
    summary = collector.collect(_MultiObjectAdapter())

    assert summary.total_samples == 22
    descriptor = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert descriptor["names"] == {0: "cube"}
    assert len(list(output.glob("metadata/*/*.json"))) == 22
    assert (output / "statistics.json").is_file()
    for label_path in output.glob("labels/*/*.txt"):
        assert all(line.split()[0] == "0" for line in label_path.read_text().splitlines())
    for metadata_path in output.glob("metadata/*/*.json"):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        rendered = json.dumps(payload).lower()
        assert '"rgb"' not in rendered
        assert '"depth"' not in rendered
        for obj in payload["objects"]:
            if obj["shape"] == "cube":
                assert obj["detector_class_id"] == 0
                assert obj["detector_class_name"] == "cube"
            else:
                assert obj["detector_class_id"] is None
                assert obj["detector_class_name"] is None
