from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from env.camera_types import CameraIntrinsics, CameraSample
import training.yolo.isaac_collector as collector_module
from training.yolo.isaac_collector import (
    CollectionLimits,
    DepthVisibilityDecision,
    EpisodeKey,
    IsaacDatasetCollectionError,
    IsaacYoloDatasetCollector,
    OracleFrameTruth,
    RandomizationBounds,
    estimate_depth_visibility,
    project_oracle_bbox,
    require_oracle_label_acknowledgements,
    split_for_episode,
)
from scripts.collect_yolo_dataset import main as collect_main


def _sample() -> CameraSample:
    return CameraSample(
        timestamp_s=1.25,
        rgb=np.full((60, 100, 3), 80, dtype=np.uint8),
        depth_to_image_plane_m=np.full((60, 100), 5.0, dtype=np.float32),
        camera_position_world_m=(1.0, 2.0, 3.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=CameraIntrinsics(
            fx=80.0,
            fy=80.0,
            cx=50.0,
            cy=30.0,
            width=100,
            height=60,
        ),
    )


def _positive_truth(*, occlusion_ratio: float = 0.0) -> OracleFrameTruth:
    return OracleFrameTruth(
        camera_sample=_sample(),
        target_position_world_m=(4.0, 5.0, 0.5),
        target_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        projected_target_pixels_uv=np.asarray(
            [[-10.0, 10.0], [30.0, 10.0], [30.0, 50.0], [-10.0, 50.0]]
        ),
        projected_target_depth_m=np.full(4, 5.0),
        occlusion_ratio=occlusion_ratio,
    )


class _FakeAdapter:
    def __init__(self) -> None:
        self.plans = []
        self.periods = []
        self._plan = None

    def begin_episode(self, randomization) -> None:
        self._plan = randomization
        self.plans.append(randomization)

    def advance_to_next_sample(self, sample_period_s: float) -> None:
        self.periods.append(sample_period_s)

    def capture_oracle_frame(self, frame_id: str) -> OracleFrameTruth:
        del frame_id
        if self._plan.sample_kind == "negative":
            return OracleFrameTruth(
                camera_sample=_sample(),
                target_position_world_m=None,
                target_orientation_world_wxyz=None,
                projected_target_pixels_uv=None,
                projected_target_depth_m=None,
                occlusion_ratio=None,
            )
        return _positive_truth(
            occlusion_ratio=0.4
            if self._plan.sample_kind == "partial_occlusion"
            else 0.0
        )


class _AlwaysNegativeAdapter(_FakeAdapter):
    def capture_oracle_frame(self, frame_id: str) -> OracleFrameTruth:
        del frame_id
        return OracleFrameTruth(
            camera_sample=_sample(),
            target_position_world_m=None,
            target_orientation_world_wxyz=None,
            projected_target_pixels_uv=None,
            projected_target_depth_m=None,
            occlusion_ratio=None,
        )


class IsaacCollectorTest(unittest.TestCase):
    def test_collection_core_has_no_isaac_import(self) -> None:
        source = inspect.getsource(collector_module)
        self.assertNotIn("import isaacsim", source)
        self.assertNotIn("from isaacsim", source)
        self.assertNotIn("from omni", source)
        self.assertNotIn("from pxr", source)

    def test_both_privileged_acknowledgements_are_required(self) -> None:
        for purpose, acknowledgement in ((False, False), (True, False), (False, True)):
            with self.subTest(purpose=purpose, acknowledgement=acknowledgement):
                with self.assertRaises(IsaacDatasetCollectionError):
                    require_oracle_label_acknowledgements(
                        oracle_label_generation=purpose,
                        acknowledge_privileged_oracle=acknowledgement,
                    )
        require_oracle_label_acknowledgements(
            oracle_label_generation=True,
            acknowledge_privileged_oracle=True,
        )

    def test_projection_clips_at_image_edge_and_normalizes(self) -> None:
        decision = project_oracle_bbox(
            _positive_truth(),
            class_id=0,
            min_bbox_area_px=16.0,
        )
        self.assertEqual(decision.visibility, "edge_clipped")
        assert decision.label is not None
        self.assertEqual(decision.label.bbox_xyxy_px, (0.0, 10.0, 30.0, 50.0))
        self.assertAlmostEqual(decision.label.center_x, 0.15)
        self.assertAlmostEqual(decision.label.center_y, 0.5)
        self.assertAlmostEqual(decision.label.width, 0.3)
        self.assertAlmostEqual(decision.label.height, 2.0 / 3.0)

    def test_fully_invisible_and_too_small_targets_do_not_get_labels(self) -> None:
        hidden = _positive_truth(occlusion_ratio=1.0)
        self.assertEqual(
            project_oracle_bbox(hidden, class_id=0, min_bbox_area_px=1.0).visibility,
            "fully_occluded",
        )
        tiny_sample = CameraSample(
            timestamp_s=1.25,
            rgb=np.full((60, 100, 3), 80, dtype=np.uint8),
            depth_to_image_plane_m=np.full((60, 100), 3.0, dtype=np.float32),
            camera_position_world_m=(1.0, 2.0, 3.0),
            camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            intrinsics=_sample().intrinsics,
        )
        tiny = OracleFrameTruth(
            camera_sample=tiny_sample,
            target_position_world_m=(1.0, 1.0, 1.0),
            target_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            projected_target_pixels_uv=np.asarray([[10.0, 10.0], [11.0, 11.0]]),
            projected_target_depth_m=np.asarray([3.0, 3.0]),
            occlusion_ratio=0.0,
        )
        decision = project_oracle_bbox(tiny, class_id=0, min_bbox_area_px=4.0)
        self.assertIsNone(decision.label)
        self.assertEqual(decision.visibility, "too_small")

    def test_synchronized_depth_detects_partial_and_full_occlusion(self) -> None:
        pixels = np.asarray([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
        projected_depth = np.full(4, 5.0)
        depth = np.full((60, 100), 12.0, dtype=np.float32)
        depth[10:30, 10:20] = 3.0
        depth[10:30, 20:30] = 5.0
        sample = CameraSample(
            timestamp_s=1.25,
            rgb=np.full((60, 100, 3), 80, dtype=np.uint8),
            depth_to_image_plane_m=depth,
            camera_position_world_m=(1.0, 2.0, 3.0),
            camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            intrinsics=_sample().intrinsics,
        )
        partial = estimate_depth_visibility(sample, pixels, projected_depth)
        self.assertIsInstance(partial, DepthVisibilityDecision)
        self.assertTrue(partial.trusted)
        self.assertEqual(partial.target_depth_pixels, 200)
        self.assertEqual(partial.occluder_depth_pixels, 200)
        self.assertAlmostEqual(partial.occlusion_ratio, 0.5)
        truth = OracleFrameTruth(
            camera_sample=sample,
            target_position_world_m=(4.0, 5.0, 0.5),
            target_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            projected_target_pixels_uv=pixels,
            projected_target_depth_m=projected_depth,
            occlusion_ratio=0.0,
        )
        projected = project_oracle_bbox(truth, class_id=0, min_bbox_area_px=1.0)
        self.assertEqual(projected.visibility, "partially_occluded")
        self.assertAlmostEqual(projected.occlusion_ratio or 0.0, 0.5)

        fully_occluded_depth = depth.copy()
        fully_occluded_depth[10:30, 10:30] = 3.0
        fully_occluded_sample = CameraSample(
            timestamp_s=1.25,
            rgb=np.full((60, 100, 3), 80, dtype=np.uint8),
            depth_to_image_plane_m=fully_occluded_depth,
            camera_position_world_m=(1.0, 2.0, 3.0),
            camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            intrinsics=_sample().intrinsics,
        )
        fully_occluded = estimate_depth_visibility(
            fully_occluded_sample,
            pixels,
            projected_depth,
        )
        self.assertTrue(fully_occluded.trusted)
        self.assertEqual(fully_occluded.reason, "fully_occluded")
        self.assertEqual(fully_occluded.occlusion_ratio, 1.0)

    def test_missing_invalid_or_unexplained_depth_fails_closed(self) -> None:
        pixels = np.asarray([[10.0, 10.0], [30.0, 30.0]])
        projected_depth = np.asarray([5.0, 5.0])
        for depth, expected_reason in (
            (None, "depth_unavailable"),
            (np.full((60, 100), np.nan, dtype=np.float32), "invalid_depth"),
            (np.full((60, 100), 12.0, dtype=np.float32), "depth_unresolved"),
        ):
            with self.subTest(reason=expected_reason):
                sample = CameraSample(
                    timestamp_s=1.25,
                    rgb=np.full((60, 100, 3), 80, dtype=np.uint8),
                    depth_to_image_plane_m=depth,
                    camera_position_world_m=(1.0, 2.0, 3.0),
                    camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
                    intrinsics=_sample().intrinsics,
                )
                visibility = estimate_depth_visibility(sample, pixels, projected_depth)
                self.assertFalse(visibility.trusted)
                self.assertEqual(visibility.reason, expected_reason)
                truth = OracleFrameTruth(
                    camera_sample=sample,
                    target_position_world_m=(4.0, 5.0, 0.5),
                    target_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
                    projected_target_pixels_uv=pixels,
                    projected_target_depth_m=projected_depth,
                    occlusion_ratio=0.0,
                )
                decision = project_oracle_bbox(truth, class_id=0, min_bbox_area_px=1.0)
                self.assertIsNone(decision.label)
                self.assertEqual(decision.visibility, expected_reason)

    def test_scene_episode_trajectory_group_has_one_stable_split(self) -> None:
        limits = CollectionLimits()
        key = EpisodeKey(7, "episode_1", "trajectory_1")
        self.assertEqual(split_for_episode(key, limits), split_for_episode(key, limits))

    def test_bounded_collection_writes_standard_labels_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            limits = CollectionLimits(
                max_samples=12,
                max_episodes=12,
                frames_per_episode=1,
                sample_hz=2.0,
            )
            collector = IsaacYoloDatasetCollector(
                output_dir=output,
                class_names=("moving_target",),
                class_id=0,
                limits=limits,
                bounds=RandomizationBounds(),
                scene_seed=42,
                oracle_label_generation=True,
                acknowledge_privileged_oracle=True,
            )
            adapter = _FakeAdapter()
            summary = collector.collect(adapter)

            self.assertEqual(summary.total_samples, 12)
            self.assertEqual(len(adapter.periods), 12)
            self.assertTrue(all(period == 0.5 for period in adapter.periods))
            records = [
                json.loads(line)
                for line in summary.manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 12)
            self.assertTrue(
                all(
                    {
                        "scene_seed",
                        "episode_id",
                        "trajectory_id",
                        "frame_id",
                        "camera_pose",
                        "target_pose",
                        "visibility",
                        "occlusion_ratio",
                        "class_id",
                    }.issubset(record)
                    for record in records
                )
            )
            episode_splits = {}
            for record in records:
                episode_splits.setdefault(record["episode_id"], set()).add(record["split"])
            self.assertTrue(all(len(splits) == 1 for splits in episode_splits.values()))
            self.assertTrue(all(count > 0 for count in summary.split_counts.values()))
            self.assertGreater(summary.positive_samples, 0)
            self.assertGreater(summary.partial_occlusion_samples, 0)
            self.assertGreater(summary.negative_samples, 0)
            self.assertTrue((output / "data.yaml").is_file())
            self.assertEqual(len(list(output.glob("images/*/*.jpg"))), 12)
            self.assertEqual(len(list(output.glob("labels/*/*.txt"))), 12)

    def test_finalize_rejects_empty_grouped_split_with_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            collector = IsaacYoloDatasetCollector(
                output_dir=Path(temporary) / "dataset",
                class_names=("moving_target",),
                class_id=0,
                limits=CollectionLimits(
                    max_samples=2,
                    max_episodes=2,
                    frames_per_episode=1,
                ),
                bounds=RandomizationBounds(),
                scene_seed=11,
                oracle_label_generation=True,
                acknowledge_privileged_oracle=True,
            )
            with self.assertRaisesRegex(
                IsaacDatasetCollectionError,
                r"split\(s\).*empty.*--max-episodes.*--scene-seed",
            ):
                collector.collect(_FakeAdapter())

    def test_finalize_requires_all_three_realized_sample_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            collector = IsaacYoloDatasetCollector(
                output_dir=Path(temporary) / "dataset",
                class_names=("red_cube",),
                class_id=0,
                limits=CollectionLimits(
                    max_samples=12,
                    max_episodes=12,
                    frames_per_episode=1,
                ),
                bounds=RandomizationBounds(),
                scene_seed=42,
                oracle_label_generation=True,
                acknowledge_privileged_oracle=True,
            )
            with self.assertRaisesRegex(
                IsaacDatasetCollectionError,
                "missing required realized sample kind.*positive.*partially_occluded",
            ):
                collector.collect(_AlwaysNegativeAdapter())

    def test_fixed_cube_collector_rejects_arbitrary_class_name_before_isaac(self) -> None:
        result = collect_main(
            [
                "--config",
                "/does/not/need/to/exist.yaml",
                "--output",
                "/tmp/not-created",
                "--class-name",
                "person",
                "--oracle-label-generation",
                "--acknowledge-privileged-oracle",
                "--validate-only",
            ]
        )
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
