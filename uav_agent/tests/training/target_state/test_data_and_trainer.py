from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from datasets.target_state.dataset import (
    compute_dataset_sha256,
    read_frame_records,
    split_for_episode,
)
from datasets.target_state.schema import (
    CameraFrameInput,
    DetectorPrediction,
    SensorInput,
    TargetStateFrameRecord,
    TargetTrainingLabel,
    UavFrameInput,
)
from training.target_state.config import TargetStateTrainingConfig, TrainingStage, load_training_config
from training.target_state.collector import (
    TargetStateCollectionError,
    TargetStateDatasetWriter,
    require_privileged_collection_acknowledgements,
)
from training.target_state.data import GEOMETRY_INPUT_FIELDS, TargetStateTorchDataset
from training.target_state.data import (
    _crop_rgbd,
    _foreground_cluster_anchor_depth,
    _relative_camera_pose as training_relative_camera_pose,
)
from training.target_state.trainer import (
    TargetStateEvaluationAccumulator,
    TargetStateTrainingError,
    _invalid_model_output_mask,
    accumulate_evaluation,
    evaluate_model,
    evaluate_promotion,
    sha256_file,
    train_target_state,
    validate_initial_checkpoint,
)
from training.target_state.model import TemporalRayDepthOutput
from training.target_state.losses import compute_target_state_losses
from perception.depth_geometry import DepthCandidateResolver
from perception.temporal_ray_depth import (
    _relative_camera_pose as production_relative_camera_pose,
)
from env.camera_types import CameraIntrinsics
from runtime.frame_store import FrameCameraGeometry, FrameStore


def _episode_for(split: str) -> str:
    for index in range(10000):
        value = f"episode_{split}_{index}"
        if split_for_episode(value, seed=42) == split:
            return value
    raise AssertionError(f"no episode found for {split}")


def _record(index: int, episode: str, prefix: str, *, missed: bool = False) -> TargetStateFrameRecord:
    return TargetStateFrameRecord(
        frame_id=f"frame_{prefix}_{index}",
        episode_id=episode,
        assignment_id=f"assignment_{prefix}",
        uav_id="uav_1",
        timestamp_s=index * 0.2,
        sensor_input=SensorInput(
            camera=CameraFrameInput(
                fx=24.0, fy=24.0, cx=15.5, cy=11.5,
                position_world_m=(0.0, 0.0, 0.0),
                orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
                resolution_wh_px=(32, 24),
            ),
            uav=UavFrameInput(
                position_world_m=(0.0, 0.0, 0.0),
                orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
                linear_velocity_world_mps=(0.1, 0.0, 0.0),
                angular_velocity_body_radps=(0.0, 0.0, 0.01),
            ),
            rgb_path=f"rgb/{prefix}_{index}.png",
            depth_path=f"depth/{prefix}_{index}.npy",
        ),
        detector_prediction=DetectorPrediction(
            detected=not missed,
            bbox_xyxy_normalized=None if missed else (0.25, 0.25, 0.75, 0.75),
            confidence=None if missed else 0.8,
            tracker_id=None if missed else "tracker_1",
            candidate_id="candidate_1",
        ),
        training_label=TargetTrainingLabel(
            position_world_m=(5.0, 0.0, 0.0),
            velocity_world_mps=(0.0, 0.0, 0.0),
            center_pixel_uv=(15.5, 11.5), visible=True,
            occlusion_ratio=0.3 if index == 6 else 0.0,
            color_name="red",
        ),
    )


def _dataset(root: Path) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    records = []
    for split in ("train", "validation", "test"):
        episode = _episode_for(split)
        for index in range(7):
            record = _record(index, episode, split, missed=index == 2)
            records.append(record)
            Image.fromarray(np.full((24, 32, 3), 80 + index, dtype=np.uint8)).save(
                root / record.sensor_input.rgb_path
            )
            np.save(root / record.sensor_input.depth_path, np.full((24, 32), 5.0, dtype=np.float32))
    (root / "frames.jsonl").write_text(
        "".join(json.dumps(item.to_dict()) + "\n" for item in records), encoding="utf-8"
    )


def _config(root: Path, output: Path, *, stage: TrainingStage = TrainingStage.YOLO_DEPLOYMENT) -> TargetStateTrainingConfig:
    return TargetStateTrainingConfig(
        dataset_root=root,
        output_dir=output,
        stage=stage,
        history_size=6,
        roi_size_px=32,
        roi_feature_dim=8,
        geometry_feature_dim=8,
        hidden_dim=8,
        gru_layers=1,
        epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        run_name="unit_run",
        save_figures=0,
        promotion_min_covariance_correlation=-1.0,
        require_dataset_manifest=False,
    )


class _FixedEvaluationModel(torch.nn.Module):
    def __init__(self, *, validity_logit: float) -> None:
        super().__init__()
        self.validity_logit = validity_logit

    def forward(self, roi_rgbd, geometry, missing_mask):
        batch_size = roi_rgbd.shape[0]
        device = roi_rgbd.device
        return TemporalRayDepthOutput(
            delta_uv_px=torch.zeros(batch_size, 2, device=device),
            depth_residual_m=torch.zeros(batch_size, device=device),
            position_log_variance=torch.zeros(batch_size, 3, device=device),
            measurement_valid_logit=torch.full(
                (batch_size,), self.validity_logit, device=device
            ),
        )


def _missing_depth_evaluation_batch() -> dict[str, torch.Tensor]:
    batch_size, steps = 1, 7
    intrinsics = torch.tensor([[24.0, 24.0, 15.5, 11.5]])
    camera_position = torch.zeros(batch_size, 3)
    camera_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return {
        "roi_rgbd": torch.zeros(batch_size, steps, 4, 32, 32),
        "geometry": torch.zeros(batch_size, steps, 25),
        "missing_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "anchor_uv_px": torch.tensor([[15.5, 11.5]]),
        "raw_depth_m": torch.zeros(batch_size),
        "intrinsics_fx_fy_cx_cy": intrinsics,
        "camera_position_world_m": camera_position,
        "camera_orientation_world_wxyz": camera_orientation,
        "target_position_world_m": torch.tensor([[5.0, 0.0, 0.0]]),
        "target_depth_m": torch.tensor([5.0]),
        "target_present_mask": torch.ones(batch_size, dtype=torch.bool),
        "label_valid_mask": torch.ones(batch_size, dtype=torch.bool),
        "measurement_valid": torch.zeros(batch_size, dtype=torch.bool),
        "valid_depth_mask": torch.zeros(batch_size, dtype=torch.bool),
        "history_intrinsics_fx_fy_cx_cy": intrinsics.unsqueeze(1).repeat(1, steps, 1),
        "history_camera_position_world_m": camera_position.unsqueeze(1).repeat(1, steps, 1),
        "history_camera_orientation_world_wxyz": camera_orientation.unsqueeze(1).repeat(1, steps, 1),
        "history_center_uv_px": torch.tensor([[[15.5, 11.5]]]).repeat(1, steps, 1),
        "history_visible_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "history_label_valid_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "occlusion_ratio": torch.zeros(batch_size),
        "history_occlusion_ratio": torch.zeros(batch_size, steps),
        "bbox_jitter_score": torch.zeros(batch_size),
        "tracker_changed": torch.zeros(batch_size, dtype=torch.bool),
    }


class TargetStateDataTest(unittest.TestCase):
    def test_stage_b_config_pins_the_deployed_yolo_identity(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        config = load_training_config(
            repository / "configs" / "target_state" / "train_yolo_deployment.yaml"
        )
        self.assertEqual(config.stage, TrainingStage.YOLO_DEPLOYMENT)
        self.assertEqual(
            config.expected_yolo_model_sha256,
            "895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07",
        )

    def test_foreground_anchor_and_roi_preprocessing_match_production_contract(self) -> None:
        depth = np.full((24, 32), 20.0, dtype=np.float32)
        depth[7:15, 10:22] = 5.0
        bbox = (0.25, 0.25, 0.75, 0.75)
        resolver = DepthCandidateResolver(FrameStore(), min_depth_m=0.2, max_depth_m=200.0)
        production = resolver._sample_depth_details(depth, bbox)
        raw_depth, anchor = _foreground_cluster_anchor_depth(
            depth, bbox, min_depth_m=0.2, max_depth_m=200.0
        )
        self.assertAlmostEqual(raw_depth, production.depth_m)
        self.assertEqual(anchor, (production.u_px, production.v_px))

        rgb = np.arange(24 * 32 * 3, dtype=np.int64).reshape(24, 32, 3).astype(np.uint8)
        actual, _, _, _ = _crop_rgbd(
            rgb, depth, bbox, size=32, min_depth_m=0.2, max_depth_m=200.0
        )
        channels = np.concatenate(
            (rgb[6:18, 8:24].astype(np.float32) / 255.0, (depth[6:18, 8:24] / 200.0)[..., None]),
            axis=-1,
        )
        expected = F.interpolate(
            torch.from_numpy(channels).permute(2, 0, 1).unsqueeze(0),
            size=(32, 32), mode="bilinear", align_corners=False,
        ).squeeze(0)
        torch.testing.assert_close(torch.from_numpy(actual), expected)

    def test_relative_camera_pose_matches_production_contract(self) -> None:
        intrinsics = CameraIntrinsics(24.0, 24.0, 15.5, 11.5, 32, 24)
        current = FrameCameraGeometry(
            1.2, intrinsics, (1.2, 1.9, 3.05), (0.99875026, 0.0, 0.0, 0.04997917)
        )
        reference = FrameCameraGeometry(
            1.4, intrinsics, (1.4, 1.8, 3.1), (0.99500417, 0.0, 0.0, 0.09983342)
        )
        expected_pose = production_relative_camera_pose(current, reference)
        actual_pose = training_relative_camera_pose(
            current.camera_position_world_m,
            current.camera_orientation_world_wxyz,
            reference.camera_position_world_m,
            reference.camera_orientation_world_wxyz,
        )
        np.testing.assert_allclose(actual_pose[0], expected_pose[0], atol=1e-7)
        np.testing.assert_allclose(actual_pose[1], expected_pose[1], atol=1e-7)

    def test_loader_emits_roi_rgbd_and_exact_25d_deployable_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _dataset(root)
            dataset = TargetStateTorchDataset(_config(root, root / "out"), split="train")
            sample = dataset[0]
            self.assertEqual(len(GEOMETRY_INPUT_FIELDS), 25)
            self.assertEqual(sample["roi_rgbd"].shape, (7, 4, 32, 32))
            self.assertEqual(sample["geometry"].shape, (7, 25))
            self.assertEqual(sample["missing_mask"].tolist(), [False, False, True, False, False, False, False])
            self.assertTrue(torch.isfinite(sample["roi_rgbd"]).all())
            self.assertTrue(torch.isfinite(sample["geometry"]).all())
            self.assertEqual(sample["geometry"][:, 5].tolist(), [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
            torch.testing.assert_close(
                sample["geometry"][:, 17:23],
                torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.01]]).repeat(7, 1),
            )
            self.assertAlmostEqual(float(sample["roi_rgbd"][-1, 3].median()), 5.0 / 200.0)
            torch.testing.assert_close(
                sample["history_occlusion_ratio"],
                torch.tensor([0.0] * 6 + [0.3]),
            )

    def test_oracle_clean_repairs_recorded_yolo_miss_without_mixing_label_into_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _dataset(root)
            clean = TargetStateTorchDataset(
                _config(root, root / "out", stage=TrainingStage.ORACLE_CLEAN), split="train"
            )[0]
            deployment = TargetStateTorchDataset(
                _config(root, root / "out", stage=TrainingStage.YOLO_DEPLOYMENT), split="train"
            )[0]
            self.assertFalse(clean["missing_mask"].any())
            self.assertTrue(deployment["missing_mask"].any())
            self.assertEqual(set(clean), set(deployment))

    def test_loader_keeps_no_target_sequence_as_masked_validity_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            episode = _episode_for("train")
            records = []
            for index in range(7):
                record = replace(
                    _record(index, episode, "negative", missed=False),
                    training_label=None,
                )
                records.append(record)
                Image.fromarray(
                    np.full((24, 32, 3), 90, dtype=np.uint8)
                ).save(root / record.sensor_input.rgb_path)
                np.save(
                    root / record.sensor_input.depth_path,
                    np.full((24, 32), 5.0, dtype=np.float32),
                )
            (root / "frames.jsonl").write_text(
                "".join(json.dumps(item.to_dict()) + "\n" for item in records),
                encoding="utf-8",
            )
            sample = TargetStateTorchDataset(
                _config(root, root / "out"), split="train"
            )[0]
            self.assertFalse(bool(sample["target_present_mask"]))
            self.assertFalse(bool(sample["label_valid_mask"]))
            self.assertFalse(bool(sample["measurement_valid"]))
            self.assertFalse(sample["history_label_valid_mask"].any())
            self.assertFalse(sample["history_visible_mask"].any())
            torch.testing.assert_close(
                sample["target_position_world_m"], torch.zeros(3)
            )
            self.assertEqual(float(sample["target_depth_m"]), 0.0)
            # The actual detector input remains present: this is a false
            # positive training example, not a manufactured missing frame.
            self.assertFalse(sample["missing_mask"].any())

    def test_config_contract_rejects_non_25d_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "must be 25"):
                TargetStateTrainingConfig(
                    dataset_root=temporary, output_dir=temporary, geometry_input_dim=24
                )

    def test_strict_writer_preserves_detector_output_and_marks_oracle_offline_only(self) -> None:
        with self.assertRaises(TargetStateCollectionError):
            require_privileged_collection_acknowledgements(
                oracle_label_generation=True, acknowledge_privileged_oracle=False
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "written"
            episode = _episode_for("train")
            writer = TargetStateDatasetWriter(root, yolo_model_sha256="a" * 64)
            camera = CameraFrameInput(
                fx=24.0,
                fy=24.0,
                cx=15.5,
                cy=11.5,
                position_world_m=(0.0, 0.0, 0.0),
                orientation_world_wxyz=(
                    0.7651479363685005,
                    -0.1989238050344461,
                    0.2793046475962796,
                    0.5449466662828217,
                ),
                resolution_wh_px=(32, 24),
            )
            for index in range(7):
                record = _record(index, episode, "writer", missed=index == 2)
                record = replace(
                    record,
                    sensor_input=replace(record.sensor_input, camera=camera),
                )
                writer.append(
                    record,
                    rgb=np.full((24, 32, 3), 100, dtype=np.uint8),
                    depth_m=np.full((24, 32), 5.0, dtype=np.float32),
                )
            manifest_path, report = writer.finalize()
            self.assertTrue(report.ok, report.errors)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["detector_prediction_source"],
                "external_capture_spool_unverified",
            )
            self.assertEqual(manifest["oracle_usage"], "offline_training_labels_only")
            lines = (root / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertFalse(json.loads(lines[2])["detector_prediction"]["detected"])


class TargetStateTrainerTest(unittest.TestCase):
    def assertAccumulatorStateEqual(
        self, left: dict[str, object], right: dict[str, object]
    ) -> None:
        self.assertEqual(set(left), set(right))
        for name in left:
            if isinstance(left[name], list):
                self.assertIsInstance(right[name], list)
                self.assertEqual(len(left[name]), len(right[name]))
                for left_tensor, right_tensor in zip(left[name], right[name]):
                    torch.testing.assert_close(left_tensor, right_tensor)
            else:
                self.assertEqual(left[name], right[name])

    @staticmethod
    def _evaluation_batches() -> list[dict[str, torch.Tensor]]:
        batches: list[dict[str, torch.Tensor]] = []
        for index in range(4):
            batch = {
                name: value.clone()
                for name, value in _missing_depth_evaluation_batch().items()
            }
            batch["target_position_world_m"][0, 0] = float(index + 1)
            batch["occlusion_ratio"][0] = 0.3 if index % 2 else 0.0
            batch["bbox_jitter_score"][0] = 0.02 if index >= 2 else 0.0
            batches.append(batch)
        no_target = {
            name: value.clone()
            for name, value in _missing_depth_evaluation_batch().items()
        }
        no_target["target_present_mask"].fill_(False)
        no_target["label_valid_mask"].fill_(False)
        no_target["target_position_world_m"].zero_()
        batches.append(no_target)
        return batches

    def test_evaluation_accumulators_merge_to_whole_loader_metrics(self) -> None:
        model = _FixedEvaluationModel(validity_logit=-10.0)
        batches = self._evaluation_batches()
        expected = evaluate_model(
            model,
            batches,
            device=torch.device("cpu"),
            maximum_depth_m=200.0,
        )
        first = accumulate_evaluation(
            model,
            batches[:2],
            device=torch.device("cpu"),
            maximum_depth_m=200.0,
        )
        second = accumulate_evaluation(
            model,
            batches[2:],
            device=torch.device("cpu"),
            maximum_depth_m=200.0,
        )
        returned = first.merge(second)
        self.assertIs(returned, first)
        self.assertEqual(first.batch_count, len(batches))
        self.assertAlmostEqual(
            first.loss_sum / first.batch_count, expected["mean_loss"]
        )
        self.assertTrue(
            all(value.device.type == "cpu" for value in first.model_errors)
        )
        self.assertEqual(first.finalize(), expected)

    def test_accumulate_evaluation_can_reuse_one_cross_shard_accumulator(self) -> None:
        model = _FixedEvaluationModel(validity_logit=-10.0)
        batches = self._evaluation_batches()
        accumulator = TargetStateEvaluationAccumulator(maximum_depth_m=200.0)
        first_result = accumulate_evaluation(
            model,
            batches[:3],
            device=torch.device("cpu"),
            maximum_depth_m=200.0,
            accumulator=accumulator,
        )
        second_result = accumulate_evaluation(
            model,
            batches[3:],
            device=torch.device("cpu"),
            maximum_depth_m=200.0,
            accumulator=accumulator,
        )
        self.assertIs(first_result, accumulator)
        self.assertIs(second_result, accumulator)
        self.assertEqual(accumulator.batch_count, len(batches))

    def test_evaluation_accumulator_state_round_trip_is_lossless(self) -> None:
        accumulator = accumulate_evaluation(
            _FixedEvaluationModel(validity_logit=-10.0),
            self._evaluation_batches(),
            device=torch.device("cpu"),
            maximum_depth_m=200.0,
        )
        state = accumulator.state_dict()
        restored = TargetStateEvaluationAccumulator.from_state_dict(
            state, maximum_depth_m=200.0
        )
        self.assertEqual(restored.finalize(), accumulator.finalize())
        self.assertEqual(restored.loss_sum, accumulator.loss_sum)
        self.assertEqual(restored.batch_count, accumulator.batch_count)
        state["model_errors"][0].fill_(999.0)
        self.assertNotEqual(
            state["model_errors"][0].tolist(), restored.model_errors[0].tolist()
        )

    def test_evaluation_accumulator_state_and_merge_validation(self) -> None:
        empty = TargetStateEvaluationAccumulator(maximum_depth_m=200.0)
        with self.assertRaisesRegex(TargetStateTrainingError, "no batches"):
            empty.finalize()
        with self.assertRaisesRegex(ValueError, "different maximum_depth_m"):
            empty.merge(TargetStateEvaluationAccumulator(maximum_depth_m=100.0))
        state = empty.state_dict()
        state["batch_count"] = 1
        with self.assertRaisesRegex(ValueError, "batch_count entries"):
            TargetStateEvaluationAccumulator.from_state_dict(
                state, maximum_depth_m=200.0
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            TargetStateEvaluationAccumulator.from_state_dict(
                empty.state_dict(), maximum_depth_m=100.0
            )

    def test_nonfinite_batch_loss_fails_before_accumulator_mutation(self) -> None:
        batch = _missing_depth_evaluation_batch()
        model = _FixedEvaluationModel(validity_logit=-10.0)
        output = model(batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
        accumulator = TargetStateEvaluationAccumulator(maximum_depth_m=200.0)
        accumulator.add_batch(batch=batch, output=output)
        before = accumulator.state_dict()
        finite_loss = compute_target_state_losses(output, batch)
        invalid_loss = replace(finite_loss, total=torch.tensor(float("nan")))

        with patch(
            "training.target_state.trainer.compute_target_state_losses",
            return_value=invalid_loss,
        ):
            with self.assertRaisesRegex(TargetStateTrainingError, "non-finite loss"):
                accumulator.add_batch(batch=batch, output=output)

        self.assertAccumulatorStateEqual(before, accumulator.state_dict())

    def test_nonfinite_metric_value_fails_before_accumulator_mutation(self) -> None:
        batch = _missing_depth_evaluation_batch()
        output = _FixedEvaluationModel(validity_logit=-10.0)(
            batch["roi_rgbd"], batch["geometry"], batch["missing_mask"]
        )
        overflowing_uncertainty = replace(
            output,
            position_log_variance=torch.full_like(
                output.position_log_variance, 1000.0
            ),
        )
        accumulator = TargetStateEvaluationAccumulator(maximum_depth_m=200.0)
        before = accumulator.state_dict()

        with self.assertRaisesRegex(
            TargetStateTrainingError, "non-finite metric values: uncertainties"
        ):
            accumulator.add_batch(batch=batch, output=overflowing_uncertainty)

        self.assertAccumulatorStateEqual(before, accumulator.state_dict())

    def test_evaluation_accumulator_restore_rejects_nonfinite_loss_sum(self) -> None:
        state = TargetStateEvaluationAccumulator(
            maximum_depth_m=200.0
        ).state_dict()
        state["loss_sum"] = float("nan")
        with self.assertRaisesRegex(ValueError, "loss_sum must be finite"):
            TargetStateEvaluationAccumulator.from_state_dict(
                state, maximum_depth_m=200.0
            )

    def test_evaluation_accumulator_restore_rejects_nonfinite_float_tensors(self) -> None:
        accumulator = accumulate_evaluation(
            _FixedEvaluationModel(validity_logit=-10.0),
            self._evaluation_batches(),
            device=torch.device("cpu"),
            maximum_depth_m=200.0,
        )
        float_fields = (
            "model_errors",
            "baseline_errors",
            "uncertainties",
            "occlusions",
            "jitters",
        )
        invalid_values = (float("nan"), float("inf"), float("-inf"))
        for index, name in enumerate(float_fields):
            with self.subTest(field=name):
                state = accumulator.state_dict()
                state[name][0][0] = invalid_values[index % len(invalid_values)]
                with self.assertRaisesRegex(
                    ValueError, rf"state {name} entries must contain only finite values"
                ):
                    TargetStateEvaluationAccumulator.from_state_dict(
                        state, maximum_depth_m=200.0
                    )

    def test_evaluate_model_counts_rejected_missing_depth_as_failure_only(self) -> None:
        metrics = evaluate_model(
            _FixedEvaluationModel(validity_logit=-10.0),
            [_missing_depth_evaluation_batch()],
            device=torch.device("cpu"),
            maximum_depth_m=200.0,
        )["model"]
        self.assertEqual(metrics["measurement_failure_rate"], 1.0)
        self.assertEqual(metrics["invalid_output_count"], 0)

    def test_evaluate_model_counts_claimed_missing_depth_as_invalid_output(self) -> None:
        metrics = evaluate_model(
            _FixedEvaluationModel(validity_logit=10.0),
            [_missing_depth_evaluation_batch()],
            device=torch.device("cpu"),
            maximum_depth_m=200.0,
        )["model"]
        self.assertEqual(metrics["measurement_failure_rate"], 1.0)
        self.assertEqual(metrics["invalid_output_count"], 1)

    def test_invalid_output_mask_treats_rejected_missing_depth_as_failure_not_numeric_violation(self) -> None:
        output = TemporalRayDepthOutput(
            delta_uv_px=torch.zeros(1, 2),
            depth_residual_m=torch.zeros(1),
            position_log_variance=torch.zeros(1, 3),
            measurement_valid_logit=torch.tensor([-10.0]),
        )
        invalid = _invalid_model_output_mask(
            output,
            predicted_position_world_m=torch.zeros(1, 3),
            corrected_depth_m=torch.tensor([0.0]),
            model_claims_valid=torch.tensor([False]),
            maximum_depth_m=200.0,
        )
        self.assertEqual(invalid.tolist(), [False])

    def test_invalid_output_mask_rejects_claimed_invalid_geometry(self) -> None:
        output = TemporalRayDepthOutput(
            delta_uv_px=torch.zeros(3, 2),
            depth_residual_m=torch.zeros(3),
            position_log_variance=torch.zeros(3, 3),
            measurement_valid_logit=torch.full((3,), 10.0),
        )
        invalid = _invalid_model_output_mask(
            output,
            predicted_position_world_m=torch.tensor(
                [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            corrected_depth_m=torch.tensor([0.0, 5.0, 201.0]),
            model_claims_valid=torch.ones(3, dtype=torch.bool),
            maximum_depth_m=200.0,
        )
        self.assertEqual(invalid.tolist(), [True, True, True])

    def test_invalid_output_mask_always_rejects_nonfinite_network_heads(self) -> None:
        output = TemporalRayDepthOutput(
            delta_uv_px=torch.tensor([[float("nan"), 0.0]]),
            depth_residual_m=torch.zeros(1),
            position_log_variance=torch.zeros(1, 3),
            measurement_valid_logit=torch.tensor([-10.0]),
        )
        invalid = _invalid_model_output_mask(
            output,
            predicted_position_world_m=torch.zeros(1, 3),
            corrected_depth_m=torch.tensor([0.0]),
            model_claims_valid=torch.tensor([False]),
            maximum_depth_m=200.0,
        )
        self.assertEqual(invalid.tolist(), [True])

    def test_one_epoch_writes_bounded_artifacts_and_full_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            output = Path(temporary) / "outputs"
            _dataset(root)
            result = train_target_state(
                _config(root, output, stage=TrainingStage.ORACLE_CLEAN)
            )
            self.assertTrue(result.best_checkpoint.is_file())
            self.assertTrue(result.latest_checkpoint.is_file())
            self.assertTrue((result.run_dir / "metrics.csv").is_file())
            manifest = json.loads(result.model_manifest.read_text(encoding="utf-8"))
            required = {
                "model_type", "schema_version", "checkpoint_path", "checkpoint_sha256",
                "dataset_sha256", "training_commit_sha", "input_fields", "output_fields",
                "history_size", "max_history_age_s", "camera_convention",
                "coordinate_convention", "validation_metrics",
            }
            self.assertTrue(required.issubset(manifest))
            self.assertEqual(len(manifest["checkpoint_sha256"]), 64)
            self.assertEqual(manifest["checkpoint_sha256"], sha256_file(result.best_checkpoint))
            self.assertEqual(manifest["training_stage"], "oracle_clean")
            self.assertFalse(manifest["promotion"]["stage_satisfied"])
            self.assertTrue(list((result.run_dir / "tensorboard").glob("events.out.tfevents.*")))
            checkpoint = torch.load(result.best_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["training_stage"], "oracle_clean")
            self.assertEqual(checkpoint["dataset_sha256"], manifest["dataset_sha256"])
            self.assertFalse(result.promoted)
            self.assertFalse(any(result.run_dir.glob("*.mp4")))

    def test_stage_b_fails_closed_without_stage_a_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _dataset(root)
            with self.assertRaisesRegex(TargetStateTrainingError, "requires --initial-checkpoint"):
                train_target_state(_config(root, Path(temporary) / "outputs"))

    def test_stage_b_rejects_a_checkpoint_not_marked_as_stage_a(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "wrong.pt"
            torch.save(
                {
                    "model_type": "temporal_ray_depth_residual",
                    "schema_version": 1,
                    "training_stage": "yolo_deployment",
                },
                wrong,
            )
            config = replace(
                _config(Path(temporary), Path(temporary) / "outputs"),
                initial_checkpoint_path=wrong,
            )
            with self.assertRaisesRegex(
                TargetStateTrainingError, "training_stage=oracle_clean"
            ):
                validate_initial_checkpoint(config)

    def test_stage_b_fails_closed_on_deployed_yolo_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            output = Path(temporary) / "outputs"
            _dataset(root)
            stage_a = train_target_state(
                _config(root, output, stage=TrainingStage.ORACLE_CLEAN)
            )
            digest = compute_dataset_sha256(
                root, read_frame_records(root / "frames.jsonl")
            )
            (root / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_sha256": digest,
                        "detector_prediction_source": "real_yolo_deployment_output",
                        "yolo_model_sha256": "b" * 64,
                        "oracle_usage": "offline_training_labels_only",
                    }
                ),
                encoding="utf-8",
            )
            unverified = _config(root, output)
            datasets = {
                split: TargetStateTorchDataset(unverified, split=split)
                for split in ("train", "validation", "test")
            }
            stage_b = replace(
                unverified,
                initial_checkpoint_path=stage_a.best_checkpoint,
                require_dataset_manifest=True,
                expected_yolo_model_sha256="a" * 64,
                run_name="stage_b",
            )
            with self.assertRaisesRegex(
                TargetStateTrainingError, "YOLO identity mismatch"
            ):
                train_target_state(
                    stage_b,
                    datasets=datasets,
                    dataset_sha256=digest,
                )
            manifest = json.loads(
                (root / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            manifest["yolo_model_sha256"] = "a" * 64
            (root / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                TargetStateTrainingError, "preflight-verified YOLO deployment receipt"
            ):
                train_target_state(
                    stage_b,
                    datasets=datasets,
                    dataset_sha256=digest,
                )

    def test_promotion_gate_checks_every_required_comparison(self) -> None:
        model = {
            "position_median_error_m": 0.5,
            "position_p95_error_m": 1.01,
            "measurement_failure_rate": 0.05,
            "no_target_false_positive_rate": 0.02,
            "occluded_position_median_error_m": 0.7,
            "jittered_position_median_error_m": 0.6,
            "covariance_error_spearman": 0.4,
            "invalid_output_count": 0,
        }
        baseline = {
            "position_median_error_m": 0.8,
            "position_p95_error_m": 1.0,
            "measurement_failure_rate": 0.05,
            "no_target_false_positive_rate": 0.03,
            "occluded_position_median_error_m": 0.9,
            "jittered_position_median_error_m": 0.7,
            "invalid_output_count": 0,
        }
        gate = evaluate_promotion(
            {"model": model, "deterministic_rgbd_baseline": baseline},
            p95_max_ratio=1.05,
            minimum_covariance_correlation=0.1,
        )
        self.assertTrue(gate["passed"], gate["reasons"])


if __name__ == "__main__":
    unittest.main()
