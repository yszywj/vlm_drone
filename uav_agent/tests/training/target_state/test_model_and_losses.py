from __future__ import annotations

import unittest

import torch

from training.target_state.geometry import (
    corrected_ray_to_world,
    diagonal_covariance,
    project_world_to_pixel,
)
from training.target_state.losses import compute_target_state_losses
from training.target_state.model import TemporalRayDepthNet, TemporalRayDepthOutput


class TemporalRayGeometryTest(unittest.TestCase):
    def test_corrected_ray_world_pixel_round_trip(self) -> None:
        world, depth, valid = corrected_ray_to_world(
            anchor_uv_px=torch.tensor([[10.0, 12.0]]),
            raw_depth_m=torch.tensor([4.0]),
            delta_uv_px=torch.tensor([[1.0, -2.0]]),
            depth_residual_m=torch.tensor([0.5]),
            intrinsics_fx_fy_cx_cy=torch.tensor([[100.0, 100.0, 10.0, 10.0]]),
            camera_position_world_m=torch.tensor([[1.0, 2.0, 3.0]]),
            camera_orientation_world_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
        uv, projected_depth, projected_valid = project_world_to_pixel(
            position_world_m=world,
            intrinsics_fx_fy_cx_cy=torch.tensor([[100.0, 100.0, 10.0, 10.0]]),
            camera_position_world_m=torch.tensor([[1.0, 2.0, 3.0]]),
            camera_orientation_world_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(uv, torch.tensor([[11.0, 10.0]]))
        torch.testing.assert_close(depth, torch.tensor([4.5]))
        torch.testing.assert_close(projected_depth, depth)
        self.assertTrue(bool(valid.item()))
        self.assertTrue(bool(projected_valid.item()))

    def test_covariance_is_finite_positive_diagonal(self) -> None:
        covariance = diagonal_covariance(torch.tensor([[0.0, -30.0, 30.0]]))
        self.assertTrue(torch.isfinite(covariance).all())
        self.assertTrue(torch.linalg.eigvalsh(covariance).min() > 0.0)


class TemporalRayDepthModelTest(unittest.TestCase):
    def test_model_has_only_residual_uncertainty_and_validity_outputs(self) -> None:
        model = TemporalRayDepthNet(
            geometry_input_dim=9,
            roi_feature_dim=24,
            geometry_feature_dim=16,
            hidden_dim=32,
            gru_layers=2,
        )
        output = model(
            torch.zeros(2, 5, 4, 32, 32),
            torch.zeros(2, 5, 9),
            torch.tensor([[False, False, True, False, False], [False] * 5]),
        )
        self.assertEqual(set(output.as_dict()), {
            "delta_uv_px", "depth_residual_m", "position_log_variance", "measurement_valid_logit"
        })
        self.assertEqual(output.delta_uv_px.shape, (2, 2))
        self.assertEqual(output.depth_residual_m.shape, (2,))
        self.assertEqual(output.position_log_variance.shape, (2, 3))
        self.assertEqual(output.measurement_valid_logit.shape, (2,))

    def test_all_required_masked_losses_are_finite_and_backward_safe(self) -> None:
        batch_size, steps = 2, 5
        output = TemporalRayDepthOutput(
            delta_uv_px=torch.zeros(batch_size, 2, requires_grad=True),
            depth_residual_m=torch.zeros(batch_size, requires_grad=True),
            position_log_variance=torch.zeros(batch_size, 3, requires_grad=True),
            measurement_valid_logit=torch.zeros(batch_size, requires_grad=True),
        )
        intrinsics = torch.tensor([[100.0, 100.0, 10.0, 10.0]]).repeat(batch_size, 1)
        camera_position = torch.zeros(batch_size, 3)
        camera_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch_size, 1)
        batch = {
            "anchor_uv_px": torch.tensor([[10.0, 10.0]]).repeat(batch_size, 1),
            "raw_depth_m": torch.tensor([4.0, 0.0]),
            "intrinsics_fx_fy_cx_cy": intrinsics,
            "camera_position_world_m": camera_position,
            "camera_orientation_world_wxyz": camera_quaternion,
            "target_position_world_m": torch.tensor([[4.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "target_depth_m": torch.tensor([4.0, 2.0]),
            "measurement_valid": torch.tensor([True, False]),
            "target_present_mask": torch.tensor([True, True]),
            "label_valid_mask": torch.tensor([True, True]),
            "valid_depth_mask": torch.tensor([True, False]),
            "occlusion_ratio": torch.tensor([0.0, 0.0]),
            "history_intrinsics_fx_fy_cx_cy": intrinsics.unsqueeze(1).repeat(1, steps, 1),
            "history_camera_position_world_m": camera_position.unsqueeze(1).repeat(1, steps, 1),
            "history_camera_orientation_world_wxyz": camera_quaternion.unsqueeze(1).repeat(1, steps, 1),
            "history_center_uv_px": torch.tensor([[[10.0, 10.0]]]).repeat(batch_size, steps, 1),
            "history_visible_mask": torch.tensor([[True] * steps, [False] * steps]),
            "history_label_valid_mask": torch.ones(batch_size, steps, dtype=torch.bool),
            "history_occlusion_ratio": torch.zeros(batch_size, steps),
        }

        losses = compute_target_state_losses(output, batch)
        for value in losses.scalars().values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertAlmostEqual(float(losses.depth_huber.detach()), 0.0)
        self.assertAlmostEqual(float(losses.position_3d_huber.detach()), 0.0)
        losses.total.backward()
        self.assertIsNotNone(output.measurement_valid_logit.grad)

    def test_occlusion_weights_regression_and_masks_fully_occluded_sample(self) -> None:
        """Partial visibility is weighted; full occlusion contributes exactly zero."""

        batch_size, steps = 3, 4
        output = TemporalRayDepthOutput(
            delta_uv_px=torch.zeros(batch_size, 2, requires_grad=True),
            depth_residual_m=torch.zeros(batch_size, requires_grad=True),
            position_log_variance=torch.zeros(batch_size, 3, requires_grad=True),
            measurement_valid_logit=torch.zeros(batch_size, requires_grad=True),
        )
        intrinsics = torch.tensor([[100.0, 100.0, 10.0, 10.0]]).repeat(batch_size, 1)
        camera_position = torch.zeros(batch_size, 3)
        camera_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch_size, 1)
        batch = {
            "anchor_uv_px": torch.tensor([[10.0, 10.0]]).repeat(batch_size, 1),
            "raw_depth_m": torch.tensor([3.0, 3.0, 3.0]),
            "intrinsics_fx_fy_cx_cy": intrinsics,
            "camera_position_world_m": camera_position,
            "camera_orientation_world_wxyz": camera_quaternion,
            # Corrected ray is [3, 0, 0].  Per-sample smooth-L1 depth error is
            # therefore [0, 1.5, 9.5]; the fully occluded third value must vanish.
            "target_position_world_m": torch.tensor(
                [[3.0, 0.0, 0.0], [5.0, 0.0, 0.0], [13.0, 0.0, 0.0]]
            ),
            "target_depth_m": torch.tensor([3.0, 5.0, 13.0]),
            "measurement_valid": torch.tensor([True, True, True]),
            "target_present_mask": torch.tensor([True, True, True]),
            "label_valid_mask": torch.tensor([True, True, True]),
            "valid_depth_mask": torch.tensor([True, True, True]),
            "occlusion_ratio": torch.tensor([0.0, 0.5, 1.0]),
            "history_intrinsics_fx_fy_cx_cy": intrinsics.unsqueeze(1).repeat(1, steps, 1),
            "history_camera_position_world_m": camera_position.unsqueeze(1).repeat(1, steps, 1),
            "history_camera_orientation_world_wxyz": camera_quaternion.unsqueeze(1).repeat(1, steps, 1),
            "history_center_uv_px": torch.tensor([[[10.0, 10.0]]]).repeat(batch_size, steps, 1),
            "history_visible_mask": torch.ones(batch_size, steps, dtype=torch.bool),
            "history_label_valid_mask": torch.ones(batch_size, steps, dtype=torch.bool),
            "history_occlusion_ratio": torch.tensor(
                [[0.0] * steps, [0.5] * steps, [1.0] * steps]
            ),
        }

        losses = compute_target_state_losses(output, batch)

        # (0 * 1.0 + 1.5 * 0.5) / (1.0 + 0.5) == 0.5.  The third sample's
        # smooth-L1 error is 9.5 and would dominate if full occlusion leaked in.
        self.assertAlmostEqual(float(losses.depth_huber.detach()), 0.5, places=6)
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()
        self.assertTrue(torch.isfinite(output.depth_residual_m.grad).all())

    def test_occlusion_weight_sanitizes_nonfinite_batch_without_nan_loss(self) -> None:
        batch_size, steps = 1, 4
        output = TemporalRayDepthOutput(
            delta_uv_px=torch.zeros(batch_size, 2, requires_grad=True),
            depth_residual_m=torch.zeros(batch_size, requires_grad=True),
            position_log_variance=torch.zeros(batch_size, 3, requires_grad=True),
            measurement_valid_logit=torch.zeros(batch_size, requires_grad=True),
        )
        intrinsics = torch.tensor([[100.0, 100.0, 10.0, 10.0]])
        camera_position = torch.zeros(batch_size, 3)
        camera_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        batch = {
            "anchor_uv_px": torch.tensor([[10.0, 10.0]]),
            "raw_depth_m": torch.tensor([3.0]),
            "intrinsics_fx_fy_cx_cy": intrinsics,
            "camera_position_world_m": camera_position,
            "camera_orientation_world_wxyz": camera_quaternion,
            "target_position_world_m": torch.tensor([[13.0, 0.0, 0.0]]),
            "target_depth_m": torch.tensor([13.0]),
            "measurement_valid": torch.tensor([True]),
            "target_present_mask": torch.tensor([True]),
            "label_valid_mask": torch.tensor([True]),
            "valid_depth_mask": torch.tensor([True]),
            "occlusion_ratio": torch.tensor([float("nan")]),
            "history_intrinsics_fx_fy_cx_cy": intrinsics.unsqueeze(1).repeat(1, steps, 1),
            "history_camera_position_world_m": camera_position.unsqueeze(1).repeat(1, steps, 1),
            "history_camera_orientation_world_wxyz": camera_quaternion.unsqueeze(1).repeat(1, steps, 1),
            "history_center_uv_px": torch.tensor([[[10.0, 10.0]]]).repeat(1, steps, 1),
            "history_visible_mask": torch.ones(1, steps, dtype=torch.bool),
            "history_label_valid_mask": torch.ones(1, steps, dtype=torch.bool),
            "history_occlusion_ratio": torch.full((1, steps), float("inf")),
        }

        losses = compute_target_state_losses(output, batch)
        self.assertEqual(float(losses.depth_huber.detach()), 0.0)
        self.assertEqual(float(losses.position_3d_huber.detach()), 0.0)
        self.assertEqual(float(losses.reprojection_huber.detach()), 0.0)
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()
        self.assertTrue(torch.isfinite(output.depth_residual_m.grad).all())

    def test_no_target_label_uses_validity_bce_but_never_geometry_regression(self) -> None:
        steps = 5
        output = TemporalRayDepthOutput(
            delta_uv_px=torch.zeros(1, 2, requires_grad=True),
            depth_residual_m=torch.zeros(1, requires_grad=True),
            position_log_variance=torch.zeros(1, 3, requires_grad=True),
            measurement_valid_logit=torch.zeros(1, requires_grad=True),
        )
        intrinsics = torch.tensor([[100.0, 100.0, 10.0, 10.0]])
        camera_position = torch.zeros(1, 3)
        camera_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        batch = {
            "anchor_uv_px": torch.tensor([[10.0, 10.0]]),
            "raw_depth_m": torch.tensor([3.0]),
            "intrinsics_fx_fy_cx_cy": intrinsics,
            "camera_position_world_m": camera_position,
            "camera_orientation_world_wxyz": camera_quaternion,
            # Finite zeros are collation placeholders only. Both explicit
            # label masks are false, so they cannot become geometric targets.
            "target_position_world_m": torch.zeros(1, 3),
            "target_depth_m": torch.zeros(1),
            "measurement_valid": torch.tensor([False]),
            "target_present_mask": torch.tensor([False]),
            "label_valid_mask": torch.tensor([False]),
            "valid_depth_mask": torch.tensor([True]),
            "occlusion_ratio": torch.zeros(1),
            "history_occlusion_ratio": torch.zeros(1, steps),
            "history_intrinsics_fx_fy_cx_cy": intrinsics.unsqueeze(1).repeat(1, steps, 1),
            "history_camera_position_world_m": camera_position.unsqueeze(1).repeat(1, steps, 1),
            "history_camera_orientation_world_wxyz": camera_quaternion.unsqueeze(1).repeat(1, steps, 1),
            "history_center_uv_px": torch.zeros(1, steps, 2),
            "history_visible_mask": torch.zeros(1, steps, dtype=torch.bool),
            "history_label_valid_mask": torch.zeros(1, steps, dtype=torch.bool),
        }
        losses = compute_target_state_losses(output, batch)
        self.assertEqual(float(losses.depth_huber.detach()), 0.0)
        self.assertEqual(float(losses.position_3d_huber.detach()), 0.0)
        self.assertEqual(float(losses.reprojection_huber.detach()), 0.0)
        self.assertEqual(float(losses.gaussian_nll.detach()), 0.0)
        self.assertGreater(float(losses.validity_bce.detach()), 0.0)
        losses.total.backward()
        self.assertIsNotNone(output.measurement_valid_logit.grad)


if __name__ == "__main__":
    unittest.main()
